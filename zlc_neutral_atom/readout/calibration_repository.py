"""Durable storage and admission for readout calibrations.

The repository has one job: make an already computed
``CalibrationAnalysisResult`` atomically visible.  Scientific validation lives
in the calibration/analysis values; canonical encoding lives in
``calibration_codec``; durability lives in ``zlc_storage`` and the generic
commit coordinator.  This module deliberately adds no second proof graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import TYPE_CHECKING

from zlc_storage import (
    ContentAddressedStore,
    ContentRef,
    ContentStoreAuthority,
    RepositoryRootLease,
    canonical_text,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping,
    positive_integer,
    positive_real,
    sha256_digest,
)

from zlc_neutral_atom.capture_reference import CaptureArtifactRef
from zlc_neutral_atom.runtime.commit import (
    CommitIntent,
    CommitTarget,
    FinalCommit,
    PersistentCommitJournal,
    PublishedManifest,
    RepositoryCommitCoordinator,
    publish_manifest_with_visibility_reconciliation,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunPlan,
)
from zlc_neutral_atom.runtime.resources import (
    ClaimMode,
    ResourceClaim,
    ResourceKey,
)

from .analysis import (
    CalibrationAnalysisResult,
    CalibrationComputation,
    CalibrationReport,
    _analyze_calibration_resolved,
    estimate_calibration_analysis_peak_bytes,
)
from .calibration import (
    CalibrationAnalysisRequest,
    CalibrationArtifact,
    ResolvedCalibration,
    _ResolvedCalibrationSource,
    _resolve_calibration_source,
    _validate_calibration_artifact_source_compatibility,
)
from .calibration_codec import (
    calibration_report_blob_refs,
    decode_calibration_artifact,
    decode_calibration_report,
    decode_calibration_report_arrays,
    encode_calibration_artifact,
    encode_calibration_reference_average,
    encode_calibration_reference_average_validity,
    encode_calibration_report_metadata,
)
from .calibration_reference import (
    CALIBRATION_ARTIFACT_NAMESPACE,
    CalibrationArtifactRef,
)
from .contracts import ReadoutBindingKey

if TYPE_CHECKING:
    from zlc_neutral_atom.artifacts.capture import AdmittedCapture, CaptureRepository


CALIBRATION_MANIFEST_FORMAT = "zlc_neutral_atom.calibration-manifest"
_CALIBRATION_ARTIFACT_KIND = "calibration"
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_REPORT_METADATA_BYTES = 64 * 1024 * 1024
_DEFAULT_DIAGNOSTIC_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_METADATA_DECODE_MULTIPLIER = 8
_MANIFEST_FIELDS = frozenset(
    {"format", "repository_id", "artifact_blob", "report_blob"}
)
_CALIBRATION_ANALYSIS_CLAIM = ResourceClaim(
    ResourceKey(("analysis", "neutral-atom-calibration")),
    ClaimMode.EXCLUSIVE,
)


@dataclass(frozen=True, slots=True)
class _PreparedCalibrationAnalysis:
    source: "AdmittedCapture"
    resolved: _ResolvedCalibrationSource


def _manifest_payload(
    repository_id: str,
    artifact_blob: ContentRef,
    report_blob: ContentRef,
) -> bytes:
    """Encode the sole current manifest shape.

    The one plain format name makes the persistent root self-identifying; the
    remaining fields are only repository ownership and two content references.
    """

    return encode(
        {
            "format": CALIBRATION_MANIFEST_FORMAT,
            "repository_id": repository_id,
            "artifact_blob": content_ref_to_tree(artifact_blob),
            "report_blob": content_ref_to_tree(report_blob),
        }
    )


def _decode_manifest(payload: bytes) -> tuple[str, ContentRef, ContentRef]:
    if not isinstance(payload, bytes):
        raise TypeError("calibration manifest payload must be bytes")
    tree = exact_mapping(
        decode(payload),
        _MANIFEST_FIELDS,
        CALIBRATION_MANIFEST_FORMAT,
        discriminator="format",
    )
    repository_id = canonical_text(tree["repository_id"], "repository_id")
    artifact_blob = content_ref_from_tree(tree["artifact_blob"])
    report_blob = content_ref_from_tree(tree["report_blob"])
    if _manifest_payload(repository_id, artifact_blob, report_blob) != payload:
        raise ValueError("calibration manifest is not canonical current format")
    return repository_id, artifact_blob, report_blob


def _target(
    repository_id: str,
    reference: CalibrationArtifactRef,
) -> CommitTarget:
    return CommitTarget(
        repository_id,
        _CALIBRATION_ARTIFACT_KIND,
        CALIBRATION_MANIFEST_FORMAT,
        reference.target_ref,
        reference.manifest_digest,
    )


def _commit_id(run_id: str, manifest_digest: str) -> str:
    return f"calibration-final-{run_id}-{manifest_digest}"


class CalibrationRepository:
    """Content-addressed calibration store with final-commit visibility."""

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-calibration",
        max_report_metadata_bytes: int = _DEFAULT_MAX_REPORT_METADATA_BYTES,
        diagnostic_memory_limit_bytes: int = _DEFAULT_DIAGNOSTIC_MEMORY_LIMIT_BYTES,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = canonical_text(repository_id, "repository_id")
        self.max_report_metadata_bytes = positive_integer(
            max_report_metadata_bytes,
            "max_report_metadata_bytes",
        )
        self.diagnostic_memory_limit_bytes = positive_integer(
            diagnostic_memory_limit_bytes,
            "diagnostic_memory_limit_bytes",
        )
        self._lock = threading.RLock()
        self._closed = False
        self._root_lease = RepositoryRootLease(self.root)
        try:
            self._store = ContentAddressedStore(self.root / "content")
            self._store_authority = self._store.authority()
            journal = PersistentCommitJournal(
                self.root / "calibration-commit.journal",
                self.repository_id,
            )
            # Construction performs synchronous recovery.  _recover therefore
            # relies only on fields initialized above, never on _coordinator.
            self._coordinator: RepositoryCommitCoordinator[
                CalibrationArtifactRef
            ] = RepositoryCommitCoordinator(
                journal,
                self._recover,
                root_lease=self._root_lease,
            )
        except BaseException:
            self._root_lease.close()
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("calibration repository is closed")
        self._root_lease.require_active()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            # The lease refuses closure while a prepared commit is outstanding.
            self._root_lease.close()
            self._closed = True

    def __enter__(self) -> "CalibrationRepository":
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def _validate_reference(self, reference: CalibrationArtifactRef) -> None:
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("CalibrationArtifactRef belongs to another repository")

    def _read_manifest(self, reference: CalibrationArtifactRef) -> bytes:
        return self._content_authority().read_manifest(
            CALIBRATION_ARTIFACT_NAMESPACE,
            reference.manifest_digest,
            max_bytes=_MAX_MANIFEST_BYTES,
        )

    def _content_authority(self) -> ContentStoreAuthority:
        with self._lock:
            self._require_open()
            return self._store_authority

    def _require_final_commit(
        self,
        reference: CalibrationArtifactRef,
    ) -> None:
        """Require the journal linearization point for one public reference."""

        with self._lock:
            self._require_open()
            self._validate_reference(reference)
            target = _target(self.repository_id, reference)
            matching = tuple(
                intent
                for intent in self._coordinator.committed_intents()
                if intent.target == target
            )
            if not matching:
                raise PermissionError(
                    "calibration lacks FINAL commit authority"
                )
            for intent in matching:
                if intent.commit_id != _commit_id(
                    intent.run_id,
                    reference.manifest_digest,
                ):
                    raise ValueError("calibration commit identity is inconsistent")

    def _artifact_and_report_ref(
        self,
        reference: CalibrationArtifactRef,
        *,
        manifest_payload: bytes | None = None,
    ) -> tuple[CalibrationArtifact, ContentRef]:
        authority = self._content_authority()
        payload = (
            authority.read_manifest(
                CALIBRATION_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
                max_bytes=_MAX_MANIFEST_BYTES,
            )
            if manifest_payload is None
            else manifest_payload
        )
        repository_id, artifact_ref, report_ref = _decode_manifest(payload)
        if repository_id != self.repository_id:
            raise ValueError("calibration manifest belongs to another repository")
        artifact = decode_calibration_artifact(
            authority.read_blob(
                artifact_ref,
                max_bytes=_MAX_ARTIFACT_BYTES,
            )
        )
        return artifact, report_ref

    def _admit_diagnostic_memory(
        self,
        average_bytes: int,
        validity_bytes: int,
        metadata_bytes: int,
        *,
        materialize: bool,
    ) -> None:
        materialization_peak = _METADATA_DECODE_MULTIPLIER * metadata_bytes
        if materialize:
            materialization_peak += 2 * (average_bytes + validity_bytes)
        if materialization_peak > self.diagnostic_memory_limit_bytes:
            raise MemoryError(
                f"calibration report diagnostics require {materialization_peak} bytes; "
                f"limit {self.diagnostic_memory_limit_bytes}"
            )

    def _report_storage_refs(
        self,
        artifact: CalibrationArtifact,
        reference: ContentRef,
        *,
        materialize: bool,
    ) -> tuple[bytes, ContentRef, ContentRef]:
        authority = self._content_authority()
        self._admit_diagnostic_memory(0, 0, reference.size, materialize=False)
        payload = authority.read_blob(reference, max_bytes=self.max_report_metadata_bytes)
        average_ref, validity_ref = calibration_report_blob_refs(payload)
        pixels = math.prod(artifact.frame_contract.frame_schema.data_shape)
        expected = (pixels * 8, pixels)
        if (average_ref.size, validity_ref.size) != expected:
            raise ValueError(
                "calibration diagnostic blob sizes differ from the FrameContract"
            )
        if materialize:
            self._admit_diagnostic_memory(*expected, len(payload), materialize=True)
        return payload, average_ref, validity_ref

    def _load_report_ref(
        self,
        artifact: CalibrationArtifact,
        reference: ContentRef,
    ) -> CalibrationReport:
        authority = self._content_authority()
        payload, average_ref, validity_ref = self._report_storage_refs(
            artifact,
            reference,
            materialize=True,
        )
        average, validity = decode_calibration_report_arrays(
            authority.read_blob(average_ref, max_bytes=average_ref.size),
            authority.read_blob(validity_ref, max_bytes=validity_ref.size),
            image_shape=artifact.frame_contract.frame_schema.data_shape,
        )
        report = decode_calibration_report(
            payload,
            reference_average=average,
            reference_average_validity=validity,
        )
        # Reuse the analysis owner's complete artifact/report binding check.
        CalibrationComputation(artifact, report)
        return report

    def load(self, reference: CalibrationArtifactRef) -> CalibrationArtifact:
        """Load an artifact after its FINAL commit is journal-linearized."""

        self._require_final_commit(reference)
        artifact, _report_ref = self._artifact_and_report_ref(reference)
        return artifact

    def load_report(self, reference: CalibrationArtifactRef) -> CalibrationReport:
        """Load the display/diagnostic report paired with an artifact."""

        self._require_final_commit(reference)
        artifact, report_ref = self._artifact_and_report_ref(reference)
        return self._load_report_ref(artifact, report_ref)

    def has(self, reference: CalibrationArtifactRef) -> bool:
        try:
            self._require_final_commit(reference)
        except PermissionError:
            return False
        return self._content_authority().has_manifest(
            CALIBRATION_ARTIFACT_NAMESPACE,
            reference.manifest_digest,
            max_bytes=_MAX_MANIFEST_BYTES,
        )

    @staticmethod
    def _validate_source_admission(
        artifact: CalibrationArtifact,
        source: "AdmittedCapture",
    ) -> _ResolvedCalibrationSource:
        from zlc_neutral_atom.artifacts.capture import AdmittedCapture

        if type(source) is not AdmittedCapture:
            raise TypeError("source must be an exact AdmittedCapture")
        return _validate_calibration_artifact_source_compatibility(
            artifact,
            source.artifact,
        )

    def admit(
        self,
        reference: CalibrationArtifactRef,
        capture_repository: "CaptureRepository",
    ) -> ResolvedCalibration:
        """Admit only a FINAL journal target whose raw source is still valid."""

        from zlc_neutral_atom.artifacts.capture import CaptureRepository

        if type(capture_repository) is not CaptureRepository:
            raise TypeError("capture_repository must be CaptureRepository")
        self._require_final_commit(reference)
        artifact, _report_ref = self._artifact_and_report_ref(reference)
        source = capture_repository.admit(
            artifact.source_binding.source_capture_ref
        )
        self._validate_source_admission(artifact, source)
        return ResolvedCalibration._from_admission(reference, artifact)

    def _stage_result(
        self,
        result: CalibrationAnalysisResult,
    ) -> tuple[CalibrationArtifactRef, bytes]:
        authority = self._content_authority()
        report = result.report
        # These two full-resolution payload sizes are known without copying a
        # pixel.  Reject an impossible diagnostic budget before encoding or
        # writing any large report blob; metadata gets an exact second check
        # once its comparatively small canonical payload exists.
        self._admit_diagnostic_memory(
            report.reference_average.nbytes,
            report.reference_average_validity.nbytes,
            0,
            materialize=True,
        )
        artifact_payload = encode_calibration_artifact(result.artifact)
        if len(artifact_payload) > _MAX_ARTIFACT_BYTES:
            raise MemoryError(
                f"calibration artifact requires {len(artifact_payload)} bytes; "
                f"limit {_MAX_ARTIFACT_BYTES}"
            )
        decode_calibration_artifact(artifact_payload)
        average_payload = encode_calibration_reference_average(
            report.reference_average
        )
        validity_payload = encode_calibration_reference_average_validity(
            report.reference_average_validity
        )
        average_blob = authority.identify_blob(average_payload)
        validity_blob = authority.identify_blob(validity_payload)
        report_payload = encode_calibration_report_metadata(
            report,
            reference_average_blob=average_blob,
            reference_average_validity_blob=validity_blob,
        )
        if len(report_payload) > self.max_report_metadata_bytes:
            raise MemoryError(
                f"calibration report metadata requires {len(report_payload)} bytes; "
                f"limit {self.max_report_metadata_bytes}"
            )
        self._admit_diagnostic_memory(
            report.reference_average.nbytes,
            report.reference_average_validity.nbytes,
            len(report_payload),
            materialize=True,
        )
        decode_calibration_report(
            report_payload,
            reference_average=report.reference_average,
            reference_average_validity=report.reference_average_validity,
        )
        if calibration_report_blob_refs(report_payload) != (
            average_blob,
            validity_blob,
        ):
            raise ValueError("calibration report failed its durable codec round-trip")
        # Only self-readable, fully admitted values reach the CAS.  Resource or
        # codec rejection above therefore cannot leave diagnostic orphan blobs.
        artifact_blob = authority.put_blob(artifact_payload)
        if authority.put_blob(average_payload) != average_blob:
            raise RuntimeError("calibration average content identity changed while staging")
        if authority.put_blob(validity_payload) != validity_blob:
            raise RuntimeError("calibration validity content identity changed while staging")
        report_blob = authority.put_blob(report_payload)
        payload = _manifest_payload(self.repository_id, artifact_blob, report_blob)
        reference = CalibrationArtifactRef(
            self.repository_id,
            sha256_digest(payload),
        )
        return reference, payload

    def final_commit(
        self,
        context: PostSafetyContext,
        result: CalibrationAnalysisResult,
        source: "AdmittedCapture",
    ) -> FinalCommit[CalibrationArtifactRef]:
        """Prepare the sole manifest publication after a source re-admission."""

        if not isinstance(context, PostSafetyContext):
            raise TypeError("calibration commit requires PostSafetyContext")
        if type(result) is not CalibrationAnalysisResult:
            raise TypeError("result must be CalibrationAnalysisResult")
        resolved = self._validate_source_admission(result.artifact, source)
        if not resolved.join.matches_contexts(result.report.group_contexts):
            raise ValueError(
                "calibration report group contexts differ from the admitted source"
            )
        result._require_source_admission(source)
        run_id, safety_bundle_id = context.authorize_commit_preparation()
        # Staging writes CAS blobs, so repository lifetime begins before the
        # first write and overlaps prepare() minting the commit-lifetime hold.
        with self._root_lease.borrow() as staging_borrow:
            staging_borrow.require_active()
            reference, payload = self._stage_result(result)
            confirmed = context.authorize_commit_preparation()
            if confirmed != (run_id, safety_bundle_id):
                raise RuntimeError("calibration commit subject changed while staging")
            target = _target(self.repository_id, reference)

            def publish() -> PublishedManifest[CalibrationArtifactRef]:
                authority = self._content_authority()
                publish_manifest_with_visibility_reconciliation(
                    authority,
                    CALIBRATION_ARTIFACT_NAMESPACE,
                    payload,
                    expected_digest=reference.manifest_digest,
                    max_bytes=_MAX_MANIFEST_BYTES,
                )
                return PublishedManifest(
                    reference.target_ref,
                    reference.manifest_digest,
                    reference,
                )

            with self._lock:
                self._require_open()
                operation = self._coordinator.prepare(
                    _commit_id(run_id, reference.manifest_digest),
                    run_id,
                    safety_bundle_id,
                    target,
                    publish,
                )
        try:
            context._track_prepared_commit(operation)
        except BaseException:
            operation.abandon()
            raise
        return operation

    def _recover(
        self,
        intent: CommitIntent,
    ) -> PublishedManifest[CalibrationArtifactRef] | None:
        """Resolve one pending intent by inspecting, never publishing, storage."""

        authority = self._content_authority()
        target = intent.target
        if (
            target.repository_id != self.repository_id
            or target.artifact_kind != _CALIBRATION_ARTIFACT_KIND
            or target.artifact_format != CALIBRATION_MANIFEST_FORMAT
        ):
            raise ValueError("commit intent is not a calibration target")
        reference = CalibrationArtifactRef(
            self.repository_id,
            target.expected_manifest_digest,
        )
        if target.target_ref != reference.target_ref:
            raise ValueError("calibration target ref and digest differ")
        if intent.commit_id != _commit_id(
            intent.run_id,
            reference.manifest_digest,
        ):
            raise ValueError("calibration commit id differs from its target")
        try:
            payload = authority.read_manifest(
                CALIBRATION_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
                max_bytes=_MAX_MANIFEST_BYTES,
            )
        except FileNotFoundError:
            return None
        # Once the visibility point exists, missing/corrupt blobs are a
        # repository fault and startup remains fail-closed.
        artifact, report_ref = self._artifact_and_report_ref(
            reference,
            manifest_payload=payload,
        )
        _report_payload, average_ref, validity_ref = self._report_storage_refs(
            artifact,
            report_ref,
            materialize=False,
        )
        authority.verify_blob(average_ref, max_bytes=average_ref.size)
        authority.verify_blob(validity_ref, max_bytes=validity_ref.size)
        confirmed = authority.confirm_manifest_durable(
            CALIBRATION_ARTIFACT_NAMESPACE,
            reference.manifest_digest,
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        if confirmed != payload:
            raise RuntimeError("recovery durability check changed manifest")
        return PublishedManifest(
            reference.target_ref,
            reference.manifest_digest,
            reference,
        )


def compile_calibration_artifact_plan(
    source_capture_ref: CaptureArtifactRef,
    capture_repository: "CaptureRepository",
    calibration_repository: CalibrationRepository,
    request: CalibrationAnalysisRequest,
    *,
    expected_readout_binding: ReadoutBindingKey,
    memory_limit_bytes: int,
    timeout_seconds: float,
) -> RunPlan:
    """Adapt one synchronous calibration calculation to the generic RunPlan."""

    from zlc_neutral_atom.artifacts.capture import AdmittedCapture, CaptureRepository

    if not isinstance(source_capture_ref, CaptureArtifactRef):
        raise TypeError("source_capture_ref must be CaptureArtifactRef")
    if type(capture_repository) is not CaptureRepository:
        raise TypeError("capture_repository must be CaptureRepository")
    if type(calibration_repository) is not CalibrationRepository:
        raise TypeError("calibration_repository must be CalibrationRepository")
    if not isinstance(request, CalibrationAnalysisRequest):
        raise TypeError("request must be CalibrationAnalysisRequest")
    if request.expected_centers_xy is None:
        raise ValueError(
            "authoritative calibration requires independent expected_centers_xy "
            "and maximum_site_residual_px"
        )
    if not isinstance(expected_readout_binding, ReadoutBindingKey):
        raise TypeError("expected_readout_binding must be ReadoutBindingKey")
    memory_limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    timeout = positive_real(timeout_seconds, "timeout_seconds")
    if source_capture_ref.repository_id != capture_repository.repository_id:
        raise ValueError("source capture belongs to another repository")

    def preflight(context: RunContext) -> _PreparedCalibrationAnalysis:
        context.checkpoint()
        source = capture_repository.admit(source_capture_ref)
        if source.artifact.camera_provenance.binding != expected_readout_binding:
            raise ValueError(
                "source capture readout binding differs from the frozen request"
            )
        frame_source = source.artifact.frame_source
        estimated_peak = estimate_calibration_analysis_peak_bytes(
            frame_source.schema,
            request,
            source_read_scratch_bytes=frame_source.max_read_scratch_bytes,
        )
        if estimated_peak > memory_limit:
            raise MemoryError(
                f"calibration analysis requires {estimated_peak} bytes; "
                f"limit {memory_limit}"
            )
        resolved = _resolve_calibration_source(source.artifact, request.layout)
        context.checkpoint()
        return _PreparedCalibrationAnalysis(source, resolved)

    def execute(
        context: RunContext,
        prepared: _PreparedCalibrationAnalysis,
    ) -> CalibrationAnalysisResult:
        if type(prepared) is not _PreparedCalibrationAnalysis:
            raise TypeError("calibration execute requires its preflight admission")
        context.checkpoint()
        result = _analyze_calibration_resolved(
            prepared.source,
            request,
            prepared.resolved,
        )
        context.checkpoint()
        return result

    def cleanup(
        _context: RunContext,
        _prepared: _PreparedCalibrationAnalysis | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        return CleanupReport()

    def finalize(
        context: PostSafetyContext,
        result: CalibrationAnalysisResult,
    ) -> CalibrationArtifactRef:
        # Re-admit at the publication boundary rather than trusting a stale
        # process-local preflight handle.
        source = capture_repository.admit(source_capture_ref)
        operation = calibration_repository.final_commit(context, result, source)
        return context.commit_final(operation)

    return RunPlan(
        name="calibrate committed camera capture",
        # The per-run estimator cannot make two concurrent 500 MiB analyses
        # safe in aggregate.  One flat non-device claim serializes this CPU and
        # memory-heavy owner without inventing a scheduler or workflow engine.
        resource_claims=(_CALIBRATION_ANALYSIS_CLAIM,),
        bound_devices=(),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        timeout_seconds=timeout,
        requires_final_commit=True,
    )


__all__ = [
    "CALIBRATION_MANIFEST_FORMAT",
    "CalibrationRepository",
    "compile_calibration_artifact_plan",
]
