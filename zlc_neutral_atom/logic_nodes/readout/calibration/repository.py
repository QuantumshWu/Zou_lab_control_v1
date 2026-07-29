"""Durable storage and admission for readout calibrations.

The repository has one job: make an already computed
``CalibrationAnalysisResult`` atomically visible.  Scientific validation lives
in the calibration/analysis values; canonical encoding lives in the
capability-local ``codec``; durability lives in ``zlc_storage``.  The canonical
CAS manifest is the sole durable visibility authority.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
import threading
from typing import TYPE_CHECKING

from zlc_storage import (
    ContentAddressedStore,
    ContentCorruptionError,
    ContentRef,
    ContentStoreAuthority,
    RepositoryRootLease,
    RepositoryRootLeaseBorrow,
    canonical_text,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping,
    positive_real,
    sha256_digest,
)

from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
)
from zlc_neutral_atom.runtime.commit import (
    PreparedArtifactCommit,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunPlan,
)
from .calibration import (
    CalibrationAnalysisRequest,
    CalibrationArtifact,
    ResolvedCalibration,
    _RESOLVED_CALIBRATION_TOKEN,
    _ResolvedCalibrationSource,
    _resolve_calibration_source,
    _validate_calibration_artifact_source_compatibility,
)
from .codec import (
    calibration_report_blob_refs,
    decode_calibration_artifact,
    decode_calibration_report,
    decode_calibration_report_arrays,
    encode_calibration_artifact,
    encode_calibration_reference_average,
    encode_calibration_reference_average_validity,
    encode_calibration_report_metadata,
)
from .reference import (
    CALIBRATION_ARTIFACT_NAMESPACE,
    CalibrationArtifactRef,
)
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.runtime.resources import (
    acquire_repository_borrows,
    release_repository_borrows,
)

if TYPE_CHECKING:
    from zlc_neutral_atom.capture.artifact import (
        AdmittedCapture,
        CaptureRepository,
    )
    from .analysis import (
        CalibrationAnalysisResult,
        CalibrationComputation,
        CalibrationReport,
    )


CALIBRATION_MANIFEST_FORMAT = "zlc_neutral_atom.logic_nodes.readout.calibration.manifest"
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "repository_id",
        "run_id",
        "artifact_blob",
        "report_blob",
    }
)
_PreparedCalibrationAnalysis = tuple[
    object,
    _ResolvedCalibrationSource,
    tuple[RepositoryRootLeaseBorrow, ...],
]


def _manifest_payload(
    repository_id: str,
    run_id: str,
    artifact_blob: ContentRef,
    report_blob: ContentRef,
) -> bytes:
    """Encode the sole current manifest shape.

    The format and repository identify the durable owner; ``run_id`` carries
    the artifact's execution provenance, and the two content references bind
    the scientific result and its report.
    """

    return encode(
        {
            "format": CALIBRATION_MANIFEST_FORMAT,
            "repository_id": repository_id,
            "run_id": canonical_text(run_id, "run_id"),
            "artifact_blob": content_ref_to_tree(artifact_blob),
            "report_blob": content_ref_to_tree(report_blob),
        },
    )


def _decode_manifest(
    payload: bytes,
) -> tuple[str, str, ContentRef, ContentRef]:
    if not isinstance(payload, bytes):
        raise TypeError("calibration manifest payload must be bytes")
    tree = exact_mapping(
        decode(payload),
        _MANIFEST_FIELDS,
        CALIBRATION_MANIFEST_FORMAT,
        discriminator="format",
    )
    repository_id = canonical_text(tree["repository_id"], "repository_id")
    run_id = canonical_text(tree["run_id"], "run_id")
    artifact_blob = content_ref_from_tree(tree["artifact_blob"])
    report_blob = content_ref_from_tree(tree["report_blob"])
    if (
        _manifest_payload(
            repository_id,
            run_id,
            artifact_blob,
            report_blob,
        )
        != payload
    ):
        raise ValueError("calibration manifest is not canonical current format")
    return repository_id, run_id, artifact_blob, report_blob


class CalibrationRepository:
    """Content-addressed calibration store with manifest-only visibility."""

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-calibration",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = canonical_text(repository_id, "repository_id")
        self._lock = threading.RLock()
        self._closed = False
        self._root_lease = RepositoryRootLease(self.root)
        try:
            self._store = ContentAddressedStore(self.root / "content")
            self._store_authority = self._store.authority()
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

    def _read_manifest(
        self,
        reference: CalibrationArtifactRef,
    ) -> bytes:
        return self._content_authority().read_manifest(
            CALIBRATION_ARTIFACT_NAMESPACE,
            reference.manifest_digest,
        )

    def _content_authority(self) -> ContentStoreAuthority:
        with self._lock:
            self._require_open()
            return self._store_authority

    def _storage_refs(
        self,
        reference: CalibrationArtifactRef,
        *,
        manifest_payload: bytes | None = None,
    ) -> tuple[str, ContentRef, ContentRef]:
        self._validate_reference(reference)
        payload = (
            self._read_manifest(reference)
            if manifest_payload is None
            else manifest_payload
        )
        repository_id, run_id, artifact_ref, report_ref = _decode_manifest(payload)
        if repository_id != self.repository_id:
            raise ValueError("calibration manifest belongs to another repository")
        return run_id, artifact_ref, report_ref

    def _materialize_artifact(
        self,
        artifact_ref: ContentRef,
    ) -> CalibrationArtifact:
        try:
            payload = self._content_authority().read_blob(artifact_ref)
        except FileNotFoundError as error:
            raise ContentCorruptionError(
                "visible calibration manifest references a missing artifact blob"
            ) from error
        return decode_calibration_artifact(payload)

    def _report_storage_refs(
        self,
        artifact: CalibrationArtifact,
        reference: ContentRef,
    ) -> tuple[bytes, ContentRef, ContentRef]:
        authority = self._content_authority()
        try:
            payload = authority.read_blob(reference)
        except FileNotFoundError as error:
            raise ContentCorruptionError(
                "visible calibration manifest references a missing report blob"
            ) from error
        average_ref, validity_ref = calibration_report_blob_refs(payload)
        pixels = math.prod(artifact.frame_contract.frame_schema.data_shape)
        expected = (pixels * 8, pixels)
        if (average_ref.size, validity_ref.size) != expected:
            raise ValueError(
                "calibration diagnostic blob sizes differ from the FrameContract"
            )
        return payload, average_ref, validity_ref

    def _load_computation_ref(
        self,
        artifact: CalibrationArtifact,
        reference: ContentRef,
    ) -> CalibrationComputation:
        from .analysis import CalibrationComputation

        authority = self._content_authority()
        payload, average_ref, validity_ref = self._report_storage_refs(
            artifact,
            reference,
        )
        try:
            average_payload = authority.read_blob(average_ref)
            validity_payload = authority.read_blob(validity_ref)
        except FileNotFoundError as error:
            raise ContentCorruptionError(
                "calibration report references missing array content"
            ) from error
        average, validity = decode_calibration_report_arrays(
            average_payload,
            validity_payload,
            image_shape=artifact.frame_contract.frame_schema.data_shape,
        )
        report = decode_calibration_report(
            payload,
            reference_average=average,
            reference_average_validity=validity,
        )
        # The analysis owner performs the complete artifact/report binding check.
        return CalibrationComputation(artifact, report)

    def load(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationArtifact:
        """Load a calibration named by an exact visible manifest."""

        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            _run_id, artifact_ref, _report_ref = self._storage_refs(reference)
            return self._materialize_artifact(artifact_ref)

    def load_computation(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationComputation:
        """Load one validated artifact/report pair."""

        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            _run_id, artifact_ref, report_ref = self._storage_refs(reference)
            artifact = self._materialize_artifact(artifact_ref)
            return self._load_computation_ref(artifact, report_ref)

    def load_report(
        self,
        reference: CalibrationArtifactRef,
    ) -> CalibrationReport:
        """Load diagnostics for one visible calibration manifest."""

        return self.load_computation(reference).report

    def has(self, reference: CalibrationArtifactRef) -> bool:
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            self._validate_reference(reference)
            try:
                payload = self._read_manifest(reference)
            except FileNotFoundError:
                return False
            try:
                _run_id, artifact_ref, report_ref = self._storage_refs(
                    reference,
                    manifest_payload=payload,
                )
                artifact = self._materialize_artifact(artifact_ref)
                _report_payload, average_ref, validity_ref = self._report_storage_refs(
                    artifact,
                    report_ref,
                )
                authority = self._content_authority()
                authority.verify_blob(average_ref)
                authority.verify_blob(validity_ref)
            except FileNotFoundError as error:
                raise ContentCorruptionError(
                    "visible calibration manifest references missing content"
                ) from error
            return True

    @staticmethod
    def _validate_source_admission(
        artifact: CalibrationArtifact,
        source: "AdmittedCapture",
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> _ResolvedCalibrationSource:
        from zlc_neutral_atom.capture.artifact import AdmittedCapture

        if type(source) is not AdmittedCapture:
            raise TypeError("source must be an exact AdmittedCapture")
        return _validate_calibration_artifact_source_compatibility(
            artifact,
            source.artifact,
            checkpoint=checkpoint,
        )

    def admit(
        self,
        reference: CalibrationArtifactRef,
        capture_repository: "CaptureRepository",
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> ResolvedCalibration:
        """Admit a visible target and validate its persisted source."""

        from zlc_neutral_atom.capture.artifact import CaptureRepository

        if type(capture_repository) is not CaptureRepository:
            raise TypeError("capture_repository must be CaptureRepository")
        with self._root_lease.borrow() as admission_borrow:
            admission_borrow.require_active()
            with capture_repository._root_lease.borrow() as source_borrow:
                source_borrow.require_active()
                _run_id, artifact_ref, _report_ref = self._storage_refs(reference)
                artifact = self._materialize_artifact(artifact_ref)
                source_capture_ref = artifact.source_binding.source_capture_ref
                source = capture_repository.admit(source_capture_ref)
                self._validate_source_admission(
                    artifact,
                    source,
                    checkpoint=checkpoint,
                )
                return ResolvedCalibration._from_admission(
                    _RESOLVED_CALIBRATION_TOKEN,
                    repository_token=self._root_lease,
                    reference=reference,
                    artifact=artifact,
                )

    def _stage_result(
        self,
        result: CalibrationAnalysisResult,
        *,
        run_id: str,
    ) -> tuple[CalibrationArtifactRef, bytes]:
        authority = self._content_authority()
        report = result.report
        artifact_payload = encode_calibration_artifact(result.artifact)
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
        payload = _manifest_payload(
            self.repository_id,
            run_id,
            artifact_blob,
            report_blob,
        )
        reference = CalibrationArtifactRef(
            self.repository_id,
            sha256_digest(payload),
        )
        return reference, payload

    def final_commit(
        self,
        context: PostSafetyContext,
        result: CalibrationAnalysisResult,
    ) -> PreparedArtifactCommit[CalibrationArtifactRef]:
        """Prepare publication from the exact admission retained by analysis."""

        from .analysis import CalibrationAnalysisResult

        if not isinstance(context, PostSafetyContext):
            raise TypeError("calibration commit requires PostSafetyContext")
        if type(result) is not CalibrationAnalysisResult:
            raise TypeError("result must be CalibrationAnalysisResult")
        source, resolved = result._source_for_commit()
        source._require_authority()
        if not resolved.join.matches_contexts(result.report.group_contexts):
            raise ValueError(
                "calibration report group contexts differ from the admitted source"
            )
        run_id = context.authorize_commit_preparation()
        # Staging writes CAS blobs, so repository lifetime begins before the
        # first write and overlaps prepare() minting the commit-lifetime hold.
        with self._root_lease.borrow() as staging_borrow:
            staging_borrow.require_active()
            reference, payload = self._stage_result(result, run_id=run_id)
            confirmed = context.authorize_commit_preparation()
            if confirmed != run_id:
                raise RuntimeError("calibration commit subject changed while staging")
            commit_borrow = self._root_lease.borrow()
        try:
            def publish(manifest_payload: bytes) -> None:
                if manifest_payload != payload:
                    raise ValueError("calibration commit payload changed after staging")
                self._content_authority().publish_manifest(
                    CALIBRATION_ARTIFACT_NAMESPACE,
                    manifest_payload,
                    expected_digest=reference.manifest_digest,
                )

            def inspect(manifest_payload: bytes) -> bool | None:
                if manifest_payload != payload:
                    raise ValueError("calibration inspection payload changed")
                authority = self._content_authority()
                try:
                    confirmed_payload = authority.confirm_manifest_durable(
                        CALIBRATION_ARTIFACT_NAMESPACE,
                        reference.manifest_digest,
                    )
                except FileNotFoundError:
                    return False
                except OSError:
                    return None
                if confirmed_payload != manifest_payload:
                    raise ContentCorruptionError(
                        "calibration manifest differs from its immutable reference"
                    )
                try:
                    stored_run_id, artifact_ref, report_ref = self._storage_refs(
                        reference,
                        manifest_payload=confirmed_payload,
                    )
                    if stored_run_id != run_id:
                        raise ValueError(
                            "visible calibration provenance belongs to another Run"
                        )
                    artifact = self._materialize_artifact(artifact_ref)
                    _report_payload, average_ref, validity_ref = (
                        self._report_storage_refs(artifact, report_ref)
                    )
                    authority.verify_blob(average_ref)
                    authority.verify_blob(validity_ref)
                except FileNotFoundError as error:
                    raise ContentCorruptionError(
                        "visible calibration manifest references missing content"
                    ) from error
                except OSError:
                    return None
                return True

            operation = PreparedArtifactCommit(
                run_id=run_id,
                result=reference,
                manifest_payload=payload,
                publish=publish,
                inspect=inspect,
                repository_borrow=commit_borrow,
            )
        except BaseException:
            commit_borrow.close()
            raise
        try:
            context.track_prepared_commit(operation)
        except BaseException:
            operation.abandon()
            raise
        return operation


def compile_calibration_artifact_plan(
    source_capture_ref: CaptureArtifactRef,
    capture_repository: "CaptureRepository",
    calibration_repository: CalibrationRepository,
    request: CalibrationAnalysisRequest,
    *,
    expected_readout_binding: ReadoutBindingKey,
    timeout_seconds: float,
    on_committed: Callable[[CalibrationArtifactRef], None] | None = None,
) -> RunPlan:
    """Adapt one synchronous calibration calculation to the generic RunPlan."""

    from zlc_neutral_atom.capture.artifact import CaptureRepository
    from .analysis import (
        CalibrationAnalysisResult,
        _analyze_calibration_resolved,
    )

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
    if on_committed is not None and not callable(on_committed):
        raise TypeError("on_committed must be callable or None")
    timeout = positive_real(timeout_seconds, "timeout_seconds")
    if source_capture_ref.repository_id != capture_repository.repository_id:
        raise ValueError("source capture belongs to another repository")

    def preflight(context: RunContext) -> _PreparedCalibrationAnalysis:
        borrows = acquire_repository_borrows(
            capture_repository._root_lease,
            calibration_repository._root_lease,
        )
        try:
            context.checkpoint()
            source = capture_repository.admit(source_capture_ref)
            if source.artifact.camera_provenance.binding != expected_readout_binding:
                raise ValueError(
                    "source capture readout binding differs from the frozen request"
                )
            resolved = _resolve_calibration_source(
                source.artifact,
                request.layout,
                checkpoint=context.checkpoint,
            )
            context.checkpoint()
            return (
                source,
                resolved,
                borrows,
            )
        except BaseException as primary:
            try:
                release_repository_borrows(borrows)
            except BaseException as close_error:
                record_secondary_failure(
                    primary,
                    "repository borrow release also failed",
                    close_error,
                )
            raise

    def execute(
        context: RunContext,
        prepared: _PreparedCalibrationAnalysis,
    ) -> tuple[
        CalibrationAnalysisResult,
        tuple[RepositoryRootLeaseBorrow, ...],
    ]:
        (
            source,
            resolved,
            borrows,
        ) = prepared
        context.checkpoint()
        result = _analyze_calibration_resolved(
            source,
            request,
            resolved,
        )
        context.checkpoint()
        return result, borrows

    def cleanup(
        _context: RunContext,
        prepared: _PreparedCalibrationAnalysis | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is not None and primary is not None:
            (
                _source,
                _resolved,
                borrows,
            ) = prepared
            release_repository_borrows(borrows)
        return CleanupReport()

    def finalize(
        context: PostSafetyContext,
        executed: tuple[
            CalibrationAnalysisResult,
            tuple[RepositoryRootLeaseBorrow, ...],
        ],
    ) -> CalibrationArtifactRef:
        result, borrows = executed
        try:
            for borrow in borrows:
                borrow.require_active()
            operation = calibration_repository.final_commit(context, result)
            reference = context.commit_final(operation)
            if on_committed is not None:
                on_committed(reference)
            return reference
        finally:
            release_repository_borrows(borrows)

    def dispose_unfinalized(
        executed: tuple[
            CalibrationAnalysisResult,
            tuple[RepositoryRootLeaseBorrow, ...],
        ],
    ) -> None:
        _result, borrows = executed
        release_repository_borrows(borrows)

    return RunPlan(
        name="calibrate committed camera capture",
        resource_claims=(),
        bound_devices=(),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        timeout_seconds=timeout,
        requires_final_commit=True,
        dispose_unfinalized=dispose_unfinalized,
    )


__all__ = [
    "CALIBRATION_MANIFEST_FORMAT",
    "CalibrationRepository",
    "compile_calibration_artifact_plan",
]
