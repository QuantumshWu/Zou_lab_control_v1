"""Durable storage and admission for readout calibrations.

The repository has one job: make an already computed
``CalibrationAnalysisResult`` atomically visible.  Scientific validation lives
in the calibration/analysis values; canonical encoding lives in
``calibration_codec``; durability lives in ``zlc_storage`` and the generic
commit coordinator.  This module deliberately adds no second proof graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import math
from pathlib import Path
import threading
from typing import TYPE_CHECKING

from zlc_storage import (
    CanonicalDecodeLimits,
    ContentAddressedStore,
    ContentRef,
    ContentSizeLimitError,
    ContentStoreAuthority,
    RepositoryRootLease,
    RepositoryRootLeaseBorrow,
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

from zlc_neutral_atom.capture_reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
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
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunPlan,
)
from .calibration import (
    CalibrationAnalysisRequest,
    CalibrationArtifact,
    ReadoutModelKind,
    ResolvedCalibration,
    _RESOLVED_CALIBRATION_TOKEN,
    _ResolvedCalibrationSource,
    _resolve_calibration_source,
    _validate_calibration_artifact_source_compatibility,
    calibration_retained_array_nbytes,
    readout_runtime_scratch_nbytes,
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
    estimate_calibration_context_roundtrip_workspace,
)
from .calibration_reference import (
    CALIBRATION_ARTIFACT_NAMESPACE,
    CalibrationArtifactRef,
)
from .contracts import ReadoutBindingKey
from .codec import readout_binding_key_from_tree, readout_binding_key_to_tree
from .physical_context import (
    estimate_readout_physical_context_peak_from_summary,
    estimate_readout_physical_context_retained_from_summary,
    readout_physical_context_retained_upper_bound_bytes,
)
from .runtime_resources import (
    READOUT_ANALYSIS_CLAIM,
    acquire_repository_borrows,
    release_repository_borrows,
)

if TYPE_CHECKING:
    from zlc_neutral_atom.artifacts.capture import AdmittedCapture, CaptureRepository
    from .analysis import (
        CalibrationAnalysisResult,
        CalibrationReport,
    )


CALIBRATION_MANIFEST_FORMAT = "zlc_neutral_atom.calibration-manifest.v2"
_CALIBRATION_ARTIFACT_KIND = "calibration"
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_REPORT_METADATA_BYTES = 64 * 1024 * 1024
_DEFAULT_MEMORY_LIMIT_BYTES = 1 << 30
_MANIFEST_INSPECTION_FIXED_BYTES = 64 * 1024
_MANIFEST_MATERIALIZATION_MULTIPLIER = 8
_METADATA_DECODE_MULTIPLIER = 8
_CANONICAL_ARTIFACT_MATERIALIZATION_MULTIPLIER = 16
_ARTIFACT_RETAINED_FIXED_BYTES = 512 * 1024
_ARTIFACT_SITE_MODEL_BYTES = 2_048
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "repository_id",
        "artifact_blob",
        "report_blob",
        "runtime_summary",
    }
)
_MANIFEST_DECODE_LIMITS = CanonicalDecodeLimits(
    max_depth=24,
    max_nodes=512,
    max_container_entries=256,
    max_arrays=0,
    max_total_array_bytes=0,
)


def _report_materialization_peak_from_sizes(
    average_bytes: int,
    validity_bytes: int,
    metadata_bytes: int,
) -> int:
    sizes = []
    for value, field in (
        (average_bytes, "average_bytes"),
        (validity_bytes, "validity_bytes"),
        (metadata_bytes, "metadata_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        sizes.append(value)
    average_size, validity_size, metadata_size = sizes
    return (
        _METADATA_DECODE_MULTIPLIER * metadata_size
        + 2 * (average_size + validity_size)
    )


def _artifact_report_load_peak(summary: "CalibrationRuntimeSummary") -> int:
    if not isinstance(summary, CalibrationRuntimeSummary):
        raise TypeError("summary must be CalibrationRuntimeSummary")
    return max(
        summary.artifact_decode_peak_upper_bound_bytes,
        summary.artifact_retained_upper_bound_bytes
        + summary.report_materialization_peak_upper_bound_bytes,
    )


def _artifact_retained_upper_bound(
    artifact: CalibrationArtifact,
    *,
    artifact_blob_bytes: int | None = None,
) -> int:
    """Bound the decoded artifact, including non-array canonical values.

    The domain term accounts for the arrays and per-site/model object graph.
    Once canonical bytes exist, the wire-size term also covers strings,
    physical-context transitions, and other scalar/container state that is not
    represented by ``ndarray.nbytes``.  Keeping this calculation beside the
    manifest summary prevents load, admission, and staging from drifting.
    """

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    present_kinds = {model.kind for model in artifact.models}
    domain_bound = (
        _ARTIFACT_RETAINED_FIXED_BYTES
        + calibration_retained_array_nbytes(artifact)
        + artifact.site_map.site_axis.size
        * len(present_kinds)
        * _ARTIFACT_SITE_MODEL_BYTES
    )
    if artifact_blob_bytes is None:
        return domain_bound
    size = positive_integer(artifact_blob_bytes, "artifact_blob_bytes")
    return max(
        domain_bound,
        _CANONICAL_ARTIFACT_MATERIALIZATION_MULTIPLIER * size,
    )


def _artifact_decode_peak_upper_bound(
    artifact: CalibrationArtifact,
    *,
    artifact_blob_bytes: int,
) -> int:
    size = positive_integer(artifact_blob_bytes, "artifact_blob_bytes")
    retained = _artifact_retained_upper_bound(
        artifact,
        artifact_blob_bytes=size,
    )
    return (
        _CANONICAL_ARTIFACT_MATERIALIZATION_MULTIPLIER * size
        + 2 * retained
    )


@dataclass(frozen=True, slots=True)
class CalibrationRuntimeSummary:
    """Small fail-fast facts recomputed from the fully decoded artifact."""

    source_capture_ref: CaptureArtifactRef
    readout_binding: ReadoutBindingKey
    site_count: int
    model_kinds: tuple[ReadoutModelKind, ...]
    default_model_kind: ReadoutModelKind
    retained_array_nbytes: int
    runtime_scratch_nbytes_by_model: tuple[tuple[ReadoutModelKind, int], ...]
    artifact_retained_upper_bound_bytes: int
    artifact_decode_peak_upper_bound_bytes: int
    report_materialization_peak_upper_bound_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        object.__setattr__(
            self,
            "site_count",
            positive_integer(self.site_count, "site_count"),
        )
        kinds = tuple(self.model_kinds)
        if not kinds or any(not isinstance(item, ReadoutModelKind) for item in kinds):
            raise TypeError("model_kinds must contain ReadoutModelKind values")
        if len(kinds) != len(set(kinds)):
            raise ValueError("model_kinds must be unique")
        if self.default_model_kind not in kinds:
            raise ValueError("default_model_kind must be present in model_kinds")
        scratch = tuple(tuple(item) for item in self.runtime_scratch_nbytes_by_model)
        if any(len(item) != 2 for item in scratch) or (
            tuple(item[0] for item in scratch) != kinds
        ) or any(
            not isinstance(item[0], ReadoutModelKind)
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 0
            for item in scratch
        ):
            raise ValueError("runtime scratch rows must align exactly with model_kinds")
        for field in (
            "retained_array_nbytes",
            "artifact_retained_upper_bound_bytes",
            "artifact_decode_peak_upper_bound_bytes",
            "report_materialization_peak_upper_bound_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.artifact_retained_upper_bound_bytes < self.retained_array_nbytes:
            raise ValueError("artifact retained bound is smaller than its arrays")
        if (
            self.artifact_decode_peak_upper_bound_bytes
            < self.artifact_retained_upper_bound_bytes
        ):
            raise ValueError("artifact decode bound is smaller than retained state")
        if self.report_materialization_peak_upper_bound_bytes == 0:
            raise ValueError("report materialization bound must be positive")
        object.__setattr__(self, "model_kinds", kinds)
        object.__setattr__(self, "runtime_scratch_nbytes_by_model", scratch)

    @property
    def inspection_retained_upper_bound_bytes(self) -> int:
        """Bound this compact summary while a later dependency is inspected."""

        return (
            _MANIFEST_INSPECTION_FIXED_BYTES
            + _MANIFEST_MATERIALIZATION_MULTIPLIER
            * len(encode(_runtime_summary_to_tree(self)))
        )


_PreparedCalibrationAnalysis = tuple[
    object,
    _ResolvedCalibrationSource,
    int,
    int,
    tuple[RepositoryRootLeaseBorrow, ...],
]


def _runtime_summary(
    artifact: CalibrationArtifact,
    *,
    artifact_blob_bytes: int,
    report_metadata_blob_bytes: int,
) -> CalibrationRuntimeSummary:
    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    size = positive_integer(artifact_blob_bytes, "artifact_blob_bytes")
    report_size = positive_integer(
        report_metadata_blob_bytes,
        "report_metadata_blob_bytes",
    )
    present_kinds = {model.kind for model in artifact.models}
    model_kinds = tuple(kind for kind in ReadoutModelKind if kind in present_kinds)
    retained_arrays = calibration_retained_array_nbytes(artifact)
    retained_upper = _artifact_retained_upper_bound(
        artifact,
        artifact_blob_bytes=size,
    )
    return CalibrationRuntimeSummary(
        artifact.source_binding.source_capture_ref,
        artifact.frame_contract.binding,
        artifact.site_map.site_axis.size,
        model_kinds,
        artifact.default_model_kind,
        retained_arrays,
        tuple(
            (kind, readout_runtime_scratch_nbytes(artifact, kind))
            for kind in model_kinds
        ),
        retained_upper,
        _artifact_decode_peak_upper_bound(
            artifact,
            artifact_blob_bytes=size,
        ),
        _report_materialization_peak_from_sizes(
            8 * math.prod(artifact.frame_contract.frame_schema.data_shape),
            math.prod(artifact.frame_contract.frame_schema.data_shape),
            report_size,
        ),
    )


def _runtime_summary_to_tree(
    value: CalibrationRuntimeSummary,
) -> dict[str, object]:
    if not isinstance(value, CalibrationRuntimeSummary):
        raise TypeError("value must be CalibrationRuntimeSummary")
    return {
        "source_capture_ref": capture_artifact_ref_to_tree(
            value.source_capture_ref
        ),
        "readout_binding": readout_binding_key_to_tree(value.readout_binding),
        "site_count": value.site_count,
        "model_kinds": [item.value for item in value.model_kinds],
        "default_model_kind": value.default_model_kind.value,
        "retained_array_nbytes": value.retained_array_nbytes,
        "runtime_scratch_nbytes_by_model": [
            [kind.value, size]
            for kind, size in value.runtime_scratch_nbytes_by_model
        ],
        "artifact_retained_upper_bound_bytes": (
            value.artifact_retained_upper_bound_bytes
        ),
        "artifact_decode_peak_upper_bound_bytes": (
            value.artifact_decode_peak_upper_bound_bytes
        ),
        "report_materialization_peak_upper_bound_bytes": (
            value.report_materialization_peak_upper_bound_bytes
        ),
    }


def _runtime_summary_from_tree(tree: object) -> CalibrationRuntimeSummary:
    data = exact_mapping(
        tree,
        {
            "source_capture_ref",
            "readout_binding",
            "site_count",
            "model_kinds",
            "default_model_kind",
            "retained_array_nbytes",
            "runtime_scratch_nbytes_by_model",
            "artifact_retained_upper_bound_bytes",
            "artifact_decode_peak_upper_bound_bytes",
            "report_materialization_peak_upper_bound_bytes",
        },
        "CalibrationRuntimeSummary",
        discriminator=None,
    )
    model_kinds = data["model_kinds"]
    scratch = data["runtime_scratch_nbytes_by_model"]
    if not isinstance(model_kinds, list) or not isinstance(scratch, list):
        raise TypeError("calibration runtime model summaries must be lists")
    scratch_rows = []
    for item in scratch:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("calibration runtime scratch row must have two items")
        scratch_rows.append((ReadoutModelKind(item[0]), item[1]))
    return CalibrationRuntimeSummary(
        capture_artifact_ref_from_tree(data["source_capture_ref"]),
        readout_binding_key_from_tree(data["readout_binding"]),
        data["site_count"],
        tuple(ReadoutModelKind(item) for item in model_kinds),
        ReadoutModelKind(data["default_model_kind"]),
        data["retained_array_nbytes"],
        tuple(scratch_rows),
        data["artifact_retained_upper_bound_bytes"],
        data["artifact_decode_peak_upper_bound_bytes"],
        data["report_materialization_peak_upper_bound_bytes"],
    )


def _manifest_payload(
    repository_id: str,
    artifact_blob: ContentRef,
    report_blob: ContentRef,
    runtime_summary: CalibrationRuntimeSummary,
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
            "runtime_summary": _runtime_summary_to_tree(runtime_summary),
        },
        limits=_MANIFEST_DECODE_LIMITS,
    )


def _decode_manifest(
    payload: bytes,
) -> tuple[str, ContentRef, ContentRef, CalibrationRuntimeSummary]:
    if not isinstance(payload, bytes):
        raise TypeError("calibration manifest payload must be bytes")
    tree = exact_mapping(
        decode(payload, limits=_MANIFEST_DECODE_LIMITS),
        _MANIFEST_FIELDS,
        CALIBRATION_MANIFEST_FORMAT,
        discriminator="format",
    )
    repository_id = canonical_text(tree["repository_id"], "repository_id")
    artifact_blob = content_ref_from_tree(tree["artifact_blob"])
    report_blob = content_ref_from_tree(tree["report_blob"])
    runtime_summary = _runtime_summary_from_tree(tree["runtime_summary"])
    if (
        _manifest_payload(
            repository_id,
            artifact_blob,
            report_blob,
            runtime_summary,
        )
        != payload
    ):
        raise ValueError("calibration manifest is not canonical current format")
    return repository_id, artifact_blob, report_blob, runtime_summary


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
        memory_limit_bytes: int = _DEFAULT_MEMORY_LIMIT_BYTES,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = canonical_text(repository_id, "repository_id")
        self.max_report_metadata_bytes = positive_integer(
            max_report_metadata_bytes,
            "max_report_metadata_bytes",
        )
        self.memory_limit_bytes = positive_integer(
            memory_limit_bytes,
            "memory_limit_bytes",
        )
        self._lock = threading.RLock()
        self._closed = False
        self._root_lease = RepositoryRootLease(self.root)
        journal = None
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
            if journal is not None:
                journal.close()
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
            self._coordinator.close()
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
        *,
        memory_limit_bytes: int | None = None,
    ) -> bytes:
        limit = self._require_memory_budget(
            _MANIFEST_INSPECTION_FIXED_BYTES
            + _MANIFEST_MATERIALIZATION_MULTIPLIER,
            "calibration manifest inspection",
            memory_limit_bytes=memory_limit_bytes,
        )
        max_payload = min(
            _MAX_MANIFEST_BYTES,
            (limit - _MANIFEST_INSPECTION_FIXED_BYTES)
            // _MANIFEST_MATERIALIZATION_MULTIPLIER,
        )
        try:
            payload = self._content_authority().read_manifest(
                CALIBRATION_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
                max_bytes=max_payload,
            )
        except ContentSizeLimitError as exc:
            raise MemoryError(
                "calibration manifest inspection exceeds memory limit"
            ) from exc
        self._require_memory_budget(
            _MANIFEST_INSPECTION_FIXED_BYTES
            + _MANIFEST_MATERIALIZATION_MULTIPLIER * len(payload),
            "calibration manifest inspection",
            memory_limit_bytes=limit,
        )
        return payload

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
            matching = self._coordinator.committed_for(target)
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

    def _storage_refs(
        self,
        reference: CalibrationArtifactRef,
        *,
        manifest_payload: bytes | None = None,
        memory_limit_bytes: int | None = None,
    ) -> tuple[ContentRef, ContentRef, CalibrationRuntimeSummary]:
        payload = (
            self._read_manifest(
                reference,
                memory_limit_bytes=memory_limit_bytes,
            )
            if manifest_payload is None
            else manifest_payload
        )
        self._require_memory_budget(
            _MANIFEST_INSPECTION_FIXED_BYTES
            + _MANIFEST_MATERIALIZATION_MULTIPLIER * len(payload),
            "calibration manifest inspection",
            memory_limit_bytes=memory_limit_bytes,
        )
        repository_id, artifact_ref, report_ref, runtime_summary = _decode_manifest(
            payload
        )
        if repository_id != self.repository_id:
            raise ValueError("calibration manifest belongs to another repository")
        if artifact_ref.size > _MAX_ARTIFACT_BYTES:
            raise MemoryError("calibration artifact exceeds repository policy")
        if report_ref.size > self.max_report_metadata_bytes:
            raise MemoryError("calibration report metadata exceeds repository policy")
        expected_decode_peak = (
            _CANONICAL_ARTIFACT_MATERIALIZATION_MULTIPLIER
            * artifact_ref.size
            + 2 * runtime_summary.artifact_retained_upper_bound_bytes
        )
        if (
            runtime_summary.artifact_decode_peak_upper_bound_bytes
            != expected_decode_peak
        ):
            raise ValueError(
                "calibration runtime summary differs from its artifact blob size"
            )
        return artifact_ref, report_ref, runtime_summary

    def _materialize_artifact(
        self,
        artifact_ref: ContentRef,
        report_ref: ContentRef,
        runtime_summary: CalibrationRuntimeSummary,
    ) -> CalibrationArtifact:
        authority = self._content_authority()
        artifact_payload = authority.read_blob(
            artifact_ref,
            max_bytes=artifact_ref.size,
        )
        artifact = decode_calibration_artifact(artifact_payload)
        if _runtime_summary(
            artifact,
            artifact_blob_bytes=len(artifact_payload),
            report_metadata_blob_bytes=report_ref.size,
        ) != runtime_summary:
            raise ValueError(
                "calibration runtime summary differs from the decoded artifact"
            )
        return artifact

    def _require_memory_budget(
        self,
        required_bytes: int,
        operation: str,
        *,
        memory_limit_bytes: int | None,
    ) -> int:
        if isinstance(required_bytes, bool) or not isinstance(required_bytes, int):
            raise TypeError("required_bytes must be an integer")
        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")
        limit = (
            self.memory_limit_bytes
            if memory_limit_bytes is None
            else min(
                self.memory_limit_bytes,
                positive_integer(memory_limit_bytes, "memory_limit_bytes"),
            )
        )
        if required_bytes > limit:
            raise MemoryError(
                f"{operation} requires {required_bytes} bytes; limit {limit}"
            )
        return limit

    def inspect_final(
        self,
        reference: CalibrationArtifactRef,
        *,
        memory_limit_bytes: int | None = None,
    ) -> CalibrationRuntimeSummary:
        """Read FINAL fail-fast facts without decoding calibration arrays."""

        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            self._require_final_commit(reference)
            payload = self._read_manifest(
                reference,
                memory_limit_bytes=memory_limit_bytes,
            )
            _artifact_ref, _report_ref, summary = self._storage_refs(
                reference,
                manifest_payload=payload,
                memory_limit_bytes=memory_limit_bytes,
            )
            return summary

    def _report_storage_refs(
        self,
        artifact: CalibrationArtifact,
        reference: ContentRef,
    ) -> tuple[bytes, ContentRef, ContentRef]:
        authority = self._content_authority()
        payload = authority.read_blob(reference, max_bytes=reference.size)
        average_ref, validity_ref = calibration_report_blob_refs(payload)
        pixels = math.prod(artifact.frame_contract.frame_schema.data_shape)
        expected = (pixels * 8, pixels)
        if (average_ref.size, validity_ref.size) != expected:
            raise ValueError(
                "calibration diagnostic blob sizes differ from the FrameContract"
            )
        return payload, average_ref, validity_ref

    def _load_report_ref(
        self,
        artifact: CalibrationArtifact,
        reference: ContentRef,
    ) -> CalibrationReport:
        from .analysis import CalibrationComputation

        authority = self._content_authority()
        payload, average_ref, validity_ref = self._report_storage_refs(
            artifact,
            reference,
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

    def load(
        self,
        reference: CalibrationArtifactRef,
        *,
        memory_limit_bytes: int | None = None,
    ) -> CalibrationArtifact:
        """Load a FINAL artifact under an explicit or repository memory limit."""

        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            self._require_final_commit(reference)
            artifact_ref, report_ref, summary = self._storage_refs(
                reference,
                memory_limit_bytes=memory_limit_bytes,
            )
            self._require_memory_budget(
                summary.inspection_retained_upper_bound_bytes
                + summary.artifact_decode_peak_upper_bound_bytes,
                "calibration artifact load",
                memory_limit_bytes=memory_limit_bytes,
            )
            return self._materialize_artifact(artifact_ref, report_ref, summary)

    def load_report(
        self,
        reference: CalibrationArtifactRef,
        *,
        memory_limit_bytes: int | None = None,
    ) -> CalibrationReport:
        """Load FINAL diagnostics under one artifact-plus-report memory limit."""

        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            self._require_final_commit(reference)
            artifact_ref, report_ref, summary = self._storage_refs(
                reference,
                memory_limit_bytes=memory_limit_bytes,
            )
            self._require_memory_budget(
                summary.inspection_retained_upper_bound_bytes
                + _artifact_report_load_peak(summary),
                "calibration report load",
                memory_limit_bytes=memory_limit_bytes,
            )
            artifact = self._materialize_artifact(
                artifact_ref,
                report_ref,
                summary,
            )
            return self._load_report_ref(artifact, report_ref)

    def has(self, reference: CalibrationArtifactRef) -> bool:
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
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
        *,
        checkpoint: Callable[[], None] | None = None,
        memory_limit_bytes: int,
    ) -> _ResolvedCalibrationSource:
        from zlc_neutral_atom.artifacts.capture import AdmittedCapture

        if type(source) is not AdmittedCapture:
            raise TypeError("source must be an exact AdmittedCapture")
        return _validate_calibration_artifact_source_compatibility(
            artifact,
            source.artifact,
            checkpoint=checkpoint,
            physical_memory_limit_bytes=memory_limit_bytes,
        )

    def admit(
        self,
        reference: CalibrationArtifactRef,
        capture_repository: "CaptureRepository",
        *,
        checkpoint: Callable[[], None] | None = None,
        memory_limit_bytes: int | None = None,
    ) -> ResolvedCalibration:
        """Admit a FINAL target and its source under one aggregate memory limit."""

        from zlc_neutral_atom.artifacts.capture import CaptureRepository

        if type(capture_repository) is not CaptureRepository:
            raise TypeError("capture_repository must be CaptureRepository")
        with self._root_lease.borrow() as admission_borrow:
            admission_borrow.require_active()
            with capture_repository._root_lease.borrow() as source_borrow:
                source_borrow.require_active()
                self._require_final_commit(reference)
                artifact_ref, report_ref, summary = self._storage_refs(
                    reference,
                    memory_limit_bytes=memory_limit_bytes,
                )
                summary_headroom = summary.inspection_retained_upper_bound_bytes
                limit = self._require_memory_budget(
                    summary_headroom
                    + summary.artifact_decode_peak_upper_bound_bytes,
                    "calibration admission",
                    memory_limit_bytes=memory_limit_bytes,
                )
                source_capture_ref = summary.source_capture_ref
                artifact_retained = summary.artifact_retained_upper_bound_bytes
                artifact_decode_peak = summary.artifact_decode_peak_upper_bound_bytes
                source_info = capture_repository.inspect_final(
                    source_capture_ref,
                    memory_limit_bytes=limit - summary_headroom,
                )
                if source_info.readout_binding != summary.readout_binding:
                    raise ValueError(
                        "calibration and training capture readout bindings differ"
                    )
                pulse_summary = source_info.pulse_runtime_summary
                if pulse_summary is None:
                    raise ValueError(
                        "authoritative calibration requires persisted pulse lineage"
                    )
                physical_peak = estimate_readout_physical_context_peak_from_summary(
                    pulse_summary
                )
                source_decode_peak = (
                    source_info.admission_decode_peak_upper_bound_bytes
                )
                source_retained = source_info.admission_retained_upper_bound_bytes
                source_read_scratch = source_info.max_read_scratch_bytes
                required_peak = max(
                    summary_headroom + artifact_decode_peak,
                    artifact_retained + source_decode_peak,
                    artifact_retained
                    + source_retained
                    + source_read_scratch
                    + physical_peak,
                )
                self._require_memory_budget(
                    required_peak,
                    "calibration admission",
                    memory_limit_bytes=limit,
                )
                del source_info, pulse_summary
                artifact = self._materialize_artifact(
                    artifact_ref,
                    report_ref,
                    summary,
                )
                del summary
                source = capture_repository.admit(source_capture_ref)
                self._validate_source_admission(
                    artifact,
                    source,
                    checkpoint=checkpoint,
                    memory_limit_bytes=limit,
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
        result_staging_peak_upper_bound_bytes: int,
        context_codec_workspace_upper_bound_bytes: int,
        memory_admission_limit_bytes: int,
    ) -> tuple[CalibrationArtifactRef, bytes]:
        authority = self._content_authority()
        report = result.report
        result_staging_peak = positive_integer(
            result_staging_peak_upper_bound_bytes,
            "result_staging_peak_upper_bound_bytes",
        )
        admission_limit = positive_integer(
            memory_admission_limit_bytes,
            "memory_admission_limit_bytes",
        )
        admitted_context_workspace = positive_integer(
            context_codec_workspace_upper_bound_bytes,
            "context_codec_workspace_upper_bound_bytes",
        )
        actual_context_workspace = (
            estimate_calibration_context_roundtrip_workspace(
                result.artifact.readout_physical_context
            )
        )
        if actual_context_workspace > admitted_context_workspace:
            raise MemoryError(
                "calibration physical-context codec workspace exceeds its "
                "preflight admission"
            )
        # One whole-run proof covers every staging phase, including the first
        # encode, typed round-trip, and report/diagnostic overlap.  Post-encode
        # wire sizes enforce repository caps but do not start a second,
        # non-equivalent memory admission model.
        self._require_memory_budget(
            result_staging_peak,
            "calibration result staging",
            memory_limit_bytes=admission_limit,
        )
        artifact_payload = encode_calibration_artifact(result.artifact)
        if len(artifact_payload) > _MAX_ARTIFACT_BYTES:
            raise MemoryError(
                f"calibration artifact requires {len(artifact_payload)} bytes; "
                f"limit {_MAX_ARTIFACT_BYTES}"
            )
        decode_calibration_artifact(artifact_payload)
        # The diagnostic image/mask payloads are raw CAS bytes rather than
        # canonical base64.  They overlap the still-live result and artifact
        # payload, so admit that actual overlap before making either copy.
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
            artifact_blob,
            report_blob,
            _runtime_summary(
                result.artifact,
                artifact_blob_bytes=len(artifact_payload),
                report_metadata_blob_bytes=len(report_payload),
            ),
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
    ) -> FinalCommit[CalibrationArtifactRef]:
        """Prepare publication from the exact admission retained by analysis."""

        from .analysis import CalibrationAnalysisResult

        if not isinstance(context, PostSafetyContext):
            raise TypeError("calibration commit requires PostSafetyContext")
        if type(result) is not CalibrationAnalysisResult:
            raise TypeError("result must be CalibrationAnalysisResult")
        (
            source,
            resolved,
            result_staging_peak,
            context_codec_workspace,
            memory_admission_limit,
        ) = result._source_for_commit()
        source._require_authority()
        if not resolved.join.matches_contexts(result.report.group_contexts):
            raise ValueError(
                "calibration report group contexts differ from the admitted source"
            )
        run_id, safety_bundle_id = context.authorize_commit_preparation()
        # Staging writes CAS blobs, so repository lifetime begins before the
        # first write and overlaps prepare() minting the commit-lifetime hold.
        with self._root_lease.borrow() as staging_borrow:
            staging_borrow.require_active()
            reference, payload = self._stage_result(
                result,
                result_staging_peak_upper_bound_bytes=result_staging_peak,
                context_codec_workspace_upper_bound_bytes=(
                    context_codec_workspace
                ),
                memory_admission_limit_bytes=memory_admission_limit,
            )
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
        artifact_ref, report_ref, summary = self._storage_refs(
            reference,
            manifest_payload=payload,
        )
        self._require_memory_budget(
            _artifact_report_load_peak(summary),
            "calibration recovery validation",
            memory_limit_bytes=None,
        )
        artifact = self._materialize_artifact(
            artifact_ref,
            report_ref,
            summary,
        )
        _report_payload, average_ref, validity_ref = self._report_storage_refs(
            artifact,
            report_ref,
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
    from .analysis import (
        CalibrationAnalysisResult,
        _analyze_calibration_resolved,
        estimate_calibration_analysis_peak_bytes,
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
    memory_limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
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
            inspected = capture_repository.inspect_final(
                source_capture_ref,
                memory_limit_bytes=memory_limit,
            )
            if inspected.readout_binding != expected_readout_binding:
                raise ValueError(
                    "source capture readout binding differs from the frozen request"
                )
            pulse_summary = inspected.pulse_runtime_summary
            if pulse_summary is None:
                raise ValueError(
                    "authoritative calibration requires persisted pulse lineage"
                )
            physical_peak = estimate_readout_physical_context_peak_from_summary(
                pulse_summary
            )
            context_retained = (
                estimate_readout_physical_context_retained_from_summary(
                    pulse_summary
                )
            )
            analysis_peak = estimate_calibration_analysis_peak_bytes(
                inspected.dataset_schema,
                request,
                source_read_scratch_bytes=inspected.max_read_scratch_bytes,
            )
            source_retained = inspected.admission_retained_upper_bound_bytes
            source_decode_peak = inspected.admission_decode_peak_upper_bound_bytes
            source_read_scratch = inspected.max_read_scratch_bytes
            workload_peak = physical_peak + source_read_scratch
            estimated_peak = max(
                source_decode_peak,
                source_retained + workload_peak,
            )
            if estimated_peak > memory_limit:
                raise MemoryError(
                    f"calibration analysis requires {estimated_peak} bytes; "
                    f"limit {memory_limit}"
                )
            del inspected, pulse_summary
            source = capture_repository.admit(source_capture_ref)
            if source.artifact.camera_provenance.binding != expected_readout_binding:
                raise ValueError(
                    "source capture readout binding differs from the frozen request"
                )
            resolved = _resolve_calibration_source(
                source.artifact,
                request.layout,
                checkpoint=context.checkpoint,
                physical_memory_limit_bytes=memory_limit,
            )
            actual_context_retained = (
                readout_physical_context_retained_upper_bound_bytes(
                    resolved.readout_physical_context
                )
            )
            if actual_context_retained > context_retained:
                raise MemoryError(
                    "resolved readout context exceeds its preflight retained bound"
                )
            actual_context_codec_workspace = (
                estimate_calibration_context_roundtrip_workspace(
                    resolved.readout_physical_context
                )
            )
            result_memory_peak = (
                source_retained
                + actual_context_retained
                + analysis_peak
                + actual_context_codec_workspace
            )
            if result_memory_peak > memory_limit:
                raise MemoryError(
                    "calibration retained result exceeds its admitted memory limit"
                )
            context.checkpoint()
            return (
                source,
                resolved,
                result_memory_peak,
                actual_context_codec_workspace,
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
            result_memory_peak,
            context_codec_workspace,
            borrows,
        ) = prepared
        context.checkpoint()
        result = _analyze_calibration_resolved(
            source,
            request,
            resolved,
            memory_admission_peak_bytes=result_memory_peak,
            context_codec_workspace_upper_bound_bytes=context_codec_workspace,
            memory_admission_limit_bytes=memory_limit,
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
                _memory_peak,
                _context_workspace,
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
            return context.commit_final(operation)
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
        # The per-run estimator cannot make two concurrent 500 MiB analyses
        # safe in aggregate.  One flat non-device claim serializes this CPU and
        # memory-heavy owner without inventing a scheduler or workflow engine.
        resource_claims=(READOUT_ANALYSIS_CLAIM,),
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
    "CalibrationRuntimeSummary",
    "compile_calibration_artifact_plan",
]
