"""Canonical FINAL dataset authority for autonomous pulse scans.

The repository owns one current format.  Direct-camera and processed exact
sources are normalized by scan application adapters before this boundary; the
persisted artifact therefore contains the canonical ``(R, P, *data_shape)``
output itself and never delegates materialization back to a source repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import numpy as np

from zlc_data import (
    READOUT_EVENT,
    BlockId,
    DataBlock,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    dataset_revision_ref_from_tree,
    dataset_revision_ref_to_tree,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
    validity_from_tree,
    validity_to_tree,
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
from zlc_neutral_atom.runtime.dataset import (
    DatasetSealProvenance,
    dataset_seal_provenance_from_tree,
    dataset_seal_provenance_to_tree,
)
from zlc_neutral_atom.runtime.run import PostSafetyContext
from zlc_neutral_atom.timing.lineage import (
    PulseCaptureEvidence,
    pulse_capture_evidence_from_tree,
    pulse_capture_evidence_to_tree,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    CompiledPulseRuntimeSummary,
    MAX_COMPILED_PULSE_ARTIFACT_BYTES,
    PulseDocument,
    PulseExecutionForm,
    compiled_pulse_runtime_summary,
    compiled_pulse_runtime_summary_from_tree,
    compiled_pulse_runtime_summary_to_tree,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
    expand_autonomous_scan_repeats,
    pulse_document_from_tree,
    pulse_document_to_tree,
)
from zlc_storage import (
    CanonicalArrayEvent,
    CanonicalDecodeLimits,
    CanonicalEncodingError,
    ContentAddressedStore,
    ContentRef,
    ContentSizeLimitError,
    ContentStoreAuthority,
    RepositoryRootLease,
    canonical_digest,
    canonical_text,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping,
    positive_integer,
    sha256_digest,
    sha256_text,
)

from .contracts import (
    ScanOutputContract,
    ScanPointTable,
    bind_scan_output_contract,
    scan_output_contract_from_tree,
    scan_output_contract_to_tree,
)
from .reference import SCAN_ARTIFACT_NAMESPACE, ScanArtifactRef


SCAN_ARTIFACT_SCHEMA = "zlc_neutral_atom.scan-storage"
SCAN_MANIFEST_SCHEMA = "zlc_neutral_atom.scan-manifest"
_SCAN_ARTIFACT_KIND = "scan"
_MANIFEST_FIELDS = frozenset({"schema", "repository_id", "metadata_blob"})
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "pulse_document_blob",
        "compiled_pulse_blob",
        "compiled_pulse_runtime_summary",
        "pulse_evidence",
        "source_dataset_ref",
        "source_dataset_schema",
        "output_contract",
        "output_dataset_ref",
        "dataset_provenance",
        "output_values_blob",
        "output_validity_blob",
        "safety_bundle_id",
    }
)
_METADATA_LIMITS = CanonicalDecodeLimits(
    max_depth=64,
    max_nodes=200_000,
    max_container_entries=200_000,
    max_arrays=0,
    max_total_array_bytes=0,
)
_FIXED_MATERIALIZATION_BYTES = 4 << 20
_CANONICAL_DECODE_MULTIPLIER = 8
_DOCUMENT_DECODE_MULTIPLIER = 16
_STATIC_LINEAGE_FIXED_BYTES = 1 << 20


class ScanResourceExceeded(RuntimeError):
    """One canonical scan exceeds an explicit storage/admission budget."""


@dataclass(frozen=True, slots=True)
class ScanRepositoryResourcePolicy:
    max_manifest_bytes: int = 1 << 20
    max_metadata_blob_bytes: int = 16 << 20
    max_pulse_document_blob_bytes: int = 16 << 20
    max_compiled_pulse_blob_bytes: int = MAX_COMPILED_PULSE_ARTIFACT_BYTES
    max_output_values_blob_bytes: int = 8 << 30
    max_output_validity_blob_bytes: int = 2 << 30

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_compiled_pulse_blob_bytes > MAX_COMPILED_PULSE_ARTIFACT_BYTES:
            raise ValueError(
                "scan compiled-pulse budget cannot exceed the pulse owner limit"
            )


DEFAULT_SCAN_REPOSITORY_RESOURCE_POLICY = ScanRepositoryResourcePolicy()


@dataclass(frozen=True, slots=True)
class _StoredScan:
    pulse_document_blob: ContentRef
    compiled_pulse_blob: ContentRef
    compiled_pulse_runtime_summary: CompiledPulseRuntimeSummary
    pulse_evidence: PulseCaptureEvidence
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance
    values_blob: ContentRef
    validity_blob: ContentRef
    safety_bundle_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_document_blob, ContentRef) or not isinstance(
            self.compiled_pulse_blob, ContentRef
        ):
            raise TypeError("scan lineage blobs must be ContentRef")
        if not isinstance(self.pulse_evidence, PulseCaptureEvidence):
            raise TypeError("pulse_evidence must be PulseCaptureEvidence")
        if not isinstance(
            self.compiled_pulse_runtime_summary,
            CompiledPulseRuntimeSummary,
        ):
            raise TypeError(
                "compiled_pulse_runtime_summary must be CompiledPulseRuntimeSummary"
            )
        if not isinstance(self.source_dataset_ref, DatasetRevisionRef) or not isinstance(
            self.output_dataset_ref, DatasetRevisionRef
        ):
            raise TypeError("scan dataset refs must be DatasetRevisionRef")
        if not isinstance(self.source_dataset_schema, DatasetSchema):
            raise TypeError("source_dataset_schema must be DatasetSchema")
        if not isinstance(self.output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(self.provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if not isinstance(self.values_blob, ContentRef) or not isinstance(
            self.validity_blob, ContentRef
        ):
            raise TypeError("scan data blobs must be ContentRef")
        canonical_text(self.safety_bundle_id, "safety_bundle_id")


@dataclass(frozen=True, slots=True)
class _StoredScanIndex:
    """Small current-format index decoded without PulseDocument or pulse IR."""

    pulse_document_blob: ContentRef
    compiled_pulse_blob: ContentRef
    compiled_pulse_runtime_summary: CompiledPulseRuntimeSummary
    pulse_evidence_tree: object
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance
    values_blob: ContentRef
    validity_blob: ContentRef
    safety_bundle_id: str


@dataclass(frozen=True, slots=True)
class _StaticScanLineageAdmission:
    """Pure pre-Run resource proof for immutable pulse/document lineage."""

    pulse_document_blob: ContentRef
    compiled_pulse_blob: ContentRef
    compiled_pulse_runtime_summary: CompiledPulseRuntimeSummary
    retained_upper_bound_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_document_blob, ContentRef) or not isinstance(
            self.compiled_pulse_blob,
            ContentRef,
        ):
            raise TypeError("staged scan lineage blobs must be ContentRef")
        if not isinstance(
            self.compiled_pulse_runtime_summary,
            CompiledPulseRuntimeSummary,
        ):
            raise TypeError("staged pulse summary must be CompiledPulseRuntimeSummary")
        positive_integer(
            self.retained_upper_bound_bytes,
            "retained_upper_bound_bytes",
        )


@dataclass(frozen=True, slots=True)
class _StagedScanLineage:
    """Pre-FIRE CAS references verified against one static admission."""

    pulse_document_blob: ContentRef
    compiled_pulse_blob: ContentRef
    compiled_pulse_runtime_summary: CompiledPulseRuntimeSummary


@dataclass(frozen=True, slots=True)
class ScanArtifact:
    """Admitted metadata for one canonical FINAL scan dataset."""

    ref: ScanArtifactRef
    pulse_document: PulseDocument
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance
    pulse_evidence: PulseCaptureEvidence
    safety_bundle_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ScanArtifactRef):
            raise TypeError("ref must be ScanArtifactRef")
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        if not isinstance(self.source_dataset_ref, DatasetRevisionRef) or not isinstance(
            self.output_dataset_ref, DatasetRevisionRef
        ):
            raise TypeError("scan dataset refs must be DatasetRevisionRef")
        if not isinstance(self.source_dataset_schema, DatasetSchema):
            raise TypeError("source_dataset_schema must be DatasetSchema")
        if not isinstance(self.output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(self.provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if not isinstance(self.pulse_evidence, PulseCaptureEvidence):
            raise TypeError("pulse_evidence must be PulseCaptureEvidence")
        canonical_text(self.safety_bundle_id, "safety_bundle_id")

    @property
    def output_schema(self) -> DatasetSchema:
        return self.output_contract.output_dataset_schema


@dataclass(frozen=True, slots=True)
class ScanArtifactInspection:
    """FINAL dataset facts obtained without decoding pulse IR or document."""

    ref: ScanArtifactRef
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance
    pulse_runtime_summary: CompiledPulseRuntimeSummary
    safety_bundle_id: str
    materialization_peak_upper_bound_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ScanArtifactRef):
            raise TypeError("ref must be ScanArtifactRef")
        if not isinstance(self.source_dataset_ref, DatasetRevisionRef) or not isinstance(
            self.output_dataset_ref,
            DatasetRevisionRef,
        ):
            raise TypeError("scan inspection dataset refs must be DatasetRevisionRef")
        if not isinstance(self.source_dataset_schema, DatasetSchema):
            raise TypeError("source_dataset_schema must be DatasetSchema")
        if not isinstance(self.output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(self.provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if not isinstance(self.pulse_runtime_summary, CompiledPulseRuntimeSummary):
            raise TypeError("pulse_runtime_summary must be CompiledPulseRuntimeSummary")
        canonical_text(self.safety_bundle_id, "safety_bundle_id")
        positive_integer(
            self.materialization_peak_upper_bound_bytes,
            "materialization_peak_upper_bound_bytes",
        )

    @property
    def output_schema(self) -> DatasetSchema:
        return self.output_contract.output_dataset_schema


@dataclass(frozen=True, eq=False, slots=True)
class MaterializedScanData:
    artifact_ref: ScanArtifactRef
    source_dataset_ref: DatasetRevisionRef
    snapshot: OwnedSnapshot
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, ScanArtifactRef):
            raise TypeError("artifact_ref must be ScanArtifactRef")
        if not isinstance(self.source_dataset_ref, DatasetRevisionRef):
            raise TypeError("source_dataset_ref must be DatasetRevisionRef")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")

    @property
    def schema(self) -> DatasetSchema:
        return self.snapshot.block.schema

    @property
    def values(self) -> np.ndarray:
        return self.snapshot.block.values

    @property
    def validity(self):
        return self.snapshot.block.validity


_SCAN_APPLICATION_TOKEN = object()


class _PreparedScanDataset:
    """Process-local output minted only from one opaque joint pipeline result."""

    __slots__ = (
        "run_id",
        "pulse_document",
        "source_snapshot",
        "output_contract",
        "output_snapshot",
        "provenance",
        "pulse_evidence",
        "staged_lineage",
        "memory_limit_bytes",
    )

    def __init__(
        self,
        token: object,
        *,
        run_id: str,
        pulse_document: PulseDocument,
        source_snapshot: OwnedSnapshot,
        output_contract: ScanOutputContract,
        output_snapshot: OwnedSnapshot,
        provenance: DatasetSealProvenance,
        pulse_evidence: PulseCaptureEvidence,
        staged_lineage: _StagedScanLineage,
        memory_limit_bytes: int,
    ) -> None:
        if token is not _SCAN_APPLICATION_TOKEN:
            raise PermissionError("prepared scan datasets are minted by scan application")
        canonical_text(run_id, "run_id")
        if not isinstance(pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        if not isinstance(source_snapshot, OwnedSnapshot):
            raise TypeError("source_snapshot must be OwnedSnapshot")
        if not isinstance(output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(output_snapshot, OwnedSnapshot):
            raise TypeError("output_snapshot must be OwnedSnapshot")
        if not isinstance(provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if not isinstance(pulse_evidence, PulseCaptureEvidence):
            raise TypeError("pulse_evidence must be PulseCaptureEvidence")
        if not isinstance(staged_lineage, _StagedScanLineage):
            raise TypeError("staged_lineage must be pre-FIRE scan lineage")
        if staged_lineage.pulse_document_blob.digest != pulse_document.fingerprint:
            raise ValueError("staged PulseDocument identity changed")
        if (
            staged_lineage.compiled_pulse_blob.digest
            != pulse_evidence.compiled_artifact.fingerprint
        ):
            raise ValueError("staged compiled pulse identity changed")
        staged_lineage.compiled_pulse_runtime_summary.require_encoded_size(
            staged_lineage.compiled_pulse_blob.size
        )
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        _require_scan_facts(
            run_id=run_id,
            document=pulse_document,
            source_ref=source_snapshot.ref,
            source_schema=source_snapshot.block.schema,
            output_contract=output_contract,
            output_ref=output_snapshot.ref,
            provenance=provenance,
            pulse_evidence=pulse_evidence,
        )
        if output_snapshot.block.schema != output_contract.output_dataset_schema:
            raise ValueError("prepared output snapshot has another ScanOutputContract")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "pulse_document", pulse_document)
        object.__setattr__(self, "source_snapshot", source_snapshot)
        object.__setattr__(self, "output_contract", output_contract)
        object.__setattr__(self, "output_snapshot", output_snapshot)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "pulse_evidence", pulse_evidence)
        object.__setattr__(self, "staged_lineage", staged_lineage)
        object.__setattr__(self, "memory_limit_bytes", limit)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("prepared scan dataset is immutable")

    def __reduce__(self):
        raise TypeError("prepared scan dataset is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("prepared scan dataset is process-local")

    @property
    def source_dataset_ref(self) -> DatasetRevisionRef:
        return self.source_snapshot.ref

    @property
    def source_dataset_schema(self) -> DatasetSchema:
        return self.source_snapshot.block.schema


def _scan_output_dataset_ref(
    document: PulseDocument,
    source_ref: DatasetRevisionRef,
    output_contract: ScanOutputContract,
) -> DatasetRevisionRef:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("source_ref must be DatasetRevisionRef")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    return _scan_output_dataset_ref_for_document_digest(
        document.fingerprint,
        source_ref,
        output_contract,
    )


def _scan_output_dataset_ref_for_document_digest(
    document_digest: str,
    source_ref: DatasetRevisionRef,
    output_contract: ScanOutputContract,
) -> DatasetRevisionRef:
    if not isinstance(document_digest, str):
        raise TypeError("document_digest must be str")
    sha256_text(document_digest, "document_digest")
    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("source_ref must be DatasetRevisionRef")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.scan-output",
            "pulse_document": document_digest,
            "source_dataset_ref": dataset_revision_ref_to_tree(source_ref),
            "output_contract": output_contract.fingerprint,
        }
    )
    return DatasetRevisionRef(
        BlockId(f"scan-output-{identity}"),
        source_ref.stream_generation,
        output_contract.output_schema_fingerprint,
        source_ref.revision,
    )


def _require_scan_facts(
    *,
    run_id: str,
    document: PulseDocument,
    source_ref: DatasetRevisionRef,
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    output_ref: DatasetRevisionRef,
    provenance: DatasetSealProvenance,
    pulse_evidence: PulseCaptureEvidence,
) -> None:
    canonical_text(run_id, "run_id")
    _require_scan_dataset_facts(
        document_digest=document.fingerprint,
        source_ref=source_ref,
        source_schema=source_schema,
        output_contract=output_contract,
        output_ref=output_ref,
        provenance=provenance,
    )
    point_table = ScanPointTable.from_pulse_document(document)
    compiled = pulse_evidence.compiled_artifact
    if compiled.execution_form is not PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        raise ValueError("scan artifact requires AUTONOMOUS_SCAN_ONCE evidence")
    expanded = expand_autonomous_scan_repeats(document)
    if compiled.source_document_digest != expanded.fingerprint:
        raise ValueError("compiled scan differs from repeat-expanded logical document")
    repeat_count = 1 if document.repeat is None else document.repeat.count
    schedule = pulse_evidence.trigger_schedule
    if (
        schedule.point_count != repeat_count * point_table.point_layout.storage_size
        or schedule.loop_count != 1
        or not schedule.full_point_loop
    ):
        raise ValueError("pulse evidence is not one complete repeat-major scan")
    if source_schema.repeat_axis.size != repeat_count:
        raise ValueError("source repeat axis differs from logical scan repeats")
    event_axes = tuple(
        axis for axis in source_schema.point_axes if axis.role == READOUT_EVENT
    )
    scan_axes = tuple(
        axis for axis in source_schema.point_axes if axis.role != READOUT_EVENT
    )
    if len(event_axes) != 1 or event_axes[0].size != 1:
        raise ValueError("current exact scan source requires one singleton READOUT_EVENT")
    if scan_axes != point_table.point_axes:
        raise ValueError("source scan axes differ from the logical ScanPointTable")
    if pulse_evidence.join_contract.scan_point_layout != point_table.point_layout:
        raise ValueError("pulse join layout differs from the logical ScanPointTable")
    expected_join = pulse_evidence.expected_cell_schedule_digest(source_schema)
    if provenance.join_plan_digest != expected_join:
        raise ValueError("source dataset schedule differs from pulse evidence")
    event_count = provenance.end_sequence - provenance.start_sequence
    if event_count != pulse_evidence.expected_trigger_count:
        raise ValueError("source provenance count differs from pulse triggers")
    if provenance.trace_binding.run_id != run_id:
        raise ValueError("source provenance belongs to another Run")
    resolved = bind_scan_output_contract(
        source_schema,
        point_table,
        output_contract.committed_transform,
    )
    if resolved != output_contract:
        raise ValueError("ScanOutputContract differs from source schema and point table")


def _require_scan_dataset_facts(
    *,
    document_digest: str,
    source_ref: DatasetRevisionRef,
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    output_ref: DatasetRevisionRef,
    provenance: DatasetSealProvenance,
) -> None:
    sha256_text(document_digest, "document_digest")
    if source_ref.stream_generation != provenance.generation:
        raise ValueError("source dataset generation differs from its provenance")
    if source_ref.schema_fingerprint != source_schema.fingerprint:
        raise ValueError("source dataset ref differs from its source schema")
    if output_ref.schema_fingerprint != output_contract.output_schema_fingerprint:
        raise ValueError("scan output ref differs from its output schema")
    if output_ref != _scan_output_dataset_ref_for_document_digest(
        document_digest,
        source_ref,
        output_contract,
    ):
        raise ValueError("scan output dataset identity differs from frozen inputs")


def _reject_arrays(events) -> None:
    if any(isinstance(event, CanonicalArrayEvent) for event in events):
        raise ScanResourceExceeded("scan metadata cannot embed ndarray payloads")


def _metadata_tree(
    value: _StoredScan | _StoredScanIndex,
) -> dict[str, object]:
    evidence_tree = (
        pulse_capture_evidence_to_tree(value.pulse_evidence)
        if isinstance(value, _StoredScan)
        else value.pulse_evidence_tree
    )
    return {
        "schema": SCAN_ARTIFACT_SCHEMA,
        "pulse_document_blob": content_ref_to_tree(value.pulse_document_blob),
        "compiled_pulse_blob": content_ref_to_tree(value.compiled_pulse_blob),
        "compiled_pulse_runtime_summary": (
            compiled_pulse_runtime_summary_to_tree(
                value.compiled_pulse_runtime_summary
            )
        ),
        "pulse_evidence": evidence_tree,
        "source_dataset_ref": dataset_revision_ref_to_tree(
            value.source_dataset_ref
        ),
        "source_dataset_schema": dataset_schema_to_tree(
            value.source_dataset_schema
        ),
        "output_contract": scan_output_contract_to_tree(value.output_contract),
        "output_dataset_ref": dataset_revision_ref_to_tree(
            value.output_dataset_ref
        ),
        "dataset_provenance": dataset_seal_provenance_to_tree(value.provenance),
        "output_values_blob": content_ref_to_tree(value.values_blob),
        "output_validity_blob": content_ref_to_tree(value.validity_blob),
        "safety_bundle_id": value.safety_bundle_id,
    }


def _encode_metadata(value: _StoredScan | _StoredScanIndex) -> bytes:
    try:
        return encode(_metadata_tree(value), limits=_METADATA_LIMITS)
    except CanonicalEncodingError as exc:
        raise ScanResourceExceeded("scan metadata exceeds canonical limits") from exc


def _decode_metadata_index(payload: bytes) -> _StoredScanIndex:
    data = exact_mapping(
        decode(
            payload,
            admit_structure=_reject_arrays,
            limits=_METADATA_LIMITS,
        ),
        _ARTIFACT_FIELDS,
        SCAN_ARTIFACT_SCHEMA,
    )
    value = _StoredScanIndex(
        content_ref_from_tree(data["pulse_document_blob"]),
        content_ref_from_tree(data["compiled_pulse_blob"]),
        compiled_pulse_runtime_summary_from_tree(
            data["compiled_pulse_runtime_summary"]
        ),
        data["pulse_evidence"],
        dataset_revision_ref_from_tree(data["source_dataset_ref"]),
        dataset_schema_from_tree(data["source_dataset_schema"]),
        scan_output_contract_from_tree(data["output_contract"]),
        dataset_revision_ref_from_tree(data["output_dataset_ref"]),
        dataset_seal_provenance_from_tree(data["dataset_provenance"]),
        content_ref_from_tree(data["output_values_blob"]),
        content_ref_from_tree(data["output_validity_blob"]),
        canonical_text(data["safety_bundle_id"], "safety_bundle_id"),
    )
    if _encode_metadata(value) != payload:
        raise ValueError("scan metadata is typed but non-canonical")
    return value


def _stored_scan_from_index(
    index: _StoredScanIndex,
    compiled_pulse: CompiledPulseArtifact,
) -> _StoredScan:
    return _StoredScan(
        index.pulse_document_blob,
        index.compiled_pulse_blob,
        index.compiled_pulse_runtime_summary,
        pulse_capture_evidence_from_tree(
            index.pulse_evidence_tree,
            compiled_pulse,
        ),
        index.source_dataset_ref,
        index.source_dataset_schema,
        index.output_contract,
        index.output_dataset_ref,
        index.provenance,
        index.values_blob,
        index.validity_blob,
        index.safety_bundle_id,
    )


def _manifest_payload(repository_id: str, metadata_blob: ContentRef) -> bytes:
    return encode(
        {
            "schema": SCAN_MANIFEST_SCHEMA,
            "repository_id": canonical_text(repository_id, "repository_id"),
            "metadata_blob": content_ref_to_tree(metadata_blob),
        }
    )


def _decode_manifest(payload: bytes) -> tuple[str, ContentRef]:
    data = exact_mapping(
        decode(payload, admit_structure=_reject_arrays, limits=_METADATA_LIMITS),
        _MANIFEST_FIELDS,
        SCAN_MANIFEST_SCHEMA,
    )
    value = (
        canonical_text(data["repository_id"], "repository_id"),
        content_ref_from_tree(data["metadata_blob"]),
    )
    if _manifest_payload(*value) != payload:
        raise ValueError("scan manifest is typed but non-canonical")
    return value


def _encode_document(document: PulseDocument) -> bytes:
    return encode(pulse_document_to_tree(document), limits=_METADATA_LIMITS)


def _decode_document(payload: bytes) -> PulseDocument:
    document = pulse_document_from_tree(
        decode(payload, admit_structure=_reject_arrays, limits=_METADATA_LIMITS)
    )
    if _encode_document(document) != payload:
        raise ValueError("PulseDocument blob is typed but non-canonical")
    return document


def _encode_validity(validity: object) -> bytes:
    return encode(validity_to_tree(validity))


def _decode_validity(payload: bytes, *, max_array_bytes: int):
    limits = CanonicalDecodeLimits(
        max_depth=16,
        max_nodes=256,
        max_container_entries=128,
        max_arrays=1,
        max_total_array_bytes=max_array_bytes,
    )
    validity = validity_from_tree(decode(payload, limits=limits))
    if _encode_validity(validity) != payload:
        raise ValueError("scan validity blob is typed but non-canonical")
    return validity


def _values_payload(snapshot: OwnedSnapshot) -> memoryview:
    values = snapshot.block.values
    if not values.flags.c_contiguous:
        raise ValueError("canonical scan output values must be C-contiguous")
    return memoryview(values).cast("B")


def _snapshot_retained_bytes(snapshot: OwnedSnapshot) -> int:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    validity = snapshot.block.validity
    mask = getattr(validity, "mask", None)
    validity_bytes = 0 if mask is None else int(mask.nbytes)
    return int(snapshot.block.values.nbytes + validity_bytes)


def _target(repository_id: str, reference: ScanArtifactRef) -> CommitTarget:
    return CommitTarget(
        repository_id,
        _SCAN_ARTIFACT_KIND,
        SCAN_MANIFEST_SCHEMA,
        reference.target_ref,
        reference.manifest_digest,
    )


def _commit_id(run_id: str, manifest_digest: str) -> str:
    return f"scan-final-{canonical_text(run_id, 'run_id')}-{manifest_digest}"


class ScanRepository:
    """Current-only CAS authority for canonical FINAL scan datasets."""

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-scan",
        resource_policy: ScanRepositoryResourcePolicy = (
            DEFAULT_SCAN_REPOSITORY_RESOURCE_POLICY
        ),
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = canonical_text(repository_id, "repository_id")
        if not isinstance(resource_policy, ScanRepositoryResourcePolicy):
            raise TypeError("resource_policy must be ScanRepositoryResourcePolicy")
        self.resource_policy = resource_policy
        self._lock = threading.RLock()
        self._closed = False
        self._root_lease = RepositoryRootLease(self.root)
        journal = None
        try:
            self._store = ContentAddressedStore(self.root / "content")
            self._store_authority = self._store.authority()
            journal = PersistentCommitJournal(
                self.root / "scan-commit.journal",
                self.repository_id,
            )
            self._coordinator: RepositoryCommitCoordinator[
                ScanArtifactRef
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
            raise RuntimeError("scan repository is closed")
        self._root_lease.require_active()

    def _require_active(self) -> None:
        """Application admission probe without exposing repository internals."""

        with self._lock:
            self._require_open()

    def _content_authority(self) -> ContentStoreAuthority:
        with self._lock:
            self._require_open()
            return self._store_authority

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._coordinator.close()
            self._closed = True

    def __enter__(self) -> "ScanRepository":
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

    def _validate_reference(self, reference: ScanArtifactRef) -> None:
        if not isinstance(reference, ScanArtifactRef):
            raise TypeError("reference must be ScanArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("ScanArtifactRef belongs to another repository")

    def _admit_static_lineage(
        self,
        document: PulseDocument,
        compiled_pulse: CompiledPulseArtifact,
        *,
        memory_limit_bytes: int,
    ) -> _StaticScanLineageAdmission:
        """Prove immutable lineage policy/peak without writing repository state."""

        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(compiled_pulse, CompiledPulseArtifact):
            raise TypeError("compiled_pulse must be CompiledPulseArtifact")
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        policy = self.resource_policy
        document_payload = _encode_document(document)
        compiled_payload = encode_compiled_pulse_artifact(compiled_pulse)
        if len(document_payload) > policy.max_pulse_document_blob_bytes:
            raise ScanResourceExceeded("scan PulseDocument exceeds repository policy")
        if len(compiled_payload) > policy.max_compiled_pulse_blob_bytes:
            raise ScanResourceExceeded("scan compiled pulse exceeds repository policy")
        summary = compiled_pulse_runtime_summary(
            compiled_pulse,
            encoded_size=len(compiled_payload),
        )
        retained = (
            _STATIC_LINEAGE_FIXED_BYTES
            + _DOCUMENT_DECODE_MULTIPLIER * len(document_payload)
            + summary.retained_upper_bound_bytes
        )
        staging_peak = retained + len(document_payload) + len(compiled_payload)
        if staging_peak > limit:
            raise MemoryError(
                f"scan static-lineage peak {staging_peak} exceeds limit {limit}"
            )
        authority = self._content_authority()
        document_ref = authority.identify_blob(document_payload)
        compiled_ref = authority.identify_blob(compiled_payload)
        if document_ref.digest != document.fingerprint:
            raise RuntimeError("PulseDocument CAS identity differs from pulse owner")
        if compiled_ref.digest != compiled_pulse.fingerprint:
            raise RuntimeError("compiled-pulse CAS identity differs from pulse owner")
        return _StaticScanLineageAdmission(
            document_ref,
            compiled_ref,
            summary,
            retained,
        )

    def _stage_static_lineage(
        self,
        admission: _StaticScanLineageAdmission,
        document: PulseDocument,
        compiled_pulse: CompiledPulseArtifact,
    ) -> _StagedScanLineage:
        """Persist admitted lineage before delegating to hardware preflight."""

        if not isinstance(admission, _StaticScanLineageAdmission):
            raise TypeError("admission must be static scan lineage admission")
        document_payload = _encode_document(document)
        compiled_payload = encode_compiled_pulse_artifact(compiled_pulse)
        authority = self._content_authority()
        if authority.identify_blob(document_payload) != admission.pulse_document_blob:
            raise RuntimeError("PulseDocument changed after static-lineage admission")
        if authority.identify_blob(compiled_payload) != admission.compiled_pulse_blob:
            raise RuntimeError("compiled pulse changed after static-lineage admission")
        if compiled_pulse_runtime_summary(
            compiled_pulse,
            encoded_size=len(compiled_payload),
        ) != admission.compiled_pulse_runtime_summary:
            raise RuntimeError("compiled pulse resource summary changed after admission")
        for expected, actual in (
            (admission.pulse_document_blob, authority.put_blob(document_payload)),
            (admission.compiled_pulse_blob, authority.put_blob(compiled_payload)),
        ):
            if actual != expected:
                raise RuntimeError("content store changed admitted static lineage identity")
        return _StagedScanLineage(
            admission.pulse_document_blob,
            admission.compiled_pulse_blob,
            admission.compiled_pulse_runtime_summary,
        )

    def _require_final_commit(self, reference: ScanArtifactRef) -> CommitIntent:
        with self._lock:
            self._require_open()
            self._validate_reference(reference)
            matching = self._coordinator.committed_for(
                _target(self.repository_id, reference)
            )
            if not matching:
                raise PermissionError("scan lacks FINAL commit authority")
            if len(matching) != 1:
                raise ValueError("scan has multiple FINAL authorities")
            intent = matching[0]
            if intent.commit_id != _commit_id(
                intent.run_id,
                reference.manifest_digest,
            ):
                raise ValueError("scan commit identity is inconsistent")
            return intent

    def _load_index(
        self,
        reference: ScanArtifactRef,
        *,
        manifest_payload: bytes | None = None,
        memory_limit_bytes: int | None = None,
    ) -> tuple[_StoredScanIndex, int, int]:
        policy = self.resource_policy
        authority = self._content_authority()
        limit = (
            None
            if memory_limit_bytes is None
            else positive_integer(memory_limit_bytes, "memory_limit_bytes")
        )
        if limit is not None and limit <= _FIXED_MATERIALIZATION_BYTES:
            raise MemoryError("scan inspection fixed state exceeds caller memory limit")
        manifest_limit = policy.max_manifest_bytes
        if limit is not None:
            manifest_limit = min(
                manifest_limit,
                (limit - _FIXED_MATERIALIZATION_BYTES)
                // _CANONICAL_DECODE_MULTIPLIER,
            )
        if manifest_payload is None:
            try:
                payload = authority.read_manifest(
                    SCAN_ARTIFACT_NAMESPACE,
                    reference.manifest_digest,
                    max_bytes=manifest_limit,
                )
            except ContentSizeLimitError as exc:
                if limit is not None and manifest_limit < policy.max_manifest_bytes:
                    raise MemoryError(
                        "scan manifest inspection exceeds caller memory limit"
                    ) from exc
                raise ScanResourceExceeded("scan manifest exceeds policy") from exc
        else:
            payload = manifest_payload
        repository_id, metadata_ref = _decode_manifest(payload)
        if repository_id != self.repository_id:
            raise ValueError("scan manifest belongs to another repository")
        if metadata_ref.size > policy.max_metadata_blob_bytes:
            raise ScanResourceExceeded("scan metadata exceeds policy")
        inspection_peak = (
            _FIXED_MATERIALIZATION_BYTES
            + _CANONICAL_DECODE_MULTIPLIER
            * (len(payload) + metadata_ref.size)
        )
        if limit is not None and inspection_peak > limit:
            raise MemoryError("scan metadata inspection exceeds caller memory limit")
        metadata = authority.read_blob(metadata_ref, max_bytes=metadata_ref.size)
        index = _decode_metadata_index(metadata)
        compiled_ref = index.compiled_pulse_blob
        if compiled_ref.size > policy.max_compiled_pulse_blob_bytes:
            raise ScanResourceExceeded("compiled pulse blob exceeds policy")
        index.compiled_pulse_runtime_summary.require_encoded_size(
            compiled_ref.size
        )
        document_ref = index.pulse_document_blob
        if document_ref.size > policy.max_pulse_document_blob_bytes:
            raise ScanResourceExceeded("pulse document blob exceeds policy")
        if index.values_blob.size > policy.max_output_values_blob_bytes:
            raise ScanResourceExceeded("scan values blob exceeds policy")
        if index.validity_blob.size > policy.max_output_validity_blob_bytes:
            raise ScanResourceExceeded("scan validity blob exceeds policy")
        expected_values = math.prod(
            index.output_contract.output_dataset_schema.physical_shape
        ) * index.output_contract.output_dataset_schema.cell_schema.dtype.itemsize
        if index.values_blob.size != expected_values:
            raise ValueError("scan values blob size differs from output schema")
        _require_scan_dataset_facts(
            document_digest=index.pulse_document_blob.digest,
            source_ref=index.source_dataset_ref,
            source_schema=index.source_dataset_schema,
            output_contract=index.output_contract,
            output_ref=index.output_dataset_ref,
            provenance=index.provenance,
        )
        if _manifest_payload(self.repository_id, metadata_ref) != payload:
            raise ValueError("scan manifest is not canonical")
        return index, len(metadata), inspection_peak

    def _load_stored(
        self,
        reference: ScanArtifactRef,
        *,
        manifest_payload: bytes | None = None,
        memory_limit_bytes: int | None = None,
    ) -> tuple[_StoredScan, PulseDocument, int]:
        index, metadata_size, inspection_peak = self._load_index(
            reference,
            manifest_payload=manifest_payload,
            memory_limit_bytes=memory_limit_bytes,
        )
        lineage_peak = (
            inspection_peak
            + index.compiled_pulse_runtime_summary.decode_peak_upper_bound_bytes
            + _DOCUMENT_DECODE_MULTIPLIER * index.pulse_document_blob.size
        )
        if memory_limit_bytes is not None and lineage_peak > memory_limit_bytes:
            raise MemoryError("scan lineage decode exceeds caller memory limit")
        authority = self._content_authority()
        compiled_payload = authority.read_blob(
            index.compiled_pulse_blob,
            max_bytes=index.compiled_pulse_blob.size,
        )
        compiled = decode_compiled_pulse_artifact(compiled_payload)
        if index.compiled_pulse_blob.digest != compiled.fingerprint:
            raise ValueError("compiled pulse blob identity differs from fingerprint")
        if compiled_pulse_runtime_summary(
            compiled,
            encoded_size=index.compiled_pulse_blob.size,
        ) != index.compiled_pulse_runtime_summary:
            raise ValueError("compiled pulse runtime summary differs from lineage")
        document = _decode_document(
            authority.read_blob(
                index.pulse_document_blob,
                max_bytes=index.pulse_document_blob.size,
            )
        )
        if index.pulse_document_blob.digest != document.fingerprint:
            raise ValueError("PulseDocument blob identity differs from fingerprint")
        stored = _stored_scan_from_index(index, compiled)
        _require_scan_facts(
            run_id=stored.provenance.trace_binding.run_id,
            document=document,
            source_ref=stored.source_dataset_ref,
            source_schema=stored.source_dataset_schema,
            output_contract=stored.output_contract,
            output_ref=stored.output_dataset_ref,
            provenance=stored.provenance,
            pulse_evidence=stored.pulse_evidence,
        )
        return stored, document, metadata_size

    @staticmethod
    def _inspection_from_index(
        reference: ScanArtifactRef,
        index: _StoredScanIndex,
        intent: CommitIntent,
        *,
        metadata_size: int,
    ) -> ScanArtifactInspection:
        if index.provenance.trace_binding.run_id != intent.run_id or (
            index.safety_bundle_id != intent.safety_bundle_id
        ):
            raise ValueError("scan index differs from its FINAL commit intent")
        peak = (
            _FIXED_MATERIALIZATION_BYTES
            + _CANONICAL_DECODE_MULTIPLIER * metadata_size
            + 2 * index.values_blob.size
            + 3 * index.validity_blob.size
        )
        return ScanArtifactInspection(
            reference,
            index.source_dataset_ref,
            index.source_dataset_schema,
            index.output_contract,
            index.output_dataset_ref,
            index.provenance,
            index.compiled_pulse_runtime_summary,
            index.safety_bundle_id,
            peak,
        )

    def inspect_final(
        self,
        reference: ScanArtifactRef,
        *,
        memory_limit_bytes: int | None = None,
    ) -> ScanArtifactInspection:
        """Read FINAL schema/resource facts without pulse IR or document decode."""

        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            intent = self._require_final_commit(reference)
            index, metadata_size, _inspection_peak = self._load_index(
                reference,
                memory_limit_bytes=memory_limit_bytes,
            )
            return self._inspection_from_index(
                reference,
                index,
                intent,
                metadata_size=metadata_size,
            )

    def admit(self, reference: ScanArtifactRef) -> ScanArtifact:
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            intent = self._require_final_commit(reference)
            stored, document, _metadata_size = self._load_stored(reference)
            return self._artifact_from_stored(
                reference,
                stored,
                document,
                intent,
            )

    @staticmethod
    def _artifact_from_stored(
        reference: ScanArtifactRef,
        stored: _StoredScan,
        document: PulseDocument,
        intent: CommitIntent,
    ) -> ScanArtifact:
        if stored.provenance.trace_binding.run_id != intent.run_id or (
            stored.safety_bundle_id != intent.safety_bundle_id
        ):
            raise ValueError("scan artifact differs from its FINAL commit intent")
        return ScanArtifact(
            reference,
            document,
            stored.source_dataset_ref,
            stored.source_dataset_schema,
            stored.output_contract,
            stored.output_dataset_ref,
            stored.provenance,
            stored.pulse_evidence,
            stored.safety_bundle_id,
        )

    def materialize(
        self,
        reference: ScanArtifactRef,
        *,
        memory_limit_bytes: int,
    ) -> MaterializedScanData:
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            intent = self._require_final_commit(reference)
            index, metadata_size, _inspection_peak = self._load_index(
                reference,
                memory_limit_bytes=limit,
            )
            inspection = self._inspection_from_index(
                reference,
                index,
                intent,
                metadata_size=metadata_size,
            )
            peak = inspection.materialization_peak_upper_bound_bytes
            if peak > limit:
                raise MemoryError(
                    f"scan materialization peak {peak} exceeds limit {limit}"
                )
            authority = self._content_authority()
            values_payload = authority.read_blob(
                index.values_blob,
                max_bytes=index.values_blob.size,
            )
            schema = inspection.output_schema
            values = np.frombuffer(
                values_payload,
                dtype=schema.cell_schema.dtype,
            ).reshape(schema.physical_shape)
            validity_payload = authority.read_blob(
                index.validity_blob,
                max_bytes=index.validity_blob.size,
            )
            validity = _decode_validity(
                validity_payload,
                max_array_bytes=index.validity_blob.size,
            )
            block = DataBlock(
                inspection.output_dataset_ref.block_id,
                inspection.output_dataset_ref.revision,
                values,
                validity,
                schema,
            )
            snapshot = OwnedSnapshot(inspection.output_dataset_ref, block)
            return MaterializedScanData(
                inspection.ref,
                inspection.source_dataset_ref,
                snapshot,
            )

    def has(self, reference: ScanArtifactRef) -> bool:
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            try:
                self._require_final_commit(reference)
            except PermissionError:
                return False
            return self._content_authority().has_manifest(
                SCAN_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
                max_bytes=self.resource_policy.max_manifest_bytes,
            )

    def _stage_result(
        self,
        prepared: _PreparedScanDataset,
        *,
        safety_bundle_id: str,
    ) -> tuple[ScanArtifactRef, bytes]:
        if type(prepared) is not _PreparedScanDataset:
            raise TypeError("prepared must be scan application output")
        policy = self.resource_policy
        authority = self._content_authority()
        lineage = prepared.staged_lineage
        validity_payload = _encode_validity(
            prepared.output_snapshot.block.validity
        )
        values_payload = _values_payload(prepared.output_snapshot)
        for payload, maximum, label in (
            (values_payload, policy.max_output_values_blob_bytes, "scan values"),
            (validity_payload, policy.max_output_validity_blob_bytes, "scan validity"),
        ):
            if len(payload) > maximum:
                raise ScanResourceExceeded(f"{label} blob exceeds repository policy")
        values_ref = authority.identify_blob(values_payload)
        validity_ref = authority.identify_blob(validity_payload)
        stored = _StoredScan(
            lineage.pulse_document_blob,
            lineage.compiled_pulse_blob,
            lineage.compiled_pulse_runtime_summary,
            prepared.pulse_evidence,
            prepared.source_dataset_ref,
            prepared.source_dataset_schema,
            prepared.output_contract,
            prepared.output_snapshot.ref,
            prepared.provenance,
            values_ref,
            validity_ref,
            safety_bundle_id,
        )
        metadata = _encode_metadata(stored)
        if len(metadata) > policy.max_metadata_blob_bytes:
            raise ScanResourceExceeded("scan metadata blob exceeds repository policy")
        retained = _snapshot_retained_bytes(prepared.output_snapshot)
        staging_peak = (
            _snapshot_retained_bytes(prepared.source_snapshot)
            + retained
            + len(validity_payload)
            + len(metadata)
        )
        if staging_peak > prepared.memory_limit_bytes:
            raise MemoryError(
                f"scan commit staging peak {staging_peak} exceeds limit "
                f"{prepared.memory_limit_bytes}"
            )
        for expected, actual in (
            (values_ref, authority.put_blob(values_payload)),
            (validity_ref, authority.put_blob(validity_payload)),
        ):
            if actual != expected:
                raise RuntimeError("content store changed a precomputed scan blob identity")
        metadata_ref = authority.put_blob(metadata)
        manifest = _manifest_payload(self.repository_id, metadata_ref)
        if len(manifest) > policy.max_manifest_bytes:
            raise ScanResourceExceeded("scan manifest exceeds repository policy")
        reference = ScanArtifactRef(self.repository_id, sha256_digest(manifest))
        return reference, manifest

    def final_commit(
        self,
        context: PostSafetyContext,
        prepared: _PreparedScanDataset,
    ) -> FinalCommit[ScanArtifactRef]:
        if not isinstance(context, PostSafetyContext):
            raise TypeError("scan commit requires PostSafetyContext")
        if type(prepared) is not _PreparedScanDataset:
            raise TypeError("prepared must be scan application output")
        run_id, safety_bundle_id = context.authorize_commit_preparation()
        if safety_bundle_id is None:
            raise ValueError("hardware scan commit requires a safety bundle")
        if prepared.run_id != run_id:
            raise ValueError("prepared scan belongs to another Run")
        with self._root_lease.borrow() as staging_borrow:
            staging_borrow.require_active()
            reference, payload = self._stage_result(
                prepared,
                safety_bundle_id=safety_bundle_id,
            )
            if context.authorize_commit_preparation() != (
                run_id,
                safety_bundle_id,
            ):
                raise RuntimeError("scan commit subject changed while staging")
            target = _target(self.repository_id, reference)

            def publish() -> PublishedManifest[ScanArtifactRef]:
                publish_manifest_with_visibility_reconciliation(
                    self._content_authority(),
                    SCAN_ARTIFACT_NAMESPACE,
                    payload,
                    expected_digest=reference.manifest_digest,
                    max_bytes=self.resource_policy.max_manifest_bytes,
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
    ) -> PublishedManifest[ScanArtifactRef] | None:
        target = intent.target
        if (
            target.repository_id != self.repository_id
            or target.artifact_kind != _SCAN_ARTIFACT_KIND
            or target.artifact_format != SCAN_MANIFEST_SCHEMA
        ):
            raise ValueError("commit intent is not a scan target")
        reference = ScanArtifactRef(
            self.repository_id,
            target.expected_manifest_digest,
        )
        if target.target_ref != reference.target_ref or intent.commit_id != _commit_id(
            intent.run_id,
            reference.manifest_digest,
        ):
            raise ValueError("scan commit identity differs from its target")
        authority = self._content_authority()
        try:
            payload = authority.read_manifest(
                SCAN_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
                max_bytes=self.resource_policy.max_manifest_bytes,
            )
        except FileNotFoundError:
            return None
        stored, _document, _metadata_size = self._load_stored(
            reference,
            manifest_payload=payload,
        )
        if stored.provenance.trace_binding.run_id != intent.run_id or (
            stored.safety_bundle_id != intent.safety_bundle_id
        ):
            raise ValueError("visible scan differs from pending commit intent")
        for blob in (
            stored.pulse_document_blob,
            stored.compiled_pulse_blob,
            stored.values_blob,
            stored.validity_blob,
        ):
            authority.verify_blob(blob, max_bytes=blob.size)
        if authority.confirm_manifest_durable(
            SCAN_ARTIFACT_NAMESPACE,
            reference.manifest_digest,
            max_bytes=self.resource_policy.max_manifest_bytes,
        ) != payload:
            raise RuntimeError("recovery durability check changed scan manifest")
        return PublishedManifest(
            reference.target_ref,
            reference.manifest_digest,
            reference,
        )


__all__ = [
    "DEFAULT_SCAN_REPOSITORY_RESOURCE_POLICY",
    "MaterializedScanData",
    "SCAN_ARTIFACT_SCHEMA",
    "ScanArtifact",
    "ScanArtifactInspection",
    "ScanRepository",
    "ScanRepositoryResourcePolicy",
    "ScanResourceExceeded",
]
