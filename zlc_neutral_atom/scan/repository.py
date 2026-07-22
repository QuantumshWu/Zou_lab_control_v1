"""Canonical FINAL dataset authority for exact pulse scans.

The repository owns one current format.  Direct-camera and processed exact
sources are normalized by scan application adapters before this boundary; the
persisted artifact therefore contains the canonical ``(R, P, *data_shape)``
output itself and never delegates materialization back to a source repository.
"""

from __future__ import annotations

from collections.abc import Callable
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
from zlc_neutral_atom.runtime.capture import (
    MAX_CAPTURE_TERMINAL_ACK_CANONICAL_BYTES,
    MAX_CAPTURE_TERMINAL_ACK_CANONICAL_NODES,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellSchedule,
    DatasetSealProvenance,
    dataset_seal_provenance_from_tree,
    dataset_seal_provenance_to_tree,
)
from zlc_neutral_atom.runtime.run import PostSafetyContext
from zlc_neutral_atom.timing.pulse import (
    MAX_PULSE_TERMINAL_ACK_CANONICAL_BYTES,
    MAX_PULSE_TERMINAL_ACK_CANONICAL_NODES,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    CompiledPulseRuntimeSummary,
    MAX_COMPILED_PULSE_ARTIFACT_BYTES,
    PulseExecutionForm,
    compiled_pulse_runtime_summary,
    compiled_pulse_runtime_summary_from_tree,
    compiled_pulse_runtime_summary_to_tree,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
    expand_autonomous_scan_repeats,
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
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseScanProgram,
    ScanOutputContract,
    bind_scan_output_contract,
    pulse_scan_program_from_tree,
    pulse_scan_program_to_tree,
    scan_output_contract_from_tree,
    scan_output_contract_to_tree,
)
from .lineage import (
    ApiSegmentedScanExecution,
    AutonomousScanExecution,
    PulseScanExecution,
    api_segmented_cell_schedule,
    api_segmented_metadata_static_shape_from_execution,
    execution_compiled_artifacts,
    pulse_scan_execution_from_tree,
    pulse_scan_execution_to_tree,
)
from .reference import SCAN_ARTIFACT_NAMESPACE, ScanArtifactRef


SCAN_ARTIFACT_SCHEMA = "zlc_neutral_atom.scan-storage"
SCAN_MANIFEST_SCHEMA = "zlc_neutral_atom.scan-manifest"
_SCAN_ARTIFACT_KIND = "scan"
_MANIFEST_FIELDS = frozenset({"schema", "repository_id", "metadata_blob"})
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "pulse_program_blob",
        "compiled_pulse_blobs",
        "compiled_pulse_runtime_summaries",
        "execution",
        "source_dataset_ref",
        "source_dataset_schema",
        "output_contract",
        "output_dataset_ref",
        "dataset_provenance",
        "output_values_blob",
        "output_validity_blob",
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
_PROGRAM_DECODE_MULTIPLIER = 16
_STATIC_LINEAGE_FIXED_BYTES = 1 << 20
_API_METADATA_DYNAMIC_FIXED_BYTES = (
    (1 << 20) + MAX_CAPTURE_TERMINAL_ACK_CANONICAL_BYTES
)
_API_METADATA_DYNAMIC_BYTES_PER_SEGMENT = (
    (4 << 10) + MAX_PULSE_TERMINAL_ACK_CANONICAL_BYTES
)
_API_METADATA_DYNAMIC_FIXED_NODES = (
    (4 << 10) + MAX_CAPTURE_TERMINAL_ACK_CANONICAL_NODES
)
_API_METADATA_DYNAMIC_NODES_PER_SEGMENT = (
    128 + MAX_PULSE_TERMINAL_ACK_CANONICAL_NODES
)


class ScanResourceExceeded(RuntimeError):
    """One canonical scan exceeds an explicit storage/admission budget."""


def _api_segmented_metadata_cardinality_floor(
    point_count: int,
    repeat_count: int,
) -> tuple[int, int]:
    """Cheap R*P-only floor; actual static metadata is admitted after binding."""

    points = positive_integer(point_count, "point_count")
    repeats = positive_integer(repeat_count, "repeat_count")
    segments = points * repeats
    return (
        _API_METADATA_DYNAMIC_FIXED_BYTES
        + segments * _API_METADATA_DYNAMIC_BYTES_PER_SEGMENT,
        _API_METADATA_DYNAMIC_FIXED_NODES
        + segments * _API_METADATA_DYNAMIC_NODES_PER_SEGMENT,
    )


@dataclass(frozen=True, slots=True)
class ScanRepositoryResourcePolicy:
    max_manifest_bytes: int = 1 << 20
    max_metadata_blob_bytes: int = 16 << 20
    max_pulse_program_blob_bytes: int = 16 << 20
    max_total_compiled_pulse_blob_bytes: int = 8 << 30
    max_output_values_blob_bytes: int = 8 << 30
    max_output_validity_blob_bytes: int = 2 << 30

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_SCAN_REPOSITORY_RESOURCE_POLICY = ScanRepositoryResourcePolicy()


@dataclass(frozen=True, slots=True)
class _StoredScan:
    pulse_program_blob: ContentRef
    compiled_pulse_blobs: tuple[ContentRef, ...]
    compiled_pulse_runtime_summaries: tuple[CompiledPulseRuntimeSummary, ...]
    execution: PulseScanExecution
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance
    values_blob: ContentRef
    validity_blob: ContentRef

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_program_blob, ContentRef):
            raise TypeError("pulse_program_blob must be ContentRef")
        blobs = tuple(self.compiled_pulse_blobs)
        summaries = tuple(self.compiled_pulse_runtime_summaries)
        if not blobs or any(not isinstance(item, ContentRef) for item in blobs):
            raise TypeError("compiled_pulse_blobs must contain ContentRef values")
        if len(summaries) != len(blobs) or any(
            not isinstance(item, CompiledPulseRuntimeSummary) for item in summaries
        ):
            raise TypeError("compiled pulse summaries must align with their blobs")
        if not isinstance(
            self.execution,
            (AutonomousScanExecution, ApiSegmentedScanExecution),
        ):
            raise TypeError("execution must be a PulseScanExecution")
        artifacts = execution_compiled_artifacts(self.execution)
        if tuple(item.digest for item in blobs) != tuple(
            item.fingerprint for item in artifacts
        ):
            raise ValueError("stored compiled blobs differ from execution evidence")
        if self.pulse_program_blob.digest != self.execution.program.fingerprint:
            raise ValueError("stored program blob differs from execution program")
        object.__setattr__(self, "compiled_pulse_blobs", blobs)
        object.__setattr__(self, "compiled_pulse_runtime_summaries", summaries)
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


@dataclass(frozen=True, slots=True)
class _StoredScanIndex:
    """Small current-format index decoded without program or pulse IR."""

    pulse_program_blob: ContentRef
    compiled_pulse_blobs: tuple[ContentRef, ...]
    compiled_pulse_runtime_summaries: tuple[CompiledPulseRuntimeSummary, ...]
    execution_tree: object
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance
    values_blob: ContentRef
    validity_blob: ContentRef


@dataclass(frozen=True, slots=True)
class _StaticScanLineageAdmission:
    """Pure pre-Run resource proof for immutable scan-program lineage."""

    pulse_program_blob: ContentRef
    compiled_pulse_blobs: tuple[ContentRef, ...]
    compiled_pulse_runtime_summaries: tuple[CompiledPulseRuntimeSummary, ...]
    retained_upper_bound_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_program_blob, ContentRef):
            raise TypeError("staged scan program blob must be ContentRef")
        blobs = tuple(self.compiled_pulse_blobs)
        summaries = tuple(self.compiled_pulse_runtime_summaries)
        if not blobs or any(not isinstance(item, ContentRef) for item in blobs):
            raise TypeError("staged compiled pulse blobs must contain ContentRef")
        if len(summaries) != len(blobs) or any(
            not isinstance(item, CompiledPulseRuntimeSummary) for item in summaries
        ):
            raise TypeError("staged summaries must align with compiled pulse blobs")
        object.__setattr__(self, "compiled_pulse_blobs", blobs)
        object.__setattr__(self, "compiled_pulse_runtime_summaries", summaries)
        positive_integer(
            self.retained_upper_bound_bytes,
            "retained_upper_bound_bytes",
        )


@dataclass(frozen=True, slots=True)
class _StagedScanLineage:
    """Pre-FIRE CAS references verified against one static admission."""

    pulse_program_blob: ContentRef
    compiled_pulse_blobs: tuple[ContentRef, ...]
    compiled_pulse_runtime_summaries: tuple[CompiledPulseRuntimeSummary, ...]


_API_FINAL_METADATA_ADMISSION_SCHEMA = (
    "zlc_neutral_atom.ApiFinalMetadataAdmission"
)


@dataclass(frozen=True, slots=True)
class _ApiFinalMetadataAdmission:
    """Process-local proof binding one API FINAL tree to pre-FIRE static facts."""

    issuer_token: object
    artifact_schema: str
    policy_max_bytes: int
    program_fingerprint: str
    source_block_id: BlockId
    source_schema_fingerprint: str
    output_contract_fingerprint: str
    static_lineage_fingerprint: str
    execution_shape_fingerprint: str
    static_charge_fingerprint: str
    point_count: int
    repeat_count: int
    admitted_max_bytes: int
    admitted_structure_limit: int

    def __post_init__(self) -> None:
        if self.artifact_schema != SCAN_ARTIFACT_SCHEMA:
            raise ValueError("API metadata admission targets another artifact schema")
        if not isinstance(self.source_block_id, BlockId):
            raise TypeError("source_block_id must be BlockId")
        for field in (
            "program_fingerprint",
            "source_schema_fingerprint",
            "output_contract_fingerprint",
            "static_lineage_fingerprint",
            "execution_shape_fingerprint",
            "static_charge_fingerprint",
        ):
            sha256_text(getattr(self, field), field)
        for field in (
            "policy_max_bytes",
            "point_count",
            "repeat_count",
            "admitted_max_bytes",
            "admitted_structure_limit",
        ):
            positive_integer(getattr(self, field), field)
        if self.admitted_max_bytes > self.policy_max_bytes:
            raise ValueError("API metadata admission exceeds repository policy")
        if self.admitted_structure_limit > min(
            _METADATA_LIMITS.max_nodes,
            _METADATA_LIMITS.max_container_entries,
        ):
            raise ValueError("API metadata admission exceeds canonical policy")


def _static_lineage_metadata_tree(
    value: _StaticScanLineageAdmission | _StagedScanLineage,
) -> dict[str, object]:
    if not isinstance(value, (_StaticScanLineageAdmission, _StagedScanLineage)):
        raise TypeError("value must be admitted or staged scan lineage")
    return {
        "pulse_program_blob": content_ref_to_tree(value.pulse_program_blob),
        "compiled_pulse_blobs": [
            content_ref_to_tree(item) for item in value.compiled_pulse_blobs
        ],
        "compiled_pulse_runtime_summaries": [
            compiled_pulse_runtime_summary_to_tree(item)
            for item in value.compiled_pulse_runtime_summaries
        ],
    }


def _api_metadata_static_charge_tree(
    program: ApiSlotSegmentedProgram,
    static_lineage: _StaticScanLineageAdmission | _StagedScanLineage,
    source_block_id: BlockId,
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    execution_shape_tree: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": _API_FINAL_METADATA_ADMISSION_SCHEMA,
        "artifact_schema": SCAN_ARTIFACT_SCHEMA,
        "program_fingerprint": program.fingerprint,
        "static_lineage": _static_lineage_metadata_tree(static_lineage),
        "source_block_id": source_block_id.value,
        "source_dataset_schema": dataset_schema_to_tree(source_schema),
        "output_contract": scan_output_contract_to_tree(output_contract),
        "execution_static_shape": execution_shape_tree,
    }


def _encode_static_metadata_charge(
    tree: dict[str, object],
) -> tuple[bytes, int]:
    """Encode once and derive the exact encoder-owned structural charge.

    The binary search deliberately delegates node/container accounting to the
    canonical owner instead of copying its traversal rules into the repository.
    """

    try:
        payload = encode(tree, limits=_METADATA_LIMITS)
    except CanonicalEncodingError as exc:
        raise ScanResourceExceeded(
            "API segmented static metadata exceeds canonical policy"
        ) from exc
    low = 1
    high = min(
        _METADATA_LIMITS.max_nodes,
        _METADATA_LIMITS.max_container_entries,
    )
    while low < high:
        middle = (low + high) // 2
        limits = CanonicalDecodeLimits(
            max_depth=_METADATA_LIMITS.max_depth,
            max_nodes=middle,
            max_container_entries=middle,
            max_arrays=0,
            max_total_array_bytes=0,
        )
        try:
            encode(tree, limits=limits)
        except CanonicalEncodingError:
            low = middle + 1
        else:
            high = middle
    return payload, low


def _api_repeated_point_shape_charge(
    execution_shape_tree: dict[str, object],
    repeat_count: int,
) -> tuple[int, int]:
    """Charge point-owned trigger/join text each time FINAL repeats it."""

    points = execution_shape_tree.get("points")
    if not isinstance(points, list) or not points or any(
        not isinstance(item, dict) for item in points
    ):
        raise ValueError("API execution static shape omits point mappings")
    bytes_per_repeat = 0
    structure_per_repeat = 0
    for point in points:
        payload, structure = _encode_static_metadata_charge({"point": point})
        bytes_per_repeat += len(payload)
        structure_per_repeat += structure
    repeats = positive_integer(repeat_count, "repeat_count")
    return bytes_per_repeat * repeats, structure_per_repeat * repeats


@dataclass(frozen=True, slots=True)
class ScanArtifact:
    """Admitted metadata for one canonical FINAL scan dataset."""

    ref: ScanArtifactRef
    execution: PulseScanExecution
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ScanArtifactRef):
            raise TypeError("ref must be ScanArtifactRef")
        if not isinstance(
            self.execution,
            (AutonomousScanExecution, ApiSegmentedScanExecution),
        ):
            raise TypeError("execution must be a PulseScanExecution")
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

    @property
    def output_schema(self) -> DatasetSchema:
        return self.output_contract.output_dataset_schema


@dataclass(frozen=True, slots=True)
class ScanArtifactInspection:
    """FINAL dataset facts obtained without decoding pulse IR or scan program."""

    ref: ScanArtifactRef
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance
    pulse_runtime_summaries: tuple[CompiledPulseRuntimeSummary, ...]
    inspection_retained_upper_bound_bytes: int
    inspection_decode_peak_upper_bound_bytes: int
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
        summaries = tuple(self.pulse_runtime_summaries)
        if not summaries or any(
            not isinstance(item, CompiledPulseRuntimeSummary) for item in summaries
        ):
            raise TypeError(
                "pulse_runtime_summaries must contain CompiledPulseRuntimeSummary values"
            )
        object.__setattr__(self, "pulse_runtime_summaries", summaries)
        positive_integer(
            self.inspection_retained_upper_bound_bytes,
            "inspection_retained_upper_bound_bytes",
        )
        positive_integer(
            self.inspection_decode_peak_upper_bound_bytes,
            "inspection_decode_peak_upper_bound_bytes",
        )
        if (
            self.inspection_decode_peak_upper_bound_bytes
            < self.inspection_retained_upper_bound_bytes
        ):
            raise ValueError("scan inspection peak is smaller than retained state")
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


def _require_api_metadata_admission_facts(
    admission: _ApiFinalMetadataAdmission,
    *,
    execution: ApiSegmentedScanExecution,
    source_ref: DatasetRevisionRef,
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    provenance: DatasetSealProvenance,
    staged_lineage: _StagedScanLineage,
) -> None:
    if not isinstance(admission, _ApiFinalMetadataAdmission):
        raise TypeError("API execution requires a FINAL metadata admission")
    if admission.artifact_schema != SCAN_ARTIFACT_SCHEMA:
        raise RuntimeError("API metadata admission artifact schema drifted")
    program = execution.program
    if (
        admission.program_fingerprint != program.fingerprint
        or admission.point_count != program.point_count
        or admission.repeat_count != program.repeat_count
    ):
        raise RuntimeError("API metadata admission belongs to another program")
    if admission.source_schema_fingerprint != source_schema.fingerprint:
        raise RuntimeError("API metadata admission source schema drifted")
    if admission.source_block_id != source_ref.block_id:
        raise RuntimeError("API metadata admission source block identity drifted")
    if admission.output_contract_fingerprint != output_contract.fingerprint:
        raise RuntimeError("API metadata admission output contract drifted")
    lineage_tree = _static_lineage_metadata_tree(staged_lineage)
    if admission.static_lineage_fingerprint != canonical_digest(lineage_tree):
        raise RuntimeError("API metadata admission static lineage drifted")
    execution_shape = api_segmented_metadata_static_shape_from_execution(
        execution,
        provenance,
    )
    if admission.execution_shape_fingerprint != canonical_digest(execution_shape):
        raise RuntimeError("API metadata admission execution shape drifted")
    static_tree = _api_metadata_static_charge_tree(
        program,
        staged_lineage,
        source_ref.block_id,
        source_schema,
        output_contract,
        execution_shape,
    )
    try:
        static_payload = encode(static_tree, limits=_METADATA_LIMITS)
    except CanonicalEncodingError as exc:
        raise RuntimeError(
            "API metadata pre-FIRE admission invariant was violated"
        ) from exc
    if admission.static_charge_fingerprint != sha256_digest(static_payload):
        raise RuntimeError("API metadata admission static charge drifted")


class _PreparedScanDataset:
    """Process-local output minted only from one opaque joint pipeline result."""

    __slots__ = (
        "run_id",
        "execution",
        "source_snapshot",
        "output_contract",
        "output_snapshot",
        "provenance",
        "staged_lineage",
        "api_metadata_admission",
        "memory_limit_bytes",
    )

    def __init__(
        self,
        token: object,
        *,
        run_id: str,
        execution: PulseScanExecution,
        source_snapshot: OwnedSnapshot,
        output_contract: ScanOutputContract,
        output_snapshot: OwnedSnapshot,
        provenance: DatasetSealProvenance,
        staged_lineage: _StagedScanLineage,
        api_metadata_admission: _ApiFinalMetadataAdmission | None,
        memory_limit_bytes: int,
    ) -> None:
        if token is not _SCAN_APPLICATION_TOKEN:
            raise PermissionError("prepared scan datasets are minted by scan application")
        canonical_text(run_id, "run_id")
        if not isinstance(
            execution,
            (AutonomousScanExecution, ApiSegmentedScanExecution),
        ):
            raise TypeError("execution must be a PulseScanExecution")
        if not isinstance(source_snapshot, OwnedSnapshot):
            raise TypeError("source_snapshot must be OwnedSnapshot")
        if not isinstance(output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(output_snapshot, OwnedSnapshot):
            raise TypeError("output_snapshot must be OwnedSnapshot")
        if not isinstance(provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if not isinstance(staged_lineage, _StagedScanLineage):
            raise TypeError("staged_lineage must be pre-FIRE scan lineage")
        if staged_lineage.pulse_program_blob.digest != execution.program.fingerprint:
            raise ValueError("staged scan-program identity changed")
        artifacts = execution_compiled_artifacts(execution)
        if tuple(item.digest for item in staged_lineage.compiled_pulse_blobs) != tuple(
            item.fingerprint for item in artifacts
        ):
            raise ValueError("staged compiled pulse identities changed")
        if len(staged_lineage.compiled_pulse_runtime_summaries) != len(artifacts):
            raise ValueError("staged pulse summaries differ from execution")
        for summary, blob in zip(
            staged_lineage.compiled_pulse_runtime_summaries,
            staged_lineage.compiled_pulse_blobs,
        ):
            summary.require_encoded_size(blob.size)
        if isinstance(execution, ApiSegmentedScanExecution):
            if not isinstance(api_metadata_admission, _ApiFinalMetadataAdmission):
                raise TypeError("API prepared scan requires metadata admission")
            _require_api_metadata_admission_facts(
                api_metadata_admission,
                execution=execution,
                source_ref=source_snapshot.ref,
                source_schema=source_snapshot.block.schema,
                output_contract=output_contract,
                provenance=provenance,
                staged_lineage=staged_lineage,
            )
        elif api_metadata_admission is not None:
            raise ValueError("autonomous scan cannot carry API metadata admission")
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        _require_scan_facts(
            run_id=run_id,
            execution=execution,
            source_ref=source_snapshot.ref,
            source_schema=source_snapshot.block.schema,
            output_contract=output_contract,
            output_ref=output_snapshot.ref,
            provenance=provenance,
        )
        if output_snapshot.block.schema != output_contract.output_dataset_schema:
            raise ValueError("prepared output snapshot has another ScanOutputContract")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "execution", execution)
        object.__setattr__(self, "source_snapshot", source_snapshot)
        object.__setattr__(self, "output_contract", output_contract)
        object.__setattr__(self, "output_snapshot", output_snapshot)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "staged_lineage", staged_lineage)
        object.__setattr__(self, "api_metadata_admission", api_metadata_admission)
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
    program: PulseScanProgram,
    source_ref: DatasetRevisionRef,
    output_contract: ScanOutputContract,
) -> DatasetRevisionRef:
    if not isinstance(
        program,
        (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
    ):
        raise TypeError("program must be a PulseScanProgram")
    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("source_ref must be DatasetRevisionRef")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    return _scan_output_dataset_ref_for_program_digest(
        program.fingerprint,
        source_ref,
        output_contract,
    )


def _scan_output_dataset_ref_for_program_digest(
    program_digest: str,
    source_ref: DatasetRevisionRef,
    output_contract: ScanOutputContract,
) -> DatasetRevisionRef:
    if not isinstance(program_digest, str):
        raise TypeError("program_digest must be str")
    sha256_text(program_digest, "program_digest")
    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("source_ref must be DatasetRevisionRef")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.scan-output",
            "pulse_scan_program": program_digest,
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
    execution: PulseScanExecution,
    source_ref: DatasetRevisionRef,
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    output_ref: DatasetRevisionRef,
    provenance: DatasetSealProvenance,
) -> None:
    canonical_text(run_id, "run_id")
    _require_scan_dataset_facts(
        program_digest=execution.program.fingerprint,
        source_ref=source_ref,
        source_schema=source_schema,
        output_contract=output_contract,
        output_ref=output_ref,
        provenance=provenance,
    )
    program = execution.program
    point_table = program.point_table
    repeat_count = program.repeat_count
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
    camera_schema = execution.camera.validate_source_schema(source_schema)
    if isinstance(execution, AutonomousScanExecution):
        if execution.evidence.join_contract.scan_point_layout != point_table.point_layout:
            raise ValueError("pulse join layout differs from the logical ScanPointTable")
        expected_schedule = DatasetCellSchedule.from_cells(
            source_schema,
            execution.evidence.join_contract.iter_cell_schedule(
                execution.evidence.trigger_schedule,
                source_schema,
            ),
        )
        expected_join = expected_schedule.digest_for_schema(source_schema)
        expected_events = execution.evidence.expected_trigger_count
    elif isinstance(execution, ApiSegmentedScanExecution):
        expected_schedule = api_segmented_cell_schedule(
            program,
            source_schema,
        )
        expected_join = expected_schedule.digest_for_schema(source_schema)
        expected_events = repeat_count * point_table.point_layout.storage_size
    else:
        raise TypeError("execution must be a PulseScanExecution")
    if provenance.join_plan_digest != expected_join:
        raise ValueError("source dataset schedule differs from pulse evidence")
    event_count = provenance.end_sequence - provenance.start_sequence
    if event_count != expected_events:
        raise ValueError("source provenance count differs from pulse triggers")
    if provenance.trace_binding.run_id != run_id:
        raise ValueError("source provenance belongs to another Run")
    execution.camera.require_schedule(expected_schedule, camera_schema)
    execution.camera.validate_dataset_provenance(provenance)
    resolved = bind_scan_output_contract(
        source_schema,
        point_table,
        output_contract.committed_transform,
    )
    if resolved != output_contract:
        raise ValueError("ScanOutputContract differs from source schema and point table")


def _require_scan_dataset_facts(
    *,
    program_digest: str,
    source_ref: DatasetRevisionRef,
    source_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    output_ref: DatasetRevisionRef,
    provenance: DatasetSealProvenance,
) -> None:
    sha256_text(program_digest, "program_digest")
    if source_ref.stream_generation != provenance.generation:
        raise ValueError("source dataset generation differs from its provenance")
    if source_ref.schema_fingerprint != source_schema.fingerprint:
        raise ValueError("source dataset ref differs from its source schema")
    if output_ref.schema_fingerprint != output_contract.output_schema_fingerprint:
        raise ValueError("scan output ref differs from its output schema")
    if output_ref != _scan_output_dataset_ref_for_program_digest(
        program_digest,
        source_ref,
        output_contract,
    ):
        raise ValueError("scan output dataset identity differs from frozen inputs")


def _require_program_artifacts(
    program: PulseScanProgram,
    compiled_pulses: tuple[CompiledPulseArtifact, ...],
) -> tuple[CompiledPulseArtifact, ...]:
    """Fail before staging when compiled pulses do not implement the program."""

    pulses = tuple(compiled_pulses)
    if any(not isinstance(item, CompiledPulseArtifact) for item in pulses):
        raise TypeError("compiled_pulses must contain CompiledPulseArtifact values")
    if isinstance(program, AutonomousScanSlotProgram):
        if len(pulses) != 1:
            raise ValueError("autonomous scan requires exactly one compiled artifact")
        artifact = pulses[0]
        if artifact.execution_form is not PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
            raise ValueError("autonomous scan requires AUTONOMOUS_SCAN_ONCE")
        expected = expand_autonomous_scan_repeats(program.execution_document)
        if artifact.source_document_digest != expected.fingerprint:
            raise ValueError("autonomous compiled pulse differs from its program")
    elif isinstance(program, ApiSlotSegmentedProgram):
        documents = program.resolved_point_documents
        if len(pulses) != len(documents):
            raise ValueError("API scan requires one compiled artifact per point")
        if any(
            artifact.execution_form is not PulseExecutionForm.STATIC_ONCE
            or artifact.source_document_digest != document.fingerprint
            for artifact, document in zip(pulses, documents)
        ):
            raise ValueError("API compiled pulses differ from resolved point documents")
    else:
        raise TypeError("program must be a PulseScanProgram")
    return pulses


def _reject_arrays(events) -> None:
    if any(isinstance(event, CanonicalArrayEvent) for event in events):
        raise ScanResourceExceeded("scan metadata cannot embed ndarray payloads")


def _metadata_tree(
    value: _StoredScan | _StoredScanIndex,
) -> dict[str, object]:
    execution_tree = (
        pulse_scan_execution_to_tree(value.execution)
        if isinstance(value, _StoredScan)
        else value.execution_tree
    )
    return {
        "schema": SCAN_ARTIFACT_SCHEMA,
        "pulse_program_blob": content_ref_to_tree(value.pulse_program_blob),
        "compiled_pulse_blobs": [
            content_ref_to_tree(item) for item in value.compiled_pulse_blobs
        ],
        "compiled_pulse_runtime_summaries": [
            compiled_pulse_runtime_summary_to_tree(item)
            for item in value.compiled_pulse_runtime_summaries
        ],
        "execution": execution_tree,
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
    }


def _encode_metadata(
    value: _StoredScan | _StoredScanIndex,
    *,
    limits: CanonicalDecodeLimits = _METADATA_LIMITS,
) -> bytes:
    try:
        return encode(_metadata_tree(value), limits=limits)
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
    blob_trees = data["compiled_pulse_blobs"]
    summary_trees = data["compiled_pulse_runtime_summaries"]
    if not isinstance(blob_trees, list) or not isinstance(summary_trees, list):
        raise TypeError("compiled pulse blobs and summaries must be lists")
    blobs = tuple(content_ref_from_tree(item) for item in blob_trees)
    summaries = tuple(
        compiled_pulse_runtime_summary_from_tree(item) for item in summary_trees
    )
    if not blobs or len(blobs) != len(summaries):
        raise ValueError("compiled pulse blobs and summaries must align")
    value = _StoredScanIndex(
        content_ref_from_tree(data["pulse_program_blob"]),
        blobs,
        summaries,
        data["execution"],
        dataset_revision_ref_from_tree(data["source_dataset_ref"]),
        dataset_schema_from_tree(data["source_dataset_schema"]),
        scan_output_contract_from_tree(data["output_contract"]),
        dataset_revision_ref_from_tree(data["output_dataset_ref"]),
        dataset_seal_provenance_from_tree(data["dataset_provenance"]),
        content_ref_from_tree(data["output_values_blob"]),
        content_ref_from_tree(data["output_validity_blob"]),
    )
    if _encode_metadata(value) != payload:
        raise ValueError("scan metadata is typed but non-canonical")
    return value


def _stored_scan_from_index(
    index: _StoredScanIndex,
    execution: PulseScanExecution,
) -> _StoredScan:
    return _StoredScan(
        index.pulse_program_blob,
        index.compiled_pulse_blobs,
        index.compiled_pulse_runtime_summaries,
        execution,
        index.source_dataset_ref,
        index.source_dataset_schema,
        index.output_contract,
        index.output_dataset_ref,
        index.provenance,
        index.values_blob,
        index.validity_blob,
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


def _encode_program(program: PulseScanProgram) -> bytes:
    return encode(pulse_scan_program_to_tree(program), limits=_METADATA_LIMITS)


def _decode_program(payload: bytes) -> PulseScanProgram:
    program = pulse_scan_program_from_tree(
        decode(payload, admit_structure=_reject_arrays, limits=_METADATA_LIMITS)
    )
    if _encode_program(program) != payload:
        raise ValueError("PulseScanProgram blob is typed but non-canonical")
    return program


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
        self._resource_policy = resource_policy
        self._lock = threading.RLock()
        self._closed = False
        self._api_metadata_admission_token = object()
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

    @property
    def resource_policy(self) -> ScanRepositoryResourcePolicy:
        """Return the immutable policy frozen when this repository was opened."""

        return self._resource_policy

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("scan repository is closed")
        self._root_lease.require_active()

    def _require_active(self) -> None:
        """Application admission probe without exposing repository internals."""

        with self._lock:
            self._require_open()

    def admit_api_execution_cardinality(
        self,
        point_count: int,
        repeat_count: int,
    ) -> int:
        """Reject an impossible R*P floor before resolving any API point."""

        projected_bytes, projected_nodes = _api_segmented_metadata_cardinality_floor(
            point_count,
            repeat_count,
        )
        with self._lock:
            self._require_open()
            maximum_bytes = self.resource_policy.max_metadata_blob_bytes
        if projected_bytes > maximum_bytes:
            raise ScanResourceExceeded(
                "API segmented FINAL metadata requires "
                f"{projected_bytes} admitted bytes; repository limit is "
                f"{maximum_bytes}"
            )
        if (
            projected_nodes > _METADATA_LIMITS.max_nodes
            or projected_nodes > _METADATA_LIMITS.max_container_entries
        ):
            raise ScanResourceExceeded(
                "API segmented FINAL metadata requires "
                f"{projected_nodes} admitted canonical nodes; repository limit is "
                f"{min(_METADATA_LIMITS.max_nodes, _METADATA_LIMITS.max_container_entries)}"
            )
        return projected_bytes

    def _admit_api_final_metadata(
        self,
        program: ApiSlotSegmentedProgram,
        static_lineage: _StaticScanLineageAdmission,
        source_block_id: BlockId,
        source_schema: DatasetSchema,
        output_contract: ScanOutputContract,
        execution_shape_tree: dict[str, object],
    ) -> _ApiFinalMetadataAdmission:
        """Bind actual frozen metadata facts before a Run can reach FIRE."""

        if not isinstance(program, ApiSlotSegmentedProgram):
            raise TypeError("program must be ApiSlotSegmentedProgram")
        if not isinstance(static_lineage, _StaticScanLineageAdmission):
            raise TypeError("static_lineage must be admitted scan lineage")
        if not isinstance(source_block_id, BlockId):
            raise TypeError("source_block_id must be BlockId")
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if not isinstance(output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(execution_shape_tree, dict):
            raise TypeError("execution_shape_tree must be a canonical mapping")
        floor_bytes, floor_nodes = _api_segmented_metadata_cardinality_floor(
            program.point_count,
            program.repeat_count,
        )
        static_tree = _api_metadata_static_charge_tree(
            program,
            static_lineage,
            source_block_id,
            source_schema,
            output_contract,
            execution_shape_tree,
        )
        static_payload, static_structure = _encode_static_metadata_charge(static_tree)
        repeated_point_bytes, repeated_point_structure = (
            _api_repeated_point_shape_charge(
                execution_shape_tree,
                program.repeat_count,
            )
        )
        projected_bytes = len(static_payload) + floor_bytes + repeated_point_bytes
        projected_structure = (
            static_structure + floor_nodes + repeated_point_structure
        )
        with self._lock:
            self._require_open()
            policy_max_bytes = self.resource_policy.max_metadata_blob_bytes
            issuer_token = self._api_metadata_admission_token
        if projected_bytes > policy_max_bytes:
            raise ScanResourceExceeded(
                "API segmented FINAL metadata requires "
                f"{projected_bytes} admitted bytes after static binding; "
                f"repository limit is {policy_max_bytes}"
            )
        canonical_limit = min(
            _METADATA_LIMITS.max_nodes,
            _METADATA_LIMITS.max_container_entries,
        )
        if projected_structure > canonical_limit:
            raise ScanResourceExceeded(
                "API segmented FINAL metadata requires "
                f"{projected_structure} admitted canonical nodes after static binding; "
                f"repository limit is {canonical_limit}"
            )
        lineage_tree = _static_lineage_metadata_tree(static_lineage)
        return _ApiFinalMetadataAdmission(
            issuer_token,
            SCAN_ARTIFACT_SCHEMA,
            policy_max_bytes,
            program.fingerprint,
            source_block_id,
            source_schema.fingerprint,
            output_contract.fingerprint,
            canonical_digest(lineage_tree),
            canonical_digest(execution_shape_tree),
            sha256_digest(static_payload),
            program.point_count,
            program.repeat_count,
            projected_bytes,
            projected_structure,
        )

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
        program: PulseScanProgram,
        compiled_pulses: tuple[CompiledPulseArtifact, ...],
        *,
        memory_limit_bytes: int,
    ) -> _StaticScanLineageAdmission:
        """Prove immutable lineage policy/peak without writing repository state."""

        if not isinstance(
            program,
            (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
        ):
            raise TypeError("program must be a PulseScanProgram")
        if isinstance(program, ApiSlotSegmentedProgram):
            self.admit_api_execution_cardinality(
                program.point_count,
                program.repeat_count,
            )
        pulses = _require_program_artifacts(program, compiled_pulses)
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        policy = self.resource_policy
        program_payload = _encode_program(program)
        compiled_payloads = tuple(
            encode_compiled_pulse_artifact(item) for item in pulses
        )
        if len(program_payload) > policy.max_pulse_program_blob_bytes:
            raise ScanResourceExceeded("scan program exceeds repository policy")
        total_compiled_size = sum(len(item) for item in compiled_payloads)
        if total_compiled_size > policy.max_total_compiled_pulse_blob_bytes:
            raise ScanResourceExceeded(
                "scan compiled pulses exceed total repository policy"
            )
        summaries = tuple(
            compiled_pulse_runtime_summary(pulse, encoded_size=len(payload))
            for pulse, payload in zip(pulses, compiled_payloads)
        )
        retained = (
            _STATIC_LINEAGE_FIXED_BYTES
            + _PROGRAM_DECODE_MULTIPLIER * len(program_payload)
            + sum(item.retained_upper_bound_bytes for item in summaries)
        )
        staging_peak = retained + len(program_payload) + total_compiled_size
        if staging_peak > limit:
            raise MemoryError(
                f"scan static-lineage peak {staging_peak} exceeds limit {limit}"
            )
        authority = self._content_authority()
        program_ref = authority.identify_blob(program_payload)
        compiled_refs = tuple(
            authority.identify_blob(item) for item in compiled_payloads
        )
        if program_ref.digest != program.fingerprint:
            raise RuntimeError("scan-program CAS identity differs from its owner")
        if tuple(item.digest for item in compiled_refs) != tuple(
            item.fingerprint for item in pulses
        ):
            raise RuntimeError("compiled-pulse CAS identity differs from pulse owner")
        return _StaticScanLineageAdmission(
            program_ref,
            compiled_refs,
            summaries,
            retained,
        )

    def _stage_static_lineage(
        self,
        admission: _StaticScanLineageAdmission,
        program: PulseScanProgram,
        compiled_pulses: tuple[CompiledPulseArtifact, ...],
    ) -> _StagedScanLineage:
        """Persist admitted lineage before delegating to hardware preflight."""

        if not isinstance(admission, _StaticScanLineageAdmission):
            raise TypeError("admission must be static scan lineage admission")
        pulses = _require_program_artifacts(program, compiled_pulses)
        program_payload = _encode_program(program)
        compiled_payloads = tuple(
            encode_compiled_pulse_artifact(item) for item in pulses
        )
        authority = self._content_authority()
        if authority.identify_blob(program_payload) != admission.pulse_program_blob:
            raise RuntimeError("scan program changed after static-lineage admission")
        identified = tuple(authority.identify_blob(item) for item in compiled_payloads)
        if identified != admission.compiled_pulse_blobs:
            raise RuntimeError("compiled pulses changed after static-lineage admission")
        summaries = tuple(
            compiled_pulse_runtime_summary(pulse, encoded_size=len(payload))
            for pulse, payload in zip(pulses, compiled_payloads)
        )
        if summaries != admission.compiled_pulse_runtime_summaries:
            raise RuntimeError("compiled pulse resource summaries changed after admission")
        pairs = (
            (admission.pulse_program_blob, authority.put_blob(program_payload)),
            *tuple(
                (expected, authority.put_blob(payload))
                for expected, payload in zip(
                    admission.compiled_pulse_blobs,
                    compiled_payloads,
                )
            ),
        )
        for expected, actual in pairs:
            if actual != expected:
                raise RuntimeError("content store changed admitted static lineage identity")
        return _StagedScanLineage(
            admission.pulse_program_blob,
            admission.compiled_pulse_blobs,
            admission.compiled_pulse_runtime_summaries,
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
        if sum(item.size for item in index.compiled_pulse_blobs) > (
            policy.max_total_compiled_pulse_blob_bytes
        ):
            raise ScanResourceExceeded("compiled pulse blobs exceed total policy")
        for compiled_ref, summary in zip(
            index.compiled_pulse_blobs,
            index.compiled_pulse_runtime_summaries,
        ):
            if compiled_ref.size > MAX_COMPILED_PULSE_ARTIFACT_BYTES:
                raise ScanResourceExceeded("one compiled pulse exceeds pulse-owner policy")
            summary.require_encoded_size(compiled_ref.size)
        program_ref = index.pulse_program_blob
        if program_ref.size > policy.max_pulse_program_blob_bytes:
            raise ScanResourceExceeded("pulse program blob exceeds policy")
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
            program_digest=index.pulse_program_blob.digest,
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
    ) -> tuple[_StoredScan, int]:
        index, metadata_size, inspection_peak = self._load_index(
            reference,
            manifest_payload=manifest_payload,
            memory_limit_bytes=memory_limit_bytes,
        )
        lineage_peak = (
            inspection_peak
            + sum(
                item.decode_peak_upper_bound_bytes
                for item in index.compiled_pulse_runtime_summaries
            )
            + _PROGRAM_DECODE_MULTIPLIER * index.pulse_program_blob.size
        )
        if memory_limit_bytes is not None and lineage_peak > memory_limit_bytes:
            raise MemoryError("scan lineage decode exceeds caller memory limit")
        authority = self._content_authority()
        compiled: list[CompiledPulseArtifact] = []
        for compiled_ref, expected_summary in zip(
            index.compiled_pulse_blobs,
            index.compiled_pulse_runtime_summaries,
        ):
            compiled_payload = authority.read_blob(
                compiled_ref,
                max_bytes=compiled_ref.size,
            )
            artifact = decode_compiled_pulse_artifact(compiled_payload)
            if compiled_ref.digest != artifact.fingerprint:
                raise ValueError("compiled pulse blob identity differs from fingerprint")
            if compiled_pulse_runtime_summary(
                artifact,
                encoded_size=compiled_ref.size,
            ) != expected_summary:
                raise ValueError("compiled pulse runtime summary differs from lineage")
            compiled.append(artifact)
        program = _decode_program(
            authority.read_blob(
                index.pulse_program_blob,
                max_bytes=index.pulse_program_blob.size,
            )
        )
        if index.pulse_program_blob.digest != program.fingerprint:
            raise ValueError("PulseScanProgram blob identity differs from fingerprint")
        execution = pulse_scan_execution_from_tree(
            index.execution_tree,
            program,
            tuple(compiled),
        )
        stored = _stored_scan_from_index(index, execution)
        _require_scan_facts(
            run_id=stored.provenance.trace_binding.run_id,
            execution=stored.execution,
            source_ref=stored.source_dataset_ref,
            source_schema=stored.source_dataset_schema,
            output_contract=stored.output_contract,
            output_ref=stored.output_dataset_ref,
            provenance=stored.provenance,
        )
        return stored, metadata_size

    @staticmethod
    def _inspection_from_index(
        reference: ScanArtifactRef,
        index: _StoredScanIndex,
        intent: CommitIntent,
        *,
        metadata_size: int,
        inspection_peak: int,
    ) -> ScanArtifactInspection:
        if index.provenance.trace_binding.run_id != intent.run_id:
            raise ValueError("scan index differs from its FINAL commit intent")
        data_peak = (
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
            index.compiled_pulse_runtime_summaries,
            inspection_peak,
            inspection_peak,
            max(inspection_peak, data_peak),
        )

    def inspect_final(
        self,
        reference: ScanArtifactRef,
        *,
        memory_limit_bytes: int | None = None,
    ) -> ScanArtifactInspection:
        """Read FINAL schema/resource facts without pulse IR or program decode."""

        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            intent = self._require_final_commit(reference)
            index, metadata_size, inspection_peak = self._load_index(
                reference,
                memory_limit_bytes=memory_limit_bytes,
            )
            return self._inspection_from_index(
                reference,
                index,
                intent,
                metadata_size=metadata_size,
                inspection_peak=inspection_peak,
            )

    def admit(self, reference: ScanArtifactRef) -> ScanArtifact:
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            intent = self._require_final_commit(reference)
            stored, _metadata_size = self._load_stored(reference)
            return self._artifact_from_stored(
                reference,
                stored,
                intent,
            )

    @staticmethod
    def _artifact_from_stored(
        reference: ScanArtifactRef,
        stored: _StoredScan,
        intent: CommitIntent,
    ) -> ScanArtifact:
        if stored.provenance.trace_binding.run_id != intent.run_id:
            raise ValueError("scan artifact differs from its FINAL commit intent")
        return ScanArtifact(
            reference,
            stored.execution,
            stored.source_dataset_ref,
            stored.source_dataset_schema,
            stored.output_contract,
            stored.output_dataset_ref,
            stored.provenance,
        )

    def materialize(
        self,
        reference: ScanArtifactRef,
        *,
        memory_limit_bytes: int,
        abort_check: Callable[[], None] | None = None,
    ) -> MaterializedScanData:
        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        if abort_check is not None:
            abort_check()
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            intent = self._require_final_commit(reference)
            index, metadata_size, inspection_peak = self._load_index(
                reference,
                memory_limit_bytes=limit,
            )
            if abort_check is not None:
                abort_check()
            inspection = self._inspection_from_index(
                reference,
                index,
                intent,
                metadata_size=metadata_size,
                inspection_peak=inspection_peak,
            )
            peak = inspection.materialization_peak_upper_bound_bytes
            if peak > limit:
                raise MemoryError(
                    f"scan materialization peak {peak} exceeds limit {limit}"
                )
            authority = self._content_authority()
            if abort_check is not None:
                abort_check()
            values_payload = authority.read_blob(
                index.values_blob,
                max_bytes=index.values_blob.size,
            )
            if abort_check is not None:
                abort_check()
            schema = inspection.output_schema
            values = np.frombuffer(
                values_payload,
                dtype=schema.cell_schema.dtype,
            ).reshape(schema.physical_shape)
            validity_payload = authority.read_blob(
                index.validity_blob,
                max_bytes=index.validity_blob.size,
            )
            if abort_check is not None:
                abort_check()
            validity = _decode_validity(
                validity_payload,
                max_array_bytes=index.validity_blob.size,
            )
            if abort_check is not None:
                abort_check()
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
            lineage.pulse_program_blob,
            lineage.compiled_pulse_blobs,
            lineage.compiled_pulse_runtime_summaries,
            prepared.execution,
            prepared.source_dataset_ref,
            prepared.source_dataset_schema,
            prepared.output_contract,
            prepared.output_snapshot.ref,
            prepared.provenance,
            values_ref,
            validity_ref,
        )
        admission = prepared.api_metadata_admission
        if isinstance(prepared.execution, ApiSegmentedScanExecution):
            if not isinstance(admission, _ApiFinalMetadataAdmission):
                raise RuntimeError("API FINAL lost its pre-FIRE metadata admission")
            if admission.issuer_token is not self._api_metadata_admission_token:
                raise PermissionError("API metadata admission belongs to another repository")
            if admission.policy_max_bytes != policy.max_metadata_blob_bytes:
                raise RuntimeError("API metadata repository policy changed after admission")
            _require_api_metadata_admission_facts(
                admission,
                execution=prepared.execution,
                source_ref=prepared.source_dataset_ref,
                source_schema=prepared.source_dataset_schema,
                output_contract=prepared.output_contract,
                provenance=prepared.provenance,
                staged_lineage=lineage,
            )
            metadata_limits = CanonicalDecodeLimits(
                max_depth=_METADATA_LIMITS.max_depth,
                max_nodes=admission.admitted_structure_limit,
                max_container_entries=admission.admitted_structure_limit,
                max_arrays=0,
                max_total_array_bytes=0,
            )
            try:
                metadata = encode(_metadata_tree(stored), limits=metadata_limits)
            except CanonicalEncodingError as exc:
                raise RuntimeError(
                    "API metadata pre-FIRE admission invariant was violated"
                ) from exc
            if len(metadata) > admission.admitted_max_bytes:
                raise RuntimeError(
                    "API metadata pre-FIRE admission byte invariant was violated"
                )
            if len(metadata) > policy.max_metadata_blob_bytes:
                raise RuntimeError(
                    "API metadata exceeded repository policy after pre-FIRE admission"
                )
        else:
            if admission is not None:
                raise RuntimeError("autonomous FINAL unexpectedly carries API admission")
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
        run_id = context.authorize_commit_preparation()
        if prepared.run_id != run_id:
            raise ValueError("prepared scan belongs to another Run")
        with self._root_lease.borrow() as staging_borrow:
            staging_borrow.require_active()
            reference, payload = self._stage_result(prepared)
            if context.authorize_commit_preparation() != run_id:
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
        stored, _metadata_size = self._load_stored(
            reference,
            manifest_payload=payload,
        )
        if stored.provenance.trace_binding.run_id != intent.run_id:
            raise ValueError("visible scan differs from pending commit intent")
        for blob in (
            stored.pulse_program_blob,
            *stored.compiled_pulse_blobs,
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
