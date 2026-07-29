"""Canonical FINAL authority for source-neutral PulseScan datasets.

The repository persists the collected ``(R, P, *data_shape)`` Dataset, the
sequencer program/terminal receipts, and compact lineage for the external
signal events.  It has no Camera, Processor, selector, or Workbench knowledge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import numpy as np

from zlc_data import (
    BlockId,
    DataBlock,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
)
from zlc_data.codec import (
    dataset_revision_ref_from_tree,
    dataset_revision_ref_to_tree,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
    validity_from_tree,
    validity_to_tree,
)
from zlc_neutral_atom.runtime.commit import PreparedArtifactCommit
from zlc_neutral_atom.runtime.dataset import (
    DatasetSealProvenance,
    dataset_seal_provenance_from_tree,
    dataset_seal_provenance_to_tree,
)
from zlc_neutral_atom.runtime.run import PostSafetyContext
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
    materialize_scan_sweeps,
)
from zlc_storage import (
    CanonicalArrayEvent,
    ContentAddressedStore,
    ContentCorruptionError,
    ContentRef,
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
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseParameterScanProgram,
    pulse_parameter_scan_program_from_tree,
    pulse_parameter_scan_program_to_tree,
)

from .contracts import (
    ScanOutputContract,
    bind_scan_output_contract,
    scan_output_contract_from_tree,
    scan_output_contract_to_tree,
)
from .lineage import (
    ApiSegmentedScanExecution,
    AutonomousScanExecution,
    PulseScanExecution,
    execution_compiled_artifacts,
    pulse_scan_execution_from_tree,
    pulse_scan_execution_to_tree,
)
from .reference import SCAN_ARTIFACT_NAMESPACE, ScanArtifactRef


SCAN_ARTIFACT_SCHEMA = "zlc_neutral_atom.logic_nodes.pulse_scan.storage"
SCAN_MANIFEST_SCHEMA = "zlc_neutral_atom.logic_nodes.pulse_scan.manifest"
_MANIFEST_FIELDS = frozenset({"schema", "repository_id", "metadata_blob"})
_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "pulse_program_blob",
        "compiled_pulse_blobs",
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
@dataclass(frozen=True, slots=True)
class _StoredScan:
    pulse_program_blob: ContentRef
    compiled_pulse_blobs: tuple[ContentRef, ...]
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
        if not blobs or any(not isinstance(item, ContentRef) for item in blobs):
            raise TypeError("compiled_pulse_blobs must contain ContentRef values")
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
    execution_tree: object
    source_dataset_ref: DatasetRevisionRef
    source_dataset_schema: DatasetSchema
    output_contract: ScanOutputContract
    output_dataset_ref: DatasetRevisionRef
    provenance: DatasetSealProvenance
    values_blob: ContentRef
    validity_blob: ContentRef


@dataclass(frozen=True, slots=True)
class _StagedScanLineage:
    """Pre-FIRE immutable scan-program CAS references."""

    pulse_program_blob: ContentRef
    compiled_pulse_blobs: tuple[ContentRef, ...]


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


@dataclass(frozen=True, eq=False, slots=True)
class MaterializedScanData:
    artifact_ref: ScanArtifactRef
    program_fingerprint: str
    source_dataset_ref: DatasetRevisionRef
    output_contract: ScanOutputContract
    snapshot: OwnedSnapshot
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, ScanArtifactRef):
            raise TypeError("artifact_ref must be ScanArtifactRef")
        sha256_text(self.program_fingerprint, "program_fingerprint")
        if not isinstance(self.source_dataset_ref, DatasetRevisionRef):
            raise TypeError("source_dataset_ref must be DatasetRevisionRef")
        if not isinstance(self.output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if self.snapshot.block.schema != self.output_contract.output_dataset_schema:
            raise ValueError("materialized scan Dataset differs from its output contract")

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
        "execution",
        "source_snapshot",
        "output_contract",
        "output_snapshot",
        "provenance",
        "staged_lineage",
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
    program: PulseParameterScanProgram,
    source_ref: DatasetRevisionRef,
    output_contract: ScanOutputContract,
) -> DatasetRevisionRef:
    if not isinstance(
        program,
        (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
    ):
        raise TypeError("program must be a PulseParameterScanProgram")
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
            "owner": "zlc_neutral_atom.logic_nodes.pulse_scan.output",
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
    sweep_count = program.sweep_count
    if source_schema.repeat_axis.size != sweep_count:
        raise ValueError("source repeat axis differs from frozen scan sweeps")
    if source_schema.point_table != point_table:
        raise ValueError("source PointTable differs from the frozen pulse rows")
    if not isinstance(
        execution,
        (AutonomousScanExecution, ApiSegmentedScanExecution),
    ):
        raise TypeError("execution must be a PulseScanExecution")
    expected_events = sweep_count * point_table.row_count
    event_count = provenance.end_sequence - provenance.start_sequence
    if event_count != expected_events:
        raise ValueError("collected Dataset count differs from logical R by P")
    if execution.source.count != expected_events:
        raise ValueError("external signal lineage count differs from logical R by P")
    projection = execution.source.projection_authority
    if source_schema.cell_schema != projection.output_value_schema:
        raise ValueError(
            "scan source schema differs from committed signal projection output"
        )
    if output_contract.committed_transform is not None:
        raise ValueError(
            "ScanOutputContract cannot duplicate signal projection authority"
        )
    if provenance.trace_binding.run_id != run_id:
        raise ValueError("collected Dataset provenance belongs to another Run")
    resolved = bind_scan_output_contract(
        source_schema,
        point_table,
        None,
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
    program: PulseParameterScanProgram,
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
        expected = materialize_scan_sweeps(
            program.execution_document,
            program.sweep_count,
        )
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
        raise TypeError("program must be a PulseParameterScanProgram")
    return pulses


def _reject_arrays(events) -> None:
    if any(isinstance(event, CanonicalArrayEvent) for event in events):
        raise ValueError("scan metadata cannot embed ndarray payloads")


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
) -> bytes:
    return encode(_metadata_tree(value))


def _decode_metadata_index(payload: bytes) -> _StoredScanIndex:
    data = exact_mapping(
        decode(
            payload,
            admit_structure=_reject_arrays,
        ),
        _ARTIFACT_FIELDS,
        SCAN_ARTIFACT_SCHEMA,
    )
    blob_trees = data["compiled_pulse_blobs"]
    if not isinstance(blob_trees, list):
        raise TypeError("compiled pulse blobs must be a list")
    blobs = tuple(content_ref_from_tree(item) for item in blob_trees)
    if not blobs:
        raise ValueError("compiled pulse blobs must not be empty")
    value = _StoredScanIndex(
        content_ref_from_tree(data["pulse_program_blob"]),
        blobs,
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
        decode(payload, admit_structure=_reject_arrays),
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


def _encode_program(program: PulseParameterScanProgram) -> bytes:
    return encode(pulse_parameter_scan_program_to_tree(program))


def _decode_program(payload: bytes) -> PulseParameterScanProgram:
    program = pulse_parameter_scan_program_from_tree(
        decode(payload, admit_structure=_reject_arrays)
    )
    if _encode_program(program) != payload:
        raise ValueError(
            "PulseParameterScanProgram blob is typed but non-canonical"
        )
    return program


def _encode_validity(validity: object) -> bytes:
    return encode(validity_to_tree(validity))


def _decode_validity(payload: bytes):
    validity = validity_from_tree(decode(payload))
    if _encode_validity(validity) != payload:
        raise ValueError("scan validity blob is typed but non-canonical")
    return validity


def _values_payload(snapshot: OwnedSnapshot) -> memoryview:
    values = snapshot.block.values
    if not values.flags.c_contiguous:
        raise ValueError("canonical scan output values must be C-contiguous")
    return memoryview(values).cast("B")


class ScanRepository:
    """Current-only CAS authority for canonical FINAL scan datasets."""

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-scan",
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
            self._root_lease.close()
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

    def _stage_static_lineage(
        self,
        program: PulseParameterScanProgram,
        compiled_pulses: tuple[CompiledPulseArtifact, ...],
    ) -> _StagedScanLineage:
        """Persist immutable lineage before delegating to hardware preflight."""

        pulses = _require_program_artifacts(program, compiled_pulses)
        program_payload = _encode_program(program)
        compiled_payloads = tuple(
            encode_compiled_pulse_artifact(item) for item in pulses
        )
        authority = self._content_authority()
        program_ref = authority.put_blob(program_payload)
        compiled_refs = tuple(authority.put_blob(item) for item in compiled_payloads)
        if program_ref.digest != program.fingerprint or tuple(
            item.digest for item in compiled_refs
        ) != tuple(item.fingerprint for item in pulses):
            raise RuntimeError("scan lineage CAS identity differs from its owner")
        return _StagedScanLineage(program_ref, compiled_refs)

    def _load_index(
        self,
        reference: ScanArtifactRef,
        *,
        manifest_payload: bytes | None = None,
    ) -> _StoredScanIndex:
        authority = self._content_authority()
        if manifest_payload is None:
            payload = authority.read_manifest(
                SCAN_ARTIFACT_NAMESPACE,
                reference.manifest_digest,
            )
        else:
            payload = manifest_payload
        repository_id, metadata_ref = _decode_manifest(payload)
        if repository_id != self.repository_id:
            raise ValueError("scan manifest belongs to another repository")
        metadata = authority.read_blob(metadata_ref)
        index = _decode_metadata_index(metadata)
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
        return index

    def _load_stored(
        self,
        reference: ScanArtifactRef,
        *,
        manifest_payload: bytes | None = None,
    ) -> _StoredScan:
        index = self._load_index(
            reference,
            manifest_payload=manifest_payload,
        )
        authority = self._content_authority()
        compiled: list[CompiledPulseArtifact] = []
        for compiled_ref in index.compiled_pulse_blobs:
            compiled_payload = authority.read_blob(compiled_ref)
            artifact = decode_compiled_pulse_artifact(compiled_payload)
            if compiled_ref.digest != artifact.fingerprint:
                raise ValueError("compiled pulse blob identity differs from fingerprint")
            compiled.append(artifact)
        program = _decode_program(
            authority.read_blob(index.pulse_program_blob)
        )
        if index.pulse_program_blob.digest != program.fingerprint:
            raise ValueError(
                "PulseParameterScanProgram blob identity differs from fingerprint"
            )
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
        for blob in (stored.values_blob, stored.validity_blob):
            authority.verify_blob(blob)
        return stored

    def admit(self, reference: ScanArtifactRef) -> ScanArtifact:
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            self._validate_reference(reference)
            stored = self._load_stored(reference)
            return self._artifact_from_stored(
                reference,
                stored,
            )

    @staticmethod
    def _artifact_from_stored(
        reference: ScanArtifactRef,
        stored: _StoredScan,
    ) -> ScanArtifact:
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
        abort_check: Callable[[], None] | None = None,
    ) -> MaterializedScanData:
        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        if abort_check is not None:
            abort_check()
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            self._validate_reference(reference)
            index = self._load_index(reference)
            if abort_check is not None:
                abort_check()
            authority = self._content_authority()
            if abort_check is not None:
                abort_check()
            values_payload = authority.read_blob(index.values_blob)
            if abort_check is not None:
                abort_check()
            schema = index.output_contract.output_dataset_schema
            values = np.frombuffer(
                values_payload,
                dtype=schema.cell_schema.dtype,
            ).reshape(schema.physical_shape)
            validity_payload = authority.read_blob(index.validity_blob)
            if abort_check is not None:
                abort_check()
            validity = _decode_validity(validity_payload)
            if abort_check is not None:
                abort_check()
            block = DataBlock(
                index.output_dataset_ref.block_id,
                index.output_dataset_ref.revision,
                values,
                validity,
                schema,
            )
            snapshot = OwnedSnapshot(index.output_dataset_ref, block)
            return MaterializedScanData(
                reference,
                index.pulse_program_blob.digest,
                index.source_dataset_ref,
                index.output_contract,
                snapshot,
            )

    def project_dataset_source(
        self,
        reference: ScanArtifactRef,
        *,
        materialize: bool,
        abort_check: Callable[[], None] | None = None,
    ) -> ArtifactDatasetSource:
        """Project the FINAL scan output as one exact generic Dataset source."""

        if type(materialize) is not bool:
            raise TypeError("materialize must be bool")
        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        if materialize:
            snapshot = self.materialize(
                reference,
                abort_check=abort_check,
            ).snapshot
            return ArtifactDatasetSource(
                snapshot.block.schema,
                snapshot.ref,
                snapshot,
            )
        if abort_check is not None:
            abort_check()
        artifact = self.admit(reference)
        if abort_check is not None:
            abort_check()
        return ArtifactDatasetSource(
            artifact.output_schema,
            artifact.output_dataset_ref,
        )

    def has(self, reference: ScanArtifactRef) -> bool:
        with self._root_lease.borrow() as borrow:
            borrow.require_active()
            self._validate_reference(reference)
            authority = self._content_authority()
            try:
                payload = authority.read_manifest(
                    SCAN_ARTIFACT_NAMESPACE,
                    reference.manifest_digest,
                )
            except FileNotFoundError:
                return False
            self._load_stored(reference, manifest_payload=payload)
            return True

    def _stage_result(
        self,
        prepared: _PreparedScanDataset,
    ) -> tuple[ScanArtifactRef, bytes]:
        if type(prepared) is not _PreparedScanDataset:
            raise TypeError("prepared must be scan application output")
        authority = self._content_authority()
        lineage = prepared.staged_lineage
        validity_payload = _encode_validity(
            prepared.output_snapshot.block.validity
        )
        values_payload = _values_payload(prepared.output_snapshot)
        values_ref = authority.identify_blob(values_payload)
        validity_ref = authority.identify_blob(validity_payload)
        stored = _StoredScan(
            lineage.pulse_program_blob,
            lineage.compiled_pulse_blobs,
            prepared.execution,
            prepared.source_dataset_ref,
            prepared.source_dataset_schema,
            prepared.output_contract,
            prepared.output_snapshot.ref,
            prepared.provenance,
            values_ref,
            validity_ref,
        )
        metadata = _encode_metadata(stored)
        for expected, actual in (
            (values_ref, authority.put_blob(values_payload)),
            (validity_ref, authority.put_blob(validity_payload)),
        ):
            if actual != expected:
                raise RuntimeError("content store changed a precomputed scan blob identity")
        metadata_ref = authority.put_blob(metadata)
        manifest = _manifest_payload(self.repository_id, metadata_ref)
        reference = ScanArtifactRef(self.repository_id, sha256_digest(manifest))
        return reference, manifest

    def final_commit(
        self,
        context: PostSafetyContext,
        prepared: _PreparedScanDataset,
    ) -> PreparedArtifactCommit[ScanArtifactRef]:
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

            def publish(manifest_payload: bytes) -> None:
                stored = self._content_authority().publish_manifest(
                    SCAN_ARTIFACT_NAMESPACE,
                    manifest_payload,
                    expected_digest=reference.manifest_digest,
                )
                if stored.content.digest != reference.manifest_digest:
                    raise RuntimeError("published scan manifest digest changed")

            def inspect(manifest_payload: bytes) -> bool | None:
                authority = self._content_authority()
                try:
                    durable_payload = authority.confirm_manifest_durable(
                        SCAN_ARTIFACT_NAMESPACE,
                        reference.manifest_digest,
                    )
                except FileNotFoundError:
                    return False
                except OSError:
                    return None
                if durable_payload != manifest_payload:
                    raise ContentCorruptionError(
                        "visible scan manifest differs from prepared payload"
                    )
                try:
                    stored_scan = self._load_stored(
                        reference,
                        manifest_payload=durable_payload,
                    )
                except FileNotFoundError as error:
                    raise ContentCorruptionError(
                        "visible scan manifest references missing content"
                    ) from error
                except OSError:
                    return None
                if stored_scan.provenance.trace_binding.run_id != run_id:
                    raise ValueError("visible scan provenance belongs to another Run")
                return True

            commit_borrow = self._root_lease.borrow()
            try:
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


__all__ = [
    "MaterializedScanData",
    "SCAN_ARTIFACT_SCHEMA",
    "ScanArtifact",
    "ScanRepository",
]
