"""Direct, record-last persistence for FINAL PulseScan datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from zlc_data import DataBlock, DatasetRevisionRef, DatasetSchema, OwnedSnapshot
from zlc_data.codec import (
    dataset_revision_ref_from_tree,
    dataset_revision_ref_to_tree,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
    validity_from_tree,
    validity_to_tree,
)
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.runtime.dataset import (
    DatasetSealProvenance,
    dataset_seal_provenance_from_tree,
    dataset_seal_provenance_to_tree,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
    materialize_scan_sweeps,
)
from zlc_storage import canonical_text, decode, encode, exact_mapping
from zlc_storage.durability import (
    atomic_write_bytes,
    atomic_write_file,
    durable_mkdir,
    durable_makedirs,
)
from zlc_storage.paths import resolve_under

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
)
from .lineage import (
    ApiSegmentedScanExecution,
    AutonomousScanExecution,
    PulseScanExecution,
    execution_compiled_artifacts,
    pulse_scan_execution_from_tree,
    pulse_scan_execution_to_tree,
)
from .reference import ScanArtifactRef


SCAN_ARTIFACT_SCHEMA = "zlc_neutral_atom.logic_nodes.pulse_scan.record"
_RECORD_FIELDS = {
    "schema",
    "run_id",
    "pulse_program_file",
    "compiled_pulse_files",
    "execution",
    "dataset_ref",
    "dataset_schema",
    "dataset_provenance",
    "values_file",
    "validity_file",
}


@dataclass(frozen=True, slots=True)
class ScanArtifact:
    """Metadata for one complete, directly persisted FINAL scan Dataset."""

    ref: ScanArtifactRef
    run_id: str
    execution: PulseScanExecution
    dataset_ref: DatasetRevisionRef
    dataset_schema: DatasetSchema
    provenance: DatasetSealProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ScanArtifactRef):
            raise TypeError("ref must be ScanArtifactRef")
        canonical_text(self.run_id, "run_id")
        _require_scan_facts(
            run_id=self.run_id,
            execution=self.execution,
            dataset_ref=self.dataset_ref,
            dataset_schema=self.dataset_schema,
            output_contract=self.output_contract,
            provenance=self.provenance,
        )

    @property
    def output_schema(self) -> DatasetSchema:
        return self.dataset_schema

    @property
    def output_contract(self) -> ScanOutputContract:
        """Derive the identity output contract instead of persisting it twice."""

        return bind_scan_output_contract(
            self.dataset_schema,
            self.execution.program.point_table,
            None,
        )


@dataclass(frozen=True, eq=False, slots=True)
class MaterializedScanData:
    artifact: ScanArtifact
    snapshot: OwnedSnapshot
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ScanArtifact):
            raise TypeError("artifact must be ScanArtifact")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if self.snapshot.ref != self.artifact.dataset_ref:
            raise ValueError("materialized scan has another Dataset revision")
        if self.snapshot.block.schema != self.artifact.dataset_schema:
            raise ValueError("materialized scan has another Dataset schema")

    @property
    def artifact_ref(self) -> ScanArtifactRef:
        return self.artifact.ref

    @property
    def schema(self) -> DatasetSchema:
        return self.snapshot.block.schema

    @property
    def values(self) -> np.ndarray:
        return self.snapshot.block.values

    @property
    def validity(self):
        return self.snapshot.block.validity


def _require_program_artifacts(
    program: PulseParameterScanProgram,
    compiled_pulses: tuple[CompiledPulseArtifact, ...],
) -> tuple[CompiledPulseArtifact, ...]:
    pulses = tuple(compiled_pulses)
    if any(not isinstance(item, CompiledPulseArtifact) for item in pulses):
        raise TypeError("compiled_pulses must contain CompiledPulseArtifact values")
    if isinstance(program, AutonomousScanSlotProgram):
        if len(pulses) != 1:
            raise ValueError("autonomous scan requires exactly one compiled artifact")
        artifact = pulses[0]
        expected = materialize_scan_sweeps(
            program.execution_document,
            program.sweep_count,
        )
        if (
            artifact.execution_form is not PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
            or artifact.source_document_digest != expected.fingerprint
        ):
            raise ValueError("autonomous compiled pulse differs from its program")
    elif isinstance(program, ApiSlotSegmentedProgram):
        documents = program.resolved_point_documents
        if len(pulses) != len(documents) or any(
            artifact.execution_form is not PulseExecutionForm.STATIC_ONCE
            or artifact.source_document_digest != document.fingerprint
            for artifact, document in zip(pulses, documents)
        ):
            raise ValueError("API compiled pulses differ from resolved point documents")
    else:
        raise TypeError("program must be a PulseParameterScanProgram")
    return pulses


def _require_scan_facts(
    *,
    run_id: str,
    execution: PulseScanExecution,
    dataset_ref: DatasetRevisionRef,
    dataset_schema: DatasetSchema,
    output_contract: ScanOutputContract,
    provenance: DatasetSealProvenance,
) -> None:
    canonical_text(run_id, "run_id")
    if not isinstance(execution, (AutonomousScanExecution, ApiSegmentedScanExecution)):
        raise TypeError("execution must be a PulseScanExecution")
    if not isinstance(dataset_ref, DatasetRevisionRef):
        raise TypeError("dataset_ref must be DatasetRevisionRef")
    if not isinstance(dataset_schema, DatasetSchema):
        raise TypeError("dataset_schema must be DatasetSchema")
    if not isinstance(output_contract, ScanOutputContract):
        raise TypeError("output_contract must be ScanOutputContract")
    if not isinstance(provenance, DatasetSealProvenance):
        raise TypeError("provenance must be DatasetSealProvenance")
    if dataset_ref.schema_fingerprint != dataset_schema.fingerprint:
        raise ValueError("scan Dataset ref differs from its schema")
    if dataset_ref.stream_generation != provenance.generation:
        raise ValueError("scan Dataset generation differs from its provenance")
    if dataset_schema != output_contract.output_dataset_schema:
        raise ValueError("scan Dataset schema differs from its output contract")
    if output_contract.committed_transform is not None:
        raise ValueError("PulseScan output cannot duplicate signal transform authority")
    program = execution.program
    if dataset_schema.repeat_axis.size != program.sweep_count:
        raise ValueError("scan repeat axis differs from frozen scan sweeps")
    if dataset_schema.point_table != program.point_table:
        raise ValueError("scan PointTable differs from frozen pulse rows")
    expected_events = program.sweep_count * program.point_table.row_count
    if provenance.end_sequence - provenance.start_sequence != expected_events:
        raise ValueError("scan Dataset count differs from R by P")
    if execution.source.count != expected_events:
        raise ValueError("external signal lineage count differs from R by P")
    if dataset_schema.cell_schema != execution.source.projection_authority.output_value_schema:
        raise ValueError("scan Dataset differs from committed signal projection")
    if bind_scan_output_contract(dataset_schema, program.point_table, None) != output_contract:
        raise ValueError("ScanOutputContract differs from the collected Dataset")
    _require_program_artifacts(program, execution_compiled_artifacts(execution))


def _encode_program(program: PulseParameterScanProgram) -> bytes:
    return encode(pulse_parameter_scan_program_to_tree(program))


def _decode_program(payload: bytes) -> PulseParameterScanProgram:
    program = pulse_parameter_scan_program_from_tree(decode(payload))
    if _encode_program(program) != payload:
        raise ValueError("PulseParameterScanProgram is not canonical current format")
    return program


def _encode_validity(validity: object) -> bytes:
    return encode(validity_to_tree(validity))


def _decode_validity(payload: bytes):
    validity = validity_from_tree(decode(payload))
    if _encode_validity(validity) != payload:
        raise ValueError("scan validity is not canonical current format")
    return validity


def _run_directory_name(run_id: str) -> str:
    canonical = canonical_text(run_id, "run_id")
    identity = "".join(character for character in canonical if character.isalnum())
    if not identity:
        raise ValueError("run_id has no filesystem-safe characters")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{identity}"


def write_scan_artifact(
    scans_root: str | Path,
    *,
    run_id: str,
    execution: PulseScanExecution,
    snapshot: OwnedSnapshot,
    output_contract: ScanOutputContract,
    provenance: DatasetSealProvenance,
) -> ScanArtifactRef:
    """Write scan arrays/lineage first and publish ``scan.json`` last."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    _require_scan_facts(
        run_id=run_id,
        execution=execution,
        dataset_ref=snapshot.ref,
        dataset_schema=snapshot.block.schema,
        output_contract=output_contract,
        provenance=provenance,
    )
    root = Path(scans_root).expanduser().resolve()
    durable_makedirs(root)
    run_directory = resolve_under(root, _run_directory_name(run_id))
    durable_mkdir(run_directory)

    program_file = "pulse-program.zlc"
    compiled = execution_compiled_artifacts(execution)
    compiled_files = tuple(
        f"compiled-pulse-{index:03d}.zlc" for index in range(len(compiled))
    )
    values_file = "values.npy"
    validity_file = "validity.zlc"
    atomic_write_bytes(run_directory / program_file, _encode_program(execution.program))
    for filename, artifact in zip(compiled_files, compiled):
        atomic_write_bytes(
            run_directory / filename,
            encode_compiled_pulse_artifact(artifact),
        )
    atomic_write_file(
        run_directory / values_file,
        lambda stream: np.save(stream, snapshot.block.values, allow_pickle=False),
    )
    atomic_write_bytes(
        run_directory / validity_file,
        _encode_validity(snapshot.block.validity),
    )
    record = {
        "schema": SCAN_ARTIFACT_SCHEMA,
        "run_id": run_id,
        "pulse_program_file": program_file,
        "compiled_pulse_files": list(compiled_files),
        "execution": pulse_scan_execution_to_tree(execution),
        "dataset_ref": dataset_revision_ref_to_tree(snapshot.ref),
        "dataset_schema": dataset_schema_to_tree(snapshot.block.schema),
        "dataset_provenance": dataset_seal_provenance_to_tree(provenance),
        "values_file": values_file,
        "validity_file": validity_file,
    }
    payload = encode(record)
    if encode(exact_mapping(decode(payload), _RECORD_FIELDS, SCAN_ARTIFACT_SCHEMA)) != payload:
        raise ValueError("scan record failed its canonical round-trip")
    record_path = run_directory / "scan.json"
    atomic_write_bytes(record_path, payload)
    return ScanArtifactRef(record_path.relative_to(root).as_posix())


def _read_scan(
    scans_root: str | Path,
    reference: ScanArtifactRef,
) -> tuple[ScanArtifact, Path, dict[str, object]]:
    if not isinstance(reference, ScanArtifactRef):
        raise TypeError("reference must be ScanArtifactRef")
    root = Path(scans_root).expanduser().resolve()
    record_path = resolve_under(root, reference.record_path)
    payload = record_path.read_bytes()
    record = exact_mapping(decode(payload), _RECORD_FIELDS, SCAN_ARTIFACT_SCHEMA)
    if encode(record) != payload:
        raise ValueError("scan record is not canonical current format")
    run_directory = record_path.parent
    program = _decode_program(
        resolve_under(run_directory, canonical_text(record["pulse_program_file"], "program file")).read_bytes()
    )
    compiled_trees = record["compiled_pulse_files"]
    if not isinstance(compiled_trees, list) or not compiled_trees:
        raise ValueError("compiled_pulse_files must be a non-empty list")
    compiled = tuple(
        decode_compiled_pulse_artifact(
            resolve_under(run_directory, canonical_text(name, "compiled pulse file")).read_bytes()
        )
        for name in compiled_trees
    )
    _require_program_artifacts(program, compiled)
    execution = pulse_scan_execution_from_tree(record["execution"], program, compiled)
    artifact = ScanArtifact(
        reference,
        canonical_text(record["run_id"], "run_id"),
        execution,
        dataset_revision_ref_from_tree(record["dataset_ref"]),
        dataset_schema_from_tree(record["dataset_schema"]),
        dataset_seal_provenance_from_tree(record["dataset_provenance"]),
    )
    return artifact, run_directory, record


def load_scan_artifact(
    scans_root: str | Path,
    reference: ScanArtifactRef,
) -> ScanArtifact:
    return _read_scan(scans_root, reference)[0]


def materialize_scan_data(
    scans_root: str | Path,
    reference: ScanArtifactRef,
    *,
    abort_check: Callable[[], None] | None = None,
) -> MaterializedScanData:
    if abort_check is not None and not callable(abort_check):
        raise TypeError("abort_check must be callable or None")
    if abort_check is not None:
        abort_check()
    artifact, run_directory, record = _read_scan(scans_root, reference)
    if abort_check is not None:
        abort_check()
    values_path = resolve_under(
        run_directory,
        canonical_text(record["values_file"], "values file"),
    )
    values = np.load(values_path, allow_pickle=False)
    if abort_check is not None:
        abort_check()
    validity = _decode_validity(
        resolve_under(
            run_directory,
            canonical_text(record["validity_file"], "validity file"),
        ).read_bytes()
    )
    if values.shape != artifact.dataset_schema.physical_shape:
        raise ValueError("scan values shape differs from the persisted schema")
    if values.dtype != artifact.dataset_schema.cell_schema.dtype:
        raise ValueError("scan values dtype differs from the persisted schema")
    block = DataBlock(
        artifact.dataset_ref.block_id,
        artifact.dataset_ref.revision,
        values,
        validity,
        artifact.dataset_schema,
    )
    return MaterializedScanData(
        artifact,
        OwnedSnapshot(artifact.dataset_ref, block),
    )


def project_scan_dataset(
    scans_root: str | Path,
    reference: ScanArtifactRef,
    *,
    materialize: bool,
    abort_check: Callable[[], None] | None = None,
) -> ArtifactDatasetSource:
    if type(materialize) is not bool:
        raise TypeError("materialize must be bool")
    if materialize:
        snapshot = materialize_scan_data(
            scans_root,
            reference,
            abort_check=abort_check,
        ).snapshot
        return ArtifactDatasetSource(snapshot.block.schema, snapshot.ref, snapshot)
    if abort_check is not None:
        abort_check()
    artifact = load_scan_artifact(scans_root, reference)
    return ArtifactDatasetSource(artifact.dataset_schema, artifact.dataset_ref)


__all__ = [
    "MaterializedScanData",
    "SCAN_ARTIFACT_SCHEMA",
    "ScanArtifact",
    "load_scan_artifact",
    "materialize_scan_data",
    "project_scan_dataset",
    "write_scan_artifact",
]
