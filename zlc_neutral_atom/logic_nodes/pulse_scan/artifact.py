"""Direct, record-last persistence for FINAL PulseScan datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from zlc_data import (
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    StreamGenerationId,
)
from zlc_data.codec import (
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
    decode_compiled_pulse_artifact,
    encode_compiled_pulse_artifact,
)
from zlc_storage import canonical_text
from zlc_storage.durability import (
    atomic_write_bytes,
    atomic_write_file,
    atomic_write_text,
    durable_mkdir,
    durable_makedirs,
)
from zlc_storage.paths import resolve_under

from zlc_neutral_atom.timing.pulse_parameter_scan import (
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
from .reference import SCAN_RECORD_PREFIX, ScanArtifactRef


SCAN_ARTIFACT_SCHEMA = "zlc.pulse_scan"
_RECORD_FIELDS = {
    "schema",
    "run_id",
    "program",
    "compiled_pulse_files",
    "execution",
    "dataset",
    "provenance",
    "values_file",
    "validity",
}

_VALUES_FILE = "values.npy"
_VALIDITY_FILE = "validity.npy"


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


def _require_scan_facts(
    *,
    run_id: str,
    execution: PulseScanExecution,
    dataset_ref: DatasetRevisionRef,
    dataset_schema: DatasetSchema,
    provenance: DatasetSealProvenance,
) -> None:
    canonical_text(run_id, "run_id")
    if not isinstance(execution, (AutonomousScanExecution, ApiSegmentedScanExecution)):
        raise TypeError("execution must be a PulseScanExecution")
    if not isinstance(dataset_ref, DatasetRevisionRef):
        raise TypeError("dataset_ref must be DatasetRevisionRef")
    if not isinstance(dataset_schema, DatasetSchema):
        raise TypeError("dataset_schema must be DatasetSchema")
    if not isinstance(provenance, DatasetSealProvenance):
        raise TypeError("provenance must be DatasetSealProvenance")
    if dataset_ref.schema_fingerprint != dataset_schema.fingerprint:
        raise ValueError("scan Dataset ref differs from its schema")
    if dataset_ref.stream_generation != provenance.generation:
        raise ValueError("scan Dataset generation differs from its provenance")
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


def _write_npy(path: Path, array: np.ndarray) -> None:
    atomic_write_file(
        path,
        lambda stream: np.save(stream, np.asarray(array), allow_pickle=False),
    )


def _write_validity(run_directory: Path, validity: object) -> dict[str, object]:
    record = validity_to_tree(validity)
    mask = record.pop("mask", None)
    filename = None
    if mask is not None:
        filename = _VALIDITY_FILE
        _write_npy(run_directory / filename, mask)
    record["file"] = filename
    return record


def _read_validity(
    run_directory: Path,
    tree: object,
):
    if not isinstance(tree, dict) or "file" not in tree:
        raise ValueError("scan validity record must name its optional mask file")
    record = dict(tree)
    filename = record.pop("file")
    if filename is not None:
        mask = np.load(
            resolve_under(
                run_directory,
                canonical_text(filename, "validity file"),
            ),
            allow_pickle=False,
        )
        if mask.dtype != np.dtype(bool):
            raise TypeError("scan validity mask must have bool dtype")
        record["mask"] = mask
    return validity_from_tree(record)


def _dataset_record(
    reference: DatasetRevisionRef,
    schema: DatasetSchema,
) -> dict[str, object]:
    return {
        "block_id": reference.block_id.value,
        "revision": reference.revision.value,
        "schema": dataset_schema_to_tree(schema),
    }


def _dataset_from_record(
    tree: object,
    generation: StreamGenerationId,
) -> tuple[DatasetRevisionRef, DatasetSchema]:
    if not isinstance(tree, dict) or set(tree) != {
        "block_id",
        "revision",
        "schema",
    }:
        raise ValueError("scan Dataset record has an unknown field set")
    schema = dataset_schema_from_tree(tree["schema"])
    revision = tree["revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("scan Dataset revision must be a nonnegative integer")
    reference = DatasetRevisionRef(
        BlockId(canonical_text(tree["block_id"], "block_id")),
        generation,
        schema.fingerprint,
        DatasetRevision(revision),
    )
    return reference, schema


def _run_directory_name(run_id: str) -> str:
    canonical = canonical_text(run_id, "run_id")
    identity = "".join(character for character in canonical if character.isalnum())
    if not identity:
        raise ValueError("run_id has no filesystem-safe characters")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{identity}"


def write_scan_artifact(
    project_root: str | Path,
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
        provenance=provenance,
    )
    if output_contract != bind_scan_output_contract(
        snapshot.block.schema,
        execution.program.point_table,
        None,
    ):
        raise ValueError("ScanOutputContract differs from the collected Dataset")
    root = Path(project_root).expanduser().resolve()
    scan_root = resolve_under(root, "/".join(SCAN_RECORD_PREFIX))
    durable_makedirs(scan_root)
    run_directory = resolve_under(scan_root, _run_directory_name(run_id))
    durable_mkdir(run_directory)

    compiled = execution_compiled_artifacts(execution)
    compiled_files = tuple(
        f"compiled-pulse-{index:03d}.bin" for index in range(len(compiled))
    )
    for filename, artifact in zip(compiled_files, compiled):
        atomic_write_bytes(
            run_directory / filename,
            encode_compiled_pulse_artifact(artifact),
        )
    _write_npy(run_directory / _VALUES_FILE, snapshot.block.values)
    validity = _write_validity(run_directory, snapshot.block.validity)
    record = {
        "schema": SCAN_ARTIFACT_SCHEMA,
        "run_id": run_id,
        "program": pulse_parameter_scan_program_to_tree(execution.program),
        "compiled_pulse_files": list(compiled_files),
        "execution": pulse_scan_execution_to_tree(execution),
        "dataset": _dataset_record(snapshot.ref, snapshot.block.schema),
        "provenance": dataset_seal_provenance_to_tree(provenance),
        "values_file": _VALUES_FILE,
        "validity": validity,
    }
    record_path = run_directory / "scan.json"
    atomic_write_text(
        record_path,
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return ScanArtifactRef(record_path.relative_to(root).as_posix())


def _read_scan(
    project_root: str | Path,
    reference: ScanArtifactRef,
) -> tuple[ScanArtifact, Path, dict[str, object]]:
    if not isinstance(reference, ScanArtifactRef):
        raise TypeError("reference must be ScanArtifactRef")
    root = Path(project_root).expanduser().resolve()
    record_path = resolve_under(root, reference.record_path)
    with record_path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError("scan record has an unknown field set")
    if record["schema"] != SCAN_ARTIFACT_SCHEMA:
        raise ValueError("scan record schema is not current")
    run_directory = record_path.parent
    program = pulse_parameter_scan_program_from_tree(record["program"])
    compiled_trees = record["compiled_pulse_files"]
    if not isinstance(compiled_trees, list) or not compiled_trees:
        raise ValueError("compiled_pulse_files must be a non-empty list")
    compiled = tuple(
        decode_compiled_pulse_artifact(
            resolve_under(run_directory, canonical_text(name, "compiled pulse file")).read_bytes()
        )
        for name in compiled_trees
    )
    execution = pulse_scan_execution_from_tree(record["execution"], program, compiled)
    provenance = dataset_seal_provenance_from_tree(record["provenance"])
    dataset_ref, dataset_schema = _dataset_from_record(
        record["dataset"],
        provenance.generation,
    )
    artifact = ScanArtifact(
        reference,
        canonical_text(record["run_id"], "run_id"),
        execution,
        dataset_ref,
        dataset_schema,
        provenance,
    )
    return artifact, run_directory, record


def load_scan_artifact(
    project_root: str | Path,
    reference: ScanArtifactRef,
) -> ScanArtifact:
    return _read_scan(project_root, reference)[0]


def materialize_scan_data(
    project_root: str | Path,
    reference: ScanArtifactRef,
    *,
    abort_check: Callable[[], None] | None = None,
) -> MaterializedScanData:
    if abort_check is not None and not callable(abort_check):
        raise TypeError("abort_check must be callable or None")
    if abort_check is not None:
        abort_check()
    artifact, run_directory, record = _read_scan(project_root, reference)
    if abort_check is not None:
        abort_check()
    values_path = resolve_under(
        run_directory,
        canonical_text(record["values_file"], "values file"),
    )
    values = np.load(values_path, allow_pickle=False)
    if abort_check is not None:
        abort_check()
    validity = _read_validity(
        run_directory,
        record["validity"],
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
    project_root: str | Path,
    reference: ScanArtifactRef,
    *,
    materialize: bool,
    abort_check: Callable[[], None] | None = None,
) -> ArtifactDatasetSource:
    if type(materialize) is not bool:
        raise TypeError("materialize must be bool")
    if materialize:
        snapshot = materialize_scan_data(
            project_root,
            reference,
            abort_check=abort_check,
        ).snapshot
        return ArtifactDatasetSource(snapshot.block.schema, snapshot.ref, snapshot)
    if abort_check is not None:
        abort_check()
    artifact = load_scan_artifact(project_root, reference)
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
