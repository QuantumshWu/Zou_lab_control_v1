"""Direct Occupancy artifact I/O and its one flat CPU RunPlan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from zlc_data import AxisId, DataBlock, DatasetComponentValidity, DatasetSchema
from zlc_data.codec import dataset_schema_from_tree, dataset_schema_to_tree
from zlc_neutral_atom.capture.artifact import CaptureArtifact, load_capture_artifact
from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.devices.camera.contract import (
    ReadoutBindingKey,
    readout_binding_key_from_tree,
    readout_binding_key_to_tree,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ResolvedCalibration,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_from_tree,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
    load_calibration_artifact,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
from zlc_storage import canonical_text, decode, encode, exact_mapping, positive_real
from zlc_storage.durability import (
    atomic_write_bytes,
    atomic_write_file,
    durable_makedirs,
    durable_mkdir,
)
from zlc_storage.paths import resolve_under

from .processor import (
    OCCUPANCY_COUNTS_BLOCK_ID,
    OCCUPANCY_OCCUPIED_BLOCK_ID,
    OccupancyArtifact,
    ResolvedOccupancy,
    _CommittedOccupancyBinding,
    _ResolvedCommittedOccupancy,
    _analyze_committed_occupancy_resolved,
    _require_committed_occupancy_context,
    _require_occupancy_output_schemas,
    _resolve_committed_occupancy_structure,
)
from .reference import OccupancyArtifactRef


OCCUPANCY_RECORD_SCHEMA = "zlc_neutral_atom.OccupancyRecord"
_RECORD_FIELDS = {
    "schema",
    "run_id",
    "source_capture_ref",
    "calibration_ref",
    "readout_binding",
    "readout_event_axis_id",
    "model_kind",
    "counts_schema",
    "occupied_schema",
}
_COUNTS_FILE = "counts.npy"
_OCCUPIED_FILE = "occupied.npy"
_VALIDITY_FILE = "validity.npy"


@dataclass(frozen=True, slots=True)
class _StoredOccupancy:
    run_id: str
    source_capture_ref: CaptureArtifactRef
    calibration_ref: CalibrationArtifactRef
    readout_binding: ReadoutBindingKey
    readout_event_axis_id: AxisId
    model_kind: ReadoutModelKind
    counts_schema: DatasetSchema
    occupied_schema: DatasetSchema

    def __post_init__(self) -> None:
        canonical_text(self.run_id, "run_id")
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        _require_occupancy_output_schemas(
            self.counts_schema,
            self.occupied_schema,
        )


def _stored_to_tree(value: _StoredOccupancy) -> dict[str, object]:
    if not isinstance(value, _StoredOccupancy):
        raise TypeError("value must be stored Occupancy metadata")
    return {
        "schema": OCCUPANCY_RECORD_SCHEMA,
        "run_id": value.run_id,
        "source_capture_ref": capture_artifact_ref_to_tree(
            value.source_capture_ref
        ),
        "calibration_ref": calibration_artifact_ref_to_tree(
            value.calibration_ref
        ),
        "readout_binding": readout_binding_key_to_tree(value.readout_binding),
        "readout_event_axis_id": value.readout_event_axis_id.value,
        "model_kind": value.model_kind.value,
        "counts_schema": dataset_schema_to_tree(value.counts_schema),
        "occupied_schema": dataset_schema_to_tree(value.occupied_schema),
    }


def _stored_from_tree(tree: object) -> _StoredOccupancy:
    data = exact_mapping(
        tree,
        _RECORD_FIELDS,
        OCCUPANCY_RECORD_SCHEMA,
        discriminator="schema",
    )
    return _StoredOccupancy(
        canonical_text(data["run_id"], "run_id"),
        capture_artifact_ref_from_tree(data["source_capture_ref"]),
        calibration_artifact_ref_from_tree(data["calibration_ref"]),
        readout_binding_key_from_tree(data["readout_binding"]),
        AxisId(
            canonical_text(
                data["readout_event_axis_id"],
                "readout_event_axis_id",
            )
        ),
        ReadoutModelKind(canonical_text(data["model_kind"], "model_kind")),
        dataset_schema_from_tree(data["counts_schema"]),
        dataset_schema_from_tree(data["occupied_schema"]),
    )


def _encode_stored(value: _StoredOccupancy) -> bytes:
    return encode(_stored_to_tree(value))


def _decode_stored(payload: bytes) -> _StoredOccupancy:
    if not isinstance(payload, bytes):
        raise TypeError("Occupancy record payload must be bytes")
    value = _stored_from_tree(decode(payload))
    if _encode_stored(value) != payload:
        raise ValueError("Occupancy record is not canonical current format")
    return value


def _absolute_root(value: str | Path, field: str) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError(f"{field} must be absolute")
    return root.resolve()


def _run_name(run_id: str) -> str:
    run = canonical_text(run_id, "run_id")
    if "/" in run or "\\" in run:
        raise ValueError("run_id cannot contain path separators")
    path = PurePosixPath(run)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise ValueError("run_id cannot name a path outside the Occupancy root")
    return path.name


def _write_npy(target: Path, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Occupancy arrays cannot contain Python objects")
    atomic_write_file(
        target,
        lambda stream: np.save(stream, array, allow_pickle=False),
    )


def _load_npy(
    target: Path,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray:
    value = np.load(target, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field} file did not contain an ndarray")
    if value.dtype != dtype or value.shape != shape:
        raise ValueError(f"{field} dtype or shape differs from its record schema")
    return value


def _validate_artifact_for_write(
    artifact: OccupancyArtifact,
    readout_binding: ReadoutBindingKey,
    *,
    run_id: str,
) -> _StoredOccupancy:
    if not isinstance(artifact, OccupancyArtifact):
        raise TypeError("artifact must be OccupancyArtifact")
    if not isinstance(readout_binding, ReadoutBindingKey):
        raise TypeError("readout_binding must be ReadoutBindingKey")
    run = canonical_text(run_id, "run_id")
    if artifact.run_id != run:
        raise ValueError("Occupancy artifact belongs to another Run")
    return _StoredOccupancy(
        run,
        artifact.source_capture_ref,
        artifact.calibration_reference,
        readout_binding,
        artifact.readout_event_axis_id,
        artifact.model_kind,
        artifact.counts.schema,
        artifact.occupied.schema,
    )


def write_occupancy_artifact(
    occupancy_root: str | Path,
    artifact: OccupancyArtifact,
    *,
    readout_binding: ReadoutBindingKey,
    run_id: str,
) -> OccupancyArtifactRef:
    """Write original-dtype arrays first and publish ``occupancy.json`` last."""

    stored = _validate_artifact_for_write(
        artifact,
        readout_binding,
        run_id=run_id,
    )
    payload = _encode_stored(stored)
    if _decode_stored(payload) != stored:
        raise ValueError("Occupancy record failed its canonical round-trip")

    root = _absolute_root(occupancy_root, "occupancy_root")
    durable_makedirs(root)
    run_directory = resolve_under(root, _run_name(stored.run_id))
    if run_directory.exists():
        raise FileExistsError(f"Occupancy run directory already exists: {run_directory}")
    durable_mkdir(run_directory)

    validity = artifact.counts.validity
    if not isinstance(validity, DatasetComponentValidity):
        raise TypeError("Occupancy artifact requires DatasetComponentValidity")
    _write_npy(run_directory / _COUNTS_FILE, artifact.counts.values)
    _write_npy(run_directory / _OCCUPIED_FILE, artifact.occupied.values)
    _write_npy(run_directory / _VALIDITY_FILE, validity.mask)
    record_path = run_directory / "occupancy.json"
    atomic_write_bytes(record_path, payload)
    return OccupancyArtifactRef(record_path.relative_to(root).as_posix())


def _validate_loaded_dependencies(
    stored: _StoredOccupancy,
    source: CaptureArtifact,
    calibration: ResolvedCalibration,
    binding: _CommittedOccupancyBinding,
) -> _ResolvedCommittedOccupancy:
    if source.ref != stored.source_capture_ref or (
        calibration.reference != stored.calibration_ref
    ):
        raise ValueError("Occupancy record names different parent artifacts")
    if source.camera_provenance.binding != stored.readout_binding:
        raise ValueError("Occupancy record names another readout binding")
    if binding.readout_event_axis_id != stored.readout_event_axis_id or (
        binding.resolved_schema.model_kind is not stored.model_kind
    ):
        raise ValueError("Occupancy record differs from its resolved model binding")
    if binding.resolved_schema.counts_schema != stored.counts_schema or (
        binding.resolved_schema.occupied_schema != stored.occupied_schema
    ):
        raise ValueError("Occupancy record schemas differ from its parent artifacts")
    return _require_committed_occupancy_context(
        source,
        calibration,
        binding,
    )


def load_occupancy_artifact(
    occupancy_root: str | Path,
    captures_root: str | Path,
    calibrations_root: str | Path,
    reference: OccupancyArtifactRef,
) -> ResolvedOccupancy:
    """Cold-open one record and validate both exact parent artifacts."""

    if not isinstance(reference, OccupancyArtifactRef):
        raise TypeError("reference must be OccupancyArtifactRef")
    occupancy = _absolute_root(occupancy_root, "occupancy_root")
    captures = _absolute_root(captures_root, "captures_root")
    calibrations = _absolute_root(calibrations_root, "calibrations_root")
    record_path = resolve_under(occupancy, reference.record_path)
    stored = _decode_stored(record_path.read_bytes())
    if PurePosixPath(reference.record_path).parts[0] != _run_name(stored.run_id):
        raise ValueError("Occupancy reference path differs from its Run provenance")

    source = load_capture_artifact(captures, stored.source_capture_ref)
    calibration = load_calibration_artifact(
        calibrations,
        captures,
        stored.calibration_ref,
    )
    binding = _resolve_committed_occupancy_structure(
        source,
        calibration,
        readout_event_axis_id=stored.readout_event_axis_id,
        model_kind=stored.model_kind,
    )
    resolved = _validate_loaded_dependencies(
        stored,
        source,
        calibration,
        binding,
    )

    shape = stored.counts_schema.physical_shape
    validity = DatasetComponentValidity(
        (binding.resolved_schema.selected_model.feature.site_axis.axis_id,),
        _load_npy(
            record_path.parent / _VALIDITY_FILE,
            dtype=np.dtype(bool),
            shape=shape,
            field="Occupancy validity",
        ),
    )
    counts = DataBlock(
        OCCUPANCY_COUNTS_BLOCK_ID,
        source.frame_source.revision,
        _load_npy(
            record_path.parent / _COUNTS_FILE,
            dtype=stored.counts_schema.cell_schema.dtype,
            shape=shape,
            field="Occupancy counts",
        ),
        validity,
        stored.counts_schema,
    )
    occupied = DataBlock(
        OCCUPANCY_OCCUPIED_BLOCK_ID,
        source.frame_source.revision,
        _load_npy(
            record_path.parent / _OCCUPIED_FILE,
            dtype=stored.occupied_schema.cell_schema.dtype,
            shape=shape,
            field="Occupancy occupied",
        ),
        validity,
        stored.occupied_schema,
    )
    artifact = OccupancyArtifact(
        stored.source_capture_ref,
        stored.calibration_ref,
        stored.readout_event_axis_id,
        stored.model_kind,
        stored.run_id,
        counts,
        occupied,
    )
    return ResolvedOccupancy(
        reference,
        artifact,
        resolved.source.camera_provenance.binding,
        stored.run_id,
    )


_ExecutedOccupancyAnalysis = tuple[OccupancyArtifact, ReadoutBindingKey]


def compile_occupancy_artifact_plan(
    source_capture_ref: CaptureArtifactRef,
    calibration_ref: CalibrationArtifactRef,
    *,
    captures_root: Path,
    calibrations_root: Path,
    occupancy_root: Path,
    expected_readout_binding: ReadoutBindingKey,
    readout_event_axis_id: AxisId,
    model_kind: ReadoutModelKind,
    timeout_seconds: float,
) -> RunPlan:
    """Compile committed raw frames to one direct-output Occupancy artifact."""

    if not isinstance(source_capture_ref, CaptureArtifactRef):
        raise TypeError("source_capture_ref must be CaptureArtifactRef")
    if not isinstance(calibration_ref, CalibrationArtifactRef):
        raise TypeError("calibration_ref must be CalibrationArtifactRef")
    roots = (
        ("captures_root", captures_root),
        ("calibrations_root", calibrations_root),
        ("occupancy_root", occupancy_root),
    )
    for field, value in roots:
        if not isinstance(value, Path) or not value.is_absolute():
            raise TypeError(f"{field} must be an absolute Path")
    if not isinstance(expected_readout_binding, ReadoutBindingKey):
        raise TypeError("expected_readout_binding must be ReadoutBindingKey")
    if not isinstance(readout_event_axis_id, AxisId):
        raise TypeError("readout_event_axis_id must be AxisId")
    if not isinstance(model_kind, ReadoutModelKind):
        raise TypeError("model_kind must be a concrete ReadoutModelKind")
    timeout = positive_real(timeout_seconds, "timeout_seconds")

    def preflight(context: RunContext) -> _ResolvedCommittedOccupancy:
        context.checkpoint()
        source = load_capture_artifact(captures_root, source_capture_ref)
        calibration = load_calibration_artifact(
            calibrations_root,
            captures_root,
            calibration_ref,
        )
        if source.camera_provenance.binding != expected_readout_binding or (
            calibration.artifact.frame_contract.binding != expected_readout_binding
        ):
            raise ValueError("Occupancy dependencies differ from the frozen binding")
        binding = _resolve_committed_occupancy_structure(
            source,
            calibration,
            readout_event_axis_id=readout_event_axis_id,
            model_kind=model_kind,
        )
        resolved = _require_committed_occupancy_context(
            source,
            calibration,
            binding,
            checkpoint=context.checkpoint,
        )
        context.checkpoint()
        return resolved

    def execute(
        context: RunContext,
        prepared: _ResolvedCommittedOccupancy,
    ) -> _ExecutedOccupancyAnalysis:
        artifact = _analyze_committed_occupancy_resolved(
            prepared,
            run_id=context.run_id.value,
            checkpoint=context.checkpoint,
        )
        return artifact, prepared.source.camera_provenance.binding

    def cleanup(
        _context: RunContext,
        _prepared: _ResolvedCommittedOccupancy | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        return CleanupReport()

    def finalize(
        context: PostSafetyContext,
        executed: _ExecutedOccupancyAnalysis,
    ) -> OccupancyArtifactRef:
        artifact, readout_binding = executed
        return write_occupancy_artifact(
            occupancy_root,
            artifact,
            readout_binding=readout_binding,
            run_id=context.run_id.value,
        )

    return RunPlan(
        name="classify committed camera capture occupancy",
        resource_claims=(),
        bound_devices=(),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        timeout_seconds=timeout,
    )


__all__ = [
    "OCCUPANCY_RECORD_SCHEMA",
    "compile_occupancy_artifact_plan",
    "load_occupancy_artifact",
    "write_occupancy_artifact",
]
