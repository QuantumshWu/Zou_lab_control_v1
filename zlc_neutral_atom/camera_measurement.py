"""Public intent and descriptor for the one Camera Measurement."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import (
    MONITOR_HISTORY,
    READOUT_EVENT,
    BlockId,
    DatasetRevisionRef,
    DatasetSchema,
    IndexSelection,
    OwnedSnapshot,
    Selection,
    dataset_revision_ref_to_tree,
    materialize_dataset_selection,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.capture_reference import (
    CaptureArtifactRef,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.dataset_output import (
    FinalDatasetOutput,
    LiveDatasetOutput,
    final_dataset_join_digest,
)
from zlc_neutral_atom.runtime.dataset import (
    MonitorCoverage,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.runtime.streams import event_ref_to_tree
from zlc_storage import canonical_digest, canonical_text, positive_integer


DEFAULT_CAMERA_MEASUREMENT_ROLE = "camera"
CAMERA_MEASUREMENT_ROLE_ORDER = ("camera", "mot_camera")
DEFAULT_CAMERA_MEASUREMENT_REPEAT = 0
DEFAULT_CAMERA_FRAMES_PER_CYCLE = 1
DEFAULT_CAMERA_MONITOR_HISTORY_CAPACITY = 8
MINIMUM_CAMERA_MEASUREMENT_REPEAT = 0
MINIMUM_CAMERA_FRAMES_PER_CYCLE = 1


def camera_frame_output_index(output_name: str) -> int:
    """Parse one canonical ``frame_i`` name without accepting aliases."""

    name = canonical_text(output_name, "camera frame output name")
    prefix = "frame_"
    if not name.startswith(prefix):
        raise ValueError("camera frame output name must start with 'frame_'")
    token = name[len(prefix) :]
    if not token.isascii() or not token.isdecimal():
        raise ValueError("camera frame output index must be a decimal integer")
    index = int(token)
    if token != str(index):
        raise ValueError("camera frame output index must use canonical decimal text")
    return index


def camera_frame_output_names(frames_per_cycle: int) -> tuple[str, ...]:
    """Return the public signal names for one camera cycle.

    A camera cycle is stored atomically as one Dataset whose named
    ``READOUT_EVENT`` axis has ``frames_per_cycle`` cells.  Presentation may
    expose those cells independently, but their names are owned here beside
    the request that defines the cycle -- never guessed from an ndarray shape
    and never supplemented with a lossy ``frame`` alias.
    """

    count = positive_integer(frames_per_cycle, "frames_per_cycle")
    return tuple(f"frame_{index}" for index in range(count))


def _camera_frame_ref(
    source: OwnedSnapshot,
    event_axis_id: str,
    event_index: int,
    output_schema: DatasetSchema,
) -> DatasetRevisionRef:
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.camera-measurement.frame-output",
            "source_block_id": source.ref.block_id.value,
            "readout_event_axis_id": event_axis_id,
            "readout_event_index": event_index,
        }
    )
    return DatasetRevisionRef(
        BlockId(f"camera-frame-{identity[:32]}"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )


def project_camera_measurement_outputs(
    source: OwnedSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, OwnedSnapshot]:
    """Materialize the Camera Measurement's declared ``frame_i`` outputs.

    The measurement contract, rather than a GUI, owns how one atomic camera
    cycle maps its named ``READOUT_EVENT`` cells to public outputs.  Repeat,
    remaining point axes, trailing data axes, component validity, revision and
    stream generation are retained exactly.
    """

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("camera frame source must be OwnedSnapshot")
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("camera output projection requires CameraMeasurementRequest")
    schema = source.block.schema
    event_axes = tuple(
        axis for axis in schema.point_axes if axis.role == READOUT_EVENT
    )
    if len(event_axes) != 1:
        raise ValueError(
            "camera Dataset must contain exactly one READOUT_EVENT point axis"
        )
    if any(axis.role == READOUT_EVENT for axis in schema.cell_schema.data_axes):
        raise ValueError("READOUT_EVENT cannot be a trailing camera data axis")
    event_axis = event_axes[0]
    if event_axis.size != request.frames_per_cycle:
        raise ValueError(
            "camera Dataset READOUT_EVENT size differs from frames_per_cycle"
        )

    expected_point_axes = tuple(
        axis for axis in schema.point_axes if axis.axis_id != event_axis.axis_id
    )
    projected: dict[str, OwnedSnapshot] = {}
    for event_index, output_name in enumerate(request.output_names):
        snapshot = materialize_dataset_selection(
            source,
            Selection.index(event_axis.axis_id, event_index),
            reference_for=lambda output_schema, index=event_index: _camera_frame_ref(
                source,
                event_axis.axis_id.value,
                index,
                output_schema,
            ),
        )
        output_schema = snapshot.block.schema
        if output_schema.repeat_axis != schema.repeat_axis:
            raise RuntimeError("camera output projection changed the repeat axis")
        if output_schema.point_axes != expected_point_axes:
            raise RuntimeError("camera output projection changed another point axis")
        if output_schema.cell_schema != schema.cell_schema:
            raise RuntimeError("camera output projection changed trailing data axes")
        if snapshot.ref.stream_generation != source.ref.stream_generation:
            raise RuntimeError("camera output projection changed stream generation")
        projected[output_name] = snapshot

    if tuple(projected) != request.output_names:
        raise RuntimeError("camera output projection order is inconsistent")
    return projected


def camera_measurement_final_outputs(
    reference: CaptureArtifactRef,
    source: OwnedSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, FinalDatasetOutput]:
    """Publish every declared Camera frame from one admitted FINAL capture."""

    if not isinstance(reference, CaptureArtifactRef):
        raise TypeError("Camera FINAL reference must be CaptureArtifactRef")
    snapshots = project_camera_measurement_outputs(source, request)
    source_identity = capture_artifact_ref_to_tree(reference)
    return {
        name: FinalDatasetOutput(
            name,
            snapshot,
            final_dataset_join_digest(
                owner="camera-measurement",
                output_name=name,
                source_identity=source_identity,
                snapshot=snapshot,
            ),
        )
        for name, snapshot in snapshots.items()
    }


def project_camera_monitor_outputs(
    source: MonitorDatasetSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, LiveDatasetOutput]:
    """Project an atomic monitor cycle without misreporting frame coverage.

    Event references are selected with the same declared READOUT_EVENT index as
    the Dataset.  Consequently each ``frame_i`` reports coverage in its own
    reduced point geometry and carries provenance for that frame rather than
    reusing the most recent event of the whole cycle.
    """

    if not isinstance(source, MonitorDatasetSnapshot):
        raise TypeError("source must be MonitorDatasetSnapshot")
    frames = project_camera_measurement_outputs(source.snapshot, request)
    source_schema = source.snapshot.block.schema
    event_positions = tuple(
        index
        for index, axis in enumerate(source_schema.point_axes)
        if axis.role == READOUT_EVENT
    )
    if len(event_positions) != 1:
        raise ValueError("Camera monitor must contain one READOUT_EVENT axis")
    event_position = event_positions[0]
    projected: dict[str, LiveDatasetOutput] = {}
    for output_name, snapshot in frames.items():
        event_index = camera_frame_output_index(output_name)
        output_schema = snapshot.block.schema
        selected_refs = []
        for repeat_index in range(output_schema.repeat_axis.size):
            for output_storage_index in range(
                output_schema.point_layout.storage_size
            ):
                output_logical = list(
                    output_schema.point_layout.multi_index(output_storage_index)
                )
                output_logical.insert(event_position, event_index)
                source_storage_index = source_schema.point_layout.storage_index(
                    tuple(output_logical)
                )
                linear_index = (
                    repeat_index * source_schema.point_layout.storage_size
                    + source_storage_index
                )
                selected_refs.append(source.event_refs[linear_index])
        total = (
            output_schema.repeat_axis.size
            * output_schema.point_layout.storage_size
        )
        if len(selected_refs) != total:
            raise RuntimeError("Camera monitor projection lost output cells")
        coverage = MonitorCoverage(
            written_cells=sum(ref is not None for ref in selected_refs),
            total_cells=total,
            missed_events=source.coverage.missed_events,
            current_gap=source.coverage.current_gap,
        )
        join_digest = canonical_digest(
            {
                "owner": "zlc_neutral_atom.camera-monitor-output",
                "source": dataset_revision_ref_to_tree(source.snapshot.ref),
                "output_name": output_name,
                "events": tuple(
                    None if ref is None else event_ref_to_tree(ref)
                    for ref in selected_refs
                ),
            }
        )
        projected[output_name] = LiveDatasetOutput(
            output_name,
            snapshot,
            coverage,
            join_digest,
        )
    return projected


def current_camera_monitor_selection(
    schema: DatasetSchema,
    coverage: MonitorCoverage,
) -> tuple[int, tuple[int, ...], Selection]:
    """Resolve the exact current Camera cell from monitor storage semantics.

    ``MonitorDataset`` presents the newest history slot at logical index zero;
    Camera's default live view presents ``frame_0`` of that cycle.  Keeping
    this rule beside Camera Measurement prevents a GUI from rediscovering the
    cell from array rank or storage order.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(coverage, MonitorCoverage):
        raise TypeError("coverage must be MonitorCoverage")
    if coverage.written_cells == 0:
        raise ValueError("the current Camera monitor has no committed cell")
    if schema.repeat_axis.size != 1:
        raise ValueError("the current Camera monitor requires one storage repeat")
    history_axes = tuple(
        axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY
    )
    if len(history_axes) != 1:
        raise ValueError(
            "the current Camera monitor requires one MONITOR_HISTORY axis"
        )
    event_axes = tuple(
        axis for axis in schema.point_axes if axis.role == READOUT_EVENT
    )
    if len(event_axes) > 1:
        raise ValueError("a Camera monitor has multiple READOUT_EVENT axes")
    if any(
        axis.role not in {MONITOR_HISTORY, READOUT_EVENT}
        for axis in schema.point_axes
    ):
        raise ValueError("a Camera monitor contains an unsupported point axis")

    logical_point = tuple(0 for _axis in schema.point_axes)
    selection = Selection(
        (
            IndexSelection(schema.repeat_axis.axis_id, 0),
            *(
                IndexSelection(axis.axis_id, index)
                for axis, index in zip(
                    schema.point_axes,
                    logical_point,
                    strict=True,
                )
            ),
        )
    )
    return (
        schema.point_layout.storage_index(logical_point),
        logical_point,
        selection,
    )


@dataclass(frozen=True)
class CameraMeasurementRequest:
    """Read raw camera cycles: ``repeat=0`` live, ``repeat=K`` finite.

    The request owns only the selected camera.  Trigger timing belongs to
    independently running hardware; Camera Measurement never acquires pulse or
    sequencer authority.
    """

    camera_ref: DeviceRef
    repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT
    history_capacity: int = DEFAULT_CAMERA_MONITOR_HISTORY_CAPACITY
    frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE

    def __post_init__(self) -> None:
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise TypeError("repeat must be an integer")
        if self.repeat < MINIMUM_CAMERA_MEASUREMENT_REPEAT:
            raise ValueError("repeat must be non-negative")
        object.__setattr__(
            self,
            "history_capacity",
            positive_integer(self.history_capacity, "history_capacity"),
        )
        object.__setattr__(
            self,
            "frames_per_cycle",
            positive_integer(self.frames_per_cycle, "frames_per_cycle"),
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        """One signal per declared readout event, in cycle order."""

        return camera_frame_output_names(self.frames_per_cycle)


@dataclass(frozen=True)
class CameraMeasurementDescriptor:
    name: str
    camera_role: str
    output_schema: DatasetSchema
    resource_claim: str

    def __post_init__(self) -> None:
        canonical_text(self.name, "camera measurement name")
        canonical_text(self.camera_role, "camera role")
        if not isinstance(self.output_schema, DatasetSchema):
            raise TypeError("output_schema must be DatasetSchema")
        canonical_text(self.resource_claim, "resource_claim")


__all__ = [
    "CAMERA_MEASUREMENT_ROLE_ORDER",
    "CameraMeasurementDescriptor",
    "CameraMeasurementRequest",
    "DEFAULT_CAMERA_FRAMES_PER_CYCLE",
    "DEFAULT_CAMERA_MEASUREMENT_REPEAT",
    "DEFAULT_CAMERA_MEASUREMENT_ROLE",
    "DEFAULT_CAMERA_MONITOR_HISTORY_CAPACITY",
    "MINIMUM_CAMERA_FRAMES_PER_CYCLE",
    "MINIMUM_CAMERA_MEASUREMENT_REPEAT",
    "camera_frame_output_index",
    "camera_frame_output_names",
    "current_camera_monitor_selection",
    "project_camera_measurement_outputs",
    "camera_measurement_final_outputs",
    "project_camera_monitor_outputs",
]
