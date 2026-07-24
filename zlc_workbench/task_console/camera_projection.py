"""Lossless TaskConsole views of Camera READOUT_EVENT cells."""

from __future__ import annotations

from zlc_data import (
    READOUT_EVENT,
    BlockId,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    Selection,
)
from zlc_neutral_atom.camera_measurement import CameraMeasurementRequest
from zlc_storage import canonical_digest

from .dataset_projection import materialize_dataset_selection

__all__ = ["project_camera_frame_snapshots"]


def _camera_frame_ref(
    source: OwnedSnapshot,
    event_axis_id: str,
    event_index: int,
    output_schema: DatasetSchema,
) -> DatasetRevisionRef:
    identity = canonical_digest(
        {
            "owner": "zlc_workbench.task-console.camera-frame",
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


def project_camera_frame_snapshots(
    source: OwnedSnapshot,
    request: CameraMeasurementRequest,
) -> dict[str, OwnedSnapshot]:
    """Project one atomic cycle Dataset to ordered, bindable ``frame_i`` views.

    Only the declared ``READOUT_EVENT`` point axis is selected.  Repeat,
    remaining point axes (for example monitor history), every trailing data
    axis, component validity, revision, and stream generation remain intact.
    """

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("camera frame source must be OwnedSnapshot")
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("camera frame projection requires CameraMeasurementRequest")
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
            raise RuntimeError("camera frame projection changed the repeat axis")
        if output_schema.point_axes != expected_point_axes:
            raise RuntimeError("camera frame projection changed another point axis")
        if output_schema.cell_schema != schema.cell_schema:
            raise RuntimeError("camera frame projection changed trailing data axes")
        if snapshot.ref.stream_generation != source.ref.stream_generation:
            raise RuntimeError("camera frame projection changed stream generation")
        projected[output_name] = snapshot

    if tuple(projected) != request.output_names:
        raise RuntimeError("camera frame projection output order is inconsistent")
    return projected
