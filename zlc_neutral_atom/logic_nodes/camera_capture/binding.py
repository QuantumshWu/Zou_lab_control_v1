"""Private composition seam for one finite pulse-triggered camera acquisition.

This module owns the one place where installation identity, physical trigger
wiring, compiled pulse cardinality, named sampling axes, the exact cell plan,
and the bound camera measurement are joined.  It is deliberately a concrete
composition helper rather than a public workflow or a generic graph builder.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraSampleContract,
)
from .pipeline import BoundMeasurement
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.timing.capture_plan import (
    CompiledCaptureCellPlan,
    compile_capture_cell_plan,
)
from zlc_neutral_atom.devices.sequencer.port import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseDocument,
    PulseExecutionForm,
    bind_pulse_document_target,
    compile_pulse_artifact,
)
from zlc_storage import canonical_text, positive_integer as _positive_int

from zlc_neutral_atom.logic_nodes.camera_measurement.binding import (
    CameraCaptureBindingRequest,
    bind_camera_measurement,
)


def _canonical_grouping(
    value: tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    try:
        grouping = tuple(tuple(pair) for pair in value)
    except TypeError as exc:
        raise TypeError(
            "within_point_grouping must contain integer (repeat, event) pairs"
        ) from exc
    if any(
        len(pair) != 2
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in pair
        )
        for pair in grouping
    ):
        raise ValueError(
            "within_point_grouping must contain non-negative two-item pairs"
        )
    return grouping


@dataclass(frozen=True, slots=True)
class TriggeredCameraLayout:
    """Named sampling intent; the compiled schedule supplies scan cardinality.

    Ordinary captures may leave ``scan_axes`` absent and receive one explicit
    ordinal axis.  A scan authority instead supplies its physical axes and
    sparse/rectangular row mapping together; this binding never infers those
    semantics from the numeric table shape.
    """

    repeat_axis: AxisSpec
    readout_event_axis_id: AxisId
    ordinal_scan_axis_id: AxisId | None = None
    readout_events_per_repeat: int | None = None
    within_point_grouping: tuple[tuple[int, int], ...] | None = None
    scan_axes: tuple[AxisSpec, ...] | None = None
    scan_point_layout: PointLayout | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repeat_axis, AxisSpec)
            or self.repeat_axis.role != REPEAT
        ):
            raise ValueError("repeat_axis must be an AxisSpec with repeat role")
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if self.readout_events_per_repeat is not None:
            object.__setattr__(
                self,
                "readout_events_per_repeat",
                _positive_int(
                    self.readout_events_per_repeat,
                    "readout_events_per_repeat",
                ),
            )
        object.__setattr__(
            self,
            "within_point_grouping",
            _canonical_grouping(self.within_point_grouping),
        )
        if self.scan_axes is None:
            if self.scan_point_layout is not None:
                raise ValueError("scan_point_layout requires declared scan_axes")
            if not isinstance(self.ordinal_scan_axis_id, AxisId):
                raise TypeError(
                    "ordinary capture requires one ordinal_scan_axis_id"
                )
            axis_ids = {
                self.repeat_axis.axis_id,
                self.ordinal_scan_axis_id,
                self.readout_event_axis_id,
            }
            if len(axis_ids) != 3:
                raise ValueError("triggered-camera sampling AxisIds must be distinct")
        else:
            if self.ordinal_scan_axis_id is not None:
                raise ValueError(
                    "physical scan_axes replace the ordinal scan axis identity"
                )
            axes = tuple(self.scan_axes)
            if not axes or any(
                not isinstance(axis, AxisSpec) or axis.role != SCAN_POINT
                for axis in axes
            ):
                raise ValueError("scan_axes must contain SCAN_POINT AxisSpec values")
            if len({axis.axis_id for axis in axes}) != len(axes):
                raise ValueError("scan_axes must have unique AxisIds")
            if len(
                {
                    self.repeat_axis.axis_id,
                    self.readout_event_axis_id,
                    *(axis.axis_id for axis in axes),
                }
            ) != len(axes) + 2:
                raise ValueError("triggered-camera sampling AxisIds must be distinct")
            if not isinstance(self.scan_point_layout, PointLayout):
                raise TypeError("declared scan_axes require a PointLayout")
            if self.scan_point_layout.logical_shape != tuple(
                axis.size for axis in axes
            ):
                raise ValueError("scan PointLayout shape differs from scan_axes")
            object.__setattr__(self, "scan_axes", axes)


@dataclass(frozen=True, slots=True)
class TriggeredCameraBinding:
    """Run-local, generation-pinned result of the concrete composition join."""

    pulse_port: BoundPulsePort
    pulse_request: FinitePulseExecutionRequest
    trigger_channel: str
    measurement: BoundMeasurement
    cell_plan: CompiledCaptureCellPlan

    @property
    def compiled_artifact(self) -> CompiledPulseArtifact:
        return self.pulse_request.artifact

    @property
    def expected_frames(self) -> int:
        return self.cell_plan.total_events




def _axis(axis_id: AxisId, role, size: int) -> AxisSpec:
    return AxisSpec(axis_id, axis_id.value, role, size, tuple(range(size)))




def bind_triggered_camera_acquisition(
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    *,
    pulse_document: PulseDocument,
    execution_form: PulseExecutionForm,
    trigger_channel: str | None,
    layout: TriggeredCameraLayout,
) -> TriggeredCameraBinding:
    """Bind one exact finite pulse/camera acquisition without starting hardware."""

    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    if not isinstance(pulse_document, PulseDocument):
        raise TypeError("pulse_document must be PulseDocument")
    if not isinstance(execution_form, PulseExecutionForm):
        raise TypeError("execution_form must be PulseExecutionForm")
    if execution_form in (
        PulseExecutionForm.CONTINUOUS_MONITOR,
        PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    ):
        raise ValueError("triggered camera acquisition requires a finite pulse form")
    if not isinstance(layout, TriggeredCameraLayout):
        raise TypeError("layout must be TriggeredCameraLayout")
    document = bind_pulse_document_target(
        pulse_document,
        pulse_port.capability.target,
    )
    camera_capability = camera_port.capability
    camera_evidence = camera_capability.camera_capability_evidence
    camera_role = camera_evidence.source_id
    camera_facts = camera_evidence.physical_facts
    camera_payload_contract = camera_capability.payload_contract
    if not isinstance(camera_payload_contract, CameraSampleContract):
        raise TypeError("camera capability payload contract has the wrong type")
    if trigger_channel is None:
        if len(camera_facts.capture_trigger_channels) != 1:
            raise ValueError(
                "exact capture requires exactly one physical camera trigger channel"
            )
        selected_trigger = camera_facts.capture_trigger_channels[0]
    else:
        selected_trigger = canonical_text(trigger_channel, "trigger_channel")
    camera_facts.require_single_capture_trigger_channel(
        selected_trigger
    )

    artifact = compile_pulse_artifact(
        document,
        clock_hz=pulse_port.capability.clock_hz,
        execution_form=execution_form,
        trigger_channels=(selected_trigger,),
        live_target=pulse_port.capability.target,
    )
    schedule = artifact.trigger_schedules[0]
    if schedule.total < 1:
        raise ValueError("compiled pulse emits no camera trigger edge")
    point_counts = [0] * schedule.point_count
    for edge in schedule.iter_edges():
        point_counts[edge.point_index] += 1
    per_point_counts = tuple(point_counts)
    if len(set(per_point_counts)) != 1:
        raise ValueError("camera trigger count must be uniform across scan points")
    per_point = per_point_counts[0]
    repeat_major_points = (
        execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
        and layout.scan_axes is not None
    )
    events_per_repeat = layout.readout_events_per_repeat
    if events_per_repeat is None:
        if not repeat_major_points and layout.repeat_axis.size != 1:
            raise ValueError(
                "readout_events_per_repeat is required when repeat_count exceeds one"
            )
        events_per_repeat = per_point
    expected_per_point = (
        events_per_repeat
        if repeat_major_points
        else layout.repeat_axis.size * events_per_repeat
    )
    if expected_per_point != per_point:
        raise ValueError(
            "declared repeat/event layout differs from per-point trigger count"
        )

    repeat_axis = layout.repeat_axis
    if layout.scan_axes is None:
        assert layout.ordinal_scan_axis_id is not None
        scan_axes = (
            (_axis(layout.ordinal_scan_axis_id, SCAN_POINT, schedule.point_count),)
            if schedule.point_count > 1
            else ()
        )
        scan_point_layout = PointLayout.rect_c(
            tuple(axis.size for axis in scan_axes)
        )
    else:
        scan_axes = layout.scan_axes
        assert layout.scan_point_layout is not None
        scan_point_layout = layout.scan_point_layout
        expected_execution_points = (
            layout.repeat_axis.size * scan_point_layout.storage_size
            if repeat_major_points
            else scan_point_layout.storage_size
        )
        if repeat_major_points and (
            schedule.loop_count != 1 or not schedule.full_point_loop
        ):
            raise ValueError(
                "autonomous scan requires one complete repeat-major finite table"
            )
        if expected_execution_points != schedule.point_count:
            raise ValueError(
                "declared scan PointLayout rows differ from compiled pulse points"
            )
    event_axis = _axis(
        layout.readout_event_axis_id,
        READOUT_EVENT,
        events_per_repeat,
    )
    point_axes = (*scan_axes, event_axis)
    point_layout = PointLayout.from_mapping(
        tuple(axis.size for axis in point_axes),
        tuple(
            (*scan_point_layout.multi_index(scan_row), event_index)
            for scan_row in range(scan_point_layout.storage_size)
            for event_index in range(event_axis.size)
        ),
    )
    dataset_schema = DatasetSchema(
        repeat_axis,
        point_axes,
        point_layout,
        camera_payload_contract.value_schema,
    )
    cell_plan = compile_capture_cell_plan(
        artifact,
        selected_trigger,
        dataset_schema,
        readout_event_axis_id=event_axis.axis_id,
        scan_point_layout=scan_point_layout,
        within_point_grouping=layout.within_point_grouping,
    )
    measurement = bind_camera_measurement(
        camera_port,
        CameraCaptureBindingRequest(
            camera_role,
            repeat_axis,
            point_axes,
            point_layout,
            cell_plan.cell_schedule,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            tuple(
                camera_facts.event_setting(index)
                for index in range(events_per_repeat)
            ),
        )
    )
    pulse_request = FinitePulseExecutionRequest(document, artifact)
    return TriggeredCameraBinding(
        pulse_port,
        pulse_request,
        selected_trigger,
        measurement,
        cell_plan,
    )


__all__ = [
    "TriggeredCameraBinding",
    "TriggeredCameraLayout",
    "bind_triggered_camera_acquisition",
]
