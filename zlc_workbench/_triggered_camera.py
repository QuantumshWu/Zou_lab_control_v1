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
from zlc_neutral_atom.acquisition import CameraAcquisitionMode, CameraSampleContract
from zlc_neutral_atom.runtime.pipeline import BoundMeasurement
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.timing.capture_plan import (
    CompiledCaptureCellPlan,
    compile_capture_cell_plan,
)
from zlc_neutral_atom.timing.pulse import (
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

from .camera_capture import (
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
class _TriggeredCameraLayout:
    """Named sampling intent; the compiled schedule supplies scan cardinality."""

    repeat_axis_id: AxisId
    scan_axis_id: AxisId
    readout_event_axis_id: AxisId
    repeat_count: int = 1
    readout_events_per_repeat: int | None = None
    within_point_grouping: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        for field in (
            "repeat_axis_id",
            "scan_axis_id",
            "readout_event_axis_id",
        ):
            if not isinstance(getattr(self, field), AxisId):
                raise TypeError(f"{field} must be AxisId")
        if len(
            {
                self.repeat_axis_id,
                self.scan_axis_id,
                self.readout_event_axis_id,
            }
        ) != 3:
            raise ValueError("triggered-camera sampling AxisIds must be distinct")
        object.__setattr__(
            self,
            "repeat_count",
            _positive_int(self.repeat_count, "repeat_count"),
        )
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


@dataclass(frozen=True, slots=True)
class _TriggeredCameraBinding:
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


def _bind_triggered_camera_acquisition(
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    *,
    pulse_document: PulseDocument,
    execution_form: PulseExecutionForm,
    trigger_channel: str | None,
    layout: _TriggeredCameraLayout,
    transport_memory_limit_bytes: int,
) -> _TriggeredCameraBinding:
    """Bind one exact finite pulse/camera acquisition without starting hardware."""

    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    if not isinstance(pulse_document, PulseDocument):
        raise TypeError("pulse_document must be PulseDocument")
    if not isinstance(execution_form, PulseExecutionForm):
        raise TypeError("execution_form must be PulseExecutionForm")
    if execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
        raise ValueError("triggered camera acquisition requires a finite pulse form")
    if not isinstance(layout, _TriggeredCameraLayout):
        raise TypeError("layout must be _TriggeredCameraLayout")
    transport_memory_limit_bytes = _positive_int(
        transport_memory_limit_bytes,
        "transport_memory_limit_bytes",
    )

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
    for edge in schedule.edges:
        point_counts[edge.point_index] += 1
    per_point_counts = tuple(point_counts)
    if len(set(per_point_counts)) != 1:
        raise ValueError("camera trigger count must be uniform across scan points")
    per_point = per_point_counts[0]
    events_per_repeat = layout.readout_events_per_repeat
    if events_per_repeat is None:
        if layout.repeat_count != 1:
            raise ValueError(
                "readout_events_per_repeat is required when repeat_count exceeds one"
            )
        events_per_repeat = per_point
    if layout.repeat_count * events_per_repeat != per_point:
        raise ValueError(
            "declared repeat/event layout differs from per-point trigger count"
        )

    repeat_axis = _axis(layout.repeat_axis_id, REPEAT, layout.repeat_count)
    scan_axes = (
        (_axis(layout.scan_axis_id, SCAN_POINT, schedule.point_count),)
        if schedule.point_count > 1
        else ()
    )
    event_axis = _axis(
        layout.readout_event_axis_id,
        READOUT_EVENT,
        events_per_repeat,
    )
    point_axes = (*scan_axes, event_axis)
    point_layout = PointLayout.rect_c(tuple(axis.size for axis in point_axes))
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
        scan_point_layout=PointLayout.rect_c(
            tuple(axis.size for axis in scan_axes)
        ),
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
            0,
            transport_memory_limit_bytes,
            tuple(
                camera_facts.event_setting(index)
                for index in range(events_per_repeat)
            ),
        )
    )
    pulse_request = FinitePulseExecutionRequest(document, artifact)
    return _TriggeredCameraBinding(
        pulse_port,
        pulse_request,
        selected_trigger,
        measurement,
        cell_plan,
    )


__all__: list[str] = []
