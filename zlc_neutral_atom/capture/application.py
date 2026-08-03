"""Shared exact camera-capture intent and physical binding."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import AxisId, AxisSpec, REPEAT
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraBinding,
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_pulse import PulseDocument, PulseExecutionForm
from zlc_storage import canonical_text, positive_integer

_CAPTURE_REPEAT_AXIS_ID = AxisId("capture.repeat")
_CAPTURE_SCAN_AXIS_ID = AxisId("capture.scan_row_ordinal")
CAPTURE_READOUT_EVENT_AXIS_ID = AxisId("capture.readout_event")


@dataclass(frozen=True)
class CaptureRequest:
    pulse_document: PulseDocument
    execution_form: PulseExecutionForm
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    trigger_channel: str | None = None
    repeat_count: int = 1
    readout_events_per_repeat: int | None = None
    within_point_grouping: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if self.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
            raise ValueError("CaptureRequest requires a finite pulse execution form")
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        if self.trigger_channel is not None:
            canonical_text(self.trigger_channel, "trigger_channel")
        object.__setattr__(
            self,
            "repeat_count",
            positive_integer(self.repeat_count, "repeat_count"),
        )
        if self.readout_events_per_repeat is not None:
            object.__setattr__(
                self,
                "readout_events_per_repeat",
                positive_integer(
                    self.readout_events_per_repeat,
                    "readout_events_per_repeat",
                ),
            )
        if self.within_point_grouping is not None:
            try:
                grouping = tuple(tuple(pair) for pair in self.within_point_grouping)
            except TypeError as exc:
                raise TypeError(
                    "within_point_grouping must be an iterable of pairs"
                ) from exc
            object.__setattr__(self, "within_point_grouping", grouping)


def bind_finite_capture_request(
    request: CaptureRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
) -> TriggeredCameraBinding:
    """Bind the physical camera/pulse source shared by capture and processors."""

    if not isinstance(request, CaptureRequest):
        raise TypeError("request must be CaptureRequest")
    return bind_triggered_camera_acquisition(
        pulse_port,
        camera_port,
        pulse_document=request.pulse_document,
        execution_form=request.execution_form,
        trigger_channel=request.trigger_channel,
        layout=TriggeredCameraLayout(
            repeat_axis=AxisSpec(
                _CAPTURE_REPEAT_AXIS_ID,
                "repeat",
                REPEAT,
                request.repeat_count,
                tuple(range(request.repeat_count)),
            ),
            readout_event_axis_id=CAPTURE_READOUT_EVENT_AXIS_ID,
            ordinal_scan_axis_id=_CAPTURE_SCAN_AXIS_ID,
            readout_events_per_repeat=request.readout_events_per_repeat,
            within_point_grouping=request.within_point_grouping,
        ),
        camera_instance_id=request.camera_ref.instance_id,
    )


__all__ = [
    "CAPTURE_READOUT_EVENT_AXIS_ID",
    "CaptureRequest",
    "bind_finite_capture_request",
]
