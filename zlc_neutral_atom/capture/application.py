"""Shared exact camera-capture request, plan, and prepared application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from zlc_data import AxisId, AxisSpec, BlockId, DatasetSchema, REPEAT
from .artifact import CaptureRepository
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.capture.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.capture.triggered import TriggeredCaptureSpec
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraBinding,
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_pulse import PulseDocument, PulseExecutionForm
from zlc_storage import canonical_text, positive_integer

from .prepared import (
    PreparedExactCapture,
)


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


@dataclass(frozen=True)
class PlanDescriptor:
    name: str
    camera_role: str
    sequencer_role: str
    execution_form: PulseExecutionForm
    trigger_channel: str
    expected_frames: int
    output_schema: DatasetSchema
    compiled_pulse_digest: str
    resource_claims: tuple[str, ...]


class PreparedFiniteCapture(PreparedExactCapture):
    """Explicit pulse-owned finite Capture command."""

    def __init__(
        self,
        triggered: TriggeredCaptureSpec,
        repository: CaptureRepository,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor: PlanDescriptor,
    ) -> None:
        if not isinstance(triggered, TriggeredCaptureSpec):
            raise TypeError("triggered must be TriggeredCaptureSpec")
        if not isinstance(descriptor, PlanDescriptor):
            raise TypeError("descriptor must be PlanDescriptor")
        super().__init__(
            triggered,
            repository,
            start_run,
            descriptor,
            one_shot_name="PreparedFiniteCapture",
        )

    @property
    def descriptor(self) -> PlanDescriptor:
        descriptor = self._descriptor
        assert isinstance(descriptor, PlanDescriptor)
        return descriptor


def bind_finite_capture_spec(
    *,
    binding: TriggeredCameraBinding,
    block_id: BlockId,
    camera_ref: DeviceRef,
    sequencer_ref: DeviceRef,
    execution_form: PulseExecutionForm,
    name_prefix: str,
) -> tuple[TriggeredCaptureSpec, PlanDescriptor]:
    """Freeze the shared exact plan inputs after use-case intent is complete."""

    pipeline = MinimalPipelineSpec(
        f"{name_prefix} {binding.pulse_request.document.name}",
        binding.capture,
        block_id,
    )
    triggered = TriggeredCaptureSpec(
        pipeline,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    descriptor = PlanDescriptor(
        pipeline.name,
        camera_ref.role,
        sequencer_ref.role,
        execution_form,
        binding.trigger_channel,
        binding.expected_frames,
        binding.capture.capture_contract.dataset_schema,
        binding.compiled_artifact.fingerprint,
        (
            str(binding.pulse_port.resource_claim.key),
            str(binding.capture.capture_port.resource_claim.key),
        ),
    )
    return triggered, descriptor


def prepare_finite_capture(
    request: CaptureRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    repository: CaptureRepository,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedFiniteCapture:
    """Bind one ordinary finite request into a narrow one-shot command."""

    binding = bind_finite_capture_request(
        request,
        pulse_port=pulse_port,
        camera_port=camera_port,
    )
    triggered, descriptor = bind_finite_capture_spec(
        binding=binding,
        block_id=BlockId(f"capture-{binding.compiled_artifact.fingerprint[:20]}"),
        camera_ref=request.camera_ref,
        sequencer_ref=request.sequencer_ref,
        execution_form=request.execution_form,
        name_prefix="Capture",
    )
    return PreparedFiniteCapture(triggered, repository, start_run, descriptor)


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
    )


__all__ = [
    "CAPTURE_READOUT_EVENT_AXIS_ID",
    "CaptureRequest",
    "PlanDescriptor",
    "PreparedFiniteCapture",
    "bind_finite_capture_request",
    "bind_finite_capture_spec",
    "prepare_finite_capture",
]
