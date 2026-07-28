"""Prepared coupled Camera + Sequencer acquisition for MOT-field optimization.

The MOT capability owns only its experiment-specific binding: three declared
coil axes, one Camera event per grid cell, and one autonomous FPGA FIRE.  Exact
camera transport, hardware coordination, cancellation, and preview remain the
single generic owners in ``capture`` and ``devices``.
"""

from __future__ import annotations

from zlc_data import AxisId, AxisSpec, BlockId, DatasetSchema, REPEAT
from zlc_neutral_atom.capture.artifact import (
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.capture.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.capture.triggered import (
    TriggeredCaptureSpec,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
)
from zlc_neutral_atom.runtime.run import RunPlan
from zlc_pulse import PulseExecutionForm

from .mot_field import (
    MotFieldRequest,
    mot_intensity_schema,
)


_MOT_REPEAT_AXIS = AxisSpec(
    AxisId("mot-field.repeat"),
    "repeat",
    REPEAT,
    1,
    (0,),
)
_MOT_READOUT_EVENT_AXIS_ID = AxisId("mot-field.readout-event")


class PreparedMotFieldAcquisition:
    """Fully bound autonomous MOT capture, still free of lifecycle ownership."""

    __slots__ = (
        "_request",
        "_source_schema",
        "_spec",
    )

    def __init__(
        self,
        request: MotFieldRequest,
        spec: TriggeredCaptureSpec,
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(spec, TriggeredCaptureSpec):
            raise TypeError("spec must be TriggeredCaptureSpec")
        source_schema = spec.capture.capture.capture_contract.dataset_schema
        mot_intensity_schema(request, source_schema)
        self._request = request
        self._spec = spec
        self._source_schema = source_schema

    @property
    def request(self) -> MotFieldRequest:
        return self._request

    @property
    def source_schema(self) -> DatasetSchema:
        return self._source_schema

    def compile_capture_plan(
        self,
        repository: CaptureRepository,
        *,
        preview: ExactDatasetPreviewPort | None = None,
    ) -> RunPlan:
        """Compile the capture into the repository-backed FINAL Run boundary."""

        if type(repository) is not CaptureRepository:
            raise TypeError("repository must be CaptureRepository")
        return compile_capture_artifact_pipeline(
            self._spec,
            repository,
            exact_preview=preview,
        )


def prepare_mot_field_acquisition(
    request: MotFieldRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
) -> PreparedMotFieldAcquisition:
    """Bind one MOT request without starting either hardware device."""

    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    program = request.program
    binding = bind_triggered_camera_acquisition(
        pulse_port,
        camera_port,
        pulse_document=program.document,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channel=request.trigger_channel,
        layout=TriggeredCameraLayout(
            repeat_axis=_MOT_REPEAT_AXIS,
            readout_event_axis_id=_MOT_READOUT_EVENT_AXIS_ID,
            readout_events_per_repeat=1,
            scan_point_table=program.point_table,
            scan_grid_topology=program.grid_topology,
        ),
    )
    if binding.expected_frames != program.point_table.row_count:
        raise RuntimeError("MOT pulse trigger count differs from its frozen grid")
    pipeline = MinimalPipelineSpec(
        "Optimize MOT field",
        binding.capture,
        BlockId(f"mot-field-source-{binding.compiled_artifact.fingerprint[:20]}"),
    )
    spec = TriggeredCaptureSpec(
        pipeline,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    return PreparedMotFieldAcquisition(request, spec)


__all__ = [
    "PreparedMotFieldAcquisition",
    "prepare_mot_field_acquisition",
]
