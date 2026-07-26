"""Prepared coupled Camera + Sequencer acquisition for MOT-field optimization.

The MOT capability owns only its experiment-specific binding: three declared
coil axes, one Camera event per grid cell, and one autonomous FPGA FIRE.  Exact
camera transport, hardware coordination, cancellation, and preview remain the
single generic owners in ``capture`` and ``devices``.
"""

from __future__ import annotations

import threading
from typing import Callable

from zlc_data import AxisId, AxisSpec, BlockId, DatasetSchema, REPEAT
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.capture.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.capture.triggered import (
    TriggeredCaptureSpec,
    TriggeredPipelineResult,
    compile_triggered_pipeline,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    notify_preview_failure,
)
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    RunHandle,
    RunId,
    RunPlan,
    RunSnapshot,
)
from zlc_pulse import PulseExecutionForm

from .mot_field import (
    MotFieldAcquisitionResult,
    MotFieldRequest,
    mot_field_source_identity,
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


class MotFieldAcquisitionHandle:
    """Run-like handle whose result is the MOT capability's exact source type."""

    __slots__ = ("_handle", "_lock", "_request", "_result")

    def __init__(self, request: MotFieldRequest, handle: RunHandle) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(handle, RunHandle):
            raise TypeError("handle must be RunHandle")
        self._request = request
        self._handle = handle
        self._lock = threading.Lock()
        self._result: MotFieldAcquisitionResult | None = None

    @property
    def run_id(self) -> RunId:
        return self._handle.run_id

    def snapshot(self) -> RunSnapshot:
        return self._handle.snapshot()

    def cancel(self, reason: str = "user requested stop") -> CancelOutcome:
        return self._handle.cancel(reason)

    def wait(self, timeout: float | None = None) -> RunSnapshot:
        return self._handle.wait(timeout)

    def result(self, timeout: float | None = None) -> MotFieldAcquisitionResult:
        with self._lock:
            cached = self._result
        if cached is not None:
            return cached
        pipeline = self._handle.result(timeout)
        if type(pipeline) is not TriggeredPipelineResult:
            raise TypeError("MOT acquisition Run returned another result type")
        dataset = pipeline.capture.dataset
        # Revalidate the MOT-specific axes/Camera frame contract at the seam;
        # TriggeredPipelineResult has already proved pulse/camera exactness.
        mot_intensity_schema(self._request, dataset.block.schema)
        result = MotFieldAcquisitionResult(
            dataset.snapshot,
            dataset.provenance,
            mot_field_source_identity(dataset.snapshot, dataset.provenance),
        )
        with self._lock:
            if self._result is None:
                self._result = result
            return self._result


class PreparedMotFieldAcquisition:
    """One-shot autonomous MOT acquisition with an optional exact live grid."""

    __slots__ = (
        "_lock",
        "_request",
        "_source_schema",
        "_spec",
        "_start_run",
        "_started",
    )

    def __init__(
        self,
        request: MotFieldRequest,
        spec: TriggeredCaptureSpec,
        start_run: Callable[[RunPlan], RunHandle],
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(spec, TriggeredCaptureSpec):
            raise TypeError("spec must be TriggeredCaptureSpec")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        source_schema = spec.capture.measurement.capture_contract.dataset_schema
        mot_intensity_schema(request, source_schema)
        self._request = request
        self._spec = spec
        self._source_schema = source_schema
        self._start_run = start_run
        self._lock = threading.Lock()
        self._started = False

    @property
    def request(self) -> MotFieldRequest:
        return self._request

    @property
    def source_schema(self) -> DatasetSchema:
        return self._source_schema

    def start(
        self,
        preview: ExactDatasetPreviewPort | None = None,
    ) -> MotFieldAcquisitionHandle:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedMotFieldAcquisition is one-shot")
            self._started = True
        try:
            plan = compile_triggered_pipeline(self._spec, exact_preview=preview)
            handle = self._start_run(plan)
            return MotFieldAcquisitionHandle(self._request, handle)
        except BaseException as error:
            notify_preview_failure(preview, error)
            raise


def prepare_mot_field_acquisition(
    request: MotFieldRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedMotFieldAcquisition:
    """Bind one MOT request without starting either hardware device."""

    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    if not callable(start_run):
        raise TypeError("start_run must be callable")
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
            scan_axes=program.point_axes,
            scan_point_layout=program.point_layout,
        ),
    )
    if binding.expected_frames != program.point_layout.storage_size:
        raise RuntimeError("MOT pulse trigger count differs from its frozen grid")
    pipeline = MinimalPipelineSpec(
        "Optimize MOT field",
        binding.measurement,
        BlockId(f"mot-field-source-{binding.compiled_artifact.fingerprint[:20]}"),
    )
    spec = TriggeredCaptureSpec(
        pipeline,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    return PreparedMotFieldAcquisition(request, spec, start_run)


__all__ = [
    "MotFieldAcquisitionHandle",
    "PreparedMotFieldAcquisition",
    "prepare_mot_field_acquisition",
]
