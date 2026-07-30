"""Prepared coupled Camera + Sequencer acquisition for MOT-field optimization.

The MOT capability owns only its experiment-specific binding: three declared
coil axes, one Camera event per grid cell, and one autonomous FPGA FIRE.  Exact
camera transport, hardware coordination, cancellation, and preview remain the
single generic owners in ``capture`` and ``devices``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from zlc_data import AxisId, AxisSpec, BlockId, DatasetSchema, REPEAT
from zlc_neutral_atom.capture.artifact import (
    CaptureArtifact,
    load_capture_artifact,
)
from zlc_neutral_atom.capture.application import (
    PreparedFiniteCapture,
    bind_finite_capture_spec,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
)
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
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
    """One fully bound autonomous MOT acquisition over generic Capture."""

    __slots__ = (
        "_capture",
        "_captures_root",
        "_request",
        "_source_schema",
    )

    def __init__(
        self,
        request: MotFieldRequest,
        capture: PreparedFiniteCapture,
        *,
        captures_root: Path,
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(capture, PreparedFiniteCapture):
            raise TypeError("capture must be PreparedFiniteCapture")
        if not isinstance(captures_root, Path):
            raise TypeError("captures_root must be Path")
        source_schema = capture.descriptor.output_schema
        mot_intensity_schema(request, source_schema)
        self._request = request
        self._capture = capture
        self._source_schema = source_schema
        self._captures_root = captures_root.expanduser().resolve()

    @property
    def request(self) -> MotFieldRequest:
        return self._request

    @property
    def source_schema(self) -> DatasetSchema:
        return self._source_schema

    def start(
        self,
        exact_preview: ExactDatasetPreviewPort | None = None,
    ) -> RunHandle:
        """Delegate start and lifecycle ownership to generic Capture."""

        return self._capture.start(exact_preview=exact_preview)

    def load_capture(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        """Load one visible direct-output Capture for MOT analysis."""

        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        artifact = load_capture_artifact(
            self._captures_root,
            reference,
            materialize=True,
        )
        if artifact.pulse_evidence is None:
            raise ValueError("MOT requires pulse-associated Capture evidence")
        if (
            artifact.pulse_evidence.trigger_channel
            != self._capture.descriptor.trigger_channel
        ):
            raise ValueError("MOT Capture trigger differs from the prepared acquisition")
        if (
            artifact.pulse_evidence.compiled_artifact.fingerprint
            != self._capture.descriptor.compiled_pulse_digest
        ):
            raise ValueError("MOT Capture pulse differs from the prepared acquisition")
        if (
            artifact.frame_source.schema.fingerprint
            != self._source_schema.fingerprint
        ):
            raise ValueError("MOT Capture schema differs from the prepared acquisition")
        return artifact


def prepare_mot_field_acquisition(
    request: MotFieldRequest,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    captures_root: Path,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedMotFieldAcquisition:
    """Bind one MOT request without starting either hardware device."""

    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    if not isinstance(captures_root, Path):
        raise TypeError("captures_root must be Path")
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
            scan_point_table=program.point_table,
            scan_grid_topology=program.grid_topology,
        ),
    )
    if binding.expected_frames != program.point_table.row_count:
        raise RuntimeError("MOT pulse trigger count differs from its frozen grid")
    triggered, descriptor = bind_finite_capture_spec(
        binding=binding,
        block_id=BlockId(
            f"mot-field-source-{binding.compiled_artifact.fingerprint[:20]}"
        ),
        camera_ref=request.camera_ref,
        sequencer_ref=request.sequencer_ref,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        name_prefix="Optimize MOT field",
    )
    capture = PreparedFiniteCapture(
        triggered,
        captures_root,
        start_run,
        descriptor,
    )
    return PreparedMotFieldAcquisition(
        request,
        capture,
        captures_root=captures_root,
    )


__all__ = [
    "PreparedMotFieldAcquisition",
    "prepare_mot_field_acquisition",
]
