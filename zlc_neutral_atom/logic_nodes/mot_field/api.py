"""Public Experiment API owned by MOT-field optimization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zlc_pulse import PulseDocument

from .application import PreparedMotFieldAcquisition
from .mot_field import (
    DEFAULT_MOT_FIELD_CAMERA_ROLE,
    DEFAULT_MOT_FIELD_CENTER_CODE,
    DEFAULT_MOT_FIELD_POINTS,
    DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
    DEFAULT_MOT_FIELD_SPAN_CODE,
    MotFieldRequest,
    build_mot_scan_program,
)
from .mot_field_task import (
    DEFAULT_MOT_FIELD_PULSE_PATH,
    MotFieldTaskIntent,
    PreparedMotFieldTask,
)


class MotFieldApi:
    __slots__ = (
        "_load_pulse",
        "_prepare_acquisition",
        "_prepare_task",
        "_resolve_camera_ref",
        "_resolve_sequencer_ref",
    )

    def __init__(
        self,
        *,
        load_pulse: Callable[[PulseDocument | str | Path], PulseDocument],
        resolve_camera_ref: Callable[[str | None], object],
        resolve_sequencer_ref: Callable[[str | None], object],
        prepare_acquisition: Callable[[MotFieldRequest], PreparedMotFieldAcquisition],
        prepare_task: Callable[[MotFieldTaskIntent, object], PreparedMotFieldTask],
    ) -> None:
        operations = (
            load_pulse,
            resolve_camera_ref,
            resolve_sequencer_ref,
            prepare_acquisition,
            prepare_task,
        )
        if any(not callable(operation) for operation in operations):
            raise TypeError("MOT-field API operations must be callable")
        self._load_pulse = load_pulse
        self._resolve_camera_ref = resolve_camera_ref
        self._resolve_sequencer_ref = resolve_sequencer_ref
        self._prepare_acquisition = prepare_acquisition
        self._prepare_task = prepare_task

    def mot_field_request(
        self,
        pulse: PulseDocument | str | Path = DEFAULT_MOT_FIELD_PULSE_PATH,
        *,
        center_x: float = DEFAULT_MOT_FIELD_CENTER_CODE,
        center_y: float = DEFAULT_MOT_FIELD_CENTER_CODE,
        center_z: float = DEFAULT_MOT_FIELD_CENTER_CODE,
        span: float = DEFAULT_MOT_FIELD_SPAN_CODE,
        points: int = DEFAULT_MOT_FIELD_POINTS,
        roi_cx: float | None = None,
        roi_cy: float | None = None,
        roi_radius: float = DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> MotFieldRequest:
        selected_camera = (
            DEFAULT_MOT_FIELD_CAMERA_ROLE if camera_role is None else camera_role
        )
        return MotFieldRequest(
            program=build_mot_scan_program(
                self._load_pulse(pulse),
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                span=span,
                points=points,
            ),
            camera_ref=self._resolve_camera_ref(selected_camera),
            sequencer_ref=self._resolve_sequencer_ref(sequencer_role),
            roi_cx=roi_cx,
            roi_cy=roi_cy,
            roi_radius=roi_radius,
            trigger_channel=trigger_channel,
        )

    def prepare_mot_field_acquisition(
        self,
        request: MotFieldRequest,
    ) -> PreparedMotFieldAcquisition:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        return self._prepare_acquisition(request)

    def prepare_mot_field_task(
        self,
        intent: MotFieldTaskIntent,
    ) -> PreparedMotFieldTask:
        return self._prepare_task(intent, self)


__all__ = ["MotFieldApi"]
