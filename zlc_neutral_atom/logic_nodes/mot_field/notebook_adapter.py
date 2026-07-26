"""Notebook surface owned by MOT-field optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from zlc_neutral_atom.installation import DeviceRef
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
    MotFieldTaskDependencies,
    MotFieldTaskIntent,
    PreparedMotFieldTask,
)


class MotFieldNotebookHost(Protocol):
    def load_mot_field_pulse(
        self,
        value: PulseDocument | str | Path,
    ) -> PulseDocument: ...

    def resolve_mot_camera_ref(self, requested: str | None) -> DeviceRef: ...

    def resolve_mot_sequencer_ref(self, requested: str | None) -> DeviceRef: ...

    def bind_mot_field_acquisition(
        self,
        request: MotFieldRequest,
    ) -> PreparedMotFieldAcquisition: ...

    def bind_mot_field_task(
        self,
        intent: MotFieldTaskIntent,
        dependencies: MotFieldTaskDependencies,
    ) -> PreparedMotFieldTask: ...


class MotFieldNotebookAdapter:
    __slots__ = ()

    @property
    def _mot_field_notebook_host(self) -> MotFieldNotebookHost:
        raise NotImplementedError

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
        host = self._mot_field_notebook_host
        selected_camera = (
            DEFAULT_MOT_FIELD_CAMERA_ROLE if camera_role is None else camera_role
        )
        return MotFieldRequest(
            program=build_mot_scan_program(
                host.load_mot_field_pulse(pulse),
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                span=span,
                points=points,
            ),
            camera_ref=host.resolve_mot_camera_ref(selected_camera),
            sequencer_ref=host.resolve_mot_sequencer_ref(sequencer_role),
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
        return self._mot_field_notebook_host.bind_mot_field_acquisition(request)

    def prepare_mot_field_task(
        self,
        intent: MotFieldTaskIntent,
    ) -> PreparedMotFieldTask:
        return self._mot_field_notebook_host.bind_mot_field_task(intent, self)


__all__ = ["MotFieldNotebookAdapter", "MotFieldNotebookHost"]
