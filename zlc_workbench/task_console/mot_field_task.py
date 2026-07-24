"""TaskConsole presentation for the typed MOT-field application intent."""

from __future__ import annotations

from typing import Mapping

from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.mot_field import (
    DEFAULT_MOT_FIELD_CAMERA_ROLE,
    DEFAULT_MOT_FIELD_CENTER_CODE,
    DEFAULT_MOT_FIELD_POINTS,
    DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
    DEFAULT_MOT_FIELD_SPAN_CODE,
    MINIMUM_MOT_FIELD_POINTS,
)
from zlc_neutral_atom.mot_field_task import (
    DEFAULT_MOT_FIELD_REPORT_FOLDER,
    MotFieldTaskIntent,
)
from zlc_neutral_atom.pulse_programs import DEFAULT_MOT_FIELD_PULSE_PATH


def mot_field_params(
    camera_roles: tuple[str, ...],
) -> FormSpec:
    """Return the familiar one-click MOT controls, with no generic timeout."""

    camera_roles = tuple(
        role
        for role in camera_roles
        if role == DEFAULT_MOT_FIELD_CAMERA_ROLE
    )
    return FormSpec((
        FormFieldProps(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_MOT_FIELD_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
            description="Autonomous SCAN_SLOT template declaring da_x, da_y and da_z",
        ),
        FormFieldProps(
            "center_x",
            "float",
            "Bx centre",
            default=DEFAULT_MOT_FIELD_CENTER_CODE,
            unit="code",
            minimum=-512.0,
            maximum=511.0,
            required=True,
            allow_blank=False,
        ),
        FormFieldProps(
            "center_y",
            "float",
            "By centre",
            default=DEFAULT_MOT_FIELD_CENTER_CODE,
            unit="code",
            minimum=-512.0,
            maximum=511.0,
            required=True,
            allow_blank=False,
        ),
        FormFieldProps(
            "center_z",
            "float",
            "Bz centre",
            default=DEFAULT_MOT_FIELD_CENTER_CODE,
            unit="code",
            minimum=-512.0,
            maximum=511.0,
            required=True,
            allow_blank=False,
        ),
        FormFieldProps(
            "span",
            "float",
            "Span (+/-)",
            default=DEFAULT_MOT_FIELD_SPAN_CODE,
            unit="code",
            minimum=0.0,
            maximum=511.0,
            required=True,
            allow_blank=False,
        ),
        FormFieldProps(
            "points",
            "int",
            "Points per axis",
            default=DEFAULT_MOT_FIELD_POINTS,
            minimum=MINIMUM_MOT_FIELD_POINTS,
            maximum=15,
            required=True,
            allow_blank=False,
            description="Total autonomous scan cells are points^3",
        ),
        FormFieldProps(
            "roi_cx",
            "float",
            "ROI centre x",
            default=None,
            unit="px",
            minimum=0.0,
            maximum=1_000_000.0,
            required=False,
            allow_blank=True,
            description="Blank uses the frame centre; 0 is the left pixel coordinate",
        ),
        FormFieldProps(
            "roi_cy",
            "float",
            "ROI centre y",
            default=None,
            unit="px",
            minimum=0.0,
            maximum=1_000_000.0,
            required=False,
            allow_blank=True,
            description="Blank uses the frame centre; 0 is the top pixel coordinate",
        ),
        FormFieldProps(
            "roi_radius",
            "float",
            "ROI radius",
            default=DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
            unit="px",
            minimum=0.1,
            maximum=1_000_000.0,
            required=True,
            allow_blank=False,
            description="The 1x..2x annulus supplies the local background",
        ),
        FormFieldProps(
            "folder",
            "path",
            "Report folder",
            default=DEFAULT_MOT_FIELD_REPORT_FOLDER,
            required=True,
            path_mode="dir",
            description=(
                "Raw intensity block, exact Bx/By/Bz axes, and refined "
                "optimum are written to mot_field_scan.npz"
            ),
        ),
        FormFieldProps(
            "camera_role",
            "choice",
            "Camera role",
            default=camera_roles[0] if camera_roles else None,
            required=True,
            choices=tuple(FormChoice(value, value) for value in camera_roles),
            description=(
                "Must be an external-trigger-capable camera physically observing "
                "the MOT; a free-running monitor cannot prove point association"
            ),
            unavailable_reason=(
                "MOT field requires the installation's external-trigger-capable "
                "mot_camera role"
                if not camera_roles
                else ""
            ),
        ),
    ))


def build_mot_field_intent(
    values: Mapping[str, object],
) -> MotFieldTaskIntent:
    """Freeze the visible form into the neutral application intent."""

    camera_role = values.get("camera_role")
    if camera_role is None:
        raise RuntimeError(
            "MOT field requires the installation's external-trigger-capable "
            "mot_camera role"
        )

    def optional_pixel(key: str) -> float | None:
        value = values.get(key)
        return None if value in (None, "") else float(value)

    return MotFieldTaskIntent(
        pulse=str(values.get("pulse") or DEFAULT_MOT_FIELD_PULSE_PATH),
        center_x=float(
            values.get("center_x", DEFAULT_MOT_FIELD_CENTER_CODE)
        ),
        center_y=float(
            values.get("center_y", DEFAULT_MOT_FIELD_CENTER_CODE)
        ),
        center_z=float(
            values.get("center_z", DEFAULT_MOT_FIELD_CENTER_CODE)
        ),
        span=float(values.get("span", DEFAULT_MOT_FIELD_SPAN_CODE)),
        points=values.get("points", DEFAULT_MOT_FIELD_POINTS),
        roi_cx=optional_pixel("roi_cx"),
        roi_cy=optional_pixel("roi_cy"),
        roi_radius=float(
            values.get("roi_radius", DEFAULT_MOT_FIELD_ROI_RADIUS_PX)
        ),
        folder=str(
            values.get("folder", DEFAULT_MOT_FIELD_REPORT_FOLDER)
        ),
        camera_role=str(camera_role),
    )


__all__ = [
    "build_mot_field_intent",
    "mot_field_params",
]
