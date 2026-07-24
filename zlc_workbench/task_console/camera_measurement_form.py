"""TaskConsole form projection for the one Camera Measurement."""

from __future__ import annotations

from typing import Mapping

from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.camera_measurement import (
    CAMERA_MEASUREMENT_ROLE_ORDER,
    DEFAULT_CAMERA_FRAMES_PER_CYCLE,
    DEFAULT_CAMERA_MEASUREMENT_REPEAT,
    DEFAULT_CAMERA_MEASUREMENT_ROLE,
    MINIMUM_CAMERA_FRAMES_PER_CYCLE,
    MINIMUM_CAMERA_MEASUREMENT_REPEAT,
    CameraMeasurementRequest,
)
from zlc_storage import canonical_text

__all__ = [
    "build_camera_measurement_request",
    "camera_measurement_form",
    "camera_measurement_roles",
]


def camera_measurement_roles(installed_roles) -> tuple[str, ...]:
    """Return supported Camera roles in the established visible order."""

    installed = set(installed_roles)
    return tuple(
        role for role in CAMERA_MEASUREMENT_ROLE_ORDER if role in installed
    )


def _camera_role_field(camera_roles: tuple[str, ...]) -> FormFieldProps:
    roles = tuple(camera_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("camera roles must be unique")
    for role in roles:
        canonical_text(role, "camera role")
    return FormFieldProps(
        "camera_role",
        "choice",
        "Camera",
        default=(
            DEFAULT_CAMERA_MEASUREMENT_ROLE
            if DEFAULT_CAMERA_MEASUREMENT_ROLE in roles
            else roles[0]
            if roles
            else None
        ),
        required=True,
        choices=tuple(FormChoice(role, role) for role in roles),
        description="Frozen camera role from the current installation",
        unavailable_reason=(
            "Camera Measurement requires an installed camera role"
            if not roles
            else ""
        ),
    )


def camera_measurement_form(camera_roles: tuple[str, ...]) -> FormSpec:
    return FormSpec((
        _camera_role_field(camera_roles),
        FormFieldProps(
            "frames_per_cycle",
            "int",
            "Frames per cycle",
            default=DEFAULT_CAMERA_FRAMES_PER_CYCLE,
            minimum=MINIMUM_CAMERA_FRAMES_PER_CYCLE,
            maximum=1_000_000,
            required=True,
            allow_blank=False,
            description=(
                "Ordered camera frames retained on an explicit READOUT_EVENT axis"
            ),
        ),
        FormFieldProps(
            "repeat",
            "int",
            "Repeat",
            default=DEFAULT_CAMERA_MEASUREMENT_REPEAT,
            minimum=MINIMUM_CAMERA_MEASUREMENT_REPEAT,
            maximum=1_000_000,
            required=True,
            allow_blank=False,
            description=(
                "0 keeps this installed camera live; a positive value performs "
                "that many exact finite capture cycles"
            ),
        ),
    ))


def build_camera_measurement_request(builder, values: Mapping[str, object]):
    """Freeze Camera form values through its explicit application builder."""

    if not callable(builder):
        raise TypeError("Camera Measurement request builder must be callable")

    selected = {}
    for key in ("camera_role", "frames_per_cycle", "repeat"):
        value = values.get(key)
        if value not in (None, ""):
            selected[key] = value
    request = builder(**selected)
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError(
            "Camera Measurement request builder returned another value"
        )
    return request
