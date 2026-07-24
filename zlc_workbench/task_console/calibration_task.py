"""TaskConsole presentation for the calibration application intent."""

from __future__ import annotations

from typing import Mapping

from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.readout.calibration_task import (
    CALIBRATION_SOURCE_MODES,
    CALIBRATION_THRESHOLD_METHODS,
    CalibrationTaskIntent,
    DEFAULT_CALIBRATION_CAMERA_ROLE,
    DEFAULT_CALIBRATION_FOLDER,
    DEFAULT_CALIBRATION_PULSE_PATH,
    DEFAULT_CALIBRATION_READOUT_EXPOSURE_S,
    DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S,
    DEFAULT_CALIBRATION_ROI_RADIUS,
    DEFAULT_CALIBRATION_SAVE_FRAMES,
    DEFAULT_CALIBRATION_SOURCE_MODE,
    DEFAULT_CALIBRATION_THRESHOLD_FRAMES,
    DEFAULT_CALIBRATION_THRESHOLD_METHOD,
    MINIMUM_CALIBRATION_ROI_RADIUS,
    MINIMUM_CALIBRATION_THRESHOLD_FRAMES,
)


def calibration_task_params(
    camera_roles: tuple[str, ...],
) -> FormSpec:
    """Return the Main calibration controls for the TaskConsole form."""

    choices = tuple(str(role) for role in camera_roles)
    camera_default = (
        DEFAULT_CALIBRATION_CAMERA_ROLE
        if DEFAULT_CALIBRATION_CAMERA_ROLE in choices
        else choices[0]
        if choices
        else None
    )
    return FormSpec((
        FormFieldProps(
            "source_mode",
            "choice",
            "source",
            default=DEFAULT_CALIBRATION_SOURCE_MODE,
            required=True,
            choices=tuple(FormChoice(value, value) for value in CALIBRATION_SOURCE_MODES),
            description="Acquire live frames now or calibrate from saved raw frames.",
        ),
        FormFieldProps(
            "folder",
            "path",
            "folder",
            default=DEFAULT_CALIBRATION_FOLDER,
            required=True,
            path_mode="dir",
            base_dir=DEFAULT_CALIBRATION_FOLDER,
            description=(
                "The one calibration directory: live writes the result and optional "
                "raw frames here; saved frames reads this directory's frames/ export."
            ),
        ),
        FormFieldProps(
            "save_frames",
            "bool",
            "save frames (live)",
            default=DEFAULT_CALIBRATION_SAVE_FRAMES,
            description="Keep raw live frames so the same acquisition can be recalibrated.",
        ),
        FormFieldProps(
            "pulse",
            "path",
            "pulse template",
            default=DEFAULT_CALIBRATION_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="zlc_neutral_atom/assets",
            file_filter="Pulse program (*.json);;All files (*)",
            description="Live only: imaging pulse used for each long-short-long bracket.",
        ),
        FormFieldProps(
            "threshold_method",
            "choice",
            "threshold",
            default=DEFAULT_CALIBRATION_THRESHOLD_METHOD,
            required=True,
            choices=tuple(FormChoice(value, value) for value in CALIBRATION_THRESHOLD_METHODS),
            description="Per-site threshold estimator.",
        ),
        FormFieldProps(
            "reference_exposure_s",
            "float",
            "reference exposure (long)",
            default=DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S,
            unit="s",
            minimum=0.0,
            maximum=10.0,
            required=True,
            allow_blank=False,
            description="Live only: long exposure for the two outer reference frames.",
        ),
        FormFieldProps(
            "readout_exposure_s",
            "float",
            "readout exposure (short)",
            default=DEFAULT_CALIBRATION_READOUT_EXPOSURE_S,
            unit="s",
            minimum=0.0,
            maximum=10.0,
            required=True,
            allow_blank=False,
            description="Live only: short exposure for the middle readout frame.",
        ),
        FormFieldProps(
            "threshold_frames",
            "int",
            "reference brackets",
            default=DEFAULT_CALIBRATION_THRESHOLD_FRAMES,
            minimum=MINIMUM_CALIBRATION_THRESHOLD_FRAMES,
            maximum=20_000,
            required=True,
            allow_blank=False,
            description="Number of long-short-long calibration shots.",
        ),
        FormFieldProps(
            "roi_radius",
            "int",
            "ROI radius",
            default=DEFAULT_CALIBRATION_ROI_RADIUS,
            unit="px",
            minimum=MINIMUM_CALIBRATION_ROI_RADIUS,
            maximum=64,
            required=True,
            allow_blank=False,
            description="Per-site square ROI half-width in pixels.",
        ),
        FormFieldProps(
            "camera_role",
            "choice",
            "Camera",
            default=camera_default,
            required=True,
            choices=tuple(FormChoice(value, value) for value in choices),
            description="Camera used for live calibration acquisition.",
            unavailable_reason=(
                "Calibrate readout requires an installed camera role with a "
                "site-map acquisition profile"
                if not choices
                else ""
            ),
        ),
    ))


def build_calibration_task_intent(
    values: Mapping[str, object],
) -> CalibrationTaskIntent:
    """Freeze the visible form into the neutral application intent."""

    camera_role = values.get("camera_role")
    if camera_role is None:
        raise RuntimeError(
            "Calibrate readout requires an installed camera role with a "
            "site-map acquisition profile"
        )

    return CalibrationTaskIntent(
        source_mode=str(
            values.get("source_mode", DEFAULT_CALIBRATION_SOURCE_MODE)
        ),
        folder=str(values.get("folder", DEFAULT_CALIBRATION_FOLDER)),
        save_frames=values.get("save_frames", DEFAULT_CALIBRATION_SAVE_FRAMES),
        pulse=str(values.get("pulse", DEFAULT_CALIBRATION_PULSE_PATH)),
        threshold_method=str(
            values.get(
                "threshold_method",
                DEFAULT_CALIBRATION_THRESHOLD_METHOD,
            )
        ),
        reference_exposure_s=float(
            values.get(
                "reference_exposure_s",
                DEFAULT_CALIBRATION_REFERENCE_EXPOSURE_S,
            )
        ),
        readout_exposure_s=float(
            values.get(
                "readout_exposure_s",
                DEFAULT_CALIBRATION_READOUT_EXPOSURE_S,
            )
        ),
        threshold_frames=values.get(
            "threshold_frames",
            DEFAULT_CALIBRATION_THRESHOLD_FRAMES,
        ),
        roi_radius=values.get(
            "roi_radius",
            DEFAULT_CALIBRATION_ROI_RADIUS,
        ),
        camera_role=str(camera_role),
    )


__all__ = [
    "build_calibration_task_intent",
    "calibration_task_params",
]
