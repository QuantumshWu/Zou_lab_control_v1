"""TaskConsole FormSpec and explicit-source binding for Occupancy."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.readout.calibration import ReadoutModelKind

from .occupancy_binding import OccupancyBindingIntent

__all__ = ["build_occupancy_binding", "occupancy_form"]


_CALIBRATION_SOURCES = ("Task output", "Saved calibration")
_DEFAULT_CALIBRATION_REF_PATH = "_output/calibrations/calibration_ref.json"
_READOUT_METHODS = (
    ("Calibration default", None),
    ("box", ReadoutModelKind.BOX),
    ("per-site PSF", ReadoutModelKind.PER_SITE_PSF),
    ("uniform PSF", ReadoutModelKind.UNIFORM_PSF),
)

if {
    kind for _label, kind in _READOUT_METHODS if kind is not None
} != set(ReadoutModelKind):
    raise RuntimeError("occupancy form labels do not cover ReadoutModelKind")


def _parse_readout_method(value: object) -> ReadoutModelKind | None:
    label = "Calibration default" if value in (None, "") else str(value)
    for candidate, kind in _READOUT_METHODS:
        if label == candidate:
            return kind
    raise ValueError(f"unknown occupancy Readout method {label!r}")


def occupancy_form() -> FormSpec:
    return FormSpec((
        FormFieldProps(
            "calibration_source",
            "choice",
            "Calibration source",
            default="Task output",
            required=True,
            choices=tuple(FormChoice(value, value) for value in _CALIBRATION_SOURCES),
            description=(
                "Use an exact TaskConsole output or an explicitly chosen saved pointer."
            ),
        ),
        FormFieldProps(
            "calibration_task",
            "signal",
            "Calibration task",
            description=(
                "FINAL calibration output of a successful Calibrate readout "
                "Task row; used only when Calibration source is Task output"
            ),
        ),
        FormFieldProps(
            "calibration_file",
            "path",
            "Saved calibration",
            default=_DEFAULT_CALIBRATION_REF_PATH,
            path_mode="file",
            base_dir="_output/calibrations",
            file_filter=(
                "Calibration pointer (calibration_ref.json);;JSON files (*.json)"
            ),
            description=(
                "Exact calibration_ref.json produced by a successful calibration; "
                "used only when Calibration source is Saved calibration"
            ),
        ),
        FormFieldProps(
            "camera_frame",
            "signal",
            "Frame source",
            required=True,
            description=(
                "Current-frame output of an already-running Camera Measurement. "
                "Live and finite Camera runs both publish this typed view; "
                "Occupancy never guesses a current cell from a full dataset, "
                "starts, or reconfigures the Camera"
            ),
        ),
        FormFieldProps(
            "readout_method",
            "choice",
            "Readout method",
            default="Calibration default",
            required=True,
            choices=tuple(FormChoice(label, label) for label, _kind in _READOUT_METHODS),
            description="Select one model already stored in the admitted calibration",
        ),
    ))


def build_occupancy_binding(
    values: Mapping[str, object],
) -> OccupancyBindingIntent:
    camera_frame = values.get("camera_frame")
    if not isinstance(camera_frame, str) or not camera_frame.strip():
        raise ValueError("occupancy requires a running Camera frame output")
    source = str(values.get("calibration_source", "Task output"))
    if source not in _CALIBRATION_SOURCES:
        raise ValueError("unknown occupancy calibration source")
    calibration_signal = None
    calibration_ref_path = None
    if source == "Task output":
        value = values.get("calibration_task")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "occupancy requires a successful Calibration task output"
            )
        calibration_signal = value.strip()
    else:
        value = values.get("calibration_file", _DEFAULT_CALIBRATION_REF_PATH)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("occupancy requires an explicit saved calibration file")
        calibration_ref_path = str(Path(value).expanduser().resolve())
    return OccupancyBindingIntent(
        camera_frame_signal=camera_frame.strip(),
        calibration_signal=calibration_signal,
        calibration_ref_path=calibration_ref_path,
        model_kind=_parse_readout_method(
            values.get("readout_method", "Calibration default")
        ),
    )
