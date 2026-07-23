"""TaskConsole forms for the three coupled readout measurements.

This module owns presentation only.  The notebook facade remains the owner of
installation role resolution and of the typed domain requests; the TaskConsole
composition root resolves the selected calibration signal to one explicit
``CalibrationArtifactRef`` before calling the builders below.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping

import numpy as np

from zlc_data.param_decl import ParamDecl
from zlc_neutral_atom.pulse_programs import (
    DEFAULT_PROBE_PULSE_PATH,
    DEFAULT_RELEASE_RECAPTURE_PULSE_PATH,
)
from zlc_neutral_atom.readout.calibration import ReadoutModelKind
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef


_MODEL_CHOICES = (
    "auto",
    *(kind.value for kind in ReadoutModelKind),
)


def _preferred(roles: tuple[str, ...], *candidates: str) -> str:
    if not roles:
        raise ValueError("measurement requires at least one configured device role")
    for candidate in candidates:
        if candidate in roles:
            return candidate
    return roles[0]


def _common_params(
    camera_roles: tuple[str, ...],
    sequencer_roles: tuple[str, ...],
) -> tuple[ParamDecl, ...]:
    cameras = tuple(camera_roles)
    sequencers = tuple(sequencer_roles)
    return (
        ParamDecl(
            "calibration",
            "Calibration",
            "signal",
            required=True,
            tooltip=(
                "FINAL calibration output of a successful Calibrate readout "
                "Task; the artifact reference is frozen into this request"
            ),
        ),
        ParamDecl(
            "model_kind",
            "Readout model",
            "choice",
            default="auto",
            choices=_MODEL_CHOICES,
            tooltip="Auto uses the calibration artifact's declared default model",
        ),
        ParamDecl(
            "camera_role",
            "Camera role",
            "choice",
            default=_preferred(cameras, "camera", "readout"),
            required=True,
            choices=cameras,
        ),
        ParamDecl(
            "sequencer_role",
            "Sequencer role",
            "choice",
            default=_preferred(sequencers, "sequencer"),
            required=True,
            choices=sequencers,
        ),
        ParamDecl(
            "trigger_channel",
            "Trigger channel",
            "text",
            default=None,
            required=False,
            tooltip="Leave blank to use the camera capability's declared trigger",
        ),
    )


def temperature_release_recapture_params(
    camera_roles: tuple[str, ...],
    sequencer_roles: tuple[str, ...],
) -> tuple[ParamDecl, ...]:
    return (
        ParamDecl(
            "pulse",
            "Pulse template",
            "path",
            default=DEFAULT_RELEASE_RECAPTURE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
            tooltip=(
                "Autonomous two-readout pulse with the declared t_off SCAN_SLOT"
            ),
        ),
        ParamDecl(
            "t_off",
            "Trap-off time",
            "axis_range",
            default=(0.02, 300.0, 13),
            unit="us",
            lo=0.02,
            hi=10_000.0,
            required=True,
            tooltip=(
                "The bundled 50 MHz pulse target requires at least one "
                "20 ns clock tick; the selected PulseDocument is validated "
                "again when the request is frozen"
            ),
        ),
        ParamDecl(
            "shots",
            "Shots / point",
            "int",
            default=16,
            lo=1,
            hi=100_000,
            required=True,
            optional=False,
        ),
        ParamDecl(
            "per_site",
            "Per-site survival",
            "bool",
            default=False,
        ),
        *_common_params(camera_roles, sequencer_roles),
    )


def readout_duration_fidelity_params(
    camera_roles: tuple[str, ...],
    sequencer_roles: tuple[str, ...],
) -> tuple[ParamDecl, ...]:
    return (
        ParamDecl(
            "pulse",
            "Pulse template",
            "path",
            default=DEFAULT_PROBE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
        ),
        ParamDecl(
            "duration",
            "Detection time",
            "axis_range",
            default=(2.0, 20_000.0, 11),
            unit="us",
            lo=0.02,
            hi=1_000_000.0,
            required=True,
        ),
        ParamDecl(
            "shots",
            "Shots / point",
            "int",
            default=60,
            lo=1,
            hi=100_000,
            required=True,
            optional=False,
        ),
        ParamDecl(
            "site",
            "Site (optional)",
            "int",
            default=None,
            lo=0,
            hi=100_000,
            required=False,
            optional=True,
        ),
        *_common_params(camera_roles, sequencer_roles),
    )


def grey_molasses_detuning_params(
    camera_roles: tuple[str, ...],
    sequencer_roles: tuple[str, ...],
    rf_roles: tuple[str, ...],
) -> tuple[ParamDecl, ...]:
    rf = tuple(rf_roles)
    rf_choices = rf or ("unavailable: no synchronized RF Port",)
    return (
        ParamDecl(
            "pulse",
            "Pulse template",
            "path",
            default=DEFAULT_RELEASE_RECAPTURE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
        ),
        ParamDecl(
            "detuning",
            "Two-photon detuning",
            "axis_range",
            default=(-0.4, 0.4, 21),
            unit="Gamma",
            lo=-50.0,
            hi=50.0,
            required=True,
        ),
        ParamDecl(
            "t_off",
            "Trap-off time",
            "float",
            default=20.0,
            unit="us",
            lo=0.02,
            hi=10_000.0,
            required=True,
            optional=False,
        ),
        ParamDecl(
            "shots",
            "Shots / point",
            "int",
            default=16,
            lo=1,
            hi=100_000,
            required=True,
            optional=False,
        ),
        ParamDecl(
            "per_site",
            "Per-site survival",
            "bool",
            default=False,
        ),
        ParamDecl(
            "rf_role",
            "RF role",
            "choice",
            default=_preferred(rf_choices, "rf"),
            required=True,
            choices=rf_choices,
            tooltip=(
                "The current installation has no hardware-synchronized RF "
                "table Port; Start will reject this Measurement explicitly"
                if not rf
                else ""
            ),
        ),
        *_common_params(camera_roles, sequencer_roles),
    )


def _axis(
    value: object,
    name: str,
    *,
    scale: float,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be (minimum, maximum, points)")
    try:
        start, stop, count = tuple(value)  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be (minimum, maximum, points)") from exc
    if (
        isinstance(start, bool)
        or not isinstance(start, Real)
        or isinstance(stop, bool)
        or not isinstance(stop, Real)
        or isinstance(count, bool)
        or not isinstance(count, Integral)
        or int(count) < 1
    ):
        raise ValueError(f"{name} must contain finite bounds and positive points")
    values = np.linspace(float(start), float(stop), int(count), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} bounds must be finite")
    return tuple(float(item) * scale for item in values)


def _model(values: Mapping[str, object]) -> ReadoutModelKind | None:
    value = values.get("model_kind", "auto")
    return None if value in (None, "", "auto") else ReadoutModelKind(str(value))


def _calibration(reference: CalibrationArtifactRef) -> CalibrationArtifactRef:
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("calibration_ref must be CalibrationArtifactRef")
    return reference


def _selected_signal(values: Mapping[str, object]) -> str:
    value = values.get("calibration")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "select a Calibrate readout Task calibration output"
        )
    return value.strip()


@dataclass(frozen=True, slots=True)
class TemperatureReleaseRecaptureIntent:
    pulse: str
    trap_off_seconds: tuple[float, ...]
    shots: int
    calibration_signal: str
    model_kind: ReadoutModelKind | None
    per_site: bool
    camera_role: str
    sequencer_role: str
    trigger_channel: str | None


@dataclass(frozen=True, slots=True)
class ReadoutDurationFidelityIntent:
    pulse: str
    duration_seconds: tuple[float, ...]
    shots: int
    calibration_signal: str
    model_kind: ReadoutModelKind | None
    site: int | None
    camera_role: str
    sequencer_role: str
    trigger_channel: str | None


@dataclass(frozen=True, slots=True)
class GreyMolassesDetuningIntent:
    pulse: str
    detuning_gamma: tuple[float, ...]
    trap_off_seconds: float
    shots: int
    rf_role: str
    calibration_signal: str
    model_kind: ReadoutModelKind | None
    per_site: bool
    camera_role: str
    sequencer_role: str
    trigger_channel: str | None


def _trigger(values: Mapping[str, object]) -> str | None:
    value = values.get("trigger_channel")
    return None if value in (None, "") else str(value)


def build_temperature_release_recapture_intent(
    values: Mapping[str, object],
) -> TemperatureReleaseRecaptureIntent:
    return TemperatureReleaseRecaptureIntent(
        str(values.get("pulse") or DEFAULT_RELEASE_RECAPTURE_PULSE_PATH),
        _axis(values.get("t_off"), "t_off", scale=1e-6),
        int(values.get("shots", 16)),
        _selected_signal(values),
        _model(values),
        bool(values.get("per_site", False)),
        str(values["camera_role"]),
        str(values["sequencer_role"]),
        _trigger(values),
    )


def build_readout_duration_fidelity_intent(
    values: Mapping[str, object],
) -> ReadoutDurationFidelityIntent:
    site = values.get("site")
    return ReadoutDurationFidelityIntent(
        str(values.get("pulse") or DEFAULT_PROBE_PULSE_PATH),
        _axis(values.get("duration"), "duration", scale=1e-6),
        int(values.get("shots", 60)),
        _selected_signal(values),
        _model(values),
        None if site in (None, "") else int(site),
        str(values["camera_role"]),
        str(values["sequencer_role"]),
        _trigger(values),
    )


def build_grey_molasses_detuning_intent(
    values: Mapping[str, object],
) -> GreyMolassesDetuningIntent:
    return GreyMolassesDetuningIntent(
        str(values.get("pulse") or DEFAULT_RELEASE_RECAPTURE_PULSE_PATH),
        _axis(values.get("detuning"), "detuning", scale=1.0),
        float(values.get("t_off", 20.0)) * 1e-6,
        int(values.get("shots", 16)),
        str(values["rf_role"]),
        _selected_signal(values),
        _model(values),
        bool(values.get("per_site", False)),
        str(values["camera_role"]),
        str(values["sequencer_role"]),
        _trigger(values),
    )


def freeze_temperature_release_recapture_request(
    experiment,
    intent: TemperatureReleaseRecaptureIntent,
    *,
    calibration_ref: CalibrationArtifactRef,
):
    if not isinstance(intent, TemperatureReleaseRecaptureIntent):
        raise TypeError("intent must be TemperatureReleaseRecaptureIntent")
    return experiment.readout.temperature_release_recapture_request(
        intent.pulse,
        trap_off_seconds=intent.trap_off_seconds,
        shots=intent.shots,
        calibration_ref=_calibration(calibration_ref),
        model_kind=intent.model_kind,
        per_site=intent.per_site,
        camera_role=intent.camera_role,
        sequencer_role=intent.sequencer_role,
        trigger_channel=intent.trigger_channel,
    )


__all__ = [
    "GreyMolassesDetuningIntent",
    "ReadoutDurationFidelityIntent",
    "TemperatureReleaseRecaptureIntent",
    "build_grey_molasses_detuning_intent",
    "build_readout_duration_fidelity_intent",
    "build_temperature_release_recapture_intent",
    "freeze_temperature_release_recapture_request",
    "grey_molasses_detuning_params",
    "readout_duration_fidelity_params",
    "temperature_release_recapture_params",
]
