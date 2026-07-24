"""TaskConsole forms and binding for coupled readout Measurements.

This module owns presentation fields and the explicit Signal(calibration)
selection only.  Axis construction, unit conversion, and physical validation
are delegated to the neutral-atom coupled-measurement owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.pulse_programs import (
    DEFAULT_PROBE_PULSE_PATH,
    DEFAULT_RELEASE_RECAPTURE_PULSE_PATH,
)
from zlc_neutral_atom.readout.coupled_measurements import (
    AutonomousMeasurementUnavailable,
    DEFAULT_GREY_MOLASSES_DETUNING_GAMMA_RANGE,
    DEFAULT_GREY_MOLASSES_PER_SITE,
    DEFAULT_GREY_MOLASSES_RF_ROLE,
    DEFAULT_GREY_MOLASSES_SHOTS,
    DEFAULT_GREY_MOLASSES_TRAP_OFF_MICROSECONDS,
    DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE,
    DEFAULT_READOUT_DURATION_SHOTS,
    DEFAULT_READOUT_DURATION_SITE,
    DEFAULT_TEMPERATURE_PER_SITE,
    DEFAULT_TEMPERATURE_SHOTS,
    DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE,
    GREY_MOLASSES_CAPABILITY_GAP,
    MINIMUM_COUPLED_MEASUREMENT_SHOTS,
    MINIMUM_READOUT_SITE_INDEX,
    GreyMolassesDetuningIntent,
    ReadoutDurationFidelityIntent,
    TemperatureReleaseRecaptureIntent,
    build_grey_molasses_detuning_intent,
    build_readout_duration_fidelity_intent,
    build_temperature_release_recapture_intent,
)
from zlc_storage import normalized_text

__all__ = [
    "CoupledMeasurementBinding",
    "build_grey_molasses_detuning_binding",
    "build_readout_duration_fidelity_binding",
    "build_temperature_release_recapture_binding",
    "grey_molasses_detuning_params",
    "readout_duration_fidelity_params",
    "temperature_release_recapture_params",
]


CoupledPhysicalIntent = (
    TemperatureReleaseRecaptureIntent
    | ReadoutDurationFidelityIntent
    | GreyMolassesDetuningIntent
)
_COUPLED_INTENT_TYPES = (
    TemperatureReleaseRecaptureIntent,
    ReadoutDurationFidelityIntent,
    GreyMolassesDetuningIntent,
)


@dataclass(frozen=True, slots=True)
class CoupledMeasurementBinding:
    """One physical intent plus its TaskConsole-only calibration producer."""

    intent: CoupledPhysicalIntent
    calibration_signal: str

    def __post_init__(self) -> None:
        if not isinstance(self.intent, _COUPLED_INTENT_TYPES):
            raise TypeError("intent must be a coupled Measurement physical intent")
        object.__setattr__(
            self,
            "calibration_signal",
            normalized_text(self.calibration_signal, "calibration_signal"),
        )


def _preferred(roles: tuple[str, ...], *candidates: str) -> str | None:
    if not roles:
        return None
    for candidate in candidates:
        if candidate in roles:
            return candidate
    return roles[0]


def _calibration_param() -> FormFieldProps:
    return FormFieldProps(
        "calibration",
        "signal",
        "Calibration",
        required=True,
        description=(
            "FINAL calibration output of a successful Calibrate readout "
            "Task; TaskConsole resolves its exact artifact reference at Start"
        ),
    )


def temperature_release_recapture_params() -> FormSpec:
    return FormSpec((
        FormFieldProps(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_RELEASE_RECAPTURE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
            description=(
                "Autonomous two-readout pulse with the declared t_off SCAN_SLOT"
            ),
        ),
        FormFieldProps(
            "t_off",
            "axis_range",
            "Trap-off time",
            default=DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE,
            unit="us",
            minimum=0.02,
            maximum=10_000.0,
            required=True,
            description=(
                "The bundled 50 MHz pulse target requires at least one "
                "20 ns clock tick; the selected PulseDocument is validated "
                "again when the request is frozen"
            ),
        ),
        FormFieldProps(
            "shots",
            "int",
            "Shots / point",
            default=DEFAULT_TEMPERATURE_SHOTS,
            minimum=MINIMUM_COUPLED_MEASUREMENT_SHOTS,
            maximum=100_000,
            required=True,
            allow_blank=False,
        ),
        FormFieldProps(
            "per_site",
            "bool",
            "Per-site survival",
            default=DEFAULT_TEMPERATURE_PER_SITE,
        ),
        _calibration_param(),
    ))


def readout_duration_fidelity_params() -> FormSpec:
    return FormSpec((
        FormFieldProps(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_PROBE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
        ),
        FormFieldProps(
            "duration",
            "axis_range",
            "Detection time",
            default=DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE,
            unit="us",
            minimum=0.02,
            maximum=1_000_000.0,
            required=True,
        ),
        FormFieldProps(
            "shots",
            "int",
            "Shots / point",
            default=DEFAULT_READOUT_DURATION_SHOTS,
            minimum=MINIMUM_COUPLED_MEASUREMENT_SHOTS,
            maximum=100_000,
            required=True,
            allow_blank=False,
        ),
        FormFieldProps(
            "site",
            "int",
            "Site (optional)",
            default=DEFAULT_READOUT_DURATION_SITE,
            minimum=MINIMUM_READOUT_SITE_INDEX,
            maximum=100_000,
            required=False,
            allow_blank=True,
        ),
        _calibration_param(),
    ))


def grey_molasses_detuning_params(rf_roles: tuple[str, ...]) -> FormSpec:
    rf = tuple(rf_roles)
    return FormSpec((
        FormFieldProps(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_RELEASE_RECAPTURE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
        ),
        FormFieldProps(
            "detuning",
            "axis_range",
            "Two-photon detuning",
            default=DEFAULT_GREY_MOLASSES_DETUNING_GAMMA_RANGE,
            unit="Γ",
            minimum=-50.0,
            maximum=50.0,
            required=True,
        ),
        FormFieldProps(
            "t_off",
            "float",
            "Trap-off time",
            default=DEFAULT_GREY_MOLASSES_TRAP_OFF_MICROSECONDS,
            unit="us",
            minimum=0.02,
            maximum=10_000.0,
            required=True,
            allow_blank=False,
        ),
        FormFieldProps(
            "shots",
            "int",
            "Shots / point",
            default=DEFAULT_GREY_MOLASSES_SHOTS,
            minimum=MINIMUM_COUPLED_MEASUREMENT_SHOTS,
            maximum=100_000,
            required=True,
            allow_blank=False,
        ),
        FormFieldProps(
            "per_site",
            "bool",
            "Per-site survival",
            default=DEFAULT_GREY_MOLASSES_PER_SITE,
        ),
        FormFieldProps(
            "rf_role",
            "choice",
            "RF role",
            default=_preferred(rf, DEFAULT_GREY_MOLASSES_RF_ROLE),
            required=True,
            choices=tuple(FormChoice(value, value) for value in rf),
            description=(
                "Hardware-synchronized RF table Port advanced by the scan clock"
            ),
            unavailable_reason=(GREY_MOLASSES_CAPABILITY_GAP if not rf else ""),
        ),
        _calibration_param(),
    ))


def _calibration_signal(values: Mapping[str, object]) -> str:
    value = values.get("calibration")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("select a Calibrate readout Task calibration output")
    return value.strip()


def build_temperature_release_recapture_binding(
    values: Mapping[str, object],
) -> CoupledMeasurementBinding:
    return CoupledMeasurementBinding(
        build_temperature_release_recapture_intent(
            pulse=str(values.get("pulse") or DEFAULT_RELEASE_RECAPTURE_PULSE_PATH),
            trap_off_microseconds=values.get(
                "t_off",
                DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE,
            ),
            shots=values.get("shots", DEFAULT_TEMPERATURE_SHOTS),
            per_site=values.get("per_site", DEFAULT_TEMPERATURE_PER_SITE),
        ),
        _calibration_signal(values),
    )


def build_readout_duration_fidelity_binding(
    values: Mapping[str, object],
) -> CoupledMeasurementBinding:
    site = values.get("site", DEFAULT_READOUT_DURATION_SITE)
    return CoupledMeasurementBinding(
        build_readout_duration_fidelity_intent(
            pulse=str(values.get("pulse") or DEFAULT_PROBE_PULSE_PATH),
            duration_microseconds=values.get(
                "duration",
                DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE,
            ),
            shots=values.get("shots", DEFAULT_READOUT_DURATION_SHOTS),
            site=None if site in (None, "") else site,
        ),
        _calibration_signal(values),
    )


def build_grey_molasses_detuning_binding(
    values: Mapping[str, object],
) -> CoupledMeasurementBinding:
    rf_role = values.get("rf_role")
    if rf_role is None:
        raise AutonomousMeasurementUnavailable(GREY_MOLASSES_CAPABILITY_GAP)
    return CoupledMeasurementBinding(
        build_grey_molasses_detuning_intent(
            pulse=str(values.get("pulse") or DEFAULT_RELEASE_RECAPTURE_PULSE_PATH),
            detuning_gamma_range=values.get(
                "detuning",
                DEFAULT_GREY_MOLASSES_DETUNING_GAMMA_RANGE,
            ),
            trap_off_microseconds=values.get(
                "t_off",
                DEFAULT_GREY_MOLASSES_TRAP_OFF_MICROSECONDS,
            ),
            shots=values.get("shots", DEFAULT_GREY_MOLASSES_SHOTS),
            rf_role=rf_role,
            per_site=values.get("per_site", DEFAULT_GREY_MOLASSES_PER_SITE),
        ),
        _calibration_signal(values),
    )
