"""Closed application commands for the three coupled readout Measurements.

The GUI owns only authored form state and its selected calibration signal.  Once
composition resolves that signal to a :class:`CalibrationArtifactRef`, this
module is the sole boundary that turns the physical intent into an installed
request, starts it, and materializes its named FINAL Dataset outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.coupled_measurements import (
    GreyMolassesDetuningIntent,
    GreyMolassesDetuningRequest,
    ReadoutDurationFidelityIntent,
    ReadoutDurationFidelityRequest,
    TemperatureReleaseRecaptureIntent,
    TemperatureReleaseRecaptureRequest,
)
from zlc_neutral_atom.readout_duration_application import (
    ReadoutDurationFidelityResult,
    readout_duration_fidelity_final_outputs,
)
from zlc_neutral_atom.release_recapture_application import (
    grey_molasses_final_outputs,
    temperature_final_outputs,
)
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_neutral_atom.timing.release_recapture import (
    TriggeredReleaseRecaptureResult,
)


CoupledMeasurementRequest = (
    TemperatureReleaseRecaptureRequest
    | ReadoutDurationFidelityRequest
    | GreyMolassesDetuningRequest
)
CoupledMeasurementResult = (
    TriggeredReleaseRecaptureResult | ReadoutDurationFidelityResult
)
_REQUEST_TYPES = (
    TemperatureReleaseRecaptureRequest,
    ReadoutDurationFidelityRequest,
    GreyMolassesDetuningRequest,
)


class CoupledMeasurementApplicationPort(Protocol):
    """Exact Readout façade surface required by coupled application commands."""

    def temperature_release_recapture_request(
        self,
        pulse: str,
        *,
        trap_off_seconds: tuple[float, ...],
        shots: int,
        calibration_ref: CalibrationArtifactRef,
        per_site: bool = False,
    ) -> TemperatureReleaseRecaptureRequest: ...

    def start_temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ) -> RunHandle: ...

    def readout_duration_fidelity_request(
        self,
        pulse: str,
        *,
        duration_seconds: tuple[float, ...],
        shots: int,
        calibration_ref: CalibrationArtifactRef,
        site: int | None = None,
    ) -> ReadoutDurationFidelityRequest: ...

    def start_readout_duration_fidelity(
        self,
        request: ReadoutDurationFidelityRequest,
    ) -> RunHandle: ...

    def grey_molasses_detuning_request(
        self,
        pulse: str,
        *,
        detuning_gamma: tuple[float, ...],
        trap_off_seconds: float,
        shots: int,
        rf_role: str,
        calibration_ref: CalibrationArtifactRef,
        per_site: bool = False,
    ) -> GreyMolassesDetuningRequest: ...

    def start_grey_molasses_detuning(
        self,
        request: GreyMolassesDetuningRequest,
    ) -> RunHandle: ...


@dataclass(frozen=True, slots=True)
class CoupledMeasurementApplicationCommand:
    """One installed request with its Start and FINAL-output semantics."""

    request: CoupledMeasurementRequest
    _application: CoupledMeasurementApplicationPort

    def __post_init__(self) -> None:
        if type(self.request) not in _REQUEST_TYPES:
            raise TypeError("request must be a coupled Measurement request")

    def start(self) -> RunHandle:
        request = self.request
        if type(request) is TemperatureReleaseRecaptureRequest:
            return self._application.start_temperature_release_recapture(request)
        if type(request) is ReadoutDurationFidelityRequest:
            return self._application.start_readout_duration_fidelity(request)
        if type(request) is GreyMolassesDetuningRequest:
            return self._application.start_grey_molasses_detuning(request)
        raise AssertionError("unreachable coupled Measurement request type")

    def final_dataset_outputs(
        self,
        result: CoupledMeasurementResult,
    ) -> dict[str, FinalDatasetOutput]:
        request = self.request
        if type(request) is ReadoutDurationFidelityRequest:
            return readout_duration_fidelity_final_outputs(result)
        if type(request) is TemperatureReleaseRecaptureRequest:
            return temperature_final_outputs(result)
        if type(request) is GreyMolassesDetuningRequest:
            return grey_molasses_final_outputs(result)
        raise AssertionError("unreachable coupled Measurement request type")


def prepare_temperature_release_recapture_application(
    intent: TemperatureReleaseRecaptureIntent,
    calibration_ref: CalibrationArtifactRef,
    application: CoupledMeasurementApplicationPort,
) -> CoupledMeasurementApplicationCommand:
    """Bind one device-independent temperature intent through Readout."""

    if not isinstance(intent, TemperatureReleaseRecaptureIntent):
        raise TypeError("intent must be TemperatureReleaseRecaptureIntent")
    if not isinstance(calibration_ref, CalibrationArtifactRef):
        raise TypeError("calibration_ref must be CalibrationArtifactRef")
    request = application.temperature_release_recapture_request(
        intent.pulse,
        trap_off_seconds=intent.trap_off_seconds,
        shots=intent.shots,
        calibration_ref=calibration_ref,
        per_site=intent.per_site,
    )
    if type(request) is not TemperatureReleaseRecaptureRequest:
        raise TypeError("Readout application returned another Temperature request")
    return CoupledMeasurementApplicationCommand(request, application)


def prepare_readout_duration_fidelity_application(
    intent: ReadoutDurationFidelityIntent,
    calibration_ref: CalibrationArtifactRef,
    application: CoupledMeasurementApplicationPort,
) -> CoupledMeasurementApplicationCommand:
    """Bind one device-independent duration intent through Readout."""

    if not isinstance(intent, ReadoutDurationFidelityIntent):
        raise TypeError("intent must be ReadoutDurationFidelityIntent")
    if not isinstance(calibration_ref, CalibrationArtifactRef):
        raise TypeError("calibration_ref must be CalibrationArtifactRef")
    request = application.readout_duration_fidelity_request(
        intent.pulse,
        duration_seconds=intent.duration_seconds,
        shots=intent.shots,
        calibration_ref=calibration_ref,
        site=intent.site,
    )
    if type(request) is not ReadoutDurationFidelityRequest:
        raise TypeError("Readout application returned another duration request")
    return CoupledMeasurementApplicationCommand(request, application)


def prepare_grey_molasses_detuning_application(
    intent: GreyMolassesDetuningIntent,
    calibration_ref: CalibrationArtifactRef,
    application: CoupledMeasurementApplicationPort,
) -> CoupledMeasurementApplicationCommand:
    """Bind one device-independent Grey-molasses intent through Readout."""

    if not isinstance(intent, GreyMolassesDetuningIntent):
        raise TypeError("intent must be GreyMolassesDetuningIntent")
    if not isinstance(calibration_ref, CalibrationArtifactRef):
        raise TypeError("calibration_ref must be CalibrationArtifactRef")
    request = application.grey_molasses_detuning_request(
        intent.pulse,
        detuning_gamma=intent.detuning_gamma,
        trap_off_seconds=intent.trap_off_seconds,
        shots=intent.shots,
        rf_role=intent.rf_role,
        calibration_ref=calibration_ref,
        per_site=intent.per_site,
    )
    if type(request) is not GreyMolassesDetuningRequest:
        raise TypeError("Readout application returned another Grey-molasses request")
    return CoupledMeasurementApplicationCommand(request, application)


__all__ = [
    "CoupledMeasurementApplicationCommand",
    "CoupledMeasurementApplicationPort",
    "CoupledMeasurementRequest",
    "CoupledMeasurementResult",
    "prepare_grey_molasses_detuning_application",
    "prepare_readout_duration_fidelity_application",
    "prepare_temperature_release_recapture_application",
]
