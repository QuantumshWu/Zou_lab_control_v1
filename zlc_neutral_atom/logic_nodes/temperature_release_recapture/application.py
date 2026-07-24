"""Application boundary for the Temperature release-recapture Measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.logic_nodes.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.release_recapture_common.application import (
    PreparedReleaseRecapture,
    final_release_recapture_output,
    prepare_release_recapture,
)
from zlc_neutral_atom.logic_nodes.release_recapture_common.timing import (
    TriggeredReleaseRecaptureResult,
)
from zlc_neutral_atom.logic_nodes.temperature_release_recapture.measurement import (
    BoundTemperatureReleaseRecapture,
    TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS,
    TemperatureReleaseRecaptureIntent,
    TemperatureReleaseRecaptureRequest,
    bind_temperature_release_recapture,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort


def temperature_final_outputs(
    result: TriggeredReleaseRecaptureResult,
) -> dict[str, FinalDatasetOutput]:
    """Publish this Measurement's exact survival curve."""

    return final_release_recapture_output(
        result,
        declaration=TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS[0],
        owner="temperature-release-recapture",
    )


def prepare_temperature_release_recapture(
    request: TemperatureReleaseRecaptureRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedReleaseRecapture:
    if not isinstance(request, TemperatureReleaseRecaptureRequest):
        raise TypeError("request must be TemperatureReleaseRecaptureRequest")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    bound = bind_temperature_release_recapture(
        request,
        calibration,
        pulse_port=pulse_port,
        camera_port=camera_port,
    )
    if not isinstance(bound, BoundTemperatureReleaseRecapture):
        raise RuntimeError("temperature binding returned another domain value")
    if calibration.reference != bound.request.calibration_ref:
        raise ValueError("calibration differs from the bound Measurement request")
    return prepare_release_recapture(
        name=f"Temperature release-recapture {bound.program.document.name}",
        owner="zlc_neutral_atom.release-recapture",
        program_fingerprint=bound.program.fingerprint,
        camera_binding=bound.camera_binding,
        calibration=calibration,
        model_kind=bound.request.model_kind,
        per_site=bound.request.per_site,
        start_run=start_run,
    )


class TemperatureReleaseRecaptureApplicationPort(Protocol):
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


@dataclass(frozen=True, slots=True)
class TemperatureReleaseRecaptureApplicationCommand:
    request: TemperatureReleaseRecaptureRequest
    _application: TemperatureReleaseRecaptureApplicationPort

    def __post_init__(self) -> None:
        if type(self.request) is not TemperatureReleaseRecaptureRequest:
            raise TypeError("request must be TemperatureReleaseRecaptureRequest")

    def start(self) -> RunHandle:
        return self._application.start_temperature_release_recapture(self.request)

    def final_dataset_outputs(
        self,
        result: TriggeredReleaseRecaptureResult,
    ) -> dict[str, FinalDatasetOutput]:
        return temperature_final_outputs(result)


def prepare_temperature_release_recapture_application(
    intent: TemperatureReleaseRecaptureIntent,
    calibration_ref: CalibrationArtifactRef,
    application: TemperatureReleaseRecaptureApplicationPort,
) -> TemperatureReleaseRecaptureApplicationCommand:
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
    return TemperatureReleaseRecaptureApplicationCommand(request, application)


__all__ = [
    "PreparedReleaseRecapture",
    "TemperatureReleaseRecaptureApplicationCommand",
    "TemperatureReleaseRecaptureApplicationPort",
    "prepare_temperature_release_recapture",
    "prepare_temperature_release_recapture_application",
    "temperature_final_outputs",
]
