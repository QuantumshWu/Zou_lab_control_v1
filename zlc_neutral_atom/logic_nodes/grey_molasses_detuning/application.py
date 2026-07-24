"""Application boundary for the Grey-molasses detuning Measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.devices.rf import BoundRfTablePort
from zlc_neutral_atom.logic_nodes.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.grey_molasses_detuning.measurement import (
    BoundGreyMolassesDetuning,
    GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS,
    GreyMolassesDetuningIntent,
    GreyMolassesDetuningRequest,
    bind_grey_molasses_detuning,
)
from zlc_neutral_atom.logic_nodes.release_recapture_common.application import (
    PreparedReleaseRecapture,
    final_release_recapture_output,
    prepare_release_recapture,
)
from zlc_neutral_atom.logic_nodes.release_recapture_common.timing import (
    TriggeredReleaseRecaptureResult,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort


def grey_molasses_final_outputs(
    result: TriggeredReleaseRecaptureResult,
) -> dict[str, FinalDatasetOutput]:
    return final_release_recapture_output(
        result,
        declaration=GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS[0],
        owner="grey-molasses-detuning",
    )


def prepare_grey_molasses_detuning(
    request: GreyMolassesDetuningRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    rf_port: BoundRfTablePort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedReleaseRecapture:
    if not isinstance(request, GreyMolassesDetuningRequest):
        raise TypeError("request must be GreyMolassesDetuningRequest")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    bound = bind_grey_molasses_detuning(
        request,
        calibration,
        pulse_port=pulse_port,
        camera_port=camera_port,
        rf_port=rf_port,
    )
    if not isinstance(bound, BoundGreyMolassesDetuning):
        raise RuntimeError("Grey-molasses binding returned another domain value")
    if calibration.reference != bound.request.calibration_ref:
        raise ValueError("calibration differs from the bound Measurement request")
    return prepare_release_recapture(
        name=f"Grey molasses detuning {bound.program.document.name}",
        owner="zlc_neutral_atom.grey-molasses-detuning",
        program_fingerprint=bound.program.fingerprint,
        camera_binding=bound.camera_binding,
        calibration=calibration,
        model_kind=bound.request.model_kind,
        per_site=bound.request.per_site,
        start_run=start_run,
        rf_port=bound.rf_port,
        rf_table=bound.rf_table,
    )


class GreyMolassesDetuningApplicationPort(Protocol):
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
class GreyMolassesDetuningApplicationCommand:
    request: GreyMolassesDetuningRequest
    _application: GreyMolassesDetuningApplicationPort

    def __post_init__(self) -> None:
        if type(self.request) is not GreyMolassesDetuningRequest:
            raise TypeError("request must be GreyMolassesDetuningRequest")

    def start(self) -> RunHandle:
        return self._application.start_grey_molasses_detuning(self.request)

    def final_dataset_outputs(
        self,
        result: TriggeredReleaseRecaptureResult,
    ) -> dict[str, FinalDatasetOutput]:
        return grey_molasses_final_outputs(result)


def prepare_grey_molasses_detuning_application(
    intent: GreyMolassesDetuningIntent,
    calibration_ref: CalibrationArtifactRef,
    application: GreyMolassesDetuningApplicationPort,
) -> GreyMolassesDetuningApplicationCommand:
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
    return GreyMolassesDetuningApplicationCommand(request, application)


__all__ = [
    "GreyMolassesDetuningApplicationCommand",
    "GreyMolassesDetuningApplicationPort",
    "PreparedReleaseRecapture",
    "grey_molasses_final_outputs",
    "prepare_grey_molasses_detuning",
    "prepare_grey_molasses_detuning_application",
]
