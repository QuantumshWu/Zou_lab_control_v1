"""Public Experiment API owned by grey-molasses detuning release-recapture."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument

from ..application import PreparedReleaseRecapture
from .application import (
    GreyMolassesDetuningApplicationCommand,
    GreyMolassesDetuningIntent,
    prepare_grey_molasses_detuning_application,
)
from .measurement import (
    GreyMolassesDetuningRequest,
)


class GreyMolassesDetuningApi:
    __slots__ = (
        "_bind_request",
        "_load_pulse",
        "_resolve_camera_ref",
        "_resolve_rf_role_operation",
        "_resolve_sequencer_ref",
    )

    def __init__(
        self,
        *,
        load_pulse: Callable,
        resolve_camera_ref: Callable,
        resolve_sequencer_ref: Callable,
        resolve_rf_role: Callable,
        bind_request: Callable,
    ) -> None:
        operations = (
            load_pulse,
            resolve_camera_ref,
            resolve_sequencer_ref,
            resolve_rf_role,
            bind_request,
        )
        if any(not callable(operation) for operation in operations):
            raise TypeError("grey-molasses API operations must be callable")
        self._load_pulse = load_pulse
        self._resolve_camera_ref = resolve_camera_ref
        self._resolve_sequencer_ref = resolve_sequencer_ref
        self._resolve_rf_role_operation = resolve_rf_role
        self._bind_request = bind_request

    def _resolve_rf_role(self, requested: str) -> str:
        return self._resolve_rf_role_operation(requested)

    def grey_molasses_detuning_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        detuning_gamma: tuple[float, ...],
        trap_off_seconds: float,
        shots: int,
        rf_role: str,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
        per_site: bool = False,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> GreyMolassesDetuningRequest:
        return GreyMolassesDetuningRequest(
            self._load_pulse(pulse),
            tuple(detuning_gamma),
            trap_off_seconds,
            shots,
            self._resolve_camera_ref(camera_role),
            self._resolve_sequencer_ref(sequencer_role),
            self._resolve_rf_role(rf_role),
            calibration_ref,
            model_kind,
            per_site,
            trigger_channel,
        )

    def start_grey_molasses_detuning(
        self,
        request: GreyMolassesDetuningRequest,
        *,
        lifecycle_owner: object | None = None,
    ) -> RunHandle:
        if not isinstance(request, GreyMolassesDetuningRequest):
            raise TypeError("request must be GreyMolassesDetuningRequest")
        return self._bind(request).start(lifecycle_owner=lifecycle_owner)

    def _bind(
        self,
        request: GreyMolassesDetuningRequest,
    ) -> PreparedReleaseRecapture:
        return self._bind_request(request)

    def prepare_grey_molasses_detuning_application(
        self,
        intent: GreyMolassesDetuningIntent,
        calibration_ref: CalibrationArtifactRef,
    ) -> GreyMolassesDetuningApplicationCommand:
        return prepare_grey_molasses_detuning_application(
            intent,
            calibration_ref,
            self,
        )


__all__ = [
    "GreyMolassesDetuningApi",
]
