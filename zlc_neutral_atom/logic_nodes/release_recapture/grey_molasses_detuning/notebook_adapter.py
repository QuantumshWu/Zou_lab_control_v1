"""Notebook surface owned by grey-molasses detuning release-recapture."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ReadoutModelKind
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument

from ..application import PreparedReleaseRecapture
from .application import (
    GreyMolassesDetuningApplicationCommand,
    GreyMolassesDetuningIntent,
    prepare_grey_molasses_detuning_application,
)
from .measurement import GreyMolassesDetuningRequest


class GreyMolassesDetuningNotebookHost(Protocol):
    def load_grey_molasses_pulse(
        self,
        value: PulseDocument | str | Path,
    ) -> PulseDocument: ...

    def resolve_grey_molasses_camera_ref(self, requested: str | None) -> DeviceRef: ...

    def resolve_grey_molasses_sequencer_ref(
        self,
        requested: str | None,
    ) -> DeviceRef: ...

    def resolve_grey_molasses_rf_role(self, requested: str) -> str: ...

    def bind_grey_molasses_detuning(
        self,
        request: GreyMolassesDetuningRequest,
    ) -> PreparedReleaseRecapture: ...


class GreyMolassesDetuningNotebookAdapter:
    __slots__ = ()

    @property
    def _grey_molasses_notebook_host(
        self,
    ) -> GreyMolassesDetuningNotebookHost:
        raise NotImplementedError

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
        host = self._grey_molasses_notebook_host
        return GreyMolassesDetuningRequest(
            host.load_grey_molasses_pulse(pulse),
            tuple(detuning_gamma),
            trap_off_seconds,
            shots,
            host.resolve_grey_molasses_camera_ref(camera_role),
            host.resolve_grey_molasses_sequencer_ref(sequencer_role),
            host.resolve_grey_molasses_rf_role(rf_role),
            calibration_ref,
            model_kind,
            per_site,
            trigger_channel,
        )

    def start_grey_molasses_detuning(
        self,
        request: GreyMolassesDetuningRequest,
    ) -> RunHandle:
        if not isinstance(request, GreyMolassesDetuningRequest):
            raise TypeError("request must be GreyMolassesDetuningRequest")
        return self._grey_molasses_notebook_host.bind_grey_molasses_detuning(
            request
        ).start()

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
    "GreyMolassesDetuningNotebookAdapter",
    "GreyMolassesDetuningNotebookHost",
]
