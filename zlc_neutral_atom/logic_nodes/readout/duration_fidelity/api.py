"""Notebook surface owned by readout-duration fidelity."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ReadoutModelKind,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument

from .application import (
    PreparedReadoutDurationFidelity,
    ReadoutDurationFidelityApplicationCommand,
    ReadoutDurationFidelityIntent,
    prepare_readout_duration_fidelity_application,
)
from .measurement import ReadoutDurationFidelityRequest


class ReadoutDurationFidelityNotebookHost(Protocol):
    def load_duration_fidelity_pulse(
        self,
        value: PulseDocument | str | Path,
    ) -> PulseDocument: ...

    def resolve_duration_fidelity_camera_ref(
        self,
        requested: str | None,
    ) -> DeviceRef: ...

    def resolve_duration_fidelity_sequencer_ref(
        self,
        requested: str | None,
    ) -> DeviceRef: ...

    def bind_readout_duration_fidelity(
        self,
        request: ReadoutDurationFidelityRequest,
    ) -> PreparedReadoutDurationFidelity: ...

    def wait_readout_duration_fidelity(self, handle: RunHandle): ...


class ReadoutDurationFidelityNotebookAdapter:
    __slots__ = ()

    @property
    def _duration_fidelity_notebook_host(
        self,
    ) -> ReadoutDurationFidelityNotebookHost:
        raise NotImplementedError

    def readout_duration_fidelity_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        duration_seconds: tuple[float, ...],
        shots: int,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
        site: int | None = None,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> ReadoutDurationFidelityRequest:
        host = self._duration_fidelity_notebook_host
        return ReadoutDurationFidelityRequest(
            host.load_duration_fidelity_pulse(pulse),
            tuple(duration_seconds),
            shots,
            host.resolve_duration_fidelity_camera_ref(camera_role),
            host.resolve_duration_fidelity_sequencer_ref(sequencer_role),
            calibration_ref,
            model_kind,
            site,
            trigger_channel,
        )

    def start_readout_duration_fidelity(
        self,
        request: ReadoutDurationFidelityRequest,
    ) -> RunHandle:
        if not isinstance(request, ReadoutDurationFidelityRequest):
            raise TypeError("request must be ReadoutDurationFidelityRequest")
        return self._duration_fidelity_notebook_host.bind_readout_duration_fidelity(
            request
        ).start()

    def prepare_readout_duration_fidelity_application(
        self,
        intent: ReadoutDurationFidelityIntent,
        calibration_ref: CalibrationArtifactRef,
    ) -> ReadoutDurationFidelityApplicationCommand:
        return prepare_readout_duration_fidelity_application(
            intent,
            calibration_ref,
            self,
        )

    def readout_duration_fidelity(
        self,
        request: ReadoutDurationFidelityRequest,
    ):
        if not isinstance(request, ReadoutDurationFidelityRequest):
            raise TypeError("request must be ReadoutDurationFidelityRequest")
        handle = self._duration_fidelity_notebook_host.bind_readout_duration_fidelity(
            request
        ).start()
        return self._duration_fidelity_notebook_host.wait_readout_duration_fidelity(
            handle
        )


__all__ = [
    "ReadoutDurationFidelityNotebookAdapter",
    "ReadoutDurationFidelityNotebookHost",
]
