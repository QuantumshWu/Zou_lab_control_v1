"""Public Experiment API owned by readout-duration fidelity."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
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


class ReadoutDurationFidelityApi:
    __slots__ = (
        "_bind_request",
        "_load_pulse",
        "_resolve_camera_ref",
        "_resolve_sequencer_ref",
        "_wait_run",
    )

    def __init__(
        self,
        *,
        load_pulse: Callable,
        resolve_camera_ref: Callable,
        resolve_sequencer_ref: Callable,
        bind_request: Callable,
        wait_run: Callable,
    ) -> None:
        operations = (
            load_pulse,
            resolve_camera_ref,
            resolve_sequencer_ref,
            bind_request,
            wait_run,
        )
        if any(not callable(operation) for operation in operations):
            raise TypeError("duration-fidelity API operations must be callable")
        self._load_pulse = load_pulse
        self._resolve_camera_ref = resolve_camera_ref
        self._resolve_sequencer_ref = resolve_sequencer_ref
        self._bind_request = bind_request
        self._wait_run = wait_run

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
        return ReadoutDurationFidelityRequest(
            self._load_pulse(pulse),
            tuple(duration_seconds),
            shots,
            self._resolve_camera_ref(camera_role),
            self._resolve_sequencer_ref(sequencer_role),
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
        return self._bind(request).start()

    def _bind(
        self,
        request: ReadoutDurationFidelityRequest,
    ) -> PreparedReadoutDurationFidelity:
        return self._bind_request(request)

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
        handle = self._bind(request).start()
        return self._wait_run(handle)


__all__ = [
    "ReadoutDurationFidelityApi",
]
