"""Public Experiment API owned by Temperature release-recapture."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument

from ..application import PreparedReleaseRecapture
from .measurement import TemperatureReleaseRecaptureRequest


class TemperatureReleaseRecaptureApi:
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
            raise TypeError("temperature API operations must be callable")
        self._load_pulse = load_pulse
        self._resolve_camera_ref = resolve_camera_ref
        self._resolve_sequencer_ref = resolve_sequencer_ref
        self._bind_request = bind_request
        self._wait_run = wait_run

    def temperature_release_recapture_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        trap_off_seconds: tuple[float, ...],
        shots: int,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
        per_site: bool = False,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> TemperatureReleaseRecaptureRequest:
        return TemperatureReleaseRecaptureRequest(
            self._load_pulse(pulse),
            tuple(trap_off_seconds),
            shots,
            self._resolve_camera_ref(camera_role),
            self._resolve_sequencer_ref(sequencer_role),
            calibration_ref,
            model_kind,
            per_site,
            trigger_channel,
        )

    def start_temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ) -> RunHandle:
        if not isinstance(request, TemperatureReleaseRecaptureRequest):
            raise TypeError("request must be TemperatureReleaseRecaptureRequest")
        return self.prepare_temperature_release_recapture(request).start()

    def prepare_temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ) -> PreparedReleaseRecapture:
        if not isinstance(request, TemperatureReleaseRecaptureRequest):
            raise TypeError("request must be TemperatureReleaseRecaptureRequest")
        return self._bind_request(request)

    def temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ):
        if not isinstance(request, TemperatureReleaseRecaptureRequest):
            raise TypeError("request must be TemperatureReleaseRecaptureRequest")
        handle = self.prepare_temperature_release_recapture(request).start()
        return self._wait_run(handle)


__all__ = [
    "TemperatureReleaseRecaptureApi",
]
