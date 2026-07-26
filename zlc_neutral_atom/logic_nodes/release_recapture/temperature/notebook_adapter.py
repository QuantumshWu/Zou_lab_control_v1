"""Notebook surface owned by Temperature release-recapture."""

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

from ..application import PreparedReleaseRecapture
from .application import (
    TemperatureReleaseRecaptureApplicationCommand,
    TemperatureReleaseRecaptureIntent,
    prepare_temperature_release_recapture_application,
)
from .measurement import TemperatureReleaseRecaptureRequest


class TemperatureReleaseRecaptureNotebookHost(Protocol):
    def load_temperature_pulse(
        self,
        value: PulseDocument | str | Path,
    ) -> PulseDocument: ...

    def resolve_temperature_camera_ref(self, requested: str | None) -> DeviceRef: ...

    def resolve_temperature_sequencer_ref(
        self,
        requested: str | None,
    ) -> DeviceRef: ...

    def bind_temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ) -> PreparedReleaseRecapture: ...

    def wait_temperature_release_recapture(self, handle: RunHandle): ...


class TemperatureReleaseRecaptureNotebookAdapter:
    __slots__ = ()

    @property
    def _temperature_notebook_host(
        self,
    ) -> TemperatureReleaseRecaptureNotebookHost:
        raise NotImplementedError

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
        host = self._temperature_notebook_host
        return TemperatureReleaseRecaptureRequest(
            host.load_temperature_pulse(pulse),
            tuple(trap_off_seconds),
            shots,
            host.resolve_temperature_camera_ref(camera_role),
            host.resolve_temperature_sequencer_ref(sequencer_role),
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
        return self._temperature_notebook_host.bind_temperature_release_recapture(
            request
        ).start()

    def prepare_temperature_release_recapture_application(
        self,
        intent: TemperatureReleaseRecaptureIntent,
        calibration_ref: CalibrationArtifactRef,
    ) -> TemperatureReleaseRecaptureApplicationCommand:
        return prepare_temperature_release_recapture_application(
            intent,
            calibration_ref,
            self,
        )

    def temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ):
        if not isinstance(request, TemperatureReleaseRecaptureRequest):
            raise TypeError("request must be TemperatureReleaseRecaptureRequest")
        handle = self._temperature_notebook_host.bind_temperature_release_recapture(
            request
        ).start()
        return self._temperature_notebook_host.wait_temperature_release_recapture(
            handle
        )


__all__ = [
    "TemperatureReleaseRecaptureNotebookAdapter",
    "TemperatureReleaseRecaptureNotebookHost",
]
