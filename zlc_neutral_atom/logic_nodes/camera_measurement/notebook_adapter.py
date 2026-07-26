"""Notebook surface owned by the Camera Measurement capability.

The capability owns its public request vocabulary.  The application composition
root contributes only two installed-runtime operations through the narrow host
port below; no Experiment, repository graph, Workbench, or service lookup crosses
into this module.
"""

from __future__ import annotations

from typing import Protocol

from zlc_neutral_atom.installation import DeviceRef

from .definition import (
    DEFAULT_CAMERA_FRAMES_PER_CYCLE,
    DEFAULT_CAMERA_MEASUREMENT_REPEAT,
    DEFAULT_CAMERA_MONITOR_HISTORY_CYCLES,
    CameraMeasurementRequest,
)
from .finite import PreparedFiniteCameraMeasurement
from .monitor import PreparedLiveCameraMeasurement


class CameraMeasurementNotebookHost(Protocol):
    """Installed operations required by the Camera notebook surface."""

    def resolve_camera_measurement_ref(
        self,
        requested_role: str | None,
    ) -> DeviceRef: ...

    def bind_camera_measurement(
        self,
        request: CameraMeasurementRequest,
    ) -> PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement: ...


class CameraMeasurementNotebookAdapter:
    """Flat ``exp.readout`` methods contributed by Camera Measurement."""

    __slots__ = ()

    @property
    def _camera_measurement_notebook_host(
        self,
    ) -> CameraMeasurementNotebookHost:
        raise NotImplementedError

    def camera_measurement_request(
        self,
        *,
        camera_role: str | None = None,
        repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT,
        history_cycles: int = DEFAULT_CAMERA_MONITOR_HISTORY_CYCLES,
        frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE,
        exposure: float | None = None,
    ) -> CameraMeasurementRequest:
        """Freeze Main's one Camera semantic: 0=live, K=finite."""

        host = self._camera_measurement_notebook_host
        return CameraMeasurementRequest(
            camera_ref=host.resolve_camera_measurement_ref(camera_role),
            repeat=repeat,
            history_cycles=history_cycles,
            frames_per_cycle=frames_per_cycle,
            exposure_seconds=exposure,
        )

    def prepare_camera_measurement(
        self,
        request: CameraMeasurementRequest,
    ) -> PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement:
        """Bind one typed Camera Measurement to the installed runtime."""

        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("request must be CameraMeasurementRequest")
        return self._camera_measurement_notebook_host.bind_camera_measurement(request)


__all__ = [
    "CameraMeasurementNotebookAdapter",
    "CameraMeasurementNotebookHost",
]
