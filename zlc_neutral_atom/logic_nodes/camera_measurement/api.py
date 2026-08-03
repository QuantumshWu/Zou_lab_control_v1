"""Public Experiment API owned by the Camera Measurement capability.

The capability owns its public request vocabulary.  The application composition
root contributes only two installed-runtime operations through the narrow host
port below; no Experiment, repository graph, Workbench, or service lookup crosses
into this module.
"""

from __future__ import annotations

from collections.abc import Callable

from .definition import (
    DEFAULT_CAMERA_FRAMES_PER_CYCLE,
    DEFAULT_CAMERA_MEASUREMENT_REPEAT,
    CameraMeasurementRequest,
)
from .finite import PreparedFiniteCameraMeasurement
from .monitor import PreparedLiveCameraMeasurement


class CameraMeasurementApi:
    """Bound ``exp.nodes.camera_measurement`` surface."""

    __slots__ = ("_prepare", "_resolve_camera_ref")

    def __init__(
        self,
        *,
        resolve_camera_ref: Callable[[str | None], object],
        prepare: Callable[
            [CameraMeasurementRequest],
            PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement,
        ],
    ) -> None:
        if not callable(resolve_camera_ref) or not callable(prepare):
            raise TypeError("Camera Measurement operations must be callable")
        self._resolve_camera_ref = resolve_camera_ref
        self._prepare = prepare

    def camera_measurement_request(
        self,
        *,
        camera_role: str | None = None,
        repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT,
        frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE,
        exposure: float | None = None,
    ) -> CameraMeasurementRequest:
        """Freeze Main's one Camera semantic: 0=live, K=finite."""

        return CameraMeasurementRequest(
            camera_ref=self._resolve_camera_ref(camera_role),
            repeat=repeat,
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
        return self._prepare(request)


__all__ = [
    "CameraMeasurementApi",
]
