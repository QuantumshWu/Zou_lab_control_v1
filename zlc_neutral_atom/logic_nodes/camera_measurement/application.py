"""Application binding for one authored Camera Measurement intent."""

from __future__ import annotations

from collections.abc import Callable

from zlc_neutral_atom.node_input import BoundNodeInputs, bind_no_node_inputs

from .definition import CameraMeasurementIntent, CameraMeasurementRequest


def bind_camera_measurement_intent(
    intent: CameraMeasurementIntent,
    inputs: BoundNodeInputs,
    *,
    request_builder: Callable[..., object],
) -> CameraMeasurementRequest:
    """Bind an authored Camera intent to the installed application request."""

    authored = bind_no_node_inputs(intent, inputs)
    if not isinstance(authored, CameraMeasurementIntent):
        raise TypeError("Camera binder requires CameraMeasurementIntent")
    if not callable(request_builder):
        raise TypeError("Camera request builder must be callable")
    request = request_builder(
        camera_role=authored.camera_role,
        repeat=authored.repeat,
        frames_per_cycle=authored.frames_per_cycle,
        exposure=authored.exposure_seconds,
    )
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("Camera request builder returned another request type")
    return request


__all__ = ["bind_camera_measurement_intent"]
