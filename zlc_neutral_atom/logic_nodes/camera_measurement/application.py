"""Application binding for one authored Camera Measurement intent."""

from __future__ import annotations

from collections.abc import Callable

from zlc_neutral_atom.node_input import BoundNodeInputs, bind_no_node_inputs

from .definition import CameraMeasurementIntent, CameraMeasurementRequest
from .finite import PreparedFiniteCameraMeasurement
from .monitor import PreparedLiveCameraMeasurement


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


def start_camera_measurement_command(
    command: PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement,
    live_output_host,
    command_context,
):
    """Start the leaf-owned live/finite shape through one generic host."""

    cancel_requested = getattr(command_context, "cancel_requested", None)
    if not callable(cancel_requested):
        raise TypeError("Camera start requires a hosted command context")
    if isinstance(command, PreparedLiveCameraMeasurement):
        factory = getattr(live_output_host, "factory", None)
        if not callable(factory):
            raise TypeError("Camera live start requires a live-output host")
        live_factory = factory(output_owner=command)
        return command.start_with_view(
            factory=live_factory,
            lifecycle_owner=command_context,
        )
    if not isinstance(command, PreparedFiniteCameraMeasurement):
        raise TypeError("Camera command has another type")
    open_exact_dataset = getattr(live_output_host, "open_exact_dataset", None)
    if not callable(open_exact_dataset):
        raise TypeError("Camera finite start requires an exact Dataset host")
    preview = open_exact_dataset(
        command.preview_spec,
        projection=command.live_projection(),
    )
    return command.start(
        preview,
        lifecycle_owner=command_context,
    )


__all__ = [
    "bind_camera_measurement_intent",
    "start_camera_measurement_command",
]
