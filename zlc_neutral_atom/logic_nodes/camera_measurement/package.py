"""Camera Measurement's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import CameraMeasurementApi
from .application import (
    bind_camera_measurement_intent,
    start_camera_measurement_command,
)
from .definition import CAMERA_MEASUREMENT_LOGIC_NODE
from .finite import prepare_finite_camera_measurement
from .monitor import prepare_live_camera_measurement


def _bind_api(
    facts: tuple[object, ...],
    _dependencies: tuple[object, ...],
) -> CameraMeasurementApi:
    (
        resolve_camera_ref,
        camera_port,
        camera_monitor_port,
        association_authorities,
        capture_repository,
        start_run,
    ) = facts

    def prepare(request):
        if request.repeat == 0:
            return prepare_live_camera_measurement(
                request,
                monitor_port=camera_monitor_port(request.camera_ref),
                start_run=start_run,
                association_authority=association_authorities.get(
                    request.camera_ref.role
                ),
            )
        return prepare_finite_camera_measurement(
            request,
            camera_port=camera_port(request.camera_ref),
            repository=capture_repository,
            start_run=start_run,
        )

    return CameraMeasurementApi(
        resolve_camera_ref=resolve_camera_ref,
        prepare=prepare,
    )


def _bind_hosted_request(api, intent, inputs):
    return bind_camera_measurement_intent(
        intent,
        inputs,
        request_builder=api.camera_measurement_request,
    )


def _prepare_hosted(api, request, event_source):
    if event_source is not None:
        raise ValueError("Camera Measurement has no event-associated input")
    return api.prepare_camera_measurement(request)


def _availability(catalog, _apparatus):
    return None if catalog.roles("camera") else "no installed Camera role"


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="camera_measurement",
    declaration=CAMERA_MEASUREMENT_LOGIC_NODE,
    api_requirements=(
        "resolve_camera_ref",
        "camera_port",
        "camera_monitor_port",
        "camera_signal_association_authorities",
        "capture_repository",
        "start_run",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    availability=_availability,
    dynamic_choice_fact="camera_roles",
    bind_hosted_request=_bind_hosted_request,
    start_prepared=start_camera_measurement_command,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
