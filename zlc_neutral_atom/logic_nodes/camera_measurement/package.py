"""Camera Measurement's complete built-in capability package."""

from __future__ import annotations

from functools import partial

from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import CameraMeasurementApi
from .application import bind_camera_measurement_intent
from .definition import CAMERA_MEASUREMENT_LOGIC_NODE
from .finite import prepare_finite_camera_measurement
from .monitor import prepare_live_camera_measurement


def _bind_api(host: object, _dependencies: tuple[object, ...]) -> CameraMeasurementApi:
    operations = host._logic_node_operations()
    association_authorities = operations.camera_signal_association_authorities
    capture_repository = operations.capture_repository
    camera_monitor_port = operations.camera_monitor_port
    camera_port = operations.camera_port
    start_run = operations.start_run

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
        resolve_camera_ref=host.resolve_readout_camera_ref,
        prepare=prepare,
    )


def _bind_task_console(api: CameraMeasurementApi, catalog: object, projection):
    from .workbench_adapter import start_camera_measurement_command

    return projection.run(
        CAMERA_MEASUREMENT_LOGIC_NODE,
        bind_request=partial(
            bind_camera_measurement_intent,
            request_builder=api.camera_measurement_request,
        ),
        prepare=api.prepare_camera_measurement,
        dynamic_choice_context=catalog.roles("camera"),
        start_with_live_output=start_camera_measurement_command,
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="camera_measurement",
    declaration=CAMERA_MEASUREMENT_LOGIC_NODE,
    bind_api=_bind_api,
    bind_task_console=_bind_task_console,
    task_console_order=10,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
