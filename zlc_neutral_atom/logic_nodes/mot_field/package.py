"""MOT Field's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import MotFieldApi
from .application import prepare_mot_field_acquisition
from .mot_field import DEFAULT_MOT_FIELD_CAMERA_ROLE
from .mot_field_task import MOT_FIELD_LOGIC_NODE, prepare_mot_field_task


def _bind_api(host: object, _dependencies: tuple[object, ...]) -> MotFieldApi:
    operations = host._logic_node_operations()
    capture_repository = operations.capture_repository
    resolve_role = operations.resolve_role
    device_ref = operations.device_ref
    pulse_port = operations.pulse_port
    camera_port = operations.camera_port
    start_run = operations.start_run

    def resolve_camera_ref(requested):
        role = resolve_role(
            requested,
            "camera",
            (DEFAULT_MOT_FIELD_CAMERA_ROLE,),
        )
        if role != DEFAULT_MOT_FIELD_CAMERA_ROLE:
            raise ValueError(
                "MOT field optimization requires the installation's "
                "'mot_camera' role; an arbitrary camera is not a "
                "coil-sensitive exact-scan sensor"
            )
        return device_ref(role)

    def bind_acquisition(request):
        return prepare_mot_field_acquisition(
            request,
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
        )

    def bind_task(intent, api):
        return prepare_mot_field_task(
            intent,
            api,
            capture_repository=capture_repository,
            output_root=operations.output_root,
            start_run=start_run,
        )

    return MotFieldApi(
        load_pulse=host.load_readout_pulse,
        resolve_camera_ref=resolve_camera_ref,
        resolve_sequencer_ref=host.resolve_readout_sequencer_ref,
        prepare_acquisition=bind_acquisition,
        prepare_task=bind_task,
    )


def _bind_task_console(api: MotFieldApi, catalog: object, projection):
    from .workbench_adapter import start_mot_field_task_command

    return projection.run(
        MOT_FIELD_LOGIC_NODE,
        prepare=api.prepare_mot_field_task,
        dynamic_choice_context=catalog.roles("camera"),
        start_with_live_output=start_mot_field_task_command,
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="mot_field",
    declaration=MOT_FIELD_LOGIC_NODE,
    bind_api=_bind_api,
    bind_task_console=_bind_task_console,
    task_console_order=70,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
