"""MOT Field's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage
from zlc_pulse import PulseDocument, load_pulse_document
from zlc_storage.paths import resolve_under

from .api import MotFieldApi
from .application import prepare_mot_field_acquisition
from .mot_field import DEFAULT_MOT_FIELD_CAMERA_ROLE
from .mot_field_task import (
    MOT_FIELD_LOGIC_NODE,
    prepare_mot_field_task,
    start_mot_field_task_command,
)


def _bind_api(
    facts: tuple[object, ...],
    _dependencies: tuple[object, ...],
) -> MotFieldApi:
    (
        pulses_root,
        output_root,
        capture_repository,
        installed_camera_ref,
        resolve_sequencer_ref,
        camera_port,
        pulse_port,
        start_run,
    ) = facts

    def resolve_camera_ref(requested):
        if requested not in (None, DEFAULT_MOT_FIELD_CAMERA_ROLE):
            raise ValueError(
                "MOT field optimization requires the installation's "
                "'mot_camera' role; an arbitrary camera is not a "
                "coil-sensitive exact-scan sensor"
            )
        return installed_camera_ref(DEFAULT_MOT_FIELD_CAMERA_ROLE)

    def load_pulse(value):
        if isinstance(value, PulseDocument):
            return value
        return load_pulse_document(resolve_under(pulses_root, value))

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
            output_root=output_root,
            start_run=start_run,
        )

    return MotFieldApi(
        load_pulse=load_pulse,
        resolve_camera_ref=resolve_camera_ref,
        resolve_sequencer_ref=resolve_sequencer_ref,
        prepare_acquisition=bind_acquisition,
        prepare_task=bind_task,
    )


def _prepare_hosted(api, request, event_source):
    if event_source is not None:
        raise ValueError("MOT Field has no event-associated input")
    return api.prepare_mot_field_task(request)


def _availability(catalog, _apparatus):
    camera = catalog.find(DEFAULT_MOT_FIELD_CAMERA_ROLE)
    sequencer = catalog.find("sequencer")
    if camera is None or camera.domain != "camera":
        return "MOT Field requires the installed mot_camera Camera role"
    if sequencer is None or sequencer.domain != "sequencer":
        return "MOT Field requires the installed Sequencer role"
    return None


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="mot_field",
    declaration=MOT_FIELD_LOGIC_NODE,
    api_requirements=(
        "pulses_root",
        "output_root",
        "capture_repository",
        "resolve_camera_ref",
        "resolve_sequencer_ref",
        "camera_port",
        "pulse_port",
        "start_run",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    availability=_availability,
    dynamic_choice_fact="camera_roles",
    start_prepared=start_mot_field_task_command,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
