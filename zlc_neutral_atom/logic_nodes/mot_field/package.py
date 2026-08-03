"""MOT Field's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage
from zlc_pulse import PulseDocument, load_pulse_document
from zlc_storage.paths import resolve_under

from .api import MotFieldApi
from .application import prepare_mot_field_acquisition
from .mot_field_task import (
    MOT_FIELD_LOGIC_NODE,
    PreparedMotFieldTask,
    prepare_mot_field_task,
    start_mot_field_task_command,
)


def _bind_api(
    facts: tuple[object, ...],
    _dependencies: tuple[object, ...],
) -> MotFieldApi:
    (
        pulses_root,
        captures_root,
        resolve_camera_ref,
        resolve_sequencer_ref,
        mot_camera_port,
        pulse_port,
        start_run,
    ) = facts

    def load_pulse(value):
        if isinstance(value, PulseDocument):
            return value
        return load_pulse_document(resolve_under(pulses_root, value))

    def bind_acquisition(request):
        return prepare_mot_field_acquisition(
            request,
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=mot_camera_port(request.camera_ref),
            captures_root=captures_root,
            start_run=start_run,
        )

    def bind_task(intent, api):
        return prepare_mot_field_task(intent, api)

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
    command = api.prepare_mot_field_task(request)
    if not isinstance(command, PreparedMotFieldTask):
        raise TypeError("MOT-field preparer returned another command type")
    return command


def _availability(catalog, _apparatus):
    if not any(
        item.domain == "camera" and "camera.mot_field_capture" in item.capabilities
        for item in catalog.values()
    ):
        return "MOT Field requires a camera with MOT-field capture capability"
    if not any(
        item.domain == "sequencer" and "pulse.execute" in item.capabilities
        for item in catalog.values()
    ):
        return "MOT Field requires a pulse sequencer"
    return None


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="mot_field",
    declaration=MOT_FIELD_LOGIC_NODE,
    api_requirements=(
        "pulses_root",
        "captures_root",
        "resolve_mot_camera_ref",
        "resolve_sequencer_ref",
        "mot_camera_port",
        "pulse_port",
        "start_run",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    availability=_availability,
    dynamic_choice_fact="mot_camera_roles",
    start_prepared=start_mot_field_task_command,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
