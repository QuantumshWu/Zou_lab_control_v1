"""Grey-molasses detuning's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import GreyMolassesDetuningApi
from .application import (
    prepare_grey_molasses_detuning,
)
from .measurement import (
    AutonomousMeasurementUnavailable,
    GREY_MOLASSES_CAPABILITY_GAP,
    GREY_MOLASSES_DETUNING_LOGIC_NODE,
)


def _bind_api(
    facts: tuple[object, ...],
    dependencies: tuple[object, ...],
) -> GreyMolassesDetuningApi:
    (calibration,) = dependencies
    (
        load_pulse,
        resolve_camera_ref,
        resolve_sequencer_ref,
        camera_port,
        pulse_port,
        rf_ports,
        start_run,
    ) = facts
    rf_by_role = dict(rf_ports)

    def resolve_rf_role(requested):
        if requested not in rf_by_role:
            raise AutonomousMeasurementUnavailable(GREY_MOLASSES_CAPABILITY_GAP)
        return requested

    def bind_request(request):
        rf_port = rf_by_role.get(request.rf_role)
        if rf_port is None:
            raise AutonomousMeasurementUnavailable(GREY_MOLASSES_CAPABILITY_GAP)
        return prepare_grey_molasses_detuning(
            request,
            calibration.load_calibration(request.calibration_ref),
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
            rf_port=rf_port,
            start_run=start_run,
        )

    return GreyMolassesDetuningApi(
        load_pulse=load_pulse,
        resolve_camera_ref=resolve_camera_ref,
        resolve_sequencer_ref=resolve_sequencer_ref,
        resolve_rf_role=resolve_rf_role,
        bind_request=bind_request,
    )


def _prepare_hosted(api, value, event_source):
    if event_source is not None:
        raise ValueError("Grey Molasses Detuning has no event-associated input")
    return api.prepare_grey_molasses_detuning_application(
        value.intent,
        value.calibration_ref,
    )


def _availability(catalog, _apparatus):
    return None if catalog.roles("rf") else "no installed RF role"


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="grey_molasses_detuning",
    declaration=GREY_MOLASSES_DETUNING_LOGIC_NODE,
    api_requirements=(
        "load_pulse",
        "resolve_camera_ref",
        "resolve_sequencer_ref",
        "camera_port",
        "pulse_port",
        "rf_ports",
        "start_run",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    api_dependencies=("calibration",),
    availability=_availability,
    dynamic_choice_fact="rf_roles",
)

__all__ = ["LOGIC_NODE_PACKAGE"]
