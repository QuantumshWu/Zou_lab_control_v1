"""Grey-molasses detuning's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage
from zlc_neutral_atom.logic_nodes.release_recapture.application import (
    prepare_release_recapture,
)

from .api import GreyMolassesDetuningApi
from .measurement import (
    BoundGreyMolassesDetuning,
    GREY_MOLASSES_CAPABILITY_GAP,
    GREY_MOLASSES_DETUNING_LOGIC_NODE,
    GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS,
    bind_grey_molasses_detuning,
    bind_grey_molasses_detuning_inputs,
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
        wait_run,
    ) = facts
    rf_by_role = dict(rf_ports)

    def resolve_rf_role(requested):
        if requested is None and len(rf_by_role) == 1:
            return next(iter(rf_by_role))
        if requested not in rf_by_role:
            raise RuntimeError(GREY_MOLASSES_CAPABILITY_GAP)
        return requested

    def bind_request(request):
        rf_port = rf_by_role.get(request.rf_role)
        if rf_port is None:
            raise RuntimeError(GREY_MOLASSES_CAPABILITY_GAP)
        resolved = calibration.load_calibration(request.calibration_ref)
        bound = bind_grey_molasses_detuning(
            request,
            resolved,
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
            rf_port=rf_port,
        )
        if not isinstance(bound, BoundGreyMolassesDetuning):
            raise TypeError("Grey-molasses binder returned another domain value")
        return prepare_release_recapture(
            name=f"Grey molasses detuning {bound.program.document.name}",
            owner="zlc_neutral_atom.grey-molasses-detuning",
            program_fingerprint=bound.program.fingerprint,
            camera_binding=bound.camera_binding,
            calibration=resolved,
            model_kind=bound.request.model_kind,
            per_site=bound.request.per_site,
            declaration=GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS[0],
            final_owner="grey-molasses-detuning",
            start_run=start_run,
            rf_port=bound.rf_port,
            rf_table=bound.rf_table,
        )

    return GreyMolassesDetuningApi(
        load_pulse=load_pulse,
        resolve_camera_ref=resolve_camera_ref,
        resolve_sequencer_ref=resolve_sequencer_ref,
        resolve_rf_role=resolve_rf_role,
        bind_request=bind_request,
        wait_run=wait_run,
    )


def _prepare_hosted(api, value, event_source):
    if event_source is not None:
        raise ValueError("Grey Molasses Detuning has no event-associated input")
    return api.prepare_grey_molasses_detuning(value)


def _bind_hosted_request(api, intent, inputs):
    return bind_grey_molasses_detuning_inputs(
        intent,
        inputs,
        request_builder=api.grey_molasses_detuning_request,
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
        "wait_run",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    api_dependencies=("calibration",),
    availability=_availability,
    dynamic_choice_fact="rf_roles",
    bind_hosted_request=_bind_hosted_request,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
