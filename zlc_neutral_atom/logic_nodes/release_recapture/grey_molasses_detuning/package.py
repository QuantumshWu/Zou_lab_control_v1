"""Grey-molasses detuning's complete built-in capability package."""

from __future__ import annotations

from functools import partial

from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import GreyMolassesDetuningApi
from .application import (
    prepare_bound_grey_molasses_detuning,
    prepare_grey_molasses_detuning,
)
from .measurement import (
    AutonomousMeasurementUnavailable,
    GREY_MOLASSES_CAPABILITY_GAP,
    GREY_MOLASSES_DETUNING_LOGIC_NODE,
)


def _bind_api(
    host: object,
    dependencies: tuple[object, ...],
) -> GreyMolassesDetuningApi:
    (calibration,) = dependencies
    operations = host._logic_node_operations()
    rf_roles = operations.roles("rf")
    resolve_role = operations.resolve_role
    device_domain = operations.device_domain
    device_ref = operations.device_ref
    pulse_port = operations.pulse_port
    camera_port = operations.camera_port
    rf_port = operations.rf_port
    start_run = operations.start_run

    def resolve_rf_role(requested):
        if not rf_roles:
            return requested
        return resolve_role(requested, "rf", ("rf",))

    def bind_request(request):
        if device_domain(request.rf_role) != "rf":
            raise AutonomousMeasurementUnavailable(GREY_MOLASSES_CAPABILITY_GAP)
        return prepare_grey_molasses_detuning(
            request,
            calibration.load_calibration(request.calibration_ref),
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
            rf_port=rf_port(device_ref(request.rf_role)),
            start_run=start_run,
        )

    return GreyMolassesDetuningApi(
        load_pulse=host.load_readout_pulse,
        resolve_camera_ref=host.resolve_readout_camera_ref,
        resolve_sequencer_ref=host.resolve_readout_sequencer_ref,
        resolve_rf_role=resolve_rf_role,
        bind_request=bind_request,
    )


def _bind_task_console(api: GreyMolassesDetuningApi, catalog: object, projection):
    return projection.run(
        GREY_MOLASSES_DETUNING_LOGIC_NODE,
        prepare=partial(
            prepare_bound_grey_molasses_detuning,
            application=api,
        ),
        dynamic_choice_context=catalog.roles("rf"),
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="grey_molasses_detuning",
    declaration=GREY_MOLASSES_DETUNING_LOGIC_NODE,
    bind_api=_bind_api,
    bind_task_console=_bind_task_console,
    task_console_order=40,
    api_dependencies=("calibration",),
)

__all__ = ["LOGIC_NODE_PACKAGE"]
