"""Readout-duration fidelity's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import ReadoutDurationFidelityApi
from .application import prepare_readout_duration_fidelity
from .measurement import (
    READOUT_DURATION_FIDELITY_LOGIC_NODE,
    bind_readout_duration_fidelity_inputs,
)


def _bind_api(
    facts: tuple[object, ...],
    dependencies: tuple[object, ...],
) -> ReadoutDurationFidelityApi:
    (calibration,) = dependencies
    (
        load_pulse,
        resolve_camera_ref,
        resolve_sequencer_ref,
        camera_port,
        pulse_port,
        start_run,
        wait_run,
    ) = facts

    def bind_request(request):
        return prepare_readout_duration_fidelity(
            request,
            calibration.load_calibration(request.calibration_ref),
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
            start_run=start_run,
        )

    return ReadoutDurationFidelityApi(
        load_pulse=load_pulse,
        resolve_camera_ref=resolve_camera_ref,
        resolve_sequencer_ref=resolve_sequencer_ref,
        bind_request=bind_request,
        wait_run=wait_run,
    )


def _prepare_hosted(api, value, event_source):
    if event_source is not None:
        raise ValueError("Readout Duration Fidelity has no event-associated input")
    return api.prepare_readout_duration_fidelity(value)


def _bind_hosted_request(api, intent, inputs):
    return bind_readout_duration_fidelity_inputs(
        intent,
        inputs,
        request_builder=api.readout_duration_fidelity_request,
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="readout_duration_fidelity",
    declaration=READOUT_DURATION_FIDELITY_LOGIC_NODE,
    api_requirements=(
        "load_pulse",
        "resolve_camera_ref",
        "resolve_sequencer_ref",
        "camera_port",
        "pulse_port",
        "start_run",
        "wait_run",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    api_dependencies=("calibration",),
    bind_hosted_request=_bind_hosted_request,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
