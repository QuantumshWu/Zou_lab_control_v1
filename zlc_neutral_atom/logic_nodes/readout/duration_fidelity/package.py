"""Readout-duration fidelity's complete built-in capability package."""

from __future__ import annotations

from functools import partial

from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import ReadoutDurationFidelityApi
from .application import (
    prepare_bound_readout_duration_fidelity,
    prepare_readout_duration_fidelity,
)
from .measurement import READOUT_DURATION_FIDELITY_LOGIC_NODE


def _bind_api(
    host: object,
    dependencies: tuple[object, ...],
) -> ReadoutDurationFidelityApi:
    (calibration,) = dependencies
    operations = host._logic_node_operations()
    pulse_port = operations.pulse_port
    camera_port = operations.camera_port
    start_run = operations.start_run

    def bind_request(request):
        return prepare_readout_duration_fidelity(
            request,
            calibration.load_calibration(request.calibration_ref),
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
            start_run=start_run,
        )

    return ReadoutDurationFidelityApi(
        load_pulse=host.load_readout_pulse,
        resolve_camera_ref=host.resolve_readout_camera_ref,
        resolve_sequencer_ref=host.resolve_readout_sequencer_ref,
        bind_request=bind_request,
        wait_run=operations.wait_run,
    )


def _bind_task_console(api: ReadoutDurationFidelityApi, _catalog: object, projection):
    return projection.run(
        READOUT_DURATION_FIDELITY_LOGIC_NODE,
        prepare=partial(
            prepare_bound_readout_duration_fidelity,
            application=api,
        ),
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="readout_duration_fidelity",
    declaration=READOUT_DURATION_FIDELITY_LOGIC_NODE,
    bind_api=_bind_api,
    bind_task_console=_bind_task_console,
    task_console_order=30,
    api_dependencies=("calibration",),
)

__all__ = ["LOGIC_NODE_PACKAGE"]
