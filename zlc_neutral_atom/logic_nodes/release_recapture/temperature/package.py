"""Temperature release-recapture's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import TemperatureReleaseRecaptureApi
from .application import (
    prepare_temperature_release_recapture,
)
from .measurement import TEMPERATURE_RELEASE_RECAPTURE_LOGIC_NODE


def _bind_api(
    facts: tuple[object, ...],
    dependencies: tuple[object, ...],
) -> TemperatureReleaseRecaptureApi:
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
        return prepare_temperature_release_recapture(
            request,
            calibration.load_calibration(request.calibration_ref),
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
            start_run=start_run,
        )

    return TemperatureReleaseRecaptureApi(
        load_pulse=load_pulse,
        resolve_camera_ref=resolve_camera_ref,
        resolve_sequencer_ref=resolve_sequencer_ref,
        bind_request=bind_request,
        wait_run=wait_run,
    )


def _prepare_hosted(api, value, event_source):
    if event_source is not None:
        raise ValueError("Temperature release-recapture has no event-associated input")
    return api.prepare_temperature_release_recapture_application(
        value.intent,
        value.calibration_ref,
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="temperature",
    declaration=TEMPERATURE_RELEASE_RECAPTURE_LOGIC_NODE,
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
)

__all__ = ["LOGIC_NODE_PACKAGE"]
