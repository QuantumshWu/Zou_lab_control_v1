"""Temperature release-recapture's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import LogicNodePackage
from zlc_neutral_atom.logic_nodes.release_recapture.application import (
    prepare_release_recapture,
)

from .api import TemperatureReleaseRecaptureApi
from .measurement import (
    BoundTemperatureReleaseRecapture,
    TEMPERATURE_RELEASE_RECAPTURE_LOGIC_NODE,
    TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS,
    bind_temperature_release_recapture,
    bind_temperature_release_recapture_inputs,
)


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
        resolved = calibration.load_calibration(request.calibration_ref)
        bound = bind_temperature_release_recapture(
            request,
            resolved,
            pulse_port=pulse_port(request.sequencer_ref),
            camera_port=camera_port(request.camera_ref),
        )
        if not isinstance(bound, BoundTemperatureReleaseRecapture):
            raise TypeError("temperature binder returned another domain value")
        return prepare_release_recapture(
            name=f"Temperature release-recapture {bound.program.document.name}",
            owner="zlc_neutral_atom.release-recapture",
            program_fingerprint=bound.program.fingerprint,
            camera_binding=bound.camera_binding,
            calibration=resolved,
            model_kind=bound.request.model_kind,
            per_site=bound.request.per_site,
            declaration=TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS[0],
            final_owner="temperature-release-recapture",
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
    return api.prepare_temperature_release_recapture(value)


def _bind_hosted_request(api, intent, inputs):
    return bind_temperature_release_recapture_inputs(
        intent,
        inputs,
        request_builder=api.temperature_release_recapture_request,
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
    bind_hosted_request=_bind_hosted_request,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
