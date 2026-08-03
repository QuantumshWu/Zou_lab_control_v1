"""The sole discovered descriptor for temperature release-recapture."""

from __future__ import annotations

from zlc_neutral_atom.logic_node import DatasetOutputSpec, LogicNodeApplicationContext, LogicNodeDescriptor
from zlc_neutral_atom.logic_nodes.readout.calibration.artifact import load_calibration_artifact
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.readout.calibration_input import CALIBRATION_INPUT_SPEC
from zlc_neutral_atom.logic_nodes.release_recapture.application import (
    compile_release_recapture,
    release_recapture_final_outputs,
)
from zlc_neutral_atom.runtime.hosted_run import LogicNodeExecutionContext
from zlc_pulse import load_pulse_document
from zlc_storage import resolve_under

from .measurement import (
    TEMPERATURE_RELEASE_RECAPTURE_DEFINITION,
    TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATION,
    TemperatureReleaseRecaptureRequest,
    bind_temperature_release_recapture,
    build_temperature_release_recapture_request,
    temperature_release_recapture_authoring_schema,
)


def _bind_execute(request: object, context: LogicNodeApplicationContext):
    if not isinstance(request, TemperatureReleaseRecaptureRequest):
        raise TypeError("temperature request has another type")
    reference = context.input(CALIBRATION_INPUT_SPEC)
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("temperature Calibration input has another type")
    calibration = load_calibration_artifact(context.project_root, reference)
    pulse_port = context.device("sequencer_instance_id", "pulse.execute")
    camera_port = context.device("camera_instance_id", "camera.capture")
    document = load_pulse_document(resolve_under(context.pulses_root, request.pulse))
    binding = bind_temperature_release_recapture(
        request,
        document,
        calibration,
        pulse_port=pulse_port,
        camera_port=camera_port,
    )

    def execute(execution: LogicNodeExecutionContext):
        plan = compile_release_recapture(
            name=f"Temperature release-recapture {document.name}",
            camera_binding=binding,
            calibration=calibration,
            model_kind=None,
            per_site=request.per_site,
        ).with_lifecycle(owner=execution, preemptible=False)
        result = execution.start_and_wait(lambda: context.start_run(plan))
        execution.publish_final(
            release_recapture_final_outputs(
                result,
                TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATION,
            )
        )
        return result

    return execute


LOGIC_NODE = LogicNodeDescriptor(
    api_name="temperature",
    definition=TEMPERATURE_RELEASE_RECAPTURE_DEFINITION,
    description="Autonomous hardware-timed release-recapture temperature scan",
    authoring_schema=temperature_release_recapture_authoring_schema(),
    input_specs=(CALIBRATION_INPUT_SPEC,),
    outputs=(
        DatasetOutputSpec(
            TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATION,
            "survival",
            "Survival",
        ),
    ),
    build_request=build_temperature_release_recapture_request,
    bind_execute=_bind_execute,
    device_requirements=(
        ("camera_instance_id", ("camera.capture",)),
        ("sequencer_instance_id", ("pulse.execute",)),
    ),
)


__all__ = ["LOGIC_NODE"]
