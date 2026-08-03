"""The sole descriptor for hardware-clocked grey-molasses detuning."""

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
    GREY_MOLASSES_DETUNING_DEFINITION,
    GREY_MOLASSES_DETUNING_OUTPUT_DECLARATION,
    GreyMolassesDetuningRequest,
    bind_grey_molasses_detuning,
    build_grey_molasses_detuning_request,
    grey_molasses_detuning_authoring_schema,
)


def _bind_execute(request: object, context: LogicNodeApplicationContext):
    if not isinstance(request, GreyMolassesDetuningRequest):
        raise TypeError("grey-molasses request has another type")
    reference = context.input(CALIBRATION_INPUT_SPEC)
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("grey-molasses Calibration input has another type")
    calibration = load_calibration_artifact(context.project_root, reference)
    pulse_port = context.device("sequencer_instance_id", "pulse.execute")
    camera_port = context.device("camera_instance_id", "camera.capture")
    rf_port = context.device("rf_instance_id", "rf.table")
    document = load_pulse_document(resolve_under(context.pulses_root, request.pulse))
    binding, rf_table = bind_grey_molasses_detuning(
        request,
        document,
        calibration,
        pulse_port=pulse_port,
        camera_port=camera_port,
    )

    def execute(execution: LogicNodeExecutionContext):
        plan = compile_release_recapture(
            name=f"Grey molasses detuning {document.name}",
            camera_binding=binding,
            calibration=calibration,
            model_kind=None,
            per_site=request.per_site,
            rf_port=rf_port,
            rf_table=rf_table,
        ).with_lifecycle(owner=execution, preemptible=False)
        result = execution.start_and_wait(lambda: context.start_run(plan))
        execution.publish_final(
            release_recapture_final_outputs(
                result,
                GREY_MOLASSES_DETUNING_OUTPUT_DECLARATION,
            )
        )
        return result

    return execute


LOGIC_NODE = LogicNodeDescriptor(
    api_name="grey_molasses_detuning",
    definition=GREY_MOLASSES_DETUNING_DEFINITION,
    description="Hardware-clocked RF detuning release-recapture scan",
    authoring_schema=grey_molasses_detuning_authoring_schema(),
    input_specs=(CALIBRATION_INPUT_SPEC,),
    outputs=(
        DatasetOutputSpec(
            GREY_MOLASSES_DETUNING_OUTPUT_DECLARATION,
            "recapture",
            "Recapture rate",
        ),
    ),
    build_request=build_grey_molasses_detuning_request,
    bind_execute=_bind_execute,
    device_requirements=(
        ("camera_instance_id", ("camera.capture",)),
        ("sequencer_instance_id", ("pulse.execute",)),
        ("rf_instance_id", ("rf.table",)),
    ),
)


__all__ = ["LOGIC_NODE"]
