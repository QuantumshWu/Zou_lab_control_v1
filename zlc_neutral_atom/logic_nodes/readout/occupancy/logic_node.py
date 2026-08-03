"""The sole discovered declaration and operation for Occupancy."""

from __future__ import annotations

from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.logic_node import (
    DatasetOutputSpec,
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.artifact import (
    load_calibration_artifact,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    calibration_artifact_ref_from_input,
)
from zlc_neutral_atom.logic_nodes.readout.calibration_input import (
    CALIBRATION_INPUT_SPEC,
)
from zlc_neutral_atom.processing.signal_plane import SignalValue
from .processor import (
    OCCUPANCY_CAMERA_INPUT_SPEC,
    OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
    OccupancyProcessorConfig,
    _evaluate_occupancy_processor,
    build_occupancy_processor_config,
    occupancy_authoring_schema,
)


def _bind_execute(
    request: object,
    context: LogicNodeApplicationContext,
):
    if not isinstance(request, OccupancyProcessorConfig):
        raise TypeError("Occupancy request must be OccupancyProcessorConfig")
    reference = calibration_artifact_ref_from_input(
        context.project_root,
        context.input(CALIBRATION_INPUT_SPEC),
    )
    calibration = load_calibration_artifact(context.project_root, reference)
    model_kind = calibration.artifact.select_model(request.model_kind).kind

    def evaluate(source: SignalValue):
        if not isinstance(source, SignalValue):
            raise TypeError("Occupancy input must be SignalValue")
        return _evaluate_occupancy_processor(
            source.snapshot,
            calibration,
            source.coverage,
            model_kind=model_kind,
        )

    return evaluate


LOGIC_NODE = LogicNodeDescriptor(
    api_name="occupancy",
    definition=LogicNodeDefinition(
        DefinitionKey(
            "zlc_neutral_atom.logic_nodes.readout.occupancy",
            "occupancy-processor",
        ),
        "Judge occupancy",
        "processor",
    ),
    description="Classify each current frame with the selected Calibration",
    authoring_schema=occupancy_authoring_schema(),
    input_specs=(OCCUPANCY_CAMERA_INPUT_SPEC, CALIBRATION_INPUT_SPEC),
    outputs=tuple(
        DatasetOutputSpec(declaration, declaration.name)
        for declaration in OCCUPANCY_LIVE_OUTPUT_DECLARATIONS
    ),
    build_request=build_occupancy_processor_config,
    bind_execute=_bind_execute,
)


__all__ = ["LOGIC_NODE"]
