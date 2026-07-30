"""Headless declaration of the Occupancy Processor capability."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_declaration import (
    DefaultOutputView,
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
)

from .processor import (
    OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
    OCCUPANCY_PROCESSOR_DEFINITION,
    OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION,
    build_occupancy_processor_config,
    occupancy_authoring_schema,
    occupancy_input_specs,
)
from .processor_application import bind_occupancy_processor_request


_OUTPUTS = OCCUPANCY_LIVE_OUTPUT_DECLARATIONS

OCCUPANCY_LOGIC_NODE = LogicNodeDeclaration(
    definition=OCCUPANCY_PROCESSOR_DEFINITION,
    description="Classify each admitted Camera revision with one Calibration",
    authoring_schema=occupancy_authoring_schema(),
    input_specs=occupancy_input_specs(),
    outputs=(
        OutputPresentation(_OUTPUTS[0], "counts", "Counts", "site counts"),
        OutputPresentation(
            _OUTPUTS[1],
            "occupied",
            "Occupancy",
            "site occupancy",
        ),
        OutputPresentation(
            _OUTPUTS[2],
            "rate",
            "Loading rate",
            "valid-site occupancy fraction for each repeat/point cell",
        ),
    ),
    build_request=build_occupancy_processor_config,
    bind_request=bind_occupancy_processor_request,
    default_views=(
        DefaultOutputView(OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION.name, "sites"),
    ),
    input_path_presentations=(
        PathPresentationHint(
            "calibration",
            file_filter=(
                "Calibration record (calibration.json);;JSON files (*.json)"
            ),
            base_dir="output/calibrations",
        ),
    ),
)


__all__ = ["OCCUPANCY_LOGIC_NODE"]
