"""Occupancy Processor presentation and lifecycle attachment."""

from __future__ import annotations

from zlc_neutral_atom.logic_nodes.occupancy.processor import (
    OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
    OCCUPANCY_PROCESSOR_DEFINITION,
    build_occupancy_processor_config,
    occupancy_authoring_schema,
    occupancy_input_specs,
)
from zlc_neutral_atom.logic_nodes.occupancy.processor_application import (
    OccupancyProcessorRequest,
    bind_occupancy_processor_request,
)
from zlc_workbench.form_projection import PathPresentation, project_authoring_form
from zlc_workbench.input_binding import project_input_fields
from zlc_workbench.logic_node_presentations.occupancy import (
    materialize_occupancy_publication,
)
from zlc_workbench.task_console.catalog_bridge import (
    ConsoleNodeSpec,
    ConsoleSignalDecl,
)

from ._common import processor_attachment


def occupancy_attachment(*, prepare):
    """Bind Occupancy's own inputs/request to the generic Processor host."""

    inputs = occupancy_input_specs()
    spec = ConsoleNodeSpec(
        definition=OCCUPANCY_PROCESSOR_DEFINITION,
        title="Judge occupancy",
        description="Classify each admitted Camera revision with one Calibration",
        form=project_authoring_form(occupancy_authoring_schema()),
        declared_outputs=(
            ConsoleSignalDecl(
                OCCUPANCY_LIVE_OUTPUT_DECLARATIONS[0],
                "counts",
                "Counts",
                "site counts",
            ),
            ConsoleSignalDecl(
                OCCUPANCY_LIVE_OUTPUT_DECLARATIONS[1],
                "occupied",
                "Occupancy",
                "site occupancy",
            ),
            ConsoleSignalDecl(
                OCCUPANCY_LIVE_OUTPUT_DECLARATIONS[2],
                "rate",
                "Loading rate",
                "valid-site occupancy fraction for each repeat/point cell",
            ),
        ),
        build_request=build_occupancy_processor_config,
        input_specs=inputs,
        input_fields=project_input_fields(
            inputs,
            path_presentations={
                "calibration": PathPresentation(
                    mode="file",
                    file_filter=(
                        "Calibration pointer (calibration_ref.json);;"
                        "JSON files (*.json)"
                    ),
                    base_dir="_output/calibrations",
                )
            },
        ),
    )

    def prepare_processor(request):
        if not isinstance(request, OccupancyProcessorRequest):
            raise TypeError("Occupancy owner returned another request type")
        return prepare(request)

    return processor_attachment(
        spec,
        bind_request=bind_occupancy_processor_request,
        prepare=prepare_processor,
        materialize_publication=materialize_occupancy_publication,
    )


__all__ = ["occupancy_attachment"]
