"""Occupancy result-to-presentation adapter for TaskConsole."""

from __future__ import annotations

from zlc_frontend.site_map_render import build_occupancy_cell_view
from zlc_neutral_atom.logic_nodes.occupancy.processor import (
    OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION,
    OccupancyProcessorEvaluation,
)

from zlc_workbench.task_console.data_plane import ConsoleSignalValue
from zlc_workbench.task_console.processor_node import ConsoleProcessorPublication

__all__ = ["materialize_occupancy_publication"]


def materialize_occupancy_publication(
    result: object,
    source: ConsoleSignalValue,
) -> ConsoleProcessorPublication:
    """Attach the typed same-shot SiteMap view to an Occupancy evaluation."""

    if not isinstance(result, OccupancyProcessorEvaluation):
        raise TypeError("Occupancy application returned another result type")
    if not isinstance(source, ConsoleSignalValue):
        raise TypeError("source must be ConsoleSignalValue")
    site_map = result.site_map
    calibration_identity = result.calibration_ref.target_ref
    presentation = build_occupancy_cell_view(
        result.background_value,
        result.background_ref,
        result.occupied_value,
        result.occupied_ref,
        result.selection,
        site_axis=site_map.site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        calibration_site_validity=site_map.validity.mask,
        calibration_identity=calibration_identity,
        run_id=source.run_id,
        provenance_epoch_id=source.epoch_id,
        summary=(
            f"source run={source.run_id} | "
            f"calibration={calibration_identity} | "
            f"revision={result.background_ref.revision.value} | "
            f"logical point={result.logical_point}"
        ),
    )
    occupied_name = OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION.name
    return ConsoleProcessorPublication(
        result.outputs,
        {occupied_name: presentation},
    )
