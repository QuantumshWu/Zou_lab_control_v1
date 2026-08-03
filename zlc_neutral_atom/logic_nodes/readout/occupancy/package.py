"""Occupancy's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.logic_node_package import (
    LogicNodePackage,
    UiContributionDescriptor,
)
from zlc_storage.paths import resolve_under

from .api import OccupancyApi
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from .declaration import OCCUPANCY_LOGIC_NODE


def _bind_api(
    facts: tuple[object, ...],
    dependencies: tuple[object, ...],
) -> OccupancyApi:
    (calibration,) = dependencies
    (
        output_root,
        start_run,
        wait_run,
        open_ui,
    ) = facts

    return OccupancyApi(
        calibration,
        captures_root=resolve_under(output_root, "captures"),
        calibrations_root=resolve_under(output_root, "calibrations"),
        occupancy_root=resolve_under(output_root, "occupancy"),
        start_run=start_run,
        wait_run=wait_run,
        open_ui=open_ui,
    )


def _prepare_hosted(api, request, event_source):
    if event_source is not None:
        raise ValueError(
            "Occupancy event source is bound by the generic Processor host"
        )
    return api.prepare_occupancy_processor_request(request)


def _resolve_artifact_reference(api, binding, resolve_final_or_saved):
    def require_reference(value: object) -> CalibrationArtifactRef:
        if not isinstance(value, CalibrationArtifactRef):
            raise TypeError("Occupancy Calibration input is not a typed reference")
        return value

    reference = resolve_final_or_saved(
        binding,
        load_saved=api._reference_from_record_path,
        extract_reference=require_reference,
    )
    return require_reference(reference)


def _close_api(api: OccupancyApi) -> tuple[Exception, ...]:
    return api.close()


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="occupancy",
    declaration=OCCUPANCY_LOGIC_NODE,
    api_requirements=(
        "output_root",
        "start_run",
        "wait_run",
        "open_ui",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    api_dependencies=("calibration",),
    resolve_artifact_reference=_resolve_artifact_reference,
    ui_contributions=(
        UiContributionDescriptor(
            "cell",
            "zlc_neutral_atom.logic_nodes.readout.occupancy.ui.workbench_window",
            "OccupancyCellWindow",
        ),
    ),
    close_api=_close_api,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
