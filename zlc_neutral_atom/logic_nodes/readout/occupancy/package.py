"""Occupancy's complete built-in capability package."""

from __future__ import annotations

from functools import partial

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.logic_node_package import LogicNodePackage

from .api import OccupancyApi
from .application import prepare_detection_plan
from .cell import (
    inspect_occupancy_cell_domain,
    load_exact_occupancy_cell_source,
)
from .declaration import OCCUPANCY_LOGIC_NODE
from .reference import OCCUPANCY_ARTIFACT_NAMESPACE, OccupancyArtifactRef


def _bind_api(host: object, dependencies: tuple[object, ...]) -> OccupancyApi:
    (calibration,) = dependencies
    operations = host._logic_node_operations()
    capture_repository = operations.capture_repository
    start_run = operations.start_run

    def calibration_repository():
        return calibration._repository_for_readout_family()

    def start_detection(request, occupancy_repository):
        plan = prepare_detection_plan(
            request,
            capture_repository=capture_repository,
            calibration_repository=calibration_repository(),
            occupancy_repository=occupancy_repository,
        )
        return start_run(plan)

    def load_occupancy(reference, occupancy_repository):
        return occupancy_repository.admit(
            reference,
            capture_repository,
            calibration_repository(),
        )

    def inspect_cell(reference, occupancy_repository):
        return inspect_occupancy_cell_domain(
            reference,
            occupancy_repository,
            capture_repository,
            calibration_repository(),
        )

    def load_cell(
        reference,
        occupancy_repository,
        selection,
        *,
        expected_domain_identity,
    ):
        return load_exact_occupancy_cell_source(
            reference,
            occupancy_repository,
            capture_repository,
            calibration_repository(),
            selection,
            expected_domain_identity=expected_domain_identity,
        )

    return OccupancyApi(
        calibration,
        repository_path=operations.repository_root / "occupancy",
        require_binding=host.require_readout_binding,
        wait_run=operations.wait_run,
        admit_capture=capture_repository.admit,
        start_detection=start_detection,
        load_occupancy=load_occupancy,
        inspect_cell=inspect_cell,
        load_cell=load_cell,
    )


def _close_api(api: OccupancyApi) -> tuple[Exception, ...]:
    return api.close()


def _bind_task_console(api: OccupancyApi, _catalog: object, projection):
    from .ui.view_projection import project_occupancy_views
    from .workbench_adapter import resolve_occupancy_calibration_input

    return projection.processor(
        OCCUPANCY_LOGIC_NODE,
        prepare=api.prepare_occupancy_processor_request,
        resolve_artifact_reference=partial(
            resolve_occupancy_calibration_input,
            resolve_final_or_saved=projection.resolve_final_or_saved,
            load_saved_calibration=api.load_saved_calibration,
        ),
        project_presentations=project_occupancy_views,
    )


def _bind_artifact_capabilities(
    api: OccupancyApi,
) -> tuple[ArtifactCapability, ...]:
    return (
        ArtifactCapability(
            format_id=(
                "zlc_neutral_atom.logic_nodes.readout.occupancy."
                + OCCUPANCY_ARTIFACT_NAMESPACE
            ),
            source_label="occupancy",
            reference_type=OccupancyArtifactRef,
            project_figure=api._project_figure,
        ),
    )


LOGIC_NODE_PACKAGE = LogicNodePackage(
    api_name="occupancy",
    declaration=OCCUPANCY_LOGIC_NODE,
    bind_api=_bind_api,
    bind_task_console=_bind_task_console,
    task_console_order=50,
    api_dependencies=("calibration",),
    close_api=_close_api,
    bind_artifact_capabilities=_bind_artifact_capabilities,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
