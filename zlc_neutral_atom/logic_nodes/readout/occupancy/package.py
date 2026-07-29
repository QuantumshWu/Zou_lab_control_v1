"""Occupancy's complete built-in capability package."""

from __future__ import annotations

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.logic_node_package import (
    LogicNodePackage,
    UiContributionDescriptor,
)

from .api import OccupancyApi
from .application import (
    prepare_detection_plan,
    resolve_occupancy_calibration_input,
)
from .cell import (
    inspect_occupancy_cell_domain,
    load_exact_occupancy_cell_source,
)
from .declaration import OCCUPANCY_LOGIC_NODE
from .reference import OCCUPANCY_ARTIFACT_NAMESPACE, OccupancyArtifactRef


def _bind_api(
    facts: tuple[object, ...],
    dependencies: tuple[object, ...],
) -> OccupancyApi:
    (calibration,) = dependencies
    (
        repository_root,
        capture_repository,
        start_run,
        wait_run,
        open_ui,
    ) = facts

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
        address,
        *,
        expected_domain_identity,
    ):
        return load_exact_occupancy_cell_source(
            reference,
            occupancy_repository,
            capture_repository,
            calibration_repository(),
            address,
            expected_domain_identity=expected_domain_identity,
        )

    return OccupancyApi(
        calibration,
        repository_path=repository_root / "occupancy",
        wait_run=wait_run,
        admit_capture=capture_repository.admit,
        start_detection=start_detection,
        load_occupancy=load_occupancy,
        inspect_cell=inspect_cell,
        load_cell=load_cell,
        open_ui=open_ui,
    )


def _prepare_hosted(api, request, event_source):
    if event_source is not None:
        raise ValueError(
            "Occupancy event source is bound by the generic Processor host"
        )
    return api.prepare_occupancy_processor_request(request)


def _resolve_artifact_reference(api, binding, resolve_final_or_saved):
    return resolve_occupancy_calibration_input(
        binding,
        resolve_final_or_saved=resolve_final_or_saved,
        load_saved_calibration=api.load_saved_calibration,
    )


def _close_api(api: OccupancyApi) -> tuple[Exception, ...]:
    return api.close()


def _project_signal_presentation(node, output_name, publication):
    from .ui.view_projection import project_occupancy_signal_presentation

    return project_occupancy_signal_presentation(node, output_name, publication)


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
    api_requirements=(
        "repository_root",
        "capture_repository",
        "start_run",
        "wait_run",
        "open_ui",
    ),
    bind_api=_bind_api,
    prepare_hosted=_prepare_hosted,
    api_dependencies=("calibration",),
    resolve_artifact_reference=_resolve_artifact_reference,
    project_signal_presentation=_project_signal_presentation,
    ui_contributions=(
        UiContributionDescriptor(
            "cell",
            "zlc_neutral_atom.logic_nodes.readout.occupancy.ui.workbench_window",
            "OccupancyCellWindow",
        ),
    ),
    close_api=_close_api,
    bind_artifact_capabilities=_bind_artifact_capabilities,
)

__all__ = ["LOGIC_NODE_PACKAGE"]
