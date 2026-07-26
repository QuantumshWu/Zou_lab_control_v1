"""Application composition for immutable Figure sources.

Generic frontend construction is shared; Occupancy's artifact projection is
an explicitly imported capability adapter here, never a policy hidden in the
public Experiment facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zlc_data import FitResultBatch
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.artifacts import (
    AdmittedFitResult,
    FitExecution,
    FitResultArtifactRef,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.pulse_scan.reference import ScanArtifactRef
from zlc_neutral_atom.logic_nodes.readout.occupancy.reference import (
    OccupancyArtifactRef,
)

from ._dataset_sources import project_final_dataset_source

if TYPE_CHECKING:
    from zlc_frontend import DataFigure
    from ._application_services import ExperimentServices

def project_notebook_figure(
    services,
    source,
    *,
    intent,
    selection,
    preferences,
    occupancy_output,
    materialize: bool,
    draft_fit_result: FitResultBatch | None = None,
    preprojected_source: ArtifactDatasetSource | None = None,
):
    """Resolve application artifacts, then delegate all Figure policy."""

    from zlc_frontend import (
        FrozenFigureSource,
        build_frozen_data_figure,
        build_frozen_figure_document,
    )

    is_occupancy = isinstance(source, OccupancyArtifactRef)
    if not is_occupancy and occupancy_output is not None:
        raise ValueError("occupancy_output is valid only for OccupancyArtifactRef")

    if draft_fit_result is not None and not isinstance(
        draft_fit_result,
        FitResultBatch,
    ):
        raise TypeError("draft_fit_result must be FitResultBatch or None")
    if preprojected_source is not None and not isinstance(
        preprojected_source,
        ArtifactDatasetSource,
    ):
        raise TypeError(
            "preprojected_source must be ArtifactDatasetSource or None"
        )
    if materialize and preprojected_source is not None:
        preprojected_source.require_owned_snapshot()

    fit_result = draft_fit_result
    dataset_source = preprojected_source
    occupancy_projection = None
    source_label = "capture"
    if draft_fit_result is not None:
        if not isinstance(source, (CaptureArtifactRef, ScanArtifactRef)):
            raise TypeError(
                "a draft fit result requires its capture or scan source"
            )
        source_ref = source
    elif isinstance(source, ScanArtifactRef):
        source_label = "scan"
        source_ref = source
    elif is_occupancy:
        from zlc_neutral_atom.logic_nodes.readout.occupancy.ui.view_projection import (
            project_occupancy_figure,
        )

        resolved_occupancy = services.readout_resources.occupancy_repository().admit(
            source,
            services.capture_repository,
            services.readout_resources.calibration_repository(),
        )
        occupancy_projection = project_occupancy_figure(
            resolved_occupancy,
            output=occupancy_output,
            materialize=materialize,
        )
        if preprojected_source is not None:
            raise ValueError(
                "Occupancy projection is owned by its capability adapter"
            )
        dataset_source = occupancy_projection.source
        source_label = occupancy_projection.label
        source_ref = None
    elif isinstance(source, CaptureArtifactRef):
        source_ref = source
    elif isinstance(source, FitExecution):
        source_ref = source.source_artifact_ref
        fit_result = source.result
    elif isinstance(source, FitResultArtifactRef):
        admitted_fit = services.fit_repository.load(
            source,
            capture_repository=services.capture_repository,
            scan_repository=services.readout_resources.scan_repository,
        )
        source_ref = admitted_fit.source_artifact_ref
        fit_result = admitted_fit.result
    elif isinstance(source, AdmittedFitResult):
        source_ref = source.source_artifact_ref
        fit_result = source.result
    else:
        raise TypeError(
            "figure source must be ScanArtifactRef, OccupancyArtifactRef, "
            "CaptureArtifactRef, FitExecution, FitResultArtifactRef, "
            "or AdmittedFitResult"
        )

    if source_ref is not None and dataset_source is None:
        dataset_source = project_final_dataset_source(
            services,
            source_ref,
            materialize=materialize,
        )
    if dataset_source is None:
        raise RuntimeError("figure source Dataset identity is unavailable")
    snapshot = (
        dataset_source.require_owned_snapshot()
        if materialize
        else dataset_source.snapshot
    )

    resolved_intent = intent
    resolved_preferences = preferences
    if fit_result is None and occupancy_projection is not None:
        if resolved_intent is None:
            resolved_intent = occupancy_projection.default_intent
        resolved_preferences = occupancy_projection.resolve_preferences(
            resolved_intent,
            resolved_preferences,
        )
    frontend_source = FrozenFigureSource(
        label=(
            source_label
            if fit_result is None
            else f"fit: {fit_result.spec.model_id}"
        ),
        schema=dataset_source.schema,
        ref=dataset_source.ref,
        snapshot=snapshot,
        fit_result=fit_result,
    )
    if not materialize:
        document = build_frozen_figure_document(
            frontend_source,
            intent=resolved_intent,
            selection=selection,
            preferences=resolved_preferences,
        )
        return document, None, fit_result
    figure = build_frozen_data_figure(
        frontend_source,
        intent=resolved_intent,
        selection=selection,
        preferences=resolved_preferences,
    )
    return figure.document, figure, fit_result


def data_figure_for_services(
    services: "ExperimentServices",
    source,
    *,
    intent,
    selection,
    preferences,
    occupancy_output,
    draft_fit_result: FitResultBatch | None = None,
    preprojected_source: ArtifactDatasetSource | None = None,
) -> "DataFigure":
    """Build one frozen DataFigure while repository authority stays private."""

    _document, figure, _fit_result = project_notebook_figure(
        services,
        source,
        intent=intent,
        selection=selection,
        preferences=preferences,
        occupancy_output=occupancy_output,
        materialize=True,
        draft_fit_result=draft_fit_result,
        preprojected_source=preprojected_source,
    )
    if figure is None:
        raise RuntimeError("frozen Figure source was not materialized")
    return figure



__all__ = ["data_figure_for_services", "project_notebook_figure"]
