"""Application composition for immutable, owner-projected Figure sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zlc_data import FitResultBatch
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.artifact_dispatch import ArtifactDispatch
from zlc_neutral_atom.artifacts import (
    FitResultArtifactRef,
    SavedFitResult,
    load_fit_result,
)

from ._dataset_sources import project_final_dataset_source

if TYPE_CHECKING:
    from zlc_frontend import DataFigure
    from ._application_services import ExperimentServices


def _special_figure_projection(
    artifacts: ArtifactDispatch,
    source: object,
    *,
    output: str | None,
    materialize: bool,
):
    projected = artifacts.project_figure(
        source,
        output=output,
        materialize=materialize,
    )
    if not isinstance(projected, tuple) or len(projected) != 2:
        raise TypeError(
            "artifact Figure projection must return (ArtifactDatasetSource, FigureIntent)"
        )
    dataset_source, figure_intent = projected
    if not isinstance(dataset_source, ArtifactDatasetSource):
        raise TypeError("artifact Figure projection lost its Dataset source")
    from zlc_frontend.plot_panel import FigureIntent

    if not isinstance(figure_intent, FigureIntent):
        raise TypeError("artifact Figure projection lost its FigureIntent")
    return dataset_source, figure_intent


def project_figure(
    services,
    artifacts: ArtifactDispatch,
    source: object,
    *,
    intent,
    point_ordinals,
    preferences,
    artifact_output: str | None,
    materialize: bool,
    draft_fit_result: FitResultBatch | None = None,
    preprojected_source: ArtifactDatasetSource | None = None,
):
    """Resolve application artifacts, then delegate all view policy to frontend."""

    from zlc_frontend.frozen_figure import (
        FrozenFigureSource,
        build_frozen_data_figure,
        build_frozen_figure_document,
        resolve_frozen_figure_intent,
    )

    if not isinstance(artifacts, ArtifactDispatch):
        raise TypeError("artifacts must be ArtifactDispatch")
    if artifact_output is not None and not isinstance(artifact_output, str):
        raise TypeError("artifact_output must be str or None")
    if draft_fit_result is not None and not isinstance(
        draft_fit_result,
        FitResultBatch,
    ):
        raise TypeError("draft_fit_result must be FitResultBatch or None")
    if preprojected_source is not None and not isinstance(
        preprojected_source,
        ArtifactDatasetSource,
    ):
        raise TypeError("preprojected_source must be ArtifactDatasetSource or None")
    if materialize and preprojected_source is not None:
        preprojected_source.require_owned_snapshot()

    fit_result = draft_fit_result
    dataset_source = preprojected_source
    owner_figure_intent = None
    source_ref = None
    source_label = "artifact"

    if draft_fit_result is not None:
        if not artifacts.can_project_dataset(source):
            raise TypeError("a draft Fit result requires its durable Dataset source")
        if artifact_output is not None:
            raise ValueError("a Fit source does not accept artifact_output")
        source_ref = source
        source_label = artifacts.source_label(source)
    elif isinstance(source, FitResultArtifactRef):
        if artifact_output is not None:
            raise ValueError("a saved Fit source does not accept artifact_output")
        saved_fit = load_fit_result(
            services.workspace_paths.output_root / "fits",
            source,
            artifacts=artifacts,
        )
        source_ref = saved_fit.source_artifact_ref
        fit_result = saved_fit.result
        source_label = artifacts.source_label(source_ref)
    elif isinstance(source, SavedFitResult):
        if artifact_output is not None:
            raise ValueError("a saved Fit source does not accept artifact_output")
        source_ref = source.source_artifact_ref
        fit_result = source.result
        source_label = artifacts.source_label(source_ref)
    elif artifacts.can_project_figure(source):
        if preprojected_source is not None:
            raise ValueError(
                "a special artifact Figure projection owns its Dataset source"
            )
        dataset_source, owner_figure_intent = _special_figure_projection(
            artifacts,
            source,
            output=artifact_output,
            materialize=materialize,
        )
        source_label = owner_figure_intent.value_label
    elif artifacts.can_project_dataset(source):
        if artifact_output is not None:
            raise ValueError(
                "artifact_output is valid only for a multi-output Figure artifact"
            )
        source_ref = source
        source_label = artifacts.source_label(source)
    else:
        raise TypeError(
            "figure source is not owned by a composed artifact capability or Fit"
        )

    if source_ref is not None and dataset_source is None:
        dataset_source = project_final_dataset_source(
            artifacts,
            source_ref,
            materialize=materialize,
        )
    if dataset_source is None:
        raise RuntimeError("Figure source Dataset identity is unavailable")
    snapshot = (
        dataset_source.require_owned_snapshot()
        if materialize
        else dataset_source.snapshot
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
    if (
        fit_result is None
        and owner_figure_intent is not None
        and intent is None
        and point_ordinals is None
        and preferences is None
    ):
        figure_intent = owner_figure_intent
    else:
        figure_intent = resolve_frozen_figure_intent(
            frontend_source,
            intent=intent,
            point_ordinals=point_ordinals,
            preferences=preferences,
            title=(
                None
                if owner_figure_intent is None
                else owner_figure_intent.title
            ),
            value_label=(
                None
                if owner_figure_intent is None
                else owner_figure_intent.value_label
            ),
        )
    if not materialize:
        document = build_frozen_figure_document(
            frontend_source,
            figure_intent,
        )
        return document, None, figure_intent, fit_result
    figure = build_frozen_data_figure(
        frontend_source,
        figure_intent,
    )
    return figure.document, figure, figure_intent, fit_result


def data_figure_for_services(
    services: "ExperimentServices",
    artifacts: ArtifactDispatch,
    source: object,
    *,
    intent,
    point_ordinals,
    preferences,
    artifact_output: str | None,
    draft_fit_result: FitResultBatch | None = None,
    preprojected_source: ArtifactDatasetSource | None = None,
) -> tuple["DataFigure", object]:
    """Build one frozen DataFigure while repository authority stays private."""

    _document, figure, figure_intent, _fit_result = project_figure(
        services,
        artifacts,
        source,
        intent=intent,
        point_ordinals=point_ordinals,
        preferences=preferences,
        artifact_output=artifact_output,
        materialize=True,
        draft_fit_result=draft_fit_result,
        preprojected_source=preprojected_source,
    )
    if figure is None:
        raise RuntimeError("frozen Figure source was not materialized")
    return figure, figure_intent


__all__ = ["data_figure_for_services", "project_figure"]
