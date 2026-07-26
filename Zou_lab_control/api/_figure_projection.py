"""Application composition for immutable, owner-projected Figure sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zlc_data import FitResultBatch
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.artifact_dispatch import ArtifactDispatch
from zlc_neutral_atom.artifacts import (
    AdmittedFitResult,
    FitExecution,
    FitResultArtifactRef,
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
    dataset_source = getattr(projected, "source", None)
    if not isinstance(dataset_source, ArtifactDatasetSource):
        raise TypeError("artifact Figure projection lost its Dataset source")
    label = str(getattr(projected, "label", "")).strip()
    if not label:
        raise ValueError("artifact Figure projection requires a visible label")
    resolver = getattr(projected, "resolve_preferences", None)
    if not callable(resolver):
        raise TypeError("artifact Figure projection must resolve view preferences")
    if not hasattr(projected, "default_intent"):
        raise TypeError("artifact Figure projection requires a default intent")
    return projected, dataset_source, label


def project_figure(
    services,
    artifacts: ArtifactDispatch,
    source: object,
    *,
    intent,
    selection,
    preferences,
    artifact_output: str | None,
    materialize: bool,
    draft_fit_result: FitResultBatch | None = None,
    preprojected_source: ArtifactDatasetSource | None = None,
):
    """Resolve application artifacts, then delegate all view policy to frontend."""

    from zlc_frontend import (
        FrozenFigureSource,
        build_frozen_data_figure,
        build_frozen_figure_document,
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
    owner_projection = None
    source_ref = None
    source_label = "artifact"

    if draft_fit_result is not None:
        if not artifacts.can_project_dataset(source):
            raise TypeError("a draft Fit result requires its durable Dataset source")
        if artifact_output is not None:
            raise ValueError("a Fit source does not accept artifact_output")
        source_ref = source
        source_label = artifacts.source_label(source)
    elif isinstance(source, FitExecution):
        if artifact_output is not None:
            raise ValueError("a Fit source does not accept artifact_output")
        source_ref = source.source_artifact_ref
        fit_result = source.result
        source_label = artifacts.source_label(source_ref)
    elif isinstance(source, FitResultArtifactRef):
        if artifact_output is not None:
            raise ValueError("a saved Fit source does not accept artifact_output")
        admitted_fit = services.fit_repository.load(
            source,
            artifacts=artifacts,
        )
        source_ref = admitted_fit.source_artifact_ref
        fit_result = admitted_fit.result
        source_label = artifacts.source_label(source_ref)
    elif isinstance(source, AdmittedFitResult):
        if artifact_output is not None:
            raise ValueError("an admitted Fit source does not accept artifact_output")
        source_ref = source.source_artifact_ref
        fit_result = source.result
        source_label = artifacts.source_label(source_ref)
    elif artifacts.can_project_figure(source):
        if preprojected_source is not None:
            raise ValueError(
                "a special artifact Figure projection owns its Dataset source"
            )
        owner_projection, dataset_source, source_label = _special_figure_projection(
            artifacts,
            source,
            output=artifact_output,
            materialize=materialize,
        )
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

    resolved_intent = intent
    resolved_preferences = preferences
    if fit_result is None and owner_projection is not None:
        if resolved_intent is None:
            resolved_intent = owner_projection.default_intent
        resolved_preferences = owner_projection.resolve_preferences(
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
    artifacts: ArtifactDispatch,
    source: object,
    *,
    intent,
    selection,
    preferences,
    artifact_output: str | None,
    draft_fit_result: FitResultBatch | None = None,
    preprojected_source: ArtifactDatasetSource | None = None,
) -> "DataFigure":
    """Build one frozen DataFigure while repository authority stays private."""

    _document, figure, _fit_result = project_figure(
        services,
        artifacts,
        source,
        intent=intent,
        selection=selection,
        preferences=preferences,
        artifact_output=artifact_output,
        materialize=True,
        draft_fit_result=draft_fit_result,
        preprojected_source=preprojected_source,
    )
    if figure is None:
        raise RuntimeError("frozen Figure source was not materialized")
    return figure


__all__ = ["data_figure_for_services", "project_figure"]
