"""Build one immutable frontend figure from an already-resolved Dataset.

The application composition layer resolves artifact repositories and hands this
module a typed Dataset identity (and, when pixels are required, its exact
``OwnedSnapshot``).  All view suggestion, FigureDocument construction and Fit
overlay binding remain here with the frontend owner; no repository or neutral
artifact type crosses this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from zlc_data import (
    MONITOR_HISTORY,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    DatasetRevisionRef,
    DatasetSchema,
    FitResultBatch,
    OwnedSnapshot,
)
from zlc_data.fit import validate_fit_result_source_binding

from .data_figure import DataFigure
from .figure import (
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    suggest_fit_view,
    suggest_view,
)
from .plot_panel import FigureIntent, figure_intent_from_view


@dataclass(frozen=True, slots=True)
class FrozenFigureSource:
    """One application-resolved Dataset revision for frontend presentation."""

    label: str
    schema: DatasetSchema
    ref: DatasetRevisionRef
    snapshot: OwnedSnapshot | None = None
    fit_result: FitResultBatch | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("figure source label must be non-empty")
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if not isinstance(self.ref, DatasetRevisionRef):
            raise TypeError("ref must be DatasetRevisionRef")
        if self.ref.schema_fingerprint != self.schema.fingerprint:
            raise ValueError("figure source ref differs from its schema")
        if self.snapshot is not None:
            if not isinstance(self.snapshot, OwnedSnapshot):
                raise TypeError("snapshot must be OwnedSnapshot or None")
            if self.snapshot.ref != self.ref:
                raise ValueError("figure source snapshot differs from its ref")
            if self.snapshot.block.schema.fingerprint != self.schema.fingerprint:
                raise ValueError("figure source snapshot differs from its schema")
        if self.fit_result is not None:
            if not isinstance(self.fit_result, FitResultBatch):
                raise TypeError("fit_result must be FitResultBatch or None")
            validate_fit_result_source_binding(
                self.fit_result,
                self.ref,
                self.schema,
            )


def _default_view_intent(schema: DatasetSchema) -> ViewIntent:
    """Choose one ordinary Figure family from declared roles only."""

    point_roles = tuple(column.role for column in schema.point_table.columns)
    data_roles = tuple(axis.role for axis in schema.cell_schema.data_axes)
    roles = (*point_roles, *data_roles)
    if roles.count(SPATIAL_X) == 1 and roles.count(SPATIAL_Y) == 1:
        return ViewIntent.IMAGE
    if any(
        role in {SCAN_POINT, SPECTRAL, MONITOR_HISTORY}
        for role in roles
    ):
        return ViewIntent.CURVE
    return ViewIntent.HISTOGRAM


def resolve_frozen_figure_intent(
    source: FrozenFigureSource,
    *,
    intent: ViewIntent | None = None,
    point_ordinals: tuple[int, ...] | None = None,
    preferences: ViewPreferences | None = None,
    title: str | None = None,
    value_label: str | None = None,
) -> FigureIntent:
    """Resolve public authoring choices once into the canonical Figure intent.

    Automatic choices are frontend policy derived from the declared schema.
    Ambiguous axes remain explicit; no renderer is allowed to repeat this
    decision from rank, singleton lengths, or data values.
    """

    if not isinstance(source, FrozenFigureSource):
        raise TypeError("source must be FrozenFigureSource")
    if point_ordinals is not None:
        point_ordinals = tuple(point_ordinals)
    if intent is not None and not isinstance(intent, ViewIntent):
        raise TypeError("intent must be ViewIntent or None")
    if preferences is not None and not isinstance(preferences, ViewPreferences):
        raise TypeError("preferences must be ViewPreferences or None")

    if source.fit_result is None:
        resolved_intent = _default_view_intent(source.schema) if intent is None else intent
        suggestion = suggest_view(
            source.schema,
            resolved_intent,
            point_ordinals,
            preferences,
        )
    else:
        if point_ordinals is not None:
            raise ValueError(
                "saved Fit display derives its exact point rows from FitSpec"
            )
        suggestion = suggest_fit_view(
            source.schema,
            source.fit_result,
            preferences,
        )
        if (
            suggestion.spec is not None
            and intent not in (None, suggestion.spec.intent)
        ):
            raise ValueError(
                "requested figure intent is incompatible with the fitted axes"
            )
    if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
        details = "; ".join(reason.message for reason in suggestion.reasons)
        raise ValueError(f"figure view needs explicit input: {details}")

    return figure_intent_from_view(
        suggestion.spec,
        title=source.label if title is None else title,
        value_label=source.label if value_label is None else value_label,
    )


def build_frozen_figure_document(
    source: FrozenFigureSource,
    figure: FigureIntent,
    *,
    document_id: str | None = None,
) -> FigureDocument:
    """Create one FigureDocument from an already-resolved Figure intent."""

    if not isinstance(source, FrozenFigureSource):
        raise TypeError("source must be FrozenFigureSource")
    if not isinstance(figure, FigureIntent) or figure.view is None:
        raise TypeError("frozen Dataset Figure requires a resolved FigureIntent")
    if figure.view.schema_fingerprint != source.schema.fingerprint:
        raise ValueError("FigureIntent view belongs to another Dataset schema")

    dataset_id = DatasetId("source")
    resolved_document_id = (
        f"figure-{uuid4().hex}" if document_id is None else document_id
    )
    return FigureDocument(
        document_id=resolved_document_id,
        revision=0,
        datasets=(
            DatasetDescriptor(
                dataset_id,
                source.label,
                source.schema.fingerprint,
            ),
        ),
        layers=(FigureLayer("data", dataset_id, figure.view),),
    )


def build_frozen_data_figure(
    source: FrozenFigureSource,
    figure: FigureIntent,
    *,
    document_id: str | None = None,
) -> DataFigure:
    """Evaluate one exact frozen source without resolving an artifact owner."""

    if not isinstance(source, FrozenFigureSource):
        raise TypeError("source must be FrozenFigureSource")
    if source.snapshot is None:
        raise ValueError("a DataFigure requires the exact source snapshot")
    document = build_frozen_figure_document(
        source,
        figure,
        document_id=document_id,
    )
    dataset_id = document.datasets[0].dataset_id
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, source.snapshot),)),
        fit_results=(
            {"data": source.fit_result}
            if source.fit_result is not None
            else None
        ),
    )


__all__ = [
    "FrozenFigureSource",
    "build_frozen_data_figure",
    "build_frozen_figure_document",
    "resolve_frozen_figure_intent",
]
