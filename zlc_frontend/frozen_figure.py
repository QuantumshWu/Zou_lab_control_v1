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
    DatasetRevisionRef,
    DatasetSchema,
    FitResultBatch,
    OwnedSnapshot,
    Selection,
    validate_fit_result_source_binding,
)

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
from .panel_policy import automatic_figure_intent


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


def build_frozen_figure_document(
    source: FrozenFigureSource,
    *,
    intent: ViewIntent | None = None,
    selection: Selection | None = None,
    preferences: ViewPreferences | None = None,
    document_id: str | None = None,
) -> FigureDocument:
    """Create the sole one-Dataset FigureDocument suggestion path.

    Automatic choices are frontend policy derived from the declared schema.
    Ambiguous axes remain an explicit error; this function never guesses from
    rank or silently reduces an informative data axis.
    """

    if not isinstance(source, FrozenFigureSource):
        raise TypeError("source must be FrozenFigureSource")
    if selection is not None and not isinstance(selection, Selection):
        raise TypeError("selection must be Selection or None")
    if intent is not None and not isinstance(intent, ViewIntent):
        raise TypeError("intent must be ViewIntent or None")
    if preferences is not None and not isinstance(preferences, ViewPreferences):
        raise TypeError("preferences must be ViewPreferences or None")

    if source.fit_result is None:
        resolved_intent = (
            automatic_figure_intent(source.schema)
            if intent is None
            else intent
        )
        suggestion = suggest_view(
            source.schema,
            resolved_intent,
            selection,
            preferences,
        )
    else:
        suggestion = suggest_fit_view(
            source.schema,
            source.fit_result,
            selection,
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

    dataset_id = DatasetId("source")
    resolved_document_id = (
        f"notebook-{uuid4().hex}" if document_id is None else document_id
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
        layers=(FigureLayer("data", dataset_id, suggestion.spec),),
    )


def build_frozen_data_figure(
    source: FrozenFigureSource,
    *,
    intent: ViewIntent | None = None,
    selection: Selection | None = None,
    preferences: ViewPreferences | None = None,
    document_id: str | None = None,
) -> DataFigure:
    """Evaluate one exact frozen source without resolving an artifact owner."""

    if not isinstance(source, FrozenFigureSource):
        raise TypeError("source must be FrozenFigureSource")
    if source.snapshot is None:
        raise ValueError("a DataFigure requires the exact source snapshot")
    document = build_frozen_figure_document(
        source,
        intent=intent,
        selection=selection,
        preferences=preferences,
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
]
