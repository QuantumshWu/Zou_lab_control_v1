"""Headless materialisation of Figure-owned selector and Fit signals.

The Figure owns Area and Cross intent; a Measurement continues to publish only
its physical dataset.  This module turns an accepted Figure gesture into typed
datasets without depending on any Workbench shell.  It deliberately contains
no Qt, renderer, runtime node, buffering, or producer-control code.

Area data is evaluated by :mod:`zlc_data.transform`, so a selection keeps every
axis it did not explicitly name and carries component validity with it.  Cross
publishes the data value selected on the displayed IMAGE, CURVE, or HISTOGRAM;
its coordinates remain derivation metadata and never masquerade as separate
signals.  Fit publishes one typed ``fit.<parameter>`` dataset per parameter
while preserving its named batch layout and failed-cell validity.  Derived
values retain the source join digest while receiving stable Figure/output
dataset identities; a composition root may attach its own run/epoch routing
metadata afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
from zlc_storage import canonical_digest, canonical_text, sha256_text

from zlc_data import (
    AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
    AxisId,
    BlockId,
    DataTransformSpec,
    DatasetRevisionRef,
    IndexRangeSelection,
    IndexSelection,
    MissingPolicy,
    OwnedSnapshot,
    ReductionMethod,
    ReductionSpec,
    Selection,
    StreamGenerationId,
    ValidityPolicy,
    apply_transform,
    commit_transform,
    dataset_revision_ref_to_tree,
    expand_dataset_validity,
    selection_to_tree,
    fit_spec_to_tree,
    materialize_dataset_acceptance_mask,
    materialize_dataset_selection,
    materialize_fit_parameter_snapshots,
    materialize_scalar_dataset,
    projected_dataset_output_contract_id,
)
from .curve_display import numeric_curve_coordinates
from .data_figure import DataFigure
from .figure import (
    AxisViewBinding,
    AxisViewRole,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    FigureDocument,
    FigureLayer,
    FixedIndex,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    figure_document_to_tree,
    view_spec_to_tree,
)
from .site_map import SiteMapPresentation
from .figure_source import FigureSource
from .render import (
    CurvePanelPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    SiteMapPanelPayload,
    SourceIdentity,
)
AREA_DATA_OUTPUT = "area.data"
CROSS_DATA_OUTPUT = "cross.data"
FIT_OUTPUT_PREFIX = "fit."
FIGURE_CROSS_DATA_OUTPUT_CONTRACT_ID = "zlc_frontend.figure.cross-data"
FIGURE_FIT_PARAMETER_OUTPUT_CONTRACT_ID = "zlc_frontend.figure.fit-parameter"


def figure_output_contract_id(
    output_name: str,
    *,
    source_contract_id: str | None = None,
) -> str:
    """Return the frontend owner's exact contract for one Figure output."""

    if output_name == AREA_DATA_OUTPUT:
        if not isinstance(source_contract_id, str) or not source_contract_id:
            raise ValueError("Figure Area output requires its source contract id")
        return projected_dataset_output_contract_id(
            source_contract_id,
            AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
        )
    if output_name == CROSS_DATA_OUTPUT:
        return FIGURE_CROSS_DATA_OUTPUT_CONTRACT_ID
    if output_name.startswith(FIT_OUTPUT_PREFIX) and output_name != FIT_OUTPUT_PREFIX:
        return FIGURE_FIT_PARAMETER_OUTPUT_CONTRACT_ID
    raise ValueError(f"unknown Figure output {output_name!r}")

__all__ = [
    "AREA_DATA_OUTPUT",
    "CROSS_DATA_OUTPUT",
    "FIT_OUTPUT_PREFIX",
    "FIGURE_CROSS_DATA_OUTPUT_CONTRACT_ID",
    "FIGURE_FIT_PARAMETER_OUTPUT_CONTRACT_ID",
    "FigureDerivedSignal",
    "FigureAreaCommit",
    "FigureCrossCommit",
    "FigureOutputFront",
    "FigureOutputPresentation",
    "FigureOutputRequest",
    "FigureOutputSession",
    "HistogramValueRangeSelection",
    "area_data_output_presentation",
    "bind_area_data_commit",
    "bind_cross_data_commit",
    "figure_derived_signal",
    "figure_derivation_digest",
    "figure_output_contract_id",
    "figure_output_revision_ref",
    "materialize_area_outputs",
    "materialize_area_snapshot",
    "materialize_cross_outputs",
    "materialize_fit_outputs",
    "source_identity_matches_snapshot",
]


@dataclass(frozen=True, slots=True)
class FigureOutputPresentation:
    """Complete frontend-owned public facts for one Figure-derived signal.

    A shell may prepend its own producer namespace, but it must not infer the
    bare name, semantic contract, labels, or description from name prefixes or
    private selector/Fit metadata.
    """

    name: str
    contract_id: str
    short: str
    axis_label: str
    description: str

    def __post_init__(self) -> None:
        for field, label in (
            ("name", "Figure output name"),
            ("contract_id", "Figure output contract id"),
            ("short", "Figure output short label"),
            ("axis_label", "Figure output axis label"),
            ("description", "Figure output description"),
        ):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), label),
            )


@dataclass(frozen=True, slots=True)
class FigureAreaCommit:
    """One completed Figure Area intent bound to a producer generation."""

    source_identity: SourceIdentity
    selection: Selection | "HistogramValueRangeSelection"

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("Area source_identity must be SourceIdentity")
        if not isinstance(
            self.selection,
            (Selection, HistogramValueRangeSelection),
        ):
            raise TypeError("Area selection is not a Figure selection")


@dataclass(frozen=True, slots=True)
class FigureCrossCommit:
    """One locked Cross value selection bound to a producer generation.

    ``sample_document`` is a narrowed copy of the exact displayed Figure: its
    focused facet, chosen batch series, and selected image/curve sample are
    explicit in the frontend-owned ViewSpec.  Re-evaluating that document on a
    newer snapshot therefore computes the same semantic value without
    rasterising a Figure or asking a Workbench to understand axis roles.

    Histogram bins are renderer display state rather than Dataset axes, so the
    selected immutable bin interval is carried separately.  ``point`` is only
    provenance for the gesture; it is never published as a signal.
    """

    source_identity: SourceIdentity
    sample_document: FigureDocument
    point: tuple[float, float]
    histogram_bin: tuple[float, float, bool] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("Cross source_identity must be SourceIdentity")
        if not isinstance(self.sample_document, FigureDocument):
            raise TypeError("Cross sample_document must be FigureDocument")
        if (
            len(self.sample_document.datasets) != 1
            or len(self.sample_document.layers) != 1
        ):
            raise ValueError("Cross sample document must contain one dataset and layer")
        intent = self.sample_document.layers[0].view.intent
        if intent not in {
            ViewIntent.IMAGE,
            ViewIntent.CURVE,
            ViewIntent.HISTOGRAM,
        }:
            raise ValueError("Cross data requires IMAGE, CURVE, or HISTOGRAM")
        unresolved = tuple(
            binding.role
            for binding in self.sample_document.layers[0].view.axis_bindings
            if binding.role in {AxisViewRole.BATCH, AxisViewRole.FACET}
        )
        if unresolved:
            raise ValueError("Cross sample document retained unresolved batch/facet axes")
        point = tuple(float(value) for value in self.point)
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise ValueError("Cross point must contain two finite coordinates")
        histogram_bin = self.histogram_bin
        if intent is ViewIntent.HISTOGRAM:
            if histogram_bin is None or len(tuple(histogram_bin)) != 3:
                raise ValueError("Histogram Cross requires one selected bin")
            lower, upper, include_upper = histogram_bin
            lower, upper = float(lower), float(upper)
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower >= upper
                or type(include_upper) is not bool
            ):
                raise ValueError("Histogram Cross bin is invalid")
            histogram_bin = (lower, upper, include_upper)
        elif histogram_bin is not None:
            raise ValueError("Only Histogram Cross may carry a bin interval")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "histogram_bin", histogram_bin)


@dataclass(frozen=True, slots=True)
class FigureOutputRequest:
    """Immutable Figure-output intent evaluated outside any GUI owner.

    The source has already been selected by the composition root.  Area and
    Cross commits are generation-scoped, while a Fit result is exact-revision
    scoped.  This request neither knows panel routing names nor publishes into
    an application data plane.
    """

    source: FigureSource
    area: FigureAreaCommit | None = None
    cross: FigureCrossCommit | None = None
    fit_result: object | None = None
    fit_result_identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, FigureSource):
            raise TypeError("Figure output request source must be FigureSource")
        snapshot = self.source.snapshot
        for label, commit, expected_type in (
            ("Area", self.area, FigureAreaCommit),
            ("Cross", self.cross, FigureCrossCommit),
        ):
            if commit is None:
                continue
            if not isinstance(commit, expected_type):
                raise TypeError(f"{label} request has another commit type")
            if not source_identity_matches_snapshot(commit.source_identity, snapshot):
                raise ValueError(f"{label} commit belongs to another source generation")
        if self.fit_result is None:
            if self.fit_result_identity is not None:
                raise ValueError("Fit identity requires a Fit result")
        else:
            from zlc_data import FitResultBatch

            if not isinstance(self.fit_result, FitResultBatch):
                raise TypeError("Figure output Fit result must be FitResultBatch")
            if self.fit_result.source_ref != snapshot.ref:
                raise ValueError("Figure output Fit belongs to another source revision")
            identity = str(self.fit_result_identity or "").strip()
            if not identity:
                raise ValueError("Figure output Fit requires an exact result identity")
            object.__setattr__(self, "fit_result_identity", identity)


@dataclass(frozen=True, slots=True)
class FigureOutputFront:
    """One complete immutable answer from :class:`FigureOutputSession`."""

    outputs: Mapping[str, "FigureDerivedSignal"]
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = dict(self.outputs)
        if any(not isinstance(value, FigureDerivedSignal) for value in values.values()):
            raise TypeError("Figure output front contains another value type")
        mismatched = tuple(
            key
            for key, value in values.items()
            if key != value.presentation.name
        )
        if mismatched:
            raise ValueError(
                "Figure output mapping keys differ from frontend presentation names: "
                f"{mismatched}"
            )
        failures = tuple(str(value) for value in self.failures)
        object.__setattr__(self, "outputs", MappingProxyType(values))
        object.__setattr__(self, "failures", failures)


def _area_dependency(request: FigureOutputRequest) -> tuple[object, ...] | None:
    commit = request.area
    if commit is None:
        return None
    selection = commit.selection
    if isinstance(selection, Selection):
        selection_identity: object = selection_to_tree(selection)
    else:
        selection_identity = {
            "histogram_value_range": [selection.lower, selection.upper],
            "source_selection": (
                None
                if selection.source_selection is None
                else selection_to_tree(selection.source_selection)
            ),
        }
    site_map = request.source.site_map
    return (
        request.source.snapshot.ref,
        request.source.source_contract_id,
        None if site_map is None else site_map.view_identity,
        canonical_digest(selection_identity),
    )


def _cross_dependency(request: FigureOutputRequest) -> tuple[object, ...] | None:
    commit = request.cross
    if commit is None:
        return None
    return (
        request.source.snapshot.ref,
        canonical_digest(
            {
                "sample_document": figure_document_to_tree(commit.sample_document),
                "point": commit.point,
                "histogram_bin": commit.histogram_bin,
            }
        ),
    )


class FigureOutputSession:
    """Persistent headless materializer for one Figure's derived signals.

    A live Area depends on each exact source revision and is therefore rebuilt
    when the source advances.  Cross and Fit work is cached by its own exact
    dependency rather than being repeated merely because a raster was painted.
    The session keeps no queue, policy, routing, or Qt state.
    """

    def __init__(self) -> None:
        self._area_key: tuple[object, ...] | None = None
        self._area_outputs: Mapping[str, FigureDerivedSignal] = MappingProxyType({})
        self._area_failures: tuple[str, ...] = ()
        self._cross_key: tuple[object, ...] | None = None
        self._cross_outputs: Mapping[str, FigureDerivedSignal] = MappingProxyType({})
        self._cross_failures: tuple[str, ...] = ()
        self._fit_key: tuple[object, ...] | None = None
        self._fit_outputs: Mapping[str, FigureDerivedSignal] = MappingProxyType({})
        self._fit_failures: tuple[str, ...] = ()

    @staticmethod
    def _materialize(
        label: str,
        operation,
    ) -> tuple[Mapping[str, "FigureDerivedSignal"], tuple[str, ...]]:
        try:
            outputs = operation()
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            return MappingProxyType({}), (f"{label}: {error}",)
        return MappingProxyType(dict(outputs)), ()

    def evaluate(self, request: FigureOutputRequest) -> FigureOutputFront:
        if not isinstance(request, FigureOutputRequest):
            raise TypeError("Figure output session requires FigureOutputRequest")

        area_key = _area_dependency(request)
        if area_key != self._area_key:
            self._area_key = area_key
            if request.area is None:
                self._area_outputs, self._area_failures = MappingProxyType({}), ()
            else:
                self._area_outputs, self._area_failures = self._materialize(
                    "Area",
                    lambda: materialize_area_outputs(
                        request.source,
                        request.area.selection,
                    ),
                )

        cross_key = _cross_dependency(request)
        if cross_key != self._cross_key:
            self._cross_key = cross_key
            if request.cross is None:
                self._cross_outputs, self._cross_failures = MappingProxyType({}), ()
            else:
                self._cross_outputs, self._cross_failures = self._materialize(
                    "Cross",
                    lambda: materialize_cross_outputs(
                        request.source,
                        request.cross,
                    ),
                )

        fit_key = (
            None
            if request.fit_result is None
            else (request.source.snapshot.ref, request.fit_result_identity)
        )
        if fit_key != self._fit_key:
            self._fit_key = fit_key
            if request.fit_result is None:
                self._fit_outputs, self._fit_failures = MappingProxyType({}), ()
            else:
                self._fit_outputs, self._fit_failures = self._materialize(
                    "Fit",
                    lambda: materialize_fit_outputs(
                        request.source,
                        request.fit_result,
                    ),
                )

        outputs: dict[str, FigureDerivedSignal] = {}
        for group in (self._area_outputs, self._cross_outputs, self._fit_outputs):
            overlap = outputs.keys() & group.keys()
            if overlap:
                raise RuntimeError(
                    f"Figure output owners overlap: {tuple(sorted(overlap))}"
                )
            outputs.update(group)
        return FigureOutputFront(
            outputs,
            self._area_failures + self._cross_failures + self._fit_failures,
        )

    def close(self) -> None:
        self._area_key = None
        self._area_outputs = MappingProxyType({})
        self._area_failures = ()
        self._cross_key = None
        self._cross_outputs = MappingProxyType({})
        self._cross_failures = ()
        self._fit_key = None
        self._fit_outputs = MappingProxyType({})
        self._fit_failures = ()


@dataclass(frozen=True, slots=True)
class FigureDerivedSignal:
    """One headless Figure-derived dataset plus derivation identity.

    Run identity and routing names intentionally do not appear here.  Those are
    application-shell concerns; the Figure owns only the immutable snapshot,
    exact derivation, complete frontend-owned presentation, and whether source
    coverage remains meaningful for the derived value.
    """

    snapshot: OwnedSnapshot
    source_ref: DatasetRevisionRef
    derivation_digest: str
    presentation: FigureOutputPresentation
    preserve_source_coverage: bool = False
    source_transform: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("Figure signal snapshot must be OwnedSnapshot")
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("Figure signal source_ref must be DatasetRevisionRef")
        if self.snapshot.ref.revision != self.source_ref.revision:
            raise ValueError("Figure signal revision differs from its source")
        sha256_text(self.derivation_digest, "Figure signal derivation_digest")
        if not isinstance(self.presentation, FigureOutputPresentation):
            raise TypeError(
                "Figure signal presentation must be FigureOutputPresentation"
            )
        if type(self.preserve_source_coverage) is not bool:
            raise TypeError("preserve_source_coverage must be bool")
        if self.source_transform is not None:
            if not isinstance(self.source_transform, DataTransformSpec):
                raise TypeError("Figure signal source_transform must be DataTransformSpec")
            if not self.source_transform.operations:
                raise ValueError("Figure signal source_transform must not be empty")


@dataclass(frozen=True, slots=True)
class HistogramValueRangeSelection:
    """A Figure Area over histogram values, not over a named source axis.

    Histogram x coordinates are physical sample values.  Pretending they are
    one of the dataset's sample axes would select the wrong dimension.  This
    narrow Figure-output intent therefore remains separate from
    :class:`zlc_data.Selection` and is consumed only by the Area materializer.
    """

    lower: float
    upper: float
    source_selection: Selection | None = None

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("histogram Area bounds must be finite")
        if lower > upper:
            raise ValueError("histogram Area lower bound exceeds upper bound")
        if self.source_selection is not None and not isinstance(
            self.source_selection,
            Selection,
        ):
            raise TypeError("histogram Area source_selection must be Selection or None")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def _area_context_selection(
    figure: DataFigure,
    *,
    gesture_axis_ids: frozenset[AxisId],
) -> Selection | None:
    """Freeze the named data context visible behind one Area gesture.

    The gesture itself owns the axes it draws over.  Every other explicit
    display restriction and evaluated SELECTED/SLIDER/focused-FACET resolution
    is part of the data the operator actually selected.  REDUCED and BATCH
    axes remain untouched: a display mean must never become authority, while
    every simultaneously visible batch series remains selected as a group.
    """

    if not isinstance(figure, DataFigure):
        raise TypeError("Area binding requires one DataFigure")
    entries = tuple(figure.datasets.entries)
    layers = tuple(figure.evaluated.layers)
    inputs = tuple(figure.evaluated.inputs)
    if (
        len(entries) != 1
        or len(figure.document.layers) != 1
        or len(layers) != 1
        or len(layers[0].cells) != 1
        or len(inputs) != 1
    ):
        raise ValueError("Area requires one resolved Figure dataset/cell")

    terms_by_axis = {
        term.axis_id: term
        for selection in figure.document.layers[0].view.display_selections
        for term in selection.terms
        if term.axis_id not in gesture_axis_ids
    }
    for resolution in layers[0].resolutions:
        if resolution.axis_id not in gesture_axis_ids:
            # The evaluated resolution is the exact fixed/latest index painted
            # on this immutable front.  Keep it as a one-element range rather
            # than dropping the named axis: Area publishes sub-data, so its
            # facet coordinate must survive in the result schema/provenance.
            terms_by_axis[resolution.axis_id] = IndexRangeSelection(
                resolution.axis_id,
                resolution.index,
                resolution.index + 1,
            )
    if not terms_by_axis:
        return None
    return Selection(tuple(terms_by_axis.values()))


def bind_area_data_commit(
    source_identity: SourceIdentity,
    selection: Selection | HistogramValueRangeSelection,
    figure: DataFigure | None,
) -> FigureAreaCommit:
    """Bind one completed Area to the exact typed data view it was drawn on.

    This is the Area counterpart of :func:`bind_cross_data_commit`.  A focused
    Grid child is therefore not merely a rectangle/range over the base source:
    its frozen facet coordinates travel in the authoritative selection and in
    the derived signal's provenance.
    """

    if not isinstance(source_identity, SourceIdentity):
        raise TypeError("Area source_identity must be SourceIdentity")
    if not isinstance(selection, (Selection, HistogramValueRangeSelection)):
        raise TypeError("Area binding requires a Figure selection")
    # SiteMap is the sole non-DataFigure surface: its frontend presentation
    # already turns the rectangle into a complete logical-site Selection.
    if figure is None:
        if not isinstance(selection, Selection):
            raise TypeError("only SiteMap Area may omit a DataFigure context")
        return FigureAreaCommit(source_identity, selection)
    entries = tuple(figure.datasets.entries)
    if len(entries) != 1:
        raise ValueError("Area requires one resolved Figure dataset")
    if not source_identity_matches_snapshot(source_identity, entries[0].snapshot):
        raise ValueError("Area Figure belongs to another source generation")

    gesture_axis_ids = (
        frozenset(term.axis_id for term in selection.terms)
        if isinstance(selection, Selection)
        else frozenset()
    )
    context = _area_context_selection(
        figure,
        gesture_axis_ids=gesture_axis_ids,
    )
    if isinstance(selection, Selection):
        terms_by_axis = (
            {}
            if context is None
            else {term.axis_id: term for term in context.terms}
        )
        # The explicit gesture is the strongest statement on an axis.
        terms_by_axis.update((term.axis_id, term) for term in selection.terms)
        bound = Selection(tuple(terms_by_axis.values()))
    else:
        bound = replace(selection, source_selection=context)
    return FigureAreaCommit(source_identity, bound)


def _nearest_axis_position(axis, coordinate: float) -> int:
    coordinates = np.asarray(
        tuple(float(value) for value in numeric_curve_coordinates(axis)),
        dtype=np.float64,
    )
    return int(np.argmin(np.abs(coordinates - float(coordinate))))


def _nearest_curve_series(payload: CurvePanelPayload, x_position: int, y: float):
    eligible: list[tuple[float, int]] = []
    for index, series in enumerate(payload.series):
        curve = series.data
        if not bool(curve.validity[x_position]):
            continue
        value = curve.values[x_position]
        scalar = value.item() if isinstance(value, np.generic) else value
        try:
            distance = abs(float(scalar) - float(y))
        except (TypeError, ValueError):
            continue
        if math.isfinite(distance):
            eligible.append((distance, index))
    # An invalid clicked sample is still a meaningful typed result: preserve
    # its false validity rather than silently retargeting to another x value.
    return payload.series[min(eligible)[1] if eligible else 0]


def _histogram_hit(
    payload: HistogramPanelPayload,
    point: tuple[float, float],
):
    x, y = point
    edges = payload.bin_edges
    if x < float(edges[0]) or x > float(edges[-1]):
        raise ValueError("Cross point lies outside the displayed histogram bins")
    if x == float(edges[-1]):
        bin_index = len(edges) - 2
    else:
        bin_index = int(np.searchsorted(edges, x, side="right") - 1)
    if not 0 <= bin_index < len(edges) - 1:
        raise ValueError("Cross point does not identify a histogram bin")
    series_index = min(
        range(len(payload.series)),
        key=lambda index: (
            abs(float(payload.bin_counts[index][bin_index]) - y),
            index,
        ),
    )
    return (
        payload.series[series_index],
        (
            float(edges[bin_index]),
            float(edges[bin_index + 1]),
            bin_index == len(edges) - 2,
        ),
    )


def _cross_sample_document(
    figure: DataFigure,
    *,
    cell,
    series,
    sampled_indices: tuple[tuple[AxisId, int], ...],
) -> FigureDocument:
    """Narrow one exact display view to the selected cell/series/sample.

    The operation remains a display-derived Figure document.  It neither
    mutates the source block nor creates an authority transform.  Evaluating
    the narrowed document on later revisions avoids a second full raster/data
    projection for IMAGE and CURVE Cross outputs.
    """

    document = figure.document
    if len(document.layers) != 1:
        raise ValueError("Cross requires one Figure layer")
    layer = document.layers[0]
    view = layer.view
    address_by_id = {
        address.axis_id: address
        for address in (*cell.facet_address, *series.batch_address)
    }
    if len(address_by_id) != len(cell.facet_address) + len(series.batch_address):
        raise ValueError("Cross cell/series addresses repeat an axis")
    bindings = []
    for binding in view.axis_bindings:
        if binding.role not in {AxisViewRole.FACET, AxisViewRole.BATCH}:
            bindings.append(binding)
            continue
        try:
            address = address_by_id[binding.axis_id]
        except KeyError as exc:
            raise ValueError("Cross display omitted a facet/batch address") from exc
        bindings.append(
            AxisViewBinding(
                binding.axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(address.index),
            )
        )

    sampled_by_id = dict(sampled_indices)
    if len(sampled_by_id) != len(sampled_indices):
        raise ValueError("Cross sample repeats an axis")
    retained_terms = tuple(
        term
        for selection in view.display_selections
        for term in selection.terms
        if term.axis_id not in sampled_by_id
    )
    sample_terms = tuple(
        IndexSelection(axis_id, index)
        for axis_id, index in sampled_indices
    )
    terms = (*retained_terms, *sample_terms)
    sample_view = replace(
        view,
        axis_bindings=tuple(bindings),
        display_selections=(() if not terms else (Selection(tuple(terms)),)),
    )
    identity = canonical_digest(
        {
            "owner": "zlc_frontend.figure-cross-sample-document",
            "source_document": figure_document_to_tree(document),
            "sample_view": view_spec_to_tree(sample_view),
        }
    )
    return FigureDocument(
        f"cross-sample-{identity}",
        0,
        document.datasets,
        (FigureLayer(layer.layer_id, layer.dataset_id, sample_view),),
        document.selections,
    )


def bind_cross_data_commit(
    source_identity: SourceIdentity,
    point: tuple[float, float],
    figure: DataFigure,
    payload: object,
) -> FigureCrossCommit:
    """Bind one painted Cross gesture to the value that Figure displayed."""

    if not isinstance(source_identity, SourceIdentity):
        raise TypeError("Cross source_identity must be SourceIdentity")
    if not isinstance(figure, DataFigure):
        raise TypeError("Cross binding requires one DataFigure")
    point = tuple(float(value) for value in point)
    if len(point) != 2 or not all(math.isfinite(value) for value in point):
        raise ValueError("Cross point must contain two finite coordinates")
    entries = tuple(figure.datasets.entries)
    layers = tuple(figure.evaluated.layers)
    inputs = tuple(figure.evaluated.inputs)
    if (
        len(entries) != 1
        or len(layers) != 1
        or len(layers[0].cells) != 1
        or len(inputs) != 1
    ):
        raise ValueError("Cross requires one resolved Figure dataset/cell")
    if not source_identity_matches_snapshot(source_identity, entries[0].snapshot):
        raise ValueError("Cross Figure belongs to another source generation")
    cell = layers[0].cells[0]
    histogram_bin = None

    if isinstance(payload, SiteMapPanelPayload):
        raise TypeError("SiteMap Cross has no unambiguous single data value")
    if not isinstance(
        payload,
        (ImagePanelPayload, CurvePanelPayload, HistogramPanelPayload),
    ):
        raise TypeError("painted payload has no Cross data semantics")
    if payload.evaluated_input.ref != inputs[0].ref:
        raise ValueError("Cross payload and Figure revisions differ")

    if isinstance(payload, ImagePanelPayload):
        matches = tuple(
            candidate for candidate in cell.series if candidate.data is payload.image
        )
        if len(matches) != 1:
            raise ValueError("Cross image is absent or ambiguous in its Figure cell")
        series = matches[0]
        x_position = _nearest_axis_position(payload.image.x_axis, point[0])
        y_position = _nearest_axis_position(payload.image.y_axis, point[1])
        sampled_indices = (
            (payload.image.x_axis.axis_id, payload.image.x_axis.indices[x_position]),
            (payload.image.y_axis.axis_id, payload.image.y_axis.indices[y_position]),
        )
    elif isinstance(payload, CurvePanelPayload):
        if tuple(candidate.batch_address for candidate in cell.series) != tuple(
            candidate.batch_address for candidate in payload.series
        ):
            raise ValueError("Cross curve series differ from their Figure cell")
        x_axis = payload.series[0].data.x_axis
        x_position = _nearest_axis_position(x_axis, point[0])
        series = _nearest_curve_series(payload, x_position, point[1])
        sampled_indices = ((x_axis.axis_id, x_axis.indices[x_position]),)
    else:
        if tuple(candidate.batch_address for candidate in cell.series) != tuple(
            candidate.batch_address for candidate in payload.series
        ):
            raise ValueError("Cross histogram series differ from their Figure cell")
        series, histogram_bin = _histogram_hit(payload, point)
        sampled_indices = ()

    return FigureCrossCommit(
        source_identity,
        _cross_sample_document(
            figure,
            cell=cell,
            series=series,
            sampled_indices=sampled_indices,
        ),
        point,
        histogram_bin,
    )


def source_identity_matches_snapshot(
    source_identity: SourceIdentity,
    snapshot: OwnedSnapshot,
) -> bool:
    """Whether a generation-scoped Figure intent applies to this revision."""

    if not isinstance(source_identity, SourceIdentity):
        raise TypeError("source_identity must be SourceIdentity")
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    ref = snapshot.ref
    return (
        source_identity.block_id == ref.block_id
        and source_identity.stream_generation == ref.stream_generation
        and source_identity.schema_fingerprint == ref.schema_fingerprint
    )


def _output_name(output_name: str) -> str:
    output = str(output_name).strip()
    if not output:
        raise ValueError("Figure output name must be non-empty")
    return output


def area_data_output_presentation(
    source_contract_id: str | None,
) -> FigureOutputPresentation:
    """Return the sole public presentation of a Figure Area dataset."""

    return FigureOutputPresentation(
        AREA_DATA_OUTPUT,
        figure_output_contract_id(
            AREA_DATA_OUTPUT,
            source_contract_id=source_contract_id,
        ),
        "Area data",
        "Area data",
        "Dataset inside the committed Figure Area selection.",
    )


def figure_output_revision_ref(
    output_name: str,
    source_ref: DatasetRevisionRef,
    output_schema,
    semantic_identity: Mapping[str, object],
) -> DatasetRevisionRef:
    output = _output_name(output_name)
    generation = canonical_digest(
        {
            "owner": "zlc_frontend.figure-output-generation",
            "output_name": output,
            "source_block_id": source_ref.block_id.value,
            "source_generation": source_ref.stream_generation.value,
            "output_schema": output_schema.fingerprint,
            "semantic_identity": dict(semantic_identity),
        }
    )
    return DatasetRevisionRef(
        BlockId(f"figure-output-{generation[:32]}"),
        StreamGenerationId(f"figure-output-{generation}"),
        output_schema.fingerprint,
        source_ref.revision,
    )


def figure_derivation_digest(
    output_name: str,
    snapshot: OwnedSnapshot,
    semantic_identity: Mapping[str, object],
) -> str:
    return canonical_digest(
        {
            "owner": "zlc_frontend.figure-output-derivation",
            "output_name": _output_name(output_name),
            "output_ref": dataset_revision_ref_to_tree(snapshot.ref),
            "semantic_identity": dict(semantic_identity),
        }
    )


def materialize_area_snapshot(
    source: OwnedSnapshot,
    selection: Selection,
    *,
    output_name: str = AREA_DATA_OUTPUT,
) -> OwnedSnapshot:
    """Materialise one accepted Area selection without flattening or reducing."""

    return materialize_dataset_selection(
        source,
        selection,
        reference_for=lambda output_schema: figure_output_revision_ref(
            output_name,
            source.ref,
            output_schema,
            {"selection": selection_to_tree(selection)},
        ),
    )


def figure_derived_signal(
    output_name: str,
    snapshot: OwnedSnapshot,
    source: FigureSource,
    *,
    preserve_source_coverage: bool,
    presentation: FigureOutputPresentation,
    source_transform: DataTransformSpec | None = None,
    derivation_digest: str | None = None,
) -> FigureDerivedSignal:
    if not isinstance(source, FigureSource):
        raise TypeError("Figure output source must be FigureSource")
    return FigureDerivedSignal(
        snapshot=snapshot,
        source_ref=source.snapshot.ref,
        derivation_digest=(
            figure_derivation_digest(
                output_name,
                snapshot,
                {"kind": "materialized-figure-output"},
            )
            if derivation_digest is None
            else derivation_digest
        ),
        presentation=presentation,
        preserve_source_coverage=preserve_source_coverage,
        source_transform=source_transform,
    )


def materialize_area_outputs(
    source: FigureSource,
    selection: Selection | HistogramValueRangeSelection,
) -> dict[str, FigureDerivedSignal]:
    """Return selected data plus one typed bound vector per selected axis."""

    if not isinstance(source, FigureSource):
        raise TypeError("Area source must be FigureSource")
    if isinstance(selection, HistogramValueRangeSelection):
        snapshot = source.snapshot
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("Histogram Area source does not own a dataset snapshot")
        source_selection = selection.source_selection
        working = snapshot
        if source_selection is not None:
            context_identity = {
                "kind": "histogram-area-context",
                "source_selection": selection_to_tree(source_selection),
            }
            working = materialize_dataset_selection(
                snapshot,
                source_selection,
                reference_for=lambda output_schema: figure_output_revision_ref(
                    AREA_DATA_OUTPUT,
                    snapshot.ref,
                    output_schema,
                    context_identity,
                ),
            )
        values = working.block.values
        if values.dtype.kind not in "biuf":
            raise TypeError("Histogram Area requires real numeric source values")
        accepted = (
            np.isfinite(values)
            & (values >= selection.lower)
            & (values <= selection.upper)
        )
        semantic_identity = {
            "histogram_value_range": [selection.lower, selection.upper],
            "source_selection": (
                None
                if source_selection is None
                else selection_to_tree(source_selection)
            ),
        }
        selected = materialize_dataset_acceptance_mask(
            working,
            accepted,
            reference_for=lambda output_schema: figure_output_revision_ref(
                AREA_DATA_OUTPUT,
                snapshot.ref,
                output_schema,
                semantic_identity,
            ),
        )
        return {
            AREA_DATA_OUTPUT: figure_derived_signal(
                AREA_DATA_OUTPUT,
                selected,
                source,
                preserve_source_coverage=True,
                presentation=area_data_output_presentation(
                    source.source_contract_id,
                ),
                # The published values include both the optional Figure
                # context selection and this histogram value-range acceptance
                # mask.  Until that mask has a first-class authoritative
                # DataTransform operation, exposing only ``source_selection``
                # would claim a replayable transform that produces different
                # data.  Keep the complete materialized output, but do not
                # advertise it as association-preserving.
                source_transform=None,
            )
        }
    site_map = source.site_map
    if site_map is not None:
        if not isinstance(selection, Selection):
            raise TypeError("SiteMap Area requires a typed Selection")
        return dict(site_map.materialize_area_outputs(source, selection))
    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("Area source signal does not own a dataset snapshot")
    selected = materialize_area_snapshot(snapshot, selection)
    return {
        AREA_DATA_OUTPUT: figure_derived_signal(
            AREA_DATA_OUTPUT,
            selected,
            source,
            preserve_source_coverage=True,
            presentation=area_data_output_presentation(
                source.source_contract_id,
            ),
            source_transform=DataTransformSpec((selection,)),
        )
    }


def materialize_cross_outputs(
    source: FigureSource,
    commit: FigureCrossCommit,
) -> dict[str, FigureDerivedSignal]:
    """Publish the data value at a locked Cross; never publish coordinates."""

    if not isinstance(source, FigureSource):
        raise TypeError("Cross source must be FigureSource")
    if not isinstance(commit, FigureCrossCommit):
        raise TypeError("Cross output requires FigureCrossCommit")
    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("Cross source signal does not own a dataset snapshot")
    if not source_identity_matches_snapshot(commit.source_identity, snapshot):
        raise ValueError("Cross commit belongs to another source generation")

    descriptor = commit.sample_document.datasets[0]
    figure = DataFigure(
        commit.sample_document,
        ResolvedDatasetMap((ResolvedDataset(descriptor.dataset_id, snapshot),)),
    )
    layers = tuple(figure.evaluated.layers)
    if (
        len(layers) != 1
        or len(layers[0].cells) != 1
        or len(layers[0].cells[0].series) != 1
    ):
        raise RuntimeError("Cross sample document did not resolve one value series")
    data = layers[0].cells[0].series[0].data
    intent = commit.sample_document.layers[0].view.intent
    if intent is ViewIntent.IMAGE:
        if not isinstance(data, EvaluatedImage):
            raise TypeError("Cross image evaluation returned another data kind")
        values = np.asarray(data.values)
        validity = np.asarray(data.validity, dtype=np.bool_)
        if values.shape != (1, 1) or validity.shape != (1, 1):
            raise RuntimeError("Cross image sample did not narrow to one pixel")
        scalar_values = values.reshape(1)
        scalar_validity = validity.reshape(1)
        unit = data.value_unit
        coordinates = (
            (data.x_axis.name, data.x_axis.coordinates[0]),
            (data.y_axis.name, data.y_axis.coordinates[0]),
        )
        description = "Cross-selected image value at " + ", ".join(
            f"{name}={value}" for name, value in coordinates
        )
    elif intent is ViewIntent.CURVE:
        if not isinstance(data, EvaluatedCurve):
            raise TypeError("Cross curve evaluation returned another data kind")
        values = np.asarray(data.values)
        validity = np.asarray(data.validity, dtype=np.bool_)
        if values.shape != (1,) or validity.shape != (1,):
            raise RuntimeError("Cross curve sample did not narrow to one point")
        scalar_values = values
        scalar_validity = validity
        unit = data.value_unit
        description = (
            "Cross-selected curve value at "
            f"{data.x_axis.name}={data.x_axis.coordinates[0]}"
        )
    elif intent is ViewIntent.HISTOGRAM:
        if not isinstance(data, EvaluatedHistogram):
            raise TypeError("Cross histogram evaluation returned another data kind")
        if commit.histogram_bin is None:
            raise RuntimeError("Histogram Cross lost its committed bin")
        lower, upper, include_upper = commit.histogram_bin
        samples = np.asarray(data.samples)
        selected = (samples >= lower) & (
            (samples <= upper) if include_upper else (samples < upper)
        )
        scalar_values = np.asarray(
            (int(np.count_nonzero(selected)),),
            dtype=np.dtype("<i8"),
        )
        scalar_validity = np.ones((1,), dtype=np.bool_)
        unit = None
        close = "]" if include_upper else ")"
        description = (
            f"Cross-selected histogram bin count for [{lower}, {upper}{close}."
        )
    else:  # FigureCrossCommit closes this; retain the materializer boundary.
        raise RuntimeError("Cross sample document has no data-value semantics")

    semantic_identity = {
        "kind": "cross-data",
        "sample_document": figure_document_to_tree(commit.sample_document),
        "point": commit.point,
        "histogram_bin": commit.histogram_bin,
    }
    output_snapshot = materialize_scalar_dataset(
        snapshot.ref,
        scalar_values,
        valid=bool(scalar_validity[0]),
        unit=unit,
        reference_for=lambda schema: figure_output_revision_ref(
            CROSS_DATA_OUTPUT,
            snapshot.ref,
            schema,
            semantic_identity,
        ),
    )
    source_transform = _cross_source_transform(
        snapshot,
        commit,
        output_snapshot,
    )
    return {
        CROSS_DATA_OUTPUT: figure_derived_signal(
            CROSS_DATA_OUTPUT,
            output_snapshot,
            source,
            preserve_source_coverage=False,
            presentation=FigureOutputPresentation(
                CROSS_DATA_OUTPUT,
                FIGURE_CROSS_DATA_OUTPUT_CONTRACT_ID,
                "Cross data",
                "Cross data",
                description,
            ),
            source_transform=source_transform,
            derivation_digest=figure_derivation_digest(
                CROSS_DATA_OUTPUT,
                output_snapshot,
                semantic_identity,
            ),
        )
    }


def _cross_source_transform(
    source: OwnedSnapshot,
    commit: FigureCrossCommit,
    output: OwnedSnapshot,
) -> DataTransformSpec | None:
    """Return the exact per-event projection behind an IMAGE/CURVE Cross.

    Association can be preserved only when the painted value is a pure
    selection/reduction inside one producer event.  A multi-cell Figure or a
    histogram bin count crosses that boundary: the former combines event
    carriers and the latter is a value predicate not expressible by
    ``DataTransformSpec``.  Those outputs remain ordinary causal signals but
    intentionally expose no exact PulseScan association capability.

    The transform is derived from the already-committed sample ``ViewSpec``;
    ndarray rank and axis spelling never participate.  A final byte/value and
    validity comparison proves that this authority computes the same scalar
    the Figure published before it is attached to the signal.
    """

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("Cross transform source must be OwnedSnapshot")
    if not isinstance(commit, FigureCrossCommit):
        raise TypeError("Cross transform requires FigureCrossCommit")
    if not isinstance(output, OwnedSnapshot):
        raise TypeError("Cross transform output must be OwnedSnapshot")
    view = commit.sample_document.layers[0].view
    if view.intent is ViewIntent.HISTOGRAM:
        return None
    schema = source.block.schema
    if schema.cell_layout.storage_size != 1:
        return None

    data_axes = tuple(schema.cell_schema.data_axes)
    data_ids = {axis.axis_id for axis in data_axes}
    selected_by_id = {
        term.axis_id: term
        for selection in view.display_selections
        for term in selection.terms
        if term.axis_id in data_ids
    }
    if len(selected_by_id) != sum(
        1
        for selection in view.display_selections
        for term in selection.terms
        if term.axis_id in data_ids
    ):
        raise RuntimeError("Cross sample repeats a data-axis selection")

    reduced: list[AxisId] = []
    reduction_method = None
    for axis in data_axes:
        binding = view.binding(axis.axis_id)
        if binding.role is AxisViewRole.REDUCED:
            if binding.reduction is None:
                raise RuntimeError("Cross reduced axis lost its reducer")
            method = {
                "MEAN": ReductionMethod.MEAN,
                "SUM": ReductionMethod.SUM,
            }.get(binding.reduction.method.value)
            if method is None:
                return None
            if reduction_method is not None and reduction_method is not method:
                return None
            reduction_method = method
            reduced.append(axis.axis_id)
            continue
        if axis.axis_id in selected_by_id:
            continue
        if binding.role in {AxisViewRole.SELECTED, AxisViewRole.SLIDER}:
            selector = binding.selector
            if isinstance(selector, FixedIndex):
                selected_by_id[axis.axis_id] = IndexSelection(
                    axis.axis_id,
                    selector.index,
                )
                continue
            if axis.size == 1:
                selected_by_id[axis.axis_id] = IndexSelection(axis.axis_id, 0)
                continue
            return None
        if axis.size == 1:
            selected_by_id[axis.axis_id] = IndexSelection(axis.axis_id, 0)
            continue
        return None

    operations = []
    if selected_by_id:
        operations.append(Selection(tuple(selected_by_id.values())))
    if reduced:
        assert reduction_method is not None
        operations.append(
            ReductionSpec(
                tuple(reduced),
                reduction_method,
                missing_policy=MissingPolicy.OMIT_MISSING,
                validity_policy=ValidityPolicy.OMIT_INVALID,
            )
        )
    if not operations:
        return None
    spec = DataTransformSpec(tuple(operations))
    transformed = apply_transform(source, commit_transform(schema, spec))
    expected = output.block
    actual_values = np.asarray(transformed.values)
    actual_validity = np.asarray(transformed.expanded_validity(), dtype=np.bool_)
    if (
        actual_values.shape != (1, 1)
        or actual_validity.shape != (1, 1)
        or expected.values.shape != (1, 1, 1)
        or not np.array_equal(actual_values.reshape(1), expected.values.reshape(1))
        or bool(actual_validity[0, 0])
        != bool(expand_dataset_validity(expected.validity, expected.schema)[0, 0, 0])
    ):
        raise RuntimeError(
            "Cross authoritative transform differs from its published scalar"
        )
    return spec


def materialize_fit_outputs(
    source: FigureSource,
    result,
) -> dict[str, FigureDerivedSignal]:
    """Publish one typed ``fit.<parameter>`` dataset per model parameter.

    Values retain the result's named batch axes and sparse layout.  A failed
    batch cell is represented by CellValidity=False, never by dropping the
    cell, averaging it, or exposing the solver's canonical numeric zero as a
    physically valid parameter.
    """

    from zlc_data import FitResultBatch

    if not isinstance(source, FigureSource):
        raise TypeError("Fit source must be FigureSource")
    if not isinstance(result, FitResultBatch):
        raise TypeError("Fit result must be FitResultBatch")
    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("Fit source signal does not own a dataset snapshot")
    if result.source_ref != snapshot.ref:
        raise ValueError("Fit result belongs to another visible source revision")

    spec_tree = fit_spec_to_tree(result.spec)
    snapshots = materialize_fit_parameter_snapshots(
        result,
        reference_for=lambda parameter_name, schema: figure_output_revision_ref(
            f"{FIT_OUTPUT_PREFIX}{parameter_name}",
            result.source_ref,
            schema,
            {
                "kind": "figure-fit-parameter",
                "fit_spec": spec_tree,
                "parameter_name": parameter_name,
            },
        ),
    )
    output: dict[str, FigureDerivedSignal] = {}
    for parameter_name, fit_snapshot in snapshots.items():
        output_name = f"{FIT_OUTPUT_PREFIX}{parameter_name}"
        output[output_name] = FigureDerivedSignal(
            snapshot=fit_snapshot,
            source_ref=result.source_ref,
            presentation=FigureOutputPresentation(
                output_name,
                FIGURE_FIT_PARAMETER_OUTPUT_CONTRACT_ID,
                parameter_name,
                parameter_name,
                (
                    f"Figure Fit parameter {parameter_name} from model "
                    f"{result.spec.model_id}."
                ),
            ),
            preserve_source_coverage=False,
            derivation_digest=canonical_digest(
                {
                    "owner": "zlc_frontend.figure-fit-output",
                    "source_ref": dataset_revision_ref_to_tree(result.source_ref),
                    "fit_spec": spec_tree,
                    "parameter_name": parameter_name,
                }
            ),
        )
    return output
