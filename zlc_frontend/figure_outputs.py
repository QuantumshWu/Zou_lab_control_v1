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

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from zlc_storage import canonical_digest, canonical_text, sha256_text

from zlc_data import (
    AxisSourceRef,
    AxisSpec,
    BlockId,
    CommittedTransform,
    DataTransformSpec,
    DataBlock,
    DatasetRevisionRef,
    IndexRangeSelection,
    IndexSelection,
    OwnedSnapshot,
    ReductionSpec,
    Selection,
    StreamGenerationId,
)
from zlc_data.codec import dataset_revision_ref_to_tree
from zlc_data.fit import fit_spec_to_tree
from zlc_data.output_contract import (
    AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
    projected_dataset_output_contract_id,
)
from zlc_data.selection import resolve_selection_indices, selection_to_tree
from zlc_data.snapshot_projection import (
    materialize_dataset_acceptance_mask,
    materialize_fit_parameter_snapshots,
    materialize_scalar_dataset,
)
from zlc_data.transform import apply_transform, commit_transform
from zlc_data.transform_codec import committed_transform_to_tree
from zlc_data.value import expand_dataset_validity
from .curve_display import numeric_curve_coordinates
from .data_figure import DataFigure
from .figure import (
    AxisViewRole,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    ViewIntent,
)
from .figure.contract import _display_reduction_spec
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
    "FigureOutputPresentation",
    "HistogramValueRangeSelection",
    "area_data_output_presentation",
    "bind_area_data_commit",
    "bind_cross_data_commit",
    "cross_data_output_presentation",
    "fit_parameter_output_presentation",
    "figure_event_transform",
    "figure_selector_identity",
    "figure_derived_signal",
    "figure_derivation_digest",
    "figure_output_contract_id",
    "figure_output_revision_ref",
    "materialize_area_outputs",
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
    authority: CommittedTransform | Selection | "HistogramValueRangeSelection"

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("Area source_identity must be SourceIdentity")
        if not isinstance(
            self.authority,
            (CommittedTransform, Selection, HistogramValueRangeSelection),
        ):
            raise TypeError("Area authority has an unsupported type")


@dataclass(frozen=True, slots=True)
class FigureCrossCommit:
    """One locked Cross value selection bound to a producer generation.

    The committed transform is the sole data authority: exact point rows,
    selected facet/batch/sample coordinates, and any display reduction needed
    to reproduce the clicked value are frozen once at gesture bind.  The
    Figure document remains presentation and is never copied into a data
    commit. Histogram bins are renderer display state rather than Dataset
    axes, so the selected immutable interval is carried separately. ``point``
    is provenance for the gesture; it is never published as a signal.
    """

    source_identity: SourceIdentity
    transform: CommittedTransform
    intent: ViewIntent
    point: tuple[float, float]
    histogram_bin: tuple[float, float, bool] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("Cross source_identity must be SourceIdentity")
        if not isinstance(self.transform, CommittedTransform):
            raise TypeError("Cross transform must be CommittedTransform")
        if self.intent not in {
            ViewIntent.IMAGE,
            ViewIntent.CURVE,
            ViewIntent.HISTOGRAM,
        }:
            raise ValueError("Cross data requires IMAGE, CURVE, or HISTOGRAM")
        point = tuple(float(value) for value in self.point)
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise ValueError("Cross point must contain two finite coordinates")
        histogram_bin = self.histogram_bin
        if self.intent is ViewIntent.HISTOGRAM:
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
class FigureDerivedSignal:
    """One headless Figure-derived dataset plus derivation identity.

    Run identity and routing names intentionally do not appear here.  Those are
    application-shell concerns; the Figure owns only the immutable snapshot,
    exact derivation, and whether source coverage remains meaningful for the
    derived value.  Presentation is generation-static topology metadata, never
    a per-revision sidecar on this value.
    """

    snapshot: OwnedSnapshot
    source_ref: DatasetRevisionRef
    derivation_digest: str
    preserve_source_coverage: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("Figure signal snapshot must be OwnedSnapshot")
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("Figure signal source_ref must be DatasetRevisionRef")
        if self.snapshot.ref.revision != self.source_ref.revision:
            raise ValueError("Figure signal revision differs from its source")
        sha256_text(self.derivation_digest, "Figure signal derivation_digest")
        if type(self.preserve_source_coverage) is not bool:
            raise TypeError("preserve_source_coverage must be bool")


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
    source_transform: CommittedTransform

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("histogram Area bounds must be finite")
        if lower > upper:
            raise ValueError("histogram Area lower bound exceeds upper bound")
        if not isinstance(self.source_transform, CommittedTransform):
            raise TypeError(
                "histogram Area source_transform must be CommittedTransform"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def _figure_dataset_context(figure: DataFigure):
    """Return the one exact Dataset/view/cell owned by a selector surface."""

    if not isinstance(figure, DataFigure):
        raise TypeError("Figure output binding requires one DataFigure")
    entries = tuple(figure.datasets.entries)
    layers = tuple(figure.evaluated.layers)
    inputs = tuple(figure.evaluated.inputs)
    if (
        len(entries) != 1
        or len(figure.document.layers) != 1
        or len(layers) != 1
        or len(layers[0].cells) != 1
        or len(inputs) != 1
        or inputs[0].ref != entries[0].snapshot.ref
    ):
        raise ValueError("Figure output requires one resolved Dataset/cell")
    return (
        entries[0].snapshot,
        figure.document.layers[0].view,
        layers[0],
        layers[0].cells[0],
    )


def _data_axes(data) -> tuple[object, ...]:
    if isinstance(data, EvaluatedImage):
        return (data.x_axis, data.y_axis)
    if isinstance(data, EvaluatedCurve):
        return (data.x_axis,)
    if isinstance(data, EvaluatedHistogram):
        return ()
    raise TypeError("Figure output surface has no selector data axes")


def _query_axis_spec(axis) -> AxisSpec:
    return AxisSpec(
        axis.axis_id,
        axis.name,
        axis.role,
        len(axis.indices),
        axis.coordinates,
        axis.unit,
        axis.coordinate_frame,
    )


def _restrict_point_rows_by_address(
    schema,
    ordinals: tuple[int, ...],
    source: AxisSourceRef,
    *,
    index: int,
    coordinate,
) -> tuple[int, ...]:
    if source.kind == AxisSourceRef.POINT_ROWS:
        wanted = int(coordinate)
        result = tuple(ordinal for ordinal in ordinals if ordinal == wanted)
    elif source.kind == AxisSourceRef.POINT_COORDINATE:
        assert source.axis_id is not None
        column = schema.point_table.column(source.axis_id)
        result = tuple(
            ordinal
            for ordinal in ordinals
            if column.values[ordinal] == coordinate
        )
    elif source.kind == AxisSourceRef.GRID_DIMENSION:
        topology = schema.grid_topology
        if topology is None or source.axis_id not in topology.dimension_ids:
            raise ValueError("Figure address refers to an absent Grid dimension")
        position = topology.dimension_ids.index(source.axis_id)
        result = tuple(
            ordinal
            for ordinal in ordinals
            if topology.row_to_cell[ordinal][position] == index
        )
    else:
        raise ValueError(f"{source.kind} cannot identify a point group")
    if not result:
        raise ValueError("Figure context resolved no physical point row")
    return result


def _restrict_point_rows_by_indices(
    schema,
    ordinals: tuple[int, ...],
    source: AxisSourceRef,
    indices: tuple[int, ...],
) -> tuple[int, ...]:
    wanted = frozenset(indices)
    if not wanted:
        raise ValueError("Figure selector resolved no source index")
    if source.kind in {
        AxisSourceRef.POINT_ORDINAL,
        AxisSourceRef.POINT_COORDINATE,
    }:
        result = tuple(ordinal for ordinal in ordinals if ordinal in wanted)
    elif source.kind == AxisSourceRef.GRID_DIMENSION:
        topology = schema.grid_topology
        if topology is None or source.axis_id not in topology.dimension_ids:
            raise ValueError("Figure selector refers to an absent Grid dimension")
        position = topology.dimension_ids.index(source.axis_id)
        result = tuple(
            ordinal
            for ordinal in ordinals
            if topology.row_to_cell[ordinal][position] in wanted
        )
    else:
        raise ValueError(f"{source.kind} cannot be a point selector axis")
    if not result:
        raise ValueError("Figure selector resolved no physical point row")
    return result


def _tensor_index_term(source: AxisSourceRef, index: int, *, preserve: bool):
    if source.kind != AxisSourceRef.TENSOR or source.axis_id is None:
        raise ValueError("tensor selection requires a tensor source")
    return (
        IndexRangeSelection(source.axis_id, index, index + 1)
        if preserve
        else IndexSelection(source.axis_id, index)
    )


def _compile_figure_transform(
    figure: DataFigure,
    *,
    series=None,
    include_batch: bool,
    include_reductions: bool,
    preserve_context_axes: bool,
    gesture_selection: Selection | None = None,
    sample_indices: tuple[tuple[AxisSourceRef, int], ...] = (),
) -> CommittedTransform:
    """Freeze one visible Figure context as data-owned authority.

    The frontend resolves display roles once, but the committed value contains
    only data concepts: exact point ordinals plus tensor selections/reductions.
    No narrowed Figure document or display-side selection survives this edge.
    """

    snapshot, view, layer, cell = _figure_dataset_context(figure)
    schema = snapshot.block.schema
    point_ordinals = tuple(
        range(schema.point_table.row_count)
        if view.point_ordinals is None
        else view.point_ordinals
    )
    terms_by_axis = {}

    def apply_address(address) -> None:
        nonlocal point_ordinals
        source = address.source
        if source.kind == AxisSourceRef.TENSOR:
            term = _tensor_index_term(
                source,
                address.index,
                preserve=preserve_context_axes,
            )
            terms_by_axis[term.axis_id] = term
        else:
            point_ordinals = _restrict_point_rows_by_address(
                schema,
                point_ordinals,
                source,
                index=address.index,
                coordinate=address.coordinate,
            )

    for resolution in layer.resolutions:
        apply_address(resolution)
    for address in cell.facet_address:
        apply_address(address)
    if include_batch:
        if series is None or series not in cell.series:
            raise ValueError("Cross series is absent from its Figure cell")
        for address in series.batch_address:
            apply_address(address)

    if gesture_selection is not None:
        axes_by_id = {}
        for candidate in cell.series:
            for axis in _data_axes(candidate.data):
                prior = axes_by_id.setdefault(axis.axis_id, axis)
                if prior.source != axis.source or prior.indices != axis.indices:
                    raise ValueError("Area axis differs across visible Figure series")
        for term in gesture_selection.terms:
            try:
                axis = axes_by_id[term.axis_id]
            except KeyError as exc:
                raise ValueError("Area gesture axis is absent from its Figure") from exc
            local, _drop = resolve_selection_indices(_query_axis_spec(axis), term)
            source_indices = tuple(axis.indices[index] for index in local)
            if axis.source.kind == AxisSourceRef.TENSOR:
                if source_indices != tuple(
                    range(source_indices[0], source_indices[0] + len(source_indices))
                ):
                    raise ValueError("Area tensor selection is not a contiguous range")
                terms_by_axis[term.axis_id] = IndexRangeSelection(
                    term.axis_id,
                    source_indices[0],
                    source_indices[-1] + 1,
                )
            else:
                point_ordinals = _restrict_point_rows_by_indices(
                    schema,
                    point_ordinals,
                    axis.source,
                    source_indices,
                )

    for source, index in sample_indices:
        if source.kind == AxisSourceRef.TENSOR:
            term = _tensor_index_term(source, index, preserve=False)
            terms_by_axis[term.axis_id] = term
        else:
            point_ordinals = _restrict_point_rows_by_indices(
                schema,
                point_ordinals,
                source,
                (index,),
            )

    operations = []
    if terms_by_axis:
        operations.append(Selection(tuple(terms_by_axis.values())))
    if include_reductions:
        reduction = _display_reduction_spec(view)
        if reduction is not None:
            operations.append(reduction)
    return commit_transform(
        schema,
        DataTransformSpec(tuple(operations)),
        point_ordinals=point_ordinals,
    )


def bind_area_data_commit(
    source_identity: SourceIdentity,
    selection: Selection | tuple[float, float],
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
    if not isinstance(selection, (Selection, tuple)):
        raise TypeError("Area binding requires a Figure selection")
    # SiteMap is the sole non-DataFigure surface: its frontend presentation
    # already turns the rectangle into a complete logical-site Selection.
    if figure is None:
        if not isinstance(selection, Selection):
            raise TypeError("only SiteMap Area may omit a DataFigure context")
        return FigureAreaCommit(source_identity, selection)
    snapshot, _view, _layer, _cell = _figure_dataset_context(figure)
    if not source_identity_matches_snapshot(source_identity, snapshot):
        raise ValueError("Area Figure belongs to another source generation")
    transform = _compile_figure_transform(
        figure,
        include_batch=False,
        include_reductions=False,
        preserve_context_axes=True,
        gesture_selection=selection if isinstance(selection, Selection) else None,
    )
    if isinstance(selection, Selection):
        return FigureAreaCommit(source_identity, transform)
    if len(selection) != 2:
        raise ValueError("Histogram Area requires lower/upper bounds")
    return FigureAreaCommit(
        source_identity,
        HistogramValueRangeSelection(selection[0], selection[1], transform),
    )


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
    snapshot, view, layer, cell = _figure_dataset_context(figure)
    if not source_identity_matches_snapshot(source_identity, snapshot):
        raise ValueError("Cross Figure belongs to another source generation")
    histogram_bin = None

    if isinstance(payload, SiteMapPanelPayload):
        raise TypeError("SiteMap Cross has no unambiguous single data value")
    if not isinstance(
        payload,
        (ImagePanelPayload, CurvePanelPayload, HistogramPanelPayload),
    ):
        raise TypeError("painted payload has no Cross data semantics")
    if payload.evaluated_input.ref != snapshot.ref:
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
            (payload.image.x_axis.source, payload.image.x_axis.indices[x_position]),
            (payload.image.y_axis.source, payload.image.y_axis.indices[y_position]),
        )
        expected_values = np.asarray(
            (payload.image.values[y_position, x_position],)
        )
        expected_validity = np.asarray(
            (payload.image.validity[y_position, x_position],),
            dtype=np.bool_,
        )
    elif isinstance(payload, CurvePanelPayload):
        if tuple(candidate.batch_address for candidate in cell.series) != tuple(
            candidate.batch_address for candidate in payload.series
        ):
            raise ValueError("Cross curve series differ from their Figure cell")
        x_axis = payload.series[0].data.x_axis
        x_position = _nearest_axis_position(x_axis, point[0])
        series = _nearest_curve_series(payload, x_position, point[1])
        sampled_indices = ((x_axis.source, x_axis.indices[x_position]),)
        expected_values = np.asarray((series.data.values[x_position],))
        expected_validity = np.asarray(
            (series.data.validity[x_position],),
            dtype=np.bool_,
        )
    else:
        if tuple(candidate.batch_address for candidate in cell.series) != tuple(
            candidate.batch_address for candidate in payload.series
        ):
            raise ValueError("Cross histogram series differ from their Figure cell")
        series, histogram_bin = _histogram_hit(payload, point)
        sampled_indices = ()
        expected_values = np.asarray(series.data.samples)
        expected_validity = np.ones(expected_values.shape, dtype=np.bool_)

    transform = _compile_figure_transform(
        figure,
        series=series,
        include_batch=True,
        include_reductions=True,
        preserve_context_axes=False,
        sample_indices=sampled_indices,
    )
    committed = FigureCrossCommit(
        source_identity,
        transform,
        view.intent,
        point,
        histogram_bin,
    )
    _validate_cross_transform(
        snapshot,
        committed,
        expected_values,
        expected_validity,
    )
    return committed


def _validate_cross_transform(
    source: OwnedSnapshot,
    commit: FigureCrossCommit,
    expected_values: np.ndarray,
    expected_validity: np.ndarray,
) -> None:
    """Prove the data authority reproduces the exact painted Cross source."""

    transformed = apply_transform(source, commit.transform)
    values = np.asarray(transformed.values)
    validity = np.asarray(
        expand_dataset_validity(transformed.validity, transformed.schema),
        dtype=np.bool_,
    )
    expected_values = np.asarray(expected_values)
    expected_validity = np.asarray(expected_validity, dtype=np.bool_)
    if commit.intent is ViewIntent.HISTOGRAM:
        values = values[validity]
        validity = np.ones(values.shape, dtype=np.bool_)
    else:
        values = values.reshape(-1)
        validity = validity.reshape(-1)
    if (
        values.shape != expected_values.shape
        or validity.shape != expected_validity.shape
        or not np.array_equal(values, expected_values, equal_nan=True)
        or not np.array_equal(validity, expected_validity)
    ):
        raise RuntimeError(
            "Cross committed transform differs from the exact painted data"
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


def _event_local_data_transform(
    source: FigureSource,
    transform: CommittedTransform,
) -> DataTransformSpec:
    """Project one Dataset transform onto a single signal event value.

    A ``SignalEvent`` carries only ``ValueSchema`` data axes; its repeat and
    point carriers are synthetic singleton axes owned by the event adapter.
    A generation-scoped Figure commit may therefore preserve association only
    when the rendered Dataset also has singleton carriers.  Operations on
    those singleton carriers are identity and are removed, while every real
    data-axis operation is retained verbatim.  Aggregated repeat/point domains
    are fan-in and must not claim one-event-to-one-output association.
    """

    schema = source.snapshot.block.schema
    if (
        schema.repeat_axis.size != 1
        or schema.point_table.row_count != 1
        or transform.exact_point_ordinals != (0,)
    ):
        raise ValueError(
            "Figure selector is not local to one repeat/point event"
        )
    repeat_axis_id = schema.repeat_axis.axis_id
    data_axis_ids = frozenset(
        axis.axis_id for axis in schema.cell_schema.data_axes
    )
    operations = []
    for operation in transform.spec.operations:
        if isinstance(operation, Selection):
            terms = []
            for term in operation.terms:
                if term.axis_id == repeat_axis_id:
                    continue
                if term.axis_id not in data_axis_ids:
                    raise ValueError(
                        "Figure selector selection is not event-local"
                    )
                terms.append(term)
            if terms:
                operations.append(Selection(tuple(terms)))
            continue
        if isinstance(operation, ReductionSpec):
            sources = []
            for axis_source in operation.sources:
                if (
                    axis_source.kind == AxisSourceRef.TENSOR
                    and axis_source.axis_id == repeat_axis_id
                ):
                    continue
                if (
                    axis_source.kind == AxisSourceRef.TENSOR
                    and axis_source.axis_id in data_axis_ids
                ):
                    sources.append(axis_source)
                    continue
                if axis_source.kind != AxisSourceRef.TENSOR:
                    # The point carrier is proven singleton above, so reducing
                    # it is the same event-local identity as reducing repeat.
                    continue
                raise ValueError(
                    "Figure selector reduction is not event-local"
                )
            if sources:
                operations.append(
                    ReductionSpec(
                        tuple(sources),
                        operation.method,
                        missing_policy=operation.missing_policy,
                        validity_policy=operation.validity_policy,
                        minimum_valid_count=operation.minimum_valid_count,
                    )
                )
            continue
        # Histogram analysis is a display fan-in, not a replayable event-cell
        # selector.  Keep this boundary closed if more operation kinds appear.
        raise ValueError("Figure selector transform is not event-local")
    return DataTransformSpec(tuple(operations))


def figure_event_transform(
    source: FigureSource,
    commit: FigureAreaCommit | FigureCrossCommit,
) -> DataTransformSpec:
    """Return the exact replayable event projection for a selector route.

    Absence of an event-local mapping is explicit: histogram value/bin
    analysis, SiteMap-specific selections, and point-row subsets raise instead
    of advertising a weaker transform.  Composition may then bind the route as
    continuous without granting FormalPulseScan association.
    """

    if not isinstance(source, FigureSource):
        raise TypeError("Figure event transform requires FigureSource")
    snapshot = source.snapshot
    if isinstance(commit, FigureAreaCommit):
        if not source_identity_matches_snapshot(commit.source_identity, snapshot):
            raise ValueError("Area commit belongs to another source generation")
        transform = commit.authority
        if not isinstance(transform, CommittedTransform):
            raise ValueError("Area selection has no replayable event transform")
    elif isinstance(commit, FigureCrossCommit):
        if not source_identity_matches_snapshot(commit.source_identity, snapshot):
            raise ValueError("Cross commit belongs to another source generation")
        if commit.intent is ViewIntent.HISTOGRAM:
            raise ValueError("Histogram Cross is event fan-in")
        transform = commit.transform
    else:
        raise TypeError("selector commit must be FigureAreaCommit or FigureCrossCommit")
    full_rows = tuple(range(snapshot.block.schema.point_table.row_count))
    if transform.exact_point_ordinals != full_rows:
        raise ValueError("point-row selection has no one-to-one event mapping")
    return _event_local_data_transform(source, transform)


def figure_selector_identity(
    commit: FigureAreaCommit | FigureCrossCommit,
) -> str:
    """Canonical generation identity for one committed selector route."""

    if isinstance(commit, FigureAreaCommit):
        authority = commit.authority
        if isinstance(authority, CommittedTransform):
            payload = {"committed_transform": committed_transform_to_tree(authority)}
        elif isinstance(authority, Selection):
            payload = {"selection": selection_to_tree(authority)}
        elif isinstance(authority, HistogramValueRangeSelection):
            payload = {
                "histogram_value_range": [authority.lower, authority.upper],
                "source_transform": committed_transform_to_tree(
                    authority.source_transform
                ),
            }
        else:  # FigureAreaCommit closes this boundary.
            raise TypeError("Area commit has another authority type")
        kind = "area"
        source_identity = commit.source_identity
    elif isinstance(commit, FigureCrossCommit):
        kind = "cross"
        source_identity = commit.source_identity
        payload = {
            "transform": committed_transform_to_tree(commit.transform),
            "intent": commit.intent.value,
            "point": commit.point,
            "histogram_bin": commit.histogram_bin,
        }
    else:
        raise TypeError("selector commit must be FigureAreaCommit or FigureCrossCommit")
    return canonical_digest(
        {
            "owner": "zlc_frontend.figure-selector-generation",
            "kind": kind,
            "source": {
                "block_id": source_identity.block_id.value,
                "stream_generation": source_identity.stream_generation.value,
                "schema_fingerprint": source_identity.schema_fingerprint,
            },
            "authority": payload,
        }
    )


def cross_data_output_presentation(
    commit: FigureCrossCommit,
) -> FigureOutputPresentation:
    """Return the sole public presentation of one committed Cross output."""

    if not isinstance(commit, FigureCrossCommit):
        raise TypeError("Cross presentation requires FigureCrossCommit")
    if commit.intent is ViewIntent.HISTOGRAM:
        if commit.histogram_bin is None:
            raise ValueError("Histogram Cross presentation requires its bin")
        lower, upper, include_upper = commit.histogram_bin
        close = "]" if include_upper else ")"
        description = (
            f"Cross-selected histogram bin count for [{lower}, {upper}{close}."
        )
    else:
        description = f"Cross-selected {commit.intent.value.lower()} value."
    return FigureOutputPresentation(
        CROSS_DATA_OUTPUT,
        FIGURE_CROSS_DATA_OUTPUT_CONTRACT_ID,
        "Cross data",
        "Cross data",
        description,
    )


def fit_parameter_output_presentation(
    parameter_name: str,
    model_id: str,
) -> FigureOutputPresentation:
    """Return the sole public presentation of one Fit parameter output."""

    parameter = canonical_text(parameter_name, "Fit parameter name")
    model = canonical_text(model_id, "Fit model id")
    return FigureOutputPresentation(
        f"{FIT_OUTPUT_PREFIX}{parameter}",
        FIGURE_FIT_PARAMETER_OUTPUT_CONTRACT_ID,
        parameter,
        parameter,
        f"Figure Fit parameter {parameter} from model {model}.",
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


def _materialize_committed_snapshot(
    source: OwnedSnapshot,
    transform: CommittedTransform,
    *,
    output_name: str,
    semantic_identity: Mapping[str, object],
) -> OwnedSnapshot:
    transformed = apply_transform(source, transform)
    output_ref = figure_output_revision_ref(
        output_name,
        source.ref,
        transformed.schema,
        semantic_identity,
    )
    return OwnedSnapshot(
        output_ref,
        DataBlock(
            output_ref.block_id,
            source.block.revision,
            transformed.values,
            transformed.validity,
            transformed.schema,
        ),
    )


def figure_derived_signal(
    output_name: str,
    snapshot: OwnedSnapshot,
    source: FigureSource,
    *,
    preserve_source_coverage: bool,
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
        preserve_source_coverage=preserve_source_coverage,
    )


def materialize_area_outputs(
    source: FigureSource,
    commit: FigureAreaCommit,
) -> dict[str, FigureDerivedSignal]:
    """Return selected data plus one typed bound vector per selected axis."""

    if not isinstance(source, FigureSource):
        raise TypeError("Area source must be FigureSource")
    if not isinstance(commit, FigureAreaCommit):
        raise TypeError("Area output requires FigureAreaCommit")
    snapshot = source.snapshot
    if not source_identity_matches_snapshot(commit.source_identity, snapshot):
        raise ValueError("Area commit belongs to another source generation")
    authority = commit.authority
    if isinstance(authority, HistogramValueRangeSelection):
        snapshot = source.snapshot
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("Histogram Area source does not own a dataset snapshot")
        context_identity = {
            "kind": "histogram-area-context",
            "source_transform": committed_transform_to_tree(
                authority.source_transform
            ),
        }
        working = _materialize_committed_snapshot(
            snapshot,
            authority.source_transform,
            output_name=AREA_DATA_OUTPUT,
            semantic_identity=context_identity,
        )
        values = working.block.values
        if values.dtype.kind not in "biuf":
            raise TypeError("Histogram Area requires real numeric source values")
        accepted = (
            np.isfinite(values)
            & (values >= authority.lower)
            & (values <= authority.upper)
        )
        semantic_identity = {
            "histogram_value_range": [authority.lower, authority.upper],
            "source_transform": committed_transform_to_tree(
                authority.source_transform
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
            )
        }
    site_map = source.site_map
    if site_map is not None:
        if not isinstance(authority, Selection):
            raise TypeError("SiteMap Area requires a typed Selection")
        return dict(site_map.materialize_area_outputs(source, authority))
    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("Area source signal does not own a dataset snapshot")
    if not isinstance(authority, CommittedTransform):
        raise TypeError("Dataset Area requires a committed transform")
    semantic_identity = {
        "kind": "area-data",
        "source_transform": committed_transform_to_tree(authority),
    }
    selected = _materialize_committed_snapshot(
        snapshot,
        authority,
        output_name=AREA_DATA_OUTPUT,
        semantic_identity=semantic_identity,
    )
    return {
        AREA_DATA_OUTPUT: figure_derived_signal(
            AREA_DATA_OUTPUT,
            selected,
            source,
            preserve_source_coverage=True,
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

    transformed = apply_transform(snapshot, commit.transform)
    values = np.asarray(transformed.values)
    validity = np.asarray(
        expand_dataset_validity(transformed.validity, transformed.schema),
        dtype=np.bool_,
    )
    if commit.intent in {ViewIntent.IMAGE, ViewIntent.CURVE}:
        if values.shape != (1, 1, 1) or validity.shape != (1, 1, 1):
            raise RuntimeError("Cross transform did not resolve one scalar value")
        scalar_values = values.reshape(1)
        scalar_validity = validity.reshape(1)
        unit = transformed.schema.cell_schema.value_unit
    elif commit.intent is ViewIntent.HISTOGRAM:
        if commit.histogram_bin is None:
            raise RuntimeError("Histogram Cross lost its committed bin")
        lower, upper, include_upper = commit.histogram_bin
        samples = values[validity]
        selected = (samples >= lower) & (
            (samples <= upper) if include_upper else (samples < upper)
        )
        scalar_values = np.asarray(
            (int(np.count_nonzero(selected)),),
            dtype=np.dtype("<i8"),
        )
        scalar_validity = np.ones((1,), dtype=np.bool_)
        unit = None
    else:  # FigureCrossCommit closes this; retain the materializer boundary.
        raise RuntimeError("Cross commit has no data-value semantics")

    semantic_identity = {
        "kind": "cross-data",
        "source_transform": committed_transform_to_tree(commit.transform),
        "intent": commit.intent.value,
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
    return {
        CROSS_DATA_OUTPUT: figure_derived_signal(
            CROSS_DATA_OUTPUT,
            output_snapshot,
            source,
            preserve_source_coverage=False,
            derivation_digest=figure_derivation_digest(
                CROSS_DATA_OUTPUT,
                output_snapshot,
                semantic_identity,
            ),
        )
    }


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
        presentation = fit_parameter_output_presentation(
            parameter_name,
            result.spec.model_id,
        )
        output_name = presentation.name
        output[output_name] = FigureDerivedSignal(
            snapshot=fit_snapshot,
            source_ref=result.source_ref,
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
