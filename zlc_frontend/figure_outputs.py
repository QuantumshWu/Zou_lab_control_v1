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
from types import MappingProxyType
from typing import Mapping

import numpy as np
from zlc_storage import canonical_digest, canonical_text, sha256_text

from zlc_data import (
    AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
    AxisSourceRef,
    AxisSpec,
    BlockId,
    CommittedTransform,
    DataTransformSpec,
    DataBlock,
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
    committed_transform_to_tree,
    dataset_revision_ref_to_tree,
    expand_dataset_validity,
    selection_to_tree,
    fit_spec_to_tree,
    materialize_dataset_acceptance_mask,
    materialize_fit_parameter_snapshots,
    materialize_scalar_dataset,
    projected_dataset_output_contract_id,
    resolve_selection_indices,
)
from .curve_display import numeric_curve_coordinates
from .data_figure import DataFigure
from .figure import (
    AxisViewRole,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    ViewIntent,
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
    authority = commit.authority
    if isinstance(authority, Selection):
        authority_identity: object = selection_to_tree(authority)
    elif isinstance(authority, CommittedTransform):
        authority_identity = committed_transform_to_tree(authority)
    else:
        authority_identity = {
            "histogram_value_range": [authority.lower, authority.upper],
            "source_transform": committed_transform_to_tree(
                authority.source_transform
            ),
        }
    site_map = request.source.site_map
    return (
        request.source.snapshot.ref,
        request.source.source_contract_id,
        None if site_map is None else site_map.view_identity,
        canonical_digest(authority_identity),
    )


def _cross_dependency(request: FigureOutputRequest) -> tuple[object, ...] | None:
    commit = request.cross
    if commit is None:
        return None
    return (
        request.source.snapshot.ref,
        canonical_digest(
            {
                "transform": committed_transform_to_tree(commit.transform),
                "intent": commit.intent.value,
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
                        request.area,
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
        reductions = tuple(
            binding
            for binding in view.source_bindings
            if binding.role is AxisViewRole.REDUCED
        )
        if reductions:
            methods = {binding.reduction.method.value for binding in reductions}
            if len(methods) != 1:
                raise ValueError("Figure reductions do not share one method")
            method = {
                "MEAN": ReductionMethod.MEAN,
                "SUM": ReductionMethod.SUM,
            }[methods.pop()]
            operations.append(
                ReductionSpec(
                    tuple(binding.source for binding in reductions),
                    method,
                    missing_policy=MissingPolicy.OMIT_MISSING,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                )
            )
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


def _replayable_association_spec(
    source: OwnedSnapshot,
    transform: CommittedTransform,
) -> DataTransformSpec | None:
    """Project only transforms the current SignalPlane can replay exactly.

    M1 authority always keeps exact point rows in ``CommittedTransform``.
    The existing association channel still accepts only operations, so a
    point-subset commit must not claim a weaker replayable transform. M2 will
    replace that channel at the transaction owner rather than adding a second
    row-selection payload here.
    """

    full_rows = tuple(range(source.block.schema.point_table.row_count))
    if transform.exact_point_ordinals != full_rows or not transform.spec.operations:
        return None
    return transform.spec


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
            presentation=area_data_output_presentation(
                source.source_contract_id,
            ),
            source_transform=_replayable_association_spec(snapshot, authority),
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
        description = f"Cross-selected {commit.intent.value.lower()} value."
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
        close = "]" if include_upper else ")"
        description = (
            f"Cross-selected histogram bin count for [{lower}, {upper}{close}."
        )
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
    source_transform = (
        None
        if commit.intent is ViewIntent.HISTOGRAM
        else _replayable_association_spec(snapshot, commit.transform)
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
