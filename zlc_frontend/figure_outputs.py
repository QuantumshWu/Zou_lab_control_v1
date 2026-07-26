"""Headless materialisation of Figure-owned selector and Fit signals.

The Figure owns Area and Cross intent; a Measurement continues to publish only
its physical dataset.  This module turns an accepted Figure gesture into typed
datasets without depending on any Workbench shell.  It deliberately contains
no Qt, renderer, runtime node, buffering, or producer-control code.

Area data is evaluated by :mod:`zlc_data.transform`, so a selection keeps every
axis it did not explicitly name and carries component validity with it.  Cross
publishes only the two coordinates of a locked click.  Fit publishes one typed
``fit.<parameter>`` dataset per parameter while preserving its named batch
layout and failed-cell validity.  Derived values retain the source join digest
while receiving stable Figure/output dataset identities; a composition root may
attach its own run/epoch routing metadata afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from zlc_storage import canonical_digest, canonical_text, sha256_text

from zlc_data import (
    AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
    COMPONENT,
    SCALAR_AXIS,
    AxisId,
    AxisSpec,
    BlockId,
    CoordinateRangeSelection,
    DataTransformSpec,
    DatasetRevisionRef,
    IndexRangeSelection,
    IndexSelection,
    OwnedSnapshot,
    Selection,
    StreamGenerationId,
    dataset_revision_ref_to_tree,
    selection_to_tree,
    fit_spec_to_tree,
    materialize_component_dataset,
    materialize_dataset_acceptance_mask,
    materialize_dataset_selection,
    materialize_fit_parameter_snapshots,
    materialize_numeric_dataset,
    projected_dataset_output_contract_id,
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
CROSS_X_OUTPUT = "cross.x"
CROSS_Y_OUTPUT = "cross.y"
FIT_OUTPUT_PREFIX = "fit."
FIGURE_AREA_RANGE_OUTPUT_CONTRACT_ID = "zlc_frontend.figure.area-range"
FIGURE_CROSS_COORDINATE_OUTPUT_CONTRACT_ID = (
    "zlc_frontend.figure.cross-coordinate"
)
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
    if output_name.startswith("area.range."):
        return FIGURE_AREA_RANGE_OUTPUT_CONTRACT_ID
    if output_name in {CROSS_X_OUTPUT, CROSS_Y_OUTPUT}:
        return FIGURE_CROSS_COORDINATE_OUTPUT_CONTRACT_ID
    if output_name.startswith(FIT_OUTPUT_PREFIX) and output_name != FIT_OUTPUT_PREFIX:
        return FIGURE_FIT_PARAMETER_OUTPUT_CONTRACT_ID
    raise ValueError(f"unknown Figure output {output_name!r}")

__all__ = [
    "AREA_DATA_OUTPUT",
    "CROSS_X_OUTPUT",
    "CROSS_Y_OUTPUT",
    "FIT_OUTPUT_PREFIX",
    "FIGURE_AREA_RANGE_OUTPUT_CONTRACT_ID",
    "FIGURE_CROSS_COORDINATE_OUTPUT_CONTRACT_ID",
    "FIGURE_FIT_PARAMETER_OUTPUT_CONTRACT_ID",
    "FigureDerivedSignal",
    "FigureAreaCommit",
    "FigureCrossCommit",
    "FigureOutputFront",
    "FigureOutputPresentation",
    "FigureOutputRequest",
    "FigureOutputSession",
    "HistogramValueRangeSelection",
    "SelectorAxisMetadata",
    "area_range_output_name",
    "area_data_output_presentation",
    "figure_derived_signal",
    "figure_derivation_digest",
    "figure_output_contract_id",
    "figure_output_revision_ref",
    "materialize_area_outputs",
    "materialize_area_range_output",
    "materialize_area_snapshot",
    "materialize_cross_outputs",
    "materialize_fit_outputs",
    "materialize_numeric_snapshot",
    "selector_axes_for_payload",
    "source_identity_matches_snapshot",
]


@dataclass(frozen=True)
class SelectorAxisMetadata:
    """The declared axis identity needed to publish one locked coordinate."""

    axis_id: AxisId
    name: str
    unit: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        name = str(self.name).strip()
        if not name:
            raise ValueError("selector axis name must not be empty")
        object.__setattr__(self, "name", name)
        if self.unit is not None:
            unit = str(self.unit).strip()
            if not unit:
                raise ValueError("selector axis unit must not be empty")
            object.__setattr__(self, "unit", unit)


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
    """One completed locked Cross intent bound to a producer generation."""

    source_identity: SourceIdentity
    point: tuple[float, float]
    axes: tuple[SelectorAxisMetadata, SelectorAxisMetadata]

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("Cross source_identity must be SourceIdentity")
        point = tuple(float(value) for value in self.point)
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise ValueError("Cross point must contain two finite coordinates")
        axes = tuple(self.axes)
        if len(axes) != 2 or any(
            not isinstance(axis, SelectorAxisMetadata) for axis in axes
        ):
            raise TypeError("Cross axes must contain x and y metadata")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "axes", axes)


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
            "histogram_value_range": [selection.lower, selection.upper]
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
        commit.point,
        tuple((axis.axis_id, axis.name, axis.unit) for axis in commit.axes),
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
                        request.cross.point,
                        request.cross.axes,
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

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("histogram Area bounds must be finite")
        if lower > upper:
            raise ValueError("histogram Area lower bound exceeds upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def selector_axes_for_payload(
    payload: object,
) -> tuple[SelectorAxisMetadata, SelectorAxisMetadata]:
    """Return the exact painted x/y coordinate semantics for Cross output.

    Synthetic display coordinates have stable Figure-owned identities.  A
    Workbench panel route never enters an AxisId, so two panels displaying the
    same source cannot manufacture different physical data identities.
    """

    if isinstance(payload, SiteMapPanelPayload):
        payload = payload.background
    if isinstance(payload, ImagePanelPayload):
        return (
            SelectorAxisMetadata(
                payload.viewport.x_axis.axis_id,
                payload.viewport.x_axis.name,
                payload.viewport.x_axis.unit,
            ),
            SelectorAxisMetadata(
                payload.viewport.y_axis.axis_id,
                payload.viewport.y_axis.name,
                payload.viewport.y_axis.unit,
            ),
        )
    if isinstance(payload, CurvePanelPayload):
        return (
            SelectorAxisMetadata(
                payload.viewport.x_axis.axis_id,
                payload.viewport.x_axis.name,
                payload.viewport.x_axis.unit,
            ),
            SelectorAxisMetadata(
                AxisId("figure-value"),
                "value",
                payload.value_unit,
            ),
        )
    if isinstance(payload, HistogramPanelPayload):
        return (
            SelectorAxisMetadata(
                AxisId("histogram-value"),
                "value",
                payload.series[0].data.value_unit,
            ),
            SelectorAxisMetadata(
                AxisId("histogram-count"),
                "count",
                None,
            ),
        )
    raise TypeError("painted payload does not expose Cross coordinates")


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


def area_range_output_name(axis_id: AxisId) -> str:
    if not isinstance(axis_id, AxisId):
        raise TypeError("axis_id must be AxisId")
    return f"area.range.{axis_id.value}"


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


def materialize_numeric_snapshot(
    output_name: str,
    source_ref: DatasetRevisionRef,
    values: object,
    *,
    unit: str | None,
    data_axes: tuple[AxisSpec, ...] = (SCALAR_AXIS,),
    semantic_identity: Mapping[str, object],
) -> OwnedSnapshot:
    """Build a typed scalar/vector selector dataset tied to one source revision."""

    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("selector source_ref must be DatasetRevisionRef")
    axes = tuple(data_axes)
    output = _output_name(output_name)
    return materialize_numeric_dataset(
        source_ref,
        values,
        data_axes=axes,
        unit=unit,
        reference_for=lambda schema: figure_output_revision_ref(
            output,
            source_ref,
            schema,
            semantic_identity,
        ),
    )


def _source_axis(source: OwnedSnapshot, axis_id: AxisId) -> AxisSpec:
    schema = source.block.schema
    for axis in (
        schema.repeat_axis,
        *schema.point_axes,
        *schema.cell_schema.data_axes,
    ):
        if axis.axis_id == axis_id:
            return axis
    raise KeyError(f"selector axis {axis_id} is absent from source schema")


def _real_coordinate(value: object) -> float | None:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, bool) or not isinstance(scalar, Real):
        return None
    numeric = float(scalar)
    return numeric if math.isfinite(numeric) else None


def _term_bounds(
    source: OwnedSnapshot,
    term: CoordinateRangeSelection | IndexRangeSelection | IndexSelection,
) -> tuple[tuple[float, ...], tuple[str, ...], str | None]:
    axis = _source_axis(source, term.axis_id)
    if isinstance(term, CoordinateRangeSelection):
        return (float(term.lower), float(term.upper)), ("lower", "upper"), axis.unit
    if isinstance(term, IndexRangeSelection):
        lower = _real_coordinate(axis.coordinate_at(term.start))
        upper = _real_coordinate(axis.coordinate_at(term.stop - 1))
        if lower is not None and upper is not None:
            return (lower, upper), ("lower", "upper"), axis.unit
        return (float(term.start), float(term.stop)), ("start", "stop"), None
    coordinate = _real_coordinate(axis.coordinate_at(term.index))
    if coordinate is not None:
        return (coordinate,), ("coordinate",), axis.unit
    return (float(term.index),), ("index",), None


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


def materialize_area_range_output(
    source: FigureSource,
    source_ref: DatasetRevisionRef,
    axis: AxisSpec,
    values: tuple[float, ...],
    labels: tuple[str, ...],
    semantic_identity: Mapping[str, object],
    *,
    unit: str | None,
    derivation_digest: str | None = None,
) -> tuple[str, FigureDerivedSignal]:
    """Build the one shared typed representation of an Area axis bound."""

    output_name = area_range_output_name(axis.axis_id)
    identity = canonical_digest(
        {
            "owner": "zlc_frontend.figure-area-bound-axis",
            "output_name": output_name,
            "source_block_id": source_ref.block_id.value,
            "semantic_identity": dict(semantic_identity),
        }
    )
    bound_axis = AxisSpec(
        AxisId(f"figure-output-{identity[:24]}-bound"),
        f"{axis.name} bound",
        COMPONENT,
        len(values),
        labels,
    )
    bound = materialize_numeric_snapshot(
        output_name,
        source_ref,
        np.asarray(values, dtype="<f8"),
        unit=unit,
        data_axes=(bound_axis,),
        semantic_identity=semantic_identity,
    )
    return output_name, figure_derived_signal(
        output_name,
        bound,
        source,
        preserve_source_coverage=False,
        presentation=FigureOutputPresentation(
            output_name,
            FIGURE_AREA_RANGE_OUTPUT_CONTRACT_ID,
            f"{axis.name} range",
            f"{axis.name} range",
            f"Committed Figure Area bounds on {axis.name}.",
        ),
        derivation_digest=derivation_digest,
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
        schema = snapshot.block.schema
        values = snapshot.block.values
        if values.dtype.kind not in "biuf":
            raise TypeError("Histogram Area requires real numeric source values")
        accepted = (
            np.isfinite(values)
            & (values >= selection.lower)
            & (values <= selection.upper)
        )
        semantic_identity = {
            "histogram_value_range": [selection.lower, selection.upper],
        }
        selected = materialize_dataset_acceptance_mask(
            snapshot,
            accepted,
            reference_for=lambda output_schema: figure_output_revision_ref(
                AREA_DATA_OUTPUT,
                snapshot.ref,
                output_schema,
                semantic_identity,
            ),
        )
        outputs = {
            AREA_DATA_OUTPUT: figure_derived_signal(
                AREA_DATA_OUTPUT,
                selected,
                source,
                preserve_source_coverage=True,
                presentation=area_data_output_presentation(
                    source.source_contract_id,
                ),
            )
        }
        value_axis = AxisSpec(
            AxisId(f"histogram-value-{schema.fingerprint[:24]}"),
            "value",
            COMPONENT,
            1,
            ("value",),
            unit=schema.cell_schema.value_unit,
        )
        range_key, range_value = materialize_area_range_output(
            source,
            snapshot.ref,
            value_axis,
            (selection.lower, selection.upper),
            ("lower", "upper"),
            semantic_identity,
            unit=schema.cell_schema.value_unit,
        )
        outputs[range_key] = range_value
        return outputs
    site_map = source.site_map
    if site_map is not None:
        if not isinstance(selection, Selection):
            raise TypeError("SiteMap Area requires a typed Selection")
        return dict(site_map.materialize_area_outputs(source, selection))
    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("Area source signal does not own a dataset snapshot")
    selected = materialize_area_snapshot(snapshot, selection)
    output: dict[str, FigureDerivedSignal] = {}
    output[AREA_DATA_OUTPUT] = figure_derived_signal(
        AREA_DATA_OUTPUT,
        selected,
        source,
        preserve_source_coverage=True,
        presentation=area_data_output_presentation(
            source.source_contract_id,
        ),
        source_transform=DataTransformSpec((selection,)),
    )
    selection_tree = selection_to_tree(selection)
    for term in selection.terms:
        axis = _source_axis(snapshot, term.axis_id)
        values, labels, unit = _term_bounds(snapshot, term)
        key, value = materialize_area_range_output(
            source,
            snapshot.ref,
            axis,
            values,
            labels,
            {
                "selection": selection_tree,
                "axis_id": term.axis_id.value,
            },
            unit=unit,
        )
        output[key] = value
    return output


def materialize_cross_outputs(
    source: FigureSource,
    point: tuple[float, float],
    axes: tuple[SelectorAxisMetadata, SelectorAxisMetadata],
) -> dict[str, FigureDerivedSignal]:
    """Publish a locked Cross point; mouse movement is intentionally irrelevant."""

    if not isinstance(source, FigureSource):
        raise TypeError("Cross source must be FigureSource")
    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("Cross source signal does not own a dataset snapshot")
    if len(tuple(point)) != 2:
        raise ValueError("Cross point must contain x and y coordinates")
    metadata = tuple(axes)
    if len(metadata) != 2 or any(
        not isinstance(axis, SelectorAxisMetadata) for axis in metadata
    ):
        raise TypeError("Cross axes must contain x and y SelectorAxisMetadata")
    coordinates = tuple(float(value) for value in point)
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("Cross coordinates must be finite")

    result: dict[str, FigureDerivedSignal] = {}
    for output_name, value, axis in zip(
        (CROSS_X_OUTPUT, CROSS_Y_OUTPUT),
        coordinates,
        metadata,
        strict=True,
    ):
        coordinate = materialize_numeric_snapshot(
            output_name,
            snapshot.ref,
            # ``SCALAR_AXIS`` is still one declared component axis.  Keep the
            # canonical physical carrier ``(R=1, P=1, data=1)`` instead of
            # passing a zero-rank ndarray and asking the data kernel to guess
            # whether an absent trailing dimension meant scalar data.
            np.asarray((value,), dtype="<f8"),
            unit=axis.unit,
            semantic_identity={
                "kind": "cross-coordinate",
                "axis_id": axis.axis_id.value,
                "value": value,
            },
        )
        result[output_name] = figure_derived_signal(
            output_name,
            coordinate,
            source,
            preserve_source_coverage=False,
            presentation=FigureOutputPresentation(
                output_name,
                FIGURE_CROSS_COORDINATE_OUTPUT_CONTRACT_ID,
                axis.name,
                axis.name,
                f"Locked Figure Cross coordinate on {axis.name}.",
            ),
        )
    return result


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
