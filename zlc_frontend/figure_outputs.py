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
from typing import Mapping

import numpy as np
from zlc_storage import canonical_digest, sha256_text

from zlc_data import (
    AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID,
    COMPONENT,
    SCALAR_AXIS,
    AxisId,
    AxisSpec,
    BlockId,
    CoordinateRangeSelection,
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
    "FigureOutputSource",
    "FitParameterMetadata",
    "HistogramValueRangeSelection",
    "SelectorAxisMetadata",
    "area_range_output_name",
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


@dataclass(frozen=True)
class FitParameterMetadata:
    """Presentation label for one typed Figure-fit parameter signal."""

    model_id: str
    parameter_name: str

    def __post_init__(self) -> None:
        for field in ("model_id", "parameter_name"):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} must not be empty")
            object.__setattr__(self, field, value)


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
class FigureOutputSource:
    """Exact immutable input accepted by a Figure output operation."""

    snapshot: OwnedSnapshot
    site_map: SiteMapPresentation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("Figure signal source must be OwnedSnapshot")
        if self.site_map is not None and not isinstance(
            self.site_map,
            SiteMapPresentation,
        ):
            raise TypeError(
                "Figure output source site_map is not a SiteMapPresentation"
            )


@dataclass(frozen=True, slots=True)
class FigureDerivedSignal:
    """One headless Figure-derived dataset plus derivation identity.

    Run identity and routing names intentionally do not appear here.  Those are
    application-shell concerns; the Figure owns only the immutable snapshot,
    exact derivation, optional display metadata, and whether source coverage
    remains meaningful for the derived value.
    """

    snapshot: OwnedSnapshot
    source_ref: DatasetRevisionRef
    derivation_digest: str
    preserve_source_coverage: bool = False
    metadata: SelectorAxisMetadata | FitParameterMetadata | None = None

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
        if self.metadata is not None and not isinstance(
            self.metadata,
            (SelectorAxisMetadata, FitParameterMetadata),
        ):
            raise TypeError("Figure output metadata is not supported")


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
    source: FigureOutputSource,
    *,
    preserve_source_coverage: bool,
    metadata: SelectorAxisMetadata | FitParameterMetadata | None = None,
    derivation_digest: str | None = None,
) -> FigureDerivedSignal:
    if not isinstance(source, FigureOutputSource):
        raise TypeError("Figure output source must be FigureOutputSource")
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
        metadata=metadata,
    )


def materialize_area_range_output(
    source: FigureOutputSource,
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
        metadata=SelectorAxisMetadata(axis.axis_id, axis.name, axis.unit),
        derivation_digest=derivation_digest,
    )




def materialize_area_outputs(
    source: FigureOutputSource,
    selection: Selection | HistogramValueRangeSelection,
) -> dict[str, FigureDerivedSignal]:
    """Return selected data plus one typed bound vector per selected axis."""

    if not isinstance(source, FigureOutputSource):
        raise TypeError("Area source must be FigureOutputSource")
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
    source: FigureOutputSource,
    point: tuple[float, float],
    axes: tuple[SelectorAxisMetadata, SelectorAxisMetadata],
) -> dict[str, FigureDerivedSignal]:
    """Publish a locked Cross point; mouse movement is intentionally irrelevant."""

    if not isinstance(source, FigureOutputSource):
        raise TypeError("Cross source must be FigureOutputSource")
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
            metadata=axis,
        )
    return result


def materialize_fit_outputs(
    source: FigureOutputSource,
    result,
) -> dict[str, FigureDerivedSignal]:
    """Publish one typed ``fit.<parameter>`` dataset per model parameter.

    Values retain the result's named batch axes and sparse layout.  A failed
    batch cell is represented by CellValidity=False, never by dropping the
    cell, averaging it, or exposing the solver's canonical numeric zero as a
    physically valid parameter.
    """

    from zlc_data import FitResultBatch

    if not isinstance(source, FigureOutputSource):
        raise TypeError("Fit source must be FigureOutputSource")
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
            preserve_source_coverage=False,
            derivation_digest=canonical_digest(
                {
                    "owner": "zlc_frontend.figure-fit-output",
                    "source_ref": dataset_revision_ref_to_tree(result.source_ref),
                    "fit_spec": spec_tree,
                    "parameter_name": parameter_name,
                }
            ),
            metadata=FitParameterMetadata(
                result.spec.model_id,
                parameter_name,
            ),
        )
    return output
