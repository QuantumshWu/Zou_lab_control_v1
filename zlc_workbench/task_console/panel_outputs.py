"""Headless materialisation of Figure-owned selector and Fit signals.

The Figure owns Area and Cross intent; a Measurement continues to publish only
its physical dataset.  This module is the one TaskConsole seam that turns an
accepted Figure gesture into typed datasets.  It deliberately contains no Qt,
renderer, hover, buffering, or producer-control code.

Area data is evaluated by :mod:`zlc_data.transform`, so a selection keeps every
axis it did not explicitly name and carries component validity with it.  Cross
publishes only the two coordinates of a locked click.  Fit publishes one typed
``fit.<parameter>`` dataset per parameter while preserving its named batch
layout and failed-cell validity.  All inherit the exact source run/join lineage
while receiving stable panel/output dataset identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

import numpy as np
from zlc_storage import canonical_digest

from zlc_data import (
    COMPONENT,
    REPEAT,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    CoordinateRangeSelection,
    DataBlock,
    DatasetRevisionRef,
    DatasetSchema,
    IndexRangeSelection,
    IndexSelection,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    dataset_revision_ref_to_tree,
    expand_dataset_validity,
    selection_to_tree,
    fit_spec_to_tree,
)
from zlc_data.console_records import panel_signal_key

from .data_plane import ConsoleSignalValue
from .dataset_projection import materialize_dataset_selection


AREA_DATA_OUTPUT = "area.data"
CROSS_X_OUTPUT = "cross.x"
CROSS_Y_OUTPUT = "cross.y"
FIT_OUTPUT_PREFIX = "fit."

__all__ = [
    "AREA_DATA_OUTPUT",
    "CROSS_X_OUTPUT",
    "CROSS_Y_OUTPUT",
    "FIT_OUTPUT_PREFIX",
    "FitParameterMetadata",
    "HistogramValueRangeSelection",
    "SelectorAxisMetadata",
    "area_range_output_name",
    "materialize_area_outputs",
    "materialize_area_snapshot",
    "materialize_cross_outputs",
    "materialize_fit_outputs",
    "materialize_numeric_snapshot",
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
    unit: str

    def __post_init__(self) -> None:
        for field in ("model_id", "parameter_name", "unit"):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} must not be empty")
            object.__setattr__(self, field, value)


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


def _panel_identity(panel_id: str, output_name: str) -> tuple[str, str, str]:
    panel = str(panel_id).strip()
    output = str(output_name).strip()
    signal = panel_signal_key(panel, output)
    identity = canonical_digest(
        {
            "owner": "zlc_workbench.task-console.panel-output.v1",
            "panel_id": panel,
            "output_name": output,
        }
    )
    return panel, signal, identity


def area_range_output_name(axis_id: AxisId) -> str:
    if not isinstance(axis_id, AxisId):
        raise TypeError("axis_id must be AxisId")
    return f"area.range.{axis_id.value}"


def _derived_ref(
    panel_id: str,
    output_name: str,
    source_ref: DatasetRevisionRef,
    output_schema: DatasetSchema,
    semantic_identity: Mapping[str, object],
) -> DatasetRevisionRef:
    _panel, _signal, identity = _panel_identity(panel_id, output_name)
    generation = canonical_digest(
        {
            "owner": "zlc_workbench.task-console.panel-output-generation.v1",
            "panel_output_identity": identity,
            "source_block_id": source_ref.block_id.value,
            "source_generation": source_ref.stream_generation.value,
            "output_schema": output_schema.fingerprint,
            "semantic_identity": dict(semantic_identity),
        }
    )
    return DatasetRevisionRef(
        BlockId(f"panel-output-{identity[:24]}"),
        StreamGenerationId(f"panel-output-{generation}"),
        output_schema.fingerprint,
        source_ref.revision,
    )


def materialize_area_snapshot(
    panel_id: str,
    source: OwnedSnapshot,
    selection: Selection,
    *,
    output_name: str = AREA_DATA_OUTPUT,
) -> OwnedSnapshot:
    """Materialise one accepted Area selection without flattening or reducing."""

    return materialize_dataset_selection(
        source,
        selection,
        reference_for=lambda output_schema: _derived_ref(
            panel_id,
            output_name,
            source.ref,
            output_schema,
            {"selection": selection_to_tree(selection)},
        ),
    )


def _numeric_array(values: object, data_axes: tuple[AxisSpec, ...]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise TypeError("selector output values must be real numeric values")
    array = np.asarray(raw, dtype="<f8")
    expected = tuple(axis.size for axis in data_axes)
    if array.shape != expected:
        raise ValueError(
            f"selector output shape {array.shape} does not match axes {expected}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("selector output values must be finite")
    return array


def materialize_numeric_snapshot(
    panel_id: str,
    output_name: str,
    source_ref: DatasetRevisionRef,
    values: object,
    *,
    unit: str | None,
    data_axes: tuple[AxisSpec, ...] = (),
    semantic_identity: Mapping[str, object],
) -> OwnedSnapshot:
    """Build a typed scalar/vector selector dataset tied to one source revision."""

    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("selector source_ref must be DatasetRevisionRef")
    axes = tuple(data_axes)
    if any(not isinstance(axis, AxisSpec) for axis in axes):
        raise TypeError("selector data_axes must contain AxisSpec values")
    array = _numeric_array(values, axes)
    _panel, _signal, identity = _panel_identity(panel_id, output_name)
    value_schema = (
        ValueSchema(
            axes,
            ValidityContract.value(),
            np.dtype("<f8"),
            unit,
        )
        if axes
        else ValueSchema.scalar(np.dtype("<f8"), unit)
    )
    schema = DatasetSchema(
        AxisSpec(
            AxisId(f"panel-output-{identity[:24]}-repeat"),
            "repeat",
            REPEAT,
            1,
            (0,),
        ),
        (),
        PointLayout.rect_c(()),
        value_schema,
    )
    ref = _derived_ref(
        panel_id,
        output_name,
        source_ref,
        schema,
        semantic_identity,
    )
    block = DataBlock(
        ref.block_id,
        ref.revision,
        array.reshape(schema.physical_shape),
        VALID,
        schema,
    )
    return OwnedSnapshot(ref, block)


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


def _signal_value(
    panel_id: str,
    output_name: str,
    snapshot: OwnedSnapshot,
    source: ConsoleSignalValue,
    *,
    coverage: object | None,
    presentation: object | None = None,
    join_digest: str | None = None,
) -> ConsoleSignalValue:
    key = panel_signal_key(panel_id, output_name)
    return ConsoleSignalValue(
        name=key,
        source=str(panel_id).strip(),
        snapshot=snapshot,
        coverage=coverage,
        run_id=source.run_id,
        epoch_id=source.epoch_id,
        join_digest=source.join_digest if join_digest is None else join_digest,
        presentation=presentation,
    )


def _area_range_output(
    panel_id: str,
    source: ConsoleSignalValue,
    source_ref: DatasetRevisionRef,
    axis: AxisSpec,
    values: tuple[float, ...],
    labels: tuple[str, ...],
    semantic_identity: Mapping[str, object],
    *,
    unit: str | None,
    join_digest: str | None = None,
) -> tuple[str, ConsoleSignalValue]:
    """Build the one shared typed representation of an Area axis bound."""

    output_name = area_range_output_name(axis.axis_id)
    _panel, _signal, identity = _panel_identity(panel_id, output_name)
    bound_axis = AxisSpec(
        AxisId(f"panel-output-{identity[:24]}-bound"),
        f"{axis.name} bound",
        COMPONENT,
        len(values),
        labels,
    )
    bound = materialize_numeric_snapshot(
        panel_id,
        output_name,
        source_ref,
        np.asarray(values, dtype="<f8"),
        unit=unit,
        data_axes=(bound_axis,),
        semantic_identity=semantic_identity,
    )
    key = panel_signal_key(panel_id, output_name)
    return key, _signal_value(
        panel_id,
        output_name,
        bound,
        source,
        coverage=None,
        presentation=axis,
        join_digest=join_digest,
    )


def _site_map_input_tree(view) -> dict[str, object]:
    """Canonical two-input lineage for one already-joined SiteMap front."""

    def evaluated_input_tree(value) -> dict[str, object]:
        return {
            "dataset_id": value.dataset_id.value,
            "ref": dataset_revision_ref_to_tree(value.ref),
        }

    return {
        "background": evaluated_input_tree(view.background_input),
        "site_state": evaluated_input_tree(view.site_state_input),
        "calibration_identity": view.calibration_identity,
        "view_identity": view.view_identity,
    }


def _site_map_data_snapshot(
    panel_id: str,
    source_ref: DatasetRevisionRef,
    site_axis: AxisSpec,
    values: np.ndarray,
    validity: np.ndarray,
    *,
    data_axes: tuple[AxisSpec, ...],
    unit: str | None,
    semantic_identity: Mapping[str, object],
) -> OwnedSnapshot:
    """Materialise SiteMap Area data without reducing SITE validity."""

    array = np.asarray(values)
    mask = np.asarray(validity, dtype=np.bool_)
    axes = tuple(data_axes)
    if not axes or axes[0] != site_axis:
        raise ValueError("SiteMap Area data must begin with its selected SITE axis")
    if array.shape != tuple(axis.size for axis in axes):
        raise ValueError("selected SiteMap data differs from its declared axes")
    if mask.shape != (site_axis.size,):
        raise ValueError("selected SiteMap validity must align to SITE")
    output_name = AREA_DATA_OUTPUT
    _panel, _signal, identity = _panel_identity(panel_id, output_name)
    schema = DatasetSchema(
        AxisSpec(
            AxisId(f"panel-output-{identity[:24]}-repeat"),
            "repeat",
            REPEAT,
            1,
            (0,),
        ),
        (),
        PointLayout.rect_c(()),
        ValueSchema(
            axes,
            ValidityContract.components(site_axis.axis_id),
            array.dtype,
            unit,
        ),
    )
    ref = _derived_ref(
        panel_id,
        output_name,
        source_ref,
        schema,
        semantic_identity,
    )
    block = DataBlock(
        ref.block_id,
        ref.revision,
        array.reshape(schema.physical_shape),
        ComponentValidity(
            (site_axis.axis_id,),
            mask.reshape(1, 1, site_axis.size),
        ),
        schema,
    )
    return OwnedSnapshot(ref, block)


def _site_map_area_outputs(
    panel_id: str,
    source: ConsoleSignalValue,
    selection: Selection,
    view,
) -> dict[str, ConsoleSignalValue]:
    """Select SiteMap state by calibrated centres, matching Main's semantics."""

    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("SiteMap Area source does not own a dataset snapshot")
    if snapshot.ref != view.site_state_input.ref:
        raise ValueError("SiteMap Area source differs from its exact site-state input")
    x_axis = view.home_viewport.x_axis
    y_axis = view.home_viewport.y_axis
    terms = {term.axis_id: term for term in selection.terms}
    if set(terms) != {x_axis.axis_id, y_axis.axis_id}:
        raise ValueError("SiteMap Area must select its painted x and y axes")
    x_term = terms[x_axis.axis_id]
    y_term = terms[y_axis.axis_id]
    if not isinstance(x_term, CoordinateRangeSelection) or not isinstance(
        y_term, CoordinateRangeSelection
    ):
        raise TypeError("SiteMap Area requires coordinate-range x and y terms")
    if any(
        term.coordinate_frame != view.coordinate_frame
        for term in (x_term, y_term)
    ):
        raise ValueError("SiteMap Area coordinate frame differs from its sites")
    centers = np.asarray(view.centers_xy, dtype="<f8")
    selected = np.flatnonzero(
        (centers[:, 0] >= float(x_term.lower))
        & (centers[:, 0] <= float(x_term.upper))
        & (centers[:, 1] >= float(y_term.lower))
        & (centers[:, 1] <= float(y_term.upper))
    )
    lineage = _site_map_input_tree(view)
    selection_tree = selection_to_tree(selection)
    join_digest = canonical_digest(
        {
            "owner": "zlc-workbench.task-console.site-map-area.v1",
            "inputs": lineage,
            "selection": selection_tree,
        }
    )
    outputs: dict[str, ConsoleSignalValue] = {}

    for axis, term in ((x_axis, x_term), (y_axis, y_term)):
        key, value = _area_range_output(
            panel_id,
            source,
            snapshot.ref,
            axis,
            (float(term.lower), float(term.upper)),
            ("lower", "upper"),
            {
                "inputs": lineage,
                "selection": selection_tree,
                "axis_id": axis.axis_id.value,
            },
            unit=axis.unit,
            join_digest=join_digest,
        )
        outputs[key] = value

    # An empty box is still a meaningful completed spatial range.  Dataset axes
    # are non-empty by contract, so it publishes only the two bounds rather than
    # inventing a sentinel site or a false valid value.
    if not selected.size:
        return outputs

    selected_indices = tuple(int(index) for index in selected)
    source_site_axis = view.site_axis
    site_axis = AxisSpec(
        source_site_axis.axis_id,
        source_site_axis.name,
        source_site_axis.role,
        len(selected_indices),
        tuple(source_site_axis.coordinate_at(index) for index in selected_indices),
        source_site_axis.unit,
        source_site_axis.coordinate_frame,
    )
    validity = np.asarray(view.site_validity, dtype=np.bool_)[selected]
    occupied = view.site_occupancy
    if occupied is not None:
        values = np.asarray(occupied, dtype=np.bool_)[selected]
        data_axes = (site_axis,)
        unit = None
        quantity = "occupancy"
    else:
        if x_axis.unit != y_axis.unit:
            raise ValueError(
                "calibration SiteMap Area cannot combine x/y coordinates with "
                "different units into one area.data signal"
            )
        _panel, _signal, identity = _panel_identity(panel_id, AREA_DATA_OUTPUT)
        coordinate_axis = AxisSpec(
            AxisId(f"panel-output-{identity[:24]}-coordinate"),
            "coordinate",
            COMPONENT,
            2,
            ("x", "y"),
        )
        values = np.asarray(centers[selected], dtype="<f8")
        data_axes = (site_axis, coordinate_axis)
        unit = x_axis.unit
        quantity = "calibrated-centers"

    result = _site_map_data_snapshot(
        panel_id,
        snapshot.ref,
        site_axis,
        values,
        validity,
        data_axes=data_axes,
        unit=unit,
        semantic_identity={
            "inputs": lineage,
            "selection": selection_tree,
            "quantity": quantity,
        },
    )
    key = panel_signal_key(panel_id, AREA_DATA_OUTPUT)
    outputs[key] = _signal_value(
        panel_id,
        AREA_DATA_OUTPUT,
        result,
        source,
        coverage=None,
        presentation=view,
        join_digest=join_digest,
    )
    return outputs


def materialize_area_outputs(
    panel_id: str,
    source: ConsoleSignalValue,
    selection: Selection | HistogramValueRangeSelection,
) -> dict[str, ConsoleSignalValue]:
    """Return selected data plus one typed bound vector per selected axis."""

    if not isinstance(source, ConsoleSignalValue):
        raise TypeError("Area source must be ConsoleSignalValue")
    if isinstance(selection, HistogramValueRangeSelection):
        snapshot = source.snapshot
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("Histogram Area source does not own a dataset snapshot")
        schema = snapshot.block.schema
        values = snapshot.block.values
        if values.dtype.kind not in "biuf":
            raise TypeError("Histogram Area requires real numeric source values")
        physical_validity = expand_dataset_validity(
            snapshot.block.validity,
            schema,
        )
        selected_validity = (
            np.asarray(physical_validity, dtype=np.bool_)
            & np.isfinite(values)
            & (values >= selection.lower)
            & (values <= selection.upper)
        )
        data_axes = tuple(schema.cell_schema.data_axes)
        if schema.cell_schema.is_scalar:
            validity_contract = ValidityContract.value()
            validity = CellValidity(selected_validity[..., 0])
        else:
            validity_contract = ValidityContract.components(
                *(axis.axis_id for axis in data_axes)
            )
            validity = ComponentValidity(
                tuple(axis.axis_id for axis in data_axes),
                selected_validity,
            )
        output_schema = DatasetSchema(
            schema.repeat_axis,
            schema.point_axes,
            schema.point_layout,
            ValueSchema(
                data_axes,
                validity_contract,
                schema.cell_schema.dtype,
                schema.cell_schema.value_unit,
            ),
        )
        semantic_identity = {
            "histogram_value_range": [selection.lower, selection.upper],
        }
        ref = _derived_ref(
            panel_id,
            AREA_DATA_OUTPUT,
            snapshot.ref,
            output_schema,
            semantic_identity,
        )
        selected = OwnedSnapshot(
            ref,
            DataBlock(
                ref.block_id,
                ref.revision,
                values,
                validity,
                output_schema,
            ),
        )
        data_key = panel_signal_key(panel_id, AREA_DATA_OUTPUT)
        outputs = {
            data_key: _signal_value(
                panel_id,
                AREA_DATA_OUTPUT,
                selected,
                source,
                coverage=source.coverage,
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
        range_key, range_value = _area_range_output(
            panel_id,
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
    from zlc_frontend.site_map_render import (
        CalibrationSiteMapView,
        OccupancyCellView,
        OccupancySummarySiteMapView,
    )

    presentation = source.presentation
    if isinstance(
        presentation,
        (
            OccupancyCellView,
            CalibrationSiteMapView,
            OccupancySummarySiteMapView,
        ),
    ):
        return _site_map_area_outputs(
            panel_id,
            source,
            selection,
            presentation,
        )
    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("Area source signal does not own a dataset snapshot")
    selected = materialize_area_snapshot(panel_id, snapshot, selection)
    output: dict[str, ConsoleSignalValue] = {}
    data_key = panel_signal_key(panel_id, AREA_DATA_OUTPUT)
    output[data_key] = _signal_value(
        panel_id,
        AREA_DATA_OUTPUT,
        selected,
        source,
        coverage=source.coverage,
    )
    selection_tree = selection_to_tree(selection)
    for term in selection.terms:
        axis = _source_axis(snapshot, term.axis_id)
        values, labels, unit = _term_bounds(snapshot, term)
        key, value = _area_range_output(
            panel_id,
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
    panel_id: str,
    source: ConsoleSignalValue,
    point: tuple[float, float],
    axes: tuple[SelectorAxisMetadata, SelectorAxisMetadata],
) -> dict[str, ConsoleSignalValue]:
    """Publish a locked Cross point; mouse movement is intentionally irrelevant."""

    if not isinstance(source, ConsoleSignalValue):
        raise TypeError("Cross source must be ConsoleSignalValue")
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

    result: dict[str, ConsoleSignalValue] = {}
    for output_name, value, axis in zip(
        (CROSS_X_OUTPUT, CROSS_Y_OUTPUT),
        coordinates,
        metadata,
        strict=True,
    ):
        coordinate = materialize_numeric_snapshot(
            panel_id,
            output_name,
            snapshot.ref,
            np.asarray(value, dtype="<f8"),
            unit=axis.unit,
            semantic_identity={
                "kind": "cross-coordinate",
                "axis_id": axis.axis_id.value,
                "value": value,
            },
        )
        key = panel_signal_key(panel_id, output_name)
        result[key] = _signal_value(
            panel_id,
            output_name,
            coordinate,
            source,
            coverage=None,
            presentation=axis,
        )
    return result


def _fit_batch_dataset_layout(result):
    """Factor a FitResultBatch into DatasetSchema's repeat/point layout.

    The fit solver stores only present batch rows.  A parameter signal keeps
    that exact sparse layout and the original named batch axes; it never
    reshapes a storage row table as though it were a dense Cartesian array.
    """

    from zlc_data import FitResultBatch

    if not isinstance(result, FitResultBatch):
        raise TypeError("fit result must be FitResultBatch")
    axes = tuple(result.batch_axis_specs)
    repeat_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == REPEAT
    )
    if len(repeat_positions) > 1:
        raise ValueError("fit result repeats the repeat axis")
    layout = result.batch_layout

    if repeat_positions:
        repeat_position = repeat_positions[0]
        repeat_axis = axes[repeat_position]
        point_axes = tuple(
            axis for index, axis in enumerate(axes) if index != repeat_position
        )
        rows: dict[tuple[int, ...], int] = {}
        by_repeat: list[set[tuple[int, ...]]] = [
            set() for _ in range(repeat_axis.size)
        ]
        for storage_index in range(layout.storage_size):
            multi = layout.multi_index(storage_index)
            repeat_index = multi[repeat_position]
            point_multi = tuple(
                value
                for index, value in enumerate(multi)
                if index != repeat_position
            )
            rows[(repeat_index, *point_multi)] = storage_index
            by_repeat[repeat_index].add(point_multi)
        point_membership = by_repeat[0]
        if any(membership != point_membership for membership in by_repeat[1:]):
            raise ValueError(
                "fit batch has repeat-dependent sparse point membership, which "
                "cannot be represented as one DatasetSchema"
            )
        point_mapping = tuple(sorted(point_membership))
        point_layout = PointLayout.from_mapping(
            tuple(axis.size for axis in point_axes),
            point_mapping,
        )
        order = np.fromiter(
            (
                rows[(repeat_index, *point_layout.multi_index(point_index))]
                for repeat_index in range(repeat_axis.size)
                for point_index in range(point_layout.storage_size)
            ),
            dtype=np.intp,
            count=repeat_axis.size * point_layout.storage_size,
        )
        return repeat_axis, point_axes, point_layout, order

    # A fit with no repeat batch axis is still a one-repeat dataset.  All
    # declared batch axes remain point axes with their exact sparse mapping.
    point_axes = axes
    used_axis_ids = {axis.axis_id.value for axis in point_axes}
    repeat_axis_id = "figure-fit-repeat"
    suffix = 2
    while repeat_axis_id in used_axis_ids:
        repeat_axis_id = f"figure-fit-repeat-{suffix}"
        suffix += 1
    repeat_axis = AxisSpec(
        AxisId(repeat_axis_id),
        "repeat",
        REPEAT,
        1,
        (0,),
    )
    point_mapping = tuple(
        layout.multi_index(storage_index)
        for storage_index in range(layout.storage_size)
    )
    point_layout = PointLayout.from_mapping(
        tuple(axis.size for axis in point_axes),
        point_mapping,
    )
    order = np.arange(layout.storage_size, dtype=np.intp)
    return repeat_axis, point_axes, point_layout, order


def materialize_fit_outputs(
    panel_id: str,
    source: ConsoleSignalValue,
    result,
) -> dict[str, ConsoleSignalValue]:
    """Publish one typed ``fit.<parameter>`` dataset per model parameter.

    Values retain the result's named batch axes and sparse layout.  A failed
    batch cell is represented by CellValidity=False, never by dropping the
    cell, averaging it, or exposing the solver's canonical numeric zero as a
    physically valid parameter.
    """

    from zlc_data import FitBatchStatus, FitResultBatch

    if not isinstance(source, ConsoleSignalValue):
        raise TypeError("Fit source must be ConsoleSignalValue")
    if not isinstance(result, FitResultBatch):
        raise TypeError("Fit result must be FitResultBatch")
    snapshot = source.snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("Fit source signal does not own a dataset snapshot")
    if result.source_ref != snapshot.ref:
        raise ValueError("Fit result belongs to another visible source revision")

    repeat_axis, point_axes, point_layout, order = _fit_batch_dataset_layout(result)
    validity_rows = np.fromiter(
        (status is FitBatchStatus.CONVERGED for status in result.statuses),
        dtype=np.bool_,
        count=len(result.statuses),
    )[order]
    physical_shape = (repeat_axis.size, point_layout.storage_size)
    validity = CellValidity(validity_rows.reshape(physical_shape))
    spec_tree = fit_spec_to_tree(result.spec)
    output: dict[str, ConsoleSignalValue] = {}
    for parameter_index, (parameter, unit) in enumerate(
        zip(result.parameter_definitions, result.parameter_units, strict=True)
    ):
        output_name = f"{FIT_OUTPUT_PREFIX}{parameter.name}"
        _panel, _signal, identity = _panel_identity(panel_id, output_name)
        schema = DatasetSchema(
            repeat_axis,
            point_axes,
            point_layout,
            ValueSchema.scalar(np.dtype("<f8"), unit),
        )
        ref = _derived_ref(
            panel_id,
            output_name,
            result.source_ref,
            schema,
            {
                "kind": "figure-fit-parameter",
                "fit_spec": spec_tree,
                "parameter_name": parameter.name,
            },
        )
        values = np.asarray(
            result.parameter_values[:, parameter_index],
            dtype="<f8",
        )[order].reshape(schema.physical_shape)
        block = DataBlock(
            ref.block_id,
            ref.revision,
            values,
            validity,
            schema,
        )
        fit_snapshot = OwnedSnapshot(ref, block)
        key = panel_signal_key(panel_id, output_name)
        output[key] = ConsoleSignalValue(
            name=key,
            source=str(panel_id).strip(),
            snapshot=fit_snapshot,
            coverage=None,
            run_id=source.run_id,
            epoch_id=source.epoch_id,
            join_digest=canonical_digest(
                {
                    "owner": "zlc_workbench.task-console.figure-fit-output.v1",
                    "source_join_digest": source.join_digest,
                    "fit_spec": spec_tree,
                }
            ),
            presentation=FitParameterMetadata(
                result.spec.model_id,
                parameter.name,
                unit,
            ),
        )
    return output
