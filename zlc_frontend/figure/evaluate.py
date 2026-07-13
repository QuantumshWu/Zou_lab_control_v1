"""Headless ViewSpec evaluation over immutable zlc_data snapshots."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
import math
from numbers import Integral, Real
import time
from typing import Any, Callable, Sequence

import numpy as np

from zlc_data import (
    AxisId,
    AxisLayout,
    AxisSpec,
    CellValidity,
    ComponentValidity,
    DataBlock,
    Invalid,
    OwnedSnapshot,
    Valid,
)
from zlc_data.numeric import (
    canonical_mean_dtype,
    canonical_sum_dtype,
    checked_numeric_sum,
)

from .contract import dataset_axes, display_axis_indices, validate_view_spec
from .model import (
    AxisAddress,
    AxisResolution,
    AxisViewRole,
    DatasetId,
    DisplayReductionMethod,
    EvaluatedAxis,
    EvaluatedCell,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedInput,
    EvaluatedLayer,
    EvaluatedMeter,
    EvaluatedSeries,
    FigureDocument,
    FixedIndex,
    LatestNonempty,
    ReductionResolution,
    SampleCoordinates,
    ViewIntent,
)


class FigureEvaluationError(ValueError):
    """The document is well-typed but cannot be evaluated on this snapshot."""


class FigureEvaluationCancelled(FigureEvaluationError):
    """Evaluation was cooperatively cancelled at a bounded checkpoint."""


class FigureEvaluationDeadlineExceeded(FigureEvaluationError):
    """Evaluation crossed its caller-provided monotonic deadline."""


class FigureEvaluationLimitExceeded(FigureEvaluationError):
    """A document exceeds the configured finite evaluation budget."""


@dataclass(frozen=True)
class FigureEvaluationPolicy:
    """Finite headless evaluation budgets; this is not an execution engine."""

    max_layers: int = 32
    max_cells: int = 256
    max_series: int = 2048
    max_output_elements: int = 4_000_000
    max_histogram_samples: int = 100_000
    max_materialized_nbytes: int = 256 * 1024 * 1024
    max_physical_rows: int = 500_000
    max_reduction_contributions: int = 32_000_000

    def __post_init__(self) -> None:
        for field in (
            "max_layers",
            "max_cells",
            "max_series",
            "max_output_elements",
            "max_histogram_samples",
            "max_materialized_nbytes",
            "max_physical_rows",
            "max_reduction_contributions",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
            object.__setattr__(self, field, int(value))


@dataclass
class _EvaluationGuard:
    policy: FigureEvaluationPolicy
    cancel_requested: Callable[[], bool] | None
    monotonic_deadline: float | None
    cells: int = 0
    series: int = 0
    output_elements: int = 0
    histogram_samples: int = 0
    materialized_nbytes: int = 0
    physical_rows: int = 0
    reduction_contributions: int = 0

    def check(self) -> None:
        if self.cancel_requested is not None and self.cancel_requested():
            raise FigureEvaluationCancelled("figure evaluation was cancelled")
        if (
            self.monotonic_deadline is not None
            and time.monotonic() >= self.monotonic_deadline
        ):
            raise FigureEvaluationDeadlineExceeded("figure evaluation deadline exceeded")

    def reserve_layer(
        self,
        *,
        cells: int,
        series: int,
        output_elements: int,
        histogram_samples: int,
        materialized_nbytes: int,
        physical_rows: int,
        reduction_contributions: int,
    ) -> None:
        additions = {
            "physical_rows": physical_rows,
            "reduction_contributions": reduction_contributions,
            "materialized_nbytes": materialized_nbytes,
            "cells": cells,
            "series": series,
            "output_elements": output_elements,
            "histogram_samples": histogram_samples,
        }
        limits = {
            "physical_rows": self.policy.max_physical_rows,
            "reduction_contributions": self.policy.max_reduction_contributions,
            "materialized_nbytes": self.policy.max_materialized_nbytes,
            "cells": self.policy.max_cells,
            "series": self.policy.max_series,
            "output_elements": self.policy.max_output_elements,
            "histogram_samples": self.policy.max_histogram_samples,
        }
        for field, amount in additions.items():
            total = getattr(self, field) + amount
            if total > limits[field]:
                raise FigureEvaluationLimitExceeded(
                    f"figure {field} budget {total} exceeds limit {limits[field]}"
                )
            setattr(self, field, total)


@dataclass(frozen=True)
class ResolvedDataset:
    dataset_id: DatasetId
    snapshot: OwnedSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be zlc_data.OwnedSnapshot")


@dataclass(frozen=True)
class ResolvedDatasetMap:
    entries: tuple[ResolvedDataset, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries or any(not isinstance(entry, ResolvedDataset) for entry in entries):
            raise ValueError("ResolvedDatasetMap requires ResolvedDataset values")
        ids = tuple(entry.dataset_id for entry in entries)
        if len(set(ids)) != len(ids):
            raise ValueError("ResolvedDatasetMap dataset ids must be unique")
        object.__setattr__(self, "entries", entries)

    def resolve(self, dataset_id: DatasetId) -> OwnedSnapshot:
        for entry in self.entries:
            if entry.dataset_id == dataset_id:
                return entry.snapshot
        raise KeyError(dataset_id)


@dataclass
class _WorkingData:
    values: np.ndarray
    validity: np.ndarray
    cell_axes: tuple[AxisSpec, ...]
    cell_coordinates: dict[AxisId, np.ndarray]
    data_axes: tuple[AxisSpec, ...]
    data_indices: dict[AxisId, Sequence[int]]


def _axis_coordinate(axis: AxisSpec, index: int) -> Any:
    return index if axis.coordinates is None else axis.coordinates[index]


def _evaluated_axis(axis: AxisSpec, indices: tuple[int, ...]) -> EvaluatedAxis:
    return EvaluatedAxis(
        axis.axis_id,
        axis.name,
        axis.unit,
        indices,
        tuple(_axis_coordinate(axis, index) for index in indices),
    )


def _selection_index_sets(block: DataBlock, view) -> dict[AxisId, Sequence[int]]:
    return {
        axis.axis_id: display_axis_indices(axis, view.display_selections)
        for axis in dataset_axes(block.schema)
    }


def _cell_layout(block: DataBlock) -> AxisLayout:
    return AxisLayout.product(
        AxisLayout.rect_c((block.schema.repeat_axis.size,)),
        block.schema.point_layout,
    )


def _component_validity(
    block: DataBlock,
    row_ids: np.ndarray,
    data_choices: dict[AxisId, Sequence[int] | int],
    remaining_data_axes: tuple[AxisSpec, ...],
    values_shape: tuple[int, ...],
) -> np.ndarray:
    validity = block.validity
    if isinstance(validity, Valid):
        return np.broadcast_to(True, values_shape)
    if isinstance(validity, Invalid):
        return np.broadcast_to(False, values_shape)
    if isinstance(validity, CellValidity):
        row_mask = validity.mask.reshape(-1)[row_ids]
        return np.broadcast_to(
            row_mask.reshape((len(row_ids),) + (1,) * len(remaining_data_axes)),
            values_shape,
        )
    if not isinstance(validity, ComponentValidity):
        raise TypeError(f"unsupported dataset validity {type(validity).__name__}")
    original_data = block.schema.cell_schema.data_axes
    component_axes = tuple(
        next(axis for axis in original_data if axis.axis_id == axis_id)
        for axis_id in validity.axis_ids
    )
    mask = validity.mask.reshape(
        (block.schema.repeat_axis.size * block.schema.point_layout.storage_size,)
        + tuple(axis.size for axis in component_axes)
    )[row_ids]
    current_component_axes = list(component_axes)
    array_axis = 1
    for axis in component_axes:
        choice = data_choices[axis.axis_id]
        if isinstance(choice, int):
            mask = np.take(mask, choice, axis=array_axis)
            current_component_axes.remove(axis)
        else:
            mask = np.take(mask, choice, axis=array_axis)
            array_axis += 1
    compact_shape = [len(row_ids)]
    current_ids = {axis.axis_id for axis in current_component_axes}
    for axis in remaining_data_axes:
        if axis.axis_id in current_ids:
            compact_shape.append(len(data_choices[axis.axis_id]))
        else:
            compact_shape.append(1)
    return np.broadcast_to(mask.reshape(tuple(compact_shape)), values_shape)


def _extract(
    block: DataBlock,
    fixed_indices: dict[AxisId, int],
    allowed_indices: dict[AxisId, Sequence[int]],
) -> _WorkingData:
    cell_axes = (block.schema.repeat_axis, *block.schema.point_axes)
    layout = _cell_layout(block)
    coordinate_columns = {
        axis.axis_id: layout.axis_indices(position)
        for position, axis in enumerate(cell_axes)
    }
    row_mask = np.ones(layout.storage_size, dtype=bool)
    for axis in cell_axes:
        allowed = allowed_indices[axis.axis_id]
        if axis.axis_id in fixed_indices:
            index = fixed_indices[axis.axis_id]
            if index not in allowed:
                raise FigureEvaluationError(
                    f"fixed index {index} is outside the display selection on {axis.axis_id}"
                )
            row_mask &= coordinate_columns[axis.axis_id] == index
        else:
            row_mask &= np.isin(coordinate_columns[axis.axis_id], allowed)
    row_ids = np.flatnonzero(row_mask)
    remaining_cell_axes = tuple(
        axis for axis in cell_axes if axis.axis_id not in fixed_indices
    )
    remaining_cell_coordinates = {
        axis.axis_id: coordinate_columns[axis.axis_id][row_ids]
        for axis in remaining_cell_axes
    }

    values = block.values.reshape(
        (layout.storage_size, *block.schema.cell_schema.data_shape)
    )[row_ids]
    data_choices: dict[AxisId, Sequence[int] | int] = {}
    remaining_data_axes: list[AxisSpec] = []
    remaining_data_indices: dict[AxisId, Sequence[int]] = {}
    array_axis = 1
    for axis in block.schema.cell_schema.data_axes:
        allowed = allowed_indices[axis.axis_id]
        if axis.axis_id in fixed_indices:
            index = fixed_indices[axis.axis_id]
            if index not in allowed:
                raise FigureEvaluationError(
                    f"fixed index {index} is outside the display selection on {axis.axis_id}"
                )
            data_choices[axis.axis_id] = index
            values = np.take(values, index, axis=array_axis)
        else:
            data_choices[axis.axis_id] = allowed
            values = np.take(values, allowed, axis=array_axis)
            remaining_data_axes.append(axis)
            remaining_data_indices[axis.axis_id] = allowed
            array_axis += 1
    validity = _component_validity(
        block,
        row_ids,
        data_choices,
        tuple(remaining_data_axes),
        values.shape,
    )
    return _WorkingData(
        np.asarray(values),
        np.asarray(validity, dtype=bool),
        remaining_cell_axes,
        remaining_cell_coordinates,
        tuple(remaining_data_axes),
        remaining_data_indices,
    )


def _slice_working(
    working: _WorkingData,
    fixed_indices: dict[AxisId, int],
) -> _WorkingData:
    """Slice an already materialized layer without revisiting DataBlock layout/validity."""

    known_ids = {
        axis.axis_id for axis in working.cell_axes + working.data_axes
    }
    unknown = set(fixed_indices) - known_ids
    if unknown:
        raise FigureEvaluationError(f"slice references absent axes: {tuple(sorted(unknown))}")

    fixed_cell_axes = tuple(
        axis for axis in working.cell_axes if axis.axis_id in fixed_indices
    )
    if fixed_cell_axes:
        row_mask = np.ones(len(working.values), dtype=bool)
        for axis in fixed_cell_axes:
            row_mask &= working.cell_coordinates[axis.axis_id] == fixed_indices[axis.axis_id]
        values = working.values[row_mask]
        validity = working.validity[row_mask]
        cell_axes = tuple(
            axis for axis in working.cell_axes if axis.axis_id not in fixed_indices
        )
        cell_coordinates = {
            axis.axis_id: working.cell_coordinates[axis.axis_id][row_mask]
            for axis in cell_axes
        }
    else:
        values = working.values
        validity = working.validity
        cell_axes = working.cell_axes
        cell_coordinates = working.cell_coordinates

    data_axes = list(working.data_axes)
    data_indices = dict(working.data_indices)
    for axis in tuple(working.data_axes):
        if axis.axis_id not in fixed_indices:
            continue
        index = fixed_indices[axis.axis_id]
        try:
            local_position = data_indices[axis.axis_id].index(index)
        except ValueError as exc:
            raise FigureEvaluationError(
                f"fixed index {index} is outside the materialized selection on {axis.axis_id}"
            ) from exc
        array_axis = 1 + next(
            position
            for position, current in enumerate(data_axes)
            if current.axis_id == axis.axis_id
        )
        indexer = [slice(None)] * values.ndim
        indexer[array_axis] = local_position
        values = values[tuple(indexer)]
        validity = validity[tuple(indexer)]
        data_axes = [current for current in data_axes if current.axis_id != axis.axis_id]
        del data_indices[axis.axis_id]

    return _WorkingData(
        np.asarray(values),
        np.asarray(validity, dtype=bool),
        cell_axes,
        cell_coordinates,
        tuple(data_axes),
        data_indices,
    )


def _layer_budget(view, allowed) -> tuple[int, int, int, int]:
    cardinality = {axis_id: len(indices) for axis_id, indices in allowed.items()}
    facet_cells = math.prod(
        cardinality[binding.axis_id]
        for binding in view.axis_bindings
        if binding.role is AxisViewRole.FACET
    )
    batch_per_cell = math.prod(
        cardinality[binding.axis_id]
        for binding in view.axis_bindings
        if binding.role is AxisViewRole.BATCH
    )
    series = facet_cells * batch_per_cell
    if view.intent is ViewIntent.IMAGE:
        per_series = math.prod(
            cardinality[binding.axis_id]
            for binding in view.axis_bindings
            if binding.role in (AxisViewRole.IMAGE_X, AxisViewRole.IMAGE_Y)
        )
    elif view.intent is ViewIntent.CURVE:
        per_series = next(
            cardinality[binding.axis_id]
            for binding in view.axis_bindings
            if binding.role is AxisViewRole.X
        )
    elif view.intent is ViewIntent.HISTOGRAM:
        per_series = math.prod(
            cardinality[binding.axis_id]
            for binding in view.axis_bindings
            if binding.role is AxisViewRole.SAMPLE
        )
    else:
        per_series = 1
    output_elements = series * per_series
    histogram_samples = output_elements if view.intent is ViewIntent.HISTOGRAM else 0
    return facet_cells, series, output_elements, histogram_samples


def _layer_resource_upper_bound(
    block: DataBlock,
    allowed,
    fixed_indices: dict[AxisId, int],
    view,
) -> tuple[int, int, int]:
    """Bound internal extraction/reduction workspace before materialization.

    Retained ``Evaluated*`` DTO arrays are governed separately by the output-
    element budget; this value is deliberately not advertised as a process RSS
    bound.
    """

    physical_rows = (
        block.schema.repeat_axis.size * block.schema.point_layout.storage_size
    )
    selected_data_elements = math.prod(
        1 if axis.axis_id in fixed_indices else len(allowed[axis.axis_id])
        for axis in block.schema.cell_schema.data_axes
    )
    elements = physical_rows * selected_data_elements
    value_bytes = block.values.dtype.itemsize
    bool_bytes = np.dtype(bool).itemsize
    int64_bytes = np.dtype(np.int64).itemsize
    intp_bytes = np.dtype(np.intp).itemsize
    cell_axis_count = 1 + len(block.schema.point_axes)

    # _extract owns full int64 coordinate columns, row filters/ids, filtered
    # coordinate copies, a values+validity base, and at least one transient
    # selection copy.  Summing them is deliberately more conservative than a
    # lifetime-tight peak model.
    cell_workspace = physical_rows * (
        (2 * cell_axis_count * int64_bytes)
        + ((1 + cell_axis_count) * bool_bytes)
        + intp_bytes
    )
    base_workspace = elements * (value_bytes + bool_bytes)
    extraction_transient = elements * (value_bytes + bool_bytes)
    workspace_nbytes = cell_workspace + base_workspace + extraction_transient

    reductions = tuple(
        binding for binding in view.axis_bindings if binding.role is AxisViewRole.REDUCED
    )
    reduction_contributions = elements if reductions else 0
    if reductions:
        method = reductions[0].reduction.method
        output_dtype = (
            canonical_mean_dtype(block.values.dtype)
            if method is DisplayReductionMethod.MEAN
            else canonical_sum_dtype(block.values.dtype)
        )
        output_bytes = output_dtype.itemsize
        # OrderedDict keys/lists and row-index matrices are Python-heavy.  A
        # 128 B/physical-row plus 40 B/cell-axis intentionally overstates
        # compact rectangular cases while covering tuple/PyLong growth; a small
        # output cannot hide grouping cost.  Array terms cover grouped
        # values/validity, np.where safe input, canonical accumulator,
        # worst-case output, counts, and output validity.
        grouping_workspace = physical_rows * (
            128 + intp_bytes + (40 * cell_axis_count)
        )
        reduction_arrays = elements * (
            value_bytes
            + bool_bytes
            + value_bytes
            + output_bytes
            + output_bytes
            + int64_bytes
            + bool_bytes
        )
        workspace_nbytes += grouping_workspace + reduction_arrays
        reduced_contributors_per_output = math.prod(
            len(allowed[binding.axis_id]) for binding in reductions
        )
        if (
            method is DisplayReductionMethod.SUM
            and block.values.dtype.kind in "iu"
            and reduced_contributors_per_output > 1
        ):
            # checked_numeric_sum may promote every contribution to a Python
            # integer for exact overflow detection.  Pointer + PyLong + allocator
            # overhead is conservatively budgeted as 64 B/contribution.
            workspace_nbytes += 64 * reduction_contributions
    return physical_rows, reduction_contributions, workspace_nbytes


def _reduce(
    working: _WorkingData,
    reduction_bindings,
    guard: _EvaluationGuard,
) -> tuple[_WorkingData, tuple[ReductionResolution, ...]]:
    if not reduction_bindings:
        return working, ()
    methods = {binding.reduction.method for binding in reduction_bindings}
    if len(methods) != 1:
        raise FigureEvaluationError(
            "baseline FigureEvaluator requires all jointly reduced axes to use one method"
        )
    method = next(iter(methods))
    output_dtype = (
        canonical_mean_dtype(working.values.dtype)
        if method is DisplayReductionMethod.MEAN
        else canonical_sum_dtype(working.values.dtype)
    )
    reduce_ids = {binding.axis_id for binding in reduction_bindings}
    cell_ids = tuple(axis.axis_id for axis in working.cell_axes)
    data_ids = tuple(axis.axis_id for axis in working.data_axes)
    reduced_cell = tuple(axis_id for axis_id in cell_ids if axis_id in reduce_ids)
    reduced_data_positions = tuple(
        position for position, axis_id in enumerate(data_ids) if axis_id in reduce_ids
    )
    if reduce_ids - set(cell_ids) - set(data_ids):
        raise FigureEvaluationError("reduction axis disappeared before evaluation")
    surviving_cell_axes = tuple(
        axis for axis in working.cell_axes if axis.axis_id not in reduce_ids
    )
    surviving_data_axes = tuple(
        axis for axis in working.data_axes if axis.axis_id not in reduce_ids
    )

    groups: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    if reduced_cell:
        for row in range(len(working.values)):
            if row % 4096 == 0:
                guard.check()
            key = tuple(
                int(working.cell_coordinates[axis.axis_id][row])
                for axis in surviving_cell_axes
            )
            groups.setdefault(key, []).append(row)
    else:
        for row in range(len(working.values)):
            if row % 4096 == 0:
                guard.check()
            key = tuple(
                int(working.cell_coordinates[axis.axis_id][row])
                for axis in surviving_cell_axes
            )
            groups[key] = [row]

    data_shape = tuple(
        len(working.data_indices[axis.axis_id]) for axis in surviving_data_axes
    )

    def reduce_arrays(values, validity, reduction_axes):
        counts = np.sum(validity, axis=reduction_axes, dtype=np.int64)
        safe = np.where(validity, values, 0)
        if method is DisplayReductionMethod.MEAN:
            safe = safe.astype(output_dtype, copy=False)
        summed = checked_numeric_sum(
            safe, reduction_axes, output_dtype=output_dtype
        )
        if method is DisplayReductionMethod.MEAN:
            result = np.zeros(np.shape(summed), dtype=output_dtype)
            np.divide(summed, counts, out=result, where=counts > 0)
        else:
            result = np.asarray(summed, dtype=output_dtype)
        valid = counts > 0
        result = np.where(valid, result, np.zeros((), dtype=output_dtype))
        return result, valid, counts

    guard.check()
    if groups:
        if reduced_cell:
            group_lengths = tuple(len(rows) for rows in groups.values())
            if len(set(group_lengths)) == 1:
                # A dense matrix is safe only when it contains exactly the
                # existing rows.  Ragged EXPLICIT layouts must never be padded
                # to num_groups * max_group_rows.
                group_rows = np.asarray(tuple(groups.values()), dtype=np.intp)
                values = working.values[group_rows]
                validity = working.validity[group_rows]
                reduction_axes = (1,) + tuple(
                    2 + position for position in reduced_data_positions
                )
                values_array, validity_array, counts_array = reduce_arrays(
                    values, validity, reduction_axes
                )
            else:
                output_values = []
                output_validity = []
                output_counts = []
                reduction_axes = (0,) + tuple(
                    1 + position for position in reduced_data_positions
                )
                for rows in groups.values():
                    guard.check()
                    row_indices = np.asarray(rows, dtype=np.intp)
                    result, valid, counts = reduce_arrays(
                        working.values[row_indices],
                        working.validity[row_indices],
                        reduction_axes,
                    )
                    output_values.append(np.asarray(result))
                    output_validity.append(np.asarray(valid, dtype=bool))
                    output_counts.append(np.asarray(counts, dtype=np.int64))
                values_array = np.stack(output_values, axis=0)
                validity_array = np.stack(output_validity, axis=0)
                counts_array = np.stack(output_counts, axis=0)
        else:
            values = working.values
            validity = working.validity
            reduction_axes = tuple(
                1 + position for position in reduced_data_positions
            )
            if not reduction_axes:
                raise FigureEvaluationError(
                    "REDUCED binding did not resolve to an array axis"
                )
            values_array, validity_array, counts_array = reduce_arrays(
                values, validity, reduction_axes
            )
    else:
        values_array = np.empty((0, *data_shape), dtype=output_dtype)
        validity_array = np.empty((0, *data_shape), dtype=bool)
        counts_array = np.empty((0, *data_shape), dtype=np.int64)
    keys = tuple(groups)
    coordinates = {
        axis.axis_id: np.asarray([key[index] for key in keys], dtype=np.int64)
        for index, axis in enumerate(surviving_cell_axes)
    }
    nonempty_counts = counts_array.reshape(-1)
    low = int(nonempty_counts.min()) if nonempty_counts.size else 0
    high = int(nonempty_counts.max()) if nonempty_counts.size else 0
    resolution = ReductionResolution(
        tuple(binding.axis_id for binding in reduction_bindings),
        method,
        low,
        high,
    )
    return (
        _WorkingData(
            values_array,
            validity_array,
            surviving_cell_axes,
            coordinates,
            surviving_data_axes,
            {
                axis.axis_id: working.data_indices[axis.axis_id]
                for axis in surviving_data_axes
            },
        ),
        (resolution,),
    )


def _image(working: _WorkingData, view, allowed) -> EvaluatedImage:
    binding_x = next(binding for binding in view.axis_bindings if binding.role is AxisViewRole.IMAGE_X)
    binding_y = next(binding for binding in view.axis_bindings if binding.role is AxisViewRole.IMAGE_Y)
    axes = {axis.axis_id: axis for axis in working.cell_axes + working.data_axes}
    x_axis, y_axis = axes[binding_x.axis_id], axes[binding_y.axis_id]
    x_indices, y_indices = allowed[x_axis.axis_id], allowed[y_axis.axis_id]
    x_out, y_out = _evaluated_axis(x_axis, x_indices), _evaluated_axis(y_axis, y_indices)
    output = np.zeros((len(y_indices), len(x_indices)), dtype=working.values.dtype)
    valid = np.zeros(output.shape, dtype=bool)
    x_pos = {index: position for position, index in enumerate(x_indices)}
    y_pos = {index: position for position, index in enumerate(y_indices)}
    cell_ids = {axis.axis_id for axis in working.cell_axes}
    data_ids = {axis.axis_id for axis in working.data_axes}
    if {x_axis.axis_id, y_axis.axis_id} <= data_ids:
        if len(working.cell_axes) or len(working.values) > 1:
            raise FigureEvaluationError("image data axes still have unresolved cell axes")
        if len(working.values) == 1:
            data_order = tuple(axis.axis_id for axis in working.data_axes)
            row = working.values[0]
            row_valid = working.validity[0]
            permutation = (data_order.index(y_axis.axis_id), data_order.index(x_axis.axis_id))
            output[...] = np.transpose(row, permutation)
            valid[...] = np.transpose(row_valid, permutation)
    elif {x_axis.axis_id, y_axis.axis_id} <= cell_ids:
        if working.data_axes:
            raise FigureEvaluationError("image cell axes still have unresolved data axes")
        for row in range(len(working.values)):
            xi = int(working.cell_coordinates[x_axis.axis_id][row])
            yi = int(working.cell_coordinates[y_axis.axis_id][row])
            output[y_pos[yi], x_pos[xi]] = working.values[row]
            valid[y_pos[yi], x_pos[xi]] = working.validity[row]
    else:
        cell_axis = x_axis if x_axis.axis_id in cell_ids else y_axis
        data_axis = y_axis if cell_axis is x_axis else x_axis
        if len(working.cell_axes) != 1 or len(working.data_axes) != 1:
            raise FigureEvaluationError("mixed image axes retain unrelated axes")
        for row in range(len(working.values)):
            ci = int(working.cell_coordinates[cell_axis.axis_id][row])
            if cell_axis is x_axis:
                output[:, x_pos[ci]] = working.values[row]
                valid[:, x_pos[ci]] = working.validity[row]
            else:
                output[y_pos[ci], :] = working.values[row]
                valid[y_pos[ci], :] = working.validity[row]
        if data_axis.axis_id not in working.data_indices:
            raise FigureEvaluationError("mixed image data axis is absent")
    return EvaluatedImage(x_out, y_out, output, valid)


def _curve(working: _WorkingData, view, allowed) -> EvaluatedCurve:
    binding = next(binding for binding in view.axis_bindings if binding.role is AxisViewRole.X)
    axes = {axis.axis_id: axis for axis in working.cell_axes + working.data_axes}
    axis = axes[binding.axis_id]
    indices = allowed[axis.axis_id]
    out_axis = _evaluated_axis(axis, indices)
    output = np.zeros((len(indices),), dtype=working.values.dtype)
    valid = np.zeros(output.shape, dtype=bool)
    positions = {index: position for position, index in enumerate(indices)}
    cell_ids = {item.axis_id for item in working.cell_axes}
    if axis.axis_id in cell_ids:
        if working.data_axes:
            raise FigureEvaluationError("curve x axis retains unrelated data axes")
        for row in range(len(working.values)):
            index = int(working.cell_coordinates[axis.axis_id][row])
            output[positions[index]] = working.values[row]
            valid[positions[index]] = working.validity[row]
    else:
        if working.cell_axes or len(working.values) > 1 or len(working.data_axes) != 1:
            raise FigureEvaluationError("curve x data axis retains unrelated axes")
        if len(working.values) == 1:
            output[...] = working.values[0]
            valid[...] = working.validity[0]
    return EvaluatedCurve(out_axis, output, valid)


def _sample_coordinate_values(
    axis: AxisSpec,
    indices: np.ndarray,
    guard: _EvaluationGuard,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for start in range(0, len(indices), 4096):
        guard.check()
        values.extend(
            _axis_coordinate(axis, int(index))
            for index in indices[start : start + 4096]
        )
    return tuple(values)


def _histogram(
    working: _WorkingData,
    view,
    guard: _EvaluationGuard,
) -> EvaluatedHistogram:
    sample_ids = {
        binding.axis_id for binding in view.axis_bindings if binding.role is AxisViewRole.SAMPLE
    }
    remaining_ids = {
        axis.axis_id for axis in working.cell_axes + working.data_axes
    }
    if sample_ids != remaining_ids:
        raise FigureEvaluationError("histogram retains a non-sample axis")
    flat_values = working.values.reshape(-1)
    flat_validity = working.validity.reshape(-1)
    data_shape = working.values.shape[1:]
    data_items = int(np.prod(data_shape, dtype=np.int64)) if data_shape else 1
    coordinate_columns: dict[AxisId, np.ndarray] = {}
    for axis in working.cell_axes:
        coordinate_columns[axis.axis_id] = np.repeat(
            working.cell_coordinates[axis.axis_id], data_items
        )
    if working.data_axes:
        grids = np.meshgrid(
            *(working.data_indices[axis.axis_id] for axis in working.data_axes),
            indexing="ij",
        )
        for axis, grid in zip(working.data_axes, grids):
            coordinate_columns[axis.axis_id] = np.tile(
                np.asarray(grid).reshape(-1), len(working.values)
            )
    keep = np.asarray(flat_validity, dtype=bool)
    coordinates = []
    axis_by_id = {axis.axis_id: axis for axis in working.cell_axes + working.data_axes}
    for binding in view.axis_bindings:
        if binding.role is not AxisViewRole.SAMPLE:
            continue
        axis = axis_by_id[binding.axis_id]
        indices = coordinate_columns[axis.axis_id][keep]
        coordinates.append(
            SampleCoordinates(
                axis.axis_id,
                _sample_coordinate_values(axis, indices, guard),
            )
        )
    return EvaluatedHistogram(
        flat_values[keep],
        tuple(coordinates),
        int(len(flat_values) - np.count_nonzero(keep)),
    )


def _meter(working: _WorkingData) -> EvaluatedMeter:
    if working.cell_axes or working.data_axes:
        raise FigureEvaluationError("meter retains unresolved axes")
    if working.values.size == 0:
        return EvaluatedMeter(0.0, False)
    if working.values.size != 1:
        raise FigureEvaluationError("meter evaluation produced more than one value")
    return EvaluatedMeter(working.values.reshape(-1)[0], bool(working.validity.reshape(-1)[0]))


def _address(axis: AxisSpec, index: int) -> AxisAddress:
    return AxisAddress(axis.axis_id, index, _axis_coordinate(axis, index))


def _combinations(axes: tuple[AxisSpec, ...], allowed):
    return product(*(allowed[axis.axis_id] for axis in axes))


class FigureEvaluator:
    """Evaluate a FigureDocument without importing any renderer or authority transform."""

    def __init__(self, policy: FigureEvaluationPolicy | None = None) -> None:
        self._policy = FigureEvaluationPolicy() if policy is None else policy
        if not isinstance(self._policy, FigureEvaluationPolicy):
            raise TypeError("policy must be FigureEvaluationPolicy or None")

    def evaluate(
        self,
        document: FigureDocument,
        datasets: ResolvedDatasetMap,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        monotonic_deadline: float | None = None,
    ) -> EvaluatedFigureData:
        if not isinstance(document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        if not isinstance(datasets, ResolvedDatasetMap):
            raise TypeError("datasets must be ResolvedDatasetMap")
        if cancel_requested is not None and not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable or None")
        if monotonic_deadline is not None:
            if (
                isinstance(monotonic_deadline, bool)
                or not isinstance(monotonic_deadline, Real)
                or not math.isfinite(monotonic_deadline)
            ):
                raise ValueError("monotonic_deadline must be finite numeric time or None")
            monotonic_deadline = float(monotonic_deadline)
        if len(document.layers) > self._policy.max_layers:
            raise FigureEvaluationLimitExceeded(
                f"figure layer count {len(document.layers)} exceeds limit {self._policy.max_layers}"
            )
        guard = _EvaluationGuard(self._policy, cancel_requested, monotonic_deadline)
        guard.check()
        evaluated_layers = []
        inputs: dict[DatasetId, EvaluatedInput] = {}
        for layer in document.layers:
            guard.check()
            snapshot = datasets.resolve(layer.dataset_id)
            descriptor = document.descriptor(layer.dataset_id)
            block = snapshot.block
            if descriptor.schema_fingerprint != snapshot.ref.schema_fingerprint:
                raise FigureEvaluationError(
                    f"dataset {layer.dataset_id} descriptor/snapshot schema mismatch"
                )
            validate_view_spec(block.schema, layer.view)
            inputs.setdefault(layer.dataset_id, EvaluatedInput(layer.dataset_id, snapshot.ref))
            evaluated_layers.append(self._evaluate_layer(layer, block, guard))
        return EvaluatedFigureData(
            document.document_id,
            document.revision,
            tuple(inputs.values()),
            tuple(evaluated_layers),
        )

    def _evaluate_layer(
        self,
        layer,
        block: DataBlock,
        guard: _EvaluationGuard,
    ) -> EvaluatedLayer:
        guard.check()
        view = layer.view
        axes = dataset_axes(block.schema)
        axis_by_id = {axis.axis_id: axis for axis in axes}
        allowed = _selection_index_sets(block, view)
        fixed: dict[AxisId, int] = {}
        dynamic = []
        for binding in view.axis_bindings:
            if isinstance(binding.selector, FixedIndex):
                fixed[binding.axis_id] = binding.selector.index
            elif isinstance(binding.selector, LatestNonempty):
                dynamic.append(binding)
        if len(dynamic) > 1:
            raise FigureEvaluationError("only one global LatestNonempty selector is supported")
        cells, series_count, output_elements, histogram_samples = _layer_budget(
            view, allowed
        )
        (
            physical_rows,
            reduction_contributions,
            materialized_nbytes,
        ) = _layer_resource_upper_bound(block, allowed, fixed, view)
        guard.reserve_layer(
            cells=cells,
            series=series_count,
            output_elements=output_elements,
            histogram_samples=histogram_samples,
            materialized_nbytes=materialized_nbytes,
            physical_rows=physical_rows,
            reduction_contributions=reduction_contributions,
        )
        guard.check()
        working_base = _extract(block, fixed, allowed)
        guard.check()
        resolutions = []
        if dynamic:
            binding = dynamic[0]
            axis = axis_by_id[binding.axis_id]
            resolved = None
            for index in reversed(allowed[axis.axis_id]):
                guard.check()
                if np.any(_slice_working(working_base, {axis.axis_id: index}).validity):
                    resolved = index
                    break
            if resolved is None:
                raise FigureEvaluationError(f"axis {axis.axis_id} has no non-empty display index")
            working_base = _slice_working(working_base, {axis.axis_id: resolved})
            resolutions.append(
                AxisResolution(
                    axis.axis_id,
                    "LATEST_NONEMPTY",
                    resolved,
                    _axis_coordinate(axis, resolved),
                )
            )

        facet_axes = tuple(
            axis_by_id[binding.axis_id]
            for binding in view.axis_bindings
            if binding.role is AxisViewRole.FACET
        )
        batch_axes = tuple(
            axis_by_id[binding.axis_id]
            for binding in view.axis_bindings
            if binding.role is AxisViewRole.BATCH
        )
        reduction_bindings = tuple(
            binding for binding in view.axis_bindings if binding.role is AxisViewRole.REDUCED
        )
        cells = []
        for facet_indices in _combinations(facet_axes, allowed):
            guard.check()
            facet_fixed = dict(
                (axis.axis_id, index) for axis, index in zip(facet_axes, facet_indices)
            )
            facet_working = _slice_working(working_base, facet_fixed)
            series = []
            for batch_indices in _combinations(batch_axes, allowed):
                guard.check()
                series_fixed = dict(
                    (axis.axis_id, index) for axis, index in zip(batch_axes, batch_indices)
                )
                working = _slice_working(facet_working, series_fixed)
                working, reduction_records = _reduce(
                    working, reduction_bindings, guard
                )
                if view.intent is ViewIntent.IMAGE:
                    data = _image(working, view, allowed)
                elif view.intent is ViewIntent.CURVE:
                    data = _curve(working, view, allowed)
                elif view.intent is ViewIntent.HISTOGRAM:
                    data = _histogram(working, view, guard)
                else:
                    data = _meter(working)
                series.append(
                    EvaluatedSeries(
                        tuple(
                            _address(axis, index)
                            for axis, index in zip(batch_axes, batch_indices)
                        ),
                        data,
                        reduction_records,
                    )
                )
            cells.append(
                EvaluatedCell(
                    tuple(
                        _address(axis, index)
                        for axis, index in zip(facet_axes, facet_indices)
                    ),
                    tuple(series),
                )
            )
        return EvaluatedLayer(
            layer.layer_id,
            layer.dataset_id,
            tuple(cells),
            tuple(resolutions),
        )


__all__ = [
    "FigureEvaluationCancelled",
    "FigureEvaluationDeadlineExceeded",
    "FigureEvaluationError",
    "FigureEvaluationLimitExceeded",
    "FigureEvaluationPolicy",
    "FigureEvaluator",
    "ResolvedDataset",
    "ResolvedDatasetMap",
]
