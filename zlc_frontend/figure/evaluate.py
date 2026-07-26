"""Headless ViewSpec evaluation over immutable zlc_data snapshots."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
import math
from numbers import Real
import time
from typing import Any, Callable, Sequence

import numpy as np

from zlc_data import (
    AxisId,
    AxisSpec,
    CellValidity,
    DataBlock,
    DatasetComponentValidity,
    DatasetSchema,
    Invalid,
    OwnedSnapshot,
    REPEAT,
    Valid,
    immutable_bool_broadcast,
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
    ViewSpec,
)


class FigureEvaluationError(ValueError):
    """The document is well-typed but cannot be evaluated on this snapshot."""


class FigureEvaluationCancelled(FigureEvaluationError):
    """Evaluation was cooperatively cancelled at a bounded checkpoint."""


class FigureEvaluationDeadlineExceeded(FigureEvaluationError):
    """Evaluation crossed its caller-provided monotonic deadline."""


@dataclass
class _EvaluationGuard:
    cancel_requested: Callable[[], bool] | None
    monotonic_deadline: float | None

    def check(self) -> None:
        if self.cancel_requested is not None and self.cancel_requested():
            raise FigureEvaluationCancelled("figure evaluation was cancelled")
        if (
            self.monotonic_deadline is not None
            and time.monotonic() >= self.monotonic_deadline
        ):
            raise FigureEvaluationDeadlineExceeded("figure evaluation deadline exceeded")

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
    return axis.coordinate_at(index)


def evaluate_axis(axis: AxisSpec, indices: tuple[int, ...]) -> EvaluatedAxis:
    """Evaluate declared coordinates for one explicit ordered index set."""

    if not isinstance(axis, AxisSpec):
        raise TypeError("axis must be AxisSpec")
    indices = tuple(indices)
    return EvaluatedAxis(
        axis.axis_id,
        axis.name,
        axis.role,
        axis.unit,
        indices,
        tuple(_axis_coordinate(axis, index) for index in indices),
        axis.coordinate_frame,
    )


def _selection_index_sets(
    schema: DatasetSchema,
    view: ViewSpec,
) -> dict[AxisId, Sequence[int]]:
    return {
        axis.axis_id: display_axis_indices(axis, view.display_selections)
        for axis in dataset_axes(schema)
    }


def _sequence_indexer(indices: Sequence[int] | np.ndarray) -> slice | np.ndarray:
    """Use a basic slice for a contiguous logical selection, else explicit indices."""

    if isinstance(indices, range) and indices.step == 1:
        return slice(indices.start, indices.stop)
    explicit = (
        np.asarray(indices, dtype=np.intp)
        if isinstance(indices, np.ndarray)
        else np.fromiter(indices, dtype=np.intp, count=len(indices))
    )
    if explicit.ndim != 1:
        raise ValueError("axis indices must be one-dimensional")
    if explicit.size == 0:
        return slice(0, 0)
    start = int(explicit[0])
    if np.array_equal(
        explicit,
        np.arange(start, start + explicit.size, dtype=np.intp),
    ):
        return slice(start, start + explicit.size)
    return explicit


def _select_axis(
    array: np.ndarray,
    choice: Sequence[int] | np.ndarray | int,
    *,
    axis: int,
) -> np.ndarray:
    """Select without copying contiguous regions; copy only true gathers."""

    indexer = choice if isinstance(choice, int) else _sequence_indexer(choice)
    if isinstance(indexer, np.ndarray):
        return np.take(array, indexer, axis=axis)
    selection = [slice(None)] * array.ndim
    selection[axis] = indexer
    return array[tuple(selection)]


def _ordered_selection_axes(
    axes: Sequence[AxisSpec],
    choices: dict[AxisId, Sequence[int] | int],
) -> tuple[AxisSpec, ...]:
    """Apply views first, then the narrowest gathers, without reordering axes."""

    def key(axis: AxisSpec) -> tuple[int, float, str]:
        choice = choices[axis.axis_id]
        if isinstance(choice, int):
            return (0, 1.0 / axis.size, axis.axis_id.value)
        selected = len(choice)
        kind = 1 if isinstance(choice, range) else 2
        return (kind, selected / axis.size, axis.axis_id.value)

    return tuple(sorted(axes, key=key))


def _select_named_axes(
    array: np.ndarray,
    axes: Sequence[AxisSpec],
    choices: dict[AxisId, Sequence[int] | int],
) -> tuple[np.ndarray, tuple[AxisSpec, ...]]:
    active = list(axes)
    for axis in _ordered_selection_axes(axes, choices):
        position = 1 + active.index(axis)
        choice = choices[axis.axis_id]
        array = _select_axis(array, choice, axis=position)
        if isinstance(choice, int):
            active.remove(axis)
    return array, tuple(active)


def _component_validity(
    block: DataBlock,
    row_ids: np.ndarray,
    data_choices: dict[AxisId, Sequence[int] | int],
    remaining_data_axes: tuple[AxisSpec, ...],
    values_shape: tuple[int, ...],
) -> np.ndarray:
    validity = block.validity
    if isinstance(validity, Valid):
        return immutable_bool_broadcast(True, values_shape)
    if isinstance(validity, Invalid):
        return immutable_bool_broadcast(False, values_shape)
    if isinstance(validity, CellValidity):
        row_mask = _select_axis(validity.mask.reshape(-1), row_ids, axis=0)
        return np.broadcast_to(
            row_mask.reshape((len(row_ids),) + (1,) * len(remaining_data_axes)),
            values_shape,
        )
    if not isinstance(validity, DatasetComponentValidity):
        raise TypeError(f"unsupported dataset validity {type(validity).__name__}")
    original_data = block.schema.cell_schema.data_axes
    component_axes = tuple(
        next(axis for axis in original_data if axis.axis_id == axis_id)
        for axis_id in validity.axis_ids
    )
    mask = _select_axis(
        validity.mask.reshape(
            (block.schema.repeat_axis.size * block.schema.point_layout.storage_size,)
            + tuple(axis.size for axis in component_axes)
        ),
        row_ids,
        axis=0,
    )
    mask, current_component_axes = _select_named_axes(
        mask,
        component_axes,
        data_choices,
    )
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
    layout = block.schema.cell_layout
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
        axis.axis_id: _select_axis(coordinate_columns[axis.axis_id], row_ids, axis=0)
        for axis in remaining_cell_axes
    }

    values = _select_axis(
        block.values.reshape(
            (layout.storage_size, *block.schema.cell_schema.data_shape)
        ),
        row_ids,
        axis=0,
    )
    data_choices: dict[AxisId, Sequence[int] | int] = {}
    remaining_data_axes: list[AxisSpec] = []
    remaining_data_indices: dict[AxisId, Sequence[int]] = {}
    for axis in block.schema.cell_schema.data_axes:
        allowed = allowed_indices[axis.axis_id]
        if axis.axis_id in fixed_indices:
            index = fixed_indices[axis.axis_id]
            if index not in allowed:
                raise FigureEvaluationError(
                    f"fixed index {index} is outside the display selection on {axis.axis_id}"
                )
            data_choices[axis.axis_id] = index
        else:
            data_choices[axis.axis_id] = allowed
            remaining_data_axes.append(axis)
            remaining_data_indices[axis.axis_id] = allowed
    values, selected_data_axes = _select_named_axes(
        values,
        block.schema.cell_schema.data_axes,
        data_choices,
    )
    assert selected_data_axes == tuple(remaining_data_axes)
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
        row_ids = np.flatnonzero(row_mask)
        values = _select_axis(working.values, row_ids, axis=0)
        validity = _select_axis(working.validity, row_ids, axis=0)
        cell_axes = tuple(
            axis for axis in working.cell_axes if axis.axis_id not in fixed_indices
        )
        cell_coordinates = {
            axis.axis_id: _select_axis(
                working.cell_coordinates[axis.axis_id], row_ids, axis=0
            )
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


def _reduction_output_dtype(
    input_dtype: np.dtype,
    method: DisplayReductionMethod,
) -> np.dtype:
    return (
        canonical_mean_dtype(input_dtype)
        if method is DisplayReductionMethod.MEAN
        else canonical_sum_dtype(input_dtype)
    )


def _single_contributor_reduction_dtype(
    input_dtype: np.dtype,
    reductions,
    axis_by_id: dict[AxisId, AxisSpec],
    maximum_contributors: int,
) -> np.dtype | None:
    """Select the one-contributor kernel by declared role, never by array rank."""

    if len(reductions) != 1 or maximum_contributors > 1:
        return None
    binding = reductions[0]
    if axis_by_id[binding.axis_id].role != REPEAT:
        return None
    assert binding.reduction is not None
    if input_dtype.kind in "biu":
        return input_dtype
    return _reduction_output_dtype(input_dtype, binding.reduction.method)


def _single_contributor_validity_extent(validity: np.ndarray) -> tuple[int, int]:
    """Resolve 0/1 contributor bounds without scanning a uniform broadcast plane."""

    validity = np.asarray(validity, dtype=bool)
    if not validity.size:
        return 0, 0
    if all(stride == 0 for stride in validity.strides):
        contributor = int(bool(validity.flat[0]))
        return contributor, contributor
    return (
        1 if bool(np.all(validity)) else 0,
        1 if bool(np.any(validity)) else 0,
    )


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
    output_dtype = _reduction_output_dtype(working.values.dtype, method)
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

    maximum_cell_contributors = max(
        (len(rows) for rows in groups.values()),
        default=0,
    )
    maximum_data_contributors = math.prod(
        len(working.data_indices[working.data_axes[position].axis_id])
        for position in reduced_data_positions
    )
    axis_by_id = {
        axis.axis_id: axis for axis in working.cell_axes + working.data_axes
    }
    singleton_dtype = _single_contributor_reduction_dtype(
        working.values.dtype,
        reduction_bindings,
        axis_by_id,
        maximum_cell_contributors * maximum_data_contributors,
    )
    if singleton_dtype is not None:
        guard.check()
        # A one-contributor repeat reduction is mathematically an identity.
        # Keep the immutable source view when its declared output dtype is also
        # unchanged; validity remains an independent mask, so an invalid sample
        # never requires rewriting its stored pixel to a sentinel value.  This
        # is the common live-camera path and avoids copying an entire frame at
        # the presentation boundary merely to remove a singleton repeat axis.
        if np.dtype(singleton_dtype) == working.values.dtype:
            values_array = working.values
            validity_array = working.validity
        else:
            values_array = np.asarray(working.values, dtype=singleton_dtype)
            validity_array = working.validity
        low, high = _single_contributor_validity_extent(validity_array)
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
                {
                    axis.axis_id: working.cell_coordinates[axis.axis_id]
                    for axis in surviving_cell_axes
                },
                surviving_data_axes,
                {
                    axis.axis_id: working.data_indices[axis.axis_id]
                    for axis in surviving_data_axes
                },
            ),
            (resolution,),
        )

    data_shape = tuple(
        len(working.data_indices[axis.axis_id]) for axis in surviving_data_axes
    )

    def reduce_arrays(values, validity, reduction_axes):
        counts = np.sum(validity, axis=reduction_axes, dtype=np.int64)
        safe = np.where(validity, values, np.zeros((), dtype=values.dtype))
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


def _image(
    working: _WorkingData,
    view,
    allowed,
    *,
    value_unit: str | None,
) -> EvaluatedImage:
    binding_x = next(binding for binding in view.axis_bindings if binding.role is AxisViewRole.IMAGE_X)
    binding_y = next(binding for binding in view.axis_bindings if binding.role is AxisViewRole.IMAGE_Y)
    axes = {axis.axis_id: axis for axis in working.cell_axes + working.data_axes}
    x_axis, y_axis = axes[binding_x.axis_id], axes[binding_y.axis_id]
    x_indices, y_indices = allowed[x_axis.axis_id], allowed[y_axis.axis_id]
    x_out, y_out = evaluate_axis(x_axis, x_indices), evaluate_axis(y_axis, y_indices)
    cell_ids = {axis.axis_id for axis in working.cell_axes}
    data_ids = {axis.axis_id for axis in working.data_axes}
    if {x_axis.axis_id, y_axis.axis_id} <= data_ids:
        if (
            len(working.cell_axes)
            or len(working.values) > 1
            or len(working.data_axes) != 2
        ):
            raise FigureEvaluationError("image data axes still have unresolved cell axes")
        if len(working.values) == 1:
            data_order = tuple(axis.axis_id for axis in working.data_axes)
            row = working.values[0]
            row_valid = working.validity[0]
            permutation = (data_order.index(y_axis.axis_id), data_order.index(x_axis.axis_id))
            return EvaluatedImage(
                x_out,
                y_out,
                np.transpose(row, permutation),
                np.transpose(row_valid, permutation),
                value_unit,
            )
        return EvaluatedImage(
            x_out,
            y_out,
            np.zeros((len(y_indices), len(x_indices)), dtype=working.values.dtype),
            np.zeros((len(y_indices), len(x_indices)), dtype=bool),
            value_unit,
        )

    output = np.zeros((len(y_indices), len(x_indices)), dtype=working.values.dtype)
    valid = np.zeros(output.shape, dtype=bool)
    x_pos = {index: position for position, index in enumerate(x_indices)}
    y_pos = {index: position for position, index in enumerate(y_indices)}
    if {x_axis.axis_id, y_axis.axis_id} <= cell_ids:
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
    return EvaluatedImage(x_out, y_out, output, valid, value_unit)


def _curve(
    working: _WorkingData,
    view,
    allowed,
    *,
    value_unit: str | None,
) -> EvaluatedCurve:
    binding = next(binding for binding in view.axis_bindings if binding.role is AxisViewRole.X)
    axes = {axis.axis_id: axis for axis in working.cell_axes + working.data_axes}
    axis = axes[binding.axis_id]
    indices = allowed[axis.axis_id]
    out_axis = evaluate_axis(axis, indices)
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
            return EvaluatedCurve(
                out_axis,
                value_unit,
                working.values[0],
                working.validity[0],
            )
    return EvaluatedCurve(out_axis, value_unit, output, valid)


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
    *,
    value_unit: str | None,
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
        value_unit,
    )


def _meter(
    working: _WorkingData,
    *,
    value_unit: str | None,
) -> EvaluatedMeter:
    if working.cell_axes or working.data_axes:
        raise FigureEvaluationError("meter retains unresolved axes")
    if working.values.size == 0:
        return EvaluatedMeter(0.0, False, value_unit)
    if working.values.size != 1:
        raise FigureEvaluationError("meter evaluation produced more than one value")
    return EvaluatedMeter(
        working.values.item(),
        bool(working.validity.item()),
        value_unit,
    )


def _address(axis: AxisSpec, index: int) -> AxisAddress:
    return AxisAddress(
        axis.axis_id,
        axis.name,
        axis.role,
        index,
        _axis_coordinate(axis, index),
    )


def _combinations(axes: tuple[AxisSpec, ...], allowed):
    return product(*(allowed[axis.axis_id] for axis in axes))


class FigureEvaluator:
    """Evaluate a FigureDocument without importing any renderer or authority transform."""

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
        guard = _EvaluationGuard(cancel_requested, monotonic_deadline)
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
        allowed = _selection_index_sets(block.schema, view)
        fixed = {
            binding.axis_id: binding.selector.index
            for binding in view.axis_bindings
            if isinstance(binding.selector, FixedIndex)
        }
        dynamic = next(
            (
                binding
                for binding in view.axis_bindings
                if isinstance(binding.selector, LatestNonempty)
            ),
            None,
        )
        working_base = _extract(block, fixed, allowed)
        guard.check()
        resolutions = [
            AxisResolution(
                axis_id,
                "FIXED_INDEX",
                index,
                _axis_coordinate(axis_by_id[axis_id], index),
            )
            for axis_id, index in fixed.items()
        ]
        if dynamic is not None:
            binding = dynamic
            axis = axis_by_id[binding.axis_id]
            resolved = None
            resolved_working = None
            for index in reversed(allowed[axis.axis_id]):
                guard.check()
                candidate = _slice_working(working_base, {axis.axis_id: index})
                if np.any(candidate.validity):
                    resolved = index
                    resolved_working = candidate
                    break
            if resolved is None:
                raise FigureEvaluationError(f"axis {axis.axis_id} has no non-empty display index")
            assert resolved_working is not None
            working_base = resolved_working
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
                    data = _image(
                        working,
                        view,
                        allowed,
                        value_unit=block.schema.cell_schema.value_unit,
                    )
                elif view.intent is ViewIntent.CURVE:
                    data = _curve(
                        working,
                        view,
                        allowed,
                        value_unit=block.schema.cell_schema.value_unit,
                    )
                elif view.intent is ViewIntent.HISTOGRAM:
                    data = _histogram(
                        working,
                        view,
                        guard,
                        value_unit=block.schema.cell_schema.value_unit,
                    )
                elif view.intent is ViewIntent.METER:
                    data = _meter(
                        working,
                        value_unit=block.schema.cell_schema.value_unit,
                    )
                else:
                    raise FigureEvaluationError(
                        f"{view.intent.value} has no DatasetSchema evaluation path"
                    )
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
            tuple(sorted(resolutions, key=lambda item: item.axis_id.value)),
        )


__all__ = [
    "FigureEvaluationCancelled",
    "FigureEvaluationDeadlineExceeded",
    "FigureEvaluationError",
    "FigureEvaluator",
    "ResolvedDataset",
    "ResolvedDatasetMap",
]
