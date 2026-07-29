"""Headless ViewSpec evaluation over immutable ``(R, P, *data)`` snapshots."""

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
    AxisSourceRef,
    DataBlock,
    DatasetSchema,
    OwnedSnapshot,
)
from zlc_data.value import expand_dataset_validity
from zlc_data.numeric import (
    canonical_mean_dtype,
    canonical_sum_dtype,
    checked_numeric_sum,
)

from .contract import (
    _dataset_sources,
    _resolved_point_group_records,
    _resolve_selected_point_ordinals,
    _resolve_view_point_rows,
    _source_cardinality,
    _source_coordinate,
    _source_coordinate_frame,
    _source_key,
    _source_name,
    _source_role,
    _source_unit,
    _tensor_axis,
    validate_view_spec,
)
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
    SourceViewBinding,
    ViewIntent,
    ViewSpec,
)


class FigureEvaluationError(ValueError):
    """The document is well-typed but cannot be evaluated on this snapshot."""


class FigureEvaluationCancelled(FigureEvaluationError):
    """Evaluation was cooperatively cancelled at a bounded checkpoint."""


class FigureEvaluationDeadlineExceeded(FigureEvaluationError):
    """Evaluation crossed its caller-provided monotonic deadline."""


def _view_bindings_in_schema_order(
    schema: DatasetSchema,
    view: ViewSpec,
) -> tuple[SourceViewBinding, ...]:
    """Separate canonical wire order from declared evaluation order."""

    by_source = {binding.source: binding for binding in view.source_bindings}
    return tuple(
        by_source[source]
        for source in _dataset_sources(schema)
        if source in by_source
    )


def _addresses_in_schema_order(
    schema: DatasetSchema,
    *groups: tuple[AxisAddress, ...],
) -> tuple[AxisAddress, ...]:
    """Merge independently resolved point/tensor addresses by producer order."""

    by_source = {item.source: item for group in groups for item in group}
    return tuple(
        by_source[source]
        for source in _dataset_sources(schema)
        if source in by_source
    )


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
            raise FigureEvaluationDeadlineExceeded(
                "figure evaluation deadline exceeded"
            )


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
        if not entries or any(
            not isinstance(entry, ResolvedDataset) for entry in entries
        ):
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
    row_indices: dict[AxisSourceRef, np.ndarray]
    point_ordinals: np.ndarray
    data_sources: tuple[AxisSourceRef, ...]
    data_indices: dict[AxisSourceRef, tuple[int, ...]]


def evaluate_axis(
    schema: DatasetSchema,
    source: AxisSourceRef,
    indices: tuple[int, ...],
) -> EvaluatedAxis:
    """Evaluate one typed source over an explicit ordered index set."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(source, AxisSourceRef):
        raise TypeError("source must be AxisSourceRef")
    indices = tuple(indices)
    return EvaluatedAxis(
        source,
        _source_name(schema, source),
        _source_role(schema, source),
        _source_unit(schema, source),
        indices,
        tuple(_source_coordinate(schema, source, index) for index in indices),
        _source_coordinate_frame(schema, source),
    )


def _take_axis(array: np.ndarray, choice: Sequence[int] | int, axis: int) -> np.ndarray:
    if isinstance(choice, int):
        selection = [slice(None)] * array.ndim
        selection[axis] = choice
        return array[tuple(selection)]
    indices = tuple(choice)
    if not indices:
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(0, 0)
        return array[tuple(selection)]
    if indices == tuple(range(indices[0], indices[0] + len(indices))):
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(indices[0], indices[0] + len(indices))
        return array[tuple(selection)]
    return np.take(array, np.asarray(indices, dtype=np.intp), axis=axis)


def _prepare_working(
    block: DataBlock,
    view: ViewSpec,
    guard: _EvaluationGuard,
) -> tuple[_WorkingData, tuple[AxisResolution, ...]]:
    schema = block.schema
    try:
        point_ordinals = _resolve_selected_point_ordinals(schema, view)
    except (TypeError, ValueError, IndexError, KeyError) as error:
        raise FigureEvaluationError(str(error)) from error
    values = block.values
    validity = expand_dataset_validity(block.validity, schema)

    repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    repeat_binding = view.binding(repeat_source)
    resolutions: list[AxisResolution] = []

    data_sources = [
        AxisSourceRef.tensor(axis.axis_id) for axis in schema.cell_schema.data_axes
    ]
    data_indices = {
        source: tuple(range(_tensor_axis(schema, source).size))
        for source in data_sources
    }
    for source in tuple(data_sources):
        binding = view.binding(source)
        if not isinstance(binding.selector, FixedIndex):
            continue
        index = binding.selector.index
        axis = 2 + data_sources.index(source)
        values = _take_axis(values, index, axis)
        validity = _take_axis(validity, index, axis)
        data_sources.remove(source)
        del data_indices[source]
        resolutions.append(
            AxisResolution(
                source,
                "FIXED_INDEX",
                index,
                _source_coordinate(schema, source, index),
            )
        )

    values = _take_axis(values, point_ordinals, 1)
    validity = _take_axis(validity, point_ordinals, 1)
    repeat_indices = tuple(range(schema.repeat_axis.size))
    if isinstance(repeat_binding.selector, FixedIndex):
        repeat_indices = (repeat_binding.selector.index,)
        resolutions.append(
            AxisResolution(
                repeat_source,
                "FIXED_INDEX",
                repeat_binding.selector.index,
                _source_coordinate(
                    schema, repeat_source, repeat_binding.selector.index
                ),
            )
        )
    elif isinstance(repeat_binding.selector, LatestNonempty):
        chosen = None
        for index in reversed(repeat_indices):
            guard.check()
            if bool(np.any(validity[index])):
                chosen = index
                break
        if chosen is None:
            raise FigureEvaluationError("repeat has no non-empty display index")
        repeat_indices = (chosen,)
        resolutions.append(
            AxisResolution(
                repeat_source,
                "LATEST_NONEMPTY",
                chosen,
                _source_coordinate(schema, repeat_source, chosen),
            )
        )
    values = _take_axis(values, repeat_indices, 0)
    validity = _take_axis(validity, repeat_indices, 0)

    repeat_count = len(repeat_indices)
    point_count = len(point_ordinals)
    values = np.asarray(values).reshape(
        (repeat_count * point_count, *np.shape(values)[2:])
    )
    validity = np.asarray(validity, dtype=bool).reshape(values.shape)
    repeated = np.repeat(np.asarray(repeat_indices, dtype=np.int64), point_count)
    tiled_points = np.tile(np.asarray(point_ordinals, dtype=np.int64), repeat_count)
    row_indices: dict[AxisSourceRef, np.ndarray] = {}
    if repeat_binding.role is not AxisViewRole.SELECTED:
        row_indices[repeat_source] = repeated
    topology = schema.grid_topology
    for binding in _view_bindings_in_schema_order(schema, view):
        source = binding.source
        if source.kind == AxisSourceRef.TENSOR or binding.role is AxisViewRole.SELECTED:
            continue
        if source.kind == AxisSourceRef.GRID_DIMENSION:
            assert topology is not None and source.axis_id is not None
            position = topology.dimension_ids.index(source.axis_id)
            lookup = np.asarray(
                tuple(cell[position] for cell in topology.row_to_cell),
                dtype=np.int64,
            )
            row_indices[source] = lookup[tiled_points]
        else:
            # PointRows, PointOrdinal, and raw coordinates all retain ordinal
            # identity.  Coordinate values are labels, never a Cartesian index.
            row_indices[source] = tiled_points

    for binding in _view_bindings_in_schema_order(schema, view):
        if (
            binding.source.kind == AxisSourceRef.GRID_DIMENSION
            and isinstance(binding.selector, FixedIndex)
        ):
            resolutions.append(
                AxisResolution(
                    binding.source,
                    "FIXED_INDEX",
                    binding.selector.index,
                    _source_coordinate(
                        schema, binding.source, binding.selector.index
                    ),
                )
            )
    return (
        _WorkingData(
            values,
            validity,
            row_indices,
            tiled_points,
            tuple(data_sources),
            data_indices,
        ),
        tuple(resolutions),
    )


def _slice_rows(working: _WorkingData, mask: np.ndarray) -> _WorkingData:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(working.values),):
        raise FigureEvaluationError("row mask does not align with working data")
    return _WorkingData(
        working.values[mask],
        working.validity[mask],
        {source: values[mask] for source, values in working.row_indices.items()},
        working.point_ordinals[mask],
        working.data_sources,
        working.data_indices,
    )


def _without_row_sources(
    working: _WorkingData,
    sources: tuple[AxisSourceRef, ...],
) -> _WorkingData:
    consumed = frozenset(sources)
    if not consumed:
        return working
    return _WorkingData(
        working.values,
        working.validity,
        {
            source: values
            for source, values in working.row_indices.items()
            if source not in consumed
        },
        working.point_ordinals,
        working.data_sources,
        working.data_indices,
    )


def _slice_data(
    working: _WorkingData,
    source: AxisSourceRef,
    logical_index: int,
) -> _WorkingData:
    if source not in working.data_sources:
        raise FigureEvaluationError(f"data source {source} is absent")
    indices = working.data_indices[source]
    try:
        local_index = indices.index(logical_index)
    except ValueError as exc:
        raise FigureEvaluationError(
            f"index {logical_index} is outside data source {source}"
        ) from exc
    axis = 1 + working.data_sources.index(source)
    values = _take_axis(working.values, local_index, axis)
    validity = _take_axis(working.validity, local_index, axis)
    sources = tuple(item for item in working.data_sources if item != source)
    return _WorkingData(
        values,
        validity,
        working.row_indices,
        working.point_ordinals,
        sources,
        {item: working.data_indices[item] for item in sources},
    )


def _slice_tensor_bindings(
    working: _WorkingData,
    schema: DatasetSchema,
    choices: tuple[tuple[AxisSourceRef, int], ...],
) -> _WorkingData:
    result = working
    repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    for source, index in choices:
        if source == repeat_source:
            column = result.row_indices.get(source)
            if column is None:
                raise FigureEvaluationError("repeat source disappeared before slicing")
            result = _slice_rows(result, column == index)
            result = _without_row_sources(result, (source,))
        else:
            result = _slice_data(result, source, index)
    return result


def _reduction_dtype(dtype: np.dtype, method: DisplayReductionMethod) -> np.dtype:
    return (
        canonical_mean_dtype(dtype)
        if method is DisplayReductionMethod.MEAN
        else canonical_sum_dtype(dtype)
    )


def _reduce_array(
    values: np.ndarray,
    validity: np.ndarray,
    axes: tuple[int, ...],
    method: DisplayReductionMethod,
    output_dtype: np.dtype,
):
    counts = np.sum(validity, axis=axes, dtype=np.int64)
    safe = np.where(validity, values, np.zeros((), dtype=values.dtype))
    if method is DisplayReductionMethod.MEAN:
        safe = safe.astype(output_dtype, copy=False)
    summed = checked_numeric_sum(safe, axes, output_dtype=output_dtype)
    if method is DisplayReductionMethod.MEAN:
        result = np.zeros(np.shape(summed), dtype=output_dtype)
        np.divide(summed, counts, out=result, where=counts > 0)
    else:
        result = np.asarray(summed, dtype=output_dtype)
    valid = counts > 0
    result = np.where(valid, result, np.zeros((), dtype=output_dtype))
    return result, valid, counts


def _reduce(
    working: _WorkingData,
    schema: DatasetSchema,
    view: ViewSpec,
    reduction_bindings: tuple[SourceViewBinding, ...],
    guard: _EvaluationGuard,
) -> tuple[_WorkingData, tuple[ReductionResolution, ...]]:
    if not reduction_bindings:
        return working, ()
    methods = {binding.reduction.method for binding in reduction_bindings}
    if len(methods) != 1:
        raise FigureEvaluationError("joint reductions require one common method")
    method = next(iter(methods))
    reduce_sources = {binding.source for binding in reduction_bindings}
    data_reduce = tuple(
        source for source in working.data_sources if source in reduce_sources
    )
    row_reduce = tuple(
        source for source in working.row_indices if source in reduce_sources
    )
    if reduce_sources - set(data_reduce) - set(row_reduce):
        raise FigureEvaluationError("reduction source disappeared before evaluation")

    surviving_data = tuple(
        source for source in working.data_sources if source not in reduce_sources
    )
    surviving_rows = tuple(
        binding.source
        for binding in _view_bindings_in_schema_order(schema, view)
        if binding.source in working.row_indices
        and binding.source not in reduce_sources
        and binding.role
        in {
            AxisViewRole.X,
            AxisViewRole.IMAGE_X,
            AxisViewRole.IMAGE_Y,
            AxisViewRole.SAMPLE,
        }
    )
    groups: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    if row_reduce:
        for row in range(len(working.values)):
            if row % 4096 == 0:
                guard.check()
            key = tuple(
                int(working.row_indices[source][row]) for source in surviving_rows
            )
            groups.setdefault(key, []).append(row)
    else:
        for row in range(len(working.values)):
            key = tuple(
                int(working.row_indices[source][row]) for source in surviving_rows
            )
            groups[key] = [row]

    maximum_row_contributors = max((len(rows) for rows in groups.values()), default=0)
    maximum_data_contributors = math.prod(
        len(working.data_indices[source]) for source in data_reduce
    )
    maximum_contributors = maximum_row_contributors * maximum_data_contributors
    only_repeat = len(reduction_bindings) == 1 and (
        reduction_bindings[0].source
        == AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    )
    output_dtype = (
        working.values.dtype
        if only_repeat and maximum_contributors <= 1
        else _reduction_dtype(working.values.dtype, method)
    )

    data_axes = tuple(
        1 + working.data_sources.index(source) for source in data_reduce
    )
    output_values = []
    output_validity = []
    output_counts = []
    for rows in groups.values():
        guard.check()
        row_indices = np.asarray(rows, dtype=np.intp)
        values = working.values[row_indices]
        validity = working.validity[row_indices]
        axes = ((0,) if row_reduce else ()) + data_axes
        if not axes:
            result = values[0]
            valid = validity[0]
            counts = valid.astype(np.int64)
        elif maximum_contributors <= 1 and output_dtype == working.values.dtype:
            result = np.asarray(values)
            valid = np.asarray(validity)
            for axis in sorted(axes, reverse=True):
                result = np.take(result, 0, axis=axis)
                valid = np.take(valid, 0, axis=axis)
            counts = valid.astype(np.int64)
        else:
            result, valid, counts = _reduce_array(
                values, validity, axes, method, output_dtype
            )
        output_values.append(np.asarray(result))
        output_validity.append(np.asarray(valid, dtype=bool))
        output_counts.append(np.asarray(counts, dtype=np.int64))

    data_shape = tuple(len(working.data_indices[source]) for source in surviving_data)
    if output_values:
        values_array = np.stack(output_values, axis=0)
        validity_array = np.stack(output_validity, axis=0)
        counts_array = np.stack(output_counts, axis=0)
    else:
        values_array = np.empty((0, *data_shape), dtype=output_dtype)
        validity_array = np.empty((0, *data_shape), dtype=bool)
        counts_array = np.empty((0, *data_shape), dtype=np.int64)
    keys = tuple(groups)
    row_indices = {
        source: np.asarray(
            [key[position] for key in keys],
            dtype=np.int64,
        )
        for position, source in enumerate(surviving_rows)
    }
    if AxisSourceRef.point_rows() in row_indices:
        ordinals = row_indices[AxisSourceRef.point_rows()]
    elif AxisSourceRef.point_ordinal() in row_indices:
        ordinals = row_indices[AxisSourceRef.point_ordinal()]
    else:
        point_source = next(
            (
                source
                for source in row_indices
                if source.kind == AxisSourceRef.POINT_COORDINATE
            ),
            None,
        )
        ordinals = (
            row_indices[point_source]
            if point_source is not None
            else np.zeros(len(values_array), dtype=np.int64)
        )
    flat_counts = counts_array.reshape(-1)
    low = int(flat_counts.min()) if flat_counts.size else 0
    high = int(flat_counts.max()) if flat_counts.size else 0
    resolution = ReductionResolution(
        tuple(binding.source for binding in reduction_bindings),
        method,
        low,
        high,
    )
    return (
        _WorkingData(
            values_array,
            validity_array,
            row_indices,
            np.asarray(ordinals, dtype=np.int64),
            surviving_data,
            {source: working.data_indices[source] for source in surviving_data},
        ),
        (resolution,),
    )


def _row_axis_indices(
    working: _WorkingData,
    source: AxisSourceRef,
) -> tuple[int, ...]:
    values = working.row_indices.get(source)
    if values is None:
        raise FigureEvaluationError(f"row source {source} is absent")
    return tuple(int(value) for value in values)


def _image(
    working: _WorkingData,
    schema: DatasetSchema,
    view: ViewSpec,
    *,
    value_unit: str | None,
) -> EvaluatedImage:
    binding_x = next(
        binding
        for binding in view.source_bindings
        if binding.role is AxisViewRole.IMAGE_X
    )
    binding_y = next(
        binding
        for binding in view.source_bindings
        if binding.role is AxisViewRole.IMAGE_Y
    )
    x_source, y_source = binding_x.source, binding_y.source
    x_data = x_source in working.data_sources
    y_data = y_source in working.data_sources
    x_row = x_source in working.row_indices
    y_row = y_source in working.row_indices

    if x_data and y_data:
        if len(working.values) != 1 or set(working.data_sources) != {
            x_source,
            y_source,
        }:
            raise FigureEvaluationError("image retains unrelated sources")
        x_indices = working.data_indices[x_source]
        y_indices = working.data_indices[y_source]
        order = working.data_sources
        row = working.values[0]
        valid = working.validity[0]
        permutation = (order.index(y_source), order.index(x_source))
        return EvaluatedImage(
            evaluate_axis(schema, x_source, x_indices),
            evaluate_axis(schema, y_source, y_indices),
            np.transpose(row, permutation),
            np.transpose(valid, permutation),
            value_unit,
        )

    if (x_row and y_data) or (y_row and x_data):
        row_source = x_source if x_row else y_source
        data_source = y_source if y_data else x_source
        if working.data_sources != (data_source,):
            raise FigureEvaluationError("mixed image retains unrelated data sources")
        row_indices = _row_axis_indices(working, row_source)
        data_indices = working.data_indices[data_source]
        if row_source is x_source:
            values = np.transpose(working.values, (1, 0))
            validity = np.transpose(working.validity, (1, 0))
            x_indices, y_indices = row_indices, data_indices
        else:
            values = working.values
            validity = working.validity
            x_indices, y_indices = data_indices, row_indices
        return EvaluatedImage(
            evaluate_axis(schema, x_source, x_indices),
            evaluate_axis(schema, y_source, y_indices),
            values,
            validity,
            value_unit,
        )

    if x_row and y_row:
        if (
            x_source.kind != AxisSourceRef.GRID_DIMENSION
            or y_source.kind != AxisSourceRef.GRID_DIMENSION
            or working.data_sources
        ):
            raise FigureEvaluationError(
                "two row image sources require GridTopology dimensions"
            )
        x_indices = tuple(range(_source_cardinality(schema, x_source)))
        y_indices = tuple(range(_source_cardinality(schema, y_source)))
        output = np.zeros((len(y_indices), len(x_indices)), dtype=working.values.dtype)
        valid = np.zeros(output.shape, dtype=bool)
        if working.values.ndim != 1:
            raise FigureEvaluationError("grid image cells must be scalar")
        for row in range(len(working.values)):
            xi = int(working.row_indices[x_source][row])
            yi = int(working.row_indices[y_source][row])
            output[yi, xi] = working.values[row]
            valid[yi, xi] = working.validity[row]
        return EvaluatedImage(
            evaluate_axis(schema, x_source, x_indices),
            evaluate_axis(schema, y_source, y_indices),
            output,
            valid,
            value_unit,
        )
    raise FigureEvaluationError("image display sources disappeared before evaluation")


def _curve(
    working: _WorkingData,
    schema: DatasetSchema,
    view: ViewSpec,
    *,
    value_unit: str | None,
) -> EvaluatedCurve:
    binding = next(
        binding for binding in view.source_bindings if binding.role is AxisViewRole.X
    )
    source = binding.source
    if source in working.row_indices:
        if working.data_sources or working.values.ndim != 1:
            raise FigureEvaluationError("curve row source retains unrelated sources")
        indices = _row_axis_indices(working, source)
        return EvaluatedCurve(
            evaluate_axis(schema, source, indices),
            value_unit,
            working.values,
            working.validity,
        )
    if source in working.data_sources:
        if len(working.values) != 1 or working.data_sources != (source,):
            raise FigureEvaluationError("curve data source retains unrelated sources")
        indices = working.data_indices[source]
        return EvaluatedCurve(
            evaluate_axis(schema, source, indices),
            value_unit,
            working.values[0],
            working.validity[0],
        )
    raise FigureEvaluationError("curve X source disappeared before evaluation")


def _histogram(
    working: _WorkingData,
    schema: DatasetSchema,
    view: ViewSpec,
    guard: _EvaluationGuard,
    *,
    value_unit: str | None,
) -> EvaluatedHistogram:
    del guard
    ordered_bindings = _view_bindings_in_schema_order(schema, view)
    sample_sources = {
        binding.source
        for binding in ordered_bindings
        if binding.role is AxisViewRole.SAMPLE
    }
    remaining = set(working.row_indices) | set(working.data_sources)
    if sample_sources != remaining:
        raise FigureEvaluationError("histogram retains a non-sample source")
    flat_values = working.values.reshape(-1)
    flat_validity = working.validity.reshape(-1)
    keep = np.asarray(flat_validity, dtype=bool)
    sample_sources = tuple(
        binding.source
        for binding in ordered_bindings
        if binding.role is AxisViewRole.SAMPLE
    )
    return EvaluatedHistogram(
        flat_values[keep],
        sample_sources,
        int(len(flat_values) - np.count_nonzero(keep)),
        value_unit,
    )


def _meter(
    working: _WorkingData,
    *,
    value_unit: str | None,
) -> EvaluatedMeter:
    if working.row_indices or working.data_sources:
        raise FigureEvaluationError("meter retains unresolved sources")
    if working.values.size == 0:
        return EvaluatedMeter(0.0, False, value_unit)
    if working.values.size != 1:
        raise FigureEvaluationError("meter evaluation produced more than one value")
    return EvaluatedMeter(
        working.values.item(),
        bool(working.validity.item()),
        value_unit,
    )


def _address(
    schema: DatasetSchema,
    source: AxisSourceRef,
    index: int,
    coordinate: Any | None = None,
) -> AxisAddress:
    return AxisAddress(
        source,
        _source_name(schema, source),
        _source_role(schema, source),
        index,
        _source_coordinate(schema, source, index)
        if coordinate is None
        else coordinate,
    )


def _tensor_choices(
    working: _WorkingData,
    schema: DatasetSchema,
    sources: tuple[AxisSourceRef, ...],
) -> tuple[tuple[tuple[AxisSourceRef, int], ...], ...]:
    repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    domains = []
    for source in sources:
        if source == repeat_source:
            indices = tuple(int(value) for value in working.row_indices[source])
            indices = tuple(dict.fromkeys(indices))
        else:
            indices = working.data_indices[source]
        domains.append(tuple((source, index) for index in indices))
    return tuple(product(*domains)) if domains else ((),)


class FigureEvaluator:
    """Evaluate a FigureDocument without importing a renderer or transform owner."""

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
                raise ValueError(
                    "monotonic_deadline must be finite numeric time or None"
                )
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
            inputs.setdefault(
                layer.dataset_id,
                EvaluatedInput(layer.dataset_id, snapshot.ref),
            )
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
        schema = block.schema
        view = layer.view
        ordered_bindings = _view_bindings_in_schema_order(schema, view)
        working_base, resolutions = _prepare_working(block, view, guard)
        repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)

        tensor_facets = tuple(
            binding.source
            for binding in ordered_bindings
            if binding.role is AxisViewRole.FACET
            and binding.source.kind == AxisSourceRef.TENSOR
        )
        tensor_batches = tuple(
            binding.source
            for binding in ordered_bindings
            if binding.role is AxisViewRole.BATCH
            and binding.source.kind == AxisSourceRef.TENSOR
        )
        reduction_bindings = tuple(
            binding
            for binding in ordered_bindings
            if binding.role is AxisViewRole.REDUCED
        )
        point_records = _resolved_point_group_records(schema, view)
        point_group_sources = tuple(
            binding.source
            for binding in ordered_bindings
            if binding.role in {AxisViewRole.FACET, AxisViewRole.BATCH}
            and binding.source.kind != AxisSourceRef.TENSOR
        )
        point_facets: OrderedDict[tuple[AxisAddress, ...], list[tuple]] = OrderedDict()
        for facet, batch, members, group_index in point_records:
            point_facets.setdefault(facet, []).append(
                (batch, members, group_index)
            )

        tensor_facet_choices = _tensor_choices(
            working_base, schema, tensor_facets
        )
        tensor_batch_choices = _tensor_choices(
            working_base, schema, tensor_batches
        )
        cells = []
        for tensor_facet in tensor_facet_choices:
            facet_working = _slice_tensor_bindings(
                working_base, schema, tensor_facet
            )
            tensor_facet_address = tuple(
                _address(schema, source, index)
                for source, index in tensor_facet
            )
            for point_facet_address, records in point_facets.items():
                series = []
                for point_batch_address, members, _group_index in records:
                    member_set = np.asarray(members, dtype=np.int64)
                    mask = np.isin(facet_working.point_ordinals, member_set)
                    if not bool(np.any(mask)):
                        continue
                    point_working = _without_row_sources(
                        _slice_rows(facet_working, mask),
                        point_group_sources,
                    )
                    for tensor_batch in tensor_batch_choices:
                        guard.check()
                        working = _slice_tensor_bindings(
                            point_working, schema, tensor_batch
                        )
                        working, reduction_records = _reduce(
                            working,
                            schema,
                            view,
                            reduction_bindings,
                            guard,
                        )
                        if view.intent is ViewIntent.IMAGE:
                            data = _image(
                                working,
                                schema,
                                view,
                                value_unit=schema.cell_schema.value_unit,
                            )
                        elif view.intent is ViewIntent.CURVE:
                            data = _curve(
                                working,
                                schema,
                                view,
                                value_unit=schema.cell_schema.value_unit,
                            )
                        elif view.intent is ViewIntent.HISTOGRAM:
                            data = _histogram(
                                working,
                                schema,
                                view,
                                guard,
                                value_unit=schema.cell_schema.value_unit,
                            )
                        elif view.intent is ViewIntent.METER:
                            data = _meter(
                                working,
                                value_unit=schema.cell_schema.value_unit,
                            )
                        else:
                            raise FigureEvaluationError(
                                f"{view.intent.value} has no DatasetSchema path"
                            )
                        tensor_batch_address = tuple(
                            _address(schema, source, index)
                            for source, index in tensor_batch
                        )
                        series.append(
                            EvaluatedSeries(
                                _addresses_in_schema_order(
                                    schema,
                                    point_batch_address,
                                    tensor_batch_address,
                                ),
                                data,
                                reduction_records,
                            )
                        )
                if series:
                    series.sort(
                        key=lambda item: tuple(
                            address.index for address in item.batch_address
                        )
                    )
                    cells.append(
                        EvaluatedCell(
                            _addresses_in_schema_order(
                                schema,
                                point_facet_address,
                                tensor_facet_address,
                            ),
                            tuple(series),
                        )
                    )
        if not cells:
            raise FigureEvaluationError("view resolved no visible cell")
        cells.sort(
            key=lambda item: tuple(
                address.index for address in item.facet_address
            )
        )
        return EvaluatedLayer(
            layer.layer_id,
            layer.dataset_id,
            tuple(cells),
            tuple(sorted(resolutions, key=lambda item: _source_key(item.source))),
        )


__all__ = [
    "FigureEvaluationCancelled",
    "FigureEvaluationDeadlineExceeded",
    "FigureEvaluationError",
    "FigureEvaluator",
    "ResolvedDataset",
    "ResolvedDatasetMap",
    "evaluate_axis",
]
