"""Fit binding and the sole authoritative dataset-to-solver packing path."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .axis import AxisSourceRef, AxisSpec, SCALAR
from .fit_contract import BoundFit, FitProblem, FitResultBatch, FitSpec
from .layout import AxisLayout, AxisLayoutMode
from .schema import DatasetSchema
from .transform import apply_transform
from .value import DatasetRevisionRef, OwnedSnapshot


_CANONICAL_TRAVERSAL_CHUNK_SIZE = 65_536
_POINT_TOKEN = object()


def bind_fit(spec: FitSpec, expected_schema: DatasetSchema) -> BoundFit:
    """Validate a serializable request against one immutable input schema."""

    return BoundFit(spec, expected_schema)


def build_fit_problem(
    bound: BoundFit,
    snapshot: OwnedSnapshot,
    *,
    abort_check: Callable[[], None] | None = None,
) -> FitProblem:
    """Pack valid observations without flattening or reinterpreting R/P authority."""

    if type(bound) is not BoundFit:
        raise TypeError("bound must be BoundFit")
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if abort_check is not None and not callable(abort_check):
        raise TypeError("abort_check must be callable or None")
    _check_abort(abort_check)
    source_fingerprint = bound.spec.committed_transform.source_schema_fingerprint
    if snapshot.ref.schema_fingerprint != source_fingerprint:
        raise ValueError("snapshot schema fingerprint disagrees with BoundFit")
    if snapshot.block.schema != bound.expected_schema:
        raise ValueError("snapshot schema value disagrees with BoundFit expected schema")

    transformed = apply_transform(snapshot, bound.spec.committed_transform)
    schema = transformed.schema
    if schema != bound.effective_schema:
        raise RuntimeError("fit binding and transform execution schemas disagree")
    values = transformed.values
    validity = transformed.expanded_validity()
    _check_abort(abort_check)

    entries, batch_layout = _batch_plan(bound)
    observations_parts: list[np.ndarray] = []
    coordinate_parts: list[list[np.ndarray]] = [
        [] for _ in bound.spec.independent_sources
    ]
    present_counts: list[int] = []
    valid_counts: list[int] = []
    used_counts: list[int] = []

    point_groups = bound._effective_point_groups
    for point_group_index, tensor_batch_indices in entries:
        _check_abort(abort_check)
        row_ids = np.asarray(
            point_groups.group_member_ordinals[point_group_index],
            dtype=np.int64,
        )
        batch_by_source = dict(
            zip(
                (
                    source
                    for source in bound.spec.batch_sources
                    if source.kind == AxisSourceRef.TENSOR
                ),
                tensor_batch_indices,
            )
        )
        observation_view, validity_view, view_tokens = _observation_view(
            bound,
            values,
            validity,
            row_ids,
            batch_by_source,
        )
        dimension_order = _canonical_dimension_order(bound, view_tokens)
        selected = _valid_positions(
            validity_view,
            dimension_order,
            abort_check,
        )
        valid_count = int(selected.size)
        physical_multi = np.unravel_index(
            selected,
            observation_view.shape,
            order="C",
        )
        coordinates = tuple(
            _independent_coordinates(
                bound,
                source,
                row_ids,
                view_tokens,
                physical_multi,
            )
            for source in bound.spec.independent_sources
        )
        usable = np.ones(valid_count, dtype=bool)
        for coordinate in coordinates:
            usable &= np.isfinite(coordinate)
        used = selected[usable]
        observations_parts.append(
            _float64_observations(np.take(observation_view, used))
        )
        for destination, coordinate in zip(coordinate_parts, coordinates):
            destination.append(np.asarray(coordinate[usable], dtype=np.dtype("<f8")))
        present_counts.append(int(observation_view.size))
        valid_counts.append(valid_count)
        used_counts.append(int(used.size))

    packed_observations = _concatenate_float64(observations_parts)
    packed_coordinates = tuple(
        _concatenate_float64(parts) for parts in coordinate_parts
    )
    offsets = np.empty(len(used_counts) + 1, dtype=np.dtype("<i8"))
    offsets[0] = 0
    np.cumsum(np.asarray(used_counts, dtype=np.dtype("<i8")), out=offsets[1:])
    _check_abort(abort_check)

    return FitProblem(
        source_ref=snapshot.ref,
        spec=bound.spec,
        fit_axis_specs=bound.fit_axis_specs,
        batch_axis_specs=bound.batch_axis_specs,
        point_groups=bound.point_groups,
        batch_layout=batch_layout,
        value_unit=schema.cell_schema.value_unit,
        batch_offsets=offsets,
        independent_values=packed_coordinates,
        observations=packed_observations,
        present_observation_counts=np.asarray(present_counts, dtype=np.dtype("<i8")),
        valid_observation_counts=np.asarray(valid_counts, dtype=np.dtype("<i8")),
        used_observation_counts=np.asarray(used_counts, dtype=np.dtype("<i8")),
    )


def validate_fit_result_source_binding(
    result: FitResultBatch,
    source_ref: DatasetRevisionRef,
    source_schema: DatasetSchema,
) -> None:
    """Validate source facts without reading or repacking capture values."""

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("source_ref must be DatasetRevisionRef")
    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    bound = bind_fit(result.spec, source_schema)
    entries, batch_layout = _batch_plan(bound)
    present_counts = tuple(
        _present_observation_count(bound, point_group_index)
        for point_group_index, _tensor_indices in entries
    )
    if result.source_ref != source_ref:
        raise ValueError("fit result source reference differs from source capture")
    if result.fit_axis_specs != bound.fit_axis_specs:
        raise ValueError("fit result axis specifications differ from source schema")
    if result.batch_axis_specs != bound.batch_axis_specs:
        raise ValueError("fit result batch axes differ from source schema")
    if result.point_groups != bound.point_groups:
        raise ValueError("fit result point groups differ from source schema")
    if result.batch_layout != batch_layout:
        raise ValueError("fit result batch layout differs from source schema")
    if result.value_unit != bound.effective_schema.cell_schema.value_unit:
        raise ValueError("fit result value unit differs from source schema")
    if not np.array_equal(
        result.present_observation_counts,
        np.asarray(present_counts, dtype=np.dtype("<i8")),
    ):
        raise ValueError(
            "fit result present_observation_counts differ from source schema"
        )


def _batch_plan(
    bound: BoundFit,
) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], AxisLayout]:
    point_groups = bound._effective_point_groups
    point_sources = point_groups.group_sources
    tensor_sources = tuple(
        source
        for source in bound.spec.batch_sources
        if source.kind == AxisSourceRef.TENSOR
    )
    tensor_axes = tuple(
        bound.batch_axis_specs[bound.spec.batch_sources.index(source)]
        for source in tensor_sources
    )
    tensor_shape = tuple(axis.size for axis in tensor_axes)
    tensor_combinations = tuple(np.ndindex(tensor_shape))
    entries: list[tuple[int, tuple[int, ...]]] = []
    mapping: list[tuple[int, ...]] = []
    for group_index in range(len(point_groups.group_member_ordinals)):
        point_multi = _point_batch_multi(bound, group_index)
        point_by_source = dict(zip(point_sources, point_multi))
        for tensor_multi in tensor_combinations:
            tensor_by_source = dict(zip(tensor_sources, tensor_multi))
            batch_multi = tuple(
                int(
                    tensor_by_source[source]
                    if source.kind == AxisSourceRef.TENSOR
                    else point_by_source[source]
                )
                for source in bound.spec.batch_sources
            )
            entries.append((group_index, tuple(int(item) for item in tensor_multi)))
            mapping.append(batch_multi)
    logical_shape = tuple(axis.size for axis in bound.batch_axis_specs)
    return tuple(entries), _batch_layout_from_mapping(logical_shape, tuple(mapping))


def _point_batch_multi(bound: BoundFit, group_index: int) -> tuple[int, ...]:
    groups = bound._effective_point_groups
    if not groups.group_sources:
        return ()
    result: list[int] = []
    for position, source in enumerate(groups.group_sources):
        if source.kind == AxisSourceRef.POINT_ROWS:
            result.append(group_index)
            continue
        if source.kind == AxisSourceRef.GRID_DIMENSION:
            result.append(int(groups.group_addresses[group_index][position]))
            continue
        axis = bound.batch_axis_specs[bound.spec.batch_sources.index(source)]
        assert axis.coordinates is not None
        value = groups.group_values[group_index][position]
        result.append(axis.coordinates.index(value))
    return tuple(result)


def _batch_layout_from_mapping(
    logical_shape: tuple[int, ...],
    mapping: tuple[tuple[int, ...], ...],
) -> AxisLayout:
    """Retain product structure while preserving genuinely sparse holes."""

    direct = AxisLayout.from_mapping(logical_shape, mapping)
    if direct.mode is not AxisLayoutMode.EXPLICIT or not mapping or len(logical_shape) < 2:
        return direct
    for split in range(1, len(logical_shape)):
        left_mapping = tuple(dict.fromkeys(multi[:split] for multi in mapping))
        right_mapping = tuple(dict.fromkeys(multi[split:] for multi in mapping))
        if tuple(left + right for left in left_mapping for right in right_mapping) != mapping:
            continue
        return AxisLayout.product(
            _batch_layout_from_mapping(logical_shape[:split], left_mapping),
            _batch_layout_from_mapping(logical_shape[split:], right_mapping),
        )
    return direct


def _observation_view(
    bound: BoundFit,
    values: np.ndarray,
    validity: np.ndarray,
    row_ids: np.ndarray,
    batch_by_source: dict[AxisSourceRef, int],
) -> tuple[np.ndarray, np.ndarray, tuple[object, ...]]:
    values = _take_rows(values, row_ids)
    validity = _take_rows(validity, row_ids)
    schema = bound.effective_schema
    requested_sources = {
        *bound.spec.independent_sources,
        *bound.spec.batch_sources,
    }
    repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    selectors: list[int | slice] = []
    tokens: list[object] = []
    if repeat_source in batch_by_source:
        selectors.append(batch_by_source[repeat_source])
    elif schema.repeat_axis.size == 1 and repeat_source not in requested_sources:
        selectors.append(0)
    else:
        selectors.append(slice(None))
        tokens.append(repeat_source)
    selectors.append(slice(None))
    tokens.append(_POINT_TOKEN)
    for axis in schema.cell_schema.data_axes:
        source = AxisSourceRef.tensor(axis.axis_id)
        if axis.role == SCALAR:
            selectors.append(0)
        elif source in batch_by_source:
            selectors.append(batch_by_source[source])
        elif axis.size == 1 and source not in requested_sources:
            selectors.append(0)
        else:
            selectors.append(slice(None))
            tokens.append(source)
    selection = tuple(selectors)
    return values[selection], validity[selection], tuple(tokens)


def _take_rows(array: np.ndarray, row_ids: np.ndarray) -> np.ndarray:
    if row_ids.size and np.array_equal(
        row_ids,
        np.arange(int(row_ids[0]), int(row_ids[0]) + row_ids.size, dtype=np.int64),
    ):
        return array[:, int(row_ids[0]) : int(row_ids[-1]) + 1, ...]
    return np.take(array, row_ids, axis=1)


def _canonical_dimension_order(
    bound: BoundFit,
    view_tokens: tuple[object, ...],
) -> tuple[int, ...]:
    desired: list[object] = []
    point_inserted = False
    for source in bound.spec.independent_sources:
        if source.kind == AxisSourceRef.TENSOR:
            desired.append(source)
        elif not point_inserted:
            desired.append(_POINT_TOKEN)
            point_inserted = True
    if not point_inserted:
        desired.insert(0, _POINT_TOKEN)
    if len(desired) != len(view_tokens) or set(desired) != set(view_tokens):
        raise RuntimeError("Fit source coverage disagrees with the physical Dataset carrier")
    return tuple(view_tokens.index(token) for token in desired)


def _valid_positions(
    validity: np.ndarray,
    dimension_order: tuple[int, ...],
    abort_check: Callable[[], None] | None,
) -> np.ndarray:
    selected_parts: list[np.ndarray] = []
    for start in range(0, validity.size, _CANONICAL_TRAVERSAL_CHUNK_SIZE):
        _check_abort(abort_check)
        stop = min(validity.size, start + _CANONICAL_TRAVERSAL_CHUNK_SIZE)
        canonical_ranks = np.arange(start, stop, dtype=np.int64)
        physical_ranks = _canonical_ranks_to_physical(
            canonical_ranks,
            validity.shape,
            dimension_order,
        )
        local = np.asarray(np.take(validity, physical_ranks), dtype=bool)
        if np.any(local):
            selected_parts.append(physical_ranks[local])
    if not selected_parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(selected_parts).astype(np.int64, copy=False)


def _canonical_ranks_to_physical(
    canonical_ranks: np.ndarray,
    physical_shape: tuple[int, ...],
    dimension_order: tuple[int, ...],
) -> np.ndarray:
    canonical_shape = tuple(physical_shape[axis] for axis in dimension_order)
    canonical_multi = np.unravel_index(canonical_ranks, canonical_shape, order="C")
    physical_multi: list[np.ndarray | None] = [None] * len(physical_shape)
    for canonical_axis, physical_axis in enumerate(dimension_order):
        physical_multi[physical_axis] = canonical_multi[canonical_axis]
    assert all(value is not None for value in physical_multi)
    return np.asarray(
        np.ravel_multi_index(tuple(physical_multi), physical_shape, order="C"),
        dtype=np.int64,
    )


def _independent_coordinates(
    bound: BoundFit,
    source: AxisSourceRef,
    row_ids: np.ndarray,
    view_tokens: tuple[object, ...],
    physical_multi: tuple[np.ndarray, ...],
) -> np.ndarray:
    if source.kind == AxisSourceRef.TENSOR:
        axis = bound.fit_axis_specs[bound.spec.independent_sources.index(source)]
        logical = physical_multi[view_tokens.index(source)]
        return _axis_coordinates(axis, logical)
    row_offsets = physical_multi[view_tokens.index(_POINT_TOKEN)]
    effective_rows = row_ids[row_offsets]
    schema = bound.effective_schema
    if source.kind == AxisSourceRef.POINT_ORDINAL:
        return np.asarray(
            [bound._source_row_members[int(row)][0] for row in effective_rows],
            dtype=np.dtype("<f8"),
        )
    if source.kind == AxisSourceRef.POINT_COORDINATE:
        column = schema.point_table.column(source.axis_id)
        return _optional_float64(column.values, effective_rows)
    if source.kind == AxisSourceRef.GRID_DIMENSION:
        topology = schema.grid_topology
        assert topology is not None and source.axis_id in topology.dimension_ids
        position = topology.dimension_ids.index(source.axis_id)
        values = tuple(
            topology.coordinate_domains[position][topology.row_to_cell[int(row)][position]]
            for row in effective_rows
        )
        return _optional_float64(values, np.arange(len(values), dtype=np.int64))
    raise RuntimeError(f"unsupported Fit independent source {source.kind}")


def _axis_coordinates(axis: AxisSpec, logical_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(logical_indices, dtype=np.int64)
    if axis.coordinates is None:
        return np.asarray(indices, dtype=np.dtype("<f8")) + axis.index_origin
    return _optional_float64(axis.coordinates, indices)


def _optional_float64(values, indices: np.ndarray) -> np.ndarray:
    declared = np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.dtype("<f8"),
    )
    return np.take(declared, indices)


def _present_observation_count(bound: BoundFit, point_group_index: int) -> int:
    point_count = len(
        bound._effective_point_groups.group_member_ordinals[point_group_index]
    )
    tensor_count = math.prod(
        bound.fit_axis_specs[position].size
        for position, source in enumerate(bound.spec.independent_sources)
        if source.kind == AxisSourceRef.TENSOR
    )
    return point_count * tensor_count


def _concatenate_float64(parts: list[np.ndarray]) -> np.ndarray:
    if not parts or not any(part.size for part in parts):
        return np.empty(0, dtype=np.dtype("<f8"))
    return np.concatenate(parts).astype(np.dtype("<f8"), copy=False)


def _float64_observations(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    converted = np.asarray(source, dtype=np.dtype("<f8"))
    if source.dtype.kind in "iu" and np.iinfo(source.dtype).bits > 53:
        with np.errstate(invalid="ignore", over="ignore"):
            round_trip = converted.astype(source.dtype)
        if not np.array_equal(round_trip, source):
            raise ValueError("fit observation is not exactly float64-representable")
    return converted


def _check_abort(abort_check: Callable[[], None] | None) -> None:
    if abort_check is not None:
        abort_check()


__all__ = [
    "bind_fit",
    "validate_fit_result_source_binding",
]
