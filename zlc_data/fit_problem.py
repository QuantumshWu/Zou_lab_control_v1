"""Fit binding and the sole authoritative dataset-to-solver packing path."""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Callable

import numpy as np

from .axis import AxisId, AxisSpec, SCALAR
from .fit_contract import (
    BoundFit,
    FitCoordinateSource,
    FitProblem,
    FitResultBatch,
    FitSpec,
)
from .layout import AxisLayout, AxisLayoutMode
from .schema import DatasetSchema
from .transform import (
    TransformedSchema,
    apply_transform,
)
from .value import DatasetRevisionRef, OwnedSnapshot, expand_dataset_validity


_CANONICAL_TRAVERSAL_CHUNK_SIZE = 65_536


def bind_fit(spec: FitSpec, expected_schema: DatasetSchema) -> BoundFit:
    """Validate a serializable request against one immutable input schema."""

    return BoundFit(spec, expected_schema)


def build_fit_problem(
    bound: BoundFit,
    snapshot: OwnedSnapshot,
    *,
    abort_check: Callable[[], None] | None = None,
) -> FitProblem:
    """Pack valid observations once into a compact ragged solver problem.

    Missing logical batch coordinates are absent from ``batch_layout``.  A present
    batch with no valid observations remains a row with a zero-length packed slice,
    so the solver can report a cell failure without densifying sparse input.
    """

    if type(bound) is not BoundFit:
        raise TypeError("bound must be BoundFit")
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if abort_check is not None and not callable(abort_check):
        raise TypeError("abort_check must be callable or None")
    _check_abort(abort_check)
    if snapshot.ref.schema_fingerprint != bound.spec.input_schema_fingerprint:
        raise ValueError("snapshot schema fingerprint disagrees with BoundFit")
    if snapshot.block.schema != bound.expected_schema:
        raise ValueError("snapshot schema value disagrees with BoundFit expected schema")
    if bound.effective_schema.dtype.kind not in "biuf":
        raise TypeError("fit packing refuses complex observations; select a real component explicitly")

    if bound.spec.committed_transform is None:
        schema = bound.effective_schema
        values = snapshot.block.values.reshape(schema.physical_shape)
        validity = expand_dataset_validity(
            snapshot.block.validity,
            snapshot.block.schema,
        ).reshape(schema.physical_shape)
    else:
        transformed = apply_transform(snapshot, bound.spec.committed_transform)
        schema = transformed.schema
        if schema != bound.effective_schema:
            raise RuntimeError("fit binding and transform execution schemas disagree")
        values = transformed.values
        validity = transformed.expanded_validity()
    _check_abort(abort_check)

    (
        fit_axes,
        batch_axes,
        cell_ids,
        data_ids,
        cell_batch_positions,
        data_batch_positions,
        data_fit_positions,
        scalar_data_positions,
        axis_indices,
        row_groups,
        data_batch_shape,
        batch_layout,
        present_counts,
    ) = _schema_batch_plan(bound, abort_check)
    coordinate_sources = bound.coordinate_sources
    coordinate_source_by_id = dict(zip(bound.spec.fit_axis_ids, coordinate_sources))
    data_dimension_indices = _data_fit_dimension_indices(
        schema,
        data_fit_positions,
        coordinate_source_by_id,
    )
    observations_parts: list[np.ndarray] = []
    coordinate_parts: list[list[np.ndarray]] = [list() for _ in fit_axes]
    valid_counts: list[int] = []
    used_counts: list[int] = []

    for row_list in row_groups.values():
        _check_abort(abort_check)
        row_ids = np.asarray(row_list, dtype=np.int64)
        row_selector = _compact_row_selector(row_ids)
        (
            dimension_order,
            dimension_indices,
        ) = _canonical_observation_order(
            schema,
            row_ids,
            axis_indices,
            bound.spec.fit_axis_ids,
            coordinate_source_by_id,
            data_dimension_indices,
        )
        for data_multi in np.ndindex(data_batch_shape):
            _check_abort(abort_check)
            data_batch_values = {
                data_ids[position]: int(data_multi[index])
                for index, position in enumerate(data_batch_positions)
            }
            data_selectors = tuple(
                (
                    data_batch_values[axis_id]
                    if axis_id in data_batch_values
                    else 0
                    if position in scalar_data_positions
                    else slice(None)
                )
                for position, axis_id in enumerate(data_ids)
            )
            selection = (row_selector, *data_selectors)
            observation_view = values[selection]
            validity_view = validity[selection]
            selected, valid_count = _all_valid_positions(
                observation_view,
                validity_view,
                dimension_order,
                dimension_indices,
                abort_check,
            )
            _check_abort(abort_check)
            used_count = int(selected.size)
            observations_parts.append(
                _float64_observations(np.take(observation_view, selected))
            )
            valid_counts.append(valid_count)
            used_counts.append(used_count)

            packed_indices = np.unravel_index(
                selected,
                observation_view.shape,
                order="C",
            )
            selected_row_offsets = packed_indices[0]
            selected_rows = row_ids[selected_row_offsets]
            for fit_position, (axis, source) in enumerate(
                zip(fit_axes, coordinate_sources)
            ):
                if axis.axis_id in cell_ids:
                    logical = axis_indices[cell_ids.index(axis.axis_id)][selected_rows]
                else:
                    data_position = data_ids.index(axis.axis_id)
                    fit_data_offset = data_fit_positions.index(data_position)
                    logical = packed_indices[1 + fit_data_offset]
                coordinate_parts[fit_position].append(
                    _coordinates_for_indices(axis, source, logical)
                )

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
        fit_axis_specs=fit_axes,
        batch_axis_specs=batch_axes,
        batch_layout=batch_layout,
        value_unit=schema.value_unit,
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
    (
        fit_axes,
        batch_axes,
        _cell_ids,
        _data_ids,
        _cell_batch_positions,
        _data_batch_positions,
        _data_fit_positions,
        _scalar_data_positions,
        _axis_indices,
        _row_groups,
        _data_batch_shape,
        batch_layout,
        present_counts,
    ) = _schema_batch_plan(bound, None)
    if result.source_ref != source_ref:
        raise ValueError("fit result source reference differs from source capture")
    if result.fit_axis_specs != fit_axes:
        raise ValueError("fit result axis specifications differ from source schema")
    if result.batch_axis_specs != batch_axes:
        raise ValueError("fit result batch axes differ from source schema")
    if result.batch_layout != batch_layout:
        raise ValueError("fit result batch layout differs from source schema")
    if result.value_unit != bound.effective_schema.value_unit:
        raise ValueError("fit result value unit differs from source schema")
    if not np.array_equal(
        result.present_observation_counts,
        np.asarray(present_counts, dtype=np.dtype("<i8")),
    ):
        raise ValueError(
            "fit result present_observation_counts differ from source schema"
        )


def _schema_batch_plan(
    bound: BoundFit,
    abort_check: Callable[[], None] | None,
):
    """Return the one schema-derived batch partition used by pack and load."""

    schema = bound.effective_schema
    fit_axes = tuple(schema.axis(axis_id) for axis_id in bound.spec.fit_axis_ids)
    batch_axes = tuple(schema.axis(axis_id) for axis_id in bound.spec.batch_axis_ids)
    cell_ids = tuple(axis.axis_id for axis in schema.cell_axes)
    data_ids = tuple(axis.axis_id for axis in schema.data_axes)
    cell_batch_positions = tuple(
        position
        for position, axis_id in enumerate(cell_ids)
        if axis_id in bound.spec.batch_axis_ids
    )
    data_batch_positions = tuple(
        position
        for position, axis_id in enumerate(data_ids)
        if axis_id in bound.spec.batch_axis_ids
    )
    data_fit_positions = tuple(
        position
        for position, axis_id in enumerate(data_ids)
        if axis_id in bound.spec.fit_axis_ids
    )
    scalar_data_positions = tuple(
        position
        for position, axis in enumerate(schema.data_axes)
        if axis.role == SCALAR
    )
    axis_indices = tuple(
        schema.cell_layout.axis_indices(position)
        for position in range(len(schema.cell_axes))
    )
    data_batch_shape = tuple(
        schema.data_axes[position].size for position in data_batch_positions
    )
    row_groups: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    for row in range(schema.cell_layout.storage_size):
        if row % 1024 == 0:
            _check_abort(abort_check)
        key = tuple(
            int(axis_indices[position][row])
            for position in cell_batch_positions
        )
        row_groups.setdefault(key, []).append(row)

    mapping: list[tuple[int, ...]] = []
    present_counts: list[int] = []
    fit_data_count = math.prod(
        schema.data_axes[position].size for position in data_fit_positions
    )
    for cell_batch_key, row_list in row_groups.items():
        cell_batch_values = {
            cell_ids[position]: cell_batch_key[index]
            for index, position in enumerate(cell_batch_positions)
        }
        for data_multi in np.ndindex(data_batch_shape):
            data_batch_values = {
                data_ids[position]: int(data_multi[index])
                for index, position in enumerate(data_batch_positions)
            }
            batch_multi = tuple(
                cell_batch_values.get(axis_id, data_batch_values.get(axis_id))
                for axis_id in bound.spec.batch_axis_ids
            )
            if any(value is None for value in batch_multi):
                raise RuntimeError("batch axis partition failed")
            mapping.append(tuple(int(value) for value in batch_multi))
            present_counts.append(len(row_list) * fit_data_count)
    batch_shape = tuple(axis.size for axis in batch_axes)
    batch_layout = _batch_layout_from_mapping(batch_shape, tuple(mapping))
    return (
        fit_axes,
        batch_axes,
        cell_ids,
        data_ids,
        cell_batch_positions,
        data_batch_positions,
        data_fit_positions,
        scalar_data_positions,
        axis_indices,
        row_groups,
        data_batch_shape,
        batch_layout,
        tuple(present_counts),
    )


def _batch_layout_from_mapping(
    logical_shape: tuple[int, ...],
    mapping: tuple[tuple[int, ...], ...],
) -> AxisLayout:
    """Retain C/F/product structure while preserving truly sparse holes."""

    direct = AxisLayout.from_mapping(logical_shape, mapping)
    if direct.mode is not AxisLayoutMode.EXPLICIT or not mapping or len(logical_shape) < 2:
        return direct
    for split in range(1, len(logical_shape)):
        left_mapping = tuple(dict.fromkeys(multi[:split] for multi in mapping))
        right_mapping = tuple(dict.fromkeys(multi[split:] for multi in mapping))
        product_mapping = tuple(
            left + right for left in left_mapping for right in right_mapping
        )
        if product_mapping != mapping:
            continue
        left = _batch_layout_from_mapping(logical_shape[:split], left_mapping)
        right = _batch_layout_from_mapping(logical_shape[split:], right_mapping)
        return AxisLayout.product(left, right)
    return direct


def _coordinates_for_indices(
    axis: AxisSpec,
    source: FitCoordinateSource,
    logical_indices: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(logical_indices, dtype=np.int64)
    if source is FitCoordinateSource.LOGICAL_INDEX:
        return np.asarray(indices, dtype=np.float64) + axis.index_origin
    assert axis.coordinates is not None
    values = np.fromiter(
        (float(axis.coordinates[int(index)]) for index in indices.reshape(-1)),
        dtype=np.dtype("<f8"),
        count=indices.size,
    )
    return values.reshape(indices.shape)


def _compact_row_selector(row_ids: np.ndarray) -> slice | np.ndarray:
    if row_ids.size and np.array_equal(
        row_ids,
        np.arange(int(row_ids[0]), int(row_ids[0]) + row_ids.size, dtype=np.int64),
    ):
        return slice(int(row_ids[0]), int(row_ids[-1]) + 1)
    return row_ids


def _all_valid_positions(
    observations: np.ndarray,
    validity: np.ndarray,
    dimension_order: tuple[int, ...],
    dimension_indices: tuple[np.ndarray | None, ...],
    abort_check: Callable[[], None] | None,
) -> tuple[np.ndarray, int]:
    selected_parts: list[np.ndarray] = []
    for start in range(0, observations.size, _CANONICAL_TRAVERSAL_CHUNK_SIZE):
        _check_abort(abort_check)
        stop = min(observations.size, start + _CANONICAL_TRAVERSAL_CHUNK_SIZE)
        canonical_ranks = np.arange(start, stop, dtype=np.int64)
        physical_ranks = _canonical_ranks_to_physical(
            canonical_ranks,
            observations.shape,
            dimension_order,
            dimension_indices,
        )
        local_validity = np.asarray(np.take(validity, physical_ranks), dtype=bool)
        if np.any(local_validity):
            selected_parts.append(physical_ranks[local_validity])
    selected = (
        np.concatenate(selected_parts).astype(np.int64, copy=False)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    return selected, int(selected.size)


def _canonical_observation_order(
    schema: TransformedSchema,
    row_ids: np.ndarray,
    axis_indices: tuple[np.ndarray, ...],
    fit_axis_ids: tuple[AxisId, ...],
    coordinate_source_by_id: dict[AxisId, FitCoordinateSource],
    data_dimension_indices: dict[AxisId, np.ndarray | None],
) -> tuple[
    tuple[int, ...],
    tuple[np.ndarray | None, ...],
]:
    """Describe logical fit-axis order without allocating full coordinate grids."""

    cell_ids = tuple(axis.axis_id for axis in schema.cell_axes)
    data_ids = tuple(axis.axis_id for axis in schema.data_axes)
    cell_fit_ids = tuple(axis_id for axis_id in fit_axis_ids if axis_id in cell_ids)
    row_keys = tuple(
        _coordinates_for_indices(
            schema.axis(axis_id),
            coordinate_source_by_id[axis_id],
            axis_indices[cell_ids.index(axis_id)][row_ids],
        )
        for axis_id in cell_fit_ids
    )
    logical_row_keys = tuple(
        axis_indices[cell_ids.index(axis_id)][row_ids]
        for axis_id in cell_fit_ids
    )
    if not row_keys:
        row_order = np.arange(row_ids.size, dtype=np.int64)
    else:
        # Declared coordinates define the physical ordering; logical indices are
        # the deterministic tie-break when coordinates repeat.  Physical storage
        # row order must never influence the packed observation order.
        row_order = np.lexsort(
            tuple(reversed((*row_keys, *logical_row_keys)))
        ).astype(np.int64, copy=False)

    row_token = "__fit_cell_rows__"
    current_tokens: list[AxisId | str] = [row_token]
    current_tokens.extend(axis_id for axis_id in data_ids if axis_id in fit_axis_ids)
    if cell_fit_ids:
        desired_tokens: list[AxisId | str] = []
        inserted_rows = False
        for axis_id in fit_axis_ids:
            if axis_id in cell_ids:
                if not inserted_rows:
                    desired_tokens.append(row_token)
                    inserted_rows = True
            else:
                desired_tokens.append(axis_id)
    else:
        desired_tokens = [row_token, *fit_axis_ids]
    dimension_order = tuple(current_tokens.index(token) for token in desired_tokens)
    dimension_indices: list[np.ndarray | None] = []
    for token in desired_tokens:
        if token == row_token:
            dimension_indices.append(row_order)
        else:
            dimension_indices.append(data_dimension_indices[token])
    return dimension_order, tuple(dimension_indices)


def _data_fit_dimension_indices(
    schema: TransformedSchema,
    data_fit_positions: tuple[int, ...],
    coordinate_source_by_id: dict[AxisId, FitCoordinateSource],
) -> dict[AxisId, np.ndarray | None]:
    """Resolve dense data-axis order once per packing operation."""

    resolved: dict[AxisId, np.ndarray | None] = {}
    for position in data_fit_positions:
        axis = schema.data_axes[position]
        source = coordinate_source_by_id[axis.axis_id]
        if source is FitCoordinateSource.LOGICAL_INDEX:
            resolved[axis.axis_id] = None
            continue
        values = _coordinates_for_indices(
            axis,
            source,
            np.arange(axis.size, dtype=np.int64),
        )
        resolved[axis.axis_id] = (
            None
            if values.size < 2 or np.all(values[:-1] <= values[1:])
            else np.argsort(values, kind="stable").astype(np.int64, copy=False)
        )
    return resolved


def _canonical_ranks_to_physical(
    canonical_ranks: np.ndarray,
    physical_shape: tuple[int, ...],
    dimension_order: tuple[int, ...],
    dimension_indices: tuple[np.ndarray | None, ...],
) -> np.ndarray:
    if canonical_ranks.size == 0:
        return np.empty(0, dtype=np.int64)
    canonical_shape = tuple(physical_shape[axis] for axis in dimension_order)
    canonical_multi = np.unravel_index(canonical_ranks, canonical_shape, order="C")
    physical_multi: list[np.ndarray | None] = [None] * len(physical_shape)
    for canonical_axis, physical_axis in enumerate(dimension_order):
        index_order = dimension_indices[canonical_axis]
        physical_multi[physical_axis] = (
            canonical_multi[canonical_axis]
            if index_order is None
            else index_order[canonical_multi[canonical_axis]]
        )
    assert all(value is not None for value in physical_multi)
    return np.asarray(
        np.ravel_multi_index(tuple(physical_multi), physical_shape, order="C"),
        dtype=np.int64,
    )


def _concatenate_float64(parts: list[np.ndarray]) -> np.ndarray:
    if not parts or not any(part.size for part in parts):
        return np.empty(0, dtype=np.dtype("<f8"))
    return np.concatenate(parts).astype(np.dtype("<f8"), copy=False)


def _float64_observations(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    converted = np.asarray(source, dtype=np.float64)
    if source.dtype.kind in "iu" and any(
        int(float(value)) != int(value) for value in source.reshape(-1)
    ):
        raise ValueError("fit observation is not exactly float64-representable")
    return converted


def _check_abort(abort_check: Callable[[], None] | None) -> None:
    if abort_check is not None:
        abort_check()


__all__ = [
    "bind_fit",
    "validate_fit_result_source_binding",
]
