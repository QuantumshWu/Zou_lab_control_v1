"""Fit binding and the sole authoritative dataset-to-solver packing path."""

from __future__ import annotations

from collections import OrderedDict
from itertools import product
import math
from typing import Callable

import numpy as np

from .axis import AxisId, AxisSpec
from .fit_contract import (
    BoundFit,
    FitCoordinateSource,
    FitProblem,
    FitResultBatch,
    FitSpec,
    _coordinate_source_for_axis,
)
from .layout import AxisLayout, AxisLayoutMode
from .schema import DatasetSchema
from .transform import (
    TransformedSchema,
    apply_transform,
)
from .value import DatasetRevisionRef, OwnedSnapshot, expand_dataset_validity


_SAMPLING_CHUNK_SIZE = 65_536
_FEATURE_SAMPLE_LIMIT = 64
_FEATURE_CANDIDATE_LIMIT = 512


def bind_fit(spec: FitSpec, expected_schema: DatasetSchema) -> BoundFit:
    """Validate a serializable request against one immutable input schema."""

    return BoundFit.bind(spec, expected_schema)


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

    if not isinstance(bound, BoundFit):
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
        axis_indices,
        row_groups,
        data_combinations,
        batch_layout,
        present_counts,
    ) = _schema_batch_plan(bound, abort_check)
    coordinate_sources = tuple(_coordinate_source_for_axis(axis) for axis in fit_axes)
    observations_parts: list[np.ndarray] = []
    coordinate_parts: list[list[np.ndarray]] = [list() for _ in fit_axes]
    valid_counts: list[int] = []
    used_counts: list[int] = []
    packed_observation_count = 0

    for row_list in row_groups.values():
        _check_abort(abort_check)
        row_ids = np.asarray(row_list, dtype=np.int64)
        row_selector = _compact_row_selector(row_ids)
        (
            dimension_order,
            dimension_indices,
            compressed_fit_coordinates,
        ) = _canonical_observation_order(
            schema,
            row_ids,
            axis_indices,
            data_fit_positions,
            bound.spec.fit_axis_ids,
        )
        for data_multi in data_combinations:
            _check_abort(abort_check)
            data_batch_values = {
                data_ids[position]: int(data_multi[index])
                for index, position in enumerate(data_batch_positions)
            }
            data_selectors = tuple(
                data_batch_values.get(axis_id, slice(None)) for axis_id in data_ids
            )
            selection = (row_selector, *data_selectors)
            observation_view = values[selection]
            validity_view = validity[selection]
            selected, valid_count = _sample_valid_positions(
                observation_view,
                validity_view,
                bound.spec.numeric_policy.sample_budget_per_batch,
                dimension_order,
                dimension_indices,
                compressed_fit_coordinates,
                abort_check,
            )
            _check_abort(abort_check)
            used_count = int(selected.size)
            next_packed_observation_count = packed_observation_count + used_count
            if (
                next_packed_observation_count
                > bound.spec.numeric_policy.max_packed_observations
            ):
                raise ValueError(
                    "fit request exceeds max_packed_observations="
                    f"{bound.spec.numeric_policy.max_packed_observations}"
                )
            packed_observation_count = next_packed_observation_count
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
        _axis_indices,
        _row_groups,
        _data_combinations,
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
    axis_indices = tuple(
        schema.cell_layout.axis_indices(position)
        for position in range(len(schema.cell_axes))
    )
    data_batch_shape = tuple(
        schema.data_axes[position].size for position in data_batch_positions
    )
    data_batch_count = math.prod(data_batch_shape)
    if data_batch_count > bound.spec.numeric_policy.max_batch_cells:
        raise ValueError(
            "data batch axes alone exceed max_batch_cells="
            f"{bound.spec.numeric_policy.max_batch_cells}"
        )
    max_cell_groups = bound.spec.numeric_policy.max_batch_cells // data_batch_count
    row_groups: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    for row in range(schema.cell_layout.storage_size):
        if row % 1024 == 0:
            _check_abort(abort_check)
        key = tuple(
            int(axis_indices[position][row])
            for position in cell_batch_positions
        )
        if key not in row_groups and len(row_groups) >= max_cell_groups:
            raise ValueError(
                "cell batch axes exceed max_batch_cells="
                f"{bound.spec.numeric_policy.max_batch_cells}"
            )
        row_groups.setdefault(key, []).append(row)

    batch_count = len(row_groups) * data_batch_count
    if batch_count > bound.spec.numeric_policy.max_batch_cells:
        raise ValueError(
            f"fit request creates {batch_count} batch cells, exceeding "
            f"max_batch_cells={bound.spec.numeric_policy.max_batch_cells}"
        )

    mapping: list[tuple[int, ...]] = []
    present_counts: list[int] = []
    data_combinations = tuple(np.ndindex(data_batch_shape))
    fit_data_count = math.prod(
        schema.data_axes[position].size for position in data_fit_positions
    )
    for cell_batch_key, row_list in row_groups.items():
        cell_batch_values = {
            cell_ids[position]: cell_batch_key[index]
            for index, position in enumerate(cell_batch_positions)
        }
        for data_multi in data_combinations:
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
        axis_indices,
        row_groups,
        data_combinations,
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
    if source is FitCoordinateSource.LOGICAL_INDEX:
        return np.asarray(logical_indices, dtype=np.float64)
    assert axis.coordinates is not None
    declared = np.asarray(axis.coordinates, dtype=np.float64)
    return np.asarray(declared[logical_indices], dtype=np.float64)


def _compact_row_selector(row_ids: np.ndarray) -> slice | np.ndarray:
    if row_ids.size and np.array_equal(
        row_ids,
        np.arange(int(row_ids[0]), int(row_ids[0]) + row_ids.size, dtype=np.int64),
    ):
        return slice(int(row_ids[0]), int(row_ids[-1]) + 1)
    return row_ids


def _sample_valid_positions(
    observations: np.ndarray,
    validity: np.ndarray,
    budget: int,
    dimension_order: tuple[int, ...],
    dimension_indices: tuple[np.ndarray, ...],
    compressed_fit_coordinates: tuple[np.ndarray, np.ndarray] | None,
    abort_check: Callable[[], None] | None,
) -> tuple[np.ndarray, int]:
    valid_count = int(np.count_nonzero(validity))
    if valid_count == 0:
        return np.empty(0, dtype=np.int64), 0

    used_count = min(valid_count, budget)
    canonical_shape = tuple(observations.shape[axis] for axis in dimension_order)
    bad_ranks, nonfinite_count, extrema = _scan_canonical_observations(
        observations,
        validity,
        used_count,
        dimension_order,
        dimension_indices,
        abort_check,
    )
    if bad_ranks.size >= used_count:
        canonical_ranks = bad_ranks[:used_count]
        selected = _canonical_ranks_to_physical(
            canonical_ranks,
            observations.shape,
            dimension_order,
            dimension_indices,
        )
        return selected, valid_count

    remaining = used_count - bad_ranks.size
    chosen_parts: list[np.ndarray] = []

    # Uniform grids can miss a narrow peak or dip entirely.  Global extrema and
    # their small deterministic neighbourhood are a bounded seed, not a hidden
    # reduction: the packed observations still retain their original coordinates.
    feature_limit = min(
        remaining,
        _FEATURE_SAMPLE_LIMIT,
        max(min(2, remaining), remaining // 16),
    )
    feature_candidates = _extremum_neighborhood_ranks(
        extrema,
        canonical_shape,
        include_neighborhood=compressed_fit_coordinates is None,
    )
    feature_ranks = _finite_candidate_ranks(
        feature_candidates,
        observations,
        validity,
        dimension_order,
        dimension_indices,
    )[:feature_limit]
    if feature_ranks.size:
        chosen_parts.append(feature_ranks)
        remaining -= feature_ranks.size

    if remaining:
        # Recompute the Cartesian grid for its own quota.  Truncating a grid
        # built for the original budget would bias a small remainder toward a
        # flattened diagonal.
        preferred = _preferred_canonical_ranks(
            canonical_shape,
            remaining,
            compressed_fit_coordinates,
        )
        preferred_finite = _finite_candidate_ranks(
            preferred,
            observations,
            validity,
            dimension_order,
            dimension_indices,
        )
        if feature_ranks.size:
            preferred_finite = preferred_finite[
                ~np.isin(preferred_finite, feature_ranks, assume_unique=True)
            ]
        if preferred_finite.size > remaining:
            preferred_finite = preferred_finite[
                _evenly_spaced_ranks(preferred_finite.size, remaining)
            ]
        if preferred_finite.size:
            chosen_parts.append(preferred_finite)
            remaining -= preferred_finite.size

    if remaining:
        chosen_finite = np.sort(np.concatenate(chosen_parts))
        available_count = valid_count - nonfinite_count - chosen_finite.size
        if available_count < remaining:  # pragma: no cover - partition invariant
            raise RuntimeError("finite fit sampling candidates were lost")
        ordinals = _evenly_spaced_ranks(available_count, remaining)
        chosen_parts.append(
            _available_ranks_at_ordinals(
                observations,
                validity,
                chosen_finite,
                ordinals,
                dimension_order,
                dimension_indices,
                abort_check,
            )
        )

    canonical_ranks = np.concatenate((bad_ranks, *chosen_parts))
    canonical_ranks.sort()
    if canonical_ranks.size != used_count or (
        canonical_ranks.size > 1
        and np.any(canonical_ranks[1:] <= canonical_ranks[:-1])
    ):  # pragma: no cover - internal partition invariant
        raise RuntimeError("fit sampler did not produce one unique rank per budget slot")
    selected = _canonical_ranks_to_physical(
        canonical_ranks,
        observations.shape,
        dimension_order,
        dimension_indices,
    )
    return selected, valid_count


def _scan_canonical_observations(
    observations: np.ndarray,
    validity: np.ndarray,
    limit: int,
    dimension_order: tuple[int, ...],
    dimension_indices: tuple[np.ndarray, ...],
    abort_check: Callable[[], None] | None,
) -> tuple[np.ndarray, int, tuple[int, ...]]:
    """Find bad ranks and finite extrema without materializing canonical data."""

    bad_parts: list[np.ndarray] = []
    nonfinite_count = 0
    maximum_rank: int | None = None
    minimum_rank: int | None = None
    maximum_value: int | float | np.generic | None = None
    minimum_value: int | float | np.generic | None = None
    for start in range(0, observations.size, _SAMPLING_CHUNK_SIZE):
        _check_abort(abort_check)
        stop = min(observations.size, start + _SAMPLING_CHUNK_SIZE)
        canonical_ranks = np.arange(start, stop, dtype=np.int64)
        physical_ranks = _canonical_ranks_to_physical(
            canonical_ranks,
            observations.shape,
            dimension_order,
            dimension_indices,
        )
        values = np.take(observations, physical_ranks)
        local_validity = np.asarray(np.take(validity, physical_ranks), dtype=bool)
        finite = np.isfinite(values)
        bad = np.flatnonzero(local_validity & ~finite)
        if not bad.size:
            pass
        else:
            take = min(int(bad.size), limit - nonfinite_count)
            if take:
                bad_parts.append(np.asarray(bad[:take] + start, dtype=np.int64))
            nonfinite_count += int(bad.size)
            if nonfinite_count >= limit:
                return (
                    np.concatenate(bad_parts).astype(np.int64, copy=False),
                    nonfinite_count,
                    (),
                )

        finite_positions = np.flatnonzero(local_validity & finite)
        if not finite_positions.size:
            continue
        finite_values = values[finite_positions]
        local_maximum = np.max(finite_values)
        local_minimum = np.min(finite_values)
        local_maximum_rank = start + int(
            finite_positions[int(np.flatnonzero(finite_values == local_maximum)[0])]
        )
        local_minimum_rank = start + int(
            finite_positions[int(np.flatnonzero(finite_values == local_minimum)[0])]
        )
        if maximum_rank is None or local_maximum > maximum_value:
            maximum_rank = local_maximum_rank
            maximum_value = local_maximum
        if minimum_rank is None or local_minimum < minimum_value:
            minimum_rank = local_minimum_rank
            minimum_value = local_minimum

    bad_ranks = (
        np.concatenate(bad_parts).astype(np.int64, copy=False)
        if bad_parts
        else np.empty(0, dtype=np.int64)
    )
    extrema = tuple(
        dict.fromkeys(
            rank for rank in (maximum_rank, minimum_rank) if rank is not None
        )
    )
    return bad_ranks, nonfinite_count, extrema


def _extremum_neighborhood_ranks(
    extrema: tuple[int, ...],
    canonical_shape: tuple[int, ...],
    *,
    include_neighborhood: bool,
) -> np.ndarray:
    """Return extrema first, then interleaved radius-one/two candidates."""

    if not extrema:
        return np.empty(0, dtype=np.int64)

    center_multi = tuple(
        tuple(int(value) for value in np.unravel_index(rank, canonical_shape, order="C"))
        for rank in extrema
    )
    active_axes = frozenset(
        axis for axis, size in enumerate(canonical_shape) if size > 1
    )
    selected: list[int] = []
    seen: set[int] = set()

    def append_multi(multi: tuple[int, ...]) -> bool:
        rank = int(np.ravel_multi_index(multi, canonical_shape, order="C"))
        if rank not in seen:
            seen.add(rank)
            selected.append(rank)
        return len(selected) == _FEATURE_CANDIDATE_LIMIT

    for multi in center_multi:
        if append_multi(multi):
            return np.asarray(selected, dtype=np.int64)
    if not include_neighborhood:
        return np.asarray(selected, dtype=np.int64)
    for radius in (1, 2):
        neighbourhoods: list[tuple[tuple[int, ...], ...]] = []
        for multi in center_multi:
            ranges = tuple(
                range(
                    max(0, coordinate - radius),
                    min(canonical_shape[axis], coordinate + radius + 1),
                )
                if axis in active_axes
                else (coordinate,)
                for axis, coordinate in enumerate(multi)
            )
            neighbourhoods.append(tuple(product(*ranges)))
        for offset in range(max(map(len, neighbourhoods))):
            for neighbourhood in neighbourhoods:
                if offset >= len(neighbourhood):
                    continue
                neighbour = neighbourhood[offset]
                if append_multi(tuple(neighbour)):
                    return np.asarray(selected, dtype=np.int64)
    return np.asarray(selected, dtype=np.int64)


def _finite_candidate_ranks(
    candidate_ranks: np.ndarray,
    observations: np.ndarray,
    validity: np.ndarray,
    dimension_order: tuple[int, ...],
    dimension_indices: tuple[np.ndarray, ...],
) -> np.ndarray:
    if not candidate_ranks.size:
        return np.empty(0, dtype=np.int64)
    physical_ranks = _canonical_ranks_to_physical(
        candidate_ranks,
        observations.shape,
        dimension_order,
        dimension_indices,
    )
    finite_validity = np.asarray(np.take(validity, physical_ranks), dtype=bool)
    finite_validity &= np.isfinite(np.take(observations, physical_ranks))
    return np.asarray(candidate_ranks[finite_validity], dtype=np.int64)


def _available_ranks_at_ordinals(
    observations: np.ndarray,
    validity: np.ndarray,
    excluded_ranks: np.ndarray,
    ordinals: np.ndarray,
    dimension_order: tuple[int, ...],
    dimension_indices: tuple[np.ndarray, ...],
    abort_check: Callable[[], None] | None,
) -> np.ndarray:
    """Map finite-valid ordinals to ranks with bounded chunk temporaries."""

    result = np.empty(ordinals.size, dtype=np.int64)
    cursor = 0
    seen = 0
    for start in range(0, observations.size, _SAMPLING_CHUNK_SIZE):
        _check_abort(abort_check)
        stop = min(observations.size, start + _SAMPLING_CHUNK_SIZE)
        canonical_ranks = np.arange(start, stop, dtype=np.int64)
        physical_ranks = _canonical_ranks_to_physical(
            canonical_ranks,
            observations.shape,
            dimension_order,
            dimension_indices,
        )
        available = np.asarray(np.take(validity, physical_ranks), dtype=bool)
        available &= np.isfinite(np.take(observations, physical_ranks))
        excluded_start = int(np.searchsorted(excluded_ranks, start, side="left"))
        excluded_stop = int(np.searchsorted(excluded_ranks, stop, side="left"))
        available[excluded_ranks[excluded_start:excluded_stop] - start] = False
        local = np.flatnonzero(available)
        next_seen = seen + int(local.size)
        end = int(np.searchsorted(ordinals, next_seen, side="left"))
        if end > cursor:
            result[cursor:end] = start + local[ordinals[cursor:end] - seen]
            cursor = end
            if cursor == ordinals.size:
                return result
        seen = next_seen
    raise RuntimeError("true-rank ordinal exceeds available candidates")


def _preferred_canonical_ranks(
    canonical_shape: tuple[int, ...],
    budget: int,
    compressed_fit_coordinates: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Choose a coordinate-stratified grid for two-dimensional fit geometry."""

    if compressed_fit_coordinates is not None:
        return _preferred_sparse_point_grid(compressed_fit_coordinates, budget)
    active = tuple(index for index, size in enumerate(canonical_shape) if size > 1)
    if len(active) != 2 or budget < 4:
        return _evenly_spaced_ranks(math.prod(canonical_shape), budget)
    first_axis, second_axis = active
    first_size = canonical_shape[first_axis]
    second_size = canonical_shape[second_axis]
    counts = _balanced_grid_counts(first_size, second_size, budget)
    if counts is None:
        return _evenly_spaced_ranks(math.prod(canonical_shape), budget)
    first_count, second_count = counts
    first = _evenly_spaced_ranks(first_size, first_count)
    second = _evenly_spaced_ranks(second_size, second_count)
    grid_first, grid_second = np.meshgrid(first, second, indexing="ij")
    multi = [np.zeros(grid_first.size, dtype=np.int64) for _ in canonical_shape]
    multi[first_axis] = grid_first.reshape(-1)
    multi[second_axis] = grid_second.reshape(-1)
    return np.asarray(
        np.ravel_multi_index(tuple(multi), canonical_shape, order="C"),
        dtype=np.int64,
    )


def _preferred_sparse_point_grid(
    coordinates: tuple[np.ndarray, np.ndarray],
    budget: int,
) -> np.ndarray:
    first_values, second_values = coordinates
    unique_first = np.unique(first_values)
    unique_second = np.unique(second_values)
    counts = _balanced_grid_counts(unique_first.size, unique_second.size, budget)
    if counts is None:
        return _evenly_spaced_ranks(first_values.size, min(first_values.size, budget))
    first_count, second_count = counts
    selected_first = unique_first[
        _evenly_spaced_ranks(unique_first.size, first_count)
    ]
    selected_second = unique_second[
        _evenly_spaced_ranks(unique_second.size, second_count)
    ]
    # row_order is lexicographic first/second coordinate order.  Binary-search
    # the bounded target grid instead of constructing an N-entry Python dict.
    selected: list[int] = []
    for first in selected_first:
        first_start = int(np.searchsorted(first_values, first, side="left"))
        first_stop = int(np.searchsorted(first_values, first, side="right"))
        second_slice = second_values[first_start:first_stop]
        for second in selected_second:
            offset = int(np.searchsorted(second_slice, second, side="left"))
            if offset < second_slice.size and second_slice[offset] == second:
                selected.append(first_start + offset)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _balanced_grid_counts(
    first_size: int,
    second_size: int,
    budget: int,
) -> tuple[int, int] | None:
    best: tuple[float, int, int, int] | None = None
    maximum_first = min(first_size, budget // 2)
    for first_count in range(2, maximum_first + 1):
        second_count = min(second_size, budget // first_count)
        if second_count < 2:
            continue
        imbalance = abs(
            math.log(
                (first_count / first_size) / (second_count / second_size)
            )
        )
        product = first_count * second_count
        candidate = (imbalance, -product, first_count, second_count)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best[2], best[3]


def _canonical_observation_order(
    schema: TransformedSchema,
    row_ids: np.ndarray,
    axis_indices: tuple[np.ndarray, ...],
    data_fit_positions: tuple[int, ...],
    fit_axis_ids: tuple[AxisId, ...],
) -> tuple[
    tuple[int, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, np.ndarray] | None,
]:
    """Describe logical fit-axis order without allocating full coordinate grids."""

    cell_ids = tuple(axis.axis_id for axis in schema.cell_axes)
    data_ids = tuple(axis.axis_id for axis in schema.data_axes)
    cell_fit_ids = tuple(axis_id for axis_id in fit_axis_ids if axis_id in cell_ids)
    row_keys = tuple(
        _resolved_axis_values(schema.axis(axis_id))[
            axis_indices[cell_ids.index(axis_id)][row_ids]
        ]
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
        # row order must never influence which y values survive sampling.
        row_order = np.lexsort(
            tuple(reversed((*row_keys, *logical_row_keys)))
        ).astype(np.int64, copy=False)

    row_token = "__fit_cell_rows__"
    current_tokens: list[AxisId | str] = [row_token]
    current_tokens.extend(data_ids[position] for position in data_fit_positions)
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
    dimension_indices: list[np.ndarray] = []
    for token in desired_tokens:
        if token == row_token:
            dimension_indices.append(row_order)
        else:
            values = _resolved_axis_values(schema.axis(token))
            dimension_indices.append(
                np.argsort(values, kind="stable").astype(np.int64, copy=False)
            )
    compressed_fit_coordinates = (
        tuple(np.asarray(key[row_order], dtype=np.float64) for key in row_keys)
        if len(cell_fit_ids) == 2
        else None
    )
    return dimension_order, tuple(dimension_indices), compressed_fit_coordinates


def _canonical_ranks_to_physical(
    canonical_ranks: np.ndarray,
    physical_shape: tuple[int, ...],
    dimension_order: tuple[int, ...],
    dimension_indices: tuple[np.ndarray, ...],
) -> np.ndarray:
    if canonical_ranks.size == 0:
        return np.empty(0, dtype=np.int64)
    canonical_shape = tuple(physical_shape[axis] for axis in dimension_order)
    canonical_multi = np.unravel_index(canonical_ranks, canonical_shape, order="C")
    physical_multi: list[np.ndarray | None] = [None] * len(physical_shape)
    for canonical_axis, physical_axis in enumerate(dimension_order):
        physical_multi[physical_axis] = dimension_indices[canonical_axis][
            canonical_multi[canonical_axis]
        ]
    assert all(value is not None for value in physical_multi)
    return np.asarray(
        np.ravel_multi_index(tuple(physical_multi), physical_shape, order="C"),
        dtype=np.int64,
    )


def _resolved_axis_values(axis: AxisSpec) -> np.ndarray:
    if axis.coordinates is None:
        return np.arange(axis.size, dtype=np.float64)
    return np.asarray(axis.coordinates, dtype=np.float64)


def _evenly_spaced_ranks(size: int, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count >= size:
        return np.arange(size, dtype=np.int64)
    if count == 1:
        return np.asarray([size // 2], dtype=np.int64)
    # Include both authoritative endpoints; integer arithmetic is deterministic
    # and cannot duplicate ranks when count <= size.
    return (
        np.arange(count, dtype=np.int64) * (size - 1) // (count - 1)
    ).astype(np.int64, copy=False)


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
