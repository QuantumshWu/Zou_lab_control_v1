"""Fit binding and the sole authoritative dataset-to-solver packing path."""

from __future__ import annotations

from collections import OrderedDict
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
from .value import OwnedSnapshot, expand_dataset_validity


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

    fit_axes = tuple(schema.axis(axis_id) for axis_id in bound.spec.fit_axis_ids)
    batch_axes = tuple(schema.axis(axis_id) for axis_id in bound.spec.batch_axis_ids)
    coordinate_sources = tuple(_coordinate_source_for_axis(axis) for axis in fit_axes)
    source_sampling_quanta = tuple(
        _source_sampling_quantum(axis, source)
        for axis, source in zip(fit_axes, coordinate_sources)
    )

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
        key = tuple(int(axis_indices[position][row]) for position in cell_batch_positions)
        if key not in row_groups and len(row_groups) >= max_cell_groups:
            raise ValueError(
                "cell batch axes exceed max_batch_cells="
                f"{bound.spec.numeric_policy.max_batch_cells}"
            )
        row_groups.setdefault(key, []).append(row)

    batch_count = len(row_groups) * data_batch_count
    if batch_count > bound.spec.numeric_policy.max_batch_cells:
        raise ValueError(
            f"fit request creates {batch_count} batch cells, exceeding max_batch_cells="
            f"{bound.spec.numeric_policy.max_batch_cells}"
        )

    mapping: list[tuple[int, ...]] = []
    observations_parts: list[np.ndarray] = []
    coordinate_parts: list[list[np.ndarray]] = [list() for _ in fit_axes]
    sampling_quanta: list[tuple[float, ...]] = []
    present_counts: list[int] = []
    valid_counts: list[int] = []
    used_counts: list[int] = []
    packed_observation_count = 0

    data_combinations = tuple(np.ndindex(data_batch_shape))
    for cell_batch_key, row_list in row_groups.items():
        _check_abort(abort_check)
        row_ids = np.asarray(row_list, dtype=np.int64)
        row_selector = _compact_row_selector(row_ids)
        cell_batch_values = {
            cell_ids[position]: cell_batch_key[index]
            for index, position in enumerate(cell_batch_positions)
        }
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
            batch_multi = tuple(
                cell_batch_values.get(axis_id, data_batch_values.get(axis_id))
                for axis_id in bound.spec.batch_axis_ids
            )
            if any(value is None for value in batch_multi):  # pragma: no cover - partition invariant
                raise RuntimeError("batch axis partition failed")
            mapping.append(tuple(int(value) for value in batch_multi))

            data_selectors = tuple(
                data_batch_values.get(axis_id, slice(None)) for axis_id in data_ids
            )
            selection = (row_selector, *data_selectors)
            observation_view = values[selection]
            validity_view = validity[selection]
            selected, present_count, valid_count = _sample_valid_positions(
                observation_view,
                validity_view,
                bound.spec.numeric_policy.sample_budget_per_batch,
                dimension_order,
                dimension_indices,
                compressed_fit_coordinates,
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
            present_counts.append(present_count)
            valid_counts.append(valid_count)
            used_counts.append(used_count)

            packed_indices = np.unravel_index(
                selected,
                observation_view.shape,
                order="C",
            )
            selected_row_offsets = packed_indices[0]
            selected_rows = row_ids[selected_row_offsets]
            cell_quanta: list[float] = []
            for fit_position, (axis, source, source_quantum) in enumerate(
                zip(fit_axes, coordinate_sources, source_sampling_quanta)
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
                cell_quanta.append(_sampling_quantum(source_quantum, logical))
            sampling_quanta.append(tuple(cell_quanta))

    batch_shape = tuple(axis.size for axis in batch_axes)
    batch_layout = _batch_layout_from_mapping(batch_shape, tuple(mapping))
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
        sampling_quanta=np.asarray(sampling_quanta, dtype=np.dtype("<f8")).reshape(
            len(mapping),
            len(fit_axes),
        ),
        observations=packed_observations,
        present_observation_counts=np.asarray(present_counts, dtype=np.dtype("<i8")),
        valid_observation_counts=np.asarray(valid_counts, dtype=np.dtype("<i8")),
        used_observation_counts=np.asarray(used_counts, dtype=np.dtype("<i8")),
    )


def validate_fit_result_binding(
    result: FitResultBatch,
    snapshot: OwnedSnapshot,
) -> None:
    """Replay the packing contract that binds a persisted result to its source.

    This does not rerun the solver or judge its numeric output.  It delegates
    dataset packing to :func:`build_fit_problem`, then verifies only the source,
    axis, layout, unit, and observation-count facts derived by that owner.
    """

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    problem = build_fit_problem(
        bind_fit(result.spec, snapshot.block.schema),
        snapshot,
    )
    if result.source_ref != problem.source_ref:
        raise ValueError("fit result source reference differs from source snapshot")
    if result.fit_axis_specs != problem.fit_axis_specs:
        raise ValueError("fit result axis specifications differ from source packing")
    if result.batch_axis_specs != problem.batch_axis_specs:
        raise ValueError("fit result batch axes differ from source packing")
    if result.batch_layout != problem.batch_layout:
        raise ValueError("fit result batch layout differs from source packing")
    if result.value_unit != problem.value_unit:
        raise ValueError("fit result value unit differs from source packing")
    for field in (
        "present_observation_counts",
        "valid_observation_counts",
        "used_observation_counts",
    ):
        if not np.array_equal(getattr(result, field), getattr(problem, field)):
            raise ValueError(
                f"fit result {field} differs from source packing"
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


def _source_sampling_quantum(
    axis: AxisSpec,
    source: FitCoordinateSource,
) -> float:
    if source is FitCoordinateSource.LOGICAL_INDEX:
        return 1.0
    if axis.coordinates is None:  # pragma: no cover - coordinate source invariant
        return 0.0
    return _uniform_source_quantum(np.asarray(axis.coordinates, dtype=np.float64))


def _sampling_quantum(source_quantum: float, logical_indices: np.ndarray) -> float:
    """Compress exact source-lattice provenance to the used stride quantum."""

    if source_quantum == 0.0:
        return 0.0
    unique = np.unique(np.asarray(logical_indices, dtype=np.int64))
    differences = np.diff(unique)
    if differences.size == 0:
        return 0.0
    stride_gcd = int(np.gcd.reduce(differences))
    if stride_gcd <= 0:  # pragma: no cover - unique sorted indices make this impossible
        return 0.0
    return float(source_quantum * stride_gcd)


def _uniform_source_quantum(values: np.ndarray) -> float:
    """Prove an affine float lattice to representation-scale precision.

    A relative tolerance would become physically huge for axes with a large
    coordinate offset.  Here every encoded coordinate must agree with one affine
    reconstruction within four local float64 ULPs; anything looser is unproven.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        return 0.0
    differences = np.diff(values)
    if np.any(differences == 0.0) or not (
        np.all(differences > 0.0) or np.all(differences < 0.0)
    ):
        return 0.0
    step = float((values[-1] - values[0]) / (values.size - 1))
    if not np.isfinite(step) or step == 0.0:
        return 0.0
    expected = values[0] + np.arange(values.size, dtype=np.float64) * step
    ulp = np.spacing(np.maximum(np.abs(values), np.abs(expected)))
    tolerance = np.maximum(4.0 * ulp, np.finfo(np.float64).tiny)
    if np.any(np.abs(values - expected) > tolerance):
        return 0.0
    return abs(step)


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
) -> tuple[np.ndarray, int, int]:
    present_count = int(observations.size)
    valid_count = int(np.count_nonzero(validity))
    if valid_count == 0:
        return np.empty(0, dtype=np.int64), present_count, 0

    nonfinite = np.logical_and(validity, np.logical_not(np.isfinite(observations)))
    has_nonfinite = bool(np.any(nonfinite))
    used_count = min(valid_count, budget)
    canonical_shape = tuple(observations.shape[axis] for axis in dimension_order)
    preferred = _preferred_canonical_ranks(
        canonical_shape,
        used_count,
        compressed_fit_coordinates,
    )
    if valid_count == present_count and not has_nonfinite:
        canonical_ranks = _fill_full_valid_ranks(preferred, present_count, used_count)
    else:
        canonical_validity = _canonical_flat_mask(
            validity,
            dimension_order,
            dimension_indices,
        )
        if has_nonfinite:
            canonical_nonfinite = _canonical_flat_mask(
                nonfinite,
                dimension_order,
                dimension_indices,
            )
            bad_ranks = np.flatnonzero(canonical_nonfinite)
        else:
            canonical_nonfinite = None
            bad_ranks = np.empty(0, dtype=np.int64)
        if bad_ranks.size >= used_count:
            canonical_ranks = bad_ranks[:used_count]
        else:
            finite_validity = canonical_validity.copy()
            if canonical_nonfinite is not None:
                finite_validity[canonical_nonfinite] = False
            finite_ranks = np.flatnonzero(finite_validity)
            remaining = used_count - bad_ranks.size
            preferred_finite = preferred[finite_validity[preferred]]
            if preferred_finite.size > remaining:
                preferred_finite = preferred_finite[
                    _evenly_spaced_ranks(preferred_finite.size, remaining)
                ]
            still_needed = remaining - preferred_finite.size
            if still_needed:
                available = np.setdiff1d(
                    finite_ranks,
                    preferred_finite,
                    assume_unique=True,
                )
                extra = available[
                    _evenly_spaced_ranks(available.size, still_needed)
                ]
                chosen_finite = np.concatenate((preferred_finite, extra))
            else:
                chosen_finite = preferred_finite
            canonical_ranks = np.concatenate((bad_ranks, chosen_finite))
            canonical_ranks.sort()
    selected = _canonical_ranks_to_physical(
        canonical_ranks,
        observations.shape,
        dimension_order,
        dimension_indices,
    )
    return selected, present_count, valid_count


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
    row_by_coordinate: dict[tuple[float, float], int] = {}
    for row, pair in enumerate(zip(first_values, second_values)):
        row_by_coordinate.setdefault((float(pair[0]), float(pair[1])), row)
    selected = tuple(
        row_by_coordinate[pair]
        for first in selected_first
        for second in selected_second
        if (pair := (float(first), float(second))) in row_by_coordinate
    )
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


def _fill_full_valid_ranks(
    preferred: np.ndarray,
    present_count: int,
    used_count: int,
) -> np.ndarray:
    if preferred.size >= used_count:
        return np.sort(preferred[:used_count])
    needed = used_count - preferred.size
    candidate_count = min(
        present_count,
        max(used_count * 2, preferred.size + needed * 4),
    )
    while True:
        candidates = _evenly_spaced_ranks(present_count, candidate_count)
        available = candidates[
            np.logical_not(np.isin(candidates, preferred, assume_unique=True))
        ]
        if available.size >= needed or candidate_count == present_count:
            break
        candidate_count = min(present_count, candidate_count * 2)
    result = np.concatenate((preferred, available[:needed]))
    result.sort()
    return result


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
    if not row_keys:
        row_order = np.arange(row_ids.size, dtype=np.int64)
    elif len(row_keys) == 1:
        row_order = np.argsort(row_keys[0], kind="stable").astype(np.int64, copy=False)
    else:
        row_order = np.lexsort(tuple(reversed(row_keys))).astype(np.int64, copy=False)

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


def _canonical_flat_mask(
    mask: np.ndarray,
    dimension_order: tuple[int, ...],
    dimension_indices: tuple[np.ndarray, ...],
) -> np.ndarray:
    ordered = mask
    if dimension_order != tuple(range(mask.ndim)):
        ordered = np.transpose(ordered, axes=dimension_order)
    for axis, indices in enumerate(dimension_indices):
        if not np.array_equal(indices, np.arange(indices.size, dtype=np.int64)):
            ordered = np.take(ordered, indices, axis=axis)
    return np.asarray(ordered, dtype=bool).reshape(-1)


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


__all__ = ["bind_fit", "build_fit_problem", "validate_fit_result_binding"]
