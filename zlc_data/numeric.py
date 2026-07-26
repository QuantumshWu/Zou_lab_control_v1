"""Shared numeric safety rules for authoritative and display reductions."""

from __future__ import annotations

import math

import numpy as np

from ._arrays import canonical_dtype


def canonical_mean_dtype(dtype: np.dtype) -> np.dtype:
    """Return the loss-aware canonical accumulator dtype for a numeric mean."""

    normalized = canonical_dtype(dtype)
    return np.dtype("<c16") if normalized.kind == "c" else np.dtype("<f8")


def canonical_sum_dtype(dtype: np.dtype) -> np.dtype:
    """Return the canonical accumulator dtype for a numeric sum."""

    normalized = canonical_dtype(dtype)
    if normalized.kind in "bi":
        return np.dtype("<i8")
    if normalized.kind == "u":
        return np.dtype("<u8")
    if normalized.kind == "f":
        return np.dtype("<f8")
    if normalized.kind == "c":
        return np.dtype("<c16")
    raise TypeError(f"unsupported reduction dtype {normalized}")  # pragma: no cover


def _integer_sum_bounds_fit(
    array: np.ndarray,
    contributors: int,
    target: np.dtype,
) -> bool:
    """Return whether ordinary fixed-width accumulation is provably exact."""

    if contributors == 0 or array.size == 0:
        return True
    minimum = int(np.min(array))
    maximum = int(np.max(array))
    limits = np.iinfo(target)
    lower_bound = min(0, minimum) * contributors
    upper_bound = max(0, maximum) * contributors
    return lower_bound >= limits.min and upper_bound <= limits.max


def _checked_integer_sum_exact(
    array: np.ndarray,
    axes: tuple[int, ...],
    target: np.dtype,
) -> np.ndarray:
    """Accumulate only ambiguous integer groups with Python's exact integers.

    This path is reached only when the actual-value bounds cannot prove that a
    fixed-width NumPy sum is safe.  It avoids both silent wraparound and the
    former full object-dtype array conversion.
    """

    reduced = set(axes)
    remaining = tuple(axis for axis in range(array.ndim) if axis not in reduced)
    ordered = np.transpose(array, remaining + axes)
    output_shape = tuple(array.shape[axis] for axis in remaining)
    result = np.empty(output_shape, dtype=target)
    limits = np.iinfo(target)
    for output_index in np.ndindex(output_shape):
        group = ordered[output_index] if output_shape else ordered
        total = sum(int(value) for value in np.asarray(group).flat)
        if total < limits.min or total > limits.max:
            raise OverflowError("integer SUM exceeds its canonical output dtype")
        result[output_index] = total
    return result


def checked_numeric_sum(
    values: np.ndarray,
    axes: tuple[int, ...],
    *,
    output_dtype: np.dtype | None = None,
) -> np.ndarray:
    """Sum without integer wraparound or finite-input floating overflow.

    The caller chooses which values contribute (normally by replacing invalid
    values with zero).  This helper owns only accumulator width and overflow
    detection, so presentation and authoritative reducers cannot drift.
    """

    array = np.asarray(values)
    input_dtype = canonical_dtype(array.dtype)
    if input_dtype != array.dtype:
        array = array.astype(input_dtype, copy=False)
    if not isinstance(axes, tuple):
        raise TypeError("sum axes must be a tuple of integer axis positions")
    if any(
        isinstance(axis, (bool, np.bool_))
        or not isinstance(axis, (int, np.integer))
        for axis in axes
    ):
        raise TypeError("sum axes must contain only integer axis positions")
    normalized_axes = tuple(sorted(int(axis) for axis in axes))
    if not normalized_axes or len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("sum axes must be a non-empty unique tuple")
    if any(axis < 0 or axis >= array.ndim for axis in normalized_axes):
        raise ValueError("sum axis is outside the input rank")
    canonical_target = canonical_sum_dtype(input_dtype)
    target = (
        canonical_target
        if output_dtype is None
        else canonical_dtype(output_dtype)
    )
    if target != canonical_target:
        raise TypeError(
            f"SUM output dtype {target} disagrees with canonical {canonical_target}"
        )

    if input_dtype.kind in "biu":
        contributor_bound = math.prod(
            array.shape[axis] for axis in normalized_axes
        )
        if _integer_sum_bounds_fit(array, contributor_bound, target):
            return np.asarray(
                np.sum(array, axis=normalized_axes, dtype=target),
                dtype=target,
            )
        return _checked_integer_sum_exact(array, normalized_axes, target)

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.sum(array, axis=normalized_axes, dtype=target)
    if np.all(np.isfinite(array)) and np.any(~np.isfinite(result)):
        raise OverflowError("floating SUM overflowed its canonical output dtype")
    return np.asarray(result, dtype=target)


__all__ = [
    "canonical_mean_dtype",
    "canonical_sum_dtype",
    "checked_numeric_sum",
]
