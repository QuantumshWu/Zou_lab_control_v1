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


def _integer_sum_requires_object(dtype: np.dtype, contributors: int) -> bool:
    normalized = canonical_dtype(dtype)
    if normalized.kind == "b":
        return False
    limits = np.iinfo(normalized)
    output = np.iinfo(canonical_sum_dtype(normalized))
    return (
        limits.max * contributors > output.max
        or (normalized.kind == "i" and limits.min * contributors < output.min)
    )


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
    normalized_axes = tuple(int(axis) for axis in axes)
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
        contributor_bound = math.prod(array.shape[axis] for axis in normalized_axes)
        if not _integer_sum_requires_object(input_dtype, contributor_bound):
            return np.asarray(
                np.sum(array, axis=normalized_axes, dtype=target),
                dtype=target,
            )
        exact = array.astype(object)
        for axis in sorted(normalized_axes, reverse=True):
            exact = np.sum(exact, axis=axis, dtype=object)
        flat = np.asarray(exact, dtype=object).reshape(-1)
        output_info = np.iinfo(target)
        if any(value < output_info.min or value > output_info.max for value in flat):
            raise OverflowError("integer SUM exceeds its canonical output dtype")
        return np.asarray(exact, dtype=target)

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
