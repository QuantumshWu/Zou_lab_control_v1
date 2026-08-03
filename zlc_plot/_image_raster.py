"""One policy-owned display front for regular two-dimensional images.

The immutable source remains authoritative for selection and analysis.  This
module prepares the scalar front handed to Matplotlib.  Cropping, optional
area reduction, and Matplotlib's interpolation stage are decided together so
the renderer cannot silently apply a second independent raster policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ImageFrontPolicy:
    """Bounded-work policy for one scalar image front.

    Area reduction is worthwhile only when a source dimension materially
    oversamples its physical output dimension.  Smaller differences stay in
    the source dtype and are resolved once by Matplotlib's scalar-data stage.
    """

    minimum_reduction_ratio: float = 1.25
    interpolation_stage: str = "data"

    def __post_init__(self) -> None:
        ratio = float(self.minimum_reduction_ratio)
        if not math.isfinite(ratio) or ratio <= 1.0:
            raise ValueError("minimum_reduction_ratio must be finite and greater than one")
        if self.interpolation_stage != "data":
            raise ValueError("a scalar image front requires interpolation_stage='data'")
        object.__setattr__(self, "minimum_reduction_ratio", ratio)


@dataclass(frozen=True, slots=True)
class PreparedImageFront:
    """One scalar display front and the exact data extent it represents."""

    values: np.ndarray | np.ma.MaskedArray
    extent: tuple[float, float, float, float]
    interpolation_stage: str


def _all_true(values: np.ndarray) -> bool:
    array = np.asarray(values, dtype=np.bool_)
    if not array.size:
        return True
    if all(stride == 0 for stride in array.strides):
        return bool(array.flat[0])
    return bool(np.all(array))


def _index_window(
    low: float,
    high: float,
    edge_start: float,
    edge_stop: float,
    count: int,
) -> tuple[int, int]:
    step = (edge_stop - edge_start) / count
    first = (low - edge_start) / step
    second = (high - edge_start) / step
    if first > second:
        first, second = second, first
    return (
        max(0, min(int(math.floor(first)), count - 1)),
        max(1, min(int(math.ceil(second)), count)),
    )


def _reduction_starts(
    size: int,
    target: int,
    minimum_ratio: float,
) -> np.ndarray:
    count = (
        target
        if size > target and size / target >= minimum_ratio
        else size
    )
    return np.arange(count, dtype=np.intp) * size // count


def _reduce_blocks(
    values: np.ndarray,
    row_starts: np.ndarray,
    column_starts: np.ndarray,
    dtype: Any,
) -> np.ndarray:
    rows, columns = values.shape
    if row_starts.size == rows:
        return np.add.reduceat(values, column_starts, axis=1, dtype=dtype)
    if column_starts.size == columns:
        return np.add.reduceat(values, row_starts, axis=0, dtype=dtype)
    if row_starts.size * columns <= rows * column_starts.size:
        reduced = np.add.reduceat(values, row_starts, axis=0, dtype=dtype)
        return np.add.reduceat(reduced, column_starts, axis=1, dtype=dtype)
    reduced = np.add.reduceat(values, column_starts, axis=1, dtype=dtype)
    return np.add.reduceat(reduced, row_starts, axis=0, dtype=dtype)


def _area_mean(
    values: np.ndarray,
    valid: np.ndarray,
    row_starts: np.ndarray,
    column_starts: np.ndarray,
) -> np.ndarray | np.ma.MaskedArray:
    all_valid = _all_true(valid)
    source = values if all_valid else np.where(valid, values, 0)
    mean_dtype = np.result_type(values.dtype, np.float32)
    summed = _reduce_blocks(source, row_starts, column_starts, mean_dtype)
    if all_valid:
        row_counts = np.diff(np.r_[row_starts, values.shape[0]])
        column_counts = np.diff(np.r_[column_starts, values.shape[1]])
        np.divide(summed, row_counts[:, np.newaxis], out=summed)
        np.divide(summed, column_counts[np.newaxis, :], out=summed)
        return summed
    counts = _reduce_blocks(valid, row_starts, column_starts, np.int64)
    means = np.zeros(summed.shape, dtype=mean_dtype)
    np.divide(summed, counts, out=means, where=counts != 0)
    return means if bool(np.all(counts)) else np.ma.array(
        means, mask=counts == 0, copy=False
    )


def prepare_image_front(
    values: Any,
    validity: Any | None,
    extent: tuple[float, float, float, float],
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    display_pixel_shape: tuple[int, int],
    policy: ImageFrontPolicy,
) -> PreparedImageFront:
    """Crop and, when policy requires it, reduce one regular scalar image.

    ``extent`` follows Matplotlib's ``(left, right, bottom, top)`` order.  The
    first source row is associated with ``top`` because image plots use an
    upper origin.  A source that does not cross the authored reduction ratio
    keeps its dtype and shares source memory.  Only a materially oversampled
    front is promoted to a floating area mean.
    """

    if not isinstance(policy, ImageFrontPolicy):
        raise TypeError("policy must be ImageFrontPolicy")
    source = np.asarray(values)
    if source.ndim != 2 or min(source.shape) < 1:
        raise ValueError("image values must be a non-empty two-dimensional array")
    if source.dtype.kind not in "biuf":
        raise TypeError("image values must be real numeric")
    if validity is None:
        valid = np.broadcast_to(np.asarray(True, dtype=np.bool_), source.shape)
    else:
        valid = np.asarray(validity)
        if valid.dtype != np.dtype(np.bool_):
            raise TypeError("image validity must have boolean dtype")
        try:
            valid = np.broadcast_to(valid, source.shape)
        except ValueError as error:
            raise ValueError(
                "image validity cannot broadcast to the image shape"
            ) from error

    prepared_extent = tuple(float(value) for value in extent)
    if len(prepared_extent) != 4 or not all(
        math.isfinite(value) for value in prepared_extent
    ):
        raise ValueError("image extent must contain four finite values")
    left, right, bottom, top = prepared_extent
    if left == right or bottom == top:
        raise ValueError("image extent spans must be non-degenerate")
    x_view = tuple(float(value) for value in x_limits)
    y_view = tuple(float(value) for value in y_limits)
    if len(x_view) != 2 or len(y_view) != 2 or not all(
        math.isfinite(value) for value in (*x_view, *y_view)
    ):
        raise ValueError("image view limits must contain finite pairs")
    width, height = display_pixel_shape
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or int(width) <= 0
        or int(height) <= 0
    ):
        raise ValueError("display pixel dimensions must be positive integers")
    display_width, display_height = int(width), int(height)

    rows, columns = source.shape
    column_start, column_stop = _index_window(
        *x_view,
        left,
        right,
        columns,
    )
    # Row zero is the image's top edge, even when the displayed Y direction is
    # reversed.  Walking from top to bottom therefore follows source order.
    row_start, row_stop = _index_window(
        *y_view,
        top,
        bottom,
        rows,
    )
    subset = source[row_start:row_stop, column_start:column_stop]
    subset_valid = valid[row_start:row_stop, column_start:column_stop]
    if source.dtype.kind == "f":
        finite = np.isfinite(subset)
        subset_valid = (
            finite
            if _all_true(subset_valid)
            else np.logical_and(subset_valid, finite)
        )

    subset_rows, subset_columns = subset.shape
    row_starts = _reduction_starts(
        subset_rows,
        display_height,
        policy.minimum_reduction_ratio,
    )
    column_starts = _reduction_starts(
        subset_columns,
        display_width,
        policy.minimum_reduction_ratio,
    )
    reduced = row_starts.size < subset_rows or column_starts.size < subset_columns
    if not reduced:
        shown: np.ndarray | np.ma.MaskedArray = (
            subset
            if _all_true(subset_valid)
            else np.ma.array(
                subset,
                mask=np.logical_not(subset_valid),
                copy=False,
            )
        )
    else:
        shown = _area_mean(subset, subset_valid, row_starts, column_starts)
    shown.setflags(write=False)

    x_step = (right - left) / columns
    y_step = (bottom - top) / rows
    shown_extent = (
        left + column_start * x_step,
        left + column_stop * x_step,
        top + row_stop * y_step,
        top + row_start * y_step,
    )
    return PreparedImageFront(
        shown,
        tuple(float(value) for value in shown_extent),
        policy.interpolation_stage,
    )


__all__ = ["ImageFrontPolicy", "PreparedImageFront", "prepare_image_front"]
