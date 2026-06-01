"""Small image-analysis helpers for the first neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Sequence

import numpy as np
from scipy import ndimage


SUPPORTED_REDUCERS = ("mean", "sum", "median", "max")
SUPPORTED_ORDERINGS = ("row-major", "serpentine", "column-major", "column-serpentine")


@dataclass(frozen=True)
class AtomDetection:
    """Counts and binary occupancy for one image."""

    counts: np.ndarray
    occupied: np.ndarray
    occupied_indices: list[int]
    thresholds: np.ndarray


@dataclass(frozen=True)
class FidelityEstimate:
    """Gaussian-split estimate for one thresholded distribution."""

    threshold: float
    fidelity: float
    left_fraction: float
    right_fraction: float
    separation: float
    dark_mean: float
    bright_mean: float
    dark_sigma: float
    bright_sigma: float


def roi_counts(image, centers, *, radius: int = 1, reducer: str = "mean") -> np.ndarray:
    """Return one scalar ROI count for every ``(x, y)`` site center."""

    img = image_array(image)
    centers = centers_array(centers)
    radius = nonnegative_int(radius, "radius")
    reducer = normalize_reducer(reducer)
    out: list[float] = []
    for index, (x, y) in enumerate(centers):
        cx, cy = int(round(float(x))), int(round(float(y)))
        if cx < 0 or cx >= img.shape[1] or cy < 0 or cy >= img.shape[0]:
            raise ValueError(f"center {index} is outside image shape {img.shape}: ({x:g}, {y:g})")
        x0, x1 = max(0, cx - radius), min(img.shape[1], cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(img.shape[0], cy + radius + 1)
        roi = img[y0:y1, x0:x1]
        finite = roi[np.isfinite(roi)]
        if finite.size == 0:
            raise ValueError(f"ROI {index} contains no finite pixels.")
        if reducer == "sum":
            out.append(float(np.sum(finite)))
        elif reducer == "median":
            out.append(float(np.median(finite)))
        elif reducer == "max":
            out.append(float(np.max(finite)))
        else:
            out.append(float(np.mean(finite)))
    return np.asarray(out, dtype=float)


def detect_atoms(image, centers, thresholds, *, radius: int = 1, reducer: str = "mean") -> AtomDetection:
    """Classify atoms by comparing ROI counts to scalar or per-site thresholds."""

    counts = roi_counts(image, centers, radius=radius, reducer=reducer)
    thresholds = threshold_array(thresholds, len(counts))
    occupied = counts > thresholds
    return AtomDetection(
        counts=counts,
        occupied=occupied,
        occupied_indices=np.flatnonzero(occupied).astype(int).tolist(),
        thresholds=thresholds,
    )


def find_site_centers(
    image,
    grid_shape: Sequence[int],
    *,
    min_distance: int | None = None,
    threshold_rel: float = 0.35,
    ordering: str = "row-major",
) -> np.ndarray:
    """Find bright trap centers and sort them into a stable site order."""

    img = image_array(image)
    ny, nx = grid_shape_tuple(grid_shape)
    need = ny * nx
    threshold_rel = finite_float(threshold_rel, "threshold_rel")
    if threshold_rel < 0 or threshold_rel > 1:
        raise ValueError("threshold_rel must be in [0, 1].")
    if min_distance is None:
        min_distance = max(3, int(min(img.shape) / max(ny, nx, 1) / 2))
    min_distance = positive_int(min_distance, "min_distance")

    smooth = ndimage.gaussian_filter(img, sigma=1.0)
    cutoff = float(np.nanmin(smooth) + threshold_rel * (np.nanmax(smooth) - np.nanmin(smooth)))
    local_max = ndimage.maximum_filter(smooth, size=min_distance)
    candidates_yx = np.argwhere((smooth == local_max) & (smooth >= cutoff))
    if len(candidates_yx) < need:
        flat = np.argsort(smooth.ravel())[::-1][:need]
        candidates_yx = np.column_stack(np.unravel_index(flat, smooth.shape))
    weights = smooth[candidates_yx[:, 0], candidates_yx[:, 1]]
    selected = candidates_yx[np.argsort(weights)[::-1]][:need]
    centers_xy = np.column_stack([selected[:, 1], selected[:, 0]]).astype(float)
    return sort_centers_grid(centers_xy, (ny, nx), ordering=ordering)


def sort_centers_grid(centers, grid_shape: Sequence[int], *, ordering: str = "row-major") -> np.ndarray:
    """Sort center coordinates into row/column-major site order."""

    centers = centers_array(centers)
    ny, nx = grid_shape_tuple(grid_shape)
    if len(centers) != ny * nx:
        raise ValueError(f"expected {ny * nx} centers for grid_shape={grid_shape}, got {len(centers)}")
    ordering = normalize_ordering(ordering)

    if ordering in {"row-major", "serpentine"}:
        rows = []
        for row_index, chunk in enumerate(np.array_split(centers[np.argsort(centers[:, 1])], ny)):
            row = chunk[np.argsort(chunk[:, 0])]
            if ordering == "serpentine" and row_index % 2:
                row = row[::-1]
            rows.append(row)
        return np.vstack(rows)

    cols = []
    for col_index, chunk in enumerate(np.array_split(centers[np.argsort(centers[:, 0])], nx)):
        col = chunk[np.argsort(chunk[:, 1])]
        if ordering == "column-serpentine" and col_index % 2:
            col = col[::-1]
        cols.append(col)
    return np.vstack(cols)


def estimate_thresholds(images, centers, *, radius: int = 1, reducer: str = "mean", bins: int = 96) -> np.ndarray:
    """Estimate one threshold per site from a stack of calibration images."""

    centers = centers_array(centers)
    bins = positive_int(bins, "bins")
    stack = [image_array(image) for image in images]
    if not stack:
        raise ValueError("images must contain at least one frame.")
    if any(frame.shape != stack[0].shape for frame in stack):
        raise ValueError("all calibration images must have the same shape.")
    counts = np.vstack([roi_counts(frame, centers, radius=radius, reducer=reducer) for frame in stack])
    return np.asarray([otsu_threshold(counts[:, i], bins=bins) for i in range(counts.shape[1])], dtype=float)


def otsu_threshold(values, *, bins: int = 96) -> float:
    """One-dimensional Otsu threshold with finite-value filtering."""

    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("threshold values contain no finite entries.")
    if float(np.min(vals)) == float(np.max(vals)):
        return float(vals[0])
    hist, edges = np.histogram(vals, bins=positive_int(bins, "bins"))
    centers = (edges[:-1] + edges[1:]) / 2
    weights = hist.astype(float)
    total = weights.sum()
    if total <= 0:
        return float(np.median(vals))
    prob = weights / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    denom = omega * (1.0 - omega)
    score = np.full_like(centers, -np.inf, dtype=float)
    valid = denom > 0
    score[valid] = (mu[-1] * omega[valid] - mu[valid]) ** 2 / denom[valid]
    return float(centers[int(np.argmax(score))])


def estimate_threshold_fidelity(values, threshold: float) -> FidelityEstimate:
    """Estimate fidelity from data split by an explicit threshold."""

    threshold = finite_float(threshold, "threshold")
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return FidelityEstimate(threshold, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    left = vals[vals <= threshold]
    right = vals[vals > threshold]
    left_fraction = float(len(left) / len(vals))
    right_fraction = float(len(right) / len(vals))
    if len(left) < 2 or len(right) < 2:
        return FidelityEstimate(threshold, np.nan, left_fraction, right_fraction, np.nan, np.nan, np.nan, np.nan, np.nan)

    mu0, mu1 = float(np.mean(left)), float(np.mean(right))
    s0 = max(float(np.std(left, ddof=1)), 1e-12)
    s1 = max(float(np.std(right, ddof=1)), 1e-12)
    w0, w1 = float(len(left)), float(len(right))
    p_dark_correct = _normal_cdf(threshold, mu0, s0)
    p_bright_correct = 1.0 - _normal_cdf(threshold, mu1, s1)
    separation = abs(mu1 - mu0) / sqrt(s0 * s0 + s1 * s1)
    raw_fidelity = (w0 * p_dark_correct + w1 * p_bright_correct) / (w0 + w1)
    balance = 2.0 * min(left_fraction, right_fraction)
    effective_separation = max(0.0, separation - 2.0)
    separation_confidence = 1.0 - np.exp(-0.5 * effective_separation * effective_separation)
    confidence = float(np.clip(balance * separation_confidence, 0.0, 1.0))
    fidelity = 0.5 + (raw_fidelity - 0.5) * confidence
    return FidelityEstimate(threshold, float(fidelity), left_fraction, right_fraction, float(separation), mu0, mu1, s0, s1)


def image_array(image) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("image must be a non-empty 2D array.")
    return arr


def centers_array(centers) -> np.ndarray:
    reject_bool_values(centers, "centers")
    arr = np.asarray(centers, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("centers must have shape (N, 2) or wider.")
    out = arr[:, :2]
    if not np.all(np.isfinite(out)):
        raise ValueError("centers must contain finite x/y coordinates.")
    return out


def threshold_array(thresholds, n: int) -> np.ndarray:
    reject_bool_values(thresholds, "thresholds")
    arr = np.asarray(thresholds, dtype=float)
    if arr.ndim == 0:
        out = np.full(int(n), float(arr), dtype=float)
    else:
        if arr.size != int(n):
            raise ValueError(f"thresholds must be scalar or length {int(n)}.")
        out = arr.reshape(-1).astype(float)
    if not np.all(np.isfinite(out)):
        raise ValueError("thresholds must contain finite values.")
    return out


def grid_shape_tuple(value, name: str = "grid_shape") -> tuple[int, int]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two positive integers.") from exc
    if len(raw) != 2:
        raise ValueError(f"{name} must contain two positive integers.")
    return positive_int(raw[0], f"{name}[0]"), positive_int(raw[1], f"{name}[1]")


def normalize_reducer(reducer: str) -> str:
    name = str(reducer).lower().replace("_", "-")
    if name not in SUPPORTED_REDUCERS:
        raise ValueError(f"reducer must be one of {', '.join(SUPPORTED_REDUCERS)}.")
    return name


def normalize_ordering(ordering: str) -> str:
    name = str(ordering).lower().replace("_", "-")
    aliases = {
        "row": "row-major",
        "row-major": "row-major",
        "snake": "serpentine",
        "serpentine": "serpentine",
        "column": "column-major",
        "col": "column-major",
        "column-major": "column-major",
        "snake-column": "column-serpentine",
        "column-serpentine": "column-serpentine",
    }
    if name not in aliases:
        raise ValueError(f"ordering must be one of {', '.join(SUPPORTED_ORDERINGS)}.")
    return aliases[name]


def finite_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite, not a boolean.")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def nonnegative_int(value, name: str) -> int:
    out = finite_float(value, name)
    if int(out) != out or out < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(out)


def positive_int(value, name: str) -> int:
    out = nonnegative_int(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return out


def reject_bool_values(values, name: str) -> None:
    arr = np.asarray(values, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in arr.reshape(-1)):
        raise ValueError(f"{name} must contain numeric values, not booleans.")


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


__all__ = [
    "AtomDetection",
    "FidelityEstimate",
    "SUPPORTED_ORDERINGS",
    "SUPPORTED_REDUCERS",
    "centers_array",
    "detect_atoms",
    "estimate_threshold_fidelity",
    "estimate_thresholds",
    "find_site_centers",
    "grid_shape_tuple",
    "image_array",
    "otsu_threshold",
    "roi_counts",
    "sort_centers_grid",
    "threshold_array",
]
