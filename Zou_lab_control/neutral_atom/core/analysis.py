"""Small image-analysis helpers for the first neutral-atom session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from Zou_lab_control._readout_math import confidence_weighted_fidelity


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


def _gauss2d_axis(coords, offset, amp, x0, y0, sx, sy):
    x, y = coords
    return (offset + amp * np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))).ravel()


def _refine_center_subpixel(image: np.ndarray, x: float, y: float, half: int = 2) -> tuple[float, float]:
    """Refine an integer peak ``(x, y)`` to its true SUB-PIXEL center by a local 2D-Gaussian fit
    (intensity-weighted centroid fallback) -- the real trap center is NOT on a pixel, so returning
    the integer argmax (or snapping to a grid) is what makes a site map look crooked.  Mirrors the
    Rb87 readout's per-site refinement; returns the input unchanged if the box is too small."""
    h, w = image.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    cut = image[y0:y1, x0:x1]
    if cut.size < 9 or not np.isfinite(cut).any():
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    bg = float(np.nanmedian(cut))
    amp = float(np.nanmax(cut) - bg)
    try:
        p0 = [bg, max(amp, 1e-6), float(x), float(y), 0.9, 0.9]
        lo = [np.nanmin(cut) - abs(amp) - 1, 0.0, x0 - 0.5, y0 - 0.5, 0.2, 0.2]
        hi = [np.nanmax(cut) + abs(amp) + 1, max(amp * 5, 1.0), x1 - 0.5, y1 - 0.5, 4.0, 4.0]
        popt, _ = curve_fit(_gauss2d_axis, (xx.ravel(), yy.ravel()), cut.ravel(), p0=p0, bounds=(lo, hi), maxfev=5000)
        return float(popt[2]), float(popt[3])
    except Exception:
        vals = np.clip(cut - np.nanpercentile(cut, 20), 0, None)
        s = float(np.sum(vals))
        if s <= 0:
            return float(x), float(y)
        return float(np.sum(xx * vals) / s), float(np.sum(yy * vals) / s)


def find_site_centers(
    image,
    grid_shape: Sequence[int],
    *,
    min_distance: int | None = None,
    threshold_rel: float = 0.35,
    ordering: str = "row-major",
    refine_half: int = 2,
) -> np.ndarray:
    """Find bright trap centers and sort them into a stable site order.

    Each detected peak is refined to its SUB-PIXEL center by a local 2D-Gaussian fit
    (:func:`_refine_center_subpixel`) -- the true trap center is not on a pixel, so the returned
    centers are the real continuous positions, NOT integer peaks forced onto a grid.  Only a
    never-loaded dark site (no peak) is interpolated onto its lattice node so its PSF box fits."""

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
    is_peak = smooth == local_max
    candidates_yx = np.argwhere(is_peak & (smooth >= cutoff))
    if len(candidates_yx) < need:
        # Fewer peaks clear the relative cutoff than there are traps -- the usual cause
        # is a dim site in a NON-UNIFORM averaged reference (each trap was loaded a
        # different number of shots).  Fall back to the ``need`` BRIGHTEST LOCAL MAXIMA,
        # which are spatially separated (one per trap), NOT the brightest PIXELS: the
        # hottest blobs span several pixels each, so a brightest-pixel pick collapses
        # every center onto a handful of spots and destroys the site grid.
        peaks_yx = np.argwhere(is_peak)
        if len(peaks_yx) >= need:
            peak_weights = smooth[peaks_yx[:, 0], peaks_yx[:, 1]]
            candidates_yx = peaks_yx[np.argsort(peak_weights)[::-1][:need]]
        else:                       # genuinely peak-poor image: last-resort brightest pixels
            flat = np.argsort(smooth.ravel())[::-1][:need]
            candidates_yx = np.column_stack(np.unravel_index(flat, smooth.shape))
    weights = smooth[candidates_yx[:, 0], candidates_yx[:, 1]]
    selected = candidates_yx[np.argsort(weights)[::-1]][:need]
    # Refine EACH detected peak from its integer argmax to its true SUB-PIXEL center (local
    # 2D-Gaussian fit) -- the real trap center is not on a pixel, so this is what keeps the site
    # map from looking crooked, WITHOUT forcing centers onto a grid.
    centers_xy = np.array([_refine_center_subpixel(img, float(c), float(r), half=refine_half)
                           for r, c in selected], dtype=float)
    # Sort into the regular array; the lattice repair only REPLACES a never-loaded dark site's
    # border artifact (a center far off its fitted node), leaving every detected sub-pixel center.
    row_major = sort_centers_grid(centers_xy, (ny, nx), ordering="row-major")
    repaired = _regularize_grid(row_major, (ny, nx), img.shape)
    return sort_centers_grid(repaired, (ny, nx), ordering=ordering)


def _robust_axis_lattice(anchors: np.ndarray, n: int) -> np.ndarray:
    """Fit ONE regular axis (origin + pitch) to ``n`` per-line anchor coordinates and return
    the lattice value for every index 0..n-1.  Uses a Theil-Sen slope (median of pairwise
    (a_j - a_i)/(j - i)) and a median-residual origin, so a few OUTLIER lines -- an
    all-dark row/column whose anchor is an image-border artifact -- cannot tip the fit;
    those lines get an INTERPOLATED interior node instead of staying at the edge.  ``n == 1``
    has no pitch to fit, so the single line keeps its anchor."""
    anchors = np.asarray(anchors, dtype=float)
    if n <= 1:
        return anchors.copy()
    idx = np.arange(n, dtype=float)
    slopes = [(anchors[j] - anchors[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    pitch = float(np.median(slopes)) if slopes else 0.0
    origin = float(np.median(anchors - pitch * idx))
    return origin + pitch * idx


def _regularize_grid(centers_row_major, grid_shape: Sequence[int], image_shape) -> np.ndarray:
    """Replace only the centers detection MISPLACED -- keep every well-detected sub-pixel center.

    The traps are an axis-aligned regular grid, so the lattice is separable: every row shares
    one y, every column one x.  A site that emitted no light this calibration has no peak to
    find, so its detected "center" lands on an image-border ``maximum_filter`` artifact and a
    downstream PSF box cannot fit.  Each axis is fit GLOBALLY (robust origin + pitch via
    :func:`_robust_axis_lattice`), so even a WHOLE dark row/column is interpolated to its
    interior node.  But a DETECTED site's own SUB-PIXEL center (``find_site_centers`` refines
    every peak with a local 2D-Gaussian fit BEFORE this) is the REAL position -- the true spot
    center is not on a pixel and not rigidly on the grid, and forcing it onto the fitted lattice
    is exactly what makes the site map look crooked/rigid.  So this only REPLACES a center more
    than half a cell from its fitted node (a dark-site / border artifact) with the lattice node;
    confidently-detected sub-pixel centers are left untouched.  Finally every node is clamped a
    quarter-cell inside the frame, so no returned center sits on the border where a box won't fit."""

    ny, nx = grid_shape_tuple(grid_shape)
    g = np.asarray(centers_row_major, dtype=float).reshape(ny, nx, 2)
    h, w = int(image_shape[0]), int(image_shape[1])
    row_y = _robust_axis_lattice(np.median(g[:, :, 1], axis=1), ny)   # one y per row
    col_x = _robust_axis_lattice(np.median(g[:, :, 0], axis=0), nx)   # one x per column
    pitches = []
    if ny > 1:
        pitches.append(abs(float(row_y[1] - row_y[0])))
    if nx > 1:
        pitches.append(abs(float(col_x[1] - col_x[0])))
    pitches = [p for p in pitches if np.isfinite(p) and p > 0]
    pitch = float(min(pitches)) if pitches else float(min(h, w))
    lattice_x = np.broadcast_to(col_x[None, :], (ny, nx))
    lattice_y = np.broadcast_to(row_y[:, None], (ny, nx))
    # Keep the sub-pixel fit for every confidently-detected site; replace ONLY a center that is
    # more than half a cell off its fitted node (a never-loaded dark site whose "peak" is a
    # border artifact) with the interpolated lattice node so its PSF box still fits.
    off_node = np.hypot(g[:, :, 0] - lattice_x, g[:, :, 1] - lattice_y) > 0.5 * pitch
    g[off_node, 0] = lattice_x[off_node]
    g[off_node, 1] = lattice_y[off_node]
    # Interior clamp: any node within a quarter-cell of the border is pulled in so a PSF box fits.
    margin = max(1.0, 0.25 * pitch)
    g[:, :, 0] = np.clip(g[:, :, 0], margin, max(margin, w - 1 - margin))
    g[:, :, 1] = np.clip(g[:, :, 1], margin, max(margin, h - 1 - margin))
    return g.reshape(ny * nx, 2)


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
    # A well-separated bimodal has an EMPTY gap between the peaks: across the gap no
    # probability mass moves, so the Otsu score is FLAT (a plateau).  ``argmax`` would
    # return the LEFTMOST plateau bin -- the threshold then sits at the top of the dark
    # peak, where a dark tail misclassifies.  Return the CENTRE of the optimal plateau
    # instead (the middle of the gap), the robust split a balanced threshold wants.
    best = float(np.max(score[valid]))
    plateau = np.flatnonzero(valid & (score >= best - 1e-9 * (abs(best) + 1.0)))
    return float(np.mean(centers[plateau]))


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
    # Confidence-weighted fidelity, shared with the live histogram (one formula,
    # so the saved-calibration fidelity and the GUI "fit F" can never disagree).
    fidelity, _raw, separation = confidence_weighted_fidelity(
        threshold, mu0, s0, float(len(left)), mu1, s1, float(len(right))
    )
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
