"""PSF-weighted site readout primitives.

The box reducer in :mod:`.analysis` (``roi_counts``) sums a square ROI equally.
A point-spread-function (PSF) extractor instead weights each pixel by the site's
own normalized spot, which is the matched filter for a known shape in additive
noise -- it suppresses background pixels far from the atom and recovers more
signal-to-noise per shot.  This is the core of the Rb87 qCMOS readout.

Two stages, both pure functions (no session, no frontend):

* ``fit_site_psfs(template, centers, ...)`` -- calibration-time: fit one
  normalized PSF weight per site from a high-SNR averaged image (every site
  occupied).  Returns a :class:`SitePSF` list and the packed ``(weights, boxes)``
  arrays a :class:`~..core.calibration.TrapCalibration` stores.
* ``psf_signals(image, weights, boxes, ...)`` -- per-shot: one matched-filter
  scalar per site (the dot product of the background-subtracted cutout with the
  site weight).

Coordinates are floating-point ``(x, y)`` pixels; numpy indexing is ``image[y, x]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit

from .analysis import centers_array, image_array, nonnegative_int


@dataclass(frozen=True)
class SitePSF:
    """One site's normalized PSF weight and the integer box it lives in."""

    index: int
    x0: float
    y0: float
    box: tuple[int, int, int, int]  # (x0, y0, w, h) integer pixel box
    weight: np.ndarray              # (h, w) >= 0, sums to 1
    sigma_x: float
    sigma_y: float
    fit_ok: bool

    @property
    def n_eff_pixels(self) -> float:
        s2 = float(np.sum(self.weight ** 2))
        return float(1.0 / s2) if s2 > 0 else float("nan")


def crop_box(shape: tuple[int, int], x: float, y: float, half_width: int) -> tuple[int, int, int, int]:
    """Integer ``(x0, y0, w, h)`` box of half-width ``half_width`` around ``(x, y)``."""

    h, w = int(shape[0]), int(shape[1])
    half = nonnegative_int(half_width, "half_width")
    xi, yi = int(round(float(x))), int(round(float(y)))
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    return x0, y0, x1 - x0, y1 - y0


def annulus_background(image: np.ndarray, box: tuple[int, int, int, int], *, pad: int = 3) -> float:
    """Median of the ring of pixels just outside ``box`` (a local background)."""

    img = image_array(image)
    h, w = img.shape
    x, y, bw, bh = (int(v) for v in box)
    x0, x1 = max(0, x - pad), min(w, x + bw + pad)
    y0, y1 = max(0, y - pad), min(h, y + bh + pad)
    region = img[y0:y1, x0:x1]
    mask = np.ones(region.shape, dtype=bool)
    mask[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = False
    vals = region[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        finite = img[np.isfinite(img)]
        return float(np.median(finite)) if finite.size else 0.0
    return float(np.median(vals))


def gaussian_psf(shape: tuple[int, int], x0: float, y0: float, sx: float, sy: float) -> np.ndarray:
    """A normalized 2D Gaussian on a ``shape`` grid centered at local ``(x0, y0)``."""

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    g = np.exp(-0.5 * (((xx - x0) / max(sx, 1e-6)) ** 2 + ((yy - y0) / max(sy, 1e-6)) ** 2))
    g = np.clip(g, 0, None)
    total = float(np.sum(g))
    return g / total if total > 0 else np.ones(shape, dtype=float) / float(np.prod(shape))


def _gauss2d(coords: tuple[np.ndarray, np.ndarray], offset: float, amp: float, x0: float, y0: float, sx: float, sy: float) -> np.ndarray:
    x, y = coords
    return (offset + amp * np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))).ravel()


def fit_site_psfs(
    template,
    centers,
    *,
    half_width: int = 3,
    bg_padding: int = 3,
    weight_smooth_sigma: float = 0.35,
    weight_source: str = "empirical",
) -> list[SitePSF]:
    """Fit one normalized PSF weight per site from a high-SNR template image.

    ``template`` is an averaged image with every site bright (e.g. the sitemap
    average).  For each center a ``(2*half_width+1)`` box is cut, a local annulus
    background subtracted, and a 2D Gaussian fit for the sub-pixel center/widths.
    The stored weight is the background-subtracted, positive, smoothed, L1-
    normalized cutout (``weight_source='empirical'``, the matched filter for the
    real spot shape) or the fitted Gaussian (``'gaussian'``).

    Every box must lie fully inside the image (uniform shape), so the weights
    pack into one ``(N, h, w)`` array; a center too close to an edge raises.
    """

    img = image_array(template)
    centers = centers_array(centers)
    from scipy.ndimage import gaussian_filter

    box_dim = 2 * nonnegative_int(half_width, "half_width") + 1
    out: list[SitePSF] = []
    for index, (x, y) in enumerate(centers):
        box = crop_box(img.shape, x, y, half_width)
        bx, by, bw, bh = box
        if (bw, bh) != (box_dim, box_dim):
            raise ValueError(
                f"site {index} at ({x:g}, {y:g}) is too close to the image edge for a "
                f"{box_dim}x{box_dim} PSF box; PSF readout needs full boxes for every site."
            )
        cut = img[by:by + bh, bx:bx + bw]
        bg = annulus_background(img, box, pad=bg_padding)
        sub = cut - bg
        yy, xx = np.mgrid[by:by + bh, bx:bx + bw]
        amp = float(np.nanmax(sub)) if np.isfinite(sub).any() else 0.0
        xf, yf, sx, sy, fit_ok = float(x), float(y), 0.9, 0.9, False
        try:
            p0 = [0.0, max(amp, 1e-6), float(x), float(y), 0.9, 0.9]
            lo = [float(np.nanmin(sub)) - abs(amp) - 1, 0.0, bx - 0.5, by - 0.5, 0.2, 0.2]
            hi = [float(np.nanmax(sub)) + abs(amp) + 1, max(amp * 5, 1.0), bx + bw - 0.5, by + bh - 0.5, 4.0, 4.0]
            popt, _ = curve_fit(_gauss2d, (xx.ravel(), yy.ravel()), sub.ravel(), p0=p0, bounds=(lo, hi), maxfev=5000)
            _, _, xf, yf, sx, sy = popt
            sx, sy, fit_ok = float(abs(sx)), float(abs(sy)), True
        except Exception:
            vals = np.clip(sub - np.nanpercentile(sub, 20), 0, None)
            if np.sum(vals) > 0:
                xf = float(np.sum(xx * vals) / np.sum(vals))
                yf = float(np.sum(yy * vals) / np.sum(vals))

        if weight_source == "gaussian":
            weight = gaussian_psf((bh, bw), xf - bx, yf - by, sx, sy)
        else:
            pos = np.clip(sub, 0, None)
            if weight_smooth_sigma and weight_smooth_sigma > 0:
                pos = gaussian_filter(pos, weight_smooth_sigma)
            total = float(np.sum(pos))
            weight = pos / total if total > 0 else gaussian_psf((bh, bw), xf - bx, yf - by, sx, sy)
        out.append(SitePSF(index=index, x0=xf, y0=yf, box=box, weight=np.ascontiguousarray(weight, dtype=float), sigma_x=sx, sigma_y=sy, fit_ok=fit_ok))
    return out


def psf_weights_array(psfs: Sequence[SitePSF]) -> np.ndarray:
    """Pack site weights into one ``(N, h, w)`` array (raises on ragged boxes)."""

    if not psfs:
        raise ValueError("psfs must contain at least one site.")
    shapes = {p.weight.shape for p in psfs}
    if len(shapes) != 1:
        raise ValueError(f"all site PSF weights must share one shape; got {sorted(shapes)}.")
    return np.stack([np.asarray(p.weight, dtype=float) for p in psfs], axis=0)


def psf_boxes_array(psfs: Sequence[SitePSF]) -> np.ndarray:
    """Pack site boxes into one ``(N, 4)`` int array of ``(x0, y0, w, h)``."""

    return np.asarray([[int(v) for v in p.box] for p in psfs], dtype=int)


def psf_signals(image, weights, boxes, *, background: str = "annulus", bg_padding: int = 3) -> np.ndarray:
    """One matched-filter scalar per site for ``image``.

    ``weights`` is ``(N, h, w)`` (each L1-normalized), ``boxes`` is ``(N, 4)`` of
    ``(x0, y0, w, h)``.  For each site the cutout is background-subtracted
    (``'annulus'`` = local ring median, ``'none'`` = raw) and dotted with the
    site weight, giving counts in the same units as the image.
    """

    img = image_array(image)
    weights = np.asarray(weights, dtype=float)
    boxes = np.asarray(boxes, dtype=int)
    if weights.ndim != 3 or boxes.ndim != 2 or boxes.shape[1] != 4 or len(weights) != len(boxes):
        raise ValueError("weights must be (N, h, w) and boxes (N, 4) with matching N.")
    mode = str(background).lower()
    if mode not in {"annulus", "none"}:
        raise ValueError("background must be 'annulus' or 'none'.")
    out = np.empty(len(weights), dtype=float)
    for i, ((x0, y0, bw, bh), weight) in enumerate(zip(boxes, weights)):
        x0, y0, bw, bh = int(x0), int(y0), int(bw), int(bh)
        if weight.shape != (bh, bw):
            raise ValueError(f"site {i} weight shape {weight.shape} != box {(bh, bw)}.")
        if x0 < 0 or y0 < 0 or y0 + bh > img.shape[0] or x0 + bw > img.shape[1]:
            raise ValueError(f"site {i} box {(x0, y0, bw, bh)} is outside image shape {img.shape}.")
        cut = img[y0:y0 + bh, x0:x0 + bw]
        bg = annulus_background(img, (x0, y0, bw, bh), pad=bg_padding) if mode == "annulus" else 0.0
        out[i] = float(np.sum(weight * (cut - bg)))
    return out


__all__ = [
    "SitePSF",
    "annulus_background",
    "crop_box",
    "fit_site_psfs",
    "gaussian_psf",
    "psf_boxes_array",
    "psf_signals",
    "psf_weights_array",
]
