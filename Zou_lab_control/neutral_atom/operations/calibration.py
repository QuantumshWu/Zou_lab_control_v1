"""Standalone sitemap and threshold calibration operations.

These build a :class:`~..core.calibration.TrapCalibration` from image stacks
without a session.  ``method`` selects the readout the calibration will carry:

* sitemap  ``method='box'`` (default) stores square-ROI readout; ``'psf'`` fits a
  per-site PSF weight from the all-sites average and stores matched-filter readout.
* threshold ``method='otsu'`` (default) is the single-split threshold; ``'bimodal'``
  fits dark/bright Gaussian cores per site (the Rb87 readout).  Either way the
  per-site signal is extracted with ``calibration.signals`` so the threshold is
  computed on exactly the quantity ``detect`` will compare against.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.analysis import (
    estimate_threshold_fidelity,
    find_site_centers,
    grid_shape_tuple,
    otsu_threshold,
)
from ..core.bimodal import fit_bimodal_per_site
from ..core.calibration import TrapCalibration
from ..core.psf import fit_site_psfs, psf_boxes_array, psf_weights_array
from ..core.results import SitemapResult, ThresholdResult
from ..core.utils import site_index
from ..views.plots import plot_image, plot_threshold_hist

SUPPORTED_THRESHOLD_METHODS = ("otsu", "bimodal")


def calibrate_sitemap_from_images(
    images,
    *,
    grid_shape: Sequence[int],
    ordering: str = "row-major",
    roi_radius: int = 1,
    reducer: str = "mean",
    method: str = "box",
    psf_half_width: int = 3,
    background: str = "annulus",
    display: bool = True,
) -> SitemapResult:
    """Calibrate site centers (and, for ``method='psf'``, per-site PSF weights)."""

    stack = [np.asarray(image, dtype=float) for image in images]
    if not stack:
        raise ValueError("images must contain at least one frame.")
    if any(frame.shape != stack[0].shape for frame in stack):
        raise ValueError("all sitemap frames must have the same shape.")
    grid_shape = grid_shape_tuple(grid_shape)
    method = str(method).lower()
    if method not in ("box", "psf"):
        raise ValueError("method must be 'box' or 'psf'.")
    average = np.mean(np.stack(stack, axis=0), axis=0)
    centers = find_site_centers(average, grid_shape, ordering=ordering)
    thresholds = np.zeros(len(centers), dtype=float)

    if method == "psf":
        psfs = fit_site_psfs(average, centers, half_width=psf_half_width)
        calibration = TrapCalibration(
            centers,
            thresholds,
            grid_shape=grid_shape,
            roi_radius=roi_radius,
            reducer=reducer,
            method="psf",
            psf_weights=psf_weights_array(psfs),
            psf_boxes=psf_boxes_array(psfs),
            background=background,
            metadata={"stage": "sitemap", "thresholds_calibrated": False, "method": "psf", "psf_half_width": int(psf_half_width)},
        )
    else:
        calibration = TrapCalibration(
            centers,
            thresholds,
            grid_shape=grid_shape,
            roi_radius=roi_radius,
            reducer=reducer,
            method="box",
            metadata={"stage": "sitemap", "thresholds_calibrated": False, "method": "box"},
        )
    plot = plot_image(average, centers=centers, roi_radius=roi_radius, display=display)
    return SitemapResult(calibration, average, stack, plot=plot)


def calibrate_threshold_from_images(
    images,
    calibration: TrapCalibration,
    *,
    site: int = 0,
    method: str = "otsu",
    display: bool = True,
) -> ThresholdResult:
    """Calibrate per-site thresholds from images and an existing sitemap.

    Signals are extracted with ``calibration.signals`` (box or PSF, matching the
    sitemap), so thresholds apply to the same quantity ``detect`` will use.
    """

    stack = [np.asarray(image, dtype=float) for image in images]
    if not stack:
        raise ValueError("images must contain at least one frame.")
    method = str(method).lower()
    if method not in SUPPORTED_THRESHOLD_METHODS:
        raise ValueError(f"threshold method must be one of {SUPPORTED_THRESHOLD_METHODS}.")
    counts = np.vstack([calibration.signals(image) for image in stack])

    if method == "bimodal":
        thresholds, fits = fit_bimodal_per_site(counts)
        # A site whose bimodal fit failed (too few/degenerate samples) falls back
        # to the single-split threshold so the calibration stays fully finite.
        for i, value in enumerate(thresholds):
            if not np.isfinite(value):
                thresholds[i] = otsu_threshold(counts[:, i])
        extra = {
            "threshold_method": "bimodal",
            "site_fidelities": [None if not np.isfinite(f.fidelity) else float(f.fidelity) for f in fits],
        }
    else:
        thresholds = np.asarray([otsu_threshold(counts[:, i]) for i in range(counts.shape[1])], dtype=float)
        extra = {"threshold_method": "otsu"}

    updated = calibration.with_thresholds(thresholds, stage="threshold", thresholds_calibrated=True, **extra)
    site = site_index(site, counts.shape[1])
    fidelity = estimate_threshold_fidelity(counts[:, site], thresholds[site])
    plot = plot_threshold_hist(
        counts[:, site],
        threshold=thresholds[site],
        labels=(f"Site {site} signal", "Shots", "Population"),
        display=display,
    )
    return ThresholdResult(updated, counts, thresholds, site, plot=plot, fidelity=fidelity)


__all__ = ["calibrate_sitemap_from_images", "calibrate_threshold_from_images", "SUPPORTED_THRESHOLD_METHODS"]
