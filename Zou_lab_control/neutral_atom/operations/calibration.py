"""Standalone sitemap and threshold calibration operations."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.analysis import (
    estimate_threshold_fidelity,
    estimate_thresholds,
    find_site_centers,
    grid_shape_tuple,
    roi_counts,
)
from ..core.calibration import TrapCalibration
from ..core.results import SitemapResult, ThresholdResult
from ..core.utils import site_index
from ..views.plots import plot_image, plot_threshold_hist


def calibrate_sitemap_from_images(
    images,
    *,
    grid_shape: Sequence[int],
    ordering: str = "row-major",
    roi_radius: int = 1,
    reducer: str = "mean",
    display: bool = True,
) -> SitemapResult:
    """Calibrate site centers from a stack of images without requiring a session."""

    stack = [np.asarray(image, dtype=float) for image in images]
    if not stack:
        raise ValueError("images must contain at least one frame.")
    if any(frame.shape != stack[0].shape for frame in stack):
        raise ValueError("all sitemap frames must have the same shape.")
    grid_shape = grid_shape_tuple(grid_shape)
    average = np.mean(np.stack(stack, axis=0), axis=0)
    centers = find_site_centers(average, grid_shape, ordering=ordering)
    thresholds = np.zeros(len(centers), dtype=float)
    calibration = TrapCalibration(
        centers,
        thresholds,
        grid_shape=grid_shape,
        roi_radius=roi_radius,
        reducer=reducer,
        metadata={"stage": "sitemap", "thresholds_calibrated": False},
    )
    plot = plot_image(average, centers=centers, roi_radius=roi_radius, display=display)
    return SitemapResult(calibration, average, stack, plot=plot)


def calibrate_threshold_from_images(
    images,
    calibration: TrapCalibration,
    *,
    site: int = 0,
    display: bool = True,
) -> ThresholdResult:
    """Calibrate per-site thresholds from images and an existing sitemap."""

    stack = [np.asarray(image, dtype=float) for image in images]
    if not stack:
        raise ValueError("images must contain at least one frame.")
    counts = np.vstack(
        [roi_counts(image, calibration.centers, radius=calibration.roi_radius, reducer=calibration.reducer) for image in stack]
    )
    thresholds = estimate_thresholds(stack, calibration.centers, radius=calibration.roi_radius, reducer=calibration.reducer)
    updated = calibration.with_thresholds(thresholds, stage="threshold", thresholds_calibrated=True)
    site = site_index(site, counts.shape[1])
    fidelity = estimate_threshold_fidelity(counts[:, site], thresholds[site])
    plot = plot_threshold_hist(counts[:, site], threshold=thresholds[site], labels=(f"Site {site} ROI counts", "Shots", "Population"), display=display)
    return ThresholdResult(updated, counts, thresholds, site, plot=plot, fidelity=fidelity)


__all__ = ["calibrate_sitemap_from_images", "calibrate_threshold_from_images"]
