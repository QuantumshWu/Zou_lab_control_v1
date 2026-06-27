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
from ..core.psf import fit_site_psfs, fit_uniform_psf, psf_boxes_array, psf_weights_array
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
    if method not in ("box", "psf", "uniform_psf"):
        raise ValueError("method must be 'box', 'psf' (per-site) or 'uniform_psf' (one shared kernel).")
    average = np.mean(np.stack(stack, axis=0), axis=0)
    centers = find_site_centers(average, grid_shape, ordering=ordering)
    thresholds = np.zeros(len(centers), dtype=float)
    # Fingerprint the frame geometry the centers were detected on, so a later
    # ROI change is caught at readout (TrapCalibration.signals) instead of
    # silently extracting the wrong pixels.
    image_shape = [int(average.shape[0]), int(average.shape[1])]

    if method in ("psf", "uniform_psf"):
        # Per-site fits one independent kernel per spot; uniform fits ONE shared
        # kernel reused by every site (right when the spots share one shape).  Both
        # store method='psf' on the calibration (the same matched-filter readout
        # path) -- the psf_mode metadata records which kernel model produced it.
        uniform = method == "uniform_psf"
        psfs = (fit_uniform_psf if uniform else fit_site_psfs)(average, centers, half_width=psf_half_width)
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
            metadata={"stage": "sitemap", "thresholds_calibrated": False, "method": "psf",
                      "psf_mode": "uniform" if uniform else "per_site",
                      "psf_half_width": int(psf_half_width), "image_shape": image_shape},
        )
    else:
        calibration = TrapCalibration(
            centers,
            thresholds,
            grid_shape=grid_shape,
            roi_radius=roi_radius,
            reducer=reducer,
            method="box",
            metadata={"stage": "sitemap", "thresholds_calibrated": False, "method": "box", "image_shape": image_shape},
        )
    plot = plot_image(average, centers=centers, roi_radius=roi_radius, display=display)
    return SitemapResult(calibration, average, stack, plot=plot)


def calibrate_threshold_from_images(
    images,
    calibration: TrapCalibration,
    *,
    site: int = 0,
    method: str = "otsu",
    exposure: float | None = None,
    display: bool = True,
) -> ThresholdResult:
    """Calibrate per-site thresholds from images and an existing sitemap.

    Signals are extracted with ``calibration.signals`` (box or PSF, matching the
    sitemap), so thresholds apply to the same quantity ``detect`` will use.

    This is the ONE place thresholds are learned (every calibration path -- the
    ``ReadoutSubsystem.thresholds`` API AND the ``CalibrateReadoutTask`` GUI flow -- routes
    through it), so it is the single source that stamps ``threshold_exposure`` (the camera
    gate time the bright counts were learnt at, passed by the caller).  A threshold is
    exposure-specific, so any later readout (e.g. the temperature survival frames) must image
    at this SAME exposure; recording it here lets that match happen automatically (#H3w-3).
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

    if exposure is not None:
        extra["threshold_exposure"] = float(exposure)   # the gate time these thresholds are valid at
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


#: The readout methods a one-shot calibration computes so the OccupancyProcessor can
#: pick any of them later (box square-ROI, per-site PSF, one-shared-kernel PSF).  Single
#: source -- the cali, the calibration's ``by_method`` and the processor choice derive
#: from this, never a retyped literal.
ALL_READOUT_METHODS = ("box", "psf", "uniform_psf")


def calibrate_all_methods_from_images(
    reference_images,
    readout_images,
    *,
    grid_shape: Sequence[int],
    ordering: str = "row-major",
    roi_radius: int = 1,
    reducer: str = "mean",
    threshold_method: str = "otsu",
    psf_half_width: int = 3,
    background: str = "annulus",
    exposure: float | None = None,
) -> TrapCalibration:
    """Calibrate ONCE into a single calibration that can be read with EVERY method.

    The site map is found once (shared centers, from the reference average); then box,
    per-site-PSF and uniform-PSF readouts are each calibrated (their kernels + per-site
    thresholds) on the SAME readout frames.  Box is the default top-level readout; the
    PSF methods are carried in ``by_method``.  So the experimenter calibrates once and the
    downstream :class:`~..logic.OccupancyProcessor` chooses box / per-site PSF / uniform
    PSF at read time -- the method is a READOUT choice, not a calibration choice."""

    cals = {}
    for method in ALL_READOUT_METHODS:
        sitemap = calibrate_sitemap_from_images(
            reference_images, grid_shape=grid_shape, ordering=ordering, roi_radius=roi_radius,
            reducer=reducer, method=method, psf_half_width=psf_half_width,
            background=background, display=False)
        result = calibrate_threshold_from_images(
            readout_images, sitemap.calibration, method=threshold_method, exposure=exposure, display=False)
        cals[method] = result.calibration
    box = cals["box"]
    by_method = {
        m: {"thresholds": np.asarray(cals[m].thresholds, dtype=float),
            "psf_weights": cals[m].psf_weights, "psf_boxes": cals[m].psf_boxes,
            # carry each method's OWN background so signals(method=m) reads on the scale its
            # thresholds were calibrated on (a psf method's annulus, not box's none).
            "background": cals[m].background}
        for m in ALL_READOUT_METHODS if m != "box"
    }
    return TrapCalibration(
        box.centers, box.thresholds, grid_shape=box.grid_shape, roi_radius=box.roi_radius,
        reducer=box.reducer, method="box", background=box.background, by_method=by_method,
        metadata={**box.metadata, "methods": list(ALL_READOUT_METHODS)})


__all__ = ["calibrate_sitemap_from_images", "calibrate_threshold_from_images",
           "calibrate_all_methods_from_images", "ALL_READOUT_METHODS", "SUPPORTED_THRESHOLD_METHODS"]
