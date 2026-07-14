"""Standalone sitemap and threshold calibration operations.

These build a :class:`~..core.calibration.TrapCalibration` from image stacks
without a session.  ``method`` selects the readout the calibration will carry:

* sitemap  ``method='box'`` (default) stores square-ROI readout; ``'psf'`` fits a
  per-site PSF weight from the all-sites average and stores matched-filter readout;
  ``'uniform_psf'`` fits ONE shared kernel reused by every site (stored verbatim --
  the calibration's ``method`` names exactly what was calibrated).
* threshold ``method='otsu'`` (default) is the single-split threshold; ``'bimodal'``
  fits dark/bright Gaussian cores per site (the Rb87 readout).  Either way the
  per-site signal is extracted with ``calibration.signals`` so the threshold is
  computed on exactly the quantity ``detect`` will compare against.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ..core.analysis import (
    centers_array,
    estimate_threshold_fidelity,
    find_site_centers,
    grid_shape_tuple,
    otsu_threshold,
)
from ..core.bimodal import fit_bimodal_per_site
from ..core.calibration import FrameContract, READOUT_KINDS, TrapCalibration, readout_kind
from ..core.psf import fit_site_psfs, fit_uniform_psf, psf_boxes_array, psf_weights_array
from ..core.results import SitemapResult, ThresholdResult
from ..core.utils import site_index
from ..views.plots import plot_image, plot_threshold_hist

SUPPORTED_THRESHOLD_METHODS = ("otsu", "bimodal")

#: Readout methods supported by the standalone sitemap builder (box square-ROI,
#: per-site PSF, one shared-kernel PSF).  The calibration and detector dispatch
#: derive from this vocabulary rather than retyping it.
ALL_READOUT_METHODS = ("box", "psf", "uniform_psf")

# Every offered method MUST declare its readout KIND in the core dispatch table, so
# ``calibration.signals`` routes it by explicit kind (box ROI vs kernel matched-filter)
# rather than a ``"psf" in name`` substring.  This guard catches a new method added here
# but not registered in READOUT_KINDS at import time (it would otherwise dispatch wrong).
_unregistered = tuple(m for m in ALL_READOUT_METHODS if m not in READOUT_KINDS)
if _unregistered:
    raise RuntimeError(
        f"readout methods {_unregistered} are not in core.calibration.READOUT_KINDS -- "
        "register each method's readout kind there so signals() dispatches it explicitly.")
del _unregistered


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
    centers=None,
) -> SitemapResult:
    """Calibrate site centers (and, for ``method='psf'``, per-site PSF weights).

    ``centers`` optionally reuses a site map already detected on the same
    reference frames: centers are a property of the atom lattice, not the readout
    method, so callers can share one detection across readout models."""

    stack = [np.asarray(image, dtype=float) for image in images]
    if not stack:
        raise ValueError("images must contain at least one frame.")
    if any(frame.shape != stack[0].shape for frame in stack):
        raise ValueError("all sitemap frames must have the same shape.")
    grid_shape = grid_shape_tuple(grid_shape)
    method = str(method).lower()
    if method not in ALL_READOUT_METHODS:
        raise ValueError("method must be 'box', 'psf' (per-site) or 'uniform_psf' (one shared kernel).")
    average = np.mean(np.stack(stack, axis=0), axis=0)
    if centers is None:
        centers = find_site_centers(average, grid_shape, ordering=ordering)
    else:
        # Reuse a site map detected on the SAME frames -- a property of the lattice, not the readout,
        # so re-detecting per method would be wasted work AND a latent single-source risk (three
        # "independent" detections that only happen to agree).  Count-checked against grid_shape.
        centers = centers_array(centers)
        if len(centers) != int(np.prod(grid_shape)):
            raise ValueError("provided centers count must match the grid_shape product.")
    thresholds = np.zeros(len(centers), dtype=float)
    # Fingerprint the frame geometry the centers were detected on, so a later
    # ROI change is caught at readout (TrapCalibration.signals) instead of
    # silently extracting the wrong pixels.
    frame_contract = FrameContract(image_shape=tuple(int(v) for v in average.shape))

    if readout_kind(method) == "kernel":
        # Per-site fits one independent kernel per spot; uniform fits ONE shared
        # kernel reused by every site (right when the spots share one shape).  The
        # calibration stores the method VERBATIM ('psf' / 'uniform_psf'), so
        # methods() offers it and signals(method=...) resolves it -- exactly like
        # the multi-method path's by_method keys (one naming, no collapse).
        uniform = method == "uniform_psf"
        psfs = (fit_uniform_psf if uniform else fit_site_psfs)(average, centers, half_width=psf_half_width)
        # roi_radius / reducer are the BOX extraction geometry; a PSF readout reads through its
        # kernels and ignores them (TrapCalibration drops them to None for a non-box method), so
        # they are NOT passed here -- a PSF calibration carries no dead box-only state.
        calibration = TrapCalibration(
            centers,
            thresholds,
            frame_contract=frame_contract,
            grid_shape=grid_shape,
            method=method,
            psf_weights=psf_weights_array(psfs),
            psf_boxes=psf_boxes_array(psfs),
            background=background,
            metadata={"stage": "sitemap", "thresholds_calibrated": False, "method": method,
                      "psf_half_width": int(psf_half_width)},
        )
    else:
        calibration = TrapCalibration(
            centers,
            thresholds,
            frame_contract=frame_contract,
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
    exposure: float | None = None,
    display: bool = True,
) -> ThresholdResult:
    """Calibrate per-site thresholds from images and an existing sitemap.

    Signals are extracted with ``calibration.signals`` (box or PSF, matching the
    sitemap), so thresholds apply to the same quantity ``detect`` will use.

    This is the standalone API's one threshold-learning boundary, so it records
    the camera gate time in the typed frame contract.  A threshold is
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

    updated = calibration.with_thresholds(thresholds, stage="threshold", thresholds_calibrated=True, **extra)
    if exposure is not None:
        updated = replace(
            updated,
            frame_contract=replace(updated.frame_contract, exposure_s=float(exposure)),
        )
    site = site_index(site, counts.shape[1])
    fidelity = estimate_threshold_fidelity(counts[:, site], thresholds[site])
    plot = plot_threshold_hist(
        counts[:, site],
        threshold=thresholds[site],
        labels=(f"Site {site} signal", "Shots", "Population"),
        display=display,
    )
    return ThresholdResult(updated, counts, thresholds, site, plot=plot, fidelity=fidelity)





__all__ = [
    "calibrate_sitemap_from_images",
    "calibrate_threshold_from_images",
    "ALL_READOUT_METHODS",
    "SUPPORTED_THRESHOLD_METHODS",
]
