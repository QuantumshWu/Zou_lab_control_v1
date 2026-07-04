"""Core data structures and image-analysis algorithms."""

from .analysis import (
    AtomDetection,
    FidelityEstimate,
    estimate_threshold_fidelity,
    find_site_centers,
    otsu_threshold,
    roi_counts,
    sort_centers_grid,
)
from .bimodal import BimodalFit, exact_otsu_threshold, fit_bimodal, fit_bimodal_per_site, gaussian_fidelity, optimal_gaussian_threshold
from .calibration import TrapCalibration
from .psf import SitePSF, fit_site_psfs, psf_boxes_array, psf_signals, psf_weights_array
from .results import (
    CaptureResult,
    DetectionResult,
    DetectionTimeScanResult,
    MeasurementTaskResult,
    PreflightReport,
    ResultObject,
    SitemapResult,
    ThresholdResult,
)

__all__ = [
    "AtomDetection",
    "BimodalFit",
    "CaptureResult",
    "DetectionResult",
    "DetectionTimeScanResult",
    "FidelityEstimate",
    "MeasurementTaskResult",
    "PreflightReport",
    "ResultObject",
    "SitePSF",
    "SitemapResult",
    "ThresholdResult",
    "TrapCalibration",
    "estimate_threshold_fidelity",
    "exact_otsu_threshold",
    "find_site_centers",
    "fit_bimodal",
    "fit_bimodal_per_site",
    "fit_site_psfs",
    "gaussian_fidelity",
    "optimal_gaussian_threshold",
    "otsu_threshold",
    "psf_boxes_array",
    "psf_signals",
    "psf_weights_array",
    "roi_counts",
    "sort_centers_grid",
]
