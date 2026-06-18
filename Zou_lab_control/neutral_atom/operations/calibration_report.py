"""Write a readout-calibration REPORT to a folder -- the Rb87-style artifacts.

After a calibration the experimenter wants the SAME things the Rb87 readout pipeline
saves: a per-site count DISTRIBUTION grid (one histogram per site, with its threshold),
a pooled GLOBAL distribution annotated with the single-shot FIDELITY, the averaged site
template image, and an ``.npz`` + ``summary.json`` bundle -- all written to one folder so
a run is reproducible and reviewable offline.

Split by layer, no drift:
  * the NUMBERS (per-site counts via the ``calibration.signals`` contract, per-site
    fidelity, the ``.npz`` + ``summary.json`` + a reloadable ``calibration.json``) are
    computed HERE -- analysis only, identical virtual vs real (only the frames differ);
  * the FIGURES are drawn by the FRONTEND plot types (the distribution grid / pooled
    histogram / site map the live console uses), routed through the viewer-registry seam
    (:func:`Zou_lab_control._viewer_registry.active_plotter`).  This layer imports no
    frontend and no camera backend, so the package decoupling holds; when no viewer is
    registered (pure headless), the data bundle is still complete and the PNGs are simply
    skipped (the registry's documented fallback).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Zou_lab_control._viewer_registry import active_plotter

from ..core.analysis import estimate_threshold_fidelity


def per_site_counts(calibration, frames) -> np.ndarray:
    """``(n_frames, n_sites)`` readout signal -- the SAME ``calibration.signals`` the
    detector thresholds, stacked over the readout frames."""
    rows = [np.asarray(calibration.signals(np.asarray(f, dtype=float)), dtype=float).reshape(-1)
            for f in frames]
    if not rows:
        return np.empty((0, len(np.asarray(calibration.centers))), dtype=float)
    return np.vstack(rows)


def per_site_fidelity(counts: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per-site single-shot readout fidelity: the two-population Gaussian-model fidelity
    of each site's count distribution about its calibrated threshold (the live quick-look
    estimate -- the SAME model ``OtsuFidelityReducer`` uses on a detection-time scan)."""
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    thr = np.asarray(thresholds, dtype=float).reshape(-1)
    out = np.full(counts.shape[1], np.nan, dtype=float)
    for j in range(counts.shape[1]):
        col = counts[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        f = float(estimate_threshold_fidelity(col, float(thr[j])).fidelity)
        out[j] = f if np.isfinite(f) else np.nan
    return out


def _render_figures(folder, *, counts, thresholds, fidelity, centers, template,
                    threshold_method, timestamp) -> dict:
    """Draw the report PNGs through the registered FRONTEND viewer; ``{}`` when headless
    (no viewer registered) -- the data bundle below is still written either way."""
    plotter = active_plotter()
    if plotter is None or not hasattr(plotter, "save_calibration_report"):
        return {}
    try:
        return dict(plotter.save_calibration_report(
            folder, counts=counts, thresholds=thresholds, fidelity=fidelity,
            centers=centers, template=template,
            threshold_method=str(threshold_method), timestamp=str(timestamp)) or {})
    except Exception:
        return {}


def write_calibration_report(folder, *, calibration, readout_frames, template=None,
                             threshold_method: str = "otsu", timestamp: str = "") -> dict:
    """Write the per-site distribution grid + global distribution(+fidelity) + site-map
    PNGs (via the frontend) and a ``calibration.npz`` + ``calibration.json`` +
    ``summary.json`` (the numbers) into ``folder`` (created if needed).

    Returns a summary dict (paths + n_sites + mean fidelity).  The numbers are derived
    purely from the calibration contract + readout frames (identical on real hardware);
    the figures come from the registered viewer, so they use the live console's plot
    types rather than hand-rolled matplotlib."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(calibration.centers, dtype=float)
    thresholds = np.asarray(calibration.thresholds, dtype=float).reshape(-1)
    counts = per_site_counts(calibration, readout_frames)
    fidelity = (per_site_fidelity(counts, thresholds) if counts.size
                else np.full(len(centers), np.nan))

    paths = _render_figures(
        folder, counts=counts, thresholds=thresholds, fidelity=fidelity, centers=centers,
        template=template, threshold_method=threshold_method, timestamp=timestamp)

    npz_path = folder / "calibration.npz"
    np.savez(npz_path, centers=centers, thresholds=thresholds,
             counts=counts, per_site_fidelity=fidelity)
    paths["npz"] = str(npz_path)
    # the loadable calibration artifact (so a downstream OccupancyProcessor can reuse it)
    cal_path = folder / "calibration.json"
    try:
        calibration.save(cal_path)
        paths["calibration"] = str(cal_path)
    except Exception:
        pass

    summary = {
        "n_sites": int(len(centers)),
        "readout_frames": int(counts.shape[0]) if counts.size else 0,
        "mean_fidelity": (float(np.nanmean(fidelity)) if np.any(np.isfinite(fidelity)) else None),
        "worst_site_fidelity": (float(np.nanmin(fidelity)) if np.any(np.isfinite(fidelity)) else None),
        "threshold_method": str(threshold_method),
        "timestamp": str(timestamp),
        "files": paths,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["folder"] = str(folder)
    return summary


__all__ = ["write_calibration_report", "per_site_counts", "per_site_fidelity"]
