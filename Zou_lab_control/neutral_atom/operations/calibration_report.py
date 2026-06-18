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


def per_site_counts(calibration, frames, *, method=None) -> np.ndarray:
    """``(n_frames, n_sites)`` readout signal -- the SAME ``calibration.signals`` the
    detector thresholds, stacked over the readout frames.  ``method`` picks the readout
    model (box / psf / uniform_psf); ``None`` = the calibration's default."""
    rows = [np.asarray(calibration.signals(np.asarray(f, dtype=float), method=method),
                       dtype=float).reshape(-1)
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
                    threshold_method, timestamp, by_method=None) -> dict:
    """Draw the report PNGs through the registered FRONTEND viewer; ``{}`` when headless
    (no viewer registered) -- the data bundle below is still written either way."""
    plotter = active_plotter()
    if plotter is None or not hasattr(plotter, "save_calibration_report"):
        return {}
    try:
        return dict(plotter.save_calibration_report(
            folder, counts=counts, thresholds=thresholds, fidelity=fidelity,
            centers=centers, template=template, by_method=by_method,
            threshold_method=str(threshold_method), timestamp=str(timestamp)) or {})
    except Exception:
        return {}


def _by_method_report(calibration, readout_frames) -> dict:
    """SELF-CONSISTENT fallback (no ground truth available): for every readout method the
    calibration carries, the per-site counts + that method's otsu threshold + the two-
    population fidelity ESTIMATE.  NOTE: that estimate is affine-invariant, so box / PSF /
    uniform-PSF come out near-identical -- it measures the empirical bimodality of the SAME
    photons, not classification accuracy.  Use :func:`_held_out_by_method` when reference
    (bracket) frames give real labels; this path is only for folder runs that kept no
    reference groups."""
    out: dict[str, dict] = {}
    try:
        methods = tuple(calibration.methods())
    except Exception:
        methods = ()
    for m in methods:
        try:
            counts = per_site_counts(calibration, readout_frames, method=m)
            thr = np.asarray(calibration.thresholds_for(m), dtype=float).reshape(-1)
        except Exception:
            continue
        fid = (per_site_fidelity(counts, thr) if counts.size
               else np.full(len(np.asarray(calibration.centers)), np.nan))
        out[str(m)] = {"counts": counts, "thresholds": thr, "fidelity": fid}
    return out


def _reference_signal_array(calibration, reference_groups, *, method=None) -> np.ndarray:
    """``(n_groups, n_ref, n_sites)`` reference signals -- the high-SNR long bracket frames
    that vote ground truth, extracted with ONE fixed method (box on the long frames) so the
    occupancy LABELS are identical across the readout methods being compared."""
    n_sites = len(np.asarray(calibration.centers))
    groups = [[np.asarray(calibration.signals(np.asarray(f, dtype=float), method=method),
                          dtype=float).reshape(-1) for f in refs]
              for refs in reference_groups]
    if not groups:
        return np.empty((0, 0, n_sites), dtype=float)
    return np.asarray(groups, dtype=float)


def _held_out_by_method(calibration, reference_groups, readout_by_group) -> dict:
    """Per-method HELD-OUT fidelity against bracket-voted ground truth (the real Rb87 flow).

    The long reference frames of each bracket vote a per-(group, site) occupancy label
    (strict consensus -- a shot where they disagree is an atom-loss event, marked ambiguous
    and dropped).  For EACH readout method the per-site threshold is then trained on a
    training split of the SHORT readout signal and the fidelity scored on a HELD-OUT test
    split.  Because PSF matched-filtering separates the labelled bright/dark populations
    better than a square box (especially on the short, noisy readout), the methods get
    DISTINCT fidelities -- the whole point of computing all three.  ``{}`` if no labels."""
    from .fidelity import characterize_readout

    reference = _reference_signal_array(calibration, reference_groups, method="box")  # fixed labels
    if reference.size == 0 or reference.shape[1] < 1 or len(readout_by_group) != reference.shape[0]:
        return {}
    out: dict[str, dict] = {}
    try:
        methods = tuple(calibration.methods())
    except Exception:
        methods = ()
    for m in methods:
        short = np.vstack([np.asarray(calibration.signals(np.asarray(f, dtype=float), method=m),
                                      dtype=float).reshape(-1) for f in readout_by_group])
        try:
            report = characterize_readout(short, reference)
        except Exception:
            continue
        out[str(m)] = {"counts": short,
                       "thresholds": np.asarray(report.thresholds, dtype=float).reshape(-1),
                       "fidelity": np.asarray(report.site_fidelities, dtype=float),
                       "held_out": True}
    return out


def write_calibration_report(folder, *, calibration, readout_frames, template=None,
                             threshold_method: str = "otsu", timestamp: str = "",
                             reference_groups=None, readout_by_group=None) -> dict:
    """Write the per-site distribution grid + global distribution(+fidelity) + site-map
    PNGs (via the frontend) and a ``calibration.npz`` + ``calibration.json`` +
    ``summary.json`` (the numbers) into ``folder`` (created if needed).

    Returns a summary dict (paths + n_sites + mean fidelity).  The numbers are derived
    purely from the calibration contract + readout frames (identical on real hardware);
    the figures come from the registered viewer, so they use the live console's plot
    types rather than hand-rolled matplotlib.

    When ``reference_groups`` + ``readout_by_group`` are given (the Rb87 reference-bracket
    flow -- long frames that vote ground truth around each short readout), the per-method
    fidelity is the HELD-OUT classification fidelity against those labels, so box / per-site
    PSF / uniform PSF get DISTINCT numbers; otherwise it falls back to the affine-invariant
    self-consistent estimate (a folder run that kept no reference groups)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(calibration.centers, dtype=float)
    thresholds = np.asarray(calibration.thresholds, dtype=float).reshape(-1)
    counts = per_site_counts(calibration, readout_frames)
    # one per-site grid PER readout method the calibration carries (box / per-site PSF /
    # uniform PSF) -- the experimenter compares the three readout models side by side.
    by_method = {}
    if reference_groups and readout_by_group:
        by_method = _held_out_by_method(calibration, reference_groups, readout_by_group)
    held_out = bool(by_method)
    if not held_out:
        by_method = _by_method_report(calibration, readout_frames)
    # the top-level fidelity (global distribution + summary) follows the BOX panel, so it is
    # the held-out classification fidelity when reference labels exist, else the estimate.
    if held_out and "box" in by_method:
        fidelity = np.asarray(by_method["box"]["fidelity"], dtype=float)
    else:
        fidelity = (per_site_fidelity(counts, thresholds) if counts.size
                    else np.full(len(centers), np.nan))

    paths = _render_figures(
        folder, counts=counts, thresholds=thresholds, fidelity=fidelity, centers=centers,
        template=template, threshold_method=threshold_method, timestamp=timestamp,
        by_method=by_method)

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
