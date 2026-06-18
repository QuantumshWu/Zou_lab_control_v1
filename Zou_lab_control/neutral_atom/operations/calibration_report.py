"""Write a readout-calibration REPORT to a folder -- the Rb87-style artifacts.

After a calibration the experimenter wants the SAME things the Rb87 readout pipeline
saves: a per-site count DISTRIBUTION grid (one histogram per site, with its threshold),
a pooled GLOBAL distribution annotated with the single-shot FIDELITY, the averaged site
template image, and an ``.npz`` + ``summary.json`` bundle -- all written to one folder so
a run is reproducible and reviewable offline.

Analysis layer ONLY: it reads the readout CONTRACT (``calibration.signals`` /
``centers`` / ``thresholds``) + the readout frames, and draws with a headless Agg
canvas (no pyplot, so it is safe in the calibrate task's worker thread).  It imports no
camera backend and no frontend, so a VIRTUAL calibration writes byte-for-byte the SAME
report a real qCMOS run would (only the frames differ).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..core.analysis import estimate_threshold_fidelity, otsu_threshold


def _grid_shape(n_sites: int) -> tuple[int, int]:
    """A near-square (rows, cols) grid that holds ``n_sites`` cells (cols >= rows)."""
    n = max(1, int(n_sites))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


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


def _save(fig: Figure, path: Path) -> None:
    FigureCanvasAgg(fig).print_figure(str(path), dpi=200, facecolor="white")


def _distribution_grid_figure(counts, thresholds, fidelity, *, bins: int = 40) -> Figure:
    n_sites = counts.shape[1]
    rows, cols = _grid_shape(n_sites)
    fig = Figure(figsize=(min(2.0 * cols, 16), min(1.6 * rows, 14)))
    axes = fig.subplots(rows, cols, squeeze=False)
    lo, hi = np.nanpercentile(counts, [0.5, 99.5]) if counts.size else (0.0, 1.0)
    edges = np.linspace(lo, hi, bins + 1)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        if k >= n_sites:
            ax.axis("off")
            continue
        col = counts[:, k]
        ax.hist(col[np.isfinite(col)], bins=edges, color="#7E9CD8", alpha=0.9)
        ax.axvline(float(thresholds[k]), color="#D07850", lw=1.3)
        fid = fidelity[k]
        ax.set_title(f"{k}: F={fid * 100:.1f}%" if np.isfinite(fid) else f"{k}",
                     fontsize=6.5)
        ax.tick_params(labelsize=5, length=1.5)
    fig.suptitle("Per-site readout count distribution (orange = threshold)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _global_distribution_figure(counts, thresholds, fidelity) -> Figure:
    fig = Figure(figsize=(5.2, 3.6))
    ax = fig.subplots()
    flat = counts.reshape(-1)
    flat = flat[np.isfinite(flat)]
    thr = float(np.nanmedian(thresholds))
    ax.hist(flat, bins=80, color="#9AA7B2", alpha=0.9)
    ax.axvline(thr, color="#D07850", lw=1.8, label=f"median threshold = {thr:.0f}")
    mean_f = float(np.nanmean(fidelity)) if np.any(np.isfinite(fidelity)) else float("nan")
    worst = float(np.nanmin(fidelity)) if np.any(np.isfinite(fidelity)) else float("nan")
    ax.set_xlabel("Readout counts")
    ax.set_ylabel("Shots x sites")
    ax.set_title("Pooled readout distribution")
    ax.text(0.97, 0.95, f"mean fidelity {mean_f * 100:.2f}%\nworst site {worst * 100:.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def _template_figure(template, centers) -> Figure:
    fig = Figure(figsize=(4.8, 3.8))
    ax = fig.subplots()
    im = ax.imshow(np.asarray(template, dtype=float), cmap="inferno", origin="upper")
    if centers is not None and len(centers):
        c = np.asarray(centers, dtype=float)
        ax.scatter(c[:, 0], c[:, 1], s=60, facecolors="none", edgecolors="#7EA5A3", lw=0.8)
    ax.set_xlabel("Camera x (px)")
    ax.set_ylabel("Camera y (px)")
    ax.set_title("Averaged reference template + sites")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def write_calibration_report(folder, *, calibration, readout_frames, template=None,
                             threshold_method: str = "otsu", timestamp: str = "") -> dict:
    """Write the per-site distribution grid + global distribution(+fidelity) + template
    PNGs and a ``calibration.npz`` + ``summary.json`` into ``folder`` (created if needed).

    Returns a summary dict (paths + n_sites + mean fidelity).  Pure: no backend / frontend
    / sim-truth -- the artifacts are derived from the calibration contract + the readout
    frames, identical on real hardware."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(calibration.centers, dtype=float)
    thresholds = np.asarray(calibration.thresholds, dtype=float).reshape(-1)
    counts = per_site_counts(calibration, readout_frames)
    fidelity = (per_site_fidelity(counts, thresholds) if counts.size
                else np.full(len(centers), np.nan))

    paths: dict[str, str] = {}
    if counts.size:
        p = folder / "site_distribution_grid.png"
        _save(_distribution_grid_figure(counts, thresholds, fidelity), p)
        paths["site_distribution_grid"] = str(p)
        p = folder / "global_distribution.png"
        _save(_global_distribution_figure(counts, thresholds, fidelity), p)
        paths["global_distribution"] = str(p)
    if template is not None:
        p = folder / "site_map.png"
        _save(_template_figure(template, centers), p)
        paths["site_map"] = str(p)

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
