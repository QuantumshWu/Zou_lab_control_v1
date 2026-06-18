"""Render the readout-calibration REPORT figures through the FRONTEND plot interface.

The experiment layer (``neutral_atom``) computes the calibration's NUMERIC artifacts
(per-site counts, per-site fidelity, the ``.npz`` + ``summary.json`` bundle) and -- when
a viewer is registered -- routes the FIGURES here, so the distribution grid / global
distribution / site map are drawn by the SAME styled plot TYPES the live console uses
(:class:`SiteHistogramGrid`, :class:`HistogramFigure`, :class:`LiveSiteMap`), not
hand-rolled matplotlib.  ``neutral_atom`` never imports this module; it reaches it through
the viewer-registry seam (the frontend self-registers ``save_calibration_report`` on
import), so the package decoupling (na never imports frontend) is preserved::

    neutral_atom --(active_plotter().save_calibration_report)--> frontend plot types

Figures are drawn on EXPLICIT Agg canvases (never pyplot / ``new_figure``), so the
calibrate task's worker thread can render them without touching the GUI's pyplot or Qt
state -- the report renders identically headless (the contract test imports the frontend
to register this renderer) and from the running console.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .live import HistogramFigure, LiveSiteMap, SiteHistogramGrid, site_ring_radius


def _agg_figure() -> Figure:
    """A bare matplotlib :class:`Figure` with an Agg canvas -- a thread-safe drawing
    surface (no pyplot global state) the live plot types draw into when handed via
    ``fig=`` (their ``show(display=False)`` then renders without a GUI)."""
    fig = Figure()
    FigureCanvasAgg(fig)
    return fig


def _save(fig: Figure, path: Path, *, dpi: int = 200) -> str:
    fig.savefig(str(path), dpi=dpi, facecolor="white")
    return str(path)


def save_calibration_report(folder, *, counts, thresholds, fidelity, centers,
                            template=None, threshold_method: str = "otsu",
                            timestamp: str = "") -> dict:
    """Render the calibration report PNGs into ``folder`` via the frontend plot types and
    return ``{name: path}``.

    * ``site_distribution_grid.png`` -- one histogram per site with its calibrated
      threshold + held-out fidelity, on the reference :class:`SiteHistogramGrid` (the
      same aligned, never-overlapping grid the readout site-grid uses).
    * ``global_distribution.png`` -- the pooled readout-count distribution on a
      :class:`HistogramFigure` (its bimodal fit + fidelity readout come for free).
    * ``site_map.png`` -- the averaged reference template with the fitted site rings on a
      :class:`LiveSiteMap` (when a template image is supplied).

    Pure rendering: it derives nothing, takes the numbers the experiment layer computed,
    and only draws -- so a virtual and a real calibration produce the same report."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    fidelity = np.asarray(fidelity, dtype=float).reshape(-1)
    centers = None if centers is None else np.asarray(centers, dtype=float)

    paths: dict[str, str] = {}
    if counts.size:
        per_site = [counts[:, k][np.isfinite(counts[:, k])] for k in range(counts.shape[1])]
        grid = SiteHistogramGrid(
            per_site, thresholds=list(thresholds), site_fidelities=list(fidelity),
            labels=("Readout counts", "Shots"),
            title="Per-site readout distribution (line = threshold)",
            fig=_agg_figure(), interactions=False).show(display=False)
        paths["site_distribution_grid"] = _save(grid.fig, folder / "site_distribution_grid.png")

        flat = counts.reshape(-1)
        flat = flat[np.isfinite(flat)]
        median_thr = float(np.nanmedian(thresholds)) if thresholds.size else None
        pooled = HistogramFigure(
            flat, bins=80, thresholds=([median_thr] if median_thr is not None else None),
            labels=("Readout counts", "Shots x sites", "Population"),
            title="Pooled readout distribution", fig=_agg_figure(),
            interactions=False).show(display=False)
        paths["global_distribution"] = _save(pooled.fig, folder / "global_distribution.png")

    if template is not None and centers is not None and len(centers):
        occupied = np.ones((len(centers), 1), dtype=float)   # every fitted site shown as found
        site_map = LiveSiteMap(
            centers[:, :2], occupied, image=np.asarray(template, dtype=float),
            roi_radius=site_ring_radius(centers), labels=("Camera x (px)", "Camera y (px)"),
            title="Reference template + fitted sites",
            fig=_agg_figure(), interactions=False).show(display=False)
        paths["site_map"] = _save(site_map.fig, folder / "site_map.png")

    return paths


__all__ = ["save_calibration_report"]
