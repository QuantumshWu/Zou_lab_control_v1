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


# Readout-method key -> human label for the per-method grid titles (the calibration carries
# all three; the report shows one grid each so the experimenter compares them side by side).
_METHOD_LABELS = {"box": "box (square ROI)", "psf": "per-site PSF", "uniform_psf": "uniform PSF"}


def _save_plot(plot, path: Path) -> str:
    """Save a report figure through the plot's OWN ``save`` (``BaseLivePlot.save`` ->
    ``DataFigure.save`` / ``GridPlot.save``) -- the SAME save path the live "Save Fig"
    button uses, at the UNIFIED high save dpi (``savefig.dpi`` = 600), not a hand-set dpi.
    Returns the PNG path (a single-axes save also drops a matching ``.npz`` of the plotted
    data beside it -- the DataFigure contract)."""
    out = plot.save(str(path))
    if isinstance(out, dict):
        return str(out.get("figure", path))
    return str(out or path)


def _site_grid(per_method_counts, thresholds, fidelity, *, title):
    """One per-site distribution grid (one histogram per site, threshold line + fidelity)."""
    return SiteHistogramGrid(
        per_method_counts, thresholds=list(thresholds), site_fidelities=list(fidelity),
        labels=("Readout counts", "Shots"), title=title,
        fig=_agg_figure(), interactions=False).show(display=False)


def save_calibration_report(folder, *, counts, thresholds, fidelity, centers,
                            template=None, threshold_method: str = "otsu",
                            timestamp: str = "", by_method=None) -> dict:
    """Render the calibration report PNGs into ``folder`` via the frontend plot types and
    return ``{name: path}``.

    * ``site_distribution_<method>.png`` -- one per-site histogram grid PER readout method
      the calibration carries (box / per-site PSF / uniform PSF), each with that method's
      per-site signals + thresholds + held-out fidelity, on the reference
      :class:`SiteHistogramGrid`.  ``by_method`` is ``{key: {counts, thresholds, fidelity}}``;
      without it a single ``site_distribution_grid.png`` is drawn from ``counts``.
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
    if by_method:
        # one per-site grid per readout method (box / per-site PSF / uniform PSF).
        for key, m in by_method.items():
            mc = np.atleast_2d(np.asarray(m["counts"], dtype=float))
            if not mc.size:
                continue
            mthr = np.asarray(m["thresholds"], dtype=float).reshape(-1)
            mfid = np.asarray(m["fidelity"], dtype=float).reshape(-1)
            per_site = [mc[:, k][np.isfinite(mc[:, k])] for k in range(mc.shape[1])]
            label = _METHOD_LABELS.get(str(key), str(key))
            grid = _site_grid(per_site, mthr, mfid,
                              title=f"Per-site readout distribution -- {label} (line = threshold)")
            paths[f"site_distribution_{key}"] = _save_plot(grid, folder / f"site_distribution_{key}.png")
    elif counts.size:
        per_site = [counts[:, k][np.isfinite(counts[:, k])] for k in range(counts.shape[1])]
        grid = _site_grid(per_site, thresholds, fidelity,
                          title="Per-site readout distribution (line = threshold)")
        paths["site_distribution_grid"] = _save_plot(grid, folder / "site_distribution_grid.png")

    if counts.size:
        # The pooled view aggregates EVERY site.  Sites have DIFFERENT dark/bright means and
        # DIFFERENT thresholds, so pooling RAW counts smears the two populations into one wide
        # blob -- a single bimodal fit then reports a falsely-low fidelity even when every site
        # is cleanly separated (the per-site grids show the real separation).  Centre each site
        # on ITS OWN threshold first (counts - threshold): all dark blobs line up below 0, all
        # bright blobs above 0, and the cut is exactly 0, so the pooled fit + its fidelity match
        # what the eye (and the per-site grids) see.
        thr = np.asarray(thresholds, dtype=float).reshape(-1)
        if thr.size == counts.shape[1] and np.isfinite(thr).any():
            centred = counts - thr[None, :]
            flat = centred.reshape(-1)
            flat = flat[np.isfinite(flat)]
            pooled = HistogramFigure(
                flat, bins=80, thresholds=[0.0],
                labels=("Readout counts - threshold", "Shots x sites", "Population"),
                title="Pooled readout distribution (per-site threshold-centred)",
                fig=_agg_figure(), interactions=False).show(display=False)
        else:
            flat = counts.reshape(-1)
            flat = flat[np.isfinite(flat)]
            pooled = HistogramFigure(
                flat, bins=80, labels=("Readout counts", "Shots x sites", "Population"),
                title="Pooled readout distribution", fig=_agg_figure(),
                interactions=False).show(display=False)
        paths["global_distribution"] = _save_plot(pooled, folder / "global_distribution.png")

    if template is not None and centers is not None and len(centers):
        occupied = np.ones((len(centers), 1), dtype=float)   # every fitted site shown as found
        site_map = LiveSiteMap(
            centers[:, :2], occupied, image=np.asarray(template, dtype=float),
            roi_radius=site_ring_radius(centers),
            labels=("Camera x (px)", "Camera y (px)", "Counts"),
            title="Reference template + fitted sites",
            fig=_agg_figure(), interactions=False).show(display=False)
        paths["site_map"] = _save_plot(site_map, folder / "site_map.png")

    return paths


__all__ = ["save_calibration_report"]
