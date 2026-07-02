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

from .live import HistogramFigure, LiveSiteMap, grid, site_ring_radius


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
    """One per-site distribution grid (one histogram per site, threshold line + fidelity) -- built through
    the UNIFIED grid factory with ``sub_plot_kind="hist"``, the SAME entry the per-site PSF kernel grid
    uses (``sub_plot_kind="2d"``): the report creates every grid the ONE way."""
    return grid(per_method_counts, sub_plot_kind="hist", thresholds=list(thresholds),
                site_fidelities=list(fidelity), labels=("Readout counts", "Shots"), title=title,
                fig=_agg_figure(), interactions=False, display=False)


def _psf_grid(psf_weights, *, title):
    """The calibration's PSF *shape* output: one ``imshow`` per site of its empirical matched-filter kernel
    (the real asymmetric, non-Gaussian spot) -- built through the SAME unified grid factory with
    ``sub_plot_kind="2d"`` (a :class:`GridPlot` of :class:`ImageCell`), NOT hand-rolled matplotlib
    (frontend rule 7), so it gets the same selectors / focus-zoom / save contract as every other report
    figure.  Mirrors the rb87 readout's ``02_psf_grid.png``."""
    return grid([np.asarray(w, dtype=float) for w in psf_weights], sub_plot_kind="2d",
                labels=("x (px)", "y (px)"), title=title, fig=_agg_figure(),
                interactions=False, display=False)


def save_calibration_report(folder, *, counts, thresholds, fidelity, centers,
                            template=None, threshold_method: str = "otsu",
                            timestamp: str = "", by_method=None,
                            psf_weights=None, psf_boxes=None) -> dict:
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

    if psf_weights is not None and len(np.asarray(psf_weights)):
        grid = _psf_grid(psf_weights,
                         title="Per-site PSF weights (empirical matched filter -- the real asymmetric spot)")
        paths["psf_grid"] = _save_plot(grid, folder / "psf_grid.png")

    return paths


def save_frame_image(path, frame, *, title: str | None = None) -> str:
    """Render ONE raw camera frame through the SAME "2D image" plot TYPE the live console uses for a
    camera 2d panel (:class:`live.Live2DDis`, ``kind="2d"``) -- REUSE, not a hand-rolled imshow -- and
    save just the PNG.  The frame (H x W counts) is raveled into the plot's ``(N, 2)`` coords +
    ``(N,)`` values exactly as :meth:`PanelCard._build_plot` does for a live 2d panel, so the saved
    picture is identical to the console's 2D camera image (same colormap + side distribution +
    colorbar + clim).  PNG only: the ``.npy`` beside it in the ``frames/`` folder IS the data (no
    redundant ``.npz``).  Drawn on an explicit Agg canvas (``fig=``), so the calibrate task's worker
    thread can render it without touching pyplot / Qt -- the same off-thread contract the report uses."""
    from pathlib import Path as _P

    from .live import Live2DDis
    from .style import DEFAULT_STYLE

    arr = np.asarray(frame, dtype=float)
    ny, nx = arr.shape
    # ravel the image into the 2d plot's scatter form (meshgrid pixel coords + counts); Live2DDis
    # grids it straight back to the H x W image -- the identical path the live 2d camera panel takes.
    xx, yy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    data_x = np.column_stack([xx.ravel(), yy.ravel()])
    plot = Live2DDis(
        data_x, arr.ravel(),
        labels=("Camera x (px)", "Camera y (px)", "Counts"),
        title=title or _P(path).stem,
        fig=_agg_figure(), interactions=False).show(display=False)
    out = _P(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.fig.savefig(str(out), dpi=DEFAULT_STYLE["savefig.dpi"])   # png only -- unified save dpi, .npy holds the data
    return str(out)


__all__ = ["save_calibration_report", "save_frame_image"]
