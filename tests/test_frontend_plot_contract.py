"""MECHANICAL guard for a core frontend design principle.

The frontend exists so every plot reuses ONE layer: selectors (zoom/pan, area,
cross, draggable lines) and the DataFigure fitting/post-processing stack.  A plot
type gets that layer by being a ``BaseLivePlot`` subclass and going through its
``show()`` lifecycle.  Hand-rolling a raw-matplotlib figure silently loses it --
that is exactly what happened to the multi-site histogram grid (2026-06).

This test encodes the principle as a contract so it cannot regress to prose:

1. structural -- no plot-shaped class in ``live.py`` may bypass ``BaseLivePlot``;
2. behavioural -- every public plot entry point produces a figure whose
   selectors are attached AND whose data exposes a working ``DataFigure``.
"""

from __future__ import annotations

import inspect
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import Zou_lab_control.frontend as zf
from Zou_lab_control.frontend import live as live_mod
from Zou_lab_control.frontend.live import BaseLivePlot, SiteHistogramGrid
from Zou_lab_control.frontend.data_figure import DataFigure


# A class whose name ends like a plot (Figure / Grid / Map / Dis / 1D / 2D /
# Plot) is a plot type and MUST be a BaseLivePlot.  This pattern matches the
# bare ``class SiteHistogramGrid:`` that shipped without selectors/data_figure.
_PLOT_NAME = re.compile(r"(Figure|Grid|Map|Dis|Plot|1D|2D)$")


def test_no_plot_class_bypasses_base_live_plot():
    offenders = []
    for name, obj in vars(live_mod).items():
        if not inspect.isclass(obj) or obj.__module__ != live_mod.__name__:
            continue
        if _PLOT_NAME.search(name) and not issubclass(obj, BaseLivePlot):
            offenders.append(name)
    assert offenders == [], (
        "these plot-shaped classes bypass BaseLivePlot, so they lose the shared "
        f"selector/DataFigure layer -- make them subclass BaseLivePlot: {offenders}"
    )


def _build_public_plots():
    """One instance of every public, data-easy plot entry point (display off)."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 50).reshape(-1, 1)
    y = np.sin(np.linspace(0, 6, 50)).reshape(-1, 1)
    xy = np.column_stack([np.tile(np.arange(7), 7), np.repeat(np.arange(7), 7)])
    z = rng.normal(size=49)
    vals = np.concatenate([rng.normal(3, 1, 150), rng.normal(40, 4, 160)])
    centers = np.column_stack([(np.arange(9) % 3) * 5.0, (np.arange(9) // 3) * 5.0])
    sites = [np.concatenate([rng.normal(3, 1, 50), rng.normal(40, 3, 60)]) for _ in range(6)]
    occ = [np.array([False] * 50 + [True] * 60) for _ in range(6)]
    return {
        "1d": zf.plot(x, y, kind="1d", display=False, update="once"),
        "2d": zf.plot(xy, z, kind="2d", display=False, update="once"),
        "monitor": zf.plot(x, y, kind="monitor", display=False, update="once"),
        "hist": zf.plot(vals, kind="hist", display=False, update="once"),
        "sites": zf.plot(centers, rng.random(9), kind="sites", display=False, update="once"),
        "grid": zf.site_histogram_grid(sites, occupied=occ, thresholds=[20.0] * 6,
                                       site_fidelities=[0.99] * 6, display=False),
    }


def test_every_public_plot_is_first_class():
    plots = _build_public_plots()
    try:
        for kind, p in plots.items():
            assert isinstance(p, BaseLivePlot), f"{kind} is not a BaseLivePlot"
            # selectors attached -> the plot is interactive
            assert len(p.interaction_handles()) >= 1, f"{kind} has no selectors attached"
            # DataFigure layer is reachable
            df = p.to_data_figure()
            assert df is not None, f"{kind} has no DataFigure"
    finally:
        for p in plots.values():
            plt.close(p.fig)


def test_site_grid_exposes_per_cell_selectors_and_fitting():
    """The exact regression: the site grid must get per-cell zoom + draggable
    threshold AND a per-cell DataFigure that can actually fit."""
    rng = np.random.default_rng(2)
    sites = [np.concatenate([rng.normal(3, 1, 60), rng.normal(50, 3, 70)]) for _ in range(8)]
    occ = [np.array([False] * 60 + [True] * 70) for _ in range(8)]
    g = zf.site_histogram_grid(sites, occupied=occ, thresholds=[25.0] * 8, display=False)
    try:
        assert isinstance(g, SiteHistogramGrid) and isinstance(g, BaseLivePlot)
        # the FULL selector bundle per visible cell (area + cross + zoom + drag),
        # same as a standalone plot -- not just zoom/drag
        for name in ("area", "cross", "zoom", "drag"):
            assert sum(getattr(b, name) is not None for b in g._cell_interactions) == 8, name
        # tools pinned to the FIGURE (the lifecycle fix: without this the per-cell
        # selectors were collected in a notebook -> "no selector" / unresponsive)
        assert getattr(g.fig, "_zlc_tools", None) is not None
        assert len(getattr(g.fig, "_zlc_grid_tools", [])) == 8
        # per-cell DataFigure exists and fits without raising
        data = g.to_data_figure()
        assert len(data.cells) == 8
        cell0 = data.cell(0)
        assert isinstance(cell0, DataFigure)
        _, popt = cell0.gaussian(is_display=False)
        assert popt is not None
        # dragging a threshold updates that site's stored cut (live reclassify)
        g._make_threshold_cb(3)(33.0)
        assert g.thresholds[3] == 33.0
    finally:
        plt.close(g.fig)


def test_grid_focus_zoom_enlarges_one_cell_and_returns():
    """Multi-panel zoom = focus one cell: focus(k) enlarges it to fill the figure
    (full selectors), unfocus()/double-click returns to the grid."""
    import numpy as np

    from matplotlib.backend_bases import MouseEvent

    rng = np.random.default_rng(3)
    sites = [np.concatenate([rng.normal(3, 1, 50), rng.normal(40, 3, 60)]) for _ in range(6)]
    g = zf.site_histogram_grid(sites, thresholds=[20.0] * 6, display=False)
    try:
        assert g.focus_ax is None and g._focused is None
        g.focus(2)                                              # enlarge cell 2
        assert g._focused == 2 and g.focus_ax is not None
        assert all(not ax.get_visible() for ax in g.site_axes)  # grid hidden
        assert g._focus_tools.zoom is not None                  # enlarged cell has its own selectors
        # grid area selectors are suspended while focused (no residual box on return)
        assert all(not b.area.selector.active for b in g._cell_interactions)

        # a NON-left double-click (scroll wheel / middle button) must NOT exit focus
        def _dbl(ax, button):
            e = MouseEvent("button_press_event", g.fig.canvas, *ax.transData.transform((20, 1)))
            e.dblclick, e.button, e.inaxes = True, button, ax
            g.fig.canvas.callbacks.process("button_press_event", e)
        _dbl(g.focus_ax, 2)
        assert g._focused == 2                                  # middle double-click did not exit
        _dbl(g.focus_ax, 1)
        assert g._focused is None and g.focus_ax is None        # left double-click returns to grid
        assert all(ax.get_visible() for ax in g.site_axes)
        # grid area selectors re-armed AND cleared of any residue
        assert all(b.area.selector.active and b.area.range == [None, None, None, None]
                   for b in g._cell_interactions)
    finally:
        plt.close(g.fig)


def test_grid_plot_is_reusable_for_other_cell_types():
    """The grid framework is general: GridPlot drives any GridCell.  Today only the
    HistogramCell exists; future Image2DCell/Line1DCell plug in the same way."""
    import numpy as np
    from Zou_lab_control.frontend.live import GridPlot, GridCell, HistogramCell

    assert issubclass(HistogramCell, GridCell)
    rng = np.random.default_rng(4)
    sites = [rng.normal(5, 1, 40) for _ in range(5)]
    g = GridPlot(HistogramCell(sites, labels=("sig", "shots")), labels=("sig", "shots")).show(display=False)
    try:
        assert isinstance(g, BaseLivePlot) and len(g.site_axes) == 5
        assert len(g.interaction_handles()) >= 5                # selectors attached
        assert g.to_data_figure().cell(0) is not None           # per-cell DataFigure
    finally:
        plt.close(g.fig)
