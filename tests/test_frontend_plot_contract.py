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


def test_plot_kind_table_classes_are_all_base_live_plot():
    """The ONE plot-kind table (``live.PLOT_KINDS`` -- the single source both ``plot()`` and the
    task_console PANEL_* lookups read) may only point at BaseLivePlot subclasses, so a kind added
    to the table cannot smuggle in a hand-rolled figure that loses the reusable layer.  (The
    richer no-drift / render-family checks live in ``tests/test_plot_kind_table.py``.)"""
    from Zou_lab_control.frontend.live import PLOT_KINDS

    bad = [pk.key for pk in PLOT_KINDS
           if not (inspect.isclass(pk.cls) and issubclass(pk.cls, BaseLivePlot))]
    assert bad == [], f"PLOT_KINDS entries whose cls is not a BaseLivePlot subclass: {bad}"


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
    psf = [rng.random((7, 7)) for _ in range(6)]            # per-site PSF kernels (image grid)
    return {
        "1d": zf.plot(x, y, kind="1d", display=False, update="once"),
        "2d": zf.plot(xy, z, kind="2d", display=False, update="once"),
        "monitor": zf.plot(x, y, kind="monitor", display=False, update="once"),
        "hist": zf.plot(vals, kind="hist", display=False, update="once"),
        "sites": zf.plot(centers, rng.random(9), kind="sites", display=False, update="once"),
        "grid": zf.site_histogram_grid(sites, occupied=occ, thresholds=[20.0] * 6,
                                       site_fidelities=[0.99] * 6, display=False),
        "psf_grid": zf.site_psf_grid(psf, display=False),
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


def test_every_public_plot_save_writes_png_AND_data_npz(tmp_path):
    """The DataFigure contract is universal: ``plot.save(path)`` ALWAYS writes BOTH a png
    AND a matching ``.npz`` of the plotted data (the confocal design).  A new plot type
    that overrides save to skip the .npz silently drops the experimenter's raw data --
    exactly the bug the multi-panel GridPlot had (2026-06-19, only the png landed for the
    per-site distribution grids).  This is the mechanical guard: every public plot kind
    saves both files."""
    plots = _build_public_plots()
    try:
        for kind, p in plots.items():
            out = p.save(str(tmp_path / kind))
            # save returns {"figure": <png>, "data": <npz>} per the DataFigure contract
            assert isinstance(out, dict), f"{kind}.save did not return a dict"
            assert "figure" in out and "data" in out, f"{kind}.save missing figure/data keys: {out}"
            png, npz = out["figure"], out["data"]
            assert str(png).endswith(".png") and str(npz).endswith(".npz"), (kind, png, npz)
            # both files exist on disk -- not just promised
            assert png.exists() and npz.exists(), f"{kind} did not write png+npz: {out}"
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


def test_fit_targets_is_the_one_fanout_seam_for_grid_and_flat():
    """The Edit-tab fit / clear / Apply-limits stack loops over ``fit_targets()`` so ONE code path works on
    a flat plot AND a grid: a plain :class:`DataFigure` is its OWN single target, and a grid's ``_GridData``
    yields its N per-cell DataFigures -- so a single ``do_fit`` fans out to EVERY subplot (the regression:
    ``do_fit`` used to build ``DataFigure(gridplot)`` and fit only cell-0's placeholder arrays).  Also pins
    the grid ``xlim``/``ylim`` fan-out (Apply-limits is global-per-cell by construction)."""
    rng = np.random.default_rng(5)
    sites = [np.concatenate([rng.normal(3, 1, 60), rng.normal(50, 3, 70)]) for _ in range(6)]
    g = zf.site_histogram_grid(sites, thresholds=[25.0] * 6, display=False)
    try:
        gd = g.to_data_figure()
        targets = list(gd.fit_targets())
        # grid multiplicity == the per-cell DataFigures (one per visible cell), each a real fit handle
        assert targets == list(gd.cells) and len(targets) == 6
        assert all(isinstance(t, DataFigure) for t in targets)
        # fan-out: fitting each target INDEPENDENTLY gives every cell a popt (all subplots fitted)
        for target in targets:
            _, popt = target.gaussian(is_display=False)
            assert popt is not None
        # a flat DataFigure is its OWN single target -- the base case do_fit shares with the grid
        assert list(targets[0].fit_targets()) == [targets[0]]
        # Apply-limits fans out over every cell (a grid Apply is global-per-cell)
        gd.xlim(0.0, 100.0)
        assert all(tuple(t._ax.get_xlim()) == (0.0, 100.0) for t in targets)
    finally:
        plt.close(g.fig)


def test_grid_cell_title_is_facet_aware_and_templated_from_one_source():
    """#5: a grid cell's title comes from the ONE :meth:`GridCell.cell_title` hook (no per-family
    ``f"s{k}"`` literal), with part 1 = the facet-aware identifier from the ONE :func:`facet_cell_labels`
    source and part 2 = a user ``title_template`` that can reference the fit ``{popt}``; the cell-title
    point size is a settable interface with a single-source default."""
    from Zou_lab_control.frontend.live import GridCell, HistogramCell, facet_cell_labels, grid

    # (1) facet_cell_labels is the ONE identifier source, facet-aware per group.
    assert facet_cell_labels("repeat", 3) == ["rep 0", "rep 1", "rep 2"]
    assert facet_cell_labels("dim", 3) == ["s0", "s1", "s2"]                     # per-site
    assert facet_cell_labels("points:0", 2, coords=[1.0, 2.5], param_names=["Bz"]) == ["Bz=1", "Bz=2.5"]
    assert facet_cell_labels(None, 2) == ["s0", "s1"]                            # recipe grid keeps the tag

    # (2) the ONE cell_title hook every family shares (base defines it; no per-family s{k}).
    assert "cell_title" in vars(GridCell) and "_tag_text" not in vars(HistogramCell)

    rng = np.random.default_rng(6)
    block = np.stack([np.concatenate([rng.normal(3, 1, 50), rng.normal(40, 3, 60)]) for _ in range(4)],
                     axis=0).reshape(4, 1, 110)
    g = grid(block, sub_plot_kind="hist", facet="repeat", points_shape=(1,), display=False)
    try:
        cell = g.cell_renderer
        assert [cell.cell_title(k) for k in range(4)] == ["rep 0", "rep 1", "rep 2", "rep 3"]  # facet-aware
        # (3) a user template rendered with the {popt} context (fit the cells first so popt exists).
        cell.consume_param("fit", "double")
        for k in range(4):
            cell._apply_fit_lines(g.site_axes[k], k, cell.cell_counts[k])
        cell.consume_param("title_template", "{id} n={k}")
        assert cell.cell_title(2) == "rep 2 n=2"
        assert cell._cell_popt(0)                                   # popt exposed for a {popt} template
        # (4) title_size is a settable interface; 0 -> the single-source default (>0), never a literal.
        assert cell.consume_param("title_size", 8.5) and cell.title_size_pt() == 8.5
        cell.consume_param("title_size", 0)
        assert cell.title_size_pt() > 0
    finally:
        plt.close(g.fig)

    # (5) the calibration hist grid keeps its s{k} + fidelity default title (part-1 fidelity suffix).
    cal = zf.site_histogram_grid([np.concatenate([rng.normal(3, 1, 60), rng.normal(50, 3, 70)]) for _ in range(3)],
                                 thresholds=[25.0] * 3, site_fidelities=[0.87, 0.85, np.nan], display=False)
    try:
        assert [cal.cell_renderer.cell_title(k) for k in range(3)] == ["s0  87%", "s1  85%", "s2"]
    finally:
        plt.close(cal.fig)


def test_double_click_focus_shares_the_thumbnail_display_contract():
    """#2: double-clicking a grid cell enlarges it to a standalone panel that reads with the SAME window
    / colour scale / title as the thumbnail, not a self-scaled one.  ``focus_data`` carries the cell's
    OWN display state -- the hist value x-window (thumbnail pooled range), the image clim (pooled scale),
    and the ONE ``cell_title`` -- so the enlarged view and the grid cell never diverge; a STANDALONE panel
    (no seed) keeps its natural auto-scale, so non-grid plots are unchanged."""
    from Zou_lab_control.frontend.live import grid, panel_plot

    # hist cell: focus value_xlim == the thumbnail's SHARED pooled x-window, title == cell_title.
    rng = np.random.default_rng(7)
    hblock = np.stack([np.concatenate([rng.normal(3, 1, 50), rng.normal(40, 3, 60)]) for _ in range(4)],
                      axis=0).reshape(4, 1, 110)
    gh = grid(hblock, sub_plot_kind="hist", facet="repeat", points_shape=(1,), display=False)
    try:
        cell = gh.cell_renderer
        fd = cell.focus_data(1, display_params={})
        assert tuple(fd["value_xlim"]) == (cell.x_lo, cell.x_hi) and fd["title"] == cell.cell_title(1)
        fp = panel_plot(fd["data_x"], kind="hist", size="2x2",
                        **{k: v for k, v in fd.items() if k != "data_x"})
        assert fp.ax.get_xlim() == (cell.x_lo, cell.x_hi)              # enlarged x-window == thumbnail
        plt.close(fp.fig)
        # a STANDALONE hist (no value_xlim) keeps the natural bin span -- non-grid unchanged.
        std = panel_plot(hblock[0].reshape(-1), kind="hist", size="2x2", bins=30)
        assert std.ax.get_xlim() != (cell.x_lo, cell.x_hi)
        plt.close(std.fig)
    finally:
        plt.close(gh.fig)

    # image cell: focus clim == the thumbnail's SHARED pooled colour scale, title == cell_title.
    iblock = np.stack([np.abs(rng.normal(0, 1, (7, 7))) * (k + 1) for k in range(4)],
                      axis=0).reshape(4, 1, 7, 7)
    gi = grid(iblock, sub_plot_kind="2d", facet="repeat", points_shape=(1,), display=False)
    try:
        cell = gi.cell_renderer
        fd = cell.focus_data(2, display_params={})
        assert tuple(fd["clim"]) == (cell.vmin, cell.vmax) and fd["title"] == cell.cell_title(2)
        fp = panel_plot(fd["data_x"], fd["data_y"], kind="2d", size="2x2",
                        **{k: v for k, v in fd.items() if k not in ("data_x", "data_y")})
        assert fp.image.get_clim() == (cell.vmin, cell.vmax)          # enlarged colour scale == thumbnail
        plt.close(fp.fig)
    finally:
        plt.close(gi.fig)


def test_grid_focus_zoom_enlarges_one_cell_and_returns():
    """Multi-panel zoom = focus one cell: a double-click enlarges the cell into its STANDALONE plot-kind
    figure (a real HistogramFigure swapped onto the grid's own canvas), and a double-click / Esc on the
    enlarged view returns to the grid.  A non-left double-click (scroll wheel / middle button) must NOT
    toggle focus."""
    import numpy as np

    from matplotlib.backend_bases import MouseEvent
    from Zou_lab_control.frontend.live import HistogramFigure

    rng = np.random.default_rng(3)
    sites = [np.concatenate([rng.normal(3, 1, 50), rng.normal(40, 3, 60)]) for _ in range(6)]
    g = zf.site_histogram_grid(sites, thresholds=[20.0] * 6, display=False)
    try:
        assert g._focused is None and g._focus_plotter is None
        cell_axes = list(g.site_axes)

        def _dbl(ax, button):
            e = MouseEvent("button_press_event", g.fig.canvas, *ax.transData.transform((20, 1)))
            e.dblclick, e.button, e.inaxes = True, button, ax
            g.fig.canvas.callbacks.process("button_press_event", e)

        # a NON-left double-click on a cell must NOT enter focus
        _dbl(cell_axes[2], 2)
        assert g._focused is None, "a middle/scroll double-click must not focus"

        # a LEFT double-click on cell 2 enlarges it into a standalone HistogramFigure
        _dbl(cell_axes[2], 1)
        assert g._focused == 2
        assert isinstance(g._focus_plotter, HistogramFigure), "focus is a real standalone plot kind"
        assert g._focus_plotter.interaction_handles(), "the enlarged cell has its own selectors"
        assert g._focus_plotter.ax.get_xlabel() and g._focus_plotter.ax.get_ylabel()

        # a middle double-click on the enlarged view must NOT exit focus
        _dbl(g._focus_plotter.ax, 2)
        assert g._focused == 2

        # a LEFT double-click on the enlarged view returns to the grid (cells + thumbnails back)
        _dbl(g._focus_plotter.ax, 1)
        assert g._focused is None and g._focus_plotter is None
        assert all(ax.get_visible() for ax in g.site_axes)
        assert getattr(g.fig, "_zlc_grid_tools", None), "the grid's per-cell selectors are re-attached"
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


def test_shipped_notebook_template_panel_sources_assign_value():
    """A shipped notebook template must actually RUN: a ``PanelConfig(source=...)`` is one
    line of Python that has to assign ``value`` (or be blank = pick later).  A bare signal
    name like ``source="frame"`` silently never sets ``value`` and the panel only ever shows
    'assign the panel data to a `value = ...`' -- exactly the kind of stale-doc footgun that
    breaks the notebook-first story.  Pin every template's panel source to the contract."""
    from pathlib import Path

    templates = Path(zf.__file__).parent / "content" / "notebook_templates"
    sources = []
    for md in templates.glob("*.cells.md"):
        text = md.read_text(encoding="utf-8")
        # a BARE `source=` kwarg only -- `trigger_source=` etc. are ordinary device params
        for m in re.finditer(r'(?<![A-Za-z0-9_])source\s*=\s*([\'"])(.*?)\1', text):
            sources.append((md.name, m.group(2)))
    assert sources, "expected at least one PanelConfig(source=...) in the shipped templates"
    bad = [(name, s) for name, s in sources if s.strip() and not s.strip().startswith("value")]
    assert bad == [], (
        "these shipped-template panel sources do not assign `value` (the panel would error "
        f"with 'assign the panel data to a value = ...'): {bad}")


def test_notebook_rolling_run_accepts_a_scalar_window():
    """Notebook-API parity: a ROLLING live trace is sized by its WINDOW, so a notebook
    user passes the history length as a scalar -- ``zf.run(300, source, kind="monitor")``
    -- exactly like ``hist`` takes a count, instead of pre-building an x array.  Pins that
    the rolling kinds accept a scalar window (they used to raise), while a FIXED kind
    (1d) still requires a real x array (the scalar shortcut is rolling/hist only)."""
    src = lambda: 0.5                                  # a per-shot scalar source
    # The rolling trace is ONE kind ("monitor"); the bare-trace variant is show_dist=False,
    # not a separate kind.  Both spellings (canonical + a rolling synonym) accept a scalar window.
    for kind, window in (("monitor", 300), ("rolling", 120), ("loading_rate", 64)):
        sess = zf.run(window, src, kind=kind, autostart=False, display=False)
        try:
            assert sess.data_y.shape == (window, 1)    # a NaN window of the asked length
            assert bool(np.all(np.isnan(sess.data_y)))  # fills in as shots arrive
        finally:
            plt.close(sess.plot.fig)
    # a fixed (append) kind has no window meaning -> a scalar is still rejected
    with pytest.raises((ValueError, TypeError)):
        zf.run(50, src, kind="1d", autostart=False, display=False)


# ---- pulse is a real plot kind: its RENDER lives in the plot layer (live.py), not the GUI app --------
# A plot kind's rendering belongs to the plot layer next to every other kind's render class -- so the
# pulse "state -> figure" render is in live.py, and pulse_gui / task_console / data_figure CONSUME it.
# The dependency must point that way (consumers import FROM live), never the reverse (the plot layer
# importing a renderer from the pulse_gui editor app).  These pins encode that so it cannot regress.
def test_pulse_render_lives_in_the_plot_layer():
    """The STRUCTURED-figure render entry points (pulse timeline + per-site grid) are DEFINED in
    ``live.py`` (the plot layer that owns every kind's render), not in the ``pulse_gui`` editor app -- so
    notebook replay, the editor preview and a seeded console panel all draw through the ONE builder.  The
    grid builder ``build_grid_figure`` mirrors the pulse builder ``build_pulse_preview_plot``."""
    from Zou_lab_control.frontend import live as live_mod
    for name in ("build_pulse_preview_plot", "annotate_pulse_variable_regions", "analog_bus_traces",
                 "build_grid_figure", "grid_recipe_from_cells", "kind_for_plotter"):
        obj = getattr(live_mod, name, None)
        assert obj is not None, f"live.py must define the render/kind entry point {name!r}"
        assert obj.__module__ == live_mod.__name__, \
            f"{name} must be DEFINED in live.py (the plot layer), got {obj.__module__}"


def test_consumers_do_not_import_pulse_render_from_the_gui_app():
    """``task_console`` and ``data_figure`` (and the plot layer itself) must NOT import any pulse RENDER
    symbol from ``pulse_gui`` -- the pulse render's single source is the plot layer.  (Importing a plain
    GUI utility like ``slot_label`` is fine; a pulse RENDERER is not.)"""
    import ast
    from pathlib import Path

    render_names = {"build_pulse_preview_plot", "analog_bus_traces", "annotate_pulse_variable_regions",
                    "_analog_bus_traces", "_annotate_variable_regions"}
    frontend = Path(live_mod.__file__).parent
    offenders = []
    for mod in ("task_console.py", "data_figure.py", "live.py"):
        tree = ast.parse((frontend / mod).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("pulse_gui"):
                for alias in node.names:
                    if alias.name in render_names:
                        offenders.append(f"{mod}: from ...pulse_gui import {alias.name}")
    assert offenders == [], (
        "a consumer imports a pulse RENDER symbol from the pulse_gui app -- the render's single source is "
        f"the plot layer (live.py); import it from there instead: {offenders}")
