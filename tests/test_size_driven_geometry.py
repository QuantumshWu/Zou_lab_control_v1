"""Contract: ONE size-driven geometry source for EVERY panel kind (incl. pulse + grid).

These pin the invariants of the size-unification round:

1. a pulse / grid panel's DATA region scales with the ``size`` preset (2x2 vs 4x4 differ), exactly like
   the single-axes kinds -- the size preset carries the content density, not a bespoke content-inches;
2. a grid's TOTAL figure px (data box + OUTER margins) EQUALS a same-size single-axes panel's to the pixel
   (#5, ONE ``panel_margins_px`` source), and a FOCUSED cell IS a standalone ``panel_plot`` of the cell's
   KIND (dispatched generically off ``sub_plot_kind`` through ``PLOT_KIND_BY_KEY`` -- hist -> HistogramFigure,
   image -> Live2DDis), so its lim edit sticks and never bounces back to the thumbnail;
3. a Monitor-card pulse / grid plotter is display-only (``interactions is False`` -> no selectors);
4. ``optimal_pulse_size`` is the ONE default-size source (the preview default and the loaded-panel default
   both derive from it), and a pulse's default tracks its drawn-row / period counts;
5. the embedded TaskConsole reserves the tab drop-shadow's top bleed above its tab strip (the figure
   viewer's Monitor-tab cut-off regression).

Runs headless (``QT_QPA_PLATFORM=offscreen``).  Values are DERIVED from the frontend's own constants /
functions, never re-typed literals, so the tests cannot silently drift from the code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from Zou_lab_control.frontend.live import (  # noqa: E402
    PANEL_SIZES,
    HistogramFigure,
    SiteHistogramGrid,
    build_grid_figure,
    build_pulse_preview_plot,
    default_pulse_size,
    grid_recipe_from_cells,
    optimal_grid_size,
    optimal_pulse_size,
    panel_margins_px,
    panel_plot_spec,
    panel_size_cells,
    pulse_drawn_rows,
    pulse_plot_spec,
)
from Zou_lab_control.frontend.canvas import design_dpi  # noqa: E402
from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod, PulseTableState  # noqa: E402


def _busy_state() -> PulseTableState:
    ch = [f"ch{i}" for i in range(8)]
    periods = [PulsePeriod(duration=5 + p, unit="us", name=f"P{p}",
                           states=tuple((p + i) % 2 for i in range(len(ch)))) for p in range(6)]
    return PulseTableState(channels=ch, periods=periods, name="busy")


def _mostly_off_state() -> PulseTableState:
    """A pulse with a FEW active channels and MANY always-off channels, chosen so ``include_always_off``
    flips the drawn-row count across a size threshold: hiding the off rows defaults to the smallest preset,
    showing them defaults to the largest.  Lets a show-all toggle observably re-derive the optimal size."""
    active = [f"a{i}" for i in range(2)]
    off = [f"off{i}" for i in range(20)]
    ch = active + off
    periods = [PulsePeriod(duration=5 + p, unit="us", name=f"P{p}",
                           states=tuple((1 if (c in active and (p + ai) % 2 == 0) else 0)
                                        for ai, c in enumerate(ch))) for p in range(3)]
    return PulseTableState(channels=ch, periods=periods, name="mostly_off")


def _grid_recipe(n=6):
    rng = np.random.default_rng(0)
    per = [np.concatenate([rng.normal(200, 25, 40), rng.normal(1200, 70, 40)]) for _ in range(n)]
    g = SiteHistogramGrid(per, grid_shape=(2, 3), thresholds=[700.0] * n, bins=30).show(display=False)
    recipe = grid_recipe_from_cells(g)
    plt.close(g.fig)
    return recipe


# --------------------------------------------------------------------------- 1. data region scales
def test_pulse_data_region_scales_with_size():
    """A pulse figure's data box is ``panel_plot_spec(size).data_px`` -- it GROWS with the size preset,
    and 2x2 != 4x4 (the whole point: size rescales the data region, not just the padding)."""
    st = _busy_state()
    small, _c, _r = build_pulse_preview_plot(st, include_always_off=True, size="2x2")
    big, _c, _r = build_pulse_preview_plot(st, include_always_off=True, size="4x4")
    # the pulse plot's axes data box (in px) equals the size-preset data_px for pulse
    for plotter, size in ((small, "2x2"), (big, "4x4")):
        exp_w, exp_h = pulse_plot_spec(size).data_px
        box_w, box_h = plotter.fig._zlc_fixed_box_in
        dpi = design_dpi(plotter.fig)
        assert round(box_w * dpi) == exp_w and round(box_h * dpi) == exp_h, \
            f"pulse {size} data box {round(box_w*dpi)}x{round(box_h*dpi)} != preset {exp_w}x{exp_h}"
    assert pulse_plot_spec("2x2").data_px != pulse_plot_spec("4x4").data_px
    plt.close(small.fig); plt.close(big.fig)


@pytest.mark.parametrize("size", ["2x2", "4x4"])
def test_grid_data_region_equals_the_size_panel_data_px(size):
    """The grid's TOTAL data region == ``panel_plot_spec(size).data_px`` -- the SAME data box every OTHER
    panel kind uses at this size (part A: the grid geometry is single-sourced off panel_plot_spec, no longer
    a bespoke fixed per-cell box).  Asserted to the PIXEL at 2x2 AND 4x4 off ``fig._zlc_fixed_box_in`` (which
    create_axes_grid(data_px=...) records for the whole cell block), so 2x2 grid data px == 2x2 hist / 2d /
    1d data px, and 4x4 likewise -- the real detection the earlier bw>sw check glossed over."""
    recipe = _grid_recipe()
    g = build_grid_figure(recipe, size=size, display=False)
    dpi = design_dpi(g.fig)
    box_w, box_h = g.fig._zlc_fixed_box_in
    got = (round(box_w * dpi), round(box_h * dpi))
    assert got == panel_plot_spec(size).data_px, \
        f"grid {size} data region {got} != panel_plot_spec({size}).data_px {panel_plot_spec(size).data_px}"
    plt.close(g.fig)


def test_grid_cells_scale_with_size():
    """A grid's data region grows with the size preset (2x2 vs 4x4 give a strictly larger data box AND
    figure), so the size preset truly rescales the cells, not just the padding."""
    recipe = _grid_recipe()
    small = build_grid_figure(recipe, size="2x2", display=False)
    big = build_grid_figure(recipe, size="4x4", display=False)
    sw, sh = small.fig.get_size_inches()
    bw, bh = big.fig.get_size_inches()
    assert bw > sw and bh > sh, f"grid 4x4 ({bw:.2f}x{bh:.2f}) must exceed 2x2 ({sw:.2f}x{sh:.2f})"
    # and the DATA region (not just the outer figure) grows -- the point of size-driven geometry
    assert panel_plot_spec("4x4").data_px[0] > panel_plot_spec("2x2").data_px[0]
    plt.close(small.fig); plt.close(big.fig)


def test_optimal_grid_size_judges_each_axis_independently():
    """``optimal_grid_size`` is the ONE grid default-size source (part A): EACH axis independently gets the
    compact half-unit (2) up to ITS OWN boundary (rows: 4, columns: 5 -- a cell is wider than tall, so more
    columns fit before the double width is needed) and the double (4) beyond it.  Rows and columns are never
    coupled, so a tall 5x3 grid opens ``4x2``, a wide 3x6 opens ``2x4``, a 5x7 opens ``4x4`` (every {2,4}^2
    combination is a real preset, no nearest-preset snap).  A grid built with ``size=None`` (the factory
    default) adopts exactly this preset (single source, no drift)."""
    # up to the per-axis boundary on BOTH axes -> compact 2x2 (<=4 rows, <=5 columns)
    for r, c in [(1, 1), (2, 2), (2, 3), (4, 4), (4, 5), (3, 5)]:
        assert panel_size_cells(optimal_grid_size(r, c)) == (2, 2), f"{r}x{c} (within bounds) -> compact 2x2"
    # each axis independently: tall -> 4x2, wide -> 2x4, both big -> 4x4
    for r, c in [(5, 3), (7, 5)]:
        assert panel_size_cells(optimal_grid_size(r, c)) == (4, 2), f"{r}x{c} (tall) -> 4x2"
    for r, c in [(3, 6), (4, 7)]:
        assert panel_size_cells(optimal_grid_size(r, c)) == (2, 4), f"{r}x{c} (wide) -> 2x4"
    for r, c in [(5, 6), (6, 8)]:
        assert panel_size_cells(optimal_grid_size(r, c)) == (4, 4), f"{r}x{c} (both big) -> double 4x4"
    # every returned preset is a real PANEL_SIZES member
    for r in range(1, 9):
        for c in range(1, 9):
            assert optimal_grid_size(r, c) in PANEL_SIZES
    # a grid built with size=None adopts the shape-driven default (the factory reads the SAME source)
    recipe = _grid_recipe(n=16)                 # 4x4 shape -> the compact 2x2 preset (both axes <= 4)
    g = build_grid_figure(recipe, size=None, display=False)
    assert g._size == optimal_grid_size(g.nrows, g.ncols)
    plt.close(g.fig)


def test_grid_focus_is_a_full_plot_kind_not_a_thumbnail():
    """A grid's FOCUSED (enlarged) cell is a FULL standalone plot-kind figure (part B), NOT the simplified
    thumbnail: focusing a distribution cell builds a real :class:`HistogramFigure` (its OWN standard panel
    figure) -- so it carries a DRAGGABLE threshold, the two-Gaussian fit and full x/y axes -- and unfocus
    rebuilds the grid cleanly (the grid cells back)."""
    recipe = _grid_recipe()
    g = build_grid_figure(recipe, size="2x2", display=False)
    n_axes_grid = len(g.fig.axes)
    g.focus(0)
    fp = g._focus_plotter
    assert isinstance(fp, HistogramFigure), "a distribution cell focuses into a full HistogramFigure"
    # the full-plot-kind machinery is present (a thumbnail has none of it)
    assert fp.threshold_draggers, "the focused cell has a DRAGGABLE threshold (not a static line)"
    assert fp.bimodal_popt is not None, "the focused cell ran the two-Gaussian fit"
    assert fp.ax.get_xlabel() and fp.ax.get_ylabel(), "the focused cell has full x/y axis labels"
    assert fp.interaction_handles(), "the focused cell attaches its own selectors (reusable layer)"
    g.unfocus()
    assert g._focused is None and g._focus_plotter is None, "unfocus returns to the grid"
    assert len(g.fig.axes) == n_axes_grid, "unfocus rebuilds the grid (its cell axes back)"
    assert all(ax.get_visible() for ax in g.site_axes), "the grid cells are visible again"
    plt.close(g.fig)


def test_grid_focus_dispatches_by_cell_kind_generically():
    """Focus dispatch is GENERIC through the ONE ``PLOT_KIND_BY_KEY`` table off each cell's ``sub_plot_kind``
    (never a hard-coded class): a HIST grid focuses into a :class:`HistogramFigure` AND an IMAGE grid focuses
    into a :class:`Live2DDis` -- proving the SAME builder resolves the right standalone plot kind per cell."""
    from Zou_lab_control.frontend.live import (
        GridPlot, HistogramCell, ImageCell, Live2DDis, PLOT_KIND_BY_KEY,
    )
    rng = np.random.default_rng(3)
    # a hist grid -> HistogramFigure
    per = [np.concatenate([rng.normal(200, 25, 40), rng.normal(1200, 70, 40)]) for _ in range(4)]
    gh = GridPlot(HistogramCell(per, thresholds=[700.0] * 4), grid_shape=(2, 2), size="2x2").show(display=False)
    assert gh.cell_renderer.sub_plot_kind == "hist" and PLOT_KIND_BY_KEY["hist"].cls is HistogramFigure
    gh.focus(0)
    assert isinstance(gh._focus_plotter, HistogramFigure), "hist cell -> standalone HistogramFigure"
    gh.unfocus(); plt.close(gh.fig)
    # an image grid -> Live2DDis (the SAME focus() path, only the cell's sub_plot_kind differs)
    imgs = [rng.random((9, 9)) for _ in range(4)]
    gi = GridPlot(ImageCell(imgs), grid_shape=(2, 2), size="2x2").show(display=False)
    assert gi.cell_renderer.sub_plot_kind == "2d" and PLOT_KIND_BY_KEY["2d"].cls is Live2DDis
    gi.focus(1)
    assert isinstance(gi._focus_plotter, Live2DDis), "image cell -> standalone Live2DDis"
    gi.unfocus(); plt.close(gi.fig)


# --------------------------------------------------------------------------- 2. focus == same-size panel
@pytest.mark.parametrize("size", ["2x2", "4x4"])
def test_grid_total_px_equals_same_size_single_axes_panel(size):
    """#5: a grid's TOTAL figure px (data box + OUTER margins) EQUALS a same-size single-axes panel's to the
    pixel -- not just the data_px.  The grid's outer margins are the ONE ``panel_margins_px`` source (no
    bespoke grid-margin tuple), so the grid thumbnail's padding matches every other kind's card."""
    from Zou_lab_control.frontend.live import panel_plot

    def _total_px(fig):
        dpi = design_dpi(fig)
        w, h = fig.get_size_inches()
        return round(w * dpi), round(h * dpi)

    recipe = _grid_recipe()
    g = build_grid_figure(recipe, size=size, display=False)
    panel = panel_plot(np.linspace(0, 1, 50), kind="hist", size=size)   # a standalone same-size hist panel
    assert _total_px(g.fig) == _total_px(panel.fig), \
        f"grid {size} total px {_total_px(g.fig)} != same-size panel {_total_px(panel.fig)}"
    plt.close(g.fig); plt.close(panel.fig)


def test_grid_focus_cell_is_a_standalone_same_size_panel(size="2x2"):
    """A focused cell is EXACTLY a standalone ``panel_plot`` of its kind at the stock ``2x2`` size, so its
    absolute px margins EQUAL ``panel_margins_px`` -- the SAME as a real single-axes panel -- WITHOUT any
    bespoke focus-bounds code (it just IS a panel_plot).  Asserted on the focus plotter's own axes."""
    recipe = _grid_recipe()
    g = build_grid_figure(recipe, size="4x4", display=False)   # grid at 4x4; focus opens at the stock 2x2
    g.focus(0)
    fp = g._focus_plotter
    dpi = design_dpi(fp.fig)
    fw = float(fp.fig.get_size_inches()[0]) * dpi
    fh = float(fp.fig.get_size_inches()[1]) * dpi
    pos = fp.ax.get_position()
    got = (round(pos.x0 * fw), round(fw - (pos.x0 + pos.width) * fw),
           round(pos.y0 * fh), round(fh - (pos.y0 + pos.height) * fh))
    assert got == panel_margins_px("default"), \
        f"focused cell margins {got} must equal a same-size panel's {panel_margins_px('default')}"
    g.unfocus()
    plt.close(g.fig)


def test_focus_lim_sticks_and_does_not_bounce_back():
    """The HARD acceptance point: after focusing, changing the standalone view's fixed lim ACTUALLY moves the
    axis (a real numeric change), and re-drawing / re-applying does NOT revert to the grid thumbnail (still
    focused, still the same standalone plotter).  A histogram's relim pins its COUNT (y) axis."""
    recipe = _grid_recipe()
    g = build_grid_figure(recipe, size="2x2", display=False)
    g.focus(0)
    fp = g._focus_plotter
    fp.relim_mode = "fixed"
    fp.fixed_lo, fp.fixed_hi = 0.0, 777.0
    fp.apply_relim_now()
    assert fp.ax.get_ylim() == (0.0, 777.0), f"fixed lim must move the axis, got {fp.ax.get_ylim()}"
    # a redraw / re-apply must NOT bounce back to the grid thumbnail
    fp.draw()
    fp.apply_relim_now()
    assert g._focused == 0 and g._focus_plotter is fp, "the focus view must NOT revert on redraw"
    assert fp.ax.get_ylim() == (0.0, 777.0), "the fixed lim must persist (no bounce-back)"
    g.unfocus()
    plt.close(g.fig)


def test_console_grid_focus_swaps_to_real_panel_lim_sticks_no_bounce():
    """The CONSOLE path: a double-click on a grid panel's canvas SWAPS ``PanelCard.plotter`` to the clicked
    cell's STANDALONE HistogramFigure; a fixed-lim edit through the ORDINARY console path moves its axis; a
    live tick does NOT bounce back to the grid; a second double-click returns to the grid."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=10, col=10, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _sig: recipe)
    try:
        card._build_plot(0.0, {})
        grid = card.plotter
        assert isinstance(grid, SiteHistogramGrid) and card._grid_focus is None

        class _Ev:
            dblclick = True
            button = 1
            def __init__(self, ax):
                self.inaxes = ax

        card._on_grid_canvas_click(_Ev(grid.site_axes[2]))
        assert isinstance(card.plotter, HistogramFigure), "double-click swaps to a standalone HistogramFigure"
        assert card._grid_focus is not None and card._grid_focus.k == 2
        fp = card.plotter

        # fixed lim through the ordinary console path (config.params + _apply_lim_to_plotter) moves the axis
        card.config.params.update({"relim": "fixed", "fixed_lo": 0.0, "fixed_hi": 888.0})
        card._apply_lim_to_plotter()
        assert fp.ax.get_ylim() == (0.0, 888.0), f"console fixed lim must move the axis, got {fp.ax.get_ylim()}"

        # a live tick is gated OFF while focused -> no bounce-back to the grid
        card._render(0.0, {})
        assert card._grid_focus is not None and card.plotter is fp, "the live tick must not revert the focus view"
        assert fp.ax.get_ylim() == (0.0, 888.0), "the fixed lim persists across a tick (no bounce-back)"

        # a second double-click returns to the grid
        card._on_grid_canvas_click(_Ev(None))
        assert card._grid_focus is None and isinstance(card.plotter, SiteHistogramGrid)
    finally:
        card.shutdown()
        plt.close("all")


def _psf_recipe(n=6):
    from Zou_lab_control.frontend.live import GridPlot, ImageCell, grid_recipe_from_cells as _rec
    rng = np.random.default_rng(1)
    g = GridPlot(ImageCell([np.abs(rng.normal(size=(7, 7))) for _ in range(n)]),
                 grid_shape=(2, 3)).show(display=False)
    recipe = _rec(g)
    plt.close(g.fig)
    return recipe


class _DblClick:
    dblclick = True
    button = 1
    def __init__(self, ax):
        self.inaxes = ax


def test_grid_panel_params_follow_sub_plot_kind():
    """#4: a grid panel's Setting / Edit param UI is driven by the grid's per-site ``sub_plot_kind`` -- a hist
    grid shows the hist params (bins/fit/ylog), a 2d (kernel) grid shows the 2d colormap -- resolved by
    ``PanelCard._param_kind``, NOT a hard-coded ``"grid"`` PANEL_PARAMS entry (which no longer exists)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS, PanelCard, PanelConfig
    ensure_qt_app()
    assert "grid" not in PANEL_PARAMS, "a grid has NO fixed params -- they come from its sub_plot_kind"
    for recipe, kind, must_have in ((_grid_recipe(), "hist", "bins"), (_psf_recipe(), "2d", "cmap")):
        cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
        card = PanelCard(cfg, grid_recipe_provider=lambda _s, _r=recipe: _r)
        try:
            card._build_plot(0.0, {})
            assert card.plotter.sub_plot_kind == kind
            assert card._param_kind() == kind, f"a {kind} grid drives its {kind} param UI"
            keys = {d.key for d in PANEL_PARAMS.get(card._param_kind(), ())}
            assert must_have in keys, f"a {kind} grid exposes the {must_have} param, got {keys}"
        finally:
            card.shutdown()
            plt.close("all")


def test_grid_param_edit_while_focused_stays_focused():
    """#4: adjusting a param while a grid cell is ENLARGED applies to the focus view + persists onto the
    parked grid, and STAYS zoomed -- it never bounces back to the main grid (the reported revert bug)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        grid = card.plotter
        card._on_grid_canvas_click(_DblClick(grid.site_axes[1]))
        assert card._grid_focus is not None, "focused into a cell"
        focus = card.plotter
        card._set_param("bins", 48)                    # a Setting edit while zoomed
        assert card._grid_focus is not None, "the param edit MUST NOT revert to the grid"
        assert card.config.params.get("bins") == 48
        assert grid.cell_renderer.bins_arg == 48, "the parked grid re-binned so it persists on unfocus"
        card._set_param("ylog", True)                  # another edit -- still focused
        assert card._grid_focus is not None and card.plotter is focus
    finally:
        card.shutdown()
        plt.close("all")


def test_relim_fixed_freezes_the_current_view():
    """Flipping relim to ``fixed`` FREEZES the view: fixed_lo/hi seed from the plotter's live limits
    (``current_lims``), NEVER the blind 0..1 default -- pinning a counts axis / camera clim to 0..1
    empties the plot (every bar/pixel outside the range), which read as "the enlarged plot died".
    The view must be pixel-identical across the flip, and a focused grid stays focused."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    for recipe, kind in ((_grid_recipe(), "hist"), (_psf_recipe(), "2d")):
        cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
        card = PanelCard(cfg, grid_recipe_provider=lambda _s, _r=recipe: _r)
        try:
            card._build_plot(0.0, {})
            card._on_grid_canvas_click(_DblClick(card.plotter.site_axes[1]))
            focus = card.plotter
            before = focus.current_lims()
            assert before != (0.0, 1.0), f"{kind}: a real view never sits exactly at the 0..1 default"
            card._set_param("relim", "fixed")
            assert card._grid_focus is not None and card.plotter is focus, \
                f"{kind}: the relim flip MUST stay zoomed on the same focus plotter"
            seeded = (card.config.params.get("fixed_lo"), card.config.params.get("fixed_hi"))
            assert seeded == before, f"{kind}: fixed seeds the CURRENT view {before}, got {seeded}"
            assert focus.current_lims() == before, f"{kind}: the view must not move on the flip"
        finally:
            card.shutdown()
            plt.close("all")


def test_relim_family_applies_in_place_at_the_plotter_layer():
    """The relim family (relim / fixed_lo / fixed_hi) is handled by ``BaseLivePlot.apply_param`` for
    EVERY kind -- the ONE in-place path all surfaces (Setting, Edit tab, a grid's focused cell) route
    through, so no surface ever needs a teardown/rebuild for a lim edit.  Flipping INTO fixed seeds
    lo/hi from the CURRENT view (fixed = freeze what you see, never a blind 0..1)."""
    from Zou_lab_control.frontend.live import panel_plot
    rng = np.random.default_rng(3)
    plotters = [
        panel_plot(np.concatenate([rng.normal(200, 20, 50), rng.normal(900, 50, 50)]),
                   kind="hist", size="2x2"),
        panel_plot(np.column_stack([np.repeat(np.arange(6.0), 6), np.tile(np.arange(6.0), 6)]),
                   rng.uniform(50, 400, 36), kind="2d", size="2x2"),
    ]
    for p in plotters:
        before = p.current_lims()
        assert p.apply_param("relim", "fixed"), f"{type(p).__name__} handles relim in place"
        assert (p.fixed_lo, p.fixed_hi) == before, \
            f"{type(p).__name__}: fixed seeds from the current view {before}"
        assert p.current_lims() == before, f"{type(p).__name__}: the view must not move on the flip"
        assert p.apply_param("fixed_hi", before[1] * 2), "fixed_hi lands in place too"
        assert p.current_lims() == (before[0], before[1] * 2)
        plt.close(p.fig)


def test_pulse_timeline_ignores_the_relim_family():
    """A pulse timeline's y axis IS the channel-row layout, not a measured quantity: the console
    pushes the relim family onto EVERY plotter after a rebuild (_apply_display_params), and that
    push must leave the timeline untouched -- the base relim against the dummy zero data_y would
    squash every channel row to (-0.1, 0.1) (the adversarial-review critical)."""
    from Zou_lab_control.frontend.live import build_pulse_preview_plot
    st = _busy_state()
    plotter, _c, _r = build_pulse_preview_plot(st, include_always_off=True, size="2x2")
    before = plotter.ax.get_ylim()
    plotter.apply_param("relim", "tight")
    plotter.apply_param("fixed_lo", 0.0)
    plotter.apply_param("fixed_hi", 1.0)
    assert plotter.ax.get_ylim() == before, "the relim family is a no-op on a pulse timeline"
    plt.close(plotter.fig)


def test_unfocused_grid_fixed_seeds_from_the_cells_auto_range():
    """Flipping relim to ``fixed`` on a grid card WITHOUT an enlarged cell seeds lo/hi from the
    cells' SHARED auto range (GridPlot.current_lims -> GridCell.auto_lims) -- never the base-class
    0..1 pair a grid's dormant ylim_min/max would give (the adversarial-review major: every
    thumbnail pinned to 0..1 = clipped to garbage)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        auto = card.plotter.cell_renderer.auto_lims()
        assert auto != (0.0, 1.0), "real data never sits exactly at the 0..1 default"
        card._set_param("relim", "fixed")
        seeded = (card.config.params["fixed_lo"], card.config.params["fixed_hi"])
        assert seeded == auto, f"fixed seeds the cells' shared auto range {auto}, got {seeded}"
        assert card.plotter.site_axes[0].get_ylim() == auto, \
            "and the thumbnails freeze AT that range (no visual jump, nothing clipped)"
    finally:
        card.shutdown()
        plt.close("all")


def test_zoomed_fixed_flip_reaches_thumbnails_without_typing_lims():
    """Flipping to ``fixed`` while a cell is ENLARGED and returning WITHOUT typing lo/hi: the seeded
    freeze (the enlarged view's own lims) must already live on the parked grid's cell state, so the
    thumbnails come back pinned to the frozen view -- never the 0..1 default (the matrix-agent
    finding: relim alone was forwarded, the just-seeded lo/hi were not)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        grid = card.plotter
        card._on_grid_canvas_click(_DblClick(grid.site_axes[1]))
        frozen = card.plotter.current_lims()               # the enlarged view the user is looking at
        card._set_param("relim", "fixed")                  # flip -- and type NOTHING
        card._unfocus_grid_cell()
        assert grid.site_axes[0].get_ylim() == frozen, \
            f"thumbnails must return pinned to the frozen view {frozen}, got {grid.site_axes[0].get_ylim()}"
    finally:
        card.shutdown()
        plt.close("all")


def test_relim_replay_with_unchanged_values_is_not_thumb_dirty():
    """The console re-pushes the relim family after EVERY grid rebuild (_apply_display_params): an
    UNCHANGED value must report not-dirty from store_display_param, or every rebuild pays an extra
    N-cell thumbnail repaint (the adversarial-review perf finding)."""
    from Zou_lab_control.frontend.live import build_grid_figure
    g = build_grid_figure(_grid_recipe(), interactions=False, size="2x2", display=False)
    g.apply_param("relim", "fixed")
    for key, value in (("relim", "fixed"),
                       ("fixed_lo", g.cell_renderer.fixed_lo),
                       ("fixed_hi", g.cell_renderer.fixed_hi)):
        assert g.store_display_param(key, value) is False, f"unchanged {key} must not dirty the thumbnails"
    plt.close(g.fig)


def test_fixed_lims_reach_the_parked_grid_thumbnails():
    """A lim edit while a dashboard grid cell is ENLARGED lands on the enlarged view AND is stored on
    the PARKED grid (the same store -> cell-state -> redraw mechanism cmap/bins use), so returning to
    the grid shows every thumbnail pinned to the operator's lo/hi -- never 'only the zoom changed'."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        grid = card.plotter
        card._on_grid_canvas_click(_DblClick(grid.site_axes[1]))
        card._set_param("relim", "fixed")                  # the Setting relim combo while zoomed
        card.apply_fixed_lims(0.0, 120.0)                  # the Setting / Edit lo-hi inputs (ONE path)
        assert card._grid_focus is not None, "still zoomed"
        assert card.plotter.current_lims() == (0.0, 120.0)
        card._unfocus_grid_cell()
        assert grid.site_axes[0].get_ylim() == (0.0, 120.0), \
            "back on the grid, the thumbnails carry the pinned count axis"
    finally:
        card.shutdown()
        plt.close("all")


def test_dashboard_focus_view_is_display_only():
    """The Monitor card is DISPLAY-ONLY (no selectors -- every _build_plot branch passes
    interactions=False): the ENLARGED grid cell on the dashboard follows the SAME rule.  Interactive
    zoom / region-select lives in the Edit tab's snapshot only."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        card._on_grid_canvas_click(_DblClick(card.plotter.site_axes[1]))
        assert card._grid_focus is not None
        assert list(card.plotter.interaction_handles()) == [], \
            "the dashboard's enlarged cell carries NO selectors (Monitor cards are display-only)"
    finally:
        card.shutdown()
        plt.close("all")


def test_focused_cell_centres_with_equal_stretch():
    """The enlarged (stock 2x2) cell sits at the CENTRE of the card's plot region: a TOP stretch with the
    SAME factor (1) as the holder's trailing one splits the spare height evenly (a factor-0 spacer wins
    nothing and pins the view to the top).  Unfocus removes the top stretch (the grid re-tops)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="4x4", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        card._on_grid_canvas_click(_DblClick(card.plotter.site_axes[1]))
        holder = card.canvas_holder
        assert holder.itemAt(0).spacerItem() is not None, "focus inserts a TOP stretch"
        assert holder.stretch(0) == holder.stretch(holder.count() - 1) == 1, \
            "top and trailing stretch factors MUST match (1) -- equal split = vertical centring"
        card._unfocus_grid_cell()
        assert holder.itemAt(0).spacerItem() is None, "unfocus removes the top stretch"
    finally:
        card.shutdown()
        plt.close("all")


def test_grid_panel_not_rebuilt_every_tick():
    """#3 perf: a grid panel is a SNAPSHOT -- a fresh hub tick (no rebuild flag) must NOT re-render all N
    cells (the "grid drawing is very slow" cause); the plotter object is unchanged across ticks."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        first = card.plotter
        assert first is not None
        card._render(0.0, {})                          # a plain tick, no _force_rebuild
        card._render(0.0, {})
        assert card.plotter is first, "a grid is not rebuilt on an ordinary tick (no per-cell re-render)"
    finally:
        card.shutdown()
        plt.close("all")


def test_grid_rebuild_while_focused_refocuses_the_same_cell():
    """A rebuild-class edit made while a grid cell is ENLARGED (a size change tears the plot down) must
    land back on the SAME enlarged cell after the rebuild -- never silently bounce to the main grid view
    (#no-focus-bounce).  An EXPLICIT unfocus (double-click) stays unfocused across later rebuilds."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    recipe = _grid_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        card._on_grid_canvas_click(_DblClick(card.plotter.site_axes[3]))
        assert card._grid_focus is not None and card._grid_focus.k == 3
        card._on_size("4x4")                    # teardown records the focused index
        card._build_plot(0.0, {})               # the next tick's rebuild
        assert card._grid_focus is not None and card._grid_focus.k == 3, \
            "a size change while zoomed re-focuses the SAME cell (no bounce to the main grid)"
        assert isinstance(card.plotter, HistogramFigure), "the enlarged view is back (a real hist panel)"
        # an explicit unfocus must NOT be resurrected by a later rebuild
        card._on_grid_canvas_click(_DblClick(None))
        assert card._grid_focus is None
        card._build_plot(0.0, {})
        assert card._grid_focus is None, "an explicit unfocus stays unfocused across rebuilds"
    finally:
        card.shutdown()
        plt.close("all")


def test_focused_2d_grid_cmap_applies_in_place_no_rebuild():
    """A cmap edit on a FOCUSED 2d (kernel) grid cell applies IN PLACE on the enlarged Live2DDis (the
    colorbar hangs off the same artist) -- the focus plotter object is UNCHANGED, so the view never
    bounces through a rebuild ('jumps back to the main view then recolours', the reported #4)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    from Zou_lab_control.frontend.live import Live2DDis
    ensure_qt_app()
    recipe = _psf_recipe()
    cfg = PanelConfig(kind="grid", row=0, col=0, size="2x2", inputs=["fig_value"], source="value = signal")
    card = PanelCard(cfg, grid_recipe_provider=lambda _s: recipe)
    try:
        card._build_plot(0.0, {})
        card._on_grid_canvas_click(_DblClick(card.plotter.site_axes[0]))
        focus = card.plotter
        assert isinstance(focus, Live2DDis)
        assert focus.apply_param("cmap", "viridis") is True, "Live2DDis handles cmap in place"
        card._set_param("cmap", "plasma")
        assert card._grid_focus is not None and card.plotter is focus, \
            "a cmap edit while zoomed stays on the SAME enlarged plotter (no refocus rebuild)"
        assert focus.image.get_cmap().name == "plasma"
        # the parked grid recorded the pick (recipe/params single source) for the unfocus redraw
        assert card._grid_focus.grid._display_params.get("cmap") == "plasma"
    finally:
        card.shutdown()
        plt.close("all")


def test_hist_grid_cells_draw_collections_not_per_bin_patches():
    """#perf contract: a hist grid cell's bars are PolyCollections (the standalone hist's bar geometry via
    _update_verts, one artist per population) -- NEVER ax.hist's per-bin Rectangle patches, which made an
    N-cell grid carry ~N x bins artists and draw far slower than a same-size single plot."""
    per = [np.random.default_rng(0).normal(0, 1, 80) for _ in range(6)]
    # interactions=False = the console Monitor card's own path (selectors add their OWN helper
    # rectangle/cross artists, which are not bin bars -- exclude them from the count).
    g = SiteHistogramGrid(per, grid_shape=(2, 3), thresholds=[0.0] * 6, bins=30,
                          interactions=False).show(display=False)
    try:
        for ax in g.site_axes:
            assert len(ax.patches) == 0, "no per-bin Rectangle patches in a grid cell"
            assert len(ax.collections) >= 1, "the bars are PolyCollection(s)"
    finally:
        plt.close(g.fig)


def test_no_external_ax_symbol_anywhere():
    """The old ``external_ax`` focus HACK is GONE (a focus is now a standalone panel_plot, not an axes
    overlay): the token must not appear in ANY frontend source or test -- a dead-symbol guard."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "Zou_lab_control" / "frontend"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"external_ax", text):
            offenders.append(str(path))
    assert not offenders, f"external_ax must be fully removed, still referenced in: {offenders}"


# --------------------------------------------------------------------------- 3. Monitor = no selectors
def test_monitor_pulse_and_grid_have_no_interactions():
    """A Monitor-card pulse / grid plotter built with ``interactions=False`` attaches NO selectors -- its
    ``interactions`` flag is False, its ``interaction_handles()`` is empty, and the figure carries no
    ``_zlc_tools`` (the read-only-card contract every kind honours)."""
    st = _busy_state()
    pulse, _c, _r = build_pulse_preview_plot(st, include_always_off=True, interactions=False)
    grid = build_grid_figure(_grid_recipe(), interactions=False, display=False)
    for plotter, name in ((pulse, "pulse"), (grid, "grid")):
        assert plotter.interactions is False, f"{name} Monitor card must have interactions=False"
        assert plotter.interaction_handles() == [], f"{name} Monitor card must attach no selectors"
        assert getattr(plotter.fig, "_zlc_tools", None) is None, f"{name} Monitor fig must carry no tools"
        plt.close(plotter.fig)
    # and the interactive (default) build DOES attach them, so the flag is what gates it
    pulse_i, _c, _r = build_pulse_preview_plot(st, include_always_off=True)
    assert pulse_i.interaction_handles(), "an interactive pulse build DOES attach selectors"
    plt.close(pulse_i.fig)


# --------------------------------------------------------------------------- 4. optimal_pulse_size single source
def test_optimal_pulse_size_is_the_default_source():
    """``default_pulse_size`` == ``optimal_pulse_size`` of the DRAWN rows + periods, and it picks a bigger
    preset for a busier pulse (monotone in content), capping at the largest preset."""
    st = _busy_state()
    rows, traces = pulse_drawn_rows(st, include_always_off=True)
    assert default_pulse_size(st, include_always_off=True) == \
        optimal_pulse_size(len(rows) + len(traces), len(st.periods))
    # monotone: a bigger content never picks a smaller preset, and the ceiling is the largest preset
    area = {s: (lambda rc: rc[0] * rc[1])(tuple(int(x) for x in s.split("x"))) for s in PANEL_SIZES}
    tiny = optimal_pulse_size(2, 2)
    huge = optimal_pulse_size(80, 80)
    assert area[tiny] <= area[huge]
    assert huge == max(PANEL_SIZES, key=lambda s: area[s]), "an over-large pulse caps at the biggest preset"


def test_preview_default_size_uses_optimal():
    """The pulse editor's preview default size is ``default_pulse_size`` -- the SAME source -- and its size
    dropdown offers exactly ``PANEL_SIZES``.  Picking a size PINS it (stops auto-tracking the content)."""
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    st = _busy_state()
    gui = PulseSequenceEditor(state=st, channels=list(st.channels))
    try:
        # unpinned: the effective preview size equals the shared default
        assert gui._preview_size_pinned is False
        eff = gui._preview_size_for(st, include_always_off=True)
        assert eff == default_pulse_size(st, include_always_off=True)
        assert [gui.preview_size_combo.itemText(i) for i in range(gui.preview_size_combo.count())] \
            == list(PANEL_SIZES)
        # picking a size pins it -> the pick wins over the optimal default
        gui.preview_size_combo.setCurrentText("4x4")
        gui._on_preview_size_picked()
        assert gui._preview_size_pinned is True
        assert gui._preview_size_for(st, include_always_off=True) == "4x4"
    finally:
        gui.deleteLater()
        plt.close("all")


# ------------------------------------- 4b. entering Preview / show-all toggle re-derive the optimal size
def test_preview_size_pin_is_transient_and_auto_recomputes():
    """The size PIN is a TRANSIENT in-Preview pick -- it is NOT kept across the two big context switches
    that change the natural size:

    (1) SWITCHING BACK to the Preview tab drops the pin and re-derives ``default_pulse_size`` for the
        current channel / period counts;
    (2) TOGGLING "show all / off channels" drops the pin and re-derives the optimal size for the NEW
        visible-channel count (asserted to actually CHANGE the size here);

    but a plain refresh WHILE staying on Preview keeps a pin (so a manual pick is sticky in-place)."""
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    st = _mostly_off_state()
    gui = PulseSequenceEditor(state=st, channels=list(st.channels))
    try:
        # (i) entering the Preview tab: pin is reset and the effective size is the shared optimal default.
        gui._preview_size_pinned = True                       # pretend a pin lingered from a prior visit
        preview_index = gui.tabs.indexOf(gui.preview_tab)
        gui.tabs.setCurrentWidget(gui.preview_tab)             # currentWidget() is preview_tab for the handler
        gui._on_tab_changed(preview_index)
        assert gui._preview_size_pinned is False, "entering Preview must drop a lingering size pin"
        assert gui._preview_size_for(st, include_always_off=False) \
            == default_pulse_size(st, include_always_off=False)

        # (ii) a manual pick PINS the size (unchanged behaviour) ...
        gui.preview_size_combo.setCurrentText("4x4")
        gui._on_preview_size_picked()
        assert gui._preview_size_pinned is True

        # ... and a plain refresh WHILE on Preview keeps that pin (a manual pick is sticky in-place).
        gui.refresh_preview()
        assert gui._preview_size_pinned is True, "a plain in-Preview refresh must NOT clear a manual pin"

        # (iii) toggling show-all drops the pin and re-derives the size for the NEW visible-channel count.
        # off channels hidden -> smallest preset; shown -> largest preset (the two defaults differ), so the
        # recompute is observable, not a no-op.
        default_hidden = default_pulse_size(st, include_always_off=False)
        default_shown = default_pulse_size(st, include_always_off=True)
        assert default_hidden != default_shown, "fixture must make show-all change the optimal size"
        gui.preview_include_off.setChecked(True)              # now showing the off rows
        gui._on_include_off_toggled()
        assert gui._preview_size_pinned is False, "a show-all toggle must drop the size pin"
        assert gui._preview_size_for(st, include_always_off=True) == default_shown, \
            "after show-all, the size must re-derive to the new (larger) optimal default"
    finally:
        gui.deleteLater()
        plt.close("all")


# --------------------------------------------------------------------------- 5. embedded console tab shadow
def test_embedded_console_reserves_tab_shadow_headroom():
    """The embedded TaskConsole leaves at least the tab drop-shadow's top bleed
    (``fluent_tab_shadow_margin``) above its tab strip, so the Monitor tab's top is not cut off (the
    figure-viewer regression).  Asserted on the real widget geometry."""
    from PyQt5 import QtCore, QtWidgets
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app, fluent_tab_shadow_margin
    from Zou_lab_control.frontend.task_console import TaskConsole, TaskConsoleState
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    con = TaskConsole(hub=SignalHub(), state=TaskConsoleState(name="t", panels=[]),
                      running_nodes=[], session=None, window_px=(900, 600), embedded=True)
    try:
        con.resize(900, 600)
        con.show()
        QtWidgets.QApplication.processEvents()
        tabs = con.tabs
        tab_top = tabs.mapTo(con, QtCore.QPoint(0, 0)).y()
        # the widget directly above the tabs is the header card; the gap between them must hold the bleed
        above_bottom = 0
        for child in con.findChildren(QtWidgets.QWidget):
            if child is tabs or not child.isVisible():
                continue
            geo_bottom = child.mapTo(con, QtCore.QPoint(0, child.height())).y()
            if geo_bottom <= tab_top and geo_bottom > above_bottom and child.parent() is con:
                above_bottom = geo_bottom
        gap = tab_top - above_bottom
        assert gap >= fluent_tab_shadow_margin(), \
            f"gap above tabs {gap} < shadow bleed {fluent_tab_shadow_margin()} (Monitor tab top clipped)"
    finally:
        con.shutdown()
        con.deleteLater()
        plt.close("all")
