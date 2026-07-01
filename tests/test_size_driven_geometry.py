"""Contract: ONE size-driven geometry source for EVERY panel kind (incl. pulse + grid).

These pin the invariants of the size-unification round:

1. a pulse / grid panel's DATA region scales with the ``size`` preset (2x2 vs 4x4 differ), exactly like
   the single-axes kinds -- the size preset carries the content density, not a bespoke content-inches;
2. a grid's FOCUSED cell fills a data box with the SAME absolute margins a same-size single-axes panel
   uses (``panel_margins_px``), so its x/y label + title never clip -- asserted to the pixel;
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
    SiteHistogramGrid,
    build_grid_figure,
    build_pulse_preview_plot,
    default_pulse_size,
    grid_recipe_from_cells,
    optimal_pulse_size,
    panel_margins_px,
    panel_plot_spec,
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


def test_grid_cells_scale_with_size():
    """A grid's per-cell box grows with the size preset (2x2 vs 4x4 give a different figure size)."""
    recipe = _grid_recipe()
    small = build_grid_figure(recipe, size="2x2", display=False)
    big = build_grid_figure(recipe, size="4x4", display=False)
    sw, sh = small.fig.get_size_inches()
    bw, bh = big.fig.get_size_inches()
    assert bw > sw and bh > sh, f"grid 4x4 ({bw:.2f}x{bh:.2f}) must exceed 2x2 ({sw:.2f}x{sh:.2f})"
    plt.close(small.fig); plt.close(big.fig)


# --------------------------------------------------------------------------- 2. focus == same-size panel
@pytest.mark.parametrize("size", ["1x2", "2x2", "4x4"])
def test_grid_focus_cell_margins_equal_same_size_panel(size):
    """A focused grid cell's absolute px margins EQUAL ``panel_margins_px`` -- the SAME margins a
    standalone single-axes panel of this size uses -- so the enlarged cell's x/y label never clips."""
    recipe = _grid_recipe()
    g = build_grid_figure(recipe, size=size, display=False)
    g.focus(0)
    fa = g.focus_ax
    dpi = design_dpi(g.fig)
    fw = float(g.fig.get_size_inches()[0]) * dpi
    fh = float(g.fig.get_size_inches()[1]) * dpi
    pos = fa.get_position()
    got = (round(pos.x0 * fw), round(fw - (pos.x0 + pos.width) * fw),
           round(pos.y0 * fh), round(fh - (pos.y0 + pos.height) * fh))
    assert got == panel_margins_px("default"), \
        f"focused cell margins {got} must equal a same-size panel's {panel_margins_px('default')}"
    g.unfocus()
    plt.close(g.fig)


def test_focus_is_visibility_flip_not_a_resize():
    """Focus does NOT resize the figure -- it hides the grid axes and adds one focus axes (a visibility
    flip + draw), so the figure size before / during / after focus is identical."""
    recipe = _grid_recipe()
    g = build_grid_figure(recipe, size="2x2", display=False)
    before = tuple(g.fig.get_size_inches())
    g.focus(0)
    during = tuple(g.fig.get_size_inches())
    g.unfocus()
    after = tuple(g.fig.get_size_inches())
    assert before == during == after, f"focus must not resize the figure: {before} {during} {after}"
    plt.close(g.fig)


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
