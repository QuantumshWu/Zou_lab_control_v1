"""Contract for the Monitor dashboard "Selectors" switch (console header).

The Monitor board is display-only BY DEFAULT: every panel now BUILDS its full selector
layer (area / cross / zoom-pan / draggable threshold+clim lines) but PARKS it inactive,
so the default board behaves exactly like the historical ``interactions=False`` cards
(no misclick can zoom / select / drag).  The header's "Selectors" switch arms the layer
IN PLACE (``BaseLivePlot.set_selectors_active`` -- never a rebuild):

  * default OFF          -> every tool inactive (area widget inactive, self-managed
                            tools disconnected) and a real click event does NOT land;
  * switch ON            -> the SAME plotter's tools respond (a synthesized right-click
                            draws the crosshair) and the wheel policy flips to in-plot;
  * add a panel while ON -> the new card inherits the armed state (2d also proves the
                            clim DragHLine -- built outside _attach_interactions -- gates);
  * switch OFF again     -> everything parks back and events stop landing.

Also: the gate is a safe no-op on a tool-less plot (``interactions=False``), so the grid
thumbnails (which stay display-only by design -- their interactive surface is the
enlarged focus cell) can never crash it.

Offscreen Qt; logic assertions only (no screenshots).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def _console_with_hub():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    hub = SignalHub()
    rng = np.random.default_rng(7)
    hub.publish({
        "vals": rng.normal(100.0, 5.0, 400),
        "frame": rng.normal(200.0, 10.0, (32, 48)),
    })
    console = TaskConsole(hub=hub, state=default_console_state(), window_px=(1100, 700))
    console._timer.stop()
    return console


def _add_card(console, kind: str, signal: str):
    kc = console.kind_combo
    idx = next(j for j in range(kc.count()) if kc.itemData(j) == kind)
    kc.setCurrentIndex(idx)
    console._add_panel()
    card = console.cards[-1]
    # wire the panel to the published signal (the state a Setting pick leaves behind:
    # the input slot names the hub signal and the source reads it)
    card.config.inputs = [signal]
    card.config.source = "value = signal"
    card._compiled_source = card.config.source
    console.refresh_once()
    assert card.plotter is not None, f"{kind} card must build from the published signal"
    return card


def _self_managed(plotter):
    """The plot's self-managed tools (cross / zoom-pan / draggable lines) -- everything
    except the AreaSelector, whose gate is the wrapped matplotlib widget's set_active."""
    handles = list(plotter.interaction_handles())
    handles += [d for d in getattr(plotter, "threshold_draggers", ()) if d is not None]
    return [h for h in handles if getattr(h, "selector", None) is None]


def _right_click(plotter):
    """Dispatch a REAL right-button press at the axes centre through the canvas callback
    registry -- the same path a user's click takes -- so active/inactive is proven by
    behaviour, not only by flags."""
    from matplotlib.backend_bases import MouseEvent

    ax = plotter.ax
    xm = float(np.mean(ax.get_xlim()))
    ym = float(np.mean(ax.get_ylim()))
    x, y = ax.transData.transform((xm, ym))
    canvas = plotter.fig.canvas
    event = MouseEvent("button_press_event", canvas, float(x), float(y), button=3)
    canvas.callbacks.process("button_press_event", event)


def test_selectors_switch_gates_dashboard_cards():
    console = _console_with_hub()
    try:
        assert not console.selectors_switch.isChecked(), "the switch defaults OFF (display-only board)"
        hist = _add_card(console, "hist", "vals")

        # (a) default OFF: the layer is BUILT (ready to arm in place) but every tool is parked
        p = hist.plotter
        assert p.area is not None and p.cross is not None and p.zoom is not None, \
            "the Monitor card builds its selector layer (parked); it is no longer tool-less"
        assert not p.area.selector.get_active(), "area selector must be inactive by default"
        assert _self_managed(p), "hist must have self-managed tools (cross / zoom / threshold drag)"
        for h in _self_managed(p):
            assert getattr(h, "_zlc_selectors_off", False), \
                f"{type(h).__name__} must be disconnected while OFF"
        _right_click(p)
        assert p.cross.xy is None, "a right-click while OFF must NOT draw the crosshair"
        assert getattr(hist.canvas, "_zlc_isolate_wheel", None) is False, \
            "while OFF the wheel keeps scrolling the board"

        # (b) switch ON: the existing card arms IN PLACE -- the SAME plotter, now responding
        console.selectors_switch.setChecked(True)
        assert hist.plotter is p, "arming is in place -- never a rebuild"
        assert p.area.selector.get_active(), "area selector must arm on switch ON"
        for h in _self_managed(p):
            assert not getattr(h, "_zlc_selectors_off", True), \
                f"{type(h).__name__} must reconnect on switch ON"
        _right_click(p)
        assert p.cross.xy is not None, "a right-click while ON must draw the crosshair"
        assert hist.canvas._zlc_isolate_wheel is True, \
            "while ON the wheel zooms in-plot (the Edit-tab rule)"

        # (c) a panel added while ON inherits the armed state
        second = _add_card(console, "hist", "vals")
        assert second.plotter.area.selector.get_active(), "a card added while ON must arm immediately"
        for h in _self_managed(second.plotter):
            # a fresh tool that was never parked carries no marker at all -- absent == live
            assert not getattr(h, "_zlc_selectors_off", False), \
                f"{type(h).__name__} on a card added while ON must be live"

        # (d) switch OFF parks every card back to the display-only board
        console.selectors_switch.setChecked(False)
        assert not p.area.selector.get_active()
        assert not second.plotter.area.selector.get_active()
        for card in (hist, second):
            for h in _self_managed(card.plotter):
                assert getattr(h, "_zlc_selectors_off", False), \
                    f"{type(h).__name__} must disconnect again on OFF"
        xy_before = p.cross.xy
        _right_click(p)
        assert p.cross.xy == xy_before, "clicks must stop landing after OFF"
        assert hist.canvas._zlc_isolate_wheel is False
    finally:
        console.shutdown()


def test_clim_drag_lines_follow_the_gate():
    """The 2D panel's clim ``DragHLine`` is constructed OUTSIDE ``_attach_interactions``
    (in the shared distribution-band builder), so it must still follow the gate -- the
    enumeration goes through ``interaction_handles`` which carries ``self.drag``."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    import matplotlib.pyplot as plt
    from Zou_lab_control.frontend.live import panel_plot

    ny, nx = 8, 10
    xx, yy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    data_x = np.column_stack([xx.ravel(), yy.ravel()])
    values = np.random.default_rng(3).normal(10.0, 2.0, ny * nx)
    p = panel_plot(data_x, values, kind="2d", interactions=True)
    try:
        assert p.drag is not None, "the 2d panel builds draggable clim lines"
        p.set_selectors_active(False)
        assert getattr(p.drag, "_zlc_selectors_off", False), "clim drag must disconnect while OFF"
        assert not p.area.selector.get_active()
        p.set_selectors_active(True)
        assert not getattr(p.drag, "_zlc_selectors_off", True), "clim drag must reconnect on ON"
        assert p.area.selector.get_active()
    finally:
        plt.close(p.fig)


def test_set_selectors_active_is_safe_on_a_toolless_plot():
    """``interactions=False`` builds no tools -- the gate must be a silent no-op both ways
    (the grid thumbnails stay display-only by design and route through this path)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    import matplotlib.pyplot as plt
    from Zou_lab_control.frontend.live import panel_plot

    p = panel_plot(np.arange(10, dtype=float), np.zeros(10), kind="1d", interactions=False)
    try:
        assert p.interaction_handles() == []
        p.set_selectors_active(True)      # nothing to arm -> must not raise
        p.set_selectors_active(False)
        assert p.interaction_handles() == []
    finally:
        plt.close(p.fig)
