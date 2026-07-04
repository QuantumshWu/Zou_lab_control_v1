"""#5: a burst of rebuild-class param edits (a fast scroll) is COALESCED into ONE race-safe rebuild.

This is a GENERAL anti-racing mechanism on the rebuild path, not a history-specific shortcut: any param
the plotter cannot apply in place (history length, colormap, ...) schedules a rebuild on a debounce timer
instead of rebuilding inline.  A fast burst restarts the timer, so the heavy teardown+rebuild runs ONCE
after the burst settles, on the LATEST config.params -- removing the per-tick rebuild race where the old
figure was torn down faster than the Qt holder could reflow (the figure flickered / grew / vanished).

The two invariants this guards:
  * coalescing -- N rapid edits cause 0 inline rebuilds; ONE rebuild when the timer fires, at the last value;
  * never-blank -- the OLD plotter stays installed through the whole burst (and the build-then-swap is
    atomic), so card.plotter is never None and the card is never a white panel mid-scroll.
History is used as the representative rebuild-class param; the mechanism is not keyed on it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _console():
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(900, 600))
    console._timer.stop()
    return exp, console


def _monitor_card(console):
    from Zou_lab_control.frontend import panel_plot
    from Zou_lab_control.frontend.task_console import PanelConfig, GAP
    card = console._new_panel_card(PanelConfig(kind="monitor", title="rate", source="value = rate",
                                               row=GAP, col=GAP, size="2x2"))
    console._attach_card(card)
    # one render so there IS a plotter to keep installed across the burst
    card.refresh({"rate": 3.0, "shot": 1})
    assert card.plotter is not None
    return card


def test_burst_of_length_edits_coalesces_into_one_rebuild():
    """Many rapid history edits (a scroll) trigger ZERO inline rebuilds -- they only (re)start the
    debounce timer; when it fires ONCE the rebuild runs with the LAST value.  The card never blanks."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()
    exp, console = _console()
    try:
        card = _monitor_card(console)
        builds = {"n": 0, "last_len": None}
        orig_build = card._build_plot

        def _counting_build(value, namespace=None):
            builds["n"] += 1
            builds["last_len"] = int(card.config.params.get("length", 300))
            return orig_build(value, namespace)

        card._build_plot = _counting_build

        # a fast scroll: many length changes in a row
        for L in (280, 260, 240, 200, 150, 120, 90, 75):
            card._set_param("length", L)
            assert card.plotter is not None, "the card must never blank mid-scroll (old plotter stays)"
            assert card._rebuild_timer.isActive(), "each edit (re)arms the coalescing timer"

        assert builds["n"] == 0, "a burst must NOT rebuild inline -- the rebuilds are coalesced"
        assert card.config.params["length"] == 75, "config.params holds the latest value immediately"

        # the timer firing once does the SINGLE rebuild, at the last value
        card._run_pending_rebuild()
        assert builds["n"] == 1, "exactly ONE rebuild after the burst settles"
        assert builds["last_len"] == 75, "the single rebuild uses the LAST scrolled value"
        assert card.plotter is not None
    finally:
        console.shutdown()


def test_rebuild_keeps_the_card_fixed_size_and_never_null_plotter():
    """The card is setFixedSize, so a rebuild swap cannot reflow it (no "图变大"); and the atomic
    build-then-swap keeps a plotter installed the whole time (no "图消失")."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()
    exp, console = _console()
    try:
        card = _monitor_card(console)
        size_before = (card.width(), card.height())
        for L in (250, 180, 120, 320):
            card._set_param("length", L)
            assert card.plotter is not None
            card._run_pending_rebuild()                 # fire each coalesced rebuild
            assert card.plotter is not None
            assert (card.width(), card.height()) == size_before, "a rebuild must not resize the card"
        # the live data stream still works after rebuilds (the monitor rolls into the fresh plotter)
        card.refresh({"rate": 9.0, "shot": 2})
        assert card.plotter is not None
    finally:
        console.shutdown()


def test_shutdown_cancels_a_pending_rebuild():
    """A pending coalesced rebuild must not fire into a torn-down card."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()
    exp, console = _console()
    card = _monitor_card(console)
    card._set_param("length", 123)
    assert card._rebuild_timer.isActive()
    card.shutdown()
    assert not card._rebuild_timer.isActive(), "shutdown must stop the debounce timer"
    assert card._force_rebuild is False
    console.shutdown()
