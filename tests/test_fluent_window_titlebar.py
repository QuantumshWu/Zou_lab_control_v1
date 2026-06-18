"""Mechanical guard for the FluentWindow title alignment + pivot-tab styling.

These broke once because the title fix was tuned to a magic offset and verified on
the WRONG path (the title set once at construction) while the real GUI re-sets the
title at runtime -- which moved it hard against the window edge.  The fix makes the
title the first title-bar item at a fixed content-column margin (deterministic), so
this test asserts that invariant DIRECTLY (label left edge == the body column) AND
on the dynamic re-title path, so a future change cannot silently regress it.

Lightweight: a bare FluentWindow over a plain QWidget + geometry assertions, NOT a
full demo GUI build (no matplotlib, no flaky panels).  Skips cleanly where the
optional frameless title bar (qframelesswindow) is absent.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5 import QtCore, QtWidgets
    from Zou_lab_control.frontend import qt_fluent as qf
except Exception:  # pragma: no cover - no Qt available
    QtWidgets = None


def _settle(app, ms=120):
    import time
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.mark.skipif(QtWidgets is None, reason="PyQt5 not available")
def test_title_label_sits_on_content_column_even_after_dynamic_retitle():
    if qf.StandardTitleBar is None:
        pytest.skip("frameless StandardTitleBar (qframelesswindow) not installed")
    app = qf.ensure_qt_app()
    body = QtWidgets.QWidget()
    body.setFixedSize(600, 200)
    win = qf.FluentWindow(widget=body, title="Initial Title")
    win.show()
    _settle(app)
    target = qf.scaled_px(qf.TITLE_LEFT_INSET)
    # FluentWindow owns the title via its OWN free-child label (the built-in one is
    # hidden); it must sit on the body content column and TRACK the window title.
    label = win._zlc_title
    assert label is not None and label.isVisible()
    x0 = label.mapTo(win, QtCore.QPoint(0, 0)).x()
    assert abs(x0 - target) <= 1, f"title label x={x0}, expected content column {target}"
    assert label.text() == "Initial Title"
    # the real GUIs re-set the window title at runtime (file name + dirty star) AND
    # some (pulse_gui) call titleBar.setTitle right after -- none of that may move
    # OUR label, because it is not in the title bar layout
    win.setWindowTitle("some_file_20260614 - App (new)*")
    if win.titleBar is not None and hasattr(win.titleBar, "setTitle"):
        win.titleBar.setTitle("some_file_20260614 - App (new)*")  # the pulse_gui double-set
    _settle(app)
    x1 = label.mapTo(win, QtCore.QPoint(0, 0)).x()
    assert abs(x1 - target) <= 1, f"after re-title, label x={x1}, expected {target}"
    assert label.text() == "some_file_20260614 - App (new)*"
    win.close()


@pytest.mark.skipif(QtWidgets is None, reason="PyQt5 not available")
def test_pivot_tabs_have_no_box_fill_and_selected_underline():
    """Tabs must be the clean pivot look (transparent, accent underline on the
    selected one), NOT filled boxes -- the regression that read as 'uglier'."""
    qf.ensure_qt_app()
    tabs = qf.FluentTabWidget()
    qss = tabs.styleSheet()
    assert "QTabBar::tab" in qss
    # selected tab is marked by an accent bottom border (underline), not a fill
    assert "border-bottom" in qss and qf.ACCENT in qss
    # no per-tab background FILL colour (BG) that would draw grey boxes on the card
    import re
    tab_block = re.search(r"QTabBar::tab \{\{?(.*?)\}\}?", qss, re.S)
    assert tab_block is not None
    assert "background: transparent" in tab_block.group(1)


@pytest.mark.skipif(QtWidgets is None, reason="PyQt5 not available")
def test_tab_overflow_uses_a_menu_button_not_scroll_arrows():
    """The tab bar has NO native scroll arrows; when the tabs overflow, ONE corner
    overflow button (...) appears and lists EVERY tab, jumping to the picked one.  Few
    tabs -> the button is hidden (no clutter).  Regression for the disliked left/right
    scroll chevrons that also crowded the last tab's close 'x'."""
    app = qf.ensure_qt_app()
    w = qf.FluentTabWidget()
    # the native QTabBar scroll arrows must be OFF (they are what crowded the close x).
    assert w.usesScrollButtons() is False

    w.add_permanent_tab(QtWidgets.QLabel("m"), "Monitor")
    w.add_closable_tab(QtWidgets.QLabel("a"), "Camera (live frames)", focus=False)
    w.resize(1400, 320); w.show(); _settle(app)
    # two short-ish tabs in a wide bar do NOT overflow -> no overflow button.
    assert w._tabs_overflow() is False
    assert w._overflow_btn.isVisible() is False

    for name in ("Readout fidelity", "Temperature", "Judge occupancy",
                 "Calibrate readout", "Loading rate", "Per-site histogram"):
        w.add_closable_tab(QtWidgets.QLabel(name), name, focus=False)
    w.resize(560, 320); _settle(app)
    # many tabs in a narrow bar overflow -> the overflow button shows.
    assert w._tabs_overflow() is True
    assert w._overflow_btn.isVisible() is True

    # the overflow menu lists EVERY tab; triggering an action selects that tab (this
    # exercises the real click->select wiring, not just a direct setCurrentIndex).
    menu = w._overflow_menu()
    actions = menu.actions()
    assert len(actions) == w.count()
    assert [a.text() for a in actions] == [w.tabText(i) for i in range(w.count())]
    w.setCurrentIndex(0)
    actions[-1].trigger()
    assert w.currentIndex() == w.count() - 1
    menu.deleteLater()
    w.deleteLater()


def test_overflowing_tabs_keep_short_tabs_full_and_never_clip_the_close_x():
    """When the tabs overflow, the bar water-fills: the WIDEST tabs are capped (their labels
    elide) while SHORT tabs (Monitor / Logic) keep their natural width, and EVERY tab's box
    -- including the right-side close 'x' slot -- stays inside the bar (no half-painted tab,
    no clipped 'x').  Regression for the disliked equal-sliver squeeze (every tab crammed to
    width//count, short ones included) AND for the original 'x cut off' at the bar edge."""
    app = qf.ensure_qt_app()
    w = qf.FluentTabWidget()
    w.add_permanent_tab(QtWidgets.QLabel("m"), "Monitor")
    w.add_permanent_tab(QtWidgets.QLabel("l"), "Logic")
    for name in ("Readout image", "Per-site occupancy", "Loading rate (dist)",
                 "Loading rate", "Counts distribution", "Per-site counts"):
        w.add_closable_tab(QtWidgets.QLabel(name), name, focus=False)
    # narrow enough that the tabs overflow, wide enough that the short tabs still fit.
    w.resize(960, 320); w.show(); _settle(app)
    bar = w.tabBar()
    assert bar.is_overflowing() is True

    cap = bar._shrink_cap()
    assert cap is not None
    naturals = [QtWidgets.QTabBar.tabSizeHint(bar, i).width() for i in range(bar.count())]
    # water-fill INVARIANT: a tab narrower than the cap keeps its natural width (Monitor /
    # Logic are not shrunk), a wider tab is capped to the shared cap (its label elides).
    for i, nat in enumerate(naturals):
        assert bar.tabSizeHint(i).width() == (nat if nat <= cap else cap)
    # the cap genuinely SPARES short tabs (some tab is below it) and is NOT the equal-sliver
    # width//count of the old squeeze that crammed every tab (short ones included).
    assert min(naturals) < cap
    assert cap > bar.width() // bar.count()
    # EVERY tab (laid out left to right) ends within the bar: no tab -- and so no close 'x'
    # -- is clipped at the right edge.
    assert all(bar.tabRect(i).right() <= bar.rect().right() + 1 for i in range(bar.count()))
    w.deleteLater()
