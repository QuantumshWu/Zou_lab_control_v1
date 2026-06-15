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
