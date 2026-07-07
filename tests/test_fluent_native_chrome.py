"""The GUI's native-Qt chrome (right-click menus, message boxes, below-anchor popups) is Fluent-styled
from ONE source.

A raw ``QMenu`` / ``QMessageBox`` pops the platform's opaque SQUARE window -- a "black box" right-click
menu, a message dialog with a native title bar + a stray Python app icon -- that clashes with the app's
rounded, translucent Fluent cards.  These contract tests pin the single-source replacements so a
regression (a new ``QMessageBox`` call, a line edit that loses its Fluent context menu, an overflow list
that stops sharing the one popup gap) fails the build instead of silently reintroducing native chrome.
"""

import inspect
import os
from pathlib import Path
import re
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

FRONTEND = REPO_ROOT / "Zou_lab_control" / "frontend"


def test_no_native_message_box_call_in_frontend():
    """No ``QMessageBox.<method>(...)`` CALL anywhere in the frontend -- ``fluent_message`` (a frameless
    rounded Fluent dialog, no native title bar / Python icon) is the ONE message/warning surface.  The
    regex requires the trailing ``(`` so a prose mention in ``fluent_message``'s own docstring (which
    names the API it replaces) is not a false positive."""
    call = re.compile(r"QMessageBox\.\w+\(")
    offenders = []
    for py in FRONTEND.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if call.search(line):
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, "use fluent_message(...) instead of a native QMessageBox:\n" + "\n".join(offenders)


@pytest.fixture
def qt():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def test_line_edit_context_menu_is_the_fluent_rounded_menu(qt):
    """Right-clicking a ``FluentLineEdit`` pops the shared ``_FluentRoundedMenu`` (the one rounded Fluent
    card, also used by the tab overflow), NOT a native square ``QMenu`` -- and Qt's standard edit actions
    (Copy / Paste / Select All, wired to the edit) survive being re-homed into it."""
    from PyQt5 import QtCore, QtGui
    from Zou_lab_control.frontend.qt_fluent import FluentLineEdit, _FluentRoundedMenu
    le = FluentLineEdit("hello world")
    captured = {}
    original = _FluentRoundedMenu.exec_
    _FluentRoundedMenu.exec_ = lambda self, *a, **k: captured.setdefault("menu", self)   # do not block
    try:
        ev = QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(3, 3), QtCore.QPoint(30, 30))
        le.contextMenuEvent(ev)
    finally:
        _FluentRoundedMenu.exec_ = original
    menu = captured.get("menu")
    assert isinstance(menu, _FluentRoundedMenu), "context menu must be the Fluent rounded card, not a native QMenu"
    texts = " ".join(a.text() for a in menu.actions())
    assert "Copy" in texts and "Paste" in texts and "Select" in texts, texts


def test_overflow_menu_shares_the_one_popup_gap(qt):
    """The ``...`` tab-overflow list and the combo drop-down open with the SAME ``_popup_gap`` off their
    anchor.  Pins BOTH that the combo reads the single source AND that the overflow exec applies it (via
    source inspection -- a real ``exec_`` blocks), so neither can drift back to flush-against-the-anchor."""
    from Zou_lab_control.frontend.qt_fluent import FluentComboBox, FluentTabWidget, _popup_gap
    assert _popup_gap() > 0
    combo = FluentComboBox()
    assert combo._gap == _popup_gap(), "the combo drop-down must read the ONE _popup_gap source"
    src = inspect.getsource(FluentTabWidget._show_overflow_menu)
    assert "_popup_gap()" in src, "the overflow menu must offset its exec position by the shared _popup_gap"


def test_fluent_message_dialog_is_frameless_and_translucent(qt):
    """The Fluent message dialog has NO native title bar (so no stray Python icon in the corner) and is
    translucent, so it paints its OWN rounded card like every other Fluent surface."""
    from PyQt5 import QtCore
    from Zou_lab_control.frontend.qt_fluent import _FluentMessageDialog
    dlg = _FluentMessageDialog(None, "Pulse", "something to report", kind="warning")
    assert bool(dlg.windowFlags() & QtCore.Qt.FramelessWindowHint)
    assert dlg.testAttribute(QtCore.Qt.WA_TranslucentBackground)
