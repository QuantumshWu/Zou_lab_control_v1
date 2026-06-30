"""Scroll-over protection for Fluent numeric spin controls: the mouse wheel only changes the
value AFTER the widget has been clicked into (has keyboard focus).  Before focus, a wheel
event is ignored (NOT accepted) so it bubbles to the parent scroll area and the PAGE scrolls
instead of the number silently nudging -- the误触 the user reported when scrolling a Setting /
Edit panel that passes over a spinbox.

The gate is ``hasFocus`` (focusPolicy is StrongFocus so a left click grabs focus); both
FluentSpinBox and FluentDoubleSpinBox inherit the single ``_WheelFocusGuardMixin``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402
from Zou_lab_control.frontend.qt_fluent import (  # noqa: E402
    ensure_qt_app,
    FluentSpinBox,
    FluentDoubleSpinBox,
)


def _wheel_event(widget) -> QtGui.QWheelEvent:
    """A one-notch upward wheel event delivered at the widget's centre."""
    pos = QtCore.QPointF(widget.rect().center())
    return QtGui.QWheelEvent(
        pos,                              # local pos
        widget.mapToGlobal(widget.rect().center()),  # global pos
        QtCore.QPoint(0, 0),              # pixelDelta
        QtCore.QPoint(0, 120),            # angleDelta: +120 = one notch up
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase,
        False,
    )


def _deliver_wheel(widget) -> QtGui.QWheelEvent:
    ev = _wheel_event(widget)
    widget.wheelEvent(ev)
    return ev


def _force_focus(widget) -> None:
    """Make ``hasFocus()`` actually true under the offscreen platform: a top-level needs to be
    shown AND activated, then the application's active window set, before setFocus() takes."""
    app = ensure_qt_app()
    widget.show()
    widget.activateWindow()
    app.setActiveWindow(widget)
    widget.setFocus(QtCore.Qt.MouseFocusReason)
    app.processEvents()


def test_spinbox_focus_policy_is_strong_so_a_click_grabs_focus():
    """StrongFocus is what lets a left click focus the box (the prerequisite for the wheel to
    then work); without it a click would never enable wheel stepping."""
    ensure_qt_app()
    for box in (FluentSpinBox(), FluentDoubleSpinBox()):
        assert box.focusPolicy() == QtCore.Qt.StrongFocus


def test_unfocused_wheel_does_not_change_value_and_bubbles():
    """Scroll-over (no focus): value unchanged AND the event is NOT accepted, so it propagates
    to the parent scroll area -> the page scrolls."""
    ensure_qt_app()
    box = FluentSpinBox()
    box.setRange(0, 100)
    box.setValue(5)
    box.clearFocus()
    assert box.hasFocus() is False

    ev = _deliver_wheel(box)

    assert box.value() == 5, "an unfocused wheel must NOT change the value"
    assert ev.isAccepted() is False, "an unfocused wheel must bubble (not be accepted)"


def test_focused_wheel_changes_value():
    """Once clicked into (focused), the wheel steps the value normally."""
    ensure_qt_app()
    box = FluentSpinBox()
    box.setRange(0, 100)
    box.setValue(5)
    _force_focus(box)          # a real, activated top-level so setFocus actually takes
    assert box.hasFocus() is True

    _deliver_wheel(box)

    assert box.value() != 5, "a focused wheel must step the value"
    box.close()


def test_doublespin_unfocused_wheel_does_not_change_value_and_bubbles():
    ensure_qt_app()
    box = FluentDoubleSpinBox()
    box.setRange(0, 100)
    box.setSingleStep(1)
    box.setValue(5)
    box.clearFocus()
    assert box.hasFocus() is False

    ev = _deliver_wheel(box)

    assert box.value() == pytest.approx(5), "an unfocused wheel must NOT change the value"
    assert ev.isAccepted() is False, "an unfocused wheel must bubble (not be accepted)"


def test_doublespin_focused_wheel_changes_value():
    ensure_qt_app()
    box = FluentDoubleSpinBox()
    box.setRange(0, 100)
    box.setSingleStep(1)
    box.setValue(5)
    _force_focus(box)
    assert box.hasFocus() is True

    _deliver_wheel(box)

    assert box.value() != pytest.approx(5), "a focused wheel must step the value"
    box.close()
