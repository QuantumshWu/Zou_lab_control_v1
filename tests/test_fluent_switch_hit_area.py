"""#4 toggle-row: a FluentSwitch must toggle ONLY when the click lands on its visible track (or label),
never on the dead padding its wider minimum width reserves -- which fills the form row's control cell,
so without this a click "anywhere on the row" flipped the switch (a clear UX/design violation)."""

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
    FluentSwitch,
    FluentTriStateToggle,
    scaled_px,
)


def test_switch_toggles_on_its_track_but_not_on_the_dead_padding():
    ensure_qt_app()
    sw = FluentSwitch("")                       # the bool param widget creates a no-text switch
    sw.resize(sw.sizeHint())
    track_w = scaled_px(60, minimum=48)
    mid_y = sw.height() // 2
    # the widget is wider than its track (reserved for column alignment) -> there IS clickable padding
    assert sw.width() > track_w + 8
    # the visible track toggles; the padding to its right (which covers the rest of the row cell) does NOT
    assert sw.hitButton(QtCore.QPoint(track_w // 2, mid_y)) is True
    assert sw.hitButton(QtCore.QPoint(sw.width() - 2, mid_y)) is False


def test_switch_with_a_label_stays_clickable_across_its_label():
    ensure_qt_app()
    sw = FluentSwitch("free run")               # a labelled switch: the label is part of the control
    sw.resize(sw.sizeHint())
    mid_y = sw.height() // 2
    track_w = scaled_px(60, minimum=48)
    assert sw.hitButton(QtCore.QPoint(track_w // 2, mid_y)) is True      # the track
    assert sw.hitButton(QtCore.QPoint(track_w + scaled_px(8) + 2, mid_y)) is True   # onto the label text


def test_a_real_click_on_the_dead_padding_does_not_flip_the_switch():
    """The end-to-end behaviour the user reported: deliver an actual left-button click (press+release)
    to the widget.  A click on the dead padding (where the form row's control cell extends past the
    track) must leave isChecked() unchanged; a click on the track must toggle it."""
    from PyQt5.QtTest import QTest
    ensure_qt_app()
    sw = FluentSwitch("")
    sw.resize(sw.sizeHint())
    sw.setChecked(False)
    track_w = scaled_px(60, minimum=48)
    mid_y = sw.height() // 2
    QTest.mouseClick(sw, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(sw.width() - 2, mid_y))
    assert sw.isChecked() is False, "a click on the dead padding must NOT toggle the switch"
    QTest.mouseClick(sw, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(track_w // 2, mid_y))
    assert sw.isChecked() is True, "a click on the track must toggle the switch"


# --- FluentTriStateToggle: segment width tracks the labels, click selects that segment ---

def test_tristate_size_hint_widens_with_the_longest_label():
    """The capsule sizes each segment to the widest label, so a toggle of long labels is wider
    than one of short labels -- never a fixed width that clips text or wastes space."""
    ensure_qt_app()
    short = FluentTriStateToggle(["a", "b", "c"])
    long = FluentTriStateToggle(["Replace", "Block", "Parallel"])
    assert long.sizeHint().width() > short.sizeHint().width()
    # the hint is derived from the real font metrics of the widest label (+ uniform padding),
    # so it is at least three times that label's text width
    fm = QtGui.QFontMetrics(long.font())
    widest = max(fm.horizontalAdvance(o) for o in ["Replace", "Block", "Parallel"])
    assert long.sizeHint().width() >= 3 * widest


def test_tristate_horizontal_size_policy_is_fixed():
    """A settings row adds the control with stretch=1; only a FIXED horizontal policy keeps the
    row from ballooning the capsule past its sizeHint."""
    ensure_qt_app()
    toggle = FluentTriStateToggle(["Replace", "Block", "Parallel"])
    assert toggle.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Fixed


def test_tristate_click_selects_the_clicked_segment_not_the_next():
    """Click-to-segment: pressing inside segment i makes state == i (not (state+1)%n)."""
    ensure_qt_app()
    toggle = FluentTriStateToggle(["Replace", "Block", "Parallel"])
    toggle.resize(toggle.sizeHint())
    seg = toggle.width() / 3
    mid_y = toggle.height() // 2

    fired: list[int] = []
    toggle.activated.connect(fired.append)

    for target in (2, 0, 1):
        x = int(seg * target + seg / 2)
        ev = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            QtCore.QPointF(x, mid_y),
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        toggle.mousePressEvent(ev)
        assert toggle.state() == target, f"click in segment {target} must select it"

    assert fired == [2, 0, 1], "each distinct pick emits activated(index)"
