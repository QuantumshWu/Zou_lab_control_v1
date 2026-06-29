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

from PyQt5 import QtCore  # noqa: E402
from Zou_lab_control.frontend.qt_fluent import ensure_qt_app, FluentSwitch, scaled_px  # noqa: E402


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
