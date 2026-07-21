"""A dropdown opens under its box.  Always.  Including at the bottom of the screen.

Qt's default is to FLIP a combo popup above its box when the space below runs out.
That is the behaviour under complaint: the list jumps somewhere else exactly when
the operator is working near the screen edge, and the box they clicked no longer
points at the list they are reading.  The rule here is unconditional -- anchored
below, always -- and overrun is absorbed by letting the popup scroll, never by
moving it.

Measured on the real editor window through the real ``showPopup()``, because the
popup is its own top-level window: nothing about its placement is observable from
the widget tree alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets
import pytest

from zlc_frontend.qt_widgets import ensure_qt_app

#: The deliberate few-px gap between the box and the card below it (the same gap the
#: Setting popup uses).  Placement is "below", not "flush", so a small positive offset
#: is expected -- what must never happen is a NEGATIVE one.
MAX_GAP_PX = 24


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture
def editor(application):
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    editor = open_pulse_editor()
    window = editor.window()
    window.show()
    for _ in range(5):
        application.processEvents()
    yield editor
    try:
        window.close()
    except Exception:                                    # pragma: no cover - teardown only
        pass
    application.processEvents()


def _anchor_report(combo: QtWidgets.QComboBox, application) -> tuple[int, int]:
    """Open the dropdown for real and return (dx, dy) from the box's bottom-left."""

    box_bottom_left = combo.mapToGlobal(QtCore.QPoint(0, combo.height()))
    combo.showPopup()
    for _ in range(4):
        application.processEvents()
    try:
        view = combo.view()
        popup = view.window() if view is not None else None
        assert popup is not None, "combo has no popup window after showPopup()"
        geometry = popup.geometry()
        return (geometry.left() - box_bottom_left.x(),
                geometry.top() - box_bottom_left.y())
    finally:
        combo.hidePopup()
        application.processEvents()


def test_every_dropdown_in_the_pulse_editor_opens_under_its_box(editor, application):
    window = editor.window()
    combos = [combo for combo in window.findChildren(QtWidgets.QComboBox) if combo.count()]
    assert combos, "the pulse editor window has no populated combo boxes to check"

    misplaced = []
    for index, combo in enumerate(combos):
        dx, dy = _anchor_report(combo, application)
        if dx != 0 or not 0 <= dy <= MAX_GAP_PX:
            misplaced.append(f"{combo.objectName() or f'combo#{index}'}: dx={dx:+d} dy={dy:+d}")
    assert not misplaced, (
        "these dropdowns did not open directly under their box:\n" + "\n".join(misplaced))


def test_a_dropdown_near_the_screen_bottom_still_opens_downward(editor, application):
    """The case Qt would flip: not enough room below.

    The popup must stay under the box and give up HEIGHT (it scrolls) rather than
    move.  Flipping is what the operator reported as the list "jumping".
    """

    window = editor.window()
    combos = [combo for combo in window.findChildren(QtWidgets.QComboBox) if combo.count() > 2]
    assert combos, "need a multi-item combo to have a popup taller than a sliver"
    combo = combos[0]

    screen = application.primaryScreen()
    available = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1280, 760)
    # Put the window's bottom edge just above the screen bottom, so this combo has only
    # a few pixels of room beneath it.
    window.move(window.x(), max(available.top(), available.bottom() - window.height() + 1))
    for _ in range(4):
        application.processEvents()

    box_bottom = combo.mapToGlobal(QtCore.QPoint(0, combo.height())).y()
    room_below = available.bottom() - box_bottom
    dx, dy = _anchor_report(combo, application)

    assert dx == 0 and dy >= 0, (
        f"with only {room_below}px below the box the dropdown moved (dx={dx:+d}, dy={dy:+d}); "
        "it must stay anchored and scroll instead")
