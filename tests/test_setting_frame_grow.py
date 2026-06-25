"""Contract: the plot-panel Setting frame GROWS to show all its content / when the panel size
grows, but NEVER shrinks back within a session (a height high-water mark, #H3i-2).  So enlarging a
panel enlarges the open Setting frame immediately, while shrinking it does NOT snap the frame
smaller; content beyond a screen-derived bound scrolls instead of clipping.
"""

import pytest


def test_setting_frame_height_is_grow_not_shrink(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    console = dt.demo_console(shots=3)
    try:
        card = console.cards[0]
        assert card._settings_h_hwm == 0                 # fresh
        card._size_settings_popup()                      # fit to content
        first = card._settings_h_hwm
        assert first > 0 and card.settings_popup.height() >= 1
        # simulate a previously-TALLER state, then re-size: the frame must NOT shrink below it
        card._settings_h_hwm = first + 400
        card._size_settings_popup()
        assert card._settings_h_hwm >= first + 400, "the high-water mark must never shrink"
        # a re-size request with the SAME content keeps it at the high-water mark (grow-not-shrink)
        card._size_settings_popup()
        assert card._settings_h_hwm >= first + 400
        # the popup never exceeds the screen-derived bound (content beyond it scrolls)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            assert card.settings_popup.maximumHeight() <= screen.availableGeometry().height()
    finally:
        console.shutdown()


def test_rebuilding_the_popup_resets_the_high_water_mark(monkeypatch):
    """A popup REBUILT with different content (e.g. a +/- signal slot) starts its height mark fresh
    so it fits the new content -- the hysteresis is per-content, not forever."""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    console = dt.demo_console(shots=3)
    try:
        card = console.cards[0]
        card._size_settings_popup()
        card._settings_h_hwm = card._settings_h_hwm + 999
        card._build_settings()                           # rebuild -> fresh content + reset mark
        assert card._settings_h_hwm == 0
    finally:
        console.shutdown()
