"""DAC bus editing works end-to-end, in the SIGNED user range, without crashing.

The operator's flow -- show a DAC bus, pick Edge, type a value, read/save/reload --
used to die or lie four separate ways, all one root: the GUI spoke UNSIGNED lane
codes while the model speaks SIGNED values around true 0 V (``bus_signed_range``,
the model's one source):

* committing any legal value fired ``_refresh_bus_displays`` which read the never-
  created ``card.bus_dots`` -- an AttributeError inside a Qt slot, killing the
  process with no traceback;
* the GUI clamped input to [0, 2^B-1], so a legal negative code could not be typed
  and +777 was accepted only to blow up in ``read_state``;
* a round-trip decoded the lanes WITHOUT subtracting ``bus_zero_code``, so 300 came
  back as 812;
* an untouched bus defaulted to ``edge`` instead of the model's ``hold``, so one
  read/load cycle stamped every visible bus edge@0 and poisoned the state forever.

Driven the way a person drives it: real widgets, real signals, real round-trips.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtTest
import pytest

from zlc_frontend.qt_widgets import ensure_qt_app


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture
def editor(application):
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    editor = open_pulse_editor()
    window = editor.window()
    window.show()
    for _ in range(6):
        application.processEvents()
    # The operator makes two DAC buses visible (the channel picker's effect on state).
    state = editor.read_state()
    state.visible_ports = list(state.visible_ports) + ["da_dipole", "da_bias_x"]
    editor.load_state(state)
    for _ in range(8):
        application.processEvents()
    yield editor
    try:
        window.close()
    except Exception:                                    # pragma: no cover - teardown only
        pass
    application.processEvents()


def _settle(application, n=6):
    for _ in range(n):
        application.processEvents()


def _first_card(editor):
    return editor.drag_container.pulse_cards()[0]


def _type_value(application, edit, text):
    edit.setFocus()
    edit.selectAll()
    QtTest.QTest.keyClicks(edit, text)
    edit.editingFinished.emit()
    _settle(application)


def test_dac_edit_round_trip_speaks_signed_values(editor, application):
    """Edge + 300 survives read_state -> load_state as 300, never the +512 wire code."""

    card = _first_card(editor)
    assert "da_dipole" in card.bus_dots, "the DAC value edit's scan dot is not registered"
    card.bus_mode_combos["da_dipole"].setCurrentText("Edge")
    _settle(application)
    _type_value(application, card.bus_value_edits["da_dipole"], "300")
    state = editor.read_state()
    assert state.analog_bus_modes["da_dipole"][0] == {"mode": "edge", "value": 300}

    editor.load_state(state)
    _settle(application, 8)
    shown = _first_card(editor).bus_value_edits["da_dipole"].text()
    assert shown == "300", (
        f"the DAC field shows {shown!r} after a round-trip -- the offset-binary wire "
        "code is leaking into the display")


def test_dac_accepts_the_signed_negative_range(editor, application):
    """-100 is a legal signed LSB value and must be typeable and stored as-is."""

    card = _first_card(editor)
    card.bus_mode_combos["da_dipole"].setCurrentText("Edge")
    _settle(application)
    _type_value(application, card.bus_value_edits["da_dipole"], "-100")
    state = editor.read_state()
    assert state.analog_bus_modes["da_dipole"][0]["value"] == -100


def test_untouched_buses_hold_and_repeated_round_trips_stay_clean(editor, application):
    """A bus the operator never touched HOLDS; read/load cycles never poison the state."""

    card = _first_card(editor)
    card.bus_mode_combos["da_dipole"].setCurrentText("Edge")
    _settle(application)
    _type_value(application, card.bus_value_edits["da_dipole"], "42")
    state = editor.read_state()
    assert state.analog_bus_modes["da_bias_x"][0] == {"mode": "hold", "value": None}, (
        "an untouched bus must default to hold -- edge@0 here is the poisoning bug")

    # Two full round-trips plus a scan-dot cycle: every step goes through read_state,
    # which used to raise forever after the first poisoned cycle.
    for _ in range(2):
        editor.load_state(editor.read_state())
        _settle(application, 8)
    card = _first_card(editor)
    card.busScanRequested.emit("da_dipole")
    _settle(application, 8)
    state = editor.read_state()
    assert state.scan_slots and state.scan_slots[0].kind == "dac"
    assert editor.preview_png_bytes(state, include_always_off=True)


def test_dac_mode_cycling_never_raises_in_a_slot(editor, application):
    """Cycling Edge/Ramp/Hold repeatedly is exception-free (the bus_dots crash)."""

    import sys

    caught = []
    previous = sys.excepthook
    sys.excepthook = lambda t, v, tb: caught.append(v)
    try:
        card = _first_card(editor)
        combo = card.bus_mode_combos["da_dipole"]
        for mode in ("Ramp", "Hold", "Edge", "Hold", "Ramp", "Edge"):
            combo.setCurrentText(mode)
            _settle(application)
    finally:
        sys.excepthook = previous
    assert not caught, f"a Qt slot raised during DAC mode cycling: {caught[0]!r}"
