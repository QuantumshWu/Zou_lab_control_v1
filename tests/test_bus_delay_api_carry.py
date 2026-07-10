"""Goal: a DAC-bus delay API slot must behave EXACTLY like an ordinary channel delay api slot --
toggle ON then OFF, and several buses each holding their own independent slot.

The bug: the pulse GUI's ``_carry_api_slots`` re-validated a DELAY slot's target against the channel
list ALONE, but a bus-delay slot's target is the BUS NAME (not a channel), so it was DROPPED on every
``read_state()`` rebuild.  Consequence: clicking the dot a second time saw "no slot" and RE-bound
(could never toggle OFF), and binding a second bus dropped the first (only one bus could hold a slot).
The fix routes both the validator and the carry through one ``PulseTableState.is_delay_target`` source.
"""
from Zou_lab_control.neutral_atom.ports import PortCatalog

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState


def _bus_state() -> PulseTableState:
    return PulseTableState(
        port_catalog=PortCatalog.from_channels(["da0", "da1", "da2", "da3", "probe", "trig"], analog_buses={"busA": ["da0", "da1"], "busB": ["da2", "da3"]}),
    )


# ---- data layer: ONE source for "is this a valid delay target" -----------------------------
def test_is_delay_target_accepts_channel_and_bus_rejects_unknown():
    st = _bus_state()
    assert st.is_delay_target("probe") is True          # an ordinary channel
    assert st.is_delay_target("busA") is True            # a DAC bus (owns one fanned-out delay)
    assert st.is_delay_target("nope") is False           # neither -> rejected


def test_two_buses_each_hold_an_independent_delay_api_slot():
    st = _bus_state()
    a = st.bind_api_field("delay", "busA")
    b = st.bind_api_field("delay", "busB")
    assert a != b                                        # unique handles, like channel delays
    assert st.api_slot_for("delay", "busA") == a
    assert st.api_slot_for("delay", "busB") == b
    st.validate()                                        # both bus-delay slots are valid


# ---- the GUI bug: read_state() must NOT drop bus-delay api slots ----------------------------
@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    return ensure_qt_app()


def _editor(_app):
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    return PulseSequenceEditor(_bus_state())


def test_read_state_preserves_bus_delay_api_slots(_app):
    """Binding two bus delays then reading the GUI state back keeps BOTH (the carry no longer
    drops a target that is a bus rather than a bare channel)."""
    ed = _editor(_app)
    ed.state.bind_api_field("delay", "busA")
    ed.state.bind_api_field("delay", "busB")
    carried = ed.read_state()
    assert carried.api_slot_for("delay", "busA") is not None
    assert carried.api_slot_for("delay", "busB") is not None


def test_toggle_bus_delay_api_cancels_on_second_click(_app):
    """The delay dot cycles none -> API -> none.  A bus-delay dot must cancel on the SECOND click
    (it could not before: read_state dropped the slot, so click 2 saw none and re-bound)."""
    ed = _editor(_app)
    ed._toggle_delay_api("busA")                          # none -> API
    assert ed.read_state().api_slot_for("delay", "busA") is not None
    ed._toggle_delay_api("busA")                          # API -> none (cancel)
    assert ed.read_state().api_slot_for("delay", "busA") is None
