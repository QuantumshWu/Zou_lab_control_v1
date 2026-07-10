"""MECHANICAL guard for #H3s-F6: a DAC-bus delay gets an API slot EXACTLY like an
ordinary channel's delay -- no special-casing, no asymmetry.

The user's complaint: "为什么 DAC 没有 delay 的 api slot" -- a plain (TTL) channel's
delay row already carries a settable-by-name handle (``aN``, NOT scannable), but the
DAC-bus delay row was left a plain field.  A bus is a GROUP, so the handle FANS OUT at
the data layer: setting ``state.aN`` writes EVERY member channel's ``delays`` +
``delay_units`` uniformly, and reading it returns the shared value.  A delay stays
``api=True, scan=False`` for a bus too -- it must NEVER get a scan slot.
"""

from __future__ import annotations
from Zou_lab_control.neutral_atom.ports import PortCatalog

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.timing import PulseTableState, enumerate_pulse_params
from Zou_lab_control.neutral_atom.timing.pulse_table import ScanSlot


BUS = "da"
MEMBERS = ["da0", "da1", "da2"]


def _bus_state() -> PulseTableState:
    """A pulse with one multi-member analog bus plus a plain TTL channel -- the minimal
    state where a bus-delay slot and a channel-delay slot can be compared side by side."""
    return PulseTableState(
        port_catalog=PortCatalog.from_channels([*MEMBERS, "ttl0"], analog_buses={BUS: MEMBERS}),
        delay_units={"da0": "us", "da1": "us", "da2": "us"},
        time_step_ns=20.0,
    )


def test_bus_delay_api_slot_fans_out_to_all_members():
    """bind -> set -> read: a bus-delay handle sets ALL member channels uniformly and reads
    back the shared value, just like a single channel's delay handle does for itself."""
    st = _bus_state()
    name = st.bind_api_field("delay", BUS)
    assert name == "a1"                                   # first free handle, like the GUI allocates
    assert st.api_slot_for("delay", BUS) == "a1"          # round-trips by (kind, target)

    st.set_api("a1", 1.5e-6)                              # set via name (notebook/Task path)
    for ch in MEMBERS:
        assert st.delays[ch] == pytest.approx(1.5e-6)     # every member written...
        assert st.delay_units[ch] == "us"                 # ...with the slot's unit
    # the read side returns the shared member value (members are uniform after any set)
    assert st._read_api_field(st.api_slots[0]) == pytest.approx(1.5e-6)
    assert st.a1 == pytest.approx(1.5e-6)                 # attribute sugar reads the same


def test_bus_delay_attribute_sugar_writes_every_member():
    """``state.a1 = ...`` (the attribute sugar) fans out identically to ``set_api``."""
    st = _bus_state()
    st.bind_api_field("delay", BUS)
    st.a1 = 3.0e-6
    assert all(st.delays[ch] == pytest.approx(3.0e-6) for ch in MEMBERS)


def test_bus_delay_unit_defaults_to_first_member():
    """With no explicit unit, a bus-delay slot inherits the first member's unit (NOT a bare
    'ns' that would mis-scale us/ms delays) -- the #4 fix in bind_api_field."""
    st = _bus_state()                                     # members are "us"
    name = st.bind_api_field("delay", BUS)                # no unit= -> derive from first member
    slot = next(s for s in st.api_slots if s.name == name)
    assert slot.unit == "us"


def test_bus_delay_slot_is_listed_by_enumerate_pulse_params():
    """The bus-delay API slot is surfaced by the single-source enumerator (the GUI dropdown +
    notebook both read it); its label carries the bus NAME (the bus name IS the information)."""
    st = _bus_state()
    triples = enumerate_pulse_params(st)
    delay_targets = [(target, label) for kind, target, label in triples if kind == "delay"]
    targets = [t for t, _ in delay_targets]
    assert BUS in targets                                 # the bus-delay target is listed
    bus_label = next(label for t, label in delay_targets if t == BUS)
    assert BUS in bus_label                               # label names the bus
    # the bus's members are exposed VIA the bus slot, not as per-member delay rows
    for ch in MEMBERS:
        assert ch not in targets
    assert "ttl0" in targets                              # a plain channel is still listed


def test_bus_delay_symmetric_with_channel_delay():
    """A bus-delay slot and a plain-channel-delay slot are the SAME kind of thing: both bind,
    both set-by-name, both read back -- the asymmetry the user flagged is gone."""
    st = _bus_state()
    st.bind_api_field("delay", BUS)
    st.bind_api_field("delay", "ttl0")
    assert st.api_names() == ["a1", "a2"]                 # two delay handles, no special-casing
    st.set_api("a1", 2.0e-6)                              # bus handle
    st.set_api("a2", 5.0e-6)                              # channel handle
    assert all(st.delays[ch] == pytest.approx(2.0e-6) for ch in MEMBERS)
    assert st.delays["ttl0"] == pytest.approx(5.0e-6)


def test_bus_delay_never_gets_a_scan_slot():
    """delay is api=True/scan=False in FIELD_KINDS -- a bus delay must NEVER be scannable.
    Both the high-level bind_field path and a hand-crafted ScanSlot are rejected."""
    st = _bus_state()
    with pytest.raises(ValueError, match="cannot be scanned"):
        st.bind_field("delay", BUS)                       # scan-bind a delay -> rejected
    # the ScanSlot type itself refuses kind="delay" (delay is not a SCAN_SLOT_KIND)
    with pytest.raises(ValueError, match="scan slot kind"):
        ScanSlot(kind="delay", target=BUS, unit="us", nominal=0.0)


def test_bind_rejects_nothing_spurious():
    """Binding a bus delay is idempotent (re-bind returns the same handle) and does not
    invent extra slots or reject a legitimate second target."""
    st = _bus_state()
    first = st.bind_api_field("delay", BUS)
    again = st.bind_api_field("delay", BUS)               # idempotent
    assert first == again
    assert len(st.api_slots) == 1
    st.bind_api_field("delay", "ttl0")                    # a different target is accepted
    assert len(st.api_slots) == 2
