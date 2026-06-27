"""Contract (#H3s-F1): FIELD_KINDS is the SINGLE source for which pulse field kinds support a
scan slot (sN, streamed variable) vs an API slot (aN, named handle keeping the concrete value).

The derived SCAN_SLOT_KINDS / API_SLOT_KINDS, the ScanSlot/ApiSlot validation, and the GUI dot-cycle
all read this one table -- so delay's "API yes, SCAN never" (a fixed output delay line) is ONE row,
not three docstrings + two rejections + a bind guard.  This guard fails if a hand-maintained tuple
drifts from the table, or if delay ever becomes scannable.
"""

from __future__ import annotations

import pytest

from Zou_lab_control.neutral_atom.timing.pulse_table import (
    API_SLOT_KINDS, FIELD_KINDS, SCAN_SLOT_KINDS, ApiSlot, ScanSlot,
)


def test_derived_sets_are_exactly_the_table():
    assert SCAN_SLOT_KINDS == tuple(k for k, v in FIELD_KINDS.items() if v.scan)
    assert API_SLOT_KINDS == tuple(k for k, v in FIELD_KINDS.items() if v.api)
    # the three bindable kinds, no more no fewer
    assert set(FIELD_KINDS) == {"duration", "delay", "dac"}


def test_delay_is_api_only_never_scan():
    assert FIELD_KINDS["delay"].api is True
    assert FIELD_KINDS["delay"].scan is False
    assert "delay" not in SCAN_SLOT_KINDS
    assert "delay" in API_SLOT_KINDS


def test_scanslot_accepts_exactly_the_scan_kinds():
    for kind in FIELD_KINDS:
        if FIELD_KINDS[kind].scan:
            ScanSlot(kind=kind, target="0")            # constructs
        else:
            with pytest.raises(ValueError):
                ScanSlot(kind=kind, target="ch00")     # delay -> rejected as a scan slot


def test_apislot_accepts_exactly_the_api_kinds():
    targets = {"duration": "0", "delay": "ch00", "dac": "da@0"}
    for kind in FIELD_KINDS:
        if FIELD_KINDS[kind].api:
            ApiSlot(name="a1", kind=kind, target=targets[kind])   # constructs (incl. delay)
    with pytest.raises(ValueError):
        ApiSlot(name="a1", kind="bogus", target="0")


def test_every_kind_round_trips_through_apislot_dict():
    targets = {"duration": "0", "delay": "ch00", "dac": "da@0"}
    for kind in API_SLOT_KINDS:
        slot = ApiSlot(name="a3", kind=kind, target=targets[kind], unit="ns")
        back = ApiSlot.from_dict(slot.to_dict())
        assert (back.name, back.kind, back.target) == (slot.name, slot.kind, slot.target)
