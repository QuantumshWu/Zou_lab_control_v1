"""MECHANICAL guards for the API-slot handle + the "GUI and API are the same data" sync
contract (the two things the user demanded be impossible to regress):

  * an API slot (a1/a2...) is a NAMED handle on a period duration / channel delay / DAC value
    that ``set_api`` (and the ``state.aN`` sugar) set BY NAME, WITHOUT rewriting the field --
    several fields can share one name (e.g. both long frames of a long-short-long bracket);
  * the SHIPPED ``pulses/imaging_template.json`` is, on disk, the continuous long-short-long
    bracket (THREE emCCD frames + api a1/a2) -- not a single window (the user's #1 complaint);
  * EVERY fire path records a syncable ``PulseTableState`` (with ``periods``) -- a bare
    ``PulseSequence`` (a Task firing ``to_sequence()``) is reconstructed into a period table,
    so the pulse GUI's Sync can ALWAYS reload the editor from whatever was fired.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.timing import (
    PulseTableState,
    default_imaging_template,
    single_imaging_template,
)
from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod


def test_dac_api_slot_unit_is_normalized_to_value():
    """CONTRACT (#E3): a DAC value is a signed code, not a time -- ApiSlot construction (the
    ONE choke point every slot passes: GUI bind, from_dict, JSON load) forces unit='value'
    for kind='dac', so a legacy file that saved unit='ns' auto-corrects on load."""
    from Zou_lab_control.neutral_atom.timing.pulse_table import ApiSlot

    legacy = ApiSlot.from_dict({"name": "a1", "kind": "dac", "target": "da_x@0", "unit": "ns"})
    assert legacy.unit == "value"
    assert ApiSlot(name="a2", kind="dac", target="da_y@0", unit="ns").unit == "value"
    assert ApiSlot(name="a3", kind="dac", target="da_z@0").unit == "value"       # default path too
    # non-dac kinds keep their declared unit untouched
    assert ApiSlot(name="a4", kind="duration", target="0", unit="us").unit == "us"


def test_shipped_pulse_tables_carry_value_unit_on_every_dac_api_slot():
    """Every shipped ``pulses/*.json`` pulse TABLE loads with unit='value' on each DAC api slot
    (the mot_field template's a1/a2/a3 carried 'ns' -- a time unit displayed on a DAC code)."""
    pulses = REPO_ROOT / "pulses"
    dac_slots = 0
    for path in sorted(pulses.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != PulseTableState.schema:
            continue                                    # runtime-program snapshots are not tables
        state = PulseTableState.load(path)
        for slot in state.api_slots:
            if slot.kind == "dac":
                dac_slots += 1
                assert slot.unit == "value", f"{path.name}: dac api slot {slot.name} unit={slot.unit!r}"
    # the sweep must not be vacuous: the mot_field template alone contributes a1/a2/a3
    mot = PulseTableState.load(pulses / "mot_field_template.json")
    assert {s.name for s in mot.api_slots if s.kind == "dac"} == {"a1", "a2", "a3"}
    assert dac_slots >= 3


def test_unrolled_bracket_preserves_api_slots_with_remapped_targets():
    """A finite-bracket compile unrolls the table first; api-slot bindings must carry through that
    unroll (with their period targets remapped), exactly like scan_slots and clk_channels.  The
    rebuild used to BUILD the remapped api_slots and then omit ``api_slots=`` from the new state,
    silently dropping every API binding on any finite-bracket compile -- the same built-but-not-
    passed bug class the adjacent clk_channels comment documents."""
    st = PulseTableState(
        channels=["ch0"],
        periods=[PulsePeriod(1.0, [1], unit="us"),
                 PulsePeriod(2.0, [0], unit="us"),
                 PulsePeriod(3.0, [1], unit="us")],
        repeat_start=0, repeat_end=0, repeat_count=3,        # bracket the FIRST period x3
    )
    st.bind_api_field("duration", 2)                          # a duration handle on period AFTER the bracket
    st.bind_api_field("delay", "ch0", unit="us")              # a delay handle on a channel (target = name)
    assert len(st.api_slots) == 2

    u = st.unrolled_bracket()
    assert len(u.api_slots) == 2, "unrolled_bracket dropped the api-slot bindings"
    by_kind = {s.kind: s.target for s in u.api_slots}
    # period 2, after a (start=end=0, count=3) bracket, shifts to 2 + (3-1)*1 = 4
    assert by_kind["duration"] == "4", "the duration handle's period target must be remapped through the unroll"
    assert by_kind["delay"] == "ch0", "a delay handle targets a channel name -> unchanged by the unroll"
    u.validate()                                              # the carried bindings are still self-consistent


def test_api_slot_names_are_unique_each_binds_one_field():
    """Each API handle binds EXACTLY ONE field (names are unique, like the GUI allocates a
    fresh ``a<N>`` per click) -- ``validate`` rejects a duplicate; ``set_api`` writes only its
    one field; ``state.aN`` sugar and a round-trip through ``to_dict``/``from_dict`` both work."""
    st = default_imaging_template()
    assert st.api_names() == ["a1", "a2", "a3"]                     # three exposure cells, three handles

    st.set_api("a1", 0.03)                                          # first long -> 30 ms
    st.set_api("a2", 0.008)                                         # short readout -> 8 ms
    st.set_api("a3", 0.025)                                         # second long -> 25 ms (independent)
    assert st.periods[1].duration == pytest.approx(0.03)
    assert st.periods[3].duration == pytest.approx(0.008)
    assert st.periods[5].duration == pytest.approx(0.025)
    assert isinstance(st.periods[1].duration, float)                # concrete value, not "aN"

    st.a1 = 0.04                                                    # attribute sugar -- only a1
    assert st.periods[1].duration == pytest.approx(0.04) and st.a1 == pytest.approx(0.04)
    assert st.periods[5].duration == pytest.approx(0.025)           # a3 untouched
    with pytest.raises(ValueError):
        st.set_api("a9", 1.0)                                       # unknown handle fails loud

    # validate REJECTS a hand-crafted duplicate name (the data layer agrees with the GUI)
    st.api_slots.append(st.api_slots[0])                            # would be two a1's
    with pytest.raises(ValueError, match="more than once"):
        st.validate()
    del st.api_slots[-1]
    r = PulseTableState.from_dict(st.to_dict())                     # round-trips
    assert [(s.name, s.target) for s in r.api_slots] == [(s.name, s.target) for s in st.api_slots]


def test_api_slot_on_delay_is_allowed():
    """A delay can be an API slot (set by name) even though it is NOT scannable."""
    st = single_imaging_template()
    ch = st.channels[0]
    st.bind_api_field("delay", ch, unit="us")
    st.set_api(st.api_slot_for("delay", ch), 1.5)
    assert st.delays[ch] == pytest.approx(1.5)


def test_shipped_imaging_template_file_is_three_emccd_bracket():
    """The FILE on disk is the long-short-long bracket (THREE emCCD frames) and exposes ONE
    api handle per exposure cell (a1 / a2 / a3, unique like the GUI allocates)."""
    data = json.loads((REPO_ROOT / "pulses" / "imaging_template.json").read_text(encoding="utf-8"))
    emccd = data["channels"].index("emCCD")
    n_triggers = sum(1 for p in data["periods"] if p["states"][emccd])
    assert n_triggers == 3                                          # long-short-long, on disk
    names = [s["name"] for s in data.get("api_slots", [])]
    assert names == ["a1", "a2", "a3"]                              # unique, one per exposure cell
    # the shipped probe template (the Pulse-scan default) is the SINGLE-image counterpart
    probe = json.loads((REPO_ROOT / "pulses" / "probe_template.json").read_text(encoding="utf-8"))
    assert sum(1 for p in probe["periods"] if p["states"][probe["channels"].index("emCCD")]) == 1


def test_every_fire_path_records_a_syncable_table():
    """The pulse GUI's Sync reads ``snapshot()['last_payload_json']`` and needs a table (a dict
    WITH ``periods``).  Firing a BARE PulseSequence (what a Task does) must still record one --
    reconstructed via ``from_sequence`` -- so sync works no matter who fired.  virtual == real:
    this is the same record seam both sequencers share."""
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (48, 60)}, seed=1)
    try:
        seqr = exp.devices.sequencer
        # a Task fires a COMPILED bracket (no periods of its own)
        tpl = default_imaging_template()
        tpl.set_api("a1", 0.02)
        tpl.set_api("a2", 0.005)
        tpl.set_api("a3", 0.02)
        seq = tpl.to_sequence(name="reference_bracket")
        seqr.prepare(seq)
        data = json.loads(seqr.snapshot()["last_payload_json"])
        assert isinstance(data, dict) and "periods" in data        # syncable (NOT a raw sequence)
        synced = PulseTableState.from_dict(data)                    # the GUI's exact sync step
        emccd = synced.channels.index("emCCD")
        assert sum(1 for p in synced.periods if p.states[emccd]) == 3   # the fired long-short-long

        # a GUI-authored PulseTableState round-trips with its api slots intact (names preserved)
        seqr.prepare(tpl)
        back = PulseTableState.from_dict(json.loads(seqr.snapshot()["last_payload_json"]))
        assert back.api_names() == ["a1", "a2", "a3"]
    finally:
        exp.close()
