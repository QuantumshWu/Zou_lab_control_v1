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


def test_api_slot_set_by_name_updates_every_tagged_field():
    """One ``set_api`` updates EVERY field sharing the handle (the long-short-long needs both
    long frames on a1); the field keeps its concrete value (not rewritten to ``aN``); the
    ``state.aN = value`` sugar and a round-trip through to_dict/from_dict both work."""
    st = default_imaging_template()
    assert st.api_names() == ["a1", "a2"]
    assert sum(1 for s in st.api_slots if s.name == "a1") == 2     # both long reference frames

    st.set_api("a1", 0.03)                                          # long -> 30 ms on BOTH frames
    st.set_api("a2", 0.008)                                         # short readout -> 8 ms
    assert st.periods[1].duration == pytest.approx(0.03)
    assert st.periods[5].duration == pytest.approx(0.03)
    assert st.periods[3].duration == pytest.approx(0.008)
    # the field still shows a concrete number (NOT an "aN" expression like a scan slot would)
    assert isinstance(st.periods[1].duration, float)

    st.a1 = 0.025                                                   # attribute sugar
    assert st.periods[1].duration == pytest.approx(0.025) and st.a1 == pytest.approx(0.025)
    with pytest.raises(ValueError):
        st.set_api("a9", 1.0)                                       # unknown handle fails loud

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
    """The user's #1 complaint: the FILE on disk must be the long-short-long bracket (THREE
    emCCD frames + the two exposure api slots), not a single-emCCD window."""
    data = json.loads((REPO_ROOT / "pulses" / "imaging_template.json").read_text(encoding="utf-8"))
    emccd = data["channels"].index("emCCD")
    n_triggers = sum(1 for p in data["periods"] if p["states"][emccd])
    assert n_triggers == 3                                          # long-short-long, on disk
    names = [s["name"] for s in data.get("api_slots", [])]
    assert names.count("a1") == 2 and names.count("a2") == 1        # a1 = long frames, a2 = readout
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
        assert back.api_names() == ["a1", "a2"]
    finally:
        exp.close()
