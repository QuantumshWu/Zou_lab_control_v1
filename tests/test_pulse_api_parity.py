"""Contract: the pulse API can do what the pulse GUI does -- programmatically.

A measurement / notebook must be able to LOAD a saved pulse program and retune it
(durations, a channel delay, append a step) WITHOUT the GUI, so the same program a
user designs in the editor can be tweaked in code (e.g. a Calibrate task sweeping the
readout duration).  These pin the programmatic mutators that mirror the GUI edits and
prove the edits survive ``to_sequence`` + a save/load round-trip.
"""

from __future__ import annotations
from Zou_lab_control.neutral_atom.ports import PortCatalog

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.timing import PulsePeriod, PulseTableState

CHANNELS = ["trap", "cooling", "probe", "emCCD"]

# A realistic neutral-atom program drives an analog DAC bus (member bits grouped under
# one ``da_*`` label); the GUI edits run on exactly this shape, so the API mutators must
# survive it too -- not just bus-free TTL channels.
DAC_CHANNELS = ["ch00", "ch01", "ch02", "ch03"]
DAC_LABELS = {"ch00": "da_test[0]", "ch01": "da_test[1]", "ch02": "da_test[2]", "ch03": "trig"}


def _dac_bus_state() -> PulseTableState:
    st = PulseTableState(
        port_catalog=PortCatalog.from_channels(DAC_CHANNELS, channel_labels=DAC_LABELS),
        visible_ports=["da_test", "ch03"],
        time_step_ns=20,
        periods=[PulsePeriod(100, (0, 0, 0, 0), unit="ns") for _ in range(3)],
    )
    st.set_analog_bus_mode(0, "da_test", "edge", value=0)   # an EXPLICIT per-period mode plan
    st.set_analog_bus_mode(2, "da_test", "ramp", value=3)
    return st


def test_pulse_state_mutators_match_gui_edits():
    """set_period_duration / set_period_name / set_channel_delay / add_period mirror the
    GUI's per-period + per-channel edits, and they compile + round-trip through dict."""
    st = PulseTableState(port_catalog=PortCatalog.from_channels(CHANNELS))
    n0 = len(st.periods)

    st.set_period_duration(0, 1234.0)
    st.set_period_name(0, "load")
    st.set_channel_delay("probe", 50.0, unit="ns")
    st.add_period(200.0, name="readout", states={"probe": 1, "emCCD": 1})

    assert len(st.periods) == n0 + 1
    assert float(st.periods[0].duration) == 1234.0 and st.periods[0].name == "load"
    assert st.delays["probe"] == 50.0
    # the appended period carries the requested per-channel states
    readout = st.periods[-1]
    assert readout.name == "readout"
    assert readout.states[st.channel_index("probe")] == 1
    assert readout.states[st.channel_index("trap")] == 0

    # compiles to a sequence (no GUI needed) + survives a JSON round-trip
    st.to_sequence()
    st2 = PulseTableState.from_dict(json.loads(json.dumps(st.to_dict())))
    assert float(st2.periods[0].duration) == 1234.0
    assert st2.delays["probe"] == 50.0
    assert st2.periods[-1].name == "readout"


def test_measurement_can_load_and_retune_a_saved_program(tmp_path):
    """The measurement use-case: a program saved from the pulse GUI is LOADED and its
    readout duration RETUNED in code, and the change reaches the compiled sequence."""
    path = tmp_path / "readout.json"
    PulseTableState(port_catalog=PortCatalog.from_channels(CHANNELS)).set_period_name(0, "readout").save(path)

    loaded = PulseTableState.load(path)
    loaded.set_period_duration(0, 5000.0)              # retune the readout duration
    assert float(loaded.periods[0].duration) == 5000.0
    # the compiled sequence reflects the retuned duration (same path the device takes)
    total = loaded.total_duration_ns()
    assert total >= 5000.0


def test_pulse_state_mutators_validate_inputs():
    """The mutators fail loud on a bad index / unknown channel (no silent corruption)."""
    st = PulseTableState(port_catalog=PortCatalog.from_channels(CHANNELS))
    with pytest.raises(ValueError):
        st.set_period_duration(999, 1.0)
    with pytest.raises(ValueError):
        st.set_channel_delay("nope", 1.0)
    with pytest.raises(ValueError):
        st.add_period(1.0, states=(1, 0))             # wrong length per-channel vector


def test_add_period_keeps_analog_bus_modes_in_step():
    """add_period on a program that drives a DAC bus must grow the per-period bus-mode
    plan in step -- else validate() rejects the modes/periods length mismatch (this is
    the realistic neutral-atom case; the earlier bus-free test could not catch it)."""
    st = _dac_bus_state()
    n0 = len(st.periods)
    assert len(st.analog_bus_modes["da_test"]) == n0   # explicit plan == period count

    st.add_period(100.0, name="extra", states={"ch03": 1})

    assert len(st.periods) == n0 + 1
    # the bus-mode plan grew with it (the appended period defaults to a HOLD), so the
    # modes stay aligned with periods and the program still compiles + round-trips
    modes = st.analog_bus_modes["da_test"]
    assert len(modes) == n0 + 1
    assert modes[-1] == {"mode": "hold", "value": None}
    st.compile(clock_hz=50_000_000, repeat_forever=False)
    st2 = PulseTableState.from_dict(json.loads(json.dumps(st.to_dict())))
    assert len(st2.analog_bus_modes["da_test"]) == n0 + 1


def test_set_period_duration_refuses_scan_bound_period():
    """A period whose duration is SCAN-BOUND carries an ``sN`` expression a scan slot
    targets; set_period_duration must fail loud (don't orphan the slot) -- unbind first."""
    st = PulseTableState(port_catalog=PortCatalog.from_channels(CHANNELS))
    st.bind_field("duration", "0")                     # bind period 0's duration to a slot
    with pytest.raises(ValueError):
        st.set_period_duration(0, 1234.0)
    st.unbind_slot(0)                                  # after unbinding it is editable again
    st.set_period_duration(0, 1234.0)
    assert float(st.periods[0].duration) == 1234.0
