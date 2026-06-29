"""#1: the pulse GUI's "Stop ▸ hold point" reloads a SINGLE-point program built from exactly the scan
point the device is on, looped forever -- so the experiment parks on the current setpoint.  The freeze
is a pure, testable transform (PulseSequenceEditor._freeze_state_to_scan_point); the full Stop handler
re-prepares + fires that frozen state through the ordinary sequencer path."""

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

import Zou_lab_control.neutral_atom as na  # noqa: E402
from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor  # noqa: E402


def _scan_state():
    return na.PulseTableState(
        channels=["ch00", "ch01", "t"],
        channel_labels={"ch00": "da[0]", "ch01": "da[1]"},
        visible_channels=["ch00", "ch01", "t"], time_step_ns=20.0,
        periods=[na.PulsePeriod(200, (0, 0, 1), unit="ns")],
        scan_slots=[{"kind": "duration", "target": "0", "unit": "ns", "nominal": 200.0}],
        scan_table=[[200.0], [400.0], [600.0]], scan_repeats=0)


def test_freeze_bakes_the_held_point_into_a_static_pulse():
    state = _scan_state()
    frozen = PulseSequenceEditor._freeze_state_to_scan_point(state, 1)   # the 400 ns point
    assert frozen.scan_table == [] and frozen.scan_slots == []           # the scan is CLEARED -> static
    assert int(frozen.scan_repeats) == 0                                 # looped forever (hold)
    assert float(frozen.periods[0].duration) == 400.0                    # held value BAKED into period 0
    # the original state is NOT mutated (the user can resume the full sweep)
    assert state.scan_table == [[200.0], [400.0], [600.0]]


def test_freeze_clamps_an_out_of_range_index_to_the_last_point():
    state = _scan_state()
    assert float(PulseSequenceEditor._freeze_state_to_scan_point(state, 99).periods[0].duration) == 600.0
    assert float(PulseSequenceEditor._freeze_state_to_scan_point(state, -5).periods[0].duration) == 200.0


def test_held_point_drives_the_compiled_frame_not_the_nominal():
    """The held point's VALUE actually drives the run: holding point 0 (200 ns) vs point 2 (600 ns)
    compiles to DIFFERENT frame lengths (10 vs 30 ticks at 20 ns), and neither is a sweep -- proving
    the freeze bakes the point's value, not the slot's nominal (a 1-row scan_table would degrade to it)."""
    state = _scan_state()
    p0 = PulseSequenceEditor._freeze_state_to_scan_point(state, 0).compile(clock_hz=50_000_000)
    p2 = PulseSequenceEditor._freeze_state_to_scan_point(state, 2).compile(clock_hz=50_000_000)
    assert p0.ticks[-1] == 10 and p2.ticks[-1] == 30
    assert not (getattr(p2, "scan_points", None) or [])                  # static -- no sweep


def test_stop_handler_reads_the_live_point_freezes_and_re_fires_it():
    """The whole Stop handler: read the device's CURRENT scan point, freeze the state to it, re-prepare
    + fire that static program, and mark the held point for the readout -- against a recording sequencer."""
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor as Editor, PulseStateUIManager
    ed = Editor.__new__(Editor)                       # bypass the heavy Qt __init__; drive the handler directly
    ed.read_state = _scan_state                       # the UI would return this fresh scan state
    ed.state = _scan_state()
    captured = {}

    class _Seq:
        def scan_progress(self):
            return {"point": 2, "n_points": 3}        # the device is on point 2 (the 600 ns setpoint)
        def prepare(self, st):
            captured["state"] = st
        def fire(self):
            captured["fired"] = True

    ed.sequencer = _Seq()
    ed._message = lambda *a, **k: None

    class _SM:
        runstate = None
    ed.stateui_manager = _SM()

    ed._stop_scan_to_current_point()
    assert captured.get("fired") is True                                  # it re-fired the held pulse
    held_state = captured["state"]
    assert held_state.scan_table == [] and float(held_state.periods[0].duration) == 600.0   # point 2 baked, static
    assert ed._held_scan_point[0] == 2 and ed._held_scan_point[2] == 3     # readout knows the held point
    assert ed.stateui_manager.runstate == PulseStateUIManager.RunState.RUNNING
