"""GUI wiring for the finite scan-repeat count (#3) + the live scan-progress readout (#4).

These guard the THREE seams the GUI adds on top of the already-committed device API
(``PulseTableState.scan_repeats`` / ``sequencer.scan_progress()``):

* the pulse_gui Scan-tab "Scan repeats (0 = ∞)" spin writes ``state.scan_repeats`` and round-trips
  through ``read_state`` -> ``load_state`` (so it reaches the device via ``prepare(state)`` + Save);
* ``_PulseSlotsWidget.values_dict()`` (the task_console pulse-scan form) carries ``scan_repeats``;
* ``_format_scan_progress`` is the SINGLE source of the live label text (blanks on idle, no "/ R"
  for an infinite scan, "/ R" for a finite one).

Offscreen, no real window for the wiring asserts; the visual three-DPR screenshots are produced
separately (the user's acceptance gate), driving a real ``show_pulse_gui`` window.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def _app(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ---- (c) the progress-label formatter: single source, idle blanks, finite shows "/ R" ----------
def test_format_scan_progress_finite_shows_point_and_sweep_over_repeats():
    from Zou_lab_control.frontend.pulse_gui import _format_scan_progress
    # point 1 (0-based) of 3, sweep 0 (0-based), K=2 repeats -> 1-based "point 2 / 3 · sweep 1 / 2"
    text = _format_scan_progress(
        {"scanning": True, "point": 1, "n_points": 3, "sweep": 0, "n_repeats": 2})
    assert text == "Scan: point 2 / 3 · sweep 1 / 2"


def test_format_scan_progress_infinite_omits_repeat_total():
    from Zou_lab_control.frontend.pulse_gui import _format_scan_progress
    # n_repeats == 0 (∞) -> NO "/ R" tail, just "sweep r"
    text = _format_scan_progress(
        {"scanning": True, "point": 0, "n_points": 3, "sweep": 1, "n_repeats": 0})
    assert text == "Scan: point 1 / 3 · sweep 2"


def test_format_scan_progress_idle_blanks():
    from Zou_lab_control.frontend.pulse_gui import _format_scan_progress
    from Zou_lab_control.neutral_atom.devices.sequencer import SCAN_PROGRESS_IDLE
    assert _format_scan_progress(SCAN_PROGRESS_IDLE) == ""   # not scanning -> blank
    assert _format_scan_progress(None) == ""                # no sequencer reading -> blank
    # scanning but no points is also idle-blank (degenerate)
    assert _format_scan_progress(
        {"scanning": True, "point": 0, "n_points": 0, "sweep": 0, "n_repeats": 0}) == ""


# ---- (b) the task_console pulse-scan form carries scan_repeats through values_dict --------------
def _pulse_slots_widget():
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget
    w = _PulseSlotsWidget()
    w.rebuild(api_rows=[("a1", "duration", "1", "us", 5.0)],
              scan_rows=[("s0", "duration", "2", "ns", "probe")])
    return w


def test_pulse_slots_values_dict_carries_scan_repeats_default_zero(_app):
    w = _pulse_slots_widget()
    v = w.values_dict()
    assert "scan_repeats" in v
    assert v["scan_repeats"] == 0          # default = sweep forever


def test_pulse_slots_values_dict_reflects_typed_repeats(_app):
    w = _pulse_slots_widget()
    w._scan_repeats_spin.setValue(4)
    assert w.values_dict()["scan_repeats"] == 4


def test_pulse_slots_scan_repeats_round_trips_through_seed_value(_app):
    """A SAVED blob restores the whole-sweep count (the spin is persistent, set in seed_value)."""
    w = _pulse_slots_widget()
    w.seed_value({"api": {"a1": 7.5}, "scan_mode": "scan",
                  "scan_code": "scan_table = [[2000.0]]", "extra_delay": 0.0, "scan_repeats": 3})
    w.rebuild(api_rows=[("a1", "duration", "1", "us", 5.0)],
              scan_rows=[("s0", "duration", "2", "ns", "probe")])
    assert w.values_dict()["scan_repeats"] == 3


# ---- (a) the pulse_gui Scan-tab spin writes state.scan_repeats and round-trips ------------------
def _pulse_editor(sequencer=None):
    """A headless PulseSequenceEditor with a tiny scan state (the Scan tab + spin are built)."""
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
    st = PulseTableState(channels=["probe", "trig"])
    st.bind_field("duration", "0", unit="us")
    st.set_scan_table([[10.0], [20.0], [30.0]])
    return PulseSequenceEditor(st, sequencer=sequencer)


def test_scan_repeats_spin_writes_state_scan_repeats(_app):
    ed = _pulse_editor()
    ed.scan_repeats_spin.setValue(3)
    state = ed.read_state()
    assert int(state.scan_repeats) == 3       # the spin's value reaches the state read_state builds


def test_scan_repeats_round_trips_through_load_state(_app):
    """load_state(state) syncs the spin to the loaded count -> a subsequent read_state reproduces
    it (the Save round-trip path)."""
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
    ed = _pulse_editor()
    loaded = PulseTableState(channels=["probe", "trig"])
    loaded.bind_field("duration", "0", unit="us")
    loaded.set_scan_table([[10.0], [20.0], [30.0]])
    loaded.scan_repeats = 5
    ed.load_state(loaded)
    assert int(ed.scan_repeats_spin.value()) == 5
    assert int(ed.read_state().scan_repeats) == 5


def test_scan_repeats_default_is_zero_forever(_app):
    ed = _pulse_editor()
    assert int(ed.scan_repeats_spin.value()) == 0     # 0 = ∞ is the default
    assert int(ed.read_state().scan_repeats) == 0


# ---- the live progress poll is defensive (no sequencer / no method -> blank, never raises) ------
def test_poll_scan_progress_blank_without_sequencer(_app):
    ed = _pulse_editor()
    ed.sequencer = None
    ed._poll_scan_progress()                          # must not raise
    assert ed.scan_progress_label.text() == ""


def test_poll_scan_progress_reads_connected_sequencer(_app):
    """With a sequencer reporting a mid-sweep scan, the poll writes the real progress text."""
    ed = _pulse_editor()

    class _FakeSeq:
        def scan_progress(self):
            return {"scanning": True, "point": 1, "n_points": 3, "sweep": 0, "n_repeats": 2}

    ed.sequencer = _FakeSeq()
    ed._poll_scan_progress()
    assert ed.scan_progress_label.text() == "Scan: point 2 / 3 · sweep 1 / 2"


def test_poll_scan_progress_swallows_reader_errors(_app):
    """A sequencer whose scan_progress() raises (a device/network blip) must not crash the timer;
    the label just blanks."""
    ed = _pulse_editor()

    class _BoomSeq:
        def scan_progress(self):
            raise RuntimeError("link dropped")

    ed.sequencer = _BoomSeq()
    ed._poll_scan_progress()                          # must not raise
    assert ed.scan_progress_label.text() == ""
