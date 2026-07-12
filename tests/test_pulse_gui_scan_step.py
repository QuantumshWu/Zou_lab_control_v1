"""Y3: the pulse GUI Scan tab's debug stepping -- "◀ step" / "step ▶" walk the HELD scan point one
row through the scan table.

The contract under guard:

* ``_hold_at_point(index)`` is the SINGLE source of the hold mechanics (clamp into the table, freeze
  that point, reload the single-point loop via the PulseController seam, remember
  ``_held_scan_point = (index, row, table length)``); ``_stop_scan_to_current_point`` is a thin shell
  over it (live point -> hold).
* stepping while HELD moves the index by ±1 with a fresh single-point prepare+fire; stepping while
  the sweep is still RUNNING stops+holds like Stop ▸ hold point and steps FROM the live point in ONE
  reload; at the table ends the index CLAMPS (no wrap) and no redundant reload is sent.
* the hold is CLEARED by every fresh device upload (On Pulse / Prepare, via ``_prepare_to_device``)
  and by Stop Pulse (``safe_state``).
* with no sequencer / no scan table the buttons are harmless (the offscreen-safe message path).

Offscreen, logic asserts only -- the visual acceptance is the user's separate three-DPR gate.
"""

from __future__ import annotations
from Zou_lab_control.neutral_atom.ports import PortCatalog

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from conftest import pulse_editor_for_test  # noqa: E402


@pytest.fixture
def _app(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _StepSeq:
    """Recording sequencer with a settable live scan position: the seam the hold/step buttons drive
    (bind_pulse(...).on_pulse = stop -> prepare -> fire) plus the ``scan_progress()`` the live-point
    read uses."""

    clock_hz = 50_000_000.0

    def __init__(self, point: int = 1, n_points: int = 4):
        self.point = point
        self.n_points = n_points
        self.prepared = []
        self.fired = 0
        self.stopped = 0

    def scan_progress(self):
        return {"scanning": True, "point": self.point, "n_points": self.n_points,
                "sweep": 0, "n_repeats": 0}

    def prepare(self, payload):
        self.prepared.append(payload)
        return payload

    def fire(self):
        self.fired += 1

    def set_safe_state(self):
        self.stopped += 1


def _scan_editor(seq):
    """A headless PulseSequenceEditor with one scanned duration and a 4-point table.

    A duration scan slot is stored in NANOSECONDS (``bind_field`` pins duration slots to ns so the
    Scan tab, the card, and the compiler agree).  The rows must be whole device ticks: the editor
    binds to the sequencer's clock at construction and re-snaps the state to that grid
    (``state.snapped(device_step_ns)``), so a sub-tick value is not a legal scan point -- it would
    snap up and silently merge with its neighbour.  ``_StepSeq`` is a 50 MHz device (20 ns tick), so
    the table is multiples of 20 ns: point k's exposure = ``20 * (k + 1)`` ns."""
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
    st = PulseTableState(port_catalog=PortCatalog.from_channels(["probe", "trig"]))
    st.bind_field("duration", "0", unit="us", name="exposure")
    st.set_scan_table([[20.0], [40.0], [60.0], [80.0]])
    if seq is None:
        return PulseSequenceEditor(st)
    seq.port_catalog = st.port_catalog
    return pulse_editor_for_test(st, sequencer=seq)


def test_scan_tab_exposes_semantic_identity_and_demotes_compiler_token(_app):
    ed = _scan_editor(None)
    ed._refresh_scan_tab()
    assert "exposure (compiler s0)" in ed.scan_slots_label.text()
    assert ed.scan_table_view.toPlainText().splitlines()[0] == "exposure"


# ---- hold -> step ▶ / ◀ step: the index moves and the device gets a fresh single-point reload -----
def test_step_forward_from_hold_advances_and_reloads_single_point(_app):
    seq = _StepSeq(point=1)
    ed = _scan_editor(seq)
    ed._stop_scan_to_current_point()                     # Stop ▸ hold point at the live point (1)
    assert ed._held_scan_point[0] == 1 and ed._held_scan_point[2] == 4
    n_prep, n_fire = len(seq.prepared), seq.fired

    ed.scan_step_forward_button.click()                  # the REAL button wiring
    assert ed._held_scan_point[0] == 2                   # index advanced by one row
    assert ed._held_scan_point[1] == [60.0]              # ...and remembers point 2's raw row (20*3 ns)
    assert len(seq.prepared) == n_prep + 1 and seq.fired == n_fire + 1   # a fresh prepare+fire
    payload = seq.prepared[-1]
    assert not payload.scan_table and not payload.scan_slots   # the reload is a STATIC single point
    # two different held points upload DIFFERENT baked programs (not the nominal twice)
    assert float(payload.periods[0].duration) != float(seq.prepared[n_prep - 1].periods[0].duration)


def test_step_back_from_hold_retreats_one_row(_app):
    seq = _StepSeq(point=2)
    ed = _scan_editor(seq)
    ed._stop_scan_to_current_point()                     # held at 2
    ed.scan_step_back_button.click()
    assert ed._held_scan_point[0] == 1 and ed._held_scan_point[1] == [40.0]  # point 1 = 20*2 ns


# ---- clamping at the table ends: no wrap-around AND no redundant reload --------------------------
def test_step_back_clamps_at_point_zero_without_reuploading(_app):
    seq = _StepSeq(point=0)
    ed = _scan_editor(seq)
    ed._stop_scan_to_current_point()                     # held at 0 (the first point)
    n_prep, n_fire = len(seq.prepared), seq.fired
    ed.scan_step_back_button.click()
    assert ed._held_scan_point[0] == 0                   # clamped, NOT wrapped to the last point
    assert len(seq.prepared) == n_prep and seq.fired == n_fire   # no redundant reload of the same point


def test_step_forward_clamps_at_last_point_without_reuploading(_app):
    seq = _StepSeq(point=3)
    ed = _scan_editor(seq)
    ed._stop_scan_to_current_point()                     # held at 3 (the last point)
    n_prep = len(seq.prepared)
    ed.scan_step_forward_button.click()
    assert ed._held_scan_point[0] == 3                   # clamped at the end
    assert len(seq.prepared) == n_prep


# ---- stepping while the sweep is still RUNNING: stop+hold like Stop ▸ hold, in ONE reload --------
def test_step_while_scanning_stops_and_holds_at_stepped_point_in_one_reload(_app):
    seq = _StepSeq(point=1)
    ed = _scan_editor(seq)
    assert ed._held_scan_point is None                   # not held -- the sweep is "running"
    ed.scan_step_forward_button.click()
    assert ed._held_scan_point[0] == 2                   # live point (1) + 1, held directly
    assert len(seq.prepared) == 1 and seq.fired == 1     # ONE reload, not hold-then-step twice
    assert seq.stopped >= 1                              # the sweep was stopped first (On Pulse seam)


def test_progress_label_shows_the_held_point_over_total_after_a_step(_app):
    """The Scan-tab label (single source: _held_scan_point_text) reads the held "point k/N" + values."""
    seq = _StepSeq(point=1)
    ed = _scan_editor(seq)
    ed.show()                                            # the poll only runs while the window is VISIBLE
    ed.tabs.setCurrentWidget(ed.scan_tab)
    ed.scan_step_forward_button.click()                  # running -> stop+hold at point 2 (0-based)
    ed._poll_scan_progress()
    assert ed.scan_progress_label.text() == "held at point 3/4: exposure = 60"


# ---- harmless without a sequencer / without a scan table -----------------------------------------
def test_step_without_scan_table_is_a_harmless_noop(_app):
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
    seq = _StepSeq()
    state = PulseTableState(port_catalog=PortCatalog.from_channels(["probe", "trig"]))
    seq.port_catalog = state.port_catalog
    ed = pulse_editor_for_test(state, sequencer=seq)   # no scan
    ed.scan_step_forward_button.click()
    ed.scan_step_back_button.click()
    assert ed._held_scan_point is None                   # nothing held
    assert seq.prepared == [] and seq.fired == 0         # nothing uploaded


def test_step_without_sequencer_never_raises(_app):
    ed = _scan_editor(None)
    ed.scan_step_forward_button.click()                  # must not raise (offscreen-safe message path)
    ed.scan_step_back_button.click()
    assert ed._held_scan_point is None


# ---- the hold is CLEARED by On Pulse / Stop Pulse / Prepare (a fresh upload supersedes it) -------
def test_hold_cleared_by_on_pulse_stop_pulse_and_prepare(_app):
    seq = _StepSeq(point=1)
    ed = _scan_editor(seq)

    ed._stop_scan_to_current_point()
    assert ed._held_scan_point is not None
    ed.fire()                                            # On Pulse resumes the full sweep
    assert ed._held_scan_point is None

    ed._stop_scan_to_current_point()
    assert ed._held_scan_point is not None
    ed.safe_state()                                      # Stop Pulse ends the held loop
    assert ed._held_scan_point is None

    ed._stop_scan_to_current_point()
    assert ed._held_scan_point is not None
    ed.prepare()                                         # Prepare uploads a fresh program
    assert ed._held_scan_point is None
