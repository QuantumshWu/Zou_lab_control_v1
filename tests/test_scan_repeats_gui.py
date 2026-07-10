"""GUI wiring for the finite scan-repeat count (#3) + the live scan-progress readout (#4).

These guard the THREE seams the GUI adds on top of the already-committed device API
(``PulseTableState.scan_repeats`` / ``sequencer.scan_progress()``):

* the pulse_gui Scan-tab "Scan repeats (0 = ∞)" spin writes ``state.scan_repeats`` and round-trips
  through ``read_state`` -> ``load_state`` (so it reaches the device via ``prepare(state)`` + Save);
* ``_format_scan_progress`` is the SINGLE source of the live label text (blanks on idle, no "/ R"
  for an infinite scan, "/ R" for a finite one).

Offscreen, no real window for the wiring asserts; the visual three-DPR screenshots are produced
separately (the user's acceptance gate), driving a real ``show_pulse_gui`` window.
"""

from __future__ import annotations
from Zou_lab_control.neutral_atom.ports import PortCatalog

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


# ---- (a) the pulse_gui Scan-tab spin writes state.scan_repeats and round-trips ------------------
def _pulse_editor(sequencer=None):
    """A headless PulseSequenceEditor with a tiny scan state (the Scan tab + spin are built)."""
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
    st = PulseTableState(port_catalog=PortCatalog.from_channels(["probe", "trig"]))
    st.bind_field("duration", "0", unit="us", name="exposure")
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
    loaded = PulseTableState(port_catalog=PortCatalog.from_channels(["probe", "trig"]))
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
    ed.tabs.setCurrentWidget(ed.scan_tab)             # Scan tab visible -> the poll actually runs
    ed.sequencer = None
    ed._poll_scan_progress()                          # must not raise
    assert ed.scan_progress_label.text() == ""


def test_poll_scan_progress_reads_connected_sequencer(_app):
    """With a sequencer reporting a mid-sweep scan, the poll writes the real progress text."""
    ed = _pulse_editor()
    ed.show()                                         # the poll only runs while the window is VISIBLE
    ed.tabs.setCurrentWidget(ed.scan_tab)             # Scan tab visible -> the poll actually runs

    class _FakeSeq:
        def scan_progress(self):
            return {"scanning": True, "point": 1, "n_points": 3, "sweep": 0, "n_repeats": 2}

    ed.sequencer = _FakeSeq()
    ed._poll_scan_progress()
    # the live position PLUS the current point's values (#1 "show the current scan points"): the
    # editor's scan_table is [[10],[20],[30]], so point 1 appends its semantic identity.
    assert ed.scan_progress_label.text() == "Scan: point 2 / 3 · sweep 1 / 2  exposure = 20"


def test_virtual_sequencer_reports_scan_progress_for_a_software_streamed_scan(_app):
    """virtual==real for the readout: the pure-software VirtualSequencer (the standalone pulse GUI's
    "Virtual (sim)" backend, no FPGA cursor) must STILL report "point K / N" for a fired streamed
    scan -- so the Scan tab shows the live position on virtual, not just on real hardware.  The
    reading is REAL-TIME PACED off the program's own per-point duration (sleep_scale=1), the same
    pacing the live camera follows -- never a display-rate fabrication."""
    import time
    from Zou_lab_control.neutral_atom.devices.sequencer import SCAN_PROGRESS_IDLE
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
    st = PulseTableState(port_catalog=PortCatalog.from_channels(["probe", "trig"]))
    st.bind_field("duration", "0", unit="us")
    st.set_scan_table([[10.0], [20.0], [30.0], [40.0]])   # 4 points, scan_repeats=0 -> repeat_forever
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=1.0)   # real-time (conftest fast-forwards by default)
    assert seq.scan_progress() == SCAN_PROGRESS_IDLE       # nothing fired yet -> idle
    seq.prepare(st)
    seq.fire()
    p = seq.scan_progress()
    assert p["scanning"] is True and p["n_points"] == 4    # a streamed scan IS reported (not idle)
    total0 = p["sweep"] * p["n_points"] + p["point"]
    time.sleep(0.12)                                       # >> the us-scale per-point duration -> the point advances
    p1 = seq.scan_progress()
    assert p1["sweep"] * p1["n_points"] + p1["point"] >= total0
    seq.set_safe_state()                                   # Stop Pulse -> back to idle
    assert seq.scan_progress() == SCAN_PROGRESS_IDLE


def test_poll_scan_progress_skips_rpc_when_scan_tab_hidden(_app):
    """The poll must NOT call scan_progress() (a remote-sequencer RPC) when the Scan tab is not the
    current view -- the label lives there, so a background tab must not keep hitting the network."""
    ed = _pulse_editor()
    ed.tabs.setCurrentWidget(ed.edit_tab)             # NOT the Scan tab

    class _CountingSeq:
        calls = 0
        def scan_progress(self):
            type(self).calls += 1
            return {"scanning": True, "point": 0, "n_points": 3, "sweep": 0, "n_repeats": 0}

    ed.sequencer = _CountingSeq()
    ed._poll_scan_progress()
    assert _CountingSeq.calls == 0                    # gated: no RPC while the Scan tab is hidden


def test_poll_scan_progress_swallows_reader_errors(_app):
    """A sequencer whose scan_progress() raises (a device/network blip) must not crash the timer;
    the label just blanks."""
    ed = _pulse_editor()
    ed.tabs.setCurrentWidget(ed.scan_tab)             # Scan tab visible -> the poll actually calls in

    class _BoomSeq:
        def scan_progress(self):
            raise RuntimeError("link dropped")

    ed.sequencer = _BoomSeq()
    ed._poll_scan_progress()                          # must not raise
    assert ed.scan_progress_label.text() == ""


# ---- (4) On Pulse routes through the SAME PulseController seam as notebook + real, and is CYCLIC ----
class _RecordingSeq:
    """Records the payload prepared + whether fire()/set_safe_state() ran -- the device seam On Pulse
    must drive (On Pulse = stop -> prepare -> fire)."""
    clock_hz = 50_000_000.0

    def __init__(self):
        self.prepared = []
        self.fired = 0
        self.stopped = 0

    def prepare(self, payload):
        self.prepared.append(payload)
        return payload

    def fire(self):
        self.fired += 1

    def set_safe_state(self):
        self.stopped += 1


def test_on_pulse_routes_through_pulse_controller_and_is_cyclic(_app):
    """The GUI On Pulse must go through bind_pulse(...).on_pulse (GUI == notebook == real one seam) and
    always ask for repeat_forever=True (On Pulse = continuous until Stop).  Even a state whose own
    repeat_forever is False fires a CYCLIC program -- so a scan with scan_repeats=0 sweeps forever and
    K>0 stops after K, exactly the user's intent (no reliance on any global data-carrier invariant)."""
    ed = _pulse_editor()
    ed.state.repeat_forever = False                   # a loaded single-shot state -- On Pulse overrides
    seq = _RecordingSeq()
    ed.sequencer = seq
    ed.fire()                                         # the real On Pulse handler
    assert seq.fired == 1                             # it prepared AND fired through the controller seam
    payload = seq.prepared[-1]
    assert bool(payload.repeat_forever) is True       # On Pulse is cyclic regardless of the state flag
    assert int(payload.scan_repeats) == 0             # scan_repeats (0=∞) is what stops a scan, not this


def test_prepare_button_also_routes_cyclic_through_controller(_app):
    """The Prepare button shares the seam: it prepares (no fire) a cyclic program through the controller
    -- so a subsequent run streams, and Prepare never leaves a contradictory one-shot scan program."""
    ed = _pulse_editor()
    ed.state.repeat_forever = False
    seq = _RecordingSeq()
    ed.sequencer = seq
    ed.prepare()                                      # the real Prepare handler
    assert seq.fired == 0                             # Prepare does NOT fire
    assert len(seq.prepared) == 1 and bool(seq.prepared[-1].repeat_forever) is True


# ---- On Pulse is ORDER-INDEPENDENT: stop -> re-read current UI -> re-upload -> fire ----------------
# The user report: change a scan slot's binding, press On Pulse -- no effect unless you first Stop and
# re-run the scan points in a specific order.  On Pulse must instead ALWAYS stop, re-derive the scan
# table for the slots as they are NOW, upload it, and fire -- so a slot move then On Pulse just works.
def _three_period_scan_editor(seq):
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod
    st = PulseTableState(port_catalog=PortCatalog.from_channels(["probe", "trig"]))
    st.periods.append(PulsePeriod(1000, tuple(0 for _ in st.port_catalog.raw_lanes), unit="ns"))
    st.periods.append(PulsePeriod(1000, tuple(0 for _ in st.port_catalog.raw_lanes), unit="ns"))
    st.bind_field("duration", "0", unit="us")
    st.set_scan_table([[10.0], [20.0], [30.0]])
    ed = PulseSequenceEditor(st, sequencer=seq)
    ed._scan_tables["generated"] = [[10.0], [20.0], [30.0]]   # the active generated source
    return ed


def test_on_pulse_stops_before_it_fires(_app):
    """On Pulse = off_pulse -> prepare -> fire: it drives the stop seam FIRST every time, so switching
    a running scan is just another On Pulse (no reliance on the user having stopped first)."""
    seq = _RecordingSeq()
    ed = _three_period_scan_editor(seq)
    ed.fire()
    assert seq.stopped == 1 and seq.fired == 1        # stopped THEN fired (one On Pulse)
    ed.fire()
    assert seq.stopped == 2 and seq.fired == 2        # a second On Pulse stops again first


def test_on_pulse_after_moving_a_scan_slot_uploads_the_current_ui_table(_app):
    """Move the scan slot to a different period, then On Pulse directly (no manual Stop, no re-run of
    the scan points): the uploaded program must target the MOVED slot and carry the real, DISTINCT
    scan points -- not a stale/degenerate table left from the old binding."""
    seq = _RecordingSeq()
    ed = _three_period_scan_editor(seq)

    # move the slot: unbind period-0's duration, bind period-2's -- the exact "change the scan slot's
    # position" the user does in the Edit tab (here via the state the read_state/load_state cycle uses).
    st = ed.read_state(); st.unbind_slot(0); ed.state = st; ed.load_state(st)
    st2 = ed.read_state(); st2.bind_field("duration", "2", unit="us"); ed.state = st2; ed.load_state(st2)

    ed.fire()                                         # On Pulse directly, no manual re-run
    payload = seq.prepared[-1]
    assert [(s.kind, s.target) for s in payload.scan_slots] == [("duration", "2")]   # the MOVED slot
    # the real distinct sweep survived the move (not [[nominal],[nominal],[nominal]])
    assert len({tuple(row) for row in payload.scan_table}) > 1
    assert seq.stopped >= 1                           # it stopped before re-uploading + firing


# ---- Scan-tab Run '*' dirty tracks scan-slot changes: a stale generated table must FLAG a re-Run -----
# The design defect: after Run generates a scan table, binding / unbinding / moving a scan slot makes the
# generated table stale (wrong column count / order), yet the Run button showed no '*', so the user
# thought the generated points were current.  Run must go dirty on any scan-slot change once a table has
# been generated, and clear again on a fresh Run.
def _scan_slot_editor():
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod
    st = PulseTableState(port_catalog=PortCatalog.from_channels(["probe", "trig"]))
    st.periods.append(PulsePeriod(1000, tuple(0 for _ in st.port_catalog.raw_lanes), unit="ns"))
    st.periods.append(PulsePeriod(1000, tuple(0 for _ in st.port_catalog.raw_lanes), unit="ns"))
    st.bind_field("duration", "0", unit="us")         # one scan slot
    ed = PulseSequenceEditor(st)
    ed.tabs.setCurrentWidget(ed.scan_tab)
    ed._refresh_scan_tab()                            # populate the default column_stack scan code
    return ed


def test_scan_run_goes_dirty_when_a_scan_slot_changes_after_run(_app):
    """After a successful Run, binding a SECOND scan slot (the generated table is now one column short)
    must mark the Run button dirty ('*'), and a fresh Run must clear it.  This is the defect: the stale
    generated table was silently kept as if current."""
    ed = _scan_slot_editor()

    ed._run_scan_code()
    assert ed.scan_run_button.is_dirty() is False, "a fresh Run clears the dirty star"
    generated = ed._scan_tables["generated"]
    assert generated and len(generated[0]) == 1, "the generated table has one column for one slot"

    # bind a SECOND scan slot via the real dot-click path -> 1 -> 2 slots, the generated table is stale
    ed._cycle_field_slot(ed.read_state(), "duration", "1", unit="us")
    assert len(ed.read_state().scan_slots) == 2, "a second scan slot is now bound"
    assert len(ed._scan_tables["generated"][0]) == 1, "the generated table is STALE (still one column)"
    assert ed.scan_run_button.is_dirty() is True, \
        "the stale generated table flags Run dirty so the user knows to re-Run"

    # a fresh Run regenerates for the new slot count and clears the star
    ed._run_scan_code()
    assert len(ed._scan_tables["generated"][0]) == 2, "re-Run generates a two-column table"
    assert ed.scan_run_button.is_dirty() is False, "the re-Run clears the dirty star"


def test_scan_run_redirties_when_a_scan_slot_is_unbound(_app):
    """Unbinding a scan slot (scan -> API on the dot) also makes the generated table stale -> Run dirty."""
    ed = _scan_slot_editor()
    ed._run_scan_code()
    assert ed.scan_run_button.is_dirty() is False
    ed._cycle_field_slot(ed.read_state(), "duration", "0", unit="us")   # scan -> API (unbinds the slot)
    assert len(ed.read_state().scan_slots) == 0
    assert ed.scan_run_button.is_dirty() is True, "unbinding the scan slot flags Run dirty"


def test_scan_slot_change_before_any_run_does_not_spuriously_dirty(_app):
    """Before the FIRST Run there is no generated table to be stale, so a scan-slot change must NOT set
    the dirty star via the slot-signature path (the code editor's own edit-dirty governs that phase)."""
    ed = _scan_slot_editor()
    ed.scan_run_button.set_dirty(False)
    assert ed._scan_generated_slots is None, "no table generated yet"
    ed._cycle_field_slot(ed.read_state(), "duration", "1", unit="us")
    assert ed.scan_run_button.is_dirty() is False, \
        "no stale generated table before the first Run -> no spurious dirty from the slot change"
