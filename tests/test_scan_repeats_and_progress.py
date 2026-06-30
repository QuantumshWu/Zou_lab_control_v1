"""#H3t-3/4: finite scan-repeat count + a live scan-progress reading.

A streamed scan used to sweep its points then repeat FOREVER.  ``scan_repeats=K`` makes it play
K whole sweeps then stop; ``scan_progress()`` reports where the scan is now (point K / N, sweep r).
The point/sweep math is single-sourced in ``scan_progress_fields`` and the VIRTUAL backend mirrors
the real streamer (same method + semantics), so the GUI poll works with no real hardware.
"""

import pytest

from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
from Zou_lab_control.neutral_atom.devices.sequencer import (
    PulseController, RuntimeSequenceProgram, SCAN_PROGRESS_IDLE, scan_progress_fields,
)
from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession


def _scan_state() -> PulseTableState:
    """A minimal 3-point, 1-slot hardware scan (period-0 duration bound as s0)."""
    st = PulseTableState(channels=["probe", "trig"])
    st.bind_field("duration", "0", unit="us")
    st.set_scan_table([[10.0], [20.0], [30.0]])
    return st


# ---- the pure point/sweep math (single source for virtual + real) ---------------------------
def test_scan_progress_fields_infinite_advances_and_wraps():
    assert scan_progress_fields(0, 3, 0) == {"scanning": True, "point": 0, "n_points": 3, "sweep": 0, "n_repeats": 0}
    assert scan_progress_fields(2, 3, 0) == {"scanning": True, "point": 2, "n_points": 3, "sweep": 0, "n_repeats": 0}
    # wraps into sweep 1, never stops (n_repeats == 0)
    assert scan_progress_fields(4, 3, 0) == {"scanning": True, "point": 1, "n_points": 3, "sweep": 1, "n_repeats": 0}


def test_scan_progress_fields_finite_saturates_and_stops():
    # K=2 sweeps of 3 points: mid-run is scanning, K*N played -> stopped at the last point.
    assert scan_progress_fields(3, 3, 2)["sweep"] == 1 and scan_progress_fields(3, 3, 2)["scanning"] is True
    done = scan_progress_fields(6, 3, 2)
    assert done == {"scanning": False, "point": 2, "n_points": 3, "sweep": 1, "n_repeats": 2}


def test_scan_progress_fields_no_points_is_idle():
    assert scan_progress_fields(0, 0, 0) == SCAN_PROGRESS_IDLE


# ---- the carrier round-trips the count ------------------------------------------------------
def test_pulse_table_state_round_trips_scan_repeats():
    st = _scan_state()
    st.scan_repeats = 5
    assert PulseTableState.from_dict(st.to_dict()).scan_repeats == 5


def test_scan_compile_carries_scan_repeats_to_program():
    # The REAL hardware path compiles the scan program (the streamer reads scan_points + the host
    # reads scan_repeats off the program); the scan compiler copies the state's count through.
    st = _scan_state()
    st.scan_repeats = 3
    program = st.compile_scan(clock_hz=50_000_000.0)
    assert program.scan_repeats == 3
    assert RuntimeSequenceProgram.from_dict(program.to_dict()).scan_repeats == 3


def test_streamed_scan_compiles_repeat_forever_even_without_the_flag():
    # The pulse GUI's On Pulse sets only scan_repeats and prepares the STATE directly (it never sets
    # repeat_forever, and bypasses PulseController).  A streamed scan must still STREAM: the compiled
    # program forces repeat_forever=True whenever scan_points is set (the single-source __post_init__
    # invariant), so K=0 sweeps forever and K>0 stops after K -- GUI, notebook and real alike.  This
    # is the "finite scan plays only one sweep / infinite scan never starts" bug, fixed at the gate.
    finite = _scan_state(); finite.scan_repeats = 3           # finite, repeat_forever NOT set
    assert finite.compile_scan(clock_hz=50_000_000.0).repeat_forever is True
    assert _scan_state().compile_scan(clock_hz=50_000_000.0).repeat_forever is True   # K=0 too


def test_gui_path_scan_streams_without_explicit_repeat_forever():
    # Reproduce the GUI On Pulse end to end on the virtual backend: prepare the scan STATE directly
    # (no PulseController, no repeat_forever flag) and fire -- it must actually STREAM (firing set,
    # scan_progress scanning), instead of a single-sweep one-shot.  virtual == real.
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    seq.prepare(_scan_state())                               # K=0 infinite scan; flag NOT set
    seq.fire()
    assert seq.firing is not None
    assert seq.scan_progress()["scanning"] is True


def test_gui_path_finite_scan_reaches_done_without_the_flag():
    # The finite case of the same GUI path: K=2 whole sweeps then stop (the saturated done reading),
    # with no explicit repeat_forever flag -- the exact user report "scan repeats > 0 plays only once".
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    st = _scan_state(); st.scan_repeats = 2                  # finite, flag NOT set
    seq.prepare(st)
    seq.fire()
    assert seq.scan_progress() == {"scanning": False, "point": 2, "n_points": 3, "sweep": 1, "n_repeats": 2}


def test_controller_payload_threads_scan_repeats_into_state():
    # The on_pulse/prepare API writes scan_repeats into the fired pulse state (the seam the scan
    # compiler then reads), so a notebook `pulse.on_pulse(scan_repeats=K)` sets it by value.
    payload = PulseController(VirtualSequencer(channels=["probe", "trig"]), _scan_state()).payload(scan_repeats=4)
    assert payload.scan_repeats == 4


# ---- virtual backend mirrors the real one (virtual==real) -----------------------------------
def test_virtual_finite_scan_reports_done():
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)  # sleep_scale=0 fast-forwards
    PulseController(seq, _scan_state()).on_pulse(repeat_forever=True, scan_repeats=2)
    progress = seq.scan_progress()        # dt scaled to 0 -> a finite scan is instantly done
    assert progress == {"scanning": False, "point": 2, "n_points": 3, "sweep": 1, "n_repeats": 2}


def test_virtual_finite_scan_wait_done_returns_true():
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    PulseController(seq, _scan_state()).on_pulse(repeat_forever=True, scan_repeats=2)
    assert seq.wait_done(timeout=10.0) is True     # a finite scan DOES finish


def test_virtual_infinite_scan_keeps_scanning():
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    PulseController(seq, _scan_state()).on_pulse(repeat_forever=True, scan_repeats=0)
    progress = seq.scan_progress()
    assert progress["scanning"] is True and progress["n_repeats"] == 0 and progress["n_points"] == 3
    assert seq.wait_done(timeout=0.05) is False     # an infinite scan never finishes


def test_virtual_idle_when_not_scanning():
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    assert seq.scan_progress() == SCAN_PROGRESS_IDLE


# ---- both backends expose the SAME scan-progress contract -----------------------------------
def test_both_backends_expose_scan_progress():
    for cls in (VirtualSequencer, VivadoAxiStreamerSession):
        assert hasattr(cls, "scan_progress"), f"{cls.__name__} must expose scan_progress()"


# ---- VirtualSequencer COMPOSES the single SequencerService state machine (no second copy) ----
def test_virtual_sequencer_composes_one_service_state_machine():
    """The virtual backend must DELEGATE prepare/fire/source-recording to a composed
    SequencerService -- the one shared state machine -- not keep a second copy that can drift
    (finding #4: the duplicated prepare/_record_source_payload/scan_progress had already diverged).
    Pin the structure so a future hand-rolled copy fails here."""
    from Zou_lab_control.neutral_atom.devices.sequencer import SequencerService
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    # composition, not a hand-rolled copy
    assert isinstance(seq.service, SequencerService)
    # the source-payload recorder lives ONLY on the service now (no virtual override to drift)
    assert "_record_source_payload" not in vars(type(seq))
    # the sync handle + history are the service's, read through, not a second store
    assert seq.last_payload_json is seq.service.last_payload_json
    assert seq.history is seq.service.history
    # prepare/fire route through the service: its state machine advances + records the source
    PulseController(seq, _scan_state()).on_pulse(repeat_forever=True, scan_repeats=0)
    assert seq.service.state == "running"
    assert seq.service.last_payload_json and '"periods"' in seq.service.last_payload_json
    # the live scan reading is the service's seam (it invokes the virtual real-time callback)
    assert seq.scan_progress()["scanning"] is True


# ---- review fixes: finite scan requires >=2 points + implies streaming + can be waited on ----
def _one_point_scan_state() -> PulseTableState:
    st = PulseTableState(channels=["probe", "trig"])
    st.bind_field("duration", "0", unit="us")
    st.set_scan_table([[10.0]])      # a single scan point
    return st


def test_finite_one_point_scan_is_rejected():
    # A single point never produces a cursor wrap, so the host could never count sweeps -> it would
    # hang forever on real hardware.  The rejection lives on the COMPILED program (__post_init__),
    # the one gate EVERY fire path passes -- so GUI, notebook and real all refuse it identically.
    one_point = _one_point_scan_state()
    one_point.scan_repeats = 2
    with pytest.raises(ValueError, match="at least 2 scan points"):
        one_point.compile_scan(clock_hz=50_000_000.0)


def test_finite_scan_repeats_implies_repeat_forever():
    # scan_repeats>0 is a streamed-then-stopped scan; a bare repeat_forever=False must NOT demote it
    # to a single sweep.  The streaming invariant lives on the COMPILED program (every fire path
    # passes __post_init__), so the payload carries the count and the compile forces streaming.
    payload = PulseController(VirtualSequencer(channels=["probe", "trig"]), _scan_state()).payload(
        repeat_forever=False, scan_repeats=2)
    assert payload.scan_repeats == 2
    assert payload.compile_scan(clock_hz=50_000_000.0).repeat_forever is True


def test_infinite_scan_streams_without_explicit_repeat_forever():
    # scan_repeats=0 (the inf default) IS a streamed scan too: it must compile as repeat_forever so
    # On Pulse keeps sweeping FOREVER.  The pulse GUI fires a scan WITHOUT toggling the whole-table
    # flag, so a bare scan_repeats=0 (repeat_forever unset) must NOT demote to a play-once program --
    # that was the "scan repeats = 0 / inf, On Pulse does nothing" bug.
    payload = PulseController(VirtualSequencer(channels=["probe", "trig"]), _scan_state()).payload(scan_repeats=0)
    assert payload.repeat_forever is True and payload.scan_repeats == 0


def test_infinite_scan_actually_runs_on_pulse_without_flag():
    # End-to-end mirror of the GUI On Pulse: firing an inf scan with NO whole-table flag must leave
    # the streamer continuously firing and the scan progressing (not fire-once-and-stop).
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    PulseController(seq, _scan_state()).on_pulse(scan_repeats=0)
    assert seq.firing is not None                          # the streamer keeps firing (it streams)
    assert seq.scan_progress()["scanning"] is True


def test_finite_scan_can_be_waited_without_timeout():
    # A finite scan DOES finish, so on_pulse(wait=True) must not raise the "cannot wait for a
    # repeat_forever pulse" guard (it would for an infinite scan).
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    PulseController(seq, _scan_state()).on_pulse(wait=True, scan_repeats=2)   # must not raise
    with pytest.raises(RuntimeError, match="cannot wait"):
        PulseController(VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0),
                        _scan_state()).on_pulse(wait=True, scan_repeats=0)     # infinite -> still guarded


def test_virtual_terminal_reading_latches_like_real():
    # After a finite scan finishes, scan_progress() keeps returning the SATURATED done reading
    # (not idle) until the next prepare/stop -- matching the real streamer's latch (virtual==real).
    seq = VirtualSequencer(channels=["probe", "trig"], sleep_scale=0.0)
    PulseController(seq, _scan_state()).on_pulse(repeat_forever=True, scan_repeats=2)
    first = seq.scan_progress()
    second = seq.scan_progress()
    assert first == second == {"scanning": False, "point": 2, "n_points": 3, "sweep": 1, "n_repeats": 2}
