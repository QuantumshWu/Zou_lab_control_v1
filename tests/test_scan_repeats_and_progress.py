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
