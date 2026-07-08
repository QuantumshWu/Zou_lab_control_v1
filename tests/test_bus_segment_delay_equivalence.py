"""Equivalence proof for the SEGMENT-DESCRIPTOR DAC delay (the instruction-level delay that replaces
the per-DA-bit value-change ``g_busdly`` FIFO).

The delay must be TRUE PHYSICAL out[t]=in[t-d] (silent BUS_SAFE until t=d, first-frame correct) --
:func:`engine_model.bus_delay_line_reference` is that unchanging ground truth.  The NEW mechanism
(:func:`engine_model.rtl_bus_segment_delay_mirror`) captures each RESOLVED segment the live engine
applies and re-runs it delayed, so its buffer holds ONE descriptor per segment (a dense ``0->1012
over 500 ticks`` ramp = 1 entry, not ~500 value events).  These tests pin that the mechanism is
byte-identical to the reference across dense/steep/gentle ramps, edges, repeat_forever spanning many
frames, a scanned ramp cycling scan points, and the finite done-tail -- and that a too-shallow
segment FIFO drops (the overflow guard) while a deep-enough one matches, and that the depth the
pulse_test-shaped stressor needs is O(segments) (~100) not O(value-changes) (~25000)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_HOST = Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer" / "host"
if str(_HOST) not in sys.path:
    sys.path.insert(0, str(_HOST))
import engine_model as em  # noqa: E402

BW = 10
SAFE = 1 << (BW - 1)
FRAME = 1000


def _seg(bus, st, sp, sv, ev, mode="ramp", vsel=0, svsel=0):
    return NS(bus_index=bus, start_tick=st, stop_tick=sp, start_value=sv, stop_value=ev,
              mode=mode, value_select=vsel, stop_value_select=svsel,
              start_tick_coeffs=[], stop_tick_coeffs=[])


def _prog(segs, frame=FRAME, repeat=True, scan_points=None):
    return NS(bus_segments=segs, loop_end_tick=frame, ticks=[frame], repeat_forever=repeat,
              scan_coeff_frac_bits=8, scan_points=scan_points or [])


# edge->0 at tick0 then a ramp of varying steepness = a dense-per-frame DAC stressor
_DENSE = _prog([_seg(0, 0, 0, 0, 0, mode="edge"), _seg(0, 1, 501, 0, 1000, mode="ramp")])
_STEEP = _prog([_seg(0, 0, 0, 0, 0, mode="edge"), _seg(0, 1, 51, 0, 1000, mode="ramp")])
_GENTLE = _prog([_seg(0, 0, 0, 0, 0, mode="edge"), _seg(0, 1, 501, 0, 400, mode="ramp")])
_EDGES = _prog([_seg(0, 100, 100, 0, 300, mode="edge"), _seg(0, 600, 600, 0, 100, mode="edge")])

_PROGS = {"dense": _DENSE, "steep": _STEEP, "gentle": _GENTLE, "edges": _EDGES}
_DELAYS = [1, 2, 7, 137, FRAME - 1, FRAME, FRAME + 3, 5 * FRAME + 250, 50 * FRAME]


@pytest.mark.parametrize("name", list(_PROGS))
@pytest.mark.parametrize("d", _DELAYS)
def test_segment_delay_equals_reference(name, d):
    """The segment-descriptor delay == the exact out[t]=in[t-d] reference, every ramp shape + delay."""
    p = _PROGS[name]
    n = max(2 * FRAME, d + 3 * FRAME)
    undelayed, _ = em.bus_undelayed_and_log(p, 0, n, bus_width=BW)
    ref = em.bus_delay_line_reference(undelayed, d, safe_value=SAFE)
    got = em.rtl_bus_segment_delay_mirror(p, 0, d, n, bus_width=BW)
    assert got == ref


def test_first_frame_silent_until_d():
    """Before t=d the delayed bus holds BUS_SAFE (silent startup), then reproduces the stream."""
    d = 250
    n = 3 * FRAME
    got = em.rtl_bus_segment_delay_mirror(_DENSE, 0, d, n, bus_width=BW)
    assert all(v == SAFE for v in got[:d])
    undelayed, _ = em.bus_undelayed_and_log(_DENSE, 0, n, bus_width=BW)
    assert got[d:] == undelayed[:n - d]


@pytest.mark.parametrize("d", [1, 50, 300, 900, 1500])
def test_finite_program_done_tail(d):
    """A finite (non-repeat) program's delay flushes the last d ticks past the program end."""
    fin = _prog([_seg(0, 0, 0, 0, 0, mode="edge"), _seg(0, 1, 501, 0, 1000, mode="ramp")], repeat=False)
    n = FRAME + 2000
    undelayed, _ = em.bus_undelayed_and_log(fin, 0, n, bus_width=BW)
    ref = em.bus_delay_line_reference(undelayed, d, safe_value=SAFE)
    assert em.rtl_bus_segment_delay_mirror(fin, 0, d, n, bus_width=BW) == ref


@pytest.mark.parametrize("d", [1, 137, FRAME + 3, 3 * FRAME + 250, 10 * FRAME])
def test_scan_cycle_stays_correct(d):
    """A scanned-target ramp cycling scan points frame-to-frame stays byte-exact under delay --
    the delay captures the ALREADY-RESOLVED per-point values, so it is scan-correct across the sweep."""
    scan = NS(bus_segments=[_seg(0, 0, 0, 0, 0, mode="edge"),
                            _seg(0, 1, 501, 0, 0, mode="ramp", svsel=1)],
              loop_end_tick=FRAME, ticks=[FRAME], repeat_forever=True,
              scan_coeff_frac_bits=8, scan_points=[[200], [900], [500]])
    seq = [0, 1, 2]
    n = d + 4 * FRAME
    undelayed, _ = em.bus_undelayed_and_log(scan, 0, n, bus_width=BW, scan_point_seq=seq)
    ref = em.bus_delay_line_reference(undelayed, d, safe_value=SAFE)
    got = em.rtl_bus_segment_delay_mirror(scan, 0, d, n, bus_width=BW, scan_point_seq=seq)
    assert got == ref


@pytest.mark.parametrize("d", [1, 2, 50, 137, FRAME + 3, 3 * FRAME + 70])
def test_ramp_reaching_loop_end_no_tick0_segment(d):
    """FRAME-BOUNDARY case: a repeat_forever bus that idles in period 0 (NO tick-0 segment) and ramps
    to stop_tick == loop_end_tick.  The live engine TRUNCATES the ramp at the wrap (holds the carried
    value, never snaps to target); the delayed re-player must FREEZE at the boundary, not ramp on into
    the next frame's idle window.  (Regression for the delay-P1 frame-boundary bug.)"""
    p = _prog([_seg(0, 30, 100, 0, 900, mode="ramp")])   # no tick-0 edge; ramp rstop == FRAME=...
    p.loop_end_tick = 100
    p.ticks = [100]
    n = d + 5 * 100
    undelayed, _ = em.bus_undelayed_and_log(p, 0, n, bus_width=BW)
    ref = em.bus_delay_line_reference(undelayed, d, safe_value=SAFE)
    assert em.rtl_bus_segment_delay_mirror(p, 0, d, n, bus_width=BW) == ref


@pytest.mark.parametrize("d", [1, 2, 50, 137, 250])
def test_segment_applies_on_final_frame_tick(d):
    """FRAME-BOUNDARY case: a segment whose effective start is the frame's LAST tick applies after
    that tick's output, so the carry into the next frame is the value REGISTER, not the last output
    sample.  (Regression for the seed-carry bug.)"""
    fr = 100
    p = _prog([_seg(0, 0, 0, 0, 200, mode="edge"), _seg(0, fr - 1, fr - 1, 0, 700, mode="edge")], frame=fr)
    n = d + 5 * fr
    undelayed, _ = em.bus_undelayed_and_log(p, 0, n, bus_width=BW)
    ref = em.bus_delay_line_reference(undelayed, d, safe_value=SAFE)
    assert em.rtl_bus_segment_delay_mirror(p, 0, d, n, bus_width=BW) == ref


def test_overflow_drops_shallow_matches_deep():
    """A too-shallow segment FIFO diverges (the RTL overflow-drop guard); a deep-enough one matches."""
    d, n = 50 * FRAME, 60 * FRAME
    undelayed, _ = em.bus_undelayed_and_log(_DENSE, 0, n, bus_width=BW)
    ref = em.bus_delay_line_reference(undelayed, d, safe_value=SAFE)
    deep, peak = em.rtl_bus_segment_delay_mirror(_DENSE, 0, d, n, bus_width=BW,
                                                 seg_depth=256, return_occupancy=True)
    shallow = em.rtl_bus_segment_delay_mirror(_DENSE, 0, d, n, bus_width=BW, seg_depth=8)
    assert deep == ref
    assert shallow != ref
    assert peak <= 256


def test_depth_is_segment_count_not_value_change_count():
    """The pulse_test-shaped stressor (a dense ramp delayed ~50 frames) needs O(segments) depth
    (~2/frame x 50 windows = ~100), NOT O(value-changes) (~500/frame x 50 = ~25000) -- the whole
    point of the instruction-level delay."""
    d, n = 50 * FRAME, 55 * FRAME
    _, peak = em.rtl_bus_segment_delay_mirror(_DENSE, 0, d, n, bus_width=BW, return_occupancy=True)
    undelayed, _ = em.bus_undelayed_and_log(_DENSE, 0, n, bus_width=BW)
    value_changes = sum(1 for i in range(1, min(FRAME, len(undelayed)))
                        if undelayed[i] != undelayed[i - 1]) * (d // FRAME)
    assert peak <= 128                      # a shallow segment FIFO covers it
    assert value_changes > 20 * peak        # the old per-value-change buffer needed >20x more
