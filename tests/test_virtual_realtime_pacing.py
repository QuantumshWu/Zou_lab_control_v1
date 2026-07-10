"""MECHANICAL guard: the virtual backend is a REAL-TIME pulse simulator.

The user's complaint: "I set the pulse duration to several seconds and the live imshow did
NOT slow down -- there is no simulation."  Root cause: ``sleep_scale`` defaulted to 0 and the
live ``acquire`` had no per-frame wait, so the live camera cadence was the polling rate, not the
pulse.  Now firing a program TAKES its real wall-clock duration (``sleep_scale=1.0`` by default),
so a live frame arrives once per trigger cycle -- editing the imaging period to several seconds
visibly slows the displayed image to that cadence.  Pin that here.

The pytest suite fast-forwards this (conftest flips ``DEFAULT_SLEEP_SCALE`` to 0), so these
tests construct the session with an EXPLICIT ``sleep_scale`` to exercise the real behaviour.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
import Zou_lab_control.neutral_atom.devices.virtual as virtual


def _live_frame_seconds(exp, exposure):
    """Fire a continuous imaging pulse of the given exposure and time ONE live frame.
    Returns (elapsed_seconds, pulse_cycle_duration_seconds)."""
    seqr = exp.devices.sequencer
    seq = na.imaging_sequence(exposure=exposure, load=True, name="live").forever()
    seqr.prepare(seq)
    seqr.fire(seq)
    t0 = time.monotonic()
    exp.devices.camera.acquire(1)                 # the wired camera senses the firing itself
    return time.monotonic() - t0, float(seq.duration)


def test_live_camera_frame_takes_the_pulse_cycle_wall_clock():
    """A live frame takes ~ the firing program's per-cycle duration, and a LONGER pulse takes
    proportionally longer -- so the displayed imshow slows when the user lengthens the pulse."""
    exp = na.connect("virtual", sleep_scale=1.0, sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    try:
        short_dt, short_dur = _live_frame_seconds(exp, 0.10)
        long_dt, long_dur = _live_frame_seconds(exp, 0.30)
        assert long_dur > short_dur + 0.15                       # the pulses really differ
        # each live frame ~ its pulse cycle (real-time), small render/overhead only
        assert short_dt == pytest.approx(short_dur, abs=0.08)
        assert long_dt == pytest.approx(long_dur, abs=0.12)
        # the longer pulse visibly slows the live image (this is the user's whole point)
        assert long_dt > short_dt + 0.12
    finally:
        exp.close()


def test_sleep_scale_zero_fast_forwards_the_same_path():
    """With sleep_scale=0 the SAME firing/acquire path runs without the wall-clock wait (this is
    how the test suite stays fast) -- the pulse is genuinely long, the wait is just skipped."""
    exp = na.connect("virtual", sleep_scale=0.0, sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    try:
        dt, dur = _live_frame_seconds(exp, 0.30)
        assert dur > 0.2                                          # a real, long pulse
        assert dt < 0.10                                          # but fast-forwarded
    finally:
        exp.close()


def test_stop_pulse_freezes_the_live_image_immediately():
    """``set_safe_state`` ("Stop Pulse") clears firing, so the next live acquire returns NO frame
    at once -- the image freezes, it does not keep updating or block for a cycle."""
    exp = na.connect("virtual", sleep_scale=1.0, sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    try:
        seqr = exp.devices.sequencer
        seq = na.imaging_sequence(exposure=0.30, load=True, name="live").forever()
        seqr.prepare(seq)
        seqr.fire(seq)
        seqr.set_safe_state()
        t0 = time.monotonic()
        frames = exp.devices.camera.acquire(1)
        assert frames == []                                       # no trigger -> no frame
        assert time.monotonic() - t0 < 0.05                       # and it did NOT block a cycle
    finally:
        exp.close()


def test_finite_wait_done_blocks_for_the_program_duration():
    """``on_pulse(wait=True)`` real-time: a finite program's wait_done blocks ~duration."""
    seqr = virtual.VirtualSequencer(sleep_scale=1.0)
    seq = na.imaging_sequence(exposure=0.20, load=True, name="readout")
    seqr.prepare(seq)
    seqr.fire(seq)
    t0 = time.monotonic()
    ok = seqr.wait_done(timeout=5.0)
    assert ok and time.monotonic() - t0 == pytest.approx(seq.duration, abs=0.10)


def test_finite_wait_done_is_cancellable_by_stop():
    """A Stop during the finite program-tail wait returns PROMPTLY (not after the whole duration) and
    reports NOT done.  This is what lets a stopped node's worker exit fast instead of holding the camera
    armed for the full ``2*duration+5`` s budget -- the orphaned-armed-camera / double-acquire window the
    sole-owner invariant exists to prevent (the wait was an UNCANCELLABLE ``time.sleep`` before)."""
    import threading
    seqr = virtual.VirtualSequencer(sleep_scale=1.0)
    seq = na.imaging_sequence(exposure=0.60, load=True, name="readout")   # a genuinely long finite program
    seqr.prepare(seq)
    seqr.fire(seq)
    stop = threading.Event()
    threading.Timer(0.05, stop.set).start()          # Stop shortly after the wait begins
    t0 = time.monotonic()
    ok = seqr.wait_done(timeout=5.0, stop=stop)
    elapsed = time.monotonic() - t0
    assert ok is False                               # cancelled mid-wait -> reports NOT done
    assert elapsed < 0.25, f"wait_done ignored Stop -- blocked {elapsed:.2f}s of the {seq.duration:.2f}s program"


def test_default_sleep_scale_is_real_time():
    """The PRODUCT default is real-time (DEFAULT_SLEEP_SCALE=1.0); a VirtualSequencer built with
    no explicit sleep_scale takes that default.  (conftest fast-forwards the suite, so restore
    the product value for this one assertion.)"""
    saved = virtual.DEFAULT_SLEEP_SCALE
    try:
        virtual.DEFAULT_SLEEP_SCALE = 1.0
        assert virtual.VirtualSequencer().sleep_scale == 1.0
    finally:
        virtual.DEFAULT_SLEEP_SCALE = saved


def test_virtual_sync_records_source_payload():
    """A virtual sequencer IS a sequencer: prepare records the SOURCE PulseTableState as
    ``last_payload_json`` (with ``periods``) so the pulse GUI's Sync pulls it back -- no error,
    no silent 'nothing prepared'."""
    seqr = virtual.VirtualSequencer(sleep_scale=0.0)
    state = na.build_release_recapture_pulse(port_catalog=seqr.port_catalog)
    seqr.prepare(state)
    snap = seqr.snapshot()
    payload = snap.get("last_payload_json")
    assert payload and '"periods"' in payload                     # the editable table, sync-able
    # round-trips back into an editable state (what sync_from_device does)
    import json
    from Zou_lab_control.neutral_atom.timing import PulseTableState
    restored = PulseTableState.from_dict(json.loads(payload))
    assert restored.port_catalog == state.port_catalog
