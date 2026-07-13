"""Contract: the camera DEVICE owns a recent-frames ring so externally-triggered
frames are never lost, and the real camera inherits it IDENTICALLY (virtual == real).

The camera is triggered by the FPGA, so a single fired shot can yield more frames
than a consumer asked for (e.g. two ``emCCD`` triggers in one pulse).  Retaining the
most-recent frames on the device base (`CameraDevice`) means a live consumer polling
`drain()` gets ALL frames captured between polls, and `latest()` always holds the
newest.  The buffer lives on the base class, so `QCMOSCamera` retains exactly like
`VirtualCamera` -- the same code path, only the frame source differs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def _rig(grid=(2, 3), image=(20, 24), seed=1):
    """A camera WIRED to an in-process streamer (the virtual trigger cable).  The camera is
    purely trigger-driven (no fabricated frames): frames exist only when the wired sequencer
    fires an imaging pulse while the camera is armed -- exactly the real-hardware topology."""
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualSequencer, VirtualTrapArray
    seqr = VirtualSequencer()
    cam = VirtualCamera(VirtualTrapArray(grid_shape=grid, image_shape=image, seed=seed),
                        exposure=0.01, sequencer=seqr)
    return cam, seqr


def _acquire(cam, seqr, frames):
    """One fired shot through the single measurement-layer helper (arm -> fire -> read)."""
    from Zou_lab_control.neutral_atom.operations.measurement import triggered_frames
    from Zou_lab_control.neutral_atom.timing import imaging_sequence
    seq = imaging_sequence(exposure=0.01, load=True, name="readout")
    return triggered_frames(cam, seqr, seq, frames)


def test_acquire_retains_and_latest_is_newest():
    cam, seqr = _rig()
    assert cam.latest() is None                      # nothing acquired yet
    frames = _acquire(cam, seqr, 2)
    assert len(frames) == 2
    np.testing.assert_array_equal(cam.latest(), frames[-1])
    assert len(cam.recent_frames()) == 2
    assert len(cam.recent_frames(1)) == 1


def test_drain_is_lossless_across_acquires():
    cam, seqr = _rig(seed=2)
    _acquire(cam, seqr, 2)
    assert len(cam.drain()) == 2                      # all frames since start
    assert cam.drain() == []                          # nothing new since last drain
    later = _acquire(cam, seqr, 3)
    new = cam.drain()
    assert len(new) == 3                              # only the new ones
    np.testing.assert_array_equal(new[-1], later[-1])


def test_recent_capacity_bounds_retention():
    cam, seqr = _rig(grid=(2, 2), image=(16, 16), seed=3)
    cam.recent_capacity = 3                           # set before first retain (lazy init)
    _acquire(cam, seqr, 5)
    assert len(cam.recent_frames()) == 3              # bounded ring
    assert len(cam.drain()) == 3                      # drain capped at what's retained
    cam.clear_recent()
    assert cam.latest() is None
    assert cam.drain() == []


def test_fast_forward_finite_source_streams_through_bounded_pending_ring():
    """Removing virtual wall time must not collapse a whole hardware run into one
    atomic host-side burst.  The finite producer and exact reader advance together
    through the same bounded driver-retention contract used by a real camera.
    """

    from Zou_lab_control.neutral_atom.devices.virtual import (
        VirtualCamera,
        VirtualSequencer,
        VirtualTrapArray,
    )
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    sequencer = VirtualSequencer(sleep_scale=0)
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(2, 2), image_shape=(6, 8), seed=31),
        exposure=0.001,
        sequencer=sequencer,
    )
    program = imaging_sequence(exposure=0.001, load=True).repeated(36)
    camera.arm(36, max_inflight_frames=16)
    try:
        sequencer.prepare(program)
        sequencer.fire(program)
        frames = camera.read_frames(36)
        assert len(frames) == 36
        state = camera._recent_state()
        with state["cond"]:
            assert state["pending_capacity"] == 16
            assert not state["pending"]
            assert state["source_ordinal"] == 36
            assert state["cond"].wait_for(
                lambda: state["_finite_delivery_phase"].value in {"done", "failed"},
                timeout=1.0,
            )
            assert state["_finite_delivery_phase"].value == "done"
    finally:
        terminal = camera.finish_record_capture()
    assert terminal.produced_count == 36
    assert terminal.source_stopped and terminal.no_more_frames and terminal.joined


def test_finite_fire_returns_before_camera_rendering_completes():
    """FIRE is a non-blocking hardware command; camera production is asynchronous."""

    import threading
    import time

    from Zou_lab_control.neutral_atom.devices.virtual import (
        VirtualCamera,
        VirtualSequencer,
        VirtualTrapArray,
    )
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    sequencer = VirtualSequencer(sleep_scale=0)
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(1, 1), image_shape=(4, 4), seed=32),
        exposure=0.001,
        sequencer=sequencer,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked_frames(_sequence, frames):
        assert frames == 1
        entered.set()
        assert release.wait(2.0)
        yield np.ones((4, 4), dtype=np.float64)

    camera._iter_render_frames = blocked_frames
    program = imaging_sequence(exposure=0.001, load=True)
    camera.arm(1)
    try:
        sequencer.prepare(program)
        started = time.monotonic()
        sequencer.fire(program)
        assert time.monotonic() - started < 0.5
        assert entered.wait(1.0)
        release.set()
        frames = camera.read_frames(1)
        assert len(frames) == 1
        np.testing.assert_array_equal(frames[0], np.ones((4, 4)))
    finally:
        release.set()
        camera.disarm()


def test_default_finite_read_timeout_bounds_a_wedged_frame_source():
    """``timeout=None`` means the camera default, never an infinite source wait."""

    import threading

    from Zou_lab_control.neutral_atom.devices.virtual import (
        VirtualCamera,
        VirtualSequencer,
        VirtualTrapArray,
    )
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    sequencer = VirtualSequencer(sleep_scale=0)
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(1, 1), image_shape=(4, 4), seed=43),
        exposure=0.001,
        timeout=0.05,
        sequencer=sequencer,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked_frames(_sequence, _frames):
        entered.set()
        assert release.wait(2.0)
        yield np.zeros((4, 4), dtype=np.float64)

    camera._iter_render_frames = blocked_frames
    program = imaging_sequence(exposure=0.001, load=True)
    camera.arm(1)
    sequencer.prepare(program)
    sequencer.fire(program)
    assert entered.wait(1.0)
    errors = []

    def read():
        try:
            camera.read_frames(1)
        except BaseException as error:
            errors.append(error)

    reader = threading.Thread(target=read)
    reader.start()
    reader.join(0.3)
    completed_within_default = not reader.is_alive()
    release.set()
    reader.join(2.0)
    try:
        assert completed_within_default
        assert len(errors) == 1
        assert isinstance(errors[0], TimeoutError)
    finally:
        camera.disarm()


def test_finite_camera_worker_failure_reaches_reader_and_terminal():
    """A background renderer may never turn into a silent short capture."""

    from Zou_lab_control.neutral_atom.devices.virtual import (
        VirtualCamera,
        VirtualSequencer,
        VirtualTrapArray,
    )
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    sequencer = VirtualSequencer(sleep_scale=0)
    camera = VirtualCamera(
        VirtualTrapArray(grid_shape=(1, 1), image_shape=(4, 4), seed=33),
        exposure=0.001,
        sequencer=sequencer,
    )

    def broken_frames(_sequence, _frames):
        raise ValueError("render failed")
        yield  # pragma: no cover - make this a generator without hiding the fault

    camera._iter_render_frames = broken_frames
    program = imaging_sequence(exposure=0.001, load=True)
    camera.arm(1)
    sequencer.prepare(program)
    sequencer.fire(program)
    with pytest.raises(RuntimeError, match="finite-frame source failed") as read_error:
        camera.read_frames(1)
    assert isinstance(read_error.value.__cause__, ValueError)
    with pytest.raises(RuntimeError, match="finite-frame source failed"):
        camera.finish_record_capture()
    state = camera._recent_state()
    with state["cond"]:
        assert not state["armed"]
        assert not state["pending"]


def test_zero_edge_finite_fire_is_immediate_loud_deficit():
    cam, seqr = _rig(seed=34)
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    cam.capture_trigger_channels = ("not-connected",)
    program = imaging_sequence(exposure=0.001, load=True)
    cam.arm(1)
    try:
        seqr.prepare(program)
        seqr.fire(program)
        with pytest.raises(TimeoutError, match="only 0 trigger edge"):
            cam.read_frames(1)
    finally:
        cam.disarm()


def test_stop_while_finite_source_is_running_is_cancellation_not_deficit():
    import threading

    from Zou_lab_control.neutral_atom.devices.base import AcquisitionCancelled
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    cam, seqr = _rig(seed=35)
    entered = threading.Event()
    release = threading.Event()

    def blocked_frames(_sequence, _frames):
        entered.set()
        assert release.wait(2.0)
        yield np.zeros((20, 24), dtype=np.float64)

    cam._iter_render_frames = blocked_frames
    program = imaging_sequence(exposure=0.001, load=True)
    cam.arm(1)
    seqr.prepare(program)
    seqr.fire(program)
    assert entered.wait(1.0)
    stop = threading.Event()
    stop.set()
    try:
        with pytest.raises(AcquisitionCancelled):
            cam.read_frames(1, stop=stop)
    finally:
        release.set()
        terminal = cam.finish_record_capture()
    assert terminal.joined
    with cam._recent_state()["cond"]:
        assert not cam._recent_state()["armed"]


def test_finite_worker_start_and_disarm_have_one_linearization_point(monkeypatch):
    """Terminal completion may not observe an installed-but-unstarted thread."""

    import threading

    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    cam, seqr = _rig(seed=36)
    entered_start = threading.Event()
    release_start = threading.Event()
    original_start = threading.Thread.start

    def gated_start(thread):
        if thread.name.startswith("zlc-virtual-camera-fire-"):
            entered_start.set()
            assert release_start.wait(2.0)
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", gated_start)
    program = imaging_sequence(exposure=0.001, load=True)
    cam.arm(1)
    seqr.prepare(program)
    fire_errors = []
    terminal_errors = []

    def fire():
        try:
            seqr.fire(program)
        except BaseException as error:  # pragma: no cover - asserted below
            fire_errors.append(error)

    def finish():
        try:
            cam.finish_record_capture()
        except BaseException as error:  # pragma: no cover - asserted below
            terminal_errors.append(error)

    fire_thread = threading.Thread(target=fire, name="test-fire")
    fire_thread.start()
    assert entered_start.wait(1.0)
    finish_thread = threading.Thread(target=finish, name="test-finish")
    finish_thread.start()
    release_start.set()
    fire_thread.join(2.0)
    finish_thread.join(2.0)
    assert not fire_thread.is_alive() and not finish_thread.is_alive()
    assert fire_errors == [] and terminal_errors == []
    # The out-of-band finisher stopped the source but could not release the
    # RLock owned by this arm thread.  Consume its cached terminal here so the
    # acquisition authority itself is also proven reusable.
    cam.finish_record_capture()
    state = cam._recent_state()
    with state["cond"]:
        assert not state["armed"]
        assert not state["pending"]


def test_disarm_of_ring_blocked_source_cannot_leak_into_rearm():
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    cam, seqr = _rig(seed=37)
    many = imaging_sequence(exposure=0.001, load=True).repeated(3)
    cam.arm(3, max_inflight_frames=1)
    seqr.prepare(many)
    seqr.fire(many)
    state = cam._recent_state()
    with state["cond"]:
        assert state["cond"].wait_for(
            lambda: state["source_ordinal"] == 1,
            timeout=1.0,
        )
    cam.disarm()

    one = imaging_sequence(exposure=0.001, load=True)
    cam.arm(1, max_inflight_frames=1)
    try:
        seqr.prepare(one)
        seqr.fire(one)
        frames = cam.read_frames(1)
        assert len(frames) == 1
        with state["cond"]:
            assert state["source_ordinal"] == 1
    finally:
        cam.disarm()


def test_close_stops_ring_blocked_source_and_unplugs_trigger_wire():
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    cam, seqr = _rig(seed=38)
    program = imaging_sequence(exposure=0.001, load=True).repeated(3)
    cam.arm(3, max_inflight_frames=1)
    seqr.prepare(program)
    seqr.fire(program)
    state = cam._recent_state()
    with state["cond"]:
        assert state["cond"].wait_for(
            lambda: state["source_ordinal"] == 1,
            timeout=1.0,
        )
        worker = state["_finite_delivery_thread"]
        assert worker.is_alive()

    cam.close()

    assert not worker.is_alive()
    with state["cond"]:
        assert not state["armed"]
        assert not state["pending"]
    assert cam._on_wire_fired not in seqr._fire_listeners
    with pytest.raises(RuntimeError, match="closed virtual camera"):
        cam.arm(1)


def test_close_retries_a_residual_subclass_source_when_epoch_is_unarmed():
    """A failed prior terminal join must not hide a subclass producer from close."""

    from Zou_lab_control.neutral_atom.devices.virtual import VirtualMotCamera

    class ResidualProducer:
        def __init__(self):
            self.alive = True
            self.join_calls = 0

        def join(self, timeout=None):
            assert timeout == 2.0
            self.join_calls += 1
            self.alive = False

        def is_alive(self):
            return self.alive

    camera = VirtualMotCamera(width=8, height=6, exposure=0.001)
    producer = ResidualProducer()
    camera._producer = producer

    camera.close()

    assert producer.join_calls == 1
    assert camera._producer is None


def test_nonzero_time_scale_ring_overrun_is_loud():
    from Zou_lab_control.neutral_atom.devices.base import CameraBufferOverrun
    from Zou_lab_control.neutral_atom.devices.virtual import (
        VirtualCamera,
        VirtualSequencer,
        VirtualTrapArray,
    )
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    seqr = VirtualSequencer(sleep_scale=1.0)
    cam = VirtualCamera(
        VirtualTrapArray(grid_shape=(1, 1), image_shape=(4, 4), seed=39),
        exposure=0.0001,
        timeout=0.2,
        sequencer=seqr,
    )
    program = imaging_sequence(exposure=0.0001, load=True).repeated(2)
    cam.arm(2, max_inflight_frames=1)
    seqr.prepare(program)
    seqr.fire(program)
    state = cam._recent_state()
    with state["cond"]:
        assert state["cond"].wait_for(
            lambda: state["_finite_delivery_phase"].value == "failed",
            timeout=2.0,
        )
        assert len(state["pending"]) == 1
        assert state["source_ordinal"] == 1
        assert isinstance(state["_finite_delivery_error"], CameraBufferOverrun)

    with pytest.raises(RuntimeError) as read_error:
        cam.read_frames(2)
    assert isinstance(read_error.value.__cause__, CameraBufferOverrun)
    with pytest.raises(RuntimeError, match="finite-frame source failed"):
        cam.finish_record_capture()


def test_close_racing_backend_arm_cannot_leave_a_closed_armed_camera():
    import threading

    cam, _seqr = _rig(seed=40)
    entered = threading.Event()
    release = threading.Event()
    real_arm = cam._arm

    def blocked_arm(frames, *, max_inflight_frames=None):
        entered.set()
        assert release.wait(2.0)
        return real_arm(frames, max_inflight_frames=max_inflight_frames)

    cam._arm = blocked_arm
    arm_errors = []
    close_errors = []

    def arm():
        try:
            cam.arm(1)
        except BaseException as error:
            arm_errors.append(error)

    def close():
        try:
            cam.close()
        except BaseException as error:  # pragma: no cover - asserted below
            close_errors.append(error)

    arm_thread = threading.Thread(target=arm)
    arm_thread.start()
    assert entered.wait(1.0)
    close_thread = threading.Thread(target=close)
    close_thread.start()
    release.set()
    arm_thread.join(2.0)
    close_thread.join(2.0)

    assert not arm_thread.is_alive() and not close_thread.is_alive()
    assert len(arm_errors) == 1
    assert "closed virtual camera" in str(arm_errors[0])
    assert close_errors == []
    state = cam._recent_state()
    with state["cond"]:
        assert not state["arming"] and not state["armed"]


def test_old_fire_schedule_parse_cannot_cross_into_a_new_arm_epoch(monkeypatch):
    import threading

    import Zou_lab_control.neutral_atom.devices.virtual as virtual_devices
    from Zou_lab_control.neutral_atom.timing import imaging_sequence

    cam, seqr = _rig(seed=42)
    entered = threading.Event()
    release = threading.Event()
    original_offsets = virtual_devices.camera_trigger_pulse_offsets

    def blocked_offsets(camera, sequence):
        if sequence.name == "epoch-a":
            entered.set()
            assert release.wait(2.0)
        return original_offsets(camera, sequence)

    monkeypatch.setattr(
        virtual_devices,
        "camera_trigger_pulse_offsets",
        blocked_offsets,
    )

    def sentinel_frames(sequence, _frames):
        value = 1.0 if sequence.name == "epoch-a" else 9.0
        yield np.full((20, 24), value, dtype=np.float64)

    cam._iter_render_frames = sentinel_frames
    first = imaging_sequence(exposure=0.001, load=True, name="epoch-a")
    second = imaging_sequence(exposure=0.001, load=True, name="epoch-b")
    cam.arm(1)
    seqr.prepare(first)
    fire_errors = []

    def fire_first():
        try:
            seqr.fire(first)
        except BaseException as error:  # pragma: no cover - asserted below
            fire_errors.append(error)

    fire_thread = threading.Thread(target=fire_first)
    fire_thread.start()
    assert entered.wait(1.0)
    cam.disarm()
    cam.arm(1)
    release.set()
    fire_thread.join(2.0)
    assert not fire_thread.is_alive() and fire_errors == []
    state = cam._recent_state()
    with state["cond"]:
        assert state["source_ordinal"] == 0
        assert not state["pending"]
        assert state["_finite_delivery_phase"].value == "idle"

    try:
        seqr.prepare(second)
        seqr.fire(second)
        frames = cam.read_frames(1)
        assert len(frames) == 1
        np.testing.assert_array_equal(frames[0], np.full((20, 24), 9.0))
    finally:
        cam.disarm()


def test_real_camera_inherits_the_same_buffer():
    # The buffer API is defined ONCE on CameraDevice; the real qCMOS adapter must
    # NOT override it -- so virtual and real retain frames through the identical path.
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera

    for name in ("_retain", "recent_frames", "latest", "drain", "clear_recent"):
        assert getattr(QCMOSCamera, name) is getattr(CameraDevice, name), (
            f"QCMOSCamera overrides {name}; the recent-frames buffer must stay single-source on the base"
        )


def _record(image_value, ordinal, *, produced_count=None):
    from Zou_lab_control.neutral_atom.devices.base import CameraFrameRecord

    return CameraFrameRecord(
        image=np.full((2, 2), image_value, dtype=np.float64),
        source_ordinal=ordinal,
        produced_count=(ordinal + 1 if produced_count is None else produced_count),
        frame_stamp=None,
        camera_stamp=None,
        timestamp_seconds=None,
        timestamp_microseconds=None,
        host_received_at_ns=1_000 + ordinal,
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_ordinal", 1.5, TypeError),
        ("source_ordinal", True, TypeError),
        ("source_ordinal", -1, ValueError),
        ("timestamp_microseconds", 1_000_000, ValueError),
        ("host_received_at_ns", 1.5, TypeError),
        ("host_received_at_ns", 0, ValueError),
    ],
)
def test_camera_frame_record_rejects_ambiguous_metadata(field, value, error):
    from Zou_lab_control.neutral_atom.devices.base import CameraFrameRecord

    kwargs = dict(
        image=np.zeros((2, 2)),
        source_ordinal=0,
        produced_count=1,
        frame_stamp=None,
        camera_stamp=None,
        timestamp_seconds=0,
        timestamp_microseconds=0,
        host_received_at_ns=1,
    )
    kwargs[field] = value
    with pytest.raises(error):
        CameraFrameRecord(**kwargs)


def test_record_and_array_readers_consume_one_armed_session_queue():
    """The transitional ndarray reader is a view of the record queue, not a fork."""

    cam, _seqr = _rig()
    cam.arm(2)
    try:
        cam._deliver_records([_record(1.0, 0), _record(2.0, 1)])
        first = cam.read_frame_records(1)
        second = cam.read_frames(1)
    finally:
        cam.disarm()

    assert [record.source_ordinal for record in first] == [0]
    np.testing.assert_array_equal(first[0].image, np.full((2, 2), 1.0))
    assert len(second) == 1
    np.testing.assert_array_equal(second[0], np.full((2, 2), 2.0))
    assert not cam._recent_state()["pending"]


def test_record_batch_overrun_is_atomic_and_never_overwrites_pending_frames():
    from Zou_lab_control.neutral_atom.devices.base import CameraBufferOverrun

    cam, _seqr = _rig()
    cam.arm(2)
    try:
        with pytest.raises(CameraBufferOverrun, match="arm budget"):
            cam._deliver_records(
                [_record(1.0, 0), _record(2.0, 1), _record(3.0, 2)]
            )
        state = cam._recent_state()
        assert list(state["pending"]) == []
        assert state["source_ordinal"] == 0
        assert cam.recent_frame_records() == []
    finally:
        cam.disarm()


def test_record_gap_is_atomic_and_preserves_existing_pending_data():
    from Zou_lab_control.neutral_atom.devices.base import CameraBufferOverrun

    cam, _seqr = _rig()
    cam.arm(3)
    try:
        cam._deliver_records([_record(1.0, 0)])
        with pytest.raises(CameraBufferOverrun, match="differs from expected 1"):
            cam._deliver_records([_record(3.0, 2)])
        state = cam._recent_state()
        assert [record.source_ordinal for record in state["pending"]] == [0]
        assert state["source_ordinal"] == 1
        assert [record.source_ordinal for record in cam.recent_frame_records()] == [0]
    finally:
        cam.disarm()


def test_virtual_record_terminal_reports_the_arm_epoch_delivery_count():
    cam, _seqr = _rig()
    cam.arm(2)
    cam._deliver_records([_record(1.0, 0), _record(2.0, 1)])
    assert len(cam.read_frame_records(2)) == 2
    terminal = cam.finish_record_capture()
    assert terminal.produced_count == 2
    assert terminal.source_stopped
    assert terminal.no_more_frames
    assert terminal.joined


def test_arm_rejects_inflight_capacity_larger_than_finite_cardinality():
    cam, _seqr = _rig()
    with pytest.raises(ValueError, match="cannot exceed"):
        cam.arm(2, max_inflight_frames=3)
    assert cam._recent_state()["armed"] is False


def test_out_of_band_terminal_stop_is_joined_by_original_arm_owner():
    import threading

    cam, _seqr = _rig()
    cam.arm(1)
    cam._deliver_records([_record(1.0, 0)])
    observed = {}

    def interrupt():
        observed["terminal"] = cam.finish_record_capture()

    worker = threading.Thread(target=interrupt)
    worker.start()
    worker.join(2.0)
    assert not worker.is_alive()
    assert cam._recent_state()["armed"] is False

    # The interrupt thread cannot release an RLock acquired by this thread.  The
    # original owner consumes the cached terminal and releases that ownership.
    assert cam.finish_record_capture() == observed["terminal"]
    cam.arm(1, timeout=0.1)
    cam.finish_record_capture()


def test_terminal_stop_wakes_an_unbounded_push_camera_read():
    import threading
    import time

    cam, _seqr = _rig()
    cam.arm(1)
    observed = {}

    def read():
        observed["records"] = cam.read_frame_records(1)

    reader = threading.Thread(target=read)
    reader.start()
    time.sleep(0.05)
    cam.finish_record_capture()
    reader.join(1.0)
    assert not reader.is_alive()
    assert observed["records"] == []


def test_lazy_state_and_lock_are_created_once_atomically():
    """F4: the lazily-created buffer state and acquisition lock are created ATOMICALLY (``setdefault``),
    so every touch returns the SAME object -- a check-then-set could build two divergent buffers /
    two locks and defeat the arm/disarm mutual exclusion.  A structural stand-in for the (hard to
    provoke) first-touch race: repeated accesses are object-identical."""
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
    cam = VirtualCamera(VirtualTrapArray(grid_shape=(2, 2), image_shape=(16, 16), seed=1), exposure=1e-3)
    assert cam._recent_state() is cam._recent_state()          # one buffer, never two
    assert cam._acquire_lock() is cam._acquire_lock()          # one lock, never two


def test_arm_is_bounded_and_cancellable_when_the_camera_is_contended():
    """#6 (MOT-field 0% hang): arm() must NOT block forever on the acquisition lock -- a Stop has to
    break a wait on a session another (possibly abandoned) consumer still holds, and a timeout must
    fail loudly instead of wedging.  arm now joins read/disarm's bounded+cancellable contract:
      * with a ``stop`` event that fires -> AcquisitionCancelled (a clean Stop, not a fault);
      * with a ``timeout`` that expires -> TimeoutError;
      * with neither (legacy) -> the old blocking acquire is preserved.
    Pinned by holding the lock from a second thread and confirming arm() returns control promptly."""
    import threading
    import time as _time

    from Zou_lab_control.neutral_atom.devices.base import AcquisitionCancelled
    cam, _seqr = _rig()

    # A rival consumer holds the acquisition lock from ANOTHER thread (the acquire lock is an RLock:
    # holding it in the calling thread would let arm re-enter it, so the contention must be cross-thread,
    # exactly the real case -- a live monitor's abandoned worker still holding the sensor).
    held = threading.Event()
    release = threading.Event()

    def _holder():
        cam._acquire_lock().acquire()
        held.set()
        release.wait(5.0)
        cam._acquire_lock().release()

    holder = threading.Thread(target=_holder, daemon=True)
    holder.start()
    assert held.wait(2.0), "rival thread failed to take the acquisition lock"
    try:
        # (a) stop cancels the wait cooperatively and promptly
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()
        t0 = _time.monotonic()
        try:
            cam.arm(1, stop=stop)
        except AcquisitionCancelled:
            pass
        else:
            raise AssertionError("contended arm(stop=...) must raise AcquisitionCancelled, not wedge")
        assert _time.monotonic() - t0 < 2.0, "cancellation must be prompt, not an unbounded wait"

        # (b) timeout fails loudly rather than blocking forever
        try:
            cam.arm(1, timeout=0.2)
        except TimeoutError:
            pass
        else:
            raise AssertionError("contended arm(timeout=...) must raise TimeoutError")
    finally:
        release.set()
        holder.join(2.0)

    # (c) once free, a plain arm still works and holds the lock (legacy contract intact)
    cam.arm(1)
    try:
        assert cam._recent_state()["armed"] is True
    finally:
        cam.disarm()


def test_triggered_frames_waits_for_the_finite_program_before_returning():
    """B2: reading the frames back is NOT the end of a finite program -- the sequence may keep
    playing past its last camera window (post-imaging reload / repump / reset segments carry no
    trigger edge).  ``triggered_frames`` must therefore ``wait_done`` the fired finite program
    BEFORE returning, so the caller's next ``prepare`` can never land on a still-RUNNING program
    (the AXI session treats that as an explicit switch and aborts it -- silently truncating the
    tail on real hardware).  Pinned on the call ORDER: every prepare after the first is preceded
    by a wait_done for the previous shot."""
    cam, seqr = _rig()
    calls: list[str] = []
    for name in ("prepare", "fire", "wait_done"):
        real = getattr(seqr, name)
        def _recorded(*a, _real=real, _name=name, **kw):
            calls.append(_name)
            return _real(*a, **kw)
        setattr(seqr, name, _recorded)

    _acquire(cam, seqr, 1)                            # two back-to-back finite shots
    _acquire(cam, seqr, 1)
    assert calls.count("wait_done") == 2, calls       # every finite shot waits its program out
    # ORDER: ... fire -> wait_done -> (next) prepare -- the wait separates consecutive programs.
    second_prepare = [i for i, c in enumerate(calls) if c == "prepare"][1]
    first_wait = calls.index("wait_done")
    assert first_wait < second_prepare, calls
