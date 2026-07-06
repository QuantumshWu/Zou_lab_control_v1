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


def test_real_camera_inherits_the_same_buffer():
    # The buffer API is defined ONCE on CameraDevice; the real qCMOS adapter must
    # NOT override it -- so virtual and real retain frames through the identical path.
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera

    for name in ("_retain", "recent_frames", "latest", "drain", "clear_recent"):
        assert getattr(QCMOSCamera, name) is getattr(CameraDevice, name), (
            f"QCMOSCamera overrides {name}; the recent-frames buffer must stay single-source on the base"
        )


def test_lazy_state_and_lock_are_created_once_atomically():
    """F4: the lazily-created buffer state and acquisition lock are created ATOMICALLY (``setdefault``),
    so every touch returns the SAME object -- a check-then-set could build two divergent buffers /
    two locks and defeat the arm/disarm mutual exclusion.  A structural stand-in for the (hard to
    provoke) first-touch race: repeated accesses are object-identical."""
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
    cam = VirtualCamera(VirtualTrapArray(grid_shape=(2, 2), image_shape=(16, 16), seed=1), exposure=1e-3)
    assert cam._recent_state() is cam._recent_state()          # one buffer, never two
    assert cam._acquire_lock() is cam._acquire_lock()          # one lock, never two


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
