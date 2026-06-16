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


def _camera(grid=(2, 3), image=(20, 24), seed=1):
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualTrapArray
    return VirtualCamera(VirtualTrapArray(grid_shape=grid, image_shape=image, seed=seed), exposure=0.01)


def test_acquire_retains_and_latest_is_newest():
    cam = _camera()
    assert cam.latest() is None                      # nothing acquired yet
    frames = cam.acquire(2)
    assert len(frames) == 2
    np.testing.assert_array_equal(cam.latest(), frames[-1])
    assert len(cam.recent_frames()) == 2
    assert len(cam.recent_frames(1)) == 1


def test_drain_is_lossless_across_acquires():
    cam = _camera(seed=2)
    cam.acquire(2)
    assert len(cam.drain()) == 2                      # all frames since start
    assert cam.drain() == []                          # nothing new since last drain
    later = cam.acquire(3)
    new = cam.drain()
    assert len(new) == 3                              # only the new ones
    np.testing.assert_array_equal(new[-1], later[-1])


def test_recent_capacity_bounds_retention():
    cam = _camera(grid=(2, 2), image=(16, 16), seed=3)
    cam.recent_capacity = 3                           # set before first retain (lazy init)
    cam.acquire(5)
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
