"""Contract: the live camera is PULSE-DRIVEN bottom-up -- it produces a frame ONLY while the
streamer is FIRING a continuous (camera-triggering) pulse.

This pins the user-visible behaviour that regressed: with a pulse running the live 2D qCMOS
image streams; the moment the pulse is stopped (``set_safe_state`` -- the GUI "Stop Pulse"
button) the camera sees no more triggers, so the live view FREEZES on its last frame instead
of fabricating a fresh one every tick.

The gate lives in the DATA SOURCE (only the lowest layer is faked, per AGENTS.md): the virtual
camera reads the sequencer's firing state; a real qCMOS learns the same thing from the absence
of hardware trigger edges -- so the camera measurement runs the SAME contract path on both
backends (``camera.acquire(...)``), only the frame source differs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

from conftest import fire_live_imaging   # the live "On Pulse" the trigger-driven camera needs


def test_live_camera_streams_only_while_the_pulse_is_firing():
    """The CameraMeasurement live monitor: no frame before On Pulse, a NEW frame each shot
    while firing, FROZEN on Stop Pulse, resuming on re-fire -- end-to-end through the node."""
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
    try:
        # 1) Nothing fired yet -> the streamer emits no camera trigger -> NO live frame.
        cam.step()
        assert "frame_0" not in hub.names()

        # 2) "On Pulse": fire the continuous imaging pulse -> the camera streams and every
        #    shot is a NEW frame (the live 2D actually updates).
        fire_live_imaging(exp)
        cam.step()
        first = np.asarray(hub.latest("frame_0")).copy()
        cam.step()
        second = np.asarray(hub.latest("frame_0")).copy()
        assert first.ndim == 4                                # the (repeat, 1, H, W) data block
        assert not np.array_equal(first, second)              # live updating while firing

        # 3) "Stop Pulse" (set_safe_state): no more triggers -> the live view FREEZES.
        #    Stepping the camera many more times never changes the last frame (NOT fabricated).
        exp.devices.sequencer.set_safe_state()
        frozen = np.asarray(hub.latest("frame_0")).copy()
        for _ in range(6):
            assert cam.step() == {}                           # no publish: the view holds
        # the (repeat, H, W) data array is frozen (equal_nan: the not-yet-filled ring rows are NaN)
        assert np.array_equal(np.asarray(hub.latest("frame_0")), frozen, equal_nan=True)

        # 4) Re-fire -> the live view resumes.
        fire_live_imaging(exp)
        cam.step()
        assert not np.array_equal(np.asarray(hub.latest("frame_0")), frozen, equal_nan=True)
    finally:
        cam.stop()
        exp.close()


def test_virtual_camera_device_does_not_fabricate_a_frame_when_idle():
    """At the DATA-SOURCE level: the virtual camera, given a sequencer that is not firing,
    returns NO frame (it never fabricates one) -- the bottom-up gate the live monitor relies
    on, and the faithful mirror of a real qCMOS that just sees no trigger edges."""
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    cam, seqr = exp.devices.camera, exp.devices.sequencer
    try:
        assert seqr.firing is None
        assert len(cam.acquire(1, sequencer=seqr)) == 0          # idle -> no fabricated frame

        fire_live_imaging(exp)
        assert seqr.firing is not None
        frames = cam.acquire(1, sequencer=seqr)
        assert len(frames) == 1 and np.asarray(frames[0]).ndim == 2

        seqr.set_safe_state()
        assert seqr.firing is None
        assert len(cam.acquire(1, sequencer=seqr)) == 0          # Stop Pulse -> no frame again
    finally:
        exp.close()
