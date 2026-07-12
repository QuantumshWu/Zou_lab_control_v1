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
from Zou_lab_control.neutral_atom.operations.measurement import triggered_frames

from conftest import fire_live_imaging, raw_device_set


def test_live_camera_streams_only_while_the_pulse_is_firing():
    """The CameraMeasurement live monitor: no frame before On Pulse, a NEW frame each shot
    while firing, FROZEN on Stop Pulse, resuming on re-fire -- end-to-end through the node."""
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    cam = CameraMeasurement(hub, raw_device_set(exp).camera, sequencer=raw_device_set(exp).sequencer)
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
        raw_device_set(exp).sequencer.set_safe_state()
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
    """At the DATA-SOURCE level: the virtual camera, wired to a sequencer that is not firing,
    returns NO frame (it never fabricates one) -- the bottom-up gate the live monitor relies
    on, and the faithful mirror of a real qCMOS that just sees no trigger edges.  The wire
    (the camera's ``sequencer`` construction parameter) is the ONLY way a pulse reaches the
    sensor -- the acquiring code never hands the camera a sequence."""
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    cam, seqr = raw_device_set(exp).camera, raw_device_set(exp).sequencer
    try:
        assert seqr.firing is None
        assert len(cam.acquire(1)) == 0          # idle wire -> no fabricated frame

        fire_live_imaging(exp)
        assert seqr.firing is not None
        frames = cam.acquire(1)
        assert len(frames) == 1 and np.asarray(frames[0]).ndim == 2

        seqr.set_safe_state()
        assert seqr.firing is None
        assert len(cam.acquire(1)) == 0          # Stop Pulse -> no frame again
    finally:
        exp.close()


def test_prepare_replaces_the_continuous_firing():
    """F2: uploading a new program (``prepare``) REPLACES whatever the streamer was continuously
    firing -- on real hardware a prepare loads the next sequence, so a continuous On Pulse is no
    longer what the streamer emits.  A measurement that prepare+fires a FINITE program right after
    an On Pulse (without an explicit Stop) therefore clears ``firing`` -- the wired camera does NOT
    keep rendering the STALE continuous program."""
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    try:
        seqr = raw_device_set(exp).sequencer
        fire_live_imaging(exp)                       # continuous On Pulse -> firing set
        assert seqr.firing is not None
        # a measurement's arm-before-fire shot uploads a FINITE program: prepare replaces the play
        triggered_frames(
            raw_device_set(exp).camera,
            seqr,
            na.imaging_sequence(exposure=1e-3, load=True),
            2,
        )
        assert seqr.firing is None                   # the continuous On Pulse is no longer firing
    finally:
        exp.close()


def test_closing_the_camera_unplugs_the_trigger_cable():
    """F6: a closed camera stops rendering on the streamer's fires -- ``close`` unplugs the trigger
    cable (removes the fire listener), symmetric with the constructor-time wire-up, so a closed
    camera does not linger as a fire listener still producing frames."""
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualSequencer, VirtualTrapArray
    seqr = VirtualSequencer()
    cam = VirtualCamera(VirtualTrapArray(grid_shape=(2, 3), image_shape=(16, 20), seed=1),
                        exposure=1e-3, sequencer=seqr)
    seq = na.imaging_sequence(exposure=1e-3, load=True, name="readout")
    cam.arm(1)
    cam.close()                                      # unplug the cable
    seqr.prepare(seq); seqr.fire(seq)                # a fire the closed camera must NOT see
    assert cam.read_frames(1, timeout=0.05) == []    # nothing rendered: the listener was removed


def test_camera_sleep_scale_is_owned_by_the_sequencer():
    """F7: the virtual time scale has ONE owner -- the wired sequencer -- and the camera reads it
    through the wire, so the two can never drift.  Setting the sequencer's ``sleep_scale`` is
    immediately what the camera reports (no separate camera-side mirror to keep in sync)."""
    from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualSequencer, VirtualTrapArray
    seqr = VirtualSequencer(sleep_scale=0.25)
    cam = VirtualCamera(VirtualTrapArray(grid_shape=(2, 3), image_shape=(16, 20), seed=1),
                        exposure=1e-3, sequencer=seqr)
    assert cam.sleep_scale == 0.25                   # read straight from the wired sequencer
    seqr.sleep_scale = 0.0                            # the sequencer is the single owner
    assert cam.sleep_scale == 0.0                    # ... and the camera tracks it with no mirror
