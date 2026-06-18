"""Contract: CameraMeasurement reads ONE frame per camera trigger (frames_per_cycle).

A pulse that triggers the camera twice per cycle (e.g. a release-recapture / two-
readout "T" sequence) needs frames_per_cycle=2; the measurement then publishes
frame_0 + frame_1 (one per emCCD trigger), so two panels can show the two triggers.
The default frames_per_cycle=1 preserves the old single-frame behaviour (only frame +
frame_0). This pins the fix for "qCMOS live always shows the first emCCD".

Virtual == real: only the camera differs; the measurement reads frames through the
same acquire(n) contract.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, OccupancyProcessor
from Zou_lab_control.neutral_atom.core.signals import SignalHub


def test_two_triggers_publish_frame_0_and_frame_1():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    cam_node = CameraMeasurement(hub, exp.camera, frames_per_cycle=2)
    cam_node.step()

    assert set(cam_node.published_signals()) == {"frame", "frame_0", "frame_1"}
    for key in ("frame", "frame_0", "frame_1"):
        assert key in hub.names()
        assert np.asarray(hub.latest(key)).ndim == 2          # a real HxW image, not a scalar
    # the default 2D panel signal aliases the FIRST trigger
    assert np.array_equal(np.asarray(hub.latest("frame")), np.asarray(hub.latest("frame_0")))


def test_default_is_single_trigger_back_compat():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    cam_node = CameraMeasurement(SignalHub(), exp.camera)        # frames_per_cycle defaults to 1
    cam_node.step()
    assert set(cam_node.published_signals()) == {"frame", "frame_0"}


def test_readout_image_frame_judged_is_synced_with_occupancy():
    """#5: a 2D 'readout image' reads the occupancy processor's ``frame_judged``, which is
    co-published ATOMICALLY with ``occupied`` (one transform dict) -- so the image and the
    site-map rings are ALWAYS the same shot.  (A raw live ``frame`` panel runs one cycle
    ahead of the judged frame; ``frame_judged`` is the bottom-up sync point.)"""
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    exp.readout.sitemap(frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    cal = exp.readout.current
    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.camera)
    det = OccupancyProcessor(hub, calibration=cal, source="frame", method="box", grid_shape=(3, 4))
    for _ in range(4):
        cam.step()
        det.step()
    frame_judged = np.asarray(hub.latest("frame_judged"))
    occupied = np.asarray(hub.latest("occupied"))
    assert frame_judged.ndim == 2                                   # the readout image
    # the readout image and the rings are the SAME shot: re-judging frame_judged reproduces
    # exactly the published occupancy (they came from one atomic publish).
    assert np.array_equal(occupied, cal.detect(frame_judged, method="box").occupied)


def test_frames_per_cycle_is_live_editable():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    cam_node = CameraMeasurement(SignalHub(), exp.camera)
    assert cam_node.acquisition_parameters()["frames_per_cycle"] == 1
    cam_node.set_acquisition_parameters(frames_per_cycle=3)     # the owner-thread apply path uses this
    assert cam_node.frames_per_cycle == 3
    assert {"frame_0", "frame_1", "frame_2"} <= set(cam_node.published_signals())
