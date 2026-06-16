"""Contract: CameraFrameFeed reads ONE frame per camera trigger (frames_per_cycle).

A pulse that triggers the camera twice per cycle (e.g. a release-recapture / two-
readout "T" sequence) needs frames_per_cycle=2; the feed then publishes frame_0 +
frame_1 (one per emCCD trigger), so two panels can show the two triggers. The
default frames_per_cycle=1 preserves the old single-frame behaviour (only frame +
frame_0). This pins the fix for "qCMOS live always shows the first emCCD".

Virtual == real: only the camera differs; the feed reads frames through the same
acquire(n) contract.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.operations.feeds import CameraFrameFeed
from Zou_lab_control.neutral_atom.core.signals import SignalHub


def test_two_triggers_publish_frame_0_and_frame_1():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    feed = CameraFrameFeed(hub, exp.camera, frames_per_cycle=2)
    feed.step()

    assert set(feed.published_signals()) == {"frame", "frame_0", "frame_1"}
    for key in ("frame", "frame_0", "frame_1"):
        assert key in hub.names()
        assert np.asarray(hub.latest(key)).ndim == 2          # a real HxW image, not a scalar
    # the default 2D panel signal aliases the FIRST trigger
    assert np.array_equal(np.asarray(hub.latest("frame")), np.asarray(hub.latest("frame_0")))


def test_default_is_single_trigger_back_compat():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    feed = CameraFrameFeed(SignalHub(), exp.camera)        # frames_per_cycle defaults to 1
    feed.step()
    assert set(feed.published_signals()) == {"frame", "frame_0"}


def test_frames_per_cycle_is_live_editable():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    feed = CameraFrameFeed(SignalHub(), exp.camera)
    assert feed.acquisition_parameters()["frames_per_cycle"] == 1
    feed.set_acquisition_parameters(frames_per_cycle=3)     # the owner-thread apply path uses this
    assert feed.frames_per_cycle == 3
    assert {"frame_0", "frame_1", "frame_2"} <= set(feed.published_signals())
