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

from conftest import fire_live_imaging   # the live "On Pulse" the trigger-driven camera needs


def test_two_triggers_publish_frame_0_and_frame_1():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    cam_node = CameraMeasurement(hub, exp.camera, sequencer=exp.devices.sequencer, frames_per_cycle=2)
    fire_live_imaging(exp)            # On Pulse: the trigger-driven camera now streams
    cam_node.step()

    # ``frame`` IS the (repeat, H, W) data array (the plot reduces its repeat axis); ``frame_i`` are
    # the single per-trigger images.
    assert set(cam_node.published_signals()) == {"frame", "frame_0", "frame_1"}
    for key in ("frame_0", "frame_1"):
        assert key in hub.names()
        assert np.asarray(hub.latest(key)).ndim == 2          # a real HxW image, not a scalar
    block = np.asarray(hub.latest("frame"))
    assert block.ndim == 4 and block.shape[1] == 1            # (repeat, 1, H, W): repeat x one point x image
    # the newest data-array slice is the FIRST trigger's frame (the default 2D panel reduces -> trigger 0)
    has = np.isfinite(block).any(axis=(1, 2, 3))
    newest = block[np.flatnonzero(has)[-1], 0]
    assert np.array_equal(newest, np.asarray(hub.latest("frame_0")))


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
    cam = CameraMeasurement(hub, exp.camera, sequencer=exp.devices.sequencer)
    det = OccupancyProcessor(hub, calibration=cal, source_expr={"inputs": ["frame"], "source": "value = signal"}, method="box", grid_shape=(3, 4))
    fire_live_imaging(exp)            # On Pulse: the trigger-driven camera now streams
    for _ in range(4):
        cam.step()
        det.step()
    frame_judged = np.asarray(hub.latest("frame_judged"))
    occupied = np.asarray(hub.latest("occupied"))
    assert frame_judged.ndim == 2                                   # the readout image
    # the readout image and the rings are the SAME shot: re-judging frame_judged reproduces
    # exactly the published occupancy (they came from one atomic publish).
    assert np.array_equal(occupied, cal.detect(frame_judged, method="box").occupied)


def test_2d_frame_panel_shows_camera_average_decoupled_from_judge():
    """#H3o DECOUPLING: a 2D-image panel bound to the camera ``frame`` shows the CAMERA's OWN
    ``(repeat,1,H,W)`` block reduced by the plot's ``repeat_mode`` (average = the long-exposure mean
    that recovers all sites), INDEPENDENT of any running Judge.  A Judge (OccupancyProcessor) is a
    SEPARATE reactive node: it publishes its OWN keys (``occupied`` / ``frame_judged``) and NEVER
    rewrites ``frame``, so it cannot change what a ``frame`` panel displays.  This pins the bug where,
    with a Judge running, a 2D(frame) panel collapsed to the judged single frame and LOST the average
    -- there is no frame-coherence rewrite; a panel reads exactly the signal it is bound to."""
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    exp.readout.sitemap(frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.camera, sequencer=exp.devices.sequencer, repeat=5, free_run=True)
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame"], "source": "value = signal"}, method="box", grid_shape=(3, 4))
    fire_live_imaging(exp)
    for _ in range(6):
        cam.step(); det.step()           # the camera fills its repeat ring WHILE the Judge runs

    ns = hub.snapshot_latest()
    block = np.asarray(ns["frame"])                       # (repeat, 1, H, W) camera block
    assert block.ndim == 4 and "frame_judged" in ns       # precondition: a Judge IS running + published

    st = {"points_shape": (1,), "data_shape": tuple(block.shape[2:]), "grid_shape": ()}
    card = PanelCard(PanelConfig(kind="2d", role="plot", source="value = signal", inputs=["frame"],
                                 params={"repeat_mode": "average", "cmap": "viridis"}),
                     structure_provider=lambda name: st)
    try:
        rendered = np.asarray(card._coerce(card._signal_then_repeat(ns)))
        expected = np.nanmean(block.reshape(block.shape[0], *block.shape[2:]), axis=0)  # camera block average
        assert rendered.shape == tuple(block.shape[2:])                    # an (H, W) image
        assert np.allclose(rendered, expected, equal_nan=True)             # == the camera average
        assert not np.allclose(rendered, np.asarray(ns["frame_judged"]))   # NOT the Judge's single frame
    finally:
        card.shutdown()


def test_occupancy_falls_back_to_session_calibration_on_shape_mismatch():
    """A stale on-disk calibration (different camera ROI) must NOT wedge the readout forever.  When
    the loaded calibration does not fit the live frame, the node falls back to the session
    calibration (built from THIS camera, so it matches) and keeps publishing occupancy/rate."""
    small = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    small.readout.sitemap(frames=4, display=False); small.readout.thresholds(frames=20, display=False)
    stale_cal = small.readout.current                    # a (40,50) calibration

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (48, 60)})
    exp.readout.sitemap(frames=4, display=False); exp.readout.thresholds(frames=20, display=False)
    session_cal = exp.readout.current                    # the matching (48,60) calibration
    hub = SignalHub()
    det = OccupancyProcessor(hub, calibration=stale_cal, session_calibration=lambda: session_cal,
                             source_expr={"inputs": ["frame"], "source": "value = signal"}, grid_shape=(3, 4))
    fire_live_imaging(exp)
    img = exp.devices.camera.acquire(1, sequencer=exp.devices.sequencer)[0]
    hub.publish({"frame": np.asarray(img, dtype=float)})
    out = det.step()                                     # stale cal mismatches (48,60) -> falls back
    assert "occupied" in out and "rate" in out           # readout keeps flowing (not wedged)
    assert det.calibration is session_cal                # adopted the matching one for next shots
    small.close(); exp.close()


def test_occupancy_nodes_publish_short_names_and_disambiguate_on_collision():
    """#7/#2: a logic node publishes its SHORT natural signal names (``occupied`` / ``rate`` /
    ... -- NOT a verbose ``judge_occupancy_rate``); the producer is shown by the signal-flow
    grouping + frame-title legend, not baked into every name.  A SECOND node whose outputs would
    COLLIDE with an already-running node's signals gets a disambiguating prefix so their hub
    signals stay disjoint.  Distinct row TITLES + distinct display LABELS too.  Verified through
    the REAL console helpers (_unique_logic_title + _logic_node_prefix), Qt-free."""
    from types import SimpleNamespace
    from Zou_lab_control.frontend.task_console import TaskConsole, LogicNodeConfig

    keys = ("occupied", "counts", "rate", "rate_sites", "rate_grid", "centers", "frame_judged")
    spec = SimpleNamespace(result_keys=keys)
    console = SimpleNamespace(logic_nodes=[], running_nodes=[], _spec_for_logic=lambda n: spec)

    t1 = TaskConsole._unique_logic_title(console, "Judge occupancy")
    cfg1 = LogicNodeConfig(kind="processor", name="Judge occupancy", title=t1)
    console.logic_nodes.append(SimpleNamespace(node=cfg1))
    # FIRST node, nothing running yet -> NO prefix: it publishes the bare short names.
    p1 = TaskConsole._logic_node_prefix(console, cfg1)
    assert p1 == ""
    a = OccupancyProcessor(SignalHub(), calibration=None, prefix=p1); a.instance_label = t1
    assert "rate" in a.published_signals() and "occupied" in a.published_signals()   # short names
    console.running_nodes.append(a)                       # now it's publishing the bare names

    # SECOND node added while the first RUNS -> its keys collide -> disambiguating prefix.
    t2 = TaskConsole._unique_logic_title(console, "Judge occupancy")
    cfg2 = LogicNodeConfig(kind="processor", name="Judge occupancy", title=t2)
    console.logic_nodes.append(SimpleNamespace(node=cfg2))
    p2 = TaskConsole._logic_node_prefix(console, cfg2)
    assert t1 != t2 and p2 and p2.endswith("_") and p2 != p1   # distinct title + disambiguating prefix
    b = OccupancyProcessor(SignalHub(), calibration=None, prefix=p2); b.instance_label = t2
    assert set(a.published_signals()).isdisjoint(b.published_signals())   # disjoint hub signals
    assert a.display_label != b.display_label


def test_frames_per_cycle_is_live_editable():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    cam_node = CameraMeasurement(SignalHub(), exp.camera)
    assert cam_node.acquisition_parameters()["frames_per_cycle"] == 1
    cam_node.set_acquisition_parameters(frames_per_cycle=3)     # the owner-thread apply path uses this
    assert cam_node.frames_per_cycle == 3
    assert {"frame_0", "frame_1", "frame_2"} <= set(cam_node.published_signals())


def test_region_is_always_exposed_even_at_full_frame():
    """A frame panel's Edit shows the camera's ROI (``region``) so the operator can crop it --
    even with NO sub-array set (``roi is None``).  At full frame ``region`` is the FULL sensor
    endpoints ``[0, W, 0, H]`` (from ``camera.sensor_shape``), so the field always exists for
    editing AND the plot's area-select / zoom writeback has a field to fill.  (Regression: when
    ``region`` was emitted only for a set ROI, a full-frame camera showed no ROI field at all.)"""
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    assert exp.camera.roi is None                                  # default: full frame, no sub-array
    assert tuple(exp.camera.sensor_shape) == (40, 50)              # (height, width)
    cam_node = CameraMeasurement(SignalHub(), exp.camera)
    params = cam_node.acquisition_parameters()
    assert params["region"] == [0, 50, 0, 40]                      # full sensor endpoints [x0,x1,y0,y1]
    # and the endpoints round-trip back to a real sub-array when applied (crop on Apply)
    cam_node.set_acquisition_parameters(region=[10, 30, 5, 25])
    assert exp.camera.roi is not None


def test_camera_exposure_setter_round_trips_on_every_backend():
    """`cam.exposure = v` works uniformly (the one intrinsic scalar that reads naturally as `=`).
    It is a write-through to configure() -- the single write path -- so it sets exposure and
    nothing else.  The base CameraDevice contract carries the setter; both concrete backends
    (VirtualCamera, QCMOSCamera) honour it.  (Guards the property/method-style alignment: the
    setter must not silently regress to getter-only on any backend.)"""
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera

    assert CameraDevice.exposure.fset is not None                  # the contract declares a setter

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    exp.camera.exposure = 5e-3                                       # virtual backend
    assert exp.camera.exposure == 5e-3

    cam = QCMOSCamera()                                             # NOT opened -> configure skips DCAM
    roi_before = cam.roi
    cam.exposure = 4e-3                                             # real backend, write-through configure
    assert cam.exposure == 4e-3
    assert cam.roi == roi_before                                   # exposure '=' touches ONLY exposure
