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
    cam = CameraMeasurement(hub, exp.camera, sequencer=exp.devices.sequencer)
    det = OccupancyProcessor(hub, calibration=cal, source="frame", method="box", grid_shape=(3, 4))
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


def test_console_resolves_2d_frame_panel_to_judged_frame_for_shot_alignment():
    """#1 alignment: a 2D-image panel bound to the LIVE camera ``frame`` and a site-map bound
    to ``occupied`` must show the SAME shot.  The camera and the occupancy processor are two
    independent producers (the camera streams newer frames while the processor judges an older
    one), so ``latest('frame')`` is a DIFFERENT shot than the occupancy.  The console resolves
    a frame panel to the consuming occupancy node's ``frame_judged`` (its shot-coherent frame),
    so 2D(frame) == site-map(occupied) == the judged shot.  Verified through the REAL resolver
    methods (TaskConsole._coherent_frame_signal + PanelCard._with_signal_slots)."""
    from types import SimpleNamespace
    from Zou_lab_control.frontend.task_console import TaskConsole, PanelCard

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    exp.readout.sitemap(frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    cal = exp.readout.current
    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.camera, sequencer=exp.devices.sequencer)
    det = OccupancyProcessor(hub, calibration=cal, source="frame", method="box", grid_shape=(3, 4))
    fire_live_imaging(exp)            # On Pulse: the trigger-driven camera now streams

    # the exact race: judge one frame, then the camera streams TWO newer frames before the
    # next occupancy tick -> latest('frame') is 2 shots ahead of the judged frame.
    cam.step(); det.step()
    cam.step(); cam.step()

    ns = hub.snapshot_latest()
    live_frame = np.asarray(ns["frame"])
    judged = np.asarray(ns["frame_judged"])
    assert not np.array_equal(live_frame, judged)        # precondition: the producers HAVE diverged

    # REAL console resolver: a frame consumed by a running occupancy node -> its frame_judged
    console = SimpleNamespace(running_nodes=[det])
    assert TaskConsole._coherent_frame_signal(console, "frame") == det.prefix + "frame_judged"
    # with no occupancy judging it, a standalone camera view keeps the LIVE frame
    assert TaskConsole._coherent_frame_signal(SimpleNamespace(running_nodes=[]), "frame") == "frame"

    # REAL panel resolution: a 2D-image panel bound to 'frame' now reads the judged frame
    panel = SimpleNamespace(
        config=SimpleNamespace(inputs=["frame"]),
        frame_coherence_provider=lambda n: TaskConsole._coherent_frame_signal(console, n))
    panel_signal = np.asarray(PanelCard._with_signal_slots(panel, ns)["signal"])

    assert np.array_equal(panel_signal, judged)          # 2D panel == site-map underlay (same shot)
    assert not np.array_equal(panel_signal, live_frame)  # NOT the camera's newer, misaligned frame
    # and that shot IS the one the occupancy was computed from (2D == occupied)
    assert np.array_equal(np.asarray(ns["occupied"]), cal.detect(panel_signal, method="box").occupied)


def test_two_occupancy_nodes_get_distinct_titles_prefixes_and_labels():
    """#2: adding multiple occupancy judges must make them distinguishable -- distinct row
    TITLES, distinct per-instance signal PREFIXES (so their hub signals don't collide), and
    distinct display LABELS (so the source combobox/legend tells them apart).  Verified
    through the REAL console helpers (_unique_logic_title + _logic_node_prefix) + the node
    instance_label, Qt-free."""
    from types import SimpleNamespace
    from Zou_lab_control.frontend.task_console import TaskConsole, LogicNodeConfig

    console = SimpleNamespace(logic_nodes=[], running_nodes=[])
    # two occupancy nodes added with the SAME default title -> the console must disambiguate
    t1 = TaskConsole._unique_logic_title(console, "Judge occupancy")
    cfg1 = LogicNodeConfig(kind="processor", name="Judge occupancy", title=t1)
    console.logic_nodes.append(SimpleNamespace(node=cfg1))
    t2 = TaskConsole._unique_logic_title(console, "Judge occupancy")
    cfg2 = LogicNodeConfig(kind="processor", name="Judge occupancy", title=t2)
    console.logic_nodes.append(SimpleNamespace(node=cfg2))

    assert t1 != t2                                      # distinct Logic-tab row titles
    p1 = TaskConsole._logic_node_prefix(console, cfg1)
    p2 = TaskConsole._logic_node_prefix(console, cfg2)
    assert p1 and p2 and p1 != p2 and p1.endswith("_") and p2.endswith("_")   # distinct signal prefixes

    # the built nodes then publish disjoint signal names + carry distinct labels
    a = OccupancyProcessor(SignalHub(), calibration=None, prefix=p1); a.instance_label = t1
    b = OccupancyProcessor(SignalHub(), calibration=None, prefix=p2); b.instance_label = t2
    assert set(a.published_signals()).isdisjoint(b.published_signals())
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
