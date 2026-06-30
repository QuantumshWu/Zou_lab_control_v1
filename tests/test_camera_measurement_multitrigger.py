"""Contract: CameraMeasurement publishes ONE signal per emCCD event of the cycle.

A cycle with ``frames_per_cycle`` camera triggers publishes ``frame_0 … frame_{N-1}``,
and NOTHING else (no lumped ``frame``).  Each ``frame_i`` is THAT event's own
``(repeat, 1, H, W)`` repeat block -- so a panel bound to ``frame_i`` reduces its repeat
axis (repeat_mode: average = the long-exposure mean of THAT emCCD event), the only way a
multi-event cycle can show the repeat_mode effect for a chosen event.

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
    cam_node = CameraMeasurement(hub, exp.camera, sequencer=exp.devices.sequencer,
                                 frames_per_cycle=2, repeat=4)
    fire_live_imaging(exp)            # On Pulse: the trigger-driven camera now streams
    for _ in range(2):
        cam_node.step()

    # ONE signal per emCCD event, each its OWN (repeat, 1, H, W) block; NO lumped ``frame``.
    assert set(cam_node.published_signals()) == {"frame_0", "frame_1"}
    b0 = np.asarray(hub.latest("frame_0"))
    b1 = np.asarray(hub.latest("frame_1"))
    for b in (b0, b1):
        assert b.shape == (4, 1, 40, 50)                       # (repeat, 1, H, W) block per event
    # the two emCCD events are DISTINCT images, each stacked across repeats in its own ring
    assert not np.array_equal(np.nan_to_num(b0), np.nan_to_num(b1))


def test_default_is_single_event_frame_0_only():
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    cam_node = CameraMeasurement(SignalHub(), exp.camera)        # frames_per_cycle defaults to 1
    cam_node.step()
    assert set(cam_node.published_signals()) == {"frame_0"}      # one event -> just frame_0; no lumped frame


def test_camera_frame_keys_is_the_single_source_for_published_and_declared_names():
    """The console's declared-signal picker (a not-yet-started camera row) and the running camera's
    ``published_signals`` must offer the EXACT same ``frame_i`` set -- so a 'waiting' name in the picker
    is always a name the camera will really emit (the #2 phantom 'frame' bug was a second, drifted
    source).  Both go through ``camera_frame_keys``; pin that they agree for every frames_per_cycle."""
    from Zou_lab_control.neutral_atom.operations.logic import camera_frame_keys
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    for n in (1, 2, 3):
        cam = CameraMeasurement(SignalHub(), exp.camera, frames_per_cycle=n)
        assert set(camera_frame_keys(n)) == set(cam.published_signals())     # helper == live publish
        assert "frame" not in camera_frame_keys(n)                            # never the lumped residue
        assert camera_frame_keys(n) == [f"frame_{i}" for i in range(n)]       # ordered, frame_0..frame_{n-1}


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
    det = OccupancyProcessor(hub, calibration=cal, source_expr={"inputs": ["frame_0"], "source": "value = signal"}, method="box", grid_shape=(3, 4))
    fire_live_imaging(exp)            # On Pulse: the trigger-driven camera now streams
    for _ in range(4):
        cam.step()
        det.step()
    frame_judged = np.asarray(hub.latest("frame_judged"))
    occupied = np.asarray(hub.latest("occupied"))
    # uniform (repeat, data_points=1, *data): frame_judged (repeat,1,H,W), occupied (repeat,1,n_sites)
    assert frame_judged.ndim == 4 and occupied.ndim == 3 and frame_judged.shape[1] == occupied.shape[1] == 1
    # the readout image and the rings are the SAME shots: re-judging each frame_judged slice
    # reproduces exactly the published occupancy slice (they came from one atomic publish).
    img = frame_judged[-1, 0]                                       # the last filled shot's (H, W) frame
    assert np.array_equal(occupied[-1, 0], cal.detect(img, method="box").occupied)


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
    cam = CameraMeasurement(hub, exp.camera, sequencer=exp.devices.sequencer, repeat=5)  # 0=∞; 5 = a 5-deep block
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame_0"], "source": "value = signal"}, method="box", grid_shape=(3, 4))
    fire_live_imaging(exp)
    for _ in range(6):
        cam.step(); det.step()           # the camera fills its repeat ring WHILE the Judge runs

    ns = hub.snapshot_latest()
    block = np.asarray(ns["frame_0"])                     # (repeat, 1, H, W) camera block
    assert block.ndim == 4 and "frame_judged" in ns       # precondition: a Judge IS running + published

    st = {"points_shape": (1,), "data_shape": tuple(block.shape[2:]), "grid_shape": ()}
    card = PanelCard(PanelConfig(kind="2d", role="plot", source="value = signal", inputs=["frame_0"],
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
                             source_expr={"inputs": ["frame_0"], "source": "value = signal"}, grid_shape=(3, 4))
    fire_live_imaging(exp)
    img = exp.devices.camera.acquire(1, sequence=getattr(exp.devices.sequencer, "firing", None))[0]
    hub.publish({"frame_0": np.asarray(img, dtype=float)})
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

    keys = ("occupied", "counts", "rate", "centers", "thresholds", "frame_judged")
    spec = SimpleNamespace(result_keys=keys)
    # _logic_node_prefix checks collisions against EVERY live hub signal (#2) -> the mock console needs a
    # hub; an empty one suffices here (the running node `a` below is detected via running_nodes).
    console = SimpleNamespace(logic_nodes=[], running_nodes=[], hub=SignalHub(), _spec_for_logic=lambda n: spec)

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


def test_arm_then_fire_is_the_single_on_armed_factory():
    """``arm_then_fire`` is the ONE arm-before-fire ``on_armed`` factory every readout/scan/capture
    path uses (no hand-copied ``lambda: (seq.prepare(s), seq.fire(s))`` per call site).  Pin its two
    behaviours: None sequencer -> None callback (a notebook readout that leans on the virtual atom
    array fires nothing); a real sequencer -> a callable that prepares THEN fires the SAME sequence."""
    from Zou_lab_control.neutral_atom.devices.base import arm_then_fire

    assert arm_then_fire(None, object()) is None          # no bound sequencer -> nothing to fire

    calls: list[tuple[str, object]] = []

    class _Spy:
        def prepare(self, s): calls.append(("prepare", s))
        def fire(self, s=None): calls.append(("fire", s))

    seq = object()
    cb = arm_then_fire(_Spy(), seq)
    assert callable(cb)
    cb()
    assert calls == [("prepare", seq), ("fire", seq)]      # prepare THEN fire, both on this sequence


def test_no_call_site_hand_copies_the_arm_then_fire_lambda():
    """Mechanical single-source guard: the arm-before-fire idiom must route through
    ``devices.base.arm_then_fire`` -- not be re-typed as an inline ``prepare(...)...fire(...)`` lambda.
    Scans the analysis/subsystem/device modules that orchestrate shots; the ONLY allowed occurrence
    of a ``prepare(...).fire(...)`` pair is the helper's own one-line body."""
    import re
    na_root = REPO_ROOT / "Zou_lab_control" / "neutral_atom"
    pattern = re.compile(r"\.prepare\([^\n]*\)[^\n]*\.fire\(")
    offenders: list[str] = []
    for py in na_root.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                # the helper body in devices/base.py is the single legitimate occurrence
                if py.name == "base.py" and "return lambda:" in line:
                    continue
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, "hand-copied arm-then-fire lambda; use arm_then_fire():\n" + "\n".join(offenders)


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
