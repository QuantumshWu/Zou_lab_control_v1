"""Contract (#shot-clock): the task console shows ONE coherent shot across ALL panels.

The user's hard requirement: with two 2-D images on ``frame_0`` / ``frame_2`` and a sitemap fed by
``frame_1`` -> occupancy, the three displays must read data from the SAME acquisition shot -- never
staggered ("同一个clock周期，不要错位").  The camera and the reactive occupancy run on SEPARATE
threads at their own rates, so the camera's raw frames race AHEAD of the occupancy's derived
``occupied`` / ``frame_judged``.  Reading each signal at its OWN latest (``snapshot_latest``) therefore
MIS-aligns the raw frames against the occupancy.

The fix is a GLOBAL display shot clock: each render tick the console picks the newest source-shot that
every DISPLAYED, still-running-producer signal has reached (the ``min`` of their provenance ids) and
reads :meth:`SignalHub.snapshot_at` there -- so a fast camera frame is HELD BACK to match the slower
occupancy and all panels show one shot.  Crucially this holds back ONLY relative to OTHER co-displayed
producers: a LONE camera panel's ``min`` is its own latest, so it stays fully live (no repeat stutter).

These guards pin:
  (A) hub-level: after the occupancy lags the camera, ``snapshot_latest`` is mis-aligned but
      ``snapshot_at(min provenance)`` resolves the raw frames AND the occupancy to one shot;
  (B) console-level: ``_display_shot`` is the ``min`` over bound-AND-running signals, and the
      expression namespace holds the camera back to the lagging occupancy (all three panels one shot);
  (C) anti-oscillation: a LONE camera panel is NOT held back -- its display shot is the camera's own
      latest, so repeat / repeat_mode stay live (the property the per-panel clock removal chased).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.core.signals import SignalHub, NO_LINEAGE
from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, OccupancyProcessor

from conftest import fire_live_imaging   # the live "On Pulse" the trigger-driven camera needs


def _calibrated(grid=(3, 4)):
    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (40, 50)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    return exp


# ----------------------------------------------------------------------------- (A) hub level
def test_snapshot_at_aligns_raw_frames_with_lagging_occupancy():
    """The architecture-level proof: a camera publishing ``frame_0/1/2`` per shot, and an occupancy
    consuming ``frame_1``, are driven so the occupancy LAGS two shots behind the camera.  Then
    ``snapshot_latest`` mixes shots (frame at S+2, occupancy at S) -- the misalignment -- while
    ``snapshot_at(min provenance)`` hands back the SAME shot for every signal (the raw frames held
    back to the occupancy's shot)."""
    exp = _calibrated()
    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                            frames_per_cycle=3, repeat=0)
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame_1"], "source": "value = signal"},
                             method="box", grid_shape=(3, 4))
    try:
        fire_live_imaging(exp)
        s1 = cam.step()                 # frames @ shot S1
        out = det.step()                # occupancy consumes frame_1 @ S1 -> occupied/frame_judged @ S1
        assert "occupied" in out and "frame_judged" in out
        cam.step(); cam.step()          # frames advance to S2, S3 -- occupancy NOT stepped -> still S1

        p_frame = hub.latest_provenance("frame_0")
        p_occ = hub.latest_provenance("occupied")
        assert p_frame > p_occ          # the camera raced ahead of the reactive occupancy (a real lag)

        # the bug: reading each at its OWN latest mixes shots (frame @ S3, occupancy @ S1).
        latest = hub.snapshot_latest()
        assert not np.array_equal(latest["frame_0"], s1["frame_0"])    # latest frame is the NEW shot

        # the fix: snapshot_at(min) = the newest shot BOTH reached = the occupancy's shot (S1).
        target = min(hub.latest_provenance(n) for n in ("frame_0", "frame_2", "occupied", "frame_judged"))
        assert target == p_occ
        snap = hub.snapshot_at(target)
        # raw frames are HELD BACK to S1 (the captured S1 block), not their S3 latest.
        assert np.array_equal(snap["frame_0"], s1["frame_0"])
        assert np.array_equal(snap["frame_2"], s1["frame_2"])
        assert not np.array_equal(snap["frame_0"], hub.latest("frame_0"))
        # occupancy IS its own latest (it never advanced past S1) -> the whole board is coherent at S1.
        assert np.array_equal(snap["occupied"], hub.latest("occupied"))
        assert np.array_equal(snap["frame_judged"], hub.latest("frame_judged"))
    finally:
        exp.close()


def test_snapshot_at_is_cheap_with_deep_no_lineage_rings():
    """Perf guard (#shot-clock froze the GUI): snapshot_at runs EVERY display tick, so it must be cheap
    even with full (2048-deep) rings.  The freeze was an O(n^2) scan -- a free-running NO_LINEAGE signal
    never early-broke, walking its whole ring via deque INDEXING (O(n) per element).  With several such
    signals at 10 Hz that wedged PyQt.  The fix skips NO_LINEAGE signals and walks via reversed iterators
    (O(1)/step).  A generous bound (the regressed cost is ~1000x over it) keeps this non-flaky."""
    import time
    hub = SignalHub()
    for _ in range(2100):                              # overflow the 2048 history -> full rings
        sid = hub.next_source_shot()
        hub.publish({"frame_0": np.zeros((1, 1, 8, 8))}, provenance=sid)
        hub.publish({"rate": 0.5, "counts": np.zeros(12), "centers": np.zeros((12, 2)),
                     "thresholds": np.zeros(12)}, provenance=None)   # 4 NO_LINEAGE deep rings
    target = hub.latest_provenance("frame_0") - 1      # an OLD shot: the held-back-camera worst case
    t = time.perf_counter()
    for _ in range(50):
        hub.snapshot_at(target)
    per_call_ms = (time.perf_counter() - t) / 50 * 1000
    assert per_call_ms < 20.0, f"snapshot_at too slow ({per_call_ms:.2f} ms) -- the O(n^2) ring scan regressed"


# ----------------------------------------------------------------------------- console helpers
@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _console(exp):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    console = TaskConsole(
        hub=SignalHub(), state=default_console_state(), session=exp,
        measurements=exp.readout.measurement_specs(), processors=exp.readout.processor_specs(),
        tasks=exp.readout.task_specs(), window_px=(1200, 800))
    console._timer.stop()
    return console


def _add_2d(console, inputs):
    """Add a 2-D plot card bound to ``inputs`` (kind is irrelevant to the shot clock -- it reads
    ``config.source`` + ``config.inputs`` via _card_reads, so a 2-D card bound to ``occupied`` stands
    for the sitemap)."""
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == "2d")
    kc.setCurrentIndex(i)
    console._add_panel()
    card = console.cards[-1]
    card.config.source = "value = signal"
    card.config.inputs = list(inputs)
    return card


# ----------------------------------------------------------------------------- (B) console: hold to occupancy
def test_console_display_shot_holds_camera_to_lagging_occupancy():
    """The console picks ONE display shot = ``min`` over bound-AND-running signals, so a board of
    {frame_0 2D, frame_2 2D, occupancy sitemap} reads all three at the occupancy's (lagging) shot --
    the camera frames are held back so nothing staggers."""
    exp = _calibrated()
    console = _console(exp)
    hub = console.hub
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                            frames_per_cycle=3, repeat=0)
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame_1"], "source": "value = signal"},
                             method="box", grid_shape=(3, 4))
    try:
        _add_2d(console, ["frame_0"])
        _add_2d(console, ["frame_2"])
        _add_2d(console, ["occupied"])           # stands for the sitemap (binds occupied)
        console.running_nodes = [cam, det]
        fire_live_imaging(exp)

        s1 = cam.step()
        det.step()                               # occupancy @ S1
        cam.step(); cam.step()                   # camera -> S3, occupancy still S1

        target = console._display_shot()
        assert target == hub.latest_provenance("occupied")          # min over bound+live = the lagging occupancy
        assert target < hub.latest_provenance("frame_0")            # strictly behind the camera (held back)

        ns = console._expression_namespace()
        # every displayed signal resolves to the ONE coherent shot S1 (no stagger).
        assert np.array_equal(ns["frame_0"], s1["frame_0"])
        assert np.array_equal(ns["frame_2"], s1["frame_2"])
        assert np.array_equal(ns["occupied"], hub.latest("occupied"))
        assert np.array_equal(ns["frame_judged"], hub.latest("frame_judged"))
        # and it is NOT just the camera's latest (that would be the misalignment).
        assert not np.array_equal(ns["frame_0"], hub.latest("frame_0"))
    finally:
        console.shutdown()
        exp.close()


# --------------------------------------------------------------- (B') faulted node excluded from the clock
def test_display_shot_excludes_a_faulted_nodes_frozen_signal():
    """A running-but-FAULTED analysis node (``consecutive_errors > 0``) whose provenance froze at an
    old shot must NOT pin the whole board's display shot -- otherwise ONE dead node freezes EVERY panel
    (#4-C).  This is the SAME principle as the existing STOPPED-node exclusion, only for a node that is
    still 'running' but erroring.  A slow-but-HEALTHY node stays in the min (coherence over freshness);
    only a faulted one is dropped."""
    exp = _calibrated()
    console = _console(exp)
    hub = console.hub
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer,
                            frames_per_cycle=3, repeat=0)
    det = OccupancyProcessor(hub, calibration=exp.readout.current,
                             source_expr={"inputs": ["frame_1"], "source": "value = signal"},
                             method="box", grid_shape=(3, 4))
    try:
        _add_2d(console, ["frame_0"])
        _add_2d(console, ["occupied"])           # the faulted node's displayed signal
        console.running_nodes = [cam, det]
        fire_live_imaging(exp)

        cam.step()
        det.step()                               # occupancy @ S1
        cam.step(); cam.step()                   # camera -> S3, occupancy frozen at S1

        # HEALTHY lag: the board is held back to the occupancy's older shot (coherence).
        assert console._display_shot() == hub.latest_provenance("occupied")

        # The occupancy node now FAULTS (its provenance stays frozen at S1).  Its signals leave the
        # min set, so the clock follows the healthy camera instead of freezing at the dead node's shot.
        det.consecutive_errors = 4
        det.last_error = "ValueError: boom"
        assert console._display_shot() == hub.latest_provenance("frame_0")   # follows the live camera
        assert console._display_shot() != hub.latest_provenance("occupied")  # not pinned to the dead node
    finally:
        console.shutdown()
        exp.close()


# ----------------------------------------------------------------------------- (C) anti-oscillation
def test_lone_camera_panel_is_not_held_back():
    """A LONE camera panel (no slower co-displayed producer) is NOT held back: its display shot is the
    camera's OWN latest, so a repeat / repeat_mode accumulation stays fully live.  This is the property
    the whole-board min must NOT break -- the hold appears only when a slower producer is shown too."""
    exp = _calibrated()
    console = _console(exp)
    hub = console.hub
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer, repeat=0)
    try:
        _add_2d(console, ["frame_0"])
        console.running_nodes = [cam]
        fire_live_imaging(exp)
        cam.step(); cam.step(); cam.step()       # frames advance S1..S3 with nothing lagging

        assert console._display_shot() == hub.latest_provenance("frame_0")   # its OWN latest -> live
        ns = console._expression_namespace()
        assert np.array_equal(ns["frame_0"], hub.latest("frame_0"))          # newest block, not held back
    finally:
        console.shutdown()
        exp.close()


# ----------------------------------------------------------------------------- scalar-only / no lineage
def test_free_running_scalar_does_not_constrain_the_clock():
    """A NO_LINEAGE signal (a free-running loading rate) never constrains the display shot -- a board
    of only such signals has no coherent shot to hold (``_display_shot`` -> None == snapshot_latest)."""
    exp = _calibrated()
    console = _console(exp)
    hub = console.hub
    try:
        card = _add_2d(console, ["rate"])
        # a node that publishes only a free-running scalar (no provenance)
        hub.publish({"rate": 0.5}, provenance=None)
        assert hub.latest_provenance("rate") == NO_LINEAGE

        from types import SimpleNamespace
        console.running_nodes = [SimpleNamespace(published_signals=lambda: frozenset({"rate"}))]
        assert console._display_shot() is None        # nothing with a lineage -> no hold
    finally:
        console.shutdown()
        exp.close()
