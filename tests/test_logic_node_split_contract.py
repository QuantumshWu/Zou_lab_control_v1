"""Contract: the loading readout is COMPOSED of separate logic nodes by the user,
not one monolithic node fabricating every signal -- and detection runs the REAL
pipeline (virtual==real).

The camera Measurement (`CameraMeasurement`) publishes ONLY ``frame``.  A separate
`OccupancyProcessor` consumes ``frame`` and runs the SAME ``calibration.detect``
contract the notebook/real readout uses to publish per-site occupancy + rate.  The
calibration comes from a `CalibrateReadoutTask` running the real sitemap/threshold
path.  This is the virtual==real split: only the camera frame is simulated; site
detection is the production code path, as a distinct graph node.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from conftest import fire_live_imaging   # the live "On Pulse" the trigger-driven camera needs


def test_camera_measurement_plus_detect_processor_runs_real_pipeline():
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, OccupancyProcessor

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    # calibration via the REAL path (site centers detected from frames; per-site thresholds learned)
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=24, display=False)
    cal = exp.readout.current
    assert cal is not None and cal.thresholds is not None

    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
    det = OccupancyProcessor(hub, calibration=cal, source_expr={"inputs": ["frame_0"], "source": "value = signal"}, grid_shape=(3, 4))
    try:
        fire_live_imaging(exp)            # On Pulse: the trigger-driven camera now streams
        # the camera measurement is just that -- a camera; it does NOT detect.
        assert "frame_0" not in det.published_signals()
        assert "occupied" in det.published_signals()

        # reactive: with no frame yet, the processor no-ops (publishes nothing)
        assert det.step() == {}

        # camera publishes ONE block per emCCD event (frames_per_cycle=1 -> just frame_0)
        named = cam.step()
        assert set(k for k in named) == {"frame_0"}
        assert "occupied" not in hub.names()

        # detect consumes the frame and produces occupancy through the REAL contract
        det.step()
        names = set(hub.names())
        assert {"occupied", "counts", "rate", "centers", "thresholds"} <= names
        occ = np.asarray(hub.latest("occupied"))
        assert occ.ndim == 3 and occ.shape[1:] == (1, 12)   # (R,P,*data_shape), P=1, data_shape=(n_sites,)
        assert np.asarray(hub.latest("rate")).shape == (1, 1, 1)

        # virtual == real: the published occupancy IS calibration.detect on that frame.  ``frame`` is
        # the camera's (repeat,1,H,W) block, so judge its last filled slice and compare per shot.
        img = np.asarray(hub.latest("frame_0"))[-1, 0]
        expected = np.asarray(cal.detect(img).occupied, dtype=float).reshape(-1)
        np.testing.assert_array_equal(occ[-1, 0], expected)   # (repeat, 1, n_sites) -> last shot's site vector

        # reactive again: stepping detect with no new frame must not republish
        v0 = hub.signal_versions().get("occupied")
        det.step()
        assert hub.signal_versions().get("occupied") == v0
    finally:
        exp.close()


def test_calibrate_task_produces_calibration_and_drives_detect_processor(tmp_path):
    """The calibrate-readout Task runs the REAL sitemap+threshold path, emits mid-run
    output (intermediate frame + progress) to its OWN buffer (NOT the hub), saves an
    npz artifact, keeps the result on the instance, and yields a calibration a
    OccupancyProcessor then consumes -- the whole loading readout composed BY THE USER
    from device + task + processor (no monolithic node)."""
    import numpy as np
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import (
        CalibrateReadoutTask, CameraMeasurement, OccupancyProcessor)

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    try:
        task = CalibrateReadoutTask(
            hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=(3, 4),
            threshold_frames=20, folder=str(tmp_path / "cal"))
        task.run_to_completion()

        assert task.finished and task.calibration is not None
        # A TASK PUBLISHES NOTHING TO THE HUB -- the hub is measurements + processors
        # only.  Mid-run output went to the task's OWN buffer; the result to the instance.
        assert hub.names() == []
        assert "frame" in task.output.names() and task.output.progress == 1.0  # mid-run buffer
        assert {"centers", "thresholds", "n_sites"} <= set(task.result)         # result on instance
        assert task.result["n_sites"] == 12
        # artifact persisted: the report's numeric bundle in the explicit run folder
        saved = np.load(Path(task.result["report_dir"]) / "calibration.npz")
        assert saved["centers"].shape == (12, 2) and saved["thresholds"].shape == (12,)

        # composition: the task's calibration drives an OccupancyProcessor on live frames
        # -- THAT (a processor) is what lands occupancy on the hub.
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        det = OccupancyProcessor(hub, calibration=task.calibration, grid_shape=(3, 4))
        fire_live_imaging(exp)            # On Pulse: the trigger-driven camera now streams
        cam.step()
        det.step()
        occ = np.asarray(hub.latest("occupied"))
        assert occ.ndim == 3 and occ.shape[1:] == (1, 12)     # (R,P,*data_shape), P=1, data_shape=(n_sites,)
    finally:
        exp.close()


def test_user_composed_loading_readout_streams_real_detect_off_camera_frames():
    """The user composes the loading readout from independent nodes -- a camera
    Measurement publishing ``frame`` + an OccupancyProcessor turning ``frame`` into
    occupancy/rate (via the REAL calibration.detect) -- and the live chain is
    camera -> frame -> real detect (virtual == real).  No monolithic node fabricates
    every signal; the user (notebook or task console) wires the three primitives."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import (
        CalibrateReadoutTask, CameraMeasurement, OccupancyProcessor)

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    try:
        # User composition step 1: run the calibrate task to get a TrapCalibration.
        # (The task publishes its mid-run output under its own cal_ prefix so its
        # template frames never clobber the live frame the user will stream next.)
        task = CalibrateReadoutTask(
            hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=(3, 4),
            threshold_frames=20, prefix="cal_")
        task.run_to_completion()
        assert task.calibration is not None

        # User composition step 2: a camera Measurement (publishes raw frames).
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        # User composition step 3: an OccupancyProcessor running the REAL contract.
        det = OccupancyProcessor(hub, calibration=task.calibration, grid_shape=(3, 4))

        fire_live_imaging(exp)                        # On Pulse: the trigger-driven camera now streams
        cam.step()                                    # publishes frame
        det.step()                                    # consumes frame -> real detect
        names = set(hub.names())
        assert "frame_0" in names and {"occupied", "rate", "centers"} <= names
        occ = np.asarray(hub.latest("occupied"))
        assert occ.ndim == 3 and occ.shape[1:] == (1, 12)     # (R,P,*data_shape), P=1, data_shape=(n_sites,)
        img = np.asarray(hub.latest("frame_0"))[-1, 0]    # judge the camera block's last filled slice
        expected = np.asarray(task.calibration.detect(img).occupied, dtype=float).reshape(-1)
        np.testing.assert_array_equal(occ[-1, 0], expected)   # (repeat, 1, n_sites) -> last shot's site vector
    finally:
        cam.stop()
        det.stop()
        exp.close()


def test_oneshot_task_marks_finished_even_when_run_raises():
    """A one-shot Task whose ``run`` RAISES must still end terminated: ``finished``
    means 'this one-shot has RUN ONCE' (success OR failure), so it is set in a
    ``finally`` alongside the stop event.  Otherwise a failed task stays
    ``finished=False`` forever and the console keeps the dashboard locked (finding 7).
    The exception still propagates to a headless ``step()`` caller (finally must not
    swallow it); failure is expressed by ``result`` staying empty."""
    import pytest

    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import Task

    class _BoomTask(Task):
        def run(self, out):
            raise RuntimeError("boom")

    hub = SignalHub()
    node = _BoomTask(hub)
    with pytest.raises(RuntimeError, match="boom"):
        node.shot()
    assert node.finished is True            # terminated despite the failure -> console can unlock
    assert node._stop.is_set()              # one-shot: the loop will not retry
    assert node.result == {}                # failure expressed by the empty result


def test_oneshot_processor_marks_finished_even_when_run_raises():
    """Same lifecycle invariant for the discrete ProcessorRun sibling: a ``spec.run``
    that RAISES sets ``finished=True`` in a ``finally`` (it ran once, no retry), the
    exception propagates, and the per-shot publish never runs (so nothing partial is
    pushed onto the hub)."""
    import pytest

    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import ProcessorRun

    class _BoomSpec:
        name = "boom"
        result_keys = ("value",)

        def run(self, ctx):
            raise RuntimeError("kaboom")

    hub = SignalHub()
    node = ProcessorRun(hub, _BoomSpec(), readout=None)
    with pytest.raises(RuntimeError, match="kaboom"):
        node.shot()
    assert node.finished is True            # terminated despite the failure -> console can unlock
    assert node._stop.is_set()
    assert hub.names() == []                # the failed shot published nothing partial


def test_calibrate_task_output_stays_off_the_hub():
    """The calibrate task's mid-run output goes to its OWN buffer (``task.output``),
    NEVER the hub -- so it can never clobber, or masquerade as, a LIVE ``frame`` the
    camera measurement publishes.  The hub carries ONLY measurement + processor
    outputs; a dedicated 'calibrating' panel watches ``task.output`` while the live
    panel watches the hub's ``frame``."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import (
        CalibrateReadoutTask, CameraMeasurement)

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    try:
        task = CalibrateReadoutTask(
            hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=(3, 4),
            threshold_frames=20, prefix="cal_")
        assert "frame_0" not in hub.names()              # nothing live yet
        task.run_to_completion()                       # runs the calibrate task
        # The task put NOTHING on the hub -- mid-run frame + progress are in its buffer.
        assert hub.names() == []
        assert "frame" in task.output.names() and task.output.progress == 1.0

        # The user adds the live camera measurement separately -- it publishes ``frame``
        # on the hub; the task's buffered frame never collided with it (no ``cal_*``).
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        fire_live_imaging(exp)            # On Pulse: the trigger-driven camera now streams
        cam.step()
        assert "frame_0" in hub.names()
        assert not any(n.startswith("cal_") for n in hub.names())
    finally:
        exp.close()
