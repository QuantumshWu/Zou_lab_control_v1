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
    det = OccupancyProcessor(hub, calibration=cal, source="frame", grid_shape=(3, 4))
    try:
        # the camera measurement is just that -- a camera; it does NOT detect.
        assert "frame" not in det.published_signals()
        assert "occupied" in det.published_signals()

        # reactive: with no frame yet, the processor no-ops (publishes nothing)
        assert det.step() == {}

        # camera publishes ONLY a frame (the monolith published 9 signals here)
        named = cam.step()
        assert set(k for k in named) == {"frame", "frame_0"}
        assert "occupied" not in hub.names()

        # detect consumes the frame and produces occupancy through the REAL contract
        det.step()
        names = set(hub.names())
        assert {"occupied", "counts", "rate", "rate_sites", "centers", "thresholds"} <= names
        occ = hub.latest("occupied")
        assert occ.shape == (12,)
        assert np.ndim(hub.latest("rate")) == 0

        # virtual == real: the published occupancy IS calibration.detect on that frame
        frame = hub.latest("frame")
        expected = np.asarray(cal.detect(frame).occupied, dtype=float).reshape(-1)
        np.testing.assert_array_equal(occ, expected)

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
        cal_path = tmp_path / "cal.npz"
        task = CalibrateReadoutTask(
            hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=(3, 4),
            calibration_frames=4, threshold_frames=20, mode="box", save_path=str(cal_path))
        task.run_to_completion()

        assert task.finished and task.calibration is not None
        # A TASK PUBLISHES NOTHING TO THE HUB -- the hub is measurements + processors
        # only.  Mid-run output went to the task's OWN buffer; the result to the instance.
        assert hub.names() == []
        assert "frame" in task.output.names() and task.output.progress == 1.0  # mid-run buffer
        assert {"centers", "thresholds", "n_sites"} <= set(task.result)         # result on instance
        assert task.result["n_sites"] == 12
        # artifact persisted
        saved = np.load(cal_path)
        assert saved["centers"].shape == (12, 2) and saved["thresholds"].shape == (12,)

        # composition: the task's calibration drives an OccupancyProcessor on live frames
        # -- THAT (a processor) is what lands occupancy on the hub.
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        det = OccupancyProcessor(hub, calibration=task.calibration, grid_shape=(3, 4))
        cam.step()
        det.step()
        assert hub.latest("occupied").shape == (12,)
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
            calibration_frames=4, threshold_frames=20, mode="box", prefix="cal_")
        task.run_to_completion()
        assert task.calibration is not None

        # User composition step 2: a camera Measurement (publishes raw frames).
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        # User composition step 3: an OccupancyProcessor running the REAL contract.
        det = OccupancyProcessor(hub, calibration=task.calibration, grid_shape=(3, 4))

        cam.step()                                    # publishes frame
        det.step()                                    # consumes frame -> real detect
        names = set(hub.names())
        assert "frame" in names and {"occupied", "rate", "centers"} <= names
        occ = hub.latest("occupied")
        assert occ.shape == (12,)
        frame = hub.latest("frame")
        expected = np.asarray(task.calibration.detect(frame).occupied, dtype=float).reshape(-1)
        np.testing.assert_array_equal(occ, expected)
    finally:
        cam.stop()
        det.stop()
        exp.close()


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
            calibration_frames=4, threshold_frames=20, prefix="cal_")
        assert "frame" not in hub.names()              # nothing live yet
        task.run_to_completion()                       # runs the calibrate task
        # The task put NOTHING on the hub -- mid-run frame + progress are in its buffer.
        assert hub.names() == []
        assert "frame" in task.output.names() and task.output.progress == 1.0

        # The user adds the live camera measurement separately -- it publishes ``frame``
        # on the hub; the task's buffered frame never collided with it (no ``cal_*``).
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        cam.step()
        assert "frame" in hub.names()
        assert not any(n.startswith("cal_") for n in hub.names())
    finally:
        exp.close()
