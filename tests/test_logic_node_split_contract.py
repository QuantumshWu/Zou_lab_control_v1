"""Contract: the loading readout is COMPOSED of separate logic nodes by the user,
not one monolithic node fabricating every signal -- and detection runs the REAL
pipeline (virtual==real).

The camera Measurement (`CameraMeasurement`) publishes ONLY ``frame``.  A separate
`DetectProcessor` consumes ``frame`` and runs the SAME ``calibration.detect``
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
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, DetectProcessor

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    # calibration via the REAL path (site centers detected from frames; per-site thresholds learned)
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=24, display=False)
    cal = exp.readout.current
    assert cal is not None and cal.thresholds is not None

    hub = SignalHub()
    cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
    det = DetectProcessor(hub, calibration=cal, source="frame", grid_shape=(3, 4))
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
    output (intermediate frame + progress) to the hub, saves an npz artifact, and
    yields a calibration a DetectProcessor then consumes -- the whole loading readout
    composed BY THE USER from device + task + processor (no monolithic node)."""
    import numpy as np
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import (
        CalibrateReadoutTask, CameraMeasurement, DetectProcessor)

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    try:
        cal_path = tmp_path / "cal.npz"
        task = CalibrateReadoutTask(
            hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=(3, 4),
            calibration_frames=4, threshold_frames=20, method="box", save_path=str(cal_path))
        task.run_to_completion()

        assert task.finished and task.calibration is not None
        # mid-run output landed on the hub (a frame to show + a progress fraction)
        assert "frame" in hub.names() and hub.latest("progress") == 1.0
        # final result + completion flag published
        assert {"centers", "thresholds", "n_sites", "task_done"} <= set(hub.names())
        assert hub.latest("n_sites") == 12
        # artifact persisted
        saved = np.load(cal_path)
        assert saved["centers"].shape == (12, 2) and saved["thresholds"].shape == (12,)

        # composition: the task's calibration drives a DetectProcessor on live frames
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        det = DetectProcessor(hub, calibration=task.calibration, grid_shape=(3, 4))
        cam.step()
        det.step()
        assert hub.latest("occupied").shape == (12,)
    finally:
        exp.close()


def test_user_composed_loading_readout_streams_real_detect_off_camera_frames():
    """The user composes the loading readout from independent nodes -- a camera
    Measurement publishing ``frame`` + a DetectProcessor turning ``frame`` into
    occupancy/rate (via the REAL calibration.detect) -- and the live chain is
    camera -> frame -> real detect (virtual == real).  No monolithic node fabricates
    every signal; the user (notebook or task console) wires the three primitives."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import (
        CalibrateReadoutTask, CameraMeasurement, DetectProcessor)

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (40, 50)})
    hub = SignalHub()
    try:
        # User composition step 1: run the calibrate task to get a TrapCalibration.
        # (The task publishes its mid-run output under its own cal_ prefix so its
        # template frames never clobber the live frame the user will stream next.)
        task = CalibrateReadoutTask(
            hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=(3, 4),
            calibration_frames=4, threshold_frames=20, method="box", prefix="cal_")
        task.run_to_completion()
        assert task.calibration is not None

        # User composition step 2: a camera Measurement (publishes raw frames).
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        # User composition step 3: a DetectProcessor running the REAL contract.
        det = DetectProcessor(hub, calibration=task.calibration, grid_shape=(3, 4))

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


def test_calibrate_task_namespaces_its_output_away_from_live_frame():
    """The calibrate task uses its OWN prefix (e.g. ``cal_``) so its mid-run + result
    signals (cal_frame / cal_progress / cal_centers / ...) never clobber a LIVE
    ``frame`` the camera measurement publishes alongside it.  A dedicated
    'calibrating' panel can watch ``cal_frame`` / ``cal_progress`` while the live
    panel watches ``frame``."""
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
        task.run_to_completion()                       # runs the calibrate task (cal_ namespace)
        names = set(hub.names())
        assert {"cal_frame", "cal_progress", "cal_centers"} <= names
        assert hub.latest("cal_progress") == 1.0
        assert "frame" not in names                    # calibration did NOT publish the live frame

        # The user adds the live camera measurement separately -- it publishes its
        # OWN ``frame``, distinct from the task's ``cal_frame``.
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
        cam.step()
        assert "frame" in hub.names() and "cal_frame" in hub.names()   # distinct, no clobber
    finally:
        exp.close()
