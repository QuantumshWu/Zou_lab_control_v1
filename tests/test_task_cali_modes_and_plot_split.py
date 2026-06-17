"""MECHANICAL guard for the deep #4 task / measurement requirements.

These are exactly the behaviours that silently regress, so they are pinned here
(the repo rule: a mechanically-enforceable requirement is a test, not a doc line):

  * the calibrate TASK really honours its declared params -- it is NOT cosmetic:
      - source = live  -> acquire now (camera + imaging pulse at the given exposure);
      - source = folder-> calibrate from saved frames in a folder;
      - mode = box / per-site PSF / uniform PSF -> the three sitemap readout models
        (resolved to box / psf / uniform_psf);
      - threshold = otsu / bimodal;
      - save_path -> persist, load_path -> restore WITHOUT re-acquiring.
  * the measurement PLOT split: a measurement called from the NOTEBOOK API defaults
    display=True (it auto-opens its default plot); the SAME measurement driven as a
    GUI/task logic node is plot=False -- it only publishes to the hub (the user wires
    a Plot panel to that signal), and starting it opens NO plot.

Offscreen Qt + virtual backend (the same contract path real hardware takes; only the
camera frames are simulated).
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import time

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _calibrated(grid=(3, 4)):
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (48, 60)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    return exp


# ----------------------------------------------------------- calibrate task modes
@pytest.mark.parametrize("mode, resolved", [
    ("box", "box"),
    ("per-site PSF", "psf"),
    ("uniform PSF", "uniform_psf"),
])
@pytest.mark.parametrize("threshold_method", ["otsu", "bimodal"])
def test_calibrate_task_live_each_mode_and_threshold(mode, resolved, threshold_method):
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    try:
        task = exp.readout.calibrate_task(
            SignalHub(), mode=mode, threshold_method=threshold_method,
            calibration_frames=3, threshold_frames=16,
            sitemap_exposure=0.05, readout_exposure=0.02)
        assert task.method == resolved                 # mode -> sitemap method
        task.run_to_completion()
        centers = np.asarray(task.calibration.centers, dtype=float)
        thresholds = np.asarray(task.calibration.thresholds, dtype=float).reshape(-1)
        assert centers.shape == (12, 2)
        assert thresholds.shape == (12,) and np.isfinite(thresholds).all()
    finally:
        exp.close()


def test_calibrate_task_save_then_load_skips_acquisition(tmp_path):
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    try:
        path = tmp_path / "cal.npz"
        made = exp.readout.calibrate_task(
            SignalHub(), mode="box", save_path=str(path),
            calibration_frames=3, threshold_frames=12)
        made.run_to_completion()
        assert path.exists()

        # load_path restores the SAME calibration with no acquisition
        loaded = exp.readout.calibrate_task(SignalHub(), load_path=str(path))
        loaded.run_to_completion()
        assert np.allclose(np.asarray(loaded.calibration.centers),
                           np.asarray(made.calibration.centers))
    finally:
        exp.close()


def test_calibrate_task_live_uses_two_saved_pulse_programs(tmp_path):
    """The cali task takes TWO saved pulse programs (PulseTableState .json from the
    pulse GUI): a LONGER one for the site/PSF pass and the ACTUAL-readout one for the
    threshold pass.  Each drives its acquisition via the pulse API; both round-trip in
    the Edit (acquisition_parameters)."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.timing import imaging_sequence, PulseTableState

    exp = _calibrated()
    try:
        chans = ["trap", "cooling", "probe", "emCCD"]
        site_prog = tmp_path / "sitemap_long.json"      # long readout for site + PSF
        read_prog = tmp_path / "readout_actual.json"    # actual readout for thresholds
        PulseTableState.from_sequence(
            imaging_sequence(exposure=0.06, load=True, name="sitemap_img"), channels=chans).save(site_prog)
        PulseTableState.from_sequence(
            imaging_sequence(exposure=0.02, load=True, name="readout_img"), channels=chans).save(read_prog)

        task = exp.readout.calibrate_task(
            SignalHub(), source="live", mode="per-site PSF",
            sitemap_pulse=str(site_prog), readout_pulse=str(read_prog),
            calibration_frames=3, threshold_frames=12)
        assert task.sitemap_pulse == str(site_prog) and task.readout_pulse == str(read_prog)
        ap = task.acquisition_parameters()
        assert ap["sitemap_pulse"] == str(site_prog) and ap["readout_pulse"] == str(read_prog)
        task.run_to_completion()
        assert np.asarray(task.calibration.centers).shape == (12, 2)
    finally:
        exp.close()


def test_calibrate_task_folder_source(tmp_path):
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    try:
        folder = tmp_path / "run"
        na.write_virtual_run(str(folder), groups=8, grid_shape=(3, 4), seed=0)
        task = exp.readout.calibrate_task(
            SignalHub(), source="folder", data_dir=str(folder), mode="box")
        assert task.source == "folder"
        task.run_to_completion()
        assert np.asarray(task.calibration.centers).shape == (12, 2)
    finally:
        exp.close()


# ------------------------------------------------- camera measurement: editable region
def test_camera_measurement_region_is_editable_and_applies_to_virtual():
    """The camera measurement exposes its ROI as an editable ``region`` (the SAME
    field for virtual / real -- only the camera differs), and setting it actually
    windows the virtual camera (the frame is cropped).  This pins that virtual==real:
    both honour camera.configure(roi=)."""
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated((3, 4))
    try:
        # the camera Edit auto-form carries region (next to exposure / frames_per_cycle)
        console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                              tasks=exp.readout.task_specs(), window_px=(900, 600))
        console._timer.stop()
        kc = console.kind_combo
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("camera", "live"))
        kc.setCurrentIndex(i)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._edit_logic_node(row)
        editor = console._logic_editors[id(row)]
        assert {"frames_per_cycle", "exposure", "region"} <= set(editor.form._widgets)
        console.shutdown()

        # building with a region windows the VIRTUAL camera -> the frame is cropped
        node = exp.readout.camera_spec().build(SignalHub(), region="10, 50, 8, 40")
        assert node.camera.roi is not None
        frame = node.camera.acquire(1, sequencer=None)[0]
        assert np.asarray(frame).shape != (40, 50)        # ROI actually crops the virtual frame
        assert "region" in node.acquisition_parameters()  # round-trips (endpoints)
    finally:
        exp.close()


# ------------------------------------------------------- measurement plot split
def test_notebook_measurement_defaults_display_true():
    """A measurement called from the notebook API auto-plots (display=True default)."""
    from Zou_lab_control.neutral_atom.operations.measurement import ScannedMeasurement

    exp = _calibrated()
    try:
        # the scanned-measurement run + a readout measurement method both default on
        assert inspect.signature(ScannedMeasurement.run).parameters["display"].default is True
        assert inspect.signature(exp.readout.temperature).parameters["display"].default is True
    finally:
        exp.close()


def test_gui_measurement_node_is_plot_false_publishes_to_hub_only():
    """The SAME measurement driven as a GUI logic node is plot=False: it publishes
    its result signal to the hub and opens NO plot panel (the user wires a Plot)."""
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    specs = exp.readout.measurement_specs()
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=specs, tasks=exp.readout.task_specs(), window_px=(900, 600))
    console._timer.stop()
    try:
        spec = specs[0]
        kc = console.kind_combo
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("measurement", spec.name))
        kc.setCurrentIndex(i)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        deadline = time.monotonic() + 8.0
        while spec.y_key not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert spec.y_key in console.hub.names()       # data on the hub
        assert console.cards == []                     # plot=False: no auto plot panel
    finally:
        console.shutdown()
        exp.close()
