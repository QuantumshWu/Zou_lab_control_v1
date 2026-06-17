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
            calibration_frames=3, threshold_frames=16, exposure=0.02)
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


def test_calibrate_task_live_uses_saved_pulse_program(tmp_path):
    """A saved pulse program (a PulseTableState .json from the pulse GUI) drives the
    LIVE imaging acquisition via the pulse API -- the cali task loads it and acquires
    under it instead of the default imaging sequence."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.timing import imaging_sequence, PulseTableState

    exp = _calibrated()
    try:
        prog = tmp_path / "imaging_program.json"
        state = PulseTableState.from_sequence(
            imaging_sequence(exposure=0.02, load=True, name="cali_img"),
            channels=["trap", "cooling", "probe", "emCCD"])
        state.save(prog)

        task = exp.readout.calibrate_task(
            SignalHub(), source="live", pulse_program=str(prog), mode="box",
            calibration_frames=3, threshold_frames=12)
        assert task.pulse_program == str(prog)                  # threaded through
        assert task.acquisition_parameters()["pulse_program"] == str(prog)  # Edit round-trips it
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
