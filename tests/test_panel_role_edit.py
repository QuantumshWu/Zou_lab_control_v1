"""MECHANICAL guard for the P6 panel ROLE model (the Add-Panel-3-types contract).

A panel's ROLE (orthogonal to its plot KIND) decides how its Edit + Setting are
composed -- this is what makes "measurement has its own Edit, no fit" / "plotter
Edit has fit + the measurement's params" / "task Edit" real instead of prose.  The
design rule (#3) is mechanically forced here so it cannot be silently violated:

  * a "measurement"-role panel's Edit carries the measurement's PARAM FORM but NO
    curve fit and NO manual-limit row (its Setting also drops the plot-display-only
    relim/unit knobs);
  * a "task"-role (data-processing) panel's Edit carries the processing param form,
    again NO fit;
  * a "plot"-role panel keeps the FULL plotter Edit -- the whole DataFigure fit set
    (task #176) -- AND, when its source reads a measurement's result signals, that
    measurement's param form too (so the plotter Edit = fit + measurement edit);
  * role round-trips through the saved-layout JSON.

Offscreen Qt + virtual backend (the same contract path as real hardware).
"""

from __future__ import annotations

from pathlib import Path
import sys

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


def _calibrated_virtual_session(grid=(2, 3)):
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (64, 80)})
    exp.readout.sitemap(method="box", frames=6, display=False)
    exp.readout.thresholds(frames=40, display=False)
    return exp


def _console(*, measurements=(), processors=(), tasks=(), session=None):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    console = TaskConsole(hub=SignalHub(), state=default_console_state(), feeds=[],
                          measurements=measurements, processors=processors, tasks=tasks,
                          session=session, window_px=(1200, 800))
    console._timer.stop()
    return console


def _open_via_add_panel(console, data):
    """Pick the Add-Panel entry whose itemData == ``data``, add it, return its card."""
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == data)
    kc.setCurrentIndex(i)
    console._add_panel()


# ------------------------------------------------------------------- round-trip
def test_panel_config_role_roundtrips_and_validates():
    from Zou_lab_control.frontend.task_console import PanelConfig

    cfg = PanelConfig(kind="1d", role="measurement", source="value = rate")
    assert cfg.role == "measurement"
    assert cfg.to_dict()["role"] == "measurement"
    assert PanelConfig.from_dict(cfg.to_dict()).role == "measurement"
    # a config WITHOUT a stored role (older layout) defaults to "plot"
    payload = cfg.to_dict(); del payload["role"]
    assert PanelConfig.from_dict(payload).role == "plot"
    with pytest.raises(ValueError):
        PanelConfig(kind="1d", role="bogus")


# ------------------------------------------------- measurement Edit: NO fit
def test_measurement_panel_edit_has_param_form_but_no_fit():
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    console = _console(measurements=specs)
    try:
        spec = specs[0]
        _open_via_add_panel(console, ("measurement", spec.name))
        card = next(c for c in console.cards if c.config.params.get("measurement") == spec.name)
        assert card.config.role == "measurement"
        editor = console._panel_editors[id(card)]
        # the measurement's OWN param form is present...
        assert editor.meas_panel is not None
        # ...but a measurement node is acquisition, not curve analysis: NO fit,
        # NO manual limits (the role gate that honours #3 without reverting #176).
        assert editor.fit_combo is None
        assert editor.xmin is None and editor.ymin is None
        # its Setting also drops the plot-display-only knobs (relim / unit)
        assert not hasattr(card, "lim_combo")
        assert not hasattr(card, "unit_button")
    finally:
        console.shutdown()


# ------------------------------------------------- plot Edit: FULL fit (#176)
def test_plot_panel_edit_keeps_full_fit_and_display_knobs():
    console = _console()
    try:
        _open_via_add_panel(console, "1d")              # a plain Plot: 1D vector
        card = console.cards[-1]
        assert card.config.role == "plot"
        editor = console._panel_editors.get(id(card))
        if editor is None:                              # plain plot opens no Edit tab itself
            console._edit_card(card)
            editor = console._panel_editors[id(card)]
        assert editor.fit_combo is not None             # the whole DataFigure fit set
        assert editor.xmin is not None                  # manual limits present
        # Setting keeps the plot-display knobs for a plot-role panel
        assert hasattr(card, "lim_combo")
        assert hasattr(card, "unit_button")
    finally:
        console.shutdown()


# --------------------------- plot reading a measurement's signals: fit + its form
def test_plot_panel_reading_measurement_signals_links_param_form_and_keeps_fit():
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    console = _console(measurements=specs)
    try:
        from Zou_lab_control.frontend.task_console import PanelConfig, PanelCard
        spec = specs[0]
        # a PLOT-role panel whose source reads the measurement's result signal
        cfg = PanelConfig(kind="1d", role="plot", source=f"value = {spec.y_key}")
        card = PanelCard(cfg, parent=console.board, names_provider=console.hub.names)
        console._attach_card(card)
        console._edit_card(card)
        editor = console._panel_editors[id(card)]
        # the plotter Edit carries the producing measurement's param form (#3)...
        assert editor.meas_panel is not None
        # ...AND the full fit (it is a plotter -- #176 stays intact)
        assert editor.fit_combo is not None
    finally:
        console.shutdown()


# ------------------------------------------------- task Edit: form + Run, NO fit
def test_data_processing_panel_edit_has_form_but_no_fit():
    from Zou_lab_control.neutral_atom.operations.processor import ProcessorSpec

    proc = ProcessorSpec(name="DemoProc", params=(),
                         run=lambda readout: {"demo": 1.0}, result_keys=("demo",))
    console = _console(processors=[proc])
    try:
        _open_via_add_panel(console, ("processor", proc.name))
        card = next(c for c in console.cards if c.config.params.get("processor") == proc.name)
        assert card.config.role == "task"
        editor = console._panel_editors[id(card)]
        assert editor.meas_panel is not None            # the data-processing form
        assert editor.fit_combo is None                 # a task node carries no fit
        assert not hasattr(card, "lim_combo")
    finally:
        console.shutdown()


# --------- Add Panel surfaces EVERY layer: camera Measurement + Task (the gap) -----
def test_camera_measurement_and_task_are_addable_layers_with_clean_labels():
    """The whole point of task_console: every architecture LAYER is an addable node.
    A standalone CONTINUOUS camera Measurement and a calibrate Task must each be in
    Add Panel (not buried in the loading composite), each with its OWN param Edit, and
    the signal-flow legend must speak in LAYER names -- never a "Feed" class name."""
    from Zou_lab_control.neutral_atom.operations.feeds import CameraFrameFeed, CalibrateReadoutTask

    exp = _calibrated_virtual_session()
    console = _console(tasks=exp.readout.task_specs(), session=exp)
    try:
        kc = console.kind_combo
        entries = {kc.itemData(i): kc.itemText(i) for i in range(kc.count())}
        # the camera live measurement + the calibrate task are BOTH offered as layers
        # (the task comes from the auto-discovered @task catalog, by name).
        assert ("camera", "live") in entries and entries[("camera", "live")].startswith("Measurement:")
        assert ("task", "Calibrate readout") in entries and entries[("task", "Calibrate readout")].startswith("Task:")

        # a continuous CAMERA measurement: its own node (CameraFrameFeed), measurement
        # role, an exposure Edit, NO fit -- and its label is the LAYER name "camera".
        _open_via_add_panel(console, ("camera", "live"))
        cam_card = next(c for c in console.cards if c.config.title == "Camera (live)")
        cam = next(f for f in console.feeds if isinstance(f, CameraFrameFeed))
        assert cam_card.config.role == "measurement"
        assert console._producer_label(cam) == "camera"        # layer name, not "CameraFrameFeed"
        cam_ed = console._panel_editors[id(cam_card)]
        assert cam_ed.fit_combo is None and "exposure" in cam_ed._feed_widgets

        # a calibrate TASK: its own node (CalibrateReadoutTask), task role, a param
        # Edit with a Run button + NO fit, label = the layer name "calibrate".
        _open_via_add_panel(console, ("task", "Calibrate readout"))
        task_card = next(c for c in console.cards if c.config.title == "Calibrate readout")
        task = next(f for f in console.feeds if isinstance(f, CalibrateReadoutTask))
        assert task_card.config.role == "task"
        assert console._producer_label(task) == "calibrate"
        task_ed = console._panel_editors[id(task_card)]
        assert task_ed.fit_combo is None and task_ed.feed_restart_button.text() == "Run"
        assert "grid_shape" in task_ed._feed_widgets

        # the footer signal legend never leaks a Python class name (the "Feed" the
        # user objected to, or DetectProcessor / CalibrateReadoutTask)
        console._refresh_signal_info()
        for c in console.cards:
            info = c._signal_info
            assert "Feed" not in info
            assert "DetectProcessor" not in info and "CalibrateReadoutTask" not in info
    finally:
        console.shutdown()
