"""MECHANICAL guard for the corrected VIEW / LOGIC split.

A Monitor-board PANEL is a pure VIEW: always role "plot", fully decoupled from
acquisition.  Its Edit carries the FULL plotter set (the whole DataFigure fit set,
task #176) + manual limits + the display knobs (relim / unit) -- but it NEVER
carries a measurement/processor param form or a Start/Stop (a plot never produces
data).  The nodes live on the LOGIC tab as logic nodes (measurement /
processor / task): each node's Edit is its auto-generated param form + Start / Stop
and NO curve fit (fitting is a plotter concern).  This is mechanically forced here
so the decoupling cannot be silently violated.

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

    console = TaskConsole(hub=SignalHub(), state=default_console_state(),
                          measurements=measurements, processors=processors, tasks=tasks,
                          session=session, window_px=(1200, 800))
    console._timer.stop()
    return console


def _open_via_add_panel(console, data):
    """Pick the Add-Panel entry whose itemData == ``data``, add it."""
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == data)
    kc.setCurrentIndex(i)
    console._add_panel()


# ------------------------------------------------------------------- round-trip
def test_panel_config_role_roundtrips_and_validates():
    from Zou_lab_control.frontend.task_console import PanelConfig

    # a board panel is always role "plot" -- it is the only valid panel role now
    cfg = PanelConfig(kind="1d", role="plot", source="value = rate")
    assert cfg.role == "plot"
    assert cfg.to_dict()["role"] == "plot"
    assert PanelConfig.from_dict(cfg.to_dict()).role == "plot"
    # a config WITHOUT a stored role (older layout) defaults to "plot"
    payload = cfg.to_dict(); del payload["role"]
    assert PanelConfig.from_dict(payload).role == "plot"
    with pytest.raises(ValueError):
        PanelConfig(kind="1d", role="bogus")


def test_logic_node_config_roundtrips_and_validates():
    from Zou_lab_control.frontend.task_console import LogicNodeConfig

    node = LogicNodeConfig(kind="measurement", name="Temperature", values={"shots": 8})
    assert node.kind == "measurement" and node.title == "Temperature"
    assert LogicNodeConfig.from_dict(node.to_dict()).values == {"shots": 8}
    with pytest.raises(ValueError):
        LogicNodeConfig(kind="bogus", name="x")


# ------------------------------------------------- plot Edit: FULL fit (#176), no node form
def test_plot_panel_edit_keeps_full_fit_no_node_form():
    console = _console()
    try:
        _open_via_add_panel(console, "1d")              # a plain Plot: 1D vector
        card = console.cards[-1]
        assert card.config.role == "plot"
        assert card.config.source == ""                 # blank pure view
        console._edit_card(card)
        editor = console._panel_editors[id(card)]
        assert editor.fit_combo is not None             # the whole DataFigure fit set
        assert editor.xmin is not None                  # manual limits present
        # a plot NEVER carries a measurement/processor param form or Start/Stop
        assert editor.meas_panel is None
        # Setting keeps the plot-display knobs for a plot panel
        assert hasattr(card, "lim_combo")
        assert hasattr(card, "unit_button")
    finally:
        console.shutdown()


def test_plot_reading_a_signal_still_has_no_node_form():
    """Even a plot whose source reads a measurement's result signal carries NO
    measurement form (fully decoupled) -- you tune the measurement from its OWN
    Logic-tab node, and you fit on the plot."""
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    console = _console(measurements=specs, session=exp)
    try:
        from Zou_lab_control.frontend.task_console import PanelConfig, PanelCard
        spec = specs[0]
        cfg = PanelConfig(kind="1d", role="plot", source=f"value = {spec.y_key}")
        card = PanelCard(cfg, parent=console.board, names_provider=console.hub.names)
        console._attach_card(card)
        console._edit_card(card)
        editor = console._panel_editors[id(card)]
        assert editor.meas_panel is None                # decoupled: no measurement form
        assert editor.fit_combo is not None             # the plotter Edit keeps the fit (#176)
    finally:
        console.shutdown()
        exp.close()


# ------------------------------------------- logic-node Edit: param form + Start/Stop, NO fit
def test_measurement_logic_node_edit_has_form_start_stop_no_fit():
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    console = _console(measurements=specs, session=exp)
    try:
        spec = specs[0]
        _open_via_add_panel(console, ("measurement", spec.name))
        row = console.logic_nodes[-1]
        assert row.node.kind == "measurement"
        editor = console._logic_editors[id(row)]
        # the measurement's OWN param form is present, with Start / Stop ...
        assert editor.form is not None
        # the spec's declared params PLUS the ONE auto-injected acquisition knob `repeat` (0 = ∞,
        # which every measurement-layer node owns -- #H3l/#H3u-2).
        assert set(editor.form._widgets) == {d.key for d in spec.params} | {"repeat"}
        assert editor.form.start_button is not None and editor.form.stop_button is not None
        # ... but a logic node carries NO curve fit, NO manual limits
        assert not hasattr(editor, "fit_combo")
    finally:
        console.shutdown()
        exp.close()


def test_processor_logic_node_edit_has_form_no_fit():
    from Zou_lab_control.neutral_atom.operations.processor import ProcessorSpec

    proc = ProcessorSpec(name="DemoProc", params=(),
                         run=lambda ctx: {"demo": 1.0}, result_keys=("demo",))
    exp = _calibrated_virtual_session()
    console = _console(processors=[proc], session=exp)
    try:
        _open_via_add_panel(console, ("processor", proc.name))
        row = console.logic_nodes[-1]
        assert row.node.kind == "processor"
        editor = console._logic_editors[id(row)]
        assert editor.form is not None                  # the data-processing form
        assert not hasattr(editor, "fit_combo")         # a logic node carries no fit
    finally:
        console.shutdown()
        exp.close()


# --------- Add Panel surfaces EVERY node layer as a clean-named logic node -----
def test_camera_and_task_are_addable_logic_layers_with_clean_labels():
    """Every node LAYER is an addable logic node (not buried in a composite):
    a camera Measurement + a calibrate Task, each with its OWN param Edit + Start/Stop,
    each STOPPED until Started, the signal-flow legend never leaking a class name."""
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, CalibrateReadoutTask

    exp = _calibrated_virtual_session()
    console = _console(tasks=exp.readout.task_specs(), session=exp)
    try:
        kc = console.kind_combo
        entries = {kc.itemData(i): kc.itemText(i) for i in range(kc.count())}
        assert ("camera", "live") in entries and entries[("camera", "live")].startswith("Measurement:")
        assert ("task", "Calibrate readout") in entries and entries[("task", "Calibrate readout")].startswith("Task:")

        # camera node: STOPPED; Start builds a CameraMeasurement; label is the LAYER name
        _open_via_add_panel(console, ("camera", "live"))
        cam_row = console.logic_nodes[-1]
        assert cam_row.node.kind == "camera"
        assert console._logic_nodes[id(cam_row)] is None        # stopped until Start
        console._start_logic_node(cam_row)
        cam = console._logic_nodes[id(cam_row)]
        assert isinstance(cam, CameraMeasurement)
        # the dashboard label is the node's DISPLAY name (the unique "<base> #N", #G306), NEVER the
        # Python class name -- no class name leaks into the UI.
        assert console._node_label(cam) == cam.display_label
        assert "CameraMeasurement" not in console._node_label(cam)
        assert console._node_label(cam).startswith("Camera")

        # calibrate task: its own node, STOPPED, with a param Edit (no fit)
        _open_via_add_panel(console, ("task", "Calibrate readout"))
        task_row = console.logic_nodes[-1]
        assert task_row.node.kind == "task"
        console._start_logic_node(task_row)
        task = console._logic_nodes[id(task_row)]
        assert isinstance(task, CalibrateReadoutTask)
        assert console._node_label(task) == task.display_label
        assert "CalibrateReadoutTask" not in console._node_label(task)
    finally:
        console.shutdown()
        exp.close()
