"""MECHANICAL guard for the task-console measurement / plot split (FRONTEND).

A measurement is a LOGIC NODE (Logic tab), added STOPPED, with an auto-generated,
validated parameter form + Start / Stop in its Edit; Start streams its scan to the
hub (display suppressed -- no auto plot).  A PLOT panel is a decoupled pure view:
its data shape handling (1d x-y, 2d ROI coordinate axes) and its 2D region->camera
writeback are plotter concerns and stay on the panel.  This pins:

1. plumbing -- passing ``measurements`` lists them in Add Panel; adding one creates
   a STOPPED logic node whose Edit form has the right typed widgets (read BY KIND,
   no free-text eval);
2. Start -- builds a ScannedMeasurementNode, registers it, runs it; nothing plots
   automatically (the user adds a Plot pointed at its result signals);
3. plot data shape -- a 1-D panel given an (N, 2) value plots y vs col 0 (when xy=True);
   a 2-D panel reads its source's ROI as real pixel axes;
4. plot 2D region writeback -- selecting on a camera-frame plot's Edit fills the
   camera node's region and Apply re-acquires.

Built directly on the offscreen Qt platform; no demo GUI fixtures.
"""

from __future__ import annotations

from pathlib import Path
import sys

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


def _calibrated_virtual_session(grid=(2, 3)):
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (64, 80)})
    exp.readout.sitemap(method="box", frames=6, display=False)
    exp.readout.thresholds(frames=40, display=False)
    return exp


def _console(measurements, *, session=None):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    console = TaskConsole(hub=SignalHub(), state=default_console_state(),
                          running_nodes=[], measurements=measurements, session=session,
                          window_px=(1200, 800))
    console._timer.stop()       # deterministic: drive refresh_once() / poll ourselves
    return console


def _open_measurement_node(console, index=0):
    """Create a measurement LOGIC node via Add Panel and return (spec, row, editor,
    form)."""
    spec = console.measurements[index]
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == ("measurement", spec.name))
    kc.setCurrentIndex(i)
    console._add_panel()
    row = console.logic_nodes[-1]
    editor = console._logic_editors[id(row)]
    return spec, row, editor, editor.form


# ----------------------------------------------------------------- plumbing
def test_no_measurements_no_measurement_entries_in_add_panel():
    """With no measurements wired, Add Panel lists no measurement entries, and there
    is no global measurement launcher."""
    console = _console(())
    try:
        kc = console.kind_combo
        data = [kc.itemData(i) for i in range(kc.count())]
        assert not any(isinstance(d, tuple) and d and d[0] == "measurement" for d in data)
        assert console.measurement_panel is None
        assert console.measurement_group is None
    finally:
        console.shutdown()


def test_measurements_listed_in_add_panel():
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    console = _console(specs, session=exp)
    try:
        kc = console.kind_combo
        entries = [kc.itemData(i) for i in range(kc.count())]
        meas = [d[1] for d in entries if isinstance(d, tuple) and d and d[0] == "measurement"]
        assert meas == [s.name for s in specs]
    finally:
        console.shutdown()
        exp.close()


def test_measurement_node_edit_generates_typed_form_with_required_param():
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    console = _console(specs, session=exp)
    try:
        spec, row, editor, form = _open_measurement_node(console, 0)
        assert form is not None
        assert set(form._widgets) == {d.key for d in spec.params}
        assert form._widgets["t_off"][0] == "axis_range"     # min/max/points triplet
        assert form._widgets["shots"][0] == "int"
        assert form._widgets["capture_radius"][0] == "float"
        assert form._widgets["per_site"][0] == "bool"
        vals = form.collect_values()
        assert set(vals) == {d.key for d in spec.params}
        assert isinstance(vals["t_off"], tuple) and len(vals["t_off"]) == 3
        assert vals["capture_radius"] == pytest.approx(spec.param("capture_radius").default)
        assert isinstance(vals["per_site"], bool)
    finally:
        console.shutdown()
        exp.close()


# ----------------------------------------------------------------- Start (no auto plot)
def test_node_start_builds_node_and_streams_no_auto_plot():
    """Start in a measurement node's Edit builds a ScannedMeasurementNode, registers
    it, and runs it -- the SAME decaying survival contract real hardware runs --
    WITHOUT creating any plot panel (display suppressed; the user adds a Plot)."""
    exp = _calibrated_virtual_session(grid=(5, 7))
    specs = exp.readout.measurement_specs()
    console = _console(specs, session=exp)
    try:
        from Zou_lab_control.neutral_atom.operations.logic import ScannedMeasurementNode

        spec, row, editor, form = _open_measurement_node(console, 0)
        _, lo, hi, pts = form._widgets["t_off"]
        lo.setValue(0.0); hi.setValue(120.0); pts.setValue(5)
        form._widgets["shots"][1].setValue(8)

        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        assert isinstance(node, ScannedMeasurementNode) and node in console.running_nodes
        # display suppressed: starting a measurement made NO plot panel
        assert console.cards == []

        node.stop()
        node.run_to_completion()
        console.refresh_once()

        x = console.hub.latest(spec.x_key)
        y = console.hub.latest(spec.y_key)
        assert x.shape == (5,) and y.shape == (5,)
        assert np.all(np.isfinite(y)) and np.all((y >= 0) & (y <= 1))
        assert y[0] > y[-1] + 0.3
    finally:
        console.shutdown()
        exp.close()


def test_node_start_then_stop_releases_controls_and_stops_node():
    exp = _calibrated_virtual_session(grid=(2, 3))
    specs = exp.readout.measurement_specs()
    console = _console(specs, session=exp)
    try:
        spec, row, editor, form = _open_measurement_node(console, 0)
        _, lo, hi, pts = form._widgets["t_off"]
        lo.setValue(0.0); hi.setValue(60.0); pts.setValue(3)
        form._widgets["shots"][1].setValue(2)

        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        console._stop_logic_node(row)
        node.stop()
        assert node.running is False
        assert form.start_button.isEnabled()
        assert not form.stop_button.isEnabled()
    finally:
        console.shutdown()
        exp.close()


# ----------------------------------------------------------------- 1d x-y (plot view)
def test_1d_panel_xy_curve_uses_col0_as_x():
    """An (N, 2) value plots y vs col 0 -- but ONLY when the panel is explicitly
    flagged xy=True (a curve view whose source is value = column_stack([x, y]))."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d", source="value = xy",
                                 params={"xy": True, "xlabel": "T (s)", "ylabel": "Surv"}))
    try:
        xy = np.column_stack([np.array([0.0, 10.0, 20.0, 30.0]),
                              np.array([1.0, 0.8, 0.5, 0.2])])
        card.refresh({"xy": xy, "shot": 1})
        assert card.plotter is not None
        assert np.allclose(card.plotter.data_x[:, 0], [0.0, 10.0, 20.0, 30.0])
        assert np.allclose(card.plotter.data_y[:, 0], [1.0, 0.8, 0.5, 0.2])
        assert card.plotter.ax.get_xlabel() == "T (s)"
    finally:
        card.shutdown()


def test_1d_panel_plain_vector_still_uses_index():
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d", source="value = vec"))
    try:
        card.refresh({"vec": np.array([3.0, 1.0, 4.0, 1.0, 5.0]), "shot": 1})
        assert card.plotter is not None
        assert np.allclose(card.plotter.data_x[:, 0], np.arange(5))
    finally:
        card.shutdown()


def test_1d_panel_without_xy_marker_flattens_n_by_2():
    """A plain 1d panel (no xy marker) that happens to produce an (N, 2) value
    flattens to a vector vs index -- the x-y meaning is opt-in, never inferred."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d", source="value = arr"))
    try:
        arr = np.column_stack([np.array([0.0, 10.0, 20.0]),
                               np.array([1.0, 0.8, 0.5])])
        card.refresh({"arr": arr, "shot": 1})
        assert card.plotter is not None
        assert np.allclose(card.plotter.data_x[:, 0], np.arange(6))
        assert np.allclose(card.plotter.data_y[:, 0], arr.reshape(-1))
    finally:
        card.shutdown()


def test_blank_panel_shows_nothing_until_a_signal_is_picked():
    """A fresh plot panel is BLANK (decoupled pure view): a blank source builds no
    plot and reports a hint, not an error."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d"))         # source defaults to blank
    try:
        assert card.config.source == ""
        card.refresh({"vec": np.array([1.0, 2.0]), "shot": 1})
        assert card.plotter is None                  # nothing shown
        assert not card._status_error                # blank is not an error
    finally:
        card.shutdown()


# ------------------------------------ coordinate axes = source param space (plot view)
def test_2d_panel_axes_use_source_roi_coordinates():
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="2d", source="value = frame"))
    try:
        frame = np.random.rand(16, 24) * 100.0           # h=16, w=24
        region = [1648, 1672, 1144, 1160]                 # origin (1648,1144), 24x16
        card.refresh({"frame": frame, "__coord_frames__": {"frame": region}, "shot": 1})
        p = card.plotter
        assert np.isclose(p.x_array[0], 1648) and np.isclose(p.x_array[-1], 1648 + 24 - 1)
        assert np.isclose(p.y_array[0], 1144) and np.isclose(p.y_array[-1], 1144 + 16 - 1)
        assert p.ax.get_xlabel() == "Camera x (px)"
        assert p.ax.get_ylabel() == "Camera y (px)"
    finally:
        card.shutdown()


def test_2d_panel_axes_default_to_index_without_coord_frame():
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="2d", source="value = frame"))
    try:
        frame = np.random.rand(16, 24) * 100.0
        card.refresh({"frame": frame, "shot": 1})
        p = card.plotter
        assert np.isclose(p.x_array[0], 0) and np.isclose(p.x_array[-1], 23)
        assert np.isclose(p.y_array[0], 0) and np.isclose(p.y_array[-1], 15)
    finally:
        card.shutdown()


def test_2d_panel_rebuilds_when_roi_shifts_same_shape():
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="2d", source="value = frame"))
    try:
        frame = np.random.rand(16, 24) * 100.0
        card.refresh({"frame": frame, "__coord_frames__": {"frame": [1648, 1672, 1144, 1160]}, "shot": 1})
        assert np.isclose(card.plotter.x_array[0], 1648)
        card.refresh({"frame": frame, "__coord_frames__": {"frame": [100, 124, 200, 216]}, "shot": 2})
        assert np.isclose(card.plotter.x_array[0], 100)
        assert np.isclose(card.plotter.y_array[0], 200)
    finally:
        card.shutdown()


def _camera_console(roi=(1648, 64, 1144, 64)):
    """A console driving a 2D PLOT panel from a faked ROI-bearing camera node, with
    the panel's Edit tab opened.  Only the lowest data source (the camera frame + its
    ROI) is faked; everything above is the real task-console plot path.  Returns
    (console, node, cam, card, editor)."""
    import Zou_lab_control.frontend as zf
    from Zou_lab_control.frontend.task_console import TaskConsole
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice

    class _RoiCamera(CameraDevice):
        def __init__(self, roi):
            self._exposure = 0.02
            self._roi = tuple(int(v) for v in roi)

        @property
        def exposure(self):
            return self._exposure

        @property
        def roi(self):
            return self._roi

        def configure(self, *, exposure=None, **kw):
            if exposure is not None:
                self._exposure = float(exposure)
            if kw.get("roi") is not None:
                self._roi = tuple(int(v) for v in kw["roi"])

        def acquire(self, frames=1, *, sequence=None, sequencer=None, stop=None, **kw):
            w, h = self._roi[1], self._roi[3]
            return [np.full((h, w), 50.0)]

    hub = SignalHub()
    cam = _RoiCamera(roi)
    node = CameraMeasurement(hub, cam)
    state = zf.TaskConsoleState(name="t", panels=[
        zf.PanelConfig(kind="2d", title="cam", size="2x2", source="value = frame")])
    console = TaskConsole(hub=hub, state=state, running_nodes=[node])
    console._timer.stop()
    node.step()
    console.refresh_once()
    card = console.cards[0]
    console._edit_card(card)
    editor = console._panel_editors[id(card)]
    editor.rebuild()
    return console, node, cam, card, editor


def test_edit_area_select_writes_camera_region_and_apply_restarts_monitor():
    """A region selected on a camera-frame 2D plot's Edit fills the acquisition
    ``region`` field as plot-coord ENDPOINTS [x_min,x_max,y_min,y_max]; Apply re-arms
    the node so the live panel re-acquires under the new window."""
    console, node, cam, card, editor = _camera_console()
    try:
        assert editor._plotter.area.callback.__name__ == "_read_region"
        assert editor._plotter.zoom.callback.__name__ == "_read_region"
        editor._plotter.area.range = [1670.0, 1690.0, 1166.0, 1186.0]
        editor._plotter.area.callback()
        assert editor._node_widgets["region"].text() == "[1670, 1690, 1166, 1186]"
        assert not node.running
        editor._restart_node()
        assert node.running                              # Apply made the source live
        assert cam.roi == (1670, 20, 1166, 20)           # device ROI = internal conversion
        assert np.isclose(card.plotter.x_array[0], 1670)
        assert np.isclose(card.plotter.x_array[-1], 1670 + 20 - 1)
        assert editor._node_now_labels["region"].text() == "now: [1670, 1690, 1166, 1186]"
    finally:
        console.shutdown()


def test_2d_zoom_updates_region_from_view_area_overrides():
    """ZOOM/PAN alone updates the region from the view; an area rectangle overrides it."""
    console, node, cam, card, editor = _camera_console()
    try:
        editor._plotter.area.range = [None, None, None, None]
        editor._plotter.ax.set_xlim(1660, 1690)
        editor._plotter.ax.set_ylim(1190, 1160)        # image y runs high->low
        editor._plotter.zoom.callback()
        assert editor._node_widgets["region"].text() == "[1660, 1690, 1160, 1190]"
        editor._plotter.area.range = [1672.0, 1682.0, 1170.0, 1180.0]
        editor._plotter.zoom.callback()
        assert editor._node_widgets["region"].text() == "[1672, 1682, 1170, 1180]"
    finally:
        console.shutdown()


def test_camera_frames_per_cycle_is_adjustable_in_the_gui():
    """frames_per_cycle auto-surfaces in the 2D plot's Edit -> Acquisition form (it is
    in acquisition_parameters()), and Apply pushes it to the node live."""
    console, node, cam, card, editor = _camera_console()
    try:
        assert "frames_per_cycle" in editor._node_widgets
        assert "frame_1" not in node.published_signals()
        editor._node_widgets["frames_per_cycle"].setText("2")
        editor._restart_node()
        assert node.frames_per_cycle == 2
        assert "frame_1" in node.published_signals()
    finally:
        console.shutdown()


def test_show_task_console_auto_starts_passed_nodes():
    """show_task_console makes a passed-in logic node LIVE on open; shutdown stops it."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import show_task_console
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    hub = SignalHub()
    node = CameraMeasurement(hub, na.connect("virtual").devices.camera)
    assert not node.running
    console = show_task_console(hub=hub, running_nodes=[node], title="autostart-test")
    try:
        assert node.running
    finally:
        console.shutdown()
    assert not node.running
