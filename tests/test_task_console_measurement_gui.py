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
    from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime

    runtime = None
    if session is not None:
        runtime = getattr(session, "_zlc_runtime_services", None)
        if runtime is None or runtime.closed:
            runtime = LegacyNeutralAtomRuntime(session.devices)
            session._zlc_runtime_services = runtime
    console = TaskConsole(hub=SignalHub(), state=default_console_state(),
                          running_nodes=[], measurements=measurements, session=session,
                          runtime_fence=None if runtime is None else runtime.fence,
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
        # the spec's declared params PLUS the auto-injected acquisition knob (the ONE `repeat`, 0 = ∞,
        # which every measurement-layer node owns -- #H3l/#H3u-2; the old separate free_run toggle is gone).
        assert set(form._widgets) == {d.key for d in spec.params} | {"repeat"}
        # the form stores the widget by key + its kind in _decls (no positional tuple now)
        assert form._decls["repeat"].kind == "int"
        assert form._decls["t_off"].kind == "axis_range"     # min/max/points triplet
        assert form._decls["shots"].kind == "int"
        assert form._decls["per_site"].kind == "bool"
        # capture_radius is an ANALYSIS/fit input, NOT an acquisition param -> it is NOT in the form (#H3q)
        assert "capture_radius" not in form._widgets
        vals = form.collect_values()
        assert set(vals) == {d.key for d in spec.params} | {"repeat"}
        assert vals["repeat"] == 1                           # acquisition default (0 = ∞)
        assert isinstance(vals["t_off"], tuple) and len(vals["t_off"]) == 3
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
        ax = form._widgets["t_off"]; lo, hi, pts = ax.min_spin, ax.max_spin, ax.pts_spin
        lo.setValue(0.0); hi.setValue(120.0); pts.setValue(5)
        form._widgets["shots"].setValue(8)

        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        assert isinstance(node, ScannedMeasurementNode) and node in console.running_nodes
        # display suppressed: starting a measurement made NO plot panel
        assert console.cards == []

        console._legacy_handles[id(node)].run_handle.wait(10.0)
        console.refresh_once()

        # the console node publishes under the measurement slug: <key>_<quantity>.
        x_tensor = np.asarray(console.hub.latest(f"{spec.key}_{spec.x_key}"), dtype=float)
        assert x_tensor.shape == (1, 5, 1)
        x = x_tensor[0, :, 0]          # explicitly select the sole R and scalar data cell
        # y is the RAW (R,P,*data_shape) block; reduce the repeat axis the way a plot does.
        from Zou_lab_control.frontend.live import reduce_repeat
        y_tensor = np.asarray(reduce_repeat(
            np.asarray(console.hub.latest(f"{spec.key}_{spec.y_key}"), dtype=float), "replace"),
            dtype=float)
        assert y_tensor.shape == (5, 1)
        y = y_tensor[:, 0]             # scalar data_shape=(1,), never flatten an unknown tail
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
        ax = form._widgets["t_off"]; lo, hi, pts = ax.min_spin, ax.max_spin, ax.pts_spin
        lo.setValue(0.0); hi.setValue(60.0); pts.setValue(3)
        form._widgets["shots"].setValue(2)

        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        console._stop_logic_node(row)
        assert node.running is False
        assert form.start_button.isEnabled()
        assert not form.stop_button.isEnabled()
    finally:
        console.shutdown()
        exp.close()


def test_stopped_node_signals_still_name_their_producer_in_picker():
    """A finished/stopped node's signals LINGER in the hub; the signal picker must still
    name WHICH node produced each (a stopped scan's signal shows "— <node>", never a bare
    name).  Regression for the picker only consulting RUNNING nodes -- a finished readout /
    stopped camera left its signals sourceless."""
    exp = _calibrated_virtual_session(grid=(2, 3))
    specs = exp.readout.measurement_specs()
    console = _console(specs, session=exp)
    try:
        spec, row, editor, form = _open_measurement_node(console, 0)
        ax = form._widgets["t_off"]; lo, hi, pts = ax.min_spin, ax.max_spin, ax.pts_spin
        lo.setValue(0.0); hi.setValue(60.0); pts.setValue(3)
        form._widgets["shots"].setValue(2)

        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        produced = sorted(node.published_signals())
        label = console._node_label(node)
        console._stop_logic_node(row)
        assert node not in console.running_nodes      # genuinely stopped

        providers = console._signal_providers()        # built AFTER the stop
        for name in produced:
            assert providers.get(name) == [label], (name, providers.get(name))
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


def test_1d_panel_draws_multi_column_value_as_multiple_lines():
    """A 1d panel given an ``(N, ncols)`` value draws each column as its OWN line -- the dimension
    axis O2 (one line per series, or one line per repeat in ``repeat_mode='new'``), NOT flattened.
    The x-y curve meaning (col0=x, col1=y) is the OPT-IN ``xy`` marker; without it, columns are
    series drawn vs the point index (#3)."""
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    card = PanelCard(PanelConfig(kind="1d", source="value = arr"))
    try:
        arr = np.column_stack([np.array([1.0, 0.8, 0.5]),
                               np.array([0.2, 0.3, 0.4])])           # (3, 2) -> 2 series
        card.refresh({"arr": arr, "shot": 1})
        assert card.plotter is not None
        assert np.allclose(card.plotter.data_x[:, 0], np.arange(3))  # x = point index
        assert card.plotter.data_y.shape == (3, 2)                   # 2 columns = 2 lines (not flattened)
        assert np.allclose(card.plotter.data_y, arr)
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
    from zlc_neutral_atom.runtime import (
        CleanupStepAck,
        DeviceBroker,
        DeviceIdentityAck,
        DeviceIdentityEvidenceKind,
        MemoryQuarantineJournal,
        ResourceArbiter,
        ResourceKey,
        RunController,
        SafeStateAck,
        SafetyOperation,
    )
    from zlc_workbench import (
        LegacyDeviceRegistration,
        LegacyDeviceRegistry,
        LegacyRuntimeFence,
    )

    broker = DeviceBroker()
    registry = LegacyDeviceRegistry(broker)
    registry.register(
        LegacyDeviceRegistration(
            device=cam,
            key=ResourceKey.parse("device/camera/roi-fixture"),
            identity_probe=lambda: DeviceIdentityAck(
                "fixture:roi-camera",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
                "fixture:roi-camera:connection",
                "test-assets-v1",
            ),
            cleanup_operations={
                SafetyOperation.SAFE_STATE: lambda: CleanupStepAck(
                    SafetyOperation.SAFE_STATE, "fixture-safe-command"
                )
            },
            cleanup_order=(SafetyOperation.SAFE_STATE,),
            verify_safe_state=lambda: SafeStateAck("fixture-safe-readback"),
        )
    )
    fence = LegacyRuntimeFence(
        RunController(ResourceArbiter(MemoryQuarantineJournal())), registry
    )
    state = zf.TaskConsoleState(name="t", panels=[
        zf.PanelConfig(kind="2d", title="cam", size="2x2", source="value = frame_0")])
    console = TaskConsole(hub=hub, state=state, runtime_fence=fence)
    console._timer.stop()
    node.step()
    handle = fence.start(node)
    handle.wait_started(2.0)
    console._legacy_handles[id(node)] = handle
    console.running_nodes.append(node)
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
        assert node.running
        apply_epoch = node.acquisition_epoch()
        editor._restart_node()
        assert node.running                              # Apply kept the sole owner live
        import time
        deadline = time.monotonic() + 2.0
        while (
            (
                cam.roi != (1670, 20, 1166, 20)
                or node.acquisition_epoch() == apply_epoch
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert cam.roi == (1670, 20, 1166, 20)           # device ROI = internal conversion
        console.refresh_once()
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
    from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime

    hub = SignalHub()
    experiment = na.connect("virtual")
    runtime = LegacyNeutralAtomRuntime(experiment.devices)
    node = CameraMeasurement(hub, experiment.devices.camera)
    assert not node.running
    console = show_task_console(
        hub=hub,
        running_nodes=[node],
        runtime_fence=runtime.fence,
        title="autostart-test",
    )
    try:
        assert node.running
    finally:
        console.shutdown()
        runtime.shutdown(timeout=2.0)
        experiment.devices.close()
    assert not node.running


def test_edit_tab_is_the_full_datafigure_ui():
    """The Edit tab carries the WHOLE DataFigure control surface (the same view knobs as the
    Setting popup, present here too): colormap (image kinds), relim (tight/normal), the unit
    cycle button, AND the save folder picker.  Each display knob drives the LIVE card via the
    card's own handlers (config.params is the single source), so Setting and Edit stay in sync."""
    console, node, cam, card, editor = _camera_console()
    try:
        # a 2D (image) panel -> colormap control present, plus relim + unit + save picker
        assert editor.ed_cmap is not None
        assert editor.ed_relim is not None
        assert editor.ed_unit_button is not None
        assert hasattr(editor, "save_dir_edit")

        # changing a display knob in Edit writes the SAME config.params key the Setting popup uses;
        # every display knob (colormap / relim) now routes through the ONE declarative _edit_param
        # path (#H3v-4b: cmap/relim are ParamDecls rendered by the shared _emit_param_rows).
        editor._edit_param("cmap", "viridis")
        assert card.config.params["cmap"] == "viridis"
        editor._edit_param("relim", "normal")
        assert card.config.params["relim"] == "normal"
        assert card.plotter is None or card.plotter.relim_mode == "normal"
    finally:
        console.shutdown()


def test_edit_save_remembers_the_chosen_folder(tmp_path):
    """The Edit save picker prefills a default folder, lets the user pick any folder, writes the
    figure there, and REMEMBERS it on the console -- so the NEXT panel's Edit opens at that same
    folder (persisted for the kernel session, not reset each time)."""
    console, node, cam, card, editor = _camera_console()
    try:
        assert editor.save_dir_edit.text()                 # prefilled (defaults to tasks/)
        editor.save_dir_edit.setText(str(tmp_path))        # user picks a folder
        editor.save()
        saved = list(tmp_path.glob("*.png"))
        assert saved, "save() wrote no figure into the chosen folder"
        assert console._last_save_dir == str(tmp_path)     # remembered on the console

        # a freshly opened Edit for another panel prefills the remembered folder
        from Zou_lab_control.frontend.task_console import PanelEditor
        editor2 = PanelEditor(card, console)
        assert editor2.save_dir_edit.text() == str(tmp_path)
    finally:
        console.shutdown()


def test_edit_command_result_is_readonly_but_copyable():
    """The Edit Command box: the `run` input fills its row, and the `result` is a
    FluentReadoutEdit -- read-only (cannot be edited) BUT selectable/copyable (a QLineEdit, so
    Ctrl-C works), unlike a plain label.  Running a command populates that copyable field."""
    from Zou_lab_control.frontend.qt_fluent import FluentReadoutEdit
    console, node, cam, card, editor = _camera_console()
    try:
        assert isinstance(editor.cmd_result, FluentReadoutEdit)
        assert editor.cmd_result.isReadOnly()                       # cannot be edited
        assert "#F3F3F3" in editor.cmd_result.styleSheet()          # distinct grey fill (not white input)
        editor.cmd_input.setText("1 + 2")
        editor._run_command()
        assert editor.cmd_result.text() == "3"                      # result lands in the copyable field
    finally:
        console.shutdown()
