"""Current C1 product flows through TaskConsole and the sole zlc_plot surface.

The tests drive the formal DeviceManager/TaskConsole widgets under the offscreen
Qt fast path.  Core projection, selector, fit, and facet mathematics are covered
by the focused zlc_plot contract tests; this module retains only cross-owner
product wiring that those tests cannot prove.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from gui_user_flow import (
    capture_offscreen_window,
    choose_combo_data as _choose_combo_data,
    choose_combo_text as _choose_combo_text,
    click_tab,
    configure_offscreen_fast_path,
    current_logic_editor as _current_logic_editor,
    drag_mouse_move,
    replace_spin_value as _replace_spin_value,
    require_offscreen_platform,
    until,
    visible_form_widgets as _visible_form_widgets,
    widget_gone,
)
from zlc_data import REPEAT, SPATIAL_X, SPATIAL_Y
from zlc_frontend.qt_widgets import FluentPlotFitPanel, ensure_qt_app
from zlc_plot import PlotKind
from zlc_workbench.task_console.console_records import (
    console_signal_key,
    panel_signal_key,
)


def _workspace_with_pulses(tmp_path: Path, *names: str) -> Path:
    workspace = tmp_path / "workspace"
    pulses = workspace / "pulses"
    pulses.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1] / "pulses"
    for name in names:
        shutil.copy2(source_root / name, pulses / name)
    return workspace


def _choose_signal_leaf(combo, signal: str, application) -> None:
    """Expand the visible producer row and click one real signal leaf."""

    ancestor = combo.parentWidget()
    while ancestor is not None and not isinstance(
        ancestor,
        QtWidgets.QAbstractScrollArea,
    ):
        ancestor = ancestor.parentWidget()
    if isinstance(ancestor, QtWidgets.QAbstractScrollArea):
        bar = ancestor.verticalScrollBar()
        if bar.maximum() > bar.minimum():
            QtTest.QTest.keyClick(bar, QtCore.Qt.Key_End)
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    view = combo.view()
    model = combo.model()
    parent_index = child_index = None
    for parent_row in range(model.rowCount()):
        candidate_parent = model.index(parent_row, 0)
        for child_row in range(model.rowCount(candidate_parent)):
            candidate_child = model.index(child_row, 0, candidate_parent)
            if candidate_child.data(QtCore.Qt.UserRole) == signal:
                parent_index = candidate_parent
                child_index = candidate_child
                break
        if child_index is not None:
            break
    assert parent_index is not None and child_index is not None
    view.scrollTo(parent_index)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    if not view.isExpanded(parent_index):
        QtTest.QTest.mouseClick(
            view.viewport(),
            QtCore.Qt.LeftButton,
            pos=view.visualRect(parent_index).center(),
        )
        until(application, lambda: view.isExpanded(parent_index))
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    if not view.isVisible():
        QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
        until(application, view.isVisible)
    view.scrollTo(child_index)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    click_position = view.visualRect(child_index).center()
    assert view.rect().contains(click_position)
    assert view.indexAt(click_position) == child_index
    QtTest.QTest.mouseClick(
        view.viewport(),
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    until(application, lambda: combo.currentData() == signal)


def _signal_leaf_keys(combo) -> set[str]:
    model = combo.model()
    found: set[str] = set()

    def visit(parent=QtCore.QModelIndex()) -> None:
        for row in range(model.rowCount(parent)):
            index = model.index(row, 0, parent)
            value = index.data(QtCore.Qt.UserRole)
            if value:
                found.add(str(value))
            visit(index)

    visit()
    return found


def _resolved_artifact(console, output_key: str):
    producer = console.resolve_console_producer(output_key)
    return producer.artifact if producer.artifact_resolved else None


def _plot_front_or_none(card):
    widget = card.plot_widget
    return None if widget is None else widget.presented_front


def _wait_for_plot(application, card, *, timeout: float = 15.0) -> None:
    try:
        until(
            application,
            lambda: card.presented_value is not None
            and _plot_front_or_none(card) is not None,
            timeout=timeout,
        )
    except AssertionError as error:
        raise AssertionError(
            f"plot did not present: status={card.status.text()!r}, "
            f"signal={card.config.signal!r}, params={card.config.params!r}"
        ) from error


def _signal_value_or_none(console, signal_key: str):
    return console._data.freeze().value(signal_key)


def _add_button(console):
    return next(
        button
        for button in console.findChildren(QtWidgets.QPushButton)
        if button.text() == "Add Panel"
    )


def _add_plot_and_bind(
    console,
    add_button,
    kind: PlotKind,
    signal: str,
    application,
):
    before = len(console.cards)
    _choose_combo_data(console.kind_combo, kind, application)
    QtTest.QTest.mouseClick(add_button, QtCore.Qt.LeftButton)
    assert len(console.cards) == before + 1
    card = console.cards[-1]
    click_tab(console, console.tabs.widget(0))
    QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
    until(application, lambda: card.settings_popup.isVisible())
    until(
        application,
        lambda: signal in _signal_leaf_keys(card.signal_combo),
        timeout=15.0,
    )
    _choose_signal_leaf(card.signal_combo, signal, application)
    assert card.config.signal == signal
    return card


def _fit_panel(card, application) -> FluentPlotFitPanel:
    if not card.settings_popup.isVisible():
        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
    until(application, lambda: card.settings_popup.isVisible())
    until(
        application,
        lambda: any(
            pane.isVisible()
            and pane.host is card.host
            and pane.fit_button.isEnabled()
            for pane in card.settings_popup.findChildren(FluentPlotFitPanel)
        ),
        timeout=15.0,
    )
    return next(
        pane
        for pane in card.settings_popup.findChildren(FluentPlotFitPanel)
        if pane.isVisible() and pane.host is card.host
    )


def _plot_point(card, x_fraction: float, y_fraction: float) -> QtCore.QPoint:
    widget = card.plot_widget
    front = widget.presented_front
    axes = next(axis for axis in front.interaction.axes if axis.role == "image")
    left, top, right, bottom = axes.bounds
    nx = left + float(x_fraction) * (right - left)
    ny = top + float(y_fraction) * (bottom - top)
    return QtCore.QPoint(
        int(round(nx * widget.width())),
        int(round(ny * widget.height())),
    )


def _close_flow(flow, console_wrapper, application) -> None:
    console = flow.console
    if console is not None:
        for row in reversed(console.logic_nodes):
            if not row.stop_button.isEnabled():
                continue
            QtTest.QTest.mouseClick(row.stop_button, QtCore.Qt.LeftButton)
            until(
                application,
                lambda current=row: not current.stop_button.isEnabled(),
                timeout=15.0,
            )
    if not widget_gone(console_wrapper):
        console_wrapper.close()
        until(application, lambda: widget_gone(console_wrapper), timeout=15.0)
    flow.finish_close(application, timeout_seconds=15.0)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_camera_live_image_area_second_image_and_fit_use_one_plot_stack(
    tmp_path,
) -> None:
    """Camera, selector-derived Dataset, and both Fits use current zlc_plot."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    require_offscreen_platform(application)
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--name",
            "camera-zlc-plot-flow",
            "--seed",
            "31",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.apply_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = _add_button(console)

        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        row = console.logic_nodes[-1]
        editor = _current_logic_editor(console, application)
        widgets = _visible_form_widgets(editor)
        _choose_combo_data(widgets["camera_role"], "mot_camera", application)
        _replace_spin_value(widgets["frames_per_cycle"], "3")
        _replace_spin_value(widgets["exposure"], "0.013")
        cards_before_start = tuple(console.cards)
        QtTest.QTest.mouseClick(editor.form.start_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: row.status_label.text() == "running",
            timeout=15.0,
        )
        assert tuple(console.cards) == cards_before_start

        frame_signals = tuple(
            console_signal_key(row.node.node_id, f"frame_{index}")
            for index in range(3)
        )
        card = _add_plot_and_bind(
            console,
            add,
            PlotKind.IMAGE,
            frame_signals[1],
            application,
        )
        _wait_for_plot(application, card)
        value = card.presented_value
        schema = value.snapshot.block.schema
        assert schema.repeat_axis.role == REPEAT
        assert schema.repeat_axis.size == 1
        assert schema.point_table.row_count == 1
        assert schema.point_table.columns == ()
        assert schema.grid_topology is None
        assert tuple(axis.role for axis in schema.cell_schema.data_axes) == (
            SPATIAL_Y,
            SPATIAL_X,
        )
        assert value.shape == (1, 1, *schema.cell_schema.data_shape)
        assert value.dtype == np.dtype("uint8")

        first_ref = value.snapshot.ref
        first_sequence = card.plot_widget.presented_front.identity.sequence
        until(
            application,
            lambda: card.presented_value is not None
            and card.presented_value.snapshot.ref != first_ref
            and card.plot_widget.presented_front.identity.sequence > first_sequence,
            timeout=10.0,
        )

        live_fit = _fit_panel(card, application)
        fit_spy = QtTest.QSignalSpy(live_fit.fitAccepted)
        fit_source_ref = card.presented_value.snapshot.ref
        fit_front_sequence = card.plot_widget.presented_front.identity.sequence
        QtTest.QTest.mouseClick(live_fit.fit_button, QtCore.Qt.LeftButton)
        center_x_signal = panel_signal_key(card.panel_id, "fit.center_x")
        until(
            application,
            lambda: len(fit_spy) == 1
            and center_x_signal in _signal_leaf_keys(card.signal_combo)
            and card.plot_widget.presented_front.identity.sequence
            > fit_front_sequence,
            timeout=20.0,
        )
        until(
            application,
            lambda: card.presented_value.snapshot.ref != fit_source_ref,
            timeout=10.0,
        )

        if card.settings_popup.isVisible():
            QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
            until(application, lambda: not card.settings_popup.isVisible())
        if not console.selectors_switch.isChecked():
            QtTest.QTest.mouseClick(console.selectors_switch, QtCore.Qt.LeftButton)
        until(application, lambda: card.plot_widget.interaction_enabled)

        area_signal = panel_signal_key(card.panel_id, "area.data")
        source_publication = card.presented_publication
        selection_spy = QtTest.QSignalSpy(card.selection_ready)
        interaction_error_spy = QtTest.QSignalSpy(card.plot_widget.errorOccurred)
        start = _plot_point(card, 0.20, 0.22)
        stop = _plot_point(card, 0.58, 0.66)
        QtTest.QTest.mousePress(card.plot_widget, QtCore.Qt.LeftButton, pos=start)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        drag_mouse_move(card.plot_widget, stop, QtCore.Qt.LeftButton)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        QtTest.QTest.mouseRelease(card.plot_widget, QtCore.Qt.LeftButton, pos=stop)
        until(
            application,
            lambda: bool(selection_spy) or bool(interaction_error_spy),
            timeout=15.0,
        )
        assert not interaction_error_spy, tuple(interaction_error_spy)
        until(
            application,
            lambda: _signal_value_or_none(console, area_signal) is not None,
            timeout=15.0,
        )

        area_card = _add_plot_and_bind(
            console,
            add,
            PlotKind.IMAGE,
            area_signal,
            application,
        )
        _wait_for_plot(application, area_card)
        area_value = area_card.presented_value
        assert area_value.dtype == value.dtype
        assert area_value.shape[:2] == (1, 1)
        assert area_value.shape[2] < value.shape[2]
        assert area_value.shape[3] < value.shape[3]
        area_publication = area_card.presented_publication
        parents = console._data.direct_parent_publications(area_publication)
        assert len(parents) == 1
        assert parents[0].event_ref.stream_id == source_publication.event_ref.stream_id
        assert area_publication.value(area_signal) is area_value

        roi_fit = _fit_panel(area_card, application)
        roi_fit_spy = QtTest.QSignalSpy(roi_fit.fitAccepted)
        roi_front_sequence = area_card.plot_widget.presented_front.identity.sequence
        QtTest.QTest.mouseClick(roi_fit.fit_button, QtCore.Qt.LeftButton)
        roi_center_signal = panel_signal_key(area_card.panel_id, "fit.center_x")
        until(
            application,
            lambda: len(roi_fit_spy) == 1
            and roi_center_signal in _signal_leaf_keys(area_card.signal_combo)
            and area_card.plot_widget.presented_front.identity.sequence
            > roi_front_sequence,
            timeout=20.0,
        )

        capture_offscreen_window(
            application,
            console,
            tmp_path / "camera-area-second-image-fit.png",
            settle_ms=50,
        )
    finally:
        _close_flow(flow, console_wrapper, application)


def test_occupancy_start_does_not_open_panel_and_manual_binding_displays(
    tmp_path,
) -> None:
    """Occupancy publishes typed signals; the operator alone creates its plot."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    require_offscreen_platform(application)
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    workspace = _workspace_with_pulses(tmp_path, "imaging_template.json")
    args = _build_parser().parse_args(
        [
            "--workspace",
            str(workspace),
            "--name",
            "occupancy-manual-panel-flow",
            "--seed",
            "43",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console = None
    console_wrapper = None
    pulse_body = None
    try:
        QtTest.QTest.mouseClick(devices.apply_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = _add_button(console)

        _choose_combo_text(console.kind_combo, "Task: Calibrate readout", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        calibration_row = console.logic_nodes[-1]
        calibration_editor = _current_logic_editor(console, application)
        calibration_widgets = _visible_form_widgets(calibration_editor)
        _replace_spin_value(calibration_widgets["threshold_frames"], "10")
        QtTest.QTest.mouseClick(
            calibration_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        calibration_signal = console_signal_key(
            calibration_row.node.node_id,
            "calibration",
        )
        until(
            application,
            lambda: _resolved_artifact(console, calibration_signal) is not None,
            timeout=25.0,
        )
        calibration_reference = _resolved_artifact(console, calibration_signal)
        calibration_root = (
            workspace
            / "_output"
            / "calibrations"
            / Path(calibration_reference.record_path).parent
        )
        calibration_runtime = console._last_node[id(calibration_row)]
        until(
            application,
            lambda: calibration_runtime.final_outputs_resolved,
            timeout=25.0,
        )
        assert calibration_runtime.final_output_error is None
        assert calibration_runtime.post_final_warning is None
        report_pages = sorted(
            path.name for path in (calibration_root / "report").glob("*.png")
        )
        assert report_pages == [
            "distribution.png",
            "fidelity.png",
            "site_map.png",
        ]
        assert all(
            (calibration_root / "report" / page).stat().st_size > 0
            for page in report_pages
        )
        assert not (calibration_root / "source_frames.npy").exists()
        assert not (calibration_root / "source_frame_validity.npy").exists()
        assert not (calibration_root / "manifest.json").exists()

        pulse_path = Path("pulses/probe_template.json").resolve()
        pulse_body = flow.pulse
        with patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(pulse_path), "ZLC pulse (*.json)"),
        ):
            QtTest.QTest.mouseClick(
                pulse_body.schedule_view.load_button,
                QtCore.Qt.LeftButton,
            )
        until(
            application,
            lambda: pulse_body._controller.current_path == pulse_path,
            timeout=15.0,
        )
        QtTest.QTest.mouseClick(
            pulse_body.schedule_view.fire_button,
            QtCore.Qt.LeftButton,
        )
        from zlc_neutral_atom.runtime.run import RunState

        until(
            application,
            lambda: pulse_body.active_snapshot is not None
            and pulse_body.active_snapshot.state is RunState.RUNNING,
            timeout=15.0,
        )

        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        camera_row = console.logic_nodes[-1]
        camera_editor = _current_logic_editor(console, application)
        camera_widgets = _visible_form_widgets(camera_editor)
        _choose_combo_data(camera_widgets["camera_role"], "camera", application)
        _replace_spin_value(camera_widgets["repeat"], "0")
        QtTest.QTest.mouseClick(camera_editor.form.start_button, QtCore.Qt.LeftButton)
        camera_signal = console_signal_key(camera_row.node.node_id, "frame_0")
        until(
            application,
            lambda: camera_row.status_label.text() == "running",
            timeout=20.0,
        )
        camera_card = _add_plot_and_bind(
            console,
            add,
            PlotKind.IMAGE,
            camera_signal,
            application,
        )
        until(
            application,
            lambda: _plot_front_or_none(camera_card) is not None,
            timeout=15.0,
        )

        _choose_combo_text(
            console.kind_combo,
            "Processor: Judge occupancy",
            application,
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        occupancy_row = console.logic_nodes[-1]
        occupancy_editor = _current_logic_editor(console, application)
        occupancy_widgets = _visible_form_widgets(occupancy_editor)
        _choose_signal_leaf(
            occupancy_widgets["camera_frame"],
            camera_signal,
            application,
        )
        _choose_signal_leaf(
            occupancy_widgets["calibration_output"],
            calibration_signal,
            application,
        )
        cards_before_start = tuple(console.cards)
        QtTest.QTest.mouseClick(
            occupancy_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        occupied_signal = console_signal_key(
            occupancy_row.node.node_id,
            "occupied",
        )
        rate_signal = console_signal_key(occupancy_row.node.node_id, "rate")
        deadline = time.monotonic() + 20.0
        while (
            occupancy_row.status_label.text() != "running"
            and not occupancy_row.status_label.text().startswith(
                ("failed", "rejected")
            )
            and time.monotonic() < deadline
        ):
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            time.sleep(0.005)
        assert occupancy_row.status_label.text() == "running", (
            occupancy_row.status_label.text()
        )
        until(
            application,
            lambda: _signal_value_or_none(console, occupied_signal) is not None,
            timeout=20.0,
        )
        assert tuple(console.cards) == cards_before_start

        occupied_card = _add_plot_and_bind(
            console,
            add,
            PlotKind.CURVE,
            occupied_signal,
            application,
        )
        _wait_for_plot(application, occupied_card, timeout=20.0)
        occupied = occupied_card.presented_value
        assert occupied.shape[:2] == (1, 1)
        assert occupied.shape[2] > 1
        assert occupied.dtype == np.dtype("bool")
        assert rate_signal in _signal_leaf_keys(occupied_card.signal_combo)
        first_ref = occupied.snapshot.ref
        until(
            application,
            lambda: occupied_card.presented_value.snapshot.ref != first_ref,
            timeout=10.0,
        )
    finally:
        if (
            pulse_body is not None
            and pulse_body.active_snapshot is not None
            and not pulse_body.active_snapshot.state.terminal
        ):
            QtTest.QTest.mouseClick(
                pulse_body.schedule_view.safe_button,
                QtCore.Qt.LeftButton,
            )
            until(
                application,
                lambda: pulse_body.active_snapshot is not None
                and pulse_body.active_snapshot.state.terminal,
                timeout=15.0,
            )
        if console is not None:
            for row in reversed(console.logic_nodes):
                if row.stop_button.isEnabled():
                    QtTest.QTest.mouseClick(
                        row.stop_button,
                        QtCore.Qt.LeftButton,
                    )
                    until(
                        application,
                        lambda current=row: not current.stop_button.isEnabled(),
                        timeout=15.0,
                    )
        _close_flow(flow, console_wrapper, application)
