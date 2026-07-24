"""Formal operator flow: DeviceManager -> Camera -> signal -> live 2-D panel.

The fixture chooses only the offscreen Qt platform.  Every product transition is
driven through the same visible controls as the desktop launcher; assertions may
inspect the resulting typed fronts, but never bypass a button to create them.
"""

from __future__ import annotations

import os
import time


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from gui_user_flow import (
    capture_offscreen_window,
    click_tab,
    configure_offscreen_fast_path,
    until,
)
from zlc_data.console_records import console_signal_key
from zlc_frontend.qt_widgets import ensure_qt_app


def _choose_combo_data(combo, value, application) -> None:
    """Choose a normal combo entry through its visible popup and keyboard."""

    row = combo.findData(value)
    assert row >= 0, (value, [combo.itemData(i) for i in range(combo.count())])
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    view = combo.view()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _ in range(row):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    assert combo.currentData() == value


def _choose_combo_text(combo, text, application) -> None:
    """Choose one visible menu label without depending on QVariant coercion."""

    row = combo.findText(text)
    assert row >= 0, (text, [combo.itemText(i) for i in range(combo.count())])
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    view = combo.view()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _ in range(row):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    assert combo.currentText() == text


def _choose_signal_leaf(combo, signal, application) -> None:
    """Expand the visible producer row and click the requested signal leaf."""

    # Logic Edit forms are real scroll pages.  A user must bring a low field
    # into view before its below-anchored popup has usable screen height.
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
    # Expanding changes the popup's fitted height.  Let that formal widget
    # transition finish, then scroll the actual leaf into the visible viewport
    # before clicking it just as an operator would.
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    if not view.isVisible():
        QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
        until(application, view.isVisible)
    view.scrollTo(child_index)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    assert view.visualRect(child_index).isValid()
    click_position = view.visualRect(child_index).center()
    assert combo.isEnabled() and view.isEnabled() and view.isVisible()
    assert view.rect().contains(click_position), (
        view.visualRect(child_index),
        view.rect(),
    )
    hit = view.indexAt(click_position)
    assert hit == child_index, (
        view.visualRect(child_index),
        view.rect(),
        hit.data(QtCore.Qt.UserRole),
        child_index.data(QtCore.Qt.UserRole),
    )
    clicked = QtTest.QSignalSpy(view.clicked)
    picked = QtTest.QSignalSpy(combo.signalPicked)
    QtTest.QTest.mouseClick(
        view.viewport(),
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    assert len(clicked) == 1, "the visible tree leaf did not receive a Qt click"
    assert len(picked) == 1, "the tree combo did not publish the selected leaf"
    until(application, lambda: combo.currentData() == signal)


def _signal_leaf_keys(combo) -> set[str]:
    """Return the exact leaves currently visible through the product picker."""

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


def _replace_spin_value(spin, text: str) -> None:
    """Edit a visible spin box exactly as an operator would."""

    edit = spin.lineEdit()
    QtTest.QTest.mouseClick(edit, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(edit, str(text))
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_Return)


def _replace_path_value(path_widget, text: str) -> None:
    """Edit the text field of the shared visible path control."""

    edit = path_widget.edit
    QtTest.QTest.mouseClick(edit, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(edit, str(text))
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_Return)


def test_device_manager_camera_signal_drives_a_changing_2d_front(tmp_path) -> None:
    """The actual standalone entry's camera chain is live and dimensioned."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "camera-user-flow",
            "--seed",
            "31",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()

        assert sum(
            console.kind_combo.itemText(index) == "Measurement: Camera"
            for index in range(console.kind_combo.count())
        ) == 1
        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        assert len(console.logic_nodes) == 1
        row = console.logic_nodes[0]
        editor = console._logic_editors[id(row)]

        # Virtual MOT camera is a true free-running source, so this is the
        # deterministic live Camera role for an operator-path acceptance run.
        role_combo = editor.form._widgets["camera_role"]
        _choose_combo_data(role_combo, "mot_camera", application)
        QtTest.QTest.mouseClick(editor.form.start_button, QtCore.Qt.LeftButton)
        signal = console_signal_key(row.node.title, "frame")
        until(
            application,
            lambda: console._data.freeze().value(signal) is not None,
            timeout=15.0,
        )
        first_value = console._data.freeze().value(signal)
        until(
            application,
            lambda: (
                (value := console._data.freeze().value(signal)) is not None
                and value.snapshot.ref != first_value.snapshot.ref
            ),
            timeout=10.0,
        )

        # The visible Logic row must expose the data dimensions, not merely an
        # unbound signal name.
        until(
            application,
            lambda: (
                "frame" in row.publishes_label.text()
                and "—" not in row.publishes_label.text()
            ),
            timeout=3.0,
        )

        _choose_combo_data(console.kind_combo, "2d", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        assert len(console.cards) == 1
        card = console.cards[0]
        click_tab(console, console.tabs.widget(0))
        QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        until(application, lambda: card.settings_popup.isVisible())
        _choose_signal_leaf(card.signal_combo, signal, application)
        assert card.config.signal == signal

        until(
            application,
            lambda: card.board is not None and card.board.front_frame is not None,
            timeout=15.0,
        )
        first_front = card.board.front_frame
        until(
            application,
            lambda: (
                card.board.front_frame is not None
                and card.board.front_frame.sequence > first_front.sequence
            ),
            timeout=10.0,
        )
        second_front = card.board.front_frame
        assert bytes(second_front.panels[0].raster.pixels) != bytes(
            first_front.panels[0].raster.pixels
        )

        # No-button movement is inert: the board does not even request tracking.
        board = card.board.board
        assert not board.hasMouseTracking()
        before = card.frozen_figure_output_state()[1:3]
        QtTest.QTest.mouseMove(board, board.rect().center())
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert card.frozen_figure_output_state()[1:3] == before

        capture_offscreen_window(
            application,
            console,
            tmp_path / "camera-live-2d.png",
            settle_ms=100,
        )

        # Open the real per-panel Edit surface from the Setting popup.  It must
        # reuse the accepted front inside the tab, not launch a second window.
        if not card.settings_popup.isVisible():
            QtTest.QTest.mouseClick(card.setting_button, QtCore.Qt.LeftButton)
        until(application, lambda: card.settings_popup.isVisible())
        QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: (
                id(card) in console._panel_editors
                and console._panel_editors[id(card)]._board is not None
            ),
        )
        edit = console._panel_editors[id(card)]
        assert edit.window() is console.window()
        assert edit._board.isVisible()
        capture_offscreen_window(
            application,
            console,
            tmp_path / "camera-live-edit.png",
            settle_ms=100,
        )
        edit_scroll = next(
            scroll
            for scroll in edit.findChildren(QtWidgets.QScrollArea)
            if scroll.isVisible()
        )
        QtTest.QTest.keyClick(
            edit_scroll.verticalScrollBar(),
            QtCore.Qt.Key_End,
        )
        capture_offscreen_window(
            application,
            console,
            tmp_path / "camera-live-edit-scrolled.png",
            settle_ms=100,
        )
    finally:
        if console_wrapper is not None and console_wrapper.isVisible():
            console_wrapper.close()
            until(application, lambda: not console_wrapper.isVisible(), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_calibration_and_mot_tasks_open_their_declared_live_panels(tmp_path) -> None:
    """The two flagship tasks run from their real forms and open typed panels."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "task-user-flow",
            "--seed",
            "37",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        _choose_combo_text(console.kind_combo, "Task: Calibrate readout", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        calibration_row = console.logic_nodes[-1]
        calibration_editor = console._logic_editors[id(calibration_row)]
        calibration_widgets = calibration_editor.form._widgets
        assert set(calibration_widgets) == {
            "source_mode",
            "folder",
            "save_frames",
            "pulse",
            "threshold_method",
            "reference_exposure_s",
            "readout_exposure_s",
            "threshold_frames",
            "roi_radius",
            "camera_role",
        }
        _replace_path_value(
            calibration_widgets["folder"],
            str(tmp_path / "calibration-output"),
        )
        _replace_spin_value(calibration_widgets["threshold_frames"], "10")
        QtTest.QTest.mouseClick(
            calibration_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )

        calibration_frame = console_signal_key(calibration_row.node.title, "frame")
        saw_calibration_panel = False
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            saw_calibration_panel = saw_calibration_panel or any(
                card.config.signal == calibration_frame
                and card.config.kind == "2d"
                for card in console.cards
            )
            if not console._task_locked:
                break
            time.sleep(0.005)
        assert saw_calibration_panel
        assert not console._task_locked
        calibration_final = console_signal_key(
            calibration_row.node.title,
            "calibration",
        )
        assert console._data.freeze().value(calibration_final) is not None

        _choose_combo_text(console.kind_combo, "Task: Optimize MOT field", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        mot_row = console.logic_nodes[-1]
        mot_editor = console._logic_editors[id(mot_row)]
        mot_widgets = mot_editor.form._widgets
        assert set(mot_widgets) == {
            "pulse",
            "center_x",
            "center_y",
            "center_z",
            "span",
            "points",
            "roi_cx",
            "roi_cy",
            "roi_radius",
            "folder",
            "camera_role",
        }
        _replace_spin_value(mot_widgets["points"], "2")
        _replace_path_value(
            mot_widgets["folder"],
            str(tmp_path / "mot-output"),
        )
        QtTest.QTest.mouseClick(
            mot_editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        mot_grid = console_signal_key(mot_row.node.title, "grid")
        saw_mot_panel = False
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            saw_mot_panel = saw_mot_panel or any(
                card.config.signal == mot_grid
                and card.config.kind == "grid"
                for card in console.cards
            )
            if not console._task_locked:
                break
            time.sleep(0.005)
        assert saw_mot_panel
        assert not console._task_locked
        mot_final = console_signal_key(mot_row.node.title, "mot_field")
        assert console._data.freeze().value(mot_final) is not None
    finally:
        if console_wrapper is not None and console_wrapper.isVisible():
            console_wrapper.close()
            until(application, lambda: not console_wrapper.isVisible(), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_pulse_scan_exposes_its_table_and_runs_from_one_camera_definition(
    tmp_path,
) -> None:
    """The formal Pulse-scan row owns the Main table editor and exact source."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    args = _build_parser().parse_args(
        [
            "--repository",
            str(tmp_path / "workspace"),
            "--name",
            "pulse-scan-user-flow",
            "--seed",
            "41",
        ]
    )
    flow = _StandaloneTaskConsoleFlow(args)
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        # There is one Camera definition.  Pulse scan resolves its declared
        # frame output into a dedicated exact acquisition; a second hidden
        # CameraCapture/CameraMonitor definition is neither needed nor offered.
        _choose_combo_text(console.kind_combo, "Measurement: Camera", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        camera_row = console.logic_nodes[-1]
        camera_signal = console_signal_key(camera_row.node.title, "frame")

        _choose_combo_text(console.kind_combo, "Measurement: Pulse scan", application)
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        scan_row = console.logic_nodes[-1]
        scan_editor = console._logic_editors[id(scan_row)]
        widgets = scan_editor.form._widgets
        assert set(widgets) == {"pulse", "pulse_slots", "y_signal"}

        slots = widgets["pulse_slots"]
        assert slots.isVisible()
        assert slots._program_code.isVisible()
        assert "scan_table" in slots._program_code.toPlainText()
        assert slots._sweep_combo.currentText() in {
            "Scan slots (hardware table)",
            "API slots (one pulse per point)",
        }
        scan_spec = console._spec_for_logic(scan_row.node)
        assert scan_spec is not None
        y_parameter = next(
            parameter for parameter in scan_spec.params
            if parameter.key == "y_signal"
        )
        assert y_parameter.label == "Exact source (y)"
        assert _signal_leaf_keys(widgets["y_signal"]) == {camera_signal}
        _choose_signal_leaf(widgets["y_signal"], camera_signal, application)

        QtTest.QTest.mouseClick(scan_editor.form.start_button, QtCore.Qt.LeftButton)
        scan_signal = console_signal_key(scan_row.node.title, "scan")
        # FINAL-only outputs do not manufacture an empty viewer at Start.
        assert not any(card.config.signal == scan_signal for card in console.cards)
        saw_declared_panel = False
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            saw_declared_panel = saw_declared_panel or any(
                card.config.signal == scan_signal
                for card in console.cards
            )
            if console._data.freeze().value(scan_signal) is not None:
                break
            time.sleep(0.005)
        assert saw_declared_panel
        value = console._data.freeze().value(scan_signal)
        if value is None:
            raise AssertionError(scan_row.status_label.text())
        scan_card = next(
            card for card in console.cards if card.config.signal == scan_signal
        )
        data_roles = {
            axis.role for axis in value.snapshot.block.schema.cell_schema.data_axes
        }
        from zlc_data import SPATIAL_X, SPATIAL_Y

        assert scan_card.config.kind == (
            "2d" if {SPATIAL_X, SPATIAL_Y}.issubset(data_roles) else "1d"
        )
        assert value.snapshot.block.schema.repeat_axis.size >= 1
        assert value.snapshot.block.schema.point_layout.storage_size >= 1
    finally:
        if console_wrapper is not None and console_wrapper.isVisible():
            console_wrapper.close()
            until(application, lambda: not console_wrapper.isVisible(), timeout=15.0)
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
