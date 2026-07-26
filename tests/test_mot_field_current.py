"""Current formal MOT-field product flow over the virtual installation."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from gui_user_flow import (
    configure_offscreen_fast_path,
    require_offscreen_platform,
    until,
    widget_gone,
)
from test_task_console_camera_user_flow import (
    _choose_combo_text,
    _replace_path_value,
    _replace_spin_value,
    _visible_form_widgets,
)
from zlc_frontend.qt_widgets import FigureSurfaceHost, ensure_qt_app
from zlc_workbench.task_console.console_records import console_signal_key


def _faceted_overview(console, signal_key: str):
    for card in console.cards:
        if card.config.signal != signal_key or card.config.kind != "grid":
            continue
        if isinstance(card.board, FigureSurfaceHost) and card.board.faceted:
            return card.board.overview_artifact
    return None


def test_mot_field_form_runs_live_and_final_named_axis_grids(tmp_path: Path) -> None:
    """The visible Start path must draw data, not merely create a blank card."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    require_offscreen_platform(application)
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    flow = _StandaloneTaskConsoleFlow(
        _build_parser().parse_args(
            [
                "--repository",
                str(tmp_path / "workspace"),
                "--name",
                "mot-field-current",
                "--seed",
                "37",
            ]
        )
    )
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

        _choose_combo_text(
            console.kind_combo,
            "Task: Optimize MOT field",
            application,
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        row = console.logic_nodes[-1]
        editor = console._logic_editors[id(row)]
        widgets = _visible_form_widgets(editor)
        _replace_spin_value(widgets["points"], "2")
        _replace_path_value(widgets["folder"], str(tmp_path / "mot-report"))

        QtTest.QTest.mouseClick(
            editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        live_key = console_signal_key(row.node.node_id, "grid")
        final_key = console_signal_key(row.node.node_id, "mot_field")
        saw_running = False
        live_value = None
        live_overview = None
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            saw_running = saw_running or (
                row.status_label.text() == "running"
                and editor.form.status.text() == "running"
            )
            live_value = console._data.freeze().value(live_key) or live_value
            live_overview = _faceted_overview(console, live_key) or live_overview
            if not console._task_locked:
                break
            time.sleep(0.005)

        assert saw_running
        assert live_value is not None
        assert live_overview is not None
        assert len(live_overview.regions) == 2
        live_schema = live_value.schema
        assert live_schema.point_layout.logical_shape == (2, 2, 2)
        assert tuple(
            (axis.axis_id.value, axis.role.value)
            for axis in live_schema.point_axes
        ) == (
            ("mot-field.da_x", "scan-point"),
            ("mot-field.da_y", "scan-point"),
            ("mot-field.da_z", "scan-point"),
        )
        assert not console._task_locked

        final_value = console._data.freeze().value(final_key)
        assert final_value is not None
        until(
            application,
            lambda: _faceted_overview(console, final_key) is not None,
            timeout=15.0,
        )
        final_overview = _faceted_overview(console, final_key)
        assert final_overview is not None
        assert len(final_overview.regions) == 2
        final_schema = final_value.schema
        assert final_schema.point_layout.logical_shape == (2, 2, 2)
        assert final_schema.physical_shape == (1, 8, 1)
        assert tuple(axis.axis_id.value for axis in final_schema.point_axes) == (
            "mot-field.da_x",
            "mot-field.da_y",
            "mot-field.da_z",
        )
        assert float(np.ptp(final_value.snapshot.block.values)) > 0.0
    finally:
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(
                application,
                lambda: widget_gone(console_wrapper),
                timeout=15.0,
            )
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
