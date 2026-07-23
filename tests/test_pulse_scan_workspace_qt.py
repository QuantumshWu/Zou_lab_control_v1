"""Human-control coverage for the formal PulseGUI scan workspace wiring."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_pulse import (
    FIELD_DURATION,
    PulseFieldRef,
    ScanParameter,
    load_deployed_pulse_target,
    new_pulse_document,
)
from zlc_workbench.pulse import PulseEditorSession
from zlc_workbench.pulse_editor.controller import PulseEditorController
from zlc_workbench.pulse_editor.window import PulseEditorWindowBody


def _until(application, predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _click_tab(body: PulseEditorWindowBody, page: QtWidgets.QWidget) -> None:
    index = body.tabs.indexOf(page)
    bar = body.tabs.tabBar()
    QtTest.QTest.mouseClick(
        bar,
        QtCore.Qt.LeftButton,
        pos=bar.tabRect(index).center(),
    )


def _body() -> tuple[PulseEditorWindowBody, PulseEditorController]:
    document = new_pulse_document(load_deployed_pulse_target(), time_step_ns=20)
    field = PulseFieldRef(FIELD_DURATION, document.periods[0].period_id)
    document = replace(
        document,
        scan_parameters=(ScanParameter("duration", field, "Duration", "ns"),),
    )
    controller = PulseEditorController(PulseEditorSession(document))
    return PulseEditorWindowBody(controller), controller


def test_formal_scan_controls_drive_workspace_and_exact_file_dialogs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    application = ensure_qt_app()
    monkeypatch.setenv("ZLC_PULSE_DIR", str(tmp_path))
    program_path = tmp_path / "scan.txt"
    program_path.write_text(
        "import time\ntime.sleep(0.15)\nscan_table = [[300], [400]]\n",
        encoding="utf-8",
    )
    array_path = tmp_path / "loaded.csv"
    array_path.write_text("500\n600\n", encoding="utf-8")
    exported_path = tmp_path / "operator_export.csv"
    dialog_calls: list[tuple[str, str, str]] = []

    def open_dialog(_parent, title, start, file_filter):
        dialog_calls.append((str(title), str(start), str(file_filter)))
        if title == "Load scan program / table":
            return str(program_path), file_filter
        if title == "Load scan array":
            return str(array_path), file_filter
        raise AssertionError(f"unexpected open dialog {title!r}")

    def save_dialog(_parent, title, start, file_filter):
        dialog_calls.append((str(title), str(start), str(file_filter)))
        assert title == "Save scan array"
        return str(exported_path), file_filter

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(open_dialog),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(save_dialog),
    )

    body, controller = _body()
    body.show()
    try:
        messages: list[str] = []
        body._message = messages.append
        _until(
            application,
            lambda: body.scan_view.scan_code.toPlainText()
            == controller.snapshot().scan_workspace.source_text,
        )
        empty_switch = body.schedule_view.channel_panel.scan_source_toggle
        QtTest.QTest.mouseClick(
            empty_switch,
            QtCore.Qt.LeftButton,
            pos=QtCore.QPoint(10, empty_switch.height() // 2),
        )
        _until(application, lambda: not empty_switch.isChecked())
        assert controller.current_scan_workspace.selected_source == "generated"
        assert messages == ["no loaded scan array is available"]

        _click_tab(body, body.scan_view)

        editor = body.scan_view.scan_code
        controller_source = controller.current_scan_workspace.source_text
        QtTest.QTest.mouseClick(editor.viewport(), QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        QtTest.QTest.keyClicks(editor, "scan_table = [[100], [200]]")
        assert body.scan_view.code_dirty
        assert controller.current_scan_workspace.source_text == controller_source

        # The UI draft is authoritative before Run: even the already-selected
        # template must replace it rather than no-op against controller state.
        QtTest.QTest.mouseClick(
            body.scan_view.scan_column_template_button,
            QtCore.Qt.LeftButton,
        )
        assert body.scan_view.scan_code.toPlainText() == controller_source
        QtTest.QTest.mouseClick(editor.viewport(), QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        QtTest.QTest.keyClicks(editor, "scan_table = [[100], [200]]")
        QtTest.QTest.mouseClick(
            body.scan_view.scan_run_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: controller.snapshot().document.scan_table is not None
            and controller.snapshot().document.scan_table.rows == ((100,), (200,)),
        )
        assert not body.scan_view.scan_run_button.text().endswith("*")

        QtTest.QTest.mouseClick(
            body.scan_view.scan_grid_template_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: "np.meshgrid" in body.scan_view.scan_code.toPlainText(),
        )
        assert body.scan_view.scan_run_button.text().endswith("*")

        QtTest.QTest.mouseClick(
            body.scan_view.scan_load_program_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.scan_view.scan_code.toPlainText().startswith("import time"),
        )
        QtTest.QTest.mouseClick(
            body.scan_view.scan_run_button,
            QtCore.Qt.LeftButton,
        )
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert not body.scan_view.scan_run_button.isEnabled()
        assert not body.scan_view.scan_load_program_button.isEnabled()
        assert not body.schedule_view.channel_panel.load_button.isEnabled()
        assert not body.schedule_view.channel_panel.scan_source_toggle.isEnabled()
        _until(
            application,
            lambda: controller.snapshot().document.scan_table is not None
            and controller.snapshot().document.scan_table.rows == ((300,), (400,)),
        )
        assert body.scan_view.scan_run_button.isEnabled()

        _click_tab(body, body.schedule_view)
        QtTest.QTest.mouseClick(
            body.schedule_view.channel_panel.load_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: controller.snapshot().scan_workspace.selected_source == "loaded"
            and controller.snapshot().document.scan_table.rows == ((500,), (600,)),
        )
        switch = body.schedule_view.channel_panel.scan_source_toggle
        assert switch.isChecked()
        QtTest.QTest.mouseClick(
            switch,
            QtCore.Qt.LeftButton,
            pos=QtCore.QPoint(10, switch.height() // 2),
        )
        _until(
            application,
            lambda: controller.snapshot().scan_workspace.selected_source == "generated"
            and controller.snapshot().document.scan_table.rows == ((300,), (400,)),
        )
        QtTest.QTest.mouseClick(
            switch,
            QtCore.Qt.LeftButton,
            pos=QtCore.QPoint(10, switch.height() // 2),
        )
        _until(
            application,
            lambda: controller.snapshot().scan_workspace.selected_source == "loaded",
        )

        _click_tab(body, body.scan_view)
        QtTest.QTest.mouseClick(
            body.scan_view.scan_save_array_button,
            QtCore.Qt.LeftButton,
        )
        _until(application, exported_path.exists)
        assert exported_path.read_text(encoding="utf-8").splitlines() == [
            "5.000000000000000000e+02",
            "6.000000000000000000e+02",
        ]

        root = str(tmp_path.resolve())
        assert dialog_calls == [
            (
                "Load scan program / table",
                root,
                "Scan program or current table (*.py *.txt *.npy *.csv *.json);;"
                "Python scan program (*.py *.txt);;"
                "Scan array (*.npy *.csv);;"
                "Current PulseDocument with scan table (*.json)",
            ),
            (
                "Load scan array",
                root,
                "Scan array (*.npy *.csv *.txt *.json)",
            ),
            (
                "Save scan array",
                str(tmp_path / "Untitled_pulse_scan.npy"),
                "Scan array (*.npy *.csv)",
            ),
        ]
        assert body.scan_view.scan_table_view.toPlainText().splitlines()[0] == "duration"
        assert body.schedule_view.channel_panel.scan_file_label.text() == str(array_path)
    finally:
        body.request_close(discard_unsaved=True)
        _until(application, lambda: controller.snapshot().close_complete)
        body.close()
        body.deleteLater()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    assert controller.worker_idle
