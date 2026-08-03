"""Human-control scan progress, hold, and step on the formal Pulse GUI."""

from __future__ import annotations

from dataclasses import replace
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest

from Zou_lab_control.api import WorkspacePaths, connect
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_neutral_atom.runtime.run import RunState
from zlc_pulse import (
    FIELD_DURATION,
    FrozenScanTable,
    PulseFieldRef,
    ScanParameter,
    load_deployed_pulse_target,
    new_pulse_document,
)


def _until(application, predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _click_tab(body, page) -> None:
    index = body.tabs.indexOf(page)
    bar = body.tabs.tabBar()
    QtTest.QTest.mouseClick(
        bar,
        QtCore.Qt.LeftButton,
        pos=bar.tabRect(index).center(),
    )


def test_operator_observes_holds_steps_and_stops_virtual_scan(tmp_path) -> None:
    application = ensure_qt_app()
    target = load_deployed_pulse_target()
    document = new_pulse_document(target, time_step_ns=20)
    field = PulseFieldRef(FIELD_DURATION, document.periods[0].period_id)
    document = replace(
        document,
        scan_parameters=(ScanParameter("duration", field, "Duration", "ns"),),
        scan_table=FrozenScanTable(
            ("duration",),
            ((1000,), (2000,), (3000,)),
        ),
    )
    experiment = connect(
        "virtual",
        workspace=WorkspacePaths.for_workspace((tmp_path / "workspace").resolve()),
    )
    body = experiment.pulse_gui(document=document)
    try:
        QtTest.QTest.mouseClick(
            body.schedule_view.fire_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.active_snapshot is not None
            and body.active_snapshot.state is RunState.RUNNING,
        )
        run_id = body.active_snapshot.run_id

        _click_tab(body, body.scan_view)
        _until(
            application,
            lambda: body.scan_view.scan_progress_label.text().startswith(
                "Scan: point "
            )
            and "duration = " in body.scan_view.scan_progress_label.text(),
        )

        QtTest.QTest.mouseClick(
            body.scan_view.scan_hold_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body._controller.runtime_update().held_scan_point is not None
            and body.scan_view.scan_progress_label.text().startswith(
                "held at point "
            ),
        )
        assert body.active_snapshot is not None
        assert body.active_snapshot.run_id == run_id
        held = body._controller.runtime_update().held_scan_point
        assert held is not None
        old_index, total, _values = held
        if old_index < total - 1:
            button = body.scan_view.scan_step_forward_button
            expected = old_index + 1
        else:
            button = body.scan_view.scan_step_back_button
            expected = old_index - 1
        QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: body._controller.runtime_update().held_scan_point is not None
            and body._controller.runtime_update().held_scan_point[0] == expected,
        )
        assert body.active_snapshot is not None
        assert body.active_snapshot.run_id == run_id
        assert body.scan_view.scan_progress_label.text().startswith(
            f"held at point {expected + 1}/{total}:"
        )

        QtTest.QTest.mouseClick(
            body.schedule_view.safe_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.active_snapshot is not None
            and body.active_snapshot.state is RunState.CANCELLED
            and not body._controller.runtime_update().run_busy,
        )
    finally:
        experiment.close()
        _until(
            application,
            lambda: body.permanently_closed and body.worker_idle,
        )
