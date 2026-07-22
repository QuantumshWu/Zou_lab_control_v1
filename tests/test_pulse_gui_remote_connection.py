"""Human-control remote execution through the one formal Pulse GUI."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from conftest import pulse_backend_completion_for
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_neutral_atom.runtime.run import RunState
from zlc_pulse import (
    FIELD_DURATION,
    FrozenScanTable,
    PulseExecutionService,
    PulseFieldRef,
    ScanParameter,
    load_deployed_pulse_target,
    new_pulse_document,
    save_pulse_document,
)
from zlc_pulse.server import serve_pulse_execution_service
from zlc_workbench.pulse_editor.app import open_pulse_editor


class _Backend:
    def __init__(self) -> None:
        self.prepared = None
        self.safe = True
        self.actions: list[str] = []

    def prepare(self, artifact) -> None:
        self.prepared = artifact
        self.safe = False
        self.actions.append("prepare")

    def fire(self, artifact) -> None:
        assert artifact is self.prepared
        self.actions.append("fire")

    def await_completion(self, artifact, _timeout):
        assert artifact is self.prepared
        self.actions.append("complete")
        return pulse_backend_completion_for(
            artifact,
            transport_id="pulse-gui-remote-test",
        )

    def wait_continuous_failure(self, artifact, timeout):
        assert artifact is self.prepared
        time.sleep(float(timeout))
        return None

    def safe_state(self) -> None:
        self.prepared = None
        self.safe = True
        self.actions.append("safe")

    def request_interrupt(self) -> None:
        return None

    def snapshot(self):
        return {"safe": self.safe}


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


def _choose_remote(body) -> None:
    combo = body.schedule_view.conn_target_combo
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    view = combo.view()
    remote_index = combo.model().index(combo.findData("remote"), 0)
    application = QtWidgets.QApplication.instance()
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    remote_position = view.visualRect(remote_index).center()
    QtTest.QTest.mouseMove(view.viewport(), remote_position)
    QtTest.QTest.mousePress(
        view.viewport(), QtCore.Qt.LeftButton, pos=remote_position
    )
    QtTest.QTest.mouseRelease(
        view.viewport(), QtCore.Qt.LeftButton, pos=remote_position
    )
    assert combo.currentData() == "remote"
    # The owner snapshot timer ticks every 40 ms.  Keep the draft selection
    # untouched for multiple ticks before the operator enters an address or
    # presses Connect; otherwise this regression can pass by racing the timer.
    deadline = time.monotonic() + 0.15
    while time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert combo.currentData() == "remote"


def _enter_address(body, endpoint: str) -> None:
    editor = body.schedule_view.conn_addr_edit
    QtTest.QTest.mouseClick(editor, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(editor, endpoint)


def _set_scan_repeats(body, count: int) -> None:
    _click_tab(body, body.scan_view)
    spin = body.scan_view.scan_repeats_spin
    QtTest.QTest.mouseClick(spin, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(spin, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(spin, str(count))
    QtTest.QTest.keyClick(spin, QtCore.Qt.Key_Return)


def _scan_document():
    document = new_pulse_document(
        load_deployed_pulse_target(),
        time_step_ns=20,
    )
    parameter = ScanParameter(
        "duration",
        PulseFieldRef(FIELD_DURATION, document.periods[0].period_id),
        "Duration",
        "ns",
    )
    return replace(
        document,
        scan_parameters=(parameter,),
        scan_table=FrozenScanTable(("duration",), ((1000,), (2000,))),
    )


def _run_remote_gui(workspace: Path) -> None:
    backend = _Backend()
    service = PulseExecutionService(
        load_deployed_pulse_target(),
        clock_hz=50e6,
        backend=backend,
    )
    server = serve_pulse_execution_service(
        service,
        host="127.0.0.1",
        port=0,
        start=False,
    )
    server_thread: threading.Thread | None = None
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        unavailable_port = reservation.getsockname()[1]

    application = ensure_qt_app()
    body = open_pulse_editor(repository=workspace, document=_scan_document())
    wrapper = body.window()
    try:
        _choose_remote(body)
        _enter_address(body, f"127.0.0.1:{unavailable_port}")
        QtTest.QTest.mouseClick(
            body.schedule_view.conn_connect_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body._controller.snapshot().connection_state == "offline"
            and "Connection failed" in body._controller.snapshot().diagnostic,
        )
        assert body.schedule_view.conn_connect_button.isEnabled()

        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        endpoint = f"127.0.0.1:{server.port}"
        _enter_address(body, endpoint)
        QtTest.QTest.mouseClick(
            body.schedule_view.conn_connect_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body._controller.snapshot().connection_state == "ready"
            and body.schedule_view.conn_status.text() == endpoint,
        )

        _set_scan_repeats(body, 1)
        _until(
            application,
            lambda: body.current_document.scan_sweep_count == 1,
        )
        _click_tab(body, body.schedule_view)
        QtTest.QTest.mouseClick(
            body.schedule_view.fire_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.active_snapshot is not None
            and body.active_snapshot.state is RunState.SUCCEEDED
            and not body._controller.snapshot().run_busy,
        )

        _set_scan_repeats(body, 0)
        _until(
            application,
            lambda: body.current_document.scan_sweep_count == 0,
        )
        _click_tab(body, body.schedule_view)
        QtTest.QTest.mouseClick(
            body.schedule_view.fire_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.active_snapshot is not None
            and body.active_snapshot.state is RunState.RUNNING,
        )
        QtTest.QTest.mouseClick(
            body.schedule_view.safe_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.active_snapshot is not None
            and body.active_snapshot.state is RunState.CANCELLED
            and not body._controller.snapshot().run_busy,
        )
    finally:
        body.request_close(discard_unsaved=True)
        _until(application, lambda: body._controller.snapshot().close_complete)
        _until(application, lambda: not wrapper.isVisible())
        server.close()
        if server_thread is not None:
            server_thread.join(timeout=3.0)

    assert body.worker_idle
    assert server_thread is not None and not server_thread.is_alive()
    assert service.snapshot()["state"] == "SAFE"
    assert backend.actions.count("prepare") == 2
    assert backend.actions.count("fire") == 2
    assert backend.actions.count("complete") == 1
    assert backend.actions[-1] == "safe"


def _run_load_before_remote(workspace: Path) -> None:
    backend = _Backend()
    deployed = load_deployed_pulse_target()
    service = PulseExecutionService(deployed, clock_hz=50e6, backend=backend)
    server = serve_pulse_execution_service(
        service,
        host="127.0.0.1",
        port=0,
        start=False,
    )
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    document_path = workspace / "incompatible-document.json"
    save_pulse_document(
        new_pulse_document(
            deployed,
            time_step_ns=10,
            name="Must load before remote preflight",
        ),
        document_path,
    )

    application = ensure_qt_app()
    body = open_pulse_editor(
        repository=workspace,
        path=document_path,
        remote_endpoint=f"127.0.0.1:{server.port}",
    )
    wrapper = body.window()
    try:
        _until(
            application,
            lambda: body.current_document.name
            == "Must load before remote preflight"
            and body._controller.snapshot().connection_state == "offline"
            and "clock grid differs" in body._controller.snapshot().diagnostic,
        )
        assert body.schedule_view.conn_connect_button.isEnabled()
    finally:
        body.request_close(discard_unsaved=True)
        _until(application, lambda: body._controller.snapshot().close_complete)
        _until(application, lambda: not wrapper.isVisible())
        server.close()
        server_thread.join(timeout=3.0)

    assert not server_thread.is_alive()
    assert service.snapshot()["state"] == "SAFE"


def test_operator_connects_remote_runs_finite_and_continuous_then_stops_safe(
    tmp_path,
) -> None:
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=Path(__file__).parents[1],
        env=environment,
        timeout=30,
        check=True,
    )


def test_document_is_loaded_before_automatic_remote_preflight(tmp_path) -> None:
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(tmp_path),
            "load-before-remote",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        timeout=30,
        check=True,
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2 and sys.argv[2] == "load-before-remote":
        _run_load_before_remote(root)
    else:
        _run_remote_gui(root)
