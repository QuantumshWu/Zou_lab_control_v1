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
from zlc_neutral_atom.timing.board_config import DEFAULT_BOARD_CONFIG
from zlc_pulse import (
    FIELD_DURATION,
    FrozenScanTable,
    PulseExecutionService,
    PulseFieldRef,
    ScanParameter,
    load_deployed_pulse_target,
    new_pulse_document,
    pulse_target_manifest_from_xdc,
    save_pulse_document,
)
from zlc_pulse.server import serve_pulse_execution_service
from zlc_workbench.pulse_editor.app import open_pulse_editor
from gui_user_flow import click_tab as _click_tab, until as _until
from pulse_gui_user_flow import (
    choose_mode as _choose_mode,
    exercise_offline_dac_round_trip as _exercise_offline_dac_round_trip,
)


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


def _choose_remote(body) -> None:
    _choose_mode(body, "remote")


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


def _server_manifest():
    return pulse_target_manifest_from_xdc(
        load_deployed_pulse_target(),
        DEFAULT_BOARD_CONFIG,
    )


def _run_offline_dac_target_gui(workspace: Path, application) -> None:
    body = open_pulse_editor(repository=workspace / "offline-target")
    wrapper = body.window()
    try:
        _exercise_offline_signal_rename(body, application)
        _exercise_offline_dac_round_trip(body, application)
    finally:
        body.request_close(discard_unsaved=True)
        _until(application, lambda: body._controller.snapshot().close_complete)
        _until(application, lambda: not wrapper.isVisible())


def _exercise_offline_signal_rename(body, application) -> None:
    """Rename one visible channel through the Target tab's real controls."""

    _click_tab(body, body.target_view)
    row = body.target_view._rows[0]
    key = row.key
    endpoint = row.endpoints.text()
    previous_label = body.current_document.target.by_key[key].label
    previous_abi = body.current_document.target.abi_fingerprint
    renamed = f"{previous_label}_renamed"

    QtTest.QTest.mouseClick(row.signal, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(row.signal, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(row.signal, renamed)
    assert body.current_document.target.by_key[key].label == previous_label
    QtTest.QTest.mouseClick(body.target_view.apply_button, QtCore.Qt.LeftButton)
    _until(
        application,
        lambda: body.current_document.target.by_key[key].label == renamed,
    )

    rebuilt = next(item for item in body.target_view._rows if item.key == key)
    assert rebuilt.signal.text() == renamed
    assert rebuilt.endpoints.text() == endpoint
    assert body.current_document.target.abi_fingerprint == previous_abi

    _click_tab(body, body.schedule_view)
    _until(
        application,
        lambda: body.schedule_view.names_panel.port_labels[key].text() == renamed,
    )
    assert body.schedule_view.names_panel.hardware_labels[key].text() == endpoint
    combo = body.schedule_view.add_channel_combo
    for index in range(combo.count()):
        hidden_key = str(combo.itemData(index))
        hidden_label = body.current_document.target.by_key[hidden_key].label
        assert combo.itemText(index).startswith(hidden_label)
        if hidden_label != hidden_key:
            assert not combo.itemText(index).startswith(f"{hidden_key}  (")


def _run_remote_gui(workspace: Path) -> None:
    backend = _Backend()
    service = PulseExecutionService(
        _server_manifest(),
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
    _run_offline_dac_target_gui(workspace, application)
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
        assert body.schedule_view.names_panel.hardware_labels["ch00"].text() == "F15"
        _click_tab(body, body.target_view)
        assert not body.target_view.apply_button.isEnabled()
        assert body.target_view._rows[0].signal.isReadOnly()

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
    service = PulseExecutionService(_server_manifest(), clock_hz=50e6, backend=backend)
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


def _run_virtual_manifest_gui(workspace: Path) -> None:
    application = ensure_qt_app()
    body = open_pulse_editor(repository=workspace)
    wrapper = body.window()
    offline_visible = body.current_document.visible_ports
    offline_available = body._controller.snapshot().target_manifest.available_port_keys
    expected = (
        "ch00",
        "ch01",
        "ch03",
        "ch09",
        "ch11",
        "da_bias_y",
        "da_bias_x",
        "da_bias_z",
    )
    try:
        _choose_mode(body, "virtual")
        QtTest.QTest.mouseClick(
            body.schedule_view.conn_connect_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body._controller.snapshot().connection_state == "ready",
        )
        assert body._controller.snapshot().target_manifest.available_port_keys == expected
        assert body.current_document.visible_ports == offline_visible
        assert body.schedule_view._visible_ports == expected

        QtTest.QTest.mouseClick(
            body.schedule_view.hide_off_button,
            QtCore.Qt.LeftButton,
        )
        _until(application, lambda: len(body.schedule_view._visible_ports) == 4)
        geometry_before = (
            body.schedule_view.dataset_scroll.geometry().getRect(),
            body.schedule_view.button_frame.geometry().getRect(),
            body.schedule_view.dataset_body.width(),
        )
        QtTest.QTest.mouseClick(
            body.schedule_view.show_all_button,
            QtCore.Qt.LeftButton,
        )
        _until(application, lambda: body.schedule_view._visible_ports == expected)
        geometry_after = (
            body.schedule_view.dataset_scroll.geometry().getRect(),
            body.schedule_view.button_frame.geometry().getRect(),
            body.schedule_view.dataset_body.width(),
        )
        assert geometry_after == geometry_before
        assert body.current_document.visible_ports == offline_visible
        assert body.schedule_view._programmable_ports == expected
        _click_tab(body, body.target_view)
        assert len(body.target_view._rows) == len(expected)
        assert not body.target_view.add_dac_button.isEnabled()

        _click_tab(body, body.schedule_view)
        _choose_mode(body, "offline")
        QtTest.QTest.mouseClick(
            body.schedule_view.conn_connect_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: (
                body._controller.snapshot().connection_state == "offline"
                and body._controller.snapshot().connection_mode == "offline"
            ),
        )
        assert body.current_document.visible_ports == offline_visible
        assert body.schedule_view._visible_ports == offline_visible
        assert body.schedule_view._programmable_ports == offline_available

        # The formal deployed target crosses the old 16-channel density
        # threshold.  Drive Hide Off -> Show All through the real controls and
        # keep both the viewport and every surviving channel row stationary.
        first_key = body.schedule_view._visible_ports[0]
        QtTest.QTest.mouseClick(
            body.schedule_view.hide_off_button,
            QtCore.Qt.LeftButton,
        )
        _until(application, lambda: len(body.schedule_view._visible_ports) == 4)
        offline_geometry_before = (
            body.schedule_view.dataset_scroll.geometry().getRect(),
            body.schedule_view.button_frame.geometry().getRect(),
            body.schedule_view.names_panel.port_labels[first_key].height(),
        )
        QtTest.QTest.mouseClick(
            body.schedule_view.show_all_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.schedule_view._visible_ports == offline_visible,
        )
        offline_geometry_after = (
            body.schedule_view.dataset_scroll.geometry().getRect(),
            body.schedule_view.button_frame.geometry().getRect(),
            body.schedule_view.names_panel.port_labels[first_key].height(),
        )
        assert offline_geometry_after == offline_geometry_before

        _click_tab(body, body.target_view)
        assert body.target_view.add_dac_button.isEnabled()
    finally:
        body.request_close(discard_unsaved=True)
        _until(application, lambda: body._controller.snapshot().close_complete)
        _until(application, lambda: not wrapper.isVisible())


def test_virtual_gui_exposes_only_simulator_wired_digital_and_dac_ports(
    tmp_path,
) -> None:
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), "virtual"],
        cwd=Path(__file__).parents[1],
        env=environment,
        timeout=30,
        check=True,
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2 and sys.argv[2] == "load-before-remote":
        _run_load_before_remote(root)
    elif len(sys.argv) > 2 and sys.argv[2] == "virtual":
        _run_virtual_manifest_gui(root)
    else:
        _run_remote_gui(root)
