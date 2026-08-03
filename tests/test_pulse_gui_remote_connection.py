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
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from conftest import private_pulse_backend_snapshot, pulse_backend_completion_for
from fpga.pulse_streamer.host.image import StreamerParams
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
    pulse_target_manifest_from_xdc,
    save_pulse_document,
)
from zlc_pulse.server import serve_pulse_execution_service
from Zou_lab_control.api import WorkspacePaths
from Zou_lab_control.workbench import open_pulse_editor
from gui_user_flow import (
    click_tab as _click_tab,
    close_pulse_editor as _close_pulse_editor,
    until as _until,
)
from pulse_gui_user_flow import (
    choose_mode as _choose_mode,
    exercise_offline_dac_round_trip as _exercise_offline_dac_round_trip,
)


class _Backend:
    def __init__(self) -> None:
        self.prepared = None
        self.safe = True
        self.actions: list[str] = []
        self.state = "IDLE"
        self.scan_points = 0
        self.cursor = 0
        self.cursor_sample_count = 0

    def prepare(self, artifact) -> None:
        self.prepared = artifact
        self.safe = False
        self.actions.append("prepare")
        self.state = "PREPARED"
        self.scan_points = len(artifact.target_ir.scan_points)
        self.cursor = 0
        self.cursor_sample_count = 0

    def fire(self, artifact) -> None:
        assert artifact is self.prepared
        self.actions.append("fire")
        self.state = "RUNNING"
        self.cursor_sample_count = 1 if self.scan_points else 0

    def await_completion(self, artifact, _timeout):
        assert artifact is self.prepared
        self.actions.append("complete")
        self.state = "DONE"
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
        self.state = "SAFE"
        self.cursor_sample_count = 0

    def request_interrupt(self) -> None:
        return None

    def snapshot(self):
        return private_pulse_backend_snapshot(
            state=self.state,
            raw_lane_count=len(_server_manifest().target.raw_lanes),
            artifact=self.prepared,
            scan_point_count=self.scan_points,
            cursor=self.cursor,
            cursor_sample_count=self.cursor_sample_count,
        )


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
        Path(__file__).resolve().parents[1] / "fpga" / "board_config" / "board.xdc",
    )


def _workspace_paths(root: Path) -> WorkspacePaths:
    return WorkspacePaths.for_workspace(root.resolve())


def _run_offline_dac_target_gui(workspace: Path, application) -> None:
    body = open_pulse_editor(
        workspace=_workspace_paths(workspace / "offline-target")
    )
    try:
        _exercise_offline_signal_rename(body, application)
        _exercise_offline_dac_round_trip(body, application)
    finally:
        _close_pulse_editor(application, body)


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

    reconciled = next(item for item in body.target_view._rows if item.key == key)
    assert reconciled is row
    assert reconciled.signal.text() == renamed
    assert reconciled.endpoints.text() == endpoint
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
        params=StreamerParams(),
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
    body = open_pulse_editor(
        workspace=_workspace_paths(workspace),
        document=_scan_document(),
    )
    try:
        _choose_remote(body)
        _enter_address(body, f"127.0.0.1:{unavailable_port}")
        QtTest.QTest.mouseClick(
            body.schedule_view.conn_connect_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body._controller.runtime_update().connection_state == "offline"
            and "Connection failed" in body._controller.runtime_update().diagnostic,
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
            lambda: body._controller.runtime_update().connection_state == "ready"
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
            and not body._controller.runtime_update().run_busy,
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
        remote_run_id = body.active_snapshot.run_id
        _click_tab(body, body.scan_view)
        _until(
            application,
            lambda: body.scan_view.scan_progress_label.text().startswith(
                "Scan: point "
            ),
        )
        QtTest.QTest.mouseClick(
            body.scan_view.scan_hold_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body._controller.runtime_update().held_scan_point is not None,
        )
        assert body.active_snapshot is not None
        assert body.active_snapshot.run_id == remote_run_id
        held = body._controller.runtime_update().held_scan_point
        assert held is not None
        held_index, total, _values = held
        step = (
            body.scan_view.scan_step_forward_button
            if held_index < total - 1
            else body.scan_view.scan_step_back_button
        )
        QtTest.QTest.mouseClick(step, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: body._controller.runtime_update().held_scan_point is not None
            and body._controller.runtime_update().held_scan_point[0] != held_index,
        )
        assert body.active_snapshot is not None
        assert body.active_snapshot.run_id == remote_run_id
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
        _close_pulse_editor(application, body)
        server.close()
        if server_thread is not None:
            server_thread.join(timeout=3.0)

    assert body.worker_idle
    assert server_thread is not None and not server_thread.is_alive()
    assert service.snapshot().state == "SAFE"
    assert service.snapshot().safe_readback_confirmed
    assert backend.actions.count("prepare") == 4
    assert backend.actions.count("fire") == 4
    assert backend.actions.count("complete") == 1
    assert backend.actions[-1] == "safe"


def _run_load_before_remote(workspace: Path) -> None:
    backend = _Backend()
    deployed = load_deployed_pulse_target()
    service = PulseExecutionService(
        _server_manifest(),
        clock_hz=50e6,
        backend=backend,
        params=StreamerParams(),
    )
    server = serve_pulse_execution_service(
        service,
        host="127.0.0.1",
        port=0,
        start=False,
    )
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    workspace_paths = _workspace_paths(workspace)
    workspace_paths.pulses_root.mkdir(parents=True, exist_ok=True)
    document_path = workspace_paths.pulses_root / "incompatible-document.json"
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
        workspace=workspace_paths,
        path=document_path.name,
        remote_endpoint=f"127.0.0.1:{server.port}",
    )
    try:
        _until(
            application,
            lambda: body.current_document.name
            == "Must load before remote preflight"
            and body._controller.runtime_update().connection_state == "offline"
            and "clock grid differs" in body._controller.runtime_update().diagnostic,
        )
        assert body.schedule_view.conn_connect_button.isEnabled()
    finally:
        _close_pulse_editor(application, body)
        server.close()
        server_thread.join(timeout=3.0)

    assert not server_thread.is_alive()
    assert service.snapshot().state == "SAFE"


def _run_c47_input_projection_gui(workspace: Path) -> None:
    """Exercise the operator-visible local-draft and stable-widget behavior."""

    application = ensure_qt_app()
    body = open_pulse_editor(
        workspace=_workspace_paths(workspace / "c47-input")
    )
    controller = body._controller
    try:
        _until(application, lambda: body.worker_idle)

        _click_tab(body, body.scan_view)
        code_editor = body.scan_view.scan_code
        source_revision = body.scan_view.source_revision
        editor_revision = controller.current_editor_revision
        committed_source = controller.current_scan_workspace.source_text
        marker = "  # local draft"
        QtTest.QTest.mouseClick(code_editor.viewport(), QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(
            code_editor,
            QtCore.Qt.Key_End,
            QtCore.Qt.ControlModifier,
        )
        QtTest.QTest.keyClicks(code_editor, marker)
        application.processEvents()
        assert marker in code_editor.toPlainText()
        assert body.scan_view.code_dirty
        assert body.scan_view.source_revision == source_revision
        assert body.scan_view.scan_code is code_editor
        assert controller.current_editor_revision == editor_revision
        assert controller.current_scan_workspace.source_text == committed_source

        _click_tab(body, body.schedule_view)
        card = body.schedule_view.period_cards()[0]
        stable_widgets = (
            body.schedule_view.names_panel,
            body.schedule_view.channel_panel,
            *body.schedule_view.period_cards(),
            card.duration_edit,
            card.unit_combo,
            card.name_edit,
        )
        card.unit_combo.currentTextChanged.emit(card.unit_combo.currentText())
        card.name_edit.editingFinished.emit()
        application.processEvents()
        assert controller.current_editor_revision == editor_revision

        prior_unit = card.unit_combo.currentText()
        QtTest.QTest.mouseClick(card.unit_combo, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(card.unit_combo, QtCore.Qt.Key_Down)
        QtTest.QTest.keyClick(card.unit_combo, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: controller.current_document.periods[0].unit != prior_unit,
        )
        edited_document = controller.current_document
        _click_tab(body, body.preview_view)
        _until(
            application,
            lambda: (
                controller.preview_update().plot is not None
                and controller.preview_update().plot.editor_revision
                == controller.current_editor_revision
            ),
        )
        assert controller.current_document is edited_document
        _click_tab(body, body.schedule_view)
        assert all(
            before_widget is after_widget
            for before_widget, after_widget in zip(
                stable_widgets,
                (
                    body.schedule_view.names_panel,
                    body.schedule_view.channel_panel,
                    *body.schedule_view.period_cards(),
                    body.schedule_view.period_cards()[0].duration_edit,
                    body.schedule_view.period_cards()[0].unit_combo,
                    body.schedule_view.period_cards()[0].name_edit,
                ),
                strict=True,
            )
        )
    finally:
        _close_pulse_editor(application, body)


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


def test_c47_idle_and_scan_typing_do_not_reproject_the_global_editor(tmp_path) -> None:
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(tmp_path),
            "c47-input",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        timeout=30,
        check=True,
    )


def _run_virtual_manifest_gui(workspace: Path) -> None:
    application = ensure_qt_app()
    body = open_pulse_editor(workspace=_workspace_paths(workspace))
    offline_visible = body.current_document.visible_ports
    offline_available = body._controller.current_target_manifest.available_port_keys
    expected = (
        "ch00",
        "ch01",
        "ch03",
        "ch06",
        "ch09",
        "ch11",
        "da_bias_y",
        "da_bias_x",
        "da_bias_z",
    )
    try:
        connected_messages: list[str] = []
        normal_message = body._message

        def message_with_nested_owner_turn(text: str) -> None:
            message = str(text)
            if message.startswith("Connected to"):
                connected_messages.append(message)
                if len(connected_messages) == 1:
                    # A desktop modal dialog runs a nested Qt event loop.  This
                    # models its timer wake without replacing the formal GUI.
                    body._owner_cycle()
            normal_message(message)

        body._message = message_with_nested_owner_turn
        _choose_mode(body, "virtual")
        QtTest.QTest.mouseClick(
            body.schedule_view.conn_connect_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body._controller.runtime_update().connection_state == "ready",
        )
        _until(application, lambda: len(connected_messages) >= 1)
        assert connected_messages == [
            "Connected to a virtual (in-memory) sequencer."
        ]
        assert body._controller.current_target_manifest.available_port_keys == expected
        assert body.current_document.visible_ports == offline_visible
        assert body.schedule_view._visible_ports == expected

        QtTest.QTest.mouseClick(
            body.schedule_view.hide_off_button,
            QtCore.Qt.LeftButton,
        )
        _until(application, lambda: len(body.schedule_view._visible_ports) == 4)
        geometry_before = (
            body.schedule_view.timeline_scroll.geometry().getRect(),
            body.schedule_view.button_frame.geometry().getRect(),
            body.schedule_view.timeline_scroll.viewport().width(),
        )
        QtTest.QTest.mouseClick(
            body.schedule_view.show_all_button,
            QtCore.Qt.LeftButton,
        )
        _until(application, lambda: body.schedule_view._visible_ports == expected)
        geometry_after = (
            body.schedule_view.timeline_scroll.geometry().getRect(),
            body.schedule_view.button_frame.geometry().getRect(),
            body.schedule_view.timeline_scroll.viewport().width(),
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
                body._controller.runtime_update().connection_state == "offline"
                and body._controller.runtime_update().connection_mode == "offline"
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
        hidden_ports = body.schedule_view._visible_ports

        # Load a different document generation with the SAME hidden projection.
        # New period widgets must be born with that complete projection; there
        # is intentionally no second visibility change for a presenter to use
        # as a repair event.
        loaded_periods = tuple(
            replace(period, period_id=f"loaded_{index}")
            for index, period in enumerate(body.current_document.periods, start=1)
        )
        loaded_document = replace(
            body.current_document,
            name="loaded hidden projection",
            periods=loaded_periods,
            visible_ports=hidden_ports,
            repeat=None,
            scan_parameters=(),
            api_parameters=(),
            scan_table=None,
            scan_recipe=None,
        )
        loaded_path = workspace / "loaded_hidden_projection.json"
        save_pulse_document(loaded_document, loaded_path)
        with patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(loaded_path), "ZLC pulse (*.json)"),
        ):
            QtTest.QTest.mouseClick(
                body.schedule_view.load_button,
                QtCore.Qt.LeftButton,
            )
        _until(
            application,
            lambda: body.current_document.name == "loaded hidden projection",
        )
        assert body.schedule_view._visible_ports == hidden_ports
        assert tuple(
            card.period_id for card in body.schedule_view.period_cards()
        ) == tuple(period.period_id for period in loaded_periods)
        for card in body.schedule_view.period_cards():
            assert all(
                (not row.isHidden()) == (key in hidden_ports)
                for key, row in card.port_rows.items()
            )
        QtTest.QTest.mouseClick(
            body.schedule_view.hide_off_button,
            QtCore.Qt.LeftButton,
        )
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert all(
            (not row.isHidden()) == (key in hidden_ports)
            for card in body.schedule_view.period_cards()
            for key, row in card.port_rows.items()
        )

        hidden_cards = body.schedule_view.period_cards()
        QtTest.QTest.mouseClick(
            body.schedule_view.add_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: len(body.schedule_view.period_cards()) == len(hidden_cards) + 1,
        )
        assert body.schedule_view.period_cards()[:-1] == hidden_cards
        added_hidden_card = body.schedule_view.period_cards()[-1]
        assert all(
            (not row.isHidden()) == (key in hidden_ports)
            for key, row in added_hidden_card.port_rows.items()
        )
        QtTest.QTest.mouseClick(
            body.schedule_view.remove_button,
            QtCore.Qt.LeftButton,
        )
        _until(application, lambda: body.schedule_view.period_cards() == hidden_cards)

        offline_geometry_before = (
            body.schedule_view.timeline_scroll.geometry().getRect(),
            body.schedule_view.button_frame.geometry().getRect(),
            body.schedule_view.names_panel.port_labels[first_key].height(),
        )
        edit_tree_before = (
            body.schedule_view.names_panel,
            body.schedule_view.channel_panel,
            *body.schedule_view.period_cards(),
            body.schedule_view.names_panel.row_widgets[first_key],
        )
        QtTest.QTest.mouseClick(
            body.schedule_view.show_all_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.schedule_view._visible_ports == offline_available,
        )
        _until(
            application,
            lambda: (
                body.schedule_view.timeline_scroll.verticalScrollBar().maximum()
                > 0
            ),
        )
        offline_geometry_after = (
            body.schedule_view.timeline_scroll.geometry().getRect(),
            body.schedule_view.button_frame.geometry().getRect(),
            body.schedule_view.names_panel.port_labels[first_key].height(),
        )
        assert offline_geometry_after == offline_geometry_before
        edit_tree_after = (
            body.schedule_view.names_panel,
            body.schedule_view.channel_panel,
            *body.schedule_view.period_cards(),
            body.schedule_view.names_panel.row_widgets[first_key],
        )
        assert all(
            before is after
            for before, after in zip(edit_tree_before, edit_tree_after, strict=True)
        )
        # Show all changes only visibility, but visibility is part of the one
        # dataset-column geometry transaction.  Every existing row must occupy
        # its own vertical slot and the same logical key must remain aligned
        # across Catalog, Delay and every Period column.
        columns = (
            body.schedule_view.names_panel.row_widgets,
            body.schedule_view.channel_panel.row_widgets,
            *(
                card.port_rows
                for card in body.schedule_view.period_cards()
            ),
        )
        for column in columns:
            widgets = [column[key] for key in offline_available]
            for previous, current in zip(widgets, widgets[1:]):
                previous_rect = QtCore.QRect(
                    previous.mapToGlobal(QtCore.QPoint(0, 0)),
                    previous.size(),
                )
                current_rect = QtCore.QRect(
                    current.mapToGlobal(QtCore.QPoint(0, 0)),
                    current.size(),
                )
                assert current_rect.top() > previous_rect.bottom()
        for key in offline_available:
            centers = tuple(
                column[key].mapToGlobal(column[key].rect().center()).y()
                for column in columns
            )
            assert max(centers) - min(centers) <= 1

        # A scalar edit is projected onto the existing controls.  The real
        # combo interaction used to trigger a full Edit-tree teardown, which
        # made a unit change visibly stall and discarded focus/widget state.
        first_card = body.schedule_view.period_cards()[0]
        stable_widgets = (
            body.schedule_view.names_panel,
            body.schedule_view.channel_panel,
            *body.schedule_view.period_cards(),
            first_card.unit_combo,
            first_card.duration_edit,
        )
        prior_unit = first_card.unit_combo.currentText()
        QtTest.QTest.mouseClick(first_card.unit_combo, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(first_card.unit_combo, QtCore.Qt.Key_Down)
        QtTest.QTest.keyClick(first_card.unit_combo, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: body.current_document.periods[0].unit != prior_unit,
        )

        original_cards = body.schedule_view.period_cards()
        QtTest.QTest.mouseClick(
            body.schedule_view.add_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: len(body.schedule_view.period_cards()) == len(original_cards) + 1,
        )
        assert body.schedule_view.period_cards()[0] is original_cards[0]
        QtTest.QTest.mouseClick(
            body.schedule_view.bracket_button,
            QtCore.Qt.LeftButton,
        )
        _until(application, lambda: body.current_document.repeat is not None)
        QtTest.QTest.mouseClick(
            body.schedule_view.remove_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: len(body.schedule_view.period_cards()) == len(original_cards),
        )
        assert body.schedule_view.period_cards() == original_cards

        digital = next(iter(first_card.checks))
        if not first_card.checks[digital].isChecked():
            QtTest.QTest.mouseClick(
                first_card.checks[digital],
                QtCore.Qt.LeftButton,
            )
            _until(application, lambda: first_card.checks[digital].isChecked())
        QtTest.QTest.mouseClick(
            body.schedule_view.channel_panel.clear_buttons[digital],
            QtCore.Qt.LeftButton,
        )
        _until(application, lambda: not first_card.checks[digital].isChecked())
        assert all(
            before is after
            for before, after in zip(
                stable_widgets,
                (
                    body.schedule_view.names_panel,
                    body.schedule_view.channel_panel,
                    *body.schedule_view.period_cards(),
                    body.schedule_view.period_cards()[0].unit_combo,
                    body.schedule_view.period_cards()[0].duration_edit,
                ),
                strict=True,
            )
        )

        # HOLD is the preceding effective signed value, and editability plus
        # scan/API decoration is one state.  Exercise the full cycle with real
        # combo, line-edit, and dot input before returning to HOLD.
        first_card, second_card = body.schedule_view.period_cards()[:2]
        dac_port = next(iter(first_card.bus_mode_combos))
        edge_combo = first_card.bus_mode_combos[dac_port]
        edge_value = first_card.bus_value_edits[dac_port]
        assert edge_combo.currentText() == "Hold"
        assert edge_value.text() == "0" and edge_value.isReadOnly()
        QtTest.QTest.mouseClick(edge_combo, QtCore.Qt.LeftButton)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        QtTest.QTest.keyClick(edge_combo.view(), QtCore.Qt.Key_Home)
        QtTest.QTest.keyClick(edge_combo.view(), QtCore.Qt.Key_Return)
        _until(application, lambda: not edge_value.isReadOnly())
        QtTest.QTest.mouseClick(edge_value, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(edge_value, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        QtTest.QTest.keyClicks(edge_value, "37")
        QtTest.QTest.keyClick(edge_value, QtCore.Qt.Key_Return)

        held_combo = second_card.bus_mode_combos[dac_port]
        held_value = second_card.bus_value_edits[dac_port]
        _until(
            application,
            lambda: held_value.text() == "37" and held_value.isReadOnly(),
        )
        assert held_combo.currentText() == "Hold"

        def held_binding_kind() -> str | None:
            field = (second_card.period_id, dac_port)
            if any(
                (item.field.period_id, item.field.port) == field
                for item in body.current_document.scan_parameters
            ):
                return "scan"
            if any(
                (item.field.period_id, item.field.port) == field
                for item in body.current_document.api_parameters
            ):
                return "api"
            return None

        for _expected_kind in ("scan", "api", None):
            QtTest.QTest.mouseClick(held_value.dot, QtCore.Qt.LeftButton)
            _until(
                application,
                lambda expected=_expected_kind: held_binding_kind() == expected,
            )
        QtTest.QTest.mouseClick(held_combo, QtCore.Qt.LeftButton)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        QtTest.QTest.keyClick(held_combo.view(), QtCore.Qt.Key_End)
        QtTest.QTest.keyClick(held_combo.view(), QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: (
                held_combo.currentText() == "Hold"
                and held_value.text() == "37"
                and held_value.isReadOnly()
            ),
        )

        _click_tab(body, body.target_view)
        assert body.target_view.add_dac_button.isEnabled()
    finally:
        _close_pulse_editor(application, body)


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
    elif len(sys.argv) > 2 and sys.argv[2] == "c47-input":
        _run_c47_input_projection_gui(root)
    elif len(sys.argv) > 2 and sys.argv[2] == "virtual":
        _run_virtual_manifest_gui(root)
    else:
        _run_remote_gui(root)
