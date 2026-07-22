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


def _run_c47_input_projection_gui(workspace: Path) -> None:
    """Exercise the formal Qt input seams without a global presentation tick."""

    application = ensure_qt_app()
    body = open_pulse_editor(repository=workspace / "c47-input")
    wrapper = body.window()
    controller = body._controller
    normal_pump = controller.pump
    normal_poll = controller.poll_runtime_change
    normal_snapshot = controller.snapshot
    normal_apply = body._apply_snapshot
    normal_set_document = body.schedule_view.set_document
    had_update_scan_source = hasattr(controller, "update_scan_source")
    normal_update_scan_source = getattr(controller, "update_scan_source", None)
    try:
        # Drain launch-time Qt events before observing the idle timer.  Preview
        # is not requested while Edit is visible, and the 40 ms runtime watcher
        # is disarmed, so neither path can manufacture a whole-window cycle.
        _until(application, lambda: body.worker_idle)
        application.processEvents()
        idle_calls = {"poll": 0, "pump": 0, "apply": 0}

        def count_poll():
            idle_calls["poll"] += 1
            return normal_poll()

        def count_pump():
            idle_calls["pump"] += 1
            return normal_pump()

        def count_apply(snapshot):
            idle_calls["apply"] += 1
            return normal_apply(snapshot)

        controller.poll_runtime_change = count_poll
        controller.pump = count_pump
        body._apply_snapshot = count_apply
        assert body._timer.interval() == 40
        assert not body._timer.isActive()
        QtTest.QTest.qWait(body._timer.interval() * 3 + 20)
        application.processEvents()
        assert idle_calls == {"poll": 0, "pump": 0, "apply": 0}
        controller.poll_runtime_change = normal_poll
        controller.pump = normal_pump
        body._apply_snapshot = normal_apply

        # The Scan Program editor owns an uncommitted draft.  Real multi-key
        # input changes only that widget and its local dirty bit: it must not
        # publish the old controller-wide source/snapshot protocol or replace
        # the code editor while the Qt event loop remains live.
        _click_tab(body, body.scan_view)
        application.processEvents()
        before = controller.snapshot()
        code_editor = body.scan_view.scan_code
        code_identity = id(code_editor)
        source_revision = body.scan_view.source_revision
        marker = "  # c47-local-draft"
        input_calls = {"update": 0, "pump": 0, "snapshot": 0, "apply": 0}

        def reject_update_scan_source(*_args, **_kwargs):
            input_calls["update"] += 1

        def count_input_pump():
            input_calls["pump"] += 1
            return normal_pump()

        def count_input_snapshot():
            input_calls["snapshot"] += 1
            return normal_snapshot()

        def count_input_apply(snapshot):
            input_calls["apply"] += 1
            return normal_apply(snapshot)

        controller.update_scan_source = reject_update_scan_source
        controller.pump = count_input_pump
        controller.snapshot = count_input_snapshot
        body._apply_snapshot = count_input_apply
        QtTest.QTest.mouseClick(code_editor.viewport(), QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(
            code_editor,
            QtCore.Qt.Key_End,
            QtCore.Qt.ControlModifier,
        )
        QtTest.QTest.keyClicks(code_editor, marker)
        QtTest.QTest.qWait(body._timer.interval() * 2 + 20)
        application.processEvents()
        assert marker in code_editor.toPlainText()
        assert body.scan_view.code_dirty
        assert body.scan_view.source_revision == source_revision
        assert body.scan_view.scan_code is code_editor
        assert id(body.scan_view.scan_code) == code_identity
        assert input_calls == {"update": 0, "pump": 0, "snapshot": 0, "apply": 0}
        controller.pump = normal_pump
        controller.snapshot = normal_snapshot
        body._apply_snapshot = normal_apply
        del controller.update_scan_source
        after = controller.snapshot()
        assert after.editor_revision == before.editor_revision
        assert after.scan_workspace.source_text == before.scan_workspace.source_text
        assert code_editor.toPlainText() != after.scan_workspace.source_text

        # Duplicate commit signals are legal at a Qt focus/selection seam.
        # They must remain no-ops and must not re-project the Edit tree.
        _click_tab(body, body.schedule_view)
        application.processEvents()
        card = body.schedule_view.period_cards()[0]
        stable_widgets = (
            body.schedule_view.names_panel,
            body.schedule_view.channel_panel,
            *body.schedule_view.period_cards(),
            card.duration_edit,
            card.unit_combo,
            card.name_edit,
        )
        revision = controller.snapshot().editor_revision
        set_document_calls = []

        def count_set_document(*args, **kwargs):
            set_document_calls.append((args, kwargs))
            return normal_set_document(*args, **kwargs)

        body.schedule_view.set_document = count_set_document
        card.unit_combo.currentTextChanged.emit(card.unit_combo.currentText())
        card.name_edit.editingFinished.emit()
        application.processEvents()
        assert controller.snapshot().editor_revision == revision
        assert set_document_calls == []

        # A real semantic edit advances the displayed editor ledger.  When the
        # asynchronous Preview completion later publishes runtime state, it
        # must not replay that already-presented revision through a full Edit
        # tree reconcile.
        prior_unit = card.unit_combo.currentText()
        QtTest.QTest.mouseClick(card.unit_combo, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(card.unit_combo, QtCore.Qt.Key_Down)
        QtTest.QTest.keyClick(card.unit_combo, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: (
                controller.current_document.periods[0].unit != prior_unit
            ),
        )
        edited_document = controller.current_document
        _click_tab(body, body.preview_view)
        _until(
            application,
            lambda: (
                controller.snapshot().rendered_preview is not None
                and controller.snapshot().rendered_preview.editor_revision
                == controller.snapshot().editor_revision
            ),
        )
        application.processEvents()
        assert controller.current_document is edited_document
        assert set_document_calls == []
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
        body.schedule_view.set_document = normal_set_document
    finally:
        controller.poll_runtime_change = normal_poll
        controller.pump = normal_pump
        controller.snapshot = normal_snapshot
        body._apply_snapshot = normal_apply
        body.schedule_view.set_document = normal_set_document
        if had_update_scan_source:
            controller.update_scan_source = normal_update_scan_source
        elif "update_scan_source" in controller.__dict__:
            del controller.update_scan_source
        body.request_close(discard_unsaved=True)
        _until(application, lambda: body._controller.snapshot().close_complete)
        if wrapper.isVisible():
            wrapper.close()
        _until(application, lambda: not wrapper.isVisible())


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
            lambda: body._controller.snapshot().connection_state == "ready",
        )
        _until(application, lambda: len(connected_messages) >= 1)
        assert connected_messages == [
            "Connected to a virtual (in-memory) sequencer."
        ]
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
        offline_geometry_after = (
            body.schedule_view.dataset_scroll.geometry().getRect(),
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
        snapshot_calls = 0
        projection_calls = 0
        set_document_calls = 0
        original_snapshot = body._controller.snapshot
        original_projection = body._controller.editor_projection
        original_set_document = body.schedule_view.set_document

        def counted_snapshot():
            nonlocal snapshot_calls
            snapshot_calls += 1
            return original_snapshot()

        def counted_projection():
            nonlocal projection_calls
            projection_calls += 1
            return original_projection()

        def counted_set_document(*args, **kwargs):
            nonlocal set_document_calls
            set_document_calls += 1
            return original_set_document(*args, **kwargs)

        body._controller.snapshot = counted_snapshot
        body._controller.editor_projection = counted_projection
        body.schedule_view.set_document = counted_set_document
        prior_unit = first_card.unit_combo.currentText()
        try:
            QtTest.QTest.mouseClick(first_card.unit_combo, QtCore.Qt.LeftButton)
            QtTest.QTest.keyClick(first_card.unit_combo, QtCore.Qt.Key_Down)
            QtTest.QTest.keyClick(first_card.unit_combo, QtCore.Qt.Key_Return)
            _until(
                application,
                lambda: body.current_document.periods[0].unit != prior_unit,
            )

            # Structural authoring uses the same local owner turn: adding,
            # bracketing, and removing a period may insert/move/delete the
            # keyed cards, but must not fall back to a whole editor projection.
            original_cards = body.schedule_view.period_cards()
            QtTest.QTest.mouseClick(
                body.schedule_view.add_button,
                QtCore.Qt.LeftButton,
            )
            _until(
                application,
                lambda: len(body.schedule_view.period_cards())
                == len(original_cards) + 1,
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
            application.processEvents()
            assert (snapshot_calls, projection_calls, set_document_calls) == (0, 0, 0)
        finally:
            body._controller.snapshot = original_snapshot
            body._controller.editor_projection = original_projection
            body.schedule_view.set_document = original_set_document
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
    elif len(sys.argv) > 2 and sys.argv[2] == "c47-input":
        _run_c47_input_projection_gui(root)
    elif len(sys.argv) > 2 and sys.argv[2] == "virtual":
        _run_virtual_manifest_gui(root)
    else:
        _run_remote_gui(root)
