"""M2e first typed ROI from the visible raw camera monitor oracles."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import ReductionMethod, Selection
from zlc_frontend.qt_widgets import QtRasterBoard
from zlc_neutral_atom.processing.roi_monitor import RoiScalarStreamProjection
from zlc_neutral_atom.runtime.control import ControlAckStatus
from zlc_neutral_atom.runtime.run import RunState
from zlc_workbench.live import LiveBoardController


_RAW_PANELS = ("camera-monitor-image",)
_SCALAR_PANELS = (
    "camera-monitor-image",
    "camera-monitor-roi-curve",
    "camera-monitor-roi-histogram",
    "camera-monitor-roi-meter",
)


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def experiment_context(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("m2e-camera-monitor")
    experiment = zlc.connect("virtual", repository=workspace)
    yield experiment, workspace
    experiment.close()


def _until(application, predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _board(window) -> QtRasterBoard:
    board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
    assert isinstance(board, QtRasterBoard)
    return board


def _close_window(application, window) -> None:
    window.close()
    _until(application, lambda: not window.isVisible(), timeout=10.0)
    window.deleteLater()
    application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    application.processEvents()


def _manifest_files(workspace: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*")
            if path.is_file() and "manifests" in path.parts
        )
    )


def _start_raw(application, window):
    start = window.findChild(QtWidgets.QPushButton, "startButton")
    _until(application, start.isEnabled)
    board = _board(window)
    assert window._board_panel_ids == _RAW_PANELS
    QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
    _until(
        application,
        lambda: (
            board.has_front
            and window._live is not None
            and window._live.front_status is not None
            and window._live.front_status.sequence == board.front_frame.sequence
            and window._selector_switch.isEnabled()
        ),
    )
    return board


def _draw_first_roi(window, board: QtRasterBoard):
    window._selector_switch.setChecked(True)
    target = board._selector_target()[0]
    start = QtCore.QPoint(
        target.left() + target.width() // 4,
        target.top() + target.height() // 4,
    )
    end = QtCore.QPoint(
        target.left() + 3 * target.width() // 4,
        target.top() + 3 * target.height() // 4,
    )
    QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start)
    QtTest.QTest.mouseMove(board, end)
    QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end)
    assert window.findChild(QtWidgets.QPushButton, "applyRoiButton") is None


def _initial_roi(experiment) -> Selection:
    schema = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=2)
    ).output_schema
    y_axis, x_axis = schema.cell_schema.data_axes
    return Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        x_axis.coordinates[0],
        x_axis.coordinates[min(7, x_axis.size - 1)],
        y_axis.coordinates[0],
        y_axis.coordinates[min(7, y_axis.size - 1)],
        coordinate_frame=x_axis.coordinate_frame,
    )


def test_rejected_initial_roi_falls_back_to_raw_and_can_recover_in_place(
    experiment_context,
    application,
    monkeypatch,
):
    experiment, workspace = experiment_context
    before_manifests = _manifest_files(workspace)
    initial_roi = _initial_roi(experiment)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        roi=initial_roi,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    reject_initial = True
    real_project = RoiScalarStreamProjection.project

    def reject_first_branch(self, update, binding, control_revision):
        if reject_initial:
            raise ValueError("synthetic initial ROI rejection")
        return real_project(self, update, binding, control_revision)

    monkeypatch.setattr(
        RoiScalarStreamProjection,
        "project",
        reject_first_branch,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        board = _board(window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.has_front
                and board.front_frame is not None
                and tuple(panel.panel_id for panel in board.front_frame.panels)
                == _RAW_PANELS
                and "REJECTED" in window._roi_status.text()
                and window._slot is not None
                and window._slot.current_camera_roi_state().binding is None
            ),
        )
        handle = window._handle
        slot = window._slot
        source_generation = window._run_owner.generation
        assert handle is not None and slot is not None
        assert not handle.snapshot().state.terminal
        assert window._applied_request.roi is None
        assert window._draft_selection == initial_roi
        _run, _causation, raw_snapshot = slot.freeze_camera_current()
        assert raw_snapshot.scalar is None

        receipts = []
        real_submit = slot.submit_camera_roi_control

        def observe_submit(candidate):
            receipt = real_submit(candidate)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(slot, "submit_camera_roi_control", observe_submit)
        reject_initial = False
        window._submit_roi_control(initial_roi)
        assert len(receipts) == 1
        _until(
            application,
            lambda: (
                receipts[0].snapshot().status is ControlAckStatus.APPLIED
                and window._handle is handle
                and window._board_panel_ids == _SCALAR_PANELS
                and board.front_frame is not None
                and tuple(panel.panel_id for panel in board.front_frame.panels)
                == _SCALAR_PANELS
            ),
        )
        _run, _causation, recovered = slot.freeze_camera_current()
        assert recovered.scalar is not None
        assert recovered.raw.ref.block_id == raw_snapshot.raw.ref.block_id
        assert recovered.raw.ref.stream_generation == raw_snapshot.raw.ref.stream_generation
        assert recovered.raw.head.sequence > raw_snapshot.raw.head.sequence
        assert window._run_owner.generation == source_generation
        assert _manifest_files(workspace) == before_manifests
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_rejected_initial_roi_and_raw_presentation_failure_keep_both_causes(
    experiment_context,
    application,
    monkeypatch,
):
    experiment, _workspace = experiment_context
    initial_roi = _initial_roi(experiment)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        roi=initial_roi,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )

    def reject_initial_branch(self, update, binding, control_revision):
        raise ValueError("synthetic initial ROI rejection with raw fallback failure")

    monkeypatch.setattr(
        RoiScalarStreamProjection,
        "project",
        reject_initial_branch,
    )
    real_reconfigure = LiveBoardController.reconfigure_scalar

    def fail_raw_fallback(
        self,
        state,
        *,
        scalar_dataset_id,
        scalar_documents,
        curve_display,
        histogram_display,
    ):
        if state.binding is None:
            raise RuntimeError("synthetic initial raw presentation failure")
        return real_reconfigure(
            self,
            state,
            scalar_dataset_id=scalar_dataset_id,
            scalar_documents=scalar_documents,
            curve_display=curve_display,
            histogram_display=histogram_display,
        )

    monkeypatch.setattr(
        LiveBoardController,
        "reconfigure_scalar",
        fail_raw_fallback,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: window._roi_presentation_failure is not None)

        diagnostic = window._roi_presentation_failure or ""
        assert "synthetic initial ROI rejection with raw fallback failure" in diagnostic
        assert "synthetic initial raw presentation failure" in diagnostic
        assert "synthetic initial ROI rejection with raw fallback failure" in (
            window._roi_status.text()
        )
        assert window._handle is not None
        assert not window._handle.snapshot().state.terminal
        assert window._live is not None and window._live._presentation_frozen
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_first_roi_clear_and_recreate_only_migrate_the_scalar_branch(
    experiment_context,
    application,
    monkeypatch,
):
    experiment, workspace = experiment_context
    before_manifests = _manifest_files(workspace)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    assert request.roi is None
    window = experiment.readout.camera_monitor_gui(request)
    try:
        raw_board = _start_raw(application, window)
        raw_frame = raw_board.front_frame
        old_handle = window._handle
        old_slot = window._slot
        old_generation = window._run_owner.generation
        assert raw_frame is not None and old_handle is not None and old_slot is not None
        assert tuple(panel.panel_id for panel in raw_frame.panels) == _RAW_PANELS
        assert window._running_binding is None
        old_run_id, _causation, old_snapshot = old_slot.freeze_camera_current()
        assert old_snapshot.scalar is None
        old_ref = old_snapshot.raw.ref
        old_dataset_id = old_slot.dataset_id

        receipts = []
        candidates = []
        real_submit = old_slot.submit_camera_roi_control

        def observe_submit(candidate):
            receipt = real_submit(candidate)
            candidates.append(candidate)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(old_slot, "submit_camera_roi_control", observe_submit)
        clear_calls = []
        real_clear = raw_board.clear

        def observe_clear(*args, **kwargs):
            clear_calls.append((args, kwargs))
            return real_clear(*args, **kwargs)

        monkeypatch.setattr(raw_board, "clear", observe_clear)

        _draw_first_roi(window, raw_board)
        assert len(receipts) == 1 and candidates[0] is not None

        def scalar_visible():
            board = _board(window)
            return (
                receipts[0].snapshot().status is ControlAckStatus.APPLIED
                and window._applied_request.roi == candidates[0].selection
                and window._handle is old_handle
                and window._board_panel_ids == _SCALAR_PANELS
                and board.has_front
                and board.front_frame is not None
                and tuple(panel.panel_id for panel in board.front_frame.panels)
                == _SCALAR_PANELS
                and window._live is not None
                and window._live.front_status is not None
                and window._live.front_status.sequence == board.front_frame.sequence
                and window._live.front_status.scalar_control_revision
                == receipts[0].snapshot().revision
            )

        _until(application, scalar_visible)
        scalar_board = _board(window)
        assert window._board_widget is raw_board is scalar_board
        assert clear_calls == []
        assert not old_handle.snapshot().state.terminal
        assert window._run_owner.generation == old_generation
        assert window._handle.run_id.value == old_run_id
        assert window._slot is old_slot
        assert window._slot.dataset_id == old_dataset_id
        _run, _causation, first_scalar = old_slot.freeze_camera_current()
        assert first_scalar.scalar is not None
        assert first_scalar.raw.ref.block_id == old_ref.block_id
        assert first_scalar.raw.ref.stream_generation == old_ref.stream_generation
        assert window._running_binding.selection == candidates[0].selection
        first_scalar_generation = first_scalar.scalar.ref.stream_generation
        first_scalar_block = first_scalar.scalar.ref.block_id
        assert scalar_board._selector_applied_bounds == (
            scalar_board._selector_viewport.normalized_bounds_for_selection(
                candidates[0].selection
            )
        )
        assert scalar_board._selector_draft_bounds is None

        clear = window.findChild(QtWidgets.QPushButton, "clearRoiButton")
        assert clear is not None and clear.isEnabled()
        QtTest.QTest.mouseClick(clear, QtCore.Qt.LeftButton)
        assert len(receipts) == 2 and candidates[1] is None
        _until(
            application,
            lambda: (
                receipts[1].snapshot().status is ControlAckStatus.APPLIED
                and window._running_binding is None
                and window._board_panel_ids == _RAW_PANELS
                and raw_board.front_frame is not None
                and tuple(panel.panel_id for panel in raw_board.front_frame.panels)
                == _RAW_PANELS
            ),
        )
        assert window._handle is old_handle and window._slot is old_slot
        _run, _causation, raw_again = old_slot.freeze_camera_current()
        assert raw_again.scalar is None
        assert raw_again.raw.ref.block_id == old_ref.block_id
        assert raw_again.raw.head.sequence > first_scalar.raw.head.sequence
        assert clear_calls == []

        _draw_first_roi(window, raw_board)
        assert len(receipts) == 3 and candidates[2] is not None
        _until(
            application,
            lambda: (
                receipts[2].snapshot().status is ControlAckStatus.APPLIED
                and window._board_panel_ids == _SCALAR_PANELS
                and raw_board.front_frame is not None
                and tuple(panel.panel_id for panel in raw_board.front_frame.panels)
                == _SCALAR_PANELS
                and window._live.front_status is not None
                and window._live.front_status.scalar_control_revision
                == receipts[2].snapshot().revision
            ),
        )
        _run, _causation, recreated = old_slot.freeze_camera_current()
        assert recreated.scalar is not None
        assert recreated.raw.ref.block_id == old_ref.block_id
        assert recreated.scalar.ref.stream_generation != first_scalar_generation
        assert recreated.scalar.ref.block_id != first_scalar_block
        assert [receipt.snapshot().revision for receipt in receipts] == sorted(
            receipt.snapshot().revision for receipt in receipts
        )
        assert clear_calls == []
        assert _manifest_files(workspace) == before_manifests
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_first_roi_presentation_failure_rolls_back_staged_layout_once(
    experiment_context,
    application,
    monkeypatch,
):
    experiment, workspace = experiment_context
    before_manifests = _manifest_files(workspace)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        board_widget = _start_raw(application, window)
        live = window._live
        board = window._board
        slot = window._slot
        handle = window._handle
        assert live is not None and board is not None
        assert slot is not None and handle is not None
        old_model = board.model
        old_configuration = live._configuration
        old_generation = window._run_owner.generation
        _run, _causation, old_snapshot = slot.freeze_camera_current()
        assert old_snapshot.scalar is None and old_snapshot.raw.head is not None

        receipts = []
        real_submit = slot.submit_camera_roi_control

        def observe_submit(candidate):
            receipt = real_submit(candidate)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(slot, "submit_camera_roi_control", observe_submit)
        staged_frames = []
        staged_generations = []
        discarded_models = []
        discarded_layouts = []
        freeze_calls = []
        board_reconfigures = []
        live_reconfigures = []
        real_stage_layout = board_widget.stage_layout
        real_discard_layout = board_widget.discard_staged_layout
        real_board_reconfigure = board.reconfigure
        real_discard_model = board.discard_staged_model
        real_freeze = live.freeze_presentation

        def observe_stage_layout(panel_ids, **kwargs):
            staged_frames.append(board_widget.front_frame)
            staged_generations.append(kwargs["layout_generation"])
            return real_stage_layout(panel_ids, **kwargs)

        def observe_discard_layout(**kwargs):
            discarded_layouts.append(kwargs["layout_generation"])
            return real_discard_layout(**kwargs)

        def observe_board_reconfigure(model):
            board_reconfigures.append(model)
            return real_board_reconfigure(model)

        def observe_discard_model(model):
            discarded_models.append(model)
            return real_discard_model(model)

        def observe_freeze():
            freeze_calls.append(None)
            return real_freeze()

        def reject_live_reconfigure(
            state,
            *,
            scalar_dataset_id,
            scalar_documents,
            curve_display,
            histogram_display,
        ):
            live_reconfigures.append(
                (
                    state,
                    scalar_dataset_id,
                    tuple(scalar_documents),
                    curve_display,
                    histogram_display,
                )
            )
            raise RuntimeError("synthetic live scalar reconfigure failure")

        monkeypatch.setattr(board_widget, "stage_layout", observe_stage_layout)
        monkeypatch.setattr(
            board_widget,
            "discard_staged_layout",
            observe_discard_layout,
        )
        monkeypatch.setattr(board, "reconfigure", observe_board_reconfigure)
        monkeypatch.setattr(board, "discard_staged_model", observe_discard_model)
        monkeypatch.setattr(live, "freeze_presentation", observe_freeze)
        monkeypatch.setattr(live, "reconfigure_scalar", reject_live_reconfigure)

        _draw_first_roi(window, board_widget)
        assert len(receipts) == 1
        _until(
            application,
            lambda: (
                receipts[0].snapshot().status is ControlAckStatus.APPLIED
                and window._roi_presentation_failure is not None
                and not window._roi_controls
            ),
        )
        assert "synthetic live scalar reconfigure failure" in (
            window._roi_presentation_failure or ""
        )
        assert len(board_reconfigures) == len(live_reconfigures) == 1
        staged_model = board_reconfigures[0]
        assert discarded_models == [staged_model]
        assert staged_generations == [old_model.layout_generation + 1]
        assert discarded_layouts == staged_generations
        assert freeze_calls == [None]
        assert board.model is old_model
        assert board._staged_model is None
        assert board_widget._staged_layout is None
        assert live._configuration is old_configuration
        assert live._presentation_frozen and live._port is None and live._sources is None
        assert window._board_panel_ids == _RAW_PANELS
        assert board_widget.panel_ids == _RAW_PANELS
        assert staged_frames[0] is not None
        staged_front = staged_frames[0]
        assert board_widget.front_frame is staged_front
        assert tuple(panel.panel_id for panel in staged_front.panels) == _RAW_PANELS
        assert live.fault is None and slot.failure is None
        assert window._handle is handle and window._slot is slot
        assert window._run_owner.generation == old_generation
        assert not handle.snapshot().state.terminal

        def raw_source_advanced():
            _run, _causation, snapshot = slot.freeze_camera_current()
            return (
                snapshot.raw.head is not None
                and snapshot.raw.head.sequence > old_snapshot.raw.head.sequence
            )

        _until(application, raw_source_advanced)
        window._drain_roi_controls()
        window._drain_roi_controls()
        assert len(board_reconfigures) == len(live_reconfigures) == 1
        assert staged_generations == [old_model.layout_generation + 1]
        assert board.model is old_model and board_widget.front_frame is staged_front
        assert receipts[0].snapshot().status is ControlAckStatus.APPLIED
        assert not window._selector_switch.isEnabled()
        assert not window._reducer_combo.isEnabled()
        assert not window._clear_roi_button.isEnabled()
        assert _manifest_files(workspace) == before_manifests
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_stop_folds_applied_roi_before_owner_drain_into_next_prepare(
    experiment_context,
    application,
    monkeypatch,
):
    experiment, workspace = experiment_context
    before_manifests = _manifest_files(workspace)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        board_widget = _start_raw(application, window)
        slot = window._slot
        handle = window._handle
        assert slot is not None and handle is not None
        old_generation = window._run_owner.generation
        receipts = []
        candidates = []
        real_submit = slot.submit_camera_roi_control

        def observe_submit(candidate):
            receipt = real_submit(candidate)
            candidates.append(candidate)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(slot, "submit_camera_roi_control", observe_submit)
        folds = []
        real_fold = window._fold_roi_state_for_detach

        def observe_fold():
            real_fold()
            folds.append((window._applied_request, window._request))

        monkeypatch.setattr(window, "_fold_roi_state_for_detach", observe_fold)

        _draw_first_roi(window, board_widget)
        assert len(receipts) == 1 and len(candidates) == 1
        candidate = candidates[0]
        assert candidate is not None
        receipt = receipts[0]
        ack = receipt.wait(10.0)
        assert ack.status is ControlAckStatus.APPLIED
        assert ack.revision in window._roi_controls
        assert window._applied_request.roi is None
        assert window._request == request

        window._cancel_monitor()
        assert window._manual_stop_requested
        terminal = handle.wait(10.0)
        assert terminal.state is RunState.CANCELLED
        terminal_state = slot.current_camera_roi_state()
        assert terminal_state.binding is None
        assert terminal_state.control_revision == ack.revision
        assert terminal_state.failure_reason is None
        # Exercise the real production ordering: receipt drain observes the
        # cleanup-published raw state before detach performs its authority fold.
        window._drain_roi_controls()
        assert ack.revision in window._roi_controls
        assert window._applied_request.roi is None
        _until(
            application,
            lambda: (
                handle.snapshot().state is RunState.CANCELLED
                and window._run_owner.owner_reaped
                and window._slot is None
                and window._live is None
                and window._prepared is not None
            ),
            timeout=10.0,
        )
        assert len(folds) == 1
        folded_applied, folded_request = folds[0]
        assert folded_applied == folded_request == window._request
        assert window._applied_request == window._request
        assert window._request.roi == candidate.selection
        assert window._request.roi_reduction is candidate.reduction
        assert window._request.roi_validity_policy is candidate.validity_policy
        assert window._prepared.command.request == window._request
        assert window._run_owner.generation == old_generation
        assert not window._roi_controls
        assert receipt.snapshot().status is ControlAckStatus.APPLIED
        assert window._running_binding is None
        assert window._applied_control_revision == ack.revision
        assert not board_widget.has_front
        assert window._visible_binding_fingerprint is None
        assert window._visible_projection_text is None
        assert window._projection_status.text().startswith(
            "Display: no coherent front · target "
        )

        assert _manifest_files(workspace) == before_manifests
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_visible_roi_stop_restart_reuses_widget_and_reearns_applied_overlay(
    experiment_context,
    application,
):
    experiment, workspace = experiment_context
    before_manifests = _manifest_files(workspace)
    initial_roi = _initial_roi(experiment)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        roi=initial_roi,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        board = _board(window)

        def scalar_front_is_visible():
            live = window._live
            binding = window._running_binding
            front = board.front_frame
            status = None if live is None else live.front_status
            return (
                live is not None
                and binding is not None
                and front is not None
                and status is not None
                and status.sequence == front.sequence
                and window._visible_binding_fingerprint == binding.fingerprint
                and window._visible_projection_text == window._projection_text
                and board._selector_applied_bounds
                == board._selector_viewport.normalized_bounds_for_selection(initial_roi)
            )

        _until(application, scalar_front_is_visible)
        first_handle = window._handle
        assert first_handle is not None
        window._cancel_monitor()
        assert first_handle.wait(10.0).state is RunState.CANCELLED
        _until(
            application,
            lambda: (
                window._run_owner.owner_reaped
                and window._slot is None
                and window._live is None
                and window._prepared is not None
            ),
            timeout=20.0,
        )
        assert window._board_widget is board
        assert not board.has_front
        assert window._visible_binding_fingerprint is None
        assert window._visible_projection_text is None
        assert window._projection_status.text().startswith(
            "Display: no coherent front · target "
        )

        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, scalar_front_is_visible, timeout=20.0)
        assert window._board_widget is board
        assert board._selector_draft_bounds is None
        assert window._local_diagnostic == ""
        assert _manifest_files(workspace) == before_manifests
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_overlapping_unpresented_roi_changes_freeze_the_old_coherent_front(
    experiment_context,
    application,
    monkeypatch,
):
    experiment, workspace = experiment_context
    before_manifests = _manifest_files(workspace)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        board_widget = _start_raw(application, window)
        slot = window._slot
        handle = window._handle
        live = window._live
        board = window._board
        assert slot is not None and handle is not None
        assert live is not None and board is not None
        _run, _causation, before = slot.freeze_camera_current()
        receipts = []
        real_submit = slot.submit_camera_roi_control

        def observe_submit(candidate):
            receipt = real_submit(candidate)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(slot, "submit_camera_roi_control", observe_submit)
        board.present_pending()
        monkeypatch.setattr(board, "present_pending", lambda: False)
        old_model = board.model
        _draw_first_roi(window, board_widget)
        assert len(receipts) == 1
        _until(
            application,
            lambda: (
                receipts[0].snapshot().status is ControlAckStatus.APPLIED
                and window._applied_request.roi is not None
                and board._staged_model is not None
                and board_widget._staged_layout is not None
            ),
            timeout=20.0,
        )
        scalar_configuration = live._configuration
        assert tuple(panel.panel_id for panel in scalar_configuration.panels) != _RAW_PANELS
        old_front = board_widget.front_frame
        assert old_front is not None
        assert tuple(panel.panel_id for panel in old_front.panels) == _RAW_PANELS

        real_reconfigure = live.reconfigure_scalar

        def reject_raw_reconfigure(
            state,
            *,
            scalar_dataset_id,
            scalar_documents,
            curve_display,
            histogram_display,
        ):
            if state.binding is None:
                raise RuntimeError("synthetic overlapping raw presentation failure")
            return real_reconfigure(
                state,
                scalar_dataset_id=scalar_dataset_id,
                scalar_documents=scalar_documents,
                curve_display=curve_display,
                histogram_display=histogram_display,
            )

        monkeypatch.setattr(live, "reconfigure_scalar", reject_raw_reconfigure)
        window._clear_roi()
        assert len(receipts) == 2
        _until(
            application,
            lambda: (
                receipts[1].snapshot().status is ControlAckStatus.APPLIED
                and window._roi_presentation_failure is not None
                and not window._roi_controls
            ),
            timeout=20.0,
        )

        assert "synthetic overlapping raw presentation failure" in (
            window._roi_presentation_failure or ""
        )
        assert board.model is old_model
        assert board._staged_model is None
        assert board_widget._staged_layout is None
        assert board_widget.front_frame is old_front
        assert live._configuration is scalar_configuration
        assert live._presentation_frozen and live._port is None and live._sources is None
        assert live.fault is None and slot.failure is None
        assert not handle.snapshot().state.terminal
        assert not window._selector_switch.isEnabled()
        assert not window._reducer_combo.isEnabled()
        assert not window._clear_roi_button.isEnabled()

        def raw_source_advanced():
            _run, _causation, after = slot.freeze_camera_current()
            return (
                after.scalar is None
                and after.raw.head is not None
                and before.raw.head is not None
                and after.raw.head.sequence > before.raw.head.sequence
            )

        _until(application, raw_source_advanced)
        assert _manifest_files(workspace) == before_manifests
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_close_terminalizes_a_pending_roi_revision_without_stale_layout_promotion(
    experiment_context,
    application,
    monkeypatch,
):
    experiment, workspace = experiment_context
    before_manifests = _manifest_files(workspace)
    request = experiment.readout.camera_monitor_request(
        history_capacity=2,
        scalar_history_capacity=8,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    release_projection = threading.Event()
    projection_entered = threading.Event()
    try:
        board = _start_raw(application, window)
        old_handle = window._handle
        old_generation = window._run_owner.generation
        slot = window._slot
        assert old_handle is not None and slot is not None
        receipts = []
        candidates = []
        real_submit = slot.submit_camera_roi_control

        def observe_submit(candidate):
            receipt = real_submit(candidate)
            candidates.append(candidate)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(slot, "submit_camera_roi_control", observe_submit)
        real_project = RoiScalarStreamProjection.project

        def hold_first_projection(self, update, binding, control_revision):
            projection_entered.set()
            if not release_projection.wait(5.0):
                raise TimeoutError("test did not release ROI projection")
            return real_project(self, update, binding, control_revision)

        monkeypatch.setattr(
            RoiScalarStreamProjection,
            "project",
            hold_first_projection,
        )
        staged_layouts = []
        real_stage = board.stage_layout

        def observe_stage(*args, **kwargs):
            staged_layouts.append((args, kwargs))
            return real_stage(*args, **kwargs)

        monkeypatch.setattr(board, "stage_layout", observe_stage)
        _draw_first_roi(window, board)
        assert len(receipts) == 1
        _until(application, projection_entered.is_set)
        second_selection = board._selector_viewport.selection_for_normalized_bounds(
            (0.1, 0.1, 0.4, 0.4)
        )
        publish_started = time.perf_counter()
        window._submit_roi_control(second_selection)
        publish_elapsed = time.perf_counter() - publish_started
        assert len(receipts) == 2
        assert publish_elapsed < 0.5
        assert receipts[1].snapshot().status is ControlAckStatus.ACCEPTED
        window.close()
        assert window.isVisible() and window._closing
        release_projection.set()
        _until(
            application,
            lambda: (
                receipts[1].snapshot().status is ControlAckStatus.TERMINATED
                and not window.isVisible()
            ),
            timeout=10.0,
        )
        assert old_handle.snapshot().state is RunState.CANCELLED
        assert window._run_owner.generation == old_generation
        assert window._handle is old_handle
        assert window._applied_request == request
        assert window._request == request
        assert staged_layouts == []
        assert _manifest_files(workspace) == before_manifests
    finally:
        release_projection.set()
        if window.isVisible():
            _close_window(application, window)
