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
import Zou_lab_control.workbench._camera_monitor as camera_workbench
from zlc_data import ReductionMethod
from zlc_frontend.qt_widgets import QtRasterBoard
from zlc_neutral_atom.runtime.run import RunState


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
    selection = window._draft_selection
    assert selection is not None
    assert window._applied_request.roi is None
    assert window._apply_roi_button.isEnabled()
    return selection


def test_raw_first_roi_rejects_without_disruption_then_replaces_the_whole_generation(
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

        draft = _draw_first_roi(window, raw_board)
        max_index = window._reducer_combo.findData(ReductionMethod.MAX)
        assert max_index >= 0
        window._reducer_combo.setCurrentIndex(max_index)

        real_prepare_view = camera_workbench._prepare_monitor_view

        def reject_scalar_candidate(command, generation):
            if command.request.roi is not None:
                raise MemoryError("synthetic first-ROI downstream rejection")
            return real_prepare_view(command, generation)

        monkeypatch.setattr(
            camera_workbench,
            "_prepare_monitor_view",
            reject_scalar_candidate,
        )
        window._apply_roi_button.click()
        _until(
            application,
            lambda: (
                window._apply_phase is None
                and "rejected before restart" in window._roi_status.text()
            ),
        )
        assert window._handle is old_handle
        assert not old_handle.snapshot().state.terminal
        assert window._run_owner.generation == old_generation
        assert window._slot is old_slot
        assert window._board_widget is raw_board
        assert window._board_panel_ids == _RAW_PANELS
        assert raw_board.has_front
        assert window._applied_request == request
        assert window._draft_selection == draft
        _run, _causation, after_reject = old_slot.freeze_camera_current()
        assert after_reject.scalar is None
        assert after_reject.raw.ref.block_id == old_ref.block_id
        assert after_reject.raw.ref.stream_generation == old_ref.stream_generation

        monkeypatch.setattr(
            camera_workbench,
            "_prepare_monitor_view",
            real_prepare_view,
        )
        topology_handoff = []
        real_configure = window._configure_board_widget

        def observe_configure(*, scalar):
            if scalar:
                topology_handoff.append(
                    (
                        old_handle.snapshot().state,
                        window._run_owner.owner_reaped,
                        window._live,
                        window._board_panel_ids,
                    )
                )
            return real_configure(scalar=scalar)

        monkeypatch.setattr(window, "_configure_board_widget", observe_configure)
        window._apply_roi_button.click()

        def replacement_visible():
            board = _board(window)
            return (
                window._apply_phase is None
                and window._applied_request.roi == draft
                and window._applied_request.roi_reduction is ReductionMethod.MAX
                and window._handle is not None
                and window._handle is not old_handle
                and window._board_panel_ids == _SCALAR_PANELS
                and board.has_front
                and board.front_frame is not None
                and tuple(panel.panel_id for panel in board.front_frame.panels)
                == _SCALAR_PANELS
                and window._live is not None
                and window._live.front_status is not None
                and window._live.front_status.sequence == board.front_frame.sequence
            )

        _until(application, replacement_visible)
        scalar_board = _board(window)
        assert topology_handoff == [
            (RunState.CANCELLED, True, None, _RAW_PANELS)
        ]
        assert old_handle.snapshot().state is RunState.CANCELLED
        assert window._run_owner.generation == old_generation + 1
        assert window._handle.run_id != old_run_id
        assert window._slot is not old_slot
        assert window._slot.dataset_id != old_dataset_id
        _run, _causation, replacement = window._slot.freeze_camera_current()
        assert replacement.scalar is not None
        assert replacement.raw.ref.block_id != old_ref.block_id
        assert replacement.raw.ref.stream_generation != old_ref.stream_generation
        assert window._running_binding.selection == draft
        assert window._running_binding.reduction is ReductionMethod.MAX
        assert scalar_board._selector_applied_bounds == (
            scalar_board._selector_viewport.normalized_bounds_for_selection(draft)
        )
        assert scalar_board._selector_draft_bounds is None
        assert _manifest_files(workspace) == before_manifests
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_close_during_first_roi_replacement_never_promotes_the_candidate(
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
    release_reap = threading.Event()
    try:
        board = _start_raw(application, window)
        _draw_first_roi(window, board)
        old_handle = window._handle
        old_generation = window._run_owner.generation
        assert old_handle is not None
        real_wait = old_handle.wait

        def hold_reap(timeout=None):
            snapshot = real_wait(timeout)
            if not release_reap.wait(5.0):
                raise TimeoutError("test did not release first-ROI reap")
            return snapshot

        monkeypatch.setattr(old_handle, "wait", hold_reap)
        scalar_topology_calls = []
        real_configure = window._configure_board_widget

        def observe_configure(*, scalar):
            scalar_topology_calls.append(scalar)
            return real_configure(scalar=scalar)

        monkeypatch.setattr(window, "_configure_board_widget", observe_configure)
        window._apply_roi_button.click()
        _until(
            application,
            lambda: (
                window._apply_phase == "REPLACING"
                and old_handle.snapshot().state.terminal
            ),
        )
        window.close()
        assert window.isVisible()
        assert window._closing is True
        assert window._prepared_apply is None
        assert window._pending_request is None
        release_reap.set()
        _until(application, lambda: not window.isVisible(), timeout=10.0)
        assert old_handle.snapshot().state is RunState.CANCELLED
        assert window._run_owner.generation == old_generation
        assert window._handle is old_handle
        assert window._applied_request == request
        assert window._request == request
        assert True not in scalar_topology_calls
        assert _manifest_files(workspace) == before_manifests
    finally:
        release_reap.set()
        if window.isVisible():
            _close_window(application, window)
