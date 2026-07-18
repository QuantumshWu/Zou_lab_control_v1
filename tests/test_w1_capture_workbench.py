"""Current W1 finite exact-capture Workbench product oracles."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.notebook.facade import _prepare_capture_for_workbench
from Zou_lab_control.workbench import open_capture_workbench
from zlc_data import BlockId, SPATIAL_X, SPATIAL_Y
from zlc_frontend.figure import DatasetId, FigureEvaluationPolicy
from zlc_frontend.qt_widgets import GREEN, ORANGE, QtImageBoard
from zlc_neutral_atom.monitor_application import (
    CameraMonitorLiveDataset,
    CameraMonitorRoiState,
)
from zlc_neutral_atom.runtime.control import ControlAckStatus, create_control_topic
import zlc_workbench.live as live_module
from zlc_workbench.live import LiveDatasetSlot, LiveImageBoardController


ROOT = Path(__file__).parents[1]
SINGLE_EVENT_PULSE = ROOT / "pulses" / "probe_template.json"
MULTI_EVENT_PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    exp = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("w1-workspace"),
    )
    yield exp
    exp.close()


def _until(application, predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _widgets(window):
    return (
        window.findChild(QtWidgets.QPushButton, "startButton"),
        window.findChild(QtWidgets.QLabel, "captureStatus"),
        window.findChild(QtWidgets.QLabel, "previewStatus"),
        window.findChild(QtWidgets.QLabel, "diagnostics"),
        window.findChild(QtImageBoard, "captureImageBoard"),
    )


def _close_window(application, window) -> None:
    window.close()
    _until(application, lambda: not window.isVisible(), timeout=5.0)
    assert window not in getattr(application, "_zlc_retained_windows", ())


def test_public_workbench_repeats_with_new_preview_and_loads_exact_artifact(
    experiment,
    application,
):
    window = None
    first = None
    second = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, capture, preview, diagnostics, board = _widgets(window)
        stop = window.findChild(QtWidgets.QPushButton, "stopButton")
        assert start._base_color == QtGui.QColor(GREEN).name(QtGui.QColor.HexRgb)
        assert stop._base_color == QtGui.QColor(ORANGE).name(QtGui.QColor.HexRgb)
        available = application.primaryScreen().availableGeometry()
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        assert available.contains(window.frameGeometry())
        _until(application, start.isEnabled)

        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: window.final_reference is not None)
        first = window.final_reference
        assert capture.text().startswith("Capture: FINAL")
        assert preview.text().startswith("Preview: DISPLAY ONLY · capture FINAL")
        assert board.has_front and diagnostics.text() == ""
        assert available.contains(window.frameGeometry())
        _until(application, start.isEnabled)

        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None
            and window.final_reference != first,
        )
        second = window.final_reference
        assert capture.text().startswith("Capture: FINAL")
        assert board.has_front
        _close_window(application, window)
        assert experiment.inspect(request).output_shape == (1, 1, 96, 128)
    finally:
        if window is not None and window.isVisible():
            _close_window(application, window)
    assert first is not None and second is not None and first != second
    artifact = experiment.readout.load_capture(second)
    schema = artifact.frame_source.schema
    assert schema.physical_shape == (1, 1, 96, 128)
    assert tuple(axis.role for axis in schema.cell_schema.data_axes) == (
        SPATIAL_Y,
        SPATIAL_X,
    )


def test_multi_event_capture_is_rejected_before_start(experiment, application):
    window = None
    try:
        request = experiment.readout.capture_request(MULTI_EVENT_PULSE)
        generic_ref = experiment.run(request)
        assert (
            experiment.readout.load_capture(generic_ref).frame_source.schema.physical_shape
            == (1, 3, 96, 128)
        )
        window = open_capture_workbench(experiment, request)
        start, capture, _preview, diagnostics, _board = _widgets(window)
        _until(application, lambda: "Preparation failed" in diagnostics.text())
        assert not start.isEnabled()
        assert capture.text() == "Capture: NOT READY · NOT FINAL"
        assert "exactly one readout event per repeat" in diagnostics.text()
    finally:
        if window is not None:
            _close_window(application, window)


def test_render_failure_does_not_change_final_capture(
    experiment,
    application,
    monkeypatch,
):
    import zlc_workbench.live as live_module

    def fail_raster(_image, _state, **_kwargs):
        raise RuntimeError("injected raster failure")

    monkeypatch.setattr(live_module, "rasterize_image_indexed8", fail_raster)
    window = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, capture, preview, _diagnostics, board = _widgets(window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: window.final_reference is not None)
        _until(application, lambda: preview.text().startswith("Preview: FAILED"))
        assert capture.text().startswith("Capture: FINAL")
        assert not board.has_front
        assert (
            experiment.readout.load_capture(window.final_reference)
            .frame_source.schema.physical_shape
            == (1, 1, 96, 128)
        )
    finally:
        if window is not None:
            _close_window(application, window)


def test_preview_budget_failure_is_visible_and_next_legal_window_succeeds(
    experiment,
    application,
):
    bad_window = None
    good_window = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        bad_window = open_capture_workbench(
            experiment,
            replace(request, pipeline_memory_limit_bytes=1),
        )
        start, capture, preview, diagnostics, board = _widgets(bad_window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: "pipeline peak budget" in diagnostics.text())
        assert "FAILED" in capture.text() and "NOT FINAL" in capture.text()
        assert preview.text().startswith("Preview: FAILED")
        assert not board.has_front
        _close_window(application, bad_window)

        good_window = open_capture_workbench(experiment, request)
        start, capture, _preview, _diagnostics, board = _widgets(good_window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: good_window.final_reference is not None)
        assert capture.text().startswith("Capture: FINAL") and board.has_front
    finally:
        if bad_window is not None and bad_window.isVisible():
            _close_window(application, bad_window)
        if good_window is not None:
            _close_window(application, good_window)


def test_close_during_start_is_nonblocking_and_does_not_close_experiment(
    experiment,
    application,
):
    request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
    window = open_capture_workbench(experiment, request)
    try:
        start, _capture, _preview, _diagnostics, _board = _widgets(window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        began = time.monotonic()
        window.close()
        assert time.monotonic() - began < 0.1
        _until(application, lambda: not window.isVisible(), timeout=5.0)
        assert window.worker_idle
        assert experiment.inspect(request).output_shape == (1, 1, 96, 128)
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_slot_replays_update_or_failure_once_to_a_late_listener(experiment):
    request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
    prepared = _prepare_capture_for_workbench(experiment, request)
    try:
        slots = []

        def factory(spec):
            slot = LiveDatasetSlot(
                spec,
                dataset_id=DatasetId("late-update"),
                evaluation_policy=FigureEvaluationPolicy(),
            )
            slots.append(slot)
            return slot

        handle = prepared.start_with_preview(
            block_id=BlockId("late-update-block"),
            downstream_peak_bytes=0,
            factory=factory,
        )
        handle.result(5.0)
        calls = []
        slots[0].set_change_listener(lambda: calls.append("update"))
        assert calls == ["update"]
        with pytest.raises(RuntimeError, match="already has"):
            slots[0].set_change_listener(lambda: None)
        slots[0].close()

        bad = _prepare_capture_for_workbench(
            experiment,
            replace(request, pipeline_memory_limit_bytes=1),
        )
        failed_slots = []

        def failed_factory(spec):
            slot = LiveDatasetSlot(
                spec,
                dataset_id=DatasetId("late-failure"),
                evaluation_policy=FigureEvaluationPolicy(),
            )
            failed_slots.append(slot)
            return slot

        with pytest.raises(MemoryError, match="pipeline peak budget"):
            bad.start_with_preview(
                block_id=BlockId("late-failure-block"),
                downstream_peak_bytes=0,
                factory=failed_factory,
            )
        failures = []
        failed_slots[0].set_change_listener(lambda: failures.append("failure"))
        assert failures == ["failure"]
        assert failed_slots[0].failure is not None
        failed_slots[0].fail("second failure must not replay")
        assert failures == ["failure"]
        failed_slots[0].close()
    finally:
        for slot in locals().get("slots", ()):
            slot.close()
        for slot in locals().get("failed_slots", ()):
            slot.close()


def test_public_import_is_lazy_and_w1_has_no_runtime_boundary_leak():
    code = (
        "import sys; import Zou_lab_control.workbench, zlc_workbench; "
        "assert not any(n == 'PyQt5' or n.startswith('PyQt5.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    source = (ROOT / "Zou_lab_control" / "workbench" / "_capture.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "MinimalPipelineSpec",
        "RunPlan",
        "BoundCapturePort",
        "BoundPulsePort",
        "compile_capture_artifact_pipeline",
        "_authority_token",
    ):
        assert forbidden not in source


class _CameraControlDataset(CameraMonitorLiveDataset):
    def __init__(self, state: CameraMonitorRoiState) -> None:
        self.state = state
        self.closed = False
        self.slot = None
        self.topic, self.consumer = create_control_topic(lambda value: value)

    def current_roi_state(self):
        return self.state

    def submit_roi_control(self, candidate):
        receipt = self.topic.publish(candidate)
        if self.slot is not None:
            self.slot.close()
        return receipt

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.consumer.terminate("camera control dataset closed")


def _camera_control_slot(dataset):
    slot = object.__new__(LiveDatasetSlot)
    slot._lock = threading.Lock()
    slot._dataset = dataset
    slot._camera_roi_state = CameraMonitorRoiState(
        None,
        None,
        None,
        None,
        None,
    )
    slot._failure = None
    slot._notification_failure = None
    slot._terminal = False
    slot._withdrawn = False
    slot._closed = False
    slot._listener = None
    slot._listener_claimed = False
    slot._pending_change = False
    slot._retain_on_terminal = False
    return slot


def test_notification_failure_is_visible_without_detaching_source_dataset():
    dataset = _CameraControlDataset(
        CameraMonitorRoiState(None, None, None, None, None)
    )
    slot = _camera_control_slot(dataset)
    wakes = []
    slot.set_change_listener(lambda: wakes.append("wake"))
    slot.notification_failed("view change listener failed")
    assert slot.failure is None
    assert slot.notification_failure == "view change listener failed"
    assert wakes == ["wake"]
    assert slot._dataset is dataset
    assert not dataset.closed
    assert slot.current_camera_roi_state() == dataset.state

    slot.notification_failed("later failure must not replace first cause")
    assert slot.failure is None
    assert slot.notification_failure == "view change listener failed"
    assert wakes == ["wake"]
    assert slot._dataset is dataset

    visible_status = object()
    controller = object.__new__(LiveImageBoardController)
    controller._slot = slot
    controller._lock = threading.Lock()
    controller._closed = False
    controller._fault = None
    controller._candidate = object()
    controller._active = True
    controller._dirty = True
    controller._port = object()
    controller._sources = (object(),)
    controller._front_status = visible_status
    revoked = []

    def revoke(before_revoke=None):
        if before_revoke is not None and not before_revoke():
            return False
        revoked.append(True)
        return True

    controller._board = SimpleNamespace(
        revoke_pending_publication=revoke
    )
    controller._request_owner_wake = lambda: wakes.append("controller-fault")
    assert controller.admit_pending() is False
    assert controller.fault is not None
    assert controller._front_status is visible_status
    assert revoked == [True]
    assert slot._dataset is dataset and not dataset.closed

    slot.source_terminal()
    assert dataset.closed and slot.terminal


def test_camera_control_submit_returns_receipt_when_detach_loses_the_race():
    state = CameraMonitorRoiState(None, None, None, None, None)
    dataset = _CameraControlDataset(state)
    slot = _camera_control_slot(dataset)
    dataset.slot = slot
    receipt = slot.submit_camera_roi_control(None)
    assert receipt.snapshot().status is ControlAckStatus.TERMINATED
    assert dataset.closed and slot.terminal


@pytest.mark.parametrize("terminal", [False, True])
def test_camera_roi_state_survives_failure_or_terminal_detach(terminal):
    initial = CameraMonitorRoiState(None, 1, None, None, None, 1)
    final = CameraMonitorRoiState(
        None,
        1,
        None,
        None,
        None,
        2,
        "scalar branch failed",
    )
    dataset = _CameraControlDataset(initial)
    slot = _camera_control_slot(dataset)
    assert slot.current_camera_roi_state() is initial
    dataset.state = final
    if terminal:
        slot.source_terminal()
    else:
        slot.fail("camera source failed")
    assert slot.current_camera_roi_state() is final
    assert dataset.closed


def test_camera_roi_state_cache_rejects_conflicting_same_revision_truth():
    first = CameraMonitorRoiState(None, 1, None, None, None, 1)
    conflicting = CameraMonitorRoiState(None, 2, None, None, None, 1)
    dataset = _CameraControlDataset(first)
    slot = _camera_control_slot(dataset)
    assert slot.current_camera_roi_state() is first
    dataset.state = conflicting
    with pytest.raises(RuntimeError, match="without advancing state_revision"):
        slot.current_camera_roi_state()
    assert slot._camera_roi_state is first


def test_freeze_presentation_revokes_work_and_preserves_the_coherent_front_state():
    controller = object.__new__(LiveImageBoardController)
    controller._owner_thread = threading.get_ident()
    controller._lock = threading.Lock()
    controller._closed = False
    controller._candidate = object()
    controller._port = object()
    controller._sources = (object(),)
    controller._dirty = True
    controller._active = True
    controller._presentation_frozen = False
    frozen = []
    controller._board = SimpleNamespace(freeze_front=lambda: frozen.append(True))
    controller.freeze_presentation()
    assert controller._candidate is None
    assert controller._port is None and controller._sources is None
    assert not controller._dirty and not controller._active
    assert controller._presentation_frozen
    assert frozen == [True]


def test_scalar_control_change_reuses_layout_when_panel_ids_are_unchanged(monkeypatch):
    class CameraSpec:
        pass

    panels = (SimpleNamespace(panel_id="image"),)
    previous = SimpleNamespace(
        board_id="camera-board",
        layout_generation=4,
        panels=panels,
        image_document=object(),
        image_display=None,
        image_viewport=None,
        scalar_documents=(object(),),
    )
    replacement = SimpleNamespace(
        board_id=previous.board_id,
        layout_generation=previous.layout_generation,
        panels=panels,
        scalar_documents=(),
        presentations=(object(),),
    )
    requests = []
    retired = []
    monkeypatch.setattr(live_module, "CameraMonitorViewSpec", CameraSpec)
    monkeypatch.setattr(
        live_module,
        "_build_live_configuration",
        lambda **_kwargs: replacement,
    )
    controller = object.__new__(LiveImageBoardController)
    controller._owner_thread = threading.get_ident()
    controller._slot = SimpleNamespace(
        spec=CameraSpec(),
        failure=None,
        withdrawn=False,
    )
    controller._worker_thread_affine = True
    controller._scalar_dataset_generations = {}
    controller._scalar_generation_datasets = {}
    controller._lock = threading.Lock()
    controller._closed = False
    controller._fault = None
    controller._configuration_epoch = 8
    controller._configuration = previous
    controller._port = None
    controller._sources = None
    controller._candidate = None
    controller._dirty = False
    controller._active = False
    controller._presentation_frozen = False
    controller._board = SimpleNamespace(
        model=SimpleNamespace(
            board_id=previous.board_id,
            layout_generation=previous.layout_generation,
            panels=panels,
        )
    )
    controller._document = object()
    controller._submit = lambda work: retired.append(work)
    controller._request_snapshot = lambda: requests.append(None)

    controller.reconfigure_scalar(
        CameraMonitorRoiState(None, 3, None, None, None, 3),
        None,
        (),
    )

    assert controller._configuration is replacement
    assert controller._configuration_epoch == 9
    assert retired and requests == [None]
