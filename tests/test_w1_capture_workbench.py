"""Current W1 finite exact-capture Workbench product oracles."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.notebook.facade import _prepare_capture_for_workbench
from Zou_lab_control.workbench import open_capture_workbench
from zlc_data import BlockId, SPATIAL_X, SPATIAL_Y
from zlc_frontend.figure import DatasetId, FigureEvaluationPolicy
from zlc_frontend.qt_widgets import GREEN, ORANGE, QtImageBoard
from zlc_workbench.live import LiveDatasetSlot


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

    def fail_raster(_image):
        raise RuntimeError("injected raster failure")

    monkeypatch.setattr(live_module, "rasterize_image_gray8", fail_raster)
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
