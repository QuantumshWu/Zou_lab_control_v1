"""M2a typed ROI history and coherent image/curve monitor oracles."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
    MONITOR_HISTORY,
    SPATIAL_X,
    SPATIAL_Y,
    ReductionMethod,
    Selection,
    resolve_selection_indices,
)
from zlc_frontend.figure import (
    AxisViewRole,
    DisplayReductionMethod,
    EvaluatedCurve,
    FigureEvaluator,
    ResolvedDataset,
    ResolvedDatasetMap,
)
from zlc_frontend.qt_widgets import QtRasterBoard
from zlc_neutral_atom.acquisition.camera import CameraFrameMetadataContract
from zlc_neutral_atom.runtime.dataset import (
    dataset_storage_nbytes,
    mutable_dataset_storage_nbytes,
)


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    exp = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("m2-camera-monitor"),
    )
    yield exp
    exp.close()


def _until(application, predicate, *, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close_window(application, window) -> None:
    window.close()
    _until(application, lambda: not window.isVisible(), timeout=8.0)
    assert window not in getattr(application, "_zlc_retained_windows", ())
    window.deleteLater()
    application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    application.processEvents()


def test_typed_roi_curve_preserves_raw_axes_and_presents_one_coherent_board(
    experiment,
    application,
):
    request = experiment.readout.camera_monitor_request(history_capacity=4)
    descriptor = experiment.readout.inspect_camera_monitor(request)
    schema = descriptor.output_schema
    assert descriptor.output_shape == schema.physical_shape
    assert descriptor.output_schema_fingerprint == schema.fingerprint
    assert schema.physical_shape[:2] == (1, 4)
    history = schema.point_axes[0]
    y_axis, x_axis = schema.cell_schema.data_axes
    assert history.role == MONITOR_HISTORY
    assert history.coordinates is None
    assert tuple(history.coordinate_at(index) for index in range(history.size)) == (
        0,
        1,
        2,
        3,
    )
    assert (y_axis.role, x_axis.role) == (SPATIAL_Y, SPATIAL_X)

    x_coordinates = x_axis.coordinates
    y_coordinates = y_axis.coordinates
    assert x_coordinates is not None and y_coordinates is not None
    roi = Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        x_coordinates[0],
        x_coordinates[min(31, x_axis.size - 1)],
        y_coordinates[0],
        y_coordinates[min(31, y_axis.size - 1)],
        coordinate_frame=x_axis.coordinate_frame,
    )
    window = experiment.readout.camera_monitor_gui(
        request,
        roi=roi,
        roi_reduction=ReductionMethod.SUM,
    )
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        projection = window.findChild(QtWidgets.QLabel, "projectionStatus")
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        assert board is not None
        assert window._run_owner.worker_thread_affine
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.has_front
                and window._live is not None
                and window._live.coverage is not None
                and window._live.coverage.written_cells >= 2
            ),
        )

        frame = board.front_frame
        assert frame is not None and len(frame.panels) == 2
        assert tuple(panel.panel_id for panel in frame.panels) == (
            "camera-monitor-image",
            "camera-monitor-roi-curve",
        )
        assert frame.panels[0].coherence_stamp == frame.panels[1].coherence_stamp
        stamp = frame.panels[0].coherence_stamp
        assert len(stamp.inputs) == 1
        assert tuple(value.panel_id for value in stamp.presentations) == (
            "camera-monitor-image",
            "camera-monitor-roi-curve",
        )
        assert "ROI sum" in projection.text()
        assert "DISPLAY ONLY" in projection.text()

        _run_id, _causation, current = window._slot.freeze_current()
        block = current.snapshot.block
        assert block.values.shape == schema.physical_shape
        assert block.values.ndim == 4
        assert current.head is not None and current.event_refs[0] == current.head
        assert len(current.event_refs) == request.history_capacity

        curve_document = window._live._curve_document
        assert curve_document is not None
        bindings = {
            binding.axis_id: binding
            for binding in curve_document.layers[0].view.axis_bindings
        }
        assert bindings[history.axis_id].role is AxisViewRole.X
        assert bindings[schema.repeat_axis.axis_id].role is AxisViewRole.SELECTED
        for axis in (x_axis, y_axis):
            assert bindings[axis.axis_id].role is AxisViewRole.REDUCED
            assert (
                bindings[axis.axis_id].reduction.method
                is DisplayReductionMethod.SUM
            )
        assert curve_document.layers[0].view.display_selections == (roi,)

        evaluated = FigureEvaluator(window._slot.evaluation_policy).evaluate(
            curve_document,
            ResolvedDatasetMap(
                (ResolvedDataset(window._slot.dataset_id, current.snapshot),)
            ),
        )
        curve = evaluated.layers[0].cells[0].series[0].data
        assert isinstance(curve, EvaluatedCurve)
        x_term = next(term for term in roi.terms if term.axis_id == x_axis.axis_id)
        y_term = next(term for term in roi.terms if term.axis_id == y_axis.axis_id)
        x_indices, _ = resolve_selection_indices(x_axis, x_term)
        y_indices, _ = resolve_selection_indices(y_axis, y_term)
        selected = np.take(block.values[0], tuple(y_indices), axis=1)
        selected = np.take(selected, tuple(x_indices), axis=2)
        expected = selected.sum(axis=(1, 2), dtype=np.float64)
        np.testing.assert_allclose(curve.values[curve.validity], expected[curve.validity])

    finally:
        if window.isVisible():
            _close_window(application, window)


def test_history_admission_counts_reorder_scratch_and_every_metadata_record(
    experiment,
):
    one = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=1)
    )
    four = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=4)
    )
    metadata = CameraFrameMetadataContract().max_retained_nbytes
    expected_delta = (
        2 * mutable_dataset_storage_nbytes(four.output_schema)
        + dataset_storage_nbytes(four.output_schema)
        + 4 * metadata
        - mutable_dataset_storage_nbytes(one.output_schema)
        - dataset_storage_nbytes(one.output_schema)
        - metadata
    )
    assert four.base_peak_bytes - one.base_peak_bytes == expected_delta

    oversized = experiment.readout.camera_monitor_request(
        history_capacity=10**9,
        memory_limit_bytes=1,
    )
    with pytest.raises(MemoryError, match="base peak"):
        experiment.readout.inspect_camera_monitor(oversized)


def test_roi_curve_rejects_untyped_or_single_slot_inputs_before_arm(
    experiment,
    application,
):
    with pytest.raises(TypeError, match="ReductionMethod"):
        experiment.readout.camera_monitor_gui(roi_reduction="MEAN")

    descriptor = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=1)
    )
    y_axis, x_axis = descriptor.output_schema.cell_schema.data_axes
    roi = Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        x_axis.coordinates[0],
        x_axis.coordinates[-1],
        y_axis.coordinates[0],
        y_axis.coordinates[-1],
        coordinate_frame=x_axis.coordinate_frame,
    )
    window = experiment.readout.camera_monitor_gui(
        history_capacity=1,
        roi=roi,
    )
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        diagnostics = window.findChild(QtWidgets.QLabel, "diagnostics")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: "history_capacity >= 2" in diagnostics.text())
        assert window._handle is None
        assert not window.findChild(QtRasterBoard, "cameraMonitorImageBoard").has_front
    finally:
        if window.isVisible():
            _close_window(application, window)
