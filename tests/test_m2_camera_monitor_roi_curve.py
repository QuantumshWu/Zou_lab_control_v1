"""M2a typed ROI history and coherent image/curve monitor oracles."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
    MONITOR_HISTORY,
    SPATIAL_X,
    SPATIAL_Y,
    ReductionMethod,
    Selection,
)
from zlc_frontend.figure import (
    AxisViewRole,
    EvaluatedCurve,
    FigureEvaluator,
    ResolvedDataset,
    ResolvedDatasetMap,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.qt_widgets import QtRasterBoard
from zlc_frontend.render import CurvePanelPayload
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


def _wheel_down(
    board: QtRasterBoard,
    target: QtCore.QRect | QtCore.QRectF,
) -> QtGui.QWheelEvent:
    center = target.center()
    position = QtCore.QPoint(int(center.x()), int(center.y()))
    event = QtGui.QWheelEvent(
        QtCore.QPointF(position),
        QtCore.QPointF(board.mapToGlobal(position)),
        QtCore.QPoint(),
        QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    board.wheelEvent(event)
    return event


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
    request = experiment.readout.camera_monitor_request(
        history_capacity=4,
        roi=roi,
        roi_reduction=ReductionMethod.SUM,
    )
    with pytest.raises(TypeError, match="request options"):
        experiment.readout.camera_monitor_gui(request, roi=roi)
    window = experiment.readout.camera_monitor_gui(request)
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
                and window._live.front_status is not None
                and window._live.front_status.sequence
                == board.front_frame.sequence
                and window._live.front_status.raw_coverage.written_cells >= 2
            ),
        )

        frame = board.front_frame
        assert frame is not None and len(frame.panels) == 4
        assert tuple(panel.panel_id for panel in frame.panels) == (
            "camera-monitor-image",
            "camera-monitor-roi-curve",
            "camera-monitor-roi-histogram",
            "camera-monitor-roi-meter",
        )
        assert all(
            panel.coherence_stamp == frame.panels[0].coherence_stamp
            for panel in frame.panels
        )
        stamp = frame.panels[0].coherence_stamp
        assert len(stamp.inputs) == 2
        assert frame.panels[0].source_identity != frame.panels[1].source_identity
        assert all(
            panel.source_identity == frame.panels[1].source_identity
            for panel in frame.panels[1:]
        )
        assert tuple(value.panel_id for value in stamp.presentations) == (
            "camera-monitor-image",
            "camera-monitor-roi-curve",
            "camera-monitor-roi-histogram",
            "camera-monitor-roi-meter",
        )
        assert "ROI sum" in projection.text()
        assert "MONITOR DERIVED" in projection.text()

        _run_id, _causation, joined = window._slot.freeze_camera_current()
        current = joined.raw
        scalar_current = joined.scalar
        assert scalar_current is not None
        block = current.snapshot.block
        assert block.values.shape == schema.physical_shape
        assert block.values.ndim == 4
        assert current.head is not None and current.event_refs[0] == current.head
        assert len(current.event_refs) == request.history_capacity

        curve_document = window._live._configuration.scalar_documents[0]
        bindings = {
            binding.axis_id: binding
            for binding in curve_document.layers[0].view.axis_bindings
        }
        scalar_schema = scalar_current.block.schema
        scalar_history = scalar_schema.point_axes[0]
        assert bindings[scalar_history.axis_id].role is AxisViewRole.X
        assert bindings[scalar_schema.repeat_axis.axis_id].role is AxisViewRole.SELECTED
        assert curve_document.layers[0].view.display_selections == ()

        evaluated = FigureEvaluator(window._slot.evaluation_policy).evaluate(
            curve_document,
            ResolvedDatasetMap(
                (
                    ResolvedDataset(
                        window._slot.scalar_dataset_id,
                        scalar_current.snapshot,
                    ),
                )
            ),
        )
        curve = evaluated.layers[0].cells[0].series[0].data
        assert isinstance(curve, EvaluatedCurve)
        expected = scalar_current.block.values[0]
        np.testing.assert_allclose(
            curve.values[curve.validity],
            expected[curve.validity],
        )

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


def test_curve_selector_and_form_change_only_the_monitor_presentation(
    experiment,
    application,
):
    descriptor = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=3)
    )
    y_axis, x_axis = descriptor.output_schema.cell_schema.data_axes
    roi = Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        x_axis.coordinates[0],
        x_axis.coordinates[min(31, x_axis.size - 1)],
        y_axis.coordinates[0],
        y_axis.coordinates[min(31, y_axis.size - 1)],
        coordinate_frame=x_axis.coordinate_frame,
    )
    window = experiment.readout.camera_monitor_gui(
        history_capacity=3,
        roi=roi,
        scalar_history_capacity=24,
    )
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "roiSelectorSwitch",
        )
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.visible_curve_origin() is not None
                and selector.isEnabled()
            ),
        )
        payload = board.visible_curve_payload()
        assert isinstance(payload, CurvePanelPayload)
        assert payload.viewport.x_axis.role == MONITOR_HISTORY
        assert payload.series and all(
            isinstance(series.data, EvaluatedCurve) for series in payload.series
        )
        assert window._edit_curve_display._form.spec is (
            window._setting_curve_display._form.spec
        )

        handle = window._handle
        slot = window._slot
        controller = window._board
        live = window._live
        assert (
            handle is not None
            and slot is not None
            and controller is not None
            and live is not None
        )
        _run, _epoch, joined_before = slot.freeze_camera_current()
        assert joined_before.scalar is not None
        initial_curve_revision = window._curve_display.revision
        initial_image_revision = window._image_display.revision
        stable_identity = (
            handle,
            handle.snapshot().run_id,
            window._run_owner.generation,
            slot,
            live,
            controller,
            controller.model,
            board,
            board.front_frame.panels[1].source_identity,
        )
        raw_revision = slot.freeze_camera_current()[2].raw.snapshot.ref.revision

        selector.setChecked(True)
        curve_target = board._curve_target()[0]
        start_point = QtCore.QPoint(
            int(curve_target.left() + 0.2 * curve_target.width()),
            int(curve_target.center().y()),
        )
        end_point = QtCore.QPoint(
            int(curve_target.left() + 0.8 * curve_target.width()),
            int(curve_target.center().y()),
        )
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start_point)
        board.mouseMoveEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove,
                QtCore.QPointF(end_point),
                QtCore.Qt.NoButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
        )
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end_point)
        assert window._curve_range_candidate is not None
        assert (
            board._curve_applied_span
            == window._curve_range_candidate[1]
        )
        assert window._curve_display.revision == initial_curve_revision

        assert _wheel_down(board, board._curve_target()[0]).isAccepted()
        authored = window._curve_display
        assert (
            authored.revision == initial_curve_revision + 1
            and authored.x_view is not None
        )
        assert board._curve_applied_span == window._curve_range_candidate[1]
        assert selector.isChecked() and selector.isEnabled()
        assert board._curve_pending_viewport is not None
        _until(
            application,
            lambda: (
                board.visible_curve_origin() is not None
                and board.visible_curve_origin().presentation.panel_revision
                == authored.revision
            ),
        )
        _until(
            application,
            lambda: (
                slot.freeze_camera_current()[2].raw.snapshot.ref.revision
                > raw_revision
            ),
        )
        assert stable_identity == (
            window._handle,
            window._handle.snapshot().run_id,
            window._run_owner.generation,
            window._slot,
            window._live,
            window._board,
            window._board.model,
            window._board_widget,
            window._board_widget.front_frame.panels[1].source_identity,
        )
        _run, _epoch, joined_after = slot.freeze_camera_current()
        assert joined_after.scalar is not None
        assert joined_after.scalar.ref.block_id == joined_before.scalar.ref.block_id
        assert (
            joined_after.scalar.ref.stream_generation
            == joined_before.scalar.ref.stream_generation
        )

        painted_y = board.visible_curve_payload().viewport.y_limits
        editor = window._setting_curve_display
        values = editor._form.read_all()
        values["relim_mode"] = RelimMode.FIXED
        window._apply_curve_display_form(editor, editor.base_revision, values)
        fixed = window._curve_display
        assert fixed.revision == authored.revision + 1
        assert fixed.relim_mode is RelimMode.FIXED
        assert fixed.fixed_y_limits == painted_y
        _until(
            application,
            lambda: (
                board.visible_curve_origin() is not None
                and board.visible_curve_origin().presentation.panel_revision
                == fixed.revision
            ),
        )

        # A pending IMAGE revision parks only IMAGE.  The same global switch
        # remains armed for the still-current CURVE panel; neither editor nor
        # interaction readiness is allowed to collapse to a board-wide lock.
        assert _wheel_down(board, board._selector_target()[0]).isAccepted()
        image_revision = window._image_display.revision
        assert image_revision == initial_image_revision + 1
        assert board._pending_viewport is not None
        assert selector.isChecked() and selector.isEnabled()
        assert not window._edit_image_display.isEnabled()
        assert window._edit_curve_display.isEnabled()
        curve_plot = board._curve_target()[0]
        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=QtCore.QPoint(
                int(curve_plot.center().x()),
                int(curve_plot.center().y()),
            ),
        )
        assert board._curve_cross is not None
        _until(
            application,
            lambda: (
                board.visible_image_origin() is not None
                and board.visible_image_origin().presentation.panel_revision
                == image_revision
            ),
        )
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_roi_curve_rejects_untyped_reducer_but_accepts_one_raw_history_slot(
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
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: board.has_front)
        assert window._handle is not None
        _run, _epoch, joined = window._slot.freeze_camera_current()
        assert joined.raw.block.values.shape[1] == 1
        assert joined.scalar is not None
        assert joined.scalar.block.values.shape[1] == 300
    finally:
        if window.isVisible():
            _close_window(application, window)
