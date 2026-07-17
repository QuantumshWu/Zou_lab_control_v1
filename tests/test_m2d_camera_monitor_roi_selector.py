"""M2d front-bound ROI selection and immutable monitor replacement oracles."""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
import Zou_lab_control.workbench._camera_monitor as camera_workbench
from zlc_data import (
    INVALID,
    VALID,
    AxisId,
    AxisSpec,
    ComponentValidity,
    CoordinateFrameId,
    ReductionMethod,
    SPATIAL_X,
    SPATIAL_Y,
    Selection,
    ValidityContract,
    ValidityPolicy,
    Value,
    ValueSchema,
)
from zlc_frontend.qt_widgets import ImageViewportTransform, QtRasterBoard
from zlc_neutral_atom.acquisition.camera import (
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.processing.roi_monitor import (
    RoiScalarBinding,
    RoiScalarMetadata,
    RoiScalarMetadataContract,
    reduce_camera_roi,
)
from zlc_neutral_atom.runtime.run import RunState
from zlc_storage import canonical_digest
from zlc_workbench.live import LiveImageBoardController


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    exp = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("m2d-camera-monitor"),
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
    window.deleteLater()
    application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    application.processEvents()


def _camera_kernel(dtype=np.dtype("<u2")):
    frame = CoordinateFrameId("m2d-camera-pixels")
    y_axis = AxisSpec(
        AxisId("m2d.y"),
        "y",
        SPATIAL_Y,
        1,
        (0,),
        unit="pixel",
        coordinate_frame=frame,
    )
    x_axis = AxisSpec(
        AxisId("m2d.x"),
        "x",
        SPATIAL_X,
        2,
        (0, 1),
        unit="pixel",
        coordinate_frame=frame,
    )
    schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        np.dtype(dtype),
        "camera-count",
    )
    selection = Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        0,
        1,
        0,
        0,
        coordinate_frame=frame,
    )
    return schema, CameraSampleContract(schema), selection, y_axis, x_axis


def _sample(schema, values, validity) -> CameraSample:
    metadata = CameraFrameMetadata(
        0,
        1,
        1,
        1,
        None,
        None,
        1,
        0,
        "m2d-kernel",
    )
    y_axis, x_axis = schema.data_axes
    return CameraSample(
        Value(
            np.asarray(values, dtype=schema.dtype).reshape(schema.data_shape),
            ComponentValidity(
                (y_axis.axis_id, x_axis.axis_id),
                np.asarray(validity, dtype=bool).reshape(schema.data_shape),
            ),
            schema,
        ),
        metadata,
    )


def test_viewport_uses_declared_pixel_roles_and_round_trips_exact_cells():
    frame = CoordinateFrameId("m2d-viewport")
    x_axis = AxisSpec(
        AxisId("viewport.x"),
        "x",
        SPATIAL_X,
        4,
        (10, 11, 12, 13),
        unit="pixel",
        coordinate_frame=frame,
    )
    y_axis = AxisSpec(
        AxisId("viewport.y"),
        "y",
        SPATIAL_Y,
        3,
        (20, 21, 22),
        unit="pixel",
        coordinate_frame=frame,
    )
    viewport = ImageViewportTransform((y_axis, x_axis), viewport_revision=7)
    bounds = (0.25, 1 / 3, 0.75, 1.0)
    selection = viewport.selection_for_normalized_bounds(bounds)
    assert viewport.axes == (x_axis, y_axis)
    assert viewport.normalized_bounds_for_selection(selection) == pytest.approx(bounds)
    assert viewport.viewport_revision == 7

    wrong_unit = AxisSpec(
        x_axis.axis_id,
        x_axis.name,
        x_axis.role,
        x_axis.size,
        x_axis.coordinates,
        unit="px",
        coordinate_frame=frame,
    )
    with pytest.raises(ValueError, match="unit='pixel'"):
        ImageViewportTransform((wrong_unit, y_axis))


def test_max_is_typed_validity_aware_and_all_roi_reducers_reject_valid_nonfinite():
    schema, contract, selection, _y_axis, _x_axis = _camera_kernel()
    omit = RoiScalarBinding(
        contract,
        selection,
        ReductionMethod.MAX,
        ValidityPolicy.OMIT_INVALID,
    )
    require_all = RoiScalarBinding(contract, selection, ReductionMethod.MAX)
    partially_valid = _sample(schema, [65535, 7], [False, True])
    omitted = reduce_camera_roi(partially_valid, omit)
    strict = reduce_camera_roi(partially_valid, require_all)
    assert omitted.values.dtype == np.dtype("<u2")
    assert omitted.values.item() == 7 and omitted.validity is VALID
    assert strict.values.item() == 7 and strict.validity is INVALID
    all_invalid = reduce_camera_roi(
        _sample(schema, [65535, 7], [False, False]),
        omit,
    )
    assert all_invalid.values.item() == 0 and all_invalid.validity is INVALID
    assert omit.reduction_scratch_nbytes == 2 * schema.dtype.itemsize

    signed_schema, signed_contract, signed_selection, *_ = _camera_kernel("<i2")
    signed = reduce_camera_roi(
        _sample(signed_schema, [-7, -2], [True, True]),
        RoiScalarBinding(
            signed_contract,
            signed_selection,
            ReductionMethod.MAX,
            ValidityPolicy.OMIT_INVALID,
        ),
    )
    assert signed.values.dtype == np.dtype("<i2") and signed.values.item() == -2

    float_schema, float_contract, float_selection, *_ = _camera_kernel("<f4")
    float_max = RoiScalarBinding(
        float_contract,
        float_selection,
        ReductionMethod.MAX,
        ValidityPolicy.OMIT_INVALID,
    )
    assert float_max.output_schema.dtype == np.dtype("<f4")
    assert float_max.reduction_scratch_nbytes == 2 * (
        np.dtype("<f4").itemsize + np.dtype(bool).itemsize
    )
    invalid_nan = _sample(float_schema, [np.nan, 3.0], [False, True])
    valid_nan = _sample(float_schema, [np.nan, 3.0], [True, True])
    for method in (
        ReductionMethod.MEAN,
        ReductionMethod.SUM,
        ReductionMethod.MAX,
    ):
        binding = RoiScalarBinding(
            float_contract,
            float_selection,
            method,
            ValidityPolicy.OMIT_INVALID,
        )
        assert reduce_camera_roi(invalid_nan, binding).values.item() == pytest.approx(3.0)
        with pytest.raises(ValueError, match="non-finite"):
            reduce_camera_roi(valid_nan, binding)

    complex_schema, complex_contract, complex_selection, *_ = _camera_kernel("<c8")
    assert complex_schema.dtype == np.dtype("<c8")
    with pytest.raises(TypeError, match="undefined for complex"):
        RoiScalarBinding(
            complex_contract,
            complex_selection,
            ReductionMethod.MAX,
        )

    assert omit.fingerprint != require_all.fingerprint
    assert omit.fingerprint != RoiScalarBinding(
        contract,
        selection,
        ReductionMethod.SUM,
        ValidityPolicy.OMIT_INVALID,
    ).fingerprint
    metadata_contract = RoiScalarMetadataContract(CameraFrameMetadataContract())
    old_layout_fingerprint = canonical_digest(
        {
            "contract": "zlc_neutral_atom.RoiScalarMetadata",
            "source_metadata": metadata_contract.source_metadata_contract.fingerprint,
            "source_reference_max_bytes": metadata_contract.source_reference_max_bytes,
        }
    )
    assert metadata_contract.fingerprint != old_layout_fingerprint


def test_pause_ends_a_candidate_waiting_for_gui_admission():
    controller = object.__new__(LiveImageBoardController)
    controller._lock = threading.Lock()
    controller._closed = False
    controller._paused = False
    controller._dirty = True
    controller._active = True
    controller._candidate = object()
    controller.pause()
    assert controller._paused is True
    assert controller._dirty is False
    assert controller._candidate is None
    assert controller._active is False


def test_manual_stop_intent_cannot_promote_a_prepared_replacement(
    experiment,
    application,
    monkeypatch,
):
    initial_roi = _initial_roi(experiment)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        roi=initial_roi,
        roi_reduction=ReductionMethod.MEAN,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    release_reap = threading.Event()
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        stop = window.findChild(QtWidgets.QPushButton, "stopButton")
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.has_front
                and window._selector_switch.isEnabled()
                and window._handle is not None
            ),
        )
        old_handle = window._handle
        old_generation = window._run_owner.generation
        assert old_handle is not None

        cancelled_snapshot = {}
        real_cancel = old_handle.cancel
        real_wait = old_handle.wait

        def capture_replacement_cancel(reason="user requested stop"):
            outcome = real_cancel(reason)
            if reason == "Workbench is applying a new immutable ROI request":
                cancelled_snapshot["value"] = old_handle.snapshot()
            return outcome

        def hold_reap(timeout=None):
            snapshot = real_wait(timeout)
            if not release_reap.wait(5.0):
                raise TimeoutError("test did not release replacement reap")
            return snapshot

        monkeypatch.setattr(old_handle, "cancel", capture_replacement_cancel)
        monkeypatch.setattr(old_handle, "wait", hold_reap)
        max_index = window._reducer_combo.findData(ReductionMethod.MAX)
        assert max_index >= 0
        window._reducer_combo.setCurrentIndex(max_index)
        window._apply_roi_button.click()
        _until(
            application,
            lambda: (
                window._apply_phase == "REPLACING"
                and "value" in cancelled_snapshot
            ),
        )

        cancelling = cancelled_snapshot["value"]
        assert cancelling.state is RunState.CANCELLING
        window._reconcile_snapshot(cancelling)
        assert not stop.isEnabled()

        # Models a queued/programmatic Stop reaching the owner after Apply's
        # first cancellation request.  Promotion must honor the intent even
        # though the second cancel outcome would be ALREADY_REQUESTED.
        window._manual_stop_requested = True
        window._reconcile_snapshot(cancelling)
        assert not stop.isEnabled()
        release_reap.set()
        _until(
            application,
            lambda: (
                window._apply_phase is None
                and "stopped by user" in window._roi_status.text()
                and window._prepared is not None
            ),
        )
        assert old_handle.snapshot().state is RunState.CANCELLED
        assert window._run_owner.generation == old_generation
        assert window._handle is old_handle
        assert window._request == request
        assert window._applied_request == request
        assert window._pending_request is None
        assert window._prepared_apply is None
        assert not board.has_front
        assert start.isEnabled()
    finally:
        release_reap.set()
        if window.isVisible():
            _close_window(application, window)


def _initial_roi(experiment):
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


def _overlay_bounds(board: QtRasterBoard, selection: Selection | None):
    if selection is None:
        return None
    return board._selector_viewport.normalized_bounds_for_selection(selection)


def test_draft_rejection_preserves_old_run_then_apply_replaces_whole_generation(
    experiment,
    application,
    monkeypatch,
):
    initial_roi = _initial_roi(experiment)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        roi=initial_roi,
        roi_reduction=ReductionMethod.MEAN,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.has_front
                and window._live is not None
                and window._live.front_status is not None
                and window._live.front_status.sequence == board.front_frame.sequence
                and window._live.front_status.scalar_coverage is not None
                and window._live.front_status.scalar_coverage.written_cells >= 4
                and window._selector_switch.isEnabled()
            ),
        )
        old_handle = window._handle
        old_generation = window._run_owner.generation
        old_binding = window._running_binding
        old_slot = window._slot
        old_frame = board.front_frame
        assert old_handle is not None and old_binding is not None and old_slot is not None
        assert old_frame is not None
        old_raw_source = old_frame.panels[0].source_identity
        old_scalar_source = old_frame.panels[1].source_identity
        _run, _epoch, old_joined = old_slot.freeze_camera_current()
        assert board._selector_applied_bounds == _overlay_bounds(board, initial_roi)

        window._selector_switch.setChecked(True)
        target = board._selector_target()[0]
        start_point = QtCore.QPoint(
            target.left() + target.width() // 2,
            target.top() + target.height() // 2,
        )
        end_point = QtCore.QPoint(
            target.left() + 3 * target.width() // 4,
            target.top() + 3 * target.height() // 4,
        )
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=start_point)
        assert window._selector_interacting is True
        QtTest.QTest.mouseMove(board, end_point)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end_point)
        draft = window._draft_selection
        assert draft is not None and draft != initial_roi
        assert window._selector_interacting is False
        assert board._selector_draft_bounds == _overlay_bounds(board, draft)
        assert window._applied_request == request
        assert window._handle is old_handle

        real_prepare_view = camera_workbench._prepare_monitor_view

        def reject_downstream(command, generation):
            if command.request != request:
                raise MemoryError("synthetic downstream peak rejection")
            return real_prepare_view(command, generation)

        monkeypatch.setattr(
            camera_workbench,
            "_prepare_monitor_view",
            reject_downstream,
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
        assert window._applied_request == request
        assert window._draft_selection == draft
        assert board._selector_applied_bounds == _overlay_bounds(board, initial_roi)
        assert board._selector_draft_bounds == _overlay_bounds(board, draft)
        _run, _epoch, after_reject = old_slot.freeze_camera_current()
        assert after_reject.raw.ref.block_id == old_joined.raw.ref.block_id
        assert after_reject.raw.ref.stream_generation == old_joined.raw.ref.stream_generation
        assert after_reject.scalar.ref.block_id == old_joined.scalar.ref.block_id
        assert (
            after_reject.scalar.ref.stream_generation
            == old_joined.scalar.ref.stream_generation
        )

        monkeypatch.setattr(
            camera_workbench,
            "_prepare_monitor_view",
            real_prepare_view,
        )
        max_index = window._reducer_combo.findData(ReductionMethod.MAX)
        assert max_index >= 0
        window._reducer_combo.setCurrentIndex(max_index)
        window._apply_roi_button.click()
        _until(
            application,
            lambda: (
                window._apply_phase is None
                and window._applied_request.roi == draft
                and window._applied_request.roi_reduction is ReductionMethod.MAX
                and window._handle is not None
                and window._handle is not old_handle
                and board.has_front
                and window._live is not None
                and window._live.front_status is not None
                and window._live.front_status.sequence == board.front_frame.sequence
                and window._live.front_status.scalar_binding_fingerprint
                == window._running_binding.fingerprint
            ),
            timeout=20.0,
        )
        assert old_handle.snapshot().state is RunState.CANCELLED
        assert window._run_owner.generation == old_generation + 1
        assert window._running_binding.input_contract.fingerprint == (
            old_binding.input_contract.fingerprint
        )
        assert window._running_binding.fingerprint != old_binding.fingerprint
        new_frame = board.front_frame
        assert new_frame.panels[0].source_identity != old_raw_source
        assert new_frame.panels[1].source_identity != old_scalar_source
        assert all(
            panel.source_identity == new_frame.panels[1].source_identity
            for panel in new_frame.panels[1:]
        )
        assert board._selector_applied_bounds == _overlay_bounds(board, draft)
        assert board._selector_draft_bounds is None
        _run, _epoch, joined = window._slot.freeze_camera_current()
        assert joined.scalar is not None
        scalar_metadata = tuple(
            metadata
            for metadata in joined.scalar.cell_metadata
            if metadata is not None
        )
        assert scalar_metadata
        assert all(
            isinstance(metadata, RoiScalarMetadata)
            and metadata.binding_fingerprint == window._running_binding.fingerprint
            for metadata in scalar_metadata
        )
        assert joined.raw.ref.block_id != old_joined.raw.ref.block_id
        assert joined.scalar.ref.block_id != old_joined.scalar.ref.block_id
    finally:
        if window.isVisible():
            _close_window(application, window)
