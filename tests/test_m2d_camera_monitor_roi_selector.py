"""M2d front-bound ROI selection and immutable monitor replacement oracles."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
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
    StreamGenerationId,
    ValidityContract,
    ValidityPolicy,
    Value,
    ValueSchema,
)
from zlc_frontend.figure import DatasetId
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
    RoiScalarStreamProjection,
    reduce_camera_roi,
)
from zlc_neutral_atom.monitor_application import (
    CameraMonitorLiveDataset,
    CameraMonitorRoiState,
)
from zlc_neutral_atom.runtime.control import ControlAckStatus, create_control_topic
import zlc_neutral_atom.runtime.dataset as dataset_runtime
from zlc_neutral_atom.runtime.run import RunState
from zlc_storage import canonical_digest
import zlc_workbench.live as live_module
from zlc_workbench.live import LiveBoardController, LiveDatasetSlot


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


def _wheel_down(board: QtRasterBoard, target: QtCore.QRectF) -> QtGui.QWheelEvent:
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


def test_sticky_presentation_freeze_ends_a_candidate_waiting_for_gui_admission():
    controller = object.__new__(LiveBoardController)
    controller._owner_thread = threading.get_ident()
    controller._lock = threading.Lock()
    controller._closed = False
    controller._presentation_frozen = False
    controller._dirty = True
    controller._active = True
    controller._candidate = object()
    controller._port = object()
    controller._sources = (object(),)
    controller._board = SimpleNamespace(freeze_front=lambda: None)
    controller.freeze_presentation()
    assert controller._presentation_frozen is True
    assert controller._dirty is False
    assert controller._candidate is None
    assert controller._active is False
    assert controller._port is None and controller._sources is None


class _NarrowCameraDataset(CameraMonitorLiveDataset):
    def __init__(self) -> None:
        self.calls = []
        self.state = CameraMonitorRoiState(None, None, None, None, None)
        topic, _consumer = create_control_topic(lambda value: value)
        self.receipt = topic.publish(None)

    def prepare_roi_control(
        self,
        selection,
        reduction=ReductionMethod.MEAN,
        validity_policy=ValidityPolicy.REQUIRE_ALL,
    ):
        self.calls.append(("prepare", selection, reduction, validity_policy))
        return None

    def submit_roi_control(self, candidate):
        self.calls.append(("submit", candidate))
        return self.receipt

    def current_roi_state(self):
        self.calls.append(("state",))
        return self.state

    @property
    def initial_roi_receipt(self):
        self.calls.append(("initial",))
        return self.receipt


def _bound_control_slot(dataset):
    slot = object.__new__(LiveDatasetSlot)
    slot._lock = threading.Lock()
    slot._closed = False
    slot._dataset = dataset
    return slot


def test_live_slot_forwards_only_the_narrow_camera_control_surface():
    dataset = _NarrowCameraDataset()
    slot = _bound_control_slot(dataset)
    assert (
        slot.prepare_camera_roi_control(
            None,
            ReductionMethod.MAX,
            ValidityPolicy.OMIT_INVALID,
        )
        is None
    )
    assert slot.submit_camera_roi_control(None) is dataset.receipt
    assert slot.current_camera_roi_state() is dataset.state
    assert slot.initial_camera_roi_receipt is dataset.receipt
    assert dataset.calls == [
        ("prepare", None, ReductionMethod.MAX, ValidityPolicy.OMIT_INVALID),
        ("submit", None),
        ("state",),
        ("initial",),
    ]


def test_live_view_fault_isolated_from_source_and_visible_status():
    class SourceSlot:
        def fail(self, _message):  # pragma: no cover - forbidden path
            raise AssertionError("downstream view fault reached the source")

    wakes = []
    visible_status = object()
    controller = object.__new__(LiveBoardController)
    controller._lock = threading.Lock()
    controller._closed = False
    controller._fault = None
    controller._candidate = object()
    controller._active = True
    controller._dirty = True
    controller._port = object()
    controller._sources = (object(),)
    controller._front_status = visible_status
    controller._slot = SourceSlot()
    revoked = []

    def revoke(before_revoke=None):
        if before_revoke is not None and not before_revoke():
            return False
        revoked.append(True)
        return True

    controller._board = SimpleNamespace(
        revoke_pending_publication=revoke
    )
    controller._request_owner_wake = lambda: wakes.append(None)
    controller._set_view_fault(ValueError("scalar renderer failed"))
    assert controller.fault is not None
    assert controller._front_status is visible_status
    assert controller._candidate is None
    assert not controller._active and not controller._dirty
    assert controller._port is None and controller._sources is None
    assert revoked == [True]
    assert wakes == [None]


def test_old_live_render_job_cannot_touch_a_new_configuration():
    old_configuration = object()
    job = SimpleNamespace(
        candidate=SimpleNamespace(configuration=old_configuration),
        port=object(),
    )
    controller = object.__new__(LiveBoardController)
    controller._lock = threading.Lock()
    controller._closed = False
    controller._fault = None
    controller._configuration = object()
    controller._port = object()
    controller._evaluator = pytest.fail
    controller._render(job)
    assert controller._fault is None
    revoked = []

    def revoke(before_revoke=None):
        if before_revoke is not None and not before_revoke():
            return False
        revoked.append(True)
        return True

    controller._board = SimpleNamespace(
        revoke_pending_publication=revoke
    )
    controller._request_owner_wake = pytest.fail
    assert not controller._set_view_fault(
        RuntimeError("stale render failed after reconfigure"),
        expected_job=job,
    )
    assert controller._fault is None and not revoked


def test_scalar_dataset_id_and_stream_generation_are_a_stable_pair():
    controller = object.__new__(LiveBoardController)
    controller._scalar_dataset_generations = {}
    controller._scalar_generation_datasets = {}
    dataset_id = DatasetId("scalar-generation-a")
    generation = StreamGenerationId("generation-a")
    controller._claim_scalar_dataset_identity(dataset_id, generation)
    controller._claim_scalar_dataset_identity(dataset_id, generation)
    with pytest.raises(RuntimeError, match="changed stream generation"):
        controller._claim_scalar_dataset_identity(
            dataset_id,
            StreamGenerationId("generation-b"),
        )
    with pytest.raises(RuntimeError, match="frontend DatasetId"):
        controller._claim_scalar_dataset_identity(
            DatasetId("another-dataset"),
            generation,
        )


def test_roi_control_never_calls_source_cancel_and_manual_stop_remains_source_owned(
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
        stop = window.findChild(QtWidgets.QPushButton, "stopButton")
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.has_front
                and window._live is not None
                and window._live.front_status is not None
                and window._live.front_status.scalar_control_revision is not None
                and window._selector_switch.isEnabled()
            ),
        )
        handle = window._handle
        slot = window._slot
        source_generation = window._run_owner.generation
        assert handle is not None and slot is not None
        _run, _epoch, before = slot.freeze_camera_current()
        assert before.scalar is not None

        cancel_reasons = []
        real_cancel = handle.cancel

        def observe_cancel(reason="user requested stop"):
            cancel_reasons.append(reason)
            return real_cancel(reason)

        monkeypatch.setattr(handle, "cancel", observe_cancel)
        staged_layouts = []
        real_stage_layout = board.stage_layout

        def observe_stage_layout(*args, **kwargs):
            staged_layouts.append((args, kwargs))
            return real_stage_layout(*args, **kwargs)

        monkeypatch.setattr(board, "stage_layout", observe_stage_layout)
        old_layout_generation = window._board.model.layout_generation
        max_index = window._reducer_combo.findData(ReductionMethod.MAX)
        assert max_index >= 0
        window._reducer_combo.setCurrentIndex(max_index)
        _until(
            application,
            lambda: (
                window._applied_request.roi_reduction is ReductionMethod.MAX
                and window._live.front_status is not None
                and window._live.front_status.scalar_control_revision
                == window._applied_control_revision
                and window._live.front_status.scalar_binding_fingerprint
                == window._running_binding.fingerprint
            ),
            timeout=20.0,
        )
        assert cancel_reasons == []
        assert window._handle is handle
        assert window._slot is slot
        assert window._run_owner.generation == source_generation
        assert staged_layouts == []
        assert window._board.model.layout_generation == old_layout_generation
        _run, _epoch, after = slot.freeze_camera_current()
        assert after.raw.ref.block_id == before.raw.ref.block_id
        assert after.raw.ref.stream_generation == before.raw.ref.stream_generation
        assert after.scalar is not None
        assert after.scalar.ref.stream_generation != before.scalar.ref.stream_generation

        QtTest.QTest.mouseClick(stop, QtCore.Qt.LeftButton)
        _until(application, lambda: handle.snapshot().state is RunState.CANCELLED)
        assert cancel_reasons == ["Workbench user requested monitor stop"]
        assert window._run_owner.generation == source_generation
    finally:
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


def _raw_image_binding(board: QtRasterBoard):
    assert tuple(board._image_bindings) == ("camera-monitor-image",)
    return board._image_bindings["camera-monitor-image"]


def _overlay_bounds(board: QtRasterBoard, selection: Selection | None):
    if selection is None:
        return None
    return _raw_image_binding(board).viewport.normalized_bounds_for_selection(selection)


def test_rectangle_release_hot_applies_same_schema_and_rejection_keeps_old_branch(
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
        live = window._live
        old_frame = board.front_frame
        assert (
            old_handle is not None
            and old_binding is not None
            and old_slot is not None
            and live is not None
        )
        assert old_frame is not None
        old_raw_source = old_frame.panels[0].source_identity
        old_scalar_source = old_frame.panels[1].source_identity
        _run, _epoch, old_joined = old_slot.freeze_camera_current()
        assert old_joined.scalar is not None
        old_control_revision = window._live.front_status.scalar_control_revision
        source_taps = tuple(
            sorted(id(tap) for tap in old_slot._dataset.raw._source._monitors)
        )
        old_visible_projection = window._visible_projection_text
        assert old_visible_projection is not None
        assert _raw_image_binding(board).applied_bounds == _overlay_bounds(
            board, initial_roi
        )
        _until(
            application,
            lambda: (
                live._curve_y_limits is not None
                and live._curve_relim_mode is not None
            ),
        )
        curve_origin = board.visible_curve_origin()
        curve_payload = board.visible_curve_payload()
        assert curve_origin is not None and curve_payload is not None
        x_low, x_high = curve_payload.viewport.x_limits
        selected_span = (
            x_low + 0.25 * (x_high - x_low),
            x_low + 0.75 * (x_high - x_low),
        )
        window._set_curve_range_candidate(curve_origin, selected_span)
        assert window._curve_range_candidate == (curve_origin, selected_span)
        assert (
            board._numeric_bindings["camera-monitor-roi-curve"].applied_span
            == selected_span
        )

        reconfigure_observations = []
        real_reconfigure_scalar = live.reconfigure_scalar

        def observe_reconfigure_scalar(*args, **kwargs):
            # Exclude a worker completion from refilling the new baseline
            # before this exact post-transaction fact is observed.
            with live._worker_gate:
                previous = live._configuration
                result = real_reconfigure_scalar(*args, **kwargs)
                current = live._configuration
                with live._lock:
                    accepted_range = live._curve_y_limits
                    accepted_mode = live._curve_relim_mode
                reconfigure_observations.append(
                    (previous, current, accepted_range, accepted_mode)
                )
                return result

        monkeypatch.setattr(live, "reconfigure_scalar", observe_reconfigure_scalar)

        receipts = []
        submitted = []
        real_submit = old_slot.submit_camera_roi_control

        def observe_submit(candidate):
            receipt = real_submit(candidate)
            submitted.append(candidate)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(old_slot, "submit_camera_roi_control", observe_submit)
        clear_calls = []
        real_clear = board.clear

        def observe_clear(*args, **kwargs):
            clear_calls.append((args, kwargs))
            return real_clear(*args, **kwargs)

        monkeypatch.setattr(board, "clear", observe_clear)
        staged_layouts = []
        real_stage_layout = board.stage_layout

        def observe_stage_layout(*args, **kwargs):
            staged_layouts.append((args, kwargs))
            return real_stage_layout(*args, **kwargs)

        monkeypatch.setattr(board, "stage_layout", observe_stage_layout)
        old_layout_generation = window._board.model.layout_generation

        retained_front = board.front_frame
        assert retained_front is not None
        old_board_sequence = retained_front.sequence
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
        held_pixels = board._selector_hold.prepared[0]
        _until(
            application,
            lambda: (
                board.front_frame.sequence > old_board_sequence
                and window._live.front_status is not None
                and window._live.front_status.sequence == board.front_frame.sequence
            ),
            timeout=20.0,
        )
        hold = board._selector_hold
        assert hold is not None
        assert hold.panel_id == "camera-monitor-image"
        assert hold.sequence == old_board_sequence
        assert board.front_frame.sequence > hold.sequence
        assert hold.prepared[0] is held_pixels
        QtTest.QTest.mouseMove(board, end_point)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=end_point)
        assert len(receipts) == 1 and len(submitted) == 1
        draft = submitted[0].selection
        assert draft != initial_roi
        assert window._handle is old_handle
        _until(
            application,
            lambda: (
                receipts[0].snapshot().status.value == "APPLIED"
                and window._applied_request.roi == draft
            ),
            timeout=20.0,
        )
        assert board.front_frame.sequence > old_board_sequence
        assert board._selector_hold is None
        assert _raw_image_binding(board).applied_bounds == _overlay_bounds(
            board, initial_roi
        )
        assert _raw_image_binding(board).draft_bounds == _overlay_bounds(board, draft)
        assert window._projection_status.text().startswith(
            f"Display: {old_visible_projection} · target "
        )
        assert window._projection_status.text().endswith(" pending")
        assert window._curve_range_candidate is None
        assert (
            board._numeric_bindings["camera-monitor-roi-curve"].applied_span
            is None
        )
        assert not window._visible_curve_matches_current_state()
        assert not window._edit_curve_display.isEnabled()
        assert window._edit_image_display.isEnabled()
        assert window._selector_switch.isChecked()
        assert window._selector_switch.isEnabled()
        assert board._image_interaction_armed(_raw_image_binding(board))
        assert not board._numeric_interaction_armed(
            board._numeric_bindings["camera-monitor-roi-curve"]
        )
        stale_curve_viewport = board.visible_curve_payload().viewport
        curve_binding = board._numeric_bindings["camera-monitor-roi-curve"]
        curve_target = board._numeric_target(curve_binding)
        assert curve_target is not None
        stale_curve_event = _wheel_down(board, curve_target.plot)
        assert not stale_curve_event.isAccepted()
        assert board.visible_curve_payload().viewport == stale_curve_viewport
        assert board.curve_selector_fault is None
        assert (
            board._numeric_bindings["camera-monitor-roi-curve"].pending_viewport
            is None
        )
        _until(
            application,
            lambda: (
                receipts[0].snapshot().status.value == "APPLIED"
                and window._applied_request.roi == draft
                and window._live.front_status is not None
                and window._live.front_status.scalar_control_revision
                == receipts[0].snapshot().revision
                and board.front_frame.sequence > old_board_sequence
                and window._live.front_status.sequence == board.front_frame.sequence
                and window._draft_selection is None
            ),
            timeout=20.0,
        )
        assert window._handle is old_handle
        assert window._live is live
        assert not old_handle.snapshot().state.terminal
        assert window._run_owner.generation == old_generation
        assert window._slot is old_slot
        assert window._applied_request.roi == draft
        assert window._draft_selection is None
        assert clear_calls == []
        assert staged_layouts == []
        assert window._board.model.layout_generation == old_layout_generation
        assert _raw_image_binding(board).applied_bounds == _overlay_bounds(board, draft)
        assert _raw_image_binding(board).draft_bounds is None
        _run, _epoch, after_apply = old_slot.freeze_camera_current()
        assert after_apply.scalar is not None
        assert after_apply.raw.ref.block_id == old_joined.raw.ref.block_id
        assert after_apply.raw.ref.stream_generation == old_joined.raw.ref.stream_generation
        assert after_apply.scalar.ref.block_id == old_joined.scalar.ref.block_id
        assert after_apply.scalar.ref.stream_generation == old_joined.scalar.ref.stream_generation
        assert after_apply.raw.head.sequence > old_joined.raw.head.sequence
        assert tuple(
            sorted(id(tap) for tap in old_slot._dataset.raw._source._monitors)
        ) == source_taps
        scalar_metadata = tuple(
            metadata for metadata in after_apply.scalar.cell_metadata if metadata is not None
        )
        assert scalar_metadata
        assert all(
            isinstance(metadata, RoiScalarMetadata)
            and metadata.binding_fingerprint == window._running_binding.fingerprint
            and metadata.control_revision == receipts[0].snapshot().revision
            for metadata in scalar_metadata
        )
        assert receipts[0].snapshot().revision > old_control_revision
        assert len(reconfigure_observations) == 1
        (
            previous_configuration,
            current_configuration,
            reset_range,
            reset_mode,
        ) = reconfigure_observations[0]
        assert (
            live_module._scalar_semantic_identity(previous_configuration)
            != live_module._scalar_semantic_identity(current_configuration)
        )
        assert (
            previous_configuration.scalar_binding_fingerprint
            != current_configuration.scalar_binding_fingerprint
        )
        assert (
            current_configuration.scalar_binding_fingerprint
            == window._running_binding.fingerprint
        )
        assert (
            current_configuration.scalar_control_revision
            == receipts[0].snapshot().revision
        )
        assert (
            current_configuration.scalar_generation
            == old_joined.scalar.ref.stream_generation
        )
        assert current_configuration.scalar_block_id == old_joined.scalar.ref.block_id
        assert reset_range is None and reset_mode is None
        assert window._visible_curve_matches_current_state()
        assert window._edit_curve_display.isEnabled()
        assert board._numeric_interaction_armed(
            board._numeric_bindings["camera-monitor-roi-curve"]
        )
        assert board.curve_selector_fault is None
        new_frame = board.front_frame
        assert (
            new_frame.panels[0].coherence_stamp.join_key_digest
            != old_frame.panels[0].coherence_stamp.join_key_digest
        )
        assert new_frame.panels[0].source_identity == old_raw_source
        assert new_frame.panels[1].source_identity == old_scalar_source

        accepted_binding = window._running_binding
        accepted_request = window._applied_request
        accepted_revision = window._applied_control_revision
        accepted_scalar_ref = after_apply.scalar.ref
        real_write_cell = dataset_runtime._write_cell
        staged_write_failed = False

        def reject_after_shadow_write(
            cell,
            value,
            validity_mask,
            values,
            written,
            validity,
        ):
            nonlocal staged_write_failed
            real_write_cell(
                cell,
                value,
                validity_mask,
                values,
                written,
                validity,
            )
            if (
                not staged_write_failed
                and values.ndim == 2
                and int(np.count_nonzero(written)) == 1
            ):
                staged_write_failed = True
                raise ValueError("synthetic candidate shadow write rejection")

        monkeypatch.setattr(dataset_runtime, "_write_cell", reject_after_shadow_write)
        target = board._selector_target()[0]
        reject_start = QtCore.QPoint(target.left() + 2, target.top() + 2)
        reject_end = QtCore.QPoint(
            target.left() + target.width() // 3,
            target.top() + target.height() // 3,
        )
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=reject_start)
        QtTest.QTest.mouseMove(board, reject_end)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=reject_end)
        assert len(receipts) == 2
        _until(
            application,
            lambda: (
                receipts[1].snapshot().status.value == "REJECTED"
                and "REJECTED" in window._roi_status.text()
            ),
            timeout=20.0,
        )
        assert window._handle is old_handle
        assert not old_handle.snapshot().state.terminal
        assert window._run_owner.generation == old_generation
        assert window._slot is old_slot
        assert window._running_binding == accepted_binding
        assert window._applied_request == accepted_request
        assert window._applied_control_revision == accepted_revision
        assert staged_write_failed
        _run, _epoch, after_reject = old_slot.freeze_camera_current()
        assert after_reject.scalar is not None
        assert after_reject.raw.ref.block_id == old_joined.raw.ref.block_id
        assert after_reject.raw.head.sequence > after_apply.raw.head.sequence
        assert after_reject.scalar.ref == accepted_scalar_ref or (
            after_reject.scalar.ref.block_id == accepted_scalar_ref.block_id
            and after_reject.scalar.ref.stream_generation
            == accepted_scalar_ref.stream_generation
        )
        assert (
            after_reject.scalar.coverage.missed_events
            == after_apply.scalar.coverage.missed_events
        )
        assert not after_reject.scalar.coverage.current_gap
        retained_metadata = tuple(
            metadata
            for metadata in after_reject.scalar.cell_metadata
            if metadata is not None
        )
        assert retained_metadata
        assert all(
            isinstance(metadata, RoiScalarMetadata)
            and metadata.binding_fingerprint == accepted_binding.fingerprint
            and metadata.control_revision == accepted_revision
            for metadata in retained_metadata
        )
        assert clear_calls == []

        # A failure before the exclusive producer mutates its sequence is
        # retryable on the old binding for this same source shot.
        retryable_branch = old_slot._dataset._branch
        assert retryable_branch is not None
        retryable_sequence = retryable_branch.stream.next_sequence
        retryable_revision = receipts[-1].snapshot().revision + 1
        original_publish = RoiScalarStreamProjection.publish

        def fail_before_real_publish(self, update, payload, output):
            if payload.metadata.control_revision == retryable_revision:
                raise RuntimeError("synthetic failure before real ROI publication")
            return original_publish(self, update, payload, output)

        monkeypatch.setattr(
            RoiScalarStreamProjection,
            "publish",
            fail_before_real_publish,
        )
        retryable_selection = _raw_image_binding(
            board
        ).viewport.selection_for_normalized_bounds(
            (0.05, 0.05, 0.3, 0.3)
        )
        window._submit_roi_control(retryable_selection)
        assert len(receipts) == 3
        _until(
            application,
            lambda: (
                receipts[2].snapshot().status is ControlAckStatus.REJECTED
                and retryable_branch.stream.next_sequence
                == retryable_sequence + 1
            ),
            timeout=20.0,
        )
        assert old_slot._dataset._branch is retryable_branch
        assert retryable_branch.stream.next_sequence == retryable_sequence + 1
        retryable_state = old_slot.current_camera_roi_state()
        assert retryable_state.binding == accepted_binding
        _run, _epoch, after_retryable_failure = old_slot.freeze_camera_current()
        assert after_retryable_failure.scalar is not None
        assert not after_retryable_failure.scalar.coverage.current_gap
        assert (
            after_retryable_failure.scalar.coverage.missed_events
            == after_reject.scalar.coverage.missed_events
        )
        assert all(
            isinstance(metadata, RoiScalarMetadata)
            and metadata.binding_fingerprint == accepted_binding.fingerprint
            and metadata.control_revision == accepted_revision
            for metadata in after_retryable_failure.scalar.cell_metadata
            if metadata is not None
        )

        # A publisher wrapper may fail only after the real stream emit has
        # returned.  That consumed sequence is authoritative: the candidate is
        # rejected and the scalar generation is retired, never followed by an
        # old-binding fallback event at N+2.
        failed_branch = old_slot._dataset._branch
        assert failed_branch is not None
        failed_stream = failed_branch.stream
        sequence_before_failure = failed_stream.next_sequence
        target_revision = receipts[-1].snapshot().revision + 1
        raised_after_publish = False

        def publish_then_raise(self, update, payload, output):
            nonlocal raised_after_publish
            envelope = original_publish(self, update, payload, output)
            if (
                not raised_after_publish
                and payload.metadata.control_revision == target_revision
            ):
                raised_after_publish = True
                raise RuntimeError("synthetic failure after real ROI publication")
            return envelope

        monkeypatch.setattr(
            RoiScalarStreamProjection,
            "publish",
            publish_then_raise,
        )
        committed_failure_selection = _raw_image_binding(
            board
        ).viewport.selection_for_normalized_bounds(
            (0.2, 0.2, 0.45, 0.45)
        )
        window._submit_roi_control(committed_failure_selection)
        assert len(receipts) == 4
        assert receipts[3].snapshot().revision == target_revision
        _until(
            application,
            lambda: (
                receipts[3].snapshot().status is ControlAckStatus.REJECTED
                and old_slot.current_camera_roi_state().failure_reason is not None
                and old_slot._dataset._branch is None
            ),
            timeout=20.0,
        )
        failed_state = old_slot.current_camera_roi_state()
        assert raised_after_publish
        assert failed_stream.next_sequence == sequence_before_failure + 1
        assert "authoritative sequence advance" in (failed_state.failure_reason or "")
        assert "synthetic failure after real ROI publication" in (
            failed_state.failure_reason or ""
        )
        assert window._handle is old_handle
        assert not old_handle.snapshot().state.terminal

        def raw_advanced_after_committed_publish_failure():
            _run, _epoch, current = old_slot.freeze_camera_current()
            return (
                current.scalar is None
                and current.raw.head is not None
                and after_reject.raw.head is not None
                and current.raw.head.sequence > after_reject.raw.head.sequence
            )

        _until(
            application,
            raw_advanced_after_committed_publish_failure,
            timeout=20.0,
        )
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_applied_receipt_drain_keeps_concurrent_branch_failure_reason(
    experiment,
    application,
    monkeypatch,
):
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
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
            lambda: board.has_front and window._selector_switch.isEnabled(),
        )
        slot = window._slot
        handle = window._handle
        assert slot is not None and handle is not None
        _run, _epoch, before = slot.freeze_camera_current()

        real_drain = window._drain_roi_controls
        monkeypatch.setattr(window, "_drain_roi_controls", lambda: None)
        selection = _initial_roi(experiment)
        window._submit_roi_control(selection)
        assert len(window._roi_controls) == 1
        revision, pending = next(iter(window._roi_controls.items()))
        ack = pending.receipt.wait(10.0)
        assert ack.status is ControlAckStatus.APPLIED
        assert ack.revision == revision

        state = slot.current_camera_roi_state()
        assert state.binding is not None
        active_fingerprint = state.binding.fingerprint
        real_project = RoiScalarStreamProjection.project

        def fail_applied_branch(self, update, binding, control_revision):
            if binding.fingerprint == active_fingerprint:
                raise ValueError("synthetic post-APPLIED scalar failure")
            return real_project(self, update, binding, control_revision)

        monkeypatch.setattr(
            RoiScalarStreamProjection,
            "project",
            fail_applied_branch,
        )
        _until(
            application,
            lambda: slot.current_camera_roi_state().failure_reason is not None,
            timeout=20.0,
        )
        def fail_failure_presentation(*_args, **_kwargs):
            raise RuntimeError("synthetic failure presentation collapse")

        monkeypatch.setattr(
            window,
            "_adopt_roi_presentation",
            fail_failure_presentation,
        )
        monkeypatch.setattr(window, "_drain_roi_controls", real_drain)
        real_drain()

        assert revision not in window._roi_controls
        assert window._applied_request.roi is None
        assert window._draft_selection == selection
        assert "synthetic post-APPLIED scalar failure" in window._roi_status.text()
        assert "synthetic failure presentation collapse" in (
            window._roi_presentation_failure or ""
        )
        assert "synthetic post-APPLIED scalar failure" in (
            window._roi_presentation_failure or ""
        )
        assert "raw source continues" in window._roi_status.text()
        assert not handle.snapshot().state.terminal

        def raw_source_advanced():
            _run, _epoch, after = slot.freeze_camera_current()
            return after.raw.head.sequence > before.raw.head.sequence

        _until(application, raw_source_advanced)
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_stop_after_same_schema_commit_keeps_the_revision_applied(
    experiment,
    application,
    monkeypatch,
):
    initial_roi = _initial_roi(experiment)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        roi=initial_roi,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    commit_returned = threading.Event()
    release_commit = threading.Event()
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.has_front
                and window._slot is not None
                and window._running_binding is not None
                and window._selector_switch.isEnabled()
            ),
        )
        slot = window._slot
        handle = window._handle
        branch = slot._dataset._branch
        assert slot is not None and handle is not None and branch is not None
        target_dataset = branch.dataset
        receipts = []
        real_submit = slot.submit_camera_roi_control

        def observe_submit(candidate):
            receipt = real_submit(candidate)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(slot, "submit_camera_roi_control", observe_submit)
        real_commit = dataset_runtime.MonitorDataset.commit_append_replacement

        def hold_after_irreversible_commit(
            self,
            replacement,
            envelope,
            *,
            timeout=0.0,
        ):
            result = real_commit(
                self,
                replacement,
                envelope,
                timeout=timeout,
            )
            if self is target_dataset:
                commit_returned.set()
                if not release_commit.wait(10.0):
                    raise TimeoutError("test did not release committed ROI replacement")
            return result

        monkeypatch.setattr(
            dataset_runtime.MonitorDataset,
            "commit_append_replacement",
            hold_after_irreversible_commit,
        )
        target_selection = _raw_image_binding(
            board
        ).viewport.selection_for_normalized_bounds(
            (0.15, 0.15, 0.55, 0.55)
        )
        window._submit_roi_control(target_selection)
        assert len(receipts) == 1
        assert commit_returned.wait(10.0)

        window._cancel_monitor()
        assert window._manual_stop_requested
        release_commit.set()
        ack = receipts[0].wait(10.0)
        assert ack.status is ControlAckStatus.APPLIED
        assert handle.wait(10.0).state is RunState.CANCELLED
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
        assert receipts[0].snapshot().status is ControlAckStatus.APPLIED
        assert window._request.roi == target_selection
        assert window._applied_request == window._request
    finally:
        release_commit.set()
        if window.isVisible():
            _close_window(application, window)


def test_spontaneous_roi_branch_failure_falls_back_to_raw_without_source_gap(
    experiment,
    application,
    monkeypatch,
):
    initial_roi = _initial_roi(experiment)
    request = experiment.readout.camera_monitor_request(
        history_capacity=3,
        roi=initial_roi,
        scalar_history_capacity=12,
        memory_limit_bytes=1 << 30,
    )
    window = experiment.readout.camera_monitor_gui(request)
    try:
        real_drain = window._drain_roi_controls
        monkeypatch.setattr(window, "_drain_roi_controls", lambda: None)
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
                and window._live.front_status.scalar_control_revision is not None
                and window._selector_switch.isEnabled()
            ),
        )
        handle = window._handle
        slot = window._slot
        live = window._live
        old_front = board.front_frame
        old_generation = window._run_owner.generation
        old_binding = window._running_binding
        assert (
            handle is not None
            and slot is not None
            and live is not None
            and old_front is not None
            and old_binding is not None
        )
        _run, _epoch, before = slot.freeze_camera_current()
        assert before.scalar is not None

        # Delay only the owner-side failure fold.  The producer keeps running;
        # adopting raw-only must retain the old coherent front until its first
        # replacement board arrives, without a reversible whole-live pause.
        real_project = RoiScalarStreamProjection.project

        def fail_applied_branch(self, update, binding, control_revision):
            if binding.fingerprint == old_binding.fingerprint:
                raise ValueError("synthetic applied branch failure")
            return real_project(self, update, binding, control_revision)

        monkeypatch.setattr(
            RoiScalarStreamProjection,
            "project",
            fail_applied_branch,
        )
        _until(
            application,
            lambda: slot.current_camera_roi_state().failure_reason is not None,
            timeout=20.0,
        )
        monkeypatch.setattr(window, "_drain_roi_controls", real_drain)
        real_drain()
        assert window._running_binding is None
        assert "raw source continues" in window._roi_status.text()
        assert window._handle is handle
        assert not handle.snapshot().state.terminal
        assert window._run_owner.generation == old_generation
        assert "synthetic applied branch failure" in window._roi_status.text()
        assert board.front_frame is old_front
        assert tuple(panel.panel_id for panel in board.front_frame.panels) == (
            "camera-monitor-image",
            "camera-monitor-roi-curve",
            "camera-monitor-roi-histogram",
            "camera-monitor-roi-meter",
        )
        window._refresh_view_status()
        assert "WAITING" in window._view_status.text()
        assert "previous front retained" in window._view_status.text()
        assert window._projection_text.startswith("latest raw frame")
        assert window._applied_request.roi is None
        assert window._draft_selection == initial_roi
        _run, _epoch, after_failure = slot.freeze_camera_current()
        assert after_failure.scalar is None
        assert after_failure.raw.ref.block_id == before.raw.ref.block_id
        assert (
            after_failure.raw.ref.stream_generation
            == before.raw.ref.stream_generation
        )
        assert after_failure.raw.head.sequence > before.raw.head.sequence

        _until(
            application,
            lambda: (
                board.front_frame is not None
                and tuple(panel.panel_id for panel in board.front_frame.panels)
                == ("camera-monitor-image",)
                and live.front_status is not None
                and live.front_status.sequence == board.front_frame.sequence
                and live.front_status.scalar_control_revision is None
            ),
            timeout=20.0,
        )
        assert window._selector_switch.isEnabled()
        assert window._reducer_combo.isEnabled()
        assert not window._clear_roi_button.isEnabled()

        # A local renderer fault disables every ROI mutation surface but does
        # not propagate to the source Run or its raw generation.
        live._set_view_fault(ValueError("synthetic detached view fault"))
        _until(
            application,
            lambda: (
                not window._selector_switch.isEnabled()
                and not window._reducer_combo.isEnabled()
                and not window._clear_roi_button.isEnabled()
            ),
        )
        assert window._handle is handle
        assert not handle.snapshot().state.terminal
        _run, _epoch, after_view_fault = slot.freeze_camera_current()
        assert after_view_fault.raw.ref.block_id == before.raw.ref.block_id
        assert (
            after_view_fault.raw.ref.stream_generation
            == before.raw.ref.stream_generation
        )
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_broken_view_notification_faults_only_presentation_while_raw_ingest_continues(
    experiment,
    application,
    monkeypatch,
):
    request = experiment.readout.camera_monitor_request(
        history_capacity=4,
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
                and window._slot is not None
                and window._selector_switch.isEnabled()
            ),
        )
        handle = window._handle
        slot = window._slot
        live = window._live
        old_generation = window._run_owner.generation
        assert handle is not None and slot is not None and live is not None
        _run, _epoch, before = slot.freeze_camera_current()
        coherent_front = board.front_frame
        assert coherent_front is not None
        calls = []

        def broken_notification():
            calls.append(None)
            raise RuntimeError("synthetic view notification failure")

        monkeypatch.setattr(slot, "updated", broken_notification)
        _until(
            application,
            lambda: (
                slot.notification_failure is not None
                and live.fault is not None
                and not window._selector_switch.isEnabled()
                and not window._reducer_combo.isEnabled()
                and not window._clear_roi_button.isEnabled()
            ),
            timeout=20.0,
        )
        fault_front = board.front_frame
        assert fault_front is not None
        assert fault_front.sequence >= coherent_front.sequence

        def raw_advanced_after_view_fault():
            _run, _epoch, current = slot.freeze_camera_current()
            return (
                current.raw.head is not None
                and before.raw.head is not None
                and current.raw.head.sequence >= before.raw.head.sequence + 2
            )

        _until(application, raw_advanced_after_view_fault, timeout=20.0)
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert calls == [None]
        assert slot.failure is None
        assert "synthetic view notification failure" in (
            slot.notification_failure or ""
        )
        assert window._handle is handle
        assert not handle.snapshot().state.terminal
        assert window._run_owner.generation == old_generation
        assert board.front_frame is fault_front
    finally:
        if window.isVisible():
            _close_window(application, window)
