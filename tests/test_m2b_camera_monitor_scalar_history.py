"""M2b typed ROI scalar events, independent history, and joined board oracles."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import (
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
    Value,
    ValueSchema,
    ValidityPolicy,
    expand_dataset_validity,
    resolve_selection_indices,
)
from zlc_frontend.qt_widgets import QtRasterBoard
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
    RoiScalarSampleContract,
    RoiScalarStreamProjection,
    reduce_camera_roi,
)
from zlc_neutral_atom.runtime.dataset import (
    dataset_storage_nbytes,
    mutable_dataset_storage_nbytes,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ProducerFlowControl,
    StreamId,
    TraceContext,
)


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    exp = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("m2b-camera-monitor"),
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


def _typed_roi(experiment):
    raw = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=4)
    ).output_schema
    y_axis, x_axis = raw.cell_schema.data_axes
    roi = Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        x_axis.coordinates[0],
        x_axis.coordinates[min(15, x_axis.size - 1)],
        y_axis.coordinates[0],
        y_axis.coordinates[min(11, y_axis.size - 1)],
        coordinate_frame=x_axis.coordinate_frame,
    )
    return roi, y_axis, x_axis


def test_roi_scalar_history_is_independent_and_board_join_is_source_exact(
    experiment,
    application,
):
    roi, y_axis, x_axis = _typed_roi(experiment)
    request = experiment.readout.camera_monitor_request(
        history_capacity=4,
        roi=roi,
        roi_reduction=ReductionMethod.SUM,
        scalar_history_capacity=24,
    )
    descriptor = experiment.readout.inspect_camera_monitor(request)
    scalar_schema = descriptor.scalar_output_schema
    assert descriptor.output_shape == (1, 4, y_axis.size, x_axis.size)
    assert scalar_schema is not None
    assert scalar_schema.physical_shape == (1, 24)
    assert scalar_schema.cell_schema.data_axes == ()

    window = experiment.readout.camera_monitor_gui(request)
    try:
        start = window.findChild(QtWidgets.QPushButton, "startButton")
        board = window.findChild(QtRasterBoard, "cameraMonitorImageBoard")
        assert board is not None
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: (
                board.has_front
                and window._live is not None
                and window._live.scalar_coverage is not None
                and window._live.scalar_coverage.written_cells >= 4
            ),
        )

        frame = board.front_frame
        assert frame is not None and len(frame.panels) == 2
        image_panel, curve_panel = frame.panels
        assert image_panel.source_identity != curve_panel.source_identity
        assert image_panel.coherence_stamp == curve_panel.coherence_stamp
        stamp = image_panel.coherence_stamp
        assert stamp.join_key_type == "camera-roi-source-event-control"
        assert len(stamp.inputs) == 2

        _run, _epoch, joined = window._slot.freeze_camera_current()
        raw = joined.raw
        scalar = joined.scalar
        metadata = joined.scalar_metadata
        assert scalar is not None and isinstance(metadata, RoiScalarMetadata)
        assert raw.block.values.shape == descriptor.output_schema.physical_shape
        assert scalar.block.values.shape == scalar_schema.physical_shape
        assert len(raw.event_refs) == request.history_capacity
        assert len(scalar.event_refs) == request.scalar_history_capacity
        assert metadata.source_event_ref == raw.head
        assert metadata.control_revision == 0
        assert metadata.control_fingerprint == window._slot.spec.roi_binding.fingerprint
        assert metadata.source_missed == 0

        terms = {term.axis_id: term for term in roi.terms}
        y_indices, _ = resolve_selection_indices(y_axis, terms[y_axis.axis_id])
        x_indices, _ = resolve_selection_indices(x_axis, terms[x_axis.axis_id])
        raw_values = raw.block.values[0, 0]
        raw_validity = expand_dataset_validity(
            raw.block.validity,
            raw.block.schema,
        )[0, 0]
        selected_values = raw_values[
            y_indices.start : y_indices.stop,
            x_indices.start : x_indices.stop,
        ]
        selected_validity = raw_validity[
            y_indices.start : y_indices.stop,
            x_indices.start : x_indices.stop,
        ]
        expected = selected_values[selected_validity].sum(dtype=np.uint64)
        assert scalar.block.values[0, 0] == expected
        assert bool(expand_dataset_validity(scalar.block.validity, scalar.block.schema)[0, 0])
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_roi_scalar_projection_omits_invalid_components_and_carries_lineage():
    frame = CoordinateFrameId("test-camera-pixels")
    y_axis = AxisSpec(AxisId("test.y"), "y", SPATIAL_Y, 3, (0, 1, 2), "px", frame)
    x_axis = AxisSpec(AxisId("test.x"), "x", SPATIAL_X, 4, (0, 1, 2, 3), "px", frame)
    schema = ValueSchema(
        (y_axis, x_axis),
        ValidityContract.components(y_axis.axis_id, x_axis.axis_id),
        np.dtype("<u2"),
        "camera-count",
    )
    contract = CameraSampleContract(schema)
    selection = Selection.rectangle(
        x_axis.axis_id,
        y_axis.axis_id,
        0,
        3,
        0,
        2,
        coordinate_frame=frame,
    )
    binding = RoiScalarBinding(
        contract,
        selection,
        ReductionMethod.MEAN,
        ValidityPolicy.OMIT_INVALID,
    )
    contributors = y_axis.size * x_axis.size
    assert binding.reduction_scratch_nbytes == contributors * (
        schema.dtype.itemsize
        + binding.output_schema.dtype.itemsize
        + np.dtype(bool).itemsize
    )
    output_contract = RoiScalarSampleContract(
        binding,
        RoiScalarMetadataContract(CameraFrameMetadataContract()),
    )
    source, source_producer = AcquisitionStream.create(
        StreamId("m2b-test-camera"),
        contract,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=1,
        retention_bytes=contract.max_retained_nbytes,
    )
    source_tap = source.monitor(
        max_events=1,
        max_bytes=contract.max_retained_nbytes,
    )
    output, output_producer = AcquisitionStream.create(
        StreamId("m2b-test-roi"),
        output_contract,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=1,
        retention_bytes=output_contract.max_retained_nbytes,
    )
    projection = RoiScalarStreamProjection(binding, source_tap, output_producer)
    metadata = CameraFrameMetadata(
        0,
        1,
        1,
        1,
        None,
        None,
        1,
        0,
        "m2b-correlation",
    )
    values = np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
    valid = np.ones((3, 4), dtype=bool)
    valid[1, 2] = False
    sample = CameraSample(
        Value(
            values,
            ComponentValidity((y_axis.axis_id, x_axis.axis_id), valid),
            schema,
        ),
        metadata,
    )
    source_envelope = source_producer.emit(
        sample,
        captured_at=metadata.captured_at,
        trace=TraceContext("m2b-run", "camera", metadata.correlation_id),
    )
    derived = projection.process_next(timeout=0.0)
    assert derived.trace.causation_refs == (source_envelope.ref,)
    assert derived.trace.control_revision == 0
    assert derived.trace.config_revision == 0
    assert derived.payload.metadata.source_event_ref == source_envelope.ref
    assert derived.payload.value.validity is VALID
    expected = values[valid].astype(np.float64).mean()
    assert derived.payload.value.values == pytest.approx(expected)
    require_all = RoiScalarBinding(contract, selection, ReductionMethod.MEAN)
    assert require_all.fingerprint != binding.fingerprint
    fail_closed = reduce_camera_roi(sample, require_all)
    assert fail_closed.validity is not VALID
    assert fail_closed.values == pytest.approx(expected)

    source_producer.finish()
    projection.finish()
    projection.close()
    assert output.next_sequence == 1


def test_scalar_history_memory_is_admitted_without_growing_raw_history(experiment):
    roi, _y_axis, _x_axis = _typed_roi(experiment)
    one = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(
            history_capacity=4,
            roi=roi,
            scalar_history_capacity=1,
        )
    )
    many = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(
            history_capacity=4,
            roi=roi,
            scalar_history_capacity=20,
        )
    )
    assert one.output_schema.physical_shape == many.output_schema.physical_shape
    assert one.scalar_output_schema is not None and many.scalar_output_schema is not None
    metadata_bytes = RoiScalarMetadataContract(
        CameraFrameMetadataContract()
    ).max_retained_nbytes
    expected_delta = (
        2 * mutable_dataset_storage_nbytes(many.scalar_output_schema)
        + dataset_storage_nbytes(many.scalar_output_schema)
        + 20 * metadata_bytes
        - mutable_dataset_storage_nbytes(one.scalar_output_schema)
        - dataset_storage_nbytes(one.scalar_output_schema)
        - metadata_bytes
    )
    assert many.base_peak_bytes - one.base_peak_bytes == expected_delta

    raw_only = experiment.readout.inspect_camera_monitor(
        experiment.readout.camera_monitor_request(history_capacity=4)
    )
    with pytest.raises(MemoryError, match="base peak"):
        experiment.readout.inspect_camera_monitor(
            experiment.readout.camera_monitor_request(
                history_capacity=4,
                roi=roi,
                scalar_history_capacity=10**7,
                memory_limit_bytes=raw_only.base_peak_bytes + 1,
            )
        )
