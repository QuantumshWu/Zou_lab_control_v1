"""DatasetBuilder contracts over exact and monitor stream deliveries."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DatasetSchema,
    PointLayout,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    StreamGenerationId,
    VALID,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetMode,
    DatasetPreviewSnapshot,
    DuplicateDatasetCell,
    MissingDatasetCells,
    SnapshotExpired,
    SealedDatasetArtifact,
    dataset_cell_key_fingerprint,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    StreamEndedEarly,
    StreamId,
    TraceContext,
    TraceBinding,
    ProducerFlowControl,
)


FINGERPRINT = "3" * 64
TRACE_BINDING = TraceBinding("run", "camera")


def axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def image_value_schema(*, component_validity: bool = False) -> ValueSchema:
    y = axis("camera.y", SPATIAL_Y, 2)
    x = axis("camera.x", SPATIAL_X, 3)
    validity = (
        ValidityContract.components(y.axis_id, x.axis_id)
        if component_validity
        else ValidityContract.value()
    )
    return ValueSchema((y, x), validity, np.dtype("<u2"), value_unit="count")


def dataset_schema(*, repeats: int = 1, points: int = 3, component_validity: bool = False):
    return DatasetSchema(
        axis("repeat", REPEAT, repeats),
        (axis("detuning", SCAN_POINT, points),),
        PointLayout.rect_c((points,)),
        image_value_schema(component_validity=component_validity),
    )


def value(number: int, *, component_validity=None) -> Value:
    schema = image_value_schema(component_validity=component_validity is not None)
    validity = VALID if component_validity is None else component_validity
    return Value(np.full((2, 3), number, dtype=np.uint16), validity, schema)


def cell_schedule(schema: DatasetSchema) -> tuple[DatasetCellAddress, ...]:
    return tuple(
        DatasetCellAddress(repeat, point)
        for repeat in range(schema.repeat_axis.size)
        for point in range(schema.point_layout.storage_size)
    )


def source(schema: DatasetSchema, *, events=8):
    contract = ValuePayloadContract(schema.cell_schema)
    return AcquisitionStream.create(
        StreamId("camera.frames"),
        contract,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=events,
        retention_bytes=events * contract.max_retained_nbytes,
        join_key_contract=DatasetCellKeyContract(schema),
    )


def emit(producer, payload, address: DatasetCellAddress, sequence_value: int):
    contract_schema = producer._stream._payload_contract.schema
    if payload.schema is not contract_schema:
        payload = Value(payload.values, payload.validity, contract_schema)
    return producer.emit(
        payload,
        captured_at=float(sequence_value),
        trace=TraceContext("run", "camera", "capture"),
        join_key=address,
    )


def test_exact_builder_preserves_all_named_data_axes_and_snapshot_revisions():
    schema = dataset_schema(repeats=1, points=3)
    stream, producer = source(schema, events=3)
    reservation = stream.reserve(
        total_events=3,
        max_inflight_events=3,
        max_inflight_bytes=36,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("capture"),
        reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        expected_cells=cell_schedule(schema),
    )

    emit(producer, value(10), DatasetCellAddress(0, 0), 0)
    progress = builder.consume(cursor.next())
    first_ref = progress.ref
    first = builder.materialize(first_ref)
    assert isinstance(first, DatasetPreviewSnapshot)
    assert first.block.values.shape == (1, 3, 2, 3)
    assert np.all(first.block.values[0, 0] == 10)
    assert progress.coverage.written_cells == 1
    assert not hasattr(progress, "patch") and not hasattr(progress, "block")

    for point, number in ((1, 20), (2, 30)):
        emit(producer, value(number), DatasetCellAddress(0, point), point)
        builder.consume(cursor.next())
    assert np.all(first.block.values[0, 0] == 10)
    assert np.all(first.block.values[0, 1:] == 0)
    with pytest.raises(SnapshotExpired):
        builder.materialize(first_ref)
    final = builder.seal(producer.finish())
    assert isinstance(final, SealedDatasetArtifact)
    assert final.block.values.shape == (1, 3, 2, 3)
    assert tuple(final.block.values[0, point, 0, 0] for point in range(3)) == (10, 20, 30)
    reservation.release()
    assert stream.retained_bytes == 0


def test_bound_builder_owns_reservation_completion():
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)
    reservation = stream.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=12,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("completion-owner"),
        reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        expected_cells=cell_schedule(schema),
    )
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    builder.consume(cursor.next())
    with pytest.raises(Exception, match="belongs to its bound DatasetBuilder"):
        reservation.complete()
    builder.seal(producer.finish())
    reservation.release()


def test_builder_context_preserves_body_error_and_releases_claim():
    schema = dataset_schema(points=1)
    stream, _producer = source(schema, events=1)
    reservation = stream.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=12,
        trace_binding=TRACE_BINDING,
    )
    reservation.activate()
    with pytest.raises(RuntimeError, match="body failure"):
        with DatasetBuilder(
            BlockId("context-owner"),
            reservation,
            schema,
            DatasetMode.FINITE_EXACT,
            expected_cells=cell_schedule(schema),
        ):
            raise RuntimeError("body failure")
    replacement = stream.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=12,
        trace_binding=TRACE_BINDING,
    )
    replacement.abort()
    replacement.release()


def test_exact_duplicate_and_missing_cells_fail_without_acknowledging_delivery():
    schema = dataset_schema(points=2)
    stream, producer = source(schema, events=2)
    reservation = stream.reserve(
        total_events=2,
        max_inflight_events=2,
        max_inflight_bytes=24,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("duplicate"),
        reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        expected_cells=cell_schedule(schema),
    )
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    builder.consume(cursor.next())
    emit(producer, value(2), DatasetCellAddress(0, 0), 1)
    with pytest.raises(DuplicateDatasetCell):
        builder.consume(cursor.next())
    assert cursor.next_sequence == 1
    builder.abort()
    reservation.release()

    missing_stream, missing_producer = source(schema, events=2)
    missing_reservation = missing_stream.reserve(
        total_events=2,
        max_inflight_events=2,
        max_inflight_bytes=24,
        trace_binding=TRACE_BINDING,
    )
    missing_cursor = missing_reservation.activate()
    missing_builder = DatasetBuilder(
        BlockId("missing"),
        missing_reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        expected_cells=cell_schedule(schema),
    )
    emit(missing_producer, value(1), DatasetCellAddress(0, 0), 0)
    missing_builder.consume(missing_cursor.next())
    with pytest.raises(StreamEndedEarly):
        missing_builder.seal(missing_producer.finish())
    missing_builder.abort()
    missing_reservation.release()


def test_component_validity_is_aligned_by_axis_id_not_trailing_shape_guess():
    schema = dataset_schema(points=1, component_validity=True)
    y_axis, x_axis = schema.cell_schema.data_axes
    component = ComponentValidity(
        (x_axis.axis_id,),
        np.array([True, False, True], dtype=bool),
    )
    stream, producer = source(schema, events=1)
    reservation = stream.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=18,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("component-validity"),
        reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        expected_cells=cell_schedule(schema),
    )
    emit(producer, value(7, component_validity=component), DatasetCellAddress(0, 0), 0)
    builder.consume(cursor.next())
    snapshot = builder.seal(producer.finish())
    assert snapshot.block.validity.axis_ids == (y_axis.axis_id, x_axis.axis_id)
    assert snapshot.block.validity.mask.shape == (1, 1, 2, 3)
    assert np.array_equal(
        snapshot.block.validity.mask[0, 0],
        np.array([[True, False, True], [True, False, True]]),
    )
    reservation.release()


def test_rolling_monitor_overwrite_expires_old_revision_without_formal_seal():
    schema = dataset_schema(points=2)
    stream, producer = source(schema, events=4)
    tap = stream.monitor(max_events=1, max_bytes=12)
    builder = DatasetBuilder(
        BlockId("rolling"),
        tap,
        schema,
        DatasetMode.ROLLING_MONITOR,
    )

    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    first = builder.ingest_monitor(tap.next()).ref
    emit(producer, value(2), DatasetCellAddress(0, 1), 1)
    builder.ingest_monitor(tap.next())
    emit(producer, value(3), DatasetCellAddress(0, 0), 2)
    builder.ingest_monitor(tap.next())

    with pytest.raises(SnapshotExpired):
        builder.materialize(first)
    current = builder.materialize()
    assert current.block.values[0, 0, 0, 0] == 3
    assert current.block.values[0, 1, 0, 0] == 2
    with pytest.raises(Exception, match="cannot become formal"):
        builder.seal(producer.finish())
    tap.close()


def test_exact_delivery_cannot_cross_source_authority_even_when_ids_match():
    schema = dataset_schema(points=1)
    stream_a, producer_a = source(schema, events=1)
    stream_b, producer_b = source(schema, events=1)
    reservation_a = stream_a.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=12,
        trace_binding=TRACE_BINDING,
    )
    reservation_b = stream_b.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=12,
        trace_binding=TRACE_BINDING,
    )
    cursor_a = reservation_a.activate()
    cursor_b = reservation_b.activate()
    builder = DatasetBuilder(
        BlockId("authority"),
        reservation_a,
        schema,
        DatasetMode.FINITE_EXACT,
        expected_cells=cell_schedule(schema),
    )
    emit(producer_a, value(1), DatasetCellAddress(0, 0), 0)
    emit(producer_b, value(999), DatasetCellAddress(0, 0), 0)
    with pytest.raises(PermissionError, match="another exact reservation"):
        builder.consume(cursor_b.next())
    assert builder.revision.value == 0
    builder.consume(cursor_a.next())
    sealed = builder.seal(producer_a.finish())
    assert sealed.block.values[0, 0, 0, 0] == 1
    reservation_a.release()
    reservation_b.abort()
    reservation_b.release()


def test_exact_join_key_must_match_the_frozen_plan_schedule():
    schema = dataset_schema(points=2)
    stream, producer = source(schema, events=2)
    reservation = stream.reserve(
        total_events=2,
        max_inflight_events=2,
        max_inflight_bytes=24,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("schedule"),
        reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        expected_cells=cell_schedule(schema),
    )
    emit(producer, value(10), DatasetCellAddress(0, 1), 0)
    delivery = cursor.next()
    with pytest.raises(Exception, match="frozen plan key"):
        builder.consume(delivery)
    assert not delivery.acknowledged
    assert builder.revision.value == 0
    builder.abort()
    reservation.release()


def test_exact_reservation_rejects_cross_run_trace_mixing():
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)
    reservation = stream.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=12,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("trace-binding"),
        reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        expected_cells=cell_schedule(schema),
    )
    payload = Value(value(7).values, VALID, schema.cell_schema)
    producer.emit(
        payload,
        captured_at=0.0,
        trace=TraceContext("another-run", "camera", "capture"),
        join_key=DatasetCellAddress(0, 0),
    )
    delivery = cursor.next()
    with pytest.raises(Exception, match="reserved formal run/source"):
        builder.consume(delivery)
    assert builder.revision.value == 0
    assert not delivery.acknowledged
    builder.abort()
    reservation.release()


def test_monitor_update_cannot_cross_tap_or_source_authority():
    schema = dataset_schema(points=1)
    stream_a, _producer_a = source(schema, events=1)
    stream_b, producer_b = source(schema, events=1)
    tap_a = stream_a.monitor(max_events=1, max_bytes=12)
    tap_b = stream_b.monitor(max_events=1, max_bytes=12)
    builder = DatasetBuilder(
        BlockId("monitor-authority"), tap_a, schema, DatasetMode.ROLLING_MONITOR
    )
    emit(producer_b, value(9), DatasetCellAddress(0, 0), 0)
    with pytest.raises(PermissionError, match="another monitor"):
        builder.ingest_monitor(tap_b.next())
    assert builder.revision.value == 0
    tap_a.close()
    tap_b.close()


def test_value_payload_contract_counts_component_validity_bytes():
    schema = dataset_schema(points=1, component_validity=True)
    x_axis = schema.cell_schema.data_axes[1]
    component = ComponentValidity(
        (x_axis.axis_id,), np.array([True, False, True], dtype=bool)
    )
    payload = Value(
        np.ones((2, 3), dtype=np.uint16),
        component,
        schema.cell_schema,
    )
    contract = ValuePayloadContract(schema.cell_schema)
    assert contract.retained_nbytes(payload) == payload.values.nbytes + component.mask.nbytes
    assert contract.max_retained_nbytes == payload.values.nbytes + 6


def test_value_payload_contract_requires_generation_owned_schema_identity():
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)
    equal_but_distinct = image_value_schema()
    assert equal_but_distinct.fingerprint == schema.cell_schema.fingerprint
    payload = Value(np.ones((2, 3), dtype=np.uint16), VALID, equal_but_distinct)
    with pytest.raises(TypeError, match="generation-owned"):
        producer.emit(
            payload,
            captured_at=0.0,
            trace=TraceContext("run", "camera", "capture"),
            join_key=DatasetCellAddress(0, 0),
        )


def test_minted_generation_prevents_revision_ref_collision_across_sources():
    schema = dataset_schema(points=1)

    def capture(number: int):
        stream, producer = source(schema, events=1)
        reservation = stream.reserve(
            total_events=1,
            max_inflight_events=1,
            max_inflight_bytes=12,
            trace_binding=TRACE_BINDING,
        )
        cursor = reservation.activate()
        builder = DatasetBuilder(
            BlockId("same-block-label"),
            reservation,
            schema,
            DatasetMode.FINITE_EXACT,
            expected_cells=cell_schedule(schema),
        )
        emit(producer, value(number), DatasetCellAddress(0, 0), 0)
        builder.consume(cursor.next())
        artifact = builder.seal(producer.finish())
        reservation.release()
        return artifact

    first = capture(1)
    second = capture(999)
    assert first.ref != second.ref
    assert first.provenance.generation != second.provenance.generation
