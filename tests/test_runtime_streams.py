"""Exact/monitor stream contracts for finite acquisition data."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    PointLayout,
    REPEAT,
    SPATIAL_X,
    StreamGenerationId,
    VALID,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
    BlockId,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ReservationCapacityExceeded,
    ReservationState,
    SchemaChanged,
    StreamBackpressure,
    StreamGap,
    StreamEndedEarly,
    AcquisitionCursor,
    EndOfStream,
    ProducerFlowControl,
    RetentionOverrun,
    SourceFailed,
    StreamId,
    TraceContext,
    TraceBinding,
)


PAYLOAD_FINGERPRINT = "1" * 64
JOIN_FINGERPRINT = "2" * 64
SCALAR_SCHEMA = ValueSchema((), ValidityContract.value(), np.dtype("<f8"))
TRACE_BINDING = TraceBinding("run-one", "camera-one")


class TupleJoinContract:
    fingerprint = JOIN_FINGERPRINT

    @staticmethod
    def snapshot(key):
        return tuple(key) if isinstance(key, tuple) else key

    @staticmethod
    def validate(key):
        if not isinstance(key, tuple):
            raise TypeError("join key must be tuple")


def trace() -> TraceContext:
    return TraceContext(
        run_id="run-one",
        source_id="camera-one",
        correlation_id="capture-one",
    )


def scalar_value(value: float) -> Value:
    return Value(
        np.asarray(value, dtype=np.float64),
        VALID,
        SCALAR_SCHEMA,
    )


def stream(
    *,
    events: int = 8,
    payload_bytes: int = 8,
    flow_control: ProducerFlowControl = ProducerFlowControl.BACKPRESSURE_CAPABLE,
):
    contract = ValuePayloadContract(SCALAR_SCHEMA)
    return AcquisitionStream.create(
        StreamId("camera.frames"),
        contract,
        flow_control=flow_control,
        retention_events=events,
        retention_bytes=events * payload_bytes,
        join_key_contract=TupleJoinContract(),
    )


def emit(producer, value: float):
    return producer.emit(
        scalar_value(value),
        captured_at=float(value),
        trace=trace(),
        join_key=(0, int(value)),
    )


def test_exact_reservation_retains_every_event_until_ordered_ack():
    source, producer = stream(events=4)
    reservation = source.reserve(
        total_events=4,
        max_inflight_events=4,
        max_inflight_bytes=32,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    emitted = [emit(producer, float(index)) for index in range(4)]

    assert cursor.next().envelope.event_id == emitted[0].event_id
    for expected in emitted:
        delivery = cursor.next(timeout=0.1)
        assert delivery.envelope is expected
        delivery.ack()
    with pytest.raises(StopIteration):
        cursor.next()
    reservation.complete()
    assert reservation.state is ReservationState.COMPLETED
    reservation.release()
    assert source.retained_bytes <= 32


def test_unreserved_cursor_gets_typed_gap_instead_of_latest_fallback():
    source, producer = stream(events=2)
    cursor = source.subscribe(start_sequence=0)
    for index in range(5):
        emit(producer, float(index))
    with pytest.raises(StreamGap) as caught:
        cursor.next()
    assert caught.value.expected == 0
    assert caught.value.earliest_retained == 3


def test_exact_backlog_fails_before_overwrite_and_monitor_still_overwrites():
    source, producer = stream(events=3)
    reservation = source.reserve(
        total_events=4,
        max_inflight_events=3,
        max_inflight_bytes=24,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    monitor = source.monitor(max_events=2, max_bytes=16)
    for index in range(3):
        emit(producer, float(index))
    with pytest.raises(StreamBackpressure):
        emit(producer, 3.0)

    first = cursor.next()
    first.ack()
    fourth = emit(producer, 3.0)
    update = monitor.latest()
    assert update.envelope is fourth
    assert update.missed == 3
    reservation.abort()
    reservation.release()
    monitor.close()


def test_reservation_admission_is_atomic_over_event_and_byte_capacity():
    source, _producer = stream(events=4)
    first = source.reserve(
        total_events=4,
        max_inflight_events=3,
        max_inflight_bytes=24,
        trace_binding=TRACE_BINDING,
    )
    with pytest.raises(ReservationCapacityExceeded, match="one formal materializer"):
        source.reserve(
            total_events=2,
            max_inflight_events=2,
            max_inflight_bytes=16,
            trace_binding=TRACE_BINDING,
        )
    first.abort()
    first.release()

    with pytest.raises(ValueError, match="max_payload_bytes"):
        source.reserve(
            total_events=2,
            max_inflight_events=2,
            max_inflight_bytes=8,
            trace_binding=TRACE_BINDING,
        )


def test_stream_rejects_materialized_dataset_payloads():
    class IllegalContract:
        fingerprint = PAYLOAD_FINGERPRINT
        max_retained_nbytes = 1024

        @staticmethod
        def snapshot(payload):
            return payload

        @staticmethod
        def validate(_payload):
            return None

        @staticmethod
        def retained_nbytes(payload):
            return payload.values.nbytes

    source, producer = AcquisitionStream.create(
        StreamId("illegal.dataset"),
        IllegalContract(),
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=1,
        retention_bytes=1024,
    )
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1)
    point = AxisSpec(AxisId("point"), "point", SPATIAL_X, 1)
    scalar = ValueSchema((), ValidityContract.value(), np.dtype("<f8"))
    schema = DatasetSchema(repeat, (point,), PointLayout.rect_c((1,)), scalar)
    block = DataBlock(
        BlockId("block"),
        DatasetRevision(0),
        np.zeros((1, 1), dtype=np.float64),
        VALID,
        schema,
    )
    with pytest.raises(TypeError, match="materialization"):
        producer.emit(
            block,
            captured_at=0.0,
            trace=trace(),
        )


def test_join_key_is_snapshotted_and_validated_by_one_contract():
    _source, producer = stream()
    with pytest.raises(TypeError, match="join key"):
        producer.emit(
            scalar_value(1.0),
            captured_at=1.0,
            trace=trace(),
            join_key=[0, 0],
        )


def test_generation_supersession_terminates_old_cursor_even_with_retained_data():
    source, producer = stream()
    cursor = source.subscribe(start_sequence=0)
    emit(producer, 0.0)
    producer.supersede(StreamGenerationId("generation-two"))
    with pytest.raises(SchemaChanged) as caught:
        cursor.next()
    assert caught.value.replacement == StreamGenerationId("generation-two")


def test_monitor_drains_retained_event_then_observes_source_eos():
    source, producer = stream()
    monitor = source.monitor(max_events=2, max_bytes=16)
    expected = emit(producer, 1.0)
    eos = producer.finish()
    assert eos.end_sequence == 1
    assert monitor.next().envelope is expected
    with pytest.raises(StreamEndedEarly, match="end-of-stream"):
        monitor.next(timeout=0.01)
    monitor.close()


def test_payload_size_is_measured_by_stream_owner_and_event_ids_do_not_repeat():
    _source, producer = stream()
    first = emit(producer, 1.0)
    second = emit(producer, 2.0)
    assert first.event_id != second.event_id


def test_cursor_and_terminal_receipt_cannot_be_fabricated():
    source, _producer = stream()
    with pytest.raises(PermissionError):
        AcquisitionCursor(
            object(),
            stream=source,
            start_sequence=0,
            end_sequence=1,
            reservation_token=object(),
        )
    with pytest.raises(PermissionError):
        EndOfStream(
            object(),
            stream_id=source.stream_id,
            stream_generation=source.generation,
            end_sequence=0,
            ended_at=0.0,
            owner=source,
            nonce=object(),
        )


def test_schema_change_discards_monitor_queue_immediately():
    source, producer = stream()
    monitor = source.monitor(max_events=2, max_bytes=16)
    emit(producer, 1.0)
    producer.supersede(StreamGenerationId("generation-two"))
    with pytest.raises(SchemaChanged):
        monitor.next()


def test_rejected_emit_has_no_retention_side_effects():
    source, producer = stream(events=1)
    cursor = source.subscribe(start_sequence=0)
    expected = emit(producer, 1.0)
    with pytest.raises(ValueError, match="finite timestamp"):
        producer.emit(
            scalar_value(2.0),
            captured_at=float("nan"),
            trace=trace(),
            join_key=(0, 2),
        )
    assert source.retained_events == 1
    assert cursor.next().envelope is expected


def test_non_backpressure_overrun_permanently_poisons_generation():
    source, producer = stream(
        events=1,
        flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
    )
    reservation = source.reserve(
        total_events=2,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    emit(producer, 10.0)
    with pytest.raises(RetentionOverrun):
        emit(producer, 20.0)
    with pytest.raises(RetentionOverrun):
        emit(producer, 30.0)
    with pytest.raises(RetentionOverrun):
        producer.finish()
    with pytest.raises(RetentionOverrun):
        cursor.next()
    assert reservation.state is ReservationState.FAILED
    reservation.release()


def test_first_terminal_fact_cannot_be_replaced():
    source, producer = stream()
    cursor = source.subscribe(start_sequence=0)
    failure = SourceFailed("driver stopped")
    producer.fail(failure)
    with pytest.raises(StreamEndedEarly, match="terminal failure"):
        producer.supersede(StreamGenerationId("replacement"))
    with pytest.raises(SourceFailed) as caught:
        cursor.next()
    assert caught.value is failure

    source2, producer2 = stream()
    replacement = StreamGenerationId("replacement-two")
    producer2.supersede(replacement)
    with pytest.raises(StreamEndedEarly, match="terminal failure"):
        producer2.fail(SourceFailed("late failure"))
    with pytest.raises(SchemaChanged):
        source2.subscribe(start_sequence=0).next()
