"""Exact/monitor stream contracts for finite acquisition data."""

from __future__ import annotations

import gc
import threading
import weakref

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
    ReservationStateError,
    ReservationState,
    SchemaChanged,
    StreamBackpressure,
    StreamGap,
    StreamEndedEarly,
    StreamError,
    AcquisitionCursor,
    EndOfStream,
    EventId,
    EventRef,
    OrderedEventSpanHasher,
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


def test_ordered_event_span_hasher_is_canonical_contiguous_and_owner_independent():
    stream_id = StreamId("camera.exact")
    generation = StreamGenerationId("generation-one")
    references = tuple(
        EventRef(
            stream_id,
            generation,
            sequence,
            EventId(f"event-{sequence}"),
            f"{sequence + 1:064x}",
        )
        for sequence in range(3)
    )

    producer_owner = OrderedEventSpanHasher(stream_id, generation, 0)
    consumer_owner = OrderedEventSpanHasher(stream_id, generation, 0)
    changed_owner = OrderedEventSpanHasher(stream_id, generation, 0)
    for reference in references:
        producer_owner.update(reference)
        consumer_owner.update(reference)
        changed_owner.update(
            EventRef(
                stream_id,
                generation,
                reference.sequence,
                EventId("changed-event")
                if reference.sequence == 1
                else reference.event_id,
                reference.payload_digest,
            )
        )

    producer_span = producer_owner.seal(3)
    assert consumer_owner.seal(3) == producer_span
    assert changed_owner.seal(3).ordered_digest != producer_span.ordered_digest

    changed_payload_owner = OrderedEventSpanHasher(stream_id, generation, 0)
    for reference in references:
        changed_payload_owner.update(
            EventRef(
                reference.stream_id,
                reference.generation,
                reference.sequence,
                reference.event_id,
                "f" * 64 if reference.sequence == 1 else reference.payload_digest,
            )
        )
    assert changed_payload_owner.seal(3).ordered_digest != producer_span.ordered_digest

    discontinuous = OrderedEventSpanHasher(stream_id, generation, 0)
    with pytest.raises(ValueError, match="stream/generation/sequence"):
        discontinuous.update(references[1])
    with pytest.raises(RuntimeError, match="already sealed"):
        producer_owner.update(references[-1])


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


def bind_terminal_consumer(source, reservation):
    owner = object()
    source._claim_consumer(
        reservation,
        owner,
        source_contract_digest="3" * 64,
        source_schedule_digest="4" * 64,
        source_key_sequence_digest="5" * 64,
        chain_contract_digest="6" * 64,
        terminal=True,
    )
    return owner


def acknowledge(source, reservation, delivery, owner) -> None:
    source._ack_consumer(reservation, delivery, owner)


def abort_consumer(source, reservation, owner) -> None:
    source._abort_consumer(reservation, owner, lambda: None)


def test_exact_reservation_retains_every_event_until_ordered_ack():
    source, producer = stream(events=4)
    reservation = source.reserve(
        total_events=4,
        max_inflight_events=4,
        max_inflight_bytes=32,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    emitted = [emit(producer, float(index)) for index in range(4)]

    assert cursor.next().envelope.event_id == emitted[0].event_id
    for expected in emitted:
        delivery = cursor.next(timeout=0.1)
        assert delivery.envelope is expected
        acknowledge(source, reservation, delivery, owner)
    with pytest.raises(StopIteration):
        cursor.next()
    eos = producer.finish()
    source._complete_consumer(reservation, eos, owner, lambda: None)
    assert reservation.state is ReservationState.COMPLETED
    reservation.release()
    assert source.retained_bytes <= 32


def test_exact_consumer_reuses_the_publication_event_ref_without_rehash(monkeypatch):
    digest_calls = 0
    original_digest = ValuePayloadContract.digest

    def counted_digest(contract, payload):
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(contract, payload)

    monkeypatch.setattr(ValuePayloadContract, "digest", counted_digest)
    source, producer = stream(events=1)
    reservation = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    emitted = emit(producer, 1.0)
    delivery = cursor.next()
    assert emitted.ref is emitted.event_ref
    assert emitted.payload_digest == emitted.event_ref.payload_digest
    assert digest_calls == 1
    source._validate_consumer_delivery(reservation, delivery, owner)
    acknowledge(source, reservation, delivery, owner)
    assert digest_calls == 1
    eos = producer.finish()
    source._complete_consumer(reservation, eos, owner, lambda: None)
    reservation.release()


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
    owner = bind_terminal_consumer(source, reservation)
    monitor = source.monitor(max_events=2, max_bytes=16)
    for index in range(3):
        emit(producer, float(index))
    with pytest.raises(StreamBackpressure):
        emit(producer, 3.0)

    first = cursor.next()
    acknowledge(source, reservation, first, owner)
    fourth = emit(producer, 3.0)
    update = monitor.latest()
    assert update.envelope is fourth
    assert update.missed == 3
    abort_consumer(source, reservation, owner)
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
    with pytest.raises(ReservationCapacityExceeded, match="one formal exact consumer"):
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


def test_unbound_exact_emit_is_side_effect_free_and_zero_event_preflight_is_retryable():
    source, producer = stream(events=1)
    reservation = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    reservation.activate()
    with pytest.raises(ReservationStateError, match="formal consumer"):
        emit(producer, 0.0)
    assert source.retained_events == 0
    assert source._next_sequence == 0
    assert reservation.acknowledged_sequence == 0
    reservation.abort()
    reservation.release()

    replacement = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    replacement.abort()
    replacement.release()


def test_failed_preclaim_is_retryable_but_first_data_tombstones_generation():
    source, producer = stream(events=1)
    reservation = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    reservation.activate()
    with pytest.raises(ValueError, match="digest"):
        source._claim_consumer(
            reservation,
            object(),
            source_contract_digest="not-a-digest",
            source_schedule_digest="4" * 64,
            source_key_sequence_digest="5" * 64,
            chain_contract_digest="6" * 64,
            terminal=True,
        )
    reservation.abort()
    reservation.release()

    claimed = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    claimed.activate()
    owner = bind_terminal_consumer(source, claimed)
    emit(producer, 0.0)
    abort_consumer(source, claimed, owner)
    claimed.release()
    with pytest.raises(
        ReservationCapacityExceeded,
        match="already had its formal exact consumer",
    ):
        source.reserve(
            total_events=1,
            max_inflight_events=1,
            max_inflight_bytes=8,
            trace_binding=TRACE_BINDING,
        )


def test_zero_event_rebind_gate_blocks_loose_emit_and_survives_failed_replacement():
    source, producer = stream(events=1)
    abandoned = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    abandoned.activate()
    owner = bind_terminal_consumer(source, abandoned)
    abort_consumer(source, abandoned, owner)
    abandoned.release()

    assert source._formal_rebind_required
    with pytest.raises(ReservationStateError, match="live bound reservation"):
        emit(producer, 0.0)
    with pytest.raises(StreamEndedEarly, match="replacement binding"):
        producer.finish()

    failed = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    failed.activate()
    with pytest.raises(ValueError, match="digest"):
        source._claim_consumer(
            failed,
            object(),
            source_contract_digest="not-a-digest",
            source_schedule_digest="4" * 64,
            source_key_sequence_digest="5" * 64,
            chain_contract_digest="6" * 64,
            terminal=True,
        )
    failed.abort()
    failed.release()
    assert source._formal_rebind_required
    with pytest.raises(ReservationStateError, match="live bound reservation"):
        emit(producer, 0.0)

    replacement = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    replacement.activate()
    replacement_owner = bind_terminal_consumer(source, replacement)
    assert not source._formal_rebind_required
    assert emit(producer, 0.0).sequence == 0
    abort_consumer(source, replacement, replacement_owner)
    replacement.release()
    assert producer.finish().end_sequence == 1


def test_zero_event_release_and_emit_are_serialized_without_an_authority_gap():
    source, producer = stream(events=1)
    reservation = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    abort_consumer(source, reservation, owner)

    barrier = threading.Barrier(3)
    emit_errors: list[BaseException] = []
    release_errors: list[BaseException] = []

    def release() -> None:
        barrier.wait()
        try:
            reservation.release()
        except BaseException as error:
            release_errors.append(error)

    def publish() -> None:
        barrier.wait()
        try:
            emit(producer, 0.0)
        except BaseException as error:
            emit_errors.append(error)

    release_thread = threading.Thread(target=release)
    emit_thread = threading.Thread(target=publish)
    release_thread.start()
    emit_thread.start()
    barrier.wait()
    release_thread.join(2.0)
    emit_thread.join(2.0)

    assert not release_thread.is_alive() and not emit_thread.is_alive()
    assert release_errors == []
    assert len(emit_errors) == 1
    assert isinstance(emit_errors[0], ReservationStateError)
    assert source.next_sequence == 0
    assert source._formal_rebind_required


def test_formal_interval_rejects_extra_or_post_failure_emit_and_short_finish():
    source, producer = stream(events=2)
    reservation = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    assert emit(producer, 0.0).sequence == 0
    with pytest.raises(ReservationStateError, match="live bound reservation"):
        emit(producer, 1.0)
    abort_consumer(source, reservation, owner)
    reservation.release()
    with pytest.raises(ReservationStateError, match="live bound reservation"):
        emit(producer, 1.0)
    assert producer.finish().end_sequence == 1

    short_source, short_producer = stream(events=2)
    short = short_source.reserve(
        total_events=2,
        max_inflight_events=2,
        max_inflight_bytes=16,
        trace_binding=TRACE_BINDING,
    )
    short.activate()
    short_owner = bind_terminal_consumer(short_source, short)
    emit(short_producer, 0.0)
    with pytest.raises(StreamEndedEarly, match="frozen interval"):
        short_producer.finish()
    abort_consumer(short_source, short, short_owner)
    short.release()


@pytest.mark.parametrize(
    "wrong_trace",
    (
        TraceContext("wrong-run", "camera-one", "capture"),
        TraceContext("run-one", "wrong-source", "capture"),
    ),
)
def test_formal_emit_rejects_wrong_trace_without_publication(wrong_trace):
    source, producer = stream(events=1)
    reservation = source.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=8,
        trace_binding=TRACE_BINDING,
    )
    reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    with pytest.raises(StreamError, match="formal run/source"):
        producer.emit(
            scalar_value(0.0),
            captured_at=0.0,
            trace=wrong_trace,
            join_key=(0, 0),
        )
    assert source.next_sequence == 0
    assert source.retained_events == 0
    assert reservation.acknowledged_sequence == 0
    abort_consumer(source, reservation, owner)
    reservation.release()


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

        @staticmethod
        def digest(_payload):
            return "0" * 64

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


def test_stream_requires_payload_contract_owned_content_digest():
    class MissingDigestContract:
        fingerprint = PAYLOAD_FINGERPRINT
        max_retained_nbytes = 8

        @staticmethod
        def snapshot(payload):
            return payload

        @staticmethod
        def validate(_payload):
            return None

        @staticmethod
        def retained_nbytes(_payload):
            return 8

    with pytest.raises(TypeError, match="payload_contract.digest"):
        AcquisitionStream.create(
            StreamId("missing.payload-digest"),
            MissingDigestContract(),
            flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
            retention_events=1,
            retention_bytes=8,
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
    bind_terminal_consumer(source, reservation)
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


def test_producer_owns_stream_but_terminal_receipt_does_not_create_a_cycle():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        source, producer = stream()
        source_reference = weakref.ref(source)

        del source
        assert source_reference() is producer._stream
        eos = producer.finish()
        del producer

        assert source_reference() is None
        assert eos._owner is None
    finally:
        if was_enabled:
            gc.enable()


def test_dead_weak_authority_never_matches_none():
    from zlc_neutral_atom.runtime.streams import _ObjectReference

    class Owner:
        pass

    owner = Owner()
    reference = _ObjectReference(owner)
    owner_reference = weakref.ref(owner)
    del owner

    assert owner_reference() is None
    assert reference.get() is None
    assert not reference.matches(None)
