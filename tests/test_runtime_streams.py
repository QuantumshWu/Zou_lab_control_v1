"""Exact/monitor stream contracts for finite acquisition data."""

from __future__ import annotations

import gc
import threading
import weakref

import numpy as np
import pytest
import zlc_neutral_atom.runtime.streams as runtime_streams

from zlc_data import (
    AxisId,
    AxisSpec,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    PointColumn,
    PointTable,
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
    ReservationStateError,
    ReservationState,
    SchemaChanged,
    StreamGap,
    StreamEndedEarly,
    StreamError,
    AcquisitionCursor,
    EndOfStream,
    SourceFailed,
    StreamId,
)


SCALAR_SCHEMA = ValueSchema.scalar(np.dtype("<f8"))


class TupleJoinContract:
    @staticmethod
    def snapshot(key):
        if not isinstance(key, tuple):
            raise TypeError("join key must be tuple")
        return tuple(key)


def scalar_value(value: float) -> Value:
    return Value(
        np.asarray([value], dtype=np.float64),
        VALID,
        SCALAR_SCHEMA,
    )


def stream():
    contract = ValuePayloadContract(SCALAR_SCHEMA)
    return AcquisitionStream.create(
        StreamId("camera.frames"),
        contract,
        join_key_contract=TupleJoinContract(),
    )


def emit(producer, value: float):
    return producer.emit(
        scalar_value(value),
        captured_at=float(value),
        join_key=(0, int(value)),
    )


def bind_terminal_consumer(source, reservation):
    owner = object()
    source._claim_consumer(
        reservation,
        owner,
        terminal=True,
    )
    return owner


def acknowledge(source, reservation, delivery, owner) -> None:
    source._ack_consumer(reservation, delivery, owner)


def abort_consumer(source, reservation, owner) -> None:
    source._abort_consumer(reservation, owner, lambda: None)


def test_exact_reservation_retains_every_event_until_ordered_ack():
    source, producer = stream()
    reservation = source.reserve(
        total_events=4,
    )
    cursor = reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    emitted = [emit(producer, float(index)) for index in range(4)]

    assert cursor.next().envelope.event_ref is emitted[0].event_ref
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


def test_exact_consumer_reuses_the_publication_event_ref() -> None:
    source, producer = stream()
    reservation = source.reserve(
        total_events=1,
    )
    cursor = reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    emitted = emit(producer, 1.0)
    delivery = cursor.next()
    assert emitted.ref is emitted.event_ref
    assert delivery.envelope.event_ref is emitted.event_ref
    source._validate_consumer_delivery(reservation, delivery, owner)
    acknowledge(source, reservation, delivery, owner)
    eos = producer.finish()
    source._complete_consumer(reservation, eos, owner, lambda: None)
    reservation.release()


def test_unreserved_cursor_gets_typed_gap_instead_of_latest_fallback():
    source, producer = stream()
    cursor = source.subscribe(start_sequence=0)
    for index in range(5):
        emit(producer, float(index))
    with pytest.raises(StreamGap) as caught:
        cursor.next()
    assert caught.value.expected == 0
    assert caught.value.earliest_retained == 5


def test_slow_exact_consumer_retains_all_events_and_display_latest_reports_skips():
    source, producer = stream()
    reservation = source.reserve(
        total_events=4,
    )
    cursor = reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    monitor = source.monitor()
    emitted = []
    for index in range(4):
        emitted.append(emit(producer, float(index)))

    assert source.retained_events == 4
    update = monitor.latest()
    assert update.envelope is emitted[-1]
    assert update.missed == 3
    for expected in emitted:
        delivery = cursor.next()
        assert delivery.envelope is expected
        acknowledge(source, reservation, delivery, owner)
    assert source.retained_events == 0
    eos = producer.finish()
    source._complete_consumer(reservation, eos, owner, lambda: None)
    reservation.release()
    monitor.close()


def test_monitor_next_never_drops_and_display_latest_does_not_change_exact_order():
    total = 100
    source, producer = stream()
    reservation = source.reserve(
        total_events=total,
    )
    cursor = reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    ordered_monitor = source.monitor()
    latest_monitor = source.monitor()
    exact_sequences = []

    for sequence in range(total):
        emit(producer, float(sequence))
        delivery = cursor.next()
        exact_sequences.append(delivery.envelope.sequence)
        acknowledge(source, reservation, delivery, owner)

    assert exact_sequences == list(range(total))
    ordered_updates = [ordered_monitor.next() for _ in range(total)]
    assert [update.envelope.sequence for update in ordered_updates] == list(range(total))
    assert all(update.missed == 0 for update in ordered_updates)
    latest = latest_monitor.latest()
    assert latest.envelope.sequence == total - 1
    assert latest.missed == total - 1
    eos = producer.finish()
    source._complete_consumer(reservation, eos, owner, lambda: None)
    reservation.release()
    ordered_monitor.close()
    latest_monitor.close()


def test_monitor_topology_cannot_expand_after_publication_begins():
    source, producer = stream()
    emit(producer, 0.0)
    with pytest.raises(ReservationStateError, match="before the first publication"):
        source.monitor()


def test_monitor_close_is_terminal_for_polling_and_wakes_blocked_reader(monkeypatch):
    source, _producer = stream()
    monitor = source.monitor()
    failures: list[BaseException] = []
    entered_wait = threading.Event()
    real_wait = monitor._condition.wait

    def announced_wait(timeout=None):
        entered_wait.set()
        return real_wait(timeout)

    monkeypatch.setattr(monitor._condition, "wait", announced_wait)

    def read() -> None:
        try:
            monitor.next()
        except BaseException as error:
            failures.append(error)

    reader = threading.Thread(target=read)
    reader.start()
    assert entered_wait.wait(timeout=1.0)
    monitor.close()
    reader.join(timeout=1.0)
    assert not reader.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], StreamEndedEarly)
    with pytest.raises(StreamEndedEarly, match="closed"):
        monitor.next()
    with pytest.raises(StreamEndedEarly, match="closed"):
        monitor.latest()


def test_reservation_allows_only_one_formal_exact_consumer():
    source, _producer = stream()
    first = source.reserve(
        total_events=4,
    )
    with pytest.raises(ReservationStateError, match="one formal exact consumer"):
        source.reserve(
            total_events=2,
        )
    first.abort()
    first.release()


def test_unbound_exact_emit_is_side_effect_free_and_zero_event_preflight_is_retryable():
    source, producer = stream()
    reservation = source.reserve(
        total_events=1,
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
    )
    replacement.abort()
    replacement.release()


def test_failed_preclaim_is_retryable_but_first_data_tombstones_generation():
    source, producer = stream()
    reservation = source.reserve(
        total_events=1,
    )
    reservation.activate()
    with pytest.raises(ValueError, match="either a terminal"):
        source._claim_consumer(
            reservation,
            object(),
        )
    reservation.abort()
    reservation.release()

    claimed = source.reserve(
        total_events=1,
    )
    claimed.activate()
    owner = bind_terminal_consumer(source, claimed)
    emit(producer, 0.0)
    abort_consumer(source, claimed, owner)
    claimed.release()
    with pytest.raises(
        ReservationStateError,
        match="already had its formal exact consumer",
    ):
        source.reserve(
            total_events=1,
        )


def test_zero_event_rebind_gate_blocks_loose_emit_and_survives_failed_replacement():
    source, producer = stream()
    abandoned = source.reserve(
        total_events=1,
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
    )
    failed.activate()
    with pytest.raises(ValueError, match="either a terminal"):
        source._claim_consumer(
            failed,
            object(),
        )
    failed.abort()
    failed.release()
    assert source._formal_rebind_required
    with pytest.raises(ReservationStateError, match="live bound reservation"):
        emit(producer, 0.0)

    replacement = source.reserve(
        total_events=1,
    )
    replacement.activate()
    replacement_owner = bind_terminal_consumer(source, replacement)
    assert not source._formal_rebind_required
    assert emit(producer, 0.0).sequence == 0
    abort_consumer(source, replacement, replacement_owner)
    replacement.release()
    assert producer.finish().end_sequence == 1


def test_zero_event_release_and_emit_are_serialized_without_an_authority_gap():
    source, producer = stream()
    reservation = source.reserve(
        total_events=1,
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
    source, producer = stream()
    reservation = source.reserve(
        total_events=1,
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

    short_source, short_producer = stream()
    short = short_source.reserve(
        total_events=2,
    )
    short.activate()
    short_owner = bind_terminal_consumer(short_source, short)
    emit(short_producer, 0.0)
    with pytest.raises(StreamEndedEarly, match="frozen interval"):
        short_producer.finish()
    abort_consumer(short_source, short, short_owner)
    short.release()


def test_stream_rejects_materialized_dataset_payloads():
    class IllegalContract:
        @staticmethod
        def snapshot(payload):
            return payload

        @staticmethod
        def validate(_payload):
            return None

    source, producer = AcquisitionStream.create(
        StreamId("illegal.dataset"),
        IllegalContract(),
    )
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1)
    point = AxisSpec(AxisId("point"), "point", SPATIAL_X, 1)
    scalar = ValueSchema.scalar(np.dtype("<f8"))
    schema = DatasetSchema(
        repeat,
        PointTable(
            1,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    (0,),
                    point.unit,
                    point.coordinate_frame,
                ),
            ),
        ),
        None,
        scalar,
    )
    block = DataBlock(
        BlockId("block"),
        DatasetRevision(0),
        np.zeros((1, 1, 1), dtype=np.float64),
        VALID,
        schema,
    )
    with pytest.raises(TypeError, match="materialization"):
        producer.emit(
            block,
            captured_at=0.0,
        )
    with pytest.raises(TypeError, match="materialization"):
        producer.emit(
            ("nested", block),
            captured_at=0.0,
        )
    assert source.next_sequence == 0
    assert source.retained_events == 0


def test_value_payload_runs_the_materialization_admission_once(monkeypatch):
    source, producer = stream()
    original = runtime_streams._contains_materialization
    calls = 0

    def counted(value, seen=None):
        nonlocal calls
        calls += 1
        return original(value, seen)

    monkeypatch.setattr(runtime_streams, "_contains_materialization", counted)
    expected = emit(producer, 1.0)

    assert expected.sequence == 0
    assert calls == 1


def test_stream_requires_payload_contract_validation():
    class MissingValidateContract:
        @staticmethod
        def snapshot(payload):
            return payload

    with pytest.raises(TypeError, match="payload_contract.validate"):
        AcquisitionStream.create(
            StreamId("missing.payload-validation"),
            MissingValidateContract(),
        )


def test_join_key_is_snapshotted_and_validated_by_one_contract():
    _source, producer = stream()
    with pytest.raises(TypeError, match="join key"):
        producer.emit(
            scalar_value(1.0),
            captured_at=1.0,
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
    monitor = source.monitor()
    expected = emit(producer, 1.0)
    assert source.retained_events == 0
    eos = producer.finish()
    assert eos.end_sequence == 1
    assert monitor.next().envelope is expected
    with pytest.raises(StreamEndedEarly, match="end-of-stream"):
        monitor.next(timeout=0.01)
    monitor.close()


def test_event_refs_advance_with_the_stream_sequence():
    _source, producer = stream()
    first = emit(producer, 1.0)
    second = emit(producer, 2.0)
    assert first.event_ref.stream_id == second.event_ref.stream_id
    assert first.event_ref.generation == second.event_ref.generation
    assert first.event_ref.sequence == 0
    assert second.event_ref.sequence == 1


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
    monitor = source.monitor()
    emit(producer, 1.0)
    producer.supersede(StreamGenerationId("generation-two"))
    with pytest.raises(SchemaChanged):
        monitor.next()


def test_rejected_emit_has_no_retention_side_effects():
    source, producer = stream()
    reservation = source.reserve(total_events=2)
    cursor = reservation.activate()
    owner = bind_terminal_consumer(source, reservation)
    expected = emit(producer, 1.0)
    with pytest.raises(ValueError):
        producer.emit(
            scalar_value(2.0),
            captured_at=float("nan"),
            join_key=(0, 2),
        )
    assert source.retained_events == 1
    delivery = cursor.next()
    assert delivery.envelope is expected
    acknowledge(source, reservation, delivery, owner)
    abort_consumer(source, reservation, owner)
    reservation.release()


def test_first_post_marker_failure_terminalizes_without_retryable_hole(monkeypatch):
    source, producer = stream()
    monitor = source.monitor()
    emit(producer, 1.0)
    real_offer = runtime_streams.MonitorTap._offer

    def fail_target_offer(self, stored):
        if self is monitor:
            raise RuntimeError("synthetic first post-marker failure")
        return real_offer(self, stored)

    monkeypatch.setattr(runtime_streams.MonitorTap, "_offer", fail_target_offer)
    with pytest.raises(SourceFailed, match="authoritative sequence 1 committed"):
        emit(producer, 2.0)

    assert source.next_sequence == 2
    with pytest.raises(SourceFailed, match="authoritative sequence 1 committed"):
        producer.emit(
            scalar_value(3.0),
            captured_at=3.0,
            join_key=(0, 3),
        )


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


def test_one_broken_terminal_tap_cannot_strand_the_remaining_fanout(monkeypatch):
    source, producer = stream()
    broken = source.monitor()
    healthy = source.monitor()
    real_source_ended = runtime_streams.MonitorTap._source_ended

    def fail_one_tap(self, error):
        if self is broken:
            raise RuntimeError("synthetic terminal tap failure")
        return real_source_ended(self, error)

    monkeypatch.setattr(
        runtime_streams.MonitorTap,
        "_source_ended",
        fail_one_tap,
    )
    failure = SourceFailed("driver terminal truth")
    producer.fail(failure)

    for tap in (broken, healthy):
        with pytest.raises(SourceFailed, match="driver terminal truth") as caught:
            tap.next(0.0)
        assert caught.value is failure
    assert source._closed and not source._monitors


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
