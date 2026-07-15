"""DatasetBuilder contracts over exact and monitor stream deliveries."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from enum import Enum
import threading

import numpy as np
import pytest
import zlc_neutral_atom.runtime.dataset as runtime_dataset

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DatasetSchema,
    MONITOR_HISTORY,
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
    DatasetCellDomain,
    DatasetCellKeyContract,
    DatasetCoverage,
    DatasetPreviewSnapshot,
    FrozenDatasetEdge,
    MissingDatasetCells,
    MonitorCoverage,
    MonitorDataset,
    OrderedDatasetEventHasher,
    OrderedDatasetMetadataHasher,
    SnapshotExpired,
    SealedDatasetArtifact,
    ValueDatasetEventAdapter,
    dataset_cell_key_fingerprint,
    dataset_cell_permutation_digest,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    EventId,
    EventRef,
    ReservationCapacityExceeded,
    ReservationStateError,
    StreamEndedEarly,
    StreamId,
    TraceContext,
    TraceBinding,
    ProducerFlowControl,
)


FINGERPRINT = "3" * 64
TRACE_BINDING = TraceBinding("run", "camera")


def test_formal_and_monitor_coverage_have_distinct_loss_semantics():
    assert DatasetCoverage(1, 1).complete
    assert not hasattr(DatasetCoverage(1, 1), "missed_events")
    assert MonitorCoverage(1, 1, 1, current_gap=False).complete
    assert not MonitorCoverage(1, 1, 1, current_gap=True).complete


def test_ordered_metadata_hasher_matches_frozen_golden_and_preserves_order():
    fingerprint = "a" * 64
    metadata_digests = ("b" * 64, "c" * 64)
    expected = "b20dd8bdd812e18599a5f4b49437265f5ef51619181f1b0f6f57775bf1fbae60"

    hasher = OrderedDatasetMetadataHasher(fingerprint)
    for digest in metadata_digests:
        hasher.update(digest)

    assert hasher.digest() == expected
    assert hasher.digest() == expected

    reversed_order = OrderedDatasetMetadataHasher(fingerprint)
    for digest in reversed(metadata_digests):
        reversed_order.update(digest)
    changed_content = OrderedDatasetMetadataHasher(fingerprint)
    for digest in (metadata_digests[0], "d" * 64):
        changed_content.update(digest)
    assert reversed_order.digest() != expected
    assert changed_content.digest() != expected

    with pytest.raises(ValueError, match="SHA-256"):
        OrderedDatasetMetadataHasher("not-a-digest")
    with pytest.raises(ValueError, match="SHA-256"):
        hasher.update("not-a-digest")


def test_ordered_dataset_event_hasher_matches_frozen_golden_and_all_identities():
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
    metadata_digests = ("b" * 64, "c" * 64, "d" * 64)

    def digest(refs, metadata):
        hasher = OrderedDatasetEventHasher(stream_id, generation, 0)
        for reference, metadata_digest in zip(refs, metadata, strict=True):
            hasher.update(reference, metadata_digest)
        return hasher.digest(3)

    expected = "bc8f115874accd1f02060c18c48939e9360992c4ab033579694b8a630a072433"
    assert digest(references, metadata_digests) == expected

    assert digest(
        (references[0], replace(references[1], event_id=EventId("changed-event")), references[2]),
        metadata_digests,
    ) != expected
    assert digest(
        (references[0], replace(references[1], payload_digest="f" * 64), references[2]),
        metadata_digests,
    ) != expected
    assert digest(references, (metadata_digests[0], "e" * 64, metadata_digests[2])) != expected

    discontinuous = OrderedDatasetEventHasher(stream_id, generation, 0)
    with pytest.raises(ValueError, match="stream/generation/sequence"):
        discontinuous.update(references[1], metadata_digests[0])
    complete = OrderedDatasetEventHasher(stream_id, generation, 0)
    for reference, metadata_digest in zip(references, metadata_digests, strict=True):
        complete.update(reference, metadata_digest)
    with pytest.raises(ValueError, match="incomplete coverage"):
        complete.digest(2)


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


def dataset_schema(
    *,
    repeats: int = 1,
    points: int = 3,
    component_validity: bool = False,
    cell_schema: ValueSchema | None = None,
):
    return DatasetSchema(
        axis("repeat", REPEAT, repeats),
        (axis("detuning", SCAN_POINT, points),),
        PointLayout.rect_c((points,)),
        cell_schema
        if cell_schema is not None
        else image_value_schema(component_validity=component_validity),
    )


def monitor_history_schema(source_schema: DatasetSchema, capacity: int) -> DatasetSchema:
    return DatasetSchema(
        axis("repeat", REPEAT, 1),
        (axis("monitor.history", MONITOR_HISTORY, capacity),),
        PointLayout.rect_c((capacity,)),
        source_schema.cell_schema,
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


def event_adapter(stream) -> ValueDatasetEventAdapter:
    return ValueDatasetEventAdapter(stream._payload_contract)


def dataset_edge(
    stream,
    schema: DatasetSchema,
    *,
    exact: bool = True,
    adapter=None,
) -> FrozenDatasetEdge:
    return FrozenDatasetEdge(
        schema,
        event_adapter(stream) if adapter is None else adapter,
        cell_schedule(schema) if exact else None,
    )


def test_cell_key_domain_excludes_value_schema_but_formal_permutation_does_not():
    first = dataset_schema(points=2)
    alternate_cell = ValueSchema(
        first.cell_schema.data_axes,
        first.cell_schema.validity_contract,
        np.dtype("<f4"),
        value_unit="probability",
    )
    second = DatasetSchema(
        first.repeat_axis,
        first.point_axes,
        first.point_layout,
        alternate_cell,
    )
    cells = cell_schedule(first)
    assert first.fingerprint != second.fingerprint
    assert dataset_cell_key_fingerprint(first) == dataset_cell_key_fingerprint(second)
    assert dataset_cell_permutation_digest(first, cells) != (
        dataset_cell_permutation_digest(second, cells)
    )


def test_frozen_edge_owner_copies_exact_schedule_addresses():
    schema = dataset_schema(points=2)
    stream, _producer = source(schema, events=2)
    schedule = cell_schedule(schema)
    edge = FrozenDatasetEdge(schema, event_adapter(stream), schedule)
    frozen_digest = edge.schedule_digest

    assert edge.expected_cells is not schedule
    assert edge.expected_cells is not None
    assert edge.expected_cells[0] is not schedule[0]
    object.__setattr__(schedule[0], "point_storage_index", 1)

    assert edge.expected_cells[0] == DatasetCellAddress(0, 0)
    assert edge.schedule_digest == frozen_digest


def test_dataset_cell_key_snapshot_detaches_the_published_envelope():
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)
    address = DatasetCellAddress(0, 0)

    envelope = emit(producer, value(1), address, 0)
    object.__setattr__(address, "point_storage_index", 7)

    assert envelope.join_key is not address
    assert envelope.join_key == DatasetCellAddress(0, 0)


def test_frozen_edge_rejects_normally_mutable_adapter_configuration():
    schema = dataset_schema(points=1)
    stream, _producer = source(schema, events=1)

    class HashableMutableScale:
        __hash__ = object.__hash__

        def __init__(self, factor: int) -> None:
            self.factor = factor

    @dataclass(frozen=True)
    class MutableAdapter:
        payload_contract: ValuePayloadContract
        scale: HashableMutableScale
        metadata_contract: object = ValueDatasetEventAdapter(
            ValuePayloadContract(image_value_schema())
        ).metadata_contract
        operator_fingerprint: str = "e" * 64

        @property
        def value_schema(self):
            return self.payload_contract.schema

        def value(self, payload):
            return Value(
                payload.values * self.scale.factor,
                payload.validity,
                payload.schema,
            )

    with pytest.raises(TypeError, match="intrinsically immutable"):
        FrozenDatasetEdge(
            schema,
            MutableAdapter(stream._payload_contract, HashableMutableScale(1)),
            cell_schedule(schema),
        )


def test_metadata_contract_validation_precedes_exact_commit(monkeypatch):
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)
    edge = dataset_edge(stream, schema)

    def reject(_metadata):
        raise ValueError("semantic metadata rejection")

    monkeypatch.setattr(type(edge.metadata_contract), "validate", staticmethod(reject))
    reservation = stream.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=12,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(BlockId("metadata-validation"), reservation, edge)
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    delivery = cursor.next()

    with pytest.raises(ValueError, match="semantic metadata rejection"):
        builder.consume(delivery)

    assert not delivery.acknowledged
    assert builder.revision.value == 0
    builder.abort()
    reservation.release()


def test_frozen_edge_validates_and_projects_one_exact_schedule_once(monkeypatch):
    schema = dataset_schema(repeats=2, points=3)
    stream, _producer = source(schema, events=6)
    validate_calls = 0
    real_validate = runtime_dataset.dataset_cell_permutation_digest

    def counted_validate(*args, **kwargs):
        nonlocal validate_calls
        validate_calls += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(
        runtime_dataset,
        "dataset_cell_permutation_digest",
        counted_validate,
    )

    edge = FrozenDatasetEdge(schema, event_adapter(stream), cell_schedule(schema))

    assert validate_calls == 1
    assert edge.schedule_digest == (
        "804db444ad86a9709c808ed1b205976ae563dd41c61de915670d64269aca9d28"
    )
    assert edge.key_sequence_digest == (
        "c55bf0d3545f3082ae0e18e19b8ac418b485354df2b53cf11fbca5029bce164a"
    )
    assert edge.consumer_contract_digest == (
        "36fce4323cd1176f8f0463e9149653c9ac2040611cf3747180f1cc2a5e290607"
    )


@pytest.mark.parametrize(
    ("schedule", "error"),
    (
        ((DatasetCellAddress(0, 0),), ValueError),
        ((DatasetCellAddress(0, 0), DatasetCellAddress(0, 0)), ValueError),
        ((DatasetCellAddress(0, 0), DatasetCellAddress(0, 2)), ValueError),
        ((DatasetCellAddress(0, 0), object()), TypeError),
    ),
)
def test_frozen_edge_rejects_incomplete_duplicate_and_foreign_schedules(
    schedule,
    error,
):
    schema = dataset_schema(points=2)
    stream, _producer = source(schema, events=2)
    with pytest.raises(error):
        FrozenDatasetEdge(schema, event_adapter(stream), schedule)


def test_cell_domain_fingerprint_is_cached_for_large_explicit_layout(monkeypatch):
    size = 2000
    source = dataset_schema(points=size)
    explicit = DatasetSchema(
        source.repeat_axis,
        source.point_axes,
        PointLayout.explicit((size,), tuple((index,) for index in reversed(range(size)))),
        source.cell_schema,
    )
    domain = DatasetCellDomain.from_schema(explicit)
    fingerprint = domain.fingerprint

    def forbidden(_layout):
        raise AssertionError("cached fingerprint reserialized the explicit layout")

    monkeypatch.setattr(
        "zlc_neutral_atom.runtime.dataset.point_layout_to_tree",
        forbidden,
    )
    assert domain.fingerprint == fingerprint
    assert dataset_cell_key_fingerprint(domain) == fingerprint


def test_cell_domain_rejects_non_repeat_repeat_axis():
    point = axis("point", SCAN_POINT, 1)
    with pytest.raises(ValueError, match="role 'repeat'"):
        DatasetCellDomain(
            axis("not-repeat", SCAN_POINT, 1),
            (point,),
            PointLayout.rect_c((1,)),
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
        dataset_edge(stream, schema),
    )

    emit(producer, value(10), DatasetCellAddress(0, 0), 0)
    progress = builder.consume(cursor.next())
    first_ref = progress.ref
    first = builder.materialize(first_ref)
    assert isinstance(first, DatasetPreviewSnapshot)
    assert first.block.values.shape == (1, 3, 2, 3)
    assert np.all(first.block.values[0, 0] == 10)
    assert progress.coverage.written_cells == 1
    assert not hasattr(progress, "block")

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
        dataset_edge(stream, schema),
    )
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    delivery = cursor.next()
    with pytest.raises(PermissionError, match="bound exact consumer"):
        delivery.ack()
    builder.consume(delivery)
    with pytest.raises(Exception, match="belongs to its bound exact consumer"):
        reservation.complete()
    builder.seal(producer.finish())
    reservation.release()


def test_builder_context_preserves_body_error_and_releases_zero_event_preflight():
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
            dataset_edge(stream, schema),
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


def test_exact_wrong_key_and_missing_cells_fail_without_acknowledging_delivery():
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
        dataset_edge(stream, schema),
    )
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    builder.consume(cursor.next())
    emit(producer, value(2), DatasetCellAddress(0, 0), 1)
    with pytest.raises(Exception, match="frozen plan key"):
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
        dataset_edge(missing_stream, schema),
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
        dataset_edge(stream, schema),
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


def test_keyed_monitor_cycle_clears_the_previous_sweep_before_point_zero():
    schema = dataset_schema(points=2)
    stream, producer = source(schema, events=4)
    tap = stream.monitor(max_events=4, max_bytes=48)
    builder = MonitorDataset.keyed_cycle(
        BlockId("rolling"),
        tap,
        dataset_edge(stream, schema),
    )

    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    first = builder.ingest_next()
    emit(producer, value(2), DatasetCellAddress(0, 1), 1)
    builder.ingest_next()
    emit(producer, value(3), DatasetCellAddress(0, 0), 2)
    builder.ingest_next()

    with pytest.raises(SnapshotExpired):
        builder.materialize(first)
    current = builder.materialize()
    assert current.block.values[0, 0, 0, 0] == 3
    assert np.count_nonzero(current.block.values[0, 1]) == 0
    assert current.block.validity.mask.tolist() == [[True, False]]
    assert not current.coverage.complete
    assert current.head.sequence == 2
    assert current.event_refs[0] == current.head
    assert current.event_refs[1] is None
    builder.close()


def test_keyed_monitor_gap_at_nonzero_offset_clears_every_stale_cell():
    schema = dataset_schema(points=3)
    stream, producer = source(schema, events=6)
    tap = stream.monitor(max_events=1, max_bytes=12)
    builder = MonitorDataset.keyed_cycle(
        BlockId("gap-cycle"),
        tap,
        dataset_edge(stream, schema),
    )

    for sequence in range(3):
        emit(
            producer,
            value(sequence + 1),
            DatasetCellAddress(0, sequence),
            sequence,
        )
        builder.ingest_next()
    assert builder.materialize().coverage.complete

    emit(producer, value(4), DatasetCellAddress(0, 0), 3)
    emit(producer, value(5), DatasetCellAddress(0, 1), 4)
    builder.ingest_latest()

    current = builder.materialize()
    assert current.block.values[0, :, 0, 0].tolist() == [0, 5, 0]
    assert current.block.validity.mask.tolist() == [[False, True, False]]
    assert current.event_refs[0] is None
    assert current.event_refs[1] == current.head
    assert current.event_refs[2] is None
    assert current.cell_metadata == (None, None, None)
    assert current.coverage.missed_events == 1
    assert not current.coverage.complete
    builder.close()


def test_append_window_owns_newest_first_order_not_producer_join_keys():
    source_schema = dataset_schema(points=2)
    window_schema = monitor_history_schema(source_schema, 3)
    stream, producer = source(source_schema, events=8)
    tap = stream.monitor(max_events=4, max_bytes=48)
    window = MonitorDataset.append_window(
        BlockId("history"),
        tap,
        dataset_edge(stream, window_schema, exact=False),
    )

    for sequence, number in enumerate((1, 2, 3, 4)):
        emit(
            producer,
            value(number),
            DatasetCellAddress(0, sequence % 2),
            sequence,
        )
        window.ingest_next()

    snapshot = window.materialize()
    assert snapshot.block.values.shape == (1, 3, 2, 3)
    assert snapshot.block.values[0, :, 0, 0].tolist() == [4, 3, 2]
    assert [reference.sequence for reference in snapshot.event_refs] == [3, 2, 1]
    assert snapshot.head == snapshot.event_refs[0]
    assert snapshot.coverage.complete
    assert snapshot.coverage.missed_events == 0
    window.close()


def test_append_window_gap_recovers_after_loss_rolls_out_of_visible_history():
    source_schema = dataset_schema(points=2)
    window_schema = monitor_history_schema(source_schema, 3)
    stream, producer = source(source_schema, events=8)
    tap = stream.monitor(max_events=1, max_bytes=12)
    window = MonitorDataset.append_window(
        BlockId("gap-recovery"),
        tap,
        dataset_edge(stream, window_schema, exact=False),
    )

    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    window.ingest_next()
    emit(producer, value(2), DatasetCellAddress(0, 1), 1)
    emit(producer, value(3), DatasetCellAddress(0, 0), 2)
    window.ingest_latest()
    with_gap = window.materialize()
    assert with_gap.coverage.missed_events == 1
    assert with_gap.coverage.current_gap
    assert not with_gap.coverage.complete

    for sequence, number in enumerate((4, 5, 6), start=3):
        emit(
            producer,
            value(number),
            DatasetCellAddress(0, sequence % 2),
            sequence,
        )
        window.ingest_next()
    recovered = window.materialize()
    assert recovered.block.values[0, :, 0, 0].tolist() == [6, 5, 4]
    assert recovered.coverage.complete
    assert not recovered.coverage.current_gap
    assert recovered.coverage.missed_events == 1
    window.close()


def test_append_window_preserves_every_data_axis_and_component_validity():
    source_schema = dataset_schema(points=1, component_validity=True)
    window_schema = monitor_history_schema(source_schema, 2)
    stream, producer = source(source_schema, events=3)
    tap = stream.monitor(max_events=3, max_bytes=39)
    window = MonitorDataset.append_window(
        BlockId("multidimensional-history"),
        tap,
        dataset_edge(stream, window_schema, exact=False),
    )
    x_axis = source_schema.cell_schema.data_axes[1]
    masks = (
        ComponentValidity((x_axis.axis_id,), np.array([True, False, True])),
        ComponentValidity((x_axis.axis_id,), np.array([False, True, True])),
        ComponentValidity((x_axis.axis_id,), np.array([True, True, False])),
    )
    for sequence, mask in enumerate(masks):
        emit(
            producer,
            value(sequence + 1, component_validity=mask),
            DatasetCellAddress(0, 0),
            sequence,
        )
        window.ingest_next()

    snapshot = window.materialize()
    assert snapshot.block.values.shape == (1, 2, 2, 3)
    assert snapshot.block.values[0, :, 0, 0].tolist() == [3, 2]
    assert snapshot.block.validity.axis_ids == tuple(
        axis.axis_id for axis in source_schema.cell_schema.data_axes
    )
    assert snapshot.block.validity.mask.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(
        snapshot.block.validity.mask[0, 0],
        np.broadcast_to(masks[2].mask, (2, 3)),
    )
    np.testing.assert_array_equal(
        snapshot.block.validity.mask[0, 1],
        np.broadcast_to(masks[1].mask, (2, 3)),
    )
    window.close()


def test_append_window_rejects_scan_axes_relabelled_as_history():
    schema = dataset_schema(points=2)
    stream, _producer = source(schema, events=2)
    tap = stream.monitor(max_events=1, max_bytes=12)

    with pytest.raises(Exception, match="MONITOR_HISTORY"):
        MonitorDataset.append_window(
            BlockId("not-history"),
            tap,
            dataset_edge(stream, schema, exact=False),
        )

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
        dataset_edge(stream_a, schema),
    )
    builder_b = DatasetBuilder(
        BlockId("authority-b"),
        reservation_b,
        dataset_edge(stream_b, schema),
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
    builder_b.abort()
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
        dataset_edge(stream, schema),
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
        dataset_edge(stream, schema),
    )
    payload = Value(value(7).values, VALID, schema.cell_schema)
    with pytest.raises(Exception, match="reserved formal run/source"):
        producer.emit(
            payload,
            captured_at=0.0,
            trace=TraceContext("another-run", "camera", "capture"),
            join_key=DatasetCellAddress(0, 0),
        )
    assert builder.revision.value == 0
    assert stream.next_sequence == 0
    builder.abort()
    reservation.release()


def test_monitor_materializer_is_the_tap_consumer_and_releases_it_on_close():
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)
    tap = stream.monitor(max_events=1, max_bytes=12)
    builder = MonitorDataset.keyed_cycle(
        BlockId("monitor-authority"),
        tap,
        dataset_edge(stream, schema),
    )
    with pytest.raises(PermissionError, match="another consumer"):
        MonitorDataset.keyed_cycle(BlockId("duplicate-owner"), tap, dataset_edge(stream, schema))
    emit(producer, value(9), DatasetCellAddress(0, 0), 0)
    with pytest.raises(PermissionError, match="another consumer"):
        tap.next()
    builder.ingest_next()
    builder.close()
    assert tap.retained_bytes == 0
    with pytest.raises(StreamEndedEarly, match="closed"):
        tap.latest()


def test_monitor_constructor_cannot_override_the_edge_cycle_schedule():
    schema = dataset_schema(points=2)
    stream, _producer = source(schema, events=2)
    tap = stream.monitor(max_events=1, max_bytes=12)
    edge = dataset_edge(stream, schema)

    with pytest.raises(TypeError, match="cycle_cells"):
        MonitorDataset(
            BlockId("no-second-cycle"),
            tap,
            edge,
            cycle_cells=tuple(reversed(edge.expected_cells)),
        )

    tap.close()


def test_monitor_claim_wakes_and_revokes_an_already_blocked_raw_reader(monkeypatch):
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)
    tap = stream.monitor(max_events=1, max_bytes=12)
    entered_wait = threading.Event()
    failures: list[BaseException] = []
    real_wait = tap._condition.wait

    def announced_wait(timeout=None):
        entered_wait.set()
        return real_wait(timeout)

    monkeypatch.setattr(tap._condition, "wait", announced_wait)

    def raw_read() -> None:
        try:
            tap.next(timeout=1.0)
        except BaseException as error:
            failures.append(error)

    reader = threading.Thread(target=raw_read)
    reader.start()
    assert entered_wait.wait(timeout=1.0)
    builder = MonitorDataset.keyed_cycle(
        BlockId("claimed-after-wait"),
        tap,
        dataset_edge(stream, schema),
    )
    reader.join(timeout=1.0)

    assert not reader.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], PermissionError)
    emit(producer, value(9), DatasetCellAddress(0, 0), 0)
    builder.ingest_next(timeout=1.0)
    assert builder.materialize().block.values[0, 0, 0, 0] == 9
    builder.close()


def test_monitor_materializer_must_bind_before_first_publication():
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)
    tap = stream.monitor(max_events=1, max_bytes=12)
    emit(producer, value(3), DatasetCellAddress(0, 0), 0)

    with pytest.raises(ReservationStateError, match="before the first publication"):
        MonitorDataset.keyed_cycle(
            BlockId("late-monitor-owner"),
            tap,
            dataset_edge(stream, schema),
        )

    assert tap.next().envelope.sequence == 0
    tap.close()


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
            dataset_edge(stream, schema),
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


def test_typed_event_adapter_seals_image_and_metadata_in_one_delivery():
    schema = dataset_schema(points=1)

    @dataclass(frozen=True)
    class FrameMetadata:
        physical_ordinal: int
        frame_stamp: int

    @dataclass(frozen=True)
    class CameraSample:
        image: Value
        metadata: FrameMetadata

    @dataclass(frozen=True)
    class CameraSampleContract:
        schema: ValueSchema
        fingerprint: str = "4" * 64

        @property
        def max_retained_nbytes(self):
            return int(np.prod(self.schema.data_shape)) * self.schema.dtype.itemsize + 16

        def snapshot(self, payload):
            self.validate(payload)
            return payload

        def validate(self, payload):
            if not isinstance(payload, CameraSample) or payload.image.schema is not self.schema:
                raise TypeError("invalid CameraSample")

        def retained_nbytes(self, payload):
            self.validate(payload)
            return payload.image.values.nbytes + 16

        def digest(self, payload):
            self.validate(payload)
            value_digest = ValuePayloadContract(self.schema).digest(payload.image)
            metadata = payload.metadata
            metadata_digest = hashlib.sha256(
                f"{metadata.physical_ordinal}:{metadata.frame_stamp}".encode("ascii")
            ).hexdigest()
            return hashlib.sha256(
                f"{value_digest}:{metadata_digest}".encode("ascii")
            ).hexdigest()

    @dataclass(frozen=True)
    class FrameMetadataContract:
        fingerprint: str = "5" * 64
        max_retained_nbytes: int = 16

        @staticmethod
        def snapshot(payload):
            return payload.metadata

        @staticmethod
        def validate(metadata):
            if not isinstance(metadata, FrameMetadata):
                raise TypeError("invalid FrameMetadata")

        @staticmethod
        def retained_nbytes(metadata):
            FrameMetadataContract.validate(metadata)
            return 16

        @staticmethod
        def digest(metadata):
            FrameMetadataContract.validate(metadata)
            encoded = f"{metadata.physical_ordinal}:{metadata.frame_stamp}".encode("ascii")
            return hashlib.sha256(encoded).hexdigest()

    @dataclass(frozen=True)
    class CameraSampleAdapter:
        payload_contract: CameraSampleContract
        metadata_contract: FrameMetadataContract = FrameMetadataContract()
        operator_fingerprint: str = "a" * 64

        @property
        def value_schema(self):
            return self.payload_contract.schema

        @staticmethod
        def value(payload):
            return payload.image

    contract = CameraSampleContract(schema.cell_schema)
    stream, producer = AcquisitionStream.create(
        StreamId("camera.samples"),
        contract,
        flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
        retention_events=1,
        retention_bytes=contract.max_retained_nbytes,
        join_key_contract=DatasetCellKeyContract(schema),
    )
    reservation = stream.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=contract.max_retained_nbytes,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("camera-sample"),
        reservation,
        dataset_edge(
            stream,
            schema,
            adapter=CameraSampleAdapter(contract),
        ),
    )
    metadata = FrameMetadata(physical_ordinal=0, frame_stamp=101)
    sample = CameraSample(Value(value(5).values, VALID, schema.cell_schema), metadata)
    producer.emit(
        sample,
        captured_at=1.0,
        trace=TraceContext("run", "camera", "capture"),
        join_key=DatasetCellAddress(0, 0),
    )
    builder.consume(cursor.next())
    assert builder.materialize().cell_metadata == (metadata,)
    artifact = builder.seal(producer.finish())
    assert artifact.block.values[0, 0, 0, 0] == 5
    assert artifact.event_metadata == (metadata,)
    assert len(artifact.provenance.ordered_metadata_digest) == 64
    reservation.release()


@pytest.mark.parametrize(
    "metadata",
    ([1], {"scale": 1}, {1}, np.asarray([1], dtype=np.int64)),
)
def test_runtime_metadata_rejects_mutable_aliases(metadata):
    assert not runtime_dataset._is_deeply_immutable(metadata)




def test_metadata_rejects_enum_with_mutable_value():
    class MutableEnum(Enum):
        ITEM = []

    assert MutableEnum.ITEM.value == []

    @dataclass(frozen=True)
    class EnumMetadataContract:
        fingerprint: str = "b" * 64
        max_retained_nbytes: int = 0

        @staticmethod
        def snapshot(_payload):
            return MutableEnum.ITEM

        @staticmethod
        def validate(_metadata):
            return None

        @staticmethod
        def retained_nbytes(_metadata):
            return 0

        @staticmethod
        def digest(_metadata):
            return "c" * 64

    @dataclass(frozen=True)
    class EnumMetadataAdapter:
        payload_contract: ValuePayloadContract
        metadata_contract: EnumMetadataContract = EnumMetadataContract()
        operator_fingerprint: str = "d" * 64

        @property
        def value_schema(self):
            return self.payload_contract.schema

        def value(self, payload):
            self.payload_contract.validate(payload)
            return payload

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
        BlockId("enum-metadata"),
        reservation,
        dataset_edge(
            stream,
            schema,
            adapter=EnumMetadataAdapter(stream._payload_contract),
        ),
    )
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    with pytest.raises(TypeError, match="deeply immutable"):
        builder.consume(cursor.next())
    builder.abort()
    reservation.release()


def test_runtime_metadata_accepts_owned_value_as_a_validated_leaf(monkeypatch):
    schema = image_value_schema()
    safe_value = Value(
        np.arange(6, dtype=np.uint16).reshape(2, 3),
        VALID,
        schema,
    )

    @dataclass(frozen=True)
    class OccupancyLikeMetadata:
        value: Value
        labels: tuple[str, ...]

    dataclass_fields = runtime_dataset.fields
    traversed_types = []

    def tracked_fields(value):
        traversed_types.append(type(value))
        return dataclass_fields(value)

    monkeypatch.setattr(runtime_dataset, "fields", tracked_fields)
    assert runtime_dataset._is_deeply_immutable(
        OccupancyLikeMetadata(safe_value, ("occupied", "empty"))
    )
    assert traversed_types == [OccupancyLikeMetadata]
    bytes_backed = np.frombuffer(
        np.arange(6, dtype=np.int64).tobytes(),
        dtype=np.dtype("<i8"),
    )
    assert runtime_dataset._is_deeply_immutable(bytes_backed)

    owning = np.arange(6, dtype=np.int64)
    owning.setflags(write=False)
    assert not runtime_dataset._is_deeply_immutable(owning)
    mutable_owner = np.arange(6, dtype=np.int64)
    readonly_alias = mutable_owner.view()
    readonly_alias.setflags(write=False)
    assert not runtime_dataset._is_deeply_immutable(readonly_alias)
