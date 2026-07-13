"""DatasetBuilder contracts over exact and monitor stream deliveries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from enum import Enum

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
    DatasetCellDomain,
    DatasetCellKeyContract,
    DatasetMode,
    DatasetPreviewSnapshot,
    DuplicateDatasetCell,
    MissingDatasetCells,
    SnapshotExpired,
    SealedDatasetArtifact,
    ValueDatasetEventAdapter,
    dataset_cell_key_fingerprint,
    dataset_cell_permutation_digest,
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


def event_adapter(stream) -> ValueDatasetEventAdapter:
    return ValueDatasetEventAdapter(stream._payload_contract)


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
        schema,
        DatasetMode.FINITE_EXACT,
        event_adapter=event_adapter(stream),
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
        event_adapter=event_adapter(stream),
        expected_cells=cell_schedule(schema),
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
            event_adapter=event_adapter(stream),
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
        event_adapter=event_adapter(stream),
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
        event_adapter=event_adapter(missing_stream),
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
        event_adapter=event_adapter(stream),
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
        event_adapter=event_adapter(stream),
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
        event_adapter=event_adapter(stream_a),
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
        event_adapter=event_adapter(stream),
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
        event_adapter=event_adapter(stream),
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
        BlockId("monitor-authority"),
        tap_a,
        schema,
        DatasetMode.ROLLING_MONITOR,
        event_adapter=event_adapter(stream_a),
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
            event_adapter=event_adapter(stream),
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
        schema,
        DatasetMode.FINITE_EXACT,
        event_adapter=CameraSampleAdapter(contract),
        expected_cells=cell_schedule(schema),
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


def test_metadata_contract_cannot_seal_a_mutable_alias():
    schema = dataset_schema(points=1)
    stream, producer = source(schema, events=1)

    @dataclass(frozen=True)
    class MutableMetadataContract:
        shared: dict
        fingerprint: str = "6" * 64
        max_retained_nbytes: int = 0

        def snapshot(self, _payload):
            return self.shared

        @staticmethod
        def validate(metadata):
            if not isinstance(metadata, dict):
                raise TypeError("metadata must be dict")

        @staticmethod
        def retained_nbytes(_metadata):
            return 0

        @staticmethod
        def digest(_metadata):
            return "7" * 64

    @dataclass(frozen=True)
    class MutableMetadataAdapter:
        payload_contract: ValuePayloadContract
        metadata_contract: MutableMetadataContract
        operator_fingerprint: str = "b" * 64

        @property
        def value_schema(self):
            return self.payload_contract.schema

        def value(self, payload):
            self.payload_contract.validate(payload)
            return payload

    reservation = stream.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=12,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    with pytest.raises(Exception, match="worst-case bytes"):
        DatasetBuilder(
            BlockId("underbudget-metadata"),
            reservation,
            schema,
            DatasetMode.FINITE_EXACT,
            event_adapter=MutableMetadataAdapter(
                stream._payload_contract,
                MutableMetadataContract(
                    {"frame": 0},
                    max_retained_nbytes=1,
                ),
            ),
            expected_cells=cell_schedule(schema),
        )
    builder = DatasetBuilder(
        BlockId("mutable-metadata"),
        reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        event_adapter=MutableMetadataAdapter(
            stream._payload_contract,
            MutableMetadataContract({"frame": 0}),
        ),
        expected_cells=cell_schedule(schema),
    )
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    delivery = cursor.next()
    with pytest.raises(TypeError, match="deeply immutable"):
        builder.consume(delivery)
    assert not delivery.acknowledged
    assert builder.revision.value == 0
    builder.abort()
    reservation.release()


def test_builder_freezes_one_metadata_contract_identity_for_the_generation():
    schema = dataset_schema(points=2)

    @dataclass(frozen=True)
    class BudgetedValueContract:
        schema: ValueSchema
        fingerprint: str = "8" * 64
        max_retained_nbytes: int = 20

        def snapshot(self, payload):
            self.validate(payload)
            return payload

        def validate(self, payload):
            if not isinstance(payload, Value) or payload.schema is not self.schema:
                raise TypeError("invalid Value")

        def retained_nbytes(self, payload):
            self.validate(payload)
            return payload.values.nbytes + 8

    @dataclass(frozen=True)
    class SwitchingMetadataContract:
        fingerprint: str
        as_text: bool
        max_retained_nbytes: int = 8

        def snapshot(self, payload):
            number = int(payload.values[0, 0])
            return str(number) if self.as_text else number

        @staticmethod
        def validate(metadata):
            if type(metadata) not in (int, str):
                raise TypeError("metadata must be canonical int or text")

        @staticmethod
        def retained_nbytes(_metadata):
            return 8

        @staticmethod
        def digest(metadata):
            return hashlib.sha256(str(metadata).encode("ascii")).hexdigest()

    @dataclass(frozen=True)
    class SwitchingAdapter:
        payload_contract: BudgetedValueContract
        metadata_contract: SwitchingMetadataContract
        operator_fingerprint: str = "c" * 64

        @property
        def value_schema(self):
            return self.payload_contract.schema

        def value(self, payload):
            self.payload_contract.validate(payload)
            return payload

    payload_contract = BudgetedValueContract(schema.cell_schema)
    first_contract = SwitchingMetadataContract("9" * 64, False)
    adapter = SwitchingAdapter(payload_contract, first_contract)
    stream, producer = AcquisitionStream.create(
        StreamId("metadata-contract-identity"),
        payload_contract,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=2,
        retention_bytes=40,
        join_key_contract=DatasetCellKeyContract(schema),
    )
    reservation = stream.reserve(
        total_events=2,
        max_inflight_events=2,
        max_inflight_bytes=40,
        trace_binding=TRACE_BINDING,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("metadata-contract-identity"),
        reservation,
        schema,
        DatasetMode.FINITE_EXACT,
        event_adapter=adapter,
        expected_cells=cell_schedule(schema),
    )
    for point, number in enumerate((1, 2)):
        if point == 1:
            object.__setattr__(
                adapter,
                "metadata_contract",
                SwitchingMetadataContract("a" * 64, True),
            )
        payload = Value(
            np.full((2, 3), number, dtype=np.uint16),
            VALID,
            schema.cell_schema,
        )
        producer.emit(
            payload,
            captured_at=float(point),
            trace=TraceContext("run", "camera", f"capture-{point}"),
            join_key=DatasetCellAddress(0, point),
        )
        builder.consume(cursor.next())
    artifact = builder.seal(producer.finish())
    assert artifact.event_metadata == (1, 2)
    assert artifact.provenance.metadata_contract_fingerprint == "9" * 64
    reservation.release()


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
        schema,
        DatasetMode.FINITE_EXACT,
        event_adapter=EnumMetadataAdapter(stream._payload_contract),
        expected_cells=cell_schedule(schema),
    )
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    with pytest.raises(TypeError, match="deeply immutable"):
        builder.consume(cursor.next())
    builder.abort()
    reservation.release()
