"""DatasetBuilder contracts over exact and monitor stream deliveries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from zlc_data.axis import (
    AxisId,
    AxisSpec,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_data.schema import DatasetSchema, PointColumn, PointTable, ValueSchema
from zlc_data.validity import ComponentValidity, VALID, ValidityContract
from zlc_data.value import (
    BlockId,
    DatasetRevision,
    StreamGenerationId,
    Value,
    ValuePayloadContract,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetCellSchedule,
    DatasetCellKeyContract,
    DatasetCoverage,
    DatasetPreviewDelta,
    DatasetPreviewSnapshot,
    FrozenDatasetEdge,
    MissingDatasetCells,
    MonitorCoverage,
    MonitorDataset,
    SnapshotExpired,
    SealedDatasetArtifact,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ReservationStateError,
    StreamEndedEarly,
    StreamId,
)


def test_formal_and_monitor_coverage_have_distinct_loss_semantics():
    assert DatasetCoverage(1, 1).complete
    assert not hasattr(DatasetCoverage(1, 1), "missed_events")
    assert MonitorCoverage(1, 1, 1, current_gap=False).complete
    assert not MonitorCoverage(1, 1, 1, current_gap=True).complete


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


_IMAGE_VALUE_SCHEMA = image_value_schema()
_COMPONENT_IMAGE_VALUE_SCHEMA = image_value_schema(component_validity=True)


def dataset_schema(
    *,
    repeats: int = 1,
    points: int = 3,
    component_validity: bool = False,
    cell_schema: ValueSchema | None = None,
):
    detuning = axis("detuning", SCAN_POINT, points)
    return DatasetSchema(
        axis("repeat", REPEAT, repeats),
        PointTable(
            points,
            (
                PointColumn(
                    detuning.axis_id,
                    detuning.name,
                    detuning.role,
                    PointColumn.NUMERIC,
                    detuning.coordinates,
                ),
            ),
        ),
        None,
        cell_schema
        if cell_schema is not None
        else (
            _COMPONENT_IMAGE_VALUE_SCHEMA
            if component_validity
            else _IMAGE_VALUE_SCHEMA
        ),
    )


def monitor_latest_schema(source_schema: DatasetSchema) -> DatasetSchema:
    return DatasetSchema(
        axis("repeat", REPEAT, 1),
        PointTable(1),
        None,
        source_schema.cell_schema,
    )


def value(number: int, *, component_validity=None) -> Value:
    schema = (
        _COMPONENT_IMAGE_VALUE_SCHEMA
        if component_validity is not None
        else _IMAGE_VALUE_SCHEMA
    )
    validity = VALID if component_validity is None else component_validity
    return Value(np.full((2, 3), number, dtype=np.uint16), validity, schema)


def cell_schedule(schema: DatasetSchema) -> DatasetCellSchedule:
    return DatasetCellSchedule.from_cells(
        schema,
        (
            DatasetCellAddress(repeat, point)
            for repeat in range(schema.repeat_axis.size)
            for point in range(schema.point_table.row_count)
        ),
    )


@dataclass(frozen=True)
class _NoDatasetMetadataContract:
    @staticmethod
    def snapshot(_payload):
        return None

    @staticmethod
    def validate(metadata):
        if metadata is not None:
            raise TypeError("value event has no metadata")

@dataclass(frozen=True)
class _ValueDatasetEventAdapter:
    payload_contract: ValuePayloadContract
    metadata_contract: _NoDatasetMetadataContract = _NoDatasetMetadataContract()

    @property
    def value_schema(self):
        return self.payload_contract.schema

    def value(self, payload):
        self.payload_contract.validate(payload)
        return payload


def source(schema: DatasetSchema):
    contract = ValuePayloadContract(schema.cell_schema)
    stream, producer = AcquisitionStream.create(
        StreamId("camera.frames"),
        contract,
        join_key_contract=DatasetCellKeyContract.from_schema(schema),
    )
    return stream, producer, contract


def emit(producer, payload, address: DatasetCellAddress, sequence_value: int):
    return producer.emit(
        payload,
        captured_at=float(sequence_value),
        join_key=address,
    )


def event_adapter(payload_contract: ValuePayloadContract) -> _ValueDatasetEventAdapter:
    return _ValueDatasetEventAdapter(payload_contract)


def dataset_edge(
    payload_contract: ValuePayloadContract,
    schema: DatasetSchema,
    *,
    exact: bool = True,
    adapter=None,
) -> FrozenDatasetEdge:
    return FrozenDatasetEdge(
        schema,
        event_adapter(payload_contract) if adapter is None else adapter,
        cell_schedule(schema) if exact else None,
    )


def test_cell_key_and_schedule_depend_only_on_repeat_and_point_rows():
    first = dataset_schema(points=2)
    alternate_cell = ValueSchema(
        first.cell_schema.data_axes,
        first.cell_schema.validity_contract,
        np.dtype("<f4"),
        value_unit="probability",
    )
    second = DatasetSchema(
        first.repeat_axis,
        first.point_table,
        first.grid_topology,
        alternate_cell,
    )
    cells = cell_schedule(first)
    assert first.fingerprint != second.fingerprint
    assert DatasetCellKeyContract.from_schema(first) == (
        DatasetCellKeyContract.from_schema(second)
    )
    cells.validate_schema(second)


def test_frozen_edge_rejects_normally_mutable_adapter_configuration():
    schema = dataset_schema(points=1)

    class HashableMutableScale:
        __hash__ = object.__hash__

        def __init__(self, factor: int) -> None:
            self.factor = factor

    @dataclass(frozen=True)
    class MutableAdapter:
        payload_contract: ValuePayloadContract
        scale: HashableMutableScale
        metadata_contract: object = _ValueDatasetEventAdapter(
            ValuePayloadContract(image_value_schema())
        ).metadata_contract

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
            MutableAdapter(
                ValuePayloadContract(schema.cell_schema),
                HashableMutableScale(1),
            ),
            cell_schedule(schema),
        )


def test_metadata_contract_validation_precedes_exact_commit(monkeypatch):
    schema = dataset_schema(points=1)
    stream, producer, payload_contract = source(schema)
    edge = dataset_edge(payload_contract, schema)

    def reject(_metadata):
        raise ValueError("semantic metadata rejection")

    monkeypatch.setattr(type(edge.metadata_contract), "validate", staticmethod(reject))
    reservation = stream.reserve(
        total_events=1,
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


def test_frozen_edge_projects_one_prevalidated_exact_schedule():
    schema = dataset_schema(repeats=2, points=3)
    schedule = cell_schedule(schema)
    edge = FrozenDatasetEdge(
        schema,
        event_adapter(ValuePayloadContract(schema.cell_schema)),
        schedule,
    )

    assert edge.cell_schedule is schedule
    assert edge.key_contract == DatasetCellKeyContract.from_schema(schema)
    assert tuple(edge.cell_schedule) == tuple(schedule)


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
    with pytest.raises(error):
        DatasetCellSchedule.from_cells(schema, schedule)


def test_cell_domain_rejects_non_repeat_repeat_axis():
    point = axis("point", SCAN_POINT, 1)
    with pytest.raises(ValueError, match="role 'repeat'"):
        DatasetCellKeyContract(
            axis("not-repeat", SCAN_POINT, 1),
            PointTable(
                1,
                (
                    PointColumn(
                        point.axis_id,
                        point.name,
                        point.role,
                        PointColumn.NUMERIC,
                        point.coordinates,
                    ),
                ),
            ),
        )


def test_exact_builder_preserves_all_named_data_axes_and_snapshot_revisions():
    schema = dataset_schema(repeats=1, points=3)
    stream, producer, payload_contract = source(schema)
    reservation = stream.reserve(
        total_events=3,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("capture"),
        reservation,
        dataset_edge(payload_contract, schema),
    )

    emit(producer, value(10), DatasetCellAddress(0, 0), 0)
    assert builder.consume(cursor.next()) is None
    first = builder.materialize()
    first_ref = first.ref
    assert isinstance(first, DatasetPreviewSnapshot)
    assert first.block.values.shape == (1, 3, 2, 3)
    assert np.all(first.block.values[0, 0] == 10)
    assert first.coverage.written_cells == 1

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
    assert stream.retained_events == 0


def test_exact_preview_delta_copies_only_new_cells_in_frozen_order():
    schema = dataset_schema(points=2, component_validity=True)
    stream, producer, payload_contract = source(schema)
    reservation = stream.reserve(
        total_events=2,
    )
    cursor = reservation.activate()
    schedule = DatasetCellSchedule.from_cells(
        schema,
        (DatasetCellAddress(0, 1), DatasetCellAddress(0, 0)),
    )
    builder = DatasetBuilder(
        BlockId("delta-camera"),
        reservation,
        FrozenDatasetEdge(schema, event_adapter(payload_contract), schedule),
    )
    reader = builder.open_preview_reader()
    mask = np.asarray(
        ((True, False, True), (False, True, False)),
        dtype=bool,
    )
    validity = ComponentValidity(
        tuple(axis.axis_id for axis in schema.cell_schema.data_axes),
        mask,
    )

    emit(producer, value(11, component_validity=validity), schedule.cell_at(0), 0)
    builder.consume(cursor.next())
    emit(producer, value(22, component_validity=validity), schedule.cell_at(1), 1)
    builder.consume(cursor.next())
    first = reader.freeze_delta(DatasetRevision(0))

    assert isinstance(first, DatasetPreviewDelta)
    assert first.after == DatasetRevision(0)
    assert first.ref.revision == DatasetRevision(1)
    assert tuple(cell.ordinal for cell in first.cells) == (0,)
    assert tuple(cell.address for cell in first.cells) == (DatasetCellAddress(0, 1),)
    assert np.all(first.cells[0].value.values == 11)
    assert np.array_equal(first.cells[0].value.validity.mask, mask)
    assert first.cells[0].metadata is None
    with pytest.raises(ValueError):
        first.cells[0].value.values.setflags(write=True)

    second = reader.freeze_delta(first.ref.revision)

    assert second.after == DatasetRevision(1)
    assert second.ref.revision == DatasetRevision(2)
    assert tuple(cell.ordinal for cell in second.cells) == (1,)
    assert tuple(cell.address for cell in second.cells) == (DatasetCellAddress(0, 0),)
    assert np.all(second.cells[0].value.values == 22)
    assert reader.freeze_delta(second.ref.revision).cells == ()

    builder.seal(producer.finish())
    reservation.release()

def test_bound_builder_owns_reservation_completion():
    schema = dataset_schema(points=1)
    stream, producer, payload_contract = source(schema)
    reservation = stream.reserve(
        total_events=1,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("completion-owner"),
        reservation,
        dataset_edge(payload_contract, schema),
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
    stream, _producer, payload_contract = source(schema)
    reservation = stream.reserve(
        total_events=1,
    )
    reservation.activate()
    with pytest.raises(RuntimeError, match="body failure"):
        with DatasetBuilder(
            BlockId("context-owner"),
            reservation,
            dataset_edge(payload_contract, schema),
        ):
            raise RuntimeError("body failure")
    replacement = stream.reserve(
        total_events=1,
    )
    replacement.abort()
    replacement.release()


def test_exact_wrong_key_and_missing_cells_fail_without_acknowledging_delivery():
    schema = dataset_schema(points=2)
    stream, producer, payload_contract = source(schema)
    reservation = stream.reserve(
        total_events=2,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("duplicate"),
        reservation,
        dataset_edge(payload_contract, schema),
    )
    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    builder.consume(cursor.next())
    emit(producer, value(2), DatasetCellAddress(0, 0), 1)
    with pytest.raises(Exception, match="frozen plan key"):
        builder.consume(cursor.next())
    assert cursor.next_sequence == 1
    builder.abort()
    reservation.release()

    missing_stream, missing_producer, missing_payload_contract = source(schema)
    missing_reservation = missing_stream.reserve(
        total_events=2,
    )
    missing_cursor = missing_reservation.activate()
    missing_builder = DatasetBuilder(
        BlockId("missing"),
        missing_reservation,
        dataset_edge(missing_payload_contract, schema),
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
    stream, producer, payload_contract = source(schema)
    reservation = stream.reserve(
        total_events=1,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("component-validity"),
        reservation,
        dataset_edge(payload_contract, schema),
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
    stream, producer, payload_contract = source(schema)
    tap = stream.monitor()
    builder = MonitorDataset.keyed_cycle(
        BlockId("rolling"),
        tap,
        dataset_edge(payload_contract, schema),
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
    stream, producer, payload_contract = source(schema)
    tap = stream.monitor()
    builder = MonitorDataset.keyed_cycle(
        BlockId("gap-cycle"),
        tap,
        dataset_edge(payload_contract, schema),
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


def test_latest_cell_replaces_the_front_without_exposing_source_join_keys():
    source_schema = dataset_schema(points=2)
    latest_schema = monitor_latest_schema(source_schema)
    stream, producer, payload_contract = source(source_schema)
    tap = stream.monitor()
    latest = MonitorDataset.latest_cell(
        BlockId("latest"),
        tap,
        dataset_edge(payload_contract, latest_schema, exact=False),
    )

    for sequence, number in enumerate((1, 2, 3, 4)):
        emit(
            producer,
            value(number),
            DatasetCellAddress(0, sequence % 2),
            sequence,
        )
        latest.ingest_next()

    snapshot = latest.materialize()
    assert snapshot.block.values.shape == (1, 1, 2, 3)
    assert snapshot.block.values[0, 0, 0, 0] == 4
    assert tuple(reference.sequence for reference in snapshot.event_refs) == (3,)
    assert snapshot.head == snapshot.event_refs[0]
    assert snapshot.ref.revision.value == 4
    assert snapshot.coverage.complete
    assert snapshot.coverage.missed_events == 0
    latest.close()


def test_latest_cell_accounts_for_a_gap_and_clears_current_gap_on_contiguous_ingest():
    source_schema = dataset_schema(points=2)
    latest_schema = monitor_latest_schema(source_schema)
    stream, producer, payload_contract = source(source_schema)
    tap = stream.monitor()
    latest = MonitorDataset.latest_cell(
        BlockId("gap-recovery"),
        tap,
        dataset_edge(payload_contract, latest_schema, exact=False),
    )

    emit(producer, value(1), DatasetCellAddress(0, 0), 0)
    latest.ingest_next()
    emit(producer, value(2), DatasetCellAddress(0, 1), 1)
    emit(producer, value(3), DatasetCellAddress(0, 0), 2)
    latest.ingest_latest()
    with_gap = latest.materialize()
    assert with_gap.coverage.missed_events == 1
    assert with_gap.coverage.current_gap
    assert not with_gap.coverage.complete
    assert with_gap.block.values[0, 0, 0, 0] == 3

    emit(producer, value(4), DatasetCellAddress(0, 1), 3)
    latest.ingest_next()
    recovered = latest.materialize()
    assert recovered.block.values[0, 0, 0, 0] == 4
    assert recovered.coverage.complete
    assert not recovered.coverage.current_gap
    assert recovered.coverage.missed_events == 1
    latest.close()


def test_latest_cell_preserves_every_data_axis_dtype_and_component_validity():
    source_schema = dataset_schema(points=1, component_validity=True)
    latest_schema = monitor_latest_schema(source_schema)
    stream, producer, payload_contract = source(source_schema)
    tap = stream.monitor()
    latest = MonitorDataset.latest_cell(
        BlockId("multidimensional-latest"),
        tap,
        dataset_edge(payload_contract, latest_schema, exact=False),
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
        latest.ingest_next()

    snapshot = latest.materialize()
    assert snapshot.block.values.shape == (1, 1, 2, 3)
    assert snapshot.block.values.dtype == np.dtype("<u2")
    assert snapshot.block.values[0, 0, 0, 0] == 3
    assert snapshot.block.validity.axis_ids == tuple(
        axis.axis_id for axis in source_schema.cell_schema.data_axes
    )
    assert snapshot.block.validity.mask.shape == (1, 1, 2, 3)
    np.testing.assert_array_equal(
        snapshot.block.validity.mask[0, 0],
        np.broadcast_to(masks[2].mask, (2, 3)),
    )
    latest.close()


def test_latest_cell_rejects_noncanonical_point_storage():
    schema = dataset_schema(points=2)
    stream, _producer, payload_contract = source(schema)
    tap = stream.monitor()

    with pytest.raises(Exception, match="single-cell"):
        MonitorDataset.latest_cell(
            BlockId("not-latest"),
            tap,
            dataset_edge(payload_contract, schema, exact=False),
        )

    tap.close()


def test_latest_cell_replacement_is_invisible_until_matching_envelope_commits():
    source_schema = dataset_schema(points=1)
    latest_schema = monitor_latest_schema(source_schema)
    stream, producer, payload_contract = source(source_schema)
    latest = MonitorDataset.latest_cell(
        BlockId("atomic-latest"),
        stream.monitor(),
        dataset_edge(payload_contract, latest_schema, exact=False),
    )
    first = value(1)
    emit(producer, first, DatasetCellAddress(0, 0), 0)
    latest.ingest_next()
    before = latest.freeze_current()

    replacement_value = value(9)
    replacement = latest.prepare_latest_cell_replacement(replacement_value)
    staged = latest.freeze_current()
    assert staged.ref == before.ref
    assert staged.block.values[0, 0, 0, 0] == 1

    envelope = emit(
        producer,
        replacement_value,
        DatasetCellAddress(0, 0),
        1,
    )
    latest.commit_latest_cell_replacement(replacement, envelope)
    committed = latest.freeze_current()
    assert committed.ref.revision.value == before.ref.revision.value + 1
    assert committed.block.values[0, 0, 0, 0] == 9
    assert committed.event_refs == (envelope.ref,)
    assert committed.head == envelope.ref
    latest.close()


def test_exact_delivery_cannot_cross_source_authority_even_when_ids_match():
    schema = dataset_schema(points=1)
    stream_a, producer_a, payload_contract_a = source(schema)
    stream_b, producer_b, payload_contract_b = source(schema)
    reservation_a = stream_a.reserve(
        total_events=1,
    )
    reservation_b = stream_b.reserve(
        total_events=1,
    )
    cursor_a = reservation_a.activate()
    cursor_b = reservation_b.activate()
    builder = DatasetBuilder(
        BlockId("authority"),
        reservation_a,
        dataset_edge(payload_contract_a, schema),
    )
    builder_b = DatasetBuilder(
        BlockId("authority-b"),
        reservation_b,
        dataset_edge(payload_contract_b, schema),
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
    stream, producer, payload_contract = source(schema)
    reservation = stream.reserve(
        total_events=2,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("schedule"),
        reservation,
        dataset_edge(payload_contract, schema),
    )
    emit(producer, value(10), DatasetCellAddress(0, 1), 0)
    delivery = cursor.next()
    with pytest.raises(Exception, match="frozen plan key"):
        builder.consume(delivery)
    assert not delivery.acknowledged
    assert builder.revision.value == 0
    builder.abort()
    reservation.release()


def test_monitor_materializer_is_the_tap_consumer_and_releases_it_on_close():
    schema = dataset_schema(points=1)
    stream, producer, payload_contract = source(schema)
    tap = stream.monitor()
    builder = MonitorDataset.keyed_cycle(
        BlockId("monitor-authority"),
        tap,
        dataset_edge(payload_contract, schema),
    )
    with pytest.raises(PermissionError, match="another consumer"):
        MonitorDataset.keyed_cycle(
            BlockId("duplicate-owner"),
            tap,
            dataset_edge(payload_contract, schema),
        )
    emit(producer, value(9), DatasetCellAddress(0, 0), 0)
    with pytest.raises(PermissionError, match="another consumer"):
        tap.next()
    builder.ingest_next()
    builder.close()
    with pytest.raises(StreamEndedEarly, match="closed"):
        tap.latest()


def test_monitor_constructor_cannot_override_the_edge_cycle_schedule():
    schema = dataset_schema(points=2)
    stream, _producer, payload_contract = source(schema)
    tap = stream.monitor()
    edge = dataset_edge(payload_contract, schema)

    with pytest.raises(TypeError, match="cycle_cells"):
        MonitorDataset(
            BlockId("no-second-cycle"),
            tap,
            edge,
            cycle_cells=tuple(reversed(tuple(edge.cell_schedule))),
        )

    tap.close()


def test_monitor_materializer_must_bind_before_first_publication():
    schema = dataset_schema(points=1)
    stream, producer, payload_contract = source(schema)
    tap = stream.monitor()
    emit(producer, value(3), DatasetCellAddress(0, 0), 0)

    with pytest.raises(ReservationStateError, match="before the first publication"):
        MonitorDataset.keyed_cycle(
            BlockId("late-monitor-owner"),
            tap,
            dataset_edge(payload_contract, schema),
        )

    assert tap.next().envelope.sequence == 0
    tap.close()


def test_value_payload_contract_requires_generation_owned_schema_identity():
    schema = dataset_schema(points=1)
    _stream, producer, _contract = source(schema)
    equal_but_distinct = image_value_schema()
    assert equal_but_distinct.fingerprint == schema.cell_schema.fingerprint
    payload = Value(np.ones((2, 3), dtype=np.uint16), VALID, equal_but_distinct)
    with pytest.raises(TypeError, match="generation-owned"):
        producer.emit(
            payload,
            captured_at=0.0,
            join_key=DatasetCellAddress(0, 0),
        )


def test_minted_generation_prevents_revision_ref_collision_across_sources():
    schema = dataset_schema(points=1)

    def capture(number: int):
        stream, producer, payload_contract = source(schema)
        reservation = stream.reserve(
            total_events=1,
        )
        cursor = reservation.activate()
        builder = DatasetBuilder(
            BlockId("same-block-label"),
            reservation,
            dataset_edge(payload_contract, schema),
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

        def snapshot(self, payload):
            self.validate(payload)
            return payload

        def validate(self, payload):
            if not isinstance(payload, CameraSample) or payload.image.schema is not self.schema:
                raise TypeError("invalid CameraSample")

    @dataclass(frozen=True)
    class FrameMetadataContract:
        @staticmethod
        def snapshot(payload):
            return payload.metadata

        @staticmethod
        def validate(metadata):
            if not isinstance(metadata, FrameMetadata):
                raise TypeError("invalid FrameMetadata")

    @dataclass(frozen=True)
    class CameraSampleAdapter:
        payload_contract: CameraSampleContract
        metadata_contract: FrameMetadataContract = FrameMetadataContract()

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
        join_key_contract=DatasetCellKeyContract.from_schema(schema),
    )
    reservation = stream.reserve(
        total_events=1,
    )
    cursor = reservation.activate()
    builder = DatasetBuilder(
        BlockId("camera-sample"),
        reservation,
        dataset_edge(
            contract,
            schema,
            adapter=CameraSampleAdapter(contract),
        ),
    )
    reader = builder.open_preview_reader()
    metadata = FrameMetadata(physical_ordinal=0, frame_stamp=101)
    sample = CameraSample(Value(value(5).values, VALID, schema.cell_schema), metadata)
    producer.emit(
        sample,
        captured_at=1.0,
        join_key=DatasetCellAddress(0, 0),
    )
    builder.consume(cursor.next())
    assert builder.materialize().cell_metadata == (metadata,)
    delta = reader.freeze_delta(DatasetRevision(0))
    assert tuple(cell.metadata for cell in delta.cells) == (metadata,)
    artifact = builder.seal(producer.finish())
    assert artifact.block.values[0, 0, 0, 0] == 5
    assert artifact.event_metadata == (metadata,)
    assert artifact.provenance.output_span.start_sequence == 0
    assert artifact.provenance.output_span.end_sequence == 1
    reservation.release()
