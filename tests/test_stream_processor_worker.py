"""Synthetic proof of one exact source -> processor -> final DatasetBuilder."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import threading
import time

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    DatasetSchema,
    PointLayout,
    REPEAT,
    SCAN_POINT,
    StreamGenerationId,
    VALID,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
)
from zlc_neutral_atom.catalog import DefinitionCatalog, DefinitionKey
from zlc_neutral_atom.processing.stream import (
    BoundStreamProcessor,
    ExactStreamProcessorWorker,
    StreamProcessorDefinition,
    StreamProcessorError,
)
from zlc_neutral_atom.runtime.cancellation import CancellationRequested
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetMode,
    ValueDatasetEventAdapter,
    dataset_cell_key_fingerprint,
    dataset_cell_permutation_digest,
    dataset_consumer_contract_digest,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ProducerFlowControl,
    ReservationState,
    RetentionOverrun,
    SourceFailed,
    SchemaChanged,
    StreamGap,
    StreamId,
    TraceBinding,
    TraceContext,
)


@dataclass(frozen=True)
class ScaleConfig:
    factor: int


@dataclass(frozen=True)
class ConvertConfig:
    output_schema: ValueSchema


def scale_value(payload: object, config: object) -> object:
    assert isinstance(payload, Value)
    assert isinstance(config, ScaleConfig)
    return Value(payload.values * config.factor, payload.validity, payload.schema)


def fail_on_two(payload: object, config: object) -> object:
    assert isinstance(payload, Value)
    if int(payload.values.item()) == 2:
        raise ArithmeticError("synthetic operator failure")
    return scale_value(payload, config)


def convert_to_probability(payload: object, config: object) -> object:
    assert isinstance(payload, Value)
    assert isinstance(config, ConvertConfig)
    return Value(
        np.array(float(payload.values.item()) / 10.0, dtype=config.output_schema.dtype),
        payload.validity,
        config.output_schema,
    )


def slow_value(payload: object, config: object) -> object:
    time.sleep(0.03)
    return scale_value(payload, config)


def operator_with_extra_default(payload: object, config: object, extra: int = 1) -> object:
    return scale_value(payload, config)


def variadic_operator(payload: object, config: object, *extra: object) -> object:
    return scale_value(payload, config)


async def async_operator(payload: object, config: object) -> object:
    return scale_value(payload, config)


def generator_operator(payload: object, config: object):
    yield scale_value(payload, config)


OPERATOR_ENTERED = threading.Event()
OPERATOR_RELEASE = threading.Event()


def cancellable_blocking_operator(payload: object, config: object) -> object:
    OPERATOR_ENTERED.set()
    if not OPERATOR_RELEASE.wait(1.0):
        raise TimeoutError("test did not release operator")
    return scale_value(payload, config)


def axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def schema(points: int) -> DatasetSchema:
    return DatasetSchema(
        axis("repeat", REPEAT, 1),
        (axis("point", SCAN_POINT, points),),
        PointLayout.rect_c((points,)),
        ValueSchema((), ValidityContract.value(), np.dtype("<i8"), value_unit="count"),
    )


def cells(schema: DatasetSchema) -> tuple[DatasetCellAddress, ...]:
    return tuple(
        DatasetCellAddress(0, point)
        for point in range(schema.point_layout.storage_size)
    )


@dataclass
class Chain:
    schema: DatasetSchema
    output_schema: DatasetSchema
    schedule: tuple[DatasetCellAddress, ...]
    source: AcquisitionStream
    producer: object
    reservation: object
    worker: ExactStreamProcessorWorker
    output: AcquisitionStream
    monitor: object
    builder: DatasetBuilder
    output_cursor: object


def chain(
    points: int = 3,
    *,
    operator=scale_value,
    config: object | None = None,
    output_cell_schema: ValueSchema | None = None,
    flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
    input_retention: int | None = None,
    terminal_wait_seconds: float = 1.0,
    operator_deadline_seconds: float = 1.0,
    share_join_owner: bool = True,
    expected_keys_override: tuple[DatasetCellAddress, ...] | None = None,
    source_schedule_digest_override: str | None = None,
    builder_schedule: tuple[DatasetCellAddress, ...] | None = None,
    tamper_output_cursor_owner: bool = False,
    output_trace_run_id: str = "synthetic-run",
    output_trace_source_id: str = "synthetic-processor",
    absolute_deadline_seconds: float = 2.0,
) -> Chain:
    data_schema = schema(points)
    result_schema = (
        data_schema
        if output_cell_schema is None
        else DatasetSchema(
            data_schema.repeat_axis,
            data_schema.point_axes,
            data_schema.point_layout,
            output_cell_schema,
        )
    )
    schedule = cells(data_schema)
    payload = ValuePayloadContract(data_schema.cell_schema)
    output_payload = ValuePayloadContract(result_schema.cell_schema)
    key_contract = DatasetCellKeyContract(data_schema)
    retention = points if input_retention is None else input_retention
    source, producer = AcquisitionStream.create(
        StreamId("synthetic.raw"),
        payload,
        flow_control=flow_control,
        retention_events=retention,
        retention_bytes=retention * payload.max_retained_nbytes,
        join_key_contract=key_contract,
    )
    reservation = source.reserve(
        total_events=points,
        max_inflight_events=retention,
        max_inflight_bytes=retention * payload.max_retained_nbytes,
        trace_binding=TraceBinding("synthetic-run", "synthetic-source"),
    )
    cursor = reservation.activate()
    output, output_producer = AcquisitionStream.create(
        StreamId("synthetic.scaled"),
        output_payload,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=points,
        retention_bytes=points * output_payload.max_retained_nbytes,
        join_key_contract=(
            key_contract if share_join_owner else DatasetCellKeyContract(result_schema)
        ),
    )
    output_reservation = output.reserve(
        total_events=points,
        max_inflight_events=points,
        max_inflight_bytes=points * output_payload.max_retained_nbytes,
        trace_binding=TraceBinding(output_trace_run_id, output_trace_source_id),
    )
    output_cursor = output_reservation.activate()
    builder = DatasetBuilder(
        BlockId("synthetic-output"),
        output_reservation,
        result_schema,
        DatasetMode.FINITE_EXACT,
        event_adapter=ValueDatasetEventAdapter(output_payload),
        expected_cells=schedule if builder_schedule is None else builder_schedule,
    )
    definition = StreamProcessorDefinition(
        DefinitionKey("test", "scale", 1),
        "Scale",
        "test.scale-config.v1",
        payload.fingerprint,
        output_payload.fingerprint,
        dataset_cell_key_fingerprint(data_schema),
        operator_deadline_seconds=operator_deadline_seconds,
        terminal_wait_seconds=terminal_wait_seconds,
    )
    assert DefinitionCatalog((definition,)).resolve(definition.key) is definition
    bound = BoundStreamProcessor(
        definition,
        ScaleConfig(10) if config is None else config,
        payload,
        output_payload,
        key_contract,
        output.stream_id,
        "synthetic-processor",
        operator,
    )
    source_contract = dataset_consumer_contract_digest(
        data_schema,
        schedule,
        ValueDatasetEventAdapter(payload).metadata_contract.fingerprint,
    )
    source_schedule = dataset_cell_permutation_digest(data_schema, schedule)
    if tamper_output_cursor_owner:
        output_reservation._cursor = object()
    worker = ExactStreamProcessorWorker(
        bound,
        reservation,
        cursor,
        source_schema=data_schema,
        source_contract_digest=source_contract,
        source_schedule_digest=(
            source_schedule
            if source_schedule_digest_override is None
            else source_schedule_digest_override
        ),
        expected_keys=(schedule if expected_keys_override is None else expected_keys_override),
        output_producer=output_producer,
        output_cursor=output_cursor,
        output_builder=builder,
        deadline_monotonic=time.monotonic() + absolute_deadline_seconds,
    )
    monitor = output.monitor(
        max_events=points,
        max_bytes=points * output_payload.max_retained_nbytes,
    )
    return Chain(
        data_schema,
        result_schema,
        schedule,
        source,
        producer,
        reservation,
        worker,
        output,
        monitor,
        builder,
        output_cursor,
    )


def emit(item: Chain, ordinal: int, *, key: DatasetCellAddress | None = None) -> object:
    payload = Value(np.array(ordinal + 1, dtype=np.int64), VALID, item.schema.cell_schema)
    return item.producer.emit(
        payload,
        captured_at=float(ordinal),
        trace=TraceContext(
            "synthetic-run",
            "synthetic-source",
            "synthetic-correlation",
        ),
        join_key=item.schedule[ordinal] if key is None else key,
    )


def test_exact_chain_preserves_keys_provenance_and_all_cells_before_input_ack():
    item = chain()
    item.worker.start()
    readiness = item.worker.exact_readiness()
    source_contract = dataset_consumer_contract_digest(
        item.schema,
        item.schedule,
        ValueDatasetEventAdapter(item.source._payload_contract).metadata_contract.fingerprint,
    )
    readiness.validate_source(
        reservation=item.reservation,
        trace_binding=item.reservation.trace_binding,
        payload_contract_fingerprint=item.source.payload_contract_fingerprint,
        join_key_contract_fingerprint=dataset_cell_key_fingerprint(item.schema),
        source_contract_digest=source_contract,
        source_schedule_digest=dataset_cell_permutation_digest(
            item.schema,
            item.schedule,
        ),
        total_events=len(item.schedule),
    )
    inputs = [emit(item, ordinal) for ordinal in range(3)]
    artifact = item.worker.finish(item.producer.finish(), 2.0)
    assert tuple(int(value) for value in artifact.block.values[0, :, ...]) == (10, 20, 30)
    outputs = [item.monitor.next().envelope for _ in range(3)]
    assert [output.join_key for output in outputs] == list(item.schedule)
    assert all(output.event_id != source.event_id for output, source in zip(outputs, inputs))
    assert [output.trace.causation_refs[0] for output in outputs] == [
        source.ref for source in inputs
    ]
    assert item.reservation.acknowledged_sequence == 3
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.worker.is_alive


def test_processor_changes_value_schema_without_changing_cell_key_domain():
    output_cell = ValueSchema(
        (),
        ValidityContract.value(),
        np.dtype("<f8"),
        value_unit="probability",
    )
    item = chain(
        points=1,
        operator=convert_to_probability,
        config=ConvertConfig(output_cell),
        output_cell_schema=output_cell,
    )
    assert item.schema.fingerprint != item.output_schema.fingerprint
    assert dataset_cell_key_fingerprint(item.schema) == dataset_cell_key_fingerprint(
        item.output_schema
    )
    item.worker.start()
    emit(item, 0)
    artifact = item.worker.finish(item.producer.finish(), 2.0)
    assert artifact.block.schema is item.output_schema
    assert float(artifact.block.values[0, 0]) == pytest.approx(0.1)


def test_operator_failure_leaves_failing_input_unacknowledged_and_joins():
    item = chain(operator=fail_on_two)
    item.worker.start()
    emit(item, 0)
    emit(item, 1)
    item.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, ArithmeticError)
    assert item.reservation.acknowledged_sequence == 1
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.worker.is_alive


def test_cancellation_while_waiting_is_bounded_and_joins():
    item = chain(points=1)
    item.worker.start()
    item.worker.cancel("synthetic cancel")
    item.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, CancellationRequested)
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.worker.is_alive


def test_preflight_close_releases_complete_chain_without_starting_thread():
    item = chain(points=1)
    item.worker.close(2.0)
    assert item.reservation.state is ReservationState.RELEASED
    assert item.worker.error is not None
    assert not item.worker.is_alive
    item.worker.close(2.0)


def test_early_eos_fails_chain_without_sealing_partial_dataset():
    item = chain(points=2)
    item.worker.start()
    emit(item, 0)
    with pytest.raises(StreamProcessorError):
        item.worker.finish(item.producer.finish(), 2.0)
    assert item.reservation.acknowledged_sequence == 1
    assert item.reservation.state is ReservationState.RELEASED


def test_source_failure_and_retention_overrun_propagate_and_join():
    item = chain(points=1)
    item.worker.start()
    item.producer.fail(SourceFailed("synthetic source failure"))
    item.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, SourceFailed)

    overrun = chain(
        points=2,
        flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
        input_retention=1,
    )
    emit(overrun, 0)
    with pytest.raises(RetentionOverrun):
        emit(overrun, 1)
    overrun.worker.start()
    overrun.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        overrun.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, RetentionOverrun)


def test_gap_and_join_key_mismatch_are_fail_closed():
    gap = chain(points=1)
    emit(gap, 0)
    with gap.source._condition:
        gap.source._records.clear()
        gap.source._order.clear()
        gap.source._retained_bytes = 0
    gap.worker.start()
    gap.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        gap.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, StreamGap)

    mismatch = chain(points=2)
    mismatch.worker.start()
    emit(mismatch, 0, key=mismatch.schedule[1])
    mismatch.worker.wait(2.0)
    with pytest.raises(StreamProcessorError, match="failed") as caught:
        mismatch.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, StreamProcessorError)
    assert mismatch.reservation.acknowledged_sequence == 0


def test_terminal_failure_and_supersede_after_last_event_wake_worker():
    failed = chain(points=1)
    failed.worker.start()
    emit(failed, 0)
    failed.producer.fail(SourceFailed("failed after last event"))
    failed.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        failed.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, SourceFailed)
    assert not failed.worker.is_alive

    superseded = chain(points=1)
    superseded.worker.start()
    emit(superseded, 0)
    superseded.producer.supersede(StreamGenerationId("replacement"))
    superseded.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        superseded.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, SchemaChanged)
    assert not superseded.worker.is_alive


def test_missing_terminal_and_late_operator_fail_with_declared_deadlines():
    missing = chain(points=1, terminal_wait_seconds=0.05)
    missing.worker.start()
    emit(missing, 0)
    missing.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        missing.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert not missing.worker.is_alive

    late = chain(
        points=1,
        operator=slow_value,
        operator_deadline_seconds=0.005,
    )
    late.worker.start()
    emit(late, 0)
    late.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        late.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert late.reservation.acknowledged_sequence == 0


def test_pass_through_preflight_cross_binds_owners_schedules_and_cursor():
    with pytest.raises(ValueError, match="output join-key contract owner"):
        chain(points=1, share_join_owner=False)
    with pytest.raises(ValueError, match="source_schedule_digest"):
        chain(points=2, source_schedule_digest_override="f" * 64)
    with pytest.raises(ValueError, match="source_schedule_digest"):
        item_schema = schema(2)
        chain(points=2, expected_keys_override=tuple(reversed(cells(item_schema))))
    with pytest.raises(ValueError, match="output builder schedule"):
        item_schema = schema(2)
        chain(points=2, builder_schedule=tuple(reversed(cells(item_schema))))
    with pytest.raises(ValueError, match="output builder reservation cursor"):
        chain(points=1, tamper_output_cursor_owner=True)
    with pytest.raises(ValueError, match="TraceBinding"):
        chain(points=1, output_trace_source_id="wrong-processor")
    with pytest.raises(ValueError, match="TraceBinding"):
        chain(points=1, output_trace_run_id="wrong-run")


def test_downstream_seal_failure_never_marks_upstream_completed(monkeypatch):
    item = chain(points=1)
    upstream_completions: list[object] = []
    original_complete = item.source._complete_consumer

    def record_upstream_completion(*args, **kwargs):
        upstream_completions.append(args)
        return original_complete(*args, **kwargs)

    def fail_seal():
        raise RuntimeError("synthetic downstream seal failure")

    monkeypatch.setattr(item.source, "_complete_consumer", record_upstream_completion)
    monkeypatch.setattr(item.builder, "_seal_locked", fail_seal)
    item.worker.start()
    emit(item, 0)
    with pytest.raises(StreamProcessorError):
        item.worker.finish(item.producer.finish(), 2.0)
    assert upstream_completions == []
    assert item.reservation.state is ReservationState.RELEASED


def test_downstream_seal_crossing_absolute_deadline_cannot_complete_upstream(monkeypatch):
    item = chain(points=1, absolute_deadline_seconds=0.05)
    upstream_completions: list[object] = []
    original_complete = item.source._complete_consumer
    original_seal = item.builder._seal_locked

    def record_upstream_completion(*args, **kwargs):
        upstream_completions.append(args)
        return original_complete(*args, **kwargs)

    def slow_seal():
        time.sleep(0.08)
        original_seal()

    monkeypatch.setattr(item.source, "_complete_consumer", record_upstream_completion)
    monkeypatch.setattr(item.builder, "_seal_locked", slow_seal)
    item.worker.start()
    emit(item, 0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.finish(item.producer.finish(), 1.0)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert upstream_completions == []
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.worker.is_alive


@pytest.mark.parametrize("operator", [operator_with_extra_default, variadic_operator])
def test_operator_signature_is_exactly_two_positional_parameters(operator):
    with pytest.raises(TypeError, match="exactly payload and frozen config"):
        chain(points=1, operator=operator)


@pytest.mark.parametrize("operator", [async_operator, generator_operator])
def test_operator_must_be_synchronous(operator):
    with pytest.raises(TypeError, match="top-level Python function"):
        chain(points=1, operator=operator)


def test_config_mapping_keys_must_be_canonical_strings():
    with pytest.raises(TypeError, match="canonical string keys"):
        chain(points=1, config=MappingProxyType({1: "integer", "1": "string"}))
    with pytest.raises(ValueError, match="canonical non-empty text"):
        chain(points=1, config=MappingProxyType({" padded ": 1}))

    first = chain(points=1, config=MappingProxyType({"a": 1, "b": 2}))
    second = chain(points=1, config=MappingProxyType({"b": 2, "a": 1}))
    assert first.worker._bound.fingerprint == second.worker._bound.fingerprint
    first.worker.close(2.0)
    second.worker.close(2.0)

    cross_type_pairs = (
        (b"\x01", MappingProxyType({"bytes_hex": "01"})),
        (np.dtype("<i8"), "<i8"),
        (
            ScaleConfig(1),
            MappingProxyType(
                {
                    "type": f"{ScaleConfig.__module__}.{ScaleConfig.__qualname__}",
                    "fields": MappingProxyType({"factor": 1}),
                }
            ),
        ),
    )
    for left_config, right_config in cross_type_pairs:
        left = chain(points=1, config=left_config)
        right = chain(points=1, config=right_config)
        assert left.worker._bound.fingerprint != right.worker._bound.fingerprint
        left.worker.close(2.0)
        right.worker.close(2.0)


def test_readiness_is_stable_but_single_bind_and_stale_proofs_are_rejected():
    item = chain(points=1)
    with pytest.raises(StreamProcessorError, match="has not started"):
        item.worker.exact_readiness()
    item.worker.start()
    readiness = item.worker.exact_readiness()
    assert item.worker.exact_readiness() is readiness
    readiness._claim_binding(item)
    with pytest.raises(Exception, match="already bound"):
        readiness._claim_binding(object())
    item.worker.close(2.0)
    with pytest.raises(StreamProcessorError, match="not live"):
        item.worker.exact_readiness()

    stale = chain(points=1)
    stale.worker.start()
    stale_readiness = stale.worker.exact_readiness()
    stale.producer.fail(SourceFailed("source failed before readiness binding"))
    stale.worker.wait(2.0)
    with pytest.raises(StreamProcessorError, match="not live"):
        stale.worker.exact_readiness()
    with pytest.raises(Exception, match="not registered|no longer live"):
        stale_readiness._claim_binding(object())
    stale.worker.close(2.0)

    stale_terminal = chain(points=1)
    stale_terminal.worker.start()
    terminal_readiness = stale_terminal.worker.exact_readiness()
    stale_terminal.worker._output_producer.fail(
        SourceFailed("terminal stream failed before readiness binding")
    )
    with pytest.raises(Exception, match="terminal dataset stream is no longer live"):
        terminal_readiness._claim_binding(object())
    stale_terminal.worker.close(2.0)

    completed = chain(points=1)
    completed.worker.start()
    emit(completed, 0)
    completed.producer.finish()
    completed.worker.wait(2.0)
    with pytest.raises(StreamProcessorError, match="not live"):
        completed.worker.exact_readiness()


def test_readiness_validation_rejects_worker_while_closing():
    OPERATOR_ENTERED.clear()
    OPERATOR_RELEASE.clear()
    item = chain(points=1, operator=cancellable_blocking_operator)
    item.worker.start()
    readiness = item.worker.exact_readiness()
    emit(item, 0)
    assert OPERATOR_ENTERED.wait(1.0)
    close_error: list[BaseException] = []

    def close_worker():
        try:
            item.worker.close(2.0)
        except BaseException as error:
            close_error.append(error)

    closer = threading.Thread(target=close_worker)
    closer.start()
    deadline = time.monotonic() + 1.0
    while not item.worker._closing and time.monotonic() < deadline:
        time.sleep(0.001)
    assert item.worker._closing
    with pytest.raises(StreamProcessorError, match="not live"):
        readiness._validate_terminal_sink()
    OPERATOR_RELEASE.set()
    closer.join(2.0)
    assert not closer.is_alive()
    assert close_error == []


def test_cancel_arriving_inside_operator_prevents_output_emit_and_input_ack():
    OPERATOR_ENTERED.clear()
    OPERATOR_RELEASE.clear()
    item = chain(
        points=1,
        operator=cancellable_blocking_operator,
        operator_deadline_seconds=1.0,
    )
    item.worker.start()
    emit(item, 0)
    assert OPERATOR_ENTERED.wait(1.0)
    item.worker.cancel("cancel during operator")
    OPERATOR_RELEASE.set()
    item.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, CancellationRequested)
    assert item.output.next_sequence == 0
    assert item.reservation.acknowledged_sequence == 0
    assert not item.worker.is_alive


def test_absolute_deadline_bounds_input_wait_and_thread_teardown():
    item = chain(points=1, absolute_deadline_seconds=0.05)
    item.worker.start()
    item.worker.wait(1.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.worker.is_alive


def test_finish_rejects_foreign_eos_even_after_autonomous_success():
    item = chain(points=1)
    item.worker.start()
    emit(item, 0)
    owned_eos = item.producer.finish()
    item.worker.wait(2.0)

    foreign = chain(points=1)
    foreign_eos = foreign.producer.finish()
    with pytest.raises(PermissionError, match="another source authority"):
        item.worker.finish(foreign_eos, 2.0)
    artifact = item.worker.finish(owned_eos, 2.0)
    assert int(artifact.block.values[0, 0]) == 10
    foreign.worker.close(2.0)
