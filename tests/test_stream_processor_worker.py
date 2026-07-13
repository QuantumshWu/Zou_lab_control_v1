"""Synthetic proof of one exact source -> processor -> final DatasetBuilder."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from types import MappingProxyType
import threading
import time
import weakref

import numpy as np
import pytest
import zlc_neutral_atom.processing.stream.contract as stream_contract
from zlc_storage import encode

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
    ProcessorExecutionGuard,
    StreamProcessorDefinition,
    StreamProcessorError,
)
from zlc_neutral_atom.runtime.cancellation import CancellationRequested
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetMode,
    FrozenDatasetEdge,
    ValueDatasetEventAdapter,
    dataset_cell_key_fingerprint,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ArtifactInputRef,
    ProducerFlowControl,
    ProcessorStageProvenance,
    ReservationState,
    RetentionOverrun,
    SourceFailed,
    SchemaChanged,
    StreamEndedEarly,
    StreamError,
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


class RecordingExecutionGuard(ProcessorExecutionGuard):
    def __init__(
        self,
        *,
        schema_id: str = "test.processor-execution-guard",
        binding_fingerprint: str = "e" * 64,
        reject: bool = False,
    ) -> None:
        self._schema_id = schema_id
        self._binding_fingerprint = binding_fingerprint
        self.reject = reject
        self.calls: list[dict[str, object]] = []

    @property
    def schema_id(self) -> str:
        return self._schema_id

    @property
    def binding_fingerprint(self) -> str:
        return self._binding_fingerprint

    def authorize_exact_worker(
        self,
        *,
        bound: BoundStreamProcessor,
        input_reservation: object,
        input_cursor: object,
        input_edge: object,
        output_producer: object,
        deadline_monotonic: float,
        output_cursor: object | None,
        output_builder: object | None,
        downstream_readiness: object | None,
        cancellation: object | None,
    ) -> None:
        self.calls.append(
            {
                "bound": bound,
                "input_reservation": input_reservation,
                "input_cursor": input_cursor,
                "input_edge": input_edge,
                "output_producer": output_producer,
                "deadline_monotonic": deadline_monotonic,
                "output_cursor": output_cursor,
                "output_builder": output_builder,
                "downstream_readiness": downstream_readiness,
                "cancellation": cancellation,
                "consumer_was_claimed": input_reservation._stream._formal_consumer_claimed,
            }
        )
        if self.reject:
            raise PermissionError("synthetic execution guard rejection")


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
CHAIN_OPERATOR_ENTERED = threading.Event()
CHAIN_OPERATOR_RELEASE = threading.Event()
CHAIN_CURSOR_ENTERED = threading.Event()
CHAIN_CURSOR_RELEASE = threading.Event()


def cancellable_blocking_operator(payload: object, config: object) -> object:
    OPERATOR_ENTERED.set()
    if not OPERATOR_RELEASE.wait(1.0):
        raise TimeoutError("test did not release operator")
    return scale_value(payload, config)


def chain_blocking_operator(payload: object, config: object) -> object:
    CHAIN_OPERATOR_ENTERED.set()
    if not CHAIN_OPERATOR_RELEASE.wait(1.0):
        raise TimeoutError("test did not release chained operator")
    return scale_value(payload, config)


def axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def schema(points: int, *, cell_schema: ValueSchema | None = None) -> DatasetSchema:
    return DatasetSchema(
        axis("repeat", REPEAT, 1),
        (axis("point", SCAN_POINT, points),),
        PointLayout.rect_c((points,)),
        cell_schema
        or ValueSchema(
            (),
            ValidityContract.value(),
            np.dtype("<i8"),
            value_unit="count",
        ),
    )


def cells(schema: DatasetSchema) -> tuple[DatasetCellAddress, ...]:
    return tuple(
        DatasetCellAddress(0, point)
        for point in range(schema.point_layout.storage_size)
    )


def artifact_ref(seed: str) -> ArtifactInputRef:
    schema_id = "tests.synthetic-artifact-ref"
    return ArtifactInputRef(
        schema_id,
        encode({"schema": schema_id, "id": seed}),
        seed * 64,
    )


def assert_failure_evidence(error: BaseException | None, expected_type: type) -> None:
    assert error is not None
    assert error.original_type == expected_type.__name__
    assert error.__traceback__ is None


def evidence_contains(error: BaseException | None, expected_type: type) -> bool:
    if error is None:
        return False
    prefix = expected_type.__name__ + ":"
    return any(
        type(summary) is str and summary.startswith(prefix)
        for summary in error.related_summaries
    )


def edge(
    data_schema: DatasetSchema,
    payload: ValuePayloadContract,
    schedule: tuple[DatasetCellAddress, ...],
) -> FrozenDatasetEdge:
    return FrozenDatasetEdge(
        data_schema,
        ValueDatasetEventAdapter(payload),
        schedule,
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


@dataclass
class TwoStageChain:
    schema: DatasetSchema
    schedule: tuple[DatasetCellAddress, ...]
    source: AcquisitionStream
    producer: object
    source_reservation: object
    first: ExactStreamProcessorWorker
    intermediate: AcquisitionStream
    intermediate_monitor: object
    intermediate_reservation: object
    second: ExactStreamProcessorWorker
    output: AcquisitionStream
    monitor: object
    builder: DatasetBuilder


def _processor_binding(
    *,
    name: str,
    factor: int,
    payload: ValuePayloadContract,
    key_contract: DatasetCellKeyContract,
    output: AcquisitionStream,
    output_source_id: str,
    operator=scale_value,
    operator_deadline_seconds: float = 1.0,
    artifact_inputs: tuple[ArtifactInputRef, ...] = (),
) -> BoundStreamProcessor:
    definition = StreamProcessorDefinition(
        DefinitionKey("test", name),
        name,
        f"test.{name}-config",
        payload.fingerprint,
        payload.fingerprint,
        key_contract.fingerprint,
        operator_deadline_seconds=operator_deadline_seconds,
        terminal_wait_seconds=1.0,
    )
    return BoundStreamProcessor(
        definition,
        ScaleConfig(factor),
        payload,
        payload,
        key_contract,
        output.stream_id,
        output_source_id,
        operator,
        artifact_inputs,
    )


def two_stage_chain(
    *,
    points: int = 3,
    second_operator=scale_value,
    intermediate_retention: int | None = None,
    cancellation=None,
    start_first: bool = True,
    gate_second_cursor: bool = False,
    second_operator_deadline_seconds: float = 1.0,
    first_artifact_inputs: tuple[ArtifactInputRef, ...] = (),
    second_artifact_inputs: tuple[ArtifactInputRef, ...] = (),
    source_pair: tuple[AcquisitionStream, object] | None = None,
) -> TwoStageChain:
    data_schema = schema(points)
    schedule = cells(data_schema)
    if source_pair is None:
        payload = ValuePayloadContract(data_schema.cell_schema)
        key_contract = DatasetCellKeyContract(data_schema)
    else:
        source, producer = source_pair
        payload = source._payload_contract
        if not isinstance(payload, ValuePayloadContract):
            raise TypeError("source_pair must use ValuePayloadContract")
        data_schema = schema(points, cell_schema=payload.schema)
        schedule = cells(data_schema)
        key_contract = source._join_key_contract
        if not isinstance(key_contract, DatasetCellKeyContract):
            raise TypeError("source_pair must use DatasetCellKeyContract")
    budget = points * payload.max_retained_nbytes
    deadline = time.monotonic() + 3.0

    if source_pair is None:
        source, producer = AcquisitionStream.create(
            StreamId("synthetic.chain.raw"),
            payload,
            flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
            retention_events=points,
            retention_bytes=budget,
            join_key_contract=key_contract,
        )
    source_reservation = source.reserve(
        total_events=points,
        max_inflight_events=points,
        max_inflight_bytes=budget,
        trace_binding=TraceBinding("synthetic-chain-run", "chain-source"),
    )
    source_cursor = source_reservation.activate()

    retained = points if intermediate_retention is None else intermediate_retention
    intermediate, intermediate_producer = AcquisitionStream.create(
        StreamId("synthetic.chain.first"),
        payload,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=retained,
        retention_bytes=retained * payload.max_retained_nbytes,
        join_key_contract=key_contract,
    )
    intermediate_reservation = intermediate.reserve(
        total_events=points,
        max_inflight_events=retained,
        max_inflight_bytes=retained * payload.max_retained_nbytes,
        trace_binding=TraceBinding("synthetic-chain-run", "chain-first"),
    )
    intermediate_cursor = intermediate_reservation.activate()
    if gate_second_cursor:
        original_next = intermediate_cursor.next

        def gated_next(timeout=None):
            CHAIN_CURSOR_ENTERED.set()
            if not CHAIN_CURSOR_RELEASE.wait(1.0):
                raise TimeoutError("test did not release chained cursor")
            return original_next(timeout)

        intermediate_cursor.next = gated_next

    output, output_producer = AcquisitionStream.create(
        StreamId("synthetic.chain.second"),
        payload,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=points,
        retention_bytes=budget,
        join_key_contract=key_contract,
    )
    output_reservation = output.reserve(
        total_events=points,
        max_inflight_events=points,
        max_inflight_bytes=budget,
        trace_binding=TraceBinding("synthetic-chain-run", "chain-second"),
    )
    output_cursor = output_reservation.activate()
    dataset_edge = edge(data_schema, payload, schedule)
    builder = DatasetBuilder(
        BlockId("synthetic-chain-output"),
        output_reservation,
        dataset_edge,
        DatasetMode.FINITE_EXACT,
    )

    second = ExactStreamProcessorWorker(
        _processor_binding(
            name="chain-second",
            factor=3,
            payload=payload,
            key_contract=key_contract,
            output=output,
            output_source_id="chain-second",
            operator=second_operator,
            operator_deadline_seconds=second_operator_deadline_seconds,
            artifact_inputs=second_artifact_inputs,
        ),
        intermediate_reservation,
        intermediate_cursor,
        input_edge=dataset_edge,
        output_producer=output_producer,
        output_cursor=output_cursor,
        output_builder=builder,
        deadline_monotonic=deadline,
        cancellation=cancellation,
    )
    second.start()
    downstream_readiness = second.exact_readiness()
    first = ExactStreamProcessorWorker(
        _processor_binding(
            name="chain-first",
            factor=2,
            payload=payload,
            key_contract=key_contract,
            output=intermediate,
            output_source_id="chain-first",
            artifact_inputs=first_artifact_inputs,
        ),
        source_reservation,
        source_cursor,
        input_edge=dataset_edge,
        output_producer=intermediate_producer,
        downstream_readiness=downstream_readiness,
        deadline_monotonic=deadline,
    )
    monitor = output.monitor(max_events=points, max_bytes=budget)
    intermediate_monitor = intermediate.monitor(
        max_events=points,
        max_bytes=budget,
    )
    if start_first:
        first.start()
    return TwoStageChain(
        data_schema,
        schedule,
        source,
        producer,
        source_reservation,
        first,
        intermediate,
        intermediate_monitor,
        intermediate_reservation,
        second,
        output,
        monitor,
        builder,
    )


def emit_two_stage(item: TwoStageChain, ordinal: int) -> object:
    return item.producer.emit(
        Value(
            np.array(ordinal + 1, dtype=np.int64),
            VALID,
            item.schema.cell_schema,
        ),
        captured_at=20.0 + ordinal,
        trace=TraceContext(
            "synthetic-chain-run",
            "chain-source",
            f"chain-shot-{ordinal}",
            config_revision=17,
            control_revision=23,
        ),
        join_key=item.schedule[ordinal],
    )


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
    input_schedule: tuple[DatasetCellAddress, ...] | None = None,
    builder_schedule: tuple[DatasetCellAddress, ...] | None = None,
    tamper_output_cursor_owner: bool = False,
    output_trace_run_id: str = "synthetic-run",
    output_trace_source_id: str = "synthetic-processor",
    absolute_deadline_seconds: float = 2.0,
    artifact_inputs: tuple[ArtifactInputRef, ...] = (),
    execution_guard_schema_id: str | None = None,
    execution_guard: ProcessorExecutionGuard | None = None,
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
    input_edge = edge(
        data_schema,
        payload,
        schedule if input_schedule is None else input_schedule,
    )
    output_edge = edge(
        result_schema,
        output_payload,
        schedule if builder_schedule is None else builder_schedule,
    )
    builder = DatasetBuilder(
        BlockId("synthetic-output"),
        output_reservation,
        output_edge,
        DatasetMode.FINITE_EXACT,
    )
    definition = StreamProcessorDefinition(
        DefinitionKey("test", "scale"),
        "Scale",
        "test.scale-config",
        payload.fingerprint,
        output_payload.fingerprint,
        dataset_cell_key_fingerprint(data_schema),
        operator_deadline_seconds=operator_deadline_seconds,
        terminal_wait_seconds=terminal_wait_seconds,
        execution_guard_schema_id=execution_guard_schema_id,
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
        artifact_inputs,
        execution_guard,
    )
    if tamper_output_cursor_owner:
        output_reservation._cursor = object()
    worker = ExactStreamProcessorWorker(
        bound,
        reservation,
        cursor,
        input_edge=input_edge,
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


def test_execution_guard_definition_and_binding_must_match_exactly():
    schema_id = "test.processor-execution-guard"
    with pytest.raises(ValueError, match="requires an execution_guard"):
        chain(execution_guard_schema_id=schema_id)

    extra = RecordingExecutionGuard()
    with pytest.raises(ValueError, match="requires execution_guard_schema_id"):
        chain(execution_guard=extra)

    wrong = RecordingExecutionGuard(schema_id="test.another-guard")
    with pytest.raises(ValueError, match="schema differs from definition"):
        chain(
            execution_guard_schema_id=schema_id,
            execution_guard=wrong,
        )

    invalid_digest = RecordingExecutionGuard(binding_fingerprint="not-a-digest")
    with pytest.raises(ValueError, match="lowercase SHA-256 digest"):
        chain(
            execution_guard_schema_id=schema_id,
            execution_guard=invalid_digest,
        )


def test_raw_worker_cannot_bypass_guard_and_guard_runs_before_consumer_claim():
    guard = RecordingExecutionGuard(reject=True)
    with pytest.raises(PermissionError, match="synthetic execution guard rejection"):
        chain(
            execution_guard_schema_id=guard.schema_id,
            execution_guard=guard,
        )

    assert len(guard.calls) == 1
    call = guard.calls[0]
    assert call["consumer_was_claimed"] is False
    reservation = call["input_reservation"]
    assert reservation._stream._formal_consumer_claimed is False
    assert call["input_cursor"] is reservation._cursor
    assert call["input_edge"].expected_cells is not None
    assert call["output_producer"] is not None
    assert call["output_cursor"] is not None
    assert call["output_builder"] is not None
    assert call["downstream_readiness"] is None
    assert call["cancellation"] is None
    assert call["bound"].execution_guard is guard


def test_bound_processor_rejects_execution_guard_identity_drift():
    guard = RecordingExecutionGuard()
    item = chain(
        execution_guard_schema_id=guard.schema_id,
        execution_guard=guard,
    )
    bound = item.worker._bound
    original_fingerprint = bound.fingerprint
    assert len(original_fingerprint) == 64

    guard._binding_fingerprint = "f" * 64
    with pytest.raises(ValueError, match="binding fingerprint changed"):
        _ = bound.fingerprint

    guard._binding_fingerprint = "e" * 64
    guard._schema_id = "test.mutated-guard"
    with pytest.raises(ValueError, match="schema changed"):
        bound._validated_execution_guard()
    guard._schema_id = "test.processor-execution-guard"

    replacement = RecordingExecutionGuard()
    object.__setattr__(bound, "execution_guard", replacement)
    with pytest.raises(ValueError, match="owner changed"):
        bound._validated_execution_guard()
    object.__setattr__(bound, "execution_guard", guard)
    item.worker.close(2.0)


def test_exact_chain_preserves_keys_provenance_and_all_cells_before_input_ack():
    item = chain()
    item.worker.start()
    readiness = item.worker.exact_readiness()
    source_edge = edge(item.schema, item.source._payload_contract, item.schedule)
    readiness.validate_source(
        reservation=item.reservation,
        trace_binding=item.reservation.trace_binding,
        payload_contract_fingerprint=item.source.payload_contract_fingerprint,
        join_key_contract_fingerprint=dataset_cell_key_fingerprint(item.schema),
        source_contract_digest=source_edge.consumer_contract_digest,
        source_schedule_digest=source_edge.schedule_digest,
        source_key_sequence_digest=source_edge.key_sequence_digest,
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
    assert (
        item.worker._input_edge.exact_key_sequence_digest
        == item.builder.edge.exact_key_sequence_digest
    )
    assert (
        item.worker._input_edge.consumer_contract_digest
        != item.builder.edge.consumer_contract_digest
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
    assert_failure_evidence(caught.value.__cause__, ArithmeticError)
    assert item.reservation.acknowledged_sequence == 1
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.worker.is_alive


def test_failed_processor_graph_releases_without_cyclic_gc():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        item = chain(operator=fail_on_two)
        source_reference = weakref.ref(item.source)
        cursor_reference = weakref.ref(item.worker._input_cursor)
        worker_reference = weakref.ref(item.worker)
        item.worker.start()
        emit(item, 0)
        emit(item, 1)
        item.worker.wait(2.0)
        assert_failure_evidence(item.worker.error, ArithmeticError)
        assert item.reservation.state is ReservationState.RELEASED

        del item

        assert source_reference() is None
        assert cursor_reference() is None
        assert worker_reference() is None
    finally:
        if was_enabled:
            gc.enable()


def test_cancellation_while_waiting_is_bounded_and_joins():
    item = chain(points=1)
    item.worker.start()
    item.worker.cancel("synthetic cancel")
    item.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert_failure_evidence(caught.value.__cause__, CancellationRequested)
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
    deadline = time.monotonic() + 1.0
    while item.reservation.acknowledged_sequence < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert item.reservation.acknowledged_sequence == 1
    with pytest.raises(StreamEndedEarly, match="frozen interval"):
        item.producer.finish()
    item.producer.fail(SourceFailed("synthetic source stopped before formal end"))
    item.worker.wait(2.0)
    with pytest.raises(StreamProcessorError):
        item.worker.raise_if_failed()
    assert item.reservation.acknowledged_sequence == 1
    assert item.reservation.state is ReservationState.RELEASED


def test_source_failure_and_retention_overrun_propagate_and_join():
    item = chain(points=1)
    item.worker.start()
    item.producer.fail(SourceFailed("synthetic source failure"))
    item.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert_failure_evidence(caught.value.__cause__, SourceFailed)

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
    assert_failure_evidence(caught.value.__cause__, RetentionOverrun)


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
    assert_failure_evidence(caught.value.__cause__, StreamGap)

    mismatch = chain(points=2)
    mismatch.worker.start()
    emit(mismatch, 0, key=mismatch.schedule[1])
    mismatch.worker.wait(2.0)
    with pytest.raises(StreamProcessorError, match="failed") as caught:
        mismatch.worker.raise_if_failed()
    assert_failure_evidence(caught.value.__cause__, StreamProcessorError)
    assert mismatch.reservation.acknowledged_sequence == 0


def test_terminal_failure_and_supersede_after_last_event_wake_worker():
    failed = chain(points=1)
    failed.worker.start()
    emit(failed, 0)
    failed.producer.fail(SourceFailed("failed after last event"))
    failed.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        failed.worker.raise_if_failed()
    assert_failure_evidence(caught.value.__cause__, SourceFailed)
    assert not failed.worker.is_alive

    superseded = chain(points=1)
    superseded.worker.start()
    emit(superseded, 0)
    superseded.producer.supersede(StreamGenerationId("replacement"))
    superseded.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        superseded.worker.raise_if_failed()
    assert_failure_evidence(caught.value.__cause__, SchemaChanged)
    assert not superseded.worker.is_alive


def test_missing_terminal_and_late_operator_fail_with_declared_deadlines():
    missing = chain(points=1, terminal_wait_seconds=0.05)
    missing.worker.start()
    emit(missing, 0)
    missing.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        missing.worker.raise_if_failed()
    assert_failure_evidence(caught.value.__cause__, TimeoutError)
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
    assert_failure_evidence(caught.value.__cause__, TimeoutError)
    assert late.reservation.acknowledged_sequence == 0


def test_pass_through_preflight_cross_binds_owners_schedules_and_cursor():
    with pytest.raises(ValueError, match="output join-key contract owner"):
        chain(points=1, share_join_owner=False)
    with pytest.raises(ValueError, match="output builder schedule"):
        item_schema = schema(2)
        chain(points=2, input_schedule=tuple(reversed(cells(item_schema))))
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
    assert_failure_evidence(caught.value.__cause__, TimeoutError)
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

    caller_owned = {"a": 1}
    detached = chain(points=1, config=MappingProxyType(caller_owned))
    frozen_fingerprint = detached.worker._bound.fingerprint
    caller_owned["a"] = 999
    assert detached.worker._bound.config["a"] == 1
    assert detached.worker._bound.fingerprint == frozen_fingerprint
    detached.worker.close(2.0)

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


def test_bound_processor_owner_copies_declarative_inputs():
    data_schema = schema(1)
    payload = ValuePayloadContract(data_schema.cell_schema)
    key_contract = DatasetCellKeyContract(data_schema)
    definition_key = DefinitionKey("test", "owned-binding")
    definition = StreamProcessorDefinition(
        definition_key,
        "Owned binding",
        "test.owned-binding-config",
        payload.fingerprint,
        payload.fingerprint,
        key_contract.fingerprint,
        operator_deadline_seconds=1.0,
        terminal_wait_seconds=1.0,
    )
    caller_config = ScaleConfig(2)
    caller_reference = artifact_ref("a")
    caller_stream_id = StreamId("synthetic.owned-binding")
    bound = BoundStreamProcessor(
        definition,
        caller_config,
        payload,
        payload,
        key_contract,
        caller_stream_id,
        "synthetic-owned-binding",
        scale_value,
        (caller_reference,),
    )
    frozen_fingerprint = bound.fingerprint

    assert bound.definition is not definition
    assert bound.definition.key is not definition_key
    assert bound.config is not caller_config
    assert bound.artifact_inputs[0] is not caller_reference
    assert bound.output_stream_id is not caller_stream_id

    object.__setattr__(definition_key, "stable_definition_id", "rewritten")
    object.__setattr__(definition, "operator_deadline_seconds", 1e-12)
    object.__setattr__(caller_config, "factor", 99)
    object.__setattr__(caller_reference, "content_digest", "b" * 64)
    object.__setattr__(caller_stream_id, "value", "rewritten-output")

    assert bound.definition.key.stable_definition_id == "owned-binding"
    assert bound.definition.operator_deadline_seconds == 1.0
    assert bound.config == ScaleConfig(2)
    assert bound.artifact_inputs[0].content_digest == "a" * 64
    assert bound.output_stream_id.value == "synthetic.owned-binding"
    assert bound.fingerprint == frozen_fingerprint


def test_processor_stage_owner_copies_direct_artifact_references():
    caller_reference = artifact_ref("c")
    stage = ProcessorStageProvenance("d" * 64, (caller_reference,))
    assert stage.direct_artifact_inputs[0] is not caller_reference
    object.__setattr__(caller_reference, "content_digest", "e" * 64)
    assert stage.direct_artifact_inputs[0].content_digest == "c" * 64


def test_artifact_input_ref_snapshots_only_canonical_owner_bytes():
    schema_id = "tests.owner-ref"
    payload = encode({"schema": schema_id, "id": "calibration-1"})
    reference = ArtifactInputRef(schema_id, payload, "c" * 64)
    assert reference.canonical_reference is payload
    assert len(reference.reference_digest) == 64

    with pytest.raises(TypeError, match="immutable bytes"):
        ArtifactInputRef(schema_id, bytearray(payload), "c" * 64)
    with pytest.raises(ValueError, match="schema"):
        ArtifactInputRef("tests.other-ref", payload, "c" * 64)
    with pytest.raises(ValueError, match="canonical owner data"):
        ArtifactInputRef(schema_id, payload + b"\x00", "c" * 64)
    oversized = encode({"schema": schema_id, "blob": "x" * (64 * 1024)})
    with pytest.raises(ValueError, match="at most 64 KiB"):
        ArtifactInputRef(schema_id, oversized, "c" * 64)
    with pytest.raises(ValueError, match="must not repeat"):
        chain(points=1, artifact_inputs=(reference, reference))

    many = tuple(
        ArtifactInputRef(
            schema_id,
            encode({"schema": schema_id, "id": index}),
            f"{index:064x}",
        )
        for index in range(33)
    )
    with pytest.raises(ValueError, match="too many direct artifact inputs"):
        ProcessorStageProvenance("d" * 64, many)

    large = tuple(
        ArtifactInputRef(
            schema_id,
            encode(
                {
                    "schema": schema_id,
                    "id": index,
                    "blob": "x" * (60 * 1024),
                }
            ),
            f"{index + 100:064x}",
        )
        for index in range(18)
    )
    with pytest.raises(ValueError, match="byte budget"):
        ProcessorStageProvenance("d" * 64, large)


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype(object),
        np.dtype([("field", "<i4")]),
        np.dtype(("<i4", (2,))),
        np.dtype("V4"),
        np.dtype(">i4"),
    ],
)
def test_config_rejects_non_scalar_or_noncanonical_dtypes(dtype):
    with pytest.raises(TypeError, match="recursively frozen binding data"):
        chain(points=1, config=dtype)


def test_config_text_and_bytes_have_per_item_and_total_byte_budgets(monkeypatch):
    monkeypatch.setattr(stream_contract, "_PROCESSOR_BINDING_MAX_SINGLE_TEXT_BYTES", 8)
    monkeypatch.setattr(stream_contract, "_PROCESSOR_BINDING_MAX_TOTAL_TEXT_BYTES", 12)
    with pytest.raises(TypeError, match="recursively frozen binding data"):
        chain(points=1, config="123456789")
    with pytest.raises(TypeError, match="recursively frozen binding data"):
        chain(points=1, config=(b"12345678", b"abcde"))


def test_config_budget_memoizes_shared_frozen_objects_but_rejects_cycles(monkeypatch):
    monkeypatch.setattr(stream_contract, "_PROCESSOR_BINDING_MAX_NODES", 8)
    shared = (1, 2, 3, 4, 5)
    accepted = chain(points=1, config=(shared, shared, shared))
    accepted.worker.close(2.0)

    backing = {}
    recursive = MappingProxyType(backing)
    backing["self"] = recursive
    with pytest.raises(TypeError, match="recursively frozen binding data"):
        chain(points=1, config=recursive)


def test_config_alias_expansion_and_integer_encoding_fail_before_digest_dos():
    shared: object = (0,)
    for _ in range(20):
        shared = (shared, shared)
    started = time.perf_counter()
    with pytest.raises(TypeError, match="recursively frozen binding data"):
        chain(points=1, config=shared)
    assert time.perf_counter() - started < 1.0

    with pytest.raises(TypeError, match="recursively frozen binding data"):
        chain(points=1, config=1 << 5000)


def test_worker_revalidates_formal_trace_before_operator_or_ack():
    item = chain(points=1)
    published = emit(item, 0)
    object.__setattr__(
        published,
        "trace",
        TraceContext("synthetic-run", "wrong-source", "capture"),
    )
    item.worker.start()
    item.producer.finish()
    item.worker.wait(2.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert_failure_evidence(caught.value.__cause__, StreamError)
    assert item.output.next_sequence == 0
    assert item.reservation.acknowledged_sequence == 0
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.worker.is_alive


def test_thread_start_failure_synchronously_releases_the_exact_graph(monkeypatch):
    item = chain(points=1)
    failure = RuntimeError("synthetic thread start failure")

    def fail_start(_thread):
        raise failure

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(StreamProcessorError, match="before its worker thread started"):
        item.worker.start()
    assert_failure_evidence(item.worker.error, RuntimeError)
    assert failure.__traceback__ is None
    assert any(
        note.startswith("detached processor traceback: ")
        for note in getattr(item.worker.error, "__notes__", ())
    )
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.source._reservations
    assert not item.output._reservations
    assert item.source._formal_rebind_required
    assert not item.worker.is_alive
    item.worker.close(0.05)


def test_thread_start_failure_releases_worker_graph_without_cyclic_gc(monkeypatch):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        item = chain(points=1)
        source_reference = weakref.ref(item.source)
        worker_reference = weakref.ref(item.worker)
        failure = RuntimeError("failure embeds worker", item.worker)

        def fail_start(_thread):
            raise failure

        monkeypatch.setattr(threading.Thread, "start", fail_start)
        try:
            item.worker.start()
        except StreamProcessorError:
            pass
        else:
            raise AssertionError("worker start failure was not propagated")

        assert item.reservation.state is ReservationState.RELEASED
        assert_failure_evidence(item.worker.error, RuntimeError)
        monkeypatch.undo()
        del failure
        del fail_start
        del item

        assert source_reference() is None
        assert worker_reference() is None
    finally:
        if was_enabled:
            gc.enable()


def test_hostile_start_exception_cannot_interrupt_worker_rollback(monkeypatch):
    class HostileStartError(RuntimeError):
        def __str__(self):
            raise ValueError("hostile __str__")

        def __repr__(self):
            raise ValueError("hostile __repr__")

    item = chain(points=1)
    failure = HostileStartError()

    def fail_start(_thread):
        raise failure

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(StreamProcessorError, match="HostileStartError"):
        item.worker.start()

    assert_failure_evidence(item.worker.error, HostileStartError)
    assert failure.__traceback__ is None
    assert item.worker._done
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.source._reservations
    assert not item.output._reservations
    assert item.source._formal_rebind_required
    item.worker.close(0.05)


def test_blocking_exception_hooks_are_never_called_during_worker_rollback(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class BlockingStartError(RuntimeError):
        @property
        def args(self):
            entered.set()
            release.wait(1.0)
            return ("blocked args",)

        @property
        def __notes__(self):
            entered.set()
            release.wait(1.0)
            return []

        def __str__(self):
            entered.set()
            release.wait(1.0)
            return "blocked str"

    item = chain(points=1)
    failure = BlockingStartError("safe base args")
    original_start = threading.Thread.start

    def fail_processor_start(thread):
        if thread.name.startswith("stream-processor:"):
            raise failure
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_processor_start)
    outcome = []

    def invoke_start():
        try:
            item.worker.start()
        except BaseException as error:
            outcome.append(error)

    caller = threading.Thread(target=invoke_start, name="test-worker-start-caller")
    caller.start()
    caller.join(0.5)
    try:
        assert not caller.is_alive()
        assert not entered.is_set()
        assert len(outcome) == 1
        assert isinstance(outcome[0], StreamProcessorError)
        assert item.worker._done
        assert item.reservation.state is ReservationState.RELEASED
        assert not item.source._reservations
        assert not item.output._reservations
    finally:
        release.set()
        caller.join(1.0)


def test_pre_start_deadline_failure_synchronously_releases_the_exact_graph():
    item = chain(points=1)
    item.worker._deadline_monotonic = time.monotonic() - 1.0

    with pytest.raises(StreamProcessorError, match="before its worker thread started"):
        item.worker.start()
    assert_failure_evidence(item.worker.error, TimeoutError)
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.source._reservations
    assert not item.output._reservations
    assert item.source._formal_rebind_required
    assert not item.worker.is_alive
    item.worker.close(0.05)


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
    assert_failure_evidence(caught.value.__cause__, CancellationRequested)
    assert item.output.next_sequence == 0
    assert item.reservation.acknowledged_sequence == 0
    assert not item.worker.is_alive


def test_absolute_deadline_bounds_input_wait_and_thread_teardown():
    item = chain(points=1, absolute_deadline_seconds=0.05)
    item.worker.start()
    item.worker.wait(1.0)
    with pytest.raises(StreamProcessorError) as caught:
        item.worker.raise_if_failed()
    assert_failure_evidence(caught.value.__cause__, TimeoutError)
    assert item.reservation.state is ReservationState.RELEASED
    assert not item.worker.is_alive


def test_finish_rejects_foreign_eos_even_after_autonomous_success():
    item = chain(points=1)
    item.worker.start()
    emit(item, 0)
    owned_eos = item.producer.finish()
    item.worker.wait(2.0)

    foreign = chain(points=1)
    emit(foreign, 0)
    foreign_eos = foreign.producer.finish()
    with pytest.raises(PermissionError, match="another source authority"):
        item.worker.finish(foreign_eos, 2.0)
    artifact = item.worker.finish(owned_eos, 2.0)
    assert int(artifact.block.values[0, 0]) == 10
    foreign.worker.close(2.0)


def test_two_live_processor_stages_reach_one_terminal_dataset_with_full_lineage():
    item = two_stage_chain(points=3, intermediate_retention=1)
    inputs = tuple(emit_two_stage(item, ordinal) for ordinal in range(3))
    artifact = item.first.finish(item.producer.finish(), 2.0)

    assert tuple(int(value) for value in artifact.block.values[0, :]) == (6, 12, 18)
    outputs = tuple(item.monitor.next().envelope for _ in range(3))
    intermediate = tuple(
        item.intermediate_monitor.next().envelope for _ in range(3)
    )
    assert tuple(output.join_key for output in outputs) == item.schedule
    assert tuple(output.captured_at for output in outputs) == (20.0, 21.0, 22.0)
    assert all(output.trace.config_revision == 17 for output in outputs)
    assert all(output.trace.control_revision == 23 for output in outputs)
    assert [event.trace.causation_refs[0] for event in intermediate] == [
        event.ref for event in inputs
    ]
    assert [event.trace.causation_refs[0] for event in outputs] == [
        event.ref for event in intermediate
    ]
    assert all(len(event.trace.causation_refs) == 1 for event in outputs)
    derivation = artifact.provenance.derivation
    assert derivation is not None
    assert derivation.root_input_span.stream_id == item.source.stream_id
    assert derivation.root_input_span.generation == item.source.generation
    assert derivation.root_input_span.start_sequence == 0
    assert derivation.root_input_span.end_sequence == len(inputs)
    assert tuple(stage.processor_binding_digest for stage in derivation.stages) == (
        item.first._bound.fingerprint,
        item.second._bound.fingerprint,
    )
    assert item.source_reservation.state is ReservationState.RELEASED
    assert item.intermediate_reservation.state is ReservationState.RELEASED
    assert not item.first.is_alive
    assert not item.second.is_alive


def test_exact_reservation_owns_downstream_stage_only_until_release_without_gc():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        item = two_stage_chain(points=2)
        second_reference = weakref.ref(item.second)
        item.second = None

        assert second_reference() is not None
        emit_two_stage(item, 0)
        emit_two_stage(item, 1)
        artifact = item.first.finish(item.producer.finish(), 2.0)

        assert tuple(int(value) for value in artifact.block.values[0, :]) == (6, 12)
        assert item.intermediate_reservation.state is ReservationState.RELEASED
        assert second_reference() is None
    finally:
        if was_enabled:
            gc.enable()


def test_artifact_causation_is_direct_while_sealed_derivation_is_root_complete():
    shared = artifact_ref("a")
    tail_only = artifact_ref("b")
    item = two_stage_chain(
        points=2,
        first_artifact_inputs=(shared,),
        second_artifact_inputs=(shared, tail_only),
    )
    inputs = tuple(emit_two_stage(item, ordinal) for ordinal in range(2))
    artifact = item.first.finish(item.producer.finish(), 2.0)
    intermediate = tuple(
        item.intermediate_monitor.next().envelope for _ in range(2)
    )
    outputs = tuple(item.monitor.next().envelope for _ in range(2))

    assert [event.trace.causation_refs for event in intermediate] == [
        (source.ref, shared) for source in inputs
    ]
    assert [event.trace.causation_refs for event in outputs] == [
        (source.ref, shared, tail_only) for source in intermediate
    ]
    derivation = artifact.provenance.derivation
    assert derivation is not None
    assert derivation.stages[0].direct_artifact_inputs == (shared,)
    assert derivation.stages[1].direct_artifact_inputs == (shared, tail_only)
    assert derivation.artifact_inputs == (shared, tail_only)


def test_upstream_ack_waits_for_real_downstream_processing_and_ack():
    CHAIN_OPERATOR_ENTERED.clear()
    CHAIN_OPERATOR_RELEASE.clear()
    item = two_stage_chain(points=1, second_operator=chain_blocking_operator)
    emit_two_stage(item, 0)
    assert CHAIN_OPERATOR_ENTERED.wait(1.0)
    assert item.source_reservation.acknowledged_sequence == 0
    assert item.intermediate_reservation.acknowledged_sequence == 0

    CHAIN_OPERATOR_RELEASE.set()
    artifact = item.first.finish(item.producer.finish(), 2.0)
    assert int(artifact.block.values[0, 0]) == 6
    assert item.source_reservation.acknowledged_sequence == 1


def test_downstream_operator_failure_propagates_without_upstream_ack():
    item = two_stage_chain(points=2, second_operator=fail_on_two)
    emit_two_stage(item, 0)
    item.first.wait(2.0)

    with pytest.raises(StreamProcessorError) as caught:
        item.first.raise_if_failed()
    assert evidence_contains(caught.value.__cause__, ArithmeticError)
    assert item.source_reservation.acknowledged_sequence == 0
    assert item.intermediate_reservation.acknowledged_sequence == 0
    assert item.source_reservation.state is ReservationState.RELEASED
    assert item.intermediate_reservation.state is ReservationState.RELEASED
    assert not item.first.is_alive
    assert not item.second.is_alive


def test_downstream_cancel_propagates_and_does_not_ack_pending_input():
    CHAIN_OPERATOR_ENTERED.clear()
    CHAIN_OPERATOR_RELEASE.clear()
    item = two_stage_chain(points=1, second_operator=chain_blocking_operator)
    emit_two_stage(item, 0)
    assert CHAIN_OPERATOR_ENTERED.wait(1.0)
    item.second.cancel("cancel downstream stage")
    CHAIN_OPERATOR_RELEASE.set()
    item.first.wait(2.0)

    with pytest.raises(StreamProcessorError):
        item.first.raise_if_failed()
    assert item.source_reservation.acknowledged_sequence == 0
    assert item.intermediate_reservation.acknowledged_sequence == 0
    assert not item.first.is_alive
    assert not item.second.is_alive


def test_downstream_operator_deadline_propagates_without_upstream_ack():
    item = two_stage_chain(
        points=1,
        second_operator=slow_value,
        second_operator_deadline_seconds=0.005,
    )
    emit_two_stage(item, 0)
    item.first.wait(2.0)

    with pytest.raises(StreamProcessorError) as caught:
        item.first.raise_if_failed()
    assert evidence_contains(caught.value.__cause__, TimeoutError)
    assert item.source_reservation.acknowledged_sequence == 0
    assert item.intermediate_reservation.acknowledged_sequence == 0
    assert not item.first.is_alive
    assert not item.second.is_alive


def test_downstream_gap_propagates_without_acknowledging_upstream_input():
    CHAIN_CURSOR_ENTERED.clear()
    CHAIN_CURSOR_RELEASE.clear()
    item = two_stage_chain(points=1, gate_second_cursor=True)
    assert CHAIN_CURSOR_ENTERED.wait(1.0)
    emit_two_stage(item, 0)
    deadline = time.monotonic() + 1.0
    while item.intermediate.next_sequence == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert item.intermediate.next_sequence == 1
    with item.intermediate._condition:
        item.intermediate._records.clear()
        item.intermediate._order.clear()
        item.intermediate._retained_bytes = 0
    CHAIN_CURSOR_RELEASE.set()
    item.first.wait(2.0)

    with pytest.raises(StreamProcessorError) as caught:
        item.first.raise_if_failed()
    assert evidence_contains(caught.value.__cause__, StreamGap)
    assert item.source_reservation.acknowledged_sequence == 0
    assert item.intermediate_reservation.acknowledged_sequence == 0
    assert not item.first.is_alive
    assert not item.second.is_alive


def test_terminal_seal_failure_prevents_every_processor_completion(monkeypatch):
    item = two_stage_chain(points=1)
    root_completions: list[object] = []
    intermediate_completions: list[object] = []
    original_root_complete = item.source._complete_consumer
    original_intermediate_complete = item.intermediate._complete_consumer

    def record_root(*args, **kwargs):
        root_completions.append(args)
        return original_root_complete(*args, **kwargs)

    def record_intermediate(*args, **kwargs):
        intermediate_completions.append(args)
        return original_intermediate_complete(*args, **kwargs)

    def fail_seal():
        raise RuntimeError("synthetic terminal seal failure")

    monkeypatch.setattr(item.source, "_complete_consumer", record_root)
    monkeypatch.setattr(item.intermediate, "_complete_consumer", record_intermediate)
    monkeypatch.setattr(item.builder, "_seal_locked", fail_seal)
    emit_two_stage(item, 0)
    with pytest.raises(StreamProcessorError):
        item.first.finish(item.producer.finish(), 2.0)

    assert root_completions == []
    assert intermediate_completions == []
    assert item.source_reservation.state is ReservationState.RELEASED
    assert item.intermediate_reservation.state is ReservationState.RELEASED
    assert not item.first.is_alive
    assert not item.second.is_alive


def test_stale_downstream_readiness_cannot_bind_an_upstream_worker():
    data_schema = schema(1)
    schedule = cells(data_schema)
    payload = ValuePayloadContract(data_schema.cell_schema)
    key_contract = DatasetCellKeyContract(data_schema)
    budget = payload.max_retained_nbytes
    deadline = time.monotonic() + 2.0
    intermediate, intermediate_producer = AcquisitionStream.create(
        StreamId("synthetic.stale.intermediate"),
        payload,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=1,
        retention_bytes=budget,
        join_key_contract=key_contract,
    )
    intermediate_reservation = intermediate.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=budget,
        trace_binding=TraceBinding("synthetic-chain-run", "chain-first"),
    )
    intermediate_cursor = intermediate_reservation.activate()
    output, output_producer = AcquisitionStream.create(
        StreamId("synthetic.stale.output"),
        payload,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=1,
        retention_bytes=budget,
        join_key_contract=key_contract,
    )
    output_reservation = output.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=budget,
        trace_binding=TraceBinding("synthetic-chain-run", "chain-second"),
    )
    output_cursor = output_reservation.activate()
    dataset_edge = edge(data_schema, payload, schedule)
    builder = DatasetBuilder(
        BlockId("synthetic-stale-output"),
        output_reservation,
        dataset_edge,
        DatasetMode.FINITE_EXACT,
    )
    second = ExactStreamProcessorWorker(
        _processor_binding(
            name="stale-second",
            factor=3,
            payload=payload,
            key_contract=key_contract,
            output=output,
            output_source_id="chain-second",
        ),
        intermediate_reservation,
        intermediate_cursor,
        input_edge=dataset_edge,
        output_producer=output_producer,
        output_cursor=output_cursor,
        output_builder=builder,
        deadline_monotonic=deadline,
    )
    second.start()
    stale = second.exact_readiness()
    second.cancel("make readiness stale")
    second.wait(2.0)

    raw, _raw_producer = AcquisitionStream.create(
        StreamId("synthetic.stale.raw"),
        payload,
        flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
        retention_events=1,
        retention_bytes=budget,
        join_key_contract=key_contract,
    )
    raw_reservation = raw.reserve(
        total_events=1,
        max_inflight_events=1,
        max_inflight_bytes=budget,
        trace_binding=TraceBinding("synthetic-chain-run", "chain-source"),
    )
    raw_cursor = raw_reservation.activate()
    with pytest.raises(Exception, match="not registered|not ACTIVE|no longer live"):
        ExactStreamProcessorWorker(
            _processor_binding(
                name="stale-first",
                factor=2,
                payload=payload,
                key_contract=key_contract,
                output=intermediate,
                output_source_id="chain-first",
            ),
            raw_reservation,
            raw_cursor,
            input_edge=dataset_edge,
            output_producer=intermediate_producer,
            downstream_readiness=stale,
            deadline_monotonic=deadline,
        )
    raw_reservation.abort(cancelled=True)
    raw_reservation.release()


def test_prestart_close_of_root_tears_down_started_tail_without_leaks():
    item = two_stage_chain(points=1, start_first=False)
    item.first.close(2.0)

    assert item.source_reservation.state is ReservationState.RELEASED
    assert item.intermediate_reservation.state is ReservationState.RELEASED
    assert item.first.error is not None
    assert item.second.error is not None
    assert not item.first.is_alive
    assert not item.second.is_alive
    item.first.close(2.0)


def test_zero_event_multistage_preflight_can_rebuild_root_and_complete():
    abandoned = two_stage_chain(points=2, start_first=False)
    source_pair = (abandoned.source, abandoned.producer)
    abandoned.first.close(2.0)

    assert abandoned.source.next_sequence == 0
    assert not abandoned.source._reservations
    assert not abandoned.source._formal_consumer_claimed
    assert abandoned.intermediate._closed
    assert abandoned.output._closed
    assert not abandoned.intermediate._reservations
    assert not abandoned.output._reservations
    with pytest.raises(Exception, match="not live|not registered|no longer live"):
        abandoned.second.exact_readiness()
    abandoned.monitor.close()
    abandoned.intermediate_monitor.close()

    rebuilt = two_stage_chain(points=2, source_pair=source_pair)
    emit_two_stage(rebuilt, 0)
    emit_two_stage(rebuilt, 1)
    artifact = rebuilt.first.finish(rebuilt.producer.finish(), 2.0)
    assert tuple(int(value) for value in artifact.block.values[0, :]) == (6, 12)
    assert rebuilt.source_reservation.state is ReservationState.RELEASED
    assert rebuilt.intermediate_reservation.state is ReservationState.RELEASED
    assert not rebuilt.first.is_alive
    assert not rebuilt.second.is_alive
    rebuilt.monitor.close()
    rebuilt.intermediate_monitor.close()


def test_downstream_readiness_is_rechecked_at_upstream_start():
    item = two_stage_chain(points=1, start_first=False)
    item.second.cancel("tail failed between bind and upstream start")
    item.second.wait(2.0)

    with pytest.raises(Exception, match="not registered|not live"):
        item.first.start()
    item.first.close(2.0)
    assert item.source_reservation.state is ReservationState.RELEASED
    assert item.intermediate_reservation.state is ReservationState.RELEASED
    assert not item.first.is_alive
    assert not item.second.is_alive
