"""Source-neutral PulseScan execution and canonical artifact commit.

PulseScan owns exactly one physical resource: the sequencer.  Its ``y`` value
comes from a producer-owned association cursor on an already-running signal
producer.  The application never opens, stops, restarts, or identifies that
producer's device, and it never branches on Camera, Occupancy, selector, or
Fit types.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
from typing import Callable

from zlc_data import (
    REPEAT,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetSchema,
    OwnedSnapshot,
    Value,
    ValuePayloadContract,
)
from zlc_neutral_atom.devices.sequencer.port import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
    PulseSession,
    PulseTerminalAck,
    pulse_terminal_ack_to_tree,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetCellSchedule,
    DatasetSealProvenance,
    FrozenDatasetEdge,
)
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
    notify_preview_failure,
)
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunHandle,
    RunPlan,
)
from zlc_neutral_atom.runtime.signal_source import (
    AuthoritativeSignalEventSource,
    SignalAssociationEvidence,
    SignalAssociationRequest,
    SignalAssociationScheduleRequirement,
    SignalAssociationUnavailable,
    SignalEvent,
    SignalEventAssociationCursor,
    SignalEventAssociationSource,
    SignalEventSource,
    SignalProjectionAuthority,
    authoritative_signal_event_source,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    AcquisitionProducer,
    AcquisitionStream,
    ArtifactInputRef,
    EventRef,
    ProcessorStageProvenance,
    StreamId,
    TraceBinding,
    TraceContext,
    event_ref_to_tree,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    bind_pulse_document_target,
    compile_pulse_artifact,
    expand_autonomous_scan_repeats,
)
from zlc_storage import (
    RepositoryRootLeaseBorrow,
    canonical_digest,
    canonical_text,
    encode,
)
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseParameterScanProgram,
)

from .contracts import (
    ScanOutputContract,
    bind_scan_output_contract,
)
from .final_output import scan_final_outputs
from .lineage import (
    ApiSegmentEvidence,
    ApiSegmentedScanExecution,
    AutonomousScanExecution,
    PulseScanExecution,
    SignalEventSequence,
)
from .reference import ScanArtifactRef
from .repository import (
    ScanRepository,
    _PreparedScanDataset,
    _SCAN_APPLICATION_TOKEN,
    _StagedScanLineage,
    _scan_output_dataset_ref,
)
from .source_binding import PulseScanBoundRequest


_SCAN_REPEAT_AXIS_ID = AxisId("pulse-scan.repeat")
_COLLECTOR_SOURCE_ID = "pulse-scan.collector"
_SOURCE_WAIT_SLICE_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class PulseScanPlanDescriptor:
    """User-facing facts for one frozen source-neutral scan command."""

    name: str
    execution_form: PulseExecutionForm
    repeat_count: int
    point_count: int
    source_definition: str
    source_output: str
    source_schema: DatasetSchema
    compiled_pulse_digest: str
    resource_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        canonical_text(self.name, "scan descriptor name")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        for field in ("repeat_count", "point_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        canonical_text(self.source_definition, "source_definition")
        canonical_text(self.source_output, "source_output")
        if not isinstance(self.source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        canonical_text(self.compiled_pulse_digest, "compiled_pulse_digest")
        claims = tuple(self.resource_claims)
        if not claims or any(not str(value).strip() for value in claims):
            raise ValueError("resource_claims must contain sequencer identity")
        object.__setattr__(self, "resource_claims", claims)


@dataclass(frozen=True, slots=True)
class _NoSignalMetadata:
    """Dataset metadata contract when source lineage lives in Scan execution."""

    fingerprint: str = canonical_digest(
        {"contract": "zlc_neutral_atom.pulse-scan.no-cell-metadata"}
    )

    @staticmethod
    def snapshot(_payload: Value) -> None:
        return None

    @staticmethod
    def validate(value: object) -> None:
        if value is not None:
            raise TypeError("PulseScan collector cell metadata must be None")

    @staticmethod
    def digest(value: object) -> str:
        _NoSignalMetadata.validate(value)
        return canonical_digest(
            {"contract": "zlc_neutral_atom.pulse-scan.no-cell-metadata-value"}
        )


@dataclass(frozen=True, slots=True)
class _SignalValueAdapter:
    """Identity Value projection for the collector's exact DatasetBuilder."""

    payload_contract: ValuePayloadContract
    metadata_contract: _NoSignalMetadata = _NoSignalMetadata()
    operator_fingerprint: str = canonical_digest(
        {"operator": "zlc_neutral_atom.pulse-scan.collect-signal-value"}
    )

    @property
    def value_schema(self):
        return self.payload_contract.schema

    def value(self, payload: Value) -> Value:
        self.payload_contract.validate(payload)
        return payload


@dataclass(slots=True)
class _PreparedScan:
    collector_producer: AcquisitionProducer[Value]
    collector_cursor: AcquisitionCursor[Value]
    builder: DatasetBuilder[Value]
    repository_borrow: RepositoryRootLeaseBorrow
    staged_lineage: _StagedScanLineage
    source_cursor: object | None = None
    autonomous_session: PulseSession | None = None
    current_api_session: PulseSession | None = None


@dataclass(frozen=True, slots=True)
class _ExecutedScan:
    prepared: _PreparedScan
    source_snapshot: OwnedSnapshot
    provenance: DatasetSealProvenance
    execution: PulseScanExecution


class _SignalSequenceAccumulator:
    """Ordered, expandable causal lineage for selected signal events."""

    __slots__ = (
        "_binding",
        "_associated_event_count",
        "_associations",
        "_count",
        "_direct_input_event_refs",
        "_event_refs",
        "_first_sequence",
        "_generation",
        "_hasher",
        "_last_sequence",
        "_processor_stages",
        "_projection_authority",
        "_source_id",
        "_source_run_id",
        "_stream_id",
    )

    def __init__(
        self,
        binding,
        cursor: object,
        projection_authority: SignalProjectionAuthority,
    ) -> None:
        stream_id = getattr(cursor, "stream_id", None)
        generation = getattr(cursor, "stream_generation", None)
        if stream_id is None or generation is None:
            raise TypeError("signal cursor does not expose stream identity")
        if not isinstance(projection_authority, SignalProjectionAuthority):
            raise TypeError(
                "projection_authority must be SignalProjectionAuthority"
            )
        if (
            getattr(cursor, "value_schema", None)
            is not projection_authority.output_value_schema
        ):
            raise RuntimeError(
                "signal cursor schema differs from committed projection authority"
            )
        self._binding = binding
        self._associated_event_count = 0
        self._associations: list[SignalAssociationEvidence] = []
        self._stream_id = stream_id
        self._generation = generation
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None
        self._count = 0
        self._event_refs: list[EventRef] = []
        self._direct_input_event_refs: list[tuple[EventRef, ...]] = []
        self._processor_stages: tuple[ProcessorStageProvenance, ...] | None = None
        self._projection_authority = projection_authority
        self._source_run_id: str | None = None
        self._source_id: str | None = None
        self._hasher = hashlib.sha256()
        self._hasher.update(b"zlc_neutral_atom.PulseScanSignalEventRefs\x00")

    def accept(self, event: SignalEvent) -> None:
        if not isinstance(event, SignalEvent):
            raise TypeError("signal cursor returned another event type")
        if event.value.schema is not self._projection_authority.output_value_schema:
            raise RuntimeError(
                "signal event schema changed after projection authority commit"
            )
        reference = event.event_ref
        if (
            reference.stream_id != self._stream_id
            or reference.generation != self._generation
        ):
            raise RuntimeError("signal source generation changed during PulseScan")
        if self._last_sequence is not None and reference.sequence <= self._last_sequence:
            raise RuntimeError("PulseScan source events are not strictly ordered")
        trace = event.trace
        if trace.run_id is None:
            raise RuntimeError("PulseScan source event has no producer Run identity")
        if self._source_run_id is None:
            self._source_run_id = trace.run_id
            self._source_id = trace.source_id
        elif trace.run_id != self._source_run_id or trace.source_id != self._source_id:
            raise RuntimeError("PulseScan source identity changed during one scan")
        processor_stages = event.processor_stages
        if self._processor_stages is None:
            self._processor_stages = processor_stages
        elif processor_stages != self._processor_stages:
            raise RuntimeError("PulseScan processor lineage changed during one scan")
        direct_inputs = tuple(
            item for item in trace.causation_refs if isinstance(item, EventRef)
        )
        artifact_inputs = tuple(
            item
            for item in trace.causation_refs
            if isinstance(item, ArtifactInputRef)
        )
        if len(direct_inputs) + len(artifact_inputs) != len(trace.causation_refs):
            raise RuntimeError(
                "PulseScan source event has unsupported direct causation"
            )
        stage_artifacts: list[ArtifactInputRef] = []
        seen_artifacts: set[str] = set()
        for stage in processor_stages:
            for item in stage.direct_artifact_inputs:
                if item.fingerprint not in seen_artifacts:
                    seen_artifacts.add(item.fingerprint)
                    stage_artifacts.append(item)
        if artifact_inputs != tuple(stage_artifacts):
            raise RuntimeError(
                "PulseScan source trace differs from its processor artifact lineage"
            )
        encoded = encode(event_ref_to_tree(reference))
        self._hasher.update(len(encoded).to_bytes(8, "big"))
        self._hasher.update(encoded)
        if self._first_sequence is None:
            self._first_sequence = reference.sequence
        self._last_sequence = reference.sequence
        self._event_refs.append(reference)
        self._direct_input_event_refs.append(direct_inputs)
        self._count += 1

    def accept_association(self, evidence: SignalAssociationEvidence) -> None:
        if not isinstance(evidence, SignalAssociationEvidence):
            raise TypeError("signal source returned another association evidence type")
        request = evidence.request
        expected_end = (
            self._associated_event_count + request.expected_event_count
        )
        if self._count != expected_end:
            raise RuntimeError(
                "signal association evidence does not align to its event group"
            )
        if any(
            item.request.association_id == request.association_id
            for item in self._associations
        ):
            raise RuntimeError("signal association id was reused within one scan")
        self._associations.append(evidence)
        self._associated_event_count = expected_end

    def finish(self, expected_count: int) -> SignalEventSequence:
        if self._count != expected_count:
            raise RuntimeError(
                f"PulseScan expected {expected_count} source events, got {self._count}"
            )
        if (
            self._first_sequence is None
            or self._last_sequence is None
            or self._source_run_id is None
            or self._source_id is None
            or self._processor_stages is None
        ):
            raise RuntimeError("PulseScan collected no source events")
        if self._associated_event_count != self._count or not self._associations:
            raise RuntimeError(
                "PulseScan source events lack complete producer association evidence"
            )
        return SignalEventSequence(
            self._binding,
            self._projection_authority,
            self._stream_id,
            self._generation,
            self._first_sequence,
            self._last_sequence,
            self._count,
            self._hasher.hexdigest(),
            self._source_run_id,
            self._source_id,
            tuple(self._event_refs),
            tuple(self._direct_input_event_refs),
            self._processor_stages,
            tuple(self._associations),
        )


class PreparedExactScan:
    """One-shot source-neutral PulseScan command."""

    __slots__ = (
        "_descriptor",
        "_lock",
        "_output_contract",
        "_plan_factory",
        "_program",
        "_repository",
        "_source_schema",
        "_start_run",
        "_started",
    )

    def __init__(
        self,
        *,
        program: PulseParameterScanProgram,
        source_schema: DatasetSchema,
        output_contract: ScanOutputContract,
        descriptor: PulseScanPlanDescriptor,
        repository: ScanRepository,
        plan_factory: Callable[[ExactDatasetPreviewPort | None], RunPlan],
        start_run: Callable[[RunPlan], RunHandle],
    ) -> None:
        if not isinstance(program, (AutonomousScanSlotProgram, ApiSlotSegmentedProgram)):
            raise TypeError("program must be a current PulseScan program")
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if not isinstance(output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if not isinstance(descriptor, PulseScanPlanDescriptor):
            raise TypeError("descriptor must be PulseScanPlanDescriptor")
        if type(repository) is not ScanRepository:
            raise TypeError("repository must be ScanRepository")
        if not callable(plan_factory) or not callable(start_run):
            raise TypeError("prepared scan requires plan_factory and start_run")
        self._program = program
        self._source_schema = source_schema
        self._output_contract = output_contract
        self._descriptor = descriptor
        self._repository = repository
        self._plan_factory = plan_factory
        self._start_run = start_run
        self._lock = threading.Lock()
        self._started = False

    @property
    def source_schema(self) -> DatasetSchema:
        return self._source_schema

    @property
    def output_contract(self) -> ScanOutputContract:
        return self._output_contract

    @property
    def descriptor(self) -> PulseScanPlanDescriptor:
        return self._descriptor

    @property
    def preview_spec(self) -> ExactDatasetPreviewSpec:
        return ExactDatasetPreviewSpec(self._source_schema.fingerprint)

    def materialize_provisional_output(self, source) -> OwnedSnapshot:
        snapshot = getattr(source, "snapshot", None)
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("source must be a DatasetPreviewSnapshot")
        if snapshot.block.schema != self._source_schema:
            raise ValueError("provisional source differs from this PulseScan")
        return _identity_output_snapshot(
            self._program,
            snapshot,
            self._output_contract,
        )

    def start(self, preview: ExactDatasetPreviewPort | None = None) -> RunHandle:
        try:
            with self._lock:
                if self._started:
                    raise RuntimeError("PreparedExactScan is one-shot")
                self._started = True
            plan = self._plan_factory(preview)
            if not isinstance(plan, RunPlan):
                raise TypeError("PulseScan plan factory returned another type")
            return self._start_run(plan)
        except BaseException as error:
            notify_preview_failure(preview, error)
            raise

    def final_dataset_outputs(self, reference: ScanArtifactRef):
        if not isinstance(reference, ScanArtifactRef):
            raise TypeError("scan FINAL result must be ScanArtifactRef")
        materialized = self._repository.materialize(reference)
        if materialized.program_fingerprint != self._program.fingerprint:
            raise ValueError("scan FINAL result belongs to another pulse program")
        if materialized.output_contract != self._output_contract:
            raise ValueError("scan FINAL result belongs to another output contract")
        return scan_final_outputs(materialized)


def prepare_exact_scan(
    request: PulseScanBoundRequest,
    source: SignalEventAssociationSource,
    *,
    pulse_port: BoundPulsePort,
    repository: ScanRepository,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedExactScan:
    """Bind an external live signal and sequencer without taking source ownership."""

    if not isinstance(request, PulseScanBoundRequest):
        raise TypeError("request must be PulseScanBoundRequest")
    if not isinstance(source, SignalEventSource):
        raise TypeError("source must implement SignalEventSource")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if type(repository) is not ScanRepository:
        raise TypeError("repository must be ScanRepository")
    if not callable(start_run):
        raise TypeError("start_run must be callable")
    if not isinstance(source, SignalEventAssociationSource):
        raise SignalAssociationUnavailable(
            "PulseScan requires producer-owned pulse association; ordinary "
            "SignalEventSource/FollowTap cursors prove software order only"
        )

    program = _bind_program_target(request.program, pulse_port)
    output_name = request.signal.output.name
    projected_source = authoritative_signal_event_source(
        source,
        output_name,
        request.signal.transform,
    )
    if not isinstance(projected_source, SignalEventAssociationSource):
        raise RuntimeError(
            "authoritative signal projection dropped producer association capability"
        )
    if request.signal.transform is None:
        value_schema = projected_source.value_schema(output_name)
        projection_authority = SignalProjectionAuthority(
            value_schema,
            None,
            value_schema,
        )
    else:
        if not isinstance(projected_source, AuthoritativeSignalEventSource):
            raise RuntimeError(
                "authoritative signal transform was not schema-committed"
            )
        projection_authority = projected_source.projection_authority
        value_schema = projection_authority.output_value_schema
    point_table = program.point_table
    repeat_axis = AxisSpec(
        _SCAN_REPEAT_AXIS_ID,
        "repeat",
        REPEAT,
        program.repeat_count,
        tuple(range(program.repeat_count)),
    )
    source_schema = DatasetSchema(
        repeat_axis,
        point_table,
        None,
        value_schema,
    )
    output_contract = bind_scan_output_contract(source_schema, point_table, None)
    schedule_requirement = (
        projected_source.signal_association_schedule_requirement(output_name)
    )
    if not isinstance(
        schedule_requirement,
        SignalAssociationScheduleRequirement,
    ):
        raise TypeError(
            "associated signal source returned another schedule requirement type"
        )
    pulse_requests = _compile_pulse_requests(
        program,
        pulse_port,
        schedule_requirement,
    )
    compiled = tuple(item.artifact for item in pulse_requests)
    compiled_digest = canonical_digest(
        {
            "owner": "zlc_neutral_atom.pulse-scan.compiled-pulses",
            "artifacts": [item.fingerprint for item in compiled],
        }
    )
    def plan_factory(preview: ExactDatasetPreviewPort | None) -> RunPlan:
        return _compile_scan_plan(
            request,
            program=program,
            source=projected_source,
            output_name=output_name,
            source_schema=source_schema,
            projection_authority=projection_authority,
            output_contract=output_contract,
            pulse_port=pulse_port,
            pulse_requests=pulse_requests,
            repository=repository,
            preview=preview,
        )
    descriptor = PulseScanPlanDescriptor(
        f"Pulse scan {program.document.name}",
        (
            PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
            if isinstance(program, AutonomousScanSlotProgram)
            else PulseExecutionForm.STATIC_ONCE
        ),
        program.repeat_count,
        point_table.row_count,
        request.signal.producer_definition.stable_definition_id,
        output_name,
        source_schema,
        compiled_digest,
        (str(pulse_port.resource_claim.key),),
    )
    return PreparedExactScan(
        program=program,
        source_schema=source_schema,
        output_contract=output_contract,
        descriptor=descriptor,
        repository=repository,
        plan_factory=plan_factory,
        start_run=start_run,
    )


def _bind_program_target(
    program: PulseParameterScanProgram,
    pulse_port: BoundPulsePort,
) -> PulseParameterScanProgram:
    document = bind_pulse_document_target(
        program.document,
        pulse_port.capability.target,
    )
    if isinstance(program, AutonomousScanSlotProgram):
        return AutonomousScanSlotProgram(document, program.api_values)
    if isinstance(program, ApiSlotSegmentedProgram):
        return ApiSlotSegmentedProgram(
            document,
            program.table,
            program.segmentation_rationale,
        )
    raise TypeError("program must be a current PulseScan program")


def _compile_pulse_requests(
    program: PulseParameterScanProgram,
    pulse_port: BoundPulsePort,
    association_requirement: SignalAssociationScheduleRequirement,
) -> tuple[FinitePulseExecutionRequest, ...]:
    if not isinstance(
        association_requirement,
        SignalAssociationScheduleRequirement,
    ):
        raise TypeError(
            "association_requirement must be SignalAssociationScheduleRequirement"
        )
    trigger_channels = association_requirement.trigger_channels
    if isinstance(program, AutonomousScanSlotProgram):
        logical = program.execution_document
        document = expand_autonomous_scan_repeats(logical)
        artifact = compile_pulse_artifact(
            document,
            clock_hz=pulse_port.capability.clock_hz,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
            trigger_channels=trigger_channels,
            live_target=pulse_port.capability.target,
        )
        return (FinitePulseExecutionRequest(document, artifact),)
    if isinstance(program, ApiSlotSegmentedProgram):
        requests = tuple(
            FinitePulseExecutionRequest(
                document,
                compile_pulse_artifact(
                    document,
                    clock_hz=pulse_port.capability.clock_hz,
                    execution_form=PulseExecutionForm.STATIC_ONCE,
                    trigger_channels=trigger_channels,
                    live_target=pulse_port.capability.target,
                ),
            )
            for document in program.resolved_point_documents
        )
        if len(requests) != program.point_count:
            raise RuntimeError("API PulseScan compilation changed point cardinality")
        return requests
    raise TypeError("program must be a current PulseScan program")


def _cell_schedule(schema: DatasetSchema) -> DatasetCellSchedule:
    return DatasetCellSchedule.from_cells(
        schema,
        (
            DatasetCellAddress(repeat, point)
            for repeat in range(schema.repeat_axis.size)
            for point in range(schema.point_table.row_count)
        ),
    )


def _open_collector(
    context: RunContext,
    schema: DatasetSchema,
) -> tuple[
    AcquisitionProducer[Value],
    AcquisitionCursor[Value],
    DatasetBuilder[Value],
]:
    payload_contract = ValuePayloadContract(schema.cell_schema)
    stream, producer = AcquisitionStream.create(
        StreamId(f"pulse-scan:{context.run_id.value}"),
        payload_contract,
        join_key_contract=DatasetCellKeyContract.from_schema(schema),
    )
    total = schema.repeat_axis.size * schema.point_table.row_count
    reservation = stream.reserve(
        total_events=total,
        trace_binding=TraceBinding(context.run_id.value, _COLLECTOR_SOURCE_ID),
    )
    cursor = reservation.activate()
    edge = FrozenDatasetEdge(
        schema,
        _SignalValueAdapter(payload_contract),
        _cell_schedule(schema),
    )
    builder = DatasetBuilder(
        BlockId(f"pulse-scan-source:{context.run_id.value}"),
        reservation,
        edge,
    )
    builder.exact_readiness()
    return producer, cursor, builder


def _next_source_event(
    context: RunContext,
    cursor: SignalEventAssociationCursor,
    token: object,
) -> SignalEvent:
    while True:
        context.checkpoint()
        try:
            event = cursor.next_associated_signal(
                token,
                _SOURCE_WAIT_SLICE_SECONDS,
            )
        except TimeoutError:
            continue
        context.checkpoint()
        if not isinstance(event, SignalEvent):
            raise TypeError("signal cursor returned another event type")
        return event


def _open_source_cursor(
    prepared: _PreparedScan,
    source: SignalEventAssociationSource,
    output_name: str,
    source_schema: DatasetSchema,
) -> SignalEventAssociationCursor:
    """Open the producer-owned association boundary before the first FIRE."""

    if prepared.source_cursor is not None:
        raise RuntimeError("PulseScan source cursor is already open")
    cursor = source.open_associated_signal_cursor(output_name)
    try:
        if not isinstance(cursor, SignalEventAssociationCursor):
            raise SignalAssociationUnavailable(
                "producer returned an ordering-only signal cursor"
            )
        if getattr(cursor, "value_schema", None) is not source_schema.cell_schema:
            raise RuntimeError("signal ValueSchema changed before PulseScan FIRE")
    except BaseException:
        cursor.close()
        raise
    prepared.source_cursor = cursor
    return cursor


def _collect_event(
    context: RunContext,
    prepared: _PreparedScan,
    accumulator: _SignalSequenceAccumulator,
    address: DatasetCellAddress,
    association_token: object,
) -> None:
    source_cursor = prepared.source_cursor
    if source_cursor is None:
        raise RuntimeError("PulseScan source cursor was not opened before FIRE")
    if not isinstance(source_cursor, SignalEventAssociationCursor):
        raise SignalAssociationUnavailable(
            "PulseScan source cursor lost association capability"
        )
    event = _next_source_event(context, source_cursor, association_token)
    accumulator.accept(event)
    prepared.collector_producer.emit(
        event.value,
        captured_at=event.captured_at,
        trace=TraceContext(
            context.run_id.value,
            _COLLECTOR_SOURCE_ID,
            f"cell:{address.repeat_index}:{address.point_ordinal}",
            causation_refs=(event.event_ref, *event.trace.causation_refs),
        ),
        join_key=address,
    )
    delivery = prepared.collector_cursor.next()
    prepared.builder.consume(delivery)


def _association_request(
    context: RunContext,
    session: PulseSession,
    artifact: CompiledPulseArtifact,
    *,
    group: str,
    expected_event_count: int,
) -> SignalAssociationRequest:
    return SignalAssociationRequest(
        f"{context.run_id.value}:{group}",
        session.session_id,
        artifact.fingerprint,
        expected_event_count,
    )


def _finish_association(
    cursor: SignalEventAssociationCursor,
    token: object,
    request: SignalAssociationRequest,
    terminal: PulseTerminalAck,
    accumulator: _SignalSequenceAccumulator,
) -> None:
    evidence = cursor.finish_signal_association(token)
    if not isinstance(evidence, SignalAssociationEvidence):
        raise TypeError("producer returned another association evidence type")
    if evidence.request != request:
        raise RuntimeError("producer association evidence belongs to another request")
    terminal_digest = canonical_digest(pulse_terminal_ack_to_tree(terminal))
    if evidence.terminal_evidence_digest != terminal_digest:
        raise RuntimeError(
            "producer association evidence is not bound to the exact pulse terminal"
        )
    accumulator.accept_association(evidence)


def _execute_scan(
    context: RunContext,
    prepared: _PreparedScan,
    *,
    request: PulseScanBoundRequest,
    program: PulseParameterScanProgram,
    source: SignalEventAssociationSource,
    output_name: str,
    source_schema: DatasetSchema,
    projection_authority: SignalProjectionAuthority,
    pulse_port: BoundPulsePort,
    pulse_requests: tuple[FinitePulseExecutionRequest, ...],
) -> _ExecutedScan:
    schedule = prepared.builder.edge.cell_schedule
    if schedule is None:
        raise RuntimeError("PulseScan collector lost its exact cell schedule")
    segments: list[ApiSegmentEvidence] = []
    try:
        if isinstance(program, AutonomousScanSlotProgram):
            session = prepared.autonomous_session
            if session is None or len(pulse_requests) != 1:
                raise RuntimeError("autonomous PulseScan preflight is incomplete")
            context.checkpoint()
            session.prepare(context)
            context.checkpoint()
            source_cursor = _open_source_cursor(
                prepared,
                source,
                output_name,
                source_schema,
            )
            accumulator = _SignalSequenceAccumulator(
                request.signal,
                source_cursor,
                projection_authority,
            )
            association = _association_request(
                context,
                session,
                pulse_requests[0].artifact,
                group="autonomous",
                expected_event_count=len(schedule),
            )
            association_token = source_cursor.arm_signal_association(association)
            session.fire(context)
            terminal = session.complete(context)
            source_cursor.bind_signal_association(association_token, terminal)
            for address in schedule:
                _collect_event(
                    context,
                    prepared,
                    accumulator,
                    address,
                    association_token,
                )
            _finish_association(
                source_cursor,
                association_token,
                association,
                terminal,
                accumulator,
            )
            source_sequence = accumulator.finish(len(schedule))
            execution: PulseScanExecution = AutonomousScanExecution(
                program,
                pulse_requests[0].artifact,
                terminal,
                source_sequence,
            )
        elif isinstance(program, ApiSlotSegmentedProgram):
            point_count = program.point_count
            if len(pulse_requests) != point_count:
                raise RuntimeError("API PulseScan preflight is incomplete")
            accumulator = None
            for address in schedule:
                context.checkpoint()
                pulse_request = pulse_requests[address.point_ordinal]
                session = pulse_port.open_session(pulse_request)
                prepared.current_api_session = session
                session.prepare(context)
                if accumulator is None:
                    source_cursor = _open_source_cursor(
                        prepared,
                        source,
                        output_name,
                        source_schema,
                    )
                    accumulator = _SignalSequenceAccumulator(
                        request.signal,
                        source_cursor,
                        projection_authority,
                    )
                association = _association_request(
                    context,
                    session,
                    pulse_request.artifact,
                    group=(
                        f"cell:{address.repeat_index}:"
                        f"{address.point_ordinal}"
                    ),
                    expected_event_count=1,
                )
                association_token = source_cursor.arm_signal_association(
                    association
                )
                session.fire(context)
                terminal = session.complete(context)
                source_cursor.bind_signal_association(
                    association_token,
                    terminal,
                )
                _collect_event(
                    context,
                    prepared,
                    accumulator,
                    address,
                    association_token,
                )
                _finish_association(
                    source_cursor,
                    association_token,
                    association,
                    terminal,
                    accumulator,
                )
                segments.append(
                    ApiSegmentEvidence(
                        address.repeat_index,
                        address.point_ordinal,
                        pulse_request.artifact,
                        terminal,
                    )
                )
            if accumulator is None:
                raise RuntimeError("API PulseScan contains no scheduled cell")
            source_sequence = accumulator.finish(len(schedule))
            execution = ApiSegmentedScanExecution(
                program,
                tuple(segments),
                source_sequence,
            )
        else:
            raise TypeError("program must be a current PulseScan program")
        eos = prepared.collector_producer.finish()
        sealed = prepared.builder.seal(eos)
        source_cursor = prepared.source_cursor
        if source_cursor is None:
            raise RuntimeError("PulseScan completed without a source cursor")
        source_cursor.close()
        return _ExecutedScan(
            prepared,
            sealed.snapshot,
            sealed.provenance,
            execution,
        )
    except BaseException:
        session = prepared.current_api_session or prepared.autonomous_session
        if session is not None:
            session.fail()
        raise


def _identity_output_snapshot(
    program: PulseParameterScanProgram,
    source: OwnedSnapshot,
    output_contract: ScanOutputContract,
) -> OwnedSnapshot:
    if output_contract.committed_transform is not None:
        raise ValueError("PulseScan collector output must already be authoritative")
    if source.block.schema != output_contract.output_dataset_schema:
        raise ValueError("PulseScan source schema differs from its identity output")
    output_ref = _scan_output_dataset_ref(program, source.ref, output_contract)
    return OwnedSnapshot(
        output_ref,
        DataBlock(
            output_ref.block_id,
            output_ref.revision,
            source.block.values,
            source.block.validity,
            output_contract.output_dataset_schema,
        ),
    )


def _compile_scan_plan(
    request: PulseScanBoundRequest,
    *,
    program: PulseParameterScanProgram,
    source: SignalEventAssociationSource,
    output_name: str,
    source_schema: DatasetSchema,
    projection_authority: SignalProjectionAuthority,
    output_contract: ScanOutputContract,
    pulse_port: BoundPulsePort,
    pulse_requests: tuple[FinitePulseExecutionRequest, ...],
    repository: ScanRepository,
    preview: ExactDatasetPreviewPort | None,
) -> RunPlan:
    compiled = tuple(item.artifact for item in pulse_requests)
    repository._require_active()

    def preflight(context: RunContext) -> _PreparedScan:
        borrow = repository._root_lease.borrow()
        builder = None
        try:
            borrow.require_active()
            staged = repository._stage_static_lineage(program, compiled)
            producer, collector_cursor, builder = _open_collector(
                context,
                source_schema,
            )
            autonomous_session = (
                pulse_port.open_session(pulse_requests[0])
                if isinstance(program, AutonomousScanSlotProgram)
                else None
            )
            if preview is not None:
                preview.bind(
                    builder.open_preview_reader(),
                    run_id=context.run_id.value,
                    causation_domain_id=builder.generation.value,
                )
            return _PreparedScan(
                producer,
                collector_cursor,
                builder,
                borrow,
                staged,
                autonomous_session=autonomous_session,
            )
        except BaseException:
            if builder is not None:
                builder.close()
            borrow.close()
            raise

    def execute(context: RunContext, prepared: _PreparedScan) -> _ExecutedScan:
        return _execute_scan(
            context,
            prepared,
            request=request,
            program=program,
            source=source,
            output_name=output_name,
            source_schema=source_schema,
            projection_authority=projection_authority,
            pulse_port=pulse_port,
            pulse_requests=pulse_requests,
        )

    def cleanup(
        context: RunContext,
        prepared: _PreparedScan | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        errors: list[BaseException] = []
        if prepared is None:
            report = pulse_port.verify_idle(context)
            if primary is not None or report.errors:
                notify_preview_failure(preview, primary or report.errors[0])
            return report
        source_cursor = prepared.source_cursor
        if source_cursor is not None:
            try:
                source_cursor.close()
            except BaseException as error:
                errors.append(error)
        session = prepared.current_api_session or prepared.autonomous_session
        try:
            report = (
                pulse_port.verify_idle(context)
                if session is None
                else session.cleanup(context)
            )
        except BaseException as error:
            errors.append(error)
        else:
            errors.extend(report.errors)
        try:
            prepared.builder.close()
        except BaseException as error:
            errors.append(error)
        result = CleanupReport.complete(errors=tuple(errors))
        if primary is not None or result.errors:
            prepared.repository_borrow.close()
            notify_preview_failure(preview, primary or result.errors[0])
        elif preview is not None:
            preview.source_terminal()
        return result

    def finalize(context: PostSafetyContext, result: _ExecutedScan) -> ScanArtifactRef:
        prepared = result.prepared
        try:
            prepared.repository_borrow.require_active()
            context.checkpoint()
            output = _identity_output_snapshot(
                program,
                result.source_snapshot,
                output_contract,
            )
            commit_value = _PreparedScanDataset(
                _SCAN_APPLICATION_TOKEN,
                run_id=context.run_id.value,
                execution=result.execution,
                source_snapshot=result.source_snapshot,
                output_contract=output_contract,
                output_snapshot=output,
                provenance=result.provenance,
                staged_lineage=prepared.staged_lineage,
            )
            operation = repository.final_commit(context, commit_value)
            return context.commit_final(operation)
        finally:
            prepared.repository_borrow.close()

    def dispose_unfinalized(result: _ExecutedScan) -> None:
        result.prepared.repository_borrow.close()

    return RunPlan(
        name=f"Pulse scan {program.document.name}",
        resource_claims=(pulse_port.resource_claim,),
        bound_devices=(pulse_port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=pulse_port.interrupt_operations,
        timeout_seconds=None,
        requires_final_commit=True,
        dispose_unfinalized=dispose_unfinalized,
    )


__all__ = [
    "PreparedExactScan",
    "PulseScanPlanDescriptor",
    "prepare_exact_scan",
]
