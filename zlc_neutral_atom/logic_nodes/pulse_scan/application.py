"""Source-neutral PulseScan execution and canonical artifact commit.

PulseScan owns exactly one physical resource: the sequencer.  Its ``y`` value
comes from a producer-owned association cursor on an already-running signal
producer.  The application never opens, stops, restarts, or identifies that
producer's device, and it never branches on Camera, Occupancy, selector, or
Fit types.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zlc_data import (
    REPEAT,
    AxisId,
    AxisSpec,
    BlockId,
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
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunPlan,
)
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationRequest,
    SignalAssociationScheduleRequirement,
    SignalAssociationUnavailable,
    SignalEvent,
    SignalEventAssociationCursor,
    SignalEventAssociationSource,
    SignalEventSource,
    SignalProjectionAuthority,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    AcquisitionProducer,
    AcquisitionStream,
    EventRef,
    StreamId,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    bind_pulse_document_target,
    compile_pulse_artifact,
    materialize_scan_sweeps,
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
from .artifact import write_scan_artifact
from .lineage import (
    ApiSegmentEvidence,
    ApiSegmentedScanExecution,
    AutonomousScanExecution,
    PulseScanExecution,
    SignalEventSequence,
)
from .reference import ScanArtifactRef
from .source_binding import PulseScanRequest


_SCAN_REPEAT_AXIS_ID = AxisId("pulse-scan.repeat")
_SOURCE_WAIT_SLICE_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class _NoSignalMetadata:
    """Dataset metadata contract when source lineage lives in Scan execution."""

    @staticmethod
    def snapshot(_payload: Value) -> None:
        return None

    @staticmethod
    def validate(value: object) -> None:
        if value is not None:
            raise TypeError("PulseScan collector cell metadata must be None")

@dataclass(frozen=True, slots=True)
class _SignalValueAdapter:
    """Identity Value projection for the collector's exact DatasetBuilder."""

    payload_contract: ValuePayloadContract
    metadata_contract: _NoSignalMetadata = _NoSignalMetadata()

    @property
    def value_schema(self):
        return self.payload_contract.schema

    def value(self, payload: Value) -> Value:
        self.payload_contract.validate(payload)
        return payload


@dataclass(slots=True)
class _ScanRunState:
    collector_producer: AcquisitionProducer[Value]
    collector_cursor: AcquisitionCursor[Value]
    builder: DatasetBuilder[Value]
    source_cursor: object | None = None
    autonomous_session: PulseSession | None = None
    current_api_session: PulseSession | None = None


@dataclass(frozen=True, slots=True)
class _CollectedScan:
    prepared: _ScanRunState
    source_snapshot: OwnedSnapshot
    provenance: DatasetSealProvenance
    execution: PulseScanExecution


class _SignalSequenceAccumulator:
    """Ordered, expandable causal lineage for selected signal events."""

    __slots__ = (
        "_binding",
        "_direct_input_event_refs",
        "_event_refs",
        "_generation",
        "_projection_authority",
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
        self._stream_id = stream_id
        self._generation = generation
        self._event_refs: list[EventRef] = []
        self._direct_input_event_refs: list[tuple[EventRef, ...]] = []
        self._projection_authority = projection_authority

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
        if self._event_refs and reference.sequence <= self._event_refs[-1].sequence:
            raise RuntimeError("PulseScan source events are not strictly ordered")
        self._event_refs.append(reference)
        self._direct_input_event_refs.append(event.direct_parent_refs)

    def finish(self, expected_count: int) -> SignalEventSequence:
        if len(self._event_refs) != expected_count:
            raise RuntimeError(
                "PulseScan expected "
                f"{expected_count} source events, got {len(self._event_refs)}"
            )
        if not self._event_refs:
            raise RuntimeError("PulseScan collected no source events")
        return SignalEventSequence(
            self._binding,
            self._projection_authority,
            tuple(self._event_refs),
            tuple(self._direct_input_event_refs),
        )


def compile_pulse_scan(
    request: PulseScanRequest,
    source: SignalEventAssociationSource,
    *,
    pulse_port: BoundPulsePort,
    project_root: Path,
) -> RunPlan:
    """Bind an external live signal and sequencer without taking source ownership."""

    if not isinstance(request, PulseScanRequest):
        raise TypeError("request must be PulseScanRequest")
    if not isinstance(source, SignalEventSource):
        raise TypeError("source must implement SignalEventSource")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path")
    project_root = project_root.resolve()
    if not isinstance(source, SignalEventAssociationSource):
        raise SignalAssociationUnavailable(
            "PulseScan requires producer-owned pulse association; ordinary "
            "SignalEventSource/FollowTap cursors prove software order only"
        )

    program = _bind_program_target(request.program, pulse_port)
    output_name = request.signal.output_name
    value_schema = source.value_schema(output_name)
    projection_authority = SignalProjectionAuthority(
        value_schema,
        None,
        value_schema,
    )
    point_table = program.point_table
    repeat_axis = AxisSpec(
        _SCAN_REPEAT_AXIS_ID,
        "repeat",
        REPEAT,
        program.sweep_count,
        tuple(range(program.sweep_count)),
    )
    source_schema = DatasetSchema(
        repeat_axis,
        point_table,
        None,
        value_schema,
    )
    output_contract = bind_scan_output_contract(source_schema, point_table, None)
    schedule_requirement = (
        source.signal_association_schedule_requirement(output_name)
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

    return _compile_scan_plan(
        request,
        program=program,
        source=source,
        output_name=output_name,
        source_schema=source_schema,
        projection_authority=projection_authority,
        output_contract=output_contract,
        pulse_port=pulse_port,
        pulse_requests=pulse_requests,
        project_root=project_root,
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
    trigger_channels = (association_requirement.trigger_channel,)
    if isinstance(program, AutonomousScanSlotProgram):
        logical = program.execution_document
        document = materialize_scan_sweeps(logical, program.sweep_count)
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
    prepared: _ScanRunState,
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
    prepared: _ScanRunState,
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
        direct_parent_refs=(event.event_ref,),
        join_key=address,
    )
    delivery = prepared.collector_cursor.next()
    prepared.builder.consume(delivery)


def _association_request(
    session: PulseSession,
    artifact: CompiledPulseArtifact,
    *,
    expected_event_count: int,
) -> SignalAssociationRequest:
    schedules = artifact.trigger_schedules
    if len(schedules) != 1:
        raise RuntimeError(
            "the current formal signal association requires exactly one "
            "compiled physical trigger schedule"
        )
    return SignalAssociationRequest(
        session.session_id,
        artifact.fingerprint,
        expected_event_count,
        schedules[0].fingerprint,
        schedules[0].channel,
        schedules[0].total,
        schedules[0].minimum_interval_ticks,
        artifact.target_ir.clock_hz,
    )


def _execute_scan(
    context: RunContext,
    prepared: _ScanRunState,
    *,
    request: PulseScanRequest,
    program: PulseParameterScanProgram,
    source: SignalEventAssociationSource,
    output_name: str,
    source_schema: DatasetSchema,
    projection_authority: SignalProjectionAuthority,
    pulse_port: BoundPulsePort,
    pulse_requests: tuple[FinitePulseExecutionRequest, ...],
) -> _CollectedScan:
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
                session,
                pulse_requests[0].artifact,
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
            source_cursor.finish_signal_association(association_token)
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
                    session,
                    pulse_request.artifact,
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
                source_cursor.finish_signal_association(association_token)
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
        return _CollectedScan(
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


def _compile_scan_plan(
    request: PulseScanRequest,
    *,
    program: PulseParameterScanProgram,
    source: SignalEventAssociationSource,
    output_name: str,
    source_schema: DatasetSchema,
    projection_authority: SignalProjectionAuthority,
    output_contract: ScanOutputContract,
    pulse_port: BoundPulsePort,
    pulse_requests: tuple[FinitePulseExecutionRequest, ...],
    project_root: Path,
) -> RunPlan:
    def preflight(context: RunContext) -> _ScanRunState:
        builder = None
        try:
            producer, collector_cursor, builder = _open_collector(
                context,
                source_schema,
            )
            autonomous_session = (
                pulse_port.open_session(pulse_requests[0])
                if isinstance(program, AutonomousScanSlotProgram)
                else None
            )
            return _ScanRunState(
                producer,
                collector_cursor,
                builder,
                autonomous_session=autonomous_session,
            )
        except BaseException:
            if builder is not None:
                builder.close()
            raise

    def execute(context: RunContext, prepared: _ScanRunState) -> _CollectedScan:
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
        prepared: _ScanRunState | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        errors: list[BaseException] = []
        if prepared is None:
            return pulse_port.verify_idle(context)
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
        return CleanupReport.complete(errors=tuple(errors))

    def finalize(context: PostSafetyContext, result: _CollectedScan) -> ScanArtifactRef:
        return write_scan_artifact(
            project_root,
            run_id=context.run_id.value,
            execution=result.execution,
            snapshot=result.source_snapshot,
            output_contract=output_contract,
            provenance=result.provenance,
        )

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
    )


__all__ = [
    "compile_pulse_scan",
]
