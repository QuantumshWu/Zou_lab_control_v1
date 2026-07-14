"""Minimal flat Measurement -> exact Dataset pipeline compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from zlc_data import BlockId, DatasetSchema, ValidityMode
from zlc_storage import (
    canonical_text as _canonical_text,
    positive_integer as _positive_int,
    sha256_text,
)

from zlc_neutral_atom.catalog import DefinitionCatalog, DefinitionKey
from zlc_neutral_atom.camera_operator import (
    CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT,
)

from ._failure import record_secondary_failure
from .capture import (
    BoundCapturePort,
    CameraCapabilityEvidence,
    CameraCaptureProvenance,
    CaptureCompletion,
    CaptureSession,
    FrozenCaptureSpec,
    CaptureStreamContract,
    CaptureTerminalAck,
)
from .dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    SealedDatasetArtifact,
)
from .run import CleanupReport, RunContext, RunMode, RunPlan
from .streams import (
    AcquisitionCursor,
    EventSpanRef,
    ExactReservation,
    ProcessorStageProvenance,
    ReservationState,
    TraceBinding,
)


@dataclass(frozen=True)
class MeasurementDefinition:
    """Pure catalog metadata; domain composition constructs BoundMeasurement."""

    key: DefinitionKey
    title: str
    request_schema_id: str
    binding_schema_id: str
    capture_spec_owner_fingerprint: str
    output_schema_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        for field in ("title", "request_schema_id", "binding_schema_id"):
            _canonical_text(getattr(self, field), field)
        for field in (
            "capture_spec_owner_fingerprint",
            "output_schema_fingerprint",
        ):
            sha256_text(getattr(self, field), field)


@dataclass(frozen=True)
class BoundMeasurement:
    definition: MeasurementDefinition
    capture_port: BoundCapturePort
    capture_contract: CaptureStreamContract
    capture_spec: FrozenCaptureSpec

    def __post_init__(self) -> None:
        if not isinstance(self.definition, MeasurementDefinition):
            raise TypeError("definition must be MeasurementDefinition")
        if not isinstance(self.capture_port, BoundCapturePort):
            raise TypeError("capture_port must be BoundCapturePort")
        if not isinstance(self.capture_contract, CaptureStreamContract):
            raise TypeError("capture_contract must be CaptureStreamContract")
        if self.capture_contract.capability is not self.capture_port.capability:
            raise ValueError("measurement contract and port must share capability owner")
        if not isinstance(self.capture_spec, FrozenCaptureSpec):
            raise TypeError("capture_spec must be FrozenCaptureSpec")
        if (
            self.definition.capture_spec_owner_fingerprint
            != self.capture_contract.capture_spec_owner_fingerprint
            or self.capture_spec.owner_fingerprint
            != self.capture_contract.capture_spec_owner_fingerprint
        ):
            raise ValueError("measurement definition/spec/contract owner differs")
        if (
            self.definition.output_schema_fingerprint
            != self.capture_contract.dataset_schema.fingerprint
        ):
            raise ValueError("measurement definition output schema differs")
        camera_provenance = self.capture_contract.camera_provenance
        if (
            camera_provenance is not None
            and camera_provenance.camera_arm_spec_fingerprint != self.capture_spec.digest
        ):
            raise ValueError(
                "camera provenance arm spec differs from FrozenCaptureSpec"
            )

    @property
    def definition_key(self) -> DefinitionKey:
        return self.definition.key


def resolve_measurement_definition(
    catalog: DefinitionCatalog,
    key: DefinitionKey,
) -> MeasurementDefinition:
    if not isinstance(catalog, DefinitionCatalog):
        raise TypeError("catalog must be DefinitionCatalog")
    definition = catalog.resolve(key, MeasurementDefinition)
    if not isinstance(definition, MeasurementDefinition):
        raise TypeError("catalog entry is not a MeasurementDefinition")
    return definition


@dataclass(frozen=True, slots=True)
class PipelineMemoryProfile:
    """User memory limit plus runtime-owned conservative Python overhead."""

    memory_limit_bytes: int
    fixed_runtime_overhead_bytes: int = field(init=False, default=1 << 20)
    per_event_reference_overhead_bytes: int = field(init=False, default=2048)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_limit_bytes",
            _positive_int(self.memory_limit_bytes, "memory_limit_bytes"),
        )


_MEMORY_ADMISSION_TOKEN = object()


class PipelineMemoryAdmission:
    """Process-local proof that one exact chain passed its memory policy."""

    __slots__ = (
        "_aggregate_peak_bytes",
        "_chain_contract_digest",
    )

    def __init__(
        self,
        authority: object,
        *,
        aggregate_peak_bytes: int,
        memory_profile: PipelineMemoryProfile,
        chain_contract_digest: str,
    ) -> None:
        if authority is not _MEMORY_ADMISSION_TOKEN:
            raise PermissionError(
                "PipelineMemoryAdmission must be minted by admit_pipeline_memory"
            )
        if not isinstance(memory_profile, PipelineMemoryProfile):
            raise TypeError("memory_profile must be PipelineMemoryProfile")
        peak = _positive_int(aggregate_peak_bytes, "aggregate_peak_bytes")
        if peak > memory_profile.memory_limit_bytes:
            raise MemoryError(
                f"pipeline peak budget {peak} exceeds "
                f"limit {memory_profile.memory_limit_bytes}"
            )
        sha256_text(chain_contract_digest, "chain_contract_digest")
        object.__setattr__(self, "_aggregate_peak_bytes", peak)
        object.__setattr__(self, "_chain_contract_digest", chain_contract_digest)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PipelineMemoryAdmission is immutable")

    def __reduce__(self):
        raise TypeError("PipelineMemoryAdmission is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("PipelineMemoryAdmission is process-local")

    @property
    def aggregate_peak_bytes(self) -> int:
        return self._aggregate_peak_bytes

    @property
    def chain_contract_digest(self) -> str:
        return self._chain_contract_digest


def admit_pipeline_memory(
    *,
    aggregate_peak_bytes: int,
    memory_profile: PipelineMemoryProfile,
    chain_contract_digest: str,
) -> PipelineMemoryAdmission:
    """Admit a compiler-derived peak before its exact chain touches hardware."""

    return PipelineMemoryAdmission(
        _MEMORY_ADMISSION_TOKEN,
        aggregate_peak_bytes=aggregate_peak_bytes,
        memory_profile=memory_profile,
        chain_contract_digest=chain_contract_digest,
    )


@dataclass(frozen=True)
class DatasetMaterializerSpec:
    block_id: BlockId
    memory: PipelineMemoryProfile

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.memory, PipelineMemoryProfile):
            raise TypeError("memory must be PipelineMemoryProfile")


@dataclass(frozen=True)
class MinimalPipelineSpec:
    name: str
    measurement: BoundMeasurement
    materializer: DatasetMaterializerSpec
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        _canonical_text(self.name, "name")
        if not isinstance(self.measurement, BoundMeasurement):
            raise TypeError("measurement must be BoundMeasurement")
        if not isinstance(self.materializer, DatasetMaterializerSpec):
            raise TypeError("materializer must be DatasetMaterializerSpec")


_PIPELINE_RESULT_TOKEN = object()


class PipelineResult:
    """Opaque compiler-minted proof that capture terminal and dataset agree."""

    __slots__ = (
        "_dataset",
        "_capture_terminal",
        "_memory_admission",
        "_aggregate_peak_bytes",
        "_run_id",
        "_source_dataset_schema",
        "_camera_provenance",
        "_camera_capability_evidence",
        "_camera_arm_spec",
        "_source_cell_schedule",
        "_source_event_span",
        "_processor_stages",
        "_chain_contract_digest",
        "_direct_raw_capture",
    )

    def __init__(
        self,
        authority: object,
        dataset: SealedDatasetArtifact,
        capture_completion: CaptureCompletion,
        memory_admission: PipelineMemoryAdmission,
    ) -> None:
        if authority is not _PIPELINE_RESULT_TOKEN:
            raise PermissionError(
                "PipelineResult can only be minted by finalize_pipeline_result"
            )
        if not isinstance(dataset, SealedDatasetArtifact):
            raise TypeError("dataset must be SealedDatasetArtifact")
        if not isinstance(capture_completion, CaptureCompletion):
            raise TypeError("capture_completion must be CaptureCompletion")
        capture_terminal = capture_completion.terminal
        _session, terminal_reservation = (
            capture_completion._validate_pipeline_authority()
        )
        if not dataset._belongs_to_terminal_reservation(terminal_reservation):
            raise RuntimeError(
                "sealed dataset belongs to another exact terminal consumer"
            )
        if type(memory_admission) is not PipelineMemoryAdmission:
            raise TypeError("memory_admission must be an exact PipelineMemoryAdmission")
        if memory_admission.chain_contract_digest != capture_completion.chain_contract_digest:
            raise RuntimeError("memory admission belongs to another exact chain")
        provenance = dataset.provenance
        if provenance.trace_binding.run_id != capture_completion.trace_binding.run_id:
            raise RuntimeError("pipeline dataset and capture completion run_id differ")
        derivation = provenance.derivation
        if capture_completion.direct_terminal_consumer:
            if derivation is not None or capture_completion.processor_stages:
                raise RuntimeError(
                    "direct capture cannot carry processor derivation provenance"
                )
        else:
            if derivation is None:
                raise RuntimeError(
                    "processed capture is missing root derivation provenance"
                )
            if (
                derivation.chain_contract_digest
                != capture_completion.chain_contract_digest
                or derivation.stages != capture_completion.processor_stages
                or derivation.root_input_span
                != capture_completion.source_event_span
            ):
                raise RuntimeError(
                    "processed dataset derivation differs from capture readiness chain"
                )
        count = provenance.end_sequence - provenance.start_sequence
        if not dataset.coverage.complete or dataset.coverage.total_cells != count:
            raise RuntimeError("pipeline dataset coverage differs from event interval")
        if (
            capture_terminal.produced_count != count
            or capture_terminal.drained_count != count
        ):
            raise RuntimeError("pipeline terminal and dataset provenance differ")
        # A direct materializer projects the source event, so its dataset
        # metadata is exactly the metadata acknowledged by the physical source.
        # A processor is allowed (and normally expected) to define a different
        # output-metadata contract.  Its source binding is instead proven by the
        # identity-bound terminal reservation plus the exact derivation/root
        # event span validated above.  Equating source and derived metadata
        # digests would make every non-identity processor impossible to compose.
        if (
            capture_completion.direct_terminal_consumer
            and capture_terminal.ordered_metadata_digest
            != provenance.ordered_metadata_digest
        ):
            raise RuntimeError("direct pipeline metadata differs from capture terminal")
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_capture_terminal", capture_terminal)
        object.__setattr__(self, "_memory_admission", memory_admission)
        object.__setattr__(
            self,
            "_aggregate_peak_bytes",
            memory_admission.aggregate_peak_bytes,
        )
        object.__setattr__(self, "_run_id", capture_completion.trace_binding.run_id)
        object.__setattr__(
            self,
            "_source_dataset_schema",
            capture_completion.source_dataset_schema,
        )
        object.__setattr__(
            self,
            "_camera_provenance",
            capture_completion.camera_provenance,
        )
        object.__setattr__(
            self,
            "_camera_capability_evidence",
            capture_completion.camera_capability_evidence,
        )
        object.__setattr__(
            self,
            "_camera_arm_spec",
            capture_completion.camera_arm_spec,
        )
        object.__setattr__(
            self,
            "_source_cell_schedule",
            capture_completion.source_cell_schedule,
        )
        object.__setattr__(
            self,
            "_source_event_span",
            capture_completion.source_event_span,
        )
        object.__setattr__(
            self,
            "_processor_stages",
            capture_completion.processor_stages,
        )
        object.__setattr__(
            self,
            "_chain_contract_digest",
            capture_completion.chain_contract_digest,
        )
        object.__setattr__(
            self,
            "_direct_raw_capture",
            capture_completion.direct_terminal_consumer
            and capture_completion.camera_provenance is not None
            and capture_completion.camera_capability_evidence is not None
            and dataset.block.schema == capture_completion.source_dataset_schema
            and capture_completion.source_event_adapter_operator_fingerprint
            == CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT,
        )
        # Every validation and value copy above must succeed before the one
        # CaptureSession-owned commit consumes dataset and completion authority
        # together.  Public results retain immutable evidence, never the live
        # CaptureSession/stream graph.
        capture_completion._commit_pipeline_result(dataset)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PipelineResult is immutable")

    @property
    def dataset(self) -> SealedDatasetArtifact:
        return self._dataset

    @property
    def capture_terminal(self) -> CaptureTerminalAck:
        return self._capture_terminal

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def source_dataset_schema(self) -> DatasetSchema:
        return self._source_dataset_schema

    @property
    def camera_provenance(self) -> CameraCaptureProvenance | None:
        return self._camera_provenance

    @property
    def camera_capability_evidence(self) -> CameraCapabilityEvidence | None:
        return self._camera_capability_evidence

    @property
    def camera_arm_spec(self) -> FrozenCaptureSpec:
        return self._camera_arm_spec

    @property
    def source_cell_schedule(self) -> tuple[DatasetCellAddress, ...]:
        return self._source_cell_schedule

    @property
    def source_event_span(self) -> EventSpanRef:
        return self._source_event_span

    @property
    def processor_stages(self) -> tuple[ProcessorStageProvenance, ...]:
        return self._processor_stages

    @property
    def chain_contract_digest(self) -> str:
        return self._chain_contract_digest

    @property
    def is_direct_raw_capture(self) -> bool:
        return self._direct_raw_capture

    @property
    def aggregate_peak_bytes(self) -> int:
        return self._aggregate_peak_bytes

def dataset_storage_nbytes(schema: DatasetSchema) -> int:
    """Return exact value-plus-validity bytes for one materialized DataBlock."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    value_bytes = math.prod(schema.physical_shape) * int(schema.cell_schema.dtype.itemsize)
    leading = (
        schema.repeat_axis.size,
        schema.point_layout.storage_size,
    )
    validity = schema.cell_schema.validity_contract
    if validity.mode is ValidityMode.VALUE:
        validity_bytes = math.prod(leading)
    else:
        component_shape = tuple(
            schema.cell_schema.axis(axis_id).size
            for axis_id in validity.component_axis_ids
        )
        validity_bytes = math.prod((*leading, *component_shape))
    return value_bytes + validity_bytes


def estimate_pipeline_peak_bytes(spec: MinimalPipelineSpec) -> int:
    if not isinstance(spec, MinimalPipelineSpec):
        raise TypeError("spec must be MinimalPipelineSpec")
    contract = spec.measurement.capture_contract
    events = contract.total_events
    dataset_bytes = dataset_storage_nbytes(contract.dataset_schema)
    metadata_bytes = (
        events * contract.dataset_edge.metadata_max_retained_nbytes
    )
    memory = spec.materializer.memory
    return (
        contract.estimated_transport_bytes
        + 2 * dataset_bytes
        + metadata_bytes
        + memory.fixed_runtime_overhead_bytes
        + events * memory.per_event_reference_overhead_bytes
    )


def _release_preflight_software(
    session: CaptureSession,
    reservation: ExactReservation | None,
    builder: DatasetBuilder | None,
    primary: BaseException,
) -> None:
    cleanup_errors: list[BaseException] = []
    try:
        session.fail(primary)
    except BaseException as error:
        cleanup_errors.append(error)
    if builder is not None:
        try:
            builder.close()
        except BaseException as error:
            cleanup_errors.append(error)
    elif reservation is not None:
        try:
            if reservation.state not in (
                ReservationState.COMPLETED,
                ReservationState.RELEASED,
            ):
                reservation.abort()
            if reservation.state is not ReservationState.RELEASED:
                reservation.release()
        except BaseException as error:
            cleanup_errors.append(error)
    for error in cleanup_errors:
        record_secondary_failure(
            primary,
            "preflight software teardown also failed",
            error,
        )


@dataclass
class ExactCaptureTransaction:
    """Concrete reusable owner of one finite exact capture/materialization path."""

    session: CaptureSession
    reservation: ExactReservation
    cursor: AcquisitionCursor
    builder: DatasetBuilder
    port: BoundCapturePort
    contract: CaptureStreamContract
    memory_admission: PipelineMemoryAdmission

    def start(self, context: RunContext) -> None:
        self.session.prepare(context)
        self.session.start(context)

    def capture_all(self, context: RunContext) -> None:
        for _cell in self.contract.expected_cells:
            context.checkpoint()
            self.session.capture_next(context)
            self.builder.consume(
                self.cursor.next(
                    timeout=self.port.capability.max_blocking_call_seconds
                )
            )

    def complete(self, context: RunContext) -> PipelineResult:
        completion: CaptureCompletion = self.session.complete(context)
        if not self.session.owns_completion(completion):
            raise RuntimeError("capture completion does not belong to this session")
        dataset = self.builder.seal(completion.eos)
        provenance = dataset.provenance
        if (
            provenance.metadata_contract_fingerprint
            != self.contract.dataset_edge.metadata_contract_fingerprint
        ):
            raise RuntimeError("sealed dataset metadata contract differs from capture")
        if (
            provenance.ordered_metadata_digest
            != completion.terminal.ordered_metadata_digest
        ):
            raise RuntimeError("sealed dataset metadata digest differs from capture")
        return finalize_pipeline_result(
            dataset=dataset,
            capture_completion=completion,
            memory_admission=self.memory_admission,
        )

    def fail(self, error: BaseException) -> None:
        try:
            self.session.fail(error)
        except BaseException as failure_error:
            record_secondary_failure(
                error,
                "capture poison also failed",
                failure_error,
            )

    def abort_preflight(self, error: BaseException) -> None:
        """Release software-only authority before any capture command was attempted."""

        _release_preflight_software(
            self.session,
            self.reservation,
            self.builder,
            error,
        )

    def cleanup(self, context: RunContext) -> CleanupReport:
        software_errors: list[BaseException] = []
        try:
            self.builder.close()
        except BaseException as error:
            software_errors.append(error)
        report = self.session.cleanup(context)
        if not software_errors:
            return report
        return CleanupReport(
            safety_proofs=report.safety_proofs,
            decisions=report.decisions,
            errors=(*report.errors, *software_errors),
        )


def open_exact_capture(
    spec: MinimalPipelineSpec,
    context: RunContext,
) -> ExactCaptureTransaction:
    """Allocate the single reservation/materializer transaction without touching hardware."""

    if not isinstance(spec, MinimalPipelineSpec):
        raise TypeError("spec must be MinimalPipelineSpec")
    aggregate_peak = estimate_pipeline_peak_bytes(spec)
    if aggregate_peak > spec.materializer.memory.memory_limit_bytes:
        raise MemoryError(
            f"pipeline peak budget {aggregate_peak} exceeds "
            f"limit {spec.materializer.memory.memory_limit_bytes}"
        )
    measurement = spec.measurement
    port = measurement.capture_port
    contract = measurement.capture_contract
    session = port.open_session(
        contract,
        TraceBinding(context.run_id.value, contract.source_id),
        measurement.capture_spec,
    )
    reservation = None
    builder = None
    try:
        reservation = session.reserve_exact()
        cursor = reservation.activate()
        builder = DatasetBuilder(
            spec.materializer.block_id,
            reservation,
            contract.dataset_edge,
        )
        readiness = builder.exact_readiness()
        memory_admission = admit_pipeline_memory(
            aggregate_peak_bytes=aggregate_peak,
            memory_profile=spec.materializer.memory,
            chain_contract_digest=readiness.chain_contract_digest,
        )
        session.bind_exact_consumer(readiness)
        return ExactCaptureTransaction(
            session,
            reservation,
            cursor,
            builder,
            port,
            contract,
            memory_admission,
        )
    except BaseException as error:
        _release_preflight_software(session, reservation, builder, error)
        raise


def compile_pipeline(spec: MinimalPipelineSpec) -> RunPlan:
    """Compile the one supported finite exact path into one flat RunPlan."""

    if not isinstance(spec, MinimalPipelineSpec):
        raise TypeError("spec must be MinimalPipelineSpec")
    aggregate_peak = estimate_pipeline_peak_bytes(spec)
    if aggregate_peak > spec.materializer.memory.memory_limit_bytes:
        raise MemoryError(
            f"pipeline peak budget {aggregate_peak} exceeds "
            f"limit {spec.materializer.memory.memory_limit_bytes}"
        )
    port = spec.measurement.capture_port

    def preflight(context: RunContext) -> ExactCaptureTransaction:
        return open_exact_capture(spec, context)

    def execute(context: RunContext, prepared: ExactCaptureTransaction) -> PipelineResult:
        try:
            prepared.start(context)
            prepared.capture_all(context)
            return prepared.complete(context)
        except BaseException as error:
            prepared.fail(error)
            raise

    def cleanup(
        context: RunContext,
        prepared: ExactCaptureTransaction | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return port.verify_idle(context)
        return prepared.cleanup(context)

    return RunPlan(
        name=spec.name,
        mode=RunMode.FINITE_EXACT,
        resource_claims=(port.resource_claim,),
        hazard_claims=(port.hazard_claim,),
        bound_devices=(port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=lambda _context, result: result,
        interrupt_operations=port.interrupt_operations,
        timeout_seconds=spec.timeout_seconds,
        requires_final_commit=False,
    )


def finalize_pipeline_result(
    *,
    dataset: SealedDatasetArtifact,
    capture_completion: CaptureCompletion,
    memory_admission: PipelineMemoryAdmission,
) -> PipelineResult:
    """Validate and mint the result of a direct or processed exact pipeline.

    Domain-specific compilers use this owner API after their live exact chain
    seals.  The public entry point does not weaken authority: ``PipelineResult``
    still proves that ``dataset`` belongs to the opaque terminal reservation in
    ``capture_completion`` and revalidates the complete derivation chain.
    """

    return PipelineResult(
        _PIPELINE_RESULT_TOKEN,
        dataset,
        capture_completion,
        memory_admission,
    )


__all__ = [
    "admit_pipeline_memory",
    "BoundMeasurement",
    "compile_pipeline",
    "dataset_storage_nbytes",
    "DatasetMaterializerSpec",
    "ExactCaptureTransaction",
    "estimate_pipeline_peak_bytes",
    "finalize_pipeline_result",
    "MeasurementDefinition",
    "MinimalPipelineSpec",
    "open_exact_capture",
    "PipelineMemoryProfile",
    "PipelineMemoryAdmission",
    "PipelineResult",
    "resolve_measurement_definition",
]
