"""Minimal flat Measurement -> exact Dataset pipeline compiler."""

from __future__ import annotations

import math
import hashlib
import struct
import sys
from dataclasses import dataclass
from numbers import Integral

from zlc_data import BlockId, DatasetSchema, ValidityMode

from zlc_neutral_atom.catalog import DefinitionCatalog, DefinitionKey

from .capture import (
    BoundCapturePort,
    CaptureCompletion,
    CaptureSession,
    FrozenCaptureSpec,
    CaptureStreamContract,
    CaptureTerminalAck,
)
from .dataset import DatasetBuilder, DatasetMode, SealedDatasetArtifact
from .run import CleanupReport, RunContext, RunMode, RunPlan
from .streams import AcquisitionCursor, ExactReservation, ReservationState, TraceBinding


def _canonical_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _positive_int(value: int, field: str) -> int:
    normalized = _nonnegative_int(value, field)
    if normalized == 0:
        raise ValueError(f"{field} must be positive")
    return normalized


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
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")


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


_MEMORY_PROFILE_TOKEN = object()


class PipelineMemoryProfile:
    """Runtime-owned conservative Python object/allocator overhead policy."""

    __slots__ = (
        "_memory_limit_bytes",
        "_overhead_profile_fingerprint",
        "_fixed_runtime_overhead_bytes",
        "_per_event_reference_overhead_bytes",
    )

    def __init__(self, authority: object, memory_limit_bytes: int) -> None:
        if authority is not _MEMORY_PROFILE_TOKEN:
            raise PermissionError(
                "PipelineMemoryProfile must be minted for the current runtime"
            )
        limit = _positive_int(memory_limit_bytes, "memory_limit_bytes")
        pointer_bytes = struct.calcsize("P")
        fixed = 1 << 20
        per_event = max(2048, 128 * pointer_bytes)
        identity = (
            f"zlc.pipeline-memory-overhead.v1|{sys.implementation.name}|"
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}|"
            f"ptr={pointer_bytes}|fixed={fixed}|per_event={per_event}"
        )
        object.__setattr__(self, "_memory_limit_bytes", limit)
        object.__setattr__(
            self,
            "_overhead_profile_fingerprint",
            hashlib.sha256(identity.encode("ascii")).hexdigest(),
        )
        object.__setattr__(self, "_fixed_runtime_overhead_bytes", fixed)
        object.__setattr__(self, "_per_event_reference_overhead_bytes", per_event)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PipelineMemoryProfile is immutable")

    @classmethod
    def for_current_runtime(cls, memory_limit_bytes: int) -> "PipelineMemoryProfile":
        return cls(_MEMORY_PROFILE_TOKEN, memory_limit_bytes)

    @property
    def memory_limit_bytes(self) -> int:
        return self._memory_limit_bytes

    @property
    def overhead_profile_fingerprint(self) -> str:
        return self._overhead_profile_fingerprint

    @property
    def fixed_runtime_overhead_bytes(self) -> int:
        return self._fixed_runtime_overhead_bytes

    @property
    def per_event_reference_overhead_bytes(self) -> int:
        return self._per_event_reference_overhead_bytes


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
        "_aggregate_peak_bytes",
        "_memory_profile_fingerprint",
    )

    def __init__(
        self,
        authority: object,
        dataset: SealedDatasetArtifact,
        capture_terminal: CaptureTerminalAck,
        aggregate_peak_bytes: int,
        memory_profile_fingerprint: str,
    ) -> None:
        if authority is not _PIPELINE_RESULT_TOKEN:
            raise PermissionError("PipelineResult can only be minted by compile_pipeline")
        if not isinstance(dataset, SealedDatasetArtifact):
            raise TypeError("dataset must be SealedDatasetArtifact")
        if not isinstance(capture_terminal, CaptureTerminalAck):
            raise TypeError("capture_terminal must be CaptureTerminalAck")
        peak = _positive_int(aggregate_peak_bytes, "aggregate_peak_bytes")
        if (
            not isinstance(memory_profile_fingerprint, str)
            or len(memory_profile_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in memory_profile_fingerprint
            )
        ):
            raise ValueError("memory profile fingerprint must be a lowercase SHA-256")
        provenance = dataset.provenance
        count = provenance.end_sequence - provenance.start_sequence
        if not dataset.coverage.complete or dataset.coverage.total_cells != count:
            raise RuntimeError("pipeline dataset coverage differs from event interval")
        if (
            capture_terminal.produced_count != count
            or capture_terminal.drained_count != count
            or capture_terminal.ordered_metadata_digest
            != provenance.ordered_metadata_digest
        ):
            raise RuntimeError("pipeline terminal and dataset provenance differ")
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_capture_terminal", capture_terminal)
        object.__setattr__(self, "_aggregate_peak_bytes", peak)
        object.__setattr__(
            self,
            "_memory_profile_fingerprint",
            memory_profile_fingerprint,
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PipelineResult is immutable")

    @property
    def dataset(self) -> SealedDatasetArtifact:
        return self._dataset

    @property
    def capture_terminal(self) -> CaptureTerminalAck:
        return self._capture_terminal

    @property
    def aggregate_peak_bytes(self) -> int:
        return self._aggregate_peak_bytes

    @property
    def memory_profile_fingerprint(self) -> str:
        return self._memory_profile_fingerprint


def _dataset_storage_bytes(schema: DatasetSchema) -> int:
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
    dataset_bytes = _dataset_storage_bytes(contract.dataset_schema)
    metadata_bytes = (
        events * contract.event_adapter.metadata_contract.max_retained_nbytes
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
        if hasattr(primary, "add_note"):
            primary.add_note(f"preflight software teardown also failed: {error!r}")


@dataclass
class ExactCaptureTransaction:
    """Concrete reusable owner of one finite exact capture/materialization path."""

    session: CaptureSession
    reservation: ExactReservation
    cursor: AcquisitionCursor
    builder: DatasetBuilder
    port: BoundCapturePort
    contract: CaptureStreamContract
    aggregate_peak_bytes: int
    memory_profile_fingerprint: str

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
            != self.contract.event_adapter.metadata_contract.fingerprint
        ):
            raise RuntimeError("sealed dataset metadata contract differs from capture")
        if (
            provenance.ordered_metadata_digest
            != completion.terminal.ordered_metadata_digest
        ):
            raise RuntimeError("sealed dataset metadata digest differs from capture")
        return PipelineResult(
            _PIPELINE_RESULT_TOKEN,
            dataset,
            completion.terminal,
            self.aggregate_peak_bytes,
            self.memory_profile_fingerprint,
        )

    def fail(self, error: BaseException) -> None:
        try:
            self.session.fail(error)
        except BaseException as failure_error:
            if hasattr(error, "add_note"):
                error.add_note(f"capture poison also failed: {failure_error!r}")

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
            contract.dataset_schema,
            DatasetMode.FINITE_EXACT,
            event_adapter=contract.event_adapter,
            expected_cells=contract.expected_cells,
        )
        session.bind_materializer(builder.exact_readiness())
        return ExactCaptureTransaction(
            session,
            reservation,
            cursor,
            builder,
            port,
            contract,
            aggregate_peak,
            spec.materializer.memory.overhead_profile_fingerprint,
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


__all__ = [
    "BoundMeasurement",
    "compile_pipeline",
    "DatasetMaterializerSpec",
    "ExactCaptureTransaction",
    "estimate_pipeline_peak_bytes",
    "MeasurementDefinition",
    "MinimalPipelineSpec",
    "open_exact_capture",
    "PipelineMemoryProfile",
    "PipelineResult",
    "resolve_measurement_definition",
]
