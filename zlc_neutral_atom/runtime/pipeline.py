"""Minimal flat Measurement -> exact Dataset pipeline compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    DatasetSchema,
    MONITOR_HISTORY,
    PointLayout,
    REPEAT,
)
from zlc_storage import (
    canonical_text as _canonical_text,
    nonnegative_integer as _nonnegative_int,
    positive_integer as _positive_int,
    sha256_text,
)

from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.camera_operator import (
    CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT,
)
from zlc_neutral_atom.acquisition.camera import (
    CameraAcquisitionMode,
    decode_camera_capture_spec,
)

from ._failure import record_secondary_failure, safe_error_summary
from .capture import (
    BoundCapturePort,
    CameraCapabilityEvidence,
    CameraCaptureProvenance,
    CaptureCompletion,
    CaptureSession,
    FrozenCaptureSpec,
    CameraCaptureContract,
    CaptureTerminalAck,
)
from .dataset import (
    DatasetBuilder,
    DatasetCellSchedule,
    FrozenDatasetEdge,
    MonitorDataset,
    SealedDatasetArtifact,
    dataset_storage_nbytes,
    mutable_dataset_storage_nbytes,
)
from .cleanup import CleanupReport
from .run import RunContext, RunPlan
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
    capture_contract: CameraCaptureContract
    capture_spec: FrozenCaptureSpec

    def __post_init__(self) -> None:
        if not isinstance(self.definition, MeasurementDefinition):
            raise TypeError("definition must be MeasurementDefinition")
        if not isinstance(self.capture_port, BoundCapturePort):
            raise TypeError("capture_port must be BoundCapturePort")
        if not isinstance(self.capture_contract, CameraCaptureContract):
            raise TypeError("capture_contract must be CameraCaptureContract")
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
        if (
            self.capture_contract.camera_provenance.camera_arm_spec_fingerprint
            != self.capture_spec.digest
        ):
            raise ValueError(
                "camera provenance arm spec differs from FrozenCaptureSpec"
            )

@dataclass(frozen=True)
class MinimalPipelineSpec:
    name: str
    measurement: BoundMeasurement
    block_id: BlockId
    memory_limit_bytes: int
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        _canonical_text(self.name, "name")
        if not isinstance(self.measurement, BoundMeasurement):
            raise TypeError("measurement must be BoundMeasurement")
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        object.__setattr__(
            self,
            "memory_limit_bytes",
            _positive_int(self.memory_limit_bytes, "memory_limit_bytes"),
        )


@dataclass(frozen=True)
class CapturePreviewSpec:
    """Process-local capacity-one live view attached to an exact capture."""

    block_id: BlockId
    dataset_edge: FrozenDatasetEdge
    downstream_peak_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.dataset_edge, FrozenDatasetEdge):
            raise TypeError("dataset_edge must be FrozenDatasetEdge")
        if self.dataset_edge.cell_schedule is not None:
            raise ValueError("capture preview requires a schedule-free dataset edge")
        schema = self.dataset_edge.schema
        if (
            schema.repeat_axis.size != 1
            or len(schema.point_axes) != 1
            or schema.point_axes[0].role != MONITOR_HISTORY
            or schema.point_axes[0].size != 1
            or schema.point_layout != PointLayout.rect_c((1,))
        ):
            raise ValueError(
                "capture preview requires capacity-one (R=1, MONITOR_HISTORY=1) storage"
            )
        object.__setattr__(
            self,
            "downstream_peak_bytes",
            _nonnegative_int(
                self.downstream_peak_bytes,
                "downstream_peak_bytes",
            ),
        )

    @classmethod
    def for_capture(
        cls,
        capture: MinimalPipelineSpec,
        *,
        block_id: BlockId,
        downstream_peak_bytes: int,
    ) -> "CapturePreviewSpec":
        """Derive the only admitted preview schema without changing cell data axes."""

        if not isinstance(capture, MinimalPipelineSpec):
            raise TypeError("capture must be MinimalPipelineSpec")
        contract = capture.measurement.capture_contract
        schema = DatasetSchema(
            AxisSpec(
                AxisId("live-preview.repeat"),
                "Live preview repeat storage",
                REPEAT,
                1,
                (0,),
            ),
            (
                AxisSpec(
                    AxisId("live-preview.history"),
                    "Live preview history",
                    MONITOR_HISTORY,
                    1,
                    (0,),
                ),
            ),
            PointLayout.rect_c((1,)),
            contract.dataset_schema.cell_schema,
        )
        return cls(
            block_id,
            FrozenDatasetEdge(schema, contract.dataset_edge.event_adapter),
            downstream_peak_bytes,
        )


class CapturePreviewPort(Protocol):
    """Workbench-owned display attachment; ``bind`` transfers dataset lifetime.

    ``fail`` is the terminal release operation after a successful bind.  The
    capture transaction retains only a non-owning ingest handle and falls back
    to closing it itself only when the port cannot accept that terminal call.
    """

    @property
    def spec(self) -> CapturePreviewSpec: ...

    def bind(
        self,
        dataset: MonitorDataset,
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None: ...

    def updated(self) -> None: ...

    def fail(self, message: str) -> None: ...

    def source_terminal(self) -> None: ...


_PIPELINE_RESULT_TOKEN = object()


class PipelineResult:
    """Opaque compiler-minted proof that capture terminal and dataset agree."""

    __slots__ = (
        "_dataset",
        "_capture_completion",
        "_direct_raw_capture",
    )

    def __init__(
        self,
        authority: object,
        dataset: SealedDatasetArtifact,
        capture_completion: CaptureCompletion,
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
        direct_raw_capture = (
            capture_completion.direct_terminal_consumer
            and dataset.block.schema == capture_completion.source_dataset_schema
            and capture_completion.source_event_adapter_operator_fingerprint
            == CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT
        )
        # Install every immutable result reference before the final no-fail
        # authority commit.  The consumed completion has its live session and
        # reservation references cleared by that commit, so retaining it avoids
        # nine mirrored evidence fields without extending the runtime graph.
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_capture_completion", capture_completion)
        object.__setattr__(self, "_direct_raw_capture", direct_raw_capture)
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
        return self._capture_completion.terminal

    @property
    def run_id(self) -> str:
        return self._capture_completion.trace_binding.run_id

    @property
    def source_dataset_schema(self) -> DatasetSchema:
        return self._capture_completion.source_dataset_schema

    @property
    def camera_provenance(self) -> CameraCaptureProvenance:
        return self._capture_completion.camera_provenance

    @property
    def camera_capability_evidence(self) -> CameraCapabilityEvidence:
        return self._capture_completion.camera_capability_evidence

    @property
    def camera_arm_spec(self) -> FrozenCaptureSpec:
        return self._capture_completion.camera_arm_spec

    @property
    def source_cell_schedule(self) -> DatasetCellSchedule:
        return self._capture_completion.source_cell_schedule

    @property
    def source_event_span(self) -> EventSpanRef:
        return self._capture_completion.source_event_span

    @property
    def processor_stages(self) -> tuple[ProcessorStageProvenance, ...]:
        return self._capture_completion.processor_stages

    @property
    def chain_contract_digest(self) -> str:
        return self._capture_completion.chain_contract_digest

    @property
    def is_direct_raw_capture(self) -> bool:
        return self._direct_raw_capture

def _estimate_capture_preview_peak_bytes(spec: CapturePreviewSpec) -> int:
    """Peak increment owned by the capacity-one monitor/display attachment."""

    if not isinstance(spec, CapturePreviewSpec):
        raise TypeError("spec must be CapturePreviewSpec")
    edge = spec.dataset_edge
    schema = edge.schema
    return (
        edge.payload_max_retained_nbytes
        + mutable_dataset_storage_nbytes(schema)
        + dataset_storage_nbytes(schema)
        + edge.metadata_max_retained_nbytes
        + spec.downstream_peak_bytes
    )


def estimate_pipeline_peak_bytes(
    spec: MinimalPipelineSpec,
    *,
    preview_spec: CapturePreviewSpec | None = None,
) -> int:
    """Conservative peak of buffers whose sizes are owned by this pipeline.

    This is not a claim about interpreter or third-party allocator overhead.
    Every term below comes from a frozen byte contract or ndarray geometry;
    guessed per-object constants are deliberately excluded.
    """

    if not isinstance(spec, MinimalPipelineSpec):
        raise TypeError("spec must be MinimalPipelineSpec")
    contract = spec.measurement.capture_contract
    events = contract.total_events
    dataset_bytes = dataset_storage_nbytes(contract.dataset_schema)
    mutable_dataset_bytes = mutable_dataset_storage_nbytes(
        contract.dataset_schema
    )
    metadata_bytes = (
        events * contract.dataset_edge.metadata_max_retained_nbytes
    )
    exact_peak = (
        contract.estimated_transport_bytes
        + mutable_dataset_bytes
        + dataset_bytes
        + metadata_bytes
    )
    return (
        exact_peak
        if preview_spec is None
        else exact_peak + _estimate_capture_preview_peak_bytes(preview_spec)
    )


def _require_pipeline_memory_budget(
    spec: MinimalPipelineSpec,
    preview_spec: CapturePreviewSpec | None = None,
) -> None:
    """Compute the owner-derived peak and reject it before any allocation."""

    peak = estimate_pipeline_peak_bytes(spec, preview_spec=preview_spec)
    limit = spec.memory_limit_bytes
    if peak > limit:
        raise MemoryError(f"pipeline peak budget {peak} exceeds limit {limit}")


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
    contract: CameraCaptureContract
    preview_dataset: MonitorDataset | None = None
    preview_port: CapturePreviewPort | None = None

    def start(self, context: RunContext) -> None:
        self.session.prepare(context)
        self.session.start(context)

    def capture_all(self, context: RunContext) -> None:
        for _ordinal in range(self.contract.total_events):
            context.checkpoint()
            self.session.capture_next(context)
            self.builder.consume(
                self.cursor.next(
                    timeout=self.port.capability.max_blocking_call_seconds
                )
            )
            preview = self.preview_dataset
            if preview is not None:
                try:
                    preview.ingest_latest()
                    port = self.preview_port
                    if port is None:
                        raise RuntimeError("capture preview port disappeared")
                    port.updated()
                except BaseException as error:
                    self._detach_preview(error)

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
        )

    def fail(self, error: BaseException) -> None:
        self._detach_preview(error)
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

        self._detach_preview(error)
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

    def _detach_preview(self, error: BaseException) -> None:
        dataset, self.preview_dataset = self.preview_dataset, None
        port, self.preview_port = self.preview_port, None
        if port is not None:
            try:
                port.fail(safe_error_summary(error))
                # bind() transferred lifetime ownership to the preview port.
                dataset = None
            except BaseException:
                pass
        if dataset is not None:
            try:
                dataset.close()
            except BaseException:
                pass

    def _finish_preview_source(self) -> None:
        # bind() transfers dataset lifetime to the Workbench slot.  A normal
        # terminal therefore drops only this transaction's non-owning handle;
        # the last visible snapshot remains available until the panel closes.
        self.preview_dataset = None
        port, self.preview_port = self.preview_port, None
        if port is not None:
            try:
                port.source_terminal()
            except BaseException:
                pass

    def settle_preview_after_cleanup(
        self,
        report: CleanupReport,
        primary: BaseException | None,
    ) -> None:
        """Publish a normal terminal only after aggregate cleanup succeeded."""

        if primary is not None:
            self._detach_preview(primary)
        elif report.errors:
            self._detach_preview(report.errors[0])
        elif report.decisions:
            decision = report.decisions[0]
            self._detach_preview(
                RuntimeError(
                    "capture cleanup reported an unsafe terminal state: "
                    f"{decision.reason}"
                )
            )
        else:
            self._finish_preview_source()


def _capture_preview_spec(
    preview: CapturePreviewPort | None,
    capture: MinimalPipelineSpec,
) -> CapturePreviewSpec | None:
    if preview is None:
        return None
    spec = getattr(preview, "spec", None)
    if not isinstance(spec, CapturePreviewSpec):
        raise TypeError("preview.spec must be CapturePreviewSpec")
    exact_edge = capture.measurement.capture_contract.dataset_edge
    if (
        spec.dataset_edge.schema.cell_schema is not exact_edge.schema.cell_schema
        or spec.dataset_edge.event_adapter is not exact_edge.event_adapter
    ):
        raise ValueError(
            "capture preview must share the exact capture cell schema and event adapter"
        )
    return spec


def _notify_preview_failure(
    preview: CapturePreviewPort | None,
    error: BaseException,
) -> None:
    if preview is None:
        return
    try:
        preview.fail(safe_error_summary(error))
    except BaseException:
        pass


def _prepare_exact_capture(
    spec: MinimalPipelineSpec,
    context: RunContext,
    preview: CapturePreviewPort | None,
    preview_spec: CapturePreviewSpec | None,
) -> ExactCaptureTransaction:
    try:
        return _allocate_exact_capture(spec, context, preview, preview_spec)
    except BaseException as error:
        _notify_preview_failure(preview, error)
        raise


def _open_exact_capture_transaction(
    spec: MinimalPipelineSpec,
    context: RunContext,
    *,
    preview: CapturePreviewPort | None,
    preview_spec: CapturePreviewSpec | None,
) -> ExactCaptureTransaction:
    """Allocate the single reservation/materializer transaction without touching hardware."""

    if not isinstance(spec, MinimalPipelineSpec):
        raise TypeError("spec must be MinimalPipelineSpec")
    _require_pipeline_memory_budget(spec, preview_spec)
    return _prepare_exact_capture(spec, context, preview, preview_spec)


def _allocate_exact_capture(
    spec: MinimalPipelineSpec,
    context: RunContext,
    preview: CapturePreviewPort | None = None,
    preview_spec: CapturePreviewSpec | None = None,
) -> ExactCaptureTransaction:
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
            spec.block_id,
            reservation,
            contract.dataset_edge,
        )
        readiness = builder.exact_readiness()
        session.bind_exact_consumer(readiness)
        preview_dataset = None
        if preview is not None:
            assert preview_spec is not None
            tap = None
            try:
                tap = session.stream.monitor(
                    max_events=1,
                    max_bytes=preview_spec.dataset_edge.payload_max_retained_nbytes,
                )
                preview_dataset = MonitorDataset.append_window(
                    preview_spec.block_id,
                    tap,
                    preview_spec.dataset_edge,
                )
                preview.bind(
                    preview_dataset,
                    run_id=context.run_id.value,
                    causation_domain_id=session.stream.generation.value,
                )
            except BaseException as preview_error:
                if preview_dataset is not None:
                    try:
                        preview_dataset.close()
                    except BaseException:
                        pass
                elif tap is not None:
                    try:
                        tap.close()
                    except BaseException:
                        pass
                preview_dataset = None
                _notify_preview_failure(preview, preview_error)
                preview = None
        return ExactCaptureTransaction(
            session,
            reservation,
            cursor,
            builder,
            port,
            contract,
            preview_dataset,
            preview,
        )
    except BaseException as error:
        _release_preflight_software(session, reservation, builder, error)
        raise


def _require_direct_capture(measurement: BoundMeasurement) -> None:
    """Reject a hardware-timed source outside its timing coordinator."""

    camera_spec = decode_camera_capture_spec(measurement.capture_spec)
    if camera_spec.mode is CameraAcquisitionMode.EXTERNAL_TRIGGERED:
        raise ValueError(
            "external-trigger capture requires a pulse timing coordinator"
        )


def compile_pipeline(
    spec: MinimalPipelineSpec,
    *,
    preview: CapturePreviewPort | None = None,
) -> RunPlan:
    """Compile the one supported finite exact path into one flat RunPlan."""

    if not isinstance(spec, MinimalPipelineSpec):
        raise TypeError("spec must be MinimalPipelineSpec")
    _require_direct_capture(spec.measurement)
    preview_spec = _capture_preview_spec(preview, spec)
    _require_pipeline_memory_budget(spec, preview_spec)
    port = spec.measurement.capture_port

    def preflight(context: RunContext) -> ExactCaptureTransaction:
        return _prepare_exact_capture(spec, context, preview, preview_spec)

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
        primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return port.verify_idle(context)
        try:
            report = prepared.cleanup(context)
        except BaseException as error:
            prepared._detach_preview(error)
            raise
        prepared.settle_preview_after_cleanup(report, primary)
        return report

    return RunPlan(
        name=spec.name,
        resource_claims=(port.resource_claim,),
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
    )


__all__ = [
    "BoundMeasurement",
    "CapturePreviewPort",
    "CapturePreviewSpec",
    "compile_pipeline",
    "ExactCaptureTransaction",
    "estimate_pipeline_peak_bytes",
    "finalize_pipeline_result",
    "MeasurementDefinition",
    "MinimalPipelineSpec",
    "PipelineResult",
]
