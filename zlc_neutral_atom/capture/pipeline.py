"""Exact camera Dataset pipeline compiler."""

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
    nonnegative_integer as _nonnegative_integer,
)

from zlc_neutral_atom.catalog import MeasurementDefinition as _MeasurementDefinition
from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraCapabilityEvidence,
    FrozenCaptureSpec,
    decode_camera_capture_spec,
)
from zlc_neutral_atom.devices.camera.capture_port import (
    BoundCapturePort,
    CaptureTerminalAck,
)
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
    notify_preview_failure,
)

from zlc_neutral_atom.runtime._failure import record_secondary_failure, safe_error_summary
from .session import (
    CameraCaptureProvenance,
    CaptureCompletion,
    CaptureSession,
    open_capture_session,
    CameraCaptureContract,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellSchedule,
    FrozenDatasetEdge,
    MonitorDataset,
    SealedDatasetArtifact,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import RunContext, RunPlan
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    EventSpanRef,
    ExactReservation,
    ProcessorStageProvenance,
    ReservationState,
    TraceBinding,
)


@dataclass(frozen=True)
class BoundMeasurement:
    definition: _MeasurementDefinition
    capture_port: BoundCapturePort
    capture_contract: CameraCaptureContract
    capture_spec: FrozenCaptureSpec

    def __post_init__(self) -> None:
        if not isinstance(self.definition, _MeasurementDefinition):
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
            self.capture_spec.owner_fingerprint
            != self.capture_contract.capture_spec_owner_fingerprint
        ):
            raise ValueError("capture spec and camera contract owner differ")
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

    def __post_init__(self) -> None:
        _canonical_text(self.name, "name")
        if not isinstance(self.measurement, BoundMeasurement):
            raise TypeError("measurement must be BoundMeasurement")
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
@dataclass(frozen=True)
class CapturePreviewSpec:
    """Process-local single-cell live view attached to an exact capture.

    ``source_ordinals=None`` publishes every physical event.  A tuple admits
    only those frozen source ordinals before the preview dataset is ingested;
    the exact DatasetBuilder remains complete in either case.
    """

    block_id: BlockId
    dataset_edge: FrozenDatasetEdge
    source_ordinals: tuple[int, ...] | None = None

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
                "capture preview requires single-cell (R=1, MONITOR_HISTORY=1) storage"
            )
        if self.source_ordinals is None:
            return
        ordinals = tuple(
            _nonnegative_integer(ordinal, "source_ordinals entry")
            for ordinal in self.source_ordinals
        )
        if not ordinals:
            raise ValueError("source_ordinals cannot be empty")
        if any(left >= right for left, right in zip(ordinals, ordinals[1:])):
            raise ValueError("source_ordinals must be strictly increasing")
        object.__setattr__(self, "source_ordinals", ordinals)

    @staticmethod
    def dataset_edge_for_capture(
        capture: MinimalPipelineSpec,
    ) -> FrozenDatasetEdge:
        """Derive the safe preview edge without exposing the bound capture Port."""

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
        return FrozenDatasetEdge(schema, contract.dataset_edge.event_adapter)

class CapturePreviewPort(Protocol):
    """Workbench-owned display attachment; ``bind`` transfers dataset lifetime.

    ``fail`` is legal before or after ``bind`` and is thread-safe, idempotent,
    and first-failure-wins.  The capture transaction retains only a non-owning
    ingest handle and falls back to closing it itself only when the port cannot
    accept that terminal call.
    """

    @property
    def spec(self) -> CapturePreviewSpec: ...

    @property
    def terminal(self) -> bool: ...

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
        output_count = provenance.end_sequence - provenance.start_sequence
        if (
            not dataset.coverage.complete
            or dataset.coverage.total_cells != output_count
        ):
            raise RuntimeError("pipeline dataset coverage differs from event interval")
        source_count = (
            capture_completion.source_event_span.end_sequence
            - capture_completion.source_event_span.start_sequence
        )
        if (
            capture_terminal.produced_count != source_count
            or capture_terminal.drained_count != source_count
        ):
            raise RuntimeError(
                "pipeline terminal and root input provenance differ"
            )
        if (
            capture_completion.direct_terminal_consumer
            and output_count != source_count
        ):
            raise RuntimeError(
                "direct capture dataset cardinality differs from its source"
            )
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
    exact_preview_port: ExactDatasetPreviewPort | None = None

    def start(self, context: RunContext) -> None:
        self.session.prepare(context)
        self.session.start(context)

    def capture_next(self, context: RunContext) -> None:
        """Consume exactly one reserved physical capture event."""

        context.checkpoint()
        self.session.capture_next(context)
        delivery = self.cursor.next(
            timeout=self.port.capability.max_blocking_call_seconds
        )
        envelope = delivery.envelope
        self.builder.consume(delivery)
        preview = self.preview_dataset
        if preview is not None:
            try:
                port = self.preview_port
                if port is None:
                    raise RuntimeError("capture preview port disappeared")
                source_ordinal = self.contract.payload_contract.source_ordinal(
                    envelope.payload
                )
                selected = port.spec.source_ordinals
                if selected is None or source_ordinal in selected:
                    preview.ingest_latest(
                        account_skipped_events=selected is None,
                        expected_event_ref=envelope.ref,
                    )
                    port.updated()
            except BaseException as error:
                self._detach_preview(error)

    def capture_all(self, context: RunContext) -> None:
        for _ordinal in range(self.contract.total_events):
            self.capture_next(context)

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
        self._fail_previews(error)
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

        self._fail_previews(error)
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
        return CleanupReport.complete(
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
    def _fail_previews(self, error: BaseException) -> None:
        """Fail independent display sinks only when the capture itself failed."""

        self._detach_preview(error)
        exact, self.exact_preview_port = self.exact_preview_port, None
        notify_preview_failure(exact, error)

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
            self._fail_previews(primary)
        elif report.errors:
            self._fail_previews(report.errors[0])
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
    if not isinstance(getattr(preview, "terminal", None), bool):
        raise TypeError("preview.terminal must be bool")
    exact_edge = capture.measurement.capture_contract.dataset_edge
    if (
        spec.dataset_edge.schema.cell_schema is not exact_edge.schema.cell_schema
        or spec.dataset_edge.payload_contract is not exact_edge.payload_contract
        or spec.dataset_edge.operator_fingerprint
        != exact_edge.operator_fingerprint
    ):
        raise ValueError(
            "capture preview must share the exact capture cell schema and event adapter"
        )
    source_ordinals = spec.source_ordinals
    schedule_size = len(capture.measurement.capture_contract.cell_schedule)
    if source_ordinals is not None and source_ordinals[-1] >= schedule_size:
        raise ValueError(
            "capture preview source_ordinals exceed the frozen cell schedule"
        )
    return spec




def _settle_unbound_preview(
    preview: CapturePreviewPort | None,
    report: CleanupReport,
    primary: BaseException | None,
) -> CleanupReport:
    """Terminate a preview when Run cleanup has no prepared transaction."""

    failure: BaseException | None = primary
    if failure is None and report.errors:
        failure = report.errors[0]
    if failure is not None:
        notify_preview_failure(preview, failure)
    return report


def _admit_capture_preview(
    spec: MinimalPipelineSpec,
    preview: CapturePreviewPort | None,
) -> CapturePreviewSpec | None:
    """Validate one attached preview exactly once."""

    try:
        return _capture_preview_spec(preview, spec)
    except BaseException as error:
        notify_preview_failure(preview, error)
        raise


def _open_exact_capture_transaction(
    spec: MinimalPipelineSpec,
    context: RunContext,
    *,
    preview: CapturePreviewPort | None,
    preview_spec: CapturePreviewSpec | None,
    exact_preview: ExactDatasetPreviewPort | None = None,
    exact_preview_spec: ExactDatasetPreviewSpec | None = None,
) -> ExactCaptureTransaction:
    """Allocate the single reservation/materializer transaction without touching hardware."""

    if not isinstance(spec, MinimalPipelineSpec):
        raise TypeError("spec must be MinimalPipelineSpec")
    try:
        return _allocate_exact_capture(
            spec,
            context,
            preview,
            preview_spec,
            exact_preview,
            exact_preview_spec,
        )
    except BaseException as error:
        notify_preview_failure(preview, error)
        notify_preview_failure(exact_preview, error)
        raise


def _allocate_exact_capture(
    spec: MinimalPipelineSpec,
    context: RunContext,
    preview: CapturePreviewPort | None = None,
    preview_spec: CapturePreviewSpec | None = None,
    exact_preview: ExactDatasetPreviewPort | None = None,
    exact_preview_spec: ExactDatasetPreviewSpec | None = None,
) -> ExactCaptureTransaction:
    if (exact_preview is None) != (exact_preview_spec is None):
        raise ValueError("exact_preview and exact_preview_spec must be present together")
    measurement = spec.measurement
    port = measurement.capture_port
    contract = measurement.capture_contract
    session = open_capture_session(
        port,
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
        bound_exact_preview = exact_preview
        if bound_exact_preview is not None:
            assert exact_preview_spec is not None
            try:
                bound_exact_preview.bind(
                    builder.open_preview_reader(),
                    run_id=context.run_id.value,
                    causation_domain_id=session.stream.generation.value,
                )
            except BaseException as preview_error:
                notify_preview_failure(bound_exact_preview, preview_error)
                bound_exact_preview = None
        preview_dataset = None
        if preview is not None:
            assert preview_spec is not None
            tap = None
            try:
                tap = session.stream.monitor()
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
                notify_preview_failure(preview, preview_error)
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
            bound_exact_preview,
        )
    except BaseException as error:
        _release_preflight_software(session, reservation, builder, error)
        raise


def _require_passive_external_capture(measurement: BoundMeasurement) -> None:
    """Admit an exact camera reader whose trigger owner is outside this Run.

    A Camera Measurement is a pure grabber: it arms the selected camera and
    drains the next exact frame group, while an independently running hardware
    pulse owns trigger timing.  This compiler therefore claims no sequencer and
    never prepares or fires one.
    """

    camera_spec = decode_camera_capture_spec(measurement.capture_spec)
    if camera_spec.mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
        raise ValueError(
            "passive finite Camera measurement requires an external-trigger source"
        )


def compile_pipeline(
    spec: MinimalPipelineSpec,
    *,
    preview: CapturePreviewPort | None = None,
) -> RunPlan:
    """Compile the passive finite Camera path into one flat RunPlan.

    This plan owns only the camera.  Independently running hardware owns the
    trigger timing; explicit pulse-owned capture uses ``TriggeredCaptureSpec``.
    """

    if not isinstance(spec, MinimalPipelineSpec):
        raise TypeError("spec must be MinimalPipelineSpec")
    preview_spec = _admit_capture_preview(spec, preview)
    try:
        _require_passive_external_capture(spec.measurement)
    except BaseException as error:
        notify_preview_failure(preview, error)
        raise
    port = spec.measurement.capture_port

    def preflight(context: RunContext) -> ExactCaptureTransaction:
        return _open_exact_capture_transaction(
            spec,
            context,
            preview=preview,
            preview_spec=preview_spec,
        )

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
            try:
                report = port.verify_idle(context)
            except BaseException as error:
                notify_preview_failure(preview, error)
                raise
            return _settle_unbound_preview(preview, report, primary)
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
        requires_final_commit=False,
    )


def _admit_exact_dataset_preview(
    spec: MinimalPipelineSpec,
    preview: ExactDatasetPreviewPort | None,
) -> ExactDatasetPreviewSpec | None:
    """Validate an exact-builder preview against the capture dataset."""

    if preview is None:
        return None
    try:
        preview_spec = getattr(preview, "spec", None)
        if not isinstance(preview_spec, ExactDatasetPreviewSpec):
            raise TypeError("exact preview.spec must be ExactDatasetPreviewSpec")
        terminal = getattr(preview, "terminal", None)
        if not isinstance(terminal, bool):
            raise TypeError("exact preview.terminal must be bool")
        if terminal:
            raise RuntimeError("exact dataset preview is already terminal")
        source_schema = spec.measurement.capture_contract.dataset_schema
        if preview_spec.source_schema_fingerprint != source_schema.fingerprint:
            raise ValueError("exact preview schema differs from capture dataset")
        return preview_spec
    except BaseException as error:
        notify_preview_failure(preview, error)
        raise


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
    "finalize_pipeline_result",
    "MinimalPipelineSpec",
    "PipelineResult",
]
