"""Finite exact camera-to-occupancy acquisition.

Generic runtime owners prove ordering, coverage, provenance, and terminal state;
this module owns only the coherent counts/occupied projection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from zlc_data import BlockId, ComponentValidity, DataBlock, DatasetSchema, OwnedSnapshot
from zlc_storage import canonical_text

from zlc_neutral_atom.devices.camera.contract import (
    CameraFrameMetadata,
    CameraFrameMetadataContract,
)
from zlc_neutral_atom.processing.stream import ExactStreamProcessorWorker
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.logic_nodes.camera_capture.session import (
    CaptureSession,
    CaptureSessionState,
    open_capture_session,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellSchedule,
    OrderedDatasetMetadataHasher,
)
from zlc_neutral_atom.logic_nodes.camera_capture.pipeline import (
    BoundMeasurement,
    PipelineResult,
    _require_passive_external_capture,
    finalize_pipeline_result,
)
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
    notify_preview_failure,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ExactReservation,
    ReservationState,
    TraceBinding,
)

from zlc_neutral_atom.logic_nodes.calibration.calibration import ReadoutModelKind
from zlc_neutral_atom.logic_nodes.calibration.reference import CalibrationArtifactRef
from .processor import (
    BoundOccupancyProcessor,
    OccupancyProcessorSpec,
    OccupancySample,
    bind_occupancy_processor,
    resolve_occupancy_processor_schema,
)


@dataclass(frozen=True, slots=True)
class OccupancyPipelineSpec:
    name: str
    measurement: BoundMeasurement
    processor: OccupancyProcessorSpec
    counts_block_id: BlockId
    occupied_block_id: BlockId

    def __post_init__(self) -> None:
        canonical_text(self.name, "name")
        if not isinstance(self.measurement, BoundMeasurement):
            raise TypeError("measurement must be BoundMeasurement")
        if not isinstance(self.processor, OccupancyProcessorSpec):
            raise TypeError("processor must be OccupancyProcessorSpec")
        if not isinstance(self.counts_block_id, BlockId) or not isinstance(self.occupied_block_id, BlockId):
            raise TypeError("counts_block_id and occupied_block_id must be BlockId")
        if self.counts_block_id == self.occupied_block_id:
            raise ValueError("counts and occupied require distinct BlockId values")


@dataclass(frozen=True, slots=True)
class OccupancyDataset:
    """Two fields and camera metadata from one exact event domain."""

    counts: OwnedSnapshot
    occupied: OwnedSnapshot
    cell_schedule: DatasetCellSchedule
    event_metadata: tuple[CameraFrameMetadata, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.counts, OwnedSnapshot) or not isinstance(self.occupied, OwnedSnapshot):
            raise TypeError("counts and occupied must be OwnedSnapshot")
        if not isinstance(self.cell_schedule, DatasetCellSchedule):
            raise TypeError("cell_schedule must be DatasetCellSchedule")
        metadata = tuple(self.event_metadata)
        if any(not isinstance(item, CameraFrameMetadata) for item in metadata):
            raise TypeError("event_metadata must contain CameraFrameMetadata")
        left, right = self.counts.block.schema, self.occupied.block.schema
        left_domain = (left.repeat_axis, left.point_axes, left.point_layout, left.cell_schema.data_axes)
        right_domain = (right.repeat_axis, right.point_axes, right.point_layout, right.cell_schema.data_axes)
        if left_domain != right_domain:
            raise ValueError("occupancy fields do not share one sampling/data domain")
        if (self.counts.ref.stream_generation, self.counts.ref.revision) != (
            self.occupied.ref.stream_generation, self.occupied.ref.revision
        ):
            raise ValueError("occupancy fields do not share generation/revision")
        if self.counts.block.validity is not self.occupied.block.validity:
            raise ValueError("occupancy fields must share one validity authority")
        self.cell_schedule.validate_schema(left)
        if len(metadata) != len(self.cell_schedule):
            raise ValueError("occupancy metadata does not cover the cell schedule")
        object.__setattr__(self, "event_metadata", metadata)

@dataclass(frozen=True, slots=True)
class OccupancyPipelineResult:
    pipeline: PipelineResult
    dataset: OccupancyDataset
    calibration_reference: CalibrationArtifactRef
    model_kind: ReadoutModelKind

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, PipelineResult) or not isinstance(self.dataset, OccupancyDataset):
            raise TypeError("pipeline and dataset have the wrong result type")
        if self.dataset.counts is not self.pipeline.dataset.snapshot:
            raise ValueError("occupancy dataset belongs to another pipeline result")
        if not isinstance(self.calibration_reference, CalibrationArtifactRef):
            raise TypeError("calibration_reference must be CalibrationArtifactRef")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")


@dataclass(slots=True)
class ExactOccupancyTransaction:
    """Run-local owner of one exact camera/processor/materializer chain."""

    spec: OccupancyPipelineSpec
    session: CaptureSession
    bound: BoundOccupancyProcessor
    worker: ExactStreamProcessorWorker | None
    worker_deadline_monotonic: float
    preview: ExactDatasetPreviewPort | None = None

    def start(self, context: RunContext) -> None:
        self.session.prepare(context)
        self.session.start(context)

    def capture_next(self, context: RunContext) -> None:
        """Request exactly one frame from the armed camera/processor chain."""

        context.checkpoint()
        self.session.capture_next(context)

    def capture_all(self, context: RunContext) -> None:
        for _ordinal in range(self.spec.measurement.capture_contract.total_events):
            self.capture_next(context)

    def complete(self, context: RunContext) -> "ExecutedOccupancy":
        if self.worker is None:
            raise RuntimeError("occupancy transaction is complete or released")
        completion = self.session.complete(context)
        sealed = self.worker.finish(
            completion.eos,
            _remaining_worker_seconds(
                context,
                self.worker_deadline_monotonic,
            ),
        )
        pipeline = finalize_pipeline_result(
            dataset=sealed,
            capture_completion=completion,
        )
        cell_schedule = self.bound.output_edge.cell_schedule
        if cell_schedule is None:
            raise RuntimeError("exact occupancy edge has no frozen cell schedule")
        metadata_contract = (
            self.spec.measurement.capture_contract.dataset_edge.metadata_contract
        )
        if not isinstance(metadata_contract, CameraFrameMetadataContract):
            raise TypeError("occupancy capture uses another metadata contract")
        return ExecutedOccupancy(
            pipeline=pipeline,
            occupied_block_id=self.spec.occupied_block_id,
            occupied_schema=_occupied_schema(self.bound),
            cell_schedule=cell_schedule,
            source_metadata_contract=metadata_contract,
            calibration_reference=self.bound.calibration_reference,
            model_kind=self.bound.model_kind,
        )

    def fail(self, error: BaseException) -> None:
        self._fail_preview(error)
        if self.session.state is CaptureSessionState.COMPLETED:
            return
        try:
            self.session.fail(error)
        except BaseException as secondary:
            record_secondary_failure(error, "capture poison also failed", secondary)

    def cleanup(self, context: RunContext) -> CleanupReport:
        errors: list[BaseException] = []
        worker = self.worker
        if worker is not None:
            try:
                worker.close(2.0)
            except BaseException as error:
                errors.append(error)
        try:
            report = self.session.cleanup(context)
        except BaseException as primary:
            for error in errors:
                record_secondary_failure(primary, "processor teardown also failed", error)
            raise
        if errors and worker is not None:
            try:
                worker.close(2.0)
            except BaseException as error:
                errors.append(error)
        self.worker = None
        if not errors:
            return report
        return CleanupReport.complete(errors=(*report.errors, *errors))

    def _fail_preview(self, error: BaseException) -> None:
        preview, self.preview = self.preview, None
        notify_preview_failure(preview, error)

    def settle_preview_after_cleanup(
        self,
        report: CleanupReport,
        primary: BaseException | None,
    ) -> None:
        """Cleanup may reject preview; only post-safety may complete it."""

        if primary is not None:
            self._fail_preview(primary)
        elif report.errors:
            self._fail_preview(report.errors[0])



@dataclass(frozen=True, slots=True)
class ExecutedOccupancy:
    """Post-safety occupancy facts with no session, worker, or reservation."""

    pipeline: PipelineResult
    occupied_block_id: BlockId
    occupied_schema: DatasetSchema
    cell_schedule: DatasetCellSchedule
    source_metadata_contract: CameraFrameMetadataContract
    calibration_reference: CalibrationArtifactRef
    model_kind: ReadoutModelKind

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, PipelineResult):
            raise TypeError("pipeline must be PipelineResult")
        if not isinstance(self.occupied_block_id, BlockId):
            raise TypeError("occupied_block_id must be BlockId")
        if not isinstance(self.occupied_schema, DatasetSchema):
            raise TypeError("occupied_schema must be DatasetSchema")
        if not isinstance(self.source_metadata_contract, CameraFrameMetadataContract):
            raise TypeError("source_metadata_contract has another owner")
        if not isinstance(self.calibration_reference, CalibrationArtifactRef):
            raise TypeError("calibration_reference must be CalibrationArtifactRef")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        if not isinstance(self.cell_schedule, DatasetCellSchedule):
            raise TypeError("cell_schedule must be DatasetCellSchedule")
        self.cell_schedule.validate_schema(self.pipeline.dataset.block.schema)
        self.cell_schedule.validate_schema(self.occupied_schema)

def _remaining_worker_seconds(
    context: RunContext,
    deadline_monotonic: float,
) -> float:
    context.checkpoint()
    remaining = float(deadline_monotonic) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("occupancy processor capability deadline expired")
    return remaining


def _worker_deadline_monotonic(spec: OccupancyPipelineSpec) -> float:
    """Derive the private worker bound from frozen capture capability.

    The public Measurement request owns no timeout.  Its camera Port already
    attests a maximum duration for every prepare/read/terminal device call, and
    the capture contract freezes the exact event count.  The worker spans one
    prepare, one start, every read, and one terminal call, so their finite sum
    is the only deadline it needs.
    """

    contract = spec.measurement.capture_contract
    call_bound = spec.measurement.capture_port.capability.max_blocking_call_seconds
    return time.monotonic() + call_bound * (contract.total_events + 3)


def _occupied_schema(bound: BoundOccupancyProcessor) -> DatasetSchema:
    counts = bound.output_schema
    return DatasetSchema(
        counts.repeat_axis, counts.point_axes, counts.point_layout,
        bound.output_payload_contract.occupied_schema,
    )


def _occupancy_preview_spec(
    spec: OccupancyPipelineSpec,
    preview: ExactDatasetPreviewPort | None,
) -> ExactDatasetPreviewSpec | None:
    if preview is None:
        return None
    try:
        preview_spec = getattr(preview, "spec", None)
        if not isinstance(preview_spec, ExactDatasetPreviewSpec):
            raise TypeError("preview.spec must be ExactDatasetPreviewSpec")
        terminal = getattr(preview, "terminal", None)
        if not isinstance(terminal, bool):
            raise TypeError("preview.terminal must be bool")
        if terminal:
            raise RuntimeError("occupancy preview is already terminal")
        source_schema = resolve_occupancy_processor_schema(
            spec.processor,
            spec.measurement.capture_contract.dataset_schema,
        ).counts_schema
        if preview_spec.source_schema_fingerprint != source_schema.fingerprint:
            raise ValueError(
                "occupancy preview schema differs from exact counts output"
            )
        return preview_spec
    except BaseException as error:
        notify_preview_failure(preview, error)
        raise


def _settle_unbound_preview(
    preview: ExactDatasetPreviewPort | None,
    report: CleanupReport,
    primary: BaseException | None,
) -> CleanupReport:
    failure = primary
    if failure is None and report.errors:
        failure = report.errors[0]
    if failure is None:
        failure = RuntimeError("occupancy preview never reached an exact source")
    if failure is not None:
        notify_preview_failure(preview, failure)
    return report


def _finish_preview_after_post_safety(
    preview: ExactDatasetPreviewPort | None,
) -> None:
    if preview is None:
        return
    try:
        if preview.terminal:
            return
        preview.source_terminal()
    except BaseException as error:
        notify_preview_failure(preview, error)


def _release(reservation: ExactReservation | None) -> None:
    if reservation is None or reservation.state is ReservationState.RELEASED:
        return
    if reservation.state not in (ReservationState.COMPLETED, ReservationState.FAILED, ReservationState.CANCELLED):
        reservation.abort()
    if reservation.state is not ReservationState.RELEASED:
        reservation.release()


def _failed_open(session: CaptureSession, worker: ExactStreamProcessorWorker | None,
                 builder: DatasetBuilder | None, source: ExactReservation | None,
                 output: ExactReservation | None, primary: BaseException) -> None:
    errors: list[BaseException] = []
    try:
        session.fail(primary)
    except BaseException as error:
        errors.append(error)
    try:
        if worker is not None:
            worker.close(2.0)
        elif builder is not None:
            builder.close()
        else:
            _release(output)
    except BaseException as error:
        errors.append(error)
    if worker is None:
        try:
            _release(source)
        except BaseException as error:
            errors.append(error)
    for error in errors:
        record_secondary_failure(primary, "preflight teardown also failed", error)


def _open_exact_occupancy(
    spec: OccupancyPipelineSpec,
    context: RunContext,
    *,
    preview: ExactDatasetPreviewPort | None = None,
    preview_spec: ExactDatasetPreviewSpec | None = None,
) -> ExactOccupancyTransaction:
    """Allocate the complete software chain without touching hardware."""

    if not isinstance(spec, OccupancyPipelineSpec):
        raise TypeError("spec must be OccupancyPipelineSpec")
    if (preview is None) != (preview_spec is None):
        raise ValueError("preview and preview_spec must be present together")
    measurement, contract = spec.measurement, spec.measurement.capture_contract
    session = open_capture_session(
        measurement.capture_port,
        contract,
        TraceBinding(context.run_id.value, contract.source_id),
        measurement.capture_spec,
    )
    worker_deadline = _worker_deadline_monotonic(spec)
    worker = builder = source = output_reservation = None
    try:
        capture_input = session.processor_input_binding
        bound = bind_occupancy_processor(spec.processor, capture_input)
        if (
            preview_spec is not None
            and preview_spec.source_schema_fingerprint
            != bound.output_schema.fingerprint
        ):
            raise ValueError(
                "occupancy preview schema differs from bound counts output"
            )
        source = session.reserve_exact()
        source_cursor = source.activate()
        payload = bound.output_payload_contract
        output_stream, producer = AcquisitionStream.create(
            bound.output_stream_id, payload,
            join_key_contract=capture_input.join_key_contract,
        )
        output_reservation = output_stream.reserve(
            total_events=contract.total_events,
            trace_binding=TraceBinding(context.run_id.value, bound.output_source_id),
        )
        output_cursor = output_reservation.activate()
        builder = DatasetBuilder(spec.counts_block_id, output_reservation, bound.output_edge)
        bound_preview = preview
        if bound_preview is not None:
            assert preview_spec is not None
            try:
                bound_preview.bind(
                    builder.open_preview_reader(),
                    run_id=context.run_id.value,
                    causation_domain_id=output_stream.generation.value,
                )
            except BaseException as preview_error:
                notify_preview_failure(bound_preview, preview_error)
                bound_preview = None
        _remaining_worker_seconds(context, worker_deadline)
        worker = bound.create_exact_worker(
            source, source_cursor, output_producer=producer, output_cursor=output_cursor,
            output_builder=builder, deadline_monotonic=worker_deadline,
            cancellation=context.cancellation,
        )
        worker.start()
        readiness = worker.exact_readiness()
        session.bind_exact_consumer(readiness)
        return ExactOccupancyTransaction(
            spec,
            session,
            bound,
            worker,
            worker_deadline,
            bound_preview,
        )
    except BaseException as error:
        _failed_open(session, worker, builder, source, output_reservation, error)
        notify_preview_failure(preview, error)
        raise


def finalize_occupancy_result(context: PostSafetyContext,
                              executed: ExecutedOccupancy) -> OccupancyPipelineResult:
    """Build occupied beside counts after generic terminal validation."""

    if not isinstance(executed, ExecutedOccupancy):
        raise TypeError("executed must be ExecutedOccupancy")
    pipeline = executed.pipeline
    terminal, counts = pipeline.dataset, pipeline.dataset.snapshot
    validity = counts.block.validity
    if not isinstance(validity, ComponentValidity):
        raise TypeError("occupancy counts block requires ComponentValidity")
    cells = executed.cell_schedule
    schema = executed.occupied_schema
    values = np.zeros(schema.physical_shape, dtype=bool)
    metadata_contract = executed.source_metadata_contract
    hasher = OrderedDatasetMetadataHasher(metadata_contract.fingerprint)
    event_metadata: list[CameraFrameMetadata] = []
    for cell, metadata in zip(cells, terminal.event_metadata, strict=True):
        context.checkpoint()
        if not isinstance(metadata, OccupancySample):
            raise TypeError("occupancy terminal contains another metadata type")
        occupied_validity = metadata.occupied.validity
        if not isinstance(occupied_validity, ComponentValidity):
            raise TypeError("occupied metadata requires ComponentValidity")
        location = (cell.repeat_index, cell.point_storage_index)
        if (validity.axis_ids != occupied_validity.axis_ids
                or not np.array_equal(validity.mask[location], occupied_validity.mask)):
            raise RuntimeError("counts and occupied validity differ at one dataset cell")
        hasher.update(metadata_contract.digest(metadata.metadata))
        values[location] = metadata.occupied.values
        event_metadata.append(metadata.metadata)
    if hasher.digest() != pipeline.capture_terminal.ordered_metadata_digest:
        raise RuntimeError("occupancy source metadata differs from physical capture")
    block = DataBlock(executed.occupied_block_id, counts.block.revision, values, validity, schema)
    occupied = OwnedSnapshot(block.ref(terminal.provenance.generation), block)
    result = OccupancyDataset(
        counts,
        occupied,
        cells,
        tuple(event_metadata),
    )
    context.checkpoint()
    return OccupancyPipelineResult(
        pipeline,
        result,
        executed.calibration_reference,
        executed.model_kind,
    )


def compile_occupancy_pipeline(
    spec: OccupancyPipelineSpec,
    *,
    preview: ExactDatasetPreviewPort | None = None,
) -> RunPlan:
    """Compile passive finite Camera -> occupancy into one flat RunPlan."""

    if not isinstance(spec, OccupancyPipelineSpec):
        raise TypeError("spec must be OccupancyPipelineSpec")
    _require_passive_external_capture(spec.measurement)
    preview_spec = _occupancy_preview_spec(spec, preview)
    port = spec.measurement.capture_port

    def execute(context: RunContext, prepared: ExactOccupancyTransaction) -> ExecutedOccupancy:
        try:
            prepared.start(context)
            prepared.capture_all(context)
            return prepared.complete(context)
        except BaseException as error:
            prepared.fail(error)
            raise

    def cleanup(context: RunContext, prepared: ExactOccupancyTransaction | None,
                primary: BaseException | None) -> CleanupReport:
        if prepared is None:
            return _settle_unbound_preview(
                preview,
                port.verify_idle(context),
                primary,
            )
        report = prepared.cleanup(context)
        prepared.settle_preview_after_cleanup(report, primary)
        return report

    def finalize(
        context: PostSafetyContext,
        executed: ExecutedOccupancy,
    ) -> OccupancyPipelineResult:
        if not isinstance(executed, ExecutedOccupancy):
            raise TypeError("occupancy finalize requires executed occupancy facts")
        if executed.pipeline.run_id != context.run_id.value:
            raise ValueError("executed occupancy result belongs to another Run")
        try:
            context.checkpoint()
            result = finalize_occupancy_result(context, executed)
        except BaseException as error:
            notify_preview_failure(preview, error)
            raise
        _finish_preview_after_post_safety(preview)
        return result

    def dispose_unfinalized(_executed: ExecutedOccupancy) -> None:
        notify_preview_failure(
            preview,
            RuntimeError(
                "passive Camera occupancy preview rejected before post-safety "
                "finalization"
            ),
        )

    return RunPlan(
        name=spec.name,
        resource_claims=(port.resource_claim,),
        bound_devices=(port.device,), preflight=lambda context: _open_exact_occupancy(
            spec,
            context,
            preview=preview,
            preview_spec=preview_spec,
        ),
        execute=execute, cleanup=cleanup, finalize=finalize,
        interrupt_operations=port.interrupt_operations,
        requires_final_commit=False,
        dispose_unfinalized=dispose_unfinalized,
    )


__all__ = [
    "compile_occupancy_pipeline",
    "ExecutedOccupancy",
    "ExactOccupancyTransaction",
    "finalize_occupancy_result",
    "OccupancyDataset",
    "OccupancyPipelineResult",
    "OccupancyPipelineSpec",
]
