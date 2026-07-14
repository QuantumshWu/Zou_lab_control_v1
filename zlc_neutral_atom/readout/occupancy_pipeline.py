"""Flat exact camera -> occupancy pipeline and coherent two-field result.

This module composes one existing camera Measurement with exactly one admitted
occupancy processor and one terminal DatasetBuilder.  It deliberately is not a
generic graph engine: one camera event produces one atomic OccupancySample, the
terminal edge materializes canonical counts, and post-safety finalization exposes
counts and occupied as two ordinary immutable snapshots from that same execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from zlc_storage import (
    canonical_text as _canonical_text,
    positive_real as _positive_seconds,
)

from zlc_data import (
    BlockId,
    ComponentValidity,
    DataBlock,
    DatasetSchema,
    OwnedSnapshot,
)

from zlc_neutral_atom.acquisition.camera import CameraFrameMetadata
from zlc_neutral_atom.processing.stream import ExactStreamProcessorWorker
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.runtime.capture import (
    CameraCapabilityEvidence,
    CameraCaptureProvenance,
    CaptureSession,
    CaptureSessionState,
    CaptureTerminalAck,
    FrozenCaptureSpec,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetMode,
    DatasetSealProvenance,
    OrderedDatasetMetadataHasher,
    SealedDatasetArtifact,
)
from zlc_neutral_atom.runtime.pipeline import (
    BoundMeasurement,
    PipelineMemoryAdmission,
    PipelineMemoryProfile,
    PipelineResult,
    admit_pipeline_memory,
    dataset_storage_nbytes,
    finalize_pipeline_result,
)
from zlc_neutral_atom.runtime.run import (
    CleanupReport,
    PostSafetyContext,
    RunContext,
    RunMode,
    RunPlan,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionStream,
    ExactReservation,
    ProducerFlowControl,
    ReservationState,
    TraceBinding,
)

from .calibration import ReadoutModelKind, calibration_retained_array_nbytes
from .calibration_reference import CalibrationArtifactRef
from .occupancy import (
    BoundOccupancyStreamProcessor,
    OccupancyDatasetField,
    OccupancyDatasetMetadata,
    OccupancyStreamProcessorSpec,
    bind_occupancy_stream_processor,
)


@dataclass(frozen=True)
class OccupancyDatasetMaterializerSpec:
    """The two user-visible block identities and one runtime memory policy."""

    counts_block_id: BlockId
    occupied_block_id: BlockId
    memory: PipelineMemoryProfile

    def __post_init__(self) -> None:
        if not isinstance(self.counts_block_id, BlockId):
            raise TypeError("counts_block_id must be BlockId")
        if not isinstance(self.occupied_block_id, BlockId):
            raise TypeError("occupied_block_id must be BlockId")
        if self.counts_block_id == self.occupied_block_id:
            raise ValueError("counts and occupied require distinct BlockId values")
        if not isinstance(self.memory, PipelineMemoryProfile):
            raise TypeError("memory must be PipelineMemoryProfile")


@dataclass(frozen=True)
class OccupancyPipelineSpec:
    """Reusable static intent for one finite exact occupancy acquisition."""

    name: str
    measurement: BoundMeasurement
    processor: OccupancyStreamProcessorSpec
    materializer: OccupancyDatasetMaterializerSpec
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.name, "name")
        if not isinstance(self.measurement, BoundMeasurement):
            raise TypeError("measurement must be BoundMeasurement")
        if not isinstance(self.processor, OccupancyStreamProcessorSpec):
            raise TypeError("processor must be OccupancyStreamProcessorSpec")
        if not isinstance(self.materializer, OccupancyDatasetMaterializerSpec):
            raise TypeError(
                "materializer must be OccupancyDatasetMaterializerSpec"
            )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_seconds(self.timeout_seconds, "timeout_seconds"),
        )
        contract = self.measurement.capture_contract
        if self.processor.output_stream_id == contract.stream_id:
            raise ValueError("occupancy output stream must differ from camera input")
        if self.processor.output_source_id == contract.source_id:
            raise ValueError("occupancy output source must differ from camera input")
        provenance = contract.camera_provenance
        if provenance is None:
            raise ValueError("occupancy pipeline requires broker-attested camera provenance")
        if provenance.binding != self.processor.readout_binding:
            raise ValueError("measurement and occupancy spec name different readout bindings")


_FROZEN_DATASET_TOKEN = object()


class FrozenOccupancyDataset:
    """Two coherent snapshots validated against one exact occupancy terminal."""

    __slots__ = (
        "_authority",
        "_counts",
        "_occupied",
        "_events",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("FrozenOccupancyDataset is final")

    def __init__(
        self,
        authority: object,
        *,
        counts: OwnedSnapshot,
        occupied: OwnedSnapshot,
        source_metadata: tuple[CameraFrameMetadata, ...],
        cell_schedule: tuple[DatasetCellAddress, ...],
    ) -> None:
        if authority is not _FROZEN_DATASET_TOKEN:
            raise PermissionError(
                "FrozenOccupancyDataset can only be minted by occupancy finalization"
            )
        if not isinstance(counts, OwnedSnapshot) or not isinstance(
            occupied,
            OwnedSnapshot,
        ):
            raise TypeError("counts and occupied must be OwnedSnapshot values")
        metadata = tuple(source_metadata)
        if any(not isinstance(item, CameraFrameMetadata) for item in metadata):
            raise TypeError("source_metadata must contain CameraFrameMetadata")
        schedule = tuple(cell_schedule)
        if any(not isinstance(item, DatasetCellAddress) for item in schedule):
            raise TypeError("cell_schedule must contain DatasetCellAddress values")
        if len(metadata) != len(schedule):
            raise ValueError("source metadata and cell schedule lengths differ")
        counts_schema = counts.block.schema
        occupied_schema = occupied.block.schema
        if (
            counts_schema.repeat_axis != occupied_schema.repeat_axis
            or counts_schema.point_axes != occupied_schema.point_axes
            or counts_schema.point_layout != occupied_schema.point_layout
        ):
            raise ValueError("occupancy fields do not share sampling axes/layout")
        if counts.ref.stream_generation != occupied.ref.stream_generation:
            raise ValueError("occupancy fields belong to different stream generations")
        if counts.ref.revision != occupied.ref.revision:
            raise ValueError("occupancy fields have different revisions")
        if counts.block.schema.cell_schema.data_axes != (
            occupied.block.schema.cell_schema.data_axes
        ):
            raise ValueError("occupancy fields do not share data axes")
        if counts.block.validity is not occupied.block.validity:
            raise ValueError("occupancy fields do not share one validity authority")
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_counts", counts)
        object.__setattr__(self, "_occupied", occupied)
        object.__setattr__(self, "_events", tuple(zip(schedule, metadata, strict=True)))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("FrozenOccupancyDataset is immutable")

    def __reduce__(self):
        raise TypeError("FrozenOccupancyDataset is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("FrozenOccupancyDataset is process-local")

    @property
    def counts(self) -> OwnedSnapshot:
        return self._counts

    @property
    def occupied(self) -> OwnedSnapshot:
        return self._occupied

    @property
    def source_metadata_in_event_order(self) -> tuple[CameraFrameMetadata, ...]:
        """Physical metadata in event order, not dataset storage order."""

        return tuple(metadata for _cell, metadata in self._events)

    @property
    def cell_schedule(self) -> tuple[DatasetCellAddress, ...]:
        """Dataset addresses in the same order as the physical events."""

        return tuple(cell for cell, _metadata in self._events)

    @property
    def events(
        self,
    ) -> tuple[tuple[DatasetCellAddress, CameraFrameMetadata], ...]:
        """Canonical event-order pairing of each cell with physical metadata."""

        return self._events

    def field(self, field: OccupancyDatasetField) -> OwnedSnapshot:
        if not isinstance(field, OccupancyDatasetField):
            raise TypeError("field must be OccupancyDatasetField")
        if field is OccupancyDatasetField.OCCUPIED:
            return self.occupied
        if field is OccupancyDatasetField.COUNTS:
            return self.counts
        raise ValueError(f"unsupported occupancy dataset field {field!r}")


_PIPELINE_RESULT_TOKEN = object()


class OccupancyPipelineResult:
    """Compacted immutable evidence plus its validated two-field dataset."""

    __slots__ = (
        "_authority",
        "_dataset",
        "_run_id",
        "_capture_terminal",
        "_source_dataset_schema",
        "_camera_provenance",
        "_camera_capability_evidence",
        "_camera_arm_spec",
        "_chain_contract_digest",
        "_aggregate_peak_bytes",
        "_terminal_provenance",
        "_model_kind",
        "_calibration_reference",
        "_calibration_artifact_fingerprint",
        "_calibration_admission_evidence_digest",
        "_processor_binding_digest",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("OccupancyPipelineResult is final")

    def __init__(
        self,
        authority: object,
        *,
        pipeline: PipelineResult,
        dataset: FrozenOccupancyDataset,
        bound: BoundOccupancyStreamProcessor,
    ) -> None:
        if authority is not _PIPELINE_RESULT_TOKEN:
            raise PermissionError(
                "OccupancyPipelineResult can only be minted by its compiler"
            )
        if not isinstance(pipeline, PipelineResult):
            raise TypeError("pipeline must be PipelineResult")
        if type(dataset) is not FrozenOccupancyDataset:
            raise TypeError("dataset must be an exact FrozenOccupancyDataset")
        if type(bound) is not BoundOccupancyStreamProcessor:
            raise TypeError("bound must be an exact BoundOccupancyStreamProcessor")
        if dataset.counts is not pipeline.dataset.snapshot:
            raise ValueError("occupancy dataset belongs to another pipeline result")
        if dataset.counts.block.schema is not bound.output_schema:
            raise ValueError("occupancy dataset belongs to another processor binding")
        if pipeline.is_direct_raw_capture:
            raise ValueError("occupancy result requires a processed pipeline")
        derivation = pipeline.dataset.provenance.derivation
        if derivation is None or len(derivation.stages) != 1:
            raise ValueError("occupancy result requires one processor derivation")
        stage = derivation.stages[0]
        if stage.processor_binding_digest != bound.processor_binding_digest:
            raise ValueError("occupancy result names another processor binding")
        if stage.direct_artifact_inputs != bound.artifact_inputs:
            raise ValueError("occupancy result names another calibration input")
        camera_provenance = pipeline.camera_provenance
        camera_capability_evidence = pipeline.camera_capability_evidence
        if camera_provenance is None or camera_capability_evidence is None:
            raise ValueError("occupancy result requires physical camera evidence")
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_run_id", pipeline.run_id)
        object.__setattr__(self, "_capture_terminal", pipeline.capture_terminal)
        object.__setattr__(
            self,
            "_source_dataset_schema",
            pipeline.source_dataset_schema,
        )
        object.__setattr__(self, "_camera_provenance", camera_provenance)
        object.__setattr__(
            self,
            "_camera_capability_evidence",
            camera_capability_evidence,
        )
        object.__setattr__(self, "_camera_arm_spec", pipeline.camera_arm_spec)
        object.__setattr__(
            self,
            "_chain_contract_digest",
            pipeline.chain_contract_digest,
        )
        object.__setattr__(
            self,
            "_aggregate_peak_bytes",
            pipeline.aggregate_peak_bytes,
        )
        object.__setattr__(
            self,
            "_terminal_provenance",
            pipeline.dataset.provenance,
        )
        object.__setattr__(self, "_model_kind", bound.model_kind)
        object.__setattr__(
            self,
            "_calibration_reference",
            bound.calibration_reference,
        )
        object.__setattr__(
            self,
            "_calibration_artifact_fingerprint",
            bound.calibration_artifact_fingerprint,
        )
        object.__setattr__(
            self,
            "_calibration_admission_evidence_digest",
            bound.calibration_admission_evidence_digest,
        )
        object.__setattr__(
            self,
            "_processor_binding_digest",
            bound.processor_binding_digest,
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("OccupancyPipelineResult is immutable")

    def __reduce__(self):
        raise TypeError("OccupancyPipelineResult is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("OccupancyPipelineResult is process-local")

    @property
    def dataset(self) -> FrozenOccupancyDataset:
        return self._dataset

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def capture_terminal(self) -> CaptureTerminalAck:
        return self._capture_terminal

    @property
    def source_dataset_schema(self) -> DatasetSchema:
        return self._source_dataset_schema

    @property
    def camera_provenance(self) -> CameraCaptureProvenance:
        return self._camera_provenance

    @property
    def camera_capability_evidence(self) -> CameraCapabilityEvidence:
        return self._camera_capability_evidence

    @property
    def camera_arm_spec(self) -> FrozenCaptureSpec:
        return self._camera_arm_spec

    @property
    def chain_contract_digest(self) -> str:
        return self._chain_contract_digest

    @property
    def aggregate_peak_bytes(self) -> int:
        return self._aggregate_peak_bytes

    @property
    def terminal_provenance(self) -> DatasetSealProvenance:
        return self._terminal_provenance

    @property
    def model_kind(self) -> ReadoutModelKind:
        return self._model_kind

    @property
    def calibration_reference(self) -> CalibrationArtifactRef:
        return self._calibration_reference

    @property
    def calibration_artifact_fingerprint(self) -> str:
        return self._calibration_artifact_fingerprint

    @property
    def calibration_admission_evidence_digest(self) -> str:
        return self._calibration_admission_evidence_digest

    @property
    def processor_binding_digest(self) -> str:
        return self._processor_binding_digest


@dataclass(frozen=True)
class _ExecutedOccupancyPipeline:
    spec: OccupancyPipelineSpec
    bound: BoundOccupancyStreamProcessor
    pipeline: PipelineResult


class _OccupancyTransaction:
    """Run-local software authority for the one fixed exact chain."""

    __slots__ = (
        "spec",
        "session",
        "bound",
        "worker",
        "memory_admission",
    )

    def __init__(
        self,
        *,
        spec: OccupancyPipelineSpec,
        session: CaptureSession,
        bound: BoundOccupancyStreamProcessor,
        worker: ExactStreamProcessorWorker,
        memory_admission: PipelineMemoryAdmission,
    ) -> None:
        self.spec = spec
        self.session = session
        self.bound = bound
        self.worker: ExactStreamProcessorWorker | None = worker
        self.memory_admission = memory_admission

    def start(self, context: RunContext) -> None:
        self.session.prepare(context)
        self.session.start(context)

    def capture_all(self, context: RunContext) -> None:
        for _cell in self.spec.measurement.capture_contract.expected_cells:
            context.checkpoint()
            self.session.capture_next(context)

    def complete(self, context: RunContext) -> _ExecutedOccupancyPipeline:
        worker = self.worker
        if worker is None:
            raise RuntimeError("occupancy worker was already released")
        completion = self.session.complete(context)
        sealed = worker.finish(completion.eos, _remaining_seconds(context))
        pipeline = finalize_pipeline_result(
            dataset=sealed,
            capture_completion=completion,
            memory_admission=self.memory_admission,
        )
        return _ExecutedOccupancyPipeline(self.spec, self.bound, pipeline)

    def fail(self, error: BaseException) -> None:
        if self.session.state is CaptureSessionState.COMPLETED:
            return
        try:
            self.session.fail(error)
        except BaseException as failure_error:
            record_secondary_failure(
                error,
                "capture poison also failed",
                failure_error,
            )

    def cleanup(self, context: RunContext) -> CleanupReport:
        software_errors: list[BaseException] = []
        worker = self.worker
        close_failed = False
        if worker is not None:
            try:
                worker.close(2.0)
            except BaseException as error:
                close_failed = True
                software_errors.append(error)

        report: CleanupReport | None = None
        port_error: BaseException | None = None
        try:
            # Physical camera stop/drain/join must run even when software graph
            # teardown failed or timed out.
            report = self.session.cleanup(context)
        except BaseException as error:
            port_error = error

        if close_failed and worker is not None:
            try:
                worker.close(2.0)
            except BaseException as error:
                software_errors.append(error)
        self.worker = None

        if port_error is not None:
            for error in software_errors:
                record_secondary_failure(
                    port_error,
                    "processor teardown also failed",
                    error,
                )
            raise port_error
        assert report is not None
        if not software_errors:
            return report
        return CleanupReport(
            safety_proofs=report.safety_proofs,
            decisions=report.decisions,
            errors=(*report.errors, *software_errors),
        )


def _remaining_seconds(context: RunContext) -> float:
    context.checkpoint()
    if context.deadline is None:
        raise RuntimeError("occupancy pipeline requires a finite Run deadline")
    remaining = float(context.deadline) - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError("occupancy pipeline deadline expired")
    return remaining


def _occupied_dataset_schema(
    bound: BoundOccupancyStreamProcessor,
) -> DatasetSchema:
    counts_schema = bound.output_schema
    return DatasetSchema(
        counts_schema.repeat_axis,
        counts_schema.point_axes,
        counts_schema.point_layout,
        bound.output_payload_contract.occupied_schema,
    )


def _estimate_occupancy_pipeline_peak_bytes(
    spec: OccupancyPipelineSpec,
    bound: BoundOccupancyStreamProcessor,
) -> int:
    """Phase-aware conservative peak for this fixed serial pipeline."""

    contract = spec.measurement.capture_contract
    events = contract.total_events
    counts_bytes = dataset_storage_nbytes(bound.output_schema)
    occupied_bytes = dataset_storage_nbytes(_occupied_dataset_schema(bound))
    metadata_bytes = events * bound.output_edge.metadata_max_retained_nbytes
    output_payload_bytes = bound.output_payload_contract.max_retained_nbytes
    calibration_bytes = calibration_retained_array_nbytes(
        spec.processor.calibration.artifact
    )
    memory = spec.materializer.memory
    runtime_bytes = (
        memory.fixed_runtime_overhead_bytes
        + events * memory.per_event_reference_overhead_bytes
    )
    execution_peak = (
        contract.estimated_transport_bytes
        + output_payload_bytes
        + 2 * counts_bytes
        + metadata_bytes
        + bound.operator_scratch_nbytes
        + calibration_bytes
        + runtime_bytes
    )
    post_safety_finalization_peak = (
        contract.estimated_transport_bytes
        + output_payload_bytes
        + 2 * counts_bytes
        + metadata_bytes
        + 2 * occupied_bytes
        + bound.output_payload_contract.finalization_scratch_nbytes
        + calibration_bytes
        + runtime_bytes
    )
    return max(execution_peak, post_safety_finalization_peak)


def _release_reservation(reservation: ExactReservation | None) -> None:
    if reservation is None or reservation.state is ReservationState.RELEASED:
        return
    if reservation.state not in (
        ReservationState.COMPLETED,
        ReservationState.FAILED,
        ReservationState.CANCELLED,
    ):
        reservation.abort()
    if reservation.state is not ReservationState.RELEASED:
        reservation.release()


def _teardown_failed_preflight(
    *,
    session: CaptureSession,
    worker: ExactStreamProcessorWorker | None,
    builder: DatasetBuilder | None,
    source_reservation: ExactReservation | None,
    output_reservation: ExactReservation | None,
    primary: BaseException,
) -> None:
    errors: list[BaseException] = []
    try:
        session.fail(primary)
    except BaseException as error:
        errors.append(error)
    if worker is not None:
        try:
            worker.close(2.0)
        except BaseException as error:
            errors.append(error)
    elif builder is not None:
        try:
            builder.close()
        except BaseException as error:
            errors.append(error)
    else:
        try:
            _release_reservation(output_reservation)
        except BaseException as error:
            errors.append(error)
    if worker is None:
        try:
            _release_reservation(source_reservation)
        except BaseException as error:
            errors.append(error)
    for error in errors:
        record_secondary_failure(
            primary,
            "preflight teardown also failed",
            error,
        )


def _open_occupancy_transaction(
    spec: OccupancyPipelineSpec,
    context: RunContext,
) -> _OccupancyTransaction:
    measurement = spec.measurement
    contract = measurement.capture_contract
    session = measurement.capture_port.open_session(
        contract,
        TraceBinding(context.run_id.value, contract.source_id),
        measurement.capture_spec,
    )
    worker: ExactStreamProcessorWorker | None = None
    builder: DatasetBuilder | None = None
    source_reservation: ExactReservation | None = None
    output_reservation: ExactReservation | None = None
    try:
        capture_input = session.processor_input_binding
        bound = bind_occupancy_stream_processor(spec.processor, capture_input)
        aggregate_peak = _estimate_occupancy_pipeline_peak_bytes(spec, bound)
        if aggregate_peak > spec.materializer.memory.memory_limit_bytes:
            raise MemoryError(
                f"occupancy pipeline peak budget {aggregate_peak} exceeds "
                f"limit {spec.materializer.memory.memory_limit_bytes}"
            )

        source_reservation = session.reserve_exact()
        source_cursor = source_reservation.activate()
        output_contract = bound.output_payload_contract
        output_retention_bytes = output_contract.max_retained_nbytes
        output_stream, output_producer = AcquisitionStream.create(
            bound.output_stream_id,
            output_contract,
            flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
            retention_events=1,
            retention_bytes=output_retention_bytes,
            join_key_contract=capture_input.join_key_contract,
        )
        output_reservation = output_stream.reserve(
            total_events=contract.total_events,
            max_inflight_events=1,
            max_inflight_bytes=output_retention_bytes,
            trace_binding=TraceBinding(
                context.run_id.value,
                bound.output_source_id,
            ),
        )
        output_cursor = output_reservation.activate()
        builder = DatasetBuilder(
            spec.materializer.counts_block_id,
            output_reservation,
            bound.output_edge,
            DatasetMode.FINITE_EXACT,
        )
        _remaining_seconds(context)
        assert context.deadline is not None
        deadline = float(context.deadline)
        worker = bound.create_exact_worker(
            source_reservation,
            source_cursor,
            output_producer=output_producer,
            output_cursor=output_cursor,
            output_builder=builder,
            deadline_monotonic=deadline,
            cancellation=context.cancellation,
        )
        worker.start()
        readiness = worker.exact_readiness()
        memory_admission = admit_pipeline_memory(
            aggregate_peak_bytes=aggregate_peak,
            memory_profile=spec.materializer.memory,
            chain_contract_digest=readiness.chain_contract_digest,
        )
        session.bind_exact_consumer(readiness)
        return _OccupancyTransaction(
            spec=spec,
            session=session,
            bound=bound,
            worker=worker,
            memory_admission=memory_admission,
        )
    except BaseException as error:
        _teardown_failed_preflight(
            session=session,
            worker=worker,
            builder=builder,
            source_reservation=source_reservation,
            output_reservation=output_reservation,
            primary=error,
        )
        raise


def _validate_terminal_identity(
    executed: _ExecutedOccupancyPipeline,
) -> tuple[
    SealedDatasetArtifact,
    tuple[DatasetCellAddress, ...],
]:
    spec = executed.spec
    bound = executed.bound
    pipeline = executed.pipeline
    terminal = pipeline.dataset
    contract = spec.measurement.capture_contract
    if pipeline.is_direct_raw_capture:
        raise RuntimeError("occupancy result cannot be a direct raw capture")
    if pipeline.source_dataset_schema is not contract.dataset_schema:
        raise RuntimeError("occupancy result names another source DatasetSchema")
    if pipeline.source_cell_schedule != contract.expected_cells:
        raise RuntimeError("occupancy result names another source cell schedule")
    expected_cells = bound.output_edge.expected_cells
    if expected_cells is None or expected_cells != contract.expected_cells:
        raise RuntimeError("occupancy output schedule differs from physical capture")
    provenance = terminal.provenance
    if provenance.stream_id != bound.output_stream_id:
        raise RuntimeError("terminal occupancy stream differs from processor binding")
    if terminal.block.block_id != spec.materializer.counts_block_id:
        raise RuntimeError("terminal counts block has another BlockId")
    if terminal.block.schema is not bound.output_schema:
        raise RuntimeError("terminal counts schema differs from occupancy binding")
    expected_count = contract.total_events
    if provenance.start_sequence < 0:
        raise RuntimeError("terminal occupancy sequence interval starts below zero")
    interval_count = provenance.end_sequence - provenance.start_sequence
    physical_cell_count = (
        contract.dataset_schema.repeat_axis.size
        * contract.dataset_schema.point_layout.storage_size
    )
    if not (
        interval_count
        == len(expected_cells)
        == terminal.coverage.total_cells
        == physical_cell_count
        == expected_count
    ):
        raise RuntimeError("terminal occupancy interval/coverage/domain sizes differ")
    if (
        not terminal.coverage.complete
        or terminal.coverage.written_cells != expected_count
        or terminal.coverage.missed_events
    ):
        raise RuntimeError("terminal occupancy coverage is incomplete")
    if terminal.ref.stream_generation != provenance.generation:
        raise RuntimeError("terminal occupancy ref generation differs from provenance")
    if terminal.ref != terminal.block.ref(provenance.generation):
        raise RuntimeError("terminal occupancy ref differs from its block/provenance")
    if terminal.block.revision.value != expected_count:
        raise RuntimeError("terminal occupancy revision differs from event count")
    validity = terminal.block.validity
    expected_validity_axes = (
        bound.output_schema.cell_schema.validity_contract.component_axis_ids
    )
    if (
        not isinstance(validity, ComponentValidity)
        or validity.axis_ids != expected_validity_axes
    ):
        raise RuntimeError("terminal occupancy validity axes differ from contract")
    if provenance.trace_binding.run_id != pipeline.run_id:
        raise RuntimeError("terminal occupancy trace belongs to another Run")
    if provenance.trace_binding.source_id != bound.output_source_id:
        raise RuntimeError("terminal occupancy trace names another processor source")
    if provenance.join_plan_digest != bound.output_edge.schedule_digest:
        raise RuntimeError("terminal occupancy join plan differs from binding")
    if (
        provenance.metadata_contract_fingerprint
        != bound.output_edge.metadata_contract_fingerprint
    ):
        raise RuntimeError("terminal occupancy metadata contract differs")
    derivation = provenance.derivation
    if derivation is None or len(derivation.stages) != 1:
        raise RuntimeError("occupancy terminal requires exactly one processor stage")
    stage = derivation.stages[0]
    if derivation.chain_contract_digest != pipeline.chain_contract_digest:
        raise RuntimeError("occupancy derivation chain digest differs from pipeline")
    if stage.processor_binding_digest != bound.processor_binding_digest:
        raise RuntimeError("occupancy derivation names another processor binding")
    if stage.direct_artifact_inputs != bound.artifact_inputs:
        raise RuntimeError("occupancy derivation names another calibration input")
    if len(terminal.event_metadata) != len(expected_cells):
        raise RuntimeError("occupancy terminal metadata length differs from schedule")
    return terminal, tuple(expected_cells)


def _finalize_occupancy_result(
    context: PostSafetyContext,
    executed: _ExecutedOccupancyPipeline,
) -> OccupancyPipelineResult:
    terminal, expected_cells = _validate_terminal_identity(executed)
    bound = executed.bound
    pipeline = executed.pipeline
    counts = terminal.snapshot
    counts_block = counts.block
    if not isinstance(counts_block.validity, ComponentValidity):
        raise TypeError("occupancy counts block requires ComponentValidity")
    occupied_schema = _occupied_dataset_schema(bound)
    occupied_values = np.zeros(occupied_schema.physical_shape, dtype=bool)
    metadata_contract = bound.output_edge.metadata_contract
    source_edge = executed.spec.measurement.capture_contract.dataset_edge
    source_contract = source_edge.metadata_contract
    source_metadata_hasher = OrderedDatasetMetadataHasher(
        source_edge.metadata_contract_fingerprint
    )
    provenance = terminal.provenance
    source_metadata: list[CameraFrameMetadata] = []

    for ordinal, (cell, metadata) in enumerate(
        zip(expected_cells, terminal.event_metadata, strict=True)
    ):
        context.checkpoint()
        if not isinstance(metadata, OccupancyDatasetMetadata):
            raise TypeError("occupancy terminal contains another metadata type")
        metadata_contract.validate(metadata)
        if metadata.source_metadata.source_ordinal != ordinal:
            raise RuntimeError("camera source ordinal differs from occupancy event order")
        location = (cell.repeat_index, cell.point_storage_index)
        occupied = metadata.occupied
        counts_values = counts_block.values[location]
        counts_validity = ComponentValidity(
            counts_block.validity.axis_ids,
            counts_block.validity.mask[location],
        )
        if (
            counts_validity.axis_ids != occupied.validity.axis_ids
            or not np.array_equal(counts_validity.mask, occupied.validity.mask)
        ):
            raise RuntimeError("counts and occupied validity differ at one dataset cell")
        source_contract.validate(metadata.source_metadata)
        source_metadata_hasher.update(source_contract.digest(metadata.source_metadata))
        occupied_values[location] = occupied.values
        source_metadata.append(metadata.source_metadata)

    context.checkpoint()
    if (
        source_metadata_hasher.digest()
        != pipeline.capture_terminal.ordered_metadata_digest
    ):
        raise RuntimeError("occupancy source metadata differs from physical capture")
    occupied_block = DataBlock(
        executed.spec.materializer.occupied_block_id,
        counts_block.revision,
        occupied_values,
        counts_block.validity,
        occupied_schema,
    )
    occupied_snapshot = OwnedSnapshot(
        occupied_block.ref(provenance.generation),
        occupied_block,
    )
    frozen = FrozenOccupancyDataset(
        _FROZEN_DATASET_TOKEN,
        counts=counts,
        occupied=occupied_snapshot,
        source_metadata=tuple(source_metadata),
        cell_schedule=expected_cells,
    )
    context.checkpoint()
    return OccupancyPipelineResult(
        _PIPELINE_RESULT_TOKEN,
        pipeline=pipeline,
        dataset=frozen,
        bound=bound,
    )


def compile_occupancy_pipeline(spec: OccupancyPipelineSpec) -> RunPlan:
    """Compile the only supported camera -> occupancy exact vertical slice."""

    if not isinstance(spec, OccupancyPipelineSpec):
        raise TypeError("spec must be OccupancyPipelineSpec")
    port = spec.measurement.capture_port

    def preflight(context: RunContext) -> _OccupancyTransaction:
        return _open_occupancy_transaction(spec, context)

    def execute(
        context: RunContext,
        prepared: _OccupancyTransaction,
    ) -> _ExecutedOccupancyPipeline:
        try:
            prepared.start(context)
            prepared.capture_all(context)
            return prepared.complete(context)
        except BaseException as error:
            prepared.fail(error)
            raise

    def cleanup(
        context: RunContext,
        prepared: _OccupancyTransaction | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return port.verify_idle(context)
        return prepared.cleanup(context)

    def finalize(
        context: PostSafetyContext,
        executed: _ExecutedOccupancyPipeline,
    ) -> OccupancyPipelineResult:
        if not isinstance(executed, _ExecutedOccupancyPipeline):
            raise TypeError("occupancy finalize requires its executed result")
        if executed.pipeline.run_id != context.run_id.value:
            raise ValueError("executed occupancy result belongs to another Run")
        context.checkpoint()
        return _finalize_occupancy_result(context, executed)

    return RunPlan(
        name=spec.name,
        mode=RunMode.FINITE_EXACT,
        resource_claims=(port.resource_claim,),
        hazard_claims=(port.hazard_claim,),
        bound_devices=(port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=port.interrupt_operations,
        timeout_seconds=spec.timeout_seconds,
        requires_final_commit=False,
    )


__all__ = [
    "compile_occupancy_pipeline",
    "FrozenOccupancyDataset",
    "OccupancyDatasetMaterializerSpec",
    "OccupancyPipelineResult",
    "OccupancyPipelineSpec",
]
