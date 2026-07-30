"""Exact two-frame release-recapture reduction in one flat capture pipeline.

The source camera capture publishes every physical frame.  This module owns
the one earned K:1 operation in the current product: adjacent, explicitly keyed
READOUT_EVENT 0/1 frames become one survival sample.  It is deliberately not a
generic workflow or configurable grouping framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from zlc_data import (
    INVALID,
    READOUT_EVENT,
    VALID,
    BlockId,
    ComponentValidity,
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ReadoutModel,
    ResolvedCalibration,
    apply_readout_model,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.runtime._failure import (
    record_secondary_failure,
    safe_error_summary,
)
from zlc_neutral_atom.capture.session import (
    CaptureCompletion,
    CaptureSession,
    CaptureSessionState,
    open_capture_session,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.dataset import (
    DatasetBuilder,
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetCellSchedule,
    FrozenDatasetEdge,
)
from zlc_neutral_atom.capture.pipeline import (
    PipelineResult,
    finalize_pipeline_result,
)
from zlc_neutral_atom.capture.binding import TriggeredCameraBinding
from zlc_neutral_atom.devices.rf import BoundRfTablePort, RfDetuningTable
from zlc_neutral_atom.runtime.run import RunContext
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    AcquisitionProducer,
    AcquisitionStream,
    Delivery,
    EndOfStream,
    Envelope,
    EventSpanRef,
    ExactConsumerReadiness,
    ExactReservation,
    ReservationState,
    SourceFailed,
    StreamEndedEarly,
    StreamId,
)
from zlc_storage import canonical_text


_CAMERA_FRAME_METADATA_CONTRACT = CameraFrameMetadataContract()


@dataclass(frozen=True, slots=True)
class ReleaseRecapturePairMetadata:
    """The two camera receipts that produced one survival sample."""

    initial: CameraFrameMetadata
    recaptured: CameraFrameMetadata

    def __post_init__(self) -> None:
        _CAMERA_FRAME_METADATA_CONTRACT.validate(self.initial)
        _CAMERA_FRAME_METADATA_CONTRACT.validate(self.recaptured)


@dataclass(frozen=True, slots=True)
class ReleaseRecaptureSample:
    survival: Value
    metadata: ReleaseRecapturePairMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.survival, Value):
            raise TypeError("survival must be zlc_data.Value")
        if not isinstance(self.metadata, ReleaseRecapturePairMetadata):
            raise TypeError("metadata must be ReleaseRecapturePairMetadata")


@dataclass(frozen=True, slots=True)
class ReleaseRecaptureMetadataContract:
    """Stateless metadata contract shared by every release-recapture sample."""

    @property
    def source(self) -> CameraFrameMetadataContract:
        return _CAMERA_FRAME_METADATA_CONTRACT

    def snapshot(
        self,
        payload: ReleaseRecaptureSample,
    ) -> ReleaseRecapturePairMetadata:
        if not isinstance(payload, ReleaseRecaptureSample):
            raise TypeError("metadata snapshot requires ReleaseRecaptureSample")
        self.validate(payload.metadata)
        return payload.metadata

    def validate(self, metadata: object | None) -> None:
        if not isinstance(metadata, ReleaseRecapturePairMetadata):
            raise TypeError("metadata must be ReleaseRecapturePairMetadata")
        self.source.validate(metadata.initial)
        self.source.validate(metadata.recaptured)

_RELEASE_RECAPTURE_METADATA_CONTRACT = ReleaseRecaptureMetadataContract()


@dataclass(frozen=True, slots=True)
class ReleaseRecaptureSampleContract:
    value_schema: ValueSchema
    value_contract: ValuePayloadContract = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value_schema, ValueSchema):
            raise TypeError("value_schema must be ValueSchema")
        object.__setattr__(
            self,
            "value_contract",
            ValuePayloadContract(self.value_schema),
        )

    @property
    def metadata_contract(self) -> ReleaseRecaptureMetadataContract:
        return _RELEASE_RECAPTURE_METADATA_CONTRACT

    def snapshot(self, payload: ReleaseRecaptureSample) -> ReleaseRecaptureSample:
        self.validate(payload)
        return payload

    def validate(self, payload: ReleaseRecaptureSample) -> None:
        if not isinstance(payload, ReleaseRecaptureSample):
            raise TypeError("payload must be ReleaseRecaptureSample")
        self.value_contract.validate(payload.survival)
        self.metadata_contract.validate(payload.metadata)


@dataclass(frozen=True, slots=True)
class ReleaseRecaptureDatasetEventAdapter:
    """Freeze the stream-payload to dataset-cell projection."""

    payload_contract: ReleaseRecaptureSampleContract

    def __post_init__(self) -> None:
        if not isinstance(self.payload_contract, ReleaseRecaptureSampleContract):
            raise TypeError(
                "payload_contract must be ReleaseRecaptureSampleContract"
            )

    @property
    def value_schema(self) -> ValueSchema:
        return self.payload_contract.value_schema

    @property
    def metadata_contract(self) -> ReleaseRecaptureMetadataContract:
        return self.payload_contract.metadata_contract

    @staticmethod
    def value(payload: ReleaseRecaptureSample) -> Value:
        return payload.survival


@dataclass(frozen=True, slots=True)
class ReleaseRecapturePipelineSpec:
    name: str
    camera_binding: TriggeredCameraBinding
    calibration: ResolvedCalibration
    model: ReadoutModel
    per_site: bool
    output_stream_id: StreamId
    output_source_id: str
    block_id: BlockId
    rf_port: BoundRfTablePort | None = None
    rf_table: RfDetuningTable | None = None

    def __post_init__(self) -> None:
        canonical_text(self.name, "name")
        if not isinstance(self.camera_binding, TriggeredCameraBinding):
            raise TypeError("camera_binding must be TriggeredCameraBinding")
        if not isinstance(self.calibration, ResolvedCalibration):
            raise TypeError("calibration must be a loaded ResolvedCalibration")
        if not isinstance(self.model, ReadoutModel):
            raise TypeError("model must be ReadoutModel")
        if not any(
            model is self.model
            for model in self.calibration.artifact.models
        ):
            raise ValueError("model must belong to the loaded calibration")
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")
        if not isinstance(self.output_stream_id, StreamId):
            raise TypeError("output_stream_id must be StreamId")
        canonical_text(self.output_source_id, "output_source_id")
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if (self.rf_port is None) != (self.rf_table is None):
            raise ValueError("RF Port and table must be supplied together")
        if self.rf_port is not None:
            if not isinstance(self.rf_port, BoundRfTablePort):
                raise TypeError("rf_port must be BoundRfTablePort")
            if not isinstance(self.rf_table, RfDetuningTable):
                raise TypeError("rf_table must be RfDetuningTable")
            if (
                self.rf_table.pulse_artifact_digest
                != self.camera_binding.pulse_request.artifact_digest
            ):
                raise ValueError("RF table belongs to another pulse artifact")


@dataclass(frozen=True, slots=True)
class _PairPlan:
    first_key: DatasetCellAddress
    second_key: DatasetCellAddress
    output_key: DatasetCellAddress


def _readout_event_column(schema: DatasetSchema) -> PointColumn:
    event_columns = tuple(
        column
        for column in schema.point_table.columns
        if column.role == READOUT_EVENT
    )
    if len(event_columns) != 1:
        raise ValueError("release-recapture source has no unique READOUT_EVENT column")
    event_column = event_columns[0]
    expected = tuple(
        event
        for _point in range(schema.point_table.row_count // 2)
        for event in (0, 1)
    )
    if event_column.values != expected:
        raise ValueError(
            "release-recapture rows must be adjacent event-0/event-1 pairs"
        )
    return event_column


def _remove_readout_event(
    source_schema: DatasetSchema,
    value_schema: ValueSchema,
    event_column: PointColumn,
) -> DatasetSchema:
    selected_ordinals = tuple(range(0, source_schema.point_table.row_count, 2))
    point_table = PointTable(
        len(selected_ordinals),
        tuple(
            PointColumn(
                column.coordinate_id,
                column.name,
                column.role,
                column.value_kind,
                tuple(column.values[ordinal] for ordinal in selected_ordinals),
                column.unit,
                column.coordinate_frame,
            )
            for column in source_schema.point_table.columns
            if column is not event_column
        ),
    )
    source_topology = source_schema.grid_topology
    topology = None
    if source_topology is not None:
        positions = tuple(
            index
            for index, dimension_id in enumerate(source_topology.dimension_ids)
            if dimension_id == event_column.coordinate_id
        )
        if len(positions) != 1:
            raise ValueError(
                "release-recapture GridTopology must name its READOUT_EVENT dimension"
            )
        event_position = positions[0]
        if source_topology.coordinate_domains[event_position] != (0, 1):
            raise ValueError(
                "release-recapture GridTopology event domain must be (0, 1)"
            )
        dimension_ids = tuple(
            dimension_id
            for position, dimension_id in enumerate(source_topology.dimension_ids)
            if position != event_position
        )
        domains = tuple(
            domain
            for position, domain in enumerate(source_topology.coordinate_domains)
            if position != event_position
        )
        cells = tuple(
            tuple(
                index
                for position, index in enumerate(
                    source_topology.row_to_cell[ordinal]
                )
                if position != event_position
            )
            for ordinal in selected_ordinals
        )
        if dimension_ids:
            topology = GridTopology(dimension_ids, domains, cells)
    return DatasetSchema(
        source_schema.repeat_axis,
        point_table,
        topology,
        value_schema,
    )


def _pair_plan(
    source_schema: DatasetSchema,
    source_schedule: DatasetCellSchedule,
    output_schema: DatasetSchema,
    event_column: PointColumn,
) -> tuple[_PairPlan, ...]:
    if len(source_schedule) != (
        output_schema.repeat_axis.size
        * output_schema.point_table.row_count
        * 2
    ):
        raise ValueError("release-recapture source cardinality is not exactly 2:1")
    plans: list[_PairPlan] = []
    source_cells = tuple(source_schedule)
    for offset in range(0, len(source_cells), 2):
        first, second = source_cells[offset : offset + 2]
        if first.repeat_index != second.repeat_index:
            raise ValueError("release-recapture pair crosses repeat cells")
        first_ordinal = first.point_ordinal
        second_ordinal = second.point_ordinal
        output_ordinal = first_ordinal // 2
        if (
            first_ordinal != output_ordinal * 2
            or second_ordinal != first_ordinal + 1
            or event_column.values[first_ordinal] != 0
            or event_column.values[second_ordinal] != 1
            or any(
                column.values[first_ordinal] != column.values[second_ordinal]
                for column in source_schema.point_table.columns
                if column is not event_column
            )
        ):
            raise ValueError(
                "release-recapture source must be adjacent event-0/event-1 pairs"
            )
        plans.append(
            _PairPlan(
                first,
                second,
                DatasetCellAddress(
                    first.repeat_index,
                    output_ordinal,
                ),
            )
        )
    return tuple(plans)


def _output_contract(
    spec: ReleaseRecapturePipelineSpec,
) -> tuple[
    DatasetSchema,
    ReleaseRecaptureSampleContract,
    tuple[_PairPlan, ...],
    DatasetCellSchedule,
]:
    source = spec.camera_binding.capture.capture_contract
    source_schema = source.dataset_schema
    event_column = _readout_event_column(source_schema)
    site_axis = spec.model.feature.site_axis
    value_schema = (
        ValueSchema(
            (site_axis,),
            ValidityContract.components(site_axis.axis_id),
            np.dtype("<f8"),
            "survival",
        )
        if spec.per_site
        else ValueSchema.scalar(np.dtype("<f8"), "survival")
    )
    output_schema = _remove_readout_event(
        source_schema,
        value_schema,
        event_column,
    )
    plans = _pair_plan(
        source_schema,
        source.cell_schedule,
        output_schema,
        event_column,
    )
    schedule = DatasetCellSchedule.from_cells(
        output_schema,
        (plan.output_key for plan in plans),
    )
    contract = ReleaseRecaptureSampleContract(value_schema)
    return output_schema, contract, plans, schedule


@dataclass(slots=True)
class ExactReleaseRecaptureTransaction:
    """Run-local owner of one exact fixed-size two-frame reduction."""

    spec: ReleaseRecapturePipelineSpec
    session: CaptureSession
    source_reservation: ExactReservation
    source_cursor: AcquisitionCursor
    output_producer: AcquisitionProducer
    output_cursor: AcquisitionCursor
    output_builder: DatasetBuilder
    pair_plan: tuple[_PairPlan, ...]
    readiness: ExactConsumerReadiness | None
    next_input_sequence: int
    _next_source_index: int = 0
    _first: Envelope[CameraSample] | None = None
    _result: PipelineResult | None = None
    _error: BaseException | None = None
    _cancelled: bool = False
    _done: bool = False

    def start(self, context: RunContext) -> None:
        self.session.prepare(context)
        self.session.start(context)

    def capture_all(self, context: RunContext) -> None:
        for _ordinal in range(
            self.spec.camera_binding.capture.capture_contract.total_events
        ):
            context.checkpoint()
            self.session.capture_next(context)
            self._consume_one(context)

    def _consume_one(self, context: RunContext) -> None:
        if self._done or self._cancelled:
            raise RuntimeError("release-recapture reducer is no longer live")
        call_bound = (
            self.spec.camera_binding.capture.capture_port.capability
            .max_blocking_call_seconds
        )
        delivery = self.source_cursor.next(timeout=call_bound)
        self.source_reservation.validate_delivery(
            delivery,
            self,
        )
        index = self._next_source_index
        group = self.pair_plan[index // 2]
        expected = group.first_key if index % 2 == 0 else group.second_key
        envelope = delivery.envelope
        if envelope.join_key != expected:
            raise ValueError(
                "release-recapture input differs from the frozen pair schedule"
            )
        if not isinstance(envelope.payload, CameraSample):
            raise TypeError("release-recapture input must be CameraSample")
        if envelope.sequence != self.next_input_sequence:
            raise RuntimeError("release-recapture input sequence is not contiguous")
        self.next_input_sequence += 1
        if index % 2 == 0:
            self._first = envelope
            self.source_reservation.acknowledge_delivery(
                delivery,
                self,
            )
        else:
            first = self._first
            if first is None:
                raise RuntimeError("release-recapture event 1 has no event 0")
            sample = self._reduce_pair(first, envelope)
            emitted = self.output_producer.emit(
                sample,
                captured_at=max(first.captured_at, envelope.captured_at),
                direct_parent_refs=(
                    first.ref,
                    envelope.ref,
                ),
                join_key=group.output_key,
            )
            output = self.output_cursor.next(timeout=call_bound)
            if output.envelope.ref != emitted.ref:
                raise RuntimeError(
                    "release-recapture output cursor lost the emitted sample"
                )
            self.output_builder.consume(output)
            self.source_reservation.acknowledge_delivery(
                delivery,
                self,
            )
            self._first = None
        self._next_source_index += 1
        context.checkpoint()

    def _reduce_pair(
        self,
        first: Envelope[CameraSample],
        second: Envelope[CameraSample],
    ) -> ReleaseRecaptureSample:
        model = self.spec.model
        frame_schema = self.spec.calibration.artifact.frame_contract.frame_schema
        initial = apply_readout_model(
            model,
            first.payload.image,
            expected_frame_schema=frame_schema,
        )
        recaptured = apply_readout_model(
            model,
            second.payload.image,
            expected_frame_schema=frame_schema,
        )
        left = initial.occupied.validity
        right = recaptured.occupied.validity
        if not isinstance(left, ComponentValidity) or not isinstance(
            right,
            ComponentValidity,
        ):
            raise TypeError("release-recapture classification requires site validity")
        if left.axis_ids != right.axis_ids:
            raise ValueError("release-recapture pair names different validity axes")
        pair_valid = np.asarray(left.mask) & np.asarray(right.mask)
        initially_occupied = (
            np.asarray(initial.occupied.values, dtype=bool) & pair_valid
        )
        survived = (
            initially_occupied
            & np.asarray(recaptured.occupied.values, dtype=bool)
        )
        schema = self.output_builder.schema.cell_schema
        if self.spec.per_site:
            values = survived.astype("<f8")
            validity = ComponentValidity(left.axis_ids, initially_occupied)
        else:
            denominator = int(np.count_nonzero(initially_occupied))
            if denominator:
                values = np.asarray(
                    [np.count_nonzero(survived) / denominator],
                    dtype="<f8",
                )
                validity = VALID
            else:
                values = np.asarray([0.0], dtype="<f8")
                validity = INVALID
        return ReleaseRecaptureSample(
            Value(values, validity, schema),
            ReleaseRecapturePairMetadata(
                first.payload.metadata,
                second.payload.metadata,
            ),
        )

    def complete(self, context: RunContext) -> PipelineResult:
        completion: CaptureCompletion = self.session.complete(context)
        if self._first is not None or self._next_source_index != (
            self.spec.camera_binding.capture.capture_contract.total_events
        ):
            raise RuntimeError("release-recapture reducer ended with an incomplete pair")
        self.source_reservation.validate_completion(
            completion.eos,
            self,
        )
        output_eos = self.output_producer.finish()
        sealed = self.output_builder.seal(output_eos)
        readiness = self.readiness
        if readiness is None:
            raise RuntimeError(
                "release-recapture exact consumer was never bound"
            )
        sealed = sealed._with_direct_parent_span(
            readiness,
            EventSpanRef(
                self.source_reservation.stream_id,
                self.source_reservation.stream_generation,
                self.source_reservation.start_sequence,
                self.source_reservation.end_sequence,
            ),
        )
        self.source_reservation.complete_consumer(
            completion.eos,
            self,
        )
        pipeline = finalize_pipeline_result(
            dataset=sealed,
            capture_completion=completion,
        )
        self._result = pipeline
        self._done = True
        return pipeline

    def fail(self, error: BaseException) -> None:
        self._error = error
        try:
            self.output_producer.fail(
                SourceFailed(
                    "release-recapture reducer failed: "
                    + safe_error_summary(error)
                )
            )
        except StreamEndedEarly:
            pass
        except BaseException as secondary:
            record_secondary_failure(
                error,
                "release-recapture output poison also failed",
                secondary,
            )
        if self.source_reservation.state in (
            ReservationState.ACTIVE,
            ReservationState.DRAINING,
        ):
            try:
                self.source_reservation.abort_consumer(
                    self,
                    cancelled=context_cancelled(error),
                )
            except BaseException as secondary:
                record_secondary_failure(
                    error,
                    "release-recapture input abort also failed",
                    secondary,
                )
        if self.session.state is not CaptureSessionState.COMPLETED:
            try:
                self.session.fail(error)
            except BaseException as secondary:
                record_secondary_failure(
                    error,
                    "release-recapture capture poison also failed",
                    secondary,
                )

    def cleanup(self, context: RunContext) -> CleanupReport:
        errors: list[BaseException] = []
        try:
            self.output_builder.close()
        except BaseException as error:
            errors.append(error)
        try:
            report = self.session.cleanup(context)
        except BaseException as primary:
            for error in errors:
                record_secondary_failure(
                    primary,
                    "release-recapture reducer cleanup also failed",
                    error,
                )
            raise
        if errors:
            return CleanupReport.complete(errors=(*report.errors, *errors))
        return report

    def _validate_liveness(self) -> None:
        if self._done or self._error is not None or self._cancelled:
            raise RuntimeError("release-recapture reducer is not live")

    def _await_completion(self, deadline_monotonic: float):
        if time.monotonic() >= float(deadline_monotonic):
            raise TimeoutError("release-recapture reducer completion expired")
        if self._error is not None:
            raise RuntimeError("release-recapture reducer failed") from self._error
        if self._result is None:
            raise RuntimeError("release-recapture reducer is not complete")
        return self._result.dataset

    def cancel(self, _reason: str | None = None) -> bool:
        changed = not self._cancelled
        self._cancelled = True
        return changed


def context_cancelled(error: BaseException) -> bool:
    from zlc_neutral_atom.runtime.cancellation import CancellationRequested

    return isinstance(error, CancellationRequested)


def open_exact_release_recapture(
    spec: ReleaseRecapturePipelineSpec,
    context: RunContext,
) -> ExactReleaseRecaptureTransaction:
    """Allocate and bind the complete exact chain without touching hardware."""

    if not isinstance(spec, ReleaseRecapturePipelineSpec):
        raise TypeError("spec must be ReleaseRecapturePipelineSpec")
    camera_capture = spec.camera_binding.capture
    contract = camera_capture.capture_contract
    session = open_capture_session(
        camera_capture.capture_port,
        contract,
        camera_capture.capture_spec,
    )
    source_reservation = output_reservation = None
    output_builder = None
    reducer = None
    try:
        source_reservation = session.reserve_exact()
        source_cursor = source_reservation.activate()
        input_edge = contract.dataset_edge
        output_schema, payload_contract, plans, output_schedule = (
            _output_contract(spec)
        )
        output_key_contract = DatasetCellKeyContract.from_schema(output_schema)
        output_stream, producer = AcquisitionStream.create(
            spec.output_stream_id,
            payload_contract,
            join_key_contract=output_key_contract,
        )
        output_reservation = output_stream.reserve(
            total_events=len(plans),
        )
        output_cursor = output_reservation.activate()
        output_edge = FrozenDatasetEdge(
            output_schema,
            ReleaseRecaptureDatasetEventAdapter(payload_contract),
            output_schedule,
        )
        output_builder = DatasetBuilder(
            spec.block_id,
            output_reservation,
            output_edge,
        )
        downstream = output_builder.exact_readiness()
        downstream._validate_emitter(
            stream=output_stream,
            total_events=len(plans),
        )
        # Initialize the callback owner before the final source-consumer claim.
        reducer = ExactReleaseRecaptureTransaction(
            spec,
            session,
            source_reservation,
            source_cursor,
            producer,
            output_cursor,
            output_builder,
            plans,
            readiness=None,
            next_input_sequence=source_reservation.start_sequence,
        )
        readiness = source_reservation.bind_consumer(
            reducer,
            downstream=downstream,
            owner_liveness=reducer._validate_liveness,
            owner_completion=reducer._await_completion,
            owner_cancel=reducer.cancel,
        )
        reducer.readiness = readiness
        session.bind_exact_consumer(readiness)
        return reducer
    except BaseException as primary:
        if reducer is not None and source_reservation is not None and (
            source_reservation.consumer_bound
        ):
            try:
                reducer.fail(primary)
            except BaseException as secondary:
                record_secondary_failure(
                    primary,
                    "release-recapture preflight abort also failed",
                    secondary,
                )
        if output_builder is not None:
            try:
                output_builder.close()
            except BaseException as secondary:
                record_secondary_failure(
                    primary,
                    "release-recapture output release also failed",
                    secondary,
                )
        elif output_reservation is not None:
            try:
                if output_reservation.state not in (
                    ReservationState.COMPLETED,
                    ReservationState.RELEASED,
                ):
                    output_reservation.abort()
                if output_reservation.state is not ReservationState.RELEASED:
                    output_reservation.release()
            except BaseException as secondary:
                record_secondary_failure(
                    primary,
                    "release-recapture output reservation release also failed",
                    secondary,
                )
        if source_reservation is not None and not source_reservation.consumer_bound:
            try:
                source_reservation.abort()
                source_reservation.release()
            except BaseException as secondary:
                record_secondary_failure(
                    primary,
                    "release-recapture source reservation release also failed",
                    secondary,
                )
        try:
            session.fail(primary)
        except BaseException as secondary:
            record_secondary_failure(
                primary,
                "release-recapture session poison also failed",
                secondary,
            )
        raise


__all__ = [
    "ExactReleaseRecaptureTransaction",
    "ReleaseRecapturePipelineSpec",
    "open_exact_release_recapture",
]
