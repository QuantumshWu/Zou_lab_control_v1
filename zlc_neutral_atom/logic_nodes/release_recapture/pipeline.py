"""Exact two-frame release-recapture reduction in one flat capture pipeline.

The source Measurement publishes every physical camera frame.  This module owns
the one earned K:1 operation in the current product: adjacent, explicitly keyed
READOUT_EVENT 0/1 frames become one survival sample.  It is deliberately not a
generic workflow or configurable grouping framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from zlc_data import (
    INVALID,
    READOUT_EVENT,
    VALID,
    BlockId,
    ComponentValidity,
    DatasetSchema,
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
    ReadoutModelKind,
    ResolvedCalibration,
    _apply_readout_model,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
    calibration_artifact_input_ref,
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
    BoundMeasurement,
    PipelineResult,
    finalize_pipeline_result,
)
from zlc_neutral_atom.runtime.run import RunContext
from zlc_neutral_atom.runtime.streams import (
    AcquisitionCursor,
    AcquisitionProducer,
    AcquisitionStream,
    Delivery,
    EndOfStream,
    Envelope,
    ExactConsumerReadiness,
    ExactReservation,
    OrderedEventSpanHasher,
    ProcessorStageProvenance,
    ReservationState,
    SourceFailed,
    StreamEndedEarly,
    StreamId,
    TraceBinding,
    TraceContext,
)
from zlc_storage import canonical_digest, canonical_text


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

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.ReleaseRecapturePairMetadata",
                "source": self.source.fingerprint,
            }
        )

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

    def digest(self, metadata: object | None) -> str:
        self.validate(metadata)
        assert isinstance(metadata, ReleaseRecapturePairMetadata)
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.ReleaseRecapturePairMetadataContent",
                "initial": self.source.digest(metadata.initial),
                "recaptured": self.source.digest(metadata.recaptured),
            }
        )


_RELEASE_RECAPTURE_METADATA_CONTRACT = ReleaseRecaptureMetadataContract()


@dataclass(frozen=True, slots=True)
class ReleaseRecaptureSampleContract:
    value_schema: ValueSchema

    def __post_init__(self) -> None:
        if not isinstance(self.value_schema, ValueSchema):
            raise TypeError("value_schema must be ValueSchema")

    @property
    def metadata_contract(self) -> ReleaseRecaptureMetadataContract:
        return _RELEASE_RECAPTURE_METADATA_CONTRACT

    @property
    def value_contract(self) -> ValuePayloadContract:
        return ValuePayloadContract(self.value_schema)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.ReleaseRecaptureSample",
                "value": self.value_contract.fingerprint,
                "metadata": self.metadata_contract.fingerprint,
            }
        )

    def snapshot(self, payload: ReleaseRecaptureSample) -> ReleaseRecaptureSample:
        self.validate(payload)
        return payload

    def validate(self, payload: ReleaseRecaptureSample) -> None:
        if not isinstance(payload, ReleaseRecaptureSample):
            raise TypeError("payload must be ReleaseRecaptureSample")
        self.value_contract.validate(payload.survival)
        self.metadata_contract.validate(payload.metadata)

    def digest(self, payload: ReleaseRecaptureSample) -> str:
        self.validate(payload)
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.ReleaseRecaptureSampleContent",
                "survival": self.value_contract.digest(payload.survival),
                "metadata": self.metadata_contract.digest(payload.metadata),
            }
        )



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

    @property
    def operator_fingerprint(self) -> str:
        return canonical_digest(
            {
                "owner": (
                    "zlc_neutral_atom.logic_nodes.release_recapture."
                    "ReleaseRecaptureDatasetEventAdapter"
                ),
                "payload": self.payload_contract.fingerprint,
            }
        )

    @staticmethod
    def value(payload: ReleaseRecaptureSample) -> Value:
        return payload.survival


@dataclass(frozen=True, slots=True)
class ReleaseRecapturePipelineSpec:
    name: str
    measurement: BoundMeasurement
    calibration: ResolvedCalibration
    model_kind: ReadoutModelKind
    per_site: bool
    output_stream_id: StreamId
    output_source_id: str
    block_id: BlockId

    def __post_init__(self) -> None:
        canonical_text(self.name, "name")
        if not isinstance(self.measurement, BoundMeasurement):
            raise TypeError("measurement must be BoundMeasurement")
        if type(self.calibration) is not ResolvedCalibration:
            raise TypeError("calibration must be an admitted ResolvedCalibration")
        self.calibration._require_authority()
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")
        if not isinstance(self.output_stream_id, StreamId):
            raise TypeError("output_stream_id must be StreamId")
        canonical_text(self.output_source_id, "output_source_id")
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")


@dataclass(frozen=True, slots=True)
class ReleaseRecapturePipelineResult:
    pipeline: PipelineResult
    calibration_reference: CalibrationArtifactRef
    model_kind: ReadoutModelKind
    per_site: bool

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, PipelineResult):
            raise TypeError("pipeline must be PipelineResult")
        if not isinstance(
            self.calibration_reference,
            CalibrationArtifactRef,
        ):
            raise TypeError("calibration_reference must be CalibrationArtifactRef")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")

    @property
    def survival(self):
        return self.pipeline.dataset.snapshot


@dataclass(frozen=True, slots=True)
class _PairPlan:
    first_key: DatasetCellAddress
    second_key: DatasetCellAddress
    output_key: DatasetCellAddress


def _pair_plan(
    source_schema: DatasetSchema,
    source_schedule: DatasetCellSchedule,
    output_schema: DatasetSchema,
) -> tuple[_PairPlan, ...]:
    event_axes = tuple(
        (position, axis)
        for position, axis in enumerate(source_schema.point_axes)
        if axis.role == READOUT_EVENT
    )
    if len(event_axes) != 1 or event_axes[0][1].size != 2:
        raise ValueError(
            "release-recapture reducer requires exactly two READOUT_EVENT cells"
        )
    event_position, _event_axis = event_axes[0]
    if len(source_schedule) != (
        output_schema.repeat_axis.size
        * output_schema.point_layout.storage_size
        * 2
    ):
        raise ValueError("release-recapture source cardinality is not exactly 2:1")
    plans: list[_PairPlan] = []
    source_cells = tuple(source_schedule)
    for offset in range(0, len(source_cells), 2):
        first, second = source_cells[offset : offset + 2]
        if first.repeat_index != second.repeat_index:
            raise ValueError("release-recapture pair crosses repeat cells")
        first_multi = source_schema.point_layout.multi_index(
            first.point_storage_index
        )
        second_multi = source_schema.point_layout.multi_index(
            second.point_storage_index
        )
        first_scan = tuple(
            value
            for position, value in enumerate(first_multi)
            if position != event_position
        )
        second_scan = tuple(
            value
            for position, value in enumerate(second_multi)
            if position != event_position
        )
        if (
            first_multi[event_position] != 0
            or second_multi[event_position] != 1
            or first_scan != second_scan
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
                    output_schema.point_layout.storage_index(first_scan),
                ),
            )
        )
    DatasetCellSchedule.from_cells(
        output_schema,
        (plan.output_key for plan in plans),
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
    source = spec.measurement.capture_contract
    source_schema = source.dataset_schema
    event_axes = tuple(
        (position, axis)
        for position, axis in enumerate(source_schema.point_axes)
        if axis.role == READOUT_EVENT
    )
    if len(event_axes) != 1:
        raise ValueError("release-recapture source has no unique READOUT_EVENT axis")
    event_position, event_axis = event_axes[0]
    if event_axis.size != 2:
        raise ValueError("release-recapture source requires two readout events")
    point_axes = tuple(
        axis
        for position, axis in enumerate(source_schema.point_axes)
        if position != event_position
    )
    selected = spec.calibration.artifact.select_model(spec.model_kind)
    if selected.kind is not spec.model_kind:
        raise ValueError("release-recapture model selection changed")
    site_axis = selected.feature.site_axis
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
    # Removing the one READOUT_EVENT axis leaves exactly the scan layout whose
    # rows were frozen by the Measurement binder.
    scan_multi = tuple(
        tuple(
            value
            for position, value in enumerate(
                source_schema.point_layout.multi_index(storage_index)
            )
            if position != event_position
        )
        for storage_index in range(
            source_schema.point_layout.storage_size
        )
        if source_schema.point_layout.multi_index(storage_index)[event_position]
        == 0
    )
    from zlc_data import PointLayout

    output_layout = PointLayout.from_mapping(
        tuple(axis.size for axis in point_axes),
        scan_multi,
    )
    output_schema = DatasetSchema(
        source_schema.repeat_axis,
        point_axes,
        output_layout,
        value_schema,
    )
    plans = _pair_plan(source_schema, source.cell_schedule, output_schema)
    schedule = DatasetCellSchedule.from_cells(
        output_schema,
        (plan.output_key for plan in plans),
    )
    contract = ReleaseRecaptureSampleContract(value_schema)
    return output_schema, contract, plans, schedule


def release_recapture_output_schema(
    spec: ReleaseRecapturePipelineSpec,
) -> DatasetSchema:
    """Resolve the exact survival dataset schema without touching hardware."""

    if not isinstance(spec, ReleaseRecapturePipelineSpec):
        raise TypeError("spec must be ReleaseRecapturePipelineSpec")
    return _output_contract(spec)[0]


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
    output_edge: FrozenDatasetEdge
    pair_plan: tuple[_PairPlan, ...]
    readiness: ExactConsumerReadiness | None
    ordered_inputs: OrderedEventSpanHasher
    _next_source_index: int = 0
    _first: Envelope[CameraSample] | None = None
    _result: ReleaseRecapturePipelineResult | None = None
    _error: BaseException | None = None
    _cancelled: bool = False
    _done: bool = False

    def start(self, context: RunContext) -> None:
        self.session.prepare(context)
        self.session.start(context)

    def capture_all(self, context: RunContext) -> None:
        for _ordinal in range(self.spec.measurement.capture_contract.total_events):
            context.checkpoint()
            self.session.capture_next(context)
            self._consume_one(context)

    def _consume_one(self, context: RunContext) -> None:
        if self._done or self._cancelled:
            raise RuntimeError("release-recapture reducer is no longer live")
        call_bound = (
            self.spec.measurement.capture_port.capability
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
        self.ordered_inputs.update(envelope.ref)
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
            if (
                first.trace.config_revision != envelope.trace.config_revision
                or first.trace.control_revision
                != envelope.trace.control_revision
            ):
                raise ValueError(
                    "release-recapture pair crosses config/control revisions"
                )
            emitted = self.output_producer.emit(
                sample,
                captured_at=max(first.captured_at, envelope.captured_at),
                trace=TraceContext(
                    run_id=envelope.trace.run_id,
                    source_id=self.spec.output_source_id,
                    correlation_id=canonical_digest(
                        {
                            "owner": "zlc.release-recapture-pair",
                            "first": first.trace.correlation_id,
                            "second": envelope.trace.correlation_id,
                        }
                    ),
                    causation_refs=(
                        first.ref,
                        envelope.ref,
                        calibration_artifact_input_ref(
                            self.spec.calibration.reference
                        ),
                    ),
                    config_revision=envelope.trace.config_revision,
                    control_revision=envelope.trace.control_revision,
                ),
                join_key=group.output_key,
            )
            output = self.output_cursor.next(timeout=call_bound)
            if output.envelope.event_id != emitted.event_id:
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
        model = self.spec.calibration.artifact.select_model(
            self.spec.model_kind
        )
        initial = _apply_readout_model(model, first.payload.image)
        recaptured = _apply_readout_model(model, second.payload.image)
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
        schema = self.output_edge.schema.cell_schema
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

    def complete(self, context: RunContext) -> ReleaseRecapturePipelineResult:
        completion: CaptureCompletion = self.session.complete(context)
        if self._first is not None or self._next_source_index != (
            self.spec.measurement.capture_contract.total_events
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
        sealed = sealed._with_derivation(
            readiness,
            self.ordered_inputs.seal(self.source_reservation.end_sequence),
        )
        self.source_reservation.complete_consumer(
            completion.eos,
            self,
        )
        pipeline = finalize_pipeline_result(
            dataset=sealed,
            capture_completion=completion,
        )
        result = ReleaseRecapturePipelineResult(
            pipeline,
            self.spec.calibration.reference,
            self.spec.model_kind,
            self.spec.per_site,
        )
        self._result = result
        self._done = True
        return result

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
        return self._result.pipeline.dataset

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
    measurement = spec.measurement
    contract = measurement.capture_contract
    session = open_capture_session(
        measurement.capture_port,
        contract,
        TraceBinding(context.run_id.value, contract.source_id),
        measurement.capture_spec,
    )
    source_reservation = output_reservation = None
    output_builder = None
    reducer = None
    try:
        capture_input = session.processor_input_binding
        source_reservation = session.reserve_exact()
        source_cursor = source_reservation.activate()
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
            trace_binding=TraceBinding(
                context.run_id.value,
                spec.output_source_id,
            ),
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
            trace_binding=TraceBinding(
                context.run_id.value,
                spec.output_source_id,
            ),
            payload_contract_fingerprint=payload_contract.fingerprint,
            join_key_contract_fingerprint=output_key_contract.fingerprint,
            source_key_sequence_digest=output_edge.exact_key_sequence_digest,
            total_events=len(plans),
        )
        stage_fingerprint = canonical_digest(
            {
                "owner": "zlc_neutral_atom.release-recapture-reducer",
                "group_size": 2,
                "source_schema": contract.dataset_schema.fingerprint,
                "output_schema": output_schema.fingerprint,
                "calibration": calibration_artifact_ref_to_tree(
                    spec.calibration.reference
                ),
                "model_kind": spec.model_kind.value,
                "per_site": spec.per_site,
            }
        )
        chain_digest = canonical_digest(
            {
                "contract": "zlc_neutral_atom.ExactReleaseRecaptureChain",
                "reducer": stage_fingerprint,
                "source_contract": capture_input.input_edge.consumer_contract_digest,
                "source_schedule": capture_input.input_edge.schedule_digest,
                "downstream": downstream.chain_contract_digest,
                "input_events": contract.total_events,
                "output_events": len(plans),
            }
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
            output_edge,
            plans,
            readiness=None,
            ordered_inputs=OrderedEventSpanHasher(
                source_reservation.stream_id,
                source_reservation.stream_generation,
                source_reservation.start_sequence,
            ),
        )
        readiness = source_reservation.bind_consumer(
            reducer,
            source_contract_digest=(
                capture_input.input_edge.consumer_contract_digest
            ),
            source_schedule_digest=capture_input.input_edge.schedule_digest,
            source_key_sequence_digest=(
                capture_input.input_edge.exact_key_sequence_digest
            ),
            chain_contract_digest=chain_digest,
            downstream=downstream,
            owner_liveness=reducer._validate_liveness,
            owner_completion=reducer._await_completion,
            owner_cancel=reducer.cancel,
            processor_stage=ProcessorStageProvenance(
                stage_fingerprint,
                (
                    calibration_artifact_input_ref(
                        spec.calibration.reference
                    ),
                ),
            ),
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
    "ReleaseRecapturePipelineResult",
    "ReleaseRecapturePipelineSpec",
    "open_exact_release_recapture",
    "release_recapture_output_schema",
]
