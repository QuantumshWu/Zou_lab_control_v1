"""Exact camera Dataset, provenance, and capture-session authority."""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass
from enum import Enum

from zlc_storage import (
    canonical_text as _canonical_text,
    exact_mapping as _exact_tree,
    nonnegative_integer as _nonnegative_int,
    sha256_text as _sha256,
)

from zlc_data import DatasetSchema, StreamGenerationId
from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT,
    CameraCapabilityEvidence,
    CameraCaptureDescriptor,
    CameraDatasetEventAdapter,
    FrozenCaptureSpec,
    ReadoutBindingKey,
    camera_capture_descriptor_from_tree,
    camera_capture_descriptor_to_tree,
    readout_binding_key_from_tree,
    readout_binding_key_to_tree,
)
from zlc_neutral_atom.devices.camera.capture_port import (
    BoundCapturePort,
    CaptureCapabilitySnapshot,
    CapturePayloadContract,
    CapturePreparedAck,
    CaptureStartedAck,
    CaptureTerminalAck,
    CapturedPayloadAck,
    CompleteCaptureCommand,
    PrepareCaptureCommand,
    ReadCaptureCommand,
    StartCaptureCommand,
)


from zlc_neutral_atom.runtime._failure import record_secondary_failure, safe_error_summary
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellKeyContract,
    DatasetCellSchedule,
    DatasetEventAdapter,
    FrozenDatasetEdge,
    OrderedDatasetMetadataHasher,
    SealedDatasetArtifact,
)
from zlc_neutral_atom.runtime.resources import (
    DeviceBindingStamp,
    device_binding_stamp_from_tree,
    device_binding_stamp_to_tree,
)
from zlc_neutral_atom.runtime.run import RunContext
from zlc_neutral_atom.runtime.streams import (
    AcquisitionProducer,
    AcquisitionStream,
    EndOfStream,
    EventSpanRef,
    ExactConsumerReadiness,
    ExactReservation,
    OrderedEventSpanHasher,
    ProcessorStageProvenance,
    ReservationState,
    SourceFailed,
    StreamError,
    StreamId,
    TraceBinding,
    TraceContext,
)


_COMPLETION_TOKEN = object()
_PROCESSOR_INPUT_BINDING_TOKEN = object()






@dataclass(frozen=True)
class CameraCaptureProvenance:
    """Owner-derived physical facts for one raw camera capture binding.

    No readout event is selected here.  A later calibration/analysis request
    combines this raw descriptor with an explicit ``CalibrationCaptureLayout``
    to derive a ``FrameContract``.  ``camera_arm_spec_fingerprint`` is only the
    frozen camera arm/capture-spec digest; it is not an FPGA pulse-schedule digest.
    """

    descriptor: CameraCaptureDescriptor
    binding: ReadoutBindingKey
    binding_stamp: DeviceBindingStamp
    capability_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CameraCaptureDescriptor):
            raise TypeError("descriptor must be CameraCaptureDescriptor")
        if not isinstance(self.binding, ReadoutBindingKey):
            raise TypeError("binding must be ReadoutBindingKey")
        settings_fingerprints = {
            setting.opaque_frame_settings_fingerprint
            for setting in self.descriptor.event_settings
        }
        if None in settings_fingerprints or len(settings_fingerprints) != 1:
            raise ValueError(
                "descriptor event settings require one active settings fingerprint"
            )
        if self.descriptor.camera_arm_spec_fingerprint is None:
            raise ValueError(
                "camera descriptor requires camera_arm_spec_fingerprint"
            )
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("binding_stamp must be DeviceBindingStamp")
        if (
            self.descriptor.camera_identity
            != self.binding_stamp.physical_identity.stable_device_identity
        ):
            raise ValueError("camera descriptor identity differs from binding stamp")
        _sha256(self.capability_fingerprint, "capability_fingerprint")

    @property
    def camera_arm_spec_fingerprint(self) -> str:
        value = self.descriptor.camera_arm_spec_fingerprint
        assert value is not None
        return value

    @property
    def active_settings_fingerprint(self) -> str:
        value = self.descriptor.event_settings[0].opaque_frame_settings_fingerprint
        assert value is not None
        return value

    def validate_schema(self, schema: DatasetSchema) -> None:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        self.descriptor.validate_schema(schema)


def camera_capture_provenance_to_tree(
    value: CameraCaptureProvenance,
) -> dict[str, object]:
    if not isinstance(value, CameraCaptureProvenance):
        raise TypeError("value must be CameraCaptureProvenance")
    return {
        "descriptor": camera_capture_descriptor_to_tree(value.descriptor),
        "binding": readout_binding_key_to_tree(value.binding),
        "binding_stamp": device_binding_stamp_to_tree(value.binding_stamp),
        "capability_fingerprint": value.capability_fingerprint,
    }


def camera_capture_provenance_from_tree(tree: object) -> CameraCaptureProvenance:
    data = _exact_tree(
        tree,
        {
            "descriptor",
            "binding",
            "binding_stamp",
            "capability_fingerprint",
        },
        "camera capture provenance",
        discriminator=None,
    )
    return CameraCaptureProvenance(
        descriptor=camera_capture_descriptor_from_tree(data["descriptor"]),
        binding=readout_binding_key_from_tree(data["binding"]),
        binding_stamp=device_binding_stamp_from_tree(data["binding_stamp"]),
        capability_fingerprint=data["capability_fingerprint"],
    )


@dataclass(frozen=True)
class CameraCaptureContract:
    stream_id: StreamId
    dataset_edge: FrozenDatasetEdge
    capability: CaptureCapabilitySnapshot
    camera_provenance: CameraCaptureProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(self.dataset_edge, FrozenDatasetEdge):
            raise TypeError("dataset_edge must be FrozenDatasetEdge")
        if not isinstance(self.capability, CaptureCapabilitySnapshot):
            raise TypeError("capability must be CaptureCapabilitySnapshot")
        edge = self.dataset_edge
        if edge.cell_schedule is None:
            raise ValueError("capture dataset edge requires an exact cell schedule")
        if self.capability.payload_contract is not edge.payload_contract:
            raise ValueError(
                "capture capability and dataset edge must share the payload owner"
            )
        for member in ("source_ordinal", "captured_at", "correlation_id"):
            if not callable(getattr(edge.payload_contract, member, None)):
                raise TypeError(f"payload_contract.{member} must be callable")
        # A raw camera CaptureArtifact is defined by the acquisition owner, not
        # by schema equality.  Require the exact identity adapter implementation.
        if type(edge.event_adapter) is not CameraDatasetEventAdapter or (
            edge.operator_fingerprint != CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT
        ):
            raise ValueError(
                "raw camera provenance requires the owner identity event adapter"
            )
        if not isinstance(self.camera_provenance, CameraCaptureProvenance):
            raise TypeError("camera_provenance must be CameraCaptureProvenance")
        self.camera_provenance.validate_schema(edge.schema)
        if self.camera_provenance.binding.value != self.source_id:
            raise ValueError("camera provenance binding differs from capture source_id")
        if self.camera_provenance.binding_stamp != self.capability.binding_stamp:
            raise ValueError("camera provenance binding differs from capability")
        if (
            self.camera_provenance.capability_fingerprint
            != self.capability.capability_fingerprint
        ):
            raise ValueError(
                "camera provenance capability fingerprint differs from capability"
            )
        self.capability.camera_physical_facts.validate_descriptor(
            self.camera_provenance.descriptor
        )

    @property
    def source_id(self) -> str:
        return self.capability.camera_capability_evidence.source_id

    @property
    def dataset_schema(self) -> DatasetSchema:
        return self.dataset_edge.schema

    @property
    def payload_contract(self) -> CapturePayloadContract:
        return self.dataset_edge.payload_contract  # type: ignore[return-value]

    @property
    def event_adapter(self) -> DatasetEventAdapter:
        return self.dataset_edge.event_adapter

    @property
    def cell_schedule(self) -> DatasetCellSchedule:
        schedule = self.dataset_edge.cell_schedule
        assert schedule is not None
        return schedule

    @property
    def capture_spec_owner_fingerprint(self) -> str:
        return self.capability.capture_spec_owner_fingerprint

    @property
    def total_events(self) -> int:
        return len(self.cell_schedule)



class CaptureSessionState(str, Enum):
    NEW = "NEW"
    PREPARED = "PREPARED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CaptureCompletion:
    __slots__ = (
        "_session",
        "_eos",
        "_terminal",
        "_trace_binding",
        "_source_dataset_schema",
        "_source_cell_schedule",
        "_camera_provenance",
        "_camera_capability_evidence",
        "_camera_arm_spec",
        "_source_event_adapter_operator_fingerprint",
        "_chain_contract_digest",
        "_source_event_span",
        "_processor_stages",
        "_terminal_reservation",
        "_direct_terminal_consumer",
    )

    def __init__(
        self,
        authority: object,
        *,
        session: "CaptureSession",
        terminal: CaptureTerminalAck,
    ) -> None:
        if authority is not _COMPLETION_TOKEN:
            raise PermissionError("CaptureCompletion can only be minted by CaptureSession")
        readiness = session._exact_consumer_readiness
        if readiness is None:
            raise RuntimeError("capture completion has no exact consumer readiness")
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_eos", None)
        object.__setattr__(self, "_terminal", terminal)
        object.__setattr__(self, "_trace_binding", session._trace_binding)
        object.__setattr__(
            self,
            "_source_dataset_schema",
            session._contract.dataset_schema,
        )
        object.__setattr__(
            self,
            "_source_cell_schedule",
            session._contract.cell_schedule,
        )
        object.__setattr__(
            self,
            "_camera_provenance",
            session._contract.camera_provenance,
        )
        object.__setattr__(
            self,
            "_camera_capability_evidence",
            session._contract.capability.camera_capability_evidence,
        )
        object.__setattr__(self, "_camera_arm_spec", session._capture_spec)
        object.__setattr__(
            self,
            "_source_event_adapter_operator_fingerprint",
            session._contract.dataset_edge.operator_fingerprint,
        )
        object.__setattr__(
            self,
            "_chain_contract_digest",
            readiness.chain_contract_digest,
        )
        source_reservation = readiness._source_reservation
        if source_reservation._stream is not session._stream:
            raise RuntimeError(
                "capture completion readiness belongs to another source stream"
            )
        source_event_span_hasher = session._source_event_span_hasher
        if source_event_span_hasher is None:
            raise RuntimeError("capture completion has no source event span owner")
        source_event_span = source_event_span_hasher.seal(
            source_reservation.end_sequence
        )
        if (
            source_event_span.stream_id != session._stream.stream_id
            or source_event_span.generation != session._stream.generation
            or source_event_span.start_sequence != source_reservation.start_sequence
            or source_event_span.end_sequence != source_reservation.end_sequence
        ):
            raise RuntimeError("capture source event span differs from its reservation")
        object.__setattr__(self, "_source_event_span", source_event_span)
        object.__setattr__(self, "_processor_stages", readiness.processor_stages)
        # Process-local identity capability.  PipelineResult uses it to prove
        # that its SealedDatasetArtifact came from this readiness chain's live
        # terminal DatasetBuilder, including through processor chains.
        object.__setattr__(
            self,
            "_terminal_reservation",
            readiness._terminal_reservation,
        )
        direct_terminal = (
            readiness._source_reservation is readiness._terminal_reservation
        )
        if direct_terminal != (not readiness.processor_stages):
            raise RuntimeError(
                "capture readiness terminal identity differs from its stage chain"
            )
        object.__setattr__(self, "_direct_terminal_consumer", direct_terminal)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CaptureCompletion is immutable")

    @property
    def eos(self) -> EndOfStream:
        if self._eos is None:
            raise RuntimeError("capture completion has not been committed")
        return self._eos

    @property
    def terminal(self) -> CaptureTerminalAck:
        return self._terminal

    @property
    def trace_binding(self) -> TraceBinding:
        return self._trace_binding

    @property
    def source_dataset_schema(self) -> DatasetSchema:
        return self._source_dataset_schema

    @property
    def source_cell_schedule(self) -> DatasetCellSchedule:
        return self._source_cell_schedule

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
    def source_event_adapter_operator_fingerprint(self) -> str:
        return self._source_event_adapter_operator_fingerprint

    @property
    def chain_contract_digest(self) -> str:
        return self._chain_contract_digest

    @property
    def direct_terminal_consumer(self) -> bool:
        return self._direct_terminal_consumer

    @property
    def source_stream_id(self) -> StreamId:
        return self._source_event_span.stream_id

    @property
    def source_generation(self) -> StreamGenerationId:
        return self._source_event_span.generation

    @property
    def source_start_sequence(self) -> int:
        return self._source_event_span.start_sequence

    @property
    def source_end_sequence(self) -> int:
        return self._source_event_span.end_sequence

    @property
    def source_event_span(self) -> EventSpanRef:
        return self._source_event_span

    @property
    def processor_stages(self) -> tuple[ProcessorStageProvenance, ...]:
        return self._processor_stages

    def _commit_pipeline_result(self, dataset: SealedDatasetArtifact) -> None:
        """Atomically consume dataset and completion live authority."""

        session = self._session
        if session is None:
            raise RuntimeError("capture completion authority was already consumed")
        session._commit_pipeline_authority(self, dataset)

    def _validate_pipeline_authority(
        self,
    ) -> tuple["CaptureSession", ExactReservation]:
        """Validate the final no-fail authority commit before any owner mutates."""

        session = self._session
        reservation = self._terminal_reservation
        if session is None or reservation is None:
            raise RuntimeError("capture completion authority was already consumed")
        session._assert_owner_thread()
        if not session.owns_completion(self):
            raise RuntimeError("capture completion authority is absent or differs")
        return session, reservation




class CaptureProcessorInputBinding:
    """The exact stream edge and reservation minted by one capture session."""

    __slots__ = (
        "_capture_contract",
        "_stream",
        "_join_key_contract",
        "_input_edge",
        "_session_reservation",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("CaptureProcessorInputBinding is sealed")

    def __init__(
        self,
        authority: object,
        *,
        capture_contract: CameraCaptureContract,
        stream: AcquisitionStream,
    ) -> None:
        if authority is not _PROCESSOR_INPUT_BINDING_TOKEN:
            raise PermissionError(
                "CaptureProcessorInputBinding can only be minted by CaptureSession"
            )
        if not isinstance(capture_contract, CameraCaptureContract):
            raise TypeError("capture_contract must be CameraCaptureContract")
        if not isinstance(stream, AcquisitionStream):
            raise TypeError("stream must be AcquisitionStream")
        input_edge = capture_contract.dataset_edge
        join_key_contract = stream._join_key_contract
        if not isinstance(join_key_contract, DatasetCellKeyContract):
            raise TypeError("capture stream has no DatasetCellKeyContract owner")
        input_edge.validate_stream(stream)
        if stream.stream_id != capture_contract.stream_id:
            raise ValueError("processor stream id differs from capture contract")
        object.__setattr__(self, "_capture_contract", capture_contract)
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_join_key_contract", join_key_contract)
        object.__setattr__(self, "_input_edge", input_edge)
        object.__setattr__(self, "_session_reservation", None)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CaptureProcessorInputBinding is immutable")

    @property
    def capture_contract(self) -> CameraCaptureContract:
        return self._capture_contract

    @property
    def stream(self) -> AcquisitionStream:
        return self._stream

    @property
    def payload_contract(self) -> CapturePayloadContract:
        return self._input_edge.payload_contract

    @property
    def join_key_contract(self) -> DatasetCellKeyContract:
        return self._join_key_contract

    @property
    def input_edge(self) -> FrozenDatasetEdge:
        return self._input_edge

    def _bind_session_reservation(
        self,
        authority: object,
        reservation: ExactReservation,
    ) -> None:
        """Record the one reservation minted through ``CaptureSession`` itself."""

        if authority is not _PROCESSOR_INPUT_BINDING_TOKEN:
            raise PermissionError("capture reservation binding is session-owned")
        if type(reservation) is not ExactReservation:
            raise TypeError("reservation must be an exact ExactReservation")
        stream = self._stream
        if reservation._stream is not stream:
            raise PermissionError("reservation belongs to another capture stream")
        with stream._condition:
            if self._session_reservation is not None:
                raise RuntimeError("capture session reservation was already bound")
            if stream._reservations.get(reservation._token) is not reservation:
                raise RuntimeError("capture session reservation is not registered")
            if reservation.state is not ReservationState.RESERVED:
                raise RuntimeError("capture session reservation is not RESERVED")
            object.__setattr__(self, "_session_reservation", reservation)

    def require_reservation(self, reservation: ExactReservation) -> None:
        """Require the active reservation minted by this exact CaptureSession."""

        if type(reservation) is not ExactReservation:
            raise TypeError("reservation must be an exact ExactReservation")
        stream = self._stream
        if reservation is not self._session_reservation:
            raise PermissionError(
                "reservation was not minted by this CaptureSession"
            )
        with stream._condition:
            if stream._reservations.get(reservation._token) is not reservation:
                raise RuntimeError(
                    "reservation is no longer registered with the capture stream"
                )
            if reservation.state is not ReservationState.ACTIVE:
                raise RuntimeError("capture session reservation is not ACTIVE")


class CaptureSession:
    """One owner of producer, device session id, ordinal, and terminal receipt."""

    def __init__(
        self,
        port: BoundCapturePort,
        contract: CameraCaptureContract,
        trace_binding: TraceBinding,
        capture_spec: FrozenCaptureSpec,
    ) -> None:
        if not isinstance(trace_binding, TraceBinding):
            raise TypeError("trace_binding must be TraceBinding")
        if trace_binding.source_id != contract.source_id:
            raise ValueError("TraceBinding source differs from CameraCaptureContract")
        if not isinstance(capture_spec, FrozenCaptureSpec):
            raise TypeError("capture_spec must be FrozenCaptureSpec")
        if capture_spec.owner_fingerprint != contract.capture_spec_owner_fingerprint:
            raise ValueError("capture spec owner differs from CameraCaptureContract")
        self._port = port
        self._contract = contract
        self._trace_binding = trace_binding
        self._capture_spec = capture_spec
        self._capture_spec_digest = capture_spec.digest
        self._session_id = uuid.uuid4().hex
        stream, producer = AcquisitionStream.create(
            contract.stream_id,
            contract.payload_contract,
            join_key_contract=DatasetCellKeyContract.from_schema(
                contract.dataset_schema
            ),
        )
        processor_input_binding = CaptureProcessorInputBinding(
            _PROCESSOR_INPUT_BINDING_TOKEN,
            capture_contract=contract,
            stream=stream,
        )
        self._stream = stream
        self._producer: AcquisitionProducer = producer
        self._processor_input_binding = processor_input_binding
        self._state = CaptureSessionState.NEW
        self._delivered = 0
        self._metadata_hasher = OrderedDatasetMetadataHasher(
            contract.dataset_edge.metadata_contract_fingerprint
        )
        self._completion: CaptureCompletion | None = None
        self._reservation: ExactReservation | None = None
        self._source_event_span_hasher: OrderedEventSpanHasher | None = None
        self._exact_consumer_readiness: ExactConsumerReadiness | None = None
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._owner_thread_id = threading.get_ident()
        self._hardware_prepare_attempted = False

    @property
    def stream(self) -> AcquisitionStream:
        return self._stream

    @property
    def processor_input_binding(self) -> CaptureProcessorInputBinding:
        with self._lock:
            if self._state is not CaptureSessionState.NEW:
                raise RuntimeError(
                    "processor input binding is only available while capture session is NEW"
                )
            binding = self._processor_input_binding
            return binding

    @property
    def state(self) -> CaptureSessionState:
        with self._lock:
            return self._state

    @property
    def session_id(self) -> str:
        return self._session_id

    def reserve_exact(self) -> ExactReservation:
        """Mint the only formal reservation that may authorize this capture."""

        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                if self._state is not CaptureSessionState.NEW:
                    raise RuntimeError("exact reservation must precede capture prepare")
                if self._reservation is not None:
                    raise RuntimeError("capture session already has an exact reservation")
            reservation = self._stream.reserve(
                total_events=self._contract.total_events,
                trace_binding=self._trace_binding,
            )
            source_event_span_hasher = OrderedEventSpanHasher(
                self._stream.stream_id,
                self._stream.generation,
                reservation.start_sequence,
            )
            try:
                self._processor_input_binding._bind_session_reservation(
                    _PROCESSOR_INPUT_BINDING_TOKEN,
                    reservation,
                )
            except BaseException:
                reservation.abort()
                reservation.release()
                raise
            with self._lock:
                self._reservation = reservation
                self._source_event_span_hasher = source_event_span_hasher
            return reservation

    def bind_exact_consumer(self, readiness: ExactConsumerReadiness) -> None:
        """Bind a complete required exact chain before any capture command."""

        with self._operation_lock:
            self._assert_owner_thread()
            if not isinstance(readiness, ExactConsumerReadiness):
                raise TypeError("readiness must be ExactConsumerReadiness")
            with self._lock:
                if self._state is not CaptureSessionState.NEW:
                    raise RuntimeError("exact consumer readiness must precede capture prepare")
                reservation = self._reservation
                if reservation is None:
                    raise RuntimeError("capture session has no exact reservation")
                if self._exact_consumer_readiness is not None:
                    raise RuntimeError("capture session already has exact consumer readiness")
            self._validate_readiness(readiness, reservation)
            readiness._claim_binding(self)
            with self._lock:
                self._exact_consumer_readiness = readiness

    def prepare(self, context: RunContext) -> None:
        with self._operation_lock:
            self._assert_owner_thread()
            if context.run_id.value != self._trace_binding.run_id:
                raise ValueError("RunContext differs from the frozen TraceBinding")
            with self._lock:
                if self._state is not CaptureSessionState.NEW:
                    raise RuntimeError("capture session can only be prepared once")
                reservation = self._reservation
                readiness = self._exact_consumer_readiness
            if reservation is None:
                raise RuntimeError("capture cannot prepare without its exact reservation")
            if readiness is None:
                raise RuntimeError("capture has no exact consumer readiness proof")
            self._validate_readiness(readiness, reservation)
            self._validate_current_capability()
            with self._lock:
                self._hardware_prepare_attempted = True
            try:
                ack = context.device(self._port.device.key).execute(
                    PrepareCaptureCommand(
                        session_id=self._session_id,
                        capture_spec_payload=self._capture_spec.payload,
                        capture_spec_owner_fingerprint=(
                            self._capture_spec.owner_fingerprint
                        ),
                        capture_spec_fingerprint=self._capture_spec_digest,
                        capability_fingerprint=self._port.capability.capability_fingerprint,
                        settings_fingerprint=self._port.capability.settings_fingerprint,
                        expected_total_events=self._contract.total_events,
                        timeout_seconds=(
                            self._port.capability.max_blocking_call_seconds
                        ),
                    )
                )
                self._validate_ack(ack, CapturePreparedAck)
                if (
                    ack.settings_fingerprint
                    != self._port.capability.settings_fingerprint
                    or ack.capability_fingerprint
                    != self._port.capability.capability_fingerprint
                    or ack.capture_spec_fingerprint != self._capture_spec_digest
                ):
                    raise RuntimeError(
                        "capture prepare acknowledgement differs from frozen contract"
                    )
            except BaseException as error:
                self._poison(
                    SourceFailed(f"capture prepare failed: {safe_error_summary(error)}")
                )
                raise
            with self._lock:
                self._state = CaptureSessionState.PREPARED

    def start(self, context: RunContext) -> None:
        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                if self._state is not CaptureSessionState.PREPARED:
                    raise RuntimeError("capture session is not prepared")
                reservation = self._reservation
                if reservation is None:
                    raise RuntimeError("capture cannot start without its exact reservation")
                if reservation.state is not ReservationState.ACTIVE:
                    raise RuntimeError("capture exact reservation is not active")
                readiness = self._exact_consumer_readiness
                if readiness is None:
                    raise RuntimeError("capture has no exact consumer readiness proof")
                self._validate_readiness(readiness, reservation)
            try:
                ack = context.device(self._port.device.key).execute(
                    StartCaptureCommand(
                        self._session_id,
                        self._port.capability.max_blocking_call_seconds,
                    )
                )
                self._validate_ack(ack, CaptureStartedAck)
            except BaseException as error:
                self._poison(
                    SourceFailed(f"capture start failed: {safe_error_summary(error)}")
                )
                raise
            with self._lock:
                self._state = CaptureSessionState.STARTED

    def _validate_readiness(
        self,
        readiness: ExactConsumerReadiness,
        reservation: ExactReservation,
    ) -> None:
        readiness.validate_source(
            reservation=reservation,
            trace_binding=self._trace_binding,
            payload_contract_fingerprint=(
                self._contract.dataset_edge.payload_contract_fingerprint
            ),
            join_key_contract_fingerprint=DatasetCellKeyContract.from_schema(
                self._contract.dataset_schema
            ).fingerprint,
            source_contract_digest=self._contract.dataset_edge.consumer_contract_digest,
            source_schedule_digest=self._contract.dataset_edge.schedule_digest,
            source_key_sequence_digest=(
                self._contract.dataset_edge.exact_key_sequence_digest
            ),
            total_events=self._contract.total_events,
        )

    def capture_next(
        self,
        context: RunContext,
    ) -> None:
        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                if self._state is not CaptureSessionState.STARTED:
                    raise RuntimeError("capture session is not started")
                expected_ordinal = self._delivered
                if expected_ordinal >= self._contract.total_events:
                    raise RuntimeError("capture already delivered its expected event count")
                join_key = self._contract.cell_schedule.cell_at(expected_ordinal)
            try:
                ack = context.device(self._port.device.key).execute(
                    ReadCaptureCommand(
                        self._session_id,
                        self._port.capability.max_blocking_call_seconds,
                    )
                )
                self._validate_ack(ack, CapturedPayloadAck)
                payload_contract = self._contract.payload_contract
                payload = ack.payload
                actual_ordinal = _nonnegative_int(
                    payload_contract.source_ordinal(payload),
                    "payload source ordinal",
                )
                if actual_ordinal != expected_ordinal:
                    raise StreamError(
                        f"payload ordinal {actual_ordinal} differs from expected "
                        f"{expected_ordinal}"
                    )
                captured_at = payload_contract.captured_at(payload)
                if not math.isfinite(float(captured_at)):
                    raise ValueError("payload captured_at must be finite")
                correlation_id = payload_contract.correlation_id(payload)
                _canonical_text(correlation_id, "payload correlation_id")
                envelope = self._producer.emit(
                    payload,
                    captured_at=float(captured_at),
                    trace=TraceContext(
                        run_id=self._trace_binding.run_id,
                        source_id=self._trace_binding.source_id,
                        correlation_id=correlation_id,
                    ),
                    join_key=join_key,
                )
                source_event_span_hasher = self._source_event_span_hasher
                if source_event_span_hasher is None:
                    raise RuntimeError("capture has no source event span owner")
                source_event_span_hasher.update(envelope.ref)
                stored = envelope.payload
                if _nonnegative_int(
                    payload_contract.source_ordinal(stored),
                    "stored payload source ordinal",
                ) != actual_ordinal:
                    raise StreamError("payload snapshot changed the physical source ordinal")
                if float(payload_contract.captured_at(stored)) != float(captured_at):
                    raise StreamError("payload snapshot changed captured_at")
                if payload_contract.correlation_id(stored) != correlation_id:
                    raise StreamError("payload snapshot changed correlation_id")
                metadata_contract = self._contract.dataset_edge.metadata_contract
                metadata = metadata_contract.snapshot(stored)
                metadata_contract.validate(metadata)
                metadata_digest = _sha256(
                    metadata_contract.digest(metadata),
                    "payload metadata digest",
                )
            except BaseException as error:
                self._poison(
                    SourceFailed(
                        "captured payload failed validation/publish: "
                        + safe_error_summary(error)
                    )
                )
                raise
            with self._lock:
                self._metadata_hasher.update(metadata_digest)
                self._delivered += 1

    def complete(self, context: RunContext) -> CaptureCompletion:
        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                if self._state is not CaptureSessionState.STARTED:
                    raise RuntimeError("capture session is not started")
                if self._delivered != self._contract.total_events:
                    raise RuntimeError(
                        "cannot complete before every scheduled payload is delivered"
                    )
            try:
                ack = context.device(self._port.device.key).execute(
                    CompleteCaptureCommand(
                        self._session_id,
                        self._contract.total_events,
                        self._port.capability.max_blocking_call_seconds,
                    )
                )
                self._validate_ack(ack, CaptureTerminalAck)
                if (
                    ack.produced_count != self._contract.total_events
                    or ack.drained_count != self._contract.total_events
                    or not ack.source_stopped
                    or not ack.no_more_frames
                    or not ack.joined
                ):
                    raise StreamError(
                        "capture terminal counters/stop/drain/join proof failed"
                    )
                expected_digest = self._metadata_hasher.digest()
                if ack.ordered_metadata_digest != expected_digest:
                    raise StreamError("capture terminal metadata digest differs")
                if (
                    ack.settings_fingerprint
                    != self._port.capability.settings_fingerprint
                    or ack.capability_fingerprint
                    != self._port.capability.capability_fingerprint
                    or ack.capture_spec_fingerprint != self._capture_spec_digest
                ):
                    raise StreamError("capture terminal contract binding differs")
                completion = CaptureCompletion(
                    _COMPLETION_TOKEN,
                    session=self,
                    terminal=ack,
                )
                eos = self._producer.finish()
                object.__setattr__(completion, "_eos", eos)
            except BaseException as error:
                self._poison(
                    SourceFailed(
                        "capture terminal validation failed: "
                        + safe_error_summary(error)
                    )
                )
                raise
            with self._lock:
                self._state = CaptureSessionState.COMPLETED
                self._completion = completion
            return completion

    def fail(self, error: BaseException) -> None:
        """Poison software authority; physical termination belongs to cleanup."""

        with self._operation_lock:
            self._assert_owner_thread()
            failure = SourceFailed(
                "capture session aborted: " + safe_error_summary(error)
            )
            self._poison(failure)

    def cleanup(self, context: RunContext) -> CleanupReport:
        """Poison unfinished data, then execute cleanup-capable stop/drain/join."""

        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                completed = self._state is CaptureSessionState.COMPLETED
            if not completed:
                self._poison(SourceFailed("capture terminated during cleanup"))
            report: CleanupReport | None = None
            port_error: BaseException | None = None
            try:
                report = (
                    self._port.cleanup(context, self._session_id)
                    if self._hardware_prepare_attempted
                    else self._port.verify_idle(context)
                )
            except BaseException as error:
                port_error = error
            release_errors: list[BaseException] = []
            reservation = self._reservation
            if reservation is not None and reservation.state is not ReservationState.RELEASED:
                try:
                    if not reservation.consumer_bound:
                        if reservation.state is not ReservationState.COMPLETED:
                            reservation.abort(cancelled=context.cancellation.is_cancelled)
                        reservation.release()
                    elif reservation.state in (
                        ReservationState.COMPLETED,
                        ReservationState.FAILED,
                        ReservationState.CANCELLED,
                    ):
                        reservation.release()
                    else:
                        raise RuntimeError(
                            "exact consumer did not terminate its reservation"
                        )
                except BaseException as error:
                    release_errors.append(error)
            if reservation is None or reservation.state is ReservationState.RELEASED:
                completion = self._completion
                if completion is not None:
                    try:
                        self._consume_completion_authority(completion)
                    except BaseException as error:
                        release_errors.append(error)
                with self._lock:
                    self._reservation = None
                    self._exact_consumer_readiness = None
                    self._source_event_span_hasher = None
            if port_error is not None:
                for error in release_errors:
                    record_secondary_failure(
                        port_error,
                        "exact reservation teardown also failed",
                        error,
                    )
                raise port_error
            assert report is not None
            if not release_errors:
                return report
            return CleanupReport.complete(
                errors=(*report.errors, *release_errors),
            )

    def _validate_current_capability(self) -> None:
        if self._port.require_current_capability() is not self._port.capability:
            raise RuntimeError("capture capability attestation snapshot changed")

    def owns_completion(self, completion: CaptureCompletion) -> bool:
        with self._lock:
            return (
                isinstance(completion, CaptureCompletion)
                and completion._session is self
                and self._completion is completion
            )

    def _consume_completion_authority(
        self,
        completion: CaptureCompletion,
    ) -> None:
        """Single no-allocation commit that breaks the completed owner graph."""

        self._assert_owner_thread()
        with self._lock:
            if (
                not isinstance(completion, CaptureCompletion)
                or completion._session is not self
                or completion._terminal_reservation is None
                or self._completion is not completion
            ):
                raise RuntimeError("capture completion authority is absent or differs")
            self._completion = None
            self._exact_consumer_readiness = None
            self._source_event_span_hasher = None
            object.__setattr__(completion, "_session", None)
            object.__setattr__(completion, "_terminal_reservation", None)

    def _commit_pipeline_authority(
        self,
        completion: CaptureCompletion,
        dataset: SealedDatasetArtifact,
    ) -> None:
        """One owner-lane commit for all process-local result capabilities."""

        self._assert_owner_thread()
        if not isinstance(dataset, SealedDatasetArtifact):
            raise TypeError("dataset must be SealedDatasetArtifact")
        with self._lock:
            reservation = completion._terminal_reservation
            if (
                completion._session is not self
                or reservation is None
                or self._completion is not completion
                or not dataset._belongs_to_terminal_reservation(reservation)
            ):
                raise RuntimeError("pipeline result authority is absent or differs")
            # Final no-fail ownership commit.  All type, identity, and graph
            # checks precede these direct reference clears under the one
            # CaptureSession owner lock.
            object.__setattr__(dataset, "_terminal_reservation", None)
            self._completion = None
            self._exact_consumer_readiness = None
            self._source_event_span_hasher = None
            object.__setattr__(completion, "_session", None)
            object.__setattr__(completion, "_terminal_reservation", None)

    def _poison(self, error: StreamError) -> None:
        with self._lock:
            if self._state is CaptureSessionState.COMPLETED:
                raise RuntimeError("completed capture session cannot fail")
            self._state = CaptureSessionState.FAILED
        self._producer.fail(error)

    def _validate_ack(self, ack: object, expected_type: type) -> None:
        if not isinstance(ack, expected_type):
            raise TypeError(
                f"capture device returned {type(ack).__name__}, expected {expected_type.__name__}"
            )
        if ack.session_id != self._session_id:
            raise RuntimeError("capture acknowledgement session_id differs")
        if ack.binding_instance_id != self._port.device.binding_instance_id:
            raise RuntimeError("capture acknowledgement binding instance differs")

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("CaptureSession operation left its owner I/O lane")

def open_capture_session(
    port: BoundCapturePort,
    contract: CameraCaptureContract,
    trace_binding: TraceBinding,
    capture_spec: FrozenCaptureSpec,
) -> CaptureSession:
    """Bind physical camera authority to one exact application session."""

    if not isinstance(port, BoundCapturePort):
        raise TypeError("port must be BoundCapturePort")
    if contract.capability is not port.capability:
        raise ValueError("CameraCaptureContract must share BoundCapturePort capability")
    return CaptureSession(port, contract, trace_binding, capture_spec)


__all__ = [
    "CameraCaptureContract",
    "CameraCaptureProvenance",
    "CaptureCompletion",
    "CaptureProcessorInputBinding",
    "CaptureSession",
    "CaptureSessionState",
    "camera_capture_provenance_from_tree",
    "camera_capture_provenance_to_tree",
    "open_capture_session",
]
