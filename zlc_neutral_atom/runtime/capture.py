"""Opaque capture-port/session authority over typed DeviceBroker commands."""

from __future__ import annotations

import hashlib
import math
import threading
import uuid
from dataclasses import dataclass, is_dataclass
from enum import Enum
from numbers import Integral
from typing import Protocol, TypeVar

from zlc_data import DatasetSchema, ValuePayloadContract
from zlc_neutral_atom.catalog import is_declarative_value

from .dataset import (
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetEventAdapter,
    ExactDatasetReadiness,
)
from .ports import (
    BoundDevice,
    SafetyInterrupt,
    SafetyOperation,
    VerifiedDeviceCapability,
)
from .resources import ClaimMode, HazardClaim, ResourceClaim
from .run import CleanupReport, RunContext
from .streams import (
    AcquisitionProducer,
    AcquisitionStream,
    EndOfStream,
    ExactReservation,
    ProducerFlowControl,
    ReservationState,
    SourceFailed,
    StreamError,
    StreamId,
    TraceBinding,
    TraceContext,
)


PayloadT = TypeVar("PayloadT")
_COMPLETION_TOKEN = object()


def _canonical_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _positive_int(value: int, field: str) -> int:
    value = _nonnegative_int(value, field)
    if value == 0:
        raise ValueError(f"{field} must be positive")
    return value


def _positive_finite(value: float, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


class CapturePayloadContract(Protocol[PayloadT]):
    fingerprint: str
    max_retained_nbytes: int

    def snapshot(self, payload: PayloadT) -> PayloadT: ...

    def validate(self, payload: PayloadT) -> None: ...

    def retained_nbytes(self, payload: PayloadT) -> int: ...

    def source_ordinal(self, payload: PayloadT) -> int: ...

    def captured_at(self, payload: PayloadT) -> float: ...

    def correlation_id(self, payload: PayloadT) -> str: ...


@dataclass(frozen=True)
class FrozenCaptureSpec:
    """Canonical owner bytes; runtime never executes an arbitrary spec codec."""

    owner_fingerprint: str
    payload: bytes
    digest: str = ""

    def __post_init__(self) -> None:
        _sha256(self.owner_fingerprint, "capture spec owner fingerprint")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("capture spec payload must be non-empty immutable bytes")
        payload = bytes(self.payload)
        digest = hashlib.sha256(payload).hexdigest()
        if self.digest and self.digest != digest:
            raise ValueError("capture spec digest differs from canonical payload")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "digest", digest)


@dataclass(frozen=True)
class CaptureCapabilitySnapshot:
    binding_id: str
    stable_device_identity: str
    connection_generation: str
    capability_fingerprint: str
    settings_fingerprint: str
    payload_contract_fingerprint: str
    capture_spec_owner_fingerprint: str
    flow_control: ProducerFlowControl
    max_source_burst_events: int
    driver_ring_bytes: int
    max_blocking_call_seconds: float
    max_capture_spec_bytes: int

    def __post_init__(self) -> None:
        for field in (
            "binding_id",
            "stable_device_identity",
            "connection_generation",
        ):
            _canonical_text(getattr(self, field), field)
        for field in (
            "capability_fingerprint",
            "settings_fingerprint",
            "payload_contract_fingerprint",
            "capture_spec_owner_fingerprint",
        ):
            _sha256(getattr(self, field), field)
        if not isinstance(self.flow_control, ProducerFlowControl):
            raise TypeError("flow_control must be ProducerFlowControl")
        object.__setattr__(
            self,
            "max_source_burst_events",
            _positive_int(self.max_source_burst_events, "max_source_burst_events"),
        )
        object.__setattr__(
            self,
            "driver_ring_bytes",
            _positive_int(self.driver_ring_bytes, "driver_ring_bytes"),
        )
        object.__setattr__(
            self,
            "max_capture_spec_bytes",
            _positive_int(self.max_capture_spec_bytes, "max_capture_spec_bytes"),
        )
        object.__setattr__(
            self,
            "max_blocking_call_seconds",
            _positive_finite(
                self.max_blocking_call_seconds,
                "max_blocking_call_seconds",
            ),
        )


@dataclass(frozen=True)
class CaptureRuntimeProfile:
    required_consumer_lag_events: int
    transport_memory_limit_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_consumer_lag_events",
            _nonnegative_int(
                self.required_consumer_lag_events,
                "required_consumer_lag_events",
            ),
        )
        object.__setattr__(
            self,
            "transport_memory_limit_bytes",
            _positive_int(
                self.transport_memory_limit_bytes,
                "transport_memory_limit_bytes",
            ),
        )


@dataclass(frozen=True)
class CaptureStreamContract:
    stream_id: StreamId
    source_id: str
    dataset_schema: DatasetSchema
    payload_contract: CapturePayloadContract
    event_adapter: DatasetEventAdapter
    expected_cells: tuple[DatasetCellAddress, ...]
    capability: CaptureCapabilitySnapshot
    runtime_profile: CaptureRuntimeProfile
    capture_spec_owner_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        _canonical_text(self.source_id, "source_id")
        if not isinstance(self.dataset_schema, DatasetSchema):
            raise TypeError("dataset_schema must be DatasetSchema")
        if not isinstance(self.capability, CaptureCapabilitySnapshot):
            raise TypeError("capability must be CaptureCapabilitySnapshot")
        if not isinstance(self.runtime_profile, CaptureRuntimeProfile):
            raise TypeError("runtime_profile must be CaptureRuntimeProfile")
        for member in ("payload_contract", "value_schema", "metadata_contract", "value"):
            if not hasattr(self.event_adapter, member):
                raise TypeError(f"event_adapter.{member} is required")
        _sha256(self.payload_contract.fingerprint, "payload contract fingerprint")
        _sha256(
            self.capture_spec_owner_fingerprint,
            "capture spec owner fingerprint",
        )
        for name, owner in (
            ("payload_contract", self.payload_contract),
            ("event_adapter", self.event_adapter),
            ("metadata_contract", self.event_adapter.metadata_contract),
        ):
            parameters = getattr(type(owner), "__dataclass_params__", None)
            if not is_dataclass(owner) or not parameters or not parameters.frozen:
                raise TypeError(f"{name} must be a frozen dataclass value")
            if not is_declarative_value(owner):
                raise TypeError(f"{name} fields must be recursively declarative data")
        for member in (
            "snapshot",
            "validate",
            "retained_nbytes",
            "source_ordinal",
            "captured_at",
            "correlation_id",
        ):
            if not callable(getattr(self.payload_contract, member, None)):
                raise TypeError(f"payload_contract.{member} must be callable")
        _positive_int(
            self.payload_contract.max_retained_nbytes,
            "payload contract max_retained_nbytes",
        )
        metadata_contract = self.event_adapter.metadata_contract
        _sha256(metadata_contract.fingerprint, "metadata contract fingerprint")
        _nonnegative_int(
            metadata_contract.max_retained_nbytes,
            "metadata contract max_retained_nbytes",
        )
        for member in ("snapshot", "validate", "retained_nbytes", "digest"):
            if not callable(getattr(metadata_contract, member, None)):
                raise TypeError(f"metadata_contract.{member} must be callable")
        value_bytes = ValuePayloadContract(
            self.dataset_schema.cell_schema
        ).max_retained_nbytes
        if value_bytes + metadata_contract.max_retained_nbytes > (
            self.payload_contract.max_retained_nbytes
        ):
            raise ValueError(
                "payload retained-byte bound must cover value plus metadata"
            )
        if self.capability.payload_contract_fingerprint != self.payload_contract.fingerprint:
            raise ValueError("capture capability and payload contract fingerprints differ")
        if (
            self.capability.capture_spec_owner_fingerprint
            != self.capture_spec_owner_fingerprint
        ):
            raise ValueError("capture capability and spec owner fingerprints differ")
        if self.event_adapter.payload_contract is not self.payload_contract:
            raise ValueError("DatasetEventAdapter must share CapturePayloadContract owner")
        if self.event_adapter.value_schema is not self.dataset_schema.cell_schema:
            raise ValueError("DatasetEventAdapter schema differs from DatasetSchema")
        cells = tuple(self.expected_cells)
        domain = {
            DatasetCellAddress(repeat, point)
            for repeat in range(self.dataset_schema.repeat_axis.size)
            for point in range(self.dataset_schema.point_layout.storage_size)
        }
        if len(cells) != len(domain) or set(cells) != domain:
            raise ValueError("expected_cells must be a complete unique dataset permutation")
        object.__setattr__(self, "expected_cells", cells)
        if self.estimated_transport_bytes > self.runtime_profile.transport_memory_limit_bytes:
            raise MemoryError(
                f"capture transport budget {self.estimated_transport_bytes} exceeds "
                f"limit {self.runtime_profile.transport_memory_limit_bytes}"
            )

    @property
    def total_events(self) -> int:
        return len(self.expected_cells)

    @property
    def max_inflight_events(self) -> int:
        return min(
            self.total_events,
            self.capability.max_source_burst_events
            + self.runtime_profile.required_consumer_lag_events,
        )

    @property
    def max_inflight_bytes(self) -> int:
        return self.max_inflight_events * int(self.payload_contract.max_retained_nbytes)

    @property
    def estimated_transport_bytes(self) -> int:
        return (
            self.capability.driver_ring_bytes
            + self.max_inflight_bytes
            + self.payload_contract.max_retained_nbytes
            + self.event_adapter.metadata_contract.max_retained_nbytes
            + self.capability.max_capture_spec_bytes
        )


@dataclass(frozen=True)
class PrepareCaptureCommand:
    session_id: str
    run_id: str
    source_id: str
    capture_spec_payload: bytes
    capture_spec_owner_fingerprint: str
    capture_spec_fingerprint: str
    capability_fingerprint: str
    settings_fingerprint: str
    expected_total_events: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        for name in ("session_id", "run_id", "source_id"):
            _canonical_text(getattr(self, name), name)
        for name in (
            "capture_spec_owner_fingerprint",
            "capture_spec_fingerprint",
            "capability_fingerprint",
            "settings_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        if not isinstance(self.capture_spec_payload, bytes) or not self.capture_spec_payload:
            raise ValueError("capture_spec_payload must be non-empty bytes")
        if hashlib.sha256(self.capture_spec_payload).hexdigest() != self.capture_spec_fingerprint:
            raise ValueError("capture spec payload digest differs")
        object.__setattr__(
            self,
            "expected_total_events",
            _positive_int(self.expected_total_events, "expected_total_events"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_finite(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CapturePreparedAck:
    session_id: str
    binding_id: str
    connection_generation: str
    settings_fingerprint: str
    capability_fingerprint: str
    capture_spec_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_id", "connection_generation"):
            _canonical_text(getattr(self, name), name)
        _sha256(self.settings_fingerprint, "settings_fingerprint")
        _sha256(self.capability_fingerprint, "capability_fingerprint")
        _sha256(self.capture_spec_fingerprint, "capture_spec_fingerprint")


@dataclass(frozen=True)
class StartCaptureCommand:
    session_id: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_finite(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CaptureStartedAck:
    session_id: str
    binding_id: str
    connection_generation: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_id", "connection_generation"):
            _canonical_text(getattr(self, name), name)


@dataclass(frozen=True)
class ReadCaptureCommand:
    session_id: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_finite(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CapturedPayloadAck:
    session_id: str
    binding_id: str
    connection_generation: str
    payload: object

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_id", "connection_generation"):
            _canonical_text(getattr(self, name), name)


@dataclass(frozen=True)
class CompleteCaptureCommand:
    session_id: str
    expected_total_events: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "expected_total_events",
            _positive_int(self.expected_total_events, "expected_total_events"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_finite(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CaptureTerminalAck:
    session_id: str
    binding_id: str
    connection_generation: str
    produced_count: int
    drained_count: int
    source_stopped: bool
    no_more_frames: bool
    joined: bool
    ordered_metadata_digest: str
    settings_fingerprint: str
    capability_fingerprint: str
    capture_spec_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_id", "connection_generation"):
            _canonical_text(getattr(self, name), name)
        object.__setattr__(
            self,
            "produced_count",
            _nonnegative_int(self.produced_count, "produced_count"),
        )
        object.__setattr__(
            self,
            "drained_count",
            _nonnegative_int(self.drained_count, "drained_count"),
        )
        for name in ("source_stopped", "no_more_frames", "joined"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        _sha256(self.ordered_metadata_digest, "ordered_metadata_digest")
        _sha256(self.settings_fingerprint, "settings_fingerprint")
        _sha256(self.capability_fingerprint, "capability_fingerprint")
        _sha256(self.capture_spec_fingerprint, "capture_spec_fingerprint")


class CaptureSessionState(str, Enum):
    NEW = "NEW"
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    STARTING = "STARTING"
    STARTED = "STARTED"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CaptureCompletion:
    __slots__ = ("_session", "_eos", "_terminal")

    def __init__(
        self,
        authority: object,
        *,
        session: "CaptureSession",
        terminal: CaptureTerminalAck,
    ) -> None:
        if authority is not _COMPLETION_TOKEN:
            raise PermissionError("CaptureCompletion can only be minted by CaptureSession")
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_eos", None)
        object.__setattr__(self, "_terminal", terminal)

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


@dataclass(frozen=True)
class BoundCapturePort:
    capability_attestation: VerifiedDeviceCapability
    cleanup_operations: tuple[SafetyOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capability_attestation, VerifiedDeviceCapability):
            raise TypeError("capability_attestation must be broker-minted")
        if not isinstance(self.capability_attestation.device, BoundDevice):
            raise TypeError("capture capability has no BoundDevice")
        if not isinstance(
            self.capability_attestation.snapshot,
            CaptureCapabilitySnapshot,
        ):
            raise TypeError("capture capability attestation has the wrong snapshot type")
        if (
            self.capability_attestation.device.validate_capability(
                self.capability_attestation
            )
            is not self.capability_attestation.snapshot
        ):
            raise RuntimeError("capture capability attestation snapshot changed")
        if self.capability.binding_id != self.device.binding_id:
            raise ValueError("capture capability binding_id differs from BoundDevice")
        if self.capability.stable_device_identity != self.device.stable_device_identity:
            raise ValueError("capture capability stable identity differs from BoundDevice")
        if self.capability.connection_generation != self.device.connection_generation:
            raise ValueError("capture capability generation differs from BoundDevice")
        if not self.device.session_cleanup_capable:
            raise ValueError("capture port requires session-specific cleanup capability")
        if not any(
            operation in self.device.interrupt_capabilities
            for operation in (SafetyOperation.ABORT, SafetyOperation.DISARM)
        ):
            raise ValueError("capture port requires a thread-safe ABORT or DISARM interrupt")
        operations = tuple(self.cleanup_operations)
        if len(set(operations)) != len(operations):
            raise ValueError("capture cleanup operations cannot contain duplicates")
        if any(
            operation not in (SafetyOperation.ABORT, SafetyOperation.DISARM)
            for operation in operations
        ):
            raise ValueError("capture cleanup pre-steps may only ABORT or DISARM")
        if any(operation not in self.device.safety_capabilities for operation in operations):
            raise ValueError("capture cleanup operation is absent from BoundDevice")
        object.__setattr__(self, "cleanup_operations", operations)

    @property
    def device(self) -> BoundDevice:
        return self.capability_attestation.device

    @property
    def capability(self) -> CaptureCapabilitySnapshot:
        snapshot = self.capability_attestation.snapshot
        assert isinstance(snapshot, CaptureCapabilitySnapshot)
        return snapshot

    @property
    def resource_claim(self) -> ResourceClaim:
        return ResourceClaim(self.device.key, ClaimMode.EXCLUSIVE)

    @property
    def hazard_claim(self) -> HazardClaim:
        return HazardClaim(
            self.device.key,
            self.device.stable_device_identity,
            self.device.connection_generation,
        )

    @property
    def interrupt_operations(self) -> tuple[SafetyInterrupt, ...]:
        preferred = tuple(
            operation
            for operation in (SafetyOperation.ABORT, SafetyOperation.DISARM)
            if operation in self.device.interrupt_capabilities
        )
        return tuple(SafetyInterrupt(self.device.key, operation) for operation in preferred)

    def open_session(
        self,
        contract: CaptureStreamContract,
        trace_binding: TraceBinding,
        capture_spec: FrozenCaptureSpec,
    ) -> "CaptureSession":
        if contract.capability is not self.capability:
            raise ValueError("CaptureStreamContract must share BoundCapturePort capability")
        return CaptureSession(self, contract, trace_binding, capture_spec)

    def cleanup(self, context: RunContext, session_id: str) -> CleanupReport:
        errors: list[BaseException] = []
        device = context.cleanup_device(self.device.key)
        for operation in self.cleanup_operations:
            try:
                device.perform(operation)
            except BaseException as error:
                errors.append(error)
        try:
            closed = device.close_session(
                session_id,
                self.capability.max_blocking_call_seconds,
            )
            if not (closed.source_stopped and closed.no_more_work and closed.joined):
                raise RuntimeError("capture session stop/drain/join acknowledgement failed")
        except BaseException as error:
            errors.append(error)
        if errors:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="capture cleanup did not acknowledge every required operation",
                recovery_action="verify camera session termination and physical safe state",
                errors=tuple(errors),
            )
        try:
            proof = device.verify_safe_state()
        except BaseException as error:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="capture device safe-state verification failed",
                recovery_action="inspect and recover the camera before reuse",
                errors=(error,),
            )
        return CleanupReport.safe((proof,))

    def verify_idle(self, context: RunContext) -> CleanupReport:
        """Verify safety when no physical capture prepare was ever attempted."""

        try:
            proof = context.cleanup_device(self.device.key).verify_safe_state()
        except BaseException as error:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="unopened capture device safe-state verification failed",
                recovery_action="inspect and recover the camera before reuse",
                errors=(error,),
            )
        return CleanupReport.safe((proof,))

class CaptureSession:
    """One owner of producer, device session id, ordinal, and terminal receipt."""

    def __init__(
        self,
        port: BoundCapturePort,
        contract: CaptureStreamContract,
        trace_binding: TraceBinding,
        capture_spec: FrozenCaptureSpec,
    ) -> None:
        if not isinstance(trace_binding, TraceBinding):
            raise TypeError("trace_binding must be TraceBinding")
        if trace_binding.source_id != contract.source_id:
            raise ValueError("TraceBinding source differs from CaptureStreamContract")
        if not isinstance(capture_spec, FrozenCaptureSpec):
            raise TypeError("capture_spec must be FrozenCaptureSpec")
        if capture_spec.owner_fingerprint != contract.capture_spec_owner_fingerprint:
            raise ValueError("capture spec owner differs from CaptureStreamContract")
        if len(capture_spec.payload) > contract.capability.max_capture_spec_bytes:
            raise MemoryError("capture spec exceeds device capability byte bound")
        self._port = port
        self._contract = contract
        self._trace_binding = trace_binding
        self._capture_spec = capture_spec
        self._capture_spec_digest = capture_spec.digest
        self._session_id = uuid.uuid4().hex
        stream, producer = AcquisitionStream.create(
            contract.stream_id,
            contract.payload_contract,
            flow_control=contract.capability.flow_control,
            retention_events=contract.max_inflight_events,
            retention_bytes=contract.max_inflight_bytes,
            join_key_contract=DatasetCellKeyContract(contract.dataset_schema),
        )
        self._stream = stream
        self._producer: AcquisitionProducer = producer
        self._state = CaptureSessionState.NEW
        self._delivered = 0
        self._metadata_hasher = hashlib.sha256()
        self._metadata_hasher.update(
            contract.event_adapter.metadata_contract.fingerprint.encode("ascii")
        )
        self._completion: CaptureCompletion | None = None
        self._reservation: ExactReservation | None = None
        self._materializer_readiness: ExactDatasetReadiness | None = None
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._owner_thread_id = threading.get_ident()
        self._hardware_prepare_attempted = False

    @property
    def stream(self) -> AcquisitionStream:
        return self._stream

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
                max_inflight_events=self._contract.max_inflight_events,
                max_inflight_bytes=self._contract.max_inflight_bytes,
                trace_binding=self._trace_binding,
            )
            with self._lock:
                self._reservation = reservation
            return reservation

    def bind_materializer(self, readiness: ExactDatasetReadiness) -> None:
        """Bind the one owner-minted exact materializer proof before prepare."""

        with self._operation_lock:
            self._assert_owner_thread()
            if not isinstance(readiness, ExactDatasetReadiness):
                raise TypeError("readiness must be ExactDatasetReadiness")
            with self._lock:
                if self._state is not CaptureSessionState.NEW:
                    raise RuntimeError("materializer readiness must precede capture prepare")
                reservation = self._reservation
                if reservation is None:
                    raise RuntimeError("capture session has no exact reservation")
                if self._materializer_readiness is not None:
                    raise RuntimeError("capture session already has materializer readiness")
            readiness.validate(
                reservation=reservation,
                schema=self._contract.dataset_schema,
                event_adapter=self._contract.event_adapter,
                expected_cells=self._contract.expected_cells,
            )
            with self._lock:
                self._materializer_readiness = readiness

    def prepare(self, context: RunContext) -> None:
        with self._operation_lock:
            self._assert_owner_thread()
            if context.run_id.value != self._trace_binding.run_id:
                raise ValueError("RunContext differs from the frozen TraceBinding")
            self._validate_current_capability()
            with self._lock:
                if self._state is not CaptureSessionState.NEW:
                    raise RuntimeError("capture session can only be prepared once")
                self._hardware_prepare_attempted = True
                self._state = CaptureSessionState.PREPARING
            try:
                ack = context.device(self._port.device.key).execute(
                    PrepareCaptureCommand(
                        session_id=self._session_id,
                        run_id=context.run_id.value,
                        source_id=self._contract.source_id,
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
                self._poison(SourceFailed(f"capture prepare failed: {error}"))
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
                readiness = self._materializer_readiness
                if readiness is None:
                    raise RuntimeError("capture has no exact materializer readiness proof")
                readiness.validate(
                    reservation=reservation,
                    schema=self._contract.dataset_schema,
                    event_adapter=self._contract.event_adapter,
                    expected_cells=self._contract.expected_cells,
                )
                self._state = CaptureSessionState.STARTING
            try:
                ack = context.device(self._port.device.key).execute(
                    StartCaptureCommand(
                        self._session_id,
                        self._port.capability.max_blocking_call_seconds,
                    )
                )
                self._validate_ack(ack, CaptureStartedAck)
            except BaseException as error:
                self._poison(SourceFailed(f"capture start failed: {error}"))
                raise
            with self._lock:
                self._state = CaptureSessionState.STARTED

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
                    raise RuntimeError("capture already delivered its frozen event budget")
                join_key = self._contract.expected_cells[expected_ordinal]
            try:
                ack = context.device(self._port.device.key).execute(
                    ReadCaptureCommand(
                        self._session_id,
                        self._port.capability.max_blocking_call_seconds,
                    )
                )
                self._validate_ack(ack, CapturedPayloadAck)
                payload_contract = self._contract.payload_contract
                payload = payload_contract.snapshot(ack.payload)
                payload_contract.validate(payload)
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
                metadata_contract = self._contract.event_adapter.metadata_contract
                metadata = metadata_contract.snapshot(stored)
                metadata_contract.validate(metadata)
                metadata_digest = _sha256(
                    metadata_contract.digest(metadata),
                    "payload metadata digest",
                )
            except BaseException as error:
                self._poison(
                    SourceFailed(f"captured payload failed validation/publish: {error}")
                )
                raise
            with self._lock:
                self._metadata_hasher.update(metadata_digest.encode("ascii"))
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
                self._state = CaptureSessionState.COMPLETING
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
                expected_digest = self._metadata_hasher.copy().hexdigest()
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
                self._poison(SourceFailed(f"capture terminal validation failed: {error}"))
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
                f"capture session aborted: {type(error).__name__}: {error}"
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
                    if not reservation.materializer_bound:
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
                            "DatasetBuilder did not terminate its exact reservation"
                        )
                except BaseException as error:
                    release_errors.append(error)
            if port_error is not None:
                for error in release_errors:
                    if hasattr(port_error, "add_note"):
                        port_error.add_note(
                            f"exact reservation teardown also failed: {error!r}"
                        )
                raise port_error
            assert report is not None
            if not release_errors:
                return report
            return CleanupReport(
                safety_proofs=report.safety_proofs,
                decisions=report.decisions,
                errors=(*report.errors, *release_errors),
            )

    def _validate_current_capability(self) -> None:
        if (
            self._port.device.validate_capability(
                self._port.capability_attestation
            )
            is not self._port.capability
        ):
            raise RuntimeError("capture capability attestation snapshot changed")

    def owns_completion(self, completion: CaptureCompletion) -> bool:
        with self._lock:
            return (
                isinstance(completion, CaptureCompletion)
                and completion._session is self
                and self._completion is completion
            )

    def _poison(self, error: StreamError) -> None:
        with self._lock:
            if self._state is CaptureSessionState.COMPLETED:
                raise RuntimeError("completed capture session cannot fail")
            self._state = CaptureSessionState.FAILED
        try:
            self._producer.fail(error)
        except BaseException:
            pass

    def _validate_ack(self, ack: object, expected_type: type) -> None:
        if not isinstance(ack, expected_type):
            raise TypeError(
                f"capture device returned {type(ack).__name__}, expected {expected_type.__name__}"
            )
        if ack.session_id != self._session_id:
            raise RuntimeError("capture acknowledgement session_id differs")
        if ack.binding_id != self._port.device.binding_id:
            raise RuntimeError("capture acknowledgement binding_id differs")
        if ack.connection_generation != self._port.device.connection_generation:
            raise RuntimeError("capture acknowledgement generation differs")

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("CaptureSession operation left its owner I/O lane")

__all__ = [
    "BoundCapturePort",
    "CaptureCapabilitySnapshot",
    "CaptureCompletion",
    "CapturePayloadContract",
    "CapturePreparedAck",
    "CaptureRuntimeProfile",
    "FrozenCaptureSpec",
    "CaptureSession",
    "CaptureSessionState",
    "CaptureStartedAck",
    "CaptureStreamContract",
    "CaptureTerminalAck",
    "CapturedPayloadAck",
    "CompleteCaptureCommand",
    "PrepareCaptureCommand",
    "ReadCaptureCommand",
    "StartCaptureCommand",
]
