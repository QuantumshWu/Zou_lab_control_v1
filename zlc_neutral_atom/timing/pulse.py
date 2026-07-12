"""Finite pulse execution as a generation-bound neutral-atom device Port."""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass
from enum import Enum

from zlc_pulse import (
    CompiledPulseArtifact,
    PulseDocument,
    PulseExecutionForm,
    PulseTarget,
    validate_target_ir_for_target,
)

from zlc_neutral_atom.runtime import (
    BoundDevice,
    ClaimMode,
    CleanupReport,
    HazardClaim,
    ResourceClaim,
    RunContext,
    SafetyInterrupt,
    SafetyOperation,
    VerifiedDeviceCapability,
)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_float(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class FinitePulseExecutionRequest:
    document: PulseDocument
    artifact: CompiledPulseArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(self.artifact, CompiledPulseArtifact):
            raise TypeError("artifact must be CompiledPulseArtifact")
        if self.artifact.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
            raise ValueError("finite pulse execution cannot use a continuous artifact")
        if self.artifact.source_document_digest != self.document.fingerprint:
            raise ValueError("compiled artifact belongs to another PulseDocument")
        if self.artifact.target_abi_fingerprint != self.document.target.abi_fingerprint:
            raise ValueError("compiled artifact target differs from PulseDocument")
        validate_target_ir_for_target(
            self.artifact.target_ir,
            self.document.target,
        )

    @property
    def artifact_digest(self) -> str:
        return self.artifact.fingerprint


@dataclass(frozen=True)
class SequencerCapabilitySnapshot:
    binding_id: str
    stable_device_identity: str
    connection_generation: str
    target: PulseTarget
    clock_hz: float
    geometry_fingerprint: int
    max_blocking_call_seconds: float
    capability_fingerprint: str

    def __post_init__(self) -> None:
        for field in ("binding_id", "stable_device_identity", "connection_generation"):
            _text(getattr(self, field), field)
        if not isinstance(self.target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        object.__setattr__(self, "clock_hz", _positive_float(self.clock_hz, "clock_hz"))
        if (
            isinstance(self.geometry_fingerprint, bool)
            or not isinstance(self.geometry_fingerprint, int)
            or not 0 <= self.geometry_fingerprint <= 0xFFFFFFFF
        ):
            raise ValueError("geometry_fingerprint must be an unsigned 32-bit integer")
        object.__setattr__(
            self,
            "max_blocking_call_seconds",
            _positive_float(self.max_blocking_call_seconds, "max_blocking_call_seconds"),
        )
        _sha256(self.capability_fingerprint, "capability_fingerprint")

    @property
    def target_abi_fingerprint(self) -> str:
        return self.target.abi_fingerprint


@dataclass(frozen=True)
class PreparePulseCommand:
    session_id: str
    run_id: str
    request: FinitePulseExecutionRequest
    capability_fingerprint: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")
        _text(self.run_id, "run_id")
        if not isinstance(self.request, FinitePulseExecutionRequest):
            raise TypeError("request must be FinitePulseExecutionRequest")
        _sha256(self.capability_fingerprint, "capability_fingerprint")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_float(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class PulsePreparedAck:
    session_id: str
    binding_id: str
    connection_generation: str
    artifact_digest: str
    capability_fingerprint: str

    def __post_init__(self) -> None:
        for field in ("session_id", "binding_id", "connection_generation"):
            _text(getattr(self, field), field)
        _sha256(self.artifact_digest, "artifact_digest")
        _sha256(self.capability_fingerprint, "capability_fingerprint")


@dataclass(frozen=True)
class FirePulseCommand:
    session_id: str
    artifact_digest: str

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")
        _sha256(self.artifact_digest, "artifact_digest")


@dataclass(frozen=True)
class PulseFiredAck:
    session_id: str
    binding_id: str
    connection_generation: str
    artifact_digest: str

    def __post_init__(self) -> None:
        for field in ("session_id", "binding_id", "connection_generation"):
            _text(getattr(self, field), field)
        _sha256(self.artifact_digest, "artifact_digest")


@dataclass(frozen=True)
class CompletePulseCommand:
    session_id: str
    artifact_digest: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")
        _sha256(self.artifact_digest, "artifact_digest")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_float(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class PulseTerminalAck:
    session_id: str
    binding_id: str
    connection_generation: str
    artifact_digest: str
    logical_done: bool
    completed_schedule_trigger_counts: tuple[tuple[str, int], ...]
    configured_output_delay_wait_seconds: float

    def __post_init__(self) -> None:
        for field in ("session_id", "binding_id", "connection_generation"):
            _text(getattr(self, field), field)
        _sha256(self.artifact_digest, "artifact_digest")
        if type(self.logical_done) is not bool:
            raise TypeError("logical_done must be bool")
        counts = tuple(self.completed_schedule_trigger_counts)
        seen: set[str] = set()
        for channel, count in counts:
            _text(channel, "trigger channel")
            if channel in seen:
                raise ValueError("completed trigger channels must be unique")
            seen.add(channel)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("completed trigger count must be non-negative int")
        object.__setattr__(self, "completed_schedule_trigger_counts", counts)
        wait = self.configured_output_delay_wait_seconds
        if (
            isinstance(wait, bool)
            or not isinstance(wait, (int, float))
            or not math.isfinite(float(wait))
            or float(wait) < 0
        ):
            raise ValueError("configured output-delay wait must be finite and non-negative")
        object.__setattr__(self, "configured_output_delay_wait_seconds", float(wait))


def pulse_terminal_ack_to_tree(value: PulseTerminalAck) -> dict[str, object]:
    if not isinstance(value, PulseTerminalAck):
        raise TypeError("value must be PulseTerminalAck")
    return {
        "session_id": value.session_id,
        "binding_id": value.binding_id,
        "connection_generation": value.connection_generation,
        "artifact_digest": value.artifact_digest,
        "logical_done": value.logical_done,
        "completed_schedule_trigger_counts": [
            [channel, count]
            for channel, count in value.completed_schedule_trigger_counts
        ],
        "configured_output_delay_wait_seconds": (
            value.configured_output_delay_wait_seconds
        ),
    }


def pulse_terminal_ack_from_tree(tree: object) -> PulseTerminalAck:
    fields = {
        "session_id",
        "binding_id",
        "connection_generation",
        "artifact_digest",
        "logical_done",
        "completed_schedule_trigger_counts",
        "configured_output_delay_wait_seconds",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PulseTerminalAck has an unknown field set")
    raw_counts = tree["completed_schedule_trigger_counts"]
    if not isinstance(raw_counts, list):
        raise TypeError("completed_schedule_trigger_counts must be a list")
    counts = []
    for item in raw_counts:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("trigger count must be [channel, count]")
        counts.append((item[0], item[1]))
    return PulseTerminalAck(
        tree["session_id"],
        tree["binding_id"],
        tree["connection_generation"],
        tree["artifact_digest"],
        tree["logical_done"],
        tuple(counts),
        tree["configured_output_delay_wait_seconds"],
    )


@dataclass(frozen=True)
class BoundPulsePort:
    capability_attestation: VerifiedDeviceCapability
    cleanup_operations: tuple[SafetyOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capability_attestation, VerifiedDeviceCapability):
            raise TypeError("pulse capability attestation must be broker-minted")
        if not isinstance(self.capability_attestation.snapshot, SequencerCapabilitySnapshot):
            raise TypeError("pulse capability attestation has the wrong snapshot type")
        if (
            self.device.validate_capability(self.capability_attestation)
            is not self.capability_attestation.snapshot
        ):
            raise RuntimeError("pulse capability attestation snapshot changed")
        capability = self.capability
        if (
            capability.binding_id != self.device.binding_id
            or capability.stable_device_identity != self.device.stable_device_identity
            or capability.connection_generation != self.device.connection_generation
        ):
            raise ValueError("pulse capability identity differs from BoundDevice")
        if not self.device.session_cleanup_capable:
            raise ValueError("pulse port requires session-specific cleanup")
        if not any(
            operation in self.device.interrupt_capabilities
            for operation in (SafetyOperation.ABORT, SafetyOperation.SAFE_STATE)
        ):
            raise ValueError("pulse port requires a thread-safe ABORT or SAFE_STATE interrupt")
        operations = tuple(self.cleanup_operations)
        if len(set(operations)) != len(operations):
            raise ValueError("pulse cleanup operations cannot contain duplicates")
        if any(
            operation not in (SafetyOperation.ABORT, SafetyOperation.SAFE_STATE)
            for operation in operations
        ):
            raise ValueError("pulse cleanup may only ABORT or enter SAFE_STATE")
        if any(operation not in self.device.safety_capabilities for operation in operations):
            raise ValueError("pulse cleanup operation is absent from BoundDevice")
        object.__setattr__(self, "cleanup_operations", operations)

    @property
    def device(self) -> BoundDevice:
        return self.capability_attestation.device

    @property
    def capability(self) -> SequencerCapabilitySnapshot:
        value = self.capability_attestation.snapshot
        assert isinstance(value, SequencerCapabilitySnapshot)
        return value

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
            for operation in (SafetyOperation.ABORT, SafetyOperation.SAFE_STATE)
            if operation in self.device.interrupt_capabilities
        )
        return tuple(SafetyInterrupt(self.device.key, operation) for operation in preferred)

    def open_session(self, request: FinitePulseExecutionRequest) -> "PulseSession":
        if request.artifact.target_abi_fingerprint != self.capability.target_abi_fingerprint:
            raise ValueError("pulse request target differs from live sequencer")
        if request.artifact.wire_image.geometry_fingerprint != self.capability.geometry_fingerprint:
            raise ValueError("pulse request wire geometry differs from live sequencer")
        return PulseSession(self, request)

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
                raise RuntimeError("sequencer session safe/terminal acknowledgement failed")
        except BaseException as error:
            errors.append(error)
        if errors:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="sequencer cleanup did not reach a verified safe terminal",
                recovery_action="verify sequencer output state and recover the connection",
                errors=tuple(errors),
            )
        try:
            proof = device.verify_safe_state()
        except BaseException as error:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="sequencer safe-state verification failed",
                recovery_action="inspect outputs and recover the sequencer before reuse",
                errors=(error,),
            )
        return CleanupReport.safe((proof,))

    def verify_idle(self, context: RunContext) -> CleanupReport:
        try:
            proof = context.cleanup_device(self.device.key).verify_safe_state()
        except BaseException as error:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="unopened sequencer safe-state verification failed",
                recovery_action="inspect and recover the sequencer before reuse",
                errors=(error,),
            )
        return CleanupReport.safe((proof,))


class PulseSessionState(str, Enum):
    NEW = "NEW"
    PREPARED = "PREPARED"
    FIRED = "FIRED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PulseSession:
    """Single-thread owner of one finite prepared pulse and terminal receipt."""

    def __init__(self, port: BoundPulsePort, request: FinitePulseExecutionRequest) -> None:
        self._port = port
        self._request = request
        self._session_id = uuid.uuid4().hex
        self._state = PulseSessionState.NEW
        self._owner_thread_id = threading.get_ident()
        self._hardware_prepare_attempted = False
        self._terminal: PulseTerminalAck | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> PulseSessionState:
        return self._state

    @property
    def terminal(self) -> PulseTerminalAck:
        if self._terminal is None:
            raise RuntimeError("pulse session has no terminal acknowledgement")
        return self._terminal

    def prepare(self, context: RunContext) -> None:
        self._assert_owner_thread()
        if self._state is not PulseSessionState.NEW:
            raise RuntimeError("pulse session can only be prepared once")
        self._validate_capability()
        self._hardware_prepare_attempted = True
        try:
            ack = context.device(self._port.device.key).execute(
                PreparePulseCommand(
                    self._session_id,
                    context.run_id.value,
                    self._request,
                    self._port.capability.capability_fingerprint,
                    self._port.capability.max_blocking_call_seconds,
                )
            )
            self._validate_ack(ack, PulsePreparedAck)
            if (
                ack.artifact_digest != self._request.artifact_digest
                or ack.capability_fingerprint != self._port.capability.capability_fingerprint
            ):
                raise RuntimeError("pulse prepare acknowledgement differs from frozen request")
        except BaseException:
            self._state = PulseSessionState.FAILED
            raise
        self._state = PulseSessionState.PREPARED

    def fire(self, context: RunContext) -> None:
        self._assert_owner_thread()
        if self._state is not PulseSessionState.PREPARED:
            raise RuntimeError("pulse session must be prepared before FIRE")
        try:
            ack = context.device(self._port.device.key).execute(
                FirePulseCommand(self._session_id, self._request.artifact_digest)
            )
            self._validate_ack(ack, PulseFiredAck)
            if ack.artifact_digest != self._request.artifact_digest:
                raise RuntimeError("FIRE acknowledgement artifact differs")
        except BaseException:
            self._state = PulseSessionState.FAILED
            raise
        self._state = PulseSessionState.FIRED

    def complete(self, context: RunContext) -> PulseTerminalAck:
        self._assert_owner_thread()
        if self._state is not PulseSessionState.FIRED:
            raise RuntimeError("pulse session must be fired before completion")
        try:
            ack = context.device(self._port.device.key).execute(
                CompletePulseCommand(
                    self._session_id,
                    self._request.artifact_digest,
                    self._port.capability.max_blocking_call_seconds,
                )
            )
            self._validate_ack(ack, PulseTerminalAck)
            if not ack.logical_done or ack.artifact_digest != self._request.artifact_digest:
                raise RuntimeError("pulse terminal acknowledgement is incomplete")
            expected = tuple(
                (schedule.channel, schedule.total)
                for schedule in self._request.artifact.trigger_schedules
            )
            if ack.completed_schedule_trigger_counts != expected:
                raise RuntimeError("pulse terminal trigger counts differ from compiled schedule")
        except BaseException:
            self._state = PulseSessionState.FAILED
            raise
        self._terminal = ack
        self._state = PulseSessionState.COMPLETED
        return ack

    def cleanup(self, context: RunContext) -> CleanupReport:
        self._assert_owner_thread()
        return (
            self._port.cleanup(context, self._session_id)
            if self._hardware_prepare_attempted
            else self._port.verify_idle(context)
        )

    def fail(self) -> None:
        self._assert_owner_thread()
        if self._state is not PulseSessionState.COMPLETED:
            self._state = PulseSessionState.FAILED

    def _validate_capability(self) -> None:
        if (
            self._port.device.validate_capability(self._port.capability_attestation)
            is not self._port.capability
        ):
            raise RuntimeError("pulse capability attestation snapshot changed")

    def _validate_ack(self, ack: object, expected_type: type) -> None:
        if not isinstance(ack, expected_type):
            raise TypeError(
                f"sequencer returned {type(ack).__name__}, expected {expected_type.__name__}"
            )
        if ack.session_id != self._session_id:
            raise RuntimeError("pulse acknowledgement session_id differs")
        if ack.binding_id != self._port.device.binding_id:
            raise RuntimeError("pulse acknowledgement binding_id differs")
        if ack.connection_generation != self._port.device.connection_generation:
            raise RuntimeError("pulse acknowledgement generation differs")

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("PulseSession operation left its owner I/O lane")


__all__ = [
    "BoundPulsePort",
    "CompletePulseCommand",
    "FinitePulseExecutionRequest",
    "FirePulseCommand",
    "PreparePulseCommand",
    "PulseFiredAck",
    "PulsePreparedAck",
    "PulseSession",
    "PulseSessionState",
    "PulseTerminalAck",
    "pulse_terminal_ack_from_tree",
    "pulse_terminal_ack_to_tree",
    "SequencerCapabilitySnapshot",
]
