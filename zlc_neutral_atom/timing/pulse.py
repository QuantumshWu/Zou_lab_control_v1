"""Finite pulse execution as a generation-bound neutral-atom device Port."""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from zlc_storage import (
    canonical_text as _text,
    positive_real as _positive_float,
    sha256_text as _sha256,
)

from zlc_pulse import (
    CompiledPulseArtifact,
    PulseBackendCompletion,
    PulseCompletion,
    PulseDocument,
    PulseExecutionForm,
    PulseTarget,
    build_pulse_playback,
    pulse_completion_from_tree,
    pulse_completion_to_tree,
    validate_backend_completion_for_artifact,
    validate_target_ir_for_target,
)

from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SafetyInterrupt,
    SafetyOperation,
    VerifiedDeviceCapability,
    admit_bound_capability,
    cleanup_device_session,
    verify_cleanup_device_safe_state,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.resources import (
    ClaimMode,
    DeviceBindingStamp,
    ResourceClaim,
)
from zlc_neutral_atom.runtime.run import RunContext


_PULSE_TERMINAL_ACK_SCHEMA = "zlc_neutral_atom.PulseTerminalAck"


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
        repeat = self.document.repeat
        expected_loop_count = 1 if repeat is None else repeat.count
        expected_full_point_loop = repeat is None or (
            repeat.start_period_id == self.document.periods[0].period_id
            and repeat.end_period_id == self.document.periods[-1].period_id
        )
        if any(
            schedule.loop_count != expected_loop_count
            or schedule.full_point_loop is not expected_full_point_loop
            for schedule in self.artifact.trigger_schedules
        ):
            raise ValueError(
                "compiled trigger execution groups differ from PulseDocument"
            )
        validate_target_ir_for_target(
            self.artifact.target_ir,
            self.document.target,
        )

    @property
    def artifact_digest(self) -> str:
        return self.artifact.fingerprint


class PulseTerminalEvidenceKind(str, Enum):
    HARDWARE_RAW_REGISTERS = "HARDWARE_RAW_REGISTERS"
    SIMULATED = "SIMULATED"


@dataclass(frozen=True)
class SequencerCapabilitySnapshot:
    binding_stamp: DeviceBindingStamp
    target: PulseTarget
    clock_hz: float
    geometry_fingerprint: int
    resident_scan_point_capacity: int
    max_blocking_call_seconds: float
    terminal_evidence_kind: "PulseTerminalEvidenceKind"
    server_connection_generation: str | None
    capability_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("binding_stamp must be DeviceBindingStamp")
        if not isinstance(self.target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        object.__setattr__(self, "clock_hz", _positive_float(self.clock_hz, "clock_hz"))
        if (
            isinstance(self.geometry_fingerprint, bool)
            or not isinstance(self.geometry_fingerprint, int)
            or not 0 <= self.geometry_fingerprint <= 0xFFFFFFFF
        ):
            raise ValueError("geometry_fingerprint must be an unsigned 32-bit integer")
        if (
            isinstance(self.resident_scan_point_capacity, bool)
            or not isinstance(self.resident_scan_point_capacity, int)
            or self.resident_scan_point_capacity < 1
        ):
            raise ValueError(
                "resident_scan_point_capacity must be a positive integer"
            )
        object.__setattr__(
            self,
            "max_blocking_call_seconds",
            _positive_float(self.max_blocking_call_seconds, "max_blocking_call_seconds"),
        )
        if not isinstance(self.terminal_evidence_kind, PulseTerminalEvidenceKind):
            raise TypeError("terminal_evidence_kind must be PulseTerminalEvidenceKind")
        if self.server_connection_generation is not None:
            _text(
                self.server_connection_generation,
                "server_connection_generation",
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
    binding_instance_id: str
    artifact_digest: str
    capability_fingerprint: str

    def __post_init__(self) -> None:
        for field in ("session_id", "binding_instance_id"):
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
    binding_instance_id: str
    artifact_digest: str

    def __post_init__(self) -> None:
        for field in ("session_id", "binding_instance_id"):
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
class SimulatedPulseReceipt:
    """Honest virtual completion that cannot impersonate hardware registers."""

    artifact_digest: str
    simulator_id: str
    expected_trigger_counts_from_completed_schedule: tuple[tuple[str, int], ...]
    logical_duration_seconds: float
    configured_output_tail_seconds: float

    def __post_init__(self) -> None:
        _sha256(self.artifact_digest, "artifact_digest")
        _text(self.simulator_id, "simulator_id")
        counts = tuple(self.expected_trigger_counts_from_completed_schedule)
        if len({channel for channel, _count in counts}) != len(counts):
            raise ValueError("simulated trigger channels must be unique")
        for channel, count in counts:
            _text(channel, "trigger channel")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("expected trigger count must be non-negative int")
        object.__setattr__(
            self,
            "expected_trigger_counts_from_completed_schedule",
            counts,
        )
        for field in ("logical_duration_seconds", "configured_output_tail_seconds"):
            raw = getattr(self, field)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) < 0
            ):
                raise ValueError(f"{field} must be finite and non-negative")
            object.__setattr__(self, field, float(raw))


PulseTerminalReceipt = PulseCompletion | SimulatedPulseReceipt


@dataclass(frozen=True)
class PulseTerminalAck:
    session_id: str
    binding_instance_id: str
    receipt: PulseTerminalReceipt

    def __post_init__(self) -> None:
        for field in ("session_id", "binding_instance_id"):
            _text(getattr(self, field), field)
        if not isinstance(self.receipt, (PulseCompletion, SimulatedPulseReceipt)):
            raise TypeError("receipt must be PulseCompletion or SimulatedPulseReceipt")

    @property
    def artifact_digest(self) -> str:
        if isinstance(self.receipt, PulseCompletion):
            return self.receipt.prepared_ref.artifact_digest
        return self.receipt.artifact_digest

    @property
    def expected_trigger_counts_from_completed_schedule(
        self,
    ) -> tuple[tuple[str, int], ...]:
        return self.receipt.expected_trigger_counts_from_completed_schedule

    @property
    def evidence_kind(self) -> PulseTerminalEvidenceKind:
        return (
            PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS
            if isinstance(self.receipt, PulseCompletion)
            else PulseTerminalEvidenceKind.SIMULATED
        )


def validate_pulse_terminal_for_artifact(
    acknowledgement: PulseTerminalAck,
    artifact: CompiledPulseArtifact,
) -> None:
    """Revalidate a terminal receipt at every neutral/artifact trust boundary."""

    if not isinstance(acknowledgement, PulseTerminalAck):
        raise TypeError("acknowledgement must be PulseTerminalAck")
    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if artifact.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
        raise ValueError("continuous monitor cannot have a finite terminal receipt")
    if acknowledgement.artifact_digest != artifact.fingerprint:
        raise ValueError("pulse terminal belongs to another compiled artifact")
    expected_counts = tuple(
        (schedule.channel, schedule.total)
        for schedule in artifact.trigger_schedules
    )
    if (
        acknowledgement.expected_trigger_counts_from_completed_schedule
        != expected_counts
    ):
        raise ValueError("pulse terminal expected counts differ from compiled schedule")

    receipt = acknowledgement.receipt
    if isinstance(receipt, PulseCompletion):
        validate_backend_completion_for_artifact(
            PulseBackendCompletion(
                receipt.hardware_terminal,
                receipt.post_terminal_tail,
            ),
            artifact,
        )
        return

    playback = build_pulse_playback(artifact)
    expected_tail = (
        artifact.max_configured_output_delay_ticks / artifact.target_ir.clock_hz
    )
    if not math.isclose(
        receipt.logical_duration_seconds,
        playback.logical_duration,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("simulated terminal duration differs from compiled pulse")
    if not math.isclose(
        receipt.configured_output_tail_seconds,
        expected_tail,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("simulated terminal tail differs from compiled pulse")


def _simulated_receipt_to_tree(value: SimulatedPulseReceipt) -> dict[str, object]:
    return {
        "artifact_digest": value.artifact_digest,
        "simulator_id": value.simulator_id,
        "expected_trigger_counts_from_completed_schedule": [
            [channel, count]
            for channel, count in value.expected_trigger_counts_from_completed_schedule
        ],
        "logical_duration_seconds": value.logical_duration_seconds,
        "configured_output_tail_seconds": value.configured_output_tail_seconds,
    }


def _simulated_receipt_from_tree(tree: object) -> SimulatedPulseReceipt:
    fields = {
        "artifact_digest",
        "simulator_id",
        "expected_trigger_counts_from_completed_schedule",
        "logical_duration_seconds",
        "configured_output_tail_seconds",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("SimulatedPulseReceipt has an unknown field set")
    raw_counts = tree["expected_trigger_counts_from_completed_schedule"]
    if not isinstance(raw_counts, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in raw_counts
    ):
        raise ValueError("simulated trigger counts must be [channel, count] rows")
    return SimulatedPulseReceipt(
        tree["artifact_digest"],
        tree["simulator_id"],
        tuple((item[0], item[1]) for item in raw_counts),
        tree["logical_duration_seconds"],
        tree["configured_output_tail_seconds"],
    )


def pulse_terminal_ack_to_tree(value: PulseTerminalAck) -> dict[str, object]:
    if not isinstance(value, PulseTerminalAck):
        raise TypeError("value must be PulseTerminalAck")
    hardware = isinstance(value.receipt, PulseCompletion)
    return {
        "schema": _PULSE_TERMINAL_ACK_SCHEMA,
        "session_id": value.session_id,
        "binding_instance_id": value.binding_instance_id,
        "receipt_kind": "HARDWARE" if hardware else "SIMULATED",
        "receipt": (
            pulse_completion_to_tree(value.receipt)
            if hardware
            else _simulated_receipt_to_tree(value.receipt)
        ),
    }


def pulse_terminal_ack_from_tree(tree: object) -> PulseTerminalAck:
    fields = {
        "schema",
        "session_id",
        "binding_instance_id",
        "receipt_kind",
        "receipt",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PulseTerminalAck has an unknown field set")
    if tree["schema"] != _PULSE_TERMINAL_ACK_SCHEMA:
        raise ValueError("PulseTerminalAck schema differs")
    kind = tree["receipt_kind"]
    if kind == "HARDWARE":
        receipt = pulse_completion_from_tree(tree["receipt"])
    elif kind == "SIMULATED":
        receipt = _simulated_receipt_from_tree(tree["receipt"])
    else:
        raise ValueError("PulseTerminalAck receipt kind differs")
    return PulseTerminalAck(
        tree["session_id"],
        tree["binding_instance_id"],
        receipt,
    )


@dataclass(frozen=True)
class BoundPulsePort:
    capability_attestation: VerifiedDeviceCapability
    cleanup_operations: tuple[SafetyOperation, ...]

    def __post_init__(self) -> None:
        admit_bound_capability(
            self.capability_attestation,
            SequencerCapabilitySnapshot,
        )
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
        device = context.cleanup_device(self.device.key)
        return cleanup_device_session(
            device,
            self.cleanup_operations,
            session_id,
            self.capability.max_blocking_call_seconds,
            termination_failure_reason=(
                "sequencer cleanup did not reach a verified safe terminal"
            ),
            termination_recovery_action=(
                "verify sequencer output state and recover the connection"
            ),
            verification_failure_reason="sequencer safe-state verification failed",
            verification_recovery_action=(
                "inspect outputs and recover the sequencer before reuse"
            ),
        )

    def verify_idle(self, context: RunContext) -> CleanupReport:
        return verify_cleanup_device_safe_state(
            context.cleanup_device(self.device.key),
            failure_reason="unopened sequencer safe-state verification failed",
            recovery_action="inspect and recover the sequencer before reuse",
        )


class _PulseSessionState(str, Enum):
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
        self._state = _PulseSessionState.NEW
        self._owner_thread_id = threading.get_ident()
        self._hardware_prepare_attempted = False
        self._terminal: PulseTerminalAck | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def terminal(self) -> PulseTerminalAck:
        if self._terminal is None:
            raise RuntimeError("pulse session has no terminal acknowledgement")
        return self._terminal

    def owns_terminal(self, terminal: PulseTerminalAck) -> bool:
        """Prove that this exact session minted and retained ``terminal``."""

        self._assert_owner_thread()
        return (
            self._state is _PulseSessionState.COMPLETED
            and self._terminal is terminal
        )

    def prepare(self, context: RunContext) -> None:
        self._assert_owner_thread()
        if self._state is not _PulseSessionState.NEW:
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
            self._state = _PulseSessionState.FAILED
            raise
        self._state = _PulseSessionState.PREPARED

    def fire(self, context: RunContext) -> None:
        self._assert_owner_thread()
        if self._state is not _PulseSessionState.PREPARED:
            raise RuntimeError("pulse session must be prepared before FIRE")
        try:
            ack = context.device(self._port.device.key).execute(
                FirePulseCommand(self._session_id, self._request.artifact_digest)
            )
            self._validate_ack(ack, PulseFiredAck)
            if ack.artifact_digest != self._request.artifact_digest:
                raise RuntimeError("FIRE acknowledgement artifact differs")
        except BaseException:
            self._state = _PulseSessionState.FAILED
            raise
        self._state = _PulseSessionState.FIRED

    def complete(self, context: RunContext) -> PulseTerminalAck:
        self._assert_owner_thread()
        if self._state is not _PulseSessionState.FIRED:
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
            if ack.evidence_kind is not self._port.capability.terminal_evidence_kind:
                raise RuntimeError("pulse terminal evidence kind differs from capability")
            validate_pulse_terminal_for_artifact(ack, self._request.artifact)
        except BaseException:
            self._state = _PulseSessionState.FAILED
            raise
        self._terminal = ack
        self._state = _PulseSessionState.COMPLETED
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
        if self._state is not _PulseSessionState.COMPLETED:
            self._state = _PulseSessionState.FAILED

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
        if ack.binding_instance_id != self._port.device.binding_instance_id:
            raise RuntimeError("pulse acknowledgement binding instance differs")

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
    "PulseTerminalAck",
    "PulseTerminalEvidenceKind",
    "PulseTerminalReceipt",
    "SimulatedPulseReceipt",
    "pulse_terminal_ack_from_tree",
    "pulse_terminal_ack_to_tree",
    "validate_pulse_terminal_for_artifact",
    "SequencerCapabilitySnapshot",
]
