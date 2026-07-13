"""Flat synchronous Run semantics hosted by one authoritative owner thread."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar

from .cancellation import CancellationRequested, CancellationToken
from .commit import (
    CheckpointCommit,
    CommitIntent,
    CommitKind,
    CommitSubject,
    CommitTarget,
    FinalCommit,
    PublishVisibilityUnknown,
    _CommitAuthoritySnapshot,
    _consume_commit_authority,
    _discard_commit_authority,
    commit_now,
)
from .ports import (
    BoundDevice,
    CleanupDevice,
    RunDevice,
    CleanupStepAck,
    SafetyProof,
    SafetyInterrupt,
    SafetyOperation,
    SessionClosedAck,
    _mint_safety_proof,
    _open_device_run,
)
from .resources import (
    ClaimMode,
    HazardClaim,
    HazardEpochExpired,
    QuarantineJournalError,
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
    ResourceQuarantined,
    SafeReceipt,
    SafetyDecision,
    SafetyDispositionBundle,
    SafetyOutcome,
    TerminalPublication,
    _mint_terminal_publication,
)


PreparedT = TypeVar("PreparedT")
ResultT = TypeVar("ResultT")
CommitT = TypeVar("CommitT")
_MISSING = object()


def _canonical_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be non-empty text without surrounding whitespace")
    return value


def _validate_timeout(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite non-negative number or None")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field} must be a finite non-negative number or None")
    return normalized


@dataclass(frozen=True, order=True)
class RunId:
    value: str

    def __post_init__(self) -> None:
        _canonical_text(self.value, "RunId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class RunRevision:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0:
            raise ValueError("RunRevision must be a non-negative integer")


class RunState(str, Enum):
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in (self.SUCCEEDED, self.FAILED, self.CANCELLED)


class RunMode(str, Enum):
    FINITE_EXACT = "FINITE_EXACT"
    CONTINUOUS_MONITOR = "CONTINUOUS_MONITOR"


class CancelOutcome(str, Enum):
    REQUESTED = "REQUESTED"
    ALREADY_REQUESTED = "ALREADY_REQUESTED"
    TOO_LATE_ALREADY_COMMITTED = "TOO_LATE_ALREADY_COMMITTED"
    TOO_LATE_FINALIZING = "TOO_LATE_FINALIZING"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"


class CheckpointDisposition(str, Enum):
    COMMITTED = "COMMITTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Public checkpoint fact; repository-native typed refs stay behind CommitTarget."""

    commit_id: str
    target: CommitTarget
    disposition: CheckpointDisposition

    def __post_init__(self) -> None:
        _canonical_text(self.commit_id, "checkpoint commit_id")
        if not isinstance(self.target, CommitTarget):
            raise TypeError("checkpoint target must be CommitTarget")
        if not isinstance(self.disposition, CheckpointDisposition):
            raise TypeError("checkpoint disposition must be CheckpointDisposition")


@dataclass(frozen=True)
class RunSnapshot:
    run_id: RunId
    revision: RunRevision
    state: RunState
    phase: str
    final_committed: bool
    committed_checkpoint: CheckpointSnapshot | None
    safety_bundle_id: str | None
    commit_recovery_warning: str | None
    primary_error: str | None
    cleanup_errors: tuple[str, ...]
    recovery_instruction: str | None


@dataclass(frozen=True)
class CleanupReport:
    """Plan-owned cleanup facts; the controller checks total hazard coverage."""

    safety_proofs: tuple[SafetyProof, ...] = ()
    decisions: tuple[SafetyDecision, ...] = ()
    errors: tuple[BaseException, ...] = ()

    def __post_init__(self) -> None:
        proofs = tuple(self.safety_proofs)
        decisions = tuple(self.decisions)
        errors = tuple(self.errors)
        if any(not isinstance(value, SafetyProof) for value in proofs):
            raise TypeError("cleanup safety_proofs must contain SafetyProof values")
        if len({id(value) for value in proofs}) != len(proofs):
            raise ValueError("cleanup safety proofs must be unique")
        if any(not isinstance(value, SafetyDecision) for value in decisions):
            raise TypeError("cleanup decisions must contain SafetyDecision values")
        if len({value.key for value in decisions}) != len(decisions):
            raise ValueError("cleanup decisions must have unique ResourceKeys")
        if any(value.outcome is SafetyOutcome.SAFE for value in decisions):
            raise ValueError("SAFE cleanup decisions require a RunContext SafetyProof")
        if any(not isinstance(error, BaseException) for error in errors):
            raise TypeError("cleanup errors must contain exceptions")
        object.__setattr__(self, "safety_proofs", proofs)
        object.__setattr__(self, "decisions", tuple(sorted(decisions, key=lambda value: value.key)))
        object.__setattr__(self, "errors", errors)

    @property
    def quarantine_keys(self) -> tuple[ResourceKey, ...]:
        return tuple(
            decision.key
            for decision in self.decisions
            if decision.outcome is SafetyOutcome.UNSAFE
        )

    @classmethod
    def safe(
        cls,
        proofs: tuple[SafetyProof, ...],
        *,
        errors: tuple[BaseException, ...] = (),
    ) -> "CleanupReport":
        return cls(safety_proofs=tuple(proofs), errors=tuple(errors))

    @classmethod
    def unsafe(
        cls,
        keys: tuple[ResourceKey, ...],
        *,
        reason: str,
        recovery_action: str,
        errors: tuple[BaseException, ...] = (),
    ) -> "CleanupReport":
        return cls(
            decisions=tuple(
                SafetyDecision.unsafe(
                    key,
                    reason=reason,
                    recovery_action=recovery_action,
                )
                for key in keys
            ),
            errors=tuple(errors),
        )

    @classmethod
    def mixed(
        cls,
        safety_proofs: tuple[SafetyProof, ...],
        decisions: tuple[SafetyDecision, ...],
        *,
        errors: tuple[BaseException, ...] = (),
    ) -> "CleanupReport":
        return cls(
            safety_proofs=tuple(safety_proofs),
            decisions=decisions,
            errors=errors,
        )


@dataclass(frozen=True)
class RunPlan(Generic[PreparedT, ResultT]):
    name: str
    mode: RunMode
    resource_claims: tuple[ResourceClaim, ...]
    hazard_claims: tuple[HazardClaim, ...]
    bound_devices: tuple[BoundDevice, ...]
    preflight: Callable[["RunContext"], PreparedT]
    execute: Callable[["RunContext", PreparedT], ResultT]
    cleanup: Callable[["RunContext", PreparedT | None, BaseException | None], CleanupReport]
    finalize: Callable[["PostSafetyContext", ResultT], ResultT]
    interrupt_operations: tuple[SafetyInterrupt, ...] = ()
    timeout_seconds: float | None = None
    requires_final_commit: bool = False

    def __post_init__(self) -> None:
        _canonical_text(self.name, "run plan name")
        if not isinstance(self.mode, RunMode):
            raise TypeError("mode must be RunMode")
        claims = tuple(self.resource_claims)
        if any(not isinstance(claim, ResourceClaim) for claim in claims):
            raise TypeError("resource_claims must contain ResourceClaim values")
        object.__setattr__(self, "resource_claims", claims)
        hazards = tuple(self.hazard_claims)
        if any(not isinstance(hazard, HazardClaim) for hazard in hazards):
            raise TypeError("hazard_claims must contain HazardClaim values")
        if len({hazard.key for hazard in hazards}) != len(hazards):
            raise ValueError("hazard_claims must have unique ResourceKeys")
        object.__setattr__(self, "hazard_claims", hazards)
        device_exclusive = {
            claim.key
            for claim in claims
            if claim.mode is ClaimMode.EXCLUSIVE and claim.key.segments[0] == "device"
        }
        if {hazard.key for hazard in hazards} != device_exclusive:
            raise ValueError(
                "every EXCLUSIVE device claim requires exactly one connection-generation hazard claim"
            )
        devices = tuple(self.bound_devices)
        if any(not isinstance(device, BoundDevice) for device in devices):
            raise TypeError("bound_devices must contain BoundDevice values")
        if len({device.key for device in devices}) != len(devices):
            raise ValueError("bound_devices must have unique ResourceKeys")
        device_by_key = {device.key: device for device in devices}
        if set(device_by_key) != device_exclusive:
            raise ValueError(
                "every EXCLUSIVE device claim requires exactly one BoundDevice"
            )
        for hazard in hazards:
            device = device_by_key[hazard.key]
            if device.stable_device_identity != hazard.stable_device_identity:
                raise ValueError(
                    f"hazard stable identity does not match BoundDevice for {hazard.key}"
                )
            if device.connection_generation != hazard.connection_generation:
                raise ValueError(
                    f"hazard generation does not match BoundDevice for {hazard.key}"
                )
        object.__setattr__(self, "bound_devices", devices)
        interrupts = tuple(self.interrupt_operations)
        if any(not isinstance(value, SafetyInterrupt) for value in interrupts):
            raise TypeError("interrupt_operations must contain SafetyInterrupt values")
        for interrupt in interrupts:
            device = device_by_key.get(interrupt.key)
            if device is None:
                raise ValueError("interrupt operation must target a bound device")
            if interrupt.operation not in device.interrupt_capabilities:
                raise ValueError(
                    f"device {interrupt.key} does not bind thread-safe interrupt {interrupt.operation.value}"
                )
        if len(set(interrupts)) != len(interrupts):
            raise ValueError("interrupt_operations cannot contain duplicates")
        object.__setattr__(self, "interrupt_operations", interrupts)
        for field, value in (
            ("preflight", self.preflight),
            ("execute", self.execute),
            ("cleanup", self.cleanup),
            ("finalize", self.finalize),
        ):
            if not callable(value):
                raise TypeError(f"{field} must be callable")
        if not isinstance(self.requires_final_commit, bool):
            raise TypeError("requires_final_commit must be bool")
        object.__setattr__(
            self,
            "timeout_seconds",
            _validate_timeout(self.timeout_seconds, "timeout_seconds"),
        )


class CapabilityRevoked(RuntimeError):
    """A caller attempted to use a process-local Run capability outside its scope."""


class _PermanentCommitRejection(CapabilityRevoked):
    """The operation can never become eligible later in this Run."""


class _CommitResolutionMode(Enum):
    FORCE_ABORT = "force-abort"
    RECOVER_VISIBILITY = "recover-visibility"
    FORCE_COMMIT = "force-commit"


_NO_COMMITTED_RESULT = object()


@dataclass(frozen=True)
class _DurableCommitOutcome(Generic[CommitT]):
    result: CommitT
    recovered_error: BaseException | None = None


@dataclass(frozen=True)
class _DurableReconciliationOutcome:
    committed: bool
    result: object = _NO_COMMITTED_RESULT


class FinalCommitReconciliationRequired(RuntimeError):
    def __init__(
        self,
        operation: FinalCommit,
        authority_snapshot: _CommitAuthoritySnapshot,
        intent: CommitIntent,
        publish_error: BaseException,
        recovery_error: BaseException,
        resolution_mode: _CommitResolutionMode,
        committed_result: object = _NO_COMMITTED_RESULT,
    ) -> None:
        self.operation = operation
        self._authority_snapshot = authority_snapshot
        self.intent = intent
        self.publish_error = publish_error
        self.recovery_error = recovery_error
        self._resolution_mode = resolution_mode
        self._committed_result = committed_result
        super().__init__(
            "final commit state is unresolved; durable reconciliation is required: "
            f"{type(recovery_error).__name__}: {recovery_error}"
        )


class CheckpointReconciliationRequired(RuntimeError):
    def __init__(
        self,
        operation: CheckpointCommit,
        authority_snapshot: _CommitAuthoritySnapshot,
        intent: CommitIntent,
        publish_error: BaseException,
        recovery_error: BaseException,
        resolution_mode: _CommitResolutionMode,
        committed_result: object = _NO_COMMITTED_RESULT,
    ) -> None:
        self.operation = operation
        self._authority_snapshot = authority_snapshot
        self.intent = intent
        self.publish_error = publish_error
        self.recovery_error = recovery_error
        self._resolution_mode = resolution_mode
        self._committed_result = committed_result
        super().__init__(
            "checkpoint commit state is unresolved; durable reconciliation is required: "
            f"{type(recovery_error).__name__}: {recovery_error}"
        )


class InterruptFailures(RuntimeError):
    def __init__(self, failures: tuple[tuple[SafetyInterrupt, BaseException], ...]) -> None:
        self.failures = failures
        detail = "; ".join(
            f"{item.key}:{item.operation.value} -> {type(error).__name__}: {error}"
            for item, error in failures
        )
        super().__init__(f"one or more hardware interrupts failed: {detail}")


class RunContext:
    """Narrow process-local capability passed to one flat RunPlan."""

    def __init__(
        self,
        handle: "RunHandle",
        token: CancellationToken,
        deadline: float | None,
        bound_devices: tuple[BoundDevice, ...],
    ) -> None:
        self._handle = handle
        self.cancellation = token
        self.deadline = deadline
        self._hardware_condition = threading.Condition(threading.RLock())
        self._execution_enabled = False
        self._execution_sealed = False
        self._cleanup_enabled = False
        self._hardware_revoked = False
        self._hardware_inflight = 0
        self._safety_bundle_id: str | None = None
        self._post_safety = False
        self._devices = {device.key: device for device in bound_devices}
        self._device_lease = _open_device_run(self.run_id.value, bound_devices)
        self._interrupt_enabled = False
        self._issued_safety_proofs: dict[object, tuple[SafetyProof, SafeReceipt]] = {}

    @property
    def run_id(self) -> RunId:
        return self._handle.run_id

    @property
    def safety_bundle_id(self) -> str | None:
        with self._hardware_condition:
            return self._safety_bundle_id

    @property
    def post_safety(self) -> bool:
        with self._hardware_condition:
            return self._post_safety

    def set_phase(self, phase: str) -> None:
        self._handle._set_phase(_canonical_text(phase, "run phase"))

    def checkpoint(self) -> None:
        self.cancellation.checkpoint()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(f"run {self.run_id} exceeded its monotonic deadline")

    def device(self, key: ResourceKey) -> RunDevice:
        if not isinstance(key, ResourceKey):
            raise TypeError("device key must be ResourceKey")
        try:
            return RunDevice(self, self._devices[key])
        except KeyError as exc:
            raise CapabilityRevoked(f"run {self.run_id} does not own device {key}") from exc

    def cleanup_device(self, key: ResourceKey) -> CleanupDevice:
        if not isinstance(key, ResourceKey):
            raise TypeError("device key must be ResourceKey")
        try:
            return CleanupDevice(self, self._devices[key])
        except KeyError as exc:
            raise CapabilityRevoked(f"run {self.run_id} does not own device {key}") from exc

    def _execute_bound_device(
        self,
        binding: BoundDevice,
        command: object,
    ) -> object:
        with self._hardware_condition:
            if (
                not self._execution_enabled
                or self._execution_sealed
                or self._hardware_revoked
                or self.cancellation.is_cancelled
            ):
                raise CapabilityRevoked(
                    f"run {self.run_id} has no active execution hardware capability"
                )
            if self._devices.get(binding.key) is not binding:
                raise CapabilityRevoked("device binding does not belong to this Run")
            self._hardware_inflight += 1
        try:
            return self._device_lease.execute(binding, command)
        finally:
            with self._hardware_condition:
                self._hardware_inflight -= 1
                self._hardware_condition.notify_all()

    def _execute_cleanup_step(
        self,
        binding: BoundDevice,
        operation: SafetyOperation,
    ) -> CleanupStepAck:
        with self._hardware_condition:
            if not self._cleanup_enabled or self._hardware_revoked:
                raise CapabilityRevoked(
                    f"run {self.run_id} has no active cleanup hardware capability"
                )
            if self._devices.get(binding.key) is not binding:
                raise CapabilityRevoked("device binding does not belong to this Run")
            self._hardware_inflight += 1
        try:
            return self._device_lease.cleanup_step(binding, operation)
        finally:
            with self._hardware_condition:
                self._hardware_inflight -= 1
                self._hardware_condition.notify_all()

    def _close_bound_device_session(
        self,
        binding: BoundDevice,
        command: object,
    ) -> SessionClosedAck:
        with self._hardware_condition:
            if not self._cleanup_enabled or self._hardware_revoked:
                raise CapabilityRevoked(
                    f"run {self.run_id} has no active cleanup hardware capability"
                )
            if self._devices.get(binding.key) is not binding:
                raise CapabilityRevoked("device binding does not belong to this Run")
            self._hardware_inflight += 1
        try:
            return self._device_lease.close_session(binding, command)
        finally:
            with self._hardware_condition:
                self._hardware_inflight -= 1
                self._hardware_condition.notify_all()

    def _verify_bound_safe_state(self, binding: BoundDevice) -> SafetyProof:
        with self._hardware_condition:
            if not self._cleanup_enabled or self._hardware_revoked:
                raise CapabilityRevoked(
                    f"run {self.run_id} has no active cleanup hardware capability"
                )
            if self._devices.get(binding.key) is not binding:
                raise CapabilityRevoked("device binding does not belong to this Run")
            self._hardware_inflight += 1
        try:
            receipt = self._device_lease.verify_safe_state(binding)
            nonce = object()
            proof = _mint_safety_proof(
                run_id=self.run_id.value,
                receipt=receipt,
                nonce=nonce,
            )
            with self._hardware_condition:
                self._issued_safety_proofs[nonce] = (proof, receipt)
            return proof
        finally:
            with self._hardware_condition:
                self._hardware_inflight -= 1
                self._hardware_condition.notify_all()

    def _consume_safety_proofs(
        self,
        proofs: tuple[SafetyProof, ...],
    ) -> tuple[SafeReceipt, ...]:
        proofs = tuple(proofs)
        with self._hardware_condition:
            receipts = []
            for proof in proofs:
                if not isinstance(proof, SafetyProof) or proof.run_id != self.run_id.value:
                    raise ValueError("cleanup safety proof belongs to another Run")
                issued = self._issued_safety_proofs.get(proof._nonce)
                if issued is None or issued[0] is not proof:
                    raise ValueError("cleanup safety proof was not issued or was already consumed")
                receipts.append(issued[1])
            for proof in proofs:
                self._issued_safety_proofs.pop(proof._nonce)
            return tuple(receipts)

    def _enable_hardware(self) -> bool:
        with self._hardware_condition:
            if self._hardware_revoked:
                raise RuntimeError("hardware capability cannot be re-enabled after safety")
            if self.cancellation.is_cancelled:
                self._execution_sealed = True
                return False
            self._execution_enabled = True
            self._interrupt_enabled = True
            return True

    def _seal_execution_on_cancel(self) -> None:
        with self._hardware_condition:
            self._execution_sealed = True
            self._execution_enabled = False
            self._hardware_condition.notify_all()

    def _enter_cleanup_hardware(self) -> None:
        with self._hardware_condition:
            self._execution_sealed = True
            self._execution_enabled = False
            self._interrupt_enabled = False
            while self._hardware_inflight:
                self._hardware_condition.wait()
            if self._hardware_revoked:
                raise RuntimeError("cleanup cannot start after hardware capability revocation")
            self._cleanup_enabled = True

    def _revoke_hardware_before_safety(self) -> None:
        with self._hardware_condition:
            self._execution_enabled = False
            self._execution_sealed = True
            self._cleanup_enabled = False
            self._interrupt_enabled = False
            self._hardware_revoked = True
            while self._hardware_inflight:
                self._hardware_condition.wait()
            self._device_lease.revoke()

    def _mark_safety_durable(self, bundle: SafetyDispositionBundle | None) -> None:
        with self._hardware_condition:
            if not self._hardware_revoked or self._hardware_inflight:
                raise RuntimeError("hardware capability must be quiescent before safety commit")
            self._safety_bundle_id = None if bundle is None else bundle.bundle_id
            self._post_safety = True
            self._hardware_condition.notify_all()

    def _run_interrupts(self, interrupts: tuple[SafetyInterrupt, ...]) -> None:
        failures: list[tuple[SafetyInterrupt, BaseException]] = []
        for interrupt in interrupts:
            try:
                with self._hardware_condition:
                    if not self._interrupt_enabled or self._hardware_revoked:
                        raise CapabilityRevoked(
                            f"run {self.run_id} interrupt capability is not active"
                        )
                    binding = self._devices.get(interrupt.key)
                    if binding is None:
                        raise CapabilityRevoked(
                            f"run {self.run_id} does not own interrupt device {interrupt.key}"
                        )
                    self._hardware_inflight += 1
                try:
                    self._device_lease.interrupt(binding, interrupt.operation)
                finally:
                    with self._hardware_condition:
                        self._hardware_inflight -= 1
                        self._hardware_condition.notify_all()
            except BaseException as exc:
                failures.append((interrupt, exc))
        if failures:
            raise InterruptFailures(tuple(failures))

    def _post_safety_context(self) -> "PostSafetyContext":
        with self._hardware_condition:
            if not self._post_safety:
                raise RuntimeError("post-safety context requires durable safety disposition")
            return PostSafetyContext(
                _POST_SAFETY_CONTEXT_TOKEN,
                run_id=self.run_id,
                cancellation=self.cancellation,
                deadline=self.deadline,
                safety_bundle_id=self._safety_bundle_id,
                handle=self._handle,
            )


_POST_SAFETY_CONTEXT_TOKEN = object()


class PostSafetyContext:
    """Finalize capability with no device/session/Port access."""

    __slots__ = (
        "_run_id",
        "_cancellation",
        "_deadline",
        "_safety_bundle_id",
        "_handle",
    )

    def __init__(
        self,
        token: object | None = None,
        *,
        run_id: RunId,
        cancellation: CancellationToken,
        deadline: float | None,
        safety_bundle_id: str | None,
        handle: "RunHandle",
    ) -> None:
        if token is not _POST_SAFETY_CONTEXT_TOKEN:
            raise PermissionError("PostSafetyContext can only be minted by RunController")
        if not isinstance(run_id, RunId):
            raise TypeError("PostSafetyContext run_id must be RunId")
        if not isinstance(cancellation, CancellationToken):
            raise TypeError("PostSafetyContext cancellation must be CancellationToken")
        if not isinstance(handle, RunHandle):
            raise TypeError("PostSafetyContext handle must be RunHandle")
        if handle.run_id != run_id:
            raise ValueError("PostSafetyContext run_id differs from RunHandle")
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_cancellation", cancellation)
        object.__setattr__(self, "_deadline", deadline)
        object.__setattr__(self, "_safety_bundle_id", safety_bundle_id)
        object.__setattr__(self, "_handle", handle)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PostSafetyContext is immutable")

    @property
    def run_id(self) -> RunId:
        return self._run_id

    @property
    def cancellation(self) -> CancellationToken:
        return self._cancellation

    @property
    def deadline(self) -> float | None:
        return self._deadline

    @property
    def safety_bundle_id(self) -> str | None:
        return self._safety_bundle_id

    def checkpoint(self) -> None:
        self.cancellation.checkpoint()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(f"run {self.run_id} exceeded its monotonic deadline")

    def authorize_commit_preparation(self, kind: CommitKind) -> CommitSubject:
        """Validate the owner gate before a repository performs staging I/O."""

        if not isinstance(kind, CommitKind):
            raise TypeError("commit preparation kind must be CommitKind")
        self._handle._validate_commit_preparation(kind, self.deadline)
        return CommitSubject(self.run_id.value, self.safety_bundle_id)

    def _validate_commit_entry(
        self,
        operation: CheckpointCommit[object] | FinalCommit[object],
    ) -> None:
        try:
            self._handle._validate_commit_caller(self.deadline)
        except BaseException as rejection:
            if isinstance(
                rejection,
                (_PermanentCommitRejection, CancellationRequested, TimeoutError),
            ):
                self._handle._discard_rejected_commit(operation, rejection)
            raise

    def commit_final(self, operation: FinalCommit[CommitT]) -> CommitT:
        if not isinstance(operation, FinalCommit):
            raise TypeError("commit_final requires a FinalCommit operation")
        self._validate_commit_entry(operation)
        return self._handle._commit_final(operation, deadline=self.deadline)

    def commit_checkpoint(self, operation: CheckpointCommit[CommitT]) -> CommitT:
        if not isinstance(operation, CheckpointCommit):
            raise TypeError("commit_checkpoint requires a CheckpointCommit operation")
        self._validate_commit_entry(operation)
        return self._handle._commit_checkpoint(operation, deadline=self.deadline)


class RunStartRejected(RuntimeError):
    def __init__(self, outcome: ResourceBusy | ResourceQuarantined) -> None:
        super().__init__(f"run start rejected: {outcome}")
        self.outcome = outcome


class RunFailed(RuntimeError):
    def __init__(self, snapshot: RunSnapshot, primary: BaseException | None) -> None:
        super().__init__(snapshot.primary_error or "run failed during cleanup")
        self.snapshot = snapshot
        self.primary = primary


class RunCancelled(RuntimeError):
    def __init__(self, snapshot: RunSnapshot) -> None:
        super().__init__(f"run {snapshot.run_id} was cancelled")
        self.snapshot = snapshot


class RunStillCancelling(RuntimeError):
    def __init__(self, handle: "RunHandle") -> None:
        super().__init__(f"run {handle.run_id} is still cancelling; retain its RunHandle")
        self.handle = handle


class RunHandle(Generic[ResultT]):
    """Queryable authoritative state; never a second lifecycle owner."""

    def __init__(
        self,
        run_id: RunId,
        token: CancellationToken,
        on_terminal: Callable[["RunHandle"], None],
    ) -> None:
        self.run_id = run_id
        self._token = token
        self._on_terminal = on_terminal
        self._condition = threading.Condition(threading.RLock())
        self._revision = 0
        self._state = RunState.RUNNING
        self._phase = "starting"
        self._final_committed = False
        self._checkpoint_attempted = False
        self._committed_checkpoint: CheckpointSnapshot | None = None
        self._commit_inflight = False
        self._commit_recovery_warning: str | None = None
        self._pending_final_commit: FinalCommitReconciliationRequired | None = None
        self._pending_checkpoint: CheckpointReconciliationRequired | None = None
        self._safety_bundle_id: str | None = None
        self._primary: BaseException | None = None
        self._cleanup_errors: tuple[BaseException, ...] = ()
        self._result: object = _MISSING
        self._interrupt: Callable[[], None] | None = None
        self._cancel_sealer: Callable[[], None] | None = None
        self._interrupt_enabled = False
        self._interrupt_errors: list[BaseException] = []
        self._interrupt_inflight = False
        self._interrupt_thread: threading.Thread | None = None
        self._cancel_gate_closed = False
        self._cleanup_started = False
        self._owner_thread: threading.Thread | None = None
        self._recovery_instruction: str | None = None
        self._retry_disposition: Callable[[], None] | None = None
        self._retry_phase = "safety-journal-failed"
        self._retry_state = RunState.RUNNING
        self._recovery_lock = threading.Lock()

    def snapshot(self) -> RunSnapshot:
        with self._condition:
            return self._snapshot_locked()

    def cancel(self, reason: str = "user requested stop") -> CancelOutcome:
        with self._condition:
            if self._state.terminal:
                return CancelOutcome.ALREADY_TERMINAL
            if self._final_committed:
                return CancelOutcome.TOO_LATE_ALREADY_COMMITTED
            if self._cancel_gate_closed:
                return CancelOutcome.TOO_LATE_FINALIZING
            first = self._token.request(reason)
            if not first:
                return CancelOutcome.ALREADY_REQUESTED
            if self._cancel_sealer is not None:
                self._cancel_sealer()
            self._state = RunState.CANCELLING
            self._revision += 1
            interrupt = (
                self._interrupt
                if self._interrupt_enabled and not self._cleanup_started
                else None
            )
            if interrupt is not None:
                self._interrupt_inflight = True
            self._condition.notify_all()
        if interrupt is not None:
            self._start_interrupt(interrupt)
        return CancelOutcome.REQUESTED

    def _start_interrupt(self, interrupt: Callable[[], None]) -> None:
        def invoke_interrupt() -> None:
            error: BaseException | None = None
            try:
                interrupt()
            except BaseException as exc:
                error = exc
            finally:
                with self._condition:
                    if error is not None:
                        self._interrupt_errors.append(error)
                    self._interrupt_inflight = False
                    self._revision += 1
                    self._condition.notify_all()

        thread = threading.Thread(
            target=invoke_interrupt,
            name=f"zlc-interrupt-{self.run_id.value[:12]}",
            daemon=False,
        )
        with self._condition:
            self._interrupt_thread = thread
        try:
            thread.start()
        except BaseException as exc:
            with self._condition:
                self._interrupt_errors.append(exc)
                self._interrupt_inflight = False
                self._revision += 1
                self._condition.notify_all()

    def wait(self, timeout: float | None = None) -> RunSnapshot:
        timeout = _validate_timeout(timeout, "wait timeout") if timeout is not None else None
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._state.terminal:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"run {self.run_id} has not reached terminal state")
                self._condition.wait(remaining)
            snapshot = self._snapshot_locked()
        thread = self._owner_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.1)
        return snapshot

    def wait_for(
        self,
        predicate: Callable[[RunSnapshot], bool],
        timeout: float | None = None,
    ) -> RunSnapshot:
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        timeout = _validate_timeout(timeout, "wait timeout") if timeout is not None else None
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                snapshot = self._snapshot_locked()
                if predicate(snapshot):
                    return snapshot
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"run {self.run_id} did not reach requested state")
                self._condition.wait(remaining)

    def result(self, timeout: float | None = None) -> ResultT:
        snapshot = self.wait(timeout)
        if snapshot.state is RunState.SUCCEEDED:
            return self._result  # type: ignore[return-value]
        if snapshot.state is RunState.CANCELLED:
            raise RunCancelled(snapshot)
        raise RunFailed(snapshot, self._primary)

    def retry_recovery(self) -> bool:
        with self._recovery_lock:
            with self._condition:
                retry = self._retry_disposition
                if retry is None:
                    return False
                instruction = self._recovery_instruction or "retry safety recovery"
                previous_errors = self._cleanup_errors
                retry_phase = self._retry_phase
                retry_state = self._retry_state
                if (
                    retry_phase == "checkpoint-reconciliation-failed"
                    and self._token.is_cancelled
                ):
                    retry_state = RunState.CANCELLING

            def retry_owner() -> None:
                try:
                    retry()
                except BaseException as exc:
                    self._install_retry(
                        retry,
                        instruction,
                        previous_errors + (exc,),
                        phase=retry_phase,
                        state=retry_state,
                    )

            thread = threading.Thread(
                target=retry_owner,
                name=f"zlc-run-{self.run_id.value[:12]}-recovery",
                daemon=False,
            )
            self._replace_owner_for_recovery(thread)
            self._resume_after_recovery(
                (
                    "retrying-final-commit-reconciliation"
                    if retry_phase == "final-commit-reconciliation-failed"
                    else (
                        "retrying-checkpoint-reconciliation"
                        if retry_phase == "checkpoint-reconciliation-failed"
                        else "retrying-safety-recovery"
                    )
                ),
                state=retry_state,
            )
            try:
                thread.start()
            except BaseException as exc:
                self._install_retry(
                    retry,
                    instruction,
                    previous_errors + (exc,),
                    phase=retry_phase,
                    state=retry_state,
                )
                raise
            return True

    def owner_thread_alive(self) -> bool:
        thread = self._owner_thread
        return thread is not None and thread.is_alive()

    def _snapshot_locked(self) -> RunSnapshot:
        return RunSnapshot(
            run_id=self.run_id,
            revision=RunRevision(self._revision),
            state=self._state,
            phase=self._phase,
            final_committed=self._final_committed,
            committed_checkpoint=self._committed_checkpoint,
            safety_bundle_id=self._safety_bundle_id,
            commit_recovery_warning=self._commit_recovery_warning,
            primary_error=None if self._primary is None else _format_error(self._primary),
            cleanup_errors=tuple(_format_error(error) for error in self._cleanup_errors),
            recovery_instruction=self._recovery_instruction,
        )

    def _bind_owner(
        self,
        thread: threading.Thread,
        interrupt: Callable[[], None] | None,
    ) -> None:
        with self._condition:
            if self._owner_thread is not None:
                raise RuntimeError("RunHandle owner already bound")
            self._owner_thread = thread
            self._interrupt = interrupt

    def _bind_cancel_sealer(self, sealer: Callable[[], None]) -> None:
        if not callable(sealer):
            raise TypeError("cancel sealer must be callable")
        with self._condition:
            if self._cancel_sealer is not None:
                raise RuntimeError("cancel sealer already bound")
            self._cancel_sealer = sealer

    def _replace_owner_for_recovery(self, thread: threading.Thread) -> None:
        with self._condition:
            if self._state.terminal:
                raise RuntimeError("cannot replace owner after terminal state")
            if (
                self._owner_thread is not None
                and self._owner_thread.is_alive()
                and self._retry_disposition is None
            ):
                raise RuntimeError("cannot replace a live owner thread")
            self._owner_thread = thread

    def _resume_after_recovery(self, phase: str, *, state: RunState | None = None) -> None:
        with self._condition:
            self._state = state or (
                RunState.CANCELLING if self._token.is_cancelled else RunState.RUNNING
            )
            self._phase = phase
            self._retry_disposition = None
            self._recovery_instruction = None
            self._cleanup_errors = ()
            self._revision += 1
            self._condition.notify_all()

    def _enable_interrupt(self) -> bool:
        with self._condition:
            if self._cleanup_started:
                return False
            if self._token.is_cancelled:
                return False
            self._interrupt_enabled = True
            return True

    def _set_phase(self, phase: str) -> None:
        with self._condition:
            if self._state.terminal:
                raise RuntimeError("cannot change phase after terminal state")
            self._phase = phase
            self._revision += 1
            self._condition.notify_all()

    def _validate_commit_caller(self, deadline: float | None) -> None:
        with self._condition:
            if self._state.terminal:
                raise _PermanentCommitRejection(
                    "post-safety commit capability is terminal"
                )
            self._token.checkpoint()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"run {self.run_id} exceeded its monotonic deadline")
            if threading.current_thread() is not self._owner_thread:
                raise CapabilityRevoked(
                    "post-safety commit capability belongs to the authoritative Run owner thread"
                )

    def _commit_gate_rejection_locked(
        self,
        kind: CommitKind,
        deadline: float | None,
    ) -> tuple[BaseException | None, bool]:
        """Return one shared preparation/admission decision while holding the Run lock."""

        if self._state.terminal:
            return (
                _PermanentCommitRejection(
                    "commit operation cannot run after terminal state"
                ),
                True,
            )
        try:
            self._token.checkpoint()
        except BaseException as error:
            return error, True
        if deadline is not None and time.monotonic() >= deadline:
            return (
                TimeoutError(
                    f"run {self.run_id} exceeded its monotonic deadline"
                ),
                True,
            )
        if threading.current_thread() is not self._owner_thread:
            return (
                CapabilityRevoked(
                    "commit operation belongs to the authoritative Run owner thread"
                ),
                False,
            )
        if self._pending_checkpoint is not None or self._pending_final_commit is not None:
            return (
                RuntimeError(
                    "cannot start a commit while durable reconciliation is pending"
                ),
                True,
            )
        if kind is CommitKind.CHECKPOINT and self._checkpoint_attempted:
            return RuntimeError("checkpoint commit may be attempted only once per Run"), True
        if kind is CommitKind.FINAL and self._final_committed:
            return RuntimeError("final commit marker may be set only once"), True
        if self._commit_inflight:
            return RuntimeError("another commit operation is already in flight"), True
        if self._cancel_gate_closed:
            return RuntimeError("run is already publishing terminal state"), True
        return None, True

    def _validate_commit_preparation(
        self,
        kind: CommitKind,
        deadline: float | None,
    ) -> None:
        with self._condition:
            rejection, _discard = self._commit_gate_rejection_locked(kind, deadline)
        if rejection is not None:
            raise rejection

    def _pause_final_commit_reconciliation(
        self,
        operation: FinalCommit,
        authority_snapshot: _CommitAuthoritySnapshot,
        intent: CommitIntent,
        primary_error: BaseException,
        reconciliation_error: BaseException,
        resolution_mode: _CommitResolutionMode,
        committed_result: object = _NO_COMMITTED_RESULT,
    ) -> FinalCommitReconciliationRequired:
        pending = FinalCommitReconciliationRequired(
            operation,
            authority_snapshot,
            intent,
            primary_error,
            reconciliation_error,
            resolution_mode,
            committed_result,
        )
        with self._condition:
            self._commit_inflight = False
            self._pending_final_commit = pending
            # Do not publish the externally retryable phase until
            # ``_install_retry`` has atomically installed its capability.
            self._phase = "final-commit-reconciliation-pending"
            self._revision += 1
            self._condition.notify_all()
        return pending

    def _pause_checkpoint_reconciliation(
        self,
        operation: CheckpointCommit,
        authority_snapshot: _CommitAuthoritySnapshot,
        intent: CommitIntent,
        primary_error: BaseException,
        reconciliation_error: BaseException,
        resolution_mode: _CommitResolutionMode,
        committed_result: object = _NO_COMMITTED_RESULT,
    ) -> CheckpointReconciliationRequired:
        pending = CheckpointReconciliationRequired(
            operation,
            authority_snapshot,
            intent,
            primary_error,
            reconciliation_error,
            resolution_mode,
            committed_result,
        )
        with self._condition:
            self._commit_inflight = False
            self._pending_checkpoint = pending
            self._committed_checkpoint = CheckpointSnapshot(
                authority_snapshot.preparation.commit_id,
                authority_snapshot.target,
                CheckpointDisposition.RECONCILIATION_REQUIRED,
            )
            # Do not publish the externally retryable phase until
            # ``_install_retry`` has atomically installed its capability.
            self._phase = "checkpoint-reconciliation-pending"
            self._revision += 1
            self._condition.notify_all()
        return pending

    @staticmethod
    def _discard_rejected_commit(
        operation: CheckpointCommit | FinalCommit,
        rejection: BaseException,
    ) -> None:
        try:
            _discard_commit_authority(operation.authority)
        except BaseException as discard_error:
            if hasattr(rejection, "add_note"):
                rejection.add_note(
                    f"rejected commit authority discard also failed: {discard_error!r}"
                )

    def _admit_commit(
        self,
        operation: CheckpointCommit[CommitT] | FinalCommit[CommitT],
        kind: CommitKind,
        deadline: float | None,
    ) -> _CommitAuthoritySnapshot[CommitT]:
        """Atomically validate the Run gate, consume authority, and reserve commit I/O."""

        rejection: BaseException | None = None
        discard_on_rejection = True
        authority: _CommitAuthoritySnapshot[CommitT] | None = None
        with self._condition:
            rejection, discard_on_rejection = self._commit_gate_rejection_locked(
                kind,
                deadline,
            )
            if rejection is None and operation.authority.kind is not kind:
                rejection = _PermanentCommitRejection(
                    "commit wrapper kind differs from its authority preparation"
                )
            if rejection is None and operation.authority.run_id != self.run_id.value:
                rejection = _PermanentCommitRejection(
                    "commit authority belongs to another Run"
                )
            if (
                rejection is None
                and operation.authority.safety_bundle_id != self._safety_bundle_id
            ):
                rejection = _PermanentCommitRejection(
                    "commit authority safety bundle differs from this Run"
                )
            if rejection is None:
                try:
                    authority = _consume_commit_authority(operation.authority)
                except BaseException as error:
                    rejection = error
                else:
                    if kind is CommitKind.CHECKPOINT:
                        self._checkpoint_attempted = True
                        self._phase = "preparing-checkpoint-intent"
                    else:
                        self._phase = "preparing-commit-intent"
                    self._commit_inflight = True
                    self._revision += 1
                    self._condition.notify_all()
        if rejection is not None:
            if discard_on_rejection:
                self._discard_rejected_commit(operation, rejection)
            raise rejection
        assert authority is not None
        return authority

    def _pause_durable_commit(
        self,
        kind: CommitKind,
        operation: CheckpointCommit | FinalCommit,
        authority: _CommitAuthoritySnapshot,
        intent: CommitIntent,
        publish_error: BaseException,
        recovery_error: BaseException,
        resolution_mode: _CommitResolutionMode,
        committed_result: object = _NO_COMMITTED_RESULT,
    ) -> CheckpointReconciliationRequired | FinalCommitReconciliationRequired:
        if kind is CommitKind.CHECKPOINT:
            assert isinstance(operation, CheckpointCommit)
            return self._pause_checkpoint_reconciliation(
                operation,
                authority,
                intent,
                publish_error,
                recovery_error,
                resolution_mode,
                committed_result,
            )
        assert isinstance(operation, FinalCommit)
        return self._pause_final_commit_reconciliation(
            operation,
            authority,
            intent,
            publish_error,
            recovery_error,
            resolution_mode,
            committed_result,
        )

    def _clear_commit_inflight_after_error(self) -> None:
        with self._condition:
            if not self._commit_inflight:
                return
            self._commit_inflight = False
            self._revision += 1
            self._condition.notify_all()

    def _execute_durable_commit(
        self,
        operation: CheckpointCommit[CommitT] | FinalCommit[CommitT],
        authority: _CommitAuthoritySnapshot[CommitT],
        kind: CommitKind,
        deadline: float | None,
    ) -> _DurableCommitOutcome[CommitT]:
        """Run one intent/publish/recover/marker transaction after atomic admission."""

        preparation = authority.preparation
        if preparation.kind is not kind:
            raise RuntimeError("consumed commit authority kind changed after admission")
        intent = CommitIntent(
            kind=preparation.kind,
            commit_id=preparation.commit_id,
            run_id=preparation.subject.run_id,
            safety_bundle_id=preparation.subject.safety_bundle_id,
            target=preparation.target,
            created_at=commit_now(),
        )
        try:
            authority.journal.begin(intent)
        except BaseException as begin_error:
            try:
                if any(value == intent for value in authority.journal.pending()):
                    authority.journal.mark_aborted(intent.commit_id)
            except BaseException as reconciliation_error:
                pending = self._pause_durable_commit(
                    kind,
                    operation,
                    authority,
                    intent,
                    begin_error,
                    reconciliation_error,
                    _CommitResolutionMode.FORCE_ABORT,
                )
                raise pending from reconciliation_error
            raise

        try:
            with self._condition:
                self._token.checkpoint()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"run {self.run_id} exceeded its monotonic deadline"
                    )
                if kind is CommitKind.FINAL:
                    self._cancel_gate_closed = True
                    self._phase = "committing"
                else:
                    self._phase = "publishing-checkpoint"
                self._revision += 1
                self._condition.notify_all()
        except BaseException as gate_error:
            try:
                authority.journal.mark_aborted(intent.commit_id)
            except BaseException as reconciliation_error:
                pending = self._pause_durable_commit(
                    kind,
                    operation,
                    authority,
                    intent,
                    gate_error,
                    reconciliation_error,
                    _CommitResolutionMode.FORCE_ABORT,
                )
                raise pending from reconciliation_error
            raise

        publish_error: BaseException | None = None
        recovered_error: BaseException | None = None
        try:
            committed = authority.publish_validated(intent)
        except PublishVisibilityUnknown as visibility_error:
            publish_error = visibility_error
            try:
                recovery = authority.recover_validated(intent)
            except BaseException as recovery_error:
                pending = self._pause_durable_commit(
                    kind,
                    operation,
                    authority,
                    intent,
                    visibility_error,
                    recovery_error,
                    _CommitResolutionMode.RECOVER_VISIBILITY,
                )
                raise pending from recovery_error
            if not recovery.committed:
                try:
                    authority.journal.mark_aborted(intent.commit_id)
                except BaseException as marker_error:
                    pending = self._pause_durable_commit(
                        kind,
                        operation,
                        authority,
                        intent,
                        visibility_error,
                        marker_error,
                        _CommitResolutionMode.FORCE_ABORT,
                    )
                    raise pending from marker_error
                if kind is CommitKind.CHECKPOINT:
                    self._token.checkpoint()
                raise visibility_error
            assert recovery.result is not None
            committed = recovery.result.result
            recovered_error = visibility_error
        except BaseException as deterministic_publish_error:
            try:
                authority.journal.mark_aborted(intent.commit_id)
            except BaseException as marker_error:
                pending = self._pause_durable_commit(
                    kind,
                    operation,
                    authority,
                    intent,
                    deterministic_publish_error,
                    marker_error,
                    _CommitResolutionMode.FORCE_ABORT,
                )
                raise pending from marker_error
            raise

        try:
            authority.journal.mark_committed(intent.commit_id)
        except BaseException as marker_error:
            pending = self._pause_durable_commit(
                kind,
                operation,
                authority,
                intent,
                publish_error or marker_error,
                marker_error,
                _CommitResolutionMode.FORCE_COMMIT,
                committed,
            )
            raise pending from marker_error
        return _DurableCommitOutcome(committed, recovered_error)

    def _reconcile_durable_commit(
        self,
        pending: CheckpointReconciliationRequired | FinalCommitReconciliationRequired,
        kind: CommitKind,
    ) -> _DurableReconciliationOutcome:
        """Resolve repository fact and its durable marker without re-running publish/finalize."""

        operation = pending.operation
        authority = pending._authority_snapshot
        mode = pending._resolution_mode
        committed_result = pending._committed_result
        if mode is _CommitResolutionMode.RECOVER_VISIBILITY:
            try:
                recovery = authority.recover_validated(pending.intent)
            except BaseException as recovery_error:
                refreshed = self._pause_durable_commit(
                    kind,
                    operation,
                    authority,
                    pending.intent,
                    pending.publish_error,
                    recovery_error,
                    _CommitResolutionMode.RECOVER_VISIBILITY,
                )
                raise refreshed from recovery_error
            if recovery.committed:
                assert recovery.result is not None
                committed_result = recovery.result.result
                mode = _CommitResolutionMode.FORCE_COMMIT
            else:
                mode = _CommitResolutionMode.FORCE_ABORT

        try:
            authority.journal.begin(pending.intent)
            if mode is _CommitResolutionMode.FORCE_COMMIT:
                if committed_result is _NO_COMMITTED_RESULT:
                    raise RuntimeError("commit reconciliation lost its validated result")
                authority.journal.mark_committed(pending.intent.commit_id)
            elif mode is _CommitResolutionMode.FORCE_ABORT:
                authority.journal.mark_aborted(pending.intent.commit_id)
            else:  # pragma: no cover - exhaustive enum guard
                raise RuntimeError(f"unknown commit resolution mode {mode!r}")
        except BaseException as marker_error:
            refreshed = self._pause_durable_commit(
                kind,
                operation,
                authority,
                pending.intent,
                pending.publish_error,
                marker_error,
                mode,
                committed_result,
            )
            raise refreshed from marker_error
        return _DurableReconciliationOutcome(
            committed=mode is _CommitResolutionMode.FORCE_COMMIT,
            result=committed_result,
        )

    def _commit_checkpoint(
        self,
        operation: CheckpointCommit[CommitT],
        *,
        deadline: float | None,
    ) -> CommitT:
        authority = self._admit_commit(operation, CommitKind.CHECKPOINT, deadline)
        try:
            outcome = self._execute_durable_commit(
                operation,
                authority,
                CommitKind.CHECKPOINT,
                deadline,
            )
        except BaseException:
            self._clear_commit_inflight_after_error()
            raise

        with self._condition:
            self._commit_inflight = False
            self._pending_checkpoint = None
            self._committed_checkpoint = CheckpointSnapshot(
                authority.preparation.commit_id,
                authority.target,
                CheckpointDisposition.COMMITTED,
            )
            self._phase = "post-checkpoint-analysis"
            if outcome.recovered_error is not None:
                self._commit_recovery_warning = _format_error(outcome.recovered_error)
            self._revision += 1
            self._condition.notify_all()
        self._token.checkpoint()
        return outcome.result

    def _commit_final(
        self,
        operation: FinalCommit[CommitT],
        *,
        deadline: float | None,
    ) -> CommitT:
        authority = self._admit_commit(operation, CommitKind.FINAL, deadline)
        try:
            outcome = self._execute_durable_commit(
                operation,
                authority,
                CommitKind.FINAL,
                deadline,
            )
        except BaseException:
            self._clear_commit_inflight_after_error()
            raise
        with self._condition:
            self._commit_inflight = False
            self._final_committed = True
            self._pending_final_commit = None
            self._result = outcome.result
            if outcome.recovered_error is not None:
                self._commit_recovery_warning = _format_error(outcome.recovered_error)
            self._revision += 1
            self._condition.notify_all()
            return outcome.result

    def _reconcile_final_commit(self) -> object:
        with self._condition:
            if threading.current_thread() is not self._owner_thread:
                raise CapabilityRevoked(
                    "final commit reconciliation belongs to the authoritative Run owner thread"
                )
            pending = self._pending_final_commit
            if pending is None:
                raise RuntimeError("Run has no pending final commit reconciliation")
            if self._commit_inflight:
                raise RuntimeError("final commit reconciliation is already in flight")
            self._commit_inflight = True
            self._phase = "reconciling-final-commit"
            self._revision += 1
            self._condition.notify_all()
        try:
            outcome = self._reconcile_durable_commit(pending, CommitKind.FINAL)
        except BaseException:
            self._clear_commit_inflight_after_error()
            raise
        if not outcome.committed:
            with self._condition:
                self._commit_inflight = False
                self._pending_final_commit = None
                self._revision += 1
                self._condition.notify_all()
            raise pending.publish_error
        with self._condition:
            self._commit_inflight = False
            self._pending_final_commit = None
            self._final_committed = True
            self._result = outcome.result
            self._commit_recovery_warning = _format_error(pending.publish_error)
            self._revision += 1
            self._condition.notify_all()
            return outcome.result

    def _reconcile_checkpoint(self) -> object:
        with self._condition:
            if threading.current_thread() is not self._owner_thread:
                raise CapabilityRevoked(
                    "checkpoint reconciliation belongs to the authoritative Run owner thread"
                )
            pending = self._pending_checkpoint
            if pending is None:
                raise RuntimeError("Run has no pending checkpoint reconciliation")
            if self._commit_inflight:
                raise RuntimeError("checkpoint reconciliation is already in flight")
            self._commit_inflight = True
            self._phase = "reconciling-checkpoint"
            self._revision += 1
            self._condition.notify_all()
        try:
            outcome = self._reconcile_durable_commit(pending, CommitKind.CHECKPOINT)
        except BaseException:
            self._clear_commit_inflight_after_error()
            raise
        if not outcome.committed:
            with self._condition:
                self._commit_inflight = False
                self._pending_checkpoint = None
                self._committed_checkpoint = None
                self._revision += 1
                self._condition.notify_all()
            raise pending.publish_error
        with self._condition:
            self._commit_inflight = False
            self._pending_checkpoint = None
            self._committed_checkpoint = CheckpointSnapshot(
                pending.intent.commit_id,
                pending._authority_snapshot.target,
                CheckpointDisposition.COMMITTED,
            )
            self._commit_recovery_warning = _format_error(pending.publish_error)
            self._revision += 1
            self._condition.notify_all()
            return outcome.result

    def _begin_cleanup_after_interrupt(self) -> None:
        with self._condition:
            self._interrupt_enabled = False
            while self._interrupt_inflight:
                self._condition.wait()
            self._cleanup_started = True

    def _mark_post_safety(self, bundle: SafetyDispositionBundle | None) -> None:
        with self._condition:
            self._interrupt_enabled = False
            while self._interrupt_inflight:
                self._condition.wait()
            self._safety_bundle_id = None if bundle is None else bundle.bundle_id
            self._revision += 1
            self._condition.notify_all()

    def _interrupt_error_snapshot(self) -> tuple[BaseException, ...]:
        with self._condition:
            return tuple(self._interrupt_errors)

    def _install_retry(
        self,
        retry: Callable[[], None],
        instruction: str,
        cleanup_errors: tuple[BaseException, ...],
        *,
        phase: str = "safety-journal-failed",
        state: RunState | None = None,
    ) -> None:
        with self._condition:
            self._state = (
                state
                if state is not None
                else (
                    RunState.CANCELLING
                    if self._token.is_cancelled
                    else RunState.RUNNING
                )
            )
            self._phase = phase
            self._retry_disposition = retry
            self._retry_phase = phase
            self._retry_state = self._state
            self._recovery_instruction = instruction
            self._cleanup_errors = cleanup_errors
            self._revision += 1
            self._condition.notify_all()

    def _pending_reconciliation_snapshot(
        self,
    ) -> tuple[
        CheckpointReconciliationRequired | None,
        FinalCommitReconciliationRequired | None,
    ]:
        with self._condition:
            return self._pending_checkpoint, self._pending_final_commit

    def _terminal_publication(
        self,
        *,
        state: RunState,
        result: object,
        primary: BaseException | None,
        cleanup_errors: tuple[BaseException, ...],
    ) -> TerminalPublication:
        if not state.terminal:
            raise ValueError("finish requires a terminal state")

        def publish() -> None:
            self._publish_terminal(
                state=state,
                result=result,
                primary=primary,
                cleanup_errors=cleanup_errors,
            )

        return _mint_terminal_publication(publish, lambda: self._on_terminal(self))

    def _publish_terminal(
        self,
        *,
        state: RunState,
        result: object,
        primary: BaseException | None,
        cleanup_errors: tuple[BaseException, ...],
    ) -> None:
        with self._condition:
            if self._state.terminal:
                raise RuntimeError("run terminal state may be published only once")
            if (
                self._commit_inflight
                or self._pending_checkpoint is not None
                or self._pending_final_commit is not None
            ):
                raise RuntimeError(
                    "cannot publish terminal state while durable commit reconciliation is pending"
                )
            self._cancel_gate_closed = True
            if self._final_committed:
                if primary is not None:
                    cleanup_errors = cleanup_errors + (primary,)
                    primary = None
                state = RunState.SUCCEEDED
                result = self._result
            elif state is RunState.SUCCEEDED and self._token.is_cancelled:
                state = RunState.CANCELLED
            self._state = state
            self._phase = "terminal"
            self._result = result
            self._primary = primary
            self._cleanup_errors = cleanup_errors
            self._retry_disposition = None
            self._recovery_instruction = None
            self._interrupt = None
            self._cancel_sealer = None
            self._interrupt_thread = None
            self._revision += 1
            self._condition.notify_all()


class RunController:
    """The sole lifecycle owner for every user-startable flat RunPlan."""

    def __init__(
        self,
        resources: ResourceArbiter,
        *,
        terminal_history_limit: int = 256,
    ) -> None:
        if not isinstance(resources, ResourceArbiter):
            raise TypeError("resources must be ResourceArbiter")
        if (
            isinstance(terminal_history_limit, bool)
            or not isinstance(terminal_history_limit, int)
            or terminal_history_limit < 0
        ):
            raise ValueError("terminal_history_limit must be a non-negative integer")
        self._resources = resources
        self._terminal_history_limit = terminal_history_limit
        self._lock = threading.Lock()
        self._handles: dict[RunId, RunHandle] = {}
        self._terminal_snapshots: dict[RunId, RunSnapshot] = {}

    def start(self, plan: RunPlan[PreparedT, ResultT]) -> RunHandle[ResultT]:
        if not isinstance(plan, RunPlan):
            raise TypeError("plan must be RunPlan")
        run_id = RunId(uuid.uuid4().hex)
        acquired = self._resources.acquire_all(run_id.value, plan.resource_claims)
        if isinstance(acquired, (ResourceBusy, ResourceQuarantined)):
            raise RunStartRejected(acquired)
        token = CancellationToken()
        handle: RunHandle[ResultT] = RunHandle(run_id, token, self._on_terminal)
        deadline = (
            None
            if plan.timeout_seconds is None
            else time.monotonic() + plan.timeout_seconds
        )
        try:
            context = RunContext(handle, token, deadline, plan.bound_devices)
        except BaseException:
            acquired._release_unarmed()
            raise
        handle._bind_cancel_sealer(context._seal_execution_on_cancel)
        thread = threading.Thread(
            target=self._run_owner,
            args=(plan, acquired, handle, context),
            name=f"zlc-run-{run_id.value[:12]}",
            daemon=False,
        )
        interrupt = (
            None
            if not plan.interrupt_operations
            else lambda: context._run_interrupts(plan.interrupt_operations)
        )
        handle._bind_owner(thread, interrupt)
        with self._lock:
            self._handles[run_id] = handle
        try:
            thread.start()
        except BaseException as start_exc:
            context._revoke_hardware_before_safety()
            acquired._commit_safety(())
            context._mark_safety_durable(acquired.safety_bundle)

            publication = handle._terminal_publication(
                    state=RunState.FAILED,
                    result=_MISSING,
                    primary=start_exc,
                    cleanup_errors=(),
            )
            acquired.release_terminal(publication, disposition="FAILED")
            raise
        return handle

    def run(
        self,
        plan: RunPlan[PreparedT, ResultT],
        *,
        cancel_join_timeout: float = 5.0,
    ) -> ResultT:
        cancel_join_timeout = _validate_timeout(
            cancel_join_timeout, "cancel_join_timeout"
        )
        assert cancel_join_timeout is not None
        handle = self.start(plan)
        try:
            return handle.result()
        except KeyboardInterrupt as interrupt:
            handle.cancel("notebook KeyboardInterrupt")
            try:
                snapshot = handle.wait(cancel_join_timeout)
            except TimeoutError as exc:
                raise RunStillCancelling(handle) from exc
            if snapshot.state is RunState.FAILED:
                raise RunFailed(snapshot, handle._primary) from interrupt
            if snapshot.state is RunState.SUCCEEDED:
                return handle.result()
            raise

    def lookup(self, run_id: RunId) -> RunHandle | RunSnapshot:
        with self._lock:
            active = self._handles.get(run_id)
            terminal = self._terminal_snapshots.get(run_id)
        if active is not None:
            return active
        if terminal is not None:
            return terminal
        raise KeyError(run_id)

    def snapshots(self) -> tuple[RunSnapshot, ...]:
        with self._lock:
            handles = tuple(self._handles.values())
            terminal = tuple(self._terminal_snapshots.values())
        return tuple(handle.snapshot() for handle in handles) + terminal

    def forget_terminal(self, run_id: RunId) -> bool:
        with self._lock:
            if run_id in self._handles:
                raise RuntimeError("cannot forget a non-terminal RunHandle")
            return self._terminal_snapshots.pop(run_id, None) is not None

    def _on_terminal(self, finished: RunHandle) -> None:
        snapshot = finished.snapshot()
        with self._lock:
            self._handles.pop(finished.run_id, None)
            self._terminal_snapshots[finished.run_id] = snapshot
            excess = len(self._terminal_snapshots) - self._terminal_history_limit
            for run_id in tuple(self._terminal_snapshots)[: max(0, excess)]:
                self._terminal_snapshots.pop(run_id, None)

    @staticmethod
    def _run_owner(
        plan: RunPlan[PreparedT, ResultT],
        lease: ResourceLease,
        handle: RunHandle[ResultT],
        context: RunContext,
    ) -> None:
        try:
            context.set_phase("hazard-journal")
            lease.activate_hazards(plan.hazard_claims)
        except HazardEpochExpired as exc:
            context._revoke_hardware_before_safety()
            bundle = lease._commit_safety(())
            context._mark_safety_durable(bundle)
            handle._mark_post_safety(bundle)
            publication = handle._terminal_publication(
                state=RunState.FAILED,
                result=_MISSING,
                primary=exc,
                cleanup_errors=(),
            )
            lease.release_terminal(publication, disposition=RunState.FAILED.value)
            return
        except QuarantineJournalError as exc:
            def retry_start() -> None:
                lease.activate_hazards(plan.hazard_claims)
                RunController._run_after_hazards(plan, lease, handle, context)

            handle._install_retry(
                retry_start,
                "repair the authoritative safety journal, then retry before hardware preflight",
                (exc,),
            )
            return
        RunController._run_after_hazards(plan, lease, handle, context)

    @staticmethod
    def _run_after_hazards(
        plan: RunPlan[PreparedT, ResultT],
        lease: ResourceLease,
        handle: RunHandle[ResultT],
        context: RunContext,
    ) -> None:
        hardware_enabled = context._enable_hardware()
        if hardware_enabled:
            handle._enable_interrupt()

        prepared: PreparedT | None = None
        result: object = _MISSING
        primary: BaseException | None = None
        try:
            context.set_phase("preflight")
            context.checkpoint()
            prepared = plan.preflight(context)
            context.checkpoint()
            context.set_phase("execute")
            result = plan.execute(context, prepared)
            if handle._token.is_cancelled and not handle.snapshot().final_committed:
                handle._token.checkpoint()
        except BaseException as exc:
            primary = exc

        handle._begin_cleanup_after_interrupt()
        context._enter_cleanup_hardware()
        try:
            context.set_phase("cleanup")
            report = plan.cleanup(context, prepared, primary)
            if not isinstance(report, CleanupReport):
                raise TypeError("RunPlan.cleanup must return CleanupReport")
        except BaseException as cleanup_exc:
            report = CleanupReport.unsafe(
                tuple(hazard.key for hazard in plan.hazard_claims),
                reason="cleanup raised before safety could be confirmed",
                recovery_action="verify device identity, generation, health, and safe state",
                errors=(cleanup_exc,),
            )

        decisions, coverage_errors = _resolve_cleanup_decisions(
            context,
            report,
            tuple(hazard.key for hazard in plan.hazard_claims),
        )
        cleanup_errors = (
            report.errors
            + coverage_errors
            + handle._interrupt_error_snapshot()
        )
        if primary is None and handle._token.is_cancelled and not handle.snapshot().final_committed:
            primary = CancellationRequested(handle._token.snapshot().reason)

        def continue_after_safety(bundle: SafetyDispositionBundle | None) -> None:
            nonlocal primary, result
            context._mark_safety_durable(bundle)
            handle._mark_post_safety(bundle)
            post_safety = context._post_safety_context()
            unsafe = tuple(
                decision.key
                for decision in decisions
                if decision.outcome is SafetyOutcome.UNSAFE
            )

            def finish_terminal() -> None:
                nonlocal primary
                if (
                    primary is None
                    and handle._token.is_cancelled
                    and not handle.snapshot().final_committed
                ):
                    primary = CancellationRequested(handle._token.snapshot().reason)
                state = _terminal_state(primary, cleanup_errors, unsafe)
                publication = handle._terminal_publication(
                    state=state,
                    result=result,
                    primary=primary,
                    cleanup_errors=cleanup_errors,
                )
                lease.release_terminal(publication, disposition=state.value)

            if primary is None and handle._token.is_cancelled and not handle.snapshot().final_committed:
                primary = CancellationRequested(handle._token.snapshot().reason)
            if primary is None and not cleanup_errors and not unsafe:
                finalize_error: BaseException | None = None
                try:
                    context.set_phase("finalize")
                    post_safety.checkpoint()
                    result = plan.finalize(post_safety, result)  # type: ignore[arg-type, assignment]
                except BaseException as exc:
                    finalize_error = exc

                checkpoint_pending, final_pending = handle._pending_reconciliation_snapshot()
                if checkpoint_pending is not None and final_pending is not None:
                    primary = RuntimeError(
                        "Run has conflicting checkpoint and final commit reconciliations"
                    )
                elif checkpoint_pending is not None:
                    preserved_finalize_error = (
                        None
                        if isinstance(finalize_error, CheckpointReconciliationRequired)
                        else finalize_error
                    )

                    def retry_checkpoint_reconciliation() -> None:
                        nonlocal primary
                        try:
                            handle._reconcile_checkpoint()
                        except CheckpointReconciliationRequired:
                            raise
                        except BaseException as definitive_failure:
                            primary = (
                                CancellationRequested(handle._token.snapshot().reason)
                                if handle._token.is_cancelled
                                else preserved_finalize_error or definitive_failure
                            )
                        else:
                            primary = (
                                CancellationRequested(handle._token.snapshot().reason)
                                if handle._token.is_cancelled
                                else (
                                    preserved_finalize_error
                                    or RuntimeError(
                                        "checkpoint reconciliation completed after finalize was "
                                        "interrupted; restart analysis from the committed checkpoint"
                                    )
                                )
                            )
                        finish_terminal()

                    checkpoint_state = (
                        RunState.CANCELLING
                        if handle._token.is_cancelled
                        else RunState.RUNNING
                    )
                    handle._install_retry(
                        retry_checkpoint_reconciliation,
                        "reconcile checkpoint visibility only; finalize will not be re-entered",
                        cleanup_errors + (checkpoint_pending.recovery_error,),
                        phase="checkpoint-reconciliation-failed",
                        state=checkpoint_state,
                    )
                    return
                elif final_pending is not None:
                    preserved_finalize_error = (
                        None
                        if isinstance(finalize_error, FinalCommitReconciliationRequired)
                        else finalize_error
                    )

                    def retry_commit_reconciliation() -> None:
                        nonlocal primary, result
                        if primary is None and not handle.snapshot().final_committed:
                            try:
                                result = handle._reconcile_final_commit()
                            except FinalCommitReconciliationRequired:
                                raise
                            except BaseException as definitive_failure:
                                primary = preserved_finalize_error or definitive_failure
                            else:
                                primary = preserved_finalize_error
                        finish_terminal()

                    handle._install_retry(
                        retry_commit_reconciliation,
                        "reconcile final artifact visibility with the repository, then publish terminal state",
                        cleanup_errors + (final_pending.recovery_error,),
                        phase="final-commit-reconciliation-failed",
                        state=RunState.RUNNING,
                    )
                    return
                elif finalize_error is not None:
                    primary = finalize_error
                elif plan.requires_final_commit and not handle.snapshot().final_committed:
                    primary = RuntimeError(
                        "RunPlan requires a final artifact commit but finalize published none"
                    )
            finish_terminal()

        try:
            context.set_phase("finalizing-safety")
            context._revoke_hardware_before_safety()
            bundle = lease._commit_safety(decisions)
        except QuarantineJournalError as exc:
            errors = cleanup_errors + (exc,)

            def retry_safety() -> None:
                bundle = lease._commit_safety(decisions)
                continue_after_safety(bundle)

            handle._install_retry(
                retry_safety,
                "repair the authoritative safety journal, then retry the same safety bundle",
                errors,
            )
            return
        continue_after_safety(bundle)


def _resolve_cleanup_decisions(
    context: RunContext,
    report: CleanupReport,
    hazard_keys: tuple[ResourceKey, ...],
) -> tuple[tuple[SafetyDecision, ...], tuple[BaseException, ...]]:
    expected = set(hazard_keys)
    try:
        receipts = context._consume_safety_proofs(report.safety_proofs)
    except BaseException as exc:
        receipts = ()
        proof_errors: tuple[BaseException, ...] = (exc,)
    else:
        proof_errors = ()
    safe_decisions = tuple(SafetyDecision.safe(receipt) for receipt in receipts)
    if len({decision.key for decision in safe_decisions}) != len(safe_decisions):
        proof_errors += (ValueError("cleanup returned multiple safety proofs for one resource"),)
    decisions = {decision.key: decision for decision in report.decisions}
    for decision in safe_decisions:
        if decision.key in decisions:
            proof_errors += (
                ValueError(f"cleanup returned both SAFE and UNSAFE for {decision.key}"),
            )
        else:
            decisions[decision.key] = decision
    extras = set(decisions) - expected
    missing = expected - set(decisions)
    errors: list[BaseException] = list(proof_errors)
    if extras:
        errors.append(
            ValueError(
                "cleanup reported undeclared hazard resources: "
                + ", ".join(str(key) for key in sorted(extras))
            )
        )
        for key in extras:
            decisions.pop(key, None)
    for key in missing:
        errors.append(RuntimeError(f"cleanup omitted safety disposition for {key}"))
        decisions[key] = SafetyDecision.unsafe(
            key,
            reason="hardware safety acknowledgement was omitted",
            recovery_action="verify device identity, generation, health, and safe state",
        )
    return tuple(decisions[key] for key in sorted(decisions)), tuple(errors)


def _terminal_state(
    primary: BaseException | None,
    cleanup_errors: tuple[BaseException, ...],
    unsafe_keys: tuple[ResourceKey, ...],
) -> RunState:
    if cleanup_errors or unsafe_keys:
        return RunState.FAILED
    if primary is None:
        return RunState.SUCCEEDED
    if isinstance(primary, CancellationRequested):
        return RunState.CANCELLED
    return RunState.FAILED


def _format_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"
