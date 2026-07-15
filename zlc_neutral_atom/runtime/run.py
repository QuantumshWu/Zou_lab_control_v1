"""Flat synchronous Run lifecycle hosted by one authoritative owner thread."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar

from zlc_storage import canonical_text as _canonical_text, finite_real

from ._failure import detach_failure, record_secondary_failure, safe_error_summary
from .cancellation import CancellationRequested, CancellationToken, _CancellationSource
from .cleanup import CleanupReport, SafetyProof, _mint_safety_proof
from .commit import (
    CommitIntent,
    FinalCommit,
    PublishVisibilityUnknown,
    _CommitAuthoritySnapshot,
    _consume_commit_authority,
    _discard_commit_authority,
)
from .ports import (
    BoundDevice,
    CleanupDevice,
    CleanupStepAck,
    RunDevice,
    SafetyInterrupt,
    SafetyOperation,
    SessionClosedAck,
    _open_device_run,
)
from .resources import (
    ClaimMode,
    HazardClaim,
    HazardEpochExpired,
    SafetyJournalWriteError,
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
ExecutedT = TypeVar("ExecutedT")
FinalT = TypeVar("FinalT")
CommitT = TypeVar("CommitT")
_MISSING = object()
_NO_COMMITTED_RESULT = object()


@dataclass(frozen=True, order=True)
class RunId:
    value: str

    def __post_init__(self) -> None:
        _canonical_text(self.value, "RunId")

    def __str__(self) -> str:
        return self.value


class RunState(str, Enum):
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in (self.SUCCEEDED, self.FAILED, self.CANCELLED)


class CancelOutcome(str, Enum):
    REQUESTED = "REQUESTED"
    ALREADY_REQUESTED = "ALREADY_REQUESTED"
    TOO_LATE_ALREADY_COMMITTED = "TOO_LATE_ALREADY_COMMITTED"
    TOO_LATE_FINALIZING = "TOO_LATE_FINALIZING"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"


@dataclass(frozen=True)
class RunSnapshot:
    run_id: RunId
    state: RunState
    phase: str
    final_committed: bool
    safety_bundle_id: str | None
    commit_recovery_warning: str | None
    primary_error: str | None
    cleanup_errors: tuple[str, ...]
    recovery_instruction: str | None


@dataclass(frozen=True)
class RunPlan(Generic[PreparedT, ExecutedT, FinalT]):
    """One finite execution with distinct preparation, execution, and final values."""

    name: str
    resource_claims: tuple[ResourceClaim, ...]
    bound_devices: tuple[BoundDevice, ...]
    preflight: Callable[["RunContext"], PreparedT]
    execute: Callable[["RunContext", PreparedT], ExecutedT]
    cleanup: Callable[["RunContext", PreparedT | None, BaseException | None], CleanupReport]
    finalize: Callable[["PostSafetyContext", ExecutedT], FinalT]
    interrupt_operations: tuple[SafetyInterrupt, ...] = ()
    timeout_seconds: float | None = None
    requires_final_commit: bool = False
    dispose_unfinalized: Callable[[ExecutedT], None] | None = None

    def __post_init__(self) -> None:
        _canonical_text(self.name, "run plan name")
        claims = tuple(self.resource_claims)
        if any(not isinstance(claim, ResourceClaim) for claim in claims):
            raise TypeError("resource_claims must contain ResourceClaim values")
        object.__setattr__(self, "resource_claims", claims)
        devices = tuple(self.bound_devices)
        if any(not isinstance(device, BoundDevice) for device in devices):
            raise TypeError("bound_devices must contain BoundDevice values")
        if len({device.key for device in devices}) != len(devices):
            raise ValueError("bound_devices must have unique ResourceKeys")
        exclusive_devices = {
            claim.key
            for claim in claims
            if claim.mode is ClaimMode.EXCLUSIVE and claim.key.segments[0] == "device"
        }
        if {device.key for device in devices} != exclusive_devices:
            raise ValueError("every EXCLUSIVE device claim requires exactly one BoundDevice")
        object.__setattr__(self, "bound_devices", devices)
        interrupts = tuple(self.interrupt_operations)
        if any(not isinstance(value, SafetyInterrupt) for value in interrupts):
            raise TypeError("interrupt_operations must contain SafetyInterrupt values")
        device_by_key = {device.key: device for device in devices}
        for interrupt in interrupts:
            device = device_by_key.get(interrupt.key)
            if device is None:
                raise ValueError("interrupt operation must target a bound device")
            if interrupt.operation not in device.interrupt_capabilities:
                raise ValueError(
                    f"device {interrupt.key} does not bind thread-safe interrupt "
                    f"{interrupt.operation.value}"
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
        if self.dispose_unfinalized is not None and not callable(
            self.dispose_unfinalized
        ):
            raise TypeError("dispose_unfinalized must be callable or None")
        if not isinstance(self.requires_final_commit, bool):
            raise TypeError("requires_final_commit must be bool")
        object.__setattr__(
            self,
            "timeout_seconds",
            None
            if self.timeout_seconds is None
            else finite_real(self.timeout_seconds, "timeout_seconds", minimum=0.0),
        )


class CapabilityRevoked(RuntimeError):
    pass


class _PermanentCommitRejection(CapabilityRevoked):
    pass


class RunStartRejected(RuntimeError):
    def __init__(self, outcome: ResourceBusy | ResourceQuarantined) -> None:
        super().__init__(f"run start rejected: {outcome}")
        self.outcome = outcome


class RunFailed(RuntimeError):
    def __init__(self, snapshot: RunSnapshot) -> None:
        super().__init__(snapshot.primary_error or "run failed during cleanup")
        self.snapshot = snapshot


class RunCancelled(RuntimeError):
    def __init__(self, snapshot: RunSnapshot) -> None:
        super().__init__(f"run {snapshot.run_id} was cancelled")
        self.snapshot = snapshot


class RunStillCancelling(RuntimeError):
    def __init__(self, handle: "RunHandle") -> None:
        super().__init__(f"run {handle.run_id} is still cancelling; retain its RunHandle")
        self.handle = handle


class _CommitResolutionMode(Enum):
    FORCE_ABORT = "force-abort"
    RECOVER_VISIBILITY = "recover-visibility"
    FORCE_COMMIT = "force-commit"


@dataclass
class _PendingCommit:
    reconciliation_id: str
    authority: _CommitAuthoritySnapshot
    intent: CommitIntent
    mode: _CommitResolutionMode
    publish_summary: str
    last_error_summary: str
    committed_result: object = _NO_COMMITTED_RESULT


class _CommitRetryRequired(RuntimeError):
    """Opaque control signal; live commit authority remains private to RunHandle."""

    def __init__(self, reconciliation_id: str, summary: str) -> None:
        self.reconciliation_id = reconciliation_id
        self.summary = summary
        super().__init__(
            f"commit reconciliation {reconciliation_id} requires an explicit retry: {summary}"
        )


class _ExecutionOwnership:
    """Mutable hand-off cell so outer owner frames never pin the live plan graph."""

    __slots__ = ("plan", "context")

    def __init__(self, plan: RunPlan, context: "RunContext") -> None:
        self.plan: RunPlan | None = plan
        self.context: RunContext | None = context

    def take(self) -> tuple[RunPlan, "RunContext"]:
        plan = self.plan
        context = self.context
        if plan is None or context is None:
            raise RuntimeError("Run execution ownership was already consumed")
        return plan, context

    def clear(self) -> None:
        self.plan = None
        self.context = None


class _FinalizationInput:
    """Exactly-once owner for an execute result until finalize accepts it."""

    __slots__ = ("_value", "_disposer")

    def __init__(
        self,
        value: object,
        disposer: Callable[[object], None] | None,
    ) -> None:
        self._value = value
        self._disposer = disposer

    def take(self) -> object:
        value = self._value
        if value is _MISSING:
            raise RuntimeError("finalization input was already consumed")
        self._value = _MISSING
        self._disposer = None
        return value

    def dispose(self) -> None:
        value = self._value
        if value is _MISSING:
            return
        disposer = self._disposer
        self._value = _MISSING
        self._disposer = None
        if disposer is not None:
            disposer(value)


class RunContext:
    """Execution/cleanup capability. Construction is pure; owner binding is explicit."""

    def __init__(
        self,
        handle: "RunHandle",
        cancellation: CancellationToken,
        deadline: float | None,
        bound_devices: tuple[BoundDevice, ...],
    ) -> None:
        self._handle: RunHandle | None = handle
        self.cancellation = cancellation
        self.deadline = deadline
        self._pending_devices = tuple(bound_devices)
        self._devices: dict[ResourceKey, BoundDevice] = {}
        self._device_lease = None
        self._hardware_condition = threading.Condition(threading.RLock())
        self._execution_enabled = False
        self._execution_sealed = False
        self._cleanup_enabled = False
        self._hardware_revoked = False
        self._hardware_inflight = 0
        self._interrupt_enabled = False
        self._issued_safety_proofs: dict[object, tuple[SafetyProof, SafeReceipt]] = {}

    @property
    def run_id(self) -> RunId:
        handle = self._handle
        if handle is None:
            raise CapabilityRevoked("RunContext has crossed its safety boundary")
        return handle.run_id

    def set_phase(self, phase: str) -> None:
        handle = self._handle
        if handle is None:
            raise CapabilityRevoked("RunContext has crossed its safety boundary")
        handle._set_phase(_canonical_text(phase, "run phase"))

    def checkpoint(self) -> None:
        self.cancellation.checkpoint()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(f"run {self.run_id} exceeded its monotonic deadline")

    def _bind_devices(self) -> None:
        """Perform the potentially blocking live identity probe on the owner thread."""

        with self._hardware_condition:
            if self._device_lease is not None or self._devices:
                raise RuntimeError("Run devices are already bound")
            pending = self._pending_devices
        lease = _open_device_run(self.run_id.value, pending)
        with self._hardware_condition:
            self._device_lease = lease
            self._devices = {device.key: device for device in pending}
            self._pending_devices = ()

    def device(self, key: ResourceKey) -> RunDevice:
        if not isinstance(key, ResourceKey):
            raise TypeError("device key must be ResourceKey")
        try:
            return RunDevice(self, self._devices[key])
        except KeyError as exc:
            raise CapabilityRevoked(f"run does not own device {key}") from exc

    def cleanup_device(self, key: ResourceKey) -> CleanupDevice:
        if not isinstance(key, ResourceKey):
            raise TypeError("device key must be ResourceKey")
        try:
            return CleanupDevice(self, self._devices[key])
        except KeyError as exc:
            raise CapabilityRevoked(f"run does not own device {key}") from exc

    def _lease(self):
        lease = self._device_lease
        if lease is None:
            raise CapabilityRevoked("Run device lease is not active")
        return lease

    def _execute_bound_device(self, binding: BoundDevice, command: object) -> object:
        with self._hardware_condition:
            if (
                not self._execution_enabled
                or self._execution_sealed
                or self._hardware_revoked
                or self.cancellation.is_cancelled
            ):
                raise CapabilityRevoked("Run has no active execution hardware capability")
            if self._devices.get(binding.key) is not binding:
                raise CapabilityRevoked("device binding does not belong to this Run")
            self._hardware_inflight += 1
        try:
            return self._lease().execute(binding, command)
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
                raise CapabilityRevoked("Run has no active cleanup hardware capability")
            if self._devices.get(binding.key) is not binding:
                raise CapabilityRevoked("device binding does not belong to this Run")
            self._hardware_inflight += 1
        try:
            return self._lease().cleanup_step(binding, operation)
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
                raise CapabilityRevoked("Run has no active cleanup hardware capability")
            if self._devices.get(binding.key) is not binding:
                raise CapabilityRevoked("device binding does not belong to this Run")
            self._hardware_inflight += 1
        try:
            return self._lease().close_session(binding, command)
        finally:
            with self._hardware_condition:
                self._hardware_inflight -= 1
                self._hardware_condition.notify_all()

    def _verify_bound_safe_state(self, binding: BoundDevice) -> SafetyProof:
        with self._hardware_condition:
            if not self._cleanup_enabled or self._hardware_revoked:
                raise CapabilityRevoked("Run has no active cleanup hardware capability")
            if self._devices.get(binding.key) is not binding:
                raise CapabilityRevoked("device binding does not belong to this Run")
            self._hardware_inflight += 1
        try:
            receipt = self._lease().verify_safe_state(binding)
            nonce = object()
            proof = _mint_safety_proof(run_id=self.run_id.value, receipt=receipt, nonce=nonce)
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
        with self._hardware_condition:
            receipts: list[SafeReceipt] = []
            for proof in tuple(proofs):
                if not isinstance(proof, SafetyProof) or proof.run_id != self.run_id.value:
                    raise ValueError("cleanup safety proof belongs to another Run")
                issued = self._issued_safety_proofs.get(proof._nonce)
                if issued is None or issued[0] is not proof:
                    raise ValueError("cleanup safety proof was not issued or was already consumed")
                receipts.append(issued[1])
            for proof in tuple(proofs):
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
                raise RuntimeError("cleanup cannot start after hardware revocation")
            self._cleanup_enabled = True

    def _run_interrupts(self, interrupts: tuple[SafetyInterrupt, ...]) -> None:
        summaries: list[str] = []
        for interrupt in interrupts:
            try:
                with self._hardware_condition:
                    if not self._interrupt_enabled or self._hardware_revoked:
                        raise CapabilityRevoked("Run interrupt capability is not active")
                    binding = self._devices.get(interrupt.key)
                    if binding is None:
                        raise CapabilityRevoked(
                            f"Run does not own interrupt device {interrupt.key}"
                        )
                    self._hardware_inflight += 1
                try:
                    self._lease().interrupt(binding, interrupt.operation)
                finally:
                    with self._hardware_condition:
                        self._hardware_inflight -= 1
                        self._hardware_condition.notify_all()
            except BaseException as error:
                summaries.append(
                    f"{interrupt.key}:{interrupt.operation.value} -> "
                    f"{safe_error_summary(error)}"
                )
                detach_failure(error, note_prefix="detached interrupt traceback")
        if summaries:
            raise RuntimeError("; ".join(summaries))

    def _revoke_hardware(self) -> None:
        with self._hardware_condition:
            self._execution_enabled = False
            self._execution_sealed = True
            self._cleanup_enabled = False
            self._interrupt_enabled = False
            self._hardware_revoked = True
            while self._hardware_inflight:
                self._hardware_condition.wait()
            lease = self._device_lease
            self._device_lease = None
            self._devices.clear()
            self._pending_devices = ()
            self._issued_safety_proofs.clear()
        if lease is not None:
            lease.revoke()

    def _detach_for_post_safety(self) -> "_PostSafetySeed":
        with self._hardware_condition:
            if not self._hardware_revoked or self._hardware_inflight:
                raise RuntimeError("hardware must be revoked before post-safety transition")
            handle = self._handle
            if handle is None:
                raise RuntimeError("RunContext already detached")
            seed = _PostSafetySeed(handle, self.cancellation, self.deadline)
            self._handle = None
            return seed


@dataclass(frozen=True)
class _PostSafetySeed:
    handle: "RunHandle"
    cancellation: CancellationToken
    deadline: float | None

    def mint(self, bundle: SafetyDispositionBundle | None) -> "PostSafetyContext":
        self.handle._mark_post_safety(bundle)
        return PostSafetyContext(
            _POST_SAFETY_CONTEXT_TOKEN,
            run_id=self.handle.run_id,
            cancellation=self.cancellation,
            deadline=self.deadline,
            safety_bundle_id=None if bundle is None else bundle.bundle_id,
            handle=self.handle,
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
        "_prepared_commits",
        "_active",
    )

    def __init__(
        self,
        token: object,
        *,
        run_id: RunId,
        cancellation: CancellationToken,
        deadline: float | None,
        safety_bundle_id: str | None,
        handle: "RunHandle",
    ) -> None:
        if token is not _POST_SAFETY_CONTEXT_TOKEN:
            raise PermissionError("PostSafetyContext is minted by RunController")
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_cancellation", cancellation)
        object.__setattr__(self, "_deadline", deadline)
        object.__setattr__(self, "_safety_bundle_id", safety_bundle_id)
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_prepared_commits", [])
        object.__setattr__(self, "_active", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PostSafetyContext is immutable")

    def _require_active(self) -> "RunHandle":
        handle = self._handle
        if not self._active or handle is None:
            raise CapabilityRevoked("PostSafetyContext has left finalize")
        return handle

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
        self._require_active()
        self.cancellation.checkpoint()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(f"run {self.run_id} exceeded its monotonic deadline")

    def authorize_commit_preparation(self) -> tuple[str, str | None]:
        self._require_active()._validate_commit_preparation(self.deadline)
        return self.run_id.value, self.safety_bundle_id

    def _track_prepared_commit(
        self,
        operation: FinalCommit[object],
    ) -> None:
        self._require_active()
        if not isinstance(operation, FinalCommit):
            raise TypeError("prepared commit tracking requires FinalCommit")
        if operation.run_id != self.run_id.value:
            raise ValueError("prepared commit belongs to another Run")
        if operation.safety_bundle_id != self.safety_bundle_id:
            raise ValueError("prepared commit safety bundle differs from this Run")
        if any(existing is operation for existing in self._prepared_commits):
            raise ValueError("prepared commit operation is already tracked")
        self._prepared_commits.append(operation)

    def _take_tracked(self, operation: object) -> None:
        for index, existing in enumerate(self._prepared_commits):
            if existing is operation:
                self._prepared_commits.pop(index)
                return

    def _abandon_unconsumed_commits(self) -> tuple[str, ...]:
        operations = tuple(self._prepared_commits)
        errors: list[str] = []
        for operation in operations:
            try:
                operation.abandon()
            except BaseException as error:
                errors.append(safe_error_summary(error))
                detach_failure(error, note_prefix="detached commit abandonment traceback")
            else:
                self._take_tracked(operation)
        return _bounded_diagnostics(tuple(errors))

    def _revoke(self) -> None:
        object.__setattr__(self, "_active", False)
        object.__setattr__(self, "_handle", None)

    def _commit_operation(
        self,
        operation: FinalCommit[CommitT],
    ) -> CommitT:
        handle = self._require_active()
        authority, rejection, discard = handle._admit_commit(
            operation,
            self.deadline,
        )
        if rejection is not None:
            if discard and handle._discard_rejected_commit(operation, rejection):
                self._take_tracked(operation)
            raise rejection
        assert authority is not None
        # _admit_commit consumed the repository capability successfully.  From
        # this point the RunHandle owns either completion or reconciliation.
        self._take_tracked(operation)
        return handle._commit_admitted(authority, self.deadline)

    def commit_final(self, operation: FinalCommit[CommitT]) -> CommitT:
        if not isinstance(operation, FinalCommit):
            raise TypeError("commit_final requires a FinalCommit operation")
        return self._commit_operation(operation)


class RunHandle(Generic[FinalT]):
    """Queryable state and the sole cancellation owner for one Run."""

    def __init__(
        self,
        run_id: RunId,
        cancellation_source: _CancellationSource,
        on_thread_joined: Callable[["RunHandle"], None],
    ) -> None:
        self.run_id = run_id
        self._cancellation_source = cancellation_source
        self._token = cancellation_source.token
        self._on_thread_joined: Callable[[RunHandle], None] | None = on_thread_joined
        self._condition = threading.Condition(threading.RLock())
        self._state = RunState.RUNNING
        self._phase = "starting"
        self._final_committed = False
        self._commit_inflight = False
        self._commit_recovery_warning: str | None = None
        self._pending_commit: _PendingCommit | None = None
        self._safety_bundle_id: str | None = None
        self._primary_error: str | None = None
        self._cleanup_errors: tuple[str, ...] = ()
        self._result: object = _MISSING
        self._interrupt: Callable[[], None] | None = None
        self._cancel_sealer: Callable[[], None] | None = None
        self._interrupt_enabled = False
        self._interrupt_errors: tuple[str, ...] = ()
        self._interrupt_inflight = False
        self._interrupt_thread: threading.Thread | None = None
        self._cancel_gate_closed = False
        self._cleanup_started = False
        self._owner_thread: threading.Thread | None = None
        self._owner_reaped = False
        self._recovery_instruction: str | None = None
        self._retry_disposition: Callable[[], None] | None = None
        self._retry_phase = "safety-journal-failed"
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
            first = self._cancellation_source.request(reason)
            if not first:
                return CancelOutcome.ALREADY_REQUESTED
            if self._cancel_sealer is not None:
                self._cancel_sealer()
            self._state = RunState.CANCELLING
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
        def invoke() -> None:
            summary: str | None = None
            try:
                interrupt()
            except BaseException as error:
                summary = safe_error_summary(error)
                detach_failure(error, note_prefix="detached interrupt traceback")
            finally:
                with self._condition:
                    if summary is not None:
                        self._interrupt_errors = _bounded_diagnostics(
                            (*self._interrupt_errors, summary)
                        )
                    self._interrupt_inflight = False
                    self._condition.notify_all()

        thread = threading.Thread(
            target=invoke,
            name=f"zlc-interrupt-{self.run_id.value[:12]}",
            daemon=False,
        )
        with self._condition:
            self._interrupt_thread = thread
        try:
            thread.start()
        except BaseException as error:
            with self._condition:
                self._interrupt_errors = _bounded_diagnostics(
                    (*self._interrupt_errors, safe_error_summary(error))
                )
                self._interrupt_inflight = False
                if self._interrupt_thread is thread:
                    self._interrupt_thread = None
                self._condition.notify_all()
            detach_failure(error, note_prefix="detached interrupt start traceback")

    def wait(self, timeout: float | None = None) -> RunSnapshot:
        timeout = None if timeout is None else finite_real(timeout, "wait timeout", minimum=0.0)
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
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
            if thread.is_alive():
                raise TimeoutError(
                    f"run {self.run_id} is terminal but its owner thread has not exited"
                )
            self._report_thread_joined(thread)
        return snapshot

    def result(self, timeout: float | None = None) -> FinalT:
        snapshot = self.wait(timeout)
        if snapshot.state is RunState.SUCCEEDED:
            return self._result  # type: ignore[return-value]
        if snapshot.state is RunState.CANCELLED:
            raise RunCancelled(snapshot)
        raise RunFailed(snapshot)

    def retry_recovery(self) -> bool:
        """Retry only an explicitly classified journal/commit ambiguity."""

        with self._recovery_lock:
            with self._condition:
                previous_owner = self._owner_thread
            if (
                previous_owner is not None
                and previous_owner is not threading.current_thread()
                and previous_owner.is_alive()
            ):
                previous_owner.join(0.25)
                if previous_owner.is_alive():
                    return False
            with self._condition:
                retry = self._retry_disposition
                if retry is None or self._state.terminal:
                    return False
                instruction = self._recovery_instruction or "retry durable recovery"
                phase = self._retry_phase
                prior = self._cleanup_errors
                self._retry_disposition = None
                self._recovery_instruction = None
                self._phase = "retrying-" + phase
                self._state = self._active_state_locked()
                self._condition.notify_all()

            launch_lock = threading.Lock()
            launch_claimed = False
            launch_cancelled = False

            def retry_owner() -> None:
                nonlocal launch_claimed
                with launch_lock:
                    if launch_cancelled:
                        return
                    launch_claimed = True
                try:
                    retry()
                except (SafetyJournalWriteError, _CommitRetryRequired) as error:
                    self._install_retry(
                        retry,
                        instruction,
                        _bounded_diagnostics((*prior, safe_error_summary(error))),
                        phase=phase,
                    )
                except BaseException as error:
                    self._block_non_retryable_recovery(
                        _bounded_diagnostics((*prior, safe_error_summary(error)))
                    )
                    detach_failure(error, note_prefix="detached recovery traceback")

            thread = threading.Thread(
                target=retry_owner,
                name=f"zlc-run-{self.run_id.value[:12]}-recovery",
                daemon=False,
            )
            with self._condition:
                self._owner_thread = thread
                self._owner_reaped = False
            try:
                thread.start()
            except BaseException as error:
                restore_retry = False
                with launch_lock:
                    if not launch_claimed:
                        launch_cancelled = True
                        restore_retry = True
                if restore_retry:
                    with self._condition:
                        if self._owner_thread is thread:
                            self._owner_thread = None
                        self._owner_reaped = False
                    # The target lost the launch claim and therefore cannot run
                    # the frozen closure.  Restore that exact closure; if the
                    # target claimed first, it remains the sole owner and this
                    # caller must not make a second copy runnable.
                    self._install_retry(
                        retry,
                        instruction,
                        _bounded_diagnostics((*prior, safe_error_summary(error))),
                        phase=phase,
                    )
                detach_failure(error, note_prefix="detached recovery start traceback")
                raise
            return True

    def _active_state_locked(self) -> RunState:
        return RunState.CANCELLING if self._token.is_cancelled else RunState.RUNNING

    def _snapshot_locked(self) -> RunSnapshot:
        return RunSnapshot(
            run_id=self.run_id,
            state=self._state,
            phase=self._phase,
            final_committed=self._final_committed,
            safety_bundle_id=self._safety_bundle_id,
            commit_recovery_warning=self._commit_recovery_warning,
            primary_error=self._primary_error,
            cleanup_errors=self._cleanup_errors,
            recovery_instruction=self._recovery_instruction,
        )

    def _bind_owner(
        self,
        thread: threading.Thread,
        interrupt: Callable[[], None] | None,
        cancel_sealer: Callable[[], None],
    ) -> None:
        with self._condition:
            if self._owner_thread is not None:
                raise RuntimeError("RunHandle owner already bound")
            self._owner_thread = thread
            self._owner_reaped = False
            self._interrupt = interrupt
            self._cancel_sealer = cancel_sealer

    def _set_phase(self, phase: str) -> None:
        with self._condition:
            if self._state.terminal:
                raise RuntimeError("cannot change phase after terminal state")
            self._phase = phase
            self._state = self._active_state_locked()
            self._condition.notify_all()

    def _enable_interrupt(self) -> bool:
        with self._condition:
            if self._cleanup_started or self._token.is_cancelled:
                return False
            self._interrupt_enabled = True
            return True

    def _begin_cleanup_after_interrupt(self) -> None:
        with self._condition:
            self._interrupt_enabled = False
            while self._interrupt_inflight:
                self._condition.wait()
            self._cleanup_started = True
            thread = self._interrupt_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._condition:
            if self._interrupt_thread is thread:
                self._interrupt_thread = None

    def _interrupt_error_snapshot(self) -> tuple[str, ...]:
        with self._condition:
            return self._interrupt_errors

    def _mark_post_safety(self, bundle: SafetyDispositionBundle | None) -> None:
        with self._condition:
            self._interrupt_enabled = False
            while self._interrupt_inflight:
                self._condition.wait()
            self._safety_bundle_id = None if bundle is None else bundle.bundle_id
            self._interrupt = None
            self._cancel_sealer = None
            self._condition.notify_all()

    def _install_retry(
        self,
        retry: Callable[[], None],
        instruction: str,
        errors: tuple[str, ...],
        *,
        phase: str = "safety-journal-failed",
    ) -> None:
        with self._condition:
            if self._state.terminal:
                return
            self._state = self._active_state_locked()
            self._phase = phase
            self._retry_disposition = retry
            self._retry_phase = phase
            self._recovery_instruction = instruction
            self._cleanup_errors = _bounded_diagnostics(errors)
            self._condition.notify_all()

    def _block_non_retryable_recovery(self, errors: tuple[str, ...]) -> None:
        with self._condition:
            self._state = self._active_state_locked()
            self._phase = "recovery-blocked-by-invariant-failure"
            self._retry_disposition = None
            self._recovery_instruction = (
                "runtime recovery failed with a non-retryable invariant error; "
                "retain this RunHandle and inspect diagnostics"
            )
            self._cleanup_errors = _bounded_diagnostics(errors)
            self._condition.notify_all()

    def _commit_gate_rejection_locked(
        self,
        deadline: float | None,
    ) -> tuple[BaseException | None, bool]:
        if self._state.terminal:
            return _PermanentCommitRejection("commit cannot run after terminal state"), True
        try:
            self._token.checkpoint()
        except BaseException as error:
            return error, True
        if deadline is not None and time.monotonic() >= deadline:
            return TimeoutError(f"run {self.run_id} exceeded its monotonic deadline"), True
        if threading.current_thread() is not self._owner_thread:
            return CapabilityRevoked("commit belongs to the Run owner thread"), False
        if self._pending_commit is not None:
            return RuntimeError("durable commit reconciliation is pending"), False
        if self._final_committed:
            return RuntimeError("final commit may be attempted only once"), True
        if self._commit_inflight:
            return RuntimeError("another commit is already in flight"), False
        if self._cancel_gate_closed:
            return RuntimeError("Run is already publishing terminal state"), True
        return None, True

    def _validate_commit_preparation(self, deadline: float | None) -> None:
        with self._condition:
            rejection, _ = self._commit_gate_rejection_locked(deadline)
        if rejection is not None:
            raise rejection

    @staticmethod
    def _discard_rejected_commit(
        operation: FinalCommit,
        rejection: BaseException,
    ) -> bool:
        try:
            _discard_commit_authority(operation)
        except BaseException as error:
            record_secondary_failure(
                rejection,
                "rejected commit authority discard also failed",
                error,
            )
            return False
        return True

    def _admit_commit(
        self,
        operation: FinalCommit[CommitT],
        deadline: float | None,
    ) -> tuple[
        _CommitAuthoritySnapshot[CommitT] | None,
        BaseException | None,
        bool,
    ]:
        rejection: BaseException | None = None
        discard = True
        authority: _CommitAuthoritySnapshot[CommitT] | None = None
        with self._condition:
            rejection, discard = self._commit_gate_rejection_locked(deadline)
            if rejection is None and operation.run_id != self.run_id.value:
                rejection = _PermanentCommitRejection("commit belongs to another Run")
            if (
                rejection is None
                and operation.safety_bundle_id != self._safety_bundle_id
            ):
                rejection = _PermanentCommitRejection("commit safety bundle differs")
            if rejection is None:
                try:
                    authority = _consume_commit_authority(operation)
                except BaseException as error:
                    rejection = error
                else:
                    self._commit_inflight = True
                    self._phase = "preparing-commit-intent"
                    self._condition.notify_all()
        return authority, rejection, discard

    def _pause_commit(
        self,
        *,
        authority: _CommitAuthoritySnapshot,
        intent: CommitIntent,
        mode: _CommitResolutionMode,
        publish_error: BaseException | str,
        recovery_error: BaseException | str,
        committed_result: object = _NO_COMMITTED_RESULT,
    ) -> _CommitRetryRequired:
        publish_summary = (
            publish_error if isinstance(publish_error, str) else safe_error_summary(publish_error)
        )
        recovery_summary = (
            recovery_error
            if isinstance(recovery_error, str)
            else safe_error_summary(recovery_error)
        )
        if isinstance(publish_error, BaseException):
            detach_failure(publish_error, note_prefix="detached commit traceback")
        if isinstance(recovery_error, BaseException):
            detach_failure(recovery_error, note_prefix="detached commit recovery traceback")
        with self._condition:
            existing = self._pending_commit
            reconciliation_id = (
                existing.reconciliation_id
                if existing is not None and existing.intent == intent
                else uuid.uuid4().hex
            )
        pending = _PendingCommit(
            reconciliation_id=reconciliation_id,
            authority=authority,
            intent=intent,
            mode=mode,
            publish_summary=publish_summary,
            last_error_summary=recovery_summary,
            committed_result=committed_result,
        )
        with self._condition:
            self._commit_inflight = False
            self._pending_commit = pending
            self._phase = "commit-reconciliation-pending"
            self._condition.notify_all()
        return _CommitRetryRequired(pending.reconciliation_id, recovery_summary)

    def _execute_commit(
        self,
        authority: _CommitAuthoritySnapshot[CommitT],
        deadline: float | None,
    ) -> tuple[CommitT, str | None]:
        authority.require_lifetime()
        preparation = authority.preparation
        intent = CommitIntent(
            commit_id=preparation.commit_id,
            run_id=preparation.run_id,
            safety_bundle_id=preparation.safety_bundle_id,
            target=preparation.target,
            created_at=time.time(),
        )
        try:
            authority.begin_intent(intent)
        except BaseException as begin_error:
            try:
                if any(value == intent for value in authority.pending_intents()):
                    authority.mark_aborted(intent.commit_id)
            except BaseException as recovery_error:
                raise self._pause_commit(
                    authority=authority,
                    intent=intent,
                    mode=_CommitResolutionMode.FORCE_ABORT,
                    publish_error=begin_error,
                    recovery_error=recovery_error,
                ) from None
            authority.release_lifetime()
            raise
        try:
            with self._condition:
                self._token.checkpoint()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"run {self.run_id} exceeded its deadline")
                self._cancel_gate_closed = True
                self._phase = "committing"
                self._condition.notify_all()
        except BaseException as gate_error:
            try:
                authority.mark_aborted(intent.commit_id)
            except BaseException as recovery_error:
                raise self._pause_commit(
                    authority=authority,
                    intent=intent,
                    mode=_CommitResolutionMode.FORCE_ABORT,
                    publish_error=gate_error,
                    recovery_error=recovery_error,
                ) from None
            authority.release_lifetime()
            raise
        publish_summary: str | None = None
        try:
            committed = authority.publish_validated(intent)
        except PublishVisibilityUnknown as visibility_error:
            publish_summary = safe_error_summary(visibility_error)
            try:
                recovery = authority.recover_validated(intent)
            except BaseException as recovery_error:
                raise self._pause_commit(
                    authority=authority,
                    intent=intent,
                    mode=_CommitResolutionMode.RECOVER_VISIBILITY,
                    publish_error=visibility_error,
                    recovery_error=recovery_error,
                ) from None
            if recovery is None:
                try:
                    authority.mark_aborted(intent.commit_id)
                except BaseException as marker_error:
                    raise self._pause_commit(
                        authority=authority,
                        intent=intent,
                        mode=_CommitResolutionMode.FORCE_ABORT,
                        publish_error=visibility_error,
                        recovery_error=marker_error,
                    ) from None
                authority.release_lifetime()
                raise
            committed = recovery.result
            detach_failure(visibility_error, note_prefix="detached visibility traceback")
        except BaseException as publish_error:
            try:
                authority.mark_aborted(intent.commit_id)
            except BaseException as marker_error:
                raise self._pause_commit(
                    authority=authority,
                    intent=intent,
                    mode=_CommitResolutionMode.FORCE_ABORT,
                    publish_error=publish_error,
                    recovery_error=marker_error,
                ) from None
            authority.release_lifetime()
            raise
        try:
            authority.mark_committed(intent.commit_id)
        except BaseException as marker_error:
            raise self._pause_commit(
                authority=authority,
                intent=intent,
                mode=_CommitResolutionMode.FORCE_COMMIT,
                publish_error=publish_summary or marker_error,
                recovery_error=marker_error,
                committed_result=committed,
            ) from None
        authority.release_lifetime()
        return committed, publish_summary

    def _commit_admitted(
        self,
        authority: _CommitAuthoritySnapshot[CommitT],
        deadline: float | None,
    ) -> CommitT:
        try:
            committed, recovered_warning = self._execute_commit(authority, deadline)
        except BaseException:
            with self._condition:
                self._commit_inflight = False
                self._condition.notify_all()
            raise
        with self._condition:
            self._commit_inflight = False
            self._pending_commit = None
            self._final_committed = True
            self._result = committed
            if recovered_warning is not None:
                self._commit_recovery_warning = recovered_warning
            self._condition.notify_all()
        return committed

    def _reconcile_pending_commit(self) -> tuple[bool, object]:
        with self._condition:
            if threading.current_thread() is not self._owner_thread:
                raise CapabilityRevoked("commit reconciliation belongs to Run owner thread")
            pending = self._pending_commit
            if pending is None:
                raise RuntimeError("Run has no pending commit reconciliation")
            if self._commit_inflight:
                raise RuntimeError("commit reconciliation is already in flight")
            self._commit_inflight = True
            self._phase = "reconciling-commit"
            self._condition.notify_all()
        authority = pending.authority
        mode = pending.mode
        committed_result = pending.committed_result
        authority.require_lifetime()
        if mode is _CommitResolutionMode.RECOVER_VISIBILITY:
            try:
                recovery = authority.recover_validated(pending.intent)
            except BaseException as error:
                raise self._pause_commit(
                    authority=authority,
                    intent=pending.intent,
                    mode=mode,
                    publish_error=pending.publish_summary,
                    recovery_error=error,
                    committed_result=committed_result,
                ) from None
            if recovery is not None:
                committed_result = recovery.result
                mode = _CommitResolutionMode.FORCE_COMMIT
            else:
                mode = _CommitResolutionMode.FORCE_ABORT
        try:
            authority.begin_intent(pending.intent)
            if mode is _CommitResolutionMode.FORCE_COMMIT:
                if committed_result is _NO_COMMITTED_RESULT:
                    raise RuntimeError("commit reconciliation lost its validated result")
                authority.mark_committed(pending.intent.commit_id)
            else:
                authority.mark_aborted(pending.intent.commit_id)
        except BaseException as error:
            raise self._pause_commit(
                authority=authority,
                intent=pending.intent,
                mode=mode,
                publish_error=pending.publish_summary,
                recovery_error=error,
                committed_result=committed_result,
            ) from None
        authority.release_lifetime()
        committed = mode is _CommitResolutionMode.FORCE_COMMIT
        with self._condition:
            self._commit_inflight = False
            self._pending_commit = None
            if committed:
                self._final_committed = True
                self._result = committed_result
            self._commit_recovery_warning = pending.publish_summary
            self._condition.notify_all()
        return committed, committed_result

    def _terminal_publication(
        self,
        *,
        state: RunState,
        result: object,
        primary_error: str | None,
        cleanup_errors: tuple[str, ...],
    ) -> TerminalPublication:
        if not state.terminal:
            raise ValueError("terminal publication requires terminal state")

        def publish() -> None:
            self._publish_terminal(
                state=state,
                result=result,
                primary_error=primary_error,
                cleanup_errors=cleanup_errors,
            )

        def after_release() -> None:
            with self._condition:
                self._condition.notify_all()

        return _mint_terminal_publication(publish, after_release)

    def _publish_terminal(
        self,
        *,
        state: RunState,
        result: object,
        primary_error: str | None,
        cleanup_errors: tuple[str, ...],
    ) -> None:
        with self._condition:
            if self._state.terminal:
                raise RuntimeError("Run terminal state may be published only once")
            if self._commit_inflight or self._pending_commit is not None:
                raise RuntimeError("cannot publish while commit reconciliation is pending")
            self._cancel_gate_closed = True
            cleanup_errors = _bounded_diagnostics(
                (*cleanup_errors, *self._interrupt_errors)
            )
            if self._final_committed:
                if primary_error is not None:
                    cleanup_errors = _bounded_diagnostics((*cleanup_errors, primary_error))
                state = RunState.SUCCEEDED
                primary_error = None
                result = self._result
            elif state is RunState.SUCCEEDED and self._token.is_cancelled:
                state = RunState.CANCELLED
            if state is not RunState.SUCCEEDED:
                result = _MISSING
            self._state = state
            self._phase = "terminal"
            self._result = result
            self._primary_error = primary_error
            self._cleanup_errors = cleanup_errors
            self._retry_disposition = None
            self._recovery_instruction = None
            self._interrupt = None
            self._cancel_sealer = None
            self._condition.notify_all()

    def _report_thread_joined(self, thread: threading.Thread) -> None:
        """Remove controller admission state only after another thread observed exit."""

        with self._condition:
            if thread is not self._owner_thread or thread.is_alive() or self._owner_reaped:
                return
            self._owner_reaped = True
            callback = self._on_thread_joined
            terminal = self._state.terminal
            if terminal:
                self._on_thread_joined = None
        if terminal and callback is not None:
            callback(self)


class RunController:
    """Admission, cancellation, and drain owner for all RunHandles it starts."""

    def __init__(self, resources: ResourceArbiter) -> None:
        if not isinstance(resources, ResourceArbiter):
            raise TypeError("resources must be ResourceArbiter")
        self._resources = resources
        self._lock = threading.RLock()
        self._accepting = True
        self._handles: dict[RunId, RunHandle] = {}

    def start(self, plan: RunPlan[PreparedT, ExecutedT, FinalT]) -> RunHandle[FinalT]:
        if not isinstance(plan, RunPlan):
            raise TypeError("plan must be RunPlan")
        run_id = RunId(uuid.uuid4().hex)
        with self._lock:
            if not self._accepting:
                raise RuntimeError("RunController is shutting down and rejects new Runs")
            acquired = self._resources.acquire_all(run_id.value, plan.resource_claims)
            if isinstance(acquired, (ResourceBusy, ResourceQuarantined)):
                raise RunStartRejected(acquired)
            source = _CancellationSource()
            handle: RunHandle[FinalT] = RunHandle(run_id, source, self._on_thread_joined)
            deadline = (
                None
                if plan.timeout_seconds is None
                else time.monotonic() + plan.timeout_seconds
            )
            context = RunContext(handle, source.token, deadline, plan.bound_devices)
            ownership = _ExecutionOwnership(plan, context)

            thread = threading.Thread(
                target=RunController._run_owner,
                args=(ownership, acquired, handle),
                name=f"zlc-run-{run_id.value[:12]}",
                daemon=False,
            )
            interrupt = (
                None
                if not plan.interrupt_operations
                else lambda: context._run_interrupts(plan.interrupt_operations)
            )
            handle._bind_owner(thread, interrupt, context._seal_execution_on_cancel)
            self._handles[run_id] = handle
        try:
            thread.start()
        except BaseException as error:
            acquired._release_unarmed()
            summary = safe_error_summary(error)
            handle._publish_terminal(
                state=RunState.FAILED,
                result=_MISSING,
                primary_error=summary,
                cleanup_errors=(),
            )
            handle._report_thread_joined(thread)
            detach_failure(error, note_prefix="detached owner start traceback")
            raise RuntimeError(f"Run owner thread failed to start: {summary}") from None
        return handle

    def run(
        self,
        plan: RunPlan[PreparedT, ExecutedT, FinalT],
        *,
        cancel_join_timeout: float = 5.0,
    ) -> FinalT:
        return self.wait(
            self.start(plan),
            cancel_join_timeout=cancel_join_timeout,
        )

    def wait(
        self,
        handle: RunHandle[FinalT],
        *,
        cancel_join_timeout: float = 5.0,
    ) -> FinalT:
        """Wait for one admitted Run while owning notebook interrupt semantics."""

        if not isinstance(handle, RunHandle):
            raise TypeError("handle must be RunHandle")
        timeout = finite_real(cancel_join_timeout, "cancel_join_timeout", minimum=0.0)
        try:
            return handle.result()
        except KeyboardInterrupt:
            handle.cancel("notebook KeyboardInterrupt")
            try:
                snapshot = handle.wait(timeout)
            except TimeoutError as error:
                raise RunStillCancelling(handle) from error
            if snapshot.state is RunState.SUCCEEDED:
                return handle.result()
            if snapshot.state is RunState.FAILED:
                raise RunFailed(snapshot)
            raise

    def shutdown(self, timeout: float | None = None) -> bool:
        """Close admission, cancel every active Run, and wait for terminal release."""

        timeout = None if timeout is None else finite_real(timeout, "shutdown timeout", minimum=0.0)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            self._accepting = False
            handles = tuple(self._handles.values())
        for handle in handles:
            handle.cancel("RunController shutdown")
        for handle in handles:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                handle.wait(remaining)
            except TimeoutError:
                return False
        with self._lock:
            return not self._handles

    def _on_thread_joined(self, finished: RunHandle) -> None:
        with self._lock:
            self._handles.pop(finished.run_id, None)

    @staticmethod
    def _hazards(plan: RunPlan) -> tuple[HazardClaim, ...]:
        return tuple(
            HazardClaim(device.key, device.binding_stamp)
            for device in plan.bound_devices
        )

    @staticmethod
    def _run_owner(
        ownership: _ExecutionOwnership,
        lease: ResourceLease,
        handle: RunHandle[FinalT],
    ) -> None:
        owned_plan, owned_context = ownership.take()
        try:
            owned_context.set_phase("binding-devices")
            owned_context.checkpoint()
            owned_context._bind_devices()
            owned_context.checkpoint()
        except BaseException as error:
            ownership.clear()
            RunController._finish_before_hazards(
                lease,
                handle,
                owned_context,
                error,
            )
            return
        hazards = RunController._hazards(owned_plan)
        try:
            owned_context.set_phase("hazard-journal")
            lease.activate_hazards(hazards)
        except HazardEpochExpired as error:
            ownership.clear()
            RunController._finish_before_hazards(
                lease,
                handle,
                owned_context,
                error,
            )
            return
        except SafetyJournalWriteError as error:
            summary = safe_error_summary(error)
            detach_failure(error, note_prefix="detached hazard journal traceback")

            def retry_hazards() -> None:
                try:
                    lease.activate_hazards(hazards)
                except HazardEpochExpired as expired:
                    RunController._finish_before_hazards(
                        lease,
                        handle,
                        ownership.take()[1],
                        expired,
                    )
                    ownership.clear()
                    return
                RunController._run_after_hazards_guarded(ownership, lease, handle)

            handle._install_retry(
                retry_hazards,
                "repair the safety journal, then retry hazard activation",
                (summary,),
            )
            return
        owned_plan = None  # type: ignore[assignment]
        owned_context = None  # type: ignore[assignment]
        RunController._run_after_hazards_guarded(ownership, lease, handle)

    @staticmethod
    def _finish_before_hazards(
        lease: ResourceLease,
        handle: RunHandle,
        context: RunContext,
        error: BaseException,
    ) -> None:
        cancelled = isinstance(error, CancellationRequested)
        summary = safe_error_summary(error)
        detach_failure(error, note_prefix="detached pre-hazard traceback")
        try:
            context._revoke_hardware()
            seed = context._detach_for_post_safety()
        except BaseException as revoke_error:
            summary = summary + "; " + safe_error_summary(revoke_error)
            detach_failure(revoke_error, note_prefix="detached revoke traceback")
            seed = _PostSafetySeed(handle, handle._token, None)

        def finish(bundle: SafetyDispositionBundle | None) -> None:
            post = seed.mint(bundle)
            post._revoke()
            state = RunState.CANCELLED if cancelled else RunState.FAILED
            publication = handle._terminal_publication(
                state=state,
                result=_MISSING,
                primary_error=summary,
                cleanup_errors=(),
            )
            lease.release_terminal(publication, disposition=state.value)

        try:
            bundle = lease._commit_safety(())
        except SafetyJournalWriteError as journal_error:
            journal_summary = safe_error_summary(journal_error)
            detach_failure(journal_error, note_prefix="detached safety journal traceback")

            def retry() -> None:
                finish(lease._commit_safety(()))

            handle._install_retry(
                retry,
                "repair the safety journal, then release the unarmed Run",
                (summary, journal_summary),
            )
            return
        finish(bundle)

    @staticmethod
    def _run_after_hazards_guarded(
        ownership: _ExecutionOwnership,
        lease: ResourceLease,
        handle: RunHandle[FinalT],
    ) -> None:
        try:
            RunController._run_after_hazards(ownership, lease, handle)
        except BaseException as error:
            RunController._fail_closed(ownership, lease, handle, error)

    @staticmethod
    def _fail_closed(
        ownership: _ExecutionOwnership,
        lease: ResourceLease,
        handle: RunHandle,
        error: BaseException,
    ) -> None:
        if lease.released or handle.snapshot().state.terminal:
            return
        try:
            plan, context = ownership.take()
        except RuntimeError:
            summary = safe_error_summary(error)
            detach_failure(error, note_prefix="detached post-safety lifecycle traceback")
            if lease.safety_committed:
                handle._mark_post_safety(lease.safety_bundle)
                publication = handle._terminal_publication(
                    state=RunState.FAILED,
                    result=_MISSING,
                    primary_error=summary,
                    cleanup_errors=(),
                )
                lease.release_terminal(publication, disposition=RunState.FAILED.value)
            else:
                handle._block_non_retryable_recovery((summary,))
            return
        summary = safe_error_summary(error)
        try:
            handle._begin_cleanup_after_interrupt()
            context._revoke_hardware()
        except BaseException as revoke_error:
            summary += "; " + safe_error_summary(revoke_error)
            detach_failure(revoke_error, note_prefix="detached fail-closed revoke traceback")
        try:
            seed = context._detach_for_post_safety()
        except BaseException:
            seed = _PostSafetySeed(handle, handle._token, None)
        decisions = tuple(
            SafetyDecision.unsafe(
                device.key,
                reason="runtime lifecycle failed before safe-state proof completed",
                recovery_action="verify physical safe state before clearing quarantine",
            )
            for device in plan.bound_devices
        )
        ownership.clear()
        plan = None  # type: ignore[assignment]
        context = None  # type: ignore[assignment]

        def finish(bundle: SafetyDispositionBundle | None) -> None:
            post = seed.mint(bundle)
            post._revoke()
            publication = handle._terminal_publication(
                state=RunState.FAILED,
                result=_MISSING,
                primary_error=summary,
                cleanup_errors=(),
            )
            lease.release_terminal(publication, disposition=RunState.FAILED.value)

        try:
            bundle = lease._commit_safety(decisions)
        except SafetyJournalWriteError as journal_error:
            journal_summary = safe_error_summary(journal_error)
            detach_failure(journal_error, note_prefix="detached fail-closed journal traceback")

            def retry() -> None:
                finish(lease._commit_safety(decisions))

            handle._install_retry(
                retry,
                "repair the safety journal, then finish fail-closed termination",
                (summary, journal_summary),
            )
            return
        detach_failure(error, note_prefix="detached lifecycle traceback")
        finish(bundle)

    @staticmethod
    def _run_after_hazards(
        ownership: _ExecutionOwnership,
        lease: ResourceLease,
        handle: RunHandle[FinalT],
    ) -> None:
        plan, context = ownership.take()
        hazards = RunController._hazards(plan)
        hazard_keys = tuple(hazard.key for hazard in hazards)
        finalize = plan.finalize
        requires_final_commit = plan.requires_final_commit
        dispose_unfinalized = plan.dispose_unfinalized
        if context._enable_hardware():
            handle._enable_interrupt()
        prepared: PreparedT | None = None
        executed: object = _MISSING
        primary: BaseException | None = None
        cancelled = False
        try:
            context.set_phase("preflight")
            context.checkpoint()
            prepared = plan.preflight(context)
            context.checkpoint()
            context.set_phase("execute")
            executed = plan.execute(context, prepared)
            context.checkpoint()
        except BaseException as error:
            primary = error
            cancelled = isinstance(error, CancellationRequested)
        finalization_input = _FinalizationInput(executed, dispose_unfinalized)
        executed = _MISSING
        try:
            handle._begin_cleanup_after_interrupt()
            context._enter_cleanup_hardware()
            try:
                context.set_phase("cleanup")
                report = plan.cleanup(context, prepared, primary)
                if not isinstance(report, CleanupReport):
                    raise TypeError("RunPlan.cleanup must return CleanupReport")
            except BaseException as cleanup_error:
                report = CleanupReport.unsafe(
                    hazard_keys,
                    reason="cleanup raised before safety could be confirmed",
                    recovery_action="verify physical safe state before clearing quarantine",
                    errors=(cleanup_error,),
                )
            decisions, coverage_errors = _resolve_cleanup_decisions(
                context,
                report,
                hazard_keys,
            )
            interrupt_errors = handle._interrupt_error_snapshot()
            cleanup_errors = tuple(
                safe_error_summary(error)
                for error in (*report.errors, *coverage_errors)
            )
            cleanup_errors = _bounded_diagnostics((*cleanup_errors, *interrupt_errors))
            unsafe = tuple(
                decision.key
                for decision in decisions
                if decision.outcome is SafetyOutcome.UNSAFE
            )
            if (
                primary is None
                and handle._token.is_cancelled
                and not handle.snapshot().final_committed
            ):
                primary = CancellationRequested(handle._token.reason)
                cancelled = True
            primary_summary = None if primary is None else safe_error_summary(primary)
            if primary is not None:
                detach_failure(primary, note_prefix="detached Run traceback")
            for error in (*report.errors, *coverage_errors):
                detach_failure(error, note_prefix="detached cleanup traceback")
            prepared = None
            report = None  # type: ignore[assignment]
            primary = None
            plan = None  # type: ignore[assignment]
            context._revoke_hardware()
            seed = context._detach_for_post_safety()
            context = None  # type: ignore[assignment]
            ownership.clear()

            def after_safety(bundle: SafetyDispositionBundle | None) -> None:
                RunController._finalize_and_publish(
                    lease=lease,
                    handle=handle,
                    seed=seed,
                    bundle=bundle,
                    finalize=finalize,
                    finalization_input=finalization_input,
                    requires_final_commit=requires_final_commit,
                    primary_error=primary_summary,
                    cancelled=cancelled,
                    cleanup_errors=cleanup_errors,
                    unsafe=unsafe,
                )

            try:
                handle._set_phase("finalizing-safety")
                bundle = lease._commit_safety(decisions)
            except SafetyJournalWriteError as journal_error:
                journal_summary = safe_error_summary(journal_error)
                detach_failure(
                    journal_error,
                    note_prefix="detached safety journal traceback",
                )

                def retry() -> None:
                    try:
                        after_safety(lease._commit_safety(decisions))
                    except SafetyJournalWriteError:
                        raise
                    except BaseException as retry_error:
                        _dispose_finalization_input(finalization_input, retry_error)
                        raise

                handle._install_retry(
                    retry,
                    "repair the safety journal, then retry the frozen safety bundle",
                    _bounded_diagnostics((*cleanup_errors, journal_summary)),
                )
                return
            after_safety(bundle)
        except BaseException as lifecycle_error:
            _dispose_finalization_input(finalization_input, lifecycle_error)
            raise

    @staticmethod
    def _finalize_and_publish(
        *,
        lease: ResourceLease,
        handle: RunHandle,
        seed: _PostSafetySeed,
        bundle: SafetyDispositionBundle | None,
        finalize: Callable[[PostSafetyContext, object], object],
        finalization_input: _FinalizationInput,
        requires_final_commit: bool,
        primary_error: str | None,
        cancelled: bool,
        cleanup_errors: tuple[str, ...],
        unsafe: tuple[ResourceKey, ...],
    ) -> None:
        post = seed.mint(bundle)
        result = _MISSING
        pending_control = False
        if primary_error is None and not cleanup_errors and not unsafe:
            try:
                handle._set_phase("finalize")
                post.checkpoint()
                result = finalize(post, finalization_input.take())
            except _CommitRetryRequired:
                pending_control = True
            except BaseException as error:
                primary_error = safe_error_summary(error)
                cancelled = isinstance(error, CancellationRequested)
                detach_failure(error, note_prefix="detached finalize traceback")
            abandonment_errors = post._abandon_unconsumed_commits()
            cleanup_errors = _bounded_diagnostics((*cleanup_errors, *abandonment_errors))
            # A broad user callback may catch the private control exception.
            # Durable ambiguity remains authoritative and cannot be swallowed.
            pending_control = pending_control or handle._pending_commit is not None
        try:
            finalization_input.dispose()
        except BaseException as error:
            cleanup_errors = _bounded_diagnostics(
                (*cleanup_errors, safe_error_summary(error))
            )
            detach_failure(error, note_prefix="detached finalization-input disposal traceback")
        post._revoke()
        post = None  # type: ignore[assignment]
        finalize = None  # type: ignore[assignment]

        def publish_terminal(
            state: RunState,
            terminal_result: object,
            error_summary: str | None,
            errors: tuple[str, ...],
        ) -> None:
            publication = handle._terminal_publication(
                state=state,
                result=terminal_result,
                primary_error=error_summary,
                cleanup_errors=errors,
            )
            lease.release_terminal(publication, disposition=state.value)

        if pending_control:
            pending = handle._pending_commit
            if pending is None:
                publish_terminal(
                    RunState.FAILED,
                    _MISSING,
                    "commit reconciliation control signal lost its private record",
                    cleanup_errors,
                )
                return
            def retry_commit() -> None:
                committed, committed_result = handle._reconcile_pending_commit()
                if committed:
                    errors = cleanup_errors
                    if primary_error is not None:
                        errors = _bounded_diagnostics((*errors, primary_error))
                    publish_terminal(RunState.SUCCEEDED, committed_result, None, errors)
                    return
                error = primary_error or "ambiguous commit was durably resolved as aborted"
                publish_terminal(RunState.FAILED, _MISSING, error, cleanup_errors)

            handle._install_retry(
                retry_commit,
                "reconcile the pending commit; finalize will not be re-entered",
                _bounded_diagnostics((*cleanup_errors, pending.last_error_summary)),
                phase="commit-reconciliation-failed",
            )
            return
        if primary_error is None and handle._token.is_cancelled and not handle.snapshot().final_committed:
            primary_error = safe_error_summary(CancellationRequested(handle._token.reason))
            cancelled = True
            result = _MISSING
        if primary_error is None and requires_final_commit and not handle.snapshot().final_committed:
            primary_error = "RuntimeError: RunPlan requires a final commit but finalize published none"
            result = _MISSING
        if cleanup_errors or unsafe:
            state = RunState.FAILED
        elif primary_error is None:
            state = RunState.SUCCEEDED
        elif cancelled:
            state = RunState.CANCELLED
        else:
            state = RunState.FAILED
        publish_terminal(state, result, primary_error, cleanup_errors)


def _dispose_finalization_input(
    finalization_input: _FinalizationInput,
    primary: BaseException,
) -> None:
    try:
        finalization_input.dispose()
    except BaseException as error:
        record_secondary_failure(
            primary,
            "unfinalized execute-result disposal also failed",
            error,
        )


def _resolve_cleanup_decisions(
    context: RunContext,
    report: CleanupReport,
    hazard_keys: tuple[ResourceKey, ...],
) -> tuple[tuple[SafetyDecision, ...], tuple[BaseException, ...]]:
    expected = set(hazard_keys)
    try:
        receipts = context._consume_safety_proofs(report.safety_proofs)
    except BaseException as error:
        receipts = ()
        errors: list[BaseException] = [error]
    else:
        errors = []
    safe = tuple(SafetyDecision.safe(receipt) for receipt in receipts)
    decisions = {decision.key: decision for decision in report.decisions}
    for decision in safe:
        if decision.key in decisions:
            errors.append(ValueError(f"cleanup returned two decisions for {decision.key}"))
        else:
            decisions[decision.key] = decision
    for key in set(decisions) - expected:
        decisions.pop(key, None)
        errors.append(ValueError(f"cleanup reported undeclared hazard resource {key}"))
    for key in expected - set(decisions):
        errors.append(RuntimeError(f"cleanup omitted safety disposition for {key}"))
        decisions[key] = SafetyDecision.unsafe(
            key,
            reason="hardware safety acknowledgement was omitted",
            recovery_action="verify physical safe state before clearing quarantine",
        )
    return tuple(decisions[key] for key in sorted(decisions)), tuple(errors)


def _bounded_diagnostics(values: tuple[str, ...]) -> tuple[str, ...]:
    """Keep first cause and latest observation; retry count cannot grow memory."""

    cleaned = tuple(value[:1024] for value in values if isinstance(value, str) and value)
    if len(cleaned) <= 2:
        return cleaned
    return cleaned[0], cleaned[-1]
