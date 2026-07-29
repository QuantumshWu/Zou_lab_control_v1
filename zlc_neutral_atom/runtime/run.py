"""Flat synchronous Run lifecycle hosted by one authoritative owner thread."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Generic, TypeVar

from zlc_storage import canonical_text as _canonical_text, finite_real

from ._failure import detach_failure, record_secondary_failure, safe_error_summary
from .cancellation import CancellationRequested, CancellationToken, _CancellationSource
from .cleanup import CleanupReport
from .commit import (
    PreparedArtifactCommit,
    _consume_prepared_artifact_commit,
    _finish_prepared_artifact_commit,
    _inspect_prepared_artifact_commit,
    _publish_prepared_artifact_commit,
)
from .ports import (
    BoundDevice,
    CleanupDevice,
    RunDevice,
    SafetyInterrupt,
    SessionClosedAck,
    _open_device_run,
)
from .resources import (
    ResourceArbiter,
    ResourceBusy,
    ResourceClaim,
    ResourceKey,
    ResourceLease,
)


PreparedT = TypeVar("PreparedT")
ExecutedT = TypeVar("ExecutedT")
FinalT = TypeVar("FinalT")
CommitT = TypeVar("CommitT")
_MISSING = object()
_COMMIT_INSPECTION_INTERVAL_SECONDS = 0.02


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
    commit_publication_warning: str | None
    primary_error: str | None
    cleanup_errors: tuple[str, ...]


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
    lifecycle_owner: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    preemptible: bool = False

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
        claimed_devices = {
            claim.key
            for claim in claims
            if claim.key.segments[0] == "device"
        }
        if {device.key for device in devices} != claimed_devices:
            raise ValueError("every device claim requires exactly one BoundDevice")
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
        if not isinstance(self.preemptible, bool):
            raise TypeError("preemptible must be bool")
        if self.preemptible and self.lifecycle_owner is None:
            raise ValueError("a preemptible RunPlan requires a lifecycle owner")
        object.__setattr__(
            self,
            "timeout_seconds",
            None
            if self.timeout_seconds is None
            else finite_real(self.timeout_seconds, "timeout_seconds", minimum=0.0),
        )

    def with_lifecycle(
        self,
        owner: object,
        *,
        preemptible: bool,
    ) -> "RunPlan[PreparedT, ExecutedT, FinalT]":
        """Freeze one application lifecycle identity onto an unbound plan."""

        if owner is None:
            raise TypeError("lifecycle owner must not be None")
        if self.lifecycle_owner is not None:
            raise RuntimeError("RunPlan lifecycle facts are already bound")
        return replace(
            self,
            lifecycle_owner=owner,
            preemptible=preemptible,
        )


class CapabilityRevoked(RuntimeError):
    pass


class _PermanentCommitRejection(CapabilityRevoked):
    pass


class RunStartRejected(RuntimeError):
    def __init__(self, blockers: tuple[ResourceBusy, ...]) -> None:
        if not isinstance(blockers, tuple):
            raise TypeError("blockers must be a tuple")
        if not blockers or any(
            not isinstance(value, ResourceBusy) for value in blockers
        ):
            raise TypeError("blockers must be a non-empty ResourceBusy tuple")
        super().__init__(f"run start rejected: {blockers}")
        self.blockers = blockers


class RunFailed(RuntimeError):
    def __init__(self, snapshot: RunSnapshot) -> None:
        message = snapshot.primary_error
        if message is None and snapshot.cleanup_errors:
            message = "; ".join(snapshot.cleanup_errors)
        super().__init__(message or "run failed")
        self.snapshot = snapshot


class RunCancelled(RuntimeError):
    def __init__(self, snapshot: RunSnapshot) -> None:
        super().__init__(f"run {snapshot.run_id} was cancelled")
        self.snapshot = snapshot


class RunStillCancelling(RuntimeError):
    def __init__(self, handle: "RunHandle") -> None:
        super().__init__(f"run {handle.run_id} is still cancelling; retain its RunHandle")
        self.handle = handle


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

    def mint(self) -> "PostSafetyContext":
        self.handle._mark_post_safety()
        return PostSafetyContext(
            _POST_SAFETY_CONTEXT_TOKEN,
            run_id=self.handle.run_id,
            cancellation=self.cancellation,
            deadline=self.deadline,
            handle=self.handle,
        )


_POST_SAFETY_CONTEXT_TOKEN = object()


class PostSafetyContext:
    """Finalize capability with no device/session/Port access."""

    __slots__ = (
        "_run_id",
        "_cancellation",
        "_deadline",
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
        handle: "RunHandle",
    ) -> None:
        if token is not _POST_SAFETY_CONTEXT_TOKEN:
            raise PermissionError("PostSafetyContext is minted by RunController")
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_cancellation", cancellation)
        object.__setattr__(self, "_deadline", deadline)
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

    def checkpoint(self) -> None:
        self._require_active()
        self.cancellation.checkpoint()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(f"run {self.run_id} exceeded its monotonic deadline")

    def authorize_commit_preparation(self) -> str:
        self._require_active()._validate_commit_preparation(self.deadline)
        return self.run_id.value

    def track_prepared_commit(
        self,
        operation: PreparedArtifactCommit[object],
    ) -> None:
        """Bind one prepared manifest publication to this Run lifetime."""

        self._require_active()
        if not isinstance(operation, PreparedArtifactCommit):
            raise TypeError(
                "prepared commit tracking requires PreparedArtifactCommit"
            )
        if operation.run_id != self.run_id.value:
            raise ValueError("prepared commit belongs to another Run")
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
        return _diagnostics(tuple(errors))

    def _revoke(self) -> None:
        object.__setattr__(self, "_active", False)
        object.__setattr__(self, "_handle", None)

    def _commit_operation(
        self,
        operation: PreparedArtifactCommit[CommitT],
    ) -> CommitT:
        handle = self._require_active()
        try:
            return handle._commit_prepared_artifact(operation, self.deadline)
        except BaseException as primary:
            try:
                operation.abandon()
            except BaseException as error:
                record_secondary_failure(
                    primary,
                    "rejected artifact commit abandonment also failed",
                    error,
                )
            raise
        finally:
            self._take_tracked(operation)

    def commit_final(self, operation: PreparedArtifactCommit[CommitT]) -> CommitT:
        if not isinstance(operation, PreparedArtifactCommit):
            raise TypeError(
                "commit_final requires a PreparedArtifactCommit operation"
            )
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
        self._commit_publication_warning: str | None = None
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
        self._shutdown_requested = False

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
                        self._interrupt_errors = _diagnostics(
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
                self._interrupt_errors = _diagnostics(
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

    def _active_state_locked(self) -> RunState:
        return RunState.CANCELLING if self._token.is_cancelled else RunState.RUNNING

    def _snapshot_locked(self) -> RunSnapshot:
        return RunSnapshot(
            run_id=self.run_id,
            state=self._state,
            phase=self._phase,
            final_committed=self._final_committed,
            commit_publication_warning=self._commit_publication_warning,
            primary_error=self._primary_error,
            cleanup_errors=self._cleanup_errors,
        )

    def _request_application_shutdown(self) -> None:
        """Wake the owner so a pending manifest gets one final inspection."""

        with self._condition:
            if self._state.terminal:
                return
            self._shutdown_requested = True
            self._condition.notify_all()

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

    def _mark_post_safety(self) -> None:
        with self._condition:
            self._interrupt_enabled = False
            while self._interrupt_inflight:
                self._condition.wait()
            self._interrupt = None
            self._cancel_sealer = None
            self._condition.notify_all()

    def _commit_gate_rejection_locked(
        self,
        deadline: float | None,
    ) -> BaseException | None:
        if self._state.terminal:
            return _PermanentCommitRejection("commit cannot run after terminal state")
        try:
            self._token.checkpoint()
        except BaseException as error:
            return error
        if deadline is not None and time.monotonic() >= deadline:
            return TimeoutError(f"run {self.run_id} exceeded its monotonic deadline")
        if threading.current_thread() is not self._owner_thread:
            return CapabilityRevoked("commit belongs to the Run owner thread")
        if self._final_committed:
            return RuntimeError("final commit may be attempted only once")
        if self._commit_inflight:
            return RuntimeError("another commit is already in flight")
        if self._cancel_gate_closed:
            return RuntimeError("Run is already finalizing")
        return None

    def _validate_commit_preparation(self, deadline: float | None) -> None:
        with self._condition:
            rejection = self._commit_gate_rejection_locked(deadline)
        if rejection is not None:
            raise rejection

    def _commit_prepared_artifact(
        self,
        operation: PreparedArtifactCommit[CommitT],
        deadline: float | None,
    ) -> CommitT:
        """Publish once, then resolve only that exact manifest target.

        Admission, operation consumption, and closure of the cancellation gate
        share one lock transition.  After publication begins it is never
        repeated: a failed acknowledgement is resolved only by exact-target
        inspection on this owner thread.
        """

        if not isinstance(operation, PreparedArtifactCommit):
            raise TypeError("operation must be PreparedArtifactCommit")
        with self._condition:
            rejection = self._commit_gate_rejection_locked(deadline)
            if rejection is None and operation.run_id != self.run_id.value:
                rejection = _PermanentCommitRejection(
                    "prepared artifact commit belongs to another Run"
                )
            if rejection is not None:
                raise rejection
            _consume_prepared_artifact_commit(operation)
            self._commit_inflight = True
            self._cancel_gate_closed = True
            self._phase = "committing"
            self._condition.notify_all()

        primary: BaseException | None = None
        publish_warning: str | None = None
        try:
            try:
                committed = _publish_prepared_artifact_commit(operation)
            except BaseException as publish_error:
                publish_warning = safe_error_summary(publish_error)
                while True:
                    try:
                        resolution = _inspect_prepared_artifact_commit(operation)
                    except BaseException as inspection_error:
                        record_secondary_failure(
                            inspection_error,
                            "manifest publication also failed before exact-target "
                            "inspection found a fatal error",
                            publish_error,
                        )
                        raise inspection_error from None
                    if resolution is True:
                        committed = operation.result
                        detach_failure(
                            publish_error,
                            note_prefix="detached manifest publication traceback",
                        )
                        break
                    if resolution is False:
                        raise
                    with self._condition:
                        if self._shutdown_requested:
                            detach_failure(
                                publish_error,
                                note_prefix="detached manifest publication traceback",
                            )
                            raise RuntimeError(
                                "artifact manifest visibility remained indeterminate "
                                "after the final shutdown inspection; process-local "
                                "commit ownership was abandoned without republication; "
                                f"original publication failure: {publish_warning}"
                            ) from None
                        self._phase = "commit-inspection-pending"
                        self._condition.notify_all()
                        self._condition.wait(_COMMIT_INSPECTION_INTERVAL_SECONDS)

            with self._condition:
                self._final_committed = True
                self._result = committed
                self._commit_publication_warning = publish_warning
                self._phase = "committed"
                self._condition.notify_all()
            return committed
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                _finish_prepared_artifact_commit(operation)
            except BaseException as finish_error:
                if primary is None:
                    raise
                record_secondary_failure(
                    primary,
                    "prepared artifact commit lifetime release also failed",
                    finish_error,
                )
            finally:
                with self._condition:
                    self._commit_inflight = False
                    self._condition.notify_all()

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
            if self._commit_inflight:
                raise RuntimeError("cannot publish while an artifact commit is active")
            self._cancel_gate_closed = True
            cleanup_errors = _diagnostics(
                (*cleanup_errors, *self._interrupt_errors)
            )
            if self._final_committed:
                if primary_error is not None:
                    cleanup_errors = _diagnostics((*cleanup_errors, primary_error))
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
            if not isinstance(acquired, ResourceLease):
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
            acquired.release()
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
        """Wait for one admitted Run while owning caller interrupt semantics."""

        if not isinstance(handle, RunHandle):
            raise TypeError("handle must be RunHandle")
        timeout = finite_real(cancel_join_timeout, "cancel_join_timeout", minimum=0.0)
        try:
            return handle.result()
        except KeyboardInterrupt:
            handle.cancel("caller KeyboardInterrupt")
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
            handle._request_application_shutdown()
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
            RunController._finish_before_execution(
                lease,
                handle,
                owned_context,
                error,
            )
            return
        owned_plan = None  # type: ignore[assignment]
        owned_context = None  # type: ignore[assignment]
        RunController._run_execution_guarded(ownership, lease, handle)

    @staticmethod
    def _finish_before_execution(
        lease: ResourceLease,
        handle: RunHandle,
        context: RunContext,
        error: BaseException,
    ) -> None:
        cancelled = isinstance(error, CancellationRequested)
        summary = safe_error_summary(error)
        detach_failure(error, note_prefix="detached pre-execution traceback")
        try:
            context._revoke_hardware()
        except BaseException as revoke_error:
            summary = summary + "; " + safe_error_summary(revoke_error)
            detach_failure(revoke_error, note_prefix="detached revoke traceback")
        try:
            lease.release()
        except BaseException as release_error:
            summary = summary + "; " + safe_error_summary(release_error)
            detach_failure(release_error, note_prefix="detached resource release traceback")
        try:
            seed = context._detach_for_post_safety()
        except BaseException as detach_error:
            summary = summary + "; " + safe_error_summary(detach_error)
            detach_failure(detach_error, note_prefix="detached post-safety transition traceback")
            seed = _PostSafetySeed(handle, handle._token, None)

        post = seed.mint()
        post._revoke()
        state = RunState.CANCELLED if cancelled else RunState.FAILED
        handle._publish_terminal(
            state=state,
            result=_MISSING,
            primary_error=summary,
            cleanup_errors=(),
        )

    @staticmethod
    def _run_execution_guarded(
        ownership: _ExecutionOwnership,
        lease: ResourceLease,
        handle: RunHandle[FinalT],
    ) -> None:
        try:
            RunController._run_execution(ownership, lease, handle)
        except BaseException as error:
            RunController._fail_closed(ownership, lease, handle, error)

    @staticmethod
    def _fail_closed(
        ownership: _ExecutionOwnership,
        lease: ResourceLease,
        handle: RunHandle,
        error: BaseException,
    ) -> None:
        if handle.snapshot().state.terminal:
            return
        try:
            plan, context = ownership.take()
        except RuntimeError:
            summary = safe_error_summary(error)
            detach_failure(error, note_prefix="detached terminal lifecycle traceback")
            handle._mark_post_safety()
            try:
                lease.release()
            except BaseException as release_error:
                summary = summary + "; " + safe_error_summary(release_error)
                detach_failure(
                    release_error,
                    note_prefix="detached fail-closed resource release traceback",
                )
            handle._publish_terminal(
                state=RunState.FAILED,
                result=_MISSING,
                primary_error=summary,
                cleanup_errors=(),
            )
            return
        summary = safe_error_summary(error)
        cleanup_summaries: list[str] = []
        try:
            handle._begin_cleanup_after_interrupt()
        except BaseException as cleanup_transition_error:
            cleanup_summaries.append(safe_error_summary(cleanup_transition_error))
            detach_failure(
                cleanup_transition_error,
                note_prefix="detached fail-closed cleanup transition traceback",
            )
        else:
            try:
                context._enter_cleanup_hardware()
            except BaseException as interrupt_error:
                cleanup_summaries.append(safe_error_summary(interrupt_error))
                detach_failure(
                    interrupt_error,
                    note_prefix="detached fail-closed cleanup entry traceback",
                )
            else:
                try:
                    context._run_interrupts(plan.interrupt_operations)
                except BaseException as interrupt_error:
                    cleanup_summaries.append(safe_error_summary(interrupt_error))
                    detach_failure(
                        interrupt_error,
                        note_prefix="detached fail-closed interrupt traceback",
                    )
        try:
            context._revoke_hardware()
        except BaseException as revoke_error:
            cleanup_summaries.append(safe_error_summary(revoke_error))
            detach_failure(revoke_error, note_prefix="detached fail-closed revoke traceback")
        try:
            lease.release()
        except BaseException as release_error:
            cleanup_summaries.append(safe_error_summary(release_error))
            detach_failure(
                release_error,
                note_prefix="detached fail-closed resource release traceback",
            )
        try:
            seed = context._detach_for_post_safety()
        except BaseException as detach_error:
            cleanup_summaries.append(safe_error_summary(detach_error))
            detach_failure(
                detach_error,
                note_prefix="detached fail-closed post-safety transition traceback",
            )
            seed = _PostSafetySeed(handle, handle._token, None)
        ownership.clear()
        plan = None  # type: ignore[assignment]
        context = None  # type: ignore[assignment]
        detach_failure(error, note_prefix="detached lifecycle traceback")
        post = seed.mint()
        post._revoke()
        handle._publish_terminal(
            state=RunState.FAILED,
            result=_MISSING,
            primary_error=summary,
            cleanup_errors=_diagnostics(tuple(cleanup_summaries)),
        )

    @staticmethod
    def _run_execution(
        ownership: _ExecutionOwnership,
        lease: ResourceLease,
        handle: RunHandle[FinalT],
    ) -> None:
        plan, context = ownership.take()
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
                report = CleanupReport(errors=(cleanup_error,))
            interrupt_errors = handle._interrupt_error_snapshot()
            cleanup_errors = tuple(
                safe_error_summary(error)
                for error in report.errors
            )
            cleanup_errors = _diagnostics((*cleanup_errors, *interrupt_errors))
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
            for error in report.errors:
                detach_failure(error, note_prefix="detached cleanup traceback")
            prepared = None
            report = None  # type: ignore[assignment]
            primary = None
            plan = None  # type: ignore[assignment]
            try:
                context._revoke_hardware()
            except BaseException as revoke_error:
                cleanup_errors = _diagnostics(
                    (*cleanup_errors, safe_error_summary(revoke_error))
                )
                detach_failure(
                    revoke_error,
                    note_prefix="detached hardware revoke traceback",
                )
            try:
                lease.release()
            except BaseException as release_error:
                cleanup_errors = _diagnostics(
                    (*cleanup_errors, safe_error_summary(release_error))
                )
                detach_failure(
                    release_error,
                    note_prefix="detached resource release traceback",
                )
            seed = context._detach_for_post_safety()
            context = None  # type: ignore[assignment]
            ownership.clear()
            handle._set_phase("post-cleanup")
            RunController._finalize_and_publish(
                handle=handle,
                seed=seed,
                finalize=finalize,
                finalization_input=finalization_input,
                requires_final_commit=requires_final_commit,
                primary_error=primary_summary,
                cancelled=cancelled,
                cleanup_errors=cleanup_errors,
            )
        except BaseException as lifecycle_error:
            _dispose_finalization_input(finalization_input, lifecycle_error)
            raise

    @staticmethod
    def _finalize_and_publish(
        *,
        handle: RunHandle,
        seed: _PostSafetySeed,
        finalize: Callable[[PostSafetyContext, object], object],
        finalization_input: _FinalizationInput,
        requires_final_commit: bool,
        primary_error: str | None,
        cancelled: bool,
        cleanup_errors: tuple[str, ...],
    ) -> None:
        post = seed.mint()
        result = _MISSING
        if primary_error is None and not cleanup_errors:
            try:
                handle._set_phase("finalize")
                post.checkpoint()
                result = finalize(post, finalization_input.take())
            except BaseException as error:
                primary_error = safe_error_summary(error)
                cancelled = isinstance(error, CancellationRequested)
                detach_failure(error, note_prefix="detached finalize traceback")
            abandonment_errors = post._abandon_unconsumed_commits()
            cleanup_errors = _diagnostics((*cleanup_errors, *abandonment_errors))
        try:
            finalization_input.dispose()
        except BaseException as error:
            cleanup_errors = _diagnostics(
                (*cleanup_errors, safe_error_summary(error))
            )
            detach_failure(error, note_prefix="detached finalization-input disposal traceback")
        post._revoke()
        post = None  # type: ignore[assignment]
        finalize = None  # type: ignore[assignment]

        if primary_error is None and handle._token.is_cancelled and not handle.snapshot().final_committed:
            primary_error = safe_error_summary(CancellationRequested(handle._token.reason))
            cancelled = True
            result = _MISSING
        if primary_error is None and requires_final_commit and not handle.snapshot().final_committed:
            primary_error = "RuntimeError: RunPlan requires a final commit but finalize published none"
            result = _MISSING
        if cleanup_errors:
            state = RunState.FAILED
        elif primary_error is None:
            state = RunState.SUCCEEDED
        elif cancelled:
            state = RunState.CANCELLED
        else:
            state = RunState.FAILED
        handle._publish_terminal(
            state=state,
            result=result,
            primary_error=primary_error,
            cleanup_errors=cleanup_errors,
        )


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


def _diagnostics(values: tuple[str, ...]) -> tuple[str, ...]:
    """Preserve every non-empty string diagnostic in observation order."""

    return tuple(value for value in values if isinstance(value, str) and value)
