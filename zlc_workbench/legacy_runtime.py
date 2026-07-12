"""Resource-safe execution facade for the temporary legacy LogicNode island.

This module deliberately contains no Qt and owns no second lifecycle state machine.
Each old node is adapted to one flat :class:`zlc_neutral_atom.runtime.RunPlan`;
the target runtime remains the sole owner of claims, hazard journalling, cancellation,
safety disposition and terminal publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from zlc_neutral_atom.runtime import (
    BoundDevice,
    CancelOutcome,
    ClaimMode,
    CleanupReport,
    CleanupStepAck,
    DeviceBroker,
    DeviceIdentityAck,
    HazardClaim,
    ResourceClaim,
    ResourceKey,
    RunContext,
    RunController,
    RunHandle,
    RunMode,
    RunPlan,
    RunSnapshot,
    RunStartRejected,
    RunState,
    SafeStateAck,
    SafetyDecision,
    SafetyOperation,
)


def _positive_timeout(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return normalized


def _canonical_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


class LegacyDeviceNotRegistered(RuntimeError):
    """A hardware-bearing legacy node referenced a device outside composition."""


class LegacyNodeAlreadyManaged(RuntimeError):
    """The same node object already has a non-terminal authoritative Run."""


class LegacyNodeStartFailed(RuntimeError):
    """The authoritative Run terminated before the old node worker started."""


@dataclass(frozen=True)
class LegacyDeviceRegistration:
    """Composition-owned mapping from one raw old device to target runtime facts.

    ``device`` is used only as an object lookup key.  Resource and safety identity are
    the typed ``key`` plus the acknowledgement returned by ``identity_probe``; Python
    ``id()`` is never persisted or used as a physical identity.
    """

    device: object
    key: ResourceKey
    identity_probe: Callable[[], DeviceIdentityAck]
    cleanup_operations: Mapping[SafetyOperation, Callable[[], CleanupStepAck]]
    cleanup_order: tuple[SafetyOperation, ...]
    verify_safe_state: Callable[[], SafeStateAck]

    def __post_init__(self) -> None:
        if self.device is None:
            raise ValueError("legacy device registration requires a device object")
        if not isinstance(self.key, ResourceKey):
            raise TypeError("legacy device key must be ResourceKey")
        if self.key.segments[0] != "device":
            raise ValueError("legacy hardware ResourceKeys must live below device/")
        if not callable(self.identity_probe):
            raise TypeError("identity_probe must be callable")
        if not callable(self.verify_safe_state):
            raise TypeError("verify_safe_state must be callable")
        operations = dict(self.cleanup_operations)
        if not operations:
            raise ValueError("hazardous legacy devices require at least one cleanup operation")
        for operation, callback in operations.items():
            if not isinstance(operation, SafetyOperation):
                raise TypeError("cleanup operation keys must be SafetyOperation")
            if not callable(callback):
                raise TypeError("cleanup operation callbacks must be callable")
        order = tuple(self.cleanup_order)
        if not order or len(set(order)) != len(order) or set(order) != set(operations):
            raise ValueError("cleanup_order must list every cleanup operation exactly once")
        object.__setattr__(self, "cleanup_operations", MappingProxyType(operations))
        object.__setattr__(self, "cleanup_order", order)


class LegacyDeviceRegistry:
    """Explicit raw-object lookup with lazy broker binding and stable typed identity."""

    def __init__(self, broker: DeviceBroker) -> None:
        if not isinstance(broker, DeviceBroker):
            raise TypeError("broker must be DeviceBroker")
        self._broker = broker
        self._lock = threading.RLock()
        self._registrations: dict[int, LegacyDeviceRegistration] = {}
        self._bindings: dict[int, BoundDevice] = {}

    @property
    def broker(self) -> DeviceBroker:
        return self._broker

    def register(self, registration: LegacyDeviceRegistration) -> None:
        if not isinstance(registration, LegacyDeviceRegistration):
            raise TypeError("registration must be LegacyDeviceRegistration")
        lookup = id(registration.device)
        with self._lock:
            previous = self._registrations.get(lookup)
            if previous is not None and previous.device is not registration.device:
                raise RuntimeError("raw device lookup identity was reused unexpectedly")
            if previous is not None and previous != registration:
                raise RuntimeError("one raw device cannot have two legacy runtime registrations")
            for other in self._registrations.values():
                if other.device is not registration.device and other.key == registration.key:
                    raise ValueError(f"legacy ResourceKey {registration.key} is already registered")
            self._registrations[lookup] = registration

    def registration_for(self, device: object) -> LegacyDeviceRegistration:
        lookup = id(device)
        with self._lock:
            registration = self._registrations.get(lookup)
        if registration is None or registration.device is not device:
            raise LegacyDeviceNotRegistered(
                f"legacy device {type(device).__name__} is not registered by the composition root"
            )
        return registration

    def binding_for(self, device: object) -> BoundDevice:
        registration = self.registration_for(device)
        lookup = id(registration.device)
        # Identity probing and broker binding are serialized so two simultaneous starts
        # cannot create competing bindings for one old raw handle.
        with self._lock:
            existing = self._bindings.get(lookup)
            if existing is not None:
                return existing
            identity = self._broker.verify_identity(registration.identity_probe)
            binding = self._broker.bind(
                key=registration.key,
                identity=identity,
                execute_command=_reject_direct_legacy_command,
                cleanup_operations=registration.cleanup_operations,
                verify_safe_state=registration.verify_safe_state,
            )
            self._bindings[lookup] = binding
            return binding

    def key_for(self, device: object) -> ResourceKey:
        return self.registration_for(device).key


def _reject_direct_legacy_command(_command: object) -> object:
    raise RuntimeError(
        "legacy LogicNode hardware calls are confined to its fenced owner thread; "
        "new RunPlans must use typed Port commands"
    )


@dataclass(frozen=True)
class LegacyNodeStarted:
    run_id: str
    node_type: str
    started_at: float

    def __post_init__(self) -> None:
        _canonical_text(self.run_id, "run_id")
        _canonical_text(self.node_type, "node_type")
        if not math.isfinite(float(self.started_at)):
            raise ValueError("started_at must be finite")


class LegacyStopStatus(str, Enum):
    NOT_RUNNING = "NOT_RUNNING"
    PENDING = "PENDING"
    TERMINATED = "TERMINATED"


@dataclass(frozen=True)
class LegacyStopReceipt:
    status: LegacyStopStatus
    snapshot: RunSnapshot | None
    cancel_outcome: CancelOutcome | None

    @property
    def terminated(self) -> bool:
        return self.status in (LegacyStopStatus.NOT_RUNNING, LegacyStopStatus.TERMINATED)


class _StartLatch:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.receipt: LegacyNodeStarted | None = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()

    def started(self, receipt: LegacyNodeStarted) -> None:
        with self._lock:
            if self.event.is_set():
                raise RuntimeError("legacy start latch may resolve only once")
            self.receipt = receipt
            self.event.set()

    def failed(self, error: BaseException) -> None:
        with self._lock:
            if not self.event.is_set():
                self.error = error
                self.event.set()


class LegacyRunHandle:
    """Read-only facade over the authoritative target-runtime RunHandle."""

    def __init__(
        self,
        *,
        node: object,
        handle: RunHandle[object],
        latch: _StartLatch,
        claims: tuple[ResourceClaim, ...],
    ) -> None:
        self._node = node
        self._handle = handle
        self._latch = latch
        self._claims = claims

    @property
    def node(self) -> object:
        return self._node

    @property
    def run_handle(self) -> RunHandle[object]:
        return self._handle

    @property
    def claims(self) -> tuple[ResourceClaim, ...]:
        return self._claims

    @property
    def started(self) -> bool:
        return self._latch.receipt is not None

    def snapshot(self) -> RunSnapshot:
        return self._handle.snapshot()

    def wait_started(self, timeout: float | None = None) -> LegacyNodeStarted:
        deadline = None
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("start wait timeout must be a finite non-negative number or None")
            normalized = float(timeout)
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError("start wait timeout must be a finite non-negative number or None")
            deadline = time.monotonic() + normalized
        while True:
            receipt = self._latch.receipt
            if receipt is not None:
                return receipt
            if self._latch.error is not None:
                raise LegacyNodeStartFailed(str(self._latch.error)) from self._latch.error
            snapshot = self._handle.snapshot()
            if snapshot.state.terminal:
                raise LegacyNodeStartFailed(
                    snapshot.primary_error or "legacy Run terminated before node start"
                )
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("legacy node has not crossed its start boundary")
            self._latch.event.wait(0.05 if remaining is None else min(0.05, remaining))

    def cancel(self, reason: str = "legacy node stop requested") -> CancelOutcome:
        return self._handle.cancel(reason)


class LegacyRuntimeFence:
    """Adapt old LogicNodes to the target runtime without becoming a lifecycle owner."""

    def __init__(
        self,
        controller: RunController,
        devices: LegacyDeviceRegistry,
        *,
        stop_attempt_timeout: float = 0.25,
        poll_interval: float = 0.01,
    ) -> None:
        if not isinstance(controller, RunController):
            raise TypeError("controller must be RunController")
        if not isinstance(devices, LegacyDeviceRegistry):
            raise TypeError("devices must be LegacyDeviceRegistry")
        self._controller = controller
        self._devices = devices
        self._stop_attempt_timeout = _positive_timeout(
            stop_attempt_timeout, "stop_attempt_timeout"
        )
        self._poll_interval = _positive_timeout(poll_interval, "poll_interval")
        self._lock = threading.RLock()
        self._handles: dict[int, LegacyRunHandle] = {}

    @property
    def controller(self) -> RunController:
        return self._controller

    @property
    def devices(self) -> LegacyDeviceRegistry:
        return self._devices

    def start(self, node: object) -> LegacyRunHandle:
        if not callable(getattr(node, "start", None)):
            raise TypeError("legacy node must provide start()")
        if not callable(getattr(node, "stop", None)):
            raise TypeError("legacy node must provide stop(timeout=...)")
        if bool(getattr(node, "running", False)):
            raise LegacyNodeAlreadyManaged(
                "an already-running legacy node cannot be adopted; stop it and restart through the fence"
            )
        lookup = id(node)
        with self._lock:
            previous = self._handles.get(lookup)
            if previous is not None and previous.node is node:
                if not previous.snapshot().state.terminal:
                    raise LegacyNodeAlreadyManaged("legacy node already has an active Run")
                self._handles.pop(lookup, None)

            claims, hazards, bindings, registrations = self._resolve_node_resources(node)
            latch = _StartLatch()
            plan = self._plan_for(
                node,
                latch=latch,
                claims=claims,
                hazards=hazards,
                bindings=bindings,
                registrations=registrations,
            )
            handle = self._controller.start(plan)
            wrapped = LegacyRunHandle(
                node=node,
                handle=handle,
                latch=latch,
                claims=claims,
            )
            self._handles[lookup] = wrapped
            return wrapped

    def handle_for(self, node: object) -> LegacyRunHandle | None:
        with self._lock:
            handle = self._handles.get(id(node))
            return handle if handle is not None and handle.node is node else None

    def stop(
        self,
        node: object,
        *,
        timeout: float = 2.0,
        reason: str = "legacy node stop requested",
    ) -> LegacyStopReceipt:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("stop timeout must be a finite non-negative number")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("stop timeout must be a finite non-negative number")
        handle = self.handle_for(node)
        if handle is None:
            if bool(getattr(node, "running", False)):
                raise LegacyNodeAlreadyManaged(
                    "running node is outside LegacyRuntimeFence authority and cannot be reported stopped"
                )
            return LegacyStopReceipt(LegacyStopStatus.NOT_RUNNING, None, None)
        outcome = handle.cancel(_canonical_text(reason, "stop reason"))
        try:
            snapshot = handle.run_handle.wait(timeout)
        except TimeoutError:
            return LegacyStopReceipt(
                LegacyStopStatus.PENDING,
                handle.snapshot(),
                outcome,
            )
        return LegacyStopReceipt(LegacyStopStatus.TERMINATED, snapshot, outcome)

    def stop_all(self, *, timeout: float = 2.0) -> tuple[LegacyStopReceipt, ...]:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("stop_all timeout must be a finite non-negative number")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("stop_all timeout must be a finite non-negative number")
        deadline = time.monotonic() + timeout
        with self._lock:
            handles = tuple(self._handles.values())
        return tuple(
            self.stop(
                handle.node,
                timeout=max(0.0, deadline - time.monotonic()),
                reason="legacy runtime fence shutdown",
            )
            for handle in handles
            if not handle.snapshot().state.terminal
        )

    def stop_nodes_using(
        self,
        devices: Sequence[object],
        *,
        timeout: float = 2.0,
    ) -> tuple[LegacyStopReceipt, ...]:
        keys = tuple(self._devices.key_for(device) for device in devices)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("stop_nodes_using timeout must be a finite non-negative number")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("stop_nodes_using timeout must be a finite non-negative number")
        deadline = time.monotonic() + timeout
        with self._lock:
            handles = tuple(self._handles.values())
        affected = tuple(
            handle
            for handle in handles
            if not handle.snapshot().state.terminal
            and any(
                claim.key.overlaps(key)
                for claim in handle.claims
                for key in keys
            )
        )
        return tuple(
            self.stop(
                handle.node,
                timeout=max(0.0, deadline - time.monotonic()),
                reason="legacy device generation is being replaced",
            )
            for handle in affected
        )

    def _resolve_node_resources(
        self,
        node: object,
    ) -> tuple[
        tuple[ResourceClaim, ...],
        tuple[HazardClaim, ...],
        tuple[BoundDevice, ...],
        tuple[LegacyDeviceRegistration, ...],
    ]:
        occupied = _unique_devices(_call_device_set(node, "occupied_devices"))
        referenced = _unique_devices(_call_device_set(node, "referenced_devices"))
        registrations = tuple(self._devices.registration_for(device) for device in occupied)
        bindings = tuple(self._devices.binding_for(device) for device in occupied)
        exclusive_keys = {registration.key for registration in registrations}
        observe_keys = {
            self._devices.key_for(device)
            for device in referenced
            if self._devices.key_for(device) not in exclusive_keys
        }
        claims = tuple(
            sorted(
                (ResourceClaim(key, ClaimMode.EXCLUSIVE) for key in exclusive_keys),
                key=lambda claim: claim.key,
            )
        ) + tuple(
            sorted(
                (ResourceClaim(key, ClaimMode.OBSERVE) for key in observe_keys),
                key=lambda claim: claim.key,
            )
        )
        by_key = {binding.key: binding for binding in bindings}
        hazards = tuple(
            HazardClaim(
                key=registration.key,
                stable_device_identity=by_key[registration.key].stable_device_identity,
                connection_generation=by_key[registration.key].connection_generation,
            )
            for registration in sorted(registrations, key=lambda value: value.key)
        )
        return claims, hazards, bindings, registrations

    def _plan_for(
        self,
        node: object,
        *,
        latch: _StartLatch,
        claims: tuple[ResourceClaim, ...],
        hazards: tuple[HazardClaim, ...],
        bindings: tuple[BoundDevice, ...],
        registrations: tuple[LegacyDeviceRegistration, ...],
    ) -> RunPlan[None, object]:
        binding_by_key = {binding.key: binding for binding in bindings}

        def preflight(_context: RunContext) -> None:
            if bool(getattr(node, "running", False)):
                raise LegacyNodeAlreadyManaged(
                    "legacy node became active outside the fence before preflight"
                )
            return None

        def execute(context: RunContext, _prepared: None) -> object:
            try:
                node.start()
                latch.started(
                    LegacyNodeStarted(
                        run_id=context.run_id.value,
                        node_type=type(node).__name__,
                        started_at=time.time(),
                    )
                )
                while bool(getattr(node, "running", False)):
                    if context.cancellation.is_cancelled:
                        stop_event = getattr(node, "_stop", None)
                        if isinstance(stop_event, threading.Event):
                            stop_event.set()
                        context.checkpoint()
                    time.sleep(self._poll_interval)
                return node
            except BaseException as exc:
                latch.failed(exc)
                raise

        def cleanup(
            context: RunContext,
            _prepared: None,
            _primary: BaseException | None,
        ) -> CleanupReport:
            stop_errors = _stop_node_until_terminated(
                node,
                attempt_timeout=self._stop_attempt_timeout,
                retry_interval=self._poll_interval,
            )
            proofs = []
            decisions = []
            errors: list[BaseException] = list(stop_errors)
            for registration in registrations:
                device = context.cleanup_device(registration.key)
                try:
                    for operation in registration.cleanup_order:
                        device.perform(operation)
                    proofs.append(device.verify_safe_state())
                except BaseException as exc:
                    errors.append(exc)
                    decisions.append(
                        SafetyDecision.unsafe(
                            registration.key,
                            reason=(
                                "legacy device safety recipe or readback failed after its "
                                "LogicNode owner terminated"
                            ),
                            recovery_action=(
                                "verify the stable device identity and connection generation, "
                                "then run the device-specific recovery recipe"
                            ),
                        )
                    )
            return CleanupReport.mixed(
                tuple(proofs),
                tuple(decisions),
                errors=tuple(errors),
            )

        def finalize(_context, result: object) -> object:
            return result

        return RunPlan(
            name=f"legacy:{type(node).__name__}",
            mode=RunMode.CONTINUOUS_MONITOR,
            resource_claims=claims,
            hazard_claims=hazards,
            bound_devices=tuple(binding_by_key[key] for key in sorted(binding_by_key)),
            preflight=preflight,
            execute=execute,
            cleanup=cleanup,
            finalize=finalize,
        )


def _call_device_set(node: object, method_name: str) -> tuple[object, ...]:
    method = getattr(node, method_name, None)
    if method is None:
        return ()
    if not callable(method):
        raise TypeError(f"legacy node {method_name} must be callable")
    return tuple(method())


def _unique_devices(devices: Sequence[object]) -> tuple[object, ...]:
    unique: list[object] = []
    seen: dict[int, object] = {}
    for device in devices:
        lookup = id(device)
        previous = seen.get(lookup)
        if previous is None:
            seen[lookup] = device
            unique.append(device)
        elif previous is not device:
            raise RuntimeError("Python object identity was reused while resolving node devices")
    return tuple(unique)


def _stop_node_until_terminated(
    node: object,
    *,
    attempt_timeout: float,
    retry_interval: float,
) -> tuple[BaseException, ...]:
    """Serialize stop attempts and never return while helper or node owner is alive."""

    errors: list[BaseException] = []
    while True:
        outcome: dict[str, object] = {}

        def invoke() -> None:
            try:
                outcome["result"] = node.stop(timeout=attempt_timeout)
            except BaseException as exc:
                outcome["error"] = exc

        helper = threading.Thread(
            target=invoke,
            name=f"zlc-legacy-stop-{type(node).__name__}",
            daemon=False,
        )
        helper.start()
        # Deliberately unbounded: a stop implementation that ignores its timeout still
        # owns hardware.  Returning would release/quarantine the claim while that helper
        # and the old node may continue touching the raw device.
        helper.join()
        error = outcome.get("error")
        if isinstance(error, BaseException):
            errors.append(error)
        try:
            running = bool(getattr(node, "running"))
        except BaseException as exc:
            errors.append(exc)
            running = True
        if not running:
            return tuple(errors)
        time.sleep(retry_interval)


__all__ = [
    "LegacyDeviceNotRegistered",
    "LegacyDeviceRegistration",
    "LegacyDeviceRegistry",
    "LegacyNodeAlreadyManaged",
    "LegacyNodeStartFailed",
    "LegacyNodeStarted",
    "LegacyRunHandle",
    "LegacyRuntimeFence",
    "LegacyStopReceipt",
    "LegacyStopStatus",
]
