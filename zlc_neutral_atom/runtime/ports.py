"""Composition-owned device broker and run-scoped capability references.

All in-process project code is trusted.  These types prevent accidental phase and
ownership violations; untrusted extension code requires a process boundary.
"""

from __future__ import annotations

import threading
import math
import uuid
from dataclasses import dataclass, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, TypeVar
from zlc_storage import canonical_text as _canonical_text

from .cleanup import CleanupReport, SafetyProof
from .resources import (
    DeviceBindingStamp,
    PhysicalDeviceIdentity,
    RecoveryAcquireResult,
    RecoveryClaim,
    RecoveryEvidence,
    RecoveryLease,
    ResourceArbiter,
    ResourceBusy,
    ResourceKey,
    SafeReceipt,
)


CommandT = TypeVar("CommandT")
ResponseT = TypeVar("ResponseT")


class SafetyOperation(str, Enum):
    ABORT = "ABORT"
    SAFE_STATE = "SAFE_STATE"
    DISARM = "DISARM"
    READ_STATUS = "READ_STATUS"


@dataclass(frozen=True)
class CleanupStepAck:
    """Adapter acknowledgement for one declared cleanup operation."""

    operation: SafetyOperation
    acknowledgement_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, SafetyOperation):
            raise TypeError("CleanupStepAck.operation must be SafetyOperation")
        _canonical_text(self.acknowledgement_digest, "acknowledgement_digest")


@dataclass(frozen=True)
class SafeStateAck:
    """Adapter readback proving the device is in its declared physical safe state."""

    acknowledgement_digest: str

    def __post_init__(self) -> None:
        _canonical_text(self.acknowledgement_digest, "acknowledgement_digest")


@dataclass(frozen=True)
class SessionCloseCommand:
    session_id: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


@dataclass(frozen=True)
class SessionClosedAck:
    session_id: str
    binding_instance_id: str
    source_stopped: bool
    no_more_work: bool
    joined: bool
    acknowledgement_digest: str

    def __post_init__(self) -> None:
        for field in ("session_id", "binding_instance_id"):
            _canonical_text(getattr(self, field), field)
        for field in ("source_stopped", "no_more_work", "joined"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be bool")
        _canonical_text(self.acknowledgement_digest, "acknowledgement_digest")

    @property
    def is_terminal(self) -> bool:
        """Whether the adapter proved stop, drain, and owner-thread join."""

        return self.source_stopped and self.no_more_work and self.joined


_INTERRUPT_OPERATIONS = frozenset(
    (SafetyOperation.ABORT, SafetyOperation.SAFE_STATE, SafetyOperation.DISARM)
)
_BOUND_DEVICE_TOKEN = object()
_BROKER_OPEN_TOKEN = object()
_VERIFIED_IDENTITY_TOKEN = object()
_VERIFIED_CAPABILITY_TOKEN = object()


@dataclass(frozen=True)
class SafetyInterrupt:
    key: ResourceKey
    operation: SafetyOperation

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("SafetyInterrupt.key must be ResourceKey")
        if self.operation not in _INTERRUPT_OPERATIONS:
            raise ValueError("interrupt operation must be ABORT, SAFE_STATE, or DISARM")


class VerifiedDeviceCapability:
    """Latest broker-minted capability contract for one device binding instance.

    The proof is a frozen fact, not an execution lease.  It remains valid across Runs until a
    successful re-probe supersedes it or the broker shuts down.
    """

    __slots__ = ("_broker", "_device", "_snapshot", "_nonce")

    def __init__(
        self,
        token: object,
        *,
        broker: "DeviceBroker",
        device: "BoundDevice",
        snapshot: object,
        nonce: object,
    ) -> None:
        if token is not _VERIFIED_CAPABILITY_TOKEN:
            raise PermissionError(
                "VerifiedDeviceCapability can only be minted by DeviceBroker"
            )
        object.__setattr__(self, "_broker", broker)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_nonce", nonce)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("VerifiedDeviceCapability is immutable")

    @property
    def device(self) -> "BoundDevice":
        return self._device

    @property
    def snapshot(self) -> object:
        return self._snapshot

class VerifiedPhysicalDeviceIdentity:
    """Opaque result of a broker-owned physical identity handshake."""

    __slots__ = (
        "_physical_identity",
        "_binding_instance_id",
        "_broker",
        "_nonce",
    )

    def __init__(
        self,
        token: object,
        *,
        broker: "DeviceBroker",
        physical_identity: PhysicalDeviceIdentity,
        binding_instance_id: str,
        nonce: object,
    ) -> None:
        if token is not _VERIFIED_IDENTITY_TOKEN:
            raise PermissionError("device identity must be verified by DeviceBroker")
        object.__setattr__(self, "_physical_identity", physical_identity)
        object.__setattr__(self, "_binding_instance_id", binding_instance_id)
        object.__setattr__(self, "_broker", broker)
        object.__setattr__(self, "_nonce", nonce)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("VerifiedPhysicalDeviceIdentity is immutable")

    @property
    def physical_identity(self) -> PhysicalDeviceIdentity:
        return self._physical_identity

    @property
    def binding_instance_id(self) -> str:
        return self._binding_instance_id

@dataclass(frozen=True)
class _DeviceEndpoint:
    key: ResourceKey
    binding_stamp: DeviceBindingStamp
    identity_probe: Callable[[], PhysicalDeviceIdentity]
    execute_command: Callable[[object], object]
    capability_probe: Callable[[], object] | None
    cleanup_operations: Mapping[SafetyOperation, Callable[[], CleanupStepAck]]
    close_session: Callable[[SessionCloseCommand], SessionClosedAck] | None
    verify_safe_state: Callable[[], SafeStateAck]
    interrupt_operations: Mapping[SafetyOperation, Callable[[], object]]


def _verify_live_identity(
    binding: "BoundDevice",
    endpoint: _DeviceEndpoint,
) -> PhysicalDeviceIdentity:
    observed = endpoint.identity_probe()
    if not isinstance(observed, PhysicalDeviceIdentity):
        raise TypeError("live identity probe must return PhysicalDeviceIdentity")
    if observed != endpoint.binding_stamp.physical_identity:
        raise RuntimeError(
            f"live device identity changed for {binding.key}; explicit re-establishment is required"
        )
    return observed


class BoundDevice:
    """Opaque binding reference; it contains no raw adapter callback."""

    __slots__ = (
        "_key",
        "_binding_stamp",
        "_safety_capabilities",
        "_interrupt_capabilities",
        "_session_cleanup_capable",
        "_broker",
    )

    def __init__(
        self,
        token: object,
        *,
        broker: "DeviceBroker",
        endpoint: _DeviceEndpoint,
    ) -> None:
        if token is not _BOUND_DEVICE_TOKEN:
            raise PermissionError("BoundDevice references are created by DeviceBroker.bind")
        object.__setattr__(self, "_key", endpoint.key)
        object.__setattr__(self, "_binding_stamp", endpoint.binding_stamp)
        object.__setattr__(
            self, "_safety_capabilities", frozenset(endpoint.cleanup_operations)
        )
        object.__setattr__(
            self, "_interrupt_capabilities", frozenset(endpoint.interrupt_operations)
        )
        object.__setattr__(
            self,
            "_session_cleanup_capable",
            endpoint.close_session is not None,
        )
        object.__setattr__(self, "_broker", broker)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("BoundDevice is immutable")

    @property
    def key(self) -> ResourceKey:
        return self._key

    @property
    def binding_stamp(self) -> DeviceBindingStamp:
        return self._binding_stamp

    @property
    def binding_instance_id(self) -> str:
        return self._binding_stamp.binding_instance_id

    @property
    def safety_capabilities(self) -> frozenset[SafetyOperation]:
        return self._safety_capabilities

    @property
    def interrupt_capabilities(self) -> frozenset[SafetyOperation]:
        return self._interrupt_capabilities

    @property
    def session_cleanup_capable(self) -> bool:
        return self._session_cleanup_capable

    def validate_capability(
        self,
        proof: VerifiedDeviceCapability,
    ) -> object:
        """Return the snapshot while this is the latest proof for this binding instance."""

        return self._broker.validate_capability(proof, self)


def admit_bound_capability(
    attestation: VerifiedDeviceCapability,
    expected_snapshot_type: type[object],
) -> None:
    """Admit one current broker proof whose snapshot names the same binding.

    Capture and pulse retain their own domain contracts.  This function owns only
    the common broker-proof and physical-binding identity boundary.
    """

    if not isinstance(attestation, VerifiedDeviceCapability):
        raise TypeError("capability attestation must be broker-minted")
    device = attestation.device
    if not isinstance(device, BoundDevice):
        raise TypeError("capability attestation has no BoundDevice")
    snapshot = attestation.snapshot
    if not isinstance(snapshot, expected_snapshot_type):
        raise TypeError(
            "capability attestation has the wrong snapshot type; expected "
            f"{expected_snapshot_type.__name__}"
        )
    if device.validate_capability(attestation) is not snapshot:
        raise RuntimeError("capability attestation snapshot changed")
    stamp = device.binding_stamp
    if getattr(snapshot, "binding_stamp", None) != stamp:
        raise ValueError("capability identity differs from BoundDevice")


class DeviceBroker:
    """Composition authority that owns raw callbacks and active run bindings."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._endpoints: dict[str, _DeviceEndpoint] = {}
        self._binding_by_key: dict[ResourceKey, BoundDevice] = {}
        self._binding_by_stable_identity: dict[str, BoundDevice] = {}
        self._verified_identities: dict[
            object,
            tuple[
                VerifiedPhysicalDeviceIdentity,
                Callable[[], PhysicalDeviceIdentity],
            ],
        ] = {}
        self._verified_capabilities: dict[object, VerifiedDeviceCapability] = {}
        self._capability_nonce_by_binding_instance: dict[str, object] = {}
        self._capability_probe_inflight: set[str] = set()
        self._identity_probe_inflight: set[str] = set()
        self._identity_handshakes = 0
        self._active: dict[str, str] = {}
        self._recovering: dict[ResourceKey, object] = {}
        self._shutdown = False

    def shutdown(self) -> None:
        """Irreversibly invalidate all bindings before raw adapters are closed."""

        with self._lock:
            if self._shutdown:
                return
            if (
                self._active
                or self._recovering
                or self._capability_probe_inflight
                or self._identity_probe_inflight
                or self._identity_handshakes
            ):
                raise RuntimeError("cannot shut down DeviceBroker with active authority")
            self._shutdown = True
            self._verified_identities.clear()
            self._verified_capabilities.clear()
            self._capability_nonce_by_binding_instance.clear()
            self._binding_by_key.clear()
            self._binding_by_stable_identity.clear()
            self._endpoints.clear()
            self._condition.notify_all()

    def _ensure_open(self) -> None:
        if self._shutdown:
            raise RuntimeError("DeviceBroker is shut down")

    def verify_identity(
        self,
        probe: Callable[[], PhysicalDeviceIdentity],
    ) -> VerifiedPhysicalDeviceIdentity:
        if not callable(probe):
            raise TypeError("identity probe must be callable")
        with self._condition:
            self._ensure_open()
            self._identity_handshakes += 1
        try:
            physical_identity = probe()
            if not isinstance(physical_identity, PhysicalDeviceIdentity):
                raise TypeError("identity probe must return PhysicalDeviceIdentity")
            nonce = object()
            identity = VerifiedPhysicalDeviceIdentity(
                _VERIFIED_IDENTITY_TOKEN,
                broker=self,
                physical_identity=physical_identity,
                binding_instance_id=uuid.uuid4().hex,
                nonce=nonce,
            )
            with self._condition:
                self._ensure_open()
                self._verified_identities[nonce] = (identity, probe)
            return identity
        finally:
            with self._condition:
                self._identity_handshakes -= 1
                self._condition.notify_all()

    def discard_verified_identity(self, identity: VerifiedPhysicalDeviceIdentity) -> bool:
        """Revoke an unconsumed identity proof when establishment rolls back."""

        if not isinstance(identity, VerifiedPhysicalDeviceIdentity) or identity._broker is not self:
            raise TypeError("identity proof does not belong to this DeviceBroker")
        with self._lock:
            pending = self._verified_identities.get(identity._nonce)
            if pending is None or pending[0] is not identity:
                return False
            self._verified_identities.pop(identity._nonce, None)
            return True

    def bind(
        self,
        *,
        key: ResourceKey,
        identity: VerifiedPhysicalDeviceIdentity,
        execute_command: Callable[[object], object],
        cleanup_operations: Mapping[SafetyOperation, Callable[[], CleanupStepAck]],
        verify_safe_state: Callable[[], SafeStateAck],
        capability_probe: Callable[[], object] | None = None,
        close_session: Callable[[SessionCloseCommand], SessionClosedAck] | None = None,
        interrupt_operations: Mapping[SafetyOperation, Callable[[], object]] = MappingProxyType({}),
    ) -> BoundDevice:
        if not isinstance(key, ResourceKey):
            raise TypeError("DeviceBroker key must be ResourceKey")
        if not isinstance(identity, VerifiedPhysicalDeviceIdentity) or identity._broker is not self:
            raise TypeError("DeviceBroker.bind requires its own VerifiedPhysicalDeviceIdentity")
        if not callable(execute_command):
            raise TypeError("execute_command must be callable")
        if capability_probe is not None and not callable(capability_probe):
            raise TypeError("capability_probe must be callable")
        if close_session is not None and not callable(close_session):
            raise TypeError("close_session must be callable")
        normalized_cleanup: dict[SafetyOperation, Callable[[], CleanupStepAck]] = {}
        for operation, callback in dict(cleanup_operations).items():
            if not isinstance(operation, SafetyOperation):
                raise TypeError("cleanup operation keys must be SafetyOperation")
            if not callable(callback):
                raise TypeError("cleanup operation values must be callable")
            normalized_cleanup[operation] = callback
        normalized_interrupt: dict[SafetyOperation, Callable[[], object]] = {}
        for operation, callback in dict(interrupt_operations).items():
            if operation not in _INTERRUPT_OPERATIONS:
                raise ValueError("interrupt operation must be ABORT, SAFE_STATE, or DISARM")
            if not callable(callback):
                raise TypeError("interrupt operation values must be callable")
            normalized_interrupt[operation] = callback
        if not callable(verify_safe_state):
            raise TypeError("verify_safe_state must be callable")
        with self._condition:
            self._ensure_open()
            pending = self._verified_identities.get(identity._nonce)
            if pending is None or pending[0] is not identity:
                raise RuntimeError("verified device identity was already consumed")
            identity_probe = pending[1]
            if key in self._binding_by_key:
                raise RuntimeError(
                    f"device {key} is already bound; installation membership is immutable"
                )
            identity_owner = self._binding_by_stable_identity.get(
                identity.physical_identity.stable_device_identity
            )
            if identity_owner is not None:
                raise ValueError(
                    "one stable physical identity cannot bind to multiple ResourceKeys"
                )
            endpoint = _DeviceEndpoint(
                key=key,
                binding_stamp=DeviceBindingStamp(
                    identity.physical_identity,
                    identity.binding_instance_id,
                ),
                identity_probe=identity_probe,
                execute_command=execute_command,
                capability_probe=capability_probe,
                cleanup_operations=MappingProxyType(normalized_cleanup),
                close_session=close_session,
                verify_safe_state=verify_safe_state,
                interrupt_operations=MappingProxyType(normalized_interrupt),
            )
            reference = BoundDevice(
                _BOUND_DEVICE_TOKEN,
                broker=self,
                endpoint=endpoint,
            )
            self._endpoints[identity.binding_instance_id] = endpoint
            self._binding_by_key[key] = reference
            self._binding_by_stable_identity[
                identity.physical_identity.stable_device_identity
            ] = reference
            self._verified_identities.pop(identity._nonce)
            return reference

    def verify_capability(
        self,
        device: BoundDevice,
    ) -> VerifiedDeviceCapability:
        """Freeze a device-owned capability readback for the current binding."""

        if not isinstance(device, BoundDevice) or device._broker is not self:
            raise TypeError("capability verification requires this broker's BoundDevice")
        with self._condition:
            self._ensure_open()
            if device.key in self._recovering:
                raise RuntimeError("cannot probe capability during device recovery")
            if device.binding_instance_id in self._active:
                raise RuntimeError("cannot probe capability while the device is active")
            if device.binding_instance_id in self._capability_probe_inflight:
                raise RuntimeError("device capability probe is already in progress")
            if device.binding_instance_id in self._identity_probe_inflight:
                raise RuntimeError("device identity probe is already in progress")
            endpoint = self._endpoints[device.binding_instance_id]
            probe = endpoint.capability_probe
            if probe is None:
                raise RuntimeError("device binding has no broker-owned capability probe")
            self._capability_probe_inflight.add(device.binding_instance_id)
        try:
            _verify_live_identity(device, endpoint)
            snapshot = probe()
            _verify_live_identity(device, endpoint)
            parameters = getattr(type(snapshot), "__dataclass_params__", None)
            if not is_dataclass(snapshot) or not parameters or not parameters.frozen:
                raise TypeError("capability probe must return a frozen dataclass snapshot")
            nonce = object()
            proof = VerifiedDeviceCapability(
                _VERIFIED_CAPABILITY_TOKEN,
                broker=self,
                device=device,
                snapshot=snapshot,
                nonce=nonce,
            )
            with self._condition:
                if device.binding_instance_id in self._active:
                    raise RuntimeError("device became active during capability probe")
                previous_nonce = self._capability_nonce_by_binding_instance.get(
                    device.binding_instance_id
                )
                if previous_nonce is not None:
                    self._verified_capabilities.pop(previous_nonce, None)
                self._verified_capabilities[nonce] = proof
                self._capability_nonce_by_binding_instance[
                    device.binding_instance_id
                ] = nonce
            return proof
        finally:
            with self._condition:
                self._capability_probe_inflight.discard(device.binding_instance_id)
                self._condition.notify_all()

    def validate_capability(
        self,
        proof: VerifiedDeviceCapability,
        device: BoundDevice,
    ) -> object:
        if (
            not isinstance(proof, VerifiedDeviceCapability)
            or proof._broker is not self
            or proof.device is not device
        ):
            raise TypeError("capability proof does not belong to this device binding")
        with self._condition:
            self._ensure_open()
            if self._verified_capabilities.get(proof._nonce) is not proof:
                raise RuntimeError("capability proof is unknown or revoked")
        return proof.snapshot

    def _open(
        self,
        token: object,
        run_id: str,
        bindings: tuple[BoundDevice, ...],
    ) -> "_DeviceRunLease":
        if token is not _BROKER_OPEN_TOKEN:
            raise PermissionError("device run leases are opened by RunController")
        _canonical_text(run_id, "run_id")
        endpoints: list[tuple[BoundDevice, _DeviceEndpoint]] = []
        with self._condition:
            self._ensure_open()
            for binding in bindings:
                if binding.binding_instance_id in self._capability_probe_inflight:
                    raise RuntimeError(
                        f"device {binding.key} capability probe is in progress"
                    )
                if binding.binding_instance_id in self._identity_probe_inflight:
                    raise RuntimeError(
                        f"device {binding.key} identity probe is already in progress"
                    )
                if binding.key in self._recovering:
                    raise RuntimeError(f"device {binding.key} is undergoing recovery")
                if binding._broker is not self:
                    raise ValueError("all BoundDevice references must belong to one broker")
                if binding.binding_instance_id in self._active:
                    raise RuntimeError(f"device binding {binding.key} is already active")
                endpoints.append(
                    (
                        binding,
                        self._endpoints[binding.binding_instance_id],
                    )
                )
            for binding, _endpoint in endpoints:
                self._identity_probe_inflight.add(binding.binding_instance_id)
        try:
            for binding, endpoint in endpoints:
                _verify_live_identity(binding, endpoint)
            with self._condition:
                for binding, _endpoint in endpoints:
                    if (
                        binding.key in self._recovering
                        or binding.binding_instance_id in self._active
                    ):
                        raise RuntimeError(
                            f"device {binding.key} became unavailable during identity probe"
                        )
                for binding, _endpoint in endpoints:
                    self._active[binding.binding_instance_id] = run_id
        finally:
            with self._condition:
                for binding, _endpoint in endpoints:
                    self._identity_probe_inflight.discard(binding.binding_instance_id)
                self._condition.notify_all()
        return _DeviceRunLease(self, run_id, bindings)

    def _endpoint_for(self, run_id: str, binding: BoundDevice) -> _DeviceEndpoint:
        with self._lock:
            if self._active.get(binding.binding_instance_id) != run_id:
                raise RuntimeError(f"device binding {binding.key} is not active for this Run")
            return self._endpoints[binding.binding_instance_id]

    def _revoke(self, run_id: str, bindings: tuple[BoundDevice, ...]) -> None:
        with self._condition:
            for binding in bindings:
                if self._active.get(binding.binding_instance_id) == run_id:
                    self._active.pop(binding.binding_instance_id, None)
            self._condition.notify_all()

    def _open_recovery(
        self,
        claim: RecoveryClaim,
        binding: BoundDevice,
    ) -> "_DeviceRecoveryLease":
        if not isinstance(claim, RecoveryClaim):
            raise TypeError("recovery verification requires RecoveryClaim")
        if not isinstance(binding, BoundDevice) or binding._broker is not self:
            raise TypeError("recovery binding must belong to this DeviceBroker")
        with self._condition:
            self._ensure_open()
            if binding.key != claim.key:
                raise ValueError("recovery claim and binding identify different resources")
            if binding.binding_instance_id in self._active:
                raise RuntimeError(f"device {claim.key} is active and cannot be recovered")
            if binding.binding_instance_id in self._capability_probe_inflight:
                raise RuntimeError(
                    f"device {claim.key} capability probe blocks recovery"
                )
            if binding.binding_instance_id in self._identity_probe_inflight:
                raise RuntimeError(
                    f"device {claim.key} identity probe blocks recovery"
                )
            if claim.key in self._recovering:
                raise RuntimeError(f"device {claim.key} already has a recovery owner")
            endpoint = self._endpoints[binding.binding_instance_id]
            if endpoint.binding_stamp.physical_identity != claim.physical_identity:
                raise ValueError("recovery binding does not match hazardous physical identity")
            token = object()
            self._recovering[claim.key] = token
            return _DeviceRecoveryLease(self, token, claim, binding, endpoint)

    def _validate_recovery_owner(
        self,
        token: object,
        binding: BoundDevice,
    ) -> None:
        with self._lock:
            if self._recovering.get(binding.key) is not token:
                raise RuntimeError("device recovery ownership is no longer active")

    def _release_recovery(self, token: object, key: ResourceKey) -> None:
        with self._condition:
            if self._recovering.get(key) is not token:
                raise RuntimeError("device recovery ownership is no longer active")
            self._recovering.pop(key)
            self._condition.notify_all()


class _DeviceRecoveryLease:
    __slots__ = ("_broker", "_token", "_claim", "_binding", "_endpoint", "_released")

    def __init__(
        self,
        broker: DeviceBroker,
        token: object,
        claim: RecoveryClaim,
        binding: BoundDevice,
        endpoint: _DeviceEndpoint,
    ) -> None:
        self._broker = broker
        self._token = token
        self._claim = claim
        self._binding = binding
        self._endpoint = endpoint
        self._released = False

    def verify(self) -> RecoveryEvidence:
        self._broker._validate_recovery_owner(
            self._token,
            self._binding,
        )
        _verify_live_identity(self._binding, self._endpoint)
        acknowledgement = self._endpoint.verify_safe_state()
        if not isinstance(acknowledgement, SafeStateAck):
            raise TypeError("device recovery safe-state verifier must return SafeStateAck")
        _verify_live_identity(self._binding, self._endpoint)
        self._broker._validate_recovery_owner(
            self._token,
            self._binding,
        )
        evidence = RecoveryEvidence(
            binding_stamp=self._binding.binding_stamp,
            safe_state_digest=acknowledgement.acknowledgement_digest,
        )
        return evidence

    def release(self) -> None:
        if self._released:
            return
        self._broker._release_recovery(self._token, self._claim.key)
        self._released = True


class _DeviceRunLease:
    __slots__ = ("_broker", "_run_id", "_bindings", "_revoked", "_lock")

    def __init__(
        self,
        broker: DeviceBroker | None,
        run_id: str,
        bindings: tuple[BoundDevice, ...],
    ) -> None:
        self._broker = broker
        self._run_id = run_id
        self._bindings = bindings
        self._revoked = False
        self._lock = threading.Lock()

    def _endpoint(self, binding: BoundDevice) -> _DeviceEndpoint:
        with self._lock:
            if self._revoked:
                raise RuntimeError("device broker lease is revoked")
            if binding not in self._bindings or self._broker is None:
                raise RuntimeError("device binding is not owned by this Run")
            broker = self._broker
        return broker._endpoint_for(self._run_id, binding)

    def execute(self, binding: BoundDevice, command: object) -> object:
        return self._endpoint(binding).execute_command(command)

    def cleanup_step(
        self, binding: BoundDevice, operation: SafetyOperation
    ) -> CleanupStepAck:
        endpoint = self._endpoint(binding)
        try:
            callback = endpoint.cleanup_operations[operation]
        except KeyError as exc:
            raise RuntimeError(
                f"device {binding.key} does not provide safety operation {operation.value}"
            ) from exc
        acknowledgement = callback()
        if not isinstance(acknowledgement, CleanupStepAck):
            raise TypeError(
                f"device {binding.key} cleanup operation {operation.value} must return CleanupStepAck"
            )
        if acknowledgement.operation is not operation:
            raise ValueError(
                f"device {binding.key} returned acknowledgement for "
                f"{acknowledgement.operation.value}, expected {operation.value}"
            )
        return acknowledgement

    def close_session(
        self,
        binding: BoundDevice,
        command: SessionCloseCommand,
    ) -> SessionClosedAck:
        endpoint = self._endpoint(binding)
        callback = endpoint.close_session
        if callback is None:
            raise RuntimeError(f"device {binding.key} has no session cleanup capability")
        acknowledgement = callback(command)
        if not isinstance(acknowledgement, SessionClosedAck):
            raise TypeError(
                f"device {binding.key} session cleanup must return SessionClosedAck"
            )
        if acknowledgement.session_id != command.session_id:
            raise ValueError("session cleanup acknowledgement belongs to another session")
        if acknowledgement.binding_instance_id != binding.binding_instance_id:
            raise ValueError("session cleanup acknowledgement binding differs")
        if not acknowledgement.is_terminal:
            raise RuntimeError(
                f"device {binding.key} session cleanup did not prove stop/drain/join"
            )
        return acknowledgement

    def verify_safe_state(self, binding: BoundDevice) -> SafeReceipt:
        endpoint = self._endpoint(binding)
        _verify_live_identity(binding, endpoint)
        acknowledgement = endpoint.verify_safe_state()
        if not isinstance(acknowledgement, SafeStateAck):
            raise TypeError(
                f"device {binding.key} safe-state verifier must return SafeStateAck"
            )
        _verify_live_identity(binding, endpoint)
        return SafeReceipt(
            key=binding.key,
            binding_stamp=binding.binding_stamp,
            operation_id="VERIFY_SAFE_STATE",
            acknowledgement_digest=acknowledgement.acknowledgement_digest,
        )

    def interrupt(self, binding: BoundDevice, operation: SafetyOperation) -> object:
        endpoint = self._endpoint(binding)
        if operation not in _INTERRUPT_OPERATIONS:
            raise ValueError("operation is not an out-of-band interrupt")
        try:
            callback = endpoint.interrupt_operations[operation]
        except KeyError as exc:
            raise RuntimeError(
                f"device {binding.key} does not provide thread-safe interrupt {operation.value}"
            ) from exc
        return callback()

    def revoke(self) -> None:
        with self._lock:
            if self._revoked:
                return
            self._revoked = True
            broker = self._broker
        if broker is not None:
            broker._revoke(self._run_id, self._bindings)


class RunDevice:
    """Execution-phase proxy; it cannot issue cleanup-only operations."""

    __slots__ = ("_context", "_binding")

    def __init__(self, context: object, binding: BoundDevice) -> None:
        self._context = context
        self._binding = binding

    def execute(self, command: CommandT) -> ResponseT:
        return self._context._execute_bound_device(  # type: ignore[attr-defined, no-any-return]
            self._binding,
            command,
        )


class CleanupDevice:
    """Cleanup-phase proxy restricted to the declared safety operation set."""

    __slots__ = ("_context", "_binding")

    def __init__(self, context: object, binding: BoundDevice) -> None:
        self._context = context
        self._binding = binding

    @property
    def capabilities(self) -> frozenset[SafetyOperation]:
        return self._binding.safety_capabilities

    def perform(self, operation: SafetyOperation) -> CleanupStepAck:
        return self._context._execute_cleanup_step(  # type: ignore[attr-defined, no-any-return]
            self._binding,
            operation,
        )

    def close_session(
        self,
        session_id: str,
        timeout_seconds: float,
    ) -> SessionClosedAck:
        return self._context._close_bound_device_session(  # type: ignore[attr-defined, no-any-return]
            self._binding,
            SessionCloseCommand(session_id, timeout_seconds),
        )

    def verify_safe_state(self) -> SafetyProof:
        return self._context._verify_bound_safe_state(  # type: ignore[attr-defined, no-any-return]
            self._binding,
        )


def verify_cleanup_device_safe_state(
    device: CleanupDevice,
    *,
    failure_reason: str,
    recovery_action: str,
) -> CleanupReport:
    """Read back one device's safe state and retain failure as cleanup evidence."""

    try:
        proof = device.verify_safe_state()
    except BaseException as error:
        return CleanupReport.unsafe(
            (device._binding.key,),
            reason=failure_reason,
            recovery_action=recovery_action,
            errors=(error,),
        )
    return CleanupReport.safe((proof,))


def cleanup_device_session(
    device: CleanupDevice,
    operations: tuple[SafetyOperation, ...],
    session_id: str,
    timeout_seconds: float,
    *,
    termination_failure_reason: str,
    termination_recovery_action: str,
    verification_failure_reason: str,
    verification_recovery_action: str,
) -> CleanupReport:
    """Execute interrupt-safe pre-steps, close one session, then prove safety."""

    errors: list[BaseException] = []
    for operation in operations:
        try:
            device.perform(operation)
        except BaseException as error:
            errors.append(error)
    try:
        device.close_session(session_id, timeout_seconds)
    except BaseException as error:
        errors.append(error)
    if errors:
        return CleanupReport.unsafe(
            (device._binding.key,),
            reason=termination_failure_reason,
            recovery_action=termination_recovery_action,
            errors=tuple(errors),
        )
    return verify_cleanup_device_safe_state(
        device,
        failure_reason=verification_failure_reason,
        recovery_action=verification_recovery_action,
    )


def _open_device_run(
    run_id: str,
    bindings: tuple[BoundDevice, ...],
) -> _DeviceRunLease:
    bindings = tuple(bindings)
    if not bindings:
        return _DeviceRunLease(None, run_id, ())
    broker = bindings[0]._broker
    if any(binding._broker is not broker for binding in bindings):
        raise ValueError("one RunPlan cannot mix DeviceBroker authorities")
    return broker._open(_BROKER_OPEN_TOKEN, run_id, bindings)


class RecoveryAttempt:
    """Exclusive recovery claim; live binding proof is acquired only on complete."""

    __slots__ = (
        "_lease",
        "_devices",
        "_device_lease",
        "_binding",
        "_proof",
        "_durable_result",
        "_result",
        "_lock",
    )

    def __init__(
        self,
        lease: RecoveryLease,
        devices: DeviceBroker,
    ) -> None:
        self._lease = lease
        self._devices = devices
        self._device_lease: _DeviceRecoveryLease | None = None
        self._binding: BoundDevice | None = None
        self._proof: RecoveryEvidence | None = None
        self._durable_result = None
        self._result = None
        self._lock = threading.Lock()

    def complete(self, binding: BoundDevice):
        with self._lock:
            if self._result is not None:
                if binding is not self._binding:
                    raise ValueError("completed recovery cannot be rebound")
                return self._result
            if self._binding is not None and binding is not self._binding:
                raise ValueError("recovery retry must reuse the original binding")
            if self._proof is None:
                device_lease = self._devices._open_recovery(
                    self._lease.claim,
                    binding,
                )
                try:
                    proof = device_lease.verify()
                except BaseException:
                    device_lease.release()
                    raise
                self._device_lease = device_lease
                self._binding = binding
                self._proof = proof
            assert self._proof is not None
            if self._durable_result is None:
                self._durable_result = self._lease._complete(self._proof)
            assert self._device_lease is not None
            self._device_lease.release()
            self._result = self._durable_result
            return self._result

    def abort(self) -> bool:
        with self._lock:
            result = self._lease.abort()
            if self._device_lease is not None:
                self._device_lease.release()
            return result

class RecoveryController:
    """Only path that turns a device recovery probe into quarantine resolution."""

    def __init__(self, resources: ResourceArbiter, devices: DeviceBroker) -> None:
        if not isinstance(resources, ResourceArbiter):
            raise TypeError("resources must be ResourceArbiter")
        if not isinstance(devices, DeviceBroker):
            raise TypeError("devices must be DeviceBroker")
        self._resources = resources
        self._devices = devices

    def begin(self, key: ResourceKey) -> RecoveryAttempt | ResourceBusy | None:
        acquired: RecoveryAcquireResult = self._resources._begin_recovery(key)
        if acquired is None or isinstance(acquired, ResourceBusy):
            return acquired
        assert isinstance(acquired, RecoveryLease)
        return RecoveryAttempt(acquired, self._devices)
