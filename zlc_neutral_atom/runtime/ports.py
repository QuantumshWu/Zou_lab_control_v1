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

from .resources import (
    RecoveryAcquireResult,
    RecoveryClaim,
    RecoveryEvidence,
    RecoveryLease,
    ResourceArbiter,
    ResourceBusy,
    ResourceKey,
    SafeReceipt,
    VerifiedRecoveryProof,
    _mint_verified_recovery,
)


CommandT = TypeVar("CommandT")
ResponseT = TypeVar("ResponseT")


def _canonical_text(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


class SafetyOperation(str, Enum):
    ABORT = "ABORT"
    SAFE_STATE = "SAFE_STATE"
    DISARM = "DISARM"
    READ_STATUS = "READ_STATUS"


class DeviceIdentityEvidenceKind(str, Enum):
    """What the live adapter actually proved about the connected asset."""

    HARDWARE_IDENTITY_READBACK = "HARDWARE_IDENTITY_READBACK"
    INSTALLATION_ASSERTED_ENDPOINT = "INSTALLATION_ASSERTED_ENDPOINT"


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
    binding_id: str
    connection_generation: str
    source_stopped: bool
    no_more_work: bool
    joined: bool
    acknowledgement_digest: str

    def __post_init__(self) -> None:
        for field in ("session_id", "binding_id", "connection_generation"):
            _canonical_text(getattr(self, field), field)
        for field in ("source_stopped", "no_more_work", "joined"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be bool")
        _canonical_text(self.acknowledgement_digest, "acknowledgement_digest")


@dataclass(frozen=True)
class RecoveryAck:
    stable_device_identity: str
    connection_generation: str
    health_digest: str
    safe_state_digest: str
    verified_at: float

    def __post_init__(self) -> None:
        _canonical_text(self.stable_device_identity, "stable_device_identity")
        _canonical_text(self.connection_generation, "connection_generation")
        _canonical_text(self.health_digest, "health_digest")
        _canonical_text(self.safe_state_digest, "safe_state_digest")
        if isinstance(self.verified_at, bool) or not isinstance(
            self.verified_at, (int, float)
        ) or not math.isfinite(float(self.verified_at)):
            raise ValueError("verified_at must be finite")


@dataclass(frozen=True)
class DeviceIdentityAck:
    stable_device_identity: str
    evidence_kind: DeviceIdentityEvidenceKind
    evidence_digest: str
    asset_map_revision: str

    def __post_init__(self) -> None:
        _canonical_text(self.stable_device_identity, "stable_device_identity")
        if not isinstance(self.evidence_kind, DeviceIdentityEvidenceKind):
            raise TypeError("evidence_kind must be DeviceIdentityEvidenceKind")
        _canonical_text(self.evidence_digest, "evidence_digest")
        _canonical_text(self.asset_map_revision, "asset_map_revision")


_INTERRUPT_OPERATIONS = frozenset(
    (SafetyOperation.ABORT, SafetyOperation.SAFE_STATE, SafetyOperation.DISARM)
)
_BOUND_DEVICE_TOKEN = object()
_BROKER_OPEN_TOKEN = object()
_SAFETY_PROOF_TOKEN = object()
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


class SafetyProof:
    """Opaque, run-scoped proof minted after an adapter safety acknowledgement."""

    __slots__ = ("_run_id", "_receipt", "_nonce")

    def __init__(
        self,
        token: object,
        *,
        run_id: str,
        receipt: SafeReceipt,
        nonce: object,
    ) -> None:
        if token is not _SAFETY_PROOF_TOKEN:
            raise PermissionError("SafetyProof can only be minted by RunContext")
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_receipt", receipt)
        object.__setattr__(self, "_nonce", nonce)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("SafetyProof is immutable")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def receipt(self) -> SafeReceipt:
        return self._receipt


class VerifiedDeviceCapability:
    """Broker-minted immutable readback bound to one device generation."""

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

class VerifiedBoundDeviceIdentity:
    """Opaque result of a broker-owned physical identity handshake."""

    __slots__ = (
        "_stable_device_identity",
        "_connection_generation",
        "_evidence_kind",
        "_evidence_digest",
        "_asset_map_revision",
        "_broker",
        "_probe",
        "_nonce",
    )

    def __init__(
        self,
        token: object,
        *,
        broker: "DeviceBroker",
        acknowledgement: DeviceIdentityAck,
        connection_generation: str,
        probe: Callable[[], DeviceIdentityAck],
        nonce: object,
    ) -> None:
        if token is not _VERIFIED_IDENTITY_TOKEN:
            raise PermissionError("device identity must be verified by DeviceBroker")
        object.__setattr__(
            self, "_stable_device_identity", acknowledgement.stable_device_identity
        )
        object.__setattr__(
            self, "_connection_generation", connection_generation
        )
        object.__setattr__(self, "_evidence_kind", acknowledgement.evidence_kind)
        object.__setattr__(self, "_evidence_digest", acknowledgement.evidence_digest)
        object.__setattr__(self, "_asset_map_revision", acknowledgement.asset_map_revision)
        object.__setattr__(self, "_broker", broker)
        object.__setattr__(self, "_probe", probe)
        object.__setattr__(self, "_nonce", nonce)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("VerifiedBoundDeviceIdentity is immutable")

    @property
    def stable_device_identity(self) -> str:
        return self._stable_device_identity

    @property
    def connection_generation(self) -> str:
        return self._connection_generation

    @property
    def evidence_kind(self) -> DeviceIdentityEvidenceKind:
        return self._evidence_kind

    @property
    def evidence_digest(self) -> str:
        return self._evidence_digest

    @property
    def asset_map_revision(self) -> str:
        return self._asset_map_revision


@dataclass(frozen=True)
class _DeviceEndpoint:
    key: ResourceKey
    stable_device_identity: str
    connection_generation: str
    identity_evidence_kind: DeviceIdentityEvidenceKind
    identity_evidence_digest: str
    asset_map_revision: str
    identity_probe: Callable[[], DeviceIdentityAck]
    execute_command: Callable[[object], object]
    capability_probe: Callable[[], object] | None
    cleanup_operations: Mapping[SafetyOperation, Callable[[], CleanupStepAck]]
    close_session: Callable[[SessionCloseCommand], SessionClosedAck] | None
    verify_safe_state: Callable[[], SafeStateAck]
    interrupt_operations: Mapping[SafetyOperation, Callable[[], object]]
    recovery_probe: Callable[[], RecoveryAck] | None


def _verify_live_identity(
    binding: "BoundDevice",
    endpoint: _DeviceEndpoint,
) -> DeviceIdentityAck:
    acknowledgement = endpoint.identity_probe()
    if not isinstance(acknowledgement, DeviceIdentityAck):
        raise TypeError("live identity probe must return DeviceIdentityAck")
    expected = (
        endpoint.stable_device_identity,
        endpoint.identity_evidence_kind,
        endpoint.identity_evidence_digest,
        endpoint.asset_map_revision,
    )
    observed = (
        acknowledgement.stable_device_identity,
        acknowledgement.evidence_kind,
        acknowledgement.evidence_digest,
        acknowledgement.asset_map_revision,
    )
    if observed != expected:
        raise RuntimeError(
            f"live device identity changed for {binding.key}; explicit re-establishment is required"
        )
    return acknowledgement


class BoundDevice:
    """Opaque binding reference; it contains no raw adapter callback."""

    __slots__ = (
        "_key",
        "_stable_device_identity",
        "_connection_generation",
        "_identity_evidence_kind",
        "_identity_evidence_digest",
        "_asset_map_revision",
        "_binding_id",
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
        binding_id: str,
    ) -> None:
        if token is not _BOUND_DEVICE_TOKEN:
            raise PermissionError("BoundDevice references are created by DeviceBroker.bind")
        object.__setattr__(self, "_key", endpoint.key)
        object.__setattr__(
            self, "_stable_device_identity", endpoint.stable_device_identity
        )
        object.__setattr__(
            self, "_connection_generation", endpoint.connection_generation
        )
        object.__setattr__(
            self, "_identity_evidence_kind", endpoint.identity_evidence_kind
        )
        object.__setattr__(
            self, "_identity_evidence_digest", endpoint.identity_evidence_digest
        )
        object.__setattr__(self, "_asset_map_revision", endpoint.asset_map_revision)
        object.__setattr__(self, "_binding_id", binding_id)
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
    def stable_device_identity(self) -> str:
        return self._stable_device_identity

    @property
    def connection_generation(self) -> str:
        return self._connection_generation

    @property
    def identity_evidence_kind(self) -> DeviceIdentityEvidenceKind:
        return self._identity_evidence_kind

    @property
    def identity_evidence_digest(self) -> str:
        return self._identity_evidence_digest

    @property
    def asset_map_revision(self) -> str:
        return self._asset_map_revision

    @property
    def binding_id(self) -> str:
        return self._binding_id

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
        """Return the snapshot only while this exact broker proof is current."""

        return self._broker.validate_capability(proof, self)


class DeviceBroker:
    """Composition authority that owns raw callbacks and active run bindings."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._endpoints: dict[str, _DeviceEndpoint] = {}
        self._current_by_key: dict[ResourceKey, BoundDevice] = {}
        self._current_by_stable_identity: dict[str, BoundDevice] = {}
        self._verified_identities: dict[object, VerifiedBoundDeviceIdentity] = {}
        self._verified_capabilities: dict[object, VerifiedDeviceCapability] = {}
        self._capability_nonce_by_binding: dict[str, object] = {}
        self._capability_probe_inflight: set[str] = set()
        self._identity_probe_inflight: set[str] = set()
        self._activity_epoch_by_binding: dict[str, int] = {}
        self._active: dict[str, str] = {}
        self._recovering: dict[ResourceKey, object] = {}

    def verify_identity(
        self,
        probe: Callable[[], DeviceIdentityAck],
    ) -> VerifiedBoundDeviceIdentity:
        if not callable(probe):
            raise TypeError("identity probe must be callable")
        acknowledgement = probe()
        if not isinstance(acknowledgement, DeviceIdentityAck):
            raise TypeError("identity probe must return DeviceIdentityAck")
        nonce = object()
        identity = VerifiedBoundDeviceIdentity(
            _VERIFIED_IDENTITY_TOKEN,
            broker=self,
            acknowledgement=acknowledgement,
            connection_generation=uuid.uuid4().hex,
            probe=probe,
            nonce=nonce,
        )
        with self._lock:
            self._verified_identities[nonce] = identity
        return identity

    def bind(
        self,
        *,
        key: ResourceKey,
        identity: VerifiedBoundDeviceIdentity,
        execute_command: Callable[[object], object],
        cleanup_operations: Mapping[SafetyOperation, Callable[[], CleanupStepAck]],
        verify_safe_state: Callable[[], SafeStateAck],
        capability_probe: Callable[[], object] | None = None,
        close_session: Callable[[SessionCloseCommand], SessionClosedAck] | None = None,
        interrupt_operations: Mapping[SafetyOperation, Callable[[], object]] = MappingProxyType({}),
        recovery_probe: Callable[[], RecoveryAck] | None = None,
    ) -> BoundDevice:
        if not isinstance(key, ResourceKey):
            raise TypeError("DeviceBroker key must be ResourceKey")
        if not isinstance(identity, VerifiedBoundDeviceIdentity) or identity._broker is not self:
            raise TypeError("DeviceBroker.bind requires its own VerifiedBoundDeviceIdentity")
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
        if recovery_probe is not None and not callable(recovery_probe):
            raise TypeError("recovery_probe must be callable")
        if not callable(verify_safe_state):
            raise TypeError("verify_safe_state must be callable")
        endpoint = _DeviceEndpoint(
            key=key,
            stable_device_identity=identity.stable_device_identity,
            connection_generation=identity.connection_generation,
            identity_evidence_kind=identity.evidence_kind,
            identity_evidence_digest=identity.evidence_digest,
            asset_map_revision=identity.asset_map_revision,
            identity_probe=identity._probe,
            execute_command=execute_command,
            capability_probe=capability_probe,
            cleanup_operations=MappingProxyType(normalized_cleanup),
            close_session=close_session,
            verify_safe_state=verify_safe_state,
            interrupt_operations=MappingProxyType(normalized_interrupt),
            recovery_probe=recovery_probe,
        )
        binding_id = uuid.uuid4().hex
        with self._lock:
            if self._verified_identities.get(identity._nonce) is not identity:
                raise RuntimeError("verified device identity was already consumed")
            if key in self._recovering:
                raise RuntimeError(f"cannot rebind device {key} during recovery")
            previous = self._current_by_key.get(key)
            if previous is not None and previous.binding_id in self._active:
                raise RuntimeError(f"cannot rebind active device {key}")
            if (
                previous is not None
                and previous.binding_id in self._capability_probe_inflight
            ):
                raise RuntimeError(f"cannot rebind device {key} during capability probe")
            if (
                previous is not None
                and previous.binding_id in self._identity_probe_inflight
            ):
                raise RuntimeError(f"cannot rebind device {key} during identity probe")
            identity_owner = self._current_by_stable_identity.get(
                identity.stable_device_identity
            )
            if identity_owner is not None and identity_owner.key != key:
                raise ValueError(
                    "one stable physical identity cannot bind to multiple ResourceKeys"
                )
            if (
                previous is not None
                and previous.stable_device_identity != identity.stable_device_identity
            ):
                raise ValueError(
                    "physical device replacement requires a new ResourceKey/asset mapping"
                )
            if previous is not None:
                previous_nonce = self._capability_nonce_by_binding.pop(
                    previous.binding_id,
                    None,
                )
                if previous_nonce is not None:
                    self._verified_capabilities.pop(previous_nonce, None)
                self._activity_epoch_by_binding.pop(previous.binding_id, None)
                self._endpoints.pop(previous.binding_id, None)
            reference = BoundDevice(
                _BOUND_DEVICE_TOKEN,
                broker=self,
                endpoint=endpoint,
                binding_id=binding_id,
            )
            self._endpoints[binding_id] = endpoint
            self._current_by_key[key] = reference
            self._current_by_stable_identity[identity.stable_device_identity] = reference
            self._activity_epoch_by_binding[binding_id] = 0
            self._verified_identities.pop(identity._nonce)
            return reference

    def verify_capability(
        self,
        device: BoundDevice,
    ) -> VerifiedDeviceCapability:
        """Freeze a device-owned capability readback for the current binding."""

        if not isinstance(device, BoundDevice) or device._broker is not self:
            raise TypeError("capability verification requires this broker's BoundDevice")
        with self._lock:
            if self._current_by_key.get(device.key) is not device:
                raise RuntimeError(f"stale device binding for {device.key}")
            if device.key in self._recovering:
                raise RuntimeError("cannot probe capability during device recovery")
            if device.binding_id in self._active:
                raise RuntimeError("cannot probe capability while the device is active")
            if device.binding_id in self._capability_probe_inflight:
                raise RuntimeError("device capability probe is already in progress")
            if device.binding_id in self._identity_probe_inflight:
                raise RuntimeError("device identity probe is already in progress")
            endpoint = self._endpoints[device.binding_id]
            probe = endpoint.capability_probe
            if probe is None:
                raise RuntimeError("device binding has no broker-owned capability probe")
            activity_epoch = self._activity_epoch_by_binding[device.binding_id]
            self._capability_probe_inflight.add(device.binding_id)
        try:
            snapshot = probe()
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
            with self._lock:
                if self._current_by_key.get(device.key) is not device:
                    raise RuntimeError("device binding changed during capability probe")
                if device.binding_id in self._active:
                    raise RuntimeError("device became active during capability probe")
                if self._activity_epoch_by_binding[device.binding_id] != activity_epoch:
                    raise RuntimeError("a device Run crossed the capability probe")
                previous_nonce = self._capability_nonce_by_binding.get(device.binding_id)
                if previous_nonce is not None:
                    self._verified_capabilities.pop(previous_nonce, None)
                self._verified_capabilities[nonce] = proof
                self._capability_nonce_by_binding[device.binding_id] = nonce
            return proof
        finally:
            with self._lock:
                self._capability_probe_inflight.discard(device.binding_id)

    def current_binding(self, key: ResourceKey) -> BoundDevice:
        """Return the broker-owned reference for the currently established endpoint."""

        if not isinstance(key, ResourceKey):
            raise TypeError("DeviceBroker key must be ResourceKey")
        with self._lock:
            binding = self._current_by_key.get(key)
            if binding is None:
                raise RuntimeError(f"no current device binding for {key}")
            return binding

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
        with self._lock:
            if self._verified_capabilities.get(proof._nonce) is not proof:
                raise RuntimeError("capability proof is unknown or revoked")
            if self._current_by_key.get(device.key) is not device:
                raise RuntimeError("capability proof belongs to a stale device binding")
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
        endpoints: list[tuple[BoundDevice, _DeviceEndpoint, int]] = []
        with self._lock:
            for binding in bindings:
                if binding.binding_id in self._capability_probe_inflight:
                    raise RuntimeError(
                        f"device {binding.key} capability probe is in progress"
                    )
                if binding.binding_id in self._identity_probe_inflight:
                    raise RuntimeError(
                        f"device {binding.key} identity probe is already in progress"
                    )
                if binding.key in self._recovering:
                    raise RuntimeError(f"device {binding.key} is undergoing recovery")
                if binding._broker is not self:
                    raise ValueError("all BoundDevice references must belong to one broker")
                if self._current_by_key.get(binding.key) is not binding:
                    raise RuntimeError(f"stale device binding for {binding.key}")
                if binding.binding_id in self._active:
                    raise RuntimeError(f"device binding {binding.key} is already active")
                endpoints.append(
                    (
                        binding,
                        self._endpoints[binding.binding_id],
                        self._activity_epoch_by_binding[binding.binding_id],
                    )
                )
            for binding, _endpoint, _epoch in endpoints:
                self._identity_probe_inflight.add(binding.binding_id)
        try:
            for binding, endpoint, _epoch in endpoints:
                _verify_live_identity(binding, endpoint)
            with self._lock:
                for binding, endpoint, epoch in endpoints:
                    if self._current_by_key.get(binding.key) is not binding:
                        raise RuntimeError(f"stale device binding for {binding.key}")
                    if self._endpoints.get(binding.binding_id) is not endpoint:
                        raise RuntimeError(f"device endpoint changed for {binding.key}")
                    if self._activity_epoch_by_binding[binding.binding_id] != epoch:
                        raise RuntimeError(
                            f"device activity changed during identity probe for {binding.key}"
                        )
                    if binding.key in self._recovering or binding.binding_id in self._active:
                        raise RuntimeError(
                            f"device {binding.key} became unavailable during identity probe"
                        )
                for binding, _endpoint, _epoch in endpoints:
                    self._active[binding.binding_id] = run_id
                    self._activity_epoch_by_binding[binding.binding_id] += 1
        finally:
            with self._lock:
                for binding, _endpoint, _epoch in endpoints:
                    self._identity_probe_inflight.discard(binding.binding_id)
        return _DeviceRunLease(self, run_id, bindings)

    def _endpoint_for(self, run_id: str, binding: BoundDevice) -> _DeviceEndpoint:
        with self._lock:
            if self._active.get(binding.binding_id) != run_id:
                raise RuntimeError(f"device binding {binding.key} is not active for this Run")
            endpoint = self._endpoints.get(binding.binding_id)
            if endpoint is None:
                raise RuntimeError(f"device binding {binding.key} is stale")
            return endpoint

    def _revoke(self, run_id: str, bindings: tuple[BoundDevice, ...]) -> None:
        with self._lock:
            for binding in bindings:
                if self._active.get(binding.binding_id) == run_id:
                    self._active.pop(binding.binding_id, None)

    def _open_recovery(self, claim: RecoveryClaim) -> "_DeviceRecoveryLease":
        if not isinstance(claim, RecoveryClaim):
            raise TypeError("recovery verification requires RecoveryClaim")
        with self._lock:
            binding = self._current_by_key.get(claim.key)
            if binding is None:
                raise RuntimeError(f"no current device binding for recovery of {claim.key}")
            if binding.binding_id in self._active:
                raise RuntimeError(f"device {claim.key} is active and cannot be recovered")
            if binding.binding_id in self._capability_probe_inflight:
                raise RuntimeError(
                    f"device {claim.key} capability probe blocks recovery"
                )
            if binding.binding_id in self._identity_probe_inflight:
                raise RuntimeError(
                    f"device {claim.key} identity probe blocks recovery"
                )
            if claim.key in self._recovering:
                raise RuntimeError(f"device {claim.key} already has a recovery owner")
            endpoint = self._endpoints[binding.binding_id]
            if endpoint.stable_device_identity != claim.stable_device_identity:
                raise ValueError("recovery binding does not match hazardous stable identity")
            token = object()
            self._recovering[claim.key] = token
            return _DeviceRecoveryLease(self, token, claim, binding, endpoint)

    def _validate_recovery_owner(
        self,
        token: object,
        binding: BoundDevice,
        endpoint: _DeviceEndpoint,
    ) -> None:
        with self._lock:
            if self._recovering.get(binding.key) is not token:
                raise RuntimeError("device recovery ownership is no longer active")
            if self._current_by_key.get(binding.key) is not binding:
                raise RuntimeError("device binding changed during recovery")
            if self._endpoints.get(binding.binding_id) is not endpoint:
                raise RuntimeError("device endpoint changed during recovery")

    def _release_recovery(self, token: object, key: ResourceKey) -> None:
        with self._lock:
            if self._recovering.get(key) is not token:
                raise RuntimeError("device recovery ownership is no longer active")
            self._recovering.pop(key)


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

    def verify(self) -> VerifiedRecoveryProof:
        probe = self._endpoint.recovery_probe
        if probe is None:
            raise RuntimeError(f"device {self._claim.key} has no recovery probe")
        acknowledgement = probe()
        if not isinstance(acknowledgement, RecoveryAck):
            raise TypeError("device recovery probe must return RecoveryAck")
        self._broker._validate_recovery_owner(
            self._token,
            self._binding,
            self._endpoint,
        )
        if acknowledgement.stable_device_identity != self._claim.stable_device_identity:
            raise ValueError("recovery stable device identity does not match hazard")
        if acknowledgement.stable_device_identity != self._endpoint.stable_device_identity:
            raise ValueError("recovery stable device identity does not match binding")
        if acknowledgement.connection_generation != self._endpoint.connection_generation:
            raise ValueError("recovery connection generation does not match binding")
        evidence = RecoveryEvidence(
            stable_device_identity=acknowledgement.stable_device_identity,
            connection_generation=acknowledgement.connection_generation,
            health_digest=acknowledgement.health_digest,
            safe_state_digest=acknowledgement.safe_state_digest,
            verified_at=float(acknowledgement.verified_at),
        )
        return _mint_verified_recovery(self._claim, evidence)

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
        if acknowledgement.binding_id != binding.binding_id:
            raise ValueError("session cleanup acknowledgement binding differs")
        if acknowledgement.connection_generation != binding.connection_generation:
            raise ValueError("session cleanup acknowledgement generation differs")
        return acknowledgement

    def verify_safe_state(self, binding: BoundDevice) -> SafeReceipt:
        endpoint = self._endpoint(binding)
        _verify_live_identity(binding, endpoint)
        acknowledgement = endpoint.verify_safe_state()
        if not isinstance(acknowledgement, SafeStateAck):
            raise TypeError(
                f"device {binding.key} safe-state verifier must return SafeStateAck"
            )
        return SafeReceipt(
            key=binding.key,
            stable_device_identity=binding.stable_device_identity,
            connection_generation=binding.connection_generation,
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


def _mint_safety_proof(
    *,
    run_id: str,
    receipt: SafeReceipt,
    nonce: object,
) -> SafetyProof:
    return SafetyProof(
        _SAFETY_PROOF_TOKEN,
        run_id=run_id,
        receipt=receipt,
        nonce=nonce,
    )


class RecoveryAttempt:
    """Holds one exclusive recovery claim and one verified, retryable proof."""

    __slots__ = ("_lease", "_device_lease", "_proof")

    def __init__(
        self,
        lease: RecoveryLease,
        device_lease: _DeviceRecoveryLease,
        proof: VerifiedRecoveryProof,
    ) -> None:
        self._lease = lease
        self._device_lease = device_lease
        self._proof = proof

    @property
    def claim(self) -> RecoveryClaim:
        return self._lease.claim

    def complete(self):
        result = self._lease.complete(self._proof)
        self._device_lease.release()
        return result

    def abort(self) -> bool:
        result = self._lease.abort()
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
        acquired: RecoveryAcquireResult = self._resources.begin_recovery(key)
        if acquired is None or isinstance(acquired, ResourceBusy):
            return acquired
        assert isinstance(acquired, RecoveryLease)
        device_lease: _DeviceRecoveryLease | None = None
        try:
            device_lease = self._devices._open_recovery(acquired.claim)
            proof = device_lease.verify()
        except BaseException:
            if device_lease is not None:
                device_lease.release()
            acquired.abort()
            raise
        return RecoveryAttempt(acquired, device_lease, proof)
