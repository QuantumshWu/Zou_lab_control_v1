"""Atomic in-process ownership for live device resources."""

from __future__ import annotations

import threading
import time
from math import isfinite
from dataclasses import dataclass
from enum import Enum

from zlc_storage import canonical_text as _canonical_text


def _canonical_segment(value: str, field: str) -> str:
    value = _canonical_text(value, field)
    if "/" in value:
        raise ValueError(f"{field} cannot contain '/'")
    return value


@dataclass(frozen=True, order=True)
class ResourceKey:
    """Canonical exact identity owned by the resource provider."""

    segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.segments, tuple):
            raise TypeError("ResourceKey.segments must be a tuple")
        if not self.segments:
            raise ValueError("ResourceKey cannot be empty")
        object.__setattr__(
            self,
            "segments",
            tuple(
                _canonical_segment(segment, "resource segment")
                for segment in self.segments
            ),
        )

    @classmethod
    def parse(cls, value: str) -> "ResourceKey":
        value = _canonical_text(value, "resource key")
        return cls(tuple(value.split("/")))

    def __str__(self) -> str:
        return "/".join(self.segments)


class DeviceIdentityEvidenceKind(str, Enum):
    """How an adapter established the identity of a physical asset."""

    HARDWARE_IDENTITY_READBACK = "HARDWARE_IDENTITY_READBACK"
    INSTALLATION_ASSERTED_ENDPOINT = "INSTALLATION_ASSERTED_ENDPOINT"


@dataclass(frozen=True, order=True)
class PhysicalDeviceIdentity:
    """Connection-independent identity and its deployment-owned source facts."""

    stable_device_identity: str
    evidence_kind: DeviceIdentityEvidenceKind

    def __post_init__(self) -> None:
        _canonical_text(self.stable_device_identity, "stable_device_identity")
        if not isinstance(self.evidence_kind, DeviceIdentityEvidenceKind):
            raise TypeError("evidence_kind must be DeviceIdentityEvidenceKind")


@dataclass(frozen=True, order=True)
class DeviceBindingStamp:
    """Exact physical identity plus one broker-minted binding instance."""

    physical_identity: PhysicalDeviceIdentity
    binding_instance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.physical_identity, PhysicalDeviceIdentity):
            raise TypeError("physical_identity must be PhysicalDeviceIdentity")
        _canonical_segment(self.binding_instance_id, "binding_instance_id")


def physical_device_identity_to_tree(
    value: PhysicalDeviceIdentity,
) -> dict[str, object]:
    if not isinstance(value, PhysicalDeviceIdentity):
        raise TypeError("value must be PhysicalDeviceIdentity")
    return {
        "stable_device_identity": value.stable_device_identity,
        "evidence_kind": value.evidence_kind.value,
    }


def physical_device_identity_from_tree(tree: object) -> PhysicalDeviceIdentity:
    fields = {
        "stable_device_identity",
        "evidence_kind",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("physical device identity has an unknown field set")
    return PhysicalDeviceIdentity(
        stable_device_identity=tree["stable_device_identity"],
        evidence_kind=DeviceIdentityEvidenceKind(tree["evidence_kind"]),
    )


def device_binding_stamp_to_tree(value: DeviceBindingStamp) -> dict[str, object]:
    if not isinstance(value, DeviceBindingStamp):
        raise TypeError("value must be DeviceBindingStamp")
    return {
        "physical_identity": physical_device_identity_to_tree(
            value.physical_identity
        ),
        "binding_instance_id": value.binding_instance_id,
    }


def device_binding_stamp_from_tree(tree: object) -> DeviceBindingStamp:
    fields = {"physical_identity", "binding_instance_id"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("device binding stamp has an unknown field set")
    return DeviceBindingStamp(
        physical_identity=physical_device_identity_from_tree(
            tree["physical_identity"]
        ),
        binding_instance_id=tree["binding_instance_id"],
    )


@dataclass(frozen=True, order=True)
class ResourceClaim:
    key: ResourceKey

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("ResourceClaim.key must be ResourceKey")


@dataclass(frozen=True)
class ResourceBusy:
    requested: ResourceClaim
    conflicting_run_id: str
    conflicting_claim: ResourceClaim


class ResourceLease:
    """Unforgeable ownership capability for one admitted in-process Run."""

    __slots__ = (
        "_arbiter",
        "_capability",
        "_run_id",
        "_release_lock",
        "_released",
    )

    def __init__(
        self,
        arbiter: "ResourceArbiter",
        capability: object,
        run_id: str,
    ) -> None:
        self._arbiter = arbiter
        self._capability = capability
        self._run_id = run_id
        self._release_lock = threading.Lock()
        self._released = False

    @property
    def released(self) -> bool:
        with self._release_lock:
            return self._released

    def release(self) -> bool:
        """Release this Run's claims exactly once.

        Resource ownership ends at the hardware safety boundary.  Run result
        publication and artifact work are deliberately outside this atomic
        resource-table transition.
        """

        with self._release_lock:
            if self._released:
                return False
            self._arbiter._release(self._capability, self._run_id)
            self._released = True
            return True


AcquireResult = ResourceLease | tuple[ResourceBusy, ...]


class ResourceArbiter:
    """One process's authoritative table of current resource owners."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._active: dict[
            str,
            tuple[object, tuple[ResourceClaim, ...]],
        ] = {}
        self._closed = False

    def shutdown(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._active:
                raise RuntimeError(
                    "cannot shut down ResourceArbiter with active ownership"
                )
            self._closed = True
            self._condition.notify_all()

    def acquire_all(
        self,
        run_id: str,
        claims: tuple[ResourceClaim, ...],
    ) -> AcquireResult:
        run_id = _canonical_text(run_id, "run_id")
        normalized = self._validate_claim_set(tuple(claims))
        capability = object()
        with self._condition:
            if self._closed:
                raise RuntimeError("ResourceArbiter is shut down")
            if run_id in self._active:
                raise RuntimeError(f"run {run_id!r} already owns a resource lease")
            blockers: list[ResourceBusy] = []
            for requested in normalized:
                for owner in sorted(self._active):
                    _token, held_claims = self._active[owner]
                    for held in held_claims:
                        if requested.key == held.key:
                            blockers.append(ResourceBusy(requested, owner, held))
            if blockers:
                return tuple(blockers)
            self._active[run_id] = (capability, normalized)
        return ResourceLease(self, capability, run_id)

    def wait_until_released(
        self,
        run_ids: tuple[str, ...],
        *,
        timeout: float | None = None,
    ) -> bool:
        """Wait until none of the exact Run owners holds a hardware lease."""

        if not isinstance(run_ids, tuple):
            raise TypeError("run_ids must be a tuple")
        owners = tuple(_canonical_text(value, "run_id") for value in run_ids)
        if len(set(owners)) != len(owners):
            raise ValueError("run_ids must be unique")
        if not owners:
            return True
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("timeout must be a number or None")
            timeout = float(timeout)
            if not isfinite(timeout) or timeout < 0.0:
                raise ValueError("timeout must be finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while any(owner in self._active for owner in owners):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def _release(self, capability: object, run_id: str) -> None:
        with self._condition:
            self._require_active(capability, run_id)
            del self._active[run_id]
            self._condition.notify_all()

    def _require_active(
        self,
        capability: object,
        run_id: str,
    ) -> tuple[ResourceClaim, ...]:
        active = self._active.get(run_id)
        if active is None or active[0] is not capability:
            raise RuntimeError("resource lease capability is no longer active")
        return active[1]

    @staticmethod
    def _validate_claim_set(
        claims: tuple[ResourceClaim, ...],
    ) -> tuple[ResourceClaim, ...]:
        if any(not isinstance(claim, ResourceClaim) for claim in claims):
            raise TypeError("claims must contain ResourceClaim values")
        normalized = tuple(sorted(claims, key=lambda claim: claim.key))
        for index, left in enumerate(normalized):
            for right in normalized[index + 1 :]:
                if left.key == right.key:
                    raise ValueError(
                        f"one run cannot request resource {left.key} twice"
                    )
        return normalized
