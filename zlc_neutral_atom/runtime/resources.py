"""Atomic in-process ownership for live device resources."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from zlc_storage import canonical_text as _canonical_text


def _canonical_segment(value: str, field: str) -> str:
    value = _canonical_text(value, field)
    if "/" in value:
        raise ValueError(f"{field} cannot contain '/'")
    return value


@dataclass(frozen=True, order=True)
class ResourceKey:
    """Canonical hierarchical identity owned by the resource provider."""

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

    def child(self, *segments: str) -> "ResourceKey":
        return ResourceKey(self.segments + tuple(segments))

    def overlaps(self, other: "ResourceKey") -> bool:
        if not isinstance(other, ResourceKey):
            return False
        common = min(len(self.segments), len(other.segments))
        return self.segments[:common] == other.segments[:common]

    def __str__(self) -> str:
        return "/".join(self.segments)


class DeviceIdentityEvidenceKind(str, Enum):
    """How an adapter established the identity of a physical asset."""

    HARDWARE_IDENTITY_READBACK = "HARDWARE_IDENTITY_READBACK"
    INSTALLATION_ASSERTED_ENDPOINT = "INSTALLATION_ASSERTED_ENDPOINT"


@dataclass(frozen=True, order=True)
class PhysicalDeviceIdentity:
    """Connection-independent identity and the evidence that supports it."""

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
        "evidence_digest": value.evidence_digest,
        "asset_map_revision": value.asset_map_revision,
    }


def physical_device_identity_from_tree(tree: object) -> PhysicalDeviceIdentity:
    fields = {
        "stable_device_identity",
        "evidence_kind",
        "evidence_digest",
        "asset_map_revision",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("physical device identity has an unknown field set")
    return PhysicalDeviceIdentity(
        stable_device_identity=tree["stable_device_identity"],
        evidence_kind=DeviceIdentityEvidenceKind(tree["evidence_kind"]),
        evidence_digest=tree["evidence_digest"],
        asset_map_revision=tree["asset_map_revision"],
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


class ClaimMode(str, Enum):
    EXCLUSIVE = "EXCLUSIVE"
    OBSERVE = "OBSERVE"


@dataclass(frozen=True, order=True)
class ResourceClaim:
    key: ResourceKey
    mode: ClaimMode = ClaimMode.EXCLUSIVE

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("ResourceClaim.key must be ResourceKey")
        if not isinstance(self.mode, ClaimMode):
            raise TypeError("ResourceClaim.mode must be ClaimMode")


@dataclass(frozen=True)
class ResourceBusy:
    requested: ResourceClaim
    conflicting_run_id: str
    conflicting_claim: ResourceClaim


@dataclass(frozen=True)
class _ActiveLease:
    run_id: str
    claims: tuple[ResourceClaim, ...]


_TERMINAL_PUBLICATION_TOKEN = object()


class TerminalPublication:
    """Runtime-owned terminal transition coupled to resource release."""

    __slots__ = ("_publish", "_after", "_published")

    def __init__(
        self,
        token: object,
        publish: Callable[[], None],
        after: Callable[[], None],
    ) -> None:
        if token is not _TERMINAL_PUBLICATION_TOKEN:
            raise PermissionError("TerminalPublication is runtime-owned")
        object.__setattr__(self, "_publish", publish)
        object.__setattr__(self, "_after", after)
        object.__setattr__(self, "_published", False)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("TerminalPublication is immutable")

    def _publish_under_resource_lock(self, token: object) -> None:
        if token is not _TERMINAL_PUBLICATION_TOKEN:
            raise PermissionError("invalid terminal publication authority")
        if self._published:
            raise RuntimeError("terminal publication may run only once")
        self._publish()
        object.__setattr__(self, "_published", True)

    def _after_resource_release(self, token: object) -> None:
        if token is not _TERMINAL_PUBLICATION_TOKEN or not self._published:
            raise PermissionError("terminal publication is not committed")
        self._after()


def _mint_terminal_publication(
    publish: Callable[[], None],
    after: Callable[[], None],
) -> TerminalPublication:
    return TerminalPublication(_TERMINAL_PUBLICATION_TOKEN, publish, after)


class ResourceLease:
    """Unforgeable ownership capability for one admitted in-process Run."""

    __slots__ = (
        "_arbiter",
        "_capability",
        "_run_id",
        "_claims",
        "_terminal_lock",
        "_released",
        "_disposition",
    )

    def __init__(
        self,
        arbiter: "ResourceArbiter",
        capability: object,
        run_id: str,
        claims: tuple[ResourceClaim, ...],
    ) -> None:
        self._arbiter = arbiter
        self._capability = capability
        self._run_id = run_id
        self._claims = claims
        self._terminal_lock = threading.Lock()
        self._released = False
        self._disposition: str | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def claims(self) -> tuple[ResourceClaim, ...]:
        return self._claims

    @property
    def released(self) -> bool:
        with self._terminal_lock:
            return self._released

    @property
    def disposition(self) -> str | None:
        with self._terminal_lock:
            return self._disposition

    def release_terminal(
        self,
        publication: TerminalPublication,
        *,
        disposition: str,
    ) -> bool:
        if not isinstance(publication, TerminalPublication):
            raise TypeError("release_terminal requires TerminalPublication")
        disposition = _canonical_text(disposition, "terminal disposition")
        with self._terminal_lock:
            if self._released:
                return False
            self._arbiter._release_terminal(
                self._capability,
                self._run_id,
                publication,
            )
            self._disposition = disposition
            self._released = True
            return True

    def _release_unarmed(self) -> bool:
        with self._terminal_lock:
            if self._released:
                return False
            self._arbiter._release_unarmed(self._capability, self._run_id)
            self._disposition = "UNARMED"
            self._released = True
            return True


AcquireResult = ResourceLease | ResourceBusy


class ResourceArbiter:
    """One process's authoritative table of current resource owners."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._active: dict[object, _ActiveLease] = {}
        self._active_by_run: dict[str, object] = {}
        self._closed = False

    def shutdown(self) -> None:
        with self._lock:
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
        with self._lock:
            if self._closed:
                raise RuntimeError("ResourceArbiter is shut down")
            if run_id in self._active_by_run:
                raise RuntimeError(f"run {run_id!r} already owns a resource lease")
            for requested in normalized:
                for active in self._active.values():
                    for held in active.claims:
                        if _claims_conflict(requested, held):
                            return ResourceBusy(requested, active.run_id, held)
            self._active[capability] = _ActiveLease(run_id, normalized)
            self._active_by_run[run_id] = capability
        return ResourceLease(self, capability, run_id, normalized)

    def active_claims(self) -> Mapping[str, tuple[ResourceClaim, ...]]:
        with self._lock:
            return MappingProxyType(
                {active.run_id: active.claims for active in self._active.values()}
            )

    def _release_terminal(
        self,
        capability: object,
        run_id: str,
        publication: TerminalPublication,
    ) -> None:
        with self._lock:
            self._require_active(capability, run_id)
            publication._publish_under_resource_lock(
                _TERMINAL_PUBLICATION_TOKEN
            )
            del self._active[capability]
            del self._active_by_run[run_id]
            self._condition.notify_all()
        publication._after_resource_release(_TERMINAL_PUBLICATION_TOKEN)

    def _release_unarmed(self, capability: object, run_id: str) -> None:
        with self._lock:
            self._require_active(capability, run_id)
            del self._active[capability]
            del self._active_by_run[run_id]
            self._condition.notify_all()

    def _require_active(self, capability: object, run_id: str) -> _ActiveLease:
        active = self._active.get(capability)
        if active is None or active.run_id != run_id:
            raise RuntimeError("resource lease capability is no longer active")
        return active

    @staticmethod
    def _validate_claim_set(
        claims: tuple[ResourceClaim, ...],
    ) -> tuple[ResourceClaim, ...]:
        if any(not isinstance(claim, ResourceClaim) for claim in claims):
            raise TypeError("claims must contain ResourceClaim values")
        normalized = tuple(
            sorted(claims, key=lambda claim: (claim.key, claim.mode.value))
        )
        for index, left in enumerate(normalized):
            for right in normalized[index + 1 :]:
                if left.key.overlaps(right.key):
                    raise ValueError(
                        "one run cannot request overlapping resources "
                        f"{left.key} and {right.key}"
                    )
        return normalized


def _claims_conflict(left: ResourceClaim, right: ResourceClaim) -> bool:
    if not left.key.overlaps(right.key):
        return False
    return not (
        left.mode is ClaimMode.OBSERVE and right.mode is ClaimMode.OBSERVE
    )
