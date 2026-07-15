"""Atomic resource ownership and crash-replayable hardware safety facts."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol
from zlc_storage import canonical_text as _canonical_text, finite_real


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
            tuple(_canonical_segment(segment, "resource segment") for segment in self.segments),
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
    """Encode the resource owner's current physical-identity value."""

    if not isinstance(value, PhysicalDeviceIdentity):
        raise TypeError("value must be PhysicalDeviceIdentity")
    return {
        "stable_device_identity": value.stable_device_identity,
        "evidence_kind": value.evidence_kind.value,
        "evidence_digest": value.evidence_digest,
        "asset_map_revision": value.asset_map_revision,
    }


def physical_device_identity_from_tree(tree: object) -> PhysicalDeviceIdentity:
    """Decode only the resource owner's current physical-identity schema."""

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
    """Encode the resource owner's current binding-stamp value."""

    if not isinstance(value, DeviceBindingStamp):
        raise TypeError("value must be DeviceBindingStamp")
    return {
        "physical_identity": physical_device_identity_to_tree(value.physical_identity),
        "binding_instance_id": value.binding_instance_id,
    }


def device_binding_stamp_from_tree(tree: object) -> DeviceBindingStamp:
    """Decode only the resource owner's current binding-stamp schema."""

    fields = {"physical_identity", "binding_instance_id"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("device binding stamp has an unknown field set")
    return DeviceBindingStamp(
        physical_identity=physical_device_identity_from_tree(tree["physical_identity"]),
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
class ResourceQuarantined:
    requested: ResourceClaim
    reason: str
    recovery_action: str


@dataclass(frozen=True, order=True)
class HazardClaim:
    key: ResourceKey
    binding_stamp: DeviceBindingStamp

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("hazard key must be ResourceKey")
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("hazard binding_stamp must be DeviceBindingStamp")


@dataclass(frozen=True)
class HazardRecord:
    record_id: str
    key: ResourceKey
    binding_stamp: DeviceBindingStamp
    run_id: str
    activated_at: float

    def __post_init__(self) -> None:
        _canonical_segment(self.record_id, "hazard record id")
        if not isinstance(self.key, ResourceKey):
            raise TypeError("hazard record key must be ResourceKey")
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("hazard record binding_stamp must be DeviceBindingStamp")
        _canonical_text(self.run_id, "hazard run_id")
        finite_real(self.activated_at, "hazard activated_at")


class SafetyOutcome(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"


@dataclass(frozen=True, order=True)
class SafeReceipt:
    key: ResourceKey
    binding_stamp: DeviceBindingStamp
    operation_id: str
    acknowledgement_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("safe receipt key must be ResourceKey")
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("safe receipt binding_stamp must be DeviceBindingStamp")
        _canonical_segment(self.operation_id, "safe operation id")
        _canonical_segment(self.acknowledgement_digest, "safe acknowledgement digest")


@dataclass(frozen=True, order=True)
class SafetyDecision:
    """One cleanup fact supplied by the device-owning plan."""

    key: ResourceKey
    outcome: SafetyOutcome
    safe_receipt: SafeReceipt | None = None
    reason: str | None = None
    recovery_action: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("safety decision key must be ResourceKey")
        if not isinstance(self.outcome, SafetyOutcome):
            raise TypeError("safety decision outcome must be SafetyOutcome")
        if self.outcome is SafetyOutcome.SAFE:
            if not isinstance(self.safe_receipt, SafeReceipt):
                raise TypeError("SAFE decision requires SafeReceipt")
            if self.safe_receipt.key != self.key:
                raise ValueError("safe receipt key does not match decision key")
            if self.reason is not None or self.recovery_action is not None:
                raise ValueError("SAFE decision cannot contain quarantine fields")
        else:
            if self.safe_receipt is not None:
                raise ValueError("UNSAFE decision cannot contain a safe receipt")
            _canonical_text(self.reason, "unsafe reason")
            _canonical_text(self.recovery_action, "recovery action")

    @classmethod
    def safe(cls, receipt: SafeReceipt) -> "SafetyDecision":
        return cls(
            key=receipt.key,
            outcome=SafetyOutcome.SAFE,
            safe_receipt=receipt,
        )

    @classmethod
    def unsafe(
        cls,
        key: ResourceKey,
        *,
        reason: str,
        recovery_action: str,
    ) -> "SafetyDecision":
        return cls(
            key=key,
            outcome=SafetyOutcome.UNSAFE,
            reason=reason,
            recovery_action=recovery_action,
        )


@dataclass(frozen=True)
class SafetyDispositionRecord:
    disposition_id: str
    key: ResourceKey
    outcome: SafetyOutcome
    hazard_record_id: str
    binding_stamp: DeviceBindingStamp
    safe_receipt: SafeReceipt | None
    reason: str | None
    recovery_action: str | None

    def __post_init__(self) -> None:
        _canonical_segment(self.disposition_id, "safety disposition id")
        if not isinstance(self.key, ResourceKey):
            raise TypeError("safety disposition key must be ResourceKey")
        if not isinstance(self.outcome, SafetyOutcome):
            raise TypeError("safety disposition outcome must be SafetyOutcome")
        _canonical_segment(self.hazard_record_id, "hazard record id")
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("binding_stamp must be DeviceBindingStamp")
        if self.outcome is SafetyOutcome.SAFE:
            if not isinstance(self.safe_receipt, SafeReceipt):
                raise TypeError("SAFE disposition requires SafeReceipt")
            if (
                self.safe_receipt.key != self.key
                or self.safe_receipt.binding_stamp != self.binding_stamp
            ):
                raise ValueError("safe receipt does not match disposition binding stamp")
            if self.reason is not None or self.recovery_action is not None:
                raise ValueError("SAFE disposition cannot contain quarantine fields")
        else:
            if self.safe_receipt is not None:
                raise ValueError("UNSAFE disposition cannot contain safe receipt")
            _canonical_text(self.reason, "unsafe reason")
            _canonical_text(self.recovery_action, "recovery action")


@dataclass(frozen=True)
class SafetyDispositionBundle:
    """Atomic durable boundary for all device safety facts of one cleanup."""

    bundle_id: str
    run_id: str
    records: tuple[SafetyDispositionRecord, ...]
    recorded_at: float

    def __post_init__(self) -> None:
        _canonical_segment(self.bundle_id, "safety bundle id")
        _canonical_text(self.run_id, "safety bundle run_id")
        records = tuple(self.records)
        if any(not isinstance(record, SafetyDispositionRecord) for record in records):
            raise TypeError("safety bundle records must be SafetyDispositionRecord")
        if not records:
            raise ValueError("safety bundle must resolve at least one hazard")
        if len({record.disposition_id for record in records}) != len(records):
            raise ValueError("safety disposition ids must be unique")
        if len({record.key for record in records}) != len(records):
            raise ValueError("safety disposition keys must be unique")
        hazard_ids = [record.hazard_record_id for record in records]
        if len(set(hazard_ids)) != len(hazard_ids):
            raise ValueError("one hazard cannot have multiple safety dispositions")
        object.__setattr__(self, "records", records)
        finite_real(self.recorded_at, "safety bundle recorded_at")


@dataclass(frozen=True)
class QuarantineRecord:
    record_id: str
    key: ResourceKey
    binding_stamp: DeviceBindingStamp
    run_id: str
    reason: str
    recovery_action: str
    recorded_at: float
    safety_bundle_id: str

    def __post_init__(self) -> None:
        _canonical_segment(self.record_id, "quarantine record id")
        if not isinstance(self.key, ResourceKey):
            raise TypeError("quarantine key must be ResourceKey")
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("quarantine binding_stamp must be DeviceBindingStamp")
        _canonical_text(self.run_id, "quarantine run_id")
        _canonical_text(self.reason, "quarantine reason")
        _canonical_text(self.recovery_action, "quarantine recovery action")
        finite_real(self.recorded_at, "quarantine recorded_at")
        _canonical_segment(self.safety_bundle_id, "safety bundle id")


@dataclass(frozen=True)
class RecoveryEvidence:
    binding_stamp: DeviceBindingStamp
    safe_state_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("recovery binding_stamp must be DeviceBindingStamp")
        _canonical_segment(self.safe_state_digest, "safe-state digest")


@dataclass(frozen=True)
class RecoveryClaim:
    key: ResourceKey
    physical_identity: PhysicalDeviceIdentity
    blocking_record_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("recovery claim key must be ResourceKey")
        if not isinstance(self.physical_identity, PhysicalDeviceIdentity):
            raise TypeError("recovery claim physical_identity must be PhysicalDeviceIdentity")
        _canonical_segment(self.blocking_record_id, "recovery blocking record id")


@dataclass(frozen=True)
class RecoveryBundle:
    bundle_id: str
    claim: RecoveryClaim
    evidence: RecoveryEvidence
    recorded_at: float

    def __post_init__(self) -> None:
        _canonical_segment(self.bundle_id, "recovery bundle id")
        if not isinstance(self.claim, RecoveryClaim):
            raise TypeError("recovery bundle claim must be RecoveryClaim")
        if not isinstance(self.evidence, RecoveryEvidence):
            raise TypeError("recovery bundle evidence must be RecoveryEvidence")
        finite_real(self.recorded_at, "recovery recorded_at")


@dataclass(frozen=True)
class SafetyJournalSnapshot:
    unresolved_hazards: tuple[HazardRecord, ...]
    unresolved_quarantines: tuple[QuarantineRecord, ...]


class HazardAppendStatus(str, Enum):
    APPENDED = "APPENDED"
    ALREADY_UNRESOLVED_SAME = "ALREADY_UNRESOLVED_SAME"
    ALREADY_RESOLVED = "ALREADY_RESOLVED"


class SafetyJournal(Protocol):
    """Each append is one atomic, idempotent framed journal transaction."""

    def snapshot(self) -> SafetyJournalSnapshot: ...

    def append_hazards(
        self, records: tuple[HazardRecord, ...]
    ) -> HazardAppendStatus: ...

    def append_safety_bundle(self, bundle: SafetyDispositionBundle) -> None: ...

    def append_recovery_bundle(self, bundle: RecoveryBundle) -> None: ...


class SafetyJournalWriteError(RuntimeError):
    """A safety fact could not be made durable, so ownership remains fail-closed."""


class HazardEpochExpired(SafetyJournalWriteError):
    """A retried hazard epoch was already durably resolved and cannot be revived."""


class _SafetyProjection:
    """Incremental materialized view of an append-only safety fact stream."""

    def __init__(self) -> None:
        self.hazards: dict[str, HazardRecord] = {}
        self.quarantines: dict[str, QuarantineRecord] = {}
        self.seen: dict[
            str, HazardRecord | SafetyDispositionBundle | RecoveryBundle
        ] = {}
        self.hazard_ids_by_run: dict[str, set[str]] = {}
        self.hazard_ids_by_key: dict[ResourceKey, set[str]] = {}
        self.quarantine_ids_by_key: dict[ResourceKey, set[str]] = {}
        self.blocker_keys: set[tuple[str, ...]] = set()
        self.blocker_prefix_counts: dict[tuple[str, ...], int] = {}

    def _has_overlapping_blocker(self, key: ResourceKey) -> bool:
        segments = key.segments
        return self.blocker_prefix_counts.get(segments, 0) > 0 or any(
            segments[:depth] in self.blocker_keys
            for depth in range(1, len(segments) + 1)
        )

    def _add_blocker(self, key: ResourceKey) -> None:
        if key.segments in self.blocker_keys:
            raise ValueError("one ResourceKey cannot have multiple safety blockers")
        self.blocker_keys.add(key.segments)
        for depth in range(1, len(key.segments) + 1):
            prefix = key.segments[:depth]
            self.blocker_prefix_counts[prefix] = (
                self.blocker_prefix_counts.get(prefix, 0) + 1
            )

    def _remove_blocker(self, key: ResourceKey) -> None:
        self.blocker_keys.remove(key.segments)
        for depth in range(1, len(key.segments) + 1):
            prefix = key.segments[:depth]
            remaining = self.blocker_prefix_counts[prefix] - 1
            if remaining:
                self.blocker_prefix_counts[prefix] = remaining
            else:
                self.blocker_prefix_counts.pop(prefix)

    def apply(
        self,
        entry: HazardRecord | SafetyDispositionBundle | RecoveryBundle,
    ) -> bool:
        entry_id = entry.record_id if isinstance(entry, HazardRecord) else entry.bundle_id
        previous = self.seen.get(entry_id)
        if previous is not None:
            if previous != entry:
                raise ValueError(f"safety journal id {entry_id} has conflicting content")
            return False
        if isinstance(entry, HazardRecord):
            if self._has_overlapping_blocker(entry.key):
                raise ValueError("hazard keys cannot overlap unresolved safety records")
            self.hazards[entry.record_id] = entry
            self.hazard_ids_by_run.setdefault(entry.run_id, set()).add(entry.record_id)
            self.hazard_ids_by_key.setdefault(entry.key, set()).add(entry.record_id)
            self._add_blocker(entry.key)
            self.seen[entry_id] = entry
            return True
        if isinstance(entry, SafetyDispositionBundle):
            expected = self.hazard_ids_by_run.get(entry.run_id, set())
            referenced = {
                record.hazard_record_id for record in entry.records
            }
            if referenced != expected:
                raise ValueError(
                    "safety bundle must exactly cover all unresolved hazards for its run"
                )
            for record in entry.records:
                hazard = self.hazards.get(record.hazard_record_id)
                if (
                    hazard is None
                    or hazard.run_id != entry.run_id
                    or hazard.key != record.key
                    or hazard.binding_stamp != record.binding_stamp
                ):
                    raise ValueError(
                        "safety disposition does not match hazard run/key/binding"
                    )
            for record in entry.records:
                hazard = self.hazards.pop(record.hazard_record_id)
                run_ids = self.hazard_ids_by_run[hazard.run_id]
                run_ids.remove(hazard.record_id)
                if not run_ids:
                    self.hazard_ids_by_run.pop(hazard.run_id)
                key_ids = self.hazard_ids_by_key[hazard.key]
                key_ids.remove(hazard.record_id)
                if not key_ids:
                    self.hazard_ids_by_key.pop(hazard.key)
                self._remove_blocker(hazard.key)
                if record.outcome is SafetyOutcome.UNSAFE:
                    assert record.reason is not None
                    assert record.recovery_action is not None
                    quarantine = QuarantineRecord(
                        record_id=record.disposition_id,
                        key=record.key,
                        binding_stamp=record.binding_stamp,
                        run_id=entry.run_id,
                        reason=record.reason,
                        recovery_action=record.recovery_action,
                        recorded_at=entry.recorded_at,
                        safety_bundle_id=entry.bundle_id,
                    )
                    self.quarantines[quarantine.record_id] = quarantine
                    self.quarantine_ids_by_key.setdefault(quarantine.key, set()).add(
                        quarantine.record_id
                    )
                    self._add_blocker(quarantine.key)
            self.seen[entry_id] = entry
            return True
        blockers = self.hazard_ids_by_key.get(
            entry.claim.key, set()
        ) | self.quarantine_ids_by_key.get(entry.claim.key, set())
        if blockers != {entry.claim.blocking_record_id}:
            raise ValueError("recovery bundle must exactly reference the blocker for its key")
        if (
            entry.evidence.binding_stamp.physical_identity
            != entry.claim.physical_identity
        ):
            raise ValueError("recovery evidence does not match claimed physical identity")
        record_id = entry.claim.blocking_record_id
        hazard = self.hazards.get(record_id)
        quarantine = self.quarantines.get(record_id)
        if hazard is not None:
            if (
                hazard.key != entry.claim.key
                or hazard.binding_stamp.physical_identity != entry.claim.physical_identity
            ):
                raise ValueError("recovery bundle references unknown or cross-key hazard")
        elif quarantine is not None:
            if (
                quarantine.key != entry.claim.key
                or quarantine.binding_stamp.physical_identity
                != entry.claim.physical_identity
            ):
                raise ValueError(
                    "recovery bundle references unknown or cross-key quarantine"
                )
        else:
            raise ValueError("recovery bundle references an unknown blocker")
        if hazard is not None:
            hazard = self.hazards.pop(record_id)
            self.hazard_ids_by_run[hazard.run_id].remove(record_id)
            if not self.hazard_ids_by_run[hazard.run_id]:
                self.hazard_ids_by_run.pop(hazard.run_id)
            self.hazard_ids_by_key[hazard.key].remove(record_id)
            if not self.hazard_ids_by_key[hazard.key]:
                self.hazard_ids_by_key.pop(hazard.key)
            self._remove_blocker(hazard.key)
        else:
            quarantine = self.quarantines.pop(record_id)
            self.quarantine_ids_by_key[quarantine.key].remove(record_id)
            if not self.quarantine_ids_by_key[quarantine.key]:
                self.quarantine_ids_by_key.pop(quarantine.key)
            self._remove_blocker(quarantine.key)
        self.seen[entry_id] = entry
        return True

    def snapshot(self) -> SafetyJournalSnapshot:
        return SafetyJournalSnapshot(
            unresolved_hazards=tuple(
                sorted(
                    self.hazards.values(),
                    key=lambda value: (str(value.key), value.record_id),
                )
            ),
            unresolved_quarantines=tuple(
                sorted(
                    self.quarantines.values(),
                    key=lambda value: (str(value.key), value.record_id),
                )
            ),
        )


@dataclass(frozen=True)
class _ActiveLease:
    run_id: str
    claims: tuple[ResourceClaim, ...]


_TERMINAL_PUBLICATION_TOKEN = object()


class TerminalPublication:
    """Runtime-owned terminal transition; never an arbitrary lease callback."""

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
    """Unforgeable in-process ownership capability with a one-way safety gate."""

    __slots__ = (
        "_arbiter",
        "_capability",
        "_run_id",
        "_claims",
        "_terminal_lock",
        "_state",
        "_pending_hazards",
        "_pending_safety",
        "_safety_bundle",
        "_final_disposition",
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
        self._state = "ACTIVE"
        self._pending_hazards: tuple[HazardRecord, ...] | None = None
        self._pending_safety: SafetyDispositionBundle | None | object = _UNSET
        self._safety_bundle: SafetyDispositionBundle | None = None
        self._final_disposition: str | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def claims(self) -> tuple[ResourceClaim, ...]:
        return self._claims

    @property
    def released(self) -> bool:
        with self._terminal_lock:
            return self._state == "RELEASED"

    @property
    def safety_committed(self) -> bool:
        with self._terminal_lock:
            return self._state in ("SAFETY_COMMITTED", "RELEASED")

    @property
    def safety_bundle(self) -> SafetyDispositionBundle | None:
        with self._terminal_lock:
            return self._safety_bundle

    @property
    def disposition(self) -> str | None:
        with self._terminal_lock:
            if self._state == "RELEASED":
                return self._final_disposition
            return self._state

    def activate_hazards(self, hazards: tuple[HazardClaim, ...]) -> bool:
        """Write HAZARD_ACTIVE before any output-changing hardware capability is enabled."""

        hazards = tuple(hazards)
        with self._terminal_lock:
            if self._state not in ("ACTIVE", "HAZARD_JOURNAL_FAILED"):
                raise RuntimeError(f"cannot activate hazards while lease is {self._state}")
            if self._pending_hazards is None:
                self._pending_hazards = self._arbiter._prepare_hazards(
                    self._capability, self._run_id, hazards
                )
            else:
                original = tuple(
                    HazardClaim(
                        record.key,
                        record.binding_stamp,
                    )
                    for record in self._pending_hazards
                )
                if hazards != original:
                    raise ValueError("hazard activation retry must reuse the original claims")
            try:
                created = self._arbiter._activate_hazards(
                    self._capability, self._run_id, self._pending_hazards
                )
            except HazardEpochExpired:
                self._pending_hazards = None
                self._state = "ACTIVE"
                raise
            except SafetyJournalWriteError:
                self._state = "HAZARD_JOURNAL_FAILED"
                raise
            self._state = "ACTIVE"
            return created

    def _commit_safety(
        self,
        decisions: tuple[SafetyDecision, ...],
    ) -> SafetyDispositionBundle | None:
        """Durably resolve/quarantine every active hazard without releasing ownership."""

        decisions = tuple(decisions)
        with self._terminal_lock:
            if self._state == "SAFETY_COMMITTED":
                expected = self._arbiter._decisions_from_bundle(self._safety_bundle)
                if decisions != expected:
                    raise ValueError("safety commit retry differs from committed bundle")
                return self._safety_bundle
            if self._state not in ("ACTIVE", "SAFETY_JOURNAL_FAILED"):
                raise RuntimeError(f"cannot commit safety while lease is {self._state}")
            if self._pending_safety is _UNSET:
                self._pending_safety = self._arbiter._prepare_safety_bundle(
                    self._capability,
                    self._run_id,
                    decisions,
                )
            else:
                expected = self._arbiter._decisions_from_bundle(
                    None if self._pending_safety is None else self._pending_safety
                )
                if decisions != expected:
                    raise ValueError("safety commit retry must reuse the original decisions")
            bundle = None if self._pending_safety is None else self._pending_safety
            try:
                self._arbiter._commit_safety_bundle(
                    self._capability,
                    self._run_id,
                    bundle,
                )
            except SafetyJournalWriteError:
                self._state = "SAFETY_JOURNAL_FAILED"
                raise
            self._safety_bundle = bundle
            self._state = "SAFETY_COMMITTED"
            return bundle

    def release_terminal(
        self,
        publication: TerminalPublication,
        *,
        disposition: str,
    ) -> bool:
        """Atomically publish runtime terminal state and release the resource claim."""

        if not isinstance(publication, TerminalPublication):
            raise TypeError("release_terminal requires TerminalPublication")
        disposition = _canonical_segment(disposition, "lease disposition")
        with self._terminal_lock:
            if self._state == "RELEASED":
                return False
            if self._state != "SAFETY_COMMITTED":
                raise RuntimeError("resource safety must be durable before terminal release")
            self._arbiter._release_terminal(
                self._capability,
                self._run_id,
                publication=publication,
            )
            self._state = "RELEASED"
            self._final_disposition = disposition
        publication._after_resource_release(_TERMINAL_PUBLICATION_TOKEN)
        return True

    def release_after_safety(self, *, disposition: str) -> bool:
        """Release a low-level lease with no RunHandle terminal transition."""

        disposition = _canonical_segment(disposition, "lease disposition")
        with self._terminal_lock:
            if self._state == "RELEASED":
                return False
            if self._state != "SAFETY_COMMITTED":
                raise RuntimeError("resource safety must be durable before release")
            self._arbiter._release_after_safety(self._capability, self._run_id)
            self._state = "RELEASED"
            self._final_disposition = disposition
            return True

    def _release_unarmed(self) -> bool:
        """Runtime-only rollback before any durable hazard epoch exists."""

        with self._terminal_lock:
            if self._state == "RELEASED":
                return False
            if self._pending_hazards:
                raise RuntimeError("a prepared hazard epoch cannot use unarmed release")
        if self._arbiter._active_hazard_records(self._capability, self._run_id):
            raise RuntimeError("an active hazard epoch cannot use unarmed release")
        self._commit_safety(())
        return self.release_after_safety(disposition="UNARMED")


_UNSET = object()


class RecoveryLease:
    """Exclusive recovery owner; ordinary runs remain blocked until evidence is durable."""

    __slots__ = (
        "_arbiter",
        "_capability",
        "claim",
        "_lock",
        "_pending_bundle",
        "_released",
    )

    def __init__(
        self,
        arbiter: "ResourceArbiter",
        capability: object,
        claim: RecoveryClaim,
    ) -> None:
        self._arbiter = arbiter
        self._capability = capability
        self.claim = claim
        self._lock = threading.Lock()
        self._pending_bundle: RecoveryBundle | None = None
        self._released = False

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def _complete(self, evidence: RecoveryEvidence) -> RecoveryBundle:
        if not isinstance(evidence, RecoveryEvidence):
            raise TypeError("recovery completion requires RecoveryEvidence")
        with self._lock:
            if self._released:
                raise RuntimeError("recovery lease is already released")
            if self._pending_bundle is None:
                self._pending_bundle = RecoveryBundle(
                    bundle_id=uuid.uuid4().hex,
                    claim=self.claim,
                    evidence=evidence,
                    recorded_at=time.time(),
                )
            elif self._pending_bundle.evidence != evidence:
                raise ValueError("recovery retry must reuse the original evidence")
            self._arbiter._complete_recovery(
                self._capability,
                self._pending_bundle,
            )
            self._released = True
            return self._pending_bundle

    def abort(self) -> bool:
        with self._lock:
            if self._released:
                return False
            if self._pending_bundle is not None:
                raise RuntimeError(
                    "recovery journal durability is ambiguous; retry the same "
                    "RecoveryBundle instead of aborting"
                )
            self._arbiter._abort_recovery(self._capability, self.claim)
            self._released = True
            return True


AcquireResult = ResourceLease | ResourceBusy | ResourceQuarantined
RecoveryAcquireResult = RecoveryLease | ResourceBusy | None


class ResourceArbiter:
    """One composition root's authoritative ownership and safety projection."""

    def __init__(self, journal: SafetyJournal) -> None:
        if journal is None:
            raise TypeError(
                "journal is required; composition must provide the persistent safety authority"
            )
        if not all(
            callable(getattr(journal, method, None))
            for method in (
                "snapshot",
                "append_hazards",
                "append_safety_bundle",
                "append_recovery_bundle",
            )
        ):
            raise TypeError("journal does not implement the SafetyJournal contract")
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._active: dict[object, _ActiveLease] = {}
        self._active_by_run: dict[str, object] = {}
        self._journal = journal
        self._journal_authority_token: object | None = None
        bind_authority = getattr(journal, "_bind_authority", None)
        if callable(bind_authority):
            self._journal_authority_token = object()
            bind_authority(self._journal_authority_token)
        try:
            snapshot = journal.snapshot()
        except BaseException:
            close_authority = getattr(journal, "_close_from_authority", None)
            if callable(close_authority) and self._journal_authority_token is not None:
                close_authority(self._journal_authority_token)
            raise
        self._unresolved_hazards: dict[str, HazardRecord] = {
            record.record_id: record for record in snapshot.unresolved_hazards
        }
        self._quarantine: dict[str, QuarantineRecord] = {
            record.record_id: record for record in snapshot.unresolved_quarantines
        }
        self._active_hazards: dict[object, tuple[HazardRecord, ...]] = {}
        self._safety_committed: dict[object, SafetyDispositionBundle | None] = {}
        self._journal_pending: dict[object, tuple[str, object]] = {}
        self._shutdown_state = "OPEN"
        self._shutdown_done = threading.Event()
        self._shutdown_error: str | None = None

    def shutdown(self) -> None:
        """Release installation authority only after all ownership has ended."""

        close_authority = None
        token = None
        owner = False
        with self._lock:
            if self._shutdown_state == "CLOSED":
                return
            if self._shutdown_state == "OPEN":
                if self._active or self._journal_pending:
                    raise RuntimeError(
                        "cannot shut down ResourceArbiter with active ownership"
                    )
                self._shutdown_state = "CLOSING"
                token = self._journal_authority_token
                close_authority = getattr(
                    self._journal, "_close_from_authority", None
                )
                owner = True
        if not owner:
            self._shutdown_done.wait()
            if self._shutdown_error is not None:
                raise RuntimeError(
                    f"ResourceArbiter shutdown failed: {self._shutdown_error}"
                )
            return
        try:
            if callable(close_authority) and token is not None:
                close_authority(token)
        except BaseException as exc:
            with self._lock:
                self._shutdown_error = f"{type(exc).__name__}: {exc}"
                self._shutdown_state = "FAILED"
                self._shutdown_done.set()
                self._condition.notify_all()
            raise
        with self._lock:
            self._shutdown_state = "CLOSED"
            self._shutdown_done.set()
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
            if self._shutdown_state != "OPEN":
                raise RuntimeError("ResourceArbiter is shut down")
            if run_id in self._active_by_run:
                raise RuntimeError(f"run {run_id!r} already owns a resource lease")
            for requested in normalized:
                hazard = self._matching_hazard(requested.key)
                if hazard is not None:
                    return ResourceQuarantined(
                        requested=requested,
                        reason=f"unresolved hazardous run {hazard.run_id}",
                        recovery_action=(
                            "verify device identity, connection generation, health, and safe state"
                        ),
                    )
                quarantined = self._matching_quarantine(requested.key)
                if quarantined is not None:
                    return ResourceQuarantined(
                        requested=requested,
                        reason=quarantined.reason,
                        recovery_action=quarantined.recovery_action,
                    )
                for active in self._active.values():
                    for held in active.claims:
                        if _claims_conflict(requested, held):
                            return ResourceBusy(requested, active.run_id, held)
            active = _ActiveLease(run_id, normalized)
            self._active[capability] = active
            self._active_by_run[run_id] = capability
        return ResourceLease(self, capability, run_id, normalized)

    def _begin_recovery(self, key: ResourceKey) -> RecoveryAcquireResult:
        if not isinstance(key, ResourceKey):
            raise TypeError("key must be ResourceKey")
        capability = object()
        with self._lock:
            if self._shutdown_state != "OPEN":
                raise RuntimeError("ResourceArbiter is shut down")
            requested = ResourceClaim(key, ClaimMode.EXCLUSIVE)
            for active in self._active.values():
                for held in active.claims:
                    if _claims_conflict(requested, held):
                        return ResourceBusy(requested, active.run_id, held)
            quarantines = tuple(
                record for record in self._quarantine.values() if record.key == key
            )
            hazards = tuple(
                record for record in self._unresolved_hazards.values() if record.key == key
            )
            if not quarantines and not hazards:
                return None
            if len(quarantines) + len(hazards) != 1:
                raise RuntimeError(
                    "safety projection has multiple blockers for one ResourceKey"
                )
            blocker = (quarantines + hazards)[0]
            claim = RecoveryClaim(
                key=key,
                physical_identity=blocker.binding_stamp.physical_identity,
                blocking_record_id=blocker.record_id,
            )
            run_id = f"recovery:{uuid.uuid4().hex}"
            self._active[capability] = _ActiveLease(run_id, (requested,))
            self._active_by_run[run_id] = capability
            return RecoveryLease(self, capability, claim)

    def active_claims(self) -> Mapping[str, tuple[ResourceClaim, ...]]:
        with self._lock:
            return MappingProxyType(
                {active.run_id: active.claims for active in self._active.values()}
            )

    def quarantine_records(self) -> tuple[QuarantineRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._quarantine.values(),
                    key=lambda record: (str(record.key), record.record_id),
                )
            )

    def unresolved_hazards(self) -> tuple[HazardRecord, ...]:
        with self._lock:
            records = tuple(self._unresolved_hazards.values())
            for active in self._active_hazards.values():
                records += active
            return tuple(
                sorted(records, key=lambda record: (str(record.key), record.record_id))
            )

    def _prepare_hazards(
        self,
        capability: object,
        run_id: str,
        hazards: tuple[HazardClaim, ...],
    ) -> tuple[HazardRecord, ...]:
        if len(set(hazards)) != len(hazards):
            raise ValueError("hazard claims must be unique")
        if any(not isinstance(hazard, HazardClaim) for hazard in hazards):
            raise TypeError("hazards must contain HazardClaim values")
        with self._lock:
            active = self._require_active(capability, run_id)
            exclusive = {
                claim.key for claim in active.claims if claim.mode is ClaimMode.EXCLUSIVE
            }
            if any(hazard.key not in exclusive for hazard in hazards):
                raise ValueError("hazard claims must match EXCLUSIVE resources owned by the run")
            now = time.time()
            return tuple(
                HazardRecord(
                    record_id=uuid.uuid4().hex,
                    key=hazard.key,
                    binding_stamp=hazard.binding_stamp,
                    run_id=run_id,
                    activated_at=now,
                )
                for hazard in hazards
            )

    def _activate_hazards(
        self,
        capability: object,
        run_id: str,
        records: tuple[HazardRecord, ...],
    ) -> bool:
        pending = ("HAZARD_ACTIVE", records)
        with self._lock:
            self._require_active(capability, run_id)
            existing = self._active_hazards.get(capability)
            if existing is not None:
                if existing != records:
                    raise RuntimeError("active hazard records differ from retry")
                return False
            if capability in self._journal_pending:
                raise RuntimeError("another safety journal transaction is already pending")
            self._journal_pending[capability] = pending
        try:
            status = self._journal.append_hazards(records)
            if not isinstance(status, HazardAppendStatus):
                raise TypeError("SafetyJournal.append_hazards must return HazardAppendStatus")
            if status is HazardAppendStatus.ALREADY_RESOLVED:
                raise HazardEpochExpired(
                    "durable hazard epoch was already resolved; start a new Run"
                )
        except HazardEpochExpired:
            with self._lock:
                if self._journal_pending.get(capability) == pending:
                    self._journal_pending.pop(capability, None)
            raise
        except Exception as exc:
            with self._lock:
                if self._journal_pending.get(capability) == pending:
                    self._journal_pending.pop(capability, None)
            raise SafetyJournalWriteError(
                "failed to persist HAZARD_ACTIVE; no hardware operation may start"
            ) from exc
        with self._lock:
            self._require_active(capability, run_id)
            if self._journal_pending.pop(capability, None) != pending:
                raise RuntimeError("hazard journal transaction identity changed")
            self._active_hazards[capability] = records
            return True

    def _active_hazard_records(
        self,
        capability: object,
        run_id: str,
    ) -> tuple[HazardRecord, ...]:
        with self._lock:
            self._require_active(capability, run_id)
            return self._active_hazards.get(capability, ())

    def _prepare_safety_bundle(
        self,
        capability: object,
        run_id: str,
        decisions: tuple[SafetyDecision, ...],
    ) -> SafetyDispositionBundle | None:
        if any(not isinstance(decision, SafetyDecision) for decision in decisions):
            raise TypeError("decisions must contain SafetyDecision values")
        if len({decision.key for decision in decisions}) != len(decisions):
            raise ValueError("safety decisions must have unique ResourceKeys")
        decisions = tuple(sorted(decisions, key=lambda value: value.key))
        with self._lock:
            active = self._require_active(capability, run_id)
            exclusive = {
                claim.key for claim in active.claims if claim.mode is ClaimMode.EXCLUSIVE
            }
            if any(decision.key not in exclusive for decision in decisions):
                raise ValueError("safety decisions must target EXCLUSIVE resources owned by run")
            hazards = self._active_hazards.get(capability, ())
            hazard_by_key = {record.key: record for record in hazards}
            decision_by_key = {decision.key: decision for decision in decisions}
            missing = set(hazard_by_key) - set(decision_by_key)
            extra = set(decision_by_key) - set(hazard_by_key)
            if missing or extra:
                raise ValueError(
                    "safety decisions must exactly cover active hazards; missing="
                    + ",".join(str(key) for key in sorted(missing))
                    + "; extra="
                    + ",".join(str(key) for key in sorted(extra))
                )
            if not decisions:
                return None
            now = time.time()
            records = []
            for decision in decisions:
                hazard = hazard_by_key[decision.key]
                records.append(
                    SafetyDispositionRecord(
                        disposition_id=uuid.uuid4().hex,
                        key=decision.key,
                        outcome=decision.outcome,
                        hazard_record_id=hazard.record_id,
                        binding_stamp=hazard.binding_stamp,
                        safe_receipt=decision.safe_receipt,
                        reason=decision.reason,
                        recovery_action=decision.recovery_action,
                    )
                )
            return SafetyDispositionBundle(
                bundle_id=uuid.uuid4().hex,
                run_id=run_id,
                records=tuple(records),
                recorded_at=now,
            )

    def _decisions_from_bundle(
        self,
        bundle: SafetyDispositionBundle | None,
    ) -> tuple[SafetyDecision, ...]:
        if bundle is None:
            return ()
        return tuple(
            SafetyDecision(
                key=record.key,
                outcome=record.outcome,
                safe_receipt=record.safe_receipt,
                reason=record.reason,
                recovery_action=record.recovery_action,
            )
            for record in bundle.records
        )

    def _commit_safety_bundle(
        self,
        capability: object,
        run_id: str,
        bundle: SafetyDispositionBundle | None,
    ) -> None:
        pending = ("SAFETY_DISPOSITION", bundle)
        with self._lock:
            self._require_active(capability, run_id)
            existing = self._safety_committed.get(capability, _UNSET)
            if existing is not _UNSET:
                if existing != bundle:
                    raise RuntimeError("committed safety bundle differs from retry")
                return
            if capability in self._journal_pending:
                raise RuntimeError("another safety journal transaction is already pending")
            self._journal_pending[capability] = pending
        if bundle is not None:
            try:
                self._journal.append_safety_bundle(bundle)
            except Exception as exc:
                with self._lock:
                    if self._journal_pending.get(capability) == pending:
                        self._journal_pending.pop(capability, None)
                raise SafetyJournalWriteError(
                    "failed to persist SafetyDispositionBundle; claims remain held"
                ) from exc
        with self._lock:
            self._require_active(capability, run_id)
            if self._journal_pending.pop(capability, None) != pending:
                raise RuntimeError("safety journal transaction identity changed")
            if bundle is not None:
                for record in bundle.records:
                    if record.outcome is not SafetyOutcome.UNSAFE:
                        continue
                    assert record.reason is not None
                    assert record.recovery_action is not None
                    self._quarantine[record.disposition_id] = QuarantineRecord(
                        record_id=record.disposition_id,
                        key=record.key,
                        binding_stamp=record.binding_stamp,
                        run_id=run_id,
                        reason=record.reason,
                        recovery_action=record.recovery_action,
                        recorded_at=bundle.recorded_at,
                        safety_bundle_id=bundle.bundle_id,
                    )
            self._active_hazards.pop(capability, None)
            self._safety_committed[capability] = bundle

    def _release_terminal(
        self,
        capability: object,
        run_id: str,
        *,
        publication: TerminalPublication,
    ) -> None:
        with self._lock:
            self._require_active(capability, run_id)
            if capability not in self._safety_committed:
                raise RuntimeError("safety disposition is not durable")
            publication._publish_under_resource_lock(_TERMINAL_PUBLICATION_TOKEN)
            del self._active[capability]
            del self._active_by_run[run_id]
            self._safety_committed.pop(capability, None)
            self._condition.notify_all()

    def _release_after_safety(self, capability: object, run_id: str) -> None:
        with self._lock:
            self._require_active(capability, run_id)
            if capability not in self._safety_committed:
                raise RuntimeError("safety disposition is not durable")
            del self._active[capability]
            del self._active_by_run[run_id]
            self._safety_committed.pop(capability, None)
            self._condition.notify_all()

    def _complete_recovery(
        self,
        capability: object,
        bundle: RecoveryBundle,
    ) -> None:
        pending = ("RECOVERY", bundle)
        with self._lock:
            active = self._active.get(capability)
            if active is None or active.claims != (
                ResourceClaim(bundle.claim.key, ClaimMode.EXCLUSIVE),
            ):
                raise RuntimeError("recovery capability is no longer active")
            current_blockers = {
                record.record_id
                for record in self._quarantine.values()
                if record.key == bundle.claim.key
            } | {
                record.record_id
                for record in self._unresolved_hazards.values()
                if record.key == bundle.claim.key
            }
            if current_blockers != {bundle.claim.blocking_record_id}:
                raise RuntimeError("safety blocker changed during recovery")
            if (
                bundle.evidence.binding_stamp.physical_identity
                != bundle.claim.physical_identity
            ):
                raise ValueError("recovery evidence does not match physical device identity")
            if capability in self._journal_pending:
                raise RuntimeError("another recovery journal transaction is pending")
            self._journal_pending[capability] = pending
        try:
            self._journal.append_recovery_bundle(bundle)
        except Exception as exc:
            with self._lock:
                if self._journal_pending.get(capability) == pending:
                    self._journal_pending.pop(capability, None)
            raise SafetyJournalWriteError(
                "failed to persist RecoveryBundle; recovery claim remains held"
            ) from exc
        with self._lock:
            active = self._active.get(capability)
            if active is None:
                raise RuntimeError("recovery capability disappeared during journal commit")
            if self._journal_pending.pop(capability, None) != pending:
                raise RuntimeError("recovery journal transaction identity changed")
            self._quarantine.pop(bundle.claim.blocking_record_id, None)
            self._unresolved_hazards.pop(bundle.claim.blocking_record_id, None)
            del self._active[capability]
            del self._active_by_run[active.run_id]
            self._condition.notify_all()

    def _abort_recovery(
        self,
        capability: object,
        claim: RecoveryClaim,
    ) -> None:
        with self._lock:
            active = self._active.get(capability)
            if active is None or active.claims != (
                ResourceClaim(claim.key, ClaimMode.EXCLUSIVE),
            ):
                raise RuntimeError("recovery capability is no longer active")
            if capability in self._journal_pending:
                raise RuntimeError("cannot abort while recovery journal I/O is pending")
            del self._active[capability]
            del self._active_by_run[active.run_id]
            self._condition.notify_all()

    def _require_active(self, capability: object, run_id: str) -> _ActiveLease:
        active = self._active.get(capability)
        if active is None or active.run_id != run_id:
            raise RuntimeError("resource lease capability is no longer active")
        return active

    def _matching_quarantine(self, key: ResourceKey) -> QuarantineRecord | None:
        matches = [record for record in self._quarantine.values() if key.overlaps(record.key)]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError("safety projection has overlapping quarantine blockers")
        return matches[0]

    def _matching_hazard(self, key: ResourceKey) -> HazardRecord | None:
        matches = [
            record for record in self._unresolved_hazards.values() if key.overlaps(record.key)
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError("safety projection has overlapping hazard blockers")
        return matches[0]

    @staticmethod
    def _validate_claim_set(
        claims: tuple[ResourceClaim, ...],
    ) -> tuple[ResourceClaim, ...]:
        if not claims:
            return ()
        if any(not isinstance(claim, ResourceClaim) for claim in claims):
            raise TypeError("claims must contain ResourceClaim values")
        normalized = tuple(sorted(claims, key=lambda claim: (claim.key, claim.mode.value)))
        for index, left in enumerate(normalized):
            for right in normalized[index + 1 :]:
                if left.key.overlaps(right.key):
                    raise ValueError(
                        f"one run cannot request overlapping resources {left.key} and {right.key}"
                    )
        return normalized


def _claims_conflict(left: ResourceClaim, right: ResourceClaim) -> bool:
    if not left.key.overlaps(right.key):
        return False
    return not (
        left.mode is ClaimMode.OBSERVE and right.mode is ClaimMode.OBSERVE
    )
