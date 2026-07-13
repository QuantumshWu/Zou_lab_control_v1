"""Atomic resource ownership and crash-replayable hardware safety facts."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Protocol
from zlc_storage import canonical_text as _canonical_text


def _canonical_segment(value: str, field: str) -> str:
    value = _canonical_text(value, field)
    if "/" in value:
        raise ValueError(f"{field} cannot contain '/'")
    return value


def _finite_timestamp(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite timestamp")
    normalized = float(value)
    if normalized != normalized or normalized in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be a finite timestamp")
    return normalized


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
class ConnectionEstablishmentClaim:
    """Exclusive, non-hazardous ownership for open + identity + generation bind."""

    attempt_id: str
    keys: tuple[ResourceKey, ...]

    def __post_init__(self) -> None:
        _canonical_segment(self.attempt_id, "connection establishment attempt id")
        if not isinstance(self.keys, tuple) or not self.keys:
            raise ValueError("connection establishment requires at least one ResourceKey")
        if any(not isinstance(key, ResourceKey) for key in self.keys):
            raise TypeError("connection establishment keys must be ResourceKey values")
        canonical = tuple(sorted(set(self.keys)))
        if canonical != self.keys:
            raise ValueError("connection establishment keys must be unique canonical order")


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
    stable_device_identity: str
    connection_generation: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("hazard key must be ResourceKey")
        _canonical_text(self.stable_device_identity, "stable_device_identity")
        _canonical_segment(self.connection_generation, "connection_generation")


@dataclass(frozen=True)
class HazardRecord:
    record_id: str
    key: ResourceKey
    stable_device_identity: str
    connection_generation: str
    run_id: str
    activated_at: float

    def __post_init__(self) -> None:
        _canonical_segment(self.record_id, "hazard record id")
        if not isinstance(self.key, ResourceKey):
            raise TypeError("hazard record key must be ResourceKey")
        _canonical_text(self.stable_device_identity, "stable_device_identity")
        _canonical_segment(self.connection_generation, "connection_generation")
        _canonical_text(self.run_id, "hazard run_id")
        _finite_timestamp(self.activated_at, "hazard activated_at")


class SafetyOutcome(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"


@dataclass(frozen=True, order=True)
class SafeReceipt:
    key: ResourceKey
    stable_device_identity: str
    connection_generation: str
    operation_id: str
    acknowledgement_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("safe receipt key must be ResourceKey")
        _canonical_text(self.stable_device_identity, "stable_device_identity")
        _canonical_segment(self.connection_generation, "connection generation")
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
    hazard_record_id: str | None
    stable_device_identity: str | None
    connection_generation: str | None
    safe_receipt: SafeReceipt | None
    reason: str | None
    recovery_action: str | None

    def __post_init__(self) -> None:
        _canonical_segment(self.disposition_id, "safety disposition id")
        if not isinstance(self.key, ResourceKey):
            raise TypeError("safety disposition key must be ResourceKey")
        if not isinstance(self.outcome, SafetyOutcome):
            raise TypeError("safety disposition outcome must be SafetyOutcome")
        hazard_identity = (
            self.hazard_record_id,
            self.stable_device_identity,
            self.connection_generation,
        )
        if any(value is None for value in hazard_identity) != all(
            value is None for value in hazard_identity
        ):
            raise ValueError("hazard record id, stable identity, and generation must appear together")
        if self.hazard_record_id is not None:
            _canonical_segment(self.hazard_record_id, "hazard record id")
            _canonical_text(self.stable_device_identity, "stable device identity")
            _canonical_segment(self.connection_generation, "connection generation")
        if self.outcome is SafetyOutcome.SAFE:
            if not isinstance(self.safe_receipt, SafeReceipt):
                raise TypeError("SAFE disposition requires SafeReceipt")
            if (
                self.safe_receipt.key != self.key
                or self.safe_receipt.stable_device_identity
                != self.stable_device_identity
                or self.safe_receipt.connection_generation != self.connection_generation
            ):
                raise ValueError("safe receipt does not match disposition key/generation")
            if self.reason is not None or self.recovery_action is not None:
                raise ValueError("SAFE disposition cannot contain quarantine fields")
            if self.hazard_record_id is None:
                raise ValueError("SAFE disposition must resolve an existing hazard")
        else:
            if self.safe_receipt is not None:
                raise ValueError("UNSAFE disposition cannot contain safe receipt")
            if self.hazard_record_id is None:
                raise ValueError("UNSAFE disposition must resolve an existing hazard")
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
        if len({record.disposition_id for record in records}) != len(records):
            raise ValueError("safety disposition ids must be unique")
        if len({record.key for record in records}) != len(records):
            raise ValueError("safety disposition keys must be unique")
        hazard_ids = [
            record.hazard_record_id
            for record in records
            if record.hazard_record_id is not None
        ]
        if len(set(hazard_ids)) != len(hazard_ids):
            raise ValueError("one hazard cannot have multiple safety dispositions")
        object.__setattr__(self, "records", records)
        _finite_timestamp(self.recorded_at, "safety bundle recorded_at")


@dataclass(frozen=True)
class QuarantineRecord:
    record_id: str
    key: ResourceKey
    stable_device_identity: str
    run_id: str
    reason: str
    recovery_action: str
    recorded_at: float
    safety_bundle_id: str

    def __post_init__(self) -> None:
        _canonical_segment(self.record_id, "quarantine record id")
        if not isinstance(self.key, ResourceKey):
            raise TypeError("quarantine key must be ResourceKey")
        _canonical_text(self.stable_device_identity, "stable_device_identity")
        _canonical_text(self.run_id, "quarantine run_id")
        _canonical_text(self.reason, "quarantine reason")
        _canonical_text(self.recovery_action, "quarantine recovery action")
        _finite_timestamp(self.recorded_at, "quarantine recorded_at")
        _canonical_segment(self.safety_bundle_id, "safety bundle id")


@dataclass(frozen=True)
class RecoveryEvidence:
    stable_device_identity: str
    connection_generation: str
    health_digest: str
    safe_state_digest: str
    verified_at: float

    def __post_init__(self) -> None:
        _canonical_text(self.stable_device_identity, "stable device identity")
        _canonical_segment(self.connection_generation, "connection generation")
        _canonical_segment(self.health_digest, "health digest")
        _canonical_segment(self.safe_state_digest, "safe-state digest")
        _finite_timestamp(self.verified_at, "recovery verified_at")


@dataclass(frozen=True)
class RecoveryClaim:
    key: ResourceKey
    stable_device_identity: str
    quarantine_record_ids: tuple[str, ...]
    hazard_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, ResourceKey):
            raise TypeError("recovery claim key must be ResourceKey")
        _canonical_text(self.stable_device_identity, "stable_device_identity")
        quarantine_ids = tuple(self.quarantine_record_ids)
        hazard_ids = tuple(self.hazard_record_ids)
        if not quarantine_ids and not hazard_ids:
            raise ValueError("recovery claim must reference unresolved safety records")
        if len(set(quarantine_ids)) != len(quarantine_ids):
            raise ValueError("recovery quarantine ids must be unique")
        if len(set(hazard_ids)) != len(hazard_ids):
            raise ValueError("recovery hazard ids must be unique")
        for value in quarantine_ids + hazard_ids:
            _canonical_segment(value, "recovery claim record id")
        object.__setattr__(self, "quarantine_record_ids", quarantine_ids)
        object.__setattr__(self, "hazard_record_ids", hazard_ids)


_VERIFIED_RECOVERY_TOKEN = object()


class VerifiedRecoveryProof:
    """Opaque authorization; RecoveryEvidence alone cannot clear safety records."""

    __slots__ = ("_claim", "_evidence")

    def __init__(
        self,
        token: object,
        *,
        claim: RecoveryClaim,
        evidence: RecoveryEvidence,
    ) -> None:
        if token is not _VERIFIED_RECOVERY_TOKEN:
            raise PermissionError("VerifiedRecoveryProof requires device authority")
        object.__setattr__(self, "_claim", claim)
        object.__setattr__(self, "_evidence", evidence)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("VerifiedRecoveryProof is immutable")

    @property
    def claim(self) -> RecoveryClaim:
        return self._claim

    @property
    def evidence(self) -> RecoveryEvidence:
        return self._evidence


def _mint_verified_recovery(
    claim: RecoveryClaim,
    evidence: RecoveryEvidence,
) -> VerifiedRecoveryProof:
    return VerifiedRecoveryProof(
        _VERIFIED_RECOVERY_TOKEN,
        claim=claim,
        evidence=evidence,
    )


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
        _finite_timestamp(self.recorded_at, "recovery recorded_at")


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


class QuarantineJournalError(RuntimeError):
    """A safety fact could not be made durable, so ownership remains fail-closed."""


class HazardEpochExpired(QuarantineJournalError):
    """A retried hazard epoch was already durably resolved and cannot be revived."""


class MemoryQuarantineJournal:
    """Contract-test implementation; real device bootstraps must use persistent storage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[
            HazardRecord | SafetyDispositionBundle | RecoveryBundle
        ] = []

    def snapshot(self) -> SafetyJournalSnapshot:
        with self._lock:
            return _replay_entries(tuple(self._entries))

    def unresolved_hazards(self) -> tuple[HazardRecord, ...]:
        return self.snapshot().unresolved_hazards

    def unresolved_quarantines(self) -> tuple[QuarantineRecord, ...]:
        return self.snapshot().unresolved_quarantines

    def append_hazards(self, records: tuple[HazardRecord, ...]) -> HazardAppendStatus:
        records = tuple(records)
        with self._lock:
            if any(not isinstance(record, HazardRecord) for record in records):
                raise TypeError("hazard append requires HazardRecord values")
            if len({record.record_id for record in records}) != len(records):
                raise ValueError("hazard append record ids must be unique")
            if len({record.key for record in records}) != len(records):
                raise ValueError("hazard append keys must be unique")
            if len({record.run_id for record in records}) > 1:
                raise ValueError("one hazard append must belong to one run")
            if not records:
                return HazardAppendStatus.APPENDED
            additions = self._missing_entries(
                tuple((record.record_id, record) for record in records)
            )
            _replay_entries(tuple(self._entries) + additions)
            self._entries.extend(additions)
            if additions:
                return HazardAppendStatus.APPENDED
            unresolved = {
                record.record_id for record in _replay_entries(tuple(self._entries)).unresolved_hazards
            }
            requested = {record.record_id for record in records}
            if requested <= unresolved:
                return HazardAppendStatus.ALREADY_UNRESOLVED_SAME
            if requested.isdisjoint(unresolved):
                return HazardAppendStatus.ALREADY_RESOLVED
            raise ValueError("hazard retry has partially resolved durable state")

    def append_safety_bundle(self, bundle: SafetyDispositionBundle) -> None:
        if not isinstance(bundle, SafetyDispositionBundle):
            raise TypeError("bundle must be SafetyDispositionBundle")
        with self._lock:
            additions = self._missing_entries(((bundle.bundle_id, bundle),))
            _replay_entries(tuple(self._entries) + additions)
            self._entries.extend(additions)

    def append_recovery_bundle(self, bundle: RecoveryBundle) -> None:
        if not isinstance(bundle, RecoveryBundle):
            raise TypeError("bundle must be RecoveryBundle")
        with self._lock:
            additions = self._missing_entries(((bundle.bundle_id, bundle),))
            _replay_entries(tuple(self._entries) + additions)
            self._entries.extend(additions)

    def entries(
        self,
    ) -> tuple[HazardRecord | SafetyDispositionBundle | RecoveryBundle, ...]:
        with self._lock:
            return tuple(self._entries)

    def _missing_entries(
        self,
        candidates: tuple[
            tuple[str, HazardRecord | SafetyDispositionBundle | RecoveryBundle], ...
        ],
    ) -> tuple[HazardRecord | SafetyDispositionBundle | RecoveryBundle, ...]:
        existing_by_id = {
            (
                entry.record_id if isinstance(entry, HazardRecord) else entry.bundle_id
            ): entry
            for entry in self._entries
        }
        additions = []
        for record_id, value in candidates:
            existing = existing_by_id.get(record_id)
            if existing is not None:
                if existing != value:
                    raise ValueError(
                        f"safety journal id {record_id} has conflicting content"
                    )
                continue
            additions.append(value)
            existing_by_id[record_id] = value
        return tuple(additions)


def _replay_entries(
    entries: tuple[HazardRecord | SafetyDispositionBundle | RecoveryBundle, ...],
) -> SafetyJournalSnapshot:
    hazards: dict[str, HazardRecord] = {}
    quarantines: dict[str, QuarantineRecord] = {}
    seen: dict[str, HazardRecord | SafetyDispositionBundle | RecoveryBundle] = {}
    for entry in entries:
        entry_id = (
            entry.record_id if isinstance(entry, HazardRecord) else entry.bundle_id
        )
        previous_entry = seen.get(entry_id)
        if previous_entry is not None:
            if previous_entry != entry:
                raise ValueError(f"safety journal id {entry_id} has conflicting content")
            continue
        seen[entry_id] = entry
        if isinstance(entry, HazardRecord):
            if any(
                existing.key.overlaps(entry.key)
                for existing in hazards.values()
            ) or any(
                existing.key.overlaps(entry.key)
                for existing in quarantines.values()
            ):
                raise ValueError("hazard keys cannot overlap unresolved safety records")
            hazards[entry.record_id] = entry
            continue
        if isinstance(entry, SafetyDispositionBundle):
            hazards_for_run = {
                record_id
                for record_id, hazard in hazards.items()
                if hazard.run_id == entry.run_id
            }
            referenced = {
                record.hazard_record_id
                for record in entry.records
                if record.hazard_record_id is not None
            }
            if referenced != hazards_for_run:
                raise ValueError(
                    "safety bundle must exactly cover all unresolved hazards for its run"
                )
            for record in entry.records:
                if record.hazard_record_id is not None:
                    hazard = hazards.get(record.hazard_record_id)
                    if hazard is None:
                        raise ValueError("safety bundle references an unknown hazard")
                    if (
                        hazard.run_id != entry.run_id
                        or hazard.key != record.key
                        or hazard.stable_device_identity
                        != record.stable_device_identity
                        or hazard.connection_generation != record.connection_generation
                    ):
                        raise ValueError(
                            "safety disposition does not match hazard run/key/generation"
                        )
                    if entry.recorded_at < hazard.activated_at:
                        raise ValueError("safety disposition predates hazard activation")
                    hazards.pop(record.hazard_record_id)
                if record.outcome is SafetyOutcome.UNSAFE:
                    assert record.reason is not None
                    assert record.recovery_action is not None
                    assert record.stable_device_identity is not None
                    quarantines[record.disposition_id] = QuarantineRecord(
                        record_id=record.disposition_id,
                        key=record.key,
                        stable_device_identity=record.stable_device_identity,
                        run_id=entry.run_id,
                        reason=record.reason,
                        recovery_action=record.recovery_action,
                        recorded_at=entry.recorded_at,
                        safety_bundle_id=entry.bundle_id,
                    )
            continue
        hazards_for_key = {
            record_id
            for record_id, hazard in hazards.items()
            if hazard.key == entry.claim.key
        }
        quarantines_for_key = {
            record_id
            for record_id, quarantine in quarantines.items()
            if quarantine.key == entry.claim.key
        }
        if set(entry.claim.hazard_record_ids) != hazards_for_key:
            raise ValueError("recovery bundle must exactly cover hazards for its key")
        if set(entry.claim.quarantine_record_ids) != quarantines_for_key:
            raise ValueError("recovery bundle must exactly cover quarantines for its key")
        if entry.recorded_at < entry.evidence.verified_at:
            raise ValueError("recovery bundle predates its verified evidence")
        if entry.evidence.stable_device_identity != entry.claim.stable_device_identity:
            raise ValueError("recovery evidence does not match claimed stable identity")
        for record_id in entry.claim.hazard_record_ids:
            hazard = hazards.get(record_id)
            if (
                hazard is None
                or hazard.key != entry.claim.key
                or hazard.stable_device_identity != entry.claim.stable_device_identity
            ):
                raise ValueError("recovery bundle references unknown or cross-key hazard")
            if entry.evidence.verified_at < hazard.activated_at:
                raise ValueError("recovery evidence predates hazard activation")
            hazards.pop(record_id)
        for record_id in entry.claim.quarantine_record_ids:
            quarantine = quarantines.get(record_id)
            if (
                quarantine is None
                or quarantine.key != entry.claim.key
                or quarantine.stable_device_identity != entry.claim.stable_device_identity
            ):
                raise ValueError("recovery bundle references unknown or cross-key quarantine")
            if entry.evidence.verified_at < quarantine.recorded_at:
                raise ValueError("recovery evidence predates quarantine")
            quarantines.pop(record_id)
    return SafetyJournalSnapshot(
        unresolved_hazards=tuple(
            sorted(hazards.values(), key=lambda value: (str(value.key), value.record_id))
        ),
        unresolved_quarantines=tuple(
            sorted(
                quarantines.values(),
                key=lambda value: (str(value.key), value.recorded_at, value.record_id),
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
                        record.stable_device_identity,
                        record.connection_generation,
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
            except QuarantineJournalError:
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
            except QuarantineJournalError:
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


class ConnectionEstablishmentLease:
    """Resource capability for a connection transaction that never changes outputs."""

    __slots__ = ("_arbiter", "_capability", "claim", "_released", "_lock")

    def __init__(
        self,
        arbiter: "ResourceArbiter",
        capability: object,
        claim: ConnectionEstablishmentClaim,
    ) -> None:
        self._arbiter = arbiter
        self._capability = capability
        self.claim = claim
        self._released = False
        self._lock = threading.Lock()

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def close(self) -> bool:
        with self._lock:
            if self._released:
                return False
            self._arbiter._release_connection_establishment(
                self._capability, self.claim
            )
            self._released = True
            return True

    def __enter__(self) -> "ConnectionEstablishmentLease":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.close()
        return False


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

    def complete(self, proof: VerifiedRecoveryProof) -> RecoveryBundle:
        if not isinstance(proof, VerifiedRecoveryProof):
            raise TypeError("recovery completion requires VerifiedRecoveryProof")
        if proof.claim != self.claim:
            raise ValueError("recovery proof does not match the held RecoveryClaim")
        evidence = proof.evidence
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
            self._arbiter._abort_recovery(self._capability, self.claim)
            self._released = True
            return True


AcquireResult = ResourceLease | ResourceBusy | ResourceQuarantined
RecoveryAcquireResult = RecoveryLease | ResourceBusy | None
ConnectionEstablishmentAcquireResult = (
    ConnectionEstablishmentLease | ResourceBusy | ResourceQuarantined
)


class ResourceArbiter:
    """One composition root's authoritative ownership and safety projection."""

    def __init__(self, journal: SafetyJournal) -> None:
        if journal is None:
            raise TypeError(
                "journal is required; real composition uses persistent storage and tests pass MemoryQuarantineJournal explicitly"
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
        self._shutdown = False

    def shutdown(self) -> None:
        """Release installation authority only after all ownership has ended."""

        with self._lock:
            if self._shutdown:
                return
            if self._active or self._journal_pending:
                raise RuntimeError("cannot shut down ResourceArbiter with active ownership")
            self._shutdown = True
            token = self._journal_authority_token
        close_authority = getattr(self._journal, "_close_from_authority", None)
        if callable(close_authority) and token is not None:
            close_authority(token)

    def acquire_all(
        self,
        run_id: str,
        claims: tuple[ResourceClaim, ...],
    ) -> AcquireResult:
        run_id = _canonical_text(run_id, "run_id")
        normalized = self._validate_claim_set(tuple(claims))
        capability = object()
        with self._lock:
            if self._shutdown:
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

    def begin_recovery(self, key: ResourceKey) -> RecoveryAcquireResult:
        if not isinstance(key, ResourceKey):
            raise TypeError("key must be ResourceKey")
        capability = object()
        with self._lock:
            if self._shutdown:
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
            identities = {
                record.stable_device_identity for record in quarantines + hazards
            }
            if len(identities) != 1:
                raise RuntimeError("recovery records disagree on stable device identity")
            claim = RecoveryClaim(
                key=key,
                stable_device_identity=next(iter(identities)),
                quarantine_record_ids=tuple(
                    sorted(record.record_id for record in quarantines)
                ),
                hazard_record_ids=tuple(sorted(record.record_id for record in hazards)),
            )
            run_id = f"recovery:{uuid.uuid4().hex}"
            self._active[capability] = _ActiveLease(run_id, (requested,))
            self._active_by_run[run_id] = capability
            return RecoveryLease(self, capability, claim)

    def begin_connection_establishment(
        self,
        keys: tuple[ResourceKey, ...],
    ) -> ConnectionEstablishmentAcquireResult:
        """Atomically exclude Runs/recovery while adapters establish live connections."""

        normalized_keys = tuple(sorted(set(keys)))
        if not normalized_keys:
            raise ValueError("connection establishment requires at least one ResourceKey")
        if any(not isinstance(key, ResourceKey) for key in normalized_keys):
            raise TypeError("connection establishment keys must be ResourceKey values")
        claims = tuple(ResourceClaim(key, ClaimMode.EXCLUSIVE) for key in normalized_keys)
        attempt_id = uuid.uuid4().hex
        claim = ConnectionEstablishmentClaim(attempt_id, normalized_keys)
        capability = object()
        run_id = f"connection:{attempt_id}"
        with self._lock:
            if self._shutdown:
                raise RuntimeError("ResourceArbiter is shut down")
            for requested in claims:
                hazard = self._matching_hazard(requested.key)
                if hazard is not None:
                    return ResourceQuarantined(
                        requested=requested,
                        reason=f"unresolved hazardous run {hazard.run_id}",
                        recovery_action=(
                            "use RecoveryClaim before reconnecting this device"
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
            self._active[capability] = _ActiveLease(run_id, claims)
            self._active_by_run[run_id] = capability
        return ConnectionEstablishmentLease(self, capability, claim)

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
                    key=lambda record: (str(record.key), record.recorded_at, record.record_id),
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
                    stable_device_identity=hazard.stable_device_identity,
                    connection_generation=hazard.connection_generation,
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
            raise QuarantineJournalError(
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
            if missing:
                raise ValueError(
                    "every active hazard requires exactly one safety decision: "
                    + ", ".join(str(key) for key in sorted(missing))
                )
            if not decisions:
                return None
            now = time.time()
            records = []
            for decision in decisions:
                hazard = hazard_by_key.get(decision.key)
                records.append(
                    SafetyDispositionRecord(
                        disposition_id=uuid.uuid4().hex,
                        key=decision.key,
                        outcome=decision.outcome,
                        hazard_record_id=None if hazard is None else hazard.record_id,
                        stable_device_identity=(
                            None if hazard is None else hazard.stable_device_identity
                        ),
                        connection_generation=(
                            None if hazard is None else hazard.connection_generation
                        ),
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
                raise QuarantineJournalError(
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
                        stable_device_identity=record.stable_device_identity,
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

    def _release_after_safety(self, capability: object, run_id: str) -> None:
        with self._lock:
            self._require_active(capability, run_id)
            if capability not in self._safety_committed:
                raise RuntimeError("safety disposition is not durable")
            del self._active[capability]
            del self._active_by_run[run_id]
            self._safety_committed.pop(capability, None)

    def _quarantines_for_bundle(self, bundle_id: str) -> tuple[QuarantineRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        record
                        for record in self._quarantine.values()
                        if record.safety_bundle_id == bundle_id
                    ),
                    key=lambda record: record.key,
                )
            )

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
            current_quarantines = {
                record.record_id
                for record in self._quarantine.values()
                if record.key == bundle.claim.key
            }
            current_hazards = {
                record.record_id
                for record in self._unresolved_hazards.values()
                if record.key == bundle.claim.key
            }
            if current_quarantines != set(bundle.claim.quarantine_record_ids):
                raise RuntimeError("quarantine records changed during recovery")
            if current_hazards != set(bundle.claim.hazard_record_ids):
                raise RuntimeError("hazard records changed during recovery")
            if bundle.evidence.stable_device_identity != bundle.claim.stable_device_identity:
                raise ValueError("recovery evidence does not match stable device identity")
            referenced_times = [
                self._quarantine[record_id].recorded_at
                for record_id in bundle.claim.quarantine_record_ids
            ] + [
                self._unresolved_hazards[record_id].activated_at
                for record_id in bundle.claim.hazard_record_ids
            ]
            if referenced_times and bundle.evidence.verified_at < max(referenced_times):
                raise ValueError("recovery evidence predates the safety records it clears")
            if capability in self._journal_pending:
                raise RuntimeError("another recovery journal transaction is pending")
            self._journal_pending[capability] = pending
        try:
            self._journal.append_recovery_bundle(bundle)
        except Exception as exc:
            with self._lock:
                if self._journal_pending.get(capability) == pending:
                    self._journal_pending.pop(capability, None)
            raise QuarantineJournalError(
                "failed to persist RecoveryBundle; recovery claim remains held"
            ) from exc
        with self._lock:
            active = self._active.get(capability)
            if active is None:
                raise RuntimeError("recovery capability disappeared during journal commit")
            if self._journal_pending.pop(capability, None) != pending:
                raise RuntimeError("recovery journal transaction identity changed")
            for record_id in bundle.claim.quarantine_record_ids:
                self._quarantine.pop(record_id, None)
            for record_id in bundle.claim.hazard_record_ids:
                self._unresolved_hazards.pop(record_id, None)
            del self._active[capability]
            del self._active_by_run[active.run_id]

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

    def _release_connection_establishment(
        self,
        capability: object,
        claim: ConnectionEstablishmentClaim,
    ) -> None:
        with self._lock:
            active = self._active.get(capability)
            expected = tuple(
                ResourceClaim(key, ClaimMode.EXCLUSIVE) for key in claim.keys
            )
            if active is None or active.claims != expected:
                raise RuntimeError("connection establishment capability is no longer active")
            if capability in self._journal_pending:
                raise RuntimeError("connection establishment cannot own journal I/O")
            del self._active[capability]
            del self._active_by_run[active.run_id]

    def _require_active(self, capability: object, run_id: str) -> _ActiveLease:
        active = self._active.get(capability)
        if active is None or active.run_id != run_id:
            raise RuntimeError("resource lease capability is no longer active")
        return active

    def _matching_quarantine(self, key: ResourceKey) -> QuarantineRecord | None:
        matches = [record for record in self._quarantine.values() if key.overlaps(record.key)]
        if not matches:
            return None
        return max(matches, key=lambda record: (record.recorded_at, record.record_id))

    def _matching_hazard(self, key: ResourceKey) -> HazardRecord | None:
        matches = [
            record for record in self._unresolved_hazards.values() if key.overlaps(record.key)
        ]
        if not matches:
            return None
        return max(matches, key=lambda record: (record.activated_at, record.record_id))

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
