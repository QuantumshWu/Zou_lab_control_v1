"""Atomic in-process resource ownership and quarantine state."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol


def _canonical_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be non-empty text without surrounding whitespace")
    return value


def _canonical_segment(value: str, field: str) -> str:
    value = _canonical_text(value, field)
    if "/" in value:
        raise ValueError(f"{field} cannot contain '/'")
    return value


@dataclass(frozen=True, order=True)
class ResourceKey:
    """A canonical hierarchy such as ``device/qcmos/serial-123``."""

    segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.segments, str):
            raise TypeError("ResourceKey segments must be a tuple, not a string")
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("ResourceKey requires at least one segment")
        segments = tuple(_canonical_segment(segment, "resource segment") for segment in segments)
        object.__setattr__(self, "segments", segments)

    @classmethod
    def parse(cls, value: str) -> "ResourceKey":
        if not isinstance(value, str):
            raise TypeError("resource key text must be str")
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
            raise TypeError("claim key must be ResourceKey")
        if not isinstance(self.mode, ClaimMode):
            raise TypeError("claim mode must be ClaimMode")


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


@dataclass(frozen=True)
class QuarantineRecord:
    record_id: str
    key: ResourceKey
    run_id: str
    reason: str
    recovery_action: str
    recorded_at: float


@dataclass(frozen=True)
class QuarantineResolution:
    record_id: str
    key: ResourceKey
    proof: str
    resolved_at: float


class QuarantineJournal(Protocol):
    """Append-only persistence seam; record-id appends must be idempotent."""

    def unresolved(self) -> tuple[QuarantineRecord, ...]: ...

    def append_quarantined(self, records: tuple[QuarantineRecord, ...]) -> None: ...

    def append_resolved(self, resolution: QuarantineResolution) -> None: ...


class QuarantineJournalError(RuntimeError):
    """A persistence failure that must leave the owning lease unavailable."""


class MemoryQuarantineJournal:
    """Thread-safe journal for virtual adapters and unit tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[QuarantineRecord | QuarantineResolution] = []

    def unresolved(self) -> tuple[QuarantineRecord, ...]:
        with self._lock:
            unresolved: dict[str, QuarantineRecord] = {}
            for entry in self._entries:
                if isinstance(entry, QuarantineRecord):
                    unresolved[entry.record_id] = entry
                else:
                    unresolved.pop(entry.record_id, None)
            return tuple(unresolved.values())

    def append_quarantined(self, records: tuple[QuarantineRecord, ...]) -> None:
        with self._lock:
            existing = {
                entry.record_id: entry
                for entry in self._entries
                if isinstance(entry, QuarantineRecord)
            }
            for record in records:
                previous = existing.get(record.record_id)
                if previous is not None:
                    if previous != record:
                        raise ValueError(
                            f"quarantine record id {record.record_id} has conflicting content"
                        )
                    continue
                self._entries.append(record)
                existing[record.record_id] = record

    def append_resolved(self, resolution: QuarantineResolution) -> None:
        with self._lock:
            self._entries.append(resolution)

    def entries(self) -> tuple[QuarantineRecord | QuarantineResolution, ...]:
        with self._lock:
            return tuple(self._entries)


@dataclass(frozen=True)
class _ActiveLease:
    capability: object
    run_id: str
    claims: tuple[ResourceClaim, ...]


_LEASE_CONSTRUCTION_TOKEN = object()


class ResourceLease:
    """The capability returned by one successful atomic acquire_all.

    Terminal disposition is an atomic choice: either ``release_safe`` or
    ``quarantine_and_release``.  There is no two-call quarantine/release protocol.
    """

    __slots__ = (
        "_arbiter",
        "_capability",
        "_run_id",
        "_claims",
        "_terminal_lock",
        "_disposition",
        "_pending_quarantine",
    )

    def __init__(
        self,
        construction_token: object,
        arbiter: "ResourceArbiter",
        capability: object,
        run_id: str,
        claims: tuple[ResourceClaim, ...],
    ) -> None:
        if construction_token is not _LEASE_CONSTRUCTION_TOKEN:
            raise TypeError("ResourceLease instances are created only by ResourceArbiter")
        self._arbiter = arbiter
        self._capability = capability
        self._run_id = run_id
        self._claims = claims
        self._terminal_lock = threading.Lock()
        self._disposition: str | None = None
        self._pending_quarantine: tuple[QuarantineRecord, ...] | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def claims(self) -> tuple[ResourceClaim, ...]:
        return self._claims

    @property
    def released(self) -> bool:
        with self._terminal_lock:
            return self._disposition in ("SAFE", "QUARANTINED")

    @property
    def disposition(self) -> str | None:
        with self._terminal_lock:
            return self._disposition

    def release_safe(self) -> bool:
        """Atomically mark the lease safely finished and release it exactly once."""

        with self._terminal_lock:
            if self._disposition is not None:
                return False
            self._arbiter._release_safe(self._capability, self._run_id)
            self._disposition = "SAFE"
            return True

    def quarantine_and_release(
        self,
        *,
        keys: tuple[ResourceKey, ...],
        reason: str,
        recovery_action: str,
    ) -> tuple[QuarantineRecord, ...]:
        """Persist quarantine records, then atomically remove the active lease."""

        with self._terminal_lock:
            if self._disposition not in (None, "JOURNAL_WRITE_FAILED"):
                raise RuntimeError(f"resource lease already ended as {self._disposition}")
            if self._pending_quarantine is None:
                self._pending_quarantine = self._arbiter._prepare_quarantine(
                    self._capability,
                    self._run_id,
                    keys=keys,
                    reason=reason,
                    recovery_action=recovery_action,
                )
            else:
                expected = tuple(
                    (record.key, record.reason, record.recovery_action)
                    for record in self._pending_quarantine
                )
                requested = tuple(
                    (key, reason, recovery_action) for key in tuple(keys)
                )
                if requested != expected:
                    raise ValueError(
                        "a quarantine retry must reuse the original keys, reason, and recovery action"
                    )
            try:
                records = self._arbiter._quarantine_and_release(
                    self._capability,
                    self._run_id,
                    records=self._pending_quarantine,
                )
            except QuarantineJournalError:
                self._disposition = "JOURNAL_WRITE_FAILED"
                raise
            self._disposition = "QUARANTINED"
            return records


AcquireResult = ResourceLease | ResourceBusy | ResourceQuarantined


class ResourceArbiter:
    """One composition root's authoritative in-process ownership table."""

    def __init__(self, journal: QuarantineJournal | None = None) -> None:
        self._lock = threading.RLock()
        self._active: dict[object, _ActiveLease] = {}
        self._active_by_run: dict[str, object] = {}
        self._journal = MemoryQuarantineJournal() if journal is None else journal
        self._quarantine: dict[ResourceKey, QuarantineRecord] = {}
        for record in self._journal.unresolved():
            existing = self._quarantine.get(record.key)
            if existing is None or record.recorded_at >= existing.recorded_at:
                self._quarantine[record.key] = record

    def acquire_all(
        self,
        run_id: str,
        claims: tuple[ResourceClaim, ...],
    ) -> AcquireResult:
        run_id = _canonical_segment(run_id, "run_id")
        normalized = self._validate_claim_set(claims)
        with self._lock:
            if run_id in self._active_by_run:
                raise RuntimeError(f"run {run_id!r} already owns a resource lease")
            for requested in normalized:
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
            capability = object()
            active = _ActiveLease(capability, run_id, normalized)
            self._active[capability] = active
            self._active_by_run[run_id] = capability
            return ResourceLease(
                _LEASE_CONSTRUCTION_TOKEN,
                self,
                capability,
                run_id,
                normalized,
            )

    def resolve_quarantine(self, key: ResourceKey, *, proof: str) -> bool:
        """Append resolution after external identity/health/safe verification."""

        proof = _canonical_text(proof, "recovery proof")
        with self._lock:
            if any(
                key.overlaps(claim.key)
                for active in self._active.values()
                for claim in active.claims
            ):
                raise RuntimeError("cannot resolve quarantine while an overlapping lease is active")
            record = self._quarantine.get(key)
            if record is None:
                return False
            resolution = QuarantineResolution(
                record_id=record.record_id,
                key=key,
                proof=proof,
                resolved_at=time.time(),
            )
            self._journal.append_resolved(resolution)
            del self._quarantine[key]
            return True

    def active_claims(self) -> Mapping[str, tuple[ResourceClaim, ...]]:
        with self._lock:
            snapshot = {active.run_id: active.claims for active in self._active.values()}
        return MappingProxyType(snapshot)

    def quarantine_records(self) -> tuple[QuarantineRecord, ...]:
        with self._lock:
            return tuple(sorted(self._quarantine.values(), key=lambda record: str(record.key)))

    def _release_safe(self, capability: object, run_id: str) -> None:
        with self._lock:
            self._require_active(capability, run_id)
            del self._active[capability]
            del self._active_by_run[run_id]

    def _prepare_quarantine(
        self,
        capability: object,
        run_id: str,
        *,
        keys: tuple[ResourceKey, ...],
        reason: str,
        recovery_action: str,
    ) -> tuple[QuarantineRecord, ...]:
        reason = _canonical_text(reason, "quarantine reason")
        recovery_action = _canonical_text(recovery_action, "recovery action")
        selected = tuple(keys)
        if not selected:
            raise ValueError("quarantine requires at least one failed resource key")
        if any(not isinstance(key, ResourceKey) for key in selected):
            raise TypeError("quarantine keys must contain ResourceKey values")
        if len(set(selected)) != len(selected):
            raise ValueError("quarantine keys must be unique")
        with self._lock:
            active = self._require_active(capability, run_id)
            exclusive = {
                claim.key for claim in active.claims if claim.mode is ClaimMode.EXCLUSIVE
            }
            if any(key not in exclusive for key in selected):
                raise ValueError(
                    "only explicitly selected EXCLUSIVE keys owned by this lease may be quarantined"
                )
            now = time.time()
            return tuple(
                QuarantineRecord(
                    record_id=uuid.uuid4().hex,
                    key=key,
                    run_id=active.run_id,
                    reason=reason,
                    recovery_action=recovery_action,
                    recorded_at=now,
                )
                for key in selected
            )

    def _quarantine_and_release(
        self,
        capability: object,
        run_id: str,
        *,
        records: tuple[QuarantineRecord, ...],
    ) -> tuple[QuarantineRecord, ...]:
        records = tuple(records)
        if not records:
            raise ValueError("quarantine commit requires prepared records")
        with self._lock:
            active = self._require_active(capability, run_id)
            exclusive = {
                claim.key for claim in active.claims if claim.mode is ClaimMode.EXCLUSIVE
            }
            if any(
                record.run_id != run_id or record.key not in exclusive
                for record in records
            ):
                raise RuntimeError("prepared quarantine records no longer match the active lease")
            # Persistence is the fail-closed linearization point.  On failure the
            # active lease remains installed and the caller cannot report terminal.
            try:
                self._journal.append_quarantined(records)
            except Exception as exc:
                raise QuarantineJournalError(
                    "failed to persist quarantine; resource lease remains active"
                ) from exc
            for record in records:
                self._quarantine[record.key] = record
            del self._active[capability]
            del self._active_by_run[run_id]
            return records

    def _require_active(self, capability: object, run_id: str) -> _ActiveLease:
        active = self._active.get(capability)
        if active is None or active.capability is not capability or active.run_id != run_id:
            raise RuntimeError("resource lease capability is not active on this arbiter")
        return active

    def _matching_quarantine(self, key: ResourceKey) -> QuarantineRecord | None:
        for quarantined_key, record in self._quarantine.items():
            if key.overlaps(quarantined_key):
                return record
        return None

    @staticmethod
    def _validate_claim_set(claims: tuple[ResourceClaim, ...]) -> tuple[ResourceClaim, ...]:
        normalized = tuple(claims)
        if any(not isinstance(claim, ResourceClaim) for claim in normalized):
            raise TypeError("claims must contain ResourceClaim values")
        for index, claim in enumerate(normalized):
            for other in normalized[index + 1 :]:
                if claim.key.overlaps(other.key):
                    raise ValueError(
                        f"one run cannot declare overlapping claims {claim.key} and {other.key}"
                    )
        return tuple(sorted(normalized))


def _claims_conflict(left: ResourceClaim, right: ResourceClaim) -> bool:
    if not left.key.overlaps(right.key):
        return False
    return left.mode is ClaimMode.EXCLUSIVE or right.mode is ClaimMode.EXCLUSIVE
