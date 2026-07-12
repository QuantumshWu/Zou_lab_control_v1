"""Durable final-artifact commit intents and restart reconciliation state."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Protocol, TypeVar

from zlc_storage.framed_journal import FramedJournal


CommitT = TypeVar("CommitT")


def _canonical(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _sha256(value: str, field: str) -> str:
    value = _canonical(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class CommitTarget:
    repository_id: str
    artifact_kind: str
    schema_version: str
    target_ref: str
    expected_manifest_digest: str

    def __post_init__(self) -> None:
        _canonical(self.repository_id, "repository_id")
        _canonical(self.artifact_kind, "artifact_kind")
        _canonical(self.schema_version, "schema_version")
        _canonical(self.target_ref, "target_ref")
        _sha256(self.expected_manifest_digest, "expected_manifest_digest")


@dataclass(frozen=True)
class CommitIntent:
    commit_id: str
    run_id: str
    safety_bundle_id: str | None
    target: CommitTarget
    created_at: float

    def __post_init__(self) -> None:
        _canonical(self.commit_id, "commit_id")
        _canonical(self.run_id, "run_id")
        if self.safety_bundle_id is not None:
            _canonical(self.safety_bundle_id, "safety_bundle_id")
        if not isinstance(self.target, CommitTarget):
            raise TypeError("target must be CommitTarget")
        if isinstance(self.created_at, bool) or not isinstance(
            self.created_at, (int, float)
        ) or not math.isfinite(float(self.created_at)):
            raise ValueError("created_at must be finite")


@dataclass(frozen=True)
class CommitRecovery(Generic[CommitT]):
    committed: bool
    result: "PublishedManifest[CommitT] | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.committed, bool):
            raise TypeError("CommitRecovery.committed must be bool")
        if self.committed and not isinstance(self.result, PublishedManifest):
            raise TypeError("committed recovery requires a PublishedManifest")
        if not self.committed and self.result is not None:
            raise ValueError("uncommitted recovery cannot contain a committed result")


@dataclass(frozen=True)
class ReconciledCommit(Generic[CommitT]):
    intent: CommitIntent
    recovery: CommitRecovery[CommitT]


class CommitJournal(Protocol):
    repository_id: str
    durable: bool

    def begin(self, intent: CommitIntent) -> None: ...

    def mark_committed(self, commit_id: str) -> None: ...

    def mark_aborted(self, commit_id: str) -> None: ...

    def pending(self) -> tuple[CommitIntent, ...]: ...


@dataclass(frozen=True)
class PublishedManifest(Generic[CommitT]):
    target_ref: str
    manifest_digest: str
    result: CommitT

    def __post_init__(self) -> None:
        _canonical(self.target_ref, "published target_ref")
        _sha256(self.manifest_digest, "published manifest_digest")


class PublishVisibilityUnknown(RuntimeError):
    """The repository cannot yet determine whether its manifest became visible."""


def _validate_published_manifest(
    published: PublishedManifest[CommitT],
    target: CommitTarget,
) -> CommitT:
    if not isinstance(published, PublishedManifest):
        raise TypeError("repository publish must return PublishedManifest")
    if published.target_ref != target.target_ref:
        raise ValueError("published manifest target_ref differs from CommitTarget")
    if published.manifest_digest != target.expected_manifest_digest:
        raise ValueError("published manifest digest differs from CommitTarget")
    return published.result


def _validate_commit_recovery(
    recovery: CommitRecovery[CommitT],
    target: CommitTarget,
) -> CommitRecovery[CommitT]:
    if not isinstance(recovery, CommitRecovery):
        raise TypeError("recover must return CommitRecovery")
    if recovery.committed:
        assert recovery.result is not None
        _validate_published_manifest(recovery.result, target)
    return recovery


_COMMIT_AUTHORITY_TOKEN = object()


class CommitAuthority(Generic[CommitT]):
    """Side-effect-free handle to coordinator-owned commit capabilities."""

    __slots__ = ("_coordinator", "_nonce", "_target")

    def __init__(
        self,
        token: object,
        *,
        coordinator: "RepositoryCommitCoordinator[CommitT]",
        nonce: object,
        target: CommitTarget,
    ) -> None:
        if token is not _COMMIT_AUTHORITY_TOKEN:
            raise PermissionError(
                "CommitAuthority must be minted by RepositoryCommitCoordinator"
            )
        if not isinstance(target, CommitTarget):
            raise TypeError("CommitAuthority.target must be CommitTarget")
        if not isinstance(coordinator, RepositoryCommitCoordinator):
            raise TypeError("CommitAuthority coordinator is invalid")
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_nonce", nonce)
        object.__setattr__(self, "_target", target)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CommitAuthority is immutable")

    @property
    def target(self) -> CommitTarget:
        return self._target


@dataclass(frozen=True)
class _CommitAuthoritySnapshot(Generic[CommitT]):
    target: CommitTarget
    journal: CommitJournal
    publish_callback: Callable[[], PublishedManifest[CommitT]]
    recover_callback: Callable[[CommitIntent], CommitRecovery[CommitT]]

    def publish_validated(self) -> CommitT:
        return _validate_published_manifest(self.publish_callback(), self.target)

    def recover_validated(self, intent: CommitIntent) -> CommitRecovery[CommitT]:
        return _validate_commit_recovery(self.recover_callback(intent), self.target)


_COMMIT_AUTHORITY_CONSUMER_TOKEN = object()


def _consume_commit_authority(
    authority: CommitAuthority[CommitT],
) -> _CommitAuthoritySnapshot[CommitT]:
    if not isinstance(authority, CommitAuthority):
        raise TypeError("final commit authority is invalid")
    return authority._coordinator._consume_authority(  # noqa: SLF001
        _COMMIT_AUTHORITY_CONSUMER_TOKEN,
        authority,
    )


@dataclass(frozen=True)
class FinalCommit(Generic[CommitT]):
    commit_id: str
    safety_bundle_id: str | None
    authority: CommitAuthority[CommitT]

    def __post_init__(self) -> None:
        _canonical(self.commit_id, "final commit_id")
        if self.safety_bundle_id is not None:
            _canonical(self.safety_bundle_id, "final commit safety_bundle_id")
        if not isinstance(self.authority, CommitAuthority):
            raise TypeError("FinalCommit.authority must be CommitAuthority")

    @property
    def target(self) -> CommitTarget:
        return self.authority.target


def reconcile_pending_commits(
    journal: CommitJournal,
    repository_id: str,
    recover: Callable[[CommitIntent], CommitRecovery[CommitT]],
) -> tuple[ReconciledCommit[CommitT], ...]:
    """Resolve durable intents by inspection only; this function never publishes."""

    if not callable(recover):
        raise TypeError("recover must be callable")
    repository_id = _canonical(repository_id, "repository_id")
    if journal.repository_id != repository_id:
        raise ValueError("startup reconciler does not own this commit journal")
    reconciled = []
    for intent in journal.pending():
        if intent.target.repository_id != repository_id:
            raise ValueError("pending commit belongs to another repository")
        resolution = _validate_commit_recovery(recover(intent), intent.target)
        if resolution.committed:
            journal.mark_committed(intent.commit_id)
        else:
            journal.mark_aborted(intent.commit_id)
        reconciled.append(ReconciledCommit(intent, resolution))
    return tuple(reconciled)


class RepositoryCommitCoordinator(Generic[CommitT]):
    """Startup gate and sole factory for one repository's commit authority."""

    def __init__(
        self,
        journal: CommitJournal,
        recover: Callable[[CommitIntent], CommitRecovery[CommitT]],
        *,
        allow_ephemeral: bool = False,
    ) -> None:
        repository_id = getattr(journal, "repository_id", None)
        if not isinstance(repository_id, str):
            raise TypeError("commit journal must expose repository_id")
        if not callable(recover):
            raise TypeError("recover must be callable")
        if not isinstance(allow_ephemeral, bool):
            raise TypeError("allow_ephemeral must be bool")
        durable = getattr(journal, "durable", None)
        if not isinstance(durable, bool):
            raise TypeError("commit journal must declare durability")
        if not durable and not allow_ephemeral:
            raise ValueError("ephemeral commit journals are forbidden for production authority")
        self.repository_id = repository_id
        self._journal = journal
        self._recover = recover
        self._authority_lock = threading.Lock()
        self._authorities: dict[object, _CommitAuthoritySnapshot[CommitT]] = {}
        self._startup_reconciliations = reconcile_pending_commits(
            journal,
            repository_id,
            recover,
        )

    @property
    def startup_reconciliations(self) -> tuple[ReconciledCommit[CommitT], ...]:
        return self._startup_reconciliations

    def prepare(
        self,
        target: CommitTarget,
        publish: Callable[[], PublishedManifest[CommitT]],
    ) -> CommitAuthority[CommitT]:
        if not isinstance(target, CommitTarget):
            raise TypeError("target must be CommitTarget")
        if target.repository_id != self.repository_id:
            raise ValueError("CommitTarget belongs to another repository")
        if not callable(publish):
            raise TypeError("publish must be callable")
        snapshot = _CommitAuthoritySnapshot(
            target=target,
            journal=self._journal,
            publish_callback=publish,
            recover_callback=self._recover,
        )
        nonce = object()
        with self._authority_lock:
            self._authorities[nonce] = snapshot
        return CommitAuthority(
            _COMMIT_AUTHORITY_TOKEN,
            coordinator=self,
            nonce=nonce,
            target=target,
        )

    def _consume_authority(
        self,
        token: object,
        authority: CommitAuthority[CommitT],
    ) -> _CommitAuthoritySnapshot[CommitT]:
        if token is not _COMMIT_AUTHORITY_CONSUMER_TOKEN:
            raise PermissionError("commit authority can only be consumed by RunController")
        if authority._coordinator is not self:  # noqa: SLF001
            raise ValueError("commit authority belongs to another coordinator")
        with self._authority_lock:
            try:
                snapshot = self._authorities.pop(authority._nonce)  # noqa: SLF001
            except KeyError as exc:
                raise RuntimeError("commit authority was already consumed") from exc
        if snapshot.target != authority.target:
            raise RuntimeError("commit authority target snapshot mismatch")
        return snapshot


class MemoryCommitJournal:
    durable = False

    def __init__(self, repository_id: str) -> None:
        self.repository_id = _canonical(repository_id, "repository_id")
        self._lock = threading.Lock()
        self._intents: dict[str, CommitIntent] = {}
        self._committed: set[str] = set()
        self._aborted: set[str] = set()

    def begin(self, intent: CommitIntent) -> None:
        if not isinstance(intent, CommitIntent):
            raise TypeError("begin requires CommitIntent")
        if intent.target.repository_id != self.repository_id:
            raise ValueError("commit intent belongs to another repository")
        with self._lock:
            previous = self._intents.get(intent.commit_id)
            if previous is not None and previous != intent:
                raise ValueError("commit_id has conflicting intent")
            self._intents[intent.commit_id] = intent

    def mark_committed(self, commit_id: str) -> None:
        commit_id = _canonical(commit_id, "commit_id")
        with self._lock:
            if commit_id not in self._intents:
                raise KeyError(f"unknown commit intent {commit_id}")
            if commit_id in self._aborted:
                raise ValueError("aborted commit cannot become committed")
            self._committed.add(commit_id)

    def mark_aborted(self, commit_id: str) -> None:
        commit_id = _canonical(commit_id, "commit_id")
        with self._lock:
            if commit_id not in self._intents:
                raise KeyError(f"unknown commit intent {commit_id}")
            if commit_id in self._committed:
                raise ValueError("committed commit cannot become aborted")
            self._aborted.add(commit_id)

    def pending(self) -> tuple[CommitIntent, ...]:
        with self._lock:
            return tuple(
                intent
                for commit_id, intent in self._intents.items()
                if commit_id not in self._committed and commit_id not in self._aborted
            )


class PersistentCommitJournal:
    """Repository-owned intent log; unresolved entries are startup reconciliation gates."""

    durable = True

    def __init__(self, path: str | Path, repository_id: str) -> None:
        self.repository_id = _canonical(repository_id, "repository_id")
        self._journal = FramedJournal(path)
        self._journal.append(
            "repository",
            {"kind": "REPOSITORY", "repository_id": self.repository_id},
        )
        self._lock = threading.Lock()
        self._intents: dict[str, CommitIntent] = {}
        self._committed: set[str] = set()
        self._aborted: set[str] = set()
        self._replay()

    def begin(self, intent: CommitIntent) -> None:
        if not isinstance(intent, CommitIntent):
            raise TypeError("begin requires CommitIntent")
        if intent.target.repository_id != self.repository_id:
            raise ValueError("commit intent belongs to another repository")
        value = {
            "kind": "INTENT",
            "commit_id": intent.commit_id,
            "run_id": intent.run_id,
            "safety_bundle_id": intent.safety_bundle_id,
            "target": {
                "repository_id": intent.target.repository_id,
                "artifact_kind": intent.target.artifact_kind,
                "schema_version": intent.target.schema_version,
                "target_ref": intent.target.target_ref,
                "expected_manifest_digest": intent.target.expected_manifest_digest,
            },
            "created_at": intent.created_at,
        }
        with self._lock:
            self._journal.append_checked(
                f"intent:{intent.commit_id}",
                value,
                self._validate_records,
            )
            self._replay()

    def mark_committed(self, commit_id: str) -> None:
        commit_id = _canonical(commit_id, "commit_id")
        with self._lock:
            self._journal.append_checked(
                f"committed:{commit_id}",
                {
                    "kind": "COMMITTED",
                    "commit_id": commit_id,
                },
                self._validate_records,
            )
            self._replay()

    def mark_aborted(self, commit_id: str) -> None:
        commit_id = _canonical(commit_id, "commit_id")
        with self._lock:
            self._journal.append_checked(
                f"aborted:{commit_id}",
                {"kind": "ABORTED", "commit_id": commit_id},
                self._validate_records,
            )
            self._replay()

    def pending(self) -> tuple[CommitIntent, ...]:
        with self._lock:
            self._replay()
            return tuple(
                intent
                for commit_id, intent in self._intents.items()
                if commit_id not in self._committed and commit_id not in self._aborted
            )

    def _replay(self) -> None:
        intents, committed, aborted = self._state_from_records(self._journal.records())
        if any(
            intent.target.repository_id != self.repository_id
            for intent in intents.values()
        ):
            raise ValueError("commit journal contains an intent for another repository")
        self._intents = intents
        self._committed = committed
        self._aborted = aborted

    @classmethod
    def _validate_records(cls, records: tuple[tuple[str, object], ...]) -> None:
        cls._state_from_records(records)

    @staticmethod
    def _state_from_records(
        records: tuple[tuple[str, object], ...],
    ) -> tuple[dict[str, CommitIntent], set[str], set[str]]:
        intents: dict[str, CommitIntent] = {}
        committed: set[str] = set()
        aborted: set[str] = set()
        for _record_id, value in records:
            if not isinstance(value, dict):
                raise ValueError("commit journal record must be an object")
            kind = value.get("kind")
            if kind == "REPOSITORY":
                if set(value) != {"kind", "repository_id"}:
                    raise ValueError("invalid repository marker fields")
                _canonical(value["repository_id"], "repository_id")
            elif kind == "INTENT":
                if set(value) != {
                    "kind",
                    "commit_id",
                    "run_id",
                    "safety_bundle_id",
                    "target",
                    "created_at",
                }:
                    raise ValueError("invalid commit intent fields")
                target = value["target"]
                if not isinstance(target, dict) or set(target) != {
                    "repository_id",
                    "artifact_kind",
                    "schema_version",
                    "target_ref",
                    "expected_manifest_digest",
                }:
                    raise ValueError("invalid commit target fields")
                intent = CommitIntent(
                    commit_id=value["commit_id"],
                    run_id=value["run_id"],
                    safety_bundle_id=value["safety_bundle_id"],
                    target=CommitTarget(**target),
                    created_at=value["created_at"],
                )
                previous = intents.get(intent.commit_id)
                if previous is not None and previous != intent:
                    raise ValueError("commit journal contains conflicting intent")
                intents[intent.commit_id] = intent
            elif kind == "COMMITTED":
                if set(value) != {"kind", "commit_id"}:
                    raise ValueError("invalid committed marker fields")
                commit_id = _canonical(value["commit_id"], "commit_id")
                if commit_id not in intents:
                    raise ValueError("commit marker precedes its intent")
                if commit_id in aborted:
                    raise ValueError("commit journal resolves one intent both ways")
                committed.add(commit_id)
            elif kind == "ABORTED":
                if set(value) != {"kind", "commit_id"}:
                    raise ValueError("invalid aborted marker fields")
                commit_id = _canonical(value["commit_id"], "commit_id")
                if commit_id not in intents:
                    raise ValueError("abort marker precedes its intent")
                if commit_id in committed:
                    raise ValueError("commit journal resolves one intent both ways")
                aborted.add(commit_id)
            else:
                raise ValueError("unknown commit journal record kind")
        return intents, committed, aborted


def commit_now() -> float:
    return time.time()
