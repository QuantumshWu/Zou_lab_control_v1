"""Durable final-artifact commit intents and restart reconciliation state."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Generic, Protocol, TypeVar

from zlc_storage.framed_journal import FramedJournal
from zlc_storage import RepositoryRootLease, RepositoryRootLeaseBorrow


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


class CommitKind(str, Enum):
    CHECKPOINT = "CHECKPOINT"
    FINAL = "FINAL"


@dataclass(frozen=True)
class CommitSubject:
    """Run identity bound into a repository-minted commit authority."""

    run_id: str
    safety_bundle_id: str | None

    def __post_init__(self) -> None:
        _canonical(self.run_id, "commit subject run_id")
        if self.safety_bundle_id is not None:
            _canonical(self.safety_bundle_id, "commit subject safety_bundle_id")


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
    kind: CommitKind
    commit_id: str
    run_id: str
    safety_bundle_id: str | None
    target: CommitTarget
    created_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CommitKind):
            raise TypeError("commit intent kind must be CommitKind")
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

    def pending(self) -> tuple[CommitIntent, ...]: ...

    def committed(self) -> tuple[CommitIntent, ...]: ...


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
_COMMIT_AUTHORITY_ABANDON_TOKEN = object()
_JOURNAL_MUTATION_TOKEN = object()


class _JournalMutationAuthority:
    """Opaque writer retained only inside coordinator/commit snapshots."""

    __slots__ = ("_journal",)

    def __init__(self, token: object, journal: CommitJournal) -> None:
        if token is not _JOURNAL_MUTATION_TOKEN:
            raise PermissionError(
                "journal mutation authority can only be minted by commit runtime"
            )
        for name in ("_begin", "_mark_committed", "_mark_aborted"):
            if not callable(getattr(journal, name, None)):
                raise TypeError("commit journal lacks internal mutation protocol")
        object.__setattr__(self, "_journal", journal)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("journal mutation authority is immutable")

    @property
    def repository_id(self) -> str:
        return self._journal.repository_id

    def begin(self, intent: CommitIntent) -> None:
        self._journal._begin(_JOURNAL_MUTATION_TOKEN, intent)

    def mark_committed(self, commit_id: str) -> None:
        self._journal._mark_committed(_JOURNAL_MUTATION_TOKEN, commit_id)

    def mark_aborted(self, commit_id: str) -> None:
        self._journal._mark_aborted(_JOURNAL_MUTATION_TOKEN, commit_id)

    def pending(self) -> tuple[CommitIntent, ...]:
        return self._journal.pending()

    def committed(self) -> tuple[CommitIntent, ...]:
        return self._journal.committed()


def _journal_mutation_authority(journal: CommitJournal) -> _JournalMutationAuthority:
    return _JournalMutationAuthority(_JOURNAL_MUTATION_TOKEN, journal)


@dataclass(frozen=True)
class _CommitPreparation:
    kind: CommitKind
    commit_id: str
    subject: CommitSubject
    target: CommitTarget

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CommitKind):
            raise TypeError("commit preparation kind must be CommitKind")
        _canonical(self.commit_id, "commit preparation commit_id")
        if not isinstance(self.subject, CommitSubject):
            raise TypeError("commit preparation subject must be CommitSubject")
        if not isinstance(self.target, CommitTarget):
            raise TypeError("commit preparation target must be CommitTarget")


class CommitAuthority(Generic[CommitT]):
    """Side-effect-free handle to coordinator-owned commit capabilities."""

    __slots__ = ("_coordinator", "_nonce", "_preparation")

    def __init__(
        self,
        token: object,
        *,
        coordinator: "RepositoryCommitCoordinator[CommitT]",
        nonce: object,
        preparation: _CommitPreparation,
    ) -> None:
        if token is not _COMMIT_AUTHORITY_TOKEN:
            raise PermissionError(
                "CommitAuthority must be minted by RepositoryCommitCoordinator"
            )
        if not isinstance(preparation, _CommitPreparation):
            raise TypeError("CommitAuthority preparation is invalid")
        if not isinstance(coordinator, RepositoryCommitCoordinator):
            raise TypeError("CommitAuthority coordinator is invalid")
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_nonce", nonce)
        object.__setattr__(self, "_preparation", preparation)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CommitAuthority is immutable")

    @property
    def target(self) -> CommitTarget:
        return self._preparation.target

    @property
    def kind(self) -> CommitKind:
        return self._preparation.kind

    @property
    def commit_id(self) -> str:
        return self._preparation.commit_id

    @property
    def run_id(self) -> str:
        return self._preparation.subject.run_id

    @property
    def safety_bundle_id(self) -> str | None:
        return self._preparation.subject.safety_bundle_id

    def abandon(self) -> None:
        """Explicitly abandon this authority if RunController has not consumed it."""

        self._coordinator._abandon_authority(  # noqa: SLF001
            _COMMIT_AUTHORITY_ABANDON_TOKEN,
            self,
        )

    def __del__(self) -> None:
        try:
            self.abandon()
        except BaseException:
            pass


@dataclass(frozen=True)
class _CommitAuthoritySnapshot(Generic[CommitT]):
    preparation: _CommitPreparation
    journal: _JournalMutationAuthority
    publish_callback: Callable[[], PublishedManifest[CommitT]]
    recover_callback: Callable[[CommitIntent], CommitRecovery[CommitT]]
    lease_borrow: RepositoryRootLeaseBorrow | None

    @property
    def target(self) -> CommitTarget:
        return self.preparation.target

    def _validate_intent(self, intent: CommitIntent) -> None:
        if not isinstance(intent, CommitIntent):
            raise TypeError("commit authority requires CommitIntent")
        preparation = self.preparation
        if (
            intent.kind is not preparation.kind
            or intent.commit_id != preparation.commit_id
            or intent.run_id != preparation.subject.run_id
            or intent.safety_bundle_id != preparation.subject.safety_bundle_id
            or intent.target != preparation.target
        ):
            raise ValueError("CommitIntent differs from authority preparation")

    def publish_validated(self, intent: CommitIntent) -> CommitT:
        self.require_lifetime()
        self._validate_intent(intent)
        return _validate_published_manifest(self.publish_callback(), self.target)

    def recover_validated(self, intent: CommitIntent) -> CommitRecovery[CommitT]:
        self.require_lifetime()
        self._validate_intent(intent)
        return _validate_commit_recovery(self.recover_callback(intent), self.target)

    def require_lifetime(self) -> None:
        if self.lease_borrow is not None:
            self.lease_borrow.require_active()

    def release_lifetime(self) -> None:
        if self.lease_borrow is not None:
            self.lease_borrow.close()


_COMMIT_AUTHORITY_CONSUMER_TOKEN = object()


def _consume_commit_authority(
    authority: CommitAuthority[CommitT],
) -> _CommitAuthoritySnapshot[CommitT]:
    if not isinstance(authority, CommitAuthority):
        raise TypeError("commit authority is invalid")
    return authority._coordinator._consume_authority(  # noqa: SLF001
        _COMMIT_AUTHORITY_CONSUMER_TOKEN,
        authority,
    )


def _discard_commit_authority(authority: CommitAuthority[object]) -> None:
    """Abandon an operation rejected before an intent so its capability cannot leak."""

    if not isinstance(authority, CommitAuthority):
        raise TypeError("commit authority is invalid")
    authority.abandon()


@dataclass(frozen=True)
class FinalCommit(Generic[CommitT]):
    authority: CommitAuthority[CommitT]

    def __post_init__(self) -> None:
        if not isinstance(self.authority, CommitAuthority):
            raise TypeError("FinalCommit.authority must be CommitAuthority")
        if self.authority.kind is not CommitKind.FINAL:
            raise TypeError("FinalCommit requires FINAL commit authority")

    @property
    def commit_id(self) -> str:
        return self.authority.commit_id

    @property
    def run_id(self) -> str:
        return self.authority.run_id

    @property
    def safety_bundle_id(self) -> str | None:
        return self.authority.safety_bundle_id

    @property
    def target(self) -> CommitTarget:
        return self.authority.target

    def abandon(self) -> None:
        self.authority.abandon()


@dataclass(frozen=True)
class CheckpointCommit(Generic[CommitT]):
    """One typed intermediate artifact commit; never a final-commit policy flag."""

    authority: CommitAuthority[CommitT]

    def __post_init__(self) -> None:
        if not isinstance(self.authority, CommitAuthority):
            raise TypeError("CheckpointCommit.authority must be CommitAuthority")
        if self.authority.kind is not CommitKind.CHECKPOINT:
            raise TypeError("CheckpointCommit requires CHECKPOINT commit authority")

    @property
    def commit_id(self) -> str:
        return self.authority.commit_id

    @property
    def run_id(self) -> str:
        return self.authority.run_id

    @property
    def safety_bundle_id(self) -> str | None:
        return self.authority.safety_bundle_id

    @property
    def target(self) -> CommitTarget:
        return self.authority.target

    def abandon(self) -> None:
        self.authority.abandon()


def reconcile_pending_commits(
    journal: _JournalMutationAuthority,
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

    __slots__ = (
        "repository_id",
        "_journal",
        "_journal_writer",
        "_root_lease",
        "_recover",
        "_authority_lock",
        "_authorities",
        "_startup_reconciliations",
        "_sealed",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError(
            "RepositoryCommitCoordinator is final and cannot be subclassed"
        )

    def __init__(
        self,
        journal: CommitJournal,
        recover: Callable[[CommitIntent], CommitRecovery[CommitT]],
        *,
        allow_ephemeral: bool = False,
        root_lease: RepositoryRootLease | None = None,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
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
        if durable:
            if type(root_lease) is not RepositoryRootLease:
                raise ValueError(
                    "durable commit coordinator requires RepositoryRootLease"
                )
            root_lease.require_active()
            repository_root = getattr(journal, "repository_root", None)
            if not isinstance(repository_root, Path):
                raise TypeError(
                    "durable commit journal must expose its resolved repository_root"
                )
            if repository_root != root_lease.root:
                raise ValueError(
                    "commit journal and repository root lease bind different roots"
                )
        elif root_lease is not None:
            raise ValueError("ephemeral commit coordinator cannot own a root lease")
        object.__setattr__(self, "repository_id", repository_id)
        object.__setattr__(self, "_journal", journal)
        journal_writer = _journal_mutation_authority(journal)
        object.__setattr__(self, "_journal_writer", journal_writer)
        object.__setattr__(self, "_root_lease", root_lease)
        object.__setattr__(self, "_recover", recover)
        object.__setattr__(self, "_authority_lock", threading.Lock())
        object.__setattr__(self, "_authorities", {})
        object.__setattr__(
            self,
            "_startup_reconciliations",
            reconcile_pending_commits(
                journal_writer,
                repository_id,
                recover,
            ),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("RepositoryCommitCoordinator wiring is immutable")
        object.__setattr__(self, _name, _value)

    @property
    def startup_reconciliations(self) -> tuple[ReconciledCommit[CommitT], ...]:
        if self._root_lease is not None:
            self._root_lease.require_active()
        return self._startup_reconciliations

    def committed_intents(self) -> tuple[CommitIntent, ...]:
        """Sealed repository-owner view used to mint downstream admission proof."""

        if self._root_lease is not None:
            self._root_lease.require_active()
        return self._journal_writer.committed()

    def prepare(
        self,
        kind: CommitKind,
        commit_id: str,
        subject: CommitSubject,
        target: CommitTarget,
        publish: Callable[[], PublishedManifest[CommitT]],
    ) -> CommitAuthority[CommitT]:
        preparation = _CommitPreparation(kind, commit_id, subject, target)
        if not isinstance(target, CommitTarget):
            raise TypeError("target must be CommitTarget")
        if target.repository_id != self.repository_id:
            raise ValueError("CommitTarget belongs to another repository")
        if not callable(publish):
            raise TypeError("publish must be callable")
        lease_borrow = (
            None if self._root_lease is None else self._root_lease.borrow()
        )
        try:
            snapshot = _CommitAuthoritySnapshot(
                preparation=preparation,
                journal=self._journal_writer,
                publish_callback=publish,
                recover_callback=self._recover,
                lease_borrow=lease_borrow,
            )
            nonce = object()
            with self._authority_lock:
                self._authorities[nonce] = snapshot
            try:
                return CommitAuthority(
                    _COMMIT_AUTHORITY_TOKEN,
                    coordinator=self,
                    nonce=nonce,
                    preparation=preparation,
                )
            except BaseException:
                with self._authority_lock:
                    self._authorities.pop(nonce, None)
                raise
        except BaseException:
            if lease_borrow is not None:
                lease_borrow.close()
            raise

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
        if snapshot.preparation != authority._preparation:  # noqa: SLF001
            snapshot.release_lifetime()
            raise RuntimeError("commit authority preparation snapshot mismatch")
        return snapshot

    def _abandon_authority(
        self,
        token: object,
        authority: CommitAuthority[object],
    ) -> None:
        if token is not _COMMIT_AUTHORITY_ABANDON_TOKEN:
            raise PermissionError("commit authority abandonment is private")
        if not isinstance(authority, CommitAuthority):
            raise TypeError("commit authority is invalid")
        if authority._coordinator is not self:  # noqa: SLF001
            raise ValueError("commit authority belongs to another coordinator")
        with self._authority_lock:
            snapshot = self._authorities.pop(authority._nonce, None)  # noqa: SLF001
        if snapshot is not None:
            snapshot.release_lifetime()


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

    def _begin(self, token: object, intent: CommitIntent) -> None:
        if token is not _JOURNAL_MUTATION_TOKEN:
            raise PermissionError("journal mutation requires coordinator authority")
        self.begin(intent)

    def mark_committed(self, commit_id: str) -> None:
        commit_id = _canonical(commit_id, "commit_id")
        with self._lock:
            if commit_id not in self._intents:
                raise KeyError(f"unknown commit intent {commit_id}")
            if commit_id in self._aborted:
                raise ValueError("aborted commit cannot become committed")
            self._committed.add(commit_id)

    def _mark_committed(self, token: object, commit_id: str) -> None:
        if token is not _JOURNAL_MUTATION_TOKEN:
            raise PermissionError("journal mutation requires coordinator authority")
        self.mark_committed(commit_id)

    def mark_aborted(self, commit_id: str) -> None:
        commit_id = _canonical(commit_id, "commit_id")
        with self._lock:
            if commit_id not in self._intents:
                raise KeyError(f"unknown commit intent {commit_id}")
            if commit_id in self._committed:
                raise ValueError("committed commit cannot become aborted")
            self._aborted.add(commit_id)

    def _mark_aborted(self, token: object, commit_id: str) -> None:
        if token is not _JOURNAL_MUTATION_TOKEN:
            raise PermissionError("journal mutation requires coordinator authority")
        self.mark_aborted(commit_id)

    def pending(self) -> tuple[CommitIntent, ...]:
        with self._lock:
            return tuple(
                intent
                for commit_id, intent in self._intents.items()
                if commit_id not in self._committed and commit_id not in self._aborted
            )

    def committed(self) -> tuple[CommitIntent, ...]:
        """Return an immutable snapshot of successfully committed intents."""

        with self._lock:
            return tuple(
                sorted(
                    (
                        intent
                        for commit_id, intent in self._intents.items()
                        if commit_id in self._committed
                    ),
                    key=lambda intent: intent.commit_id,
                )
            )


class PersistentCommitJournal:
    """Repository-owned intent log; unresolved entries are startup reconciliation gates."""

    durable = True

    __slots__ = (
        "repository_id",
        "repository_root",
        "_journal",
        "_lock",
        "_intents",
        "_committed",
        "_aborted",
        "_sealed",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("PersistentCommitJournal is final and cannot be subclassed")

    def __init__(self, path: str | Path, repository_id: str) -> None:
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(
            self,
            "repository_id",
            _canonical(repository_id, "repository_id"),
        )
        journal = FramedJournal(path)
        object.__setattr__(self, "_journal", journal)
        object.__setattr__(self, "repository_root", journal.path.parent)
        self._journal.append(
            "repository",
            {"kind": "REPOSITORY", "repository_id": self.repository_id},
        )
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_intents", {})
        object.__setattr__(self, "_committed", set())
        object.__setattr__(self, "_aborted", set())
        self._replay()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PersistentCommitJournal wiring is immutable")
        object.__setattr__(self, _name, _value)

    def _begin(self, token: object, intent: CommitIntent) -> None:
        if token is not _JOURNAL_MUTATION_TOKEN:
            raise PermissionError("journal mutation requires coordinator authority")
        if not isinstance(intent, CommitIntent):
            raise TypeError("begin requires CommitIntent")
        if intent.target.repository_id != self.repository_id:
            raise ValueError("commit intent belongs to another repository")
        value = {
            "kind": "INTENT",
            "operation_kind": intent.kind.value,
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

    def _mark_committed(self, token: object, commit_id: str) -> None:
        if token is not _JOURNAL_MUTATION_TOKEN:
            raise PermissionError("journal mutation requires coordinator authority")
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

    def _mark_aborted(self, token: object, commit_id: str) -> None:
        if token is not _JOURNAL_MUTATION_TOKEN:
            raise PermissionError("journal mutation requires coordinator authority")
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

    def committed(self) -> tuple[CommitIntent, ...]:
        """Replay and return an immutable snapshot of committed intent records."""

        with self._lock:
            self._replay()
            return tuple(
                sorted(
                    (
                        intent
                        for commit_id, intent in self._intents.items()
                        if commit_id in self._committed
                    ),
                    key=lambda intent: intent.commit_id,
                )
            )

    def _replay(self) -> None:
        intents, committed, aborted = self._state_from_records(
            self._journal.records(),
            expected_repository_id=self.repository_id,
        )
        if any(
            intent.target.repository_id != self.repository_id
            for intent in intents.values()
        ):
            raise ValueError("commit journal contains an intent for another repository")
        object.__setattr__(self, "_intents", intents)
        object.__setattr__(self, "_committed", committed)
        object.__setattr__(self, "_aborted", aborted)

    def _validate_records(self, records: tuple[tuple[str, object], ...]) -> None:
        self._state_from_records(
            records,
            expected_repository_id=self.repository_id,
        )

    @staticmethod
    def _state_from_records(
        records: tuple[tuple[str, object], ...],
        *,
        expected_repository_id: str,
    ) -> tuple[dict[str, CommitIntent], set[str], set[str]]:
        expected_repository_id = _canonical(
            expected_repository_id,
            "expected_repository_id",
        )
        intents: dict[str, CommitIntent] = {}
        committed: set[str] = set()
        aborted: set[str] = set()
        repository_markers = 0
        for index, (record_id, value) in enumerate(records):
            if not isinstance(value, dict):
                raise ValueError("commit journal record must be an object")
            kind = value.get("kind")
            if kind == "REPOSITORY":
                repository_markers += 1
                if index != 0 or repository_markers != 1 or record_id != "repository":
                    raise ValueError(
                        "commit journal repository marker must be unique and first"
                    )
                if set(value) != {"kind", "repository_id"}:
                    raise ValueError("invalid repository marker fields")
                if (
                    _canonical(value["repository_id"], "repository_id")
                    != expected_repository_id
                ):
                    raise ValueError(
                        "commit journal repository marker belongs to another repository"
                    )
            elif kind == "INTENT":
                if set(value) != {
                    "kind",
                    "operation_kind",
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
                    kind=CommitKind(value["operation_kind"]),
                    commit_id=value["commit_id"],
                    run_id=value["run_id"],
                    safety_bundle_id=value["safety_bundle_id"],
                    target=CommitTarget(**target),
                    created_at=value["created_at"],
                )
                if record_id != f"intent:{intent.commit_id}":
                    raise ValueError("commit intent record_id differs from payload")
                previous = intents.get(intent.commit_id)
                if previous is not None and previous != intent:
                    raise ValueError("commit journal contains conflicting intent")
                intents[intent.commit_id] = intent
            elif kind == "COMMITTED":
                if set(value) != {"kind", "commit_id"}:
                    raise ValueError("invalid committed marker fields")
                commit_id = _canonical(value["commit_id"], "commit_id")
                if record_id != f"committed:{commit_id}":
                    raise ValueError("committed record_id differs from payload")
                if commit_id not in intents:
                    raise ValueError("commit marker precedes its intent")
                if commit_id in aborted:
                    raise ValueError("commit journal resolves one intent both ways")
                committed.add(commit_id)
            elif kind == "ABORTED":
                if set(value) != {"kind", "commit_id"}:
                    raise ValueError("invalid aborted marker fields")
                commit_id = _canonical(value["commit_id"], "commit_id")
                if record_id != f"aborted:{commit_id}":
                    raise ValueError("aborted record_id differs from payload")
                if commit_id not in intents:
                    raise ValueError("abort marker precedes its intent")
                if commit_id in committed:
                    raise ValueError("commit journal resolves one intent both ways")
                aborted.add(commit_id)
            else:
                raise ValueError("unknown commit journal record kind")
        if repository_markers != 1:
            raise ValueError("commit journal requires exactly one repository marker")
        return intents, committed, aborted


def commit_now() -> float:
    return time.time()


# Durable journal mutation and coordinator construction are repository-owner
# implementation details.  They intentionally remain importable by the domain
# repository modules that own them, but are excluded from both this module's
# declared public API and ``zlc_neutral_atom.runtime``'s package facade.
__all__ = [
    "CheckpointCommit",
    "CommitAuthority",
    "CommitIntent",
    "CommitJournal",
    "CommitKind",
    "CommitRecovery",
    "CommitSubject",
    "CommitTarget",
    "FinalCommit",
    "MemoryCommitJournal",
    "PublishVisibilityUnknown",
    "PublishedManifest",
    "ReconciledCommit",
]
