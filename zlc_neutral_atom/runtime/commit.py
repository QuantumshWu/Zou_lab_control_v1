"""Durable final-artifact commit intents and restart reconciliation state."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

from zlc_storage.framed_journal import FramedJournal
from zlc_storage import (
    ContentStoreAuthority,
    RepositoryRootLease,
    RepositoryRootLeaseBorrow,
    StoredManifest,
    canonical_text as _canonical,
    sha256_text as _sha256,
)


CommitT = TypeVar("CommitT")


@dataclass(frozen=True)
class CommitTarget:
    repository_id: str
    artifact_kind: str
    artifact_format: str
    target_ref: str
    expected_manifest_digest: str

    def __post_init__(self) -> None:
        _canonical(self.repository_id, "repository_id")
        _canonical(self.artifact_kind, "artifact_kind")
        _canonical(self.artifact_format, "artifact_format")
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
class PublishedManifest(Generic[CommitT]):
    target_ref: str
    manifest_digest: str
    result: CommitT

    def __post_init__(self) -> None:
        _canonical(self.target_ref, "published target_ref")
        _sha256(self.manifest_digest, "published manifest_digest")


class PublishVisibilityUnknown(RuntimeError):
    """The repository cannot yet determine whether its manifest became visible."""


def publish_manifest_with_visibility_reconciliation(
    authority: ContentStoreAuthority,
    namespace: str,
    payload: bytes,
    *,
    expected_digest: str,
    max_bytes: int,
) -> StoredManifest:
    """Publish once and classify an ordinary failure from the visible target.

    An explicit :class:`PublishVisibilityUnknown` already carries the storage
    owner's verdict and is propagated unchanged.  For every other publication
    error, only a verified absent target is a known failure.  A visible expected
    target, an unreadable target, or unexpected visible bytes all leave commit
    visibility unknown and must be reconciled by the commit coordinator.
    """

    try:
        return authority.publish_manifest(
            namespace,
            payload,
            expected_digest=expected_digest,
        )
    except PublishVisibilityUnknown:
        raise
    except BaseException as publish_error:
        try:
            visible = authority.read_manifest(
                namespace,
                expected_digest,
                max_bytes=max_bytes,
            )
        except FileNotFoundError:
            raise publish_error
        except BaseException as visibility_error:
            raise PublishVisibilityUnknown(
                "manifest visibility could not be verified"
            ) from visibility_error
        if visible != payload:
            raise PublishVisibilityUnknown(
                "manifest target is visible with unexpected bytes"
            ) from publish_error
        raise PublishVisibilityUnknown(
            "manifest became visible before publication acknowledgement completed"
        ) from publish_error


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


def _validate_recovered_manifest(
    recovery: PublishedManifest[CommitT] | None,
    target: CommitTarget,
) -> PublishedManifest[CommitT] | None:
    if recovery is not None:
        _validate_published_manifest(recovery, target)
    return recovery


_COMMIT_AUTHORITY_TOKEN = object()
_COMMIT_AUTHORITY_ABANDON_TOKEN = object()
_JOURNAL_MUTATION_TOKEN = object()


@dataclass(frozen=True)
class _CommitPreparation:
    commit_id: str
    run_id: str
    safety_bundle_id: str | None
    target: CommitTarget


class FinalCommit(Generic[CommitT]):
    """Single-use final-artifact authority minted by a repository coordinator."""

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
                "FinalCommit must be minted by RepositoryCommitCoordinator"
            )
        if not isinstance(preparation, _CommitPreparation):
            raise TypeError("FinalCommit preparation is invalid")
        if not isinstance(coordinator, RepositoryCommitCoordinator):
            raise TypeError("FinalCommit coordinator is invalid")
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_nonce", nonce)
        object.__setattr__(self, "_preparation", preparation)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("FinalCommit is immutable")

    @property
    def target(self) -> CommitTarget:
        return self._preparation.target

    @property
    def commit_id(self) -> str:
        return self._preparation.commit_id

    @property
    def run_id(self) -> str:
        return self._preparation.run_id

    @property
    def safety_bundle_id(self) -> str | None:
        return self._preparation.safety_bundle_id

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
    journal: "PersistentCommitJournal"
    publish_callback: Callable[[], PublishedManifest[CommitT]]
    recover_callback: Callable[[CommitIntent], PublishedManifest[CommitT] | None]
    lease_borrow: RepositoryRootLeaseBorrow

    @property
    def target(self) -> CommitTarget:
        return self.preparation.target

    def _validate_intent(self, intent: CommitIntent) -> None:
        if not isinstance(intent, CommitIntent):
            raise TypeError("commit authority requires CommitIntent")
        preparation = self.preparation
        if (
            intent.commit_id != preparation.commit_id
            or intent.run_id != preparation.run_id
            or intent.safety_bundle_id != preparation.safety_bundle_id
            or intent.target != preparation.target
        ):
            raise ValueError("CommitIntent differs from authority preparation")

    def begin_intent(self, intent: CommitIntent) -> None:
        self.journal._begin(_JOURNAL_MUTATION_TOKEN, intent)

    def pending_intents(self) -> tuple[CommitIntent, ...]:
        return self.journal.pending()

    def mark_committed(self, commit_id: str) -> None:
        self.journal._mark_committed(_JOURNAL_MUTATION_TOKEN, commit_id)

    def mark_aborted(self, commit_id: str) -> None:
        self.journal._mark_aborted(_JOURNAL_MUTATION_TOKEN, commit_id)

    def publish_validated(self, intent: CommitIntent) -> CommitT:
        self.require_lifetime()
        self._validate_intent(intent)
        return _validate_published_manifest(self.publish_callback(), self.target)

    def recover_validated(
        self,
        intent: CommitIntent,
    ) -> PublishedManifest[CommitT] | None:
        self.require_lifetime()
        self._validate_intent(intent)
        return _validate_recovered_manifest(self.recover_callback(intent), self.target)

    def require_lifetime(self) -> None:
        self.lease_borrow.require_active()

    def release_lifetime(self) -> None:
        self.lease_borrow.close()


_COMMIT_AUTHORITY_CONSUMER_TOKEN = object()


def _consume_commit_authority(
    authority: FinalCommit[CommitT],
) -> _CommitAuthoritySnapshot[CommitT]:
    if not isinstance(authority, FinalCommit):
        raise TypeError("commit authority is invalid")
    return authority._coordinator._consume_authority(  # noqa: SLF001
        _COMMIT_AUTHORITY_CONSUMER_TOKEN,
        authority,
    )


def _discard_commit_authority(authority: FinalCommit[object]) -> None:
    """Abandon an operation rejected before an intent so its capability cannot leak."""

    if not isinstance(authority, FinalCommit):
        raise TypeError("commit authority is invalid")
    authority.abandon()


def _reconcile_pending_commits(
    journal: "PersistentCommitJournal",
    repository_id: str,
    recover: Callable[[CommitIntent], PublishedManifest[CommitT] | None],
) -> None:
    """Resolve durable intents by inspection only; this function never publishes."""

    if not callable(recover):
        raise TypeError("recover must be callable")
    repository_id = _canonical(repository_id, "repository_id")
    if journal.repository_id != repository_id:
        raise ValueError("startup reconciler does not own this commit journal")
    for intent in journal.pending():
        if intent.target.repository_id != repository_id:
            raise ValueError("pending commit belongs to another repository")
        resolution = _validate_recovered_manifest(recover(intent), intent.target)
        if resolution is not None:
            journal._mark_committed(_JOURNAL_MUTATION_TOKEN, intent.commit_id)
        else:
            journal._mark_aborted(_JOURNAL_MUTATION_TOKEN, intent.commit_id)


class RepositoryCommitCoordinator(Generic[CommitT]):
    """Startup gate and sole factory for one repository's commit authority."""

    __slots__ = (
        "repository_id",
        "_journal",
        "_root_lease",
        "_recover",
        "_authority_lock",
        "_authorities",
        "_sealed",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError(
            "RepositoryCommitCoordinator is final and cannot be subclassed"
        )

    def __init__(
        self,
        journal: "PersistentCommitJournal",
        recover: Callable[[CommitIntent], PublishedManifest[CommitT] | None],
        *,
        root_lease: RepositoryRootLease,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if type(journal) is not PersistentCommitJournal:
            raise TypeError("journal must be PersistentCommitJournal")
        repository_id = journal.repository_id
        if not callable(recover):
            raise TypeError("recover must be callable")
        if type(root_lease) is not RepositoryRootLease:
            raise TypeError("commit coordinator requires RepositoryRootLease")
        root_lease.require_active()
        if journal.repository_root != root_lease.root:
            raise ValueError(
                "commit journal and repository root lease bind different roots"
            )
        object.__setattr__(self, "repository_id", repository_id)
        object.__setattr__(self, "_journal", journal)
        object.__setattr__(self, "_root_lease", root_lease)
        object.__setattr__(self, "_recover", recover)
        object.__setattr__(self, "_authority_lock", threading.Lock())
        object.__setattr__(self, "_authorities", {})
        _reconcile_pending_commits(journal, repository_id, recover)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("RepositoryCommitCoordinator wiring is immutable")
        object.__setattr__(self, _name, _value)

    def committed_intents(self) -> tuple[CommitIntent, ...]:
        """Sealed repository-owner view used to mint downstream admission proof."""

        self._root_lease.require_active()
        return self._journal.committed()

    def prepare(
        self,
        commit_id: str,
        run_id: str,
        safety_bundle_id: str | None,
        target: CommitTarget,
        publish: Callable[[], PublishedManifest[CommitT]],
    ) -> FinalCommit[CommitT]:
        preparation = _CommitPreparation(
            _canonical(commit_id, "commit_id"),
            _canonical(run_id, "run_id"),
            (
                None
                if safety_bundle_id is None
                else _canonical(safety_bundle_id, "safety_bundle_id")
            ),
            target,
        )
        if not isinstance(target, CommitTarget):
            raise TypeError("target must be CommitTarget")
        if target.repository_id != self.repository_id:
            raise ValueError("CommitTarget belongs to another repository")
        if not callable(publish):
            raise TypeError("publish must be callable")
        lease_borrow = self._root_lease.borrow()
        try:
            snapshot = _CommitAuthoritySnapshot(
                preparation=preparation,
                journal=self._journal,
                publish_callback=publish,
                recover_callback=self._recover,
                lease_borrow=lease_borrow,
            )
            nonce = object()
            with self._authority_lock:
                self._authorities[nonce] = snapshot
            try:
                return FinalCommit(
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
            lease_borrow.close()
            raise

    def _consume_authority(
        self,
        token: object,
        authority: FinalCommit[CommitT],
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
        authority: FinalCommit[object],
    ) -> None:
        if token is not _COMMIT_AUTHORITY_ABANDON_TOKEN:
            raise PermissionError("commit authority abandonment is private")
        if not isinstance(authority, FinalCommit):
            raise TypeError("commit authority is invalid")
        if authority._coordinator is not self:  # noqa: SLF001
            raise ValueError("commit authority belongs to another coordinator")
        with self._authority_lock:
            snapshot = self._authorities.pop(authority._nonce, None)  # noqa: SLF001
        if snapshot is not None:
            snapshot.release_lifetime()


class PersistentCommitJournal:
    """Repository-owned intent log; unresolved entries are startup reconciliation gates."""

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
            "commit_id": intent.commit_id,
            "run_id": intent.run_id,
            "safety_bundle_id": intent.safety_bundle_id,
            "target": {
                "repository_id": intent.target.repository_id,
                "artifact_kind": intent.target.artifact_kind,
                "artifact_format": intent.target.artifact_format,
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
                    "artifact_format",
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


# Durable journal mutation and coordinator construction are repository-owner
# implementation details.  They intentionally remain importable by the domain
# repository modules that own them, but are excluded from both this module's
# declared public API and ``zlc_neutral_atom.runtime``'s package facade.
__all__ = [
    "CommitIntent",
    "CommitTarget",
    "FinalCommit",
    "PublishVisibilityUnknown",
    "PublishedManifest",
]
