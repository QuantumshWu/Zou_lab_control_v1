"""Process-local authority for one final artifact manifest publication.

The canonical CAS manifest is the only durable commit record.  This module
therefore owns no intent log, startup recovery projection, commit identifier or
repository coordinator.  A repository stages immutable blobs, freezes the
canonical manifest bytes, and lends one single-use operation to RunController.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Generic, TypeVar

from zlc_storage import RepositoryRootLeaseBorrow, canonical_text


CommitT = TypeVar("CommitT")
_COMMIT_CONSUMER_TOKEN = object()


class PreparedArtifactCommit(Generic[CommitT]):
    """Single-use publication capability backed by one repository-root borrow.

    ``inspect`` returns ``True`` only after the exact expected manifest and its
    referenced content are readable and durable, ``False`` only when the exact
    manifest is confirmed absent, and ``None`` when storage is temporarily
    unreadable.  Corruption and domain-lineage mismatches are raised.
    """

    __slots__ = (
        "_run_id",
        "_result",
        "_manifest_payload",
        "_publish",
        "_inspect",
        "_borrow",
        "_lock",
        "_state",
        "_publish_attempted",
    )

    def __init__(
        self,
        *,
        run_id: str,
        result: CommitT,
        manifest_payload: bytes,
        publish: Callable[[bytes], None],
        inspect: Callable[[bytes], bool | None],
        repository_borrow: RepositoryRootLeaseBorrow,
    ) -> None:
        if not isinstance(manifest_payload, bytes):
            raise TypeError("manifest_payload must be canonical bytes")
        if not callable(publish) or not callable(inspect):
            raise TypeError("publish and inspect must be callable")
        if type(repository_borrow) is not RepositoryRootLeaseBorrow:
            raise TypeError("repository_borrow must be RepositoryRootLeaseBorrow")
        repository_borrow.require_active()
        object.__setattr__(self, "_run_id", canonical_text(run_id, "run_id"))
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_manifest_payload", manifest_payload)
        object.__setattr__(self, "_publish", publish)
        object.__setattr__(self, "_inspect", inspect)
        object.__setattr__(self, "_borrow", repository_borrow)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_state", "prepared")
        object.__setattr__(self, "_publish_attempted", False)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PreparedArtifactCommit is immutable")

    def __reduce__(self):
        raise TypeError("PreparedArtifactCommit is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("PreparedArtifactCommit is process-local")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def result(self) -> CommitT:
        """The exact typed reference named by the frozen manifest payload."""

        return self._result

    def abandon(self) -> bool:
        """Release an operation that RunController never consumed."""

        borrow = None
        with self._lock:
            if self._state != "prepared":
                return False
            object.__setattr__(self, "_state", "abandoned")
            borrow = self._borrow
        borrow.close()
        return True

    def _consume(self, token: object) -> None:
        if token is not _COMMIT_CONSUMER_TOKEN:
            raise PermissionError("only RunController may consume a final commit")
        with self._lock:
            if self._state != "prepared":
                raise RuntimeError("prepared artifact commit was already consumed")
            self._borrow.require_active()
            object.__setattr__(self, "_state", "consumed")

    def _publish_once(self, token: object) -> CommitT:
        if token is not _COMMIT_CONSUMER_TOKEN:
            raise PermissionError("only RunController may publish a final commit")
        with self._lock:
            if self._state != "consumed":
                raise RuntimeError("artifact commit is not owned by RunController")
            if self._publish_attempted:
                raise RuntimeError("artifact manifest publication may be attempted once")
            self._borrow.require_active()
            object.__setattr__(self, "_publish_attempted", True)
            payload = self._manifest_payload
            publish = self._publish
        publish(payload)
        return self._result

    def _inspect_exact_target(self, token: object) -> bool | None:
        if token is not _COMMIT_CONSUMER_TOKEN:
            raise PermissionError("only RunController may inspect a final commit")
        with self._lock:
            if self._state != "consumed" or not self._publish_attempted:
                raise RuntimeError("artifact inspection requires one publication attempt")
            self._borrow.require_active()
            payload = self._manifest_payload
            inspect = self._inspect
        resolution = inspect(payload)
        if resolution is not None and type(resolution) is not bool:
            raise TypeError("manifest inspection must return bool or None")
        return resolution

    def _finish(self, token: object) -> None:
        if token is not _COMMIT_CONSUMER_TOKEN:
            raise PermissionError("only RunController may finish a final commit")
        with self._lock:
            if self._state != "consumed":
                raise RuntimeError("artifact commit is not owned by RunController")
            object.__setattr__(self, "_state", "finished")
            borrow = self._borrow
        borrow.close()

    def __del__(self) -> None:
        try:
            self.abandon()
        except BaseException:
            pass


def _consume_prepared_artifact_commit(
    operation: PreparedArtifactCommit[CommitT],
) -> PreparedArtifactCommit[CommitT]:
    if not isinstance(operation, PreparedArtifactCommit):
        raise TypeError("operation must be PreparedArtifactCommit")
    operation._consume(_COMMIT_CONSUMER_TOKEN)
    return operation


def _publish_prepared_artifact_commit(
    operation: PreparedArtifactCommit[CommitT],
) -> CommitT:
    return operation._publish_once(_COMMIT_CONSUMER_TOKEN)


def _inspect_prepared_artifact_commit(
    operation: PreparedArtifactCommit[object],
) -> bool | None:
    return operation._inspect_exact_target(_COMMIT_CONSUMER_TOKEN)


def _finish_prepared_artifact_commit(
    operation: PreparedArtifactCommit[object],
) -> None:
    operation._finish(_COMMIT_CONSUMER_TOKEN)


__all__ = ["PreparedArtifactCommit"]
