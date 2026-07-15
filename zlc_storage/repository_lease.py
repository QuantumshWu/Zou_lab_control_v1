"""Process-lifetime exclusive ownership of one durable repository root."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

from .durability import durable_mkdir
from .file_lock import (
    FileLockBusy,
    acquire_file_lock,
    open_durable_lock_file,
    release_file_lock,
)


_LEASE_FILENAME = ".zlc-repository-root.lock"
_PROCESS_LEASE_LOCK = threading.Lock()
_PROCESS_LEASES: set[str] = set()
_BORROW_TOKEN = object()


class RepositoryRootBusy(RuntimeError):
    """Another live repository owner already holds this root."""


def _root_key(root: Path) -> str:
    return os.path.normcase(str(root))


class RepositoryRootLease:
    """Exclusive OS + in-process lease held for a repository owner's lifetime."""

    __slots__ = (
        "_root",
        "_creator_pid",
        "_key",
        "_stream",
        "_owner_closed",
        "_borrow_count",
        "_state_lock",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("RepositoryRootLease is final and cannot be subclassed")

    def __init__(self, root: str | os.PathLike[str]) -> None:
        resolved = Path(root).expanduser().resolve()
        durable_mkdir(resolved)
        key = _root_key(resolved)
        with _PROCESS_LEASE_LOCK:
            if key in _PROCESS_LEASES:
                raise RepositoryRootBusy(
                    f"repository root already has a live owner: {resolved}"
                )
            _PROCESS_LEASES.add(key)

        stream = None
        try:
            lock_path = resolved / _LEASE_FILENAME
            stream = open_durable_lock_file(lock_path)
            try:
                acquire_file_lock(stream, blocking=False)
            except FileLockBusy as exc:
                raise RepositoryRootBusy(
                    "repository root is owned by another live process"
                ) from exc
        except BaseException:
            if stream is not None:
                stream.close()
            with _PROCESS_LEASE_LOCK:
                _PROCESS_LEASES.discard(key)
            raise

        object.__setattr__(self, "_root", resolved)
        object.__setattr__(self, "_creator_pid", os.getpid())
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_owner_closed", False)
        object.__setattr__(self, "_borrow_count", 0)
        object.__setattr__(self, "_state_lock", threading.Lock())

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("RepositoryRootLease is immutable")

    def __reduce__(self):
        raise TypeError("RepositoryRootLease is process-local and cannot be serialized")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("RepositoryRootLease is process-local and cannot be serialized")

    @property
    def root(self) -> Path:
        self.require_active()
        return self._root

    @property
    def active(self) -> bool:
        if not self._in_creator_process():
            return False
        with self._state_lock:
            return not self._owner_closed

    def _in_creator_process(self) -> bool:
        return os.getpid() == self._creator_pid

    def _require_creator_process(self) -> None:
        if not self._in_creator_process():
            raise RuntimeError("repository root lease belongs to another process")

    def require_active(self) -> None:
        self._require_creator_process()
        with self._state_lock:
            if self._owner_closed or self._stream.closed:
                raise RuntimeError("repository root lease owner is closed")
        with _PROCESS_LEASE_LOCK:
            if self._key not in _PROCESS_LEASES:
                raise RuntimeError("repository root lease lost process ownership")

    def borrow(self) -> "RepositoryRootLeaseBorrow":
        self._require_creator_process()
        with self._state_lock:
            if self._owner_closed or self._stream.closed:
                raise RuntimeError("repository root lease owner is closed")
            object.__setattr__(self, "_borrow_count", self._borrow_count + 1)
        return RepositoryRootLeaseBorrow(_BORROW_TOKEN, self)

    def _require_borrow_active(self) -> None:
        self._require_creator_process()
        with self._state_lock:
            if self._owner_closed or self._stream.closed or self._borrow_count <= 0:
                raise RuntimeError("repository root lease borrow is no longer held")
        with _PROCESS_LEASE_LOCK:
            if self._key not in _PROCESS_LEASES:
                raise RuntimeError("repository root lease lost process ownership")

    def _release_borrow(self, token: object) -> None:
        if token is not _BORROW_TOKEN:
            raise PermissionError("repository lease borrow release is private")
        self._require_creator_process()
        with self._state_lock:
            if self._borrow_count <= 0:
                raise RuntimeError("repository lease borrow count underflow")
            object.__setattr__(self, "_borrow_count", self._borrow_count - 1)

    def close(self) -> None:
        self.close_guarded(lambda: None)

    def close_guarded(self, finalize_owner: Callable[[], None]) -> None:
        """Finalize dependent owners while new borrows remain atomically barred.

        Repository-owned journals and other lifetime resources must close while
        the root's OS lease is still held.  The state lock makes the quiescence
        check, dependent finalization, and root release one exclusion region, so
        an operation cannot acquire a late borrow between those steps.
        """

        if not callable(finalize_owner):
            raise TypeError("finalize_owner must be callable")
        if not self._in_creator_process():
            # ``flock`` ownership is shared by open-file descriptions after
            # fork.  Close only this inherited descriptor; explicit LOCK_UN
            # here would release the live parent's authority.
            if not self._owner_closed:
                object.__setattr__(self, "_owner_closed", True)
                self._stream.close()
                _PROCESS_LEASES.discard(self._key)
            raise RuntimeError("repository root lease belongs to another process")
        with self._state_lock:
            if self._owner_closed:
                return
            if self._borrow_count:
                raise RuntimeError(
                    "repository root lease has outstanding operations"
                )
            finalize_owner()
            object.__setattr__(self, "_owner_closed", True)
            # Keep the state lock through the OS unlock/close.  A concurrent
            # close may return only after this one has actually released the
            # process and kernel ownership, so immediate reopen is reliable.
            self._release_os_resources()

    def _release_os_resources(self) -> None:
        error: BaseException | None = None
        try:
            release_file_lock(self._stream)
        except BaseException as exc:
            error = exc
        try:
            self._stream.close()
        except BaseException as exc:
            if error is None:
                error = exc
        with _PROCESS_LEASE_LOCK:
            _PROCESS_LEASES.discard(self._key)
        if error is not None:
            raise error

    def __enter__(self) -> "RepositoryRootLease":
        self.require_active()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class RepositoryRootLeaseBorrow:
    """One explicit in-flight commit hold on a repository root lease."""

    __slots__ = ("_lease", "_closed", "_close_lock")

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("RepositoryRootLeaseBorrow is final and cannot be subclassed")

    def __init__(self, token: object, lease: RepositoryRootLease) -> None:
        if token is not _BORROW_TOKEN:
            raise PermissionError(
                "RepositoryRootLeaseBorrow can only be minted by RepositoryRootLease"
            )
        object.__setattr__(self, "_lease", lease)
        object.__setattr__(self, "_closed", False)
        object.__setattr__(self, "_close_lock", threading.Lock())

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("RepositoryRootLeaseBorrow is immutable")

    def __reduce__(self):
        raise TypeError(
            "RepositoryRootLeaseBorrow is process-local and cannot be serialized"
        )

    def __reduce_ex__(self, _protocol: int):
        raise TypeError(
            "RepositoryRootLeaseBorrow is process-local and cannot be serialized"
        )

    @property
    def root(self) -> Path:
        self.require_active()
        return self._lease._root

    @property
    def active(self) -> bool:
        if not self._lease._in_creator_process():
            return False
        with self._close_lock:
            return not self._closed

    def require_active(self) -> None:
        self._lease._require_creator_process()
        with self._close_lock:
            if self._closed:
                raise RuntimeError("repository root lease borrow is closed")
            self._lease._require_borrow_active()

    def close(self) -> None:
        if not self._lease._in_creator_process():
            object.__setattr__(self, "_closed", True)
            raise RuntimeError("repository root lease borrow belongs to another process")
        with self._close_lock:
            if self._closed:
                return
            self._lease._release_borrow(_BORROW_TOKEN)
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> "RepositoryRootLeaseBorrow":
        self.require_active()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


__all__ = [
    "RepositoryRootBusy",
    "RepositoryRootLease",
    "RepositoryRootLeaseBorrow",
]
