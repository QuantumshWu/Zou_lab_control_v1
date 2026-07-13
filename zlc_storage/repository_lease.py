"""Process-lifetime exclusive ownership of one durable repository root."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .durability import durable_mkdir, flush_directory


_LEASE_FILENAME = ".zlc-repository-root.lock"
_PROCESS_LEASE_LOCK = threading.Lock()
_PROCESS_LEASES: set[str] = set()
_BORROW_TOKEN = object()


class RepositoryRootBusy(RuntimeError):
    """Another live repository owner already holds this root."""


def _root_key(root: Path) -> str:
    return os.path.normcase(str(root))


def _lock_file(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RepositoryRootBusy(
                "repository root is owned by another live process"
            ) from exc
        return
    import fcntl

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RepositoryRootBusy(
            "repository root is owned by another live process"
        ) from exc


def _unlock_file(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RepositoryRootLease:
    """Exclusive OS + in-process lease held for a repository owner's lifetime."""

    __slots__ = (
        "_root",
        "_owner",
        "_key",
        "_stream",
        "_owner_closed",
        "_borrow_count",
        "_state_lock",
        "_os_released",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("RepositoryRootLease is final and cannot be subclassed")

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        owner: str,
    ) -> None:
        if not isinstance(owner, str) or not owner or owner.strip() != owner:
            raise ValueError("repository lease owner must be canonical non-empty text")
        resolved = Path(root).expanduser().resolve()
        durable_mkdir(resolved)
        # This is also the construction-time backend preflight for an already
        # existing hierarchy: no repository starts if directory flush is fake.
        flush_directory(resolved)
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
            existed = lock_path.exists()
            stream = lock_path.open("a+b")
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
                if not existed:
                    flush_directory(resolved)
            _lock_file(stream)
        except BaseException:
            if stream is not None:
                stream.close()
            with _PROCESS_LEASE_LOCK:
                _PROCESS_LEASES.discard(key)
            raise

        object.__setattr__(self, "_root", resolved)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_owner_closed", False)
        object.__setattr__(self, "_borrow_count", 0)
        object.__setattr__(self, "_state_lock", threading.Lock())
        object.__setattr__(self, "_os_released", False)

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
    def owner(self) -> str:
        self.require_active()
        return self._owner

    @property
    def active(self) -> bool:
        with self._state_lock:
            return not self._owner_closed

    def require_active(self) -> None:
        with self._state_lock:
            if self._owner_closed or self._os_released or self._stream.closed:
                raise RuntimeError("repository root lease owner is closed")
        with _PROCESS_LEASE_LOCK:
            if self._key not in _PROCESS_LEASES:
                raise RuntimeError("repository root lease lost process ownership")

    def borrow(self) -> "RepositoryRootLeaseBorrow":
        with self._state_lock:
            if self._owner_closed or self._os_released or self._stream.closed:
                raise RuntimeError("repository root lease owner is closed")
            object.__setattr__(self, "_borrow_count", self._borrow_count + 1)
        return RepositoryRootLeaseBorrow(_BORROW_TOKEN, self)

    def _require_borrow_active(self) -> None:
        with self._state_lock:
            if self._os_released or self._stream.closed or self._borrow_count <= 0:
                raise RuntimeError("repository root lease borrow is no longer held")
        with _PROCESS_LEASE_LOCK:
            if self._key not in _PROCESS_LEASES:
                raise RuntimeError("repository root lease lost process ownership")

    def _release_borrow(self, token: object) -> None:
        if token is not _BORROW_TOKEN:
            raise PermissionError("repository lease borrow release is private")
        with self._state_lock:
            if self._borrow_count <= 0:
                raise RuntimeError("repository lease borrow count underflow")
            object.__setattr__(self, "_borrow_count", self._borrow_count - 1)

    def close(self) -> None:
        with self._state_lock:
            if self._owner_closed:
                return
            if self._borrow_count:
                raise RuntimeError(
                    "repository root lease has outstanding commit authorities"
                )
            object.__setattr__(self, "_owner_closed", True)
            object.__setattr__(self, "_os_released", True)
        self._release_os_resources()

    def _release_os_resources(self) -> None:
        error: BaseException | None = None
        try:
            _unlock_file(self._stream)
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

    __slots__ = ("_lease", "_closed")

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("RepositoryRootLeaseBorrow is final and cannot be subclassed")

    def __init__(self, token: object, lease: RepositoryRootLease) -> None:
        if token is not _BORROW_TOKEN:
            raise PermissionError(
                "RepositoryRootLeaseBorrow can only be minted by RepositoryRootLease"
            )
        object.__setattr__(self, "_lease", lease)
        object.__setattr__(self, "_closed", False)

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
        return not self._closed

    def require_active(self) -> None:
        if self._closed:
            raise RuntimeError("repository root lease borrow is closed")
        self._lease._require_borrow_active()

    def close(self) -> None:
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        self._lease._release_borrow(_BORROW_TOKEN)

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
