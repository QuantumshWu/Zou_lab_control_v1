"""Platform mechanics for one-byte advisory file locks."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import BinaryIO

from .durability import flush_directory


class FileLockBusy(RuntimeError):
    """A non-blocking advisory lock is held by another live owner."""


def open_durable_lock_file(path: str | os.PathLike[str]) -> BinaryIO:
    """Open an existing-parent lock file and persist its file and name.

    Preparation is repeated even when the file already exists.  A retry after
    a failed directory flush therefore cannot confuse a visible lock file with
    a durably acknowledged one.  Lock paths are permanent identities: owners
    must never unlink or replace a lock file while any process may hold it.
    """

    lock_path = Path(path).expanduser().resolve()
    if not lock_path.parent.is_dir():
        raise FileNotFoundError(
            f"lock-file parent directory does not exist: {lock_path.parent}"
        )
    stream = lock_path.open("a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
        stream.flush()
        os.fsync(stream.fileno())
        flush_directory(lock_path.parent)
        return stream
    except BaseException:
        stream.close()
        raise


def _is_contention(error: OSError) -> bool:
    return error.errno in {
        errno.EACCES,
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EDEADLK,
    } or getattr(error, "winerror", None) == 33


def acquire_file_lock(stream: BinaryIO, *, blocking: bool) -> None:
    """Acquire the first byte using the platform's blocking/nonblocking mode.

    Windows ``LK_LOCK`` has a bounded CRT retry window; exhaustion propagates
    as an OS error and therefore remains fail-closed.
    """

    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            msvcrt.locking(stream.fileno(), mode, 1)
        else:
            import fcntl

            mode = fcntl.LOCK_EX
            if not blocking:
                mode |= fcntl.LOCK_NB
            fcntl.flock(stream.fileno(), mode)
    except OSError as exc:
        if not blocking and _is_contention(exc):
            raise FileLockBusy("advisory file lock is already held") from exc
        raise


def release_file_lock(stream: BinaryIO) -> None:
    """Release the first-byte advisory lock held by ``stream``."""

    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "FileLockBusy",
    "acquire_file_lock",
    "open_durable_lock_file",
    "release_file_lock",
]
