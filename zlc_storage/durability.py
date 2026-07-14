"""Filesystem directory durability primitives shared by storage owners."""

from __future__ import annotations

import os
from pathlib import Path


class DirectoryDurabilityError(RuntimeError):
    """The filesystem could not durably flush a directory entry."""


def _flush_windows_directory(directory: Path) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flush_file_buffers = kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = (wintypes.HANDLE,)
    flush_file_buffers.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        str(directory),
        0x40000000,  # GENERIC_WRITE, required by FlushFileBuffers.
        0x00000001 | 0x00000002 | 0x00000004,  # share read/write/delete
        None,
        3,  # OPEN_EXISTING
        0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS opens a directory handle.
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    flush_error: OSError | None = None
    try:
        if not flush_file_buffers(handle):
            flush_error = ctypes.WinError(ctypes.get_last_error())
    finally:
        if not close_handle(handle) and flush_error is None:
            flush_error = ctypes.WinError(ctypes.get_last_error())
    if flush_error is not None:
        raise flush_error


def _flush_posix_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def flush_directory(directory: str | os.PathLike[str]) -> None:
    """Durably flush one existing directory or fail explicitly.

    Windows uses a real directory handle plus ``FlushFileBuffers``; POSIX uses
    ``fsync`` on an open directory descriptor.  There is deliberately no
    best-effort/no-op backend because callers use this as commit evidence.
    """

    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    try:
        if os.name == "nt":
            _flush_windows_directory(path)
        else:
            _flush_posix_directory(path)
    except DirectoryDurabilityError:
        raise
    except OSError as exc:
        raise DirectoryDurabilityError(
            f"directory durability flush failed for {path}"
        ) from exc


def durable_mkdir(directory: str | os.PathLike[str]) -> Path:
    """Create or re-acknowledge one directory below an existing parent.

    The child and parent are flushed on every call, including retries after a
    previous parent flush failed.  Hierarchy owners must call this once per
    level; hiding recursive creation here would make a visible child look
    durable on retry without re-acknowledging the entry whose flush failed.
    """

    target = Path(directory).expanduser().resolve()
    parent = target.parent
    if parent == target:
        if not target.is_dir():
            raise NotADirectoryError(target)
        flush_directory(target)
        return target
    if not parent.exists():
        raise FileNotFoundError(
            f"parent directory does not exist for durable mkdir: {parent}"
        )
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    if not target.exists():
        try:
            target.mkdir()
        except FileExistsError:
            if not target.is_dir():
                raise NotADirectoryError(target)
    if not target.is_dir():
        raise NotADirectoryError(target)
    flush_directory(target)
    flush_directory(parent)
    return target


__all__ = [
    "DirectoryDurabilityError",
    "durable_mkdir",
    "flush_directory",
]
