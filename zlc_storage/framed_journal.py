"""Crash-repairable, idempotent append journal for canonical primitive records."""

from __future__ import annotations

import hashlib
import os
import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .canonical import decode, encode


_MAGIC = b"ZLCJNL1\n"
_HEADER = struct.Struct(">8sQ32s")
_DEFAULT_MAX_RECORD_BYTES = 64 * 1024 * 1024


class JournalCorruptionError(RuntimeError):
    pass


def _fsync_directory(path: Path) -> None:
    """Persist directory entries where the host exposes directory fsync."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
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
        handle = create_file(
            str(path),
            0x40000000,  # GENERIC_WRITE is required for FlushFileBuffers on directories.
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,  # OPEN_EXISTING
            0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
            None,
        )
        invalid = wintypes.HANDLE(-1).value
        if handle == invalid:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not ctypes.WinDLL("kernel32", use_last_error=True).FlushFileBuffers(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_id(value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("journal record_id must be canonical non-empty text")
    return value


@contextmanager
def _interprocess_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    lock_file = path.open("a+b")
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
            os.fsync(lock_file.fileno())
            if not existed:
                _fsync_directory(path.parent)
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


class FramedJournal:
    """One file, canonical payloads, SHA-256 frames, and safe torn-tail repair."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES,
    ) -> None:
        if isinstance(max_record_bytes, bool) or not isinstance(max_record_bytes, int):
            raise TypeError("max_record_bytes must be an integer")
        if max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.max_record_bytes = max_record_bytes
        self._thread_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        with self._thread_lock, _interprocess_lock(self.lock_path):
            with self.path.open("a+b") as stream:
                self._scan(stream, repair_torn_tail=True)
                if not existed:
                    stream.flush()
                    os.fsync(stream.fileno())
            if not existed:
                _fsync_directory(self.path.parent)

    def append(self, record_id: str, value: Any) -> bool:
        return self.append_checked(record_id, value, lambda _records: None)

    def append_checked(
        self,
        record_id: str,
        value: Any,
        validate: Callable[[tuple[tuple[str, Any], ...]], None],
    ) -> bool:
        """Append after validating the prospective log under the process lock."""

        record_id = _record_id(record_id)
        if not callable(validate):
            raise TypeError("validate must be callable")
        payload = encode({"record_id": record_id, "value": value})
        if len(payload) > self.max_record_bytes:
            raise ValueError("journal record exceeds max_record_bytes")
        frame = _HEADER.pack(_MAGIC, len(payload), hashlib.sha256(payload).digest()) + payload
        with self._thread_lock, _interprocess_lock(self.lock_path):
            with self.path.open("r+b") as stream:
                existing = self._scan(stream, repair_torn_tail=True)
                previous = existing.get(record_id)
                if previous is not None:
                    if previous != payload:
                        raise ValueError(
                            f"journal record id {record_id!r} has conflicting content"
                        )
                    validate(self._decode_payloads(existing))
                    return False
                decoded = self._decode_payloads(existing)
                candidate_value = decode(payload)["value"]
                validate(decoded + ((record_id, candidate_value),))
                stream.seek(0, os.SEEK_END)
                stream.write(frame)
                stream.flush()
                os.fsync(stream.fileno())
                return True

    def records(self) -> tuple[tuple[str, Any], ...]:
        with self._thread_lock, _interprocess_lock(self.lock_path):
            with self.path.open("r+b") as stream:
                payloads = self._scan(stream, repair_torn_tail=True)
        return self._decode_payloads(payloads)

    @staticmethod
    def _decode_payloads(payloads: dict[str, bytes]) -> tuple[tuple[str, Any], ...]:
        result = []
        for record_id, payload in payloads.items():
            decoded = decode(payload)
            result.append((record_id, decoded["value"]))
        return tuple(result)

    def _scan(self, stream, *, repair_torn_tail: bool) -> dict[str, bytes]:
        stream.seek(0)
        payloads: dict[str, bytes] = {}
        valid_end = 0
        while True:
            frame_start = stream.tell()
            header = stream.read(_HEADER.size)
            if not header:
                valid_end = frame_start
                break
            if len(header) < _HEADER.size:
                valid_end = frame_start
                break
            magic, size, digest = _HEADER.unpack(header)
            if magic != _MAGIC:
                raise JournalCorruptionError(
                    f"invalid journal frame magic at byte {frame_start}"
                )
            if size > self.max_record_bytes:
                raise JournalCorruptionError(
                    f"journal frame at byte {frame_start} exceeds configured limit"
                )
            payload = stream.read(size)
            if len(payload) < size:
                valid_end = frame_start
                break
            if hashlib.sha256(payload).digest() != digest:
                raise JournalCorruptionError(
                    f"journal frame digest mismatch at byte {frame_start}"
                )
            try:
                decoded = decode(payload)
                record_id = _record_id(decoded["record_id"])
                if set(decoded) != {"record_id", "value"}:
                    raise ValueError("unexpected journal record fields")
            except Exception as exc:
                raise JournalCorruptionError(
                    f"invalid canonical journal record at byte {frame_start}"
                ) from exc
            previous = payloads.get(record_id)
            if previous is not None and previous != payload:
                raise JournalCorruptionError(
                    f"journal record id {record_id!r} has conflicting content"
                )
            payloads[record_id] = payload
            valid_end = stream.tell()
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if valid_end != end:
            if not repair_torn_tail:
                raise JournalCorruptionError("journal has a torn final frame")
            stream.truncate(valid_end)
            stream.flush()
            os.fsync(stream.fileno())
        return payloads
