"""Crash-repairable, idempotent append journal for canonical primitive records."""

from __future__ import annotations

import hashlib
import os
import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .canonical import canonical_text, decode, encode
from . import durability
from .file_lock import (
    acquire_file_lock,
    open_durable_lock_file,
    release_file_lock,
)


_MAGIC = b"ZLCJNL1\n"
_HEADER = struct.Struct(">8sQ32s")
_DEFAULT_MAX_RECORD_BYTES = 64 * 1024 * 1024


class JournalCorruptionError(RuntimeError):
    pass


def _record_id(value: str) -> str:
    return canonical_text(value, "journal record_id")


@contextmanager
def _interprocess_lock(path: Path) -> Iterator[None]:
    lock_file = path.open("r+b")
    try:
        acquire_file_lock(lock_file, blocking=True)
        try:
            yield
        finally:
            release_file_lock(lock_file)
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
        durability.durable_mkdir(self.path.parent)
        open_durable_lock_file(self.lock_path).close()
        existed = self.path.exists()
        with self._thread_lock, _interprocess_lock(self.lock_path):
            with self.path.open("a+b") as stream:
                self._scan(stream, repair_torn_tail=True)
                if not existed:
                    stream.flush()
                    os.fsync(stream.fileno())
            if not existed:
                durability.flush_directory(self.path.parent)

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
                    previous_payload, _previous_value = previous
                    if previous_payload != payload:
                        raise ValueError(
                            f"journal record id {record_id!r} has conflicting content"
                        )
                    validate(self._record_values(existing))
                    return False
                decoded = self._record_values(existing)
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
                records = self._scan(stream, repair_torn_tail=True)
        return self._record_values(records)

    @staticmethod
    def _record_values(
        records: dict[str, tuple[bytes, Any]],
    ) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (record_id, value)
            for record_id, (_payload, value) in records.items()
        )

    def _scan(
        self,
        stream,
        *,
        repair_torn_tail: bool,
    ) -> dict[str, tuple[bytes, Any]]:
        stream.seek(0)
        records: dict[str, tuple[bytes, Any]] = {}
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
            previous = records.get(record_id)
            if previous is not None and previous[0] != payload:
                raise JournalCorruptionError(
                    f"journal record id {record_id!r} has conflicting content"
                )
            records[record_id] = (payload, decoded["value"])
            valid_end = stream.tell()
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if valid_end != end:
            if not repair_torn_tail:
                raise JournalCorruptionError("journal has a torn final frame")
            stream.truncate(valid_end)
            stream.flush()
            os.fsync(stream.fileno())
        return records
