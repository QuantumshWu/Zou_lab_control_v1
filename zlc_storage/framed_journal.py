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
    ) -> None:
        self._initialize(path, scan=True)

    @classmethod
    def open_exclusive(
        cls,
        path: str | os.PathLike[str],
    ) -> "FramedJournalSession":
        """Create a lifetime-exclusive session with one startup scan."""

        journal = cls.__new__(cls)
        journal._initialize(path, scan=False)
        return FramedJournalSession(journal)

    def _initialize(
        self,
        path: str | os.PathLike[str],
        *,
        scan: bool,
    ) -> None:
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._thread_lock = threading.Lock()
        durability.durable_mkdir(self.path.parent)
        open_durable_lock_file(self.lock_path).close()
        if not scan:
            return
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


class FramedJournalSession:
    """Exclusive cached writer for one :class:`FramedJournal` lifetime."""

    def __init__(self, journal: FramedJournal) -> None:
        if not isinstance(journal, FramedJournal):
            raise TypeError("journal must be FramedJournal")
        self._journal = journal
        self._creator_pid = os.getpid()
        self._closed = False
        self._owns_lock = False
        self._lock_stream = None
        self._stream = None
        try:
            self._lock_stream = journal.lock_path.open("r+b")
            acquire_file_lock(self._lock_stream, blocking=False)
            self._owns_lock = True
            existed = journal.path.exists()
            self._stream = journal.path.open("a+b")
            self._records = journal._scan(self._stream, repair_torn_tail=True)
            if not existed:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                durability.flush_directory(journal.path.parent)
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise cleanup_error from exc
            raise

    def records(self) -> tuple[tuple[str, Any], ...]:
        self._ensure_open()
        return self._journal._record_values(self._records)

    def append(self, record_id: str, value: Any) -> bool:
        """Append against the cached projection without rescanning old frames."""

        self._ensure_open()
        record_id = _record_id(record_id)
        payload = encode({"record_id": record_id, "value": value})
        previous = self._records.get(record_id)
        if previous is not None:
            if previous[0] != payload:
                raise ValueError(
                    f"journal record id {record_id!r} has conflicting content"
                )
            return False
        frame = (
            _HEADER.pack(_MAGIC, len(payload), hashlib.sha256(payload).digest())
            + payload
        )
        candidate_value = decode(payload)["value"]
        assert self._stream is not None
        try:
            self._stream.seek(0, os.SEEK_END)
            self._stream.write(frame)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            # A failed write/fsync has ambiguous durability.  Rebuild the cache
            # from the same locked file before returning control to the caller.
            self._records = self._journal._scan(
                self._stream,
                repair_torn_tail=True,
            )
            raise
        self._records[record_id] = (payload, candidate_value)
        return True

    @property
    def owns_file_lock(self) -> bool:
        """Whether this process still owns the journal authority lock."""

        return os.getpid() == self._creator_pid and self._owns_lock

    def close(self) -> None:
        if self._closed:
            return
        if self._stream is not None:
            stream = self._stream
            try:
                stream.close()
            except BaseException:
                if stream.closed:
                    self._stream = None
                raise
            self._stream = None
        if self._lock_stream is not None:
            lock_stream = self._lock_stream
            release_error: BaseException | None = None
            if os.getpid() == self._creator_pid and self._owns_lock:
                try:
                    release_file_lock(lock_stream)
                except BaseException as error:
                    release_error = error
                else:
                    self._owns_lock = False
            elif os.getpid() != self._creator_pid:
                # A fork child owns only duplicated descriptors.  Explicitly
                # unlocking their shared file description would unlock the parent.
                self._owns_lock = False
            try:
                lock_stream.close()
            except BaseException as error:
                if lock_stream.closed:
                    self._lock_stream = None
                    # Closing the last creator descriptor releases the OS lock
                    # even when the explicit unlock call itself reported an
                    # error.  Logical authority must never revive after that.
                    self._owns_lock = False
                if release_error is not None:
                    error.add_note(
                        "explicit journal lock release also failed: "
                        f"{type(release_error).__name__}: {release_error}"
                    )
                raise
            else:
                self._lock_stream = None
                self._owns_lock = False
            if release_error is not None:
                self._closed = True
                raise release_error
        self._closed = not self._owns_lock
        if not self._closed:
            raise RuntimeError("framed journal session still owns its file lock")

    def _ensure_open(self) -> None:
        if os.getpid() != self._creator_pid:
            raise RuntimeError("framed journal session belongs to another process")
        if self._closed or not self._owns_lock or self._stream is None:
            raise RuntimeError("framed journal session is closed")

    def __enter__(self) -> "FramedJournalSession":
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
