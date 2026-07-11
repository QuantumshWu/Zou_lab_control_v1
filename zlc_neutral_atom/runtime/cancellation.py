"""Monotonic cooperative cancellation for one run invocation."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CancellationSnapshot:
    requested: bool
    reason: str | None
    requested_at: float | None


class CancellationRequested(RuntimeError):
    """Raised at a cooperative cancellation checkpoint."""

    def __init__(self, reason: str | None = None) -> None:
        super().__init__("run cancellation requested" if reason is None else reason)
        self.reason = reason


class CancellationToken:
    """A one-way token: active -> cancelled, never reset or reused."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._requested_at: float | None = None

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def request(self, reason: str | None = None) -> bool:
        """Request cancellation once; return True only for the first request."""

        if reason is not None:
            if not isinstance(reason, str) or not reason or reason.strip() != reason:
                raise ValueError("cancellation reason must be non-empty canonical text or None")
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._requested_at = time.monotonic()
            self._event.set()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def checkpoint(self) -> None:
        if self._event.is_set():
            raise CancellationRequested(self.snapshot().reason)

    def snapshot(self) -> CancellationSnapshot:
        with self._lock:
            return CancellationSnapshot(
                requested=self._event.is_set(),
                reason=self._reason,
                requested_at=self._requested_at,
            )
