"""Process-scoped exclusive ownership of the single installed pulse streamer."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Protocol


class DeviceLease(Protocol):
    """Fail-closed process lease acquired before opening any hardware transport."""

    def acquire(self) -> None: ...

    def release(self) -> None: ...


class InterprocessDeviceLease:
    """OS-released exclusive lock shared by the AXI and UART server modes."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.environ.get("ZLC_PULSE_DEVICE_LOCK")
        self.path = Path(configured) if configured else (
            Path(tempfile.gettempdir()) / "zlc-pulse-streamer-device.lock"
        )
        self._lock = threading.RLock()
        self._handle = None

    def acquire(self) -> None:
        with self._lock:
            if self._handle is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            try:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                _lock_handle(handle)
            except BaseException:
                handle.close()
                raise RuntimeError(
                    "pulse streamer is already owned by another AXI/UART process"
                ) from None
            self._handle = handle

    def release(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            try:
                handle.seek(0)
                _unlock_handle(handle)
            finally:
                self._handle = None
                handle.close()


def _lock_handle(handle) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["DeviceLease", "InterprocessDeviceLease"]
