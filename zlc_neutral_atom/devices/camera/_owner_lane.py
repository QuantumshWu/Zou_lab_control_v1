"""Single-owner execution lane shared by owner-affine camera SDKs."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar


_T = TypeVar("_T")


class CameraSdkOwnerLane:
    """Serialize SDK calls on one stable non-daemon thread.

    Camera adapters keep their own acquisition semantics.  This class owns only
    thread affinity and synchronous command handoff; it is not a scheduler,
    buffer, registry, or lifecycle state machine.
    """

    _STOP = object()

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("camera SDK owner lane name must be non-empty text")
        self._name = name
        self._commands: queue.Queue[object] = queue.Queue()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._owner_ident: int | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=False)
        self._thread.start()

    @property
    def owner_ident(self) -> int | None:
        return self._owner_ident

    def _run(self) -> None:
        self._owner_ident = threading.get_ident()
        while True:
            item = self._commands.get()
            if item is self._STOP:
                return
            function, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function())
            except BaseException as error:
                future.set_exception(error)

    def call(self, function: Callable[[], _T]) -> _T:
        if not callable(function):
            raise TypeError("camera SDK owner command must be callable")
        if threading.get_ident() == self._owner_ident:
            return function()
        future: Future[_T] = Future()
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError(f"{self._name} is closed")
            self._commands.put((function, future))
        return future.result()

    def close(self) -> None:
        if threading.get_ident() == self._owner_ident:
            raise RuntimeError("camera SDK owner lane cannot join itself")
        with self._lifecycle_lock:
            if self._closed:
                should_join = self._thread.is_alive()
            else:
                self._closed = True
                self._commands.put(self._STOP)
                should_join = True
        if should_join:
            self._thread.join()


__all__ = ["CameraSdkOwnerLane"]
