"""Queued GUI-owner wake primitive."""

from __future__ import annotations

import threading
from typing import Callable

from PyQt5 import QtCore

from ..render import detached_render_fault


class QtOwnerWake(QtCore.QObject):
    """No-payload queued wake bound once to a GUI-owner callback."""

    requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._callback: Callable[[], object] | None = None
        self._fault: BaseException | None = None
        self._pending_lock = threading.Lock()
        self._scheduled = False
        self._dispatching = False
        self._replay = False
        self.requested.connect(self._dispatch, QtCore.Qt.QueuedConnection)

    @property
    def fault(self) -> BaseException | None:
        return self._fault

    def bind(self, callback: Callable[[], object]) -> None:
        self._require_owner()
        if not callable(callback):
            raise TypeError("callback must be callable")
        if self._callback is not None:
            raise RuntimeError("QtOwnerWake is already bound")
        self._callback = callback
        self._fault = None

    def request_owner_wake(self) -> None:
        """Queue at most one owner turn while retaining a concurrent replay.

        Worker completions are level-triggered: the owner callback drains every
        result currently available.  Emitting one queued Qt event per completed
        future therefore manufactures redundant owner turns when several
        futures finish together.  A completion arriving while the callback is
        running requests exactly one replay, so no result can be stranded.
        """

        with self._pending_lock:
            if self._scheduled:
                # Several completions can arrive before Qt begins the already
                # queued owner turn.  That turn drains all of them, so another
                # event would be empty work.  Replay is needed only when a
                # completion races with the callback that is already draining.
                if self._dispatching:
                    self._replay = True
                return
            self._scheduled = True
        self.requested.emit()

    def detach(self) -> None:
        self._require_owner()
        self._callback = None

    @QtCore.pyqtSlot()
    def _dispatch(self) -> None:
        with self._pending_lock:
            self._dispatching = True
        try:
            callback = self._callback
            if callback is not None:
                callback()
        except BaseException as error:
            self._fault = detached_render_fault(error)
        finally:
            with self._pending_lock:
                self._dispatching = False
                if self._replay:
                    self._replay = False
                    replay = True
                else:
                    self._scheduled = False
                    replay = False
            if replay:
                self.requested.emit()

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtOwnerWake binding is GUI-thread affine")

__all__ = ["QtOwnerWake"]
