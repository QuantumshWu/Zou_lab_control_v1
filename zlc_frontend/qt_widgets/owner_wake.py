"""Queued GUI-owner wake primitive."""

from __future__ import annotations

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
        self.requested.emit()

    def detach(self) -> None:
        self._require_owner()
        self._callback = None

    @QtCore.pyqtSlot()
    def _dispatch(self) -> None:
        callback = self._callback
        if callback is None:
            return
        try:
            callback()
        except BaseException as error:
            self._fault = detached_render_fault(error)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtOwnerWake binding is GUI-thread affine")

__all__ = ["QtOwnerWake"]
