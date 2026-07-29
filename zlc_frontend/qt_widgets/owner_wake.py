"""Queued GUI-owner wake and capacity-one worker lifetime primitives."""

from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
import threading
import time
from typing import Callable

from PyQt5 import QtCore, QtWidgets

from ..render import detached_render_fault
from .fluent import release_window


RASTER_WORK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zlc-raster-work",
)


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


def wait_for_owner_retirement(
    owner: QtCore.QObject,
    completed: threading.Event,
    *,
    timeout: float,
) -> bool:
    """Wait for an application-owned Qt object without starving its owner."""

    if not isinstance(owner, QtCore.QObject):
        raise TypeError("owner must be a Qt object")
    if not isinstance(completed, threading.Event):
        raise TypeError("completed must be a threading.Event")
    seconds = float(timeout)
    if not seconds > 0.0:
        raise ValueError("owner retirement timeout must be positive")
    if completed.is_set():
        return True
    if QtCore.QThread.currentThread() is not owner.thread():
        return completed.wait(seconds)
    application = QtWidgets.QApplication.instance()
    if application is None:
        raise RuntimeError("Qt owner retirement requires the existing QApplication")
    deadline = time.monotonic() + seconds
    while not completed.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        completed.wait(min(0.005, remaining))
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    return completed.is_set()


class SerialWorkerWindow(QtWidgets.QWidget):
    """Capacity-one worker handoff with Qt-owner cancellation and close."""

    def __init__(
        self,
        *,
        executor: Executor | None = None,
        worker_release: Callable[[], object] | None = None,
    ) -> None:
        super().__init__()
        if executor is not None and not isinstance(executor, Executor):
            raise TypeError("executor must implement concurrent.futures.Executor")
        if worker_release is not None and not callable(worker_release):
            raise TypeError("worker_release must be callable or None")
        self._executor = RASTER_WORK_EXECUTOR if executor is None else executor
        self._worker_release = worker_release
        self._worker_release_pending = False
        self._future: Future | None = None
        self._cancelled = threading.Event()
        self._closing = False
        self._closed = False
        self._allow_close = False
        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)

    def _set_worker_release(self, release: Callable[[], object]) -> None:
        if not callable(release):
            raise TypeError("worker release must be callable")
        if self._worker_release is not None:
            raise RuntimeError("worker release is already installed")
        if self._closing:
            raise RuntimeError("worker release cannot change during close")
        self._worker_release = release

    def _submit_future(self, function, *args) -> bool:
        if self._future is not None:
            raise RuntimeError("worker session already has active work")
        try:
            future = self._executor.submit(function, *args)
        except BaseException as error:
            self._worker_submit_failed(error)
            return False
        self._future = future
        future.add_done_callback(lambda _done: self._wake.request_owner_wake())
        return True

    def _worker_submit_failed(self, error: BaseException) -> None:
        raise error

    def _accept_finished_future(self, future: Future) -> None:
        raise NotImplementedError

    def _worker_release_failed(self, error: BaseException) -> None:
        """Report a final release failure without throwing through a Qt slot."""

    def _report_worker_release_failure(self, error: BaseException) -> None:
        try:
            self._worker_release_failed(error)
        except BaseException:
            pass

    def _consume_worker_release_future(self, future: Future) -> bool:
        if not self._worker_release_pending:
            return False
        self._worker_release_pending = False
        try:
            future.result()
        except BaseException as error:
            self._report_worker_release_failure(error)
        return True

    def _after_worker_completion(self) -> None:
        """Let an open subclass enqueue its next already-authored request."""

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        future = self._future
        if future is not None and future.done():
            self._future = None
            if not self._consume_worker_release_future(future):
                self._accept_finished_future(future)
                self._after_worker_completion()
        self._finish_close_if_ready()

    @property
    def worker_idle(self) -> bool:
        return self._future is None

    @property
    def closed(self) -> bool:
        return self._closed

    def _before_worker_shutdown(self) -> None:
        """Clear subclass-owned fronts and disable UI after close linearizes."""

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        self._closing = True
        self._cancelled.set()
        self._before_worker_shutdown()
        future = self._future
        if future is not None:
            future.cancel()
        self._finish_close_if_ready()

    def _finish_close_if_ready(self) -> None:
        if not self._closing or self._future is not None or self._closed:
            return
        release, self._worker_release = self._worker_release, None
        if release is not None:
            try:
                future = self._executor.submit(release)
            except BaseException as error:
                self._report_worker_release_failure(error)
            else:
                self._worker_release_pending = True
                self._future = future
                future.add_done_callback(
                    lambda _done: self._wake.request_owner_wake()
                )
                return
        self._wake.detach()
        self._closed = True
        self._allow_close = True
        QtCore.QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:
        if self._allow_close:
            release_window(self)
            event.accept()
            return
        event.ignore()
        self.shutdown()


def error_summary(error: BaseException) -> str:
    """Format one exception for a compact operator diagnostic."""

    message = str(error).strip()
    return type(error).__name__ if not message else f"{type(error).__name__}: {message}"


__all__ = [
    "error_summary",
    "QtOwnerWake",
    "RASTER_WORK_EXECUTOR",
    "SerialWorkerWindow",
    "wait_for_owner_retirement",
]
