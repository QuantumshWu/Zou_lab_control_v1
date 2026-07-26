"""Shared Qt-owner launch and capacity-one raster-work runtime."""

from __future__ import annotations

from concurrent.futures import CancelledError, Executor, Future, ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.qt_widgets import (
    WINDOW_SCREEN_FRACTION,
    QtOwnerWake,
    center_window_on_primary_screen,
    ensure_qt_app,
    release_window,
    retain_window,
    screen_fit_window_size,
    set_fluent_scale,
)


RASTER_WORK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zlc-raster-work",
)


def wait_for_owner_retirement(
    owner: QtCore.QObject,
    completed: threading.Event,
    *,
    timeout: float,
) -> bool:
    """Wait for an application-owned Qt window without starving its owner."""

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
    """One Qt-owner handoff for a capacity-one worker session.

    Subclasses own their widgets and accepted result semantics.  This base is
    the sole owner of future handoff, cancellation, worker-affine final
    release, and the asynchronous QWidget close boundary.  Interactive
    ``BoardFrame`` hosts and immutable encoded-raster hosts therefore share the
    lifecycle without pretending that their presentation payloads are alike.
    """

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
        """Install the one final worker-affine release before shutdown."""

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
        """Keep final-release diagnostics outside the Qt lifecycle boundary."""

        try:
            self._worker_release_failed(error)
        except BaseException:
            # Final release is terminal: a diagnostic hook must never turn its
            # own failure into an uncaught Qt exception or prevent shell close.
            pass

    def _consume_worker_release_future(self, future: Future) -> bool:
        """Consume the tagged final-release Future before payload dispatch."""

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
    message = str(error).strip()
    return type(error).__name__ if not message else f"{type(error).__name__}: {message}"


def load_raster_bundle(
    loader: Callable[[threading.Event], EncodedRasterDocument],
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    """Load one immutable raster bundle unless the window was cancelled."""

    if cancelled.is_set():
        raise CancelledError()
    bundle = loader(cancelled)
    if not isinstance(bundle, EncodedRasterDocument):
        raise TypeError("raster loader must return EncodedRasterDocument")
    if cancelled.is_set():
        raise CancelledError()
    return bundle


def stage_and_replace_export(
    destination: Path,
    *,
    write_staged: Callable[[Path], None],
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> Path:
    """Publish one staged display export atomically relative to window close."""

    if not callable(write_staged):
        raise TypeError("staged export writer must be callable")
    if cancelled.is_set():
        raise CancelledError()
    target = Path(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_staged(temporary)
        with commit_lock:
            if cancelled.is_set():
                raise CancelledError()
            os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cancel_export_commits(
    *,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> None:
    """Linearize close before exports that have not replaced their target."""

    with commit_lock:
        cancelled.set()


def open_workbench_window(factory):
    """Construct, size, retain, and show one Workbench window on the Qt owner."""

    if not callable(factory):
        raise TypeError("Workbench window factory must be callable")
    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("Workbench must be opened on the Qt GUI thread")
    set_fluent_scale(None)
    window = factory()
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


__all__ = [
    "RASTER_WORK_EXECUTOR",
    "SerialWorkerWindow",
    "cancel_export_commits",
    "error_summary",
    "load_raster_bundle",
    "open_workbench_window",
    "release_window",
    "stage_and_replace_export",
    "wait_for_owner_retirement",
]
