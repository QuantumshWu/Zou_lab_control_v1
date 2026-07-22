"""Shared Qt-owner launch and capacity-one raster-work runtime."""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable

from PyQt5 import QtCore

from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.qt_widgets import (
    WINDOW_SCREEN_FRACTION,
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
    "cancel_export_commits",
    "error_summary",
    "load_raster_bundle",
    "open_workbench_window",
    "release_window",
    "stage_and_replace_export",
]
