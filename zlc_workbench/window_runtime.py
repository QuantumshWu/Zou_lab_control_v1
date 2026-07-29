"""Workbench-owned compute submission and atomic export publication."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable


_COMPUTE_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="zlc-compute-work",
)


def submit_compute(function, *args) -> Future:
    """Submit non-raster compute through the Workbench composition owner."""

    if not callable(function):
        raise TypeError("compute work must be callable")
    return _COMPUTE_EXECUTOR.submit(function, *args)


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


__all__ = [
    "cancel_export_commits",
    "stage_and_replace_export",
    "submit_compute",
]
