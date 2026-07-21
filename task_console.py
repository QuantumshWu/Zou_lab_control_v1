"""Standalone launcher for the current typed TaskConsole workbench."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the current Zou_lab_control TaskConsole.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Open with a saved console layout (a tasks/<name>.json path).",
    )
    parser.add_argument(
        "--task",
        help="Open with a saved console layout BY NAME from the workspace's tasks/.",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.home() / ".zlc" / "task_console",
        help="Virtual Experiment workspace.",
    )
    parser.add_argument(
        "--name",
        default="task_console",
        help="Virtual Experiment name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Virtual installation seed.",
    )
    return parser


def _close_window(application, window, *, timeout_seconds: float = 10.0) -> None:
    """Close the console's TOP-LEVEL window and wait for it to actually go.

    ``open_task_console`` returns the console BODY, not the frame around it (the body's
    ``isWindow()`` is False; its ``window()`` is the Fluent frame).  Closing the body leaves
    the frame open, the app keeps running, and the launcher hangs forever in ``exec_()`` --
    which is exactly what happened the first time this entry was switched.
    """

    if window is None:
        return
    window = window.window()
    if not window.isVisible():
        return
    window.close()
    deadline = time.monotonic() + timeout_seconds
    while window.isVisible() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    if window.isVisible():
        raise RuntimeError("TaskConsole did not finish its bounded shutdown")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore

    from Zou_lab_control.notebook import connect
    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_storage import durable_makedirs
    from zlc_workbench.task_console.app import open_task_console

    application = ensure_qt_app()
    experiment = None
    window = None
    try:
        # The composition root owns one repository below an existing
        # parent, so this launcher owns the workspace levels above it:
        # a first run on a machine with no ~/.zlc must not die.
        durable_makedirs(args.repository.expanduser().parent)
        experiment = connect(
            "virtual",
            repository=args.repository,
            name=args.name,
            seed=args.seed,
        )
        window = open_task_console(experiment, state=args.state,
                                   task=args.task)
        auto_close_ms = os.environ.get("ZLC_TASK_CONSOLE_AUTO_CLOSE_MS")
        if auto_close_ms:
            QtCore.QTimer.singleShot(
                max(0, int(auto_close_ms)),
                window.window().close,
            )
        return int(application.exec_())
    finally:
        try:
            _close_window(application, window)
        finally:
            if experiment is not None:
                experiment.close()


if __name__ == "__main__":
    raise SystemExit(main())
