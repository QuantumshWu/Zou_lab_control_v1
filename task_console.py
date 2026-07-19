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
        help="Load one current TaskConsole scan-intent JSON file, stopped.",
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
    if window is None or not window.isVisible():
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
    from zlc_workbench.task_console import load_task_console_scan_intent

    application = ensure_qt_app()
    initial_intent = (
        None
        if args.state is None
        else load_task_console_scan_intent(args.state)
    )
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
        window = experiment.task_console(initial_intent)
        auto_close_ms = os.environ.get("ZLC_TASK_CONSOLE_AUTO_CLOSE_MS")
        if auto_close_ms:
            QtCore.QTimer.singleShot(
                max(0, int(auto_close_ms)),
                window.close,
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
