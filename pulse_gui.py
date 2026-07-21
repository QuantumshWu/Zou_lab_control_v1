"""Standalone launcher for the pulse-sequence editor (double-click ``pulse_gui.bat``).

Opens the OFFLINE editor -- no device is created or discovered.  Executable use on virtual
or real hardware enters through a session (``exp.pulse_gui()`` in a notebook), exactly as
before the migration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the Zou_lab_control pulse-sequence editor (offline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Load a saved pulse program (a PulseTableState JSON).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Display scale override (default: screen-derived).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore

    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    application = ensure_qt_app()
    editor = open_pulse_editor(state=args.state, scale=args.scale)
    auto_close_ms = os.environ.get("ZLC_PULSE_GUI_AUTO_CLOSE_MS")
    if auto_close_ms:
        # The editor is the window BODY; its Fluent frame is the top-level window, and
        # closing the body alone leaves the frame open and the launcher hung in exec_()
        # (the exact task_console lesson).
        QtCore.QTimer.singleShot(max(0, int(auto_close_ms)), editor.window().close)
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
