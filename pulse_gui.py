"""Standalone current PulseDocument editor and remote-pulse launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the Zou_lab_control PulseDocument editor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--document",
        type=Path,
        help="Load a current PulseDocument JSON.",
    )
    parser.add_argument(
        "--remote-host",
        help="Connect to the current pulse server after the window opens.",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=18861,
        help="Remote pulse server port.",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        help=(
            "Standalone runtime workspace; defaults to ZLC_PULSE_WORKSPACE or "
            "~/.zlc/pulse-workbench."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore

    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.pulse_editor.app import open_pulse_editor

    application = ensure_qt_app()
    remote_endpoint = None
    if args.remote_host:
        host = str(args.remote_host)
        remote_endpoint = (
            f"[{host}]:{args.remote_port}" if ":" in host else f"{host}:{args.remote_port}"
        )
    editor = open_pulse_editor(
        path=args.document,
        remote_endpoint=remote_endpoint,
        repository=args.repository,
    )
    auto_close_ms = os.environ.get("ZLC_PULSE_GUI_AUTO_CLOSE_MS")
    if auto_close_ms:
        QtCore.QTimer.singleShot(
            max(0, int(auto_close_ms)),
            lambda: editor.request_close(discard_unsaved=True),
        )
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
