"""Standalone launcher for the current installation config manager."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zou-lab device manager.")
    parser.add_argument(
        "config",
        nargs="?",
        default="virtual",
        help="current installation config JSON, or 'virtual'",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=None,
        help="runtime workspace (default: ~/.zlc/device-manager)",
    )
    parser.add_argument("--name", default="device-manager")
    args = parser.parse_args(argv)
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore

    from Zou_lab_control.notebook import device_manager
    from zlc_frontend.qt_widgets import ensure_qt_app

    application = ensure_qt_app()
    body = device_manager(
        args.config,
        repository=args.repository,
        name=args.name,
    )
    auto_close_ms = os.environ.get("ZLC_DEVICE_MANAGER_AUTO_CLOSE_MS")
    if auto_close_ms:
        QtCore.QTimer.singleShot(
            max(0, int(auto_close_ms)),
            body.window().close,
        )
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
