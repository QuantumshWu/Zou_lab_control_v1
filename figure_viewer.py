"""Standalone launcher for the saved-figure viewer.

Double-click ``figure_viewer.bat`` (or run ``python figure_viewer.py``) to open the
viewer window: the Info column and the board, with no session and no hardware.

With no path the window opens an empty, session-independent browser.  Passing a
current ``.npz`` archive opens it directly; passing its PNG/JPEG companion resolves
the same-stem ``.npz``.  The archive retains the typed source, figure document,
validity, fit overlays, and display state rather than reconstructing semantics from
image pixels.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zou-lab saved-figure viewer.")
    parser.add_argument("path", nargs="?", default=None,
                        help="current Figure .npz, or a PNG/JPEG with a same-stem .npz")
    parser.add_argument("--scale", type=float, default=None, help="UI scale factor (default: auto).")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore

    from zlc_frontend.qt_widgets import ensure_qt_app
    from zlc_workbench.figure_viewer.app import open_figure_viewer

    app = ensure_qt_app()
    open_figure_viewer(path=args.path, scale=args.scale)

    auto_close_ms = os.environ.get("ZLC_FIGURE_VIEWER_AUTO_CLOSE_MS")
    if auto_close_ms:
        QtCore.QTimer.singleShot(int(auto_close_ms), app.quit)
    app.exec_()
    return 0


if __name__ == "__main__":
    sys.exit(main())
