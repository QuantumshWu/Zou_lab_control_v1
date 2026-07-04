"""Standalone launcher for the saved-figure viewer.

Double-click ``figure_viewer.bat`` (or run ``python figure_viewer.py``) to open the
window that RE-OPENS a saved figure: load its ``.npz`` (the data + the full snapshot
``info`` a Save wrote), see exactly what was stored, re-plot the SAME data with a
DIFFERENT plotter, re-edit (relim / fit / limits) and re-save.  It is
session-INDEPENDENT -- it only reads files, so it needs no connected experiment::

    python figure_viewer.py                          # open empty; Browse to a .npz
    python figure_viewer.py path\\to\\snap.npz        # open that saved figure
    python figure_viewer.py path\\to\\run_folder      # open the folder as a gallery

The window is the SAME as ``exp.figure_viewer(path)`` from a notebook / a running
session; this launcher is just the double-click entry that needs no session.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zou-lab saved-figure viewer.")
    parser.add_argument("path", nargs="?", default=None,
                        help="a saved figure .npz (or a folder of saved .npz files) to open on "
                             "start; omit to open empty and Browse to one.")
    parser.add_argument("--scale", type=float, default=None, help="UI scale factor (default: auto).")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore

    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend import show_figure_viewer

    app = ensure_qt_app()
    show_figure_viewer(path=args.path, scale=args.scale)

    auto_close_ms = os.environ.get("ZLC_FIGURE_VIEWER_AUTO_CLOSE_MS")
    if auto_close_ms:
        QtCore.QTimer.singleShot(int(auto_close_ms), app.quit)
    app.exec_()
    return 0


if __name__ == "__main__":
    sys.exit(main())
