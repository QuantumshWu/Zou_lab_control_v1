"""Standalone launcher for the task console (live experiment dashboard).

By default it runs against the VIRTUAL atom-loading experiment (one primary feed
plus a ``b_``-prefixed secondary feed so cross-signal expressions like
``value = rate_grid - b_rate_grid`` have data).  Wire a real experiment from a
notebook instead with::

    from Zou_lab_control import frontend as zf
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    hub = SignalHub()
    console = zf.show_task_console(hub=hub)
    hub.publish({"frame": image, "counts": counts, ...})   # once per shot
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zou-lab task console (live dashboard).")
    parser.add_argument("--state", type=str, default=None, help="task layout JSON to load on start.")
    parser.add_argument("--task", type=str, default=None,
                        help="named task dashboard: a built-in (atom_loading_monitor, loading_rate_live),"
                             " a layout saved in tasks/, or a JSON path.")
    parser.add_argument("--scale", type=float, default=None, help="UI scale factor (default: auto).")
    parser.add_argument("--rate", type=float, default=4.0, help="virtual feed rate in shots/s (default 4).")
    parser.add_argument("--seed", type=int, default=None, help="virtual feed seed (default: random).")
    parser.add_argument("--single", action="store_true",
                        help="start only the primary virtual feed (no b_* signals).")
    parser.add_argument("--no-feed", action="store_true",
                        help="open the console without any virtual feed (wire a hub yourself).")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore, QtWidgets

    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        TaskConsoleState, default_console_state, resolve_task_state, show_task_console)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.feeds import VirtualLoadingFeed

    app = ensure_qt_app()
    hub = SignalHub()
    feeds = []
    if not args.no_feed:
        print("Starting virtual atom-loading feed (calibrating site map + thresholds)...")
        feeds.append(VirtualLoadingFeed(hub, seed=args.seed))
        if not args.single:
            seed_b = None if args.seed is None else args.seed + 11
            feeds.append(VirtualLoadingFeed(hub, prefix="b_", seed=seed_b, loading_probability=0.35))
        for feed in feeds:
            feed.start(rate_hz=args.rate)

    if args.state:
        state = TaskConsoleState.load(args.state)
    elif args.task:
        state = resolve_task_state(args.task)
    else:
        state = default_console_state()
    show_task_console(hub=hub, state=state, feeds=feeds, scale=args.scale)

    auto_close_ms = os.environ.get("ZLC_TASK_CONSOLE_AUTO_CLOSE_MS")
    if auto_close_ms:
        QtCore.QTimer.singleShot(int(auto_close_ms), app.quit)
    app.exec_()
    for feed in feeds:
        feed.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
