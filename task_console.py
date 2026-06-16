"""Standalone launcher for the task console (live experiment dashboard).

By default it runs against the VIRTUAL atom-loading experiment (one primary feed
plus a ``b_``-prefixed secondary feed so cross-signal expressions like
``value = rate_grid - b_rate_grid`` have data).

Run on REAL hardware with one command -- ``--config`` connects the devices from
a device config and streams live atom-loading shots into the console::

    python task_console.py --config remote_template.json --grid 5x7

That opens the camera + remote sequencer through the SAME composed loading
readout (``build_loading_readout``) the virtual path uses (only the data source
differs), and wires the
auto-discovered readout measurement catalog (every ``@measurement`` in
``operations/measurements/`` + anything ``register_measurement``-ed) into the
Add-Panel list, so you can pick any available measurement to connect.
First-light site/threshold
CALIBRATION still belongs in a notebook (``exp.readout.sitemap(display=True)`` /
``exp.readout.thresholds(display=True)``, where you eyeball the loading image and
the count histograms); the ``--config`` path self-calibrates through that same
contract for routine running once you trust the setup.

Or wire a real experiment from a notebook for full control::

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


def _parse_grid(text: str) -> tuple[int, int]:
    """Parse ``ROWSxCOLS`` (e.g. ``5x7``) into a (rows, cols) tuple."""

    parts = str(text).lower().replace(" ", "").split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise SystemExit(f"--grid must be ROWSxCOLS (e.g. 5x7), got {text!r}.")
    rows, cols = int(parts[0]), int(parts[1])
    if rows < 1 or cols < 1:
        raise SystemExit(f"--grid dimensions must be >= 1, got {text!r}.")
    return rows, cols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zou-lab task console (live dashboard).")
    parser.add_argument("--state", type=str, default=None, help="task layout JSON to load on start.")
    parser.add_argument("--task", type=str, default=None,
                        help="named task dashboard: a built-in (atom_loading_monitor, loading_rate_live),"
                             " a layout saved in tasks/, or a JSON path.")
    parser.add_argument("--config", type=str, default=None,
                        help="connect a REAL experiment from this device config (a config file path, e.g."
                             " remote_template.json, or a named backend) and stream live atom-loading shots"
                             " into the console instead of the virtual feed.  First-light site/threshold"
                             " calibration still belongs in a notebook (exp.readout.sitemap/thresholds with"
                             " display=True); this path self-calibrates through the SAME contract.")
    parser.add_argument("--grid", type=str, default="5x7",
                        help="atom grid as ROWSxCOLS for the --config loading feed (default 5x7).")
    parser.add_argument("--scale", type=float, default=None, help="UI scale factor (default: auto).")
    parser.add_argument("--rate", type=float, default=4.0, help="feed rate in shots/s (default 4).")
    parser.add_argument("--seed", type=int, default=None, help="virtual feed seed (default: random).")
    parser.add_argument("--single", action="store_true",
                        help="start only the primary virtual feed (no b_* signals).")
    parser.add_argument("--no-feed", action="store_true",
                        help="open the console without any feed (wire a hub yourself).")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore, QtWidgets

    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        TaskConsoleState, default_console_state, resolve_task_state, show_task_console)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    app = ensure_qt_app()
    hub = SignalHub()
    feeds = []
    measurements = ()
    processors = ()
    tasks = ()
    exp = None
    # Everything that may open a device session lives in try/finally: a
    # calibration / readout / show failure must STILL stop the producers and
    # close the session (a real camera + remote FPGA sequencer must reach a clean
    # safe_state, not leak open), then re-raise so the operator sees the error.
    try:
        if args.config:
            # REAL hardware (or any device config): one-click direct run.  The loading
            # readout is COMPOSED (calibrate task + camera measurement + detect
            # processor), touching ONLY the camera.acquire contract -- same nodes the
            # virtual path uses, only the data source differs.
            import Zou_lab_control.neutral_atom as na
            from Zou_lab_control.neutral_atom.operations.feeds import build_loading_readout

            grid = _parse_grid(args.grid)
            print(f"Connecting devices from {args.config!r} (grid {grid[0]}x{grid[1]}); "
                  "self-calibrating site map + thresholds through the readout contract...")
            exp = na.connect(args.config, open_devices=True)
            readout = build_loading_readout(hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=grid)
            feeds.extend([readout.calibrate_task, readout.camera, readout.detect])
            measurements = exp.readout.measurement_specs()
            processors = exp.readout.processor_specs()
            tasks = exp.readout.task_specs()
            readout.start(rate_hz=args.rate)   # non-blocking: calibrate on its own thread, then stream
        elif not args.no_feed:
            # VIRTUAL experiment through the SAME connect() contract the real path uses
            # (only the camera frames are simulated): the primary loading readout is the
            # IDENTICAL composed chain as --config, and the readout measurement catalog
            # is wired in.  An optional b_* readout adds a second signal source for
            # cross-signal expressions (e.g. ``value = rate_grid - b_rate_grid``).
            import Zou_lab_control.neutral_atom as na
            from Zou_lab_control.neutral_atom.devices.virtual import virtual_loading_readout
            from Zou_lab_control.neutral_atom.operations.feeds import build_loading_readout

            grid = _parse_grid(args.grid)
            print(f"Starting VIRTUAL experiment (grid {grid[0]}x{grid[1]}; only camera frames are simulated); "
                  "calibrating site map + thresholds so measurements can run...")
            exp = na.connect("virtual", sitemap={"grid_shape": grid})
            exp.readout.sitemap(method="box", frames=4, display=False)
            exp.readout.thresholds(frames=24, display=False)
            readouts = [build_loading_readout(hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=grid)]
            if not args.single:
                seed_b = None if args.seed is None else args.seed + 11
                readouts.append(virtual_loading_readout(hub, prefix="b_", seed=seed_b,
                                                        loading_probability=0.35, grid_shape=grid))
            for readout in readouts:
                feeds.extend([readout.calibrate_task, readout.camera, readout.detect])
            measurements = exp.readout.measurement_specs()
            processors = exp.readout.processor_specs()
            tasks = exp.readout.task_specs()
            for readout in readouts:
                readout.start(rate_hz=args.rate)

        if args.state:
            state = TaskConsoleState.load(args.state)
        elif args.task:
            state = resolve_task_state(args.task)
        else:
            state = default_console_state()
        show_task_console(hub=hub, state=state, feeds=feeds, measurements=measurements,
                          processors=processors, tasks=tasks, session=exp, scale=args.scale)

        auto_close_ms = os.environ.get("ZLC_TASK_CONSOLE_AUTO_CLOSE_MS")
        if auto_close_ms:
            QtCore.QTimer.singleShot(int(auto_close_ms), app.quit)
        app.exec_()
    finally:
        for feed in feeds:
            feed.stop()
        if exp is not None:
            exp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
