"""Standalone launcher for the task console (live experiment dashboard).

By default it connects the VIRTUAL atom-loading experiment (the SAME ``connect()``
contract as real hardware -- only the camera frames are simulated), self-calibrates
the site map + thresholds, and opens the console EMPTY and STOPPED.  You build the
dashboard yourself: add logic nodes (camera / measurement / processor / task) on the
Logic tab and Start them, then add Plot panels wired to the signals they publish.
There is NO auto-built, auto-started readout -- the hub is empty until you Start a node.

Run on REAL hardware with one command -- ``--config`` connects the devices from a
device config instead of the virtual session::

    python task_console.py --config remote_template.json --grid 5x7

That opens the camera + remote sequencer through the SAME ``connect()`` contract the
virtual path uses (only the data source differs), and wires the auto-discovered
catalogs (every ``@measurement`` / ``@processor`` / task in ``operations/`` +
anything registered) into the Add-Panel list, so you can add any of them as a logic
node.  First-light site/threshold CALIBRATION still belongs in a notebook
(``exp.readout.sitemap(display=True)`` / ``exp.readout.thresholds(display=True)``,
where you eyeball the loading image and the count histograms); the ``--config`` path
self-calibrates through that same contract for routine running once you trust the setup.

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
                        help="a task layout YOU saved (tasks/<name>.json) or a JSON path "
                             "(there are no built-in presets; the console opens empty).")
    parser.add_argument("--config", type=str, default=None,
                        help="connect a REAL experiment from this device config (a config file path, e.g."
                             " remote_template.json, or a named backend) instead of the virtual session."
                             "  First-light site/threshold calibration still belongs in a notebook"
                             " (exp.readout.sitemap/thresholds with display=True); this path self-calibrates"
                             " through the SAME contract.")
    parser.add_argument("--grid", type=str, default="5x7",
                        help="atom grid as ROWSxCOLS for the session site map (default 5x7).")
    parser.add_argument("--scale", type=float, default=None, help="UI scale factor (default: auto).")
    parser.add_argument("--no-connect", action="store_true",
                        help="open the console without connecting any session (wire a hub yourself).")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore

    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        TaskConsoleState, default_console_state, resolve_task_state, show_task_console)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    app = ensure_qt_app()
    hub = SignalHub()
    measurements = ()
    processors = ()
    tasks = ()
    exp = None
    # Opening a device session lives in try/finally: a calibration / show failure must
    # STILL close the session (a real camera + remote FPGA sequencer must reach a clean
    # safe_state, not leak open), then re-raise so the operator sees the error.
    try:
        if not args.no_connect:
            # Connect a session (real devices from --config, else a virtual one that
            # only simulates camera frames -- SAME connect() contract).  Self-calibrate
            # the site map + thresholds so the catalogs (measurements / processors /
            # tasks) can run.  Then open the console EMPTY + STOPPED: the user adds
            # logic nodes (camera / measurement / processor / task) to the Logic tab
            # and Starts them, then adds Plot panels wired to the signals they publish.
            # There is NO auto-built / auto-started readout -- the hub opens empty
            # (the decoupled VIEW / LOGIC model).
            import Zou_lab_control.neutral_atom as na

            grid = _parse_grid(args.grid)
            if args.config:
                print(f"Connecting devices from {args.config!r} (grid {grid[0]}x{grid[1]})...")
                exp = na.connect(args.config, open_devices=True)
            else:
                print(f"Starting VIRTUAL experiment (grid {grid[0]}x{grid[1]}; only camera frames simulated)...")
                # Real-time sequence timing (sleep_scale=1.0) so a fired calibration /
                # temperature scan takes a BELIEVABLE wall-clock (each frame ~ its real
                # exposure) and its mid-run panel is actually watchable -- instead of the
                # CPU finishing the render instantly.  Tests build the virtual backend
                # with the default sleep_scale=0, so they stay fast.
                exp = na.connect("virtual", sitemap={"grid_shape": grid}, sleep_scale=1.0)
            exp.readout.sitemap(method="box", frames=4, display=False)
            exp.readout.thresholds(frames=24, display=False)
            measurements = exp.readout.measurement_specs()
            processors = exp.readout.processor_specs()
            tasks = exp.readout.task_specs()

        if args.state:
            state = TaskConsoleState.load(args.state)
        elif args.task:
            state = resolve_task_state(args.task)
        else:
            state = default_console_state()
        show_task_console(hub=hub, state=state, measurements=measurements,
                          processors=processors, tasks=tasks, session=exp, scale=args.scale)

        auto_close_ms = os.environ.get("ZLC_TASK_CONSOLE_AUTO_CLOSE_MS")
        if auto_close_ms:
            QtCore.QTimer.singleShot(int(auto_close_ms), app.quit)
        app.exec_()
    finally:
        if exp is not None:
            exp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
