"""Standalone launcher for the current PulseWorkbench product surface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the current Zou_lab_control PulseWorkbench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Open one current zlc_pulse.PulseDocument JSON file.",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.home() / ".zlc" / "pulse_gui",
        help="Virtual Experiment workspace. Ignored in offline mode.",
    )
    parser.add_argument("--name", default="pulse_gui", help="Virtual Experiment name.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Author and preview only; no fake execution backend is constructed.",
    )
    parser.add_argument(
        "--clock-hz",
        type=_positive_float,
        default=50_000_000.0,
        help="Clock for a new offline document. Loaded documents retain their own grid.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore, QtWidgets

    from Zou_lab_control.workbench import (
        open_offline_pulse_workbench,
        open_pulse_workbench,
    )

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    experiment = None
    if args.offline:
        from zlc_pulse import load_deployed_pulse_target

        window = open_offline_pulse_workbench(
            load_deployed_pulse_target(),
            time_step_ns=1e9 / args.clock_hz,
            path=args.state,
        )
    else:
        from Zou_lab_control.notebook import connect

        experiment = connect(
            "virtual",
            repository=args.repository,
            name=args.name,
        )
        window = open_pulse_workbench(experiment, path=args.state)

    auto_close_ms = os.environ.get("ZLC_PULSE_GUI_AUTO_CLOSE_MS")
    if auto_close_ms:
        QtCore.QTimer.singleShot(
            max(0, int(auto_close_ms)),
            lambda: window.request_close(discard_unsaved=True),
        )
    try:
        return int(application.exec_())
    finally:
        if experiment is not None:
            experiment.close()


if __name__ == "__main__":
    raise SystemExit(main())
