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
        help="Open with a saved console layout (a tasks/<name>.json path).",
    )
    parser.add_argument(
        "--task",
        help="Open with a saved console layout BY NAME from the workspace's tasks/.",
    )
    parser.add_argument(
        "--config",
        default="virtual",
        help=(
            "Installation config shown by DeviceManager before TaskConsole "
            "starts: 'virtual' or a saved config JSON path."
        ),
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.home() / ".zlc" / "task_console",
        help="Experiment workspace used after DeviceManager initializes devices.",
    )
    parser.add_argument(
        "--name",
        default="task_console",
        help="Experiment name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Virtual installation seed (used only when --config=virtual).",
    )
    return parser


def _close_window(application, window, *, timeout_seconds: float = 10.0) -> None:
    """Close the console's TOP-LEVEL window and wait for it to actually go.

    ``open_task_console`` returns the console BODY, not the frame around it (the body's
    ``isWindow()`` is False; its ``window()`` is the Fluent frame).  Closing the body leaves
    the frame open, the app keeps running, and the launcher hangs forever in ``exec_()`` --
    which is exactly what happened the first time this entry was switched.
    """

    if window is None:
        return
    window = window.window()
    if not window.isVisible():
        return
    window.close()
    deadline = time.monotonic() + timeout_seconds
    while window.isVisible() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    if window.isVisible():
        raise RuntimeError("TaskConsole did not finish its bounded shutdown")


class _StandaloneTaskConsoleFlow:
    """Own one DeviceManager-created Experiment, TaskConsole, and Pulse GUI.

    The standalone GUI has no hidden process registry and does not manufacture
    a second Experiment.  DeviceManager owns installation composition; after
    its successful Init transition the exact same Experiment is borrowed by
    both application windows.  Closing TaskConsole first closes its sibling
    Pulse GUI, then retires DeviceManager authority and that one Experiment.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.devices = None
        self.closing_devices = None
        self.console = None
        self.pulse = None
        self.experiment = None
        self.failure: BaseException | None = None

    def open(self):
        from Zou_lab_control.notebook import device_manager

        kwargs = {
            "repository": self.args.repository,
            "name": self.args.name,
        }
        if self.args.config == "virtual":
            kwargs["seed"] = self.args.seed
        self.devices = device_manager(
            self.args.config,
            on_initialized=self._open_console,
            **kwargs,
        )
        return self.devices

    def _open_console(self, experiment) -> None:
        if self.console is not None:
            return
        if self.devices is None:
            return
        console = None
        try:
            console = experiment.task_console(
                state=self.args.state,
                task=self.args.task,
            )
            pulse = experiment.pulse_gui()
        except BaseException as error:
            if console is not None:
                console.window().close()
            self.failure = error
            self.devices.status_strip.show_message(
                f"{type(error).__name__}: {error}",
                severity="error",
            )
            return
        self.experiment = experiment
        self.console = console
        self.pulse = pulse
        console.window().closed.connect(self.close)
        self.devices.window().hide()

    def close(self) -> None:
        pulse, self.pulse = self.pulse, None
        if pulse is not None:
            pulse_window = pulse.window()
            if pulse_window.isVisible():
                pulse_window.close()
        self.experiment = None
        devices, self.devices = self.devices, None
        if devices is not None:
            self.closing_devices = devices
            devices.request_owner_close()

    def finish_close(
        self,
        application,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Wait for DeviceManager's asynchronous installation shutdown."""

        self.close()
        devices = self.closing_devices
        if devices is None:
            return
        deadline = time.monotonic() + float(timeout_seconds)
        while not devices.permanently_closed and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.005)
        if not devices.permanently_closed:
            raise RuntimeError(
                "DeviceManager did not finish its bounded installation shutdown"
            )
        self.closing_devices = None


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

    from PyQt5 import QtCore

    from zlc_frontend.qt_widgets import ensure_qt_app

    application = ensure_qt_app()
    flow = _StandaloneTaskConsoleFlow(args)
    try:
        flow.open()
        auto_close_ms = os.environ.get("ZLC_TASK_CONSOLE_AUTO_CLOSE_MS")
        if auto_close_ms:
            QtCore.QTimer.singleShot(
                max(0, int(auto_close_ms)),
                application.quit,
            )
        status = int(application.exec_())
        if flow.failure is not None:
            raise flow.failure
        return status
    finally:
        try:
            _close_window(application, flow.console)
        finally:
            flow.finish_close(application)


if __name__ == "__main__":
    raise SystemExit(main())
