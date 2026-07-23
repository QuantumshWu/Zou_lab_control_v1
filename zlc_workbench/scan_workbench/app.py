"""Qt composition entry point for the pulse-scan Workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Zou_lab_control.notebook.facade import (
        Experiment,
        OccupancyScanRequest,
        ScanRequest,
    )

    from .window import ScanWorkbenchWindow


def open_scan_workbench(
    experiment: Experiment,
    request: ScanRequest | OccupancyScanRequest,
) -> ScanWorkbenchWindow:
    from PyQt5 import QtCore

    from zlc_frontend.qt_widgets import (
        WINDOW_SCREEN_FRACTION,
        center_window_on_primary_screen,
        ensure_qt_app,
        retain_window,
        screen_fit_window_size,
        set_fluent_scale,
    )

    from .window import ScanWorkbenchWindow

    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("scan Workbench must be opened on the Qt GUI thread")
    set_fluent_scale(None)
    window = ScanWorkbenchWindow(experiment, request)
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window
