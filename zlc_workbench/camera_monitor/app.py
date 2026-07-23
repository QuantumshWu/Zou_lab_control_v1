"""Composition root for the camera-monitor application."""

from __future__ import annotations

from PyQt5 import QtCore

from zlc_frontend.qt_widgets import (
    WINDOW_SCREEN_FRACTION,
    center_window_on_primary_screen,
    ensure_qt_app,
    retain_window,
    screen_fit_window_size,
    set_fluent_scale,
)
from zlc_neutral_atom.monitor_application import CameraMonitorRequest

from .window import CameraMonitorWorkbenchWindow


def open_camera_monitor_workbench(
    prepare,
    request: CameraMonitorRequest,
) -> CameraMonitorWorkbenchWindow:
    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("camera monitor Workbench must open on the Qt GUI thread")
    set_fluent_scale(None)
    window = CameraMonitorWorkbenchWindow(prepare, request)
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


__all__ = ["open_camera_monitor_workbench"]
