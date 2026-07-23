"""Composition root for the finite-capture application."""

from __future__ import annotations

from PyQt5 import QtCore

from Zou_lab_control.notebook.facade import Experiment, _prepare_capture_for_workbench
from zlc_frontend.qt_widgets import (
    WINDOW_SCREEN_FRACTION,
    center_window_on_primary_screen,
    ensure_qt_app,
    retain_window,
    screen_fit_window_size,
    set_fluent_scale,
)
from zlc_neutral_atom.capture_application import CaptureRequest

from .window import CaptureWorkbenchWindow


def open_capture_workbench(
    experiment: Experiment,
    request: CaptureRequest,
) -> CaptureWorkbenchWindow:
    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("capture Workbench must be opened on the Qt GUI thread")
    set_fluent_scale(None)
    window = CaptureWorkbenchWindow(
        lambda: _prepare_capture_for_workbench(experiment, request),
        request,
    )
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


__all__ = ["open_capture_workbench"]
