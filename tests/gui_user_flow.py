"""Shared offscreen fast path for every GUI debug/acceptance flow.

Application-specific flows supply only their real input sequence.  This owner
selects the offscreen Qt backend *before* ``ensure_qt_app`` is called, supplies
event waiting and whole-window capture, and never changes product sizing,
styling, or display scale.  The slow path launches the same product entry on the
desktop and is deliberately kept outside this helper.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets


_FAST_PLATFORM = "offscreen"


def configure_offscreen_fast_path() -> None:
    """Select offscreen Qt before the sole QApplication owner is invoked.

    This function owns only the platform choice.  High-DPI attributes and
    QApplication construction remain exclusively owned by ``ensure_qt_app``.
    Calling it after a non-offscreen application already exists is an error,
    because that process can no longer represent the fast-path contract.
    """

    application = QtWidgets.QApplication.instance()
    if application is not None:
        require_offscreen_platform(application)
        return
    configured = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if configured not in {"", _FAST_PLATFORM}:
        raise RuntimeError(
            "GUI fast path requires QT_QPA_PLATFORM=offscreen before "
            "ensure_qt_app()"
        )
    os.environ["QT_QPA_PLATFORM"] = _FAST_PLATFORM


def until(
    application: QtWidgets.QApplication,
    predicate,
    *,
    timeout: float = 10.0,
) -> None:
    deadline = time.monotonic() + float(timeout)
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    if not predicate():
        raise AssertionError("GUI user-flow condition did not become true")


def click_tab(body, page) -> None:
    """Select a visible product tab through its real tab-bar hit target."""

    index = body.tabs.indexOf(page)
    if index < 0:
        raise ValueError("page does not belong to the product tab widget")
    bar = body.tabs.tabBar()
    QtTest.QTest.mouseClick(
        bar,
        QtCore.Qt.LeftButton,
        pos=bar.tabRect(index).center(),
    )


def drag_mouse_move(widget, position, button) -> None:
    """Deliver one real Qt mouse-move while ``button`` remains held."""

    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(position),
        QtCore.Qt.NoButton,
        button,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)


def require_offscreen_platform(
    application: QtWidgets.QApplication,
) -> str:
    """Return the Qt platform name or reject a non-fast-path application."""

    platform = application.platformName().lower()
    if platform != _FAST_PLATFORM:
        raise RuntimeError(
            "GUI fast path requires the offscreen Qt backend; use the formal "
            "desktop launcher for the slow human-flow path"
        )
    return platform


def _settle(application: QtWidgets.QApplication, settle_ms: int) -> None:
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    QtTest.QTest.qWait(max(800, int(settle_ms)))
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def _geometry(rect: QtCore.QRect) -> dict[str, int]:
    return {
        "x": rect.x(),
        "y": rect.y(),
        "width": rect.width(),
        "height": rect.height(),
    }


def capture_offscreen_window(
    application: QtWidgets.QApplication,
    body,
    output: str | Path,
    *,
    settle_ms: int = 1000,
) -> dict[str, object]:
    """Capture the untouched formal outer window through offscreen Qt."""

    platform = require_offscreen_platform(application)
    _settle(application, settle_ms)
    wrapper = body.window()
    if wrapper is None or not wrapper.isVisible():
        raise RuntimeError("formal GUI wrapper is not visible")
    path = Path(output).expanduser().resolve()
    pixmap = wrapper.grab()
    path.parent.mkdir(parents=True, exist_ok=True)
    if pixmap.isNull() or not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"could not save GUI screenshot to {path}")
    screen = wrapper.screen() or application.primaryScreen()
    return {
        "path": str(path),
        "qt_platform": platform,
        "window_frame": _geometry(wrapper.frameGeometry()),
        "window_client": _geometry(wrapper.geometry()),
        "image_pixels": {"width": pixmap.width(), "height": pixmap.height()},
        "device_pixel_ratio": float(pixmap.devicePixelRatio()),
        "screen": (
            None
            if screen is None
            else {
                "name": screen.name(),
                "available_geometry": _geometry(screen.availableGeometry()),
                "logical_dpi": [
                    screen.logicalDotsPerInchX(),
                    screen.logicalDotsPerInchY(),
                ],
                "physical_dpi": [
                    screen.physicalDotsPerInchX(),
                    screen.physicalDotsPerInchY(),
                ],
                "device_pixel_ratio": float(screen.devicePixelRatio()),
            }
        ),
    }


__all__ = [
    "capture_offscreen_window",
    "click_tab",
    "configure_offscreen_fast_path",
    "drag_mouse_move",
    "require_offscreen_platform",
    "until",
]
