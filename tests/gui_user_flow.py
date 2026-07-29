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

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets, sip


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


def widget_gone(widget: QtWidgets.QWidget | None) -> bool:
    """Accept either hidden or QObject-deleted as a completed window close."""

    return widget is None or sip.isdeleted(widget) or not widget.isVisible()


def close_pulse_editor(
    application: QtWidgets.QApplication,
    body,
    *,
    timeout: float = 10.0,
) -> None:
    """Close a Pulse editor through its lifecycle owner, discarding test edits."""

    wrapper = body.window()
    body.request_close(discard_unsaved=True)
    until(
        application,
        lambda: body.permanently_closed and widget_gone(wrapper),
        timeout=timeout,
    )
    # ``deleteLater`` is the product's safe post-close QObject boundary.  The
    # desktop event loop naturally consumes it; a short-lived offscreen process
    # must do the same before Python/SIP interpreter teardown begins.
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def close_task_console(
    application: QtWidgets.QApplication,
    body,
    *,
    timeout: float = 10.0,
) -> None:
    """Close TaskConsole through its non-blocking product lifecycle."""

    if getattr(body, "_window", None) is None:
        deadline = time.monotonic() + float(timeout)
        while not body.shutdown() and time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            time.sleep(0.005)
        if getattr(body, "_shutdown_state", None) != "TERMINATED":
            raise AssertionError(
                "direct TaskConsole body did not terminate: "
                f"state={getattr(body, '_shutdown_state', None)!r}"
            )
        body.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        return

    body.request_window_close()
    deadline = time.monotonic() + float(timeout)
    while not body.permanently_closed and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    if not body.permanently_closed:
        raise AssertionError(
            "TaskConsole did not close: "
            f"state={getattr(body, '_shutdown_state', None)!r}, "
            f"window_visible={body.isVisible()!r}"
        )


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


def choose_combo_data(
    combo: QtWidgets.QComboBox,
    value: object,
    application: QtWidgets.QApplication,
) -> None:
    """Choose a typed combo entry through its visible popup and keyboard."""

    row = combo.findData(value)
    if row < 0:
        row = next(
            (
                index
                for index in range(combo.count())
                if combo.itemData(index) == value
            ),
            -1,
        )
    assert row >= 0, (
        value,
        [combo.itemData(index) for index in range(combo.count())],
    )
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    view = combo.view()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _ in range(row):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    assert combo.currentData() == value


def choose_combo_text(
    combo: QtWidgets.QComboBox,
    value: str,
    application: QtWidgets.QApplication,
) -> None:
    """Choose one visible combo label without QVariant coercion."""

    row = combo.findText(value)
    assert row >= 0, (
        value,
        [combo.itemText(index) for index in range(combo.count())],
    )
    QtTest.QTest.mouseClick(combo, QtCore.Qt.LeftButton)
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)
    view = combo.view()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Home)
    for _ in range(row):
        QtTest.QTest.keyClick(view, QtCore.Qt.Key_Down)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Return)
    assert combo.currentText() == value


def current_logic_editor(body, application: QtWidgets.QApplication):
    """Resolve the Logic Edit page reached through the visible Add/Edit flow."""

    from zlc_workbench.task_console.logic_node_editor import LogicNodeEditor

    until(
        application,
        lambda: isinstance(body.tabs.currentWidget(), LogicNodeEditor),
    )
    editor = body.tabs.currentWidget()
    assert editor.isVisible() and not editor.isWindow()
    return editor


def visible_form_widgets(editor) -> dict[str, QtWidgets.QWidget]:
    """Resolve a visible generated form through its public form API."""

    candidates = tuple(
        widget
        for widget in editor.form.findChildren(QtWidgets.QWidget)
        if (
            widget.parentWidget() is editor.form
            and widget.isVisible()
            and hasattr(widget, "spec")
            and callable(getattr(widget, "widget_for", None))
        )
    )
    assert len(candidates) == 1, candidates
    form = candidates[0]
    return {key: form.widget_for(key) for key in form.spec.keys}


def replace_spin_value(spin: QtWidgets.QWidget, text: str) -> None:
    """Edit a visible numeric form control exactly as an operator would."""

    edit = spin.lineEdit() if hasattr(spin, "lineEdit") else spin
    QtTest.QTest.mouseClick(edit, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(edit, str(text))
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_Return)


def replace_path_value(path_widget: QtWidgets.QWidget, text: str) -> None:
    """Edit the text field of the shared visible path control."""

    edit = path_widget.edit
    QtTest.QTest.mouseClick(edit, QtCore.Qt.LeftButton)
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
    QtTest.QTest.keyClicks(edit, str(text))
    QtTest.QTest.keyClick(edit, QtCore.Qt.Key_Return)


def point_in_rect(
    rect: QtCore.QRect | QtCore.QRectF,
    x_fraction: float,
    y_fraction: float,
) -> QtCore.QPoint:
    """Resolve a fractional hit target in one painted Qt rectangle."""

    return QtCore.QPoint(
        int(round(rect.left() + float(x_fraction) * rect.width())),
        int(round(rect.top() + float(y_fraction) * rect.height())),
    )


def normalized_subrect(
    rect: QtCore.QRect | QtCore.QRectF,
    bounds: tuple[float, float, float, float],
) -> QtCore.QRectF:
    """Map top-origin normalized bounds into one painted Qt rectangle."""

    left, top, right, bottom = (float(value) for value in bounds)
    if not 0.0 <= left < right <= 1.0 or not 0.0 <= top < bottom <= 1.0:
        raise ValueError("normalized bounds must be a nonempty unit rectangle")
    return QtCore.QRectF(
        rect.x() + left * rect.width(),
        rect.y() + top * rect.height(),
        (right - left) * rect.width(),
        (bottom - top) * rect.height(),
    )


def raster_subrect(
    rect: QtCore.QRect,
    bounds: tuple[float, float, float, float],
) -> QtCore.QRect:
    """Map worker-authored raster bounds to the exact integer Qt hit box."""

    normalized_subrect(rect, bounds)
    left, top, right, bottom = (float(value) for value in bounds)
    x0 = rect.x() + round(left * rect.width())
    y0 = rect.y() + round(top * rect.height())
    x1 = rect.x() + round(right * rect.width())
    y1 = rect.y() + round(bottom * rect.height())
    return QtCore.QRect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def send_wheel(
    widget: QtWidgets.QWidget,
    position: QtCore.QPoint,
    delta: int,
) -> QtGui.QWheelEvent:
    """Deliver one real wheel event at a widget-local painted position."""

    event = QtGui.QWheelEvent(
        QtCore.QPointF(position),
        QtCore.QPointF(widget.mapToGlobal(position)),
        QtCore.QPoint(),
        QtCore.QPoint(0, int(delta)),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    return event


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


def send_mouse_double_click(widget, position, button) -> None:
    """Deliver only the double-click phase at a widget-local position."""

    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonDblClick,
        QtCore.QPointF(position),
        button,
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
    "choose_combo_data",
    "choose_combo_text",
    "click_tab",
    "close_pulse_editor",
    "close_task_console",
    "configure_offscreen_fast_path",
    "current_logic_editor",
    "drag_mouse_move",
    "normalized_subrect",
    "point_in_rect",
    "raster_subrect",
    "replace_path_value",
    "replace_spin_value",
    "require_offscreen_platform",
    "send_mouse_double_click",
    "send_wheel",
    "until",
    "visible_form_widgets",
]
