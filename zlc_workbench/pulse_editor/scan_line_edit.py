"""Pulse-editor field with one inline Scan/API binding control."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_frontend.qt_widgets import (
    API_VIOLET,
    API_VIOLET_DARK,
    BG,
    EDIT_PADDING_H,
    FONT,
    ORANGE,
    ORANGE_DARK,
    ORANGE_TINT,
    PADDING_V,
    PLACEHOLDER,
    RADIUS,
    FluentLineEdit,
    Metrics,
    fluent_font_size,
    scaled_px,
)

__all__ = ["FluentScanLineEdit"]


class _FluentScanDot(QtWidgets.QAbstractButton):
    """Pulse-local Scan/API binding button embedded in one numeric field."""

    def __init__(self, parent=None, *, tooltip: str):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._number: int | None = None
        self._api = False
        diameter = Metrics.dot()
        self.setFixedSize(diameter, diameter)
        self.setToolTip(tooltip)

    def set_number(self, number: int | None) -> None:
        self._number = None if number is None else int(number)
        self.update()

    def set_api(self, api: bool) -> None:
        self._api = bool(api)
        self.update()

    def nextCheckState(self) -> None:  # noqa: N802 - model owns binding state
        pass

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = min(self.width(), self.height()) / 2.0 - max(1.0, scaled_px(1))
        if self.isChecked() or self._api:
            painter.setBrush(QtGui.QColor(API_VIOLET if self._api else ORANGE))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(center, radius, radius)
            if self._number is not None:
                painter.setPen(QtGui.QColor("#FFFFFF"))
                font = QtGui.QFont(FONT, max(6, fluent_font_size() - 5))
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, str(self._number))
        else:
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(
                QtGui.QPen(QtGui.QColor(PLACEHOLDER), max(1, scaled_px(1)))
            )
            painter.drawEllipse(center, radius, radius)
            painter.setBrush(QtGui.QColor(PLACEHOLDER))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(center, radius * 0.42, radius * 0.42)
        painter.end()


def _bound_field_style(
    *,
    text: str,
    border: str,
    fill: str | None,
) -> str:
    background = f"background: {fill}; " if fill else ""
    return (
        f"QLineEdit {{ {background}color: {text}; border: 1px solid {border}; "
        f"border-radius: {scaled_px(RADIUS)}px; "
        f"padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px; "
        f'font: {fluent_font_size()}pt "{FONT}"; }}'
    )


def _muted_line_style() -> str:
    return (
        f"QLineEdit {{ background: {BG}; color: {PLACEHOLDER}; "
        f"border: 1px solid {PLACEHOLDER}; "
        f"border-radius: {scaled_px(RADIUS)}px; "
        f"padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px; "
        f'font: {fluent_font_size()}pt "{FONT}"; }}'
    )


class FluentScanLineEdit(FluentLineEdit):
    """Numeric Pulse field whose dot cycles unbound -> Scan -> API."""

    scanClicked = QtCore.pyqtSignal()

    def __init__(
        self,
        text: str = "",
        parent=None,
        *,
        tooltip: str = "Click the dot to scan this field",
    ) -> None:
        super().__init__(text, parent)
        self._base_style = self.styleSheet()
        self._dot = _FluentScanDot(self, tooltip=tooltip)
        self._dot.clicked.connect(self.scanClicked)
        self._field_state: tuple[bool, str | None, int | None] | None = None
        self._reserve_right()

    @property
    def dot(self) -> _FluentScanDot:
        return self._dot

    @staticmethod
    def _dot_size() -> int:
        return Metrics.dot()

    def _reserve_right(self) -> None:
        self.setTextMargins(0, 0, self._dot_size() + scaled_px(3), 0)

    def _place_dot(self) -> None:
        diameter = self._dot_size()
        self._dot.setGeometry(
            int(self.width() - diameter - scaled_px(4)),
            int((self.height() - diameter) // 2),
            int(diameter),
            int(diameter),
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_dot()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._place_dot()

    def set_field_state(
        self,
        *,
        editable: bool,
        binding: str | None = None,
        number: int | None = None,
    ) -> None:
        normalized = None if binding is None else str(binding).strip().lower()
        if normalized not in (None, "scan", "api"):
            raise ValueError("binding must be 'scan', 'api', or None")
        if normalized is None and number is not None:
            raise ValueError("an unbound field cannot have a binding number")
        state = (bool(editable), normalized, number)
        if state == self._field_state:
            return
        self._field_state = state

        is_scan = normalized == "scan"
        is_api = normalized == "api"
        self._dot.set_api(is_api)
        self._dot.setChecked(is_scan)
        self._dot.set_number(number if normalized is not None else None)
        self.setReadOnly(is_scan or not state[0])
        if is_scan:
            style = _bound_field_style(
                text=ORANGE_DARK,
                border=ORANGE,
                fill=ORANGE_TINT,
            )
        elif is_api:
            style = _bound_field_style(
                text=API_VIOLET_DARK,
                border=API_VIOLET,
                fill=None,
            )
        else:
            style = self._base_style if state[0] else _muted_line_style()
        self.setStyleSheet(style)
        self._reserve_right()
        self.update()
