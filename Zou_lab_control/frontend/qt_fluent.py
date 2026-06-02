"""Reusable PyQt Fluent-style widgets for Zou_lab_control frontends.

The visual constants and widget shapes follow the original Confocal_GUIv2
Fluent layer: pale blue accent, Segoe UI text, white cards, small radii, and
soft shadows.
"""

from __future__ import annotations

import math
import re
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

try:  # Optional, but used when available to match Confocal_GUIv2 windows.
    from qframelesswindow import FramelessWindow, StandardTitleBar
except Exception:  # pragma: no cover - depends on optional desktop package.
    FramelessWindow = QtWidgets.QWidget
    StandardTitleBar = None


ACCENT = "#77AADD"
HOVER = "#004578"
BG = "#F3F3F3"
TEXT = "#323130"
HINT = "#F0a150"
PLACEHOLDER = "#A19F9D"
GREEN = "#7FC2AD"
RED = "#CD7380"
ORANGE = "#D69A6E"
YELLOW = "#E5C85B"
GREY = "#A2A2A2"
RADIUS = 4
FONT = "Segoe UI"
FONT_SIZE = 12
PADDING_V = 1
PADDING_H = 1
EDIT_PADDING_H = 4
COMBO_WIDTH = 16
COMBO_TRI_SIZE = 8
STEP_WIDTH = 6

_QT_APP = None
_FLUENT_SCALE = 1.0
_FLOAT_OR_X_RE = re.compile(
    r"""
    ^\s*
    (?:
        [+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?
        |
        [+-]?x
        |
        [+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*[-+*/]\s*x
        |
        [+-]?x\s*[-+*/]\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?
        |
        [+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*[-+]\s*x
        |
        [+-]?x\s*[-+]\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?
    )
    \s*$
    """,
    re.VERBOSE,
)


def fluent_scale() -> float:
    return _FLUENT_SCALE


def set_fluent_scale(scale: float | None = None) -> float:
    """Set the scale used by subsequently created Fluent widgets."""

    global _FLUENT_SCALE
    if scale is None:
        scale = 1.0
    _FLUENT_SCALE = max(0.72, min(1.25, float(scale)))
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.setFont(QtGui.QFont(FONT, fluent_font_size()))
    return _FLUENT_SCALE


def scaled_px(value: int | float, *, minimum: int = 1) -> int:
    return max(int(minimum), int(round(float(value) * _FLUENT_SCALE)))


def fluent_font_size() -> int:
    return max(8, int(round(FONT_SIZE * _FLUENT_SCALE)))


def fluent_text_width(metrics: QtGui.QFontMetrics, text: str) -> int:
    """Return text width on old and new PyQt5 builds."""

    if hasattr(metrics, "horizontalAdvance"):
        return int(metrics.horizontalAdvance(text))
    return int(metrics.width(text))


def _radius() -> int:
    return scaled_px(RADIUS)


def ensure_qt_app() -> QtWidgets.QApplication:
    """Return a QApplication, creating and keeping one alive when needed."""

    global _QT_APP
    app = QtWidgets.QApplication.instance()
    if app is not None:
        _QT_APP = app
        return app
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    _QT_APP = QtWidgets.QApplication(sys.argv)
    _QT_APP.setFont(QtGui.QFont(FONT, fluent_font_size()))
    return _QT_APP


def fluent_widget_stylesheet() -> str:
    return f'QWidget {{ background: {BG}; color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; }}'


def fluent_scrollbar_stylesheet(selector: str = "QScrollBar") -> str:
    """Return shared Fluent scrollbar CSS for scroll areas and popup lists."""

    thickness = scaled_px(12, minimum=10)
    return f"""
    {selector}:vertical {{
        background: transparent;
        border: none;
        width: {thickness}px;
        margin: 0px;
    }}
    {selector}:horizontal {{
        background: transparent;
        border: none;
        height: {thickness}px;
        margin: 0px;
    }}
    {selector}::handle:vertical, {selector}::handle:horizontal {{
        background: #C8C6C4;
        border: none;
        border-radius: {_radius()}px;
        min-height: {scaled_px(28)}px;
        min-width: {scaled_px(28)}px;
    }}
    {selector}::handle:vertical:hover, {selector}::handle:horizontal:hover {{
        background: {ACCENT};
    }}
    {selector}::add-line, {selector}::sub-line {{
        width: 0px;
        height: 0px;
        border: none;
        background: transparent;
    }}
    {selector}::add-page, {selector}::sub-page {{
        background: transparent;
    }}
    """


def status_dot_stylesheet(color: str, *, radius: int = 8) -> str:
    return f"background:{color}; border-radius:{scaled_px(radius)}px;"


def add_fluent_shadow(widget: QtWidgets.QWidget, *, blur: int = 20, alpha: int = 50, offset: int = 0) -> None:
    shadow = QtWidgets.QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(scaled_px(blur))
    shadow.setColor(QtGui.QColor(0, 0, 0, alpha))
    shadow.setOffset(0, scaled_px(offset, minimum=0))
    widget.setGraphicsEffect(shadow)


def format_compact_number(value: float, *, digits: int = 12) -> str:
    if not math.isfinite(float(value)):
        return str(value)
    text = f"{float(value):.{digits}g}"
    return text.replace("e+0", "e").replace("e+", "e")


def _confocal_float2str(value: float, *, length: int | None = None) -> str:
    """Match Confocal_GUIv2.helper.float2str for numeric spin boxes."""

    if length is not None:
        length = max(int(length), 5)
        sign = "-" if value < 0 else ""
        abs_val = abs(float(value))
        int_part = str(int(abs_val))
        dec_places = length - len(sign) - len(int_part) - 1
        if dec_places < 0:
            dec_places = 0
        text = f"{float(value):.{dec_places}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text[:length]
    return str(value)


def align_to_resolution(value: str | float, resolution: float, *, allow_any: bool = True) -> str:
    """Snap the numeric part of a simple value/x expression to ``resolution``."""

    if resolution is None or float(resolution) <= 0:
        return str(value)
    text = str(value).strip()
    if not text:
        return text

    def snap_number(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            number = float(token)
        except ValueError:
            return token
        snapped = round(number / float(resolution)) * float(resolution)
        if not allow_any and snapped <= 0:
            snapped = float(resolution)
        return format_compact_number(snapped)

    if "x" in text.lower():
        return re.sub(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", snap_number, text)
    return snap_number(re.match(r".+", text))


class FluentStatusDot(QtWidgets.QLabel):
    def __init__(self, parent=None, *, color: str = GREY, size: int = 16):
        super().__init__(parent)
        self._size = scaled_px(size)
        self.setFixedSize(self._size, self._size)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        self.setStyleSheet(status_dot_stylesheet(color, radius=max(1, self._size // 2)))


class FluentLabel(QtWidgets.QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f'QLabel {{ color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; background: transparent; }}')


class FluentFrame(QtWidgets.QFrame):
    def __init__(self, parent=None, *, shadow: bool = True, round: tuple[str, ...] = ("NW", "NE", "SE", "SW")):
        super().__init__(parent)
        corners = {corner.upper() for corner in round}
        top_left = _radius() if "NW" in corners else 0
        top_right = _radius() if "NE" in corners else 0
        bottom_right = _radius() if "SE" in corners else 0
        bottom_left = _radius() if "SW" in corners else 0
        self.setStyleSheet(
            f"""
            QFrame {{
                background: white;
                border: none;
                border-top-left-radius: {top_left}px;
                border-top-right-radius: {top_right}px;
                border-bottom-right-radius: {bottom_right}px;
                border-bottom-left-radius: {bottom_left}px;
            }}
            """
        )
        if shadow:
            add_fluent_shadow(self)


class FluentGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title: str = "", parent=None, *, shadow: bool = True):
        super().__init__(title, parent)
        self.setStyleSheet(
            f"""
            QGroupBox {{
                background: white;
                border: none;
                border-radius: {_radius()}px;
                margin-top: 0px;
                padding-top: {scaled_px(32)}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                background: {BG};
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                border-radius: {_radius()}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            """
        )
        if shadow:
            add_fluent_shadow(self)


class FluentButton(QtWidgets.QPushButton):
    def __init__(self, text: str = "", parent=None, *, color: str = ACCENT):
        super().__init__(text, parent)
        self._current_bg = None
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.set_color(color)

    def set_color(self, color: str) -> None:
        bg = QtGui.QColor(color).name(QtGui.QColor.HexRgb)
        if bg == self._current_bg:
            return
        self._current_bg = bg
        hover = QtGui.QColor(bg).darker(184).name(QtGui.QColor.HexRgb)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: {bg};
                color: white;
                border: none;
                border-radius: {_radius()}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:disabled {{ background: {PLACEHOLDER}; color: {BG}; }}
            """
        )
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)


class FluentLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._res_step: float | None = None
        self._allow_any = True
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.setStyleSheet(
            f"""
            QLineEdit {{
                background: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {_radius()}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
            QLineEdit:disabled {{ background: {BG}; color: {PLACEHOLDER}; }}
            """
        )
        self.setText(str(text))
        self.editingFinished.connect(self._snap_to_resolution)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        super().setText(str(text))
        if not self.hasFocus():
            self.setCursorPosition(0)

    def set_resolution(self, step: float | None) -> None:
        self._res_step = None if step in (None, 0) else float(step)

    def set_allow_any(self, allow_any: bool = True) -> None:
        self._allow_any = bool(allow_any)

    def _snap_to_resolution(self) -> None:
        if not self._res_step:
            return
        before = self.text()
        after = align_to_resolution(before, self._res_step, allow_any=self._allow_any)
        if after != before:
            self.setText(after)


class FloatLineEdit(FluentLineEdit):
    pass


class FloatOrXLineEdit(FluentLineEdit):
    def has_acceptable_text(self) -> bool:
        return bool(_FLOAT_OR_X_RE.fullmatch(self.text() or ""))


class FluentComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.setEditable(True)
        line_edit = self.lineEdit()
        line_edit.setReadOnly(True)
        line_edit.setFrame(False)
        line_edit.setCursor(QtCore.Qt.ArrowCursor)
        line_edit.setFocusPolicy(QtCore.Qt.NoFocus)
        line_edit.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        line_edit.setStyleSheet("background: transparent; border: none; padding: 0;")
        line_edit.setTextMargins(scaled_px(EDIT_PADDING_H), scaled_px(PADDING_V), scaled_px(EDIT_PADDING_H), scaled_px(PADDING_V))
        line_edit.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.currentTextChanged.connect(self._reset_cursor)
        self.editTextChanged.connect(self._reset_cursor)
        self.setStyleSheet(
            f"""
            QComboBox {{
                background-color: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {_radius()}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
                padding: 0px;
            }}
            QComboBox:disabled {{
                background-color: {BG};
                border: 1px solid {PLACEHOLDER};
                color: {PLACEHOLDER};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: border;
                subcontrol-position: right;
                width: {scaled_px(COMBO_WIDTH)}px;
                border: none;
                background-color: {ACCENT};
                border-top-right-radius: {_radius()}px;
                border-bottom-right-radius: {_radius()}px;
            }}
            QComboBox::drop-down:disabled {{ background-color: {PLACEHOLDER}; }}
            QComboBox::drop-down:hover {{ background-color: {HOVER}; }}
            QComboBox::down-arrow {{ image: none; }}
            QComboBox QAbstractItemView {{
                border: 1px solid {PLACEHOLDER};
                border-radius: {_radius()}px;
                background: white;
                color: {TEXT};
                outline: 0;
                selection-background-color: {ACCENT};
                selection-color: white;
                font: {fluent_font_size()}pt "{FONT}";
                padding: {scaled_px(2)}px;
            }}
            """
        )
        self.view().setMouseTracking(True)
        self.view().setStyleSheet(
            f"""
            QListView {{
                background: white;
                color: {TEXT};
                border: 1px solid {PLACEHOLDER};
                border-radius: {_radius()}px;
                outline: 0;
                padding: {scaled_px(2)}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QListView::item {{
                min-height: {scaled_px(24, minimum=18)}px;
                padding: {scaled_px(2)}px {scaled_px(6)}px;
                border-radius: {_radius()}px;
            }}
            QListView::item:hover {{
                background: {BG};
                color: {TEXT};
            }}
            QListView::item:selected {{
                background: {ACCENT};
                color: white;
            }}
            {fluent_scrollbar_stylesheet("QScrollBar")}
            """
        )

    def _reset_cursor(self, *_):
        if self.isEditable() and self.lineEdit():
            self.lineEdit().setCursorPosition(0)

    def _layout_lineedit(self) -> None:
        if not self.isEditable():
            return
        rect = QtCore.QRect(0, 0, max(0, self.width() - scaled_px(COMBO_WIDTH)), self.height())
        self.lineEdit().setGeometry(rect)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_lineedit()

    def showEvent(self, event):
        super().showEvent(event)
        self._layout_lineedit()
        self._reset_cursor()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#FFFFFF")))
        cx = int(self.width() - scaled_px(COMBO_WIDTH) / 2)
        cy = int(self.height() / 2)
        size = scaled_px(COMBO_TRI_SIZE)
        points = [
            QtCore.QPoint(cx - size // 2, cy - size // 4),
            QtCore.QPoint(cx + size // 2, cy - size // 4),
            QtCore.QPoint(cx, cy + size // 4),
        ]
        painter.drawPolygon(QtGui.QPolygon(points))
        painter.end()

    def wheelEvent(self, event):
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class FluentTabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabBar().setExpanding(False)
        self.setUsesScrollButtons(True)
        self.setStyleSheet(
            f"""
            QTabWidget::pane {{
                background: white;
                margin: 0px;
                padding: 0px;
                border: none;
                border-top-right-radius: {_radius()}px;
                border-bottom-left-radius: {_radius()}px;
                border-bottom-right-radius: {_radius()}px;
            }}
            QTabWidget QStackedWidget {{
                background: white;
                border: none;
                margin: 0px;
                padding: 0px;
                border-top-right-radius: {_radius()}px;
                border-bottom-left-radius: {_radius()}px;
                border-bottom-right-radius: {_radius()}px;
            }}
            QTabWidget {{
                margin: 0px;
                padding: 0px;
            }}
            QTabWidget::tab-bar {{
                margin: 0px;
                padding: 0px;
            }}
            QTabBar::tab {{
                background: {BG};
                color: {TEXT};
                border: none;
                border-top-left-radius: {_radius()}px;
                border-top-right-radius: {_radius()}px;
                min-width: {scaled_px(82, minimum=68)}px;
                height: {scaled_px(30, minimum=24)}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(PADDING_H)}px;
                margin-right: {scaled_px(2)}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QTabBar::tab:selected {{
                background: white;
                color: {TEXT};
            }}
            QTabBar::tab:!selected:hover {{
                background: {ACCENT};
                color: white;
            }}
            """
        )
        add_fluent_shadow(self, blur=10, alpha=50, offset=2)


class FluentSwitch(QtWidgets.QAbstractButton):
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFont(QtGui.QFont(FONT, fluent_font_size()))
        self._offset = float(scaled_px(3, minimum=2))
        self._animating = False
        self._anim = QtCore.QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.stateChanged.connect(self._on_anim_state_changed)
        self.toggled.connect(self._start_animation)
        self.setMinimumSize(scaled_px(126, minimum=96), scaled_px(30, minimum=24))

    def sizeHint(self) -> QtCore.QSize:
        text_w = fluent_text_width(QtGui.QFontMetrics(self.font()), self.text())
        return QtCore.QSize(max(self.minimumWidth(), scaled_px(60) + text_w), self.minimumHeight())

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        track_w = scaled_px(60, minimum=48)
        track_h = min(self.height(), scaled_px(30, minimum=24))
        y = int((self.height() - track_h) / 2)
        track_color = ACCENT if self.isChecked() and self.isEnabled() else (PLACEHOLDER if self.isEnabled() else BG)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(track_color)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(0, y, track_w, track_h, track_h / 2, track_h / 2)

        margin = scaled_px(3, minimum=2)
        thumb_d = max(1, track_h - margin * 2)
        offset = self._offset if self._animating else self._checked_offset(track_w, thumb_d, margin)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#FFFFFF")))
        painter.drawEllipse(int(offset), y + margin, thumb_d, thumb_d)

        if self.text():
            painter.setPen(QtGui.QColor(TEXT))
            text_rect = QtCore.QRect(track_w + scaled_px(8), 0, max(0, self.width() - track_w - scaled_px(8)), self.height())
            painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.text())
        painter.end()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)

    def _checked_offset(self, track_w: int | None = None, thumb_d: int | None = None, margin: int | None = None) -> float:
        track_w = scaled_px(60, minimum=48) if track_w is None else int(track_w)
        track_h = min(self.height() or scaled_px(30, minimum=24), scaled_px(30, minimum=24))
        margin = scaled_px(3, minimum=2) if margin is None else int(margin)
        thumb_d = max(1, track_h - margin * 2) if thumb_d is None else int(thumb_d)
        return float(track_w - thumb_d - margin if self.isChecked() else margin)

    def _start_animation(self, checked: bool) -> None:
        del checked
        end_pos = self._checked_offset()
        if not self.isEnabled() or self.width() <= 0 or self.height() <= 0:
            self._anim.stop()
            self._offset = end_pos
            self.update()
            return
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(end_pos)
        self._anim.start()

    def _on_anim_state_changed(self, new_state, _old_state) -> None:
        self._animating = new_state == QtCore.QAbstractAnimation.Running
        if not self._animating:
            self._offset = self._checked_offset()
            self.update()

    def getOffset(self) -> float:
        return float(self._offset)

    def setOffset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = QtCore.pyqtProperty(float, fget=getOffset, fset=setOffset)


def fluent_spinbox_stylesheet(selector: str) -> str:
    return f"""
    {selector} {{
        background: white;
        border: 1px solid {PLACEHOLDER};
        border-radius: {_radius()}px;
        padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
        font: {fluent_font_size()}pt "{FONT}";
        color: {TEXT};
    }}
    {selector}:focus {{ border: 1px solid {ACCENT}; }}
    {selector}:disabled {{ background: {BG}; color: {PLACEHOLDER}; }}
    {selector}::up-button, {selector}::down-button {{
        subcontrol-origin: border;
        width: {scaled_px(COMBO_WIDTH)}px;
        border: none;
        background-color: {ACCENT};
    }}
    {selector}::up-button {{
        subcontrol-position: top right;
        border-top-right-radius: {_radius()}px;
    }}
    {selector}::down-button {{
        subcontrol-position: bottom right;
        border-bottom-right-radius: {_radius()}px;
    }}
    {selector}::up-button:hover, {selector}::down-button:hover {{
        background-color: {HOVER};
    }}
    {selector}::up-arrow, {selector}::down-arrow {{ image: none; }}
    """


class FluentSpinBox(QtWidgets.QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet(fluent_spinbox_stylesheet("QSpinBox"))


class FluentInputDialog(QtWidgets.QDialog):
    """Small Confocal-style numeric input dialog used by step editors."""

    def __init__(self, prompt: str, default: float, parent=None):
        super().__init__(parent, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self.setFont(QtGui.QFont(FONT, fluent_font_size()))
        self.setStyleSheet("QDialog { background: white; }")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(scaled_px(12), scaled_px(12), scaled_px(12), scaled_px(12))
        layout.setSpacing(scaled_px(8, minimum=5))

        layout.addWidget(FluentLabel(prompt, self))
        self._edit = FluentLineEdit(str(default), self)
        layout.addWidget(self._edit)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        ok = FluentButton("OK", self)
        cancel = FluentButton("Cancel", self)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)

    def getValue(self):
        if self.exec_() == QtWidgets.QDialog.Accepted:
            try:
                return float(self._edit.text()), True
            except ValueError:
                return None, False
        return None, False


class FluentDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """Confocal_GUIv2-style double spinbox with an inline step editor."""

    def __init__(self, length=5, allow_minus: bool = False, parent=None):
        if isinstance(length, QtWidgets.QWidget) and parent is None:
            parent = length
            length = 5
        base_class = type(self).mro()[1]
        if "Confocal_GUIv2" in getattr(base_class, "__module__", ""):
            super().__init__(length=length, allow_minus=allow_minus, parent=parent)
            self.setMinimumHeight(scaled_px(30, minimum=22))
            return
        super().__init__(parent)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet(fluent_spinbox_stylesheet("QDoubleSpinBox"))

        self._step_btn = QtWidgets.QToolButton(self)
        self._step_btn.setText(".")
        self._step_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._step_btn.setStyleSheet(
            f"""
            QToolButton {{
                background: {ACCENT};
                color: white;
                border: none;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QToolButton:hover {{
                background: {HOVER};
            }}
            """
        )
        self._step_btn.clicked.connect(self._on_edit_step)

        self.length = max(int(length), 5)
        if allow_minus:
            self.setRange(1 - 10 ** (self.length - 1), 10**self.length - 1)
        else:
            self.setRange(10 ** -(self.length - 2), 10**self.length - 1)
        self.setSingleStep(1)
        self.setDecimals(self.length - 2)
        self.lineEdit().setMaxLength(self.length)

    def setValue(self, value: float) -> None:
        rounded = float(_confocal_float2str(value, length=self.length))
        super().setValue(rounded)

    def setSingleStep(self, step: float) -> None:
        rounded = float(_confocal_float2str(step, length=self.length))
        super().setSingleStep(rounded)

    def stepBy(self, steps: int) -> None:
        current = self.value()
        step = self.singleStep()
        target = float(_confocal_float2str(current + steps * step, length=self.length))
        if self.minimum() <= target <= self.maximum():
            super(FluentDoubleSpinBox, self).setValue(target)

    def textFromValue(self, value: float) -> str:
        return _confocal_float2str(value, length=self.length)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        btn_w = scaled_px(COMBO_WIDTH)
        step_w = scaled_px(STEP_WIDTH)
        self._step_btn.setGeometry(self.width() - btn_w - step_w, 0, step_w, self.height())

    def _on_edit_step(self) -> None:
        current = self.singleStep()
        dialog = FluentInputDialog("Edit step", current, self)
        dialog._edit.setMaxLength(self.lineEdit().maxLength())
        value, ok = dialog.getValue()
        if ok and value is not None:
            self.setSingleStep(value)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        size = scaled_px(COMBO_TRI_SIZE)
        btn_w = scaled_px(COMBO_WIDTH)
        cx = self.width() - btn_w / 2
        height = self.height()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#FFFFFF")))

        y_up = height * 0.25
        up_points = [
            QtCore.QPoint(int(cx - size / 2), int(y_up + size / 4)),
            QtCore.QPoint(int(cx + size / 2), int(y_up + size / 4)),
            QtCore.QPoint(int(cx), int(y_up - size / 4)),
        ]
        painter.drawPolygon(QtGui.QPolygon(up_points))

        y_down = height * 0.75
        down_points = [
            QtCore.QPoint(int(cx - size / 2), int(y_down - size / 4)),
            QtCore.QPoint(int(cx + size / 2), int(y_down - size / 4)),
            QtCore.QPoint(int(cx), int(y_down + size / 4)),
        ]
        painter.drawPolygon(QtGui.QPolygon(down_points))
        painter.end()


class FluentCheckBox(QtWidgets.QCheckBox):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.setStyleSheet(
            f"""
            QCheckBox {{
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
                spacing: {scaled_px(6)}px;
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: {scaled_px(16)}px;
                height: {scaled_px(16)}px;
                border-radius: {scaled_px(8)}px;
                border: 1px solid {PLACEHOLDER};
                background: white;
            }}
            QCheckBox::indicator:checked {{
                background: {ACCENT};
                border: 1px solid {ACCENT};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {HOVER};
            }}
            """
        )


class FluentScrollArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet(
            f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollArea > QWidget > QWidget {{ background: transparent; }}
            {fluent_scrollbar_stylesheet("QScrollBar")}
            """
        )


class FluentWindow(FramelessWindow):
    """Frameless Confocal-style wrapper for PyQt frontends.

    The original Confocal GUI uses ``qframelesswindow`` with a 32 px custom
    titlebar.  This wrapper keeps that shape when the optional package is
    installed and falls back to a regular QWidget otherwise.
    """

    hidden = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        widget: QtWidgets.QWidget | None = None,
        widget_class: type | None = None,
        widget_kwargs: dict | None = None,
        title: str = "",
        hide_on_close: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._hide_on_close = bool(hide_on_close)
        self.setObjectName("FluentWindow")
        self.setStyleSheet(f"QWidget#FluentWindow {{ background: {BG}; }}")

        if StandardTitleBar is not None:
            title_bar = StandardTitleBar(self)
            title_bar.setTitle(title)
            title_bar.iconLabel.setFixedSize(0, 0)
            title_bar.titleLabel.setStyleSheet(
                f"""
                QLabel {{
                    background: transparent;
                    color: {TEXT};
                    font: {fluent_font_size()}pt "{FONT}";
                    padding: 0 {scaled_px(4)}px;
                }}
                """
            )
            self.setTitleBar(title_bar)
            top_margin = scaled_px(32)
        else:
            self.setWindowTitle(title)
            top_margin = 0

        if widget is not None:
            self.loaded = widget
            self.loaded.setParent(self)
        elif widget_class is not None:
            kwargs = dict(widget_kwargs or {})
            kwargs.setdefault("parent", self)
            self.loaded = widget_class(**kwargs)
        else:
            raise ValueError("FluentWindow needs widget or widget_class.")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, top_margin, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.loaded)
        if StandardTitleBar is not None:
            self.titleBar.raise_()
        self.loaded.adjustSize()
        self.resize(max(scaled_px(900, minimum=680), self.loaded.width()), self.loaded.height() + top_margin)

    def closeEvent(self, event):
        self.hidden.emit()
        if self._hide_on_close:
            event.ignore()
            self.hide()
            return
        super().closeEvent(event)

    def hideEvent(self, event):
        self.hidden.emit()
        super().hideEvent(event)


def run_fluent_window(
    *,
    widget_class: type | None = None,
    widget_kwargs: dict | None = None,
    widget: QtWidgets.QWidget | None = None,
    title: str = "",
    in_GUI: bool = False,
    window_handle: FluentWindow | None = None,
) -> FluentWindow:
    """Open a Confocal-style Fluent window and keep the QApplication alive."""

    app = ensure_qt_app()
    window = window_handle or FluentWindow(
        widget=widget,
        widget_class=widget_class,
        widget_kwargs=widget_kwargs,
        title=title,
        hide_on_close=not in_GUI,
    )

    screen = app.primaryScreen()
    if screen is not None:
        screen_geo = screen.availableGeometry()
        frame = window.frameGeometry()
        frame.moveCenter(screen_geo.center())
        window.move(frame.topLeft())
    window.show()

    if in_GUI:
        return window

    loop = QtCore.QEventLoop()
    window.hidden.connect(loop.quit)
    loop.exec_()
    return window


__all__ = [
    "ACCENT",
    "BG",
    "FONT",
    "FONT_SIZE",
    "GREEN",
    "GREY",
    "HINT",
    "HOVER",
    "ORANGE",
    "PLACEHOLDER",
    "RADIUS",
    "RED",
    "TEXT",
    "YELLOW",
    "FluentButton",
    "FluentCheckBox",
    "FluentComboBox",
    "FluentDoubleSpinBox",
    "FluentFrame",
    "FluentGroupBox",
    "FluentInputDialog",
    "FluentLabel",
    "FluentLineEdit",
    "FluentSpinBox",
    "FluentSwitch",
    "FluentTabWidget",
    "FloatLineEdit",
    "FloatOrXLineEdit",
    "FluentScrollArea",
    "FluentStatusDot",
    "FluentWindow",
    "add_fluent_shadow",
    "align_to_resolution",
    "ensure_qt_app",
    "fluent_font_size",
    "fluent_scale",
    "fluent_scrollbar_stylesheet",
    "fluent_spinbox_stylesheet",
    "fluent_text_width",
    "fluent_widget_stylesheet",
    "format_compact_number",
    "run_fluent_window",
    "scaled_px",
    "set_fluent_scale",
    "status_dot_stylesheet",
]
