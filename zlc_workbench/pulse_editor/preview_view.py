"""Frozen Preview tab for the formal Pulse GUI."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_frontend.panel_size import PANEL_SIZES
from zlc_frontend.qt_widgets import (
    ACCENT,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentSwitch,
    signals_blocked,
)

from ._layout import px


class PulsePreviewView(QtWidgets.QWidget):
    """Exact formal Preview layout; rendering remains a separate owner."""

    includeOffToggled = QtCore.pyqtSignal(bool)
    selectorsToggled = QtCore.pyqtSignal(bool)
    sizeActivated = QtCore.pyqtSignal(str)
    saveFigureRequested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._preview_size_pinned = False
        self._content_widget: QtWidgets.QWidget | None = None
        self._wheel_target: QtWidgets.QWidget | None = None

        tab_margin = px(8, minimum=5)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(tab_margin, tab_margin, tab_margin, tab_margin)
        layout.setSpacing(px(8, minimum=5))

        controls = FluentFrame()
        controls.setFixedHeight(px(48, minimum=40))
        row = QtWidgets.QHBoxLayout(controls)
        row.setContentsMargins(px(12), px(6), px(12), px(6))
        row.setSpacing(px(10, minimum=6))
        control_h = px(32, minimum=28)

        self.preview_include_off = FluentSwitch("Show off rows")
        self.preview_include_off.setFixedSize(
            px(198, minimum=178),
            control_h,
        )
        self.preview_include_off.setToolTip(
            "Show channels that are always off in the preview."
        )
        self.preview_include_off.toggled.connect(self._include_off_changed)

        self.preview_selectors_switch = FluentSwitch("Selectors")
        self.preview_selectors_switch.setFixedSize(
            px(168, minimum=150),
            control_h,
        )
        self.preview_selectors_switch.setChecked(False)
        self.preview_selectors_switch.setToolTip(
            "OFF: the preview is display-only (wheel scrolls).\n"
            "ON: zoom / pan / area / cross work on the preview plot."
        )
        self.preview_selectors_switch.toggled.connect(self.selectorsToggled)

        self.preview_size_label = FluentLabel("Size")
        self.preview_size_combo = FluentComboBox()
        self.preview_size_combo.addItems(list(PANEL_SIZES))
        self.preview_size_combo.setCurrentText("2x2")
        self.preview_size_combo.setFixedSize(px(80, minimum=66), control_h)
        self.preview_size_combo.setToolTip(
            "Plot size preset -- a busy pulse defaults to a bigger size; pick "
            "one to pin it (saved with the figure)."
        )
        self.preview_size_combo.activated.connect(self._size_picked)

        self.preview_save_figure_button = FluentButton(
            "Save Figure",
            color=ACCENT,
        )
        self.preview_save_figure_button.setFixedSize(
            px(124, minimum=108),
            control_h,
        )
        self.preview_save_figure_button.clicked.connect(
            self.saveFigureRequested
        )
        self.preview_status = FluentLineEdit("")
        self.preview_status.setEnabled(False)
        self.preview_status.setFixedHeight(control_h)

        row.addWidget(self.preview_include_off)
        row.addWidget(self.preview_selectors_switch)
        row.addWidget(self.preview_size_label)
        row.addWidget(self.preview_size_combo)
        row.addWidget(self.preview_status, 1)
        row.addWidget(self.preview_save_figure_button)
        layout.addWidget(controls)

        self.preview_scroll = FluentScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.preview_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_body = QtWidgets.QWidget()
        self.preview_body_layout = QtWidgets.QVBoxLayout(self.preview_body)
        self.preview_body_layout.setContentsMargins(
            px(8),
            px(8),
            px(8),
            px(8),
        )
        self.preview_body_layout.setSpacing(0)
        self.preview_placeholder = FluentLabel(
            "Open Preview to render the pulse plot."
        )
        self.preview_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_body_layout.addWidget(self.preview_placeholder)
        self.preview_scroll.setWidget(self.preview_body)
        layout.addWidget(self.preview_scroll, 1)

    @property
    def preview_size_pinned(self) -> bool:
        return self._preview_size_pinned

    @property
    def preview_size(self) -> str:
        return self.preview_size_combo.currentText()

    @property
    def include_off_rows(self) -> bool:
        return self.preview_include_off.isChecked()

    def _include_off_changed(self, checked: bool) -> None:
        self._preview_size_pinned = False
        self.includeOffToggled.emit(bool(checked))

    def _size_picked(self, _index: int) -> None:
        self._preview_size_pinned = True
        self.sizeActivated.emit(self.preview_size_combo.currentText())

    def set_preview_size(self, size: str, *, pinned: bool | None = None) -> None:
        with signals_blocked(self.preview_size_combo):
            self.preview_size_combo.setCurrentText(str(size))
        if pinned is not None:
            self._preview_size_pinned = bool(pinned)

    def reset_preview_size_pin(self) -> None:
        """Return the transient size selector to content-derived ownership."""

        self._preview_size_pinned = False

    def set_status(self, text: str) -> None:
        self.preview_status.setText(str(text))

    def mount_content(
        self,
        widget: QtWidgets.QWidget,
        *,
        logical_size: tuple[int, int] | None = None,
        wheel_target: QtWidgets.QWidget | None = None,
    ) -> None:
        if not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("preview content must be a QWidget")
        if self._content_widget is not None and self._content_widget is not widget:
            retired = self._content_widget
            self.preview_body_layout.removeWidget(retired)
            retired.hide()
            retired.deleteLater()
        if self._content_widget is not widget:
            self.preview_body_layout.addWidget(
                widget,
                0,
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
            )
            self._content_widget = widget
        self.preview_placeholder.hide()
        self._wheel_target = wheel_target
        if logical_size is not None:
            width, height = (int(value) for value in logical_size)
            margins = self.preview_body_layout.contentsMargins()
            self.preview_body.resize(
                width + margins.left() + margins.right(),
                height + margins.top() + margins.bottom(),
            )

    def show_placeholder(self, text: str) -> None:
        self.preview_placeholder.setText(str(text))
        self.preview_placeholder.show()

    def eventFilter(self, obj, event):  # noqa: N802
        if (
            event.type() == QtCore.QEvent.Wheel
            and obj is self.preview_scroll.viewport()
        ):
            canvas = self._wheel_target
            if canvas is not None:
                global_pos = (
                    event.globalPos()
                    if hasattr(event, "globalPos")
                    else event.globalPosition().toPoint()
                )
                local = canvas.mapFromGlobal(global_pos)
                if canvas.rect().contains(local):
                    mapped = QtGui.QWheelEvent(
                        QtCore.QPointF(local),
                        QtCore.QPointF(global_pos),
                        event.pixelDelta(),
                        event.angleDelta(),
                        event.buttons(),
                        event.modifiers(),
                        event.phase(),
                        event.inverted(),
                    )
                    QtWidgets.QApplication.sendEvent(canvas, mapped)
                    return True
        return super().eventFilter(obj, event)


__all__ = ["PulsePreviewView"]
