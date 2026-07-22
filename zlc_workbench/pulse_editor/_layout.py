"""Frozen geometry helpers shared by the formal Pulse GUI views.

These functions are a mechanical extraction from the established PulseGUI.
They own no document or runtime state; keeping them here prevents the Edit,
Preview, and Scan pages from drifting while their controllers are separated.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_frontend.qt_widgets import (
    FONT,
    FluentLabel,
    fluent_font_size,
    measure_text_width,
    scaled_px,
)


ROW_HEIGHT = 30
CHANNEL_LABEL_WIDTH = 100
TIME_UNIT_WIDTH = 60
HIDE_BUTTON_WIDTH = 26
PANEL_TOP_HEIGHT = 178
CHANNEL_ROW_SPACING = 4
PERIOD_CARD_WIDTH = 158


def px(value: int | float, *, minimum: int = 1) -> int:
    return scaled_px(value, minimum=minimum)


def font_metrics() -> QtGui.QFontMetrics:
    return QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))


def row_height() -> int:
    return px(ROW_HEIGHT, minimum=22)


def row_spacing() -> int:
    return px(CHANNEL_ROW_SPACING, minimum=3)


def row_region_vmetrics() -> tuple[int, int]:
    return px(8), row_spacing()


def channel_label_width() -> int:
    return px(CHANNEL_LABEL_WIDTH, minimum=84)


def channel_name_edit_width() -> int:
    return px(108, minimum=88)


def time_unit_width() -> int:
    return px(TIME_UNIT_WIDTH, minimum=62)


def hide_button_width() -> int:
    return px(HIDE_BUTTON_WIDTH, minimum=22)


def panel_top_height() -> int:
    return px(PANEL_TOP_HEIGHT, minimum=138)


def card_gutter() -> int:
    return px(5, minimum=4)


def period_card_width() -> int:
    return px(PERIOD_CARD_WIDTH, minimum=112)


def period_control_width(card_width: int) -> int:
    return max(px(76, minimum=68), card_width - 2 * px(7) - px(4))


def bus_mode_combo_width() -> int:
    return measure_text_width(["Edge", "Ramp", "Hold"], padding=34)


def set_fixed_height(
    widget: QtWidgets.QWidget,
    height: int | None = None,
) -> QtWidgets.QWidget:
    widget.setFixedHeight(row_height() if height is None else height)
    return widget


def set_form_label_geometry(label: FluentLabel) -> FluentLabel:
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setFixedSize(channel_label_width(), row_height())
    return label


def form_control_cell(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    widget.setFixedHeight(row_height())
    cell = QtWidgets.QWidget()
    cell.setStyleSheet("background: transparent;")
    layout = QtWidgets.QHBoxLayout(cell)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    fixed_width = (
        widget.minimumWidth() > 0
        and widget.maximumWidth() == widget.minimumWidth()
    )
    if fixed_width:
        layout.addWidget(widget, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.addStretch(1)
    else:
        layout.addWidget(widget, 1)
    return cell


def add_labeled_widget(
    layout,
    label_text: str,
    widget: QtWidgets.QWidget,
) -> FluentLabel:
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(px(5, minimum=3))
    label = FluentLabel(label_text)
    set_form_label_geometry(label)
    row.addWidget(label)
    row.addWidget(form_control_cell(widget), 1)
    layout.addLayout(row)
    return label


def channel_row_height() -> int:
    """Return the one formal channel-row height used by every pulse column.

    The number of visible channels controls only the scroll extent.  It must
    never select a denser layout: doing so moves every existing control when
    Show All crosses an arbitrary channel-count threshold.
    """

    return px(ROW_HEIGHT, minimum=22)


def elide_text(text: object, width: int) -> str:
    return font_metrics().elidedText(
        str(text),
        QtCore.Qt.ElideRight,
        max(8, int(width)),
    )


__all__ = [
    "add_labeled_widget",
    "bus_mode_combo_width",
    "card_gutter",
    "channel_label_width",
    "channel_name_edit_width",
    "channel_row_height",
    "elide_text",
    "font_metrics",
    "form_control_cell",
    "hide_button_width",
    "panel_top_height",
    "period_card_width",
    "period_control_width",
    "px",
    "row_height",
    "row_region_vmetrics",
    "row_spacing",
    "set_fixed_height",
    "set_form_label_geometry",
    "time_unit_width",
]
