"""TaskConsole card geometry and the absolute-positioned board surface.

This module owns the one conversion between semantic panel sizes and Qt pixel
geometry.  Card order remains the layout authority; the shared headless board
packer computes positions and this Qt surface only places already-existing
cards at those positions.
"""

from __future__ import annotations

from collections.abc import Sequence

from PyQt5 import QtGui, QtWidgets

from zlc_data.console_records import PanelConfig
from zlc_data.panel_size import panel_size_cells
from zlc_frontend import board_layout as _layout
from zlc_frontend.qt_widgets import CARD_PAD, CARD_TITLE_PX, scaled_px
from zlc_frontend.render_style import panel_display_size


GRID_UNIT = 8
GAP = GRID_UNIT


def cell_size() -> tuple[int, int]:
    """Return the pixel footprint of the narrowest ``1x2`` card."""

    width = panel_display_size("1x2")[0] + 2 * CARD_PAD
    height = (
        scaled_px(CARD_TITLE_PX)
        + scaled_px(2)
        + panel_display_size("1x2")[1]
        + CARD_PAD
    )
    return width, height


def card_size(size: str) -> tuple[int, int]:
    """Convert one declared panel-size preset to its exact card pixels."""

    _rows, cols = panel_size_cells(size)
    width_units = max(1, cols // 2)
    cell_width, _cell_height = cell_size()
    width = width_units * cell_width + (width_units - 1) * GAP
    height = (
        scaled_px(CARD_TITLE_PX)
        + scaled_px(2)
        + panel_display_size(size)[1]
        + CARD_PAD
    )
    return width, height


def _metrics() -> _layout.BoardMetrics:
    # Read live: Qt scale can change before a window is composed.
    return _layout.BoardMetrics(gap=GAP, card_size=card_size)


def board_width(configs: Sequence[PanelConfig]) -> int:
    return _layout.board_width(configs, _metrics())


def pack(order: Sequence[PanelConfig], board_width_px: int | None = None) -> bool:
    return _layout.pack(order, _metrics(), board_width_px)


def drop_index(
    config: PanelConfig,
    others: Sequence[PanelConfig],
    board_width_px: int | None = None,
) -> int:
    return _layout.drop_index(config, others, _metrics(), board_width_px)


def opaque_white_composite(pixmap: QtGui.QPixmap) -> QtGui.QPixmap:
    """Flatten a grabbed HiDPI board onto white without changing its DPR."""

    canvas = QtGui.QPixmap(pixmap.size())
    canvas.setDevicePixelRatio(pixmap.devicePixelRatio())
    canvas.fill(QtGui.QColor("#FFFFFF"))
    painter = QtGui.QPainter(canvas)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()
    return canvas


class PanelBoard(QtWidgets.QWidget):
    """Transparent surface that places stable card widgets at packed positions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

    def arrange(self, cards: Sequence[QtWidgets.QWidget]) -> None:
        max_x = max_y = 0
        for card in cards:
            x, y = card.config.col, card.config.row
            card.move(x, y)
            max_x = max(max_x, x + card.width())
            max_y = max(max_y, y + card.height())
        self.setMinimumSize(max_x + GAP, max_y + GAP)
