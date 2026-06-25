"""Contract: a panel card's WIDTH is on a clean COLUMN grid (cols // 2 base-width columns) but its
HEIGHT HUGS the plot -- the card is exactly tall enough for its figure + chrome, with NO blank
padding below (every size hugs like 1x2, #H3i-3).  Because heights are therefore VARIABLE, the
vertical layout is pixel-based (``col`` snaps to the column grid; ``row`` is a pixel y, cards
free-stack and ``_compact`` pushes an overlap down by the blocker's ACTUAL pixel height).

The card's FORMAT (rounded corners / shadow / grey title strip / content padding) is owned by the
FluentGroupBox component (qt_fluent.CARD_PAD / CARD_TITLE_PX); this module only sizes + places cards.
Pure functions (no Qt window beyond the QApplication scaled_px needs) so this runs in the suite.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.frontend.live import PANEL_SIZES, panel_display_size, panel_size_cells
from Zou_lab_control.frontend.qt_fluent import CARD_PAD, CARD_TITLE_PX, ensure_qt_app, scaled_px
from Zou_lab_control.frontend.task_console import (
    _GRID_GAP, _card_size, _cell_size, _columns_overlap, _compact, _pos_to_slot, _slot_to_pos)

ensure_qt_app()                       # the cell size reads scaled_px (needs the QApplication)


def _chrome_h() -> int:
    return scaled_px(CARD_TITLE_PX) + scaled_px(2) + CARD_PAD


def _w_units(size):
    return max(1, panel_size_cells(size)[1] // 2)


def test_one_by_two_is_exactly_one_cell():
    """The base 1x2 panel IS one grid cell, and the cell hugs the plot (figure + thin border)."""
    cw, ch = _cell_size()
    assert _card_size("1x2") == (cw, ch)
    assert cw == panel_display_size("1x2")[0] + 2 * CARD_PAD   # cell width hugs the figure L/R


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_width_is_whole_columns(size):
    """Card WIDTH == w_units base columns + the gaps between them (the clean horizontal grid)."""
    cw, _ch = _cell_size()
    w = _w_units(size)
    assert _card_size(size)[0] == w * cw + (w - 1) * _GRID_GAP


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_height_hugs_the_plot_at_every_size(size):
    """Card HEIGHT == chrome + THIS size's figure height, with ZERO blank padding below the plot
    (the bottom edge hugs the plot) -- for EVERY size, not only 1x2."""
    _cw, fh = panel_display_size(size)
    assert _card_size(size)[1] == _chrome_h() + fh           # slack == 0 -> hug


def test_taller_sizes_stay_distinct():
    """A taller preset (more rows) gives a TALLER card (the figure still scales with size); width
    only grows with columns -- so 1x2 / 2x2 / 4x4 remain visually different."""
    assert _card_size("2x2")[1] > _card_size("1x2")[1]       # 2x2 hugs a taller figure
    assert _card_size("4x4")[1] > _card_size("2x2")[1]
    assert _card_size("1x4")[0] > _card_size("1x2")[0]       # 1x4 is wider (more columns)


def test_vertical_layout_is_pixels_columns_snap():
    """``col`` snaps to the column grid; ``row`` is a pixel y (round-trips through the drop snap)."""
    pitch_x = _cell_size()[0] + _GRID_GAP
    x, y = _slot_to_pos(120, 2)                              # row=120 px, col=2
    assert x == _GRID_GAP + 2 * pitch_x and y == 120
    # a dropped pixel position resolves to (snapped y, col); col is exact, y lands on the quantum
    row, col = _pos_to_slot(x, 121)
    assert col == 2 and row >= _GRID_GAP and (row - _GRID_GAP) % _GRID_GAP == 0


def test_compact_pushes_overlap_down_by_actual_pixel_height():
    """Two cards in the same column that overlap: the lower one is pushed to just below the upper
    card's ACTUAL pixel height (+ one gap), since heights are variable -- no overlap remains."""
    from Zou_lab_control.frontend.task_console import PanelConfig
    a = PanelConfig(kind="1d", size="2x2", row=_GRID_GAP, col=0)        # taller card on top
    b = PanelConfig(kind="1d", size="1x2", row=_GRID_GAP, col=0)        # dropped overlapping it
    _compact([a, b], active=a)                                          # a pinned, b reflows
    assert b.row == a.row + _card_size(a.size)[1] + _GRID_GAP           # below A's real height
    assert not (a.row < b.row + _card_size(b.size)[1] and b.row < a.row + _card_size(a.size)[1])
