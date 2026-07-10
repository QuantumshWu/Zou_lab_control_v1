"""Contract: a panel card's WIDTH scales with the size (``cols // 2`` base widths joined by ``GAP``,
so 1x4 is wider than 1x2) but its HEIGHT HUGS the plot -- the card is exactly tall enough for its
figure + chrome, with NO blank padding below (every size hugs like 1x2, #H3i-3).  Because heights
are therefore VARIABLE, the board is a pure pixel plane: ``col``/``row`` are the card's pixel
top-left and ``pack`` gravity-packs them (no column grid, #H3s-F8).

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
    GAP, PanelConfig, _aabb, _card_size, _cell_size, pack)

ensure_qt_app()                       # the cell size reads scaled_px (needs the QApplication)


def _chrome_h() -> int:
    return scaled_px(CARD_TITLE_PX) + scaled_px(2) + CARD_PAD


def _w_units(size):
    return max(1, panel_size_cells(size)[1] // 2)


def test_one_by_two_is_exactly_one_base_width():
    """The base 1x2 panel IS one width unit, and that unit hugs the plot (figure + thin border)."""
    cw, ch = _cell_size()
    assert _card_size("1x2") == (cw, ch)
    assert cw == panel_display_size("1x2")[0] + 2 * CARD_PAD   # base width hugs the figure L/R


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_width_is_whole_base_widths(size):
    """Card WIDTH == w_units base widths + the GAP between them (the clean horizontal rhythm)."""
    cw, _ch = _cell_size()
    w = _w_units(size)
    assert _card_size(size)[0] == w * cw + (w - 1) * GAP


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


def _one_col(size="2x2"):
    """A board width that fits exactly ONE card of ``size`` (+ both GAP margins), forcing a stack."""
    return _card_size(size)[0] + 2 * GAP


def test_board_is_pixel_plane_col_and_row_are_pixels():
    """``col``/``row`` ARE the card's pixel top-left -- on a one-card-wide board a freshly placed
    pair packs to ``(GAP, GAP)`` and just below it (no column grid, no row snap; pure pixels)."""
    a = PanelConfig(kind="1d", size="2x2", col=0, row=0)
    b = PanelConfig(kind="1d", size="2x2", col=0, row=5)
    pack([a, b], board_w=_one_col())                     # one column -> they stack
    top, bot = sorted([a, b], key=lambda c: c.row)
    assert (top.col, top.row) == (GAP, GAP)                  # top-left margin = GAP
    assert bot.row == top.row + _card_size(top.size)[1] + GAP  # next one exactly one GAP below


def testpack_pushes_overlap_down_by_actual_pixel_height():
    """Two same-column cards on a one-card-wide board: the lower one ends up just below the upper
    card's ACTUAL pixel height + one GAP, since heights are variable -- no overlap remains."""
    a = PanelConfig(kind="1d", size="2x2", col=0, row=0)        # taller card on top
    b = PanelConfig(kind="1d", size="1x2", col=0, row=5)        # dropped overlapping it
    pack([a, b], board_w=_one_col())                        # one column -> they stack
    top = min([a, b], key=lambda c: c.row)
    other = b if top is a else a
    assert other.row == top.row + _card_size(top.size)[1] + GAP    # below the top card's real height
    ax0, ay0, ax1, ay1 = _aabb(a)
    bx0, by0, bx1, by1 = _aabb(b)
    assert not (ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1)   # no overlap
