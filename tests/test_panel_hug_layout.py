"""Contract: panel cards tile on a CLEAN GRID whose cell is the base 1x2 panel.

Every preset is a whole number of cells -- 1x2 = 1x1, 2x2 = 1 wide x 2 tall, 1x4 = 2 wide x 1
tall, 2x4 = 2x2, 4x4 = 2 wide x 4 tall -- so a card's size is an exact whole-cell multiple and
the only drag-snap points are whole-cell positions.  Cards therefore tile with EXACTLY one
_GRID_GAP between them (no slot-rounding slack).  The 1x2 cell hugs the plot; multi-cell cards
carry blank padding (the plot's fixed margins cannot be subdivided to fill the extra cells).

The card's FORMAT (rounded corners / shadow / grey title strip / content padding) is owned by
the FluentGroupBox component (qt_fluent.CARD_PAD / CARD_TITLE_PX); this module only sizes the
cells.  Pure functions (no Qt window) so this runs in the normal suite.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.frontend.live import PANEL_SIZES, panel_display_size, panel_size_cells
from Zou_lab_control.frontend.qt_fluent import CARD_PAD, ensure_qt_app
from Zou_lab_control.frontend.task_console import _GRID_GAP, _card_size, _cell_size

ensure_qt_app()                       # the cell size reads scaled_px (needs the QApplication)


def _units(size):
    rows, cols = panel_size_cells(size)
    return cols // 2, rows                       # (w_units, h_units) in whole cells


def test_one_by_two_is_exactly_one_cell():
    """The base 1x2 panel IS one grid cell, and the cell hugs the plot (figure + thin border)."""
    cw, ch = _cell_size()
    assert _card_size("1x2") == (cw, ch)
    assert cw == panel_display_size("1x2")[0] + 2 * CARD_PAD   # cell width hugs the figure L/R


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_card_is_a_whole_number_of_cells(size):
    """Card size == w_units cells wide x h_units cells tall, plus the gaps between cells."""
    cw, ch = _cell_size()
    w_units, h_units = _units(size)
    assert _card_size(size) == (w_units * cw + (w_units - 1) * _GRID_GAP,
                                h_units * ch + (h_units - 1) * _GRID_GAP)


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_card_always_holds_its_plot(size):
    """The card is never smaller than the design-size plot it holds (so the plot never clips);
    any surplus is the blank padding that fills the extra cells."""
    cw, ch = _card_size(size)
    fw, fh = panel_display_size(size)
    assert cw >= fw and ch >= fh


def test_cells_tile_with_exactly_one_gap():
    """Stacking / abutting cards leaves EXACTLY _GRID_GAP between them -- a 2-cell card spans the
    same pixels as two 1-cell cards plus the gap, so there is no rounding slack."""
    cw, ch = _cell_size()
    assert _card_size("2x2")[1] == 2 * ch + _GRID_GAP        # 2 tall == two stacked cells + gap
    assert _card_size("1x4")[0] == 2 * cw + _GRID_GAP        # 2 wide == two side cells + gap
    assert _card_size("4x4") == (2 * cw + _GRID_GAP, 4 * ch + 3 * _GRID_GAP)


def test_cell_span_matches_the_preset():
    """PanelConfig.rows/cols report the WHOLE-CELL span the layout packs in (rows tall, cols//2
    wide) -- the gravity/overlap layout works in these cells."""
    from Zou_lab_control.frontend.task_console import PanelConfig
    for size in PANEL_SIZES:
        w_units, h_units = _units(size)
        cfg = PanelConfig(kind="1d", size=size)
        assert (cfg.rows, cfg.cols) == (h_units, w_units)
