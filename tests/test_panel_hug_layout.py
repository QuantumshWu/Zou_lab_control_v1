"""Contract: a Monitor panel CARD keeps the FluentGroupBox chrome and is sized in proportion.

The card's DISPLAY / ART is the stock FluentGroupBox (rounded corners, drop shadow, the grey
QGroupBox title strip) -- unchanged.  Only the LAYOUT is sized here:

  * WIDTH  = the figure width + a thin L/R border (the drag grip).  The figure width has only a
    few finite values (one per column count), so panel widths are in proportion.
  * HEIGHT = the grey title strip + a plot region proportional to the preset's ROW count
    (the 1-row figure height x rows) + the bottom border.  The design-size canvas pins to the
    top of that region (right under the title strip) and the remainder is bottom padding, so a
    2-/4-row card is taller in proportion to its rows (no footer, no slot-multiple slack).

Pure functions (no Qt window), so this runs in the normal suite -- it pins the proportion
mechanically.  The chrome itself is exercised by the GUI smoke tests (shadow / modular).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.frontend.live import PANEL_SIZES, panel_display_size, panel_size_cells
# CARD_PAD / CARD_TITLE_PX are the card's FORMAT -- owned by the component library, not task_console.
from Zou_lab_control.frontend.qt_fluent import CARD_PAD, CARD_TITLE_PX, ensure_qt_app, scaled_px
from Zou_lab_control.frontend.task_console import _card_size

ensure_qt_app()                       # scaled_px needs the QApplication / fluent scale


def _expected_height(size: str) -> int:
    one_row_h = panel_display_size("1x2")[1]
    rows = panel_size_cells(size)[0]
    return scaled_px(CARD_TITLE_PX) + scaled_px(2) + one_row_h * rows + CARD_PAD


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_card_width_is_figure_plus_border(size):
    """Card width == the figure width + a thin L/R border (the plot hugs L/R)."""
    assert _card_size(size)[0] == panel_display_size(size)[0] + 2 * CARD_PAD


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_card_height_is_strip_plus_row_proportional(size):
    """Card height == the grey title strip + (1-row figure height x rows) + bottom border, so
    the plot region grows in proportion to the rows."""
    assert _card_size(size)[1] == _expected_height(size)


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_card_always_holds_its_canvas(size):
    """The card is tall enough to hold the design-size canvas below the title strip (never
    clips), with the leftover as bottom padding."""
    ch = _card_size(size)[1]
    fh = panel_display_size(size)[1]
    assert ch >= scaled_px(CARD_TITLE_PX) + fh


def test_plot_region_scales_linearly_with_rows():
    """The plot region (card height minus the fixed title-strip chrome) is exactly proportional
    to the rows: a 2-row card's region is 2x a 1-row card's, a 4-row card's is 4x."""
    chrome = scaled_px(CARD_TITLE_PX) + scaled_px(2) + CARD_PAD
    region = {s: _card_size(s)[1] - chrome for s in ("1x2", "2x2", "4x4")}
    assert region["2x2"] == 2 * region["1x2"]
    assert region["4x4"] == 4 * region["1x2"]


def test_plot_canvas_size_is_unchanged():
    """The plot itself is UNTOUCHED: panel_display_size (the figure's on-screen size) is the
    stock frontend geometry -- the card is sized around it, the plot is not stretched."""
    assert panel_display_size("2x2")[0] == panel_display_size("1x2")[0]   # same cols => same width
    assert panel_display_size("2x2")[1] > panel_display_size("1x2")[1]    # more rows => taller plot
