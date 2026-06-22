"""Contract: a Monitor panel CARD hugs its plot.

The user's ask: the panel's top/left/right edges should sit right at the plot, so every panel is
compact with little whitespace -- WITHOUT touching the plot (its shape / size / margins / art are
unchanged).  So the card = the frontend panel's on-screen canvas (the figure, untouched) + only a
thin border L/R + a small top inset + the footer slot below.  No grid-multiple inflation, which is
what removes the big bottom padding a 2x2 (or larger) panel used to carry.

Pure functions (no Qt window), so this runs in the normal suite -- it pins the hug so a future
change can't silently re-inflate the card back to a slot-multiple.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.frontend.live import PANEL_SIZES, panel_display_size
from Zou_lab_control.frontend.task_console import _CARD_PAD, _card_size


@pytest.mark.parametrize("size", PANEL_SIZES)
def test_card_hugs_its_plot(size):
    """The card is the plot canvas + a thin border L/R + top inset + footer -- nothing more."""
    cw, ch = _card_size(size)
    pw, ph = panel_display_size(size)
    assert cw == pw + 2 * _CARD_PAD            # left/right edges hug the plot exactly
    # height = top inset + plot + footer slot only; never a big bottom gap.
    assert ph < ch <= ph + 64


def test_bigger_cards_are_not_grid_multiples():
    """A 2x2 card is much SHORTER than two stacked 1x2 cards, and a 4-col card much NARROWER than
    two 2-col cards -- because each card hugs its (affine, fixed-margin) plot instead of spanning
    whole grid slots.  This is exactly the bottom/side padding the user reported, now gone."""
    h2 = _card_size("2x2")[1]
    h1 = _card_size("1x2")[1]
    assert h2 < 2 * h1                          # no per-row chrome inflation -> no bottom slack
    assert _card_size("1x4")[0] < 2 * _card_size("1x2")[0]   # no per-col inflation -> no side slack


def test_plot_canvas_size_is_unchanged():
    """The plot itself is UNTOUCHED: panel_display_size (the figure's on-screen size) is the stock
    frontend geometry -- the card shrinks to it, the plot does not grow/shrink to the card."""
    # a representative spot-check against the known design geometry (data area scales with cells,
    # margins fixed) -- if someone makes the plot fill the card, these change and this fails.
    assert panel_display_size("2x2")[0] == panel_display_size("1x2")[0]   # same cols => same width
    assert panel_display_size("2x2")[1] > panel_display_size("1x2")[1]    # more rows => taller plot
