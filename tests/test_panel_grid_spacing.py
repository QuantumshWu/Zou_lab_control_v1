"""Contract (#H3s-F8): inter-panel spacing is a UNIFORM ``GAP`` on all four sides.

The user dropped the column grid (#H3r had a column grid + a vertical-gap snap).  The board is now a
pure pixel plane: ``pack`` gravity-packs cards top-left and the clear distance between any two
neighbours -- and the margin from the top-left origin -- is exactly the single ``GAP`` constant
(which equals the horizontal inter-card gap the user liked).  This guard fails if any placement path
leaves a ragged gap, or if the spacing stops scaling with the one ``GAP`` knob.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Zou_lab_control.frontend.task_console as tc
from Zou_lab_control.frontend.task_console import (
    GAP, PANEL_KINDS, PanelConfig, _aabb, _card_size, pack,
)

_KIND = sorted(PANEL_KINDS)[0]


def _stack(rows, col=0, size="2x2"):
    return [PanelConfig(kind=_KIND, row=r, col=col, size=size) for r in rows]


def _one_col(size="2x2"):
    """A board width that fits exactly ONE card of ``size`` (+ both GAP margins), forcing a stack."""
    return _card_size(size)[0] + 2 * GAP


def _stacked_gaps(cfgs):
    """Vertical gaps between cards that end up stacked in the same x-column after packing."""
    cfgs = sorted(cfgs, key=lambda c: c.row)
    gaps = []
    for a, b in zip(cfgs, cfgs[1:]):
        ax0, _ay0, ax1, ay1 = _aabb(a)
        bx0, by0, bx1, _by1 = _aabb(b)
        if ax0 < bx1 and bx0 < ax1:                 # share an x-column -> stacked
            gaps.append(by0 - ay1)
    return gaps


def test_stacked_gaps_are_exactly_one_gap():
    """Several cards on a one-card-wide board stack one ``GAP`` apart (uniform, never ragged)."""
    cfgs = _stack([5, 137, 281, 410])               # arbitrary pixel rows in one column
    pack(cfgs, board_w=_one_col())              # board fits one column -> they stack
    gaps = _stacked_gaps(cfgs)
    assert gaps, "expected the cards to stack in one column"
    for gap in gaps:
        assert gap == GAP, f"stacked gap {gap} != the uniform GAP {GAP}"


def test_top_left_margin_is_one_gap():
    """The top-most/left-most card sits exactly ``GAP`` from the (0, 0) origin on both axes."""
    cfgs = _stack([5, 137, 281])
    pack(cfgs)
    top_left = min(cfgs, key=lambda c: (c.row, c.col))
    assert (top_left.col, top_left.row) == (GAP, GAP)


def test_changing_gap_rescales_spacing(monkeypatch):
    """GAP is the ONE knob: retune it and every margin + inter-card gap follows."""
    monkeypatch.setattr(tc, "GAP", 25)
    cfgs = _stack([7, 190, 360])
    pack(cfgs, board_w=_one_col())              # one column -> stacked, gaps follow the new GAP
    top_left = min(cfgs, key=lambda c: (c.row, c.col))
    assert (top_left.col, top_left.row) == (25, 25)            # margin = retuned GAP
    for gap in _stacked_gaps(cfgs):
        assert gap == 25, f"stacked gap {gap} != the retuned GAP 25"


def test_side_by_side_cards_one_gap_apart():
    """Two cards the user PLACED in adjacent columns stay side by side, one ``GAP`` apart, both risen
    to the top margin.  Gravity pulls each UP/LEFT from where it is -- it does NOT auto-stack two
    distinct columns into one, nor fling either across the board (#1: no auto-relocation)."""
    w = _card_size("1x2")[0]
    cfgs = [PanelConfig(kind=_KIND, row=5 * GAP, col=GAP, size="1x2"),
            PanelConfig(kind=_KIND, row=9 * GAP, col=GAP + w + GAP, size="1x2")]   # adjacent columns
    pack(cfgs)
    cfgs.sort(key=lambda c: c.col)
    left, right = cfgs
    assert left.row == right.row == GAP
    assert right.col - (left.col + _card_size(left.size)[0]) == GAP


def test_demo_board_seeds_pixel_positions_from_the_geometry_single_source():
    """The 1480 px demo window starts as a two-column board, not six overlapping grid-number
    seeds that gravity turns into one tall column.  The seed is already a packing fixed point and
    every offset is derived from the real card AABBs plus GAP."""
    from Zou_lab_control.frontend.devtools import _demo_board_state

    board_width = 1480
    configs = _demo_board_state(board_width=board_width).panels
    assert pack(configs, board_w=board_width) is False

    by_title = {cfg.title: cfg for cfg in configs}
    left = by_title["Camera image"]
    right = by_title["Mean intensity (dist)"]
    assert (left.col, left.row) == (GAP, GAP)
    assert right.row == GAP
    assert right.col == left.col + _card_size(left.size)[0] + GAP

    for index, first in enumerate(configs):
        x0, y0, x1, y1 = _aabb(first)
        assert GAP <= x0 and x1 <= board_width - GAP
        for second in configs[index + 1:]:
            a0, b0, a1, b1 = _aabb(second)
            separated = x1 + GAP <= a0 or a1 + GAP <= x0 \
                or y1 + GAP <= b0 or b1 + GAP <= y0
            assert separated, (first.title, second.title)
