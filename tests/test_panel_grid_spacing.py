"""Contract (#H3s-F8): inter-panel spacing is a UNIFORM ``GAP`` on all four sides.

The user dropped the column grid (#H3r had a column grid + a vertical-gap snap).  The board is now a
pure pixel plane: ``_compact`` gravity-packs cards top-left and the clear distance between any two
neighbours -- and the margin from the top-left origin -- is exactly the single ``GAP`` constant
(which equals the horizontal inter-card gap the user liked).  This guard fails if any placement path
leaves a ragged gap, or if the spacing stops scaling with the one ``GAP`` knob.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Zou_lab_control.frontend.task_console as tc
from Zou_lab_control.frontend.task_console import (
    GAP, PANEL_KINDS, PanelConfig, _aabb, _card_size, _compact,
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
    _compact(cfgs, board_w=_one_col())              # board fits one column -> they stack
    gaps = _stacked_gaps(cfgs)
    assert gaps, "expected the cards to stack in one column"
    for gap in gaps:
        assert gap == GAP, f"stacked gap {gap} != the uniform GAP {GAP}"


def test_top_left_margin_is_one_gap():
    """The top-most/left-most card sits exactly ``GAP`` from the (0, 0) origin on both axes."""
    cfgs = _stack([5, 137, 281])
    _compact(cfgs)
    top_left = min(cfgs, key=lambda c: (c.row, c.col))
    assert (top_left.col, top_left.row) == (GAP, GAP)


def test_changing_gap_rescales_spacing(monkeypatch):
    """GAP is the ONE knob: retune it and every margin + inter-card gap follows."""
    monkeypatch.setattr(tc, "GAP", 25)
    cfgs = _stack([7, 190, 360])
    _compact(cfgs, board_w=_one_col())              # one column -> stacked, gaps follow the new GAP
    top_left = min(cfgs, key=lambda c: (c.row, c.col))
    assert (top_left.col, top_left.row) == (25, 25)            # margin = retuned GAP
    for gap in _stacked_gaps(cfgs):
        assert gap == 25, f"stacked gap {gap} != the retuned GAP 25"


def test_side_by_side_cards_one_gap_apart():
    """Two narrow cards that DO fit side by side end up one ``GAP`` apart horizontally (the user's
    'horizontal minimal gap'), both at the top margin."""
    cfgs = _stack([5, 9], size="1x2")               # 1x2 is the base width -> two fit on one row
    _compact(cfgs)
    cfgs.sort(key=lambda c: c.col)
    left, right = cfgs
    assert left.row == right.row == GAP
    assert right.col - (left.col + _card_size(left.size)[0]) == GAP
