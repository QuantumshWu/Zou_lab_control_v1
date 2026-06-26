"""Contract (#H3r-F6): inter-panel vertical spacing is always an integer multiple of GRID_UNIT.

The user disliked the old "random" spacing: a card's pixel-y was snapped, but card heights vary, so
the GAP below a card was any leftover pixel value.  `_compact` now snaps every vertical inter-card
gap (the distance from one card's bottom to the next card's top, in the same column) to a clean
multiple of the single `GRID_UNIT` setting.  This guard fails if any placement path leaves a ragged
gap, or if the spacing stops scaling with GRID_UNIT.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Zou_lab_control.frontend.task_console as tc
from Zou_lab_control.frontend.task_console import (
    PANEL_KINDS, PanelConfig, _card_size, _columns_overlap, _compact, GRID_UNIT,
)

_KIND = sorted(PANEL_KINDS)[0]


def _stack(rows, col=0, size="2x2"):
    return [PanelConfig(kind=_KIND, row=r, col=col, size=size) for r in rows]


def _in_column_gaps(cfgs):
    cfgs = sorted(cfgs, key=lambda c: c.row)
    gaps = []
    for a, b in zip(cfgs, cfgs[1:]):
        if _columns_overlap(a, b):
            gaps.append(b.row - (a.row + _card_size(a.size)[1]))
    return gaps


def test_vertical_gaps_are_multiples_of_grid_unit():
    cfgs = _stack([5, 137, 281, 410])      # arbitrary pixel rows in one column
    _compact(cfgs)
    gaps = _in_column_gaps(cfgs)
    assert gaps, "expected stacked cards to share a column"
    for gap in gaps:
        assert gap >= 0, "cards must not overlap"
        assert gap % GRID_UNIT == 0, f"inter-panel gap {gap} is not a multiple of GRID_UNIT={GRID_UNIT}"


def test_two_columns_are_independent_but_each_tidy():
    cfgs = _stack([10, 200], col=0) + _stack([55, 333], col=2)
    _compact(cfgs)
    for gap in _in_column_gaps(cfgs):
        assert gap % GRID_UNIT == 0


def test_changing_grid_unit_rescales_spacing(monkeypatch):
    # GRID_UNIT is the ONE knob: snap to it and every gap becomes a multiple of the new value.
    monkeypatch.setattr(tc, "GRID_UNIT", 25)
    cfgs = _stack([7, 190, 360])
    _compact(cfgs)
    cfgs.sort(key=lambda c: c.row)
    for a, b in zip(cfgs, cfgs[1:]):
        if _columns_overlap(a, b):
            gap = b.row - (a.row + _card_size(a.size)[1])
            assert gap % 25 == 0, f"gap {gap} not a multiple of the retuned unit 25"


def test_snap_gap_rounds_to_unit_and_floors_at_one_unit():
    assert tc._snap_gap(0) == GRID_UNIT
    assert tc._snap_gap(GRID_UNIT * 3) == GRID_UNIT * 3
    assert tc._snap_gap(GRID_UNIT * 2 + 1) == GRID_UNIT * 2     # rounds down
    assert tc._snap_gap(-5) == GRID_UNIT                        # never negative/zero
