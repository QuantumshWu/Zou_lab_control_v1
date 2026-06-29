"""Contract (#H3s-F8): the Monitor board is a TOP-LEFT GRAVITY packer over pixel AABBs.

The user dropped the column grid: after any add / drag / resize every panel floats UP then LEFT
until blocked by another panel or the board edge, and the clear gap on ALL FOUR sides -- top,
bottom, left, right -- and the margin from the top-left origin is a UNIFORM ``GAP`` (the horizontal
inter-card gap the user liked).  ``_compact`` re-packs ``PanelConfig.col`` (pixel x) / ``row``
(pixel y) in place; there is NO column grid.

These guards pin the five properties the redesign must hold:
  (a) no two card AABBs overlap;
  (b) every neighbour and the board edges are separated by exactly ``GAP`` (a card with nothing
      above or left of it sits at ``(GAP, GAP)``);
  (c) determinism -- packing a settled board a second time is a fixed point (returns False);
  (d) stability -- moving one card and dropping it back reproduces the original layout;
  (e) a dragged ``active`` card wins its spot and the others reflow around it.
Pure functions (no Qt window beyond the QApplication ``scaled_px`` needs), so this runs in the suite.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
from Zou_lab_control.frontend.task_console import (
    GAP, PanelConfig, _aabb, _card_size, _compact, _min_board_width,
)


def test_narrow_viewport_reflows_to_single_column():
    """A narrowed viewport collapses a two-column layout into a single column: each card keeps its
    column under gravity, but the board-width clamp pulls every card's x back inside the board, so at
    one-card width both land in the one column and stack (the clamp must NOT ratchet to the cards'
    current right-extent, else narrowing could never re-pack)."""
    w = _card_size("1x2")[0]
    cfgs = [PanelConfig(kind="1d", col=GAP, row=GAP, size="1x2"),
            PanelConfig(kind="1d", col=GAP + w + GAP, row=GAP, size="1x2")]   # two adjacent columns
    _compact(cfgs, board_w=4000)                          # wide: the two distinct columns stay put
    assert len({c.col for c in cfgs}) == 2, "wide board keeps the two columns the user placed"
    _compact(cfgs, board_w=_min_board_width(cfgs))        # narrow to one-card width
    assert len({c.col for c in cfgs}) == 1, "narrow board clamps both cards into a single column"
    assert len({c.row for c in cfgs}) == 2, "and they stack vertically"

ensure_qt_app()                       # _card_size reads scaled_px (needs the QApplication)


def _cfg(col, row, size="2x2", kind="1d"):
    return PanelConfig(kind=kind, col=col, row=row, size=size)


def _overlap(a, b) -> bool:
    ax0, ay0, ax1, ay1 = _aabb(a)
    bx0, by0, bx1, by1 = _aabb(b)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _assert_no_overlap(cfgs):
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            assert not _overlap(a, b), f"cards overlap: {_aabb(a)} vs {_aabb(b)}"


def _clear_gap(a, b) -> int | None:
    """The clear separation between two non-overlapping AABBs along whichever axis they are
    separated on (the axis on which they DON'T project-overlap); None if they share neither axis."""
    ax0, ay0, ax1, ay1 = _aabb(a)
    bx0, by0, bx1, by1 = _aabb(b)
    x_proj = ax0 < bx1 and bx0 < ax1
    y_proj = ay0 < by1 and by0 < ay1
    if y_proj and not x_proj:                       # side-by-side -> horizontal gap
        return bx0 - ax1 if bx0 >= ax1 else ax0 - bx1
    if x_proj and not y_proj:                       # stacked -> vertical gap
        return by0 - ay1 if by0 >= ay1 else ay0 - by1
    return None


# ----------------------------------------------------------------------------- (b) top-left origin
def test_single_card_lands_at_gap_gap():
    """A lone card -- nothing above or left -- packs to exactly the top-left origin ``(GAP, GAP)``."""
    c = _cfg(500, 700, size="2x2")
    _compact([c])
    assert (c.col, c.row) == (GAP, GAP)


def test_top_card_sits_at_gap_margin():
    """The TOP-MOST/LEFT-MOST card of any board is at ``(GAP, GAP)`` -- the board margin is GAP."""
    cfgs = [_cfg(200, 9), _cfg(40, 333), _cfg(600, 120)]
    _compact(cfgs)
    top_left = min(cfgs, key=lambda c: (c.row, c.col))
    assert (top_left.col, top_left.row) == (GAP, GAP)


# ----------------------------------------------------------------------------- (a)+(b) gaps uniform
def test_no_overlap_and_neighbours_exactly_one_gap():
    """Scatter several cards, pack, and assert NO overlap and that every adjacent pair is separated
    by exactly ``GAP`` on the axis they neighbour on (the packer never leaves a ragged gap)."""
    cfgs = [_cfg(300, 5, "2x2"), _cfg(20, 260, "1x2"), _cfg(640, 88, "1x4"),
            _cfg(120, 500, "2x2"), _cfg(700, 410, "1x2")]
    _compact(cfgs)
    _assert_no_overlap(cfgs)
    seen_neighbour = False
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            g = _clear_gap(a, b)
            if g is not None and g >= 0 and g < 4 * GAP:     # an actual touching neighbour
                seen_neighbour = True
                assert g == GAP, f"neighbour gap {g} != GAP {GAP}: {_aabb(a)} vs {_aabb(b)}"
    assert seen_neighbour, "expected at least one touching neighbour pair"


def _one_col_width(size):
    """A board width that fits exactly ONE card of ``size`` (+ both GAP margins), forcing a stack."""
    return _card_size(size)[0] + 2 * GAP


def test_two_stacked_cards_one_gap_apart():
    """Two same-width cards on a ONE-card-wide board stack with exactly one GAP between them."""
    a = _cfg(0, 0, "2x2")
    b = _cfg(0, 5, "2x2")
    _compact([a, b], board_w=_one_col_width("2x2"))   # board fits only one column -> they stack
    top, bot = sorted([a, b], key=lambda c: c.row)
    assert top.row == GAP
    assert bot.row == top.row + _card_size(top.size)[1] + GAP


# ----------------------------------------------------------------------------- (c) determinism
def test_packing_is_a_fixed_point():
    """Packing a settled board again moves nothing (returns False) -- the layout is a fixed point."""
    cfgs = [_cfg(300, 5, "2x2"), _cfg(20, 260, "1x2"), _cfg(640, 88, "2x2"), _cfg(120, 500, "1x4")]
    assert _compact(cfgs) in (True, False)            # first pass settles it
    before = [(c.col, c.row) for c in cfgs]
    assert _compact(cfgs) is False                    # second pass is a no-op
    assert [(c.col, c.row) for c in cfgs] == before


def test_deterministic_same_input_same_output():
    """Two independent boards with the same seed positions pack to identical layouts."""
    seeds = [(300, 5, "2x2"), (20, 260, "1x2"), (640, 88, "2x2"), (120, 500, "1x4")]
    a = [_cfg(*s) for s in seeds]
    b = [_cfg(*s) for s in seeds]
    _compact(a)
    _compact(b)
    assert [(c.col, c.row) for c in a] == [(c.col, c.row) for c in b]


# ----------------------------------------------------------------------------- (d) stability
def test_drop_back_reproduces_layout():
    """Settle a board, move ONE card and drop it back at its settled spot (as the active card):
    the whole layout reproduces -- dropping a card where it was must NOT reshuffle the others."""
    cfgs = [_cfg(300, 5, "2x2"), _cfg(20, 260, "1x2"), _cfg(640, 88, "2x2"), _cfg(120, 500, "1x4")]
    _compact(cfgs)
    settled = [(c.col, c.row) for c in cfgs]
    mover = cfgs[1]
    # "pick up and drop back" = re-seed it at its own settled position and re-pack with it active.
    mover.col, mover.row = settled[1]
    moved = _compact(cfgs, active=mover)
    assert moved is False
    assert [(c.col, c.row) for c in cfgs] == settled


# ----------------------------------------------------------------------------- (e) active wins
def test_active_card_wins_a_contested_slot():
    """A dragged ``active`` card dropped ONTO another card's slot WINS it (sorts first on the tie),
    so the other card reflows around it instead of the active card yielding."""
    a = _cfg(400, 400, "2x2")
    b = _cfg(0, 0, "2x2")
    c = _cfg(10, 10, "2x2")
    a.col, a.row = 0, 0                               # drag ``a`` onto the very top-left
    _compact([a, b, c], active=a)
    assert (a.col, a.row) == (GAP, GAP)              # active claimed the origin
    _assert_no_overlap([a, b, c])
    assert (b.col, b.row) != (GAP, GAP)              # b and c reflowed off the origin (a took it)
    assert (c.col, c.row) != (GAP, GAP)


def test_dragged_card_snaps_up_left():
    """A card dropped LOW-RIGHT on an otherwise-empty board snaps UP-LEFT to the origin -- the whole
    point of gravity packing (the dragged card floats up + left until blocked, not pinned)."""
    a = _cfg(700, 600, "2x2")
    _compact([a], active=a)
    assert (a.col, a.row) == (GAP, GAP)


# ------------------------------------------------------- (#2) NW gravity is BLOCKED, never teleports
def test_drag_to_bottom_left_stays_below_the_top_left_card():
    """#2: a card dropped at the BOTTOM-LEFT, with a panel directly ABOVE it and the LEFT BOUNDARY on
    its left, STAYS bottom-left -- the NW gravity is blocked by the panel above and the boundary, so it
    must NOT be flung up-and-over to the empty TOP-RIGHT slot.  (The candidate-sweep packer wrongly
    teleported it to the first free top-most slot = top-right; anchored local gravity keeps it put.)"""
    w, h = _card_size("1x2")
    board_w = 2 * w + 3 * GAP                          # room for TWO columns (so a top-right gap exists)
    top = _cfg(GAP, GAP, "1x2")
    drop = _cfg(GAP, GAP + h + 50, "1x2")              # the user drops it just below the top-left card
    _compact([top, drop], active=drop, board_w=board_w)
    assert (drop.col, drop.row) == (GAP, GAP + h + GAP)   # stays bottom-left, one GAP under the top card
    assert drop.row > GAP                                 # NOT risen to the top row
    assert drop.col == GAP                                # NOT slid right to the top-right gap


def test_first_free_slot_tiles_the_top_row_then_wraps():
    """A freshly-Added panel SPAWNS at the first free top-left slot (``_first_free_slot``) so an Add
    TILES the board -- fills the top row left-to-right, then wraps -- instead of starting a fresh column.
    (Local NW gravity then keeps it there; this is the seed that makes Add fill the width.)"""
    from Zou_lab_control.frontend.task_console import _first_free_slot
    w, h = _card_size("1x2")
    board_w = 2 * w + 3 * GAP                          # exactly two columns fit
    placed = []
    spots = []
    for _ in range(4):                                 # add four 1x2 panels one at a time
        cfg = _cfg(GAP, GAP, "1x2")
        cfg.col, cfg.row = _first_free_slot(cfg, placed, board_w)
        spots.append((cfg.col, cfg.row)); placed.append(cfg)
    assert spots[0] == (GAP, GAP)                       # first -> origin
    assert spots[1] == (GAP + w + GAP, GAP)             # second -> top row, second column (TILES, not stacks)
    assert spots[2] == (GAP, GAP + h + GAP)             # third -> wraps to row 2, column 1
    assert spots[3] == (GAP + w + GAP, GAP + h + GAP)   # fourth -> row 2, column 2
