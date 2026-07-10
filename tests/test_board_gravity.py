"""Contract (#2): the Monitor board is an ORDER-DRIVEN top-left gravity packer.

The user dropped the column grid AND (this round) the path-dependent LOCAL gravity that let a card
rest wherever it was dropped -- which left middle holes and reflowed surprisingly on a click.  Now the
board is a single ordered list; :func:`pack` places each card, IN LIST ORDER, at the first free
top-most-then-left-most GAP-clear slot.  Placement is a PURE function of the order (the layout's single
source of truth), the sizes, and the board width -- it never reads a card's current pixel position.

These guards pin the properties the board must hold:
  (a) no two card AABBs overlap;
  (b) every neighbour and the board edges are separated by exactly ``GAP`` (the top-left-most card
      sits at ``(GAP, GAP)``);
  (c) determinism / idempotence -- packing a settled board again is a fixed point (returns False),
      and the same order packs to the same layout every time;
  (d) strict NW gravity -- a card is placed at the FIRST free slot, so an appended (Add) card lands
      in the next bottom slot and holes are filled, never left (the #2 fix);
  (e) a DRAG reorders the card (:func:`drop_index`): a drop onto a card's slot DISPLACES it (insert
      before it), a drop past the last card appends to the bottom; pack then recomputes the pixels.
A fuzz sweep asserts (a)+(b)+(c) hold for random orders/sizes/board widths.
Pure functions (no Qt window beyond the QApplication ``scaled_px`` needs), so this runs in the suite.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
from Zou_lab_control.frontend.task_console import (
    GAP, PanelConfig, _aabb, _card_size, _board_width, _min_board_width, drop_index, pack,
)

ensure_qt_app()                       # _card_size reads scaled_px (needs the QApplication)

SIZES = ("1x2", "2x2", "1x4")


def _cfg(col=GAP, row=GAP, size="2x2", kind="1d", title="p"):
    return PanelConfig(kind=kind, title=title, col=col, row=row, size=size)


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


def _one_col_width(size):
    """A board width that fits exactly ONE card of ``size`` (+ both GAP margins), forcing a stack."""
    return _card_size(size)[0] + 2 * GAP


# ----------------------------------------------------------------------------- (b) top-left origin
def test_single_card_lands_at_gap_gap():
    """A lone card packs to exactly the top-left origin ``(GAP, GAP)`` regardless of its seed pixel."""
    c = _cfg(500, 700, size="2x2")
    pack([c])
    assert (c.col, c.row) == (GAP, GAP)


def test_first_in_order_is_the_top_left_card():
    """The FIRST card in order is always at ``(GAP, GAP)`` -- pack ignores seed pixels and lays out by
    order, so the board margin is a uniform GAP from the origin."""
    cfgs = [_cfg(200, 9), _cfg(40, 333), _cfg(600, 120)]
    pack(cfgs)
    assert (cfgs[0].col, cfgs[0].row) == (GAP, GAP)


# ----------------------------------------------------------------------------- (a)+(b) gaps uniform
def test_no_overlap_and_neighbours_exactly_one_gap():
    """Several cards pack with NO overlap and every adjacent pair separated by exactly ``GAP`` on the
    axis they neighbour on (the packer never leaves a ragged gap)."""
    cfgs = [_cfg(size="2x2"), _cfg(size="1x2"), _cfg(size="1x4"),
            _cfg(size="2x2"), _cfg(size="1x2")]
    pack(cfgs, board_w=3 * _card_size("1x4")[0])
    _assert_no_overlap(cfgs)
    seen_neighbour = False
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            g = _clear_gap(a, b)
            if g is not None and 0 <= g < 4 * GAP:     # an actual touching neighbour
                seen_neighbour = True
                assert g == GAP, f"neighbour gap {g} != GAP {GAP}: {_aabb(a)} vs {_aabb(b)}"
    assert seen_neighbour, "expected at least one touching neighbour pair"


def test_two_stacked_cards_one_gap_apart():
    """Two same-width cards on a ONE-card-wide board stack with exactly one GAP between them."""
    a, b = _cfg(size="2x2"), _cfg(size="2x2")
    pack([a, b], board_w=_one_col_width("2x2"))   # board fits only one column -> they stack
    top, bot = sorted([a, b], key=lambda c: c.row)
    assert top.row == GAP
    assert bot.row == top.row + _card_size(top.size)[1] + GAP


# ----------------------------------------------------------------------------- (c) determinism
def test_packing_is_a_fixed_point():
    """Packing a settled board again moves nothing (returns False) -- the layout is a fixed point."""
    cfgs = [_cfg(size="2x2"), _cfg(size="1x2"), _cfg(size="2x2"), _cfg(size="1x4")]
    pack(cfgs)                                         # first pass settles it
    before = [(c.col, c.row) for c in cfgs]
    assert pack(cfgs) is False                         # second pass is a no-op
    assert [(c.col, c.row) for c in cfgs] == before


def test_deterministic_same_order_same_output():
    """Two independent boards with the SAME ORDER pack to identical layouts, whatever the seed cols."""
    a = [_cfg(300, 5, "2x2"), _cfg(20, 260, "1x2"), _cfg(640, 88, "2x2"), _cfg(120, 500, "1x4")]
    b = [_cfg(9, 9, "2x2"), _cfg(0, 0, "1x2"), _cfg(1, 1, "2x2"), _cfg(2, 2, "1x4")]   # same order, other pixels
    pack(a, board_w=4000)
    pack(b, board_w=4000)
    assert [(c.col, c.row) for c in a] == [(c.col, c.row) for c in b]


# ----------------------------------------------------------------------------- (d) strict NW gravity
def test_appended_card_lands_at_the_bottom_never_a_middle_hole():
    """#2: with the middle card of a one-column stack removed (a hole), an APPENDED (Add) card lands
    at the BOTTOM -- pack fills in order, so the new card is placed after the survivors, never floated
    into the middle hole the old local gravity would have dropped it into."""
    board_w = _one_col_width("1x2")
    a, b, c = _cfg(size="1x2", title="a"), _cfg(size="1x2", title="b"), _cfg(size="1x2", title="c")
    pack([a, b, c], board_w)                           # a,b,c stacked top->bottom
    order = [a, c, _cfg(size="1x2", title="NEW")]      # drop 'b' (a hole opens), APPEND a new card
    pack(order, board_w)
    new = order[-1]
    assert new.row == max(x.row for x in order), "the appended card is at the bottom, not the hole"


def test_narrow_viewport_reflows_to_single_column():
    """A wide board packs two 1x2 cards side by side; narrowing to one-card width re-packs them into a
    single stacked column (pack honours board_w, clamped up to one-card-wide)."""
    two = [_cfg(size="1x2", title="a"), _cfg(size="1x2", title="b")]
    pack(two, board_w=4000)
    assert len({c.col for c in two}) == 2, "wide board packs the two cards side by side"
    pack(two, board_w=_min_board_width(two))
    assert len({c.col for c in two}) == 1 and len({c.row for c in two}) == 2, "narrow board stacks them"


def test_resizing_one_card_does_not_disturb_the_other_column():
    """#H4c: growing ONE card must not scramble a card in a DIFFERENT column.  Order [TL, TR, BL] on a
    two-column board: TL top-left, TR top-right, BL under TL.  Growing TR (taller) re-packs by the SAME
    order, so TL and BL (left column, placed before/after TR by order) keep their slots."""
    w1, h1 = _card_size("1x2")
    board_w = GAP + _card_size("2x2")[0] + GAP + _card_size("2x2")[0] + GAP
    TL, TR, BL = _cfg(size="1x2", title="TL"), _cfg(size="1x2", title="TR"), _cfg(size="1x2", title="BL")
    cfgs = [TL, TR, BL]
    pack(cfgs, board_w=board_w)
    tl0, bl0 = (TL.col, TL.row), (BL.col, BL.row)
    TR.size = "2x2"                                     # user enlarges the top-right card
    moved = pack(cfgs, board_w=board_w)
    assert (TL.col, TL.row) == tl0 and (BL.col, BL.row) == bl0, "the left column is undisturbed"
    assert (TR.col, TR.row) == (GAP + w1 + GAP, GAP), "the grown card keeps its own column/top"
    assert moved is False
    _assert_no_overlap(cfgs)


# ----------------------------------------------------------------------------- (e) drop reorders
def test_drop_onto_a_cards_slot_displaces_it():
    """A drop with its top-left ON another card's slot returns THAT card's ORDER index, so inserting
    the dragged card there displaces the occupant down the order (and pack re-lays it out below)."""
    others = [_cfg(size="2x2", title="A"), _cfg(size="2x2", title="B")]
    board_w = _board_width(others + [_cfg()])
    pack(others, board_w)                              # A at origin, B beside it
    drop = _cfg(col=others[1].col + 3, row=others[1].row + 3, size="2x2")
    assert drop_index(drop, others, board_w) == 1, "dropping onto B's slot inserts before B (displaces)"
    drop2 = _cfg(col=others[0].col + 2, row=others[0].row + 2, size="2x2")
    assert drop_index(drop2, others, board_w) == 0, "dropping onto A's slot inserts before A"


def test_drop_below_everything_appends():
    """A drop below the last card returns the END index -- the dragged card appends to the bottom."""
    others = [_cfg(size="2x2", title="A"), _cfg(size="2x2", title="B")]
    board_w = _board_width(others + [_cfg()])
    pack(others, board_w)
    _bw, ch = _card_size("2x2")
    drop = _cfg(col=GAP, row=max(c.row for c in others) + 5 * ch, size="2x2")
    assert drop_index(drop, others, board_w) == len(others)


def test_drop_reorder_then_pack_keeps_the_contract():
    """The full drop pipeline: reorder by drop_index then pack -> still no overlaps, still GAP-clean,
    and the displaced card sinks below the card that took its slot (one column, clean vertical stack)."""
    board_w = _one_col_width("2x2")
    A = _cfg(size="2x2", title="A")
    C = _cfg(size="2x2", title="C")
    pack([A, C], board_w)                              # A on top, C below
    B = _cfg(col=A.col + 6, row=A.row + 4, size="2x2", title="B")   # dropped onto A's slot
    order = [A, C]
    idx = drop_index(B, order, board_w)
    order.insert(idx, B)
    pack(order, board_w)
    _assert_no_overlap(order)
    top, mid, bot = sorted(order, key=lambda c: c.row)
    assert top is B, "the dropped card took the top slot A occupied"
    h = _card_size("2x2")[1]
    assert mid.row == top.row + h + GAP and bot.row == mid.row + h + GAP, "the rest stack one GAP apart"


# ----------------------------------------------------------------------------- fuzz property sweep
def test_fuzz_pack_is_always_clean_and_idempotent():
    """Random orders / sizes / board widths: pack ALWAYS yields no overlaps, a top-left-origin first
    card, GAP-clean touching neighbours, and is idempotent (a second pass is a fixed point)."""
    rng = random.Random(20260710)
    widest = _card_size("1x4")[0]
    for _ in range(200):
        n = rng.randint(1, 8)
        cfgs = [_cfg(size=rng.choice(SIZES), title=str(i)) for i in range(n)]
        board_w = rng.choice([_one_col_width("1x4"), widest + 2 * GAP + 40,
                              2 * widest + 3 * GAP, 4000])
        pack(cfgs, board_w)
        _assert_no_overlap(cfgs)
        assert (cfgs[0].col, cfgs[0].row) == (GAP, GAP)
        for i, a in enumerate(cfgs):
            for b in cfgs[i + 1:]:
                g = _clear_gap(a, b)
                if g is not None and 0 <= g < 2 * GAP:
                    assert g == GAP, f"ragged neighbour gap {g}"
        assert pack(cfgs, board_w) is False, "a settled board is a fixed point"


# ----------------------------------------------------------------------------- (#H4c) save composite
def test_save_composite_fills_a_hidpi_grab_with_no_white_margin():
    """#H4c save: the monitor 'Save image' composites the grabbed board onto an opaque white canvas.  On
    a SCALED display the grab is a HiDPI pixmap (physical = logical × dpr); the canvas MUST carry that
    dpr or the pixmap paints at its smaller LOGICAL size into the top-left and leaves a giant blank
    margin (the bug).  Build a fully-RED dpr=2 pixmap, composite, and assert the BOTTOM-RIGHT pixel is
    red -- i.e. the content fills the whole image, no margin."""
    from PyQt5 import QtGui, QtWidgets
    from Zou_lab_control.frontend.task_console import _opaque_white_composite
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pm = QtGui.QPixmap(40, 20)                  # physical 40x20 ...
    pm.setDevicePixelRatio(2.0)                 # ... = logical 20x10 at dpr 2 (a scaled-display grab)
    pm.fill(QtGui.QColor("#FF0000"))
    out = _opaque_white_composite(pm)
    assert out.size() == pm.size()                          # same physical pixels
    assert out.devicePixelRatio() == pm.devicePixelRatio()  # dpr carried -> logical sizes match
    img = out.toImage()
    br = QtGui.QColor(img.pixel(img.width() - 1, img.height() - 1)).getRgb()[:3]
    assert br == (255, 0, 0), f"bottom-right is {br}, not red -> the HiDPI composite left a blank margin"


def test_first_free_slot_tiles_the_top_row_then_wraps():
    """pack's per-card placer (:func:`_first_free_slot`) TILES the board: fills the top row
    left-to-right, then wraps to the next shelf -- the seed that makes an ordered board fill the width."""
    from Zou_lab_control.frontend.task_console import _first_free_slot
    w, h = _card_size("1x2")
    board_w = 2 * w + 3 * GAP                          # exactly two columns fit
    placed = []
    spots = []
    for _ in range(4):                                 # place four 1x2 panels in order
        cfg = _cfg(size="1x2")
        cfg.col, cfg.row = _first_free_slot(cfg, placed, board_w)
        spots.append((cfg.col, cfg.row)); placed.append(cfg)
    assert spots[0] == (GAP, GAP)                       # first -> origin
    assert spots[1] == (GAP + w + GAP, GAP)             # second -> top row, second column (TILES, not stacks)
    assert spots[2] == (GAP, GAP + h + GAP)             # third -> wraps to row 2, column 1
    assert spots[3] == (GAP + w + GAP, GAP + h + GAP)   # fourth -> row 2, column 2
