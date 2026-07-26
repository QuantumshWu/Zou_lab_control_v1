"""Pure north-west-gravity placement for a panel board.

The packer depends only on rectangle geometry.  Qt chrome and renderer size
policy enter through :class:`BoardMetrics`; this module imports neither UI nor
rendering backends.  ``card_size`` remains callable because the answer depends
on the current display scale and must not be captured as stale construction
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = ["BoardMetrics", "GeomProxy", "board_width", "drop_index",
           "first_free_slot", "min_board_width", "pack"]


@dataclass(frozen=True)
class BoardMetrics:
    """The two facts the packer cannot derive: the clear gap, and how big a card is.

    ``card_size`` maps a panel-size preset to the card's outer ``(width, height)`` in
    pixels.  It stays a callable so the answer always reflects the CURRENT display
    scale; see the module docstring for why a mapping would be wrong.
    """

    gap: int
    card_size: Callable[[str], "tuple[int, int]"]

    def __post_init__(self) -> None:
        if int(self.gap) != self.gap or self.gap < 0:
            raise ValueError("gap must be a non-negative whole number of pixels")
        if not callable(self.card_size):
            raise TypeError("card_size must be callable: a snapshot would go stale")


class GeomProxy:
    """A placement-only stand-in, so a trial pack never mutates a real card."""

    __slots__ = ("size", "col", "row")

    def __init__(self, size: str, col: int = 0, row: int = 0) -> None:
        self.size = size
        self.col = col
        self.row = row


def _aabb(cfg, metrics: BoardMetrics) -> tuple[int, int, int, int]:
    """The card's pixel AABB ``(x0, y0, x1, y1)`` -- top-left ``(col, row)`` plus its size."""
    w, h = metrics.card_size(cfg.size)
    return (cfg.col, cfg.row, cfg.col + w, cfg.row + h)


def _overlaps_with_gap(box: tuple[int, int, int, int], placed, metrics: BoardMetrics) -> bool:
    """True when ``box`` (an ``(x0, y0, x1, y1)`` AABB), EXPANDED by the gap on all sides,
    intersects any already-placed card.  Equivalently: the clear distance between ``box`` and a
    placed card is < gap on the axis where they overlap, so leaving a card exactly gap away
    counts as clear."""
    gap = metrics.gap
    x0, y0, x1, y1 = box
    for p in placed:
        px0, py0, px1, py1 = _aabb(p, metrics)
        if x0 < px1 + gap and px0 < x1 + gap and y0 < py1 + gap and py0 < y1 + gap:
            return True
    return False


def first_free_slot(cfg, placed, board_w: int, metrics: BoardMetrics) -> tuple[int, int]:
    """The TOP-MOST then LEFT-MOST free ``(col, row)`` where ``cfg`` fits clear of every ``placed``
    card (gap apart, inside ``board_w``) -- the per-card north-west placement :func:`pack` applies to
    EVERY card in order (so the board tiles the top row left-to-right, wraps to the next shelf, and
    never leaves a middle hole).  Candidate points are the gap (origin) plus each placed card's
    right/bottom edge (``+gap``) and its left/top edge (so a card can tuck under a wider one); swept
    by y then x, first feasible wins."""
    gap = metrics.gap
    w, _h = metrics.card_size(cfg.size)
    xs = {gap}
    ys = {gap}
    for p in placed:
        px0, py0, px1, py1 = _aabb(p, metrics)
        xs.add(px1 + gap)
        ys.add(py1 + gap)
        xs.add(px0)            # also align left edges, so a card can tuck under a wider one
        ys.add(py0)
    max_x = max(gap, board_w - gap - w)
    cand_x = sorted(x for x in xs if gap <= x <= max_x) or [gap]
    for y in sorted(ys):
        for x in cand_x:
            if not _overlaps_with_gap((x, y, x + w, y + _h), placed, metrics):
                return (x, y)
    # No candidate fit (should not happen -- placing past the lowest card always clears).
    bottom = max((py1 for *_rest, py1 in (_aabb(p, metrics) for p in placed)), default=0)
    return (gap, bottom + gap if placed else gap)


def min_board_width(configs: Sequence, metrics: BoardMetrics) -> int:
    """The NARROWEST a board may pack to: one WIDEST card plus both gap margins.  A viewport thinner
    than this still has to fit the widest card, so we clamp up to it -- but NOT to the cards' current
    right-extent: clamping to the extent would RATCHET (once cards spread wide the board could never
    pack narrower), so narrowing the window would never reflow into a single column.  At one-card
    width the gravity packer simply stacks every card in one column, which is the correct reflow."""
    widest = max((metrics.card_size(c.size)[0] for c in configs),
                 default=metrics.card_size("1x2")[0])
    return widest + 2 * metrics.gap


def board_width(configs: Sequence, metrics: BoardMetrics) -> int:
    """A fallback packing width for callers without a live viewport (the pure-function tests): two
    of the WIDEST card side by side plus the gap margins, so cards CAN pack side by side.  The real
    GUI passes the scroll viewport width to :func:`pack` instead, so the board wraps at the edge."""
    widest = max((metrics.card_size(c.size)[0] for c in configs),
                 default=metrics.card_size("1x2")[0])
    return max(2 * widest + 3 * metrics.gap, min_board_width(configs, metrics))


def pack(order: Sequence, metrics: BoardMetrics, board_w: int | None = None) -> bool:
    """The ONE board packer: place each card, IN THE GIVEN LIST ORDER, at the TOP-MOST then
    LEFT-MOST gap-clear slot (:func:`first_free_slot`).  Strict north-west gravity as a PURE function
    of the ORDER (the board's single source of truth), the sizes, and ``board_w`` -- it does NOT read
    any card's current pixel position, so it is deterministic and idempotent.

    Placement depends only on order: the first card lands at ``(gap, gap)``; every later card fills
    the first free NW slot clearing all already-placed cards by the gap within ``board_w`` (else drops
    to a new shelf below).  An Add appended LAST therefore always lands in the next bottom slot --
    never a middle hole -- and re-packing a settled board moves nothing.  A drop REORDERS the list
    (:func:`drop_index`); pack recomputes every pixel from the new order.  ``board_w`` None -> a
    two-wide headless fallback; a given width is honoured but clamped up to one-card-wide.  Returns
    True if any card's ``(col, row)`` changed."""
    order = list(order)
    board_w = (board_width(order, metrics) if board_w is None
               else max(board_w, min_board_width(order, metrics)))
    placed: list = []
    moved = False
    for cfg in order:
        col, row = first_free_slot(cfg, placed, board_w, metrics)
        if (cfg.col, cfg.row) != (col, row):
            cfg.col, cfg.row = col, row
            moved = True
        placed.append(cfg)
    return moved


def drop_index(cfg, others: Sequence, metrics: BoardMetrics, board_w: int | None = None) -> int:
    """The ORDER index at which to insert a card DROPPED at its raw pixel ``(cfg.col, cfg.row)`` among
    ``others`` (already in order), so it lands NEAREST the drop point under :func:`pack` gravity.

    The drop rule is expressed THROUGH the one packer instead of a separate placement math: for every
    candidate insertion index we pack a trial order (proxies, so the real configs are never mutated)
    and measure where the dropped card ends up; the index whose resulting top-left is closest to the
    raw drop wins.  A drop near an existing card's slot lands ON it (index before it -> that card and
    everything after shift DOWN the order and re-pack = "displace"); a drop past the last card lands
    at the bottom (append).  Ties -> the earliest index, so dropping squarely onto a card displaces
    it.  Board width None -> the same headless fallback :func:`pack` uses."""
    board_w = (board_width(list(others) + [cfg], metrics) if board_w is None else board_w)
    drop_x, drop_y = int(round(cfg.col)), int(round(cfg.row))
    proxies = [GeomProxy(o.size) for o in others]
    best_i, best_d = 0, None
    for k in range(len(proxies) + 1):
        probe = GeomProxy(cfg.size)
        trial = proxies[:k] + [probe] + proxies[k:]
        pack(trial, metrics, board_w)
        d = (probe.col - drop_x) ** 2 + (probe.row - drop_y) ** 2
        if best_d is None or d < best_d:
            best_d, best_i = d, k
    return best_i
