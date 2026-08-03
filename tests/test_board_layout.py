"""The frontend board packer places every panel by its current pixel contract.

``zlc_frontend.board_layout`` is the console's shelf packer and drop-placement rule.
Qt chrome tokens and figure size arrive explicitly through :class:`BoardMetrics`.

The layouts below are the current product golden at the default display scale. The
real card-size table is literal test data, so the packer can be checked without Qt or
Matplotlib while preserving the exact pixels the GUI promises.

These captured layouts remain the product oracle for ordering, width, and pixels.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_frontend.board_layout import (
    BoardMetrics, GeomProxy, board_width, drop_index, min_board_width, pack)

#: Current card outer sizes at the default scale.
CARD_PX = {
    "1x2": (500, 275), "2x2": (500, 401), "4x2": (500, 653),
    "1x4": (1008, 275), "2x4": (1008, 401), "4x4": (1008, 653),
    "4x8": (2024, 653), "8x4": (1008, 1157), "8x8": (2024, 1157),
}
GAP = 8
METRICS = BoardMetrics(gap=GAP, card_size=lambda size: CARD_PX[size])

CASES = {
    ("1x2", "1x2", "1x2"): {
        None: [(8, 8), (516, 8), (8, 291)],
        1200: [(8, 8), (516, 8), (8, 291)],
        700: [(8, 8), (8, 291), (8, 574)],
    },
    ("2x2", "1x2", "4x2", "2x2"): {
        None: [(8, 8), (516, 8), (516, 291), (8, 417)],
        1200: [(8, 8), (516, 8), (516, 291), (8, 417)],
        700: [(8, 8), (8, 417), (8, 700), (8, 1361)],
    },
    ("4x4", "1x2", "1x2", "2x4", "1x2"): {
        None: [(8, 8), (1024, 8), (1532, 8), (1024, 291), (8, 669)],
        1200: [(8, 8), (8, 669), (516, 669), (8, 952), (8, 1361)],
        700: [(8, 8), (8, 669), (516, 669), (8, 952), (8, 1361)],
    },
    ("8x8", "1x2"): {
        None: [(8, 8), (2040, 8)],
        1200: [(8, 8), (8, 1173)],
        700: [(8, 8), (8, 1173)],
    },
    ("1x4",) * 5: {
        None: [(8, 8), (1024, 8), (8, 291), (1024, 291), (8, 574)],
        1200: [(8, 8), (8, 291), (8, 574), (8, 857), (8, 1140)],
        700: [(8, 8), (8, 291), (8, 574), (8, 857), (8, 1140)],
    },
}


def _packed(sizes, board_w):
    order = [GeomProxy(size) for size in sizes]
    pack(order, METRICS, board_w)
    return [(card.col, card.row) for card in order]


@pytest.mark.parametrize("sizes", list(CASES))
def test_every_captured_layout_is_reproduced_pixel_for_pixel(sizes):
    """The current contract fixes order, width, and pixels."""

    for board_w, expected in CASES[sizes].items():
        assert _packed(sizes, board_w) == expected, f"width {board_w}"


def test_packing_a_settled_board_moves_nothing():
    """Idempotence is what makes a resize or a click stop reflowing surprisingly."""

    order = [GeomProxy(size) for size in ("2x2", "1x2", "4x2", "2x2")]
    assert pack(order, METRICS, 1200) is True          # first pack places them
    assert pack(order, METRICS, 1200) is False         # second changes nothing


def test_no_two_cards_come_closer_than_the_gap():
    """The invariant every captured layout is an instance of, checked directly."""

    for sizes in CASES:
        for board_w in CASES[sizes]:
            order = [GeomProxy(size) for size in sizes]
            pack(order, METRICS, board_w)
            boxes = [(c.col, c.row, c.col + CARD_PX[c.size][0], c.row + CARD_PX[c.size][1])
                     for c in order]
            for i, (ax0, ay0, ax1, ay1) in enumerate(boxes):
                for bx0, by0, bx1, by1 in boxes[i + 1:]:
                    apart = (ax1 + GAP <= bx0 or bx1 + GAP <= ax0
                             or ay1 + GAP <= by0 or by1 + GAP <= ay0)
                    assert apart, f"cards closer than the gap at width {board_w}"


def test_a_drop_lands_where_it_was_dropped():
    """The drop rule expressed through the packer, on the captured board."""

    sizes = ("2x2", "1x2", "4x2", "2x2")
    others = [GeomProxy(size) for size in sizes]
    pack(others, METRICS, 1200)

    for (drop_x, drop_y), expected in [((0, 0), 0), ((600, 0), 1),
                                       ((0, 400), 3), ((2000, 2000), 2)]:
        probe = GeomProxy("1x2", drop_x, drop_y)
        assert drop_index(probe, others, METRICS, 1200) == expected, (drop_x, drop_y)


def test_drop_position_can_be_supplied_without_mutating_the_layout_record():
    """A live Qt drag uses widget pixels until release, not stale config pixels."""

    others = [GeomProxy(size) for size in ("1x2", "1x2", "1x2")]
    pack(others, METRICS, 1200)
    probe = GeomProxy("1x2", 8, 8)
    assert drop_index(
        probe,
        others,
        METRICS,
        1200,
        raw_position=(600, 0),
    ) == 1
    assert (probe.col, probe.row) == (8, 8)


def test_a_snapshotted_card_size_is_refused():
    """The reason card_size is a callable, made mechanical.

    A card's pixels follow the live Qt scale.  A mapping captured at construction would
    keep answering the old scale after the user moved the window to another screen -
    exactly how the forwarding shims went stale before they were made to proxy live.
    """

    with pytest.raises(TypeError):
        BoardMetrics(gap=GAP, card_size=CARD_PX)       # a Mapping, not a callable

    with pytest.raises(ValueError):
        BoardMetrics(gap=-1, card_size=lambda size: (1, 1))


def test_the_widths_scale_with_the_widest_card():
    configs = [GeomProxy("1x2"), GeomProxy("4x8")]
    assert min_board_width(configs, METRICS) == 2024 + 2 * GAP
    assert board_width(configs, METRICS) == 2 * 2024 + 3 * GAP


def test_the_module_reaches_for_no_renderer_and_no_toolkit():
    """The headless layout owner must remain independent of rendering and Qt."""

    import zlc_frontend.board_layout as layout_owner

    tree = ast.parse(pathlib.Path(layout_owner.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert modules <= {"__future__", "dataclasses", "typing"}, modules


