"""Grid cell typography + corridor budget -- the Y2 contracts (user-fixed rules).

Rule 1 (two tiers, no continuous shrink): a grid squeezed into a panel size SMALLER than its
recommended default (``optimal_grid_size``; "smaller" = EITHER side below the recommendation)
uses HALF the standard 2x2 cell text (``font_scale = 0.5``); every other size uses the plot
kind's standard style sizes unscaled (``font_scale = 1.0``).

Rule 2 (space fully used, titles never cut): the row corridor's budget is exactly one cell-title
line + a few px of air -- the rendered title must clear the cell above it, and the corridor must
not be meaningfully wider than the title needs (spare pixels belong to the CELLS).  The column
corridor carries no text (edge-label + centered-single-tick policies) and stays a thin separator.

All geometry is asserted on REAL rendered figures (bbox pixels), not on formula echoes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.frontend as zf
from Zou_lab_control.frontend.live import (
    PANEL_SIZES,
    _cell_font_scale,
    _site_grid_geometry,
    grid,
    optimal_grid_size,
)


@pytest.fixture(scope="module", autouse=True)
def _style():
    zf.apply_style()


def test_cell_font_scale_has_exactly_two_tiers():
    """The cell-text scale is BINARY on the current-vs-recommended relation: EITHER side below
    the grid's recommended default size -> half the standard 2x2 text (0.5); equal or larger on
    both sides -> the standard sizes unscaled (1.0).  The recommendation DERIVES from the grid
    shape through ``optimal_grid_size`` (the ONE recommended-size source) -- the test never
    re-types its thresholds."""
    # {grid shape: {panel size: expected scale}} -- (3, 3) recommends the compact preset,
    # (5, 7) the double one (each axis past its boundary); pinned by test_size_driven_geometry.
    cases = {
        (3, 3): {"1x2": 0.5, "2x2": 1.0, "4x4": 1.0},
        (5, 7): {"2x2": 0.5, "4x2": 0.5, "2x4": 0.5, "4x4": 1.0},   # ONE small side already halves
    }
    seen: set[float] = set()
    for shape, by_size in cases.items():
        recommended = optimal_grid_size(*shape)
        for size, expected in by_size.items():
            assert _cell_font_scale(size, recommended) == expected, (shape, recommended, size)
        seen |= {_cell_font_scale(s, recommended) for s in PANEL_SIZES}
    assert seen == {0.5, 1.0}                       # exactly two tiers, nothing in between


@pytest.mark.parametrize("size,kind,n_points", [
    ("1x2", "hist", 6),      # squeezed below the compact recommendation -> half text
    ("2x2", "hist", 6),      # exactly the recommendation -> standard text
    ("4x4", "2d", 6),        # larger than the recommendation -> standard text
    ("2x2", "hist", 40),     # a 6x7 grid (recommends the double preset) squeezed -> half text
])
def test_row_corridor_fits_title_exactly(size, kind, n_points):
    """Rendered contract, at BOTH font tiers: every cell title clears the cell above it (no
    overlap / no cutoff), and the corridor is no more than ~8 px wider than the tallest title
    band (space belongs to the cells, not to the corridor)."""
    rng = np.random.default_rng(0)
    block = (rng.normal(120, 18, size=(20, n_points, 1)) if kind == "hist"
             else rng.normal(120, 18, size=(6, n_points, 10, 14)))
    g = grid(block, sub_plot_kind=kind, facet="points:0", points_shape=(n_points,),
             size=size, display=False)
    try:
        fig = g.fig
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes = list(np.ravel(g.axes))
        recommended = optimal_grid_size(g.nrows, g.ncols)   # same ONE source the grid stored
        _, _, row_gap, _, _ = _site_grid_geometry(size, recommended)

        title_band = 0.0
        for ax in axes:
            ab = ax.get_window_extent(renderer)
            tb = ax.title.get_window_extent(renderer)
            assert tb.y1 <= fig.get_size_inches()[1] * fig.dpi + 0.5   # never off the canvas
            title_band = max(title_band, tb.y1 - ab.y1)
            # the title must not intrude into the axes directly above (same column)
            for other in axes:
                ob = other.get_window_extent(renderer)
                if other is not ax and abs(ob.x0 - ab.x0) < 2 and ob.y0 > ab.y1:
                    assert tb.y1 <= ob.y0 + 0.5, f"title of a {size} cell clips the cell above"
        # corridor == title band + a few px of air, never a wide waste band
        assert row_gap >= title_band + 2
        assert row_gap <= title_band + 8, (row_gap, title_band)
    finally:
        import matplotlib.pyplot as plt
        plt.close(g.fig)
