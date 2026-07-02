"""Grid cell typography + corridor budget -- the Y2 contracts (user-fixed rules).

Rule 1 (two tiers, no continuous shrink): a panel SMALLER than 2x2 (either side < 2 half-units)
uses ONE NOTCH smaller cell text (``smaller_fontsize()``); 2x2 and larger use the plot kind's
standard style sizes unscaled.

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
from Zou_lab_control.frontend.live import PANEL_SIZES, _cell_font_scale, _site_grid_geometry, grid, panel_size_cells
from Zou_lab_control.frontend.style import small_fontsize, smaller_fontsize


@pytest.fixture(scope="module", autouse=True)
def _style():
    zf.apply_style()


def test_cell_font_scale_has_exactly_two_tiers():
    """Every curated panel size lands on one of exactly TWO scales: one-notch-down below 2x2,
    the standard sizes from 2x2 up -- derived from style, never a literal."""
    compact = smaller_fontsize() / small_fontsize()
    for size in PANEL_SIZES:
        rows, cols = panel_size_cells(size)
        expected = 1.0 if (rows >= 2 and cols >= 2) else compact
        assert _cell_font_scale(size) == expected, size
    assert {_cell_font_scale(s) for s in PANEL_SIZES} == {1.0, compact}


@pytest.mark.parametrize("size,kind", [("1x2", "hist"), ("2x2", "hist"), ("4x4", "2d")])
def test_row_corridor_fits_title_exactly(size, kind):
    """Rendered contract: every cell title clears the cell above it (no overlap / no cutoff),
    and the corridor is no more than ~8 px wider than the tallest title band (space belongs to
    the cells, not to the corridor)."""
    rng = np.random.default_rng(0)
    block = (rng.normal(120, 18, size=(20, 6, 1)) if kind == "hist"
             else rng.normal(120, 18, size=(6, 6, 10, 14)))
    g = grid(block, sub_plot_kind=kind, facet="points:0", points_shape=(6,),
             size=size, display=False)
    try:
        fig = g.fig
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes = list(np.ravel(g.axes))
        _, _, row_gap, _, _ = _site_grid_geometry(size)

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
