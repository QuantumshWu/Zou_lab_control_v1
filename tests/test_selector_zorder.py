"""Selector artists must sit ABOVE every plot data overlay.

The plot overlays (site occupancy rings, threshold cut lines, pulse brackets) draw at
zorder 1-12; a transient selector (area box + handles, crosshair, coordinate text) drawn
at matplotlib's default zorder (1-3) would be HIDDEN behind them, so the user can't see
what they're selecting.  `selectors.SELECTOR_ZORDER` is the single contract that pins
every selector artist clear of the data layer; these tests guard it.

The axes come from the real `zf.plot` entry point so matplotlib's rcParams carry the
frontend style (numeric font sizes the selectors read), exactly as on a live plot.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import Zou_lab_control.frontend as zf  # noqa: E402
from Zou_lab_control.frontend.selectors import (  # noqa: E402
    AreaSelector,
    CrossSelector,
    SELECTOR_ZORDER,
)

# The highest data-overlay zorder in the repo is the pulse-bracket label at 12.
HIGHEST_DATA_OVERLAY_ZORDER = 12


def _live_overlay_ax():
    """A real plot's axes, carrying a data overlay at the repo's highest overlay zorder."""
    plot = zf.plot(np.arange(5.0), np.arange(5.0), display=False)
    plot.ax.axhline(2.0, zorder=HIGHEST_DATA_OVERLAY_ZORDER)   # e.g. a threshold overlay
    return plot


def test_selector_zorder_clears_every_data_overlay():
    assert SELECTOR_ZORDER > HIGHEST_DATA_OVERLAY_ZORDER


def test_area_selector_artists_sit_above_overlays():
    plot = _live_overlay_ax()
    try:
        area = AreaSelector(plot.ax)
        artists = list(getattr(area.selector, "artists", ()))
        assert artists, "RectangleSelector should expose its artists"
        for artist in artists:
            assert artist.get_zorder() >= SELECTOR_ZORDER
            assert artist.get_zorder() > HIGHEST_DATA_OVERLAY_ZORDER
        # the coordinate label drawn on a drag sits one above the box
        area.selector.extents = (0.5, 2.5, 0.5, 2.5)
        area.onselect(SimpleNamespace(xdata=0.5, ydata=0.5), SimpleNamespace(xdata=2.5, ydata=2.5))
        assert area.text is not None
        assert area.text.get_zorder() == SELECTOR_ZORDER + 1
    finally:
        plt.close(plot.ax.figure)


def test_cross_selector_artists_sit_above_overlays():
    plot = _live_overlay_ax()
    try:
        cross = CrossSelector(plot.ax)
        event = SimpleNamespace(inaxes=plot.ax, button=3, xdata=1.0, ydata=1.0, dblclick=False)
        cross.on_press(event)
        for artist in (cross.vline, cross.hline, cross.point):
            assert artist.get_zorder() == SELECTOR_ZORDER
            assert artist.get_zorder() > HIGHEST_DATA_OVERLAY_ZORDER
        assert cross.text.get_zorder() == SELECTOR_ZORDER + 1
    finally:
        plt.close(plot.ax.figure)


def test_cross_selector_distribution_line_sits_above_overlays():
    """The crosshair's companion line on the side distribution axes (2-D image) is also pinned."""
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-1, 1, 7)
    xx, yy = np.meshgrid(x, y)
    z = np.exp(-(xx**2 + yy**2))
    data_x = np.column_stack([xx.ravel(), yy.ravel()])
    plot = zf.plot(data_x, z.ravel(), labels=("X", "Y", "Counts"), display=False)
    ax = plot.ax
    fig = ax.figure
    axdis = fig._zlc_state.axdis
    assert axdis is not None, "a 2-D plot exposes a distribution axes"
    axdis.axhline(0.5, zorder=HIGHEST_DATA_OVERLAY_ZORDER)
    try:
        cross = CrossSelector(ax)
        event = SimpleNamespace(inaxes=ax, button=3, xdata=0.0, ydata=0.0, dblclick=False)
        cross.on_press(event)
        assert cross._dis_line is not None, "a 2-D crosshair must draw a distribution line"
        assert cross._dis_line.get_zorder() == SELECTOR_ZORDER
        assert cross._dis_line.get_zorder() > HIGHEST_DATA_OVERLAY_ZORDER
    finally:
        plt.close(fig)
