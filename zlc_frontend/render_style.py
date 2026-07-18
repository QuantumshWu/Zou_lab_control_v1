"""The sole Matplotlib/render style owner for current and migrating frontends."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import matplotlib
from matplotlib import font_manager as fm

from zlc_storage import canonical_text

from .site_map import (
    SITE_EMPTY_ALPHA,
    SITE_EMPTY_COLOR,
    SITE_EMPTY_LINEWIDTH,
    SITE_INVALID_COLOR,
    SITE_OCCUPIED_ALPHA,
    SITE_OCCUPIED_COLOR,
    SITE_OCCUPIED_LINEWIDTH,
)

NEW_BLACK = "black"
FONT_PATH = Path(__file__).resolve().parent / "assets" / "helvetica-light-587ebe5a59211.ttf"

_FONT_NAME = None
if FONT_PATH.exists():
    try:
        fm.fontManager.addfont(str(FONT_PATH))
        _FONT_NAME = fm.FontProperties(fname=str(FONT_PATH)).get_name()
    except Exception:
        _FONT_NAME = None

SANS_SERIF = ((_FONT_NAME,) if _FONT_NAME else ()) + ("Arial",)

# --------------------------------------------------------------------------- #
# Geometry design tokens.  render_style.py is the lowest-level Matplotlib module (no
# internal imports), so the ONE stock-figure geometry + the ONE design dpi live
# here and every other module (canvas.FigureSpec, live's panel/pulse specs) reads
# them -- a value is written ONCE and never re-typed, so nothing can drift.
# These are OWNED constants of the frontend visual system, NOT per-call knobs.
# --------------------------------------------------------------------------- #
DESIGN_DPI = 300                          # the one DESIGN dpi; layout geometry + saved figures use it
# The ONE panel display scale: how large an embedded live panel APPEARS on screen, as a fraction of
# its DESIGN_DPI size (widget logical px = inches x DESIGN_DPI x PANEL_DISPLAY_SCALE).  It is the
# single knob qt_canvas.EmbeddedFigureCanvas takes -- the Agg buffer is ALWAYS rendered at the
# matching resolution (figure.dpi = DESIGN_DPI x real screen ratio x this), so the on-screen blit is
# 1:1 crisp, never rendered small and stretched up.  SAVED figures use savefig.dpi, independent of
# this.  (Lives HERE, the frontend's single source of truth -- NOT scattered across live.py /
# qt_canvas as a display_scale/render_scale pair -- a value written ONCE so display size can't drift.)
PANEL_DISPLAY_SCALE = 0.7
STOCK_DATA_PX = (480, 360)                # the stock single-axes data region (confocal)
STOCK_MARGINS_PX = (110, 110, 100, 40)    # confocal stock margins (L, R, B, T)

# Stock figure size in inches = (data + L + R, data + B + T) / dpi.  Derived, so
# it can never disagree with FigureSpec's defaults (which read the same tokens).
_STOCK_FIGSIZE = (
    (STOCK_DATA_PX[0] + STOCK_MARGINS_PX[0] + STOCK_MARGINS_PX[1]) / DESIGN_DPI,
    (STOCK_DATA_PX[1] + STOCK_MARGINS_PX[2] + STOCK_MARGINS_PX[3]) / DESIGN_DPI,
)

# The ONE typography/dpi system.  The mutable dict is PRIVATE so the owned
# defaults can never be mutated in place from outside (the public name below is
# a read-only view); see the frontend/__init__.py design contract.
_DEFAULT_STYLE: dict[str, Any] = {
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "figure.figsize": _STOCK_FIGSIZE,
    "lines.linewidth": 1,
    "scatter.edgecolors": NEW_BLACK,
    "legend.numpoints": 1,
    "lines.markersize": 2,
    "ytick.major.size": 1.5,
    "ytick.major.width": 0.4,
    "xtick.major.size": 1.5,
    "xtick.major.width": 0.4,
    "axes.linewidth": 0.4,
    "figure.subplot.left": 0,
    "figure.subplot.right": 1,
    "figure.subplot.bottom": 0,
    "figure.subplot.top": 1,
    "axes.titlepad": 1.5,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.labelpad": 1.5,
    "grid.linestyle": "--",
    "axes.grid": False,
    "text.usetex": False,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.top": False,
    "ytick.minor.right": False,
    "xtick.minor.bottom": False,
    "ytick.minor.left": False,
    "font.family": "sans-serif",
    "font.sans-serif": SANS_SERIF,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "savefig.dpi": 600,
    "figure.dpi": 300,
    "text.color": NEW_BLACK,
    "patch.edgecolor": NEW_BLACK,
    "patch.force_edgecolor": False,
    "hatch.color": NEW_BLACK,
    "axes.edgecolor": NEW_BLACK,
    "axes.titlecolor": NEW_BLACK,
    "axes.labelcolor": NEW_BLACK,
    "xtick.color": NEW_BLACK,
    "ytick.color": NEW_BLACK,
}

# Public, READ-ONLY view of the owned style (mutating it raises TypeError).
DEFAULT_STYLE: Mapping[str, Any] = MappingProxyType(_DEFAULT_STYLE)
_MATPLOTLIB_COMPOSE_LOCK = threading.RLock()


def apply_style(overrides: Mapping[str, Any] | None = None) -> None:
    """Apply the Confocal_GUIv2-derived publication/notebook style."""
    style = dict(_DEFAULT_STYLE)
    if overrides:
        style.update(dict(overrides))
    with _MATPLOTLIB_COMPOSE_LOCK:
        matplotlib.rcParams.update(style)


@contextmanager
def style_context(overrides: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Temporarily apply the style on the single product Matplotlib compose lane.

    ``matplotlib.rc_context`` mutates process-global rcParams and is not thread-local.  Holding
    this re-entrant lock for the complete context prevents two product render workers from
    restoring each other's snapshots out of order.  Direct third-party Matplotlib calls remain
    outside this product-owned lane and must not mutate rcParams concurrently with a render.
    """
    style = dict(_DEFAULT_STYLE)
    if overrides:
        style.update(dict(overrides))
    with _MATPLOTLIB_COMPOSE_LOCK:
        with matplotlib.rc_context(style):
            yield


# --------------------------------------------------------------------------- #
# Shared design tokens: ONE source for the sizes / colours / title that every
# plot type reads, so a new plot type cannot re-pick (and mis-pick) them.  All
# values are the established ones, so routing through these changes no pixels.
# --------------------------------------------------------------------------- #

def title_fontsize() -> float:
    """Point size for a plot title (every plot title reads this -- NOT the
    undefined ``axes.titlesize``, which would silently be matplotlib's larger
    stock default)."""

    return float(matplotlib.rcParams["axes.labelsize"])


def axis_label_fontsize() -> float:
    """Point size for an axis label (x / y / colorbar / outer grid label)."""

    return float(matplotlib.rcParams["axes.labelsize"])


def small_fontsize() -> float:
    """Point size for small in-plot annotations (stats text, per-cell tags)."""

    return float(matplotlib.rcParams["legend.fontsize"])


def tick_fontsize(axis: str = "x") -> float:
    """Point size for tick labels on ``axis`` ('x' or 'y')."""

    return float(matplotlib.rcParams[f"{axis}tick.labelsize"])


def smaller_fontsize(delta: float = 1.0, floor: float = 5.5) -> float:
    """A touch smaller than the small annotation size, never below ``floor``
    (the pulse plot's shrink-and-floor, written once)."""

    return max(float(floor), small_fontsize() - float(delta))


def apply_title(target, title: str, *, pad: float | None = None, size: float | None = None):
    """Render a plot title consistently on an Axes OR a Figure.

    One title mechanism, so single-axes plots and figure-level grids never drift
    on title size/position.  Returns the artist, or ``None`` for an empty title.
    ``size`` overrides the title point size (still read from render_style.py, never a raw
    literal) -- a grid CELL identifies itself with a small ``small_fontsize()`` title
    ABOVE its axes through this SAME mechanism, just smaller, so the site index never
    sits inside the cell's data area.
    """

    if not title:
        return None
    if size is None:
        size = title_fontsize()
    if hasattr(target, "set_title"):  # an Axes
        if pad is None:
            pad = max(float(matplotlib.rcParams["axes.titlepad"]), 2.5)
        return target.set_title(str(title), fontsize=size, pad=pad)
    return target.text(0.5, 0.992, str(title), ha="center", va="top", fontsize=size)


# The lab accent + population palette.  Colours are ART, owned here, never a
# per-call knob (sealed-API contract).  Read-only view, like DEFAULT_STYLE.
_PALETTE: dict[str, Any] = {
    # 1D / monitor multi-curve cycle: the Okabe-Ito COLOUR-BLIND-SAFE qualitative set, ordered so the
    # common 2-5 curve case gets maximally different HUES (blue / orange / green / purple / red ...)
    # -- never two near-identical colours (the old grey + sky-blue + tab:blue were indistinguishable).
    # Guarded by tests/test_series_palette_distinct.py (>=8 colours, pairwise perceptually distinct).
    "series": ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#666666", "#F0E442"),
    "line_single": "#808080",  # a lone 1-D curve / rolling trace -- matplotlib 'grey', EXACTLY confocal's repeat=1 line
    "pulse_cycle": (
        "#5D7583", "#C37D5A", "#6F8D73", "#A66E87", "#7A6FA4", "#B5A262",
        "#5E9A9A", "#9A765E", "#7890B5", "#8B8B8B", "#B97878", "#679174",
    ),
    "bracket_cycle": ("#6A6A6A", "#C96F3D", "#4F7EA8", "#8B6BB8"),
    "hist_fill": "grey",     # histogram / side-distribution bar fill
    "dark": "grey",          # dark population
    "bright": "skyblue",     # bright population
    "fit_left": "skyblue",   # left / dark Gaussian fit curve
    "fit_right": "orange",   # right / bright Gaussian fit curve
    "fit_total": "black",    # summed bimodal fit curve
    "threshold": "orange",   # threshold cut line + Gaussian-width fit line
    "annotation": NEW_BLACK,  # stats / tag text
    "readout": "grey",       # newest-value readout text
    "fit_text": "blue",      # post-run fit-result value text (DataFigure)
    "guide": "grey",         # faint min/max reference guide lines
    "data_scatter": "lightgrey",  # raw data points under a post-run fit
    "pulse_name": "white",   # name drawn inside a coloured pulse bar
    "pulse_grid": "0.88",
    "pulse_repeat_note": "0.35",
    "bad": "#9a9a9a",        # NaN / masked / not-yet-measured cell: a neutral mid-grey background so an
                             # in-band value-mask silhouette (or a partly-filled scan grid) reads AGAINST
                             # it -- white made a masked frame invisible (white-on-white blank, #4)
    "cmap_scan": "inferno",
    "cmap_camera": "gray",
}
PALETTE: Mapping[str, Any] = MappingProxyType(_PALETTE)


def _argb32(red: int, green: int, blue: int, alpha: int = 255) -> int:
    return (
        (int(alpha) << 24)
        | (int(red) << 16)
        | (int(green) << 8)
        | int(blue)
    )


@lru_cache(maxsize=8)
def indexed_colormap(name: str) -> tuple[int, ...]:
    """Return the canonical invalid + 255-sample QRgb table for one cmap.

    The render owner samples Matplotlib once per closed colormap name.  Qt
    installs this immutable table directly; a clim edit re-quantizes pixels in
    the worker and never asks Qt to invent a second value-to-colour mapping.
    """

    name = canonical_text(name, "colormap name")
    try:
        cmap = matplotlib.colormaps[name]
    except KeyError as exc:
        raise ValueError(f"unknown Matplotlib colormap {name!r}") from exc
    rgba = cmap([index / 254.0 for index in range(255)], bytes=True)
    bad_rgba = tuple(
        round(channel * 255)
        for channel in matplotlib.colors.to_rgba(_PALETTE["bad"])
    )
    return (
        _argb32(*bad_rgba),
        *(
            _argb32(int(red), int(green), int(blue), int(alpha))
            for red, green, blue, alpha in rgba
        ),
    )

# Current headless FigureDocument/Agg tokens extend the mature plotting system above; they do
# not establish a second palette or rcParams owner.  Every renderer imports from this module.
RENDER_TEXT = NEW_BLACK
RENDER_BACKGROUND = "#FFFFFF"
RENDER_GRID = _PALETTE["pulse_grid"]
FIT_FAILURE_COLOR = SITE_INVALID_COLOR
FIT_CONTOUR_COLOR = "#FFFFFF"
FIT_CONTOUR_LINEWIDTH = 0.8
FIT_LINESTYLE = "--"
CURVE_LINESTYLE = "-"
CURVE_MARKER = "o"
ANNOTATION_FONT_SIZE = "small"
PULSE_SCAN_REGION_COLOR = "#D69A6E"
PULSE_SCAN_ANNOTATION_COLOR = "#FFFFFF"
PULSE_SCAN_ANNOTATION_FONT_SIZE = 3.36
SERIES_COLORS = tuple(_PALETTE["series"])
RENDER_RCPARAMS: Mapping[str, Any] = MappingProxyType(dict(_DEFAULT_STYLE))


@contextmanager
def render_style_context() -> Iterator[None]:
    """Apply the canonical series cycle on the serialized Matplotlib compose lane."""

    from cycler import cycler

    with style_context({"axes.prop_cycle": cycler(color=SERIES_COLORS)}):
        yield

# Occupancy overlay on a site map: a 2D camera frame with one hollow ring per
# tweezer -- WHITE/grey for an EMPTY site, a bold warm ring for an OCCUPIED one (the
# Rb87 readout look the user prefers).  Single source for the site-map plot.  An empty
# site sits in the DARK background, so a near-opaque WHITE ring reads clearly there
# (the earlier faint alpha=0.22 vanished on grey -- the fix is opacity + a real stroke,
# NOT a coloured ring); the occupied ring stays warm-orange so the two never confuse.
SITE_OCCUPANCY_STYLE: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "empty": MappingProxyType({
        "color": SITE_EMPTY_COLOR,
        "alpha": SITE_EMPTY_ALPHA,
        "linewidth": SITE_EMPTY_LINEWIDTH,
    }),
    "occupied": MappingProxyType({
        "color": SITE_OCCUPIED_COLOR,
        "alpha": SITE_OCCUPIED_ALPHA,
        "linewidth": SITE_OCCUPIED_LINEWIDTH,
    }),
})


def threshold_line_kwargs(linewidth: float = 1.9) -> dict[str, Any]:
    """``axvline`` kwargs for a threshold cut line: one owned colour/alpha/zorder,
    ``linewidth`` per call (a narrow grid cell may use a thinner line)."""

    return {"color": _PALETTE["threshold"], "linewidth": float(linewidth), "alpha": 0.95, "zorder": 5}


# The ONE histogram bar-fill opacity.  A dis bin is drawn SLIGHTLY translucent so an overlapped
# distribution (a 'create' repeat overlay) stays legible and the threshold/fit lines read THROUGH
# the bars -- and so the lone (repeat=1) histogram reads the SAME as an overlay, never an opaque
# block beside translucent ones.  Owned here (ART), read by every histogram poly (primary + overlay)
# so the two can never drift to different opacities.
HIST_FILL_ALPHA = 0.4

#: While a fit overlay is drawn the data lines dim to this so the fit curve reads clearly; clearing
#: the fit restores each line's ORIGINAL alpha (the single source both halves of the overlay use).
FIT_DIM_ALPHA = 0.5


def bimodal_fit_line_specs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """``ax.plot`` style kwargs for the THREE bimodal fit curves (left / right / total), in that
    order.  ONE owned source of the fit-line COLOUR (from PALETTE) plus its linewidth/alpha, so the
    standalone :class:`HistogramFigure` and the grid hist thumbnail draw the SAME fit lines and can
    never drift to two different weights/opacities the way a re-typed triple could."""

    return (
        {"color": _PALETTE["fit_left"], "linewidth": 1, "alpha": 0.8},
        {"color": _PALETTE["fit_right"], "linewidth": 1, "alpha": 0.8},
        {"color": _PALETTE["fit_total"], "linewidth": 1, "alpha": 0.35},
    )


__all__ = [
    "ANNOTATION_FONT_SIZE",
    "bimodal_fit_line_specs",
    "CURVE_LINESTYLE",
    "CURVE_MARKER",
    "DEFAULT_STYLE",
    "DESIGN_DPI",
    "FIT_CONTOUR_COLOR",
    "FIT_CONTOUR_LINEWIDTH",
    "HIST_FILL_ALPHA",
    "FIT_FAILURE_COLOR",
    "FIT_LINESTYLE",
    "FONT_PATH",
    "NEW_BLACK",
    "PALETTE",
    "indexed_colormap",
    "PANEL_DISPLAY_SCALE",
    "PULSE_SCAN_ANNOTATION_COLOR",
    "PULSE_SCAN_ANNOTATION_FONT_SIZE",
    "PULSE_SCAN_REGION_COLOR",
    "RENDER_BACKGROUND",
    "RENDER_GRID",
    "RENDER_RCPARAMS",
    "RENDER_TEXT",
    "SANS_SERIF",
    "SERIES_COLORS",
    "SITE_OCCUPANCY_STYLE",
    "STOCK_DATA_PX",
    "STOCK_MARGINS_PX",
    "apply_style",
    "apply_title",
    "axis_label_fontsize",
    "small_fontsize",
    "smaller_fontsize",
    "render_style_context",
    "style_context",
    "threshold_line_kwargs",
    "tick_fontsize",
    "title_fontsize",
]

