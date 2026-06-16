"""Notebook plotting style for Zou lab front-end figures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import matplotlib
from matplotlib import font_manager as fm

try:
    from IPython import get_ipython
    from IPython.display import HTML, display
except Exception:  # pragma: no cover - only absent outside IPython.
    get_ipython = None
    HTML = None
    display = None


NEW_BLACK = "black"
FONT_PATH = Path(__file__).resolve().parent / "assets" / "helvetica-light-587ebe5a59211.ttf"

_FONT_NAME = None
if FONT_PATH.exists():
    try:
        fm.fontManager.addfont(str(FONT_PATH))
        _FONT_NAME = fm.FontProperties(fname=str(FONT_PATH)).get_name()
    except Exception:
        _FONT_NAME = None

SANS_SERIF = ([_FONT_NAME] if _FONT_NAME else []) + ["Arial"]

# --------------------------------------------------------------------------- #
# Geometry design tokens.  style.py is the lowest-level frontend module (no
# internal imports), so the ONE stock-figure geometry + the ONE design dpi live
# here and every other module (canvas.FigureSpec, live's panel/pulse specs) reads
# them -- a value is written ONCE and never re-typed, so nothing can drift.
# These are OWNED constants of the frontend visual system, NOT per-call knobs.
# --------------------------------------------------------------------------- #
DESIGN_DPI = 300                          # the one design dpi; every figure renders at this
STOCK_DATA_PX = (480, 360)                # the stock single-axes data region (confocal)
STOCK_MARGINS_PX = (110, 110, 100, 40)    # confocal stock margins (L, R, B, T)

# Stock figure size in inches = (data + L + R, data + B + T) / dpi.  Derived, so
# it can never disagree with FigureSpec's defaults (which read the same tokens).
_STOCK_FIGSIZE = [
    (STOCK_DATA_PX[0] + STOCK_MARGINS_PX[0] + STOCK_MARGINS_PX[1]) / DESIGN_DPI,
    (STOCK_DATA_PX[1] + STOCK_MARGINS_PX[2] + STOCK_MARGINS_PX[3]) / DESIGN_DPI,
]

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


def use_widget_backend() -> None:
    """Switch Matplotlib to the Jupyter widget backend."""
    if get_ipython is None:
        raise RuntimeError("IPython is not available.")
    ip = get_ipython()
    if ip is None:
        raise RuntimeError("No active IPython shell is available.")
    ip.run_line_magic("matplotlib", "widget")


def enable_long_output() -> None:
    """Remove notebook output scroll boxes when the frontend supports it."""
    if display is None or HTML is None:
        return
    display(
        HTML(
            """
            <style>
            .output_scroll {
                height: auto !important;
                max-height: none !important;
            }
            </style>
            """
        )
    )


def apply_style(overrides: Mapping[str, Any] | None = None) -> None:
    """Apply the Confocal_GUIv2-derived publication/notebook style."""
    style = dict(_DEFAULT_STYLE)
    if overrides:
        style.update(dict(overrides))
    matplotlib.rcParams.update(style)


@contextmanager
def style_context(overrides: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Temporarily apply the front-end plotting style."""
    style = dict(_DEFAULT_STYLE)
    if overrides:
        style.update(dict(overrides))
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


def apply_title(target, title: str, *, pad: float | None = None):
    """Render a plot title consistently on an Axes OR a Figure.

    One title mechanism, so single-axes plots and figure-level grids never drift
    on title size/position.  Returns the artist, or ``None`` for an empty title.
    """

    if not title:
        return None
    size = title_fontsize()
    if hasattr(target, "set_title"):  # an Axes
        if pad is None:
            pad = max(float(matplotlib.rcParams["axes.titlepad"]), 2.5)
        return target.set_title(str(title), fontsize=size, pad=pad)
    return target.text(0.5, 0.992, str(title), ha="center", va="top", fontsize=size)


# The lab accent + population palette.  Colours are ART, owned here, never a
# per-call knob (sealed-API contract).  Read-only view, like DEFAULT_STYLE.
_PALETTE: dict[str, Any] = {
    "series": ["grey", "skyblue", "tab:blue", "tab:orange"],     # 1D / monitor line cycle
    "pulse_cycle": [
        "#5D7583", "#C37D5A", "#6F8D73", "#A66E87", "#7A6FA4", "#B5A262",
        "#5E9A9A", "#9A765E", "#7890B5", "#8B8B8B", "#B97878", "#679174",
    ],
    "bracket_cycle": ["#6A6A6A", "#C96F3D", "#4F7EA8", "#8B6BB8"],
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
    "bad": "white",          # NaN / masked cell
    "site_ring": "white",    # site-circle stroke
    "cmap_scan": "inferno",
    "cmap_site": "viridis",
    "cmap_camera": "gray",
}
PALETTE: Mapping[str, Any] = MappingProxyType(_PALETTE)


def threshold_line_kwargs(linewidth: float = 1.9) -> dict[str, Any]:
    """``axvline`` kwargs for a threshold cut line: one owned colour/alpha/zorder,
    ``linewidth`` per call (a narrow grid cell may use a thinner line)."""

    return {"color": _PALETTE["threshold"], "linewidth": float(linewidth), "alpha": 0.95, "zorder": 5}


__all__ = [
    "DEFAULT_STYLE",
    "DESIGN_DPI",
    "FONT_PATH",
    "NEW_BLACK",
    "PALETTE",
    "SANS_SERIF",
    "STOCK_DATA_PX",
    "STOCK_MARGINS_PX",
    "apply_style",
    "apply_title",
    "axis_label_fontsize",
    "enable_long_output",
    "small_fontsize",
    "smaller_fontsize",
    "style_context",
    "threshold_line_kwargs",
    "tick_fontsize",
    "title_fontsize",
    "use_widget_backend",
]

