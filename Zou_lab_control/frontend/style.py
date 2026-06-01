"""Notebook plotting style for Zou lab front-end figures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
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

DEFAULT_STYLE: dict[str, Any] = {
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "figure.figsize": [700 / 300, 500 / 300],
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
    style = DEFAULT_STYLE.copy()
    if overrides:
        style.update(dict(overrides))
    matplotlib.rcParams.update(style)


@contextmanager
def style_context(overrides: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Temporarily apply the front-end plotting style."""
    style = DEFAULT_STYLE.copy()
    if overrides:
        style.update(dict(overrides))
    with matplotlib.rc_context(style):
        yield


__all__ = [
    "DEFAULT_STYLE",
    "FONT_PATH",
    "NEW_BLACK",
    "SANS_SERIF",
    "apply_style",
    "enable_long_output",
    "style_context",
    "use_widget_backend",
]

