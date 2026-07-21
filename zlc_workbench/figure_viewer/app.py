"""The figure viewer's composition root -- the one place that opens the viewer window.

The window is the ORIGINAL saved-figure viewer UI -- Browse/gallery, re-plot,
re-edit, Flow tab, Fluent chrome -- hosted in :mod:`.plot_bridge_figure_viewer`
(the UI skeleton is kept BY DIRECTIVE 2026-07-21; it is never redesigned).  Its
remaining legacy-tree imports are a listed rewiring debt being replaced by the
zlc_* stack, tracked to zero by ``tests/test_workbench_zero_legacy.py``.
"""

from __future__ import annotations

__all__ = ["open_figure_viewer"]


def open_figure_viewer(path=None, *, scale=None, **kwargs):
    """Open the saved-figure viewer and return the viewer widget."""

    from .plot_bridge_figure_viewer import show_figure_viewer

    return show_figure_viewer(path=path, scale=scale, **kwargs)
