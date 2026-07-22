"""Qt-lazy composition root for the session-independent FigureViewer."""

from __future__ import annotations

__all__ = ["open_figure_viewer"]


def open_figure_viewer(path=None, *, scale=None, **kwargs):
    """Open the saved-figure viewer, optionally committing one initial path."""

    from .plot_bridge_figure_viewer import show_figure_viewer

    return show_figure_viewer(path=path, scale=scale, **kwargs)
