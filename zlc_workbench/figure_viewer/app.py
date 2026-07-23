"""Qt-lazy composition root for the session-independent FigureViewer."""

from __future__ import annotations

__all__ = ["open_figure_viewer"]


def open_figure_viewer(
    path=None,
    *,
    scale=None,
    window_ratio=None,
    hide_on_close=False,
):
    """Open the saved-figure viewer, optionally committing one initial path."""

    from zlc_frontend.qt_widgets import (
        WINDOW_SCREEN_FRACTION,
        ensure_qt_app,
        launch_fluent_window,
    )

    from .window import FigureViewer

    ensure_qt_app()
    viewer = FigureViewer(
        path,
        scale=scale,
        window_ratio=(
            WINDOW_SCREEN_FRACTION
            if window_ratio is None
            else float(window_ratio)
        ),
    )

    def _wire(window):
        if not hide_on_close:
            window.set_close_guard(viewer.teardown)

    window = launch_fluent_window(
        viewer,
        title="FigureViewer@Zou lab",
        hide_on_close=hide_on_close,
        wire=_wire,
    )
    viewer._zlc_window = window
    return viewer
