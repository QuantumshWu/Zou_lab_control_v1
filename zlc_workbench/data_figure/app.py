"""Qt-lazy composition for one standalone ``zlc_plot`` Figure."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from zlc_data import OwnedSnapshot
from zlc_plot import PlotSpec

from .window import DataFigureWindow


def create_data_figure_pane(
    snapshot: OwnedSnapshot,
    spec: PlotSpec,
    *,
    output_root: Path,
    size: str | None = None,
    parameters: Mapping[str, object] | None = None,
    archive_path: str | Path | None = None,
    metadata: Mapping[str, object] | None = None,
    open_fit: bool = False,
    embedded: bool = True,
    parent=None,
) -> DataFigureWindow:
    """Build the sole interactive surface used by Viewer and snapshot Edit."""

    from zlc_frontend.qt_widgets import ensure_qt_app

    ensure_qt_app()
    return DataFigureWindow(
        snapshot,
        spec,
        output_root=output_root,
        size=size,
        parameters=parameters,
        archive_path=archive_path,
        metadata=metadata,
        open_fit=open_fit,
        embedded=embedded,
        parent=parent,
    )


def open_figure_workbench(
    snapshot: OwnedSnapshot,
    spec: PlotSpec,
    *,
    output_root: Path,
    size: str | None = None,
    parameters: Mapping[str, object] | None = None,
    archive_path: str | Path | None = None,
    metadata: Mapping[str, object] | None = None,
    open_fit: bool = False,
    hide_on_close: bool = False,
) -> DataFigureWindow:
    """Open the formal Data Figure window over one already-frozen snapshot."""

    from zlc_frontend.qt_widgets import (
        WINDOW_SCREEN_FRACTION,
        ensure_qt_app,
        launch_fluent_window,
        screen_fit_window_size,
    )

    ensure_qt_app()
    pane = create_data_figure_pane(
        snapshot,
        spec,
        output_root=output_root,
        size=size,
        parameters=parameters,
        archive_path=archive_path,
        metadata=metadata,
        open_fit=open_fit,
        embedded=False,
    )

    def wire(window) -> None:
        if not hide_on_close:
            window.set_close_guard(pane.teardown)

    initial_size = screen_fit_window_size(WINDOW_SCREEN_FRACTION)
    window = launch_fluent_window(
        pane,
        title="Data Figure@Zou lab",
        hide_on_close=hide_on_close,
        fixed_size=False,
        size=(initial_size.width(), initial_size.height()),
        wire=wire,
    )
    pane._zlc_window = window
    return pane


__all__ = ["create_data_figure_pane", "open_figure_workbench"]
