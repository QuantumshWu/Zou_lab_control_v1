"""Nonblocking Qt viewer for one frozen current ``DataFigure``."""

from __future__ import annotations

from concurrent.futures import CancelledError
import threading

from zlc_frontend import DataFigure
from zlc_frontend.encoded_raster import EncodedRasterDocument, EncodedRasterPage
from zlc_storage import positive_integer

from ._frozen_raster import (
    FrozenRasterWindow,
    open_frozen_raster_window,
)


_DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024


def _figure_summary(figure: DataFigure) -> str:
    document = figure.document
    intents = tuple(dict.fromkeys(layer.view.intent.value for layer in document.layers))
    panel_count = sum(len(layer.cells) for layer in figure.evaluated.layers)
    return (
        f"{'/'.join(value.lower() for value in intents)} · {panel_count} panel(s) · "
        f"document revision {document.revision}"
    )


def _render_figure(
    loader,
    memory_limit_bytes: int,
    cancelled: threading.Event | None = None,
) -> EncodedRasterDocument:
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()
    figure = loader()
    if not isinstance(figure, DataFigure):
        raise TypeError("figure loader must return DataFigure")
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()
    frozen_limit = figure.render_memory_limit_bytes
    render_limit = (
        memory_limit_bytes
        if frozen_limit is None
        else min(memory_limit_bytes, frozen_limit)
    )
    payload = figure.to_png_bytes(memory_limit_bytes=render_limit)
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()
    return EncodedRasterDocument(
        _figure_summary(figure),
        (EncodedRasterPage("figure", "Figure", payload),),
    )


def _open_figure_window(loader, *, memory_limit_bytes: int) -> FrozenRasterWindow:
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    return open_frozen_raster_window(
        lambda cancelled: _render_figure(loader, limit, cancelled),
        window_title="Data Figure",
        mode_text="FROZEN DATA FIGURE · DISPLAY ONLY",
        loading_summary="Resolving immutable input…",
        object_prefix="figureViewer",
        subject="figure",
        memory_limit_bytes=limit,
    )


def open_data_figure_workbench(
    figure: DataFigure,
    *,
    memory_limit_bytes: int = _DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES,
) -> FrozenRasterWindow:
    """Open an already-resolved DataFigure on the shared frozen-raster lane."""

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    return _open_figure_window(
        lambda: figure,
        memory_limit_bytes=memory_limit_bytes,
    )


def open_figure_workbench(
    figure_factory,
    source,
    *,
    intent=None,
    selection=None,
    preferences=None,
    memory_limit_bytes: int = _DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES,
) -> FrozenRasterWindow:
    """Resolve and render a current artifact entirely on the bounded worker."""

    if not callable(figure_factory):
        raise TypeError("figure_factory must be callable")
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    return _open_figure_window(
        lambda: figure_factory(
            source,
            intent=intent,
            selection=selection,
            preferences=preferences,
            memory_limit_bytes=limit,
        ),
        memory_limit_bytes=limit,
    )


__all__ = [
    "open_data_figure_workbench",
    "open_figure_workbench",
]
