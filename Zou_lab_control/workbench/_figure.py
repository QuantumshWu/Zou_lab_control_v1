"""Nonblocking Qt viewer for one frozen current ``DataFigure``."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import threading
from typing import Callable

from PyQt5 import QtCore, QtWidgets

from zlc_frontend import DataFigure
from zlc_frontend.image_raster import (
    estimate_encoded_png_front_peak_nbytes,
    png_raster_size,
)
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentLabel,
    GREY,
    QtImageBoard,
    QtOwnerWake,
    WINDOW_SCREEN_FRACTION,
    center_window_on_primary_screen,
    ensure_qt_app,
    release_window,
    retain_window,
    screen_fit_window_size,
    set_fluent_scale,
)
from zlc_storage import positive_integer


_DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_FIGURE_RENDER_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zlc-figure-render",
)


def _error_summary(error: BaseException) -> str:
    message = str(error).strip()
    return type(error).__name__ if not message else f"{type(error).__name__}: {message}"


def _figure_summary(figure: DataFigure) -> str:
    document = figure.document
    intents = tuple(dict.fromkeys(layer.view.intent.value for layer in document.layers))
    panel_count = sum(len(layer.cells) for layer in figure.evaluated.layers)
    intent_text = "/".join(value.lower() for value in intents)
    return (
        f"{intent_text} · {panel_count} panel(s) · "
        f"document revision {document.revision}"
    )


@dataclass(frozen=True, slots=True)
class _RenderedFigure:
    png_bytes: bytes
    summary: str

    def __post_init__(self) -> None:
        png_raster_size(self.png_bytes)
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("rendered figure summary must be non-empty text")
        object.__setattr__(self, "summary", self.summary.strip())

    @property
    def source_front_peak_nbytes(self) -> int:
        return estimate_encoded_png_front_peak_nbytes(self.png_bytes)


def _render_figure(
    loader: Callable[[], DataFigure],
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> _RenderedFigure:
    if cancelled.is_set():
        raise CancelledError()
    figure = loader()
    if not isinstance(figure, DataFigure):
        raise TypeError("figure loader must return DataFigure")
    if cancelled.is_set():
        raise CancelledError()
    frozen_limit = figure.render_memory_limit_bytes
    render_limit = (
        memory_limit_bytes
        if frozen_limit is None
        else min(memory_limit_bytes, frozen_limit)
    )
    payload = figure.to_png_bytes(memory_limit_bytes=render_limit)
    if cancelled.is_set():
        raise CancelledError()
    rendered = _RenderedFigure(payload, _figure_summary(figure))
    if rendered.source_front_peak_nbytes > memory_limit_bytes:
        raise MemoryError(
            "Qt figure front requires "
            f"{rendered.source_front_peak_nbytes} bytes; limit is "
            f"{memory_limit_bytes}"
        )
    return rendered


class FigureWorkbenchWindow(QtWidgets.QWidget):
    """Display one frozen DataFigure without blocking the Qt owner."""

    def __init__(
        self,
        loader: Callable[[], DataFigure],
        *,
        memory_limit_bytes: int = _DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES,
    ) -> None:
        super().__init__()
        if not callable(loader):
            raise TypeError("loader must be callable")
        self._memory_limit_bytes = positive_integer(
            memory_limit_bytes,
            "memory_limit_bytes",
        )
        self._future: Future[_RenderedFigure] | None = None
        self._cancelled = threading.Event()
        self._rendered: _RenderedFigure | None = None
        self._closing = False
        self._closed = False
        self._allow_close = False

        self.setWindowTitle("Data Figure")
        self._mode = FluentLabel("FROZEN DATA FIGURE · DISPLAY ONLY", self)
        self._mode.setObjectName("figureViewerMode")
        self._status = FluentLabel("BUILDING FIGURE", self)
        self._status.setObjectName("figureViewerStatus")
        self._summary = FluentLabel("Resolving immutable input…", self)
        self._summary.setObjectName("figureViewerSummary")
        self._summary.setWordWrap(True)
        self._board = QtImageBoard(
            "data-figure",
            self,
            empty_text="Building frozen figure…",
        )
        self._board.setObjectName("figureViewerBoard")
        self._board.setMinimumSize(320, 240)
        self._diagnostic = FluentLabel("", self)
        self._diagnostic.setObjectName("figureViewerDiagnostic")
        self._diagnostic.setWordWrap(True)
        self._diagnostic.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._close_button = FluentButton("Close", self, color=GREY)
        self._close_button.setObjectName("closeFigureViewerButton")

        controls = QtWidgets.QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(self._close_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._mode)
        layout.addWidget(self._status)
        layout.addWidget(self._summary)
        layout.addWidget(self._board, 1)
        layout.addWidget(self._diagnostic)
        layout.addLayout(controls)

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._close_button.clicked.connect(self.shutdown)
        try:
            future = _FIGURE_RENDER_EXECUTOR.submit(
                _render_figure,
                loader,
                self._memory_limit_bytes,
                self._cancelled,
            )
        except BaseException as error:
            self._status.setText("FIGURE FAILED")
            self._diagnostic.setText(_error_summary(error))
        else:
            self._future = future
            future.add_done_callback(lambda _done: self._wake.request_owner_wake())

    @property
    def worker_idle(self) -> bool:
        return self._future is None

    @property
    def raster_ready(self) -> bool:
        return self._rendered is not None and self._board.has_front

    @property
    def closed(self) -> bool:
        return self._closed

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        future = self._future
        if future is not None and future.done():
            self._future = None
            try:
                rendered = future.result()
            except CancelledError:
                if not self._closing:
                    self._status.setText("FIGURE CANCELLED")
            except BaseException as error:
                if not self._closing:
                    self._status.setText("FIGURE FAILED")
                    self._summary.setText("No raster was admitted")
                    self._diagnostic.setText(_error_summary(error))
            else:
                if not self._closing:
                    try:
                        device_ratio = float(self._board.devicePixelRatioF())
                        presentation_size = (
                            max(1, math.ceil(self._board.width() * device_ratio)),
                            max(1, math.ceil(self._board.height() * device_ratio)),
                        )
                        required = estimate_encoded_png_front_peak_nbytes(
                            rendered.png_bytes,
                            presentation_size=presentation_size,
                        )
                        if required > self._memory_limit_bytes:
                            raise MemoryError(
                                "Qt figure presentation requires "
                                f"{required} bytes; limit is "
                                f"{self._memory_limit_bytes}"
                            )
                        self._board.present_encoded(
                            rendered.png_bytes,
                            image_format="PNG",
                        )
                    except BaseException as error:
                        self._status.setText("DISPLAY FAILED")
                        self._summary.setText("The frozen figure remains valid")
                        self._diagnostic.setText(_error_summary(error))
                    else:
                        self._rendered = rendered
                        self._status.setText("READY")
                        self._summary.setText(rendered.summary)
                        self._diagnostic.setText("")
            future = None
        self._finish_close_if_ready()

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        self._closing = True
        self._cancelled.set()
        self._status.setText("CLOSING")
        self._close_button.setEnabled(False)
        self._rendered = None
        self._board.clear()
        future = self._future
        if future is not None:
            future.cancel()
        self._finish_close_if_ready()

    def _finish_close_if_ready(self) -> None:
        if not self._closing or self._future is not None or self._closed:
            return
        self._wake.detach()
        self._closed = True
        self._allow_close = True
        QtCore.QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:
        if self._allow_close:
            release_window(self)
            event.accept()
            return
        event.ignore()
        self.shutdown()


def _open_figure_window(
    loader: Callable[[], DataFigure],
    *,
    memory_limit_bytes: int,
) -> FigureWorkbenchWindow:
    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("DataFigure Workbench must be opened on the Qt GUI thread")
    set_fluent_scale(None)
    window = FigureWorkbenchWindow(
        loader,
        memory_limit_bytes=memory_limit_bytes,
    )
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


def open_data_figure_workbench(
    figure: DataFigure,
    *,
    memory_limit_bytes: int = _DEFAULT_FIGURE_GUI_MEMORY_LIMIT_BYTES,
) -> FigureWorkbenchWindow:
    """Open an already-resolved DataFigure on the shared Qt viewer."""

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
) -> FigureWorkbenchWindow:
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
    "FigureWorkbenchWindow",
    "open_data_figure_workbench",
    "open_figure_workbench",
]
