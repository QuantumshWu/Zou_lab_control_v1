"""Shared nonblocking Qt shell for immutable, worker-rendered raster pages."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import math
import threading
from typing import Callable

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.image_raster import estimate_encoded_png_front_peak_nbytes
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentLabel,
    FluentTabWidget,
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
from zlc_storage import canonical_text, positive_integer


_FROZEN_RASTER_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zlc-frozen-raster",
)


def _error_summary(error: BaseException) -> str:
    message = str(error).strip()
    return type(error).__name__ if not message else f"{type(error).__name__}: {message}"


def _load_bundle(
    loader: Callable[[threading.Event], EncodedRasterDocument],
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    if cancelled.is_set():
        raise CancelledError()
    bundle = loader(cancelled)
    if not isinstance(bundle, EncodedRasterDocument):
        raise TypeError("frozen raster loader must return EncodedRasterDocument")
    if cancelled.is_set():
        raise CancelledError()
    if bundle.source_front_peak_nbytes > memory_limit_bytes:
        raise MemoryError(
            "encoded raster fronts require "
            f"{bundle.source_front_peak_nbytes} bytes; limit is {memory_limit_bytes}"
        )
    return bundle


class FrozenRasterWindow(QtWidgets.QWidget):
    """Present one atomic set of immutable PNG pages on the Qt owner thread."""

    def __init__(
        self,
        loader: Callable[[threading.Event], EncodedRasterDocument],
        *,
        window_title: str,
        mode_text: str,
        loading_summary: str,
        object_prefix: str,
        subject: str,
        memory_limit_bytes: int,
    ) -> None:
        super().__init__()
        if not callable(loader):
            raise TypeError("loader must be callable")
        self._memory_limit_bytes = positive_integer(
            memory_limit_bytes,
            "memory_limit_bytes",
        )
        self._prefix = canonical_text(object_prefix, "object_prefix")
        self._subject = canonical_text(subject, "subject").upper()
        self._future: Future[EncodedRasterDocument] | None = None
        self._cancelled = threading.Event()
        self._bundle: EncodedRasterDocument | None = None
        self._boards: tuple[QtImageBoard, ...] = ()
        self._closing = False
        self._closed = False
        self._allow_close = False

        self.setWindowTitle(canonical_text(window_title, "window_title"))
        self._mode = FluentLabel(canonical_text(mode_text, "mode_text"), self)
        self._mode.setObjectName(f"{self._prefix}Mode")
        self._status = FluentLabel(f"BUILDING {self._subject}", self)
        self._status.setObjectName(f"{self._prefix}Status")
        self._summary = FluentLabel(
            canonical_text(loading_summary, "loading_summary"),
            self,
        )
        self._summary.setObjectName(f"{self._prefix}Summary")
        self._summary.setWordWrap(True)
        self._tabs = FluentTabWidget(self)
        self._tabs.setObjectName(f"{self._prefix}Tabs")
        self._placeholder = QtImageBoard(
            f"{self._prefix}-loading",
            self._tabs,
            empty_text=f"Building frozen {self._subject.lower()}…",
        )
        self._placeholder.setMinimumSize(320, 240)
        self._tabs.addTab(self._placeholder, "Loading")
        self._diagnostic = FluentLabel("", self)
        self._diagnostic.setObjectName(f"{self._prefix}Diagnostic")
        self._diagnostic.setWordWrap(True)
        self._diagnostic.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._close_button = FluentButton("Close", self, color=GREY)
        self._close_button.setObjectName(f"close{self._subject.title()}Button")

        controls = QtWidgets.QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(self._close_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._mode)
        layout.addWidget(self._status)
        layout.addWidget(self._summary)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._diagnostic)
        layout.addLayout(controls)

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._close_button.clicked.connect(self.shutdown)
        try:
            future = _FROZEN_RASTER_EXECUTOR.submit(
                _load_bundle,
                loader,
                self._memory_limit_bytes,
                self._cancelled,
            )
        except BaseException as error:
            self._status.setText(f"{self._subject} FAILED")
            self._diagnostic.setText(_error_summary(error))
        else:
            self._future = future
            future.add_done_callback(lambda _done: self._wake.request_owner_wake())

    @property
    def worker_idle(self) -> bool:
        return self._future is None

    @property
    def raster_ready(self) -> bool:
        return self._bundle is not None and bool(self._boards) and all(
            board.has_front for board in self._boards
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def _build_boards(self, bundle: EncodedRasterDocument) -> tuple[QtImageBoard, ...]:
        self._tabs.clear()
        boards = []
        one_page = len(bundle.pages) == 1
        for page in bundle.pages:
            board = QtImageBoard(
                f"{self._prefix}-{page.key}",
                self._tabs,
                empty_text="Raster unavailable",
            )
            board.setMinimumSize(320, 240)
            board.setObjectName(
                f"{self._prefix}Board"
                if one_page
                else f"{self._prefix}Board_{page.key}"
            )
            self._tabs.addTab(board, page.title)
            boards.append(board)
        if one_page:
            self._tabs.tabBar().hide()
        return tuple(boards)

    def _presentation_peak(
        self,
        bundle: EncodedRasterDocument,
        boards: tuple[QtImageBoard, ...],
    ) -> int:
        total = 0
        for page, board in zip(bundle.pages, boards, strict=True):
            ratio = float(board.devicePixelRatioF())
            size = (
                max(1, math.ceil(board.width() * ratio)),
                max(1, math.ceil(board.height() * ratio)),
            )
            total += estimate_encoded_png_front_peak_nbytes(
                page.png_bytes,
                presentation_size=size,
            )
        return total

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        future = self._future
        if future is not None and future.done():
            self._future = None
            try:
                bundle = future.result()
            except CancelledError:
                if not self._closing:
                    self._status.setText(f"{self._subject} CANCELLED")
            except BaseException as error:
                if not self._closing:
                    self._status.setText(f"{self._subject} FAILED")
                    self._summary.setText("No raster was admitted")
                    self._diagnostic.setText(_error_summary(error))
            else:
                if not self._closing:
                    boards: tuple[QtImageBoard, ...] = ()
                    try:
                        boards = self._build_boards(bundle)
                        required = self._presentation_peak(bundle, boards)
                        if required > self._memory_limit_bytes:
                            raise MemoryError(
                                f"Qt {self._subject.lower()} presentation requires "
                                f"{required} bytes; limit is {self._memory_limit_bytes}"
                            )
                        for page, board in zip(bundle.pages, boards, strict=True):
                            board.present_encoded(page.png_bytes, image_format="PNG")
                    except BaseException as error:
                        for board in boards:
                            board.clear()
                        self._status.setText("DISPLAY FAILED")
                        self._summary.setText(f"The frozen {self._subject.lower()} remains valid")
                        self._diagnostic.setText(_error_summary(error))
                    else:
                        self._bundle = bundle
                        self._boards = boards
                        self._status.setText("READY")
                        self._summary.setText(bundle.summary)
                        self._diagnostic.setText("")
        self._finish_close_if_ready()

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        self._closing = True
        self._cancelled.set()
        self._status.setText("CLOSING")
        self._close_button.setEnabled(False)
        self._bundle = None
        for board in self._boards:
            board.clear()
        self._boards = ()
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


def open_frozen_raster_window(
    loader: Callable[[threading.Event], EncodedRasterDocument],
    **options,
) -> FrozenRasterWindow:
    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("frozen raster Workbench must be opened on the Qt GUI thread")
    set_fluent_scale(None)
    window = FrozenRasterWindow(loader, **options)
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


__all__ = [
    "FrozenRasterWindow",
    "open_frozen_raster_window",
]
