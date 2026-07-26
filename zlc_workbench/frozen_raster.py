"""Shared nonblocking Qt shell for immutable, worker-rendered raster pages."""

from __future__ import annotations

from concurrent.futures import CancelledError, Executor, Future
import threading
from typing import Callable

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentLabel,
    FluentScrollArea,
    FluentTabWidget,
    GREY,
    FrozenRasterView,
)
from zlc_storage import canonical_text

from .window_runtime import (
    SerialWorkerWindow,
    error_summary,
    load_raster_bundle,
    open_workbench_window,
)


class FrozenRasterWindow(SerialWorkerWindow):
    """Present one atomic set of immutable PNG pages on the Qt owner thread."""

    def __init__(
        self,
        loader: Callable[[threading.Event], EncodedRasterDocument] | None,
        *,
        window_title: str,
        mode_text: str,
        loading_summary: str,
        object_prefix: str,
        subject: str,
        executor: Executor | None = None,
        worker_release: Callable[[], object] | None = None,
    ) -> None:
        super().__init__(executor=executor, worker_release=worker_release)
        if loader is not None and not callable(loader):
            raise TypeError("loader must be callable or None")
        self._prefix = canonical_text(object_prefix, "object_prefix")
        self._subject = canonical_text(subject, "subject").upper()
        self._bundle: EncodedRasterDocument | None = None
        self._boards: tuple[FrozenRasterView, ...] = ()
        self._tab_hosts: tuple[
            tuple[FrozenRasterView, QtWidgets.QWidget], ...
        ] = ()

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
        self._placeholder = FrozenRasterView(
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
        self._controls = controls
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.addWidget(self._mode)
        self._layout.addWidget(self._status)
        self._layout.addWidget(self._summary)
        self._layout.addWidget(self._tabs, 1)
        self._layout.addWidget(self._diagnostic)
        self._layout.addLayout(controls)

        self._close_button.clicked.connect(self.shutdown)
        if loader is not None:
            self._submit_future(
                load_raster_bundle,
                loader,
                self._cancelled,
            )

    @property
    def raster_ready(self) -> bool:
        return self._bundle is not None and bool(self._boards) and all(
            board.has_front for board in self._boards
        )

    def _worker_submit_failed(self, error: BaseException) -> None:
        self._status.setText(f"{self._subject} FAILED")
        self._diagnostic.setText(error_summary(error))

    def _worker_release_failed(self, error: BaseException) -> None:
        self._status.setText("CLOSE FAILED")
        self._diagnostic.setText(error_summary(error))

    def _build_boards(self, bundle: EncodedRasterDocument) -> tuple[FrozenRasterView, ...]:
        self._retire_tab_pages()
        boards = []
        tab_hosts = []
        one_page = len(bundle.pages) == 1
        for page in bundle.pages:
            scroll = FluentScrollArea(self._tabs)
            scroll.setWidgetResizable(False)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            board = FrozenRasterView(
                f"{self._prefix}-{page.key}",
                scroll,
                empty_text="Raster unavailable",
            )
            board.setMinimumSize(320, 240)
            board.setObjectName(
                f"{self._prefix}Board"
                if one_page
                else f"{self._prefix}Board_{page.key}"
            )
            scroll.setWidget(board)
            self._tabs.addTab(scroll, page.title)
            boards.append(board)
            tab_hosts.append((board, scroll))
        self._tab_hosts = tuple(tab_hosts)
        self._tabs.tabBar().setVisible(not one_page)
        return tuple(boards)

    def _tab_host_for_board(self, board: FrozenRasterView) -> QtWidgets.QWidget:
        """Return the actual tab page that owns one encoded raster board.

        Frozen pages are mounted inside non-resizing scroll areas.  Consumers
        that temporarily switch away from an encoded page must therefore move
        the scroll host, never detach its child raster and bypass scrolling.
        """

        for candidate, host in self._tab_hosts:
            if candidate is board:
                return host
        raise ValueError("raster board does not belong to this window")

    def _retire_tab_pages(self) -> None:
        """Drop every old Qt front before admitting a replacement bundle."""

        old_boards = self._boards
        self._boards = ()
        self._tab_hosts = ()
        for board in old_boards:
            board.clear()
        while self._tabs.count():
            widget = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if isinstance(widget, FrozenRasterView):
                widget.clear()
            widget.hide()
            widget.deleteLater()
        if self._placeholder is not None:
            self._placeholder = None

    def _present_bundle(self, bundle: EncodedRasterDocument) -> bool:
        boards: tuple[FrozenRasterView, ...] = ()
        try:
            boards = self._build_boards(bundle)
            self._boards = boards
            for page, board in zip(bundle.pages, boards, strict=True):
                board.present_encoded(page.png_bytes, image_format="PNG")
                board.adjustSize()
        except BaseException as error:
            for board in boards:
                board.clear()
            self._status.setText("DISPLAY FAILED")
            self._summary.setText(f"The frozen {self._subject.lower()} remains valid")
            self._diagnostic.setText(error_summary(error))
            return False
        self._bundle = bundle
        self._boards = boards
        self._status.setText("READY")
        self._summary.setText(bundle.summary)
        self._diagnostic.setText("")
        return True

    def _accept_finished_future(self, future: Future[EncodedRasterDocument]) -> None:
        try:
            bundle = future.result()
        except CancelledError:
            if not self._closing:
                self._status.setText(f"{self._subject} CANCELLED")
        except BaseException as error:
            if not self._closing:
                self._status.setText(f"{self._subject} FAILED")
                self._summary.setText("No raster was admitted")
                self._diagnostic.setText(error_summary(error))
        else:
            if not self._closing:
                self._present_bundle(bundle)

    def _clear_bundle(self) -> None:
        self._bundle = None
        for board in self._boards:
            board.clear()
        self._boards = ()
        self._tab_hosts = ()

    def _before_worker_shutdown(self) -> None:
        self._status.setText("CLOSING")
        self._close_button.setEnabled(False)
        self._clear_bundle()


def open_frozen_raster_window(
    loader: Callable[[threading.Event], EncodedRasterDocument],
    **options,
) -> FrozenRasterWindow:
    return open_workbench_window(lambda: FrozenRasterWindow(loader, **options))


__all__ = [
    "FrozenRasterWindow",
    "open_frozen_raster_window",
]
