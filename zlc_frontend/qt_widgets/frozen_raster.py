"""Frozen raster presenters and their shared nonblocking Qt shell."""

from __future__ import annotations

from concurrent.futures import CancelledError, Executor, Future
import threading
from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_storage import canonical_text

from ..encoded_raster import EncodedRasterDocument
from ..render import BoardFrame, RasterBuffer
from ._raster_front import _prepared_qimage
from .fluent import (
    FluentButton,
    FluentLabel,
    FluentScrollArea,
    FluentTabWidget,
)
from .owner_wake import SerialWorkerWindow, error_summary
from .style import BG, GREY


def _full_image_rect(image: QtGui.QImage) -> QtCore.QRectF:
    return QtCore.QRectF(
        0.0,
        0.0,
        float(image.width()),
        float(image.height()),
    )


class FrozenRasterView(QtWidgets.QWidget):
    """Single-panel raster widget that paints directly from immutable bytes."""

    normalizedDoubleClicked = QtCore.pyqtSignal(float, float)

    def __init__(
        self,
        panel_id: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        empty_text: str = "",
    ) -> None:
        super().__init__(parent)
        # Plot pointer motion has no standalone meaning.  Qt still delivers
        # move events while a pressed button owns a drag without mouse
        # tracking, so disabling it preserves pan while making ordinary moves
        # inert by construction.
        self.setMouseTracking(False)
        self._panel_id = canonical_text(panel_id, "panel_id")
        self._empty_text = str(empty_text)
        self._front: tuple[bytes, QtGui.QImage] | None = None
        self._front_frame: BoardFrame | None = None
        self.setMinimumSize(64, 64)

    def present(self, frame: BoardFrame) -> None:
        self._require_owner()
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        if len(frame.panels) != 1 or frame.panels[0].panel_id != self._panel_id:
            raise ValueError("FrozenRasterView requires its one configured panel")
        self._front = _prepared_qimage(frame.panels[0])
        self._front_frame = frame
        self.updateGeometry()
        self.update()

    def present_raster(self, raster: RasterBuffer) -> None:
        """Present one frontend-owned immutable RGBA raster without decoding."""

        self._require_owner()
        if not isinstance(raster, RasterBuffer):
            raise TypeError("raster must be RasterBuffer")
        self._front = _prepared_qimage(raster)
        self._front_frame = None
        self.updateGeometry()
        self.update()

    def present_encoded(self, payload: bytes, *, image_format: str = "PNG") -> None:
        """Decode one owned display artifact without changing board geometry."""
        self._require_owner()
        if not isinstance(payload, bytes):
            raise TypeError("payload must be owned immutable bytes")
        image_format = canonical_text(image_format, "image_format")
        image = QtGui.QImage.fromData(payload, image_format.encode("ascii"))
        if image.isNull():
            raise RuntimeError("Qt rejected the encoded immutable raster")
        self._front = (payload, image)
        self._front_frame = None
        self.updateGeometry()
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self._front_frame = None
        self.updateGeometry()
        self.update()

    @property
    def has_front(self) -> bool:
        return self._front is not None

    @property
    def front_frame(self) -> BoardFrame | None:
        return self._front_frame

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)
        front = self._front
        if front is None:
            if self._empty_text:
                painter.setPen(QtGui.QColor(BG))
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self._empty_text)
            return
        image = front[1]
        source = _full_image_rect(image)
        target = self._native_target(image)
        painter.drawImage(
            target,
            image,
            source,
        )

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        front = self._front
        if front is not None:
            target = self._native_target(front[1])
            position = event.pos()
            if target.contains(position) and target.width() > 0 and target.height() > 0:
                self.normalizedDoubleClicked.emit(
                    (position.x() - target.x()) / target.width(),
                    (position.y() - target.y()) / target.height(),
                )
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def _native_target(self, image: QtGui.QImage) -> QtCore.QRectF:
        """Centre a raster without resampling any source pixel.

        Qt widget geometry is logical pixels while the immutable raster is in
        physical pixels.  Dividing only by the live device-pixel ratio makes
        one source pixel map to one display pixel; surplus widget space is
        letterboxed and undersized hosts clip instead of inventing detail.
        """

        ratio = max(float(self.devicePixelRatioF()), 1.0)
        width = float(image.width()) / ratio
        height = float(image.height()) / ratio
        bounds = QtCore.QRectF(self.rect())
        return QtCore.QRectF(
            bounds.x() + (bounds.width() - width) / 2.0,
            bounds.y() + (bounds.height() - height) / 2.0,
            width,
            height,
        )

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        front = self._front
        if front is None:
            return super().sizeHint()
        ratio = max(float(self.devicePixelRatioF()), 1.0)
        image = front[1]
        return QtCore.QSize(
            max(1, int(round(image.width() / ratio))),
            max(1, int(round(image.height() / ratio))),
        )

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("FrozenRasterView presentation is GUI-thread affine")


def _load_raster_bundle(
    loader: Callable[[threading.Event], EncodedRasterDocument],
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    if cancelled.is_set():
        raise CancelledError()
    bundle = loader(cancelled)
    if not isinstance(bundle, EncodedRasterDocument):
        raise TypeError("raster loader must return EncodedRasterDocument")
    if cancelled.is_set():
        raise CancelledError()
    return bundle


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
                _load_raster_bundle,
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

    def _build_boards(
        self,
        bundle: EncodedRasterDocument,
    ) -> tuple[FrozenRasterView, ...]:
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
        for candidate, host in self._tab_hosts:
            if candidate is board:
                return host
        raise ValueError("raster board does not belong to this window")

    def _retire_tab_pages(self) -> None:
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

    def _accept_finished_future(
        self,
        future: Future[EncodedRasterDocument],
    ) -> None:
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


__all__ = ["FrozenRasterView", "FrozenRasterWindow"]
