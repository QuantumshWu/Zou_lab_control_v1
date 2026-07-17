"""Qt widgets for one immutable live-image board front."""

from __future__ import annotations

import math
from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_storage import canonical_text

from ..render import BoardFrame, PixelFormat, detached_render_fault
from .style import BG


def _prepared_qimage(raster) -> tuple[bytes, QtGui.QImage]:
    formats = {
        PixelFormat.GRAY8: QtGui.QImage.Format_Grayscale8,
        PixelFormat.RGB888: QtGui.QImage.Format_RGB888,
        PixelFormat.RGBA8888: QtGui.QImage.Format_RGBA8888,
    }
    image = QtGui.QImage(
        raster.pixels,
        raster.width,
        raster.height,
        raster.stride_bytes,
        formats[raster.pixel_format],
    )
    if image.isNull():
        raise RuntimeError("Qt rejected the prepared immutable raster front")
    return raster.pixels, image


def _aspect_target(bounds: QtCore.QRect, image: QtGui.QImage) -> QtCore.QRect:
    scaled = image.size()
    scaled.scale(bounds.size(), QtCore.Qt.KeepAspectRatio)
    return QtCore.QRect(
        bounds.x() + (bounds.width() - scaled.width()) // 2,
        bounds.y() + (bounds.height() - scaled.height()) // 2,
        scaled.width(),
        scaled.height(),
    )


class QtOwnerWake(QtCore.QObject):
    """No-payload queued wake bound once to a GUI-owner callback."""

    requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._callback: Callable[[], object] | None = None
        self._fault: BaseException | None = None
        self.requested.connect(self._dispatch, QtCore.Qt.QueuedConnection)

    @property
    def fault(self) -> BaseException | None:
        return self._fault

    def bind(self, callback: Callable[[], object]) -> None:
        self._require_owner()
        if not callable(callback):
            raise TypeError("callback must be callable")
        if self._callback is not None:
            raise RuntimeError("QtOwnerWake is already bound")
        self._callback = callback
        self._fault = None

    def request_owner_wake(self) -> None:
        self.requested.emit()

    def detach(self) -> None:
        self._require_owner()
        self._callback = None

    @QtCore.pyqtSlot()
    def _dispatch(self) -> None:
        callback = self._callback
        if callback is None:
            return
        try:
            callback()
        except BaseException as error:
            self._fault = detached_render_fault(error)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtOwnerWake binding is GUI-thread affine")


class QtImageBoard(QtWidgets.QWidget):
    """Single-panel BoardPresenter that paints directly from immutable bytes."""

    def __init__(
        self,
        panel_id: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        empty_text: str = "",
    ) -> None:
        super().__init__(parent)
        self._panel_id = canonical_text(panel_id, "panel_id")
        self._empty_text = str(empty_text)
        self._front: tuple[bytes, QtGui.QImage] | None = None
        self.setMinimumSize(64, 64)

    def present(self, frame: BoardFrame) -> None:
        self._require_owner()
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        if len(frame.panels) != 1 or frame.panels[0].panel_id != self._panel_id:
            raise ValueError("QtImageBoard requires its one configured panel")
        self._front = _prepared_qimage(frame.panels[0].raster)
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
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self.update()

    @property
    def has_front(self) -> bool:
        return self._front is not None

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
        painter.drawImage(_aspect_target(self.rect(), image), image)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtImageBoard presentation is GUI-thread affine")


class QtRasterBoard(QtWidgets.QWidget):
    """Atomic multi-panel presenter for immutable worker-owned raster fronts."""

    def __init__(
        self,
        panel_ids: tuple[str, ...],
        parent: QtWidgets.QWidget | None = None,
        *,
        columns: int = 2,
        empty_text: str = "",
    ) -> None:
        super().__init__(parent)
        ids = tuple(canonical_text(value, "panel_id") for value in panel_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("QtRasterBoard requires unique panel ids")
        if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
            raise ValueError("columns must be a positive integer")
        self._panel_ids = ids
        self._columns = min(columns, len(ids))
        self._empty_text = str(empty_text)
        self._front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None = None
        self.setMinimumSize(128, 64)

    def present(self, frame: BoardFrame) -> None:
        self._require_owner()
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        if tuple(panel.panel_id for panel in frame.panels) != self._panel_ids:
            raise ValueError("QtRasterBoard frame does not match its configured panel order")
        prepared = tuple(_prepared_qimage(panel.raster) for panel in frame.panels)
        self._front = (frame, prepared)
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self.update()

    @property
    def has_front(self) -> bool:
        return self._front is not None

    @property
    def front_frame(self) -> BoardFrame | None:
        return None if self._front is None else self._front[0]

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
        images = front[1]
        rows = math.ceil(len(images) / self._columns)
        cell_width = self.width() // self._columns
        cell_height = self.height() // rows
        for index, (_pixels, image) in enumerate(images):
            row, column = divmod(index, self._columns)
            right = self.width() if column == self._columns - 1 else (column + 1) * cell_width
            bottom = self.height() if row == rows - 1 else (row + 1) * cell_height
            bounds = QtCore.QRect(
                column * cell_width,
                row * cell_height,
                right - column * cell_width,
                bottom - row * cell_height,
            )
            painter.drawImage(_aspect_target(bounds, image), image)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtRasterBoard presentation is GUI-thread affine")


__all__ = ["QtImageBoard", "QtOwnerWake", "QtRasterBoard"]
