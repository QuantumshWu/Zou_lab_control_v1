"""Optional Qt leaf for one immutable live-image board front."""

from __future__ import annotations

from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_storage import canonical_text

from .render import BoardFrame, PixelFormat, detached_render_fault


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
    ) -> None:
        super().__init__(parent)
        self._panel_id = canonical_text(panel_id, "panel_id")
        self._front: tuple[bytes, QtGui.QImage] | None = None
        self.setMinimumSize(64, 64)

    def present(self, frame: BoardFrame) -> None:
        self._require_owner()
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        if len(frame.panels) != 1 or frame.panels[0].panel_id != self._panel_id:
            raise ValueError("QtImageBoard requires its one configured panel")
        raster = frame.panels[0].raster
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
        prepared = (raster.pixels, image)
        self._front = prepared
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
            return
        image = front[1]
        scaled = image.size()
        scaled.scale(self.size(), QtCore.Qt.KeepAspectRatio)
        target = QtCore.QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )
        painter.drawImage(target, image)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtImageBoard presentation is GUI-thread affine")


__all__ = ["QtImageBoard", "QtOwnerWake"]
