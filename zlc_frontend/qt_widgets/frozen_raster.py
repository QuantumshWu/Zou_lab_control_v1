"""Frozen or encoded raster presenter with physical-pixel fidelity."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_storage import canonical_text

from ..render import BoardFrame, RasterBuffer
from ._raster_front import _prepared_qimage
from .style import BG


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

__all__ = ["FrozenRasterView"]
