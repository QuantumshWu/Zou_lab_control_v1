"""Frozen or encoded raster presenter with pixel-only zoom and pan."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_storage import canonical_text

from ..render import BoardFrame
from ._raster_front import (
    _aspect_target_for_source,
    _prepared_qimage,
)
from .style import BG


def _full_image_rect(image: QtGui.QImage) -> QtCore.QRectF:
    return QtCore.QRectF(
        0.0,
        0.0,
        float(image.width()),
        float(image.height()),
    )


class FrozenRasterView(QtWidgets.QWidget):
    """Single-panel BoardPresenter that paints directly from immutable bytes."""

    normalizedDoubleClicked = QtCore.pyqtSignal(float, float)

    def __init__(
        self,
        panel_id: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        empty_text: str = "",
        zoomable: bool = False,
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
        if not isinstance(zoomable, bool):
            raise TypeError("zoomable must be bool")
        self._zoomable = zoomable
        # Normalised view of the source rect: centre plus a magnification.  A
        # frozen report page is the one raster the operator cannot re-render at
        # a larger size, so being unable to magnify it is the difference between
        # reading a per-site fit and guessing at it.  Off by default: a live
        # board's zoom belongs to its ViewportTransform, not to the presenter.
        self._view_center = (0.5, 0.5)
        self._view_scale = 1.0
        self._pan_from: QtCore.QPoint | None = None
        self._front_size: tuple[int, int] | None = None
        self.setMinimumSize(64, 64)

    def present(self, frame: BoardFrame) -> None:
        self._require_owner()
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        if len(frame.panels) != 1 or frame.panels[0].panel_id != self._panel_id:
            raise ValueError("FrozenRasterView requires its one configured panel")
        self._front = _prepared_qimage(frame.panels[0])
        self._front_frame = frame
        self._note_front_geometry(self._front[1])
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
        self._note_front_geometry(image)
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self._front_frame = None
        self._front_size = None
        self._reset_view()
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
        painter.drawImage(
            QtCore.QRectF(_aspect_target_for_source(self.rect(), source)),
            image,
            self._magnified_source(source),
        )

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._zoomable and self._view_scale > 1.0:
            # Zoomed in, double-click means "show me the whole page again".
            # Un-zoomed it keeps the meaning every existing consumer relies on.
            self._reset_view()
            event.accept()
            self.update()
            return
        front = self._front
        if front is not None:
            source = _full_image_rect(front[1])
            target = _aspect_target_for_source(self.rect(), source)
            position = event.pos()
            if target.contains(position) and target.width() > 0 and target.height() > 0:
                self.normalizedDoubleClicked.emit(
                    (position.x() - target.x()) / target.width(),
                    (position.y() - target.y()) / target.height(),
                )
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------ zoom / pan
    #: A report page is worth magnifying, not resampling; 16x is already past
    #: the point where one source pixel fills a screen tile.
    MAX_VIEW_SCALE = 16.0

    @property
    def view_scale(self) -> float:
        """Current magnification; 1.0 means the whole page is shown."""

        return self._view_scale

    @property
    def view_center(self) -> tuple[float, float]:
        """Normalised centre of the visible sub-rect within the source raster."""

        return self._view_center

    def _reset_view(self) -> None:
        self._view_center = (0.5, 0.5)
        self._view_scale = 1.0
        self._pan_from = None

    def _note_front_geometry(self, image: QtGui.QImage) -> None:
        size = (int(image.width()), int(image.height()))
        # A refreshed page of the same geometry keeps the operator's zoom; a
        # different raster is a different picture, so the view starts over.
        if size != self._front_size:
            self._front_size = size
            self._reset_view()

    def _magnified_source(self, source: QtCore.QRect) -> QtCore.QRectF:
        base = QtCore.QRectF(source)
        if not self._zoomable or self._view_scale <= 1.0:
            return base
        span = 1.0 / self._view_scale
        cx, cy = self._view_center
        return QtCore.QRectF(
            base.x() + (cx - span / 2.0) * base.width(),
            base.y() + (cy - span / 2.0) * base.height(),
            span * base.width(),
            span * base.height(),
        )

    def _clamped_center(
        self,
        cx: float,
        cy: float,
        scale: float,
    ) -> tuple[float, float]:
        """Keep the visible sub-rect inside the page: no empty margins."""

        half = 0.5 / scale
        low, high = half, 1.0 - half
        if low > high:
            return (0.5, 0.5)
        return (min(max(cx, low), high), min(max(cy, low), high))

    def _target_rect(self):
        front = self._front
        if front is None:
            return None
        source = _full_image_rect(front[1])
        target = _aspect_target_for_source(self.rect(), source)
        if target.width() <= 0 or target.height() <= 0:
            return None
        return target

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        target = self._target_rect() if self._zoomable else None
        if target is None or not target.contains(event.pos()):
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            super().wheelEvent(event)
            return
        scale = min(
            max(self._view_scale * (1.25 ** steps), 1.0),
            self.MAX_VIEW_SCALE,
        )
        # Hold the page point under the cursor still, so magnifying reads as
        # "look closer HERE" rather than "jump to the middle".
        u = (event.pos().x() - target.x()) / target.width()
        v = (event.pos().y() - target.y()) / target.height()
        cx, cy = self._view_center
        px = cx - 0.5 / self._view_scale + u / self._view_scale
        py = cy - 0.5 / self._view_scale + v / self._view_scale
        self._view_scale = scale
        self._view_center = self._clamped_center(
            px + 0.5 / scale - u / scale,
            py + 0.5 / scale - v / scale,
            scale,
        )
        event.accept()
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._zoomable
            and event.button() == QtCore.Qt.LeftButton
            and self._view_scale > 1.0
        ):
            self._pan_from = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        # Current button state is authoritative.  A retained press origin must
        # never turn a no-button move into an implicit plot interaction when a
        # release was delivered elsewhere or the window lost capture.
        if event.buttons() == QtCore.Qt.NoButton:
            super().mouseMoveEvent(event)
            return
        target = None if self._pan_from is None else self._target_rect()
        if target is None:
            super().mouseMoveEvent(event)
            return
        delta = event.pos() - self._pan_from
        self._pan_from = event.pos()
        cx, cy = self._view_center
        self._view_center = self._clamped_center(
            cx - (delta.x() / target.width()) / self._view_scale,
            cy - (delta.y() / target.height()) / self._view_scale,
            self._view_scale,
        )
        event.accept()
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pan_from is not None:
            self._pan_from = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("FrozenRasterView presentation is GUI-thread affine")

__all__ = ["FrozenRasterView"]
