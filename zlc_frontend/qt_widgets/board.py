"""Qt widgets for one immutable live-image board front."""

from __future__ import annotations

import math
from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data import Selection
from zlc_storage import canonical_text, nonnegative_integer

from ..render import BoardFrame, PixelFormat, SourceIdentity, detached_render_fault
from ..selector import (
    ImageViewportTransform,
    NormalizedRectangle,
    RectangleGesture,
)
from .style import BG, GREEN, ORANGE


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


def _panel_bounds(
    bounds: QtCore.QRect,
    *,
    index: int,
    count: int,
    columns: int,
) -> QtCore.QRect:
    """Return the one grid cell used by both raster paint and hit testing."""

    rows = math.ceil(count / columns)
    cell_width = bounds.width() // columns
    cell_height = bounds.height() // rows
    row, column = divmod(index, columns)
    right = (
        bounds.right() + 1
        if column == columns - 1
        else bounds.x() + (column + 1) * cell_width
    )
    bottom = (
        bounds.bottom() + 1
        if row == rows - 1
        else bounds.y() + (row + 1) * cell_height
    )
    return QtCore.QRect(
        bounds.x() + column * cell_width,
        bounds.y() + row * cell_height,
        right - (bounds.x() + column * cell_width),
        bottom - (bounds.y() + row * cell_height),
    )


def _validated_panel_layout(
    panel_ids: tuple[str, ...],
    columns: int,
) -> tuple[tuple[str, ...], int]:
    ids = tuple(canonical_text(value, "panel_id") for value in panel_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("QtRasterBoard requires unique panel ids")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError("columns must be a positive integer")
    return ids, min(columns, len(ids))


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

    normalizedDoubleClicked = QtCore.pyqtSignal(float, float)

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
        self._front_frame: BoardFrame | None = None
        self.setMinimumSize(64, 64)

    def present(self, frame: BoardFrame) -> None:
        self._require_owner()
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        if len(frame.panels) != 1 or frame.panels[0].panel_id != self._panel_id:
            raise ValueError("QtImageBoard requires its one configured panel")
        self._front = _prepared_qimage(frame.panels[0].raster)
        self._front_frame = frame
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
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self._front_frame = None
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
        painter.drawImage(_aspect_target(self.rect(), image), image)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        front = self._front
        if front is not None:
            target = _aspect_target(self.rect(), front[1])
            position = event.pos()
            if target.contains(position) and target.width() > 0 and target.height() > 0:
                self.normalizedDoubleClicked.emit(
                    (position.x() - target.x()) / target.width(),
                    (position.y() - target.y()) / target.height(),
                )
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

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
        self._panel_ids, self._columns = _validated_panel_layout(panel_ids, columns)
        self._active_layout_identity: tuple[str, int] | None = None
        self._staged_layout: tuple[str, int, tuple[str, ...], int] | None = None
        self._empty_text = str(empty_text)
        self._front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None = None
        self._selector_panel_id: str | None = None
        self._selector_viewport: ImageViewportTransform | None = None
        self._selector_callback: Callable[[RectangleGesture], object] | None = None
        self._selector_interaction_callback: Callable[[bool], object] | None = None
        self._selector_interaction_announced = False
        self._selector_enabled = False
        self._selector_applied_bounds: NormalizedRectangle | None = None
        self._selector_draft_bounds: NormalizedRectangle | None = None
        self._selector_drag_anchor: tuple[float, float] | None = None
        self._selector_drag_front: tuple[
            str,
            int,
            int,
            SourceIdentity,
            int,
        ] | None = None
        self._selector_fault: RuntimeError | None = None
        self._closed = False
        self.setMinimumSize(128, 64)

    @property
    def panel_ids(self) -> tuple[str, ...]:
        """Panel order of the currently visible/active layout."""

        self._require_owner()
        return self._panel_ids

    @property
    def columns(self) -> int:
        self._require_owner()
        return self._columns

    def stage_layout(
        self,
        panel_ids: tuple[str, ...],
        *,
        board_id: str,
        layout_generation: int,
        columns: int = 2,
    ) -> None:
        """Admit one newer layout without disturbing the currently painted front."""

        self._require_owner()
        self._ensure_open()
        ids, column_count = _validated_panel_layout(panel_ids, columns)
        identity = (
            canonical_text(board_id, "board_id"),
            nonnegative_integer(layout_generation, "layout_generation"),
        )
        current = self._staged_layout
        floor = (
            self._active_layout_identity
            if current is None
            else (current[0], current[1])
        )
        if floor is not None:
            if identity[0] != floor[0]:
                raise ValueError("QtRasterBoard cannot change board identity")
            if identity[1] <= floor[1]:
                raise ValueError("staged layout_generation must increase")
        self._staged_layout = (*identity, ids, column_count)
        if self._selector_interaction_announced:
            self._cancel_rectangle_gesture(clear_draft=True)
            self.update()

    def discard_staged_layout(
        self,
        *,
        board_id: str,
        layout_generation: int,
    ) -> bool:
        """Discard only the named unpresented layout, preserving the old front."""

        self._require_owner()
        if self._closed:
            return False
        identity = (
            canonical_text(board_id, "board_id"),
            nonnegative_integer(layout_generation, "layout_generation"),
        )
        staged = self._staged_layout
        if staged is None or (staged[0], staged[1]) != identity:
            return False
        self._staged_layout = None
        self.update()
        return True

    def present(self, frame: BoardFrame) -> None:
        self._require_owner()
        self._ensure_open()
        interaction_was_active = self._selector_interaction_announced
        promoting = False
        target_panel_ids = self._panel_ids
        target_columns = self._columns
        target_identity = self._active_layout_identity
        try:
            if not isinstance(frame, BoardFrame):
                raise TypeError("frame must be BoardFrame")
            frame_identity = (frame.board_id, frame.layout_generation)
            frame_panel_ids = tuple(panel.panel_id for panel in frame.panels)
            staged = self._staged_layout
            if staged is not None:
                staged_identity = (staged[0], staged[1])
                if frame_identity != staged_identity or frame_panel_ids != staged[2]:
                    raise ValueError(
                        "QtRasterBoard frame does not match its staged layout identity"
                    )
                promoting = True
                target_identity = staged_identity
                target_panel_ids = staged[2]
                target_columns = staged[3]
            elif frame_panel_ids != self._panel_ids:
                raise ValueError(
                    "QtRasterBoard frame does not match its configured panel order"
                )
            elif (
                self._active_layout_identity is not None
                and frame_identity != self._active_layout_identity
            ):
                raise ValueError(
                    "QtRasterBoard frame does not match its active layout identity"
                )
            else:
                target_identity = frame_identity
            prepared = tuple(_prepared_qimage(panel.raster) for panel in frame.panels)
            if (
                self._selector_panel_id is not None
                and self._selector_panel_id in target_panel_ids
                and self._selector_viewport is not None
            ):
                self._validate_selector_binding(
                    self._selector_panel_id,
                    self._selector_viewport,
                    frame,
                    prepared,
                    panel_ids=target_panel_ids,
                )
        except BaseException:
            if promoting:
                self._staged_layout = None
            if interaction_was_active:
                self._cancel_rectangle_gesture(clear_draft=True)
                self.update()
            raise
        previous = self._front
        if self._selector_panel_id is not None:
            panel_id = self._selector_panel_id
            if panel_id not in target_panel_ids:
                self._reset_rectangle_selector()
            elif previous is not None:
                old_index = self._panel_ids.index(panel_id)
                new_index = target_panel_ids.index(panel_id)
                if (
                    previous[0].panels[old_index].source_identity
                    != frame.panels[new_index].source_identity
                ):
                    self._selector_applied_bounds = None
                    self._selector_draft_bounds = None
        if promoting:
            self._panel_ids = target_panel_ids
            self._columns = target_columns
            self._staged_layout = None
        self._active_layout_identity = target_identity
        self._front = (frame, prepared)
        if interaction_was_active:
            self._cancel_rectangle_gesture(clear_draft=True)
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self._active_layout_identity = None
        self._staged_layout = None
        self._cancel_rectangle_gesture(clear_draft=True)
        self._selector_applied_bounds = None
        self.update()

    @property
    def has_front(self) -> bool:
        return self._front is not None

    @property
    def front_frame(self) -> BoardFrame | None:
        return None if self._front is None else self._front[0]

    @property
    def selector_fault(self) -> RuntimeError | None:
        self._require_owner()
        return self._selector_fault

    def bind_rectangle_selector(
        self,
        panel_id: str,
        viewport: ImageViewportTransform,
        callback: Callable[[RectangleGesture], object],
        *,
        interaction_callback: Callable[[bool], object],
        enabled: bool = True,
    ) -> None:
        """Bind one image panel without giving the widget a runtime control sink.

        ``interaction_callback`` is a synchronous, balanced owner-thread gate:
        ``True`` freezes live-front presentation before press handling returns;
        every release or cancellation reports ``False``.  Live consumers must
        gate presentation here rather than poll widget state.
        """

        self._require_owner()
        panel_id = canonical_text(panel_id, "selector panel_id")
        if panel_id not in self._panel_ids:
            raise ValueError("selector panel_id is absent from this board")
        if not isinstance(viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        if not callable(callback):
            raise TypeError("selector callback must be callable")
        if not callable(interaction_callback):
            raise TypeError("selector interaction_callback must be callable")
        if not isinstance(enabled, bool):
            raise TypeError("selector enabled must be bool")
        if self._front is not None:
            self._validate_selector_binding(
                panel_id,
                viewport,
                self._front[0],
                self._front[1],
            )
        if not self._reset_rectangle_selector():
            self.update()
            raise RuntimeError(
                "previous selector interaction could not be released; "
                "inspect selector_fault"
            )
        self._selector_fault = None
        self._selector_panel_id = panel_id
        self._selector_viewport = viewport
        self._selector_callback = callback
        self._selector_interaction_callback = interaction_callback
        self._selector_enabled = enabled
        self.update()

    def set_rectangle_selector_enabled(self, enabled: bool) -> None:
        self._require_owner()
        if not isinstance(enabled, bool):
            raise TypeError("selector enabled must be bool")
        if enabled and (
            self._selector_panel_id is None
            or self._selector_viewport is None
            or self._selector_callback is None
        ):
            raise RuntimeError("rectangle selector is not bound")
        if enabled and self._selector_fault is not None:
            raise RuntimeError("rectangle selector must be rebound after a callback fault")
        self._selector_enabled = enabled
        if not enabled:
            self._cancel_rectangle_gesture(
                clear_draft=self._selector_drag_anchor is not None
            )
        self.update()

    def set_selector_applied_selection(self, selection: Selection | None) -> None:
        self._require_owner()
        viewport = self._require_selector_viewport()
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("applied selection must be zlc_data.Selection or None")
        self._selector_applied_bounds = (
            None
            if selection is None
            else viewport.normalized_bounds_for_selection(selection)
        )
        self.update()

    def set_selector_draft_selection(self, selection: Selection | None) -> None:
        self._require_owner()
        viewport = self._require_selector_viewport()
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("draft selection must be zlc_data.Selection or None")
        self._selector_draft_bounds = (
            None
            if selection is None
            else viewport.normalized_bounds_for_selection(selection)
        )
        self._cancel_rectangle_gesture(clear_draft=False)
        self.update()

    def unbind_rectangle_selector(self) -> None:
        self._require_owner()
        self._reset_rectangle_selector()
        self.update()

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
        for index, (_pixels, image) in enumerate(images):
            bounds = _panel_bounds(
                self.rect(),
                index=index,
                count=len(images),
                columns=self._columns,
            )
            painter.drawImage(_aspect_target(bounds, image), image)
        self._paint_selector_overlays(painter)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._selector_enabled or event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        target = self._selector_target()
        if target is None or not target[0].contains(event.pos()):
            super().mousePressEvent(event)
            return
        image_target = target[0]
        point = self._normalized_point(event.localPos(), image_target, clamp=False)
        bounds = self._selector_draft_bounds or self._selector_applied_bounds
        handle = None if bounds is None else self._hit_corner_handle(event.pos(), bounds, image_target)
        self._selector_drag_anchor = (
            point if handle is None else self._opposite_corner_anchor(bounds, handle)
        )
        self._selector_drag_front = self._selector_front_identity(target)
        if not self._notify_selector_interaction(True):
            self._cancel_rectangle_gesture(clear_draft=True)
            self.update()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        anchor = self._selector_drag_anchor
        if anchor is None:
            super().mouseMoveEvent(event)
            return
        target = self._selector_target()
        if target is None:
            event.accept()
            return
        point = self._normalized_point(event.localPos(), target[0], clamp=True)
        if point[0] == anchor[0] or point[1] == anchor[1]:
            self._selector_draft_bounds = None
        else:
            self._selector_draft_bounds = self._require_selector_viewport().snapped_bounds_for_drag(
                anchor,
                point,
            )
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        anchor = self._selector_drag_anchor
        if anchor is None or event.button() != QtCore.Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        target = self._selector_target()
        if target is not None:
            point = self._normalized_point(event.localPos(), target[0], clamp=True)
            if point[0] == anchor[0] or point[1] == anchor[1]:
                self._selector_draft_bounds = None
            else:
                self._selector_draft_bounds = self._require_selector_viewport().snapped_bounds_for_drag(
                    anchor,
                    point,
                )
        current = None if target is None else self._selector_front_identity(target)
        frozen = self._selector_drag_front
        bounds = self._selector_draft_bounds
        callback = self._selector_callback
        stale = current != frozen
        delivered = False
        # Geometry is complete before the consumer callback runs, but the
        # synchronous front gate remains active until that callback returns.
        # A re-entrant PREPARING/disable transition must therefore preserve
        # this completed draft rather than classify it as a partial drag.
        self._selector_drag_anchor = None
        self._selector_drag_front = None
        if not stale and bounds is not None and frozen is not None and callback is not None:
            assert self._selector_panel_id is not None
            gesture = RectangleGesture(
                panel_id=self._selector_panel_id,
                board_id=frozen[0],
                layout_generation=frozen[1],
                sequence=frozen[2],
                source_identity=frozen[3],
                normalized_bounds=bounds,
                viewport_revision=frozen[4],
            )
            try:
                callback(gesture)
                delivered = True
            except BaseException as error:
                if self._selector_fault is None:
                    self._selector_fault = detached_render_fault(error)
                self._selector_enabled = False
        self._cancel_rectangle_gesture(
            clear_draft=stale or (bounds is not None and not delivered)
        )
        self.update()
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self._selector_interaction_announced:
            self._cancel_rectangle_gesture(clear_draft=True)
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._reset_rectangle_selector()
        self._front = None
        self._active_layout_identity = None
        self._staged_layout = None
        self._closed = True
        super().closeEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.DeferredDelete:
            self._reset_rectangle_selector()
            self._front = None
            self._active_layout_identity = None
            self._staged_layout = None
            self._closed = True
        elif event.type() in (QtCore.QEvent.Hide, QtCore.QEvent.WindowDeactivate) and (
            getattr(self, "_selector_interaction_announced", False)
        ):
            self._cancel_rectangle_gesture(clear_draft=True)
            self.update()
        return super().event(event)

    def _require_selector_viewport(self) -> ImageViewportTransform:
        viewport = self._selector_viewport
        if viewport is None:
            raise RuntimeError("rectangle selector is not bound")
        return viewport

    def _validate_selector_binding(
        self,
        panel_id: str,
        viewport: ImageViewportTransform,
        frame,
        prepared,
        *,
        panel_ids: tuple[str, ...] | None = None,
    ) -> None:
        configured_ids = self._panel_ids if panel_ids is None else panel_ids
        index = configured_ids.index(panel_id)
        image = prepared[index][1]
        expected_height, expected_width = viewport.raster_shape
        if image.width() != expected_width or image.height() != expected_height:
            raise ValueError(
                "selector viewport axes do not match the selected panel raster geometry"
            )
        if frame.panels[index].panel_id != panel_id:
            raise ValueError("selector panel identity changed")

    def _selector_target(self):
        front = self._front
        panel_id = self._selector_panel_id
        viewport = self._selector_viewport
        if front is None or panel_id is None or viewport is None:
            return None
        index = self._panel_ids.index(panel_id)
        image = front[1][index][1]
        bounds = _panel_bounds(
            self.rect(),
            index=index,
            count=len(front[1]),
            columns=self._columns,
        )
        target = _aspect_target(bounds, image)
        return target, front[0], front[0].panels[index]

    def _selector_front_identity(self, target):
        viewport = self._require_selector_viewport()
        frame, panel = target[1], target[2]
        return (
            frame.board_id,
            frame.layout_generation,
            frame.sequence,
            panel.source_identity,
            viewport.viewport_revision,
        )

    @staticmethod
    def _normalized_point(
        point: QtCore.QPointF,
        target: QtCore.QRect,
        *,
        clamp: bool,
    ) -> tuple[float, float]:
        x = (float(point.x()) - target.x()) / max(1, target.width())
        y = (float(point.y()) - target.y()) / max(1, target.height())
        if clamp:
            return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("pointer lies outside the selected image viewport")
        return x, y

    def _opposite_corner_anchor(
        self,
        bounds: NormalizedRectangle,
        handle: int,
    ) -> tuple[float, float]:
        left, top, right, bottom = bounds
        viewport = self._require_selector_viewport()
        x_half_cell = 0.5 / viewport.x_axis.size
        y_half_cell = 0.5 / viewport.y_axis.size
        return (
            (
                (right - x_half_cell, bottom - y_half_cell),
                (left + x_half_cell, bottom - y_half_cell),
                (right - x_half_cell, top + y_half_cell),
                (left + x_half_cell, top + y_half_cell),
            )[handle]
        )

    @staticmethod
    def _overlay_rect(
        bounds: NormalizedRectangle,
        target: QtCore.QRect,
    ) -> QtCore.QRectF:
        left, top, right, bottom = bounds
        return QtCore.QRectF(
            target.x() + left * target.width(),
            target.y() + top * target.height(),
            (right - left) * target.width(),
            (bottom - top) * target.height(),
        )

    def _hit_corner_handle(
        self,
        point: QtCore.QPoint,
        bounds: NormalizedRectangle,
        target: QtCore.QRect,
    ) -> int | None:
        rectangle = self._overlay_rect(bounds, target)
        corners = (
            rectangle.topLeft(),
            rectangle.topRight(),
            rectangle.bottomLeft(),
            rectangle.bottomRight(),
        )
        radius = 7.0
        for index, corner in enumerate(corners):
            if (
                abs(point.x() - corner.x()) <= radius
                and abs(point.y() - corner.y()) <= radius
            ):
                return index
        return None

    def _paint_selector_overlays(self, painter: QtGui.QPainter) -> None:
        target = self._selector_target()
        if target is None:
            return
        image_target = target[0]
        painter.save()
        painter.setClipRect(image_target)
        if self._selector_applied_bounds is not None:
            self._paint_selector_rectangle(
                painter,
                self._selector_applied_bounds,
                image_target,
                QtGui.QColor(GREEN),
                dashed=False,
                handles=(
                    self._selector_enabled
                    and self._selector_draft_bounds is None
                ),
            )
        if self._selector_draft_bounds is not None:
            self._paint_selector_rectangle(
                painter,
                self._selector_draft_bounds,
                image_target,
                QtGui.QColor(ORANGE),
                dashed=True,
                handles=self._selector_enabled,
            )
        painter.restore()

    def _paint_selector_rectangle(
        self,
        painter: QtGui.QPainter,
        bounds: NormalizedRectangle,
        target: QtCore.QRect,
        color: QtGui.QColor,
        *,
        dashed: bool,
        handles: bool,
    ) -> None:
        rectangle = self._overlay_rect(bounds, target)
        pen = QtGui.QPen(color, 2.0)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(rectangle)
        if not handles:
            return
        painter.setPen(QtGui.QPen(color, 1.0))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
        size = 8.0
        for corner in (
            rectangle.topLeft(),
            rectangle.topRight(),
            rectangle.bottomLeft(),
            rectangle.bottomRight(),
        ):
            painter.drawRect(
                QtCore.QRectF(
                    corner.x() - size / 2,
                    corner.y() - size / 2,
                    size,
                    size,
                )
            )

    def _cancel_rectangle_gesture(self, *, clear_draft: bool) -> bool:
        self._selector_drag_anchor = None
        self._selector_drag_front = None
        if clear_draft:
            self._selector_draft_bounds = None
        return self._notify_selector_interaction(False)

    def _notify_selector_interaction(self, active: bool) -> bool:
        callback = self._selector_interaction_callback
        if active:
            if self._selector_interaction_announced:
                return True
            if callback is None:
                return False
            self._selector_interaction_announced = True
        elif not self._selector_interaction_announced:
            return True
        else:
            self._selector_interaction_announced = False
        try:
            callback(active)
        except BaseException as error:
            if self._selector_fault is None:
                self._selector_fault = detached_render_fault(error)
            self._selector_enabled = False
            if active and self._selector_interaction_announced:
                self._selector_interaction_announced = False
                try:
                    callback(False)
                except BaseException:
                    pass
            return False
        return not active or (
            self._selector_interaction_announced
            and self._selector_drag_anchor is not None
        )

    def _reset_rectangle_selector(self) -> bool:
        released = self._cancel_rectangle_gesture(clear_draft=True)
        self._selector_applied_bounds = None
        self._selector_enabled = False
        self._selector_panel_id = None
        self._selector_viewport = None
        self._selector_callback = None
        self._selector_interaction_callback = None
        return released

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtRasterBoard presentation is GUI-thread affine")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("QtRasterBoard is closed")


__all__ = ["QtImageBoard", "QtOwnerWake", "QtRasterBoard"]
