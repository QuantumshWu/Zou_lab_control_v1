"""Qt widgets for one immutable live-image board front."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data import Selection
from zlc_storage import canonical_text, nonnegative_integer

from ..image_raster import indexed8_code_for_value
from ..render import (
    BoardFrame,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    PixelFormat,
    SourceIdentity,
    detached_render_fault,
)
from ..selector import (
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageInteractionOrigin,
    ImageViewportTransform,
    ImageViewportCommit,
    NormalizedRectangle,
    RectangleGesture,
)
from .style import BG, GREEN, ORANGE


def _prepared_qimage(panel_or_raster) -> tuple[bytes, QtGui.QImage]:
    """Prepare one owned Qt front, applying an IMAGE payload's exact LUT.

    Accepting a bare ``RasterBuffer`` preserves the small legacy presenter
    surface.  Interactive IMAGE panels pass ``PanelFrame`` so INDEXED8 colour
    limits are resolved from its paired immutable payload rather than guessed
    from display codes.
    """

    if isinstance(panel_or_raster, PanelFrame):
        raster = panel_or_raster.raster
        payload = panel_or_raster.image_payload
    else:
        raster = panel_or_raster
        payload = None
    formats = {
        PixelFormat.GRAY8: QtGui.QImage.Format_Grayscale8,
        PixelFormat.INDEXED8: QtGui.QImage.Format_Indexed8,
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
    if raster.pixel_format is PixelFormat.INDEXED8:
        if payload is None:
            raise ValueError(
                "INDEXED8 has no intrinsic colours and requires ImagePanelPayload"
            )
        # Worker quantization already spans the committed clim, so the one
        # sampled palette is installed directly.  A second LUT remap would
        # reintroduce the narrow-window precision loss this path avoids.
        image.setColorTable(list(payload.base_palette))
    return raster.pixels, image


def _image_source_rect(
    image: QtGui.QImage,
    viewport: ImageViewportTransform | None,
) -> QtCore.QRectF:
    if viewport is None:
        return QtCore.QRectF(0.0, 0.0, float(image.width()), float(image.height()))
    left, top, right, bottom = viewport.visible_bounds
    return QtCore.QRectF(
        left * image.width(),
        top * image.height(),
        (right - left) * image.width(),
        (bottom - top) * image.height(),
    )


def _aspect_target_for_source(
    bounds: QtCore.QRect,
    source: QtCore.QRectF,
) -> QtCore.QRect:
    if source.width() <= 0.0 or source.height() <= 0.0:
        raise ValueError("image source viewport must have positive geometry")
    source_ratio = source.width() / source.height()
    bounds_ratio = bounds.width() / max(1, bounds.height())
    if bounds_ratio > source_ratio:
        height = bounds.height()
        width = max(1, int(round(height * source_ratio)))
    else:
        width = bounds.width()
        height = max(1, int(round(width / source_ratio)))
    return QtCore.QRect(
        bounds.x() + (bounds.width() - width) // 2,
        bounds.y() + (bounds.height() - height) // 2,
        width,
        height,
    )


def _panel_image_geometry(
    bounds: QtCore.QRect,
    image: QtGui.QImage,
    payload: ImagePanelPayload | None,
) -> tuple[QtCore.QRect, QtCore.QRectF, QtCore.QRect | None]:
    source = _image_source_rect(
        image,
        None if payload is None else payload.viewport,
    )
    if payload is None:
        return _aspect_target_for_source(bounds, source), source, None
    rail_width = min(34, max(20, bounds.width() // 10))
    gap = 5
    image_bounds = QtCore.QRect(
        bounds.x(),
        bounds.y(),
        max(1, bounds.width() - rail_width - gap),
        bounds.height(),
    )
    target = _aspect_target_for_source(image_bounds, source)
    rail = QtCore.QRect(
        bounds.right() - rail_width + 1,
        target.top(),
        rail_width,
        target.height(),
    )
    return target, source, rail


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


def _panel_presentation(panel: PanelFrame) -> PanelPresentationIdentity:
    matches = tuple(
        value
        for value in panel.coherence_stamp.presentations
        if value.panel_id == panel.panel_id
    )
    if len(matches) != 1:
        raise RuntimeError("panel coherence stamp has no unique presentation identity")
    return matches[0]


def _raster_geometry(panel: PanelFrame) -> tuple[int, int, int, PixelFormat]:
    raster = panel.raster
    return (
        raster.width,
        raster.height,
        raster.stride_bytes,
        raster.pixel_format,
    )


@dataclass(frozen=True, slots=True)
class _ImageSample:
    """Exact painted sample used only by this board's visual overlays."""

    x_index: int
    y_index: int
    x_coordinate: object
    y_coordinate: object
    value: object
    valid: bool


@dataclass(frozen=True, slots=True)
class _HeldPanelFront:
    """One GUI-owned display overlay; it is never an authoritative BoardFrame."""

    panel_id: str
    board_id: str
    layout_generation: int
    sequence: int
    coherence_group: str
    source_identity: SourceIdentity
    presentation: PanelPresentationIdentity
    viewport_revision: int
    raster_geometry: tuple[int, int, int, PixelFormat]
    prepared: tuple[bytes, QtGui.QImage]
    image_payload: ImagePanelPayload | None

    @property
    def gesture_identity(self) -> tuple[str, int, int, SourceIdentity, int]:
        return (
            self.board_id,
            self.layout_generation,
            self.sequence,
            self.source_identity,
            self.viewport_revision,
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
        self._front = _prepared_qimage(frame.panels[0])
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
        payload = (
            None
            if self._front_frame is None
            else self._front_frame.panels[0].image_payload
        )
        source = _image_source_rect(
            image,
            None if payload is None else payload.viewport,
        )
        painter.drawImage(
            QtCore.QRectF(_aspect_target_for_source(self.rect(), source)),
            image,
            source,
        )

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        front = self._front
        if front is not None:
            payload = (
                None
                if self._front_frame is None
                else self._front_frame.panels[0].image_payload
            )
            source = _image_source_rect(
                front[1],
                None if payload is None else payload.viewport,
            )
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
        self._interaction_callback: Callable[[ImageInteractionCommit], object] | None = None
        self._selector_enabled = False
        self._selector_applied_bounds: NormalizedRectangle | None = None
        self._selector_draft_bounds: NormalizedRectangle | None = None
        self._selector_drag_anchor: tuple[float, float] | None = None
        self._pan_anchor: QtCore.QPointF | None = None
        self._pan_origin: ImageViewportTransform | None = None
        self._pan_target_size: tuple[int, int] | None = None
        self._pan_candidate: ImageViewportTransform | None = None
        self._pending_viewport: ImageViewportTransform | None = None
        self._pending_color_limits: tuple[float, float] | None = None
        self._pending_origin: ImageInteractionOrigin | None = None
        self._clim_drag: str | None = None
        self._clim_origin_limits: tuple[float, float] | None = None
        self._clim_candidate: tuple[float, float] | None = None
        self._clim_domain: tuple[float, float] | None = None
        self._selector_hold: _HeldPanelFront | None = None
        self._cross_sample: _ImageSample | None = None
        self._hover_sample: _ImageSample | None = None
        self._hover_position: QtCore.QPointF | None = None
        self._selector_fault: RuntimeError | None = None
        self._closed = False
        self.setMouseTracking(True)
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
        if self._selector_hold is not None:
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
        interaction_was_active = self._selector_hold is not None
        promoting = False
        cancel_interaction = False
        target_panel_ids = self._panel_ids
        target_columns = self._columns
        target_identity = self._active_layout_identity
        target_viewport = self._selector_viewport
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
            if (
                self._selector_panel_id is not None
                and self._selector_panel_id in target_panel_ids
                and self._selector_viewport is not None
            ):
                target_viewport = self._viewport_for_presented_panel(
                    self._selector_panel_id,
                    self._selector_viewport,
                    frame,
                    panel_ids=target_panel_ids,
                )
                self._validate_selector_binding(
                    self._selector_panel_id,
                    target_viewport,
                    frame,
                    panel_ids=target_panel_ids,
                )
                pending_origin = self._pending_origin
                pending_limits = self._pending_color_limits
                target_panel = frame.panels[
                    target_panel_ids.index(self._selector_panel_id)
                ]
                if (
                    self._interaction_callback is not None
                    and target_panel.image_payload is None
                ):
                    raise ValueError(
                        "image interaction callback requires exact ImagePanelPayload"
                    )
                if (
                    pending_origin is not None
                    and pending_limits is not None
                    and target_panel.image_payload is not None
                    and target_viewport.viewport_revision
                    == pending_origin.presentation.panel_revision + 1
                    and target_panel.image_payload.color_limits != pending_limits
                ):
                    raise ValueError(
                        "pending image color-limit revision returned conflicting limits"
                    )
            if interaction_was_active:
                hold = self._selector_hold
                if hold is None:
                    raise RuntimeError(
                        "active rectangle interaction has no held panel front"
                    )
                cancel_interaction = not self._hold_matches_frame(
                    hold,
                    frame,
                    panel_ids=target_panel_ids,
                )
            # Only after all cheap identity/revision checks pass may INDEXED8
            # setColorTable detach a potentially multi-megapixel QImage plane.
            prepared = tuple(_prepared_qimage(panel) for panel in frame.panels)
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
                old_panel = previous[0].panels[old_index]
                new_panel = frame.panels[new_index]
                if self._panel_semantics_changed(old_panel, new_panel):
                    self._selector_applied_bounds = None
                    self._selector_draft_bounds = None
                    self._set_cross_sample(None)
                    self._pending_viewport = None
                    self._pending_color_limits = None
                    self._pending_origin = None
                    cancel_interaction = interaction_was_active
        if cancel_interaction:
            self._cancel_rectangle_gesture(clear_draft=True)
        if promoting:
            self._panel_ids = target_panel_ids
            self._columns = target_columns
            self._staged_layout = None
        self._active_layout_identity = target_identity
        self._selector_viewport = (
            target_viewport if self._selector_panel_id is not None else None
        )
        pending = self._pending_viewport
        if pending is not None and target_viewport is not None:
            if (
                target_viewport == pending
                or target_viewport.viewport_revision > pending.viewport_revision
            ):
                self._pending_viewport = None
        pending_origin = self._pending_origin
        if (
            self._pending_color_limits is not None
            and target_viewport is not None
            and pending_origin is not None
            and target_viewport.viewport_revision
            > pending_origin.presentation.panel_revision
        ):
            self._pending_color_limits = None
        if not self._interaction_is_pending():
            self._pending_origin = None
        hover_position = self._hover_position
        self._front = (frame, prepared)
        if (
            self._selector_enabled
            and not self._interaction_is_pending()
            and hover_position is not None
        ):
            target = self._selector_target()
            sample = (
                None
                if target is None or not target[0].contains(hover_position.toPoint())
                else self._sample_for_target(target, hover_position)
            )
            if sample is not None:
                self._hover_position = QtCore.QPointF(hover_position)
            self._set_hover_sample(sample)
        else:
            self._set_hover_sample(None)
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self._active_layout_identity = None
        self._staged_layout = None
        self._cancel_rectangle_gesture(clear_draft=True)
        self._pending_viewport = None
        self._pending_color_limits = None
        self._pending_origin = None
        self._selector_applied_bounds = None
        self._set_cross_sample(None)
        self._set_hover_sample(None)
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

    def visible_image_payload(self) -> ImagePanelPayload | None:
        """Return the exact samples paired with the currently painted IMAGE.

        During A/pan interaction this is the held target payload, not the
        advancing board front.  Setting/Edit can therefore freeze FIXED limits
        from exactly what the operator sees without retaining a BoardFrame.
        """

        self._require_owner()
        hold = self._selector_hold
        if hold is not None:
            return hold.image_payload
        front = self._front
        panel_id = self._selector_panel_id
        if front is None or panel_id is None or panel_id not in self._panel_ids:
            return None
        return front[0].panels[self._panel_ids.index(panel_id)].image_payload

    def visible_image_origin(self) -> ImageInteractionOrigin | None:
        """Return provenance for the exact held/current IMAGE being painted."""

        self._require_owner()
        hold = self._selector_hold
        if hold is not None:
            if hold.image_payload is None:
                return None
            return ImageInteractionOrigin(
                hold.panel_id,
                hold.board_id,
                hold.layout_generation,
                hold.sequence,
                hold.source_identity,
                hold.presentation,
                hold.image_payload.evaluated_input,
            )
        front = self._front
        panel_id = self._selector_panel_id
        if front is None or panel_id is None or panel_id not in self._panel_ids:
            return None
        panel = front[0].panels[self._panel_ids.index(panel_id)]
        payload = panel.image_payload
        if payload is None:
            return None
        return ImageInteractionOrigin(
            panel.panel_id,
            front[0].board_id,
            front[0].layout_generation,
            front[0].sequence,
            panel.source_identity,
            _panel_presentation(panel),
            payload.evaluated_input,
        )

    def discard_pending_image_interaction(
        self,
        origin: ImageInteractionOrigin,
    ) -> bool:
        """Release only one exact failed display intent.

        The owner calls this after an asynchronously accepted reconfigure ends
        in a terminal render fault.  A delayed failure cannot clear a newer
        pending command because sequence, source, presentation revision, and
        exact evaluated input all participate in ``origin`` equality.
        """

        self._require_owner()
        if not isinstance(origin, ImageInteractionOrigin):
            raise TypeError("origin must be ImageInteractionOrigin")
        if not self._interaction_is_pending() or origin != self._pending_origin:
            return False
        self._pending_viewport = None
        self._pending_color_limits = None
        self._pending_origin = None
        self.update()
        return True

    def selection_for_rectangle_gesture(self, gesture: RectangleGesture) -> Selection:
        """Resolve a gesture only while its exact display-only origin is held."""

        self._require_owner()
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("gesture must be RectangleGesture")
        hold = self._selector_hold
        if (
            hold is None
            or self._selector_drag_anchor is not None
            or self._pan_anchor is not None
        ):
            raise RuntimeError("rectangle gesture has no completed held origin")
        if gesture.panel_id != hold.panel_id or (
            gesture.board_id,
            gesture.layout_generation,
            gesture.sequence,
            gesture.source_identity,
            gesture.viewport_revision,
        ) != hold.gesture_identity:
            raise RuntimeError("rectangle gesture differs from its held panel origin")
        viewport = self._require_selector_viewport()
        if viewport.viewport_revision != hold.viewport_revision:
            raise RuntimeError("rectangle gesture viewport changed before dispatch")
        front = self._front
        if front is None or not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            raise RuntimeError("rectangle gesture origin is stale for this panel binding")
        return viewport.selection_for_normalized_bounds(gesture.normalized_bounds)

    def bind_rectangle_selector(
        self,
        panel_id: str,
        viewport: ImageViewportTransform,
        callback: Callable[[RectangleGesture], object],
        *,
        enabled: bool = True,
        interaction_callback: Callable[[ImageInteractionCommit], object] | None = None,
    ) -> None:
        """Bind one image panel without giving the widget a runtime control sink."""

        self._require_owner()
        panel_id = canonical_text(panel_id, "selector panel_id")
        if panel_id not in self._panel_ids:
            raise ValueError("selector panel_id is absent from this board")
        if not isinstance(viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        if not callable(callback):
            raise TypeError("selector callback must be callable")
        if interaction_callback is not None and not callable(interaction_callback):
            raise TypeError("interaction_callback must be callable or None")
        if not isinstance(enabled, bool):
            raise TypeError("selector enabled must be bool")
        if self._front is not None:
            self._validate_selector_binding(
                panel_id,
                viewport,
                self._front[0],
            )
            if interaction_callback is not None:
                index = self._panel_ids.index(panel_id)
                if self._front[0].panels[index].image_payload is None:
                    raise ValueError(
                        "image interaction callback requires exact ImagePanelPayload"
                    )
        self._reset_rectangle_selector()
        self._selector_fault = None
        self._selector_panel_id = panel_id
        self._selector_viewport = viewport
        self._selector_callback = callback
        self._interaction_callback = interaction_callback
        self._pending_viewport = None
        self._pending_color_limits = None
        self._pending_origin = None
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
            self._set_hover_sample(None)
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
        hold = self._selector_hold
        if hold is not None and not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            hold = None
        held_target = None
        for index, (_pixels, latest_image) in enumerate(images):
            panel_id = self._panel_ids[index]
            panel = front[0].panels[index]
            image = (
                hold.prepared[1]
                if hold is not None and hold.panel_id == panel_id
                else latest_image
            )
            payload = (
                hold.image_payload
                if hold is not None and hold.panel_id == panel_id
                else panel.image_payload
            )
            bounds = _panel_bounds(
                self.rect(),
                index=index,
                count=len(images),
                columns=self._columns,
            )
            target, source, rail = _panel_image_geometry(
                bounds,
                image,
                payload,
            )
            painter.drawImage(QtCore.QRectF(target), image, source)
            if payload is not None and rail is not None:
                self._paint_color_rail(painter, payload, rail)
            if hold is not None and hold.panel_id == panel_id:
                held_target = target
        if hold is not None and held_target is not None:
            self._paint_hold_badge(
                painter,
                hold,
                held_target,
                live_sequence=front[0].sequence,
            )
        self._paint_selector_overlays(painter)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._selector_enabled:
            super().mousePressEvent(event)
            return
        target = self._selector_target()
        rail_target = self._clim_rail_target()
        hits_image = target is not None and target[0].contains(event.pos())
        hits_rail = rail_target is not None and rail_target[0].contains(event.pos())
        if self._interaction_is_pending() or self._selector_hold is not None:
            if hits_image or hits_rail:
                event.accept()
            else:
                super().mousePressEvent(event)
            return
        if (
            target is not None
            and rail_target is not None
            and event.button() == QtCore.Qt.LeftButton
            and self._interaction_callback is not None
            and rail_target[0].contains(event.pos())
        ):
            handle = self._clim_handle_at(event.pos(), rail_target[0], rail_target[4])
            if handle is not None:
                self._selector_hold = self._held_panel_from_target(target)
                self._clim_drag = handle
                self._clim_origin_limits = rail_target[4].color_limits
                self._clim_candidate = rail_target[4].color_limits
                self._clim_domain = self._color_rail_domain(rail_target[4])
                self._set_hover_sample(None)
                self.update()
                event.accept()
                return
        if not hits_image:
            super().mousePressEvent(event)
            return
        if event.button() == QtCore.Qt.RightButton:
            sample = self._sample_for_target(target, event.localPos())
            if sample is None:
                super().mousePressEvent(event)
                return
            self._set_cross_sample(sample)
            self._set_hover_sample(None)
            self.update()
            event.accept()
            return
        if event.button() == QtCore.Qt.MiddleButton:
            if self._interaction_callback is None:
                super().mousePressEvent(event)
                return
            self._selector_hold = self._held_panel_from_target(target)
            self._pan_anchor = QtCore.QPointF(event.localPos())
            self._pan_origin = self._viewport_for_target(target)
            self._pan_target_size = (
                max(1, target[0].width()),
                max(1, target[0].height()),
            )
            self._pan_candidate = self._pan_origin
            self._set_hover_sample(None)
            self.update()
            event.accept()
            return
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        image_target = target[0]
        point = self._normalized_point(event.localPos(), image_target, clamp=False)
        bounds = self._selector_draft_bounds or self._selector_applied_bounds
        handle = None if bounds is None else self._hit_corner_handle(event.pos(), bounds, image_target)
        self._selector_drag_anchor = (
            point if handle is None else self._opposite_corner_anchor(bounds, handle)
        )
        self._selector_hold = self._held_panel_from_target(target)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._clim_drag is not None:
            rail_target = self._clim_rail_target()
            if (
                rail_target is not None
                and self._clim_origin_limits is not None
                and self._clim_domain is not None
            ):
                value = self._rail_value(
                    float(event.localPos().y()),
                    self._clim_domain,
                    rail_target[0],
                )
                low, high = self._clim_origin_limits
                if self._clim_drag == "low":
                    low = min(value, math.nextafter(high, -math.inf))
                else:
                    high = max(value, math.nextafter(low, math.inf))
                self._clim_candidate = (low, high)
                self.update()
            event.accept()
            return
        anchor = self._selector_drag_anchor
        if anchor is not None:
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
            return

        pan_anchor = self._pan_anchor
        pan_origin = self._pan_origin
        pan_size = self._pan_target_size
        if pan_anchor is not None and pan_origin is not None and pan_size is not None:
            delta = (
                float(event.localPos().x() - pan_anchor.x()),
                float(event.localPos().y() - pan_anchor.y()),
            )
            self._pan_candidate = pan_origin.panned_by_pixels(delta, pan_size)
            event.accept()
            return

        if self._selector_enabled and not self._interaction_is_pending():
            target = self._selector_target()
            sample = (
                None
                if target is None or not target[0].contains(event.pos())
                else self._sample_for_target(target, event.localPos())
            )
            self._hover_position = (
                None if sample is None else QtCore.QPointF(event.localPos())
            )
            self._set_hover_sample(sample)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._clim_drag is not None and event.button() == QtCore.Qt.LeftButton:
            candidate = self._clim_candidate
            hold = self._selector_hold
            try:
                if candidate is not None and hold is not None:
                    self._commit_color_limits(candidate, hold=hold)
            finally:
                self._cancel_rectangle_gesture(clear_draft=False)
                self.update()
            event.accept()
            return
        if self._pan_anchor is not None and event.button() == QtCore.Qt.MiddleButton:
            candidate = self._pan_candidate
            hold = self._selector_hold
            try:
                if candidate is not None and hold is not None:
                    self._commit_viewport(candidate, hold=hold)
            finally:
                self._cancel_rectangle_gesture(clear_draft=False)
                self.update()
            event.accept()
            return
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
        hold = self._selector_hold
        bounds = self._selector_draft_bounds
        callback = self._selector_callback
        delivered = False
        # Geometry is complete before the consumer callback runs, but the
        # synchronous held origin remains alive until that callback returns.
        # A re-entrant PREPARING/disable transition must therefore preserve
        # this completed draft rather than classify it as a partial drag.
        self._selector_drag_anchor = None
        try:
            if bounds is not None and hold is not None and callback is not None:
                gesture = RectangleGesture(
                    panel_id=hold.panel_id,
                    board_id=hold.board_id,
                    layout_generation=hold.layout_generation,
                    sequence=hold.sequence,
                    source_identity=hold.source_identity,
                    normalized_bounds=bounds,
                    viewport_revision=hold.viewport_revision,
                )
                callback(gesture)
                delivered = True
        except BaseException as error:
            if self._selector_fault is None:
                self._selector_fault = detached_render_fault(error)
            self._selector_enabled = False
        finally:
            self._cancel_rectangle_gesture(
                clear_draft=(bounds is not None and not delivered)
            )
            self.update()
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self._selector_enabled or self._interaction_callback is None:
            super().wheelEvent(event)
            return
        target = self._selector_target()
        rail_target = self._clim_rail_target()
        position = event.pos()
        hits_image = target is not None and target[0].contains(position)
        hits_rail = rail_target is not None and rail_target[0].contains(position)
        if self._interaction_is_pending() or self._selector_hold is not None:
            if hits_image or hits_rail:
                event.accept()
            else:
                super().wheelEvent(event)
            return
        if not hits_image:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        point = self._normalized_point(event.posF(), target[0], clamp=False)
        # Preserve the established lab convention: wheel DOWN zooms in and
        # wheel UP zooms out.
        scale = 1.0 / 1.1 if delta < 0 else 1.1
        candidate = self._viewport_for_target(target).centered_zoom(point, scale)
        self._commit_viewport(candidate)
        self._set_hover_sample(None)
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._selector_enabled:
            super().mouseDoubleClickEvent(event)
            return
        target = self._selector_target()
        rail_target = self._clim_rail_target()
        hits_image = target is not None and target[0].contains(event.pos())
        hits_rail = rail_target is not None and rail_target[0].contains(event.pos())
        if self._interaction_is_pending() or self._selector_hold is not None:
            if hits_image or hits_rail:
                event.accept()
            else:
                super().mouseDoubleClickEvent(event)
            return
        if not hits_image:
            super().mouseDoubleClickEvent(event)
            return
        if event.button() == QtCore.Qt.RightButton:
            self._set_cross_sample(None)
            self._set_hover_sample(None)
            self.update()
            event.accept()
            return
        if event.button() == QtCore.Qt.MiddleButton and self._interaction_callback is not None:
            viewport = self._viewport_for_target(target)
            area = self._selector_draft_bounds or self._selector_applied_bounds
            candidate = viewport.with_visible_bounds(
                (0.0, 0.0, 1.0, 1.0) if area is None else area
            )
            self._commit_viewport(candidate)
            self._set_hover_sample(None)
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._set_hover_sample(None)
        self._hover_position = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self._selector_hold is not None:
            self._cancel_rectangle_gesture(clear_draft=True)
        self._set_hover_sample(None)
        self._hover_position = None
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
        elif event.type() in (
            QtCore.QEvent.Hide,
            QtCore.QEvent.WindowDeactivate,
            QtCore.QEvent.UngrabMouse,
        ):
            changed = getattr(self, "_selector_hold", None) is not None
            if changed:
                self._cancel_rectangle_gesture(clear_draft=True)
            if getattr(self, "_hover_sample", None) is not None:
                self._set_hover_sample(None)
                changed = True
            if changed:
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
        *,
        panel_ids: tuple[str, ...] | None = None,
    ) -> None:
        configured_ids = self._panel_ids if panel_ids is None else panel_ids
        index = configured_ids.index(panel_id)
        raster = frame.panels[index].raster
        expected_height, expected_width = viewport.raster_shape
        if raster.width != expected_width or raster.height != expected_height:
            raise ValueError(
                "selector viewport axes do not match the selected panel raster geometry"
            )
        if frame.panels[index].panel_id != panel_id:
            raise ValueError("selector panel identity changed")
        payload = frame.panels[index].image_payload
        if payload is not None and payload.viewport != viewport:
            raise ValueError(
                "selector viewport differs from the exact image payload viewport"
            )

    def _viewport_for_presented_panel(
        self,
        panel_id: str,
        current: ImageViewportTransform,
        frame: BoardFrame,
        *,
        panel_ids: tuple[str, ...],
    ) -> ImageViewportTransform:
        index = panel_ids.index(panel_id)
        panel = frame.panels[index]
        payload = panel.image_payload
        if payload is None:
            return current
        candidate = payload.viewport
        previous = self._front
        structurally_new = previous is None or panel_id not in self._panel_ids
        if not structurally_new and previous is not None:
            old_panel = previous[0].panels[self._panel_ids.index(panel_id)]
            structurally_new = self._panel_semantics_changed(old_panel, panel)
        if structurally_new:
            return candidate
        if candidate.axes != current.axes:
            raise ValueError("image viewport axes changed without panel structure change")
        if candidate.viewport_revision < current.viewport_revision:
            raise ValueError("stale image viewport revision cannot replace the visible front")
        pending = self._pending_viewport
        if (
            pending is not None
            and candidate.viewport_revision == pending.viewport_revision
            and candidate != pending
        ):
            raise ValueError("pending image viewport revision returned conflicting bounds")
        if (
            candidate.viewport_revision == current.viewport_revision
            and candidate != current
        ):
            raise ValueError("one image viewport revision describes conflicting bounds")
        return candidate

    def _selector_target(self):
        front = self._front
        panel_id = self._selector_panel_id
        viewport = self._selector_viewport
        if front is None or panel_id is None or viewport is None:
            return None
        index = self._panel_ids.index(panel_id)
        hold = self._selector_hold
        prepared = (
            hold.prepared
            if hold is not None and hold.panel_id == panel_id
            else front[1][index]
        )
        image = prepared[1]
        bounds = _panel_bounds(
            self.rect(),
            index=index,
            count=len(front[1]),
            columns=self._columns,
        )
        payload = (
            hold.image_payload
            if hold is not None and hold.panel_id == panel_id
            else front[0].panels[index].image_payload
        )
        target, _source, _rail = _panel_image_geometry(bounds, image, payload)
        return target, front[0], front[0].panels[index], prepared

    def _clim_rail_target(self):
        front = self._front
        panel_id = self._selector_panel_id
        if front is None or panel_id is None:
            return None
        index = self._panel_ids.index(panel_id)
        hold = self._selector_hold
        prepared = (
            hold.prepared
            if hold is not None and hold.panel_id == panel_id
            else front[1][index]
        )
        panel = front[0].panels[index]
        payload = (
            hold.image_payload
            if hold is not None and hold.panel_id == panel_id
            else panel.image_payload
        )
        if payload is None:
            return None
        bounds = _panel_bounds(
            self.rect(),
            index=index,
            count=len(front[1]),
            columns=self._columns,
        )
        _target, _source, rail = _panel_image_geometry(bounds, prepared[1], payload)
        if rail is None:
            return None
        return rail, front[0], panel, prepared, payload

    def _viewport_for_target(self, target) -> ImageViewportTransform:
        hold = self._selector_hold
        if hold is not None and hold.panel_id == target[2].panel_id:
            payload = hold.image_payload
            if payload is not None:
                return payload.viewport
        payload = target[2].image_payload
        if payload is not None:
            return payload.viewport
        return self._require_selector_viewport()

    def _sample_for_target(
        self,
        target,
        point: QtCore.QPointF,
    ) -> _ImageSample | None:
        image_target, frame, panel = target[0], target[1], target[2]
        hold = self._selector_hold
        if hold is not None and hold.panel_id == panel.panel_id:
            payload = hold.image_payload
            presentation = hold.presentation
        else:
            payload = panel.image_payload
            presentation = _panel_presentation(panel)
        if payload is None:
            return None
        viewport = payload.viewport
        if presentation.panel_revision != viewport.viewport_revision:
            return None
        normalized = self._normalized_point(point, image_target, clamp=False)
        y_index, x_index = viewport.sample_indices_for_visible_point(normalized)
        value = payload.image.values[y_index, x_index]
        if hasattr(value, "item"):
            value = value.item()
        valid = payload.image.validity[y_index, x_index]
        if hasattr(valid, "item"):
            valid = valid.item()
        try:
            finite_value = math.isfinite(value)
        except TypeError:
            finite_value = False
        x_coordinate = viewport.x_axis.coordinate_at(x_index)
        y_coordinate = viewport.y_axis.coordinate_at(y_index)
        return _ImageSample(
            x_index=x_index,
            y_index=y_index,
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
            value=value,
            valid=bool(valid) and finite_value,
        )

    def _commit_viewport(
        self,
        candidate: ImageViewportTransform,
        *,
        hold: _HeldPanelFront | None = None,
    ) -> bool:
        current = self._require_selector_viewport()
        if candidate == current:
            return False
        if candidate.axes != current.axes:
            raise ValueError("viewport commit cannot change image axes")
        if candidate.viewport_revision <= current.viewport_revision:
            raise ValueError("viewport commit revision must increase")
        if self._interaction_is_pending():
            return False
        front = self._front
        if front is None:
            return False
        if hold is not None and not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            return False
        callback = self._interaction_callback
        if callback is None:
            return False
        origin = self._interaction_origin(hold=hold)
        command = ImageViewportCommit(origin, candidate)
        self._pending_viewport = candidate
        self._pending_origin = origin
        try:
            callback(command)
        except BaseException as error:
            self._pending_viewport = None
            self._pending_origin = None
            if self._selector_fault is None:
                self._selector_fault = detached_render_fault(error)
            self._selector_enabled = False
            self._set_hover_sample(None)
            return False
        return True

    def _clim_handle_at(
        self,
        point: QtCore.QPoint,
        rail: QtCore.QRect,
        payload: ImagePanelPayload,
    ) -> str | None:
        domain = self._color_rail_domain(payload)
        candidates = (
            (abs(point.y() - self._rail_y(payload.color_limits[0], domain, rail)), "low"),
            (abs(point.y() - self._rail_y(payload.color_limits[1], domain, rail)), "high"),
        )
        distance, handle = min(candidates)
        return handle if distance <= 7.0 else None

    def _commit_color_limits(
        self,
        limits: tuple[float, float],
        *,
        hold: _HeldPanelFront,
    ) -> bool:
        payload = hold.image_payload
        if payload is None or limits == payload.color_limits:
            return False
        if self._interaction_is_pending():
            return False
        front = self._front
        if front is None or not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            return False
        callback = self._interaction_callback
        if callback is None:
            return False
        origin = self._interaction_origin(hold=hold)
        command = ImageColorLimitsCommit(origin, limits)
        self._pending_color_limits = command.color_limits
        self._pending_origin = origin
        try:
            callback(command)
        except BaseException as error:
            self._pending_color_limits = None
            self._pending_origin = None
            if self._selector_fault is None:
                self._selector_fault = detached_render_fault(error)
            self._selector_enabled = False
            self._set_hover_sample(None)
            return False
        return True

    def _interaction_is_pending(self) -> bool:
        return (
            self._pending_viewport is not None
            or self._pending_color_limits is not None
        )

    def _interaction_origin(
        self,
        *,
        hold: _HeldPanelFront | None = None,
    ) -> ImageInteractionOrigin:
        if hold is not None and hold is not self._selector_hold:
            raise RuntimeError("image interaction hold is no longer painted")
        origin = self.visible_image_origin()
        if origin is None:
            raise RuntimeError("image interaction origin has no exact payload")
        return origin

    def _set_cross_sample(self, sample: _ImageSample | None) -> None:
        if sample is self._cross_sample:
            return
        self._cross_sample = sample

    def _set_hover_sample(self, sample: _ImageSample | None) -> None:
        if sample is self._hover_sample:
            return
        self._hover_sample = sample
        if sample is None:
            self._hover_position = None

    def _held_panel_from_target(self, target) -> _HeldPanelFront:
        frame, panel, prepared = target[1], target[2], target[3]
        viewport = self._viewport_for_target(target)
        return _HeldPanelFront(
            panel_id=panel.panel_id,
            board_id=frame.board_id,
            layout_generation=frame.layout_generation,
            sequence=frame.sequence,
            coherence_group=panel.coherence_group,
            source_identity=panel.source_identity,
            presentation=_panel_presentation(panel),
            viewport_revision=viewport.viewport_revision,
            raster_geometry=_raster_geometry(panel),
            prepared=prepared,
            image_payload=panel.image_payload,
        )

    @staticmethod
    def _panel_semantics_changed(old: PanelFrame, new: PanelFrame) -> bool:
        old_presentation = _panel_presentation(old)
        new_presentation = _panel_presentation(new)
        old_axes = None if old.image_payload is None else old.image_payload.viewport.axes
        new_axes = None if new.image_payload is None else new.image_payload.viewport.axes
        return (
            old.panel_id != new.panel_id
            or old.coherence_group != new.coherence_group
            or old.source_identity != new.source_identity
            or old_presentation.panel_id != new_presentation.panel_id
            or old_presentation.document_id != new_presentation.document_id
            or old_presentation.document_revision != new_presentation.document_revision
            or old_presentation.selection_revision
            != new_presentation.selection_revision
            or _raster_geometry(old) != _raster_geometry(new)
            or old_axes != new_axes
        )

    def _hold_matches_frame(
        self,
        hold: _HeldPanelFront,
        frame: BoardFrame,
        *,
        panel_ids: tuple[str, ...],
    ) -> bool:
        if (
            frame.board_id != hold.board_id
            or frame.layout_generation != hold.layout_generation
            or hold.panel_id not in panel_ids
        ):
            return False
        viewport = self._selector_viewport
        if viewport is None or viewport.viewport_revision != hold.viewport_revision:
            return False
        index = panel_ids.index(hold.panel_id)
        panel = frame.panels[index]
        return (
            panel.panel_id == hold.panel_id
            and panel.coherence_group == hold.coherence_group
            and panel.source_identity == hold.source_identity
            and _panel_presentation(panel) == hold.presentation
            and _raster_geometry(panel) == hold.raster_geometry
        )

    @staticmethod
    def _paint_hold_badge(
        painter: QtGui.QPainter,
        hold: _HeldPanelFront,
        target: QtCore.QRect,
        *,
        live_sequence: int,
    ) -> None:
        painter.save()
        try:
            painter.setClipRect(target)
            label = f"H {hold.sequence}→{live_sequence}"
            metrics = painter.fontMetrics()
            label_bounds = metrics.boundingRect(label).adjusted(-6, -3, 6, 3)
            label_bounds.moveBottomRight(target.bottomRight() + QtCore.QPoint(-6, -6))
            painter.fillRect(label_bounds, QtGui.QColor(0, 0, 0, 190))
            painter.setPen(QtGui.QColor(ORANGE))
            painter.drawText(label_bounds, QtCore.Qt.AlignCenter, label)
        finally:
            painter.restore()

    @staticmethod
    def _color_rail_domain(payload: ImagePanelPayload) -> tuple[float, float]:
        low, high = payload.color_limits
        span = high - low
        padding = max(
            span * 0.08,
            math.ulp(max(1.0, abs(low), abs(high))) * 16.0,
        )
        return low - padding, high + padding

    @staticmethod
    def _rail_y(value: float, domain: tuple[float, float], rail: QtCore.QRect) -> float:
        low, high = domain
        fraction = (value - low) / (high - low)
        return rail.bottom() - min(1.0, max(0.0, fraction)) * rail.height()

    @staticmethod
    def _rail_value(y: float, domain: tuple[float, float], rail: QtCore.QRect) -> float:
        fraction = (rail.bottom() - y) / max(1, rail.height())
        low, high = domain
        return low + min(1.0, max(0.0, fraction)) * (high - low)

    def _paint_color_rail(
        self,
        painter: QtGui.QPainter,
        payload: ImagePanelPayload,
        rail: QtCore.QRect,
    ) -> None:
        painter.save()
        try:
            painter.setClipRect(rail)
            painter.fillRect(rail, QtGui.QColor(12, 12, 12, 230))
            gradient_left = rail.right() - min(9, max(5, rail.width() // 3)) + 1
            denominator = max(1, rail.height() - 1)
            domain = self._color_rail_domain(payload)
            domain_low, domain_high = domain
            for offset in range(rail.height()):
                fraction = 1.0 - offset / denominator
                value = domain_low + fraction * (domain_high - domain_low)
                painter.setPen(
                    QtGui.QColor.fromRgba(
                        self._color_rail_argb(payload, value)
                    )
                )
                y = rail.top() + offset
                painter.drawLine(gradient_left, y, rail.right(), y)

            if any(payload.histogram_counts):
                maximum = max(payload.histogram_counts)
                histogram_width = max(1, gradient_left - rail.left() - 2)
                value_low, value_high = payload.color_limits
                for code, count in enumerate(payload.histogram_counts, start=1):
                    if count == 0:
                        continue
                    value = (
                        value_low
                        if value_high == value_low
                        else value_low + (value_high - value_low) * (code - 1) / 254.0
                    )
                    y = int(round(self._rail_y(value, domain, rail)))
                    width = max(1, int(round(histogram_width * count / maximum)))
                    painter.fillRect(
                        QtCore.QRect(gradient_left - width - 1, y, width, 1),
                        QtGui.QColor(210, 210, 210, 150),
                    )

            if payload.data_range is not None:
                guide_pen = QtGui.QPen(QtGui.QColor(180, 180, 180, 150), 1.0)
                guide_pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(guide_pen)
                for value in payload.data_range:
                    if not payload.color_limits[0] <= value <= payload.color_limits[1]:
                        continue
                    y = self._rail_y(value, domain, rail)
                    painter.drawLine(
                        QtCore.QPointF(rail.left(), y),
                        QtCore.QPointF(rail.right(), y),
                    )

            limits = (
                self._clim_candidate
                if self._selector_hold is not None
                and self._selector_hold.image_payload is payload
                and self._clim_candidate is not None
                else payload.color_limits
            )
            for value in limits:
                y = self._rail_y(value, domain, rail)
                painter.setPen(QtGui.QPen(QtGui.QColor(ORANGE), 2.0))
                painter.drawLine(
                    QtCore.QPointF(rail.left(), y),
                    QtCore.QPointF(rail.right(), y),
                )
        finally:
            painter.restore()

    @staticmethod
    def _color_rail_argb(payload: ImagePanelPayload, value: float) -> int:
        """Map one physical rail value through the painted image's clim."""

        index = indexed8_code_for_value(value, payload.color_limits)
        return payload.base_palette[index]

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
        viewport = self._require_selector_viewport()
        left, top, right, bottom = viewport.visible_bounds_for_full_bounds(bounds)
        visible_left, visible_top, visible_right, visible_bottom = (
            viewport.visible_bounds
        )
        x_half_cell = 0.5 / (
            viewport.x_axis.size * (visible_right - visible_left)
        )
        y_half_cell = 0.5 / (
            viewport.y_axis.size * (visible_bottom - visible_top)
        )
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
        try:
            visible_bounds = self._require_selector_viewport().visible_bounds_for_full_bounds(
                bounds
            )
        except ValueError:
            return None
        rectangle = self._overlay_rect(visible_bounds, target)
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
            visible = self._require_selector_viewport().clipped_visible_bounds_for_full_bounds(
                self._selector_applied_bounds
            )
            if visible is not None:
                self._paint_selector_rectangle(
                    painter,
                    visible,
                    image_target,
                    QtGui.QColor(GREEN),
                    dashed=False,
                    handles=(
                        self._selector_enabled
                        and self._selector_draft_bounds is None
                        and self._rectangle_fully_visible(
                            self._selector_applied_bounds
                        )
                    ),
                    endpoint_bounds=(
                        self._selector_applied_bounds
                        if self._selector_draft_bounds is None
                        else None
                    ),
                )
        if self._selector_draft_bounds is not None:
            visible = self._require_selector_viewport().clipped_visible_bounds_for_full_bounds(
                self._selector_draft_bounds
            )
            if visible is not None:
                self._paint_selector_rectangle(
                    painter,
                    visible,
                    image_target,
                    QtGui.QColor(ORANGE),
                    dashed=True,
                    handles=(
                        self._selector_enabled
                        and self._rectangle_fully_visible(
                            self._selector_draft_bounds
                        )
                    ),
                    endpoint_bounds=self._selector_draft_bounds,
                )
        hold = self._selector_hold
        if (
            hold is not None
            and hold.image_payload is not None
            and self._clim_candidate is not None
        ):
            self._paint_clim_candidate_label(
                painter,
                hold.image_payload,
                image_target,
            )
        if self._cross_sample is not None:
            self._paint_cross_sample(
                painter,
                self._cross_sample,
                image_target,
            )
        if self._hover_sample is not None and self._hover_position is not None:
            self._paint_hover_sample(
                painter,
                self._hover_sample,
                self._hover_position,
                image_target,
            )
        painter.restore()

    def _rectangle_fully_visible(self, bounds: NormalizedRectangle) -> bool:
        try:
            self._require_selector_viewport().visible_bounds_for_full_bounds(bounds)
        except ValueError:
            return False
        return True

    def _visible_point_for_sample(
        self,
        sample: _ImageSample,
    ) -> tuple[float, float] | None:
        viewport = self._require_selector_viewport()
        if (
            sample.x_index >= viewport.x_axis.size
            or sample.y_index >= viewport.y_axis.size
        ):
            return None
        full = (
            (sample.x_index + 0.5) / viewport.x_axis.size,
            (sample.y_index + 0.5) / viewport.y_axis.size,
        )
        try:
            return viewport.visible_point_for_full_point(full)
        except ValueError:
            return None

    def _paint_cross_sample(
        self,
        painter: QtGui.QPainter,
        sample: _ImageSample,
        target: QtCore.QRect,
    ) -> None:
        point = self._visible_point_for_sample(sample)
        color = QtGui.QColor(GREEN)
        if point is not None:
            x = target.x() + point[0] * target.width()
            y = target.y() + point[1] * target.height()
            painter.setPen(QtGui.QPen(color, 1.5))
            painter.drawLine(
                QtCore.QPointF(x, target.top()),
                QtCore.QPointF(x, target.bottom()),
            )
            painter.drawLine(
                QtCore.QPointF(target.left(), y),
                QtCore.QPointF(target.right(), y),
            )
            painter.setBrush(QtGui.QBrush(color))
            painter.drawEllipse(QtCore.QPointF(x, y), 3.5, 3.5)
        value = self._formatted_sample_value(sample)
        suffix = " · off-view" if point is None else ""
        label = f"({sample.x_coordinate}, {sample.y_coordinate}, {value}){suffix}"
        metrics = painter.fontMetrics()
        bounds = metrics.boundingRect(label).adjusted(-5, -2, 5, 2)
        bounds.moveTopRight(target.topRight() + QtCore.QPoint(-5, 5))
        painter.fillRect(bounds, QtGui.QColor(0, 0, 0, 190))
        painter.setPen(color)
        painter.drawText(bounds, QtCore.Qt.AlignCenter, label)

    def _paint_hover_sample(
        self,
        painter: QtGui.QPainter,
        sample: _ImageSample,
        position: QtCore.QPointF,
        target: QtCore.QRect,
    ) -> None:
        label = (
            f"x={sample.x_coordinate}  y={sample.y_coordinate}  "
            f"z={self._formatted_sample_value(sample)}"
        )
        metrics = painter.fontMetrics()
        bounds = metrics.boundingRect(label).adjusted(-5, -2, 5, 2)
        x = min(target.right() - bounds.width(), int(position.x()) + 12)
        y = min(target.bottom() - bounds.height(), int(position.y()) + 12)
        bounds.moveTopLeft(QtCore.QPoint(max(target.left(), x), max(target.top(), y)))
        painter.fillRect(bounds, QtGui.QColor(0, 0, 0, 190))
        painter.setPen(QtGui.QColor(ORANGE))
        painter.drawText(bounds, QtCore.Qt.AlignCenter, label)

    def _paint_clim_candidate_label(
        self,
        painter: QtGui.QPainter,
        payload: ImagePanelPayload,
        target: QtCore.QRect,
    ) -> None:
        label = self._clim_candidate_label(payload)
        metrics = painter.fontMetrics()
        bounds = metrics.boundingRect(label).adjusted(-6, -3, 6, 3)
        bounds.moveBottomLeft(target.bottomLeft() + QtCore.QPoint(7, -7))
        painter.fillRect(bounds, QtGui.QColor(0, 0, 0, 190))
        painter.setPen(QtGui.QColor(ORANGE))
        painter.drawText(bounds, QtCore.Qt.AlignCenter, label)

    def _clim_candidate_label(self, payload: ImagePanelPayload) -> str:
        limits = self._clim_candidate
        if limits is None:
            raise RuntimeError("H candidate label requires an active limit draft")
        low, high = self._color_rail_domain(payload)
        span = high - low
        gap = span / 1000.0 if span else 0.01
        precision = max(0, -int(math.ceil(math.log10(gap))))

        def formatted(value: float) -> str:
            return (
                f"{value:.{precision}f}"
                if precision <= 6 and abs(value) < 1.0e9
                else f"{value:.6g}"
            )

        return f"H low={formatted(limits[0])}  high={formatted(limits[1])}"

    @staticmethod
    def _formatted_sample_value(sample: _ImageSample) -> str:
        if not sample.valid:
            return "invalid"
        value = sample.value
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _paint_selector_rectangle(
        self,
        painter: QtGui.QPainter,
        bounds: NormalizedRectangle,
        target: QtCore.QRect,
        color: QtGui.QColor,
        *,
        dashed: bool,
        handles: bool,
        endpoint_bounds: NormalizedRectangle | None,
    ) -> None:
        rectangle = self._overlay_rect(bounds, target)
        pen = QtGui.QPen(color, 2.0)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(rectangle)
        if endpoint_bounds is not None:
            label = self._selection_endpoint_label(endpoint_bounds)
            flags = QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
            label_area = target.adjusted(7, 7, -7, -7)
            label_bounds = painter.fontMetrics().boundingRect(
                label_area,
                flags,
                label,
            ).adjusted(-5, -3, 5, 3)
            painter.fillRect(label_bounds, QtGui.QColor(0, 0, 0, 190))
            painter.setPen(color)
            painter.drawText(label_bounds.adjusted(5, 3, -5, -3), flags, label)
        if handles:
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

    def _selection_endpoint_label(self, bounds: NormalizedRectangle) -> str:
        viewport = self._require_selector_viewport()
        selected_terms = {
            term.axis_id: term
            for term in viewport.selection_for_normalized_bounds(bounds).terms
        }
        visible_cell_bounds = viewport.snapped_bounds_for_drag(
            (0.0, 0.0),
            (1.0, 1.0),
        )
        visible_terms = {
            term.axis_id: term
            for term in viewport.selection_for_normalized_bounds(
                visible_cell_bounds
            ).terms
        }
        selected_x = selected_terms[viewport.x_axis.axis_id]
        selected_y = selected_terms[viewport.y_axis.axis_id]
        visible_x = visible_terms[viewport.x_axis.axis_id]
        visible_y = visible_terms[viewport.y_axis.axis_id]

        def precision(span: float) -> int:
            gap = abs(float(span)) / 1000.0 if span else 0.01
            return max(0, -int(math.ceil(math.log10(gap))))

        x_precision = precision(float(visible_x.upper) - float(visible_x.lower))
        y_precision = precision(float(visible_y.upper) - float(visible_y.lower))
        return (
            f"({float(selected_x.lower):.{x_precision}f}, "
            f"{float(selected_y.lower):.{y_precision}f})\n"
            f"({float(selected_x.upper):.{x_precision}f}, "
            f"{float(selected_y.upper):.{y_precision}f})"
        )

    def _cancel_rectangle_gesture(self, *, clear_draft: bool) -> None:
        self._selector_drag_anchor = None
        self._pan_anchor = None
        self._pan_origin = None
        self._pan_target_size = None
        self._pan_candidate = None
        self._clim_drag = None
        self._clim_origin_limits = None
        self._clim_candidate = None
        self._clim_domain = None
        self._selector_hold = None
        if clear_draft:
            self._selector_draft_bounds = None

    def _reset_rectangle_selector(self) -> None:
        self._cancel_rectangle_gesture(clear_draft=True)
        self._selector_applied_bounds = None
        self._pending_viewport = None
        self._pending_color_limits = None
        self._pending_origin = None
        self._set_cross_sample(None)
        self._set_hover_sample(None)
        self._selector_enabled = False
        self._selector_panel_id = None
        self._selector_viewport = None
        self._selector_callback = None
        self._interaction_callback = None

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtRasterBoard presentation is GUI-thread affine")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("QtRasterBoard is closed")


__all__ = ["QtImageBoard", "QtOwnerWake", "QtRasterBoard"]
