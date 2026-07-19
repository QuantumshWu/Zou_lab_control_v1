"""Qt widgets for one immutable live-image board front."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Literal, TypeAlias

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data import Selection
from zlc_storage import canonical_text, nonnegative_integer

from ..curve_display import CurveViewportTransform
from ..display_range import RelimMode, validated_display_range
from ..histogram_display import HistogramViewportTransform
from ..image_raster import indexed8_code_for_value
from ..image_view import validate_normalized_rectangle
from ..render import (
    BoardFrame,
    CurvePanelPayload,
    DisplayPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    PixelFormat,
    SiteMapPanelPayload,
    SourceIdentity,
    detached_render_fault,
)
from ..site_map import (
    SITE_EMPTY_ALPHA,
    SITE_EMPTY_COLOR,
    SITE_EMPTY_LINEWIDTH,
    SITE_INVALID_ALPHA,
    SITE_INVALID_COLOR,
    SITE_INVALID_LINEWIDTH,
    SITE_OCCUPIED_ALPHA,
    SITE_OCCUPIED_COLOR,
    SITE_OCCUPIED_LINEWIDTH,
)
from ..selector import (
    CurveInteractionIntent,
    CurveRangeGesture,
    CurveViewportCommit,
    HistogramInteractionIntent,
    HistogramRangeGesture,
    HistogramViewportCommit,
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageViewportTransform,
    ImageViewportCommit,
    NormalizedRectangle,
    PanelInteractionOrigin,
    RectangleGesture,
)
from .style import (
    BG,
    GREEN,
    ORANGE,
)


def _prepared_qimage(panel_or_raster) -> tuple[bytes, QtGui.QImage]:
    """Prepare one owned Qt front, applying an IMAGE payload's exact LUT.

    Accepting a bare ``RasterBuffer`` preserves the small legacy presenter
    surface.  Interactive IMAGE panels pass ``PanelFrame`` so INDEXED8 colour
    limits are resolved from its paired immutable payload rather than guessed
    from display codes.
    """

    if isinstance(panel_or_raster, PanelFrame):
        raster = panel_or_raster.raster
        payload = panel_or_raster.display_payload
        if isinstance(payload, SiteMapPanelPayload):
            payload = payload.background
        elif not isinstance(payload, ImagePanelPayload):
            payload = None
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
    *,
    aspect_ratio: float | None = None,
    anchor_west: bool = False,
) -> QtCore.QRect:
    if source.width() <= 0.0 or source.height() <= 0.0:
        raise ValueError("image source viewport must have positive geometry")
    source_ratio = (
        source.width() / source.height()
        if aspect_ratio is None
        else float(aspect_ratio)
    )
    if not math.isfinite(source_ratio) or source_ratio <= 0.0:
        raise ValueError("image aspect ratio must be finite and positive")
    bounds_ratio = bounds.width() / max(1, bounds.height())
    if bounds_ratio > source_ratio:
        height = bounds.height()
        width = max(1, int(round(height * source_ratio)))
    else:
        width = bounds.width()
        height = max(1, int(round(width / source_ratio)))
    return QtCore.QRect(
        bounds.x()
        if anchor_west
        else bounds.x() + (bounds.width() - width) // 2,
        bounds.y() + (bounds.height() - height) // 2,
        width,
        height,
    )


def _panel_image_geometry(
    bounds: QtCore.QRect,
    image: QtGui.QImage,
    payload: ImagePanelPayload | None,
    *,
    site_map_payload: SiteMapPanelPayload | None = None,
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
    target = _aspect_target_for_source(
        image_bounds,
        source,
        aspect_ratio=(
            None
            if site_map_payload is None
            else site_map_payload.visible_coordinate_aspect_ratio
        ),
        anchor_west=site_map_payload is not None,
    )
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


def _image_payload(
    panel_or_hold: PanelFrame | _HeldPanelFront,
) -> ImagePanelPayload | None:
    payload = panel_or_hold.display_payload
    if isinstance(payload, SiteMapPanelPayload):
        return payload.background
    return payload if isinstance(payload, ImagePanelPayload) else None


def _site_map_payload(
    panel_or_hold: PanelFrame | _HeldPanelFront,
) -> SiteMapPanelPayload | None:
    payload = panel_or_hold.display_payload
    return payload if isinstance(payload, SiteMapPanelPayload) else None


def _payload_input(
    payload: (
        ImagePanelPayload
        | CurvePanelPayload
        | HistogramPanelPayload
        | SiteMapPanelPayload
    ),
):
    return (
        payload.occupancy_input
        if isinstance(payload, SiteMapPanelPayload)
        else payload.evaluated_input
    )


def _input_structure(evaluated_input) -> tuple[object, object, object, str]:
    """Return producer structure while deliberately excluding normal revisions."""

    ref = evaluated_input.ref
    return (
        evaluated_input.dataset_id,
        ref.block_id,
        ref.stream_generation,
        ref.schema_fingerprint,
    )


def _curve_payload(
    panel_or_hold: PanelFrame | _HeldPanelFront,
) -> CurvePanelPayload | None:
    payload = panel_or_hold.display_payload
    return payload if isinstance(payload, CurvePanelPayload) else None


def _histogram_payload(
    panel_or_hold: PanelFrame | _HeldPanelFront,
) -> HistogramPanelPayload | None:
    payload = panel_or_hold.display_payload
    return payload if isinstance(payload, HistogramPanelPayload) else None


_NumericKind: TypeAlias = Literal["curve", "histogram"]
_NumericPayload: TypeAlias = CurvePanelPayload | HistogramPanelPayload
_NumericViewport: TypeAlias = CurveViewportTransform | HistogramViewportTransform
_NumericIntent: TypeAlias = CurveInteractionIntent | HistogramInteractionIntent


def _numeric_payload(
    panel_or_hold: PanelFrame | _HeldPanelFront,
    kind: _NumericKind,
) -> _NumericPayload | None:
    if kind == "curve":
        return _curve_payload(panel_or_hold)
    return _histogram_payload(panel_or_hold)


def _numeric_plot_geometry(
    panel_bounds: QtCore.QRect,
    viewport: _NumericViewport,
) -> QtCore.QRectF:
    """Map the worker's exact top-origin Agg axes bbox into this Qt cell."""

    left, top, right, bottom = viewport.plot_bounds
    return QtCore.QRectF(
        panel_bounds.x() + left * panel_bounds.width(),
        panel_bounds.y() + top * panel_bounds.height(),
        (right - left) * panel_bounds.width(),
        (bottom - top) * panel_bounds.height(),
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
class _NumericCross:
    """One arbitrary continuous numeric cursor, never a snapped sample."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _CurveSample:
    """Nearest valid sample borrowed from one exact immutable curve payload."""

    series_label: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _HistogramBinSample:
    """One bin borrowed from a frozen HistogramPanelPayload projection."""

    series_label: str
    left: float
    right: float
    count: int
    right_closed: bool

    @property
    def x(self) -> float:
        return 0.5 * (self.left + self.right)

    @property
    def y(self) -> float:
        return float(self.count)


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
    raster_geometry: tuple[int, int, int, PixelFormat]
    prepared: tuple[bytes, QtGui.QImage]
    display_payload: (
        ImagePanelPayload
        | CurvePanelPayload
        | HistogramPanelPayload
        | SiteMapPanelPayload
        | None
    )

    @property
    def gesture_identity(self) -> tuple[str, int, int, SourceIdentity]:
        return (
            self.board_id,
            self.layout_generation,
            self.sequence,
            self.source_identity,
        )


@dataclass(slots=True)
class _NumericPanelBinding:
    """The sole mutable owner for one bound numeric panel."""

    kind: _NumericKind
    panel_id: str
    callback: Callable[[_NumericIntent], object]
    viewport: _NumericViewport | None = None
    binding_enabled: bool = True
    interaction_ready: bool = False
    pending_viewport: _NumericViewport | None = None
    pending_origin: PanelInteractionOrigin | None = None
    applied_span: tuple[float, float] | None = None
    span_anchor: float | None = None
    span_candidate: tuple[float, float] | None = None
    pan_anchor: float | None = None
    pan_origin: _NumericViewport | None = None
    pan_candidate: tuple[float, float] | None = None
    cross: _NumericCross | None = None
    hover: _CurveSample | _HistogramBinSample | None = None
    hover_position: QtCore.QPointF | None = None
    fault: RuntimeError | None = None


@dataclass(frozen=True, slots=True)
class _NumericTarget:
    plot: QtCore.QRectF
    frame: BoardFrame
    panel: PanelFrame
    prepared: tuple[bytes, QtGui.QImage]
    payload: _NumericPayload
    bounds: QtCore.QRect
    binding: _NumericPanelBinding


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
            else _image_payload(self._front_frame.panels[0])
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
                else _image_payload(self._front_frame.panels[0])
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
        self._image_binding_enabled = False
        self._image_interaction_ready = False
        self._selector_applied_bounds: NormalizedRectangle | None = None
        self._selector_draft_bounds: NormalizedRectangle | None = None
        self._selector_drag_anchor: tuple[float, float] | None = None
        self._pan_anchor: QtCore.QPointF | None = None
        self._pan_origin: ImageViewportTransform | None = None
        self._pan_target_size: tuple[int, int] | None = None
        self._pan_candidate: ImageViewportTransform | None = None
        self._pending_viewport: ImageViewportTransform | None = None
        self._pending_color_limits: tuple[float, float] | None = None
        self._pending_origin: PanelInteractionOrigin | None = None
        self._clim_drag: str | None = None
        self._clim_origin_limits: tuple[float, float] | None = None
        self._clim_candidate: tuple[float, float] | None = None
        self._clim_domain: tuple[float, float] | None = None
        self._selector_hold: _HeldPanelFront | None = None
        self._cross_sample: _ImageSample | None = None
        self._hover_sample: _ImageSample | None = None
        self._hover_position: QtCore.QPointF | None = None
        self._selector_fault: RuntimeError | None = None
        self._numeric_bindings: dict[str, _NumericPanelBinding] = {}
        self._closed = False
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
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
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
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
        target_numeric_viewports: dict[str, _NumericViewport] = {}
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
                    and _image_payload(target_panel) is None
                ):
                    raise ValueError(
                        "image interaction callback requires exact ImagePanelPayload"
                    )
                if (
                    pending_origin is not None
                    and pending_limits is not None
                    and _image_payload(target_panel) is not None
                    and target_viewport.viewport_revision
                    == pending_origin.presentation.panel_revision + 1
                    and _image_payload(target_panel).color_limits != pending_limits
                ):
                    raise ValueError(
                        "pending image color-limit revision returned conflicting limits"
                    )
            for panel_id, binding in self._numeric_bindings.items():
                if panel_id in target_panel_ids:
                    target_numeric_viewports[panel_id] = (
                        self._numeric_viewport_for_presented_panel(
                            binding,
                            frame,
                            panel_ids=target_panel_ids,
                        )
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
                self._cancel_active_gesture(
                    clear_image_draft=True,
                    clear_numeric_spans=True,
                )
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
                    cancel_interaction = (
                        interaction_was_active
                        and self._selector_hold is not None
                        and self._selector_hold.panel_id == panel_id
                    )
        for panel_id, binding in tuple(self._numeric_bindings.items()):
            if panel_id not in target_panel_ids:
                self._reset_numeric_binding(panel_id)
            elif previous is not None and panel_id in self._panel_ids:
                old_panel = previous[0].panels[self._panel_ids.index(panel_id)]
                new_panel = frame.panels[target_panel_ids.index(panel_id)]
                if self._panel_semantics_changed(old_panel, new_panel):
                    self._clear_numeric_transient(
                        binding,
                        clear_applied_span=True,
                        clear_pending=True,
                    )
                    cancel_interaction = (
                        interaction_was_active
                        and self._selector_hold is not None
                        and self._selector_hold.panel_id == panel_id
                    )
        if cancel_interaction:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
        if promoting:
            self._panel_ids = target_panel_ids
            self._columns = target_columns
            self._staged_layout = None
        self._active_layout_identity = target_identity
        self._selector_viewport = (
            target_viewport if self._selector_panel_id is not None else None
        )
        for panel_id, viewport in target_numeric_viewports.items():
            binding = self._numeric_bindings.get(panel_id)
            if binding is not None:
                binding.viewport = viewport
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
        if not self._image_interaction_is_pending():
            self._pending_origin = None
        for panel_id, binding in self._numeric_bindings.items():
            pending = binding.pending_viewport
            candidate = target_numeric_viewports.get(panel_id)
            if (
                pending is not None
                and candidate is not None
                and candidate.display_revision >= pending.display_revision
            ):
                binding.pending_viewport = None
            if binding.pending_viewport is None:
                binding.pending_origin = None
        hover_position = self._hover_position
        numeric_hover_positions = {
            panel_id: binding.hover_position
            for panel_id, binding in self._numeric_bindings.items()
        }
        self._front = (frame, prepared)
        if (
            self._image_interaction_armed()
            and not self._image_interaction_is_pending()
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
        for panel_id, binding in self._numeric_bindings.items():
            position = numeric_hover_positions.get(panel_id)
            target = self._numeric_target(binding)
            sample = None
            if (
                self._numeric_interaction_armed(binding)
                and binding.pending_viewport is None
                and position is not None
                and target is not None
                and target.plot.contains(position)
            ):
                sample = self._numeric_sample_for_target(target, position)
            if sample is not None and position is not None:
                binding.hover_position = QtCore.QPointF(position)
            self._set_numeric_hover(binding, sample)
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self._active_layout_identity = None
        self._staged_layout = None
        self._cancel_active_gesture(
            clear_image_draft=True,
            clear_numeric_spans=True,
        )
        self._pending_viewport = None
        self._pending_color_limits = None
        self._pending_origin = None
        self._selector_applied_bounds = None
        self._set_cross_sample(None)
        self._set_hover_sample(None)
        for binding in self._numeric_bindings.values():
            self._clear_numeric_transient(
                binding,
                clear_applied_span=True,
                clear_pending=True,
            )
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

    @property
    def curve_selector_fault(self) -> RuntimeError | None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("curve")
        return None if binding is None else binding.fault

    @property
    def histogram_selector_fault(self) -> RuntimeError | None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram")
        return None if binding is None else binding.fault

    @property
    def selectors_enabled(self) -> bool:
        """Return the effective board-wide interaction intent."""

        self._require_owner()
        return self._selector_enabled

    def _visible_display(
        self,
        panel_id: str | None,
        payload_type: type | tuple[type, ...],
    ) -> tuple[
        DisplayPayload | None,
        PanelInteractionOrigin | None,
    ]:
        """Resolve one typed payload and origin from the exact painted panel."""

        hold = self._selector_hold
        if hold is not None and hold.panel_id == panel_id:
            payload = hold.display_payload
            if not isinstance(payload, payload_type):
                return None, None
            return payload, PanelInteractionOrigin(
                hold.panel_id,
                hold.board_id,
                hold.layout_generation,
                hold.sequence,
                hold.source_identity,
                hold.presentation,
                _payload_input(payload),
            )
        front = self._front
        if front is None or panel_id is None or panel_id not in self._panel_ids:
            return None, None
        panel = front[0].panels[self._panel_ids.index(panel_id)]
        payload = panel.display_payload
        if not isinstance(payload, payload_type):
            return None, None
        return payload, PanelInteractionOrigin(
            panel.panel_id,
            front[0].board_id,
            front[0].layout_generation,
            front[0].sequence,
            panel.source_identity,
            _panel_presentation(panel),
            _payload_input(payload),
        )

    def visible_image_payload(self) -> ImagePanelPayload | None:
        """Return the exact samples paired with the currently painted IMAGE.

        During A/pan interaction this is the held target payload, not the
        advancing board front.  Setting/Edit can therefore freeze FIXED limits
        from exactly what the operator sees without retaining a BoardFrame.
        """

        self._require_owner()
        payload, _origin = self._visible_display(
            self._selector_panel_id,
            (ImagePanelPayload, SiteMapPanelPayload),
        )
        if isinstance(payload, SiteMapPanelPayload):
            return payload.background
        return payload if isinstance(payload, ImagePanelPayload) else None

    def visible_image_origin(self) -> PanelInteractionOrigin | None:
        """Return provenance for the exact held/current IMAGE being painted."""

        self._require_owner()
        _payload, origin = self._visible_display(
            self._selector_panel_id,
            (ImagePanelPayload, SiteMapPanelPayload),
        )
        return origin

    def visible_site_map_payload(self) -> SiteMapPanelPayload | None:
        """Return the exact composite payload painted by the image-family panel."""

        self._require_owner()
        payload, _origin = self._visible_display(
            self._selector_panel_id,
            SiteMapPanelPayload,
        )
        return payload if isinstance(payload, SiteMapPanelPayload) else None

    def discard_pending_image_interaction(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        """Release only one exact failed display intent.

        The owner calls this after an asynchronously accepted reconfigure ends
        in a terminal render fault.  A delayed failure cannot clear a newer
        pending command because sequence, source, presentation revision, and
        exact evaluated input all participate in ``origin`` equality.
        """

        self._require_owner()
        if not isinstance(origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        if not self._image_interaction_is_pending() or origin != self._pending_origin:
            return False
        self._pending_viewport = None
        self._pending_color_limits = None
        self._pending_origin = None
        self.update()
        return True

    def visible_curve_payload(
        self,
        panel_id: str | None = None,
    ) -> CurvePanelPayload | None:
        """Return the exact held/current CURVE payload currently painted."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("curve", panel_id=panel_id)
        payload, _origin = self._visible_display(
            None if binding is None else binding.panel_id, CurvePanelPayload
        )
        return payload if isinstance(payload, CurvePanelPayload) else None

    def visible_curve_origin(
        self,
        panel_id: str | None = None,
    ) -> PanelInteractionOrigin | None:
        """Return provenance for the exact held/current CURVE being painted."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("curve", panel_id=panel_id)
        _payload, origin = self._visible_display(
            None if binding is None else binding.panel_id, CurvePanelPayload
        )
        return origin

    def visible_histogram_payload(
        self,
        panel_id: str | None = None,
    ) -> HistogramPanelPayload | None:
        """Return the exact held/current HISTOGRAM payload currently painted."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram", panel_id=panel_id)
        payload, _origin = self._visible_display(
            None if binding is None else binding.panel_id, HistogramPanelPayload
        )
        return payload if isinstance(payload, HistogramPanelPayload) else None

    def visible_histogram_origin(
        self,
        panel_id: str | None = None,
    ) -> PanelInteractionOrigin | None:
        """Return provenance for the exact held/current HISTOGRAM front."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram", panel_id=panel_id)
        _payload, origin = self._visible_display(
            None if binding is None else binding.panel_id, HistogramPanelPayload
        )
        return origin

    def discard_pending_curve_interaction(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        """Discard only the exact failed CURVE display intent."""

        self._require_owner()
        if not isinstance(origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        binding = self._numeric_bindings.get(origin.panel_id)
        if (
            binding is None
            or binding.kind != "curve"
            or binding.pending_viewport is None
            or origin != binding.pending_origin
        ):
            return False
        binding.pending_viewport = None
        binding.pending_origin = None
        self.update()
        return True

    def discard_pending_histogram_interaction(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        """Discard only the exact failed HISTOGRAM display intent."""

        self._require_owner()
        if not isinstance(origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        binding = self._numeric_bindings.get(origin.panel_id)
        if (
            binding is None
            or binding.kind != "histogram"
            or binding.pending_viewport is None
            or origin != binding.pending_origin
        ):
            return False
        binding.pending_viewport = None
        binding.pending_origin = None
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
        ) != hold.gesture_identity:
            raise RuntimeError("rectangle gesture differs from its held panel origin")
        if isinstance(hold.display_payload, SiteMapPanelPayload):
            raise RuntimeError(
                "site-map rectangles are display-only candidates; "
                "a spatial box cannot be promoted to authoritative SITE selection"
            )
        viewport = self._require_selector_viewport()
        if viewport.viewport_revision != gesture.viewport_revision:
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
        if self._numeric_bindings and enabled != self._selector_enabled:
            raise ValueError(
                "a second selector family must match the board-wide enabled state; "
                "call set_selectors_enabled explicitly"
            )
        if self._front is not None:
            self._validate_selector_binding(
                panel_id,
                viewport,
                self._front[0],
            )
            if interaction_callback is not None:
                index = self._panel_ids.index(panel_id)
                if _image_payload(self._front[0].panels[index]) is None:
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
        self._image_binding_enabled = True
        self._image_interaction_ready = enabled
        if not self._numeric_bindings:
            self._selector_enabled = enabled
        self.update()

    def set_interaction_readiness(
        self,
        *,
        image: bool,
        curve: bool,
        histogram: bool = False,
    ) -> None:
        """Arm only panel families whose painted provenance is current.

        ``set_selectors_enabled`` carries the operator's board-wide intent.
        Readiness is a separate presentation fact supplied by the owner after
        comparing each painted payload with its current semantic state.  A
        stale sibling therefore cannot emit an intent merely because another
        panel on the same board is current.
        """

        self._require_owner()
        if (
            not isinstance(image, bool)
            or not isinstance(curve, bool)
            or not isinstance(histogram, bool)
        ):
            raise TypeError("interaction readiness values must be bool")
        if not image and self._image_interaction_ready:
            self._cancel_image_gesture(clear_draft=True)
            self._set_hover_sample(None)
        self._image_interaction_ready = image
        readiness = {"curve": curve, "histogram": histogram}
        for binding in self._numeric_bindings.values():
            ready = readiness[binding.kind]
            if not ready and binding.interaction_ready:
                self._cancel_numeric_gesture(binding, clear_span=True)
                self._set_numeric_hover(binding, None)
            binding.interaction_ready = ready
        self.update()

    def set_selectors_enabled(self, enabled: bool) -> None:
        """Park or arm all healthy bound selector families without rebuilding."""

        self._require_owner()
        if not isinstance(enabled, bool):
            raise TypeError("selector enabled must be bool")
        healthy_image = (
            self._image_binding_enabled
            and self._image_interaction_ready
            and self._selector_panel_id is not None
            and self._selector_viewport is not None
            and self._selector_callback is not None
            and self._selector_fault is None
        )
        healthy_numeric = any(
            binding.binding_enabled
            and binding.interaction_ready
            and binding.viewport is not None
            and binding.fault is None
            for binding in self._numeric_bindings.values()
        )
        if enabled and not (healthy_image or healthy_numeric):
            raise RuntimeError("no healthy selector binding is available")
        self._selector_enabled = enabled
        if not enabled:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
            self._set_hover_sample(None)
            for binding in self._numeric_bindings.values():
                self._set_numeric_hover(binding, None)
        self.update()

    def _image_interaction_armed(self) -> bool:
        return (
            self._selector_enabled
            and self._image_binding_enabled
            and self._image_interaction_ready
        )

    def _numeric_interaction_armed(self, binding: _NumericPanelBinding) -> bool:
        return (
            self._selector_enabled
            and binding.binding_enabled
            and binding.interaction_ready
            and binding.viewport is not None
            and binding.fault is None
        )

    def bind_curve_interaction(
        self,
        panel_id: str,
        callback: Callable[[CurveInteractionIntent], object],
        *,
        enabled: bool = True,
    ) -> None:
        """Bind one CURVE panel to display-only typed intents."""

        self._bind_numeric_interaction(
            "curve", panel_id, callback, enabled=enabled
        )

    def bind_histogram_interaction(
        self,
        panel_id: str,
        callback: Callable[[HistogramInteractionIntent], object],
        *,
        enabled: bool = True,
    ) -> None:
        """Bind one HISTOGRAM panel to display-only typed intents."""

        self._bind_numeric_interaction(
            "histogram", panel_id, callback, enabled=enabled
        )

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
        self._cancel_image_gesture(clear_draft=False)
        self.update()

    def set_site_map_rectangle_candidate(
        self,
        bounds: NormalizedRectangle | None,
    ) -> None:
        """Retain one display-only spatial candidate without forging SITE authority."""

        self._require_owner()
        if self.visible_site_map_payload() is None:
            raise RuntimeError("no exact SiteMap payload is currently painted")
        self._selector_applied_bounds = (
            None if bounds is None else validate_normalized_rectangle(bounds)
        )
        self._cancel_image_gesture(clear_draft=True)
        self.update()

    def set_curve_range_candidate(
        self,
        x_span: tuple[float, float] | None,
        *,
        panel_id: str | None = None,
    ) -> None:
        """Project the Workbench-owned display-only CURVE range candidate."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("curve", panel_id=panel_id)
        if binding is None:
            if x_span is None:
                return
            raise RuntimeError("no curve panel is bound")
        binding.applied_span = (
            None
            if x_span is None
            else validated_display_range(x_span, "curve range candidate")
        )
        self.update()

    def set_histogram_range_candidate(
        self,
        x_span: tuple[float, float] | None,
        *,
        panel_id: str | None = None,
    ) -> None:
        """Project one display-only HISTOGRAM value-range candidate."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram", panel_id=panel_id)
        if binding is None:
            if x_span is None:
                return
            raise RuntimeError("no histogram panel is bound")
        binding.applied_span = (
            None
            if x_span is None
            else validated_display_range(x_span, "histogram range candidate")
        )
        self.update()

    def unbind_rectangle_selector(self) -> None:
        self._require_owner()
        self._reset_rectangle_selector()
        self.update()

    def unbind_curve_interaction(self, panel_id: str | None = None) -> None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("curve", panel_id=panel_id)
        if binding is not None:
            self._reset_numeric_binding(binding.panel_id)
        self.update()

    def unbind_histogram_interaction(self, panel_id: str | None = None) -> None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram", panel_id=panel_id)
        if binding is not None:
            self._reset_numeric_binding(binding.panel_id)
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
                hold.display_payload
                if hold is not None and hold.panel_id == panel_id
                else panel.display_payload
            )
            bounds = _panel_bounds(
                self.rect(),
                index=index,
                count=len(images),
                columns=self._columns,
            )
            image_payload = None
            if isinstance(payload, (CurvePanelPayload, HistogramPanelPayload)):
                target = bounds
                source = QtCore.QRectF(
                    0.0,
                    0.0,
                    float(image.width()),
                    float(image.height()),
                )
                rail = None
            else:
                image_payload = (
                    payload.background
                    if isinstance(payload, SiteMapPanelPayload)
                    else payload
                    if isinstance(payload, ImagePanelPayload)
                    else None
                )
                target, source, rail = _panel_image_geometry(
                    bounds,
                    image,
                    image_payload,
                    site_map_payload=(
                        payload
                        if isinstance(payload, SiteMapPanelPayload)
                        else None
                    ),
                )
            painter.drawImage(QtCore.QRectF(target), image, source)
            if isinstance(payload, SiteMapPanelPayload):
                self._paint_site_map_rings(painter, payload, target)
            if image_payload is not None and rail is not None:
                self._paint_color_rail(painter, image_payload, rail)
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
        self._paint_numeric_overlays(painter)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._selector_enabled:
            super().mousePressEvent(event)
            return
        numeric_target = self._numeric_target_at(event.localPos())
        if (
            numeric_target is not None
            and self._numeric_interaction_armed(numeric_target.binding)
        ):
            binding = numeric_target.binding
            if (
                binding.pending_viewport is not None
                or self._selector_hold is not None
            ):
                event.accept()
                return
            point = self._numeric_normalized_point(
                numeric_target, event.localPos()
            )
            viewport = numeric_target.payload.viewport
            if event.button() == QtCore.Qt.RightButton:
                x, y = viewport.widget_normalized_to_data(*point)
                binding.cross = _NumericCross(x, y)
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            if event.button() == QtCore.Qt.MiddleButton:
                self._selector_hold = self._held_panel_from_numeric_target(
                    numeric_target
                )
                binding.pan_anchor = point[0]
                binding.pan_origin = viewport
                binding.pan_candidate = viewport.x_limits
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                self._selector_hold = self._held_panel_from_numeric_target(
                    numeric_target
                )
                binding.span_anchor = point[0]
                binding.span_candidate = None
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            super().mousePressEvent(event)
            return
        if not self._image_interaction_armed():
            super().mousePressEvent(event)
            return
        target = self._selector_target()
        rail_target = self._clim_rail_target()
        hits_image = target is not None and target[0].contains(event.pos())
        hits_rail = rail_target is not None and rail_target[0].contains(event.pos())
        if self._image_interaction_is_pending() or self._selector_hold is not None:
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
        numeric_binding = self._active_numeric_binding()
        if numeric_binding is not None and numeric_binding.span_anchor is not None:
            target = self._numeric_target(numeric_binding)
            if target is not None:
                viewport = target.payload.viewport
                point = self._numeric_normalized_point(
                    target,
                    event.localPos(),
                    clamp_to_plot=True,
                )
                if point[0] == numeric_binding.span_anchor:
                    numeric_binding.span_candidate = None
                else:
                    try:
                        numeric_binding.span_candidate = viewport.selection_x_span(
                            numeric_binding.span_anchor,
                            point[0],
                        )
                    except ValueError:
                        numeric_binding.span_candidate = None
                self.update()
            event.accept()
            return
        if (
            numeric_binding is not None
            and numeric_binding.pan_anchor is not None
            and numeric_binding.pan_origin is not None
        ):
            target = self._numeric_target(numeric_binding)
            if target is not None:
                point = self._numeric_normalized_point(target, event.localPos())
                try:
                    numeric_binding.pan_candidate = (
                        numeric_binding.pan_origin.panned_x_limits(
                        numeric_binding.pan_anchor,
                        point[0],
                        start_x_limits=numeric_binding.pan_origin.x_limits,
                        )
                    )
                except ValueError:
                    numeric_binding.pan_candidate = None
            event.accept()
            return
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

        numeric_target = self._numeric_target_at(event.localPos())
        hovered_numeric = None
        if (
            numeric_target is not None
            and self._numeric_interaction_armed(numeric_target.binding)
            and numeric_target.binding.pending_viewport is None
        ):
            hovered_numeric = numeric_target.binding
            sample = self._numeric_sample_for_target(
                numeric_target,
                event.localPos(),
            )
            hovered_numeric.hover_position = (
                None if sample is None else QtCore.QPointF(event.localPos())
            )
            self._set_numeric_hover(hovered_numeric, sample)
            self._set_hover_sample(None)
        for binding in self._numeric_bindings.values():
            if binding is not hovered_numeric:
                self._set_numeric_hover(binding, None)
        if hovered_numeric is not None:
            self.update()
            super().mouseMoveEvent(event)
            return
        if (
            self._image_interaction_armed()
            and not self._image_interaction_is_pending()
        ):
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
        numeric_binding = self._active_numeric_binding()
        if (
            numeric_binding is not None
            and numeric_binding.pan_anchor is not None
            and event.button() == QtCore.Qt.MiddleButton
        ):
            candidate = numeric_binding.pan_candidate
            hold = self._selector_hold
            try:
                if candidate is not None and hold is not None:
                    self._commit_numeric_viewport(
                        numeric_binding,
                        candidate,
                        hold=hold,
                    )
            finally:
                self._cancel_active_gesture(
                    clear_image_draft=False,
                    clear_numeric_spans=False,
                )
                self.update()
            event.accept()
            return
        if (
            numeric_binding is not None
            and numeric_binding.span_anchor is not None
            and event.button() == QtCore.Qt.LeftButton
        ):
            candidate = numeric_binding.span_candidate
            hold = self._selector_hold
            numeric_binding.span_anchor = None
            try:
                if hold is not None:
                    origin = self._numeric_interaction_origin(
                        numeric_binding,
                        hold=hold,
                    )
                    gesture: _NumericIntent = (
                        CurveRangeGesture(origin, candidate)
                        if numeric_binding.kind == "curve"
                        else HistogramRangeGesture(origin, candidate)
                    )
                    numeric_binding.callback(gesture)
            except BaseException as error:
                if numeric_binding.fault is None:
                    numeric_binding.fault = detached_render_fault(error)
                numeric_binding.binding_enabled = False
            finally:
                self._cancel_active_gesture(
                    clear_image_draft=False,
                    clear_numeric_spans=True,
                )
                self.update()
            event.accept()
            return
        if self._clim_drag is not None and event.button() == QtCore.Qt.LeftButton:
            candidate = self._clim_candidate
            hold = self._selector_hold
            try:
                if candidate is not None and hold is not None:
                    self._commit_color_limits(candidate, hold=hold)
            finally:
                self._cancel_image_gesture(clear_draft=False)
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
                self._cancel_image_gesture(clear_draft=False)
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
                    viewport_revision=self._require_selector_viewport().viewport_revision,
                )
                callback(gesture)
                delivered = True
        except BaseException as error:
            if self._selector_fault is None:
                self._selector_fault = detached_render_fault(error)
            self._image_binding_enabled = False
        finally:
            self._cancel_image_gesture(
                clear_draft=(bounds is not None and not delivered)
            )
            self.update()
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self._selector_enabled:
            super().wheelEvent(event)
            return
        numeric_target = self._numeric_target_at(event.posF())
        if (
            numeric_target is not None
            and self._numeric_interaction_armed(numeric_target.binding)
        ):
            binding = numeric_target.binding
            if (
                binding.pending_viewport is not None
                or self._selector_hold is not None
            ):
                event.accept()
                return
            delta = event.angleDelta().y()
            if delta == 0:
                super().wheelEvent(event)
                return
            point = self._numeric_normalized_point(numeric_target, event.posF())
            viewport = numeric_target.payload.viewport
            anchor_x = viewport.widget_normalized_to_data(*point)[0]
            factor = 1.0 / 1.1 if delta < 0 else 1.1
            try:
                candidate = viewport.zoomed_x_limits(anchor_x, factor)
            except ValueError:
                candidate = None
            if candidate is not None:
                self._commit_numeric_viewport(binding, candidate)
            self._set_numeric_hover(binding, None)
            self.update()
            event.accept()
            return
        if not self._image_interaction_armed() or self._interaction_callback is None:
            super().wheelEvent(event)
            return
        target = self._selector_target()
        rail_target = self._clim_rail_target()
        position = event.pos()
        hits_image = target is not None and target[0].contains(position)
        hits_rail = rail_target is not None and rail_target[0].contains(position)
        if self._image_interaction_is_pending() or self._selector_hold is not None:
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
        numeric_target = self._numeric_target_at(event.localPos())
        if (
            numeric_target is not None
            and self._numeric_interaction_armed(numeric_target.binding)
        ):
            binding = numeric_target.binding
            if (
                binding.pending_viewport is not None
                or self._selector_hold is not None
            ):
                event.accept()
                return
            if event.button() == QtCore.Qt.RightButton:
                binding.cross = None
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            if (
                event.button() == QtCore.Qt.MiddleButton
            ):
                viewport = numeric_target.payload.viewport
                self._commit_numeric_viewport(
                    binding,
                    viewport.home_x_limits
                    if binding.applied_span is None
                    else binding.applied_span,
                )
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            super().mouseDoubleClickEvent(event)
            return
        if not self._image_interaction_armed():
            super().mouseDoubleClickEvent(event)
            return
        target = self._selector_target()
        rail_target = self._clim_rail_target()
        hits_image = target is not None and target[0].contains(event.pos())
        hits_rail = rail_target is not None and rail_target[0].contains(event.pos())
        if self._image_interaction_is_pending() or self._selector_hold is not None:
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
        for binding in self._numeric_bindings.values():
            self._set_numeric_hover(binding, None)
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and self._selector_hold is not None:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self._selector_hold is not None:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
        self._set_hover_sample(None)
        self._hover_position = None
        for binding in self._numeric_bindings.values():
            self._set_numeric_hover(binding, None)
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._reset_rectangle_selector()
        self._reset_all_numeric_bindings()
        self._front = None
        self._active_layout_identity = None
        self._staged_layout = None
        self._closed = True
        super().closeEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.DeferredDelete:
            self._reset_rectangle_selector()
            self._reset_all_numeric_bindings()
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
                self._cancel_active_gesture(
                    clear_image_draft=True,
                    clear_numeric_spans=True,
                )
            if getattr(self, "_hover_sample", None) is not None:
                self._set_hover_sample(None)
                changed = True
            for binding in getattr(self, "_numeric_bindings", {}).values():
                if binding.hover is not None:
                    self._set_numeric_hover(binding, None)
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
        payload = _image_payload(frame.panels[index])
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
        payload = _image_payload(panel)
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

    def _numeric_binding_for_kind(
        self,
        kind: _NumericKind,
        *,
        panel_id: str | None = None,
    ) -> _NumericPanelBinding | None:
        if panel_id is not None:
            panel_id = canonical_text(panel_id, f"{kind} panel_id")
            binding = self._numeric_bindings.get(panel_id)
            return binding if binding is not None and binding.kind == kind else None
        matches = tuple(
            binding
            for binding in self._numeric_bindings.values()
            if binding.kind == kind
        )
        if len(matches) > 1:
            raise ValueError(f"multiple {kind} panels are bound; panel_id is required")
        return None if not matches else matches[0]

    def _bind_numeric_interaction(
        self,
        kind: _NumericKind,
        panel_id: str,
        callback: Callable[[_NumericIntent], object],
        *,
        enabled: bool,
    ) -> None:
        self._require_owner()
        panel_id = canonical_text(panel_id, f"{kind} panel_id")
        if panel_id not in self._panel_ids:
            raise ValueError(f"{kind} panel_id is absent from this board")
        if not callable(callback):
            raise TypeError(f"{kind} callback must be callable")
        if not isinstance(enabled, bool):
            raise TypeError("selector enabled must be bool")
        has_other_family = (
            self._selector_panel_id is not None
            or any(value.panel_id != panel_id for value in self._numeric_bindings.values())
        )
        if has_other_family and enabled != self._selector_enabled:
            raise ValueError(
                "a second selector family must match the board-wide enabled state; "
                "call set_selectors_enabled explicitly"
            )
        viewport = None
        if self._front is not None:
            panel = self._front[0].panels[self._panel_ids.index(panel_id)]
            payload = _numeric_payload(panel, kind)
            if payload is None:
                raise ValueError(
                    f"{kind} interaction requires exact {kind.title()}PanelPayload"
                )
            if _panel_presentation(panel).panel_revision != payload.viewport.display_revision:
                raise ValueError(
                    f"{kind} payload viewport revision differs from its presentation"
                )
            viewport = payload.viewport
        if panel_id in self._numeric_bindings:
            self._reset_numeric_binding(panel_id)
        self._numeric_bindings[panel_id] = _NumericPanelBinding(
            kind,
            panel_id,
            callback,
            viewport=viewport,
            interaction_ready=enabled,
        )
        if not has_other_family:
            self._selector_enabled = enabled
        self.update()

    def _numeric_viewport_for_presented_panel(
        self,
        binding: _NumericPanelBinding,
        frame: BoardFrame,
        *,
        panel_ids: tuple[str, ...],
    ) -> _NumericViewport:
        panel = frame.panels[panel_ids.index(binding.panel_id)]
        payload = _numeric_payload(panel, binding.kind)
        if payload is None:
            raise ValueError(
                f"{binding.kind} interaction requires its exact typed payload"
            )
        candidate = payload.viewport
        if _panel_presentation(panel).panel_revision != candidate.display_revision:
            raise ValueError(
                f"{binding.kind} viewport revision differs from its presentation"
            )
        current = binding.viewport
        previous = self._front
        structurally_new = previous is None or binding.panel_id not in self._panel_ids
        if not structurally_new and previous is not None:
            old_panel = previous[0].panels[
                self._panel_ids.index(binding.panel_id)
            ]
            structurally_new = self._panel_semantics_changed(old_panel, panel)
        if current is None or structurally_new:
            return candidate
        if type(candidate) is not type(current):
            raise ValueError("numeric viewport type changed without panel structure change")
        if (
            isinstance(candidate, CurveViewportTransform)
            and isinstance(current, CurveViewportTransform)
            and candidate.x_axis != current.x_axis
        ):
            raise ValueError("curve x axis changed without panel structure change")
        if candidate.display_revision < current.display_revision:
            raise ValueError(
                f"stale {binding.kind} display revision cannot replace the visible front"
            )
        pending = binding.pending_viewport
        if (
            pending is not None
            and candidate.display_revision == pending.display_revision
            and candidate.x_limits != pending.x_limits
        ):
            raise ValueError(
                f"pending {binding.kind} viewport returned conflicting x bounds"
            )
        if (
            isinstance(candidate, HistogramViewportTransform)
            and isinstance(pending, HistogramViewportTransform)
            and candidate.display_revision == pending.display_revision
            and (
                candidate.count_scale is not pending.count_scale
                or candidate.relim_mode is not pending.relim_mode
                or candidate.x_limits_are_auto != pending.x_limits_are_auto
                or candidate.bin_count != pending.bin_count
                or (
                    candidate.relim_mode is RelimMode.FIXED
                    and candidate.count_limits != pending.count_limits
                )
            )
        ):
            raise ValueError(
                "pending histogram viewport returned conflicting authored state"
            )
        if candidate.display_revision == current.display_revision:
            if isinstance(candidate, CurveViewportTransform) and (
                candidate.x_limits != current.x_limits
                or candidate.home_x_limits != current.home_x_limits
            ):
                raise ValueError(
                    "one curve display revision describes conflicting x bounds"
                )
            if (
                isinstance(candidate, HistogramViewportTransform)
                and isinstance(current, HistogramViewportTransform)
                and (
                    candidate.count_scale is not current.count_scale
                    or candidate.relim_mode is not current.relim_mode
                    or candidate.x_limits_are_auto != current.x_limits_are_auto
                    or candidate.bin_count != current.bin_count
                    or (
                        not candidate.x_limits_are_auto
                        and candidate.x_limits != current.x_limits
                    )
                    or (
                        candidate.relim_mode is RelimMode.FIXED
                        and candidate.count_limits != current.count_limits
                    )
                )
            ):
                raise ValueError(
                    "one histogram display revision describes conflicting authored state"
                )
            # Histogram home/x/count limits are data-derived in AUTO modes and
            # may legitimately advance at one authored display revision.
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
        composite = (
            _site_map_payload(hold)
            if hold is not None and hold.panel_id == panel_id
            else _site_map_payload(front[0].panels[index])
        )
        payload = (
            _image_payload(hold)
            if hold is not None and hold.panel_id == panel_id
            else _image_payload(front[0].panels[index])
        )
        target, _source, _rail = _panel_image_geometry(
            bounds,
            image,
            payload,
            site_map_payload=composite,
        )
        return target, front[0], front[0].panels[index], prepared

    def _numeric_target(
        self,
        binding: _NumericPanelBinding,
    ) -> _NumericTarget | None:
        front = self._front
        panel_id = binding.panel_id
        if front is None or panel_id is None or panel_id not in self._panel_ids:
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
            _numeric_payload(hold, binding.kind)
            if hold is not None and hold.panel_id == panel_id
            else _numeric_payload(panel, binding.kind)
        )
        if payload is None:
            return None
        bounds = _panel_bounds(
            self.rect(),
            index=index,
            count=len(front[1]),
            columns=self._columns,
        )
        plot = _numeric_plot_geometry(bounds, payload.viewport)
        return _NumericTarget(plot, front[0], panel, prepared, payload, bounds, binding)

    def _numeric_target_at(self, point: QtCore.QPointF) -> _NumericTarget | None:
        for binding in self._numeric_bindings.values():
            target = self._numeric_target(binding)
            if target is not None and target.plot.contains(point):
                return target
        return None

    @staticmethod
    def _numeric_normalized_point(
        target: _NumericTarget,
        point: QtCore.QPointF,
        *,
        clamp_to_plot: bool = False,
    ) -> tuple[float, float]:
        bounds = target.bounds
        x = (float(point.x()) - bounds.x()) / max(1, bounds.width())
        y = (float(point.y()) - bounds.y()) / max(1, bounds.height())
        if clamp_to_plot:
            left, top, right, bottom = target.payload.viewport.plot_bounds
            x = min(right, max(left, x))
            y = min(bottom, max(top, y))
        return x, y

    def _numeric_sample_for_target(
        self,
        target: _NumericTarget,
        point: QtCore.QPointF,
    ) -> _CurveSample | _HistogramBinSample | None:
        if isinstance(target.payload, HistogramPanelPayload):
            return self._histogram_sample_for_target(target, point)
        return self._curve_sample_for_numeric_target(target, point)

    def _curve_sample_for_numeric_target(
        self,
        target: _NumericTarget,
        point: QtCore.QPointF,
    ) -> _CurveSample | None:
        payload = target.payload
        assert isinstance(payload, CurvePanelPayload)
        viewport = payload.viewport
        bounds = target.bounds
        best: tuple[float, int, int, _CurveSample] | None = None
        coordinates = np.asarray(
            payload.series[0].data.x_axis.coordinates,
            dtype=np.float64,
        )
        x_low, x_high = viewport.x_limits
        y_low, y_high = viewport.y_limits
        left, top, right, bottom = viewport.plot_bounds
        x_widget = bounds.x() + (
            left
            + (coordinates - x_low) / (x_high - x_low) * (right - left)
        ) * bounds.width()
        for series_index, (series, label) in enumerate(
            zip(payload.series, payload.series_labels)
        ):
            curve = series.data
            values = np.asarray(curve.values, dtype=np.float64)
            valid = np.asarray(curve.validity, dtype=bool)
            visible = (
                valid
                & np.isfinite(values)
                & (coordinates >= x_low)
                & (coordinates <= x_high)
                & (values >= y_low)
                & (values <= y_high)
            )
            sample_indices = np.flatnonzero(visible)
            if not sample_indices.size:
                continue
            visible_values = values[sample_indices]
            y_widget = bounds.y() + (
                top
                + (y_high - visible_values) / (y_high - y_low) * (bottom - top)
            ) * bounds.height()
            distances = (
                (x_widget[sample_indices] - point.x()) ** 2
                + (y_widget - point.y()) ** 2
            )
            local_index = int(np.argmin(distances))
            sample_index = int(sample_indices[local_index])
            sample = _CurveSample(
                label,
                float(coordinates[sample_index]),
                float(values[sample_index]),
            )
            candidate = (
                float(distances[local_index]),
                series_index,
                sample_index,
                sample,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        return None if best is None else best[3]

    def _histogram_sample_for_target(
        self,
        target: _NumericTarget,
        point: QtCore.QPointF,
    ) -> _HistogramBinSample | None:
        payload = target.payload
        assert isinstance(payload, HistogramPanelPayload)
        viewport = payload.viewport
        normalized = self._numeric_normalized_point(target, point)
        x_value, _count_value = viewport.widget_normalized_to_data(*normalized)
        edges = np.asarray(payload.bin_edges, dtype=np.float64)
        index = int(np.searchsorted(edges, x_value, side="right") - 1)
        if x_value == float(edges[-1]):
            index = len(edges) - 2
        if not 0 <= index < len(edges) - 1:
            return None
        best: tuple[float, int, _HistogramBinSample] | None = None
        for series_index, (counts, label) in enumerate(
            zip(payload.bin_counts, payload.series_labels, strict=True)
        ):
            count = int(counts[index])
            if viewport.count_scale.value == "log" and count <= 0:
                continue
            if not viewport.count_limits[0] <= count <= viewport.count_limits[1]:
                continue
            sample = _HistogramBinSample(
                label,
                float(edges[index]),
                float(edges[index + 1]),
                count,
                index == len(edges) - 2,
            )
            widget = viewport.data_to_widget_normalized(sample.x, sample.y)
            widget_y = target.bounds.y() + widget[1] * target.bounds.height()
            candidate = (abs(widget_y - point.y()), series_index, sample)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return None if best is None else best[2]

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
            _image_payload(hold)
            if hold is not None and hold.panel_id == panel_id
            else _image_payload(panel)
        )
        composite = (
            _site_map_payload(hold)
            if hold is not None and hold.panel_id == panel_id
            else _site_map_payload(panel)
        )
        if payload is None:
            return None
        bounds = _panel_bounds(
            self.rect(),
            index=index,
            count=len(front[1]),
            columns=self._columns,
        )
        _target, _source, rail = _panel_image_geometry(
            bounds,
            prepared[1],
            payload,
            site_map_payload=composite,
        )
        if rail is None:
            return None
        return rail, front[0], panel, prepared, payload

    def _viewport_for_target(self, target) -> ImageViewportTransform:
        hold = self._selector_hold
        if hold is not None and hold.panel_id == target[2].panel_id:
            payload = _image_payload(hold)
            if payload is not None:
                return payload.viewport
        payload = _image_payload(target[2])
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
            payload = _image_payload(hold)
            presentation = hold.presentation
        else:
            payload = _image_payload(panel)
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
        if self._image_interaction_is_pending():
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
            self._image_binding_enabled = False
            self._set_hover_sample(None)
            return False
        return True

    def _commit_numeric_viewport(
        self,
        binding: _NumericPanelBinding,
        x_limits: tuple[float, float],
        *,
        hold: _HeldPanelFront | None = None,
    ) -> bool:
        payload = (
            _numeric_payload(hold, binding.kind)
            if hold is not None
            else self._visible_display(
                binding.panel_id,
                CurvePanelPayload
                if binding.kind == "curve"
                else HistogramPanelPayload,
            )[0]
        )
        if payload is None or x_limits == payload.viewport.x_limits:
            return False
        assert isinstance(payload, (CurvePanelPayload, HistogramPanelPayload))
        if binding.pending_viewport is not None:
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
        origin = self._numeric_interaction_origin(binding, hold=hold)
        candidate = replace(
            payload.viewport,
            display_revision=payload.viewport.display_revision + 1,
            x_limits=x_limits,
            **(
                {"x_limits_are_auto": False}
                if isinstance(payload, HistogramPanelPayload)
                else {}
            ),
        )
        command: _NumericIntent = (
            CurveViewportCommit(origin, candidate)
            if binding.kind == "curve"
            else HistogramViewportCommit(origin, candidate)
        )
        binding.pending_viewport = candidate
        binding.pending_origin = origin
        try:
            binding.callback(command)
        except BaseException as error:
            binding.pending_viewport = None
            binding.pending_origin = None
            if binding.fault is None:
                binding.fault = detached_render_fault(error)
            binding.binding_enabled = False
            self._set_numeric_hover(binding, None)
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
        payload = _image_payload(hold)
        if payload is None or limits == payload.color_limits:
            return False
        if self._image_interaction_is_pending():
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
            self._image_binding_enabled = False
            self._set_hover_sample(None)
            return False
        return True

    def _image_interaction_is_pending(self) -> bool:
        return (
            self._pending_viewport is not None
            or self._pending_color_limits is not None
        )

    def _interaction_origin(
        self,
        *,
        hold: _HeldPanelFront | None = None,
    ) -> PanelInteractionOrigin:
        return self._require_interaction_origin(
            panel_id=self._selector_panel_id,
            payload_type=(ImagePanelPayload, SiteMapPanelPayload),
            hold=hold,
            kind="image",
        )

    def _numeric_interaction_origin(
        self,
        binding: _NumericPanelBinding,
        *,
        hold: _HeldPanelFront | None = None,
    ) -> PanelInteractionOrigin:
        return self._require_interaction_origin(
            panel_id=binding.panel_id,
            payload_type=(
                CurvePanelPayload
                if binding.kind == "curve"
                else HistogramPanelPayload
            ),
            hold=hold,
            kind=binding.kind,
        )

    def _require_interaction_origin(
        self,
        *,
        panel_id: str | None,
        payload_type: type | tuple[type, ...],
        hold: _HeldPanelFront | None,
        kind: str,
    ) -> PanelInteractionOrigin:
        if hold is not None and hold is not self._selector_hold:
            raise RuntimeError(f"{kind} interaction hold is no longer painted")
        _payload, origin = self._visible_display(panel_id, payload_type)
        if origin is None:
            raise RuntimeError(f"{kind} interaction origin has no exact payload")
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

    def _set_numeric_hover(
        self,
        binding: _NumericPanelBinding,
        sample: _CurveSample | _HistogramBinSample | None,
    ) -> None:
        binding.hover = sample
        if sample is None:
            binding.hover_position = None

    def _active_numeric_binding(self) -> _NumericPanelBinding | None:
        hold = self._selector_hold
        if hold is None:
            return None
        return self._numeric_bindings.get(hold.panel_id)

    def _held_panel_from_target(self, target) -> _HeldPanelFront:
        frame, panel, prepared = target[1], target[2], target[3]
        return _HeldPanelFront(
            panel_id=panel.panel_id,
            board_id=frame.board_id,
            layout_generation=frame.layout_generation,
            sequence=frame.sequence,
            coherence_group=panel.coherence_group,
            source_identity=panel.source_identity,
            presentation=_panel_presentation(panel),
            raster_geometry=_raster_geometry(panel),
            prepared=prepared,
            display_payload=(
                target[4]
                if len(target) > 4
                else panel.display_payload
            ),
        )

    def _held_panel_from_numeric_target(
        self,
        target: _NumericTarget,
    ) -> _HeldPanelFront:
        return _HeldPanelFront(
            panel_id=target.panel.panel_id,
            board_id=target.frame.board_id,
            layout_generation=target.frame.layout_generation,
            sequence=target.frame.sequence,
            coherence_group=target.panel.coherence_group,
            source_identity=target.panel.source_identity,
            presentation=_panel_presentation(target.panel),
            raster_geometry=_raster_geometry(target.panel),
            prepared=target.prepared,
            display_payload=target.payload,
        )

    @staticmethod
    def _panel_semantics_changed(old: PanelFrame, new: PanelFrame) -> bool:
        old_presentation = _panel_presentation(old)
        new_presentation = _panel_presentation(new)
        old_payload = old.display_payload
        new_payload = new.display_payload

        def interaction_geometry(payload):
            if isinstance(payload, ImagePanelPayload):
                return (ImagePanelPayload, payload.viewport.axes)
            if isinstance(payload, CurvePanelPayload):
                return (CurvePanelPayload, payload.viewport.x_axis)
            if isinstance(payload, HistogramPanelPayload):
                return (
                    HistogramPanelPayload,
                    payload.value_unit,
                    payload.series_labels,
                )
            if isinstance(payload, SiteMapPanelPayload):
                return (
                    SiteMapPanelPayload,
                    payload.background.viewport.axes,
                    _input_structure(payload.background.evaluated_input),
                    payload.site_axis,
                    payload.coordinate_frame,
                    payload.geometry_identity,
                )
            return (None,)

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
            or interaction_geometry(old_payload) != interaction_geometry(new_payload)
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
        index = panel_ids.index(hold.panel_id)
        panel = frame.panels[index]
        held_payload = hold.display_payload
        current_payload = panel.display_payload
        if isinstance(held_payload, ImagePanelPayload):
            payload_matches = (
                isinstance(current_payload, ImagePanelPayload)
                and current_payload.viewport.axes == held_payload.viewport.axes
            )
        elif isinstance(held_payload, CurvePanelPayload):
            payload_matches = (
                isinstance(current_payload, CurvePanelPayload)
                and current_payload.viewport.x_axis == held_payload.viewport.x_axis
            )
        elif isinstance(held_payload, HistogramPanelPayload):
            payload_matches = (
                isinstance(current_payload, HistogramPanelPayload)
                and current_payload.value_unit == held_payload.value_unit
                and current_payload.series_labels == held_payload.series_labels
            )
        elif isinstance(held_payload, SiteMapPanelPayload):
            payload_matches = (
                isinstance(current_payload, SiteMapPanelPayload)
                and current_payload.background.viewport.axes
                == held_payload.background.viewport.axes
                and _input_structure(current_payload.background.evaluated_input)
                == _input_structure(held_payload.background.evaluated_input)
                and current_payload.site_axis == held_payload.site_axis
                and current_payload.coordinate_frame == held_payload.coordinate_frame
                and current_payload.geometry_identity == held_payload.geometry_identity
            )
        else:
            payload_matches = current_payload is None
        return (
            panel.panel_id == hold.panel_id
            and panel.coherence_group == hold.coherence_group
            and panel.source_identity == hold.source_identity
            and _panel_presentation(panel) == hold.presentation
            and _raster_geometry(panel) == hold.raster_geometry
            and payload_matches
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
    def _paint_site_map_rings(
        painter: QtGui.QPainter,
        payload: SiteMapPanelPayload,
        target: QtCore.QRect,
    ) -> None:
        """Paint calibrated rings over the exact background front in Qt."""

        viewport = payload.background.viewport
        width, height = payload.visible_ring_span
        ring_width = width * target.width()
        ring_height = height * target.height()
        left, top, right, bottom = viewport.visible_bounds
        occupied = payload.site_validity & payload.occupied
        empty = payload.site_validity & ~payload.occupied
        invalid = ~payload.site_validity
        styles = (
            (
                empty,
                SITE_EMPTY_COLOR,
                SITE_EMPTY_ALPHA,
                SITE_EMPTY_LINEWIDTH,
                False,
            ),
            (
                occupied,
                SITE_OCCUPIED_COLOR,
                SITE_OCCUPIED_ALPHA,
                SITE_OCCUPIED_LINEWIDTH,
                False,
            ),
            (
                invalid,
                SITE_INVALID_COLOR,
                SITE_INVALID_ALPHA,
                SITE_INVALID_LINEWIDTH,
                True,
            ),
        )
        painter.save()
        try:
            painter.setClipRect(target)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setBrush(QtCore.Qt.NoBrush)
            for mask, color_name, alpha, linewidth, dashed in styles:
                color = QtGui.QColor(color_name)
                color.setAlphaF(alpha)
                pen = QtGui.QPen(color, linewidth)
                if dashed:
                    pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(pen)
                for full_x, full_y in payload.full_normalized_centers_xy[mask]:
                    x = (float(full_x) - left) / (right - left)
                    y = (float(full_y) - top) / (bottom - top)
                    painter.drawEllipse(
                        QtCore.QRectF(
                            target.x() + x * target.width() - ring_width / 2.0,
                            target.y() + y * target.height() - ring_height / 2.0,
                            ring_width,
                            ring_height,
                        )
                    )
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
                and _image_payload(self._selector_hold) is payload
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
                        self._image_interaction_armed()
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
                        self._image_interaction_armed()
                        and self._rectangle_fully_visible(
                            self._selector_draft_bounds
                        )
                    ),
                    endpoint_bounds=self._selector_draft_bounds,
                )
        hold = self._selector_hold
        held_image_payload = None if hold is None else _image_payload(hold)
        if (
            held_image_payload is not None
            and self._clim_candidate is not None
        ):
            self._paint_clim_candidate_label(
                painter,
                held_image_payload,
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

    def _paint_numeric_overlays(self, painter: QtGui.QPainter) -> None:
        for binding in self._numeric_bindings.values():
            self._paint_numeric_binding_overlay(painter, binding)

    def _paint_numeric_binding_overlay(
        self,
        painter: QtGui.QPainter,
        binding: _NumericPanelBinding,
    ) -> None:
        target = self._numeric_target(binding)
        if target is None:
            return
        plot, payload, bounds = target.plot, target.payload, target.bounds
        viewport = payload.viewport
        x_unit_value = (
            viewport.x_axis.unit
            if isinstance(viewport, CurveViewportTransform)
            else payload.value_unit
        )
        x_unit = "" if x_unit_value is None else f" {x_unit_value}"
        y_unit = (
            ""
            if isinstance(payload, HistogramPanelPayload)
            else "" if payload.value_unit is None else f" {payload.value_unit}"
        )

        def widget_point(x: float, y: float) -> QtCore.QPointF:
            normalized = viewport.data_to_widget_normalized(x, y)
            return QtCore.QPointF(
                bounds.x() + normalized[0] * bounds.width(),
                bounds.y() + normalized[1] * bounds.height(),
            )

        painter.save()
        try:
            painter.setClipRect(plot)
            span = binding.span_candidate or binding.applied_span
            if span is not None:
                y_low = (
                    viewport.y_limits[0]
                    if isinstance(viewport, CurveViewportTransform)
                    else viewport.count_limits[0]
                )
                left = widget_point(span[0], y_low).x()
                right = widget_point(span[1], y_low).x()
                rectangle = QtCore.QRectF(
                    min(left, right),
                    plot.top(),
                    abs(right - left),
                    plot.height(),
                ).intersected(plot)
                painter.fillRect(rectangle, QtGui.QColor(245, 166, 35, 38))
                pen = QtGui.QPen(QtGui.QColor(ORANGE), 1.5)
                pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(rectangle)

            cross = binding.cross
            if cross is not None:
                point = widget_point(cross.x, cross.y)
                if plot.contains(point):
                    painter.setPen(QtGui.QPen(QtGui.QColor(GREEN), 1.5))
                    painter.drawLine(
                        QtCore.QPointF(point.x(), plot.top()),
                        QtCore.QPointF(point.x(), plot.bottom()),
                    )
                    painter.drawLine(
                        QtCore.QPointF(plot.left(), point.y()),
                        QtCore.QPointF(plot.right(), point.y()),
                    )
                self._paint_curve_label(
                    painter,
                    (
                        f"x={cross.x:.6g}{x_unit}  count={cross.y:.6g}"
                        if isinstance(payload, HistogramPanelPayload)
                        else f"x={cross.x:.6g}{x_unit}  y={cross.y:.6g}{y_unit}"
                    ),
                    plot,
                    QtGui.QColor(GREEN),
                    top_right=True,
                )

            sample = binding.hover
            position = binding.hover_position
            if sample is not None and position is not None:
                point = None
                try:
                    point = widget_point(sample.x, sample.y)
                except ValueError:
                    pass
                if point is not None and plot.contains(point):
                    painter.setPen(QtGui.QPen(QtGui.QColor(ORANGE), 1.5))
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(ORANGE)))
                    painter.drawEllipse(point, 3.5, 3.5)
                label = (
                    (
                        f"{sample.series_label}  "
                        f"[{sample.left:.6g}, {sample.right:.6g}"
                        f"{']' if sample.right_closed else ')'}{x_unit}  "
                        f"count={sample.count}"
                    )
                    if isinstance(sample, _HistogramBinSample)
                    else (
                        f"{sample.series_label}  x={sample.x:.6g}{x_unit}  "
                        f"y={sample.y:.6g}{y_unit}"
                    )
                )
                self._paint_curve_label(
                    painter,
                    label,
                    plot,
                    QtGui.QColor(ORANGE),
                    anchor=position,
                )
        finally:
            painter.restore()

    @staticmethod
    def _paint_curve_label(
        painter: QtGui.QPainter,
        label: str,
        plot: QtCore.QRectF,
        color: QtGui.QColor,
        *,
        anchor: QtCore.QPointF | None = None,
        top_right: bool = False,
    ) -> None:
        metrics = painter.fontMetrics()
        label_bounds = metrics.boundingRect(label).adjusted(-5, -2, 5, 2)
        if top_right:
            label_bounds.moveTopRight(plot.topRight().toPoint() + QtCore.QPoint(-5, 5))
        else:
            if anchor is None:
                anchor = plot.topLeft()
            x = min(int(plot.right()) - label_bounds.width(), int(anchor.x()) + 12)
            y = min(int(plot.bottom()) - label_bounds.height(), int(anchor.y()) + 12)
            label_bounds.moveTopLeft(
                QtCore.QPoint(max(int(plot.left()), x), max(int(plot.top()), y))
            )
        painter.fillRect(label_bounds, QtGui.QColor(0, 0, 0, 190))
        painter.setPen(color)
        painter.drawText(label_bounds, QtCore.Qt.AlignCenter, label)

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
        label = (
            f"({sample.x_coordinate}, {sample.y_coordinate}){suffix}"
            if self.visible_site_map_payload() is not None
            else f"({sample.x_coordinate}, {sample.y_coordinate}, {value}){suffix}"
        )
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
        site_map = self.visible_site_map_payload()
        if site_map is None:
            label = (
                f"x={sample.x_coordinate}  y={sample.y_coordinate}  "
                f"z={self._formatted_sample_value(sample)}"
            )
        else:
            point = np.asarray(
                (float(sample.x_coordinate), float(sample.y_coordinate)),
                dtype=np.float64,
            )
            distances = np.sum(np.square(site_map.centers_xy - point), axis=1)
            site_index = int(np.argmin(distances))
            state = (
                "invalid"
                if not site_map.site_validity[site_index]
                else "occupied"
                if site_map.occupied[site_index]
                else "empty"
            )
            site_label = site_map.site_axis.coordinate_at(site_index)
            label = (
                f"x={sample.x_coordinate}  y={sample.y_coordinate}  "
                f"z={self._formatted_sample_value(sample)}  "
                f"nearest={site_label} ({state})"
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

    def _cancel_image_gesture(self, *, clear_draft: bool) -> None:
        self._selector_drag_anchor = None
        self._pan_anchor = None
        self._pan_origin = None
        self._pan_target_size = None
        self._pan_candidate = None
        self._clim_drag = None
        self._clim_origin_limits = None
        self._clim_candidate = None
        self._clim_domain = None
        if (
            self._selector_hold is not None
            and self._selector_hold.panel_id == self._selector_panel_id
        ):
            self._selector_hold = None
        if clear_draft:
            self._selector_draft_bounds = None

    def _cancel_numeric_gesture(
        self,
        binding: _NumericPanelBinding,
        *,
        clear_span: bool,
    ) -> None:
        binding.span_anchor = None
        binding.pan_anchor = None
        binding.pan_origin = None
        binding.pan_candidate = None
        if (
            self._selector_hold is not None
            and self._selector_hold.panel_id == binding.panel_id
        ):
            self._selector_hold = None
        if clear_span:
            binding.span_candidate = None

    def _clear_numeric_transient(
        self,
        binding: _NumericPanelBinding,
        *,
        clear_applied_span: bool,
        clear_pending: bool,
    ) -> None:
        self._cancel_numeric_gesture(binding, clear_span=True)
        if clear_applied_span:
            binding.applied_span = None
        if clear_pending:
            binding.pending_viewport = None
            binding.pending_origin = None
        binding.cross = None
        self._set_numeric_hover(binding, None)

    def _cancel_active_gesture(
        self,
        *,
        clear_image_draft: bool,
        clear_numeric_spans: bool,
    ) -> None:
        self._cancel_image_gesture(clear_draft=clear_image_draft)
        for binding in self._numeric_bindings.values():
            self._cancel_numeric_gesture(
                binding,
                clear_span=clear_numeric_spans,
            )
        self._selector_hold = None

    def _reset_rectangle_selector(self) -> None:
        self._cancel_image_gesture(clear_draft=True)
        self._selector_applied_bounds = None
        self._pending_viewport = None
        self._pending_color_limits = None
        self._pending_origin = None
        self._set_cross_sample(None)
        self._set_hover_sample(None)
        self._image_binding_enabled = False
        self._image_interaction_ready = False
        self._selector_panel_id = None
        self._selector_viewport = None
        self._selector_callback = None
        self._interaction_callback = None
        if not self._numeric_bindings:
            self._selector_enabled = False

    def _reset_numeric_binding(self, panel_id: str) -> None:
        binding = self._numeric_bindings.get(panel_id)
        if binding is None:
            return
        self._clear_numeric_transient(
            binding,
            clear_applied_span=True,
            clear_pending=True,
        )
        binding.binding_enabled = False
        binding.interaction_ready = False
        del self._numeric_bindings[panel_id]
        if self._selector_panel_id is None and not self._numeric_bindings:
            self._selector_enabled = False

    def _reset_all_numeric_bindings(self) -> None:
        for panel_id in tuple(self._numeric_bindings):
            self._reset_numeric_binding(panel_id)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtRasterBoard presentation is GUI-thread affine")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("QtRasterBoard is closed")


__all__ = ["QtImageBoard", "QtOwnerWake", "QtRasterBoard"]
