"""Internal geometry and interaction state used by :class:`QtRasterBoard`."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Literal, TypeAlias

from PyQt5 import QtCore, QtGui

from zlc_storage import canonical_text

from ..curve_display import CurveViewportTransform, NumericViewportTransform
from ..histogram_display import HistogramViewportTransform
from ..render import (
    BoardFrame,
    CurvePanelPayload,
    DocumentInputIdentity,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    PixelFormat,
    PulsePanelPayload,
    SiteMapPanelPayload,
    SourceIdentity,
)
from ..selector import (
    CurveInteractionIntent,
    HistogramInteractionIntent,
    ImageInteractionCommit,
    ImageViewportTransform,
    NormalizedRectangle,
    PanelInteractionOrigin,
    RectangleGesture,
)
from .style import SELECTOR_ALPHA, SELECTOR_COLOR

def _selector_pen_color() -> QtGui.QColor:
    color = QtGui.QColor(SELECTOR_COLOR)
    color.setAlpha(SELECTOR_ALPHA)
    return color


def _selector_precision(span: float) -> int:
    """Decimal places for a selector coordinate label, the reference's rule
    exactly: enough digits to resolve 1/1000 of the visible span."""

    gap = abs(span) / 1000 if span else 0.01
    return max(0, -int(math.ceil(math.log10(gap))))


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
    fit_aspect = None
    if payload.fit_overlay is not None:
        unit_span = payload.viewport.visible_span_for_coordinate_span(
            (1.0, 1.0),
            coordinate_frame=payload.fit_overlay.coordinate_frame,
        )
        fit_aspect = unit_span[1] / unit_span[0]
    target = _aspect_target_for_source(
        image_bounds,
        source,
        aspect_ratio=(
            site_map_payload.visible_coordinate_aspect_ratio
            if site_map_payload is not None
            else fit_aspect
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
        | MeterPanelPayload
        | PulsePanelPayload
        | SiteMapPanelPayload
    ),
):
    if isinstance(payload, SiteMapPanelPayload):
        return payload.occupancy_input
    if isinstance(payload, PulsePanelPayload):
        return payload.document_input
    return payload.evaluated_input


def _input_structure(evaluated_input) -> tuple[object, object, object, str]:
    """Return producer structure while deliberately excluding normal revisions."""

    ref = evaluated_input.ref
    return (
        evaluated_input.dataset_id,
        ref.block_id,
        ref.stream_generation,
        ref.schema_fingerprint,
    )


_NumericKind: TypeAlias = Literal["curve", "histogram", "pulse"]
_NumericPayload: TypeAlias = (
    CurvePanelPayload | HistogramPanelPayload | PulsePanelPayload
)
_NumericViewport: TypeAlias = (
    CurveViewportTransform | NumericViewportTransform | HistogramViewportTransform
)
_NumericIntent: TypeAlias = CurveInteractionIntent | HistogramInteractionIntent

#: The ONE kind->payload map for the numeric interaction family.  PULSE is a
#: member because the pulse timeline is an x-only interactive surface: it
#: reuses the CURVE viewport transform and the CURVE intent vocabulary, so the
#: only fact that distinguishes it here is which payload type its front carries.
_NUMERIC_PAYLOAD_TYPES: dict[_NumericKind, type] = {
    "curve": CurvePanelPayload,
    "histogram": HistogramPanelPayload,
    "pulse": PulsePanelPayload,
}


def _numeric_payload(
    panel_or_hold: PanelFrame | _HeldPanelFront,
    kind: _NumericKind,
) -> _NumericPayload | None:
    payload = panel_or_hold.display_payload
    return payload if isinstance(payload, _NUMERIC_PAYLOAD_TYPES[kind]) else None


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
    source_identity: SourceIdentity | DocumentInputIdentity
    presentation: PanelPresentationIdentity
    raster_geometry: tuple[int, int, int, PixelFormat]
    prepared: tuple[bytes, QtGui.QImage]
    display_payload: (
        ImagePanelPayload
        | CurvePanelPayload
        | HistogramPanelPayload
        | PulsePanelPayload
        | SiteMapPanelPayload
        | None
    )

    @property
    def gesture_identity(
        self,
    ) -> tuple[str, int, int, SourceIdentity | DocumentInputIdentity]:
        return (
            self.board_id,
            self.layout_generation,
            self.sequence,
            self.source_identity,
        )


def _presented_revision_state(
    pending_revision: int | None,
    candidate_revision: int | None,
    held_revision: int | None,
) -> tuple[bool, bool]:
    """Classify one worker answer as final or useful intermediate."""

    if pending_revision is None or candidate_revision is None:
        return (False, False)
    if candidate_revision >= pending_revision:
        return (True, False)
    return (
        False,
        held_revision is not None and candidate_revision > held_revision,
    )


def _advance_held_front(
    hold: _HeldPanelFront,
    frame: BoardFrame,
    panel: PanelFrame,
    prepared: tuple[bytes, QtGui.QImage],
) -> _HeldPanelFront:
    """Absorb one same-gesture raster answer into the painted hold."""

    return replace(
        hold,
        sequence=frame.sequence,
        source_identity=panel.source_identity,
        presentation=_panel_presentation(panel),
        raster_geometry=_raster_geometry(panel),
        prepared=prepared,
        display_payload=panel.display_payload,
    )


@dataclass(slots=True)
class _ImagePanelBinding:
    """The sole mutable owner for one bound image-family panel."""

    panel_id: str
    viewport: ImageViewportTransform
    selection_callback: Callable[[RectangleGesture], object]
    interaction_callback: Callable[[ImageInteractionCommit], object] | None = None
    revision_floor: int = 0
    binding_enabled: bool = True
    interaction_ready: bool = False
    applied_bounds: NormalizedRectangle | None = None
    draft_bounds: NormalizedRectangle | None = None
    drag_anchor: tuple[float, float] | None = None
    drag_prior_draft: NormalizedRectangle | None = None
    pan_anchor: QtCore.QPointF | None = None
    pan_origin: ImageViewportTransform | None = None
    pan_target_size: tuple[int, int] | None = None
    pan_candidate: ImageViewportTransform | None = None
    pending_viewport: ImageViewportTransform | None = None
    pending_color_limits: tuple[float, float] | None = None
    pending_origin: PanelInteractionOrigin | None = None
    clim_drag: str | None = None
    clim_origin_limits: tuple[float, float] | None = None
    clim_candidate: tuple[float, float] | None = None
    clim_domain: tuple[float, float] | None = None
    cross: _ImageSample | None = None
    hover: _ImageSample | None = None
    hover_position: QtCore.QPointF | None = None
    fault: RuntimeError | None = None


@dataclass(slots=True)
class _NumericPanelBinding:
    """The sole mutable owner for one bound numeric panel."""

    kind: _NumericKind
    panel_id: str
    callback: Callable[[_NumericIntent], object]
    viewport: _NumericViewport | None = None
    revision_floor: int = 0
    binding_enabled: bool = True
    interaction_ready: bool = False
    pending_viewport: _NumericViewport | None = None
    pending_origin: PanelInteractionOrigin | None = None
    applied_span: tuple[float, float] | None = None
    span_anchor: float | None = None
    span_candidate: tuple[float, float] | None = None
    # The DRAWN selection rectangle in normalized plot coordinates
    # (x0, y0, x1, y1): live during the drag, kept after release (the
    # reference's RectangleSelector leaves its box + label standing).
    span_rect: tuple[float, float, float, float] | None = None
    # Interactive-handle modes on a STANDING box, the reference's
    # RectangleSelector semantics: an edge grab resizes one dimension only
    # ('x' keeps the y ends frozen, 'y' keeps the x span frozen) and a center
    # grab ('span_move_grab' = pointer offset from the box origin) moves the
    # whole box.  Both are None during a plain corner drag / fresh pull.
    span_resize_lock: str | None = None
    span_move_grab: tuple[float, float] | None = None
    # A live threshold-line drag (histogram only): the grabbed index into the
    # payload's authored thresholds, the in-flight authored set, and the
    # display revision the last commit expects back -- present() absorbs the
    # answer into the hold when it arrives (the reference's DragVLine redraws
    # per motion).
    threshold_drag: int | None = None
    threshold_candidate: tuple[float, ...] | None = None
    threshold_pending_revision: int | None = None
    threshold_pending_origin: PanelInteractionOrigin | None = None
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
