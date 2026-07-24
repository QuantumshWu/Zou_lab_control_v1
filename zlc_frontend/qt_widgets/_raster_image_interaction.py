"""Image-family gesture state and pure interaction geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from PyQt5 import QtCore, QtGui

from ..render import (
    BoardFrame,
    ImagePanelPayload,
    SiteMapPanelPayload,
    detached_render_fault,
)
from ..render_style import colormap_argb_at
from ..selector import (
    ImageInteractionCommit,
    ImageColorLimitsCommit,
    ImageViewportTransform,
    ImageViewportCommit,
    NormalizedRectangle,
    PanelInteractionOrigin,
    RectangleGesture,
)
from ._raster_front import (
    _HeldPanelFront,
    _image_payload,
    _hold_matches_frame,
    _panel_bounds,
    _panel_image_geometry,
    _panel_presentation,
    _panel_semantics_changed,
    _raster_geometry,
    _site_map_payload,
    _visible_display,
)
from ._rectangle_selector import (
    RectangleDrag,
    paint_cross_selector,
    paint_rectangle_selector,
    paint_selector_text,
    selector_pen_color,
)
from .style import ORANGE


@dataclass(frozen=True, slots=True)
class _ImageSample:
    """Exact painted sample used only by one board's visual overlays."""

    x_index: int
    y_index: int
    x_coordinate: object
    y_coordinate: object
    value: object
    valid: bool


@dataclass(slots=True)
class _ImagePanelBinding:
    """The sole mutable state for one bound image-family panel."""

    panel_id: str
    viewport: ImageViewportTransform
    selection_callback: Callable[[RectangleGesture], object]
    interaction_callback: Callable[[ImageInteractionCommit], object] | None = None
    revision_floor: int = 0
    binding_enabled: bool = True
    interaction_ready: bool = False
    applied_bounds: NormalizedRectangle | None = None
    draft_bounds: NormalizedRectangle | None = None
    rectangle_drag: RectangleDrag | None = None
    drag_prior_draft: NormalizedRectangle | None = None
    drag_start_bounds: NormalizedRectangle | None = None
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
    fault: RuntimeError | None = None


def _image_interaction_is_pending(binding: _ImagePanelBinding) -> bool:
    return (
        binding.pending_viewport is not None
        or binding.pending_color_limits is not None
    )


def _image_interaction_armed(
    selector_enabled: bool,
    binding: _ImagePanelBinding,
) -> bool:
    return (
        selector_enabled
        and binding.binding_enabled
        and binding.interaction_ready
        and binding.fault is None
    )


def _validate_selector_binding(
    panel_id: str,
    viewport: ImageViewportTransform,
    frame: BoardFrame,
    *,
    panel_ids: tuple[str, ...],
) -> None:
    index = panel_ids.index(panel_id)
    panel = frame.panels[index]
    if panel.panel_id != panel_id:
        raise ValueError("selector panel identity changed")
    payload = _image_payload(panel)
    if payload is not None:
        if payload.viewport != viewport:
            raise ValueError(
                "selector viewport differs from the exact image payload viewport"
            )
        return
    raster = panel.raster
    expected_height, expected_width = viewport.raster_shape
    if raster.width != expected_width or raster.height != expected_height:
        raise ValueError(
            "selector viewport axes do not match the selected raw raster geometry"
        )


def _viewport_for_presented_panel(
    binding: _ImagePanelBinding,
    frame: BoardFrame,
    *,
    panel_ids: tuple[str, ...],
    previous: tuple[
        BoardFrame,
        tuple[tuple[bytes, QtGui.QImage], ...],
    ] | None,
    previous_panel_ids: tuple[str, ...],
) -> ImageViewportTransform:
    panel_id = binding.panel_id
    current = binding.viewport
    panel = frame.panels[panel_ids.index(panel_id)]
    payload = _image_payload(panel)
    if payload is None:
        return current
    candidate = payload.viewport
    structurally_new = previous is None or panel_id not in previous_panel_ids
    if not structurally_new and previous is not None:
        old_panel = previous[0].panels[previous_panel_ids.index(panel_id)]
        structurally_new = _panel_semantics_changed(old_panel, panel)
    if structurally_new:
        return candidate
    if candidate.axes != current.axes:
        raise ValueError("image viewport axes changed without panel structure change")
    if candidate.viewport_revision < current.viewport_revision:
        raise ValueError("stale image viewport revision cannot replace the visible front")
    pending = binding.pending_viewport
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


def _selector_target(
    binding: _ImagePanelBinding | None,
    *,
    widget_rect: QtCore.QRect,
    panel_ids: tuple[str, ...],
    columns: int,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    hold: _HeldPanelFront | None,
):
    if front is None or binding is None:
        return None
    panel_id = binding.panel_id
    if panel_id not in panel_ids:
        return None
    index = panel_ids.index(panel_id)
    prepared = (
        hold.prepared
        if hold is not None and hold.panel_id == panel_id
        else front[1][index]
    )
    image = prepared[1]
    bounds = _panel_bounds(
        widget_rect,
        index=index,
        count=len(front[1]),
        columns=columns,
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
    geometry = _panel_image_geometry(
        bounds,
        image,
        payload,
        site_map_payload=composite,
    )
    return geometry.target, front[0], front[0].panels[index], prepared


def _image_target_at(
    point: QtCore.QPointF,
    *,
    include_rail: bool,
    widget_rect: QtCore.QRect,
    panel_ids: tuple[str, ...],
    columns: int,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    hold: _HeldPanelFront | None,
    bindings: dict[str, _ImagePanelBinding],
):
    for binding in bindings.values():
        target = _selector_target(
            binding,
            widget_rect=widget_rect,
            panel_ids=panel_ids,
            columns=columns,
            front=front,
            hold=hold,
        )
        if target is None:
            continue
        rail_target = _clim_rail_target(
            binding,
            widget_rect=widget_rect,
            panel_ids=panel_ids,
            columns=columns,
            front=front,
            hold=hold,
        )
        integer_point = point.toPoint()
        if target[0].contains(integer_point) or (
            include_rail
            and rail_target is not None
            and rail_target[0].contains(integer_point)
        ):
            return binding, target, rail_target
    return None


def _painted_image_panel_id_at(
    point: QtCore.QPointF,
    *,
    widget_rect: QtCore.QRect,
    columns: int,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
) -> str | None:
    if front is None:
        return None
    for index, panel in enumerate(front[0].panels):
        payload = panel.display_payload
        if not isinstance(payload, (ImagePanelPayload, SiteMapPanelPayload)):
            continue
        bounds = _panel_bounds(
            widget_rect,
            index=index,
            count=len(front[0].panels),
            columns=columns,
        )
        image = front[1][index][1]
        image_payload = (
            payload.background
            if isinstance(payload, SiteMapPanelPayload)
            else payload
        )
        geometry = _panel_image_geometry(
            bounds,
            image,
            image_payload,
            site_map_payload=(
                payload if isinstance(payload, SiteMapPanelPayload) else None
            ),
        )
        if geometry.target.contains(point.toPoint()):
            return panel.panel_id
    return None


def _clim_rail_target(
    binding: _ImagePanelBinding,
    *,
    widget_rect: QtCore.QRect,
    panel_ids: tuple[str, ...],
    columns: int,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    hold: _HeldPanelFront | None,
):
    panel_id = binding.panel_id
    if front is None or panel_id not in panel_ids:
        return None
    index = panel_ids.index(panel_id)
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
        widget_rect,
        index=index,
        count=len(front[1]),
        columns=columns,
    )
    geometry = _panel_image_geometry(
        bounds,
        prepared[1],
        payload,
        site_map_payload=composite,
    )
    if geometry.distribution is None:
        return None
    return geometry.distribution, front[0], panel, prepared, payload


def _viewport_for_target(
    binding: _ImagePanelBinding,
    target,
    hold: _HeldPanelFront | None,
) -> ImageViewportTransform:
    # Wheel commits are latest-intent, not render-paced.  While Agg is still
    # answering an earlier wheel step, the next step must be composed from the
    # already-authored viewport rather than replaying the still-painted one.
    # The exact painted front remains the CAS origin; only the display intent
    # accumulates here.
    if binding.pending_viewport is not None:
        return binding.pending_viewport
    if hold is not None and hold.panel_id == target[2].panel_id:
        payload = _image_payload(hold)
        if payload is not None:
            return payload.viewport
    payload = _image_payload(target[2])
    if payload is not None:
        return payload.viewport
    return binding.viewport


def _sample_for_target(
    target,
    point: QtCore.QPointF,
    *,
    hold: _HeldPanelFront | None,
) -> _ImageSample | None:
    image_target, _frame, panel = target[0], target[1], target[2]
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
    normalized = _normalized_point(point, image_target, clamp=False)
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
    x_coordinate, y_coordinate = viewport.coordinate_for_visible_point(normalized)
    return _ImageSample(
        x_index=x_index,
        y_index=y_index,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
        value=value,
        valid=bool(valid) and finite_value,
    )


def _held_panel_from_target(target) -> _HeldPanelFront:
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
            target[4] if len(target) > 4 else panel.display_payload
        ),
    )


def _commit_viewport(
    binding: _ImagePanelBinding,
    candidate: ImageViewportTransform,
    *,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    panel_ids: tuple[str, ...],
    hold: _HeldPanelFront | None,
    painted_hold: _HeldPanelFront | None,
) -> bool:
    current = binding.viewport
    authored = binding.pending_viewport or current
    if candidate.axes != current.axes:
        raise ValueError("viewport commit cannot change image axes")
    # A gesture candidate carries desired BOUNDS; this owner assigns its
    # revision.  Live pan deliberately keeps calculating those bounds from the
    # press-time transform, so an intervening worker answer can make the
    # candidate's inherited revision older than the currently authored view.
    # Equality must therefore be about the authored bounds, not the candidate
    # dataclass (whose stale revision is not a second piece of intent).
    if candidate.visible_bounds == authored.visible_bounds:
        return False
    # Compare with the latest authored view, not merely the still-painted
    # worker answer.  Otherwise wheel-in followed immediately by wheel-out
    # would be mistaken for a no-op when it reaches the painted home bounds,
    # leaving the earlier zoom pending and making one wheel step appear lost.
    base_revision = max(
        current.viewport_revision,
        authored.viewport_revision,
        binding.revision_floor,
    )
    # Rebase from ``authored``.  Calling ``candidate.with_visible_bounds`` with
    # candidate's own bounds is a no-op by contract and used to leave its stale
    # revision unchanged after a render answer landed mid-gesture.
    candidate = authored.with_visible_bounds(
        candidate.visible_bounds,
        viewport_revision=base_revision + 1,
    )
    if candidate.viewport_revision <= current.viewport_revision:
        raise ValueError("viewport commit revision must increase")
    if front is None:
        return False
    if hold is not None and not _hold_matches_frame(
        hold,
        front[0],
        panel_ids=panel_ids,
    ):
        return False
    callback = binding.interaction_callback
    if callback is None:
        return False
    _payload, origin = _visible_display(
        binding.panel_id,
        (ImagePanelPayload, SiteMapPanelPayload),
        front=front,
        panel_ids=panel_ids,
        hold=painted_hold,
    )
    if origin is None:
        raise RuntimeError("image interaction origin has no exact payload")
    command = ImageViewportCommit(origin, candidate)
    binding.revision_floor = max(
        binding.revision_floor,
        candidate.viewport_revision,
    )
    binding.pending_viewport = candidate
    binding.pending_origin = origin
    try:
        callback(command)
    except BaseException as error:
        binding.pending_viewport = None
        binding.pending_origin = None
        if binding.fault is None:
            binding.fault = detached_render_fault(error)
        binding.binding_enabled = False
        return False
    return True


def _commit_color_limits(
    binding: _ImagePanelBinding,
    limits: tuple[float, float],
    *,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    panel_ids: tuple[str, ...],
    hold: _HeldPanelFront,
    painted_hold: _HeldPanelFront | None,
) -> bool:
    payload = _image_payload(hold)
    if payload is None or limits == payload.color_limits:
        return False
    if front is None or not _hold_matches_frame(
        hold,
        front[0],
        panel_ids=panel_ids,
    ):
        return False
    callback = binding.interaction_callback
    if callback is None:
        return False
    _visible_payload, origin = _visible_display(
        binding.panel_id,
        (ImagePanelPayload, SiteMapPanelPayload),
        front=front,
        panel_ids=panel_ids,
        hold=painted_hold,
    )
    if origin is None:
        raise RuntimeError("image interaction origin has no exact payload")
    command = ImageColorLimitsCommit(origin, limits)
    binding.pending_color_limits = command.color_limits
    binding.pending_origin = origin
    try:
        callback(command)
    except BaseException as error:
        binding.pending_color_limits = None
        binding.pending_origin = None
        if binding.fault is None:
            binding.fault = detached_render_fault(error)
        binding.binding_enabled = False
        return False
    return True


def _set_cross_sample(
    binding: _ImagePanelBinding,
    sample: _ImageSample | None,
) -> None:
    if sample is binding.cross:
        return
    binding.cross = sample


def _active_image_binding(
    bindings: dict[str, _ImagePanelBinding],
    hold,
) -> _ImagePanelBinding | None:
    return None if hold is None else bindings.get(hold.panel_id)


def _color_rail_domain(payload: ImagePanelPayload) -> tuple[float, float]:
    return payload.color_limits


def _rail_y(
    value: float,
    domain: tuple[float, float],
    rail: QtCore.QRect,
) -> float:
    low, high = domain
    fraction = (value - low) / (high - low)
    return rail.bottom() - min(1.0, max(0.0, fraction)) * rail.height()


def _rail_value(
    y: float,
    domain: tuple[float, float],
    rail: QtCore.QRect,
) -> float:
    fraction = (rail.bottom() - y) / max(1, rail.height())
    low, high = domain
    return low + min(1.0, max(0.0, fraction)) * (high - low)


def _clim_handle_at(
    point: QtCore.QPoint,
    rail: QtCore.QRect,
    payload: ImagePanelPayload,
) -> str | None:
    domain = _color_rail_domain(payload)
    candidates = (
        (abs(point.y() - _rail_y(payload.color_limits[0], domain, rail)), "low"),
        (abs(point.y() - _rail_y(payload.color_limits[1], domain, rail)), "high"),
    )
    distance, handle = min(candidates)
    return handle if distance <= 7.0 else None


def _color_rail_argb(payload: ImagePanelPayload, value: float) -> int:
    """Map one physical rail value through the painted image's limits."""

    low, high = payload.color_limits
    return colormap_argb_at(
        payload.colormap,
        (float(value) - low) / (high - low),
    )


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


def _image_bounds_for_rectangle_drag(
    viewport: ImageViewportTransform,
    visible_bounds: NormalizedRectangle,
) -> NormalizedRectangle | None:
    """Resolve one visible-window drag through its exact painted viewport."""

    return viewport.clipped_full_bounds_for_visible_bounds(
        visible_bounds
    )


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


def _rectangle_fully_visible(
    viewport: ImageViewportTransform,
    bounds: NormalizedRectangle,
) -> bool:
    try:
        viewport.visible_bounds_for_full_bounds(bounds)
    except ValueError:
        return False
    return True


def _visible_point_for_sample(
    viewport: ImageViewportTransform,
    sample: _ImageSample,
) -> tuple[float, float] | None:
    point = viewport.unbounded_visible_point_for_coordinate(
        (sample.x_coordinate, sample.y_coordinate),
        coordinate_frame=viewport.coordinate_frame,
    )
    return point if 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0 else None


def _clim_candidate_label(
    binding: _ImagePanelBinding,
    payload: ImagePanelPayload,
) -> str:
    limits = binding.clim_candidate
    if limits is None:
        raise RuntimeError("H candidate label requires an active limit draft")
    low, high = _color_rail_domain(payload)
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


def _formatted_sample_value(sample: _ImageSample) -> str:
    if not sample.valid:
        return "invalid"
    value = sample.value
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _selection_endpoint_label(
    viewport: ImageViewportTransform,
    bounds: NormalizedRectangle,
) -> str:
    selected_x_low, selected_y_low, selected_x_high, selected_y_high = (
        viewport.coordinate_rectangle_for_full_bounds(bounds)
    )
    # Label precision follows the physical window under the pointer.  Do not
    # feed ``visible_bounds`` back into a normalized-selection API: those
    # source-relative values are intentionally unbounded after a legal pan.
    visible_x_low, visible_x_high = viewport.x_limits
    visible_y_low, visible_y_high = viewport.y_limits

    def precision(span: float) -> int:
        gap = abs(float(span)) / 1000.0 if span else 0.01
        return max(0, -int(math.ceil(math.log10(gap))))

    x_precision = precision(visible_x_high - visible_x_low)
    y_precision = precision(visible_y_high - visible_y_low)
    return (
        f"({selected_x_low:.{x_precision}f}, "
        f"{selected_y_low:.{y_precision}f})\n"
        f"({selected_x_high:.{x_precision}f}, "
        f"{selected_y_high:.{y_precision}f})"
    )


def _visible_site_map_payload(
    panel_id: str,
    *,
    panel_ids: tuple[str, ...],
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    hold: _HeldPanelFront | None,
) -> SiteMapPanelPayload | None:
    if hold is not None and hold.panel_id == panel_id:
        return _site_map_payload(hold)
    if front is None or panel_id not in panel_ids:
        return None
    return _site_map_payload(front[0].panels[panel_ids.index(panel_id)])


def _paint_clim_draft_lines(
    painter: QtGui.QPainter,
    payload: ImagePanelPayload,
    distribution: QtCore.QRect,
    binding: _ImagePanelBinding | None,
    *,
    hold: _HeldPanelFront | None,
) -> None:
    """Overlay only an in-flight H drag; base band and chrome are Agg-owned."""

    if (
        binding is None
        or hold is None
        or _image_payload(hold) is not payload
        or binding.clim_candidate is None
    ):
        return
    painter.save()
    try:
        domain = _color_rail_domain(payload)
        for value, cmap_value in zip(
            binding.clim_candidate,
            payload.color_limits,
            strict=True,
        ):
            y = _rail_y(value, domain, distribution)
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor.fromRgba(
                        _color_rail_argb(payload, cmap_value)
                    ),
                    1.0,
                )
            )
            painter.drawLine(
                QtCore.QPointF(distribution.left(), y),
                QtCore.QPointF(distribution.right(), y),
            )
    finally:
        painter.restore()


def _paint_cross_sample(
    painter: QtGui.QPainter,
    viewport: ImageViewportTransform,
    sample: _ImageSample,
    target: QtCore.QRect,
    *,
    site_map: SiteMapPanelPayload | None,
    color: QtGui.QColor | str | None = None,
) -> None:
    visible = _visible_point_for_sample(viewport, sample)
    point = None if visible is None else QtCore.QPointF(
        target.x() + visible[0] * target.width(),
        target.y() + visible[1] * target.height(),
    )
    value = _formatted_sample_value(sample)
    label = (
        f"({sample.x_coordinate:g}, {sample.y_coordinate:g})"
        if site_map is not None
        else f"({sample.x_coordinate:g}, {sample.y_coordinate:g}, {value})"
    )
    paint_cross_selector(
        painter,
        QtCore.QRectF(target),
        point,
        label,
        color=color,
    )


def _paint_clim_candidate_label(
    painter: QtGui.QPainter,
    binding: _ImagePanelBinding,
    payload: ImagePanelPayload,
    target: QtCore.QRect,
) -> None:
    label = _clim_candidate_label(binding, payload)
    paint_selector_text(
        painter,
        label,
        QtCore.QRectF(target),
        QtGui.QColor(ORANGE),
        corner="top_left",
    )


def _paint_image_overlays(
    painter: QtGui.QPainter,
    *,
    selector_enabled: bool,
    widget_rect: QtCore.QRect,
    panel_ids: tuple[str, ...],
    columns: int,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    hold: _HeldPanelFront | None,
    bindings: dict[str, _ImagePanelBinding],
) -> None:
    for binding in bindings.values():
        target = _selector_target(
            binding,
            widget_rect=widget_rect,
            panel_ids=panel_ids,
            columns=columns,
            front=front,
            hold=hold,
        )
        if target is None:
            continue
        image_target = target[0]
        image_payload = (
            _image_payload(hold)
            if hold is not None and hold.panel_id == binding.panel_id
            else _image_payload(target[2])
        )
        # Overlay coordinates and the data box are one painted fact.  A
        # pending viewport is only the newest authored worker intent; using it
        # here would move Area/Cross over the still-old raster (and over the
        # old equal-aspect bbox) before that viewport's raster+geometry was
        # admitted.  Keep every visible overlay on the exact payload under it;
        # rapid wheel/pan may still accumulate through ``_viewport_for_target``.
        viewport = (
            binding.viewport
            if image_payload is None
            else image_payload.viewport
        )
        selector_color = (
            None
            if image_payload is None
            else QtGui.QColor.fromRgba(
                colormap_argb_at(image_payload.colormap, 0.95)
            )
        )
        site_map = _visible_site_map_payload(
            binding.panel_id,
            panel_ids=panel_ids,
            front=front,
            hold=hold,
        )
        painter.save()
        painter.setClipRect(image_target)
        selected_bounds = binding.draft_bounds or binding.applied_bounds
        if selected_bounds is not None:
            visible = viewport.clipped_visible_bounds_for_full_bounds(
                selected_bounds
            )
            if visible is not None:
                rectangle = _overlay_rect(visible, image_target)
                paint_rectangle_selector(
                    painter,
                    handles=(
                        _image_interaction_armed(selector_enabled, binding)
                        and _rectangle_fully_visible(viewport, selected_bounds)
                    ),
                    rectangle=rectangle,
                    color=selector_color,
                )
                if binding.rectangle_drag is None:
                    paint_selector_text(
                        painter,
                        _selection_endpoint_label(viewport, selected_bounds),
                        QtCore.QRectF(image_target),
                        selector_pen_color(selector_color),
                        corner="top_left",
                    )
        held_image_payload = None if hold is None else _image_payload(hold)
        if (
            held_image_payload is not None
            and hold is not None
            and hold.panel_id == binding.panel_id
            and binding.clim_candidate is not None
        ):
            _paint_clim_candidate_label(
                painter,
                binding,
                held_image_payload,
                image_target,
            )
        if binding.cross is not None:
            _paint_cross_sample(
                painter,
                viewport,
                binding.cross,
                image_target,
                site_map=site_map,
                color=selector_color,
            )
        painter.restore()


def _cancel_image_gesture(
    binding: _ImagePanelBinding,
    *,
    clear_draft: bool,
) -> None:
    binding.rectangle_drag = None
    binding.drag_prior_draft = None
    binding.drag_start_bounds = None
    binding.pan_anchor = None
    binding.pan_origin = None
    binding.pan_target_size = None
    binding.pan_candidate = None
    binding.clim_drag = None
    binding.clim_origin_limits = None
    binding.clim_candidate = None
    binding.clim_domain = None
    if clear_draft:
        binding.draft_bounds = None


def _clear_image_transient(
    binding: _ImagePanelBinding,
    *,
    clear_applied_bounds: bool,
    clear_pending: bool,
) -> None:
    _cancel_image_gesture(binding, clear_draft=True)
    if clear_applied_bounds:
        binding.applied_bounds = None
    if clear_pending:
        binding.pending_viewport = None
        binding.pending_color_limits = None
        binding.pending_origin = None
    binding.cross = None
