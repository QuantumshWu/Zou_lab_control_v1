"""Curve, histogram, and pulse gesture state and pure interaction geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Literal, TypeAlias

import numpy as np
from PyQt5 import QtCore, QtGui

from ..curve_display import CurveViewportTransform, NumericViewportTransform
from ..display_range import RelimMode
from ..histogram_display import HistogramViewportTransform
from ..render import (
    BoardFrame,
    CurvePanelPayload,
    HistogramPanelPayload,
    PanelFrame,
    PulsePanelPayload,
    detached_render_fault,
)
from ..selector import (
    CurveInteractionIntent,
    CurveViewportCommit,
    HistogramInteractionIntent,
    HistogramThresholdCommit,
    HistogramViewportCommit,
    PanelInteractionOrigin,
)
from ._raster_front import (
    _HeldPanelFront,
    _panel_bounds,
    _panel_presentation,
    _panel_semantics_changed,
    _hold_matches_frame,
    _raster_geometry,
    _visible_display,
)
from ._rectangle_selector import (
    RectangleDrag,
    paint_rectangle_selector,
    paint_selector_hover_label,
    paint_selector_text,
    selector_pen_color,
    selector_precision,
)
from .style import ORANGE, SELECTOR_DOT_PX, SELECTOR_LINE_PX


_NumericKind: TypeAlias = Literal["curve", "histogram", "pulse"]
_NumericPayload: TypeAlias = (
    CurvePanelPayload | HistogramPanelPayload | PulsePanelPayload
)
_NumericViewport: TypeAlias = (
    CurveViewportTransform | NumericViewportTransform | HistogramViewportTransform
)
_NumericIntent: TypeAlias = CurveInteractionIntent | HistogramInteractionIntent

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


def _numeric_viewport_for_presented_panel(
    binding: _NumericPanelBinding,
    frame: BoardFrame,
    *,
    panel_ids: tuple[str, ...],
    previous: tuple[
        BoardFrame,
        tuple[tuple[bytes, QtGui.QImage], ...],
    ] | None,
    previous_panel_ids: tuple[str, ...],
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
    structurally_new = previous is None or binding.panel_id not in previous_panel_ids
    if not structurally_new and previous is not None:
        old_panel = previous[0].panels[
            previous_panel_ids.index(binding.panel_id)
        ]
        structurally_new = _panel_semantics_changed(old_panel, panel)
    if current is None or structurally_new:
        return candidate
    if type(candidate) is not type(current):
        raise ValueError("numeric viewport type changed without panel structure change")
    if (
        isinstance(candidate, NumericViewportTransform)
        and isinstance(current, NumericViewportTransform)
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
        if isinstance(candidate, NumericViewportTransform) and (
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
    return candidate


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


@dataclass(slots=True)
class _NumericPanelBinding:
    """The sole mutable state for one bound numeric panel."""

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
    span_candidate: tuple[float, float] | None = None
    span_rect: tuple[float, float, float, float] | None = None
    rectangle_drag: RectangleDrag | None = None
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


def _numeric_target(
    *,
    widget_rect: QtCore.QRect,
    panel_ids: tuple[str, ...],
    columns: int,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    hold: _HeldPanelFront | None,
    binding: _NumericPanelBinding,
) -> _NumericTarget | None:
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
        _numeric_payload(hold, binding.kind)
        if hold is not None and hold.panel_id == panel_id
        else _numeric_payload(panel, binding.kind)
    )
    if payload is None:
        return None
    bounds = _panel_bounds(
        widget_rect,
        index=index,
        count=len(front[1]),
        columns=columns,
    )
    plot = _numeric_plot_geometry(bounds, payload.viewport)
    return _NumericTarget(plot, front[0], panel, prepared, payload, bounds, binding)


def _numeric_target_at(
    point: QtCore.QPointF,
    *,
    widget_rect: QtCore.QRect,
    panel_ids: tuple[str, ...],
    columns: int,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    hold: _HeldPanelFront | None,
    bindings: dict[str, _NumericPanelBinding],
) -> _NumericTarget | None:
    for binding in bindings.values():
        target = _numeric_target(
            widget_rect=widget_rect,
            panel_ids=panel_ids,
            columns=columns,
            front=front,
            hold=hold,
            binding=binding,
        )
        if target is not None and target.plot.contains(point):
            return target
    return None


def _held_panel_from_numeric_target(
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


def _threshold_line_hit(
    target: _NumericTarget,
    pos: QtCore.QPointF,
) -> int | None:
    payload = target.payload
    if not isinstance(payload, HistogramPanelPayload):
        return None
    thresholds = payload.thresholds
    if not thresholds:
        return None
    viewport = payload.viewport
    counts_mid = 0.5 * (
        viewport.count_limits[0] + viewport.count_limits[1]
    )
    tolerance = 0.02 * target.plot.width()
    pressed_x = float(pos.x())
    best_index: int | None = None
    best_distance: float | None = None
    for index, threshold in enumerate(thresholds):
        normalized = viewport.data_to_widget_normalized(
            float(threshold), counts_mid
        )
        line_x = target.bounds.x() + normalized[0] * target.bounds.width()
        distance = abs(pressed_x - line_x)
        if distance <= tolerance and (
            best_distance is None or distance < best_distance
        ):
            best_index, best_distance = index, distance
    return best_index


def _span_rect_widget_extents(
    target: _NumericTarget,
    rect: tuple[float, float, float, float],
) -> tuple[list[float], list[float]]:
    viewport = target.payload.viewport
    bounds = target.bounds
    first = viewport.data_to_widget_normalized(rect[0], rect[1])
    second = viewport.data_to_widget_normalized(rect[2], rect[3])
    xs = sorted(
        (
            bounds.x() + first[0] * bounds.width(),
            bounds.x() + second[0] * bounds.width(),
        )
    )
    ys = sorted(
        (
            bounds.y() + first[1] * bounds.height(),
            bounds.y() + second[1] * bounds.height(),
        )
    )
    return xs, ys


def _span_data_candidate(
    first: float | None,
    second: float | None,
) -> tuple[float, float] | None:
    if first is None or second is None or first == second:
        return None
    low, high = sorted((float(first), float(second)))
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    return (low, high)


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
    target: _NumericTarget,
    point: QtCore.QPointF,
) -> _CurveSample | _HistogramBinSample | None:
    if isinstance(target.payload, HistogramPanelPayload):
        return _histogram_sample_for_target(target, point)
    if isinstance(target.payload, PulsePanelPayload):
        return None
    return _curve_sample_for_numeric_target(target, point)


def _curve_sample_for_numeric_target(
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
    target: _NumericTarget,
    point: QtCore.QPointF,
) -> _HistogramBinSample | None:
    payload = target.payload
    assert isinstance(payload, HistogramPanelPayload)
    viewport = payload.viewport
    normalized = _numeric_normalized_point(target, point)
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


def _set_numeric_hover(
    binding: _NumericPanelBinding,
    sample: _CurveSample | _HistogramBinSample | None,
) -> None:
    binding.hover = sample
    if sample is None:
        binding.hover_position = None


def _numeric_interaction_armed(
    selector_enabled: bool,
    binding: _NumericPanelBinding,
) -> bool:
    return (
        selector_enabled
        and binding.binding_enabled
        and binding.interaction_ready
        and binding.viewport is not None
        and binding.fault is None
    )


def _active_numeric_binding(
    bindings: dict[str, _NumericPanelBinding],
    hold: _HeldPanelFront | None,
) -> _NumericPanelBinding | None:
    return None if hold is None else bindings.get(hold.panel_id)


def _cancel_numeric_gesture(
    binding: _NumericPanelBinding,
    *,
    clear_span: bool,
) -> None:
    if clear_span and binding.rectangle_drag is not None:
        binding.span_rect = None
    binding.rectangle_drag = None
    binding.threshold_drag = None
    binding.threshold_candidate = None
    binding.threshold_pending_revision = None
    binding.threshold_pending_origin = None
    binding.pan_anchor = None
    binding.pan_origin = None
    binding.pan_candidate = None
    if clear_span:
        binding.span_candidate = None


def _clear_numeric_transient(
    binding: _NumericPanelBinding,
    *,
    clear_applied_span: bool,
    clear_pending: bool,
) -> None:
    _cancel_numeric_gesture(binding, clear_span=True)
    if clear_applied_span:
        binding.applied_span = None
    if clear_pending:
        binding.pending_viewport = None
        binding.pending_origin = None
    binding.cross = None
    _set_numeric_hover(binding, None)


def _commit_histogram_thresholds(
    binding: _NumericPanelBinding,
    thresholds: tuple[float, ...],
    *,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    panel_ids: tuple[str, ...],
    hold: _HeldPanelFront | None,
    painted_hold: _HeldPanelFront | None,
) -> bool:
    payload = (
        _numeric_payload(hold, binding.kind)
        if hold is not None
        else _visible_display(
            binding.panel_id,
            _NUMERIC_PAYLOAD_TYPES[binding.kind],
            front=front,
            panel_ids=panel_ids,
            hold=painted_hold,
        )[0]
    )
    if payload is None or tuple(thresholds) == tuple(payload.thresholds):
        return False
    if front is None:
        return False
    if hold is not None and not _hold_matches_frame(
        hold,
        front[0],
        panel_ids=panel_ids,
    ):
        return False
    _visible_payload, origin = _visible_display(
        binding.panel_id,
        _NUMERIC_PAYLOAD_TYPES[binding.kind],
        front=front,
        panel_ids=panel_ids,
        hold=painted_hold,
    )
    if origin is None:
        raise RuntimeError(
            f"{binding.kind} interaction origin has no exact payload"
        )
    command = HistogramThresholdCommit(origin, tuple(thresholds))
    expected = payload.viewport.display_revision
    if binding.viewport is not None:
        expected = max(expected, binding.viewport.display_revision)
    expected = max(expected, binding.revision_floor)
    if binding.threshold_pending_revision is not None:
        expected = max(expected, binding.threshold_pending_revision)
    binding.threshold_pending_revision = expected + 1
    binding.revision_floor = binding.threshold_pending_revision
    binding.threshold_pending_origin = origin
    try:
        binding.callback(command)
    except BaseException as error:
        binding.threshold_pending_revision = None
        binding.threshold_pending_origin = None
        if binding.fault is None:
            binding.fault = detached_render_fault(error)
        binding.binding_enabled = False
        _set_numeric_hover(binding, None)
        return False
    return True


def _commit_numeric_viewport(
    binding: _NumericPanelBinding,
    x_limits: tuple[float, float],
    *,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    panel_ids: tuple[str, ...],
    hold: _HeldPanelFront | None,
    painted_hold: _HeldPanelFront | None,
) -> bool:
    payload = (
        _numeric_payload(hold, binding.kind)
        if hold is not None
        else _visible_display(
            binding.panel_id,
            _NUMERIC_PAYLOAD_TYPES[binding.kind],
            front=front,
            panel_ids=panel_ids,
            hold=painted_hold,
        )[0]
    )
    if payload is None or x_limits == payload.viewport.x_limits:
        return False
    assert isinstance(payload, _NUMERIC_PAYLOAD_TYPES[binding.kind])
    base_revision = payload.viewport.display_revision
    if binding.viewport is not None:
        base_revision = max(
            base_revision,
            binding.viewport.display_revision,
        )
    base_revision = max(base_revision, binding.revision_floor)
    if binding.pending_viewport is not None:
        base_revision = max(
            base_revision,
            binding.pending_viewport.display_revision,
        )
    if front is None:
        return False
    if hold is not None and not _hold_matches_frame(
        hold,
        front[0],
        panel_ids=panel_ids,
    ):
        return False
    _visible_payload, origin = _visible_display(
        binding.panel_id,
        _NUMERIC_PAYLOAD_TYPES[binding.kind],
        front=front,
        panel_ids=panel_ids,
        hold=painted_hold,
    )
    if origin is None:
        raise RuntimeError(
            f"{binding.kind} interaction origin has no exact payload"
        )
    candidate = replace(
        payload.viewport,
        display_revision=base_revision + 1,
        x_limits=x_limits,
        **(
            {"x_limits_are_auto": False}
            if isinstance(payload, HistogramPanelPayload)
            else {}
        ),
    )
    command: _NumericIntent = (
        HistogramViewportCommit(origin, candidate)
        if binding.kind == "histogram"
        else CurveViewportCommit(origin, candidate)
    )
    binding.revision_floor = candidate.display_revision
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
        _set_numeric_hover(binding, None)
        return False
    return True


def _paint_numeric_overlays(
    painter: QtGui.QPainter,
    *,
    widget_rect: QtCore.QRect,
    panel_ids: tuple[str, ...],
    columns: int,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    hold: _HeldPanelFront | None,
    bindings: dict[str, _NumericPanelBinding],
) -> None:
    for binding in bindings.values():
        target = _numeric_target(
            widget_rect=widget_rect,
            panel_ids=panel_ids,
            columns=columns,
            front=front,
            hold=hold,
            binding=binding,
        )
        if target is None:
            continue
        plot, payload, bounds = target.plot, target.payload, target.bounds
        viewport = payload.viewport
        x_unit_value = (
            viewport.x_axis.unit
            if isinstance(viewport, NumericViewportTransform)
            else payload.value_unit
        )
        x_unit = "" if x_unit_value is None else f" {x_unit_value}"
        y_unit = (
            ""
            if isinstance(payload, (HistogramPanelPayload, PulsePanelPayload))
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
            rect_norm = binding.span_rect
            if rect_norm is not None:
                selector_color = selector_pen_color()
                xs_px, ys_px = _span_rect_widget_extents(target, rect_norm)
                rectangle = QtCore.QRectF(
                    QtCore.QPointF(xs_px[0], ys_px[0]),
                    QtCore.QPointF(xs_px[1], ys_px[1]),
                )
                paint_rectangle_selector(
                    painter,
                    rectangle,
                    handles=True,
                )
                if binding.rectangle_drag is None:
                    lo_x, hi_x = sorted((rect_norm[0], rect_norm[2]))
                    lo_y, hi_y = sorted((rect_norm[1], rect_norm[3]))
                    dx = selector_precision(
                        viewport.x_limits[1] - viewport.x_limits[0]
                    )
                    y_span = (
                        viewport.y_limits
                        if isinstance(viewport, NumericViewportTransform)
                        else viewport.count_limits
                    )
                    dy = selector_precision(y_span[1] - y_span[0])
                    label = (
                        f"({lo_x:.{dx}f}, {lo_y:.{dy}f})\n"
                        f"({hi_x:.{dx}f}, {hi_y:.{dy}f})"
                    )
                    paint_selector_text(
                        painter,
                        label,
                        plot,
                        selector_color,
                        corner="top_left",
                    )

            cross = binding.cross
            if cross is not None:
                selector_color = selector_pen_color()
                point = widget_point(cross.x, cross.y)
                if plot.contains(point):
                    painter.setPen(QtGui.QPen(selector_color, SELECTOR_LINE_PX))
                    painter.drawLine(
                        QtCore.QPointF(point.x(), plot.top()),
                        QtCore.QPointF(point.x(), plot.bottom()),
                    )
                    painter.drawLine(
                        QtCore.QPointF(plot.left(), point.y()),
                        QtCore.QPointF(plot.right(), point.y()),
                    )
                    painter.setBrush(QtGui.QBrush(selector_color))
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.drawEllipse(
                        point,
                        SELECTOR_DOT_PX / 2.0,
                        SELECTOR_DOT_PX / 2.0,
                    )
                dx = selector_precision(
                    viewport.x_limits[1] - viewport.x_limits[0]
                )
                y_span = (
                    viewport.y_limits
                    if isinstance(viewport, NumericViewportTransform)
                    else viewport.count_limits
                )
                dy = selector_precision(y_span[1] - y_span[0])
                paint_selector_text(
                    painter,
                    f"({cross.x:.{dx}f}, {cross.y:.{dy}f})",
                    plot,
                    selector_color,
                    corner="top_right",
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
                paint_selector_hover_label(
                    painter,
                    label,
                    plot,
                    QtGui.QColor(ORANGE),
                    anchor=position,
                )
        finally:
            painter.restore()
