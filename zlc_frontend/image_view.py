"""Headless coordinate transforms for one regular two-axis pixel image.

The transform is immutable and contains the committed visible window.  Every
pointer and rectangle is first expressed in that visible window, then mapped
back to the complete raster.  Qt, render workers, and selection consumers can
therefore share one exact mapping without sharing a Figure, artist, or widget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TypeAlias

import numpy as np

from zlc_data import (
    AxisSpec,
    CoordinateFrameId,
    CoordinateRangeSelection,
    Selection,
    SPATIAL_X,
    SPATIAL_Y,
    immutable_array,
    resolve_selection_indices,
)
from zlc_storage import nonnegative_integer, positive_integer


NormalizedPoint: TypeAlias = tuple[float, float]
NormalizedRectangle: TypeAlias = tuple[float, float, float, float]
CoordinateView: TypeAlias = tuple[float, float]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return 0.0 if number == 0.0 else number


def validate_normalized_point(
    value: object,
    name: str = "normalized point",
) -> NormalizedPoint:
    """Return one finite ``(x, y)`` point inside the closed unit square."""

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item tuple")
    x = _finite_number(value[0], f"{name} x")
    y = _finite_number(value[1], f"{name} y")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError(f"{name} must lie inside [0, 1] x [0, 1]")
    return x, y


def validate_normalized_rectangle(value: object) -> NormalizedRectangle:
    """Return one finite, non-degenerate rectangle inside the unit square."""

    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("normalized bounds must be a four-item tuple")
    names = ("left", "top", "right", "bottom")
    normalized = tuple(
        _finite_number(item, f"normalized {name}")
        for name, item in zip(names, value, strict=True)
    )
    left, top, right, bottom = normalized
    if any(number < 0.0 or number > 1.0 for number in normalized):
        raise ValueError("normalized bounds must lie inside [0, 1]")
    if not left < right or not top < bottom:
        raise ValueError("normalized bounds must contain a non-degenerate rectangle")
    return left, top, right, bottom


def _coordinate_view(value: object, name: str) -> CoordinateView:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item tuple")
    low = _finite_number(value[0], f"{name} low")
    high = _finite_number(value[1], f"{name} high")
    if not low < high:
        raise ValueError(f"{name} low must be below high")
    if not math.isfinite(high - low):
        raise ValueError(f"{name} span must be finite")
    return low, high


def _axis_step(axis: AxisSpec) -> int | float:
    if axis.unit != "pixel":
        raise ValueError(f"axis {axis.axis_id} must declare unit='pixel'")
    if axis.coordinates is None:
        raise ValueError(f"axis {axis.axis_id} requires explicit pixel coordinates")
    first = axis.coordinate_at(0)
    if (
        isinstance(first, bool)
        or not isinstance(first, Real)
        or not math.isfinite(float(first))
    ):
        raise TypeError(f"axis {axis.axis_id} requires finite numeric coordinates")
    if axis.size == 1:
        return 1
    second = axis.coordinate_at(1)
    if (
        isinstance(second, bool)
        or not isinstance(second, Real)
        or not math.isfinite(float(second))
    ):
        raise TypeError(f"axis {axis.axis_id} requires finite numeric coordinates")
    step = second - first
    if step == 0:
        raise ValueError(f"axis {axis.axis_id} coordinates must be strictly monotonic")
    previous = second
    for index in range(2, axis.size):
        current = axis.coordinate_at(index)
        if (
            isinstance(current, bool)
            or not isinstance(current, Real)
            or not math.isfinite(float(current))
        ):
            raise TypeError(f"axis {axis.axis_id} requires finite numeric coordinates")
        if current - previous != step:
            raise ValueError(
                f"axis {axis.axis_id} coordinates are not exactly regular; "
                "the pixel viewport refuses an affine approximation"
            )
        previous = current
    return step


def _axis_edge_coordinates(axis: AxisSpec) -> tuple[float, float]:
    """Return index-edge coordinates without inventing singleton spacing."""

    if axis.size < 2:
        raise ValueError(
            f"axis {axis.axis_id} needs at least two coordinates to define "
            "a coordinate viewport"
        )
    step = float(_axis_step(axis))
    first = float(axis.coordinate_at(0))
    start = first - 0.5 * step
    stop = start + axis.size * step
    if not math.isfinite(start) or not math.isfinite(stop) or start == stop:
        raise ValueError(f"axis {axis.axis_id} has degenerate coordinate edges")
    return start, stop


def _normalized_interval_for_coordinate_view(
    axis: AxisSpec,
    view: object,
    name: str,
) -> tuple[float, float]:
    low, high = _coordinate_view(view, name)
    start, stop = _axis_edge_coordinates(axis)
    domain_low, domain_high = sorted((start, stop))
    # Absorb only arithmetic round-off at the domain's magnitude.  A relative
    # tolerance would become a physically meaningful out-of-bounds allowance
    # for cameras whose absolute ROI coordinates are large.
    tolerance = 8.0 * math.ulp(
        max(1.0, abs(domain_low), abs(domain_high), abs(low), abs(high))
    )
    if low < domain_low - tolerance or high > domain_high + tolerance:
        raise ValueError(
            f"{name} lies outside axis {axis.axis_id} edge extent "
            f"[{domain_low}, {domain_high}]"
        )
    low = min(domain_high, max(domain_low, low))
    high = min(domain_high, max(domain_low, high))
    first_fraction = (low - start) / (stop - start)
    second_fraction = (high - start) / (stop - start)
    normalized_low, normalized_high = sorted((first_fraction, second_fraction))
    normalized_low = min(1.0, max(0.0, normalized_low))
    normalized_high = min(1.0, max(0.0, normalized_high))
    if (normalized_high - normalized_low) * axis.size < 1.0 - 1e-12:
        raise ValueError(f"{name} must contain at least one raster cell")
    return normalized_low, normalized_high


def _coordinate_view_for_normalized_interval(
    axis: AxisSpec,
    low: float,
    high: float,
) -> CoordinateView:
    start, stop = _axis_edge_coordinates(axis)
    first = start + low * (stop - start)
    second = start + high * (stop - start)
    coordinate_low, coordinate_high = sorted((first, second))
    return (
        0.0 if coordinate_low == 0.0 else coordinate_low,
        0.0 if coordinate_high == 0.0 else coordinate_high,
    )


def _edge_index(value: float, size: int, name: str) -> int:
    scaled = value * size
    candidate = int(round(scaled))
    # Bounds emitted by this module are integer pixel edges divided by size.
    # The tolerance only absorbs that division's floating-point round-trip.
    if not math.isclose(scaled, candidate, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"normalized {name} is not aligned to a pixel-cell edge")
    if not 0 <= candidate <= size:
        raise ValueError(f"normalized {name} lies outside the raster")
    return candidate


def _sample_index(
    full_coordinate: float,
    size: int,
    *,
    at_visible_upper_edge: bool,
) -> int:
    if at_visible_upper_edge:
        candidate = math.ceil(full_coordinate * size) - 1
    else:
        candidate = math.floor(full_coordinate * size)
    return min(size - 1, max(0, candidate))


def _clamped_interval(start: float, span: float) -> tuple[float, float]:
    start = max(0.0, min(1.0 - span, start))
    return start, start + span


@dataclass(frozen=True, slots=True)
class ImageViewportTransform:
    """Exact map between a committed visible window and a complete pixel raster.

    ``axes`` must contain exactly one named ``SPATIAL_X`` and ``SPATIAL_Y``
    axis.  ``visible_bounds`` uses complete-raster edge coordinates; the home
    view is ``(0, 0, 1, 1)``.  Rank, singleton shape, and tuple position never
    supply an axis role.
    """

    axes: tuple[AxisSpec, AxisSpec]
    viewport_revision: int = 0
    visible_bounds: NormalizedRectangle = (0.0, 0.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if len(axes) != 2 or any(not isinstance(axis, AxisSpec) for axis in axes):
            raise ValueError("ImageViewportTransform requires exactly two AxisSpec values")
        by_role = {axis.role: axis for axis in axes}
        if len(by_role) != 2 or set(by_role) != {SPATIAL_X, SPATIAL_Y}:
            raise ValueError(
                "ImageViewportTransform requires exactly one named SPATIAL_X "
                "and one named SPATIAL_Y axis"
            )
        x_axis, y_axis = by_role[SPATIAL_X], by_role[SPATIAL_Y]
        if x_axis.axis_id == y_axis.axis_id:
            raise ValueError("image axes require distinct AxisId values")
        if (
            x_axis.coordinate_frame is None
            or y_axis.coordinate_frame is None
            or x_axis.coordinate_frame != y_axis.coordinate_frame
        ):
            raise ValueError("image axes require one shared, explicit coordinate frame")
        _axis_step(x_axis)
        _axis_step(y_axis)
        bounds = validate_normalized_rectangle(self.visible_bounds)
        if (bounds[2] - bounds[0]) * x_axis.size < 1.0 - 1e-12:
            raise ValueError("visible x bounds must contain at least one raster cell")
        if (bounds[3] - bounds[1]) * y_axis.size < 1.0 - 1e-12:
            raise ValueError("visible y bounds must contain at least one raster cell")
        object.__setattr__(self, "axes", (x_axis, y_axis))
        object.__setattr__(
            self,
            "viewport_revision",
            nonnegative_integer(self.viewport_revision, "viewport_revision"),
        )
        object.__setattr__(self, "visible_bounds", bounds)

    @property
    def x_axis(self) -> AxisSpec:
        return self.axes[0]

    @property
    def y_axis(self) -> AxisSpec:
        return self.axes[1]

    @property
    def raster_shape(self) -> tuple[int, int]:
        return self.y_axis.size, self.x_axis.size

    @property
    def coordinate_frame(self) -> CoordinateFrameId:
        assert self.x_axis.coordinate_frame is not None
        return self.x_axis.coordinate_frame

    def normalized_bounds_for_optional_coordinate_views(
        self,
        x_view: CoordinateView | None,
        y_view: CoordinateView | None,
    ) -> NormalizedRectangle:
        """Map authored axis pins while an omitted axis remains at home.

        This is the exact bridge from :class:`ImageDisplayState`: ``None`` is
        not guessed from data or shape; it explicitly means the complete axis.
        Treating the axes independently also keeps a singleton unpinned axis
        usable even though it cannot define a non-degenerate public range.
        """

        left, right = (
            (0.0, 1.0)
            if x_view is None
            else _normalized_interval_for_coordinate_view(
                self.x_axis,
                x_view,
                "x_view",
            )
        )
        top, bottom = (
            (0.0, 1.0)
            if y_view is None
            else _normalized_interval_for_coordinate_view(
                self.y_axis,
                y_view,
                "y_view",
            )
        )
        return validate_normalized_rectangle((left, top, right, bottom))

    def optional_coordinate_views_for_normalized_bounds(
        self,
        bounds: NormalizedRectangle | None = None,
    ) -> tuple[CoordinateView | None, CoordinateView | None]:
        """Return authored pins, using ``None`` for each complete axis.

        Each axis is handled independently, so a complete singleton axis does
        not need an invented public coordinate span.
        """

        checked = self.visible_bounds if bounds is None else validate_normalized_rectangle(
            bounds
        )
        left, top, right, bottom = checked
        x_view = (
            None
            if (left, right) == (0.0, 1.0)
            else _coordinate_view_for_normalized_interval(self.x_axis, left, right)
        )
        y_view = (
            None
            if (top, bottom) == (0.0, 1.0)
            else _coordinate_view_for_normalized_interval(self.y_axis, top, bottom)
        )
        return x_view, y_view

    def full_point_for_visible_point(
        self,
        point: NormalizedPoint,
    ) -> NormalizedPoint:
        """Map a point in the visible window to complete-raster coordinates."""

        x, y = validate_normalized_point(point, "visible point")
        left, top, right, bottom = self.visible_bounds
        return (
            left + x * (right - left),
            top + y * (bottom - top),
        )

    def visible_point_for_full_point(
        self,
        point: NormalizedPoint,
    ) -> NormalizedPoint:
        """Map a complete-raster point contained by this visible window."""

        x, y = validate_normalized_point(point, "full-raster point")
        left, top, right, bottom = self.visible_bounds
        tolerance = 1e-12
        if (
            x < left - tolerance
            or x > right + tolerance
            or y < top - tolerance
            or y > bottom + tolerance
        ):
            raise ValueError("full-raster point lies outside the visible window")
        return (
            min(1.0, max(0.0, (x - left) / (right - left))),
            min(1.0, max(0.0, (y - top) / (bottom - top))),
        )

    def full_points_for_coordinates(
        self,
        coordinates_xy: object,
        *,
        coordinate_frame: CoordinateFrameId,
    ) -> np.ndarray:
        """Freeze many physical points in complete-raster normalized coordinates.

        Axis regularity and edge geometry are resolved once per axis, not once
        per point.  The returned float64 matrix is backed by immutable bytes so
        a composite payload can safely retain it across worker/GUI hand-off.
        """

        if not isinstance(coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        if coordinate_frame != self.coordinate_frame:
            raise ValueError("coordinates belong to another coordinate frame")
        source = np.asarray(coordinates_xy)
        if source.ndim != 2 or source.shape[1:] != (2,):
            raise ValueError("coordinates_xy must have shape (points, 2)")
        if source.dtype.kind not in "iuf" or source.dtype.kind == "b":
            raise TypeError("coordinates_xy must contain real numeric values")
        values = np.asarray(source, dtype=np.float64, order="C")
        if not np.all(np.isfinite(values)):
            raise ValueError("coordinates_xy must be finite")

        normalized = np.empty(values.shape, dtype=np.float64)
        for column, axis, name in (
            (0, self.x_axis, "x coordinates"),
            (1, self.y_axis, "y coordinates"),
        ):
            start, stop = _axis_edge_coordinates(axis)
            low, high = sorted((start, stop))
            coordinates = values[:, column]
            magnitude = max(
                1.0,
                abs(low),
                abs(high),
                0.0 if not len(coordinates) else float(np.max(np.abs(coordinates))),
            )
            tolerance = 8.0 * math.ulp(magnitude)
            if np.any(coordinates < low - tolerance) or np.any(
                coordinates > high + tolerance
            ):
                raise ValueError(
                    f"{name} lie outside axis {axis.axis_id} edge extent "
                    f"[{low}, {high}]"
                )
            target = normalized[:, column]
            np.subtract(coordinates, start, out=target)
            np.divide(target, stop - start, out=target)
            np.clip(target, 0.0, 1.0, out=target)

        return immutable_array(
            normalized,
            dtype=np.dtype("<f8"),
            shape=normalized.shape,
        )

    def visible_span_for_coordinate_span(
        self,
        span_xy: tuple[object, object],
        *,
        coordinate_frame: CoordinateFrameId,
    ) -> tuple[float, float]:
        """Return visible-normalized width/height for a physical coordinate span."""

        if not isinstance(span_xy, tuple) or len(span_xy) != 2:
            raise TypeError("span_xy must be a two-item tuple")
        if not isinstance(coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId")
        if coordinate_frame != self.coordinate_frame:
            raise ValueError("coordinate span belongs to another coordinate frame")
        span_x = _finite_number(span_xy[0], "x coordinate span")
        span_y = _finite_number(span_xy[1], "y coordinate span")
        if span_x <= 0.0 or span_y <= 0.0:
            raise ValueError("coordinate spans must be positive")
        x_start, x_stop = _axis_edge_coordinates(self.x_axis)
        y_start, y_stop = _axis_edge_coordinates(self.y_axis)
        left, top, right, bottom = self.visible_bounds
        return (
            span_x / abs(x_stop - x_start) / (right - left),
            span_y / abs(y_stop - y_start) / (bottom - top),
        )

    def visible_bounds_for_full_bounds(
        self,
        bounds: NormalizedRectangle,
    ) -> NormalizedRectangle:
        """Map a fully visible complete-raster rectangle to visible coordinates."""

        left, top, right, bottom = validate_normalized_rectangle(bounds)
        visible_left, visible_top = self.visible_point_for_full_point((left, top))
        visible_right, visible_bottom = self.visible_point_for_full_point(
            (right, bottom)
        )
        return validate_normalized_rectangle(
            (visible_left, visible_top, visible_right, visible_bottom)
        )

    def clipped_visible_bounds_for_full_bounds(
        self,
        bounds: NormalizedRectangle,
    ) -> NormalizedRectangle | None:
        """Project the intersection with this window, or return ``None``."""

        left, top, right, bottom = validate_normalized_rectangle(bounds)
        view_left, view_top, view_right, view_bottom = self.visible_bounds
        clipped = (
            max(left, view_left),
            max(top, view_top),
            min(right, view_right),
            min(bottom, view_bottom),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            return None
        return self.visible_bounds_for_full_bounds(clipped)

    def sample_indices_for_visible_point(
        self,
        point: NormalizedPoint,
    ) -> tuple[int, int]:
        """Return exact ``(y_index, x_index)`` for one visible point.

        The visible right/bottom edges belong to the last cell inside the
        window, not the first cell immediately outside it.
        """

        visible_x, visible_y = validate_normalized_point(point, "visible point")
        full_x, full_y = self.full_point_for_visible_point((visible_x, visible_y))
        x_index = _sample_index(
            full_x,
            self.x_axis.size,
            at_visible_upper_edge=visible_x == 1.0,
        )
        y_index = _sample_index(
            full_y,
            self.y_axis.size,
            at_visible_upper_edge=visible_y == 1.0,
        )
        return y_index, x_index

    def sample_coordinates_for_visible_point(
        self,
        point: NormalizedPoint,
    ) -> tuple[object, object]:
        """Return exact ``(x_coordinate, y_coordinate)`` for one sample cell."""

        y_index, x_index = self.sample_indices_for_visible_point(point)
        return (
            self.x_axis.coordinate_at(x_index),
            self.y_axis.coordinate_at(y_index),
        )

    def normalized_bounds_for_selection(
        self,
        selection: Selection,
    ) -> NormalizedRectangle:
        """Resolve a typed rectangle to complete-raster pixel-cell edges."""

        if not isinstance(selection, Selection):
            raise TypeError("selection must be zlc_data.Selection")
        terms = {term.axis_id: term for term in selection.terms}
        expected = {self.x_axis.axis_id, self.y_axis.axis_id}
        if set(terms) != expected or any(
            not isinstance(term, CoordinateRangeSelection) for term in terms.values()
        ):
            raise ValueError("image selection must be one typed spatial rectangle")
        x_indices, x_drop = resolve_selection_indices(
            self.x_axis,
            terms[self.x_axis.axis_id],
        )
        y_indices, y_drop = resolve_selection_indices(
            self.y_axis,
            terms[self.y_axis.axis_id],
        )
        if (
            x_drop
            or y_drop
            or not isinstance(x_indices, range)
            or not isinstance(y_indices, range)
            or x_indices.step != 1
            or y_indices.step != 1
        ):
            raise ValueError("image rectangle must resolve to contiguous pixel cells")
        return validate_normalized_rectangle(
            (
                x_indices.start / self.x_axis.size,
                y_indices.start / self.y_axis.size,
                x_indices.stop / self.x_axis.size,
                y_indices.stop / self.y_axis.size,
            )
        )

    def selection_for_normalized_bounds(
        self,
        bounds: NormalizedRectangle,
    ) -> Selection:
        """Convert complete-raster edge-aligned bounds to a closed Selection."""

        left, top, right, bottom = validate_normalized_rectangle(bounds)
        x_start = _edge_index(left, self.x_axis.size, "left")
        x_stop = _edge_index(right, self.x_axis.size, "right")
        y_start = _edge_index(top, self.y_axis.size, "top")
        y_stop = _edge_index(bottom, self.y_axis.size, "bottom")
        if x_start >= x_stop or y_start >= y_stop:
            raise ValueError("normalized rectangle contains no pixel cells")
        x_first = self.x_axis.coordinate_at(x_start)
        x_last = self.x_axis.coordinate_at(x_stop - 1)
        y_first = self.y_axis.coordinate_at(y_start)
        y_last = self.y_axis.coordinate_at(y_stop - 1)
        return Selection.rectangle(
            self.x_axis.axis_id,
            self.y_axis.axis_id,
            min(x_first, x_last),
            max(x_first, x_last),
            min(y_first, y_last),
            max(y_first, y_last),
            coordinate_frame=self.coordinate_frame,
        )

    def snapped_bounds_for_drag(
        self,
        start: NormalizedPoint,
        end: NormalizedPoint,
    ) -> NormalizedRectangle:
        """Snap two visible-window points to complete-raster cell edges."""

        y0, x0 = self.sample_indices_for_visible_point(start)
        y1, x1 = self.sample_indices_for_visible_point(end)
        return validate_normalized_rectangle(
            (
                min(x0, x1) / self.x_axis.size,
                min(y0, y1) / self.y_axis.size,
                (max(x0, x1) + 1) / self.x_axis.size,
                (max(y0, y1) + 1) / self.y_axis.size,
            )
        )

    def with_visible_bounds(
        self,
        bounds: NormalizedRectangle,
        *,
        viewport_revision: int | None = None,
    ) -> ImageViewportTransform:
        """Commit another visible window, increasing its revision if it changed."""

        checked = validate_normalized_rectangle(bounds)
        if checked == self.visible_bounds:
            return self
        revision = self._replacement_revision(viewport_revision)
        return ImageViewportTransform(self.axes, revision, checked)

    def centered_zoom(
        self,
        point: NormalizedPoint,
        span_scale: float,
        *,
        viewport_revision: int | None = None,
    ) -> ImageViewportTransform:
        """Scale both spans around one visible point.

        ``span_scale < 1`` zooms in and ``span_scale > 1`` zooms out.  Each
        span is clamped to the complete raster and to at least one sample cell.
        """

        visible_x, visible_y = validate_normalized_point(point, "zoom point")
        scale = _finite_number(span_scale, "span_scale")
        if scale <= 0.0:
            raise ValueError("span_scale must be positive")
        if scale == 1.0:
            return self
        left, top, right, bottom = self.visible_bounds
        old_width, old_height = right - left, bottom - top
        new_width = min(1.0, max(1.0 / self.x_axis.size, old_width * scale))
        new_height = min(1.0, max(1.0 / self.y_axis.size, old_height * scale))
        anchor_x, anchor_y = self.full_point_for_visible_point((visible_x, visible_y))
        new_left, new_right = _clamped_interval(
            anchor_x - visible_x * new_width,
            new_width,
        )
        new_top, new_bottom = _clamped_interval(
            anchor_y - visible_y * new_height,
            new_height,
        )
        return self.with_visible_bounds(
            (new_left, new_top, new_right, new_bottom),
            viewport_revision=viewport_revision,
        )

    def panned_by_pixels(
        self,
        delta_pixels: tuple[float, float],
        viewport_size_pixels: tuple[int, int],
        *,
        viewport_revision: int | None = None,
    ) -> ImageViewportTransform:
        """Pan from a press-time pixel delta while preserving both spans.

        Positive pointer motion moves the image with the pointer, so the source
        window moves in the opposite direction.  Call this on the transform
        frozen at press time rather than feeding each result into the next move.
        """

        if not isinstance(delta_pixels, tuple) or len(delta_pixels) != 2:
            raise TypeError("delta_pixels must be a two-item tuple")
        if not isinstance(viewport_size_pixels, tuple) or len(viewport_size_pixels) != 2:
            raise TypeError("viewport_size_pixels must be a (width, height) tuple")
        delta_x = _finite_number(delta_pixels[0], "horizontal pixel delta")
        delta_y = _finite_number(delta_pixels[1], "vertical pixel delta")
        width = positive_integer(viewport_size_pixels[0], "viewport pixel width")
        height = positive_integer(viewport_size_pixels[1], "viewport pixel height")
        if delta_x == 0.0 and delta_y == 0.0:
            return self
        left, top, right, bottom = self.visible_bounds
        span_x, span_y = right - left, bottom - top
        new_left, new_right = _clamped_interval(
            left - delta_x * span_x / width,
            span_x,
        )
        new_top, new_bottom = _clamped_interval(
            top - delta_y * span_y / height,
            span_y,
        )
        return self.with_visible_bounds(
            (new_left, new_top, new_right, new_bottom),
            viewport_revision=viewport_revision,
        )

    def _replacement_revision(self, revision: int | None) -> int:
        if revision is None:
            return self.viewport_revision + 1
        checked = nonnegative_integer(revision, "viewport_revision")
        if checked <= self.viewport_revision:
            raise ValueError("replacement viewport_revision must increase")
        return checked


__all__ = [
    "CoordinateView",
    "ImageViewportTransform",
    "NormalizedPoint",
    "NormalizedRectangle",
    "validate_normalized_point",
    "validate_normalized_rectangle",
]
