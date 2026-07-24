"""Headless coordinate transforms for one regular numeric two-axis raster.

The transform is immutable and contains the committed visible window.  Every
pointer and rectangle is first expressed in that visible window, then mapped
back to the complete raster.  Qt, render workers, and selection consumers can
therefore share one exact mapping without sharing a Figure, artist, or widget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import TypeAlias

import numpy as np

from zlc_data import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    CoordinateRangeSelection,
    IndexRangeSelection,
    Selection,
    SPATIAL_X,
    SPATIAL_Y,
    immutable_array,
    resolve_selection_indices,
)
from zlc_storage import nonnegative_integer, positive_integer

from .figure.model import EvaluatedAxis, EvaluatedImage
from .plot_layout import square_image_extent


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


def _unbounded_normalized_rectangle(
    value: object,
    name: str = "normalized bounds",
) -> NormalizedRectangle:
    """Return finite ordered bounds without clipping them to the source raster."""

    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError(f"{name} must be a four-item tuple")
    left, top, right, bottom = (
        _finite_number(item, f"{name} {edge}")
        for edge, item in zip(("left", "top", "right", "bottom"), value, strict=True)
    )
    if not left < right or not top < bottom:
        raise ValueError(f"{name} must contain a non-degenerate rectangle")
    return left, top, right, bottom


def _axis_step(axis: AxisSpec) -> int | float:
    if axis.coordinates is None:
        raise ValueError(f"axis {axis.axis_id} requires explicit numeric coordinates")
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
                "the IMAGE viewport refuses an affine approximation"
            )
        previous = current
    return step


def _axis_edge_coordinates_from_step(
    axis: AxisSpec,
    step: float,
) -> tuple[float, float]:
    if axis.size < 2:
        raise ValueError(
            f"axis {axis.axis_id} needs at least two coordinates to define "
            "a coordinate viewport"
        )
    first = float(axis.coordinate_at(0))
    start = first - 0.5 * step
    stop = start + axis.size * step
    if not math.isfinite(start) or not math.isfinite(stop) or start == stop:
        raise ValueError(f"axis {axis.axis_id} has degenerate coordinate edges")
    return start, stop


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


@dataclass(frozen=True, slots=True, init=False)
class ImageViewportTransform:
    """Exact map between a committed visible window and a numeric 2-D raster.

    ``display_axis_ids`` is the explicit ``(IMAGE_X, IMAGE_Y)`` binding when
    the source axes do not identify their display direction themselves.  A
    traditional spatial image may omit it because the declared
    ``SPATIAL_X``/``SPATIAL_Y`` roles are already unambiguous.  The source
    roles remain untouched: two scan-point axes stay scan-point axes while the
    Figure binding says which is horizontal and which is vertical.

    Physical ``x_limits``/``y_limits`` are the sole viewport authority.  The
    normalized ``visible_bounds`` bridge is derived for raster-cell selections;
    it is allowed to extend beyond the source when Main's square home padding,
    zoom-out, or pan exposes axes background.  Rank, singleton shape, and tuple
    position never supply an axis role or display direction.
    """

    axes: tuple[AxisSpec, AxisSpec]
    viewport_revision: int = 0
    x_limits: CoordinateView = field(init=False)
    y_limits: CoordinateView = field(init=False)
    _home_x_limits: CoordinateView = field(init=False, repr=False, compare=False)
    _home_y_limits: CoordinateView = field(init=False, repr=False, compare=False)
    _x_edge_coordinates: tuple[float, float] | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    _y_edge_coordinates: tuple[float, float] | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        axes: tuple[AxisSpec, AxisSpec],
        viewport_revision: int = 0,
        visible_bounds: NormalizedRectangle | None = None,
        display_axis_ids: tuple[AxisId, AxisId] | None = None,
        *,
        x_limits: CoordinateView | None = None,
        y_limits: CoordinateView | None = None,
    ) -> None:
        axes = tuple(axes)
        if len(axes) != 2 or any(not isinstance(axis, AxisSpec) for axis in axes):
            raise ValueError("ImageViewportTransform requires exactly two AxisSpec values")
        if display_axis_ids is None:
            by_role = {axis.role: axis for axis in axes}
            if len(by_role) != 2 or set(by_role) != {SPATIAL_X, SPATIAL_Y}:
                raise ValueError(
                    "non-spatial IMAGE axes require explicit IMAGE_X/IMAGE_Y bindings"
                )
            x_axis, y_axis = by_role[SPATIAL_X], by_role[SPATIAL_Y]
        else:
            display_ids = tuple(display_axis_ids)
            if (
                len(display_ids) != 2
                or any(not isinstance(axis_id, AxisId) for axis_id in display_ids)
                or display_ids[0] == display_ids[1]
            ):
                raise ValueError(
                    "display_axis_ids must contain distinct IMAGE_X and IMAGE_Y AxisId values"
                )
            by_id = {axis.axis_id: axis for axis in axes}
            if len(by_id) != 2 or set(display_ids) != set(by_id):
                raise ValueError(
                    "IMAGE_X/IMAGE_Y bindings must identify the two viewport axes"
                )
            x_axis, y_axis = by_id[display_ids[0]], by_id[display_ids[1]]
        if x_axis.axis_id == y_axis.axis_id:
            raise ValueError("image axes require distinct AxisId values")
        if x_axis.role == SPATIAL_Y or y_axis.role == SPATIAL_X:
            raise ValueError(
                "spatial IMAGE roles disagree with the explicit IMAGE_X/IMAGE_Y bindings"
            )
        spatial_axes = tuple(
            axis for axis in (x_axis, y_axis) if axis.role in (SPATIAL_X, SPATIAL_Y)
        )
        if any(axis.coordinate_frame is None for axis in spatial_axes):
            raise ValueError("spatial image axes require an explicit coordinate frame")
        if any(axis.unit != "pixel" for axis in spatial_axes):
            raise ValueError("spatial image axes must declare unit='pixel'")
        if (
            x_axis.role == SPATIAL_X
            and y_axis.role == SPATIAL_Y
            and x_axis.coordinate_frame != y_axis.coordinate_frame
        ):
            raise ValueError("spatial image axes require one shared coordinate frame")
        x_step = float(_axis_step(x_axis))
        y_step = float(_axis_step(y_axis))
        x_edge_coordinates = (
            None
            if x_axis.size == 1
            else _axis_edge_coordinates_from_step(x_axis, x_step)
        )
        y_edge_coordinates = (
            None
            if y_axis.size == 1
            else _axis_edge_coordinates_from_step(y_axis, y_step)
        )
        object.__setattr__(self, "axes", (x_axis, y_axis))
        object.__setattr__(
            self,
            "viewport_revision",
            nonnegative_integer(viewport_revision, "viewport_revision"),
        )
        object.__setattr__(self, "_x_edge_coordinates", x_edge_coordinates)
        object.__setattr__(self, "_y_edge_coordinates", y_edge_coordinates)
        home = square_image_extent(self.data_extent)
        home_x = tuple(sorted(home[:2]))
        home_y = tuple(sorted(home[2:]))
        object.__setattr__(self, "_home_x_limits", home_x)
        object.__setattr__(self, "_home_y_limits", home_y)
        if visible_bounds is not None:
            if x_limits is not None or y_limits is not None:
                raise ValueError(
                    "visible_bounds cannot be combined with physical x/y limits"
                )
            left, top, right, bottom = _unbounded_normalized_rectangle(
                visible_bounds
            )
            x_start, x_stop = self._cached_axis_edges(x_axis)
            y_start, y_stop = self._cached_axis_edges(y_axis)
            x_limits = tuple(sorted((
                x_start + left * (x_stop - x_start),
                x_start + right * (x_stop - x_start),
            )))
            y_limits = tuple(sorted((
                y_start + top * (y_stop - y_start),
                y_start + bottom * (y_stop - y_start),
            )))
        object.__setattr__(
            self,
            "x_limits",
            home_x if x_limits is None else _coordinate_view(x_limits, "x_limits"),
        )
        object.__setattr__(
            self,
            "y_limits",
            home_y if y_limits is None else _coordinate_view(y_limits, "y_limits"),
        )

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
    def coordinate_frame(self) -> CoordinateFrameId | None:
        """Return a shared 2-D frame, or ``None`` for independent numeric axes."""

        frame = self.x_axis.coordinate_frame
        return frame if frame == self.y_axis.coordinate_frame else None

    def _cached_axis_edges(self, axis: AxisSpec) -> tuple[float, float]:
        """Return the affine cell edges validated when this viewport was built.

        Camera axes can contain thousands of explicit coordinates.  Rewalking
        that immutable tuple every time a frame asks for its extent (and again
        for every pointer mapping) made the supposedly cheap view transform an
        O(width + height) live-render operation.  Construction already
        proves regularity once, so every subsequent mapping must consume that
        proof rather than repeat it.
        """

        if axis.axis_id == self.x_axis.axis_id:
            cached = self._x_edge_coordinates
        elif axis.axis_id == self.y_axis.axis_id:
            cached = self._y_edge_coordinates
        else:  # internal misuse; no role/rank guessing
            raise ValueError("axis does not belong to this image viewport")
        if cached is not None:
            return cached
        center = float(axis.coordinate_at(0))
        return center - 0.5, center + 0.5

    @property
    def home_x_limits(self) -> CoordinateView:
        return self._home_x_limits

    @property
    def home_y_limits(self) -> CoordinateView:
        return self._home_y_limits

    @property
    def visible_bounds(self) -> NormalizedRectangle:
        """Physical limits expressed against the complete raster, top-origin."""

        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
        x_first = (self.x_limits[0] - x_start) / (x_stop - x_start)
        x_second = (self.x_limits[1] - x_start) / (x_stop - x_start)
        y_first = (self.y_limits[0] - y_start) / (y_stop - y_start)
        y_second = (self.y_limits[1] - y_start) / (y_stop - y_start)
        left, right = sorted((x_first, x_second))
        top, bottom = sorted((y_first, y_second))
        return _unbounded_normalized_rectangle((left, top, right, bottom))

    def _validated_replacement(
        self,
        x_limits: CoordinateView,
        y_limits: CoordinateView,
        revision: int,
    ) -> ImageViewportTransform:
        """Clone physical limits without revalidating immutable source axes."""

        candidate = object.__new__(type(self))
        object.__setattr__(candidate, "axes", self.axes)
        object.__setattr__(candidate, "viewport_revision", revision)
        object.__setattr__(candidate, "x_limits", _coordinate_view(x_limits, "x_limits"))
        object.__setattr__(candidate, "y_limits", _coordinate_view(y_limits, "y_limits"))
        object.__setattr__(candidate, "_home_x_limits", self._home_x_limits)
        object.__setattr__(candidate, "_home_y_limits", self._home_y_limits)
        object.__setattr__(
            candidate, "_x_edge_coordinates", self._x_edge_coordinates
        )
        object.__setattr__(
            candidate, "_y_edge_coordinates", self._y_edge_coordinates
        )
        return candidate

    def _replacement_with_visible_bounds(
        self,
        bounds: NormalizedRectangle,
        revision: int,
    ) -> ImageViewportTransform:
        """Convert a normalized selection bridge into physical limits once."""

        left, top, right, bottom = _unbounded_normalized_rectangle(bounds)
        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
        x_limits = tuple(sorted((
            x_start + left * (x_stop - x_start),
            x_start + right * (x_stop - x_start),
        )))
        y_limits = tuple(sorted((
            y_start + top * (y_stop - y_start),
            y_start + bottom * (y_stop - y_start),
        )))
        return self._validated_replacement(x_limits, y_limits, revision)

    @property
    def data_extent(self) -> tuple[float, float, float, float]:
        """The true raw ``left,right,bottom,upper`` sample-cell extent."""

        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
        return (x_start, x_stop, y_stop, y_start)

    @property
    def visible_data_extent(self) -> tuple[float, float, float, float]:
        """Current physical Matplotlib ``left,right,bottom,upper`` limits."""

        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
        x_left_right = (
            self.x_limits
            if x_stop > x_start
            else (self.x_limits[1], self.x_limits[0])
        )
        y_bottom_upper = (
            (self.y_limits[1], self.y_limits[0])
            if y_stop > y_start
            else self.y_limits
        )
        return (*x_left_right, *y_bottom_upper)

    @property
    def display_extent(self) -> tuple[float, float, float, float]:
        """Alias for the one physical viewport consumed by the renderer."""

        return self.visible_data_extent

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

        x_limits = self.home_x_limits if x_view is None else _coordinate_view(x_view, "x_view")
        y_limits = self.home_y_limits if y_view is None else _coordinate_view(y_view, "y_view")
        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
        left, right = sorted(
            ((value - x_start) / (x_stop - x_start) for value in x_limits)
        )
        top, bottom = sorted(
            ((value - y_start) / (y_stop - y_start) for value in y_limits)
        )
        return _unbounded_normalized_rectangle((left, top, right, bottom))

    def optional_coordinate_views_for_normalized_bounds(
        self,
        bounds: NormalizedRectangle | None = None,
    ) -> tuple[CoordinateView | None, CoordinateView | None]:
        """Return authored pins, using ``None`` for each complete axis.

        Each axis is handled independently, so a complete singleton axis does
        not need an invented public coordinate span.
        """

        if bounds is None:
            x_limits, y_limits = self.x_limits, self.y_limits
        else:
            left, top, right, bottom = _unbounded_normalized_rectangle(bounds)
            x_start, x_stop = self._cached_axis_edges(self.x_axis)
            y_start, y_stop = self._cached_axis_edges(self.y_axis)
            x_limits = tuple(sorted((
                x_start + left * (x_stop - x_start),
                x_start + right * (x_stop - x_start),
            )))
            y_limits = tuple(sorted((
                y_start + top * (y_stop - y_start),
                y_start + bottom * (y_stop - y_start),
            )))
        x_view = None if x_limits == self.home_x_limits else x_limits
        y_view = None if y_limits == self.home_y_limits else y_limits
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

    def coordinate_for_visible_point(
        self,
        point: NormalizedPoint,
    ) -> tuple[float, float]:
        """Map one visible pointer position to continuous physical coordinates.

        This is deliberately different from ``sample_coordinates_for_visible_point``:
        a Cross cursor keeps the operator's exact clicked coordinate while the
        nearest sample index is used only to read an optional value.
        """

        full_x, full_y = self.full_point_for_visible_point(point)
        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
        return (
            x_start + full_x * (x_stop - x_start),
            y_start + full_y * (y_stop - y_start),
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

    def coordinate_rectangle_for_visible_bounds(
        self,
        bounds: NormalizedRectangle,
    ) -> tuple[float, float, float, float]:
        """Map visible-window bounds to continuous axis coordinates.

        These bounds live in the currently painted axes square, where ``0..1``
        describes the *visible window*.  They are deliberately not the same
        coordinate space as a standing image selection, whose normalized
        bounds describe the complete source raster and remain inside that
        raster even when zoom or pan exposes axes background.
        """

        left, top, right, bottom = validate_normalized_rectangle(bounds)

        first_x, first_y = self.coordinate_for_visible_point((left, top))
        second_x, second_y = self.coordinate_for_visible_point((right, bottom))
        x_low, x_high = sorted((first_x, second_x))
        y_low, y_high = sorted((first_y, second_y))
        return x_low, y_low, x_high, y_high

    def coordinate_rectangle_for_full_bounds(
        self,
        bounds: NormalizedRectangle,
    ) -> tuple[float, float, float, float]:
        """Map complete-raster bounds to continuous axis coordinates.

        A Figure Area is stored in this source-relative coordinate space so it
        is invariant under later zoom/pan.  The viewport itself may extend
        beyond the source, but an Area cannot: accepting unbounded viewport
        bounds here would silently turn display background into data
        authority.
        """

        left, top, right, bottom = validate_normalized_rectangle(bounds)
        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
        first_x = x_start + left * (x_stop - x_start)
        second_x = x_start + right * (x_stop - x_start)
        first_y = y_start + top * (y_stop - y_start)
        second_y = y_start + bottom * (y_stop - y_start)
        x_low, x_high = sorted((first_x, second_x))
        y_low, y_high = sorted((first_y, second_y))
        return x_low, y_low, x_high, y_high

    def unbounded_visible_point_for_coordinate(
        self,
        coordinate_xy: object,
        *,
        coordinate_frame: CoordinateFrameId | None,
    ) -> tuple[float, float]:
        """Map one physical point through this view without clipping it.

        The returned values use the visible window as ``[0, 1] x [0, 1]``,
        but may lie outside that square.  This is the geometry needed for a
        vector overlay whose center can be off-screen while the raster remains
        clipped by the presentation surface.  Axis direction and half-cell
        edges come from the same exact affine geometry as the bounded APIs.
        """

        if not isinstance(coordinate_xy, tuple) or len(coordinate_xy) != 2:
            raise TypeError("coordinate_xy must be a two-item tuple")
        if coordinate_frame is None:
            if (
                self.x_axis.coordinate_frame is not None
                or self.y_axis.coordinate_frame is not None
            ):
                raise ValueError(
                    "unframed coordinates require two explicitly unframed axes"
                )
        elif not isinstance(coordinate_frame, CoordinateFrameId):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")
        elif coordinate_frame != self.coordinate_frame:
            raise ValueError("coordinate belongs to another coordinate frame")
        coordinate_x = _finite_number(coordinate_xy[0], "x coordinate")
        coordinate_y = _finite_number(coordinate_xy[1], "y coordinate")
        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
        left, top, right, bottom = self.visible_bounds
        visible_x = (
            (coordinate_x - x_start) / (x_stop - x_start) - left
        ) / (right - left)
        visible_y = (
            (coordinate_y - y_start) / (y_stop - y_start) - top
        ) / (bottom - top)
        if not math.isfinite(visible_x) or not math.isfinite(visible_y):
            raise ValueError("coordinate mapping must be finite")
        return (
            0.0 if visible_x == 0.0 else visible_x,
            0.0 if visible_y == 0.0 else visible_y,
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
            start, stop = self._cached_axis_edges(axis)
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
        x_start, x_stop = self._cached_axis_edges(self.x_axis)
        y_start, y_stop = self._cached_axis_edges(self.y_axis)
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

    def clipped_full_bounds_for_visible_bounds(
        self,
        bounds: NormalizedRectangle,
    ) -> NormalizedRectangle | None:
        """Project a visible drag onto source data, or return no Area.

        Visible axes may contain background after square padding, zoom-out, or
        pan.  Rectangle gestures begin in that visible coordinate space, while
        the retained Figure Area must stay in complete-raster ``[0, 1]``
        coordinates.  This is the single semantic intersection between those
        spaces; Qt paint code neither clamps nor guesses data bounds.
        """

        left, top, right, bottom = validate_normalized_rectangle(bounds)
        full_left, full_top = self.full_point_for_visible_point((left, top))
        full_right, full_bottom = self.full_point_for_visible_point(
            (right, bottom)
        )
        clipped = (
            max(0.0, full_left),
            max(0.0, full_top),
            min(1.0, full_right),
            min(1.0, full_bottom),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            return None
        return validate_normalized_rectangle(clipped)

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
        """Resolve a typed rectangle to complete-raster cell edges."""

        if not isinstance(selection, Selection):
            raise TypeError("selection must be zlc_data.Selection")
        terms = {term.axis_id: term for term in selection.terms}
        expected = {self.x_axis.axis_id, self.y_axis.axis_id}
        if set(terms) != expected or any(
            not isinstance(term, (CoordinateRangeSelection, IndexRangeSelection))
            for term in terms.values()
        ):
            raise ValueError("image selection must be one typed IMAGE rectangle")
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
            raise ValueError("image rectangle must resolve to contiguous raster cells")
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
        """Convert one continuous complete-raster rectangle to a Selection.

        Painting remains faithful to the operator's unsnapped RectangleSelector
        extents.  Dataset selection resolves those named coordinate ranges only
        when a consumer asks for selected values.
        """

        x_low, y_low, x_high, y_high = self.coordinate_rectangle_for_full_bounds(
            bounds
        )
        return Selection(
            (
                CoordinateRangeSelection(
                    self.x_axis.axis_id,
                    x_low,
                    x_high,
                    self.x_axis.coordinate_frame,
                ),
                CoordinateRangeSelection(
                    self.y_axis.axis_id,
                    y_low,
                    y_high,
                    self.y_axis.coordinate_frame,
                ),
            )
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

    def snapped_bounds_for_visible_rectangle(
        self,
        bounds: NormalizedRectangle,
    ) -> NormalizedRectangle:
        """Snap visible rectangle *edges* to complete-raster cell edges.

        Unlike :meth:`snapped_bounds_for_drag`, both inputs here are already
        rectangle edges rather than two sampled pixels.  This is the stable
        path for moving or resizing a standing selector: an unchanged right
        or bottom edge must not grow the selection by one cell.
        """

        left, top, right, bottom = validate_normalized_rectangle(bounds)
        full_left, full_top = self.full_point_for_visible_point((left, top))
        full_right, full_bottom = self.full_point_for_visible_point(
            (right, bottom)
        )

        def snapped_edges(
            start: float,
            stop: float,
            size: int,
        ) -> tuple[int, int]:
            tolerance = 1e-12
            first = max(0, min(size - 1, math.floor(start * size + tolerance)))
            last = max(
                first + 1,
                min(size, math.ceil(stop * size - tolerance)),
            )
            return first, last

        x_start, x_stop = snapped_edges(
            full_left,
            full_right,
            self.x_axis.size,
        )
        y_start, y_stop = snapped_edges(
            full_top,
            full_bottom,
            self.y_axis.size,
        )
        return validate_normalized_rectangle(
            (
                x_start / self.x_axis.size,
                y_start / self.y_axis.size,
                x_stop / self.x_axis.size,
                y_stop / self.y_axis.size,
            )
        )

    def with_visible_bounds(
        self,
        bounds: NormalizedRectangle,
        *,
        viewport_revision: int | None = None,
    ) -> ImageViewportTransform:
        """Commit normalized raster bounds as physical limits once."""

        checked = _unbounded_normalized_rectangle(bounds)
        if checked == self.visible_bounds:
            return self
        revision = self._replacement_revision(viewport_revision)
        return self._replacement_with_visible_bounds(checked, revision)

    def home(
        self,
        *,
        viewport_revision: int | None = None,
    ) -> ImageViewportTransform:
        """Return Main's once-padded square home limits."""

        if self.x_limits == self.home_x_limits and self.y_limits == self.home_y_limits:
            return self
        revision = self._replacement_revision(viewport_revision)
        return self._validated_replacement(
            self.home_x_limits,
            self.home_y_limits,
            revision,
        )

    def centered_zoom(
        self,
        point: NormalizedPoint,
        span_scale: float,
        *,
        viewport_revision: int | None = None,
    ) -> ImageViewportTransform:
        """Scale both spans around one visible point.

        ``span_scale < 1`` zooms in and ``span_scale > 1`` zooms out.  Like
        Main's physical axes, the result is not cropped to the source raster.
        """

        visible_x, visible_y = validate_normalized_point(point, "zoom point")
        scale = _finite_number(span_scale, "span_scale")
        if scale <= 0.0:
            raise ValueError("span_scale must be positive")
        if scale == 1.0:
            return self
        anchor_x, anchor_y = self.coordinate_for_visible_point((visible_x, visible_y))
        x_low, x_high = self.x_limits
        y_low, y_high = self.y_limits
        candidate = (
            (
                anchor_x - (anchor_x - x_low) * scale,
                anchor_x + (x_high - anchor_x) * scale,
            ),
            (
                anchor_y - (anchor_y - y_low) * scale,
                anchor_y + (y_high - anchor_y) * scale,
            ),
        )
        checked_x = _coordinate_view(candidate[0], "zoom x_limits")
        checked_y = _coordinate_view(candidate[1], "zoom y_limits")
        revision = self._replacement_revision(viewport_revision)
        return self._validated_replacement(checked_x, checked_y, revision)

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
        return self.with_visible_bounds(
            (
                left - delta_x * span_x / width,
                top - delta_y * span_y / height,
                right - delta_x * span_x / width,
                bottom - delta_y * span_y / height,
            ),
            viewport_revision=viewport_revision,
        )

    def _replacement_revision(self, revision: int | None) -> int:
        if revision is None:
            return self.viewport_revision + 1
        checked = nonnegative_integer(revision, "viewport_revision")
        if checked <= self.viewport_revision:
            raise ValueError("replacement viewport_revision must increase")
        return checked


def _effective_image_axis(axis: EvaluatedAxis) -> AxisSpec:
    if len(set(axis.indices)) != len(axis.indices):
        raise ValueError(
            f"evaluated IMAGE axis {axis.axis_id} contains duplicate source indices"
        )
    return AxisSpec(
        axis_id=axis.axis_id,
        name=axis.name,
        role=axis.role,
        size=len(axis.indices),
        coordinates=axis.coordinates,
        unit=axis.unit,
        coordinate_frame=axis.coordinate_frame,
    )


def image_viewport_for_evaluated_image(
    image: EvaluatedImage,
) -> ImageViewportTransform:
    """Build the exact numeric viewport admitted by an evaluated IMAGE DTO.

    ``EvaluatedImage.x_axis`` and ``y_axis`` are the evaluator's resolved
    ``IMAGE_X`` and ``IMAGE_Y`` bindings.  They supply display direction while
    each declared domain role remains authoritative and unchanged.  The
    projection never infers either fact from rank, shape, singleton axes, or
    fit metadata.  :class:`AxisSpec` and :class:`ImageViewportTransform` then
    fail closed on nonnumeric, nonfinite, duplicate, or irregular coordinate
    metadata.
    """

    if not isinstance(image, EvaluatedImage):
        raise TypeError("image must be EvaluatedImage")
    x_axis = _effective_image_axis(image.x_axis)
    y_axis = _effective_image_axis(image.y_axis)
    return ImageViewportTransform(
        (x_axis, y_axis),
        display_axis_ids=(x_axis.axis_id, y_axis.axis_id),
    )


__all__ = [
    "CoordinateView",
    "ImageViewportTransform",
    "NormalizedPoint",
    "NormalizedRectangle",
    "image_viewport_for_evaluated_image",
    "validate_normalized_point",
    "validate_normalized_rectangle",
]
