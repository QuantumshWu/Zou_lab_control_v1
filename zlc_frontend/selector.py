"""Headless, front-bound rectangle selection for regular pixel images.

This module owns no Qt, Matplotlib, runtime control, or analysis authority.
It only converts between one explicitly named two-axis pixel viewport and a
typed :class:`zlc_data.Selection`.  Normalized rectangles use raster pixel-cell
edges: ``(0, 0, 1, 1)`` covers the complete image.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TypeAlias

from zlc_data import (
    AxisSpec,
    CoordinateFrameId,
    CoordinateRangeSelection,
    Selection,
    SPATIAL_X,
    SPATIAL_Y,
    resolve_selection_indices,
)
from zlc_storage import canonical_text, nonnegative_integer

from .render import SourceIdentity


NormalizedRectangle: TypeAlias = tuple[float, float, float, float]


def _normalized_rectangle(value: object) -> NormalizedRectangle:
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("normalized bounds must be a four-item tuple")
    names = ("left", "top", "right", "bottom")
    normalized: list[float] = []
    for name, item in zip(names, value, strict=True):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"normalized {name} must be numeric")
        number = float(item)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"normalized {name} must be finite and inside [0, 1]")
        normalized.append(0.0 if number == 0.0 else number)
    left, top, right, bottom = normalized
    if not left < right or not top < bottom:
        raise ValueError("normalized bounds must contain a non-degenerate rectangle")
    return left, top, right, bottom


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


def _edge_index(value: float, size: int, name: str) -> int:
    scaled = value * size
    candidate = int(round(scaled))
    # Bounds emitted by this module are integer pixel edges divided by size.
    # The tolerance only absorbs that division's floating-point round-trip; it
    # never relaxes the exact-regular-axis admission above.
    if not math.isclose(scaled, candidate, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"normalized {name} is not aligned to a pixel-cell edge")
    if not 0 <= candidate <= size:
        raise ValueError(f"normalized {name} lies outside the raster")
    return candidate


@dataclass(frozen=True, slots=True)
class RectangleGesture:
    """One immutable rectangle tied to the exact front on which it was drawn."""

    panel_id: str
    board_id: str
    layout_generation: int
    sequence: int
    source_identity: SourceIdentity
    normalized_bounds: NormalizedRectangle
    viewport_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", canonical_text(self.panel_id, "panel_id"))
        object.__setattr__(self, "board_id", canonical_text(self.board_id, "board_id"))
        for field in ("layout_generation", "sequence", "viewport_revision"):
            object.__setattr__(
                self,
                field,
                nonnegative_integer(getattr(self, field), field),
            )
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be zlc_frontend.render.SourceIdentity")
        object.__setattr__(
            self,
            "normalized_bounds",
            _normalized_rectangle(self.normalized_bounds),
        )


@dataclass(frozen=True, slots=True)
class ImageViewportTransform:
    """Exact affine map for one explicitly named two-axis pixel image.

    ``axes`` must contain exactly one SPATIAL_X and one SPATIAL_Y axis.  Rank,
    singleton shape, or tuple position never supplies an axis role.
    """

    axes: tuple[AxisSpec, AxisSpec]
    viewport_revision: int = 0

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
        object.__setattr__(self, "axes", (x_axis, y_axis))
        object.__setattr__(
            self,
            "viewport_revision",
            nonnegative_integer(self.viewport_revision, "viewport_revision"),
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
    def coordinate_frame(self) -> CoordinateFrameId:
        assert self.x_axis.coordinate_frame is not None
        return self.x_axis.coordinate_frame

    def normalized_bounds_for_selection(
        self,
        selection: Selection,
    ) -> NormalizedRectangle:
        """Resolve a typed rectangle to the exact pixel cells it includes."""

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
        return _normalized_rectangle(
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
        """Convert an edge-aligned normalized rectangle to a closed Selection."""

        left, top, right, bottom = _normalized_rectangle(bounds)
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
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> NormalizedRectangle:
        """Snap two normalized raster points to their inclusive pixel cells."""

        if (
            not isinstance(start, tuple)
            or len(start) != 2
            or not isinstance(end, tuple)
            or len(end) != 2
        ):
            raise TypeError("drag endpoints must be normalized (x, y) tuples")
        points: list[float] = []
        for name, value in zip(("start x", "start y", "end x", "end y"), (*start, *end)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be finite and inside [0, 1]")
            points.append(number)
        start_x, start_y, end_x, end_y = points
        x0 = min(self.x_axis.size - 1, math.floor(start_x * self.x_axis.size))
        x1 = min(self.x_axis.size - 1, math.floor(end_x * self.x_axis.size))
        y0 = min(self.y_axis.size - 1, math.floor(start_y * self.y_axis.size))
        y1 = min(self.y_axis.size - 1, math.floor(end_y * self.y_axis.size))
        return _normalized_rectangle(
            (
                min(x0, x1) / self.x_axis.size,
                min(y0, y1) / self.y_axis.size,
                (max(x0, x1) + 1) / self.x_axis.size,
                (max(y0, y1) + 1) / self.y_axis.size,
            )
        )


__all__ = [
    "ImageViewportTransform",
    "NormalizedRectangle",
    "RectangleGesture",
]
