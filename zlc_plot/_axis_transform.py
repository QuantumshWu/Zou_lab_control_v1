"""One immutable axes transform shared by native and raster interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import math

from ._validation import finite_real, integer, nonempty_text
from .selectors import CrosshairPoint


def _finite_tuple(
    values: Iterable[object],
    length: int,
    field: str,
) -> tuple[float, ...]:
    try:
        selected = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field} must be an iterable of numbers") from error
    if len(selected) != length:
        raise ValueError(f"{field} must contain {length} values")
    return tuple(finite_real(value, field) for value in selected)


def canvas_physical_size(canvas: Any) -> tuple[float, float]:
    """Return the physical-pixel extent used by Matplotlib pointer events."""

    get_width_height = getattr(canvas, "get_width_height", None)
    if not callable(get_width_height):
        raise TypeError("interactive canvas has no pixel-size query")
    try:
        size = get_width_height(physical=True)
    except TypeError:
        width, height = get_width_height()
        ratio = float(getattr(canvas, "device_pixel_ratio", 1.0))
        size = (float(width) * ratio, float(height) * ratio)
    try:
        width, height = map(float, size)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "interactive canvas pixel size must contain two numbers"
        ) from error
    if not math.isfinite(width) or not math.isfinite(height):
        raise ValueError("interactive canvas physical size must be positive and finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("interactive canvas physical size must be positive and finite")
    return width, height


@dataclass(frozen=True, slots=True)
class AxisTransform:
    """Exact top-origin plot box plus display and canonical coordinates."""

    role: str
    cell_index: int | None
    bounds: tuple[float, float, float, float]
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    canonical_x_limits: tuple[float, float]
    canonical_y_limits: tuple[float, float]

    def __post_init__(self) -> None:
        role = nonempty_text(self.role, "axes role")
        cell_index = integer(
            self.cell_index,
            "cell_index",
            minimum=0,
            optional=True,
        )
        bounds = _finite_tuple(self.bounds, 4, "axes bounds")
        left, top, right, bottom = bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError("axes bounds must be a non-empty top-origin unit box")
        x_limits = _finite_tuple(self.x_limits, 2, "x limits")
        y_limits = _finite_tuple(self.y_limits, 2, "y limits")
        if x_limits[0] == x_limits[1] or y_limits[0] == y_limits[1]:
            raise ValueError("axes limits must be non-degenerate")
        canonical_x = _finite_tuple(
            self.canonical_x_limits,
            2,
            "canonical x limits",
        )
        canonical_y = _finite_tuple(
            self.canonical_y_limits,
            2,
            "canonical y limits",
        )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "cell_index", cell_index)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "x_limits", x_limits)
        object.__setattr__(self, "y_limits", y_limits)
        object.__setattr__(self, "canonical_x_limits", canonical_x)
        object.__setattr__(self, "canonical_y_limits", canonical_y)

    def display_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        """Map display-space axes data into top-origin widget coordinates."""

        left, top, right, bottom = self.bounds
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        nx = left + (float(x) - x0) / (x1 - x0) * (right - left)
        ny = top + (float(y) - y1) / (y0 - y1) * (bottom - top)
        return nx, ny

    def canonical_from_normalized(self, nx: float, ny: float) -> CrosshairPoint:
        """Map top-origin normalized widget coordinates into canonical data."""

        left, top, right, bottom = self.bounds
        tx = (float(nx) - left) / (right - left)
        ty = (float(ny) - top) / (bottom - top)
        x0, x1 = self.canonical_x_limits
        y0, y1 = self.canonical_y_limits
        if self.role == "distribution":
            return CrosshairPoint(x1 + ty * (x0 - x1), 0.0)
        return CrosshairPoint(
            x0 + tx * (x1 - x0),
            y1 + ty * (y0 - y1),
        )

    def display_from_normalized(self, nx: float, ny: float) -> CrosshairPoint:
        """Map top-origin normalized widget coordinates into display data."""

        left, top, right, bottom = self.bounds
        tx = (float(nx) - left) / (right - left)
        ty = (float(ny) - top) / (bottom - top)
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        if self.role == "distribution":
            return CrosshairPoint(y1 + ty * (y0 - y1), 0.0)
        return CrosshairPoint(
            x0 + tx * (x1 - x0),
            y1 + ty * (y0 - y1),
        )

    @staticmethod
    def _event_normalized(event: Any, canvas: Any) -> tuple[float, float] | None:
        pixel_x = getattr(event, "x", None)
        pixel_y = getattr(event, "y", None)
        if pixel_x is None or pixel_y is None:
            return None
        width, height = canvas_physical_size(canvas)
        return float(pixel_x) / width, 1.0 - float(pixel_y) / height

    def canonical_point(self, event: Any, canvas: Any) -> CrosshairPoint | None:
        """Map one Matplotlib event into canonical data coordinates."""

        normalized = self._event_normalized(event, canvas)
        return (
            None
            if normalized is None
            else self.canonical_from_normalized(*normalized)
        )

    def display_point(self, event: Any, canvas: Any) -> CrosshairPoint | None:
        """Map one Matplotlib event into display-space data coordinates."""

        normalized = self._event_normalized(event, canvas)
        return None if normalized is None else self.display_from_normalized(*normalized)

    def display_to_pixel(
        self,
        x: float,
        y: float,
        canvas: Any,
    ) -> tuple[float, float]:
        """Map display-space data into bottom-origin physical canvas pixels."""

        width, height = canvas_physical_size(canvas)
        nx, ny = self.display_to_normalized(x, y)
        return nx * width, (1.0 - ny) * height


__all__ = ["AxisTransform", "canvas_physical_size"]
