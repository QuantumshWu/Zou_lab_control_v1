"""Shared pure x-axis geometry for numeric raster interactions."""

from __future__ import annotations

from .display_range import DisplayRange, validated_display_range
from zlc_storage import finite_real


NormalizedPlotBounds = tuple[float, float, float, float]


def normalized_plot_bounds(value: object) -> NormalizedPlotBounds:
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("plot_bounds must be (left, top, right, bottom)")
    left, top, right, bottom = (
        finite_real(item, f"plot_bounds[{index}]")
        for index, item in enumerate(value)
    )
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError("plot_bounds must be a nonempty top-origin unit rectangle")
    return left, top, right, bottom


def normalized_plot_contains(
    bounds: NormalizedPlotBounds,
    x: object,
    y: object,
) -> bool:
    left, top, right, bottom = normalized_plot_bounds(bounds)
    x = finite_real(x, "widget x")
    y = finite_real(y, "widget y")
    return left <= x <= right and top <= y <= bottom


def normalized_widget_x_to_data(
    bounds: NormalizedPlotBounds,
    limits: DisplayRange,
    x: object,
) -> float:
    left, _top, right, _bottom = normalized_plot_bounds(bounds)
    low, high = validated_display_range(limits, "x_limits")
    x = finite_real(x, "widget x")
    return low + (x - left) / (right - left) * (high - low)


def data_x_to_normalized_widget(
    bounds: NormalizedPlotBounds,
    limits: DisplayRange,
    x: object,
) -> float:
    left, _top, right, _bottom = normalized_plot_bounds(bounds)
    low, high = validated_display_range(limits, "x_limits")
    x = finite_real(x, "data x")
    return left + (x - low) / (high - low) * (right - left)


def zoomed_x_limits(
    limits: DisplayRange,
    anchor_x: object,
    factor: object,
) -> DisplayRange:
    low, high = validated_display_range(limits, "x_limits")
    anchor = finite_real(anchor_x, "zoom anchor x")
    factor = finite_real(factor, "zoom factor")
    if factor <= 0.0:
        raise ValueError("zoom factor must be positive")
    return validated_display_range(
        (
            anchor + (low - anchor) * factor,
            anchor + (high - anchor) * factor,
        ),
        "zoomed x limits",
    )


def panned_x_limits(
    bounds: NormalizedPlotBounds,
    limits: DisplayRange,
    press_widget_x: object,
    current_widget_x: object,
) -> DisplayRange:
    left, _top, right, _bottom = normalized_plot_bounds(bounds)
    low, high = validated_display_range(limits, "start_x_limits")
    press = finite_real(press_widget_x, "press widget x")
    current = finite_real(current_widget_x, "current widget x")
    shift = -(current - press) / (right - left) * (high - low)
    return validated_display_range(
        (low + shift, high + shift),
        "panned x limits",
    )


__all__ = [
    "NormalizedPlotBounds",
    "data_x_to_normalized_widget",
    "normalized_plot_bounds",
    "normalized_plot_contains",
    "normalized_widget_x_to_data",
    "panned_x_limits",
    "zoomed_x_limits",
]
