"""One rectangle-selector gesture and paint primitive for every plot family."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias

from PyQt5 import QtCore, QtGui

from .style import (
    SELECTOR_ALPHA,
    SELECTOR_COLOR,
    SELECTOR_FONT_FAMILY,
    SELECTOR_FONT_PX,
    SELECTOR_HANDLE_PX,
    SELECTOR_LINE_PX,
)


RectangleHandle: TypeAlias = Literal[
    "C",
    "NW",
    "N",
    "NE",
    "W",
    "E",
    "SW",
    "S",
    "SE",
]
RectangleTuple: TypeAlias = tuple[float, float, float, float]


def selector_precision(span: float) -> int:
    """Digits needed to resolve one thousandth of the visible span."""

    gap = abs(span) / 1000 if span else 0.01
    return max(0, -int(math.ceil(math.log10(gap))))


def selector_pen_color(
    value: QtGui.QColor | str | None = None,
) -> QtGui.QColor:
    color = QtGui.QColor(SELECTOR_COLOR if value is None else value)
    color.setAlpha(SELECTOR_ALPHA)
    return color


def normalized_rectangle(rectangle: RectangleTuple) -> RectangleTuple:
    """Sort one possibly inverted rectangle without inventing a minimum size."""

    if not isinstance(rectangle, tuple) or len(rectangle) != 4:
        raise TypeError("rectangle must be a four-item tuple")
    left, top, right, bottom = (float(value) for value in rectangle)
    if any(
        not math.isfinite(value)
        for value in (left, top, right, bottom)
    ):
        raise ValueError("rectangle values must be finite")
    return (
        min(left, right),
        min(top, bottom),
        max(left, right),
        max(top, bottom),
    )


def rectangle_handle_points(
    rectangle: QtCore.QRectF,
) -> tuple[tuple[RectangleHandle, QtCore.QPointF], ...]:
    """Return the center plus the established eight resize handles."""

    center = rectangle.center()
    return (
        ("C", center),
        ("NW", rectangle.topLeft()),
        ("N", QtCore.QPointF(center.x(), rectangle.top())),
        ("NE", rectangle.topRight()),
        ("W", QtCore.QPointF(rectangle.left(), center.y())),
        ("E", QtCore.QPointF(rectangle.right(), center.y())),
        ("SW", rectangle.bottomLeft()),
        ("S", QtCore.QPointF(center.x(), rectangle.bottom())),
        ("SE", rectangle.bottomRight()),
    )


def hit_rectangle_handle(
    rectangle: QtCore.QRectF,
    point: QtCore.QPointF,
    *,
    grab_px: float = 10.0,
) -> RectangleHandle | None:
    """Hit-test with center priority, matching the established selector."""

    grab_px = float(grab_px)
    if not math.isfinite(grab_px) or grab_px <= 0.0:
        raise ValueError("rectangle handle grab radius must be positive")
    handles = rectangle_handle_points(rectangle)
    center = handles[0][1]
    if math.hypot(point.x() - center.x(), point.y() - center.y()) < 2.0 * grab_px:
        return "C"
    best_name: RectangleHandle | None = None
    best_distance: float | None = None
    for name, handle_point in handles[1:]:
        distance = math.hypot(
            point.x() - handle_point.x(),
            point.y() - handle_point.y(),
        )
        if best_distance is None or distance < best_distance:
            best_name = name
            best_distance = distance
    return (
        best_name
        if best_distance is not None and best_distance <= grab_px
        else None
    )


@dataclass(frozen=True, slots=True)
class RectangleDrag:
    """A stable rectangle plus the handle grabbed at gesture start.

    Coordinates are whatever normalized display space the caller uses.  Both
    image and numeric surfaces therefore get exactly the same center move,
    edge/corner resize, crossing, and clamp behavior while retaining their own
    data-coordinate authority.
    """

    initial: RectangleTuple
    handle: RectangleHandle
    grab_offset: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        initial = normalized_rectangle(self.initial)
        if self.handle not in {
            "C", "NW", "N", "NE", "W", "E", "SW", "S", "SE"
        }:
            raise ValueError("unknown rectangle handle")
        object.__setattr__(self, "initial", initial)
        if self.handle == "C":
            if (
                not isinstance(self.grab_offset, tuple)
                or len(self.grab_offset) != 2
            ):
                raise TypeError("center rectangle drag requires a grab offset")
            offset = tuple(float(value) for value in self.grab_offset)
            if any(not math.isfinite(value) for value in offset):
                raise ValueError("rectangle grab offset must be finite")
            object.__setattr__(self, "grab_offset", offset)
        elif self.grab_offset is not None:
            raise ValueError("only a center rectangle drag carries a grab offset")

    @classmethod
    def begin(
        cls,
        rectangle: RectangleTuple,
        handle: RectangleHandle,
        point: tuple[float, float],
    ) -> "RectangleDrag":
        initial = normalized_rectangle(rectangle)
        x, y = (float(value) for value in point)
        grab = (
            (x - initial[0], y - initial[1])
            if handle == "C"
            else None
        )
        return cls(initial, handle, grab)

    @classmethod
    def fresh(cls, point: tuple[float, float]) -> "RectangleDrag":
        x, y = (float(value) for value in point)
        return cls((x, y, x, y), "SE")

    def moved(
        self,
        point: tuple[float, float],
        *,
        clamp: RectangleTuple,
    ) -> RectangleTuple:
        """Move/resize continuously and clamp the result to the plot box."""

        x, y = (float(value) for value in point)
        clamp_left, clamp_top, clamp_right, clamp_bottom = normalized_rectangle(
            clamp
        )
        x = min(clamp_right, max(clamp_left, x))
        y = min(clamp_bottom, max(clamp_top, y))
        left, top, right, bottom = self.initial
        if self.handle == "C":
            assert self.grab_offset is not None
            width = right - left
            height = bottom - top
            new_left = x - self.grab_offset[0]
            new_top = y - self.grab_offset[1]
            new_left = min(
                clamp_right - width,
                max(clamp_left, new_left),
            )
            new_top = min(
                clamp_bottom - height,
                max(clamp_top, new_top),
            )
            return (
                new_left,
                new_top,
                new_left + width,
                new_top + height,
            )
        if "W" in self.handle:
            left = x
        if "E" in self.handle:
            right = x
        if "N" in self.handle:
            top = y
        if "S" in self.handle:
            bottom = y
        return normalized_rectangle((left, top, right, bottom))


def paint_rectangle_selector(
    painter: QtGui.QPainter,
    rectangle: QtCore.QRectF,
    *,
    handles: bool,
    dashed: bool = False,
    color: QtGui.QColor | str | None = None,
) -> None:
    """Draw the shared outline and white eight-handle appearance."""

    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen_color = selector_pen_color(color)
        pen = QtGui.QPen(pen_color, SELECTOR_LINE_PX)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(rectangle)
        if not handles:
            return
        painter.setPen(QtGui.QPen(pen_color, SELECTOR_LINE_PX / 2.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        half = SELECTOR_HANDLE_PX / 2.0
        for name, point in rectangle_handle_points(rectangle):
            if name == "C":
                continue
            painter.drawRect(
                QtCore.QRectF(
                    point.x() - half,
                    point.y() - half,
                    SELECTOR_HANDLE_PX,
                    SELECTOR_HANDLE_PX,
                )
            )
    finally:
        painter.restore()


def paint_selector_text(
    painter: QtGui.QPainter,
    label: str,
    plot: QtCore.QRectF,
    color: QtGui.QColor,
    *,
    corner: str,
) -> None:
    """Paint the established unboxed selector coordinate label."""

    painter.save()
    try:
        font = painter.font()
        font.setFamily(SELECTOR_FONT_FAMILY)
        font.setPixelSize(SELECTOR_FONT_PX)
        painter.setFont(font)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        painter.setPen(color)
        inset_x = 0.025 * plot.width()
        inset_y = 0.025 * plot.height()
        metrics = QtGui.QFontMetricsF(font)
        first_line = label.splitlines()[0] if label else " "
        ink = metrics.tightBoundingRect(first_line)
        ink_top_padding = metrics.ascent() + ink.top()
        area = QtCore.QRectF(
            plot.left() + inset_x,
            plot.top() + inset_y - ink_top_padding,
            plot.width() - 2 * inset_x,
            plot.height() - 2 * inset_y + ink_top_padding,
        )
        flags = QtCore.Qt.AlignTop | (
            QtCore.Qt.AlignRight
            if corner == "top_right"
            else QtCore.Qt.AlignLeft
        )
        painter.drawText(area, flags, label)
    finally:
        painter.restore()


def paint_selector_hover_label(
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
        label_bounds.moveTopRight(
            plot.topRight().toPoint() + QtCore.QPoint(-5, 5)
        )
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


__all__ = [
    "RectangleDrag",
    "RectangleHandle",
    "hit_rectangle_handle",
    "normalized_rectangle",
    "paint_rectangle_selector",
    "paint_selector_hover_label",
    "paint_selector_text",
    "rectangle_handle_points",
    "selector_precision",
    "selector_pen_color",
]
