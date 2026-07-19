"""Unbounded physical geometry for the saved-fit grid image overlay."""

from __future__ import annotations

import pytest

from zlc_data import AxisId, AxisSpec, CoordinateFrameId, SPATIAL_X, SPATIAL_Y
from zlc_frontend.image_view import ImageViewportTransform


def _viewport(
    *,
    descending_x: bool = False,
    descending_y: bool = False,
    visible_bounds=(0.0, 0.0, 1.0, 1.0),
) -> tuple[ImageViewportTransform, CoordinateFrameId]:
    frame = CoordinateFrameId("w7-fit-grid-camera")
    x_coordinates = (10.0, 12.0, 14.0, 16.0)
    y_coordinates = (20.0, 24.0, 28.0, 32.0)
    if descending_x:
        x_coordinates = tuple(reversed(x_coordinates))
    if descending_y:
        y_coordinates = tuple(reversed(y_coordinates))
    x_axis = AxisSpec(
        AxisId("w7-fit-grid.x"),
        "camera x",
        SPATIAL_X,
        len(x_coordinates),
        x_coordinates,
        unit="pixel",
        coordinate_frame=frame,
    )
    y_axis = AxisSpec(
        AxisId("w7-fit-grid.y"),
        "camera y",
        SPATIAL_Y,
        len(y_coordinates),
        y_coordinates,
        unit="pixel",
        coordinate_frame=frame,
    )
    return (
        ImageViewportTransform(
            (y_axis, x_axis),
            viewport_revision=4,
            visible_bounds=visible_bounds,
        ),
        frame,
    )


@pytest.mark.parametrize(
    ("descending_x", "descending_y", "coordinate", "expected"),
    [
        (False, False, (11.0, 22.0), (0.25, 0.25)),
        (True, False, (11.0, 22.0), (0.75, 0.25)),
        (False, True, (11.0, 22.0), (0.25, 0.75)),
        (True, True, (11.0, 22.0), (0.75, 0.75)),
    ],
)
def test_unbounded_visible_point_uses_declared_axis_edge_direction(
    descending_x,
    descending_y,
    coordinate,
    expected,
):
    viewport, frame = _viewport(
        descending_x=descending_x,
        descending_y=descending_y,
    )

    assert viewport.unbounded_visible_point_for_coordinate(
        coordinate,
        coordinate_frame=frame,
    ) == pytest.approx(expected)


def test_unbounded_visible_point_projects_through_zoom_without_clamping():
    viewport, frame = _viewport(visible_bounds=(0.25, 0.125, 0.75, 0.625))

    assert viewport.unbounded_visible_point_for_coordinate(
        (13.0, 22.0),
        coordinate_frame=frame,
    ) == pytest.approx((0.5, 0.25))
    assert viewport.unbounded_visible_point_for_coordinate(
        (1.0, 39.0),
        coordinate_frame=frame,
    ) == pytest.approx((-2.5, 2.375))


@pytest.mark.parametrize(
    ("coordinate", "error", "message"),
    [
        ([13.0, 23.0], TypeError, "two-item tuple"),
        ((True, 23.0), TypeError, "numeric"),
        ((13.0, object()), TypeError, "numeric"),
        ((float("nan"), 23.0), ValueError, "finite"),
        ((13.0, float("inf")), ValueError, "finite"),
    ],
)
def test_unbounded_visible_point_rejects_invalid_coordinates(
    coordinate,
    error,
    message,
):
    viewport, frame = _viewport()

    with pytest.raises(error, match=message):
        viewport.unbounded_visible_point_for_coordinate(
            coordinate,
            coordinate_frame=frame,
        )


def test_unbounded_visible_point_rejects_wrong_or_untyped_frame():
    viewport, frame = _viewport()

    with pytest.raises(ValueError, match="another coordinate frame"):
        viewport.unbounded_visible_point_for_coordinate(
            (13.0, 23.0),
            coordinate_frame=CoordinateFrameId("another-camera"),
        )
    with pytest.raises(TypeError, match="CoordinateFrameId"):
        viewport.unbounded_visible_point_for_coordinate(
            (13.0, 23.0),
            coordinate_frame=str(frame),  # type: ignore[arg-type]
        )
