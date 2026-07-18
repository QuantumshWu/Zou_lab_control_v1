"""Pure contracts for the front-bound image viewport transform."""

from __future__ import annotations

import random

import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_frontend.image_view import ImageViewportTransform


def _viewport(
    *,
    width: int = 8,
    height: int = 6,
    revision: int = 7,
    visible_bounds=(0.0, 0.0, 1.0, 1.0),
) -> ImageViewportTransform:
    frame = CoordinateFrameId("image-view-test")
    x_axis = AxisSpec(
        AxisId("image-view.x"),
        "x",
        SPATIAL_X,
        width,
        tuple(10 + 2 * index for index in range(width)),
        unit="pixel",
        coordinate_frame=frame,
    )
    y_axis = AxisSpec(
        AxisId("image-view.y"),
        "y",
        SPATIAL_Y,
        height,
        tuple(100 - 3 * index for index in range(height)),
        unit="pixel",
        coordinate_frame=frame,
    )
    return ImageViewportTransform(
        (y_axis, x_axis),
        viewport_revision=revision,
        visible_bounds=visible_bounds,
    )


def test_selector_and_package_exports_share_the_single_image_view_owner():
    from zlc_frontend import ImageViewportTransform as package_transform
    from zlc_frontend.selector import ImageViewportTransform as selector_transform

    assert package_transform is ImageViewportTransform
    assert selector_transform is ImageViewportTransform


def test_full_visible_point_and_bounds_mapping_round_trip_and_clip():
    viewport = _viewport(visible_bounds=(0.25, 0.125, 0.75, 0.875))

    assert viewport.full_point_for_visible_point((0.0, 0.0)) == (0.25, 0.125)
    assert viewport.full_point_for_visible_point((1.0, 1.0)) == (0.75, 0.875)
    point = (0.31, 0.73)
    assert viewport.visible_point_for_full_point(
        viewport.full_point_for_visible_point(point)
    ) == pytest.approx(point)

    visible_bounds = (0.2, 0.4, 0.8, 0.6)
    full_bounds = (
        *viewport.full_point_for_visible_point(visible_bounds[:2]),
        *viewport.full_point_for_visible_point(visible_bounds[2:]),
    )
    assert full_bounds == pytest.approx((0.35, 0.425, 0.65, 0.575))
    assert viewport.visible_bounds_for_full_bounds(full_bounds) == pytest.approx(
        visible_bounds
    )
    assert viewport.clipped_visible_bounds_for_full_bounds(
        (0.0, 0.0, 0.5, 0.5)
    ) == pytest.approx((0.0, 0.0, 0.5, 0.5))
    assert viewport.clipped_visible_bounds_for_full_bounds(
        (0.0, 0.0, 0.2, 0.1)
    ) is None
    with pytest.raises(ValueError, match="outside the visible window"):
        viewport.visible_bounds_for_full_bounds((0.0, 0.0, 0.5, 0.5))


def test_coordinate_views_map_pixel_edges_for_ascending_and_descending_axes():
    viewport = _viewport(width=8, height=6)

    assert viewport.normalized_bounds_for_optional_coordinate_views(
        (9.0, 25.0),
        (83.5, 101.5),
    ) == (0.0, 0.0, 1.0, 1.0)
    bounds = viewport.normalized_bounds_for_optional_coordinate_views(
        (11.0, 21.0),
        (88.0, 97.0),
    )
    assert bounds == pytest.approx((0.125, 0.25, 0.75, 0.75))
    x_view, y_view = viewport.optional_coordinate_views_for_normalized_bounds(bounds)
    assert x_view == pytest.approx((11.0, 21.0))
    assert y_view == pytest.approx((88.0, 97.0))

    committed = ImageViewportTransform(viewport.axes, 19, bounds)
    assert committed.viewport_revision == 19
    assert committed.visible_bounds == pytest.approx(bounds)
    x_view, y_view = committed.optional_coordinate_views_for_normalized_bounds()
    assert x_view == pytest.approx((11.0, 21.0))
    assert y_view == pytest.approx((88.0, 97.0))


@pytest.mark.parametrize(
    ("x_view", "y_view", "error", "message"),
    [
        ((8.999, 25.0), (83.5, 101.5), ValueError, "outside"),
        ((9.0, 25.001), (83.5, 101.5), ValueError, "outside"),
        ((9.0, 25.0), (83.499, 101.5), ValueError, "outside"),
        ((9.0, 25.0), (83.5, 101.501), ValueError, "outside"),
        ((10.0, 10.0), (83.5, 101.5), ValueError, "below high"),
        ((9.0, 9.5), (83.5, 101.5), ValueError, "at least one raster cell"),
        ((9.0, 25.0), (90.0, 91.0), ValueError, "at least one raster cell"),
        ([9.0, 25.0], (83.5, 101.5), TypeError, "two-item tuple"),
        ((9.0, float("nan")), (83.5, 101.5), ValueError, "finite"),
    ],
)
def test_coordinate_views_fail_closed_for_invalid_or_unrepresentable_ranges(
    x_view,
    y_view,
    error,
    message,
):
    with pytest.raises(error, match=message):
        _viewport().normalized_bounds_for_optional_coordinate_views(x_view, y_view)


def test_coordinate_views_fail_closed_for_irregular_and_singleton_axes():
    frame = CoordinateFrameId("coordinate-view-invalid")
    irregular_x = AxisSpec(
        AxisId("coordinate-view-invalid.x"),
        "x",
        SPATIAL_X,
        4,
        (0.0, 1.0, 2.25, 3.0),
        unit="pixel",
        coordinate_frame=frame,
    )
    regular_y = AxisSpec(
        AxisId("coordinate-view-invalid.y"),
        "y",
        SPATIAL_Y,
        3,
        (0.0, 1.0, 2.0),
        unit="pixel",
        coordinate_frame=frame,
    )
    with pytest.raises(ValueError, match="not exactly regular"):
        ImageViewportTransform((irregular_x, regular_y))

    singleton_x = AxisSpec(
        AxisId("coordinate-view-singleton.x"),
        "x",
        SPATIAL_X,
        1,
        (42.0,),
        unit="pixel",
        coordinate_frame=frame,
    )
    singleton_viewport = ImageViewportTransform((singleton_x, regular_y))
    with pytest.raises(ValueError, match="at least two coordinates"):
        singleton_viewport.normalized_bounds_for_optional_coordinate_views(
            (41.5, 42.5),
            (-0.5, 2.5),
        )


def test_random_coordinate_view_round_trips_preserve_physical_limits():
    generator = random.Random(2026071801)
    viewport = _viewport(width=31, height=23)
    x_domain = (9.0, 71.0)
    y_domain = (32.5, 101.5)
    for _ in range(200):
        x_low = generator.uniform(x_domain[0], x_domain[1] - 2.0)
        x_high = generator.uniform(x_low + 2.0, x_domain[1])
        y_low = generator.uniform(y_domain[0], y_domain[1] - 3.0)
        y_high = generator.uniform(y_low + 3.0, y_domain[1])
        physical = ((x_low, x_high), (y_low, y_high))
        normalized = viewport.normalized_bounds_for_optional_coordinate_views(*physical)
        round_trip_x, round_trip_y = viewport.optional_coordinate_views_for_normalized_bounds(
            normalized
        )
        assert round_trip_x == pytest.approx(physical[0])
        assert round_trip_y == pytest.approx(physical[1])


def test_visible_edge_points_map_to_the_last_sample_inside_the_window():
    viewport = _viewport(visible_bounds=(0.25, 1 / 6, 0.75, 5 / 6))

    assert viewport.sample_indices_for_visible_point((0.0, 0.0)) == (1, 2)
    assert viewport.sample_coordinates_for_visible_point((0.0, 0.0)) == (14, 97)
    assert viewport.sample_indices_for_visible_point((1.0, 1.0)) == (4, 5)
    assert viewport.sample_coordinates_for_visible_point((1.0, 1.0)) == (20, 88)
    assert viewport.sample_indices_for_visible_point((0.5, 0.5)) == (3, 4)

    bounds = viewport.snapped_bounds_for_drag((0.0, 0.0), (1.0, 1.0))
    assert bounds == pytest.approx((2 / 8, 1 / 6, 6 / 8, 5 / 6))
    selection = viewport.selection_for_normalized_bounds(bounds)
    assert viewport.normalized_bounds_for_selection(selection) == pytest.approx(bounds)


def test_centered_zoom_preserves_anchor_clamps_to_home_and_one_cell():
    viewport = _viewport(width=100, height=80, revision=5)
    anchor = (0.25, 0.75)
    full_anchor = viewport.full_point_for_visible_point(anchor)

    zoomed = viewport.centered_zoom(anchor, 0.5)
    assert zoomed.visible_bounds == pytest.approx((0.125, 0.375, 0.625, 0.875))
    assert zoomed.viewport_revision == 6
    assert zoomed.full_point_for_visible_point(anchor) == pytest.approx(full_anchor)

    home = zoomed.centered_zoom(anchor, 100.0)
    assert home.visible_bounds == (0.0, 0.0, 1.0, 1.0)
    assert home.viewport_revision == 7

    one_cell = viewport.centered_zoom((0.5, 0.5), 1e-12)
    left, top, right, bottom = one_cell.visible_bounds
    assert right - left == pytest.approx(1 / 100)
    assert bottom - top == pytest.approx(1 / 80)
    assert viewport.centered_zoom(anchor, 1.0) is viewport
    with pytest.raises(ValueError, match="positive"):
        viewport.centered_zoom(anchor, 0.0)


def test_pan_uses_press_time_pixel_delta_preserves_span_and_clamps():
    viewport = _viewport(
        revision=3,
        visible_bounds=(0.25, 0.25, 0.75, 0.75),
    )

    panned = viewport.panned_by_pixels((20.0, -10.0), (200, 100))
    assert panned.visible_bounds == pytest.approx((0.2, 0.3, 0.7, 0.8))
    assert panned.viewport_revision == 4
    assert panned.visible_bounds[2] - panned.visible_bounds[0] == pytest.approx(0.5)
    assert panned.visible_bounds[3] - panned.visible_bounds[1] == pytest.approx(0.5)
    assert viewport.panned_by_pixels((0.0, 0.0), (200, 100)) is viewport

    assert viewport.panned_by_pixels((10_000.0, 10_000.0), (200, 100)).visible_bounds \
        == pytest.approx((0.0, 0.0, 0.5, 0.5))
    assert viewport.panned_by_pixels((-10_000.0, -10_000.0), (200, 100)).visible_bounds \
        == pytest.approx((0.5, 0.5, 1.0, 1.0))
    explicit = viewport.panned_by_pixels(
        (20.0, -10.0),
        (200, 100),
        viewport_revision=11,
    )
    assert explicit.viewport_revision == 11
    with pytest.raises(ValueError, match="must increase"):
        viewport.panned_by_pixels(
            (20.0, -10.0),
            (200, 100),
            viewport_revision=3,
        )


def test_visible_window_rejects_less_than_one_sample_cell():
    with pytest.raises(ValueError, match="at least one raster cell"):
        _viewport(width=8, visible_bounds=(0.0, 0.0, 0.1, 1.0))
    with pytest.raises(ValueError, match="at least one raster cell"):
        _viewport(height=6, visible_bounds=(0.0, 0.0, 1.0, 0.1))


def test_randomized_full_visible_round_trips_and_sample_cells_are_bounded():
    generator = random.Random(20260718)
    width, height = 17, 13
    for revision in range(1, 101):
        left = generator.uniform(0.0, 0.65)
        top = generator.uniform(0.0, 0.65)
        right = generator.uniform(left + 1 / width, 1.0)
        bottom = generator.uniform(top + 1 / height, 1.0)
        viewport = _viewport(
            width=width,
            height=height,
            revision=revision,
            visible_bounds=(left, top, right, bottom),
        )
        point = (generator.random(), generator.random())
        full = viewport.full_point_for_visible_point(point)
        assert viewport.visible_point_for_full_point(full) == pytest.approx(point)
        y_index, x_index = viewport.sample_indices_for_visible_point(point)
        assert 0 <= y_index < height
        assert 0 <= x_index < width

        x0, x1 = sorted((generator.random(), generator.random()))
        y0, y1 = sorted((generator.random(), generator.random()))
        if x0 == x1 or y0 == y1:
            continue
        visible = (x0, y0, x1, y1)
        mapped = (
            *viewport.full_point_for_visible_point(visible[:2]),
            *viewport.full_point_for_visible_point(visible[2:]),
        )
        assert viewport.visible_bounds_for_full_bounds(mapped) == pytest.approx(visible)
