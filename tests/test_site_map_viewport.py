"""Exact camera-coordinate projection used by physical SiteMap overlays."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import AxisId, AxisSpec, CoordinateFrameId, SPATIAL_X, SPATIAL_Y
from zlc_frontend.image_view import ImageViewportTransform


def _viewport(*, visible=(0.0, 0.0, 1.0, 1.0)):
    frame = CoordinateFrameId("camera.roi-12.binning-2")
    x = AxisSpec(
        AxisId("site-map.frame-x"),
        "camera x",
        SPATIAL_X,
        4,
        (100.0, 102.0, 104.0, 106.0),
        unit="pixel",
        coordinate_frame=frame,
    )
    y = AxisSpec(
        AxisId("site-map.frame-y"),
        "camera y",
        SPATIAL_Y,
        4,
        (50.0, 46.0, 42.0, 38.0),
        unit="pixel",
        coordinate_frame=frame,
    )
    return ImageViewportTransform((y, x), 3, visible), frame


def test_site_centers_keep_subpixel_roi_binning_and_descending_axis_geometry():
    viewport, frame = _viewport()

    batch = viewport.full_points_for_coordinates(
        np.asarray(((100.0, 50.0), (103.0, 44.0), (106.75, 36.25))),
        coordinate_frame=frame,
    )
    np.testing.assert_allclose(
        batch,
        ((0.125, 0.125), (0.5, 0.5), (0.96875, 0.984375)),
    )
    assert batch.dtype == np.dtype("<f8")
    assert not batch.flags.writeable
    with pytest.raises(ValueError):
        batch.setflags(write=True)


def test_site_centers_project_through_the_committed_visible_window():
    viewport, frame = _viewport(visible=(0.25, 0.25, 0.75, 0.75))
    full = viewport.full_points_for_coordinates(
        np.asarray(((103.0, 44.0), (100.0, 50.0))),
        coordinate_frame=frame,
    )

    assert viewport.visible_point_for_full_point(tuple(full[0])) == pytest.approx(
        (0.5, 0.5)
    )
    with pytest.raises(ValueError, match="outside the visible window"):
        viewport.visible_point_for_full_point(tuple(full[1]))
    assert viewport.visible_span_for_coordinate_span(
        (2.0, 4.0), coordinate_frame=frame
    ) == pytest.approx((0.5, 0.5))


def test_site_coordinate_projection_rejects_wrong_frame_and_outside_values():
    viewport, frame = _viewport()

    with pytest.raises(ValueError, match="another coordinate frame"):
        viewport.full_points_for_coordinates(
            np.asarray(((103.0, 44.0),)),
            coordinate_frame=CoordinateFrameId("another-camera-roi"),
        )
    with pytest.raises(ValueError, match="another coordinate frame"):
        viewport.visible_span_for_coordinate_span(
            (2.0, 4.0),
            coordinate_frame=CoordinateFrameId("another-camera-roi"),
        )
    with pytest.raises(ValueError, match="positive"):
        viewport.visible_span_for_coordinate_span(
            (0.0, 4.0), coordinate_frame=frame
        )
    with pytest.raises(ValueError, match="outside axis"):
        viewport.full_points_for_coordinates(
            np.asarray(((103.0, 44.0), (98.5, 44.0))),
            coordinate_frame=frame,
        )
    with pytest.raises(TypeError, match="real numeric"):
        viewport.full_points_for_coordinates(
            np.asarray(((True, False),)),
            coordinate_frame=frame,
        )
    with pytest.raises(ValueError, match=r"shape \(points, 2\)"):
        viewport.full_points_for_coordinates(
            np.asarray((103.0, 44.0)), coordinate_frame=frame
        )
    with pytest.raises(ValueError, match="finite"):
        viewport.full_points_for_coordinates(
            np.asarray(((float("nan"), 44.0),)), coordinate_frame=frame
        )
