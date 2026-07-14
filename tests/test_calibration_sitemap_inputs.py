"""Independent input contracts for the retained site-map calibration function."""

from __future__ import annotations

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.operations import calibration as opcal


def _grid_frame(grid, image_shape, *, seed=0, sigma=1.6, background=200.0):
    ny, nx = grid
    height, width = image_shape
    ys = np.linspace(0.18 * height, 0.82 * height, ny)
    xs = np.linspace(0.18 * width, 0.82 * width, nx)
    rng = np.random.default_rng(seed)
    frame = np.full((height, width), background) + rng.normal(0.0, 4.0, (height, width))
    yy, xx = np.mgrid[0:height, 0:width]
    for cy in ys:
        for cx in xs:
            frame += 700.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return frame


def test_provided_centers_skip_detection_and_are_count_checked(monkeypatch):
    grid = (2, 3)
    frame = _grid_frame(grid, (60, 90))
    centers = opcal.find_site_centers(frame, grid)

    calls = 0
    real = opcal.find_site_centers

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(opcal, "find_site_centers", spy)
    result = opcal.calibrate_sitemap_from_images(
        [frame], grid_shape=grid, method="box", display=False, centers=centers)
    assert calls == 0
    np.testing.assert_allclose(result.calibration.centers, centers)

    with pytest.raises(ValueError, match="centers count"):
        opcal.calibrate_sitemap_from_images(
            [frame], grid_shape=grid, method="box", display=False, centers=centers[:-1])
