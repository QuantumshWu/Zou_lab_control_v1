"""``calibrate_all_methods_from_images`` detects the site map ONCE and shares it.

Centers are a property of the atom lattice, not the readout method, so the expensive
``find_site_centers`` (filters + per-site 2D-Gaussian refine) runs a single time -- was three
times, once per box/psf/uniform_psf -- and every readout method the calibration carries reads off
that ONE set of centers (a single source, not three detections that only happen to agree).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.operations import calibration as opcal
from Zou_lab_control.neutral_atom.operations.calibration import (
    ALL_READOUT_METHODS,
    calibrate_all_methods_from_images,
)


def _grid_frame(grid, image_shape, amps, *, seed, sigma=1.6, background=200.0):
    """A Gaussian spot per trap on a noisy background -- a synthetic averaged reference frame."""
    ny, nx = grid
    h, w = image_shape
    ys = np.linspace(0.18 * h, 0.82 * h, ny)
    xs = np.linspace(0.18 * w, 0.82 * w, nx)
    rng = np.random.default_rng(seed)
    img = np.full((h, w), float(background)) + rng.normal(0.0, 4.0, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.asarray(amps, dtype=float).reshape(ny, nx)
    for r, cy in enumerate(ys):
        for c, cx in enumerate(xs):
            img += a[r, c] * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return img


def test_site_map_detected_once_and_shared(monkeypatch):
    grid = (3, 4)
    shape = (72, 96)
    n = grid[0] * grid[1]
    bright = np.full(n, 700.0)
    reference = [_grid_frame(grid, shape, bright, seed=s) for s in range(3)]      # all sites loaded
    # readout frames: each site bright in half the frames, dark in the other half -> a clean per-site
    # bimodal split for the threshold stage (and nothing to detect centers from -- that is the point).
    readout = [_grid_frame(grid, shape,
                           np.array([700.0 if (i + s) % 2 == 0 else 0.0 for i in range(n)]),
                           seed=100 + s)
               for s in range(6)]

    calls = {"n": 0}
    real = opcal.find_site_centers

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(opcal, "find_site_centers", spy)

    cal = calibrate_all_methods_from_images(reference, readout, grid_shape=grid)

    assert calls["n"] == 1, f"find_site_centers ran {calls['n']}x (want 1: detect once, share across methods)"
    # the one calibration reads with every method, all off the SAME top-level centers
    assert set(cal.methods()) == set(ALL_READOUT_METHODS)
    assert cal.n_sites == n
    # by_method carries per-method thresholds/kernels but NOT its own centers -> one center source
    for entry in (cal.by_method or {}).values():
        assert "centers" not in entry


def test_provided_centers_skip_detection_and_are_count_checked(monkeypatch):
    """Passing ``centers`` into ``calibrate_sitemap_from_images`` bypasses ``find_site_centers``
    entirely, and a wrong count is rejected (not silently mis-shaped)."""
    import pytest

    grid = (2, 3)
    shape = (60, 90)
    frame = _grid_frame(grid, shape, np.full(grid[0] * grid[1], 700.0), seed=0)
    centers = opcal.find_site_centers(frame, grid)

    calls = {"n": 0}
    real = opcal.find_site_centers

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(opcal, "find_site_centers", spy)
    result = opcal.calibrate_sitemap_from_images([frame], grid_shape=grid, method="box",
                                                 display=False, centers=centers)
    assert calls["n"] == 0, "provided centers must skip find_site_centers"
    np.testing.assert_allclose(result.calibration.centers, centers)

    with pytest.raises(ValueError, match="centers count"):
        opcal.calibrate_sitemap_from_images([frame], grid_shape=grid, method="box",
                                            display=False, centers=centers[:-1])
