"""Contract: site detection spreads ONE center per trap across the whole array, even
when the reference template is NON-UNIFORM (each trap loaded a different number of shots,
so the dim traps sit below a relative brightness cutoff).

This pins the regression where ``find_site_centers`` fell back to the ``need`` brightest
PIXELS when fewer peaks cleared the relative cutoff -- the hottest blobs span several
pixels each, so every center collapsed onto a handful of spots and the site grid was
destroyed (the "site map completely broken" failure).  The fallback must instead pick the
``need`` brightest LOCAL MAXIMA, which are spatially separated (one per trap).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.analysis import find_site_centers


def _grid_template(grid_shape, image_shape, *, amplitudes, sigma=1.6, background=200.0):
    """A synthetic averaged reference: a Gaussian spot per trap on a constant background,
    with per-site ``amplitudes`` so some traps are much dimmer than others."""
    ny, nx = grid_shape
    h, w = image_shape
    ys = np.linspace(0.18 * h, 0.82 * h, ny)
    xs = np.linspace(0.18 * w, 0.82 * w, nx)
    img = np.full((h, w), float(background))
    yy, xx = np.mgrid[0:h, 0:w]
    truth = []
    amps = np.asarray(amplitudes, dtype=float).reshape(ny, nx)
    for r, cy in enumerate(ys):
        for c, cx in enumerate(xs):
            img += amps[r, c] * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
            truth.append((cx, cy))
    return img, np.asarray(truth, dtype=float)


def _is_one_to_one(detected, truth, *, tol):
    """Every true site owns exactly one detected center within ``tol`` (no two detected
    centers share a true site -> no clustering)."""
    assigned = set()
    for cx, cy in detected:
        d = np.hypot(truth[:, 0] - cx, truth[:, 1] - cy)
        j = int(np.argmin(d))
        if d[j] > tol or j in assigned:
            return False
        assigned.add(j)
    return len(assigned) == len(truth)


def test_find_site_centers_spreads_over_nonuniform_template():
    grid = (4, 5)
    # amplitudes spanning 6x: the dimmest traps fall well under the 0.35 relative cutoff,
    # which used to drop them and trigger the brightest-PIXEL clustering fallback.
    rng = np.random.default_rng(3)
    amps = rng.uniform(150.0, 900.0, size=grid[0] * grid[1])
    img, truth = _grid_template(grid, (80, 100), amplitudes=amps)

    centers = find_site_centers(img, grid)

    assert centers.shape == (grid[0] * grid[1], 2)
    # the cell pitch is ~12 px; a correct detection lands each center within ~half a cell.
    assert _is_one_to_one(centers, truth, tol=5.0), (
        "detected centers clustered instead of one-per-trap:\n" + np.array2string(centers))
    # no duplicate centers (the clustering signature was many identical coordinates).
    uniq = {(round(x), round(y)) for x, y in centers}
    assert len(uniq) == grid[0] * grid[1]


def test_find_site_centers_uniform_template_still_one_to_one():
    grid = (3, 6)
    img, truth = _grid_template(grid, (72, 132), amplitudes=np.full(grid[0] * grid[1], 700.0))
    centers = find_site_centers(img, grid)
    assert _is_one_to_one(centers, truth, tol=5.0)


def test_find_site_centers_recovers_genuinely_dark_corner_sites():
    """A trap that emitted NO light this calibration (a low-loading reference where a corner
    site never loaded across the averaged frames) still sits at its lattice node.  Detection
    has no peak to find there, so without the lattice repair its center lands on an image-
    border ``maximum_filter`` artifact -- and a 7x7 PSF box (which needs a full window around
    every site) cannot fit, crashing the calibration.  The detector must instead return a
    COMPLETE, in-bounds regular grid: every center far enough from the edge for a PSF box."""
    grid = (5, 7)
    image_shape = (96, 128)
    amps = np.full(grid[0] * grid[1], 700.0)
    dark = (0, grid[1] - 1, (grid[0] - 1) * grid[1], grid[0] * grid[1] - 1)  # four corners dark
    for i in dark:
        amps[i] = 0.0
    img, truth = _grid_template(grid, image_shape, amplitudes=amps)
    # a REAL averaged reference frame carries qCMOS read noise, so the background is not a
    # dead-flat plateau -- without it every background pixel is a tie-peak, a path the live
    # camera never takes.  Match reality so the test exercises the genuine dark-site case.
    img = img + np.random.default_rng(0).normal(0.0, 4.0, size=img.shape)

    centers = find_site_centers(img, grid)

    assert centers.shape == (grid[0] * grid[1], 2)
    half = 3                                       # the PSF half-width every box must clear
    h, w = image_shape
    assert np.all(centers[:, 0] >= half) and np.all(centers[:, 0] <= w - 1 - half)
    assert np.all(centers[:, 1] >= half) and np.all(centers[:, 1] <= h - 1 - half)
    # the recovered dark-site centers land on their true lattice nodes (within half a cell)
    assert _is_one_to_one(centers, truth, tol=7.0)
