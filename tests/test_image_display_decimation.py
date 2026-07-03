"""Camera-frame display resolution -- the Z-round contracts.

A full camera frame (1920x1200+) fed to a 2D live panel used to be resampled by
matplotlib on EVERY draw (~90 ms) through a render buffer SMALLER than the screen
area it was stretched onto (render_scale 0.5 vs display_scale 0.7) with unfiltered
"none" interpolation -- three independent resolution losses.  The contracts here pin
the redesign:

1. the display layer keeps the FULL array and shows a block-mean (area-averaged)
   view within the axes' pixel budget -- photometric, never subsampled;
2. zooming re-slices from the full array, so detail returns progressively until the
   view fits the budget 1:1 (exact values, a no-copy slice);
3. the live render buffer matches the on-screen device pixels EXACTLY
   (render_scale == display_scale in panel_canvas) -- one resample end to end;
4. small scan grids pass through untouched (crisp nearest pixels, zero change).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

from Zou_lab_control.frontend.live import (
    _decimate_image_view, _image_axes_px_budget, panel_plot)


def _camera_frame(h=1200, w=1920, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(200.0, 20.0, (h, w))


def _camera_panel(frame):
    h, w = frame.shape
    ys, xs = np.mgrid[0:h, 0:w]
    data_x = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    return panel_plot(data_x, frame.ravel().astype(float), kind="2d", size="2x2")


# --------------------------------------------------------------- pure helper
def test_decimation_is_a_block_mean_never_a_subsample():
    pat = np.arange(16.0).reshape(4, 4)
    small, ext = _decimate_image_view(pat, (0, 4, 4, 0), (0, 4), (4, 0), 2)
    expect = np.array([[pat[0:2, 0:2].mean(), pat[0:2, 2:4].mean()],
                       [pat[2:4, 0:2].mean(), pat[2:4, 2:4].mean()]])
    assert np.allclose(small, expect)
    assert ext == (0.0, 4.0, 4.0, 0.0)          # full view -> full (inverted-y) extent
    # photometric: with a divisible budget the global mean survives exactly (area
    # average, not nearest picking); a non-divisible factor only trims <factor px
    # off the far edge, and the returned extent maps exactly the pixels kept.
    frame = _camera_frame()
    full, fext = _decimate_image_view(frame, (0, 1920, 1200, 0), (0, 1920), (1200, 0), 240)
    assert full.shape == (240, 240)
    assert abs(float(full.mean()) - float(frame.mean())) < 1e-9
    trimmed, text = _decimate_image_view(frame, (0, 1920, 1200, 0), (0, 1920), (1200, 0), 250)
    assert trimmed.shape == (300, 274)               # 1200//4, (1920//7*7)//7
    assert text[1] == 274 * 7                        # extent ends at the kept edge


def test_decimated_size_is_never_below_the_budget():
    # floor-based factor: the result is >= the budget on both axes (never softer
    # than the screen), and below 2x the budget (bounded draw cost).
    frame = _camera_frame()
    small, _ = _decimate_image_view(frame, (0, 1920, 1200, 0), (0, 1920), (1200, 0), (380, 300))
    assert 380 <= small.shape[1] < 2 * 380
    assert 300 <= small.shape[0] < 2 * 300


def test_zoom_returns_full_resolution_exactly():
    # a view small enough for the budget comes back as the UNTOUCHED slice: pixel
    # values equal the full array (this is the "zoom brings the detail back" law).
    frame = _camera_frame()
    small, ext = _decimate_image_view(frame, (0, 1920, 1200, 0), (100.2, 300.7), (150.9, 50.5), 512)
    assert small.base is not None                    # a no-copy view of the full array
    jx0 = int(np.floor(100.2)); jy0 = int(np.floor(50.5))
    assert np.array_equal(small, frame[jy0:jy0 + small.shape[0], jx0:jx0 + small.shape[1]])
    assert ext[0] <= 100.2 and ext[1] >= 300.7       # kept region covers the view
    assert ext[3] <= 50.5 and ext[2] >= 150.9        # (inverted y: top=ext[3], bottom=ext[2])


def test_small_scan_grids_pass_through_untouched():
    grid = np.arange(600.0).reshape(20, 30)
    small, ext = _decimate_image_view(grid, (0, 3, 2, 0), (0, 3), (2, 0), 512)
    assert np.array_equal(small, grid)
    assert ext == (0.0, 3.0, 2.0, 0.0)


# ------------------------------------------------------------ Live2DDis wiring
def test_live2d_displays_a_budgeted_view_and_keeps_the_full_grid():
    frame = _camera_frame()
    p = _camera_panel(frame)
    assert p.grid.shape == frame.shape               # analysis-side array stays full-res
    p.fig.canvas.draw()                              # settle the split-axes layout
    p._refresh_display_image()                       # budget is read per refresh (post-layout)
    shown = np.asarray(p.image.get_array())
    bx, by = _image_axes_px_budget(p.ax)             # the axes' design px = screen upper bound
    assert shown.shape[1] < 2 * bx and shown.shape[0] < 2 * by
    assert shown.shape[1] >= min(bx, frame.shape[1]) and shown.shape[0] >= min(by, frame.shape[0])
    assert shown.shape[0] < frame.shape[0]           # a camera frame IS decimated for display
    assert p.image.get_interpolation() == "antialiased"  # filtered when minified; portable ("auto" only exists in mpl>=3.5)


def test_live2d_zoom_reslices_to_full_resolution():
    frame = _camera_frame()
    p = _camera_panel(frame)
    coarse = np.asarray(p.image.get_array()).shape
    p.ax.set_xlim(200, 380)                           # a window well inside the budget
    p.ax.set_ylim(300, 160)                           # (this axis draws top-to-bottom)
    fine = np.asarray(p.image.get_array())
    assert fine.shape != coarse
    jx0, jy0 = 200, 160
    assert np.array_equal(fine, p.grid[jy0:jy0 + fine.shape[0], jx0:jx0 + fine.shape[1]])


def test_sitemap_background_uses_the_same_display_policy():
    from Zou_lab_control.frontend.live import LiveSiteMap

    frame = _camera_frame(h=600, w=960, seed=3)
    centers = np.array([[100.0, 100.0], [200.0, 100.0], [300.0, 200.0]])
    occ = np.array([[1.0], [0.0], [1.0]])
    p = LiveSiteMap(centers, occ, image=frame, interactions=False).show(display=False)
    p.fig.canvas.draw()
    p._refresh_display_image()
    shown = np.asarray(p._bg_image.get_array())
    bx, by = _image_axes_px_budget(p.ax)
    assert shown.shape[1] < 2 * bx and shown.shape[0] < 2 * by
    assert p._bg_image.get_interpolation() == "antialiased"
    assert p.background.shape == frame.shape
    # a new frame refreshes the displayed view through the same decimation
    p.set_background(frame + 100.0)
    assert abs(float(np.asarray(p._bg_image.get_array()).mean()) - (float(frame.mean()) + 100.0)) < 1e-6


# ------------------------------------------------------- canvas: one resample
def test_panel_canvas_buffer_matches_the_screen_exactly():
    """render_scale == display_scale: the Agg buffer is byte-for-byte the size the
    screen shows, so the blit is 1:1 and the ONLY resample in the whole chain is
    the budgeted decimation above.  (The old 0.5-render/0.7-display mismatch blit
    every live panel through a permanent 1.4x upscale.)"""
    pytest.importorskip("PyQt5")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend.qt_canvas import panel_canvas

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    del app
    p = _camera_panel(_camera_frame(h=120, w=192, seed=5))
    cv = panel_canvas(p.fig)
    assert abs(cv._zlc_render_scale - (1.0 / cv._zlc_ratio)) < 1e-9
    cv.draw()
    real = QtWidgets.QWidget.devicePixelRatioF(cv) or 1.0     # the true screen ratio
    size = cv._zlc_design_size()
    assert abs(cv.renderer.width - size.width() * real) <= 1.0
    assert abs(cv.renderer.height - size.height() * real) <= 1.0
