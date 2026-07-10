"""Dense images keep O(1) coordinate geometry until an explicit export boundary."""

from __future__ import annotations

import os
import tracemalloc

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Zou_lab_control.neutral_atom.core.fitting import FitRequest, fit_image
from Zou_lab_control.neutral_atom.core.raster import RegularRaster


def _spot(shape=(31, 41), center=(17.0, 12.0), sigma=3.0):
    yy, xx = np.indices(shape, dtype=float)
    return 0.2 + 4.0 * np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
                              / (2.0 * sigma ** 2))


def test_regular_raster_is_constant_size_and_materializes_correct_row_order():
    tracemalloc.start()
    grid = RegularRaster((1200, 1920), origin=(100.0, 200.0), spacing=(0.5, 2.0))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 16 * 1024
    assert grid.point_count == 2_304_000
    np.testing.assert_allclose(
        grid.coordinates([0, 1, 1920, grid.point_count - 1]),
        [[100.0, 200.0], [100.5, 200.0], [100.0, 202.0], [1059.5, 2598.0]],
    )


def test_fit_image_reads_regular_grid_origin_and_spacing_without_coordinate_table():
    image = _spot(center=(17.0, 12.0))
    grid = RegularRaster(image.shape, origin=(100.0, 200.0), spacing=(0.5, 2.0))
    result = fit_image(image, FitRequest("center"), grid=grid)
    assert result.valid, result.status
    params = result.parameter_map()
    np.testing.assert_allclose(params["x0"], 108.5, atol=0.15)
    np.testing.assert_allclose(params["y0"], 224.0, atol=0.3)


def test_live_2d_and_data_figure_keep_the_compact_grid(monkeypatch):
    from Zou_lab_control.frontend.live import plot

    image = _spot()
    grid = RegularRaster(image.shape, origin=(300.0, 400.0))
    plotter = plot(
        grid, image.ravel(), kind="2d", display=False, data_figure=True, update=False)
    try:
        assert plotter.data_x is grid
        assert plotter.raster_grid is grid
        assert plotter.data_figure.raster_grid is grid
        adapted = plotter.selection_rows()
        assert adapted["raster"] is grid and adapted["coordinates"] == {}

        # Fingerprinting and fitting must not call the explicit full-coordinate export.
        monkeypatch.setattr(
            RegularRaster,
            "coordinates",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("dense coordinate table was materialized")),
        )
        key = plotter._fit_data_fingerprint()
        assert key[0] == ("regular_raster", grid.shape, grid.origin, grid.spacing)
        result = plotter.data_figure.fit("center", request=FitRequest("center"))
        assert result.valid, result.status
    finally:
        plt.close(plotter.fig)


def test_frontend_image_builders_do_not_expand_pixel_meshes():
    from pathlib import Path
    import Zou_lab_control.frontend.live as live
    import Zou_lab_control.frontend.task_console as task_console

    live_source = Path(live.__file__).read_text(encoding="utf-8")
    console_source = Path(task_console.__file__).read_text(encoding="utf-8")
    assert "np.meshgrid(np.arange(nx" not in live_source
    assert "np.meshgrid(x0 + np.arange(nx" not in console_source
