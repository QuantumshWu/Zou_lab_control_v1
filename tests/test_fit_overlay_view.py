"""Contracts for the fit overlay + the model-agnostic frame-fit node.

* #11: a 2-D image centre fit is a READOUT overlay -- drawing the dot/ring must NOT change the
  image's displayed range (xlim/ylim) or its colour limit.  ``data_figure._draw_fit_result`` snapshots
  and restores the view, so the ONE overlay primitive (used by both the flat panel and the grid) is
  strictly non-mutating of the view.
* #9: the frame-fit node's published keys derive from the fitted MODEL's parameter ``names`` (single
  source), never a hardwired ``x0/y0/size`` centre layout, and its spec name is type-agnostic.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import numpy as np

from Zou_lab_control._readout_math import gaussian2d_center
from Zou_lab_control.frontend.live import Live2DDis, apply_fit_to_figure
from Zou_lab_control.neutral_atom.core.fitting import FitRequest, fit_model
from Zou_lab_control.neutral_atom.core.raster import RegularRaster
from Zou_lab_control.neutral_atom.operations.processors.fit import (
    FIT_SPEC_NAME, FitProcessor, _param_keys, _provided_keys)


def _image_plot(h=120, w=160):
    ys, xs = np.mgrid[0:h, 0:w]
    img = gaussian2d_center((xs, ys), 150.0, 12.0, 15.0, 100.0, 60.0).astype(np.float64)
    raster = RegularRaster(shape=(h, w))
    return Live2DDis(raster, img.reshape(-1, 1), labels=["x", "y", "counts"]).show(display=False)


def test_2d_centre_fit_overlay_does_not_change_the_view():
    plot = _image_plot()
    ax = plot.ax
    xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
    clim0 = ax.get_images()[0].get_clim()

    df = plot.to_data_figure()
    result = apply_fit_to_figure(df, FitRequest("center").to_dict())

    assert result.valid, result.status
    # the fit CONVERGED near the planted centre -- so this is a real overlay, not a no-op
    params = result.parameter_map()
    assert abs(params["x0"] - 100.0) < 2.0 and abs(params["y0"] - 60.0) < 2.0
    # ...yet the view is byte-for-byte preserved: no dataLim expansion, no re-clim (#11)
    assert ax.get_xlim() == xlim0
    assert ax.get_ylim() == ylim0
    assert ax.get_images()[0].get_clim() == clim0
    # the overlay drew exactly the centre dot + the radius ring
    assert len(df._fit_artists) == 2


def test_frame_fit_keys_derive_from_the_model_not_a_centre_layout():
    # provides is EXACTLY the model's fit_<name> parameters + the fixed quality/validity suffix,
    # derived from FitModel.names -- so a non-centre model would publish ITS parameters.
    assert _param_keys("center") == tuple(f"fit_{n}" for n in fit_model("center").names)
    assert FitProcessor.provides == _provided_keys("center")
    assert set(FitProcessor.provides) == {
        "fit_amplitude", "fit_offset", "fit_radius", "fit_x0", "fit_y0",
        "fit_valid", "fit_rmse", "fit_r2", "fit_status", "fit_points"}
    # no hardwired centre-only 'fit_size' key survives the generic derivation
    assert "fit_size" not in FitProcessor.provides
    # the spec + node label are type-agnostic (no longer named "center")
    assert FIT_SPEC_NAME == "Frame fit"
    assert "center" not in FitProcessor.node_label.lower()
