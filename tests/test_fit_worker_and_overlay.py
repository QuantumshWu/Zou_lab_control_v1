"""Contracts for the per-panel worker fit NODE + the DISPLAY-only overlay (#5/#6/#7).

* EVERY fit family is a per-panel :class:`FitProcessor` node that consumes TWO signals -- the source
  block AND the panel's ``<slug>_region`` signal -- and fits on its WORKER, never the Qt event loop.
* A hist fit fits the BINNED ``(bin_centers, counts)`` carried on the region (NEVER the raw per-pixel
  samples), so it is O(bins) -- instant -- and its parameters are finite.
* The panel overlay reconstructs the curve / centre from PUBLISHED parameters via
  ``apply_published_fit`` -- it calls NO ``curve_fit`` (the solve is on the node worker).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control._readout_math import gaussian2d_center
from Zou_lab_control.neutral_atom.core.fitting import FitRequest, FitResult, fit_model
from Zou_lab_control.neutral_atom.core.selection import (
    Selection, axis_crop_binding, region_tensor, value_mask_binding)
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.core.signal_tensor import SignalSchema
from Zou_lab_control.neutral_atom.operations.processors.fit import FitProcessor


def _publish_region(hub, name, selection, *, bins=None):
    tensor = region_tensor(selection, bins=bins)
    hub.register_signal(name, tensor.schema)
    hub.publish({name: tensor})


def _hub_with(name, block):
    hub = SignalHub()
    hub.register_signal(name, SignalSchema(
        point_shape=(block.shape[1],), data_shape=block.shape[2:], dtype=block.dtype,
        repeat_capacity=block.shape[0]))
    hub.publish({name: block})
    return hub


def test_fit_processor_consumes_source_and_region_signals():
    """#7: the fit node consumes TWO signals -- the source frame AND the panel's region control signal."""
    ys, xs = np.mgrid[0:120, 0:160]
    frame = gaussian2d_center((xs, ys), 150.0, 12.0, 15.0, 100.0, 60.0).astype(np.float64).reshape(1, 1, 120, 160)
    hub = _hub_with("frame_0", frame)
    _publish_region(hub, "p_region", Selection.rectangle(0, 160, 0, 120,
        metadata={"binding": axis_crop_binding([("x", -1, 0.0, 1.0), ("y", -2, 0.0, 1.0)])}))
    node = FitProcessor(hub, source_expr={"inputs": ["frame_0"], "source": "value = signal"},
                        region="p_region", fit_request={"model": "center"})
    assert node.consumes == ("frame_0", "p_region")
    hub.publish({"frame_0": frame})
    node.step()
    assert abs(hub.latest("fit_x0")[0, 0, 0] - 100.0) < 2.0
    assert abs(hub.latest("fit_y0")[0, 0, 0] - 60.0) < 2.0


def test_hist_fit_uses_binned_data_not_raw_samples():
    """#5: a distribution fit runs on the BINNED (centers, counts) carried on the region -- finite
    params -- and the solver sees ONLY the bins (fit_points <= bins), NEVER the millions of raw
    per-pixel samples.  ``fit_points`` is the DETERMINISTIC proof the fit is O(bins), not a wall clock."""
    rng = np.random.default_rng(0)
    samples = rng.normal(600.0, 25.0, size=(1200, 1920)).astype(np.float64).reshape(1, 1, 1200, 1920)
    hub = _hub_with("frame_h", samples)
    _publish_region(hub, "h_region", Selection.value(400.0, 800.0,
        metadata={"binding": value_mask_binding()}), bins=60)
    node = FitProcessor(hub, source_expr={"inputs": ["frame_h"], "source": "value = signal"},
                        region="h_region", fit_request={"model": "gaussian"})
    hub.publish({"frame_h": samples})
    node.step()

    params = {name: float(hub.latest(f"fit_{name}")[0, 0, 0]) for name in fit_model("gaussian").names}
    assert bool(hub.latest("fit_valid")[0, 0, 0])
    assert np.isfinite(list(params.values())).all()
    assert abs(params["x0"] - 600.0) < 5.0 and abs(params["sigma"] - 25.0) < 5.0
    # the solver saw ONLY the binned points (<= bins), not the 2.3M raw samples -- the O(bins) proof
    assert int(hub.latest("fit_points")[0, 0, 0]) <= 60


def test_hist_fit_output_names_derive_from_the_model():
    """A gaussian (1-D family) panel publishes its OWN parameter names (fit_sigma), not a fixed
    centre layout -- the keys derive from the solved model (single source)."""
    samples = np.random.default_rng(1).normal(5.0, 1.0, size=(1, 1, 4000)).astype(np.float64)
    hub = _hub_with("s", samples)
    _publish_region(hub, "s_region", Selection.value(0.0, 12.0,
        metadata={"binding": value_mask_binding()}), bins=40)
    node = FitProcessor(hub, source_expr={"inputs": ["s"], "source": "value = signal"},
                        region="s_region", fit_request={"model": "gaussian"})
    hub.publish({"s": samples}); node.step()
    published = {s for s in node.published_signals()}
    assert {"fit_amplitude", "fit_offset", "fit_sigma", "fit_x0"} <= published
    assert "fit_radius" not in published and "fit_y0" not in published


def test_overlay_draws_from_published_params_without_curve_fit(monkeypatch):
    """#6: the panel overlay reconstructs the curve from a PUBLISHED FitResult (apply_published_fit) and
    calls NO curve_fit -- the solve is on the node worker, never the Qt event loop."""
    import scipy.optimize as opt
    from Zou_lab_control.frontend.live import HistogramFigure

    called = {"n": 0}
    real = opt.curve_fit
    monkeypatch.setattr(opt, "curve_fit", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or real(*a, **k))

    samples = np.random.default_rng(2).normal(5.0, 1.0, size=6000).astype(np.float64)
    hist = HistogramFigure(samples, bins=40, fit="none").show(display=False)  # fit=none: no built-in bimodal solve
    model = fit_model("gaussian")
    # a published result the console would hand the overlay (params solved on the worker)
    result = FitResult("gaussian", model.names, np.array([1200.0, 0.0, 1.0, 5.0]), None, True, "ok",
                       {"rmse": 1.0, "r2": 0.99}, 40, "value")
    hist.apply_published_fit(result)
    assert hist._fit_overlay_result is result
    assert len(hist._fit_overlay_artists) == 1              # the reconstructed model curve
    hist.apply_published_fit(None)                          # clearing drops the overlay
    assert hist._fit_overlay_artists == []
    assert called["n"] == 0, "the overlay must NOT solve -- it only draws published parameters"
