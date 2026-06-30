"""#H3v-4a: the `dis` plot (a histogram) bound to a raw camera FRAME must show the dark/bright BIMODAL
readout, NOT a single background blob.

A frame's raw PIXELS are not bimodal (background dominates; the atom pixels blur into a tail) -- the
readout bimodal lives in the per-site box-summed COUNTS.  So a histogram panel, when bound to a camera
frame and a calibration is running, reduces each frame to its per-site counts via
``TrapCalibration.signals`` (the SAME readout the detection uses) before binning.  This pins that a
frame -> dis yields the per-site counts (cleanly bimodal), while a per-site signal passes through.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na  # noqa: E402

GRID = (5, 7)
N_SITES = GRID[0] * GRID[1]


def _fitted_sep(vals: np.ndarray) -> float:
    """The two-Gaussian fitted separation (summed-width units) -- the SAME gate the dis plot uses to
    decide a real bimodal; >= 1.5 = two separated peaks."""
    from scipy.optimize import curve_fit
    from Zou_lab_control._readout_math import bimodal_model
    vals = np.asarray(vals, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    n, edges = np.histogram(vals, bins=40)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = n.astype(float)
    span = float(np.ptp(vals)) or 1.0
    xs = np.sort(vals); cs = np.cumsum(xs); k = np.arange(1, xs.size, dtype=float)
    score = k * (xs.size - k) * ((cs[-1] - cs[:-1]) / (xs.size - k) - cs[:-1] / k) ** 2
    si = int(np.argmax(score)); split = float(0.5 * (xs[si] + xs[si + 1]))
    left, right = vals[vals <= split], vals[vals > split]
    mu0, mu1 = float(np.mean(left)), float(np.mean(right))
    if mu0 > mu1:
        mu0, mu1 = mu1, mu0
    s0 = max(float(np.std(left)), span / 40, 1e-9)
    s1 = max(float(np.std(right)), span / 40, 1e-9)
    amp = lambda mu: max(float(counts[int(np.clip(np.searchsorted(centers, mu), 0, len(counts) - 1))]), 1.0)
    p0 = [amp(mu0), mu0, s0, amp(mu1), mu1, s1]
    popt, _ = curve_fit(bimodal_model, centers, counts, p0=p0, maxfev=20000)
    if popt[1] > popt[4]:
        popt = np.array([popt[3], popt[4], popt[5], popt[0], popt[1], popt[2]])
    return abs(popt[4] - popt[1]) / (abs(popt[2]) + abs(popt[5]) + 1e-12)


def _hist_card_with_calibration():
    """A `dis` PanelCard whose calibration_provider returns a REAL virtual calibration -- exactly the
    wiring the console gives every panel (its `_active_calibration`)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    exp = na.connect("virtual", sitemap={"grid_shape": GRID, "image_shape": (64, 80)})
    exp.readout.sitemap(method="box", frames=6, display=False)
    exp.readout.thresholds(frames=40, display=False)
    cal = exp.readout.current
    card = PanelCard(PanelConfig(kind="hist", inputs=["frame_0"], source="value = signal"),
                     calibration_provider=lambda: cal)
    return exp, cal, card


def test_dis_on_a_camera_frame_yields_the_per_site_bimodal_not_pixels():
    exp, cal, card = _hist_card_with_calibration()
    try:
        from conftest import fire_live_imaging
        from Zou_lab_control.neutral_atom.core.signals import SignalHub
        from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement
        hub = SignalHub()
        cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer, repeat=8)
        fire_live_imaging(exp)
        for _ in range(8):
            cam.step()
        frame_block = np.asarray(hub.latest("frame_0"))             # (repeat, 1, H, W)

        # a raw camera frame -> dis reduces it to per-site COUNTS (repeat x n_sites of them), CLEANLY
        # bimodal (dark vs bright sites), NOT the single-blob raw-pixel histogram.
        samples = np.asarray(card._hist_samples(frame_block), dtype=float)
        assert samples.ndim == 1 and samples.size == 8 * N_SITES   # repeat x per-site counts
        assert _fitted_sep(samples) >= 1.5                          # two separated peaks (the readout)
        # the raw PIXELS are NOT bimodal (single background blob) -- proving the reduction is what fixes it
        assert _fitted_sep(np.asarray(frame_block, float).reshape(-1)) < 1.5

        # a per-site COUNTS array (already 1-D) passes through unchanged (no double reduction)
        counts = cal.signals(np.squeeze(frame_block[-1]))
        passthrough = card._hist_samples(np.asarray(counts, dtype=float))
        np.testing.assert_allclose(np.asarray(passthrough, dtype=float).reshape(-1), np.asarray(counts, float).reshape(-1))
    finally:
        if hasattr(card, "shutdown"):
            card.shutdown()
        exp.close()


def test_dis_without_calibration_keeps_raw_pixels():
    """No running calibration -> the histogram bins the raw value (a frame's pixels) unchanged: the
    per-site reduction only kicks in when a calibration is available to define the sites."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    card = PanelCard(PanelConfig(kind="hist", inputs=["frame_0"], source="value = signal"),
                     calibration_provider=lambda: None)
    try:
        frame = np.random.default_rng(0).normal(300, 30, size=(64, 80))
        out = np.asarray(card._hist_samples(frame), dtype=float)
        np.testing.assert_array_equal(out.reshape(-1), frame.reshape(-1))
    finally:
        if hasattr(card, "shutdown"):
            card.shutdown()


def test_active_calibration_falls_back_to_session_with_no_running_node():
    """#3A root fix: a frame -> dis must be bimodal even with NO Judge node running.  The console's
    ``_active_calibration`` prefers a running node's calibration but falls back to the SESSION's current
    calibration (the last sitemap/thresholds the operator ran) -- so the per-site reduction (and thus the
    bimodal dis) happens without requiring a running processor.  Before, it required a running node, so a
    plain frame -> dis defaulted to the single-blob raw pixels (fit as ONE gaussian)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    ensure_qt_app()
    exp = na.connect("virtual", sitemap={"grid_shape": GRID, "image_shape": (64, 80)})
    exp.readout.sitemap(method="box", frames=6, display=False)
    exp.readout.thresholds(frames=40, display=False)
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(), window_px=(900, 600))
    console._timer.stop()
    try:
        assert not console.running_nodes                       # NO Judge node running ...
        cal = console._active_calibration()                    # ... yet a calibration is available
        assert cal is not None and callable(getattr(cal, "signals", None))
        assert cal is exp.readout.current                      # it's the SESSION's loaded calibration
    finally:
        console.shutdown()
        exp.close()
