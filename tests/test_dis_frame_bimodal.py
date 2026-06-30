"""A `dis` plot (histogram) bins EXACTLY the array it is bound to.

DECOUPLING: the dis must NOT reach into any calibration / processor to transform its input -- downstream
never knows an upstream node's internals.  Bind a dis to a processor's per-site COUNTS to see the
dark/bright bimodal readout; bind it to a raw camera FRAME and it histograms the frame's pixels,
honestly.  There is NO hidden per-site reduction and NO calibration hook on the panel: whatever the
source gives the dis is what the dis bins (so the fit is a DISPLAY choice on that data, never decided by
secret knowledge of where the data came from).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def _hist_card():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig
    ensure_qt_app()
    return PanelCard(PanelConfig(kind="hist", inputs=["frame_0"], source="value = signal"))


def test_dis_bins_its_raw_input_no_calibration_reach():
    """A 2D camera FRAME bound to a dis histograms ALL its pixels (size H*W) -- it is NOT secretly
    reduced to per-site counts.  And the panel carries no calibration hook at all (structural decoupling:
    the dis cannot reach into a processor/session for 'extra data')."""
    card = _hist_card()
    try:
        frame = np.random.default_rng(0).normal(300.0, 30.0, size=(24, 30))
        card._render(frame)
        assert card.plotter is not None
        binned = np.asarray(card.plotter.values, dtype=float).reshape(-1)
        assert binned.size == frame.size                       # every pixel binned -- NO per-site reduction
        np.testing.assert_allclose(np.sort(binned), np.sort(frame.reshape(-1)))
        # the decoupling is structural, not just behavioural: the panel has no calibration provider
        # and no per-site reducer to reach through.
        assert not hasattr(card, "calibration_provider")
        assert not hasattr(card, "_hist_samples")
    finally:
        if hasattr(card, "shutdown"):
            card.shutdown()


def test_dis_bins_per_site_counts_unchanged_when_bound_to_counts():
    """Bound to a 1-D per-site COUNTS array (what a processor publishes), the dis bins those values
    unchanged -- the bimodal readout comes from the DATA the source provides, not from the plot reaching
    for it."""
    card = _hist_card()
    try:
        counts = np.concatenate([np.random.default_rng(0).normal(300.0, 20.0, 40),
                                 np.random.default_rng(1).normal(460.0, 20.0, 30)])
        card._render(counts)
        assert card.plotter is not None
        binned = np.asarray(card.plotter.values, dtype=float).reshape(-1)
        np.testing.assert_allclose(np.sort(binned), np.sort(counts))
    finally:
        if hasattr(card, "shutdown"):
            card.shutdown()
