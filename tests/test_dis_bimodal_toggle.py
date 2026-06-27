"""#H3w-1: the `dis` histogram is CONSISTENTLY bimodal by default (every signal), with a toggle.

The bug: the fit auto-chose single-vs-double per data (the fitted-separation gate decided the VISUAL), so
the same panel showed one peak for some signals and two for others -- inconsistent.  Now `bimodal`
(default True) is an explicit param: ON always draws the dark/bright two-Gaussian decomposition; OFF
draws a single Gaussian.  The fidelity STAT stays honest -- reported only when the two peaks cleanly
separate, else 'fit F=N/A' -- but that no longer changes whether one or two curves are drawn.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.frontend.qt_fluent import ensure_qt_app  # noqa: E402
from Zou_lab_control.frontend import plot  # noqa: E402


def _two_peaks_drawn(fig) -> bool:
    return len(fig.fit_line_left.get_xdata()) > 0 and len(fig.fit_line_right.get_xdata()) > 0


def _only_single_drawn(fig) -> bool:
    return (len(fig.fit_line_total.get_xdata()) > 0
            and len(fig.fit_line_left.get_xdata()) == 0
            and len(fig.fit_line_right.get_xdata()) == 0)


def test_default_bimodal_is_consistent_even_on_unimodal_data():
    """Default (bimodal=True) draws the two-Gaussian decomposition for ANY data -- including a single
    broad blob (raw frame pixels) -- so the dis reads consistently, never auto-collapsed to one peak."""
    ensure_qt_app()
    rng = np.random.default_rng(0)
    unimodal = rng.normal(300.0, 20.0, 400)
    fig = plot(unimodal, kind="hist")                       # bimodal defaults to True
    assert _two_peaks_drawn(fig), "default dis must draw the bimodal decomposition even on unimodal data"
    # but the fidelity is honest: an unseparated blob reports N/A, NOT a fake number
    assert not fig._fit_separated
    assert fig._fit_fidelity(fig.fit_threshold or float(np.median(unimodal))) is None


def test_toggle_off_draws_a_single_gaussian():
    ensure_qt_app()
    rng = np.random.default_rng(1)
    unimodal = rng.normal(300.0, 20.0, 400)
    fig = plot(unimodal, kind="hist", bimodal=False)
    assert _only_single_drawn(fig), "bimodal=False must draw ONE Gaussian, not the two-peak split"


def test_separated_readout_reports_a_fidelity():
    """A genuinely bimodal readout (dark + bright) draws two peaks AND reports a finite fidelity (the
    separation gate is met)."""
    ensure_qt_app()
    rng = np.random.default_rng(2)
    readout = np.concatenate([rng.normal(300.0, 15.0, 300), rng.normal(460.0, 15.0, 200)])
    fig = plot(readout, kind="hist")
    assert _two_peaks_drawn(fig) and fig._fit_separated
    fidelity = fig._fit_fidelity(fig.fit_threshold or 380.0)
    assert fidelity is not None and 0.5 <= fidelity <= 1.0
