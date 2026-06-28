"""#issue-2: a single-Gaussian histogram must NOT draw a threshold (a cut between two populations is
meaningless when there is one) -- it shows the FIT params + the out-of-fit fraction instead.  A genuinely
SEPARATED bimodal keeps its threshold + fidelity readout, and an EXPLICIT (calibration) cut always shows.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

from Zou_lab_control.frontend.live import HistogramFigure


def _stats(h) -> str:
    return h.stats_text.get_text()


def _visible_threshold(h) -> bool:
    return bool(h.threshold_lines) and any(ln.get_visible() for ln in h.threshold_lines)


def test_single_gaussian_has_no_threshold_shows_fit_and_out_of_fit():
    rng = np.random.default_rng(0)
    vals = rng.normal(100.0, 8.0, size=4000)          # ONE population -> no separable cut
    h = HistogramFigure(vals, bimodal=False).show()
    try:
        assert h._has_threshold is False
        assert not _visible_threshold(h), "single-Gaussian must NOT show a threshold line"
        text = _stats(h)
        assert "out-of-fit" in text and ("gauss" in text or "single" in text), text
        assert "fit F=" not in text and "th=" not in text, text
    finally:
        import matplotlib.pyplot as plt; plt.close(h.fig)


def test_separated_bimodal_keeps_threshold_and_fidelity():
    rng = np.random.default_rng(1)
    vals = np.concatenate([rng.normal(20.0, 3.0, 2000), rng.normal(80.0, 4.0, 2000)])   # two clear peaks
    h = HistogramFigure(vals, bimodal=True).show()
    try:
        assert h._fit_separated and h._has_threshold
        assert _visible_threshold(h), "a separated bimodal must show its threshold line"
        assert "th=" in _stats(h) and "fit F=" in _stats(h)
    finally:
        import matplotlib.pyplot as plt; plt.close(h.fig)


def test_explicit_threshold_shows_even_on_single_gaussian():
    rng = np.random.default_rng(2)
    vals = rng.normal(50.0, 6.0, size=3000)           # single population, but an explicit cut is given
    h = HistogramFigure(vals, bimodal=False, thresholds=[55.0]).show()
    try:
        assert h._has_threshold and _visible_threshold(h)
        assert "th=" in _stats(h)
    finally:
        import matplotlib.pyplot as plt; plt.close(h.fig)
