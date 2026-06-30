"""Single-source guard + regression for the Otsu (between-class-variance) split.

The maximum-between-class-variance threshold used to exist twice: a 96-bin
histogram version in ``core/analysis.py`` and an exact sorted-sample version in
``core/bimodal.py``.  They are now ONE source: ``bimodal.exact_otsu_threshold``
is authoritative and ``analysis.otsu_threshold`` is a thin wrapper that delegates
to it (``mode="binned"`` keeps the histogram fast path for very large samples).

These tests pin:
  * delegation -- the default path is literally the exact function's output;
  * gap-centre robustness -- on a well-separated bimodal (including degenerate
    integer counts) the split lands in the MIDDLE of the empty gap, not on the
    top of the dark peak.  ``exact`` gets this for free from the sorted samples;
    ``binned`` reproduces it via plateau-centre averaging.  Both must agree.
"""

from __future__ import annotations

import math

import numpy as np

from Zou_lab_control.neutral_atom.core.analysis import otsu_threshold
from Zou_lab_control.neutral_atom.core.bimodal import exact_otsu_threshold


def test_default_mode_delegates_to_exact():
    """The public ``otsu_threshold`` default IS the exact single source."""
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(10.0, 1.0, 400), rng.normal(40.0, 1.5, 400)])
    assert otsu_threshold(values) == exact_otsu_threshold(values)


def test_bad_mode_raises():
    with np.errstate(all="ignore"):
        try:
            otsu_threshold([0.0, 1.0, 2.0, 3.0], mode="bogus")
        except ValueError:
            return
    raise AssertionError("unknown mode must raise ValueError")


def test_single_finite_value_degenerates_cleanly():
    """A flat (min==max) sample returns that value, never NaN."""
    assert otsu_threshold([5.0, 5.0, 5.0, 5.0]) == 5.0
    assert otsu_threshold([5.0, 5.0, 5.0, 5.0], mode="binned") == 5.0


def test_gap_centre_on_separated_bimodal_both_modes():
    """A wide empty gap -> threshold sits in the MIDDLE of the gap.

    The dark-peak top vs gap-centre distinction is the whole robustness point:
    a threshold at the top of the dark peak misclassifies the dark tail.  Both
    the exact and the binned paths must place it near the gap centre.
    """
    rng = np.random.default_rng(11)
    dark = rng.normal(10.0, 1.0, 500)
    bright = rng.normal(40.0, 1.5, 500)
    values = np.concatenate([dark, bright])
    gap_lo, gap_hi = float(dark.max()), float(bright.min())
    gap_mid = 0.5 * (gap_lo + gap_hi)

    t_exact = otsu_threshold(values)            # exact
    t_binned = otsu_threshold(values, mode="binned")

    # Inside the empty gap (above every dark sample, below every bright sample).
    assert gap_lo < t_exact < gap_hi
    assert gap_lo < t_binned < gap_hi
    # Near the gap centre, not hugging the dark-peak edge.
    half_gap = 0.5 * (gap_hi - gap_lo)
    assert abs(t_exact - gap_mid) <= 0.5 * half_gap
    assert abs(t_binned - gap_mid) <= 0.5 * half_gap


def test_integer_count_gap_degeneracy_takes_gap_centre():
    """Degenerate integer counts (the real qCMOS case): dark in {0..3}, bright
    in {20..25}.  A naive argmax over equal-score adjacent pairs would put the
    split just above the dark peak (~3.5); the exact between-class-variance peak
    is unique at the gap centre (11.5).  This is the regression the finding asks
    for: the gap-degeneration case must resolve to the gap centre.
    """
    dark = np.array([0, 1, 1, 2, 2, 2, 3, 3] * 30, dtype=float)
    bright = np.array([20, 21, 22, 22, 23, 24, 25] * 30, dtype=float)
    values = np.concatenate([dark, bright])

    t = otsu_threshold(values)
    assert math.isclose(t, 11.5, abs_tol=1e-9)
    # Well clear of the dark-peak top (3) -- not the argmax-first failure mode.
    assert t > 3.5
    assert exact_otsu_threshold(values) == t
