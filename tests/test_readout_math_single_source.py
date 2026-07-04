"""MECHANICAL guard for the 复用 / single-source principle (AGENTS.md §2).

The dark/bright Gaussian, the normal CDF, and the confidence-weighted readout
fidelity formula must have exactly ONE implementation -- Zou_lab_control/
_readout_math.py -- imported by BOTH the calibration side (neutral_atom) and the
live-plot side (frontend).  Two copies drift: the saved-calibration fidelity and
the GUI "fit F=..%" would disagree for the same data (the 2026-06 finding).

This module is a dependency-free leaf (the _viewer_registry precedent) so the
sealed frontend can import it without importing neutral_atom, and neutral_atom
stays headless.  This test fails if the math is re-implemented anywhere else.
"""

from __future__ import annotations

import re
from pathlib import Path

import Zou_lab_control


_PKG = Path(Zou_lab_control.__file__).resolve().parent
_SHARED = (_PKG / "_readout_math.py").resolve()
# The readout analysis side + the frontend plot side -- the two places the math
# used to be duplicated across.
_LAYERS = ("neutral_atom/core", "neutral_atom/operations", "neutral_atom/subsystems", "frontend")


def _py_files():
    for sub in _LAYERS:
        yield from (_PKG / sub).rglob("*.py")


def test_normal_cdf_kernel_lives_only_in_shared_module():
    """``erf(`` is the signature of the normal CDF; it must appear only in the
    shared module, never re-implemented in core or the frontend."""
    offenders = [
        str(f.relative_to(_PKG))
        for f in _py_files()
        if f.resolve() != _SHARED and re.search(r"\berf\s*\(", f.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"normal-CDF (erf) re-implemented outside _readout_math.py: {offenders}"


def test_shared_math_functions_are_not_redefined():
    for name in ("def normal_cdf", "def confidence_weighted_fidelity",
                 "def gaussian_jacobian_columns", "def bimodal_model"):
        hits = [
            str(f.relative_to(_PKG))
            for f in _py_files()
            if f.resolve() != _SHARED and name in f.read_text(encoding="utf-8")
        ]
        assert hits == [], f"{name!r} duplicated outside _readout_math.py: {hits}"


def test_both_sides_use_the_same_shared_objects():
    """Same function objects on both sides -> they cannot diverge."""
    from Zou_lab_control import _readout_math as rm
    from Zou_lab_control.neutral_atom.core import analysis, bimodal
    from Zou_lab_control.frontend import live

    assert analysis.confidence_weighted_fidelity is rm.confidence_weighted_fidelity
    assert bimodal.normal_cdf is rm.normal_cdf
    assert live.confidence_weighted_fidelity is rm.confidence_weighted_fidelity
    assert live.gaussian is rm.gaussian and live.bimodal_model is rm.bimodal_model


def test_finite_mean_is_the_one_gap_safe_average_on_both_sides():
    """``finite_mean`` -- the ONE gap-safe average shared by the live plots and the per-site shot
    reduce -- equals ``np.nanmean`` on finite data, yields NaN for an all-NaN slice, EXCLUDES inf, and
    (unlike ``np.nanmean``) NEVER emits a RuntimeWarning.  The SAME object on both layers, so the
    live-plot empty-grid collapse and the na per-site reduce cannot diverge (and neither suppresses a
    warning after the fact)."""
    import warnings

    import numpy as np

    import sys

    from Zou_lab_control import _readout_math as rm
    from Zou_lab_control.frontend import live
    # ``operations.measurement`` (the ATTRIBUTE) re-exports the @measurement decorator, shadowing the
    # same-named submodule -- so reach the real module via ``sys.modules`` to read its ``finite_mean``.
    import Zou_lab_control.neutral_atom.operations.measurement  # noqa: F401 -- register the submodule
    meas_mod = sys.modules["Zou_lab_control.neutral_atom.operations.measurement"]

    assert live.finite_mean is rm.finite_mean is meas_mod.finite_mean   # ONE object, both layers

    a = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 5.0], [2.0, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)                     # NO warning may be raised
        got = rm.finite_mean(a, 0)
        assert rm.finite_mean(np.full(4, np.nan), 0).shape == ()           # scalar reduce, all-NaN -> NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                    # nanmean itself warns; ignore ONLY here
        want = np.nanmean(a, 0)
    np.testing.assert_allclose(got, want, equal_nan=True)                  # equals nanmean on finite
    assert np.isnan(got[1])                                                # all-NaN column -> NaN
    assert rm.finite_mean(np.array([1.0, np.inf, 3.0]), 0) == 2.0          # inf excluded (not just NaN)
