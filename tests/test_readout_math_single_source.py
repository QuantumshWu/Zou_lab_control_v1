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
