"""Shared readout math: the ONE source for the Gaussian / bimodal / fidelity
primitives used on BOTH sides of the readout.

This module is a dependency-free leaf (numpy + scipy only) at the same tier as
``_viewer_registry``.  It exists because the same dark/bright Gaussian model,
normal CDF, and confidence-weighted fidelity formula were implemented twice --
once in ``neutral_atom.core`` (the saved-calibration fidelity) and once in
``frontend.live`` (the live histogram's "fit F=..%").  Two copies drift: a tweak
to one silently disagrees with the other for the same data.  The frontend may
not import ``neutral_atom`` (sealed-layer contract) and ``neutral_atom`` stays
headless, so neither can own this -- it lives here and both import it.

Keep this module free of any ``Zou_lab_control`` import.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.special import erf


_SIGMA_FLOOR = 1e-12


def gaussian(x, amp, mu, sigma):
    """``amp * exp(-(x-mu)^2 / (2 sigma^2))`` with a tiny sigma floor.

    Within every caller's fit bounds sigma > 0, so the floor never changes a
    result -- it only guards a degenerate sigma == 0."""

    sigma = max(abs(float(sigma)), _SIGMA_FLOOR)
    return amp * np.exp(-((np.asarray(x, dtype=float) - mu) ** 2) / (2.0 * sigma * sigma))


def gaussian_jacobian_columns(x, amp, mu, sigma):
    """Analytic gradient columns ``(d/d_amp, d/d_mu, d/d_sigma)`` of :func:`gaussian`.

    Returned as a 3-tuple of arrays so a single-Gaussian fit can stack them and a
    bimodal fit can concatenate two sets.  Matches :func:`gaussian` exactly for
    sigma > 0, so handing it to ``curve_fit`` removes the finite-difference
    Jacobian (the bulk of the fit cost) without changing the converged params."""

    x = np.asarray(x, dtype=float)
    s = max(abs(float(sigma)), _SIGMA_FLOOR)
    e = np.exp(-((x - mu) ** 2) / (2.0 * s * s))
    g = amp * e
    return e, g * (x - mu) / (s * s), g * (x - mu) ** 2 / (s ** 3)


def gaussian_jacobian(x, amp, mu, sigma):
    """Single-Gaussian Jacobian matrix (columns stacked on the last axis)."""

    return np.stack(gaussian_jacobian_columns(x, amp, mu, sigma), axis=-1)


def bimodal_model(x, amp0, mu0, sigma0, amp1, mu1, sigma1):
    """Sum of a dark and a bright Gaussian."""

    return gaussian(x, amp0, mu0, sigma0) + gaussian(x, amp1, mu1, sigma1)


def bimodal_jacobian(x, amp0, mu0, sigma0, amp1, mu1, sigma1):
    """Jacobian of :func:`bimodal_model` (six columns on the last axis)."""

    a0, m0, s0 = gaussian_jacobian_columns(x, amp0, mu0, sigma0)
    a1, m1, s1 = gaussian_jacobian_columns(x, amp1, mu1, sigma1)
    return np.stack([a0, m0, s0, a1, m1, s1], axis=-1)


def normal_cdf(x, mu, sigma):
    """``0.5 (1 + erf((x-mu)/(sigma sqrt 2)))``; scalar in -> float out."""

    sigma = max(abs(float(sigma)), _SIGMA_FLOOR)
    out = 0.5 * (1.0 + erf((np.asarray(x, dtype=float) - mu) / (sigma * sqrt(2.0))))
    return float(out) if out.ndim == 0 else out


def confidence_weighted_fidelity(threshold, mu0, sigma0, weight0, mu1, sigma1, weight1):
    """Confidence-weighted single-threshold readout fidelity for two Gaussians.

    ``weight0/weight1`` are the dark/bright populations (sample counts, or
    Gaussian areas ``amp*sigma`` -- the formula only uses their ratio).  Returns
    ``(fidelity, raw_fidelity, separation)``: the raw threshold fidelity is pulled
    toward 0.5 by a confidence in ``[0, 1]`` that drops when the peaks are poorly
    separated or badly imbalanced, so a single noisy blob does not report a
    falsely high fidelity.  This is the EXACT formula both the saved-calibration
    fidelity (core) and the live histogram fidelity (frontend) use."""

    total_w = float(weight0) + float(weight1)
    if total_w <= 0:
        return float("nan"), float("nan"), float("nan")
    dark_ok = normal_cdf(threshold, mu0, sigma0)
    bright_ok = 1.0 - normal_cdf(threshold, mu1, sigma1)
    raw = (float(weight0) * float(dark_ok) + float(weight1) * float(bright_ok)) / total_w
    separation = abs(float(mu1) - float(mu0)) / sqrt(float(sigma0) ** 2 + float(sigma1) ** 2)
    balance = 2.0 * min(float(weight0), float(weight1)) / total_w
    effective_separation = max(0.0, separation - 2.0)
    confidence = float(np.clip(balance * (1.0 - np.exp(-0.5 * effective_separation * effective_separation)), 0.0, 1.0))
    fidelity = 0.5 + (raw - 0.5) * confidence
    return float(fidelity), float(raw), float(separation)


def finite_mean(a, axis=None):
    """Mean over the FINITE (non-NaN, non-inf) entries along ``axis``; a slice with NO finite entry
    yields NaN.

    Computed DIRECTLY as sum-of-finite / count-of-finite, with the empty (count == 0) slices left NaN
    through a masked ``divide`` -- so, unlike ``np.nanmean``, it NEVER computes a 0/0 and NEVER emits the
    "Mean of empty slice" ``RuntimeWarning``.  An all-NaN slice is a legitimate GAP (a not-yet-measured
    LIVE scan point, an unfilled grid cell, a site empty across every shot), so the right result is a
    silent NaN produced by construction -- not a warning suppressed after the fact.  The ONE gap-safe
    average shared by the live-plot repeat/facet collapse and the per-site shot reduce, so a partly
    filled scan / grid never spams the console with one warning per empty cell, and no ``catch_warnings``
    (global, non-thread-safe, mask-everything) is needed on any hot render / reduce path."""
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    count = np.count_nonzero(finite, axis=axis)
    total = np.sum(np.where(finite, a, 0.0), axis=axis)
    out = np.full(np.shape(count), np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)   # empty slices stay NaN -- no 0/0, no warning
    return out


__all__ = [
    "bimodal_jacobian",
    "bimodal_model",
    "confidence_weighted_fidelity",
    "finite_mean",
    "gaussian",
    "gaussian_jacobian",
    "gaussian_jacobian_columns",
    "normal_cdf",
]
