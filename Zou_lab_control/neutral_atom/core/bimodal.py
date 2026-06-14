"""Bimodal (dark/bright) threshold fitting for single-site readout.

``otsu_threshold`` in :mod:`.analysis` picks a single split point; the Gaussian
split in ``estimate_threshold_fidelity`` then characterizes the two sides.  For
low-photon qCMOS readout the Rb87 pipeline does better: it fits robust dark and
bright peak *cores* (one-sided widths, so a partial-loss bridge between the peaks
does not inflate the bright width), then places the threshold at the point of
minimum total error between the two Gaussians and reports the model fidelity
``0.5*(F_dark + F_bright)``.

Pure functions, numpy + scipy only.  ``fit_bimodal(values) -> BimodalFit`` is the
public entry; per-site thresholds come from :func:`fit_bimodal_per_site`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf


@dataclass(frozen=True)
class BimodalFit:
    """Dark/bright Gaussian-core fit and the optimal threshold between them."""

    threshold: float
    fidelity: float
    dark_mean: float
    dark_sigma: float
    bright_mean: float
    bright_sigma: float
    bright_fraction: float
    dark_fidelity: float
    bright_fidelity: float
    bright_above: bool
    ok: bool
    message: str = ""


def _norm_cdf(x, mu: float, sigma: float):
    sigma = max(float(sigma), 1e-12)
    arr = 0.5 * (1.0 + erf((np.asarray(x, dtype=float) - mu) / (sigma * math.sqrt(2.0))))
    return float(arr) if arr.ndim == 0 else arr


def exact_otsu_threshold(values, *, min_fraction: float = 0.02) -> float:
    """Exact 1-D between-class-variance split on the sorted samples (not binned)."""

    xs = np.asarray(values, dtype=float).ravel()
    xs = np.sort(xs[np.isfinite(xs)])
    n = int(xs.size)
    if n < 4:
        return float("nan")
    min_n = max(2, int(math.ceil(float(min_fraction) * n)))
    if n < 2 * min_n + 1:
        min_n = max(1, n // 4)
    k = np.arange(1, n, dtype=float)
    cs = np.cumsum(xs)
    total = float(cs[-1])
    left_count, right_count = k, float(n) - k
    valid = (left_count >= min_n) & (right_count >= min_n) & (xs[:-1] < xs[1:])
    if not np.any(valid):
        return float(np.median(xs))
    m0 = cs[:-1] / left_count
    m1 = (total - cs[:-1]) / right_count
    score = left_count * right_count * (m1 - m0) ** 2
    score[~valid] = -np.inf
    idx = int(np.argmax(score))
    if not np.isfinite(score[idx]):
        return float(np.median(xs))
    return float(0.5 * (xs[idx] + xs[idx + 1]))


def _one_sided_core_stats(samples: np.ndarray, side: str, sigma_floor: float) -> tuple[float, float, bool]:
    """Robust peak location + one-sided width (away from the inter-peak bridge)."""

    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 4:
        return float("nan"), float("nan"), False
    q16, q50, q84 = np.percentile(s, [15.865525393145708, 50.0, 84.1344746068543])
    sigma = float(q50 - q16) if str(side).lower().startswith("low") else float(q84 - q50)
    alt = float(0.5 * (q84 - q16))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = alt
    if not np.isfinite(sigma) or sigma <= 0:
        med = float(np.median(s))
        sigma = 1.482602218505602 * float(np.median(np.abs(s - med)))
    sigma = max(float(sigma), max(float(sigma_floor), 1e-12))
    return float(q50), sigma, True


def optimal_gaussian_threshold(mu_dark, sigma_dark, mu_bright, sigma_bright, *, prior_dark=0.5, prior_bright=0.5) -> tuple[float, bool]:
    """Threshold minimizing prior-weighted total misclassification of two Gaussians."""

    bright_above = bool(mu_bright >= mu_dark)
    lo, hi = min(mu_dark, mu_bright), max(mu_dark, mu_bright)
    if not np.isfinite([lo, hi, sigma_dark, sigma_bright]).all() or hi <= lo:
        return float(0.5 * (mu_dark + mu_bright)), bright_above

    def err(t: float) -> float:
        if bright_above:
            dark_err = 1.0 - float(_norm_cdf(t, mu_dark, sigma_dark))
            bright_err = float(_norm_cdf(t, mu_bright, sigma_bright))
        else:
            dark_err = float(_norm_cdf(t, mu_dark, sigma_dark))
            bright_err = 1.0 - float(_norm_cdf(t, mu_bright, sigma_bright))
        return prior_dark * dark_err + prior_bright * bright_err

    res = minimize_scalar(err, bounds=(lo, hi), method="bounded")
    return float(res.x if res.success else 0.5 * (lo + hi)), bright_above


def classify_threshold(values, threshold: float, bright_above: bool = True) -> np.ndarray:
    """Boolean bright/dark classification of ``values`` against ``threshold``."""

    arr = np.asarray(values, dtype=float)
    return arr > threshold if bright_above else arr < threshold


def gaussian_fidelity(mu_dark, sigma_dark, mu_bright, sigma_bright, threshold, bright_above) -> tuple[float, float, float]:
    """``(F_dark, F_bright, 0.5*(F_dark+F_bright))`` for a threshold between two Gaussians."""

    if not np.isfinite([mu_dark, sigma_dark, mu_bright, sigma_bright, threshold]).all():
        return float("nan"), float("nan"), float("nan")
    if bright_above:
        f_dark = float(_norm_cdf(threshold, mu_dark, sigma_dark))
        f_bright = 1.0 - float(_norm_cdf(threshold, mu_bright, sigma_bright))
    else:
        f_dark = 1.0 - float(_norm_cdf(threshold, mu_dark, sigma_dark))
        f_bright = float(_norm_cdf(threshold, mu_bright, sigma_bright))
    return f_dark, f_bright, 0.5 * (f_dark + f_bright)


def fit_bimodal(values, *, min_component_fraction: float = 0.01) -> BimodalFit:
    """Fit dark/bright Gaussian cores and the optimal threshold for one site.

    Steps: empirical Otsu split -> one-sided peak-core mean/sigma for each side
    -> optimal Gaussian threshold -> model fidelity.  Falls back to the empirical
    split with NaN fidelity when the sample is too small or one side is empty.
    """

    xx = np.asarray(values, dtype=float).ravel()
    xx = xx[np.isfinite(xx)]
    split = exact_otsu_threshold(xx)
    if xx.size < 8 or not np.isfinite(split):
        return BimodalFit(float(split), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), float("nan"), True, False, "too few samples for bimodal fit")
    lo = xx[xx <= split]
    hi = xx[xx > split]
    min_count = max(4, int(math.ceil(float(min_component_fraction) * xx.size)))
    if lo.size < min_count or hi.size < min_count:
        return BimodalFit(float(split), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                          float(hi.size / xx.size) if xx.size else float("nan"),
                          float("nan"), float("nan"), True, False, "one split side too small")
    full_sd = float(np.std(xx)) if xx.size > 1 else 1.0
    floor = max(1e-6 * full_sd, 1e-12)
    mu0, s0, ok0 = _one_sided_core_stats(lo, "low", floor)
    mu1, s1, ok1 = _one_sided_core_stats(hi, "high", floor)
    bright_fraction = float(hi.size / xx.size)
    if not (ok0 and ok1) or not np.isfinite([mu0, s0, mu1, s1]).all() or mu1 <= mu0:
        return BimodalFit(float(split), float("nan"), float(mu0), float(s0), float(mu1), float(s1),
                          bright_fraction, float("nan"), float("nan"), True, False, "core fit failed")
    threshold, bright_above = optimal_gaussian_threshold(mu0, s0, mu1, s1)
    f_dark, f_bright, fidelity = gaussian_fidelity(mu0, s0, mu1, s1, threshold, bright_above)
    sep = float((mu1 - mu0) / max(s0 + s1, 1e-12))
    return BimodalFit(
        threshold=float(threshold),
        fidelity=float(fidelity),
        dark_mean=float(mu0),
        dark_sigma=float(s0),
        bright_mean=float(mu1),
        bright_sigma=float(s1),
        bright_fraction=bright_fraction,
        dark_fidelity=float(f_dark),
        bright_fidelity=float(f_bright),
        bright_above=bool(bright_above),
        ok=bool(sep > 0.5),
        message=f"bimodal core fit; sep={sep:.2f}",
    )


def fit_bimodal_per_site(signals) -> tuple[np.ndarray, list[BimodalFit]]:
    """Fit a bimodal threshold for every site column of a ``(shots, N_sites)`` array.

    Returns ``(thresholds, fits)`` where ``thresholds`` is ``(N_sites,)``.
    """

    arr = np.asarray(signals, dtype=float)
    if arr.ndim != 2:
        raise ValueError("signals must be a (shots, n_sites) 2D array.")
    fits = [fit_bimodal(arr[:, i]) for i in range(arr.shape[1])]
    thresholds = np.asarray([f.threshold for f in fits], dtype=float)
    return thresholds, fits


__all__ = [
    "BimodalFit",
    "classify_threshold",
    "exact_otsu_threshold",
    "fit_bimodal",
    "fit_bimodal_per_site",
    "gaussian_fidelity",
    "optimal_gaussian_threshold",
]
