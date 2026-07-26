"""Pure bimodal readout statistics shared by readout capabilities.

Calibration uses the fitted components to establish reference labels, while
Readout Duration Fidelity uses the same physical discriminator for each scan
point.  Neither leaf owns this algorithm and this module has no capture,
artifact, runtime, or presentation dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BimodalFit:
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


def _normal_cdf(x, mean: float, sigma: float):
    from scipy.special import erf

    sigma = max(abs(float(sigma)), 1e-12)
    result = 0.5 * (
        1.0
        + erf((np.asarray(x, dtype=float) - mean) / (sigma * math.sqrt(2.0)))
    )
    return float(result) if result.ndim == 0 else result


def _exact_otsu_threshold(values: np.ndarray, min_fraction: float = 0.02) -> float:
    samples = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    count = int(samples.size)
    if count < 4:
        return float("nan")
    minimum = max(2, int(math.ceil(min_fraction * count)))
    if count < 2 * minimum + 1:
        minimum = max(1, count // 4)
    positions = np.arange(1, count, dtype=float)
    cumulative = np.cumsum(samples)
    total = float(cumulative[-1])
    left_count = positions
    right_count = float(count) - positions
    valid = (
        (left_count >= minimum)
        & (right_count >= minimum)
        & (samples[:-1] < samples[1:])
    )
    if not np.any(valid):
        return float(np.median(samples))
    left_mean = cumulative[:-1] / left_count
    right_mean = (total - cumulative[:-1]) / right_count
    score = left_count * right_count * (right_mean - left_mean) ** 2
    score[~valid] = -np.inf
    index = int(np.argmax(score))
    if not np.isfinite(score[index]):
        return float(np.median(samples))
    return float(0.5 * (samples[index] + samples[index + 1]))


def _one_sided_core_stats(
    samples: np.ndarray,
    side: str,
    sigma_floor: float,
) -> tuple[float, float, bool]:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < 4:
        return float("nan"), float("nan"), False
    q16, q50, q84 = np.percentile(
        samples,
        [15.865525393145708, 50.0, 84.1344746068543],
    )
    sigma = float(q50 - q16) if side == "low" else float(q84 - q50)
    alternative = float(0.5 * (q84 - q16))
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = alternative
    if not math.isfinite(sigma) or sigma <= 0:
        median = float(np.median(samples))
        sigma = 1.482602218505602 * float(np.median(np.abs(samples - median)))
    return float(q50), max(float(sigma), sigma_floor, 1e-12), True


def optimal_gaussian_threshold(
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
) -> tuple[float, bool]:
    """Return the equal-prior threshold and physical bright direction."""

    from scipy.optimize import minimize_scalar

    bright_above = bool(bright_mean >= dark_mean)
    lower, upper = min(dark_mean, bright_mean), max(dark_mean, bright_mean)
    if (
        not np.isfinite([lower, upper, dark_sigma, bright_sigma]).all()
        or upper <= lower
    ):
        return float(0.5 * (dark_mean + bright_mean)), bright_above

    def error(threshold: float) -> float:
        if bright_above:
            dark_error = 1.0 - float(_normal_cdf(threshold, dark_mean, dark_sigma))
            bright_error = float(_normal_cdf(threshold, bright_mean, bright_sigma))
        else:
            dark_error = float(_normal_cdf(threshold, dark_mean, dark_sigma))
            bright_error = 1.0 - float(
                _normal_cdf(threshold, bright_mean, bright_sigma)
            )
        return 0.5 * (dark_error + bright_error)

    result = minimize_scalar(error, bounds=(lower, upper), method="bounded")
    threshold = result.x if result.success else 0.5 * (lower + upper)
    return float(threshold), bright_above


def gaussian_fidelity(
    dark_mean: float,
    dark_sigma: float,
    bright_mean: float,
    bright_sigma: float,
    threshold: float,
    bright_above: bool,
) -> tuple[float, float, float]:
    """Return dark, bright, and balanced fidelity for two Gaussians."""

    if not np.isfinite(
        [dark_mean, dark_sigma, bright_mean, bright_sigma, threshold]
    ).all():
        return float("nan"), float("nan"), float("nan")
    if bright_above:
        dark = float(_normal_cdf(threshold, dark_mean, dark_sigma))
        bright = 1.0 - float(_normal_cdf(threshold, bright_mean, bright_sigma))
    else:
        dark = 1.0 - float(_normal_cdf(threshold, dark_mean, dark_sigma))
        bright = float(_normal_cdf(threshold, bright_mean, bright_sigma))
    return dark, bright, 0.5 * (dark + bright)


def fit_bimodal(
    values: object,
    *,
    min_component_fraction: float = 0.01,
) -> BimodalFit:
    """Fit a robust two-state fluorescence discriminator to finite samples."""

    samples = np.asarray(values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    split = _exact_otsu_threshold(samples)
    if samples.size < 8 or not math.isfinite(split):
        return BimodalFit(
            split,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            True,
            False,
        )
    low = samples[samples <= split]
    high = samples[samples > split]
    minimum = max(4, int(math.ceil(min_component_fraction * samples.size)))
    if low.size < minimum or high.size < minimum:
        return BimodalFit(
            split,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float(high.size / samples.size),
            float("nan"),
            float("nan"),
            True,
            False,
        )
    full_sigma = float(np.std(samples)) if samples.size > 1 else 1.0
    floor = max(1e-6 * full_sigma, 1e-12)
    dark_mean, dark_sigma, dark_ok = _one_sided_core_stats(low, "low", floor)
    bright_mean, bright_sigma, bright_ok = _one_sided_core_stats(high, "high", floor)
    bright_fraction = float(high.size / samples.size)
    if (
        not (dark_ok and bright_ok)
        or not np.isfinite(
            [dark_mean, dark_sigma, bright_mean, bright_sigma]
        ).all()
        or bright_mean <= dark_mean
    ):
        return BimodalFit(
            split,
            float("nan"),
            dark_mean,
            dark_sigma,
            bright_mean,
            bright_sigma,
            bright_fraction,
            float("nan"),
            float("nan"),
            True,
            False,
        )
    threshold, bright_above = optimal_gaussian_threshold(
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
    )
    dark_fidelity, bright_fidelity, fidelity = gaussian_fidelity(
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
        threshold,
        bright_above,
    )
    separation = (bright_mean - dark_mean) / max(dark_sigma + bright_sigma, 1e-12)
    return BimodalFit(
        threshold,
        fidelity,
        dark_mean,
        dark_sigma,
        bright_mean,
        bright_sigma,
        bright_fraction,
        dark_fidelity,
        bright_fidelity,
        bright_above,
        bool(separation > 0.5),
    )


__all__ = [
    "BimodalFit",
    "fit_bimodal",
    "gaussian_fidelity",
    "optimal_gaussian_threshold",
]
