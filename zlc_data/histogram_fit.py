"""Headless binned-histogram fit used by presentation projections.

This is the single data-layer owner of the Gaussian fit drawn by the rolling
monitor and Distribution plot. It consumes already-binned counts; it is not an
authoritative experiment fit or readout calibration and never changes an
experiment result. Frontends only draw this result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, sqrt
from numbers import Integral

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from ._arrays import immutable_array


_SIGMA_FLOOR = 1e-12


def _gaussian(x, amplitude, mean, sigma):
    """Evaluate the presentation histogram's Gaussian component."""

    scale = max(abs(float(sigma)), _SIGMA_FLOOR)
    coordinate = np.asarray(x, dtype=np.float64)
    return amplitude * np.exp(-np.square(coordinate - mean) / (2.0 * scale * scale))


def _gaussian_jacobian_columns(x, amplitude, mean, sigma):
    coordinate = np.asarray(x, dtype=np.float64)
    scale = max(abs(float(sigma)), _SIGMA_FLOOR)
    exponential = np.exp(-np.square(coordinate - mean) / (2.0 * scale * scale))
    gaussian = amplitude * exponential
    return (
        exponential,
        gaussian * (coordinate - mean) / (scale * scale),
        gaussian * np.square(coordinate - mean) / (scale**3),
    )


def _gaussian_jacobian(x, amplitude, mean, sigma):
    return np.stack(
        _gaussian_jacobian_columns(x, amplitude, mean, sigma),
        axis=-1,
    )


def _bimodal_model(x, amp0, mu0, sigma0, amp1, mu1, sigma1):
    return _gaussian(x, amp0, mu0, sigma0) + _gaussian(
        x,
        amp1,
        mu1,
        sigma1,
    )


def _bimodal_jacobian(x, amp0, mu0, sigma0, amp1, mu1, sigma1):
    return np.stack(
        (
            *_gaussian_jacobian_columns(x, amp0, mu0, sigma0),
            *_gaussian_jacobian_columns(x, amp1, mu1, sigma1),
        ),
        axis=-1,
    )


def confidence_weighted_fidelity(
    threshold,
    mu0,
    sigma0,
    weight0,
    mu1,
    sigma1,
    weight1,
):
    """Return the established display-only confidence-weighted fidelity."""

    values = tuple(
        float(value)
        for value in (threshold, mu0, sigma0, weight0, mu1, sigma1, weight1)
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("fidelity inputs must be finite")
    threshold, mu0, sigma0, weight0, mu1, sigma1, weight1 = values
    if sigma0 < 0.0 or sigma1 < 0.0:
        raise ValueError("fidelity sigmas must be non-negative")
    if weight0 < 0.0 or weight1 < 0.0:
        raise ValueError("fidelity weights must be non-negative")

    def normal_cdf(value, mean, sigma):
        scale = max(abs(float(sigma)), _SIGMA_FLOOR)
        result = 0.5 * (
            1.0
            + erf(
                (np.asarray(value, dtype=np.float64) - mean)
                / (scale * sqrt(2.0))
            )
        )
        return float(result) if result.ndim == 0 else result

    total_weight = weight0 + weight1
    if total_weight <= 0:
        return float("nan"), float("nan"), float("nan")
    dark_ok = normal_cdf(threshold, mu0, sigma0)
    bright_ok = 1.0 - normal_cdf(threshold, mu1, sigma1)
    raw = (
        float(weight0) * float(dark_ok)
        + float(weight1) * float(bright_ok)
    ) / total_weight
    scale0 = max(abs(float(sigma0)), _SIGMA_FLOOR)
    scale1 = max(abs(float(sigma1)), _SIGMA_FLOOR)
    separation = abs(float(mu1) - float(mu0)) / hypot(scale0, scale1)
    balance = 2.0 * min(float(weight0), float(weight1)) / total_weight
    effective_separation = max(0.0, separation - 2.0)
    confidence = float(
        np.clip(
            balance
            * (
                1.0
                - np.exp(
                    -0.5 * effective_separation * effective_separation
                )
            ),
            0.0,
            1.0,
        )
    )
    fidelity = 0.5 + (raw - 0.5) * confidence
    return float(fidelity), float(raw), float(separation)


@dataclass(frozen=True, slots=True)
class HistogramFitResult:
    requested_mode: str
    component_count: int
    parameters: np.ndarray
    valid: bool
    separated: bool
    threshold: float | None
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.requested_mode, str):
            raise TypeError("histogram fit requested_mode must be text")
        mode = str(self.requested_mode).strip().lower()
        if mode not in {"none", "single", "double"}:
            raise ValueError(f"unknown histogram fit mode {self.requested_mode!r}")
        expected = {0: 0, 1: 3, 2: 6}
        if isinstance(self.component_count, (bool, np.bool_)) or not isinstance(
            self.component_count,
            Integral,
        ):
            raise TypeError("component_count must be an integer")
        count = int(self.component_count)
        if count not in expected:
            raise ValueError("component_count must be 0, 1, or 2")
        source = np.asarray(self.parameters, dtype=np.dtype("<f8")).reshape(-1)
        if source.size != expected[count]:
            raise ValueError("histogram fit parameter count does not match its model")
        if not isinstance(self.valid, (bool, np.bool_)) or not isinstance(
            self.separated,
            (bool, np.bool_),
        ):
            raise TypeError("histogram fit validity flags must be bool")
        valid = bool(self.valid)
        separated = bool(self.separated)
        if valid != (count > 0):
            raise ValueError(
                "a valid histogram fit must carry one fitted model and an "
                "invalid fit must carry none"
            )
        if mode == "none" and valid:
            raise ValueError("disabled histogram fitting cannot carry a valid result")
        if mode == "single" and count not in (0, 1):
            raise ValueError("single histogram mode cannot carry a bimodal result")
        if source.size and not np.all(np.isfinite(source)):
            raise ValueError("valid histogram fit parameters must be finite")
        if count:
            amplitudes = source[0::3]
            sigmas = source[2::3]
            if np.any(amplitudes < 0.0):
                raise ValueError("histogram fit amplitudes must be non-negative")
            if np.any(sigmas <= 0.0):
                raise ValueError("histogram fit sigmas must be positive")
        if count == 2 and source[1] > source[4]:
            raise ValueError("bimodal histogram means must be ordered")
        if separated and (not valid or count != 2):
            raise ValueError("only a valid bimodal fit can be separated")
        threshold = self.threshold
        if (threshold is not None) != separated:
            raise ValueError(
                "a separated bimodal fit must carry exactly one threshold"
            )
        if threshold is not None:
            threshold = float(threshold)
            if not isfinite(threshold):
                raise ValueError("histogram fit threshold must be finite")
            if not source[1] < threshold < source[4]:
                raise ValueError(
                    "histogram fit threshold must lie between ordered means"
                )
        if not isinstance(self.status, str):
            raise TypeError("histogram fit status must be text")
        status = self.status.strip()
        if not status:
            raise ValueError("histogram fit status must not be empty")
        owned = immutable_array(
            source,
            dtype=np.dtype("<f8"),
            shape=(expected[count],),
        )
        object.__setattr__(self, "requested_mode", mode)
        object.__setattr__(self, "component_count", count)
        object.__setattr__(self, "parameters", owned)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "separated", separated)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "status", status)

    @classmethod
    def invalid(cls, mode: str, status: str) -> "HistogramFitResult":
        return cls(mode, 0, np.empty(0), False, False, None, status)

    @property
    def single_parameters(self) -> tuple[float, float, float] | None:
        if not self.valid or self.component_count != 1:
            return None
        return tuple(float(value) for value in self.parameters)

    @property
    def bimodal_parameters(self) -> np.ndarray | None:
        return self.parameters if self.valid and self.component_count == 2 else None

    def curves(
        self,
        x,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        if not self.valid:
            raise ValueError(f"cannot evaluate invalid histogram fit: {self.status}")
        x = np.asarray(x, dtype=np.float64)
        if self.component_count == 1:
            return None, None, _gaussian(x, *self.parameters)
        left = _gaussian(x, *self.parameters[:3])
        right = _gaussian(x, *self.parameters[3:])
        return left, right, left + right

    def evaluate(self, x) -> np.ndarray:
        return self.curves(x)[2]


def _solve(function, jacobian, x, y, seed, bounds):
    try:
        parameters, _covariance = curve_fit(
            function,
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            p0=np.asarray(seed, dtype=np.float64),
            bounds=bounds,
            jac=jacobian,
            maxfev=20_000,
        )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"
    prediction = np.asarray(function(x, *parameters), dtype=np.float64)
    if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(prediction)):
        return None, "solver returned non-finite values"
    return np.asarray(parameters, dtype=np.float64), "ok"


def fit_histogram(
    edges,
    counts,
    mode: str,
) -> HistogramFitResult:
    """Fit one or two zero-offset Gaussians using main's exact policy."""

    if not isinstance(mode, str):
        raise TypeError("histogram fit mode must be text")
    requested = str(mode).strip().lower()
    if requested not in {"none", "single", "double"}:
        raise ValueError("histogram fit mode must be 'none', 'single', or 'double'")
    edges = np.asarray(edges, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if requested == "none":
        return HistogramFitResult.invalid(requested, "fit disabled")
    if edges.ndim != 1 or counts.ndim != 1 or edges.size != counts.size + 1:
        return HistogramFitResult.invalid(requested, "histogram edges/counts mismatch")
    if (
        counts.size < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.isfinite(counts))
        or np.any(np.diff(edges) <= 0)
        or np.any(counts < 0)
    ):
        return HistogramFitResult.invalid(
            requested,
            "histogram needs increasing finite edges and non-negative counts",
        )
    sample_count = float(np.sum(counts))
    if sample_count < 6 or np.count_nonzero(counts) < 2:
        return HistogramFitResult.invalid(
            requested,
            "histogram has insufficient occupied bins",
        )

    centers = (edges[:-1] + edges[1:]) / 2.0
    span = float(edges[-1] - edges[0]) or 1.0

    def weighted_stats(coordinates, weights) -> tuple[float, float]:
        total = float(np.sum(weights))
        mean = float(np.sum(weights * coordinates) / total)
        sigma = float(
            np.sqrt(
                max(
                    float(np.sum(weights * np.square(coordinates - mean)) / total),
                    0.0,
                )
            )
        )
        return mean, sigma

    def single(status: str = "ok") -> HistogramFitResult:
        mean, sigma = weighted_stats(centers, counts)
        sigma = max(sigma, span / 40.0, 1e-9)
        peak = max(float(np.max(counts)), 1.0)
        parameters, diagnostic = _solve(
            _gaussian,
            _gaussian_jacobian,
            centers,
            counts,
            (peak, mean, sigma),
            (
                np.asarray((0.0, edges[0], max(span / 200.0, 1e-12))),
                np.asarray((
                    max(peak * 5.0, 1.0),
                    edges[-1],
                    max(span * 2.0, 1e-12),
                )),
            ),
        )
        if parameters is None:
            return HistogramFitResult.invalid(
                requested,
                f"single histogram fit failed: {diagnostic}",
            )
        return HistogramFitResult(
            requested,
            1,
            parameters,
            True,
            False,
            None,
            status,
        )

    if requested == "single":
        return single()

    weight_left = np.cumsum(counts)[:-1]
    moment_left = np.cumsum(counts * centers)[:-1]
    weight_right = sample_count - weight_left
    total_moment = float(np.sum(counts * centers))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(
            (weight_left > 0) & (weight_right > 0),
            weight_left
            * weight_right
            * np.square(
                moment_left / weight_left
                - (total_moment - moment_left) / weight_right
            ),
            0.0,
        )
    split = int(np.argmax(score))
    left_x, left_counts = centers[: split + 1], counts[: split + 1]
    right_x, right_counts = centers[split + 1 :], counts[split + 1 :]
    if float(np.sum(left_counts)) < 2 or float(np.sum(right_counts)) < 2:
        half = int(np.searchsorted(np.cumsum(counts), sample_count / 2.0))
        left_x, left_counts = centers[: half + 1], counts[: half + 1]
        right_x, right_counts = centers[half + 1 :], counts[half + 1 :]
        if float(np.sum(left_counts)) <= 0 or float(np.sum(right_counts)) <= 0:
            return single("double seed split failed; single fallback")

    mean_left, sigma_left = weighted_stats(left_x, left_counts)
    mean_right, sigma_right = weighted_stats(right_x, right_counts)
    if mean_left > mean_right:
        mean_left, mean_right = mean_right, mean_left
        sigma_left, sigma_right = sigma_right, sigma_left
    sigma_left = max(sigma_left, span / 40.0, 1e-9)
    sigma_right = max(sigma_right, span / 40.0, 1e-9)

    def amplitude_near(mean: float) -> float:
        index = int(np.clip(np.searchsorted(centers, mean), 0, counts.size - 1))
        return max(float(counts[index]), 1.0)

    amplitude_left = amplitude_near(mean_left)
    amplitude_right = amplitude_near(mean_right)
    parameters, diagnostic = _solve(
        _bimodal_model,
        _bimodal_jacobian,
        centers,
        counts,
        (
            amplitude_left,
            mean_left,
            sigma_left,
            amplitude_right,
            mean_right,
            sigma_right,
        ),
        (
            np.asarray((
                0.0,
                edges[0],
                span / 200.0,
                0.0,
                edges[0],
                span / 200.0,
            )),
            np.asarray((
                max(amplitude_left * 5.0, 1.0),
                edges[-1],
                span * 2.0,
                max(amplitude_right * 5.0, 1.0),
                edges[-1],
                span * 2.0,
            )),
        ),
    )
    if parameters is None:
        return single(f"double fit failed ({diagnostic}); single fallback")
    if parameters[1] > parameters[4]:
        parameters = parameters[np.asarray((3, 4, 5, 0, 1, 2))]
    separation = abs(parameters[4] - parameters[1]) / (
        abs(parameters[2]) + abs(parameters[5]) + 1e-12
    )
    separated = bool(separation >= 1.5)
    threshold = None
    if separated and parameters[4] > parameters[1]:
        between = np.linspace(float(parameters[1]), float(parameters[4]), 400)
        candidate = float(
            between[
                int(
                    np.argmin(
                        np.abs(
                            _gaussian(between, *parameters[:3])
                            - _gaussian(between, *parameters[3:])
                        )
                    )
                )
            ]
        )
        if float(parameters[1]) < candidate < float(parameters[4]):
            threshold = candidate
        else:
            # A visually separated pair with no interior component crossing
            # has no defensible cut.  Keep the two fitted curves but do not
            # claim threshold/fidelity semantics.
            separated = False
    return HistogramFitResult(
        requested,
        2,
        parameters,
        True,
        separated,
        threshold,
        "ok",
    )


__all__ = [
    "HistogramFitResult",
    "confidence_weighted_fidelity",
    "fit_histogram",
]
