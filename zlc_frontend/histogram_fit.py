"""Bounded presentation fit used by the established histogram surfaces.

This is the single owner of the small O(bin-count) Gaussian fit drawn by the
rolling monitor and Distribution plot.  It consumes already-binned counts; it
is not the general dataset fit API and never changes an experiment result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from zlc_data.readout_math import (
    bimodal_jacobian,
    bimodal_model,
    gaussian,
    gaussian_jacobian,
)


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
        mode = str(self.requested_mode).strip().lower()
        if mode not in {"none", "single", "double"}:
            raise ValueError(f"unknown histogram fit mode {self.requested_mode!r}")
        expected = {0: 0, 1: 3, 2: 6}
        count = int(self.component_count)
        if count not in expected:
            raise ValueError("component_count must be 0, 1, or 2")
        source = np.asarray(self.parameters, dtype=np.float64).reshape(-1)
        if source.size != expected[count]:
            raise ValueError("histogram fit parameter count does not match its model")
        owned = np.frombuffer(source.tobytes(), dtype=np.dtype("<f8"))
        owned.setflags(write=False)
        object.__setattr__(self, "requested_mode", mode)
        object.__setattr__(self, "component_count", count)
        object.__setattr__(self, "parameters", owned)
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "separated", bool(self.separated))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "status", str(self.status))

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
            return None, None, gaussian(x, *self.parameters)
        left = gaussian(x, *self.parameters[:3])
        right = gaussian(x, *self.parameters[3:])
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
    except Exception as error:
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
            gaussian,
            gaussian_jacobian,
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
        bimodal_model,
        bimodal_jacobian,
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
        threshold = float(
            between[
                int(
                    np.argmin(
                        np.abs(
                            gaussian(between, *parameters[:3])
                            - gaussian(between, *parameters[3:])
                        )
                    )
                )
            ]
        )
    return HistogramFitResult(
        requested,
        2,
        parameters,
        True,
        separated,
        threshold,
        "ok",
    )


__all__ = ["HistogramFitResult", "fit_histogram"]
