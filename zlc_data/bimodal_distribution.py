"""Typed, model-owned bimodal analysis for histogram presentation.

The frontend owns drawing, but it must not assemble a synthetic dataset or
reimplement Gaussian formulae merely to analyse one already-binned
distribution.  This module is the narrow bridge: it binds a legitimate
histogram fit internally, delegates the catalogue's exact single-cell solver,
and returns immutable values that a renderer can only present.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from zlc_storage.canonical import finite_real

from ._arrays import immutable_array
from .axis import HISTOGRAM_BIN, REPEAT, AxisId, AxisSpec
from .fit_contract import (
    BoundFit,
    FitBatchStatus,
    FitCancelled,
    FitDeadlineExceeded,
    FitSpec,
)
from .fit_model import evaluate_fit_model_components
from .fit_problem import bind_fit
from .layout import PointLayout
from .schema import DatasetSchema, ValueSchema


_FLOAT64 = np.dtype("<f8")
_HISTOGRAM_REPEAT_AXIS_ID = AxisId("zlc_data.bimodal-distribution.repeat")
_HISTOGRAM_BIN_AXIS_ID = AxisId("zlc_data.bimodal-distribution.bin")
# Main's established display contract calls the populations resolved only when
# their mean splitting is at least 1.5 times the sum of their fitted sigmas.
# Keep that physical threshold here with the model rather than letting each
# renderer invent a visually similar but scientifically different rule.
_MINIMUM_SIGMA_SUM_SEPARATION = 1.5


def _immutable_float64(values: object, shape: tuple[int, ...]) -> np.ndarray:
    return immutable_array(
        np.asarray(values, dtype=_FLOAT64),
        dtype=_FLOAT64,
        shape=shape,
    )


@dataclass(frozen=True, eq=False)
class BimodalDistributionAnalysis:
    """One discriminated, immutable result for an already-binned distribution.

    ``status`` is the discriminator.  A converged result carries ordered
    ``(left, right, total)`` predictions aligned with ``coordinates``.  A
    failed result carries a stable diagnostic and no invented curves.
    A fit may converge while ``threshold`` remains ``None``: that means two
    mathematical components were fitted but the evidence did not describe a
    resolved, uniquely separable distribution.
    """

    status: FitBatchStatus
    diagnostic: str
    coordinates: np.ndarray
    component_predictions: tuple[np.ndarray, ...]
    threshold: float | None
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, FitBatchStatus):
            raise TypeError("status must be FitBatchStatus")
        if not isinstance(self.diagnostic, str):
            raise TypeError("diagnostic must be str")
        diagnostic = " ".join(self.diagnostic.split())
        object.__setattr__(self, "diagnostic", diagnostic)
        coordinates = np.asarray(self.coordinates, dtype=_FLOAT64)
        if coordinates.ndim != 1:
            raise ValueError("coordinates must be one-dimensional")
        object.__setattr__(
            self,
            "coordinates",
            _immutable_float64(coordinates, coordinates.shape),
        )
        converged = self.status is FitBatchStatus.CONVERGED
        predictions = tuple(
            _immutable_float64(values, coordinates.shape)
            for values in self.component_predictions
        )
        object.__setattr__(self, "component_predictions", predictions)
        threshold = self.threshold
        if threshold is not None:
            threshold = finite_real(threshold, "threshold")
            object.__setattr__(self, "threshold", threshold)

        if converged:
            if diagnostic:
                raise ValueError(
                    "converged bimodal analysis cannot carry a failure diagnostic"
                )
            if len(predictions) != 3:
                raise ValueError(
                    "converged bimodal analysis requires left/right/total predictions"
                )
            if any(not np.all(np.isfinite(values)) for values in predictions):
                raise ValueError("converged bimodal analysis payload must be finite")
        else:
            if not diagnostic:
                raise ValueError("failed bimodal analysis requires a diagnostic")
            if predictions:
                raise ValueError("failed bimodal analysis cannot carry fitted curves")
            if threshold is not None:
                raise ValueError("failed bimodal analysis cannot publish a threshold")

    @property
    def separated(self) -> bool:
        return self.threshold is not None


def analyze_bimodal_distribution(
    bin_centers: object,
    bin_counts: object,
    *,
    cancel_check: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> BimodalDistributionAnalysis:
    """Fit and analyse one already-binned distribution through zlc_data.

    The input vectors are the authoritative histogram-bin centres and counts;
    the function neither reconstructs samples nor creates a ``DatasetSnapshot``.
    Cancellation/deadline abort the call just like :meth:`BoundFit.run`.
    Expected fit failures remain typed results; malformed API inputs raise.
    """

    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable or None")
    if deadline_monotonic is not None:
        deadline_monotonic = finite_real(
            deadline_monotonic,
            "deadline_monotonic",
        )
    centers = _real_vector(bin_centers, "bin_centers")
    counts = _real_vector(bin_counts, "bin_counts")
    if centers.shape != counts.shape:
        raise ValueError("bin_centers and bin_counts must have the same shape")
    if np.any(counts < 0.0):
        raise ValueError("bin_counts cannot be negative")
    if centers.size > 1 and np.any(np.diff(centers) <= 0.0):
        raise ValueError("bin_centers must be strictly increasing")

    # Keep importing zlc_data metadata-only.  SciPy enters the process only
    # when a caller actually asks to solve a distribution, just as BoundFit.run
    # does for formal dataset fitting.
    from .fit_solver import (
        _CellFailure,
        _check_host_abort,
        _fit_cell,
        _fit_error_text,
    )

    _check_host_abort(cancel_check, deadline_monotonic)
    if centers.size == 0:
        return _failed_analysis(
            FitBatchStatus.NO_VALID_DATA,
            "distribution has no authoritative histogram bins",
            centers,
        )
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(counts)):
        return _failed_analysis(
            FitBatchStatus.NUMERIC_ERROR,
            "authoritative histogram coordinates or counts contain non-finite values",
            centers,
        )

    bound = _bind_bimodal_histogram(centers)
    if counts.size < bound.minimum_observation_count:
        return _failed_analysis(
            FitBatchStatus.INSUFFICIENT_POINTS,
            (
                f"bimodal fit requires at least {bound.minimum_observation_count} "
                f"histogram bins; got {counts.size}"
            ),
            centers,
        )
    try:
        success = _fit_cell(
            bound,
            (centers,),
            counts,
            cancel_check=cancel_check,
            host_deadline=deadline_monotonic,
        )
    except (FitCancelled, FitDeadlineExceeded):
        raise
    except _CellFailure as exc:
        return _failed_analysis(
            exc.status,
            _fit_error_text(str(exc)),
            centers,
        )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return _failed_analysis(
            FitBatchStatus.NUMERIC_ERROR,
            _fit_error_text(f"numeric failure: {exc}"),
            centers,
        )

    components = evaluate_fit_model_components(
        bound.model,
        (centers,),
        success.parameters,
    )
    threshold = _resolved_bimodal_threshold(
        success.parameters,
        support=(float(centers[0]), float(centers[-1])),
    )
    return BimodalDistributionAnalysis(
        status=FitBatchStatus.CONVERGED,
        diagnostic="",
        coordinates=centers,
        component_predictions=components,
        threshold=threshold,
    )


def _bind_bimodal_histogram(coordinates: np.ndarray) -> BoundFit:
    repeat_axis = AxisSpec(
        _HISTOGRAM_REPEAT_AXIS_ID,
        "Repeat",
        REPEAT,
        1,
    )
    bin_axis = AxisSpec(
        _HISTOGRAM_BIN_AXIS_ID,
        "Histogram bin",
        HISTOGRAM_BIN,
        coordinates.size,
        tuple(float(value) for value in coordinates),
    )
    schema = DatasetSchema(
        repeat_axis,
        (bin_axis,),
        PointLayout.rect_c((bin_axis.size,)),
        ValueSchema.scalar(_FLOAT64, "count"),
    )
    return bind_fit(
        FitSpec(
            schema.fingerprint,
            None,
            (bin_axis.axis_id,),
            (repeat_axis.axis_id,),
            "bimodal_gaussian",
        ),
        schema,
    )


def _real_vector(value: object, field: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    if source.dtype.kind not in "biuf":
        raise TypeError(f"{field} must contain real numeric values")
    converted = np.asarray(source, dtype=_FLOAT64)
    if source.dtype.kind in "iu" and any(
        int(float(item)) != int(item) for item in source
    ):
        raise ValueError(
            f"{field} contains an integer not exactly representable as float64"
        )
    return converted


def _failed_analysis(
    status: FitBatchStatus,
    diagnostic: str,
    coordinates: np.ndarray,
) -> BimodalDistributionAnalysis:
    return BimodalDistributionAnalysis(
        status=status,
        diagnostic=diagnostic,
        coordinates=coordinates,
        component_predictions=(),
        threshold=None,
    )


def _resolved_bimodal_threshold(
    parameters: np.ndarray,
    *,
    support: tuple[float, float],
) -> float | None:
    """Return the sole component-height crossing for a resolved Gaussian pair.

    This is a display diagnostic for the fitted histogram components.  It is
    deliberately not the equal-prior decision threshold owned by neutral-atom
    readout calibration.
    """

    (
        center,
        splitting,
        left_amplitude,
        left_sigma,
        right_amplitude,
        right_sigma,
    ) = (float(value) for value in parameters)
    if not np.all(
        np.isfinite(
            (
                center,
                splitting,
                left_amplitude,
                left_sigma,
                right_amplitude,
                right_sigma,
            )
        )
    ):
        return None
    if (
        splitting <= 0.0
        or left_amplitude <= 0.0
        or right_amplitude <= 0.0
        or left_sigma <= 0.0
        or right_sigma <= 0.0
    ):
        return None
    support_low, support_high = support
    left_mean = center - splitting / 2.0
    right_mean = center + splitting / 2.0
    if not support_low <= left_mean < right_mean <= support_high:
        return None
    separation = splitting / (left_sigma + right_sigma)
    if (
        not math.isfinite(separation)
        or separation < _MINIMUM_SIGMA_SUM_SEPARATION
    ):
        return None

    # Work with log(component_left/component_right).  Strict opposite endpoint
    # signs prove that each component dominates at its own mean and, because
    # this log ratio is quadratic, that exactly one simple root lies between
    # the means.  Bisection in normalized t avoids cancellation from a large
    # absolute coordinate offset.
    log_amplitude_ratio = math.log(left_amplitude / right_amplitude)
    q_left = log_amplitude_ratio + splitting**2 / (2.0 * right_sigma**2)
    q_right = log_amplitude_ratio - splitting**2 / (2.0 * left_sigma**2)
    if not q_left > 0.0 or not q_right < 0.0:
        return None

    def log_ratio(t: float) -> float:
        return (
            log_amplitude_ratio
            - (splitting * t) ** 2 / (2.0 * left_sigma**2)
            + (splitting * (1.0 - t)) ** 2 / (2.0 * right_sigma**2)
        )

    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if middle == low or middle == high:
            break
        if log_ratio(middle) > 0.0:
            low = middle
        else:
            high = middle
    threshold = left_mean + splitting * ((low + high) / 2.0)
    if not support_low < threshold < support_high or not math.isfinite(threshold):
        return None
    return float(threshold)


__all__ = [
    "BimodalDistributionAnalysis",
    "analyze_bimodal_distribution",
]
