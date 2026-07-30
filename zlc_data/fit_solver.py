"""Nonlinear fit execution and solver-owned distribution analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np
from zlc_storage.canonical import finite_real

from ._arrays import immutable_array
from .axis import HISTOGRAM_BIN, REPEAT, AxisId, AxisSourceRef, AxisSpec
from .fit_contract import (
    BoundFit,
    FitBatchStatus,
    FitCancelled,
    FitDeadlineExceeded,
    FitParameterConstraint,
    FitResultBatch,
    FitSpec,
)
from .fit_model import (
    FitParameterDomain,
    _radial_gaussian_center_axis_basis,
    _radial_gaussian_center_jacobian,
    _spatial_radial_center_seeds,
    evaluate_fit_model,
    evaluate_fit_model_components,
    initialize_fit_model,
)
from .fit_problem import (
    _RegularFitProblem,
    _build_solver_problem,
    _float64_observations,
    bind_fit,
)
from .schema import DatasetSchema, PointColumn, PointTable, ValueSchema
from .transform import DataTransformSpec, commit_transform
from .value import OwnedSnapshot


_FLOAT64 = np.dtype("<f8")
_HISTOGRAM_REPEAT_AXIS_ID = AxisId("zlc_data.bimodal-distribution.repeat")
_HISTOGRAM_BIN_AXIS_ID = AxisId("zlc_data.bimodal-distribution.bin")
_MINIMUM_SIGMA_SUM_SEPARATION = 1.5


def least_squares(*args, **kwargs):
    """Enter SciPy only when nonlinear solving is actually requested."""

    from scipy.optimize import least_squares as scipy_least_squares

    return scipy_least_squares(*args, **kwargs)


def minimize(*args, **kwargs):
    """Enter SciPy only when full-image solving is actually requested."""

    from scipy.optimize import minimize as scipy_minimize

    return scipy_minimize(*args, **kwargs)


def _scipy_version() -> str:
    import scipy

    return scipy.__version__


def _immutable_float64(values: object, shape: tuple[int, ...]) -> np.ndarray:
    return immutable_array(
        np.asarray(values, dtype=_FLOAT64),
        dtype=_FLOAT64,
        shape=shape,
    )


@dataclass(frozen=True, eq=False)
class BimodalDistributionAnalysis:
    """One immutable result for an already-binned distribution analysis."""

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


class _CellFailure(RuntimeError):
    status: FitBatchStatus

    def __init__(
        self,
        status: FitBatchStatus,
        message: str,
        *,
        evaluations: int = 0,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.evaluations = int(evaluations)


@dataclass(frozen=True)
class _ResolvedInitialization:
    seeds: tuple[np.ndarray, ...]
    lower: np.ndarray
    upper: np.ndarray
    fixed: np.ndarray
    fixed_values: np.ndarray
    free_indices: np.ndarray


@dataclass(frozen=True)
class _CellSuccess:
    parameters: np.ndarray
    covariance: np.ndarray
    covariance_valid: bool
    evaluations: int
    residual_sum_squares: float
    r_squared: float
    r_squared_valid: bool


def analyze_bimodal_distribution(
    bin_centers: object,
    bin_counts: object,
    *,
    cancel_check: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> BimodalDistributionAnalysis:
    """Fit one authoritative already-binned distribution.

    Cancellation and deadlines abort the call like :meth:`BoundFit.run`.
    Expected solve failures return a typed result; malformed inputs raise.
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
        PointTable(
            bin_axis.size,
            (
                PointColumn(
                    bin_axis.axis_id,
                    bin_axis.name,
                    bin_axis.role,
                    PointColumn.NUMERIC,
                    bin_axis.coordinates or (),
                    bin_axis.unit,
                    bin_axis.coordinate_frame,
                ),
            ),
        ),
        None,
        ValueSchema.scalar(_FLOAT64, "count"),
    )
    return bind_fit(
        FitSpec(
            committed_transform=commit_transform(schema, DataTransformSpec()),
            independent_sources=(
                AxisSourceRef.point_coordinate(bin_axis.axis_id),
            ),
            batch_sources=(),
            model_id="bimodal_gaussian",
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
    """Return the sole component-height crossing for a resolved Gaussian pair."""

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


def _fit_analysis(
    bound: BoundFit,
    snapshot: OwnedSnapshot,
    *,
    cancel_check: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> FitResultBatch:
    """Execute every physically present batch independently.

    Hosting cancellation and its absolute deadline abort the whole call.
    Deterministic evaluation limits produce typed per-batch failures, preserving
    all other independently successful grid/site fits.
    """

    if type(bound) is not BoundFit:
        raise TypeError("bound must be BoundFit")
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable or None")
    if deadline_monotonic is not None:
        deadline_monotonic = finite_real(
            deadline_monotonic,
            "deadline_monotonic",
        )

    def packing_abort() -> None:
        _check_host_abort(cancel_check, deadline_monotonic)

    _check_host_abort(cancel_check, deadline_monotonic)
    problem = _build_solver_problem(bound, snapshot, abort_check=packing_abort)
    _check_host_abort(cancel_check, deadline_monotonic)

    batch_size = problem.batch_layout.storage_size
    parameter_count = len(bound.model.parameters)
    parameter_values = np.zeros((batch_size, parameter_count), dtype=np.dtype("<f8"))
    covariance = np.zeros(
        (batch_size, parameter_count, parameter_count),
        dtype=np.dtype("<f8"),
    )
    covariance_valid = np.zeros(batch_size, dtype=bool)
    evaluation_counts = np.zeros(batch_size, dtype=np.dtype("<i8"))
    rss = np.zeros(batch_size, dtype=np.dtype("<f8"))
    r_squared = np.zeros(batch_size, dtype=np.dtype("<f8"))
    r_squared_valid = np.zeros(batch_size, dtype=bool)
    statuses: list[FitBatchStatus] = []
    errors: list[str | None] = []
    for batch_index in range(batch_size):
        _check_host_abort(cancel_check, deadline_monotonic)
        valid_count = int(problem.valid_observation_counts[batch_index])
        used_count = int(problem.used_observation_counts[batch_index])
        if valid_count == 0:
            statuses.append(FitBatchStatus.NO_VALID_DATA)
            errors.append("batch has no authoritative valid observations")
            continue
        if used_count < bound.minimum_observation_count:
            statuses.append(FitBatchStatus.INSUFFICIENT_POINTS)
            errors.append(
                f"fit request requires at least {bound.minimum_observation_count} used points; "
                f"got {used_count}"
            )
            continue
        try:
            if isinstance(problem, _RegularFitProblem):
                coordinates, observations, validity = problem.cells[batch_index]
                if len(coordinates) == 2:
                    success = _fit_regular_radial_cell(
                        bound,
                        coordinates,
                        observations,
                        validity,
                        cancel_check=cancel_check,
                        host_deadline=deadline_monotonic,
                    )
                else:
                    valid = np.asarray(validity, dtype=bool).reshape(-1)
                    selected: slice | np.ndarray = (
                        slice(None) if bool(np.all(valid)) else valid
                    )
                    success = _fit_cell(
                        bound,
                        (np.asarray(coordinates[0]).reshape(-1)[selected],),
                        _float64_observations(
                            np.asarray(observations).reshape(-1)[selected]
                        ),
                        cancel_check=cancel_check,
                        host_deadline=deadline_monotonic,
                    )
            else:
                start = int(problem.batch_offsets[batch_index])
                stop = int(problem.batch_offsets[batch_index + 1])
                success = _fit_cell(
                    bound,
                    tuple(
                        values[start:stop]
                        for values in problem.independent_values
                    ),
                    problem.observations[start:stop],
                    cancel_check=cancel_check,
                    host_deadline=deadline_monotonic,
                )
        except (FitCancelled, FitDeadlineExceeded):
            raise
        except _CellFailure as exc:
            statuses.append(exc.status)
            errors.append(_fit_error_text(str(exc)))
            evaluation_counts[batch_index] = exc.evaluations
            continue
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            statuses.append(FitBatchStatus.NUMERIC_ERROR)
            errors.append(_fit_error_text(f"numeric failure: {exc}"))
            continue

        statuses.append(FitBatchStatus.CONVERGED)
        errors.append(None)
        parameter_values[batch_index] = success.parameters
        if success.covariance_valid:
            covariance[batch_index] = success.covariance
            covariance_valid[batch_index] = True
        evaluation_counts[batch_index] = success.evaluations
        rss[batch_index] = success.residual_sum_squares
        if success.r_squared_valid:
            r_squared[batch_index] = success.r_squared
            r_squared_valid[batch_index] = True

    return FitResultBatch(
        source_ref=problem.source_ref,
        spec=problem.spec,
        fit_axis_specs=problem.fit_axis_specs,
        batch_axis_specs=problem.batch_axis_specs,
        point_groups=problem.point_groups,
        batch_layout=problem.batch_layout,
        value_unit=problem.value_unit,
        parameter_values=parameter_values,
        covariance=covariance,
        covariance_valid=covariance_valid,
        statuses=tuple(statuses),
        errors=tuple(errors),
        present_observation_counts=problem.present_observation_counts,
        valid_observation_counts=problem.valid_observation_counts,
        used_observation_counts=problem.used_observation_counts,
        evaluation_counts=evaluation_counts,
        residual_sum_squares=rss,
        r_squared=r_squared,
        r_squared_valid=r_squared_valid,
        scipy_version=_scipy_version(),
    )


def _fit_cell(
    bound: BoundFit,
    coordinates: tuple[np.ndarray, ...],
    observations: np.ndarray,
    *,
    cancel_check: Callable[[], bool] | None,
    host_deadline: float | None,
) -> _CellSuccess:
    model_coordinates = coordinates
    if not np.all(np.isfinite(observations)) or any(
        not np.all(np.isfinite(values)) for values in coordinates
    ):
        raise _CellFailure(
            FitBatchStatus.NUMERIC_ERROR,
            "authoritative valid observations or coordinates contain non-finite values",
        )

    _check_host_abort(cancel_check, host_deadline)
    try:
        user_seed = _complete_user_seed(bound)
        model_seeds = (
            initialize_fit_model(
                bound.model,
                model_coordinates,
                observations,
            )
            if user_seed is None
            else (user_seed,)
        )
        initialization = _resolve_initialization(
            bound,
            model_seeds,
        )
        _check_host_abort(cancel_check, host_deadline)
    except _CellFailure:
        raise
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise _CellFailure(
            FitBatchStatus.INITIALIZATION_FAILED,
            f"initialization failed: {exc}",
        ) from exc

    evaluation_count = 0

    def evaluated_residual(
        free: np.ndarray,
        evaluation_coordinates: tuple[np.ndarray, ...],
        evaluation_observations: np.ndarray,
    ) -> np.ndarray:
        nonlocal evaluation_count
        _check_host_abort(cancel_check, host_deadline)
        if evaluation_count >= bound.spec.numeric_policy.max_evaluations:
            raise _CellFailure(
                FitBatchStatus.EVALUATION_LIMIT,
                "per-batch model evaluation limit exceeded",
                evaluations=evaluation_count,
            )
        evaluation_count += 1
        parameters = _full_parameter_vector(initialization, free)
        with np.errstate(all="ignore"):
            predicted = evaluate_fit_model(
                bound.model,
                evaluation_coordinates,
                parameters,
            )
        values = (
            np.asarray(predicted, dtype=np.float64).reshape(-1)
            - evaluation_observations
        )
        if not np.all(np.isfinite(values)):
            raise _CellFailure(
                FitBatchStatus.NUMERIC_ERROR,
                "model evaluation produced a non-finite residual",
                evaluations=evaluation_count,
            )
        return values

    def residual(free: np.ndarray) -> np.ndarray:
        return evaluated_residual(
            free,
            model_coordinates,
            observations,
        )

    if initialization.free_indices.size == 0:
        residual_values = evaluated_residual(
            np.empty(0, dtype=np.float64),
            model_coordinates,
            observations,
        )
        return _success_metrics(
            initialization.fixed_values,
            np.zeros((len(bound.model.parameters), len(bound.model.parameters))),
            True,
            evaluation_count,
            observations,
            residual_values,
        )

    free_lower = initialization.lower[initialization.free_indices]
    free_upper = initialization.upper[initialization.free_indices]
    best = None
    best_rss = math.inf
    last_message = "least-squares did not run"
    for seed in initialization.seeds:
        _check_host_abort(cancel_check, host_deadline)
        try:
            solved = least_squares(
                residual,
                seed[initialization.free_indices],
                jac="2-point",
                bounds=(free_lower, free_upper),
                method="trf",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                x_scale=1.0,
                loss="linear",
                f_scale=1.0,
                diff_step=None,
                tr_solver="exact",
                tr_options=None,
                max_nfev=bound.spec.numeric_policy.max_evaluations,
            )
        except _CellFailure as exc:
            if best is not None and exc.status in {
                FitBatchStatus.EVALUATION_LIMIT,
                FitBatchStatus.NUMERIC_ERROR,
            }:
                break
            raise
        if not solved.success:
            last_message = str(solved.message)
            if solved.status == 0:
                if best is not None:
                    break
                raise _CellFailure(
                    FitBatchStatus.EVALUATION_LIMIT,
                    "scipy least-squares exhausted its evaluation limit",
                    evaluations=evaluation_count,
                )
            continue
        parameters = _full_parameter_vector(initialization, solved.x)
        residual_values = np.asarray(solved.fun, dtype=np.float64).reshape(-1)
        candidate_rss = _sum_squares(residual_values)
        if not np.isfinite(candidate_rss):
            continue
        if candidate_rss < best_rss:
            best = (
                parameters,
                solved.jac,
                residual_values,
                _has_authoritative_active_bound(
                    bound,
                    initialization,
                    np.asarray(solved.active_mask),
                ),
            )
            best_rss = candidate_rss

    if best is None:
        raise _CellFailure(
            FitBatchStatus.SOLVER_FAILED,
            f"least-squares failed for every initializer: {last_message}",
            evaluations=evaluation_count,
        )
    parameters, jacobian, residual_values, active_bound = best
    covariance, covariance_is_valid = _covariance(
        np.asarray(jacobian, dtype=np.float64),
        residual_values,
        initialization.free_indices,
        len(bound.model.parameters),
        bound.spec.numeric_policy.covariance_rcond,
    )
    if active_bound:
        covariance = np.zeros_like(covariance)
        covariance_is_valid = False
    return _success_metrics(
        parameters,
        covariance,
        covariance_is_valid,
        evaluation_count,
        observations,
        residual_values,
    )


def _fit_regular_radial_cell(
    bound: BoundFit,
    coordinates: tuple[np.ndarray, ...],
    observations: np.ndarray,
    validity: np.ndarray,
    *,
    cancel_check: Callable[[], bool] | None,
    host_deadline: float | None,
) -> _CellSuccess:
    """Fit the exact raster objective, with a bounded masked-data fallback."""

    if bound.model.model_id != "radial_gaussian_center" or len(coordinates) != 2:
        raise RuntimeError("regular two-axis Fit requires the radial image model")
    x_values, y_values = coordinates
    image = np.asarray(observations)
    valid = np.broadcast_to(np.asarray(validity, dtype=bool), image.shape)
    if image.shape != (x_values.size, y_values.size):
        raise RuntimeError("regular Fit image shape disagrees with its axes")
    _check_host_abort(cancel_check, host_deadline)
    all_valid = bool(np.all(valid))
    observation_scale = 0.0
    observation_min = math.inf
    observation_max = -math.inf
    for start in range(0, image.shape[0], 64):
        _check_host_abort(cancel_check, host_deadline)
        source = np.asarray(image[start : start + 64]).reshape(-1)
        if all_valid:
            values = _float64_observations(source)
        else:
            mask = np.asarray(valid[start : start + 64], dtype=bool).reshape(-1)
            values = _float64_observations(source[mask])
        if values.size:
            if not np.all(np.isfinite(values)):
                raise _CellFailure(
                    FitBatchStatus.NUMERIC_ERROR,
                    "authoritative valid observations contain non-finite values",
                )
            observation_scale = max(observation_scale, float(np.max(np.abs(values))))
            observation_min = min(observation_min, float(np.min(values)))
            observation_max = max(observation_max, float(np.max(values)))
    observation_scale = observation_scale or 1.0
    if observation_min >= 0.0 or observation_max <= 0.0:
        optimizer_scale = observation_max - observation_min
    else:
        # A cross-zero range can overflow even when max-absolute scaling is safe.
        optimizer_scale = observation_scale
    optimizer_scale = optimizer_scale or observation_scale
    optimizer_ratio = observation_scale / optimizer_scale
    optimizer_factor = optimizer_ratio * optimizer_ratio
    if not np.isfinite(optimizer_factor):
        raise _CellFailure(
            FitBatchStatus.NUMERIC_ERROR,
            "full-image objective scale is not finite",
        )
    observation_sum = observation_square_sum = observation_total = 0.0
    observation_count = 0
    if all_valid:
        sums: list[float] = []
        squares: list[float] = []
        mean = 0.0
        for start in range(0, image.shape[0], 64):
            _check_host_abort(cancel_check, host_deadline)
            values = np.asarray(image[start : start + 64]).reshape(-1).astype(
                np.float64, copy=False
            ) / observation_scale
            sums.append(float(np.sum(values)))
            squares.append(float(np.dot(values, values)))
            local_count = values.size
            local_mean = float(np.mean(values))
            centered = values - local_mean
            delta = local_mean - mean
            combined = observation_count + local_count
            observation_total += float(np.dot(centered, centered)) + (
                delta * delta * observation_count * local_count / combined
            )
            mean += delta * local_count / combined
            observation_count = combined
        observation_sum = math.fsum(sums)
        observation_square_sum = math.fsum(squares)

    user_seed = _complete_user_seed(bound)
    try:
        seeds = (
            (user_seed,)
            if user_seed is not None
            else _spatial_radial_center_seeds(
                x_values,
                y_values,
                image,
                valid,
                lambda: _check_host_abort(cancel_check, host_deadline),
            )
        )
        if not seeds:
            raise ValueError("radial center fit requires coherent contrast")
        initialization = _resolve_initialization(bound, seeds)
    except _CellFailure:
        raise
    except (ValueError, FloatingPointError, OverflowError) as exc:
        raise _CellFailure(
            FitBatchStatus.INITIALIZATION_FAILED,
            f"initialization failed: {exc}",
        ) from exc

    free_indices = initialization.free_indices
    evaluation_count = 0

    def full_pass(
        parameters: np.ndarray,
        *,
        collect_metrics: bool,
        collect_information: bool,
    ) -> tuple[float, np.ndarray, np.ndarray, float, int]:
        _check_host_abort(cancel_check, host_deadline)
        rss = 0.0
        gradient = np.zeros(free_indices.size, dtype=np.float64)
        information = np.zeros(
            (free_indices.size, free_indices.size), dtype=np.float64
        )
        if all_valid:
            with np.errstate(all="ignore"):
                radial_x, radial_y, radius_x, radius_y, center_x, center_y = (
                    _radial_gaussian_center_axis_basis(x_values, y_values, parameters)
                )
            amplitude, offset = float(parameters[0]), float(parameters[1])
            inverse_scale = 1.0 / observation_scale
            # Basis order is radial, constant, radius derivative, center derivative.
            row_basis = np.stack(
                (radial_x, np.ones_like(radial_x), radius_x, center_x)
            )
            column_basis = np.stack(
                (radial_y, np.ones_like(radial_y), radius_y, center_y)
            )
            coefficients = np.zeros((6, 4, 4), dtype=np.float64)
            coefficients[0, 0, 0] = amplitude * inverse_scale
            coefficients[0, 1, 1] = offset * inverse_scale
            coefficients[1, 0, 0] = inverse_scale
            coefficients[2, 1, 1] = inverse_scale
            coefficients[3, 2, 0] = coefficients[3, 0, 2] = (
                amplitude * inverse_scale
            )
            coefficients[4, 3, 0] = amplitude * inverse_scale
            coefficients[5, 0, 3] = amplitude * inverse_scale
            data_projection = np.einsum(
                "ij,bj->bi",
                image,
                column_basis,
                optimize=False,
                dtype=np.float64,
                casting="unsafe",
            ) / observation_scale
            data_matrix = row_basis @ data_projection.T
            data_matrix[1, 1] = observation_sum
            data_inner = np.einsum(
                "kab,ab->k", coefficients, data_matrix, optimize=False
            )
            function_inner = np.einsum(
                "kab,lcd,ac,bd->kl",
                coefficients,
                coefficients,
                row_basis @ row_basis.T,
                column_basis @ column_basis.T,
                optimize=False,
            )
            model_square = float(function_inner[0, 0])
            model_data = float(data_inner[0])
            rss = model_square - 2.0 * model_data + observation_square_sum
            cancellation = 64.0 * np.finfo(np.float64).eps * max(
                model_square,
                2.0 * abs(model_data),
                observation_square_sum,
                1.0,
            )
            gradient = (function_inner[0, 1:] - data_inner[1:])[free_indices]
            if collect_information:
                information = function_inner[1:, 1:][
                    np.ix_(free_indices, free_indices)
                ]
            if not np.isfinite(rss) or not np.all(
                np.isfinite(gradient)
            ) or not np.all(np.isfinite(information)):
                raise _CellFailure(
                    FitBatchStatus.NUMERIC_ERROR,
                    "separable full-image objective produced non-finite values",
                    evaluations=evaluation_count,
                )
            if abs(rss) > cancellation:
                if rss < 0.0:
                    raise _CellFailure(
                        FitBatchStatus.NUMERIC_ERROR,
                        "separable full-image objective lost numeric precision",
                        evaluations=evaluation_count,
                    )
                return (
                    rss,
                    gradient,
                    information,
                    observation_total if collect_metrics else 0.0,
                    observation_count if collect_metrics else 0,
                )
            # The separable expansion subtracts three nearly equal inner
            # products near a good fit.  Once their rounding envelope can hide
            # the residual, use the same bounded exact pass as masked rasters.
            rss = 0.0
            gradient.fill(0.0)
            information.fill(0.0)
        mean = total = 0.0
        count = 0
        need_jacobian = bool(free_indices.size) and (
            not collect_metrics or collect_information
        )
        for start in range(0, x_values.size, 64):
            _check_host_abort(cancel_check, host_deadline)
            stop = min(x_values.size, start + 64)
            mask = np.asarray(valid[start:stop], dtype=bool).reshape(-1)
            if not bool(np.any(mask)):
                continue
            coordinates = np.broadcast_arrays(
                x_values[start:stop, None],
                y_values[None, :],
            )
            with np.errstate(all="ignore"):
                predicted = evaluate_fit_model(bound.model, coordinates, parameters)
                jacobian = (
                    _radial_gaussian_center_jacobian(coordinates, parameters)[
                        :, free_indices
                    ]
                    if need_jacobian
                    else None
                )
            observed = np.asarray(image[start:stop]).reshape(-1)[mask].astype(
                np.float64, copy=False
            )
            residual = (predicted.reshape(-1)[mask] - observed) / observation_scale
            if jacobian is not None:
                jacobian = jacobian[mask] / observation_scale
            if not np.all(np.isfinite(residual)) or (
                jacobian is not None and not np.all(np.isfinite(jacobian))
            ):
                raise _CellFailure(
                    FitBatchStatus.NUMERIC_ERROR,
                    "full-image model evaluation produced non-finite values",
                    evaluations=evaluation_count,
                )
            rss += float(np.dot(residual, residual))
            if jacobian is not None:
                gradient += jacobian.T @ residual
                if collect_information:
                    information += jacobian.T @ jacobian
            if collect_metrics:
                observed = observed / observation_scale
                local_count = observed.size
                local_mean = float(np.mean(observed))
                centered = observed - local_mean
                delta = local_mean - mean
                combined = count + local_count
                total += float(np.dot(centered, centered)) + (
                    delta * delta * count * local_count / combined
                )
                mean += delta * local_count / combined
                count = combined
        if not np.isfinite(rss) or not np.all(np.isfinite(gradient)):
            raise _CellFailure(
                FitBatchStatus.NUMERIC_ERROR,
                "full-image objective overflowed",
                evaluations=evaluation_count,
            )
        return rss, gradient, information, total, count

    best_parameters: np.ndarray | None = None
    covariance_allowed = True
    if free_indices.size == 0:
        best_parameters = initialization.fixed_values
        evaluation_count = 1
    else:
        free_lower = initialization.lower[free_indices]
        free_upper = initialization.upper[free_indices]
        best_objective = math.inf
        last_message = "full-image optimizer did not run"
        for seed in initialization.seeds:
            _check_host_abort(cancel_check, host_deadline)
            remaining = bound.spec.numeric_policy.max_evaluations - evaluation_count
            if remaining <= 0:
                break

            def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
                nonlocal evaluation_count
                _check_host_abort(cancel_check, host_deadline)
                if evaluation_count >= bound.spec.numeric_policy.max_evaluations:
                    raise _CellFailure(
                        FitBatchStatus.EVALUATION_LIMIT,
                        "per-batch model evaluation limit exceeded",
                        evaluations=evaluation_count,
                    )
                evaluation_count += 1
                rss, gradient, _information, _total, _count = full_pass(
                    _full_parameter_vector(initialization, free),
                    collect_metrics=False,
                    collect_information=False,
                )
                return 0.5 * rss * optimizer_factor, gradient * optimizer_factor

            try:
                solved = minimize(
                    objective,
                    seed[free_indices],
                    method="L-BFGS-B",
                    jac=True,
                    bounds=tuple(zip(free_lower, free_upper)),
                    options={
                        "ftol": 1e-8,
                        "gtol": 1e-8,
                        "maxfun": remaining,
                        "maxiter": remaining,
                        "maxls": 50,
                    },
                )
            except _CellFailure as exc:
                if best_parameters is not None and exc.status in {
                    FitBatchStatus.EVALUATION_LIMIT,
                    FitBatchStatus.NUMERIC_ERROR,
                }:
                    break
                raise
            last_message = str(solved.message)
            if not solved.success:
                continue
            candidate = float(solved.fun)
            if not np.isfinite(candidate) or candidate >= best_objective:
                continue
            best_objective = candidate
            best_parameters = _full_parameter_vector(initialization, solved.x)
            tolerance = np.sqrt(np.finfo(np.float64).eps) * np.maximum(
                1.0, np.abs(solved.x)
            )
            active = np.zeros(solved.x.shape, dtype=np.int8)
            active[
                np.isfinite(free_lower) & (solved.x - free_lower <= tolerance)
            ] = -1
            active[
                np.isfinite(free_upper) & (free_upper - solved.x <= tolerance)
            ] = 1
            covariance_allowed = not _has_authoritative_active_bound(
                bound, initialization, active
            )

        if best_parameters is None:
            status = (
                FitBatchStatus.EVALUATION_LIMIT
                if evaluation_count >= bound.spec.numeric_policy.max_evaluations
                else FitBatchStatus.SOLVER_FAILED
            )
            message = (
                "full-image optimizer exhausted its evaluation limit"
                if status is FitBatchStatus.EVALUATION_LIMIT
                else f"full-image optimizer failed for every initializer: {last_message}"
            )
            raise _CellFailure(status, message, evaluations=evaluation_count)

    assert best_parameters is not None
    normalized_rss, _gradient, information, total, observation_count = full_pass(
        best_parameters,
        collect_metrics=True,
        collect_information=covariance_allowed,
    )
    rss = _restore_scaled_square_sum(normalized_rss, observation_scale)
    if not np.isfinite(rss):
        raise _CellFailure(
            FitBatchStatus.NUMERIC_ERROR,
            "residual sum of squares overflowed",
            evaluations=evaluation_count,
        )
    total = max(0.0, float(total))
    r_squared_valid = total > 0.0 and np.isfinite(normalized_rss)
    r_squared = float(1.0 - normalized_rss / total) if r_squared_valid else 0.0
    if r_squared_valid and -64.0 * np.finfo(np.float64).eps <= r_squared < 0.0:
        r_squared = 0.0
    r_squared_valid = r_squared_valid and np.isfinite(r_squared)

    covariance = np.zeros(
        (len(bound.model.parameters), len(bound.model.parameters)),
        dtype=np.float64,
    )
    covariance_valid = False
    if covariance_allowed:
        covariance, covariance_valid = _covariance_from_information(
            information,
            observation_count,
            normalized_rss,
            free_indices,
            len(bound.model.parameters),
            bound.spec.numeric_policy.covariance_rcond,
        )
    return _CellSuccess(
        best_parameters,
        covariance,
        covariance_valid,
        evaluation_count,
        rss,
        r_squared if r_squared_valid else 0.0,
        r_squared_valid,
    )


def _full_parameter_vector(
    initialization: _ResolvedInitialization,
    free: np.ndarray,
) -> np.ndarray:
    result = initialization.fixed_values.copy()
    result[initialization.free_indices] = np.asarray(free, dtype=np.float64)
    return result


def _restore_scaled_square_sum(normalized_sum: float, scale: float) -> float:
    if not np.isfinite(normalized_sum) or normalized_sum < 0.0:
        return math.inf
    if normalized_sum == 0.0 or scale == 0.0:
        return 0.0
    if scale > math.sqrt(np.finfo(np.float64).max) / math.sqrt(normalized_sum):
        return math.inf
    return float(scale * scale * normalized_sum)


def _complete_user_seed(bound: BoundFit) -> tuple[float, ...] | None:
    """Bypass data heuristics when every parameter already has a caller seed."""

    constraints = {
        constraint.parameter_name: constraint for constraint in bound.spec.constraints
    }
    values: list[float] = []
    for parameter in bound.model.parameters:
        constraint = constraints.get(parameter.name)
        if constraint is None:
            return None
        if constraint.fixed is not None:
            values.append(constraint.fixed)
        elif constraint.initial is not None:
            values.append(constraint.initial)
        else:
            return None
    return tuple(values)


def _has_authoritative_active_bound(
    bound: BoundFit,
    initialization: _ResolvedInitialization,
    active_mask: np.ndarray,
) -> bool:
    mask = np.asarray(active_mask, dtype=np.int8).reshape(-1)
    if mask.shape != initialization.free_indices.shape:
        return True
    constraints = {
        constraint.parameter_name: constraint for constraint in bound.spec.constraints
    }
    for free_position in np.flatnonzero(mask):
        parameter_index = int(initialization.free_indices[free_position])
        parameter = bound.model.parameters[parameter_index]
        constraint = constraints.get(parameter.name)
        if (
            parameter.domain is FitParameterDomain.PHASE_RADIANS
            and (constraint is None or (constraint.lower is None and constraint.upper is None))
        ):
            continue
        return True
    return False


def _resolve_initialization(
    bound: BoundFit,
    model_seeds: tuple[tuple[float, ...], ...],
) -> _ResolvedInitialization:
    parameter_count = len(bound.model.parameters)
    lower = np.asarray(
        [parameter.solver_bounds[0] for parameter in bound.model.parameters],
        dtype=np.float64,
    )
    upper = np.asarray(
        [parameter.solver_bounds[1] for parameter in bound.model.parameters],
        dtype=np.float64,
    )
    seeds = [np.asarray(seed, dtype=np.float64).copy() for seed in model_seeds]
    if lower.shape != (parameter_count,) or upper.shape != (parameter_count,) or any(
        seed.shape != (parameter_count,) for seed in seeds
    ):
        raise _CellFailure(
            FitBatchStatus.INITIALIZATION_FAILED,
            "model initializer dimensions disagree with the model catalog",
        )

    constraints: dict[str, FitParameterConstraint] = {
        constraint.parameter_name: constraint for constraint in bound.spec.constraints
    }
    fixed = np.zeros(parameter_count, dtype=bool)
    fixed_values = np.zeros(parameter_count, dtype=np.float64)
    explicit_initial = np.zeros(parameter_count, dtype=bool)
    for index, parameter in enumerate(bound.model.parameters):
        constraint = constraints.get(parameter.name)
        if constraint is None:
            continue
        if constraint.lower is not None:
            lower[index] = max(lower[index], constraint.lower)
        if constraint.upper is not None:
            upper[index] = min(upper[index], constraint.upper)
        if not lower[index] < upper[index] and constraint.fixed is None:
            raise _CellFailure(
                FitBatchStatus.INITIALIZATION_FAILED,
                f"constraint intersection is empty for parameter {parameter.name!r}",
            )
        if constraint.initial is not None:
            explicit_initial[index] = True
            for seed in seeds:
                seed[index] = constraint.initial
        if constraint.fixed is not None:
            fixed[index] = True
            fixed_values[index] = constraint.fixed
            for seed in seeds:
                seed[index] = constraint.fixed

    for index, parameter in enumerate(bound.model.parameters):
        if fixed[index]:
            if not lower[index] <= fixed_values[index] <= upper[index]:
                raise _CellFailure(
                    FitBatchStatus.INITIALIZATION_FAILED,
                    f"fixed value is outside model bounds for parameter {parameter.name!r}",
                )
            continue
        for seed in seeds:
            if explicit_initial[index] and not lower[index] <= seed[index] <= upper[index]:
                raise _CellFailure(
                    FitBatchStatus.INITIALIZATION_FAILED,
                    f"explicit initial value is outside model bounds for parameter {parameter.name!r}",
                )
            seed[index] = _inside_bounds(seed[index], lower[index], upper[index])

    free_indices = np.flatnonzero(np.logical_not(fixed)).astype(np.int64, copy=False)
    # Fixed values occupy their exact slots; free slots are overwritten on every evaluation.
    fixed_values[free_indices] = 0.0
    return _ResolvedInitialization(
        tuple(seeds),
        lower,
        upper,
        fixed,
        fixed_values,
        free_indices,
    )


def _inside_bounds(value: float, lower: float, upper: float) -> float:
    if not np.isfinite(value):
        raise _CellFailure(
            FitBatchStatus.INITIALIZATION_FAILED,
            "model initializer produced a non-finite seed",
        )
    low_inside = lower if np.isneginf(lower) else np.nextafter(lower, upper)
    high_inside = upper if np.isposinf(upper) else np.nextafter(upper, lower)
    return float(np.clip(value, low_inside, high_inside))


def _covariance(
    jacobian: np.ndarray,
    residual: np.ndarray,
    free_indices: np.ndarray,
    parameter_count: int,
    rcond: float,
) -> tuple[np.ndarray, bool]:
    result = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    free_count = int(free_indices.size)
    if free_count == 0:
        return result, True
    if jacobian.shape != (residual.size, free_count):
        return result, False
    if not np.all(np.isfinite(jacobian)):
        return result, False
    column_norms = np.linalg.norm(jacobian, axis=0)
    if np.any(~np.isfinite(column_norms)) or np.any(
        column_norms <= np.finfo(np.float64).tiny
    ):
        return result, False
    normalized = jacobian / column_norms
    return _covariance_from_information(
        normalized.T @ normalized,
        residual.size,
        _sum_squares(residual),
        free_indices,
        parameter_count,
        rcond,
        column_norms=column_norms,
    )


def _covariance_from_information(
    information: np.ndarray,
    observation_count: int,
    rss: float,
    free_indices: np.ndarray,
    parameter_count: int,
    rcond: float,
    *,
    column_norms: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    result = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    free_count = int(free_indices.size)
    if free_count == 0:
        return result, True
    if (
        information.shape != (free_count, free_count)
        or observation_count <= free_count
        or not np.all(np.isfinite(information))
        or not np.isfinite(rss)
    ):
        return result, False
    if column_norms is None:
        column_norms = np.sqrt(np.diag(information))
        normalized_information = None
    else:
        column_norms = np.asarray(column_norms, dtype=np.float64)
        normalized_information = information
    if column_norms.shape != (free_count,) or np.any(
        ~np.isfinite(column_norms)
    ) or np.any(column_norms <= np.finfo(np.float64).tiny):
        return result, False
    if normalized_information is None:
        normalized_information = information / np.outer(column_norms, column_norms)
    rank = int(
        np.linalg.matrix_rank(
            normalized_information,
            tol=rcond**2 * np.linalg.norm(normalized_information, ord=2),
        )
    )
    if rank != free_count:
        return result, False
    variance = rss / (observation_count - free_count)
    if not np.isfinite(variance):
        return result, False
    try:
        normalized_covariance = (
            np.linalg.pinv(normalized_information, rcond=rcond) * variance
        )
    except np.linalg.LinAlgError:
        return result, False
    free_covariance = normalized_covariance / np.outer(column_norms, column_norms)
    free_covariance = (free_covariance + free_covariance.T) / 2.0
    if not np.all(np.isfinite(free_covariance)):
        return result, False
    result[np.ix_(free_indices, free_indices)] = free_covariance
    # Canonical serialization requires exact symmetry, not tolerance-based symmetry.
    result = (result + result.T) / 2.0
    return result, True


def _success_metrics(
    parameters: np.ndarray,
    covariance: np.ndarray,
    covariance_valid: bool,
    evaluations: int,
    observations: np.ndarray,
    residual: np.ndarray,
) -> _CellSuccess:
    if not np.all(np.isfinite(parameters)):
        raise _CellFailure(FitBatchStatus.NUMERIC_ERROR, "solver returned non-finite parameters")
    rss = _sum_squares(residual)
    if not np.isfinite(rss):
        raise _CellFailure(FitBatchStatus.NUMERIC_ERROR, "residual sum of squares overflowed")
    r_squared = 0.0
    r_squared_is_valid = False
    observation_scale = float(np.max(np.abs(observations)))
    if observation_scale > 0.0 and np.isfinite(observation_scale):
        normalized_observations = observations / observation_scale
        centered = normalized_observations - float(np.mean(normalized_observations))
        normalized_total = _sum_squares(centered)
        normalized_residual = _sum_squares(residual / observation_scale)
        if normalized_total > 0.0 and np.isfinite(normalized_residual):
            r_squared = float(1.0 - normalized_residual / normalized_total)
            r_squared_is_valid = bool(np.isfinite(r_squared))
            if r_squared_is_valid and -64.0 * np.finfo(np.float64).eps <= r_squared < 0.0:
                r_squared = 0.0
    if not covariance_valid:
        covariance = np.zeros_like(covariance)
    return _CellSuccess(
        np.asarray(parameters, dtype=np.float64),
        np.asarray(covariance, dtype=np.float64),
        bool(covariance_valid),
        int(evaluations),
        rss,
        r_squared if r_squared_is_valid else 0.0,
        r_squared_is_valid,
    )


def _sum_squares(values: np.ndarray) -> float:
    """Return a finite sum of squares or ``inf`` without overflowing en route."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale):
        return math.inf
    if scale == 0.0:
        return 0.0
    normalized = values / scale
    return _restore_scaled_square_sum(float(np.dot(normalized, normalized)), scale)


def _check_host_abort(
    cancel_check: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> None:
    if cancel_check is not None and cancel_check():
        raise FitCancelled("fit was cancelled")
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit hosting deadline exceeded")


def _fit_error_text(message: str) -> str:
    compact = " ".join(str(message).split()) or "unspecified fit failure"
    return compact
