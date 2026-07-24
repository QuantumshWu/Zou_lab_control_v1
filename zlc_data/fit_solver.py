"""Bound nonlinear least-squares execution with per-batch isolation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np
import scipy
from scipy.optimize import least_squares
from zlc_storage.canonical import finite_real

from .fit_contract import (
    BoundFit,
    FitBatchStatus,
    FitCancelled,
    FitDeadlineExceeded,
    FitParameterConstraint,
    FitResultBatch,
)
from .fit_model import FitParameterDomain, evaluate_fit_model, initialize_fit_model
from .fit_problem import build_fit_problem
from .value import OwnedSnapshot


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
    problem = build_fit_problem(bound, snapshot, abort_check=packing_abort)
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
            start = int(problem.batch_offsets[batch_index])
            stop = int(problem.batch_offsets[batch_index + 1])
            success = _fit_cell(
                bound,
                tuple(values[start:stop] for values in problem.independent_values),
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
        scipy_version=scipy.__version__,
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
        if user_seed is None:
            model_seeds = initialize_fit_model(
                bound.model,
                model_coordinates,
                observations,
            )
        else:
            model_seeds = (user_seed,)
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

    def full_parameters(free: np.ndarray) -> np.ndarray:
        result = initialization.fixed_values.copy()
        result[initialization.free_indices] = free
        return result

    def residual(free: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count
        _check_host_abort(cancel_check, host_deadline)
        if evaluation_count >= bound.spec.numeric_policy.max_evaluations:
            raise _CellFailure(
                FitBatchStatus.EVALUATION_LIMIT,
                "per-batch model evaluation limit exceeded",
                evaluations=evaluation_count,
            )
        evaluation_count += 1
        parameters = full_parameters(np.asarray(free, dtype=np.float64))
        with np.errstate(all="ignore"):
            predicted = evaluate_fit_model(bound.model, model_coordinates, parameters)
        values = np.asarray(predicted, dtype=np.float64).reshape(-1) - observations
        if not np.all(np.isfinite(values)):
            raise _CellFailure(
                FitBatchStatus.NUMERIC_ERROR,
                "model evaluation produced a non-finite residual",
                evaluations=evaluation_count,
            )
        return values

    if initialization.free_indices.size == 0:
        residual_values = residual(np.empty(0, dtype=np.float64))
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
        parameters = full_parameters(solved.x)
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
    if jacobian.shape != (residual.size, free_count) or residual.size <= free_count:
        return result, False
    if not np.all(np.isfinite(jacobian)):
        return result, False
    column_norms = np.linalg.norm(jacobian, axis=0)
    if np.any(~np.isfinite(column_norms)) or np.any(column_norms <= np.finfo(np.float64).tiny):
        return result, False
    normalized = jacobian / column_norms
    rank = int(
        np.linalg.matrix_rank(
            normalized,
            tol=rcond * np.linalg.norm(normalized, ord=2),
        )
    )
    if rank != free_count:
        return result, False
    information = normalized.T @ normalized
    variance = _sum_squares(residual) / (residual.size - free_count)
    if not np.isfinite(variance):
        return result, False
    try:
        normalized_covariance = np.linalg.pinv(information, rcond=rcond) * variance
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
    normalized_sum = float(np.dot(normalized, normalized))
    if not np.isfinite(normalized_sum) or normalized_sum <= 0.0:
        return math.inf
    if scale > math.sqrt(np.finfo(np.float64).max / normalized_sum):
        return math.inf
    return float(scale * scale * normalized_sum)


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
