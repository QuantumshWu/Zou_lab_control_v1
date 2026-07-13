"""Bound nonlinear least-squares execution with per-batch isolation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import time
from typing import Callable

import numpy as np
import scipy
from scipy.optimize import least_squares

from .fit_contract import (
    BoundFit,
    FitBatchStatus,
    FitCancelled,
    FitCellProblem,
    FitDeadlineExceeded,
    FitParameterConstraint,
    FitResultBatch,
    resolve_parameter_units,
)
from .fit_model import evaluate_fit_model, initialize_fit_model
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
    rmse: float
    r_squared: float
    r_squared_valid: bool


def fit_analysis(
    bound: BoundFit,
    snapshot: OwnedSnapshot,
    *,
    cancel_check: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> FitResultBatch:
    """Execute every physically present batch independently.

    Hosting cancellation and its absolute deadline abort the whole call.  Numeric
    budgets owned by ``FitSpec`` produce typed per-batch failures, preserving all
    other independently successful grid/site fits.
    """

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable or None")
    if deadline_monotonic is not None:
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, Real)
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise ValueError("deadline_monotonic must be a finite absolute deadline")
        deadline_monotonic = float(deadline_monotonic)

    analysis_start = time.monotonic()
    analysis_deadline = analysis_start + bound.spec.numeric_policy.max_total_seconds

    def packing_abort() -> None:
        _check_host_abort(cancel_check, deadline_monotonic)
        if time.monotonic() >= analysis_deadline:
            raise FitDeadlineExceeded("fit numeric total deadline exceeded during packing")

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
    rmse = np.zeros(batch_size, dtype=np.dtype("<f8"))
    r_squared = np.zeros(batch_size, dtype=np.dtype("<f8"))
    r_squared_valid = np.zeros(batch_size, dtype=bool)
    statuses: list[FitBatchStatus] = []
    errors: list[str | None] = []

    for batch_index in range(batch_size):
        _check_host_abort(cancel_check, deadline_monotonic)
        cell = problem.cell(batch_index)
        if cell.valid_observation_count == 0:
            statuses.append(FitBatchStatus.NO_VALID_DATA)
            errors.append("batch has no authoritative valid observations")
            continue
        if cell.used_observation_count < bound.model.minimum_observations:
            statuses.append(FitBatchStatus.INSUFFICIENT_POINTS)
            errors.append(
                f"model requires at least {bound.model.minimum_observations} used points; "
                f"got {cell.used_observation_count}"
            )
            continue
        if time.monotonic() >= analysis_deadline:
            statuses.append(FitBatchStatus.TIMEOUT)
            errors.append("analysis numeric time budget exceeded")
            continue
        try:
            success = _fit_cell(
                bound,
                cell,
                cancel_check=cancel_check,
                host_deadline=deadline_monotonic,
                analysis_deadline=analysis_deadline,
            )
        except (FitCancelled, FitDeadlineExceeded):
            raise
        except _CellFailure as exc:
            statuses.append(exc.status)
            errors.append(_bounded_error(str(exc)))
            evaluation_counts[batch_index] = exc.evaluations
            continue
        except (FloatingPointError, OverflowError) as exc:
            statuses.append(FitBatchStatus.NUMERIC_ERROR)
            errors.append(_bounded_error(f"numeric failure: {exc}"))
            continue
        except Exception as exc:  # solver isolation: one grid cell cannot poison siblings
            statuses.append(FitBatchStatus.SOLVER_FAILED)
            errors.append(_bounded_error(f"solver failure: {type(exc).__name__}: {exc}"))
            continue

        statuses.append(FitBatchStatus.SUCCESS)
        errors.append(None)
        parameter_values[batch_index] = success.parameters
        if success.covariance_valid:
            covariance[batch_index] = success.covariance
            covariance_valid[batch_index] = True
        evaluation_counts[batch_index] = success.evaluations
        rss[batch_index] = success.residual_sum_squares
        rmse[batch_index] = success.rmse
        if success.r_squared_valid:
            r_squared[batch_index] = success.r_squared
            r_squared_valid[batch_index] = True

    return FitResultBatch(
        source_ref=problem.source_ref,
        spec=problem.spec,
        effective_schema_fingerprint=problem.effective_schema_fingerprint,
        fit_axis_specs=problem.fit_axis_specs,
        coordinate_sources=problem.coordinate_sources,
        batch_axis_specs=problem.batch_axis_specs,
        batch_layout=problem.batch_layout,
        value_unit=problem.value_unit,
        parameter_definitions=bound.model.parameters,
        parameter_units=resolve_parameter_units(
            bound.model.parameters,
            problem.fit_axis_specs,
            problem.coordinate_sources,
            problem.value_unit,
        ),
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
        rmse=rmse,
        r_squared=r_squared,
        r_squared_valid=r_squared_valid,
        solver_contract_id=bound.model.solver_contract_id,
        scipy_version=scipy.__version__,
        initializer_id=bound.model.initializer_id,
    )


def _fit_cell(
    bound: BoundFit,
    cell: FitCellProblem,
    *,
    cancel_check: Callable[[], bool] | None,
    host_deadline: float | None,
    analysis_deadline: float,
) -> _CellSuccess:
    coordinates = cell.independent_values
    model_coordinates = _model_coordinates(bound, coordinates)
    observations = cell.observations
    if not np.all(np.isfinite(observations)) or any(
        not np.all(np.isfinite(values)) for values in coordinates
    ):
        raise _CellFailure(
            FitBatchStatus.NUMERIC_ERROR,
            "authoritative valid observations or coordinates contain non-finite values",
        )
    if len(model_coordinates) > 1:
        geometry = np.column_stack(model_coordinates)
        geometry -= np.mean(geometry, axis=0, keepdims=True)
        spans = np.ptp(geometry, axis=0)
        if np.any(spans == 0):
            raise _CellFailure(
                FitBatchStatus.INITIALIZATION_FAILED,
                "fit coordinates do not span every independent axis",
            )
        geometry /= spans
        if np.linalg.matrix_rank(geometry) < len(model_coordinates):
            raise _CellFailure(
                FitBatchStatus.INITIALIZATION_FAILED,
                "fit coordinate geometry is rank deficient",
            )

    cell_start = time.monotonic()
    cell_deadline = min(
        analysis_deadline,
        cell_start + bound.spec.numeric_policy.max_seconds_per_batch,
    )
    _check_host_abort(cancel_check, host_deadline)
    try:
        model_initialization = initialize_fit_model(
            bound.model,
            model_coordinates,
            observations,
        )
        initialization = _resolve_initialization(
            bound,
            model_initialization.seeds,
            model_initialization.lower_bounds,
            model_initialization.upper_bounds,
        )
        _check_host_abort(cancel_check, host_deadline)
        if time.monotonic() >= cell_deadline:
            raise _CellFailure(
                FitBatchStatus.TIMEOUT,
                "per-batch numeric time budget exceeded during initialization",
            )
    except _CellFailure:
        raise
    except Exception as exc:
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
        if time.monotonic() >= cell_deadline:
            raise _CellFailure(
                FitBatchStatus.TIMEOUT,
                "per-batch numeric time budget exceeded",
                evaluations=evaluation_count,
            )
        if evaluation_count >= bound.spec.numeric_policy.max_evaluations:
            raise _CellFailure(
                FitBatchStatus.EVALUATION_LIMIT,
                "per-batch model evaluation budget exceeded",
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
                bounds=(free_lower, free_upper),
                method="trf",
                max_nfev=bound.spec.numeric_policy.max_evaluations,
            )
        except _CellFailure as exc:
            if best is not None and exc.status in {
                FitBatchStatus.EVALUATION_LIMIT,
                FitBatchStatus.TIMEOUT,
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
        candidate_rss = float(np.dot(residual_values, residual_values))
        if not np.isfinite(candidate_rss):
            continue
        if candidate_rss < best_rss:
            best = (parameters, solved.jac, residual_values)
            best_rss = candidate_rss

    if best is None:
        raise _CellFailure(
            FitBatchStatus.SOLVER_FAILED,
            f"least-squares failed for every initializer: {last_message}",
            evaluations=evaluation_count,
        )
    parameters, jacobian, residual_values = best
    covariance, covariance_is_valid = _covariance(
        np.asarray(jacobian, dtype=np.float64),
        residual_values,
        initialization.free_indices,
        len(bound.model.parameters),
        bound.spec.numeric_policy.covariance_rcond,
    )
    return _success_metrics(
        parameters,
        covariance,
        covariance_is_valid,
        evaluation_count,
        observations,
        residual_values,
    )


def _resolve_initialization(
    bound: BoundFit,
    model_seeds: tuple[tuple[float, ...], ...],
    model_lower: tuple[float, ...],
    model_upper: tuple[float, ...],
) -> _ResolvedInitialization:
    parameter_count = len(bound.model.parameters)
    lower = np.asarray(model_lower, dtype=np.float64).copy()
    upper = np.asarray(model_upper, dtype=np.float64).copy()
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


def _model_coordinates(
    bound: BoundFit,
    coordinates: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Apply model-owned coordinate reference semantics without changing lineage."""

    if bound.model.model_id not in {"damped_sine", "exponential_decay"}:
        return coordinates
    axis = bound.effective_schema.axis(bound.spec.fit_axis_ids[0])
    reference = (
        0.0
        if axis.coordinates is None
        else min(float(value) for value in axis.coordinates)
    )
    return (coordinates[0] - reference,)


def _inside_bounds(value: float, lower: float, upper: float) -> float:
    if not np.isfinite(value):
        raise _CellFailure(
            FitBatchStatus.INITIALIZATION_FAILED,
            "model initializer produced a non-finite seed",
        )
    low_inside = np.nextafter(lower, upper)
    high_inside = np.nextafter(upper, lower)
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
    rank = int(np.linalg.matrix_rank(jacobian, tol=rcond * np.linalg.norm(jacobian, ord=2)))
    if rank != free_count:
        return result, False
    information = jacobian.T @ jacobian
    variance = float(np.dot(residual, residual) / (residual.size - free_count))
    try:
        free_covariance = np.linalg.pinv(information, rcond=rcond) * variance
    except np.linalg.LinAlgError:
        return result, False
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
    rss = float(np.dot(residual, residual))
    rmse = float(math.sqrt(rss / observations.size))
    centered = observations - float(np.mean(observations))
    total = float(np.dot(centered, centered))
    if total > 0 and np.isfinite(total):
        r_squared = float(1.0 - rss / total)
        r_squared_is_valid = bool(np.isfinite(r_squared))
    else:
        r_squared = 0.0
        r_squared_is_valid = False
    if not covariance_valid:
        covariance = np.zeros_like(covariance)
    return _CellSuccess(
        np.asarray(parameters, dtype=np.float64),
        np.asarray(covariance, dtype=np.float64),
        bool(covariance_valid),
        int(evaluations),
        rss,
        rmse,
        r_squared if r_squared_is_valid else 0.0,
        r_squared_is_valid,
    )


def _check_host_abort(
    cancel_check: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> None:
    if cancel_check is not None and cancel_check():
        raise FitCancelled("fit was cancelled")
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit hosting deadline exceeded")


def _bounded_error(message: str) -> str:
    compact = " ".join(str(message).split()) or "unspecified fit failure"
    return compact[:512]


__all__ = ["fit_analysis"]
