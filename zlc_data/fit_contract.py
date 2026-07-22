"""Serializable fit requests/results and non-serializable bound problem values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from typing import Callable

import numpy as np

from zlc_storage.canonical import canonical_text, sha256_text

from ._arrays import immutable_array, immutable_bool_array
from .axis import AxisId, AxisSpec
from .layout import AxisLayout
from .schema import DatasetSchema
from .transform import (
    CommittedTransform,
    TransformedSchema,
    _source_transformed_schema,
    resolve_transformed_schema,
)
from .value import DatasetRevisionRef, OwnedSnapshot
from .fit_model import (
    FitModelDefinition,
    FitParameterDefinition,
    ParameterUnitRelation,
    fit_model_definition,
)


class FitCoordinateSource(str, Enum):
    DECLARED = "DECLARED"
    LOGICAL_INDEX = "LOGICAL_INDEX"


_MAX_CONSECUTIVE_FLOAT64_INTEGER = 1 << 53


class FitBatchStatus(str, Enum):
    CONVERGED = "CONVERGED"
    NO_VALID_DATA = "NO_VALID_DATA"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    EVALUATION_LIMIT = "EVALUATION_LIMIT"
    SOLVER_FAILED = "SOLVER_FAILED"
    NUMERIC_ERROR = "NUMERIC_ERROR"


class FitCancelled(RuntimeError):
    """Raised when the hosting layer cancels an in-flight BoundFit."""


class FitDeadlineExceeded(RuntimeError):
    """Raised when the hosting layer's whole-analysis deadline expires."""


@dataclass(frozen=True)
class FitParameterConstraint:
    parameter_name: str
    initial: float | None = None
    lower: float | None = None
    upper: float | None = None
    fixed: float | None = None

    def __post_init__(self) -> None:
        canonical_text(self.parameter_name, "parameter_name")
        for field in ("initial", "lower", "upper", "fixed"):
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"constraint {field} must be a finite real number or None")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"constraint {field} must be finite")
            if normalized == 0.0:
                normalized = 0.0
            object.__setattr__(self, field, normalized)
        if self.lower is not None and self.upper is not None and not self.lower < self.upper:
            raise ValueError("constraint lower must be below upper")
        for field in ("initial", "fixed"):
            value = getattr(self, field)
            if value is None:
                continue
            if self.lower is not None and value < self.lower:
                raise ValueError(f"constraint {field} lies below lower")
            if self.upper is not None and value > self.upper:
                raise ValueError(f"constraint {field} lies above upper")
        if self.fixed is not None and self.initial is not None and self.fixed != self.initial:
            raise ValueError("fixed and initial values disagree")


@dataclass(frozen=True)
class FitNumericPolicy:
    max_evaluations: int = 4_000
    covariance_rcond: float = 1e-12

    def __post_init__(self) -> None:
        value = self.max_evaluations
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError("max_evaluations must be a positive integer")
        object.__setattr__(self, "max_evaluations", int(value))
        value = self.covariance_rcond
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("covariance_rcond must be a positive finite real number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError("covariance_rcond must be a positive finite real number")
        object.__setattr__(self, "covariance_rcond", normalized)


@dataclass(frozen=True)
class FitSpec:
    input_schema_fingerprint: str
    committed_transform: CommittedTransform | None
    fit_axis_ids: tuple[AxisId, ...]
    batch_axis_ids: tuple[AxisId, ...]
    model_id: str
    constraints: tuple[FitParameterConstraint, ...] = ()
    numeric_policy: FitNumericPolicy = FitNumericPolicy()

    def __post_init__(self) -> None:
        sha256_text(self.input_schema_fingerprint, "input_schema_fingerprint")
        if self.committed_transform is not None:
            if not isinstance(self.committed_transform, CommittedTransform):
                raise TypeError("committed_transform must be CommittedTransform or None")
            if self.committed_transform.input_schema_fingerprint != self.input_schema_fingerprint:
                raise ValueError("committed transform is bound to another input schema")
        fit_axes = tuple(self.fit_axis_ids)
        batch_axes = tuple(self.batch_axis_ids)
        if not fit_axes or any(not isinstance(axis_id, AxisId) for axis_id in fit_axes):
            raise ValueError("fit_axis_ids must contain named AxisId values")
        if any(not isinstance(axis_id, AxisId) for axis_id in batch_axes):
            raise TypeError("batch_axis_ids must contain AxisId values")
        if len(set(fit_axes)) != len(fit_axes) or len(set(batch_axes)) != len(batch_axes):
            raise ValueError("fit and batch axis ids must each be unique")
        if set(fit_axes) & set(batch_axes):
            raise ValueError("fit and batch axes cannot overlap")
        canonical_text(self.model_id, "model_id")
        model = fit_model_definition(self.model_id)
        constraints = tuple(self.constraints)
        if any(not isinstance(item, FitParameterConstraint) for item in constraints):
            raise TypeError("constraints must contain FitParameterConstraint values")
        names = tuple(item.parameter_name for item in constraints)
        if len(set(names)) != len(names):
            raise ValueError("a fit parameter may be constrained only once")
        parameters = {parameter.name: parameter for parameter in model.parameters}
        unknown = tuple(name for name in names if name not in parameters)
        if unknown:
            raise ValueError(f"constraints name unknown model parameters: {unknown!r}")
        for constraint in constraints:
            parameter = parameters[constraint.parameter_name]
            if (
                constraint.fixed is None
                and not parameter.has_free_interval(constraint.lower, constraint.upper)
            ):
                raise ValueError(
                    f"constraint bounds for {parameter.name!r} have no free interval in its "
                    f"{parameter.domain.value} domain"
                )
            for field in ("initial", "fixed"):
                value = getattr(constraint, field)
                if value is not None and not parameter.accepts(value):
                    raise ValueError(
                        f"constraint {field} for {parameter.name!r} violates its "
                        f"{parameter.domain.value} domain"
                    )
        if not isinstance(self.numeric_policy, FitNumericPolicy):
            raise TypeError("numeric_policy must be FitNumericPolicy")
        object.__setattr__(self, "fit_axis_ids", fit_axes)
        object.__setattr__(self, "batch_axis_ids", batch_axes)
        object.__setattr__(self, "constraints", tuple(sorted(constraints, key=lambda item: item.parameter_name)))


def _minimum_observation_count(spec: FitSpec, model: FitModelDefinition) -> int:
    """Smallest sample count that leaves one residual degree of freedom.

    Fixed parameters are hypotheses supplied by the caller, not quantities that
    the observations must identify.  The requirement therefore belongs to the
    bound request rather than to the model catalogue.
    """

    fixed = {
        constraint.parameter_name
        for constraint in spec.constraints
        if constraint.fixed is not None
    }
    free_count = sum(parameter.name not in fixed for parameter in model.parameters)
    return max(2, free_count + 1)


@dataclass(frozen=True)
class BoundFit:
    spec: FitSpec
    expected_schema: DatasetSchema
    effective_schema: TransformedSchema = field(init=False)
    model: FitModelDefinition = field(init=False)
    _coordinate_sources: tuple[FitCoordinateSource, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if not isinstance(self.expected_schema, DatasetSchema):
            raise TypeError("expected_schema must be DatasetSchema")
        if self.expected_schema.fingerprint != self.spec.input_schema_fingerprint:
            raise ValueError("BoundFit expected schema disagrees with FitSpec")
        effective_schema = _resolve_fit_effective_schema(self.spec, self.expected_schema)
        model = fit_model_definition(self.spec.model_id)
        object.__setattr__(self, "effective_schema", effective_schema)
        object.__setattr__(self, "model", model)
        if effective_schema.dtype.kind not in "biuf":
            raise TypeError("fit observations must use a real numeric dtype")
        effective_ids = tuple(axis.axis_id for axis in effective_schema.axes)
        requested_ids = self.spec.fit_axis_ids + self.spec.batch_axis_ids
        if len(requested_ids) != len(effective_ids) or set(requested_ids) != set(effective_ids):
            raise ValueError("BoundFit axes do not cover the effective schema exactly")
        if len(self.spec.fit_axis_ids) != model.independent_arity:
            raise ValueError(
                f"model {model.model_id!r} requires "
                f"{model.independent_arity} fit axes"
            )
        fit_axes = tuple(
            effective_schema.axis(axis_id) for axis_id in self.spec.fit_axis_ids
        )
        sources = tuple(_coordinate_source_for_axis(axis) for axis in fit_axes)
        for position, (axis, requirement) in enumerate(
            zip(fit_axes, model.axis_requirements)
        ):
            if axis.role not in requirement:
                roles = ", ".join(role.value for role in requirement)
                raise ValueError(
                    f"fit axis {position} ({axis.axis_id}) role {axis.role.value!r} "
                    f"does not satisfy model roles [{roles}]"
                )
        if model.require_common_axis_unit:
            units = tuple(
                _coordinate_unit(axis, source)
                for axis, source in zip(fit_axes, sources)
            )
            if len(set(units)) != 1:
                raise ValueError("model fit axes require compatible coordinate units")
        if model.require_common_coordinate_frame and len(
            set(axis.coordinate_frame for axis in fit_axes)
        ) != 1:
            raise ValueError("model fit axes require the same coordinate frame")
        object.__setattr__(self, "_coordinate_sources", sources)

    def run(
        self,
        snapshot: OwnedSnapshot,
        *,
        cancel_check: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
    ) -> "FitResultBatch":
        from .fit_solver import _fit_analysis

        return _fit_analysis(
            self,
            snapshot,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )

    @property
    def minimum_observation_count(self) -> int:
        return _minimum_observation_count(self.spec, self.model)

    @property
    def parameter_definitions(self) -> tuple[FitParameterDefinition, ...]:
        return self.model.parameters

    @property
    def coordinate_sources(self) -> tuple[FitCoordinateSource, ...]:
        return self._coordinate_sources

    @property
    def parameter_units(self) -> tuple[str, ...]:
        fit_axes = tuple(
            self.effective_schema.axis(axis_id) for axis_id in self.spec.fit_axis_ids
        )
        return resolve_parameter_units(
            self.parameter_definitions,
            fit_axes,
            self.coordinate_sources,
            self.effective_schema.value_unit,
        )


@dataclass(frozen=True)
class FitProblem:
    source_ref: DatasetRevisionRef
    spec: FitSpec
    fit_axis_specs: tuple[AxisSpec, ...]
    batch_axis_specs: tuple[AxisSpec, ...]
    batch_layout: AxisLayout
    value_unit: str | None
    batch_offsets: np.ndarray
    independent_values: tuple[np.ndarray, ...]
    observations: np.ndarray
    present_observation_counts: np.ndarray
    valid_observation_counts: np.ndarray
    used_observation_counts: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("source_ref must be DatasetRevisionRef")
        if not isinstance(self.spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if self.source_ref.schema_fingerprint != self.spec.input_schema_fingerprint:
            raise ValueError("fit problem source lineage disagrees with FitSpec")
        fit_axes = tuple(self.fit_axis_specs)
        batch_axes = tuple(self.batch_axis_specs)
        if any(not isinstance(axis, AxisSpec) for axis in fit_axes + batch_axes):
            raise TypeError("fit problem axes must contain AxisSpec values")
        if tuple(axis.axis_id for axis in fit_axes) != self.spec.fit_axis_ids:
            raise ValueError("fit problem axis order disagrees with FitSpec")
        if tuple(axis.axis_id for axis in batch_axes) != self.spec.batch_axis_ids:
            raise ValueError("batch problem axis order disagrees with FitSpec")
        if not isinstance(self.batch_layout, AxisLayout):
            raise TypeError("batch_layout must be AxisLayout")
        if self.batch_layout.logical_shape != tuple(axis.size for axis in batch_axes):
            raise ValueError("batch layout shape does not match batch axes")
        if self.value_unit is not None:
            canonical_text(self.value_unit, "value_unit")
        batch_size = self.batch_layout.storage_size
        offsets = _immutable_numeric(self.batch_offsets, "<i8", (batch_size + 1,))
        if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("fit problem batch offsets must be monotonic and start at zero")
        packed_size = int(offsets[-1])
        observations = _immutable_numeric(self.observations, "<f8", (packed_size,))
        independent = tuple(
            _immutable_numeric(values, "<f8", (packed_size,))
            for values in self.independent_values
        )
        if len(independent) != len(fit_axes):
            raise ValueError("fit problem needs one packed coordinate vector per fit axis")
        counts = {}
        for field in (
            "present_observation_counts",
            "valid_observation_counts",
            "used_observation_counts",
        ):
            counts[field] = _immutable_numeric(getattr(self, field), "<i8", (batch_size,))
            if np.any(counts[field] < 0):
                raise ValueError(f"{field} cannot contain negative counts")
            object.__setattr__(self, field, counts[field])
        if np.any(counts["valid_observation_counts"] > counts["present_observation_counts"]):
            raise ValueError("valid observation count exceeds present count")
        if np.any(counts["used_observation_counts"] > counts["valid_observation_counts"]):
            raise ValueError("used observation count exceeds valid count")
        if not np.array_equal(np.diff(offsets), counts["used_observation_counts"]):
            raise ValueError("packed offsets disagree with used observation counts")
        object.__setattr__(self, "fit_axis_specs", fit_axes)
        object.__setattr__(self, "batch_axis_specs", batch_axes)
        object.__setattr__(self, "batch_offsets", offsets)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "independent_values", independent)

    @property
    def effective_schema_fingerprint(self) -> str:
        if self.spec.committed_transform is None:
            return self.spec.input_schema_fingerprint
        return self.spec.committed_transform.output_schema_fingerprint

@dataclass(frozen=True, eq=False)
class FitResultBatch:
    source_ref: DatasetRevisionRef
    spec: FitSpec
    fit_axis_specs: tuple[AxisSpec, ...]
    batch_axis_specs: tuple[AxisSpec, ...]
    batch_layout: AxisLayout
    value_unit: str | None
    parameter_values: np.ndarray
    covariance: np.ndarray
    covariance_valid: np.ndarray
    statuses: tuple[FitBatchStatus, ...]
    errors: tuple[str | None, ...]
    present_observation_counts: np.ndarray
    valid_observation_counts: np.ndarray
    used_observation_counts: np.ndarray
    evaluation_counts: np.ndarray
    residual_sum_squares: np.ndarray
    r_squared: np.ndarray
    r_squared_valid: np.ndarray
    scipy_version: str
    _coordinate_sources: tuple[FitCoordinateSource, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("source_ref must be DatasetRevisionRef")
        if not isinstance(self.spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if self.source_ref.schema_fingerprint != self.spec.input_schema_fingerprint:
            raise ValueError("fit result source lineage disagrees with FitSpec")
        fit_axes = tuple(self.fit_axis_specs)
        batch_axes = tuple(self.batch_axis_specs)
        if any(not isinstance(axis, AxisSpec) for axis in fit_axes + batch_axes):
            raise TypeError("fit result axes must contain AxisSpec values")
        sources = tuple(_coordinate_source_for_axis(axis) for axis in fit_axes)
        model = fit_model_definition(self.spec.model_id)
        parameters = model.parameters
        if tuple(axis.axis_id for axis in fit_axes) != self.spec.fit_axis_ids:
            raise ValueError("fit result axis order disagrees with FitSpec")
        if tuple(axis.axis_id for axis in batch_axes) != self.spec.batch_axis_ids:
            raise ValueError("batch result axis order disagrees with FitSpec")
        if not isinstance(self.batch_layout, AxisLayout):
            raise TypeError("batch_layout must be AxisLayout")
        if self.batch_layout.logical_shape != tuple(axis.size for axis in batch_axes):
            raise ValueError("batch layout shape does not match batch axes")
        if self.value_unit is not None:
            canonical_text(self.value_unit, "value_unit")
        minimum_observations = _minimum_observation_count(self.spec, model)
        batch_size = self.batch_layout.storage_size
        parameter_count = len(parameters)
        object.__setattr__(
            self,
            "parameter_values",
            _immutable_numeric(self.parameter_values, "<f8", (batch_size, parameter_count)),
        )
        object.__setattr__(
            self,
            "covariance",
            _immutable_numeric(self.covariance, "<f8", (batch_size, parameter_count, parameter_count)),
        )
        object.__setattr__(
            self,
            "covariance_valid",
            immutable_bool_array(self.covariance_valid, shape=(batch_size,)),
        )
        for field in (
            "present_observation_counts",
            "valid_observation_counts",
            "used_observation_counts",
            "evaluation_counts",
        ):
            object.__setattr__(self, field, _immutable_numeric(getattr(self, field), "<i8", (batch_size,)))
            if np.any(getattr(self, field) < 0):
                raise ValueError(f"{field} cannot contain negative counts")
        for field in ("residual_sum_squares", "r_squared"):
            object.__setattr__(self, field, _immutable_numeric(getattr(self, field), "<f8", (batch_size,)))
        object.__setattr__(
            self,
            "r_squared_valid",
            immutable_bool_array(self.r_squared_valid, shape=(batch_size,)),
        )
        statuses = tuple(self.statuses)
        errors = tuple(self.errors)
        if len(statuses) != batch_size or any(not isinstance(item, FitBatchStatus) for item in statuses):
            raise ValueError("statuses must match batch layout")
        if len(errors) != batch_size:
            raise ValueError("errors must contain text or None per batch")
        for value in errors:
            if value is None:
                continue
            canonical_text(value, "errors item")
        fixed_indices = tuple(
            index
            for index, parameter in enumerate(parameters)
            if any(
                constraint.parameter_name == parameter.name and constraint.fixed is not None
                for constraint in self.spec.constraints
            )
        )
        for index, status in enumerate(statuses):
            if status is FitBatchStatus.CONVERGED:
                if errors[index] is not None or not np.all(np.isfinite(self.parameter_values[index])):
                    raise ValueError("converged batch result must have finite parameters and no execution error")
                for parameter, value in zip(parameters, self.parameter_values[index]):
                    if not parameter.accepts(float(value)):
                        raise ValueError(
                            f"converged batch parameter {parameter.name!r} violates its "
                            f"{parameter.domain.value} domain"
                        )
                values_by_name = {
                    parameter.name: float(value)
                    for parameter, value in zip(parameters, self.parameter_values[index])
                }
                for constraint in self.spec.constraints:
                    value = values_by_name[constraint.parameter_name]
                    if constraint.fixed is not None and value != constraint.fixed:
                        raise ValueError(
                            f"converged batch violates fixed {constraint.parameter_name!r}"
                        )
                    if constraint.lower is not None and value < constraint.lower:
                        raise ValueError(
                            f"converged batch violates lower bound for "
                            f"{constraint.parameter_name!r}"
                        )
                    if constraint.upper is not None and value > constraint.upper:
                        raise ValueError(
                            f"converged batch violates upper bound for "
                            f"{constraint.parameter_name!r}"
                        )
                if self.used_observation_counts[index] < minimum_observations:
                    raise ValueError("converged batch result has too few used observations")
                if self.evaluation_counts[index] <= 0:
                    raise ValueError("converged batch result requires model evaluations")
                if (
                    not np.isfinite(self.residual_sum_squares[index])
                    or self.residual_sum_squares[index] < 0
                ):
                    raise ValueError("converged batch result requires finite non-negative RSS")
                covariance = self.covariance[index]
                if self.covariance_valid[index]:
                    if not np.all(np.isfinite(covariance)) or not np.allclose(
                        covariance,
                        covariance.T,
                        rtol=0.0,
                        atol=0.0,
                    ):
                        raise ValueError("valid covariance must be finite and exactly symmetric")
                    if np.any(np.diag(covariance) < 0):
                        raise ValueError("valid covariance cannot have negative diagonal variance")
                    diagonal = np.diag(covariance)
                    zero_variance = np.flatnonzero(diagonal == 0.0)
                    if zero_variance.size and (
                        np.any(covariance[zero_variance, :])
                        or np.any(covariance[:, zero_variance])
                    ):
                        raise ValueError("zero-variance covariance rows and columns must be zero")
                    positive_variance = np.flatnonzero(diagonal > 0.0)
                    if positive_variance.size:
                        submatrix = covariance[np.ix_(positive_variance, positive_variance)]
                        scales = np.sqrt(diagonal[positive_variance])
                        correlation = submatrix / np.outer(scales, scales)
                        eigenvalues = np.linalg.eigvalsh(correlation)
                        tolerance = (
                            64.0
                            * np.finfo(np.float64).eps
                            * positive_variance.size
                            * max(float(np.max(np.abs(eigenvalues))), 1.0)
                        )
                        if float(np.min(eigenvalues)) < -tolerance:
                            raise ValueError("valid covariance must be positive semidefinite")
                    if fixed_indices and (
                        np.any(covariance[list(fixed_indices), :])
                        or np.any(covariance[:, list(fixed_indices)])
                    ):
                        raise ValueError("fixed parameter covariance rows and columns must be zero")
                elif np.any(covariance):
                    raise ValueError("invalid covariance payload must be canonical zero")
                if self.r_squared_valid[index]:
                    if not np.isfinite(self.r_squared[index]) or self.r_squared[index] > 1.0:
                        raise ValueError("valid r_squared must be finite and no greater than one")
                elif self.r_squared[index] != 0:
                    raise ValueError("invalid r_squared payload must be canonical zero")
            else:
                if errors[index] is None:
                    raise ValueError("failed batch result requires an error")
                if np.any(self.parameter_values[index]) or np.any(self.covariance[index]):
                    raise ValueError("failed batch result numeric payload must be canonical zero")
                if self.covariance_valid[index] or self.r_squared_valid[index]:
                    raise ValueError("failed batch result cannot mark optional metrics valid")
                if (
                    self.residual_sum_squares[index] != 0
                    or self.r_squared[index] != 0
                ):
                    raise ValueError("failed batch metrics must be canonical zero")
            valid_count = int(self.valid_observation_counts[index])
            used_count = int(self.used_observation_counts[index])
            if (status is FitBatchStatus.NO_VALID_DATA) != (valid_count == 0):
                raise ValueError("NO_VALID_DATA status disagrees with validity counts")
            insufficient = valid_count > 0 and used_count < minimum_observations
            if (status is FitBatchStatus.INSUFFICIENT_POINTS) != insufficient:
                raise ValueError("INSUFFICIENT_POINTS status disagrees with used counts")
        if np.any(self.valid_observation_counts > self.present_observation_counts):
            raise ValueError("valid observation count exceeds present count")
        if np.any(self.used_observation_counts > self.valid_observation_counts):
            raise ValueError("used observation count exceeds valid count")
        if np.any(self.evaluation_counts > self.spec.numeric_policy.max_evaluations):
            raise ValueError("evaluation count exceeds max_evaluations")
        canonical_text(self.scipy_version, "scipy_version")
        object.__setattr__(self, "fit_axis_specs", fit_axes)
        object.__setattr__(self, "batch_axis_specs", batch_axes)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "_coordinate_sources", sources)

    @property
    def rmse(self) -> np.ndarray:
        """Canonical derived metric; RSS/count is the only stored authority."""

        values = np.zeros_like(self.residual_sum_squares)
        converged = np.fromiter(
            (status is FitBatchStatus.CONVERGED for status in self.statuses),
            dtype=bool,
            count=len(self.statuses),
        )
        values[converged] = np.sqrt(
            self.residual_sum_squares[converged]
            / self.used_observation_counts[converged]
        )
        return _immutable_numeric(values, "<f8", values.shape)

    @property
    def effective_schema_fingerprint(self) -> str:
        if self.spec.committed_transform is None:
            return self.spec.input_schema_fingerprint
        return self.spec.committed_transform.output_schema_fingerprint

    @property
    def coordinate_sources(self) -> tuple[FitCoordinateSource, ...]:
        return self._coordinate_sources

    @property
    def parameter_definitions(self) -> tuple[FitParameterDefinition, ...]:
        return fit_model_definition(self.spec.model_id).parameters

    @property
    def parameter_units(self) -> tuple[str, ...]:
        return resolve_parameter_units(
            self.parameter_definitions,
            self.fit_axis_specs,
            self.coordinate_sources,
            self.value_unit,
        )

    def evaluate_batch(
        self,
        batch_storage_index: int,
        coordinates: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        """Evaluate one numerically converged batch on declared-axis coordinates.

        This is the result-owned path used by overlays and replay.  Coordinates
        retain their absolute declared-axis meaning; selection never rebases
        amplitude or phase.
        """

        if (
            isinstance(batch_storage_index, bool)
            or not isinstance(batch_storage_index, Integral)
            or not 0 <= int(batch_storage_index) < self.batch_layout.storage_size
        ):
            raise IndexError("batch_storage_index is outside the result layout")
        index = int(batch_storage_index)
        if self.statuses[index] is not FitBatchStatus.CONVERGED:
            raise ValueError("cannot evaluate a failed fit batch")
        from .fit_model import evaluate_fit_model

        model = fit_model_definition(self.spec.model_id)
        return evaluate_fit_model(model, coordinates, self.parameter_values[index])


def resolve_parameter_units(
    parameters: tuple[FitParameterDefinition, ...],
    fit_axes: tuple[AxisSpec, ...],
    coordinate_sources: tuple[FitCoordinateSource, ...],
    value_unit: str | None,
) -> tuple[str, ...]:
    """Resolve model-relative units into self-contained artifact metadata."""

    if len(coordinate_sources) != len(fit_axes):
        raise ValueError("coordinate sources must describe the fit axes")
    axis_units = tuple(
        _coordinate_unit(axis, source)
        for axis, source in zip(fit_axes, coordinate_sources)
    )
    units: list[str] = []
    for parameter in parameters:
        relation = parameter.unit_relation
        if relation is ParameterUnitRelation.VALUE:
            unit = value_unit or "1"
        elif relation is ParameterUnitRelation.AXIS_0:
            unit = axis_units[0]
        elif relation is ParameterUnitRelation.AXIS_1:
            if len(fit_axes) < 2:
                raise ValueError("AXIS_1 parameter requires a second fit axis")
            unit = axis_units[1]
        elif relation is ParameterUnitRelation.INVERSE_AXIS_0:
            axis_unit = axis_units[0]
            unit = "1" if axis_unit == "1" else f"1/({axis_unit})"
        elif relation is ParameterUnitRelation.RADIAN:
            unit = "rad"
        else:  # pragma: no cover - closed enum
            raise RuntimeError(f"unsupported parameter unit relation {relation!r}")
        units.append(unit)
    return tuple(units)


def _immutable_numeric(values, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "f" and np.any(array == 0.0):
        array = array.copy()
        array[array == 0.0] = 0.0
    return immutable_array(array, dtype=np.dtype(dtype), shape=shape)


def _coordinate_source_for_axis(axis: AxisSpec) -> FitCoordinateSource:
    if axis.coordinates is None:
        last = axis.index_origin + axis.size - 1
        if axis.size > 1 and last > _MAX_CONSECUTIVE_FLOAT64_INTEGER:
            raise ValueError(
                "implicit fit coordinates are not consecutively float64-representable"
            )
        for value in {axis.index_origin, last}:
            try:
                converted = float(value)
            except OverflowError as exc:
                raise ValueError(
                    "implicit fit coordinate is not float64-representable"
                ) from exc
            if not math.isfinite(converted) or int(converted) != value:
                raise ValueError("implicit fit coordinate is not exactly float64-representable")
        return FitCoordinateSource.LOGICAL_INDEX
    if not all(
        not isinstance(value, bool) and isinstance(value, Real)
        for value in axis.coordinates
    ):
        raise TypeError(
            f"declared coordinates for fit axis {axis.axis_id} must be entirely numeric"
        )
    for value in axis.coordinates:
        try:
            converted = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("fit coordinate is not float64-representable") from exc
        if not math.isfinite(converted) or (
            isinstance(value, Integral) and int(converted) != int(value)
        ):
            raise ValueError("fit coordinate is not exactly float64-representable")
    return FitCoordinateSource.DECLARED


def _coordinate_unit(axis: AxisSpec, source: FitCoordinateSource) -> str:
    if axis.unit is not None:
        return axis.unit
    return "index" if source is FitCoordinateSource.LOGICAL_INDEX else "1"


def _resolve_fit_effective_schema(
    spec: FitSpec,
    expected_schema: DatasetSchema,
) -> TransformedSchema:
    if spec.committed_transform is not None:
        return resolve_transformed_schema(expected_schema, spec.committed_transform)
    return _source_transformed_schema(expected_schema)


__all__ = [
    "BoundFit",
    "FitBatchStatus",
    "FitCancelled",
    "FitCoordinateSource",
    "FitDeadlineExceeded",
    "FitNumericPolicy",
    "FitParameterConstraint",
    "FitResultBatch",
    "FitSpec",
]
