"""Serializable fit requests/results and non-serializable bound problem values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
from typing import Callable

import numpy as np

from zlc_storage.canonical import canonical_digest

from ._arrays import immutable_array, immutable_bool_array
from .axis import AxisId, AxisSpec
from .layout import AxisLayout, AxisLayoutMode
from .schema import DatasetSchema
from .transform import CommittedTransform, TransformedSchema, resolve_transformed_schema
from .validity import ValidityMode
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


class FitBatchStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_VALID_DATA = "NO_VALID_DATA"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    EVALUATION_LIMIT = "EVALUATION_LIMIT"
    TIMEOUT = "TIMEOUT"
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
        if (
            not isinstance(self.parameter_name, str)
            or not self.parameter_name
            or self.parameter_name.strip() != self.parameter_name
        ):
            raise ValueError("parameter_name must be canonical non-empty text")
        for field in ("initial", "lower", "upper", "fixed"):
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"constraint {field} must be a finite real number or None")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"constraint {field} must be finite")
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
    max_seconds_per_batch: float = 5.0
    max_total_seconds: float = 120.0
    max_batch_cells: int = 100_000
    sample_budget_per_batch: int = 12_000
    covariance_rcond: float = 1e-12

    def __post_init__(self) -> None:
        for field in ("max_evaluations", "max_batch_cells", "sample_budget_per_batch"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
            object.__setattr__(self, field, int(value))
        for field in ("max_seconds_per_batch", "max_total_seconds", "covariance_rcond"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field} must be a positive finite real number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(f"{field} must be a positive finite real number")
            object.__setattr__(self, field, normalized)


@dataclass(frozen=True)
class FitSpec:
    input_schema_fingerprint: str
    committed_transform: CommittedTransform | None
    fit_axis_ids: tuple[AxisId, ...]
    batch_axis_ids: tuple[AxisId, ...]
    model_id: str
    model_version: int = 1
    constraints: tuple[FitParameterConstraint, ...] = ()
    numeric_policy: FitNumericPolicy = FitNumericPolicy()

    def __post_init__(self) -> None:
        _require_digest(self.input_schema_fingerprint, "input_schema_fingerprint")
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
        if not isinstance(self.model_id, str) or not self.model_id or self.model_id.strip() != self.model_id:
            raise ValueError("model_id must be canonical non-empty text")
        if isinstance(self.model_version, bool) or not isinstance(self.model_version, Integral) or self.model_version <= 0:
            raise ValueError("model_version must be a positive integer")
        constraints = tuple(self.constraints)
        if any(not isinstance(item, FitParameterConstraint) for item in constraints):
            raise TypeError("constraints must contain FitParameterConstraint values")
        names = tuple(item.parameter_name for item in constraints)
        if len(set(names)) != len(names):
            raise ValueError("a fit parameter may be constrained only once")
        if not isinstance(self.numeric_policy, FitNumericPolicy):
            raise TypeError("numeric_policy must be FitNumericPolicy")
        object.__setattr__(self, "fit_axis_ids", fit_axes)
        object.__setattr__(self, "batch_axis_ids", batch_axes)
        object.__setattr__(self, "model_version", int(self.model_version))
        object.__setattr__(self, "constraints", tuple(sorted(constraints, key=lambda item: item.parameter_name)))

    @property
    def digest(self) -> str:
        from .fit_codec import fit_spec_to_tree

        return canonical_digest(fit_spec_to_tree(self))


@dataclass(frozen=True)
class BoundFit:
    spec: FitSpec
    expected_schema: DatasetSchema
    effective_schema: TransformedSchema
    model: FitModelDefinition

    @classmethod
    def bind(cls, spec: FitSpec, expected_schema: DatasetSchema) -> "BoundFit":
        if not isinstance(spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if not isinstance(expected_schema, DatasetSchema):
            raise TypeError("expected_schema must be DatasetSchema")
        if expected_schema.fingerprint != spec.input_schema_fingerprint:
            raise ValueError("FitSpec input schema fingerprint is stale")
        return cls(
            spec,
            expected_schema,
            _resolve_fit_effective_schema(spec, expected_schema),
            fit_model_definition(spec.model_id, spec.model_version),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if not isinstance(self.expected_schema, DatasetSchema):
            raise TypeError("expected_schema must be DatasetSchema")
        if not isinstance(self.effective_schema, TransformedSchema):
            raise TypeError("effective_schema must be TransformedSchema")
        if not isinstance(self.model, FitModelDefinition):
            raise TypeError("model must be FitModelDefinition")
        if self.expected_schema.fingerprint != self.spec.input_schema_fingerprint:
            raise ValueError("BoundFit expected schema disagrees with FitSpec")
        if self.model != fit_model_definition(self.spec.model_id, self.spec.model_version):
            raise ValueError("BoundFit model disagrees with FitSpec")
        derived_schema = _resolve_fit_effective_schema(self.spec, self.expected_schema)
        if self.effective_schema != derived_schema:
            raise ValueError("BoundFit effective schema is not derived from FitSpec authority")
        if self.effective_schema.dtype.kind not in "biuf":
            raise TypeError("fit observations must use a real numeric dtype")
        effective_ids = tuple(axis.axis_id for axis in self.effective_schema.axes)
        requested_ids = self.spec.fit_axis_ids + self.spec.batch_axis_ids
        if len(requested_ids) != len(effective_ids) or set(requested_ids) != set(effective_ids):
            raise ValueError("BoundFit axes do not cover the effective schema exactly")
        if self.spec.numeric_policy.sample_budget_per_batch < self.model.minimum_observations:
            raise ValueError(
                "sample_budget_per_batch is below the selected model's minimum_observations"
            )
        if len(self.spec.fit_axis_ids) != self.model.independent_arity:
            raise ValueError(
                f"model {self.model.model_id!r} requires "
                f"{self.model.independent_arity} fit axes"
            )
        fit_axes = tuple(
            self.effective_schema.axis(axis_id) for axis_id in self.spec.fit_axis_ids
        )
        sources = tuple(_coordinate_source_for_axis(axis) for axis in fit_axes)
        for position, (axis, requirement) in enumerate(
            zip(fit_axes, self.model.axis_requirements)
        ):
            if axis.role not in requirement.allowed_roles:
                roles = ", ".join(role.value for role in requirement.allowed_roles)
                raise ValueError(
                    f"fit axis {position} ({axis.axis_id}) role {axis.role.value!r} "
                    f"does not satisfy model roles [{roles}]"
                )
        if self.model.require_common_axis_unit:
            units = tuple(
                "index"
                if source is FitCoordinateSource.LOGICAL_INDEX
                else (axis.unit or "1")
                for axis, source in zip(fit_axes, sources)
            )
            if len(set(units)) != 1:
                raise ValueError("model fit axes require compatible coordinate units")
        if self.model.require_common_coordinate_frame and len(
            set(axis.coordinate_frame for axis in fit_axes)
        ) != 1:
            raise ValueError("model fit axes require the same coordinate frame")
        parameter_names = set(self.model.parameter_names)
        unknown = tuple(
            constraint.parameter_name
            for constraint in self.spec.constraints
            if constraint.parameter_name not in parameter_names
        )
        if unknown:
            raise ValueError(f"constraints name unknown model parameters: {unknown!r}")

    def run(
        self,
        snapshot: OwnedSnapshot,
        *,
        cancel_check: Callable[[], bool] | None = None,
        deadline_monotonic: float | None = None,
    ) -> "FitResultBatch":
        from .fit_solver import fit_analysis

        return fit_analysis(
            self,
            snapshot,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )


@dataclass(frozen=True, eq=False)
class FitCellProblem:
    """A transient zero-copy view of one row in a packed :class:`FitProblem`."""

    batch_multi_index: tuple[int, ...]
    independent_values: tuple[np.ndarray, ...]
    observations: np.ndarray
    present_observation_count: int
    valid_observation_count: int
    used_observation_count: int
    __hash__ = None

    def __post_init__(self) -> None:
        multi = tuple(self.batch_multi_index)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in multi):
            raise ValueError("batch_multi_index must contain non-negative integers")
        observations = np.asarray(self.observations)
        coords = tuple(np.asarray(value) for value in self.independent_values)
        if observations.dtype != np.dtype("<f8") or observations.ndim != 1:
            raise TypeError("fit cell observations must be a canonical float64 vector")
        if any(value.dtype != np.dtype("<f8") or value.ndim != 1 for value in coords):
            raise TypeError("fit cell coordinates must be canonical float64 vectors")
        if not coords or any(value.shape != observations.shape for value in coords):
            raise ValueError("fit cell coordinates and observations must share one shape")
        for field in (
            "present_observation_count",
            "valid_observation_count",
            "used_observation_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            object.__setattr__(self, field, int(value))
        if not (
            observations.size
            == self.used_observation_count
            <= self.valid_observation_count
            <= self.present_observation_count
        ):
            raise ValueError("fit cell observation counts disagree")
        object.__setattr__(self, "batch_multi_index", tuple(int(value) for value in multi))
        if observations.flags.writeable or any(value.flags.writeable for value in coords):
            raise ValueError("fit cell views must be read-only")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "independent_values", coords)


@dataclass(frozen=True)
class FitProblem:
    source_ref: DatasetRevisionRef
    spec: FitSpec
    effective_schema_fingerprint: str
    fit_axis_specs: tuple[AxisSpec, ...]
    coordinate_sources: tuple[FitCoordinateSource, ...]
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
        _require_digest(self.effective_schema_fingerprint, "effective_schema_fingerprint")
        _validate_effective_lineage(self.spec, self.effective_schema_fingerprint)
        fit_axes = tuple(self.fit_axis_specs)
        batch_axes = tuple(self.batch_axis_specs)
        sources = tuple(self.coordinate_sources)
        if any(not isinstance(axis, AxisSpec) for axis in fit_axes + batch_axes):
            raise TypeError("fit problem axes must contain AxisSpec values")
        if tuple(axis.axis_id for axis in fit_axes) != self.spec.fit_axis_ids:
            raise ValueError("fit problem axis order disagrees with FitSpec")
        if tuple(axis.axis_id for axis in batch_axes) != self.spec.batch_axis_ids:
            raise ValueError("batch problem axis order disagrees with FitSpec")
        if len(sources) != len(fit_axes) or any(not isinstance(item, FitCoordinateSource) for item in sources):
            raise ValueError("coordinate_sources must describe every fit axis")
        _validate_coordinate_sources(fit_axes, sources)
        if not isinstance(self.batch_layout, AxisLayout):
            raise TypeError("batch_layout must be AxisLayout")
        if self.batch_layout.logical_shape != tuple(axis.size for axis in batch_axes):
            raise ValueError("batch layout shape does not match batch axes")
        if self.value_unit is not None and (
            not isinstance(self.value_unit, str)
            or not self.value_unit
            or self.value_unit.strip() != self.value_unit
        ):
            raise ValueError("value_unit must be canonical non-empty text or None")
        batch_size = self.batch_layout.storage_size
        if batch_size > self.spec.numeric_policy.max_batch_cells:
            raise ValueError("fit problem exceeds its declared batch-cell budget")
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
        if np.any(
            counts["used_observation_counts"]
            > self.spec.numeric_policy.sample_budget_per_batch
        ):
            raise ValueError("used observation count exceeds the declared sampling budget")
        if not np.array_equal(np.diff(offsets), counts["used_observation_counts"]):
            raise ValueError("packed offsets disagree with used observation counts")
        object.__setattr__(self, "fit_axis_specs", fit_axes)
        object.__setattr__(self, "batch_axis_specs", batch_axes)
        object.__setattr__(self, "coordinate_sources", sources)
        object.__setattr__(self, "batch_offsets", offsets)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "independent_values", independent)

    def cell(self, storage_index: int) -> FitCellProblem:
        """Create a transient read-only view without retaining per-batch objects."""

        multi = self.batch_layout.multi_index(storage_index)
        start = int(self.batch_offsets[storage_index])
        stop = int(self.batch_offsets[storage_index + 1])
        return FitCellProblem(
            multi,
            tuple(values[start:stop] for values in self.independent_values),
            self.observations[start:stop],
            int(self.present_observation_counts[storage_index]),
            int(self.valid_observation_counts[storage_index]),
            int(self.used_observation_counts[storage_index]),
        )


@dataclass(frozen=True, eq=False)
class FitResultBatch:
    source_ref: DatasetRevisionRef
    spec: FitSpec
    effective_schema_fingerprint: str
    fit_axis_specs: tuple[AxisSpec, ...]
    coordinate_sources: tuple[FitCoordinateSource, ...]
    batch_axis_specs: tuple[AxisSpec, ...]
    batch_layout: AxisLayout
    value_unit: str | None
    parameter_definitions: tuple[FitParameterDefinition, ...]
    parameter_units: tuple[str, ...]
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
    rmse: np.ndarray
    r_squared: np.ndarray
    r_squared_valid: np.ndarray
    solver_contract_id: str
    scipy_version: str
    initializer_id: str
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("source_ref must be DatasetRevisionRef")
        if not isinstance(self.spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if self.source_ref.schema_fingerprint != self.spec.input_schema_fingerprint:
            raise ValueError("fit result source lineage disagrees with FitSpec")
        _require_digest(self.effective_schema_fingerprint, "effective_schema_fingerprint")
        _validate_effective_lineage(self.spec, self.effective_schema_fingerprint)
        fit_axes = tuple(self.fit_axis_specs)
        batch_axes = tuple(self.batch_axis_specs)
        sources = tuple(self.coordinate_sources)
        parameters = tuple(self.parameter_definitions)
        parameter_units = tuple(self.parameter_units)
        if any(not isinstance(axis, AxisSpec) for axis in fit_axes + batch_axes):
            raise TypeError("fit result axes must contain AxisSpec values")
        if tuple(axis.axis_id for axis in fit_axes) != self.spec.fit_axis_ids:
            raise ValueError("fit result axis order disagrees with FitSpec")
        if tuple(axis.axis_id for axis in batch_axes) != self.spec.batch_axis_ids:
            raise ValueError("batch result axis order disagrees with FitSpec")
        if len(sources) != len(fit_axes) or any(not isinstance(item, FitCoordinateSource) for item in sources):
            raise ValueError("coordinate_sources must describe every fit axis")
        _validate_coordinate_sources(fit_axes, sources)
        if not isinstance(self.batch_layout, AxisLayout):
            raise TypeError("batch_layout must be AxisLayout")
        if self.batch_layout.logical_shape != tuple(axis.size for axis in batch_axes):
            raise ValueError("batch layout shape does not match batch axes")
        if self.value_unit is not None and (
            not isinstance(self.value_unit, str)
            or not self.value_unit
            or self.value_unit.strip() != self.value_unit
        ):
            raise ValueError("value_unit must be canonical non-empty text or None")
        if any(not isinstance(item, FitParameterDefinition) for item in parameters):
            raise TypeError("parameter_definitions must contain FitParameterDefinition values")
        model = fit_model_definition(self.spec.model_id, self.spec.model_version)
        if parameters != model.parameters:
            raise ValueError("fit result parameter schema disagrees with model version")
        if len(parameter_units) != len(parameters) or any(
            not isinstance(value, str)
            or not value
            or value.strip() != value
            for value in parameter_units
        ):
            raise ValueError("parameter_units must contain canonical strings")
        if parameter_units != resolve_parameter_units(
            parameters,
            fit_axes,
            sources,
            self.value_unit,
        ):
            raise ValueError("fit result parameter units disagree with authoritative axes")
        batch_size = self.batch_layout.storage_size
        if batch_size > self.spec.numeric_policy.max_batch_cells:
            raise ValueError("fit result exceeds its declared batch-cell budget")
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
        for field in ("residual_sum_squares", "rmse", "r_squared"):
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
        if len(errors) != batch_size or any(
            value is not None and (not isinstance(value, str) or not value or len(value) > 512)
            for value in errors
        ):
            raise ValueError("errors must contain bounded text or None per batch")
        for index, status in enumerate(statuses):
            if status is FitBatchStatus.SUCCESS:
                if errors[index] is not None or not np.all(np.isfinite(self.parameter_values[index])):
                    raise ValueError("successful batch result must have finite parameters and no error")
                if self.used_observation_counts[index] < model.minimum_observations:
                    raise ValueError("successful batch result has too few used observations")
                if self.evaluation_counts[index] <= 0:
                    raise ValueError("successful batch result requires solver evaluations")
                if (
                    not np.isfinite(self.residual_sum_squares[index])
                    or self.residual_sum_squares[index] < 0
                    or not np.isfinite(self.rmse[index])
                    or self.rmse[index] < 0
                ):
                    raise ValueError("successful batch result requires finite non-negative metrics")
                covariance = self.covariance[index]
                if self.covariance_valid[index]:
                    if not np.all(np.isfinite(covariance)) or not np.allclose(
                        covariance,
                        covariance.T,
                        rtol=0.0,
                        atol=0.0,
                    ):
                        raise ValueError("valid covariance must be finite and exactly symmetric")
                elif np.any(covariance):
                    raise ValueError("invalid covariance payload must be canonical zero")
                if self.r_squared_valid[index]:
                    if not np.isfinite(self.r_squared[index]):
                        raise ValueError("valid r_squared must be finite")
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
                    or self.rmse[index] != 0
                    or self.r_squared[index] != 0
                ):
                    raise ValueError("failed batch metrics must be canonical zero")
            valid_count = int(self.valid_observation_counts[index])
            used_count = int(self.used_observation_counts[index])
            if (status is FitBatchStatus.NO_VALID_DATA) != (valid_count == 0):
                raise ValueError("NO_VALID_DATA status disagrees with validity counts")
            insufficient = valid_count > 0 and used_count < model.minimum_observations
            if (status is FitBatchStatus.INSUFFICIENT_POINTS) != insufficient:
                raise ValueError("INSUFFICIENT_POINTS status disagrees with used counts")
        if np.any(self.valid_observation_counts > self.present_observation_counts):
            raise ValueError("valid observation count exceeds present count")
        if np.any(self.used_observation_counts > self.valid_observation_counts):
            raise ValueError("used observation count exceeds valid count")
        if np.any(
            self.used_observation_counts
            > self.spec.numeric_policy.sample_budget_per_batch
        ):
            raise ValueError("used observation count exceeds the declared sampling budget")
        if np.any(self.evaluation_counts > self.spec.numeric_policy.max_evaluations):
            raise ValueError("evaluation count exceeds the declared numeric budget")
        for field in ("solver_contract_id", "scipy_version", "initializer_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{field} must be canonical non-empty text")
        if self.solver_contract_id != model.solver_contract_id:
            raise ValueError("solver_contract_id disagrees with the model catalog")
        if self.initializer_id != model.initializer_id:
            raise ValueError("initializer_id disagrees with the model catalog")
        object.__setattr__(self, "fit_axis_specs", fit_axes)
        object.__setattr__(self, "batch_axis_specs", batch_axes)
        object.__setattr__(self, "coordinate_sources", sources)
        object.__setattr__(self, "parameter_definitions", parameters)
        object.__setattr__(self, "parameter_units", parameter_units)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "errors", errors)

    @property
    def digest(self) -> str:
        from .fit_codec import fit_result_batch_to_tree

        return canonical_digest(fit_result_batch_to_tree(self))


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
        "index"
        if source is FitCoordinateSource.LOGICAL_INDEX
        else (axis.unit or "1")
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
    array = np.asarray(values, dtype=np.dtype(dtype))
    return immutable_array(array, dtype=np.dtype(dtype), shape=shape)


def _validate_effective_lineage(spec: FitSpec, effective_fingerprint: str) -> None:
    expected = (
        spec.input_schema_fingerprint
        if spec.committed_transform is None
        else spec.committed_transform.output_schema_fingerprint
    )
    if effective_fingerprint != expected:
        raise ValueError("effective schema fingerprint disagrees with FitSpec authority")


def _validate_coordinate_sources(
    axes: tuple[AxisSpec, ...],
    sources: tuple[FitCoordinateSource, ...],
) -> None:
    for axis, source in zip(axes, sources):
        expected = _coordinate_source_for_axis(axis)
        if source is not expected:
            raise ValueError("coordinate source disagrees with authoritative AxisSpec")


def _coordinate_source_for_axis(axis: AxisSpec) -> FitCoordinateSource:
    if axis.coordinates is None:
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


def _resolve_fit_effective_schema(
    spec: FitSpec,
    expected_schema: DatasetSchema,
) -> TransformedSchema:
    if spec.committed_transform is not None:
        return resolve_transformed_schema(expected_schema, spec.committed_transform)
    cell_axes = (expected_schema.repeat_axis, *expected_schema.point_axes)
    point_layout = expected_schema.point_layout
    if point_layout.mode is AxisLayoutMode.RECT_C or (
        point_layout.mode is AxisLayoutMode.RECT_F
        and len(expected_schema.point_axes) <= 1
    ):
        cell_layout = AxisLayout.rect_c(tuple(axis.size for axis in cell_axes))
    else:
        cell_layout = AxisLayout.product(
            AxisLayout.rect_c((expected_schema.repeat_axis.size,)),
            point_layout,
        )
    validity_axes = (
        expected_schema.cell_schema.validity_contract.component_axis_ids
        if expected_schema.cell_schema.validity_contract.mode is ValidityMode.COMPONENTS
        else ()
    )
    return TransformedSchema(
        cell_axes,
        cell_layout,
        expected_schema.cell_schema.data_axes,
        validity_axes,
        expected_schema.cell_schema.dtype,
        expected_schema.cell_schema.value_unit,
    )


def _require_digest(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


__all__ = [
    "BoundFit",
    "FitBatchStatus",
    "FitCancelled",
    "FitCellProblem",
    "FitCoordinateSource",
    "FitDeadlineExceeded",
    "FitNumericPolicy",
    "FitParameterConstraint",
    "FitProblem",
    "FitResultBatch",
    "FitSpec",
]
