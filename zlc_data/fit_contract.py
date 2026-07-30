"""Serializable fit requests/results and non-serializable bound problem values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from typing import Callable

import numpy as np

from zlc_storage.canonical import (
    canonical_text,
    finite_real,
    integer,
    positive_integer,
    positive_real,
)

from ._arrays import immutable_array, immutable_bool_array
from .axis import (
    AxisId,
    AxisSourceRef,
    AxisSpec,
    SCALAR,
    point_ordinal_axis,
)
from .layout import AxisLayout
from .schema import DatasetSchema, ResolvedPointRows, resolve_point_rows
from .transform import (
    CommittedTransform,
    _resolve_transform_state,
)
from .value import DatasetRevisionRef, OwnedSnapshot
from .fit_model import (
    FitModelDefinition,
    FitParameterDefinition,
    ParameterUnitRelation,
    fit_model_definition,
)


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
            object.__setattr__(
                self,
                field,
                finite_real(value, f"constraint {field}"),
            )
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
        object.__setattr__(
            self,
            "max_evaluations",
            positive_integer(self.max_evaluations, "max_evaluations"),
        )
        object.__setattr__(
            self,
            "covariance_rcond",
            positive_real(self.covariance_rcond, "covariance_rcond"),
        )


@dataclass(frozen=True)
class FitSpec:
    committed_transform: CommittedTransform
    independent_sources: tuple[AxisSourceRef, ...]
    batch_sources: tuple[AxisSourceRef, ...]
    model_id: str
    constraints: tuple[FitParameterConstraint, ...] = ()
    numeric_policy: FitNumericPolicy = FitNumericPolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.committed_transform, CommittedTransform):
            raise TypeError("committed_transform must be CommittedTransform")
        independent = tuple(self.independent_sources)
        batch = tuple(self.batch_sources)
        if not independent or any(
            not isinstance(source, AxisSourceRef) for source in independent
        ):
            raise ValueError("independent_sources must contain AxisSourceRef values")
        if any(not isinstance(source, AxisSourceRef) for source in batch):
            raise TypeError("batch_sources must contain AxisSourceRef values")
        if len(set(independent)) != len(independent) or len(set(batch)) != len(batch):
            raise ValueError("independent and batch sources must each be unique")
        if set(independent) & set(batch):
            raise ValueError("independent and batch sources cannot overlap")
        if any(source.kind == AxisSourceRef.POINT_ROWS for source in independent):
            raise ValueError("POINT_ROWS cannot be a Fit independent source")
        if any(source.kind == AxisSourceRef.POINT_ORDINAL for source in batch):
            raise ValueError("POINT_ORDINAL cannot be a Fit batch source")
        canonical_text(self.model_id, "model_id")
        model = fit_model_definition(self.model_id)
        if len(independent) != model.independent_arity:
            raise ValueError(
                f"model {model.model_id!r} requires "
                f"{model.independent_arity} independent sources"
            )
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
        object.__setattr__(self, "independent_sources", independent)
        object.__setattr__(self, "batch_sources", batch)
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
    effective_schema: DatasetSchema = field(init=False)
    fit_axis_specs: tuple[AxisSpec, ...] = field(init=False)
    batch_axis_specs: tuple[AxisSpec, ...] = field(init=False)
    point_groups: ResolvedPointRows = field(init=False)
    model: FitModelDefinition = field(init=False)
    _effective_point_groups: ResolvedPointRows = field(
        init=False,
        repr=False,
        compare=False,
    )
    _source_row_members: tuple[tuple[int, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if not isinstance(self.expected_schema, DatasetSchema):
            raise TypeError("expected_schema must be DatasetSchema")
        if (
            self.expected_schema.fingerprint
            != self.spec.committed_transform.source_schema_fingerprint
        ):
            raise ValueError("BoundFit expected schema disagrees with FitSpec")
        state = _resolve_transform_state(
            self.expected_schema,
            self.spec.committed_transform,
        )
        effective_schema = state.schema
        model = fit_model_definition(self.spec.model_id)
        if effective_schema.cell_schema.dtype.kind not in "biuf":
            raise TypeError("fit observations must use a real numeric dtype")
        _validate_fit_source_coverage(effective_schema, self.spec)
        point_batch_sources = tuple(
            source
            for source in self.spec.batch_sources
            if source.kind != AxisSourceRef.TENSOR
        )
        effective_groups = resolve_point_rows(
            effective_schema.point_table,
            effective_schema.grid_topology,
            group_sources=point_batch_sources,
        )
        if effective_groups.group_sources != point_batch_sources:
            raise ValueError("Fit point batch sources are not in canonical owner order")
        fit_axes = tuple(
            _independent_axis_spec(effective_schema, state.source_row_members, source)
            for source in self.spec.independent_sources
        )
        source_groups = _source_point_groups(
            effective_groups,
            state.source_row_members,
        )
        batch_axes = tuple(
            _batch_axis_spec(effective_schema, source_groups, source)
            for source in self.spec.batch_sources
        )
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
            units = tuple(_coordinate_unit(axis) for axis in fit_axes)
            if len(set(units)) != 1:
                raise ValueError("model fit axes require compatible coordinate units")
        if model.require_common_coordinate_frame and len(
            set(axis.coordinate_frame for axis in fit_axes)
        ) != 1:
            raise ValueError("model fit axes require the same coordinate frame")
        object.__setattr__(self, "effective_schema", effective_schema)
        object.__setattr__(self, "fit_axis_specs", fit_axes)
        object.__setattr__(self, "batch_axis_specs", batch_axes)
        object.__setattr__(
            self,
            "point_groups",
            source_groups,
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "_effective_point_groups", effective_groups)
        object.__setattr__(self, "_source_row_members", state.source_row_members)

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
    def parameter_units(self) -> tuple[str, ...]:
        return resolve_parameter_units(
            self.parameter_definitions,
            self.fit_axis_specs,
            self.effective_schema.cell_schema.value_unit,
        )


def _validate_fit_source_coverage(schema: DatasetSchema, spec: FitSpec) -> None:
    requested = spec.independent_sources + spec.batch_sources
    requested_tensor = tuple(
        source for source in requested if source.kind == AxisSourceRef.TENSOR
    )
    expected_tensor = (
        *(
            (AxisSourceRef.tensor(schema.repeat_axis.axis_id),)
            if schema.repeat_axis.size > 1
            else ()
        ),
        *(
            AxisSourceRef.tensor(axis.axis_id)
            for axis in schema.cell_schema.data_axes
            if axis.role != SCALAR and axis.size > 1
        ),
    )
    if len(requested_tensor) != len(expected_tensor) or set(requested_tensor) != set(
        expected_tensor
    ):
        raise ValueError(
            "Fit tensor sources must cover every effective information axis exactly"
        )

    point_independent = tuple(
        source
        for source in spec.independent_sources
        if source.kind != AxisSourceRef.TENSOR
    )
    point_batch = tuple(
        source for source in spec.batch_sources if source.kind != AxisSourceRef.TENSOR
    )
    point_sources = point_independent + point_batch
    if schema.point_table.row_count > 1 and not point_sources:
        raise ValueError(
            "Fit must explicitly use or batch a multi-row point domain"
        )
    if any(source.kind == AxisSourceRef.POINT_ROWS for source in point_batch) and point_independent:
        raise ValueError("POINT_ROWS batch cannot also expose a point independent source")
    if any(source.kind == AxisSourceRef.POINT_ROWS for source in point_batch) and len(
        point_batch
    ) != 1:
        raise ValueError("POINT_ROWS is the sole point-domain batch source")
    kinds = {source.kind for source in point_sources}
    if AxisSourceRef.GRID_DIMENSION in kinds and kinds != {
        AxisSourceRef.GRID_DIMENSION
    }:
        raise ValueError("raw point and GridTopology Fit sources cannot mix")
    if AxisSourceRef.POINT_ROWS in kinds and AxisSourceRef.POINT_COORDINATE in kinds:
        raise ValueError("POINT_ROWS and POINT_COORDINATE Fit sources cannot mix")


def _tensor_axis(schema: DatasetSchema, axis_id: AxisId | None) -> AxisSpec:
    if axis_id == schema.repeat_axis.axis_id:
        return schema.repeat_axis
    if axis_id is not None:
        for axis in schema.cell_schema.data_axes:
            if axis.axis_id == axis_id:
                return axis
    raise KeyError(f"tensor source {axis_id!r} is absent from effective Dataset")


def _point_column(schema: DatasetSchema, axis_id: AxisId | None):
    if axis_id is None:
        raise TypeError("point coordinate source requires an AxisId")
    return schema.point_table.column(axis_id)


def _independent_axis_spec(
    schema: DatasetSchema,
    source_row_members: tuple[tuple[int, ...], ...],
    source: AxisSourceRef,
) -> AxisSpec:
    if source.kind == AxisSourceRef.TENSOR:
        axis = _tensor_axis(schema, source.axis_id)
        if axis.role == SCALAR:
            raise ValueError("the scalar carrier cannot be a Fit independent source")
    elif source.kind == AxisSourceRef.POINT_ORDINAL:
        if any(len(members) != 1 for members in source_row_members):
            raise ValueError(
                "POINT_ORDINAL cannot identify an output row aggregated from source rows"
            )
        coordinates = tuple(members[0] for members in source_row_members)
        axis = point_ordinal_axis(len(coordinates), coordinates)
    elif source.kind == AxisSourceRef.POINT_COORDINATE:
        column = _point_column(schema, source.axis_id)
        if column.value_kind != column.NUMERIC:
            raise TypeError("Fit point-coordinate source must be numeric")
        axis = AxisSpec(
            column.coordinate_id,
            column.name,
            column.role,
            schema.point_table.row_count,
            column.values,
            column.unit,
            column.coordinate_frame,
        )
    elif source.kind == AxisSourceRef.GRID_DIMENSION:
        topology = schema.grid_topology
        if topology is None or source.axis_id not in topology.dimension_ids:
            raise KeyError("Fit grid source is absent from GridTopology")
        position = topology.dimension_ids.index(source.axis_id)
        column = _point_column(schema, source.axis_id)
        axis = AxisSpec(
            column.coordinate_id,
            column.name,
            column.role,
            len(topology.coordinate_domains[position]),
            topology.coordinate_domains[position],
            column.unit,
            column.coordinate_frame,
        )
    else:
        raise ValueError(f"{source.kind} is not a Fit independent source")
    _validate_numeric_axis(axis)
    return axis


def _batch_axis_spec(
    schema: DatasetSchema,
    point_groups: ResolvedPointRows,
    source: AxisSourceRef,
) -> AxisSpec:
    if source.kind == AxisSourceRef.TENSOR:
        axis = _tensor_axis(schema, source.axis_id)
        if axis.role == SCALAR:
            raise ValueError("the scalar carrier cannot be a Fit batch source")
        return axis
    if source.kind == AxisSourceRef.POINT_ROWS:
        coordinates = tuple(row[0] for row in point_groups.group_values)
        return point_ordinal_axis(len(coordinates), coordinates)
    try:
        position = point_groups.group_sources.index(source)
    except ValueError as exc:
        raise KeyError("point batch source is absent from resolved groups") from exc
    column = _point_column(schema, source.axis_id)
    if source.kind == AxisSourceRef.POINT_COORDINATE:
        domain = tuple(
            dict.fromkeys(row[position] for row in point_groups.group_values)
        )
    elif source.kind == AxisSourceRef.GRID_DIMENSION:
        topology = schema.grid_topology
        if topology is None or source.axis_id not in topology.dimension_ids:
            raise KeyError("Fit grid batch source is absent from GridTopology")
        domain = topology.coordinate_domains[topology.dimension_ids.index(source.axis_id)]
    else:
        raise ValueError(f"{source.kind} is not a Fit batch source")
    return AxisSpec(
        column.coordinate_id,
        column.name,
        column.role,
        len(domain),
        domain,
        column.unit,
        column.coordinate_frame,
    )


def _source_point_groups(
    effective: ResolvedPointRows,
    source_row_members: tuple[tuple[int, ...], ...],
) -> ResolvedPointRows:
    surviving = tuple(
        sorted(item for members in source_row_members for item in members)
    )
    mapped_groups = tuple(
        tuple(
            sorted(
                source_ordinal
                for effective_ordinal in group
                for source_ordinal in source_row_members[effective_ordinal]
            )
        )
        for group in effective.group_member_ordinals
    )
    grouped = {item for group in mapped_groups for item in group}
    addresses = effective.group_addresses
    values = effective.group_values
    if effective.group_sources == (AxisSourceRef.point_rows(),):
        if any(len(group) != 1 for group in mapped_groups):
            raise ValueError(
                "POINT_ROWS batch cannot identify an aggregated source row"
            )
        addresses = tuple((group[0],) for group in mapped_groups)
        values = addresses
    return ResolvedPointRows(
        surviving,
        effective.group_sources,
        addresses,
        values,
        mapped_groups,
        len(surviving) - len(grouped),
    )


def _validate_numeric_axis(axis: AxisSpec) -> None:
    if axis.coordinates is None:
        last = axis.index_origin + axis.size - 1
        if axis.size > 1 and last > _MAX_CONSECUTIVE_FLOAT64_INTEGER:
            raise ValueError(
                "implicit fit coordinates are not consecutively float64-representable"
            )
        values = (axis.index_origin, last)
    else:
        values = tuple(value for value in axis.coordinates if value is not None)
        if not values:
            raise ValueError("fit axis has no numeric coordinates")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise TypeError(
                f"declared coordinates for fit axis {axis.axis_id} must be numeric"
            )
    for value in values:
        try:
            converted = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("fit coordinate is not float64-representable") from exc
        if not math.isfinite(converted) or (
            isinstance(value, Integral) and int(converted) != int(value)
        ):
            raise ValueError("fit coordinate is not exactly float64-representable")


@dataclass(frozen=True)
class FitProblem:
    source_ref: DatasetRevisionRef
    spec: FitSpec
    fit_axis_specs: tuple[AxisSpec, ...]
    batch_axis_specs: tuple[AxisSpec, ...]
    point_groups: ResolvedPointRows
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
        if (
            self.source_ref.schema_fingerprint
            != self.spec.committed_transform.source_schema_fingerprint
        ):
            raise ValueError("fit problem source lineage disagrees with FitSpec")
        fit_axes = tuple(self.fit_axis_specs)
        batch_axes = tuple(self.batch_axis_specs)
        if any(not isinstance(axis, AxisSpec) for axis in fit_axes + batch_axes):
            raise TypeError("fit problem axes must contain AxisSpec values")
        if len(fit_axes) != len(self.spec.independent_sources):
            raise ValueError("fit problem axis arity disagrees with FitSpec")
        if len(batch_axes) != len(self.spec.batch_sources):
            raise ValueError("batch problem axis arity disagrees with FitSpec")
        if not isinstance(self.point_groups, ResolvedPointRows):
            raise TypeError("point_groups must be ResolvedPointRows")
        point_batch_sources = tuple(
            source
            for source in self.spec.batch_sources
            if source.kind != AxisSourceRef.TENSOR
        )
        if self.point_groups.group_sources != point_batch_sources:
            raise ValueError("fit problem point groups disagree with FitSpec")
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

@dataclass(frozen=True, eq=False)
class FitResultBatch:
    source_ref: DatasetRevisionRef
    spec: FitSpec
    fit_axis_specs: tuple[AxisSpec, ...]
    batch_axis_specs: tuple[AxisSpec, ...]
    point_groups: ResolvedPointRows
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
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("source_ref must be DatasetRevisionRef")
        if not isinstance(self.spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if (
            self.source_ref.schema_fingerprint
            != self.spec.committed_transform.source_schema_fingerprint
        ):
            raise ValueError("fit result source lineage disagrees with FitSpec")
        fit_axes = tuple(self.fit_axis_specs)
        batch_axes = tuple(self.batch_axis_specs)
        if any(not isinstance(axis, AxisSpec) for axis in fit_axes + batch_axes):
            raise TypeError("fit result axes must contain AxisSpec values")
        model = fit_model_definition(self.spec.model_id)
        parameters = model.parameters
        if len(fit_axes) != len(self.spec.independent_sources):
            raise ValueError("fit result axis arity disagrees with FitSpec")
        if len(batch_axes) != len(self.spec.batch_sources):
            raise ValueError("batch result axis arity disagrees with FitSpec")
        if not isinstance(self.point_groups, ResolvedPointRows):
            raise TypeError("point_groups must be ResolvedPointRows")
        point_batch_sources = tuple(
            source
            for source in self.spec.batch_sources
            if source.kind != AxisSourceRef.TENSOR
        )
        if self.point_groups.group_sources != point_batch_sources:
            raise ValueError("fit result point groups disagree with FitSpec")
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
    def parameter_definitions(self) -> tuple[FitParameterDefinition, ...]:
        return fit_model_definition(self.spec.model_id).parameters

    @property
    def parameter_units(self) -> tuple[str, ...]:
        return resolve_parameter_units(
            self.parameter_definitions,
            self.fit_axis_specs,
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

        normalized_index = integer(batch_storage_index, "batch_storage_index")
        assert normalized_index is not None
        if not 0 <= normalized_index < self.batch_layout.storage_size:
            raise IndexError("batch_storage_index is outside the result layout")
        index = normalized_index
        if self.statuses[index] is not FitBatchStatus.CONVERGED:
            raise ValueError("cannot evaluate a failed fit batch")
        from .fit_model import evaluate_fit_model

        model = fit_model_definition(self.spec.model_id)
        return evaluate_fit_model(model, coordinates, self.parameter_values[index])


def resolve_parameter_units(
    parameters: tuple[FitParameterDefinition, ...],
    fit_axes: tuple[AxisSpec, ...],
    value_unit: str | None,
) -> tuple[str, ...]:
    """Resolve model-relative units into self-contained artifact metadata."""

    axis_units = tuple(_coordinate_unit(axis) for axis in fit_axes)
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


def _coordinate_unit(axis: AxisSpec) -> str:
    if axis.unit is not None:
        return axis.unit
    return "index" if axis.coordinates is None else "1"


__all__ = [
    "BoundFit",
    "FitBatchStatus",
    "FitCancelled",
    "FitDeadlineExceeded",
    "FitNumericPolicy",
    "FitParameterConstraint",
    "FitResultBatch",
    "FitSpec",
]
