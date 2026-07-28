"""Authoritative transforms over the fixed ``(R, P, *data_shape)`` carrier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from zlc_storage.canonical import positive_integer, sha256_text

from ._arrays import canonical_dtype, immutable_array
from .axis import HISTOGRAM_BIN, AxisId, AxisSourceRef, AxisSpec, SCALAR_AXIS
from .numeric import canonical_mean_dtype, canonical_sum_dtype, checked_numeric_sum
from .schema import DatasetSchema, GridTopology, PointColumn, PointTable, resolve_point_rows
from .selection import Selection, resolve_selection_indices
from .validity import (
    CellValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
    ValidityContract,
    ValidityMode,
)
from .value import (
    DatasetRevisionRef,
    OwnedSnapshot,
    compact_dataset_validity,
    expand_dataset_validity,
)


TRANSFORM_SPEC_SCHEMA = "zlc_data.DataTransformSpec"
COMMITTED_TRANSFORM_SCHEMA = "zlc_data.CommittedTransform"
HISTOGRAM_BIN_AXIS_ID = AxisId("zlc_data.histogram-bin")

DatasetValidity = Valid | Invalid | CellValidity | DatasetComponentValidity


class ReductionMethod(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"
    MIN = "MIN"
    MAX = "MAX"


class MissingPolicy(str, Enum):
    """How absent topology contributors affect a reduction output row."""

    REQUIRE_ALL = "REQUIRE_ALL"
    OMIT_MISSING = "OMIT_MISSING"


class ValidityPolicy(str, Enum):
    """How present-but-invalid components affect a reduction value."""

    REQUIRE_ALL = "REQUIRE_ALL"
    OMIT_INVALID = "OMIT_INVALID"
    MIN_COUNT = "MIN_COUNT"


@dataclass(frozen=True)
class ReductionSpec:
    sources: tuple[AxisSourceRef, ...]
    method: ReductionMethod
    missing_policy: MissingPolicy = MissingPolicy.REQUIRE_ALL
    validity_policy: ValidityPolicy = ValidityPolicy.REQUIRE_ALL
    minimum_valid_count: int | None = None

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if not sources or any(not isinstance(source, AxisSourceRef) for source in sources):
            raise ValueError("ReductionSpec sources must contain AxisSourceRef values")
        if len(set(sources)) != len(sources):
            raise ValueError("ReductionSpec sources must be unique")
        if not isinstance(self.method, ReductionMethod):
            raise TypeError("method must be ReductionMethod")
        if not isinstance(self.missing_policy, MissingPolicy):
            raise TypeError("missing_policy must be MissingPolicy")
        if not isinstance(self.validity_policy, ValidityPolicy):
            raise TypeError("validity_policy must be ValidityPolicy")
        if self.validity_policy is ValidityPolicy.MIN_COUNT:
            object.__setattr__(
                self,
                "minimum_valid_count",
                positive_integer(self.minimum_valid_count, "MIN_COUNT minimum_valid_count"),
            )
        elif self.minimum_valid_count is not None:
            raise ValueError("minimum_valid_count is only valid with MIN_COUNT")
        object.__setattr__(self, "sources", sources)


@dataclass(frozen=True)
class HistogramSpec:
    """Freeze exact sample sources and bin edges as analysis authority."""

    sources: tuple[AxisSourceRef, ...]
    bin_edges: tuple[float, ...]

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if not sources or any(not isinstance(source, AxisSourceRef) for source in sources):
            raise ValueError("HistogramSpec sources must contain AxisSourceRef values")
        if len(set(sources)) != len(sources):
            raise ValueError("HistogramSpec sources must be unique")
        edges = tuple(float(value) for value in self.bin_edges)
        if len(edges) < 2 or not all(math.isfinite(value) for value in edges):
            raise ValueError("HistogramSpec bin_edges must be finite boundaries")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("HistogramSpec bin_edges must be strictly increasing")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "bin_edges", edges)

    @property
    def bin_axis_id(self) -> AxisId:
        return HISTOGRAM_BIN_AXIS_ID


@dataclass(frozen=True)
class DataTransformSpec:
    operations: tuple[Selection | ReductionSpec | HistogramSpec, ...] = ()

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if any(
            not isinstance(operation, (Selection, ReductionSpec, HistogramSpec))
            for operation in operations
        ):
            raise TypeError("DataTransformSpec contains an unsupported operation")
        histogram_positions = tuple(
            index
            for index, operation in enumerate(operations)
            if isinstance(operation, HistogramSpec)
        )
        if len(histogram_positions) > 1:
            raise ValueError("DataTransformSpec permits at most one HistogramSpec")
        if histogram_positions and histogram_positions[0] != len(operations) - 1:
            raise ValueError("HistogramSpec must be the terminal transform operation")
        object.__setattr__(self, "operations", operations)


@dataclass(frozen=True)
class CommittedTransform:
    source_schema_fingerprint: str
    exact_point_ordinals: tuple[int, ...]
    spec: DataTransformSpec
    effective_output_schema: DatasetSchema

    def __post_init__(self) -> None:
        sha256_text(self.source_schema_fingerprint, "source_schema_fingerprint")
        ordinals = tuple(self.exact_point_ordinals)
        if not ordinals or tuple(sorted(set(ordinals))) != ordinals:
            raise ValueError("exact_point_ordinals must be non-empty, unique, and increasing")
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in ordinals):
            raise TypeError("exact_point_ordinals must contain nonnegative integers")
        if not isinstance(self.spec, DataTransformSpec):
            raise TypeError("spec must be DataTransformSpec")
        if not isinstance(self.effective_output_schema, DatasetSchema):
            raise TypeError("effective_output_schema must be DatasetSchema")
        object.__setattr__(self, "exact_point_ordinals", ordinals)

    @property
    def output_schema_fingerprint(self) -> str:
        return self.effective_output_schema.fingerprint


@dataclass(frozen=True, eq=False)
class TransformedData:
    source_ref: DatasetRevisionRef
    transform: CommittedTransform
    values: np.ndarray
    validity: DatasetValidity
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("source_ref must be DatasetRevisionRef")
        if not isinstance(self.transform, CommittedTransform):
            raise TypeError("transform must be CommittedTransform")
        if self.source_ref.schema_fingerprint != self.transform.source_schema_fingerprint:
            raise ValueError("source_ref schema differs from the committed transform source")
        expand_dataset_validity(self.validity, self.schema)
        object.__setattr__(
            self,
            "values",
            immutable_array(
                self.values,
                dtype=self.schema.cell_schema.dtype,
                shape=self.schema.physical_shape,
            ),
        )

    @property
    def schema(self) -> DatasetSchema:
        return self.transform.effective_output_schema

    def expanded_validity(self) -> np.ndarray:
        return expand_dataset_validity(self.validity, self.schema)


@dataclass
class _State:
    schema: DatasetSchema
    values: np.ndarray | None
    validity: DatasetValidity | None
    source_row_members: tuple[tuple[int, ...], ...]


def commit_transform(
    schema: DatasetSchema,
    authoritative_spec: DataTransformSpec,
    *,
    point_ordinals: tuple[int, ...] | None = None,
) -> CommittedTransform:
    """Validate and freeze one transform without touching source values."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(authoritative_spec, DataTransformSpec):
        raise TypeError("authoritative_spec must be DataTransformSpec")
    exact = resolve_point_rows(
        schema.point_table,
        schema.grid_topology,
        point_ordinals=point_ordinals,
    ).surviving_ordinals
    source = _source_state(schema, exact, values=None, validity=None)
    canonical_spec = _canonicalize_spec(source.schema, authoritative_spec)
    output = _run_operations(source, canonical_spec)
    return CommittedTransform(schema.fingerprint, exact, canonical_spec, output.schema)


def resolve_transformed_schema(
    schema: DatasetSchema,
    transform: CommittedTransform,
) -> DatasetSchema:
    """Recompile and verify one committed effective Dataset schema."""

    state = _resolve_transform_state(schema, transform)
    return state.schema


def apply_transform(
    snapshot: OwnedSnapshot,
    transform: CommittedTransform,
) -> TransformedData:
    """Execute one committed transform against an identified snapshot."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if not isinstance(transform, CommittedTransform):
        raise TypeError("transform must be CommittedTransform")
    block = snapshot.block
    if block.schema.fingerprint != transform.source_schema_fingerprint:
        raise ValueError("CommittedTransform source schema fingerprint is stale")
    state = _source_state(
        block.schema,
        transform.exact_point_ordinals,
        values=block.values,
        validity=block.validity,
    )
    state = _run_operations(state, transform.spec)
    if state.schema != transform.effective_output_schema:
        raise RuntimeError("transform execution disagrees with its committed output schema")
    assert state.values is not None and state.validity is not None
    return TransformedData(snapshot.ref, transform, state.values, state.validity)


def _resolve_transform_state(
    schema: DatasetSchema,
    transform: CommittedTransform,
) -> _State:
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(transform, CommittedTransform):
        raise TypeError("transform must be CommittedTransform")
    if schema.fingerprint != transform.source_schema_fingerprint:
        raise ValueError("CommittedTransform source schema fingerprint is stale")
    source = _source_state(
        schema,
        transform.exact_point_ordinals,
        values=None,
        validity=None,
    )
    canonical_spec = _canonicalize_spec(source.schema, transform.spec)
    if canonical_spec != transform.spec:
        raise ValueError("CommittedTransform operation sources are non-canonical")
    output = _run_operations(source, transform.spec)
    if output.schema != transform.effective_output_schema:
        raise ValueError("CommittedTransform effective schema is inconsistent")
    return output


def _source_state(
    schema: DatasetSchema,
    ordinals: tuple[int, ...],
    *,
    values: np.ndarray | None,
    validity: DatasetValidity | None,
) -> _State:
    selected_schema = _subset_point_schema(schema, ordinals)
    full = ordinals == tuple(range(schema.point_table.row_count))
    if values is None:
        selected_values = None
        selected_validity = None
    elif full:
        selected_values = values
        selected_validity = validity
    else:
        selected_values = np.take(values, ordinals, axis=1)
        assert validity is not None
        selected_validity = _select_validity(validity, axis=1, indices=ordinals)
    return _State(
        selected_schema,
        selected_values,
        selected_validity,
        tuple((ordinal,) for ordinal in ordinals),
    )


def _subset_point_schema(
    schema: DatasetSchema,
    ordinals: tuple[int, ...],
) -> DatasetSchema:
    if ordinals == tuple(range(schema.point_table.row_count)):
        return schema
    columns = tuple(
        PointColumn(
            column.coordinate_id,
            column.name,
            column.role,
            column.value_kind,
            tuple(column.values[index] for index in ordinals),
            column.unit,
            column.coordinate_frame,
        )
        for column in schema.point_table.columns
    )
    topology = schema.grid_topology
    selected_topology = (
        None
        if topology is None
        else GridTopology(
            topology.dimension_ids,
            topology.coordinate_domains,
            tuple(topology.row_to_cell[index] for index in ordinals),
        )
    )
    return DatasetSchema(
        schema.repeat_axis,
        PointTable(len(ordinals), columns),
        selected_topology,
        schema.cell_schema,
    )


def _canonicalize_spec(schema: DatasetSchema, spec: DataTransformSpec) -> DataTransformSpec:
    operations: list[Selection | ReductionSpec | HistogramSpec] = []
    for operation in spec.operations:
        if isinstance(operation, Selection):
            operations.append(operation)
            continue
        sources = tuple(sorted(operation.sources, key=lambda item: _source_order(schema, item)))
        if isinstance(operation, ReductionSpec):
            operations.append(
                ReductionSpec(
                    sources,
                    operation.method,
                    operation.missing_policy,
                    operation.validity_policy,
                    operation.minimum_valid_count,
                )
            )
        else:
            operations.append(HistogramSpec(sources, operation.bin_edges))
    return DataTransformSpec(tuple(operations))


def _source_order(schema: DatasetSchema, source: AxisSourceRef) -> tuple[int, int]:
    if source.kind == AxisSourceRef.TENSOR:
        if source.axis_id == schema.repeat_axis.axis_id:
            return (0, 0)
        data_ids = tuple(axis.axis_id for axis in schema.cell_schema.data_axes)
        try:
            return (2, data_ids.index(source.axis_id))
        except ValueError as exc:
            raise KeyError(f"tensor source {source.axis_id} is absent from Dataset") from exc
    if source.kind == AxisSourceRef.POINT_ROWS:
        return (1, 0)
    if source.kind == AxisSourceRef.GRID_DIMENSION:
        topology = schema.grid_topology
        if topology is None:
            raise ValueError("GRID_DIMENSION source requires GridTopology")
        try:
            return (1, topology.dimension_ids.index(source.axis_id))
        except ValueError as exc:
            raise KeyError(f"grid source {source.axis_id} is absent from GridTopology") from exc
    raise ValueError(f"{source.kind} is not a transform operation source")


def _run_operations(state: _State, spec: DataTransformSpec) -> _State:
    for operation in spec.operations:
        if isinstance(operation, Selection):
            state = _apply_selection(state, operation)
        elif isinstance(operation, ReductionSpec):
            state = _apply_reduction(state, operation)
        else:
            state = _apply_histogram(state, operation)
    return state


def _apply_selection(state: _State, selection: Selection) -> _State:
    for term in selection.terms:
        if term.axis_id == state.schema.repeat_axis.axis_id:
            state = _select_repeat(state, term)
            continue
        data_ids = tuple(axis.axis_id for axis in state.schema.cell_schema.data_axes)
        if term.axis_id not in data_ids:
            raise KeyError(
                "transform Selection is tensor-only; point queries must commit exact_point_ordinals"
            )
        state = _select_data_axis(state, data_ids.index(term.axis_id), term)
    return state


def _selected_axis(
    axis: AxisSpec,
    indices: range | tuple[int, ...],
) -> AxisSpec:
    if isinstance(indices, range) and indices.start == 0 and indices.stop == axis.size:
        return axis
    if axis.coordinates is None:
        if not isinstance(indices, range):
            raise RuntimeError("implicit-coordinate selection is not contiguous")
        coordinates = None
        index_origin = axis.index_origin + indices.start
    elif isinstance(indices, range):
        coordinates = axis.coordinates[indices.start : indices.stop]
        index_origin = 0
    else:
        coordinates = tuple(axis.coordinates[index] for index in indices)
        index_origin = 0
    return AxisSpec(
        axis.axis_id,
        axis.name,
        axis.role,
        len(indices),
        coordinates,
        axis.unit,
        axis.coordinate_frame,
        index_origin,
    )


def _reduced_axis(axis: AxisSpec) -> AxisSpec:
    return AxisSpec(
        axis.axis_id,
        axis.name,
        axis.role,
        1,
        None,
        axis.unit,
        axis.coordinate_frame,
        0,
    )


def _take_indices(
    array: np.ndarray,
    indices: range | tuple[int, ...],
    *,
    axis: int,
    drop: bool = False,
) -> np.ndarray:
    if drop:
        return np.take(array, indices[0], axis=axis)
    if isinstance(indices, range):
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(indices.start, indices.stop)
        return array[tuple(selection)]
    return np.take(array, indices, axis=axis)


def _select_validity(
    validity: DatasetValidity,
    *,
    axis: int,
    indices: range | tuple[int, ...],
    data_axis_id: AxisId | None = None,
    drop: bool = False,
) -> DatasetValidity:
    if isinstance(validity, (Valid, Invalid)):
        return validity
    if axis < 2:
        return type(validity)(
            validity.axis_ids,
            _take_indices(validity.mask, indices, axis=axis, drop=False),
        ) if isinstance(validity, DatasetComponentValidity) else CellValidity(
            _take_indices(validity.mask, indices, axis=axis, drop=False)
        )
    if not isinstance(validity, DatasetComponentValidity) or data_axis_id not in validity.axis_ids:
        return validity
    position = validity.axis_ids.index(data_axis_id)
    mask = _take_indices(validity.mask, indices, axis=2 + position, drop=drop)
    if not drop:
        return DatasetComponentValidity(validity.axis_ids, mask)
    axis_ids = validity.axis_ids[:position] + validity.axis_ids[position + 1 :]
    return DatasetComponentValidity(axis_ids, mask) if axis_ids else CellValidity(mask)


def _select_repeat(state: _State, term: object) -> _State:
    indices, _drop = resolve_selection_indices(state.schema.repeat_axis, term)
    if isinstance(indices, range) and indices.start == 0 and indices.stop == state.schema.repeat_axis.size:
        return state
    axis = _selected_axis(state.schema.repeat_axis, indices)
    schema = DatasetSchema(
        axis,
        state.schema.point_table,
        state.schema.grid_topology,
        state.schema.cell_schema,
    )
    if state.values is None:
        return _State(schema, None, None, state.source_row_members)
    assert state.validity is not None
    return _State(
        schema,
        _take_indices(state.values, indices, axis=0),
        _select_validity(state.validity, axis=0, indices=indices),
        state.source_row_members,
    )


def _value_schema(
    data_axes: tuple[AxisSpec, ...],
    validity_axis_ids: tuple[AxisId, ...],
    dtype: np.dtype,
    value_unit: str | None,
):
    from .schema import ValueSchema

    axes = data_axes or (SCALAR_AXIS,)
    validity_ids = tuple(
        axis.axis_id for axis in axes if axis.axis_id in validity_axis_ids
    )
    contract = (
        ValidityContract.components(*validity_ids)
        if validity_ids
        else ValidityContract.value()
    )
    return ValueSchema(axes, contract, dtype, value_unit)


def _select_data_axis(state: _State, position: int, term: object) -> _State:
    axes = list(state.schema.cell_schema.data_axes)
    axis = axes[position]
    indices, drop = resolve_selection_indices(axis, term)
    if not drop and isinstance(indices, range) and indices.start == 0 and indices.stop == axis.size:
        return state
    if drop:
        axes.pop(position)
    else:
        axes[position] = _selected_axis(axis, indices)
    contract = state.schema.cell_schema.validity_contract
    validity_ids = (
        tuple(item for item in contract.component_axis_ids if not (drop and item == axis.axis_id))
        if contract.mode is ValidityMode.COMPONENTS
        else ()
    )
    cell_schema = _value_schema(
        tuple(axes),
        validity_ids,
        state.schema.cell_schema.dtype,
        state.schema.cell_schema.value_unit,
    )
    schema = DatasetSchema(
        state.schema.repeat_axis,
        state.schema.point_table,
        state.schema.grid_topology,
        cell_schema,
    )
    if state.values is None:
        return _State(schema, None, None, state.source_row_members)
    values = _take_indices(state.values, indices, axis=2 + position, drop=drop)
    if drop and not axes:
        values = np.expand_dims(values, axis=-1)
    assert state.validity is not None
    validity = _select_validity(
        state.validity,
        axis=2 + position,
        indices=indices,
        data_axis_id=axis.axis_id,
        drop=drop,
    )
    return _State(schema, values, validity, state.source_row_members)


def _apply_reduction(state: _State, reduction: ReductionSpec) -> _State:
    repeat_source = AxisSourceRef.tensor(state.schema.repeat_axis.axis_id)
    reduce_repeat = repeat_source in reduction.sources
    data_ids = tuple(axis.axis_id for axis in state.schema.cell_schema.data_axes)
    reduced_data_ids = tuple(
        source.axis_id
        for source in reduction.sources
        if source.kind == AxisSourceRef.TENSOR and source != repeat_source
    )
    if any(axis_id not in data_ids for axis_id in reduced_data_ids):
        raise KeyError("reduction tensor source is absent from effective Dataset")
    point_rows = tuple(
        source for source in reduction.sources if source.kind == AxisSourceRef.POINT_ROWS
    )
    grid_sources = tuple(
        source for source in reduction.sources if source.kind == AxisSourceRef.GRID_DIMENSION
    )
    if point_rows and grid_sources:
        raise ValueError("POINT_ROWS and GRID_DIMENSION cannot share a reduction")
    if len(point_rows) > 1:
        raise ValueError("reduction accepts at most one POINT_ROWS source")
    allowed_count = int(reduce_repeat) + len(reduced_data_ids) + len(point_rows) + len(grid_sources)
    if allowed_count != len(reduction.sources):
        raise ValueError("reduction contains an unsupported source kind")

    groups: tuple[tuple[int, ...], ...] | None = None
    source_members = state.source_row_members
    point_table = state.schema.point_table
    topology = state.schema.grid_topology
    point_maximum = 1
    if point_rows:
        groups = (tuple(range(point_table.row_count)),)
        point_table = PointTable(1)
        topology = None
        point_maximum = state.schema.point_table.row_count
        source_members = (
            tuple(item for group in state.source_row_members for item in group),
        )
    elif grid_sources:
        point_table, topology, groups, source_members, point_maximum = _reduce_grid_domain(
            state,
            grid_sources,
            reduction.missing_policy,
        )

    data_positions = tuple(data_ids.index(axis_id) for axis_id in reduced_data_ids)
    maximum = (
        (state.schema.repeat_axis.size if reduce_repeat else 1)
        * point_maximum
        * math.prod(state.schema.cell_schema.data_axes[index].size for index in data_positions)
    )
    if (
        reduction.validity_policy is ValidityPolicy.MIN_COUNT
        and reduction.minimum_valid_count is not None
        and reduction.minimum_valid_count > maximum
    ):
        raise ValueError("minimum_valid_count exceeds the reduction maximum")
    if reduction.method in (ReductionMethod.MIN, ReductionMethod.MAX) and state.schema.cell_schema.dtype.kind == "c":
        raise TypeError("MIN/MAX reductions are undefined for complex values")

    remaining_axes = tuple(
        axis for axis in state.schema.cell_schema.data_axes if axis.axis_id not in reduced_data_ids
    )
    contract = state.schema.cell_schema.validity_contract
    validity_ids = (
        tuple(item for item in contract.component_axis_ids if item not in reduced_data_ids)
        if contract.mode is ValidityMode.COMPONENTS
        else ()
    )
    only_single_repeat = (
        reduction.sources == (repeat_source,)
        and state.schema.repeat_axis.size == 1
    )
    output_dtype = (
        state.schema.cell_schema.dtype
        if only_single_repeat
        else _reduction_dtype(state.schema.cell_schema.dtype, reduction.method)
    )
    cell_schema = _value_schema(
        remaining_axes,
        validity_ids,
        output_dtype,
        state.schema.cell_schema.value_unit,
    )
    schema = DatasetSchema(
        _reduced_axis(state.schema.repeat_axis) if reduce_repeat else state.schema.repeat_axis,
        point_table,
        topology,
        cell_schema,
    )
    if state.values is None:
        return _State(schema, None, None, source_members)
    assert state.validity is not None
    if only_single_repeat:
        return _State(
            schema,
            state.values,
            state.validity,
            source_members,
        )
    expanded = expand_dataset_validity(state.validity, state.schema)
    tensor_axes = tuple(
        ([0] if reduce_repeat else []) + [2 + position for position in data_positions]
    )
    if groups is None:
        values, validity = _reduce_arrays(
            state.values,
            expanded,
            tensor_axes,
            reduction.method,
            reduction.validity_policy,
            reduction.minimum_valid_count,
            output_dtype,
        )
        if reduce_repeat:
            values = np.expand_dims(values, axis=0)
            validity = np.expand_dims(validity, axis=0)
        if not remaining_axes:
            values = np.expand_dims(values, axis=-1)
            validity = np.expand_dims(validity, axis=-1)
    else:
        values, validity = _reduce_point_groups(
            state.values,
            expanded,
            groups,
            tensor_axes,
            reduce_repeat,
            bool(remaining_axes),
            reduction,
            output_dtype,
        )
    if values.shape != schema.physical_shape:
        raise RuntimeError("reduction output shape disagrees with effective Dataset schema")
    return _State(
        schema,
        values.astype(output_dtype, copy=False),
        compact_dataset_validity(validity, schema),
        source_members,
    )


def _reduce_grid_domain(
    state: _State,
    reduced_sources: tuple[AxisSourceRef, ...],
    missing_policy: MissingPolicy,
) -> tuple[
    PointTable,
    GridTopology | None,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    int,
]:
    topology = state.schema.grid_topology
    if topology is None:
        raise ValueError("GRID_DIMENSION reduction requires GridTopology")
    reduced_ids = {source.axis_id for source in reduced_sources}
    if any(axis_id not in topology.dimension_ids for axis_id in reduced_ids):
        raise KeyError("reduction grid source is absent from GridTopology")
    remaining_positions = tuple(
        index
        for index, axis_id in enumerate(topology.dimension_ids)
        if axis_id not in reduced_ids
    )
    remaining_sources = tuple(
        AxisSourceRef.grid_dimension(topology.dimension_ids[index])
        for index in remaining_positions
    )
    resolved = resolve_point_rows(
        state.schema.point_table,
        topology,
        group_sources=remaining_sources,
    )
    expected = math.prod(
        len(topology.coordinate_domains[index])
        for index, axis_id in enumerate(topology.dimension_ids)
        if axis_id in reduced_ids
    )
    retained = tuple(
        index
        for index, members in enumerate(resolved.group_member_ordinals)
        if missing_policy is MissingPolicy.OMIT_MISSING or len(members) == expected
    )
    if not retained:
        raise ValueError("Grid reduction has no complete output rows")
    groups = tuple(resolved.group_member_ordinals[index] for index in retained)
    addresses = tuple(resolved.group_addresses[index] for index in retained)
    columns: list[PointColumn] = []
    for column in state.schema.point_table.columns:
        if column.coordinate_id in reduced_ids:
            continue
        values: list[object] = []
        for members in groups:
            first = column.values[members[0]]
            if any(column.values[index] != first for index in members[1:]):
                break
            values.append(first)
        else:
            columns.append(
                PointColumn(
                    column.coordinate_id,
                    column.name,
                    column.role,
                    column.value_kind,
                    tuple(values),
                    column.unit,
                    column.coordinate_frame,
                )
            )
    remaining_ids = tuple(topology.dimension_ids[index] for index in remaining_positions)
    if any(axis_id not in {column.coordinate_id for column in columns} for axis_id in remaining_ids):
        raise RuntimeError("Grid reduction lost an unconsumed topology column")
    output_topology = (
        GridTopology(
            remaining_ids,
            tuple(topology.coordinate_domains[index] for index in remaining_positions),
            addresses,
        )
        if remaining_ids
        else None
    )
    output_members = tuple(
        tuple(
            source_ordinal
            for local_ordinal in members
            for source_ordinal in state.source_row_members[local_ordinal]
        )
        for members in groups
    )
    return PointTable(len(groups), tuple(columns)), output_topology, groups, output_members, expected


def _reduce_point_groups(
    values: np.ndarray,
    validity: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    tensor_axes: tuple[int, ...],
    repeat_reduced: bool,
    has_data_axes: bool,
    reduction: ReductionSpec,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    output_values: list[np.ndarray] = []
    output_validity: list[np.ndarray] = []
    axes = tuple(sorted((1, *tensor_axes)))
    for members in groups:
        indices = np.asarray(members, dtype=np.intp)
        reduced_values, reduced_validity = _reduce_arrays(
            values[:, indices],
            validity[:, indices],
            axes,
            reduction.method,
            reduction.validity_policy,
            reduction.minimum_valid_count,
            output_dtype,
        )
        if repeat_reduced:
            reduced_values = np.expand_dims(reduced_values, axis=0)
            reduced_validity = np.expand_dims(reduced_validity, axis=0)
        if not has_data_axes:
            reduced_values = np.expand_dims(reduced_values, axis=-1)
            reduced_validity = np.expand_dims(reduced_validity, axis=-1)
        output_values.append(reduced_values)
        output_validity.append(reduced_validity)
    return (
        np.stack(output_values, axis=1),
        np.stack(output_validity, axis=1),
    )


def _apply_histogram(state: _State, histogram: HistogramSpec) -> _State:
    repeat_source = AxisSourceRef.tensor(state.schema.repeat_axis.axis_id)
    sample_repeat = repeat_source in histogram.sources
    sample_points = AxisSourceRef.point_rows() in histogram.sources
    data_ids = tuple(axis.axis_id for axis in state.schema.cell_schema.data_axes)
    sample_data_ids = tuple(
        source.axis_id
        for source in histogram.sources
        if source.kind == AxisSourceRef.TENSOR and source != repeat_source
    )
    if any(axis_id not in data_ids for axis_id in sample_data_ids):
        raise KeyError("histogram tensor source is absent from effective Dataset")
    allowed = int(sample_repeat) + int(sample_points) + len(sample_data_ids)
    if allowed != len(histogram.sources):
        raise ValueError("Histogram supports only repeat/data TENSOR and POINT_ROWS sources")
    if histogram.bin_axis_id in data_ids:
        raise ValueError("histogram bin axis collides with an input axis")
    if state.schema.cell_schema.dtype.kind not in "biuf":
        raise TypeError("histogram projection requires real numeric or boolean values")

    # The canonical trailing scalar axis is a storage carrier, not an
    # independent physical dimension.  Histogramming scalar observations must
    # therefore consume that singleton automatically; otherwise a perfectly
    # ordinary scalar Dataset would incorrectly produce ``scalar x bin`` and
    # violate the scalar-carrier invariant.  Named size-one data axes remain
    # real dimensions and still require an explicit sample source.
    physical_sample_data_ids = sample_data_ids
    if state.schema.cell_schema.data_axes == (SCALAR_AXIS,):
        physical_sample_data_ids = (SCALAR_AXIS.axis_id,)
    remaining_axes = tuple(
        axis
        for axis in state.schema.cell_schema.data_axes
        if axis.axis_id not in physical_sample_data_ids
    )
    edges = np.asarray(histogram.bin_edges, dtype=np.dtype("<f8"))
    centers = tuple(((edges[:-1] + edges[1:]) * 0.5).tolist())
    bin_axis = AxisSpec(
        histogram.bin_axis_id,
        "value",
        HISTOGRAM_BIN,
        len(edges) - 1,
        centers,
        state.schema.cell_schema.value_unit,
    )
    validity_ids = tuple(axis.axis_id for axis in remaining_axes)
    cell_schema = _value_schema(
        (*remaining_axes, bin_axis),
        validity_ids,
        np.dtype("<i8"),
        "count",
    )
    point_table = PointTable(1) if sample_points else state.schema.point_table
    topology = None if sample_points else state.schema.grid_topology
    source_members = (
        (tuple(item for group in state.source_row_members for item in group),)
        if sample_points
        else state.source_row_members
    )
    schema = DatasetSchema(
        _reduced_axis(state.schema.repeat_axis) if sample_repeat else state.schema.repeat_axis,
        point_table,
        topology,
        cell_schema,
    )
    if state.values is None:
        return _State(schema, None, None, source_members)
    assert state.validity is not None
    expanded = expand_dataset_validity(state.validity, state.schema)
    sample_axes = tuple(
        sorted(
            ([0] if sample_repeat else [])
            + ([1] if sample_points else [])
            + [
                2 + data_ids.index(axis_id)
                for axis_id in physical_sample_data_ids
            ]
        )
    )
    remaining_physical_axes = tuple(
        index for index in range(state.values.ndim) if index not in sample_axes
    )
    permutation = remaining_physical_axes + sample_axes
    remaining_shape = tuple(state.values.shape[index] for index in remaining_physical_axes)
    sample_size = math.prod(state.values.shape[index] for index in sample_axes)
    rows = math.prod(remaining_shape)
    grouped_values = np.transpose(state.values, permutation).reshape(rows, sample_size)
    grouped_validity = np.transpose(expanded, permutation).reshape(rows, sample_size)
    bin_count = len(edges) - 1
    counts = np.zeros((rows, bin_count), dtype=np.dtype("<i8"))
    valid_rows = np.zeros(rows, dtype=np.bool_)
    for index in range(rows):
        samples = grouped_values[index, grouped_validity[index]]
        if not samples.size:
            continue
        if not bool(np.all(np.isfinite(samples))):
            raise ValueError("histogram projection received a valid non-finite value")
        bins = np.searchsorted(edges, samples, side="right") - 1
        bins[samples == edges[-1]] = bin_count - 1
        if not bool(np.all((bins >= 0) & (bins < bin_count))):
            raise ValueError("histogram bin edges do not retain every authoritative sample")
        counts[index] = np.bincount(bins, minlength=bin_count)
        valid_rows[index] = True
    values = counts.reshape((*remaining_shape, bin_count))
    validity = np.broadcast_to(
        valid_rows.reshape((*remaining_shape, 1)),
        values.shape,
    )
    if sample_repeat:
        values = np.expand_dims(values, axis=0)
        validity = np.expand_dims(validity, axis=0)
    if sample_points:
        values = np.expand_dims(values, axis=1)
        validity = np.expand_dims(validity, axis=1)
    if values.shape != schema.physical_shape:
        raise RuntimeError("histogram output shape disagrees with effective Dataset schema")
    return _State(
        schema,
        values,
        compact_dataset_validity(validity, schema),
        source_members,
    )


def _reduce_arrays(
    values: np.ndarray,
    validity: np.ndarray,
    axes: tuple[int, ...],
    method: ReductionMethod,
    policy: ValidityPolicy,
    minimum_valid_count: int | None,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    if not axes:
        raise RuntimeError("ReductionSpec did not resolve to a physical axis")
    if values.dtype.kind in "fc" and np.any(np.logical_and(validity, ~np.isfinite(values))):
        raise ValueError("reduction received a valid non-finite value")
    counts = np.sum(validity, axis=axes, dtype=np.int64)
    if policy is ValidityPolicy.REQUIRE_ALL:
        output_validity = counts == math.prod(values.shape[axis] for axis in axes)
    elif policy is ValidityPolicy.MIN_COUNT:
        assert minimum_valid_count is not None
        output_validity = counts >= minimum_valid_count
    else:
        output_validity = counts > 0
    if method in (ReductionMethod.MEAN, ReductionMethod.SUM):
        safe = np.where(validity, values, 0)
        if method is ReductionMethod.SUM and values.dtype.kind in "biu":
            summed = checked_numeric_sum(safe, axes, output_dtype=output_dtype)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                summed = np.sum(safe, axis=axes, dtype=output_dtype)
        if values.dtype.kind in "fc" and np.all(np.isfinite(safe)):
            if np.any((counts > 0) & ~np.isfinite(summed)):
                raise OverflowError("floating reduction overflowed its canonical output dtype")
        if method is ReductionMethod.MEAN:
            result = np.zeros(np.shape(summed), dtype=output_dtype)
            np.divide(summed, counts, out=result, where=counts > 0)
        else:
            result = np.asarray(summed, dtype=output_dtype)
    else:
        masked = np.ma.array(values, mask=~validity)
        reduced = masked.min(axis=axes) if method is ReductionMethod.MIN else masked.max(axis=axes)
        result = np.asarray(np.ma.filled(reduced, 0), dtype=output_dtype)
    result = np.where(output_validity, result, np.zeros((), dtype=output_dtype))
    return np.asarray(result, dtype=output_dtype), np.asarray(output_validity, dtype=bool)


def _reduction_dtype(dtype: np.dtype, method: ReductionMethod) -> np.dtype:
    dtype = canonical_dtype(dtype)
    if method is ReductionMethod.MEAN:
        return canonical_mean_dtype(dtype)
    if method is ReductionMethod.SUM:
        return canonical_sum_dtype(dtype)
    return dtype


__all__ = [
    "CommittedTransform",
    "DataTransformSpec",
    "HistogramSpec",
    "HISTOGRAM_BIN_AXIS_ID",
    "MissingPolicy",
    "ReductionMethod",
    "ReductionSpec",
    "TransformedData",
    "ValidityPolicy",
    "apply_transform",
    "commit_transform",
    "resolve_transformed_schema",
]
