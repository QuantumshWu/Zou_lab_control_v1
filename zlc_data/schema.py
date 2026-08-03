"""Value and materialized dataset schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from zlc_storage.canonical import canonical_text, nonnegative_integer, positive_integer

from ._arrays import canonical_dtype
from .axis import (
    _POINT_ORDINAL_AXIS_ID,
    HISTOGRAM_BIN,
    READOUT_EVENT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    AxisId,
    AxisRoleId,
    AxisSourceRef,
    AxisSpec,
    CoordinateFrameId,
    CoordinateScalar,
    REPEAT,
    SCALAR,
    SCALAR_AXIS,
    canonical_coordinate_scalar,
)
from .validity import ValidityContract, ValidityMode


def _unique_axis_ids(axes: tuple[AxisSpec, ...], *, context: str) -> None:
    ids = tuple(axis.axis_id for axis in axes)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{context} axis ids must be unique")


def _ordered_subset(candidate: tuple[AxisId, ...], available: tuple[AxisId, ...]) -> bool:
    positions = []
    for axis_id in candidate:
        try:
            positions.append(available.index(axis_id))
        except ValueError:
            return False
    return positions == sorted(positions)


_POINT_ROLES = frozenset(
    {
        SCAN_POINT,
        READOUT_EVENT,
        SPATIAL_X,
        SPATIAL_Y,
        SPECTRAL,
        HISTOGRAM_BIN,
        SITE,
    }
)


@dataclass(frozen=True)
class PointColumn:
    """One correlated metadata column over the shared point-row domain."""

    coordinate_id: AxisId
    name: str
    role: AxisRoleId
    value_kind: str
    values: tuple[CoordinateScalar, ...]
    unit: str | None = None
    coordinate_frame: CoordinateFrameId | None = None

    NUMERIC: ClassVar[str] = "NUMERIC"
    TEXT: ClassVar[str] = "TEXT"

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_id, AxisId):
            raise TypeError("coordinate_id must be AxisId")
        canonical_text(self.name, "point column name")
        if not isinstance(self.role, AxisRoleId) or self.role not in _POINT_ROLES:
            raise ValueError("point column role is outside the point-domain role set")
        if self.value_kind not in {self.NUMERIC, self.TEXT}:
            raise ValueError("point column value_kind must be NUMERIC or TEXT")
        values = tuple(
            canonical_coordinate_scalar(value, "point coordinate")
            for value in self.values
        )
        if not values:
            raise ValueError("point column values must be non-empty")
        if self.value_kind == self.NUMERIC:
            if any(
                value is not None and not isinstance(value, (int, float))
                for value in values
            ):
                raise TypeError("NUMERIC point columns accept only numbers or None")
        else:
            if any(value is not None and not isinstance(value, str) for value in values):
                raise TypeError("TEXT point columns accept only text or None")
            if self.unit is not None:
                raise ValueError("TEXT point columns cannot declare a unit")
        object.__setattr__(self, "values", values)
        if self.unit is not None:
            canonical_text(self.unit, "point column unit")
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame, CoordinateFrameId
        ):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")


@dataclass(frozen=True)
class PointTable:
    """The authored ordered point rows shared by every repeat of a Dataset."""

    row_count: int
    columns: tuple[PointColumn, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "row_count",
            positive_integer(self.row_count, "point row_count"),
        )
        columns = tuple(self.columns)
        if any(not isinstance(column, PointColumn) for column in columns):
            raise TypeError("columns must contain PointColumn values")
        ids = tuple(column.coordinate_id for column in columns)
        if len(set(ids)) != len(ids):
            raise ValueError("point coordinate ids must be unique")
        if any(len(column.values) != self.row_count for column in columns):
            raise ValueError("every point column must contain exactly row_count values")
        object.__setattr__(self, "columns", columns)

    def column(self, coordinate_id: AxisId) -> PointColumn:
        if not isinstance(coordinate_id, AxisId):
            raise TypeError("coordinate_id must be AxisId")
        for column in self.columns:
            if column.coordinate_id == coordinate_id:
                return column
        raise KeyError(coordinate_id)


@dataclass(frozen=True)
class GridTopology:
    """Explicit producer-owned mapping from point ordinals to logical grid cells."""

    dimension_ids: tuple[AxisId, ...]
    coordinate_domains: tuple[tuple[CoordinateScalar, ...], ...]
    row_to_cell: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        dimensions = tuple(self.dimension_ids)
        if not dimensions or any(not isinstance(item, AxisId) for item in dimensions):
            raise TypeError("dimension_ids must contain at least one AxisId")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("grid dimension ids must be unique")
        domains = tuple(
            tuple(
                canonical_coordinate_scalar(value, "grid coordinate")
                for value in domain
            )
            for domain in self.coordinate_domains
        )
        if len(domains) != len(dimensions):
            raise ValueError("grid domains must match dimension_ids")
        if any(not domain for domain in domains):
            raise ValueError("grid coordinate domains must be non-empty")
        if any(any(value is None for value in domain) for domain in domains):
            raise ValueError("grid coordinate domains cannot contain missing values")
        if any(len(set(domain)) != len(domain) for domain in domains):
            raise ValueError("grid coordinate domains must contain unique values")
        mapping: list[tuple[int, ...]] = []
        for cell in self.row_to_cell:
            normalized = tuple(
                nonnegative_integer(index, "grid cell index") for index in cell
            )
            if len(normalized) != len(dimensions):
                raise ValueError("grid cell rank must match dimension_ids")
            if any(index >= len(domain) for index, domain in zip(normalized, domains)):
                raise ValueError("grid cell index is outside its coordinate domain")
            mapping.append(normalized)
        if not mapping:
            raise ValueError("row_to_cell must be non-empty")
        if len(set(mapping)) != len(mapping):
            raise ValueError("row_to_cell must be injective")
        object.__setattr__(self, "dimension_ids", dimensions)
        object.__setattr__(self, "coordinate_domains", domains)
        object.__setattr__(self, "row_to_cell", tuple(mapping))

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return tuple(len(domain) for domain in self.coordinate_domains)


def _validate_grid_topology(point_table: PointTable, topology: GridTopology) -> None:
    if len(topology.row_to_cell) != point_table.row_count:
        raise ValueError("GridTopology must map every PointTable row")
    for position, dimension_id in enumerate(topology.dimension_ids):
        column = point_table.column(dimension_id)
        domain = topology.coordinate_domains[position]
        if any(
            column.values[ordinal] != domain[cell[position]]
            for ordinal, cell in enumerate(topology.row_to_cell)
        ):
            raise ValueError(
                "GridTopology cell values must match their PointTable columns"
            )


@dataclass(frozen=True)
class ResolvedPointRows:
    """Canonical point-row grouping resolved once for View and Fit consumers."""

    surviving_ordinals: tuple[int, ...]
    group_sources: tuple[AxisSourceRef, ...]
    group_addresses: tuple[tuple[CoordinateScalar, ...], ...]
    group_values: tuple[tuple[CoordinateScalar, ...], ...]
    group_member_ordinals: tuple[tuple[int, ...], ...]
    dropped_count: int = 0

    def __post_init__(self) -> None:
        surviving = tuple(
            nonnegative_integer(item, "surviving point ordinal")
            for item in self.surviving_ordinals
        )
        if not surviving or tuple(sorted(set(surviving))) != surviving:
            raise ValueError("surviving_ordinals must be non-empty, unique, and increasing")
        sources = tuple(self.group_sources)
        if any(not isinstance(source, AxisSourceRef) for source in sources):
            raise TypeError("group_sources must contain AxisSourceRef values")
        addresses = tuple(
            tuple(canonical_coordinate_scalar(value, "point group address") for value in row)
            for row in self.group_addresses
        )
        values = tuple(
            tuple(canonical_coordinate_scalar(value, "point group value") for value in row)
            for row in self.group_values
        )
        members = tuple(
            tuple(nonnegative_integer(item, "point group member") for item in row)
            for row in self.group_member_ordinals
        )
        if not (len(addresses) == len(values) == len(members)) or not members:
            raise ValueError("point group fields must describe the same non-empty groups")
        if any(not row or tuple(sorted(set(row))) != row for row in members):
            raise ValueError("each point group must contain increasing unique ordinals")
        if any(len(row) != len(sources) for row in addresses + values):
            raise ValueError("point group address/value arity must match group_sources")
        flattened = tuple(item for row in members for item in row)
        if len(set(flattened)) != len(flattened) or any(
            item not in surviving for item in flattened
        ):
            raise ValueError("point groups must be disjoint subsets of surviving rows")
        dropped = nonnegative_integer(self.dropped_count, "dropped_count")
        if len(flattened) + dropped != len(surviving):
            raise ValueError("point groups and dropped_count must cover surviving rows")
        object.__setattr__(self, "surviving_ordinals", surviving)
        object.__setattr__(self, "group_sources", sources)
        object.__setattr__(self, "group_addresses", addresses)
        object.__setattr__(self, "group_values", values)
        object.__setattr__(self, "group_member_ordinals", members)
        object.__setattr__(self, "dropped_count", dropped)


def resolve_point_rows(
    point_table: PointTable,
    grid_topology: GridTopology | None,
    *,
    point_ordinals: tuple[int, ...] | None = None,
    group_sources: tuple[AxisSourceRef, ...] = (),
) -> ResolvedPointRows:
    """Resolve one exact row filter and grouping without Cartesian expansion."""

    if not isinstance(point_table, PointTable):
        raise TypeError("point_table must be PointTable")
    if grid_topology is not None and not isinstance(grid_topology, GridTopology):
        raise TypeError("grid_topology must be GridTopology or None")
    if grid_topology is not None:
        _validate_grid_topology(point_table, grid_topology)
    if point_ordinals is None:
        ordinals = tuple(range(point_table.row_count))
    else:
        ordinals = tuple(
            nonnegative_integer(item, "point ordinal") for item in point_ordinals
        )
        if not ordinals or tuple(sorted(set(ordinals))) != ordinals:
            raise ValueError("point_ordinals must be non-empty, unique, and increasing")
        if ordinals[-1] >= point_table.row_count:
            raise IndexError("point ordinal is outside PointTable")
    requested = tuple(group_sources)
    if any(not isinstance(source, AxisSourceRef) for source in requested):
        raise TypeError("group_sources must contain AxisSourceRef values")
    if len(set(requested)) != len(requested):
        raise ValueError("group_sources must be unique")
    if not requested:
        return ResolvedPointRows(ordinals, (), ((),), ((),), (ordinals,), 0)
    kinds = {source.kind for source in requested}
    if kinds == {AxisSourceRef.POINT_ROWS}:
        if len(requested) != 1:
            raise ValueError("PointRows grouping accepts exactly one source")
        groups = tuple((ordinal,) for ordinal in ordinals)
        return ResolvedPointRows(
            ordinals,
            requested,
            groups,
            groups,
            groups,
            0,
        )
    if kinds == {AxisSourceRef.POINT_COORDINATE}:
        requested_ids = {source.axis_id for source in requested}
        columns = tuple(
            column
            for column in point_table.columns
            if column.coordinate_id in requested_ids
        )
        if len(columns) != len(requested):
            raise KeyError("point coordinate group source is absent from PointTable")
        sources = tuple(AxisSourceRef.point_coordinate(column.coordinate_id) for column in columns)
        grouped: dict[tuple[CoordinateScalar, ...], list[int]] = {}
        dropped = 0
        for ordinal in ordinals:
            key = tuple(column.values[ordinal] for column in columns)
            if any(value is None for value in key):
                dropped += 1
                continue
            grouped.setdefault(key, []).append(ordinal)
        if not grouped:
            raise ValueError("point grouping dropped every selected row")
        keys = tuple(grouped)
        return ResolvedPointRows(
            ordinals,
            sources,
            keys,
            keys,
            tuple(tuple(grouped[key]) for key in keys),
            dropped,
        )
    if kinds == {AxisSourceRef.GRID_DIMENSION}:
        if grid_topology is None:
            raise ValueError("grid dimension grouping requires GridTopology")
        requested_ids = {source.axis_id for source in requested}
        positions = tuple(
            index
            for index, dimension_id in enumerate(grid_topology.dimension_ids)
            if dimension_id in requested_ids
        )
        if len(positions) != len(requested):
            raise KeyError("grid dimension group source is absent from GridTopology")
        sources = tuple(
            AxisSourceRef.grid_dimension(grid_topology.dimension_ids[index])
            for index in positions
        )
        grouped_cells: dict[tuple[int, ...], list[int]] = {}
        for ordinal in ordinals:
            cell = grid_topology.row_to_cell[ordinal]
            address = tuple(cell[index] for index in positions)
            grouped_cells.setdefault(address, []).append(ordinal)
        addresses = tuple(sorted(grouped_cells))
        values = tuple(
            tuple(
                grid_topology.coordinate_domains[position][cell_index]
                for position, cell_index in zip(positions, address)
            )
            for address in addresses
        )
        return ResolvedPointRows(
            ordinals,
            sources,
            addresses,
            values,
            tuple(tuple(grouped_cells[address]) for address in addresses),
            0,
        )
    raise ValueError("point row grouping cannot mix raw, topology, or tensor sources")


@dataclass(frozen=True)
class ValueSchema:
    data_axes: tuple[AxisSpec, ...]
    validity_contract: ValidityContract
    dtype: np.dtype
    value_unit: str | None = None
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        axes = tuple(self.data_axes)
        if any(not isinstance(axis, AxisSpec) for axis in axes):
            raise TypeError("data_axes must contain AxisSpec values")
        if not axes:
            raise ValueError(
                "data_axes must be non-empty; use ValueSchema.scalar() for a scalar value"
            )
        scalar_axes = tuple(axis for axis in axes if axis.role == SCALAR)
        if scalar_axes and axes != (SCALAR_AXIS,):
            raise ValueError(
                "the canonical scalar carrier must be the sole data axis"
            )
        _unique_axis_ids(axes, context="data")
        object.__setattr__(self, "data_axes", axes)
        if not isinstance(self.validity_contract, ValidityContract):
            raise TypeError("validity_contract must be ValidityContract")
        if scalar_axes and self.validity_contract.mode is not ValidityMode.VALUE:
            raise ValueError("the scalar carrier uses value-level validity")
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))
        if self.value_unit is not None:
            canonical_text(self.value_unit, "value_unit")
        available = tuple(axis.axis_id for axis in axes)
        declared = self.validity_contract.component_axis_ids
        if self.validity_contract.mode is ValidityMode.COMPONENTS and not _ordered_subset(
            declared, available
        ):
            raise ValueError("validity component axes must be an ordered subset of data axes")
        from .codec import value_schema_fingerprint

        object.__setattr__(self, "_fingerprint", value_schema_fingerprint(self))

    @property
    def data_shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.data_axes)

    @property
    def is_scalar(self) -> bool:
        return self.data_axes == (SCALAR_AXIS,)

    @classmethod
    def scalar(
        cls,
        dtype: np.dtype,
        value_unit: str | None = None,
    ) -> "ValueSchema":
        return cls(
            (SCALAR_AXIS,),
            ValidityContract.value(),
            dtype,
            value_unit,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def axis(self, axis_id: AxisId) -> AxisSpec:
        for axis in self.data_axes:
            if axis.axis_id == axis_id:
                return axis
        raise KeyError(axis_id)


@dataclass(frozen=True)
class DatasetSchema:
    repeat_axis: AxisSpec
    point_table: PointTable
    grid_topology: GridTopology | None
    cell_schema: ValueSchema
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.repeat_axis, AxisSpec) or self.repeat_axis.role != REPEAT:
            raise ValueError("repeat_axis must be an AxisSpec with role 'repeat'")
        if not isinstance(self.point_table, PointTable):
            raise TypeError("point_table must be PointTable")
        if self.grid_topology is not None and not isinstance(
            self.grid_topology, GridTopology
        ):
            raise TypeError("grid_topology must be GridTopology or None")
        if not isinstance(self.cell_schema, ValueSchema):
            raise TypeError("cell_schema must be ValueSchema")
        if any(axis.role == REPEAT for axis in self.cell_schema.data_axes):
            raise ValueError("REPEAT role belongs only to DatasetSchema.repeat_axis")
        all_ids = (
            self.repeat_axis.axis_id,
            *(column.coordinate_id for column in self.point_table.columns),
            *(axis.axis_id for axis in self.cell_schema.data_axes),
        )
        if _POINT_ORDINAL_AXIS_ID in all_ids:
            raise ValueError(
                "dataset axis ids cannot use the reserved point-ordinal identity"
            )
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("dataset tensor and point coordinate ids must be unique")
        if self.grid_topology is not None:
            _validate_grid_topology(self.point_table, self.grid_topology)
        from .codec import dataset_schema_fingerprint

        object.__setattr__(self, "_fingerprint", dataset_schema_fingerprint(self))

    @property
    def physical_shape(self) -> tuple[int, ...]:
        return (
            self.repeat_axis.size,
            self.point_table.row_count,
            *self.cell_schema.data_shape,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint
