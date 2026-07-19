"""Value and materialized dataset schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from zlc_storage.canonical import canonical_text

from ._arrays import canonical_dtype
from .axis import AxisId, AxisSpec, REPEAT
from .layout import AxisLayout, PointLayout
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


def integer_retained_upper_bound_nbytes(value: int) -> int:
    """Bound one Python integer without allocating its decimal spelling."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    integer = int(value)
    # Cover CPython/PyPy bigint limbs and allocator slack without decimalizing.
    return 128 + 4 * max(1, (abs(integer).bit_length() + 7) // 8)


def axis_spec_retained_upper_bound_nbytes(axis: AxisSpec) -> int:
    """Bound one immutable AxisSpec owner graph without materializing text."""

    if not isinstance(axis, AxisSpec):
        raise TypeError("axis must be AxisSpec")
    text = (
        len(axis.axis_id.value)
        + len(axis.name)
        + len(axis.role.value)
        + (0 if axis.unit is None else len(axis.unit))
        + (
            0
            if axis.coordinate_frame is None
            else len(axis.coordinate_frame.value)
        )
    )
    coordinates = 0
    if axis.coordinates is not None:
        coordinates = 128 + sum(
            256
            + (
                4 * len(value)
                if isinstance(value, str)
                else (
                    integer_retained_upper_bound_nbytes(value)
                    if isinstance(value, int)
                    else 64
                )
            )
            for value in axis.coordinates
        )
    return int(
        2048
        + 4 * text
        + coordinates
        + integer_retained_upper_bound_nbytes(axis.size)
        + integer_retained_upper_bound_nbytes(axis.index_origin)
    )


def axis_layout_retained_upper_bound_nbytes(layout: AxisLayout) -> int:
    """Bound one AxisLayout, including its explicit reverse-index mapping."""

    if not isinstance(layout, AxisLayout):
        raise TypeError("layout must be AxisLayout")
    retained = 4096 + sum(
        256 + integer_retained_upper_bound_nbytes(size)
        for size in layout.logical_shape
    )
    retained += integer_retained_upper_bound_nbytes(layout.storage_size)
    if layout.storage_to_multi is not None:
        # The canonical tuple table and its reverse MappingProxy/dict coexist.
        # Every logical index is bounded by its declared shape; use that fact
        # instead of rescanning a potentially million-row immutable table on
        # every Workbench residual calculation.
        per_row = (
            1024
            + sum(
                256 + integer_retained_upper_bound_nbytes(size - 1)
                for size in layout.logical_shape
            )
            + integer_retained_upper_bound_nbytes(layout.storage_size - 1)
        )
        retained += layout.storage_size * per_row
    if layout.factors is not None:
        retained += sum(
            axis_layout_retained_upper_bound_nbytes(factor)
            for factor in layout.factors
        )
    return int(retained)


def dataset_schema_retained_upper_bound_nbytes(schema: "DatasetSchema") -> int:
    """Conservatively bound one decoded schema/layout/coordinate owner graph."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    axes = (
        schema.repeat_axis,
        *schema.point_axes,
        *schema.cell_schema.data_axes,
    )
    validity_text = sum(
        4 * len(axis_id.value) + 512
        for axis_id in schema.cell_schema.validity_contract.component_axis_ids
    )
    value_unit = (
        0
        if schema.cell_schema.value_unit is None
        else 4 * len(schema.cell_schema.value_unit)
    )
    return int(
        64 * 1024
        + sum(axis_spec_retained_upper_bound_nbytes(axis) for axis in axes)
        + axis_layout_retained_upper_bound_nbytes(schema.point_layout)
        + axis_layout_retained_upper_bound_nbytes(schema.cell_layout)
        + validity_text
        + value_unit
        + 4 * (
            len(schema.fingerprint) + len(schema.cell_schema.fingerprint)
        )
    )


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
        _unique_axis_ids(axes, context="data")
        object.__setattr__(self, "data_axes", axes)
        if not isinstance(self.validity_contract, ValidityContract):
            raise TypeError("validity_contract must be ValidityContract")
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
    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout
    cell_schema: ValueSchema
    _fingerprint: str = field(init=False, repr=False, compare=False)
    _cell_layout: AxisLayout = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.repeat_axis, AxisSpec) or self.repeat_axis.role != REPEAT:
            raise ValueError("repeat_axis must be an AxisSpec with role 'repeat'")
        points = tuple(self.point_axes)
        if any(not isinstance(axis, AxisSpec) for axis in points):
            raise TypeError("point_axes must contain AxisSpec values")
        object.__setattr__(self, "point_axes", points)
        if not isinstance(self.point_layout, PointLayout):
            raise TypeError("point_layout must be PointLayout")
        if not isinstance(self.cell_schema, ValueSchema):
            raise TypeError("cell_schema must be ValueSchema")
        if any(
            axis.role == REPEAT
            for axis in points + self.cell_schema.data_axes
        ):
            raise ValueError("REPEAT role belongs only to DatasetSchema.repeat_axis")
        logical_shape = tuple(axis.size for axis in points)
        if self.point_layout.logical_shape != logical_shape:
            raise ValueError(
                f"point layout shape {self.point_layout.logical_shape} does not match axes {logical_shape}"
            )
        all_axes = (self.repeat_axis,) + points + self.cell_schema.data_axes
        _unique_axis_ids(all_axes, context="dataset")
        object.__setattr__(
            self,
            "_cell_layout",
            AxisLayout.product(
                AxisLayout.rect_c((self.repeat_axis.size,)),
                self.point_layout,
            ),
        )
        from .codec import dataset_schema_fingerprint

        object.__setattr__(self, "_fingerprint", dataset_schema_fingerprint(self))

    @property
    def physical_shape(self) -> tuple[int, ...]:
        return (
            self.repeat_axis.size,
            self.point_layout.storage_size,
            *self.cell_schema.data_shape,
        )

    @property
    def cell_layout(self) -> AxisLayout:
        """Canonical physical-row mapping over repeat and logical point axes."""

        return self._cell_layout

    @property
    def fingerprint(self) -> str:
        return self._fingerprint
