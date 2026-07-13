"""Value and materialized dataset schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from zlc_storage.canonical import canonical_text

from ._arrays import canonical_dtype
from .axis import AxisId, AxisSpec, REPEAT
from .layout import PointLayout
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
        logical_shape = tuple(axis.size for axis in points)
        if self.point_layout.logical_shape != logical_shape:
            raise ValueError(
                f"point layout shape {self.point_layout.logical_shape} does not match axes {logical_shape}"
            )
        all_axes = (self.repeat_axis,) + points + self.cell_schema.data_axes
        _unique_axis_ids(all_axes, context="dataset")
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
    def fingerprint(self) -> str:
        return self._fingerprint
