"""Materialize derived Dataset snapshots without losing physical axes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._arrays import canonical_dtype
from .axis import REPEAT, AxisId, AxisSpec
from .schema import DatasetSchema, PointTable, ValueSchema
from .selection import Selection
from .transform import DataTransformSpec, apply_transform, commit_transform
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
    ValidityContract,
)
from .value import (
    DataBlock,
    DatasetRevisionRef,
    OwnedSnapshot,
    Value,
    expand_dataset_validity,
)

__all__ = [
    "materialize_component_dataset",
    "materialize_dataset_acceptance_mask",
    "materialize_dataset_selection",
    "materialize_derived_dataset",
    "materialize_scalar_dataset",
    "materialize_value_dataset",
]


def _derived_reference(
    source_ref: DatasetRevisionRef,
    schema: DatasetSchema,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> DatasetRevisionRef:
    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("derived dataset source_ref must be DatasetRevisionRef")
    if not callable(reference_for):
        raise TypeError("derived dataset reference_for must be callable")
    ref = reference_for(schema)
    if not isinstance(ref, DatasetRevisionRef):
        raise TypeError("reference_for must return DatasetRevisionRef")
    if ref.block_id == source_ref.block_id:
        raise ValueError("a derived dataset cannot reuse its source BlockId")
    if ref.revision != source_ref.revision:
        raise ValueError("a derived dataset must retain its source revision")
    if ref.schema_fingerprint != schema.fingerprint:
        raise ValueError("derived reference schema differs from derived data")
    return ref


def _single_cell_schema(cell_schema: ValueSchema) -> DatasetSchema:
    return DatasetSchema(
        AxisSpec(AxisId("derived.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(1),
        None,
        cell_schema,
    )


def materialize_derived_dataset(
    source_ref: DatasetRevisionRef,
    values: object,
    *,
    schema: DatasetSchema,
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Materialize one typed derived Dataset without interpreting its domain.

    The caller owns the physical layout semantics.  This data-owned boundary
    only validates the canonical Dataset carrier and derives a distinct block
    reference at the exact source revision.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("derived dataset schema must be DatasetSchema")
    ref = _derived_reference(source_ref, schema, reference_for)
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            np.asarray(values),
            validity,
            schema,
        ),
    )


def materialize_scalar_dataset(
    source_ref: DatasetRevisionRef,
    value: object,
    *,
    valid: bool,
    unit: str | None,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Materialize one typed scalar in the canonical ``(1,1,1)`` carrier."""

    if type(valid) is not bool:
        raise TypeError("scalar dataset valid must be bool")
    raw = np.asarray(value)
    if raw.shape not in {(), (1,)}:
        raise ValueError("scalar dataset value must contain exactly one item")
    dtype = canonical_dtype(raw.dtype)
    if dtype.kind not in "biuf":
        raise TypeError("scalar dataset value must be real numeric or boolean")
    array = np.asarray(raw, dtype=dtype).reshape(1)
    if valid and dtype.kind == "f" and not bool(np.isfinite(array[0])):
        raise ValueError("valid scalar dataset value must be finite")
    schema = _single_cell_schema(ValueSchema.scalar(dtype, unit))
    ref = _derived_reference(source_ref, schema, reference_for)
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            array.reshape(schema.physical_shape),
            CellValidity(np.asarray([[valid]], dtype=np.bool_)),
            schema,
        ),
    )


def materialize_value_dataset(
    source_ref: DatasetRevisionRef,
    value: Value,
    *,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Place one typed :class:`Value` in the canonical single-cell carrier.

    Cross selectors return the value that is painted, not renderer coordinates.
    This data-owned boundary preserves its dtype, trailing axes and named
    component validity while adding only the required ``R=1, P=1`` carrier.
    """

    if not isinstance(value, Value):
        raise TypeError("single-cell dataset materialization requires Value")
    schema = _single_cell_schema(value.schema)
    ref = _derived_reference(source_ref, schema, reference_for)
    if isinstance(value.validity, Valid):
        validity = VALID
    elif isinstance(value.validity, Invalid):
        validity = INVALID
    elif isinstance(value.validity, ComponentValidity):
        validity = DatasetComponentValidity(
            value.validity.axis_ids,
            value.validity.mask.reshape(1, 1, *value.validity.mask.shape),
        )
    else:  # the closed Value validity vocabulary is enforced by Value itself
        raise TypeError("Value contains another validity representation")
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            value.values.reshape(schema.physical_shape),
            validity,
            schema,
        ),
    )


def materialize_component_dataset(
    source_ref: DatasetRevisionRef,
    values: object,
    *,
    data_axes: tuple[AxisSpec, ...],
    validity_axis_ids: tuple[AxisId, ...],
    validity: object,
    unit: str | None,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Materialize one Dataset cell with named component validity."""

    axes = tuple(data_axes)
    if not axes or any(not isinstance(axis, AxisSpec) for axis in axes):
        raise TypeError("component dataset data_axes must contain AxisSpec values")
    validity_ids = tuple(validity_axis_ids)
    if not validity_ids or any(
        not isinstance(axis_id, AxisId) for axis_id in validity_ids
    ):
        raise TypeError("validity_axis_ids must contain AxisId values")
    available = tuple(axis.axis_id for axis in axes)
    try:
        positions = tuple(available.index(axis_id) for axis_id in validity_ids)
    except ValueError as exc:
        raise ValueError("component validity axis is absent from data_axes") from exc
    if positions != tuple(sorted(positions)):
        raise ValueError("component validity axes must follow data-axis order")
    array = np.asarray(values)
    expected_values = tuple(axis.size for axis in axes)
    if array.shape != expected_values:
        raise ValueError(
            f"component dataset shape {array.shape} does not match axes {expected_values}"
        )
    mask = np.asarray(validity, dtype=np.bool_)
    expected_mask = tuple(axes[position].size for position in positions)
    if mask.shape != expected_mask:
        raise ValueError(
            f"component validity shape {mask.shape} does not match {expected_mask}"
        )
    schema = _single_cell_schema(
        ValueSchema(
            axes,
            ValidityContract.components(*validity_ids),
            array.dtype,
            unit,
        )
    )
    ref = _derived_reference(source_ref, schema, reference_for)
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            array.reshape(schema.physical_shape),
            DatasetComponentValidity(
                validity_ids,
                mask.reshape(1, 1, *mask.shape),
            ),
            schema,
        ),
    )


def materialize_dataset_acceptance_mask(
    source: OwnedSnapshot,
    accepted: object,
    *,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Preserve the Dataset carrier while intersecting one physical mask."""

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("masked dataset source must be OwnedSnapshot")
    source_schema = source.block.schema
    accepted_mask = np.asarray(accepted, dtype=np.bool_)
    if accepted_mask.shape != source_schema.physical_shape:
        raise ValueError(
            f"acceptance mask shape {accepted_mask.shape} does not match source "
            f"shape {source_schema.physical_shape}"
        )
    physical_validity = (
        np.asarray(expand_dataset_validity(source.block.validity, source_schema))
        & accepted_mask
    )
    data_axes = source_schema.cell_schema.data_axes
    if source_schema.cell_schema.is_scalar:
        cell_schema = ValueSchema.scalar(
            source_schema.cell_schema.dtype,
            source_schema.cell_schema.value_unit,
        )
        validity = CellValidity(physical_validity[..., 0])
    else:
        validity_ids = tuple(axis.axis_id for axis in data_axes)
        cell_schema = ValueSchema(
            data_axes,
            ValidityContract.components(*validity_ids),
            source_schema.cell_schema.dtype,
            source_schema.cell_schema.value_unit,
        )
        validity = DatasetComponentValidity(validity_ids, physical_validity)
    schema = DatasetSchema(
        source_schema.repeat_axis,
        source_schema.point_table,
        source_schema.grid_topology,
        cell_schema,
    )
    ref = _derived_reference(source.ref, schema, reference_for)
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            source.block.values,
            validity,
            schema,
        ),
    )


def materialize_dataset_selection(
    source: OwnedSnapshot,
    selection: Selection,
    *,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Materialize one explicit tensor-axis selection without reduction."""

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("dataset projection source must be OwnedSnapshot")
    if not isinstance(selection, Selection):
        raise TypeError("dataset projection selection must be Selection")
    transformed = apply_transform(
        source,
        commit_transform(source.block.schema, DataTransformSpec((selection,))),
    )
    output_ref = _derived_reference(source.ref, transformed.schema, reference_for)
    return OwnedSnapshot(
        output_ref,
        DataBlock(
            output_ref.block_id,
            output_ref.revision,
            transformed.values,
            transformed.validity,
            transformed.schema,
        ),
    )
