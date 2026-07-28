"""Materialize derived Dataset snapshots without losing physical axes."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real

import numpy as np

from ._arrays import canonical_dtype
from .axis import REPEAT, AxisId, AxisSpec
from .fit_contract import FitBatchStatus, FitResultBatch
from .schema import DatasetSchema, GridTopology, PointColumn, PointTable, ValueSchema
from .selection import Selection
from .transform import DataTransformSpec, apply_transform, commit_transform
from .validity import CellValidity, DatasetComponentValidity, ValidityContract
from .value import (
    DataBlock,
    DatasetRevisionRef,
    OwnedSnapshot,
    expand_dataset_validity,
)

__all__ = [
    "materialize_component_dataset",
    "materialize_dataset_acceptance_mask",
    "materialize_dataset_selection",
    "materialize_fit_parameter_snapshots",
    "materialize_scalar_dataset",
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


def _fit_parameter_dataset_layout(
    result: FitResultBatch,
) -> tuple[AxisSpec, PointTable, GridTopology | None, np.ndarray | None]:
    """Project a sparse Fit batch layout into one exact R/P row table."""

    axes = result.batch_axis_specs
    repeat_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == REPEAT
    )
    if len(repeat_positions) > 1:
        raise ValueError("fit result repeats the repeat axis")
    layout = result.batch_layout
    if repeat_positions:
        repeat_position = repeat_positions[0]
        repeat_axis = axes[repeat_position]
        row_axes = tuple(
            axis for index, axis in enumerate(axes) if index != repeat_position
        )
        rows: dict[tuple[int, ...], int] = {}
        by_repeat: list[set[tuple[int, ...]]] = [
            set() for _ in range(repeat_axis.size)
        ]
        for storage_index in range(layout.storage_size):
            multi = layout.multi_index(storage_index)
            repeat_index = multi[repeat_position]
            point_multi = tuple(
                value for index, value in enumerate(multi) if index != repeat_position
            )
            key = (repeat_index, *point_multi)
            if key in rows:
                raise ValueError("fit batch layout contains a duplicate cell")
            rows[key] = storage_index
            by_repeat[repeat_index].add(point_multi)
        point_membership = by_repeat[0]
        if any(membership != point_membership for membership in by_repeat[1:]):
            raise ValueError(
                "fit batch has repeat-dependent point membership"
            )
        point_mapping = tuple(sorted(point_membership))
        order = np.fromiter(
            (
                rows[(repeat_index, *point_multi)]
                for repeat_index in range(repeat_axis.size)
                for point_multi in point_mapping
            ),
            dtype=np.intp,
            count=repeat_axis.size * len(point_mapping),
        )
        if np.array_equal(order, np.arange(order.size, dtype=np.intp)):
            order = None
    else:
        row_axes = axes
        used_ids = {axis.axis_id.value for axis in row_axes}
        repeat_id = "fit-result-repeat"
        suffix = 2
        while repeat_id in used_ids:
            repeat_id = f"fit-result-repeat-{suffix}"
            suffix += 1
        repeat_axis = AxisSpec(AxisId(repeat_id), "repeat", REPEAT, 1, (0,))
        point_mapping = tuple(
            layout.multi_index(storage_index)
            for storage_index in range(layout.storage_size)
        )
        order = None
    if not point_mapping:
        raise ValueError("Fit result contains no materializable batch cells")
    columns = tuple(
        _point_column_for_axis(
            axis,
            tuple(axis.coordinate_at(multi[position]) for multi in point_mapping),
        )
        for position, axis in enumerate(row_axes)
    )
    point_table = PointTable(len(point_mapping), columns)
    topology = _fit_batch_topology(row_axes, point_mapping)
    return repeat_axis, point_table, topology, order


def _point_column_for_axis(
    axis: AxisSpec,
    values: tuple[object, ...],
) -> PointColumn:
    value_kind = (
        PointColumn.NUMERIC
        if all(
            value is None
            or (not isinstance(value, bool) and isinstance(value, Real))
            for value in values
        )
        else PointColumn.TEXT
    )
    return PointColumn(
        axis.axis_id,
        axis.name,
        axis.role,
        value_kind,
        values,
        axis.unit,
        axis.coordinate_frame,
    )


def _fit_batch_topology(
    axes: tuple[AxisSpec, ...],
    mapping: tuple[tuple[int, ...], ...],
) -> GridTopology | None:
    if not axes:
        return None
    domains = tuple(
        tuple(axis.coordinate_at(index) for index in range(axis.size))
        for axis in axes
    )
    if any(
        any(value is None for value in domain) or len(set(domain)) != len(domain)
        for domain in domains
    ):
        return None
    return GridTopology(
        tuple(axis.axis_id for axis in axes),
        domains,
        mapping,
    )


def materialize_fit_parameter_snapshots(
    result: FitResultBatch,
    *,
    reference_for: Callable[[str, DatasetSchema], DatasetRevisionRef],
) -> dict[str, OwnedSnapshot]:
    """Materialize every Fit parameter as a scalar Dataset over its batches."""

    if not isinstance(result, FitResultBatch):
        raise TypeError("fit result must be FitResultBatch")
    if not callable(reference_for):
        raise TypeError("fit parameter reference_for must be callable")
    repeat_axis, point_table, topology, order = _fit_parameter_dataset_layout(result)
    validity_rows = np.fromiter(
        (status is FitBatchStatus.CONVERGED for status in result.statuses),
        dtype=np.bool_,
        count=len(result.statuses),
    )
    if order is not None:
        validity_rows = validity_rows[order]
    physical_shape = (repeat_axis.size, point_table.row_count)
    validity = CellValidity(validity_rows.reshape(physical_shape))
    output: dict[str, OwnedSnapshot] = {}
    identities: set[tuple[object, object]] = set()
    for parameter_index, (parameter, unit) in enumerate(
        zip(result.parameter_definitions, result.parameter_units, strict=True)
    ):
        schema = DatasetSchema(
            repeat_axis,
            point_table,
            topology,
            ValueSchema.scalar(np.dtype("<f8"), unit),
        )
        ref = reference_for(parameter.name, schema)
        if not isinstance(ref, DatasetRevisionRef):
            raise TypeError("reference_for must return DatasetRevisionRef")
        if ref.block_id == result.source_ref.block_id:
            raise ValueError("a Fit parameter cannot reuse its source BlockId")
        if ref.revision != result.source_ref.revision:
            raise ValueError("a Fit parameter must retain its source revision")
        if ref.schema_fingerprint != schema.fingerprint:
            raise ValueError("Fit parameter reference schema differs from data")
        identity = (ref.block_id, ref.stream_generation)
        if identity in identities:
            raise ValueError("Fit parameters must have distinct dataset identities")
        identities.add(identity)
        parameter_values = np.asarray(
            result.parameter_values[:, parameter_index],
            dtype="<f8",
        )
        values = (
            parameter_values if order is None else parameter_values[order]
        ).reshape(*physical_shape, 1)
        output[parameter.name] = OwnedSnapshot(
            ref,
            DataBlock(
                ref.block_id,
                ref.revision,
                values,
                validity,
                schema,
            ),
        )
    return output
