"""One lossless bridge from named-axis selections to Dataset snapshots.

The producer keeps one atomic :class:`zlc_data.OwnedSnapshot`.  Presentation
owners may expose an explicitly selected named axis as another snapshot, but
they must not flatten ``(R, P, *data_shape)``, infer an axis from rank, or
reimplement validity reshaping.  This module owns that mechanical projection
once for Camera ``frame_i`` views and Figure selector outputs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from zlc_data import (
    REPEAT,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DataTransformSpec,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    RowComponentValidity,
    Selection,
    ValidityContract,
    ValueSchema,
    apply_transform,
    commit_transform,
)

__all__ = ["materialize_dataset_selection"]


def _selected_dataset_schema(transformed) -> tuple[DatasetSchema, np.ndarray]:
    """Factor repeat from transformed rows and preserve physical point order."""

    cell_axes = tuple(transformed.schema.cell_axes)
    repeat_positions = tuple(
        index for index, axis in enumerate(cell_axes) if axis.role == REPEAT
    )
    if repeat_positions != (0,):
        raise ValueError(
            "dataset projection must preserve exactly one leading repeat axis"
        )
    repeat_axis = cell_axes[0]
    point_axes = cell_axes[1:]
    layout = transformed.schema.cell_layout
    row_by_multi = {
        layout.multi_index(row): row for row in range(layout.storage_size)
    }
    if len(row_by_multi) != layout.storage_size:
        raise RuntimeError("transformed cell layout contains duplicate rows")

    point_mapping: tuple[tuple[int, ...], ...] | None = None
    for repeat_index in range(repeat_axis.size):
        current = tuple(
            multi[1:]
            for multi in (
                layout.multi_index(row) for row in range(layout.storage_size)
            )
            if multi[0] == repeat_index
        )
        if point_mapping is None:
            point_mapping = current
        elif current != point_mapping:
            raise ValueError(
                "dataset projection produced repeat-dependent point membership"
            )
    if not point_mapping:
        raise ValueError("dataset projection produced no points")

    point_layout = PointLayout.from_mapping(
        tuple(axis.size for axis in point_axes),
        point_mapping,
    )
    if not isinstance(point_layout, PointLayout):
        raise RuntimeError("point layout factory returned the wrong layout type")
    order = np.fromiter(
        (
            row_by_multi[(repeat_index, *point_layout.multi_index(point_index))]
            for repeat_index in range(repeat_axis.size)
            for point_index in range(point_layout.storage_size)
        ),
        dtype=np.intp,
        count=repeat_axis.size * point_layout.storage_size,
    )
    validity_ids = tuple(transformed.schema.validity_axis_ids)
    validity_contract = (
        ValidityContract.components(*validity_ids)
        if validity_ids
        else ValidityContract.value()
    )
    schema = DatasetSchema(
        repeat_axis,
        point_axes,
        point_layout,
        ValueSchema(
            tuple(transformed.schema.data_axes),
            validity_contract,
            transformed.schema.dtype,
            transformed.schema.value_unit,
        ),
    )
    return schema, order


def materialize_dataset_selection(
    source: OwnedSnapshot,
    selection: Selection,
    *,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Materialize one explicit named-axis selection without reduction.

    ``reference_for`` remains with the presentation owner because it knows the
    stable identity of the derived signal.  This function owns only the generic
    shape/layout/validity mechanics and requires the derived revision to remain
    aligned with its atomic source revision.
    """

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("dataset projection source must be OwnedSnapshot")
    if not isinstance(selection, Selection):
        raise TypeError("dataset projection selection must be Selection")
    if not callable(reference_for):
        raise TypeError("dataset projection reference_for must be callable")

    transform = commit_transform(
        source.block.schema,
        DataTransformSpec((selection,)),
    )
    transformed = apply_transform(source, transform)
    output_schema, order = _selected_dataset_schema(transformed)
    output_ref = reference_for(output_schema)
    if not isinstance(output_ref, DatasetRevisionRef):
        raise TypeError("reference_for must return DatasetRevisionRef")
    if output_ref.block_id == source.ref.block_id:
        raise ValueError("a projected dataset cannot reuse its source BlockId")
    if output_ref.revision != source.ref.revision:
        raise ValueError("a projected dataset must retain its source revision")
    if output_ref.schema_fingerprint != output_schema.fingerprint:
        raise ValueError("projected reference schema differs from projected data")

    values = transformed.values[order].reshape(output_schema.physical_shape)
    transformed_validity = transformed.validity
    if isinstance(transformed_validity, RowComponentValidity):
        mask = transformed_validity.mask[order]
        if transformed_validity.axis_ids:
            validity = ComponentValidity(
                transformed_validity.axis_ids,
                mask.reshape(
                    output_schema.repeat_axis.size,
                    output_schema.point_layout.storage_size,
                    *mask.shape[1:],
                ),
            )
        else:
            validity = CellValidity(
                mask.reshape(
                    output_schema.repeat_axis.size,
                    output_schema.point_layout.storage_size,
                )
            )
    else:
        validity = transformed_validity
    block = DataBlock(
        output_ref.block_id,
        output_ref.revision,
        values,
        validity,
        output_schema,
    )
    return OwnedSnapshot(output_ref, block)
