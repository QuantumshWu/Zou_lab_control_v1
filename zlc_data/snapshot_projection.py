"""Materialize derived Dataset snapshots without losing physical axes.

These are data-kernel operations.  Callers supply only the semantic identity of
each derived revision; selection/Fit layout, sparse point order, validity and
the physical ``(R, P, *data_shape)`` carrier remain owned here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._arrays import canonical_dtype
from .axis import REPEAT, AxisId, AxisSpec
from .fit_contract import FitBatchStatus, FitResultBatch
from .layout import PointLayout
from .schema import DatasetSchema, ValueSchema
from .selection import Selection
from .transform import DataTransformSpec, apply_transform, commit_transform
from .validity import (
    VALID,
    CellValidity,
    DatasetComponentValidity,
    RowComponentValidity,
    ValidityContract,
)
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
    """Accept one caller-owned identity without giving it carrier authority."""

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
    """Own the canonical physical ``R=1, P=1`` carrier boundary."""

    return DatasetSchema(
        AxisSpec(
            AxisId("derived.repeat"),
            "repeat",
            REPEAT,
            1,
            (0,),
        ),
        (),
        PointLayout.rect_c(()),
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
    """Materialize one typed scalar with its value-level validity.

    A scalar's canonical physical carrier remains ``(R=1, P=1, data=1)``;
    unlike a multi-component value it cannot use component validity.  Invalid
    floating payloads may retain a non-finite diagnostic value, while a value
    declared valid must be finite.
    """

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
    cell_schema = ValueSchema.scalar(dtype, unit)
    schema = _single_cell_schema(cell_schema)
    ref = _derived_reference(source_ref, schema, reference_for)
    block = DataBlock(
        ref.block_id,
        ref.revision,
        array.reshape(schema.physical_shape),
        CellValidity(np.asarray([[valid]], dtype=np.bool_)),
        schema,
    )
    return OwnedSnapshot(ref, block)


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
    """Materialize one Dataset cell with named component validity.

    Callers declare data axes and a compact mask over named axes.  This owner
    alone embeds both values and validity into the physical ``(R, P, ...)``
    carrier, preserving every declared trailing data axis.
    """

    axes = tuple(data_axes)
    if not axes or any(not isinstance(axis, AxisSpec) for axis in axes):
        raise TypeError(
            "component dataset data_axes must contain at least one AxisSpec"
        )
    validity_ids = tuple(validity_axis_ids)
    if not validity_ids or any(
        not isinstance(axis_id, AxisId) for axis_id in validity_ids
    ):
        raise TypeError(
            "component dataset validity_axis_ids must contain AxisId values"
        )
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
            f"component dataset shape {array.shape} does not match axes "
            f"{expected_values}"
        )
    mask = np.asarray(validity, dtype=np.bool_)
    expected_mask = tuple(axes[position].size for position in positions)
    if mask.shape != expected_mask:
        raise ValueError(
            f"component validity shape {mask.shape} does not match named axes "
            f"{expected_mask}"
        )

    schema = _single_cell_schema(
        ValueSchema(
            axes,
            ValidityContract.components(*validity_ids),
            array.dtype,
            unit,
        ),
    )
    ref = _derived_reference(source_ref, schema, reference_for)
    block = DataBlock(
        ref.block_id,
        ref.revision,
        array.reshape(schema.physical_shape),
        DatasetComponentValidity(
            validity_ids,
            mask.reshape(1, 1, *mask.shape),
        ),
        schema,
    )
    return OwnedSnapshot(ref, block)


def materialize_dataset_acceptance_mask(
    source: OwnedSnapshot,
    accepted: object,
    *,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """Preserve a Dataset carrier while intersecting one physical value mask.

    ``accepted`` is aligned with the source's complete physical shape.  The
    data owner combines it with source validity, chooses value- versus
    component-valid storage from the declared schema, and preserves all repeat,
    point and trailing data axes without flattening or reduction.
    """

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
    data_axes = tuple(source_schema.cell_schema.data_axes)
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
        source_schema.point_axes,
        source_schema.point_layout,
        cell_schema,
    )
    ref = _derived_reference(source.ref, schema, reference_for)
    block = DataBlock(
        ref.block_id,
        ref.revision,
        source.block.values,
        validity,
        schema,
    )
    return OwnedSnapshot(ref, block)


def _selected_dataset_schema(
    transformed,
) -> tuple[DatasetSchema, np.ndarray | None]:
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
    row_by_multi: dict[tuple[int, ...], int] = {}
    point_rows_by_repeat: list[list[tuple[int, ...]]] = [
        [] for _ in range(repeat_axis.size)
    ]
    for row in range(layout.storage_size):
        multi = layout.multi_index(row)
        if multi in row_by_multi:
            raise RuntimeError("transformed cell layout contains duplicate rows")
        row_by_multi[multi] = row
        point_rows_by_repeat[multi[0]].append(multi[1:])

    point_mapping = tuple(point_rows_by_repeat[0])
    for current_rows in point_rows_by_repeat[1:]:
        if tuple(current_rows) != point_mapping:
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
    # NumPy advanced indexing always copies the complete selected row payload.
    # The common single-cell/rectangular case is already in canonical physical
    # order, so represent that permutation as ``None`` and leave DataBlock as
    # the one immutable ownership copy.  Non-identity sparse/reordered layouts
    # still take the explicit indexed copy below.
    if all(int(row) == index for index, row in enumerate(order)):
        order = None
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

    ordered_values = (
        transformed.values
        if order is None
        else transformed.values[order]
    )
    values = ordered_values.reshape(output_schema.physical_shape)
    transformed_validity = transformed.validity
    if isinstance(transformed_validity, RowComponentValidity):
        mask = (
            transformed_validity.mask
            if order is None
            else transformed_validity.mask[order]
        )
        if transformed_validity.axis_ids:
            validity = DatasetComponentValidity(
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


def _fit_parameter_dataset_layout(
    result: FitResultBatch,
) -> tuple[AxisSpec, tuple[AxisSpec, ...], PointLayout, np.ndarray | None]:
    """Factor named Fit batch axes into Dataset repeat/point storage."""

    axes = tuple(result.batch_axis_specs)
    repeat_positions = tuple(
        index for index, axis in enumerate(axes) if axis.role == REPEAT
    )
    if len(repeat_positions) > 1:
        raise ValueError("fit result repeats the repeat axis")
    layout = result.batch_layout

    if repeat_positions:
        repeat_position = repeat_positions[0]
        repeat_axis = axes[repeat_position]
        point_axes = tuple(
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
                value
                for index, value in enumerate(multi)
                if index != repeat_position
            )
            key = (repeat_index, *point_multi)
            if key in rows:
                raise ValueError("fit batch layout contains a duplicate cell")
            rows[key] = storage_index
            by_repeat[repeat_index].add(point_multi)
        point_membership = by_repeat[0]
        if any(membership != point_membership for membership in by_repeat[1:]):
            raise ValueError(
                "fit batch has repeat-dependent sparse point membership, which "
                "cannot be represented as one DatasetSchema"
            )
        point_mapping = tuple(sorted(point_membership))
        point_layout = PointLayout.from_mapping(
            tuple(axis.size for axis in point_axes),
            point_mapping,
        )
        order = np.fromiter(
            (
                rows[(repeat_index, *point_layout.multi_index(point_index))]
                for repeat_index in range(repeat_axis.size)
                for point_index in range(point_layout.storage_size)
            ),
            dtype=np.intp,
            count=repeat_axis.size * point_layout.storage_size,
        )
        if all(int(row) == index for index, row in enumerate(order)):
            order = None
        return repeat_axis, point_axes, point_layout, order

    # Dataset's physical carrier always has one repeat axis.  When repeat was
    # not a Fit batch axis, introduce the one semantic carrier axis here rather
    # than letting each Figure or Workbench shell invent its own AxisId.
    point_axes = axes
    used_axis_ids = {axis.axis_id.value for axis in point_axes}
    repeat_axis_id = "fit-result-repeat"
    suffix = 2
    while repeat_axis_id in used_axis_ids:
        repeat_axis_id = f"fit-result-repeat-{suffix}"
        suffix += 1
    repeat_axis = AxisSpec(
        AxisId(repeat_axis_id),
        "repeat",
        REPEAT,
        1,
        (0,),
    )
    point_mapping = tuple(
        layout.multi_index(storage_index)
        for storage_index in range(layout.storage_size)
    )
    point_layout = PointLayout.from_mapping(
        tuple(axis.size for axis in point_axes),
        point_mapping,
    )
    return repeat_axis, point_axes, point_layout, None


def materialize_fit_parameter_snapshots(
    result: FitResultBatch,
    *,
    reference_for: Callable[[str, DatasetSchema], DatasetRevisionRef],
) -> dict[str, OwnedSnapshot]:
    """Materialize every Fit parameter as a typed scalar Dataset.

    The data owner preserves named batch axes, sparse membership and failed-cell
    validity.  The caller owns only each derived revision identity; it cannot
    alter the resulting schema or detach the output from the source revision.
    """

    if not isinstance(result, FitResultBatch):
        raise TypeError("fit result must be FitResultBatch")
    if not callable(reference_for):
        raise TypeError("fit parameter reference_for must be callable")

    repeat_axis, point_axes, point_layout, order = _fit_parameter_dataset_layout(
        result
    )
    validity_rows = np.fromiter(
        (status is FitBatchStatus.CONVERGED for status in result.statuses),
        dtype=np.bool_,
        count=len(result.statuses),
    )
    if order is not None:
        validity_rows = validity_rows[order]
    physical_shape = (repeat_axis.size, point_layout.storage_size)
    validity = CellValidity(validity_rows.reshape(physical_shape))

    output: dict[str, OwnedSnapshot] = {}
    identities: set[tuple[object, object]] = set()
    for parameter_index, (parameter, unit) in enumerate(
        zip(result.parameter_definitions, result.parameter_units, strict=True)
    ):
        parameter_name = parameter.name
        schema = DatasetSchema(
            repeat_axis,
            point_axes,
            point_layout,
            ValueSchema.scalar(np.dtype("<f8"), unit),
        )
        ref = reference_for(parameter_name, schema)
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
            parameter_values
            if order is None
            else parameter_values[order]
        ).reshape(schema.physical_shape)
        block = DataBlock(
            ref.block_id,
            ref.revision,
            values,
            validity,
            schema,
        )
        output[parameter_name] = OwnedSnapshot(ref, block)
    return output
