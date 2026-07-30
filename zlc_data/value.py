"""Immutable event values and materialized dataset revisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from zlc_storage.canonical import (
    canonical_text,
    integer,
    nonnegative_integer,
    sha256_text,
)

from ._arrays import canonical_dtype, immutable_array
from .axis import AxisId
from .schema import DatasetSchema, ValueSchema
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
    ValidityMode,
)


@dataclass(frozen=True, order=True)
class BlockId:
    value: str

    def __post_init__(self) -> None:
        canonical_text(self.value, "BlockId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class DatasetRevision:
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            nonnegative_integer(self.value, "DatasetRevision"),
        )


@dataclass(frozen=True, order=True)
class StreamGenerationId:
    value: str

    def __post_init__(self) -> None:
        canonical_text(self.value, "StreamGenerationId")


@dataclass(frozen=True)
class DatasetRevisionRef:
    block_id: BlockId
    stream_generation: StreamGenerationId
    schema_fingerprint: str
    revision: DatasetRevision

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.stream_generation, StreamGenerationId):
            raise TypeError("stream_generation must be StreamGenerationId")
        sha256_text(self.schema_fingerprint, "schema_fingerprint")
        if not isinstance(self.revision, DatasetRevision):
            raise TypeError("revision must be DatasetRevision")


@dataclass(frozen=True, eq=False)
class Value:
    values: np.ndarray
    validity: Valid | Invalid | ComponentValidity
    schema: ValueSchema
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ValueSchema):
            raise TypeError("schema must be ValueSchema")
        _validate_value_validity(self.validity, self.schema)
        array = immutable_array(
            self.values,
            dtype=self.schema.dtype,
            shape=self.schema.data_shape,
        )
        object.__setattr__(self, "values", array)


def canonical_value_array(
    values: np.ndarray,
    validity: Valid | Invalid | ComponentValidity,
    schema: ValueSchema,
) -> np.ndarray | None:
    """Return canonical values, or ``None`` for schema-level invalid values."""

    if not isinstance(schema, ValueSchema):
        raise TypeError("schema must be ValueSchema")
    array = np.asarray(values)
    if canonical_dtype(array.dtype) != schema.dtype:
        raise TypeError(
            f"values dtype {array.dtype} does not match schema dtype {schema.dtype}"
        )
    if array.shape != schema.data_shape:
        raise ValueError(
            f"values shape {array.shape} does not match expected {schema.data_shape}"
        )
    _validate_value_validity(validity, schema)
    if (
        isinstance(validity, Invalid)
        and schema.validity_contract.mode is not ValidityMode.COMPONENTS
    ):
        return None
    normalized: np.ndarray
    owns_normalized = False
    if isinstance(validity, (Invalid, ComponentValidity)):
        expanded = expand_value_validity(validity, schema)
        if not np.all(expanded):
            # Build the canonical array from zero rather than copying and then
            # indexing with ``~expanded``.  The latter materializes another
            # dense boolean frame for component validity and doubles that
            # normalization step's large-image scratch work.
            normalized = np.zeros(schema.data_shape, dtype=schema.dtype, order="C")
            np.copyto(normalized, array, where=expanded)
            owns_normalized = True
        else:
            normalized = np.ascontiguousarray(array, dtype=schema.dtype)
    else:
        normalized = np.ascontiguousarray(array, dtype=schema.dtype)
    if not owns_normalized:
        owns_normalized = not np.shares_memory(normalized, array)
    if normalized.dtype.kind in "fc" and np.any(np.isnan(normalized)):
        if not owns_normalized:
            normalized = np.array(array, copy=True, order="C")
            owns_normalized = True
        if normalized.dtype.kind == "f":
            canonical_nan = np.array(float("nan"), dtype=normalized.dtype)
            np.copyto(normalized, canonical_nan, where=np.isnan(normalized))
        else:
            component_dtype = np.dtype(
                "<f4" if normalized.dtype.itemsize == 8 else "<f8"
            )
            components = normalized.reshape(-1).view(component_dtype)
            canonical_nan = np.array(float("nan"), dtype=component_dtype)
            np.copyto(components, canonical_nan, where=np.isnan(components))
    return normalized


@dataclass(frozen=True)
class ValuePayloadContract:
    """Single owner for Value snapshot and schema validation."""

    schema: ValueSchema

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ValueSchema):
            raise TypeError("schema must be ValueSchema")

    def snapshot(self, payload: Value) -> Value:
        self.validate(payload)
        return payload

    def validate(self, payload: Value) -> None:
        if not isinstance(payload, Value):
            raise TypeError("ValuePayloadContract accepts only Value payloads")
        if payload.schema is not self.schema:
            raise TypeError(
                "Value payload must share the generation-owned ValueSchema instance"
            )

@dataclass(frozen=True, eq=False)
class DataBlock:
    block_id: BlockId
    revision: DatasetRevision
    values: np.ndarray
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity
    schema: DatasetSchema
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.revision, DatasetRevision):
            raise TypeError("revision must be DatasetRevision")
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        _validate_dataset_validity(self.validity, self.schema)
        array = immutable_array(
            self.values,
            dtype=self.schema.cell_schema.dtype,
            shape=self.schema.physical_shape,
        )
        object.__setattr__(self, "values", array)

    def ref(self, stream_generation: StreamGenerationId) -> DatasetRevisionRef:
        return DatasetRevisionRef(
            block_id=self.block_id,
            stream_generation=stream_generation,
            schema_fingerprint=self.schema.fingerprint,
            revision=self.revision,
        )


@dataclass(frozen=True)
class OwnedSnapshot:
    ref: DatasetRevisionRef
    block: DataBlock

    def __post_init__(self) -> None:
        if self.ref.block_id != self.block.block_id:
            raise ValueError("snapshot ref block_id does not match DataBlock")
        if self.ref.revision != self.block.revision:
            raise ValueError("snapshot ref revision does not match DataBlock")
        if self.ref.schema_fingerprint != self.block.schema.fingerprint:
            raise ValueError("snapshot ref schema fingerprint does not match DataBlock")


def dataset_cell_value(
    block: DataBlock,
    repeat_index: int,
    point_ordinal: int,
) -> Value:
    """Extract one exact physical Dataset cell as its declared ``Value``.

    This is the sole in-memory Dataset-to-Value boundary.  It preserves the
    cell schema and projects dataset validity without asking a presentation or
    application shell to interpret validity masks or trailing dimensions.
    """

    if not isinstance(block, DataBlock):
        raise TypeError("block must be DataBlock")
    normalized_repeat = integer(repeat_index, "repeat_index")
    normalized_point = integer(point_ordinal, "point_ordinal")
    assert normalized_repeat is not None and normalized_point is not None
    repeat_index = normalized_repeat
    point_ordinal = normalized_point
    for name, index, size in (
        ("repeat_index", repeat_index, block.schema.repeat_axis.size),
        (
            "point_ordinal",
            point_ordinal,
            block.schema.point_table.row_count,
        ),
    ):
        if not 0 <= index < size:
            raise IndexError(f"{name} is outside the Dataset")

    validity = block.validity
    if isinstance(validity, (Valid, Invalid)):
        cell_validity = validity
    elif isinstance(validity, CellValidity):
        cell_validity = (
            VALID
            if bool(validity.mask[repeat_index, point_ordinal])
            else INVALID
        )
    elif isinstance(validity, DatasetComponentValidity):
        cell_validity = ComponentValidity(
            validity.axis_ids,
            validity.mask[repeat_index, point_ordinal],
        )
    else:  # pragma: no cover - DataBlock validation closes the union
        raise TypeError("DataBlock validity has an unsupported type")
    return Value(
        block.values[repeat_index, point_ordinal],
        cell_validity,
        block.schema.cell_schema,
    )


def expand_value_validity(
    validity: Valid | Invalid | ComponentValidity,
    schema: ValueSchema,
) -> np.ndarray:
    """Return an AxisId-aware read-only validity view over ``schema.data_shape``."""

    _validate_value_validity(validity, schema)
    if isinstance(validity, (Valid, Invalid)):
        return np.broadcast_to(isinstance(validity, Valid), schema.data_shape)
    positions = _axis_positions(validity.axis_ids, schema)
    broadcast_shape = [1] * len(schema.data_axes)
    for mask_index, axis_position in enumerate(positions):
        broadcast_shape[axis_position] = validity.mask.shape[mask_index]
    return np.broadcast_to(validity.mask.reshape(tuple(broadcast_shape)), schema.data_shape)


def expand_component_validity(
    validity: Valid | Invalid | ComponentValidity,
    schema: ValueSchema,
) -> np.ndarray:
    """Return validity over the schema's declared component axes only.

    This is the canonical mask stored by materialization, independent of whether
    a producer supplied a scalar Valid/Invalid marker or a mask over a smaller
    named-axis subset.
    """

    _validate_value_validity(validity, schema)
    contract = schema.validity_contract
    if contract.mode is not ValidityMode.COMPONENTS:
        raise ValueError("component validity expansion requires a COMPONENTS contract")
    declared = contract.component_axis_ids
    declared_shape = tuple(schema.axis(axis_id).size for axis_id in declared)
    if isinstance(validity, (Valid, Invalid)):
        return np.broadcast_to(isinstance(validity, Valid), declared_shape)
    positions = tuple(declared.index(axis_id) for axis_id in validity.axis_ids)
    broadcast_shape = [1] * len(declared)
    for mask_index, declared_position in enumerate(positions):
        broadcast_shape[declared_position] = validity.mask.shape[mask_index]
    return np.broadcast_to(validity.mask.reshape(tuple(broadcast_shape)), declared_shape)


def expand_dataset_validity(
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    schema: DatasetSchema,
) -> np.ndarray:
    """Return validity aligned to ``(R, P, *data_shape)`` by named axes."""

    _validate_dataset_validity(validity, schema)
    if isinstance(validity, (Valid, Invalid)):
        return np.broadcast_to(isinstance(validity, Valid), schema.physical_shape)
    if isinstance(validity, CellValidity):
        shape = validity.mask.shape + (1,) * len(schema.cell_schema.data_axes)
        return np.broadcast_to(validity.mask.reshape(shape), schema.physical_shape)
    positions = _axis_positions(validity.axis_ids, schema.cell_schema)
    broadcast_shape = [schema.repeat_axis.size, schema.point_table.row_count]
    broadcast_shape.extend([1] * len(schema.cell_schema.data_axes))
    for mask_index, axis_position in enumerate(positions):
        broadcast_shape[2 + axis_position] = validity.mask.shape[2 + mask_index]
    return np.broadcast_to(validity.mask.reshape(tuple(broadcast_shape)), schema.physical_shape)


def compact_dataset_validity(
    mask: np.ndarray,
    schema: DatasetSchema,
) -> Valid | Invalid | CellValidity | DatasetComponentValidity:
    """Compact one full physical validity mask under its Dataset contract."""

    array = np.asarray(mask, dtype=np.bool_)
    if array.shape != schema.physical_shape:
        raise ValueError("dataset validity mask shape disagrees with schema")
    if bool(np.all(array)):
        return VALID
    if not bool(np.any(array)):
        return INVALID
    component_ids = (
        schema.cell_schema.validity_contract.component_axis_ids
        if schema.cell_schema.validity_contract.mode is ValidityMode.COMPONENTS
        else ()
    )
    compact = array
    for position in range(len(schema.cell_schema.data_axes) - 1, -1, -1):
        axis = schema.cell_schema.data_axes[position]
        if axis.axis_id in component_ids:
            continue
        array_axis = 2 + position
        first = np.take(compact, 0, axis=array_axis)
        if not np.array_equal(
            compact,
            np.broadcast_to(np.expand_dims(first, array_axis), compact.shape),
        ):
            raise RuntimeError(
                "validity varies along an axis absent from its declared contract"
            )
        compact = first
    if component_ids:
        return DatasetComponentValidity(component_ids, compact)
    return CellValidity(compact)


def _axis_positions(axis_ids: tuple[AxisId, ...], schema: ValueSchema) -> tuple[int, ...]:
    available = tuple(axis.axis_id for axis in schema.data_axes)
    try:
        positions = tuple(available.index(axis_id) for axis_id in axis_ids)
    except ValueError as exc:
        raise ValueError("validity axis is absent from ValueSchema") from exc
    if positions != tuple(sorted(positions)):
        raise ValueError("validity axes must follow ValueSchema data-axis order")
    return positions


def _validate_component_axes(
    validity: ComponentValidity | DatasetComponentValidity,
    schema: ValueSchema,
) -> tuple[int, ...]:
    if schema.validity_contract.mode is not ValidityMode.COMPONENTS:
        raise ValueError("component validity is forbidden by VALUE validity contract")
    declared = schema.validity_contract.component_axis_ids
    if any(axis_id not in declared for axis_id in validity.axis_ids):
        raise ValueError("component validity uses an axis absent from the schema contract")
    return _axis_positions(validity.axis_ids, schema)


def _validate_value_validity(
    validity: Valid | Invalid | ComponentValidity,
    schema: ValueSchema,
) -> None:
    if isinstance(validity, (Valid, Invalid)):
        return
    if not isinstance(validity, ComponentValidity):
        raise TypeError("Value validity must be Valid, Invalid, or ComponentValidity")
    positions = _validate_component_axes(validity, schema)
    expected = tuple(schema.data_axes[index].size for index in positions)
    if validity.mask.shape != expected:
        raise ValueError(
            f"component validity shape {validity.mask.shape} does not match named axes {expected}"
        )


def _validate_dataset_validity(
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    schema: DatasetSchema,
) -> None:
    leading = (schema.repeat_axis.size, schema.point_table.row_count)
    if isinstance(validity, (Valid, Invalid)):
        return
    if isinstance(validity, CellValidity):
        if validity.mask.shape != leading:
            raise ValueError(
                f"cell validity shape {validity.mask.shape} does not match dataset cells {leading}"
            )
        return
    if not isinstance(validity, DatasetComponentValidity):
        raise TypeError("DataBlock validity has an unsupported type")
    positions = _validate_component_axes(validity, schema.cell_schema)
    expected = leading + tuple(schema.cell_schema.data_axes[index].size for index in positions)
    if validity.mask.shape != expected:
        raise ValueError(
            f"component validity shape {validity.mask.shape} does not match named axes {expected}"
        )


__all__ = [
    "BlockId",
    "DataBlock",
    "DatasetRevision",
    "DatasetRevisionRef",
    "INVALID",
    "OwnedSnapshot",
    "StreamGenerationId",
    "VALID",
    "Value",
    "ValuePayloadContract",
    "canonical_value_array",
    "compact_dataset_validity",
    "dataset_cell_value",
    "expand_dataset_validity",
    "expand_value_validity",
]
