"""Immutable event values and materialized dataset revisions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ._arrays import immutable_array
from .axis import AxisId
from .schema import DatasetSchema, ValueSchema
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    Invalid,
    Valid,
    ValidityMode,
)


@dataclass(frozen=True, order=True)
class BlockId:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value or self.value.strip() != self.value:
            raise ValueError("BlockId must be non-empty text without surrounding whitespace")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class DatasetRevision:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0:
            raise ValueError("DatasetRevision must be a non-negative integer")


@dataclass(frozen=True, order=True)
class StreamGenerationId:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value or self.value.strip() != self.value:
            raise ValueError(
                "StreamGenerationId must be non-empty text without surrounding whitespace"
            )


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
        if (
            not isinstance(self.schema_fingerprint, str)
            or len(self.schema_fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in self.schema_fingerprint)
        ):
            raise ValueError("schema_fingerprint must be a lowercase SHA-256 digest")
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


@dataclass(frozen=True)
class ValuePayloadContract:
    """Single owner for Value snapshot, schema validation, and retained-byte accounting."""

    schema: ValueSchema

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ValueSchema):
            raise TypeError("schema must be ValueSchema")

    @property
    def fingerprint(self) -> str:
        source = f"zlc.value-payload.v1:{self.schema.fingerprint}".encode("ascii")
        return hashlib.sha256(source).hexdigest()

    @property
    def max_retained_nbytes(self) -> int:
        value_bytes = int(np.prod(self.schema.data_shape, dtype=np.int64)) * self.schema.dtype.itemsize
        validity = self.schema.validity_contract
        if validity.mode is not ValidityMode.COMPONENTS:
            return value_bytes
        component_shape = tuple(self.schema.axis(axis_id).size for axis_id in validity.component_axis_ids)
        return value_bytes + int(np.prod(component_shape, dtype=np.int64))

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

    def retained_nbytes(self, payload: Value) -> int:
        self.validate(payload)
        validity_bytes = (
            payload.validity.mask.nbytes
            if isinstance(payload.validity, ComponentValidity)
            else 0
        )
        return int(payload.values.nbytes + validity_bytes)


@dataclass(frozen=True, eq=False)
class DataBlock:
    block_id: BlockId
    revision: DatasetRevision
    values: np.ndarray
    validity: Valid | Invalid | CellValidity | ComponentValidity
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


def expand_dataset_validity(
    validity: Valid | Invalid | CellValidity | ComponentValidity,
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
    broadcast_shape = [schema.repeat_axis.size, schema.point_layout.storage_size]
    broadcast_shape.extend([1] * len(schema.cell_schema.data_axes))
    for mask_index, axis_position in enumerate(positions):
        broadcast_shape[2 + axis_position] = validity.mask.shape[2 + mask_index]
    return np.broadcast_to(validity.mask.reshape(tuple(broadcast_shape)), schema.physical_shape)


def _axis_positions(axis_ids: tuple[AxisId, ...], schema: ValueSchema) -> tuple[int, ...]:
    available = tuple(axis.axis_id for axis in schema.data_axes)
    try:
        positions = tuple(available.index(axis_id) for axis_id in axis_ids)
    except ValueError as exc:
        raise ValueError("validity axis is absent from ValueSchema") from exc
    if positions != tuple(sorted(positions)):
        raise ValueError("validity axes must follow ValueSchema data-axis order")
    return positions


def _validate_component_axes(validity: ComponentValidity, schema: ValueSchema) -> tuple[int, ...]:
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
    validity: Valid | Invalid | CellValidity | ComponentValidity,
    schema: DatasetSchema,
) -> None:
    leading = (schema.repeat_axis.size, schema.point_layout.storage_size)
    if isinstance(validity, (Valid, Invalid)):
        return
    if isinstance(validity, CellValidity):
        if validity.mask.shape != leading:
            raise ValueError(
                f"cell validity shape {validity.mask.shape} does not match dataset cells {leading}"
            )
        return
    if not isinstance(validity, ComponentValidity):
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
    "expand_dataset_validity",
    "expand_value_validity",
]
