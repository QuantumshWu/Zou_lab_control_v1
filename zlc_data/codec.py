"""Canonical codecs owned by the zlc_data bounded context."""

from __future__ import annotations

from typing import Any

import numpy as np

from zlc_storage.canonical import canonical_digest, decode, encode

from .axis import AxisId, AxisRoleId, AxisSpec, CoordinateFrameId
from .layout import PointLayout, PointLayoutMode
from .schema import DatasetSchema, ValueSchema
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    Invalid,
    Valid,
    ValidityContract,
    ValidityMode,
)
from .value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    Value,
    expand_dataset_validity,
    expand_value_validity,
)


AXIS_SCHEMA = "zlc_data.AxisSpec/v1"
VALUE_SCHEMA = "zlc_data.ValueSchema/v1"
DATASET_SCHEMA = "zlc_data.DatasetSchema/v1"
VALUE = "zlc_data.Value/v1"
DATA_BLOCK = "zlc_data.DataBlock/v1"


class TypedCodecError(ValueError):
    """Raised when primitive bytes do not have one unique typed representation."""


def axis_to_tree(axis: AxisSpec) -> dict[str, Any]:
    return {
        "schema": AXIS_SCHEMA,
        "axis_id": axis.axis_id.value,
        "name": axis.name,
        "role": axis.role.value,
        "size": axis.size,
        "coordinates": None if axis.coordinates is None else list(axis.coordinates),
        "unit": axis.unit,
        "coordinate_frame": None
        if axis.coordinate_frame is None
        else axis.coordinate_frame.value,
    }


def axis_from_tree(tree: Any) -> AxisSpec:
    data = _exact_map(
        tree,
        {
            "schema",
            "axis_id",
            "name",
            "role",
            "size",
            "coordinates",
            "unit",
            "coordinate_frame",
        },
        AXIS_SCHEMA,
    )
    coordinates = data["coordinates"]
    if coordinates is not None and not isinstance(coordinates, list):
        raise ValueError("AxisSpec coordinates must be a list or null")
    frame = data["coordinate_frame"]
    return AxisSpec(
        axis_id=AxisId(_text(data["axis_id"], "axis_id")),
        name=_text(data["name"], "name"),
        role=AxisRoleId(_text(data["role"], "role")),
        size=_integer(data["size"], "size"),
        coordinates=None if coordinates is None else tuple(coordinates),
        unit=None if data["unit"] is None else _text(data["unit"], "unit"),
        coordinate_frame=None if frame is None else CoordinateFrameId(_text(frame, "coordinate_frame")),
    )


def value_schema_to_tree(schema: ValueSchema) -> dict[str, Any]:
    return {
        "schema": VALUE_SCHEMA,
        "data_axes": [axis_to_tree(axis) for axis in schema.data_axes],
        "validity_contract": {
            "mode": schema.validity_contract.mode.value,
            "component_axis_ids": [
                axis_id.value for axis_id in schema.validity_contract.component_axis_ids
            ],
        },
        "dtype": schema.dtype.str,
        "value_unit": schema.value_unit,
    }


def value_schema_from_tree(tree: Any) -> ValueSchema:
    data = _exact_map(
        tree,
        {"schema", "data_axes", "validity_contract", "dtype", "value_unit"},
        VALUE_SCHEMA,
    )
    axes = data["data_axes"]
    if not isinstance(axes, list):
        raise ValueError("ValueSchema data_axes must be a list")
    validity = data["validity_contract"]
    if not isinstance(validity, dict) or set(validity) != {"mode", "component_axis_ids"}:
        raise ValueError("invalid ValueSchema validity_contract")
    mode = ValidityMode(_text(validity["mode"], "validity mode"))
    component_ids = validity["component_axis_ids"]
    if not isinstance(component_ids, list):
        raise ValueError("component_axis_ids must be a list")
    contract = ValidityContract(mode, tuple(AxisId(_text(item, "component axis")) for item in component_ids))
    unit = data["value_unit"]
    return ValueSchema(
        data_axes=tuple(axis_from_tree(axis) for axis in axes),
        validity_contract=contract,
        dtype=np.dtype(_text(data["dtype"], "dtype")),
        value_unit=None if unit is None else _text(unit, "value_unit"),
    )


def dataset_schema_to_tree(schema: DatasetSchema) -> dict[str, Any]:
    layout = schema.point_layout
    return {
        "schema": DATASET_SCHEMA,
        "repeat_axis": axis_to_tree(schema.repeat_axis),
        "point_axes": [axis_to_tree(axis) for axis in schema.point_axes],
        "point_layout": {
            "logical_shape": list(layout.logical_shape),
            "mode": layout.mode.value,
            "storage_size": layout.storage_size,
            "storage_to_multi": None
            if layout.storage_to_multi is None
            else [list(index) for index in layout.storage_to_multi],
        },
        "cell_schema": value_schema_to_tree(schema.cell_schema),
    }


def dataset_schema_from_tree(tree: Any) -> DatasetSchema:
    data = _exact_map(
        tree,
        {"schema", "repeat_axis", "point_axes", "point_layout", "cell_schema"},
        DATASET_SCHEMA,
    )
    point_axes = data["point_axes"]
    if not isinstance(point_axes, list):
        raise ValueError("DatasetSchema point_axes must be a list")
    layout_data = data["point_layout"]
    if not isinstance(layout_data, dict) or set(layout_data) != {
        "logical_shape",
        "mode",
        "storage_size",
        "storage_to_multi",
    }:
        raise ValueError("invalid DatasetSchema point_layout")
    shape_data = layout_data["logical_shape"]
    mapping_data = layout_data["storage_to_multi"]
    if not isinstance(shape_data, list):
        raise ValueError("point layout logical_shape must be a list")
    if mapping_data is not None and not isinstance(mapping_data, list):
        raise ValueError("point layout storage_to_multi must be a list or null")
    layout = PointLayout(
        logical_shape=tuple(_integer(size, "logical shape") for size in shape_data),
        mode=PointLayoutMode(_text(layout_data["mode"], "layout mode")),
        storage_size=_integer(layout_data["storage_size"], "storage_size"),
        storage_to_multi=None
        if mapping_data is None
        else tuple(tuple(_integer(index, "multi-index") for index in item) for item in mapping_data),
    )
    return DatasetSchema(
        repeat_axis=axis_from_tree(data["repeat_axis"]),
        point_axes=tuple(axis_from_tree(axis) for axis in point_axes),
        point_layout=layout,
        cell_schema=value_schema_from_tree(data["cell_schema"]),
    )


def value_schema_fingerprint(schema: ValueSchema) -> str:
    return canonical_digest(value_schema_to_tree(schema))


def dataset_schema_fingerprint(schema: DatasetSchema) -> str:
    return canonical_digest(dataset_schema_to_tree(schema))


def encode_dataset_schema(schema: DatasetSchema) -> bytes:
    return encode(dataset_schema_to_tree(schema))


def decode_dataset_schema(payload: bytes) -> DatasetSchema:
    schema = dataset_schema_from_tree(decode(payload))
    _require_typed_canonical(payload, encode_dataset_schema(schema), DATASET_SCHEMA)
    return schema


def value_to_tree(value: Value) -> dict[str, Any]:
    return {
        "schema": VALUE,
        "value_schema": value_schema_to_tree(value.schema),
        "values": _normalized_values(value.values, expand_value_validity(value.validity, value.schema)),
        "validity": validity_to_tree(value.validity),
    }


def value_from_tree(tree: Any) -> Value:
    data = _exact_map(tree, {"schema", "value_schema", "values", "validity"}, VALUE)
    schema = value_schema_from_tree(data["value_schema"])
    values = data["values"]
    if not isinstance(values, np.ndarray):
        raise ValueError("Value values must be an ndarray")
    return Value(values, validity_from_tree(data["validity"]), schema)


def data_block_to_tree(block: DataBlock) -> dict[str, Any]:
    return {
        "schema": DATA_BLOCK,
        "block_id": block.block_id.value,
        "revision": block.revision.value,
        "dataset_schema": dataset_schema_to_tree(block.schema),
        "values": _normalized_values(
            block.values,
            expand_dataset_validity(block.validity, block.schema),
        ),
        "validity": validity_to_tree(block.validity),
    }


def data_block_from_tree(tree: Any) -> DataBlock:
    data = _exact_map(
        tree,
        {"schema", "block_id", "revision", "dataset_schema", "values", "validity"},
        DATA_BLOCK,
    )
    schema = dataset_schema_from_tree(data["dataset_schema"])
    values = data["values"]
    if not isinstance(values, np.ndarray):
        raise ValueError("DataBlock values must be an ndarray")
    return DataBlock(
        block_id=BlockId(_text(data["block_id"], "block_id")),
        revision=DatasetRevision(_integer(data["revision"], "revision")),
        values=values,
        validity=validity_from_tree(data["validity"]),
        schema=schema,
    )


def encode_value(value: Value) -> bytes:
    return encode(value_to_tree(value))


def decode_value(payload: bytes) -> Value:
    value = value_from_tree(decode(payload))
    _require_typed_canonical(payload, encode_value(value), VALUE)
    return value


def encode_data_block(block: DataBlock) -> bytes:
    return encode(data_block_to_tree(block))


def decode_data_block(payload: bytes) -> DataBlock:
    block = data_block_from_tree(decode(payload))
    _require_typed_canonical(payload, encode_data_block(block), DATA_BLOCK)
    return block


def validity_to_tree(
    validity: Valid | Invalid | CellValidity | ComponentValidity,
) -> dict[str, Any]:
    if isinstance(validity, Valid):
        return {"kind": "valid"}
    if isinstance(validity, Invalid):
        return {"kind": "invalid"}
    if isinstance(validity, CellValidity):
        return {"kind": "cell", "mask": validity.mask}
    if isinstance(validity, ComponentValidity):
        return {
            "kind": "component",
            "axis_ids": [axis_id.value for axis_id in validity.axis_ids],
            "mask": validity.mask,
        }
    raise TypeError(f"unsupported validity type {type(validity).__name__}")


def validity_from_tree(tree: Any) -> Valid | Invalid | CellValidity | ComponentValidity:
    if not isinstance(tree, dict) or not isinstance(tree.get("kind"), str):
        raise ValueError("validity must be a tagged map")
    kind = tree["kind"]
    if kind == "valid" and set(tree) == {"kind"}:
        return VALID
    if kind == "invalid" and set(tree) == {"kind"}:
        return INVALID
    if kind == "cell" and set(tree) == {"kind", "mask"}:
        if not isinstance(tree["mask"], np.ndarray):
            raise ValueError("cell validity mask must be an ndarray")
        return CellValidity(tree["mask"])
    if kind == "component" and set(tree) == {"kind", "axis_ids", "mask"}:
        axis_ids = tree["axis_ids"]
        if not isinstance(axis_ids, list):
            raise ValueError("component validity axis_ids must be a list")
        if not isinstance(tree["mask"], np.ndarray):
            raise ValueError("component validity mask must be an ndarray")
        return ComponentValidity(
            tuple(AxisId(_text(axis_id, "validity axis id")) for axis_id in axis_ids),
            tree["mask"],
        )
    raise ValueError(f"invalid validity payload for kind {kind!r}")


def _normalized_values(values: np.ndarray, validity_mask: np.ndarray) -> np.ndarray:
    normalized = np.array(values, copy=True, order="C")
    normalized[~validity_mask] = 0
    return normalized


def _require_typed_canonical(got: bytes, expected: bytes, schema_id: str) -> None:
    if bytes(got) != expected:
        raise TypedCodecError(
            f"{schema_id} payload uses a non-canonical typed representation"
        )


def _exact_map(tree: Any, fields: set[str], schema_id: str) -> dict[str, Any]:
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError(f"{schema_id} must contain exactly {sorted(fields)}")
    if tree["schema"] != schema_id:
        raise ValueError(f"expected schema {schema_id!r}, got {tree['schema']!r}")
    return tree


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value
