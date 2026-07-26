"""Canonical codecs owned by the zlc_data bounded context."""

from __future__ import annotations

from typing import Any

import numpy as np

from zlc_storage.canonical import (
    canonical_digest,
    canonical_text as _text,
    encode as _encode,
    exact_mapping as _exact_map,
)

from .axis import AxisId, AxisRoleId, AxisSpec, CoordinateFrameId
from .layout import AxisLayout, AxisLayoutMode, PointLayout
from .schema import DatasetSchema, ValueSchema
from .value import BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
    ValidityContract,
    ValidityMode,
)


AXIS_SCHEMA = "zlc_data.AxisSpec"
VALUE_SCHEMA = "zlc_data.ValueSchema"
DATASET_SCHEMA = "zlc_data.DatasetSchema"
DATASET_REVISION_REF_SCHEMA = "zlc_data.DatasetRevisionRef"


class TypedCodecError(ValueError):
    """Raised when primitive bytes do not have one unique typed representation."""


def dataset_revision_ref_to_tree(value: DatasetRevisionRef) -> dict[str, Any]:
    """Project one revision identity through its data-owner field model."""

    if not isinstance(value, DatasetRevisionRef):
        raise TypeError("value must be DatasetRevisionRef")
    return {
        "schema": DATASET_REVISION_REF_SCHEMA,
        "block_id": value.block_id.value,
        "stream_generation": value.stream_generation.value,
        "schema_fingerprint": value.schema_fingerprint,
        "revision": value.revision.value,
    }


def dataset_revision_ref_from_tree(tree: Any) -> DatasetRevisionRef:
    """Decode only the current exact DatasetRevisionRef representation."""

    data = _exact_map(
        tree,
        {
            "schema",
            "block_id",
            "stream_generation",
            "schema_fingerprint",
            "revision",
        },
        DATASET_REVISION_REF_SCHEMA,
    )
    value = DatasetRevisionRef(
        block_id=BlockId(data["block_id"]),
        stream_generation=StreamGenerationId(data["stream_generation"]),
        schema_fingerprint=data["schema_fingerprint"],
        revision=DatasetRevision(data["revision"]),
    )
    if _encode(dataset_revision_ref_to_tree(value)) != _encode(tree):
        raise ValueError("DatasetRevisionRef tree is typed but non-canonical")
    return value


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
        "index_origin": axis.index_origin,
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
            "index_origin",
        },
        AXIS_SCHEMA,
    )
    coordinates = data["coordinates"]
    if coordinates is not None and not isinstance(coordinates, list):
        raise ValueError("AxisSpec coordinates must be a list or null")
    frame = data["coordinate_frame"]
    axis = AxisSpec(
        axis_id=AxisId(data["axis_id"]),
        name=data["name"],
        role=AxisRoleId(data["role"]),
        size=data["size"],
        coordinates=None if coordinates is None else tuple(coordinates),
        unit=data["unit"],
        coordinate_frame=None if frame is None else CoordinateFrameId(frame),
        index_origin=data["index_origin"],
    )
    if _encode(axis_to_tree(axis)) != _encode(tree):
        raise ValueError("AxisSpec tree is typed but non-canonical")
    return axis


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
    mode = ValidityMode(validity["mode"])
    component_ids = validity["component_axis_ids"]
    if not isinstance(component_ids, list):
        raise ValueError("component_axis_ids must be a list")
    contract = ValidityContract(mode, tuple(AxisId(item) for item in component_ids))
    unit = data["value_unit"]
    schema = ValueSchema(
        data_axes=tuple(axis_from_tree(axis) for axis in axes),
        validity_contract=contract,
        dtype=np.dtype(_text(data["dtype"], "dtype")),
        value_unit=unit,
    )
    if _encode(value_schema_to_tree(schema)) != _encode(tree):
        raise ValueError("ValueSchema tree is typed but non-canonical")
    return schema


def dataset_schema_to_tree(schema: DatasetSchema) -> dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA,
        "repeat_axis": axis_to_tree(schema.repeat_axis),
        "point_axes": [axis_to_tree(axis) for axis in schema.point_axes],
        "point_layout": point_layout_to_tree(schema.point_layout),
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
    layout = point_layout_from_tree(data["point_layout"])
    schema = DatasetSchema(
        repeat_axis=axis_from_tree(data["repeat_axis"]),
        point_axes=tuple(axis_from_tree(axis) for axis in point_axes),
        point_layout=layout,
        cell_schema=value_schema_from_tree(data["cell_schema"]),
    )
    if _encode(dataset_schema_to_tree(schema)) != _encode(tree):
        raise ValueError("DatasetSchema tree is typed but non-canonical")
    return schema


def axis_layout_to_tree(layout: AxisLayout) -> dict[str, Any]:
    """Project any zlc_data axis layout using its single canonical field model."""

    if not isinstance(layout, AxisLayout):
        raise TypeError("layout must be AxisLayout")
    if layout.mode is AxisLayoutMode.PRODUCT:
        assert layout.factors is not None
        return {
            "logical_shape": list(layout.logical_shape),
            "mode": layout.mode.value,
            "storage_size": layout.storage_size,
            "factors": [axis_layout_to_tree(factor) for factor in layout.factors],
        }
    return {
        "logical_shape": list(layout.logical_shape),
        "mode": layout.mode.value,
        "storage_size": layout.storage_size,
        "storage_to_multi": None
        if layout.storage_to_multi is None
        else [list(index) for index in layout.storage_to_multi],
    }


def axis_layout_from_tree(tree: Any) -> AxisLayout:
    """Reconstruct a generic axis layout from its exact owner-defined fields."""

    if isinstance(tree, dict) and tree.get("mode") == AxisLayoutMode.PRODUCT.value:
        if set(tree) != {"logical_shape", "mode", "storage_size", "factors"}:
            raise ValueError("invalid PRODUCT AxisLayout field set")
        factors = tree["factors"]
        shape = tree["logical_shape"]
        if not isinstance(factors, list) or not isinstance(shape, list):
            raise ValueError("PRODUCT AxisLayout factors/shape must be lists")
        decoded_factors = tuple(axis_layout_from_tree(item) for item in factors)
        layout = AxisLayout.product(*decoded_factors)
        if layout.mode is not AxisLayoutMode.PRODUCT:
            raise ValueError("PRODUCT AxisLayout has a non-canonical factorization")
        declared = AxisLayout(
            tuple(shape),
            AxisLayoutMode.PRODUCT,
            tree["storage_size"],
            factors=decoded_factors,
        )
        if declared != layout:
            raise ValueError("PRODUCT AxisLayout fields are non-canonical")
        if _encode(axis_layout_to_tree(layout)) != _encode(tree):
            raise ValueError("PRODUCT AxisLayout tree is non-canonical")
        return layout
    shape, mode, storage_size, mapping = _axis_layout_fields(tree)
    layout = AxisLayout(shape, mode, storage_size, mapping)
    if _encode(axis_layout_to_tree(layout)) != _encode(tree):
        raise ValueError("AxisLayout tree is non-canonical")
    return layout


def point_layout_to_tree(layout: PointLayout) -> dict[str, Any]:
    """Project a dataset PointLayout through the generic owner serializer."""

    if not isinstance(layout, PointLayout):
        raise TypeError("layout must be PointLayout")
    return axis_layout_to_tree(layout)


def point_layout_from_tree(tree: Any) -> PointLayout:
    """Reconstruct the non-empty dataset specialization of AxisLayout."""

    shape, mode, storage_size, mapping = _axis_layout_fields(tree)
    if mode is AxisLayoutMode.PRODUCT:
        raise ValueError("PointLayout cannot be PRODUCT")
    layout = PointLayout(shape, mode, storage_size, mapping)
    if _encode(point_layout_to_tree(layout)) != _encode(tree):
        raise ValueError("PointLayout tree is non-canonical")
    return layout


def _axis_layout_fields(
    tree: Any,
) -> tuple[tuple[int, ...], AxisLayoutMode, int, tuple[tuple[int, ...], ...] | None]:
    if not isinstance(tree, dict) or set(tree) != {
        "logical_shape",
        "mode",
        "storage_size",
        "storage_to_multi",
    }:
        raise ValueError("invalid PointLayout field set")
    shape_data = tree["logical_shape"]
    mapping_data = tree["storage_to_multi"]
    if not isinstance(shape_data, list):
        raise ValueError("point layout logical_shape must be a list")
    if mapping_data is not None and not isinstance(mapping_data, list):
        raise ValueError("point layout storage_to_multi must be a list or null")
    if mapping_data is not None and any(not isinstance(item, list) for item in mapping_data):
        raise ValueError("point layout multi-indices must be lists")
    return (
        tuple(shape_data),
        AxisLayoutMode(tree["mode"]),
        tree["storage_size"],
        None
        if mapping_data is None
        else tuple(tuple(item) for item in mapping_data),
    )


def value_schema_fingerprint(schema: ValueSchema) -> str:
    return canonical_digest(value_schema_to_tree(schema))


def dataset_schema_fingerprint(schema: DatasetSchema) -> str:
    return canonical_digest(dataset_schema_to_tree(schema))


def validity_to_tree(
    validity: Valid | Invalid | CellValidity | ComponentValidity | DatasetComponentValidity,
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
    if isinstance(validity, DatasetComponentValidity):
        return {
            "kind": "dataset_component",
            "axis_ids": [axis_id.value for axis_id in validity.axis_ids],
            "mask": validity.mask,
        }
    raise TypeError(f"unsupported validity type {type(validity).__name__}")


def validity_from_tree(
    tree: Any,
) -> Valid | Invalid | CellValidity | ComponentValidity | DatasetComponentValidity:
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
            tuple(AxisId(axis_id) for axis_id in axis_ids),
            tree["mask"],
        )
    if kind == "dataset_component" and set(tree) == {"kind", "axis_ids", "mask"}:
        axis_ids = tree["axis_ids"]
        if not isinstance(axis_ids, list):
            raise ValueError("dataset component validity axis_ids must be a list")
        if not isinstance(tree["mask"], np.ndarray):
            raise ValueError("dataset component validity mask must be an ndarray")
        return DatasetComponentValidity(
            tuple(AxisId(axis_id) for axis_id in axis_ids),
            tree["mask"],
        )
    raise ValueError(f"invalid validity payload for kind {kind!r}")


def _require_typed_canonical(got: bytes, expected: bytes, schema_id: str) -> None:
    if bytes(got) != expected:
        raise TypedCodecError(
            f"{schema_id} payload uses a non-canonical typed representation"
        )
