"""Canonical current-schema codecs for zlc_data transform authority values."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from zlc_storage.canonical import decode, encode

from .axis import AxisId
from .codec import axis_from_tree, axis_layout_from_tree, axis_layout_to_tree, axis_to_tree
from .selection import selection_from_tree, selection_to_tree
from .transform import (
    COMMITTED_TRANSFORM_SCHEMA,
    TRANSFORMED_SCHEMA_SCHEMA,
    TRANSFORM_SPEC_SCHEMA,
    CommittedTransform,
    DataTransformSpec,
    MissingPolicy,
    Reduce,
    ReductionMethod,
    ReductionSpec,
    Select,
    TransformOperation,
    TransformOrigin,
    TransformRevision,
    TransformedSchema,
    ValidityPolicy,
)


def transformed_schema_to_tree(schema: TransformedSchema) -> dict[str, Any]:
    return {
        "schema": TRANSFORMED_SCHEMA_SCHEMA,
        "cell_axes": [axis_to_tree(axis) for axis in schema.cell_axes],
        "cell_layout": axis_layout_to_tree(schema.cell_layout),
        "data_axes": [axis_to_tree(axis) for axis in schema.data_axes],
        "validity_axis_ids": [axis_id.value for axis_id in schema.validity_axis_ids],
        "dtype": schema.dtype.str,
        "value_unit": schema.value_unit,
    }


def transformed_schema_from_tree(tree: Any) -> TransformedSchema:
    if not isinstance(tree, dict) or set(tree) != {
        "schema",
        "cell_axes",
        "cell_layout",
        "data_axes",
        "validity_axis_ids",
        "dtype",
        "value_unit",
    }:
        raise ValueError("invalid TransformedSchema field set")
    if tree["schema"] != TRANSFORMED_SCHEMA_SCHEMA:
        raise ValueError(f"expected schema {TRANSFORMED_SCHEMA_SCHEMA!r}")
    if (
        not isinstance(tree["cell_axes"], list)
        or not isinstance(tree["data_axes"], list)
        or not isinstance(tree["validity_axis_ids"], list)
    ):
        raise ValueError("transformed axes must be lists")
    unit = tree["value_unit"]
    return TransformedSchema(
        tuple(axis_from_tree(item) for item in tree["cell_axes"]),
        axis_layout_from_tree(tree["cell_layout"]),
        tuple(axis_from_tree(item) for item in tree["data_axes"]),
        tuple(AxisId(_text(item, "validity_axis_id")) for item in tree["validity_axis_ids"]),
        np.dtype(_text(tree["dtype"], "dtype")),
        None if unit is None else _text(unit, "value_unit"),
    )


def data_transform_spec_to_tree(spec: DataTransformSpec) -> dict[str, Any]:
    if not isinstance(spec, DataTransformSpec):
        raise TypeError("spec must be DataTransformSpec")
    operations: list[dict[str, Any]] = []
    for operation in spec.operations:
        if isinstance(operation, Select):
            operations.append(
                {"kind": "SELECT", "selection": selection_to_tree(operation.selection)}
            )
        else:
            reduction = operation.reduction
            operations.append(
                {
                    "kind": "REDUCE",
                    "axis_ids": [axis_id.value for axis_id in reduction.axis_ids],
                    "method": reduction.method.value,
                    "missing_policy": reduction.missing_policy.value,
                    "validity_policy": reduction.validity_policy.value,
                    "minimum_valid_count": reduction.minimum_valid_count,
                }
            )
    return {"schema": TRANSFORM_SPEC_SCHEMA, "operations": operations}


def data_transform_spec_from_tree(tree: Any) -> DataTransformSpec:
    if not isinstance(tree, dict) or set(tree) != {"schema", "operations"}:
        raise ValueError("DataTransformSpec must contain exactly schema and operations")
    if tree["schema"] != TRANSFORM_SPEC_SCHEMA or not isinstance(tree["operations"], list):
        raise ValueError("invalid DataTransformSpec payload")
    operations: list[TransformOperation] = []
    for raw in tree["operations"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
            raise ValueError("transform operation must be a tagged map")
        if raw["kind"] == "SELECT" and set(raw) == {"kind", "selection"}:
            operations.append(Select(selection_from_tree(raw["selection"])))
        elif raw["kind"] == "REDUCE" and set(raw) == {
            "kind",
            "axis_ids",
            "method",
            "missing_policy",
            "validity_policy",
            "minimum_valid_count",
        }:
            if not isinstance(raw["axis_ids"], list):
                raise ValueError("reduction axis_ids must be a list")
            operations.append(
                Reduce(
                    ReductionSpec(
                        tuple(AxisId(_text(item, "axis_id")) for item in raw["axis_ids"]),
                        ReductionMethod(_text(raw["method"], "method")),
                        MissingPolicy(_text(raw["missing_policy"], "missing_policy")),
                        ValidityPolicy(_text(raw["validity_policy"], "validity_policy")),
                        None
                        if raw["minimum_valid_count"] is None
                        else _integer(raw["minimum_valid_count"], "minimum_valid_count"),
                    )
                )
            )
        else:
            raise ValueError(f"invalid transform operation for kind {raw['kind']!r}")
    return DataTransformSpec(tuple(operations))


def committed_transform_payload_tree(transform: CommittedTransform) -> dict[str, Any]:
    return {
        "schema": COMMITTED_TRANSFORM_SCHEMA,
        "input_schema_fingerprint": transform.input_schema_fingerprint,
        "spec": data_transform_spec_to_tree(transform.spec),
        "output_schema_fingerprint": transform.output_schema_fingerprint,
        "revision": transform.revision.value,
        "origin": transform.origin.value,
    }


def committed_transform_to_tree(transform: CommittedTransform) -> dict[str, Any]:
    if not isinstance(transform, CommittedTransform):
        raise TypeError("transform must be CommittedTransform")
    tree = committed_transform_payload_tree(transform)
    tree["transform_digest"] = transform.transform_digest
    return tree


def committed_transform_from_tree(tree: Any) -> CommittedTransform:
    if not isinstance(tree, dict) or set(tree) != {
        "schema",
        "input_schema_fingerprint",
        "spec",
        "output_schema_fingerprint",
        "revision",
        "origin",
        "transform_digest",
    }:
        raise ValueError("invalid CommittedTransform field set")
    if tree["schema"] != COMMITTED_TRANSFORM_SCHEMA:
        raise ValueError(f"expected schema {COMMITTED_TRANSFORM_SCHEMA!r}")
    return CommittedTransform(
        _text(tree["input_schema_fingerprint"], "input_schema_fingerprint"),
        data_transform_spec_from_tree(tree["spec"]),
        _text(tree["output_schema_fingerprint"], "output_schema_fingerprint"),
        TransformRevision(_integer(tree["revision"], "revision")),
        TransformOrigin(_text(tree["origin"], "origin")),
        _text(tree["transform_digest"], "transform_digest"),
    )


def encode_committed_transform(transform: CommittedTransform) -> bytes:
    return encode(committed_transform_to_tree(transform))


def decode_committed_transform(payload: bytes) -> CommittedTransform:
    transform = committed_transform_from_tree(decode(payload))
    if bytes(payload) != encode_committed_transform(transform):
        raise ValueError("CommittedTransform payload uses a non-canonical typed representation")
    return transform


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)


__all__ = [
    "committed_transform_from_tree",
    "committed_transform_payload_tree",
    "committed_transform_to_tree",
    "data_transform_spec_from_tree",
    "data_transform_spec_to_tree",
    "decode_committed_transform",
    "encode_committed_transform",
    "transformed_schema_from_tree",
    "transformed_schema_to_tree",
]
