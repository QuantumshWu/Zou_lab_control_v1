"""Canonical current-schema trees for zlc_data transform authority values."""

from __future__ import annotations

from typing import Any

from zlc_storage.canonical import (
    encode as _encode,
    exact_mapping as _exact_map,
)

from .axis import AxisId
from .codec import axis_layout_to_tree, axis_to_tree
from .selection import Selection, selection_from_tree, selection_to_tree
from .transform import (
    COMMITTED_TRANSFORM_SCHEMA,
    TRANSFORMED_SCHEMA_SCHEMA,
    TRANSFORM_SPEC_SCHEMA,
    CommittedTransform,
    DataTransformSpec,
    HistogramSpec,
    MissingPolicy,
    ReductionMethod,
    ReductionSpec,
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


def data_transform_spec_to_tree(spec: DataTransformSpec) -> dict[str, Any]:
    if not isinstance(spec, DataTransformSpec):
        raise TypeError("spec must be DataTransformSpec")
    operations: list[dict[str, Any]] = []
    for operation in spec.operations:
        if isinstance(operation, Selection):
            operations.append(
                {"kind": "SELECT", "selection": selection_to_tree(operation)}
            )
        elif isinstance(operation, ReductionSpec):
            operations.append(
                {
                    "kind": "REDUCE",
                    "axis_ids": [axis_id.value for axis_id in operation.axis_ids],
                    "method": operation.method.value,
                    "missing_policy": operation.missing_policy.value,
                    "validity_policy": operation.validity_policy.value,
                    "minimum_valid_count": operation.minimum_valid_count,
                }
            )
        else:
            operations.append(
                {
                    "kind": "HISTOGRAM",
                    "axis_ids": [axis_id.value for axis_id in operation.axis_ids],
                    "bin_edges": list(operation.bin_edges),
                }
            )
    return {"schema": TRANSFORM_SPEC_SCHEMA, "operations": operations}


def data_transform_spec_from_tree(tree: Any) -> DataTransformSpec:
    data = _exact_map(tree, {"schema", "operations"}, TRANSFORM_SPEC_SCHEMA)
    if not isinstance(data["operations"], list):
        raise ValueError("DataTransformSpec operations must be a list")
    operations: list[Selection | ReductionSpec | HistogramSpec] = []
    for raw in data["operations"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
            raise ValueError("transform operation must be a tagged map")
        if raw["kind"] == "SELECT":
            item = _exact_map(
                raw,
                {"kind", "selection"},
                "SELECT",
                discriminator="kind",
            )
            operations.append(selection_from_tree(item["selection"]))
        elif raw["kind"] == "REDUCE":
            item = _exact_map(
                raw,
                {
                    "kind",
                    "axis_ids",
                    "method",
                    "missing_policy",
                    "validity_policy",
                    "minimum_valid_count",
                },
                "REDUCE",
                discriminator="kind",
            )
            if not isinstance(item["axis_ids"], list):
                raise ValueError("reduction axis_ids must be a list")
            operations.append(
                ReductionSpec(
                    tuple(AxisId(value) for value in item["axis_ids"]),
                    ReductionMethod(item["method"]),
                    MissingPolicy(item["missing_policy"]),
                    ValidityPolicy(item["validity_policy"]),
                    item["minimum_valid_count"],
                )
            )
        elif raw["kind"] == "HISTOGRAM":
            item = _exact_map(
                raw,
                {"kind", "axis_ids", "bin_edges"},
                "HISTOGRAM",
                discriminator="kind",
            )
            if not isinstance(item["axis_ids"], list):
                raise ValueError("histogram axis_ids must be a list")
            if not isinstance(item["bin_edges"], list):
                raise ValueError("histogram bin_edges must be a list")
            operations.append(
                HistogramSpec(
                    tuple(AxisId(value) for value in item["axis_ids"]),
                    tuple(item["bin_edges"]),
                )
            )
        else:
            raise ValueError(f"invalid transform operation kind {raw['kind']!r}")
    spec = DataTransformSpec(tuple(operations))
    if _encode(data_transform_spec_to_tree(spec)) != _encode(tree):
        raise ValueError("DataTransformSpec tree is typed but non-canonical")
    return spec


def committed_transform_to_tree(transform: CommittedTransform) -> dict[str, Any]:
    if not isinstance(transform, CommittedTransform):
        raise TypeError("transform must be CommittedTransform")
    return {
        "schema": COMMITTED_TRANSFORM_SCHEMA,
        "input_schema_fingerprint": transform.input_schema_fingerprint,
        "spec": data_transform_spec_to_tree(transform.spec),
        "output_schema_fingerprint": transform.output_schema_fingerprint,
    }


def committed_transform_from_tree(tree: Any) -> CommittedTransform:
    data = _exact_map(
        tree,
        {
            "schema",
            "input_schema_fingerprint",
            "spec",
            "output_schema_fingerprint",
        },
        COMMITTED_TRANSFORM_SCHEMA,
    )
    transform = CommittedTransform(
        data["input_schema_fingerprint"],
        data_transform_spec_from_tree(data["spec"]),
        data["output_schema_fingerprint"],
    )
    if _encode(committed_transform_to_tree(transform)) != _encode(tree):
        raise ValueError("CommittedTransform tree is typed but non-canonical")
    return transform


__all__ = [
    "committed_transform_from_tree",
    "committed_transform_to_tree",
    "data_transform_spec_from_tree",
    "data_transform_spec_to_tree",
    "transformed_schema_to_tree",
]
