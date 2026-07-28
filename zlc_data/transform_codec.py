"""Canonical current-schema trees for zlc_data transform authority values."""

from __future__ import annotations

from typing import Any

from zlc_storage.canonical import (
    encode as _encode,
    exact_mapping as _exact_map,
)

from .codec import (
    axis_source_ref_from_tree,
    axis_source_ref_to_tree,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
)
from .selection import Selection, selection_from_tree, selection_to_tree
from .transform import (
    COMMITTED_TRANSFORM_SCHEMA,
    TRANSFORM_SPEC_SCHEMA,
    CommittedTransform,
    DataTransformSpec,
    HistogramSpec,
    MissingPolicy,
    ReductionMethod,
    ReductionSpec,
    ValidityPolicy,
)


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
                    "sources": [
                        axis_source_ref_to_tree(source)
                        for source in operation.sources
                    ],
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
                    "sources": [
                        axis_source_ref_to_tree(source)
                        for source in operation.sources
                    ],
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
                    "sources",
                    "method",
                    "missing_policy",
                    "validity_policy",
                    "minimum_valid_count",
                },
                "REDUCE",
                discriminator="kind",
            )
            if not isinstance(item["sources"], list):
                raise ValueError("reduction sources must be a list")
            operations.append(
                ReductionSpec(
                    sources=tuple(
                        axis_source_ref_from_tree(source)
                        for source in item["sources"]
                    ),
                    method=ReductionMethod(item["method"]),
                    missing_policy=MissingPolicy(item["missing_policy"]),
                    validity_policy=ValidityPolicy(item["validity_policy"]),
                    minimum_valid_count=item["minimum_valid_count"],
                )
            )
        elif raw["kind"] == "HISTOGRAM":
            item = _exact_map(
                raw,
                {"kind", "sources", "bin_edges"},
                "HISTOGRAM",
                discriminator="kind",
            )
            if not isinstance(item["sources"], list):
                raise ValueError("histogram sources must be a list")
            if not isinstance(item["bin_edges"], list):
                raise ValueError("histogram bin_edges must be a list")
            operations.append(
                HistogramSpec(
                    sources=tuple(
                        axis_source_ref_from_tree(source)
                        for source in item["sources"]
                    ),
                    bin_edges=tuple(item["bin_edges"]),
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
        "source_schema_fingerprint": transform.source_schema_fingerprint,
        "exact_point_ordinals": list(transform.exact_point_ordinals),
        "spec": data_transform_spec_to_tree(transform.spec),
        "effective_output_schema": dataset_schema_to_tree(
            transform.effective_output_schema
        ),
    }


def committed_transform_from_tree(tree: Any) -> CommittedTransform:
    data = _exact_map(
        tree,
        {
            "schema",
            "source_schema_fingerprint",
            "exact_point_ordinals",
            "spec",
            "effective_output_schema",
        },
        COMMITTED_TRANSFORM_SCHEMA,
    )
    if not isinstance(data["exact_point_ordinals"], list):
        raise ValueError("CommittedTransform exact_point_ordinals must be a list")
    transform = CommittedTransform(
        source_schema_fingerprint=data["source_schema_fingerprint"],
        exact_point_ordinals=tuple(data["exact_point_ordinals"]),
        spec=data_transform_spec_from_tree(data["spec"]),
        effective_output_schema=dataset_schema_from_tree(
            data["effective_output_schema"]
        ),
    )
    if _encode(committed_transform_to_tree(transform)) != _encode(tree):
        raise ValueError("CommittedTransform tree is typed but non-canonical")
    return transform


__all__ = [
    "committed_transform_from_tree",
    "committed_transform_to_tree",
    "data_transform_spec_from_tree",
    "data_transform_spec_to_tree",
]
