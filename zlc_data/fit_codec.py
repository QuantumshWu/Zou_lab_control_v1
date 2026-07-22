"""Strict canonical codecs for FitSpec and FitResultBatch."""

from __future__ import annotations

from typing import Any

import numpy as np

from zlc_storage.canonical import (
    decode,
    encode,
    exact_mapping as _exact_map,
)

from .axis import AxisId
from .codec import (
    _require_typed_canonical,
    axis_from_tree,
    axis_layout_from_tree,
    axis_layout_to_tree,
    axis_to_tree,
    dataset_revision_ref_from_tree,
    dataset_revision_ref_to_tree,
)
from .fit_contract import (
    FitBatchStatus,
    FitNumericPolicy,
    FitParameterConstraint,
    FitResultBatch,
    FitSpec,
)
from .transform_codec import committed_transform_from_tree, committed_transform_to_tree


FIT_SPEC_SCHEMA = "zlc_data.FitSpec"
FIT_RESULT_BATCH_SCHEMA = "zlc_data.FitResultBatch"

def fit_spec_to_tree(spec: FitSpec) -> dict[str, Any]:
    if not isinstance(spec, FitSpec):
        raise TypeError("spec must be FitSpec")
    return {
        "schema": FIT_SPEC_SCHEMA,
        "input_schema_fingerprint": spec.input_schema_fingerprint,
        "committed_transform": None
        if spec.committed_transform is None
        else committed_transform_to_tree(spec.committed_transform),
        "fit_axis_ids": [axis_id.value for axis_id in spec.fit_axis_ids],
        "batch_axis_ids": [axis_id.value for axis_id in spec.batch_axis_ids],
        "model_id": spec.model_id,
        "constraints": [_constraint_to_tree(item) for item in spec.constraints],
        "numeric_policy": _numeric_policy_to_tree(spec.numeric_policy),
    }


def fit_spec_from_tree(tree: Any) -> FitSpec:
    data = _exact_map(
        tree,
        {
            "schema",
            "input_schema_fingerprint",
            "committed_transform",
            "fit_axis_ids",
            "batch_axis_ids",
            "model_id",
            "constraints",
            "numeric_policy",
        },
        FIT_SPEC_SCHEMA,
    )
    fit_axes = data["fit_axis_ids"]
    batch_axes = data["batch_axis_ids"]
    constraints = data["constraints"]
    if not isinstance(fit_axes, list) or not isinstance(batch_axes, list) or not isinstance(constraints, list):
        raise ValueError("FitSpec axes and constraints must be lists")
    transform = data["committed_transform"]
    return FitSpec(
        input_schema_fingerprint=data["input_schema_fingerprint"],
        committed_transform=(
            None if transform is None else committed_transform_from_tree(transform)
        ),
        fit_axis_ids=tuple(AxisId(value) for value in fit_axes),
        batch_axis_ids=tuple(AxisId(value) for value in batch_axes),
        model_id=data["model_id"],
        constraints=tuple(_constraint_from_tree(value) for value in constraints),
        numeric_policy=_numeric_policy_from_tree(data["numeric_policy"]),
    )


def encode_fit_spec(spec: FitSpec) -> bytes:
    return encode(fit_spec_to_tree(spec))


def decode_fit_spec(payload: bytes) -> FitSpec:
    spec = fit_spec_from_tree(decode(payload))
    _require_typed_canonical(payload, encode_fit_spec(spec), FIT_SPEC_SCHEMA)
    return spec


def _fit_result_batch_to_tree(result: FitResultBatch) -> dict[str, Any]:
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    return {
        "schema": FIT_RESULT_BATCH_SCHEMA,
        "source_ref": dataset_revision_ref_to_tree(result.source_ref),
        "fit_spec": fit_spec_to_tree(result.spec),
        "fit_axis_specs": [axis_to_tree(axis) for axis in result.fit_axis_specs],
        "batch_axis_specs": [axis_to_tree(axis) for axis in result.batch_axis_specs],
        "batch_layout": axis_layout_to_tree(result.batch_layout),
        "value_unit": result.value_unit,
        "parameter_values": result.parameter_values,
        "covariance": result.covariance,
        "covariance_valid": result.covariance_valid,
        "statuses": [value.value for value in result.statuses],
        "errors": list(result.errors),
        "present_observation_counts": result.present_observation_counts,
        "valid_observation_counts": result.valid_observation_counts,
        "used_observation_counts": result.used_observation_counts,
        "evaluation_counts": result.evaluation_counts,
        "residual_sum_squares": result.residual_sum_squares,
        "r_squared": result.r_squared,
        "r_squared_valid": result.r_squared_valid,
        "scipy_version": result.scipy_version,
    }


def _fit_result_batch_from_tree(tree: Any) -> FitResultBatch:
    fields = {
        "schema",
        "source_ref",
        "fit_spec",
        "fit_axis_specs",
        "batch_axis_specs",
        "batch_layout",
        "value_unit",
        "parameter_values",
        "covariance",
        "covariance_valid",
        "statuses",
        "errors",
        "present_observation_counts",
        "valid_observation_counts",
        "used_observation_counts",
        "evaluation_counts",
        "residual_sum_squares",
        "r_squared",
        "r_squared_valid",
        "scipy_version",
    }
    data = _exact_map(tree, fields, FIT_RESULT_BATCH_SCHEMA)
    for field in (
        "fit_axis_specs",
        "batch_axis_specs",
        "statuses",
        "errors",
    ):
        if not isinstance(data[field], list):
            raise ValueError(f"FitResultBatch {field} must be a list")
    arrays = (
        "parameter_values",
        "covariance",
        "covariance_valid",
        "present_observation_counts",
        "valid_observation_counts",
        "used_observation_counts",
        "evaluation_counts",
        "residual_sum_squares",
        "r_squared",
        "r_squared_valid",
    )
    if any(not isinstance(data[field], np.ndarray) for field in arrays):
        raise ValueError("FitResultBatch numeric fields must be ndarrays")
    return FitResultBatch(
        source_ref=dataset_revision_ref_from_tree(data["source_ref"]),
        spec=fit_spec_from_tree(data["fit_spec"]),
        fit_axis_specs=tuple(axis_from_tree(value) for value in data["fit_axis_specs"]),
        batch_axis_specs=tuple(axis_from_tree(value) for value in data["batch_axis_specs"]),
        batch_layout=axis_layout_from_tree(data["batch_layout"]),
        value_unit=data["value_unit"],
        parameter_values=data["parameter_values"],
        covariance=data["covariance"],
        covariance_valid=data["covariance_valid"],
        statuses=tuple(FitBatchStatus(value) for value in data["statuses"]),
        errors=tuple(data["errors"]),
        present_observation_counts=data["present_observation_counts"],
        valid_observation_counts=data["valid_observation_counts"],
        used_observation_counts=data["used_observation_counts"],
        evaluation_counts=data["evaluation_counts"],
        residual_sum_squares=data["residual_sum_squares"],
        r_squared=data["r_squared"],
        r_squared_valid=data["r_squared_valid"],
        scipy_version=data["scipy_version"],
    )


def encode_fit_result_batch(result: FitResultBatch) -> bytes:
    return encode(_fit_result_batch_to_tree(result))


def decode_fit_result_batch(payload: bytes) -> FitResultBatch:
    result = _fit_result_batch_from_tree(decode(payload))
    _require_typed_canonical(
        payload,
        encode_fit_result_batch(result),
        FIT_RESULT_BATCH_SCHEMA,
    )
    return result


def _constraint_to_tree(value: FitParameterConstraint) -> dict[str, Any]:
    return {
        "parameter_name": value.parameter_name,
        "initial": value.initial,
        "lower": value.lower,
        "upper": value.upper,
        "fixed": value.fixed,
    }


def _constraint_from_tree(tree: Any) -> FitParameterConstraint:
    data = _exact_map(
        tree,
        {"parameter_name", "initial", "lower", "upper", "fixed"},
        "FitParameterConstraint",
        discriminator=None,
    )
    return FitParameterConstraint(
        data["parameter_name"],
        data["initial"],
        data["lower"],
        data["upper"],
        data["fixed"],
    )


def _numeric_policy_to_tree(value: FitNumericPolicy) -> dict[str, Any]:
    return {
        "max_evaluations": value.max_evaluations,
        "covariance_rcond": value.covariance_rcond,
    }


def _numeric_policy_from_tree(tree: Any) -> FitNumericPolicy:
    fields = {
        "max_evaluations",
        "covariance_rcond",
    }
    data = _exact_map(
        tree,
        fields,
        "FitNumericPolicy",
        discriminator=None,
    )
    return FitNumericPolicy(
        max_evaluations=data["max_evaluations"],
        covariance_rcond=data["covariance_rcond"],
    )


__all__ = [
    "decode_fit_spec",
    "decode_fit_result_batch",
    "encode_fit_spec",
    "encode_fit_result_batch",
    "fit_spec_from_tree",
    "fit_spec_to_tree",
]
