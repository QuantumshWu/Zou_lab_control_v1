"""Strict canonical codecs for FitSpec and FitResultBatch."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from zlc_storage.canonical import (
    canonical_text as _text,
    decode,
    encode,
    exact_mapping as _exact_map,
    integer as _integer,
)

from .axis import AxisId
from .codec import TypedCodecError, axis_from_tree, axis_layout_from_tree, axis_layout_to_tree, axis_to_tree
from .fit_contract import (
    FitBatchStatus,
    FitCoordinateSource,
    FitNumericPolicy,
    FitParameterConstraint,
    FitResultBatch,
    FitSpec,
)
from .fit_model import FitParameterDefinition, ParameterUnitRelation
from .transform_codec import committed_transform_from_tree, committed_transform_to_tree
from .value import BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId


FIT_SPEC_SCHEMA = "zlc_data.FitSpec/v1"
FIT_RESULT_BATCH_SCHEMA = "zlc_data.FitResultBatch/v1"
DATASET_REVISION_REF_SCHEMA = "zlc_data.DatasetRevisionRef/v1"


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
        "model_version": spec.model_version,
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
            "model_version",
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
        _text(data["input_schema_fingerprint"], "input_schema_fingerprint"),
        None if transform is None else committed_transform_from_tree(transform),
        tuple(AxisId(_text(value, "fit_axis_id")) for value in fit_axes),
        tuple(AxisId(_text(value, "batch_axis_id")) for value in batch_axes),
        _text(data["model_id"], "model_id"),
        _integer(data["model_version"], "model_version"),
        tuple(_constraint_from_tree(value) for value in constraints),
        _numeric_policy_from_tree(data["numeric_policy"]),
    )


def encode_fit_spec(spec: FitSpec) -> bytes:
    return encode(fit_spec_to_tree(spec))


def decode_fit_spec(payload: bytes) -> FitSpec:
    spec = fit_spec_from_tree(decode(payload))
    _require_canonical(payload, encode_fit_spec(spec), FIT_SPEC_SCHEMA)
    return spec


def fit_result_batch_to_tree(result: FitResultBatch) -> dict[str, Any]:
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    return {
        "schema": FIT_RESULT_BATCH_SCHEMA,
        "source_ref": _revision_ref_to_tree(result.source_ref),
        "fit_spec": fit_spec_to_tree(result.spec),
        "effective_schema_fingerprint": result.effective_schema_fingerprint,
        "fit_axis_specs": [axis_to_tree(axis) for axis in result.fit_axis_specs],
        "coordinate_sources": [value.value for value in result.coordinate_sources],
        "batch_axis_specs": [axis_to_tree(axis) for axis in result.batch_axis_specs],
        "batch_layout": axis_layout_to_tree(result.batch_layout),
        "value_unit": result.value_unit,
        "parameter_definitions": [_parameter_definition_to_tree(value) for value in result.parameter_definitions],
        "parameter_units": list(result.parameter_units),
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
        "rmse": result.rmse,
        "r_squared": result.r_squared,
        "r_squared_valid": result.r_squared_valid,
        "solver_contract_id": result.solver_contract_id,
        "scipy_version": result.scipy_version,
        "initializer_id": result.initializer_id,
    }


def fit_result_batch_from_tree(tree: Any) -> FitResultBatch:
    fields = {
        "schema",
        "source_ref",
        "fit_spec",
        "effective_schema_fingerprint",
        "fit_axis_specs",
        "coordinate_sources",
        "batch_axis_specs",
        "batch_layout",
        "value_unit",
        "parameter_definitions",
        "parameter_units",
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
        "rmse",
        "r_squared",
        "r_squared_valid",
        "solver_contract_id",
        "scipy_version",
        "initializer_id",
    }
    data = _exact_map(tree, fields, FIT_RESULT_BATCH_SCHEMA)
    for field in (
        "fit_axis_specs",
        "coordinate_sources",
        "batch_axis_specs",
        "parameter_definitions",
        "parameter_units",
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
        "rmse",
        "r_squared",
        "r_squared_valid",
    )
    if any(not isinstance(data[field], np.ndarray) for field in arrays):
        raise ValueError("FitResultBatch numeric fields must be ndarrays")
    errors = tuple(None if value is None else _text(value, "fit error") for value in data["errors"])
    value_unit = data["value_unit"]
    if value_unit is not None:
        value_unit = _text(value_unit, "value_unit")
    parameter_units = tuple(_text(value, "parameter unit") for value in data["parameter_units"])
    return FitResultBatch(
        source_ref=_revision_ref_from_tree(data["source_ref"]),
        spec=fit_spec_from_tree(data["fit_spec"]),
        effective_schema_fingerprint=_text(
            data["effective_schema_fingerprint"], "effective_schema_fingerprint"
        ),
        fit_axis_specs=tuple(axis_from_tree(value) for value in data["fit_axis_specs"]),
        coordinate_sources=tuple(
            FitCoordinateSource(_text(value, "coordinate_source"))
            for value in data["coordinate_sources"]
        ),
        batch_axis_specs=tuple(axis_from_tree(value) for value in data["batch_axis_specs"]),
        batch_layout=axis_layout_from_tree(data["batch_layout"]),
        value_unit=value_unit,
        parameter_definitions=tuple(
            _parameter_definition_from_tree(value) for value in data["parameter_definitions"]
        ),
        parameter_units=parameter_units,
        parameter_values=data["parameter_values"],
        covariance=data["covariance"],
        covariance_valid=data["covariance_valid"],
        statuses=tuple(
            FitBatchStatus(_text(value, "fit_status")) for value in data["statuses"]
        ),
        errors=errors,
        present_observation_counts=data["present_observation_counts"],
        valid_observation_counts=data["valid_observation_counts"],
        used_observation_counts=data["used_observation_counts"],
        evaluation_counts=data["evaluation_counts"],
        residual_sum_squares=data["residual_sum_squares"],
        rmse=data["rmse"],
        r_squared=data["r_squared"],
        r_squared_valid=data["r_squared_valid"],
        solver_contract_id=_text(data["solver_contract_id"], "solver_contract_id"),
        scipy_version=_text(data["scipy_version"], "scipy_version"),
        initializer_id=_text(data["initializer_id"], "initializer_id"),
    )


def encode_fit_result_batch(result: FitResultBatch) -> bytes:
    return encode(fit_result_batch_to_tree(result))


def decode_fit_result_batch(payload: bytes) -> FitResultBatch:
    result = fit_result_batch_from_tree(decode(payload))
    _require_canonical(payload, encode_fit_result_batch(result), FIT_RESULT_BATCH_SCHEMA)
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
    if not isinstance(tree, dict) or set(tree) != {"parameter_name", "initial", "lower", "upper", "fixed"}:
        raise ValueError("invalid FitParameterConstraint fields")
    return FitParameterConstraint(
        _text(tree["parameter_name"], "parameter_name"),
        _optional_real(tree["initial"], "initial"),
        _optional_real(tree["lower"], "lower"),
        _optional_real(tree["upper"], "upper"),
        _optional_real(tree["fixed"], "fixed"),
    )


def _numeric_policy_to_tree(value: FitNumericPolicy) -> dict[str, Any]:
    return {
        "max_evaluations": value.max_evaluations,
        "max_seconds_per_batch": value.max_seconds_per_batch,
        "max_total_seconds": value.max_total_seconds,
        "max_batch_cells": value.max_batch_cells,
        "sample_budget_per_batch": value.sample_budget_per_batch,
        "covariance_rcond": value.covariance_rcond,
    }


def _numeric_policy_from_tree(tree: Any) -> FitNumericPolicy:
    fields = {
        "max_evaluations",
        "max_seconds_per_batch",
        "max_total_seconds",
        "max_batch_cells",
        "sample_budget_per_batch",
        "covariance_rcond",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("invalid FitNumericPolicy fields")
    return FitNumericPolicy(
        _integer(tree["max_evaluations"], "max_evaluations"),
        _real(tree["max_seconds_per_batch"], "max_seconds_per_batch"),
        _real(tree["max_total_seconds"], "max_total_seconds"),
        _integer(tree["max_batch_cells"], "max_batch_cells"),
        _integer(tree["sample_budget_per_batch"], "sample_budget_per_batch"),
        _real(tree["covariance_rcond"], "covariance_rcond"),
    )


def _parameter_definition_to_tree(value: FitParameterDefinition) -> dict[str, Any]:
    return {"name": value.name, "unit_relation": value.unit_relation.value}


def _parameter_definition_from_tree(tree: Any) -> FitParameterDefinition:
    if not isinstance(tree, dict) or set(tree) != {"name", "unit_relation"}:
        raise ValueError("invalid FitParameterDefinition fields")
    return FitParameterDefinition(
        _text(tree["name"], "parameter name"),
        ParameterUnitRelation(_text(tree["unit_relation"], "parameter unit relation")),
    )


def _revision_ref_to_tree(value: DatasetRevisionRef) -> dict[str, Any]:
    return {
        "schema": DATASET_REVISION_REF_SCHEMA,
        "block_id": value.block_id.value,
        "stream_generation": value.stream_generation.value,
        "schema_fingerprint": value.schema_fingerprint,
        "revision": value.revision.value,
    }


def _revision_ref_from_tree(tree: Any) -> DatasetRevisionRef:
    data = _exact_map(
        tree,
        {"schema", "block_id", "stream_generation", "schema_fingerprint", "revision"},
        DATASET_REVISION_REF_SCHEMA,
    )
    return DatasetRevisionRef(
        BlockId(_text(data["block_id"], "block_id")),
        StreamGenerationId(_text(data["stream_generation"], "stream_generation")),
        _text(data["schema_fingerprint"], "schema_fingerprint"),
        DatasetRevision(_integer(data["revision"], "revision")),
    )


def _require_canonical(got: bytes, expected: bytes, schema_id: str) -> None:
    if bytes(got) != expected:
        raise TypedCodecError(f"{schema_id} payload uses a non-canonical typed representation")


def _real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_real(value: Any, field: str) -> float | None:
    return None if value is None else _real(value, field)


__all__ = [
    "decode_fit_result_batch",
    "decode_fit_spec",
    "encode_fit_result_batch",
    "encode_fit_spec",
    "fit_result_batch_from_tree",
    "fit_result_batch_to_tree",
    "fit_spec_from_tree",
    "fit_spec_to_tree",
]
