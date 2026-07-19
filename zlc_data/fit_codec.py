"""Strict canonical codecs for FitSpec and FitResultBatch."""

from __future__ import annotations

from numbers import Integral
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
    fit_result_retained_upper_bound_nbytes,
)
from .selection import Selection
from .transform_codec import committed_transform_from_tree, committed_transform_to_tree


FIT_SPEC_SCHEMA = "zlc_data.FitSpec"
FIT_RESULT_BATCH_SCHEMA = "zlc_data.FitResultBatch"

_FIT_CODEC_ENCODE_FIXED_WORKSPACE_BYTES = 8 * 1024 * 1024
_FIT_CODEC_DECODE_FIXED_WORKSPACE_BYTES = 16 * 1024 * 1024


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


def _fit_result_codec_text_upper_bound_nbytes(result: FitResultBatch) -> int:
    """Inventory all unbounded owner text without allocating encoded copies."""

    characters = (
        len(result.source_ref.block_id.value)
        + len(result.source_ref.stream_generation.value)
        + len(result.source_ref.schema_fingerprint)
        + len(result.spec.input_schema_fingerprint)
        + len(result.spec.model_id)
        + len(result.scipy_version)
        + sum(len(axis_id.value) for axis_id in result.spec.fit_axis_ids)
        + sum(len(axis_id.value) for axis_id in result.spec.batch_axis_ids)
        + sum(
            len(constraint.parameter_name)
            for constraint in result.spec.constraints
        )
        + sum(len(status.value) for status in result.statuses)
        + sum(len(error) for error in result.errors if error is not None)
    )
    if result.value_unit is not None:
        characters += len(result.value_unit)
    for axis_group in (result.fit_axis_specs, result.batch_axis_specs):
        for axis in axis_group:
            characters += (
                len(axis.axis_id.value) + len(axis.name) + len(axis.role.value)
            )
            if axis.unit is not None:
                characters += len(axis.unit)
            if axis.coordinate_frame is not None:
                characters += len(axis.coordinate_frame.value)
            if axis.coordinates is not None:
                characters += sum(
                    len(value)
                    for value in axis.coordinates
                    if isinstance(value, str)
                )
    transform = result.spec.committed_transform
    if transform is not None:
        characters += len(transform.input_schema_fingerprint) + len(
            transform.output_schema_fingerprint
        )
        for operation in transform.spec.operations:
            if isinstance(operation, Selection):
                for term in operation.terms:
                    characters += len(term.axis_id.value)
                    frame = getattr(term, "coordinate_frame", None)
                    if frame is not None:
                        characters += len(frame.value)
            else:
                characters += sum(
                    len(axis_id.value) for axis_id in operation.axis_ids
                )
    # UTF-8 uses at most four bytes per Unicode code point.  No encoded strings
    # are built here, so a tiny rejected budget cannot itself trigger a copy of
    # an unbounded identifier or coordinate label.
    return 4 * characters


def fit_result_encode_additional_peak_upper_bound_nbytes(
    result: FitResultBatch,
) -> int:
    """Bound additional workspace for encoding an already-resident result.

    The returned value deliberately excludes the :class:`FitResultBatch`
    itself.  A composition root may therefore subtract a Figure front which
    already retains that result, then pass the remaining operation budget to
    the artifact owner without counting the result twice.

    The canonical encoder can simultaneously retain accumulated base64 text,
    its tagged Python tree, a JSON unicode string, UTF-8 bytes, framed bytes,
    and one ndarray normalization/conversion scratch set.  The retained-result
    estimator supplies a conservative inventory of rows, axes, coordinates,
    and text; here it sizes only those *new codec copies*.
    """

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    arrays = (
        result.parameter_values,
        result.covariance,
        result.covariance_valid,
        result.present_observation_counts,
        result.valid_observation_counts,
        result.used_observation_counts,
        result.evaluation_counts,
        result.residual_sum_squares,
        result.r_squared,
        result.r_squared_valid,
    )
    array_nbytes = tuple(int(value.nbytes) for value in arrays)
    base64_nbytes = tuple(4 * ((size + 2) // 3) for size in array_nbytes)
    total_arrays = sum(array_nbytes)
    total_base64 = sum(base64_nbytes)
    largest_array = max(array_nbytes, default=0)
    largest_base64 = max(base64_nbytes, default=0)
    retained = fit_result_retained_upper_bound_nbytes(result)
    structured_inventory = max(0, retained - total_arrays)
    owner_text = _fit_result_codec_text_upper_bound_nbytes(result)

    # The structured inventory covers canonical tags/keys; six copies of the
    # complete owner-text inventory cover JSON's worst control-character escape
    # expansion.  Six concurrent payload-sized copies then cover the widest
    # Python unicode representation, UTF-8 bytes, and final framed bytes.
    payload_upper_bound = (
        total_base64 + 2 * structured_inventory + 6 * owner_text
    )
    tagged_tree_upper_bound = total_base64 + 2 * structured_inventory
    active_array_scratch = 2 * largest_array + 2 * largest_base64
    return int(
        _FIT_CODEC_ENCODE_FIXED_WORKSPACE_BYTES
        + tagged_tree_upper_bound
        + 6 * payload_upper_bound
        + active_array_scratch
    )


def fit_result_decode_additional_peak_upper_bound_nbytes(
    encoded_payload_nbytes: int,
) -> int:
    """Bound load-time codec allocations from one not-yet-read result blob.

    This includes the newly read payload and decoded result because neither is
    resident when repository admission begins.  It also covers canonical JSON
    parsing, structure inspection, ndarray base64 decode/copy, the generic
    canonical rebuild, and the typed result rebuild.  Caller-owned Figure/front
    memory and later source-artifact admission are intentionally excluded.
    """

    if isinstance(encoded_payload_nbytes, bool) or not isinstance(
        encoded_payload_nbytes,
        Integral,
    ):
        raise TypeError("encoded_payload_nbytes must be an integer")
    if encoded_payload_nbytes <= 0:
        raise ValueError("encoded_payload_nbytes must be a positive integer")
    payload = int(encoded_payload_nbytes)
    # The fit payload has a closed field set and bounded canonical node model.
    # This factor is intentionally above both canonical validation passes'
    # simultaneous UTF-8/tagged/tree/array/re-encode allocation sets.
    return _FIT_CODEC_DECODE_FIXED_WORKSPACE_BYTES + 64 * payload


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
        "max_batch_cells": value.max_batch_cells,
        "sample_budget_per_batch": value.sample_budget_per_batch,
        "max_packed_observations": value.max_packed_observations,
        "covariance_rcond": value.covariance_rcond,
    }


def _numeric_policy_from_tree(tree: Any) -> FitNumericPolicy:
    fields = {
        "max_evaluations",
        "max_batch_cells",
        "sample_budget_per_batch",
        "max_packed_observations",
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
        max_batch_cells=data["max_batch_cells"],
        sample_budget_per_batch=data["sample_budget_per_batch"],
        max_packed_observations=data["max_packed_observations"],
        covariance_rcond=data["covariance_rcond"],
    )


__all__ = [
    "decode_fit_spec",
    "decode_fit_result_batch",
    "encode_fit_spec",
    "encode_fit_result_batch",
    "fit_result_decode_additional_peak_upper_bound_nbytes",
    "fit_result_encode_additional_peak_upper_bound_nbytes",
    "fit_spec_from_tree",
    "fit_spec_to_tree",
]
