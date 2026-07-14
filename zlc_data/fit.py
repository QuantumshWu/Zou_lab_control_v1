"""Public headless fit API assembled from the focused fit submodules."""

from .axis import AxisId

from .fit_codec import (
    decode_fit_result_batch,
    decode_fit_spec,
    encode_fit_result_batch,
    encode_fit_spec,
    fit_result_batch_from_tree,
    fit_result_batch_to_tree,
    fit_spec_from_tree,
    fit_spec_to_tree,
)
from .fit_contract import (
    BoundFit,
    FitAcceptance,
    FitBatchStatus,
    FitCancelled,
    FitCoordinateSource,
    FitDeadlineExceeded,
    FitNumericPolicy,
    FitParameterConstraint,
    FitProblem,
    FitResultBatch,
    FitSpec,
    resolve_parameter_units,
)
from .fit_model import (
    FitAxisRequirement,
    FitModelDefinition,
    FitParameterDomain,
    FitParameterDefinition,
    ParameterUnitRelation,
    evaluate_fit_model,
    fit_model_catalog,
    fit_model_definition,
)
from .fit_reference import (
    FIT_RESULT_ARTIFACT_NAMESPACE,
    FIT_RESULT_ARTIFACT_REF_SCHEMA,
    FitResultArtifactRef,
    decode_fit_result_artifact_ref,
    encode_fit_result_artifact_ref,
    fit_result_artifact_ref_from_tree,
    fit_result_artifact_ref_to_tree,
)
from .fit_problem import (
    bind_fit,
    build_fit_problem,
    validate_fit_result_binding,
)
from .schema import DatasetSchema
from .transform import CommittedTransform, resolve_transformed_schema


def _unique_role_matching(model, effective_axes) -> tuple[AxisId, ...]:
    """Return the sole complete model/axis role matching.

    A stable preference is useful for a display suggestion, but it cannot
    decide which physical independent variable an authoritative fit answers.
    Model arity is deliberately bounded to one or two, so enumerating every
    complete matching is both simpler and safer than a greedy priority rule.
    """

    matches: list[tuple[AxisId, ...]] = []

    def visit(position: int, selected: tuple[AxisId, ...]) -> None:
        if position == len(model.axis_requirements):
            matches.append(selected)
            return
        requirement = model.axis_requirements[position]
        for axis in effective_axes:
            if (
                axis.axis_id not in selected
                and axis.role in requirement.allowed_roles
            ):
                visit(position + 1, (*selected, axis.axis_id))

    visit(0, ())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"model {model.model_id!r} has no complete declared-role axis "
            "matching; pass fit_axis_ids explicitly"
        )
    identities = tuple(
        tuple(axis_id.value for axis_id in matching)
        for matching in matches
    )
    raise ValueError(
        f"model {model.model_id!r} has ambiguous declared-role axis "
        f"matchings {identities!r}; pass fit_axis_ids explicitly"
    )


def fit_spec_for(
    schema: DatasetSchema,
    model_id: str,
    *,
    committed_transform: CommittedTransform | None = None,
    fit_axis_ids: tuple[AxisId, ...] | None = None,
    constraints: tuple[FitParameterConstraint, ...] = (),
    numeric_policy: FitNumericPolicy = FitNumericPolicy(),
) -> FitSpec:
    """Build a total named-axis fit request without changing any data.

    Explicit fit axes are kept in caller order.  When they are omitted, the
    closed model's declared axis roles must have exactly one complete matching.
    Any second physically valid matching is ambiguous and rejected.
    Every effective axis not selected for fitting is preserved as a batch axis.
    This helper never selects, reduces, flattens, or otherwise changes data.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    model = fit_model_definition(model_id)
    if committed_transform is None:
        effective_axes = (
            schema.repeat_axis,
            *schema.point_axes,
            *schema.cell_schema.data_axes,
        )
    else:
        effective_axes = resolve_transformed_schema(
            schema,
            committed_transform,
        ).axes

    if fit_axis_ids is None:
        resolved_fit_axis_ids = _unique_role_matching(model, effective_axes)
    else:
        resolved_fit_axis_ids = tuple(fit_axis_ids)

    batch_axis_ids = tuple(
        axis.axis_id
        for axis in effective_axes
        if axis.axis_id not in resolved_fit_axis_ids
    )
    spec = FitSpec(
        input_schema_fingerprint=schema.fingerprint,
        committed_transform=committed_transform,
        fit_axis_ids=resolved_fit_axis_ids,
        batch_axis_ids=batch_axis_ids,
        model_id=model.model_id,
        constraints=constraints,
        numeric_policy=numeric_policy,
    )
    return bind_fit(spec, schema).spec


def fit_analysis(
    bound: BoundFit,
    snapshot,
    *,
    cancel_check=None,
    deadline_monotonic=None,
) -> FitResultBatch:
    """Lazy public solver entrypoint; importing ``zlc_data`` does not import SciPy."""

    from .fit_solver import fit_analysis as _fit_analysis

    return _fit_analysis(
        bound,
        snapshot,
        cancel_check=cancel_check,
        deadline_monotonic=deadline_monotonic,
    )


__all__ = [
    "BoundFit",
    "FIT_RESULT_ARTIFACT_NAMESPACE",
    "FIT_RESULT_ARTIFACT_REF_SCHEMA",
    "FitAcceptance",
    "FitAxisRequirement",
    "FitBatchStatus",
    "FitCancelled",
    "FitCoordinateSource",
    "FitDeadlineExceeded",
    "FitModelDefinition",
    "FitNumericPolicy",
    "FitParameterConstraint",
    "FitParameterDomain",
    "FitParameterDefinition",
    "FitProblem",
    "FitResultArtifactRef",
    "FitResultBatch",
    "FitSpec",
    "ParameterUnitRelation",
    "bind_fit",
    "build_fit_problem",
    "decode_fit_result_batch",
    "decode_fit_result_artifact_ref",
    "decode_fit_spec",
    "encode_fit_result_batch",
    "encode_fit_result_artifact_ref",
    "encode_fit_spec",
    "evaluate_fit_model",
    "fit_model_catalog",
    "fit_model_definition",
    "fit_result_artifact_ref_from_tree",
    "fit_result_artifact_ref_to_tree",
    "fit_analysis",
    "fit_spec_for",
    "fit_result_batch_from_tree",
    "fit_result_batch_to_tree",
    "fit_spec_from_tree",
    "fit_spec_to_tree",
    "resolve_parameter_units",
    "validate_fit_result_binding",
]
