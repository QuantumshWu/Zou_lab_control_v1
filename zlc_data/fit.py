"""Public headless fit API assembled from the focused fit submodules."""

from .axis import AxisId, SCALAR

from .fit_codec import (
    decode_fit_result_batch,
    decode_fit_spec,
    encode_fit_result_batch,
    encode_fit_spec,
    fit_spec_from_tree,
    fit_spec_to_tree,
)
from .fit_contract import (
    BoundFit,
    FitBatchStatus,
    FitCancelled,
    FitCoordinateSource,
    FitDeadlineExceeded,
    FitNumericPolicy,
    FitParameterConstraint,
    FitResultBatch,
    FitSpec,
)
from .fit_model import (
    FitModelDefinition,
    FitParameterDefinition,
    FitParameterDomain,
    ParameterUnitRelation,
    fit_model_catalog,
    fit_model_definition,
)
from .fit_problem import (
    bind_fit,
    validate_fit_result_source_binding,
)
from .schema import DatasetSchema
from .selection import (
    CoordinateRangeSelection,
    IndexRangeSelection,
    Selection,
)
from .transform import (
    CommittedTransform,
    DataTransformSpec,
    commit_transform,
    resolve_transformed_schema,
)


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
                and axis.role in requirement
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


def _unbound_fit_spec_for(
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
        if axis.axis_id not in resolved_fit_axis_ids and axis.role != SCALAR
    )
    return FitSpec(
        input_schema_fingerprint=schema.fingerprint,
        committed_transform=committed_transform,
        fit_axis_ids=resolved_fit_axis_ids,
        batch_axis_ids=batch_axis_ids,
        model_id=model.model_id,
        constraints=constraints,
        numeric_policy=numeric_policy,
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
    """Build and validate one total named-axis Fit request."""
    spec = _unbound_fit_spec_for(
        schema,
        model_id,
        committed_transform=committed_transform,
        fit_axis_ids=fit_axis_ids,
        constraints=constraints,
        numeric_policy=numeric_policy,
    )
    return bind_fit(spec, schema).spec


def suggest_fit_draft(
    schema: DatasetSchema,
    model_id: str,
    *,
    fit_axis_ids: tuple[AxisId, ...],
    selection: Selection | None = None,
    constraints: tuple[FitParameterConstraint, ...] = (),
    numeric_policy: FitNumericPolicy = FitNumericPolicy(),
) -> BoundFit:
    """Build one authoritative Fit draft from named axes and an optional ROI.

    This is deliberately a data-owned pure function.  It accepts no view or
    display-reduction state: every non-fit axis remains a named batch unless an
    independently authored authority transform says otherwise.  The only
    selector-derived transform admitted here is one range-preserving Selection
    over axes that the caller explicitly designated as fit axes.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    resolved_fit_axis_ids = tuple(fit_axis_ids)
    if not resolved_fit_axis_ids or any(
        not isinstance(axis_id, AxisId) for axis_id in resolved_fit_axis_ids
    ):
        raise ValueError("fit_axis_ids must explicitly contain named AxisId values")
    if len(set(resolved_fit_axis_ids)) != len(resolved_fit_axis_ids):
        raise ValueError("fit_axis_ids must be unique")

    preliminary_spec = _unbound_fit_spec_for(
        schema,
        model_id,
        fit_axis_ids=resolved_fit_axis_ids,
        constraints=constraints,
        numeric_policy=numeric_policy,
    )
    # Reject an incompatible model before constructing selection-derived
    # transformed metadata.  The temporary binding is released immediately.
    bind_fit(preliminary_spec, schema)

    committed_transform = None
    if selection is not None:
        if not isinstance(selection, Selection):
            raise TypeError("selection must be zlc_data.Selection or None")
        if any(
            not isinstance(term, (IndexRangeSelection, CoordinateRangeSelection))
            for term in selection.terms
        ):
            raise ValueError(
                "Fit draft selection supports only range-preserving terms"
            )
        selected_axis_ids = {term.axis_id for term in selection.terms}
        if not selected_axis_ids <= set(resolved_fit_axis_ids):
            raise ValueError(
                "Fit draft selection may name only explicit fit axes"
            )
        committed_transform = commit_transform(
            schema,
            DataTransformSpec((selection,)),
        )

    spec = _unbound_fit_spec_for(
        schema,
        model_id,
        committed_transform=committed_transform,
        fit_axis_ids=resolved_fit_axis_ids,
        constraints=constraints,
        numeric_policy=numeric_policy,
    )
    return bind_fit(spec, schema)


__all__ = [
    "BoundFit",
    "FitBatchStatus",
    "FitCancelled",
    "FitCoordinateSource",
    "FitDeadlineExceeded",
    "FitModelDefinition",
    "FitNumericPolicy",
    "FitParameterConstraint",
    "FitParameterDefinition",
    "FitParameterDomain",
    "FitResultBatch",
    "FitSpec",
    "ParameterUnitRelation",
    "bind_fit",
    "decode_fit_result_batch",
    "decode_fit_spec",
    "encode_fit_result_batch",
    "encode_fit_spec",
    "fit_model_catalog",
    "fit_model_definition",
    "fit_spec_from_tree",
    "fit_spec_for",
    "suggest_fit_draft",
    "fit_spec_to_tree",
    "validate_fit_result_source_binding",
]
