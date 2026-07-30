"""Public headless fit API assembled from the focused fit submodules."""

from .axis import AxisSourceRef
from .fit_solver import (
    BimodalDistributionAnalysis,
    analyze_bimodal_distribution,
)
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
    evaluate_fit_model_components,
    fit_model_catalog,
    fit_model_definition,
    histogram_gaussian_display_diagnostic,
)
from .fit_problem import bind_fit, validate_fit_result_source_binding
from .schema import DatasetSchema
from .selection import CoordinateRangeSelection, IndexRangeSelection, Selection
from .transform import CommittedTransform, DataTransformSpec, commit_transform


def fit_spec_for(
    schema: DatasetSchema,
    model_id: str,
    *,
    independent_sources: tuple[AxisSourceRef, ...],
    batch_sources: tuple[AxisSourceRef, ...] = (),
    committed_transform: CommittedTransform | None = None,
    constraints: tuple[FitParameterConstraint, ...] = (),
    numeric_policy: FitNumericPolicy = FitNumericPolicy(),
) -> FitSpec:
    """Build and bind one explicit source-based Fit request."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    transform = (
        commit_transform(schema, DataTransformSpec())
        if committed_transform is None
        else committed_transform
    )
    spec = FitSpec(
        committed_transform=transform,
        independent_sources=tuple(independent_sources),
        batch_sources=tuple(batch_sources),
        model_id=model_id,
        constraints=constraints,
        numeric_policy=numeric_policy,
    )
    return bind_fit(spec, schema).spec


def suggest_fit_draft(
    schema: DatasetSchema,
    model_id: str,
    *,
    independent_sources: tuple[AxisSourceRef, ...],
    batch_sources: tuple[AxisSourceRef, ...] = (),
    selection: Selection | None = None,
    point_ordinals: tuple[int, ...] | None = None,
    constraints: tuple[FitParameterConstraint, ...] = (),
    numeric_policy: FitNumericPolicy = FitNumericPolicy(),
) -> BoundFit:
    """Freeze one explicit Fit draft; display state is never copied implicitly."""

    independent = tuple(independent_sources)
    if selection is not None:
        if not isinstance(selection, Selection):
            raise TypeError("selection must be zlc_data.Selection or None")
        if any(
            not isinstance(term, (IndexRangeSelection, CoordinateRangeSelection))
            for term in selection.terms
        ):
            raise ValueError("Fit selection supports only range-preserving terms")
        tensor_ids = {
            source.axis_id
            for source in independent
            if source.kind == AxisSourceRef.TENSOR
        }
        if any(term.axis_id not in tensor_ids for term in selection.terms):
            raise ValueError(
                "Fit selection may name only explicit tensor independent sources"
            )
    operations = () if selection is None else (selection,)
    transform = commit_transform(
        schema,
        DataTransformSpec(operations),
        point_ordinals=point_ordinals,
    )
    return bind_fit(
        FitSpec(
            committed_transform=transform,
            independent_sources=independent,
            batch_sources=tuple(batch_sources),
            model_id=model_id,
            constraints=constraints,
            numeric_policy=numeric_policy,
        ),
        schema,
    )


__all__ = [
    "BimodalDistributionAnalysis",
    "BoundFit",
    "FitBatchStatus",
    "FitCancelled",
    "FitDeadlineExceeded",
    "FitModelDefinition",
    "FitNumericPolicy",
    "FitParameterConstraint",
    "FitParameterDefinition",
    "FitParameterDomain",
    "FitResultBatch",
    "FitSpec",
    "ParameterUnitRelation",
    "analyze_bimodal_distribution",
    "bind_fit",
    "decode_fit_result_batch",
    "decode_fit_spec",
    "encode_fit_result_batch",
    "encode_fit_spec",
    "evaluate_fit_model_components",
    "fit_model_catalog",
    "fit_model_definition",
    "fit_spec_for",
    "fit_spec_from_tree",
    "fit_spec_to_tree",
    "histogram_gaussian_display_diagnostic",
    "suggest_fit_draft",
    "validate_fit_result_source_binding",
]
