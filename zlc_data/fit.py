"""Public headless fit API assembled from the focused fit submodules."""

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
    FitFamily,
    FitModelDefinition,
    FitParameterDefinition,
    ParameterUnitRelation,
    evaluate_fit_model,
    fit_model_catalog,
    fit_model_definition,
)
from .fit_problem import bind_fit, build_fit_problem


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
    "FitAxisRequirement",
    "FitBatchStatus",
    "FitCancelled",
    "FitCoordinateSource",
    "FitDeadlineExceeded",
    "FitFamily",
    "FitModelDefinition",
    "FitNumericPolicy",
    "FitParameterConstraint",
    "FitParameterDefinition",
    "FitProblem",
    "FitResultBatch",
    "FitSpec",
    "ParameterUnitRelation",
    "bind_fit",
    "build_fit_problem",
    "decode_fit_result_batch",
    "decode_fit_spec",
    "encode_fit_result_batch",
    "encode_fit_spec",
    "evaluate_fit_model",
    "fit_model_catalog",
    "fit_model_definition",
    "fit_analysis",
    "fit_result_batch_from_tree",
    "fit_result_batch_to_tree",
    "fit_spec_from_tree",
    "fit_spec_to_tree",
    "resolve_parameter_units",
]
