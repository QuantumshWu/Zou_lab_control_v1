"""Minimal source identity contract for reactive Processor evaluations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from zlc_neutral_atom.dataset_output import LiveDatasetOutput


@runtime_checkable
class CausalProcessorEvaluation(Protocol):
    """One atomic output bundle paired with its parent by the worker lane."""

    outputs: Mapping[str, LiveDatasetOutput]


def require_causal_processor_evaluation(
    value: object,
) -> CausalProcessorEvaluation:
    """Validate the output bundle; its exact parent lives on the work token."""

    if not isinstance(value, CausalProcessorEvaluation):
        raise TypeError(
            "reactive Processor result must implement CausalProcessorEvaluation"
        )
    if not isinstance(value.outputs, Mapping) or not value.outputs:
        raise ValueError("Processor result must carry one atomic output mapping")
    outputs = dict(value.outputs)
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(output, LiveDatasetOutput)
        or output.name != name
        for name, output in outputs.items()
    ):
        raise TypeError("Processor outputs must be named LiveDatasetOutput values")
    # The worker lane retains the exact immutable source publication beside the
    # Future and supplies it when this bundle is published.  Asking the result
    # to echo the same source identity would create a second, forgeable lineage
    # owner without proving that the algorithm consumed that source.
    return value


__all__ = [
    "CausalProcessorEvaluation",
    "require_causal_processor_evaluation",
]
