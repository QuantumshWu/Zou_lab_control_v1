"""Minimal source identity contract for reactive Processor evaluations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from zlc_data import DatasetRevisionRef
from zlc_neutral_atom.dataset_output import LiveDatasetOutput
from zlc_storage import canonical_digest, sha256_text


def derive_dataset_event_digest(
    source_event_digest: str,
    derivation_digest: str,
) -> str:
    """Create the sole causal identity for a deterministic derived Dataset."""

    source = sha256_text(source_event_digest, "source_event_digest")
    derivation = sha256_text(derivation_digest, "derivation_digest")
    assert source is not None and derivation is not None
    return canonical_digest(
        {
            "owner": "zlc_neutral_atom.processing.derived-dataset-event",
            "source_event_digest": source,
            "derivation_digest": derivation,
        }
    )


@runtime_checkable
class CausalProcessorEvaluation(Protocol):
    """One atomic publication that names the immutable event it consumed."""

    @property
    def source_ref(self) -> DatasetRevisionRef: ...

    source_event_digest: str
    outputs: Mapping[str, LiveDatasetOutput]


def require_causal_processor_evaluation(
    value: object,
    *,
    source_ref: DatasetRevisionRef,
    source_event_digest: str,
) -> CausalProcessorEvaluation:
    """Reject a result that cannot prove which admitted source produced it."""

    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("source_ref must be DatasetRevisionRef")
    digest = sha256_text(source_event_digest, "source_event_digest")
    assert digest is not None
    if not isinstance(value, CausalProcessorEvaluation):
        raise TypeError(
            "reactive Processor result must implement CausalProcessorEvaluation"
        )
    if value.source_ref != source_ref or value.source_event_digest != digest:
        raise ValueError("Processor result belongs to another source revision/event")
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
    # A derived Dataset is a different stream and therefore owns its own
    # StreamGenerationId.  Equating that generation with the input stream used
    # to make every legitimate derived Processor publication impossible.  The
    # cross-stream causal proof is the exact source ref/event carried above;
    # each concrete Processor remains responsible for validating the shape,
    # revision and generation relationships among its own sibling outputs.
    if len({output.join_digest for output in outputs.values()}) != 1:
        raise ValueError("Processor outputs do not form one atomic causal join")
    return value


__all__ = [
    "CausalProcessorEvaluation",
    "derive_dataset_event_digest",
    "require_causal_processor_evaluation",
]
