"""Node-neutral application projection of durable Dataset artifacts."""

from __future__ import annotations

from collections.abc import Callable

from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.artifact_dispatch import ArtifactDispatch


def project_final_dataset_source(
    artifacts: ArtifactDispatch,
    reference: object,
    *,
    materialize: bool,
    abort_check: Callable[[], None] | None = None,
) -> ArtifactDatasetSource:
    """Ask the single frozen owner dispatch for an exact Dataset source."""

    if not isinstance(artifacts, ArtifactDispatch):
        raise TypeError("artifacts must be ArtifactDispatch")
    return artifacts.project_dataset(
        reference,
        materialize=materialize,
        abort_check=abort_check,
    )


__all__ = ["project_final_dataset_source"]
