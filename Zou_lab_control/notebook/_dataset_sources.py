"""Static application wiring for durable Dataset-producing artifact owners.

This module selects one of the two built-in source owners by its public typed
reference and immediately delegates.  It contains no artifact field names,
storage interpretation, dynamic registry, or fallback discovery.
"""

from __future__ import annotations

from collections.abc import Callable

from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.pulse_scan.reference import ScanArtifactRef


def project_final_dataset_source(
    services,
    reference: CaptureArtifactRef | ScanArtifactRef,
    *,
    materialize: bool,
    abort_check: Callable[[], None] | None = None,
) -> ArtifactDatasetSource:
    """Delegate exact Dataset projection to the reference's static owner."""

    if isinstance(reference, CaptureArtifactRef):
        return services.capture_repository.project_dataset_source(
            reference,
            materialize=materialize,
            abort_check=abort_check,
        )
    if isinstance(reference, ScanArtifactRef):
        return services.readout_resources.scan_repository.project_dataset_source(
            reference,
            materialize=materialize,
            abort_check=abort_check,
        )
    raise TypeError("Dataset source must be CaptureArtifactRef or ScanArtifactRef")


__all__ = ["project_final_dataset_source"]
