"""Named FINAL Dataset publication for committed pulse scans."""

from __future__ import annotations

from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
)
from zlc_neutral_atom.artifact_output import ArtifactOutputDeclaration

from .artifact import MaterializedScanData
from .reference import SCAN_ARTIFACT_REF_SCHEMA


PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration("scan", "zlc_neutral_atom.pulse-scan.final"),
)
PULSE_SCAN_ARTIFACT_OUTPUT_DECLARATION = ArtifactOutputDeclaration(
    "scan_artifact",
    SCAN_ARTIFACT_REF_SCHEMA,
)


def scan_final_outputs(
    materialized: MaterializedScanData,
) -> dict[str, FinalDatasetOutput]:
    """Publish the canonical scan Dataset without a GUI-side materializer."""

    if not isinstance(materialized, MaterializedScanData):
        raise TypeError("materialized must be MaterializedScanData")
    snapshot = materialized.snapshot
    declaration = PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS[0]
    output = FinalDatasetOutput(
        declaration,
        snapshot,
    )
    return {output.name: output}


__all__ = [
    "PULSE_SCAN_ARTIFACT_OUTPUT_DECLARATION",
    "PULSE_SCAN_FINAL_OUTPUT_DECLARATIONS",
    "scan_final_outputs",
]
