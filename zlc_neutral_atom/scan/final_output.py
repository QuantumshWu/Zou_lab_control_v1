"""Named FINAL Dataset publication for committed pulse scans."""

from __future__ import annotations

from zlc_neutral_atom.dataset_output import (
    FinalDatasetOutput,
    final_dataset_join_digest,
)

from .reference import scan_artifact_ref_to_tree
from .repository import MaterializedScanData


PULSE_SCAN_FINAL_OUTPUT_NAMES = ("scan",)


def scan_final_outputs(
    materialized: MaterializedScanData,
) -> dict[str, FinalDatasetOutput]:
    """Publish the canonical scan Dataset without a GUI-side materializer."""

    if not isinstance(materialized, MaterializedScanData):
        raise TypeError("materialized must be MaterializedScanData")
    snapshot = materialized.snapshot
    output_name = PULSE_SCAN_FINAL_OUTPUT_NAMES[0]
    output = FinalDatasetOutput(
        output_name,
        snapshot,
        final_dataset_join_digest(
            owner="pulse-scan",
            output_name=output_name,
            source_identity=scan_artifact_ref_to_tree(materialized.artifact_ref),
            snapshot=snapshot,
        ),
    )
    return {output.name: output}


__all__ = ["PULSE_SCAN_FINAL_OUTPUT_NAMES", "scan_final_outputs"]
