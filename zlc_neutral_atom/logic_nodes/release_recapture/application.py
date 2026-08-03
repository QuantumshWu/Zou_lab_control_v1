"""Shared flat Run construction for the two release-recapture Measurements."""

from __future__ import annotations

from uuid import uuid4

from zlc_data import BlockId
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration, FinalDatasetOutput
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.release_recapture.pipeline import (
    ReleaseRecapturePipelineSpec,
)
from zlc_neutral_atom.logic_nodes.release_recapture.timing import (
    TriggeredReleaseRecaptureResult,
    compile_triggered_release_recapture_pipeline,
)
from zlc_neutral_atom.runtime.run import RunPlan
from zlc_neutral_atom.runtime.streams import StreamId


def compile_release_recapture(
    *,
    name: str,
    camera_binding,
    calibration: ResolvedCalibration,
    model_kind,
    per_site: bool,
    rf_port=None,
    rf_table=None,
) -> RunPlan:
    """Build the one physical pair-reduction Run without another lifecycle owner."""

    selected = calibration.artifact.select_model(model_kind)
    identity = uuid4().hex
    spec = ReleaseRecapturePipelineSpec(
        name,
        camera_binding,
        calibration,
        selected,
        per_site,
        StreamId(f"release-recapture-{identity}"),
        BlockId(f"release-recapture-{identity[:20]}"),
        rf_port=rf_port,
        rf_table=rf_table,
    )
    return compile_triggered_release_recapture_pipeline(spec)


def release_recapture_final_outputs(
    result: TriggeredReleaseRecaptureResult,
    declaration: DatasetOutputDeclaration,
) -> dict[str, FinalDatasetOutput]:
    if type(result) is not TriggeredReleaseRecaptureResult:
        raise TypeError("result must be TriggeredReleaseRecaptureResult")
    if not isinstance(declaration, DatasetOutputDeclaration):
        raise TypeError("declaration must be DatasetOutputDeclaration")
    output = FinalDatasetOutput(declaration, result.survival)
    return {output.name: output}


__all__ = ["compile_release_recapture", "release_recapture_final_outputs"]
