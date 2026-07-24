"""The sole calibration-domain to frontend SiteMap composition seam."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zlc_frontend.site_map_render import (
    SiteMapView,
    build_calibration_site_map_view,
)
from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_neutral_atom.logic_nodes.calibration.projection import (
    CALIBRATION_FINAL_OUTPUT_DECLARATIONS,
)
from zlc_neutral_atom.logic_nodes.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.calibration.task import PreparedCalibrationTask

if TYPE_CHECKING:
    from zlc_neutral_atom.logic_nodes.calibration.application import (
        CalibrationArtifactRequest,
    )


def calibration_authority_summary(
    request: CalibrationArtifactRequest,
    previous_reference: CalibrationArtifactRef | None = None,
) -> str:
    """Describe frozen Calibration authority without duplicating its physics."""

    from zlc_neutral_atom.logic_nodes.calibration.application import (
        CalibrationArtifactRequest,
    )

    if not isinstance(request, CalibrationArtifactRequest):
        raise TypeError("request must be CalibrationArtifactRequest")
    if previous_reference is not None and not isinstance(
        previous_reference,
        CalibrationArtifactRef,
    ):
        raise TypeError("previous_reference must be CalibrationArtifactRef or None")
    analysis = request.analysis
    layout = analysis.layout
    centers = analysis.expected_centers_xy
    center_text = (
        "missing (formal Run will reject)"
        if centers is None
        else f"{analysis.site_count} independent centers"
    )
    residual_text = (
        "none"
        if analysis.maximum_site_residual_px is None
        else f"{analysis.maximum_site_residual_px:.6g} px"
    )
    previous = (
        "new calibration"
        if previous_reference is None
        else f"editing {previous_reference.target_ref} into a new artifact"
    )
    return (
        f"source={request.source_capture_ref.target_ref} · "
        f"binding={request.readout_binding.value} · {previous}\n"
        f"READOUT_EVENT={layout.readout_event_axis_id.value} · "
        f"reference={layout.reference_event_indices} · "
        f"readout={layout.readout_event_index}\n"
        f"grid={analysis.grid_shape_yx} {analysis.ordering.value} · "
        f"sites={analysis.site_count} · {center_text} · max residual={residual_text}\n"
        "Spatial authority is frozen; detector/display output cannot rewrite it."
    )


def materialize_calibration_final_presentations(
    command: PreparedCalibrationTask,
    result: CalibrationArtifactRef,
    outputs: Mapping[str, FinalDatasetOutput],
) -> dict[str, SiteMapView]:
    """Render a prepared task's typed FINAL context without redoing physics."""

    if not isinstance(command, PreparedCalibrationTask):
        raise TypeError("command must be PreparedCalibrationTask")
    if not isinstance(result, CalibrationArtifactRef):
        raise TypeError("result must be CalibrationArtifactRef")
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must be a mapping")

    context = command.site_map_context(result)
    declaration = CALIBRATION_FINAL_OUTPUT_DECLARATIONS[0]
    calibration_name = declaration.name
    output = outputs.get(calibration_name)
    if (
        not isinstance(output, FinalDatasetOutput)
        or output.declaration != declaration
    ):
        raise ValueError("calibration SiteMap requires the calibration output")
    snapshot = output.snapshot
    valid_count = int(context.site_validity.sum())
    view = build_calibration_site_map_view(
        snapshot,
        site_axis=context.site_axis,
        coordinate_frame=context.coordinate_frame,
        centers_xy=context.centers_xy,
        site_validity=context.site_validity,
        calibration_identity=context.calibration_identity,
        run_id=f"calibration-{output.join_digest}",
        provenance_epoch_id=snapshot.ref.stream_generation.value,
        summary=(
            f"{context.calibration_identity} | reference average | "
            f"valid sites={valid_count}/{context.site_axis.size}"
        ),
    )
    return {calibration_name: view}


__all__ = [
    "calibration_authority_summary",
    "materialize_calibration_final_presentations",
]
