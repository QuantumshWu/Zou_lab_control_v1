"""Thin Calibration-domain projections into frontend-owned view values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zlc_data.value import OwnedSnapshot
from zlc_frontend.figure_source import FigureSource
from zlc_frontend.plot_kind import PlotKind
from zlc_frontend.plot_panel import FigureIntent
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.readout.calibration.projection import (
    CalibrationSiteMapContext,
)
from zlc_frontend.site_map_view import (
    build_site_map_snapshot_view,
)

if TYPE_CHECKING:
    from zlc_neutral_atom.logic_nodes.readout.calibration.application import (
        CalibrationArtifactRequest,
    )


_SITE_MAP_FIGURE = FigureIntent(
    PlotKind.SITE_MAP,
    "Reference average | calibrated sites",
    "Counts",
)


def calibration_authority_summary(
    request: CalibrationArtifactRequest,
    previous_reference: CalibrationArtifactRef | None = None,
) -> str:
    """Describe frozen Calibration authority without duplicating its physics."""

    from zlc_neutral_atom.logic_nodes.readout.calibration.application import (
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


def calibration_site_map_figure(
    snapshot: OwnedSnapshot,
    context: CalibrationSiteMapContext,
    *,
    coherence_identity: str,
    run_id: str,
    provenance_epoch_id: str,
) -> tuple[FigureIntent, FigureSource]:
    """Return Calibration's sole semantic SiteMap intent and exact source.

    Live panels and immutable reports supply only lineage facts.  The visible
    title, axis label, SiteMap projection, and default display all originate
    from this one leaf intent plus the frontend PlotPanel compiler.
    """

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if not isinstance(context, CalibrationSiteMapContext):
        raise TypeError("context must be CalibrationSiteMapContext")
    valid_count = int(context.site_validity.sum())
    view = build_site_map_snapshot_view(
        snapshot,
        site_axis=context.site_axis,
        coordinate_frame=context.coordinate_frame,
        centers_xy=context.centers_xy,
        site_validity=context.site_validity,
        site_geometry_identity=context.calibration_identity,
        coherence_identity=coherence_identity,
        run_id=run_id,
        provenance_epoch_id=provenance_epoch_id,
        summary=(
            f"{context.calibration_identity} | reference average | "
            f"valid sites={valid_count}/{context.site_axis.size}"
        ),
        presentation_kind="calibration-sites",
    )
    return _SITE_MAP_FIGURE, FigureSource(snapshot, view)


__all__ = [
    "calibration_authority_summary",
    "calibration_site_map_figure",
]
