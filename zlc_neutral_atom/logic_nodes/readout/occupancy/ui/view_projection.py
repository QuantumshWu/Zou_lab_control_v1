"""Thin Occupancy-domain projections into frontend-owned view values."""

from __future__ import annotations

from dataclasses import dataclass, replace

from zlc_data import (
    MONITOR_HISTORY,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    DatasetSchema,
    OwnedSnapshot,
)
from zlc_frontend.figure import RepeatViewMode, ViewIntent, ViewPreferences
from zlc_neutral_atom.logic_nodes.readout.occupancy.cell import ExactOccupancyCellSource
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import ResolvedOccupancy
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import (
    OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION,
    OccupancyProcessorEvaluation,
)
from zlc_frontend.site_map_view import (
    SiteMapView,
    build_site_map_cell_view,
)
from zlc_storage import canonical_text


@dataclass(frozen=True, slots=True)
class OccupancyFigureProjection:
    """One admitted Occupancy output ready for generic Figure suggestion."""

    schema: DatasetSchema
    snapshot: OwnedSnapshot | None
    label: str
    default_intent: ViewIntent
    default_preferences: ViewPreferences | None

    def __post_init__(self) -> None:
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if self.snapshot is not None and not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot or None")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be non-empty")
        if not isinstance(self.default_intent, ViewIntent):
            raise TypeError("default_intent must be ViewIntent")
        if self.default_preferences is not None and not isinstance(
            self.default_preferences,
            ViewPreferences,
        ):
            raise TypeError("default_preferences must be ViewPreferences or None")

    def resolve_preferences(
        self,
        intent: ViewIntent,
        requested: ViewPreferences | None,
    ) -> ViewPreferences | None:
        """Apply Occupancy's default only when the caller left it unspecified."""

        if not isinstance(intent, ViewIntent):
            raise TypeError("intent must be ViewIntent")
        if requested is not None and not isinstance(requested, ViewPreferences):
            raise TypeError("requested must be ViewPreferences or None")
        defaults = self.default_preferences
        if (
            defaults is None
            or intent is not self.default_intent
            or (requested is not None and requested.repeat_mode is not None)
        ):
            return requested
        return replace(
            ViewPreferences() if requested is None else requested,
            repeat_mode=defaults.repeat_mode,
        )


def build_exact_occupancy_cell_view(
    source: ExactOccupancyCellSource,
) -> SiteMapView:
    """Project one admitted same-shot cell without leaking fields into the facade."""

    if not isinstance(source, ExactOccupancyCellSource):
        raise TypeError("source must be ExactOccupancyCellSource")
    domain = source.domain
    site_map = domain.site_map
    metadata = source.frame_metadata
    address = source.address
    summary = (
        f"{domain.artifact_identity} | "
        f"source={domain.source_capture_identity} | "
        f"calibration={domain.calibration_identity}\n"
        f"model={domain.model_kind.value} | "
        f"revision={domain.occupancy_ref.revision.value} | "
        f"source generation={domain.source_generation.value} | "
        f"occupancy generation={domain.occupancy_generation.value} | "
        f"address=({address.repeat_index}, {address.point_storage_index}) | "
        f"logical_point={source.logical_point}\n"
        f"frame ordinal={metadata.source_ordinal} | "
        f"frame_stamp={metadata.frame_stamp} | "
        f"camera_stamp={metadata.camera_stamp} | "
        f"captured_at={metadata.captured_at:.9f}s | "
        f"correlation={metadata.correlation_id}"
    )
    return build_site_map_cell_view(
        source.image,
        domain.source_ref,
        source.occupied,
        domain.occupancy_ref,
        source.selection,
        site_axis=site_map.site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        site_geometry_identity=domain.calibration_identity,
        coherence_identity=domain.artifact_identity,
        run_id=domain.run_id,
        provenance_epoch_id=domain.provenance_epoch_id,
        summary=summary,
        presentation_kind="occupancy-cell",
    )


def project_occupancy_views(
    result: object,
    *,
    run_id: str,
    provenance_epoch_id: str,
) -> dict[str, SiteMapView]:
    """Map an already-closed evaluation into a frontend SiteMap view."""

    if not isinstance(result, OccupancyProcessorEvaluation):
        raise TypeError("Occupancy application returned another result type")
    run_id = canonical_text(run_id, "run_id")
    provenance_epoch_id = canonical_text(
        provenance_epoch_id,
        "provenance_epoch_id",
    )
    site_map = result.site_map
    calibration_identity = result.calibration_ref.target_ref
    occupied_name = OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION.name
    presentation = build_site_map_cell_view(
        result.background_value,
        result.background_ref,
        result.occupied_value,
        result.occupied_ref,
        result.selection,
        site_axis=site_map.site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        site_geometry_identity=calibration_identity,
        coherence_identity=result.outputs[occupied_name].join_digest,
        run_id=run_id,
        provenance_epoch_id=provenance_epoch_id,
        summary=(
            f"source run={run_id} | "
            f"calibration={calibration_identity} | "
            f"revision={result.background_ref.revision.value} | "
            f"logical point={result.logical_point}"
        ),
        presentation_kind="occupancy-cell",
    )
    return {occupied_name: presentation}


def project_occupancy_figure(
    resolved: ResolvedOccupancy,
    *,
    output: str | None,
    materialize: bool,
) -> OccupancyFigureProjection:
    """Choose one explicit Figure output from an admitted Occupancy artifact."""

    if type(resolved) is not ResolvedOccupancy:
        raise TypeError("resolved must be an exact ResolvedOccupancy")
    selected_output = "occupied" if output is None else str(output)
    if selected_output not in ("occupied", "counts"):
        raise ValueError("occupancy output must be 'occupied' or 'counts'")
    artifact = resolved.artifact
    if selected_output == "occupied":
        block = artifact.occupied
        snapshot = artifact.occupied_snapshot if materialize else None
    else:
        block = artifact.counts
        snapshot = artifact.counts_snapshot if materialize else None
    schema = block.schema
    roles = {
        axis.role
        for axis in (
            schema.repeat_axis,
            *schema.point_axes,
            *schema.cell_schema.data_axes,
        )
    }
    if selected_output == "occupied":
        default_intent = (
            ViewIntent.CURVE
            if roles.intersection((SCAN_POINT, SPECTRAL, MONITOR_HISTORY))
            else ViewIntent.METER
        )
        default_preferences = (
            ViewPreferences(repeat_mode=RepeatViewMode.MEAN)
            if default_intent is ViewIntent.METER
            else None
        )
    elif SPATIAL_X in roles and SPATIAL_Y in roles:
        default_intent = ViewIntent.IMAGE
        default_preferences = None
    elif roles.intersection((SCAN_POINT, SPECTRAL, MONITOR_HISTORY)):
        default_intent = ViewIntent.CURVE
        default_preferences = None
    else:
        default_intent = ViewIntent.HISTOGRAM
        default_preferences = None
    return OccupancyFigureProjection(
        schema=schema,
        snapshot=snapshot,
        label=f"occupancy {selected_output} | {artifact.model_kind.value}",
        default_intent=default_intent,
        default_preferences=default_preferences,
    )


__all__ = [
    "OccupancyFigureProjection",
    "build_exact_occupancy_cell_view",
    "project_occupancy_views",
    "project_occupancy_figure",
]
