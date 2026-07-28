"""Thin Occupancy-domain projections into frontend-owned view values."""

from __future__ import annotations

from dataclasses import dataclass, replace

from zlc_data import IndexSelection, Selection, dataset_cell_value
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_frontend import automatic_figure_view
from zlc_frontend.figure import ViewIntent, ViewPreferences
from zlc_neutral_atom.logic_nodes.readout.occupancy.cell import ExactOccupancyCellSource
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import ResolvedOccupancy
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import (
    occupancy_artifact_output_name,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor_application import (
    PreparedOccupancyProcessor,
)
from zlc_neutral_atom.processing.signal_plane import SignalValue
from zlc_frontend.site_map_view import (
    SiteMapView,
    build_site_map_cell_view,
)
from zlc_storage import canonical_digest
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress


def _occupancy_cell_coherence_identity(
    artifact_identity: str,
    address: DatasetCellAddress,
) -> str:
    if not isinstance(address, DatasetCellAddress):
        raise TypeError("address must be DatasetCellAddress")
    return canonical_digest(
        {
            "owner": "zlc_neutral_atom.occupancy-cell",
            "artifact": artifact_identity,
            "repeat_index": address.repeat_index,
            "point_ordinal": address.point_ordinal,
        }
    )


@dataclass(frozen=True, slots=True)
class OccupancyFigureProjection:
    """One admitted Occupancy output ready for generic Figure suggestion."""

    source: ArtifactDatasetSource
    label: str
    default_intent: ViewIntent
    default_preferences: ViewPreferences | None

    def __post_init__(self) -> None:
        if not isinstance(self.source, ArtifactDatasetSource):
            raise TypeError("source must be ArtifactDatasetSource")
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
    cell_selection = Selection(
        (
            IndexSelection(
                domain.occupancy_schema.repeat_axis.axis_id,
                address.repeat_index,
            ),
        )
    )
    summary = (
        f"{domain.artifact_identity} | "
        f"source={domain.source_capture_identity} | "
        f"calibration={domain.calibration_identity}\n"
        f"model={domain.model_kind.value} | "
        f"revision={domain.occupancy_ref.revision.value} | "
        f"source generation={domain.source_generation.value} | "
        f"occupancy generation={domain.occupancy_generation.value} | "
        f"address=({address.repeat_index}, {address.point_ordinal}) | "
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
        cell_selection,
        site_axis=site_map.site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        site_geometry_identity=domain.calibration_identity,
        coherence_identity=_occupancy_cell_coherence_identity(
            domain.artifact_identity,
            address,
        ),
        run_id=domain.run_id,
        provenance_epoch_id=domain.provenance_epoch_id,
        summary=summary,
        presentation_kind="occupancy-cell",
    )


def project_occupancy_site_map(
    application: PreparedOccupancyProcessor,
    background: SignalValue,
    occupied: SignalValue,
) -> SiteMapView:
    """Project one exact parent/result publication with no revision side index."""

    if not isinstance(application, PreparedOccupancyProcessor):
        raise TypeError("application must be PreparedOccupancyProcessor")
    if not isinstance(background, SignalValue) or not isinstance(
        occupied,
        SignalValue,
    ):
        raise TypeError("Occupancy presentation requires exact SignalValues")
    source_schema = background.schema
    occupied_schema = occupied.schema
    if (
        source_schema.repeat_axis.size != 1
        or source_schema.point_table.row_count != 1
        or occupied_schema.repeat_axis.size != 1
        or occupied_schema.point_table.row_count != 1
    ):
        raise ValueError("live Occupancy SiteMap requires R=1 and P=1")
    source_value = dataset_cell_value(background.block, 0, 0)
    occupied_value = dataset_cell_value(occupied.block, 0, 0)
    selection = Selection(
        (IndexSelection(source_schema.repeat_axis.axis_id, 0),)
    )
    logical_point = (
        source_schema.grid_topology.row_to_cell[0]
        if source_schema.grid_topology is not None
        else (0,)
    )
    site_map = application.site_map
    calibration_identity = application.request.calibration_ref.target_ref
    return build_site_map_cell_view(
        source_value,
        background.snapshot.ref,
        occupied_value,
        occupied.snapshot.ref,
        selection,
        site_axis=site_map.site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        site_geometry_identity=calibration_identity,
        coherence_identity=occupied.join_digest,
        run_id=occupied.run_id,
        provenance_epoch_id=occupied.epoch_id,
        summary=(
            f"source run={occupied.run_id} | "
            f"calibration={calibration_identity} | "
            f"revision={background.snapshot.ref.revision.value} | "
            f"logical point={logical_point}"
        ),
        presentation_kind="occupancy-cell",
    )


def project_occupancy_figure(
    resolved: ResolvedOccupancy,
    *,
    output: str | None,
    materialize: bool,
) -> OccupancyFigureProjection:
    """Choose one explicit Figure output from an admitted Occupancy artifact."""

    if type(resolved) is not ResolvedOccupancy:
        raise TypeError("resolved must be an exact ResolvedOccupancy")
    selected_output = occupancy_artifact_output_name(output)
    artifact = resolved.artifact
    source = resolved.project_dataset_source(
        output=selected_output,
        materialize=materialize,
    )
    default_intent, default_preferences = automatic_figure_view(
        source.schema,
        prefer_meter=selected_output == "occupied",
    )
    return OccupancyFigureProjection(
        source=source,
        label=f"occupancy {selected_output} | {artifact.model_kind.value}",
        default_intent=default_intent,
        default_preferences=default_preferences,
    )


__all__ = [
    "OccupancyFigureProjection",
    "build_exact_occupancy_cell_view",
    "project_occupancy_site_map",
    "project_occupancy_figure",
]
