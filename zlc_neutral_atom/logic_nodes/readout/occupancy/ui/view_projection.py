"""The single Occupancy adapter into frontend-owned Figure contracts."""

from __future__ import annotations

from zlc_data import IndexSelection, Selection
from zlc_data.value import dataset_cell_value
from zlc_frontend.figure_source import FigureSource
from zlc_frontend.frozen_figure import (
    FrozenFigureSource,
    resolve_frozen_figure_intent,
)
from zlc_frontend.plot_kind import PlotKind
from zlc_frontend.plot_panel import FigureIntent
from zlc_frontend.site_map_view import SiteMapView, build_site_map_cell_view
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.logic_nodes.readout.occupancy.cell import (
    ExactOccupancyCellSource,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import (
    OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION,
    ResolvedOccupancy,
    occupancy_artifact_output_name,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor_application import (
    PreparedOccupancyProcessor,
)
from zlc_neutral_atom.processing.hosted_processor import HostedProcessor
from zlc_neutral_atom.processing.signal_plane import SignalPublication, SignalValue


OCCUPANCY_SITE_MAP_FIGURE = FigureIntent(
    PlotKind.SITE_MAP,
    "Occupancy | exact same-shot sites",
    "Occupancy",
)


def _build_exact_occupancy_cell_view(
    source: ExactOccupancyCellSource,
) -> SiteMapView:
    """Project one admitted exact cell through frontend's SiteMap contract."""

    if not isinstance(source, ExactOccupancyCellSource):
        raise TypeError("source must be ExactOccupancyCellSource")
    domain = source.domain
    site_map = domain.site_map
    metadata = source.frame_metadata
    address = source.address
    selection = Selection(
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
        f"source generation={domain.source_ref.stream_generation.value} | "
        f"occupancy generation={domain.occupancy_ref.stream_generation.value} | "
        f"address=({address.repeat_index}, {address.point_ordinal}) | "
        f"logical_point={source.logical_point}\n"
        f"frame ordinal={metadata.source_ordinal} | "
        f"frame_stamp={metadata.frame_stamp} | "
        f"camera_stamp={metadata.camera_stamp} | "
        f"captured_at={metadata.captured_at:.9f}s"
    )
    return build_site_map_cell_view(
        source.image,
        domain.source_ref,
        source.occupied,
        domain.occupancy_ref,
        selection,
        site_axis=site_map.site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        summary=summary,
        presentation_kind="occupancy-cell",
    )


def project_exact_occupancy_cell(
    source: ExactOccupancyCellSource,
) -> tuple[FigureIntent, FigureSource]:
    """Pair one exact cell with the generic Figure source consumed by every host."""

    view = _build_exact_occupancy_cell_view(source)
    return OCCUPANCY_SITE_MAP_FIGURE, FigureSource(source.occupied_snapshot, view)


def project_occupancy_site_map(
    application: PreparedOccupancyProcessor,
    background: SignalValue,
    occupied: SignalValue,
) -> tuple[FigureIntent, SiteMapView]:
    """Project one exact parent/result publication without a revision side index."""

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
    view = build_site_map_cell_view(
        source_value,
        background.snapshot.ref,
        occupied_value,
        occupied.snapshot.ref,
        selection,
        site_axis=site_map.site_axis,
        coordinate_frame=site_map.coordinate_frame,
        centers_xy=site_map.coordinates_xy,
        summary=(
            f"calibration={calibration_identity} | "
            f"generation={occupied.snapshot.ref.stream_generation.value} | "
            f"revision={background.snapshot.ref.revision.value} | "
            f"logical point={logical_point}"
        ),
        presentation_kind="occupancy-cell",
    )
    return OCCUPANCY_SITE_MAP_FIGURE, view


def project_occupancy_signal_presentation(
    node: object,
    output_name: str,
    publication: SignalPublication,
    direct_parents: tuple[SignalPublication, ...],
):
    """Project one same-shot Occupancy SiteMap from its exact signal front."""

    if output_name != OCCUPANCY_SITE_MAP_OUTPUT_DECLARATION.name:
        return None
    if not isinstance(node, HostedProcessor):
        raise TypeError("Occupancy presentation requires HostedProcessor")
    parent = direct_parents[0] if len(direct_parents) == 1 else None
    background = None if parent is None else parent.value(node.source_signal)
    occupied = publication.value(node.signal_key(output_name))
    if background is None or occupied is None:
        return None
    return project_occupancy_site_map(
        node.prepared_application,
        background,
        occupied,
    )


def project_occupancy_figure(
    resolved: ResolvedOccupancy,
    *,
    output: str | None,
    materialize: bool,
) -> tuple[ArtifactDatasetSource, FigureIntent]:
    """Project one admitted Occupancy artifact through the canonical Figure path."""

    if type(resolved) is not ResolvedOccupancy:
        raise TypeError("resolved must be an exact ResolvedOccupancy")
    selected_output = occupancy_artifact_output_name(output)
    artifact = resolved.artifact
    source = resolved.project_dataset_source(
        output=selected_output,
        materialize=materialize,
    )
    label = f"occupancy {selected_output} | {artifact.model_kind.value}"
    figure = resolve_frozen_figure_intent(
        FrozenFigureSource(
            label,
            source.schema,
            source.ref,
            snapshot=source.snapshot,
        ),
        title=label,
        value_label=selected_output,
    )
    return source, figure


__all__ = [
    "project_exact_occupancy_cell",
    "project_occupancy_figure",
    "project_occupancy_signal_presentation",
]
