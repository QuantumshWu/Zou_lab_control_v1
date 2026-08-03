"""Project exact Occupancy facts through the sole :mod:`zlc_plot` API."""

from __future__ import annotations

import numpy as np

from zlc_data import (
    SPATIAL_X,
    SPATIAL_Y,
    BlockId,
    ComponentValidity,
    DatasetRevisionRef,
    materialize_value_dataset,
)
from zlc_plot import (
    AxisRef,
    ImageFrame,
    ImagePlot,
    ImagePointOverlay,
    PlotLabels,
    PlotSession,
    PointStatus,
    RasterPlotHost,
)

from ..cell import ExactOccupancyCellSource


def occupancy_cell_summary(source: ExactOccupancyCellSource) -> str:
    """Describe provenance without creating a second presentation model."""

    if not isinstance(source, ExactOccupancyCellSource):
        raise TypeError("source must be ExactOccupancyCellSource")
    domain = source.domain
    address = source.address
    metadata = source.frame_metadata
    return (
        f"{domain.artifact_identity} | source={domain.source_capture_identity} | "
        f"calibration={domain.calibration_identity}\n"
        f"model={domain.model_kind.value} | "
        f"revision={domain.occupancy_ref.revision.value} | "
        f"address=({address.repeat_index}, {address.point_ordinal}) | "
        f"logical point={source.logical_point}\n"
        f"frame ordinal={metadata.source_ordinal} | "
        f"frame_stamp={metadata.frame_stamp} | "
        f"camera_stamp={metadata.camera_stamp} | "
        f"captured_at={metadata.captured_at:.9f}s"
    )


def _cell_image_snapshot(source: ExactOccupancyCellSource):
    address = source.address
    block_id = BlockId(
        f"occupancy-cell-frame-r{address.repeat_index}-p{address.point_ordinal}"
    )

    def reference_for(schema) -> DatasetRevisionRef:
        ref = source.domain.source_ref
        return DatasetRevisionRef(
            block_id,
            ref.stream_generation,
            schema.fingerprint,
            ref.revision,
        )

    return materialize_value_dataset(
        source.domain.source_ref,
        source.image,
        reference_for=reference_for,
    )


def _site_overlay(source: ExactOccupancyCellSource) -> ImagePointOverlay:
    site_map = source.domain.site_map
    occupied = source.occupied
    validity = occupied.validity
    if not isinstance(validity, ComponentValidity):
        raise TypeError("Occupancy cell requires per-site validity")
    values = np.asarray(occupied.values, dtype=np.bool_)
    valid = np.asarray(validity.mask, dtype=np.bool_)
    statuses = tuple(
        PointStatus.INVALID
        if not bool(valid[index])
        else PointStatus.OCCUPIED
        if bool(values[index])
        else PointStatus.EMPTY
        for index in range(site_map.site_axis.size)
    )
    return ImagePointOverlay(
        revision=source.domain.occupancy_ref.revision.value,
        coordinates=site_map.coordinates_xy,
        point_ids=tuple(
            f"site-{index}" for index in range(site_map.site_axis.size)
        ),
        labels=tuple(
            str(site_map.site_axis.coordinate_at(index))
            for index in range(site_map.site_axis.size)
        ),
        statuses=statuses,
    )


def occupancy_cell_plot(
    source: ExactOccupancyCellSource,
) -> tuple[ImageFrame, ImagePlot]:
    """Return one zlc_plot input/spec pair for an admitted exact cell."""

    if not isinstance(source, ExactOccupancyCellSource):
        raise TypeError("source must be ExactOccupancyCellSource")
    axes = source.image.schema.data_axes
    x_axes = tuple(axis for axis in axes if axis.role == SPATIAL_X)
    y_axes = tuple(axis for axis in axes if axis.role == SPATIAL_Y)
    if len(axes) != 2 or len(x_axes) != 1 or len(y_axes) != 1:
        raise ValueError(
            "Occupancy cell image requires one SPATIAL_X and SPATIAL_Y axis"
        )
    return (
        ImageFrame(_cell_image_snapshot(source), _site_overlay(source)),
        ImagePlot(
            AxisRef.data(x_axes[0].axis_id.value),
            AxisRef.data(y_axes[0].axis_id.value),
            labels=PlotLabels(
                title="Occupancy | exact same-shot sites",
                value="Occupancy",
            ),
        ),
    )


def occupancy_cell_session(source: ExactOccupancyCellSource) -> PlotSession:
    """Create the headless/session view returned by the public Occupancy API."""

    data, spec = occupancy_cell_plot(source)
    return PlotSession(data, spec)


def occupancy_cell_host(source: ExactOccupancyCellSource) -> RasterPlotHost:
    """Create the worker-owned surface used by the optional Qt navigator."""

    data, spec = occupancy_cell_plot(source)
    return RasterPlotHost.from_plot(data, spec)


__all__ = [
    "occupancy_cell_host",
    "occupancy_cell_plot",
    "occupancy_cell_session",
    "occupancy_cell_summary",
]
