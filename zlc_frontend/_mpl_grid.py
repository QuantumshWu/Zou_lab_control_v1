"""Internal Matplotlib implementation owner: grid."""

from __future__ import annotations

import matplotlib
from zlc_data import FitBatchStatus, FitResultBatch
from .figure import (
    EvaluatedAxis,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    FigureDocument,
)
from .fit_projection import (
    evaluated_figure_panels as _panels,
    panel_focus_address as _panel_focus_address,
    fit_batch_storage_index as _batch_storage_index,
)
from .data_figure import FigurePanelRegion
from .plot_layout import (
    grid_shape_for,
    grid_shape_for_aspect,
    image_panel_layout,
    image_panel_layout_for_raster,
    LIVE_PANEL_DPI,
    optimal_grid_size_for_cells,
    panel_data_box,
    panel_data_box_for_raster,
    panel_figure_size_inches,
    rolling_panel_layout,
    rolling_panel_layout_for_raster,
    site_grid_geometry,
)


def _figure_panel_regions(
    figure,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
) -> tuple[FigurePanelRegion, ...]:
    panels = _panels(evaluated)
    axes = tuple(figure.axes[: len(panels)])
    if len(axes) != len(panels):
        raise RuntimeError("rendered figure lost one or more data-panel axes")
    regions = []
    for index, (axis, (layer, cell, series_group)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        bounds = axis.get_position()
        fit_result = fit_results.get(layer.layer_id)
        focus_address = _panel_focus_address(
            layer,
            cell,
            series_group,
        )
        fit_storage_index = None
        if fit_result is not None and len(series_group) == 1:
            fit_storage_index = _batch_storage_index(
                fit_result,
                layer,
                cell,
                series_group[0],
            )
        regions.append(
            FigurePanelRegion(
                f"panel-{index}",
                focus_address,
                fit_storage_index,
                float(bounds.x0),
                float(1.0 - bounds.y1),
                float(bounds.x1),
                float(1.0 - bounds.y0),
            )
        )
    return tuple(regions)

def _live_grid_axes(
    figure,
    *,
    size: str,
    cell_count: int,
    cell_aspect: float | None = None,
) -> tuple[tuple[object, ...], int, int, float]:
    """Build main-equivalent fixed-box grid axes for one live panel."""

    import matplotlib

    tick_points = float(matplotlib.rcParams["xtick.labelsize"])
    rows, columns = grid_shape_for(cell_count)
    # Main's recommendation belongs to facet cardinality, before an IMAGE's
    # fixed aspect repacks those same cells.  Keep that one recommendation for
    # both the initial size policy and the two-tier cell-font comparison;
    # deriving it again from renderer packing would create a second owner.
    recommended = optimal_grid_size_for_cells(cell_count)
    if cell_aspect is not None:
        region_px, _column_gap, _row_gap, _margins, _font_scale = (
            site_grid_geometry(
                size,
                recommended,
                tick_font_points=tick_points,
            )
        )
        rows, columns = grid_shape_for_aspect(
            cell_count,
            cell_aspect,
            region_px,
        )
    data_px, column_gap, row_gap, margins, font_scale = site_grid_geometry(
        size,
        recommended,
        tick_font_points=tick_points,
    )
    data_width, data_height = data_px
    left, right, bottom, top = margins
    cell_width = max(
        (data_width - (columns - 1) * column_gap) / columns,
        1.0,
    )
    cell_height = max(
        (data_height - (rows - 1) * row_gap) / rows,
        1.0,
    )
    figure_width = left + data_width + right
    figure_height = bottom + data_height + top
    axes = []
    for row in range(rows):
        for column in range(columns):
            x = left + column * (cell_width + column_gap)
            y = bottom + (rows - 1 - row) * (cell_height + row_gap)
            axes.append(
                figure.add_axes(
                    (
                        x / figure_width,
                        y / figure_height,
                        cell_width / figure_width,
                        cell_height / figure_height,
                    )
                )
            )
    return tuple(axes), rows, columns, font_scale

def _live_grid_cell_title(cell, series_group) -> str:
    """Return Main's role-authored facet identifier."""

    from .fit_projection import address_label

    addresses = tuple(cell.facet_address)
    if len(series_group) == 1:
        addresses = (*addresses, *series_group[0].batch_address)
    return address_label(addresses)

__all__ = [
]
