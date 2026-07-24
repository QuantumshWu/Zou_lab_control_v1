"""Internal Matplotlib implementation owner: document."""

from __future__ import annotations

import gc
from io import BytesIO
import math
import numpy as np
from zlc_data import FitBatchStatus, FitResultBatch
from zlc_storage import positive_integer
from .figure import (
    EvaluatedAxis,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    FigureDocument,
)
from .fit_image_projection import (
    RadialGaussianImageFitPanel,
    address_label as _address_label,
    evaluated_figure_panels as _panels,
    figure_panel_title as _panel_title,
    fit_batch_storage_index as _batch_storage_index,
    fit_panel_selection as _fit_panel_selection,
    panel_focus_selection as _panel_focus_selection,
    radial_gaussian_fit_geometry,
    reduction_label as _reduction_label,
)
from .image_display import (
    ImageDisplayState,
    evaluated_image_data_range,
    image_viewport_for_display_state,
)
from .image_view import image_viewport_for_evaluated_image
from .histogram_display import (
    DEFAULT_HISTOGRAM_BINS,
    FacetedHistogramDisplayState,
    HistogramBinProjection,
    HistogramCountScale,
    HistogramDisplayState,
    HistogramFitMode,
    HistogramViewportTransform,
    _WindowedHistogramProjection,
    histogram_count_limits,
    histogram_home_x_limits,
)
from .data_figure import FigurePanelRegion
from .display_range import (
    RelimMode,
    deadband_display_range,
    validated_display_range,
)
from .plot_layout import (
    grid_shape_for,
    grid_shape_for_aspect,
    image_panel_layout,
    image_panel_layout_for_raster,
    LIVE_PANEL_DPI,
    optimal_grid_size,
    panel_data_box,
    panel_data_box_for_raster,
    panel_figure_size_inches,
    rolling_panel_layout,
    rolling_panel_layout_for_raster,
    site_grid_geometry,
)
from .render_style import (
    ANNOTATION_FONT_SIZE,
    CURVE_LINESTYLE,
    CURVE_MARKER,
    FIT_CONTOUR_COLOR,
    FIT_CONTOUR_LINEWIDTH,
    FIT_FAILURE_COLOR,
    FIT_LINESTYLE,
    HIST_FILL_ALPHA,
    LINE_CYCLE,
    PALETTE,
    SITE_OCCUPANCY_STYLE,
    apply_title,
    axis_label_fontsize,
    bimodal_fit_line_specs,
    render_style_context,
    small_fontsize,
    threshold_line_kwargs,
    tick_fontsize,
)

from ._mpl_common import (
    _render_dpi,
    _require_evaluated_identity,
    raster_from_agg,
    release_agg_figure,
)
from ._mpl_curve import (
    _curve,
    _meter,
    _shared_curve_limits,
)
from ._mpl_histogram import (
    _draw_histogram_projection,
    _grid_histogram_value_range,
    _histogram,
    _update_histogram_presentation,
)
from ._mpl_image import (
    _draw_projected_image,
    _image,
    _radial_image_color_limits_by_layer,
)
from ._mpl_grid import (
    _figure_panel_regions,
    _live_grid_axes,
    _live_grid_cell_title,
)

def render_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    dpi: float = 100.0,
):
    """Construct a caller-owned Figure with canonical artist styles frozen in.

    Later caller-driven draw/save operations are outside the product compose lane.  Product PNG
    and file export must use :func:`save_evaluated_figure`, which keeps construction *and* Agg
    draw under the serialized style context.
    """

    dpi = _render_dpi(dpi)
    with render_style_context():
        return _render_evaluated_figure(document, evaluated, fit_results, dpi=dpi)

def save_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    destination,
    *,
    image_format: str,
    dpi: float,
) -> None:
    """Construct, draw, and release one product figure on the Matplotlib compose lane."""

    _save_evaluated_figure(
        document,
        evaluated,
        fit_results,
        destination,
        image_format=image_format,
        dpi=dpi,
        include_panel_regions=False,
    )

def encode_evaluated_figure_with_panel_regions(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    dpi: float,
) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
    """Encode a product PNG and the final Agg panel rectangles from one draw."""

    output = BytesIO()
    regions = _save_evaluated_figure(
        document,
        evaluated,
        fit_results,
        output,
        image_format="png",
        dpi=dpi,
        include_panel_regions=True,
    )
    return output.getvalue(), regions

def encode_evaluated_panel_with_regions(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    size: str,
    width: int,
    height: int,
    dpi: float,
    display_state: object,
    title: str,
    value_label: str,
) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
    """Encode one live faceted panel at its exact fixed Qt geometry."""

    width = positive_integer(width, "width")
    height = positive_integer(height, "height")
    output = BytesIO()
    regions = _save_evaluated_figure(
        document,
        evaluated,
        fit_results,
        output,
        image_format="png",
        dpi=dpi,
        include_panel_regions=True,
        panel_geometry=(str(size), width, height),
        panel_display_state=display_state,
        panel_title=str(title),
        panel_value_label=str(value_label),
    )
    return output.getvalue(), regions

def _save_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    destination,
    *,
    image_format: str,
    dpi: float,
    include_panel_regions: bool,
    panel_geometry: tuple[str, int, int] | None = None,
    panel_display_state: object | None = None,
    panel_title: str = "",
    panel_value_label: str = "Signal",
) -> tuple[FigurePanelRegion, ...]:
    dpi = _render_dpi(dpi)
    regions: tuple[FigurePanelRegion, ...] = ()
    with render_style_context():
        figure = None
        try:
            figure = _render_evaluated_figure(
                document,
                evaluated,
                fit_results,
                dpi=dpi,
                panel_geometry=panel_geometry,
                panel_display_state=panel_display_state,
                panel_title=panel_title,
                panel_value_label=panel_value_label,
            )
            if panel_geometry is None:
                figure.savefig(destination, format=image_format, dpi=dpi)
            else:
                from PIL import Image

                _size, width, height = panel_geometry
                raster = raster_from_agg(
                    figure,
                    physical_size=(width, height),
                )
                Image.frombytes(
                    "RGBA",
                    (raster.width, raster.height),
                    raster.pixels,
                ).save(destination, format=image_format.upper())
            if include_panel_regions:
                regions = _figure_panel_regions(figure, evaluated, fit_results)
        finally:
            if figure is not None:
                release_agg_figure(figure)
            figure = None
            # The caller's last strong local must be gone before collecting the
            # remaining Matplotlib artist-parent cycles.
            gc.collect()
    return regions

def _render_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    dpi: float,
    panel_geometry: tuple[str, int, int] | None = None,
    panel_display_state: object | None = None,
    panel_title: str = "",
    panel_value_label: str = "Signal",
):
    """Render immutable DTOs without pyplot or shared Figures."""

    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _require_evaluated_identity(document, evaluated)
    panels = _panels(evaluated)
    shared_histogram_projection = None
    shared_histogram_analysis_counts = None
    shared_histogram_analysis_edges = None
    shared_histogram_counts = None
    shared_histogram_edges = None
    shared_histogram_x_limits = None
    shared_histogram_count_limits = None
    histogram_display = None
    faceted_histogram_display = None
    if (
        len({layer.layer_id for layer, _cell, _series in panels}) == 1
        and all(
            isinstance(series.data, EvaluatedHistogram)
            for _layer, _cell, series_group in panels
            for series in series_group
        )
    ):
        if panel_geometry is not None:
            if isinstance(
                panel_display_state,
                FacetedHistogramDisplayState,
            ):
                faceted_histogram_display = panel_display_state
                histogram_display = panel_display_state.display
            elif (
                isinstance(panel_display_state, HistogramDisplayState)
                and not panel_display_state.thresholds
            ):
                faceted_histogram_display = FacetedHistogramDisplayState(
                    panel_display_state
                )
                histogram_display = panel_display_state
            else:
                raise TypeError(
                    "live histogram grid requires per-cell histogram display state"
                )
        else:
            histogram_display = HistogramDisplayState()
        histogram_samples = tuple(
            series.data.samples
            for _layer, _cell, series_group in panels
            for series in series_group
        )
        if panel_geometry is None:
            shared_histogram_projection = HistogramBinProjection(
                histogram_samples,
                bins=histogram_display.bin_count,
            )
            shared_histogram_counts = shared_histogram_projection.bin_counts
            shared_histogram_edges = shared_histogram_projection.bin_edges
            shared_histogram_analysis_counts = shared_histogram_counts
            shared_histogram_analysis_edges = shared_histogram_edges
        else:
            low, high = _grid_histogram_value_range(panels)
            shared_histogram_projection = _WindowedHistogramProjection(
                histogram_samples,
                histogram_display.bin_count,
                visible_range=(low, high),
            )
            shared_histogram_edges = (
                shared_histogram_projection.visible_bin_edges
            )
            shared_histogram_counts = (
                shared_histogram_projection.visible_bin_counts
            )
            # A Grid's bounded fit is a DISPLAY-ONLY overlay of these exact
            # visible bars.  Fit the bins it annotates; a separate full-sample
            # projection belongs to authoritative analysis, not this thumbnail.
            shared_histogram_analysis_counts = shared_histogram_counts
            shared_histogram_analysis_edges = shared_histogram_edges
        shared_histogram_x_limits = (
            histogram_display.x_view
            or histogram_home_x_limits(shared_histogram_edges)
        )
        peak_count = max(
            (
                int(np.max(counts, initial=0))
                for counts in shared_histogram_counts
            ),
            default=0,
        )
        if panel_geometry is None:
            shared_histogram_count_limits = histogram_count_limits(
                histogram_display,
                peak_count,
            )
        elif histogram_display.relim_mode is RelimMode.FIXED:
            assert histogram_display.fixed_count_limits is not None
            shared_histogram_count_limits = (
                histogram_display.fixed_count_limits
            )
        elif histogram_display.count_scale is HistogramCountScale.LOG:
            shared_histogram_count_limits = (
                0.5,
                max(1.08 * peak_count, 1.0),
            )
        else:
            shared_histogram_count_limits = (
                0.0,
                max(1.08 * peak_count, 1.0),
            )
    live_image_display = None
    live_image_color_limits = None
    live_image_aspect = None
    if panel_geometry is not None and all(
        isinstance(series.data, EvaluatedImage)
        for _layer, _cell, series_group in panels
        for series in series_group
    ):
        if not isinstance(panel_display_state, ImageDisplayState):
            raise TypeError("live image grid requires ImageDisplayState")
        live_image_display = panel_display_state
        images = tuple(
            series.data
            for _layer, _cell, series_group in panels
            for series in series_group
        )
        data_range = evaluated_image_data_range(images)
        if data_range is None:
            live_image_color_limits = (
                live_image_display.fixed_color_limits
                if live_image_display.relim_mode is RelimMode.FIXED
                else (0.0, 1.0)
            )
        else:
            live_image_color_limits = deadband_display_range(
                live_image_display.relim_mode,
                None,
                data_range[0],
                data_range[1],
                fixed_range=live_image_display.fixed_color_limits,
                force=True,
            )
        first_image = images[0]
        live_image_aspect = (
            len(first_image.x_axis.indices)
            / max(len(first_image.y_axis.indices), 1)
        )
    shared_curve_limits = _shared_curve_limits(
        panels,
        fit_results,
        live_grid=panel_geometry is not None,
    )
    radial_color_limits = _radial_image_color_limits_by_layer(
        panels,
        fit_results,
    )
    if panel_geometry is None:
        columns = min(3, max(1, len(panels)))
        rows = math.ceil(len(panels) / columns)
        figure = Figure(
            figsize=(5.0 * columns, 4.0 * rows),
            dpi=dpi,
            constrained_layout=True,
        )
        live_font_scale = None
    else:
        size, width, height = panel_geometry
        figure = Figure(
            # Keep Main's authored design inches.  ``width``/``height`` are the
            # Qt owner's requested physical raster and are deliberately only
            # the target contract: deriving figsize from their rounded values
            # changes subpixel transforms (480/210 is not 686/300).
            figsize=panel_figure_size_inches(size),
            dpi=dpi,
            constrained_layout=False,
        )
        live_font_scale = 1.0
    axes = None
    try:
        FigureCanvasAgg(figure)
        if panel_geometry is None:
            axes = figure.subplots(rows, columns, squeeze=False).reshape(-1)
        else:
            axes, rows, columns, live_font_scale = _live_grid_axes(
                figure,
                size=size,
                cell_count=len(panels),
                cell_aspect=live_image_aspect,
            )

        histogram_series_offset = 0
        live_xlabel = None
        live_ylabel = None
        for panel_index, (
            target,
            (layer, cell, series_group),
        ) in enumerate(zip(axes, panels)):
            fit_result = fit_results.get(layer.layer_id)
            kind = series_group[0].data
            if isinstance(kind, EvaluatedCurve):
                _curve(target, layer, cell, series_group, fit_result)
                if panel_geometry is not None:
                    target.set_ylabel(panel_value_label)
                if shared_curve_limits is not None:
                    shared_x_limits, shared_y_limits = shared_curve_limits
                    target.set_xlim(*shared_x_limits)
                    target.set_ylim(*shared_y_limits)
            elif isinstance(kind, EvaluatedImage):
                if panel_geometry is None:
                    _image(
                        target,
                        figure,
                        layer,
                        cell,
                        series_group[0],
                        fit_result,
                        radial_color_limits=radial_color_limits.get(
                            layer.layer_id
                        ),
                    )
                else:
                    assert live_image_display is not None
                    assert live_image_color_limits is not None
                    viewport = image_viewport_for_display_state(
                        live_image_display,
                        image_viewport_for_evaluated_image(kind),
                    )
                    center = radius = diagnostic = None
                    if fit_result is not None:
                        if fit_result.spec.model_id != "radial_gaussian_center":
                            raise ValueError(
                                "live compact image grid currently accepts "
                                "only radial image fit overlays"
                            )
                        fit_index = _batch_storage_index(
                            fit_result,
                            layer,
                            cell,
                            series_group[0],
                        )
                        if fit_index is None:
                            diagnostic = "NOT_PRESENT"
                        elif (
                            fit_result.statuses[fit_index]
                            is FitBatchStatus.CONVERGED
                        ):
                            center, radius = radial_gaussian_fit_geometry(
                                fit_result,
                                fit_index,
                            )
                        else:
                            diagnostic = fit_result.statuses[fit_index].value
                            if fit_result.errors[fit_index]:
                                diagnostic = (
                                    f"{diagnostic}: "
                                    f"{fit_result.errors[fit_index]}"
                                )
                    _draw_projected_image(
                        target,
                        figure,
                        kind,
                        colormap=live_image_display.colormap.value,
                        color_limits=live_image_color_limits,
                        visible_bounds=viewport.visible_bounds,
                        regular_axis_contract=True,
                        center=center,
                        radius=radius,
                        diagnostic=diagnostic,
                        show_colorbar=False,
                    )
            elif isinstance(kind, EvaluatedHistogram):
                if fit_result is not None:
                    raise ValueError("fit overlays require a curve or image view")
                if shared_histogram_counts is None:
                    _histogram(target, series_group)
                else:
                    next_offset = histogram_series_offset + len(series_group)
                    _draw_histogram_projection(
                        target,
                        series_group,
                        shared_histogram_counts[
                            histogram_series_offset:next_offset
                        ],
                        shared_histogram_edges,
                        fill_alpha=(
                            1.0
                            if panel_geometry is not None
                            else HIST_FILL_ALPHA
                        ),
                    )
                    histogram_series_offset = next_offset
                    assert shared_histogram_x_limits is not None
                    assert shared_histogram_count_limits is not None
                    assert histogram_display is not None
                    target.set_yscale(histogram_display.count_scale.value)
                    target.set_xlim(*shared_histogram_x_limits)
                    target.set_ylim(*shared_histogram_count_limits)
                    if panel_geometry is not None:
                        assert faceted_histogram_display is not None
                        cell_selection = _panel_focus_selection(
                            layer,
                            cell,
                            series_group,
                        )
                        if cell_selection is None:
                            raise RuntimeError(
                                "live histogram grid cell has no logical selection"
                            )
                        cell_histogram_display = (
                            faceted_histogram_display.display_for(
                                cell_selection
                            )
                        )
                        assert shared_histogram_analysis_counts is not None
                        assert shared_histogram_analysis_edges is not None
                        _update_histogram_presentation(
                            target,
                            cell_histogram_display,
                            shared_histogram_counts[
                                histogram_series_offset
                                - len(series_group):histogram_series_offset
                            ],
                            shared_histogram_edges,
                            analysis_counts_group=(
                                shared_histogram_analysis_counts[
                                    histogram_series_offset
                                    - len(series_group):histogram_series_offset
                                ]
                            ),
                            analysis_edges=shared_histogram_analysis_edges,
                            show_stats=False,
                            infer_fit_threshold=False,
                            threshold_linewidth=1.4,
                        )
                    target.set_xlabel(panel_value_label)
                    target.set_ylabel("Shots")
            elif isinstance(kind, EvaluatedMeter):
                if fit_result is not None:
                    raise ValueError("fit overlays require a curve or image view")
                _meter(target, series_group)
            else:  # pragma: no cover - closed EvaluatedLayerData union
                raise TypeError(f"unsupported evaluated data {type(kind).__name__}")
            if live_font_scale is None:
                target.set_title(
                    _panel_title(document, layer, cell, series_group),
                )
            else:
                apply_title(
                    target,
                    _live_grid_cell_title(cell, series_group),
                    size=tick_fontsize() * live_font_scale,
                    pad=1.5,
                )
                if live_xlabel is None and target.get_xlabel():
                    live_xlabel = target.get_xlabel()
                if live_ylabel is None and target.get_ylabel():
                    live_ylabel = target.get_ylabel()
                target.set_xlabel("")
                target.set_ylabel("")
                target.tick_params(
                    axis="both",
                    labelsize=tick_fontsize() * live_font_scale,
                    length=2,
                )
                from matplotlib.ticker import (
                    FixedLocator,
                    LogLocator,
                    MaxNLocator,
                    NullLocator,
                )

                if panel_index % columns:
                    target.set_yticks([])
                    if target.get_yscale() != "linear":
                        target.yaxis.set_minor_locator(NullLocator())
                elif target.get_yscale() == "linear":
                    target.yaxis.set_major_locator(MaxNLocator(nbins=3))
                    target.tick_params(axis="y", labelleft=True)
                else:
                    target.yaxis.set_major_locator(LogLocator(numticks=3))
                    target.yaxis.set_minor_locator(NullLocator())
                    target.tick_params(axis="y", labelleft=True)
                if panel_index + columns < len(panels):
                    target.set_xticks([])
                else:
                    low, high = target.get_xlim()
                    interior = tuple(
                        value
                        for value in MaxNLocator(nbins=4).tick_values(low, high)
                        if low < value < high
                    )
                    picked = (
                        (
                            min(
                                interior,
                                key=lambda value: abs(
                                    value - (low + high) / 2.0
                                ),
                            ),
                        )
                        if interior
                        else ()
                    )
                    target.xaxis.set_major_locator(FixedLocator(picked))
                    target.tick_params(axis="x", labelbottom=True)
        for unused in axes[len(panels):]:
            unused.set_visible(False)
        if live_font_scale is not None:
            apply_title(figure, panel_title)
            if live_xlabel:
                figure.text(
                    0.5,
                    0.012,
                    live_xlabel,
                    ha="center",
                    va="bottom",
                    fontsize=axis_label_fontsize(),
                )
            if live_ylabel:
                figure.text(
                    0.008,
                    0.5,
                    live_ylabel,
                    ha="left",
                    va="center",
                    rotation="vertical",
                    fontsize=axis_label_fontsize(),
                )
        return figure
    except BaseException:
        release_agg_figure(figure)
        figure = axes = None
        gc.collect()
        raise

__all__ = [
    "render_evaluated_figure",
    "save_evaluated_figure",
    "encode_evaluated_figure_with_panel_regions",
    "encode_evaluated_panel_with_regions",
]
