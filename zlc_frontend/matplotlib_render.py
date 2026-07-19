"""Optional Matplotlib/Agg rendering for frozen :mod:`zlc_frontend` values."""

from __future__ import annotations

import gc
from io import BytesIO
import math
import threading
from dataclasses import fields, is_dataclass
from numbers import Number

import numpy as np

from zlc_data import FitBatchStatus, FitResultBatch
from zlc_storage import nonnegative_integer, positive_integer

from .figure import (
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
    fit_panel_selection as _panel_selection,
    radial_gaussian_fit_geometry,
    reduction_label as _reduction_label,
)
from .image_display import ImageDisplayState, image_viewport_for_display_state
from .curve_display import (
    CurveDisplayState,
    CurveViewportTransform,
    curve_home_x_limits,
    numeric_curve_coordinates,
)
from .histogram_display import (
    DEFAULT_HISTOGRAM_BINS,
    HistogramBinProjection,
    HistogramCountScale,
    HistogramDisplayState,
    HistogramViewportTransform,
    histogram_count_limits,
    histogram_home_x_limits,
)
from .image_raster import evaluated_image_data_range
from .data_figure import FigurePanelRegion
from .display_range import (
    RelimMode,
    deadband_display_range,
    validated_display_range,
)
from .render import (
    CurvePanelPayload,
    HistogramPanelPayload,
    PixelFormat,
    RasterBuffer,
)
from .render_style import (
    ANNOTATION_FONT_SIZE,
    CURVE_LINESTYLE,
    CURVE_MARKER,
    FIT_CONTOUR_COLOR,
    FIT_CONTOUR_LINEWIDTH,
    FIT_FAILURE_COLOR,
    FIT_LINESTYLE,
    PALETTE,
    render_style_context,
)


_RASTER_FIXED_BYTES = 8 << 20
_RASTER_BUFFER_MULTIPLIER = 8
_ARTIST_ARRAY_MULTIPLIER = 8


def _render_dpi(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Number)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("dpi must be a finite positive number")
    return float(value)


def _array_nbytes(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_array_nbytes(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, dict):
        return sum(_array_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_nbytes(item) for item in value)
    return 0


def evaluated_figure_array_nbytes(evaluated: EvaluatedFigureData) -> int:
    """Return the exact ndarray bytes retained by one frozen evaluation.

    The value is intentionally narrower than a render-peak estimate: product
    composition uses it as the data term of
    :func:`estimate_live_panel_raster_peak_nbytes`, so the same immutable
    arrays are not guessed again from rank or dtype at the Workbench boundary.
    """

    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    return _array_nbytes(evaluated)


def estimate_render_peak_nbytes(
    evaluated: EvaluatedFigureData,
    *,
    dpi: float,
) -> int:
    """Conservative Agg/PNG peak from immutable evaluated data and canvas size."""

    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    dpi = _render_dpi(dpi)
    panels = _panels(evaluated)
    columns = min(3, max(1, len(panels)))
    rows = math.ceil(len(panels) / columns)
    width = math.ceil(5.0 * columns * dpi)
    height = math.ceil(4.0 * rows * dpi)
    rgba_bytes = width * height * 4
    return int(
        _RASTER_FIXED_BYTES
        + _RASTER_BUFFER_MULTIPLIER * rgba_bytes
        + _ARTIST_ARRAY_MULTIPLIER * _array_nbytes(evaluated)
    )


def estimate_live_panel_raster_peak_nbytes(
    width: int,
    height: int,
    *,
    evaluated_data_upper_bound_bytes: int = 0,
    histogram_bins: int | None = None,
    histogram_series_count: int = 1,
    extra_retained_fronts: int = 0,
    extra_retained_evaluated_data_bytes: int = 0,
) -> int:
    """Static preflight bound for one coalesced single-panel Agg front.

    The caller supplies a schema-derived upper bound for evaluator-owned data;
    A live renderer retains the previous artist arrays while Matplotlib copies
    the next evaluated revision into those artists.  The bound also covers the
    persistent Agg canvas plus queued/visible immutable raster fronts.  A live
    histogram additionally admits its bounded counts, edges, and filled-step
    vertices; ``None`` means that the panel is a curve or meter.  Pointer-hold
    retention is explicit: every extra immutable RGBA front, bin payload, and
    older exact evaluated payload is added directly rather than hidden in a
    multiplier.
    """

    width = positive_integer(width, "width")
    height = positive_integer(height, "height")
    data_bytes = nonnegative_integer(
        evaluated_data_upper_bound_bytes,
        "evaluated_data_upper_bound_bytes",
    )
    extra_fronts = nonnegative_integer(
        extra_retained_fronts,
        "extra_retained_fronts",
    )
    extra_data_bytes = nonnegative_integer(
        extra_retained_evaluated_data_bytes,
        "extra_retained_evaluated_data_bytes",
    )
    histogram_geometry_bytes = 0
    histogram_payload_bytes = 0
    if histogram_bins is not None:
        bins = positive_integer(histogram_bins, "histogram_bins")
        series_count = positive_integer(
            histogram_series_count,
            "histogram_series_count",
        )
        counts_bytes = series_count * bins * np.dtype("<i8").itemsize
        edges_bytes = (bins + 1) * np.dtype(np.float64).itemsize
        vertices_bytes = (
            series_count
            * 2
            * (2 * bins + 2)
            * np.dtype(np.float64).itemsize
        )
        histogram_geometry_bytes = (
            counts_bytes + edges_bytes + vertices_bytes
        )
        histogram_payload_bytes = counts_bytes + edges_bytes
    return (
        _RASTER_FIXED_BYTES
        + _RASTER_BUFFER_MULTIPLIER * width * height * 4
        + _ARTIST_ARRAY_MULTIPLIER
        * (data_bytes + histogram_geometry_bytes)
        + extra_fronts * width * height * 4
        + extra_fronts * histogram_payload_bytes
        + extra_data_bytes
    )


def _series_label(series, *, include_reductions: bool) -> str | None:
    parts = [_address_label(series.batch_address)]
    if include_reductions:
        parts.append(_reduction_label(series.reductions))
    label = " | ".join(part for part in parts if part)
    return label or None


def _display_series_label(
    layer_id: str,
    series,
    index: int,
    *,
    multiple_series: bool,
) -> str:
    """Return the nonempty label used by both artists and exact payloads."""

    label = _series_label(series, include_reductions=multiple_series)
    if label is not None:
        return label
    return layer_id if not multiple_series else f"{layer_id} {index + 1}"


def _axis_label(axis) -> str:
    return axis.name if axis.unit is None else f"{axis.name} [{axis.unit}]"


def _numeric_centers(axis):
    values = tuple(axis.coordinates)
    if not values or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Number)
        for value in values
    ):
        return None
    centers = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(centers)):
        return None
    if len(centers) > 1:
        delta = np.diff(centers)
        if not (np.all(delta > 0) or np.all(delta < 0)):
            return None
    return centers


def _image_axis(axis):
    centers = _numeric_centers(axis)
    if centers is None:
        positions = np.arange(len(axis.coordinates), dtype=np.float64)
        return (
            np.arange(len(axis.coordinates) + 1, dtype=np.float64) - 0.5,
            positions,
            tuple(str(value) for value in axis.coordinates),
        )
    if len(centers) == 1:
        edges = np.asarray((centers[0] - 0.5, centers[0] + 0.5))
    else:
        middle = (centers[:-1] + centers[1:]) / 2.0
        edges = np.concatenate(
            ((centers[0] - (middle[0] - centers[0]),), middle,
             (centers[-1] + (centers[-1] - middle[-1]),))
        )
    return edges, centers, None


def _fit_status(axis, result: FitResultBatch, index: int | None) -> bool:
    if index is None:
        message = "NOT_PRESENT"
    else:
        status = result.statuses[index]
        if status is FitBatchStatus.CONVERGED:
            return True
        message = status.value
        if result.errors[index]:
            message = f"{message}: {result.errors[index]}"
    axis.text(
        0.02,
        0.98,
        f"fit {message}",
        transform=axis.transAxes,
        va="top",
        color=FIT_FAILURE_COLOR,
        fontsize=ANNOTATION_FONT_SIZE,
    )
    return False


def _curve_values(series):
    data = series.data
    assert isinstance(data, EvaluatedCurve)
    if np.iscomplexobj(data.values):
        raise ValueError(
            "complex curves require an explicit real-valued display transform"
        )
    valid_values = np.asarray(data.values)[np.asarray(data.validity, dtype=bool)]
    if valid_values.size and not bool(np.all(np.isfinite(valid_values))):
        raise ValueError("valid curve values must all be finite")
    return np.ma.array(data.values, mask=~data.validity)


def _curve(axis, layer, cell, series_group, fit_result):
    multiple_series = len(series_group) > 1
    for index, series in enumerate(series_group):
        data = series.data
        assert isinstance(data, EvaluatedCurve)
        x = np.asarray(data.x_axis.coordinates)
        values = _curve_values(series)
        label = _display_series_label(
            layer.layer_id,
            series,
            index,
            multiple_series=multiple_series,
        )
        style = {}
        if not multiple_series:
            style["color"] = PALETTE["line_single"]
        axis.plot(
            x,
            values,
            marker=CURVE_MARKER,
            linestyle=CURVE_LINESTYLE,
            label=label,
            **style,
        )
        if fit_result is not None:
            index = _batch_storage_index(fit_result, layer, cell, series)
            if _fit_status(axis, fit_result, index):
                coordinates = np.asarray(data.x_axis.coordinates, dtype=np.float64)
                predicted = fit_result.evaluate_batch(index, (coordinates,))
                axis.plot(
                    coordinates,
                    predicted,
                    linestyle=FIT_LINESTYLE,
                    label=("fit" if label is None else f"fit {label}"),
                )
    data = series_group[0].data
    axis.set_xlabel(_axis_label(data.x_axis))
    axis.set_ylabel(
        "Value" if data.value_unit is None else f"Value [{data.value_unit}]"
    )
    if len(series_group) > 1 or any(series.batch_address for series in series_group):
        axis.legend(fontsize=ANNOTATION_FONT_SIZE)


def _draw_radial_image(
    axis,
    figure,
    data: EvaluatedImage,
    *,
    colormap: str,
    color_limits: tuple[float, float],
    visible_bounds=(0.0, 0.0, 1.0, 1.0),
    center: tuple[float, float] | None,
    radius: float | None,
    diagnostic: str | None,
):
    """Draw the one radial IMAGE presentation used by canonical and typed export."""

    x_edges, x_centers, x_labels = _image_axis(data.x_axis)
    y_edges, y_centers, y_labels = _image_axis(data.y_axis)
    invalid = np.logical_not(data.validity)
    if data.values.dtype.kind == "f":
        invalid = np.logical_or(invalid, ~np.isfinite(data.values))
    values = np.ma.array(data.values, mask=invalid)
    mesh = axis.pcolormesh(
        x_edges,
        y_edges,
        values,
        shading="flat",
        cmap=colormap,
        vmin=color_limits[0],
        vmax=color_limits[1],
        rasterized=True,
    )
    figure.colorbar(mesh, ax=axis)
    if x_labels is not None:
        axis.set_xticks(x_centers, x_labels)
    if y_labels is not None:
        axis.set_yticks(y_centers, y_labels)
    axis.set_xlabel(_axis_label(data.x_axis))
    axis.set_ylabel(_axis_label(data.y_axis))
    left, top, right, bottom = visible_bounds
    x_start, x_stop = float(x_edges[0]), float(x_edges[-1])
    y_start, y_stop = float(y_edges[0]), float(y_edges[-1])
    x_limits = (
        x_start + left * (x_stop - x_start),
        x_start + right * (x_stop - x_start),
    )
    # The first declared Y coordinate/raster row is always at the top.
    y_limits = (
        y_start + bottom * (y_stop - y_start),
        y_start + top * (y_stop - y_start),
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    if center is not None:
        from matplotlib.patches import Circle

        assert radius is not None and diagnostic is None
        axis.scatter(*center, color=PALETTE["fit_right"], s=8, clip_on=True)
        axis.add_patch(
            Circle(
                center,
                radius=radius,
                edgecolor=PALETTE["fit_right"],
                facecolor="none",
                linewidth=1.8,
                alpha=0.9,
                clip_on=True,
            )
        )
        # Off-screen saved geometry is annotation, never an autoscale input.
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
    else:
        assert radius is None and diagnostic is not None
        axis.text(
            0.02,
            0.98,
            f"fit {diagnostic}",
            transform=axis.transAxes,
            va="top",
            color=FIT_FAILURE_COLOR,
            fontsize=ANNOTATION_FONT_SIZE,
        )


def _image(
    axis,
    figure,
    layer,
    cell,
    series,
    fit_result,
    *,
    radial_color_limits: tuple[float, float] | None = None,
):
    data = series.data
    assert isinstance(data, EvaluatedImage)
    if np.iscomplexobj(data.values):
        raise ValueError("complex images require an explicit real-valued display transform")
    radial_fit = (
        fit_result is not None
        and fit_result.spec.model_id == "radial_gaussian_center"
    )
    if radial_fit:
        if radial_color_limits is None:
            raise RuntimeError("radial image grid omitted its shared color limits")
        index = _batch_storage_index(fit_result, layer, cell, series)
        center = None
        radius = None
        if index is None:
            diagnostic = "NOT_PRESENT"
        else:
            status = fit_result.statuses[index]
            if status is FitBatchStatus.CONVERGED:
                center, radius = radial_gaussian_fit_geometry(fit_result, index)
                diagnostic = None
            else:
                diagnostic = status.value
                if fit_result.errors[index]:
                    diagnostic = f"{diagnostic}: {fit_result.errors[index]}"
        _draw_radial_image(
            axis,
            figure,
            data,
            colormap="gray",
            color_limits=radial_color_limits,
            center=center,
            radius=radius,
            diagnostic=diagnostic,
        )
        return

    x_edges, x_centers, x_labels = _image_axis(data.x_axis)
    y_edges, y_centers, y_labels = _image_axis(data.y_axis)
    values = np.ma.array(data.values, mask=~data.validity)
    mesh = axis.pcolormesh(x_edges, y_edges, values, shading="flat")
    figure.colorbar(mesh, ax=axis)
    if x_labels is not None:
        axis.set_xticks(x_centers, x_labels)
    if y_labels is not None:
        axis.set_yticks(y_centers, y_labels)
    axis.set_xlabel(_axis_label(data.x_axis))
    axis.set_ylabel(_axis_label(data.y_axis))
    if fit_result is not None:
        index = _batch_storage_index(fit_result, layer, cell, series)
        if _fit_status(axis, fit_result, index):
            x_grid, y_grid = np.meshgrid(
                np.asarray(data.x_axis.coordinates, dtype=np.float64),
                np.asarray(data.y_axis.coordinates, dtype=np.float64),
            )
            grids = {
                data.x_axis.axis_id: x_grid,
                data.y_axis.axis_id: y_grid,
            }
            coordinates = tuple(
                grids[item.axis_id] for item in fit_result.fit_axis_specs
            )
            predicted = fit_result.evaluate_batch(index, coordinates)
            if np.ptp(predicted) > 0:
                axis.contour(
                    x_grid,
                    y_grid,
                    predicted,
                    colors=FIT_CONTOUR_COLOR,
                    linewidths=FIT_CONTOUR_LINEWIDTH,
                )


def _radial_image_color_limits_by_layer(
    panels,
    fit_results: dict[str, FitResultBatch],
) -> dict[str, tuple[float, float]]:
    """Resolve one default TIGHT range for every radial IMAGE grid layer."""

    images_by_layer: dict[str, list[EvaluatedImage]] = {}
    for layer, _cell, series_group in panels:
        fit_result = fit_results.get(layer.layer_id)
        if (
            fit_result is None
            or fit_result.spec.model_id != "radial_gaussian_center"
        ):
            continue
        for series in series_group:
            if not isinstance(series.data, EvaluatedImage):
                raise ValueError("radial fit overlays require an IMAGE view")
            images_by_layer.setdefault(layer.layer_id, []).append(series.data)

    limits = {}
    for layer_id, images in images_by_layer.items():
        data_range = evaluated_image_data_range(images)
        limits[layer_id] = (
            (0.0, 1.0)
            if data_range is None
            else deadband_display_range(
                RelimMode.TIGHT,
                None,
                data_range[0],
                data_range[1],
                force=True,
            )
        )
    return limits


def _radial_projected_image(
    axis,
    figure,
    panel: RadialGaussianImageFitPanel,
    display: ImageDisplayState,
    color_limits: tuple[float, float],
):
    """Draw one already-projected saved-fit cell without fit authority."""

    viewport = image_viewport_for_display_state(display, panel.home_viewport)
    overlay = panel.fit_overlay
    _draw_radial_image(
        axis,
        figure,
        panel.image,
        colormap=display.colormap.value,
        color_limits=color_limits,
        visible_bounds=viewport.visible_bounds,
        center=overlay.center_xy,
        radius=overlay.one_over_e_radius,
        diagnostic=(
            None
            if overlay.status is FitBatchStatus.CONVERGED
            else overlay.diagnostic
        ),
    )
    status = "NOT_PRESENT" if overlay.status is None else overlay.status.value
    # The canonical Helvetica face lacks U+00B7; mathtext preserves the same
    # visual separator without dropping a glyph in PDF/SVG/JPEG export.
    axis.set_title(f"{overlay.caption} $\\cdot$ {status}")


def _validated_radial_panels(
    panels: tuple[RadialGaussianImageFitPanel, ...],
) -> tuple[RadialGaussianImageFitPanel, ...]:
    if not isinstance(panels, tuple) or not panels or any(
        not isinstance(panel, RadialGaussianImageFitPanel) for panel in panels
    ):
        raise TypeError("panels must be a non-empty radial panel tuple")
    if len(panels) > 36:
        raise ValueError("radial saved-fit export exceeded 36 panels")
    first = panels[0].fit_overlay
    if any(
        panel.fit_overlay.artifact_identity != first.artifact_identity
        or panel.fit_overlay.source_ref != first.source_ref
        for panel in panels[1:]
    ):
        raise ValueError("radial saved-fit export cannot mix artifact revisions")
    return panels


def _validated_radial_grid_columns(columns: int, panel_count: int) -> int:
    columns = positive_integer(columns, "columns")
    if columns > panel_count:
        raise ValueError("columns cannot exceed the radial panel count")
    return columns


def estimate_projected_radial_fit_render_peak_nbytes(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    *,
    dpi: float,
    columns: int,
) -> int:
    """Bound incremental Agg export memory after source panels are retained.

    The caller already owns the immutable image samples.  This estimate charges
    the Agg canvas/encoder fronts and conservative artist/mask/coordinate copies
    created specifically by the export.
    """

    prepared = _validated_radial_panels(panels)
    dpi = _render_dpi(dpi)
    columns = _validated_radial_grid_columns(columns, len(prepared))
    rows = math.ceil(len(prepared) / columns)
    width = math.ceil(5.0 * columns * dpi)
    height = math.ceil(4.0 * rows * dpi)
    rgba_bytes = width * height * 4
    artist_input_bytes = sum(
        panel.image.values.nbytes
        + panel.image.validity.nbytes
        + 8
        * (
            len(panel.image.x_axis.coordinates)
            + len(panel.image.y_axis.coordinates)
        )
        for panel in prepared
    )
    return int(
        _RASTER_FIXED_BYTES
        + _RASTER_BUFFER_MULTIPLIER * rgba_bytes
        + _ARTIST_ARRAY_MULTIPLIER * artist_input_bytes
    )


def render_radial_gaussian_image_fit_panels(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float],
    *,
    columns: int,
    dpi: float = 100.0,
    memory_limit_bytes: int | None = None,
):
    """Render the current typed saved-fit IMAGE view from immutable projections.

    No dataset lookup, view evaluation, predicted image, or solver is reachable
    from this path.  The caller owns the returned Figure and must release it
    with :func:`release_agg_figure`.
    """

    prepared = _validated_radial_panels(panels)
    if not isinstance(display, ImageDisplayState):
        raise TypeError("display must be ImageDisplayState")
    limits = validated_display_range(
        current_color_limits,
        "current_color_limits",
    )
    dpi = _render_dpi(dpi)
    columns = _validated_radial_grid_columns(columns, len(prepared))
    if memory_limit_bytes is not None:
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        required = estimate_projected_radial_fit_render_peak_nbytes(
            prepared,
            dpi=dpi,
            columns=columns,
        )
        if required > limit:
            raise MemoryError(
                f"projected radial render peak {required} exceeds limit {limit}"
            )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = math.ceil(len(prepared) / columns)
    with render_style_context():
        figure = Figure(
            figsize=(5.0 * columns, 4.0 * rows),
            dpi=dpi,
            constrained_layout=True,
        )
        axes = None
        try:
            FigureCanvasAgg(figure)
            axes = figure.subplots(rows, columns, squeeze=False).reshape(-1)
            for axis, panel in zip(axes, prepared, strict=False):
                _radial_projected_image(axis, figure, panel, display, limits)
            for unused in axes[len(prepared):]:
                unused.set_visible(False)
            return figure
        except BaseException:
            release_agg_figure(figure)
            figure = axes = None
            gc.collect()
            raise


def save_radial_gaussian_image_fit_panels(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float],
    destination,
    *,
    image_format: str,
    columns: int,
    dpi: float = 100.0,
    memory_limit_bytes: int | None = None,
) -> None:
    """Save the exact committed typed page/focus display.

    Viewport, colormap, shared limits, saved-fit overlay, status, and frozen
    board columns are preserved for PNG/PDF/SVG/JPEG.  A transient rectangle
    selection candidate is intentionally absent: it is an uncommitted pointer
    draft, not part of :class:`ImageDisplayState`.
    """

    if not isinstance(image_format, str):
        raise TypeError("image_format must be str")
    if image_format not in {"png", "pdf", "svg", "jpg", "jpeg"}:
        raise ValueError("radial image export format must be png, pdf, svg, jpg, or jpeg")
    figure = None
    try:
        figure = render_radial_gaussian_image_fit_panels(
            panels,
            display,
            current_color_limits,
            columns=columns,
            dpi=dpi,
            memory_limit_bytes=memory_limit_bytes,
        )
        with render_style_context():
            figure.savefig(destination, format=image_format, dpi=dpi)
    finally:
        if figure is not None:
            release_agg_figure(figure)
        figure = None
        gc.collect()


def _histogram_projection(series_group, bins: int):
    return HistogramBinProjection(
        tuple(series.data.samples for series in series_group), bins=bins
    )


def _histogram(
    axis,
    series_group,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    projection=None,
):
    """Draw filled, shared-edge distributions and return their exact bins."""

    multiple_series = len(series_group) > 1
    all_boolean = all(
        isinstance(series.data, EvaluatedHistogram)
        and np.issubdtype(series.data.samples.dtype, np.bool_)
        for series in series_group
    )
    bin_projection = (
        _histogram_projection(series_group, bins)
        if projection is None
        else projection
    )
    counts_group = bin_projection.bin_counts
    edges = bin_projection.bin_edges
    for series, counts in zip(series_group, counts_group, strict=True):
        data = series.data
        assert isinstance(data, EvaluatedHistogram)
        label = _series_label(series, include_reductions=multiple_series)
        axis.stairs(
            counts,
            edges,
            fill=True,
            alpha=0.4,
            label=label,
        )
    if all_boolean:
        axis.set_xticks((0, 1), ("false", "true"))
    if len(series_group) > 1 or any(series.batch_address for series in series_group):
        axis.legend(fontsize=ANNOTATION_FONT_SIZE)
    return counts_group, edges


def _meter_text(series_group) -> str:
    lines = []
    multiple_series = len(series_group) > 1
    for series in series_group:
        data = series.data
        assert isinstance(data, EvaluatedMeter)
        label = _series_label(series, include_reductions=multiple_series) or ""
        if data.valid and not _finite_numeric_scalar(data.value):
            raise ValueError("valid meter values must be finite")
        value = str(data.value) if data.valid else "invalid"
        lines.append(f"{label}: {value}" if label else value)
    return "\n".join(lines)


def _finite_numeric_scalar(value: Number) -> bool:
    try:
        return bool(np.isfinite(value))
    except TypeError:
        predicate = getattr(value, "is_finite", None)
        if callable(predicate):
            return bool(predicate())
        return bool(math.isfinite(value))


def _meter(axis, series_group):
    axis.set_axis_off()
    axis.text(0.5, 0.5, _meter_text(series_group), ha="center", va="center")


def _require_evaluated_identity(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
) -> None:
    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")


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


def release_agg_figure(figure) -> None:
    """Clear artists and sever the Figure/Canvas ownership edge."""
    canvas = getattr(figure, "canvas", None)
    try:
        figure.clear()
    finally:
        try:
            figure.set_canvas(None)
            if getattr(canvas, "figure", None) is figure:
                canvas.figure = None
        finally:
            figure = canvas = None


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
        selection = _panel_selection(layer, cell, series_group, fit_result)
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
                selection,
                fit_storage_index,
                float(bounds.x0),
                float(1.0 - bounds.y1),
                float(bounds.x1),
                float(1.0 - bounds.y0),
            )
        )
    return tuple(regions)


def _save_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    destination,
    *,
    image_format: str,
    dpi: float,
    include_panel_regions: bool,
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
            )
            figure.savefig(destination, format=image_format, dpi=dpi)
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
):
    """Render immutable DTOs without pyplot or shared Figures."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _require_evaluated_identity(document, evaluated)
    panels = _panels(evaluated)
    radial_color_limits = _radial_image_color_limits_by_layer(
        panels,
        fit_results,
    )
    columns = min(3, max(1, len(panels)))
    rows = math.ceil(len(panels) / columns)
    figure = Figure(
        figsize=(5.0 * columns, 4.0 * rows),
        dpi=dpi,
        constrained_layout=True,
    )
    axes = None
    try:
        FigureCanvasAgg(figure)
        axes = figure.subplots(rows, columns, squeeze=False).reshape(-1)

        for target, (layer, cell, series_group) in zip(axes, panels):
            fit_result = fit_results.get(layer.layer_id)
            kind = series_group[0].data
            if isinstance(kind, EvaluatedCurve):
                _curve(target, layer, cell, series_group, fit_result)
            elif isinstance(kind, EvaluatedImage):
                _image(
                    target,
                    figure,
                    layer,
                    cell,
                    series_group[0],
                    fit_result,
                    radial_color_limits=radial_color_limits.get(layer.layer_id),
                )
            elif isinstance(kind, EvaluatedHistogram):
                if fit_result is not None:
                    raise ValueError("fit overlays require a curve or image view")
                _histogram(target, series_group)
            elif isinstance(kind, EvaluatedMeter):
                if fit_result is not None:
                    raise ValueError("fit overlays require a curve or image view")
                _meter(target, series_group)
            else:  # pragma: no cover - closed EvaluatedLayerData union
                raise TypeError(f"unsupported evaluated data {type(kind).__name__}")
            target.set_title(_panel_title(document, layer, cell, series_group))
        for unused in axes[len(panels):]:
            unused.set_visible(False)
        return figure
    except BaseException:
        release_agg_figure(figure)
        figure = axes = None
        gc.collect()
        raise


class SinglePanelAggRenderer:
    """Worker-affine live Agg surface for one curve/histogram/meter panel."""

    __slots__ = (
        "_axis",
        "_document",
        "_figure",
        "_artists",
        "_owner_thread",
        "_topology",
    )

    def __init__(
        self,
        document: FigureDocument,
        *,
        width: int,
        height: int,
        dpi: float = 100.0,
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        if not isinstance(document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        width = positive_integer(width, "width")
        height = positive_integer(height, "height")
        dpi = _render_dpi(dpi)
        with render_style_context():
            figure = None
            axis = None
            try:
                figure = Figure(
                    figsize=(width / dpi, height / dpi),
                    dpi=dpi,
                    constrained_layout=True,
                )
                FigureCanvasAgg(figure)
                axis = figure.subplots()
            except BaseException:
                if figure is not None:
                    release_agg_figure(figure)
                figure = axis = None
                gc.collect()
                raise
        self._owner_thread = threading.get_ident()
        self._document = document
        self._figure = figure
        self._axis = axis
        self._artists = ()
        self._topology = None

    def render(self, evaluated: EvaluatedFigureData) -> RasterBuffer:
        with render_style_context():
            return self._render(evaluated)

    def _render(self, evaluated: EvaluatedFigureData) -> RasterBuffer:
        figure, axis, _layer, _cell, series_group = self._prepare_panel(evaluated)
        first = series_group[0].data
        if isinstance(first, (EvaluatedCurve, EvaluatedHistogram)):
            axis.set_autoscalex_on(True)
            axis.set_autoscaley_on(True)
            axis.relim(visible_only=True)
            axis.autoscale_view()
        return self._draw_raster(figure)

    def render_interactive_curve(
        self,
        evaluated: EvaluatedFigureData,
        state: CurveDisplayState,
        *,
        current_y_limits: tuple[float, float] | None,
        previous_relim_mode: RelimMode | None,
    ) -> tuple[RasterBuffer, CurvePanelPayload]:
        """Render one exact interactive curve front on this worker owner.

        The caller owns the previously accepted y baseline.  This method uses
        it to resolve hysteresis but never stores or advances it internally;
        the returned viewport is the only candidate the caller may accept.
        """

        with render_style_context():
            return self._render_interactive_curve(
                evaluated,
                state,
                current_y_limits=current_y_limits,
                previous_relim_mode=previous_relim_mode,
            )

    def _render_interactive_curve(
        self,
        evaluated: EvaluatedFigureData,
        state: CurveDisplayState,
        *,
        current_y_limits: tuple[float, float] | None,
        previous_relim_mode: RelimMode | None,
    ) -> tuple[RasterBuffer, CurvePanelPayload]:
        if not isinstance(state, CurveDisplayState):
            raise TypeError("state must be CurveDisplayState")
        if previous_relim_mode is not None and not isinstance(
            previous_relim_mode,
            RelimMode,
        ):
            raise TypeError("previous_relim_mode must be RelimMode or None")
        # Validate the closed interactive contract before mutating this
        # persistent Agg surface.  A rejected categorical/non-monotonic front
        # must not leave a unit converter or partial artist topology behind.
        pre_layer, _pre_cell, pre_series_group = self._one_panel(evaluated)
        curves = tuple(series.data for series in pre_series_group)
        if any(not isinstance(curve, EvaluatedCurve) for curve in curves):
            raise ValueError("interactive render requires one CURVE panel")
        first_curve = curves[0]
        assert isinstance(first_curve, EvaluatedCurve)
        numeric_curve_coordinates(first_curve.x_axis)
        if any(curve.x_axis != first_curve.x_axis for curve in curves[1:]):
            raise ValueError("interactive curve series must share one exact x axis")
        value_unit = first_curve.value_unit
        if any(curve.value_unit != value_unit for curve in curves[1:]):
            raise ValueError("interactive curve series must share value_unit")
        for series in pre_series_group:
            _curve_values(series)

        figure, axis, layer, _cell, series_group = self._prepare_panel(evaluated)
        if layer is not pre_layer or series_group is not pre_series_group:
            raise RuntimeError("interactive panel identity changed during preparation")

        finite_groups: list[np.ndarray] = []
        for series in series_group:
            data = series.data
            assert isinstance(data, EvaluatedCurve)
            values = np.asarray(data.values)
            valid = np.asarray(data.validity, dtype=bool)
            if np.any(valid):
                finite_groups.append(np.asarray(values[valid], dtype=np.float64))

        home_x_limits = curve_home_x_limits(first_curve.x_axis)
        x_limits = state.x_view or home_x_limits
        if finite_groups:
            data_min = min(float(np.min(group)) for group in finite_groups)
            data_max = max(float(np.max(group)) for group in finite_groups)
            y_limits = deadband_display_range(
                state.relim_mode,
                current_y_limits,
                data_min,
                data_max,
                fixed_range=state.fixed_y_limits,
                force=(
                    previous_relim_mode is None
                    or previous_relim_mode is not state.relim_mode
                ),
            )
        elif state.relim_mode is RelimMode.FIXED:
            assert state.fixed_y_limits is not None
            y_limits = state.fixed_y_limits
        elif current_y_limits is not None:
            y_limits = validated_display_range(
                current_y_limits,
                "current_y_limits",
            )
        else:
            y_limits = (0.0, 1.0)

        # Authored x and resolved y are applied after every artist has received
        # the same evaluated revision.  No autoscale call follows these pins.
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        raster = self._draw_raster(figure)
        actual_x_limits = validated_display_range(
            tuple(float(value) for value in axis.get_xlim()),
            "drawn curve x limits",
        )
        actual_y_limits = validated_display_range(
            tuple(float(value) for value in axis.get_ylim()),
            "drawn curve y limits",
        )
        x0, y0, width, height = (
            float(value) for value in axis.bbox.bounds
        )
        plot_bounds = (
            x0 / raster.width,
            1.0 - (y0 + height) / raster.height,
            (x0 + width) / raster.width,
            1.0 - y0 / raster.height,
        )
        viewport = CurveViewportTransform(
            first_curve.x_axis,
            state.revision,
            plot_bounds,
            actual_x_limits,
            actual_y_limits,
            home_x_limits,
        )
        try:
            evaluated_input = next(
                item for item in evaluated.inputs if item.dataset_id == layer.dataset_id
            )
        except StopIteration as exc:
            raise ValueError(
                "interactive curve layer dataset is absent from evaluated inputs"
            ) from exc
        if sum(
            item.dataset_id == layer.dataset_id for item in evaluated.inputs
        ) != 1:
            raise ValueError(
                "interactive curve layer requires one exact evaluated input"
            )
        labels = self._curve_series_labels(layer.layer_id, series_group)
        return raster, CurvePanelPayload(
            evaluated_input,
            viewport,
            tuple(series_group),
            labels,
        )

    def render_interactive_histogram(
        self,
        evaluated: EvaluatedFigureData,
        state: HistogramDisplayState,
        *,
        current_count_limits: tuple[float, float] | None,
        previous_relim_mode: RelimMode | None,
        previous_count_scale: HistogramCountScale | None,
    ) -> tuple[RasterBuffer, HistogramPanelPayload]:
        """Render one front-bound shared-bin Histogram projection."""

        with render_style_context():
            return self._render_interactive_histogram(
                evaluated,
                state,
                current_count_limits=current_count_limits,
                previous_relim_mode=previous_relim_mode,
                previous_count_scale=previous_count_scale,
            )

    def _render_interactive_histogram(
        self,
        evaluated: EvaluatedFigureData,
        state: HistogramDisplayState,
        *,
        current_count_limits: tuple[float, float] | None,
        previous_relim_mode: RelimMode | None,
        previous_count_scale: HistogramCountScale | None,
    ) -> tuple[RasterBuffer, HistogramPanelPayload]:
        if not isinstance(state, HistogramDisplayState):
            raise TypeError("state must be HistogramDisplayState")
        pre_layer, _pre_cell, pre_series_group = self._one_panel(evaluated)
        histograms = tuple(series.data for series in pre_series_group)
        if any(not isinstance(item, EvaluatedHistogram) for item in histograms):
            raise ValueError("interactive render requires one HISTOGRAM panel")
        value_unit = histograms[0].value_unit
        if any(item.value_unit != value_unit for item in histograms[1:]):
            raise ValueError("interactive histogram series must share value_unit")
        # Validate finite samples and freeze one common edge vector before this
        # persistent Agg surface is changed.
        bin_projection = HistogramBinProjection(
            tuple(item.samples for item in histograms),
            bins=state.bin_count,
        )
        figure, axis, layer, _cell, series_group = self._prepare_panel(
            evaluated,
            histogram_bins=state.bin_count,
            histogram_projection=bin_projection,
        )
        if layer is not pre_layer or series_group is not pre_series_group:
            raise RuntimeError("interactive panel identity changed during preparation")

        home_x_limits = histogram_home_x_limits(bin_projection.bin_edges)
        x_limits = state.x_view or home_x_limits
        peak_count = max(
            (
                int(np.max(counts)) if counts.size else 0
                for counts in bin_projection.bin_counts
            ),
            default=0,
        )
        count_limits = histogram_count_limits(
            state,
            peak_count,
            current_count_limits=current_count_limits,
            previous_relim_mode=previous_relim_mode,
            previous_count_scale=previous_count_scale,
        )
        axis.set_yscale(state.count_scale.value)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*count_limits)
        raster = self._draw_raster(figure)
        actual_x_limits = validated_display_range(
            tuple(float(value) for value in axis.get_xlim()),
            "drawn histogram x limits",
        )
        actual_count_limits = validated_display_range(
            tuple(float(value) for value in axis.get_ylim()),
            "drawn histogram count limits",
        )
        x0, y0, width, height = (float(value) for value in axis.bbox.bounds)
        plot_bounds = (
            x0 / raster.width,
            1.0 - (y0 + height) / raster.height,
            (x0 + width) / raster.width,
            1.0 - y0 / raster.height,
        )
        viewport = HistogramViewportTransform(
            state.revision,
            plot_bounds,
            actual_x_limits,
            actual_count_limits,
            home_x_limits,
            state.count_scale,
            state.relim_mode,
            state.x_view is None,
            state.bin_count,
        )
        try:
            evaluated_input = next(
                item for item in evaluated.inputs if item.dataset_id == layer.dataset_id
            )
        except StopIteration as exc:
            raise ValueError(
                "interactive histogram layer dataset is absent from evaluated inputs"
            ) from exc
        if sum(item.dataset_id == layer.dataset_id for item in evaluated.inputs) != 1:
            raise ValueError(
                "interactive histogram layer requires one exact evaluated input"
            )
        labels = self._curve_series_labels(layer.layer_id, series_group)
        return raster, HistogramPanelPayload(
            evaluated_input,
            viewport,
            tuple(series_group),
            labels,
            bin_projection,
        )

    def _prepare_panel(
        self,
        evaluated: EvaluatedFigureData,
        *,
        histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
        histogram_projection=None,
    ):
        self._require_owner()
        figure = self._figure
        axis = self._axis
        if figure is None or axis is None:
            raise RuntimeError("single-panel renderer is closed")
        layer, cell, series_group = self._one_panel(evaluated)
        first = series_group[0].data
        kind = type(first)
        if isinstance(first, EvaluatedCurve):
            series_topology = tuple(
                (
                    series.batch_address,
                    series.data.x_axis.axis_id,
                    series.data.x_axis.role,
                    len(series.data.x_axis.indices),
                    series.data.value_unit,
                )
                for series in series_group
            )
        elif isinstance(first, EvaluatedHistogram):
            series_topology = tuple(
                (series.batch_address, series.data.samples.dtype.str)
                for series in series_group
            )
        else:
            assert isinstance(first, EvaluatedMeter)
            series_topology = tuple(series.batch_address for series in series_group)
        topology = (
            layer.layer_id,
            layer.dataset_id,
            layer.resolutions,
            cell.facet_address,
            kind,
            series_topology,
        )
        if self._topology is None:
            if isinstance(first, EvaluatedCurve):
                _curve(axis, layer, cell, series_group, None)
                self._artists = tuple(axis.lines)
            elif isinstance(first, EvaluatedHistogram):
                _histogram(
                    axis,
                    series_group,
                    bins=histogram_bins,
                    projection=histogram_projection,
                )
                self._artists = tuple(axis.patches)
            else:
                _meter(axis, series_group)
                self._artists = tuple(axis.texts)
            expected_artists = 1 if isinstance(first, EvaluatedMeter) else len(series_group)
            if len(self._artists) != expected_artists:
                raise RuntimeError("single-panel renderer created another artist topology")
            self._topology = topology
        else:
            expected_artists = 1 if isinstance(first, EvaluatedMeter) else len(series_group)
            if topology != self._topology or len(self._artists) != expected_artists:
                raise RuntimeError("progressive panel topology changed between revisions")
            if isinstance(first, EvaluatedCurve):
                for line, series in zip(self._artists, series_group, strict=True):
                    data = series.data
                    assert isinstance(data, EvaluatedCurve)
                    line.set_data(
                        np.asarray(data.x_axis.coordinates),
                        _curve_values(series),
                    )
            elif isinstance(first, EvaluatedHistogram):
                bin_projection = (
                    _histogram_projection(series_group, histogram_bins)
                    if histogram_projection is None
                    else histogram_projection
                )
                for artist, counts in zip(
                    self._artists,
                    bin_projection.bin_counts,
                    strict=True,
                ):
                    artist.set_data(counts, bin_projection.bin_edges)
            else:
                assert isinstance(first, EvaluatedMeter)
                self._artists[0].set_text(_meter_text(series_group))

        multiple_series = len(series_group) > 1
        if isinstance(first, (EvaluatedCurve, EvaluatedHistogram)):
            for index, (line, series) in enumerate(
                zip(self._artists, series_group, strict=True)
            ):
                line.set_label(
                    _display_series_label(
                        layer.layer_id,
                        series,
                        index,
                        multiple_series=multiple_series,
                    )
                )
        if isinstance(first, EvaluatedCurve):
            axis.set_xlabel(_axis_label(first.x_axis))
            axis.set_ylabel(
                "Value"
                if first.value_unit is None
                else f"Value [{first.value_unit}]"
            )
        elif isinstance(first, EvaluatedHistogram):
            axis.set_xlabel(
                "Value"
                if first.value_unit is None
                else f"Value [{first.value_unit}]"
            )
            axis.set_ylabel("Count")
        axis.set_title(_panel_title(self._document, layer, cell, series_group))
        if isinstance(first, (EvaluatedCurve, EvaluatedHistogram)) and (
            multiple_series or any(series.batch_address for series in series_group)
        ):
            legend = axis.get_legend()
            if legend is None:
                legend = axis.legend(fontsize=ANNOTATION_FONT_SIZE)
            else:
                texts = legend.get_texts()
                if len(texts) != len(self._artists):
                    raise RuntimeError("progressive panel legend topology changed")
                for text, line in zip(texts, self._artists, strict=True):
                    text.set_text(line.get_label())
        return figure, axis, layer, cell, series_group

    @staticmethod
    def _curve_series_labels(layer_id: str, series_group) -> tuple[str, ...]:
        multiple_series = len(series_group) > 1
        labels = []
        for index, series in enumerate(series_group):
            label = _display_series_label(
                layer_id,
                series,
                index,
                multiple_series=multiple_series,
            )
            labels.append(label)
        return tuple(labels)

    @staticmethod
    def _draw_raster(figure) -> RasterBuffer:
        figure.canvas.draw()
        actual_width, actual_height = figure.canvas.get_width_height()
        return RasterBuffer(
            actual_width,
            actual_height,
            actual_width * 4,
            PixelFormat.RGBA8888,
            bytes(figure.canvas.buffer_rgba()),
        )

    def close(self) -> None:
        with render_style_context():
            self._close()

    def _close(self) -> None:
        self._require_owner()
        figure = self._figure
        if figure is None:
            return
        self._figure = None
        self._axis = None
        self._artists = ()
        self._topology = None
        # Collect before the worker reports done so the FINAL renderer cannot
        # overlap a provisional Agg surface.
        release_agg_figure(figure)
        figure = None
        gc.collect()

    def _one_panel(self, evaluated: EvaluatedFigureData):
        _require_evaluated_identity(self._document, evaluated)
        panels = _panels(evaluated)
        if len(panels) != 1:
            raise ValueError("progressive raster requires exactly one panel")
        series_group = panels[0][2]
        if not series_group:
            raise ValueError("progressive raster panel has no series")
        kind = type(series_group[0].data)
        if kind not in (EvaluatedCurve, EvaluatedHistogram, EvaluatedMeter) or any(
            type(series.data) is not kind for series in series_group
        ):
            raise ValueError(
                "progressive raster requires one homogeneous curve, histogram, or meter panel"
            )
        return panels[0]

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("single-panel Agg renderer used from another thread")


__all__ = [
    "encode_evaluated_figure_with_panel_regions",
    "evaluated_figure_array_nbytes",
    "estimate_live_panel_raster_peak_nbytes",
    "estimate_projected_radial_fit_render_peak_nbytes",
    "estimate_render_peak_nbytes",
    "render_radial_gaussian_image_fit_panels",
    "render_evaluated_figure",
    "release_agg_figure",
    "save_radial_gaussian_image_fit_panels",
    "save_evaluated_figure",
    "SinglePanelAggRenderer",
]
