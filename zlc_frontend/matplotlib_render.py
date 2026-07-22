"""Optional Matplotlib/Agg rendering for frozen :mod:`zlc_frontend` values."""

from __future__ import annotations

import gc
from decimal import Decimal
from io import BytesIO
import math
import threading
from dataclasses import fields, is_dataclass
from numbers import Integral, Number

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
from .image_view import image_viewport_for_evaluated_image
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
    CurveFitOverlay,
    CurvePanelPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
    PixelFormat,
    RasterBuffer,
    _validated_curve_fit_overlays,
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
    indexed_colormap,
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


def _histogram_render_geometry_nbytes(
    evaluated: EvaluatedFigureData,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> int:
    """Conservatively charge bounded histogram counts/edges/step vertices."""

    total = 0
    for _layer, _cell, series_group in _panels(evaluated):
        if not isinstance(series_group[0].data, EvaluatedHistogram):
            continue
        series_count = len(series_group)
        counts_bytes = series_count * bins * np.dtype("<i8").itemsize
        edges_bytes = (bins + 1) * np.dtype(np.float64).itemsize
        vertices_bytes = (
            series_count
            * 2
            * (2 * bins + 2)
            * np.dtype(np.float64).itemsize
        )
        total += counts_bytes + edges_bytes + vertices_bytes
    return int(total)


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
        + _ARTIST_ARRAY_MULTIPLIER
        * (
            _array_nbytes(evaluated)
            + _histogram_render_geometry_nbytes(evaluated)
        )
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
    fit_overlay_retained_upper_bound_bytes: int = 0,
    fit_prediction_upper_bound_bytes: int = 0,
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
    multiplier.  A curve fit front separately charges its already-owned DTO
    retention and the prediction bytes copied through Matplotlib's artist
    workspace; callers must not hide either term in a second render budget.
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
    fit_overlay_bytes = nonnegative_integer(
        fit_overlay_retained_upper_bound_bytes,
        "fit_overlay_retained_upper_bound_bytes",
    )
    fit_prediction_bytes = nonnegative_integer(
        fit_prediction_upper_bound_bytes,
        "fit_prediction_upper_bound_bytes",
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
        + fit_overlay_bytes
        + _ARTIST_ARRAY_MULTIPLIER * fit_prediction_bytes
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


def _shared_curve_limits(
    panels,
    fit_results,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return comparison x/y ranges for a homogeneous source-only curve grid.

    The standalone focused panel deliberately does not consume this range: it
    keeps the ordinary Curve relim policy, matching main's LineCell/GridPlot
    split between comparable thumbnails and a detailed focus view.
    """

    if fit_results or len(panels) <= 1:
        return None
    if len({layer.layer_id for layer, _cell, _series in panels}) != 1:
        return None
    curves = tuple(
        series.data
        for _layer, _cell, series_group in panels
        for series in series_group
    )
    if not curves or any(not isinstance(curve, EvaluatedCurve) for curve in curves):
        return None
    first = curves[0]
    if any(
        curve.x_axis != first.x_axis or curve.value_unit != first.value_unit
        for curve in curves[1:]
    ):
        return None
    try:
        shared_x_limits = curve_home_x_limits(first.x_axis)
    except (TypeError, ValueError):
        # Numeric/strictly-monotonic X is an interactive-CURVE requirement,
        # not a generic Figure requirement.  Categorical/repeated/nonmonotonic
        # curves remain complete encoded panels with ordinary Matplotlib relim.
        return None

    data_min = math.inf
    data_max = -math.inf
    for curve in curves:
        values = np.asarray(curve.values)
        if np.iscomplexobj(values):
            raise ValueError(
                "complex curves require an explicit real-valued display transform"
            )
        valid = np.asarray(curve.validity, dtype=bool)
        if not bool(np.any(valid)):
            continue
        selected = values[valid]
        if not bool(np.all(np.isfinite(selected))):
            raise ValueError("valid curve values must all be finite")
        data_min = min(data_min, float(np.min(selected)))
        data_max = max(data_max, float(np.max(selected)))
    y_limits = (
        (0.0, 1.0)
        if not math.isfinite(data_min)
        else deadband_display_range(
            RelimMode.TIGHT,
            None,
            data_min,
            data_max,
            force=True,
        )
    )
    return shared_x_limits, y_limits


def _draw_projected_image(
    axis,
    figure,
    data: EvaluatedImage,
    *,
    colormap: str,
    color_limits: tuple[float, float],
    visible_bounds=(0.0, 0.0, 1.0, 1.0),
    regular_pixel_contract: bool,
    center: tuple[float, float] | None,
    radius: float | None,
    diagnostic: str | None,
):
    """Draw one exact IMAGE projection with an optional radial-fit annotation."""

    if np.iscomplexobj(data.values):
        raise ValueError("complex images require an explicit real-valued display transform")
    x_edges, _x_centers, x_labels = _image_axis(data.x_axis)
    y_edges, _y_centers, y_labels = _image_axis(data.y_axis)
    invalid = np.logical_not(data.validity)
    if data.values.dtype.kind == "f":
        invalid = np.logical_or(invalid, ~np.isfinite(data.values))
    values = np.ma.array(data.values, mask=invalid)
    if not isinstance(regular_pixel_contract, bool):
        raise TypeError("regular_pixel_contract must be bool")
    if regular_pixel_contract:
        # Typed IMAGE and saved-fit panels have already crossed the strict
        # regular-pixel viewport boundary.  Revalidate that declared contract
        # here before taking the low-memory path; never infer it from shape.
        image_viewport_for_evaluated_image(data)
        if x_labels is not None or y_labels is not None:
            raise ValueError("projected IMAGE export requires numeric pixel axes")
        image_artist = axis.imshow(
            values,
            origin="upper",
            extent=(x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]),
            interpolation="nearest",
            cmap=colormap,
            vmin=color_limits[0],
            vmax=color_limits[1],
            rasterized=True,
        )
        colorbar = figure.colorbar(image_artist, ax=axis)
    else:
        # Canonical/encoded radial figures also admit irregular coordinates.
        # Their geometry must remain cell-edge exact even though QuadMesh is
        # more expensive; silently spreading those cells uniformly would move
        # the physical image underneath an otherwise correct fit overlay.
        image_artist = axis.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="flat",
            cmap=colormap,
            vmin=color_limits[0],
            vmax=color_limits[1],
            rasterized=True,
        )
        colorbar = figure.colorbar(image_artist, ax=axis)
        if x_labels is not None:
            axis.set_xticks(_x_centers, x_labels)
        if y_labels is not None:
            axis.set_yticks(_y_centers, y_labels)
    colorbar.set_label(
        "Value" if data.value_unit is None else f"Value [{data.value_unit}]"
    )
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
    elif diagnostic is not None:
        assert radius is None
        axis.text(
            0.02,
            0.98,
            f"fit {diagnostic}",
            transform=axis.transAxes,
            va="top",
            color=FIT_FAILURE_COLOR,
            fontsize=ANNOTATION_FONT_SIZE,
        )
    else:
        assert radius is None


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
        _draw_projected_image(
            axis,
            figure,
            data,
            colormap="gray",
            color_limits=radial_color_limits,
            regular_pixel_contract=False,
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
    _draw_projected_image(
        axis,
        figure,
        panel.image,
        colormap=display.colormap.value,
        color_limits=color_limits,
        visible_bounds=viewport.visible_bounds,
        regular_pixel_contract=True,
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


def _estimate_projected_image_agg_peak_nbytes(
    images: tuple[EvaluatedImage, ...],
    *,
    dpi: float,
    columns: int,
) -> int:
    """Shared conservative Agg/encoder peak after exact images are retained."""

    images = tuple(images)
    if not images or any(not isinstance(image, EvaluatedImage) for image in images):
        raise TypeError("images must be a non-empty EvaluatedImage tuple")
    dpi = _render_dpi(dpi)
    columns = positive_integer(columns, "columns")
    if columns > len(images):
        raise ValueError("columns cannot exceed the image count")
    rows = math.ceil(len(images) / columns)
    width = math.ceil(5.0 * columns * dpi)
    height = math.ceil(4.0 * rows * dpi)
    rgba_bytes = width * height * 4
    # The exact values/validity planes are retained by the caller and are not
    # charged again here.  The strict regular-pixel ``imshow`` path still owns
    # dtype-independent per-cell mask, normalization/resampling, RGBA and
    # backend workspaces.  Keep that measured term separate from the Agg/PNG
    # canvas fronts and the small one-dimensional edge arrays.  A 2304-square
    # uint16 witness peaked at 373,728,906 incremental bytes; this formula
    # admits it below the 512 MiB product limit with a conservative margin.
    cell_count = sum(image.values.size for image in images)
    axis_coordinate_count = sum(
        len(image.x_axis.coordinates) + len(image.y_axis.coordinates)
        for image in images
    )
    return int(
        _RASTER_FIXED_BYTES
        + _RASTER_BUFFER_MULTIPLIER * rgba_bytes
        + 88 * cell_count
        + 16 * axis_coordinate_count
    )


def estimate_image_png_export_peak_nbytes(
    image: EvaluatedImage,
    *,
    dpi: float = 100.0,
) -> int:
    """Bound incremental PNG export memory after one exact IMAGE is retained."""

    if not isinstance(image, EvaluatedImage):
        raise TypeError("image must be EvaluatedImage")
    image_viewport_for_evaluated_image(image)
    if np.iscomplexobj(image.values):
        raise ValueError("complex images require an explicit real-valued display transform")
    return _estimate_projected_image_agg_peak_nbytes(
        (image,),
        dpi=dpi,
        columns=1,
    )


def _validated_image_panel_export(
    payload: ImagePanelPayload,
    display: ImageDisplayState,
) -> None:
    if not isinstance(payload, ImagePanelPayload):
        raise TypeError("payload must be ImagePanelPayload")
    if not isinstance(display, ImageDisplayState):
        raise TypeError("display must be ImageDisplayState")
    home_viewport = image_viewport_for_evaluated_image(payload.image)
    expected_viewport = image_viewport_for_display_state(display, home_viewport)
    if expected_viewport != payload.viewport:
        raise ValueError("image display state differs from the exact payload viewport")
    if payload.base_palette != indexed_colormap(display.colormap.value):
        raise ValueError("image display colormap differs from the exact payload palette")
    if (
        display.relim_mode is RelimMode.FIXED
        and display.fixed_color_limits != payload.color_limits
    ):
        raise ValueError("fixed image display limits differ from the exact payload front")


def _image_panel_fit_annotation(
    payload: ImagePanelPayload,
) -> tuple[
    tuple[float, float] | None,
    float | None,
    str | None,
    str | None,
]:
    """Project one immutable radial-fit DTO through the exact payload view.

    The Qt front and this headless export share the same authority boundary:
    fit geometry is expressed in the declared coordinate frame and is mapped
    by :class:`ImageViewportTransform`.  Mapping the resulting visible point
    and span back through the imshow extent keeps descending axes, cropped
    viewports, and half-cell edges exact without letting Matplotlib infer a
    pixel convention from array shape.

    The returned tuple is ``(center, radius, diagnostic, title)``.  Failed and
    sparse cells intentionally return no geometry, so a diagnostic can never
    inherit a successful centre/ring.
    """

    overlay = payload.fit_overlay
    if overlay is None:
        return None, None, None, None
    status_label = (
        "NOT_PRESENT" if overlay.status is None else overlay.status.value
    )
    title = f"{overlay.caption} $\\cdot$ {status_label}"
    if overlay.status is not FitBatchStatus.CONVERGED:
        return None, None, overlay.diagnostic or status_label, title

    center = overlay.center_xy
    radius = overlay.one_over_e_radius
    if center is None or radius is None:
        raise RuntimeError("converged radial fit overlay lost its geometry")
    viewport = payload.viewport
    visible_center = viewport.unbounded_visible_point_for_coordinate(
        center,
        coordinate_frame=overlay.coordinate_frame,
    )
    visible_diameter = viewport.visible_span_for_coordinate_span(
        (2.0 * radius, 2.0 * radius),
        coordinate_frame=overlay.coordinate_frame,
    )

    x_edges, _x_centers, _x_labels = _image_axis(payload.image.x_axis)
    y_edges, _y_centers, _y_labels = _image_axis(payload.image.y_axis)
    left, top, right, bottom = viewport.visible_bounds
    x_full_start, x_full_stop = float(x_edges[0]), float(x_edges[-1])
    y_full_start, y_full_stop = float(y_edges[0]), float(y_edges[-1])
    x_view_start = x_full_start + left * (x_full_stop - x_full_start)
    x_view_stop = x_full_start + right * (x_full_stop - x_full_start)
    y_view_top = y_full_start + top * (y_full_stop - y_full_start)
    y_view_bottom = y_full_start + bottom * (y_full_stop - y_full_start)
    projected_center = (
        x_view_start + visible_center[0] * (x_view_stop - x_view_start),
        y_view_top + visible_center[1] * (y_view_bottom - y_view_top),
    )
    projected_diameters = (
        visible_diameter[0] * abs(x_view_stop - x_view_start),
        visible_diameter[1] * abs(y_view_bottom - y_view_top),
    )
    tolerance = 16.0 * math.ulp(
        max(1.0, abs(projected_diameters[0]), abs(projected_diameters[1]))
    )
    if not math.isclose(
        projected_diameters[0],
        projected_diameters[1],
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        raise ValueError("radial fit viewport mapping produced anisotropic geometry")
    return projected_center, projected_diameters[0] / 2.0, None, title


def save_image_panel_png(
    payload: ImagePanelPayload,
    display: ImageDisplayState,
    destination,
    *,
    dpi: float = 100.0,
    memory_limit_bytes: int | None = None,
) -> None:
    """Save one exact current IMAGE front without re-evaluation or fit authority.

    The committed viewport, colormap, effective colour limits, and value unit
    all come from the exact payload/display pair.  An attached immutable radial
    fit DTO is exported as its centre/radius and status only; no fitted image or
    contour is synthesized.  Pointer-drag rectangles remain absent because
    they are transient Qt overlays, not committed display state.
    """

    _validated_image_panel_export(payload, display)
    center, radius, diagnostic, title = _image_panel_fit_annotation(payload)
    dpi = _render_dpi(dpi)
    required = estimate_image_png_export_peak_nbytes(payload.image, dpi=dpi)
    if memory_limit_bytes is not None:
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        if required > limit:
            raise MemoryError(
                f"image panel PNG export peak {required} exceeds limit {limit}"
            )

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = None
    try:
        with render_style_context():
            figure = Figure(figsize=(5.0, 4.0), dpi=dpi, constrained_layout=True)
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            _draw_projected_image(
                axis,
                figure,
                payload.image,
                colormap=display.colormap.value,
                color_limits=payload.color_limits,
                visible_bounds=payload.viewport.visible_bounds,
                regular_pixel_contract=True,
                center=center,
                radius=radius,
                diagnostic=diagnostic,
            )
            if title is not None:
                axis.set_title(title)
            figure.savefig(destination, format="png", dpi=dpi)
    finally:
        if figure is not None:
            release_agg_figure(figure)
        figure = None
        gc.collect()


def _pulse_time_unit(span_s: float) -> tuple[float, str]:
    """Seconds -> (scale, unit) so a pulse timeline reads in ns / us / ms / s.

    The same thresholds the reference uses, so the x axis label and tick values
    match: sub-us spans stay in ns, sub-ms in us, sub-s in ms, else seconds."""
    span = abs(float(span_s))
    if span < 1e-6:
        return 1e-9, "ns"
    if span < 1e-3:
        return 1e-6, "us"
    if span < 1.0:
        return 1e-3, "ms"
    return 1.0, "s"


def _draw_pulse_repeat_brackets(axis, repeat_markers, n_channels, colors) -> None:
    """The grey square brackets that enclose a repeated span, with its ``×N`` /
    ``×∞`` label, exactly as the reference draws them: two vertical stems with
    short inward feet at each end, nested outward for multiple brackets."""

    import numpy as _np

    from .render_style import smaller_fontsize

    markers = [m for m in repeat_markers if m is not None]
    if not markers:
        return
    xlim = axis.get_xlim()
    span = max(float(xlim[1] - xlim[0]), 1e-12)
    tick_base = span * 0.024
    bracket_count = max(1, len(markers))
    for index, marker in enumerate(markers):
        try:
            start, stop, label = float(marker[0]), float(marker[1]), str(marker[2])
        except Exception:
            continue
        if not _np.isfinite(start) or not _np.isfinite(stop) or stop <= start:
            continue
        color = colors[index % len(colors)]
        alpha = 0.58
        outer_depth = max(0, bracket_count - 1 - index)
        y_low = -0.42 - 0.13 * outer_depth
        y_high = float(n_channels) - 0.10 + 0.34 * outer_depth
        tick = min(tick_base, max(stop - start, 0.0) * 0.2)
        tick = max(tick, span * 0.006)
        axis.plot([start + tick, start, start, start + tick], [y_high, y_high, y_low, y_low],
                  color=color, alpha=alpha, linewidth=1.05, solid_capstyle="round",
                  clip_on=True, zorder=8 + index)
        axis.plot([stop - tick, stop, stop, stop - tick], [y_high, y_high, y_low, y_low],
                  color=color, alpha=alpha, linewidth=1.05, solid_capstyle="round",
                  clip_on=True, zorder=8 + index)
        if label:
            # Label placement matches the reference EXACTLY: just to the RIGHT of the right stem
            # (``stop + tick*0.12``, left-aligned), not centred above the span.  DejaVu Sans supplies
            # the U+221E glyph the design's Helvetica Light lacks, so ×∞ reads as infinity, not tofu.
            axis.text(stop + tick * 0.12, y_high + 0.055, label,
                      ha="left", va="bottom", color=color, alpha=alpha, fontfamily="DejaVu Sans",
                      fontsize=smaller_fontsize(0.8, 5.5), clip_on=False, zorder=9 + index)


def render_pulse_timeline_png(
    *,
    pulses,
    channels,
    channel_labels,
    total_duration: float,
    title: str = "",
    repeat_markers=(),
    repeat_notation: str = "",
    size: str = "2x2",
    analog_traces=(),
    scan_regions=(),
    scan_dac_segments=(),
) -> bytes:
    """The pulse-timeline figure as PNG bytes -- the reference's faithful preview.

    Each digital channel is one row: a coloured OFF baseline with the ON spans
    drawn as FILLED blocks (``Rectangle`` patches), the board name on the y axis
    (tinted to its row), a ``Time (unit)`` x axis, the pulse name inside long
    blocks, and the repeat span shown as grey brackets (or a top-right ``×∞`` note).

    ``analog_traces`` fold each DAC bus into ONE extra row below the digital
    channels: a dashed 0 V reference where the SIGNED zero maps inside the row
    (mid-row for a bipolar bus) plus a solid step/staircase line of the actual
    values -- dicts of ``{name, label, min, max, starts, values}`` with ``starts``
    in seconds (one more than ``values``) and signed values.

    ``scan_regions`` shade the time span a scanned DURATION slot affects in
    transparent orange across every channel; ``scan_dac_segments`` draw a thick
    orange level segment on the affected bus row -- dicts of
    ``{start, stop, number}`` and ``{trace_name, start, stop, value, number}``,
    each carrying its 1-based slot number drawn once as a circled badge.

    ``size`` is one of the panel-size presets (``PANEL_SIZES``); the figure geometry is derived from
    it through the frontend's ONE size source (:func:`panel_figure_size_inches`), so the preview
    rescales with the preset exactly like every other panel kind -- the raster is emitted at the same
    on-screen resolution a panel of that size reserves, never a bespoke inch/dpi pair.

    Plain data only (no ``PulseTableState``/pulse imports) so the plot layer stays
    free of the domain packages; the editor extracts these lists from its sequence.
    """

    import io

    import matplotlib

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    from .render_style import (
        DESIGN_DPI, PALETTE, PANEL_DISPLAY_SCALE, PULSE_SCAN_ANNOTATION_COLOR,
        PULSE_SCAN_ANNOTATION_FONT_SIZE, PULSE_SCAN_REGION_COLOR, apply_title,
        panel_figure_size_inches, render_style_context, smaller_fontsize,
    )

    # The size PRESET is the one geometry knob: inches come from the frontend's single size source and
    # the raster dpi matches a panel's on-screen scale, so the emitted PNG is exactly the pixel box a
    # card of this size reserves (panel_display_size(size)) -- preview, Save Figure and a seeded panel
    # all agree, and a bigger preset yields a bigger picture.
    size_inches = panel_figure_size_inches(size)
    dpi = _render_dpi(DESIGN_DPI * PANEL_DISPLAY_SCALE)
    pulses = [dict(row) for row in pulses]
    channels = [str(channel) for channel in channels]
    labels = dict(channel_labels or {})
    colors = list(PALETTE["pulse_cycle"])
    bracket_colors = list(PALETTE["bracket_cycle"])

    # DAC buses draw BELOW the digital channels, one row each; every row key -- a
    # channel name or an analog trace key -- shares one index/colour space so the
    # colour cycle simply continues past the channels (the reference's layout).
    analog_traces = [dict(trace) for trace in analog_traces]
    analog_keys = [f"analog:{i}:{trace.get('name', 'analog')}"
                   for i, trace in enumerate(analog_traces)]
    row_keys = channels + analog_keys
    n_channels = len(channels)
    n_rows = len(row_keys)
    index_map = {key: n_rows - 1 - i for i, key in enumerate(row_keys)}
    color_map = {key: colors[i % len(colors)] for i, key in enumerate(row_keys)}
    row_height = 0.64 if n_rows <= 10 else max(0.42, 6.4 / max(1, n_rows))

    # The axis spans the whole FRAME (total_duration), not just to the last pulse edge --
    # a trailing all-off stretch is real programme time and must stay visible.
    start_min = min([0.0] + [float(row["start"]) for row in pulses])
    stop_max = max([1e-12, float(total_duration)] + [float(row["stop"]) for row in pulses])
    bracket_bounds = []
    for marker in repeat_markers:
        try:
            b_start, b_stop = float(marker[0]), float(marker[1])
        except Exception:
            continue
        if b_stop > b_start:
            bracket_bounds.append((b_start, b_stop))
            start_min = min(start_min, b_start)
            stop_max = max(stop_max, b_stop)
    span = max(stop_max - start_min, 1e-12)
    scale, unit = _pulse_time_unit(span)
    margin_x = max(span * 0.04, 1e-12)
    left_limit = start_min - margin_x
    right_limit = stop_max + margin_x
    if bracket_bounds:
        right_limit += span * 0.05

    figure = None
    try:
        with render_style_context():
            figure = Figure(figsize=size_inches, dpi=dpi, constrained_layout=True)
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            axis.set_ylabel("")
            baseline_offset = row_height / 2
            pulse_zorder = 3
            baseline_y = {}
            for channel in channels:
                y = index_map[channel]
                baseline_y[channel] = y - baseline_offset
                axis.hlines(baseline_y[channel], left_limit, right_limit,
                            color=color_map[channel], linewidth=0.65, alpha=1.0, zorder=pulse_zorder)
            for row in pulses:
                if not row.get("value") or row["channel"] not in index_map or row["duration"] <= 0:
                    continue
                channel = row["channel"]
                axis.add_patch(Rectangle(
                    (row["start"], baseline_y[channel]), row["duration"], row_height,
                    facecolor=color_map[channel], edgecolor="none", linewidth=0.0,
                    alpha=1.0, zorder=pulse_zorder))
                if row.get("name") and row["duration"] >= 0.09 * max(total_duration, 1e-12):
                    axis.text(row["start"] + row["duration"] / 2.0, index_map[channel], str(row["name"]),
                              ha="center", va="center", color=PALETTE["pulse_name"],
                              fontsize=smaller_fontsize(1.2, 4.8), clip_on=True, zorder=pulse_zorder + 1)

            # DAC bus rows: a dashed 0 V reference where SIGNED zero maps inside the row
            # (mid-row for a bipolar bus), plus a solid staircase of the actual values --
            # same weight and opacity as the digital off-lines, per the reference.
            analog_zero_y: dict[str, float] = {}
            analog_geom: dict[str, tuple[float, int, int]] = {}
            for key, trace in zip(analog_keys, analog_traces):
                y_base = index_map[key] - baseline_offset
                baseline_y[key] = y_base
                v_max = int(trace.get("max", 1))
                v_min = int(trace.get("min", 0))
                v_span = max(1, v_max - v_min)
                zero_frac = min(1.0, max(0.0, (0.0 - v_min) / v_span))
                zero_y = y_base + row_height * zero_frac
                name = str(trace.get("name", key))
                analog_zero_y[name] = zero_y
                analog_geom[name] = (y_base, v_min, v_span)
                color = color_map[key]
                axis.plot([left_limit, right_limit], [zero_y, zero_y],
                          color=color, linewidth=0.65, alpha=0.5, linestyle=(0, (4, 3)),
                          zorder=pulse_zorder + 1)
                starts = np.asarray(trace.get("starts", []), dtype=float)
                values = np.asarray(trace.get("values", []), dtype=float)
                if starts.size >= 2 and values.size >= 1:
                    x = starts[: values.size + 1]
                    y_values = y_base + row_height * np.clip(
                        (values[: x.size - 1] - v_min) / v_span, 0.0, 1.0)
                    axis.plot(np.repeat(x, 2)[1:-1], np.repeat(y_values, 2),
                              color=color, linewidth=0.65, alpha=1.0, zorder=pulse_zorder + 2)
                else:
                    axis.plot([left_limit, right_limit], [zero_y, zero_y],
                              color=color, linewidth=0.65, alpha=1.0, zorder=pulse_zorder + 2)

            # Scanned regions, exactly the reference's annotation: a transparent orange
            # band across every row for a scanned DURATION, a thick orange level segment
            # on the bus row for a scanned DAC value, each numbered once with a circled
            # badge so several slots stay tellable apart.
            badge_kwargs = dict(
                ha="center", va="center", color=PULSE_SCAN_ANNOTATION_COLOR,
                fontsize=max(2.6, float(PULSE_SCAN_ANNOTATION_FONT_SIZE)), zorder=12,
                bbox=dict(boxstyle="circle,pad=0.3", facecolor=PULSE_SCAN_REGION_COLOR,
                          edgecolor="none"))
            area_bottom = min(baseline_y.values()) if baseline_y else -baseline_offset
            area_top = (max(baseline_y.values()) + row_height) if baseline_y else row_height
            for region in scan_regions:
                r_start, r_stop = float(region["start"]), float(region["stop"])
                if r_stop <= r_start:
                    continue
                axis.add_patch(Rectangle(
                    (r_start, area_bottom), r_stop - r_start, area_top - area_bottom,
                    facecolor=PULSE_SCAN_REGION_COLOR, edgecolor="none", alpha=0.18,
                    zorder=6))
                number = region.get("number")
                if number is not None:
                    axis.text((r_start + r_stop) / 2.0, area_top - row_height / 2.0,
                              str(number), **badge_kwargs)
            for segment in scan_dac_segments:
                name = str(segment.get("trace_name", ""))
                geom = analog_geom.get(name)
                if geom is None:
                    continue
                y_base, v_min, v_span = geom
                s_start, s_stop = float(segment["start"]), float(segment["stop"])
                level = y_base + row_height * min(1.0, max(0.0, (float(segment["value"]) - v_min) / v_span))
                axis.plot([s_start, s_stop], [level, level],
                          color=PULSE_SCAN_REGION_COLOR, linewidth=3.0, alpha=0.9, zorder=8)
                number = segment.get("number")
                if number is not None:
                    axis.text((s_start + s_stop) / 2.0, y_base + row_height / 2.0,
                              str(number), **badge_kwargs)

            axis.set_xlim(left_limit, right_limit)
            ylim_top = n_rows - 0.38
            if bracket_bounds:
                ylim_top = n_rows + 0.78 + 0.26 * max(0, len(bracket_bounds) - 1)
            axis.set_ylim(-0.62, ylim_top)
            axis.set_yticks([index_map[key] for key in row_keys])
            axis.set_yticklabels(
                [labels.get(channel, channel) for channel in channels]
                + [str(trace.get("label") or trace.get("name") or "analog")
                   for trace in analog_traces])
            # Channel names sit ONE row apart, so the reference shrinks the y-tick label a notch below
            # the stock tick size (ytick.labelsize - 1.2, floored at 4.8) to keep long board names from
            # crowding their neighbours -- match it exactly.
            axis.tick_params(
                axis="y",
                labelsize=max(4.8, matplotlib.rcParams["ytick.labelsize"] - 1.2))
            for tick, key in zip(axis.get_yticklabels(), row_keys):
                tick.set_color(color_map[key])
            axis.set_xlabel(f"Time ({unit})")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
            # Blank the cosmetic negative-time headroom ticks (the left margin lets a t=0 edge breathe
            # off the spine); the reference never prints a negative tick, so nor does the preview.
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _pos, s=scale: "" if value < 0 else f"{value / s:.4g}"))
            axis.tick_params(axis="x", which="both", bottom=True, top=False,
                             labelbottom=True, labeltop=False, pad=2)
            axis.set_axisbelow(True)
            axis.grid(axis="x", color=PALETTE["pulse_grid"], linewidth=0.35, zorder=0)
            axis.spines[["top", "right"]].set_visible(False)
            if repeat_notation and not bracket_bounds:
                axis.text(0.995, 1.012, str(repeat_notation), transform=axis.transAxes,
                          ha="right", va="bottom", color=PALETTE["pulse_repeat_note"],
                          fontsize=smaller_fontsize(1.0, 5.5))
            _draw_pulse_repeat_brackets(axis, repeat_markers, n_rows, bracket_colors)
            if title:
                # The ONE title mechanism (apply_title = title_fontsize, the stock label size); NOT
                # axis.set_title(), whose default 'large' titlesize dwarfs the compact preview and
                # crowds the repeat bracket right below it.
                apply_title(axis, str(title))
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=dpi)
            return buffer.getvalue()
    finally:
        if figure is not None:
            release_agg_figure(figure)
        gc.collect()


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
    return _estimate_projected_image_agg_peak_nbytes(
        tuple(panel.image for panel in prepared),
        dpi=dpi,
        columns=columns,
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

    bin_projection = (
        _histogram_projection(series_group, bins)
        if projection is None
        else projection
    )
    counts_group = bin_projection.bin_counts
    edges = bin_projection.bin_edges
    _draw_histogram_projection(axis, series_group, counts_group, edges)
    return counts_group, edges


def _draw_histogram_projection(axis, series_group, counts_group, edges) -> None:
    """Draw one panel from already-frozen shared histogram geometry."""

    multiple_series = len(series_group) > 1
    all_boolean = all(
        isinstance(series.data, EvaluatedHistogram)
        and np.issubdtype(series.data.samples.dtype, np.bool_)
        for series in series_group
    )
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


def _meter_text(series_group) -> str:
    lines = []
    multiple_series = len(series_group) > 1
    for series in series_group:
        data = series.data
        assert isinstance(data, EvaluatedMeter)
        label = _series_label(series, include_reductions=multiple_series) or ""
        if data.valid and not _finite_numeric_scalar(data.value):
            raise ValueError("valid meter values must be finite")
        if data.valid:
            value = (
                "true"
                if data.value is True
                else "false"
                if data.value is False
                else format(Decimal(int(data.value)), ".6g")
                if isinstance(data.value, Integral)
                else format(data.value, ".6g")
            )
            if data.value_unit is not None:
                value = f"{value} {data.value_unit}"
        else:
            value = "invalid"
        lines.append(f"{label}: {value}" if label else value)
    return "\n".join(lines)


def _finite_numeric_scalar(value: Number) -> bool:
    if isinstance(value, Integral):
        return True
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
    shared_histogram_projection = None
    shared_histogram_x_limits = None
    shared_histogram_count_limits = None
    if (
        len({layer.layer_id for layer, _cell, _series in panels}) == 1
        and all(
            isinstance(series.data, EvaluatedHistogram)
            for _layer, _cell, series_group in panels
            for series in series_group
        )
    ):
        shared_histogram_projection = HistogramBinProjection(
            tuple(
                series.data.samples
                for _layer, _cell, series_group in panels
                for series in series_group
            ),
            bins=DEFAULT_HISTOGRAM_BINS,
        )
        shared_histogram_x_limits = histogram_home_x_limits(
            shared_histogram_projection.bin_edges
        )
        shared_histogram_count_limits = histogram_count_limits(
            HistogramDisplayState(),
            max(
                (
                    int(np.max(counts, initial=0))
                    for counts in shared_histogram_projection.bin_counts
                ),
                default=0,
            ),
        )
    shared_curve_limits = _shared_curve_limits(panels, fit_results)
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

        histogram_series_offset = 0
        for target, (layer, cell, series_group) in zip(axes, panels):
            fit_result = fit_results.get(layer.layer_id)
            kind = series_group[0].data
            if isinstance(kind, EvaluatedCurve):
                _curve(target, layer, cell, series_group, fit_result)
                if shared_curve_limits is not None:
                    shared_x_limits, shared_y_limits = shared_curve_limits
                    target.set_xlim(*shared_x_limits)
                    target.set_ylim(*shared_y_limits)
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
                if shared_histogram_projection is None:
                    _histogram(target, series_group)
                else:
                    next_offset = histogram_series_offset + len(series_group)
                    _draw_histogram_projection(
                        target,
                        series_group,
                        shared_histogram_projection.bin_counts[
                            histogram_series_offset:next_offset
                        ],
                        shared_histogram_projection.bin_edges,
                    )
                    histogram_series_offset = next_offset
                    assert shared_histogram_x_limits is not None
                    assert shared_histogram_count_limits is not None
                    target.set_xlim(*shared_histogram_x_limits)
                    target.set_ylim(*shared_histogram_count_limits)
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
        "_fit_artists",
        "_fit_diagnostic_artists",
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
        self._fit_artists = ()
        self._fit_diagnostic_artists = ()
        self._topology = None

    def render(self, evaluated: EvaluatedFigureData) -> RasterBuffer:
        with render_style_context():
            return self._render(evaluated)

    def _render(self, evaluated: EvaluatedFigureData) -> RasterBuffer:
        figure, axis, layer, _cell, series_group = self._prepare_panel(evaluated)
        first = series_group[0].data
        if isinstance(first, EvaluatedCurve):
            self._update_curve_fit_artists(axis, layer, series_group, ())
        if isinstance(first, (EvaluatedCurve, EvaluatedHistogram)):
            axis.set_autoscalex_on(True)
            axis.set_autoscaley_on(True)
            axis.relim(visible_only=True)
            axis.autoscale_view()
        return self._draw_raster(figure)

    def render_meter(
        self,
        evaluated: EvaluatedFigureData,
        *,
        display_revision: int,
    ) -> tuple[RasterBuffer, MeterPanelPayload]:
        """Render one exact display-only METER front and its immutable payload."""

        revision = nonnegative_integer(display_revision, "meter display_revision")
        with render_style_context():
            layer, _cell, series_group = self._one_panel(evaluated)
            if not series_group or any(
                not isinstance(series.data, EvaluatedMeter)
                for series in series_group
            ):
                raise ValueError("typed meter render requires METER series")
            value_unit = series_group[0].data.value_unit
            if any(
                series.data.value_unit != value_unit
                for series in series_group[1:]
            ):
                raise ValueError("typed meter series must share value_unit")
            # Formatting is pure and performs the valid/non-finite check before
            # the persistent Agg surface can acquire or mutate artists.
            _meter_text(series_group)
            evaluated_input = self._evaluated_input_for_layer(evaluated, layer)
            figure, _axis, prepared_layer, _cell, prepared_series = (
                self._prepare_panel(evaluated)
            )
            if prepared_layer is not layer or prepared_series is not series_group:
                raise RuntimeError("meter panel identity changed during preparation")
            raster = self._draw_raster(figure)
            labels = self._series_labels(layer.layer_id, series_group)
            return raster, MeterPanelPayload(
                evaluated_input,
                revision,
                tuple(series_group),
                labels,
            )

    def render_interactive_curve(
        self,
        evaluated: EvaluatedFigureData,
        state: CurveDisplayState,
        *,
        current_y_limits: tuple[float, float] | None,
        previous_relim_mode: RelimMode | None,
        fit_overlays: tuple[CurveFitOverlay, ...] = (),
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
                fit_overlays=fit_overlays,
            )

    def _render_interactive_curve(
        self,
        evaluated: EvaluatedFigureData,
        state: CurveDisplayState,
        *,
        current_y_limits: tuple[float, float] | None,
        previous_relim_mode: RelimMode | None,
        fit_overlays: tuple[CurveFitOverlay, ...],
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
        evaluated_input = self._evaluated_input_for_layer(evaluated, pre_layer)
        fit_overlays = _validated_curve_fit_overlays(
            evaluated_input,
            tuple(pre_series_group),
            fit_overlays,
        )

        figure, axis, layer, _cell, series_group = self._prepare_panel(evaluated)
        if layer is not pre_layer or series_group is not pre_series_group:
            raise RuntimeError("interactive panel identity changed during preparation")
        self._update_curve_fit_artists(
            axis,
            layer,
            series_group,
            fit_overlays,
        )

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
        labels = self._series_labels(layer.layer_id, series_group)
        return raster, CurvePanelPayload(
            evaluated_input,
            viewport,
            tuple(series_group),
            labels,
            fit_overlays,
        )

    def render_interactive_histogram(
        self,
        evaluated: EvaluatedFigureData,
        state: HistogramDisplayState,
        *,
        current_count_limits: tuple[float, float] | None,
        previous_relim_mode: RelimMode | None,
        previous_count_scale: HistogramCountScale | None,
        projection_value_range: tuple[float, float] | None = None,
    ) -> tuple[RasterBuffer, HistogramPanelPayload]:
        """Render one front-bound shared-bin Histogram projection."""

        with render_style_context():
            return self._render_interactive_histogram(
                evaluated,
                state,
                current_count_limits=current_count_limits,
                previous_relim_mode=previous_relim_mode,
                previous_count_scale=previous_count_scale,
                projection_value_range=projection_value_range,
            )

    def _render_interactive_histogram(
        self,
        evaluated: EvaluatedFigureData,
        state: HistogramDisplayState,
        *,
        current_count_limits: tuple[float, float] | None,
        previous_relim_mode: RelimMode | None,
        previous_count_scale: HistogramCountScale | None,
        projection_value_range: tuple[float, float] | None,
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
            value_range=projection_value_range,
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
        labels = self._series_labels(layer.layer_id, series_group)
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
            series_topology = tuple(
                (series.batch_address, series.data.value_unit)
                for series in series_group
            )
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
                fit_artists = []
                diagnostic_artists = []
                for index, source_artist in enumerate(self._artists):
                    fit_artist, = axis.plot(
                        (),
                        (),
                        color=source_artist.get_color(),
                        linestyle=FIT_LINESTYLE,
                        marker=None,
                        label="_nolegend_",
                    )
                    fit_artists.append(fit_artist)
                    diagnostic_artists.append(
                        axis.text(
                            0.02,
                            0.98 - 0.06 * index,
                            "",
                            transform=axis.transAxes,
                            va="top",
                            color=FIT_FAILURE_COLOR,
                            fontsize=ANNOTATION_FONT_SIZE,
                        )
                    )
                self._fit_artists = tuple(fit_artists)
                self._fit_diagnostic_artists = tuple(diagnostic_artists)
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
                if (
                    len(self._fit_artists) != len(series_group)
                    or len(self._fit_diagnostic_artists) != len(series_group)
                ):
                    raise RuntimeError("progressive fit artist topology changed")
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
        if isinstance(first, EvaluatedCurve):
            # A prior accepted overlay may have added fit handles.  Rebuild
            # from the currently accepted source artists here, then let
            # ``_update_curve_fit_artists`` add only this revision's converged
            # fits.  This keeps legend state from becoming a hidden cache.
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
            if multiple_series or any(
                series.batch_address for series in series_group
            ):
                axis.legend(
                    handles=self._artists,
                    labels=tuple(line.get_label() for line in self._artists),
                    fontsize=ANNOTATION_FONT_SIZE,
                )
        elif isinstance(first, EvaluatedHistogram) and (
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

    def _update_curve_fit_artists(
        self,
        axis,
        layer,
        series_group,
        overlays: tuple[CurveFitOverlay, ...],
    ) -> None:
        """Replace the complete fit front; absent/failed rows clear old lines."""

        overlays = tuple(overlays)
        if overlays and len(overlays) != len(series_group):
            raise ValueError("curve fit overlays must align one-for-one with series")
        if (
            len(self._fit_artists) != len(series_group)
            or len(self._fit_diagnostic_artists) != len(series_group)
        ):
            raise RuntimeError("curve fit artist topology differs from its series")
        labels = self._series_labels(layer.layer_id, series_group)
        active_fit_artists = []
        for index, (fit_artist, diagnostic_artist, series, label) in enumerate(
            zip(
                self._fit_artists,
                self._fit_diagnostic_artists,
                series_group,
                labels,
                strict=True,
            )
        ):
            overlay = None if not overlays else overlays[index]
            if overlay is not None and overlay.status is FitBatchStatus.CONVERGED:
                curve = series.data
                assert isinstance(curve, EvaluatedCurve)
                start, stop = overlay.source_sample_span
                fit_artist.set_data(
                    np.asarray(
                        curve.x_axis.coordinates[start:stop],
                        dtype=np.float64,
                    ),
                    overlay.predicted_y,
                )
                fit_artist.set_visible(True)
                fit_artist.set_label(f"fit {label}")
                diagnostic_artist.set_text("")
                active_fit_artists.append(fit_artist)
            else:
                # Clearing both data and visibility is intentional: either fact
                # alone is too easy for a later Matplotlib mutation to undo.
                fit_artist.set_data((), ())
                fit_artist.set_visible(False)
                fit_artist.set_label("_nolegend_")
                diagnostic_artist.set_text(
                    ""
                    if overlay is None
                    else f"fit {label}: {overlay.diagnostic}"
                )

        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        if len(series_group) > 1 or any(
            series.batch_address for series in series_group
        ):
            handles = (*self._artists, *active_fit_artists)
            axis.legend(
                handles=handles,
                labels=tuple(item.get_label() for item in handles),
                fontsize=ANNOTATION_FONT_SIZE,
            )

    @staticmethod
    def _evaluated_input_for_layer(evaluated: EvaluatedFigureData, layer):
        matches = tuple(
            item for item in evaluated.inputs if item.dataset_id == layer.dataset_id
        )
        if len(matches) != 1:
            raise ValueError(
                "interactive curve layer requires one exact evaluated input"
            )
        return matches[0]

    @staticmethod
    def _series_labels(layer_id: str, series_group) -> tuple[str, ...]:
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
        return RasterBuffer.from_agg_rgba(
            actual_width, actual_height, figure.canvas.buffer_rgba())

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
        self._fit_artists = ()
        self._fit_diagnostic_artists = ()
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
    "estimate_image_png_export_peak_nbytes",
    "estimate_live_panel_raster_peak_nbytes",
    "estimate_projected_radial_fit_render_peak_nbytes",
    "estimate_render_peak_nbytes",
    "render_radial_gaussian_image_fit_panels",
    "render_evaluated_figure",
    "release_agg_figure",
    "save_image_panel_png",
    "save_radial_gaussian_image_fit_panels",
    "save_evaluated_figure",
    "SinglePanelAggRenderer",
]
