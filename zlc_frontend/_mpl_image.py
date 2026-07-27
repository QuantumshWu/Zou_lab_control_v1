"""Internal Matplotlib implementation owner: image."""

from __future__ import annotations

import gc
from io import BytesIO
import math
import threading
from numbers import Number
import matplotlib
import numpy as np
from zlc_data import FitBatchStatus, FitResultBatch
from zlc_storage import positive_integer
from .figure import (
    EvaluatedImage,
    EvaluatedProjectionIdentity,
)
from .fit_image_projection import (
    RadialGaussianImageFitPanel,
    radial_gaussian_fit_geometry,
)
from .fit_projection import (
    evaluated_figure_panels as _panels,
    fit_batch_storage_index as _batch_storage_index,
)
from .image_display import (
    ImageDisplayState,
    evaluated_image_data_range,
    image_viewport_for_display_state,
)
from .image_view import ImageViewportTransform, image_viewport_for_evaluated_image
from .display_range import (
    RelimMode,
    deadband_display_range,
    validated_display_range,
)
from .render import (
    ImagePanelRasterGeometry,
    RadialGaussianImageFitOverlay,
    RasterBuffer,
    _validated_curve_fit_overlays,
)
from .axis_display import axis_label as _axis_label
from .plot_layout import (
    image_panel_layout,
    image_panel_layout_for_raster,
    LIVE_PANEL_DPI,
    panel_figure_size_inches,
)
from .render_style import (
    ANNOTATION_FONT_SIZE,
    FIT_CONTOUR_COLOR,
    FIT_CONTOUR_LINEWIDTH,
    FIT_FAILURE_COLOR,
    FIT_RADIAL_CENTER_SIZE,
    FIT_RADIAL_COLOR,
    FIT_RADIAL_RING_ALPHA,
    FIT_RADIAL_RING_LINEWIDTH,
    PALETTE,
    SITE_OCCUPANCY_STYLE,
    apply_title,
    render_style_context,
    small_fontsize,
)
from .site_map import (
    SITE_INVALID_ALPHA,
    SITE_INVALID_COLOR,
    SITE_INVALID_LINEWIDTH,
)

from ._mpl_common import (
    _AggBlitCache,
    _agg_chrome_key,
    _agg_layout_key,
    _fit_status,
    _raster_from_drawn_agg,
    _render_dpi,
    raster_from_agg,
    release_agg_figure,
)

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

def _draw_projected_image(
    axis,
    figure,
    data: EvaluatedImage,
    *,
    colormap: str,
    color_limits: tuple[float, float],
    visible_bounds=(0.0, 0.0, 1.0, 1.0),
    regular_axis_contract: bool,
    center: tuple[float, float] | None,
    radius: float | None,
    diagnostic: str | None,
    show_colorbar: bool = True,
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
    if not isinstance(regular_axis_contract, bool):
        raise TypeError("regular_axis_contract must be bool")
    if regular_axis_contract:
        # Typed IMAGE and saved-fit panels have already crossed the strict
        # regular numeric-axis viewport boundary.  Revalidate that contract
        # here before projection; never infer it from shape.
        image_viewport_for_evaluated_image(data)
        if x_labels is not None or y_labels is not None:
            raise ValueError("projected IMAGE export requires numeric axes")
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
        colorbar = figure.colorbar(image_artist, ax=axis) if show_colorbar else None
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
        colorbar = figure.colorbar(image_artist, ax=axis) if show_colorbar else None
        if x_labels is not None:
            axis.set_xticks(_x_centers, x_labels)
        if y_labels is not None:
            axis.set_yticks(_y_centers, y_labels)
    if colorbar is not None:
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
    axis.set_anchor("W")
    axis.set_aspect("equal", adjustable="box")
    if center is not None:
        from matplotlib.patches import Circle

        assert radius is not None and diagnostic is None
        axis.scatter(
            *center,
            color=FIT_RADIAL_COLOR,
            s=FIT_RADIAL_CENTER_SIZE,
            clip_on=True,
        )
        axis.add_patch(
            Circle(
                center,
                radius=radius,
                edgecolor=FIT_RADIAL_COLOR,
                facecolor="none",
                linewidth=FIT_RADIAL_RING_LINEWIDTH,
                alpha=FIT_RADIAL_RING_ALPHA,
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
            regular_axis_contract=False,
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
        regular_axis_contract=True,
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

def render_radial_gaussian_image_fit_panels(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float],
    *,
    columns: int,
    dpi: float = 100.0,
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

def encode_radial_gaussian_image_fit_panels(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float],
    *,
    image_format: str,
    columns: int,
    dpi: float = 100.0,
) -> bytes:
    """Encode the exact committed typed page/focus display.

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
    output = BytesIO()
    try:
        figure = render_radial_gaussian_image_fit_panels(
            panels,
            display,
            current_color_limits,
            columns=columns,
            dpi=dpi,
        )
        with render_style_context():
            figure.savefig(output, format=image_format, dpi=dpi)
    finally:
        if figure is not None:
            release_agg_figure(figure)
        figure = None
        gc.collect()
    return output.getvalue()

def _decimate_image_view(
    grid: np.ndarray,
    validity: np.ndarray,
    extent: tuple[float, float, float, float],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    display_pixel_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Slice and area-average one typed image without promoting its source.

    Invalid samples remain a boolean mask.  Only an actually decimated,
    display-sized result uses floating-point means; a full-resolution uint8 or
    uint16 view is handed to Matplotlib with its native dtype.
    """

    grid = np.asarray(grid)
    validity = np.asarray(validity, dtype=bool)
    if validity.shape != grid.shape:
        raise ValueError("image validity shape differs from image values")
    rows, columns = grid.shape
    x0, x1 = float(extent[0]), float(extent[1])
    y1, y0 = float(extent[2]), float(extent[3])

    def index_window(low, high, edge0, edge1, count):
        step = (edge1 - edge0) / count
        first, second = (low - edge0) / step, (high - edge0) / step
        if first > second:
            first, second = second, first
        return (
            max(0, min(int(np.floor(first)), count - 1)),
            max(1, min(int(np.ceil(second)), count)),
        )

    col0, col1 = index_window(*x_limits, x0, x1, columns)
    row0, row1 = index_window(*y_limits, y0, y1, rows)
    subset = grid[row0:row1, col0:col1]
    subset_validity = validity[row0:row1, col0:col1]
    subset_rows, subset_columns = subset.shape
    display_width, display_height = (
        max(1, int(display_pixel_shape[0])),
        max(1, int(display_pixel_shape[1])),
    )
    factor_x = max(1, subset_columns // display_width)
    factor_y = max(1, subset_rows // display_height)
    if factor_x == 1 and factor_y == 1:
        small = (
            subset
            if _all_true(subset_validity)
            else np.ma.array(
                subset,
                mask=np.logical_not(subset_validity),
                copy=False,
            )
        )
        kept_rows, kept_columns = subset_rows, subset_columns
    else:
        kept_rows = (subset_rows // factor_y) * factor_y
        kept_columns = (subset_columns // factor_x) * factor_x
        blocks = _block_view_2d(
            subset[:kept_rows, :kept_columns],
            (factor_y, factor_x),
        )
        valid_blocks = _block_view_2d(
            subset_validity[:kept_rows, :kept_columns],
            (factor_y, factor_x),
        )
        if _all_true(valid_blocks):
            small = blocks.mean(axis=(1, 3))
        else:
            counts = np.sum(valid_blocks, axis=(1, 3), dtype=np.int64)
            summed = np.sum(
                blocks,
                axis=(1, 3),
                dtype=np.float64,
                where=valid_blocks,
            )
            means = np.zeros(counts.shape, dtype=np.float64)
            np.divide(summed, counts, out=means, where=counts != 0)
            small = np.ma.array(means, mask=counts == 0, copy=False)
    x_step = (x1 - x0) / columns
    y_step = (y1 - y0) / rows
    return small, (
        x0 + col0 * x_step,
        x0 + (col0 + kept_columns) * x_step,
        y0 + (row0 + kept_rows) * y_step,
        y0 + row0 * y_step,
    )


def _block_view_2d(
    array: np.ndarray,
    block_shape: tuple[int, int],
) -> np.ndarray:
    """Group a regular 2-D view into blocks without materialising its strides."""

    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("image block input must be two-dimensional")
    block_rows, block_columns = (int(value) for value in block_shape)
    if block_rows <= 0 or block_columns <= 0:
        raise ValueError("image block shape must be positive")
    rows, columns = array.shape
    if rows % block_rows or columns % block_columns:
        raise ValueError("image dimensions must be divisible by block shape")
    row_stride, column_stride = array.strides
    return np.lib.stride_tricks.as_strided(
        array,
        shape=(
            rows // block_rows,
            block_rows,
            columns // block_columns,
            block_columns,
        ),
        strides=(
            row_stride * block_rows,
            row_stride,
            column_stride * block_columns,
            column_stride,
        ),
        writeable=False,
    )


def _all_true(value: np.ndarray) -> bool:
    """Read a compact broadcast scalar without scanning its logical plane."""

    array = np.asarray(value, dtype=bool)
    if not array.size:
        return True
    if all(stride == 0 for stride in array.strides):
        return bool(array.flat[0])
    return bool(np.all(array))


_DISTRIBUTION_SAMPLE_TARGET = 200_000


def _image_distribution_values(
    values: np.ndarray,
    validity: np.ndarray,
) -> np.ndarray:
    """Return Main's bounded, even-stride display sample.

    The side band is presentation, not an authoritative reduction.  A dense
    camera frame therefore uses the established representative sample instead
    of copying/histogramming every pixel on every live revision.  If a sparse
    validity mask leaves that stride empty, fall back to all valid pixels so a
    small ROI cannot render as a blank distribution.
    """

    values = np.asarray(values)
    validity = np.asarray(validity, dtype=bool)
    flatten_order = (
        "F"
        if values.flags.f_contiguous and not values.flags.c_contiguous
        else "C"
    )
    flat_values = np.ravel(values, order=flatten_order)
    flat_validity = np.ravel(validity, order=flatten_order)
    if flat_values.size > _DISTRIBUTION_SAMPLE_TARGET:
        step = flat_values.size // _DISTRIBUTION_SAMPLE_TARGET + 1
        sampled = flat_values[::step]
        sampled_validity = flat_validity[::step]
        if _all_true(sampled_validity):
            return sampled
        if bool(np.any(sampled_validity)):
            return sampled[sampled_validity]
    if _all_true(flat_validity):
        return flat_values
    return flat_values[flat_validity]

def _compact_engineering(value: float, length: int = 5) -> str:
    if not np.isfinite(value):
        return "nan"
    text = f"{float(value):.{max(0, length - 1)}g}"
    if "e" not in text.lower():
        return text
    mantissa, exponent = f"{float(value):.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"

def _normalized_axis_bbox(axis, width: int, height: int):
    x, y, box_width, box_height = (
        float(value) for value in axis.bbox.bounds
    )
    return (
        x / width,
        1.0 - (y + box_height) / height,
        (x + box_width) / width,
        1.0 - y / height,
    )


class _ImageAxesBlitCache:
    """Keep unchanged side bands while redrawing one complete image Axes.

    A viewport changes the image axes limits, ticks and (with equal/box) its
    physical bbox, but it does not change the distribution or colourbar for
    the same immutable source revision.  Main mutates that one persistent Axes
    in place.  The worker equivalent captures the Figure with only that Axes
    hidden, then restores the owned background and asks Matplotlib to draw the
    complete Axes -- image, ticks, labels, title and vector overlays -- for
    every subsequent viewport.  No old image pixels are transformed or
    previewed by Qt.
    """

    __slots__ = ("_axis", "_background", "_background_key")

    def __init__(self, axis) -> None:
        self._axis = axis
        self._background = None
        self._background_key = None

    def clear(self) -> None:
        self._background = None
        self._background_key = None

    def raster(self, figure, *, background_key, physical_size) -> RasterBuffer:
        try:
            if background_key != self._background_key:
                self.clear()
                self._background_key = background_key
            if self._background is None:
                visible = bool(self._axis.get_visible())
                try:
                    self._axis.set_visible(False)
                    figure.canvas.draw()
                    self._background = figure.canvas.copy_from_bbox(figure.bbox)
                finally:
                    self._axis.set_visible(visible)
            figure.canvas.restore_region(self._background)
            self._axis.draw(figure.canvas.get_renderer())
            return _raster_from_drawn_agg(
                figure,
                physical_size=physical_size,
            )
        except BaseException:
            # This is only an optimisation boundary.  A backend that cannot
            # restore/draw one Axes still returns the ordinary exact raster.
            self.clear()
            return raster_from_agg(figure, physical_size=physical_size)


class ImagePanelAggRenderer:
    """Worker-affine full-panel image renderer using main's exact visible owner."""

    __slots__ = (
        "_axis",
        "_axes_blit_cache",
        "_blit_cache",
        "_colorbar",
        "_colorbar_state",
        "_count_ceiling",
        "_distribution",
        "_distribution_artist",
        "_distribution_cache_key",
        "_distribution_cache_value",
        "_figure",
        "_fit_center_artist",
        "_fit_diagnostic_artist",
        "_fit_ring_artist",
        "_guide_lines",
        "_image_artist",
        "_layout",
        "_limit_lines",
        "_last_side_key",
        "_last_viewport_key",
        "_prepared_image_key",
        "_prepared_image_value",
        "_owner_thread",
        "_site_map",
        "_site_artist",
        "_size",
        "_size_name",
    )

    def __init__(
        self,
        *,
        width: int,
        height: int,
        dpi: float = LIVE_PANEL_DPI,
        size_name: str | None = None,
        site_map: bool = False,
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import EllipseCollection, PolyCollection
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        width = positive_integer(width, "width")
        height = positive_integer(height, "height")
        dpi = _render_dpi(dpi)
        if not isinstance(site_map, bool):
            raise TypeError("site_map must be bool")
        with render_style_context():
            figure = Figure(
                figsize=(
                    panel_figure_size_inches(size_name)
                    if size_name is not None
                    else (width / dpi, height / dpi)
                ),
                dpi=dpi,
            )
            FigureCanvasAgg(figure)
            layout = (
                image_panel_layout(size_name)
                if size_name is not None
                else image_panel_layout_for_raster(width, height)
            )
            axis = figure.add_axes(layout.image.matplotlib_bounds())
            distribution = figure.add_axes(
                layout.distribution.matplotlib_bounds()
            )
            color_axis = figure.add_axes(layout.colorbar.matplotlib_bounds())
            cmap = matplotlib.colormaps[PALETTE["cmap_camera"]].copy()
            cmap.set_bad(PALETTE["bad"])
            image_artist = axis.imshow(
                np.zeros((1, 1), dtype=np.float64),
                cmap=cmap,
                origin="upper",
                interpolation="antialiased",
                extent=(-0.5, 0.5, 0.5, -0.5),
            )
            axis.set_anchor("W")
            axis.set_aspect("equal", adjustable="box")
            from .ticks import apply_smart_ticks

            if site_map:
                apply_smart_ticks(axis)
            else:
                apply_smart_ticks(axis, max_ticks_x=4, max_ticks_y=4)
            distribution.tick_params(
                axis="x",
                which="both",
                bottom=True,
                top=False,
                labelbottom=True,
                labeltop=False,
            )
            distribution.tick_params(
                axis="y",
                which="both",
                left=True,
                right=False,
                labelleft=False,
                labelright=False,
            )
            distribution.xaxis.set_major_locator(
                MaxNLocator(nbins=1, prune="lower")
            )
            distribution.xaxis.set_major_formatter(ScalarFormatter())
            empty_vertices = np.zeros((1, 4, 2), dtype=np.float64)
            distribution_artist = PolyCollection(
                empty_vertices,
                facecolors=PALETTE["hist_fill"],
            )
            distribution.add_collection(distribution_artist)
            guide_lines = (
                distribution.axhline(
                    0.0,
                    color=PALETTE["guide"],
                    linewidth=small_fontsize() / 2.0,
                    alpha=0.3,
                ),
                distribution.axhline(
                    1.0,
                    color=PALETTE["guide"],
                    linewidth=small_fontsize() / 2.0,
                    alpha=0.3,
                ),
            )
            limit_lines = (
                distribution.axhline(
                    0.0,
                    color=cmap(0.0),
                    linewidth=small_fontsize() / 2.0,
                ),
                distribution.axhline(
                    1.0,
                    color=cmap(0.95),
                    linewidth=small_fontsize() / 2.0,
                ),
            )
            colorbar = figure.colorbar(image_artist, cax=color_axis)
            site_artist = EllipseCollection(
                widths=(1.0,),
                heights=(1.0,),
                angles=(0.0,),
                units="xy",
                offsets=np.empty((0, 2), dtype=np.float64),
                transOffset=axis.transData,
                facecolors="none",
                edgecolors="none",
                linewidths=(0.0,),
                zorder=5,
            )
            site_artist.set_visible(False)
            axis.add_collection(site_artist)
            fit_center_artist = axis.scatter(
                (),
                (),
                color=FIT_RADIAL_COLOR,
                s=FIT_RADIAL_CENTER_SIZE,
                clip_on=True,
                zorder=6,
            )
            fit_center_artist.set_visible(False)
            fit_ring_artist = Circle(
                (0.0, 0.0),
                radius=1.0,
                edgecolor=FIT_RADIAL_COLOR,
                facecolor="none",
                linewidth=FIT_RADIAL_RING_LINEWIDTH,
                alpha=FIT_RADIAL_RING_ALPHA,
                clip_on=True,
                zorder=6,
            )
            fit_ring_artist.set_visible(False)
            axis.add_patch(fit_ring_artist)
            fit_diagnostic_artist = axis.text(
                0.02,
                0.98,
                "",
                transform=axis.transAxes,
                va="top",
                color=FIT_FAILURE_COLOR,
                fontsize=ANNOTATION_FONT_SIZE,
                zorder=6,
            )
            fit_diagnostic_artist.set_visible(False)
        self._owner_thread = threading.get_ident()
        self._figure = figure
        self._fit_center_artist = fit_center_artist
        self._fit_diagnostic_artist = fit_diagnostic_artist
        self._fit_ring_artist = fit_ring_artist
        self._axis = axis
        self._axes_blit_cache = _ImageAxesBlitCache(axis)
        self._blit_cache = _AggBlitCache()
        self._distribution = distribution
        self._image_artist = image_artist
        self._layout = layout
        self._distribution_artist = distribution_artist
        self._distribution_cache_key = None
        self._distribution_cache_value = None
        self._guide_lines = guide_lines
        self._limit_lines = limit_lines
        self._last_side_key = None
        self._last_viewport_key = None
        self._prepared_image_key = None
        self._prepared_image_value = None
        self._colorbar = colorbar
        self._colorbar_state = None
        self._site_artist = site_artist
        self._site_map = site_map
        self._size = (width, height)
        self._size_name = None if size_name is None else str(size_name)
        self._count_ceiling = 0.0

    def render(
        self,
        image: EvaluatedImage,
        viewport,
        display: ImageDisplayState,
        *,
        color_limits: tuple[float, float],
        data_range: tuple[float, float] | None,
        title: str,
        value_label: str = "Signal",
        distribution_guides: bool = True,
        distribution_bins: int | None = None,
        projection_identity: EvaluatedProjectionIdentity,
        site_centers_xy: np.ndarray | None = None,
        site_radius: float | None = None,
        site_occupied: np.ndarray | None = None,
        site_validity: np.ndarray | None = None,
        colorbar_endpoints: bool | None = None,
        fit_overlay: RadialGaussianImageFitOverlay | None = None,
    ) -> tuple[RasterBuffer, ImagePanelRasterGeometry]:
        self._require_owner()
        if not isinstance(image, EvaluatedImage):
            raise TypeError("image must be EvaluatedImage")
        if not isinstance(viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        if not isinstance(display, ImageDisplayState):
            raise TypeError("display must be ImageDisplayState")
        if not isinstance(projection_identity, EvaluatedProjectionIdentity):
            raise TypeError(
                "projection_identity must be EvaluatedProjectionIdentity"
            )
        if projection_identity.data is not image:
            raise ValueError(
                "projection_identity does not name this exact EvaluatedImage"
            )
        with render_style_context():
            width, height = self._size
            extent = viewport.data_extent
            visible_extent = viewport.visible_data_extent
            display_extent = viewport.display_extent
            visible_x_limits = visible_extent[:2]
            visible_y_limits = visible_extent[2:]
            x_limits = display_extent[:2]
            y_limits = display_extent[2:]
            prepared_key = projection_identity
            if prepared_key == self._prepared_image_key:
                (
                    values,
                    finite_validity,
                ) = self._prepared_image_value
            else:
                values = np.asarray(image.values)
                validity = np.asarray(image.validity, dtype=bool)
                # Integer/bool camera samples are finite by construction.  Do
                # not allocate and scan a second megapixel boolean plane merely
                # to rediscover that fact on every viewport answer.
                finite_validity = (
                    validity
                    if values.dtype.kind in "biu"
                    else validity & np.isfinite(values)
                )
                self._prepared_image_key = prepared_key
                self._prepared_image_value = (
                    values,
                    finite_validity,
                )
            colormap_name = str(display.colormap.value)
            if self._image_artist.get_cmap().name != colormap_name:
                cmap = matplotlib.colormaps[colormap_name].copy()
                cmap.set_bad(PALETTE["bad"])
                self._image_artist.set_cmap(cmap)
            else:
                cmap = self._image_artist.get_cmap()
            if tuple(float(value) for value in self._image_artist.get_clim()) != tuple(
                float(value) for value in color_limits
            ):
                self._image_artist.set_clim(*color_limits)
            self._axis.set_xlim(*x_limits)
            self._axis.set_ylim(*y_limits)
            # ``equal`` + ``adjustable='box'`` makes the actual data box a
            # function of this viewport.  Resolve that box before decimating:
            # the worker then samples only to the pixels this exact draw can
            # show, and the published selector geometry comes from the same
            # post-draw bbox below.
            self._axis.apply_aspect()
            display_pixel_shape = (
                max(1, round(float(self._axis.bbox.width))),
                max(1, round(float(self._axis.bbox.height))),
            )
            shown, shown_extent = _decimate_image_view(
                values,
                finite_validity,
                extent,
                visible_x_limits,
                visible_y_limits,
                display_pixel_shape,
            )
            self._image_artist.set_data(shown)
            self._image_artist.set_extent(shown_extent)
            self._axis.set_xlabel(_axis_label(image.x_axis))
            self._axis.set_ylabel(_axis_label(image.y_axis))
            if title:
                apply_title(self._axis, title)
            else:
                self._axis.title.set_text("")
            self._update_site_overlay(
                site_centers_xy,
                site_radius,
                site_occupied,
                site_validity,
            )
            self._update_radial_fit_overlay(
                fit_overlay,
                viewport,
                fallback_title=title,
            )
            # Static annotations are presentation only.  Adding or moving them
            # may never participate in image autoscaling.
            self._axis.set_xlim(*x_limits)
            self._axis.set_ylim(*y_limits)

            # The side distribution is display-only.  Main's bounded
            # representative sample is part of its live-image performance and
            # visual contract; authoritative data remains untouched.
            bin_count = (
                max(8, min(max(image.values.size, 1) // 4, 50))
                if distribution_bins is None
                else positive_integer(
                    distribution_bins,
                    "distribution_bins",
                )
            )
            distribution_cache_key = (
                projection_identity,
                bin_count,
                tuple(float(value) for value in color_limits),
            )
            if distribution_cache_key == self._distribution_cache_key:
                counts, edges = self._distribution_cache_value
            else:
                finite_values = _image_distribution_values(
                    values,
                    finite_validity,
                )
                counts, edges = np.histogram(
                    (
                        finite_values
                        if finite_values.size
                        else np.asarray([color_limits[0]])
                    ),
                    bins=bin_count,
                    range=color_limits,
                )
                self._distribution_cache_key = distribution_cache_key
                self._distribution_cache_value = (counts, edges)
            vertices = np.empty((bin_count, 4, 2), dtype=np.float64)
            for index, count in enumerate(counts):
                low, high = edges[index], edges[index + 1]
                vertices[index] = (
                    (0.0, low),
                    (float(count), low),
                    (float(count), high),
                    (0.0, high),
                )
            self._distribution_artist.set_verts(vertices)
            peak = float(np.max(counts)) if counts.size else 0.0
            wanted = float(max(10, int(max(peak + 5.0, peak * 1.5))))
            if (
                self._count_ceiling <= 0.0
                or wanted > self._count_ceiling
                or wanted < 0.6 * self._count_ceiling
            ):
                self._count_ceiling = wanted
            self._distribution.set_xlim(0.0, self._count_ceiling)
            self._distribution.set_ylim(*color_limits)
            for line, value in zip(
                self._limit_lines,
                color_limits,
                strict=True,
            ):
                line.set_ydata((value, value))
            self._limit_lines[0].set_color(cmap(0.0))
            self._limit_lines[1].set_color(cmap(0.95))
            for line in self._guide_lines:
                line.set_visible(
                    bool(distribution_guides and data_range is not None)
                )
            if distribution_guides and data_range is not None:
                for line, value in zip(
                    self._guide_lines,
                    data_range,
                    strict=True,
                ):
                    line.set_ydata((value, value))

            # ``set_cmap`` / ``set_clim`` notify the attached colorbar when the
            # mapping actually changes.  Calling ``update_normal`` regardless
            # rebuilt its QuadMesh on every camera frame and defeated the
            # otherwise steady Agg path.
            # Endpoint labels are chrome, so their presence must be stable for
            # the lifetime of this renderer.  Deferring them until the second
            # frame made the first live update change the chrome key and pay
            # another complete Agg draw before the steady blit path could be
            # entered.  ``None`` is the ordinary PlotPanel policy (enabled);
            # specialised surfaces such as SiteMap opt out explicitly.
            endpoints_enabled = colorbar_endpoints is not False
            colorbar_state = (
                colormap_name,
                tuple(float(value) for value in color_limits),
                str(value_label),
                bool(endpoints_enabled),
            )
            if colorbar_state != self._colorbar_state:
                previous_colorbar_state = self._colorbar_state
                self._colorbar.set_label(str(value_label))
                if endpoints_enabled:
                    self._colorbar.set_ticks(color_limits)
                    self._colorbar.set_ticklabels(
                        [_compact_engineering(value) for value in color_limits]
                    )
                elif (
                    previous_colorbar_state is not None
                    and previous_colorbar_state[3]
                ):
                    from matplotlib.ticker import AutoLocator, ScalarFormatter

                    self._colorbar.locator = AutoLocator()
                    self._colorbar.formatter = ScalarFormatter()
                    self._colorbar.update_ticks()
                self._colorbar_state = colorbar_state
            side_key = (
                projection_identity,
                bin_count,
                str(display.colormap.value),
                tuple(float(value) for value in color_limits),
                None
                if data_range is None
                else tuple(float(value) for value in data_range),
                bool(distribution_guides),
                str(value_label),
                bool(endpoints_enabled),
                self._count_ceiling,
            )
            viewport_key = (
                tuple(float(value) for value in x_limits),
                tuple(float(value) for value in y_limits),
            )
            viewport_only = (
                side_key == self._last_side_key
                and viewport_key != self._last_viewport_key
            )
            if viewport_only:
                raster = self._axes_blit_cache.raster(
                    self._figure,
                    background_key=side_key,
                    physical_size=self._size,
                )
            else:
                if side_key != self._last_side_key:
                    self._axes_blit_cache.clear()
                raster = self._blit_cache.raster(
                    self._figure,
                    (
                        self._image_artist,
                        self._distribution_artist,
                        *self._guide_lines,
                        *self._limit_lines,
                        self._site_artist,
                        self._fit_center_artist,
                        self._fit_ring_artist,
                        self._fit_diagnostic_artist,
                    ),
                    layout_key=_agg_layout_key(
                        self._figure,
                        extra=(self._site_map,),
                    ),
                    chrome_key=_agg_chrome_key(
                        self._figure,
                        extra=(
                            self._site_map,
                            str(display.colormap.value),
                            tuple(float(value) for value in color_limits),
                            str(value_label),
                            bool(endpoints_enabled),
                            self._count_ceiling,
                        ),
                    ),
                    physical_size=self._size,
                )
            self._last_side_key = side_key
            self._last_viewport_key = viewport_key
            actual_width, actual_height = raster.width, raster.height
            # ``aspect='equal', adjustable='box'`` can shrink the live image
            # axes after limits change.  The drawn Agg bbox is therefore the
            # only mapping authority; authored layout boxes are merely inputs.
            geometry = ImagePanelRasterGeometry(
                _normalized_axis_bbox(
                    self._axis,
                    actual_width,
                    actual_height,
                ),
                _normalized_axis_bbox(
                    self._distribution,
                    actual_width,
                    actual_height,
                ),
                _normalized_axis_bbox(
                    self._colorbar.ax,
                    actual_width,
                    actual_height,
                ),
            )
            return raster, geometry

    def _update_site_overlay(
        self,
        centers_xy: np.ndarray | None,
        radius: float | None,
        occupied: np.ndarray | None,
        validity: np.ndarray | None,
    ) -> None:
        """Update Main's one hollow-ring collection, or hide it for IMAGE."""

        artist = self._site_artist
        if centers_xy is None:
            if any(value is not None for value in (radius, occupied, validity)):
                raise ValueError(
                    "site overlay values require site_centers_xy"
                )
            artist.set_offsets(np.empty((0, 2), dtype=np.float64))
            artist.set_visible(False)
            return
        centers = np.asarray(centers_xy, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1:] != (2,):
            raise ValueError("site_centers_xy must have shape (sites, 2)")
        if not len(centers) or not np.all(np.isfinite(centers)):
            raise ValueError("site_centers_xy must be nonempty and finite")
        if (
            radius is None
            or not math.isfinite(float(radius))
            or float(radius) <= 0.0
        ):
            raise ValueError("site_radius must be finite and positive")
        if validity is None:
            valid = np.ones(len(centers), dtype=np.bool_)
        else:
            valid = np.asarray(validity)
            if valid.dtype != np.dtype(bool) or valid.shape != (len(centers),):
                raise ValueError(
                    "site_validity must have bool shape (sites,)"
                )
        if occupied is None:
            states = np.zeros(len(centers), dtype=np.bool_)
        else:
            states = np.asarray(occupied)
            if states.dtype != np.dtype(bool) or states.shape != (len(centers),):
                raise ValueError(
                    "site_occupied must have bool shape (sites,)"
                )
        from matplotlib.colors import to_rgba

        empty_style = SITE_OCCUPANCY_STYLE["empty"]
        occupied_style = SITE_OCCUPANCY_STYLE["occupied"]
        invalid_color = to_rgba(SITE_INVALID_COLOR, SITE_INVALID_ALPHA)
        edgecolors = []
        linewidths = []
        linestyles = []
        for is_valid, is_occupied in zip(valid, states, strict=True):
            if not is_valid:
                edgecolors.append(invalid_color)
                linewidths.append(float(SITE_INVALID_LINEWIDTH))
                linestyles.append("dashed")
                continue
            style = occupied_style if is_occupied else empty_style
            edgecolors.append(
                to_rgba(style["color"], float(style["alpha"]))
            )
            linewidths.append(float(style["linewidth"]))
            linestyles.append("solid")
        diameter = 2.0 * float(radius)
        artist.set_offsets(centers)
        artist.set_widths(np.full(len(centers), diameter))
        artist.set_heights(np.full(len(centers), diameter))
        artist.set_angles(np.zeros(len(centers)))
        artist.set_edgecolors(edgecolors)
        artist.set_linewidths(linewidths)
        artist.set_linestyles(linestyles)
        artist.set_visible(True)

    def _update_radial_fit_overlay(
        self,
        overlay: RadialGaussianImageFitOverlay | None,
        viewport,
        *,
        fallback_title: str,
    ) -> None:
        """Update Main's radial-fit dot/ring/status on the same Agg axes."""

        center_artist = self._fit_center_artist
        ring_artist = self._fit_ring_artist
        diagnostic_artist = self._fit_diagnostic_artist
        center_artist.set_visible(False)
        ring_artist.set_visible(False)
        diagnostic_artist.set_visible(False)
        if overlay is None:
            if fallback_title:
                apply_title(self._axis, fallback_title)
            else:
                self._axis.title.set_text("")
            return
        if not isinstance(overlay, RadialGaussianImageFitOverlay):
            raise TypeError(
                "fit_overlay must be RadialGaussianImageFitOverlay or None"
            )
        if overlay.coordinate_frame != viewport.coordinate_frame:
            raise ValueError("fit overlay belongs to another coordinate frame")
        status_label = (
            "NOT_PRESENT"
            if overlay.status is None
            else overlay.status.value
        )
        apply_title(
            self._axis,
            f"{overlay.caption} $\\cdot$ {status_label}",
        )
        if overlay.status is FitBatchStatus.CONVERGED:
            center = overlay.center_xy
            radius = overlay.one_over_e_radius
            if center is None or radius is None:
                raise RuntimeError(
                    "converged radial fit overlay lost its geometry"
                )
            center_artist.set_offsets(np.asarray((center,), dtype=np.float64))
            center_artist.set_visible(True)
            ring_artist.set_center(center)
            ring_artist.set_radius(radius)
            ring_artist.set_visible(True)
            # Vector annotation can be off-screen but may never autoscale IMAGE.
            return
        diagnostic_artist.set_text(
            f"fit {overlay.diagnostic or status_label}"
        )
        diagnostic_artist.set_visible(True)

    def close(self) -> None:
        self._require_owner()
        figure, self._figure = self._figure, None
        if figure is None:
            return
        self._axes_blit_cache.clear()
        self._blit_cache.clear()
        self._distribution_cache_key = None
        self._distribution_cache_value = None
        self._prepared_image_key = None
        self._prepared_image_value = None
        release_agg_figure(figure)
        gc.collect()

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("image-panel Agg renderer used from another thread")
        if self._figure is None:
            raise RuntimeError("image-panel Agg renderer is closed")

__all__ = [
    "encode_radial_gaussian_image_fit_panels",
    "render_radial_gaussian_image_fit_panels",
    "ImagePanelAggRenderer",
]
