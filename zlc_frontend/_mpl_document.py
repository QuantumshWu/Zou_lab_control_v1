"""Internal Matplotlib implementation owner: document."""

from __future__ import annotations

import gc
from io import BytesIO
import math
import threading
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
from .curve_display import (
    CurveDisplayState,
    curve_home_x_limits,
    numeric_curve_coordinates,
)
from .fit_projection import (
    evaluated_figure_panels as _panels,
    figure_panel_title as _panel_title,
    panel_focus_selection as _panel_focus_selection,
    fit_batch_storage_index as _batch_storage_index,
)
from .fit_image_projection import radial_gaussian_fit_geometry
from .image_display import (
    ImageDisplayState,
    evaluated_image_data_range,
    image_viewport_for_display_state,
    resolve_image_color_limits_from_range,
)
from .image_view import image_viewport_for_evaluated_image
from .meter_display import MeterDisplayState
from .histogram_display import (
    FacetedHistogramDisplayState,
    HistogramBinProjection,
    HistogramDisplayState,
    _windowed_histogram_projection,
    histogram_count_limits,
    histogram_home_x_limits,
)
from .data_figure import FigurePanelRegion
from .axis_display import axis_label as _axis_label
from .render import RasterBuffer
from .display_range import (
    RelimMode,
    deadband_display_range,
)
from .plot_layout import (
    panel_figure_size_inches,
)
from .render_style import (
    ANNOTATION_FONT_SIZE,
    FIT_FAILURE_COLOR,
    FIT_RADIAL_CENTER_SIZE,
    FIT_RADIAL_COLOR,
    FIT_RADIAL_RING_ALPHA,
    FIT_RADIAL_RING_LINEWIDTH,
    FIT_LINESTYLE,
    HIST_FILL_ALPHA,
    apply_title,
    axis_label_fontsize,
    render_style_context,
    tick_fontsize,
)

from ._mpl_common import (
    _AggBlitCache,
    _agg_chrome_key,
    _agg_layout_key,
    _display_series_label,
    _render_dpi,
    _require_evaluated_identity,
    release_agg_figure,
)
from ._mpl_curve import (
    _curve,
    _curve_values,
    _meter,
    _meter_text,
    _shared_curve_limits,
)
from ._mpl_histogram import (
    _draw_histogram_projection,
    _grid_histogram_value_range,
    _histogram,
    _update_histogram_artist,
    _update_histogram_presentation,
)
from ._mpl_image import (
    _draw_projected_image,
    _image_axis,
    _image,
    _radial_image_color_limits_by_layer,
)
from ._mpl_grid import (
    _figure_panel_regions,
    _live_grid_axes,
    _live_grid_cell_title,
)


def _faceted_panel_topology(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
) -> tuple[object, ...]:
    """Return only the artist/layout facts of one faceted live document.

    Dataset revisions and evaluated numeric values deliberately do not enter
    this identity.  They are the exact payload painted *onto* an Agg surface,
    not a reason to replace that surface.
    """

    _require_evaluated_identity(document, evaluated)

    def axis_topology(axis: EvaluatedAxis) -> tuple[object, ...]:
        return (
            axis.axis_id,
            axis.role,
            axis.unit,
            axis.coordinate_frame,
            len(axis.indices),
        )

    def data_topology(data) -> tuple[object, ...]:
        if isinstance(data, EvaluatedCurve):
            return (
                EvaluatedCurve,
                axis_topology(data.x_axis),
                data.value_unit,
            )
        if isinstance(data, EvaluatedImage):
            return (
                EvaluatedImage,
                axis_topology(data.x_axis),
                axis_topology(data.y_axis),
                data.value_unit,
            )
        if isinstance(data, EvaluatedHistogram):
            return (
                EvaluatedHistogram,
                tuple(item.axis_id for item in data.sample_coordinates),
                data.value_unit,
            )
        if isinstance(data, EvaluatedMeter):
            return (EvaluatedMeter, data.value_unit)
        raise TypeError(f"unsupported evaluated data {type(data).__name__}")

    panels = _panels(evaluated)
    return (
        tuple(document.datasets),
        tuple(document.layers),
        tuple(
            (
                layer.layer_id,
                layer.dataset_id,
                tuple((item.axis_id, item.index) for item in cell.facet_address),
                tuple(
                    (
                        tuple(
                            (item.axis_id, item.index)
                            for item in series.batch_address
                        ),
                        data_topology(series.data),
                    )
                    for series in series_group
                ),
            )
            for layer, cell, series_group in panels
        ),
    )


class _FacetedCellArtists:
    """The fixed artist slots of one cell in a persistent live grid."""

    __slots__ = (
        "diagnostics",
        "fit",
        "histogram_fit",
        "histogram_thresholds",
        "image_center",
        "image_diagnostic",
        "image_ring",
        "kind",
        "source",
    )

    def __init__(self, kind: type) -> None:
        self.kind = kind
        self.source = ()
        self.fit = ()
        self.diagnostics = ()
        self.histogram_fit = ()
        self.histogram_thresholds = []
        self.image_center = None
        self.image_ring = None
        self.image_diagnostic = None

    def dynamic_artists(self) -> tuple[object, ...]:
        return (
            *self.source,
            *self.fit,
            *self.diagnostics,
            *self.histogram_fit,
            *self.histogram_thresholds,
            self.image_center,
            self.image_ring,
            self.image_diagnostic,
        )


class FacetedPanelAggRenderer:
    """Worker-affine persistent artist owner for one live faceted panel.

    Schema/view/facet layout constructs the Figure, Axes, and fixed artist
    slots exactly once.  Source and display revisions only reconcile values,
    limits, labels, and variable overlays into those slots.  Exact source refs
    remain in ``EvaluatedFigureData`` and the caller's coherence stamp; they
    never participate in renderer ownership.
    """

    __slots__ = (
        "_axes",
        "_blit_cache",
        "_cells",
        "_closed",
        "_columns",
        "_curve_relim_mode",
        "_curve_y_limits",
        "_dpi",
        "_figure",
        "_font_scale",
        "_height",
        "_histogram_count_limits",
        "_histogram_count_scale",
        "_histogram_relim_mode",
        "_image_color_limits",
        "_image_relim_mode",
        "_image_range_evaluated",
        "_image_range_value",
        "_kind",
        "_owner_thread",
        "_size_name",
        "_title",
        "_topology",
        "_value_label",
        "_width",
    )

    def __init__(
        self,
        *,
        size_name: str,
        width: int,
        height: int,
        dpi: float,
        title: str,
        value_label: str,
    ) -> None:
        self._size_name = str(size_name)
        self._width = positive_integer(width, "width")
        self._height = positive_integer(height, "height")
        self._dpi = _render_dpi(dpi)
        self._title = str(title)
        self._value_label = str(value_label)
        self._owner_thread = threading.get_ident()
        self._closed = False
        self._figure = None
        self._font_scale = 1.0
        self._axes = ()
        self._cells = ()
        self._columns = 0
        self._curve_relim_mode = None
        self._curve_y_limits = None
        self._kind = None
        self._topology = None
        self._blit_cache = _AggBlitCache()
        self._histogram_count_limits = None
        self._histogram_relim_mode = None
        self._histogram_count_scale = None
        self._image_color_limits = None
        self._image_relim_mode = None
        self._image_range_evaluated = None
        self._image_range_value = None

    def render(
        self,
        document: FigureDocument,
        evaluated: EvaluatedFigureData,
        fit_results: dict[str, FitResultBatch],
        *,
        display_state: object,
    ) -> tuple[RasterBuffer, tuple[FigurePanelRegion, ...]]:
        self._require_owner()
        if self._closed:
            raise RuntimeError("faceted Agg renderer is closed")
        topology = _faceted_panel_topology(document, evaluated)
        if self._topology is not None and topology != self._topology:
            raise RuntimeError("faceted panel topology changed between revisions")
        with render_style_context():
            try:
                if self._figure is None:
                    self._build(document, evaluated)
                    self._topology = topology
                self._update(document, evaluated, fit_results, display_state)
                figure = self._figure
                assert figure is not None
                dynamic = tuple(
                    artist
                    for cell in self._cells
                    for artist in cell.dynamic_artists()
                    if artist is not None
                )
                raster = self._blit_cache.raster(
                    figure,
                    dynamic,
                    layout_key=_agg_layout_key(
                        figure,
                        extra=(self._topology, self._size_name),
                    ),
                    chrome_key=_agg_chrome_key(
                        figure,
                        extra=(self._topology, self._title, self._value_label),
                    ),
                    physical_size=(self._width, self._height),
                )
                regions = _figure_panel_regions(
                    figure,
                    evaluated,
                    fit_results,
                )
                return raster, regions
            except BaseException:
                self.close()
                raise

    def _build(
        self,
        document: FigureDocument,
        evaluated: EvaluatedFigureData,
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        _require_evaluated_identity(document, evaluated)
        panels = _panels(evaluated)
        if len(panels) <= 1:
            raise ValueError("faceted renderer requires multiple panels")
        kinds = {
            type(series.data)
            for _layer, _cell, series_group in panels
            for series in series_group
        }
        if len(kinds) != 1:
            raise ValueError("faceted renderer requires homogeneous panel data")
        kind = next(iter(kinds))
        if kind not in (
            EvaluatedCurve,
            EvaluatedHistogram,
            EvaluatedImage,
            EvaluatedMeter,
        ):
            raise ValueError(
                "faceted renderer requires CURVE, HISTOGRAM, IMAGE, or METER"
            )
        image_aspect = None
        if kind is EvaluatedImage:
            first = panels[0][2][0].data
            image_aspect = len(first.x_axis.indices) / max(
                len(first.y_axis.indices),
                1,
            )
        figure = Figure(
            figsize=panel_figure_size_inches(self._size_name),
            dpi=self._dpi,
            constrained_layout=False,
        )
        FigureCanvasAgg(figure)
        axes, _rows, columns, font_scale = _live_grid_axes(
            figure,
            size=self._size_name,
            cell_count=len(panels),
            cell_aspect=image_aspect,
        )
        for unused in axes[len(panels):]:
            unused.set_visible(False)
        self._figure = figure
        self._axes = axes[: len(panels)]
        self._columns = columns
        self._font_scale = font_scale
        self._kind = kind
        self._cells = tuple(_FacetedCellArtists(kind) for _ in panels)
        apply_title(figure, self._title)

    def _update(
        self,
        document: FigureDocument,
        evaluated: EvaluatedFigureData,
        fit_results: dict[str, FitResultBatch],
        display_state: object,
    ) -> None:
        _require_evaluated_identity(document, evaluated)
        panels = _panels(evaluated)
        if len(panels) != len(self._cells):
            raise RuntimeError("faceted cell count changed between revisions")
        if self._kind is EvaluatedCurve:
            self._update_curves(panels, fit_results, display_state)
        elif self._kind is EvaluatedHistogram:
            self._update_histograms(panels, fit_results, display_state)
        elif self._kind is EvaluatedImage:
            self._update_images(evaluated, panels, fit_results, display_state)
        elif self._kind is EvaluatedMeter:
            self._update_meters(panels, fit_results, display_state)
        else:  # pragma: no cover - _build closes the set
            raise RuntimeError("faceted renderer lost its data kind")

        first_xlabel = first_ylabel = ""
        for index, (axis, state, (_layer, cell, series_group)) in enumerate(
            zip(self._axes, self._cells, panels, strict=True)
        ):
            if index == 0:
                first_xlabel = axis.get_xlabel()
                first_ylabel = axis.get_ylabel()
            self._apply_cell_chrome(axis, index, cell, series_group)
        self._set_outer_label("x", first_xlabel)
        self._set_outer_label("y", first_ylabel)

    def _update_meters(self, panels, fit_results, display_state) -> None:
        if not isinstance(display_state, MeterDisplayState):
            raise TypeError("live meter grid requires MeterDisplayState")
        if fit_results:
            raise ValueError("METER display cannot carry a Fit overlay")

        # Validate the complete new front before touching the persistent artist
        # tree.  A bad value in one cell must not leave the preceding cells at
        # the new revision while the rest still show the old revision.
        texts = []
        for _layer, _cell, series_group in panels:
            if any(
                not isinstance(series.data, EvaluatedMeter)
                for series in series_group
            ):
                raise ValueError("meter grid changed data kind")
            texts.append(_meter_text(series_group))

        for axis, state, panel, text_value in zip(
            self._axes,
            self._cells,
            panels,
            texts,
            strict=True,
        ):
            if not state.source:
                before = len(axis.texts)
                _meter(axis, panel[2])
                state.source = tuple(axis.texts[before:])
            if len(state.source) != 1:
                raise RuntimeError("meter grid artist topology changed")
            state.source[0].set_text(text_value)

    def _update_curves(self, panels, fit_results, display_state) -> None:
        if not isinstance(display_state, CurveDisplayState):
            raise TypeError("live curve grid requires CurveDisplayState")
        curves = tuple(
            series.data
            for _layer, _cell, series_group in panels
            for series in series_group
        )
        first_curve = curves[0]
        numeric_curve_coordinates(first_curve.x_axis)
        if any(curve.x_axis != first_curve.x_axis for curve in curves[1:]):
            raise ValueError("live curve grid requires one exact shared x axis")
        finite_groups = []
        for curve in curves:
            values = np.asarray(curve.values)
            if np.iscomplexobj(values):
                raise ValueError(
                    "complex curves require an explicit real-valued display transform"
                )
            valid = np.asarray(curve.validity, dtype=bool)
            if np.any(valid):
                if not bool(np.all(np.isfinite(values[valid]))):
                    raise ValueError("valid curve values must all be finite")
                finite_groups.append(np.asarray(values[valid], dtype=np.float64))
        if finite_groups:
            data_min = min(float(np.min(group)) for group in finite_groups)
            data_max = max(float(np.max(group)) for group in finite_groups)
            y_limits = deadband_display_range(
                display_state.relim_mode,
                self._curve_y_limits,
                data_min,
                data_max,
                fixed_range=display_state.fixed_y_limits,
                force=(
                    self._curve_relim_mode is None
                    or self._curve_relim_mode is not display_state.relim_mode
                ),
            )
        elif display_state.relim_mode is RelimMode.FIXED:
            assert display_state.fixed_y_limits is not None
            y_limits = display_state.fixed_y_limits
        elif self._curve_y_limits is not None:
            y_limits = self._curve_y_limits
        else:
            y_limits = (0.0, 1.0)
        self._curve_y_limits = y_limits
        self._curve_relim_mode = display_state.relim_mode
        x_limits = display_state.x_view or curve_home_x_limits(first_curve.x_axis)

        for axis, state, (layer, cell, series_group) in zip(
            self._axes,
            self._cells,
            panels,
            strict=True,
        ):
            if any(not isinstance(series.data, EvaluatedCurve) for series in series_group):
                raise ValueError("curve grid changed data kind")
            if not state.source:
                _curve(axis, layer, cell, series_group, None)
                state.source = tuple(axis.lines)
                fit_artists = []
                diagnostics = []
                for index, source_artist in enumerate(state.source):
                    fit_artist, = axis.plot(
                        (),
                        (),
                        color=source_artist.get_color(),
                        linestyle=FIT_LINESTYLE,
                        marker=None,
                        label="_nolegend_",
                    )
                    fit_artists.append(fit_artist)
                    diagnostics.append(
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
                state.fit = tuple(fit_artists)
                state.diagnostics = tuple(diagnostics)
            if (
                len(state.source) != len(series_group)
                or len(state.fit) != len(series_group)
                or len(state.diagnostics) != len(series_group)
            ):
                raise RuntimeError("curve grid artist topology changed")
            fit_result = fit_results.get(layer.layer_id)
            multiple = len(series_group) > 1
            for index, (source_artist, fit_artist, diagnostic, series) in enumerate(
                zip(
                    state.source,
                    state.fit,
                    state.diagnostics,
                    series_group,
                    strict=True,
                )
            ):
                data = series.data
                source_artist.set_data(
                    np.asarray(data.x_axis.coordinates),
                    _curve_values(series),
                )
                label = _display_series_label(
                    layer.layer_id,
                    series,
                    index,
                    multiple_series=multiple,
                )
                source_artist.set_label(label)
                fit_artist.set_data((), ())
                fit_artist.set_visible(False)
                fit_artist.set_label("_nolegend_")
                diagnostic.set_text("")
                if fit_result is None:
                    continue
                storage = _batch_storage_index(fit_result, layer, cell, series)
                if storage is None:
                    diagnostic.set_text("fit NOT_PRESENT")
                    continue
                status = fit_result.statuses[storage]
                if status is not FitBatchStatus.CONVERGED:
                    message = status.value
                    if fit_result.errors[storage]:
                        message = f"{message}: {fit_result.errors[storage]}"
                    diagnostic.set_text(f"fit {message}")
                    continue
                coordinates = np.asarray(data.x_axis.coordinates, dtype=np.float64)
                fit_artist.set_data(
                    coordinates,
                    fit_result.evaluate_batch(storage, (coordinates,)),
                )
                fit_artist.set_visible(True)
                fit_artist.set_label(f"fit {label}")
            axis.set_xlabel(_axis_label(series_group[0].data.x_axis))
            axis.set_ylabel(self._value_label)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)

    def _histogram_projection(self, panels, display_state):
        if isinstance(display_state, FacetedHistogramDisplayState):
            faceted = display_state
            display = display_state.display
        elif isinstance(display_state, HistogramDisplayState) and not display_state.thresholds:
            display = display_state
            faceted = FacetedHistogramDisplayState(display)
        else:
            raise TypeError("live histogram grid requires per-cell display state")
        samples = tuple(
            series.data.samples
            for _layer, _cell, series_group in panels
            for series in series_group
        )
        low, high = _grid_histogram_value_range(panels)
        counts, edges = _windowed_histogram_projection(
            samples,
            display.bin_count,
            visible_range=(low, high),
        )
        x_limits = display.x_view or histogram_home_x_limits(edges)
        peak = max((int(np.max(item, initial=0)) for item in counts), default=0)
        count_limits = histogram_count_limits(
            display,
            peak,
            current_count_limits=self._histogram_count_limits,
            previous_relim_mode=self._histogram_relim_mode,
            previous_count_scale=self._histogram_count_scale,
        )
        self._histogram_count_limits = count_limits
        self._histogram_relim_mode = display.relim_mode
        self._histogram_count_scale = display.count_scale
        return faceted, display, counts, edges, x_limits, count_limits

    def _update_histograms(self, panels, fit_results, display_state) -> None:
        if fit_results:
            raise ValueError("fit overlays require a curve or image view")
        faceted, display, counts, edges, x_limits, count_limits = (
            self._histogram_projection(panels, display_state)
        )
        offset = 0
        for axis, state, (layer, cell, series_group) in zip(
            self._axes,
            self._cells,
            panels,
            strict=True,
        ):
            if any(not isinstance(series.data, EvaluatedHistogram) for series in series_group):
                raise ValueError("histogram grid changed data kind")
            next_offset = offset + len(series_group)
            panel_counts = counts[offset:next_offset]
            offset = next_offset
            if not state.source:
                state.source = _draw_histogram_projection(
                    axis,
                    series_group,
                    panel_counts,
                    edges,
                    fill_alpha=1.0,
                )
            if len(state.source) != len(panel_counts):
                raise RuntimeError("histogram grid artist topology changed")
            for artist, current in zip(state.source, panel_counts, strict=True):
                _update_histogram_artist(artist, current, edges)
            selection = _panel_focus_selection(layer, cell, series_group)
            if selection is None:
                raise RuntimeError("live histogram grid cell has no selection")
            cell_display = faceted.display_for(selection)
            (
                state.histogram_fit,
                thresholds,
                _unused_stats,
                _effective_thresholds,
            ) = _update_histogram_presentation(
                axis,
                cell_display,
                panel_counts,
                edges,
                fit_artists=state.histogram_fit,
                threshold_artists=state.histogram_thresholds,
                stats_text=None,
                show_stats=False,
                threshold_linewidth=1.4,
            )
            state.histogram_thresholds = list(thresholds)
            axis.set_yscale(display.count_scale.value)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*count_limits)
            axis.set_xlabel(self._value_label)
            axis.set_ylabel("Shots")

    def _resolved_image_color_limits(self, evaluated, panels, display):
        if not isinstance(display, ImageDisplayState):
            raise TypeError("live image grid requires ImageDisplayState")
        if self._image_range_evaluated is not evaluated:
            images = tuple(
                series.data
                for _layer, _cell, series_group in panels
                for series in series_group
            )
            self._image_range_evaluated = evaluated
            self._image_range_value = evaluated_image_data_range(images)
        _data_range, color_limits = resolve_image_color_limits_from_range(
            self._image_range_value,
            display,
            current_color_limits=self._image_color_limits,
            previous_relim_mode=self._image_relim_mode,
        )
        self._image_color_limits = color_limits
        self._image_relim_mode = display.relim_mode
        return color_limits

    def _update_images(self, evaluated, panels, fit_results, display) -> None:
        from matplotlib.patches import Circle

        color_limits = self._resolved_image_color_limits(
            evaluated,
            panels,
            display,
        )
        for axis, state, (layer, cell, series_group) in zip(
            self._axes,
            self._cells,
            panels,
            strict=True,
        ):
            if len(series_group) != 1 or not isinstance(
                series_group[0].data,
                EvaluatedImage,
            ):
                raise ValueError("image grid requires one image per cell")
            series = series_group[0]
            data = series.data
            viewport = image_viewport_for_display_state(
                display,
                image_viewport_for_evaluated_image(data),
            )
            if not state.source:
                _draw_projected_image(
                    axis,
                    self._figure,
                    data,
                    colormap=display.colormap.value,
                    color_limits=color_limits,
                    visible_bounds=viewport.visible_bounds,
                    regular_axis_contract=True,
                    center=None,
                    radius=None,
                    diagnostic=None,
                    show_colorbar=False,
                )
                state.source = (axis.images[0],)
                state.image_center = axis.scatter(
                    (), (), color=FIT_RADIAL_COLOR,
                    s=FIT_RADIAL_CENTER_SIZE, clip_on=True
                )
                state.image_center.set_visible(False)
                state.image_ring = Circle(
                    (0.0, 0.0),
                    radius=1.0,
                    edgecolor=FIT_RADIAL_COLOR,
                    facecolor="none",
                    linewidth=FIT_RADIAL_RING_LINEWIDTH,
                    alpha=FIT_RADIAL_RING_ALPHA,
                    clip_on=True,
                )
                state.image_ring.set_visible(False)
                axis.add_patch(state.image_ring)
                state.image_diagnostic = axis.text(
                    0.02,
                    0.98,
                    "",
                    transform=axis.transAxes,
                    va="top",
                    color=FIT_FAILURE_COLOR,
                    fontsize=ANNOTATION_FONT_SIZE,
                )
                state.image_diagnostic.set_visible(False)
            artist = state.source[0]
            x_edges, _x_centers, x_labels = _image_axis(data.x_axis)
            y_edges, _y_centers, y_labels = _image_axis(data.y_axis)
            if x_labels is not None or y_labels is not None:
                raise ValueError("live image grid requires regular numeric axes")
            invalid = np.logical_not(data.validity)
            if data.values.dtype.kind == "f":
                invalid = np.logical_or(invalid, ~np.isfinite(data.values))
            artist.set_data(np.ma.array(data.values, mask=invalid))
            artist.set_extent((x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]))
            artist.set_cmap(display.colormap.value)
            artist.set_clim(*color_limits)
            left, top, right, bottom = viewport.visible_bounds
            x_start, x_stop = float(x_edges[0]), float(x_edges[-1])
            y_start, y_stop = float(y_edges[0]), float(y_edges[-1])
            x_limits = (
                x_start + left * (x_stop - x_start),
                x_start + right * (x_stop - x_start),
            )
            y_limits = (
                y_start + bottom * (y_stop - y_start),
                y_start + top * (y_stop - y_start),
            )
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_anchor("W")
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(_axis_label(data.x_axis))
            axis.set_ylabel(_axis_label(data.y_axis))
            state.image_center.set_offsets(np.empty((0, 2), dtype=np.float64))
            state.image_center.set_visible(False)
            state.image_ring.set_visible(False)
            state.image_diagnostic.set_text("")
            state.image_diagnostic.set_visible(False)
            fit_result = fit_results.get(layer.layer_id)
            if fit_result is not None:
                if fit_result.spec.model_id != "radial_gaussian_center":
                    raise ValueError(
                        "live image grid accepts only radial image fit overlays"
                    )
                storage = _batch_storage_index(fit_result, layer, cell, series)
                if storage is None:
                    diagnostic = "NOT_PRESENT"
                elif fit_result.statuses[storage] is FitBatchStatus.CONVERGED:
                    center, radius = radial_gaussian_fit_geometry(
                        fit_result,
                        storage,
                    )
                    state.image_center.set_offsets(np.asarray((center,)))
                    state.image_center.set_visible(True)
                    state.image_ring.center = center
                    state.image_ring.set_radius(radius)
                    state.image_ring.set_visible(True)
                    diagnostic = None
                else:
                    diagnostic = fit_result.statuses[storage].value
                    if fit_result.errors[storage]:
                        diagnostic = f"{diagnostic}: {fit_result.errors[storage]}"
                if diagnostic is not None:
                    state.image_diagnostic.set_text(f"fit {diagnostic}")
                    state.image_diagnostic.set_visible(True)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)

    def _apply_cell_chrome(self, axis, index, cell, series_group) -> None:
        from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullLocator

        apply_title(
            axis,
            _live_grid_cell_title(cell, series_group),
            size=tick_fontsize() * self._font_scale,
            pad=1.5,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(
            axis="both",
            labelsize=tick_fontsize() * self._font_scale,
            length=2,
        )
        if index % self._columns:
            axis.set_yticks([])
            if axis.get_yscale() != "linear":
                axis.yaxis.set_minor_locator(NullLocator())
        elif axis.get_yscale() == "linear":
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.tick_params(axis="y", labelleft=True)
        else:
            axis.yaxis.set_major_locator(LogLocator(numticks=3))
            axis.yaxis.set_minor_locator(NullLocator())
            axis.tick_params(axis="y", labelleft=True)
        if index + self._columns < len(self._cells):
            axis.set_xticks([])
        else:
            low, high = axis.get_xlim()
            interior = tuple(
                value
                for value in MaxNLocator(nbins=4).tick_values(low, high)
                if low < value < high
            )
            picked = (
                (min(interior, key=lambda value: abs(value - (low + high) / 2.0)),)
                if interior
                else ()
            )
            axis.xaxis.set_major_locator(FixedLocator(picked))
            axis.tick_params(axis="x", labelbottom=True)

    def _set_outer_label(self, axis_name: str, value: str) -> None:
        figure = self._figure
        assert figure is not None
        gid = f"zlc-faceted-{axis_name}-label"
        existing = next(
            (artist for artist in figure.texts if artist.get_gid() == gid),
            None,
        )
        if existing is not None:
            existing.set_text(value)
            return
        if not value:
            return
        if axis_name == "x":
            artist = figure.text(
                0.5,
                0.012,
                value,
                ha="center",
                va="bottom",
                fontsize=axis_label_fontsize(),
            )
        else:
            artist = figure.text(
                0.008,
                0.5,
                value,
                ha="left",
                va="center",
                rotation="vertical",
                fontsize=axis_label_fontsize(),
            )
        artist.set_gid(gid)

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("faceted Agg renderer used from another thread")

    def close(self) -> None:
        self._require_owner()
        if self._closed:
            return
        self._closed = True
        figure, self._figure = self._figure, None
        self._axes = ()
        self._cells = ()
        self._topology = None
        self._curve_y_limits = None
        self._curve_relim_mode = None
        self._histogram_count_limits = None
        self._histogram_relim_mode = None
        self._histogram_count_scale = None
        self._image_color_limits = None
        self._image_relim_mode = None
        self._image_range_evaluated = None
        self._image_range_value = None
        self._blit_cache.clear()
        if figure is not None:
            release_agg_figure(figure)
            figure = None
            gc.collect()

def render_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    dpi: float = 100.0,
):
    """Construct a caller-owned Figure with canonical artist styles frozen in.

    Later caller-driven draw/save operations are outside the product compose lane.  Product
    encoding must use :func:`encode_evaluated_figure`, which keeps construction and Agg draw
    under the serialized style context.
    """

    dpi = _render_dpi(dpi)
    with render_style_context():
        return _render_evaluated_figure(document, evaluated, fit_results, dpi=dpi)

def encode_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    image_format: str,
    dpi: float,
) -> bytes:
    """Encode one product figure into immutable bytes on the compose lane."""

    payload, _regions = _encode_evaluated_figure(
        document,
        evaluated,
        fit_results,
        image_format=image_format,
        dpi=dpi,
        include_panel_regions=False,
    )
    return payload

def encode_evaluated_figure_with_panel_regions(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    dpi: float,
) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
    """Encode a product PNG and the final Agg panel rectangles from one draw."""

    return _encode_evaluated_figure(
        document,
        evaluated,
        fit_results,
        image_format="png",
        dpi=dpi,
        include_panel_regions=True,
    )

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
    renderer = FacetedPanelAggRenderer(
        size_name=str(size),
        width=width,
        height=height,
        dpi=dpi,
        title=str(title),
        value_label=str(value_label),
    )
    try:
        raster, regions = renderer.render(
            document,
            evaluated,
            fit_results,
            display_state=display_state,
        )
        from .encoded_raster import encode_raster_buffer_png

        return encode_raster_buffer_png(raster), regions
    finally:
        renderer.close()

def _encode_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    image_format: str,
    dpi: float,
    include_panel_regions: bool,
) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
    dpi = _render_dpi(dpi)
    regions: tuple[FigurePanelRegion, ...] = ()
    output = BytesIO()
    with render_style_context():
        figure = None
        try:
            figure = _render_evaluated_figure(
                document,
                evaluated,
                fit_results,
                dpi=dpi,
            )
            figure.savefig(output, format=image_format, dpi=dpi)
            if include_panel_regions:
                regions = _figure_panel_regions(figure, evaluated, fit_results)
        finally:
            if figure is not None:
                release_agg_figure(figure)
            figure = None
            # The caller's last strong local must be gone before collecting the
            # remaining Matplotlib artist-parent cycles.
            gc.collect()
    return output.getvalue(), regions

def _render_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
    *,
    dpi: float,
):
    """Render the generic one-shot figure used by archive/export surfaces.

    Live faceted panels are owned exclusively by
    :class:`FacetedPanelAggRenderer`; keeping their geometry and display-state
    branches here would create a second presentation implementation.
    """

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _require_evaluated_identity(document, evaluated)
    panels = _panels(evaluated)
    shared_histogram_projection = None
    shared_histogram_counts = None
    shared_histogram_edges = None
    shared_histogram_x_limits = None
    shared_histogram_count_limits = None
    histogram_display = None
    if (
        len({layer.layer_id for layer, _cell, _series in panels}) == 1
        and all(
            isinstance(series.data, EvaluatedHistogram)
            for _layer, _cell, series_group in panels
            for series in series_group
        )
    ):
        histogram_display = HistogramDisplayState()
        histogram_samples = tuple(
            series.data.samples
            for _layer, _cell, series_group in panels
            for series in series_group
        )
        shared_histogram_projection = HistogramBinProjection(
            histogram_samples,
            bins=histogram_display.bin_count,
        )
        shared_histogram_counts = shared_histogram_projection.bin_counts
        shared_histogram_edges = shared_histogram_projection.bin_edges
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
        shared_histogram_count_limits = histogram_count_limits(
            histogram_display,
            peak_count,
        )
    shared_curve_limits = _shared_curve_limits(
        panels,
        fit_results,
        live_grid=False,
    )
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
                        fill_alpha=HIST_FILL_ALPHA,
                    )
                    histogram_series_offset = next_offset
                    assert shared_histogram_x_limits is not None
                    assert shared_histogram_count_limits is not None
                    assert histogram_display is not None
                    target.set_yscale(histogram_display.count_scale.value)
                    target.set_xlim(*shared_histogram_x_limits)
                    target.set_ylim(*shared_histogram_count_limits)
            elif isinstance(kind, EvaluatedMeter):
                if fit_result is not None:
                    raise ValueError("fit overlays require a curve or image view")
                _meter(target, series_group)
            else:  # pragma: no cover - closed EvaluatedLayerData union
                raise TypeError(f"unsupported evaluated data {type(kind).__name__}")
            target.set_title(
                _panel_title(document, layer, cell, series_group),
            )
        for unused in axes[len(panels):]:
            unused.set_visible(False)
        return figure
    except BaseException:
        release_agg_figure(figure)
        figure = axes = None
        gc.collect()
        raise

__all__ = [
    "encode_evaluated_figure",
    "render_evaluated_figure",
    "encode_evaluated_figure_with_panel_regions",
    "encode_evaluated_panel_with_regions",
    "FacetedPanelAggRenderer",
]
