"""Internal Matplotlib implementation owner: live."""

from __future__ import annotations

import gc
import threading
import numpy as np
from zlc_data import FitBatchStatus, FitResultBatch
from zlc_storage import nonnegative_integer, positive_integer
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
from .curve_display import (
    CurveDisplayState,
    CurveViewportTransform,
    NumericDisplayAxis,
    NumericViewportTransform,
    curve_home_x_limits,
    numeric_curve_coordinates,
)
from .histogram_display import (
    DEFAULT_HISTOGRAM_BINS,
    FacetedHistogramDisplayState,
    HistogramBinProjection,
    HistogramCountScale,
    HistogramDisplayState,
    HistogramFitMode,
    HistogramViewportTransform,
    histogram_count_limits,
    histogram_home_x_limits,
)
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
    ImagePanelRasterGeometry,
    MeterPanelPayload,
    PulsePanelPayload,
    RadialGaussianImageFitOverlay,
    RasterBuffer,
    _validated_curve_fit_overlays,
)
from .axis_display import axis_label as _axis_label
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
)
from ._mpl_histogram import (
    _histogram,
    _histogram_projection,
    _update_histogram_artist,
    _update_histogram_presentation,
)

class SinglePanelAggRenderer:
    """Worker-affine live Agg surface for one curve/histogram/meter panel."""

    __slots__ = (
        "_axis",
        "_blit_cache",
        "_document",
        "_figure",
        "_artists",
        "_fit_artists",
        "_fit_diagnostic_artists",
        "_distribution_artist",
        "_gauss_artist",
        "_gauss_text",
        "_hist_fit_artists",
        "_hist_stats_text",
        "_hist_threshold_artists",
        "_latest_text",
        "_owner_thread",
        "_rolling_distribution",
        "_rolling_trace",
        "_side_axis",
        "_side_count_ceiling",
        "_size",
        "_topology",
        "_title_override",
        "_value_label",
    )

    def __init__(
        self,
        document: FigureDocument,
        *,
        width: int,
        height: int,
        dpi: float = LIVE_PANEL_DPI,
        rolling_trace: bool = False,
        rolling_distribution: bool = False,
        value_label: str = "Signal",
        title: str | None = None,
        size_name: str | None = None,
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
                    figsize=(
                        panel_figure_size_inches(size_name)
                        if size_name is not None
                        else (width / dpi, height / dpi)
                    ),
                    dpi=dpi,
                )
                FigureCanvasAgg(figure)
                if rolling_distribution:
                    layout = (
                        rolling_panel_layout(size_name)
                        if size_name is not None
                        else rolling_panel_layout_for_raster(width, height)
                    )
                    axis = figure.add_axes(layout.history.matplotlib_bounds())
                    side_axis = figure.add_axes(
                        layout.distribution.matplotlib_bounds(),
                        sharey=axis,
                    )
                    side_axis.tick_params(
                        axis="y",
                        which="both",
                        left=False,
                        right=False,
                        labelleft=False,
                    )
                    side_axis.tick_params(
                        axis="both",
                        which="both",
                        bottom=False,
                        top=False,
                    )
                else:
                    axis = figure.add_axes(
                        (
                            panel_data_box(size_name)
                            if size_name is not None
                            else panel_data_box_for_raster(width, height)
                        ).matplotlib_bounds()
                    )
                    side_axis = None
                from .ticks import apply_smart_ticks

                apply_smart_ticks(axis)
            except BaseException:
                if figure is not None:
                    release_agg_figure(figure)
                figure = axis = None
                gc.collect()
                raise
        self._owner_thread = threading.get_ident()
        self._size = (width, height)
        self._rolling_distribution = bool(rolling_distribution)
        self._rolling_trace = bool(rolling_trace or rolling_distribution)
        self._value_label = str(value_label)
        self._title_override = None if title is None else str(title)
        self._document = document
        self._figure = figure
        self._axis = axis
        self._blit_cache = _AggBlitCache()
        self._side_axis = side_axis
        self._artists = ()
        self._fit_artists = ()
        self._fit_diagnostic_artists = ()
        self._distribution_artist = None
        self._gauss_artist = None
        self._gauss_text = None
        self._hist_fit_artists = ()
        self._hist_stats_text = None
        self._hist_threshold_artists = []
        self._latest_text = None
        self._side_count_ceiling = 0.0
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
        self._update_rolling_trace(
            tuple(pre_series_group),
            y_limits,
        )
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

    def _update_rolling_trace(
        self,
        series_group,
        y_limits: tuple[float, float],
    ) -> None:
        """Move Main's rolling readout and its optional side distribution."""

        if not self._rolling_trace:
            return

        first = series_group[0].data
        if not isinstance(first, EvaluatedCurve):
            raise TypeError("rolling trace requires EvaluatedCurve")
        values = np.asarray(first.values)
        validity = np.asarray(first.validity, dtype=bool)

        latest = None
        if values.size:
            indices = np.flatnonzero(validity)
            if indices.size:
                latest = float(values[int(indices[0])])
        label = "" if latest is None else f"{latest:.6g}"
        if self._latest_text is None:
            self._latest_text = self._axis.text(
                0.97,
                0.95,
                label,
                transform=self._axis.transAxes,
                color=PALETTE["readout"],
                ha="right",
                va="top",
                fontsize=small_fontsize(),
            )
        else:
            self._latest_text.set_text(label)

        side = self._side_axis
        if side is None:
            return
        from matplotlib.collections import PolyCollection
        from matplotlib.ticker import MaxNLocator

        finite = np.asarray(values[validity], dtype=np.float64)
        bin_count = max(3, min(len(first.x_axis.indices) // 4, 50))
        counts, edges = np.histogram(
            finite if finite.size else np.asarray([y_limits[0]]),
            bins=bin_count,
            range=y_limits,
        )
        vertices = np.empty((bin_count, 4, 2), dtype=np.float64)
        for index, count in enumerate(counts):
            low, high = edges[index], edges[index + 1]
            vertices[index] = (
                (0.0, low),
                (float(count), low),
                (float(count), high),
                (0.0, high),
            )
        if self._distribution_artist is None:
            self._distribution_artist = PolyCollection(
                vertices,
                facecolors=PALETTE["hist_fill"],
            )
            side.add_collection(self._distribution_artist)
            side.xaxis.set_major_locator(MaxNLocator(nbins=1, prune="lower"))
        else:
            self._distribution_artist.set_verts(vertices)
        peak = float(np.max(counts)) if counts.size else 0.0
        wanted = float(max(10, int(max(peak + 5.0, peak * 1.5))))
        if (
            self._side_count_ceiling <= 0.0
            or wanted > self._side_count_ceiling
            or wanted < 0.6 * self._side_count_ceiling
        ):
            self._side_count_ceiling = wanted
        side.set_xlim(0.0, self._side_count_ceiling)
        side.set_ylim(*y_limits)

        from zlc_data import fit_histogram

        fit = fit_histogram(edges, counts, "single")
        if self._gauss_artist is None:
            (self._gauss_artist,) = side.plot(
                (),
                (),
                color=PALETTE["fit_right"],
                alpha=1.0,
            )
        if fit.valid:
            coordinates = np.linspace(y_limits[0], y_limits[1], 100)
            self._gauss_artist.set_data(fit.evaluate(coordinates), coordinates)
            parameters = fit.single_parameters
            assert parameters is not None
            _amplitude, mean, sigma = parameters
            label = (
                r"$\sigma/\sqrt{\mu}$=N/A"
                if mean <= 0.0
                else rf"$\sigma$={sigma / np.sqrt(mean):.2f}$\sqrt{{\mu}}$"
            )
        else:
            self._gauss_artist.set_data((), ())
            label = ""
        if self._gauss_text is None:
            self._gauss_text = side.text(
                0.5,
                1.005,
                label,
                transform=side.transAxes,
                color=PALETTE["fit_right"],
                ha="center",
                va="bottom",
                fontsize=small_fontsize(),
            )
        else:
            self._gauss_text.set_text(label)

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
        effective_thresholds = self._update_histogram_presentation(
            axis,
            state,
            bin_projection,
        )
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
            thresholds=effective_thresholds,
        )

    def _update_histogram_presentation(
        self,
        axis,
        state: HistogramDisplayState,
        projection,
    ) -> tuple[float, ...]:
        """Update the shared full-panel presentation in place."""

        (
            self._hist_fit_artists,
            threshold_artists,
            self._hist_stats_text,
            thresholds,
        ) = _update_histogram_presentation(
            axis,
            state,
            projection.bin_counts,
            projection.bin_edges,
            fit_artists=self._hist_fit_artists,
            threshold_artists=self._hist_threshold_artists,
            stats_text=self._hist_stats_text,
            show_stats=True,
            threshold_linewidth=1.9,
        )
        self._hist_threshold_artists = list(threshold_artists)
        return thresholds

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
            # Resolution *values* are data, not artist topology.  A
            # LatestNonempty index normally advances every live revision and a
            # Fixed selector may resolve to another value without changing the
            # number or kind of artists.  Only the structural selector roles
            # belong in the persistent Agg topology key.
            tuple(
                (resolution.axis_id, resolution.selector)
                for resolution in layer.resolutions
            ),
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
                self._artists, _counts_group, _edges = _histogram(
                    axis,
                    series_group,
                    bins=histogram_bins,
                    projection=histogram_projection,
                )
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
                    _update_histogram_artist(
                        artist,
                        counts,
                        bin_projection.bin_edges,
                    )
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
            axis.set_xlabel(
                "Shots ago"
                if self._rolling_trace
                else _axis_label(first.x_axis)
            )
            axis.set_ylabel(self._value_label)
        elif isinstance(first, EvaluatedHistogram):
            axis.set_xlabel(self._value_label)
            axis.set_ylabel("Shots")
        title = (
            _panel_title(self._document, layer, cell, series_group)
            if self._title_override is None
            else self._title_override
        )
        current_title = axis.title
        if title:
            apply_title(axis, title)
        elif current_title.get_text():
            current_title.set_text("")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
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

    def _draw_raster(self, figure) -> RasterBuffer:
        dynamic = (
            *self._artists,
            *self._fit_artists,
            *self._fit_diagnostic_artists,
            self._distribution_artist,
            self._gauss_artist,
            self._gauss_text,
            *self._hist_fit_artists,
            self._hist_stats_text,
            *self._hist_threshold_artists,
            self._latest_text,
        )
        return self._blit_cache.raster(
            figure,
            dynamic,
            layout_key=_agg_layout_key(
                figure,
                extra=(
                    self._topology,
                    self._rolling_trace,
                    self._rolling_distribution,
                ),
            ),
            chrome_key=_agg_chrome_key(
                figure,
                extra=(
                    self._topology,
                    self._rolling_trace,
                    self._rolling_distribution,
                    self._value_label,
                    self._title_override,
                    self._side_count_ceiling,
                ),
            ),
            physical_size=self._size,
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
        self._fit_artists = ()
        self._fit_diagnostic_artists = ()
        self._topology = None
        self._blit_cache.clear()
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
    "SinglePanelAggRenderer",
]
