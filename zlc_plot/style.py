"""Immutable visual design tokens shared by every plotting backend.

The module intentionally contains no Matplotlib artists.  Backends consume a
single :class:`PlotStyleConfig`, which keeps Qt raster, browser raster, Agg export and
FacetGrid cells on the same visual contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading
from typing import Any, Iterator, Mapping, TypeAlias

from ._image_raster import ImageFrontPolicy
from ._validation import finite_real, nonempty_text as _text
from .assets import HELVETICA_LIGHT_FAMILY


RcScalar: TypeAlias = str | int | float | bool | None
RcValue: TypeAlias = RcScalar | tuple[str, ...] | tuple[float, ...]


def _finite(value: object, field: str, *, positive: bool = False) -> float:
    result = finite_real(value, field)
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _unit_interval(value: object, field: str, *, include_zero: bool = True) -> float:
    result = _finite(value, field)
    lower_ok = result >= 0.0 if include_zero else result > 0.0
    if not lower_ok or result > 1.0:
        raise ValueError(f"{field} must be in {'[' if include_zero else '('}0, 1]")
    return result


def _color(value: object, field: str) -> str:
    return _text(value, field)


@dataclass(frozen=True, slots=True)
class RcParam:
    """One serialisable Matplotlib rcParam entry."""

    name: str
    value: RcValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "rcParam name"))
        value = self.value
        if isinstance(value, list):
            value = tuple(value)
            object.__setattr__(self, "value", value)
        if isinstance(value, tuple):
            if not value:
                raise ValueError("rcParam tuple values must not be empty")
            if all(isinstance(item, str) for item in value):
                if any(not item for item in value):
                    raise ValueError("rcParam string tuples must not contain empty values")
            elif all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in value
            ):
                value = tuple(float(item) for item in value)
                object.__setattr__(self, "value", value)
            else:
                raise TypeError("rcParam tuple values must be all strings or all finite numbers")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("rcParam numeric values must be finite")
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError("rcParam value is not JSON serialisable")


@dataclass(frozen=True, slots=True)
class FontStyleConfig:
    family: str
    fallbacks: tuple[str, ...]
    weight: str
    figure_title_pt: float
    axis_label_pt: float
    tick_pt: float
    legend_pt: float
    annotation_pt: float
    fit_annotation_pt: float
    facet_normal_scale: float
    facet_compact_scale: float
    pulse_repeat_pt: float
    pulse_bar_label_pt: float
    pulse_scan_annotation_pt: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _text(self.family, "font family"))
        object.__setattr__(self, "weight", _text(self.weight, "font weight"))
        fallbacks = tuple(self.fallbacks)
        if not fallbacks or any(not isinstance(item, str) or not item.strip() for item in fallbacks):
            raise ValueError("font fallbacks must contain non-empty names")
        object.__setattr__(self, "fallbacks", tuple(item.strip() for item in fallbacks))
        for field in (
            "figure_title_pt",
            "axis_label_pt",
            "tick_pt",
            "legend_pt",
            "annotation_pt",
            "fit_annotation_pt",
            "facet_normal_scale",
            "facet_compact_scale",
            "pulse_repeat_pt",
            "pulse_bar_label_pt",
            "pulse_scan_annotation_pt",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        if self.facet_compact_scale >= self.facet_normal_scale:
            raise ValueError("compact facet typography must be smaller than normal typography")

    @property
    def sans_serif(self) -> tuple[str, ...]:
        return (self.family, *self.fallbacks)

@dataclass(frozen=True, slots=True)
class PaletteConfig:
    series: tuple[str, ...]
    line_single: str
    pulse_cycle: tuple[str, ...]
    bracket_cycle: tuple[str, ...]
    hist_fill: str
    bright: str
    fit_left: str
    fit_right: str
    fit_total: str
    threshold: str
    annotation: str
    readout: str
    fit_text: str
    guide: str
    data_scatter: str
    pulse_name: str
    pulse_grid: str
    pulse_repeat_note: str
    bad: str

    def __post_init__(self) -> None:
        for field in ("series", "pulse_cycle", "bracket_cycle"):
            values = tuple(getattr(self, field))
            if not values:
                raise ValueError(f"{field} palette must not be empty")
            values = tuple(_color(item, f"{field} color") for item in values)
            object.__setattr__(self, field, values)
        for field in (
            "line_single",
            "hist_fill",
            "bright",
            "fit_left",
            "fit_right",
            "fit_total",
            "threshold",
            "annotation",
            "readout",
            "fit_text",
            "guide",
            "data_scatter",
            "pulse_name",
            "pulse_grid",
            "pulse_repeat_note",
            "bad",
        ):
            object.__setattr__(self, field, _color(getattr(self, field), field))

    @property
    def line_cycle(self) -> tuple[str, ...]:
        """Canonical Curve cycle: grey, sky blue, then qualitative colours."""

        return (self.line_single, self.bright, *self.series)

    def line_color(self, index: int) -> str:
        """Return the canonical colour for a zero-based source-series index."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("line colour index must be an integer")
        if index < 0:
            raise ValueError("line colour index must be non-negative")
        cycle = self.line_cycle
        return cycle[index % len(cycle)]


@dataclass(frozen=True, slots=True)
class LineToken:
    color: str
    linewidth: float
    alpha: float = 1.0
    linestyle: str = "-"
    marker: str | None = None
    zorder: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", _color(self.color, "line color"))
        object.__setattr__(self, "linewidth", _finite(self.linewidth, "line linewidth", positive=True))
        object.__setattr__(self, "alpha", _unit_interval(self.alpha, "line alpha"))
        object.__setattr__(self, "linestyle", _text(self.linestyle, "line linestyle"))
        if self.marker is not None and not isinstance(self.marker, str):
            raise TypeError("line marker must be text or None")
        if self.zorder is not None:
            object.__setattr__(self, "zorder", _finite(self.zorder, "line zorder"))

    def kwargs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "color": self.color,
            "linewidth": self.linewidth,
            "alpha": self.alpha,
            "linestyle": self.linestyle,
            "marker": self.marker,
        }
        if self.zorder is not None:
            values["zorder"] = self.zorder
        return values


@dataclass(frozen=True, slots=True)
class PointRingToken:
    color: str
    alpha: float
    linewidth: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", _color(self.color, "point ring color"))
        object.__setattr__(
            self,
            "alpha",
            _unit_interval(self.alpha, "point ring alpha"),
        )
        object.__setattr__(
            self,
            "linewidth",
            _finite(self.linewidth, "point ring linewidth", positive=True),
        )


@dataclass(frozen=True, slots=True)
class ArtistStyleConfig:
    curve: LineToken
    curve_marker_size_pt: float
    histogram_fill_alpha: float
    fit_dim_alpha: float
    threshold_line: LineToken
    curve_fit_line: LineToken
    bimodal_fit_lines: tuple[LineToken, LineToken, LineToken]
    fit_failure_color: str
    fit_radial_color: str
    fit_radial_center_area_pt2: float
    fit_radial_ring_linewidth: float
    fit_radial_ring_alpha: float
    side_distribution_linewidth: float
    pulse_scan_region_color: str
    pulse_scan_annotation_color: str
    point_unknown: PointRingToken
    point_empty: PointRingToken
    point_occupied: PointRingToken
    point_invalid: PointRingToken
    point_auto_radius_fraction: float = 0.3
    point_single_radius_fraction: float = 0.03
    point_zorder: float = 5.0
    point_label_zorder: float = 6.0
    selector_line_width: float = 1.0
    selector_alpha: float = 0.8
    selector_handle_size_pt: float = 3.25
    selector_handle_edge_width: float = 0.5
    selector_zorder: float = 20.0
    fit_annotation_zorder: float = 15.0
    crosshair_marker_size_pt: float = 2.0
    fit_source_scatter_area_pt2: float = 20.0
    fit_source_scatter_zorder: float = 4.0
    fit_radial_zorder: float = 6.0
    fit_component_sample_count: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.curve, LineToken):
            raise TypeError("curve must be LineToken")
        if not isinstance(self.threshold_line, LineToken):
            raise TypeError("threshold_line must be LineToken")
        if not isinstance(self.curve_fit_line, LineToken):
            raise TypeError("curve_fit_line must be LineToken")
        lines = tuple(self.bimodal_fit_lines)
        if len(lines) != 3 or any(not isinstance(item, LineToken) for item in lines):
            raise ValueError("bimodal_fit_lines must contain exactly three LineToken values")
        object.__setattr__(self, "bimodal_fit_lines", lines)
        if (
            isinstance(self.fit_component_sample_count, bool)
            or not isinstance(self.fit_component_sample_count, int)
            or self.fit_component_sample_count < 2
        ):
            raise ValueError("fit_component_sample_count must be an integer of at least two")
        for field in (
            "curve_marker_size_pt",
            "fit_radial_center_area_pt2",
            "fit_radial_ring_linewidth",
            "side_distribution_linewidth",
            "selector_line_width",
            "selector_handle_size_pt",
            "selector_handle_edge_width",
            "crosshair_marker_size_pt",
            "fit_source_scatter_area_pt2",
            "point_auto_radius_fraction",
            "point_single_radius_fraction",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        for field in (
            "histogram_fill_alpha",
            "fit_dim_alpha",
            "fit_radial_ring_alpha",
            "selector_alpha",
        ):
            object.__setattr__(self, field, _unit_interval(getattr(self, field), field))
        object.__setattr__(
            self,
            "selector_zorder",
            _finite(self.selector_zorder, "selector_zorder"),
        )
        object.__setattr__(
            self,
            "fit_annotation_zorder",
            _finite(self.fit_annotation_zorder, "fit_annotation_zorder"),
        )
        for field in (
            "point_zorder",
            "point_label_zorder",
            "fit_source_scatter_zorder",
            "fit_radial_zorder",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        for field in (
            "fit_failure_color",
            "fit_radial_color",
            "pulse_scan_region_color",
            "pulse_scan_annotation_color",
        ):
            object.__setattr__(self, field, _color(getattr(self, field), field))
        for field in (
            "point_unknown",
            "point_empty",
            "point_occupied",
            "point_invalid",
        ):
            if not isinstance(getattr(self, field), PointRingToken):
                raise TypeError(f"{field} must be PointRingToken")


@dataclass(frozen=True, slots=True)
class PulseStyleConfig:
    """Reference pulse-timeline geometry and stroke tokens."""

    row_height: float = 0.64
    dense_row_threshold: int = 10
    dense_min_row_height: float = 0.42
    dense_total_height: float = 6.4
    x_margin_fraction: float = 0.04
    repeat_right_margin_fraction: float = 0.05
    grid_linewidth: float = 0.35
    trace_linewidth: float = 0.65
    block_label_min_span_fraction: float = 0.09
    analog_zero_alpha: float = 0.5
    analog_zero_dash: tuple[float, tuple[float, float]] = (0.0, (4.0, 3.0))
    scan_region_alpha: float = 0.18
    scan_badge_pad: float = 0.3
    scan_dac_linewidth: float = 3.0
    scan_dac_alpha: float = 0.9
    repeat_tick_fraction: float = 0.024
    repeat_max_foot_fraction: float = 0.2
    repeat_min_foot_fraction: float = 0.006
    repeat_bottom: float = -0.42
    repeat_bottom_step: float = 0.13
    repeat_top_offset: float = -0.10
    repeat_top_step: float = 0.34
    repeat_alpha: float = 0.58
    repeat_linewidth: float = 1.05
    repeat_label_x_fraction: float = 0.12
    repeat_label_y_offset: float = 0.055
    repeat_note_position: tuple[float, float] = (0.995, 1.012)
    ylim_bottom: float = -0.62
    ylim_top_offset: float = -0.38
    repeat_ylim_top_offset: float = 0.78
    repeat_ylim_top_step: float = 0.26
    ytick_font_floor_pt: float = 4.8
    ytick_font_delta_pt: float = 1.2
    xtick_count: int = 5
    xtick_pad_pt: float = 2.0
    base_zorder: float = 3.0
    scan_region_zorder: float = 6.0
    scan_dac_zorder: float = 8.0
    annotation_zorder: float = 12.0
    repeat_bracket_zorder: float = 8.0
    repeat_label_zorder: float = 9.0

    def __post_init__(self) -> None:
        for field in ("dense_row_threshold", "xtick_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in (
            "row_height",
            "dense_min_row_height",
            "dense_total_height",
            "grid_linewidth",
            "trace_linewidth",
            "scan_badge_pad",
            "scan_dac_linewidth",
            "repeat_bottom_step",
            "repeat_top_step",
            "repeat_linewidth",
            "repeat_label_y_offset",
            "repeat_ylim_top_step",
            "ytick_font_floor_pt",
            "ytick_font_delta_pt",
            "xtick_pad_pt",
        ):
            object.__setattr__(
                self, field, _finite(getattr(self, field), field, positive=True)
            )
        for field in (
            "x_margin_fraction",
            "repeat_right_margin_fraction",
            "block_label_min_span_fraction",
            "repeat_tick_fraction",
            "repeat_max_foot_fraction",
            "repeat_min_foot_fraction",
            "repeat_label_x_fraction",
        ):
            object.__setattr__(
                self,
                field,
                _unit_interval(getattr(self, field), field, include_zero=False),
            )
        if self.dense_min_row_height > self.row_height:
            raise ValueError("dense_min_row_height must not exceed row_height")
        if self.repeat_min_foot_fraction > self.repeat_max_foot_fraction:
            raise ValueError("repeat foot minimum must not exceed its maximum")
        for field in (
            "repeat_bottom",
            "repeat_top_offset",
            "ylim_bottom",
            "ylim_top_offset",
            "repeat_ylim_top_offset",
            "base_zorder",
            "scan_region_zorder",
            "scan_dac_zorder",
            "annotation_zorder",
            "repeat_bracket_zorder",
            "repeat_label_zorder",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        for field in ("analog_zero_alpha", "scan_region_alpha", "scan_dac_alpha", "repeat_alpha"):
            object.__setattr__(self, field, _unit_interval(getattr(self, field), field))
        dash_offset, dash_pattern = self.analog_zero_dash
        dash = (
            _finite(dash_offset, "analog_zero_dash offset"),
            tuple(
                _finite(item, "analog_zero_dash item", positive=True)
                for item in dash_pattern
            ),
        )
        if not dash[1]:
            raise ValueError("analog_zero_dash pattern must not be empty")
        object.__setattr__(self, "analog_zero_dash", dash)
        note_position = tuple(
            _finite(item, "repeat_note_position") for item in self.repeat_note_position
        )
        if len(note_position) != 2:
            raise ValueError("repeat_note_position must contain x and y")
        object.__setattr__(self, "repeat_note_position", note_position)


@dataclass(frozen=True, slots=True)
class RenderPolicyConfig:
    """Renderer-wide visual geometry, image defaults and bounded-work policy."""

    image_default_colormap: str = "gray"
    image_colormaps: tuple[str, ...] = ("inferno", "viridis", "magma", "plasma", "gray")
    image_default_interpolation: str = "antialiased"
    facet_image_interpolation: str = "nearest"
    image_front: ImageFrontPolicy = ImageFrontPolicy()
    image_interpolations: tuple[str, ...] = (
        "none", "auto", "antialiased", "nearest", "bilinear", "bicubic",
        "spline16", "spline36", "hanning", "hamming", "hermite", "kaiser",
        "quadric", "catrom", "gaussian", "bessel", "mitchell", "sinc",
        "lanczos", "blackman",
    )
    image_origin: str = "upper"
    image_anchor: str = "W"
    axes_title_pad_pt: float = 2.5
    compact_axes_title_pad_pt: float = 1.5
    figure_title_y: float = 0.992
    colorbar_endpoint_label_pad_pt: float = -2.5
    colorbar_endpoint_label_chars: int = 5
    axes_text_inset_fraction: float = 0.025
    selector_label_line_fraction: float = 0.11
    selector_sample_colormap_fraction: float = 0.55
    selector_sample_alpha: float = 0.3
    colormap_low_fraction: float = 0.0
    colormap_high_fraction: float = 0.95
    image_spatial_max_ticks: int = 4
    side_distribution_fill_alpha: float = 1.0
    rolling_distribution_min_bins: int = 3
    image_distribution_min_bins: int = 8
    distribution_max_bins: int = 50
    distribution_bin_divisor: int = 4
    distribution_axis_max_ticks: int = 1
    distribution_guide_alpha: float = 0.3
    image_distribution_sample_target: int = 200_000
    image_color_padding_fraction: float = 0.10
    image_color_deadband_fraction: float = 0.35
    distribution_count_floor: int = 10
    distribution_count_headroom: float = 5.0
    distribution_count_growth: float = 1.5
    distribution_count_shrink_ratio: float = 0.6
    fit_source_scatter_max_points: int = 2000

    def __post_init__(self) -> None:
        if not isinstance(self.image_front, ImageFrontPolicy):
            raise TypeError("image_front must be ImageFrontPolicy")
        for field in ("image_colormaps", "image_interpolations"):
            values = tuple(_text(value, field) for value in getattr(self, field))
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{field} must contain unique non-empty values")
            object.__setattr__(self, field, values)
        for field in (
            "image_default_colormap",
            "image_default_interpolation",
            "facet_image_interpolation",
            "image_origin",
            "image_anchor",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.image_default_colormap not in self.image_colormaps:
            raise ValueError("image_default_colormap must be one of image_colormaps")
        if any(
            value not in self.image_interpolations
            for value in (
                self.image_default_interpolation,
                self.facet_image_interpolation,
            )
        ):
            raise ValueError("image interpolation defaults must be declared choices")
        if self.image_origin not in {"lower", "upper"}:
            raise ValueError("image_origin must be 'lower' or 'upper'")
        if self.image_anchor not in {"C", "SW", "S", "SE", "E", "NE", "N", "NW", "W"}:
            raise ValueError("image_anchor must be a Matplotlib cardinal anchor")
        for field in (
            "axes_title_pad_pt",
            "compact_axes_title_pad_pt",
            "figure_title_y",
            "colorbar_endpoint_label_pad_pt",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        for field in (
            "axes_text_inset_fraction",
            "selector_label_line_fraction",
            "selector_sample_colormap_fraction",
            "selector_sample_alpha",
            "colormap_low_fraction",
            "colormap_high_fraction",
            "side_distribution_fill_alpha",
            "distribution_guide_alpha",
            "image_color_padding_fraction",
            "image_color_deadband_fraction",
        ):
            object.__setattr__(self, field, _unit_interval(getattr(self, field), field))
        if self.colormap_low_fraction >= self.colormap_high_fraction:
            raise ValueError("colormap fractions must be strictly increasing")
        for field in (
            "image_spatial_max_ticks",
            "colorbar_endpoint_label_chars",
            "rolling_distribution_min_bins",
            "image_distribution_min_bins",
            "distribution_max_bins",
            "distribution_bin_divisor",
            "distribution_axis_max_ticks",
            "image_distribution_sample_target",
            "distribution_count_floor",
            "fit_source_scatter_max_points",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if max(
            self.rolling_distribution_min_bins,
            self.image_distribution_min_bins,
        ) > self.distribution_max_bins:
            raise ValueError("distribution minimum bins must not exceed the maximum")
        headroom = _finite(self.distribution_count_headroom, "distribution_count_headroom")
        growth = _finite(self.distribution_count_growth, "distribution_count_growth")
        shrink = _unit_interval(
            self.distribution_count_shrink_ratio,
            "distribution_count_shrink_ratio",
            include_zero=False,
        )
        if headroom < 0.0:
            raise ValueError("distribution_count_headroom must be non-negative")
        if growth < 1.0:
            raise ValueError("distribution_count_growth must be at least one")
        object.__setattr__(self, "distribution_count_headroom", headroom)
        object.__setattr__(self, "distribution_count_growth", growth)
        object.__setattr__(self, "distribution_count_shrink_ratio", shrink)


@dataclass(frozen=True, slots=True)
class PlotStyleConfig:
    """Complete backend-neutral visual style."""

    fonts: FontStyleConfig
    palette: PaletteConfig
    artists: ArtistStyleConfig
    pulse: PulseStyleConfig
    render: RenderPolicyConfig
    rc_params: tuple[RcParam, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fonts, FontStyleConfig):
            raise TypeError("fonts must be FontStyleConfig")
        if not isinstance(self.palette, PaletteConfig):
            raise TypeError("palette must be PaletteConfig")
        if not isinstance(self.artists, ArtistStyleConfig):
            raise TypeError("artists must be ArtistStyleConfig")
        if not isinstance(self.pulse, PulseStyleConfig):
            raise TypeError("pulse must be PulseStyleConfig")
        if not isinstance(self.render, RenderPolicyConfig):
            raise TypeError("render must be RenderPolicyConfig")
        params = tuple(self.rc_params)
        if any(not isinstance(item, RcParam) for item in params):
            raise TypeError("rc_params must contain RcParam values")
        names = tuple(item.name for item in params)
        if len(names) != len(set(names)):
            raise ValueError("rc_params names must be unique")
        conflicts = set(names) & set(self._derived_rc_params())
        if conflicts:
            raise ValueError(
                "rc_params duplicate typed style policy: "
                + ", ".join(sorted(conflicts))
            )
        object.__setattr__(self, "rc_params", params)

    def _derived_rc_params(self) -> dict[str, RcValue]:
        fonts = self.fonts
        curve = self.artists.curve
        render = self.render
        return {
            "font.size": fonts.annotation_pt,
            "font.weight": fonts.weight,
            "font.family": fonts.sans_serif,
            "font.sans-serif": fonts.sans_serif,
            "axes.labelsize": fonts.axis_label_pt,
            "axes.labelweight": fonts.weight,
            "axes.titlesize": fonts.figure_title_pt,
            "axes.titleweight": fonts.weight,
            "axes.titlepad": render.compact_axes_title_pad_pt,
            "figure.titlesize": fonts.figure_title_pt,
            "figure.titleweight": fonts.weight,
            "legend.fontsize": fonts.legend_pt,
            "legend.title_fontsize": fonts.legend_pt,
            "xtick.labelsize": fonts.tick_pt,
            "ytick.labelsize": fonts.tick_pt,
            "lines.linewidth": curve.linewidth,
            "lines.linestyle": curve.linestyle,
            "lines.marker": curve.marker or "None",
            "lines.markersize": self.artists.curve_marker_size_pt,
            "image.cmap": render.image_default_colormap,
            "image.origin": render.image_origin,
            "image.interpolation": render.image_default_interpolation,
        }

    def matplotlib_rc_params(self) -> dict[str, RcValue]:
        """Return a fresh dict suitable for ``matplotlib.rc_context``."""

        values = {item.name: item.value for item in self.rc_params}
        values.update(self._derived_rc_params())
        return values

def build_plot_style() -> PlotStyleConfig:
    """Build the package's immutable plotting style."""

    fonts = FontStyleConfig(
        family=HELVETICA_LIGHT_FAMILY,
        fallbacks=("Arial",),
        weight="normal",
        figure_title_pt=7.5,
        axis_label_pt=7.5,
        tick_pt=6.5,
        legend_pt=6.5,
        annotation_pt=6.5,
        fit_annotation_pt=3.25,
        facet_normal_scale=1.0,
        facet_compact_scale=0.5,
        pulse_repeat_pt=5.7,
        pulse_bar_label_pt=5.3,
        pulse_scan_annotation_pt=3.36,
    )
    palette = PaletteConfig(
        series=("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#666666", "#F0E442"),
        line_single="#808080",
        pulse_cycle=("#5D7583", "#C37D5A", "#6F8D73", "#A66E87", "#7A6FA4", "#B5A262", "#5E9A9A", "#9A765E", "#7890B5", "#8B8B8B", "#B97878", "#679174"),
        bracket_cycle=("#6A6A6A", "#C96F3D", "#4F7EA8", "#8B6BB8"),
        hist_fill="grey",
        bright="skyblue",
        fit_left="skyblue",
        fit_right="orange",
        fit_total="black",
        threshold="orange",
        annotation="black",
        readout="grey",
        fit_text="blue",
        guide="grey",
        data_scatter="lightgrey",
        pulse_name="white",
        pulse_grid="0.88",
        pulse_repeat_note="0.35",
        bad="#9a9a9a",
    )
    artists = ArtistStyleConfig(
        curve=LineToken(
            palette.line_single,
            1.0,
            alpha=0.8,
            marker=None,
        ),
        curve_marker_size_pt=2.0,
        histogram_fill_alpha=0.4,
        fit_dim_alpha=0.5,
        threshold_line=LineToken(palette.threshold, 1.9, alpha=0.95, zorder=5.0),
        curve_fit_line=LineToken(palette.fit_right, 2.0, alpha=0.5, zorder=5.0),
        bimodal_fit_lines=(
            LineToken(palette.fit_left, 1.0, alpha=0.8),
            LineToken(palette.fit_right, 1.0, alpha=0.8),
            LineToken(palette.fit_total, 1.0, alpha=0.35),
        ),
        fit_failure_color="#CD7380",
        fit_radial_color=palette.fit_right,
        fit_radial_center_area_pt2=50.0,
        fit_radial_ring_linewidth=2.0,
        fit_radial_ring_alpha=0.5,
        side_distribution_linewidth=3.25,
        pulse_scan_region_color="#D69A6E",
        pulse_scan_annotation_color="#FFFFFF",
        point_unknown=PointRingToken("#FFFFFF", 0.85, 0.6),
        point_empty=PointRingToken("#FFFFFF", 0.85, 0.6),
        point_occupied=PointRingToken("#D07850", 0.95, 0.9),
        point_invalid=PointRingToken("#CD7380", 0.95, 0.9),
    )
    pulse = PulseStyleConfig()
    render = RenderPolicyConfig()
    rc_params = (
        RcParam("scatter.edgecolors", "black"),
        RcParam("legend.numpoints", 1),
        RcParam("ytick.major.size", 1.5),
        RcParam("ytick.major.width", 0.4),
        RcParam("xtick.major.size", 1.5),
        RcParam("xtick.major.width", 0.4),
        RcParam("axes.linewidth", 0.4),
        RcParam("figure.subplot.left", 0.0),
        RcParam("figure.subplot.right", 1.0),
        RcParam("figure.subplot.bottom", 0.0),
        RcParam("figure.subplot.top", 1.0),
        RcParam("xtick.major.pad", 1.5),
        RcParam("ytick.major.pad", 1.5),
        RcParam("axes.labelpad", 1.5),
        RcParam("grid.linestyle", "--"),
        RcParam("axes.grid", False),
        RcParam("axes.facecolor", "white"),
        RcParam("figure.facecolor", "white"),
        RcParam("figure.autolayout", False),
        RcParam("figure.constrained_layout.use", False),
        RcParam("text.usetex", False),
        RcParam("xtick.top", False),
        RcParam("ytick.right", False),
        RcParam("xtick.minor.top", False),
        RcParam("ytick.minor.right", False),
        RcParam("xtick.minor.bottom", False),
        RcParam("ytick.minor.left", False),
        RcParam("xtick.direction", "in"),
        RcParam("ytick.direction", "in"),
        RcParam("legend.frameon", False),
        RcParam("savefig.facecolor", "white"),
        RcParam("savefig.edgecolor", "white"),
        RcParam("savefig.transparent", False),
        RcParam("text.color", "black"),
        RcParam("patch.edgecolor", "black"),
        RcParam("patch.force_edgecolor", False),
        RcParam("hatch.color", "black"),
        RcParam("axes.edgecolor", "black"),
        RcParam("axes.titlecolor", "black"),
        RcParam("axes.labelcolor", "black"),
        RcParam("xtick.color", "black"),
        RcParam("ytick.color", "black"),
    )
    return PlotStyleConfig(fonts, palette, artists, pulse, render, rc_params)


_MATPLOTLIB_COMPOSE_LOCK = threading.RLock()
_CANONICAL_FONT_REGISTERED = False


def _ensure_canonical_font_registered() -> None:
    global _CANONICAL_FONT_REGISTERED
    if _CANONICAL_FONT_REGISTERED:
        return
    from .assets import register_helvetica_light

    register_helvetica_light()
    _CANONICAL_FONT_REGISTERED = True


@contextmanager
def style_context(
    style: PlotStyleConfig,
    overrides: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Apply one immutable style on the serial Matplotlib compose lane."""

    if not isinstance(style, PlotStyleConfig):
        raise TypeError("style must be PlotStyleConfig")
    import matplotlib
    from cycler import cycler

    values = style.matplotlib_rc_params()
    values["axes.prop_cycle"] = cycler(color=style.palette.line_cycle)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must be a mapping or None")
        values.update(overrides)
    with _MATPLOTLIB_COMPOSE_LOCK:
        # Font registration belongs to the same serialized compose lane as the
        # rc mutation.  Any renderer that enters this context therefore gets
        # the package-owned face even when it did not pre-register the asset.
        _ensure_canonical_font_registered()
        with matplotlib.rc_context(values):
            yield


__all__ = [
    "ArtistStyleConfig",
    "FontStyleConfig",
    "ImageFrontPolicy",
    "LineToken",
    "PaletteConfig",
    "PointRingToken",
    "PlotStyleConfig",
    "PulseStyleConfig",
    "RcParam",
    "RcValue",
    "RenderPolicyConfig",
    "build_plot_style",
    "style_context",
]
