"""Internal Matplotlib implementation owner: pulse."""

from __future__ import annotations

import gc
from io import BytesIO
import math
import matplotlib
import numpy as np
from .curve_display import (
    CurveDisplayState,
    CurveViewportTransform,
    NumericDisplayAxis,
    NumericViewportTransform,
    curve_home_x_limits,
    numeric_curve_coordinates,
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
    release_agg_figure,
)

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

def _draw_pulse_repeat_brackets(
    axis, repeat_markers, n_channels, colors, home_x_limits
) -> None:
    """The grey square brackets that enclose a repeated span, with its ``×N`` /
    ``×∞`` label, exactly as the reference draws them: two vertical stems with
    short inward feet at each end, nested outward for multiple brackets.

    ``home_x_limits`` is the full-frame HOME span: the reference bakes its
    bracket artists ONCE at the home xlim and a zoom only re-windows them, so
    the feet/label geometry must always derive from the home span -- never the
    currently zoomed ``axis.get_xlim()`` -- or zooming would grow the feet and
    shift the label instead of just magnifying the picture."""

    import numpy as _np

    from .render_style import smaller_fontsize

    markers = [m for m in repeat_markers if m is not None]
    if not markers:
        return
    span = max(float(home_x_limits[1] - home_x_limits[0]), 1e-12)
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
            text = axis.text(stop + tick * 0.12, y_high + 0.055, label,
                             ha="left", va="bottom", color=color, alpha=alpha,
                             fontfamily="DejaVu Sans",
                             fontsize=smaller_fontsize(0.8, 5.5),
                             clip_on=False, zorder=9 + index)
            # clip_on=False is the reference's look (a zoomed-out label floats
            # in the margin), but this preview re-renders per zoom with
            # constrained layout while the reference lays out ONCE -- the
            # escaped label must not feed the layout solver or the axes width
            # breathes with every zoom step.
            text.set_in_layout(False)

def _draw_pulse_timeline(
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
    pixel_ratio: float = 1.0,
    x_limits=None,
    screen_pixel_exact: bool = True,
):
    """Draw the pulse-timeline figure once -- the reference's faithful preview.

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

    Returns ``(figure, axis, row_keys, row_labels)``.  The CALLER owns the
    figure's style context and release; the two public exits (PNG bytes and
    interactive raster+payload) wrap this one drawing so they can never
    disagree on a single pixel.
    """

    import matplotlib

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    from .render_style import (
        DESIGN_DPI, PALETTE, PANEL_DISPLAY_SCALE, PULSE_SCAN_ANNOTATION_COLOR,
        PULSE_SCAN_ANNOTATION_FONT_SIZE, PULSE_SCAN_REGION_COLOR, apply_title,
        panel_axes_bounds, panel_display_size, panel_figure_size_inches,
        smaller_fontsize,
    )

    # The size PRESET is the one geometry knob: inches come from the frontend's single size source and
    # the raster dpi matches a panel's on-screen scale, so the emitted PNG is exactly the pixel box a
    # card of this size reserves (panel_display_size(size)) -- preview, Save Figure and a seeded panel
    # all agree, and a bigger preset yields a bigger picture.
    size_inches = panel_figure_size_inches(size, kind="pulse")
    # ``pixel_ratio`` is the caller's SCREEN device-pixel ratio: the reference
    # renders every on-screen Agg buffer at exactly the widget's device pixels
    # (design dpi x display scale x real screen ratio) so the blit is 1:1
    # crisp, never rendered at logical pixels and stretched up by Qt.
    ratio = _render_dpi(pixel_ratio)
    dpi = _render_dpi(DESIGN_DPI * PANEL_DISPLAY_SCALE * ratio)
    if screen_pixel_exact:
        # Qt scales a positive logical extent with qRound semantics (half up),
        # while Matplotlib otherwise rounds each ``figsize * dpi`` edge by its
        # own floating-point path.  At fractional DPR (notably 125/150/175%)
        # those paths can differ by one physical pixel, forcing Qt to resample
        # the entire preview and making text/lines look soft.  Freeze the
        # existing logical panel size and make Agg's device raster exactly the
        # physical box Qt will paint.  This changes no layout token or UI size.
        logical_width, logical_height = panel_display_size(size, kind="pulse")
        physical_width = max(1, math.floor(logical_width * ratio + 0.5))
        physical_height = max(1, math.floor(logical_height * ratio + 0.5))
        size_inches = (physical_width / dpi, physical_height / dpi)
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

    # The formal plot geometry is a pure function of size/kind tokens.  A
    # viewport changes only xlim; it can never feed a layout solver.  This is
    # the headless equivalent of the reference FigureSpec + Divider path and
    # keeps axes, title and labels immobile through every live drag revision.
    figure = Figure(figsize=size_inches, dpi=dpi, constrained_layout=False)
    FigureCanvasAgg(figure)
    axis = figure.add_axes(panel_axes_bounds(size, kind="pulse"))
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

    # ``x_limits`` is a display-only VIEW override (the unified zoom/pan owner's
    # commit); the drawn geometry and the home span stay the full frame either way.
    if x_limits is not None:
        axis.set_xlim(float(x_limits[0]), float(x_limits[1]))
    else:
        axis.set_xlim(left_limit, right_limit)
    ylim_top = n_rows - 0.38
    if bracket_bounds:
        ylim_top = n_rows + 0.78 + 0.26 * max(0, len(bracket_bounds) - 1)
    axis.set_ylim(-0.62, ylim_top)
    axis.set_yticks([index_map[key] for key in row_keys])
    row_labels = (
        [labels.get(channel, channel) for channel in channels]
        + [str(trace.get("label") or trace.get("name") or "analog")
           for trace in analog_traces])
    axis.set_yticklabels(row_labels)
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
    _draw_pulse_repeat_brackets(
        axis, repeat_markers, n_rows, bracket_colors,
        (left_limit, right_limit))
    if title:
        # The ONE title mechanism (apply_title = title_fontsize, the stock label size); NOT
        # axis.set_title(), whose default 'large' titlesize dwarfs the compact preview and
        # crowds the repeat bracket right below it.
        apply_title(axis, str(title))
    return figure, axis, row_keys, row_labels, (left_limit, right_limit)

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
    pixel_ratio: float = 1.0,
    export: bool = False,
) -> bytes:
    """The pulse-timeline figure as PNG bytes (one exit of the shared drawing).

    Two dpi principles, both the reference's: an ON-SCREEN raster is emitted at
    the panel's logical pixel box times the caller's screen ``pixel_ratio`` (so
    a HiDPI display blits device pixels 1:1 instead of stretching a soft
    logical-pixel image), while an ``export`` write ignores the screen entirely
    and saves at the style's ``savefig.dpi`` (600) -- the same split as the
    reference's live canvas vs its bare ``savefig`` call.
    """

    if export and float(pixel_ratio) != 1.0:
        raise ValueError("an exported figure is screen-independent; leave pixel_ratio at 1")
    figure = None
    try:
        with render_style_context():
            figure, _axis, _row_keys, _row_labels, _home = _draw_pulse_timeline(
                pulses=pulses,
                channels=channels,
                channel_labels=channel_labels,
                total_duration=total_duration,
                title=title,
                repeat_markers=repeat_markers,
                repeat_notation=repeat_notation,
                size=size,
                analog_traces=analog_traces,
                scan_regions=scan_regions,
                scan_dac_segments=scan_dac_segments,
                pixel_ratio=pixel_ratio,
                screen_pixel_exact=not export,
            )
            buffer = BytesIO()
            if export:
                # No explicit dpi: the style context's savefig.dpi (600) applies.
                figure.savefig(buffer, format="png")
            else:
                figure.savefig(buffer, format="png", dpi=figure.get_dpi())
            return buffer.getvalue()
    finally:
        if figure is not None:
            release_agg_figure(figure)
        gc.collect()

def render_pulse_timeline_panel(
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
    document_input,
    display_revision: int = 0,
    pixel_ratio: float = 1.0,
    x_limits=None,
) -> "tuple[RasterBuffer, PulsePanelPayload]":
    """The SAME pulse-timeline picture as an interactive raster front.

    Twin exit of :func:`render_pulse_timeline_png`: one shared drawing, read
    back as an owned :class:`RasterBuffer` plus the draw-frozen x mapping
    (:class:`PulsePanelPayload`) the unified interaction owner consumes -- the
    pulse preview presents on the same board, with the same selector overlay,
    as every other panel kind instead of a bespoke picture label.  Gestures are
    x-only (time), so the home limits pin to the drawn frame span.

    ``document_input`` is the caller-owned exact pulse-document revision this
    picture was drawn from.  It is forwarded without inventing dataset/run/
    join/schema provenance.
    """

    figure = None
    try:
        with render_style_context():
            figure, axis, row_keys, row_labels, home = _draw_pulse_timeline(
                pulses=pulses,
                channels=channels,
                channel_labels=channel_labels,
                total_duration=total_duration,
                title=title,
                repeat_markers=repeat_markers,
                repeat_notation=repeat_notation,
                size=size,
                analog_traces=analog_traces,
                scan_regions=scan_regions,
                scan_dac_segments=scan_dac_segments,
                pixel_ratio=pixel_ratio,
                x_limits=x_limits,
            )
            figure.canvas.draw()
            width, height = figure.canvas.get_width_height()
            raster = RasterBuffer.from_agg_rgba(
                width, height, figure.canvas.buffer_rgba())
            drawn_x_limits = validated_display_range(
                tuple(float(value) for value in axis.get_xlim()),
                "drawn pulse x limits",
            )
            y_limits = validated_display_range(
                tuple(float(value) for value in axis.get_ylim()),
                "drawn pulse y limits",
            )
            home_x_limits = validated_display_range(
                tuple(float(value) for value in home),
                "pulse home x limits",
            )
            x0, y0, box_width, box_height = (
                float(value) for value in axis.bbox.bounds
            )
            plot_bounds = (
                x0 / raster.width,
                1.0 - (y0 + box_height) / raster.height,
                (x0 + box_width) / raster.width,
                1.0 - y0 / raster.height,
            )
            # The axis IDENTITY pins to the HOME (full-frame) span, never the
            # current zoomed view: hold matching and semantics checks compare
            # this axis, and a zoom must not read as a different panel.
            time_axis = NumericDisplayAxis("pulse.preview.time", "Time", "s")
            viewport = NumericViewportTransform(
                time_axis,
                int(display_revision),
                plot_bounds,
                drawn_x_limits,
                y_limits,
                home_x_limits,
            )
            return raster, PulsePanelPayload(
                document_input, viewport, tuple(row_keys), tuple(row_labels))
    finally:
        if figure is not None:
            release_agg_figure(figure)
        gc.collect()

__all__ = [
    "render_pulse_timeline_png",
    "render_pulse_timeline_panel",
]
