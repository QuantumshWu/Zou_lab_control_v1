"""Pulse timeline artist updates."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from .._pulse_time import pulse_content_bounds, pulse_time_scale
from ..primitives import PulseAnalogTrace, PulseTimelineData
from ..state import DisplayState
from ..style import PlotStyleConfig


def _sync_lines(axis: Any, artists: MutableMapping[str, Any], key: str, count: int) -> list[Any]:
    lines: list[Any] = artists.setdefault(key, [])
    while len(lines) < count:
        (line,) = axis.plot([], [])
        line.set_clip_on(True)
        lines.append(line)
    for index, line in enumerate(lines):
        line.set_visible(index < count)
    return lines


def _sync_rectangles(
    axis: Any,
    artists: MutableMapping[str, Any],
    key: str,
    count: int,
) -> list[Any]:
    from matplotlib.patches import Rectangle

    rectangles: list[Any] = artists.setdefault(key, [])
    while len(rectangles) < count:
        rectangle = Rectangle((0.0, 0.0), 0.0, 0.0)
        rectangle.set_clip_on(True)
        axis.add_patch(rectangle)
        rectangles.append(rectangle)
    for index, rectangle in enumerate(rectangles):
        rectangle.set_visible(index < count)
    return rectangles


def _sync_texts(axis: Any, artists: MutableMapping[str, Any], key: str, count: int) -> list[Any]:
    texts: list[Any] = artists.setdefault(key, [])
    while len(texts) < count:
        text = axis.text(0.0, 0.0, "")
        text.set_clip_on(True)
        texts.append(text)
    for index, text in enumerate(texts):
        text.set_visible(index < count)
    return texts


def _analog_geometry(
    trace: PulseAnalogTrace,
    row_base: float,
    row_height: float,
) -> tuple[str, float, float, float, np.ndarray, np.ndarray]:
    span = trace.maximum - trace.minimum
    zero_fraction = float(np.clip((0.0 - trace.minimum) / span, 0.0, 1.0))
    zero_y = row_base + row_height * zero_fraction
    starts = np.asarray(trace.starts, dtype=float)
    values = np.asarray(trace.values, dtype=float)
    return trace.name, trace.minimum, span, zero_y, starts, values


def update_pulse_timeline(
    axis: Any,
    payload: PulseTimelineData,
    state: DisplayState,
    style: PlotStyleConfig,
    artists: MutableMapping[str, Any],
) -> None:
    """Update a pulse timeline without replacing its data artists."""

    if not isinstance(payload, PulseTimelineData):
        raise TypeError("pulse timeline requires PulseTimelineData")
    if not isinstance(state, DisplayState):
        raise TypeError("state must be DisplayState")
    if not isinstance(style, PlotStyleConfig):
        raise TypeError("style must be PlotStyleConfig")
    if not isinstance(artists, MutableMapping):
        raise TypeError("artists must be a mutable mapping")

    channels = payload.channels
    analog_traces = payload.analog_traces
    repeat_markers = payload.repeat_markers
    scan_dac_segments = payload.scan_dac_segments
    pulse = style.pulse

    analog_keys = tuple(
        f"analog:{index}:{trace.name}"
        for index, trace in enumerate(analog_traces)
    )
    row_keys = tuple(channel.channel_id for channel in channels) + analog_keys
    row_count = len(row_keys)
    row_index = {key: row_count - 1 - index for index, key in enumerate(row_keys)}
    row_colors = {
        key: style.palette.pulse_cycle[index % len(style.palette.pulse_cycle)]
        for index, key in enumerate(row_keys)
    }
    row_height = (
        pulse.row_height
        if row_count <= pulse.dense_row_threshold
        else max(pulse.dense_min_row_height, pulse.dense_total_height / row_count)
    )
    baseline_offset = row_height / 2.0

    total_duration = 0.0 if payload.total_duration is None else payload.total_duration
    start_min, stop_max = pulse_content_bounds(payload)
    span = stop_max - start_min
    margin = max(span * pulse.x_margin_fraction, np.finfo(float).eps * span)
    left_limit = start_min - margin
    right_limit = stop_max + margin
    if repeat_markers:
        right_limit += span * pulse.repeat_right_margin_fraction

    baseline_y: dict[str, float] = {}
    baselines = _sync_lines(axis, artists, "pulse:baselines", len(channels))
    for index, channel in enumerate(channels):
        y = row_index[channel.channel_id] - baseline_offset
        baseline_y[channel.channel_id] = y
        line = baselines[index]
        line.set_data((left_limit, right_limit), (y, y))
        line.set_color(row_colors[channel.channel_id])
        line.set_linewidth(pulse.trace_linewidth)
        line.set_alpha(1.0)
        line.set_linestyle("-")
        line.set_zorder(pulse.base_zorder)

    blocks = _sync_rectangles(axis, artists, "pulse:blocks", len(payload.blocks))
    block_labels = _sync_texts(axis, artists, "pulse:block_labels", len(payload.blocks))
    label_style = {
        "fontsize": style.fonts.pulse_bar_label_pt,
        "color": style.palette.pulse_name,
    }
    label_duration = total_duration if total_duration > 0.0 else span
    for index, block in enumerate(payload.blocks):
        baseline = baseline_y[block.channel_id]
        color = row_colors[block.channel_id]
        rectangle = blocks[index]
        rectangle.set_xy((block.start, baseline))
        rectangle.set_width(block.stop - block.start)
        rectangle.set_height(row_height)
        rectangle.set_facecolor(color)
        rectangle.set_edgecolor("none")
        rectangle.set_linewidth(0.0)
        rectangle.set_alpha(1.0)
        rectangle.set_zorder(pulse.base_zorder)
        label = block_labels[index]
        label.set_position(((block.start + block.stop) / 2.0, row_index[block.channel_id]))
        label.set_text(block.label)
        label.set_ha("center")
        label.set_va("center")
        label.set_clip_on(True)
        label.set_zorder(pulse.base_zorder + 1.0)
        label.update(label_style)
        label.set_visible(
            bool(block.label)
            and (block.stop - block.start)
            >= pulse.block_label_min_span_fraction * label_duration
        )

    analog_zero_lines = _sync_lines(
        axis,
        artists,
        "pulse:analog_zero",
        len(analog_traces),
    )
    analog_value_lines = _sync_lines(
        axis,
        artists,
        "pulse:analog_values",
        len(analog_traces),
    )
    analog_ranges: dict[str, tuple[float, float, float]] = {}
    for index, (key, trace) in enumerate(zip(analog_keys, analog_traces)):
        row_base = row_index[key] - baseline_offset
        baseline_y[key] = row_base
        name, minimum, value_span, zero_y, trace_starts, trace_values = _analog_geometry(
            trace,
            row_base,
            row_height,
        )
        analog_ranges[name] = (row_base, minimum, value_span)
        color = row_colors[key]
        zero_line = analog_zero_lines[index]
        zero_line.set_data((left_limit, right_limit), (zero_y, zero_y))
        zero_line.set_color(color)
        zero_line.set_linewidth(pulse.trace_linewidth)
        zero_line.set_alpha(pulse.analog_zero_alpha)
        zero_line.set_linestyle(pulse.analog_zero_dash)
        zero_line.set_zorder(pulse.base_zorder + 1.0)
        value_line = analog_value_lines[index]
        count = min(trace_values.size, max(0, trace_starts.size - 1))
        if count:
            x = np.repeat(trace_starts[: count + 1], 2)[1:-1]
            y_values = row_base + row_height * np.clip(
                (trace_values[:count] - minimum) / value_span,
                0.0,
                1.0,
            )
            value_line.set_data(x, np.repeat(y_values, 2))
            value_line.set_visible(True)
        else:
            value_line.set_data((left_limit, right_limit), (zero_y, zero_y))
            value_line.set_visible(True)
        value_line.set_color(color)
        value_line.set_linewidth(pulse.trace_linewidth)
        value_line.set_alpha(1.0)
        value_line.set_linestyle("-")
        value_line.set_zorder(pulse.base_zorder + 2.0)

    show_scan = bool(state["show_scan_regions"])
    scan_rectangles = _sync_rectangles(
        axis,
        artists,
        "pulse:scan_regions",
        len(payload.scan_regions),
    )
    scan_labels = _sync_texts(
        axis,
        artists,
        "pulse:scan_labels",
        len(payload.scan_regions),
    )
    area_bottom = min(baseline_y.values(), default=-baseline_offset)
    area_top = max(baseline_y.values(), default=0.0) + row_height
    scan_text = {
        "fontsize": style.fonts.pulse_scan_annotation_pt,
        "color": style.artists.pulse_scan_annotation_color,
    }
    badge = {
        "boxstyle": f"circle,pad={pulse.scan_badge_pad:g}",
        "facecolor": style.artists.pulse_scan_region_color,
        "edgecolor": "none",
    }
    for index, region in enumerate(payload.scan_regions):
        rectangle = scan_rectangles[index]
        rectangle.set_xy((region.start, area_bottom))
        rectangle.set_width(region.stop - region.start)
        rectangle.set_height(area_top - area_bottom)
        rectangle.set_facecolor(style.artists.pulse_scan_region_color)
        rectangle.set_edgecolor("none")
        rectangle.set_alpha(pulse.scan_region_alpha)
        rectangle.set_zorder(pulse.scan_region_zorder)
        rectangle.set_visible(show_scan)
        text = scan_labels[index]
        text.set_position(((region.start + region.stop) / 2.0, area_top - row_height / 2.0))
        text.set_text(str(region.number))
        text.set_ha("center")
        text.set_va("center")
        text.set_zorder(pulse.annotation_zorder)
        text.set_bbox(badge)
        text.update(scan_text)
        text.set_visible(show_scan)

    dac_lines = _sync_lines(
        axis,
        artists,
        "pulse:scan_dac",
        len(scan_dac_segments),
    )
    dac_labels = _sync_texts(
        axis,
        artists,
        "pulse:scan_dac_labels",
        len(scan_dac_segments),
    )
    for index, segment in enumerate(scan_dac_segments):
        geometry = analog_ranges[segment.trace_name]
        line = dac_lines[index]
        text = dac_labels[index]
        row_base, minimum, value_span = geometry
        start = segment.start
        stop = segment.stop
        value = segment.value
        y = row_base + row_height * float(np.clip((value - minimum) / value_span, 0.0, 1.0))
        line.set_data((start, stop), (y, y))
        line.set_color(style.artists.pulse_scan_region_color)
        line.set_linewidth(pulse.scan_dac_linewidth)
        line.set_alpha(pulse.scan_dac_alpha)
        line.set_solid_capstyle("butt")
        line.set_zorder(pulse.scan_dac_zorder)
        line.set_visible(show_scan)
        number = segment.number
        text.set_position(((start + stop) / 2.0, row_base + row_height / 2.0))
        text.set_text(str(number) if number is not None else "")
        text.set_ha("center")
        text.set_va("center")
        text.set_zorder(pulse.annotation_zorder)
        text.set_bbox(badge)
        text.update(scan_text)
        text.set_visible(show_scan and number is not None)

    left_brackets = _sync_lines(
        axis,
        artists,
        "pulse:repeat_left",
        len(repeat_markers),
    )
    right_brackets = _sync_lines(
        axis,
        artists,
        "pulse:repeat_right",
        len(repeat_markers),
    )
    bracket_labels = _sync_texts(
        axis,
        artists,
        "pulse:repeat_labels",
        len(repeat_markers),
    )
    home_span = right_limit - left_limit
    tick_base = home_span * pulse.repeat_tick_fraction
    for index, marker in enumerate(repeat_markers):
        start, stop, label_value = marker.start, marker.stop, marker.label
        color = style.palette.bracket_cycle[index % len(style.palette.bracket_cycle)]
        outer_depth = max(0, len(repeat_markers) - 1 - index)
        y_low = pulse.repeat_bottom - pulse.repeat_bottom_step * outer_depth
        y_high = row_count + pulse.repeat_top_offset + pulse.repeat_top_step * outer_depth
        tick = min(
            max(tick_base, home_span * pulse.repeat_min_foot_fraction),
            (stop - start) * pulse.repeat_max_foot_fraction,
        )
        left_line = left_brackets[index]
        right_line = right_brackets[index]
        left_line.set_data(
            (start + tick, start, start, start + tick),
            (y_high, y_high, y_low, y_low),
        )
        right_line.set_data(
            (stop - tick, stop, stop, stop - tick),
            (y_high, y_high, y_low, y_low),
        )
        for line in (left_line, right_line):
            line.set_color(color)
            line.set_alpha(pulse.repeat_alpha)
            line.set_linewidth(pulse.repeat_linewidth)
            line.set_solid_capstyle("round")
            line.set_clip_on(True)
            line.set_zorder(pulse.repeat_bracket_zorder + index)
        text = bracket_labels[index]
        text.set_position(
            (
                stop + tick * pulse.repeat_label_x_fraction,
                y_high + pulse.repeat_label_y_offset,
            )
        )
        text.set_text(label_value)
        text.set_ha("left")
        text.set_va("bottom")
        text.set_alpha(pulse.repeat_alpha)
        text.set_clip_on(True)
        text.set_in_layout(False)
        text.set_zorder(pulse.repeat_label_zorder + index)
        text.update({"fontsize": style.fonts.pulse_repeat_pt, "color": color})
        text.set_visible(bool(label_value))

    repeat_note = artists.get("pulse:repeat_note")
    if repeat_note is None:
        repeat_note = axis.text(*pulse.repeat_note_position, "", transform=axis.transAxes)
        repeat_note.set_clip_on(True)
        repeat_note.set_in_layout(False)
        artists["pulse:repeat_note"] = repeat_note
    notation = payload.repeat_notation
    repeat_note.set_text(notation)
    repeat_note.set_ha("right")
    repeat_note.set_va("bottom")
    repeat_note.update(
        {
            "fontsize": style.fonts.pulse_repeat_pt,
            "color": style.palette.pulse_repeat_note,
        }
    )
    repeat_note.set_visible(bool(notation) and not repeat_markers)

    axis.set_xlim(left_limit, right_limit)
    top_limit = row_count + pulse.ylim_top_offset
    if repeat_markers:
        top_limit = (
            row_count
            + pulse.repeat_ylim_top_offset
            + pulse.repeat_ylim_top_step * max(0, len(repeat_markers) - 1)
        )
    axis.set_ylim(pulse.ylim_bottom, top_limit)
    axis.set_yticks([row_index[key] for key in row_keys])
    row_labels = [channel.label for channel in channels] + [
        trace.label for trace in analog_traces
    ]
    axis.set_yticklabels(
        row_labels,
        fontsize=max(
            pulse.ytick_font_floor_pt,
            style.fonts.tick_pt - pulse.ytick_font_delta_pt,
        ),
    )
    for tick, key in zip(axis.get_yticklabels(), row_keys):
        tick.set_color(row_colors[key])

    display_factor, display_unit = pulse_time_scale(
        payload,
        state["x_display_unit"],
        source_span=span,
    )
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    axis.set_xlabel(
        f"Time ({display_unit})",
        fontsize=style.fonts.axis_label_pt,
    )
    axis.set_ylabel("")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=pulse.xtick_count, prune="both"))
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: ""
            if value < 0.0
            else f"{value * display_factor:.4g}"
        )
    )
    axis.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
        pad=pulse.xtick_pad_pt,
    )
    axis.set_axisbelow(True)
    axis.grid(False, axis="y")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(True)
    axis.spines["left"].set_visible(True)
__all__ = ["update_pulse_timeline"]
