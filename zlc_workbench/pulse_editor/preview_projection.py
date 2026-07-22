"""Pure projection from the pulse-owned timeline to the existing renderer API.

This seam contains no Qt, Matplotlib, compiler, or authoring logic.  It merely
selects visible rows and translates the renderer-neutral, exact timeline into
the plain-data arguments consumed by the single pulse drawing implementation.
"""

from __future__ import annotations

from zlc_frontend.render_style import optimal_pulse_size
from zlc_pulse import PORT_DAC, PORT_DIGITAL, PulseTimelineDocument

from .repeat_presentation import pulse_repeat_presentation


def pulse_timeline_render_kwargs(
    timeline: PulseTimelineDocument,
    *,
    include_off_rows: bool,
    size: str | None = None,
) -> dict[str, object]:
    """Return the one plain-data rendering projection for screen and export."""

    if not isinstance(timeline, PulseTimelineDocument):
        raise TypeError("timeline must be PulseTimelineDocument")
    if type(include_off_rows) is not bool:
        raise TypeError("include_off_rows must be bool")

    digital, analog = _preview_rows(
        timeline,
        include_off_rows=include_off_rows,
    )
    tick_seconds = 1.0 / timeline.clock_hz

    pulses = []
    for row in digital:
        for segment in row.segments:
            if segment.start_value == 0:
                continue
            start = segment.start_tick * tick_seconds
            stop = segment.stop_tick * tick_seconds
            pulses.append(
                {
                    "channel": row.row_id,
                    "start": start,
                    "stop": stop,
                    "duration": stop - start,
                    "value": True,
                    # The formal source sequence never assigned per-edge names;
                    # period names remain represented by the period cards.
                    "name": "",
                }
            )

    analog_traces = []
    for row in analog:
        starts = [segment.start_tick * tick_seconds for segment in row.segments]
        starts.append(row.segments[-1].stop_tick * tick_seconds)
        analog_traces.append(
            {
                "name": row.row_id,
                "label": row.label,
                "min": row.value_range[0],
                "max": row.value_range[1],
                "starts": starts,
                "values": [segment.start_value for segment in row.segments],
            }
        )

    repeat_summary, repeat_index_spans, periods = _timeline_repeat_presentation(
        timeline
    )
    repeat_markers = []
    for start_index, end_index, label in repeat_index_spans:
        start_tick = periods[start_index].start_tick
        # The outer infinite bracket encloses the complete physical frame,
        # including a delayed-output tail.  A finite inner bracket remains on
        # its authored period boundaries.
        stop_tick = (
            timeline.duration_ticks
            if label == "×∞"
            else periods[end_index].stop_tick
        )
        repeat_markers.append(
            (start_tick * tick_seconds, stop_tick * tick_seconds, label)
        )
    scan_regions = []
    scan_dac_segments = []
    for annotation in timeline.annotations:
        start = annotation.start_tick * tick_seconds
        stop = annotation.stop_tick * tick_seconds
        if annotation.kind == "scan-duration":
            scan_regions.append(
                {"start": start, "stop": stop, "number": annotation.number}
            )
        elif annotation.kind == "scan-dac":
            scan_dac_segments.append(
                {
                    "trace_name": annotation.row_id,
                    "start": start,
                    "stop": stop,
                    "value": annotation.value,
                    "number": annotation.number,
                }
            )

    effective_size = size or optimal_pulse_size(len(digital), len(periods))
    return {
        "pulses": pulses,
        "channels": [row.row_id for row in digital],
        "channel_labels": {row.row_id: row.label for row in digital},
        "total_duration": timeline.logical_duration_ticks * tick_seconds,
        "title": timeline.title,
        "repeat_markers": repeat_markers,
        "repeat_notation": repeat_summary,
        "size": effective_size,
        "analog_traces": analog_traces,
        "scan_regions": scan_regions,
        "scan_dac_segments": scan_dac_segments,
    }


def pulse_repeat_summary(timeline: PulseTimelineDocument) -> str:
    """Return the formal editor wording derived by the shared repeat policy."""

    summary, _spans, _periods = _timeline_repeat_presentation(timeline)
    return summary


def pulse_preview_status(
    timeline: PulseTimelineDocument,
    *,
    include_off_rows: bool,
) -> str:
    """Format the exact formal status using digital rows as its denominator."""

    digital = tuple(
        row for row in timeline.rows if row.port_kind == PORT_DIGITAL
    )
    drawn, _analog = _preview_rows(
        timeline,
        include_off_rows=include_off_rows,
    )
    mode = "all channels" if include_off_rows else "active channels"
    notation = pulse_repeat_summary(timeline)
    return (
        f"{len(drawn)}/{len(digital)} plotted ({mode})"
        + (f" | {notation}" if notation else "")
    )


def _preview_rows(
    timeline: PulseTimelineDocument,
    *,
    include_off_rows: bool,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Select rows once for render geometry and its operator-facing count."""

    all_digital = tuple(
        row for row in timeline.rows if row.port_kind == PORT_DIGITAL
    )
    all_analog = tuple(row for row in timeline.rows if row.port_kind == PORT_DAC)
    if include_off_rows:
        return all_digital, all_analog
    active_digital = tuple(row for row in all_digital if row.active)
    # The formal preview has always retained one digital baseline as a spatial
    # reference when the entire TTL table is off.  This is a display fallback,
    # never a mutation or claim that the channel is physically active.
    digital = active_digital or all_digital[:1]
    return digital, tuple(row for row in all_analog if row.active)


def _timeline_repeat_presentation(
    timeline: PulseTimelineDocument,
) -> tuple[str, tuple[tuple[int, int, str], ...], tuple[object, ...]]:
    periods = tuple(
        annotation
        for annotation in timeline.annotations
        if annotation.kind == "period"
    )
    if not periods:
        raise RuntimeError("pulse timeline has no authored period annotations")
    annotations = tuple(
        annotation
        for annotation in timeline.annotations
        if annotation.kind == "repeat"
    )
    if len(annotations) > 1:
        raise RuntimeError("pulse timeline has more than one repeat annotation")
    repeat_spec = None
    if annotations:
        repeat = annotations[0]
        try:
            start = next(
                index
                for index, period in enumerate(periods)
                if period.start_tick == repeat.start_tick
            )
            end = next(
                index
                for index, period in enumerate(periods)
                if period.stop_tick == repeat.stop_tick
            )
            label = repeat.label[1:] if repeat.label.startswith("×") else repeat.label
            count = int(label)
        except (StopIteration, TypeError, ValueError) as exc:
            raise RuntimeError(
                "repeat annotation does not align with authored periods"
            ) from exc
        repeat_spec = (start, end, count)
    summary, spans = pulse_repeat_presentation(len(periods), repeat_spec)
    return summary, spans, periods


__all__ = [
    "pulse_preview_status",
    "pulse_repeat_summary",
    "pulse_timeline_render_kwargs",
]
