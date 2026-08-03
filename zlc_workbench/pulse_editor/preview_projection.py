"""Pure Pulse-domain projection into the public :mod:`zlc_plot` vocabulary.

Pulse authoring owns the exact timeline.  ``zlc_plot`` owns every drawing,
layout, selector, raster and export decision.  This module is the deliberately
small seam between those owners: it selects visible Pulse rows and constructs
the immutable plot input/spec consumed by every presentation backend.
"""

from __future__ import annotations

from zlc_plot import (
    DEFAULTS,
    PlotLabels,
    PulseAnalogTrace,
    PulseBlock,
    PulseChannel,
    PulseDacScanSegment,
    PulseRepeatMarker,
    PulseScanRegion,
    PulseTimelineData,
    PulseTimelinePlot,
)
from zlc_pulse import PORT_DAC, PORT_DIGITAL, PulseTimelineDocument

from .repeat_presentation import pulse_repeat_presentation


def pulse_timeline_plot(
    timeline: PulseTimelineDocument,
    *,
    include_off_rows: bool,
) -> tuple[PulseTimelineData, PulseTimelinePlot]:
    """Project one exact Pulse timeline into the sole plotting API."""

    if not isinstance(timeline, PulseTimelineDocument):
        raise TypeError("timeline must be PulseTimelineDocument")
    if type(include_off_rows) is not bool:
        raise TypeError("include_off_rows must be bool")

    digital, analog = _preview_rows(
        timeline,
        include_off_rows=include_off_rows,
    )
    tick_seconds = 1.0 / timeline.clock_hz

    blocks = tuple(
        PulseBlock(
            channel_id=row.row_id,
            start=segment.start_tick * tick_seconds,
            stop=segment.stop_tick * tick_seconds,
        )
        for row in digital
        for segment in row.segments
        if segment.start_value != 0
    )
    analog_traces = tuple(
        PulseAnalogTrace(
            name=row.row_id,
            label=row.label,
            minimum=row.value_range[0],
            maximum=row.value_range[1],
            starts=tuple(
                [
                    segment.start_tick * tick_seconds
                    for segment in row.segments
                ]
                + [row.segments[-1].stop_tick * tick_seconds]
            ),
            values=tuple(segment.start_value for segment in row.segments),
        )
        for row in analog
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
            PulseRepeatMarker(
                start_tick * tick_seconds,
                stop_tick * tick_seconds,
                label,
            )
        )

    scan_regions = []
    scan_dac_segments = []
    for annotation in timeline.annotations:
        start = annotation.start_tick * tick_seconds
        stop = annotation.stop_tick * tick_seconds
        if annotation.kind == "scan-duration":
            scan_regions.append(
                PulseScanRegion(start, stop, annotation.number)
            )
        elif annotation.kind == "scan-dac":
            scan_dac_segments.append(
                PulseDacScanSegment(
                    annotation.row_id,
                    start,
                    stop,
                    annotation.value,
                    annotation.number,
                )
            )

    data = PulseTimelineData(
        channels=tuple(PulseChannel(row.row_id, row.label) for row in digital),
        blocks=blocks,
        scan_regions=tuple(scan_regions),
        time_unit="s",
        total_duration=timeline.logical_duration_ticks * tick_seconds,
        analog_traces=analog_traces,
        repeat_markers=tuple(repeat_markers),
        repeat_notation=repeat_summary,
        scan_dac_segments=tuple(scan_dac_segments),
    )
    return data, PulseTimelinePlot(labels=PlotLabels(title=timeline.title))


def recommended_pulse_size(
    timeline: PulseTimelineDocument,
    *,
    include_off_rows: bool,
) -> str:
    """Choose the smallest declared zlc_plot preset that keeps rows legible."""

    if not isinstance(timeline, PulseTimelineDocument):
        raise TypeError("timeline must be PulseTimelineDocument")
    if type(include_off_rows) is not bool:
        raise TypeError("include_off_rows must be bool")
    digital, analog = _preview_rows(
        timeline,
        include_off_rows=include_off_rows,
    )
    _summary, _spans, periods = _timeline_repeat_presentation(timeline)
    layout = DEFAULTS.layout
    row_count = max(1, len(digital) + len(analog))
    period_count = max(1, len(periods))
    ordered = sorted(
        layout.presets,
        key=lambda preset: preset.rows * preset.columns,
    )
    for preset in ordered:
        if (
            preset.rows * layout.panel_unit.height
            >= row_count * layout.pulse_row_min_px
            and preset.columns * layout.panel_unit.width
            >= period_count * layout.pulse_period_min_px
        ):
            return preset.name
    return ordered[-1].name


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
    # Preserve the formal preview's one digital spatial reference when every
    # TTL row is off.  It is a display row only, never an authored state change.
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
    "pulse_timeline_plot",
    "recommended_pulse_size",
]
