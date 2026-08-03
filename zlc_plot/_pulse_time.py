"""Time-coordinate conversion shared by pulse rendering and interaction."""

from __future__ import annotations

import math
from numbers import Real

from .primitives import PulseTimelineData


_SECONDS_PER_UNIT = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
}
_MINIMUM_SOURCE_SPAN = 1e-12


def _normalized_unit(unit: str) -> str:
    if not isinstance(unit, str):
        raise TypeError("pulse time unit must be text")
    normalized = unit.strip().lower().replace("μ", "µ")
    return "us" if normalized == "µs" else normalized


def pulse_content_bounds(payload: PulseTimelineData) -> tuple[float, float]:
    """Return the complete authored pulse bounds in the source time unit."""

    if not isinstance(payload, PulseTimelineData):
        raise TypeError("payload must be PulseTimelineData")
    starts = [0.0]
    stops = [_MINIMUM_SOURCE_SPAN]
    starts.extend(block.start for block in payload.blocks)
    stops.extend(block.stop for block in payload.blocks)
    starts.extend(region.start for region in payload.scan_regions)
    stops.extend(region.stop for region in payload.scan_regions)
    for trace in payload.analog_traces:
        if trace.starts:
            starts.append(trace.starts[0])
            stops.append(trace.starts[-1])
    starts.extend(marker.start for marker in payload.repeat_markers)
    stops.extend(marker.stop for marker in payload.repeat_markers)
    starts.extend(segment.start for segment in payload.scan_dac_segments)
    stops.extend(segment.stop for segment in payload.scan_dac_segments)
    if payload.total_duration is not None:
        stops.append(payload.total_duration)
    start = min(starts)
    stop = max(stops)
    if stop - start < _MINIMUM_SOURCE_SPAN:
        stop = start + _MINIMUM_SOURCE_SPAN
    return float(start), float(stop)


def pulse_time_scale(
    payload: PulseTimelineData,
    display_unit: str | None = None,
    *,
    source_span: float | None = None,
) -> tuple[float, str]:
    """Return ``source value * factor -> displayed value`` and its unit."""

    if not isinstance(payload, PulseTimelineData):
        raise TypeError("payload must be PulseTimelineData")
    source_unit = _normalized_unit(payload.time_unit)
    seconds_per_source = _SECONDS_PER_UNIT.get(source_unit)
    if seconds_per_source is None:
        if display_unit is None or _normalized_unit(display_unit) == source_unit:
            return 1.0, payload.time_unit
        raise ValueError(
            f"cannot convert pulse source unit {payload.time_unit!r} to {display_unit!r}"
        )
    if display_unit is None:
        if source_span is None:
            start, stop = pulse_content_bounds(payload)
            span = stop - start
        else:
            if isinstance(source_span, bool) or not isinstance(source_span, Real):
                raise TypeError("pulse source span must be a real number")
            span = float(source_span)
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError("pulse source span must be positive and finite")
        seconds = abs(span) * seconds_per_source
        if seconds < 1e-6:
            target = "ns"
        elif seconds < 1e-3:
            target = "us"
        elif seconds < 1.0:
            target = "ms"
        else:
            target = "s"
    else:
        target = _normalized_unit(display_unit)
    seconds_per_target = _SECONDS_PER_UNIT.get(target)
    if seconds_per_target is None:
        raise ValueError("pulse display unit must be ns, us, ms, or s")
    return seconds_per_source / seconds_per_target, target


__all__ = ["pulse_content_bounds", "pulse_time_scale"]
