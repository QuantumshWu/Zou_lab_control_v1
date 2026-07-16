"""Exact, renderer-neutral pulse timeline projection for the authoring UI."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from zlc_storage import canonical_text, nonnegative_integer, positive_integer

from .artifact import CompiledPulseArtifact
from .document import PulseDocument, PulsePeriod, _exact_ticks
from .ir import TargetBusSegment, TargetIR
from .physical import (
    iter_physical_digital_high_intervals,
    physical_digital_playback_terminal_tick,
)
from .target import PORT_DAC, PORT_DIGITAL


@dataclass(frozen=True, slots=True)
class PulseTimelineSegment:
    start_tick: int
    stop_tick: int
    start_value: int
    stop_value: int

    def __post_init__(self) -> None:
        start = nonnegative_integer(self.start_tick, "timeline segment start_tick")
        stop = positive_integer(self.stop_tick, "timeline segment stop_tick")
        if stop <= start:
            raise ValueError("timeline segment must have positive duration")
        for field in ("start_value", "stop_value"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
        object.__setattr__(self, "start_tick", start)
        object.__setattr__(self, "stop_tick", stop)


@dataclass(frozen=True, slots=True)
class PulseTimelineRow:
    row_id: str
    label: str
    unit: str
    value_range: tuple[int, int]
    segments: tuple[PulseTimelineSegment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", canonical_text(self.row_id, "timeline row_id"))
        object.__setattr__(self, "label", canonical_text(self.label, "timeline row label"))
        object.__setattr__(self, "unit", canonical_text(self.unit, "timeline row unit"))
        value_range = tuple(self.value_range)
        if (
            len(value_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in value_range)
            or value_range[1] < value_range[0]
        ):
            raise ValueError("timeline row value_range must be an ordered integer pair")
        segments = tuple(self.segments)
        if not segments or any(not isinstance(item, PulseTimelineSegment) for item in segments):
            raise ValueError("timeline row requires PulseTimelineSegment values")
        if segments[0].start_tick != 0 or any(
            left.stop_tick != right.start_tick
            for left, right in zip(segments, segments[1:])
        ):
            raise ValueError("timeline row segments must be contiguous from tick zero")
        object.__setattr__(self, "value_range", value_range)
        object.__setattr__(self, "segments", segments)


@dataclass(frozen=True, slots=True)
class PulseTimelineAnnotation:
    kind: str
    start_tick: int
    stop_tick: int
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", canonical_text(self.kind, "annotation kind"))
        start = nonnegative_integer(self.start_tick, "annotation start_tick")
        stop = positive_integer(self.stop_tick, "annotation stop_tick")
        if stop <= start:
            raise ValueError("timeline annotation must have positive duration")
        object.__setattr__(self, "start_tick", start)
        object.__setattr__(self, "stop_tick", stop)
        object.__setattr__(self, "label", canonical_text(self.label, "annotation label"))


@dataclass(frozen=True, slots=True)
class PulseTimelineDocument:
    title: str
    source_document_digest: str
    target_ir_digest: str
    clock_hz: float
    logical_duration_ticks: int
    duration_ticks: int
    reference_label: str
    rows: tuple[PulseTimelineRow, ...]
    annotations: tuple[PulseTimelineAnnotation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", canonical_text(self.title, "timeline title"))
        canonical_text(self.source_document_digest, "source_document_digest")
        canonical_text(self.target_ir_digest, "target_ir_digest")
        if isinstance(self.clock_hz, bool) or not isinstance(self.clock_hz, (int, float)):
            raise TypeError("clock_hz must be numeric")
        if float(self.clock_hz) <= 0:
            raise ValueError("clock_hz must be positive")
        object.__setattr__(self, "clock_hz", float(self.clock_hz))
        logical = positive_integer(self.logical_duration_ticks, "logical_duration_ticks")
        duration = positive_integer(self.duration_ticks, "duration_ticks")
        if duration < logical:
            raise ValueError("physical timeline cannot end before logical execution")
        object.__setattr__(self, "logical_duration_ticks", logical)
        object.__setattr__(self, "duration_ticks", duration)
        object.__setattr__(
            self,
            "reference_label",
            canonical_text(self.reference_label, "reference_label"),
        )
        rows = tuple(self.rows)
        annotations = tuple(self.annotations)
        if not rows or any(not isinstance(item, PulseTimelineRow) for item in rows):
            raise ValueError("pulse timeline requires at least one row")
        if len({item.row_id for item in rows}) != len(rows):
            raise ValueError("pulse timeline row_id values must be unique")
        if any(row.segments[-1].stop_tick != duration for row in rows):
            raise ValueError("every timeline row must cover the physical duration")
        if any(not isinstance(item, PulseTimelineAnnotation) for item in annotations):
            raise TypeError("annotations must contain PulseTimelineAnnotation values")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "annotations", annotations)


class _ItemBudget:
    __slots__ = ("_limit", "_used")

    def __init__(self, limit: int) -> None:
        self._limit = positive_integer(limit, "max_timeline_items")
        self._used = 0

    def take(self, count: int = 1) -> None:
        self._used += count
        if self._used > self._limit:
            raise ValueError(
                f"pulse preview requires more than {self._limit} timeline items"
            )


def build_pulse_timeline(
    document: PulseDocument,
    artifact: CompiledPulseArtifact,
    *,
    reference_label: str,
    max_timeline_items: int = 50_000,
) -> PulseTimelineDocument:
    """Project one finite static/reference compile without approximating scan rows."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if artifact.source_document_digest != document.fingerprint:
        raise ValueError("compiled pulse artifact belongs to another PulseDocument")
    target_ir = artifact.target_ir
    if target_ir.scan_points:
        raise ValueError("timeline preview requires a static or nominal-reference compile")
    if target_ir.repeat_forever:
        raise ValueError("timeline preview displays one finite compile, not an unbounded Run")

    budget = _ItemBudget(max_timeline_items)
    logical_duration = physical_digital_playback_terminal_tick(target_ir)
    high_by_output: dict[str, list[tuple[int, int]]] = {
        key: [] for key, _lane in target_ir.logical_digital_outputs
    }
    for interval in iter_physical_digital_high_intervals(
        target_ir,
        max_assignments=max_timeline_items * 4,
        max_work_items=max_timeline_items * 8,
    ):
        budget.take()
        high_by_output[interval.output_key].append(
            (interval.start_tick, interval.stop_tick)
        )

    bus_actions = _bus_actions(target_ir, budget)
    physical_duration = max(
        (
            logical_duration,
            *(stop for values in high_by_output.values() for _start, stop in values),
            *(
                stop
                for values in bus_actions.values()
                for _start, stop, _mode, _value in values
            ),
        )
    )
    visible = document.visible_ports or tuple(
        port.key
        for port in document.target.ports
        if port.kind in (PORT_DIGITAL, PORT_DAC)
    )
    rows: list[PulseTimelineRow] = []
    for key in visible:
        port = document.target.by_key[key]
        if port.kind == PORT_DIGITAL:
            segments = _digital_segments(
                high_by_output.get(key, ()),
                physical_duration,
                budget,
            )
            rows.append(
                PulseTimelineRow(key, port.label, "logic", (0, 1), segments)
            )
        elif port.kind == PORT_DAC:
            assert port.bus_index is not None and port.signed_range is not None
            segments = _bus_segments(
                bus_actions.get(port.bus_index, ()),
                safe_value=port.safe_value,
                duration=physical_duration,
                budget=budget,
            )
            rows.append(
                PulseTimelineRow(
                    key,
                    port.label,
                    "DAC code",
                    port.signed_range,
                    segments,
                )
            )
    if not rows:
        raise ValueError("PulseDocument has no visible digital or DAC rows")
    annotations = _period_annotations(document, budget)
    return PulseTimelineDocument(
        document.name,
        document.fingerprint,
        target_ir.fingerprint,
        target_ir.clock_hz,
        logical_duration,
        physical_duration,
        reference_label,
        tuple(rows),
        annotations,
    )


def _bus_actions(
    ir: TargetIR,
    budget: _ItemBudget,
) -> dict[int, tuple[tuple[int, int, str, int], ...]]:
    if any(item.value_select or item.stop_value_select for item in ir.bus_segments):
        raise ValueError("static timeline contains unresolved DAC selectors")
    delay_by_bus = {item.bus_index: item.delay_ticks for item in ir.bus_delays}
    actions: dict[int, list[tuple[int, int, str, int]]] = {
        index: [] for index in range(len(ir.bus_names))
    }
    loop_start = ir.ticks[ir.loop_start_index]
    loop_end = ir.loop_end_tick
    loop_span = loop_end - loop_start
    for segment in ir.bus_segments:
        delay = delay_by_bus.get(segment.bus_index, 0)
        for start, stop in _expanded_segment_times(
            segment,
            loop_start=loop_start,
            loop_end=loop_end,
            loop_span=loop_span,
            loop_count=ir.loop_count,
        ):
            budget.take()
            actions[segment.bus_index].append(
                (start + delay, stop + delay, segment.mode, segment.stop_value)
            )
    return {
        index: tuple(sorted(values, key=lambda item: (item[0], item[1])))
        for index, values in actions.items()
    }


def _expanded_segment_times(
    segment: TargetBusSegment,
    *,
    loop_start: int,
    loop_end: int,
    loop_span: int,
    loop_count: int,
) -> Iterator[tuple[int, int]]:
    if loop_count == 1 or segment.start_tick < loop_start:
        yield segment.start_tick, segment.stop_tick
        return
    if segment.start_tick < loop_end:
        for iteration in range(loop_count):
            yield (
                segment.start_tick + iteration * loop_span,
                segment.stop_tick + iteration * loop_span,
            )
        return
    shift = (loop_count - 1) * loop_span
    yield segment.start_tick + shift, segment.stop_tick + shift


def _digital_segments(
    highs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    duration: int,
    budget: _ItemBudget,
) -> tuple[PulseTimelineSegment, ...]:
    result: list[PulseTimelineSegment] = []
    cursor = 0
    for start, stop in highs:
        if start < cursor or stop <= start:
            raise ValueError("compiled digital intervals overlap or reverse")
        if start > cursor:
            budget.take()
            result.append(PulseTimelineSegment(cursor, start, 0, 0))
        budget.take()
        result.append(PulseTimelineSegment(start, stop, 1, 1))
        cursor = stop
    if cursor < duration:
        budget.take()
        result.append(PulseTimelineSegment(cursor, duration, 0, 0))
    return tuple(result)


def _bus_segments(
    actions: tuple[tuple[int, int, str, int], ...] | list[tuple[int, int, str, int]],
    *,
    safe_value: int,
    duration: int,
    budget: _ItemBudget,
) -> tuple[PulseTimelineSegment, ...]:
    result: list[PulseTimelineSegment] = []
    cursor = 0
    current = safe_value
    for start, stop, mode, target in actions:
        if start < cursor:
            raise ValueError("compiled DAC actions overlap")
        if start > cursor:
            budget.take()
            result.append(
                PulseTimelineSegment(
                    cursor,
                    start,
                    current - safe_value,
                    current - safe_value,
                )
            )
        if mode == "edge":
            current = target
            cursor = start
            continue
        if stop <= start:
            raise ValueError("compiled DAC ramp has no duration")
        budget.take()
        result.append(
            PulseTimelineSegment(
                start,
                stop,
                current - safe_value,
                target - safe_value,
            )
        )
        current = target
        cursor = stop
    if cursor < duration:
        budget.take()
        result.append(
            PulseTimelineSegment(
                cursor,
                duration,
                current - safe_value,
                current - safe_value,
            )
        )
    return tuple(result)


def _period_annotations(
    document: PulseDocument,
    budget: _ItemBudget,
) -> tuple[PulseTimelineAnnotation, ...]:
    periods = document.periods
    repeat = document.repeat
    if repeat is None:
        expanded: tuple[PulsePeriod, ...] = periods
        repeat_bounds = None
        budget.take(len(expanded))
    else:
        start = next(
            index for index, period in enumerate(periods)
            if period.period_id == repeat.start_period_id
        )
        end = next(
            index for index, period in enumerate(periods)
            if period.period_id == repeat.end_period_id
        )
        expanded_count = start + (end - start + 1) * repeat.count + len(periods) - end - 1
        budget.take(expanded_count)
        expanded = (
            periods[:start]
            + periods[start : end + 1] * repeat.count
            + periods[end + 1 :]
        )
        prefix_ticks = sum(
            _exact_ticks(item.duration, item.unit, document.time_step_ns, "period duration")
            for item in periods[:start]
        )
        body_ticks = sum(
            _exact_ticks(item.duration, item.unit, document.time_step_ns, "period duration")
            for item in periods[start : end + 1]
        )
        repeat_bounds = (prefix_ticks, prefix_ticks + repeat.count * body_ticks, repeat.count)
    annotations: list[PulseTimelineAnnotation] = []
    cursor = 0
    for period in expanded:
        stop = cursor + _exact_ticks(
            period.duration,
            period.unit,
            document.time_step_ns,
            f"period {period.period_id} duration",
        )
        annotations.append(
            PulseTimelineAnnotation(
                "period",
                cursor,
                stop,
                period.name or period.period_id,
            )
        )
        cursor = stop
    if repeat_bounds is not None:
        budget.take()
        start, stop, count = repeat_bounds
        annotations.append(
            PulseTimelineAnnotation("repeat", start, stop, f"repeat ×{count}")
        )
    return tuple(annotations)


__all__ = [
    "PulseTimelineAnnotation",
    "PulseTimelineDocument",
    "PulseTimelineRow",
    "PulseTimelineSegment",
    "build_pulse_timeline",
]
