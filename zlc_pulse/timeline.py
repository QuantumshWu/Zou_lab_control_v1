"""Exact renderer-neutral projection of one authored pulse document.

The timeline is a read-only view of :class:`PulseDocument`, never a second
authoring or persistence model.  It deliberately keeps the authored period
table compact: a :class:`RepeatRegion` is an annotation over the original
period span, while the rows describe one authored pass.  Physical output
delays and the DAC engine's integer staircase remain pulse-owned facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from zlc_storage import (
    canonical_digest,
    canonical_text,
    nonnegative_integer,
    positive_integer,
    sha256_text,
)

from .artifact import CompiledPulseArtifact, PulseExecutionForm
from .document import FIELD_DAC, FIELD_DURATION, PulseDocument, _exact_ticks
from .ir import TargetIR
from .physical import (
    iter_physical_digital_high_intervals,
    physical_digital_playback_terminal_tick,
)
from .target import PORT_DAC, PORT_DIGITAL


_ANNOTATION_KINDS = frozenset(("period", "repeat", "scan-duration", "scan-dac"))


@dataclass(frozen=True, slots=True)
class PulseTimelineSegment:
    """One constant half-open output interval ``[start_tick, stop_tick)``."""

    start_tick: int
    stop_tick: int
    start_value: int
    stop_value: int

    def __post_init__(self) -> None:
        start = nonnegative_integer(self.start_tick, "timeline segment start_tick")
        stop = positive_integer(self.stop_tick, "timeline segment stop_tick")
        if stop <= start:
            raise ValueError("timeline segment must have positive duration")
        for name in ("start_value", "stop_value"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.start_value != self.stop_value:
            raise ValueError("timeline segments must be constant; ramps are exact staircases")
        object.__setattr__(self, "start_tick", start)
        object.__setattr__(self, "stop_tick", stop)


@dataclass(frozen=True, slots=True)
class PulseTimelineRow:
    row_id: str
    label: str
    port_kind: str
    active: bool
    unit: str
    value_range: tuple[int, int]
    segments: tuple[PulseTimelineSegment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", canonical_text(self.row_id, "timeline row_id"))
        object.__setattr__(self, "label", canonical_text(self.label, "timeline row label"))
        kind = canonical_text(self.port_kind, "timeline row port_kind")
        if kind not in (PORT_DIGITAL, PORT_DAC):
            raise ValueError("timeline row port_kind must be digital or dac")
        object.__setattr__(self, "port_kind", kind)
        if type(self.active) is not bool:
            raise TypeError("timeline row active must be bool")
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
        if any(
            not value_range[0] <= value <= value_range[1]
            for segment in segments
            for value in (segment.start_value, segment.stop_value)
        ):
            raise ValueError("timeline row segment value is outside value_range")
        object.__setattr__(self, "value_range", value_range)
        object.__setattr__(self, "segments", segments)


@dataclass(frozen=True, slots=True)
class PulseTimelineAnnotation:
    """Typed period/repeat/scan overlay in authored timeline coordinates."""

    kind: str
    start_tick: int
    stop_tick: int
    label: str
    parameter_id: str | None = None
    number: int | None = None
    row_id: str | None = None
    value: int | float | None = None

    def __post_init__(self) -> None:
        kind = canonical_text(self.kind, "annotation kind")
        if kind not in _ANNOTATION_KINDS:
            raise ValueError(f"unsupported timeline annotation kind {kind!r}")
        object.__setattr__(self, "kind", kind)
        start = nonnegative_integer(self.start_tick, "annotation start_tick")
        stop = positive_integer(self.stop_tick, "annotation stop_tick")
        if stop <= start:
            raise ValueError("timeline annotation must have positive duration")
        object.__setattr__(self, "start_tick", start)
        object.__setattr__(self, "stop_tick", stop)
        object.__setattr__(self, "label", canonical_text(self.label, "annotation label"))

        parameter_id = self.parameter_id
        if parameter_id is not None:
            parameter_id = canonical_text(parameter_id, "annotation parameter_id")
        row_id = self.row_id
        if row_id is not None:
            row_id = canonical_text(row_id, "annotation row_id")
        number = self.number
        if number is not None:
            number = positive_integer(number, "annotation number")
        value = self.value
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("annotation value must be numeric or None")
            if not math.isfinite(float(value)):
                raise ValueError("annotation value must be finite")
            value = int(value) if isinstance(value, int) or float(value).is_integer() else float(value)

        if kind in ("period", "repeat"):
            if any(item is not None for item in (parameter_id, number, row_id, value)):
                raise ValueError(f"{kind} annotation cannot carry scan fields")
        elif kind == "scan-duration":
            if parameter_id is None or number is None or row_id is not None or value is not None:
                raise ValueError("scan-duration requires parameter_id/number only")
        else:
            if (
                parameter_id is None
                or number is None
                or row_id is None
                or isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise ValueError("scan-dac requires parameter_id/number/row_id/integer value")
        object.__setattr__(self, "parameter_id", parameter_id)
        object.__setattr__(self, "number", number)
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "value", value)


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
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", canonical_text(self.title, "timeline title"))
        object.__setattr__(
            self,
            "source_document_digest",
            sha256_text(self.source_document_digest, "source_document_digest"),
        )
        object.__setattr__(
            self,
            "target_ir_digest",
            sha256_text(self.target_ir_digest, "target_ir_digest"),
        )
        if isinstance(self.clock_hz, bool) or not isinstance(self.clock_hz, (int, float)):
            raise TypeError("clock_hz must be numeric")
        if not math.isfinite(float(self.clock_hz)) or float(self.clock_hz) <= 0:
            raise ValueError("clock_hz must be finite and positive")
        object.__setattr__(self, "clock_hz", float(self.clock_hz))
        logical = positive_integer(self.logical_duration_ticks, "logical_duration_ticks")
        duration = positive_integer(self.duration_ticks, "duration_ticks")
        if duration < logical:
            raise ValueError("physical timeline cannot end before authored execution")
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
        row_by_id = {item.row_id: item for item in rows}
        if len(row_by_id) != len(rows):
            raise ValueError("pulse timeline row_id values must be unique")
        if any(row.segments[-1].stop_tick != duration for row in rows):
            raise ValueError("every timeline row must cover the physical duration")
        if any(not isinstance(item, PulseTimelineAnnotation) for item in annotations):
            raise TypeError("annotations must contain PulseTimelineAnnotation values")
        if any(item.stop_tick > logical for item in annotations):
            raise ValueError("timeline annotation exceeds the authored period span")
        scan_numbers = [item.number for item in annotations if item.number is not None]
        if len(scan_numbers) != len(set(scan_numbers)):
            raise ValueError("scan annotation numbers must be unique")
        for item in annotations:
            if item.kind == "scan-dac":
                row = row_by_id.get(item.row_id)
                if row is None or row.port_kind != PORT_DAC:
                    raise ValueError("scan-dac annotation must reference one DAC row")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "_fingerprint", canonical_digest(_timeline_tree(self)))

    @property
    def fingerprint(self) -> str:
        """Digest of every value that can change the rendered pulse picture."""

        return self._fingerprint


def build_pulse_timeline(
    document: PulseDocument,
    artifact: CompiledPulseArtifact,
    *,
    reference_label: str,
) -> PulseTimelineDocument:
    """Project the one authored pass represented by a finite static compile.

    The supplied artifact proves that the exact source document compiled.  A
    repeat bracket is then removed only for the renderer-neutral authoring
    projection and compiled again through the same pulse compiler.  There is no
    expanded-preview mode and no frontend reconstruction of hardware behavior.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if artifact.source_document_digest != document.fingerprint:
        raise ValueError("compiled pulse artifact belongs to another PulseDocument")
    if artifact.execution_form not in (
        PulseExecutionForm.STATIC_ONCE,
        PulseExecutionForm.STATIC_REFERENCE_POINT,
    ):
        raise ValueError("timeline preview requires a finite static/reference compile")
    if artifact.target_ir.scan_points or artifact.target_ir.repeat_forever:
        raise ValueError("timeline preview requires a finite static/reference TargetIR")

    return build_authored_pulse_timeline(
        document,
        artifact.target_ir,
        execution_form=artifact.execution_form,
        reference_label=reference_label,
    )


def build_authored_pulse_timeline(
    document: PulseDocument,
    target_ir: TargetIR,
    *,
    execution_form: PulseExecutionForm,
    reference_label: str,
) -> PulseTimelineDocument:
    """Project an already-compiled logical TargetIR for authoring.

    Unlike :func:`build_pulse_timeline`, this boundary requires no wire image or
    deployed streamer geometry.  It is the Offline editor path: the pulse
    compiler remains authoritative for physical timing, while frozen hardware
    admission belongs exclusively to artifact packing and Run preflight.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(target_ir, TargetIR):
        raise TypeError("target_ir must be TargetIR")
    if not isinstance(execution_form, PulseExecutionForm):
        raise TypeError("execution_form must be PulseExecutionForm")
    if execution_form not in (
        PulseExecutionForm.STATIC_ONCE,
        PulseExecutionForm.STATIC_REFERENCE_POINT,
    ):
        raise ValueError("authored timeline requires a finite static/reference compile")
    if target_ir.scan_points or target_ir.repeat_forever:
        raise ValueError("authored timeline requires a finite static/reference TargetIR")
    if target_ir.target_abi_fingerprint != document.target.abi_fingerprint:
        raise ValueError("TargetIR belongs to another PulseTarget")

    target_ir = _authored_projection_ir(
        document,
        target_ir,
        execution_form,
    )
    logical_duration = _authored_period_starts(document)[-1]
    compiled_terminal = physical_digital_playback_terminal_tick(target_ir)
    if compiled_terminal != logical_duration:
        raise RuntimeError("authored timeline duration differs from the pulse compiler")

    high_by_output: dict[str, list[tuple[int, int]]] = {
        key: [] for key, _lane in target_ir.logical_digital_outputs
    }
    for interval in iter_physical_digital_high_intervals(target_ir):
        high_by_output[interval.output_key].append(
            (interval.start_tick, interval.stop_tick)
        )

    physical_duration = max(
        logical_duration + _max_output_delay(target_ir),
        *(stop for values in high_by_output.values() for _start, stop in values),
        _bus_projection_terminal(target_ir),
    )
    rows: list[PulseTimelineRow] = []
    segments_by_bus = {
        index: tuple(
            segment for segment in target_ir.bus_segments if segment.bus_index == index
        )
        for index in range(len(target_ir.bus_names))
    }
    for port in document.target.ports:
        if port.kind == PORT_DIGITAL:
            highs = high_by_output.get(port.key, ())
            rows.append(
                PulseTimelineRow(
                    row_id=port.key,
                    label=port.label,
                    port_kind=PORT_DIGITAL,
                    active=bool(highs),
                    unit="logic",
                    value_range=(0, 1),
                    segments=_digital_segments(highs, physical_duration),
                )
            )
        elif port.kind == PORT_DAC:
            assert port.bus_index is not None and port.signed_range is not None
            source_segments = segments_by_bus[port.bus_index]
            rows.append(
                PulseTimelineRow(
                    row_id=port.key,
                    label=port.label,
                    port_kind=PORT_DAC,
                    active=bool(source_segments),
                    unit="DAC code",
                    value_range=port.signed_range,
                    segments=_bus_segments(
                        target_ir,
                        port.bus_index,
                        safe_value=port.safe_value,
                        duration=physical_duration,
                    ),
                )
            )
    if not rows:
        raise ValueError("PulseDocument target has no digital or DAC rows")
    annotations = (*_period_annotations(document), *_scan_annotations(document))
    return PulseTimelineDocument(
        title=document.name,
        source_document_digest=document.fingerprint,
        target_ir_digest=target_ir.fingerprint,
        clock_hz=target_ir.clock_hz,
        logical_duration_ticks=logical_duration,
        duration_ticks=physical_duration,
        reference_label=reference_label,
        rows=tuple(rows),
        annotations=annotations,
    )


def _authored_projection_ir(
    document: PulseDocument,
    target_ir: TargetIR,
    execution_form: PulseExecutionForm,
) -> TargetIR:
    if document.repeat is None:
        return target_ir
    # Import at the projection seam so compiler.py remains independent of the
    # renderer-neutral value types exported by this module.
    from .compiler import compile_pulse_document

    authored = replace(document, repeat=None)
    return compile_pulse_document(
        authored,
        clock_hz=target_ir.clock_hz,
        execution_form=execution_form,
        live_target=document.target,
    )


def _authored_period_starts(document: PulseDocument) -> tuple[int, ...]:
    starts = [0]
    for period in document.periods:
        starts.append(
            starts[-1]
            + _exact_ticks(
                period.duration,
                period.unit,
                document.time_step_ns,
                f"period {period.period_id} duration",
            )
        )
    return tuple(starts)


def _digital_segments(
    highs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    duration: int,
) -> tuple[PulseTimelineSegment, ...]:
    result: list[PulseTimelineSegment] = []
    cursor = 0
    for start, stop in highs:
        if start < cursor or stop <= start or stop > duration:
            raise ValueError("compiled digital intervals overlap or exceed the timeline")
        if start > cursor:
            result.append(PulseTimelineSegment(cursor, start, 0, 0))
        result.append(PulseTimelineSegment(start, stop, 1, 1))
        cursor = stop
    if cursor < duration:
        result.append(PulseTimelineSegment(cursor, duration, 0, 0))
    return tuple(result)


def _bus_projection_terminal(ir: TargetIR) -> int:
    delays = {item.bus_index: item.delay_ticks for item in ir.bus_delays}
    terminal = 0
    for segment in ir.bus_segments:
        delay = delays.get(segment.bus_index, 0)
        source = segment.stop_tick if segment.mode == "ramp" else segment.start_tick
        terminal = max(terminal, source + delay)
    return terminal


def _max_output_delay(ir: TargetIR) -> int:
    return max(
        (0, *ir.channel_delays, *(item.delay_ticks for item in ir.bus_delays))
    )


def _bus_segments(
    ir: TargetIR,
    bus_index: int,
    *,
    safe_value: int,
    duration: int,
) -> tuple[PulseTimelineSegment, ...]:
    events = _bus_value_events(ir, bus_index, safe_value=safe_value)
    normalized: list[tuple[int, int]] = []
    for tick, value in events:
        if tick < 0:
            raise ValueError("compiled DAC event precedes tick zero")
        if tick >= duration:
            continue
        if normalized and normalized[-1][0] == tick:
            normalized[-1] = (tick, value)
        elif not normalized or normalized[-1][1] != value:
            normalized.append((tick, value))
    if not normalized or normalized[0][0] != 0:
        normalized.insert(0, (0, safe_value))

    result: list[PulseTimelineSegment] = []
    for index, (start, encoded) in enumerate(normalized):
        stop = normalized[index + 1][0] if index + 1 < len(normalized) else duration
        if stop > start:
            value = encoded - safe_value
            result.append(PulseTimelineSegment(start, stop, value, value))
    if not result:
        result.append(PulseTimelineSegment(0, duration, 0, 0))
    return tuple(result)


def _bus_value_events(
    ir: TargetIR,
    bus_index: int,
    *,
    safe_value: int,
) -> tuple[tuple[int, int], ...]:
    segments = sorted(
        (item for item in ir.bus_segments if item.bus_index == bus_index),
        key=lambda item: (item.start_tick, item.stop_tick),
    )
    if any(item.value_select or item.stop_value_select for item in segments):
        raise ValueError("authored timeline contains unresolved DAC selectors")
    delay = next(
        (item.delay_ticks for item in ir.bus_delays if item.bus_index == bus_index),
        0,
    )
    events: list[tuple[int, int]] = [(0, safe_value)]
    carried = safe_value
    for segment in segments:
        visible_start = segment.start_tick + delay + (0 if segment.start_tick == 0 else 1)
        target = segment.stop_value
        if segment.mode == "edge" or segment.stop_tick <= segment.start_tick:
            events.append((visible_start, target))
            carried = target
            continue
        events.append((visible_start, carried))
        span = segment.stop_tick - segment.start_tick
        delta = abs(target - carried)
        direction = 1 if target >= carried else -1
        if delta <= span:
            for move in range(1, delta + 1):
                elapsed = (move * span + delta - 1) // delta
                events.append(
                    (
                        segment.start_tick + delay + elapsed + 1,
                        carried + direction * move,
                    )
                )
        else:
            for elapsed in range(1, span + 1):
                move = (elapsed * delta) // span
                events.append(
                    (
                        segment.start_tick + delay + elapsed + 1,
                        carried + direction * move,
                    )
                )
        carried = target
    # The live engine freezes a ramp at the shifted frame boundary.  Another
    # output's longer delay tail may keep the shared preview time axis open,
    # but it must not make this bus take extra staircase steps after its own
    # frame ended.
    frame_end = physical_digital_playback_terminal_tick(ir) + delay
    finite = [item for item in events if item[0] <= frame_end]
    # Finite DONE clears the live bus and its delayed descriptor stream drains
    # that SAFE hold one registered tick after the shifted terminal.  Usually
    # this is exactly the right edge of the picture; if another output owns a
    # longer tail, the bus must visibly remain SAFE across that extra span.
    finite.append((frame_end + 1, safe_value))
    return tuple(finite)


def _period_annotations(document: PulseDocument) -> tuple[PulseTimelineAnnotation, ...]:
    starts = _authored_period_starts(document)
    annotations = [
        PulseTimelineAnnotation(
            kind="period",
            start_tick=starts[index],
            stop_tick=starts[index + 1],
            label=period.name or period.period_id,
        )
        for index, period in enumerate(document.periods)
    ]
    repeat = document.repeat
    if repeat is not None:
        by_id = {period.period_id: index for index, period in enumerate(document.periods)}
        start = by_id[repeat.start_period_id]
        end = by_id[repeat.end_period_id]
        annotations.append(
            PulseTimelineAnnotation(
                kind="repeat",
                start_tick=starts[start],
                stop_tick=starts[end + 1],
                label=f"×{repeat.count}",
            )
        )
    return tuple(annotations)


def _scan_annotations(document: PulseDocument) -> tuple[PulseTimelineAnnotation, ...]:
    if not document.scan_parameters:
        return ()
    starts = _authored_period_starts(document)
    period_index = {
        period.period_id: index for index, period in enumerate(document.periods)
    }
    by_id = document.scan_parameter_by_id
    ordered_ids = (
        document.scan_table.columns
        if document.scan_table is not None
        else tuple(parameter.parameter_id for parameter in document.scan_parameters)
    )
    annotations: list[PulseTimelineAnnotation] = []
    for number, parameter_id in enumerate(ordered_ids, start=1):
        parameter = by_id[parameter_id]
        reference = parameter.field
        index = period_index[reference.period_id]
        if reference.kind == FIELD_DURATION:
            annotations.append(
                PulseTimelineAnnotation(
                    kind="scan-duration",
                    start_tick=starts[index],
                    stop_tick=starts[index + 1],
                    label=parameter.label or parameter.parameter_id,
                    parameter_id=parameter.parameter_id,
                    number=number,
                )
            )
        elif reference.kind == FIELD_DAC:
            value, _unit = document.field_value(reference)
            annotations.append(
                PulseTimelineAnnotation(
                    kind="scan-dac",
                    start_tick=starts[index],
                    stop_tick=starts[index + 1],
                    label=parameter.label or parameter.parameter_id,
                    parameter_id=parameter.parameter_id,
                    number=number,
                    row_id=reference.port,
                    value=int(value),
                )
            )
        else:
            raise ValueError("timeline scan overlay supports duration and DAC fields only")
    return tuple(annotations)


def _timeline_tree(value: PulseTimelineDocument) -> dict[str, object]:
    return {
        "kind": "pulse-timeline",
        "title": value.title,
        "source_document_digest": value.source_document_digest,
        "target_ir_digest": value.target_ir_digest,
        "clock_hz": value.clock_hz,
        "logical_duration_ticks": value.logical_duration_ticks,
        "duration_ticks": value.duration_ticks,
        "reference_label": value.reference_label,
        "rows": [
            {
                "row_id": row.row_id,
                "label": row.label,
                "port_kind": row.port_kind,
                "active": row.active,
                "unit": row.unit,
                "value_range": list(row.value_range),
                "segments": [
                    {
                        "start_tick": segment.start_tick,
                        "stop_tick": segment.stop_tick,
                        "start_value": segment.start_value,
                        "stop_value": segment.stop_value,
                    }
                    for segment in row.segments
                ],
            }
            for row in value.rows
        ],
        "annotations": [
            {
                "kind": annotation.kind,
                "start_tick": annotation.start_tick,
                "stop_tick": annotation.stop_tick,
                "label": annotation.label,
                "parameter_id": annotation.parameter_id,
                "number": annotation.number,
                "row_id": annotation.row_id,
                "value": annotation.value,
            }
            for annotation in value.annotations
        ],
    }


__all__ = [
    "PulseTimelineAnnotation",
    "PulseTimelineDocument",
    "PulseTimelineRow",
    "PulseTimelineSegment",
    "build_authored_pulse_timeline",
    "build_pulse_timeline",
]
