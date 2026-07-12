"""Native compiler from current :class:`PulseDocument` values to FPGA TargetIR.

This module is the sole semantic compiler.  It consumes only pulse-owned current
values; historical table objects, neutral-atom runtime payloads, GUI state, and
device services are deliberately outside the dependency graph.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import replace
from functools import lru_cache
from numbers import Number
from typing import Mapping, Sequence

from fpga.pulse_streamer.host.image import StreamerParams

from .artifact import CompiledPulseArtifact, PulseExecutionForm
from .document import AnalogBusStep, PulseDocument, PulsePeriod
from .fpga import pack_target_ir
from .ir import TargetBusDelay, TargetBusSegment, TargetIR
from .schedule import build_digital_trigger_schedules
from .target import PORT_CLOCK, PORT_DAC, PORT_DIGITAL, PulsePortSpec, PulseTarget


COMPILER_ID = "zlc-pulse-native"
COMPILER_VERSION = "1"
_UNITS_TO_NS = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
_SLOT_NAME = re.compile(r"s([0-9]+)\Z")


def compile_pulse_document(
    document: PulseDocument,
    *,
    clock_hz: float,
    execution_form: PulseExecutionForm,
    live_target: PulseTarget | None = None,
    params: StreamerParams | None = None,
) -> TargetIR:
    """Compile a current pulse document into one immutable target program.

    ``execution_form`` makes scan/reference/cyclic intent explicit.  A scan is
    never silently collapsed to its first point, and a finite formal scan never
    inherits the historical cursor-wrap ``scan_repeats`` behaviour.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(execution_form, PulseExecutionForm):
        raise TypeError("execution_form must be PulseExecutionForm")
    frequency = _positive_float(clock_hz, "clock_hz")
    geometry = params or StreamerParams()
    if live_target is not None:
        if not isinstance(live_target, PulseTarget):
            raise TypeError("live_target must be PulseTarget")
        if live_target.abi_fingerprint != document.target.abi_fingerprint:
            raise ValueError("pulse document target ABI differs from live target")
    _validate_target_geometry(document.target, geometry)

    if execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        if not document.scan_slots or not document.scan_table:
            raise ValueError("AUTONOMOUS_SCAN_ONCE requires scan slots and points")
        if document.scan_repeats != 0:
            raise ValueError(
                "finite autonomous scan expands repeats in its frozen plan; "
                "cursor-wrap scan_repeats is forbidden"
            )
        return _compile_scan(
            document,
            clock_hz=frequency,
            coeff_frac_bits=int(geometry.coeff_frac_bits),
        )

    reference = execution_form is PulseExecutionForm.STATIC_REFERENCE_POINT
    if (document.scan_slots or document.scan_table) and not reference:
        raise ValueError("static execution cannot silently ignore a scan definition")
    if reference and not document.scan_slots:
        raise ValueError("STATIC_REFERENCE_POINT requires at least one scan slot")
    return _compile_static(
        document,
        clock_hz=frequency,
        repeat_forever=execution_form is PulseExecutionForm.CONTINUOUS_MONITOR,
    )


def compile_pulse_artifact(
    document: PulseDocument,
    *,
    clock_hz: float,
    execution_form: PulseExecutionForm,
    trigger_channels: tuple[str, ...] = (),
    live_target: PulseTarget | None = None,
    params: StreamerParams | None = None,
) -> CompiledPulseArtifact:
    """Compile source, target program, wire image, and trigger truth together."""

    channels = tuple(trigger_channels)
    if len(channels) != len(set(channels)):
        raise ValueError("trigger_channels must be unique")
    if execution_form is PulseExecutionForm.CONTINUOUS_MONITOR and channels:
        raise ValueError("continuous monitor cannot publish a finite trigger schedule")
    lane_owners = {
        lane: port for port in document.target.ports for lane in port.lanes
    }
    for channel in channels:
        owner = lane_owners.get(channel)
        if owner is None:
            raise ValueError(f"unknown physical trigger lane {channel!r}")
        if owner.kind != PORT_DIGITAL:
            raise ValueError(
                f"trigger lane {channel!r} belongs to {owner.kind!r}, not a digital port"
            )
    geometry = params or StreamerParams()
    target_ir = compile_pulse_document(
        document,
        clock_hz=clock_hz,
        execution_form=execution_form,
        live_target=live_target,
        params=geometry,
    )
    schedules = (
        ()
        if execution_form is PulseExecutionForm.CONTINUOUS_MONITOR
        else build_digital_trigger_schedules(target_ir, channels)
    )
    return CompiledPulseArtifact(
        source_document_digest=document.fingerprint,
        compiler_id=COMPILER_ID,
        compiler_version=COMPILER_VERSION,
        execution_form=execution_form,
        target_ir=target_ir,
        wire_image=pack_target_ir(target_ir, geometry),
        trigger_schedules=schedules,
    )


def _compile_static(
    document: PulseDocument,
    *,
    clock_hz: float,
    repeat_forever: bool,
) -> TargetIR:
    step_ns = 1e9 / clock_hz
    slots = _reference_slots(document)
    work = document
    has_bracket = _has_bracket(work)
    if has_bracket and _has_any_delay(work, slots, step_ns):
        work = _unroll_bracket(work)
        has_bracket = False
        slots = _reference_slots(work)

    starts = _period_starts(work, slots, step_ns)
    bus_names, bus_segments, raw_bus_delays = _bus_segments(
        work,
        starts=starts,
        slots=slots,
        step_ns=step_ns,
        slot_vars=(),
        coeff_frac_bits=0,
    )
    if has_bracket and bus_segments and work.repeat_count > 1:
        raise ValueError("DAC bus segments do not support a compact finite inner repeat bracket")
    driven_bus_raw = {
        segment.bus_index: int(raw_bus_delays.get(segment.bus_index, 0))
        for segment in bus_segments
    }
    ticks, masks, channel_delays, shifted_bus_delays = _static_edges(
        work,
        starts=starts,
        slots=slots,
        step_ns=step_ns,
        extra_raw_bus_delays=driven_bus_raw,
    )

    if has_bracket:
        assert work.repeat_start is not None and work.repeat_end is not None
        loop_start_tick = starts[work.repeat_start]
        loop_start_index = ticks.index(loop_start_tick)
        loop_end_tick = starts[work.repeat_end + 1]
        loop_count = work.repeat_count
    else:
        loop_start_index = 0
        loop_end_tick = starts[-1]
        loop_count = 1
    effective_ticks = starts[-1]
    if has_bracket:
        assert work.repeat_start is not None and work.repeat_end is not None
        effective_ticks += (work.repeat_count - 1) * (
            starts[work.repeat_end + 1] - starts[work.repeat_start]
        )
    clk_enable = _clock_enable(work.target)
    if clk_enable:
        masks = [mask & ~clk_enable for mask in masks]
    return TargetIR(
        clock_hz=clock_hz,
        target_abi_fingerprint=work.target.abi_fingerprint,
        channels=work.target.raw_lanes,
        ticks=tuple(ticks),
        masks=tuple(masks),
        duration_seconds=effective_ticks / clock_hz,
        repeat_forever=repeat_forever,
        loop_start_index=loop_start_index,
        loop_end_tick=loop_end_tick,
        loop_count=loop_count,
        slot_kinds=(),
        loop_end_slot_coeffs=(),
        tick_slot_coeffs=tuple(() for _ in ticks),
        scan_points=(),
        scan_point_durations=(),
        scan_coeff_frac_bits=0,
        scan_repeats=0,
        bus_names=tuple(bus_names),
        bus_segments=tuple(bus_segments),
        bus_delays=tuple(
            TargetBusDelay(index, delay)
            for index, delay in sorted(shifted_bus_delays.items())
            if delay
        ),
        channel_delays=tuple(channel_delays),
        clk_enable=clk_enable,
    )


def _compile_scan(
    document: PulseDocument,
    *,
    clock_hz: float,
    coeff_frac_bits: int,
) -> TargetIR:
    step_ns = 1e9 / clock_hz
    work = document
    if _has_bracket(work) and _has_any_delay(work, _reference_slots(work), step_ns):
        work = _unroll_bracket(work)
    table = _snap_scan_table(work, step_ns)
    scan_points = _wire_scan_points(work, table, step_ns)
    slot_vars = tuple(f"s{index}" for index in range(len(work.scan_slots)))
    reference_slots = _reference_slots(work, row=table[0])
    reference_starts = _period_starts(work, reference_slots, step_ns)
    affine_starts = _affine_period_starts(
        work,
        slot_vars=slot_vars,
        step_ns=step_ns,
        coeff_frac_bits=coeff_frac_bits,
    )
    bus_names, bus_segments, raw_bus_delays = _bus_segments(
        work,
        starts=reference_starts,
        slots=reference_slots,
        step_ns=step_ns,
        slot_vars=slot_vars,
        coeff_frac_bits=coeff_frac_bits,
        affine_starts=affine_starts,
    )
    bus_members = {
        lane
        for port in work.target.ports
        if port.kind == PORT_DAC
        for lane in port.lanes
    }
    rows = _affine_edge_rows(
        work,
        affine_starts=affine_starts,
        scan_points=scan_points,
        coeff_frac_bits=coeff_frac_bits,
        excluded=bus_members,
    )
    ticks = [row[0] for row in rows]
    masks = [row[1] for row in rows]
    tick_coeffs = [row[2] for row in rows]
    clk_enable = _clock_enable(work.target)
    if clk_enable:
        masks = [mask & ~clk_enable for mask in masks]

    if _has_bracket(work) and work.repeat_count > 1:
        assert work.repeat_start is not None and work.repeat_end is not None
        loop_expr = affine_starts[work.repeat_start]
        loop_start_index = next(
            index
            for index, row in enumerate(rows)
            if (row[0], row[2]) == loop_expr
        )
        loop_end_tick, loop_end_coeffs = affine_starts[work.repeat_end + 1]
        loop_count = work.repeat_count
    else:
        loop_start_index = 0
        loop_end_tick, loop_end_coeffs = rows[-1][0], rows[-1][2]
        loop_count = 1

    raw_ttl_delays: dict[str, int] = {}
    clock_lanes = {
        port.lanes[0] for port in work.target.ports if port.kind == PORT_CLOCK
    }
    for lane_index, lane in enumerate(work.target.raw_lanes):
        if lane in bus_members or lane in clock_lanes:
            continue
        if not any(period.states[lane_index] for period in work.periods):
            continue
        raw = dict(work.delays).get(lane, 0)
        if isinstance(raw, str):
            _base, coeffs = _affine_coeffs(
                raw,
                unit=dict(work.delay_units).get(lane, "ns"),
                slot_vars=slot_vars,
                step_ns=step_ns,
                coeff_frac_bits=coeff_frac_bits,
            )
            if any(coeffs):
                raise ValueError(
                    f"channel {lane!r} delay references a scan slot; delays are fixed"
                )
        raw_ttl_delays[lane] = _delay_ticks(work, lane, reference_slots, step_ns)
    driven_bus_indices = {segment.bus_index for segment in bus_segments}
    bus_raw = {
        index: int(raw_bus_delays.get(index, 0)) for index in driven_bus_indices
    }
    ttl_shifted, bus_shifted, _global = _fold_global_delay_shift(
        raw_ttl_delays, bus_raw
    )
    lane_bits = {lane: index for index, lane in enumerate(work.target.raw_lanes)}
    channel_delays = [0] * len(work.target.raw_lanes)
    for lane, delay in ttl_shifted.items():
        channel_delays[lane_bits[lane]] = delay

    point_durations = tuple(
        _apply_affine(ticks[-1], tick_coeffs[-1], point, coeff_frac_bits)
        / clock_hz
        for point in scan_points
    )
    return TargetIR(
        clock_hz=clock_hz,
        target_abi_fingerprint=work.target.abi_fingerprint,
        channels=work.target.raw_lanes,
        ticks=tuple(ticks),
        masks=tuple(masks),
        duration_seconds=sum(point_durations),
        repeat_forever=False,
        loop_start_index=loop_start_index,
        loop_end_tick=loop_end_tick,
        loop_count=loop_count,
        slot_kinds=tuple(slot.kind for slot in work.scan_slots),
        loop_end_slot_coeffs=tuple(loop_end_coeffs),
        tick_slot_coeffs=tuple(tuple(row) for row in tick_coeffs),
        scan_points=tuple(tuple(point) for point in scan_points),
        scan_point_durations=point_durations,
        scan_coeff_frac_bits=coeff_frac_bits,
        scan_repeats=0,
        bus_names=tuple(bus_names),
        bus_segments=tuple(bus_segments),
        bus_delays=tuple(
            TargetBusDelay(index, delay)
            for index, delay in sorted(bus_shifted.items())
            if delay
        ),
        channel_delays=tuple(channel_delays),
        clk_enable=clk_enable,
    )


def _static_edges(
    document: PulseDocument,
    *,
    starts: Sequence[int],
    slots: Mapping[str, float],
    step_ns: float,
    extra_raw_bus_delays: Mapping[int, int],
) -> tuple[list[int], list[int], list[int], dict[int, int]]:
    bus_members = {
        lane
        for port in document.target.ports
        if port.kind == PORT_DAC
        for lane in port.lanes
    }
    clock_lanes = {
        port.lanes[0] for port in document.target.ports if port.kind == PORT_CLOCK
    }
    intervals: dict[str, list[tuple[int, int]]] = {}
    raw_delays: dict[str, int] = {}
    for lane_index, lane in enumerate(document.target.raw_lanes):
        if lane in bus_members or lane in clock_lanes:
            continue
        active: int | None = None
        lane_intervals: list[tuple[int, int]] = []
        for period_index, period in enumerate(document.periods):
            state = period.states[lane_index]
            if state and active is None:
                active = starts[period_index]
            elif not state and active is not None:
                lane_intervals.append((active, starts[period_index]))
                active = None
        if active is not None:
            lane_intervals.append((active, starts[-1]))
        if lane_intervals:
            intervals[lane] = lane_intervals
            raw_delays[lane] = _delay_ticks(document, lane, slots, step_ns)
    shifted, shifted_buses, _global = _fold_global_delay_shift(
        raw_delays, extra_raw_bus_delays
    )
    events: dict[int, list[tuple[str, int]]] = {int(tick): [] for tick in starts}
    for lane, lane_intervals in intervals.items():
        for begin, end in lane_intervals:
            events.setdefault(begin, []).append((lane, 1))
            events.setdefault(end, []).append((lane, 0))
    bit = {lane: index for index, lane in enumerate(document.target.raw_lanes)}
    ticks: list[int] = []
    masks: list[int] = []
    current = 0
    for tick in sorted(events):
        for lane, value in events[tick]:
            current = (current | (1 << bit[lane])) if value else (current & ~(1 << bit[lane]))
        ticks.append(tick)
        masks.append(current)
    if ticks[0] != 0:
        ticks.insert(0, 0)
        masks.insert(0, 0)
    if ticks[-1] != starts[-1]:
        ticks.append(starts[-1])
        masks.append(0)
    else:
        masks[-1] = 0
    channel_delays = [0] * len(document.target.raw_lanes)
    for lane, delay in shifted.items():
        channel_delays[bit[lane]] = delay
    return ticks, masks, channel_delays, shifted_buses


def _affine_edge_rows(
    document: PulseDocument,
    *,
    affine_starts: Sequence[tuple[int, tuple[int, ...]]],
    scan_points: Sequence[Sequence[int]],
    coeff_frac_bits: int,
    excluded: set[str],
) -> list[tuple[int, int, tuple[int, ...]]]:
    events: dict[
        tuple[int, tuple[int, ...]], list[tuple[str | None, int | None]]
    ] = {}
    for lane_index, lane in enumerate(document.target.raw_lanes):
        if lane in excluded:
            continue
        active: tuple[int, tuple[int, ...]] | None = None
        for period_index, period in enumerate(document.periods):
            state = period.states[lane_index]
            if state and active is None:
                active = affine_starts[period_index]
            elif not state and active is not None:
                events.setdefault(active, []).append((lane, 1))
                events.setdefault(affine_starts[period_index], []).append((lane, 0))
                active = None
        if active is not None:
            events.setdefault(active, []).append((lane, 1))
            events.setdefault(affine_starts[-1], []).append((lane, 0))
    events.setdefault((0, tuple(0 for _ in document.scan_slots)), []).append((None, None))
    events.setdefault(affine_starts[-1], []).append((None, None))
    if _has_bracket(document) and document.repeat_count > 1:
        assert document.repeat_start is not None
        events.setdefault(affine_starts[document.repeat_start], []).append((None, None))

    point0 = scan_points[0]
    ordered = sorted(
        events,
        key=lambda expr: (
            _apply_affine(expr[0], expr[1], point0, coeff_frac_bits),
            not any(lane is not None for lane, _value in events[expr]),
            expr,
        ),
    )
    reference_ticks = [
        _apply_affine(expr[0], expr[1], point0, coeff_frac_bits) for expr in ordered
    ]
    if any(right <= left for left, right in zip(reference_ticks, reference_ticks[1:])):
        raise ValueError("scan moves or collides distinct edge rows at its reference point")
    bit = {lane: index for index, lane in enumerate(document.target.raw_lanes)}
    current = 0
    rows: list[tuple[int, int, tuple[int, ...]]] = []
    for expr in ordered:
        for lane, value in events[expr]:
            if lane is None or value is None:
                continue
            current = (current | (1 << bit[lane])) if value else (current & ~(1 << bit[lane]))
        rows.append((expr[0], current, expr[1]))
    if rows[-1][1] != 0:
        raise ValueError("hardware scan template does not return every digital lane to zero")
    for point_index, point in enumerate(scan_points):
        effective = [
            _apply_affine(base, coeffs, point, coeff_frac_bits)
            for base, _mask, coeffs in rows
        ]
        if effective[0] != 0 or any(
            right <= left for left, right in zip(effective, effective[1:])
        ):
            raise ValueError(
                f"scan point {point_index} produces non-increasing effective edge ticks"
            )
    return rows


def _bus_segments(
    document: PulseDocument,
    *,
    starts: Sequence[int],
    slots: Mapping[str, float],
    step_ns: float,
    slot_vars: Sequence[str],
    coeff_frac_bits: int,
    affine_starts: Sequence[tuple[int, tuple[int, ...]]] | None = None,
) -> tuple[list[str], list[TargetBusSegment], dict[int, int]]:
    programs = dict(document.analog_bus_programs)
    bus_ports = sorted(
        (port for port in document.target.ports if port.kind == PORT_DAC),
        key=lambda port: int(port.bus_index),
    )
    names = [port.key for port in bus_ports]
    segments: list[TargetBusSegment] = []
    bus_delays: dict[int, int] = {}
    zero_coeffs = tuple(0 for _ in slot_vars)
    for port in bus_ports:
        assert port.bus_index is not None and port.signed_range is not None
        plan = programs.get(
            port.key,
            tuple(AnalogBusStep("hold", None) for _ in document.periods),
        )
        lane_delays = {
            _delay_ticks(document, lane, slots, step_ns) for lane in port.lanes
        }
        if len(lane_delays) != 1:
            raise ValueError("all raw lanes in one DAC bus must share the same delay")
        delay = next(iter(lane_delays))
        if delay:
            bus_delays[port.bus_index] = delay
        zero_code = -port.signed_range[0]
        code_max = (1 << port.width) - 1
        for period_index, step in enumerate(plan):
            if step.mode == "hold":
                continue
            selector = _slot_reference(step.value, slot_vars)
            if selector is None:
                value = step.value
                unresolved = _slot_reference(value, tuple(f"s{i}" for i in range(len(document.scan_slots))))
                if unresolved is not None:
                    key = f"s{unresolved}"
                    if key not in slots:
                        raise ValueError(
                            f"DAC bus {port.key!r} references unresolved scan slot {key!r}"
                        )
                    value = slots[key]
                code = max(0, min(code_max, int(round(float(value))) + zero_code))
                value_select = 0
            else:
                code = 0
                value_select = selector + 1
            if affine_starts is None:
                start_tick = starts[period_index]
                start_coeffs = zero_coeffs
                stop_tick = starts[period_index + 1]
                stop_coeffs = zero_coeffs
            else:
                start_tick, start_coeffs = affine_starts[period_index]
                stop_tick, stop_coeffs = affine_starts[period_index + 1]
            if step.mode == "edge":
                segments.append(
                    TargetBusSegment(
                        port.bus_index,
                        port.key,
                        start_tick,
                        start_tick,
                        code,
                        code,
                        "edge",
                        value_select,
                        value_select,
                        tuple(start_coeffs),
                        tuple(start_coeffs),
                    )
                )
            else:
                segments.append(
                    TargetBusSegment(
                        port.bus_index,
                        port.key,
                        start_tick,
                        stop_tick,
                        0,
                        code,
                        "ramp",
                        0,
                        value_select,
                        tuple(start_coeffs),
                        tuple(stop_coeffs),
                    )
                )
    return names, segments, bus_delays


def _period_starts(
    document: PulseDocument, slots: Mapping[str, float], step_ns: float
) -> list[int]:
    starts = [0]
    for period in document.periods:
        value_ns = _eval_expr(period.duration, slots) * _UNITS_TO_NS[period.unit]
        starts.append(starts[-1] + max(1, _round_ties_away(value_ns / step_ns)))
    return starts


def _affine_period_starts(
    document: PulseDocument,
    *,
    slot_vars: Sequence[str],
    step_ns: float,
    coeff_frac_bits: int,
) -> list[tuple[int, tuple[int, ...]]]:
    starts = [(0, tuple(0 for _ in slot_vars))]
    for period in document.periods:
        term = _affine_coeffs(
            period.duration,
            unit=period.unit,
            slot_vars=slot_vars,
            step_ns=step_ns,
            coeff_frac_bits=coeff_frac_bits,
        )
        starts.append(
            (
                starts[-1][0] + term[0],
                tuple(a + b for a, b in zip(starts[-1][1], term[1])),
            )
        )
    return starts


def _snap_scan_table(document: PulseDocument, step_ns: float) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in document.scan_table:
        if len(row) != len(document.scan_slots):
            raise ValueError("scan point width differs from scan slot count")
        snapped: list[float] = []
        for slot, value in zip(document.scan_slots, row):
            if slot.kind == "duration":
                factor = _UNITS_TO_NS[slot.unit]
                ticks = max(1, _round_ties_away(float(value) * factor / step_ns))
                snapped.append(ticks * step_ns / factor)
            else:
                port = _dac_port_for_slot(document.target, slot.target)
                assert port.signed_range is not None
                snapped.append(
                    float(max(port.signed_range[0], min(port.signed_range[1], round(float(value)))))
                )
        rows.append(snapped)
    return rows


def _wire_scan_points(
    document: PulseDocument, rows: Sequence[Sequence[float]], step_ns: float
) -> list[list[int]]:
    points: list[list[int]] = []
    for row in rows:
        point: list[int] = []
        for slot, value in zip(document.scan_slots, row):
            if slot.kind == "duration":
                point.append(
                    _round_ties_away(float(value) * _UNITS_TO_NS[slot.unit] / step_ns)
                )
            else:
                port = _dac_port_for_slot(document.target, slot.target)
                assert port.signed_range is not None
                point.append(int(round(float(value))) - port.signed_range[0])
        points.append(point)
    return points


def _reference_slots(
    document: PulseDocument, *, row: Sequence[float] | None = None
) -> dict[str, float]:
    values = row if row is not None else tuple(slot.nominal for slot in document.scan_slots)
    return {
        f"s{index}": (
            float(value) * _UNITS_TO_NS[slot.unit]
            if slot.kind == "duration"
            else float(value)
        )
        for index, (slot, value) in enumerate(zip(document.scan_slots, values))
    }


def _delay_ticks(
    document: PulseDocument,
    lane: str,
    slots: Mapping[str, float],
    step_ns: float,
) -> int:
    raw = dict(document.delays).get(lane, 0)
    unit = dict(document.delay_units).get(lane, "ns")
    return _round_ties_away(_eval_expr(raw, slots) * _UNITS_TO_NS[unit] / step_ns)


def _fold_global_delay_shift(
    lanes: Mapping[str, int], buses: Mapping[int, int]
) -> tuple[dict[str, int], dict[int, int], int]:
    values = list(lanes.values()) + list(buses.values())
    global_shift = max(0, -min(values)) if values else 0
    return (
        {key: value + global_shift for key, value in lanes.items() if value + global_shift},
        {key: value + global_shift for key, value in buses.items() if value + global_shift},
        global_shift,
    )


def _has_bracket(document: PulseDocument) -> bool:
    return (
        document.repeat_start is not None
        and document.repeat_end is not None
        and document.repeat_count > 1
    )


def _has_any_delay(
    document: PulseDocument, slots: Mapping[str, float], step_ns: float
) -> bool:
    for lane in document.target.raw_lanes:
        raw = dict(document.delays).get(lane, 0)
        if isinstance(raw, str) and _SLOT_NAME.search(raw):
            return True
        if _delay_ticks(document, lane, slots, step_ns):
            return True
    return False


def _unroll_bracket(document: PulseDocument) -> PulseDocument:
    if not _has_bracket(document):
        return document
    assert document.repeat_start is not None and document.repeat_end is not None
    begin, end = document.repeat_start, document.repeat_end

    def expand(values: Sequence[object]) -> tuple[object, ...]:
        return tuple(values[:begin]) + tuple(values[begin : end + 1]) * document.repeat_count + tuple(values[end + 1 :])

    programs = tuple(
        (key, tuple(expand(steps))) for key, steps in document.analog_bus_programs
    )
    return replace(
        document,
        periods=tuple(expand(document.periods)),
        analog_bus_programs=programs,
        repeat_start=None,
        repeat_end=None,
        repeat_count=1,
    )


def _clock_enable(target: PulseTarget) -> int:
    lane_index = {lane: index for index, lane in enumerate(target.raw_lanes)}
    return sum(
        1 << lane_index[port.lanes[0]]
        for port in target.ports
        if port.kind == PORT_CLOCK
    )


def _dac_port_for_slot(target: PulseTarget, slot_target: str) -> PulsePortSpec:
    key, separator, _period = slot_target.partition("@")
    if not separator:
        raise ValueError(f"DAC slot target {slot_target!r} must be '<port>@<period>'")
    port = target.by_key.get(key)
    if port is None or port.kind != PORT_DAC:
        raise ValueError(f"DAC slot references unknown DAC port {key!r}")
    return port


def _slot_reference(value: object, slot_vars: Sequence[str]) -> int | None:
    if not isinstance(value, str):
        return None
    match = _SLOT_NAME.fullmatch(value.strip())
    if match is None:
        return None
    index = int(match.group(1))
    return index if index < len(slot_vars) and slot_vars[index] == f"s{index}" else None


def _validate_target_geometry(target: PulseTarget, params: StreamerParams) -> None:
    if len(target.raw_lanes) > int(params.channel_count):
        raise ValueError("PulseTarget has more raw lanes than the frozen streamer geometry")
    dac = [port for port in target.ports if port.kind == PORT_DAC]
    if len(dac) > int(params.bus_count):
        raise ValueError("PulseTarget has more DAC buses than the frozen streamer geometry")
    if any(port.width > int(params.bus_width) for port in dac):
        raise ValueError("PulseTarget DAC width exceeds the frozen streamer geometry")


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _round_ties_away(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def _apply_affine(
    base: int, coeffs: Sequence[int], point: Sequence[int], frac_bits: int
) -> int:
    return int(base) + (
        sum(int(coeff) * int(value) for coeff, value in zip(coeffs, point))
        >> int(frac_bits)
    )


def _affine_coeffs(
    value: int | float | str,
    *,
    unit: str,
    slot_vars: Sequence[str],
    step_ns: float,
    coeff_frac_bits: int,
) -> tuple[int, tuple[int, ...]]:
    base, coefficients = _SafeExpression().affine(value)
    unit_scale = _UNITS_TO_NS[unit]
    raw_base = base * unit_scale / step_ns
    base_ticks = int(round(raw_base))
    if not math.isclose(raw_base, base_ticks, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("affine timing base is not on the FPGA clock grid")
    unknown = set(coefficients) - set(slot_vars)
    if unknown:
        raise ValueError(f"expression references unbound scan variables {sorted(unknown)}")
    scale = 1 << coeff_frac_bits
    fixed: list[int] = []
    for name in slot_vars:
        coefficient = coefficients.get(name, 0.0) * unit_scale
        encoded = int(round(coefficient * scale))
        if not math.isclose(encoded / scale, coefficient, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("scan coefficient cannot be represented by the target ABI")
        fixed.append(encoded)
    return base_ticks, tuple(fixed)


def _eval_expr(value: int | float | str, slots: Mapping[str, float]) -> float:
    if isinstance(value, Number):
        result = float(value)
    else:
        result = _SafeExpression(slots).evaluate(str(value))
    if not math.isfinite(result):
        raise ValueError("pulse expression must be finite")
    return result


@lru_cache(maxsize=1024)
def _parse_expression(text: str) -> ast.expr:
    normalized = text.strip()
    if not normalized:
        raise ValueError("pulse expression must not be empty")
    variable = r"(?:s\d+)"
    normalized = re.sub(r"(\d|\.|\))\s*(" + variable + r")", r"\1*\2", normalized)
    normalized = re.sub(r"(" + variable + r"|\)|\d|\.)\s*(\()", r"\1*\2", normalized)
    return ast.parse(normalized, mode="eval").body


class _SafeExpression:
    def __init__(self, slots: Mapping[str, float] | None = None) -> None:
        self._slots = {str(key): float(value) for key, value in dict(slots or {}).items()}

    def evaluate(self, text: str) -> float:
        return float(self._visit(_parse_expression(text)))

    def _visit(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and _SLOT_NAME.fullmatch(node.id):
            if node.id not in self._slots:
                raise ValueError(f"pulse expression references unbound scan slot {node.id!r}")
            return self._slots[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = self._visit(node.left), self._visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
        raise ValueError("pulse expressions allow only numbers, sN, arithmetic, and parentheses")

    def affine(self, value: int | float | str) -> tuple[float, dict[str, float]]:
        if isinstance(value, Number):
            return float(value), {}
        return self._affine(_parse_expression(str(value)))

    def _affine(self, node: ast.AST) -> tuple[float, dict[str, float]]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value), {}
        if isinstance(node, ast.Name) and _SLOT_NAME.fullmatch(node.id):
            return 0.0, {node.id: 1.0}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            base, coeffs = self._affine(node.operand)
            if isinstance(node.op, ast.USub):
                return -base, {name: -coefficient for name, coefficient in coeffs.items()}
            return base, coeffs
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left_base, left = self._affine(node.left)
            right_base, right = self._affine(node.right)
            sign = -1.0 if isinstance(node.op, ast.Sub) else 1.0
            merged = dict(left)
            for name, coefficient in right.items():
                merged[name] = merged.get(name, 0.0) + sign * coefficient
            return left_base + sign * right_base, merged
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left_base, left = self._affine(node.left)
            right_base, right = self._affine(node.right)
            if not left:
                return right_base * left_base, {
                    name: coefficient * left_base for name, coefficient in right.items()
                }
            if not right:
                return left_base * right_base, {
                    name: coefficient * right_base for name, coefficient in left.items()
                }
            raise ValueError("scan timing only supports affine slot expressions")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left_base, left = self._affine(node.left)
            right_base, right = self._affine(node.right)
            if right or right_base == 0:
                raise ValueError("scan timing only supports division by a nonzero constant")
            return left_base / right_base, {
                name: coefficient / right_base for name, coefficient in left.items()
            }
        raise ValueError("scan timing only supports affine expressions in sN slots")


__all__ = [
    "COMPILER_ID",
    "COMPILER_VERSION",
    "compile_pulse_artifact",
    "compile_pulse_document",
]
