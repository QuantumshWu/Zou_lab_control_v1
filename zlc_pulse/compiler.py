"""Sole compiler from current PulseDocument values to FPGA TargetIR.

Stable ``ParameterId -> PulseFieldRef`` bindings are lowered to dense target
slots only inside this module.  The authoring model never contains ``sN``
expressions or position-parsed targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from fpga.pulse_streamer.host.image import StreamerParams
from zlc_storage import positive_real as _positive_float

from .artifact import CompiledPulseArtifact, PulseExecutionForm
from .document import (
    FIELD_DAC,
    FIELD_DURATION,
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    PulseFieldRef,
    PulsePeriod,
    ScanParameter,
    _exact_ticks,
    _integral_code,
)
from .fpga import pack_target_ir
from .ir import TargetBusDelay, TargetBusSegment, TargetIR, evaluate_affine_tick
from .schedule import DigitalTriggerSchedule, build_digital_trigger_schedules
from .target import PORT_CLOCK, PORT_DAC, PORT_DIGITAL, PulseTarget
from .validation import validate_target_ir_for_target


COMPILER_ID = "zlc-pulse-native"


@dataclass(frozen=True)
class _ProgramView:
    target: PulseTarget
    periods: tuple[PulsePeriod, ...]
    period_origins: tuple[str, ...]
    delays: Mapping[str, OutputDelay]
    scan_parameters: tuple[ScanParameter, ...]
    scan_table: FrozenScanTable | None
    repeat_start: int | None
    repeat_end: int | None
    repeat_count: int


def _program_view(document: PulseDocument) -> _ProgramView:
    index = {period.period_id: position for position, period in enumerate(document.periods)}
    if document.repeat is None:
        start = end = None
        count = 1
    else:
        start = index[document.repeat.start_period_id]
        end = index[document.repeat.end_period_id]
        count = document.repeat.count
    return _ProgramView(
        target=document.target,
        periods=document.periods,
        period_origins=tuple(period.period_id for period in document.periods),
        delays={delay.port: delay for delay in document.delays},
        scan_parameters=document.scan_parameters,
        scan_table=document.scan_table,
        repeat_start=start,
        repeat_end=end,
        repeat_count=count,
    )


def compile_pulse_document(
    document: PulseDocument,
    *,
    clock_hz: float,
    execution_form: PulseExecutionForm,
    live_target: PulseTarget | None = None,
    params: StreamerParams | None = None,
) -> TargetIR:
    """Compile one fully frozen current document without changing its values."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(execution_form, PulseExecutionForm):
        raise TypeError("execution_form must be PulseExecutionForm")
    frequency = _positive_float(clock_hz, "clock_hz")
    step_ns = 1e9 / frequency
    if not math.isclose(document.time_step_ns, step_ns, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("pulse document clock grid differs from compile clock")
    geometry = params or StreamerParams()
    if live_target is not None:
        if not isinstance(live_target, PulseTarget):
            raise TypeError("live_target must be PulseTarget")
        if live_target.abi_fingerprint != document.target.abi_fingerprint:
            raise ValueError("pulse document target ABI differs from live target")
    _validate_target_geometry(document.target, geometry)

    if execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        if document.scan_table is None or not document.scan_table.rows:
            raise ValueError("AUTONOMOUS_SCAN_ONCE requires a frozen scan table with rows")
        return _compile_scan(
            _program_view(document),
            clock_hz=frequency,
            coeff_frac_bits=int(geometry.coeff_frac_bits),
        )

    reference = execution_form is PulseExecutionForm.STATIC_REFERENCE_POINT
    if (document.scan_parameters or document.scan_table is not None) and not reference:
        raise ValueError("static execution cannot silently ignore a scan definition")
    if reference and not document.scan_parameters:
        raise ValueError("STATIC_REFERENCE_POINT requires at least one scan parameter")
    return _compile_static(
        _program_view(document),
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
    validate_target_ir_for_target(target_ir, document.target)
    schedules = (
        ()
        if execution_form is PulseExecutionForm.CONTINUOUS_MONITOR or not channels
        else build_digital_trigger_schedules(target_ir, channels)
    )
    source_loop_count = 1 if document.repeat is None else document.repeat.count
    if schedules and target_ir.loop_count != source_loop_count:
        # Output-delay lowering may have to unroll the hardware loop, but that
        # wire-level decision must not erase the source execution groups used
        # by host simulation and capture cross-validation.  Recompile only the
        # host-side grouping projection without output delays, then join its
        # point/loop ownership onto the unchanged physical ticks by ordinal.
        grouping_document = replace(document, delays=())
        grouping_ir = compile_pulse_document(
            grouping_document,
            clock_hz=clock_hz,
            execution_form=execution_form,
            live_target=live_target,
            params=geometry,
        )
        grouping_schedules = build_digital_trigger_schedules(
            grouping_ir,
            channels,
        )
        schedules = _join_source_trigger_groups(schedules, grouping_schedules)
    return CompiledPulseArtifact(
        source_document_digest=document.fingerprint,
        compiler_id=COMPILER_ID,
        execution_form=execution_form,
        target_ir=target_ir,
        wire_image=pack_target_ir(target_ir, geometry),
        trigger_schedules=schedules,
    )


def _join_source_trigger_groups(
    physical: tuple[DigitalTriggerSchedule, ...],
    grouped: tuple[DigitalTriggerSchedule, ...],
) -> tuple[DigitalTriggerSchedule, ...]:
    if tuple(item.channel for item in physical) != tuple(
        item.channel for item in grouped
    ):
        raise RuntimeError("source and physical trigger schedule channels differ")
    joined = []
    for physical_schedule, grouped_schedule in zip(
        physical,
        grouped,
        strict=True,
    ):
        if (
            physical_schedule.point_count != grouped_schedule.point_count
            or len(physical_schedule.edges) != len(grouped_schedule.edges)
        ):
            raise RuntimeError("source and physical trigger cardinality differ")
        edges = []
        for physical_edge, grouped_edge in zip(
            physical_schedule.edges,
            grouped_schedule.edges,
            strict=True,
        ):
            if (
                physical_edge.channel != grouped_edge.channel
                or physical_edge.point_index != grouped_edge.point_index
                or physical_edge.trigger_ordinal != grouped_edge.trigger_ordinal
                or physical_edge.point_trigger_ordinal
                != grouped_edge.point_trigger_ordinal
            ):
                raise RuntimeError("source and physical trigger order differ")
            edges.append(
                replace(
                    physical_edge,
                    loop_iteration=grouped_edge.loop_iteration,
                )
            )
        joined.append(
            DigitalTriggerSchedule(
                physical_schedule.channel,
                tuple(edges),
                physical_schedule.point_count,
                grouped_schedule.loop_count,
                grouped_schedule.full_point_loop,
            )
        )
    return tuple(joined)


def _compile_static(
    view: _ProgramView,
    *,
    clock_hz: float,
    repeat_forever: bool,
) -> TargetIR:
    step_ns = 1e9 / clock_hz
    work = view
    has_bracket = _has_bracket(work)
    if has_bracket and _has_any_delay(work, step_ns):
        work = _unroll_bracket(work)
        has_bracket = False

    starts = _period_starts(work, step_ns)
    bus_names, bus_segments, raw_bus_delays = _bus_segments(
        work,
        starts=starts,
        step_ns=step_ns,
        binding={},
        coeff_frac_bits=0,
    )
    driven_bus_raw = {
        segment.bus_index: int(raw_bus_delays.get(segment.bus_index, 0))
        for segment in bus_segments
    }
    ticks, masks, channel_delays, shifted_bus_delays = _static_edges(
        work,
        starts=starts,
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
    view: _ProgramView,
    *,
    clock_hz: float,
    coeff_frac_bits: int,
) -> TargetIR:
    step_ns = 1e9 / clock_hz
    work = view
    if _has_bracket(work) and _has_any_delay(work, step_ns):
        work = _unroll_bracket(work)
    assert work.scan_table is not None and work.scan_table.rows
    columns = _scan_columns(work)
    binding = {parameter.field: index for index, parameter in enumerate(columns)}
    scan_points = _wire_scan_points(work, columns, step_ns)
    reference_starts = _period_starts(
        work,
        step_ns,
        binding=binding,
        scan_row=work.scan_table.rows[0],
    )
    affine_starts = _affine_period_starts(
        work,
        binding=binding,
        step_ns=step_ns,
        coeff_frac_bits=coeff_frac_bits,
    )
    bus_names, bus_segments, raw_bus_delays = _bus_segments(
        work,
        starts=reference_starts,
        step_ns=step_ns,
        binding=binding,
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
        slot_count=len(columns),
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
            index for index, row in enumerate(rows) if (row[0], row[2]) == loop_expr
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
    owner_by_lane = {
        lane: port for port in work.target.ports for lane in port.lanes
    }
    for lane_index, lane in enumerate(work.target.raw_lanes):
        if lane in bus_members or lane in clock_lanes:
            continue
        if not any(period.states[lane_index] for period in work.periods):
            continue
        raw_ttl_delays[lane] = _delay_ticks(
            work, owner_by_lane[lane].key, step_ns
        )
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

    loop_start_base = ticks[loop_start_index]
    loop_start_coeffs = tick_coeffs[loop_start_index]
    point_durations = tuple(
        (
            evaluate_affine_tick(ticks[-1], tick_coeffs[-1], point, coeff_frac_bits)
            + (loop_count - 1)
            * (
                evaluate_affine_tick(
                    loop_end_tick,
                    loop_end_coeffs,
                    point,
                    coeff_frac_bits,
                )
                - evaluate_affine_tick(
                    loop_start_base,
                    loop_start_coeffs,
                    point,
                    coeff_frac_bits,
                )
            )
        )
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
        slot_kinds=tuple(parameter.field.kind for parameter in columns),
        loop_end_slot_coeffs=tuple(loop_end_coeffs),
        tick_slot_coeffs=tuple(tuple(row) for row in tick_coeffs),
        scan_points=tuple(tuple(point) for point in scan_points),
        scan_point_durations=point_durations,
        scan_coeff_frac_bits=coeff_frac_bits,
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


def _scan_columns(view: _ProgramView) -> tuple[ScanParameter, ...]:
    assert view.scan_table is not None
    by_id = {parameter.parameter_id: parameter for parameter in view.scan_parameters}
    return tuple(by_id[parameter_id] for parameter_id in view.scan_table.columns)


def _duration_ref(view: _ProgramView, period_index: int) -> PulseFieldRef:
    return PulseFieldRef(FIELD_DURATION, view.period_origins[period_index], None)


def _dac_ref(view: _ProgramView, period_index: int, port: str) -> PulseFieldRef:
    return PulseFieldRef(FIELD_DAC, view.period_origins[period_index], port)


def _static_edges(
    view: _ProgramView,
    *,
    starts: Sequence[int],
    step_ns: float,
    extra_raw_bus_delays: Mapping[int, int],
) -> tuple[list[int], list[int], list[int], dict[int, int]]:
    bus_members = {
        lane
        for port in view.target.ports
        if port.kind == PORT_DAC
        for lane in port.lanes
    }
    clock_lanes = {
        port.lanes[0] for port in view.target.ports if port.kind == PORT_CLOCK
    }
    owner_by_lane = {
        lane: port for port in view.target.ports for lane in port.lanes
    }
    intervals: dict[str, list[tuple[int, int]]] = {}
    raw_delays: dict[str, int] = {}
    for lane_index, lane in enumerate(view.target.raw_lanes):
        if lane in bus_members or lane in clock_lanes:
            continue
        active: int | None = None
        lane_intervals: list[tuple[int, int]] = []
        for period_index, period in enumerate(view.periods):
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
            raw_delays[lane] = _delay_ticks(
                view, owner_by_lane[lane].key, step_ns
            )
    shifted, shifted_buses, _global = _fold_global_delay_shift(
        raw_delays, extra_raw_bus_delays
    )
    events: dict[int, list[tuple[str, int]]] = {int(tick): [] for tick in starts}
    for lane, lane_intervals in intervals.items():
        for begin, end in lane_intervals:
            events.setdefault(begin, []).append((lane, 1))
            events.setdefault(end, []).append((lane, 0))
    bit = {lane: index for index, lane in enumerate(view.target.raw_lanes)}
    ticks: list[int] = []
    masks: list[int] = []
    current = 0
    for tick in sorted(events):
        for lane, value in events[tick]:
            current = (
                current | (1 << bit[lane])
                if value
                else current & ~(1 << bit[lane])
            )
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
    channel_delays = [0] * len(view.target.raw_lanes)
    for lane, delay in shifted.items():
        channel_delays[bit[lane]] = delay
    return ticks, masks, channel_delays, shifted_buses


def _affine_edge_rows(
    view: _ProgramView,
    *,
    affine_starts: Sequence[tuple[int, tuple[int, ...]]],
    scan_points: Sequence[Sequence[int]],
    coeff_frac_bits: int,
    excluded: set[str],
    slot_count: int,
) -> list[tuple[int, int, tuple[int, ...]]]:
    events: dict[
        tuple[int, tuple[int, ...]], list[tuple[str | None, int | None]]
    ] = {}
    for lane_index, lane in enumerate(view.target.raw_lanes):
        if lane in excluded:
            continue
        active: tuple[int, tuple[int, ...]] | None = None
        for period_index, period in enumerate(view.periods):
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
    events.setdefault((0, tuple(0 for _ in range(slot_count))), []).append((None, None))
    events.setdefault(affine_starts[-1], []).append((None, None))
    if _has_bracket(view) and view.repeat_count > 1:
        assert view.repeat_start is not None
        events.setdefault(affine_starts[view.repeat_start], []).append((None, None))

    point0 = scan_points[0]
    ordered = sorted(
        events,
        key=lambda expression: (
            evaluate_affine_tick(
                expression[0],
                expression[1],
                point0,
                coeff_frac_bits,
            ),
            not any(lane is not None for lane, _value in events[expression]),
            expression,
        ),
    )
    reference_ticks = [
        evaluate_affine_tick(
            expression[0],
            expression[1],
            point0,
            coeff_frac_bits,
        )
        for expression in ordered
    ]
    if any(right <= left for left, right in zip(reference_ticks, reference_ticks[1:])):
        raise ValueError("scan moves or collides distinct edge rows at its reference point")
    bit = {lane: index for index, lane in enumerate(view.target.raw_lanes)}
    current = 0
    rows: list[tuple[int, int, tuple[int, ...]]] = []
    for expression in ordered:
        for lane, value in events[expression]:
            if lane is None or value is None:
                continue
            current = (
                current | (1 << bit[lane])
                if value
                else current & ~(1 << bit[lane])
            )
        rows.append((expression[0], current, expression[1]))
    if rows[-1][1] != 0:
        raise ValueError("hardware scan template does not return every digital lane to zero")
    for point_index, point in enumerate(scan_points):
        effective = [
            evaluate_affine_tick(base, coeffs, point, coeff_frac_bits)
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
    view: _ProgramView,
    *,
    starts: Sequence[int],
    step_ns: float,
    binding: Mapping[PulseFieldRef, int],
    coeff_frac_bits: int,
    affine_starts: Sequence[tuple[int, tuple[int, ...]]] | None = None,
) -> tuple[list[str], list[TargetBusSegment], dict[int, int]]:
    bus_ports = sorted(
        (port for port in view.target.ports if port.kind == PORT_DAC),
        key=lambda port: int(port.bus_index),
    )
    names = [port.key for port in bus_ports]
    segments: list[TargetBusSegment] = []
    bus_delays: dict[int, int] = {}
    zero_coeffs = tuple(0 for _ in binding)
    for port in bus_ports:
        assert port.bus_index is not None and port.signed_range is not None
        delay = _delay_ticks(view, port.key, step_ns)
        if delay:
            bus_delays[port.bus_index] = delay
        zero_code = -port.signed_range[0]
        code_max = (1 << port.width) - 1
        for period_index, period in enumerate(view.periods):
            step = next(
                (item for item in period.analog_steps if item.port == port.key),
                None,
            )
            if step is None:
                continue
            selector = binding.get(_dac_ref(view, period_index, port.key))
            if selector is None:
                code = step.value + zero_code
                if not 0 <= code <= code_max:
                    raise ValueError(f"DAC value for {port.key!r} exceeds target range")
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
    view: _ProgramView,
    step_ns: float,
    *,
    binding: Mapping[PulseFieldRef, int] | None = None,
    scan_row: Sequence[float] | None = None,
) -> list[int]:
    active = dict(binding or {})
    starts = [0]
    columns = _scan_columns(view) if active else ()
    for period_index, period in enumerate(view.periods):
        selector = active.get(_duration_ref(view, period_index))
        if selector is None:
            value = period.duration
            unit = period.unit
        else:
            if scan_row is None:
                raise ValueError("scan-bound period start requires a frozen scan row")
            value = scan_row[selector]
            unit = columns[selector].unit
        ticks = _exact_ticks(value, unit, step_ns, "period duration")
        starts.append(starts[-1] + ticks)
    return starts


def _affine_period_starts(
    view: _ProgramView,
    *,
    binding: Mapping[PulseFieldRef, int],
    step_ns: float,
    coeff_frac_bits: int,
) -> list[tuple[int, tuple[int, ...]]]:
    slot_count = len(binding)
    starts = [(0, tuple(0 for _ in range(slot_count)))]
    scale = 1 << coeff_frac_bits
    for period_index, period in enumerate(view.periods):
        selector = binding.get(_duration_ref(view, period_index))
        if selector is None:
            base = _exact_ticks(
                period.duration, period.unit, step_ns, "period duration"
            )
            coefficients = tuple(0 for _ in range(slot_count))
        else:
            base = 0
            coefficients = tuple(
                scale if index == selector else 0 for index in range(slot_count)
            )
        starts.append(
            (
                starts[-1][0] + base,
                tuple(a + b for a, b in zip(starts[-1][1], coefficients)),
            )
        )
    return starts


def _wire_scan_points(
    view: _ProgramView,
    columns: Sequence[ScanParameter],
    step_ns: float,
) -> list[list[int]]:
    assert view.scan_table is not None
    points: list[list[int]] = []
    for row in view.scan_table.rows:
        point: list[int] = []
        for parameter, value in zip(columns, row):
            if parameter.field.kind == FIELD_DURATION:
                point.append(
                    _exact_ticks(value, parameter.unit, step_ns, "scan duration")
                )
            else:
                port = view.target.by_key[parameter.field.port]
                assert port.signed_range is not None
                code = _integral_code(value, "scan DAC value")
                if not port.signed_range[0] <= code <= port.signed_range[1]:
                    raise ValueError("scan DAC value exceeds target range")
                point.append(code - port.signed_range[0])
        points.append(point)
    return points


def _delay_ticks(view: _ProgramView, port: str, step_ns: float) -> int:
    delay = view.delays.get(port)
    if delay is None:
        return 0
    return _exact_ticks(
        delay.value,
        delay.unit,
        step_ns,
        f"delay {port!r}",
        minimum=None,
    )


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


def _has_bracket(view: _ProgramView) -> bool:
    return (
        view.repeat_start is not None
        and view.repeat_end is not None
        and view.repeat_count > 1
    )


def _has_any_delay(view: _ProgramView, step_ns: float) -> bool:
    return any(_delay_ticks(view, port, step_ns) for port in view.delays)


def _unroll_bracket(view: _ProgramView) -> _ProgramView:
    if not _has_bracket(view):
        return view
    assert view.repeat_start is not None and view.repeat_end is not None
    begin, end = view.repeat_start, view.repeat_end

    def expand(values: Sequence[object]) -> tuple[object, ...]:
        return (
            tuple(values[:begin])
            + tuple(values[begin : end + 1]) * view.repeat_count
            + tuple(values[end + 1 :])
        )

    return replace(
        view,
        periods=tuple(expand(view.periods)),
        period_origins=tuple(expand(view.period_origins)),
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


def _validate_target_geometry(target: PulseTarget, params: StreamerParams) -> None:
    if len(target.raw_lanes) > int(params.channel_count):
        raise ValueError("PulseTarget has more raw lanes than the frozen streamer geometry")
    dac = [port for port in target.ports if port.kind == PORT_DAC]
    if len(dac) > int(params.bus_count):
        raise ValueError("PulseTarget has more DAC buses than the frozen streamer geometry")
    if any(port.width > int(params.bus_width) for port in dac):
        raise ValueError("PulseTarget DAC width exceeds the frozen streamer geometry")


__all__ = [
    "COMPILER_ID",
    "compile_pulse_artifact",
    "compile_pulse_document",
]
