"""Frozen-bitstream capability gate for current pulse artifacts.

The host image writer is intentionally a serializer: several wire fields are
packed with masks.  Every value must therefore pass this module before packing,
otherwise an oversized value could become a different physical pulse.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
import math
from typing import Sequence

from fpga.pulse_streamer.host.image import (
    StreamerParams,
    check_rtl_assumptions,
)
from zlc_storage import positive_real

from .document import PulseDocument
from .ir import TargetIR, evaluate_affine_tick
from .target import PORT_CLOCK, PORT_DAC, PORT_DIGITAL, PulseTarget


SCAN_READ_LATENCY_TICKS = 2
SLOT_MULTIPLIER_WIDTH = 25
COUNTER_LIMIT = (1 << 32) - 1


def validate_pulse_document_clock_grid(
    document: PulseDocument,
    clock_hz: int | float,
) -> float:
    """Return the validated clock shared by preflight and the sole compiler."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    frequency = positive_real(clock_hz, "clock_hz")
    step_ns = 1e9 / frequency
    if not math.isclose(
        document.time_step_ns,
        step_ns,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("pulse document clock grid differs from compile clock")
    return frequency


def validate_target_ir_for_geometry(
    ir: TargetIR,
    params: StreamerParams | None = None,
) -> None:
    """Reject every TargetIR value the frozen RTL would truncate or misplay."""

    if not isinstance(ir, TargetIR):
        raise TypeError("validate_target_ir_for_geometry requires TargetIR")
    geometry = params or StreamerParams()
    check_rtl_assumptions(geometry)
    operand_width = SLOT_MULTIPLIER_WIDTH

    if len(ir.channels) > geometry.channel_count:
        raise ValueError(
            f"TargetIR uses {len(ir.channels)} channels, but geometry has "
            f"{geometry.channel_count}"
        )
    if len(ir.ticks) > geometry.max_edges:
        raise ValueError(
            f"TargetIR has {len(ir.ticks)} edges, above max_edges={geometry.max_edges}"
        )
    if ir.slot_count > geometry.num_slots:
        raise ValueError(
            f"TargetIR uses {ir.slot_count} scan slots, but geometry has "
            f"{geometry.num_slots}"
        )
    if len(ir.bus_names) > geometry.bus_count:
        raise ValueError(
            f"TargetIR uses {len(ir.bus_names)} DAC buses, but geometry has "
            f"{geometry.bus_count}"
        )
    if len(ir.scan_points) > COUNTER_LIMIT:
        raise ValueError("scan point count exceeds the 32-bit hardware counter")
    if (
        ir.repeat_forever
        and ir.scan_points
        and len(ir.scan_points) > (1 << 32) - geometry.bank_size
    ):
        raise ValueError(
            "cyclic scan point count would overflow the frozen RTL chunk-count addition"
        )
    if ir.loop_count > COUNTER_LIMIT:
        raise ValueError("loop_count exceeds the 32-bit hardware counter")

    tick_limit = (1 << geometry.tick_width) - 1
    if any(tick > tick_limit for tick in ir.ticks):
        raise ValueError(f"TargetIR base tick does not fit {geometry.tick_width} bits")
    if ir.loop_end_tick > tick_limit:
        raise ValueError(f"TargetIR loop end does not fit {geometry.tick_width} bits")
    coeff_min = -(1 << (geometry.coeff_width - 1))
    coeff_max = (1 << (geometry.coeff_width - 1)) - 1
    coefficients = [
        *ir.loop_end_slot_coeffs,
        *(value for row in ir.tick_slot_coeffs for value in row),
        *(
            value
            for segment in ir.bus_segments
            for row in (segment.start_tick_coeffs, segment.stop_tick_coeffs)
            for value in row
        ),
    ]
    if any(value < coeff_min or value > coeff_max for value in coefficients):
        raise ValueError(
            f"TargetIR scan coefficient does not fit signed {geometry.coeff_width} bits"
        )

    if ir.scan_points and ir.scan_coeff_frac_bits != geometry.coeff_frac_bits:
        raise ValueError(
            "TargetIR scan coefficient scale differs from the frozen hardware geometry"
        )
    slot_min = -(1 << (operand_width - 1))
    slot_max = (1 << (operand_width - 1)) - 1
    bus_value_limit = (1 << geometry.bus_width) - 1
    for point_index, point in enumerate(ir.scan_points):
        for slot_index, (kind, value) in enumerate(zip(ir.slot_kinds, point)):
            if not slot_min <= value <= slot_max:
                raise ValueError(
                    f"scan point {point_index} slot {slot_index} does not fit signed "
                    f"{operand_width}-bit RTL multiplier input"
                )
            if kind == "duration" and value < 1:
                raise ValueError(
                    f"scan point {point_index} duration slot {slot_index} must be positive"
                )
            if kind == "dac" and not 0 <= value <= bus_value_limit:
                raise ValueError(
                    f"scan point {point_index} DAC slot {slot_index} does not fit "
                    f"{geometry.bus_width} bits"
                )
        effective = _effective_ticks(ir, point)
        if effective[-1] > tick_limit:
            raise ValueError(
                f"scan point {point_index} effective tick does not fit "
                f"{geometry.tick_width} bits"
            )
        loop_end = evaluate_affine_tick(
            ir.loop_end_tick,
            ir.loop_end_slot_coeffs,
            point,
            ir.scan_coeff_frac_bits,
        )
        point_wall_ticks = effective[-1] + (ir.loop_count - 1) * (
            loop_end - effective[ir.loop_start_index]
        )
        if len(ir.scan_points) > 1 and point_wall_ticks < SCAN_READ_LATENCY_TICKS:
            raise ValueError(
                f"scan point {point_index} frame is shorter than the "
                f"{SCAN_READ_LATENCY_TICKS}-tick scan-BRAM read latency"
            )
        if loop_end > tick_limit:
            raise ValueError(
                f"scan point {point_index} loop end does not fit "
                f"{geometry.tick_width} bits"
            )

    counts = [0] * geometry.bus_count
    for segment in ir.bus_segments:
        if segment.bus_index >= geometry.bus_count:
            raise ValueError("bus segment index exceeds frozen geometry")
        counts[segment.bus_index] += 1
        if counts[segment.bus_index] > geometry.max_bus_segments:
            raise ValueError(
                f"bus {segment.bus_index} has more than "
                f"{geometry.max_bus_segments} hardware segment rows"
            )
        if segment.start_tick > tick_limit or segment.stop_tick > tick_limit:
            raise ValueError("bus segment tick does not fit the hardware field")
        if not 0 <= segment.start_value <= bus_value_limit or not (
            0 <= segment.stop_value <= bus_value_limit
        ):
            raise ValueError(
                f"bus segment literal value does not fit {geometry.bus_width} bits"
            )
        selector_limit = (1 << geometry.bus_sel_width) - 1
        if (
            segment.value_select > selector_limit
            or segment.stop_value_select > selector_limit
        ):
            raise ValueError("bus segment selector does not fit the hardware field")

    for bit, delay in enumerate(ir.channel_delays):
        if delay > geometry.ttl_delay_max_ticks:
            raise ValueError(
                f"channel {bit} delay exceeds the frozen 32-bit delay contract"
            )
        if delay and bit >= geometry.num_delay_ch:
            raise ValueError(
                f"channel {bit} has a delay but no hardware event FIFO"
            )
    for delay in ir.bus_delays:
        if delay.bus_index >= geometry.bus_count:
            raise ValueError("bus delay index exceeds frozen geometry")
        if delay.delay_ticks > geometry.ttl_delay_max_ticks:
            raise ValueError("bus delay exceeds the frozen 32-bit delay contract")

    _validate_bus_delay_capacity(ir, geometry)
    _validate_channel_delay_capacity(ir, geometry)


def validate_target_ir_for_target(ir: TargetIR, target: PulseTarget) -> None:
    """Verify that IR metadata cannot contradict its attested physical target."""

    if not isinstance(ir, TargetIR):
        raise TypeError("validate_target_ir_for_target requires TargetIR")
    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    if ir.target_abi_fingerprint != target.abi_fingerprint:
        raise ValueError("TargetIR ABI fingerprint differs from PulseTarget")
    if ir.channels != target.raw_lanes:
        raise ValueError("TargetIR channel order differs from PulseTarget raw lanes")
    expected_digital_outputs = tuple(
        sorted(
            (port.key, port.lanes[0])
            for port in target.ports
            if port.kind == PORT_DIGITAL
        )
    )
    if ir.logical_digital_outputs != expected_digital_outputs:
        raise ValueError(
            "TargetIR logical digital outputs differ from PulseTarget topology"
        )

    lane_index = {lane: index for index, lane in enumerate(target.raw_lanes)}
    digital_mask = sum(
        1 << lane_index[port.lanes[0]]
        for port in target.ports
        if port.kind == PORT_DIGITAL
    )
    if any(mask & ~digital_mask for mask in ir.masks):
        raise ValueError("TargetIR drives a non-digital physical lane as TTL")
    expected_clocks = sum(
        1 << lane_index[port.lanes[0]]
        for port in target.ports
        if port.kind == PORT_CLOCK
    )
    if ir.clk_enable != expected_clocks:
        raise ValueError("TargetIR clock-enable mask differs from PulseTarget topology")

    buses = tuple(
        sorted(
            (port for port in target.ports if port.kind == PORT_DAC),
            key=lambda port: int(port.bus_index),
        )
    )
    if ir.bus_names != tuple(port.key for port in buses):
        raise ValueError("TargetIR DAC bus order differs from PulseTarget")
    if ir.bus_safe_values != tuple(port.safe_value for port in buses):
        raise ValueError("TargetIR DAC safe values differ from PulseTarget")
    for segment in ir.bus_segments:
        port = buses[segment.bus_index]
        limit = (1 << port.width) - 1
        if not 0 <= segment.start_value <= limit or not (
            0 <= segment.stop_value <= limit
        ):
            raise ValueError(
                f"bus segment literal exceeds target port {port.key!r} width"
            )
        selectors = {
            selector - 1
            for selector in (segment.value_select, segment.stop_value_select)
            if selector
        }
        for slot_index in selectors:
            if any(point[slot_index] > limit for point in ir.scan_points):
                raise ValueError(
                    f"DAC scan slot exceeds target port {port.key!r} width"
                )

    owner_by_lane = {
        lane: port for port in target.ports for lane in port.lanes
    }
    for lane, delay in zip(target.raw_lanes, ir.channel_delays):
        if delay and owner_by_lane[lane].kind != PORT_DIGITAL:
            raise ValueError("TargetIR delays a non-digital physical lane")


def _validate_bus_delay_capacity(ir: TargetIR, geometry: StreamerParams) -> None:
    for delay in ir.bus_delays:
        if delay.delay_ticks <= 1:
            continue
        tracker = _CapacityTracker(
            width=delay.delay_ticks - 1,
            limit=geometry.bus_evt_fifo_depth,
        )
        try:
            period = _play_bus_sweep(ir, delay.bus_index, tracker, start=0)
            if ir.repeat_forever:
                if tracker.width < period:
                    _play_bus_sweep(ir, delay.bus_index, tracker, start=period)
                else:
                    steady = _BoundedEventCollector(geometry.bus_evt_fifo_depth)
                    _play_bus_sweep(ir, delay.bus_index, steady, start=0)
                    if steady.events:
                        tracker.add_periodic_block(
                            period,
                            steady.events,
                            period,
                            tracker.width // period + 2,
                        )
            else:
                # zlc_bus_capture_safe_hold enqueues this even if the bus was
                # already at midpoint.  It is a real FIFO descriptor.
                tracker.add_event(period)
        except _CapacityExceeded as overflow:
            raise ValueError(
                f"bus {delay.bus_index} delay keeps at least {overflow.count} "
                f"descriptors in flight, above FIFO depth "
                f"{geometry.bus_evt_fifo_depth}"
            ) from None


def _validate_channel_delay_capacity(ir: TargetIR, geometry: StreamerParams) -> None:
    for bit, delay in enumerate(ir.channel_delays):
        if delay <= 1:
            continue
        tracker = _CapacityTracker(
            width=delay - 1,
            limit=geometry.evt_fifo_depth,
        )
        try:
            period, end_state = _play_channel_sweep(
                ir,
                bit,
                tracker,
                start=0,
                initial_state=0,
            )
            if ir.repeat_forever:
                if tracker.width < period:
                    _play_channel_sweep(
                        ir,
                        bit,
                        tracker,
                        start=period,
                        initial_state=end_state,
                    )
                else:
                    steady = _BoundedEventCollector(geometry.evt_fifo_depth)
                    _play_channel_sweep(
                        ir,
                        bit,
                        steady,
                        start=0,
                        initial_state=end_state,
                    )
                    if steady.events:
                        tracker.add_periodic_block(
                            period,
                            steady.events,
                            period,
                            tracker.width // period + 2,
                        )
            elif end_state:
                # Finite DONE changes state_mask directly to zero.  The final
                # TargetIR row is a boundary sentinel, not a played edge.
                tracker.add_event(period)
        except _CapacityExceeded as overflow:
            raise ValueError(
                f"channel {bit} delay keeps at least {overflow.count} edges in "
                f"flight, above FIFO depth {geometry.evt_fifo_depth}"
            ) from None


class _CapacityExceeded(Exception):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.count = int(count)


class _CapacityTracker:
    """Exact bounded-window tracker with compressed periodic-block ingestion."""

    def __init__(self, *, width: int, limit: int) -> None:
        self.width = int(width)
        self.limit = int(limit)
        self._recent: deque[int] = deque()
        self._cursor = 0

    def add_event(self, tick: int) -> None:
        value = int(tick)
        if value < self._cursor:
            raise AssertionError("capacity events must be emitted in time order")
        self._expire(value)
        self._recent.append(value)
        self._cursor = value
        if len(self._recent) > self.limit:
            raise _CapacityExceeded(len(self._recent))

    def advance_to(self, tick: int) -> None:
        value = int(tick)
        if value < self._cursor:
            raise AssertionError("capacity stream cannot move backwards")
        self._expire(value)
        self._cursor = value

    def add_periodic_block(
        self,
        start: int,
        phases: Sequence[int],
        period: int,
        repetitions: int,
    ) -> None:
        origin = int(start)
        cycle = int(period)
        count = int(repetitions)
        ordered = tuple(sorted(int(phase) for phase in phases))
        if cycle <= 0 or count < 0 or any(
            phase < 0 or phase >= cycle for phase in ordered
        ):
            raise AssertionError("invalid compressed periodic capacity block")
        self.advance_to(origin)
        end = origin + cycle * count
        if not ordered or count == 0:
            self.advance_to(end)
            return
        total = len(ordered) * count
        internal = _max_finite_periodic_window(
            ordered,
            cycle,
            self.width,
            count,
        )
        if internal > self.limit:
            raise _CapacityExceeded(internal)

        prior_latest = self._recent[-1] if self._recent else None
        index = 0
        while prior_latest is not None and index < total:
            tick = _periodic_event_at(origin, ordered, cycle, index)
            if tick - prior_latest > self.width:
                break
            self.add_event(tick)
            index += 1
        if index == total:
            self.advance_to(end)
            return

        suffix = []
        index = total - 1
        while index >= 0:
            tick = _periodic_event_at(origin, ordered, cycle, index)
            if end - tick > self.width:
                break
            suffix.append(tick)
            if len(suffix) > self.limit:
                raise _CapacityExceeded(len(suffix))
            index -= 1
        self._recent = deque(reversed(suffix))
        self._cursor = end

    def _expire(self, terminal: int) -> None:
        while self._recent and terminal - self._recent[0] > self.width:
            self._recent.popleft()


class _BoundedEventCollector:
    """Materialize at most one FIFO's worth of a compressed steady sweep."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.events: list[int] = []
        self._cursor = 0

    def add_event(self, tick: int) -> None:
        value = int(tick)
        if value < self._cursor:
            raise AssertionError("collected events must be time ordered")
        if len(self.events) >= self.limit:
            raise _CapacityExceeded(self.limit + 1)
        self.events.append(value)
        self._cursor = value

    def advance_to(self, tick: int) -> None:
        value = int(tick)
        if value < self._cursor:
            raise AssertionError("collected stream cannot move backwards")
        self._cursor = value

    def add_periodic_block(
        self,
        start: int,
        phases: Sequence[int],
        period: int,
        repetitions: int,
    ) -> None:
        ordered = tuple(sorted(int(phase) for phase in phases))
        total = len(ordered) * int(repetitions)
        if len(self.events) + total > self.limit:
            raise _CapacityExceeded(self.limit + 1)
        self.advance_to(start)
        for index in range(total):
            self.add_event(_periodic_event_at(start, ordered, period, index))
        self.advance_to(int(start) + int(period) * int(repetitions))


def _play_bus_sweep(
    ir: TargetIR,
    bus_index: int,
    sink: _CapacityTracker | _BoundedEventCollector,
    *,
    start: int,
) -> int:
    offset = int(start)
    segments = tuple(
        segment for segment in ir.bus_segments if segment.bus_index == bus_index
    )
    for point in ir.scan_points or ((),):
        effective, loop_start, loop_end, loop_span, duration = _point_timing(ir, point)
        starts = tuple(
            evaluate_affine_tick(
                segment.start_tick,
                segment.start_tick_coeffs,
                point,
                ir.scan_coeff_frac_bits,
            )
            for segment in segments
        )
        if ir.loop_count <= 1:
            for tick in starts:
                sink.add_event(offset + tick)
        else:
            repeated = tuple(tick - loop_start for tick in starts if tick < loop_end)
            sink.add_periodic_block(
                offset + loop_start,
                repeated,
                loop_span,
                ir.loop_count,
            )
            tail_shift = (ir.loop_count - 1) * loop_span
            for tick in starts:
                if tick >= loop_end:
                    sink.add_event(offset + tick + tail_shift)
        offset += duration
        sink.advance_to(offset)
    return offset


def _play_channel_sweep(
    ir: TargetIR,
    bit: int,
    sink: _CapacityTracker | _BoundedEventCollector,
    *,
    start: int,
    initial_state: int,
) -> tuple[int, int]:
    offset = int(start)
    state = int(initial_state)
    for point in ir.scan_points or ((),):
        effective, loop_start, loop_end, loop_span, duration = _point_timing(ir, point)
        final = effective[-1]
        prefix = tuple(
            (effective[index], ir.masks[index])
            for index in range(ir.loop_start_index)
            if effective[index] < final
        )
        body = tuple(
            (effective[index], ir.masks[index])
            for index in range(ir.loop_start_index, len(ir.ticks))
            if effective[index] < loop_end and effective[index] < final
        )
        tail = tuple(
            (effective[index], ir.masks[index])
            for index in range(ir.loop_start_index, len(ir.ticks))
            if loop_end <= effective[index] < final
        )
        state = _emit_channel_assignments(sink, offset, prefix, bit, state)
        state = _emit_channel_assignments(sink, offset, body, bit, state)
        if ir.loop_count > 1:
            steady_phases, steady_state = _channel_toggle_phases(
                body,
                bit,
                state,
                origin=loop_start,
            )
            sink.add_periodic_block(
                offset + loop_start + loop_span,
                steady_phases,
                loop_span,
                ir.loop_count - 1,
            )
            state = steady_state
        tail_shift = (ir.loop_count - 1) * loop_span
        state = _emit_channel_assignments(
            sink,
            offset + tail_shift,
            tail,
            bit,
            state,
        )
        offset += duration
        sink.advance_to(offset)
    return offset, state


def _emit_channel_assignments(
    sink: _CapacityTracker | _BoundedEventCollector,
    offset: int,
    assignments: Sequence[tuple[int, int]],
    bit: int,
    initial_state: int,
) -> int:
    state = int(initial_state)
    for tick, mask in assignments:
        current = (int(mask) >> bit) & 1
        if current != state:
            sink.add_event(int(offset) + int(tick))
        state = current
    return state


def _channel_toggle_phases(
    assignments: Sequence[tuple[int, int]],
    bit: int,
    initial_state: int,
    *,
    origin: int,
) -> tuple[tuple[int, ...], int]:
    state = int(initial_state)
    phases = []
    for tick, mask in assignments:
        current = (int(mask) >> bit) & 1
        if current != state:
            phases.append(int(tick) - int(origin))
        state = current
    return tuple(phases), state


def _point_timing(
    ir: TargetIR,
    point: Sequence[int],
) -> tuple[tuple[int, ...], int, int, int, int]:
    effective = _effective_ticks(ir, point)
    loop_start = effective[ir.loop_start_index]
    loop_end = evaluate_affine_tick(
        ir.loop_end_tick,
        ir.loop_end_slot_coeffs,
        point,
        ir.scan_coeff_frac_bits,
    )
    loop_span = loop_end - loop_start
    duration = effective[-1] + (ir.loop_count - 1) * loop_span
    return effective, loop_start, loop_end, loop_span, duration


def _periodic_event_at(
    start: int,
    phases: Sequence[int],
    period: int,
    index: int,
) -> int:
    width = len(phases)
    return (
        int(start)
        + (int(index) // width) * int(period)
        + int(phases[int(index) % width])
    )


def _max_finite_periodic_window(
    phases: Sequence[int],
    period: int,
    width: int,
    repetitions: int,
) -> int:
    """Exact closed-window maximum without expanding repeated cycles."""

    ordered = tuple(sorted(int(phase) for phase in phases))
    count = int(repetitions)
    if not ordered or count <= 0:
        return 0
    cycle = int(period)
    terminal_cycle = count - 1

    def prefix(terminal: int) -> int:
        if terminal < 0:
            return 0
        complete = terminal // cycle
        if complete >= count:
            return count * len(ordered)
        return complete * len(ordered) + bisect_right(
            ordered,
            terminal - complete * cycle,
        )

    worst = 0
    for phase in ordered:
        terminal = terminal_cycle * cycle + phase
        worst = max(
            worst,
            prefix(terminal) - prefix(terminal - int(width) - 1),
        )
    return worst


def _effective_ticks(ir: TargetIR, point: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        evaluate_affine_tick(base, coefficients, point, ir.scan_coeff_frac_bits)
        for base, coefficients in zip(ir.ticks, ir.tick_slot_coeffs)
    )


__all__ = [
    "SCAN_READ_LATENCY_TICKS",
    "SLOT_MULTIPLIER_WIDTH",
    "validate_pulse_document_clock_grid",
    "validate_target_ir_for_geometry",
]
