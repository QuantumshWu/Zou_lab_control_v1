"""Property checks for the compressed frozen-RTL FIFO capacity oracle."""

from __future__ import annotations

import random

import pytest

from fpga.pulse_streamer.host.image import StreamerParams
from zlc_pulse import TargetBusDelay, TargetBusSegment, TargetIR
from zlc_pulse.validation import (
    _CapacityExceeded,
    _CapacityTracker,
    _max_finite_periodic_window,
    validate_target_ir_for_geometry,
)


def _brute_window(events, width):
    ordered = sorted(events)
    return max(
        (sum(terminal - width <= event <= terminal for event in ordered)
         for terminal in ordered),
        default=0,
    )


def test_compressed_finite_tracker_matches_expanded_random_streams():
    assert _max_finite_periodic_window((0,), 7, 447, (1 << 32) - 1) == 64
    assert _max_finite_periodic_window((0,), 7, 448, (1 << 32) - 1) == 65
    randomizer = random.Random(0xC0DE)
    for _ in range(500):
        width = randomizer.randrange(0, 60)
        period = randomizer.randrange(1, 16)
        phases = tuple(
            sorted(
                randomizer.randrange(period)
                for _ in range(randomizer.randrange(0, 8))
            )
        )
        repetitions = randomizer.randrange(0, 20)
        prior = tuple(
            sorted(
                set(
                    randomizer.randrange(0, 10)
                    for _ in range(randomizer.randrange(0, 5))
                )
            )
        )
        start = 10
        expanded_block = tuple(
            start + cycle * period + phase
            for cycle in range(repetitions)
            for phase in phases
        )
        tail = tuple(
            sorted(
                start
                + repetitions * period
                + randomizer.randrange(0, 10)
                for _ in range(randomizer.randrange(0, 4))
            )
        )
        expanded = (*prior, *expanded_block, *tail)
        expected = _brute_window(expanded, width)
        assert _max_finite_periodic_window(
            phases,
            period,
            width,
            repetitions,
        ) == _brute_window(
            tuple(
                cycle * period + phase
                for cycle in range(repetitions)
                for phase in phases
            ),
            width,
        )

        tracker = _CapacityTracker(width=width, limit=expected)
        for event in prior:
            tracker.add_event(event)
        tracker.add_periodic_block(start, phases, period, repetitions)
        for event in tail:
            tracker.add_event(event)

        if expected:
            tracker = _CapacityTracker(width=width, limit=expected - 1)
            with pytest.raises(_CapacityExceeded):
                for event in prior:
                    tracker.add_event(event)
                tracker.add_periodic_block(start, phases, period, repetitions)
                for event in tail:
                    tracker.add_event(event)


def _one_static_sweep(ir):
    final = ir.ticks[-1]
    loop_start = ir.ticks[ir.loop_start_index]
    loop_end = ir.loop_end_tick
    loop_span = loop_end - loop_start
    assignments = [
        (ir.ticks[index], ir.masks[index])
        for index in range(ir.loop_start_index)
        if ir.ticks[index] < final
    ]
    body = [
        (ir.ticks[index], ir.masks[index])
        for index in range(ir.loop_start_index, len(ir.ticks))
        if ir.ticks[index] < loop_end and ir.ticks[index] < final
    ]
    for iteration in range(ir.loop_count):
        assignments.extend(
            (tick + iteration * loop_span, mask) for tick, mask in body
        )
    tail_shift = (ir.loop_count - 1) * loop_span
    assignments.extend(
        (ir.ticks[index] + tail_shift, ir.masks[index])
        for index in range(ir.loop_start_index, len(ir.ticks))
        if loop_end <= ir.ticks[index] < final
    )
    return assignments, final + tail_shift


def _brute_channel_capacity(ir, delay):
    sweep, period = _one_static_sweep(ir)
    repetitions = delay // period + 3 if ir.repeat_forever else 1
    state = 0
    toggles = []
    for repetition in range(repetitions):
        for tick, mask in sweep:
            current = mask & 1
            if current != state:
                toggles.append(repetition * period + tick)
            state = current
    if not ir.repeat_forever and state:
        toggles.append(period)
    return _brute_window(toggles, delay - 1)


def _brute_bus_capacity(ir, delay):
    starts = tuple(segment.start_tick for segment in ir.bus_segments)
    loop_span = ir.loop_end_tick
    one = []
    if ir.loop_count <= 1:
        one.extend(starts)
    else:
        for tick in starts:
            if tick < ir.loop_end_tick:
                one.extend(
                    tick + iteration * loop_span
                    for iteration in range(ir.loop_count)
                )
            else:
                one.append(tick + (ir.loop_count - 1) * loop_span)
    period = ir.ticks[-1] + (ir.loop_count - 1) * loop_span
    repetitions = delay // period + 3 if ir.repeat_forever else 1
    events = [
        repetition * period + tick
        for repetition in range(repetitions)
        for tick in one
    ]
    if not ir.repeat_forever:
        events.append(period)
    return _brute_window(events, delay - 1)


def test_capacity_gate_matches_independent_small_ir_expansion():
    randomizer = random.Random(0xF1F0)
    geometry = StreamerParams(evt_fifo_depth=4, bus_evt_fifo_depth=4)
    for case in range(250):
        final = randomizer.randrange(3, 25)
        inner = sorted(
            set(
                randomizer.randrange(1, final)
                for _ in range(randomizer.randrange(0, 7))
            )
        )
        ticks = (0, *inner, final)
        loop_start_index = randomizer.randrange(0, len(ticks) - 1)
        loop_end = randomizer.randrange(ticks[loop_start_index] + 1, final + 1)
        loop_count = randomizer.randrange(1, 9)
        repeat_forever = bool(randomizer.randrange(2))
        delay = randomizer.randrange(2, 80)
        masks = tuple(randomizer.randrange(2) for _ in ticks[:-1]) + (0,)
        wall = final + (loop_count - 1) * (
            loop_end - ticks[loop_start_index]
        )
        channel_ir = TargetIR(
            clock_hz=50e6,
            target_abi_fingerprint=f"{case:064x}",
            channels=("trigger",),
            ticks=ticks,
            masks=masks,
            duration_seconds=wall / 50e6,
            repeat_forever=repeat_forever,
            loop_start_index=loop_start_index,
            loop_end_tick=loop_end,
            loop_count=loop_count,
            tick_slot_coeffs=tuple(() for _ in ticks),
            channel_delays=(delay,),
        )
        expected = _brute_channel_capacity(channel_ir, delay) <= 4
        try:
            validate_target_ir_for_geometry(channel_ir, geometry)
            accepted = True
        except ValueError as error:
            assert "in flight" in str(error)
            accepted = False
        assert accepted is expected

        starts = tuple(
            sorted(
                set(
                    randomizer.randrange(0, final)
                    for _ in range(randomizer.randrange(0, 7))
                )
            )
        )
        bus_loop_end = randomizer.randrange(1, final + 1)
        bus_loop_count = randomizer.randrange(1, 9)
        wall = final + (bus_loop_count - 1) * bus_loop_end
        bus_ir = TargetIR(
            clock_hz=50e6,
            target_abi_fingerprint=f"{case + 1000:064x}",
            channels=("trigger",),
            ticks=(0, final),
            masks=(0, 0),
            duration_seconds=wall / 50e6,
            repeat_forever=repeat_forever,
            loop_start_index=0,
            loop_end_tick=bus_loop_end,
            loop_count=bus_loop_count,
            tick_slot_coeffs=((), ()),
            bus_names=("dac",),
            bus_segments=tuple(
                TargetBusSegment(
                    0,
                    "dac",
                    tick,
                    tick,
                    0,
                    0,
                    "edge",
                    0,
                    0,
                    (),
                    (),
                )
                for tick in starts
            ),
            bus_delays=(TargetBusDelay(0, delay),),
            channel_delays=(0,),
        )
        expected = _brute_bus_capacity(bus_ir, delay) <= 4
        try:
            validate_target_ir_for_geometry(bus_ir, geometry)
            accepted = True
        except ValueError as error:
            assert "in flight" in str(error)
            accepted = False
        assert accepted is expected
