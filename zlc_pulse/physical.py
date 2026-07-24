"""Compact pulse-owned index of the exact finite physical output waveform.

This module deliberately indexes the compiled TargetIR in fixed-width columns.
It does not construct one Python tuple or dataclass per played edge.  Consumers
ask for one small, normalized integration window; interpretation of scan points,
the compact loop, output delays, DAC register timing, and finite-DONE safety
therefore remains owned by :mod:`zlc_pulse`.

The generic digital playback projection is deliberately streaming and works for
finite or cyclic programs regardless of their DAC program.  The stricter
readout-context index below can describe held digital values and held/edge DAC
values only.  Live DAC ramps and compact repeated DAC programs are rejected by
that optional capability before allocation rather than approximated.  Adding
either readout capability requires a segment-level window value and a
corresponding artifact schema; it must not be implemented by materializing one
value per hardware tick.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from zlc_storage.canonical import (
    canonical_text as _text,
    nonnegative_integer as _nonnegative_integer,
    nonnegative_real as _nonnegative_real,
    positive_integer as _positive_integer,
    positive_real as _positive_real,
)

from .ir import TargetIR, evaluate_affine_tick


_U64_MAX = (1 << 64) - 1
_U64 = np.dtype("<u8")


class PhysicalReadoutContextUnsupportedError(ValueError):
    """The pulse is valid, but the optional readout-window index cannot model it."""


def _callback(value: object, field: str) -> Callable[[], None]:
    if not callable(value):
        raise TypeError(f"{field} must be callable")
    return value


@dataclass(frozen=True)
class PhysicalDigitalHighInterval:
    """One delayed high interval on an external logical digital output."""

    output_key: str
    lane: str
    start_tick: int
    stop_tick: int

    def __post_init__(self) -> None:
        _text(self.output_key, "output_key")
        _text(self.lane, "lane")
        start = _nonnegative_integer(self.start_tick, "start_tick")
        stop = _positive_integer(self.stop_tick, "stop_tick")
        if stop <= start:
            raise ValueError("digital high interval must have positive duration")


@dataclass(frozen=True)
class PhysicalDigitalWindow:
    """One logical digital output inside a trigger-normalized window."""

    output_key: str
    lane: str
    high_at_window_start: bool
    transitions: tuple[tuple[int, bool], ...] = ()

    def __post_init__(self) -> None:
        _text(self.output_key, "output_key")
        _text(self.lane, "lane")
        if type(self.high_at_window_start) is not bool:
            raise TypeError("high_at_window_start must be bool")
        transitions = tuple(tuple(item) for item in self.transitions)
        previous_tick = 0
        state = self.high_at_window_start
        normalized: list[tuple[int, bool]] = []
        for item in transitions:
            if len(item) != 2:
                raise ValueError("digital transitions must be (relative_tick, high) pairs")
            tick = _positive_integer(item[0], "digital relative tick")
            next_state = item[1]
            if type(next_state) is not bool:
                raise TypeError("digital transition state must be bool")
            if tick <= previous_tick:
                raise ValueError("digital transition ticks must strictly increase")
            if next_state is state:
                raise ValueError("digital window contains a redundant transition")
            normalized.append((tick, next_state))
            previous_tick = tick
            state = next_state
        object.__setattr__(self, "transitions", tuple(normalized))


@dataclass(frozen=True)
class PhysicalBusWindow:
    """One decoded held/edge DAC bus inside a normalized window."""

    bus_name: str
    value_at_window_start: int
    changes: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        _text(self.bus_name, "bus_name")
        start = _nonnegative_integer(
            self.value_at_window_start,
            "value_at_window_start",
        )
        if start > _U64_MAX:
            raise ValueError("value_at_window_start exceeds the uint64 index domain")
        previous_tick = 0
        current = start
        normalized: list[tuple[int, int]] = []
        for raw in self.changes:
            item = tuple(raw)
            if len(item) != 2:
                raise ValueError("bus changes must be (relative_tick, value) pairs")
            tick = _positive_integer(item[0], "bus relative tick")
            value = _nonnegative_integer(item[1], "bus change value")
            if value > _U64_MAX:
                raise ValueError("bus change value exceeds the uint64 index domain")
            if tick <= previous_tick:
                raise ValueError("bus change ticks must strictly increase")
            if value == current:
                raise ValueError("bus window contains a redundant change")
            normalized.append((tick, value))
            previous_tick = tick
            current = value
        object.__setattr__(self, "value_at_window_start", start)
        object.__setattr__(self, "changes", tuple(normalized))


@dataclass(frozen=True)
class PhysicalWindowProjection:
    """Exact external output waveform in one camera-integration window."""

    clock_hz: float
    target_abi_fingerprint: str
    integration_start_offset_seconds: float
    integration_seconds: float
    digital: tuple[PhysicalDigitalWindow, ...]
    buses: tuple[PhysicalBusWindow, ...]

    def __post_init__(self) -> None:
        clock = _positive_real(self.clock_hz, "clock_hz")
        target = _text(self.target_abi_fingerprint, "target_abi_fingerprint")
        offset = _nonnegative_real(
            self.integration_start_offset_seconds,
            "integration_start_offset_seconds",
        )
        duration = _positive_real(self.integration_seconds, "integration_seconds")
        digital = tuple(self.digital)
        buses = tuple(self.buses)
        if any(not isinstance(item, PhysicalDigitalWindow) for item in digital):
            raise TypeError("digital must contain PhysicalDigitalWindow values")
        if any(not isinstance(item, PhysicalBusWindow) for item in buses):
            raise TypeError("buses must contain PhysicalBusWindow values")
        if tuple(item.output_key for item in digital) != tuple(
            sorted({item.output_key for item in digital})
        ):
            raise ValueError("digital windows must have unique sorted output keys")
        if tuple(item.bus_name for item in buses) != tuple(
            sorted({item.bus_name for item in buses})
        ):
            raise ValueError("bus windows must have unique sorted names")
        object.__setattr__(self, "clock_hz", clock)
        object.__setattr__(self, "target_abi_fingerprint", target)
        object.__setattr__(self, "integration_start_offset_seconds", offset)
        object.__setattr__(self, "integration_seconds", duration)
        object.__setattr__(self, "digital", digital)
        object.__setattr__(self, "buses", buses)


@dataclass(frozen=True)
class _DigitalBinding:
    output_key: str
    lane: str
    bit_index: int
    delay_ticks: int


@dataclass(frozen=True, eq=False)
class _BusTimeline:
    bus_name: str
    safe_value: int
    ticks: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        _text(self.bus_name, "bus_name")
        safe = _nonnegative_integer(self.safe_value, "safe_value")
        if safe > _U64_MAX:
            raise ValueError("safe_value exceeds the uint64 index domain")
        for field in ("ticks", "values"):
            value = getattr(self, field)
            if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != _U64:
                raise TypeError(f"{field} must be a one-dimensional {_U64.str} ndarray")
            if value.flags.writeable or not isinstance(value.base, bytes):
                raise ValueError(f"{field} must be backed by immutable owned bytes")
        if len(self.ticks) != len(self.values):
            raise ValueError("bus timeline ticks and values must have equal length")
        if len(self.ticks) > 1 and any(
            int(right) < int(left)
            for left, right in zip(self.ticks[:-1], self.ticks[1:], strict=True)
        ):
            raise ValueError("bus timeline ticks must be nondecreasing")
        object.__setattr__(self, "safe_value", safe)


class _Checkpoint:
    __slots__ = ("_callback", "_rows")

    def __init__(self, callback: Callable[[], None] | None) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("checkpoint must be callable or None")
        self._callback = callback
        self._rows = 0

    def boundary(self) -> None:
        if self._callback is not None:
            self._callback()

    def row(self) -> None:
        self._rows += 1
        if self._callback is not None and self._rows % 4_096 == 0:
            self._callback()


class PhysicalWaveformIndex:
    """Packed, immutable index for exact finite physical-window queries."""

    __slots__ = (
        "_ir_fingerprint",
        "_clock_hz",
        "_target_abi_fingerprint",
        "_terminal_tick",
        "_digital_bindings",
        "_digital_ticks",
        "_digital_masks",
        "_bus_timelines",
    )

    def __init__(
        self,
        ir: TargetIR,
        *,
        terminal_tick: int,
        digital_ticks: np.ndarray,
        digital_masks: np.ndarray,
        bus_timelines: tuple[_BusTimeline, ...],
    ) -> None:
        if not isinstance(ir, TargetIR):
            raise TypeError("ir must be TargetIR")
        terminal = _nonnegative_integer(terminal_tick, "terminal_tick")
        if terminal > _U64_MAX:
            raise ValueError("terminal_tick exceeds the uint64 index domain")
        for field, value in (
            ("digital_ticks", digital_ticks),
            ("digital_masks", digital_masks),
        ):
            if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != _U64:
                raise TypeError(f"{field} must be a one-dimensional {_U64.str} ndarray")
            if value.flags.writeable or not isinstance(value.base, bytes):
                raise ValueError(f"{field} must be backed by immutable owned bytes")
        if len(digital_ticks) != len(digital_masks):
            raise ValueError("digital index ticks and masks must have equal length")
        if len(digital_ticks) > 1 and any(
            int(right) <= int(left)
            for left, right in zip(digital_ticks[:-1], digital_ticks[1:], strict=True)
        ):
            raise ValueError("digital index ticks must strictly increase")
        timelines = tuple(bus_timelines)
        if any(not isinstance(item, _BusTimeline) for item in timelines):
            raise TypeError("bus_timelines must contain internal bus timelines")
        if tuple(item.bus_name for item in timelines) != tuple(
            sorted({item.bus_name for item in timelines})
        ):
            raise ValueError("bus timelines must have unique sorted names")
        lane_bits = {lane: index for index, lane in enumerate(ir.channels)}
        bindings = tuple(
            _DigitalBinding(
                output_key,
                lane,
                lane_bits[lane],
                ir.channel_delays[lane_bits[lane]],
            )
            for output_key, lane in ir.logical_digital_outputs
        )
        object.__setattr__(self, "_ir_fingerprint", ir.fingerprint)
        object.__setattr__(self, "_clock_hz", ir.clock_hz)
        object.__setattr__(self, "_target_abi_fingerprint", ir.target_abi_fingerprint)
        object.__setattr__(self, "_terminal_tick", terminal)
        object.__setattr__(self, "_digital_bindings", bindings)
        object.__setattr__(self, "_digital_ticks", digital_ticks)
        object.__setattr__(self, "_digital_masks", digital_masks)
        object.__setattr__(self, "_bus_timelines", timelines)

    @property
    def source_ir_fingerprint(self) -> str:
        return self._ir_fingerprint

    @property
    def terminal_tick(self) -> int:
        return self._terminal_tick

    def iter_digital_high_intervals(
        self,
    ) -> Iterator[PhysicalDigitalHighInterval]:
        """Yield delayed high intervals without materializing transition rows."""

        for binding in self._digital_bindings:
            active: int | None = None
            previous = False
            for raw_tick, raw_mask in zip(
                self._digital_ticks,
                self._digital_masks,
                strict=True,
            ):
                tick = int(raw_tick) + binding.delay_ticks
                current = bool((int(raw_mask) >> binding.bit_index) & 1)
                if current and not previous:
                    active = tick
                elif previous and not current:
                    if active is None or tick <= active:
                        raise ValueError(
                            "compiled digital waveform contains an invalid high interval"
                        )
                    yield PhysicalDigitalHighInterval(
                        binding.output_key,
                        binding.lane,
                        active,
                        tick,
                    )
                    active = None
                previous = current
            if active is not None:
                raise ValueError("finite digital waveform has no terminal safe edge")

    def window(
        self,
        anchor_tick: int,
        *,
        integration_start_offset_seconds: float,
        integration_seconds: float,
        exclude_output_key: str,
        checkpoint: Callable[[], None],
    ) -> PhysicalWindowProjection:
        """Project one strict ``[start, stop)`` integration window.

        ``anchor_tick`` is already the physically delayed trigger rising edge.
        The trigger output is excluded because EDGE-trigger pulse width is an
        anchor, not a measured physical condition.
        """

        anchor = _nonnegative_integer(anchor_tick, "anchor_tick")
        offset = _nonnegative_real(
            integration_start_offset_seconds,
            "integration_start_offset_seconds",
        )
        duration = _positive_real(integration_seconds, "integration_seconds")
        excluded = _text(exclude_output_key, "exclude_output_key")
        callback = _callback(checkpoint, "checkpoint")
        matches = tuple(
            item for item in self._digital_bindings if item.output_key == excluded
        )
        if len(matches) != 1:
            raise ValueError("exclude_output_key must name exactly one logical output")
        window_start = anchor + offset * self._clock_hz
        window_stop = window_start + duration * self._clock_hz
        if not math.isfinite(window_start) or not math.isfinite(window_stop):
            raise ValueError("physical integration window exceeds the finite time domain")
        callback()

        digital: list[PhysicalDigitalWindow] = []
        for binding in self._digital_bindings:
            callback()
            if binding.output_key == excluded:
                continue
            source_start = window_start - binding.delay_ticks
            source_stop = window_stop - binding.delay_ticks
            prior = _search_right(self._digital_ticks, source_start) - 1
            state = (
                bool((int(self._digital_masks[prior]) >> binding.bit_index) & 1)
                if prior >= 0
                else False
            )
            changes: list[tuple[int, bool]] = []
            start = prior + 1
            stop = _search_left(self._digital_ticks, source_stop)
            for index in range(start, stop):
                if index and index % 4_096 == 0:
                    callback()
                next_state = bool(
                    (int(self._digital_masks[index]) >> binding.bit_index) & 1
                )
                if next_state is state:
                    continue
                relative_tick = (
                    int(self._digital_ticks[index])
                    + binding.delay_ticks
                    - anchor
                )
                changes.append((relative_tick, next_state))
                state = next_state
            digital.append(
                PhysicalDigitalWindow(
                    binding.output_key,
                    binding.lane,
                    bool(
                        (int(self._digital_masks[prior]) >> binding.bit_index) & 1
                    )
                    if prior >= 0
                    else False,
                    tuple(changes),
                )
            )

        buses: list[PhysicalBusWindow] = []
        for timeline in self._bus_timelines:
            callback()
            prior = _search_right(timeline.ticks, window_start) - 1
            initial = int(timeline.values[prior]) if prior >= 0 else timeline.safe_value
            current = initial
            changes: list[tuple[int, int]] = []
            start = prior + 1
            stop = _search_left(timeline.ticks, window_stop)
            for index in range(start, stop):
                if index and index % 4_096 == 0:
                    callback()
                next_value = int(timeline.values[index])
                if next_value == current:
                    continue
                changes.append((int(timeline.ticks[index]) - anchor, next_value))
                current = next_value
            buses.append(
                PhysicalBusWindow(
                    timeline.bus_name,
                    initial,
                    tuple(changes),
                )
            )
        callback()
        return PhysicalWindowProjection(
            self._clock_hz,
            self._target_abi_fingerprint,
            offset,
            duration,
            tuple(sorted(digital, key=lambda item: item.output_key)),
            tuple(sorted(buses, key=lambda item: item.bus_name)),
        )


def physical_digital_playback_terminal_tick(ir: TargetIR) -> int:
    """Return the exact logical terminal of one finite pass/base cycle.

    This is intentionally independent of the readout-context index.  It is
    valid for cyclic programs, arbitrary-width digital masks, DAC ramps, and
    compact repeated DAC programs because none of those alter the digital
    TargetIR schedule owned here.
    """

    if not isinstance(ir, TargetIR):
        raise TypeError("ir must be TargetIR")
    terminal = 0
    for point in ir.scan_points or ((),):
        effective_final = evaluate_affine_tick(
            ir.ticks[-1],
            ir.tick_slot_coeffs[-1],
            point,
            ir.scan_coeff_frac_bits,
        )
        loop_start = evaluate_affine_tick(
            ir.ticks[ir.loop_start_index],
            ir.tick_slot_coeffs[ir.loop_start_index],
            point,
            ir.scan_coeff_frac_bits,
        )
        loop_end = evaluate_affine_tick(
            ir.loop_end_tick,
            ir.loop_end_slot_coeffs,
            point,
            ir.scan_coeff_frac_bits,
        )
        terminal += effective_final + (ir.loop_count - 1) * (
            loop_end - loop_start
        )
    return terminal


def iter_physical_digital_high_intervals(
    ir: TargetIR,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> Iterator[PhysicalDigitalHighInterval]:
    """Stream exact delayed digital high intervals without an assignment graph.

    One compact state cell is retained per declared logical digital output.
    Scan points and finite inner loops are interpreted by the same pulse-owned
    assignment iterator as the readout index.  For ``repeat_forever`` this
    yields one exact base cycle; the caller keeps the cyclic execution flag and
    repeats it at :func:`physical_digital_playback_terminal_tick`.

    DAC structure is intentionally irrelevant here: a ramp or compact repeated
    DAC program must not prevent a digital-only adapter from playing the pulse.
    """

    if not isinstance(ir, TargetIR):
        raise TypeError("ir must be TargetIR")
    counter = _Checkpoint(checkpoint)
    counter.boundary()
    lane_bits = {lane: index for index, lane in enumerate(ir.channels)}
    bindings = tuple(
        _DigitalBinding(
            output_key,
            lane,
            lane_bits[lane],
            ir.channel_delays[lane_bits[lane]],
        )
        for output_key, lane in ir.logical_digital_outputs
    )
    if not bindings:
        counter.boundary()
        return
    active: list[int | None] = [None] * len(bindings)
    states = [False] * len(bindings)
    for tick, mask in _iter_digital_mask_assignments(ir, counter):
        for index, binding in enumerate(bindings):
            current = bool((int(mask) >> binding.bit_index) & 1)
            if current == states[index]:
                continue
            shifted = int(tick) + binding.delay_ticks
            if current:
                active[index] = shifted
            else:
                start = active[index]
                if start is None or shifted <= start:
                    raise ValueError(
                        "compiled digital waveform contains an invalid high interval"
                    )
                yield PhysicalDigitalHighInterval(
                    binding.output_key,
                    binding.lane,
                    start,
                    shifted,
                )
                active[index] = None
            states[index] = current
    if any(value is not None for value in active) or any(states):
        raise ValueError("digital waveform has no terminal safe edge")
    counter.boundary()


def build_physical_waveform_index(
    ir: TargetIR,
    *,
    checkpoint: Callable[[], None],
) -> PhysicalWaveformIndex:
    """Build an immutable packed index with cancellation checkpoints."""

    if not isinstance(ir, TargetIR):
        raise TypeError("ir must be TargetIR")
    callback = _callback(checkpoint, "checkpoint")
    callback()
    digital_rows = _digital_index_row_count(ir, checkpoint=callback)

    digital_ticks_mutable = np.empty(digital_rows, dtype=_U64)
    digital_masks_mutable = np.empty(digital_rows, dtype=_U64)
    terminal_tick = 0
    written = 0
    counter = _Checkpoint(callback)
    for tick, mask in _iter_digital_mask_assignments(ir, counter):
        digital_ticks_mutable[written] = tick
        digital_masks_mutable[written] = mask
        terminal_tick = tick
        written += 1
    if written != digital_rows:
        raise RuntimeError("physical digital counting/build passes diverged")
    digital_ticks = _freeze_u64(digital_ticks_mutable)
    digital_masks = _freeze_u64(digital_masks_mutable)
    del digital_ticks_mutable, digital_masks_mutable

    points = ir.scan_points or ((),)
    delay_by_bus = {item.bus_index: item.delay_ticks for item in ir.bus_delays}
    timelines: list[_BusTimeline] = []
    for bus_index, bus_name in enumerate(ir.bus_names):
        callback()
        segments = tuple(
            item for item in ir.bus_segments if item.bus_index == bus_index
        )
        row_count = len(points) * len(segments) + (1 if segments else 0)
        ticks_mutable = np.empty(row_count, dtype=_U64)
        values_mutable = np.empty(row_count, dtype=_U64)
        row = 0
        run_offset = 0
        delay = delay_by_bus.get(bus_index, 0)
        for point in points:
            callback()
            final_tick = evaluate_affine_tick(
                ir.ticks[-1],
                ir.tick_slot_coeffs[-1],
                point,
                ir.scan_coeff_frac_bits,
            )
            for segment in segments:
                source_tick = evaluate_affine_tick(
                    segment.start_tick,
                    segment.start_tick_coeffs,
                    point,
                    ir.scan_coeff_frac_bits,
                )
                selector = segment.stop_value_select
                value = (
                    int(point[selector - 1])
                    if selector
                    else int(segment.stop_value)
                )
                visible_tick = (
                    run_offset
                    + source_tick
                    + delay
                    + (1 if source_tick != 0 else 0)
                )
                ticks_mutable[row] = _u64(visible_tick, "physical bus tick")
                values_mutable[row] = _u64(value, "physical bus value")
                row += 1
                counter.row()
            run_offset += final_tick
        if segments:
            ticks_mutable[row] = _u64(
                run_offset + delay + 1,
                "finite-DONE physical bus tick",
            )
            values_mutable[row] = _u64(
                ir.bus_safe_values[bus_index],
                "physical bus safe value",
            )
            row += 1
        if row != row_count:
            raise RuntimeError("physical bus counting/build passes diverged")
        timelines.append(
            _BusTimeline(
                bus_name,
                ir.bus_safe_values[bus_index],
                _freeze_u64(ticks_mutable),
                _freeze_u64(values_mutable),
            )
        )
        del ticks_mutable, values_mutable
    callback()
    return PhysicalWaveformIndex(
        ir,
        terminal_tick=terminal_tick,
        digital_ticks=digital_ticks,
        digital_masks=digital_masks,
        bus_timelines=tuple(sorted(timelines, key=lambda item: item.bus_name)),
    )


def _digital_index_row_count(
    ir: TargetIR,
    *,
    checkpoint: Callable[[], None] | None,
) -> int:
    _validate_supported_domain(ir)
    counter = _Checkpoint(checkpoint)
    counter.boundary()
    digital_rows = _digital_assignment_count(
        ir,
        checkpoint=counter.boundary,
    )
    counter.boundary()
    return digital_rows


def _validate_supported_domain(ir: TargetIR) -> None:
    if not isinstance(ir, TargetIR):
        raise TypeError("ir must be TargetIR")
    if ir.repeat_forever:
        raise PhysicalReadoutContextUnsupportedError(
            "a cyclic TargetIR has no finite physical waveform index"
        )
    if len(ir.channels) > 64 or any(mask > _U64_MAX for mask in ir.masks):
        raise PhysicalReadoutContextUnsupportedError(
            "physical waveform masks exceed the uint64 packed domain"
        )
    if any(segment.mode != "edge" for segment in ir.bus_segments):
        raise PhysicalReadoutContextUnsupportedError(
            "physical readout context cannot represent a live-state DAC ramp exactly"
        )
    if ir.bus_segments and ir.loop_count != 1:
        raise PhysicalReadoutContextUnsupportedError(
            "physical readout context cannot represent compact repeated DAC updates exactly"
        )
    for value in ir.bus_safe_values:
        _u64(value, "physical bus safe value")
    for segment in ir.bus_segments:
        if not segment.stop_value_select:
            _u64(segment.stop_value, "physical bus literal value")


def _iter_digital_mask_assignments(
    ir: TargetIR,
    checkpoint: _Checkpoint,
) -> Iterator[tuple[int, int]]:
    run_offset = 0
    points = ir.scan_points or ((),)
    for point in points:
        checkpoint.boundary()
        effective = tuple(
            evaluate_affine_tick(base, coeffs, point, ir.scan_coeff_frac_bits)
            for base, coeffs in zip(ir.ticks, ir.tick_slot_coeffs, strict=True)
        )
        final_tick = effective[-1]
        loop_start_tick = effective[ir.loop_start_index]
        loop_end_tick = evaluate_affine_tick(
            ir.loop_end_tick,
            ir.loop_end_slot_coeffs,
            point,
            ir.scan_coeff_frac_bits,
        )
        loop_span = loop_end_tick - loop_start_tick
        body_stop = ir.loop_start_index
        terminal_index = len(effective) - 1
        while body_stop < terminal_index and effective[body_stop] < loop_end_tick:
            body_stop += 1
        for index in range(ir.loop_start_index):
            tick = _u64(run_offset + effective[index], "physical digital tick")
            yield tick, int(ir.masks[index])
            checkpoint.row()
        if body_stop > ir.loop_start_index:
            for iteration in range(ir.loop_count):
                shift = iteration * loop_span
                for index in range(ir.loop_start_index, body_stop):
                    tick = _u64(
                        run_offset + effective[index] + shift,
                        "physical digital tick",
                    )
                    yield tick, int(ir.masks[index])
                    checkpoint.row()
        tail_shift = (ir.loop_count - 1) * loop_span
        for index in range(body_stop, terminal_index):
            tick = _u64(
                run_offset + effective[index] + tail_shift,
                "physical digital tick",
            )
            yield tick, int(ir.masks[index])
            checkpoint.row()
        run_offset += final_tick + tail_shift
    yield _u64(run_offset, "physical waveform terminal tick"), 0
    checkpoint.row()


def _digital_assignment_count(
    ir: TargetIR,
    *,
    checkpoint: Callable[[], None] | None,
) -> int:
    """Count expanded digital rows without iterating the hardware loop count."""

    if not isinstance(ir, TargetIR):
        raise TypeError("ir must be TargetIR")
    counter = _Checkpoint(checkpoint)
    terminal_index = len(ir.ticks) - 1
    total = 1  # One final safe assignment after every scan point.
    for point in ir.scan_points or ((),):
        counter.boundary()
        effective = tuple(
            evaluate_affine_tick(base, coeffs, point, ir.scan_coeff_frac_bits)
            for base, coeffs in zip(ir.ticks, ir.tick_slot_coeffs, strict=True)
        )
        loop_end_tick = evaluate_affine_tick(
            ir.loop_end_tick,
            ir.loop_end_slot_coeffs,
            point,
            ir.scan_coeff_frac_bits,
        )
        body_stop = ir.loop_start_index
        while body_stop < terminal_index and effective[body_stop] < loop_end_tick:
            body_stop += 1
        prefix_rows = ir.loop_start_index
        body_rows = body_stop - ir.loop_start_index
        tail_rows = terminal_index - body_stop
        total += prefix_rows + body_rows * ir.loop_count + tail_rows
    counter.boundary()
    return total


def _freeze_u64(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != _U64:
        raise TypeError("packed physical column must be a one-dimensional <u8 ndarray")
    return np.frombuffer(value.tobytes(order="C"), dtype=_U64)


def _u64(value: object, field: str) -> int:
    result = _nonnegative_integer(value, field)
    if result > _U64_MAX:
        raise ValueError(f"{field} exceeds the uint64 packed domain")
    return result


def _search_right(values: np.ndarray, boundary: float | int) -> int:
    """Bisect uint64 values without coercing a large boundary through float64."""

    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if int(values[middle]) <= boundary:
            low = middle + 1
        else:
            high = middle
    return low


def _search_left(values: np.ndarray, boundary: float | int) -> int:
    """Bisect uint64 values without coercing a large boundary through float64."""

    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if int(values[middle]) < boundary:
            low = middle + 1
        else:
            high = middle
    return low


__all__ = [
    "PhysicalBusWindow",
    "PhysicalDigitalHighInterval",
    "PhysicalDigitalWindow",
    "PhysicalReadoutContextUnsupportedError",
    "PhysicalWaveformIndex",
    "PhysicalWindowProjection",
    "build_physical_waveform_index",
    "iter_physical_digital_high_intervals",
    "physical_digital_playback_terminal_tick",
]
