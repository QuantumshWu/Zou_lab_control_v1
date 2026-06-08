"""Sequencer devices and the runtime pulse-table service boundary."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from ..core.analysis import positive_int
from .base import SequencerDevice
from ..timing import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS,
    PulseSequence,
    PulseTableState,
    affine_coeffs,
    channel_names,
    count_trigger_pulses,
    positive_float,
    sequence_for_frame_count,
    slot_var,
)
from ..timing.pulse_table import UNITS_TO_NS
from ..timing.verilog import VerilogBuild, VerilogFiles, generate_verilog, write_verilog_bundle


DEFAULT_RUNTIME_CLOCK_HZ = 50_000_000.0
DEFAULT_RUNTIME_BUS_NAMES = ("da_dipole", "da_bias_y", "da_bias_x", "da_bias_z")
BUS_SEGMENT_MODES = {"edge": 1, "ramp": 2}


@dataclass(frozen=True)
class RuntimeDelayInterval:
    """One undelayed ON interval ``[start_tick, stop_tick)`` of a delayed channel,
    affine in the scanned DURATION slots (``effective = start_tick + (sum coeff_j*
    slot_j) >> frac``) so the interval moves with a scanned duration -- exactly like a
    bus segment's start/stop ticks."""

    start_tick: int
    stop_tick: int
    start_tick_coeffs: list[int] | None = None
    stop_tick_coeffs: list[int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "start_tick": int(self.start_tick),
            "stop_tick": int(self.stop_tick),
            "start_tick_coeffs": list(self.start_tick_coeffs or []),
            "stop_tick_coeffs": list(self.stop_tick_coeffs or []),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeDelayInterval":
        return cls(
            start_tick=int(payload.get("start_tick", 0)),
            stop_tick=int(payload.get("stop_tick", payload.get("start_tick", 0))),
            start_tick_coeffs=[int(v) for v in payload.get("start_tick_coeffs", [])] or None,
            stop_tick_coeffs=[int(v) for v in payload.get("stop_tick_coeffs", [])] or None,
        )


@dataclass(frozen=True)
class RuntimeDelayChannel:
    """One delayed output channel for the UNBOUNDED membership delay player.

    ``bit`` is the output channel bit; ``delay`` is the physical delay ``d`` in ticks
    (>= 0 after the host folds the global negative-delay shift G); ``intervals`` are
    the channel's OWN undelayed ON intervals over ``[0, T)``.  The engine stores the
    intervals in a per-channel LUTRAM and produces the delayed bit by evaluating
    membership at the shifted phase ``(time_count - (d mod T)) mod T`` (skip =
    floor(d/T) whole periods gate the startup) -- NO buffer, so T and d are unbounded."""

    bit: int
    delay: int
    intervals: list[RuntimeDelayInterval] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "bit": int(self.bit),
            "delay": int(self.delay),
            "intervals": [iv.to_dict() for iv in (self.intervals or [])],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeDelayChannel":
        return cls(
            bit=int(payload.get("bit", 0)),
            delay=int(payload.get("delay", 0)),
            intervals=[RuntimeDelayInterval.from_dict(iv) for iv in (payload.get("intervals") or [])] or None,
        )


@dataclass(frozen=True)
class RuntimeBusSegment:
    """One runtime analog-bus segment uploaded beside the digital edge table."""

    bus_index: int
    start_tick: int
    stop_tick: int
    start_value: int
    stop_value: int
    mode: str = "edge"
    bus_name: str = ""
    value_select: int = 0
    """START-endpoint scan-slot select.  0 = use ``start_value``; ``j+1`` = read the
    DAC code from scan slot ``j`` at runtime.  For edge/hold segments (start==stop)
    this is THE held-value select."""
    start_tick_coeffs: list[int] | None = None
    stop_tick_coeffs: list[int] | None = None
    """Per-slot affine coefficients for the segment's start/stop tick.  The FPGA
    computes ``effective_tick = start_tick + (sum coeff_j*slot_j) >> frac`` so a
    scanned duration/delay moves the segment in lockstep with the digital edges
    -- this is what lets DAC value + duration + delay scan simultaneously."""
    stop_value_select: int = 0
    """STOP-endpoint scan-slot select (``j+1`` = stop value reads slot ``j``).  For
    edge/hold segments it equals ``value_select`` (start==stop).  Independent from
    ``value_select`` so a RAMP can scan BOTH endpoints: ramp scanned-A -> scanned-B."""

    def to_dict(self) -> dict[str, object]:
        return {
            "bus_index": int(self.bus_index),
            "bus_name": str(self.bus_name),
            "start_tick": int(self.start_tick),
            "stop_tick": int(self.stop_tick),
            "start_value": int(self.start_value),
            "stop_value": int(self.stop_value),
            "mode": str(self.mode),
            "value_select": int(self.value_select),
            "stop_value_select": int(self.stop_value_select),
            "start_tick_coeffs": list(self.start_tick_coeffs or []),
            "stop_tick_coeffs": list(self.stop_tick_coeffs or []),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeBusSegment":
        return cls(
            bus_index=int(payload.get("bus_index", 0)),
            bus_name=str(payload.get("bus_name", "")),
            start_tick=int(payload.get("start_tick", 0)),
            stop_tick=int(payload.get("stop_tick", payload.get("start_tick", 0))),
            start_value=int(payload.get("start_value", payload.get("stop_value", 0))),
            stop_value=int(payload.get("stop_value", payload.get("start_value", 0))),
            mode=str(payload.get("mode", "edge")).strip().lower(),
            value_select=int(payload.get("value_select", 0)),
            # stop select defaults to the start select so an edge/hold held-value scan
            # (start==stop) stays correct when only value_select is given.
            stop_value_select=int(payload.get("stop_value_select", payload.get("value_select", 0))),
            start_tick_coeffs=[int(v) for v in payload.get("start_tick_coeffs", [])] or None,
            stop_tick_coeffs=[int(v) for v in payload.get("stop_tick_coeffs", [])] or None,
        )


@dataclass(frozen=True)
class RuntimeSequenceProgram:
    """Runtime edge-table program uploaded to a pulse-streamer-like FPGA."""

    sequence_id: str
    sequence_name: str
    clock_hz: float
    channels: list[str]
    ticks: list[int]
    masks: list[int]
    duration: float
    trigger_count: int
    source_sequence: dict[str, Any] | None = None
    source_table: dict[str, Any] | None = None
    repeat_forever: bool = False
    loop_start_index: int = 0
    loop_end_tick: int = 0
    loop_count: int = 1
    repeat_from_index: int = 0
    slot_count: int = 0
    slot_kinds: list[str] | None = None
    loop_end_slot_coeffs: list[int] | None = None
    tick_slot_coeffs: list[list[int]] | None = None
    scan_points: list[list[int]] | None = None
    scan_point_durations: list[float] | None = None
    scan_coeff_frac_bits: int = 8
    bus_names: list[str] | None = None
    bus_segments: list[RuntimeBusSegment] | None = None
    # PHYSICAL CHANNEL DELAY: per-channel-bit delay in ticks, applied to the engine OUTPUT
    # (a delay line), NOT baked into ``ticks``.  ``ticks``/``masks`` are the UNDELAYED frame;
    # the engine delays bit ``b`` by ``channel_delays[b]`` (startup counter + sub-period phase
    # shift -- see engine_model.membership_delay_play).  Any length; never disturbs another
    # channel; first frame real.  Empty/None = no channel delayed.  ``channel_delays`` keeps
    # the delay-per-bit (the source of d); ``delay_channels`` carries the same delays PLUS each
    # delayed channel's ON intervals -- what the UNBOUNDED membership player needs.
    channel_delays: list[int] | None = None
    delay_channels: list[RuntimeDelayChannel] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema": "Zou_lab_control.neutral_atom.RuntimeSequenceProgram",
            "version": 2,
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "clock_hz": self.clock_hz,
            "channels": list(self.channels),
            "ticks": list(self.ticks),
            "masks": list(self.masks),
            "duration": self.duration,
            "trigger_count": self.trigger_count,
            "repeat_forever": bool(self.repeat_forever),
            "loop_start_index": int(self.loop_start_index),
            "loop_end_tick": int(self.loop_end_tick),
            "loop_count": int(self.loop_count),
            "repeat_from_index": int(self.repeat_from_index),
            "slot_count": int(self.slot_count),
            "slot_kinds": list(self.slot_kinds or []),
            "loop_end_slot_coeffs": list(self.loop_end_slot_coeffs or [0] * int(self.slot_count)),
            "tick_slot_coeffs": [list(row) for row in (self.tick_slot_coeffs or [[0] * int(self.slot_count) for _ in self.ticks])],
            "scan_points": [list(point) for point in (self.scan_points or [])],
            "scan_point_durations": list(self.scan_point_durations or []),
            "scan_coeff_frac_bits": int(self.scan_coeff_frac_bits),
            "bus_names": list(self.bus_names or []),
            "bus_segments": [segment.to_dict() for segment in (self.bus_segments or [])],
            "channel_delays": list(self.channel_delays or []),
            "delay_channels": [dc.to_dict() for dc in (self.delay_channels or [])],
        }
        if self.source_sequence is not None:
            payload["source_sequence"] = self.source_sequence
        if self.source_table is not None:
            payload["source_table"] = self.source_table
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeSequenceProgram":
        if payload.get("schema") != "Zou_lab_control.neutral_atom.RuntimeSequenceProgram":
            raise ValueError("unsupported runtime sequence program schema.")
        slot_count = int(payload.get("slot_count", 0))
        tick_slot_coeffs = [[int(v) for v in row] for row in payload.get("tick_slot_coeffs", [])]
        if tick_slot_coeffs and not any(any(row) for row in tick_slot_coeffs):
            tick_slot_coeffs = []
        return cls(
            sequence_id=str(payload["sequence_id"]),
            sequence_name=str(payload["sequence_name"]),
            clock_hz=positive_float(payload["clock_hz"], "clock_hz"),
            channels=list(channel_names(payload["channels"], "channels")),
            ticks=[int(v) for v in payload["ticks"]],
            masks=[int(v) for v in payload["masks"]],
            duration=float(payload["duration"]),
            trigger_count=int(payload["trigger_count"]),
            source_sequence=None if payload.get("source_sequence") is None else dict(payload["source_sequence"]),
            source_table=None if payload.get("source_table") is None else dict(payload["source_table"]),
            repeat_forever=bool(payload.get("repeat_forever", False)),
            loop_start_index=int(payload.get("loop_start_index", 0)),
            loop_end_tick=int(payload.get("loop_end_tick", 0)),
            loop_count=int(payload.get("loop_count", 1)),
            repeat_from_index=int(payload.get("repeat_from_index", 0)),
            slot_count=slot_count,
            slot_kinds=[str(v) for v in payload.get("slot_kinds", [])] or None,
            loop_end_slot_coeffs=[int(v) for v in payload.get("loop_end_slot_coeffs", [])] or None,
            tick_slot_coeffs=tick_slot_coeffs or None,
            scan_points=[[int(v) for v in item] for item in payload.get("scan_points", [])] or None,
            scan_point_durations=[float(v) for v in payload.get("scan_point_durations", [])] or None,
            scan_coeff_frac_bits=int(payload.get("scan_coeff_frac_bits", 8)),
            bus_names=[str(item) for item in payload.get("bus_names", [])] or None,
            bus_segments=[RuntimeBusSegment.from_dict(item) for item in payload.get("bus_segments", [])] or None,
            channel_delays=[int(v) for v in payload.get("channel_delays", [])] or None,
            delay_channels=[RuntimeDelayChannel.from_dict(item) for item in payload.get("delay_channels", [])] or None,
        )

    @property
    def scan_enabled(self) -> bool:
        return bool(self.scan_points)


def compile_runtime_program(
    sequence: PulseSequence,
    *,
    channels: Sequence[str],
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> RuntimeSequenceProgram:
    """Compile a ``PulseSequence`` into an uploadable edge table."""

    channels = list(channel_names(channels, "channels"))
    clock_hz = positive_float(clock_hz, "clock_hz")
    base_sequence = sequence.without_repeat()
    ticks, masks, channels = base_sequence.edges(clock_hz=clock_hz, channels=channels)
    repeat_period = sequence.repeat_period or base_sequence.duration
    loop_end_tick = _time_to_ticks(repeat_period, clock_hz, "repeat_period") if repeat_period > 0 else (int(ticks[-1]) if ticks else 0)
    ticks, masks = _ensure_final_off_edge(ticks, masks, loop_end_tick)
    # Anchor an all-off edge at tick 0: the engine seeds its time counter from edge 0, so a
    # sequence whose first pulse starts after t=0 would otherwise slip every edge one tick
    # on hardware (same invariant the pulse-table compilers enforce).
    if not ticks or int(ticks[0]) != 0:
        ticks = [0] + list(ticks)
        masks = [0] + list(masks)
    payload = {
        "sequence": sequence.to_dict(),
        "clock_hz": clock_hz,
        "channels": channels,
        "ticks": ticks,
        "masks": masks,
        "repeat_count": sequence.repeat_count,
        "repeat_forever": sequence.repeat_forever,
    }
    sequence_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return RuntimeSequenceProgram(
        sequence_id=sequence_id,
        sequence_name=sequence.name,
        clock_hz=clock_hz,
        channels=list(channels),
        ticks=list(ticks),
        masks=list(masks),
        duration=sequence.duration,
        trigger_count=count_trigger_pulses(sequence, trigger_channels=trigger_channels),
        source_sequence=sequence.to_dict(),
        repeat_forever=bool(sequence.repeat_forever),
        loop_start_index=0,
        loop_end_tick=loop_end_tick,
        loop_count=int(sequence.repeat_count),
    )


def compile_pulse_table_runtime_program(
    state: PulseTableState,
    *,
    channels: Sequence[str] | None = None,
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
    slots: Mapping[str, float] | None = None,
    repeat_forever: bool = True,
) -> RuntimeSequenceProgram:
    """Compile GUI period-card state into an unexpanded FPGA loop program.

    ``PulseTableState`` carries the frontend repeat-bracket semantics.  The
    runtime FPGA should receive one copy of the period table plus loop metadata,
    not a fully expanded edge table.  A bracket becomes one finite inner loop;
    the whole table may still be repeated forever by the FPGA.  Any bound scan
    slots are resolved to constants using ``slots`` (default: the reference
    scan point), so this path emits a single static program.
    """

    channels = list(channel_names(state.channels if channels is None else channels, "channels"))
    unknown_channels = [channel for channel in state.channels if channel not in channels]
    if unknown_channels:
        raise ValueError(f"pulse table channels are not in hardware channels: {unknown_channels}.")
    clock_hz = positive_float(clock_hz, "clock_hz")
    clock_step_ns = 1e9 / clock_hz
    slot_values = state.reference_slots() if slots is None else dict(slots)

    # A (constant) channel delay inside a finite repeat bracket: there is no inner-loop
    # boundary for an additively-shifted edge to cross once the bracket is UNROLLED into a
    # flat period list, so a delay in ANY form works.  Compile the unrolled state with the
    # existing flat additive machinery (loop_count becomes 1; the flat frame can still
    # repeat_forever).  No delay, or no bracket -> compile the state as-is (compact loop).
    has_bracket = state.repeat_start is not None and state.repeat_end is not None
    has_delays = _pulse_table_has_delays(state, slots=slot_values, time_step_ns=clock_step_ns)
    if has_delays and has_bracket:
        unrolled = state.unrolled_bracket()
        _check_unrolled_edge_budget(unrolled, slots=slot_values, time_step_ns=clock_step_ns)
        return compile_pulse_table_runtime_program(
            unrolled,
            channels=channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            slots=slots,
            repeat_forever=repeat_forever,
        )

    state.validate(slots=slot_values, time_step_ns=clock_step_ns)
    sequence = state.to_sequence(slots=slot_values, time_step_ns=clock_step_ns, expand_repeat=False)
    period_starts = _pulse_table_period_starts_ticks(state, slots=slot_values, time_step_ns=clock_step_ns)
    bus_names, bus_segments = _pulse_table_bus_segments(
        state,
        slots=slot_values,
        time_step_ns=clock_step_ns,
    )
    ticks, masks, channels, loop_end, repeat_from_index, channel_delays, delay_channels = _pulse_table_edge_table(
        state,
        channels=channels,
        slots=slot_values,
        time_step_ns=clock_step_ns,
        fold_analog_buses=not bool(bus_segments),
        repeat_forever=bool(repeat_forever) and not has_bracket,
    )
    repeat_count = int(state.repeat_count)
    if not has_bracket:
        loop_start_index = 0
        # The loop period is the steady frame end; with a delay the engine rewinds to
        # repeat_from_index (the steady-frame start) so the real-startup preamble plays
        # exactly once.  With no delay repeat_from_index == 0 (the whole frame loops).
        loop_end_tick = int(loop_end)
        loop_count = 1
    else:
        loop_start_tick = int(period_starts[int(state.repeat_start)])
        loop_end_tick = int(period_starts[int(state.repeat_end) + 1])
        ticks, masks, loop_start_index = _insert_mask_edge_at_tick(ticks, masks, loop_start_tick)
        loop_count = repeat_count
        repeat_from_index = 0   # a finite bracket replays the whole program on repeat

    effective_duration_ticks = _pulse_table_effective_duration_ticks(state, slots=slot_values, time_step_ns=clock_step_ns)
    if has_delays and not has_bracket:
        effective_duration_ticks = int(loop_end)
    payload = {
        "table": state.to_dict(),
        "clock_hz": clock_hz,
        "channels": channels,
        "ticks": ticks,
        "masks": masks,
        "repeat_forever": bool(repeat_forever),
        "loop_start_index": loop_start_index,
        "loop_end_tick": loop_end_tick,
        "loop_count": loop_count,
        "repeat_from_index": repeat_from_index,
        "bus_names": bus_names,
        "bus_segments": [segment.to_dict() for segment in bus_segments],
        "channel_delays": [int(channel_delays.get(bit, 0)) for bit in range(len(channels))],
        "delay_channels": [dc.to_dict() for dc in delay_channels],
    }
    channel_delays_list = [int(channel_delays.get(bit, 0)) for bit in range(len(channels))] if channel_delays else None
    sequence_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return RuntimeSequenceProgram(
        sequence_id=sequence_id,
        sequence_name=state.name,
        clock_hz=clock_hz,
        channels=list(channels),
        ticks=list(ticks),
        masks=list(masks),
        duration=effective_duration_ticks / clock_hz,
        trigger_count=_pulse_table_trigger_count(state, trigger_channels=trigger_channels),
        source_sequence=sequence.to_dict(),
        source_table=state.to_dict(),
        repeat_forever=bool(repeat_forever),
        loop_start_index=loop_start_index,
        loop_end_tick=loop_end_tick,
        loop_count=loop_count,
        repeat_from_index=repeat_from_index,
        bus_names=bus_names or None,
        bus_segments=bus_segments or None,
        channel_delays=channel_delays_list,
        delay_channels=delay_channels or None,
    )


def compile_pulse_table_scan_runtime_program(
    state: PulseTableState,
    *,
    scan_table: Sequence[Sequence[float]] | None = None,
    channels: Sequence[str] | None = None,
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
    repeat_forever: bool = False,
    coeff_frac_bits: int = 8,
) -> RuntimeSequenceProgram:
    """Compile a ``PulseTableState`` with bound scan slots into a scan program.

    Each ``scan_table`` row is one scan point; column ``j`` is the value of
    slot ``j`` (named ``s{j}``, in the slot's display unit).  Bound durations and
    delays are *time slots*: they feed the affine tick formula
    ``tick = base + (sum_j coeff_j * slot_tick_j) >> coeff_frac_bits``.  The
    hardware iterates the scan points seamlessly; only the template and the
    parameter table are uploaded.
    """

    channels = list(channel_names(state.channels if channels is None else channels, "channels"))
    unknown_channels = [channel for channel in state.channels if channel not in channels]
    if unknown_channels:
        raise ValueError(f"pulse table channels are not in hardware channels: {unknown_channels}.")
    clock_hz = positive_float(clock_hz, "clock_hz")
    clock_step_ns = 1e9 / clock_hz
    if not state.scan_slots:
        raise ValueError("hardware scan requires at least one bound scan slot; bind a duration/delay/DAC first.")

    # A channel delay (constant OR scanned) inside a finite repeat bracket: UNROLL the
    # bracket into a flat period list first, then compile with the flat affine machinery
    # (additive global shift G + affine scan + reordering delay lanes).  Once flat there
    # is no inner-loop boundary for a delayed edge to cross, so a scanned delay works in
    # ANY form -- crossing the (former) boundary, reordering, negative, or frame-extending.
    # This also closes a latent hole: the scan path never validated a delay crossing the
    # bracket boundary, so scan+bracket+crossing-delay was silently wrong before.
    has_bracket = state.repeat_start is not None and state.repeat_end is not None
    if has_bracket and _pulse_table_has_any_delay(state):
        unrolled = state.unrolled_bracket()
        _check_unrolled_edge_budget(unrolled, slots=unrolled.reference_slots(), time_step_ns=clock_step_ns)
        return compile_pulse_table_scan_runtime_program(
            unrolled,
            scan_table=scan_table,
            channels=channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            repeat_forever=repeat_forever,
            coeff_frac_bits=coeff_frac_bits,
        )

    # DAC value + duration + delay scan simultaneously: every analog-bus segment's
    # ticks are emitted as affine expressions (base + per-slot coeffs), so a scanned
    # duration/delay moves the segment -- and any ramp's start/stop ticks -- in
    # lockstep with the digital edges.  Ramps with fixed value endpoints therefore
    # scan their TIMING freely; ramps whose value endpoints are themselves scanned
    # use the dual start/stop value_select (see _pulse_table_bus_segments).
    table = [[float(value) for value in row] for row in (state.scan_table if scan_table is None else scan_table)]
    if not table:
        raise ValueError("hardware scan requires at least one scan-table row.")
    for index, row in enumerate(table):
        if len(row) != len(state.scan_slots):
            raise ValueError(f"scan table row {index} has {len(row)} values but {len(state.scan_slots)} slots.")
    slot_vars = state.scan_var_names

    def point_slots_ns(row: Sequence[float]) -> dict[str, float]:
        # Time slots carry a physical time (-> ns); DAC slots carry a raw 10-bit
        # code that must pass through untouched so the bus engine reads it directly.
        return {
            slot_var(index): float(row[index]) * (1.0 if slot.kind == "dac" else UNITS_TO_NS.get(slot.unit, 1.0))
            for index, slot in enumerate(state.scan_slots)
        }

    def point_slot_value(point_index: int, slot_index: int, ns: Mapping[str, float]) -> int:
        slot = state.scan_slots[slot_index]
        if slot.kind == "dac":
            # Store the DAC code verbatim (no ns->tick conversion); its affine
            # coefficient is 0 so it never enters the edge-tick formula.
            return int(round(float(ns[slot_var(slot_index)])))
        return _time_ns_to_ticks(
            ns[slot_var(slot_index)], clock_step_ns, f"scan point {point_index} slot {slot_index}", allow_negative=True
        )

    # Validate the slot bindings + the full scan TABLE once (slot-independent), then each
    # scan point only re-checks its RESOLVED state (durations/DAC/delays at that point) with
    # validate_scan_slots=False.  Validating the whole table per point was O(N^2) and made
    # on_pulse very slow for thousands of points.
    state.validate(slots=state.reference_slots(), time_step_ns=clock_step_ns)
    points_ticks: list[list[int]] = []
    for point_index, row in enumerate(table):
        ns = point_slots_ns(row)
        points_ticks.append([
            point_slot_value(point_index, index, ns) for index in range(len(state.scan_slots))
        ])
        state.validate(slots=ns, time_step_ns=clock_step_ns, validate_scan_slots=False)

    # Analog buses are driven by the hardware bus engine, not the TTL edge table.
    # A scanned DAC value becomes a bus segment whose value_select reads the slot
    # per scan point; we exclude bus member channels from the affine edge rows so
    # they are not also driven as TTL bits.
    bus_names: list[str] = []
    bus_segments: list[RuntimeBusSegment] = []
    bus_members: list[str] = []
    if _pulse_table_has_analog_activity(state):
        bus_names, bus_segments = _pulse_table_bus_segments(
            state,
            slots=state.reference_slots(),
            time_step_ns=clock_step_ns,
            slot_vars=slot_vars,
            coeff_frac_bits=coeff_frac_bits,
        )
        bus_members = [channel for members in state.bus_channels().values() for channel in members]

    # PHYSICAL CHANNEL DELAY: a delay is NOT scanned and NOT baked into the edges -- it is a
    # CONSTANT per-channel OUTPUT delay (a delay line; see engine_model.delay_line_reference).
    # Compute it over the TTL (non-bus) channels, folding the negative-delay global shift
    # G = max(0, -min delay) into every channel's delay (a causal delay line cannot lead, so
    # shifting all of them by G keeps relative timing while every delay stays >= 0).  The edge
    # table is emitted UNDELAYED and the loop period is the plain (affine-in-duration) frame,
    # so a delay of ANY length never disturbs another channel and never changes the period.
    hardware_bits = {ch: index for index, ch in enumerate(channel_names(channels, "channels"))}
    raw_delay = {
        ch: state.delay_steps(ch, slots=state.reference_slots(), time_step_ns=clock_step_ns)
        for ch in state.channels if ch not in bus_members and ch in hardware_bits
    }
    global_shift = max(0, -min(raw_delay.values())) if raw_delay else 0
    channel_delays = {
        hardware_bits[ch]: raw_delay[ch] + global_shift
        for ch in raw_delay if (raw_delay[ch] + global_shift) != 0
    }

    rows = _pulse_table_affine_rows(
        state,
        channels=channels,
        scan_points=points_ticks,
        slot_vars=slot_vars,
        time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits,
        exclude_channels=bus_members,
    )
    ticks = [row[0] for row in rows]
    masks = [row[1] for row in rows]
    tick_slot_coeffs = [list(row[2]) for row in rows]
    loop_start_index, loop_end_tick, loop_end_slot_coeffs, loop_count = _pulse_table_affine_loop_metadata(
        state,
        rows=rows,
        slot_vars=slot_vars,
        time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits,
    )
    # Per-delayed-channel ON intervals (affine in the scanned durations) for the UNBOUNDED
    # membership delay player -- the engine evaluates membership at the shifted phase, never
    # buffers, so the delay/period are unbounded across the scan too.
    delay_channels = _pulse_table_affine_delay_channels(
        state,
        channel_delays=channel_delays,
        hardware_bits=hardware_bits,
        exclude_channels=bus_members,
        slot_vars=slot_vars,
        time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits,
    )
    point_durations = [
        float(_apply_affine_ticks(ticks[-1], tick_slot_coeffs[-1], point, coeff_frac_bits)) / clock_hz
        for point in points_ticks
    ]
    sequence = state.to_sequence(slots=point_slots_ns(table[0]), time_step_ns=clock_step_ns, expand_repeat=False)
    trigger_count = _pulse_table_trigger_count(state, trigger_channels=trigger_channels) * len(points_ticks)
    slot_kinds = [slot.kind for slot in state.scan_slots]
    source_table = state.to_dict()
    source_table["scan_table"] = [list(row) for row in table]
    payload = {
        "table": state.to_dict(),
        "clock_hz": clock_hz,
        "channels": channels,
        "ticks": ticks,
        "masks": masks,
        "tick_slot_coeffs": tick_slot_coeffs,
        "scan_points": points_ticks,
        "slot_kinds": slot_kinds,
        "repeat_forever": bool(repeat_forever),
        "loop_start_index": loop_start_index,
        "loop_end_tick": loop_end_tick,
        "loop_end_slot_coeffs": loop_end_slot_coeffs,
        "loop_count": loop_count,
        "scan_coeff_frac_bits": coeff_frac_bits,
        "bus_names": bus_names,
        "bus_segments": [segment.to_dict() for segment in bus_segments],
        "channel_delays": [int(channel_delays.get(bit, 0)) for bit in range(len(channels))],
        "delay_channels": [dc.to_dict() for dc in delay_channels],
    }
    channel_delays_list = [int(channel_delays.get(bit, 0)) for bit in range(len(channels))] if channel_delays else None
    sequence_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return RuntimeSequenceProgram(
        sequence_id=sequence_id,
        sequence_name=state.name,
        clock_hz=clock_hz,
        channels=list(channels),
        ticks=ticks,
        masks=masks,
        duration=sum(point_durations),
        trigger_count=trigger_count,
        source_sequence=sequence.to_dict(),
        source_table=source_table,
        repeat_forever=bool(repeat_forever),
        loop_start_index=loop_start_index,
        loop_end_tick=loop_end_tick,
        loop_count=loop_count,
        slot_count=len(state.scan_slots),
        slot_kinds=slot_kinds,
        loop_end_slot_coeffs=loop_end_slot_coeffs,
        tick_slot_coeffs=tick_slot_coeffs,
        scan_points=points_ticks,
        scan_point_durations=point_durations,
        scan_coeff_frac_bits=coeff_frac_bits,
        bus_names=bus_names or None,
        bus_segments=bus_segments or None,
        channel_delays=channel_delays_list,
        delay_channels=delay_channels or None,
    )


def compile_runtime_program_for_payload(
    payload: PulseSequence | PulseTableState,
    *,
    channels: Sequence[str],
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> RuntimeSequenceProgram:
    """Compile either finite sequence data or GUI pulse-table data."""

    if isinstance(payload, PulseTableState):
        if payload.scan_slots and payload.scan_table:
            return compile_pulse_table_scan_runtime_program(
                payload,
                channels=channels,
                clock_hz=clock_hz,
                trigger_channels=trigger_channels,
                scan_table=payload.scan_table,
                repeat_forever=payload.repeat_forever,
            )
        return compile_pulse_table_runtime_program(
            payload,
            channels=channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            repeat_forever=payload.repeat_forever,
        )
    return compile_runtime_program(payload, channels=channels, clock_hz=clock_hz, trigger_channels=trigger_channels)


def finite_frame_sequence(
    payload: PulseSequence | PulseTableState,
    frames: int,
    *,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> PulseSequence:
    """Return a finite ``PulseSequence`` with exactly ``frames`` trigger rises."""

    frames = positive_int(frames, "frames")
    trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
    if isinstance(payload, PulseTableState):
        slots = payload.reference_slots()
        sequence = payload.to_sequence(slots=slots, time_step_ns=payload.time_step_ns, expand_repeat=False)
        base_period_s = sum(
            period.duration_steps(slots=slots, time_step_ns=payload.time_step_ns) for period in payload.periods
        ) * payload.time_step_ns * 1e-9
        triggers = count_trigger_pulses(sequence, trigger_channels=trigger_channels)
        if triggers == frames:
            return sequence
        if triggers == 1 and frames > 1:
            return sequence.repeated(frames, period=base_period_s)
        raise ValueError(
            f"sequence {sequence.name!r} has {triggers} camera trigger pulses, "
            f"but acquisition requested {frames} frame(s)."
        )
    if isinstance(payload, PulseSequence):
        return sequence_for_frame_count(payload, frames, trigger_channels=trigger_channels)
    raise TypeError("frame acquisition sequence must be a PulseSequence or PulseTableState.")


class SequencerService:
    """Stateful service that mirrors the final FPGA runtime protocol.

    The same object can run in-process for tests, or be exposed over RPyC on
    the FPGA/Vivado computer.  Hardware-specific callbacks can be attached
    later without changing the client-side ``SequencerDevice`` contract.
    """

    def __init__(
        self,
        *,
        channels: Sequence[str],
        clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        prepare_callback: Callable[[RuntimeSequenceProgram], None] | None = None,
        fire_callback: Callable[[RuntimeSequenceProgram], None] | None = None,
        wait_done_callback: Callable[[RuntimeSequenceProgram, float | None], bool] | None = None,
        safe_state_callback: Callable[[], None] | None = None,
        sleep_scale: float = 0.0,
        cache_prepared: bool = True,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
        self.prepare_callback = prepare_callback
        self.fire_callback = fire_callback
        self.wait_done_callback = wait_done_callback
        self.safe_state_callback = safe_state_callback
        self.sleep_scale = nonnegative_float(sleep_scale, "sleep_scale")
        self.cache_prepared = bool(cache_prepared)
        self._lock = threading.RLock()
        self.prepared_program: RuntimeSequenceProgram | None = None
        self.state = "idle"
        self.history: list[dict[str, object]] = []

    def prepare(self, sequence_payload) -> dict[str, object]:
        timing_payload = timing_from_payload(sequence_payload)
        program = compile_runtime_program_for_payload(
            timing_payload,
            channels=self.channels,
            clock_hz=self.clock_hz,
            trigger_channels=self.trigger_channels,
        )
        with self._lock:
            cached = (
                self.cache_prepared
                and self.prepared_program is not None
                and self.prepared_program.sequence_id == program.sequence_id
            )
            if self.prepare_callback is not None and not cached:
                self.prepare_callback(program)
            self.prepared_program = program
            self.state = "prepared"
            self.history.append(
                {
                    "action": "prepare",
                    "sequence_id": program.sequence_id,
                    "triggers": program.trigger_count,
                    "cached": cached,
                }
            )
        return program.to_dict()

    def fire(self, sequence_payload=None) -> dict[str, object]:
        with self._lock:
            program = self._require_prepared()
            if sequence_payload is not None:
                requested = compile_runtime_program_for_payload(
                    timing_from_payload(sequence_payload),
                    channels=self.channels,
                    clock_hz=self.clock_hz,
                    trigger_channels=self.trigger_channels,
                )
                if requested.sequence_id != program.sequence_id:
                    raise RuntimeError("fire(sequence) does not match the prepared runtime program.")
            if self.fire_callback is not None:
                self.fire_callback(program)
            self.state = "running"
            self.history.append({"action": "fire", "sequence_id": program.sequence_id})
            return program.to_dict()

    def wait_done(self, timeout: float | None = None) -> bool:
        with self._lock:
            program = self._require_prepared()
        if program.repeat_forever and timeout is None:
            raise RuntimeError("sequencer wait_done cannot wait forever for a repeat_forever program; pass a timeout or stop the pulse.")
        if self.wait_done_callback is not None:
            ok = bool(self.wait_done_callback(program, timeout))
        elif program.repeat_forever:
            ok = False
        else:
            delay = program.duration * self.sleep_scale
            if timeout is not None and delay > float(timeout):
                ok = False
            else:
                if delay > 0:
                    time.sleep(delay)
                ok = True
        with self._lock:
            self.state = "done" if ok else "timeout"
            self.history.append({"action": "wait_done", "sequence_id": program.sequence_id, "ok": ok})
        return ok

    def abort(self) -> None:
        with self._lock:
            self.prepared_program = None
            self.state = "aborted"
            self.history.append({"action": "abort", "invalidated": True})
        if self.safe_state_callback is not None:
            self.safe_state_callback()

    def set_safe_state(self) -> None:
        with self._lock:
            self.prepared_program = None
            self.state = "safe"
            self.history.append({"action": "safe", "invalidated": True})
        if self.safe_state_callback is not None:
            self.safe_state_callback()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "type": type(self).__name__,
                "channels": list(self.channels),
                "clock_hz": self.clock_hz,
                "trigger_channels": list(self.trigger_channels),
                "state": self.state,
                "cache_prepared": self.cache_prepared,
                "prepared_program": None if self.prepared_program is None else self.prepared_program.to_dict(),
                "history_length": len(self.history),
            }

    def _require_prepared(self) -> RuntimeSequenceProgram:
        if self.prepared_program is None:
            raise RuntimeError("sequencer service has no prepared sequence.")
        return self.prepared_program


class RuntimeSequencer(SequencerDevice):
    """Local device adapter for the runtime edge-table protocol."""

    def __init__(
        self,
        *,
        channels: Sequence[str],
        clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        sleep_scale: float = 0.0,
    ):
        self.service = SequencerService(
            channels=channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            sleep_scale=sleep_scale,
        )
        self.channels = self.service.channels
        self.clock_hz = self.service.clock_hz
        self.trigger_channels = self.service.trigger_channels
        self.last_program: RuntimeSequenceProgram | None = None

    def prepare(self, sequence: PulseSequence | PulseTableState) -> RuntimeSequenceProgram:
        self.last_program = RuntimeSequenceProgram.from_dict(self.service.prepare(timing_payload_to_dict(sequence)))
        return self.last_program

    def fire(self, sequence: PulseSequence | PulseTableState | None = None) -> None:
        self.service.fire(None if sequence is None else timing_payload_to_dict(sequence))

    def wait_done(self, timeout: float | None = None) -> bool:
        return self.service.wait_done(timeout)

    def abort(self) -> None:
        self.service.abort()

    def set_safe_state(self) -> None:
        self.service.set_safe_state()

    def snapshot(self) -> dict[str, object]:
        out = self.service.snapshot()
        out["type"] = type(self).__name__
        return out


class ManualSequencer(SequencerDevice):
    """Sequencer adapter for first-light tests with a manually started FPGA.

    ``fire`` intentionally does not drive hardware.  It records that the camera
    is armed and that the operator or an external free-running FPGA must provide
    the trigger pulses before the qCMOS timeout expires.
    """

    def __init__(
        self,
        *,
        channels: Sequence[str],
        clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        message: str | None = None,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
        self.message = message or "Camera is armed. Start the FPGA/manual trigger sequence now."
        self.prepared_sequence: PulseSequence | None = None
        self.state = "idle"
        self.history: list[dict[str, object]] = []

    def prepare(self, sequence: PulseSequence) -> RuntimeSequenceProgram:
        sequence.validate(clock_hz=self.clock_hz, channels=self.channels).raise_if_failed()
        program = compile_runtime_program(
            sequence,
            channels=self.channels,
            clock_hz=self.clock_hz,
            trigger_channels=self.trigger_channels,
        )
        self.prepared_sequence = sequence
        self.state = "prepared"
        self.history.append({"action": "prepare", "sequence_id": program.sequence_id, "triggers": program.trigger_count})
        return program

    def fire(self, sequence: PulseSequence | None = None) -> None:
        if self.prepared_sequence is None:
            raise RuntimeError("ManualSequencer.fire() called before prepare().")
        if sequence is not None and sequence is not self.prepared_sequence:
            raise RuntimeError("ManualSequencer.fire() received a sequence that was not prepared.")
        self.state = "manual_trigger_wait"
        self.history.append({"action": "fire_manual", "message": self.message})
        print(self.message)

    def wait_done(self, timeout: float | None = None) -> bool:
        self.state = "unknown_done"
        self.history.append({"action": "wait_done_manual", "timeout": timeout})
        return True

    def abort(self) -> None:
        self.state = "aborted"
        self.history.append({"action": "abort"})

    def set_safe_state(self) -> None:
        self.state = "safe_requested"
        self.history.append({"action": "safe"})

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "channels": list(self.channels),
            "clock_hz": self.clock_hz,
            "trigger_channels": list(self.trigger_channels),
            "state": self.state,
            "prepared": self.prepared_sequence is not None,
            "history_length": len(self.history),
        }


class RemoteSequencer(SequencerDevice):
    """RPyC client-side sequencer proxy for the FPGA/Vivado computer."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        channels: Sequence[str],
        clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        ssl: bool = False,
        ca_certs: str | None = None,
        connect_on_init: bool = False,
    ):
        self.host = str(host).strip()
        if self.host in {"", "0.0.0.0", "::"}:
            raise ValueError("RemoteSequencer host must be the server address reachable from the control computer.")
        self.port = int(port)
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.trigger_channels = tuple(channel_names(trigger_channels, "trigger_channels"))
        self.ssl = bool(ssl)
        self.ca_certs = ca_certs
        self._conn = None
        self._last_program: RuntimeSequenceProgram | None = None
        if connect_on_init:
            self.open()

    def open(self) -> "RemoteSequencer":
        if self._conn is not None:
            return self
        try:
            import rpyc
        except ImportError as exc:  # pragma: no cover - depends on lab install
            raise RuntimeError("RemoteSequencer requires `rpyc`. Install it on the control computer.") from exc
        if self.ssl:
            self._conn = rpyc.utils.classic.ssl_connect(host=self.host, port=self.port, ca_certs=self.ca_certs)
        else:
            # Finite backstop timeout (generous: must exceed the longest server-side
            # action, e.g. wait_done on a big finite scan) so a genuinely wedged server
            # cannot block the caller -- the GUI worker thread -- forever.  prepare/fire
            # return in seconds; this only bounds a truly stuck request.
            self._conn = rpyc.connect(self.host, self.port, config={"allow_pickle": True, "sync_request_timeout": 3600.0})
        snap = self._conn.root.snapshot()
        self.channels = list(snap.get("channels", self.channels))
        self.clock_hz = float(snap.get("clock_hz", self.clock_hz))
        self.trigger_channels = tuple(channel_names(snap.get("trigger_channels", self.trigger_channels), "trigger_channels"))
        return self

    def prepare(self, sequence: PulseSequence | PulseTableState) -> RuntimeSequenceProgram:
        self.open()
        program = self._conn.root.prepare(json.dumps(timing_payload_to_dict(sequence)))
        payload = json.loads(program) if isinstance(program, (str, bytes)) else dict(program)
        self._last_program = RuntimeSequenceProgram.from_dict(payload)
        return self._last_program

    def fire(self, sequence: PulseSequence | PulseTableState | None = None) -> None:
        self.open()
        self._conn.root.fire(None if sequence is None else json.dumps(timing_payload_to_dict(sequence)))

    def wait_done(self, timeout: float | None = None) -> bool:
        self.open()
        return bool(self._conn.root.wait_done(timeout))

    def abort(self) -> None:
        if self._conn is not None:
            self._conn.root.abort()

    def set_safe_state(self) -> None:
        self.open()
        self._conn.root.set_safe_state()

    def snapshot(self) -> dict[str, object]:
        out = {
            "type": type(self).__name__,
            "host": self.host,
            "port": self.port,
            "channels": list(self.channels),
            "clock_hz": self.clock_hz,
            "trigger_channels": list(self.trigger_channels),
            "connected": self._conn is not None,
            "last_program": None if self._last_program is None else self._last_program.to_dict(),
        }
        if self._conn is not None:
            try:
                out["remote"] = dict(self._conn.root.snapshot())
            except Exception as exc:
                out["remote_error"] = str(exc)
        return out

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


class PulseController:
    """Notebook helper that binds a pulse payload to a sequencer.

    It keeps readout scans terse.  A single-point scan sets one slot per shot::

        pulse.set_time(200)          # ns into the first duration/delay slot
        pulse.on_pulse()

    A multi-point hardware scan uploads a whole scan table once::

        pulse.set_scan_table([[10], [20], [30]]).on_pulse()

    The controller owns no hardware; it delegates to the supplied local or
    remote ``SequencerDevice``.
    """

    def __init__(self, sequencer: SequencerDevice, pulse: PulseSequence | PulseTableState):
        self.sequencer = sequencer
        self.pulse = pulse
        self.scan_table = [list(row) for row in (getattr(pulse, "scan_table", []) or [])]
        self.slots: dict[str, float] = {}
        self.last_program: RuntimeSequenceProgram | None = None

    def set_slot(self, key: int | str, value: float) -> "PulseController":
        name = key if isinstance(key, str) else slot_var(int(key))
        self.slots[name] = float(value)
        return self

    def set_time(self, value_ns: float) -> "PulseController":
        """Set the first duration/delay scan slot (in ns) for the next shot."""

        name = self.pulse.primary_time_slot() if isinstance(self.pulse, PulseTableState) else None
        if name is None:
            raise TypeError("pulse has no duration/delay scan slot; bind one via the GUI scan dot or state.bind_field(...).")
        return self.set_slot(name, value_ns)

    def set_scan_table(self, rows: Sequence[Sequence[float]] | None) -> "PulseController":
        self.scan_table = [list(map(float, row)) for row in (rows or [])]
        return self

    def payload(
        self,
        *,
        slots: Mapping[str, float] | None = None,
        scan_table: Sequence[Sequence[float]] | None = None,
        repeat_forever: bool | None = None,
    ) -> PulseSequence | PulseTableState:
        if isinstance(self.pulse, PulseTableState):
            table = self.scan_table if scan_table is None else scan_table
            merged = dict(self.slots)
            merged.update(slots or {})
            if table:
                data = self.pulse.to_dict()
                data["scan_table"] = [list(row) for row in table]
                payload = PulseTableState.from_dict(data)
            elif merged:
                payload = self.pulse.with_slots_resolved(merged)
            else:
                payload = self.pulse
            if repeat_forever is not None:
                data = payload.to_dict()
                data["repeat_forever"] = bool(repeat_forever)
                payload = PulseTableState.from_dict(data)
            return payload
        if repeat_forever is not None:
            data = self.pulse.to_dict()
            data["repeat_forever"] = bool(repeat_forever)
            return PulseSequence.from_dict(data)
        return self.pulse

    def frame_sequence(
        self,
        frames: int,
        *,
        time_ns: float | None = None,
        slots: Mapping[str, float] | None = None,
        trigger_channels: Sequence[str] | None = None,
    ) -> PulseSequence:
        """Return a finite ``PulseSequence`` with exactly ``frames`` triggers."""

        frames = positive_int(frames, "frames")
        trigger_channels = tuple(channel_names(
            getattr(self.sequencer, "trigger_channels", DEFAULT_CAMERA_TRIGGER_CHANNELS) if trigger_channels is None else trigger_channels,
            "trigger_channels",
        ))
        merged = dict(slots or {})
        if time_ns is not None:
            name = self.pulse.primary_time_slot() if isinstance(self.pulse, PulseTableState) else None
            if name is None:
                raise TypeError("pulse has no duration/delay scan slot to set from time_ns.")
            merged[name] = float(time_ns)
        payload = self.payload(slots=merged, scan_table=[], repeat_forever=False)
        return finite_frame_sequence(payload, frames, trigger_channels=trigger_channels)

    def prepare(
        self,
        *,
        scan_table: Sequence[Sequence[float]] | None = None,
        repeat_forever: bool | None = None,
    ) -> RuntimeSequenceProgram:
        self.last_program = self.sequencer.prepare(self.payload(scan_table=scan_table, repeat_forever=repeat_forever))
        return self.last_program

    def on_pulse(
        self,
        *,
        wait: bool = False,
        timeout: float | None = None,
        scan_table: Sequence[Sequence[float]] | None = None,
        repeat_forever: bool | None = None,
    ) -> RuntimeSequenceProgram:
        payload = self.payload(scan_table=scan_table, repeat_forever=repeat_forever)
        if wait and timeout is None and bool(getattr(payload, "repeat_forever", False)):
            raise RuntimeError(
                "pulse.on_pulse(wait=True) cannot wait for a repeat_forever pulse without a timeout. "
                "Use pulse.on_pulse(wait=False, repeat_forever=True) for continuous scope output, "
                "or pulse.on_pulse(wait=True, repeat_forever=False) for a finite shot."
            )
        self.last_program = self.sequencer.prepare(payload)
        program = self.last_program
        self.sequencer.fire()
        if wait:
            if not self.sequencer.wait_done(timeout=timeout):
                raise TimeoutError(f"sequencer did not report done for pulse {program.sequence_name!r}.")
        return program

    def wait_done(self, timeout: float | None = None) -> bool:
        return bool(self.sequencer.wait_done(timeout=timeout))

    def stop(self) -> None:
        if hasattr(self.sequencer, "set_safe_state"):
            self.sequencer.set_safe_state()
        elif hasattr(self.sequencer, "abort"):
            self.sequencer.abort()

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe summary for notebook/debug display."""

        last = None
        if self.last_program is not None:
            last = {
                "sequence_name": self.last_program.sequence_name,
                "channels": list(self.last_program.channels),
                "edge_count": len(self.last_program.ticks),
                "trigger_count": int(self.last_program.trigger_count),
                "duration": float(self.last_program.duration),
                "repeat_forever": bool(self.last_program.repeat_forever),
                "loop_count": int(self.last_program.loop_count),
            }
        return {
            "type": type(self).__name__,
            "pulse_type": type(self.pulse).__name__,
            "slots": dict(self.slots),
            "scan_table": [list(row) for row in self.scan_table],
            "sequencer_type": type(self.sequencer).__name__,
            "sequencer_channels": list(getattr(self.sequencer, "channels", [])),
            "clock_hz": float(getattr(self.sequencer, "clock_hz", 0.0)),
            "trigger_channels": list(getattr(self.sequencer, "trigger_channels", [])),
            "last_program": last,
        }


def bind_pulse(sequencer: SequencerDevice, pulse: PulseSequence | PulseTableState) -> PulseController:
    """Return a ``PulseController`` for concise notebook pulse scans."""

    return PulseController(sequencer, pulse)


class VerilogSequencer(SequencerDevice):
    """Prepare writes generated Verilog; fire calls an optional hardware hook."""

    def __init__(
        self,
        *,
        channels: Sequence[str],
        clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
        output_dir: str | Path = "generated_sequences",
        module_name: str = "zlc_sequence",
        pin_map: Mapping[str, str] | None = None,
        fire_callback: Callable[[VerilogBuild], None] | None = None,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.output_dir = Path(output_dir)
        self.module_name = str(module_name)
        self.pin_map = None if pin_map is None else dict(pin_map)
        self.fire_callback = fire_callback
        self.last_build: VerilogBuild | None = None
        self.last_files: VerilogFiles | None = None
        self.prepared_sequence: PulseSequence | None = None

    def prepare(self, sequence: PulseSequence) -> VerilogBuild:
        build = generate_verilog(sequence, channels=self.channels, clock_hz=self.clock_hz, module_name=self.module_name)
        self.last_build = build
        self.last_files = write_verilog_bundle(build, self.output_dir, pin_map=self.pin_map)
        self.prepared_sequence = sequence
        return build

    def fire(self, sequence: PulseSequence | None = None) -> None:
        if self.last_build is None or self.prepared_sequence is None:
            raise RuntimeError("VerilogSequencer.fire() called before prepare().")
        if sequence is not None and sequence is not self.prepared_sequence:
            raise RuntimeError("VerilogSequencer.fire() received a sequence that was not prepared.")
        if self.fire_callback is None:
            raise RuntimeError(
                "Verilog files were generated, but no fire_callback is configured. "
                "For real hardware, pass a callback that starts the FPGA after the camera is armed."
            )
        self.fire_callback(self.last_build)

    def snapshot(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "channels": self.channels,
            "clock_hz": self.clock_hz,
            "output_dir": str(self.output_dir),
            "module_name": self.module_name,
            "last_verilog": None if self.last_files is None else str(self.last_files.verilog_path),
            "last_manifest": None if self.last_files is None else str(self.last_files.manifest_path),
            "prepared": self.prepared_sequence is not None,
        }

    def close(self) -> None:
        pass


def timing_payload_to_dict(payload: PulseSequence | PulseTableState) -> dict[str, object]:
    """Return the JSON-safe timing payload for a sequence or pulse table.

    A ``PulseTableState`` is SNAPPED to the clock-tick grid before serialization, so
    the pulse transferred to the server/hardware carries the same whole-tick values
    the GUI displays and the compiler would land on -- there is no place where an
    off-grid value silently slips through the pulse-transfer API."""

    if isinstance(payload, PulseTableState):
        return payload.snapped().to_dict()
    if isinstance(payload, PulseSequence):
        return payload.to_dict()
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError("timing payload must be a PulseSequence, PulseTableState, or mapping.")


def timing_from_payload(payload) -> PulseSequence | PulseTableState:
    """Accept local timing objects or their JSON/RPyC-safe dict payload."""

    if isinstance(payload, PulseSequence):
        return payload
    if isinstance(payload, PulseTableState):
        return payload
    if isinstance(payload, (str, bytes)):
        return timing_from_payload(json.loads(payload))
    if isinstance(payload, Mapping):
        data = dict(payload)
        schema = data.get("schema", "Zou_lab_control.neutral_atom.PulseSequence")
        if schema == "Zou_lab_control.neutral_atom.PulseTableState":
            return PulseTableState.from_dict(data)
        if schema == "Zou_lab_control.neutral_atom.PulseSequence":
            return PulseSequence.from_dict(data)
        raise ValueError(f"unsupported timing payload schema {schema!r}.")
    if hasattr(payload, "items"):
        return timing_from_payload(_plain_rpc_payload(payload))
    raise TypeError("timing payload must be a PulseSequence/PulseTableState or a to_dict() mapping.")


def sequence_from_payload(payload) -> PulseSequence:
    """Accept a local ``PulseSequence`` or its JSON/RPyC-safe dict payload."""

    timing = timing_from_payload(payload)
    if not isinstance(timing, PulseSequence):
        raise TypeError("sequence payload must be a PulseSequence or PulseSequence.to_dict() mapping.")
    return timing


def _time_to_ticks(value_s: float, clock_hz: float, name: str) -> int:
    raw = float(value_s) * float(clock_hz)
    ticks = int(round(raw))
    if not math.isclose(raw, ticks, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{name}={value_s:g} s is not on the {clock_hz:g} Hz clock grid.")
    if ticks <= 0:
        raise ValueError(f"{name} must be at least one clock tick.")
    return ticks


def _ensure_final_off_edge(ticks: Sequence[int], masks: Sequence[int], final_tick: int) -> tuple[list[int], list[int]]:
    ticks = [int(tick) for tick in ticks]
    masks = [int(mask) for mask in masks]
    final_tick = int(final_tick)
    if not ticks:
        return [final_tick], [0]
    if final_tick < ticks[-1]:
        raise ValueError("repeat period is shorter than the base sequence edge table.")
    if final_tick == ticks[-1]:
        masks[-1] = 0
        return ticks, masks
    ticks.append(final_tick)
    masks.append(0)
    return ticks, masks


def _insert_mask_edge_at_tick(ticks: Sequence[int], masks: Sequence[int], tick: int) -> tuple[list[int], list[int], int]:
    """Insert a snapshot edge at ``tick`` and return its index.

    Hardware loops restart by loading ``mask_mem[loop_start_index]``.  Delayed
    pulse sequences may not naturally have an edge at the GUI repeat-bracket
    boundary, so the compiler inserts a complete state snapshot there.
    """

    out_ticks = [int(item) for item in ticks]
    out_masks = [int(item) for item in masks]
    tick = int(tick)
    current_mask = 0
    for index, candidate in enumerate(out_ticks):
        candidate = int(candidate)
        if candidate == tick:
            return out_ticks, out_masks, index
        if candidate > tick:
            out_ticks.insert(index, tick)
            out_masks.insert(index, current_mask)
            return out_ticks, out_masks, index
        current_mask = out_masks[index]
    out_ticks.append(tick)
    out_masks.append(current_mask)
    return out_ticks, out_masks, len(out_ticks) - 1


def _pulse_table_period_starts_ticks(
    state: PulseTableState,
    *,
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
) -> list[int]:
    starts = [0]
    for period in state.periods:
        starts.append(starts[-1] + period.duration_steps(slots=slots, time_step_ns=time_step_ns))
    return starts


def _affine_expr(
    value: float | str,
    unit: str,
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
) -> tuple[int, tuple[int, ...]]:
    base, coeffs = affine_coeffs(value, slot_vars=slot_vars, unit=unit, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    return base, tuple(coeffs)


def _pulse_table_affine_period_starts(
    state: PulseTableState,
    *,
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
) -> list[tuple[int, tuple[int, ...]]]:
    starts = [(0, tuple(0 for _ in slot_vars))]
    for period in state.periods:
        starts.append(_affine_add(starts[-1], _affine_expr(period.duration, period.unit, slot_vars, time_step_ns, coeff_frac_bits)))
    return starts


def _pulse_table_affine_all_edge_exprs(
    state: PulseTableState,
    *,
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
    exclude_channels: Sequence[str] = (),
) -> list[tuple[int, tuple[int, ...]]]:
    """Every channel's delayed rise/fall edge expr (period_start +/- delay), affine in the
    scan slots, over ALL non-excluded channels INCLUDING ones that will become delay
    lanes.  The shared root for the global shift G and the global frame end so the main
    table and the lanes agree on both."""

    exclude = set(exclude_channels)
    starts = _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    exprs: list[tuple[int, tuple[int, ...]]] = [(0, tuple(0 for _ in slot_vars)), starts[-1]]
    for channel_index, channel in enumerate(state.channels):
        if channel in exclude:
            continue
        delay = _affine_expr(state.delays.get(channel, 0.0), state.delay_units.get(channel, "ns"), slot_vars, time_step_ns, coeff_frac_bits)
        active_start: tuple[int, tuple[int, ...]] | None = None
        for period_index, period in enumerate(state.periods):
            value = int(period.states[channel_index])
            if value and active_start is None:
                active_start = starts[period_index]
            elif not value and active_start is not None:
                exprs.append(_affine_add(active_start, delay))
                exprs.append(_affine_add(starts[period_index], delay))
                active_start = None
        if active_start is not None:
            exprs.append(_affine_add(active_start, delay))
            exprs.append(_affine_add(starts[-1], delay))
    return exprs


def _pulse_table_affine_rows(
    state: PulseTableState,
    *,
    channels: Sequence[str],
    scan_points: Sequence[Sequence[int]],
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
    exclude_channels: Sequence[str] = (),
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Return one affine edge row ``(base_tick, mask, slot_coeffs)`` per edge -- the
    UNDELAYED template.

    Every channel's rise/fall edge is a period boundary ``period_start`` evaluated affinely
    in the bound scan slots (the scanned DURATIONS).  Channel DELAYS are NOT applied here:
    a delay is a per-channel OUTPUT delay (``channel_delays``, a delay line), never baked
    into the edges.  Because every edge sits on a monotone period boundary, the merged edge
    list is globally tick-monotone at every scan point automatically -- no channel reorders,
    so no global shift G and no delay lane are needed.  ``_stable_affine_groups`` still
    validates per-channel + cross-channel ordering at every scan point as a safety net."""

    hardware_channels = list(channel_names(channels, "channels"))
    exclude = set(exclude_channels)
    starts = _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    events: list[tuple[tuple[int, tuple[int, ...]], str | None, int | None]] = []

    for channel_index, channel in enumerate(state.channels):
        if channel in exclude:
            continue  # analog-bus members are driven by the bus engine, not TTL edges
        active_start: tuple[int, tuple[int, ...]] | None = None
        for period_index, period in enumerate(state.periods):
            value = int(period.states[channel_index])
            if value and active_start is None:
                active_start = starts[period_index]
            elif not value and active_start is not None:
                events.append((active_start, channel, 1))
                events.append((starts[period_index], channel, 0))
                active_start = None
        if active_start is not None:
            events.append((active_start, channel, 1))
            events.append((starts[-1], channel, 0))

    if state.repeat_start is not None and state.repeat_end is not None and state.repeat_count > 1:
        events.append((starts[int(state.repeat_start)], None, None))

    # Final all-off marker at the nominal frame end (a channel ON through the last period has
    # its fall at the SAME expr, so they group into one all-off row -- no bump needed).
    events.append((starts[-1], None, None))

    grouped = _stable_affine_groups(events, scan_points=scan_points, coeff_frac_bits=coeff_frac_bits)
    current_mask = 0
    rows: list[tuple[int, int, tuple[int, ...]]] = []
    for expr, group_events in grouped:
        for channel, value in group_events:
            if channel is None or value is None:
                continue
            bit = hardware_channels.index(channel) if channel in hardware_channels else None
            if bit is None:
                continue
            if value:
                current_mask |= 1 << bit
            else:
                current_mask &= ~(1 << bit)
        rows.append((expr[0], current_mask, expr[1]))
    if not rows:
        rows.append((0, 0, tuple(0 for _ in slot_vars)))
    # Anchor an edge at ABSOLUTE tick 0 (all-off if nothing starts there) for EVERY scan
    # point: the engine seeds its time counter from edge 0, so the table must begin at tick
    # 0 with zero slot coeffs or every edge slips by the prefetch latency on hardware.  A
    # delayed channel, a global-shift G, or an all-off opening period all push the first
    # real edge past 0 -- exactly when this anchor is required.  (Mirror of the non-scan
    # _pulse_table_edge_table tick-0 anchor.)
    if rows[0][0] != 0 or any(rows[0][2]):
        rows.insert(0, (0, 0, tuple(0 for _ in slot_vars)))
    if int(rows[-1][1]) != 0:
        raise ValueError("hardware scan template final row must return every channel to 0.")
    return rows


def _pulse_table_affine_delay_channels(
    state: PulseTableState,
    *,
    channel_delays: Mapping[int, int],
    hardware_bits: Mapping[str, int],
    exclude_channels: Sequence[str],
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
) -> list[RuntimeDelayChannel]:
    """Build the per-delayed-channel ON intervals for the UNBOUNDED membership player on
    the SCAN path.  Each interval's start/stop tick is a period boundary evaluated affinely
    in the scanned DURATION slots (base + per-slot coeffs), so a scanned duration moves the
    interval in lockstep with the edge table -- exactly the affine bus-segment ticks.  The
    delays themselves are constant (a delay is not scannable)."""
    if not channel_delays:
        return []
    starts = _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    exclude = set(exclude_channels)
    bit_to_channel = {hardware_bits[ch]: ch for ch in hardware_bits}
    out: list[RuntimeDelayChannel] = []
    for bit, d in channel_delays.items():
        channel = bit_to_channel.get(bit)
        if channel is None or channel in exclude:
            continue
        channel_index = state.channels.index(channel)
        intervals: list[RuntimeDelayInterval] = []
        active: tuple[int, tuple[int, ...]] | None = None
        for period_index, period in enumerate(state.periods):
            value = int(period.states[channel_index])
            if value and active is None:
                active = starts[period_index]
            elif not value and active is not None:
                stop = starts[period_index]
                intervals.append(RuntimeDelayInterval(
                    start_tick=int(active[0]), stop_tick=int(stop[0]),
                    start_tick_coeffs=list(active[1]) or None, stop_tick_coeffs=list(stop[1]) or None))
                active = None
        if active is not None:
            stop = starts[-1]
            intervals.append(RuntimeDelayInterval(
                start_tick=int(active[0]), stop_tick=int(stop[0]),
                start_tick_coeffs=list(active[1]) or None, stop_tick_coeffs=list(stop[1]) or None))
        out.append(RuntimeDelayChannel(bit=int(bit), delay=int(d), intervals=intervals or None))
    return out


def _pulse_table_affine_loop_metadata(
    state: PulseTableState,
    *,
    rows: Sequence[tuple[int, int, tuple[int, ...]]],
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
) -> tuple[int, int, list[int], int]:
    if state.repeat_start is None or state.repeat_end is None or state.repeat_count <= 1:
        return 0, int(rows[-1][0]), list(rows[-1][2]), 1
    starts = _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    loop_start = starts[int(state.repeat_start)]
    loop_end = starts[int(state.repeat_end) + 1]
    loop_start_index = _affine_row_index(rows, loop_start)
    return loop_start_index, int(loop_end[0]), list(loop_end[1]), int(state.repeat_count)


def _stable_affine_groups(
    events: Sequence[tuple[tuple[int, tuple[int, ...]], str | None, int | None]],
    *,
    scan_points: Sequence[Sequence[int]],
    coeff_frac_bits: int,
) -> list[tuple[tuple[int, tuple[int, ...]], list[tuple[str | None, int | None]]]]:
    if not scan_points:
        raise ValueError("hardware scan requires at least one scan point.")
    point0 = scan_points[0]
    by_ref: dict[int, list[tuple[tuple[int, tuple[int, ...]], str | None, int | None]]] = {}
    for expr, channel, value in events:
        tick0 = _apply_affine_ticks(expr[0], expr[1], point0, coeff_frac_bits)
        if tick0 < 0:
            raise ValueError("hardware scan produced a negative edge tick at the first scan point.")
        by_ref.setdefault(tick0, []).append((expr, channel, value))
    grouped: list[tuple[tuple[int, tuple[int, ...]], list[tuple[str | None, int | None]]]] = []
    for _tick0, items in sorted(by_ref.items(), key=lambda item: item[0]):
        # Events that share a reference tick but differ in affine expr only conflict if
        # MORE THAN ONE distinct expr carries a real CHANNEL transition (channel != None):
        # that is a genuine cross-channel reorder the single sorted table cannot play.  A
        # no-op ANCHOR (final/loop marker, channel None) coinciding with a real edge at the
        # reference point but diverging elsewhere is NOT a reorder -- it just lands on its
        # own row at its own (extending) expr.
        by_expr: dict[tuple[int, tuple[int, ...]], list[tuple[str | None, int | None]]] = {}
        for expr, channel, value in items:
            by_expr.setdefault((int(expr[0]), tuple(int(c) for c in expr[1])), []).append((channel, value))
        channel_exprs = [key for key, evs in by_expr.items() if any(ch is not None for ch, _v in evs)]
        if len(channel_exprs) > 1:
            raise ValueError(
                "this scan moves one channel's edges PAST another channel's edges as the "
                "scanned delay sweeps (the channels reorder), which the single global edge "
                "table cannot play.  Keep the scanned delay small enough that the channel "
                "stays in its own slot relative to the others, OR scan a DAC delay (analog "
                "buses are independent timelines and may reorder freely).  Reordering "
                "digital-delay scans need the per-channel delay-lane path (planned)."
            )
        # All events here share a reference tick.  A no-op ANCHOR (final/loop marker) that
        # coincides with a real edge at the reference but DIVERGES at other points is kept
        # as its OWN row at its own (extending) expr -- it carries no channel transition, so
        # it never moves a real edge; the table is still strictly increasing per point
        # because at the reference the rows are at the same tick (deduped downstream) and
        # the per-point monotonicity is enforced on real edges only.  Real channel edges go
        # first; a diverging anchor follows.
        ordered = sorted(by_expr, key=lambda k: (not any(ch is not None for ch, _v in by_expr[k]), k))
        for key in ordered:
            grouped.append(((key[0], key[1]), by_expr[key]))
    # Per-channel monotonicity: each channel's OWN edges must stay strictly ordered
    # (and non-negative) at every scan point -- a channel reversing/colliding its own
    # edges is unrepresentable.  This is NECESSARY but NOT sufficient for the FINAL
    # design: the engine is a single GLOBAL edge-table player, so the MERGED edge list
    # must also stay globally tick-monotone at every scan point.  That global check is
    # enforced downstream by ``validate_pulse_streamer_program`` (host prepare), which
    # rejects a scan that reorders edges across channels rather than dropping them.
    for point_index, point in enumerate(scan_points):
        per_chan_last: dict[str, int] = {}
        for expr, items in grouped:
            tick = _apply_affine_ticks(expr[0], expr[1], point, coeff_frac_bits)
            if tick < 0:
                raise ValueError(f"hardware scan produced a negative edge tick at scan point {point_index}.")
            for channel, _value in items:
                if channel is None:
                    continue  # final/loop markers are not channel edges
                previous = per_chan_last.get(channel)
                if previous is not None and tick <= previous:
                    raise ValueError(
                        f"hardware scan reverses or collides channel '{channel}'s own edges at "
                        f"scan point {point_index}; simplify that channel's delay/duration scan "
                        "(a single channel cannot run its own pulses backwards)."
                    )
                per_chan_last[channel] = tick
    # CROSS-CHANNEL reorder/collision at ANY scan point (not just the reference): the
    # single global sorted table needs the rows STRICTLY increasing AND in a fixed order at
    # every scan point.  If two rows that BOTH carry a real channel transition swap order or
    # collide (same effective tick) at some point as the scanned delay sweeps, the channels
    # reorder -- raise so the caller pulls the scanned-delay channel onto its own lane (the
    # reference-only check above misses a reorder that only appears at a later point).
    edge_rows = [(expr, items) for expr, items in grouped if any(ch is not None for ch, _v in items)]
    for point_index, point in enumerate(scan_points):
        last = None
        for expr, _items in edge_rows:
            tick = _apply_affine_ticks(expr[0], expr[1], point, coeff_frac_bits)
            if last is not None and tick <= last:
                raise ValueError(
                    "this scan moves one channel's edges PAST another channel's edges as the "
                    "scanned delay sweeps (the channels reorder), which the single global edge "
                    "table cannot play.  Reordering digital-delay scans use the per-channel "
                    "delay-lane path."
                )
            last = tick
    return grouped


def _affine_row_index(rows: Sequence[tuple[int, int, tuple[int, ...]]], expr: tuple[int, tuple[int, ...]]) -> int:
    target = (int(expr[0]), tuple(int(coeff) for coeff in expr[1]))
    for index, row in enumerate(rows):
        if (int(row[0]), tuple(int(coeff) for coeff in row[2])) == target:
            return index
    raise ValueError("repeat bracket start does not match a hardware scan edge row.")


def _affine_add(left: tuple[int, tuple[int, ...]], right: tuple[int, tuple[int, ...]]) -> tuple[int, tuple[int, ...]]:
    return int(left[0]) + int(right[0]), tuple(int(a) + int(b) for a, b in zip(left[1], right[1]))


def _apply_affine_ticks(base: int, coeffs: Sequence[int], slot_ticks: Sequence[int], coeff_frac_bits: int) -> int:
    total = sum(int(coeff) * int(tick) for coeff, tick in zip(coeffs, slot_ticks))
    return int(base) + (total >> int(coeff_frac_bits))


def _affine_ticks_matrix(exprs, scan_points, coeff_frac_bits):
    """``(len(exprs), N)`` numpy int64 array of effective ticks for every expr at every
    scan point -- the VECTORISED form of ``_apply_affine_ticks`` over a whole sweep.  The
    affine compile evaluates the same exprs at every scan point several times (global
    shift G, frame-end domination, global monotonicity); at thousands of points a Python
    double loop dominates ``compile_scan`` (~0.7 s at 4096 pts).  numpy's ``@`` + arithmetic
    ``>>`` (sign-extending = floor, identical to Python ``>>``) make it ~milliseconds and
    BIT-IDENTICAL to ``_apply_affine_ticks``.  ``np.int64`` holds the worst-case dot
    (coeff 2^15 x slot 2^24 x 4 slots = 2^41)."""
    import numpy as np

    points = [list(p) for p in scan_points] or [[]]
    n = len(points)
    frac = int(coeff_frac_bits)
    slots = len(points[0]) if points and points[0] else max((len(c) for _b, c in exprs), default=0)
    if not exprs:
        return np.zeros((0, n), dtype=np.int64)
    if slots == 0:
        return np.array([[int(base)] * n for base, _c in exprs], dtype=np.int64)
    pts = np.array([(list(p) + [0] * slots)[:slots] for p in points], dtype=np.int64)  # (N, slots)
    rows = []
    for base, coeffs in exprs:
        c = np.array((list(coeffs) + [0] * slots)[:slots], dtype=np.int64)
        rows.append(int(base) + ((pts @ c) >> frac))
    return np.stack(rows)


def _time_ns_to_ticks(value_ns: float, time_step_ns: float, name: str, *, allow_negative: bool = False) -> int:
    raw = float(value_ns) / float(time_step_ns)
    # Auto-snap to the nearest tick (ties away from zero) instead of rejecting an
    # off-grid value.  Scan-table points are arbitrary floats; the clock can only
    # land on whole ticks, so we round rather than raise.
    ticks = int(math.floor(raw + 0.5)) if raw >= 0 else int(math.ceil(raw - 0.5))
    if ticks < 0 and not allow_negative:
        ticks = 0
    return ticks


def _pulse_table_effective_duration_ticks(
    state: PulseTableState,
    *,
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
) -> int:
    starts = _pulse_table_period_starts_ticks(state, slots=slots, time_step_ns=time_step_ns)
    if state.repeat_start is None or state.repeat_end is None or state.repeat_count <= 1:
        return starts[-1]
    loop_ticks = starts[int(state.repeat_end) + 1] - starts[int(state.repeat_start)]
    return starts[-1] + (int(state.repeat_count) - 1) * loop_ticks


def _pulse_table_edge_table(
    state: PulseTableState,
    *,
    channels: Sequence[str],
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
    fold_analog_buses: bool = True,
    repeat_forever: bool = True,
) -> tuple[list[int], list[int], list[str], int, int, dict[int, int]]:
    """Build ``(ticks, masks, channels, loop_end, repeat_from_index, channel_delays)``.

    The edge table is UNDELAYED: every channel sits at its nominal position and the loop
    period is the plain frame end ``table_end`` (``repeat_from_index`` always 0).  A channel
    delay is NOT baked into the ticks -- it is applied to the engine OUTPUT by a per-channel
    delay line (output_delayed[t] = output_undelayed[t-d], zero before fire).  This is the
    literal physical delay: ANY length, never disturbs another channel, first frame real.
    ``channel_delays`` maps output-bit -> delay in ticks (only nonzero entries).

    A NEGATIVE delay re-translates the WHOLE frame, so the global shift ``G = max(0, -min
    delay)`` is FOLDED INTO every channel's delay (a causal delay line cannot lead): every
    returned delay is ``raw_delay + G >= 0``, preserving relative timing."""
    hardware_channels = list(channel_names(channels, "channels"))
    starts = _pulse_table_period_starts_ticks(state, slots=slots, time_step_ns=time_step_ns)
    table_end = int(starts[-1])
    channel_bits = {channel: index for index, channel in enumerate(hardware_channels)}
    bus_groups = state.bus_channels()
    bus_members = {channel for members in bus_groups.values() for channel in members}

    # --- per-channel UN-delayed ON intervals over [0, T) + each channel's raw delay ---
    base_intervals: dict[str, list[tuple[int, int]]] = {}
    raw_delay: dict[str, int] = {}
    for channel_index, channel in enumerate(state.channels):
        if channel in bus_members or channel not in channel_bits:
            continue
        raw_delay[channel] = state.delay_steps(channel, slots=slots, time_step_ns=time_step_ns)
        ivals, active = [], None
        for period_index, period in enumerate(state.periods):
            if int(period.states[channel_index]) and active is None:
                active = int(starts[period_index])
            elif not int(period.states[channel_index]) and active is not None:
                ivals.append((active, int(starts[period_index]))); active = None
        if active is not None:
            ivals.append((active, table_end))
        base_intervals[channel] = ivals
    if fold_analog_buses:
        for bus_name, members in bus_groups.items():
            bus_delay = _pulse_table_bus_delay_steps(state, members, slots=slots, time_step_ns=time_step_ns)
            plan = state.analog_bus_plan(bus_name)
            bus_ticks = sorted(set(_pulse_table_analog_bus_ticks(plan, starts)) | {0})
            for tick in bus_ticks:
                if tick < 0 or tick > table_end:
                    raise ValueError(f"analog bus {bus_name!r} produced edge tick {tick} outside the uploaded table.")
            for bit, channel in enumerate(members):
                if channel not in channel_bits:
                    continue
                raw_delay[channel] = bus_delay
                ivals, active = [], None
                for tick in bus_ticks:
                    on = (int(_pulse_table_analog_bus_value_at_tick(plan, starts, tick)) >> bit) & 1
                    if on and active is None:
                        active = int(tick)
                    elif not on and active is not None:
                        ivals.append((active, int(tick))); active = None
                if active is not None:
                    ivals.append((active, table_end))
                base_intervals[channel] = ivals

    # PHYSICAL DELAY: edges are emitted UNDELAYED (every channel at its nominal position);
    # each channel's delay is applied to the engine OUTPUT (a per-channel delay line), NOT
    # baked into the ticks.  ``channel_delays`` carries it.  A NEGATIVE delay re-translates
    # the WHOLE frame, so fold the global shift G = max(0, -min delay) into EVERY channel's
    # delay -- a causal delay line cannot lead, so shifting everyone by G makes all delays
    # >= 0 while preserving relative timing (the old in-edge G, now an output delay).
    global_shift = max(0, -min(raw_delay.values())) if raw_delay else 0
    channel_delays = {channel: raw_delay[channel] + global_shift for channel in raw_delay}

    # --- emit UNDELAYED ON/OFF events (channel=None entries are period-boundary anchors) ---
    events: list[tuple[int, str | None, int | None]] = []
    loop_end = table_end
    for tick in starts:
        events.append((int(tick), None, None))
    for channel, ivals in base_intervals.items():
        for a, b in ivals:
            events.append((a, channel, 1)); events.append((b, channel, 0))

    grouped: dict[int, list[tuple[str | None, int | None]]] = {}
    for tick, channel, value in events:
        if tick < 0 or tick > loop_end:
            raise ValueError(f"pulse table edge tick {tick} is outside the uploaded table [0, {loop_end}].")
        grouped.setdefault(int(tick), []).append((channel, value))

    ticks: list[int] = []
    masks: list[int] = []
    current_mask = 0
    for tick in sorted(grouped):
        for channel, value in grouped[tick]:
            if channel is None or value is None:
                continue
            bit = channel_bits[channel]
            if int(value):
                current_mask |= 1 << bit
            else:
                current_mask &= ~(1 << bit)
        ticks.append(int(tick))
        masks.append(int(current_mask))
    ticks, masks = _dedupe_same_tick_edges(ticks, masks)
    # Anchor an edge at tick 0 (all-off if nothing starts there): the engine seeds its
    # time counter from edge 0, so the table must begin at tick 0 or every edge slips a
    # tick.  A delayed channel that starts later, or an all-off opening period, both need
    # this explicit tick-0 anchor.
    if not ticks or ticks[0] != 0:
        ticks = [0] + ticks
        masks = [0] + masks
    ticks, masks = _ensure_final_off_edge(ticks, masks, loop_end)

    # The frame is UNDELAYED, so the loop always replays the WHOLE frame (period = table_end);
    # the per-channel output delay line, not a steady-frame rewind, produces the real startup.
    repeat_from_index = 0
    channel_delays_by_bit = {channel_bits[ch]: int(d) for ch, d in channel_delays.items() if int(d) != 0}
    # The UNBOUNDED membership delay player needs each delayed channel's OWN undelayed ON
    # intervals over [0, T) -- the engine evaluates membership at the shifted phase instead of
    # buffering.  No scan here -> the intervals are LITERAL (zero affine coeffs).
    delay_channels = [
        RuntimeDelayChannel(
            bit=channel_bits[ch],
            delay=int(channel_delays[ch]),
            intervals=[RuntimeDelayInterval(start_tick=int(a), stop_tick=int(b)) for a, b in base_intervals.get(ch, [])] or None,
        )
        for ch in channel_delays if int(channel_delays[ch]) != 0
    ]
    return ticks, masks, hardware_channels, loop_end, repeat_from_index, channel_delays_by_bit, delay_channels


def _dedupe_same_tick_edges(ticks: Sequence[int], masks: Sequence[int]) -> tuple[list[int], list[int]]:
    out_ticks: list[int] = []
    out_masks: list[int] = []
    for tick, mask in zip(ticks, masks):
        tick = int(tick)
        mask = int(mask)
        if out_ticks and out_ticks[-1] == tick:
            out_masks[-1] = mask
            continue
        out_ticks.append(tick)
        out_masks.append(mask)
    return out_ticks, out_masks


def _pulse_table_bus_delay_steps(
    state: PulseTableState,
    members: Sequence[str],
    *,
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
) -> int:
    delays = {
        state.delay_steps(channel, slots=slots, time_step_ns=time_step_ns)
        for channel in members
    }
    if len(delays) > 1:
        raise ValueError("all bit channels in one analog bus must share the same delay.")
    return next(iter(delays), 0)


def _pulse_table_analog_bus_ticks(plan: Sequence[Mapping[str, object]], starts: Sequence[int]) -> list[int]:
    ticks = {int(starts[index]) for index in range(max(0, len(starts) - 1))}
    anchors: list[tuple[int, int, str, int]] = []
    for index, entry in enumerate(plan):
        mode = str(entry.get("mode", "hold")).lower()
        if mode not in {"edge", "ramp"} or entry.get("value") is None:
            continue
        anchors.append((index, int(starts[index]), mode, int(entry["value"])))
    if anchors:
        ticks.add(anchors[0][1])
    previous = anchors[0] if anchors else None
    for anchor in anchors[1:]:
        ticks.add(anchor[1])
        if previous is not None and anchor[2] == "ramp":
            start_tick = previous[1]
            stop_tick = anchor[1]
            start_value = previous[3]
            stop_value = anchor[3]
            span = stop_tick - start_tick
            steps = abs(stop_value - start_value)
            if span > 0 and steps > 0:
                last_tick = start_tick
                for step in range(1, steps + 1):
                    tick = int(round(start_tick + span * (step / steps)))
                    tick = max(start_tick, min(stop_tick, tick))
                    if tick <= last_tick and last_tick < stop_tick:
                        tick = last_tick + 1
                    if tick <= stop_tick:
                        ticks.add(tick)
                        last_tick = tick
        previous = anchor
    ticks.add(int(starts[-1]))
    return sorted(ticks)


def _pulse_table_analog_bus_value_at_tick(plan: Sequence[Mapping[str, object]], starts: Sequence[int], tick: int) -> int:
    anchors: list[tuple[int, int, str, int]] = []
    for index, entry in enumerate(plan):
        mode = str(entry.get("mode", "hold")).lower()
        value = entry.get("value")
        if mode in {"edge", "ramp"} and value is not None:
            anchors.append((index, int(starts[index]), mode, int(value)))
    if not anchors:
        return 0
    tick = int(tick)
    if tick < anchors[0][1]:
        return 0
    previous = anchors[0]
    for anchor in anchors[1:]:
        if tick < anchor[1]:
            if anchor[2] == "ramp" and anchor[1] > previous[1]:
                fraction = (tick - previous[1]) / (anchor[1] - previous[1])
                return int(round(previous[3] + (anchor[3] - previous[3]) * fraction))
            return int(previous[3])
        previous = anchor
    return int(previous[3])


def _pulse_table_bus_order(bus_groups: Mapping[str, Sequence[str]]) -> list[str]:
    """Return the HDL bus order, keeping address-switch buses stable."""

    names = [name for name in DEFAULT_RUNTIME_BUS_NAMES if name in bus_groups]
    names.extend(name for name in bus_groups if name not in names)
    return names


def _slot_ref_index(value: object, slot_vars: Sequence[str]) -> int | None:
    """Return the scan-slot column index a bus value references, else ``None``.

    A scanned DAC level is stored in the analog-bus plan as a slot variable name
    such as ``"s2"``; this maps it back to its column index so a bus segment can
    carry ``value_select = index + 1`` instead of a literal DAC code.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in slot_vars:
        return list(slot_vars).index(text)
    if len(text) >= 2 and text[0] == "s" and text[1:].isdigit():
        index = int(text[1:])
        if 0 <= index < len(slot_vars):
            return index
    return None


def _pulse_table_bus_segments(
    state: PulseTableState,
    *,
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
    slot_vars: Sequence[str] | None = None,
    coeff_frac_bits: int = 8,
) -> tuple[list[str], list[RuntimeBusSegment]]:
    """Compile logical analog buses into hardware bus segments.

    A ramp consumes one segment regardless of how many 10-bit stair steps it
    produces.  Digital edge rows are left for ordinary TTL outputs.  When a bus
    value references a scan slot (``slot_vars`` given), the segment carries a
    ``value_select`` so the DAC level is read from that slot per scan point.

    With ``slot_vars`` the segment *ticks* are emitted as affine expressions
    (base + per-slot coefficients), exactly like the digital edges, so a scanned
    duration/delay moves the analog segment in lockstep -- this is what lets DAC
    value + duration + delay scan simultaneously.
    """

    slot_vars = list(slot_vars or [])
    affine = bool(slot_vars)
    zero_coeffs = tuple(0 for _ in slot_vars)
    starts = _pulse_table_period_starts_ticks(state, slots=slots, time_step_ns=time_step_ns)
    table_end = int(starts[-1])
    affine_starts = (
        _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
        if affine
        else None
    )
    bus_groups = state.bus_channels()
    bus_names = _pulse_table_bus_order(bus_groups)
    segments: list[RuntimeBusSegment] = []
    for bus_index, bus_name in enumerate(bus_names):
        members = bus_groups[bus_name]
        plan = state.analog_bus_plan(bus_name)
        if bus_name not in state.analog_bus_modes and all(state.bus_value(index, bus_name) == 0 for index in range(len(state.periods))):
            continue
        delay_steps = _pulse_table_bus_delay_steps(state, members, slots=slots, time_step_ns=time_step_ns)
        delay_aff = (
            _affine_expr(state.delays.get(members[0], 0.0), state.delay_units.get(members[0], "ns"), slot_vars, time_step_ns, coeff_frac_bits)
            if affine
            else (delay_steps, zero_coeffs)
        )
        # anchor: (period_index, ref_tick, base_tick, coeffs, mode, value_int, value_select)
        anchors: list[tuple[int, int, int, tuple[int, ...], str, int, int]] = []
        max_value = (1 << len(members)) - 1
        for period_index, entry in enumerate(plan):
            mode = str(entry.get("mode", "hold")).strip().lower()
            value = entry.get("value")
            if mode not in {"edge", "ramp"} or value is None:
                continue
            ref_index = _slot_ref_index(value, slot_vars)
            if ref_index is not None:
                value_select = ref_index + 1
                value_int = 0  # placeholder; the FPGA reads the slot at runtime
            else:
                value_select = 0
                value_int = max(0, min(max_value, int(value)))
            ref_tick = int(starts[period_index]) + delay_steps
            if affine:
                base, coeffs = _affine_add(affine_starts[period_index], delay_aff)
            else:
                base, coeffs = ref_tick, zero_coeffs
            anchors.append((period_index, ref_tick, int(base), tuple(coeffs), mode, value_int, value_select))

        def _coeffs(values: tuple[int, ...]) -> list[int] | None:
            return list(values) if affine else None

        previous: tuple[int, int, int, tuple[int, ...], str, int, int] | None = None
        for anchor_index, anchor in enumerate(anchors):
            _period_index, ref_tick, base, coeffs, mode, value, value_select = anchor
            if ref_tick < 0 or ref_tick > table_end:
                raise ValueError(f"analog bus {bus_name!r} produced segment tick {ref_tick} outside the uploaded table.")
            if previous is None:
                next_anchor = anchors[anchor_index + 1] if anchor_index + 1 < len(anchors) else None
                if next_anchor is None or str(next_anchor[4]).lower() != "ramp":
                    segments.append(RuntimeBusSegment(bus_index, base, base, value, value, "edge", bus_name, value_select, _coeffs(coeffs), _coeffs(coeffs), stop_value_select=value_select))
            elif mode == "ramp":
                start_ref = int(previous[1])
                start_base = int(previous[2])
                start_coeffs = previous[3]
                start_value = int(previous[5])
                if ref_tick < start_ref:
                    raise ValueError(f"analog bus {bus_name!r} ramp end precedes its start.")
                # A ramp may scan EITHER endpoint independently: the start value reads
                # the previous anchor's slot (previous[6]); the stop value reads this
                # anchor's slot (value_select).  The RTL bus engine's dual value_select
                # makes a ramp scanned-A -> scanned-B seamless.
                segments.append(RuntimeBusSegment(
                    bus_index, start_base, base, start_value, value, "ramp", bus_name,
                    previous[6], _coeffs(start_coeffs), _coeffs(coeffs),
                    stop_value_select=value_select,
                ))
            else:
                next_anchor = anchors[anchor_index + 1] if anchor_index + 1 < len(anchors) else None
                if next_anchor is None or str(next_anchor[4]).lower() != "ramp":
                    segments.append(RuntimeBusSegment(bus_index, base, base, value, value, "edge", bus_name, value_select, _coeffs(coeffs), _coeffs(coeffs), stop_value_select=value_select))
            previous = anchor
    return bus_names, segments


def _pulse_table_has_analog_activity(state: PulseTableState) -> bool:
    for bus_name in state.bus_channels():
        if bus_name in state.analog_bus_modes:
            return True
        if any(state.bus_value(index, bus_name) != 0 for index in range(len(state.periods))):
            return True
    return False


def _pulse_table_has_analog_ramp(state: PulseTableState) -> bool:
    for bus_name in state.bus_channels():
        for entry in state.analog_bus_plan(bus_name):
            if str(entry.get("mode", "hold")).lower() == "ramp":
                return True
    return False


def _pulse_table_has_any_delay(state: PulseTableState) -> bool:
    """True if any channel has a delay that is nonzero OR scanned (a slot expression).

    The scan path must UNROLL a bracket when ANY channel carries a delay -- including a
    SCANNED delay whose reference value is 0 (so :func:`_pulse_table_has_delays`, which
    resolves the reference slot, would miss it).  A delay bound to a slot is always
    treated as present; a literal delay counts only when it rounds to a nonzero tick.
    """

    for channel in state.channels:
        raw = state.delays.get(channel, 0.0)
        if isinstance(raw, str) and not _is_plain_number(raw):
            return True  # scanned / expression delay (e.g. "s0", "20+s1")
        if state.delay_steps(channel, slots=state.reference_slots(), time_step_ns=state.time_step_ns) != 0:
            return True
    return False


def _is_plain_number(value: object) -> bool:
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _check_unrolled_edge_budget(
    state: PulseTableState,
    *,
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
) -> None:
    """Raise a clear, actionable error if unrolling the bracket would overflow the edge
    budget (a large ``repeat_count`` makes a flat edge table that the streamer cannot
    hold).  ``validate_pulse_streamer_program`` is the authoritative gate; this just
    front-loads a friendlier message that names the inner repeat as the cause."""

    from .fpga_pulse_streamer import DEFAULT_MAX_EDGES

    # 2 edges per ON run + a tick-0 anchor + a final off edge is a generous upper bound;
    # the real count is <= this, so we never reject a program the streamer could hold.
    n_periods = len(state.periods)
    upper_bound_edges = 2 * len(state.channels) * n_periods + 2
    if upper_bound_edges > DEFAULT_MAX_EDGES:
        raise ValueError(
            f"unrolling the inner repeat bracket would make up to {upper_bound_edges} edges, "
            f"above the FPGA streamer budget of {DEFAULT_MAX_EDGES}.  Use repeat_forever for the "
            "OUTER loop, fewer inner iterations, or remove the channel delay so the bracket can "
            "stay a compact hardware loop instead of being unrolled."
        )


def _edge_index_at_or_after(ticks: Sequence[int], tick: int) -> int:
    for index, candidate in enumerate(ticks):
        if int(candidate) >= int(tick):
            return index
    raise ValueError(f"repeat bracket starts at tick {tick}, but no edge exists at or after that tick.")


def _pulse_table_trigger_count(
    state: PulseTableState,
    *,
    trigger_channels: Sequence[str],
) -> int:
    trigger_channels = list(channel_names(trigger_channels, "trigger_channels", allow_empty=True))
    total = 0
    counted: set[str] = set()
    for trigger in trigger_channels:
        candidates = [trigger] if trigger in state.channels else []
        trigger_label = str(trigger).strip().lower()
        candidates.extend(
            channel
            for channel, label in state.channel_labels.items()
            if channel in state.channels and str(label).strip().lower() == trigger_label
        )
        for channel in candidates:
            if channel in counted:
                continue
            counted.add(channel)
            total += _pulse_table_channel_rises(state, channel)
    return total


def _pulse_table_channel_rises(state: PulseTableState, channel: str) -> int:
    index = state.channel_index(channel)
    states = [int(period.states[index]) for period in state.periods]
    if state.repeat_start is None or state.repeat_end is None or state.repeat_count <= 1:
        count, _last = _count_rises(states, initial=0)
        return count

    repeat_start = int(state.repeat_start)
    repeat_end = int(state.repeat_end)
    repeat_count = int(state.repeat_count)
    pre = states[:repeat_start]
    loop = states[repeat_start : repeat_end + 1]
    post = states[repeat_end + 1 :]

    count, last = _count_rises(pre, initial=0)
    first_count, last_after_loop = _count_rises(loop, initial=last)
    count += first_count
    if repeat_count > 1:
        loop_again_count, last_after_loop = _count_rises(loop, initial=last_after_loop)
        count += (repeat_count - 1) * loop_again_count
    post_count, _last = _count_rises(post, initial=last_after_loop)
    return count + post_count


def _count_rises(states: Sequence[int], *, initial: int) -> tuple[int, int]:
    last = 1 if int(initial) else 0
    count = 0
    for state in states:
        state = 1 if int(state) else 0
        if state and not last:
            count += 1
        last = state
    return count, last


def _pulse_table_has_delays(
    state: PulseTableState,
    *,
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
) -> bool:
    return any(state.delay_steps(channel, slots=slots, time_step_ns=time_step_ns) != 0 for channel in state.channels)


def _plain_rpc_payload(value):
    """Recursively convert RPyC netrefs/proxies into local JSON-like objects."""

    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _plain_rpc_payload(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return [_plain_rpc_payload(item) for item in value]
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return [_plain_rpc_payload(item) for item in iterator]


def nonnegative_float(value, name: str) -> float:
    out = float(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0.")
    return out


def serve_runtime_sequencer(
    service: SequencerService,
    *,
    host: str = "0.0.0.0",
    port: int = 18861,
    start: bool = True,
):
    """Expose ``SequencerService`` over RPyC on the FPGA/Vivado computer."""

    try:
        import rpyc
        from rpyc.utils.server import ThreadedServer
    except ImportError as exc:  # pragma: no cover - depends on lab install
        raise RuntimeError("serve_runtime_sequencer requires `rpyc` on the FPGA computer.") from exc

    class RPyCSequencerService(rpyc.Service):
        def exposed_prepare(self, sequence_payload):
            return json.dumps(service.prepare(sequence_payload))

        def exposed_fire(self, sequence_payload=None):
            return json.dumps(service.fire(sequence_payload))

        def exposed_wait_done(self, timeout=None):
            return service.wait_done(timeout)

        def exposed_abort(self):
            return service.abort()

        def exposed_set_safe_state(self):
            return service.set_safe_state()

        def exposed_snapshot(self):
            return service.snapshot()

    server = ThreadedServer(
        RPyCSequencerService,
        hostname=host,
        port=int(port),
        protocol_config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": None},
    )
    if start:
        server.start()
    return server


__all__ = [
    "ManualSequencer",
    "PulseController",
    "RemoteSequencer",
    "RuntimeBusSegment",
    "RuntimeSequenceProgram",
    "RuntimeSequencer",
    "SequencerService",
    "VerilogSequencer",
    "bind_pulse",
    "compile_pulse_table_runtime_program",
    "compile_pulse_table_scan_runtime_program",
    "compile_runtime_program",
    "compile_runtime_program_for_payload",
    "serve_runtime_sequencer",
]
