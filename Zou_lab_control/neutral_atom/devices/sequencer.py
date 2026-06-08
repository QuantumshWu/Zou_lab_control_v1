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
class DelayLane:
    """One digital channel whose SCANNED delay reorders its edges past other channels.

    Such a channel cannot stay in the single global sorted edge table (the merged order
    changes per scan point).  It is pulled out onto its own DISJOINT output bit and
    played by a tiny per-channel sub-player -- structurally the per-bus DAC engine
    (``zlc_bus_tick``) specialised to a 1-bit value.  Because the hardware delay is
    ADDITIVE (not cyclic), each edge tick is ``period_start +/- delay`` -- AFFINE in the
    scan slots -- so the lane reuses the same ``effective_tick`` MAC and the same 4
    gapless boundary reseeds as the main table and the bus engine; the disjoint bit means
    no global re-sort is ever needed.

    ``ticks[i]`` / ``coeffs[i]`` are the affine base tick + per-slot coefficients of
    edge ``i`` (sorted by reference-point effective tick); ``values[i]`` is 0/1.  The
    final edge is the channel's last fall, so the lane returns to 0 within the frame.
    """

    channel: str
    channel_bit: int
    ticks: list[int]
    coeffs: list[list[int]]
    values: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": str(self.channel),
            "channel_bit": int(self.channel_bit),
            "ticks": [int(t) for t in self.ticks],
            "coeffs": [[int(c) for c in row] for row in self.coeffs],
            "values": [int(v) for v in self.values],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DelayLane":
        return cls(
            channel=str(payload.get("channel", "")),
            channel_bit=int(payload.get("channel_bit", 0)),
            ticks=[int(t) for t in payload.get("ticks", [])],
            coeffs=[[int(c) for c in row] for row in payload.get("coeffs", [])],
            values=[int(v) for v in payload.get("values", [])],
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
    delay_lanes: list[DelayLane] | None = None

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
            "delay_lanes": [lane.to_dict() for lane in (self.delay_lanes or [])],
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
            delay_lanes=[DelayLane.from_dict(item) for item in payload.get("delay_lanes", [])] or None,
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
    ticks, masks, channels, loop_end, repeat_from_index = _pulse_table_edge_table(
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
    }
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

    points_ticks: list[list[int]] = []
    for point_index, row in enumerate(table):
        ns = point_slots_ns(row)
        points_ticks.append([
            point_slot_value(point_index, index, ns) for index in range(len(state.scan_slots))
        ])
        state.validate(slots=ns, time_step_ns=clock_step_ns)

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

    # GLOBAL SHIFT G + FRAME END, computed ONCE over ALL non-bus channels (the union of
    # main-table and would-be lane channels) so the main table and the lanes agree on the
    # frame re-translation (negative scanned delay) and on the frame length (a large
    # scanned delay extends it).  ``frame_end`` is a single affine expr that dominates
    # every channel's latest delayed edge at every scan point, or None (the affine
    # candidates cross -> fall back to the main rows' own per-reference max).
    global_shift = _pulse_table_affine_global_shift(
        state, scan_points=points_ticks, slot_vars=slot_vars, time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits, exclude_channels=bus_members,
    )
    frame_end = _pulse_table_affine_frame_end(
        state, scan_points=points_ticks, slot_vars=slot_vars, time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits, exclude_channels=bus_members, global_shift=global_shift,
    )

    # Every channel with a SCANNED delay rides its own DISJOINT-bit delay lane (a 1-bit
    # affine sub-player), and is excluded from the main global edge table.  This is not an
    # optimisation choice but a CORRECTNESS one: a scanned-delay edge sweeps through tick
    # values and, after the global shift G, the minimum edge lands on tick 0 at the extreme
    # scan point -- which would collide with the table's mandatory tick-0 seed anchor, and
    # the single-edge-per-tick prefetch engine would slip that edge by one tick anyway.  A
    # lane plays at its exact effective tick every scan point (its own sub-player reseeds to
    # lane-tick 0, no FIFO, no anchor), so a scanned delay of ANY form -- reordering past
    # other channels, crossing the (unrolled) bracket, negative (shared G), or frame-
    # extending (shared frame_end) -- is exact.  The main table then carries only constant/
    # zero-delay and non-delayed channels, whose period-0 edges share the anchor's zero-
    # coeff expression and merge cleanly.
    lane_channels, delay_lanes = _pulse_table_delay_lanes(
        state,
        channels=channels,
        scan_points=points_ticks,
        slot_vars=slot_vars,
        time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits,
        exclude_channels=bus_members,
        global_shift=global_shift,
        frame_end=frame_end,
    )
    rows = _pulse_table_affine_rows(
        state,
        channels=channels,
        scan_points=points_ticks,
        slot_vars=slot_vars,
        time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits,
        exclude_channels=list(bus_members) + lane_channels,
        global_shift=global_shift,
        frame_end=frame_end,
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
        "delay_lanes": [lane.to_dict() for lane in delay_lanes],
    }
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
        delay_lanes=delay_lanes or None,
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


def _pulse_table_affine_global_shift(
    state: PulseTableState,
    *,
    scan_points: Sequence[Sequence[int]],
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
    exclude_channels: Sequence[str] = (),
) -> int:
    """Single global shift G = max(0, -(min effective edge tick over ALL non-excluded
    channels AND all scan points)), including channels that will become delay LANES.

    The main table and the lanes reset their frame tick at the SAME boundary, so they
    must share ONE G; this computes it over the union of both, before the table/lane
    split.  ``exclude_channels`` drops analog-bus members (driven by the bus engine).
    """

    exprs = _pulse_table_affine_all_edge_exprs(
        state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits, exclude_channels=exclude_channels
    )
    return max(0, -_affine_min_over_points(exprs, scan_points, coeff_frac_bits))


def _pulse_table_affine_frame_end(
    state: PulseTableState,
    *,
    scan_points: Sequence[Sequence[int]],
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
    exclude_channels: Sequence[str],
    global_shift: int,
) -> tuple[int, tuple[int, ...]] | None:
    """The single AFFINE frame-end expr (with G applied) that DOMINATES every channel's
    latest delayed edge at EVERY scan point, or ``None`` if no single affine expr does.

    The engine's frame length for a scan point is the last main-table row's effective
    tick; the lanes reset their frame tick at the same boundary.  So the frame end must,
    at every scan point, be >= every (main + lane) edge -- a single affine expr -- for the
    frame to extend over a large/frame-extending scanned delay without cutting it off.
    A LARGER frame end at one point but a smaller candidate at another (the affine
    candidates cross) -> ``None`` (a genuinely unrepresentable frame; the caller falls
    back to the per-row reference max).

    The candidate set is every edge expr PLUS ``table_end + delay_ch`` for each channel:
    the latter is >= every edge of that channel (a delayed stop is at most the delayed
    table end), so the dominating frame end exists whenever a single channel carries the
    largest delay at every scan point -- the common case (one trigger/probe delay swept).
    """

    edge_exprs = _pulse_table_affine_all_edge_exprs(
        state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits, exclude_channels=exclude_channels
    )
    starts = _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    table_end = starts[-1]
    exclude = set(exclude_channels)
    candidates = list(edge_exprs)
    for channel in state.channels:
        if channel in exclude:
            continue
        delay = _affine_expr(state.delays.get(channel, 0.0), state.delay_units.get(channel, "ns"), slot_vars, time_step_ns, coeff_frac_bits)
        candidates.append(_affine_add(table_end, delay))   # frame end if THIS channel's delay dominates
    g = (int(global_shift), tuple(0 for _ in slot_vars))
    candidates = [_affine_add(expr, g) for expr in candidates]
    points = [list(point) for point in scan_points] or [[0] * len(slot_vars)]
    for cand in candidates:
        cand_ticks = [_apply_affine_ticks(cand[0], cand[1], point, coeff_frac_bits) for point in points]
        if all(
            cand_ticks[p] >= _apply_affine_ticks(other[0], other[1], point, coeff_frac_bits)
            for other in edge_exprs
            for p, point in enumerate(points)
        ):
            return cand
    return None


def _pulse_table_affine_rows(
    state: PulseTableState,
    *,
    channels: Sequence[str],
    scan_points: Sequence[Sequence[int]],
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
    exclude_channels: Sequence[str] = (),
    global_shift: int | None = None,
    frame_end: tuple[int, tuple[int, ...]] | None = None,
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Return one affine edge row ``(base_tick, mask, slot_coeffs)`` per edge.

    Every channel's rise/fall edge is ``period_start +/- delay`` evaluated
    affinely in the bound scan slots.  Delays compose additively with period
    durations, so a bound delay and a bound duration on the same edge sum their
    coefficients.  ``_stable_affine_groups`` validates PER-CHANNEL edge ordering and
    non-negativity at *every* scan point.  NOTE: the FINAL engine is a single GLOBAL
    edge-table player, so the merged edge list must ALSO stay globally tick-monotone
    at every scan point; ``validate_pulse_streamer_program`` enforces that (a scan
    that reorders the merged edges across channels is rejected, not silently dropped).

    ``global_shift`` (G) is the additive frame re-translation for negative scanned
    delays.  Pass it explicitly when delay LANES are also in play so the main table and
    the lanes share ONE G (their frame ticks must align); ``None`` computes G from these
    rows' own edges (the no-lane path).  ``frame_end`` (ALREADY shifted by G) forces the
    final-row tick so the frame EXTENDS to cover lane edges that push past these rows'
    own latest edge (the large/frame-extending scanned-delay-on-a-lane case).
    """

    hardware_channels = list(channel_names(channels, "channels"))
    exclude = set(exclude_channels)
    starts = _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    events: list[tuple[tuple[int, tuple[int, ...]], str | None, int | None]] = []
    final_expr = starts[-1]

    for channel_index, channel in enumerate(state.channels):
        if channel in exclude:
            continue  # analog-bus members are driven by the bus engine, not TTL edges
        delay = _affine_expr(state.delays.get(channel, 0.0), state.delay_units.get(channel, "ns"), slot_vars, time_step_ns, coeff_frac_bits)
        active_start: tuple[int, tuple[int, ...]] | None = None
        for period_index, period in enumerate(state.periods):
            value = int(period.states[channel_index])
            if value and active_start is None:
                active_start = starts[period_index]
            elif not value and active_start is not None:
                events.append((_affine_add(active_start, delay), channel, 1))
                events.append((_affine_add(starts[period_index], delay), channel, 0))
                final_expr = _affine_max_reference(final_expr, _affine_add(starts[period_index], delay), scan_points, coeff_frac_bits)
                active_start = None
        if active_start is not None:
            events.append((_affine_add(active_start, delay), channel, 1))
            events.append((_affine_add(starts[-1], delay), channel, 0))
            final_expr = _affine_max_reference(final_expr, _affine_add(starts[-1], delay), scan_points, coeff_frac_bits)

    if state.repeat_start is not None and state.repeat_end is not None and state.repeat_count > 1:
        events.append((starts[int(state.repeat_start)], None, None))

    # GLOBAL SHIFT G (negative scanned delay): G = max(0, -(min effective edge tick over
    # ALL channels AND all scan points)) -- the affine min over the scan points is at an
    # extreme point, so G is a single compile-time constant.  Adding G to every edge BASE
    # re-translates the WHOLE frame so the earliest event is >= 0 at every scan point; a
    # uniform base offset never reorders anything and the edges stay affine.  Mirrors the
    # additive-G logic in _pulse_table_edge_table.
    if global_shift is None:
        shift = max(0, -_affine_min_over_points([expr for expr, _c, _v in events] + [final_expr], scan_points, coeff_frac_bits))
    else:
        shift = int(global_shift)
    if shift:
        g_expr = (shift, tuple(0 for _ in slot_vars))
        events = [(_affine_add(expr, g_expr), channel, value) for expr, channel, value in events]
        final_expr = _affine_add(final_expr, g_expr)
    # The final (all-off) marker uses the EXTENDED, already-shifted frame end when one is
    # supplied (covers lane edges past these rows' own latest edge); else this table's own
    # shifted final.  If that frame end COINCIDES with a real edge at the reference point
    # (e.g. a lane delay is 0 at the reference, so the frame does not extend there) it would
    # produce two rows at the same tick -- bump it one tick later (a single harmless idle
    # tick at the frame end) so the table stays strictly increasing; it still extends to fit
    # the lane edges at the other scan points.
    final_marker = frame_end if frame_end is not None else final_expr
    ref_point = scan_points[0] if scan_points else ()
    marker_ref = _apply_affine_ticks(final_marker[0], final_marker[1], ref_point, coeff_frac_bits)
    real_edge_refs = {_apply_affine_ticks(expr[0], expr[1], ref_point, coeff_frac_bits)
                      for expr, channel, _v in events if channel is not None}
    if marker_ref in real_edge_refs:
        final_marker = (final_marker[0] + 1, final_marker[1])
    events.append((final_marker, None, None))

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


def _pulse_table_delay_lanes(
    state: PulseTableState,
    *,
    channels: Sequence[str],
    scan_points: Sequence[Sequence[int]],
    slot_vars: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
    exclude_channels: Sequence[str] = (),
    global_shift: int = 0,
    frame_end: tuple[int, tuple[int, ...]] | None = None,
) -> tuple[list[str], list[DelayLane]]:
    """Build a DELAY LANE for each digital channel whose DELAY is SCANNED.

    Because the hardware delay is ADDITIVE, a delayed edge tick is ``period_start +
    delay`` -- affine in the scan slots -- so the lane is a 1-bit affine sub-player on a
    DISJOINT output bit (structurally the per-bus DAC engine).  Pulling the channel out
    of the global sorted table is what lets a scanned delay REORDER its edges past other
    channels (the failing case the single table rejects), and a LARGE scanned delay push
    edges past the nominal frame -- the frame end EXTENDS to fit them.

    ``global_shift`` (G) is added to every lane edge so a NEGATIVE scanned delay aligns
    with the same re-translated frame as the main table; the lane and the main table share
    ONE G and reset their frame tick at the same boundary.  ``frame_end`` (the program's
    ACTUAL final tick expr, already extended to fit positive delayed edges and shifted by
    G) bounds the per-point in-frame check; an edge may extend the frame but not run
    backwards or leave the (extended) frame.  Returns the lane channel names (to exclude
    from the main edge rows) and the lanes."""

    hardware_channels = list(channel_names(channels, "channels"))
    channel_bits = {channel: index for index, channel in enumerate(hardware_channels)}
    exclude = set(exclude_channels)
    affine_starts = _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    g_expr = (int(global_shift), tuple(0 for _ in slot_vars))
    if frame_end is None:
        frame_end = _affine_add(affine_starts[-1], g_expr)
    frame_base, frame_coeffs = frame_end
    points = [list(point) for point in scan_points] or [[0] * len(slot_vars)]
    lane_channels: list[str] = []
    lanes: list[DelayLane] = []
    for channel_index, channel in enumerate(state.channels):
        if channel in exclude or channel not in channel_bits:
            continue
        delay_aff = _affine_expr(state.delays.get(channel, 0.0), state.delay_units.get(channel, "ns"), slot_vars, time_step_ns, coeff_frac_bits)
        if not any(delay_aff[1]):
            continue  # delay not scanned -> stays in the (cheaper) main edge table
        # the channel's ON period-runs -> affine rise(=1)/fall(=0) edges shifted by delay
        # AND by the shared global shift G (so the lane frame aligns with the main frame).
        shift_aff = _affine_add(delay_aff, g_expr)
        edges: list[tuple[tuple[int, tuple[int, ...]], int]] = []
        active: int | None = None
        for period_index, period in enumerate(state.periods):
            on = int(period.states[channel_index])
            if on and active is None:
                active = period_index
            elif not on and active is not None:
                edges.append((_affine_add(affine_starts[active], shift_aff), 1))
                edges.append((_affine_add(affine_starts[period_index], shift_aff), 0))
                active = None
        if active is not None:
            edges.append((_affine_add(affine_starts[active], shift_aff), 1))
            edges.append((_affine_add(affine_starts[len(state.periods)], shift_aff), 0))
        if not edges:
            continue
        ref = points[0]
        edges.sort(key=lambda e: _apply_affine_ticks(e[0][0], e[0][1], ref, coeff_frac_bits))
        for point_index, point in enumerate(points):
            frame_tick = _apply_affine_ticks(frame_base, frame_coeffs, point, coeff_frac_bits)
            prev = -1
            for (base, coeffs), _value in edges:
                tick = _apply_affine_ticks(base, coeffs, point, coeff_frac_bits)
                if tick < 0 or tick > frame_tick:
                    raise ValueError(
                        f"scanned delay on channel {channel!r} pushes an edge to tick {tick}, "
                        f"outside the (extended) frame [0, {frame_tick}] at scan point {point_index}; "
                        "the frame end could not be made to dominate this edge at every scan point."
                    )
                if tick <= prev:
                    raise ValueError(
                        f"scanned delay on channel {channel!r} reverses its own edges at scan "
                        f"point {point_index} (a channel cannot run its own pulses backwards)."
                    )
                prev = tick
        lanes.append(DelayLane(
            channel=channel,
            channel_bit=channel_bits[channel],
            ticks=[int(base) for (base, _c), _v in edges],
            coeffs=[list(coeffs) for (_b, coeffs), _v in edges],
            values=[int(value) for (_bc), value in edges],
        ))
        lane_channels.append(channel)
    return lane_channels, lanes


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


def _affine_min_over_points(
    exprs: Sequence[tuple[int, tuple[int, ...]]],
    scan_points: Sequence[Sequence[int]],
    coeff_frac_bits: int,
) -> int:
    """Smallest effective tick of any of ``exprs`` over ALL scan points.

    Used to size the global shift G for a negative scanned delay: the affine min over
    the scan points is reached at an extreme point, so the result is a single constant.
    """

    points = list(scan_points) or [()]
    best: int | None = None
    for base, coeffs in exprs:
        for point in points:
            tick = _apply_affine_ticks(base, coeffs, point, coeff_frac_bits)
            if best is None or tick < best:
                best = tick
    return 0 if best is None else int(best)


def _affine_max_reference(
    left: tuple[int, tuple[int, ...]],
    right: tuple[int, tuple[int, ...]],
    scan_points: Sequence[Sequence[int]],
    coeff_frac_bits: int,
) -> tuple[int, tuple[int, ...]]:
    if not scan_points:
        return right if right[0] > left[0] else left
    point = scan_points[0]
    left_tick = _apply_affine_ticks(left[0], left[1], point, coeff_frac_bits)
    right_tick = _apply_affine_ticks(right[0], right[1], point, coeff_frac_bits)
    return right if right_tick > left_tick else left


def _apply_affine_ticks(base: int, coeffs: Sequence[int], slot_ticks: Sequence[int], coeff_frac_bits: int) -> int:
    total = sum(int(coeff) * int(tick) for coeff, tick in zip(coeffs, slot_ticks))
    return int(base) + (total >> int(coeff_frac_bits))


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
) -> tuple[list[int], list[int], list[str], int, int]:
    """Build ``(ticks, masks, channels, loop_end, repeat_from_index)`` for one frame.

    A channel/DAC delay is a PURE ADDITIVE delay (NOT the cyclic ``%total`` view the
    preview shows): a channel delayed by ``d`` is its periodic signal shifted RIGHT by
    ``d`` with ZERO before fire -- so the FIRST pulses after fire are the real
    delayed sequence (the cyclic view would fake a wrapped-in tail at t=0, corrupting
    the experiment startup).  The loop PERIOD stays the original frame ``T`` (delaying
    one channel does not change another's period -- the physically-intuitive delay).

    A NEGATIVE delay re-translates the WHOLE frame: ``G = max(0, -min delay)`` is added
    to every channel so the earliest event is ``>= 0`` (a uniform base offset cannot
    reorder anything).

    For ``repeat_forever`` this emits a real-startup PREAMBLE then one steady-state
    frame, and returns ``repeat_from_index`` = the edge at the steady frame start so the
    engine loops ONLY the steady frame (period ``T``), playing the preamble once.  For a
    finite/one-shot run it emits the additively-shifted edges once (the frame extends to
    fit), ``loop_end`` = last edge, ``repeat_from_index`` = 0."""
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

    global_shift = max(0, -min(raw_delay.values())) if raw_delay else 0
    eff_delay = {channel: raw_delay[channel] + global_shift for channel in raw_delay}
    max_eff = max(eff_delay.values()) if eff_delay else 0

    # --- emit ON/OFF events (additive, period-preserving) ---
    # channel=None entries are period-boundary anchors (no-op edges) that keep the
    # frame structure (and bracket loop boundaries) explicit.
    events: list[tuple[int, str | None, int | None]] = []
    if repeat_forever and table_end > 0 and max_eff > 0:
        # PREAMBLE + steady frame: enough whole frames so the last one is steady-state.
        num_frames = (max_eff + table_end - 1) // table_end + 1   # ceil(max_eff/T) + 1
        loop_end = num_frames * table_end
        for channel, ivals in base_intervals.items():
            d = eff_delay[channel]
            for k in range(num_frames + 1):
                for a, b in ivals:
                    s, e = a + d + k * table_end, b + d + k * table_end
                    s, e = max(0, s), min(loop_end, e)
                    if s < e:
                        events.append((s, channel, 1)); events.append((e, channel, 0))
    else:
        # finite / one-shot / bracket / no-delay: the additively-shifted frame played
        # once; the frame extends only to fit the latest shifted edge (a bracket's
        # delays are validated to fit, so this stays the bracket frame end).
        for tick in starts:
            events.append((int(tick) + global_shift, None, None))
        for channel, ivals in base_intervals.items():
            d = eff_delay[channel]
            for a, b in ivals:
                events.append((a + d, channel, 1)); events.append((b + d, channel, 0))
        loop_end = max([table_end + global_shift] + [int(t) for t, _, _ in events])

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

    repeat_from_index = 0
    if repeat_forever and table_end > 0 and max_eff > 0:
        # the loop replays ONLY the steady frame [loop_end - T, loop_end); insert an
        # anchor edge there carrying the steady-state-start mask so repeat_forever
        # rewinds to it (NOT to edge 0 -- that would replay the startup preamble).
        ticks, masks, repeat_from_index = _insert_mask_edge_at_tick(ticks, masks, loop_end - table_end)
    return ticks, masks, hardware_channels, loop_end, repeat_from_index


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
