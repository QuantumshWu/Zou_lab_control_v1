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
    ScanParameterTable,
    affine_named_time_expr,
    channel_names,
    count_trigger_pulses,
    normalize_scan_parameter_name,
    positive_float,
    sequence_for_frame_count,
    scan_numeric_value,
    scan_parameter_names_from_expr,
)
from ..timing.verilog import VerilogBuild, VerilogFiles, generate_verilog, write_verilog_bundle


DEFAULT_RUNTIME_CLOCK_HZ = 50_000_000.0
DEFAULT_RUNTIME_BUS_NAMES = ("da_dipole", "da_bias_y", "da_bias_x", "da_bias_z")
DEFAULT_RUNTIME_BUS_WIDTH = 10
DEFAULT_RUNTIME_SCAN_PARAMETER_SLOTS = 5
DEFAULT_RUNTIME_SCAN_CHUNK_ROWS = 256
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

    def to_dict(self) -> dict[str, object]:
        return {
            "bus_index": int(self.bus_index),
            "bus_name": str(self.bus_name),
            "start_tick": int(self.start_tick),
            "stop_tick": int(self.stop_tick),
            "start_value": int(self.start_value),
            "stop_value": int(self.stop_value),
            "mode": str(self.mode),
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
    loop_end_x_coeff: int = 0
    loop_end_y_coeff: int = 0
    tick_x_coeffs: list[int] | None = None
    tick_y_coeffs: list[int] | None = None
    scan_points: list[tuple[int, int]] | None = None
    scan_axis_names: list[str] | None = None
    scan_point_durations: list[float] | None = None
    scan_coeff_frac_bits: int = 8
    bus_names: list[str] | None = None
    bus_segments: list[RuntimeBusSegment] | None = None
    scan_bus_values: list[int] | None = None

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
            "loop_end_x_coeff": int(self.loop_end_x_coeff),
            "loop_end_y_coeff": int(self.loop_end_y_coeff),
            "tick_x_coeffs": list(self.tick_x_coeffs or [0 for _ in self.ticks]),
            "tick_y_coeffs": list(self.tick_y_coeffs or [0 for _ in self.ticks]),
            "scan_points": [list(point) for point in (self.scan_points or [])],
            "scan_axis_names": list(self.scan_axis_names or []),
            "scan_point_durations": list(self.scan_point_durations or []),
            "scan_coeff_frac_bits": int(self.scan_coeff_frac_bits),
            "bus_names": list(self.bus_names or []),
            "bus_segments": [segment.to_dict() for segment in (self.bus_segments or [])],
            "scan_bus_values": [int(value) for value in (self.scan_bus_values or [])],
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
        tick_x_coeffs = [int(v) for v in payload.get("tick_x_coeffs", [])]
        tick_y_coeffs = [int(v) for v in payload.get("tick_y_coeffs", [])]
        if tick_x_coeffs and not any(tick_x_coeffs):
            tick_x_coeffs = []
        if tick_y_coeffs and not any(tick_y_coeffs):
            tick_y_coeffs = []
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
            loop_end_x_coeff=int(payload.get("loop_end_x_coeff", 0)),
            loop_end_y_coeff=int(payload.get("loop_end_y_coeff", 0)),
            tick_x_coeffs=tick_x_coeffs or None,
            tick_y_coeffs=tick_y_coeffs or None,
            scan_points=[(int(item[0]), int(item[1])) for item in payload.get("scan_points", [])] or None,
            scan_axis_names=[str(item) for item in payload.get("scan_axis_names", [])] or None,
            scan_point_durations=[float(v) for v in payload.get("scan_point_durations", [])] or None,
            scan_coeff_frac_bits=int(payload.get("scan_coeff_frac_bits", 8)),
            bus_names=[str(item) for item in payload.get("bus_names", [])] or None,
            bus_segments=[RuntimeBusSegment.from_dict(item) for item in payload.get("bus_segments", [])] or None,
            scan_bus_values=[int(item) for item in payload.get("scan_bus_values", [])] or None,
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
    x_ns: float | None = None,
    y_ns: float | None = None,
    variables: Mapping[str, float] | None = None,
    repeat_forever: bool = True,
) -> RuntimeSequenceProgram:
    """Compile GUI period-card state into an unexpanded FPGA loop program.

    ``PulseTableState`` carries the frontend repeat-bracket semantics.  The
    runtime FPGA should receive one copy of the period table plus loop metadata,
    not a fully expanded edge table.  A bracket becomes one finite inner loop;
    the whole table may still be repeated forever by the FPGA.
    """

    channels = list(channel_names(state.channels if channels is None else channels, "channels"))
    unknown_channels = [channel for channel in state.channels if channel not in channels]
    if unknown_channels:
        raise ValueError(f"pulse table channels are not in hardware channels: {unknown_channels}.")
    clock_hz = positive_float(clock_hz, "clock_hz")
    clock_step_ns = 1e9 / clock_hz
    x_value = state.x_ns if x_ns is None else x_ns
    y_value = state.y_ns if y_ns is None else y_ns
    expression_vars = state.scan_variable_values(x_ns=x_value, y_ns=y_value)
    expression_vars.update({str(k): float(v) for k, v in dict(variables or {}).items()})
    state.validate(x_ns=x_value, y_ns=y_value, time_step_ns=clock_step_ns, variables=expression_vars)

    sequence = state.to_sequence(x_ns=x_value, y_ns=y_value, time_step_ns=clock_step_ns, variables=expression_vars, expand_repeat=False)
    period_starts = _pulse_table_period_starts_ticks(state, x_ns=x_value, y_ns=y_value, time_step_ns=clock_step_ns, variables=expression_vars)
    has_delays = _pulse_table_has_delays(state, x_ns=x_value, y_ns=y_value, time_step_ns=clock_step_ns, variables=expression_vars)
    if has_delays:
        _validate_pulse_table_delays_for_hardware_loop(
            state,
            period_starts=period_starts,
            x_ns=x_value,
            y_ns=y_value,
            time_step_ns=clock_step_ns,
            variables=expression_vars,
        )
    bus_names, bus_segments = _pulse_table_bus_segments(
        state,
        x_ns=x_value,
        y_ns=y_value,
        time_step_ns=clock_step_ns,
        variables=expression_vars,
    )
    ticks, masks, channels = _pulse_table_edge_table(
        state,
        channels=channels,
        x_ns=x_value,
        y_ns=y_value,
        time_step_ns=clock_step_ns,
        variables=expression_vars,
        fold_analog_buses=not bool(bus_segments),
    )
    repeat_count = int(state.repeat_count)
    if state.repeat_start is None or state.repeat_end is None:
        loop_start_index = 0
        loop_end_tick = int(ticks[-1]) if has_delays else int(period_starts[-1])
        loop_count = 1
    else:
        loop_start_tick = int(period_starts[int(state.repeat_start)])
        loop_end_tick = int(period_starts[int(state.repeat_end) + 1])
        ticks, masks, loop_start_index = _insert_mask_edge_at_tick(ticks, masks, loop_start_tick)
        loop_count = repeat_count

    effective_duration_ticks = _pulse_table_effective_duration_ticks(state, x_ns=x_value, y_ns=y_value, time_step_ns=clock_step_ns, variables=expression_vars)
    if has_delays and state.repeat_start is None and state.repeat_end is None:
        effective_duration_ticks = int(ticks[-1])
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
        bus_names=bus_names or None,
        bus_segments=bus_segments or None,
    )


def compile_pulse_table_scan_runtime_program(
    state: PulseTableState,
    *,
    scan_points: Sequence[Sequence[float] | float] | None = None,
    scan_parameter_table: ScanParameterTable | None = None,
    channels: Sequence[str] | None = None,
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
    repeat_forever: bool = False,
    coeff_frac_bits: int = 8,
) -> RuntimeSequenceProgram:
    """Compile a ``PulseTableState`` into a parameterized FPGA scan program.

    Named duration and delay parameters are mapped onto at most two hardware
    timing axes.  Named analog-bus values are packed into one per-scan-row bus
    value word instead of expanding the edge table.
    """

    channels = list(channel_names(state.channels if channels is None else channels, "channels"))
    unknown_channels = [channel for channel in state.channels if channel not in channels]
    if unknown_channels:
        raise ValueError(f"pulse table channels are not in hardware channels: {unknown_channels}.")
    clock_hz = positive_float(clock_hz, "clock_hz")
    clock_step_ns = 1e9 / clock_hz
    scan_table, scan_axis_names = _pulse_table_scan_table_for_compile(
        state,
        scan_points=scan_points,
        scan_parameter_table=scan_parameter_table,
        time_step_ns=clock_step_ns,
    )
    if not scan_table.rows:
        raise ValueError("hardware scan requires at least one scan point.")
    if len(scan_axis_names) > 2:
        raise ValueError(
            f"the current FPGA scan table has two hardware axes, but this pulse uses {len(scan_axis_names)} active scan parameters: {scan_axis_names}. "
            "Use at most two time-scan parameters for this bitstream, or split the scan."
        )
    bus_names, scan_bus_values = _pulse_table_scan_bus_values(state, scan_table.rows)
    if _pulse_table_has_analog_activity(state) and scan_bus_values is None:
        raise ValueError(
            "hardware scan array cannot currently combine with analog bus segment output. "
            "Run one prepared analog-bus pulse per scan point, or use ordinary TTL channels for the scan template."
        )
    points_ticks = [
        _scan_table_row_to_axis_ticks(row, scan_axis_names, time_step_ns=clock_step_ns, point_index=index)
        for index, row in enumerate(scan_table.rows)
    ]
    point_variables = [
        _scan_table_row_variables(state, row)
        for row in scan_table.rows
    ]
    for variables in point_variables:
        state.validate(
            x_ns=variables.get("x", state.x_ns),
            y_ns=variables.get("y", state.y_ns),
            time_step_ns=clock_step_ns,
            variables=variables,
        )

    rows = _pulse_table_affine_rows(
        state,
        channels=channels,
        scan_points=points_ticks,
        scan_axis_names=scan_axis_names,
        time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits,
    )
    ticks = [row[0] for row in rows]
    masks = [row[1] for row in rows]
    tick_x_coeffs = [row[2] for row in rows]
    tick_y_coeffs = [row[3] for row in rows]
    loop_start_index, loop_end_tick, loop_end_x_coeff, loop_end_y_coeff, loop_count = _pulse_table_affine_loop_metadata(
        state,
        rows=rows,
        scan_axis_names=scan_axis_names,
        time_step_ns=clock_step_ns,
        coeff_frac_bits=coeff_frac_bits,
    )
    point_durations = [float(_apply_affine_ticks(ticks[-1], tick_x_coeffs[-1], tick_y_coeffs[-1], x_tick, y_tick, coeff_frac_bits)) / clock_hz for x_tick, y_tick in points_ticks]
    first_variables = point_variables[0]
    first_x = first_variables.get("x", state.x_ns)
    first_y = first_variables.get("y", state.y_ns)
    sequence = state.to_sequence(x_ns=first_x, y_ns=first_y, time_step_ns=clock_step_ns, variables=first_variables, expand_repeat=False)
    trigger_count = _pulse_table_trigger_count(state, trigger_channels=trigger_channels) * len(points_ticks)
    payload = {
        "table": state.to_dict(),
        "clock_hz": clock_hz,
        "channels": channels,
        "ticks": ticks,
        "masks": masks,
        "tick_x_coeffs": tick_x_coeffs,
        "tick_y_coeffs": tick_y_coeffs,
        "scan_points": points_ticks,
        "scan_axis_names": scan_axis_names,
        "repeat_forever": bool(repeat_forever),
        "loop_start_index": loop_start_index,
        "loop_end_tick": loop_end_tick,
        "loop_end_x_coeff": loop_end_x_coeff,
        "loop_end_y_coeff": loop_end_y_coeff,
        "loop_count": loop_count,
        "scan_coeff_frac_bits": coeff_frac_bits,
        "bus_names": bus_names,
        "scan_bus_values": scan_bus_values or [],
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
        source_table=state.with_scan_table_path(scan_table.path).to_dict(),
        repeat_forever=bool(repeat_forever),
        loop_start_index=loop_start_index,
        loop_end_tick=loop_end_tick,
        loop_count=loop_count,
        loop_end_x_coeff=loop_end_x_coeff,
        loop_end_y_coeff=loop_end_y_coeff,
        tick_x_coeffs=tick_x_coeffs,
        tick_y_coeffs=tick_y_coeffs,
        scan_points=points_ticks,
        scan_axis_names=list(scan_axis_names),
        scan_point_durations=point_durations,
        scan_coeff_frac_bits=coeff_frac_bits,
        bus_names=bus_names or None,
        scan_bus_values=scan_bus_values,
    )


def compile_runtime_program_for_payload(
    payload: PulseSequence | PulseTableState | RuntimeSequenceProgram,
    *,
    channels: Sequence[str],
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> RuntimeSequenceProgram:
    """Compile either finite sequence data or GUI pulse-table data."""

    if isinstance(payload, RuntimeSequenceProgram):
        expected_channels = list(channel_names(channels, "channels"))
        if list(payload.channels) != expected_channels:
            raise ValueError("runtime program channels do not match this sequencer service.")
        if not math.isclose(float(payload.clock_hz), float(clock_hz), rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("runtime program clock_hz does not match this sequencer service.")
        return payload
    if isinstance(payload, PulseTableState):
        if getattr(payload, "scan_table_path", ""):
            return compile_pulse_table_scan_runtime_program(
                payload,
                channels=channels,
                clock_hz=clock_hz,
                trigger_channels=trigger_channels,
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
        variables = payload.scan_variable_values()
        sequence = payload.to_sequence(
            x_ns=payload.x_ns,
            y_ns=payload.y_ns,
            time_step_ns=payload.time_step_ns,
            variables=variables,
            expand_repeat=False,
        )
        base_period_s = sum(
            period.duration_steps(x_ns=payload.x_ns, y_ns=payload.y_ns, time_step_ns=payload.time_step_ns, variables=variables)
            for period in payload.periods
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
            self._conn = rpyc.connect(self.host, self.port, config={"allow_pickle": True, "sync_request_timeout": None})
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

    The controller owns no hardware; it delegates to the supplied local or
    remote ``SequencerDevice`` and keeps named scan variables with the pulse
    payload.
    """

    def __init__(self, sequencer: SequencerDevice, pulse: PulseSequence | PulseTableState):
        self.sequencer = sequencer
        self.pulse = pulse
        self.variables = dict(pulse.scan_variable_values()) if isinstance(pulse, PulseTableState) else {}
        self.scan_table_path = str(getattr(pulse, "scan_table_path", "") or "")
        self.last_program: RuntimeSequenceProgram | None = None
        self.last_programs: list[RuntimeSequenceProgram] = []

    @staticmethod
    def _legacy_xy_error() -> ValueError:
        return ValueError("legacy x/y scan variables are no longer supported; use set_variable(name, value).")

    @property
    def x(self) -> float:
        raise self._legacy_xy_error()

    @x.setter
    def x(self, value: float) -> None:
        raise self._legacy_xy_error()

    @property
    def y(self) -> float:
        raise self._legacy_xy_error()

    @y.setter
    def y(self, value: float) -> None:
        raise self._legacy_xy_error()

    @property
    def x_ns(self) -> float:
        raise self._legacy_xy_error()

    @x_ns.setter
    def x_ns(self, value: float) -> None:
        raise self._legacy_xy_error()

    @property
    def y_ns(self) -> float:
        raise self._legacy_xy_error()

    @y_ns.setter
    def y_ns(self, value: float) -> None:
        raise self._legacy_xy_error()

    def set_variable(self, name: str, value: float) -> "PulseController":
        self.variables[normalize_scan_parameter_name(name)] = float(value)
        return self

    def set_variables(self, variables: Mapping[str, float] | None = None, **kwargs: float) -> "PulseController":
        updates = dict(variables or {})
        updates.update(kwargs)
        for name, value in updates.items():
            self.set_variable(str(name), float(value))
        return self

    def set_scan_table_path(self, path: str | Path | None) -> "PulseController":
        self.scan_table_path = "" if path is None else str(path)
        return self

    def set_scan_points(self, scan_points: Sequence[Sequence[float] | float] | None) -> "PulseController":
        raise ValueError("legacy x/y scan_points are no longer supported; use set_scan_table_path(path).")

    def payload(
        self,
        *,
        variables: Mapping[str, float] | None = None,
        scan_table_path: str | Path | None = None,
        scan_points: Sequence[Sequence[float] | float] | None = None,
        repeat_forever: bool | None = None,
    ) -> PulseSequence | PulseTableState:
        if scan_points is not None:
            raise ValueError("legacy x/y scan_points are no longer supported; use scan_table_path.")
        if isinstance(self.pulse, PulseTableState):
            merged = dict(self.variables)
            merged.update({normalize_scan_parameter_name(k): float(v) for k, v in dict(variables or {}).items()})
            payload = self.pulse.with_scan_variables(**merged)
            table_path = self.scan_table_path if scan_table_path is None else str(scan_table_path)
            payload = payload.with_scan_table_path(table_path)
            if payload.scan_points:
                data = payload.to_dict()
                data["scan_points"] = []
                payload = PulseTableState.from_dict(data)
            if repeat_forever is not None:
                data = payload.to_dict()
                data["repeat_forever"] = bool(repeat_forever)
                payload = PulseTableState.from_dict(data)
            return payload
        if variables:
            raise ValueError("named scan variables require a PulseTableState payload.")
        if scan_table_path:
            raise ValueError("scan_table_path requires a PulseTableState payload.")
        if repeat_forever is not None:
            data = self.pulse.to_dict()
            data["repeat_forever"] = bool(repeat_forever)
            return PulseSequence.from_dict(data)
        return self.pulse

    def frame_sequence(
        self,
        frames: int,
        *,
        variables: Mapping[str, float] | None = None,
        trigger_channels: Sequence[str] | None = None,
    ) -> PulseSequence:
        """Return a finite ``PulseSequence`` with exactly ``frames`` triggers."""

        frames = positive_int(frames, "frames")
        trigger_channels = tuple(channel_names(
            getattr(self.sequencer, "trigger_channels", DEFAULT_CAMERA_TRIGGER_CHANNELS) if trigger_channels is None else trigger_channels,
                "trigger_channels",
        ))
        payload = self.payload(variables=variables, repeat_forever=False)
        return finite_frame_sequence(payload, frames, trigger_channels=trigger_channels)

    def prepare(
        self,
        *,
        variables: Mapping[str, float] | None = None,
        scan_table_path: str | Path | None = None,
        scan_points: Sequence[Sequence[float] | float] | None = None,
        repeat_forever: bool | None = None,
    ) -> RuntimeSequenceProgram:
        payload = self.payload(
            variables=variables,
            scan_table_path=scan_table_path,
            scan_points=scan_points,
            repeat_forever=repeat_forever,
        )
        program_payload = self._payload_for_single_prepare(payload)
        self.last_program = self.sequencer.prepare(program_payload)
        self.last_programs = [self.last_program]
        return self.last_program

    def on_pulse(
        self,
        *,
        wait: bool = False,
        timeout: float | None = None,
        variables: Mapping[str, float] | None = None,
        scan_table_path: str | Path | None = None,
        scan_points: Sequence[Sequence[float] | float] | None = None,
        repeat_forever: bool | None = None,
        scan_chunk_size: int | None = None,
    ) -> RuntimeSequenceProgram:
        payload = self.payload(
            variables=variables,
            scan_table_path=scan_table_path,
            scan_points=scan_points,
            repeat_forever=repeat_forever,
        )
        if wait and timeout is None and bool(getattr(payload, "repeat_forever", False)):
            raise RuntimeError(
                "pulse.on_pulse(wait=True) cannot wait for a repeat_forever pulse without a timeout. "
                "Use pulse.on_pulse(wait=False, repeat_forever=True) for continuous scope output, "
                "or pulse.on_pulse(wait=True, repeat_forever=False) for a finite shot."
            )
        if isinstance(payload, PulseTableState) and payload.scan_table_path and payload.active_scan_parameters():
            table = self._scan_table_for_runtime(payload)
            chunk_size = self._scan_chunk_size(scan_chunk_size)
            if len(table.rows) > chunk_size:
                if bool(getattr(payload, "repeat_forever", False)):
                    raise RuntimeError("automatic scan chunking requires repeat_forever=False.")
                if not wait:
                    raise RuntimeError("automatic scan chunking requires wait=True so each chunk can finish before the next prepare.")
                return self._run_scan_chunks(payload, table, chunk_size=chunk_size, timeout=timeout)
        program_payload = self._payload_for_single_prepare(payload, scan_chunk_size=scan_chunk_size)
        self.last_program = self.sequencer.prepare(program_payload)
        program = self.last_program
        self.last_programs = [program]
        self.sequencer.fire()
        if wait:
            if not self.sequencer.wait_done(timeout=timeout):
                raise TimeoutError(f"sequencer did not report done for pulse {program.sequence_name!r}.")
        return program

    def _scan_chunk_size(self, scan_chunk_size: int | None = None) -> int:
        if scan_chunk_size is not None:
            return positive_int(scan_chunk_size, "scan_chunk_size")
        value = getattr(self.sequencer, "max_scan_points", None)
        if value is not None:
            return positive_int(value, "sequencer.max_scan_points")
        return DEFAULT_RUNTIME_SCAN_CHUNK_ROWS

    def _scan_table_for_runtime(self, payload: PulseTableState) -> ScanParameterTable:
        clock_hz = positive_float(getattr(self.sequencer, "clock_hz", DEFAULT_RUNTIME_CLOCK_HZ), "clock_hz")
        return payload.scan_table(time_step_ns=1e9 / clock_hz)

    def _compile_scan_payload(
        self,
        payload: PulseTableState,
        *,
        scan_parameter_table: ScanParameterTable | None = None,
    ) -> RuntimeSequenceProgram:
        return compile_pulse_table_scan_runtime_program(
            payload,
            channels=list(getattr(self.sequencer, "channels", payload.channels)),
            clock_hz=float(getattr(self.sequencer, "clock_hz", DEFAULT_RUNTIME_CLOCK_HZ)),
            trigger_channels=list(getattr(self.sequencer, "trigger_channels", DEFAULT_CAMERA_TRIGGER_CHANNELS)),
            repeat_forever=payload.repeat_forever,
            scan_parameter_table=scan_parameter_table,
        )

    def _payload_for_single_prepare(
        self,
        payload: PulseSequence | PulseTableState,
        *,
        scan_chunk_size: int | None = None,
    ) -> PulseSequence | PulseTableState | RuntimeSequenceProgram:
        if isinstance(payload, PulseTableState) and payload.scan_table_path and payload.active_scan_parameters():
            table = self._scan_table_for_runtime(payload)
            chunk_size = self._scan_chunk_size(scan_chunk_size)
            if len(table.rows) > chunk_size:
                raise RuntimeError(
                    f"scan table has {len(table.rows)} rows, but one FPGA program accepts {chunk_size}. "
                    "Use pulse.on_pulse(wait=True, repeat_forever=False) to run it with automatic scan chunking."
                )
            return self._compile_scan_payload(payload, scan_parameter_table=table)
        return payload

    def _run_scan_chunks(
        self,
        payload: PulseTableState,
        table: ScanParameterTable,
        *,
        chunk_size: int,
        timeout: float | None,
    ) -> RuntimeSequenceProgram:
        programs: list[RuntimeSequenceProgram] = []
        rows = list(table.rows)
        for start in range(0, len(rows), chunk_size):
            stop = min(start + chunk_size, len(rows))
            chunk_table = ScanParameterTable(
                names=tuple(table.names),
                rows=tuple(dict(row) for row in rows[start:stop]),
                units=dict(table.units),
                path=f"{table.path or payload.scan_table_path} rows {start + 1}-{stop}",
            )
            program = self._compile_scan_payload(payload, scan_parameter_table=chunk_table)
            prepared = self.sequencer.prepare(program)
            self.last_program = prepared
            programs.append(prepared)
            self.sequencer.fire()
            if not self.sequencer.wait_done(timeout=timeout):
                self.last_programs = programs
                raise TimeoutError(f"sequencer did not report done for scan chunk {start // chunk_size + 1}.")
        self.last_programs = programs
        if not programs:
            raise RuntimeError("scan table did not contain any rows.")
        return programs[-1]

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
            "scan_variables": dict(self.variables),
            "active_scan_parameters": self.pulse.active_scan_parameters() if isinstance(self.pulse, PulseTableState) else [],
            "scan_table_path": self.scan_table_path,
            "sequencer_type": type(self.sequencer).__name__,
            "sequencer_channels": list(getattr(self.sequencer, "channels", [])),
            "clock_hz": float(getattr(self.sequencer, "clock_hz", 0.0)),
            "trigger_channels": list(getattr(self.sequencer, "trigger_channels", [])),
            "last_program": last,
            "last_program_count": len(self.last_programs),
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


def timing_payload_to_dict(payload: PulseSequence | PulseTableState | RuntimeSequenceProgram) -> dict[str, object]:
    """Return the JSON-safe timing payload for a sequence or pulse table."""

    if isinstance(payload, (PulseSequence, PulseTableState, RuntimeSequenceProgram)):
        return payload.to_dict()
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError("timing payload must be a PulseSequence, PulseTableState, or mapping.")


def timing_from_payload(payload) -> PulseSequence | PulseTableState | RuntimeSequenceProgram:
    """Accept local timing objects or their JSON/RPyC-safe dict payload."""

    if isinstance(payload, RuntimeSequenceProgram):
        return payload
    if isinstance(payload, PulseSequence):
        return payload
    if isinstance(payload, PulseTableState):
        return payload
    if isinstance(payload, (str, bytes)):
        return timing_from_payload(json.loads(payload))
    if isinstance(payload, Mapping):
        data = dict(payload)
        schema = data.get("schema", "Zou_lab_control.neutral_atom.PulseSequence")
        if schema == "Zou_lab_control.neutral_atom.RuntimeSequenceProgram":
            return RuntimeSequenceProgram.from_dict(data)
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


def _pulse_table_scan_table_for_compile(
    state: PulseTableState,
    *,
    scan_points: Sequence[Sequence[float] | float] | None,
    scan_parameter_table: ScanParameterTable | None = None,
    time_step_ns: float,
):
    active = state.active_scan_parameters()
    timing_active = _pulse_table_timing_scan_parameters(state)
    if scan_points is not None or state.scan_points:
        raise ValueError("legacy x/y scan_points are no longer supported; link a named scan table file.")
    if scan_parameter_table is None and not state.scan_table_path:
        raise ValueError("hardware scan requires a named scan table file.")
    if not active:
        raise ValueError("hardware scan requires at least one active named scan parameter.")
    if len(active) > DEFAULT_RUNTIME_SCAN_PARAMETER_SLOTS:
        raise ValueError(
            f"the current Artix-7 35T scan profile supports at most {DEFAULT_RUNTIME_SCAN_PARAMETER_SLOTS} active named scan parameters, "
            f"but this pulse uses {len(active)}: {active}. Split the scan or bind fewer fields for one prepared program."
        )
    table = scan_parameter_table if scan_parameter_table is not None else state.scan_table(time_step_ns=time_step_ns)
    required = [name for name in active if name not in table.names]
    if required:
        raise ValueError(
            f"scan table is missing active parameter column(s): {required}. "
            f"Available columns are {list(table.names)}. Make the scan file '# vars:' header match the GUI Params row."
        )
    axis_names = [name for name in table.names if name in timing_active]
    return table, axis_names


def _pulse_table_timing_scan_parameters(state: PulseTableState) -> list[str]:
    active: set[str] = set()
    for period in state.periods:
        active.update(scan_parameter_names_from_expr(period.duration))
    for value in state.delays.values():
        active.update(scan_parameter_names_from_expr(value))
    for key, variable in state.scan_bindings.items():
        key_text = str(key)
        if key_text.startswith("period:") or key_text.startswith("delay:"):
            active.add(normalize_scan_parameter_name(variable))
    return sorted(active)


def _scan_table_row_variables(state: PulseTableState, row: Mapping[str, float]) -> dict[str, float]:
    variables = state.scan_variable_values()
    variables.update({str(k): float(v) for k, v in row.items()})
    return variables


def _scan_table_row_to_axis_ticks(
    row: Mapping[str, float],
    axis_names: Sequence[str],
    *,
    time_step_ns: float,
    point_index: int,
) -> tuple[int, int]:
    values = []
    for name in axis_names[:2]:
        values.append(_time_ns_to_ticks(float(row[name]), time_step_ns, f"scan row {point_index} {name}", allow_negative=True))
    while len(values) < 2:
        values.append(0)
    return int(values[0]), int(values[1])


def _pulse_table_period_starts_ticks(
    state: PulseTableState,
    *,
    x_ns: float,
    y_ns: float = 0.0,
    time_step_ns: float,
    variables: Mapping[str, float] | None = None,
) -> list[int]:
    starts = [0]
    expression_vars = state.scan_variable_values(x_ns=x_ns, y_ns=y_ns)
    expression_vars.update(dict(variables or {}))
    for period in state.periods:
        starts.append(starts[-1] + period.duration_steps(x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=expression_vars))
    return starts


def _pulse_table_affine_period_starts(
    state: PulseTableState,
    *,
    scan_axis_names: Sequence[str] = ("x", "y"),
    time_step_ns: float,
    coeff_frac_bits: int,
) -> list[tuple[int, int, int]]:
    starts = [(0, 0, 0)]
    for period in state.periods:
        starts.append(_affine_add(starts[-1], _pulse_table_axis_affine_time_expr(period.duration, unit=period.unit, scan_axis_names=scan_axis_names, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)))
    return starts


def _pulse_table_affine_rows(
    state: PulseTableState,
    *,
    channels: Sequence[str],
    scan_points: Sequence[tuple[int, int]],
    scan_axis_names: Sequence[str] = ("x", "y"),
    time_step_ns: float,
    coeff_frac_bits: int,
) -> list[tuple[int, int, int, int]]:
    hardware_channels = list(channel_names(channels, "channels"))
    state_index = {channel: index for index, channel in enumerate(state.channels)}
    starts = _pulse_table_affine_period_starts(state, scan_axis_names=scan_axis_names, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    events: list[tuple[tuple[int, int, int], str | None, int | None]] = []
    final_expr = starts[-1]

    for channel_index, channel in enumerate(state.channels):
        delay = _pulse_table_axis_affine_time_expr(state.delays.get(channel, 0.0), unit=state.delay_units.get(channel, "ns"), scan_axis_names=scan_axis_names, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
        active_start: tuple[int, int, int] | None = None
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

    events.append((final_expr, None, None))
    if state.repeat_start is not None and state.repeat_end is not None and state.repeat_count > 1:
        events.append((starts[int(state.repeat_start)], None, None))

    grouped = _stable_affine_groups(events, scan_points=scan_points, coeff_frac_bits=coeff_frac_bits)
    current_mask = 0
    rows: list[tuple[int, int, int, int]] = []
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
        rows.append((expr[0], current_mask, expr[1], expr[2]))
    if not rows:
        rows.append((0, 0, 0, 0))
    if int(rows[-1][1]) != 0:
        raise ValueError("hardware scan template final row must return every channel to 0.")
    return rows


def _pulse_table_affine_loop_metadata(
    state: PulseTableState,
    *,
    rows: Sequence[tuple[int, int, int, int]],
    scan_axis_names: Sequence[str] = ("x", "y"),
    time_step_ns: float,
    coeff_frac_bits: int,
) -> tuple[int, int, int, int, int]:
    if state.repeat_start is None or state.repeat_end is None or state.repeat_count <= 1:
        return 0, int(rows[-1][0]), int(rows[-1][2]), int(rows[-1][3]), 1
    starts = _pulse_table_affine_period_starts(state, scan_axis_names=scan_axis_names, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    loop_start = starts[int(state.repeat_start)]
    loop_end = starts[int(state.repeat_end) + 1]
    loop_start_index = _affine_row_index(rows, loop_start)
    return loop_start_index, int(loop_end[0]), int(loop_end[1]), int(loop_end[2]), int(state.repeat_count)


def _stable_affine_groups(
    events: Sequence[tuple[tuple[int, int, int], str | None, int | None]],
    *,
    scan_points: Sequence[tuple[int, int]],
    coeff_frac_bits: int,
) -> list[tuple[tuple[int, int, int], list[tuple[str | None, int | None]]]]:
    if not scan_points:
        raise ValueError("hardware scan requires at least one scan point.")
    x0, y0 = scan_points[0]
    by_ref: dict[int, list[tuple[tuple[int, int, int], str | None, int | None]]] = {}
    for expr, channel, value in events:
        tick0 = _apply_affine_ticks(expr[0], expr[1], expr[2], x0, y0, coeff_frac_bits)
        if tick0 < 0:
            raise ValueError("hardware scan produced a negative edge tick at the first scan point.")
        by_ref.setdefault(tick0, []).append((expr, channel, value))
    grouped: list[tuple[tuple[int, int, int], list[tuple[str | None, int | None]]]] = []
    for _tick0, items in sorted(by_ref.items(), key=lambda item: item[0]):
        expr = items[0][0]
        if any(item[0] != expr for item in items):
            raise ValueError("hardware scan has events that coincide only for the first scan point; split the scan or simplify timing.")
        grouped.append((expr, [(channel, value) for _expr, channel, value in items]))
    for point_index, (x_tick, y_tick) in enumerate(scan_points):
        last_tick = -1
        for expr, _items in grouped:
            tick = _apply_affine_ticks(expr[0], expr[1], expr[2], x_tick, y_tick, coeff_frac_bits)
            if tick < 0:
                raise ValueError(f"hardware scan produced a negative edge tick at scan row {point_index}.")
            if tick <= last_tick:
                raise ValueError(
                    "hardware scan changes edge order for at least one timing-axis row; "
                    "split the scan into multiple templates or use simpler delay/duration expressions."
                )
            last_tick = tick
    return grouped


def _affine_row_index(rows: Sequence[tuple[int, int, int, int]], expr: tuple[int, int, int]) -> int:
    for index, row in enumerate(rows):
        if (int(row[0]), int(row[2]), int(row[3])) == (int(expr[0]), int(expr[1]), int(expr[2])):
            return index
    raise ValueError("repeat bracket start does not match a hardware scan edge row.")


def _pulse_table_axis_affine_time_expr(
    value: float | str,
    *,
    unit: str,
    scan_axis_names: Sequence[str],
    time_step_ns: float,
    coeff_frac_bits: int,
) -> tuple[int, int, int]:
    axis_names = [str(name) for name in scan_axis_names]
    base, coeffs = affine_named_time_expr(
        value,
        variable_names=axis_names,
        unit=unit,
        time_step_ns=time_step_ns,
        coeff_frac_bits=coeff_frac_bits,
    )
    x_coeff = int(coeffs.get(axis_names[0], 0)) if len(axis_names) >= 1 else 0
    y_coeff = int(coeffs.get(axis_names[1], 0)) if len(axis_names) >= 2 else 0
    return int(base), x_coeff, y_coeff


def _affine_add(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(left[0]) + int(right[0]), int(left[1]) + int(right[1]), int(left[2]) + int(right[2])


def _affine_max_reference(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    scan_points: Sequence[tuple[int, int]],
    coeff_frac_bits: int,
) -> tuple[int, int, int]:
    if not scan_points:
        return right if right[0] > left[0] else left
    x_tick, y_tick = scan_points[0]
    left_tick = _apply_affine_ticks(left[0], left[1], left[2], x_tick, y_tick, coeff_frac_bits)
    right_tick = _apply_affine_ticks(right[0], right[1], right[2], x_tick, y_tick, coeff_frac_bits)
    return right if right_tick > left_tick else left


def _apply_affine_ticks(base: int, x_coeff: int, y_coeff: int, x_tick: int, y_tick: int, coeff_frac_bits: int) -> int:
    return int(base) + ((int(x_coeff) * int(x_tick) + int(y_coeff) * int(y_tick)) >> int(coeff_frac_bits))


def _time_ns_to_ticks(value_ns: float, time_step_ns: float, name: str, *, allow_negative: bool = False) -> int:
    raw = float(value_ns) / float(time_step_ns)
    ticks = int(round(raw))
    if not math.isclose(raw, ticks, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{name}={value_ns:g} ns is not on the {time_step_ns:g} ns clock grid.")
    if ticks < 0 and not allow_negative:
        raise ValueError(f"{name} must be >= 0 ns.")
    return ticks


def _pulse_table_effective_duration_ticks(
    state: PulseTableState,
    *,
    x_ns: float,
    y_ns: float = 0.0,
    time_step_ns: float,
    variables: Mapping[str, float] | None = None,
) -> int:
    starts = _pulse_table_period_starts_ticks(state, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=variables)
    if state.repeat_start is None or state.repeat_end is None or state.repeat_count <= 1:
        return starts[-1]
    loop_ticks = starts[int(state.repeat_end) + 1] - starts[int(state.repeat_start)]
    return starts[-1] + (int(state.repeat_count) - 1) * loop_ticks


def _pulse_table_edge_table(
    state: PulseTableState,
    *,
    channels: Sequence[str],
    x_ns: float,
    y_ns: float = 0.0,
    time_step_ns: float,
    variables: Mapping[str, float] | None = None,
    fold_analog_buses: bool = True,
) -> tuple[list[int], list[int], list[str]]:
    hardware_channels = list(channel_names(channels, "channels"))
    state_index = {channel: index for index, channel in enumerate(state.channels)}
    starts = _pulse_table_period_starts_ticks(state, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=variables)
    table_end = int(starts[-1])
    bus_groups = state.bus_channels()
    bus_members = {channel for members in bus_groups.values() for channel in members}
    events: list[tuple[int, str | None, int | None]] = []
    for tick in starts:
        events.append((int(tick), None, None))

    for channel_index, channel in enumerate(state.channels):
        if channel in bus_members:
            continue
        delay_steps = state.delay_steps(channel, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=variables)
        active_start: int | None = None
        for period_index, period in enumerate(state.periods):
            value = int(period.states[channel_index])
            if value and active_start is None:
                active_start = int(starts[period_index])
            elif not value and active_start is not None:
                events.append((active_start + delay_steps, channel, 1))
                events.append((int(starts[period_index]) + delay_steps, channel, 0))
                active_start = None
        if active_start is not None:
            events.append((active_start + delay_steps, channel, 1))
            events.append((table_end + delay_steps, channel, 0))

    if fold_analog_buses:
        for bus_name, members in bus_groups.items():
            delay_steps = _pulse_table_bus_delay_steps(state, members, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=variables)
            plan = state.analog_bus_plan(bus_name)
            bus_ticks = _pulse_table_analog_bus_ticks(plan, starts)
            for tick in bus_ticks:
                if tick < 0 or tick > table_end:
                    raise ValueError(f"analog bus {bus_name!r} produced edge tick {tick} outside the uploaded table.")
                value = _pulse_table_analog_bus_value_at_tick(plan, starts, tick, variables=variables)
                for bit, channel in enumerate(members):
                    events.append((int(tick) + delay_steps, channel, 1 if (int(value) >> bit) & 1 else 0))
            for bit, channel in enumerate(members):
                events.append((table_end + delay_steps, channel, 0))

    grouped: dict[int, list[tuple[str | None, int | None]]] = {}
    for tick, channel, value in events:
        tick = int(tick)
        if tick < 0 or tick > table_end:
            raise ValueError(f"pulse table edge tick {tick} is outside the uploaded table [0, {table_end}].")
        grouped.setdefault(tick, []).append((channel, value))

    ticks: list[int] = []
    masks: list[int] = []
    current_mask = 0
    channel_bits = {channel: index for index, channel in enumerate(hardware_channels)}
    for tick in sorted(grouped):
        for channel, value in grouped[tick]:
            if channel is None or value is None:
                continue
            bit = channel_bits.get(channel)
            if bit is None:
                continue
            if int(value):
                current_mask |= 1 << bit
            else:
                current_mask &= ~(1 << bit)
        ticks.append(int(tick))
        masks.append(int(current_mask))
    ticks, masks = _dedupe_same_tick_edges(ticks, masks)
    ticks, masks = _ensure_final_off_edge(ticks, masks, table_end)
    return ticks, masks, hardware_channels


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
    x_ns: float,
    y_ns: float,
    time_step_ns: float,
    variables: Mapping[str, float] | None = None,
) -> int:
    delays = {
        state.delay_steps(channel, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=variables)
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


def _pulse_table_analog_bus_value_at_tick(
    plan: Sequence[Mapping[str, object]],
    starts: Sequence[int],
    tick: int,
    *,
    variables: Mapping[str, float] | None = None,
) -> int:
    anchors: list[tuple[int, int, str, int]] = []
    for index, entry in enumerate(plan):
        mode = str(entry.get("mode", "hold")).lower()
        value = entry.get("value")
        if mode in {"edge", "ramp"} and value is not None:
            anchors.append((index, int(starts[index]), mode, scan_numeric_value(value, variables=variables, name="analog bus value")))
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


def _pulse_table_scan_bus_values(
    state: PulseTableState,
    scan_rows: Sequence[Mapping[str, float]],
) -> tuple[list[str], list[int] | None]:
    """Return per-scan-row packed static bus values for scan-mode upload.

    The Artix-7 35T path keeps scan DA output intentionally compact: one
    packed bus word is uploaded for each scan row and held for the full row.
    Period-internal bus ramps or multiple bus changes still use the non-scan
    bus-segment path and are rejected here.
    """

    if not _pulse_table_has_analog_activity(state):
        return [], None
    bus_groups = state.bus_channels()
    if not bus_groups:
        return [], None
    bus_names = _pulse_table_bus_order(bus_groups)
    active_entries: dict[str, object] = {}
    for bus_name in bus_names:
        plan = state.analog_bus_plan(bus_name)
        bus_entries: list[tuple[int, object]] = []
        for period_index, entry in enumerate(plan):
            mode = str(entry.get("mode", "hold")).strip().lower()
            value = entry.get("value")
            bound_variable = state.scan_bindings.get(f"bus:{bus_name}:{period_index}:value")
            if bound_variable:
                mode = "edge"
                value = normalize_scan_parameter_name(bound_variable)
            if mode == "hold" or value is None:
                continue
            if mode == "ramp":
                raise ValueError(
                    f"analog bus {bus_name!r} ramp values cannot be combined with a scan table; "
                    "use one static edge value per scan row or run separate prepared pulses."
                )
            if mode != "edge":
                raise ValueError(f"analog bus {bus_name!r} has unsupported scan mode {mode!r}.")
            bus_entries.append((int(period_index), value))
        if len(bus_entries) > 1:
            raise ValueError(
                f"analog bus {bus_name!r} has multiple edge values inside one scan row; "
                "the compact FPGA scan path supports one static DA value per bus per row."
            )
        if bus_entries:
            active_entries[bus_name] = bus_entries[0][1]
    packed_rows: list[int] = []
    for row_index, row in enumerate(scan_rows):
        variables = _scan_table_row_variables(state, row)
        word = 0
        for bus_index, bus_name in enumerate(bus_names):
            members = bus_groups[bus_name]
            value_expr = active_entries.get(bus_name, 0)
            max_value = min((1 << len(members)) - 1, (1 << DEFAULT_RUNTIME_BUS_WIDTH) - 1)
            value = scan_numeric_value(
                value_expr,
                variables=variables,
                name=f"scan row {row_index} analog bus {bus_name!r} value",
            )
            if value < 0 or value > max_value:
                raise ValueError(f"scan row {row_index} analog bus {bus_name!r} value {value} is outside 0..{max_value}.")
            word |= int(value) << (bus_index * DEFAULT_RUNTIME_BUS_WIDTH)
        packed_rows.append(int(word))
    return bus_names, packed_rows


def _pulse_table_bus_segments(
    state: PulseTableState,
    *,
    x_ns: float,
    y_ns: float,
    time_step_ns: float,
    variables: Mapping[str, float] | None = None,
) -> tuple[list[str], list[RuntimeBusSegment]]:
    """Compile logical analog buses into hardware bus segments.

    A ramp consumes one segment regardless of how many 10-bit stair steps it
    produces.  Digital edge rows are left for ordinary TTL outputs.
    """

    starts = _pulse_table_period_starts_ticks(state, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=variables)
    expression_vars = state.scan_variable_values(x_ns=x_ns, y_ns=y_ns)
    expression_vars.update(dict(variables or {}))
    table_end = int(starts[-1])
    bus_groups = state.bus_channels()
    bus_names = _pulse_table_bus_order(bus_groups)
    segments: list[RuntimeBusSegment] = []
    for bus_index, bus_name in enumerate(bus_names):
        members = bus_groups[bus_name]
        plan = state.analog_bus_plan(bus_name)
        if bus_name not in state.analog_bus_modes and all(state.bus_value(index, bus_name) == 0 for index in range(len(state.periods))):
            continue
        delay_steps = _pulse_table_bus_delay_steps(state, members, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=expression_vars)
        anchors: list[tuple[int, int, str, int]] = []
        max_value = (1 << len(members)) - 1
        for period_index, entry in enumerate(plan):
            mode = str(entry.get("mode", "hold")).strip().lower()
            value = entry.get("value")
            if mode not in {"edge", "ramp"} or value is None:
                continue
            value_int = max(0, min(max_value, scan_numeric_value(value, variables=expression_vars, name=f"analog bus {bus_name!r} value")))
            anchors.append((period_index, int(starts[period_index]) + delay_steps, mode, value_int))
        previous: tuple[int, int, str, int] | None = None
        for anchor_index, anchor in enumerate(anchors):
            _period_index, tick, mode, value = anchor
            if tick < 0 or tick > table_end:
                raise ValueError(f"analog bus {bus_name!r} produced segment tick {tick} outside the uploaded table.")
            if previous is None:
                next_anchor = anchors[anchor_index + 1] if anchor_index + 1 < len(anchors) else None
                if next_anchor is None or str(next_anchor[2]).lower() != "ramp":
                    segments.append(RuntimeBusSegment(bus_index, tick, tick, value, value, "edge", bus_name))
            elif mode == "ramp":
                start_tick = int(previous[1])
                start_value = int(previous[3])
                if tick < start_tick:
                    raise ValueError(f"analog bus {bus_name!r} ramp end precedes its start.")
                segments.append(RuntimeBusSegment(bus_index, start_tick, tick, start_value, value, "ramp", bus_name))
            else:
                next_anchor = anchors[anchor_index + 1] if anchor_index + 1 < len(anchors) else None
                if next_anchor is None or str(next_anchor[2]).lower() != "ramp":
                    segments.append(RuntimeBusSegment(bus_index, tick, tick, value, value, "edge", bus_name))
            previous = anchor
    return bus_names, segments


def _pulse_table_has_analog_activity(state: PulseTableState) -> bool:
    for bus_name in state.bus_channels():
        if bus_name in state.analog_bus_modes:
            return True
        if any(state.bus_value(index, bus_name) != 0 for index in range(len(state.periods))):
            return True
    return False


def _pulse_table_has_analog_scan_value(state: PulseTableState) -> bool:
    active = set(state.active_scan_parameters())
    if not active:
        return False
    for entries in state.analog_bus_modes.values():
        for entry in entries:
            value = dict(entry).get("value")
            if scan_parameter_names_from_expr(value) & active:
                return True
    for key, variable in state.scan_bindings.items():
        if str(key).startswith("bus:") and variable in active:
            return True
    return False


def _pulse_table_has_analog_ramp(state: PulseTableState) -> bool:
    for bus_name in state.bus_channels():
        for entry in state.analog_bus_plan(bus_name):
            if str(entry.get("mode", "hold")).lower() == "ramp":
                return True
    return False


def _validate_pulse_table_delays_for_hardware_loop(
    state: PulseTableState,
    *,
    period_starts: Sequence[int],
    x_ns: float,
    y_ns: float = 0.0,
    time_step_ns: float,
    variables: Mapping[str, float] | None = None,
) -> None:
    """Reject delayed edges that a compact FPGA loop cannot replay correctly."""

    table_start = 0
    table_end = int(period_starts[-1])
    has_bracket = state.repeat_start is not None and state.repeat_end is not None
    loop_start = int(period_starts[int(state.repeat_start)]) if has_bracket else table_start
    loop_end = int(period_starts[int(state.repeat_end) + 1]) if has_bracket else table_end
    delayed_spans = _pulse_table_delayed_channel_spans(
        state,
        period_starts=period_starts,
        x_ns=x_ns,
        y_ns=y_ns,
        time_step_ns=time_step_ns,
        variables=variables,
    )
    for channel, raw_start, raw_stop, delay_steps in delayed_spans:
        shifted_start = raw_start + delay_steps
        shifted_stop = raw_stop + delay_steps
        if shifted_start < table_start or shifted_stop > table_end:
            raise ValueError(
                f"delay for {channel!r} moves a pulse outside the uploaded period table; "
                "add guard/idle periods or reduce the delay before using FPGA repeat."
            )
        if not has_bracket:
            continue
        raw_inside = raw_start >= loop_start and raw_stop <= loop_end
        shifted_inside = shifted_start >= loop_start and shifted_stop <= loop_end
        shifted_intersects = shifted_start < loop_end and shifted_stop > loop_start
        if raw_inside != shifted_inside or (not raw_inside and shifted_intersects):
            raise ValueError(
                f"delay for {channel!r} moves a pulse across the repeat bracket boundary; "
                "move the bracket, add guard periods, or keep delayed edges inside the bracket."
            )


def _pulse_table_delayed_channel_spans(
    state: PulseTableState,
    *,
    period_starts: Sequence[int],
    x_ns: float,
    y_ns: float = 0.0,
    time_step_ns: float,
    variables: Mapping[str, float] | None = None,
) -> list[tuple[str, int, int, int]]:
    spans: list[tuple[str, int, int, int]] = []
    for channel_index, channel in enumerate(state.channels):
        delay_steps = state.delay_steps(channel, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=variables)
        if delay_steps == 0:
            continue
        active_start: int | None = None
        for period_index, period in enumerate(state.periods):
            state_value = int(period.states[channel_index])
            if state_value and active_start is None:
                active_start = int(period_starts[period_index])
            elif not state_value and active_start is not None:
                spans.append((channel, active_start, int(period_starts[period_index]), delay_steps))
                active_start = None
        if active_start is not None:
            spans.append((channel, active_start, int(period_starts[-1]), delay_steps))
    return spans


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
    x_ns: float,
    y_ns: float = 0.0,
    time_step_ns: float,
    variables: Mapping[str, float] | None = None,
) -> bool:
    return any(state.delay_steps(channel, x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=variables) != 0 for channel in state.channels)


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
