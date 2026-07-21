"""The pulse-table runtime compiler: authoring model -> FPGA-ready wire program.

This is the machine-verified compile math behind every real shot: period edge tables,
DAC bus segments with per-bus delays, the streamed affine scan program, and the wire
codec the FPGA server consumes.  It lives beside the authoring model
(:mod:`.pulse_table`) because they share one lifecycle: a table edit changes what this
module emits, nothing else does.  Transport (who ships the program to which socket) is
someone else's job and imports from here, never the reverse.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .clock import default_clock_hz
from .clock import default_clock_hz as _default_clock_hz
from .streamer_geometry import DEFAULT_COEFF_FRAC_BITS
from .serialization import (require_array, require_bool, require_exact_fields,
                            require_int, require_number, require_object, require_string)
from zlc_data.validation import nonnegative_float, positive_int
from .ports import PortCatalog, coerce_port_catalog
from .sequence_model import (PulseSequence, channel_names, positive_float, round_ticks)
from .pulse_table import (UNITS_TO_NS, PulseTableState, analog_bus_ticks,
                          _analog_bus_value_at_tick, bus_zero_code, snap_scan_table,
                          slot_var, is_slot_ref, slot_ref_index as table_slot_ref_index,
                          affine_coeffs)


DEFAULT_RUNTIME_CLOCK_HZ = _default_clock_hz()

# Serialized RuntimeSequenceProgram schema version -- ONE source, written by to_dict AND checked by
# from_dict (#G4) so a future schema bump fails fast with a rebuild message instead of mis-decoding.
_RUNTIME_PROGRAM_VERSION = 4

#: The idle scan-progress reading -- no scan running.  SINGLE SOURCE for the dict shape every
#: SequencerDevice.scan_progress() returns, so the GUI poll + the virtual/real backends agree.
SCAN_PROGRESS_IDLE: dict[str, object] = {
    "scanning": False, "point": 0, "n_points": 0, "sweep": 0, "n_repeats": 0,
}

#: The ONE deadlock-guard message for "wait forever on a repeat_forever program" -- shared by the
#: service's wait_done and the virtual backend's real-time override, so the two can never drift.
WAIT_FOREVER_MESSAGE = ("sequencer wait_done cannot wait forever for a repeat_forever program; "
                        "pass a timeout or stop the pulse.")

# RPyC-transport array fields.  A large numeric list (the scan_table forward, the compiled
# scan_points / scan_point_durations on return) crosses the RPyC wire as ONE raw little-endian
# ndarray buffer instead of a per-number JSON list -- removing the O(N) json.dumps/loads over the
# tens of thousands of scan numbers (measured ~40 ms round-trip at 20000 points).  The rebuilt value
# is a NATIVE Python list identical to what json.loads would have produced, so the compiled program
# is byte-for-byte the same as the all-JSON path (proven by an in-process RPyC-loopback hash check).
_WIRE_ARRAY_FIELDS_PAYLOAD = ("scan_table",)

_WIRE_ARRAY_FIELDS_PROGRAM = ("scan_points", "scan_point_durations")



def _channel_delays_list(channel_delays: Mapping[int, int] | None, n_channels: int) -> list[int]:
    """Dense per-channel delay list in FPGA bit order (0 where unset).

    The host<->program packing of the ``{bit: delay}`` map, factored out so both
    compilers' payload-build sites share one implementation."""
    cd = channel_delays or {}
    return [int(cd.get(bit, 0)) for bit in range(n_channels)]

def _clk_enable_mask_for_channels(channels: Sequence[str], clk_channels: Sequence[str]) -> int:
    """clk-enable bitmask in the COMPILED hardware channel order (bit n = ``channels[n]``).

    The program's edge masks are in the order passed to the compiler (which may differ from
    ``state.port_catalog.raw_lanes``); using a mask in another order would point it
    at the wrong bits and could clear a real engine bit.  Always derive it from the order the
    masks actually use."""
    clk_set = {str(c) for c in clk_channels}
    mask = 0
    for index, channel in enumerate(channels):
        if str(channel) in clk_set:
            mask |= 1 << index
    return mask

@dataclass(frozen=True)
class RuntimeBusDelay:
    """One delayed analog DAC bus for the per-bus EVENT-SCHEDULED output delay.

    ``bus_index`` is the hardware bus; ``delay`` is the physical delay ``d`` in ticks
    (>= 0 after the host folds the global negative-delay shift G, capped by the host's
    ``TTL_DELAY_MAX_TICKS`` default -- the same 32-bit-field range as TTL delays).  Each
    DA bit is its own 1-bit event-scheduler channel: when the UNDELAYED bus value changes,
    each bit's new level is queued in that bit's event FIFO and pops exactly ``d`` ticks
    later (``bus_out[t] = bus_undelayed[t - d]``); the 10 bits of a bus share one ``d`` so
    the DAC value shifts coherently.  Storage scales with value-change events IN FLIGHT
    (host-validated <= the per-bit FIFO depth), not with ``d`` -- the DAC-value counterpart
    of a per-channel ``channel_delays`` entry."""

    bus_index: int
    delay: int

    def to_dict(self) -> dict[str, object]:
        return {"bus_index": int(self.bus_index), "delay": int(self.delay)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeBusDelay":
        require_exact_fields(payload, {"bus_index", "delay"}, what="runtime bus delay")
        return cls(
            bus_index=require_int(payload["bus_index"], what="runtime bus delay.bus_index"),
            delay=require_int(payload["delay"], what="runtime bus delay.delay"),
        )

def _fold_global_delay_shift(
    channel_raw: Mapping[Any, int],
    bus_raw: Mapping[int, int],
) -> tuple[dict[Any, int], dict[int, int], int]:
    """Fold the negative-delay global shift ``G`` into TTL channels and DAC buses.

    A causal delay line cannot LEAD, so a negative delay re-translates the WHOLE
    frame: shift every channel and bus by ``G = max(0, -min(all raw delays))`` so
    every realised delay lands ``>= 0`` while relative timing is preserved (the
    "-2 s emCCD shows DA_bias_y first" bug needs G to reach the buses too).  ``G``
    is computed over the TTL ``channel_raw`` and DAC ``bus_raw`` delays TOGETHER --
    they are one physical schedule.  A folded delay equal to 0 is dropped from both
    maps (an undelayed signal carries no delay word).  This is the SINGLE source for
    both the scanned and non-scanned pulse-table compilers; the keys of ``channel_raw``
    (channel name vs hardware-bit index) are opaque and round-trip unchanged."""

    all_raw = list(channel_raw.values()) + list(bus_raw.values())
    global_shift = max(0, -min(all_raw)) if all_raw else 0
    channel_delays = {
        ch: channel_raw[ch] + global_shift
        for ch in channel_raw
        if (channel_raw[ch] + global_shift) != 0
    }
    bus_delays = {
        bus: bus_raw[bus] + global_shift
        for bus in bus_raw
        if (bus_raw[bus] + global_shift) != 0
    }
    return channel_delays, bus_delays, global_shift

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
    scanned DURATION moves the segment in lockstep with the digital edges -- this is
    what lets a DAC value + duration scan simultaneously.  (A channel DELAY is a fixed
    per-channel value and is NOT a scan slot.)"""
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
        fields = {
            "bus_index", "bus_name", "start_tick", "stop_tick",
            "start_value", "stop_value", "mode", "value_select",
            "stop_value_select", "start_tick_coeffs", "stop_tick_coeffs",
        }
        require_exact_fields(payload, fields, what="runtime bus segment")
        return cls(
            bus_index=require_int(payload["bus_index"], what="runtime bus segment.bus_index"),
            bus_name=require_string(payload["bus_name"], what="runtime bus segment.bus_name"),
            start_tick=require_int(payload["start_tick"], what="runtime bus segment.start_tick"),
            stop_tick=require_int(payload["stop_tick"], what="runtime bus segment.stop_tick"),
            start_value=require_int(payload["start_value"], what="runtime bus segment.start_value"),
            stop_value=require_int(payload["stop_value"], what="runtime bus segment.stop_value"),
            mode=require_string(payload["mode"], what="runtime bus segment.mode").strip().lower(),
            value_select=require_int(
                payload["value_select"], what="runtime bus segment.value_select"),
            stop_value_select=require_int(
                payload["stop_value_select"], what="runtime bus segment.stop_value_select"),
            start_tick_coeffs=[
                require_int(v, what="runtime bus segment.start_tick_coeffs item")
                for v in require_array(
                    payload["start_tick_coeffs"],
                    what="runtime bus segment.start_tick_coeffs")
            ] or None,
            stop_tick_coeffs=[
                require_int(v, what="runtime bus segment.stop_tick_coeffs item")
                for v in require_array(
                    payload["stop_tick_coeffs"],
                    what="runtime bus segment.stop_tick_coeffs")
            ] or None,
        )

@dataclass(frozen=True)
class RuntimeSequenceProgram:
    """Runtime edge-table program uploaded to a pulse-streamer-like FPGA."""

    # The on-disk/wire identity of this payload -- the CLASS owns it (same pattern as
    # PulseSequence.schema / PulseTableState.schema): to_dict writes it, from_dict checks it,
    # and the timing_from_payload dispatcher compares against it, so the string has exactly
    # one spelling.  (No type annotation: a dataclass must not treat it as a field.)
    schema = "Zou_lab_control.neutral_atom.RuntimeSequenceProgram"

    sequence_id: str
    sequence_name: str
    clock_hz: float
    channels: list[str]
    ticks: list[int]
    masks: list[int]
    duration: float
    source_sequence: dict[str, Any] | None = None
    source_table: dict[str, Any] | None = None
    port_catalog_fingerprint: str = ""
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
    scan_coeff_frac_bits: int = DEFAULT_COEFF_FRAC_BITS   # config single source, not a bare literal 8
    # Number of FULL scan sweeps before the scan stops: 0 = sweep forever (seamless cyclic
    # streaming, the default), K>=1 = play every scan point K times then halt (the host counts
    # sweeps from the CURSOR wrap and issues the engine stop -- no RTL change).  Inert unless
    # ``scan_points`` is set.
    scan_repeats: int = 0
    bus_names: list[str] | None = None
    bus_segments: list[RuntimeBusSegment] | None = None
    # PHYSICAL DAC-BUS DELAY: per-bus delay in ticks, realised by the per-signal EVENT SCHEDULER
    # (each DA bit its own event FIFO; one delay shared by all 10 bits).  The DAC-value counterpart
    # of ``channel_delays``; empty/None = no bus delayed.  Bounded by events in flight, not length.
    bus_delays: list[RuntimeBusDelay] | None = None
    # PHYSICAL CHANNEL DELAY: per-channel-bit delay in ticks, applied to the engine OUTPUT by the
    # per-signal EVENT SCHEDULER, NOT baked into ``ticks``.
    # ``ticks``/``masks`` are the UNDELAYED frame; the engine delays bit ``b`` by
    # ``channel_delays[b]`` -- out[t]=in[t-d], 0 before fire (see engine_model.delay_line_reference).
    # Any d in [0, the 32-bit field cap]; never disturbs another channel; first frame real.  The
    # global negative-delay shift G is folded in so every entry is >= 0.  Empty/None = none delayed.
    channel_delays: list[int] | None = None
    # Bitmask (bit b = channel b) of channels wired directly to the FPGA clk.  These bits
    # are forced 0 in every edge mask (the engine does not drive them); the top muxes clk
    # onto their pins via this mask.  0 = no clk channels.
    clk_enable: int = 0

    def __post_init__(self) -> None:
        # repeat_forever and scan_points are ORTHOGONAL here, and this pure data contract does NOT
        # decide one from the other.  ``repeat_forever`` is the host-side CYCLIC intent: 0-repeat
        # seamless forever / K whole sweeps then stop.  ``scan_points`` is the table fact: any bound
        # scan carries its physical points.  A program can legitimately be a finite single-pass scan
        # (repeat_forever=False + scan_points), so this carrier must never force one value from the
        # other.  The current hardware owner admits only fully resident autonomous tables; this
        # legacy carrier does not decide deployment capacity.  The CYCLIC intent for a GUI/notebook
        # scan is owned by the fire seam
        # (PulseController.on_pulse passes repeat_forever for On Pulse "continuous until Stop"); the
        # streaming/progress gate downstream keys off scan_points, not this flag.
        # A finite K-sweep streamed scan (scan_repeats>0) needs >= 2 scan points: the host counts whole
        # sweeps from the streamed cursor's WRAP, which a single point never produces -- it would never
        # stop on real hardware.  Enforced HERE, on the COMPILED program, because EVERY fire path (real
        # and virtual both compile to a RuntimeSequenceProgram) passes through it and it is NEVER built
        # during GUI mid-edit (the editor holds a PulseTableState, not this) -- so the rule cannot be
        # bypassed and virtual == real, without a transient 1-row edit ever raising (#G3).
        if int(self.scan_repeats) > 0 and self.scan_points is not None and len(self.scan_points) < 2:
            raise ValueError(
                "a finite scan-repeat (scan_repeats > 0) needs at least 2 scan points: the host counts "
                "whole sweeps from the streamed cursor's wrap, which a single point never produces (it "
                "would never stop). Add scan points, or use scan_repeats=0 for a seamless cyclic scan.")

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "version": _RUNTIME_PROGRAM_VERSION,
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "clock_hz": self.clock_hz,
            "channels": list(self.channels),
            "ticks": list(self.ticks),
            "masks": list(self.masks),
            "duration": self.duration,
            "port_catalog_fingerprint": str(self.port_catalog_fingerprint),
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
            "scan_repeats": int(self.scan_repeats),
            "bus_names": list(self.bus_names or []),
            "bus_segments": [segment.to_dict() for segment in (self.bus_segments or [])],
            "bus_delays": [bd.to_dict() for bd in (self.bus_delays or [])],
            "channel_delays": list(self.channel_delays or []),
            "clk_enable": int(self.clk_enable),
        }
        if self.source_sequence is not None:
            payload["source_sequence"] = self.source_sequence
        if self.source_table is not None:
            payload["source_table"] = self.source_table
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeSequenceProgram":
        required = {
            "schema", "version", "sequence_id", "sequence_name", "clock_hz",
            "channels", "ticks", "masks", "duration", "port_catalog_fingerprint",
            "repeat_forever", "loop_start_index", "loop_end_tick", "loop_count",
            "repeat_from_index", "slot_count", "slot_kinds", "loop_end_slot_coeffs",
            "tick_slot_coeffs", "scan_points", "scan_point_durations",
            "scan_coeff_frac_bits", "scan_repeats", "bus_names", "bus_segments",
            "bus_delays", "channel_delays", "clk_enable",
        }
        require_exact_fields(
            payload, required, optional={"source_sequence", "source_table"},
            what="runtime sequence program")
        if payload["schema"] != cls.schema:
            raise ValueError(
                f"unsupported runtime sequence program schema {payload['schema']!r}; "
                f"expected {cls.schema!r}.")
        # ``version`` is load-bearing, not a written-but-unread field (#G4): a future schema bump
        # fails fast here with a clear rebuild message instead of silently mis-decoding an old payload.
        version = require_int(payload["version"], what="runtime sequence program.version")
        if version != _RUNTIME_PROGRAM_VERSION:
            raise ValueError(
                f"unsupported runtime sequence program version {version} "
                f"(this host reads/writes version {_RUNTIME_PROGRAM_VERSION}); rebuild the program.")
        slot_count = require_int(
            payload["slot_count"], what="runtime sequence program.slot_count")
        tick_slot_coeffs = [
            [
                require_int(v, what="runtime sequence program.tick_slot_coeffs item")
                for v in require_array(
                    row, what="runtime sequence program.tick_slot_coeffs row")
            ]
            for row in require_array(
                payload["tick_slot_coeffs"],
                what="runtime sequence program.tick_slot_coeffs")
        ]
        if tick_slot_coeffs and not any(any(row) for row in tick_slot_coeffs):
            tick_slot_coeffs = []
        source_sequence = payload.get("source_sequence")
        if source_sequence is not None:
            source_sequence = require_object(
                source_sequence, what="runtime sequence program.source_sequence")
        source_table = payload.get("source_table")
        if source_table is not None:
            source_table = require_object(
                source_table, what="runtime sequence program.source_table")
        return cls(
            sequence_id=require_string(
                payload["sequence_id"], what="runtime sequence program.sequence_id"),
            sequence_name=require_string(
                payload["sequence_name"], what="runtime sequence program.sequence_name"),
            clock_hz=positive_float(
                require_number(payload["clock_hz"], what="runtime sequence program.clock_hz"),
                "clock_hz"),
            channels=list(channel_names([
                require_string(v, what="runtime sequence program.channels item")
                for v in require_array(
                    payload["channels"], what="runtime sequence program.channels")
            ], "channels")),
            ticks=[
                require_int(v, what="runtime sequence program.ticks item")
                for v in require_array(
                    payload["ticks"], what="runtime sequence program.ticks")
            ],
            masks=[
                require_int(v, what="runtime sequence program.masks item")
                for v in require_array(
                    payload["masks"], what="runtime sequence program.masks")
            ],
            duration=require_number(
                payload["duration"], what="runtime sequence program.duration"),
            source_sequence=None if source_sequence is None else dict(source_sequence),
            source_table=None if source_table is None else dict(source_table),
            port_catalog_fingerprint=require_string(
                payload["port_catalog_fingerprint"],
                what="runtime sequence program.port_catalog_fingerprint"),
            repeat_forever=require_bool(
                payload["repeat_forever"], what="runtime sequence program.repeat_forever"),
            loop_start_index=require_int(
                payload["loop_start_index"], what="runtime sequence program.loop_start_index"),
            loop_end_tick=require_int(
                payload["loop_end_tick"], what="runtime sequence program.loop_end_tick"),
            loop_count=require_int(
                payload["loop_count"], what="runtime sequence program.loop_count"),
            repeat_from_index=require_int(
                payload["repeat_from_index"], what="runtime sequence program.repeat_from_index"),
            slot_count=slot_count,
            slot_kinds=[
                require_string(v, what="runtime sequence program.slot_kinds item")
                for v in require_array(
                    payload["slot_kinds"], what="runtime sequence program.slot_kinds")
            ] or None,
            loop_end_slot_coeffs=[
                require_int(v, what="runtime sequence program.loop_end_slot_coeffs item")
                for v in require_array(
                    payload["loop_end_slot_coeffs"],
                    what="runtime sequence program.loop_end_slot_coeffs")
            ] or None,
            tick_slot_coeffs=tick_slot_coeffs or None,
            scan_points=[
                [
                    require_int(v, what="runtime sequence program.scan_points item")
                    for v in require_array(
                        row, what="runtime sequence program.scan_points row")
                ]
                for row in require_array(
                    payload["scan_points"], what="runtime sequence program.scan_points")
            ] or None,
            scan_point_durations=[
                require_number(v, what="runtime sequence program.scan_point_durations item")
                for v in require_array(
                    payload["scan_point_durations"],
                    what="runtime sequence program.scan_point_durations")
            ] or None,
            scan_coeff_frac_bits=require_int(
                payload["scan_coeff_frac_bits"],
                what="runtime sequence program.scan_coeff_frac_bits"),
            scan_repeats=require_int(
                payload["scan_repeats"], what="runtime sequence program.scan_repeats"),
            bus_names=[
                require_string(v, what="runtime sequence program.bus_names item")
                for v in require_array(
                    payload["bus_names"], what="runtime sequence program.bus_names")
            ] or None,
            bus_segments=[
                RuntimeBusSegment.from_dict(require_object(
                    item, what="runtime sequence program.bus segment"))
                for item in require_array(
                    payload["bus_segments"], what="runtime sequence program.bus_segments")
            ] or None,
            bus_delays=[
                RuntimeBusDelay.from_dict(require_object(
                    item, what="runtime sequence program.bus delay"))
                for item in require_array(
                    payload["bus_delays"], what="runtime sequence program.bus_delays")
            ] or None,
            channel_delays=[
                require_int(v, what="runtime sequence program.channel_delays item")
                for v in require_array(
                    payload["channel_delays"],
                    what="runtime sequence program.channel_delays")
            ] or None,
            clk_enable=require_int(
                payload["clk_enable"], what="runtime sequence program.clk_enable"),
        )

    @property
    def scan_enabled(self) -> bool:
        return bool(self.scan_points)

def scan_progress_fields(point_total: int, n_points: int, scan_repeats: int) -> dict[str, object]:
    """Turn a MONOTONIC played-point count into the scan-progress reading.

    ``point_total`` = how many scan points have been played since the scan started (it keeps
    counting across sweep wraps: sweep s, point p -> s*N + p).  Returns the SINGLE-source dict
    {scanning, point (0-based in the current sweep), n_points N, sweep (0-based), n_repeats K}.
    For a finite scan (K>=1) the reading saturates at the last point of sweep K-1 and reports
    ``scanning=False`` once K full sweeps are done; an infinite scan (K=0) never stops.  Shared
    by the virtual sequencer (point_total from wall-clock) and the real streamer (point_total =
    CURSOR + wraps), so both report progress identically (virtual==real)."""

    n = int(n_points)
    k = max(0, int(scan_repeats))
    if n <= 0:
        return dict(SCAN_PROGRESS_IDLE)
    total = max(0, int(point_total))
    if k > 0 and total >= k * n:
        # K full sweeps done: saturate at the last point and stop.
        return {"scanning": False, "point": n - 1, "n_points": n, "sweep": k - 1, "n_repeats": k}
    return {"scanning": True, "point": total % n, "n_points": n, "sweep": total // n, "n_repeats": k}

def compile_runtime_program(
    sequence: PulseSequence,
    *,
    channels: Sequence[str],
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    port_catalog: PortCatalog | Mapping[str, object] | None = None,
) -> RuntimeSequenceProgram:
    """Compile a ``PulseSequence`` into an uploadable edge table."""

    catalog = coerce_port_catalog(port_catalog, channels=channels)
    channels = list(catalog.raw_lanes)
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
        "port_catalog_fingerprint": catalog.fingerprint,
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
        source_sequence=sequence.to_dict(),
        source_table=sequence.source_table,
        port_catalog_fingerprint=catalog.fingerprint,
        repeat_forever=bool(sequence.repeat_forever),
        loop_start_index=0,
        loop_end_tick=loop_end_tick,
        loop_count=int(sequence.repeat_count),
    )

def _resolve_hardware_catalog(
    state: "PulseTableState",
    port_catalog: PortCatalog | Mapping[str, object] | None = None,
) -> tuple[list[str], PortCatalog]:
    """Resolve the one catalog a pulse-table compiler is allowed to use.

    A pulse table is already bound to a complete immutable topology.  Passing
    a connected device catalog is only a fingerprint assertion; it cannot
    supply a second raw-lane universe or silently align the document.
    """

    catalog = state.port_catalog
    if port_catalog is not None:
        device_catalog = (
            port_catalog if isinstance(port_catalog, PortCatalog)
            else PortCatalog.from_dict(port_catalog)
        )
        if device_catalog.fingerprint != catalog.fingerprint:
            pulse_only = sorted(set(catalog.raw_lanes) - set(device_catalog.raw_lanes))
            device_only = sorted(set(device_catalog.raw_lanes) - set(catalog.raw_lanes))
            raise ValueError(
                "pulse port topology does not match the connected sequencer: "
                f"pulse={catalog.fingerprint}, device={device_catalog.fingerprint}. "
                f"Pulse-only raw lanes: {pulse_only}; device-only raw lanes: {device_only}. "
                "Align the template to the device PortCatalog before compiling.")
    if not catalog.raw_lanes:
        raise ValueError(
            "pulse PortCatalog must contain at least one raw lane.")
    return list(catalog.raw_lanes), catalog

def compile_pulse_table_runtime_program(
    state: PulseTableState,
    *,
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    slots: Mapping[str, float] | None = None,
    repeat_forever: bool = True,
    port_catalog: PortCatalog | Mapping[str, object] | None = None,
) -> RuntimeSequenceProgram:
    """Compile GUI period-card state into an unexpanded FPGA loop program.

    ``PulseTableState`` carries the frontend repeat-bracket semantics.  The
    runtime FPGA should receive one copy of the period table plus loop metadata,
    not a fully expanded edge table.  A bracket becomes one finite inner loop;
    the whole table may still be repeated forever by the FPGA.  Any bound scan
    slots are resolved to constants using ``slots`` (default: the reference
    scan point), so this path emits a single static program.
    """

    channels, catalog = _resolve_hardware_catalog(state, port_catalog)
    clock_hz = positive_float(clock_hz, "clock_hz")
    clock_step_ns = 1e9 / clock_hz
    slot_values = state._reference_slots() if slots is None else state._compiler_values_for(slots)

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
            clock_hz=clock_hz,
            slots=slots,
            repeat_forever=repeat_forever,
            port_catalog=catalog,
        )

    state.validate(
        slots=slot_values, time_step_ns=clock_step_ns, _slots_are_compiler=True)
    sequence = state.to_sequence(
        slots=state._semantic_values_from_compiler(slot_values),
        time_step_ns=clock_step_ns, expand_repeat=False)
    period_starts = state.period_start_steps(slots=slot_values, time_step_ns=clock_step_ns)
    bus_names, bus_segments, raw_bus_delays = _pulse_table_bus_segments(
        state,
        slots=slot_values,
        time_step_ns=clock_step_ns,
    )
    # When buses are emitted as SEGMENTS, their (now nominal-phase) delay is realised by the
    # LITERAL per-bus delay line; pass the raw bus delays so they share the SAME global shift G
    # as the TTL channels (a negative bus delay also lands >= 0).  EVERY DRIVEN bus (one that
    # emits >= 1 segment) must be passed -- even with NO explicit delay (raw 0) -- so a negative
    # TTL delay that shifts the whole frame by G also shifts the DAC buses; otherwise they keep
    # their nominal phase and visibly LEAD (the "-2 s emCCD shows DA_bias_y first" bug).
    driven_bus_raw = (
        {seg.bus_index: int(raw_bus_delays.get(seg.bus_index, 0)) for seg in bus_segments}
        if bus_segments else None
    )
    ticks, masks, channels, loop_end, repeat_from_index, channel_delays, bus_delays_by_index = _pulse_table_edge_table(
        state,
        channels=channels,
        slots=slot_values,
        time_step_ns=clock_step_ns,
        fold_analog_buses=not bool(bus_segments),
        repeat_forever=bool(repeat_forever) and not has_bracket,
        extra_raw_delays=driven_bus_raw,
    )
    bus_delays = [
        RuntimeBusDelay(bus_index=bus_index, delay=int(bus_delays_by_index[bus_index]))
        for bus_index in sorted(bus_delays_by_index)
    ]
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

    # Channels wired to clk are driven by the top's clk mux, NOT the engine: force their
    # bits to 0 in every edge mask so the engine never fights the clk routing.  Compute the
    # mask in the COMPILED channel order (masks use that order, which may differ from state).
    clock_lanes = [port.lanes[0] for port in state.port_catalog.clock_ports]
    clk_enable = _clk_enable_mask_for_channels(channels, clock_lanes)
    if clk_enable:
        masks = [int(mask) & ~clk_enable for mask in masks]
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
        "bus_delays": [bd.to_dict() for bd in bus_delays],
        "channel_delays": _channel_delays_list(channel_delays, len(channels)),
        "clk_enable": int(clk_enable),
        "port_catalog_fingerprint": catalog.fingerprint,
    }
    channel_delays_list = _channel_delays_list(channel_delays, len(channels)) if channel_delays else None
    sequence_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    # The physical channel-delay bound (and the full edge/FIFO/monotonic capacity contract) is
    # enforced once by validate_pulse_streamer_program -- the authoritative gate every program
    # passes through at host prepare / AXI upload (#G1, single source for the bound).
    return RuntimeSequenceProgram(
        sequence_id=sequence_id,
        sequence_name=state.name,
        clock_hz=clock_hz,
        channels=list(channels),
        ticks=list(ticks),
        masks=list(masks),
        duration=effective_duration_ticks / clock_hz,
        source_sequence=sequence.to_dict(),
        source_table=state.to_dict(),
        port_catalog_fingerprint=catalog.fingerprint,
        repeat_forever=bool(repeat_forever),
        loop_start_index=loop_start_index,
        loop_end_tick=loop_end_tick,
        loop_count=loop_count,
        repeat_from_index=repeat_from_index,
        bus_names=bus_names or None,
        bus_segments=bus_segments or None,
        bus_delays=bus_delays or None,
        channel_delays=channel_delays_list,
        clk_enable=int(clk_enable),
    )

def compile_pulse_table_scan_runtime_program(
    state: PulseTableState,
    *,
    scan_table: Sequence[Sequence[float]] | None = None,
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    repeat_forever: bool = False,
    coeff_frac_bits: int = DEFAULT_COEFF_FRAC_BITS,   # config single source (bitstream-affecting)
    port_catalog: PortCatalog | Mapping[str, object] | None = None,
) -> RuntimeSequenceProgram:
    """Compile a ``PulseTableState`` with bound scan slots into a scan program.

    Each ``scan_table`` row is one scan point; column ``j`` is the value of
    slot ``j`` (named ``s{j}``, in the slot's display unit).  Bound durations and
    delays are *time slots*: they enter the affine tick formula
    ``tick = base + (sum_j coeff_j * slot_tick_j) >> coeff_frac_bits``.  The
    hardware iterates the scan points seamlessly; only the template and the
    parameter table are uploaded.
    """

    channels, catalog = _resolve_hardware_catalog(state, port_catalog)
    clock_hz = positive_float(clock_hz, "clock_hz")
    clock_step_ns = 1e9 / clock_hz
    if not state.scan_slots:
        raise ValueError("hardware scan requires at least one bound scan slot; bind a duration or DAC value first.")

    # A (constant) channel delay inside a finite repeat bracket: UNROLL the bracket into a flat
    # period list first, then compile with the flat affine machinery (additive global shift G +
    # affine duration scan).  Once flat there is no inner-loop boundary
    # for a delayed edge to cross, so a constant delay works in ANY form -- crossing the (former)
    # boundary, reordering, negative, or frame-extending.  (A channel delay is a FIXED value and
    # cannot itself be scanned; a delay expression referencing a scanned slot is rejected below.)
    has_bracket = state.repeat_start is not None and state.repeat_end is not None
    if has_bracket and _pulse_table_has_any_delay(state):
        unrolled = state.unrolled_bracket()
        _check_unrolled_edge_budget(unrolled, slots=unrolled._reference_slots(), time_step_ns=clock_step_ns)
        return compile_pulse_table_scan_runtime_program(
            unrolled,
            scan_table=scan_table,
            clock_hz=clock_hz,
            repeat_forever=repeat_forever,
            coeff_frac_bits=coeff_frac_bits,
            port_catalog=catalog,
        )

    # DAC value + duration scan simultaneously: every analog-bus segment's ticks are
    # emitted as affine expressions (base + per-slot coeffs), so a scanned DURATION moves
    # the segment -- and any ramp's start/stop ticks -- in lockstep with the digital edges.
    # Ramps with fixed value endpoints therefore scan their TIMING freely; ramps whose value
    # endpoints are themselves scanned use the dual start/stop value_select (see
    # _pulse_table_bus_segments).  (A channel delay is fixed and is not a scan slot.)
    raw_rows = [[float(value) for value in row] for row in (state.scan_table if scan_table is None else scan_table)]
    if not raw_rows:
        raise ValueError("hardware scan requires at least one scan-table row.")
    # Snap + clamp every scan point the SAME way PulseTableState.compile_scan / the GUI /
    # the server do, so the uploaded program matches the hardware REGARDLESS of entry point
    # (a scanned duration of 0 ns becomes >= 1 tick, DAC codes clamp to the bus width).  This
    # also validates the column count against the slot count -- a width mismatch in EITHER
    # direction raises (#C3) -- instead of the old zip() silently truncating.  Previously the
    # snap invariant held only via compile_scan; a direct call (e.g. compile_runtime_program_
    # _for_payload) used the raw table and could emit a zero-length period.
    table = _snap_scan_table(
        raw_rows, state.scan_slots, time_step_ns=clock_step_ns, dac_ranges=state.scan_slot_dac_ranges()
    )
    slot_vars = state.compiler_scan_vars

    def point_slots_ns(row: Sequence[float]) -> dict[str, float]:
        # Time slots carry a physical time (-> ns); DAC slots carry the user-facing
        # SIGNED value (0 = true 0 V) untouched -- the offset-binary wire code is
        # produced below in point_slot_value (the ONE signed->code conversion point).
        return {
            slot_var(index): float(row[index]) * (1.0 if slot.kind == "dac" else UNITS_TO_NS.get(slot.unit, 1.0))
            for index, slot in enumerate(state.scan_slots)
        }

    # Per-slot signed range + zero code so the SIGNED user value becomes the legal
    # offset-binary CODE the bus engine reads raw via value_select (defense in depth --
    # the GUI + snapped() already clamp the signed value; this guarantees a legal code
    # no matter how the program was built).
    dac_slot_ranges = state.scan_slot_dac_ranges()

    def point_slot_value(point_index: int, slot_index: int, ns: Mapping[str, float]) -> int:
        slot = state.scan_slots[slot_index]
        if slot.kind == "dac":
            # WIRE CONVERSION: signed user value -> offset-binary code (no ns->tick
            # conversion; the affine coefficient is 0 so it never enters the edge-tick
            # formula).  code = signed + 2^(B-1), clamped to [0, 2^B-1].
            signed = int(round(float(ns[slot_var(slot_index)])))
            rng = dac_slot_ranges[slot_index]
            if rng is None:
                # scan_slot_dac_ranges() returns a range for EVERY dac slot; a missing one
                # means the slot/bus wiring is broken -- never leak a SIGNED value into the
                # wire-layer scan_points (the FPGA would read it as a raw code).
                raise ValueError(f"dac scan slot {slot_index} has no bus range; cannot convert to a wire code.")
            lo, hi = int(rng[0]), int(rng[1])
            signed = max(lo, min(hi, signed))
            zero_code = -lo                           # lo == -2^(B-1)
            return signed + zero_code
        return _time_ns_to_ticks(
            ns[slot_var(slot_index)], clock_step_ns, f"scan point {point_index} slot {slot_index}", allow_negative=True
        )

    # Validate the whole state ONCE at the reference slots (structure + every period duration and
    # channel delay).  Per scan point, re-validate ONLY the timing that can actually VARY point to
    # point: a period whose duration EXPRESSION references a scan slot (``sN``).  A constant-duration
    # period is slot-invariant, and a channel delay can NEVER reference a scan slot (delays are fixed
    # -- a slot-referencing delay is rejected up front), so both are already proven by the reference
    # validate; re-resolving all periods + delays at every point (the old per-point
    # _validate_slot_timing) was pure O(N) waste.  ``_validate_slot_timing`` remains the single
    # source the reference validate uses.  (Empty ``scan_dep_*`` -> the per-point body is a no-op,
    # exactly matching a scan whose only slots are DAC values.)
    state.validate(
        slots=state._reference_slots(), time_step_ns=clock_step_ns,
        _slots_are_compiler=True)
    _refs_slot = re.compile(r"s\d").search
    scan_dep_periods = [period for period in state.periods if _refs_slot(str(period.duration))]
    scan_dep_delays = [channel for channel, value in state.delays.items() if _refs_slot(str(value))]
    points_ticks: list[list[int]] = []
    for point_index, row in enumerate(table):
        ns = point_slots_ns(row)
        points_ticks.append([
            point_slot_value(point_index, index, ns) for index in range(len(state.scan_slots))
        ])
        for period in scan_dep_periods:
            period.duration_steps(slots=ns, time_step_ns=clock_step_ns)   # raises on a <0 duration here
        for channel in scan_dep_delays:
            state.delay_steps(channel, slots=ns, time_step_ns=clock_step_ns)

    # Analog buses are driven by the hardware bus engine, not the TTL edge table.
    # A scanned DAC value becomes a bus segment whose value_select reads the slot
    # per scan point; we exclude bus member channels from the affine edge rows so
    # they are not also driven as TTL bits.
    bus_names: list[str] = []
    bus_segments: list[RuntimeBusSegment] = []
    # DAC membership is topology, never inferred from whether this particular
    # pulse happens to contain a waveform plan.  Its raw lanes must never leak
    # back into the TTL edge engine when a bus is idle.
    bus_members: list[str] = [
        lane for port in state.port_catalog.dac_ports for lane in port.lanes]
    raw_bus_delays: dict[int, int] = {}
    if _pulse_table_has_analog_activity(state):
        bus_names, bus_segments, raw_bus_delays = _pulse_table_bus_segments(
            state,
            slots=state._reference_slots(),
            time_step_ns=clock_step_ns,
            slot_vars=slot_vars,
            coeff_frac_bits=coeff_frac_bits,
        )

    # PHYSICAL CHANNEL DELAY: a delay is NOT scanned and NOT baked into the edges -- it is a
    # CONSTANT per-channel OUTPUT delay (a delay line; see engine_model.delay_line_reference).
    # Compute it over the TTL (non-bus) channels, folding the negative-delay global shift
    # G = max(0, -min delay) into every channel's delay (a causal delay line cannot lead, so
    # shifting all of them by G keeps relative timing while every delay stays >= 0).  The edge
    # table is emitted UNDELAYED and the loop period is the plain (affine-in-duration) frame,
    # so a delay of ANY length never disturbs another channel and never changes the period.
    # The DAC BUSES go through the SAME global shift G (their delays are folded with the TTL
    # delays so a negative bus delay also lands >= 0), then are realised by the SAME per-signal
    # event scheduler as the TTL channels (32-bit range; the host validates events-in-flight).
    hardware_bits = {ch: index for index, ch in enumerate(channel_names(channels, "channels"))}
    # A channel delay is a FIXED per-channel OUTPUT delay -- it cannot vary per scan point.
    # Reject a delay EXPRESSION that references a scanned slot (a nonzero affine coeff): it would
    # otherwise be silently FROZEN at the reference value (channel_delays is one constant array,
    # not per-point).  Scan the duration instead.
    for ch, raw in state.delays.items():
        if isinstance(raw, str):
            _, dcoeffs = affine_coeffs(raw, slot_vars=slot_vars,
                                       unit=state.delay_units.get(ch, "ns"),
                                       time_step_ns=clock_step_ns, coeff_frac_bits=coeff_frac_bits)
            if any(int(c) != 0 for c in dcoeffs):
                raise ValueError(
                    f"channel {ch!r} delay {raw!r} references a scanned slot; a channel delay is "
                    "a fixed per-channel value and cannot be scanned (scan the duration instead).")
    clk_set = {port.lanes[0] for port in state.port_catalog.clock_ports}
    raw_delay = {}
    for ch in state.port_catalog.raw_lanes:
        # clk channels are clk-mux driven (no engine output); OFF channels emit nothing -- neither
        # may contribute a delay that would shift the global frame G and delay ACTIVE channels.
        if ch in bus_members or ch not in hardware_bits or ch in clk_set:
            continue
        ch_index = state.channel_index(ch)
        if not any(int(period.states[ch_index]) for period in state.periods):
            continue
        raw_delay[ch] = state.delay_steps(ch, slots=state._reference_slots(), time_step_ns=clock_step_ns)
    # EVERY DRIVEN bus (emits >= 1 segment) must inherit the global shift G, even with no
    # explicit delay of its own -- so a negative TTL delay shifts the DAC buses in lockstep
    # with the TTLs instead of letting them lead (the "-2 s emCCD shows DA_bias_y first" bug).
    # Seed those buses at raw 0 so they fold like an explicitly-zero delay.
    driven_bus_indices = sorted({seg.bus_index for seg in bus_segments})
    bus_raw = {b: raw_bus_delays.get(b, 0) for b in driven_bus_indices}
    channel_delays_by_name, bus_delays_by_index, _global_shift = _fold_global_delay_shift(
        raw_delay, bus_raw)
    channel_delays = {hardware_bits[ch]: d for ch, d in channel_delays_by_name.items()}
    bus_delays = [
        RuntimeBusDelay(bus_index=b, delay=d) for b, d in sorted(bus_delays_by_index.items())
    ]

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
    # Channels wired to clk are driven by the top's clk mux, not the engine -> 0 in masks.
    # Mask in the compiled raw-lane order so it lands on the right bits.
    clk_enable = _clk_enable_mask_for_channels(channels, clk_set)
    if clk_enable:
        masks = [int(mask) & ~clk_enable for mask in masks]
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
    sequence = state.to_sequence(
        slots=state._semantic_values_from_compiler(point_slots_ns(table[0])),
        time_step_ns=clock_step_ns, expand_repeat=False)
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
        "bus_delays": [bd.to_dict() for bd in bus_delays],
        "channel_delays": _channel_delays_list(channel_delays, len(channels)),
        "clk_enable": int(clk_enable),
        "port_catalog_fingerprint": catalog.fingerprint,
    }
    channel_delays_list = _channel_delays_list(channel_delays, len(channels)) if channel_delays else None
    sequence_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    # The physical channel-delay bound (and the full edge/FIFO/monotonic capacity contract) is
    # enforced once by validate_pulse_streamer_program -- the authoritative gate every program
    # passes through at host prepare / AXI upload (#G1, single source for the bound).
    return RuntimeSequenceProgram(
        sequence_id=sequence_id,
        sequence_name=state.name,
        clock_hz=clock_hz,
        channels=list(channels),
        ticks=ticks,
        masks=masks,
        duration=sum(point_durations),
        source_sequence=sequence.to_dict(),
        source_table=source_table,
        port_catalog_fingerprint=catalog.fingerprint,
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
        scan_repeats=max(0, int(getattr(state, "scan_repeats", 0))),
        bus_names=bus_names or None,
        bus_segments=bus_segments or None,
        bus_delays=bus_delays or None,
        channel_delays=channel_delays_list,
        clk_enable=int(clk_enable),
    )

def compile_runtime_program_for_payload(
    payload: PulseSequence | PulseTableState,
    *,
    channels: Sequence[str] | None = None,
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    port_catalog: PortCatalog | Mapping[str, object] | None = None,
) -> RuntimeSequenceProgram:
    """Compile either finite sequence data or GUI pulse-table data."""

    if isinstance(payload, PulseTableState):
        # Scan path only when there are BOTH bound slots AND at least one scan-table row.
        # Bound slots with an EMPTY table INTENTIONALLY degrade to a single static program
        # at the slots' reference values (compile_pulse_table_runtime_program resolves them)
        # -- a run is never blocked just because the table has not been filled yet.  (A
        # DIRECT compile_scan call still errors on an empty table, which is the right strict
        # behavior for that explicit "I am scanning" entry point.)
        if payload.scan_slots and payload.scan_table:
            return compile_pulse_table_scan_runtime_program(
                payload,
                clock_hz=clock_hz,
                scan_table=payload.scan_table,
                repeat_forever=payload.repeat_forever,
                port_catalog=port_catalog,
            )
        return compile_pulse_table_runtime_program(
            payload,
            clock_hz=clock_hz,
            repeat_forever=payload.repeat_forever,
            port_catalog=port_catalog,
        )
    if channels is None:
        if port_catalog is None:
            raise ValueError("compiling a PulseSequence requires a PortCatalog")
        catalog = (
            port_catalog if isinstance(port_catalog, PortCatalog)
            else PortCatalog.from_dict(port_catalog)
        )
        channels = catalog.raw_lanes
    return compile_runtime_program(
        payload, channels=channels, clock_hz=clock_hz, port_catalog=port_catalog)

def timing_payload_to_dict(payload: PulseSequence | PulseTableState, *, time_step_ns: float | None = None) -> dict[str, object]:
    """Return the JSON-safe timing payload for a sequence or pulse table.

    A ``PulseTableState`` is SNAPPED to the clock-tick grid before serialization, so
    the pulse transferred to the server/hardware carries the same whole-tick values
    the GUI displays and the compiler would land on -- there is no place where an
    off-grid value silently slips through the pulse-transfer API.

    ``time_step_ns`` MUST be the TARGET sequencer's tick (``1e9 / clock_hz``) so the
    snap lands on the grid the SERVER will compile at -- otherwise a state saved at a
    different ``time_step_ns`` would be pre-snapped on the wrong grid and the Remote/Virtual
    result would diverge from a direct local compile at the same clock.  Defaults to the
    payload's own ``time_step_ns`` only when no target is supplied."""

    if isinstance(payload, PulseTableState):
        return payload.snapped(time_step_ns=time_step_ns).to_dict()
    if isinstance(payload, PulseSequence):
        return payload.to_dict()
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError("timing payload must be a PulseSequence, PulseTableState, or mapping.")

def encode_wire_payload(data: Mapping[str, object], array_fields: Sequence[str]) -> tuple[str, tuple]:
    """Split only NON-EMPTY ``array_fields`` out of a JSON-safe timing dict into raw ndarray buffers,
    JSON the rest.

    Returns ``(head_json, blobs)`` where ``blobs`` is a TUPLE of ``(field, raw_bytes, shape, dtype)``
    -- a tuple (not a dict) so RPyC transfers it BY VALUE (brine serialises tuple/bytes/str/int; a
    dict would netref).  An empty or absent array field STAYS in the JSON head (as ``[]``), so the
    decoded payload is byte-for-byte the all-JSON payload -- a no-scan program/table still carries its
    ``scan_points`` / ``scan_point_durations`` / ``scan_table`` keys and passes the strict exact-field
    decoder, instead of being silently dropped (only large non-empty arrays take the binary fast path)."""
    array_field_set = set(array_fields)
    head: dict[str, object] = {}
    blobs = []
    for key, value in data.items():
        if key in array_field_set and value is not None and hasattr(value, "__len__") and len(value) != 0:
            arr = np.asarray(value)
            blobs.append((key, arr.tobytes(), tuple(int(s) for s in arr.shape), arr.dtype.str))
        else:
            head[key] = value
    return json.dumps(head), tuple(blobs)

def decode_wire_payload(head_json: str, blobs: Sequence[tuple]) -> dict:
    """Inverse of :func:`encode_wire_payload`: JSON-load the head and rebuild each array field from
    its buffer as a NATIVE Python list -- identical to the all-JSON payload, so downstream compile is
    unchanged."""
    out = json.loads(head_json)
    for field, buf, shape, dtype in (blobs or ()):
        out[field] = np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(tuple(shape)).tolist()
    return out

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
        schema = data.get("schema", PulseSequence.schema)
        if schema == PulseTableState.schema:
            return PulseTableState.from_dict(data)
        if schema == PulseSequence.schema:
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
    """Affine form of ``PulseTableState.period_start_steps`` (the same period-start prefix
    sum): each entry is ``(base_tick, slot_coeffs)`` rather than a scalar tick, so the period
    boundaries scan affinely in the bound slots.  Iterates ``state.periods`` in the same order;
    a scalar evaluation of these coeffs at any slot point equals ``period_start_steps`` there."""
    starts = [(0, tuple(0 for _ in slot_vars))]
    for period in state.periods:
        starts.append(_affine_add(starts[-1], _affine_expr(period.duration, period.unit, slot_vars, time_step_ns, coeff_frac_bits)))
    return starts

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
    a delay is a per-channel OUTPUT delay (``channel_delays``, the FPGA event scheduler),
    never baked into the edges.  Because every edge sits on a monotone period boundary, the merged edge
    list is globally tick-monotone at every scan point automatically -- no channel reorders,
    so no global shift G is needed.  ``_stable_affine_groups`` still
    validates per-channel + cross-channel ordering at every scan point as a safety net."""

    hardware_channels = list(channel_names(channels, "channels"))
    exclude = set(exclude_channels)
    starts = _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
    events: list[tuple[tuple[int, tuple[int, ...]], str | None, int | None]] = []

    for channel_index, channel in enumerate(state.port_catalog.raw_lanes):
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
                "scan slots sweep (the channels reorder), which the single global edge "
                "table cannot play.  Narrow the scan range so every channel stays in its "
                "own slot relative to the others, OR scan a DAC value instead (analog "
                "buses are independent timelines and may reorder freely)."
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
    # Per-scan-point monotonicity, VECTORISED.  Every edge tick is the SAME pure-integer affine
    # MAC ``base + (coeffs . slot_ticks) >> frac`` the scalar ``_apply_affine_ticks`` computes, so
    # one int64 matmul over all points x rows yields tick[N, G] BIT-IDENTICAL to the per-point loop
    # (int64 has ample headroom; numpy ``>>`` on signed ints is the same arithmetic shift toward
    # -inf as Python ``>>``, so a negative coefficient sum floors identically).  This validation
    # EMITS NOTHING -- it only raises on an unrepresentable scan -- so the compiled image is
    # unchanged; it just replaces the O(N x rows) Python loop that dominated large-scan compile.
    pts = np.asarray(scan_points, dtype=np.int64)                                  # (N, S)
    bases = np.array([int(expr[0]) for expr, _items in grouped], dtype=np.int64)   # (G,)
    coeffs = np.array([[int(c) for c in expr[1]] for expr, _items in grouped], dtype=np.int64)  # (G, S)
    ticks = bases[None, :] + ((pts @ coeffs.T) >> int(coeff_frac_bits))            # (N, G)
    # 1) Every row's tick is non-negative at every point (mirror of the scalar ``tick < 0`` guard,
    #    raising at the FIRST offending point in scan order).
    neg_rows = (ticks < 0).any(axis=1)
    if neg_rows.any():
        raise ValueError(
            f"hardware scan produced a negative edge tick at scan point {int(np.argmax(neg_rows))}.")
    # 2) Per-channel: each channel's OWN edges (its ordered subsequence of grouped rows) must be
    #    strictly increasing at every point -- a channel reversing/colliding its own edges is
    #    unrepresentable.  NECESSARY but not sufficient; the global merge is re-checked below and
    #    downstream by ``validate_pulse_streamer_program``.
    chan_rows: dict[str, list[int]] = {}
    for row_index, (_expr, items) in enumerate(grouped):
        for channel, _value in items:
            if channel is not None:
                chan_rows.setdefault(channel, []).append(row_index)
    for channel, row_indices in chan_rows.items():
        if len(row_indices) < 2:
            continue
        seq = ticks[:, row_indices]                                                # (N, k), grouped order
        bad = (seq[:, 1:] <= seq[:, :-1]).any(axis=1)
        if bad.any():
            raise ValueError(
                f"hardware scan reverses or collides channel '{channel}'s own edges at "
                f"scan point {int(np.argmax(bad))}; simplify that channel's delay/duration scan "
                "(a single channel cannot run its own pulses backwards)."
            )
    # 3) CROSS-CHANNEL reorder/collision at ANY scan point: the single global sorted table needs
    #    the real-edge rows STRICTLY increasing at every point.  Two rows that both carry a channel
    #    transition swapping/colliding as the slots sweep means the channels reorder (the
    #    reference-only grouping above misses a reorder that only appears at a later point).
    edge_indices = [row_index for row_index, (_e, items) in enumerate(grouped)
                    if any(ch is not None for ch, _v in items)]
    if len(edge_indices) >= 2:
        seq = ticks[:, edge_indices]                                               # (N, E), grouped order
        if bool((seq[:, 1:] <= seq[:, :-1]).any()):
            raise ValueError(
                "this scan moves one channel's edges PAST another channel's edges as the "
                "scan slots sweep (the channels reorder), which the single global edge "
                "table cannot play.  Narrow the scan range so every channel stays in "
                "its own slot relative to the others."
            )
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
    starts = state.period_start_steps(slots=slots, time_step_ns=time_step_ns)
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
    extra_raw_delays: Mapping[int, int] | None = None,
) -> tuple[list[int], list[int], list[str], int, int, dict[int, int], list]:
    """Build ``(ticks, masks, channels, loop_end, repeat_from_index, channel_delays,
    bus_delays)`` -- 7 values (the annotation matches the return exactly).

    The edge table is UNDELAYED: every channel sits at its nominal position and the loop
    period is the plain frame end ``table_end`` (``repeat_from_index`` always 0).  A channel
    delay is NOT baked into the ticks -- it is applied to the engine OUTPUT by a per-channel
    delay line (output_delayed[t] = output_undelayed[t-d], zero before fire).  This is the
    literal physical delay: ANY length, never disturbs another channel, first frame real.
    ``channel_delays`` maps output-bit -> delay in ticks (only nonzero entries).

    A NEGATIVE delay re-translates the WHOLE frame, so the global shift ``G = max(0, -min
    delay)`` is FOLDED INTO every channel's delay (a causal delay line cannot lead): every
    returned delay is ``raw_delay + G >= 0``, preserving relative timing.

    ``extra_raw_delays`` (bus_index -> raw delay in ticks) are DAC buses emitted as bus
    SEGMENTS, not folded into the TTL mask.  They share the SAME global shift G (so a
    negative bus delay also lands >= 0) and are returned shifted as ``bus_delays``
    (bus_index -> delay) for the LITERAL per-bus delay line."""
    hardware_channels = list(channel_names(channels, "channels"))
    starts = state.period_start_steps(slots=slots, time_step_ns=time_step_ns)
    table_end = int(starts[-1])
    channel_bits = {channel: index for index, channel in enumerate(hardware_channels)}
    bus_groups = state.bus_channels()
    bus_members = {channel for members in bus_groups.values() for channel in members}

    # --- per-channel UN-delayed ON intervals over [0, T) + each channel's raw delay ---
    base_intervals: dict[str, list[tuple[int, int]]] = {}
    raw_delay: dict[str, int] = {}
    clk_set = {port.lanes[0] for port in state.port_catalog.clock_ports}
    for channel_index, channel in enumerate(state.port_catalog.raw_lanes):
        # clk channels are driven by the top's clk mux, not the engine -> no edges, no delay.
        if channel in bus_members or channel not in channel_bits or channel in clk_set:
            continue
        ivals, active = [], None
        for period_index, period in enumerate(state.periods):
            if int(period.states[channel_index]) and active is None:
                active = int(starts[period_index])
            elif not int(period.states[channel_index]) and active is not None:
                ivals.append((active, int(starts[period_index]))); active = None
        if active is not None:
            ivals.append((active, table_end))
        # An OFF channel (no ON interval) emits nothing; its (possibly negative) delay must NOT
        # enter raw_delay -- otherwise it would shift the global frame G and delay other ACTIVE
        # channels for no physical reason.  Only delay channels that actually output.
        if not ivals:
            continue
        raw_delay[channel] = state.delay_steps(channel, slots=slots, time_step_ns=time_step_ns)
        base_intervals[channel] = ivals
    if fold_analog_buses:
        for bus_name, members in bus_groups.items():
            bus_delay = _pulse_table_bus_delay_steps(state, members, slots=slots, time_step_ns=time_step_ns)
            plan = state.analog_bus_plan(bus_name)
            # An UNTOUCHED bus (all-hold plan, resting at 0 V) folds NOTHING -- the
            # hardware idles at the mid code by itself (BUS_SAFE_VALUE); folding it
            # would put phantom mid-code bits into the masks.
            if all(str(entry.get("mode", "hold")).lower() == "hold" for entry in plan):
                continue
            # plan values are SIGNED (0 = 0 V); the folded member bits carry the
            # offset-binary CODE = signed + zero_code (the ONE mid-scale helper, not re-typed).
            fold_zero_code = bus_zero_code(len(members))
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
                    on = ((int(_pulse_table_analog_bus_value_at_tick(plan, starts, tick)) + fold_zero_code) >> bit) & 1
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
    extra_raw_delays = dict(extra_raw_delays or {})
    channel_delays, bus_delays_shifted, _global_shift = _fold_global_delay_shift(
        raw_delay, extra_raw_delays)

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
    return ticks, masks, hardware_channels, loop_end, repeat_from_index, channel_delays_by_bit, bus_delays_shifted

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
    # Shared "sN" parser (the single slot-reference spelling, owned by the timing layer).
    index = _parse_slot_ref_index(text)
    if index is not None and 0 <= index < len(slot_vars):
        return index
    return None

def _pulse_table_bus_segments(
    state: PulseTableState,
    *,
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
    slot_vars: Sequence[str] | None = None,
    coeff_frac_bits: int = DEFAULT_COEFF_FRAC_BITS,
) -> tuple[list[str], list[RuntimeBusSegment], dict[int, int]]:
    """Compile logical analog buses into hardware bus segments.

    A ramp consumes one segment regardless of how many 10-bit stair steps it
    produces.  Digital edge rows are left for ordinary TTL outputs.  When a bus
    value references a scan slot (``slot_vars`` given), the segment carries a
    ``value_select`` so the DAC level is read from that slot per scan point.

    With ``slot_vars`` the segment *ticks* are emitted as affine expressions
    (base + per-slot coefficients), exactly like the digital edges, so a scanned
    DURATION moves the analog segment in lockstep -- this is what lets DAC value +
    duration scan simultaneously.

    The per-bus DELAY is NOT baked into the segment ticks (that capped it at one
    frame).  Segments are emitted at their NOMINAL phase and the bus delay is
    returned as ``{bus_index: delay_steps}`` (third element), realised by the SAME
    per-signal event scheduler as the TTL channels -- so a DAC value can be delayed by
    more than one frame, exactly like a TTL channel.
    """

    slot_vars = list(slot_vars or [])
    affine = bool(slot_vars)
    zero_coeffs = tuple(0 for _ in slot_vars)
    starts = state.period_start_steps(slots=slots, time_step_ns=time_step_ns)
    table_end = int(starts[-1])
    affine_starts = (
        _pulse_table_affine_period_starts(state, slot_vars=slot_vars, time_step_ns=time_step_ns, coeff_frac_bits=coeff_frac_bits)
        if affine
        else None
    )
    bus_groups = state.bus_channels()
    # Physical bus order/index belongs to PortCatalog.  Never sort by a
    # hand-written tuple or template dict insertion order: either would silently
    # route a logical DAC to the wrong zlc_bus_out slice.
    bus_specs = [port for port in state.port_catalog.dac_ports if port.key in bus_groups]
    bus_names = [port.key for port in bus_specs]
    segments: list[RuntimeBusSegment] = []
    bus_delays: dict[int, int] = {}
    for bus_spec in bus_specs:
        bus_index = int(bus_spec.bus_index)
        bus_name = bus_spec.key
        members = bus_groups[bus_name]
        plan = state.analog_bus_plan(bus_name)
        # A bus delay is NOT baked into the segment ticks (that capped it at one frame).
        # Segments are emitted at their NOMINAL (undelayed) phase; the per-bus delay is
        # returned separately and realised by the engine's per-signal EVENT SCHEDULER (each DA
        # bit queues its value-changes against the global tick counter; out[t]=in[t-d]).
        # This is the SAME mechanism as the TTL channels, so a DAC value can be delayed by more
        # than one frame (storage scales with events in flight, not with the delay length).
        delay_steps = _pulse_table_bus_delay_steps(state, members, slots=slots, time_step_ns=time_step_ns)
        if delay_steps:
            bus_delays[bus_index] = int(delay_steps)
        max_value = (1 << len(members)) - 1

        def _coeffs(values: tuple[int, ...]) -> list[int] | None:
            return list(values) if affine else None

        def _boundary(boundary_index: int) -> tuple[int, int, tuple[int, ...]]:
            """(ref_tick, base_tick, coeffs) for period boundary i in [0, n_periods]."""
            ref = int(starts[boundary_index])
            if affine:
                base, coeffs = affine_starts[boundary_index]
                return ref, int(base), tuple(coeffs)
            return ref, ref, zero_coeffs

        # Forward-propagate the DAC value through the periods, so each mode controls the
        # CURRENT period:
        #   edge v -> step to v at the period start and HOLD v (a point segment);
        #   ramp v -> ramp linearly from the value carried INTO the period to v by the
        #             period END (a [start_i, start_{i+1}) segment); the RTL/engine
        #             interpolate within that window and hold v afterwards;
        #   hold   -> emit NO segment (the engine keeps the carried value).
        # The value entering each period is the engine's LIVE register (carried across periods and
        # across the frame/loop/scan wrap), so a ramp's START is NOT computed here -- the ramp engine
        # reads its current value at runtime (#ramp-carry).  This is the one model correct for
        # fire-once (ramp from idle), loop (converge to flat) AND scan (staircase from the prior point).
        # WIRE CONVERSION: plan values are SIGNED (0 = true 0 V); segments carry the offset-binary
        # CODE = signed + zero_code.  The hardware idles at zero_code (BUS_SAFE_VALUE).
        zero_code = bus_zero_code(len(members))          # the ONE mid-scale helper (not re-typed)

        def _emit():
            """Walk this bus's periods, emitting one segment per edge/ramp period (hold emits none).

            The DAC value is CARRIED by the engine/hardware value register across periods AND across the
            frame / loop / scan-point boundary (#ramp-carry).  A ramp therefore ALWAYS ramps from the
            CURRENT register value -- the end of the previous period within a frame, or the previous
            frame / scan-point across a wrap -- to its target, so the segment bakes NO start endpoint
            (start_value / start_value_select = 0, ignored by the RTL ramp engine and the host mirror
            alike).  An edge sets the register to its value; a hold persists it.  The compiler thus
            never has to GUESS the carry: the first fire ramps from idle 0 V, a looping [ramp V, hold V]
            converges to FLAT V (the wrap carries V in), and a SCANNED ramp staircases from the previous
            scan point's value -- the one model that is correct for fire-once, loop AND scan."""
            bus_segs: list[RuntimeBusSegment] = []
            for period_index in range(len(state.periods)):
                entry = plan[period_index] if period_index < len(plan) else {"mode": "hold", "value": None}
                mode = str(entry.get("mode", "hold")).strip().lower()
                value = entry.get("value")
                if mode not in {"edge", "ramp"} or value is None:
                    continue  # hold -> carried value persists in the engine
                ref_index = _slot_ref_index(value, slot_vars)
                if ref_index is not None:
                    value_select = ref_index + 1
                    value_int = 0  # placeholder; the FPGA reads the (code) slot at runtime
                else:
                    value_select = 0
                    # On the STATIC path (slot_vars empty -- e.g. the payload dispatcher degrades a
                    # bound-but-unfilled DAC scan to a static program), a value may still be a slot
                    # ref like "s0".  Resolve it from the provided slots (reference values) instead
                    # of int("s0")-crashing.
                    slot_idx = _parse_slot_ref_index(value)
                    if slot_idx is not None:
                        key = str(value).strip()
                        if slots is None or key not in slots:
                            raise ValueError(
                                f"analog bus {bus_name!r} references unresolved scan slot {key!r}; "
                                "provide its value or a scan table.")
                        value = slots[key]
                    # signed user value -> offset-binary wire code
                    value_int = max(0, min(max_value, int(round(float(value))) + zero_code))
                start_ref, start_base, start_coeffs = _boundary(period_index)
                if start_ref < 0 or start_ref > table_end:
                    raise ValueError(f"analog bus {bus_name!r} produced segment tick {start_ref} outside the uploaded table.")
                if mode == "edge":
                    # Edge: step to value at the period start and HOLD it.  Always emitted -- with the
                    # carried register an edge to the currently-held value is harmlessly redundant, but
                    # under a loop / scan the held value differs per frame / point, so an edge must NOT
                    # be STATICALLY dropped (e.g. a period-0 edge-to-0 that re-zeroes the bus each loop
                    # is exactly what stops the previous frame's value carrying in) (#ramp-carry).
                    bus_segs.append(RuntimeBusSegment(
                        bus_index, start_base, start_base, value_int, value_int, "edge", bus_name,
                        value_select, _coeffs(start_coeffs), _coeffs(start_coeffs),
                        stop_value_select=value_select,
                    ))
                else:  # ramp from the CURRENT register value -> value, over [start_i, start_{i+1})
                    end_ref, end_base, end_coeffs = _boundary(period_index + 1)
                    if end_ref < start_ref:
                        raise ValueError(f"analog bus {bus_name!r} ramp end precedes its start.")
                    # The ramp START is the LIVE register (carry) -- NOT baked: start_value=0,
                    # start_value_select=0, ignored by the ramp engine, which reads its current value.
                    bus_segs.append(RuntimeBusSegment(
                        bus_index, start_base, end_base, 0, value_int, "ramp", bus_name,
                        0, _coeffs(start_coeffs), _coeffs(end_coeffs),
                        stop_value_select=value_select,
                    ))
            return bus_segs

        segments.extend(_emit())
    return bus_names, segments, bus_delays

def _pulse_table_has_analog_activity(state: PulseTableState) -> bool:
    """Whether any catalog DAC has an edge/ramp to emit through the bus engine."""

    return any(
        any(str(entry.get("mode", "hold")).lower() in {"edge", "ramp"}
            for entry in state.analog_bus_plan(port.key))
        for port in state.port_catalog.dac_ports
    )

def _pulse_table_has_any_delay(state: PulseTableState) -> bool:
    """True if any channel has a delay that is nonzero OR scanned (a slot expression).

    The scan path must UNROLL a bracket when ANY channel carries a delay -- including a
    SCANNED delay whose reference value is 0 (so :func:`_pulse_table_has_delays`, which
    resolves the reference slot, would miss it).  A delay bound to a slot is always
    treated as present; a literal delay counts only when it rounds to a nonzero tick.
    """

    for channel in state.port_catalog.raw_lanes:
        raw = state.delays.get(channel, 0.0)
        if isinstance(raw, str) and not _is_plain_number(raw):
            return True  # scanned / expression delay (e.g. "s0", "20+s1")
        if state.delay_steps(channel, slots=state._reference_slots(), time_step_ns=state.time_step_ns) != 0:
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
    upper_bound_edges = 2 * len(state.port_catalog.raw_lanes) * n_periods + 2
    if upper_bound_edges > DEFAULT_MAX_EDGES:
        raise ValueError(
            f"unrolling the inner repeat bracket would make up to {upper_bound_edges} edges, "
            f"above the FPGA streamer budget of {DEFAULT_MAX_EDGES}.  Use repeat_forever for the "
            "OUTER loop, fewer inner iterations, or remove the channel delay so the bracket can "
            "stay a compact hardware loop instead of being unrolled."
        )

def _pulse_table_has_delays(
    state: PulseTableState,
    *,
    slots: Mapping[str, float] | None = None,
    time_step_ns: float,
) -> bool:
    return any(
        state.delay_steps(channel, slots=slots, time_step_ns=time_step_ns) != 0
        for channel in state.port_catalog.raw_lanes
    )

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
