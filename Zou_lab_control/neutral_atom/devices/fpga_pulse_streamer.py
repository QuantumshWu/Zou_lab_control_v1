"""FPGA host helpers: board-XDC channel/trigger inference + program validation.

The FINAL pulse streamer is driven over JTAG-to-AXI by
``axi_session.VivadoAxiStreamerSession`` (which packs the BRAM image from
``fpga.pulse_streamer.host.image`` and uploads it).  This module keeps only the
host-side helpers shared by the server + GUI launcher: inferring channel names,
labels, pins and the camera trigger from a board XDC, and validating a compiled
``RuntimeSequenceProgram`` before upload.  (The old VIO/Vivado-Tcl HDL generator +
session were removed with the rest of the legacy control path.)
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from typing import Mapping, Sequence

from .sequencer import RuntimeSequenceProgram
from ..timing import channel_names
from ..timing.verilog import CONTROL_PORTS, safe_identifier


# The reconfigurable, compile-affecting specifics (channel/edge/scan/bus/delay geometry,
# widths, FPGA part) come from the SINGLE user-editable config file
# fpga/board_config/streamer_config.json, loaded through host.image -- so a board or
# geometry change is edited in ONE place and flows to validation + capacity together.
# These module constants MIRROR that config; the literal fallback (kept identical to the
# shipped config) keeps this module importable if the fpga package is unavailable.
# NOTE: max_edges/bank_size/evt_fifo_depth etc. are baked into the synthesized bitstream
# (zlc_pulse_streamer_top.v localparams) -- changing the JSON does NOT re-synthesize; it
# only re-aligns host validation/estimation.  Double-click estimate_resources.bat after
# editing to check the part still fits, and rebuild the RTL to actually change the geometry.
try:  # pragma: no cover - exercised whenever the fpga package is importable (the norm)
    from fpga.pulse_streamer.host.image import load_streamer_config as _load_streamer_config

    _STREAMER_CFG = _load_streamer_config()
    _CFG_PARAMS = _STREAMER_CFG["params"]
    DEFAULT_FPGA_CHANNEL_COUNT = int(_CFG_PARAMS.channel_count)
    DEFAULT_MAX_EDGES = int(_CFG_PARAMS.max_edges)
    DEFAULT_MAX_SCAN_POINTS = int(2 * _CFG_PARAMS.bank_size)   # resident 2-bank window; more stream
    DEFAULT_MAX_BUS_SEGMENTS = int(_CFG_PARAMS.max_bus_segments)
    DEFAULT_TICK_WIDTH = int(_CFG_PARAMS.tick_width)
    DEFAULT_SCAN_COEFF_WIDTH = int(_CFG_PARAMS.coeff_width)
    DEFAULT_SCAN_COEFF_FRAC_BITS = int(_CFG_PARAMS.coeff_frac_bits)
    DEFAULT_NUM_SLOTS = int(_CFG_PARAMS.num_slots)
    TTL_DELAY_MAX_TICKS = int(getattr(_CFG_PARAMS, "ttl_delay_max_ticks", (1 << 31) - 1))
    EVT_FIFO_DEPTH = int(getattr(_CFG_PARAMS, "evt_fifo_depth", 16))
    BUS_EVT_FIFO_DEPTH = int(getattr(_CFG_PARAMS, "bus_evt_fifo_depth", 64))
    DEFAULT_SLOT_MUL_WIDTH = int(_STREAMER_CFG["slot_mul_width"])
    DEFAULT_BUS_COUNT = int(_CFG_PARAMS.bus_count)
    DEFAULT_BUS_WIDTH = int(_CFG_PARAMS.bus_width)
    DEFAULT_FPGA_PART = str(_STREAMER_CFG["fpga_part"])
except Exception:  # pragma: no cover - fpga package not importable; use shipped-config literals
    DEFAULT_FPGA_CHANNEL_COUNT = 62
    DEFAULT_MAX_EDGES = 4096
    DEFAULT_MAX_SCAN_POINTS = 4096        # 2 * bank_size (2048)
    DEFAULT_MAX_BUS_SEGMENTS = 64
    DEFAULT_TICK_WIDTH = 32
    DEFAULT_SCAN_COEFF_WIDTH = 16
    DEFAULT_SCAN_COEFF_FRAC_BITS = 8
    DEFAULT_NUM_SLOTS = 4
    TTL_DELAY_MAX_TICKS = (1 << 31) - 1
    EVT_FIFO_DEPTH = 16
    BUS_EVT_FIFO_DEPTH = 64
    # Affine-MAC slot operand width -- MUST match zlc_edge_streamer.v SLOT_MUL_WIDTH and
    # engine_model.SLOT_MUL_WIDTH.  Each scan slot VALUE x a 16-bit coeff fits one DSP48E1
    # (25x18), so the slot operand is the low 25 bits as signed; the validator rejects a
    # scan value outside +/-2^24 ticks (the coeff still scales it, so the tick OFFSET keeps
    # the full 32-bit range).
    DEFAULT_SLOT_MUL_WIDTH = 25
    DEFAULT_BUS_COUNT = 4
    DEFAULT_BUS_WIDTH = 10
    DEFAULT_FPGA_PART = "xc7a35tfgg484-2"
DEFAULT_CHANNELS = [f"ch{index:02d}" for index in range(DEFAULT_FPGA_CHANNEL_COUNT)]
# The scan/edge BRAMs are forced to READ_LATENCY_B = 2 (create_project.tcl zlc_force_latency2),
# and the engine reads the NEXT scan point's slot vector during the CURRENT frame.  A scanned
# frame shorter than this many ticks would play the next point with the previous point's slot.
SCAN_READ_LATENCY = 2


def hardware_channel_names(count: int = DEFAULT_FPGA_CHANNEL_COUNT) -> list[str]:
    """Return hardware channel names in FPGA bit order."""

    count = _positive_int(count, "count")
    return [f"ch{index:02d}" for index in range(count)]


def _xdc_ports(text: str) -> list[str]:
    ports: list[str] = []
    pattern = re.compile(r"get_ports\s+(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_\[\]]*))\s*\]")
    for match in pattern.finditer(text):
        port = (match.group(1) or match.group(2) or "").strip()
        if port:
            ports.append(port)
    return ports


def _xdc_output_port_labels(text: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for port in _xdc_ports(text):
        if re.fullmatch(r"ch\[\d+\]", port):
            continue
        if port == "clk" or port.startswith("led[") or re.fullmatch(r"GND\d*", port, re.IGNORECASE):
            continue
        if port in CONTROL_PORTS or port in {"zlc_running_led", "zlc_done_led"}:
            continue
        if port in seen:
            continue
        seen.add(port)
        labels.append(port)
    return labels


def infer_xdc_channel_count(
    xdc_path: str | Path | None = None,
    *,
    default: int = DEFAULT_FPGA_CHANNEL_COUNT,
    max_count: int | None = DEFAULT_FPGA_CHANNEL_COUNT,
) -> int:
    """Infer the full FPGA output width from ``get_ports {ch[n]}`` constraints.

    A completed board XDC is the most concrete source of the physical output
    contract.  If no XDC is present, the caller gets ``default`` so GUI/offline
    workflows still open cleanly.
    """

    default = _positive_int(default, "default")
    path = _resolve_xdc_path(xdc_path)
    if path is None or not path.exists():
        return default
    text = path.read_text(encoding="utf-8", errors="replace")
    indices = [int(match.group(1)) for match in re.finditer(r"get_ports\s+\{?ch\[(\d+)\]\}?", text)]
    if indices:
        count = max(indices) + 1
        missing = sorted(set(range(count)) - set(indices))
        if missing:
            raise ValueError(f"XDC channel constraints are not contiguous; missing ch indices: {missing}.")
    else:
        count = len(_xdc_output_port_labels(text))
        if count <= 0:
            return default
    if max_count is not None and count > int(max_count):
        raise ValueError(f"XDC defines {count} channels, above max_count={int(max_count)}.")
    return count


def infer_xdc_channels(
    xdc_path: str | Path | None = None,
    *,
    default: int = DEFAULT_FPGA_CHANNEL_COUNT,
    max_count: int | None = DEFAULT_FPGA_CHANNEL_COUNT,
) -> list[str]:
    """Return ``ch00..`` names inferred from a completed pulse-streamer XDC."""

    return hardware_channel_names(infer_xdc_channel_count(xdc_path, default=default, max_count=max_count))


def infer_xdc_channel_labels(
    xdc_path: str | Path | None = None,
    *,
    default: int = DEFAULT_FPGA_CHANNEL_COUNT,
    max_count: int | None = DEFAULT_FPGA_CHANNEL_COUNT,
) -> dict[str, str]:
    """Return display labels parsed from address-switch XDC comments.

    The hardware channel names stay ``ch00..`` in FPGA bit order.  Labels are
    only front-end names, parsed from comments such as ``;# ch04 <- cooling_pgc``.
    Existing pulse JSON labels should take priority over this default map.
    """

    channels = infer_xdc_channels(xdc_path, default=default, max_count=max_count)
    labels = {channel: channel for channel in channels}
    path = _resolve_xdc_path(xdc_path)
    if path is None or not path.exists():
        return labels
    text = path.read_text(encoding="utf-8", errors="replace")
    ch_style = False
    for line in text.splitlines():
        port_match = re.search(r"get_ports\s+\{?ch\[(\d+)\]\}?", line)
        if not port_match:
            continue
        ch_style = True
        channel = f"ch{int(port_match.group(1)):02d}"
        if channel not in labels:
            continue
        comment = ""
        if ";#" in line:
            comment = line.split(";#", 1)[1]
        elif "#" in line:
            comment = line.split("#", 1)[1]
        label = _label_from_xdc_comment(comment, channel)
        if label:
            labels[channel] = label
    if ch_style:
        return labels
    for index, label in enumerate(_xdc_output_port_labels(text)):
        channel = f"ch{index:02d}"
        if channel in labels:
            labels[channel] = label
    return labels


def infer_xdc_channel_pins(
    xdc_path: str | Path | None = None,
    *,
    default: int = DEFAULT_FPGA_CHANNEL_COUNT,
    max_count: int | None = DEFAULT_FPGA_CHANNEL_COUNT,
) -> dict[str, str]:
    """Return FPGA package pins for inferred ``chNN`` hardware channels."""

    channels = infer_xdc_channels(xdc_path, default=default, max_count=max_count)
    pins = {channel: channel for channel in channels}
    path = _resolve_xdc_path(xdc_path)
    if path is None or not path.exists():
        return pins
    text = path.read_text(encoding="utf-8", errors="replace")
    labels = infer_xdc_channel_labels(xdc_path, default=default, max_count=max_count)
    label_to_channel = {label: channel for channel, label in labels.items()}
    for line in text.splitlines():
        pin_match = re.search(r"PACKAGE_PIN\s+([A-Za-z0-9_]+)", line)
        ports = _xdc_ports(line)
        if not pin_match or not ports:
            continue
        pin = pin_match.group(1)
        port = ports[0]
        channel = None
        ch_match = re.fullmatch(r"ch\[(\d+)\]", port)
        if ch_match:
            channel = f"ch{int(ch_match.group(1)):02d}"
        else:
            channel = label_to_channel.get(port)
        if channel in pins:
            pins[channel] = pin
    return pins


def infer_xdc_trigger_channels(
    xdc_path: str | Path | None = None,
    *,
    default: int = DEFAULT_FPGA_CHANNEL_COUNT,
    max_count: int | None = None,
    preferred_labels: Sequence[str] = ("emccd",),
) -> list[str]:
    """Infer hardware trigger channels from XDC display labels.

    The returned names are hardware bit names such as ``ch11``.  The default
    camera trigger is the physical ``emCCD`` output; labels such as ``trig``
    remain ordinary output names and are not selected implicitly.
    """

    labels = infer_xdc_channel_labels(xdc_path, default=default, max_count=max_count)
    preferred = [str(label).strip().lower() for label in preferred_labels if str(label).strip()]
    for target in preferred:
        for channel, label in labels.items():
            if str(label).strip().lower() == target:
                return [channel]
    return []


def validate_pulse_streamer_program(
    program: RuntimeSequenceProgram,
    *,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_scan_points: int = DEFAULT_MAX_SCAN_POINTS,
    max_bus_segments: int = DEFAULT_MAX_BUS_SEGMENTS,
    tick_width: int = DEFAULT_TICK_WIDTH,
    channel_count: int | None = None,
    coeff_width: int = DEFAULT_SCAN_COEFF_WIDTH,
    num_slots: int = DEFAULT_NUM_SLOTS,
    bus_count: int = DEFAULT_BUS_COUNT,
    bus_width: int = DEFAULT_BUS_WIDTH,
    slot_mul_width: int = DEFAULT_SLOT_MUL_WIDTH,
    max_validated_scan_points: int | None = None,
) -> None:
    """Validate that a runtime edge table fits the fixed FPGA streamer.

    ``max_validated_scan_points`` bounds the O(points*edges) effective-tick
    monotonicity sweep for a large STREAMED scan: when there are more points than
    the cap, only a representative subset (an even stride plus the per-slot extreme
    points, which the affine ticks make the most likely to reorder) is checked, so
    the validator never hangs on a million-point sweep.  ``None`` checks every point
    (shape/value checks always run for every point)."""

    max_edges = _positive_int(max_edges, "max_edges")
    max_scan_points = _positive_int(max_scan_points, "max_scan_points")
    max_bus_segments = _positive_int(max_bus_segments, "max_bus_segments")
    tick_width = _positive_int(tick_width, "tick_width")
    coeff_width = _positive_int(coeff_width, "coeff_width")
    num_slots = _positive_int(num_slots, "num_slots")
    bus_count = _positive_int(bus_count, "bus_count")
    bus_width = _positive_int(bus_width, "bus_width")
    channel_count = len(program.channels) if channel_count is None else _positive_int(channel_count, "channel_count")
    if len(set(program.channels)) != len(program.channels):
        raise ValueError("program channels must be unique.")
    if len(program.ticks) != len(program.masks):
        raise ValueError("program ticks and masks must have the same length.")
    if len(program.ticks) > max_edges:
        raise ValueError(f"program has {len(program.ticks)} edges, but the FPGA streamer only accepts {max_edges}.")
    if len(program.channels) > channel_count:
        raise ValueError(f"program uses {len(program.channels)} channels, but the FPGA streamer has {channel_count}.")
    # PER-CHANNEL TTL OUTPUT DELAY -- the EVENT SCHEDULER.  A delay is constant (never
    # scanned), bounded only by its 32-bit field (TTL_DELAY_MAX_TICKS ~ 42.9 s) -- e.g.
    # millisecond emCCD delays.  The hardware constraint moves from delay LENGTH to
    # toggles IN FLIGHT: a channel may have at most EVT_FIFO_DEPTH toggles inside any
    # window of its own delay length (each in-flight toggle holds one event-FIFO slot).
    channel_delays = [int(d) for d in (getattr(program, "channel_delays", None) or [])]
    for b, d in enumerate(channel_delays):
        if d < 0 or d > TTL_DELAY_MAX_TICKS:
            raise ValueError(
                f"channel-delay output bit {b}: delay {d} ticks is outside "
                f"[0, {TTL_DELAY_MAX_TICKS}] (~{TTL_DELAY_MAX_TICKS * 20e-9:.1f} s at 20 ns/tick).")
    # (Delay-ELIGIBILITY -- only the real TTL outputs have an event FIFO, not the bus
    # bits / da_clk pins -- is enforced user-facing at the GUI/API level by HARDWARE
    # position; the RTL also gates a stray non-eligible delay to a passthrough.  It is
    # not re-checked here: this validator sees only the program's channel SUBSET, whose
    # index does not map to the hardware position.)
    # (the event-FIFO capacity analysis runs at the END of this validator, after
    # the tick/loop/scan geometry checks -- it assumes validated geometry, e.g.
    # loop_end > loop_start at every scan point)
    tick_limit = (1 << tick_width) - 1
    mask_limit = (1 << channel_count) - 1
    scan_points = list(getattr(program, "scan_points", None) or [])
    # 32-bit COUNTER guards (the FPGA's SCAN_COUNT / LOOP_COUNT CTRL words and the
    # frame time counter are 32 bits): reject anything that would silently wrap.
    counter_limit = (1 << 32) - 1
    if len(scan_points) > counter_limit:
        raise ValueError(f"{len(scan_points)} scan points exceed the 32-bit SCAN_COUNT counter.")
    loop_count_value = int(getattr(program, "loop_count", 1) or 1)
    if loop_count_value > counter_limit:
        raise ValueError(f"loop_count {loop_count_value} exceeds the 32-bit LOOP_COUNT counter.")
    require_base_ticks_increasing = not scan_points
    last_tick = -1
    for tick in program.ticks:
        tick = int(tick)
        if require_base_ticks_increasing and tick <= last_tick:
            raise ValueError("program ticks must be strictly increasing.")
        if tick < 0 or tick > tick_limit:
            # tick_width=32 at 20 ns/tick caps one frame at ~85.9 s -- say so, the raw
            # bit-count message reads as a bug rather than a physical limit.
            seconds = tick * 20e-9
            raise ValueError(
                f"program tick {tick} does not fit {tick_width} bits: the frame time "
                f"counter is {tick_width}-bit, one frame must stay under "
                f"{((1 << tick_width) - 1) * 20e-9:.1f} s at 20 ns/tick (this edge is at "
                f"~{seconds:.1f} s). Split the sequence or use repeat/loop instead.")
        last_tick = tick
    for mask in program.masks:
        mask = int(mask)
        if mask < 0 or mask > mask_limit:
            raise ValueError(f"program mask {mask} does not fit {channel_count} channels.")
    if program.masks and int(program.masks[-1]) != 0:
        raise ValueError("program final mask must be 0 so the streamer returns to a safe idle state.")
    # The streamer SEEDS its time counter from edge 0, so the table MUST begin at tick 0
    # (a table starting after tick 0 slips every edge one tick on hardware -- the prefetch
    # startup invariant).  Backstop both compiler paths here so a forgotten anchor can
    # never reach the FPGA silently.
    if program.ticks and int(program.ticks[0]) != 0:
        raise ValueError(
            "program edge 0 must be at tick 0 (the streamer seeds its time counter from "
            "edge 0; a table starting after tick 0 slips every edge one tick)."
        )
    bus_segments = list(getattr(program, "bus_segments", None) or [])
    program_slot_count = int(getattr(program, "slot_count", 0))
    bus_segment_counts = [0 for _ in range(bus_count)]
    bus_value_limit = (1 << bus_width) - 1
    for index, segment in enumerate(bus_segments):
        bus_index = int(getattr(segment, "bus_index", segment.get("bus_index") if isinstance(segment, Mapping) else 0))
        start_tick = int(getattr(segment, "start_tick", segment.get("start_tick") if isinstance(segment, Mapping) else 0))
        stop_tick = int(getattr(segment, "stop_tick", segment.get("stop_tick", start_tick) if isinstance(segment, Mapping) else start_tick))
        start_value = int(getattr(segment, "start_value", segment.get("start_value", 0) if isinstance(segment, Mapping) else 0))
        stop_value = int(getattr(segment, "stop_value", segment.get("stop_value", start_value) if isinstance(segment, Mapping) else start_value))
        mode = str(getattr(segment, "mode", segment.get("mode", "edge") if isinstance(segment, Mapping) else "edge")).lower()
        value_select = int(getattr(segment, "value_select", segment.get("value_select", 0) if isinstance(segment, Mapping) else 0))
        if bus_index < 0 or bus_index >= bus_count:
            raise ValueError(f"bus segment {index} bus_index {bus_index} is outside bus_count={bus_count}.")
        if start_tick < 0 or stop_tick < start_tick or stop_tick > tick_limit:
            raise ValueError(f"bus segment {index} has invalid tick range {start_tick}..{stop_tick}.")
        if start_value < 0 or start_value > bus_value_limit or stop_value < 0 or stop_value > bus_value_limit:
            raise ValueError(f"bus segment {index} value does not fit {bus_width} bits.")
        if mode not in {"edge", "ramp"}:
            raise ValueError(f"bus segment {index} has unsupported mode {mode!r}.")
        # value_select==0 uses the literal value above; j+1 makes the segment read
        # its DAC code from scan slot j at runtime (the seamless DAC-value scan).
        stop_value_select = int(getattr(segment, "stop_value_select", segment.get("stop_value_select", value_select) if isinstance(segment, Mapping) else value_select))
        for sel_name, sel in (("value_select", value_select), ("stop_value_select", stop_value_select)):
            if sel < 0 or sel > program_slot_count:
                raise ValueError(
                    f"bus segment {index} {sel_name} {sel} must be 0 (literal) "
                    f"or 1..{program_slot_count} (scan slot index + 1)."
                )
            if sel > 0 and not scan_points:
                raise ValueError(f"bus segment {index} {sel_name} requires a scan-point table.")
        bus_segment_counts[bus_index] += 1
        if bus_segment_counts[bus_index] > max_bus_segments:
            raise ValueError(f"bus {bus_index} has {bus_segment_counts[bus_index]} segments, above max_bus_segments={max_bus_segments}.")
    if bus_segments and (int(getattr(program, "loop_count", 1)) > 1 and int(getattr(program, "loop_start_index", 0)) != 0):
        raise ValueError("bus_segments do not currently support finite inner repeat brackets.")
    # RAMP SLOPE: deliberately NOT validated.  The hardware ramp engine is a Bresenham
    # stepper -- per tick it moves floor-line-tracking increments (multiple LSBs for steep
    # ramps), so ANY duration yields the closest realizable staircase to the ideal line,
    # landing exactly on the target at stop_tick.  The preview draws the same staircase
    # (pulse_table._analog_bus_value_at_tick), so what you see is what the DAC does.
    # Per-bus DAC DELAY -- each DA bit is its OWN event-scheduler channel (the bus's 10 bits
    # share one delay), so a bus delay has the SAME 32-bit physical range as a TTL channel.
    # Capacity is bounded by value-change events in flight per bit
    # (<= the event-FIFO depth), like TTL -- sparse DAC use is far below it; a delayed long ramp is
    # the only stressor.  Here we enforce the 32-bit field bound; the per-bit in-flight count rides
    # the same event-FIFO contract as TTL.
    bus_delays = list(getattr(program, "bus_delays", None) or [])
    for bd in bus_delays:
        bdi = int(getattr(bd, "bus_index", bd.get("bus_index") if isinstance(bd, Mapping) else 0))
        bdd = int(getattr(bd, "delay", bd.get("delay") if isinstance(bd, Mapping) else 0))
        if bdi < 0 or bdi >= bus_count:
            raise ValueError(f"bus delay bus_index {bdi} is outside bus_count={bus_count}.")
        if bdd < 0 or bdd > TTL_DELAY_MAX_TICKS:
            raise ValueError(
                f"bus delay bus_index {bdi}: delay {bdd} ticks is outside [0, {TTL_DELAY_MAX_TICKS}] "
                f"(~{TTL_DELAY_MAX_TICKS * 20e-9:.1f} s at 20 ns/tick); reduce the delay.")
        if bdd == 0:
            continue
        # DELAYED-RAMP capacity (the one realistic per-DA-bit event-FIFO stressor): a ramp
        # changes the bus value on up to min(delta, span) ticks (the Bresenham stepping
        # ticks), and every change inside the delay window d is one in-flight event per
        # changing bit.  Conservative per-bit bound for a ramp on a DELAYED bus:
        # min(d, span, delta); a SCANNED endpoint (value_select != 0) is unknown at
        # compile time, so delta takes its full-scale worst case.  Undelayed buses and
        # edge segments are not the stressor (an edge is one event per bit; segment count
        # is capped at max_bus_segments).
        for seg in (getattr(program, "bus_segments", None) or []):
            sbi = int(getattr(seg, "bus_index", seg.get("bus_index") if isinstance(seg, Mapping) else 0))
            smode = str(getattr(seg, "mode", seg.get("mode", "edge") if isinstance(seg, Mapping) else "edge")).lower()
            if sbi != bdi or smode != "ramp":
                continue
            s_tick = int(getattr(seg, "start_tick", seg.get("start_tick") if isinstance(seg, Mapping) else 0))
            e_tick = int(getattr(seg, "stop_tick", seg.get("stop_tick", s_tick) if isinstance(seg, Mapping) else s_tick))
            v0 = int(getattr(seg, "start_value", seg.get("start_value", 0) if isinstance(seg, Mapping) else 0))
            v1 = int(getattr(seg, "stop_value", seg.get("stop_value", v0) if isinstance(seg, Mapping) else v0))
            sel0 = int(getattr(seg, "value_select", seg.get("value_select", 0) if isinstance(seg, Mapping) else 0))
            sel1 = int(getattr(seg, "stop_value_select", seg.get("stop_value_select", 0) if isinstance(seg, Mapping) else 0))
            full_scale = (1 << bus_width) - 1
            delta = full_scale if (sel0 or sel1) else abs(v1 - v0)
            span = max(0, e_tick - s_tick)
            bound = min(bdd, span, delta)
            if bound > BUS_EVT_FIFO_DEPTH:
                raise ValueError(
                    f"bus {bdi}: a DELAYED ramp can hold ~{bound} value-change events in flight "
                    f"per DA bit (delay {bdd} ticks, ramp span {span} ticks, swing {delta} codes"
                    f"{' worst-case: a scanned endpoint' if (sel0 or sel1) else ''}), above the "
                    f"per-bit event FIFO depth {BUS_EVT_FIFO_DEPTH}. Shorten the ramp, reduce the "
                    f"bus delay, lower the swing, or raise bus_evt_fifo_depth (rebuild).")
    slot_count = int(getattr(program, "slot_count", 0))
    tick_slot_coeffs = list(getattr(program, "tick_slot_coeffs", None) or [[0] * slot_count for _ in program.ticks])
    if len(tick_slot_coeffs) != len(program.ticks):
        raise ValueError("scan tick coefficient rows must match the edge table length.")
    # Edge 0 must be at tick 0 at EVERY scan point, not just the reference -- so its slot
    # coefficients must all be 0 (otherwise the seed edge moves with the scan and slips).
    if tick_slot_coeffs and any(int(c) != 0 for c in tick_slot_coeffs[0]):
        raise ValueError("program edge 0 must be at tick 0 at every scan point (edge-0 slot coefficients must be 0).")
    if slot_count > num_slots:
        raise ValueError(f"program uses {slot_count} scan slots, but the FPGA streamer has {num_slots}.")
    loop_end_slot_coeffs = list(getattr(program, "loop_end_slot_coeffs", None) or [0] * slot_count)
    coeff_min = -(1 << (coeff_width - 1))
    coeff_max = (1 << (coeff_width - 1)) - 1
    for coeff in [int(c) for row in tick_slot_coeffs for c in row] + [int(c) for c in loop_end_slot_coeffs]:
        if coeff < coeff_min or coeff > coeff_max:
            raise ValueError(f"scan coefficient {coeff} does not fit signed {coeff_width} bits.")
    frac_bits = int(getattr(program, "scan_coeff_frac_bits", DEFAULT_SCAN_COEFF_FRAC_BITS))
    if scan_points:
        if len(scan_points) > max_scan_points:
            raise ValueError(f"program has {len(scan_points)} scan points, but the FPGA streamer only accepts {max_scan_points}.")
        # Each slot VALUE goes through the affine MAC's single-DSP 16x25 product, so
        # it must fit signed slot_mul_width bits (the RTL/model narrow it identically).
        # This is a tighter bound than tick_width; +/-2^24 ticks (~335 ms) covers any
        # real scan, and the resulting tick offset still spans the full tick_width.
        signed_slot_min = -(1 << (slot_mul_width - 1))
        signed_slot_max = (1 << (slot_mul_width - 1)) - 1
        # cheap shape/value check on EVERY point (O(points*slots)).
        slot_kinds = list(getattr(program, "slot_kinds", []) or [])
        for point_index, point in enumerate(scan_points):
            if len(point) != slot_count:
                raise ValueError(f"scan point {point_index} must have {slot_count} slot value(s).")
            for slot_j, value in enumerate(point):
                v = int(value)
                if v < signed_slot_min or v > signed_slot_max:
                    raise ValueError(
                        f"scan point {point_index} value {v} does not fit signed {slot_mul_width} bits "
                        f"(the affine slot-multiply bound, +/-2^{slot_mul_width - 1} ticks)."
                    )
                # a DAC slot value is a raw DAC code read straight onto the bus, so it
                # must fit bus_width (else it would silently truncate on hardware).
                if slot_j < len(slot_kinds) and slot_kinds[slot_j] == "dac" and not (0 <= v <= bus_value_limit):
                    raise ValueError(f"scan point {point_index} DAC slot {slot_j} value {v} does not fit {bus_width} bits.")
        # effective-tick monotonicity + range is O(edges) per point; bound it for a
        # large streamed scan (the affine ticks make the per-slot EXTREME points the
        # ones most likely to reorder, so always include those).
        if max_validated_scan_points is None or len(scan_points) <= max_validated_scan_points:
            check_indices = range(len(scan_points))
        else:
            stride = max(1, len(scan_points) // max_validated_scan_points)
            picked = set(range(0, len(scan_points), stride))
            picked.add(len(scan_points) - 1)
            for j in range(slot_count):
                col_min = min(range(len(scan_points)), key=lambda i: int(scan_points[i][j]))
                col_max = max(range(len(scan_points)), key=lambda i: int(scan_points[i][j]))
                picked.add(col_min); picked.add(col_max)
            check_indices = sorted(picked)
        for point_index in check_indices:
            point = scan_points[point_index]
            last_effective_tick = -1
            for tick, coeffs in zip(program.ticks, tick_slot_coeffs):
                effective_tick = _apply_scan_tick(tick, coeffs, point, frac_bits)
                if effective_tick <= last_effective_tick:
                    raise ValueError(f"scan point {point_index} produces non-increasing effective edge ticks.")
                if effective_tick < 0 or effective_tick > tick_limit:
                    raise ValueError(f"scan point {point_index} effective tick {effective_tick} does not fit {tick_width} bits.")
                last_effective_tick = effective_tick
            # SAME-CLASS GUARD as the edge tick/coeff/mask read-latency fix: the scan BRAM
            # is read with a fixed latency (SCAN_READ_LATENCY ticks), and the engine reads
            # the NEXT point's slot vector during the CURRENT frame (lead time == this
            # frame's length).  A scanned frame shorter than that latency would be played
            # with the PREVIOUS point's scanned values (a whole-frame stale slot), so
            # reject it with a clear message instead of silently mis-scanning.  Real
            # experiment frames are micro/milliseconds (thousands of ticks); this only
            # ever trips a pathological sub-100 ns scanned period.
            if 0 < last_effective_tick < SCAN_READ_LATENCY:
                raise ValueError(
                    f"scan point {point_index} frame is only {last_effective_tick} tick(s); a scanned "
                    f"frame must be >= {SCAN_READ_LATENCY} ticks (the scan-BRAM read latency) so the next "
                    "point's slot vector is read in time -- a shorter frame plays it with the PREVIOUS "
                    "point's scanned values.  Lengthen the scanned period.")
    loop_count = int(getattr(program, "loop_count", 1))
    loop_start_index = int(getattr(program, "loop_start_index", 0))
    loop_end_tick = int(getattr(program, "loop_end_tick", 0))
    repeat_forever = bool(getattr(program, "repeat_forever", False))
    if loop_count < 1:
        raise ValueError("program loop_count must be >= 1.")
    if repeat_forever or loop_count > 1:
        if not program.ticks:
            raise ValueError("hardware repeat requires at least one uploaded edge.")
        if loop_start_index < 0 or loop_start_index >= len(program.ticks):
            raise ValueError("program loop_start_index must select an uploaded edge.")
        if loop_end_tick > tick_limit:
            raise ValueError(f"program loop_end_tick {loop_end_tick} does not fit {tick_width} bits.")
        if scan_points:
            for point_index in check_indices:   # sampled subset (bounds a huge streamed scan)
                point = scan_points[point_index]
                loop_start_tick = _apply_scan_tick(program.ticks[loop_start_index], tick_slot_coeffs[loop_start_index], point, frac_bits)
                effective_loop_end = _apply_scan_tick(loop_end_tick, loop_end_slot_coeffs, point, frac_bits)
                effective_final = _apply_scan_tick(program.ticks[-1], tick_slot_coeffs[-1], point, frac_bits)
                if effective_loop_end <= loop_start_tick:
                    raise ValueError(f"scan point {point_index} loop_end_tick must be after the loop start tick.")
                if effective_loop_end > effective_final:
                    raise ValueError(f"scan point {point_index} loop_end_tick must not exceed the uploaded final tick.")
        else:
            if loop_end_tick <= int(program.ticks[loop_start_index]):
                raise ValueError("program loop_end_tick must be after the loop start tick.")
            if loop_end_tick > int(program.ticks[-1]):
                raise ValueError("program loop_end_tick must not exceed the uploaded final tick.")
    if any(channel_delays):
        # Exact in-flight toggle analysis over the FULL schedule (every scan
        # point at its affine-shifted times, bracket loops, repeat-forever
        # seams, windows longer than one frame).  The old single-frame sliding
        # window undercounted all of those.  Runs LAST: it assumes the tick /
        # loop / scan geometry above already validated.
        _check_delay_event_capacity(
            program,
            evt_depth=EVT_FIFO_DEPTH,
            frac_bits=int(getattr(program, "scan_coeff_frac_bits", DEFAULT_SCAN_COEFF_FRAC_BITS)))


def capacity_estimate_text(
    *,
    channel_count: int = DEFAULT_FPGA_CHANNEL_COUNT,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_scan_points: int = DEFAULT_MAX_SCAN_POINTS,
    tick_width: int = DEFAULT_TICK_WIDTH,
    num_slots: int = DEFAULT_NUM_SLOTS,
    coeff_width: int = DEFAULT_SCAN_COEFF_WIDTH,
    target_pct: float = 70.0,
) -> str:
    """Return a conservative capacity note for the configured FPGA profile.

    Models the N-slot scan design: every edge row carries one signed coefficient
    PER scan slot, and every scan-table row carries one value per slot -- so the
    LUTRAM cost grows with ``num_slots``, not a fixed 2-variable (x/y) pair.
    """

    channel_count = _positive_int(channel_count, "channel_count")
    max_edges = _positive_int(max_edges, "max_edges")
    max_scan_points = _positive_int(max_scan_points, "max_scan_points")
    tick_width = _positive_int(tick_width, "tick_width")
    num_slots = _positive_int(num_slots, "num_slots")
    coeff_width = _positive_int(coeff_width, "coeff_width")
    target_pct = float(target_pct)
    lut_total = 20_800
    bram_total = 50
    baseline_lut = 2406.0
    baseline_edges = 1024.0
    baseline_row_bits = 32.0 + 40.0
    fixed_lut = max(900.0, baseline_lut - (baseline_edges * baseline_row_bits / 64.0))
    # Per-edge LUTRAM row: abs tick + channel mask + one coeff per scan slot.
    scan_row_bits = tick_width + channel_count + num_slots * coeff_width
    # Per scan-table row: one tick-domain value per scan slot.
    point_bits = num_slots * tick_width
    lut_budget = lut_total * max(1.0, min(95.0, target_pct)) / 100.0
    usable_lutram = max(0.0, lut_budget - fixed_lut)
    configured_lutram = (max_edges * scan_row_bits + max_scan_points * point_bits) / 64.0
    configured_lut_est = fixed_lut + configured_lutram
    max_edges_if_points_fixed = max(0, int((usable_lutram * 64.0 - max_scan_points * point_bits) // scan_row_bits))
    max_points_if_edges_fixed = max(0, int((usable_lutram * 64.0 - max_edges * scan_row_bits) // point_bits))
    bram_for_points = math.ceil(max_scan_points * point_bits / 36_864)
    return "\n".join(
        [
            "ZLC pulse-streamer capacity estimate",
            f"  target:          {target_pct:g}% LUT budget ({lut_budget:.0f}/{lut_total})",
            f"  configured:      channels={channel_count} edges={max_edges} scan_points={max_scan_points} "
            f"slots={num_slots} tick_width={tick_width}",
            f"  row bits:        edge_template={scan_row_bits} scan_point={point_bits}",
            f"  LUT estimate:    {configured_lut_est:.0f}/{lut_total} ({configured_lut_est / lut_total * 100.0:.1f}%)",
            f"  at target:       with {max_scan_points} scan points, approx max_edges={max_edges_if_points_fixed}",
            f"  at target:       with {max_edges} edges, approx max_scan_points={max_points_if_edges_fixed}",
            f"  BRAM note:       scan-point RAM would be about {bram_for_points}/{bram_total} 36Kb blocks if mapped to BRAM",
            "  final evidence:  use Vivado report_utilization after synthesis; this is a design-budget estimate.",
        ]
    )


def _apply_scan_tick(base_tick: int, coeffs, slot_ticks, frac_bits: int) -> int:
    total = sum(int(coeff) * int(tick) for coeff, tick in zip(coeffs, slot_ticks))
    return int(base_tick) + (total >> int(frac_bits))


# ---------------------------------------------------------------------------
# TTL delay EVENT-SCHEDULER capacity analysis.
#
# The per-channel delay FIFO holds the channel's toggles that are IN FLIGHT: a
# toggle pushed at (undelayed) wall tick t pops at t+d-1, so at any moment the
# occupancy equals the number of toggles inside a window of the channel's own
# delay length.  The analysis below reconstructs the channel's UNDELAYED toggle
# stream over the WHOLE program -- every scan point at its affine-shifted edge
# times, bracket loops, and the repeat-forever wrap -- and takes the exact
# maximum window count.  (The old per-frame heuristic undercounted whenever a
# window crossed scan-point seams or spanned more than one frame: d > frame.)
# ---------------------------------------------------------------------------


def _program_scan_slots(program) -> list[list[int]]:
    points = [[int(v) for v in row] for row in (getattr(program, "scan_points", None) or [])]
    if points:
        return points
    slot_count = int(getattr(program, "slot_count", 0) or 0)
    return [[0] * slot_count]


def _program_tick_coeffs(program) -> list:
    coeffs = getattr(program, "tick_slot_coeffs", None)
    if coeffs:
        return [list(row) for row in coeffs]
    return [[] for _ in program.ticks]


def _frame_events_for_point(ticks, masks, coeffs, slots, frac_bits,
                            loop_start_index, loop_end_tick, loop_end_coeffs,
                            loop_count, start_index=0):
    """One frame's played ``[(wall_tick, mask)]`` (in play order) + wall duration.

    Mirrors the engine: edges play at their affine effective ticks; a finite
    bracket (``loop_count > 1``) replays the [loop_start, loop_end) time slice,
    each iteration advancing the wall clock by the slice length while the edge
    timeline rewinds.  ``start_index`` plays the frame from that edge (the
    repeat-forever rewind point), supported for plain frames."""
    n = len(ticks)
    if n == 0:
        return [], 0
    eff = [_apply_scan_tick(int(ticks[i]), coeffs[i], slots, frac_bits) for i in range(n)]
    final = eff[-1]
    loop_count = max(1, int(loop_count))
    if loop_count <= 1:
        s = max(0, min(int(start_index), n - 1))
        base = eff[s] if s > 0 else 0
        return [(eff[i] - base, int(masks[i])) for i in range(s, n)], max(1, final - base)
    if start_index:
        # No compiler emits a rewind together with a finite bracket
        # (repeat_from_index is forced to 0 there); fall back to the full frame,
        # which only ADDS the preamble toggles -- conservative for capacity.
        start_index = 0
    ls_i = max(0, min(int(loop_start_index), n - 1))
    ls_t = eff[ls_i]
    le_t = _apply_scan_tick(int(loop_end_tick), loop_end_coeffs or (), slots, frac_bits)
    slice_len = max(0, le_t - ls_t)
    events = [(eff[i], int(masks[i])) for i in range(ls_i)]
    loop_idx = [i for i in range(ls_i, n) if eff[i] < le_t]
    for k in range(loop_count):
        off = k * slice_len
        events.extend((eff[i] + off, int(masks[i])) for i in loop_idx)
    tail_off = (loop_count - 1) * slice_len
    events.extend((eff[i] + tail_off, int(masks[i])) for i in range(ls_i, n) if eff[i] >= le_t)
    return events, max(1, final + tail_off)


def _frame_duration_for_point(program, slots, frac_bits, start_index=0) -> int:
    ticks = program.ticks
    n = len(ticks)
    if n == 0:
        return 0
    coeffs = _program_tick_coeffs(program)
    final = _apply_scan_tick(int(ticks[-1]), coeffs[-1], slots, frac_bits)
    base = 0
    s = max(0, min(int(start_index), n - 1))
    if s > 0:
        base = _apply_scan_tick(int(ticks[s]), coeffs[s], slots, frac_bits)
    loop_count = max(1, int(getattr(program, "loop_count", 1) or 1))
    tail = 0
    if loop_count > 1:
        ls_i = max(0, min(int(getattr(program, "loop_start_index", 0) or 0), n - 1))
        ls_t = _apply_scan_tick(int(ticks[ls_i]), coeffs[ls_i], slots, frac_bits)
        le_t = _apply_scan_tick(int(getattr(program, "loop_end_tick", 0) or 0),
                                getattr(program, "loop_end_slot_coeffs", None) or (), slots, frac_bits)
        tail = (loop_count - 1) * max(0, le_t - ls_t)
    return max(1, final - base + tail)


def _sweep_events(program, frac_bits, start_index=0):
    """``[(wall, mask)]`` across one full sweep (every scan point) + sweep ticks."""
    points = _program_scan_slots(program)
    coeffs = _program_tick_coeffs(program)
    loop_start_index = int(getattr(program, "loop_start_index", 0) or 0)
    loop_end_tick = int(getattr(program, "loop_end_tick", 0) or 0)
    loop_end_coeffs = getattr(program, "loop_end_slot_coeffs", None) or ()
    loop_count = max(1, int(getattr(program, "loop_count", 1) or 1))
    events: list = []
    offset = 0
    for index, slots in enumerate(points):
        frame, duration = _frame_events_for_point(
            program.ticks, program.masks, coeffs, slots, frac_bits,
            loop_start_index, loop_end_tick, loop_end_coeffs, loop_count,
            start_index=start_index if index == 0 else 0)
        events.extend((offset + t, m) for t, m in frame)
        offset += duration
    return events, offset


def steady_sweep_ticks(program, *, frac_bits: int | None = None) -> int:
    """Wall ticks of one steady-state sweep (all scan points, bracket loops in).

    This is the period of the undelayed output stream under ``repeat_forever``
    (the wrap rewinds to ``repeat_from_index``, so the steady first frame may be
    shorter than the very first one)."""
    frac = int(getattr(program, "scan_coeff_frac_bits", DEFAULT_SCAN_COEFF_FRAC_BITS)
               if frac_bits is None else frac_bits)
    start = int(getattr(program, "repeat_from_index", 0) or 0)
    total = 0
    for index, slots in enumerate(_program_scan_slots(program)):
        total += _frame_duration_for_point(program, slots, frac,
                                           start_index=start if index == 0 else 0)
    return total


def effective_channel_delays(program, *, frac_bits: int | None = None) -> list[int]:
    """Channel delays as PROGRAMMED -- the TRUE physical delays, unchanged.

    The delay is genuinely physical (``out[t]=in[t-d]``, silent for the first
    ``d`` ticks; the first frame is correct).  There is NO modulo-by-period
    reduction -- whether the in-flight edge count fits the event FIFO is a
    capacity question enforced by ``validate_pulse_streamer_program``, not
    something papered over by rewriting the delay.  Kept as an identity helper
    so callers have one named place for "the delays that hit the hardware"."""
    return [int(d) for d in (getattr(program, "channel_delays", None) or [])]


def _max_in_flight_toggles(toggles: Sequence[int], d: int) -> int:
    """Max number of toggles inside any CLOSED window [t-d, t] (conservative)."""
    worst = 0
    lo = 0
    for hi, t in enumerate(toggles):
        while t - toggles[lo] > d:
            lo += 1
        worst = max(worst, hi - lo + 1)
    return worst


def _max_in_flight_periodic(steady_toggles: Sequence[int], period: int, d: int) -> int:
    """Exact max toggle count in a closed d-window over a PERIODIC toggle stream.

    The stream is ``{t + k*period}`` for each ``t`` in ``steady_toggles`` and all
    integers ``k >= 0`` (steady state).  A maximal window ends at a toggle, so it
    suffices to test windows ending at each toggle position (mod period).
    Needed when ``d >= period``: a finite unrolling can undercount there."""
    if not steady_toggles or period <= 0:
        return 0
    best = 0
    for x in steady_toggles:
        count = 0
        for t in steady_toggles:
            # integers k with x - d <= t + k*period <= x
            hi_k = (x - t) // period
            lo_k = -((d - x + t) // period)     # ceil((x - d - t) / period)
            if hi_k >= lo_k:
                count += hi_k - lo_k + 1
        best = max(best, count)
    return best


def _frame_toggle_count_bound(program, slots, frac_bits: int) -> dict:
    """Per-bit upper bound on toggles in ONE frame, WITHOUT unrolling the bracket.

    Masks are absolute, so the bit trajectory of one bracket iteration is the
    same in every iteration except possibly at the iteration seam; counting the
    slice toggles once (from either entry state) and multiplying by loop_count,
    plus one seam toggle per iteration, bounds the total."""
    ticks, masks = program.ticks, program.masks
    n = len(ticks)
    loop_count = max(1, int(getattr(program, "loop_count", 1) or 1))
    coeffs = _program_tick_coeffs(program)
    bits = {b for mask in masks for b in range(int(mask).bit_length())}
    out = {}
    if loop_count <= 1:
        for b in bits:
            toggles, _ = _channel_toggle_times([(0, int(m)) for m in masks], b, 0)
            out[b] = len(toggles)
        return out
    ls_i = max(0, min(int(getattr(program, "loop_start_index", 0) or 0), n - 1))
    le_t = _apply_scan_tick(int(getattr(program, "loop_end_tick", 0) or 0),
                            getattr(program, "loop_end_slot_coeffs", None) or (), slots, frac_bits)
    eff = [_apply_scan_tick(int(ticks[i]), coeffs[i], slots, frac_bits) for i in range(n)]
    pre = [int(masks[i]) for i in range(ls_i)]
    slice_masks = [int(masks[i]) for i in range(ls_i, n) if eff[i] < le_t]
    post = [int(masks[i]) for i in range(ls_i, n) if eff[i] >= le_t]
    for b in bits:
        pre_t, prev = _channel_toggle_times([(0, m) for m in pre], b, 0)
        slice_t, _ = _channel_toggle_times([(0, m) for m in slice_masks], b, prev)
        post_t, _ = _channel_toggle_times([(0, m) for m in post], b, prev)
        # slice toggles per iteration vary by at most 1 with the entry state
        out[b] = len(pre_t) + (len(slice_t) + 1) * loop_count + len(post_t)
    return out


def _channel_toggle_times(events, bit: int, prev_bit: int = 0):
    toggles = []
    for t, mask in events:
        value = (mask >> bit) & 1
        if value != prev_bit:
            toggles.append(t)
            prev_bit = value
    return toggles, prev_bit


def _check_delay_event_capacity(program, *, evt_depth: int, frac_bits: int,
                                max_work: int = 4_000_000) -> None:
    """Reject programs whose toggle stream would overflow a delay event FIFO.

    Exact analysis over the real schedule: every scan point (affine-shifted
    edge times), bracket loops, and the repeat-forever wrap.  For gigantic
    sweeps (work > ``max_work`` played events) a conservative frame-count bound
    is used instead, and the error says so."""
    delays = [int(d) for d in (getattr(program, "channel_delays", None) or [])]
    checked = [(b, d) for b, d in enumerate(delays) if d > 1]   # d==0 bypass; d==1 register
    if not checked or not program.ticks:
        return
    if int(getattr(program, "repeat_from_index", 0) or 0) != 0:
        # The analysis models the rewind only partially (the first sweep can be
        # denser than the steady state and would escape the d>=period check).
        # No compiler emits a rewind together with channel delays; reject the
        # combination outright rather than risk a silent FIFO overflow.
        raise ValueError(
            "channel delays combined with repeat_from_index != 0 are not supported "
            "(the delay event-FIFO capacity cannot be verified for a rewound preamble).")
    forever = bool(getattr(program, "repeat_forever", False))
    points = _program_scan_slots(program)
    loop_count = max(1, int(getattr(program, "loop_count", 1) or 1))
    work = len(points) * len(program.ticks) * loop_count

    if work > max_work:
        # Conservative path: a d-window spans at most floor(d/min_frame)+2 frames,
        # and the per-frame toggle count is point-independent (masks are absolute,
        # edges play in the same order at every point).  The bracket loop is NOT
        # unrolled (loop_count can be 2^32): count slice toggles arithmetically.
        durations = [_frame_duration_for_point(program, slots, frac_bits) for slots in points]
        min_frame = max(1, min(durations))
        per_frame_counts = _frame_toggle_count_bound(program, points[0], frac_bits)
        for b, d in checked:
            per_frame = per_frame_counts.get(b, 0) + 1   # +1 for a possible seam toggle
            bound = per_frame * (d // min_frame + 2)
            if bound > evt_depth:
                max_d = max(0, evt_depth * min_frame // max(1, per_frame))
                raise ValueError(
                    f"channel-delay output bit {b}: a PHYSICAL delay of {d} ticks could keep up to "
                    f"~{bound} edges in flight (conservative estimate; the sweep is too large to "
                    f"verify exactly), but the per-channel event FIFO holds {evt_depth}. The longest "
                    f"physical delay this channel's toggle rate allows is about {max_d} ticks. "
                    f"Reduce the delay or the channel's toggle rate, or rebuild the bitstream with a "
                    f"larger evt_fifo_depth.")
        return

    sweep_a, period_a = _sweep_events(program, frac_bits, start_index=0)
    rewind = int(getattr(program, "repeat_from_index", 0) or 0)
    if forever:
        if rewind:
            sweep_b, period_b = _sweep_events(program, frac_bits, start_index=rewind)
        else:
            sweep_b, period_b = sweep_a, period_a
    for b, d in checked:
        toggles, end_bit = _channel_toggle_times(sweep_a, b, 0)
        if forever and sweep_b:
            if d >= period_b > 0:
                # TRUE PHYSICAL delay spanning whole sweeps (no modulo reduction):
                # the steady stream is periodic, so the exact in-flight count is the
                # periodic d-window maximum.  The longest delay that fits the FIFO is
                # ~ depth * period / toggles_per_period.
                steady_toggles, _ = _channel_toggle_times(sweep_b, b, end_bit)
                per_period = max(1, len(steady_toggles))
                cheap = per_period * (d // period_b + 2)   # cheap upper bound (exact is O(n^2))
                if cheap <= evt_depth:
                    continue
                worst = (cheap if per_period ** 2 > max_work
                         else _max_in_flight_periodic(steady_toggles, period_b, d))
                if worst > evt_depth:
                    max_d = max(0, evt_depth * period_b // per_period)
                    raise ValueError(
                        f"channel-delay output bit {b}: a PHYSICAL delay of {d} ticks "
                        f"({d * 20e-9 * 1e3:.3f} ms) spans {d // period_b} repeating frames "
                        f"(period {period_b} ticks) and keeps up to {worst} edges in flight, but "
                        f"the per-channel event FIFO holds {evt_depth}. The longest physical delay "
                        f"this channel's toggle rate allows is about {max_d} ticks "
                        f"({max_d * 20e-9 * 1e3:.3f} ms). Reduce the delay or the channel's toggle "
                        f"rate, or rebuild the bitstream with a larger evt_fifo_depth.")
                continue
            # d < period: the first sweep plus TWO steady sweeps cover every
            # window, including all wrap seams.
            offset = period_a
            for _ in range(2):
                more, end_bit = _channel_toggle_times(sweep_b, b, end_bit)
                toggles.extend(offset + t for t in more)
                offset += period_b
        worst = _max_in_flight_toggles(toggles, d)
        if worst > evt_depth:
            # longest delay that fits: shrink d until the worst window holds <= depth.
            span = max(1, (toggles[-1] - toggles[0]) if len(toggles) > 1 else d)
            density = worst / max(1, d)
            max_d = max(0, int(evt_depth / density)) if density > 0 else d
            raise ValueError(
                f"channel-delay output bit {b}: a PHYSICAL delay of {d} ticks "
                f"({d * 20e-9 * 1e3:.3f} ms) keeps {worst} output edges in flight (counted over "
                f"the full schedule: every scan point, bracket loops and the repeat seam), but the "
                f"per-channel event FIFO holds {evt_depth}. The longest physical delay this "
                f"channel's toggle rate allows is about {max_d} ticks ({max_d * 20e-9 * 1e3:.3f} ms). "
                f"Reduce the delay or the channel's toggle rate, or rebuild the bitstream with a "
                f"larger evt_fifo_depth.")


def _safe_channel_identifiers(channels: Sequence[str], *, reserved: set[str]) -> list[str]:
    safe_channels = [safe_identifier(channel) for channel in channels]
    if len(set(safe_channels)) != len(safe_channels):
        raise ValueError("channel names collide after Verilog identifier sanitization.")
    collisions = sorted(set(safe_channels) & set(reserved))
    if collisions:
        raise ValueError(f"channel names collide with pulse-streamer top-level names: {collisions}")
    return safe_channels


def _label_from_xdc_comment(comment: str, channel: str) -> str:
    text = str(comment or "").strip()
    if not text:
        return ""
    if "<-" in text:
        text = text.split("<-", 1)[1].strip()
    elif "=" in text:
        left, right = text.split("=", 1)
        if channel in left or re.search(r"\bch\d+\b", left):
            text = right.strip()
    text = text.split(",", 1)[0].strip()
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_\[\]-]+", "", text)
    return text


def delay_eligible_channel_count(channel_count: int, bus_count: int = DEFAULT_BUS_COUNT,
                                 bus_width: int = DEFAULT_BUS_WIDTH) -> int:
    """Number of leading channels that can carry a TTL output delay.

    A channel is delay-eligible iff its engine bit drives a real TTL pin -- i.e. it is
    NOT a bus-member bit (``bus_count*bus_width`` of them, pins driven by ``bus_out``)
    and NOT a per-bus ``da_clk`` pin (``bus_count`` of them).  The board lays the real
    TTL outputs out FIRST, so the eligible set is the leading
    ``channel_count - bus_count*(bus_width+1)`` indices -- matching the RTL's compacted
    (identity) event-FIFO map.  Only these get an event FIFO (deep enough at depth 256
    only because the bus/clk channels are excluded)."""
    return max(0, int(channel_count) - int(bus_count) * (int(bus_width) + 1))


def _positive_int(value, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive.")
    return out


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _resolve_xdc_path(xdc_path: str | Path | None) -> Path | None:
    if xdc_path is not None and str(xdc_path).strip():
        return Path(xdc_path)
    value = os.environ.get("ZLC_PS_XDC")
    if value:
        return Path(value)
    # Default board pin map: the in-repo platform-config copy (fpga/board_config/board.xdc,
    # see its README).  The old references/ copy is deprecated and no longer consulted.
    relative = Path("fpga") / "board_config" / "board.xdc"
    cwd_candidate = Path.cwd() / relative
    if cwd_candidate.exists():
        return cwd_candidate
    package_candidate = Path(__file__).resolve().parents[3] / relative
    if package_candidate.exists():
        return package_candidate
    return None




__all__ = [
    "DEFAULT_FPGA_CHANNEL_COUNT",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_SCAN_POINTS",
    "DEFAULT_MAX_BUS_SEGMENTS",
    "DEFAULT_SCAN_COEFF_FRAC_BITS",
    "DEFAULT_SCAN_COEFF_WIDTH",
    "DEFAULT_TICK_WIDTH",
    "DEFAULT_NUM_SLOTS",
    "DEFAULT_BUS_COUNT",
    "DEFAULT_BUS_WIDTH",
    "effective_channel_delays",
    "delay_eligible_channel_count",
    "steady_sweep_ticks",
    "hardware_channel_names",
    "capacity_estimate_text",
    "infer_xdc_channel_count",
    "infer_xdc_channel_labels",
    "infer_xdc_channel_pins",
    "infer_xdc_channels",
    "infer_xdc_trigger_channels",
    "validate_pulse_streamer_program",
    "build_arg_parser",
    "main",
]


def build_arg_parser() -> ArgumentParser:
    """Infer-only CLI used by run_server.bat.  The final design has NO HDL/VIO
    generator (the host drives the bitstream over JTAG-to-AXI via axi_session),
    so the only CLI actions are reading channel/trigger info from a board XDC."""

    parser = ArgumentParser(description="Infer FPGA channel/trigger info from a board XDC.")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("infer_channel_count", "infer_channels", "infer_channel_labels",
                 "infer_channel_pins", "infer_trigger_channels"):
        p = sub.add_parser(name)
        p.add_argument("--xdc", default=None)
        p.add_argument("--default-count", type=int, default=DEFAULT_FPGA_CHANNEL_COUNT)
        p.add_argument("--max-channel-count", type=int, default=None)


    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    xdc = args.xdc or None
    default = int(args.default_count)
    max_count = args.max_channel_count
    if args.action == "infer_channel_count":
        print(infer_xdc_channel_count(xdc, default=default, max_count=max_count))
    elif args.action == "infer_channels":
        print(" ".join(infer_xdc_channels(xdc, default=default, max_count=max_count)))
    elif args.action == "infer_channel_labels":
        labels = infer_xdc_channel_labels(xdc, default=default, max_count=max_count)
        print(" ".join(f"{k}={v}" for k, v in labels.items()))
    elif args.action == "infer_channel_pins":
        pins = infer_xdc_channel_pins(xdc, default=default, max_count=max_count)
        print(" ".join(f"{k}={v}" for k, v in pins.items()))
    elif args.action == "infer_trigger_channels":
        print(" ".join(infer_xdc_trigger_channels(xdc, default=default, max_count=max_count)))
    else:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
