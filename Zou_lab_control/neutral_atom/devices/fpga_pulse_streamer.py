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


DEFAULT_FPGA_CHANNEL_COUNT = 62
DEFAULT_CHANNELS = [f"ch{index:02d}" for index in range(DEFAULT_FPGA_CHANNEL_COUNT)]
DEFAULT_MAX_EDGES = 1024
DEFAULT_MAX_SCAN_POINTS = 1024
DEFAULT_MAX_BUS_SEGMENTS = 64
DEFAULT_TICK_WIDTH = 32
DEFAULT_SCAN_COEFF_WIDTH = 16
DEFAULT_SCAN_COEFF_FRAC_BITS = 8
DEFAULT_NUM_SLOTS = 4
# Affine-MAC slot operand width -- MUST match zlc_edge_streamer.v SLOT_MUL_WIDTH
# and engine_model.SLOT_MUL_WIDTH.  Each scan slot VALUE is multiplied by a 16-bit
# coeff on a single DSP48E1 (25x18), so the slot operand is the low 25 bits taken
# as signed; a scan value outside +/-2^24 ticks (~+/-335 ms @ 20 ns) would diverge
# from the model, so the validator rejects it.  (The coeff still scales the slot,
# so the resulting tick OFFSET keeps the full 32-bit range.)
DEFAULT_SLOT_MUL_WIDTH = 25
DEFAULT_BUS_COUNT = 4
DEFAULT_BUS_WIDTH = 10


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
    if getattr(program, "delay_lanes", None):
        # SAFETY GATE (removed when the lane RTL lands): a reordering scanned-delay
        # program compiles to disjoint-bit delay lanes and is proven cycle-accurate by
        # engine_model, but the lane sub-players are not yet in the bitstream -- uploading
        # would leave the lane channel silent.  Reject at upload (never silently wrong)
        # until the RTL lane players are built.
        raise NotImplementedError(
            "this program uses delay lanes (a scanned digital delay whose edges reorder "
            "past other channels); the RTL lane players are the next build step.  Until "
            "then keep the scanned delay non-reordering so it stays in the main edge table."
        )
    if len(set(program.channels)) != len(program.channels):
        raise ValueError("program channels must be unique.")
    if len(program.ticks) != len(program.masks):
        raise ValueError("program ticks and masks must have the same length.")
    if len(program.ticks) > max_edges:
        raise ValueError(f"program has {len(program.ticks)} edges, but the FPGA streamer only accepts {max_edges}.")
    if len(program.channels) > channel_count:
        raise ValueError(f"program uses {len(program.channels)} channels, but the FPGA streamer has {channel_count}.")
    tick_limit = (1 << tick_width) - 1
    mask_limit = (1 << channel_count) - 1
    scan_points = list(getattr(program, "scan_points", None) or [])
    require_base_ticks_increasing = not scan_points
    last_tick = -1
    for tick in program.ticks:
        tick = int(tick)
        if require_base_ticks_increasing and tick <= last_tick:
            raise ValueError("program ticks must be strictly increasing.")
        if tick < 0 or tick > tick_limit:
            raise ValueError(f"program tick {tick} does not fit {tick_width} bits.")
        last_tick = tick
    for mask in program.masks:
        mask = int(mask)
        if mask < 0 or mask > mask_limit:
            raise ValueError(f"program mask {mask} does not fit {channel_count} channels.")
    if program.masks and int(program.masks[-1]) != 0:
        raise ValueError("program final mask must be 0 so the streamer returns to a safe idle state.")
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
    slot_count = int(getattr(program, "slot_count", 0))
    tick_slot_coeffs = list(getattr(program, "tick_slot_coeffs", None) or [[0] * slot_count for _ in program.ticks])
    if len(tick_slot_coeffs) != len(program.ticks):
        raise ValueError("scan tick coefficient rows must match the edge table length.")
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
            for point_index, point in enumerate(scan_points):
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
    cwd_candidate = Path.cwd() / "references" / "source_archives" / "address_switch" / "address_switch.srcs" / "constrs_1" / "new" / "addre.xdc"
    if cwd_candidate.exists():
        return cwd_candidate
    package_candidate = Path(__file__).resolve().parents[3] / "references" / "source_archives" / "address_switch" / "address_switch.srcs" / "constrs_1" / "new" / "addre.xdc"
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
