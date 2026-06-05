"""Vivado/VIO runtime pulse-streamer backend for the FPGA computer.

The control computer sends a ``PulseSequence`` over RPyC.  The FPGA computer
compiles it into a ``RuntimeSequenceProgram`` and this backend uploads the
resulting edge table to a fixed pulse-streamer bitstream.
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
DEFAULT_VIO_FILTER = 'CELL_NAME=~"*vio*"'
DEFAULT_MAX_EDGES = 512
DEFAULT_MAX_SCAN_POINTS = 256
DEFAULT_MAX_BUS_SEGMENTS = 64
DEFAULT_TICK_WIDTH = 32
DEFAULT_SCAN_COEFF_WIDTH = 16
DEFAULT_SCAN_COEFF_FRAC_BITS = 8
DEFAULT_BUS_COUNT = 4
DEFAULT_BUS_WIDTH = 10
DEFAULT_PROJECT_NAME = "address_switch"
DEFAULT_TOP_NAME = "zlc_pulse_streamer_top_address_switch"


@dataclass(frozen=True)
class PulseStreamerProbeNames:
    """Names of VIO probes connected to ``zlc_pulse_streamer``."""

    reset: str = "zlc_reset"
    start: str = "zlc_start"
    prog_we: str = "zlc_prog_we"
    prog_addr: str = "zlc_prog_addr"
    prog_tick: str = "zlc_prog_tick"
    prog_mask: str = "zlc_prog_mask"
    prog_count: str = "zlc_prog_count"
    repeat_forever: str = "zlc_repeat_forever"
    loop_start_addr: str = "zlc_loop_start_addr"
    loop_end_tick: str = "zlc_loop_end_tick"
    loop_count: str = "zlc_loop_count"
    prog_tick_x_coeff: str = "zlc_prog_tick_x_coeff"
    prog_tick_y_coeff: str = "zlc_prog_tick_y_coeff"
    scan_enable: str = "zlc_scan_enable"
    scan_prog_we: str = "zlc_scan_prog_we"
    scan_prog_addr: str = "zlc_scan_prog_addr"
    scan_prog_x: str = "zlc_scan_prog_x"
    scan_prog_y: str = "zlc_scan_prog_y"
    scan_prog_bus_values: str = "zlc_scan_prog_bus_values"
    scan_count: str = "zlc_scan_count"
    loop_end_x_coeff: str = "zlc_loop_end_x_coeff"
    loop_end_y_coeff: str = "zlc_loop_end_y_coeff"
    bus_prog_we: str = "zlc_bus_prog_we"
    bus_prog_bus: str = "zlc_bus_prog_bus"
    bus_prog_addr: str = "zlc_bus_prog_addr"
    bus_prog_start_tick: str = "zlc_bus_prog_start_tick"
    bus_prog_stop_tick: str = "zlc_bus_prog_stop_tick"
    bus_prog_start_value: str = "zlc_bus_prog_start_value"
    bus_prog_stop_value: str = "zlc_bus_prog_stop_value"
    bus_prog_mode: str = "zlc_bus_prog_mode"
    bus_counts: str = "zlc_bus_counts"
    running: str = "zlc_running"
    done: str = "zlc_done"


@dataclass(frozen=True)
class PulseStreamerHDLFiles:
    core_path: Path
    top_example_path: Path
    manifest_path: Path


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
    bus_count: int = DEFAULT_BUS_COUNT,
    bus_width: int = DEFAULT_BUS_WIDTH,
) -> None:
    """Validate that a runtime edge table fits the fixed FPGA streamer."""

    max_edges = _positive_int(max_edges, "max_edges")
    max_scan_points = _positive_int(max_scan_points, "max_scan_points")
    max_bus_segments = _positive_int(max_bus_segments, "max_bus_segments")
    tick_width = _positive_int(tick_width, "tick_width")
    coeff_width = _positive_int(coeff_width, "coeff_width")
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
    tick_limit = (1 << tick_width) - 1
    mask_limit = (1 << channel_count) - 1
    scan_points = list(getattr(program, "scan_points", None) or [])
    scan_bus_values = list(getattr(program, "scan_bus_values", None) or [])
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
    if bus_segments and scan_points:
        raise ValueError("program cannot combine scan_points with bus_segments in the current FPGA bitstream.")
    if scan_bus_values and not scan_points:
        raise ValueError("scan_bus_values require scan_points rows.")
    if scan_bus_values and len(scan_bus_values) != len(scan_points):
        raise ValueError("scan_bus_values length must match scan_points length.")
    scan_bus_value_limit = (1 << (bus_count * bus_width)) - 1
    for point_index, value in enumerate(scan_bus_values):
        value = int(value)
        if value < 0 or value > scan_bus_value_limit:
            raise ValueError(f"scan_bus_values[{point_index}] does not fit {bus_count * bus_width} bits.")
    bus_segment_counts = [0 for _ in range(bus_count)]
    bus_value_limit = (1 << bus_width) - 1
    for index, segment in enumerate(bus_segments):
        bus_index = int(getattr(segment, "bus_index", segment.get("bus_index") if isinstance(segment, Mapping) else 0))
        start_tick = int(getattr(segment, "start_tick", segment.get("start_tick") if isinstance(segment, Mapping) else 0))
        stop_tick = int(getattr(segment, "stop_tick", segment.get("stop_tick", start_tick) if isinstance(segment, Mapping) else start_tick))
        start_value = int(getattr(segment, "start_value", segment.get("start_value", 0) if isinstance(segment, Mapping) else 0))
        stop_value = int(getattr(segment, "stop_value", segment.get("stop_value", start_value) if isinstance(segment, Mapping) else start_value))
        mode = str(getattr(segment, "mode", segment.get("mode", "edge") if isinstance(segment, Mapping) else "edge")).lower()
        if bus_index < 0 or bus_index >= bus_count:
            raise ValueError(f"bus segment {index} bus_index {bus_index} is outside bus_count={bus_count}.")
        if start_tick < 0 or stop_tick < start_tick or stop_tick > tick_limit:
            raise ValueError(f"bus segment {index} has invalid tick range {start_tick}..{stop_tick}.")
        if start_value < 0 or start_value > bus_value_limit or stop_value < 0 or stop_value > bus_value_limit:
            raise ValueError(f"bus segment {index} value does not fit {bus_width} bits.")
        if mode not in {"edge", "ramp"}:
            raise ValueError(f"bus segment {index} has unsupported mode {mode!r}.")
        bus_segment_counts[bus_index] += 1
        if bus_segment_counts[bus_index] > max_bus_segments:
            raise ValueError(f"bus {bus_index} has {bus_segment_counts[bus_index]} segments, above max_bus_segments={max_bus_segments}.")
    if bus_segments and (int(getattr(program, "loop_count", 1)) > 1 and int(getattr(program, "loop_start_index", 0)) != 0):
        raise ValueError("bus_segments do not currently support finite inner repeat brackets.")
    tick_x_coeffs = list(getattr(program, "tick_x_coeffs", None) or [0 for _ in program.ticks])
    tick_y_coeffs = list(getattr(program, "tick_y_coeffs", None) or [0 for _ in program.ticks])
    if len(tick_x_coeffs) != len(program.ticks) or len(tick_y_coeffs) != len(program.ticks):
        raise ValueError("scan tick coefficient arrays must match the edge table length.")
    coeff_min = -(1 << (coeff_width - 1))
    coeff_max = (1 << (coeff_width - 1)) - 1
    for coeff in [*tick_x_coeffs, *tick_y_coeffs, int(getattr(program, "loop_end_x_coeff", 0)), int(getattr(program, "loop_end_y_coeff", 0))]:
        if int(coeff) < coeff_min or int(coeff) > coeff_max:
            raise ValueError(f"scan coefficient {int(coeff)} does not fit signed {coeff_width} bits.")
    frac_bits = int(getattr(program, "scan_coeff_frac_bits", DEFAULT_SCAN_COEFF_FRAC_BITS))
    if scan_points:
        if len(scan_points) > max_scan_points:
            raise ValueError(f"program has {len(scan_points)} scan points, but the FPGA streamer only accepts {max_scan_points}.")
        signed_tick_min = -(1 << (tick_width - 1))
        signed_tick_max = (1 << (tick_width - 1)) - 1
        for point_index, point in enumerate(scan_points):
            if len(point) != 2:
                raise ValueError(f"scan row {point_index} must contain two timing-axis tick slots.")
            x_tick, y_tick = int(point[0]), int(point[1])
            if x_tick < signed_tick_min or x_tick > signed_tick_max or y_tick < signed_tick_min or y_tick > signed_tick_max:
                raise ValueError(f"scan row {point_index} timing-axis tick does not fit signed {tick_width} bits.")
            last_effective_tick = -1
            for tick, x_coeff, y_coeff in zip(program.ticks, tick_x_coeffs, tick_y_coeffs):
                effective_tick = _apply_scan_tick(tick, x_coeff, y_coeff, x_tick, y_tick, frac_bits)
                if effective_tick <= last_effective_tick:
                    raise ValueError(f"scan row {point_index} produces non-increasing effective edge ticks.")
                if effective_tick < 0 or effective_tick > tick_limit:
                    raise ValueError(f"scan row {point_index} effective tick {effective_tick} does not fit {tick_width} bits.")
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
            loop_end_x_coeff = int(getattr(program, "loop_end_x_coeff", 0))
            loop_end_y_coeff = int(getattr(program, "loop_end_y_coeff", 0))
            for point_index, (x_tick, y_tick) in enumerate(scan_points):
                loop_start_tick = _apply_scan_tick(
                    program.ticks[loop_start_index],
                    tick_x_coeffs[loop_start_index],
                    tick_y_coeffs[loop_start_index],
                    x_tick,
                    y_tick,
                    frac_bits,
                )
                effective_loop_end = _apply_scan_tick(loop_end_tick, loop_end_x_coeff, loop_end_y_coeff, x_tick, y_tick, frac_bits)
                effective_final = _apply_scan_tick(program.ticks[-1], tick_x_coeffs[-1], tick_y_coeffs[-1], x_tick, y_tick, frac_bits)
                if effective_loop_end <= loop_start_tick:
                    raise ValueError(f"scan row {point_index} loop_end_tick must be after the loop start tick.")
                if effective_loop_end > effective_final:
                    raise ValueError(f"scan row {point_index} loop_end_tick must not exceed the uploaded final tick.")
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
    target_pct: float = 70.0,
) -> str:
    """Return a conservative capacity note for the configured FPGA profile."""

    channel_count = _positive_int(channel_count, "channel_count")
    max_edges = _positive_int(max_edges, "max_edges")
    max_scan_points = _positive_int(max_scan_points, "max_scan_points")
    tick_width = _positive_int(tick_width, "tick_width")
    target_pct = float(target_pct)
    lut_total = 20_800
    bram_total = 50
    baseline_lut = 2406.0
    baseline_edges = 1024.0
    baseline_row_bits = 32.0 + 40.0
    fixed_lut = max(900.0, baseline_lut - (baseline_edges * baseline_row_bits / 64.0))
    scan_row_bits = tick_width + channel_count + 2 * DEFAULT_SCAN_COEFF_WIDTH
    point_bits = 2 * tick_width + DEFAULT_BUS_COUNT * DEFAULT_BUS_WIDTH
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
            f"  configured:      channels={channel_count} edges={max_edges} scan_rows={max_scan_points} tick_width={tick_width}",
            f"  row bits:        edge_template={scan_row_bits} scan_pair_plus_bus={point_bits}",
            f"  LUT estimate:    {configured_lut_est:.0f}/{lut_total} ({configured_lut_est / lut_total * 100.0:.1f}%)",
            f"  at target:       with {max_scan_points} scan rows, approx max_edges={max_edges_if_points_fixed}",
            f"  at target:       with {max_edges} edges, approx max_scan_rows={max_points_if_edges_fixed}",
            f"  BRAM note:       scan row plus DA-value RAM would be about {bram_for_points}/{bram_total} 36Kb blocks if mapped to BRAM",
            "  final evidence:  use Vivado report_utilization after synthesis; this is a design-budget estimate.",
        ]
    )


def generate_pulse_streamer_core(
    *,
    module_name: str = "zlc_pulse_streamer",
    channel_count: int = len(DEFAULT_CHANNELS),
    max_edges: int = DEFAULT_MAX_EDGES,
    max_scan_points: int = DEFAULT_MAX_SCAN_POINTS,
    max_bus_segments: int = DEFAULT_MAX_BUS_SEGMENTS,
    tick_width: int = DEFAULT_TICK_WIDTH,
    coeff_width: int = DEFAULT_SCAN_COEFF_WIDTH,
    coeff_frac_bits: int = DEFAULT_SCAN_COEFF_FRAC_BITS,
    bus_count: int = DEFAULT_BUS_COUNT,
    bus_width: int = DEFAULT_BUS_WIDTH,
) -> str:
    """Return synthesizable Verilog for a runtime edge-table pulse streamer."""

    module_name = safe_identifier(module_name)
    channel_count = _positive_int(channel_count, "channel_count")
    tick_width = _positive_int(tick_width, "tick_width")
    edge_addr_width = _edge_addr_width(max_edges)
    scan_addr_width = _edge_addr_width(max_scan_points)
    bus_seg_addr_width = _edge_addr_width(max_bus_segments)
    template_path = Path(__file__).resolve().parents[3] / "fpga" / "pulse_streamer" / "zlc_pulse_streamer.v"
    if not template_path.exists():
        raise FileNotFoundError(f"pulse-streamer HDL template not found: {template_path}")
    text = template_path.read_text(encoding="utf-8")
    text = re.sub(r"\bmodule\s+zlc_pulse_streamer\b", f"module {module_name}", text, count=1)
    replacements = {
        "CHANNEL_COUNT": channel_count,
        "EDGE_ADDR_WIDTH": edge_addr_width,
        "TICK_WIDTH": tick_width,
        "SCAN_ADDR_WIDTH": scan_addr_width,
        "COEFF_WIDTH": _positive_int(coeff_width, "coeff_width"),
        "COEFF_FRAC_BITS": _positive_int(coeff_frac_bits, "coeff_frac_bits"),
        "BUS_COUNT": _positive_int(bus_count, "bus_count"),
        "BUS_INDEX_WIDTH": _edge_addr_width(bus_count),
        "BUS_WIDTH": _positive_int(bus_width, "bus_width"),
        "BUS_SEG_ADDR_WIDTH": bus_seg_addr_width,
    }
    for name, value in replacements.items():
        text = re.sub(
            rf"parameter integer {name} = \d+",
            f"parameter integer {name} = {int(value)}",
            text,
            count=1,
        )
    return text


def generate_pulse_streamer_top_example(
    *,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    core_module_name: str = "zlc_pulse_streamer",
    top_module_name: str = "zlc_pulse_streamer_top_example",
    max_edges: int = DEFAULT_MAX_EDGES,
    max_scan_points: int = DEFAULT_MAX_SCAN_POINTS,
    max_bus_segments: int = DEFAULT_MAX_BUS_SEGMENTS,
    tick_width: int = DEFAULT_TICK_WIDTH,
    coeff_width: int = DEFAULT_SCAN_COEFF_WIDTH,
    bus_count: int = DEFAULT_BUS_COUNT,
    bus_width: int = DEFAULT_BUS_WIDTH,
    probe_names: PulseStreamerProbeNames | None = None,
) -> str:
    """Return an example top module showing the VIO probe contract."""

    channels = list(channel_names(channels, "channels"))
    core_module_name = safe_identifier(core_module_name)
    top_module_name = safe_identifier(top_module_name)
    probe_names = probe_names or PulseStreamerProbeNames()
    reserved = {
        *CONTROL_PORTS,
        "out",
        "zlc_streamer_i",
        "zlc_vio_i",
        "zlc_running_led",
        "zlc_done_led",
        *(safe_identifier(name) for name in probe_names.__dict__.values()),
    }
    safe_channels = _safe_channel_identifiers(channels, reserved=reserved)
    edge_addr_width = _edge_addr_width(max_edges)
    scan_addr_width = _edge_addr_width(max_scan_points)
    bus_seg_addr_width = _edge_addr_width(max_bus_segments)
    bus_index_width = _edge_addr_width(bus_count)
    bus_counts_width = bus_count * (bus_seg_addr_width + 1)
    scan_bus_values_width = bus_count * bus_width
    assigns = "\n".join(f"    assign {name} = out[{index}];" for index, name in enumerate(safe_channels))
    outputs = "\n".join(f"    output wire {name}," for name in safe_channels)
    return f"""`timescale 1ns / 1ps
// Example top-level wrapper for zlc_pulse_streamer.
// Create a Vivado VIO IP named vio_0 with these probes:
//   probe_out0 {probe_names.reset}      width 1
//   probe_out1 {probe_names.start}      width 1
//   probe_out2 {probe_names.prog_we}    width 1
//   probe_out3 {probe_names.prog_addr}  width {edge_addr_width}
//   probe_out4 {probe_names.prog_tick}  width {tick_width}
//   probe_out5 {probe_names.prog_mask}  width {len(channels)}
//   probe_out6 {probe_names.prog_count} width {edge_addr_width + 1}
//   probe_out7 {probe_names.repeat_forever} width 1
//   probe_out8 {probe_names.loop_start_addr} width {edge_addr_width}
//   probe_out9 {probe_names.loop_end_tick} width {tick_width}
//   probe_out10 {probe_names.loop_count} width 32
//   probe_out11 {probe_names.prog_tick_x_coeff} width {coeff_width}
//   probe_out12 {probe_names.prog_tick_y_coeff} width {coeff_width}
//   probe_out13 {probe_names.scan_enable} width 1
//   probe_out14 {probe_names.scan_prog_we} width 1
//   probe_out15 {probe_names.scan_prog_addr} width {scan_addr_width}
//   probe_out16 {probe_names.scan_prog_x} width {tick_width}
//   probe_out17 {probe_names.scan_prog_y} width {tick_width}
//   probe_out18 {probe_names.scan_count} width {scan_addr_width + 1}
//   probe_out19 {probe_names.loop_end_x_coeff} width {coeff_width}
//   probe_out20 {probe_names.loop_end_y_coeff} width {coeff_width}
//   probe_out21 {probe_names.bus_prog_we} width 1
//   probe_out22 {probe_names.bus_prog_bus} width {bus_index_width}
//   probe_out23 {probe_names.bus_prog_addr} width {bus_seg_addr_width}
//   probe_out24 {probe_names.bus_prog_start_tick} width {tick_width}
//   probe_out25 {probe_names.bus_prog_stop_tick} width {tick_width}
//   probe_out26 {probe_names.bus_prog_start_value} width {bus_width}
//   probe_out27 {probe_names.bus_prog_stop_value} width {bus_width}
//   probe_out28 {probe_names.bus_prog_mode} width 2
//   probe_out29 {probe_names.bus_counts} width {bus_counts_width}
//   probe_out30 {probe_names.scan_prog_bus_values} width {scan_bus_values_width}
//   probe_in0  {probe_names.running}    width 1
//   probe_in1  {probe_names.done}       width 1

module {top_module_name}(
    input wire clk,
{outputs}
    output wire zlc_running_led,
    output wire zlc_done_led
);

    wire {probe_names.reset};
    wire {probe_names.start};
    wire {probe_names.prog_we};
    wire [{edge_addr_width - 1}:0] {probe_names.prog_addr};
    wire [{tick_width - 1}:0] {probe_names.prog_tick};
    wire [{len(channels) - 1}:0] {probe_names.prog_mask};
    wire [{edge_addr_width}:0] {probe_names.prog_count};
    wire {probe_names.repeat_forever};
    wire [{edge_addr_width - 1}:0] {probe_names.loop_start_addr};
    wire [{tick_width - 1}:0] {probe_names.loop_end_tick};
    wire signed [{coeff_width - 1}:0] {probe_names.loop_end_x_coeff};
    wire signed [{coeff_width - 1}:0] {probe_names.loop_end_y_coeff};
    wire [31:0] {probe_names.loop_count};
    wire signed [{coeff_width - 1}:0] {probe_names.prog_tick_x_coeff};
    wire signed [{coeff_width - 1}:0] {probe_names.prog_tick_y_coeff};
    wire {probe_names.scan_enable};
    wire {probe_names.scan_prog_we};
    wire [{scan_addr_width - 1}:0] {probe_names.scan_prog_addr};
    wire signed [{tick_width - 1}:0] {probe_names.scan_prog_x};
    wire signed [{tick_width - 1}:0] {probe_names.scan_prog_y};
    wire [{scan_bus_values_width - 1}:0] {probe_names.scan_prog_bus_values};
    wire [{scan_addr_width}:0] {probe_names.scan_count};
    wire {probe_names.bus_prog_we};
    wire [{bus_index_width - 1}:0] {probe_names.bus_prog_bus};
    wire [{bus_seg_addr_width - 1}:0] {probe_names.bus_prog_addr};
    wire [{tick_width - 1}:0] {probe_names.bus_prog_start_tick};
    wire [{tick_width - 1}:0] {probe_names.bus_prog_stop_tick};
    wire [{bus_width - 1}:0] {probe_names.bus_prog_start_value};
    wire [{bus_width - 1}:0] {probe_names.bus_prog_stop_value};
    wire [1:0] {probe_names.bus_prog_mode};
    wire [{bus_counts_width - 1}:0] {probe_names.bus_counts};
    wire [{len(channels) - 1}:0] out;
    wire [{bus_count * bus_width - 1}:0] bus_out;
    wire {probe_names.running};
    wire {probe_names.done};

{assigns}
    assign zlc_running_led = {probe_names.running};
    assign zlc_done_led = {probe_names.done};

    {core_module_name} #(
        .CHANNEL_COUNT({len(channels)}),
        .EDGE_ADDR_WIDTH({edge_addr_width}),
        .TICK_WIDTH({tick_width}),
        .SCAN_ADDR_WIDTH({scan_addr_width}),
        .COEFF_WIDTH({coeff_width}),
        .COEFF_FRAC_BITS({DEFAULT_SCAN_COEFF_FRAC_BITS}),
        .BUS_COUNT({bus_count}),
        .BUS_INDEX_WIDTH({bus_index_width}),
        .BUS_WIDTH({bus_width}),
        .BUS_SEG_ADDR_WIDTH({bus_seg_addr_width})
    ) zlc_streamer_i (
        .clk(clk),
        .reset({probe_names.reset}),
        .start({probe_names.start}),
        .prog_we({probe_names.prog_we}),
        .prog_addr({probe_names.prog_addr}),
        .prog_tick({probe_names.prog_tick}),
        .prog_tick_x_coeff({probe_names.prog_tick_x_coeff}),
        .prog_tick_y_coeff({probe_names.prog_tick_y_coeff}),
        .prog_mask({probe_names.prog_mask}),
        .prog_count({probe_names.prog_count}),
        .repeat_forever({probe_names.repeat_forever}),
        .loop_start_addr({probe_names.loop_start_addr}),
        .loop_end_tick({probe_names.loop_end_tick}),
        .loop_end_x_coeff({probe_names.loop_end_x_coeff}),
        .loop_end_y_coeff({probe_names.loop_end_y_coeff}),
        .loop_count({probe_names.loop_count}),
        .scan_enable({probe_names.scan_enable}),
        .scan_prog_we({probe_names.scan_prog_we}),
        .scan_prog_addr({probe_names.scan_prog_addr}),
        .scan_prog_x({probe_names.scan_prog_x}),
        .scan_prog_y({probe_names.scan_prog_y}),
        .scan_prog_bus_values({probe_names.scan_prog_bus_values}),
        .scan_count({probe_names.scan_count}),
        .bus_prog_we({probe_names.bus_prog_we}),
        .bus_prog_bus({probe_names.bus_prog_bus}),
        .bus_prog_addr({probe_names.bus_prog_addr}),
        .bus_prog_start_tick({probe_names.bus_prog_start_tick}),
        .bus_prog_stop_tick({probe_names.bus_prog_stop_tick}),
        .bus_prog_start_value({probe_names.bus_prog_start_value}),
        .bus_prog_stop_value({probe_names.bus_prog_stop_value}),
        .bus_prog_mode({probe_names.bus_prog_mode}),
        .bus_counts({probe_names.bus_counts}),
        .out(out),
        .bus_out(bus_out),
        .running({probe_names.running}),
        .done({probe_names.done})
    );

    vio_0 zlc_vio_i (
        .clk(clk),
        .probe_in0({probe_names.running}),
        .probe_in1({probe_names.done}),
        .probe_out0({probe_names.reset}),
        .probe_out1({probe_names.start}),
        .probe_out2({probe_names.prog_we}),
        .probe_out3({probe_names.prog_addr}),
        .probe_out4({probe_names.prog_tick}),
        .probe_out5({probe_names.prog_mask}),
        .probe_out6({probe_names.prog_count}),
        .probe_out7({probe_names.repeat_forever}),
        .probe_out8({probe_names.loop_start_addr}),
        .probe_out9({probe_names.loop_end_tick}),
        .probe_out10({probe_names.loop_count}),
        .probe_out11({probe_names.prog_tick_x_coeff}),
        .probe_out12({probe_names.prog_tick_y_coeff}),
        .probe_out13({probe_names.scan_enable}),
        .probe_out14({probe_names.scan_prog_we}),
        .probe_out15({probe_names.scan_prog_addr}),
        .probe_out16({probe_names.scan_prog_x}),
        .probe_out17({probe_names.scan_prog_y}),
        .probe_out18({probe_names.scan_count}),
        .probe_out19({probe_names.loop_end_x_coeff}),
        .probe_out20({probe_names.loop_end_y_coeff}),
        .probe_out21({probe_names.bus_prog_we}),
        .probe_out22({probe_names.bus_prog_bus}),
        .probe_out23({probe_names.bus_prog_addr}),
        .probe_out24({probe_names.bus_prog_start_tick}),
        .probe_out25({probe_names.bus_prog_stop_tick}),
        .probe_out26({probe_names.bus_prog_start_value}),
        .probe_out27({probe_names.bus_prog_stop_value}),
        .probe_out28({probe_names.bus_prog_mode}),
        .probe_out29({probe_names.bus_counts}),
        .probe_out30({probe_names.scan_prog_bus_values})
    );
endmodule
"""


def write_pulse_streamer_hdl_bundle(
    output_dir: str | Path,
    *,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    core_module_name: str = "zlc_pulse_streamer",
    top_module_name: str = "zlc_pulse_streamer_top_example",
    max_edges: int = DEFAULT_MAX_EDGES,
    max_scan_points: int = DEFAULT_MAX_SCAN_POINTS,
    max_bus_segments: int = DEFAULT_MAX_BUS_SEGMENTS,
    tick_width: int = DEFAULT_TICK_WIDTH,
) -> PulseStreamerHDLFiles:
    """Write the pulse-streamer core, an example top, and a manifest."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = list(channel_names(channels, "channels"))
    core_path = output_dir / f"{safe_identifier(core_module_name)}.v"
    top_path = output_dir / f"{safe_identifier(top_module_name)}.v"
    manifest_path = output_dir / "zlc_pulse_streamer.manifest.json"
    core_path.write_text(
        generate_pulse_streamer_core(
            module_name=core_module_name,
            channel_count=len(channels),
            max_edges=max_edges,
            max_scan_points=max_scan_points,
            max_bus_segments=max_bus_segments,
            tick_width=tick_width,
        ),
        encoding="utf-8",
        newline="\n",
    )
    top_path.write_text(
        generate_pulse_streamer_top_example(
            channels=channels,
            core_module_name=core_module_name,
            top_module_name=top_module_name,
            max_edges=max_edges,
            max_scan_points=max_scan_points,
            max_bus_segments=max_bus_segments,
            tick_width=tick_width,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema": "Zou_lab_control.neutral_atom.PulseStreamerHDL",
        "version": 1,
        "channels": channels,
        "max_edges": int(max_edges),
        "max_scan_points": int(max_scan_points),
        "max_bus_segments": int(max_bus_segments),
        "edge_addr_width": _edge_addr_width(max_edges),
        "scan_addr_width": _edge_addr_width(max_scan_points),
        "bus_seg_addr_width": _edge_addr_width(max_bus_segments),
        "bus_count": DEFAULT_BUS_COUNT,
        "bus_width": DEFAULT_BUS_WIDTH,
        "tick_width": int(tick_width),
        "core": core_path.name,
        "top_example": top_path.name,
        "probe_names": PulseStreamerProbeNames().__dict__,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return PulseStreamerHDLFiles(core_path, top_path, manifest_path)


def write_vivado_pulse_streamer_tcl(
    path: str | Path,
    action: str,
    *,
    program: RuntimeSequenceProgram | None = None,
    project: str | None = None,
    bitstream: str | None = None,
    probes: str | None = None,
    vio_filter: str = DEFAULT_VIO_FILTER,
    program_on_run: str | None = None,
    probe_names: PulseStreamerProbeNames | None = None,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_scan_points: int = DEFAULT_MAX_SCAN_POINTS,
    tick_width: int = DEFAULT_TICK_WIDTH,
    channel_count: int | None = None,
    timeout: float | None = None,
    poll_interval: float = 0.02,
) -> Path:
    """Write a Vivado Tcl action for the runtime pulse streamer."""

    action = _normalize_action(action)
    probe_names = probe_names or PulseStreamerProbeNames()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if program is not None:
        validate_pulse_streamer_program(
            program,
            max_edges=max_edges,
            max_scan_points=max_scan_points,
            tick_width=tick_width,
            channel_count=channel_count,
        )
    channel_count = len(program.channels) if channel_count is None and program is not None else _positive_int(channel_count or len(DEFAULT_CHANNELS), "channel_count")

    lines = _vivado_common_tcl(
        project=project,
        bitstream=bitstream,
        probes=probes,
        vio_filter=vio_filter,
        program_on_run=program_on_run,
        probe_names=probe_names,
    )
    if action == "prepare":
        if program is None:
            raise ValueError("prepare requires a RuntimeSequenceProgram.")
        lines.extend(_prepare_tcl(program, probe_names=probe_names))
    elif action == "fire":
        lines.extend(_fire_tcl(probe_names=probe_names))
    elif action == "wait_done":
        lines.extend(_wait_done_tcl(probe_names=probe_names, timeout=timeout, poll_interval=poll_interval))
    elif action in {"safe_state", "abort"}:
        lines.extend(_safe_state_tcl(probe_names=probe_names))
    else:
        raise ValueError(f"unknown pulse-streamer action {action!r}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class VivadoPulseStreamerSession:
    """Persistent Vivado/VIO transport for the runtime pulse streamer.

    The batch backend is reliable for bring-up, but it pays Vivado startup,
    hardware-target open, and probe discovery on every action.  This session
    performs that setup once, then executes prepare/fire/wait/safe Tcl snippets
    in the same Vivado Tcl process.
    """

    def __init__(
        self,
        *,
        state_dir: str | Path,
        vivado: str | None = None,
        project: str | None = None,
        bitstream: str | None = None,
        probes: str | None = None,
        vio_filter: str = DEFAULT_VIO_FILTER,
        program_on_run: str | None = None,
        probe_names: PulseStreamerProbeNames | None = None,
        max_edges: int | None = None,
        max_scan_points: int | None = None,
        tick_width: int | None = None,
        channel_count: int | None = None,
        startup_timeout: float = 90.0,
        action_timeout: float | None = None,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.program_path = self.state_dir / "prepared_program.json"
        self.vivado = vivado or os.environ.get("ZLC_PS_VIVADO_BIN", os.environ.get("ZLC_VIVADO_BIN", "vivado"))
        self.project, self.bitstream, self.probes = _resolve_vivado_artifacts(project=project, bitstream=bitstream, probes=probes)
        self.vio_filter = str(vio_filter)
        self.program_on_run = program_on_run if program_on_run is not None else _env_first("ZLC_PS_VIVADO_PROGRAM_ON_RUN", "ZLC_VIVADO_PROGRAM_ON_RUN")
        self.probe_names = probe_names or PulseStreamerProbeNames()
        self.max_edges = _env_int("ZLC_PS_MAX_EDGES", DEFAULT_MAX_EDGES) if max_edges is None else _positive_int(max_edges, "max_edges")
        self.max_scan_points = _env_int("ZLC_PS_MAX_SCAN_POINTS", DEFAULT_MAX_SCAN_POINTS) if max_scan_points is None else _positive_int(max_scan_points, "max_scan_points")
        self.tick_width = _env_int("ZLC_PS_TICK_WIDTH", DEFAULT_TICK_WIDTH) if tick_width is None else _positive_int(tick_width, "tick_width")
        self.channel_count = None if channel_count is None else _positive_int(channel_count, "channel_count")
        self.startup_timeout = float(startup_timeout)
        self.action_timeout = action_timeout
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._counter = 0
        self._closed = False
        self._uploaded_program: RuntimeSequenceProgram | None = None
        self._log_path = self.state_dir / "vivado_session.log"

    def start(self) -> "VivadoPulseStreamerSession":
        if self._process is not None:
            return self
        try:
            self._process = subprocess.Popen(
                [self.vivado, "-mode", "tcl", "-nolog", "-nojournal"],
                cwd=self.state_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            message = (
                f"Vivado executable was not found: {self.vivado!r}.\n"
                "Set ZLC_PS_VIVADO_BIN or ZLC_VIVADO_BIN to the full Vivado executable path."
            )
            self._write_action_log("vivado_session_start", message)
            raise RuntimeError(f"pulse-streamer could not start persistent Vivado. See {self.state_dir / 'vivado_session_start.log'}.") from exc
        self._reader = threading.Thread(target=self._read_stdout, name="zlc-vivado-session-reader", daemon=True)
        self._reader.start()
        init_lines = _vivado_common_tcl(
            project=self.project,
            bitstream=self.bitstream,
            probes=self.probes,
            vio_filter=self.vio_filter,
            program_on_run=self.program_on_run,
            probe_names=self.probe_names,
        )
        self._execute(init_lines, action="vivado_session_start", timeout=self.startup_timeout)
        return self

    def prepare(self, program: RuntimeSequenceProgram) -> None:
        channel_count = len(program.channels) if self.channel_count is None else self.channel_count
        validate_pulse_streamer_program(
            program,
            max_edges=self.max_edges,
            max_scan_points=self.max_scan_points,
            tick_width=self.tick_width,
            channel_count=channel_count,
        )
        self._write_program(program)
        previous = self._uploaded_program
        self._execute(_prepare_tcl(program, probe_names=self.probe_names, previous_program=previous), action="prepare", timeout=self.action_timeout)
        self._uploaded_program = program

    def fire(self, program: RuntimeSequenceProgram | None = None) -> None:
        if program is not None:
            self._write_program(program)
        self._execute(_fire_tcl(probe_names=self.probe_names), action="fire", timeout=self.action_timeout)
        (self.state_dir / "pulse_streamer_fire_time.txt").write_text(str(time.monotonic()), encoding="utf-8")

    def wait_done(self, program: RuntimeSequenceProgram | None = None, timeout: float | None = None) -> bool:
        if program is not None:
            self._write_program(program)
        read_timeout = None if timeout is None else float(timeout) + 5.0
        self._execute(
            _wait_done_tcl(probe_names=self.probe_names, timeout=timeout, poll_interval=0.02),
            action="wait_done",
            timeout=read_timeout,
        )
        return True

    def safe_state(self) -> None:
        self._execute(_safe_state_tcl(probe_names=self.probe_names), action="safe_state", timeout=self.action_timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write("exit\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
        self._process = None
        self._uploaded_program = None

    def _write_program(self, program: RuntimeSequenceProgram) -> None:
        self.program_path.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")
        (self.state_dir / "last_sequence_id.txt").write_text(program.sequence_id, encoding="utf-8")

    def _execute(self, lines: Sequence[str], *, action: str, timeout: float | None) -> str:
        self.start() if self._process is None else None
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("persistent Vivado session is not running.")
        self._counter += 1
        marker = f"ZLC_SESSION_{self._counter:06d}"
        script = self._wrap_tcl(lines, marker)
        try:
            process.stdin.write(script)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            message = f"persistent Vivado session stopped before {action}. See {self._log_path}."
            self._write_action_log(action, message)
            raise RuntimeError(message) from exc
        output = self._read_until_marker(marker, timeout=timeout)
        self._write_action_log(action, output)
        if f"{marker}_ERROR" in output:
            tail = _log_tail(output)
            message = f"persistent Vivado {action} failed. See {self.state_dir / (action + '.log')}."
            if tail:
                message = f"{message}\n\n--- {action}.log tail ---\n{tail}"
            raise RuntimeError(message)
        return output

    @staticmethod
    def _wrap_tcl(lines: Sequence[str], marker: str) -> str:
        body = "\n".join(lines)
        return (
            f"puts \"{marker}_BEGIN\"\n"
            "if {[catch {\n"
            f"{body}\n"
            "} zlc_session_result zlc_session_options]} {\n"
            f"    puts \"{marker}_ERROR $zlc_session_result\"\n"
            "    if {[dict exists $zlc_session_options -errorinfo]} { puts [dict get $zlc_session_options -errorinfo] }\n"
            "} else {\n"
            f"    puts \"{marker}_OK\"\n"
            "}\n"
            f"puts \"{marker}_END\"\n"
            "flush stdout\n"
        )

    def _read_until_marker(self, marker: str, *, timeout: float | None) -> str:
        deadline = None if timeout is None else time.monotonic() + max(0.1, float(timeout))
        lines: list[str] = []
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                self.close()
                raise TimeoutError(f"persistent Vivado action timed out waiting for {marker}.")
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError(f"persistent Vivado action timed out waiting for {marker}.") from exc
            if item is None:
                raise RuntimeError("persistent Vivado exited unexpectedly.")
            lines.append(item)
            if f"{marker}_END" in item:
                return "".join(lines)

    def _read_stdout(self) -> None:
        assert self._process is not None
        stdout = self._process.stdout
        if stdout is None:
            self._queue.put(None)
            return
        with self._log_path.open("a", encoding="utf-8", errors="replace") as log:
            for line in stdout:
                log.write(line)
                log.flush()
                self._queue.put(line)
        self._queue.put(None)

    def _write_action_log(self, action: str, text: str) -> None:
        (self.state_dir / f"{action}.log").write_text(text, encoding="utf-8", errors="replace")


def run_action(
    action: str,
    *,
    program_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    vivado: str = "vivado",
    dry_run: bool = False,
    timeout: float | None = None,
    max_edges: int | None = None,
    max_scan_points: int | None = None,
    tick_width: int | None = None,
    channel_count: int | None = None,
) -> Path | None:
    """Run one pulse-streamer action from a sequencer-server command."""

    action = _normalize_action(action)
    state = Path(state_dir or os.environ.get("ZLC_STATE_DIR", "."))
    state.mkdir(parents=True, exist_ok=True)
    program = _read_program(program_path) if action == "prepare" else None
    if action == "wait_done" and timeout is None:
        timeout = _optional_float(os.environ.get("ZLC_TIMEOUT"))
    max_edges = _env_int("ZLC_PS_MAX_EDGES", DEFAULT_MAX_EDGES) if max_edges is None else max_edges
    max_scan_points = _env_int("ZLC_PS_MAX_SCAN_POINTS", DEFAULT_MAX_SCAN_POINTS) if max_scan_points is None else max_scan_points
    tick_width = _env_int("ZLC_PS_TICK_WIDTH", DEFAULT_TICK_WIDTH) if tick_width is None else tick_width
    if channel_count is None:
        channel_count = _env_int("ZLC_PS_CHANNEL_COUNT", len(program.channels) if program is not None else len(DEFAULT_CHANNELS))
    project, bitstream, probes = _resolve_vivado_artifacts(project=None, bitstream=None, probes=None)
    tcl_path = write_vivado_pulse_streamer_tcl(
        state / f"pulse_streamer_{action}.tcl",
        action,
        program=program,
        project=project,
        bitstream=bitstream,
        probes=probes,
        vio_filter=os.environ.get("ZLC_PS_VIO_FILTER", os.environ.get("ZLC_VIO_FILTER", DEFAULT_VIO_FILTER)),
        program_on_run=_env_first("ZLC_PS_VIVADO_PROGRAM_ON_RUN", "ZLC_VIVADO_PROGRAM_ON_RUN"),
        max_edges=max_edges,
        max_scan_points=max_scan_points,
        tick_width=tick_width,
        channel_count=channel_count,
        timeout=timeout,
    )
    if dry_run:
        return tcl_path
    result = _run_vivado(vivado, tcl_path, state=state, timeout=None)
    log_path = state / f"pulse_streamer_{action}.log"
    log_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        message = f"pulse-streamer {action} failed with code {result.returncode}. See {log_path}."
        tail = _log_tail(result.stdout)
        if tail:
            message = f"{message}\n\n--- {log_path.name} tail ---\n{tail}"
        raise RuntimeError(message)
    if action == "fire":
        (state / "pulse_streamer_fire_time.txt").write_text(str(time.monotonic()), encoding="utf-8")
    return tcl_path


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare/fire a runtime-programmable FPGA pulse streamer.")
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "fire",
            "wait_done",
            "safe_state",
            "abort",
            "generate_hdl",
            "capacity_estimate",
            "infer_channel_count",
            "infer_channels",
            "infer_trigger_channels",
        ],
        nargs="?",
        default=None,
    )
    parser.add_argument("--program", default=None, help="RuntimeSequenceProgram JSON path. Defaults to ZLC_SEQUENCE_PROGRAM.")
    parser.add_argument("--state-dir", default=None, help="State/log directory. Defaults to ZLC_STATE_DIR.")
    parser.add_argument("--vivado", default=os.environ.get("ZLC_PS_VIVADO_BIN", os.environ.get("ZLC_VIVADO_BIN", "vivado")))
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only write the generated Tcl file.")
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--max-scan-points", type=int, default=None)
    parser.add_argument("--max-bus-segments", type=int, default=None)
    parser.add_argument("--resource-target-pct", type=float, default=None)
    parser.add_argument("--tick-width", type=int, default=None)
    parser.add_argument("--channel-count", type=int, default=None)
    parser.add_argument("--output-dir", default="generated_pulse_streamer")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--xdc", default=None, help="Completed pulse-streamer XDC used to infer ch00.. width.")
    parser.add_argument("--default-count", type=int, default=DEFAULT_FPGA_CHANNEL_COUNT)
    parser.add_argument("--max-channel-count", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    action = args.action or os.environ.get("ZLC_SEQUENCER_ACTION", "prepare")
    if action == "generate_hdl":
        files = write_pulse_streamer_hdl_bundle(
            args.output_dir,
            channels=args.channels,
            max_edges=args.max_edges or _env_int("ZLC_PS_MAX_EDGES", DEFAULT_MAX_EDGES),
            max_scan_points=args.max_scan_points or _env_int("ZLC_PS_MAX_SCAN_POINTS", DEFAULT_MAX_SCAN_POINTS),
            max_bus_segments=args.max_bus_segments or _env_int("ZLC_PS_MAX_BUS_SEGMENTS", DEFAULT_MAX_BUS_SEGMENTS),
            tick_width=args.tick_width or _env_int("ZLC_PS_TICK_WIDTH", DEFAULT_TICK_WIDTH),
        )
        print(files.core_path)
        print(files.top_example_path)
        print(files.manifest_path)
        return 0
    if action == "capacity_estimate":
        print(capacity_estimate_text(
            channel_count=args.channel_count or DEFAULT_FPGA_CHANNEL_COUNT,
            max_edges=args.max_edges or _env_int("ZLC_PS_MAX_EDGES", DEFAULT_MAX_EDGES),
            max_scan_points=args.max_scan_points or _env_int("ZLC_PS_MAX_SCAN_POINTS", DEFAULT_MAX_SCAN_POINTS),
            tick_width=args.tick_width or _env_int("ZLC_PS_TICK_WIDTH", DEFAULT_TICK_WIDTH),
            target_pct=args.resource_target_pct if args.resource_target_pct is not None else _env_float("ZLC_PS_RESOURCE_TARGET_PCT", 70.0),
        ))
        return 0
    if action == "infer_channel_count":
        print(infer_xdc_channel_count(args.xdc, default=args.default_count, max_count=args.max_channel_count))
        return 0
    if action == "infer_channels":
        print(" ".join(infer_xdc_channels(args.xdc, default=args.default_count, max_count=args.max_channel_count)))
        return 0
    if action == "infer_trigger_channels":
        print(" ".join(infer_xdc_trigger_channels(args.xdc, default=args.default_count, max_count=args.max_channel_count)))
        return 0
    run_action(
        action,
        program_path=args.program,
        state_dir=args.state_dir,
        vivado=args.vivado,
        dry_run=args.dry_run,
        timeout=args.timeout,
        max_edges=args.max_edges,
        max_scan_points=args.max_scan_points,
        tick_width=args.tick_width,
        channel_count=args.channel_count,
    )
    return 0


def _vivado_common_tcl(
    *,
    project: str | None,
    bitstream: str | None,
    probes: str | None,
    vio_filter: str,
    program_on_run: str | None,
    probe_names: PulseStreamerProbeNames,
) -> list[str]:
    project_line = "set project [env_or ZLC_PS_VIVADO_PROJECT [env_or ZLC_VIVADO_PROJECT \"\"]]" if project is None else f"set project {{{project}}}"
    bitstream_line = "set bitstream [env_or ZLC_PS_VIVADO_BIT [env_or ZLC_VIVADO_BIT \"\"]]" if bitstream is None else f"set bitstream {{{bitstream}}}"
    probes_line = "set probes [env_or ZLC_PS_VIVADO_LTX [env_or ZLC_VIVADO_LTX \"\"]]" if probes is None else f"set probes {{{probes}}}"
    program_line = (
        "set program_on_run [env_or ZLC_PS_VIVADO_PROGRAM_ON_RUN [env_or ZLC_VIVADO_PROGRAM_ON_RUN \"0\"]]"
        if program_on_run is None
        else f"set program_on_run {{{program_on_run}}}"
    )
    return [
        "proc env_or {name default} {",
        "    if {[info exists ::env($name)]} { return $::env($name) }",
        "    return $default",
        "}",
        project_line,
        bitstream_line,
        probes_line,
        f"set vio_filter {{{vio_filter}}}",
        "set hw_server_url [env_or ZLC_PS_HW_SERVER_URL [env_or ZLC_HW_SERVER_URL \"\"]]",
        "set zlc_verbose_vio [env_or ZLC_PS_VERBOSE_VIO \"0\"]",
        program_line,
        "if {$project ne \"\" && ![file exists $project]} {",
        "    puts \"Vivado project not found; continuing without open_project: $project\"",
        "    set project \"\"",
        "}",
        "if {$probes eq \"\"} { error \"Vivado .ltx probe file is required for VIO control. Set ZLC_PS_VIVADO_LTX to the Probes file used when programming the FPGA, or check the address-switch XDC pin map and run fpga/build_and_program.bat.\" }",
        "if {![file exists $probes]} { error \"Vivado probe file not found: $probes\" }",
        "if {$program_on_run ne \"0\" && ($bitstream eq \"\" || ![file exists $bitstream])} {",
        "    error \"Vivado bitstream not found for programming: $bitstream\"",
        "}",
        "if {$project ne \"\" && [file exists $project]} {",
        "    if {[catch {open_project $project} zlc_open_project_error]} {",
        "        puts \"open_project failed; continuing without open_project: $zlc_open_project_error\"",
        "        set project \"\"",
        "    }",
        "}",
        "if {[llength [info commands load_features]]} { catch {load_features labtools} }",
        "if {[llength [info commands open_hw_manager]]} {",
        "    open_hw_manager",
        "} elseif {[llength [info commands open_hw]]} {",
        "    open_hw",
        "}",
        "if {![llength [info commands connect_hw_server]]} {",
        "    error \"Vivado hardware Tcl commands are unavailable. Install/enable Vivado LabTools or set ZLC_PS_VIVADO_BIN to a Vivado with Hardware Manager support.\"",
        "}",
        "if {$hw_server_url ne \"\"} {",
        "    connect_hw_server -url $hw_server_url",
        "} elseif {[catch {connect_hw_server} zlc_connect_error]} {",
        "    error \"connect_hw_server failed: $zlc_connect_error\"",
        "}",
        "catch {refresh_hw_server}",
        "set zlc_targets {}",
        "if {[catch {set zlc_targets [get_hw_targets]} zlc_target_error]} {",
        "    puts \"get_hw_targets failed after refresh: $zlc_target_error\"",
        "    set zlc_targets {}",
        "}",
        "puts \"Available hardware targets: $zlc_targets\"",
        "set zlc_target [lindex $zlc_targets 0]",
        "if {$zlc_target eq \"\"} { error \"No Vivado hardware target found. Check the USB/JTAG cable, board power, and hw_server connection.\" }",
        "current_hw_target $zlc_target",
        "if {[catch {open_hw_target $zlc_target} zlc_open_target_error]} {",
        "    puts \"open_hw_target failed: $zlc_open_target_error\"",
        "    catch {close_hw_target}",
        "    puts \"Retrying open_hw_target with -jtag_mode on.\"",
        "    if {[catch {open_hw_target -jtag_mode on $zlc_target} zlc_open_target_jtag_error]} {",
        "        error \"Vivado sees hardware target '$zlc_target' but no FPGA device could be opened. Check board power, JTAG chain/mode jumpers, power-source jumper, cable seating, then disconnect/reconnect hw_server. Last error: $zlc_open_target_jtag_error\"",
        "    }",
        "}",
        "set device [lindex [get_hw_devices] 0]",
        "if {$device eq \"\"} { error \"Vivado opened the hardware target but found no FPGA device. Check board power, JTAG chain/mode jumpers, power-source jumper, and Hardware Manager Auto Connect.\" }",
        "if {$program_on_run ne \"0\" && $bitstream ne \"\" && [file exists $bitstream]} {",
        "    set_property PROGRAM.FILE $bitstream $device",
        "    if {$probes ne \"\" && [file exists $probes]} {",
        "        set_property PROBES.FILE $probes $device",
        "        set_property FULL_PROBES.FILE $probes $device",
        "    }",
        "    program_hw_devices $device",
        "    refresh_hw_device $device",
        "} elseif {$probes ne \"\" && [file exists $probes]} {",
        "    set_property PROBES.FILE $probes $device",
        "    set_property FULL_PROBES.FILE $probes $device",
        "    refresh_hw_device $device",
        "}",
        "set available_vios [get_hw_vios -of_objects $device]",
        "set filtered_vios {}",
        "if {[catch {set filtered_vios [get_hw_vios -of_objects $device -filter $vio_filter]} zlc_vio_filter_error]} {",
        "    puts \"VIO filter '$vio_filter' failed: $zlc_vio_filter_error\"",
        "    set filtered_vios {}",
        "}",
        "set vio [lindex $filtered_vios 0]",
        "if {$vio eq \"\" && [llength $available_vios] == 1} {",
        "    puts \"VIO filter did not select a core; using the only available VIO core.\"",
        "    set vio [lindex $available_vios 0]",
        "}",
        "if {$vio eq \"\"} {",
        "    puts \"Available VIO cores:\"",
        "    foreach candidate $available_vios {",
        "        puts \"  NAME=[get_property NAME $candidate] CELL_NAME=[get_property CELL_NAME $candidate]\"",
        "    }",
        "    if {[llength $available_vios] == 0} {",
        "        error \"No VIO core was found on the FPGA. The current FPGA image is not the ZLC pulse-streamer bitstream, or the matching .ltx probes were not loaded. Program zlc_pulse_streamer_top_address_switch.bit with zlc_pulse_streamer_top_address_switch.ltx, or set ZLC_PS_VIVADO_LTX to the exact Probes file used in Vivado Program Device.\"",
        "    }",
        "    error \"No VIO core matched filter '$vio_filter'.\"",
        "}",
        "proc zlc_list_probes {vio} {",
        "    puts \"Available probes on matched VIO:\"",
        "    foreach candidate [get_hw_probes -of_objects $vio] {",
        "        set line \"\"",
        "        foreach prop {NAME PROBE_TYPE DIRECTION INPUT_VALUE OUTPUT_VALUE WIDTH} {",
        "            if {![catch {get_property $prop $candidate} value]} { append line \" $prop=$value\" }",
        "        }",
        "        puts \"  $line\"",
        "    }",
        "}",
        "proc zlc_probe {vio names} {",
        "    global zlc_probe_cache",
        "    set cache_key \"$vio|[join $names {|}]\"",
        "    if {[info exists zlc_probe_cache($cache_key)]} { return $zlc_probe_cache($cache_key) }",
        "    set matches {}",
        "    foreach name $names {",
        "        set probe [lindex [get_hw_probes $name -of_objects $vio] 0]",
        "        if {$probe ne \"\"} { lappend matches $probe }",
        "    }",
        "    if {[llength $matches] == 1} {",
        "        set zlc_probe_cache($cache_key) [lindex $matches 0]",
        "        return [lindex $matches 0]",
        "    }",
        "    if {[llength $matches] == 0} {",
        "        foreach candidate [get_hw_probes -of_objects $vio] {",
        "            set candidate_name [get_property NAME $candidate]",
        "            foreach name $names {",
        "                if {$candidate_name eq $name || [string match \"*/$name\" $candidate_name] || [string match \"*$name*\" $candidate_name]} {",
        "                    lappend matches $candidate",
        "                    break",
        "                }",
        "            }",
        "        }",
        "    }",
        "    set unique_matches {}",
        "    foreach probe $matches {",
        "        if {[lsearch -exact $unique_matches $probe] < 0} { lappend unique_matches $probe }",
        "    }",
        "    if {[llength $unique_matches] == 1} {",
        "        set zlc_probe_cache($cache_key) [lindex $unique_matches 0]",
        "        return [lindex $unique_matches 0]",
        "    }",
        "    zlc_list_probes $vio",
        "    if {[llength $unique_matches] > 1} { error \"VIO probe aliases '$names' matched multiple probes.\" }",
        "    error \"VIO probe aliases '$names' were not found.\"",
        "}",
        "proc zlc_stage_probe {vio name value} {",
        "    global zlc_verbose_vio",
        "    set probe [zlc_probe $vio $name]",
        "    set_property OUTPUT_VALUE_RADIX UNSIGNED $probe",
        "    set_property OUTPUT_VALUE $value $probe",
        "    if {$zlc_verbose_vio ne \"\" && $zlc_verbose_vio ne \"0\"} {",
        "        puts \"ZLC pulse-streamer VIO: [lindex $name 0]=$value\"",
        "    }",
        "    return $probe",
        "}",
        "proc zlc_commit_probes {probes} {",
        "    set unique_probes {}",
        "    foreach probe $probes {",
        "        if {[lsearch -exact $unique_probes $probe] < 0} { lappend unique_probes $probe }",
        "    }",
        "    if {[llength $unique_probes] > 0} { commit_hw_vio $unique_probes }",
        "}",
        "proc zlc_set_probe {vio name value} {",
        "    zlc_commit_probes [list [zlc_stage_probe $vio $name $value]]",
        "}",
        "proc zlc_read_probe {vio name} {",
        "    refresh_hw_vio $vio",
        "    set probe [zlc_probe $vio $name]",
        "    return [get_property INPUT_VALUE $probe]",
        "}",
        "proc zlc_output_probe_bool {vio name} {",
        "    refresh_hw_vio $vio",
        "    set probe [zlc_probe $vio $name]",
        "    set value [get_property OUTPUT_VALUE $probe]",
        "    if {$value eq \"\"} { return 0 }",
        "    if {[string is integer -strict $value]} { return [expr {int($value) != 0}] }",
        "    if {[regexp {([01])$} $value _ bit]} { return [expr {int($bit) != 0}] }",
        "    return 0",
        "}",
        "proc zlc_probe_value_bool {value} {",
        "    if {$value eq \"\"} { return 0 }",
        "    if {[string is integer -strict $value]} { return [expr {int($value) != 0}] }",
        "    if {[regexp {([01])$} $value _ bit]} { return [expr {int($bit) != 0}] }",
        "    return 0",
        "}",
        f"set zlc_reset_probe {{{probe_names.reset} probe_out0}}",
        f"set zlc_start_probe {{{probe_names.start} probe_out1}}",
        f"set zlc_prog_we_probe {{{probe_names.prog_we} probe_out2}}",
        f"set zlc_prog_addr_probe {{{probe_names.prog_addr} probe_out3}}",
        f"set zlc_prog_tick_probe {{{probe_names.prog_tick} probe_out4}}",
        f"set zlc_prog_mask_probe {{{probe_names.prog_mask} probe_out5}}",
        f"set zlc_prog_count_probe {{{probe_names.prog_count} probe_out6}}",
        f"set zlc_repeat_forever_probe {{{probe_names.repeat_forever} probe_out7}}",
        f"set zlc_loop_start_addr_probe {{{probe_names.loop_start_addr} probe_out8}}",
        f"set zlc_loop_end_tick_probe {{{probe_names.loop_end_tick} probe_out9}}",
        f"set zlc_loop_count_probe {{{probe_names.loop_count} probe_out10}}",
        f"set zlc_prog_tick_x_coeff_probe {{{probe_names.prog_tick_x_coeff} probe_out11}}",
        f"set zlc_prog_tick_y_coeff_probe {{{probe_names.prog_tick_y_coeff} probe_out12}}",
        f"set zlc_scan_enable_probe {{{probe_names.scan_enable} probe_out13}}",
        f"set zlc_scan_prog_we_probe {{{probe_names.scan_prog_we} probe_out14}}",
        f"set zlc_scan_prog_addr_probe {{{probe_names.scan_prog_addr} probe_out15}}",
        f"set zlc_scan_prog_x_probe {{{probe_names.scan_prog_x} probe_out16}}",
        f"set zlc_scan_prog_y_probe {{{probe_names.scan_prog_y} probe_out17}}",
        f"set zlc_scan_count_probe {{{probe_names.scan_count} probe_out18}}",
        f"set zlc_loop_end_x_coeff_probe {{{probe_names.loop_end_x_coeff} probe_out19}}",
        f"set zlc_loop_end_y_coeff_probe {{{probe_names.loop_end_y_coeff} probe_out20}}",
        f"set zlc_bus_prog_we_probe {{{probe_names.bus_prog_we} probe_out21}}",
        f"set zlc_bus_prog_bus_probe {{{probe_names.bus_prog_bus} probe_out22}}",
        f"set zlc_bus_prog_addr_probe {{{probe_names.bus_prog_addr} probe_out23}}",
        f"set zlc_bus_prog_start_tick_probe {{{probe_names.bus_prog_start_tick} probe_out24}}",
        f"set zlc_bus_prog_stop_tick_probe {{{probe_names.bus_prog_stop_tick} probe_out25}}",
        f"set zlc_bus_prog_start_value_probe {{{probe_names.bus_prog_start_value} probe_out26}}",
        f"set zlc_bus_prog_stop_value_probe {{{probe_names.bus_prog_stop_value} probe_out27}}",
        f"set zlc_bus_prog_mode_probe {{{probe_names.bus_prog_mode} probe_out28}}",
        f"set zlc_bus_counts_probe {{{probe_names.bus_counts} probe_out29}}",
        f"set zlc_scan_prog_bus_values_probe {{{probe_names.scan_prog_bus_values} probe_out30}}",
        f"set zlc_running_probe {{{probe_names.running} probe_in0}}",
        f"set zlc_done_probe {{{probe_names.done} probe_in1}}",
    ]


def _prepare_edge_indices(program: RuntimeSequenceProgram, previous_program: RuntimeSequenceProgram | None = None) -> list[int]:
    if not program.ticks:
        return []
    full = list(range(len(program.ticks)))
    if previous_program is None:
        return full
    if list(previous_program.channels) != list(program.channels):
        return full
    if float(previous_program.clock_hz) != float(program.clock_hz):
        return full
    previous_x_coeffs = list(previous_program.tick_x_coeffs or [0 for _ in previous_program.ticks])
    previous_y_coeffs = list(previous_program.tick_y_coeffs or [0 for _ in previous_program.ticks])
    current_x_coeffs = list(program.tick_x_coeffs or [0 for _ in program.ticks])
    current_y_coeffs = list(program.tick_y_coeffs or [0 for _ in program.ticks])
    changed = {
        index
        for index, (tick, mask) in enumerate(zip(program.ticks, program.masks))
        if index >= len(previous_program.ticks)
        or index >= len(previous_program.masks)
        or index >= len(previous_x_coeffs)
        or index >= len(previous_y_coeffs)
        or index >= len(current_x_coeffs)
        or index >= len(current_y_coeffs)
        or int(previous_program.ticks[index]) != int(tick)
        or int(previous_program.masks[index]) != int(mask)
        or int(previous_x_coeffs[index]) != int(current_x_coeffs[index])
        or int(previous_y_coeffs[index]) != int(current_y_coeffs[index])
    }
    loop_start_index = int(program.loop_start_index) if program.ticks else 0
    critical = {0, max(0, min(loop_start_index, len(program.ticks) - 1)), len(program.ticks) - 1}
    return sorted(index for index in (changed | critical) if 0 <= index < len(program.ticks))


def _prepare_scan_indices(program: RuntimeSequenceProgram, previous_program: RuntimeSequenceProgram | None = None) -> list[int]:
    points = list(getattr(program, "scan_points", None) or [])
    if not points:
        return []
    values = _scan_bus_values_for_program(program)
    full = list(range(len(points)))
    if previous_program is None:
        return full
    previous_points = list(getattr(previous_program, "scan_points", None) or [])
    previous_values = _scan_bus_values_for_program(previous_program)
    if len(previous_points) != len(points) or len(previous_values) != len(values):
        return full
    return [
        index
        for index, point in enumerate(points)
        if index >= len(previous_points)
        or int(previous_points[index][0]) != int(point[0])
        or int(previous_points[index][1]) != int(point[1])
        or int(previous_values[index]) != int(values[index])
    ]


def _scan_bus_values_for_program(program: RuntimeSequenceProgram) -> list[int]:
    points = list(getattr(program, "scan_points", None) or [])
    values = list(getattr(program, "scan_bus_values", None) or [])
    if not points:
        return []
    if not values:
        return [0 for _ in points]
    return [int(value) for value in values]


def _bus_segment_mode_value(mode: str) -> int:
    mode = str(mode).strip().lower()
    if mode == "edge":
        return 1
    if mode == "ramp":
        return 2
    raise ValueError(f"unsupported bus segment mode {mode!r}.")


def _bus_segments_by_bus(program: RuntimeSequenceProgram, *, bus_count: int = DEFAULT_BUS_COUNT) -> list[list[object]]:
    groups: list[list[object]] = [[] for _ in range(bus_count)]
    for segment in list(getattr(program, "bus_segments", None) or []):
        bus_index = int(getattr(segment, "bus_index", segment.get("bus_index") if isinstance(segment, Mapping) else 0))
        if 0 <= bus_index < bus_count:
            groups[bus_index].append(segment)
    for bus_segments in groups:
        bus_segments.sort(
            key=lambda segment: (
                int(getattr(segment, "start_tick", segment.get("start_tick") if isinstance(segment, Mapping) else 0)),
                int(getattr(segment, "stop_tick", segment.get("stop_tick") if isinstance(segment, Mapping) else 0)),
            )
        )
    return groups


def _bus_counts_word(program: RuntimeSequenceProgram, *, bus_count: int = DEFAULT_BUS_COUNT, addr_width: int = 6) -> int:
    word = 0
    field_width = int(addr_width) + 1
    for bus_index, segments in enumerate(_bus_segments_by_bus(program, bus_count=bus_count)):
        word |= int(len(segments)) << (bus_index * field_width)
    return word


def _prepare_bus_rows(
    program: RuntimeSequenceProgram,
    previous_program: RuntimeSequenceProgram | None = None,
    *,
    bus_count: int = DEFAULT_BUS_COUNT,
) -> list[tuple[int, int, object]]:
    groups = _bus_segments_by_bus(program, bus_count=bus_count)
    previous_groups = _bus_segments_by_bus(previous_program, bus_count=bus_count) if previous_program is not None else None
    rows: list[tuple[int, int, object]] = []
    for bus_index, segments in enumerate(groups):
        previous_segments = [] if previous_groups is None else previous_groups[bus_index]
        if previous_groups is not None and len(previous_segments) == len(segments):
            changed = [
                addr
                for addr, segment in enumerate(segments)
                if getattr(previous_segments[addr], "to_dict", lambda: previous_segments[addr])()
                != getattr(segment, "to_dict", lambda: segment)()
            ]
        else:
            changed = list(range(len(segments)))
        for addr in changed:
            rows.append((bus_index, addr, segments[addr]))
    return rows


def _prepare_tcl(
    program: RuntimeSequenceProgram,
    *,
    probe_names: PulseStreamerProbeNames,
    previous_program: RuntimeSequenceProgram | None = None,
) -> list[str]:
    loop_end_tick = int(program.loop_end_tick) if int(program.loop_end_tick) > 0 else (int(program.ticks[-1]) if program.ticks else 0)
    loop_start_index = int(program.loop_start_index) if program.ticks else 0
    loop_count = max(1, int(program.loop_count))
    tick_x_coeffs = list(program.tick_x_coeffs or [0 for _ in program.ticks])
    tick_y_coeffs = list(program.tick_y_coeffs or [0 for _ in program.ticks])
    scan_points = list(program.scan_points or [])
    scan_bus_values = _scan_bus_values_for_program(program)
    scan_indices = _prepare_scan_indices(program, previous_program)
    write_indices = _prepare_edge_indices(program, previous_program)
    bus_rows = _prepare_bus_rows(program, previous_program)
    bus_counts_word = _bus_counts_word(program)
    first_write_index = write_indices[0] if write_indices else None
    remaining_write_indices = write_indices[1:] if first_write_index is not None else []
    lines = [
        "set zlc_prog_we_toggle_value [zlc_output_probe_bool $vio $zlc_prog_we_probe]",
        "set zlc_scan_prog_we_toggle_value [zlc_output_probe_bool $vio $zlc_scan_prog_we_probe]",
        "set zlc_bus_prog_we_toggle_value [zlc_output_probe_bool $vio $zlc_bus_prog_we_probe]",
    ]
    lines.extend(
        [
        "set zlc_batch {}",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_reset_probe 1]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_start_probe 0]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_repeat_forever_probe {1 if program.repeat_forever else 0}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_loop_start_addr_probe {loop_start_index}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_loop_end_tick_probe {loop_end_tick}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_loop_count_probe {loop_count}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_loop_end_x_coeff_probe {_signed_to_unsigned(int(getattr(program, 'loop_end_x_coeff', 0)), DEFAULT_SCAN_COEFF_WIDTH)}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_loop_end_y_coeff_probe {_signed_to_unsigned(int(getattr(program, 'loop_end_y_coeff', 0)), DEFAULT_SCAN_COEFF_WIDTH)}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_scan_enable_probe {1 if scan_points else 0}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_scan_count_probe {len(scan_points)}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_counts_probe {bus_counts_word}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_count_probe {len(program.ticks)}]",
        "zlc_commit_probes $zlc_batch",
        "set zlc_prepare_reset_settle_ms [expr {max(1, int([env_or ZLC_PS_PREPARE_RESET_SETTLE_MS 5]))}]",
        "after $zlc_prepare_reset_settle_ms",
        ]
    )
    if first_write_index is not None:
        tick = int(program.ticks[first_write_index])
        mask = int(program.masks[first_write_index])
        x_coeff = _signed_to_unsigned(int(tick_x_coeffs[first_write_index]), DEFAULT_SCAN_COEFF_WIDTH)
        y_coeff = _signed_to_unsigned(int(tick_y_coeffs[first_write_index]), DEFAULT_SCAN_COEFF_WIDTH)
        lines.extend(
            [
                "set zlc_prog_we_toggle_value [expr {$zlc_prog_we_toggle_value ? 0 : 1}]",
                "set zlc_batch {}",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_addr_probe {first_write_index}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_tick_probe {tick}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_mask_probe {mask}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_tick_x_coeff_probe {x_coeff}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_tick_y_coeff_probe {y_coeff}]",
                "lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_we_probe $zlc_prog_we_toggle_value]",
                "zlc_commit_probes $zlc_batch",
            ]
        )
    for index in remaining_write_indices:
        tick = int(program.ticks[index])
        mask = int(program.masks[index])
        x_coeff = _signed_to_unsigned(int(tick_x_coeffs[index]), DEFAULT_SCAN_COEFF_WIDTH)
        y_coeff = _signed_to_unsigned(int(tick_y_coeffs[index]), DEFAULT_SCAN_COEFF_WIDTH)
        lines.extend(
            [
                "set zlc_prog_we_toggle_value [expr {$zlc_prog_we_toggle_value ? 0 : 1}]",
                "set zlc_batch {}",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_addr_probe {index}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_tick_probe {tick}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_mask_probe {mask}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_tick_x_coeff_probe {x_coeff}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_tick_y_coeff_probe {y_coeff}]",
                "lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_we_probe $zlc_prog_we_toggle_value]",
                "zlc_commit_probes $zlc_batch",
            ]
        )
    for bus_index, bus_addr, segment in bus_rows:
        start_tick = int(getattr(segment, "start_tick", segment.get("start_tick") if isinstance(segment, Mapping) else 0))
        stop_tick = int(getattr(segment, "stop_tick", segment.get("stop_tick", start_tick) if isinstance(segment, Mapping) else start_tick))
        start_value = int(getattr(segment, "start_value", segment.get("start_value", 0) if isinstance(segment, Mapping) else 0))
        stop_value = int(getattr(segment, "stop_value", segment.get("stop_value", start_value) if isinstance(segment, Mapping) else start_value))
        mode_value = _bus_segment_mode_value(str(getattr(segment, "mode", segment.get("mode", "edge") if isinstance(segment, Mapping) else "edge")))
        lines.extend(
            [
                "set zlc_bus_prog_we_toggle_value [expr {$zlc_bus_prog_we_toggle_value ? 0 : 1}]",
                "set zlc_batch {}",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_prog_bus_probe {bus_index}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_prog_addr_probe {bus_addr}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_prog_start_tick_probe {start_tick}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_prog_stop_tick_probe {stop_tick}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_prog_start_value_probe {start_value}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_prog_stop_value_probe {stop_value}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_prog_mode_probe {mode_value}]",
                "lappend zlc_batch [zlc_stage_probe $vio $zlc_bus_prog_we_probe $zlc_bus_prog_we_toggle_value]",
                "zlc_commit_probes $zlc_batch",
            ]
        )
    for index in scan_indices:
        x_tick = _signed_to_unsigned(int(scan_points[index][0]), DEFAULT_TICK_WIDTH)
        y_tick = _signed_to_unsigned(int(scan_points[index][1]), DEFAULT_TICK_WIDTH)
        bus_values = int(scan_bus_values[index]) if index < len(scan_bus_values) else 0
        lines.extend(
            [
                "set zlc_scan_prog_we_toggle_value [expr {$zlc_scan_prog_we_toggle_value ? 0 : 1}]",
                "set zlc_batch {}",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_scan_prog_addr_probe {index}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_scan_prog_x_probe {x_tick}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_scan_prog_y_probe {y_tick}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_scan_prog_bus_values_probe {bus_values}]",
                "lappend zlc_batch [zlc_stage_probe $vio $zlc_scan_prog_we_probe $zlc_scan_prog_we_toggle_value]",
                "zlc_commit_probes $zlc_batch",
            ]
        )
    lines.extend(
        [
            "set zlc_batch {}",
            "lappend zlc_batch [zlc_stage_probe $vio $zlc_reset_probe 0]",
            "zlc_commit_probes $zlc_batch",
            f"puts \"ZLC pulse-streamer prepared sequence {program.sequence_id} wrote {len(write_indices)}/{len(program.ticks)} edge rows, {len(scan_indices)}/{len(scan_points)} scan points, and {len(bus_rows)}/{len(getattr(program, 'bus_segments', None) or [])} bus segments reset_settle_ms=$zlc_prepare_reset_settle_ms repeat_forever={int(program.repeat_forever)} scan={int(bool(scan_points))} loop_start={loop_start_index} loop_end={loop_end_tick} loop_count={loop_count} bus_counts={bus_counts_word}\"",
        ]
    )
    return lines


def _fire_tcl(*, probe_names: PulseStreamerProbeNames) -> list[str]:
    return [
        "if {[zlc_output_probe_bool $vio $zlc_start_probe]} {",
        "    zlc_set_probe $vio $zlc_start_probe 0",
        "}",
        "set zlc_batch {}",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_reset_probe 0]",
        "zlc_commit_probes $zlc_batch",
        "set zlc_fire_arm_delay_ms [expr {max(0, int([env_or ZLC_PS_FIRE_ARM_DELAY_MS 5]))}]",
        "set zlc_fire_hold_ms [expr {max(1, int([env_or ZLC_PS_FIRE_HOLD_MS 5]))}]",
        "if {$zlc_fire_arm_delay_ms > 0} { after $zlc_fire_arm_delay_ms }",
        "set zlc_batch {}",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_start_probe 1]",
        "zlc_commit_probes $zlc_batch",
        "after $zlc_fire_hold_ms",
        "zlc_set_probe $vio $zlc_start_probe 0",
        "after 1",
        "set zlc_running_value [zlc_read_probe $vio $zlc_running_probe]",
        "set zlc_done_value [zlc_read_probe $vio $zlc_done_probe]",
        "puts \"ZLC pulse-streamer start pulse sent running=$zlc_running_value done=$zlc_done_value arm_delay_ms=$zlc_fire_arm_delay_ms hold_ms=$zlc_fire_hold_ms\"",
        "if {![zlc_probe_value_bool $zlc_running_value] && ![zlc_probe_value_bool $zlc_done_value]} {",
        "    error \"ZLC pulse-streamer start was not observed by FPGA: running=$zlc_running_value done=$zlc_done_value. Check bitstream/probes/clock/reset and try a longer ZLC_PS_FIRE_ARM_DELAY_MS.\"",
        "}",
    ]


def _wait_done_tcl(*, probe_names: PulseStreamerProbeNames, timeout: float | None, poll_interval: float) -> list[str]:
    timeout_expr = str(float(timeout)) if timeout is not None else "[env_or ZLC_TIMEOUT 10.0]"
    poll_ms = max(1, int(round(float(poll_interval) * 1000)))
    return [
        f"set timeout_s {timeout_expr}",
        "set deadline [expr {[clock milliseconds] + int(1000.0 * double($timeout_s))}]",
        "while {1} {",
        "    set done_value [zlc_read_probe $vio $zlc_done_probe]",
        "    if {$done_value ne \"0\"} {",
        "        puts \"ZLC pulse-streamer done\"",
        "        break",
        "    }",
        "    if {[clock milliseconds] > $deadline} { error \"ZLC pulse-streamer wait_done timed out.\" }",
        f"    after {poll_ms}",
        "}",
    ]


def _safe_state_tcl(*, probe_names: PulseStreamerProbeNames) -> list[str]:
    return [
        "set zlc_batch {}",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_start_probe 0]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_repeat_forever_probe 0]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_scan_enable_probe 0]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_reset_probe 1]",
        "zlc_commit_probes $zlc_batch",
        "puts \"ZLC pulse-streamer safe state requested\"",
    ]


def _read_program(program_path: str | Path | None) -> RuntimeSequenceProgram:
    path = Path(program_path or os.environ["ZLC_SEQUENCE_PROGRAM"])
    return RuntimeSequenceProgram.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _run_vivado(vivado: str, tcl_path: Path, *, state: Path, timeout: float | None):
    try:
        return subprocess.run(
            [vivado, "-mode", "batch", "-source", str(tcl_path)],
            cwd=state,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        log_path = state / f"{tcl_path.stem}.log"
        message = (
            f"Vivado executable was not found: {vivado!r}.\n"
            "Set ZLC_PS_VIVADO_BIN or ZLC_VIVADO_BIN to the full Vivado executable path."
        )
        log_path.write_text(message, encoding="utf-8", errors="replace")
        raise RuntimeError(f"pulse-streamer could not start Vivado. See {log_path}.") from exc


def _normalize_action(action: str) -> str:
    action = str(action).strip()
    return "safe_state" if action == "abort" else action


def _edge_addr_width(max_edges: int) -> int:
    max_edges = _positive_int(max_edges, "max_edges")
    return max(1, (max_edges - 1).bit_length())


def _apply_scan_tick(base_tick: int, x_coeff: int, y_coeff: int, x_tick: int, y_tick: int, frac_bits: int) -> int:
    return int(base_tick) + ((int(x_coeff) * int(x_tick) + int(y_coeff) * int(y_tick)) >> int(frac_bits))


def _signed_to_unsigned(value: int, width: int) -> int:
    value = int(value)
    width = int(width)
    limit = 1 << width
    if value < 0:
        value = limit + value
    if value < 0 or value >= limit:
        raise ValueError(f"signed value does not fit {width} bits.")
    return value


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


def _optional_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _log_tail(text: str, *, max_lines: int = 80, max_chars: int = 12_000) -> str:
    tail = "\n".join(str(text).splitlines()[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default if raw is None or raw == "" else raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    out = float(default if raw is None or raw == "" else raw)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _path_is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _default_project_root() -> Path:
    root = _env_first("ZLC_PS_PROJECT_ROOT")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[3] / "fpga" / "build"


def _old_pulse_streamer_build_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "fpga" / "pulse_streamer" / "build"


def _safe_project_dir(path: str | Path | None) -> Path:
    fallback = _default_project_root() / DEFAULT_PROJECT_NAME
    candidate = Path(path) if path else fallback
    if _path_is_under(candidate, _old_pulse_streamer_build_dir()):
        return fallback
    return candidate


def _safe_vivado_artifact_env(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    if _path_is_under(candidate, _old_pulse_streamer_build_dir()):
        return None
    return str(candidate)


def _default_vivado_artifact_paths(project_dir: Path) -> tuple[Path, Path, Path]:
    impl_dir = project_dir / f"{DEFAULT_PROJECT_NAME}.runs" / "impl_1"
    return (
        project_dir / f"{DEFAULT_PROJECT_NAME}.xpr",
        impl_dir / f"{DEFAULT_TOP_NAME}.bit",
        impl_dir / f"{DEFAULT_TOP_NAME}.ltx",
    )


def _resolve_vivado_artifacts(
    *,
    project: str | None,
    bitstream: str | None,
    probes: str | None,
) -> tuple[str | None, str | None, str | None]:
    project_dir = _safe_project_dir(_env_first("ZLC_PS_PROJECT_DIR"))
    default_project, default_bitstream, default_probes = _default_vivado_artifact_paths(project_dir)
    resolved_project = project if project is not None else _safe_vivado_artifact_env(_env_first("ZLC_PS_VIVADO_PROJECT", "ZLC_VIVADO_PROJECT"))
    resolved_bitstream = bitstream if bitstream is not None else _safe_vivado_artifact_env(_env_first("ZLC_PS_VIVADO_BIT", "ZLC_VIVADO_BIT"))
    resolved_probes = probes if probes is not None else _safe_vivado_artifact_env(_env_first("ZLC_PS_VIVADO_LTX", "ZLC_VIVADO_LTX"))
    if not resolved_project:
        resolved_project = str(default_project)
    if not resolved_bitstream:
        resolved_bitstream = str(default_bitstream)
    if not resolved_probes:
        resolved_probes = str(default_probes)
    return resolved_project, resolved_bitstream, resolved_probes


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FPGA_CHANNEL_COUNT",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_SCAN_POINTS",
    "DEFAULT_SCAN_COEFF_FRAC_BITS",
    "DEFAULT_SCAN_COEFF_WIDTH",
    "DEFAULT_TICK_WIDTH",
    "DEFAULT_VIO_FILTER",
    "PulseStreamerHDLFiles",
    "PulseStreamerProbeNames",
    "build_arg_parser",
    "capacity_estimate_text",
    "generate_pulse_streamer_core",
    "generate_pulse_streamer_top_example",
    "hardware_channel_names",
    "infer_xdc_channel_count",
    "infer_xdc_channel_labels",
    "infer_xdc_channel_pins",
    "infer_xdc_channels",
    "infer_xdc_trigger_channels",
    "main",
    "run_action",
    "validate_pulse_streamer_program",
    "write_pulse_streamer_hdl_bundle",
    "write_vivado_pulse_streamer_tcl",
]
