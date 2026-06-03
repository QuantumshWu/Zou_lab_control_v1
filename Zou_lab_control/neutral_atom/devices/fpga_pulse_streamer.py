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
import os
import queue
import subprocess
import threading
import time
from typing import Mapping, Sequence

from .sequencer import RuntimeSequenceProgram
from ..timing import channel_names
from ..timing.verilog import CONTROL_PORTS, safe_identifier


DEFAULT_CHANNELS = ["trap", "cooling", "probe", "qcm_trigger"]
DEFAULT_VIO_FILTER = 'CELL_NAME=~"*vio*"'
DEFAULT_MAX_EDGES = 1024
DEFAULT_TICK_WIDTH = 32


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
    running: str = "zlc_running"
    done: str = "zlc_done"


@dataclass(frozen=True)
class PulseStreamerHDLFiles:
    core_path: Path
    top_example_path: Path
    manifest_path: Path


def validate_pulse_streamer_program(
    program: RuntimeSequenceProgram,
    *,
    max_edges: int = DEFAULT_MAX_EDGES,
    tick_width: int = DEFAULT_TICK_WIDTH,
    channel_count: int | None = None,
) -> None:
    """Validate that a runtime edge table fits the fixed FPGA streamer."""

    max_edges = _positive_int(max_edges, "max_edges")
    tick_width = _positive_int(tick_width, "tick_width")
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
    last_tick = -1
    for tick in program.ticks:
        tick = int(tick)
        if tick <= last_tick:
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
        if loop_end_tick <= int(program.ticks[loop_start_index]):
            raise ValueError("program loop_end_tick must be after the loop start tick.")
        if loop_end_tick > int(program.ticks[-1]):
            raise ValueError("program loop_end_tick must not exceed the uploaded final tick.")
        if loop_end_tick > tick_limit:
            raise ValueError(f"program loop_end_tick {loop_end_tick} does not fit {tick_width} bits.")


def generate_pulse_streamer_core(
    *,
    module_name: str = "zlc_pulse_streamer",
    channel_count: int = len(DEFAULT_CHANNELS),
    max_edges: int = DEFAULT_MAX_EDGES,
    tick_width: int = DEFAULT_TICK_WIDTH,
) -> str:
    """Return synthesizable Verilog for a runtime edge-table pulse streamer."""

    module_name = safe_identifier(module_name)
    channel_count = _positive_int(channel_count, "channel_count")
    tick_width = _positive_int(tick_width, "tick_width")
    edge_addr_width = _edge_addr_width(max_edges)
    actual_edges = 1 << edge_addr_width
    return f"""`timescale 1ns / 1ps
// Generated by Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer.
// Runtime-programmable edge-table pulse streamer.

module {module_name} #(
    parameter integer CHANNEL_COUNT = {channel_count},
    parameter integer EDGE_ADDR_WIDTH = {edge_addr_width},
    parameter integer TICK_WIDTH = {tick_width}
)(
    input wire clk,
    input wire reset,
    input wire start,
    input wire prog_we,
    input wire [EDGE_ADDR_WIDTH-1:0] prog_addr,
    input wire [TICK_WIDTH-1:0] prog_tick,
    input wire [CHANNEL_COUNT-1:0] prog_mask,
    input wire [EDGE_ADDR_WIDTH:0] prog_count,
    input wire repeat_forever,
    input wire [EDGE_ADDR_WIDTH-1:0] loop_start_addr,
    input wire [TICK_WIDTH-1:0] loop_end_tick,
    input wire [31:0] loop_count,
    output wire [CHANNEL_COUNT-1:0] out,
    output reg running = 1'b0,
    output reg done = 1'b0
);

    localparam integer MAX_EDGES = {actual_edges};

    (* ram_style = "distributed" *) reg [TICK_WIDTH-1:0] tick_mem [0:MAX_EDGES-1];
    (* ram_style = "distributed" *) reg [CHANNEL_COUNT-1:0] mask_mem [0:MAX_EDGES-1];

    reg [CHANNEL_COUNT-1:0] state_mask = {{CHANNEL_COUNT{{1'b0}}}};
    reg [TICK_WIDTH-1:0] time_count = {{TICK_WIDTH{{1'b0}}}};
    reg [TICK_WIDTH-1:0] final_tick = {{TICK_WIDTH{{1'b0}}}};
    reg [EDGE_ADDR_WIDTH:0] edge_index = {{(EDGE_ADDR_WIDTH + 1){{1'b0}}}};
    reg [EDGE_ADDR_WIDTH:0] active_count = {{(EDGE_ADDR_WIDTH + 1){{1'b0}}}};
    reg repeat_forever_active = 1'b0;
    reg [EDGE_ADDR_WIDTH-1:0] loop_start_active = {{EDGE_ADDR_WIDTH{{1'b0}}}};
    reg [TICK_WIDTH-1:0] loop_end_active = {{TICK_WIDTH{{1'b0}}}};
    reg [31:0] loop_count_active = 32'd1;
    reg [31:0] loops_remaining = 32'd1;

    reg start_sync = 1'b0;
    reg start_prev = 1'b0;

    wire start_event = start_sync != start_prev;
    wire [EDGE_ADDR_WIDTH-1:0] edge_addr = edge_index[EDGE_ADDR_WIDTH-1:0];

    assign out = state_mask;

    always @(posedge clk) begin
        start_sync <= start;
        start_prev <= start_sync;

        if (reset && prog_we) begin
            tick_mem[prog_addr] <= prog_tick;
            mask_mem[prog_addr] <= prog_mask;
        end

        if (reset) begin
            running <= 1'b0;
            done <= 1'b0;
            state_mask <= {{CHANNEL_COUNT{{1'b0}}}};
            time_count <= {{TICK_WIDTH{{1'b0}}}};
            final_tick <= {{TICK_WIDTH{{1'b0}}}};
            edge_index <= {{(EDGE_ADDR_WIDTH + 1){{1'b0}}}};
            active_count <= {{(EDGE_ADDR_WIDTH + 1){{1'b0}}}};
            repeat_forever_active <= 1'b0;
            loop_start_active <= {{EDGE_ADDR_WIDTH{{1'b0}}}};
            loop_end_active <= {{TICK_WIDTH{{1'b0}}}};
            loop_count_active <= 32'd1;
            loops_remaining <= 32'd1;
        end else if (start_event && !running) begin
            running <= (prog_count != 0);
            done <= (prog_count == 0);
            final_tick <= (prog_count == 0) ? {{TICK_WIDTH{{1'b0}}}} : tick_mem[prog_count[EDGE_ADDR_WIDTH-1:0] - 1'b1];
            active_count <= prog_count;
            repeat_forever_active <= repeat_forever;
            loop_start_active <= loop_start_addr;
            loop_end_active <= loop_end_tick;
            loop_count_active <= (loop_count == 0) ? 32'd1 : loop_count;
            loops_remaining <= (loop_count == 0) ? 32'd1 : loop_count;
            if (prog_count != 0 && tick_mem[0] == {{TICK_WIDTH{{1'b0}}}}) begin
                state_mask <= mask_mem[0];
                time_count <= {{{{(TICK_WIDTH-1){{1'b0}}}}, 1'b1}};
                edge_index <= {{{{EDGE_ADDR_WIDTH{{1'b0}}}}, 1'b1}};
            end else begin
                state_mask <= {{CHANNEL_COUNT{{1'b0}}}};
                time_count <= {{TICK_WIDTH{{1'b0}}}};
                edge_index <= {{(EDGE_ADDR_WIDTH + 1){{1'b0}}}};
            end
        end else if (running) begin
            if (loop_count_active > 32'd1 && loops_remaining > 32'd1 && time_count >= loop_end_active) begin
                state_mask <= mask_mem[loop_start_active];
                time_count <= tick_mem[loop_start_active] + 1'b1;
                edge_index <= {{1'b0, loop_start_active}} + 1'b1;
                loops_remaining <= loops_remaining - 1'b1;
            end else if (time_count >= final_tick) begin
                if (repeat_forever_active) begin
                    if (tick_mem[0] == {{TICK_WIDTH{{1'b0}}}}) begin
                        state_mask <= mask_mem[0];
                        time_count <= {{{{(TICK_WIDTH-1){{1'b0}}}}, 1'b1}};
                        edge_index <= {{{{EDGE_ADDR_WIDTH{{1'b0}}}}, 1'b1}};
                    end else begin
                        state_mask <= {{CHANNEL_COUNT{{1'b0}}}};
                        time_count <= {{TICK_WIDTH{{1'b0}}}};
                        edge_index <= {{(EDGE_ADDR_WIDTH + 1){{1'b0}}}};
                    end
                    loops_remaining <= loop_count_active;
                    done <= 1'b0;
                end else begin
                    running <= 1'b0;
                    done <= 1'b1;
                    state_mask <= {{CHANNEL_COUNT{{1'b0}}}};
                end
            end else begin
                if (edge_index < active_count && time_count == tick_mem[edge_addr]) begin
                    state_mask <= mask_mem[edge_addr];
                    edge_index <= edge_index + 1'b1;
                end
                time_count <= time_count + 1'b1;
            end
        end
    end
endmodule
"""


def generate_pulse_streamer_top_example(
    *,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    core_module_name: str = "zlc_pulse_streamer",
    top_module_name: str = "zlc_pulse_streamer_top_example",
    max_edges: int = DEFAULT_MAX_EDGES,
    tick_width: int = DEFAULT_TICK_WIDTH,
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
    wire [31:0] {probe_names.loop_count};
    wire [{len(channels) - 1}:0] out;
    wire {probe_names.running};
    wire {probe_names.done};

{assigns}
    assign zlc_running_led = {probe_names.running};
    assign zlc_done_led = {probe_names.done};

    {core_module_name} #(
        .CHANNEL_COUNT({len(channels)}),
        .EDGE_ADDR_WIDTH({edge_addr_width}),
        .TICK_WIDTH({tick_width})
    ) zlc_streamer_i (
        .clk(clk),
        .reset({probe_names.reset}),
        .start({probe_names.start}),
        .prog_we({probe_names.prog_we}),
        .prog_addr({probe_names.prog_addr}),
        .prog_tick({probe_names.prog_tick}),
        .prog_mask({probe_names.prog_mask}),
        .prog_count({probe_names.prog_count}),
        .repeat_forever({probe_names.repeat_forever}),
        .loop_start_addr({probe_names.loop_start_addr}),
        .loop_end_tick({probe_names.loop_end_tick}),
        .loop_count({probe_names.loop_count}),
        .out(out),
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
        .probe_out10({probe_names.loop_count})
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
        "edge_addr_width": _edge_addr_width(max_edges),
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
        validate_pulse_streamer_program(program, max_edges=max_edges, tick_width=tick_width, channel_count=channel_count)
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
        tick_width: int | None = None,
        channel_count: int | None = None,
        startup_timeout: float = 90.0,
        action_timeout: float | None = None,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.program_path = self.state_dir / "prepared_program.json"
        self.vivado = vivado or os.environ.get("ZLC_PS_VIVADO_BIN", os.environ.get("ZLC_VIVADO_BIN", "vivado"))
        self.project = project if project is not None else _env_first("ZLC_PS_VIVADO_PROJECT", "ZLC_VIVADO_PROJECT")
        self.bitstream = bitstream if bitstream is not None else _env_first("ZLC_PS_VIVADO_BIT", "ZLC_VIVADO_BIT")
        self.probes = probes if probes is not None else _env_first("ZLC_PS_VIVADO_LTX", "ZLC_VIVADO_LTX")
        self.vio_filter = str(vio_filter)
        self.program_on_run = program_on_run if program_on_run is not None else _env_first("ZLC_PS_VIVADO_PROGRAM_ON_RUN", "ZLC_VIVADO_PROGRAM_ON_RUN")
        self.probe_names = probe_names or PulseStreamerProbeNames()
        self.max_edges = _env_int("ZLC_PS_MAX_EDGES", DEFAULT_MAX_EDGES) if max_edges is None else _positive_int(max_edges, "max_edges")
        self.tick_width = _env_int("ZLC_PS_TICK_WIDTH", DEFAULT_TICK_WIDTH) if tick_width is None else _positive_int(tick_width, "tick_width")
        self.channel_count = None if channel_count is None else _positive_int(channel_count, "channel_count")
        self.startup_timeout = float(startup_timeout)
        self.action_timeout = action_timeout
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._counter = 0
        self._closed = False
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
        validate_pulse_streamer_program(program, max_edges=self.max_edges, tick_width=self.tick_width, channel_count=channel_count)
        self._write_program(program)
        self._execute(_prepare_tcl(program, probe_names=self.probe_names), action="prepare", timeout=self.action_timeout)

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
    tick_width = _env_int("ZLC_PS_TICK_WIDTH", DEFAULT_TICK_WIDTH) if tick_width is None else tick_width
    if channel_count is None:
        channel_count = _env_int("ZLC_PS_CHANNEL_COUNT", len(program.channels) if program is not None else len(DEFAULT_CHANNELS))
    tcl_path = write_vivado_pulse_streamer_tcl(
        state / f"pulse_streamer_{action}.tcl",
        action,
        program=program,
        project=_env_first("ZLC_PS_VIVADO_PROJECT", "ZLC_VIVADO_PROJECT"),
        bitstream=_env_first("ZLC_PS_VIVADO_BIT", "ZLC_VIVADO_BIT"),
        probes=_env_first("ZLC_PS_VIVADO_LTX", "ZLC_VIVADO_LTX"),
        vio_filter=os.environ.get("ZLC_PS_VIO_FILTER", os.environ.get("ZLC_VIO_FILTER", DEFAULT_VIO_FILTER)),
        program_on_run=_env_first("ZLC_PS_VIVADO_PROGRAM_ON_RUN", "ZLC_VIVADO_PROGRAM_ON_RUN"),
        max_edges=max_edges,
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
    parser.add_argument("action", choices=["prepare", "fire", "wait_done", "safe_state", "abort", "generate_hdl"], nargs="?", default=None)
    parser.add_argument("--program", default=None, help="RuntimeSequenceProgram JSON path. Defaults to ZLC_SEQUENCE_PROGRAM.")
    parser.add_argument("--state-dir", default=None, help="State/log directory. Defaults to ZLC_STATE_DIR.")
    parser.add_argument("--vivado", default=os.environ.get("ZLC_PS_VIVADO_BIN", os.environ.get("ZLC_VIVADO_BIN", "vivado")))
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only write the generated Tcl file.")
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--tick-width", type=int, default=None)
    parser.add_argument("--channel-count", type=int, default=None)
    parser.add_argument("--output-dir", default="generated_pulse_streamer")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    action = args.action or os.environ.get("ZLC_SEQUENCER_ACTION", "prepare")
    if action == "generate_hdl":
        files = write_pulse_streamer_hdl_bundle(
            args.output_dir,
            channels=args.channels,
            max_edges=args.max_edges or _env_int("ZLC_PS_MAX_EDGES", DEFAULT_MAX_EDGES),
            tick_width=args.tick_width or _env_int("ZLC_PS_TICK_WIDTH", DEFAULT_TICK_WIDTH),
        )
        print(files.core_path)
        print(files.top_example_path)
        print(files.manifest_path)
        return 0
    run_action(
        action,
        program_path=args.program,
        state_dir=args.state_dir,
        vivado=args.vivado,
        dry_run=args.dry_run,
        timeout=args.timeout,
        max_edges=args.max_edges,
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
        program_line,
        "if {$project ne \"\" && ![file exists $project]} {",
        "    puts \"Vivado project not found; continuing without open_project: $project\"",
        "    set project \"\"",
        "}",
        "if {$probes eq \"\"} { error \"Vivado .ltx probe file is required for VIO control. Set ZLC_PS_VIVADO_LTX to the Probes file used when programming the FPGA, or run fpga/build_and_program.bat after completing the 40ch XDC.\" }",
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
        "        error \"No VIO core was found on the FPGA. The current FPGA image is not the ZLC pulse-streamer bitstream, or the matching .ltx probes were not loaded. Program zlc_pulse_streamer_top_40ch.bit with zlc_pulse_streamer_top_40ch.ltx, or set ZLC_PS_VIVADO_LTX to the exact Probes file used in Vivado Program Device.\"",
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
        "    set matches {}",
        "    foreach name $names {",
        "        set probe [lindex [get_hw_probes $name -of_objects $vio] 0]",
        "        if {$probe ne \"\"} { lappend matches $probe }",
        "    }",
        "    if {[llength $matches] == 1} { return [lindex $matches 0] }",
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
        "    if {[llength $unique_matches] == 1} { return [lindex $unique_matches 0] }",
        "    zlc_list_probes $vio",
        "    if {[llength $unique_matches] > 1} { error \"VIO probe aliases '$names' matched multiple probes.\" }",
        "    error \"VIO probe aliases '$names' were not found.\"",
        "}",
        "proc zlc_stage_probe {vio name value} {",
        "    set probe [zlc_probe $vio $name]",
        "    set_property OUTPUT_VALUE_RADIX UNSIGNED $probe",
        "    set_property OUTPUT_VALUE $value $probe",
        "    puts \"ZLC pulse-streamer VIO: [lindex $name 0]=$value\"",
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
        f"set zlc_running_probe {{{probe_names.running} probe_in0}}",
        f"set zlc_done_probe {{{probe_names.done} probe_in1}}",
    ]


def _prepare_tcl(program: RuntimeSequenceProgram, *, probe_names: PulseStreamerProbeNames) -> list[str]:
    loop_end_tick = int(program.loop_end_tick) if int(program.loop_end_tick) > 0 else (int(program.ticks[-1]) if program.ticks else 0)
    loop_start_index = int(program.loop_start_index) if program.ticks else 0
    loop_count = max(1, int(program.loop_count))
    lines = [
        "set zlc_batch {}",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_reset_probe 1]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_start_probe 0]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_we_probe 0]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_repeat_forever_probe {1 if program.repeat_forever else 0}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_loop_start_addr_probe {loop_start_index}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_loop_end_tick_probe {loop_end_tick}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_loop_count_probe {loop_count}]",
        f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_count_probe {len(program.ticks)}]",
        "zlc_commit_probes $zlc_batch",
    ]
    for index, (tick, mask) in enumerate(zip(program.ticks, program.masks)):
        lines.extend(
            [
                "set zlc_batch {}",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_addr_probe {index}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_tick_probe {int(tick)}]",
                f"lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_mask_probe {int(mask)}]",
                "lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_we_probe 1]",
                "zlc_commit_probes $zlc_batch",
            ]
        )
    lines.extend(
        [
            "set zlc_batch {}",
            "lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_we_probe 0]",
            "lappend zlc_batch [zlc_stage_probe $vio $zlc_reset_probe 0]",
            "zlc_commit_probes $zlc_batch",
            "set zlc_start_toggle_value 0",
            f"puts \"ZLC pulse-streamer prepared sequence {program.sequence_id} with {len(program.ticks)} edges repeat_forever={int(program.repeat_forever)} loop_start={loop_start_index} loop_end={loop_end_tick} loop_count={loop_count}\"",
        ]
    )
    return lines


def _fire_tcl(*, probe_names: PulseStreamerProbeNames) -> list[str]:
    return [
        "if {![info exists zlc_start_toggle_value]} { set zlc_start_toggle_value [zlc_output_probe_bool $vio $zlc_start_probe] }",
        "set zlc_start_next [expr {$zlc_start_toggle_value ? 0 : 1}]",
        "set zlc_batch {}",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_reset_probe 0]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_start_probe $zlc_start_next]",
        "zlc_commit_probes $zlc_batch",
        "set zlc_start_toggle_value $zlc_start_next",
        "puts \"ZLC pulse-streamer start toggle sent value=$zlc_start_next\"",
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
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_prog_we_probe 0]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_repeat_forever_probe 0]",
        "lappend zlc_batch [zlc_stage_probe $vio $zlc_reset_probe 1]",
        "zlc_commit_probes $zlc_batch",
        "set zlc_start_toggle_value 0",
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


def _safe_channel_identifiers(channels: Sequence[str], *, reserved: set[str]) -> list[str]:
    safe_channels = [safe_identifier(channel) for channel in channels]
    if len(set(safe_channels)) != len(safe_channels):
        raise ValueError("channel names collide after Verilog identifier sanitization.")
    collisions = sorted(set(safe_channels) & set(reserved))
    if collisions:
        raise ValueError(f"channel names collide with pulse-streamer top-level names: {collisions}")
    return safe_channels


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


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_EDGES",
    "DEFAULT_TICK_WIDTH",
    "DEFAULT_VIO_FILTER",
    "PulseStreamerHDLFiles",
    "PulseStreamerProbeNames",
    "build_arg_parser",
    "generate_pulse_streamer_core",
    "generate_pulse_streamer_top_example",
    "main",
    "run_action",
    "validate_pulse_streamer_program",
    "write_pulse_streamer_hdl_bundle",
    "write_vivado_pulse_streamer_tcl",
]
