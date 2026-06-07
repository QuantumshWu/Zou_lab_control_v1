"""Persistent Vivado JTAG-to-AXI runtime session for the FINAL pulse streamer.

Drives ``zlc_pulse_streamer_top`` + ``zlc_edge_streamer`` (1-tick FIFO prefetch +
2-bank streaming scan) over the JTAG-to-AXI master.  It holds one persistent
``vivado -mode tcl`` process, connects to the programmed FPGA, and:

  * packs the compiled program into the BRAM image (:mod:`fpga.pulse_streamer.host.image`):
    edges -> the 3 parallel TICK/COEFF/MASK BRAMs, the first two scan chunks ->
    the ping-pong banks, the bus segments -> the bus-image BRAM, scalars -> CTRL,
  * writes that image over AXI (``create_hw_axi_txn`` / ``run_hw_axi``),
  * drives the CTRL COMMAND/STATUS mailbox (LOAD -> the top's mini-loader copies
    the bus image into the engine + asserts LOADED; FIRE -> release reset + pulse
    start; SAFE -> halt + reset),
  * for an UNBOUNDED scan (N > 2*bank_size points) STREAMS: it polls the engine's
    CURSOR, and as each ping-pong bank is freed behind the cursor it rewrites that
    bank with the next chunk and re-arms its BANK_READY bit, so the scan-point
    count is limited only by host memory.  A late refill makes the engine STALL
    (STATUS underflow) -- it never emits a wrong point.

There is NO min-edge-spacing constraint: the engine is 1-tick seamless.

The Tcl execution is injectable (``tcl_executor``) so the whole
prepare/fire/stream/wait_done/safe_state flow is tested without Vivado or
hardware.  ``wait_done`` always bounds its poll, so the server can never busy-poll
forever.
"""

from __future__ import annotations

import math
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from fpga.pulse_streamer.host.image import (
    StreamerParams,
    CtrlWords,
    pack_program,
    scan_bank_words,
    region_bases,
    CMD_LOAD,
    CMD_FIRE,
    CMD_SAFE,
    CMD_RESET,
    STATUS_LOADED,
    STATUS_RUNNING,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_UNDERFLOW,
)

DEFAULT_RUNTIME_CLOCK_HZ = 50_000_000.0

# The default host geometry MUST match the built bitstream (create_project.tcl /
# host.image.solve_capacity for the 35T): 4096 edges + bank_size 2048.
DEFAULT_PARAMS = StreamerParams(max_edges=4096, bank_size=2048)


def _default_vivado() -> str:
    for name in ("ZLC_PS_VIVADO_BIN", "ZLC_VIVADO_BIN"):
        value = os.environ.get(name)
        if value:
            return value
    return "vivado"


def _default_artifact(suffix: str) -> str | None:
    """Default bit/ltx path from the in-repo final build (fpga/build/pulse_streamer)."""

    env = {
        ".bit": ("ZLC_PS_VIVADO_BIT", "ZLC_PS_BIT"),
        ".ltx": ("ZLC_PS_VIVADO_LTX", "ZLC_PS_LTX"),
    }.get(suffix, ())
    for name in env:
        value = os.environ.get(name)
        if value:
            return value
    root = os.environ.get("ZLC_PS_PROJECT_DIR")
    if not root:
        return None
    candidate = Path(root) / "pulse_streamer.runs" / "impl_1" / f"zlc_pulse_streamer_top{suffix}"
    return str(candidate) if candidate.exists() else None


class VivadoAxiStreamerSession:
    """Persistent Vivado hw_axi transport for the final pulse streamer."""

    def __init__(
        self,
        *,
        state_dir: str | Path,
        vivado: str | None = None,
        bitstream: str | None = None,
        probes: str | None = None,
        hw_server_url: str | None = None,
        program_on_start: bool = False,
        clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
        params: StreamerParams | None = None,
        startup_timeout: float = 180.0,
        action_timeout: float | None = 120.0,
        write_batch: int = 200,
        stream_poll_interval: float = 0.005,
        tcl_executor: Callable[[Sequence[str], str, float | None], str] | None = None,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.vivado = vivado or _default_vivado()
        self.bitstream = bitstream if bitstream is not None else _default_artifact(".bit")
        self.probes = probes if probes is not None else _default_artifact(".ltx")
        self.hw_server_url = hw_server_url or os.environ.get("ZLC_PS_HW_SERVER_URL") or os.environ.get("ZLC_HW_SERVER_URL") or ""
        self.program_on_start = bool(program_on_start)
        self.clock_hz = float(clock_hz)
        self.params = params or DEFAULT_PARAMS
        self.startup_timeout = float(startup_timeout)
        self.action_timeout = action_timeout
        self.write_batch = max(1, int(write_batch))
        self.stream_poll_interval = float(stream_poll_interval)

        self._pending: list[tuple[int, int]] = []  # (byte_addr, value) queued writes
        self._repeat_forever = False
        self._program = None          # last prepared program (for streaming refills)
        self._total_points = 0

        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._counter = 0
        self._closed = False
        self._log_path = self.state_dir / "vivado_axi_session.log"
        self._external_executor = tcl_executor

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "VivadoAxiStreamerSession":
        if self._external_executor is not None or self._process is not None:
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
            self._write_action_log("vivado_axi_session_start", message)
            raise RuntimeError(
                f"pulse-streamer session could not start persistent Vivado. See {self.state_dir / 'vivado_axi_session_start.log'}."
            ) from exc
        self._reader = threading.Thread(target=self._read_stdout, name="zlc-vivado-axi-reader", daemon=True)
        self._reader.start()
        self._run_tcl(self._init_tcl(), action="vivado_axi_session_start", timeout=self.startup_timeout)
        return self

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

    # ------------------------------------------------------------------ Tcl build
    def _init_tcl(self) -> list[str]:
        lines = [
            "if {[llength [info commands load_features]]} { catch {load_features labtools} }",
            "if {[llength [info commands open_hw_manager]]} { open_hw_manager } elseif {[llength [info commands open_hw]]} { open_hw }",
        ]
        if self.hw_server_url:
            lines.append(f"connect_hw_server -url {self.hw_server_url}")
        else:
            lines.append("if {[catch {connect_hw_server}]} { connect_hw_server }")
        lines += [
            "catch {refresh_hw_server}",
            "set zlc_targets [get_hw_targets]",
            'if {$zlc_targets eq ""} { error "No Vivado hardware target. Check the JTAG cable and board power." }',
            "current_hw_target [lindex $zlc_targets 0]",
            "if {[catch {open_hw_target}]} { catch {close_hw_target}; open_hw_target -jtag_mode on }",
            "current_hw_device [lindex [get_hw_devices] 0]",
        ]
        if self.probes:
            lines.append(f'set_property PROBES.FILE {{{self.probes}}} [current_hw_device]')
            lines.append(f'set_property FULL_PROBES.FILE {{{self.probes}}} [current_hw_device]')
        if self.program_on_start and self.bitstream:
            lines.append(f'set_property PROGRAM.FILE {{{self.bitstream}}} [current_hw_device]')
            lines.append("program_hw_devices [current_hw_device]")
        lines.append("refresh_hw_device [current_hw_device]")
        lines += [
            "set zlc_axi [get_hw_axis]",
            'if {$zlc_axi eq ""} { error "No JTAG-to-AXI (hw_axi) core found. Program the bitstream first (build_and_program.bat), and check the .ltx probes file." }',
            'puts "ZLC hw_axi cores: $zlc_axi"',
        ]
        return lines

    @staticmethod
    def _write_txn_tcl(byte_addr: int, value: int) -> list[str]:
        addr = f"{byte_addr & 0xFFFFFFFF:08X}"
        data = f"{value & 0xFFFFFFFF:08X}"
        return [
            f"create_hw_axi_txn zlc_w [get_hw_axis] -address {addr} -data {data} -len 1 -type write",
            "run_hw_axi zlc_w",
            "delete_hw_axi_txn zlc_w",
        ]

    @staticmethod
    def _read_txn_tcl(byte_addr: int, marker: str) -> list[str]:
        addr = f"{byte_addr & 0xFFFFFFFF:08X}"
        return [
            f"create_hw_axi_txn zlc_r [get_hw_axis] -address {addr} -len 1 -type read",
            "run_hw_axi zlc_r",
            f'puts "{marker} [get_property DATA [get_hw_axi_txns zlc_r]]"',
            "delete_hw_axi_txn zlc_r",
        ]

    # ------------------------------------------------------------- word I/O
    def _queue_word(self, word_offset: int, value: int) -> None:
        self._pending.append((int(word_offset) * 4, int(value) & 0xFFFFFFFF))

    def _read_word(self, word_offset: int) -> int:
        self._flush()
        marker = "ZLCDATA"
        out = self._run_tcl(self._read_txn_tcl(int(word_offset) * 4, marker), action="axi_read", timeout=self.action_timeout)
        return self._parse_read(out, marker)

    @staticmethod
    def _parse_read(output: str, marker: str) -> int:
        match = None
        for line in output.splitlines():
            if marker in line:
                match = line.split(marker, 1)[1].strip()
        if not match:
            raise RuntimeError("hw_axi read returned no DATA.")
        token = match.split()[-1].replace("0x", "").replace("0X", "")
        token = re.sub(r"[^0-9a-fA-F]", "", token) or "0"
        return int(token, 16) & 0xFFFFFFFF

    def _flush(self) -> None:
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        for start in range(0, len(pending), self.write_batch):
            batch = pending[start : start + self.write_batch]
            lines: list[str] = []
            for byte_addr, value in batch:
                lines.extend(self._write_txn_tcl(byte_addr, value))
            self._run_tcl(lines, action="axi_write", timeout=self.action_timeout)

    # --------------------------------------------------------------- sequencer API
    def prepare(self, program) -> None:
        """Pack the program into the BRAM image, upload it over AXI, then command
        the top's mini-loader to copy the bus image into the engine (LOAD)."""

        p = self.params
        points = list(getattr(program, "scan_points", []) or [])
        self._program = program
        self._total_points = len(points)
        self._repeat_forever = bool(getattr(program, "repeat_forever", False))
        if self._repeat_forever and self._total_points > 2 * p.bank_size:
            raise ValueError(
                "repeat_forever with a streaming scan (> 2*bank_size points) is not "
                "supported: the ping-pong banks are overwritten ahead of the cursor, so "
                "a wrap to point 0 cannot be served seamlessly. Use a finite scan for "
                f"unbounded streaming, or keep <= {2 * p.bank_size} points for repeat."
            )
        image = pack_program(program, p)
        # Halt + reset first so a prior run cannot drive outputs while we rewrite BRAM.
        self._command(CMD_SAFE)
        for word_offset in sorted(image):
            self._queue_word(word_offset, image[word_offset])
        # banks 0 and 1 are resident after pack_program -> arm both ready bits.
        self._queue_word(CtrlWords.BANK_READY, 0b11)
        self._flush()
        if not self._command(CMD_LOAD, wait_mask=STATUS_LOADED):
            raise RuntimeError(
                "pulse streamer did not report LOADED after the program upload "
                "(check the .bit/.ltx, the JTAG cable, and the STATUS word)."
            )

    def fire(self, program=None) -> None:
        if program is not None:
            self.prepare(program)
        self._command(CMD_FIRE)
        (self.state_dir / "fire_time.txt").write_text(str(time.monotonic()), encoding="utf-8")

    def wait_done(self, program=None, timeout: float | None = None) -> bool:
        """Poll to completion.  For an unbounded scan (N > 2*bank_size) this also
        STREAMS: it refills each ping-pong bank as the cursor frees it, so the host
        keeps the engine fed for the whole N-point sweep."""

        effective = float(timeout) if timeout is not None else float(self.action_timeout or 600.0)
        deadline = time.monotonic() + max(0.0, effective)
        p = self.params
        bank_size = p.bank_size
        total_chunks = max(1, math.ceil(self._total_points / bank_size)) if self._total_points else 1
        next_chunk = 2                # chunks 0,1 are already resident
        bank_ready = 0b11

        while True:
            status = self._read_word(CtrlWords.STATUS)
            if status & STATUS_DONE:
                return True
            if self._repeat_forever and (status & STATUS_RUNNING):
                return True
            if status & STATUS_ERROR:
                raise RuntimeError("pulse streamer reported STATUS_ERROR (bad image magic?).")

            # --- streaming refill: load the next chunk into the freed bank ---
            if next_chunk < total_chunks:
                cursor = self._read_word(CtrlWords.CURSOR)
                # chunk (next_chunk-2) lives in bank next_chunk%2; it is fully
                # consumed once cursor >= (next_chunk-1)*bank_size, freeing that bank.
                if cursor >= (next_chunk - 1) * bank_size:
                    bank = next_chunk % 2
                    refill = scan_bank_words(self._program, p, next_chunk)
                    bank_ready &= ~(1 << bank)
                    self._queue_word(CtrlWords.BANK_READY, bank_ready)   # de-arm during rewrite
                    for off in sorted(refill):
                        self._queue_word(off, refill[off])
                    bank_ready |= (1 << bank)
                    self._queue_word(CtrlWords.BANK_READY, bank_ready)   # re-arm
                    self._flush()
                    next_chunk += 1
                    continue   # re-check status/cursor promptly while streaming

            if time.monotonic() >= deadline:
                return False
            time.sleep(self.stream_poll_interval if next_chunk < total_chunks else 0.01)

    def safe_state(self) -> None:
        self._command(CMD_SAFE)

    # ------------------------------------------------------------------ mailbox
    def _command(self, command: int, *, wait_mask: int | None = None,
                 timeout: float | None = None) -> bool:
        """Drive a rising edge on the COMMAND word, optionally waiting for a STATUS
        mask.  The top edge-detects commands, so write 0 first to guarantee a clean
        0->cmd transition even if the same command was issued before."""

        self._queue_word(CtrlWords.COMMAND, 0)
        self._queue_word(CtrlWords.COMMAND, int(command) & 0xF)
        self._flush()
        if wait_mask is None:
            return True
        effective = float(timeout) if timeout is not None else float(self.action_timeout or 30.0)
        deadline = time.monotonic() + max(0.0, effective)
        while True:
            status = self._read_word(CtrlWords.STATUS)
            if status & STATUS_ERROR:
                raise RuntimeError("pulse streamer reported STATUS_ERROR (bad image magic?).")
            if (status & wait_mask) == wait_mask:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    # ------------------------------------------------------------------ Tcl plumbing
    def _run_tcl(self, lines: Sequence[str], *, action: str, timeout: float | None) -> str:
        if self._external_executor is not None:
            return self._external_executor(list(lines), action, timeout)
        return self._execute(lines, action=action, timeout=timeout)

    def _execute(self, lines: Sequence[str], *, action: str, timeout: float | None) -> str:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("persistent Vivado hw_axi session is not running.")
        self._counter += 1
        marker = f"ZLC_AXI_{self._counter:06d}"
        script = self._wrap_tcl(lines, marker)
        try:
            process.stdin.write(script)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            message = f"persistent Vivado hw_axi session stopped before {action}. See {self._log_path}."
            self._write_action_log(action, message)
            raise RuntimeError(message) from exc
        output = self._read_until_marker(marker, timeout=timeout)
        self._write_action_log(action, output)
        if f"{marker}_ERROR" in output:
            tail = "\n".join(output.splitlines()[-25:])
            raise RuntimeError(f"persistent Vivado hw_axi {action} failed. See {self.state_dir / (action + '.log')}.\n\n{tail}")
        return output

    @staticmethod
    def _wrap_tcl(lines: Sequence[str], marker: str) -> str:
        body = "\n".join(lines)
        return (
            f'puts "{marker}_BEGIN"\n'
            "if {[catch {\n"
            f"{body}\n"
            "} zlc_axi_result zlc_axi_options]} {\n"
            f'    puts "{marker}_ERROR $zlc_axi_result"\n'
            "    if {[dict exists $zlc_axi_options -errorinfo]} { puts [dict get $zlc_axi_options -errorinfo] }\n"
            "} else {\n"
            f'    puts "{marker}_OK"\n'
            "}\n"
            f'puts "{marker}_END"\n'
            "flush stdout\n"
        )

    def _read_until_marker(self, marker: str, *, timeout: float | None) -> str:
        deadline = None if timeout is None else time.monotonic() + max(0.1, float(timeout))
        lines: list[str] = []
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                self.close()
                raise TimeoutError(f"persistent Vivado hw_axi action timed out waiting for {marker}.")
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError(f"persistent Vivado hw_axi action timed out waiting for {marker}.") from exc
            if item is None:
                raise RuntimeError("persistent Vivado hw_axi process exited unexpectedly.")
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


__all__ = ["VivadoAxiStreamerSession", "DEFAULT_RUNTIME_CLOCK_HZ", "DEFAULT_PARAMS"]
