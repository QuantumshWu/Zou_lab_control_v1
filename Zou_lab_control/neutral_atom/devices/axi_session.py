"""Persistent Vivado JTAG-to-AXI runtime session for the edge-table loader.

This drives the affine edge-table pulse streamer (``zlc_pulse_streamer``, the
validated seamless engine) through the on-chip loader (``zlc_axi_program_loader``)
over the JTAG-to-AXI master.  It holds one persistent ``vivado -mode tcl`` process,
connects to the programmed FPGA, and:

  * packs the compiled :class:`RuntimeSequenceProgram` into the BRAM program image
    (:mod:`edgetable_image`),
  * writes that image into the program BRAM over AXI (``create_hw_axi_txn`` /
    ``run_hw_axi``),
  * drives the loader's COMMAND/STATUS mailbox (LOAD -> the loader copies the image
    into the engine's prog_* ports and asserts LOADED; FIRE -> the loader releases
    reset and pulses start; SAFE -> halt + reset), and
  * polls STATUS (DONE for a finite scan; a repeat_forever program never asserts
    DONE so the host treats RUNNING as success).

The Tcl execution is injectable (``tcl_executor``) so the whole
prepare/fire/wait_done/safe_state flow is tested without Vivado or hardware.
``wait_done`` always bounds its poll, so the server can never busy-poll forever.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from .edgetable_image import (
    EdgeTableImageParams,
    CtrlWords,
    pack_program,
    pack_program_d,
    CMD_LOAD,
    CMD_FIRE,
    CMD_SAFE,
    CMD_RESET,
    STATUS_LOADED,
    STATUS_RUNNING,
    STATUS_DONE,
    STATUS_ERROR,
)
from .edgetable_engine_model import min_edge_spacing

# Architecture-D depth-1 prefetch settle: the host must keep min edge spacing >=
# this many ticks so the BRAM read always lands before the edge fires.
D_MIN_EDGE_SPACING = 3

DEFAULT_RUNTIME_CLOCK_HZ = 50_000_000.0


def _default_vivado() -> str:
    for name in ("ZLC_PS_VIVADO_BIN", "ZLC_VIVADO_BIN"):
        value = os.environ.get(name)
        if value:
            return value
    return "vivado"


def _default_artifact(suffix: str) -> str | None:
    """Default bit/ltx path from the in-repo edge-table loader build (fpga/build/l)."""

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
    candidate = Path(root) / "l.runs" / "impl_1" / f"zlc_pulse_streamer_loader_top{suffix}"
    return str(candidate) if candidate.exists() else None


class VivadoAxiStreamerSession:
    """Persistent Vivado hw_axi transport for the edge-table loader pulse streamer."""

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
        params: EdgeTableImageParams | None = None,
        variant: str = "loader",
        startup_timeout: float = 180.0,
        action_timeout: float | None = 120.0,
        write_batch: int = 200,
        tcl_executor: Callable[[Sequence[str], str, float | None], str] | None = None,
    ):
        # variant "loader" = the LUTRAM edge-table loader path (pack_program);
        # "d" = the Architecture-D BRAM-table path (pack_program_d + min edge
        # spacing enforcement).  Both share the COMMAND/STATUS mailbox + Tcl plumbing.
        self.variant = str(variant).strip().lower()
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.vivado = vivado or _default_vivado()
        self.bitstream = bitstream if bitstream is not None else _default_artifact(".bit")
        self.probes = probes if probes is not None else _default_artifact(".ltx")
        self.hw_server_url = hw_server_url or os.environ.get("ZLC_PS_HW_SERVER_URL") or os.environ.get("ZLC_HW_SERVER_URL") or ""
        self.program_on_start = bool(program_on_start)
        self.clock_hz = float(clock_hz)
        self.params = params or EdgeTableImageParams()
        self.startup_timeout = float(startup_timeout)
        self.action_timeout = action_timeout
        self.write_batch = max(1, int(write_batch))

        self._pending: list[tuple[int, int]] = []  # (byte_addr, value) queued writes
        self._repeat_forever = False  # remembers the last prepared program's loop mode

        # Persistent Vivado Tcl session plumbing (mirrors VivadoPulseStreamerSession).
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
                f"run-length session could not start persistent Vivado. See {self.state_dir / 'vivado_axi_session_start.log'}."
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
            'if {$zlc_axi eq ""} { error "No JTAG-to-AXI (hw_axi) core found. Program the run-length bitstream first (build_and_program.bat), and check the .ltx probes file." }',
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
        # DATA comes back as a hex string (possibly with 0x / spaces); take the last.
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
        """Pack the edge-table program into the BRAM image, upload it over AXI, then
        command the on-chip loader to copy it into the engine (LOAD)."""

        if self.variant == "d":
            # depth-1 prefetch requires the host to keep edges >= D_MIN_EDGE_SPACING
            # ticks apart (else the BRAM read cannot land before the edge fires).
            spacing = min_edge_spacing(program)
            if spacing < D_MIN_EDGE_SPACING:
                raise ValueError(
                    f"Architecture-D engine needs edges >= {D_MIN_EDGE_SPACING} ticks apart "
                    f"(found {spacing}). Widen the closest pulse/delay or use the loader build."
                )
            image = pack_program_d(program, self.params)
        else:
            image = pack_program(program, self.params)
        self._repeat_forever = bool(getattr(program, "repeat_forever", False))
        # Halt + reset first so a prior run cannot drive outputs while we rewrite BRAM.
        self._command(CMD_SAFE)
        # Upload the program image (only the used words are emitted by pack_program).
        for word_offset in sorted(image):
            self._queue_word(word_offset, image[word_offset])
        self._flush()
        # Tell the loader to copy the image into the engine's prog_* tables.
        if not self._command(CMD_LOAD, wait_mask=STATUS_LOADED):
            raise RuntimeError(
                "edge-table loader did not report LOADED after the program upload "
                "(check the .bit/.ltx, the JTAG cable, and the loader STATUS word)."
            )

    def fire(self, program=None) -> None:
        if program is not None:
            self.prepare(program)
        # FIRE: the loader releases the engine reset and pulses start.  RUNNING is set
        # by the loader; DONE is cleared at fire time inside the loader.
        self._command(CMD_FIRE)
        (self.state_dir / "loader_fire_time.txt").write_text(str(time.monotonic()), encoding="utf-8")

    def wait_done(self, program=None, timeout: float | None = None) -> bool:
        # Always bound the poll: with timeout=None fall back to a finite ceiling so the
        # server can never busy-poll the engine forever (honours "never hang").
        effective = float(timeout) if timeout is not None else float(self.action_timeout or 600.0)
        deadline = time.monotonic() + max(0.0, effective)
        while True:
            status = self._read_word(CtrlWords.STATUS)
            if status & STATUS_DONE:
                return True
            # A repeat_forever program never asserts DONE; treat RUNNING as success so
            # the caller is not blocked for the full timeout on an infinite loop.
            if self._repeat_forever and (status & STATUS_RUNNING):
                return True
            if status & STATUS_ERROR:
                raise RuntimeError("edge-table loader reported STATUS_ERROR (bad image magic?).")
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def safe_state(self) -> None:
        self._command(CMD_SAFE)

    # ------------------------------------------------------------------ mailbox
    def _command(self, command: int, *, wait_mask: int | None = None,
                 timeout: float | None = None) -> bool:
        """Drive a rising edge on the loader COMMAND word, optionally waiting for a
        STATUS mask.  The loader edge-detects commands, so write 0 first to guarantee
        a clean 0->cmd transition even if the same command was issued before."""

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
                raise RuntimeError("edge-table loader reported STATUS_ERROR (bad image magic?).")
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


__all__ = ["VivadoAxiStreamerSession", "DEFAULT_RUNTIME_CLOCK_HZ"]
