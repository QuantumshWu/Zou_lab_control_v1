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

from .fpga_pulse_streamer import validate_pulse_streamer_program
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


class _AxiAborted(Exception):
    """Raised when an in-flight hw_axi read is interrupted by a stop request (the
    streaming-refill thread being torn down on Off/prepare).  Distinct from
    TimeoutError, which signals a real hardware fault."""


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
    """Default bit/ltx path from the in-repo final build (fpga/build/ps)."""

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
    candidate = Path(root) / "ps.runs" / "impl_1" / f"zlc_pulse_streamer_top{suffix}"
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
        action_timeout: float | None = 30.0,
        load_timeout: float = 5.0,
        write_batch: int = 200,
        stream_poll_interval: float = 0.005,
        burst_max: int = 256,
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
        self.load_timeout = float(load_timeout)
        self.write_batch = max(1, int(write_batch))
        # AXI4 INCR burst beats per transaction (AWLEN max => 256).  Address-contiguous
        # queued words are coalesced into bursts so one ``run_hw_axi`` moves up to 256
        # words instead of one -- the difference between a multi-second upload and a
        # ~100 ms one over JTAG-to-AXI.  Requires the AXI4 (not AXI4-Lite) bitstream.
        self.burst_max = max(1, min(256, int(burst_max)))
        self.stream_poll_interval = float(stream_poll_interval)

        self._pending: list[tuple[int, int]] = []  # (byte_addr, value) queued writes
        self._repeat_forever = False
        self._program = None          # last prepared program (for streaming refills)
        self._total_points = 0
        self._total_chunks = 1
        self._next_chunk = 2          # finite-streaming cursor (instance state -> re-entrant)
        self._bank_ready = 0b11
        self._io_lock = threading.Lock()    # serialise AXI access (main + stream thread)
        self._stream_thread: threading.Thread | None = None
        self._stream_stop: threading.Event | None = None

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
        self._stop_stream_thread()
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
    def _write_burst_tcl(byte_addr: int, values: Sequence[int]) -> list[str]:
        """One AXI write transaction for ``values`` at consecutive word addresses
        starting at ``byte_addr``.  For len > 1 this is a single INCR burst (needs the
        AXI4 bitstream).  Vivado ``create_hw_axi_txn -data`` for a burst is ONE
        concatenated hex value whose LEAST-significant (rightmost) word goes to the
        BASE address, so the per-beat words are emitted high-address-first."""

        addr = f"{byte_addr & 0xFFFFFFFF:08X}"
        n = len(values)
        if n == 1:
            data = f"{int(values[0]) & 0xFFFFFFFF:08X}"
        else:
            data = "".join(f"{int(v) & 0xFFFFFFFF:08X}" for v in reversed(values))
        # -burst INCR is explicit: the default burst type is not guaranteed INCR across
        # Vivado versions, and FIXED would write every beat to the BASE address (silent
        # corruption).  -size is omitted (defaults to the 32-bit bus width).
        return [
            f"create_hw_axi_txn zlc_w [get_hw_axis] -address {addr} -data {data} -len {n} -type write -burst INCR",
            "run_hw_axi zlc_w",
            "delete_hw_axi_txn zlc_w",
        ]

    def _burst_runs(self, pending: Sequence[tuple[int, int]]) -> list[tuple[int, list[int]]]:
        """Coalesce queued (byte_addr, value) writes into ``(base, [values])`` bursts.

        Runs are built in INSERTION ORDER -- never globally sorted -- so an
        order-dependent command sequence (e.g. COMMAND 0 then COMMAND cmd, or
        BANK_READY de-arm then re-arm at the SAME address) keeps its order and stays a
        sequence of len-1 writes.  Only consecutive, strictly address-contiguous
        (stride 4) entries merge, capped at ``burst_max`` beats."""

        runs: list[tuple[int, list[int]]] = []
        i = 0
        n = len(pending)
        while i < n:
            base, val = pending[i]
            vals = [val]
            j = i + 1
            while (
                j < n
                and len(vals) < self.burst_max
                and pending[j][0] == base + 4 * len(vals)
            ):
                vals.append(pending[j][1])
                j += 1
            runs.append((base, vals))
            i = j
        return runs

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

    def _read_word(self, word_offset: int, *, stop: "threading.Event | None" = None) -> int:
        self._flush(stop=stop)
        marker = "ZLCDATA"
        out = self._run_tcl(self._read_txn_tcl(int(word_offset) * 4, marker), action="axi_read", timeout=self.action_timeout, stop=stop)
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

    def _flush(self, *, stop: "threading.Event | None" = None) -> None:
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        runs = self._burst_runs(pending)
        # Send several bursts per Vivado round-trip (amortise the host<->Tcl latency);
        # write_batch bounds the bursts-per-round-trip, not the words.
        lines: list[str] = []
        bursts = 0
        for base, values in runs:
            lines.extend(self._write_burst_tcl(base, values))
            bursts += 1
            if bursts >= self.write_batch:
                self._run_tcl(lines, action="axi_write", timeout=self.action_timeout, stop=stop)
                lines = []
                bursts = 0
        if lines:
            self._run_tcl(lines, action="axi_write", timeout=self.action_timeout, stop=stop)

    # --------------------------------------------------------------- sequencer API
    def prepare(self, program) -> None:
        """Pack the program into the BRAM image, upload it over AXI, then command
        the top's mini-loader to copy the bus image into the engine (LOAD)."""

        self._stop_stream_thread()             # any prior streaming refill must end first
        p = self.params
        points = list(getattr(program, "scan_points", []) or [])
        self._program = program
        self._total_points = len(points)
        self._total_chunks = max(1, math.ceil(self._total_points / p.bank_size)) if self._total_points else 1
        self._repeat_forever = bool(getattr(program, "repeat_forever", False))
        # Independently validate before upload (defence in depth -- a non-monotonic
        # effective-tick program would silently drop edges on hardware).  Allow the
        # full scan-point count (streaming is unbounded) by raising the cap to N.
        validate_pulse_streamer_program(
            program,
            max_edges=p.max_edges,
            max_scan_points=max(1, self._total_points),
            max_bus_segments=p.max_bus_segments,
            tick_width=p.tick_width,
            channel_count=len(program.channels),
            coeff_width=p.coeff_width,
            num_slots=p.num_slots,
            bus_count=p.bus_count,
            bus_width=p.bus_width,
            num_lanes=p.num_lanes,
            max_lane_edges=p.max_lane_edges,
            # bound the monotonicity sweep so a million-point streamed scan does not
            # hang prepare(); the per-slot extreme points are always included.
            max_validated_scan_points=max(4096, 2 * p.bank_size),
        )
        image = pack_program(program, p)
        # Halt + reset first so a prior run cannot drive outputs while we rewrite BRAM.
        self._command(CMD_SAFE)
        for word_offset in sorted(image):
            self._queue_word(word_offset, image[word_offset])
        # banks 0 and 1 are resident after pack_program (bank_chunk 0/1 set in the
        # image) -> arm both ready bits.
        self._queue_word(CtrlWords.BANK_READY, 0b11)
        self._flush()
        # The mini-loader asserts LOADED within microseconds of CMD_LOAD on real
        # hardware, so wait only a few seconds: a missing LOADED means a wedged
        # bring-up (bad .bit/.ltx, JTAG, or AXI), which should surface as a prompt
        # error -- not a 120 s freeze.
        if not self._command(CMD_LOAD, wait_mask=STATUS_LOADED, timeout=self.load_timeout):
            status = self._read_word(CtrlWords.STATUS)
            raise RuntimeError(
                "pulse streamer did not report LOADED after the program upload "
                f"(STATUS=0x{status:08X}; check the .bit/.ltx, the JTAG cable, and that "
                "run_server programmed the current bitstream)."
            )

    def axi_self_test(self, *, count: int = 16) -> bool:
        """Bring-up check for the AXI4 burst path: burst-write a known ramp into the
        scan-BRAM region, read it back single-beat, and confirm it matches.  This
        catches the one silent failure mode -- a wrong ``create_hw_axi_txn -data`` burst
        byte order (or a still-AXI4-Lite bitstream that ignores ``-len``) -- BEFORE any
        real pulse upload.  Returns True on success; raises on mismatch.  Safe to call
        before ``prepare`` (the scan region is overwritten by the next upload)."""

        base = region_bases(self.params)["scan"]
        n = max(2, int(count))
        pattern = [(0xC0DE0000 + i) & 0xFFFFFFFF for i in range(n)]
        self._stop_stream_thread()
        for offset, value in enumerate(pattern):
            self._queue_word(base + offset, value)
        self._flush()                      # one contiguous burst
        read = [self._read_word(base + offset) for offset in range(n)]
        if read != pattern:
            raise RuntimeError(
                "AXI burst self-test FAILED -- the uploaded ramp read back scrambled, "
                "so the burst -data byte order is wrong or the bitstream is still "
                f"AXI4-Lite (no burst).  wrote={[hex(v) for v in pattern[:4]]}... "
                f"read={[hex(v) for v in read[:4]]}..."
            )
        return True

    def fire(self, program=None) -> None:
        if program is not None:
            self.prepare(program)
        # finite-streaming cursor state lives on the instance so wait_done is
        # re-entrant (a second call resumes the refill instead of reloading chunk 2
        # over whatever the freed bank now holds).
        self._next_chunk = 2          # chunks 0,1 resident after prepare
        self._bank_ready = 0b11
        # Re-arm BOTH ping-pong banks on the FPGA before firing.  prepare() already
        # set this, but a standalone fire() (program=None, e.g. re-fire after a prior
        # repeat_forever STREAMED run that left BANK_READY de-armed mid-refill) must
        # not start the engine against a stale, half-armed bank mask.
        self._queue_word(CtrlWords.BANK_READY, 0b11)
        self._flush()
        self._command(CMD_FIRE)
        (self.state_dir / "fire_time.txt").write_text(str(time.monotonic()), encoding="utf-8")
        # a repeat_forever STREAMED scan (> 2 banks) must be fed continuously while it
        # re-sweeps; a background thread keeps the ping-pong banks loaded.
        self._start_stream_thread()

    # --- streaming refill primitive -----------------------------------------
    def _load_chunk(self, sweep_chunk: int, bank_ready: int, *, stop: "threading.Event | None" = None) -> int:
        """Load sweep-chunk ``sweep_chunk`` into its ping-pong bank and return the
        new BANK_READY mask.  De-arm the bank, write its scan rows, record the
        chunk index (bank_chunk handshake), then re-arm -- so the engine STALLS on
        a bank mid-rewrite and only accepts it once it truly holds this chunk."""

        p = self.params
        bank = sweep_chunk % 2
        refill = scan_bank_words(self._program, p, sweep_chunk)
        bank_ready &= ~(1 << bank)
        self._queue_word(CtrlWords.BANK_READY, bank_ready)        # de-arm during rewrite
        for off in sorted(refill):
            self._queue_word(off, refill[off])
        self._queue_word(CtrlWords.BANK0_CHUNK if bank == 0 else CtrlWords.BANK1_CHUNK, sweep_chunk)
        bank_ready |= (1 << bank)
        self._queue_word(CtrlWords.BANK_READY, bank_ready)        # re-arm
        self._flush(stop=stop)
        return bank_ready

    def wait_done(self, program=None, timeout: float | None = None) -> bool:
        """Poll to completion.  A finite scan with N > 2*bank_size STREAMS here:
        each ping-pong bank is refilled as the cursor frees it (with the bank_chunk
        handshake) so the whole N-point sweep plays gaplessly.  A repeat_forever
        streamed scan is fed by a background thread (started at fire); this returns
        once RUNNING is observed (DONE never asserts for repeat_forever)."""

        effective = float(timeout) if timeout is not None else float(self.action_timeout or 600.0)
        deadline = time.monotonic() + max(0.0, effective)
        p = self.params
        bank_size = p.bank_size
        total_chunks = self._total_chunks

        while True:
            status = self._read_word(CtrlWords.STATUS)
            if status & STATUS_DONE:
                return True
            if self._repeat_forever and (status & STATUS_RUNNING):
                return True            # background thread keeps a streamed re-sweep fed
            if status & STATUS_ERROR:
                raise RuntimeError("pulse streamer reported STATUS_ERROR (bad image magic?).")
            # STATUS_UNDERFLOW is a TRANSIENT streaming stall (the engine reached a
            # bank not yet refilled), NOT a fatal error -- a distinct bit from
            # STATUS_ERROR on purpose; keep polling/refilling and it resumes the
            # instant the bank is armed.  (Guarded by test_final_status_bits_match_host.)

            # --- finite streaming refill: load the next chunk into the freed bank ---
            # (next_chunk / bank_ready live on the instance so this is re-entrant.)
            if self._next_chunk < total_chunks:
                cursor = self._read_word(CtrlWords.CURSOR)
                # chunk (next_chunk-2) lives in bank next_chunk%2; it is fully
                # consumed once cursor >= (next_chunk-1)*bank_size, freeing that bank.
                if cursor >= (self._next_chunk - 1) * bank_size:
                    self._bank_ready = self._load_chunk(self._next_chunk, self._bank_ready)
                    self._next_chunk += 1
                    continue           # re-check status/cursor promptly while streaming

            if time.monotonic() >= deadline:
                return False
            time.sleep(self.stream_poll_interval if self._next_chunk < total_chunks else 0.01)

    # --- repeat_forever streamed re-sweep: background cyclic refill ----------
    def _stream_refill_loop(self) -> None:
        """Keep a repeat_forever STREAMED scan fed forever: refill chunks 2..K-1 as
        the cursor frees their banks, and at each sweep end (cursor reaches N) reload
        chunks 0 and 1 so the engine's wrap (gated on bank_chunk0==0) is served.
        Within a sweep this is gapless; the inter-sweep seam is a brief safe hold
        (never a wrong point).  Runs until _stop_stream_thread()."""

        p = self.params
        bank_size = p.bank_size
        total_chunks = self._total_chunks
        N = self._total_points
        next_chunk = 2
        bank_ready = 0b11
        stop = self._stream_stop
        try:
            while not stop.is_set():
                cursor = self._read_word(CtrlWords.CURSOR, stop=stop)
                if cursor >= N:
                    # sweep finished: the engine is holding at the wrap until chunk 0
                    # (and chunk 1) are resident again.  Reload them, then it re-sweeps.
                    bank_ready = self._load_chunk(0, bank_ready, stop=stop)
                    if total_chunks > 1:
                        bank_ready = self._load_chunk(1, bank_ready, stop=stop)
                    next_chunk = 2
                    # wait for the engine to wrap (cursor returns below N), then refill.
                    # Bounded so a hardware fault (engine never wraps) cannot spin-poll
                    # the AXI link forever -- fall back to the outer loop, which re-checks.
                    for _ in range(2000):
                        if stop.is_set() or self._read_word(CtrlWords.CURSOR, stop=stop) < N:
                            break
                        time.sleep(self.stream_poll_interval)
                    continue
                if next_chunk < total_chunks and cursor >= (next_chunk - 1) * bank_size:
                    bank_ready = self._load_chunk(next_chunk, bank_ready, stop=stop)
                    next_chunk += 1
                    continue
                time.sleep(self.stream_poll_interval)
        except _AxiAborted:             # Off/prepare tore us down mid-read: clean exit
            pass
        except Exception as exc:        # never let the daemon die silently
            self._write_action_log("stream_refill", f"streaming refill thread stopped: {exc!r}")

    def _start_stream_thread(self) -> None:
        if self._repeat_forever and self._total_points > 2 * self.params.bank_size:
            self._stop_stream_thread()
            self._stream_stop = threading.Event()
            self._stream_thread = threading.Thread(
                target=self._stream_refill_loop, name="zlc-scan-stream", daemon=True)
            self._stream_thread.start()

    def _stop_stream_thread(self) -> None:
        if self._stream_stop is not None:
            self._stream_stop.set()
        thread = self._stream_thread
        if thread is not None and thread.is_alive():
            # The refill thread can be mid-AXI-read holding _io_lock; that read is
            # itself bounded by action_timeout and then self-clears.  Wait long
            # enough that the thread is truly dead before reusing the session --
            # if it is still alive, KEEP the handle so a later stop retries the
            # join instead of orphaning a thread that still holds _io_lock (which
            # would make the NEXT prepare/safe_state block on the lock).
            thread.join(timeout=float(self.action_timeout or 120.0) + 5.0)
            if thread.is_alive():
                return
        self._stream_thread = None
        self._stream_stop = None

    def safe_state(self) -> None:
        self._stop_stream_thread()
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
    def _run_tcl(self, lines: Sequence[str], *, action: str, timeout: float | None,
                 stop: "threading.Event | None" = None) -> str:
        # One AXI transaction at a time: the main thread and the streaming-refill
        # thread share the single Vivado Tcl process / marker stream.
        with self._io_lock:
            if self._external_executor is not None:
                return self._external_executor(list(lines), action, timeout)
            return self._execute(lines, action=action, timeout=timeout, stop=stop)

    def _execute(self, lines: Sequence[str], *, action: str, timeout: float | None,
                 stop: "threading.Event | None" = None) -> str:
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
        output = self._read_until_marker(marker, timeout=timeout, stop=stop)
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

    def _read_until_marker(self, marker: str, *, timeout: float | None,
                           stop: "threading.Event | None" = None) -> str:
        # Poll the queue in short slices so a ``stop`` request (from the streaming
        # refill thread being torn down on Off/prepare) is honoured PROMPTLY instead
        # of blocking for the full action_timeout.  Aborting mid-read is safe: each
        # Tcl command carries a UNIQUE marker, so any stale response is harmlessly
        # accumulated-and-skipped by the next read (which matches on its own marker).
        deadline = None if timeout is None else time.monotonic() + max(0.1, float(timeout))
        lines: list[str] = []
        while True:
            if stop is not None and stop.is_set():
                raise _AxiAborted(f"hw_axi read aborted (stop requested) waiting for {marker}.")
            if deadline is None:
                slice_s = 0.2
            else:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    self.close()
                    raise TimeoutError(f"persistent Vivado hw_axi action timed out waiting for {marker}.")
                slice_s = min(remaining, 0.2)   # wake at least every 0.2 s to re-check stop/deadline
            try:
                item = self._queue.get(timeout=slice_s)
            except queue.Empty:
                continue                        # re-check stop / deadline, then keep waiting
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
