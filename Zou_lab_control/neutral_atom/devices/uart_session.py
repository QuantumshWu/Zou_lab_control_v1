"""UART fast-control transport session -- a drop-in twin of ``VivadoAxiStreamerSession``.

Same device API (prepare / fire / safe_state / wait_done / scan_progress / snapshot / close) and the
SAME register/BRAM protocol -- it INHERITS all of that transport-agnostic logic from the AXI session
unchanged (prepare/fire/wait_done/_command/_load_chunk/streaming/check_register_layout only ever call
``_queue_word`` / ``_read_word`` / ``_flush``).  Only the transport primitives are overridden: instead
of coalescing into AXI bursts through a persistent Vivado-Tcl process, they frame ``uart_frame`` packets
over a serial link to the ``zlc_uart_bridge``.  So a program is BYTE-IDENTICAL on either transport (a
pure transport swap), but a pulse apply is ~82 ms / a scan-point step ~sub-ms instead of ~1 s -- the
per-transaction Vivado-Tcl + JTAG overhead is gone.

The serial transport is INJECTABLE (like the AXI ``tcl_executor``) so tests run end-to-end with a
``FakeUartTransport`` backed by the RTL-mirroring ``uart_bridge_model`` -- no serial port, virtual==real.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Protocol, Sequence

from .axi_session import VivadoAxiStreamerSession, DEFAULT_PARAMS, DEFAULT_RUNTIME_CLOCK_HZ
from fpga.pulse_streamer.host import uart_frame as uf
from fpga.pulse_streamer.host.image import CtrlWords, region_bases
from fpga.pulse_streamer.host.uart_bridge_model import UartBridgeModel

try:  # image.py constants used by the inherited protocol
    from fpga.pulse_streamer.host.image import (
        CMD_LOAD, CMD_FIRE, CMD_SAFE, STATUS_LOADED, STATUS_RUNNING, STATUS_DONE, REGISTER_LAYOUT_ID,
    )
except ImportError:  # pragma: no cover
    from image import CMD_LOAD, CMD_FIRE, CMD_SAFE, STATUS_LOADED, STATUS_RUNNING, STATUS_DONE, REGISTER_LAYOUT_ID  # type: ignore


class UartError(RuntimeError):
    pass


# --------------------------------------------------------------------------- transport seam
class UartTransport(Protocol):
    name: str
    baud: int
    def open(self) -> None: ...
    def close(self) -> None: ...
    def exchange(self, request: bytes, *, timeout: float, stop: "threading.Event | None" = None) -> bytes: ...
    # Write a RUN of request frames back-to-back and collect their replies IN ORDER.  Used for the
    # WRITE path (program upload / scan refill): streaming all frames without a per-frame ACK
    # round-trip collapses N x USB-latency into ~one, so a whole upload is bounded by wire time
    # (~82 ms) instead of N x ~5 ms round-trips.  Proven safe on the RTL bridge (tb_uart_pipeline.v):
    # a 9-byte WRITE ack always finishes before the next >=16-byte frame completes, so no ack is lost.
    def write_batch(self, requests: "Sequence[bytes]", *, timeout: float, stop: "threading.Event | None" = None) -> "list[bytes]": ...


class PySerialTransport:
    """Real serial link to the bridge.  8N1 at ``baud``; a request gets exactly one reply frame
    (WRITE ack or READ data).  Resynchronises on the SYNC pair and honours ``stop`` in short slices."""

    def __init__(self, port: str | None, baud: int = 3_000_000):
        self.name = str(port or "")
        self.baud = int(baud)
        self._ser = None

    def open(self) -> None:
        import serial   # pyserial; only needed on the real control computer
        # ``timeout`` is only the OUTER read cap; exchange() reads in_waiting-sized slices so it never
        # blocks the whole timeout (that fixed-count read was the ~50 ms/transaction stall).  A USB
        # latency-timer knob is FTDI-only and out-of-band (Device Manager > Port Settings > Advanced,
        # or the FTDIBUS LatencyTimer registry value) -- there is no pyserial call for it (the old
        # self._ser.set_latency_timer() did not exist on pyserial and silently raised every time), and
        # a CH340 has no equivalent register, so we do not attempt it here.
        self._ser = serial.Serial(self.name, self.baud, timeout=0.05, write_timeout=1.0)

    def close(self) -> None:
        if self._ser is not None:
            try: self._ser.close()
            finally: self._ser = None

    def exchange(self, request: bytes, *, timeout: float, stop: "threading.Event | None" = None) -> bytes:
        if self._ser is None:
            raise UartError("serial link not open.")
        import time
        self._ser.reset_input_buffer()
        self._ser.write(request); self._ser.flush()
        deadline = time.monotonic() + max(0.05, float(timeout))
        buf = bytearray()
        while time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                raise UartError("uart exchange aborted (stop requested).")
            # Read only the bytes ALREADY buffered (in_waiting), NEVER a fixed count.  pyserial's
            # read(n) on a total-timeout port blocks the WHOLE timeout whenever fewer than n bytes
            # arrive, and a reply is only 9-13 bytes, so the old `read(64)` stalled the full ~50 ms on
            # EVERY transaction -> ~30 round-trips per pulse apply x 50 ms ~= 1.5 s, SLOWER than JTAG.
            # read(in_waiting) is satisfied instantly, so each exchange costs ~one reply transit
            # (sub-ms at 3 Mbaud + USB latency) and a whole apply drops to ~80 ms.  The 0.5 ms poll
            # slice keeps `stop` responsive (a blocking read would ignore it for up to the timeout).
            n = self._ser.in_waiting
            if n:
                buf += self._ser.read(n)
                frame = _extract_reply(buf)
                if frame is not None:
                    return frame
            else:
                time.sleep(0.0005)
        raise TimeoutError("uart reply timed out.")

    def write_batch(self, requests, *, timeout: float, stop: "threading.Event | None" = None) -> "list[bytes]":
        if self._ser is None:
            raise UartError("serial link not open.")
        import time
        want = len(requests)
        self._ser.reset_input_buffer()
        # INTER-FRAME GAP: the bridge decoder cannot consume RX bytes while it commits a multi-word frame
        # (D_COMMIT loops f_count cycles, and a byte arriving then is dropped).  Committing a full 256-word
        # frame takes ~256 clocks = ~5 us > one 3 Mbaud byte-time (~3.3 us), so a zero-gap next frame would
        # lose its leading SYNC and never ack.  Pad each boundary with idle 0xFF bytes (ignored in D_HUNT):
        # 8 bytes (~27 us) safely covers the worst 256-word commit while adding negligible wire time.
        self._ser.write((b"\xff" * 8).join(requests)); self._ser.flush()
        deadline = time.monotonic() + max(0.05, float(timeout))
        buf = bytearray(); replies: list[bytes] = []
        while len(replies) < want and time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                raise UartError("uart batch aborted (stop requested).")
            n = self._ser.in_waiting                              # in_waiting-sized reads (never a fixed count)
            if n:
                buf += self._ser.read(n)
                while len(replies) < want:
                    frame = _extract_reply(buf)
                    if frame is None:
                        break
                    replies.append(frame)
            else:
                time.sleep(0.0005)
        if len(replies) < want:
            raise TimeoutError(f"uart batch timed out: {len(replies)}/{want} replies.")
        return replies


class FakeUartTransport:
    """Test transport: apply the request through the RTL-mirroring model, simulate the FPGA STATUS
    transitions (LOADED after CMD_LOAD, RUNNING after CMD_FIRE, cleared by CMD_SAFE) so the inherited
    prepare/fire/wait_done handshakes run to completion, and return the bridge's reply frame."""

    def __init__(self, *, layout_id: int = REGISTER_LAYOUT_ID):
        self.name = "fake"; self.baud = 3_000_000
        self.model = UartBridgeModel(layout_id=layout_id)
        self.writes: dict[int, int] = self.model.regfile   # committed word map (== pack_program image)

    def open(self) -> None: pass
    def close(self) -> None: pass

    def exchange(self, request: bytes, *, timeout: float, stop=None) -> bytes:
        events = [e for e in self.model.feed(request) if e.op in ("write", "read")]
        if not events:
            raise UartError("fake uart: request produced no reply (corrupt frame).")
        # simulate STATUS after a COMMAND write so the inherited poll loops terminate
        for e in events:
            if e.op == "write" and e.base == CtrlWords.COMMAND and e.values:
                cmd = e.values[-1] & 0xF
                if cmd == CMD_LOAD:   self.model.regfile[CtrlWords.STATUS] = STATUS_LOADED
                elif cmd == CMD_FIRE: self.model.regfile[CtrlWords.STATUS] = STATUS_LOADED | STATUS_RUNNING
                elif cmd == CMD_SAFE: self.model.regfile[CtrlWords.STATUS] = 0
        return events[-1].reply

    def write_batch(self, requests, *, timeout: float, stop=None) -> "list[bytes]":
        # Feed each frame through the model in order -- byte-identical to N separate exchanges (the RTL
        # commits each frame the same whether the host waited for the ack or not; tb_uart_pipeline.v).
        return [self.exchange(req, timeout=timeout, stop=stop) for req in requests]


def _extract_reply(buf: bytearray) -> bytes | None:
    """Pull one complete reply frame out of ``buf`` (consuming it), resyncing on the SYNC pair."""
    while len(buf) >= 2 and not (buf[0] == uf.SYNC0 and buf[1] == uf.SYNC1):
        del buf[0]
    if len(buf) < 7:          # SYNC0 SYNC1 RESP SEQ STATUS COUNT[2]
        return None
    count = int.from_bytes(buf[5:7], "little")
    need = uf.reply_frame_len(count)
    if len(buf) < need:
        return None
    frame = bytes(buf[:need]); del buf[:need]
    return frame


# --------------------------------------------------------------------------- session
class UartStreamerSession(VivadoAxiStreamerSession):
    """The AXI session's protocol logic over a UART link instead of Vivado hw_axi."""

    def __init__(self, *, state_dir, port: str | None = None, baud: int = 3_000_000,
                 clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ, params=None,
                 action_timeout: float | None = 5.0, load_timeout: float = 1.0,
                 stream_poll_interval: float = 0.001, write_words_max: int = uf.MAX_FRAME_WORDS,
                 transport: UartTransport | None = None):
        # Deliberately NOT calling super().__init__ (it spins up Vivado); set the SHARED state the
        # inherited protocol methods use.  A cross-transport contract test guards against drift.
        self.state_dir = Path(state_dir); self.state_dir.mkdir(parents=True, exist_ok=True)
        self.params = params or DEFAULT_PARAMS
        self.clock_hz = float(clock_hz)
        self.action_timeout = action_timeout
        self.load_timeout = float(load_timeout)
        self.stream_poll_interval = float(stream_poll_interval)
        self.write_words_max = max(1, min(int(write_words_max), uf.MAX_FRAME_WORDS))
        self._link = transport or PySerialTransport(port, baud)

        self._pending: list[tuple[int, int]] = []          # (WORD addr, value) -- word-addressed, not byte
        self._io_lock = threading.RLock()
        self._seq = 0
        self._tail_seconds = 0.0; self._drain_until = 0.0; self._repeat_forever = False
        self._program = None; self._total_points = 0; self._total_chunks = 1; self._scan_repeats = 0
        self._prepared_artifact_digest = None; self._prepared_duration_seconds = 0.0
        self._scan_point = 0; self._scan_sweep = 0; self._scan_finished = False
        self._next_chunk = 2; self._bank_ready = 0b11
        self._stream_thread = None; self._stream_stop = None
        self._closed = False

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "UartStreamerSession":
        self._link.open(); self._closed = False
        return self

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try: self._stop_stream_thread()
        except Exception: pass
        try: self._link.close()
        except Exception: pass

    # ------------------------------------------------------------ transport primitives (override)
    def _queue_word(self, word_offset: int, value: int) -> None:
        self._pending.append((int(word_offset), int(value) & 0xFFFFFFFF))   # WORD addr (bridge is word-addressed)

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFF
        return self._seq

    def _txn(self, request: bytes, *, expect_words: int = 0, stop=None) -> list[int]:
        reply = self._link.exchange(request, timeout=float(self.action_timeout or 5.0), stop=stop)
        _, status, words = uf.decode_reply(reply)
        if status != uf.ST_OK:
            raise UartError(f"uart NAK status 0x{status:02X}")
        return words

    def _flush(self, *, stop=None) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        with self._io_lock:
            frames = [uf.encode_write(base, vals, seq=self._next_seq())
                      for base, vals in uf.coalesce_runs(pending, max_words=self.write_words_max)]
            if not frames:
                return
            # PIPELINE the writes: stream ALL frames back-to-back and collect their acks together, instead
            # of a blocking ACK round-trip per frame.  A whole program upload (~47 frames for a 2000-point
            # scan) drops from N x USB-round-trip (~250 ms) to ~wire time (~80 ms).  Safe on the RTL bridge
            # -- a 9-byte WRITE ack always clears before the next >=16-byte frame completes, so no ack is
            # lost (tb_uart_pipeline.v).  READS stay single (a 13-byte read reply is longer than a request).
            for reply in self._link.write_batch(frames, timeout=float(self.action_timeout or 5.0), stop=stop):
                _, status, _ = uf.decode_reply(reply)
                if status != uf.ST_OK:
                    raise UartError(f"uart NAK status 0x{status:02X}")

    def _read_word(self, word_offset: int, *, stop=None) -> int:
        self._flush(stop=stop)
        with self._io_lock:
            words = self._txn(uf.encode_read(int(word_offset), 1, seq=self._next_seq()), expect_words=1, stop=stop)
        return int(words[0]) & 0xFFFFFFFF if words else 0

    # ------------------------------------------------------------ bring-up self-test (UART analogue)
    def link_self_test(self, *, count: int = 8) -> bool:
        """Write a ramp into the CTRL scratch words (above every defined register, no CLK_ENABLE bit)
        and read it back over the link -- proves baud lock + framing + the bridge write/read path
        before any pulse.  Mirrors ``axi_self_test`` but for the serial transport."""
        base = region_bases(self.params)["ctrl"] + int(self.params.ctrl_scratch_base)
        for i in range(count):
            self._queue_word(base + i, 0xC0DE0000 | i)
        self._flush()
        ok = all(self._read_word(base + i) == (0xC0DE0000 | i) for i in range(count))
        for i in range(count):
            self._queue_word(base + i, 0)          # leave scratch clean
        self._flush()
        if not ok:
            raise UartError("uart link self-test read-back mismatch (baud / framing / wiring).")
        return True

    def current_snapshot(self) -> dict[str, object]:
        snapshot = super().current_snapshot()
        snapshot["transport"] = "uart"
        return snapshot
