"""UART word transport for the frozen pulse-streamer bridge."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Protocol, Sequence

from fpga.pulse_streamer.host import uart_frame as framing
from fpga.pulse_streamer.host.image import CtrlWords

from .session import TransportAborted


class UartError(RuntimeError):
    pass


class UartLink(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def exchange(
        self,
        request: bytes,
        *,
        deadline: float,
        stop: threading.Event | None = None,
    ) -> bytes: ...

    def write_batch(
        self,
        requests: Sequence[bytes],
        *,
        deadline: float,
        stop: threading.Event | None = None,
    ) -> list[bytes]: ...


class PySerialLink:
    """8N1 serial link with bounded, cancellation-aware reply framing."""

    def __init__(self, port: str, baud: int = 3_000_000) -> None:
        if not port:
            raise ValueError("UART port is required")
        self.port = str(port)
        self.baud = int(baud)
        self._serial = None

    def open(self) -> None:
        import serial

        self._serial = serial.Serial(
            self.port,
            self.baud,
            timeout=0.05,
            write_timeout=1.0,
        )

    def close(self) -> None:
        serial_port = self._serial
        self._serial = None
        if serial_port is not None:
            serial_port.close()

    def exchange(
        self,
        request: bytes,
        *,
        deadline: float,
        stop: threading.Event | None = None,
    ) -> bytes:
        if stop is not None and stop.is_set():
            raise TransportAborted("UART transaction cancelled before issue")
        serial_port = self._require_open()
        serial_port.reset_input_buffer()
        serial_port.write_timeout = _remaining(deadline, "UART write")
        serial_port.write(request)
        serial_port.flush()
        replies = self._read_replies(1, deadline=deadline, stop=stop)
        return replies[0]

    def write_batch(
        self,
        requests: Sequence[bytes],
        *,
        deadline: float,
        stop: threading.Event | None = None,
    ) -> list[bytes]:
        if stop is not None and stop.is_set():
            raise TransportAborted("UART transaction cancelled before issue")
        serial_port = self._require_open()
        if not requests:
            return []
        serial_port.reset_input_buffer()
        # The decoder commits a 256-word frame for several microseconds.  Idle
        # padding prevents the next SYNC pair from arriving during that commit.
        serial_port.write_timeout = _remaining(deadline, "UART batch write")
        serial_port.write((b"\xff" * 8).join(requests))
        serial_port.flush()
        return self._read_replies(len(requests), deadline=deadline, stop=stop)

    def _read_replies(
        self,
        count: int,
        *,
        deadline: float,
        stop: threading.Event | None,
    ) -> list[bytes]:
        serial_port = self._require_open()
        _remaining(deadline, "UART reply")
        buffer = bytearray()
        replies: list[bytes] = []
        while len(replies) < count and time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                raise TransportAborted("UART transaction cancelled")
            available = serial_port.in_waiting
            if available:
                buffer.extend(serial_port.read(available))
                while len(replies) < count:
                    frame = _extract_reply(buffer)
                    if frame is None:
                        break
                    replies.append(frame)
            else:
                time.sleep(min(0.0005, max(0.0, deadline - time.monotonic())))
        if len(replies) != count:
            raise TimeoutError(f"UART replies timed out: {len(replies)}/{count}")
        return replies

    def _require_open(self):
        if self._serial is None:
            raise UartError("serial link is not open")
        return self._serial


class UartRegisterTransport:
    """Ordered word-addressed transport over CRC/sequence-framed UART."""

    transport_id = "uart"
    def __init__(
        self,
        *,
        state_dir: str | Path,
        link: UartLink | None = None,
        port: str | None = None,
        baud: int = 3_000_000,
        action_timeout: float = 5.0,
        max_frame_words: int = framing.MAX_FRAME_WORDS,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if (
            isinstance(action_timeout, bool)
            or not isinstance(action_timeout, (int, float))
            or not math.isfinite(float(action_timeout))
            or action_timeout <= 0
        ):
            raise ValueError("UART action_timeout must be finite and positive")
        self.action_timeout = float(action_timeout)
        self.max_frame_words = max(
            1,
            min(int(max_frame_words), framing.MAX_FRAME_WORDS),
        )
        self._link = link or PySerialLink(str(port or ""), baud)
        self._lock = threading.RLock()
        self._sequence = 0
        self._closed = False

    def start(self) -> None:
        with self._lock:
            self._link.open()
            self._closed = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._link.close()

    def write_words(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> None:
        absolute_deadline = self._effective_deadline(deadline)
        pending = tuple(
            (int(address), int(value) & 0xFFFFFFFF)
            for address, value in rows
        )
        if not pending:
            return
        if not self._lock.acquire(
            timeout=_remaining(absolute_deadline, "UART write lock")
        ):
            raise TimeoutError("UART write timed out waiting for the I/O owner")
        try:
            if self._closed:
                raise UartError(
                    "UART transport is closed; call start() before another transaction"
                )
            if stop is not None and stop.is_set():
                raise TransportAborted("UART write cancelled before issue")
            frames = [
                framing.encode_write(base, values, seq=self._next_sequence())
                for base, values in framing.coalesce_runs(
                    pending,
                    max_words=self.max_frame_words,
                )
            ]
            replies = self._link.write_batch(
                frames,
                deadline=absolute_deadline,
                stop=stop,
            )
            if len(replies) != len(frames):
                raise UartError("UART write reply count differs from request count")
            for request, reply in zip(frames, replies, strict=True):
                sequence, status, words = framing.decode_reply(reply)
                if sequence != request[3]:
                    raise UartError(
                        "UART write reply sequence differs from its request"
                    )
                if status != framing.ST_OK:
                    raise UartError(f"UART write NAK 0x{status:02X}")
                if words:
                    raise UartError("UART write ACK unexpectedly carried data words")
        finally:
            self._lock.release()

    def read_word(
        self,
        word_offset: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> int:
        absolute_deadline = self._effective_deadline(deadline)
        if not self._lock.acquire(
            timeout=_remaining(absolute_deadline, "UART read lock")
        ):
            raise TimeoutError("UART read timed out waiting for the I/O owner")
        try:
            if self._closed:
                raise UartError(
                    "UART transport is closed; call start() before another transaction"
                )
            if stop is not None and stop.is_set():
                raise TransportAborted("UART read cancelled before issue")
            request = framing.encode_read(
                int(word_offset),
                1,
                seq=self._next_sequence(),
            )
            reply = self._link.exchange(
                request,
                deadline=absolute_deadline,
                stop=stop,
            )
            sequence, status, words = framing.decode_reply(reply)
            if sequence != request[3]:
                raise UartError("UART read reply sequence differs from its request")
            if status != framing.ST_OK:
                raise UartError(f"UART read NAK 0x{status:02X}")
            if len(words) != 1:
                raise UartError("UART read returned the wrong word count")
            return int(words[0]) & 0xFFFFFFFF
        finally:
            self._lock.release()

    def rewrite_scan_bank(
        self,
        *,
        unarmed_bank_ready: int,
        bank_words: Sequence[tuple[int, int]],
        chunk_word: int,
        chunk_index: int,
        rearmed_bank_ready: int,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> None:
        """Ack every safety boundary before UART may re-arm a rewritten bank."""

        if chunk_word not in (CtrlWords.BANK0_CHUNK, CtrlWords.BANK1_CHUNK):
            raise ValueError("scan-bank rewrite has an invalid chunk register")
        absolute_deadline = self._effective_deadline(deadline)
        self.write_words(
            ((CtrlWords.BANK_READY, unarmed_bank_ready),),
            stop=stop,
            deadline=absolute_deadline,
        )
        self.write_words(
            tuple(bank_words),
            stop=stop,
            deadline=absolute_deadline,
        )
        self.write_words(
            ((chunk_word, chunk_index),),
            stop=stop,
            deadline=absolute_deadline,
        )
        self.write_words(
            ((CtrlWords.BANK_READY, rearmed_bank_ready),),
            stop=stop,
            deadline=absolute_deadline,
        )

    def record_diagnostic(self, name: str, text: str) -> None:
        try:
            (self.state_dir / f"{name}.log").write_text(
                text,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            pass

    def _effective_deadline(self, value: float | None) -> float:
        deadline = (
            time.monotonic() + self.action_timeout
            if value is None
            else float(value)
        )
        if not math.isfinite(deadline):
            raise ValueError("UART transaction deadline must be finite")
        if deadline <= time.monotonic():
            raise TimeoutError("UART transaction deadline expired before issue")
        return deadline

    def _next_sequence(self) -> int:
        self._sequence = (self._sequence + 1) & 0xFF
        return self._sequence


def _extract_reply(buffer: bytearray) -> bytes | None:
    while len(buffer) >= 2 and not (
        buffer[0] == framing.SYNC0 and buffer[1] == framing.SYNC1
    ):
        del buffer[0]
    if len(buffer) < 7:
        return None
    word_count = int.from_bytes(buffer[5:7], "little")
    frame_length = framing.reply_frame_len(word_count)
    if len(buffer) < frame_length:
        return None
    frame = bytes(buffer[:frame_length])
    del buffer[:frame_length]
    return frame


def _remaining(deadline: float, action: str) -> float:
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{action} exceeded its absolute deadline")
    return remaining


__all__ = [
    "PySerialLink",
    "UartError",
    "UartLink",
    "UartRegisterTransport",
]
