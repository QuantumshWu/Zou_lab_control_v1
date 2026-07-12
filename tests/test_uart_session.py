"""Regression tests for the real pyserial link boundary.

The deployed UART register transport is covered separately.  These tests stay at
the byte-stream boundary where pyserial's timeout semantics and the frozen RTL
decoder's inter-frame idle requirement matter.
"""

from __future__ import annotations

from collections import deque
import time

import pytest

from fpga.pulse_streamer.host import uart_frame as framing
from fpga.pulse_streamer.host.uart_bridge_model import UartBridgeModel
from zlc_pulse.transport import PySerialLink


class _ModelSerial:
    """Small pyserial stand-in backed by the RTL-mirroring bridge model."""

    def __init__(self) -> None:
        self.timeout = 0.05
        self.model = UartBridgeModel()
        self.written = b""
        self._rx = bytearray()

    def reset_input_buffer(self) -> None:
        self._rx.clear()

    def write(self, data: bytes) -> None:
        wire = bytes(data)
        self.written += wire
        self._rx.extend(
            b"".join(
                event.reply
                for event in self.model.feed(wire)
                if event.op == "write"
            )
        )

    def flush(self) -> None:
        pass

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def read(self, size: int) -> bytes:
        result = bytes(self._rx[:size])
        del self._rx[:size]
        return result


def test_write_batch_pads_frame_boundaries_and_collects_every_ack() -> None:
    """Back-to-back writes retain the idle bytes required by the RTL decoder."""

    requests = [
        bytes(framing.encode_write(40 + index, [0x100 + index], seq=index + 1))
        for index in range(3)
    ]
    serial_port = _ModelSerial()
    link = PySerialLink("COM_TEST")
    link._serial = serial_port

    replies = link.write_batch(requests, deadline=time.monotonic() + 1.0)

    assert len(replies) == len(requests)
    assert all(
        framing.decode_reply(reply)[1] == framing.ST_OK for reply in replies
    )
    assert serial_port.written == (b"\xff" * 8).join(requests)
    assert [serial_port.model.regfile[address] for address in range(40, 43)] == [
        0x100,
        0x101,
        0x102,
    ]


class _IncrementalSerial:
    """Expose one reply fragment at a time and emulate total-timeout reads."""

    def __init__(self, fragments: list[bytes]) -> None:
        self.timeout = 0.05
        self._fragments = deque(bytearray(fragment) for fragment in fragments)
        self.stalled = 0.0
        self.max_read = 0

    def reset_input_buffer(self) -> None:
        pass

    def write(self, _data: bytes) -> None:
        pass

    def flush(self) -> None:
        pass

    @property
    def in_waiting(self) -> int:
        return len(self._fragments[0]) if self._fragments else 0

    def read(self, size: int) -> bytes:
        self.max_read = max(self.max_read, int(size))
        fragment = self._fragments[0]
        count = min(int(size), len(fragment))
        if count < int(size):
            self.stalled += self.timeout
        result = bytes(fragment[:count])
        del fragment[:count]
        if not fragment:
            self._fragments.popleft()
        return result


def test_exchange_assembles_incremental_reply_without_oversized_reads() -> None:
    """Only currently available bytes are read, avoiding pyserial timeout stalls."""

    request = bytes(framing.encode_read(0, 1, seq=7))
    reply = [
        event.reply
        for event in UartBridgeModel().feed(request)
        if event.op == "read"
    ][-1]
    fragments = [bytes(reply[:1]), bytes(reply[1:6]), bytes(reply[6:])]
    serial_port = _IncrementalSerial(fragments)
    link = PySerialLink("COM_TEST")
    link._serial = serial_port

    received = link.exchange(request, deadline=time.monotonic() + 1.0)

    assert received == bytes(reply)
    assert serial_port.stalled == 0.0
    assert serial_port.max_read <= max(map(len, fragments))


def test_exchange_times_out_when_no_complete_reply_arrives() -> None:
    """A truncated serial response is never accepted as a complete reply."""

    request = bytes(framing.encode_read(0, 1, seq=9))
    reply = [
        event.reply
        for event in UartBridgeModel().feed(request)
        if event.op == "read"
    ][-1]
    link = PySerialLink("COM_TEST")
    link._serial = _IncrementalSerial([bytes(reply[:-1])])

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="0/1"):
        link.exchange(request, deadline=time.monotonic() + 0.01)
    assert time.monotonic() - started < 0.04
