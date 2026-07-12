"""Behavioural model of the RTL UART bridge decoder -- the EQUIVALENCE ORACLE.

It mirrors ``zlc_uart_bridge.v``'s decoder FSM byte-for-byte at the protocol level: hunt for the
SYNC pair, parse the header, buffer the payload, verify CRC, and ONLY THEN commit writes (a corrupt
frame NEVER mis-writes -- the strongest guarantee, and what the RTL is built to match).  A WRITE
commits value[i] to word_addr+i (auto-increment); a READ returns CTRL words (STATUS/CURSOR/LAYOUT_ID
-- the BRAM regions are host-write-only, so only CTRL is readable).  On any fault (bad SYNC, bad
opcode, CRC fail, truncation) it resynchronises by advancing one byte and re-hunting for SYNC.

Two uses:
  * the equivalence oracle -- feed the host's encoded byte stream, get the committed ``{word:value}``
    map, compare to ``image.pack_program`` (proves the UART path is a byte-identical transport swap);
  * a test-only backing store for the current UART register transport -- apply
    writes into a register file and serve READ replies without a serial port.

Mirrors the role of ``engine_model.py`` for the engine: one Python truth the RTL is checked against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import uart_frame as uf

try:  # package import (tests) vs bare script
    from .image import REGISTER_LAYOUT_ID, CtrlWords
except ImportError:  # pragma: no cover
    from image import REGISTER_LAYOUT_ID, CtrlWords  # type: ignore


@dataclass
class BridgeEvent:
    op: str                       # "write" | "read" | "reject"
    ok: bool
    base: int = 0                 # first word address
    values: list[int] = field(default_factory=list)   # write values / read reply words
    count: int = 0
    seq: int = 0
    reply: bytes = b""            # the reply frame the device would send (READ + WRITE ack)


class UartBridgeModel:
    """Consume request BYTES exactly as the RTL bridge would; commit writes; answer reads."""

    def __init__(self, regfile: dict[int, int] | None = None, *, layout_id: int = REGISTER_LAYOUT_ID):
        self.regfile: dict[int, int] = {} if regfile is None else regfile
        self.layout_id = int(layout_id)
        self._buf = bytearray()

    # -- read tap: CTRL word 63 is the hardwired layout id, else the regfile (0 default) --
    def read_word(self, word: int) -> int:
        if int(word) == int(CtrlWords.LAYOUT_ID):
            return self.layout_id & uf.MASK32
        return int(self.regfile.get(int(word), 0)) & uf.MASK32

    def feed(self, data: bytes) -> list[BridgeEvent]:
        """Process a (possibly partial / concatenated) byte stream; commit writes into ``regfile``.

        Returns one BridgeEvent per fully-parsed frame (write/read) plus a ``reject`` event whenever a
        byte is dropped during resync-after-fault (so tests can assert corrupt frames commit nothing)."""
        self._buf += bytes(data)
        events: list[BridgeEvent] = []
        while True:
            ev = self._try_one()
            if ev is None:
                break
            events.append(ev)
        return events

    def _try_one(self) -> BridgeEvent | None:
        buf = self._buf
        # HUNT: advance to a SYNC0,SYNC1 pair (idle 0xFF never matches), dropping bytes as we go.
        while len(buf) >= 2 and not (buf[0] == uf.SYNC0 and buf[1] == uf.SYNC1):
            del buf[0]
            return BridgeEvent(op="reject", ok=False)   # one dropped byte == one resync step
        if len(buf) < uf.HDR_LEN:
            return None                                  # need the full fixed header
        op = buf[2]
        seq = buf[3]
        addr = int.from_bytes(buf[4:8], "little")
        count = int.from_bytes(buf[8:10], "little")
        if op == uf.OP_WRITE:
            need = uf.HDR_LEN + 4 * count + uf.CRC_LEN
        elif op == uf.OP_READ:
            need = uf.HDR_LEN + uf.CRC_LEN
        else:
            del buf[0]                                   # bad opcode -> resync past this SYNC0
            return BridgeEvent(op="reject", ok=False, seq=seq)
        if len(buf) < need:
            return None                                  # wait for the rest of the frame
        span = bytes(buf[uf.OFF_OP:need - uf.CRC_LEN])
        got = int.from_bytes(buf[need - uf.CRC_LEN:need], "little")
        if uf.crc16_ccitt(span) != got:
            del buf[0]                                   # CRC fault -> drop one byte, re-hunt
            return BridgeEvent(op="reject", ok=False, seq=seq)
        # a VALID frame: commit / answer, consume it whole
        if op == uf.OP_WRITE:
            values = [int.from_bytes(buf[uf.HDR_LEN + 4 * i: uf.HDR_LEN + 4 * i + 4], "little")
                      for i in range(count)]
            for i, v in enumerate(values):
                self.regfile[addr + i] = v & uf.MASK32   # auto-increment word address
            reply = uf.encode_reply(seq, uf.ST_OK, ())
            del buf[:need]
            return BridgeEvent(op="write", ok=True, base=addr, values=values, count=count, seq=seq, reply=reply)
        # OP_READ
        words = [self.read_word(addr + i) for i in range(count)]
        reply = uf.encode_reply(seq, uf.ST_OK, words)
        del buf[:need]
        return BridgeEvent(op="read", ok=True, base=addr, values=words, count=count, seq=seq, reply=reply)


def committed_writes(stream: bytes, *, layout_id: int = REGISTER_LAYOUT_ID) -> dict[int, int]:
    """Convenience oracle: the ``{word:value}`` map a byte stream commits (ignoring reads/rejects).
    Used by the equivalence test to compare against ``image.pack_program``."""
    model = UartBridgeModel(layout_id=layout_id)
    model.feed(stream)
    return dict(model.regfile)
