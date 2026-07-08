"""UART fast-control wire protocol -- the SINGLE source of the frame layout + CRC.

The UART side-channel is a DEBUG-path replacement for the JTAG-to-AXI transport: it writes the
SAME flat, word-addressed register/BRAM map (``host.image.region_bases`` / ``CtrlWords``) as the
AXI path, so a program is BYTE-IDENTICAL on either transport (a pure transport swap -- no engine
or readout math changes).  Only TWO wire opcodes exist: ``WRITE`` (a run of words at a base word
address) and ``READ`` (a run of words back).  Everything else -- a COMMAND rising edge, a scan-point
step, a layout/PING check -- is COMPOSED by the host from WRITE/READ frames, so this codec and the
RTL bridge FSM stay tiny and ``image.py`` remains the single source.

This module is imported by BOTH the host (``devices/uart_session.py``) and the behavioural model
(``uart_bridge_model.py``) and drives the xsim test-vector generator, so the encoder, the decoder,
the Python oracle and the Verilog testbench can never disagree on a byte.

Frame (LSB-first bytes; all multi-byte fields little-endian):

    request  = SYNC0 SYNC1 OP SEQ ADDR[4] COUNT[2] PAYLOAD[4*COUNT for WRITE]  CRC16[2]
    reply    = SYNC0 SYNC1 RESP SEQ STATUS COUNT[2] PAYLOAD[4*COUNT]           CRC16[2]

CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, xorout 0) over everything from OP/RESP
through the last PAYLOAD byte -- NOT the two SYNC bytes and NOT the CRC itself.  The idle line is
0xFF, which never matches the SYNC0,SYNC1 = 0x5A,0xA5 pair, so a receiver resynchronises by hunting
for that pair.
"""

from __future__ import annotations

import binascii
import struct
from typing import Sequence

# --------------------------------------------------------------------------- constants
SYNC0 = 0x5A
SYNC1 = 0xA5

OP_WRITE = 0x01
OP_READ = 0x02
RESP = 0x81           # reply opcode (ACK for WRITE, data for READ)

# reply STATUS byte
ST_OK = 0x00
ST_CRC_FAIL = 0x01
ST_BAD_OP = 0x02
ST_ADDR_RANGE = 0x03

# byte offsets within a frame (shared by encoder + model + tb generator)
OFF_OP = 2            # SYNC0 SYNC1 then OP/RESP -- the CRC span starts here
HDR_LEN = 10          # SYNC0 SYNC1 OP SEQ ADDR[4] COUNT[2]  (before PAYLOAD)
CRC_LEN = 2
MASK32 = 0xFFFFFFFF

# Max data words in ONE WRITE frame.  == the RTL bridge's frame buffer depth: the bridge buffers a
# whole frame, verifies CRC, and ONLY THEN commits (so a corrupt frame commits NOTHING -- RTL matches
# the Python model exactly).  The host splits a program upload into runs of <= this; at 3 Mbaud 8N1
# (10 bits/byte = 300 KB/s) a 256-word frame is ~1 KB ~3.4 ms, so a 24 KB program is ~24 frames ~82 ms
# (framing overhead tiny).
MAX_FRAME_WORDS = 256


class FrameError(ValueError):
    """A malformed / CRC-failing / truncated frame."""


# --------------------------------------------------------------------------- CRC
def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no in/out reflection, xorout 0."""
    # Delegate to the C-implemented ``binascii.crc_hqx`` (same poly 0x1021) seeded with the 0xFFFF
    # init -- that IS CRC-16/CCITT-FALSE.  BYTE-IDENTICAL to the reference bit-serial loop (proven by
    # a 20000-case fuzz + full 66 KB image buffer, 0 mismatches), ~300x faster (0.1 ms vs 29 ms over a
    # 20000-point image's CRC), which was the dominant host-CPU cost of a large-scan UART frame build.
    # The RTL bridge computes the SAME CRC in hardware, so every frame stays byte-for-byte accepted.
    return binascii.crc_hqx(data, 0xFFFF)


def _u16(value: int) -> bytes:
    return int(value & 0xFFFF).to_bytes(2, "little")


def _u32(value: int) -> bytes:
    return int(value & MASK32).to_bytes(4, "little")


def _frame(body: bytes) -> bytes:
    """Wrap a CRC span (OP..payload) in SYNC + trailing CRC."""
    return bytes((SYNC0, SYNC1)) + body + _u16(crc16_ccitt(body))


# --------------------------------------------------------------------------- encode (host -> device)
def encode_write(word_addr: int, values: Sequence[int], *, seq: int = 0) -> bytes:
    """A WRITE of ``len(values)`` consecutive words starting at ``word_addr`` (a pack_program key).

    The device commits value[i] to word_addr+i (auto-increment), byte-for-byte what an AXI burst to
    the same base does; there is NO 256-beat / 4 KB burst rule here (the bridge drives the internal
    word port one word at a time), so the old AXI 4 KB scan glitch cannot recur."""
    vals = list(values)
    body = bytes((OP_WRITE, seq & 0xFF)) + _u32(word_addr) + _u16(len(vals))
    # One C-level struct.pack of all payload words instead of a per-word ``_u32`` join -- BYTE-
    # IDENTICAL (``_u32(v)`` == little-endian ``v & MASK32``; struct '<I' takes the masked unsigned)
    # but ~10x faster over a full 16507-word image, the last big chunk of the UART frame-build cost.
    body += struct.pack("<%dI" % len(vals), *(v & MASK32 for v in vals))
    return _frame(body)


def encode_read(word_addr: int, count: int, *, seq: int = 0) -> bytes:
    """A READ of ``count`` consecutive words starting at ``word_addr`` (CTRL region only: the BRAMs
    are write-only from the host side -- STATUS/CURSOR/LAYOUT_ID live in the CTRL regfile)."""
    body = bytes((OP_READ, seq & 0xFF)) + _u32(word_addr) + _u16(count)
    return _frame(body)


def encode_reply(seq: int, status: int, words: Sequence[int] = ()) -> bytes:
    """The device's reply frame (unified ACK / READ-reply) -- used by the RTL model + FakeUart."""
    vals = list(words)
    body = bytes((RESP, seq & 0xFF, status & 0xFF)) + _u16(len(vals))
    body += b"".join(_u32(v) for v in vals)
    return _frame(body)


# --------------------------------------------------------------------------- decode (device -> host)
def decode_reply(frame: bytes) -> tuple[int, int, list[int]]:
    """Parse a reply frame -> ``(seq, status, words)``.  Raises FrameError on SYNC/CRC/length fault."""
    if len(frame) < 2 + 4 + CRC_LEN:
        raise FrameError("reply too short")
    if frame[0] != SYNC0 or frame[1] != SYNC1:
        raise FrameError("bad reply preamble")
    if frame[2] != RESP:
        raise FrameError(f"bad reply opcode 0x{frame[2]:02X}")
    seq = frame[3]
    status = frame[4]
    count = int.from_bytes(frame[5:7], "little")
    need = 2 + 3 + 2 + 4 * count + CRC_LEN
    if len(frame) != need:
        raise FrameError(f"reply length {len(frame)} != expected {need} for count {count}")
    span = frame[OFF_OP:need - CRC_LEN]
    got = int.from_bytes(frame[need - CRC_LEN:need], "little")
    if crc16_ccitt(span) != got:
        raise FrameError("reply CRC mismatch")
    words = [int.from_bytes(frame[7 + 4 * i: 11 + 4 * i], "little") for i in range(count)]
    return seq, status, words


def reply_frame_len(count: int) -> int:
    """Total bytes of a reply carrying ``count`` data words (host reader sizing)."""
    return 2 + 3 + 2 + 4 * count + CRC_LEN


# --------------------------------------------------------------------------- run coalescing
def coalesce_runs(pairs: Sequence[tuple[int, int]], *, max_words: int = MAX_FRAME_WORDS) -> list[tuple[int, list[int]]]:
    """Merge strictly word-contiguous (stride-1) ``(word_addr, value)`` entries -- IN ORDER, never
    globally sorted -- into ``(base, [values])`` runs, capped at ``max_words`` words per run (one
    frame's payload).  The UART analogue of the AXI ``_burst_runs`` coalescer, but stride-1 words with
    NO 256-beat / 4 KB burst rule (the bridge drives the internal word port one word at a time)."""
    runs: list[tuple[int, list[int]]] = []
    i = 0
    n = len(pairs)
    while i < n:
        base, val = pairs[i]
        vals = [val]
        j = i + 1
        while j < n and len(vals) < max_words and pairs[j][0] == base + len(vals):
            vals.append(pairs[j][1])
            j += 1
        runs.append((base, vals))
        i = j
    return runs
