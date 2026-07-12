"""MECHANICAL guard: the UART side-channel is a BYTE-IDENTICAL transport swap for the AXI path.

The whole safety of the UART fast-control channel (root fix for the ~1 s pulse-apply latency) rests
on ONE invariant: the host emits, over UART, the SAME ``{word:value}`` register/BRAM writes that the
JTAG-to-AXI path emits from ``image.pack_program`` -- so the engine + readout math are untouched (a
pure transport swap, unlike a calibration change).  These tests prove that end-to-end in pure Python
(host encoder -> the RTL-mirroring ``UartBridgeModel`` decoder -> the committed word map), plus the
framing / CRC / command-ordering / resync properties the RTL bridge must also satisfy.  No hardware,
no Vivado -- the same ``uart_frame`` codec drives the host, this model, and the xsim vector generator,
so they can never disagree on a byte.
"""

from __future__ import annotations
from Zou_lab_control.neutral_atom.ports import PortCatalog

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from fpga.pulse_streamer.host import image as im
from fpga.pulse_streamer.host import uart_frame as uf
from fpga.pulse_streamer.host.uart_bridge_model import UartBridgeModel, committed_writes


# --------------------------------------------------------------- helpers
def _encode_image(image: dict[int, int], *, max_words: int = 4096, seq0: int = 0) -> bytes:
    """The host's upload byte stream: sorted image words -> contiguous runs -> one WRITE frame each
    (the current ``UartRegisterTransport.write_words`` emits the same frames)."""
    pairs = sorted(image.items())
    out = bytearray()
    seq = seq0
    for base, vals in uf.coalesce_runs(pairs, max_words=max_words):
        out += uf.encode_write(base, vals, seq=seq & 0xFF)
        seq += 1
    return bytes(out)


def _programs():
    """A battery of real compiled programs exercising every region."""
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod
    from Zou_lab_control.neutral_atom.devices.sequencer import compile_runtime_program_for_payload
    chans = [f"ch{i:02d}" for i in range(62)]
    progs = []

    # (a) plain TTL, a few periods + a channel delay -> ctrl/tick/coeff/mask/delay
    st = PulseTableState(port_catalog=PortCatalog.from_channels(chans))
    st.periods = [PulsePeriod(10.0, tuple(1 if i == 0 else 0 for i in range(62)), unit="us", name="MOT"),
                  PulsePeriod(3.0, tuple(1 if i == 3 else 0 for i in range(62)), unit="ms", name="image"),
                  PulsePeriod(5.0, tuple(0 for _ in range(62)), unit="us", name="dead")]
    progs.append(("ttl", compile_runtime_program_for_payload(st, channels=chans, clock_hz=50e6)))

    # (b) a scan (duration slot) with a real table -> scan region + repeat_forever
    st2 = PulseTableState(port_catalog=PortCatalog.from_channels(chans))
    st2.periods = [PulsePeriod(10.0, tuple(1 if i == 0 else 0 for i in range(62)), unit="us", name="p"),
                   PulsePeriod(1.0, tuple(1 if i == 3 else 0 for i in range(62)), unit="us", name="probe")]
    st2.bind_field("duration", "1", unit="us")
    st2.set_scan_table([[float(v)] for v in range(20, 20 + 300)])   # 300 scan points, 1 slot
    progs.append(("scan", compile_runtime_program_for_payload(st2, channels=chans, clock_hz=50e6)))
    return progs


# --------------------------------------------------------------- core equivalence
@pytest.mark.parametrize("name", ["ttl", "scan"])
def test_host_encode_roundtrips_through_model_bytes_identical(name):
    """pack_program -> host UART encode -> RTL-model decode -> committed word map == pack_program.
    THE load-bearing invariant: UART writes exactly what AXI writes."""
    prog = dict(_programs())[name]
    p = im.default_params()
    image = im.pack_program(prog, p)
    stream = _encode_image(image)
    assert committed_writes(stream) == image, f"UART transport is not byte-identical to the AXI image ({name})"


def test_framing_roundtrip_edge_values_and_noncontiguous():
    """Arbitrary word maps incl. 0 / 0xFFFFFFFF / sparse gaps / a max-length run survive encode->decode."""
    image = {0: 0, 1: 0xFFFFFFFF, 2: 0x5A5A5A5A, 100: 1, 101: 2, 102: 3, 5000: 0xDEADBEEF}
    image.update({20544 + i: (i * 2654435761) & 0xFFFFFFFF for i in range(600)})   # a long contiguous run
    assert committed_writes(_encode_image(image)) == image


def test_single_word_and_full_run_cap():
    """A run longer than write_words_max splits into multiple frames but commits identically."""
    image = {7: 0x11111111}
    image.update({1000 + i: (i + 1) & 0xFFFFFFFF for i in range(9000)})   # > 4096 -> multiple frames
    assert committed_writes(_encode_image(image, max_words=4096)) == image


# --------------------------------------------------------------- composed host ops (not wire opcodes)
def test_command_is_a_zero_then_cmd_write_pair_in_order():
    """A COMMAND rising edge is composed as WRITE(COMMAND,0) then WRITE(COMMAND,cmd) -- the model sees
    the two writes to the SAME word in that order (reproduces the loader's 0->cmd edge detect)."""
    seq_stream = uf.encode_write(im.CtrlWords.COMMAND, [0], seq=1) + \
        uf.encode_write(im.CtrlWords.COMMAND, [im.CMD_FIRE], seq=2)
    model = UartBridgeModel()
    events = [e for e in model.feed(seq_stream) if e.op == "write"]
    assert [(e.base, e.values) for e in events] == [(im.CtrlWords.COMMAND, [0]),
                                                    (im.CtrlWords.COMMAND, [im.CMD_FIRE])]
    assert model.regfile[im.CtrlWords.COMMAND] == im.CMD_FIRE   # last write wins


def test_scan_point_step_writes_only_scan_and_ctrl_words():
    """The step/hold fast path writes point k's 4 slots + SCAN_COUNT=1 + arm -- and NOTHING in the edge
    region (tick/coeff/mask/bus), which is why a step is a handful of tiny frames, not a full re-upload."""
    p = im.default_params()
    bases = im.region_bases(p)
    slots = [123, 456, 789, 0]
    stream = (uf.encode_write(bases["scan"], slots)
              + uf.encode_write(im.CtrlWords.SCAN_COUNT, [1])
              + uf.encode_write(im.CtrlWords.SCAN_ENABLE, [1])
              + uf.encode_write(im.CtrlWords.BANK0_CHUNK, [0])
              + uf.encode_write(im.CtrlWords.BANK_READY, [0b01]))
    writes = committed_writes(stream)
    assert [writes[bases["scan"] + j] for j in range(4)] == slots
    assert writes[im.CtrlWords.SCAN_COUNT] == 1
    # no edge-region word was touched
    assert all(not (bases["tick"] <= w < bases["scan"]) for w in writes), "a step must not write the edge regions"


# --------------------------------------------------------------- reliability
def test_corrupt_crc_commits_nothing_then_resyncs():
    """A CRC-flipped frame commits ZERO writes; a following good frame still decodes (byte-hunt resync)."""
    good = uf.encode_write(50, [0xAAAA0000, 0xAAAA0001])
    bad = bytearray(uf.encode_write(50, [0x12345678]))
    bad[-1] ^= 0xFF                                   # corrupt the CRC
    model = UartBridgeModel()
    model.feed(bytes(bad))
    assert model.regfile == {}, "a corrupt frame must commit nothing"
    model.feed(good)
    assert model.regfile == {50: 0xAAAA0000, 51: 0xAAAA0001}, "the next good frame resyncs and commits"


def test_truncated_frame_waits_then_completes():
    """A frame split across two feed() calls (partial arrival) commits once fully received, not before."""
    frame = uf.encode_write(9, [0xCAFEBABE, 0x0BADF00D])
    model = UartBridgeModel()
    assert model.feed(frame[:6]) == [] or model.regfile == {}   # nothing committed on the partial
    assert model.regfile == {}
    model.feed(frame[6:])
    assert model.regfile == {9: 0xCAFEBABE, 10: 0x0BADF00D}


def test_read_reply_parses_layout_and_status():
    """A READ(LAYOUT_ID) reply carries REGISTER_LAYOUT_ID; a READ(STATUS) reply carries the regfile word."""
    model = UartBridgeModel(regfile={im.CtrlWords.STATUS: im.STATUS_LOADED})
    (ev_layout,) = [e for e in model.feed(uf.encode_read(im.CtrlWords.LAYOUT_ID, 1)) if e.op == "read"]
    seq, status, words = uf.decode_reply(ev_layout.reply)
    assert status == uf.ST_OK and words == [im.REGISTER_LAYOUT_ID]
    (ev_status,) = [e for e in model.feed(uf.encode_read(im.CtrlWords.STATUS, 1)) if e.op == "read"]
    _, _, words2 = uf.decode_reply(ev_status.reply)
    assert words2 == [im.STATUS_LOADED]


def test_crc16_ccitt_known_answer():
    """CRC-16/CCITT-FALSE reference vector: '123456789' -> 0x29B1 (locks the poly/seed against drift)."""
    assert uf.crc16_ccitt(b"123456789") == 0x29B1


def test_count_zero_frame_commits_nothing_and_acks():
    """A protocol-legal count==0 WRITE (COUNT is a 16-bit field, 0 is representable) must commit NO
    words and reply an ACK -- the contract the RTL bridge's D_CRC count==0 branch now matches (it
    guards the ``w_idx==f_count-1`` underflow that would else drive 65536 spurious writes across the
    whole register/BRAM map).  The READ count==0 replies with an empty payload."""
    model = UartBridgeModel()
    model.regfile[7] = 0xDEADBEEF                       # a pre-existing word must survive untouched
    frame = uf.encode_write(0, [], seq=3)               # count == 0 WRITE at base 0
    events = [ev for ev in model.feed(frame) if ev is not None]
    assert model.regfile == {7: 0xDEADBEEF}             # scribbled NOTHING
    assert events and events[-1].op == "write" and events[-1].ok and events[-1].count == 0
    read0 = uf.encode_read(0, 0, seq=4)                 # count == 0 READ
    revs = [ev for ev in UartBridgeModel().feed(read0) if ev is not None]
    assert revs and revs[-1].op == "read" and revs[-1].ok and revs[-1].count == 0 and revs[-1].values == []
