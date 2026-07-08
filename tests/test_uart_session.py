"""End-to-end guard: the UART session runs the REAL prepare/fire/wait_done protocol (inherited from
the AXI session) over the serial link and writes the BYTE-IDENTICAL pack_program image.

Uses ``FakeUartTransport`` backed by the RTL-mirroring ``uart_bridge_model`` -- no serial port, no
Vivado -- so the whole host path (image pack -> UART frames -> bridge model -> register map) is proven
in pure Python.  The load-bearing invariant: every pack_program word lands in the model's regfile at
the same value the AXI path would write, and the LOAD/FIRE STATUS handshakes complete.  Real 3 Mbaud
baud-lock + FT2232 wiring are the rig steps (this cannot prove those).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from fpga.pulse_streamer.host import image as im
from Zou_lab_control.neutral_atom.devices.uart_session import UartStreamerSession, FakeUartTransport


def _program():
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod
    from Zou_lab_control.neutral_atom.devices.sequencer import compile_runtime_program_for_payload
    chans = [f"ch{i:02d}" for i in range(62)]
    st = PulseTableState(channels=chans)
    st.periods = [PulsePeriod(10.0, tuple(1 if i == 0 else 0 for i in range(62)), unit="us", name="MOT"),
                  PulsePeriod(3.0, tuple(1 if i == 3 else 0 for i in range(62)), unit="ms", name="image")]
    return compile_runtime_program_for_payload(st, channels=chans, clock_hz=50e6)


def _session():
    fake = FakeUartTransport()
    sess = UartStreamerSession(state_dir=Path(_tmp()), params=im.default_params(), transport=fake,
                               load_timeout=1.0, action_timeout=2.0)
    sess.start()
    return sess, fake


_TMP = []
def _tmp():
    import tempfile
    d = tempfile.mkdtemp(prefix="zlc_uart_")
    _TMP.append(d)
    return d


def test_prepare_writes_the_byte_identical_pack_program_image_and_reaches_loaded():
    """prepare() over UART commits EXACTLY the AXI pack_program image into the register map (byte-
    identical transport swap), and the inherited LOAD handshake reaches STATUS.LOADED."""
    prog = _program()
    p = im.default_params()
    image = im.pack_program(prog, p)
    sess, fake = _session()
    sess.prepare(prog)
    # every image word landed at the same value the AXI path would write
    for word, val in image.items():
        assert fake.writes.get(word) == val, f"UART wrote word {word} != AXI image"
    # the inherited CMD_LOAD poll saw STATUS.LOADED (else prepare would have raised)
    assert fake.writes[im.CtrlWords.STATUS] & im.STATUS_LOADED


def test_fire_sets_running_and_wait_done_returns():
    prog = _program()
    sess, fake = _session()
    sess.fire(prog)                                   # prepare + fire
    assert fake.writes[im.CtrlWords.COMMAND] == im.CMD_FIRE
    assert fake.writes[im.CtrlWords.STATUS] & im.STATUS_RUNNING
    # a repeat_forever program: wait_done returns once RUNNING is observed
    assert sess.wait_done(timeout=1.0) is True


def test_command_is_a_zero_then_cmd_frame_pair():
    """A COMMAND over UART is composed as WRITE(COMMAND,0) then WRITE(COMMAND,cmd) -- the 0->cmd rising
    edge the bridge/loader edge-detects, identical to the AXI path."""
    sess, fake = _session()
    sess._command(im.CMD_SAFE)
    assert fake.writes[im.CtrlWords.COMMAND] == im.CMD_SAFE
    assert fake.writes[im.CtrlWords.STATUS] == 0     # CMD_SAFE clears STATUS (simulated)


def test_read_word_round_trips_status_and_layout():
    sess, fake = _session()
    fake.model.regfile[im.CtrlWords.STATUS] = im.STATUS_LOADED
    assert sess._read_word(im.CtrlWords.STATUS) == im.STATUS_LOADED
    assert sess._read_word(im.CtrlWords.LAYOUT_ID) == im.REGISTER_LAYOUT_ID


def test_check_register_layout_accepts_matching_bitstream():
    """The inherited layout guard reads LAYOUT_ID over UART and accepts the matching id (an old/wrong
    bitstream would return 0 and be refused -- same handshake as AXI)."""
    sess, fake = _session()
    sess.check_register_layout()                      # must not raise (fake serves REGISTER_LAYOUT_ID)


def test_link_self_test_writes_and_reads_scratch_then_cleans():
    sess, fake = _session()
    assert sess.link_self_test(count=8) is True
    base = im.region_bases(im.default_params())["ctrl"] + im.default_params().ctrl_scratch_base
    assert all(fake.writes.get(base + i, 0) == 0 for i in range(8)), "scratch left dirty after self-test"


def test_writes_are_word_addressed_not_byte():
    """The UART bridge is WORD-addressed (matches pack_program keys directly); a plain scan-point write
    lands at the scan region's WORD base, not base*4 (the AXI byte-addr is transport-internal)."""
    p = im.default_params()
    sess, fake = _session()
    scan_base = im.region_bases(p)["scan"]
    sess._queue_word(scan_base, 0x1234)
    sess._flush()
    assert fake.writes.get(scan_base) == 0x1234 and fake.writes.get(scan_base * 4) is None


def test_pyserial_exchange_reads_incrementally_never_a_fixed_oversized_read():
    """PERF REGRESSION: PySerialTransport.exchange must read only what is AVAILABLE (in_waiting),
    NEVER a fixed large count.  pyserial's read(n) on a total-timeout port blocks the WHOLE timeout
    whenever fewer than n bytes arrive, and a reply is only 9-13 bytes -- so the old `read(64)` stalled
    the full ~50 ms on EVERY transaction (~30 round-trips per pulse apply => ~1.5 s, SLOWER than JTAG).
    A FakeSerial mimicking that timeout semantics proves the fix incurs ZERO such stall."""
    from Zou_lab_control.neutral_atom.devices.uart_session import PySerialTransport
    from fpga.pulse_streamer.host.uart_bridge_model import UartBridgeModel
    from fpga.pulse_streamer.host import uart_frame as uf

    req = bytes(uf.encode_read(im.CtrlWords.LAYOUT_ID, 1, seq=1))
    reply = [e.reply for e in UartBridgeModel().feed(req) if e.op == "read"][-1]

    class FakeSerial:
        def __init__(self, reply):
            self.timeout = 0.05; self._buf = b""; self._reply = bytes(reply)
            self.stalled = 0.0; self.max_read = 0
        def reset_input_buffer(self): pass
        def write(self, data): self._buf = self._reply     # the bridge's reply is now buffered to read
        def flush(self): pass
        @property
        def in_waiting(self): return len(self._buf)
        def read(self, size):
            self.max_read = max(self.max_read, int(size))
            n = min(int(size), len(self._buf))
            if n < int(size):                              # pyserial: waits the FULL timeout for the rest
                self.stalled += self.timeout
            out, self._buf = self._buf[:n], self._buf[n:]
            return out

    link = PySerialTransport("COM_TEST")
    link._ser = FakeSerial(reply)
    got = link.exchange(req, timeout=1.0)
    assert got == bytes(reply)
    assert link._ser.stalled == 0.0, "exchange stalled on a fixed oversized read (the read(64) 50 ms bug)"
    assert link._ser.max_read <= len(reply), "exchange must not request more bytes than a reply can hold"
