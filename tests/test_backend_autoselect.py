"""`--backend auto`: probe fastest-first and use the first transport that VERIFIES against this host's
register layout, so on_pulse/fire ride the best available link with no client change.

Hardware-free: the UART probe uses a real ``UartStreamerSession`` over ``FakeUartTransport`` (backed by
the RTL-mirroring ``uart_bridge_model``, which serves the layout id on CTRL word 63), so the full
start -> check_register_layout -> clear_host_config -> link_self_test bring-up runs for real; JTAG is a
stub.  Selection/discovery are injected via monkeypatch, never touching Vivado or a serial port.
"""

from __future__ import annotations

import types

import pytest

from Zou_lab_control.neutral_atom.devices import sequencer_server as ss
from Zou_lab_control.neutral_atom.devices.uart_session import UartStreamerSession, FakeUartTransport
from fpga.pulse_streamer.host import image as im
from fpga.pulse_streamer.host.image import REGISTER_LAYOUT_ID


def _make_uart(tmp_path, *, layout_id=REGISTER_LAYOUT_ID):
    """A real UART session whose FakeUartTransport answers reads with ``layout_id`` on word 63."""
    return UartStreamerSession(
        state_dir=tmp_path, params=im.default_params(),
        transport=FakeUartTransport(layout_id=layout_id),
    )


class _FakeJtag:
    """A JTAG session stub that verifies + arms (records the bring-up it received)."""

    def __init__(self):
        self.started = self.cleared = self.tested = self.closed = False

    def start(self):
        self.started = True
        return self

    def check_register_layout(self):
        pass

    def clear_host_config(self):
        self.cleared = True

    def axi_self_test(self, **_):
        self.tested = True
        return True

    def close(self):
        self.closed = True

    # device callbacks (present so run_server-style wiring would work; unused here)
    def prepare(self, *a, **k): ...
    def fire(self, *a, **k): ...
    def wait_done(self, *a, **k): ...
    def safe_state(self, *a, **k): ...
    def scan_progress(self, *a, **k): return {}


def _auto(tmp_path, **over):
    kw = dict(channels=["ch00"], clock_hz=50_000_000, state_dir=tmp_path,
              uart_port=None, uart_baud=3_000_000, warm_start=True, log=lambda *a: None)
    kw.update(over)
    return ss._auto_select_backend(**kw)


# ------------------------------------------------------------------ port discovery
def test_discover_uart_ports_explicit_wins(monkeypatch):
    monkeypatch.setenv("ZLC_PS_UART_PORT", "COM_ENV")     # explicit arg still wins over the env
    assert ss._discover_uart_ports("COM6") == ["COM6"]


def test_discover_uart_ports_env(monkeypatch):
    monkeypatch.setenv("ZLC_PS_UART_PORT", "COM9")
    assert ss._discover_uart_ports(None) == ["COM9"]


def test_discover_uart_ports_scans_ch340_first(monkeypatch):
    monkeypatch.delenv("ZLC_PS_UART_PORT", raising=False)
    lp = pytest.importorskip("serial.tools.list_ports")
    fake = [
        types.SimpleNamespace(vid=0x10C4, device="COM3"),   # CP210x
        types.SimpleNamespace(vid=0x1A86, device="COM6"),   # CH340 (Da Vinci USB_UART)
        types.SimpleNamespace(vid=0x1234, device="COM7"),   # unrelated -> excluded
    ]
    monkeypatch.setattr(lp, "comports", lambda: fake)
    assert ss._discover_uart_ports(None) == ["COM6", "COM3"]   # CH340 ranked first, unrelated dropped


# ------------------------------------------------------------------ auto selection
def test_auto_selects_uart_when_link_verifies(monkeypatch, tmp_path):
    made: dict[str, int] = {}

    def fake_make(name, **kw):
        made[name] = made.get(name, 0) + 1
        if name == "uart":
            return _make_uart(tmp_path)
        raise AssertionError("jtag-axi must NOT be constructed when the UART link verifies")

    monkeypatch.setattr(ss, "_make_hardware_backend", fake_make)
    monkeypatch.setattr(ss, "_discover_uart_ports", lambda explicit: ["COM_FAKE"])

    name, sess = _auto(tmp_path)
    assert name == "uart"
    assert made == {"uart": 1}          # Vivado/JTAG never even constructed -> no cost when UART works
    sess.close()


def test_auto_falls_back_to_jtag_on_uart_layout_mismatch(monkeypatch, tmp_path):
    jtag = _FakeJtag()

    def fake_make(name, **kw):
        if name == "uart":
            return _make_uart(tmp_path, layout_id=0xDEADBEEF)   # wrong layout -> check_register_layout raises
        if name == "jtag-axi":
            return jtag
        raise AssertionError(name)

    monkeypatch.setattr(ss, "_make_hardware_backend", fake_make)
    monkeypatch.setattr(ss, "_discover_uart_ports", lambda explicit: ["COM_FAKE"])

    name, sess = _auto(tmp_path)
    assert name == "jtag-axi"
    assert sess is jtag and jtag.started and jtag.cleared and jtag.tested


def test_auto_raises_when_nothing_verifies(monkeypatch, tmp_path):
    def fake_make(name, **kw):
        if name == "uart":
            return _make_uart(tmp_path, layout_id=0xBAD1)       # UART answers wrong layout
        if name == "jtag-axi":
            class _J:
                def start(self): raise RuntimeError("Vivado cannot open a hw_target")
                def close(self): ...
            return _J()
        raise AssertionError(name)

    monkeypatch.setattr(ss, "_make_hardware_backend", fake_make)
    monkeypatch.setattr(ss, "_discover_uart_ports", lambda explicit: ["COM_FAKE"])

    with pytest.raises(RuntimeError, match="no usable transport"):
        _auto(tmp_path)


def test_auto_no_warm_start_selects_jtag_lazily(monkeypatch, tmp_path):
    jtag = _FakeJtag()

    def fake_make(name, **kw):
        if name == "jtag-axi":
            return jtag
        raise AssertionError("UART must not be constructed when no serial port is present")

    monkeypatch.setattr(ss, "_make_hardware_backend", fake_make)
    monkeypatch.setattr(ss, "_discover_uart_ports", lambda explicit: [])   # no UART bridge found

    name, sess = _auto(tmp_path, warm_start=False)
    assert name == "jtag-axi"
    assert sess is jtag
    assert jtag.started is False          # --no-warm-start: Vivado is NOT spun up at startup (deferred)


# ------------------------------------------------------------------ bring-up invariants
def test_warm_start_explicit_preserves_bringup_order_without_layout_check():
    """Explicit paths (verify_layout=False) keep the pinned start -> clear -> self-test order and NEVER
    call check_register_layout -- guards the run_server jtag warm-start contract test from regressing."""
    calls: list[str] = []

    class _S:
        def start(self): calls.append("start")
        def check_register_layout(self): calls.append("layout")
        def clear_host_config(self): calls.append("clear")
        def axi_self_test(self, **_): calls.append("selftest")

    ss._warm_start_hardware(_S(), "jtag-axi", log=lambda *a: None)
    assert calls == ["start", "clear", "selftest"]


def test_warm_start_auto_inserts_layout_check():
    calls: list[str] = []

    class _S:
        action_timeout = 5.0
        def start(self): calls.append("start")
        def check_register_layout(self): calls.append("layout")
        def clear_host_config(self): calls.append("clear")
        def link_self_test(self, **_): calls.append("selftest")

    ss._warm_start_hardware(_S(), "uart", log=lambda *a: None, verify_layout=True, layout_timeout=0.6)
    assert calls == ["start", "layout", "clear", "selftest"]


# ------------------------------------------------------------------ CLI default
def test_cli_default_backend_is_auto():
    args = ss.build_arg_parser().parse_args(["--channels", "ch00"])   # --channels is required; backend defaults
    assert args.backend == "auto"


# ------------------------------------------------------------------ run_server integration
def test_run_server_auto_wires_selected_session_not_the_name(monkeypatch, tmp_path):
    """Regression: run_server(backend='auto') must unpack (name, session) and wire the SESSION's
    callbacks.  An earlier swap wired the backend-NAME string, so hardware_backend.prepare crashed
    with 'str' object has no attribute 'prepare' at server start."""
    sess = _FakeJtag()   # exposes prepare/fire/wait_done/safe_state/scan_progress
    monkeypatch.setattr(ss, "_auto_select_backend", lambda **kw: ("uart", sess))
    served = {}

    def fake_serve(service, *, host, port, start):
        served["service"] = service
        return "SERVED"

    monkeypatch.setattr(ss, "serve_runtime_sequencer", fake_serve)
    out = ss.run_server(channels=["ch00", "ch01"], host="127.0.0.1", port=18871,
                        state_dir=tmp_path / "st", backend="auto", warm_start=True)
    assert out == "SERVED"            # no AttributeError: 'str' object has no attribute 'prepare'
    assert "service" in served


# ------------------------------------------------------------------ XDC channel inference
def test_infer_channel_count_excludes_uart_pins(tmp_path):
    """board.xdc's uart_rx/uart_tx are infrastructure pins (like clk/led/GND), NOT user channels --
    adding them must not bump the inferred channel count (regression: Da Vinci reported 64ch not 62)."""
    from Zou_lab_control.neutral_atom.devices import fpga_pulse_streamer as fps
    xdc = tmp_path / "b.xdc"
    xdc.write_text(
        "set_property -dict {PACKAGE_PIN R4 IOSTANDARD LVCMOS33} [get_ports clk]\n"
        "set_property -dict {PACKAGE_PIN A1 IOSTANDARD LVCMOS33} [get_ports cooling]\n"
        "set_property -dict {PACKAGE_PIN A2 IOSTANDARD LVCMOS33} [get_ports repump]\n"
        "set_property -dict {PACKAGE_PIN A3 IOSTANDARD LVCMOS33} [get_ports {led[0]}]\n"
        "set_property -dict {PACKAGE_PIN A4 IOSTANDARD LVCMOS33} [get_ports GND1]\n"
        "set_property -dict {PACKAGE_PIN U5 IOSTANDARD LVCMOS33} [get_ports uart_rx]\n"
        "set_property -dict {PACKAGE_PIN T6 IOSTANDARD LVCMOS33} [get_ports uart_tx]\n",
        encoding="utf-8",
    )
    labels = fps._xdc_output_port_labels(xdc.read_text(encoding="utf-8"))
    assert labels == ["cooling", "repump"]                 # clk/led/GND/uart_rx/uart_tx all excluded
    assert fps.infer_xdc_channel_count(xdc, default=62, max_count=None) == 2
