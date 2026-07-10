"""Command-line sequencer service for the FPGA/Vivado computer."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import os
import socket
import subprocess
from typing import Sequence

from ..ports import PortCatalog, coerce_port_catalog
from .sequencer import DEFAULT_RUNTIME_CLOCK_HZ, RuntimeSequenceProgram, SequencerService, serve_runtime_sequencer


@dataclass
class CommandSequencerBackend:
    """Bridge ``SequencerService`` callbacks to lab-owned hardware commands.

    The command interface is deliberately new and narrow: the service writes a
    JSON ``RuntimeSequenceProgram`` file, exports its path and metadata through
    environment variables, then runs the configured command.  A Vivado Tcl
    script, vendor Python script, or future FPGA runtime uploader can be placed
    behind that command without changing the control-computer notebook API.
    """

    state_dir: Path
    prepare_command: str | None = None
    fire_command: str | None = None
    wait_done_command: str | None = None
    safe_state_command: str | None = None
    timeout: float | None = None

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.program_path = self.state_dir / "prepared_program.json"

    def prepare(self, program: RuntimeSequenceProgram) -> None:
        self._write_program(program)
        self._run(self.prepare_command, program, action="prepare")

    def fire(self, program: RuntimeSequenceProgram) -> None:
        self._write_program(program)
        self._run(self.fire_command, program, action="fire")

    def wait_done(self, program: RuntimeSequenceProgram, timeout: float | None) -> bool:
        self._write_program(program)
        if self.wait_done_command is None:
            return True
        self._run(self.wait_done_command, program, action="wait_done", timeout=timeout)
        return True

    def safe_state(self) -> None:
        self._run(self.safe_state_command, None, action="safe_state")

    def _write_program(self, program: RuntimeSequenceProgram) -> None:
        self.program_path.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")
        (self.state_dir / "last_sequence_id.txt").write_text(program.sequence_id, encoding="utf-8")

    def _run(
        self,
        command: str | None,
        program: RuntimeSequenceProgram | None,
        *,
        action: str,
        timeout: float | None = None,
    ) -> None:
        if command is None:
            return
        env = os.environ.copy()
        env["ZLC_SEQUENCER_ACTION"] = action
        env["ZLC_STATE_DIR"] = str(self.state_dir)
        env["ZLC_SEQUENCE_PROGRAM"] = str(self.program_path)
        if timeout is not None:
            env["ZLC_TIMEOUT"] = str(timeout)
        if program is not None:
            env["ZLC_SEQUENCE_ID"] = program.sequence_id
            env["ZLC_SEQUENCE_NAME"] = program.sequence_name
            env["ZLC_CLOCK_HZ"] = str(program.clock_hz)
            env["ZLC_DURATION"] = str(program.duration)
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.state_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Honour the PER-CALL timeout (e.g. a long wait_done) for the actual process kill, not just
            # the ZLC_TIMEOUT env var -- else a slow command is killed at the instance default while the
            # env says otherwise.  Falls back to the instance timeout when the caller gives none.
            timeout=timeout if timeout is not None else self.timeout,
        )
        log_path = self.state_dir / f"{action}.log"
        log_path.write_text(result.stdout, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            message = f"sequencer {action} command failed with code {result.returncode}. See {log_path}."
            tail = _log_tail(result.stdout)
            if tail:
                message = f"{message}\n\n--- {log_path.name} tail ---\n{tail}"
            raise RuntimeError(message)


def _log_tail(text: str, *, max_lines: int = 80, max_chars: int = 12_000) -> str:
    tail = "\n".join(str(text).splitlines()[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------- backend auto-selection
# Prefer the FASTEST transport that actually VERIFIES against this host's register layout, so the
# GUI on_pulse / API fire transparently ride the best available link with no client-side change.
_AUTO_BACKEND_PRIORITY = ("uart", "jtag-axi")            # fastest first; "command" is never auto-selected
_AUTO_UART_PROBE_TIMEOUT = 0.6                           # s -- an absent/dead UART must fail fast, not hang startup
_UART_BRIDGE_VIDS = (0x1A86, 0x10C4, 0x0403, 0x067B)    # CH340, CP210x, FTDI-VCP, PL2303


def _err_line(exc: Exception) -> str:
    return (str(exc).strip().splitlines() or [""])[0][:200]


def _solve_backend_params(channels):
    from fpga.pulse_streamer.host.image import solve_capacity
    part = os.environ.get("ZLC_PS_FPGA_PART", "xc7a35tfgg484-2")
    return solve_capacity(part, channel_count=max(1, len(list(channels)))).params


def _make_hardware_backend(name, *, channels, clock_hz, state_dir, uart_port, uart_baud):
    """Construct (do NOT start) the hardware session for a concrete backend name.  Single source of the
    session geometry (BRAM image solved from part + channel count) shared by the explicit and auto paths."""
    params = _solve_backend_params(channels)
    if name == "jtag-axi":
        from .axi_session import VivadoAxiStreamerSession
        return VivadoAxiStreamerSession(
            state_dir=state_dir, clock_hz=clock_hz,
            program_on_start=_env_bool("ZLC_PS_VIVADO_PROGRAM_ON_RUN", False), params=params,
        )
    if name == "uart":
        from .uart_session import UartStreamerSession
        return UartStreamerSession(
            state_dir=state_dir, clock_hz=clock_hz, params=params,
            port=uart_port or os.environ.get("ZLC_PS_UART_PORT"),
            baud=int(uart_baud or os.environ.get("ZLC_PS_UART_BAUD", 3_000_000)),
        )
    raise ValueError(f"unknown hardware backend {name!r}")


def _discover_uart_ports(explicit):
    """Ordered candidate serial ports for the UART link: an explicit --uart-port / ZLC_PS_UART_PORT wins;
    otherwise scan for a USB-UART bridge (CH340 first) so ``auto`` is plug-and-play.  Never raises."""
    if explicit:
        return [str(explicit)]
    env = os.environ.get("ZLC_PS_UART_PORT")
    if env:
        return [env]
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    ranked = []
    for p in list_ports.comports():
        vid = getattr(p, "vid", None)
        if vid in _UART_BRIDGE_VIDS:
            ranked.append((0 if vid == 0x1A86 else 1, str(p.device)))   # CH340 (the Da Vinci USB_UART) first
    ranked.sort()
    return [dev for _, dev in ranked]


def _warm_start_hardware(session, name, *, log, verify_layout=False, layout_timeout=None):
    """Bring a hardware session up to a verified, safe-idle state: open the link, (auto only) verify the
    programmed bitstream matches this host's register layout, zero the host-owned CTRL config, and run the
    transport self-test.  ``verify_layout`` stays False on the EXPLICIT paths so their bring-up ORDER
    (start -> clear -> self-test) is byte-for-byte unchanged.  Raises on any failure (caller closes)."""
    log(f"[{name}] opening control link before accepting clients...")
    session.start()
    if verify_layout:
        # reads CTRL LAYOUT_ID; an old/absent bitstream returns 0 (or the link times out) -> raises,
        # which is exactly how ``auto`` decides this transport is not usable and falls back.
        saved = getattr(session, "action_timeout", None)
        if layout_timeout is not None:
            try: session.action_timeout = layout_timeout
            except Exception: pass
        try:
            session.check_register_layout()
        finally:
            if layout_timeout is not None:
                try: session.action_timeout = saved
                except Exception: pass
    log(f"[{name}] clearing host-owned CTRL config (delays + clk mask) to a safe idle state...")
    session.clear_host_config()
    self_test_env = {"jtag-axi": "ZLC_PS_AXI_SELF_TEST", "uart": "ZLC_PS_UART_SELF_TEST"}[name]
    if _env_bool(self_test_env, True):
        log(f"[{name}] verifying link (write + read-back self-test)...")
        (session.axi_self_test if name == "jtag-axi" else session.link_self_test)()
        log(f"[{name}] self-test OK.")
    return session


def _auto_select_backend(*, channels, clock_hz, state_dir, uart_port, uart_baud, warm_start, log):
    """Probe transports fastest-first; return (name, started_session) for the first that verifies.
    UART is a CHEAP probe (open serial + read LAYOUT_ID); Vivado JTAG-AXI is only brought up if no UART
    link answers.  With --no-warm-start and no UART, JTAG is selected LAZILY (no Vivado spin at startup)."""
    tried: list[tuple[str, str]] = []
    ports = _discover_uart_ports(uart_port)
    if not ports:
        reason = ("no UART serial port found (need pyserial installed + a CH340/CP210x/FTDI USB-UART "
                  "plugged, or pass --uart-port)")
        tried.append(("uart", reason))
        log(f"  auto: skipping UART -- {reason}")
    for port in ports:
        session = _make_hardware_backend("uart", channels=channels, clock_hz=clock_hz,
                                         state_dir=state_dir, uart_port=port, uart_baud=uart_baud)
        try:
            _warm_start_hardware(session, "uart", log=log, verify_layout=True,
                                 layout_timeout=_AUTO_UART_PROBE_TIMEOUT)
            log(f"Auto-selected backend: UART on {port} -- fastest verified link "
                "(~82 ms apply / ~sub-ms scan step).")
            return "uart", session
        except Exception as exc:
            try: session.close()
            except Exception: pass
            tried.append((f"uart:{port}", _err_line(exc)))
            log(f"  auto: UART on {port} not usable -- {_err_line(exc)}")
    # No UART link answered -> fall back to the (slower) Vivado JTAG-to-AXI path.
    session = _make_hardware_backend("jtag-axi", channels=channels, clock_hz=clock_hz,
                                     state_dir=state_dir, uart_port=None, uart_baud=uart_baud)
    if not warm_start:
        log("Auto-selected backend: jtag-axi (deferred bring-up; --no-warm-start) -- no UART link verified.")
        return "jtag-axi", session
    try:
        _warm_start_hardware(session, "jtag-axi", log=log, verify_layout=True)
        log("Auto-selected backend: jtag-axi (Vivado hw_axi, ~1 s apply) -- no faster UART link verified.")
        return "jtag-axi", session
    except Exception as exc:
        try: session.close()
        except Exception: pass
        tried.append(("jtag-axi", _err_line(exc)))
    raise RuntimeError(
        "auto backend: no usable transport verified.\n  "
        + "\n  ".join(f"{n}: {r}" for n, r in tried)
        + "\n  Fix: program the FPGA (fpga/build_and_program.bat) and/or plug the USB_UART cable, "
        "or choose one explicitly with --backend jtag-axi|uart."
    )


def run_server(
    *,
    channels: Sequence[str],
    port_catalog: PortCatalog | dict | None = None,
    channel_labels: dict[str, str] | None = None,
    xdc: str | Path | None = None,
    host: str = "0.0.0.0",
    port: int = 18861,
    clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
    state_dir: str | Path = "zlc_sequencer_state",
    prepare_command: str | None = None,
    fire_command: str | None = None,
    wait_done_command: str | None = None,
    safe_state_command: str | None = None,
    command_timeout: float | None = None,
    backend: str = "command",
    warm_start: bool = True,
    uart_port: str | None = None,
    uart_baud: int = 3_000_000,
):
    """Start the RPyC sequencer service used by ``RemoteSequencer``."""

    channels = _split_channels(channels)
    if port_catalog is None and channel_labels is None and xdc is not None:
        from .fpga_pulse_streamer import infer_xdc_channel_labels

        channel_labels = infer_xdc_channel_labels(
            xdc, default=len(channels), max_count=len(channels))
    catalog = coerce_port_catalog(
        port_catalog, channels=channels, channel_labels=channel_labels)
    channels = list(catalog.raw_lanes)
    backend_name = str(backend).strip().lower().replace("_", "-")
    if backend_name == "auto":
        # Probe transports fastest-first and use the first that VERIFIES against this host's register
        # layout, so on_pulse/fire transparently ride the best available link (UART ~82 ms > JTAG ~1 s).
        backend_name, hardware_backend = _auto_select_backend(
            channels=channels, clock_hz=clock_hz, state_dir=state_dir,
            uart_port=uart_port, uart_baud=uart_baud, warm_start=warm_start, log=print,
        )
        prepare_callback = hardware_backend.prepare
        fire_callback = hardware_backend.fire
        wait_done_callback = hardware_backend.wait_done
        safe_state_callback = hardware_backend.safe_state
        scan_progress_callback = hardware_backend.scan_progress
    elif backend_name in {"jtag-axi", "axi", "loader", "edge-table"}:
        # The FINAL affine edge-table engine (1-tick FIFO prefetch + 2-bank streaming scan), driven over
        # JTAG-to-AXI (hw_axi).  Construction geometry + bring-up (start -> clear host CTRL config ->
        # AXI burst self-test) are shared with ``auto`` via the helpers -- single source of truth.
        hardware_backend = _make_hardware_backend(
            "jtag-axi", channels=channels, clock_hz=clock_hz, state_dir=state_dir,
            uart_port=uart_port, uart_baud=uart_baud,
        )
        if warm_start:
            _warm_start_hardware(hardware_backend, "jtag-axi", log=print)
        prepare_callback = hardware_backend.prepare
        fire_callback = hardware_backend.fire
        wait_done_callback = hardware_backend.wait_done
        safe_state_callback = hardware_backend.safe_state
        scan_progress_callback = hardware_backend.scan_progress   # live scan cursor for the GUI poll
    elif backend_name in {"uart", "serial"}:
        # SAME edge-table engine + register map over the UART fast-control side-channel instead of
        # Vivado-Tcl JTAG -- a byte-identical transport swap (~82 ms apply / ~sub-ms scan step vs ~1 s).
        # Construction + bring-up shared with ``auto`` via the helpers.
        hardware_backend = _make_hardware_backend(
            "uart", channels=channels, clock_hz=clock_hz, state_dir=state_dir,
            uart_port=uart_port, uart_baud=uart_baud,
        )
        if warm_start:
            _warm_start_hardware(hardware_backend, "uart", log=print)
        prepare_callback = hardware_backend.prepare
        fire_callback = hardware_backend.fire
        wait_done_callback = hardware_backend.wait_done
        safe_state_callback = hardware_backend.safe_state
        scan_progress_callback = hardware_backend.scan_progress
    elif backend_name == "command":
        hardware_backend = CommandSequencerBackend(
            Path(state_dir),
            prepare_command=prepare_command,
            fire_command=fire_command,
            wait_done_command=wait_done_command,
            safe_state_command=safe_state_command,
            timeout=command_timeout,
        )
        prepare_callback = hardware_backend.prepare
        fire_callback = hardware_backend.fire
        wait_done_callback = hardware_backend.wait_done
        safe_state_callback = hardware_backend.safe_state
        scan_progress_callback = None        # the command backend has no live scan cursor
    else:
        raise ValueError("backend must be 'auto', 'jtag-axi', 'uart', or 'command'.")
    cache_prepared = _env_bool("ZLC_SEQUENCER_CACHE_PREPARED", False)
    service = SequencerService(
        channels=channels,
        port_catalog=catalog,
        clock_hz=clock_hz,
        prepare_callback=prepare_callback,
        fire_callback=fire_callback,
        wait_done_callback=wait_done_callback,
        safe_state_callback=safe_state_callback,
        scan_progress_callback=scan_progress_callback,
        cache_prepared=cache_prepared,
    )
    print("Zou_lab_control sequencer service")
    print(json.dumps(service.snapshot(), indent=2))
    print(f"Listening on {host}:{port}")
    _print_client_endpoints(host, port)
    print(f"State directory: {Path(state_dir).resolve()}")
    print(f"Backend: {backend_name}")
    print(f"Prepare cache: {'on' if cache_prepared else 'off'}")
    return serve_runtime_sequencer(service, host=host, port=port, start=True)


def _print_client_endpoints(host: str, port: int) -> None:
    addresses = _client_addresses(host)
    if not addresses:
        print("Client endpoints: no non-loopback IPv4 address detected")
        return
    print("Client endpoints:")
    for address in addresses:
        print(f"  {address}:{int(port)}")
    print("Notebook connect example:")
    print(f'  exp = na.connect("remote_template", sequencer={{"host": "{addresses[0]}", "port": {int(port)}}}, open_devices=True)')


def _client_addresses(bind_host: str) -> list[str]:
    host = str(bind_host).strip()
    if host and host not in {"0.0.0.0", "::"}:
        return [host]
    return _local_ipv4_addresses()


def _local_ipv4_addresses() -> list[str]:
    addresses: list[str] = []

    def add(value) -> None:
        try:
            ip = str(value).strip()
            packed = socket.inet_aton(ip)
        except OSError:
            return
        if ip == "0.0.0.0" or ip.startswith("127."):
            return
        if packed not in [socket.inet_aton(existing) for existing in addresses]:
            addresses.append(ip)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            add(sock.getsockname()[0])
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            add(info[4][0])
    except OSError:
        pass

    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            add(ip)
    except OSError:
        pass

    return addresses


def _split_channels(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = []
        for item in value:
            raw.extend(str(item).replace(",", " ").split())
    if not raw:
        raise ValueError("channels must not be empty.")
    return raw


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Start the Zou_lab_control neutral-atom sequencer service on the FPGA/Vivado computer.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18861)
    parser.add_argument("--channels", nargs="+", required=True, help="Sequencer channels, e.g. ch00 ch01 ... inferred from the selected XDC.")
    parser.add_argument(
        "--xdc", default=None,
        help="Board XDC used once at startup to build the logical PortCatalog (DAC/clock/digital).")
    parser.add_argument("--clock-hz", type=float, default=DEFAULT_RUNTIME_CLOCK_HZ)
    parser.add_argument("--state-dir", default="zlc_sequencer_state")
    parser.add_argument("--prepare-command", default=None)
    parser.add_argument("--fire-command", default=None)
    parser.add_argument("--wait-done-command", default=None)
    parser.add_argument("--safe-state-command", default=None)
    parser.add_argument("--command-timeout", type=float, default=None)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "jtag-axi", "uart", "command"],
        help="Hardware backend.  auto (default) probes fastest-first and uses the first link that "
        "VERIFIES against this host's register layout: the UART fast-control side-channel (~82 ms apply) "
        "if a responding CH340/USB-UART is present, else JTAG-to-AXI over a persistent Vivado hw_axi "
        "session (~1 s).  jtag-axi / uart force one; command shells out per action.",
    )
    parser.add_argument("--uart-port", default=None, help="Serial port for --backend uart (e.g. COM3, /dev/ttyUSB1).")
    parser.add_argument("--uart-baud", type=int, default=3_000_000, help="UART baud for --backend uart (default 3 Mbaud).")
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="For session backends, delay Vivado startup until the first prepare call.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_server(
        channels=_split_channels(args.channels),
        xdc=args.xdc,
        host=args.host,
        port=args.port,
        clock_hz=args.clock_hz,
        state_dir=args.state_dir,
        prepare_command=args.prepare_command,
        fire_command=args.fire_command,
        wait_done_command=args.wait_done_command,
        safe_state_command=args.safe_state_command,
        command_timeout=args.command_timeout,
        backend=args.backend,
        warm_start=not args.no_warm_start,
        uart_port=args.uart_port,
        uart_baud=args.uart_baud,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["CommandSequencerBackend", "build_arg_parser", "main", "run_server"]
