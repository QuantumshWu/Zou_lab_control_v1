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

from .sequencer import RuntimeSequenceProgram, SequencerService, serve_runtime_sequencer


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
            env["ZLC_TRIGGER_COUNT"] = str(program.trigger_count)
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.state_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout,
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


def run_server(
    *,
    channels: Sequence[str],
    host: str = "0.0.0.0",
    port: int = 18861,
    clock_hz: float = 250e6,
    trigger_channels: Sequence[str] = ("qcm_trigger",),
    state_dir: str | Path = "zlc_sequencer_state",
    prepare_command: str | None = None,
    fire_command: str | None = None,
    wait_done_command: str | None = None,
    safe_state_command: str | None = None,
    command_timeout: float | None = None,
    backend: str = "command",
    warm_start: bool = True,
):
    """Start the RPyC sequencer service used by ``RemoteSequencer``."""

    backend_name = str(backend).strip().lower().replace("_", "-")
    if backend_name in {"vivado-session", "persistent-vivado", "fpga-pulse-streamer"}:
        from .fpga_pulse_streamer import VivadoPulseStreamerSession

        hardware_backend = VivadoPulseStreamerSession(state_dir=state_dir)
        if warm_start:
            print("Starting persistent Vivado session before accepting clients...")
            hardware_backend.start()
        prepare_callback = hardware_backend.prepare
        fire_callback = hardware_backend.fire
        wait_done_callback = hardware_backend.wait_done
        safe_state_callback = hardware_backend.safe_state
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
    else:
        raise ValueError("backend must be 'vivado-session' or 'command'.")
    service = SequencerService(
        channels=channels,
        clock_hz=clock_hz,
        trigger_channels=trigger_channels,
        prepare_callback=prepare_callback,
        fire_callback=fire_callback,
        wait_done_callback=wait_done_callback,
        safe_state_callback=safe_state_callback,
    )
    print("Zou_lab_control sequencer service")
    print(json.dumps(service.snapshot(), indent=2))
    print(f"Listening on {host}:{port}")
    _print_client_endpoints(host, port)
    print(f"State directory: {Path(state_dir).resolve()}")
    print(f"Backend: {backend_name}")
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
    parser.add_argument("--channels", nargs="+", required=True, help="Sequencer channels, e.g. ch00 ch01 ch02 ch03 ... ch39.")
    parser.add_argument("--trigger-channels", nargs="+", default=["qcm_trigger"])
    parser.add_argument("--clock-hz", type=float, default=250e6)
    parser.add_argument("--state-dir", default="zlc_sequencer_state")
    parser.add_argument("--prepare-command", default=None)
    parser.add_argument("--fire-command", default=None)
    parser.add_argument("--wait-done-command", default=None)
    parser.add_argument("--safe-state-command", default=None)
    parser.add_argument("--command-timeout", type=float, default=None)
    parser.add_argument(
        "--backend",
        default="command",
        choices=["command", "vivado-session"],
        help="Hardware backend. vivado-session keeps one Vivado Tcl process alive for lower runtime latency.",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="For vivado-session, delay Vivado startup until the first prepare call.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_server(
        channels=_split_channels(args.channels),
        trigger_channels=_split_channels(args.trigger_channels),
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
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["CommandSequencerBackend", "build_arg_parser", "main", "run_server"]
