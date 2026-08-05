"""Standalone composition root for the current frozen-bitstream pulse server."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fpga.pulse_streamer.host.image import (
    FROZEN_CLOCK_HZ,
    StreamerParams,
    build_fingerprint,
)

from .artifact import CompiledPulseArtifact
from .deployment import (
    APPROVED_DEPLOYED_TARGET_ABI,
    _load_deployed_streamer_config,
    validate_deployed_target,
)
from .evidence import PulseBackendCompletion
from .manifest import PulseTargetManifest, pulse_target_manifest_from_xdc
from .server import PulseExecutionService, serve_pulse_execution_service
from .target import PulseTarget, load_pulse_target
from .transport import (
    DeployedStreamerSession as _CurrentDeployedStreamerSession,
    InterprocessDeviceLease,
    RegisterLayoutMismatch,
    UartRegisterTransport,
    VivadoAxiRegisterTransport,
    verify_register_layout,
)


# The deployment policy, not a taste setting: UART is the ordinary control path
# because a resident vivado.exe costs 1-2 GB and one JTAG transaction costs
# milliseconds.  ``auto`` therefore probes UART first and only keeps JTAG-to-AXI
# as the fallback for a machine whose bridge cable is absent.
BACKEND_CHOICES = ("auto", "uart", "jtag-axi")
DEFAULT_UART_BAUD = 3_000_000
UART_PROBE_TIMEOUT = 0.5
# A missing pyserial silently disqualifies every UART candidate, which reads as
# "the cable does nothing".  Name the exact interpreter that must grow the
# dependency instead of letting the fallback swallow the cause.
_PYSERIAL_HINT = (
    "pyserial is missing; install it for this interpreter with: "
    f'"{sys.executable}" -m pip install pyserial'
)


class _DeployedStreamerSessionPort(Protocol):
    def start(self): ...

    def clear_host_config(self) -> None: ...

    def check_register_layout(self) -> None: ...

    def transport_self_test(self) -> None: ...

    def safe_state(self) -> None: ...

    def request_interrupt(self) -> None: ...

    def prepare(self, artifact: CompiledPulseArtifact) -> None: ...

    def fire(self, artifact: CompiledPulseArtifact) -> None: ...

    def await_completion(
        self,
        artifact: CompiledPulseArtifact,
        timeout: float | None,
    ) -> PulseBackendCompletion | None: ...

    def wait_continuous_failure(
        self,
        artifact: CompiledPulseArtifact,
        timeout: float,
    ) -> str | None: ...

    def snapshot(self) -> dict[str, object]: ...

    def close(self) -> None: ...


@dataclass
class PulseServerRuntime:
    service: PulseExecutionService
    rpc_server: object
    session: _DeployedStreamerSessionPort
    _closed: bool = field(default=False, init=False, repr=False)
    _rpc_closed: bool = field(default=False, init=False, repr=False)
    _session_closed: bool = field(default=False, init=False, repr=False)
    _close_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            close_server = getattr(self.rpc_server, "close", None)
            failure: BaseException | None = None
            if not self._rpc_closed:
                try:
                    if callable(close_server):
                        close_server()
                    self._rpc_closed = True
                except BaseException as error:
                    failure = error
            if not self._session_closed:
                try:
                    self.service.safe_state()
                except BaseException as error:
                    if failure is None:
                        failure = error
                    else:
                        failure.add_note(
                            "pulse physical SAFE also failed: "
                            f"{type(error).__name__}: {error}"
                        )
                else:
                    try:
                        # The session owns the physical SAFE receipt.  close()
                        # observes that receipt and revokes transport ownership;
                        # it must not issue the same physical SAFE a second time.
                        self.session.close()
                        self._session_closed = True
                    except BaseException as error:
                        if failure is None:
                            failure = error
                        else:
                            failure.add_note(
                                "pulse session close also failed: "
                                f"{type(error).__name__}: {error}"
                            )
            self._closed = self._rpc_closed and self._session_closed
            if failure is not None:
                raise failure


def build_service_for_session(
    manifest: PulseTargetManifest,
    session: _DeployedStreamerSessionPort,
    *,
    params: StreamerParams,
    clock_hz: float,
) -> PulseExecutionService:
    if not isinstance(manifest, PulseTargetManifest):
        raise TypeError("manifest must be PulseTargetManifest")
    target = manifest.target
    if not isinstance(params, StreamerParams):
        raise TypeError("params must be StreamerParams")
    geometry = params
    clock = float(clock_hz)
    if clock != FROZEN_CLOCK_HZ:
        raise ValueError(
            f"deployment clock must match the frozen RTL ({FROZEN_CLOCK_HZ:g} Hz)"
        )
    validate_deployed_target(target, geometry)
    if getattr(session, "params", None) != geometry:
        raise ValueError("hardware session geometry differs from server deployment geometry")
    if float(getattr(session, "clock_hz", 0.0)) != clock:
        raise ValueError("hardware session clock differs from server deployment clock")
    return PulseExecutionService(
        manifest,
        clock_hz=clock,
        backend=session,
        params=geometry,
    )


def bring_up_frozen_session(session: _DeployedStreamerSessionPort) -> None:
    """Verify the approved deployment without synthesizing or programming hardware."""

    session.start()
    layout_verified = False
    try:
        session.check_register_layout()
        layout_verified = True
        session.clear_host_config()
        session.transport_self_test()
        session.safe_state()
    except BaseException as primary:
        if layout_verified:
            try:
                session.safe_state()
            except BaseException as error:
                primary.add_note(
                    "pulse bring-up SAFE also failed: "
                    f"{type(error).__name__}: {error}"
                )
        try:
            session.close()
        except BaseException as error:
            primary.add_note(
                "pulse bring-up session close also failed: "
                f"{type(error).__name__}: {error}"
            )
        raise


@dataclass(frozen=True)
class BackendResolution:
    """The one startup decision that selects the physical transport."""

    requested: str
    backend: str
    uart_port: str | None
    reason: str
    attempts: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()


class BackendResolutionError(RuntimeError):
    """An explicitly requested backend could not be established."""

    def __init__(self, message: str, *, attempts: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


def _list_uart_ports() -> tuple[str, ...]:
    """Enumerate serial ports lazily, physical USB bridges before virtual ones.

    Ordering only, never exclusion: an ordinary workstation also publishes
    Bluetooth and serial-over-LAN COM ports, and the streamer's USB-UART is
    indistinguishable from them by name alone.  Probing the ports that carry a
    USB VID/PID first means the board normally answers on the first attempt, so
    the decoys are never opened at all -- but they are still probed if nothing
    on USB answers, because the operator may not know which port this is.
    """

    from serial.tools import list_ports

    usb: list[str] = []
    virtual: list[str] = []
    for descriptor in list_ports.comports():
        port = str(getattr(descriptor, "device", descriptor)).strip()
        if not port or port in usb or port in virtual:
            continue
        group = usb if getattr(descriptor, "vid", None) is not None else virtual
        group.append(port)
    return tuple(usb + virtual)


def _uart_candidates(
    uart_port: str | None,
    port_provider: Callable[[], Iterable[object]] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the ports worth probing plus an enumeration failure, if one occurred."""

    selected = str(uart_port).strip() if uart_port is not None else ""
    if selected:
        return (selected,), ()
    provider = port_provider or _list_uart_ports
    try:
        raw_ports = provider()
    except ModuleNotFoundError as exc:
        name = str(getattr(exc, "name", "") or "")
        if name.startswith("serial") or "serial" in str(exc).lower():
            return (), (_PYSERIAL_HINT,)
        return (), (f"port enumeration failed: {type(exc).__name__}",)
    except Exception as exc:
        return (), (f"port enumeration failed: {type(exc).__name__}: {exc}",)
    if isinstance(raw_ports, str):
        raw_ports = (raw_ports,)
    ports: list[str] = []
    for descriptor in raw_ports:
        port = str(getattr(descriptor, "device", descriptor)).strip()
        if port and port not in ports:
            ports.append(port)
    return tuple(ports), (() if ports else ("no UART ports detected",))


def _probe_uart_port(
    port: str,
    timeout: float,
    *,
    params: StreamerParams,
    state_dir: str | Path,
    baud: int,
) -> None:
    """Decide one candidate port by the deployed geometry handshake alone."""

    transport = UartRegisterTransport(
        state_dir=state_dir,
        port=port,
        baud=baud,
        action_timeout=timeout,
    )
    try:
        transport.start()
        # One READ frame and nothing else: the probe never writes a register,
        # LOADs or FIREs, so a port owned by another instrument keeps its state.
        verify_register_layout(transport, params)
    finally:
        # A rejected candidate must not stay held open: the next attempt, and a
        # later real session on the winning port, both need the device released.
        transport.close()


def _probe_failure_reason(exc: BaseException) -> str:
    """Collapse probe exceptions into the operator-facing failure categories."""

    message = " ".join(str(exc).split())
    lower = message.lower()
    if isinstance(exc, ModuleNotFoundError) and (
        str(getattr(exc, "name", "") or "").startswith("serial") or "serial" in lower
    ):
        return _PYSERIAL_HINT
    if isinstance(exc, RegisterLayoutMismatch):
        return "geometry fingerprint mismatch"
    if type(exc).__name__ == "FrameError" or "crc" in lower:
        return "CRC error"
    if isinstance(exc, TimeoutError) or "timeout" in lower or "timed out" in lower:
        return "no reply before timeout"
    if type(exc).__name__ in {"SerialException", "FileNotFoundError", "PermissionError"}:
        return "port open failed"
    if "could not open port" in lower or "access is denied" in lower:
        return "port open failed"
    return f"{type(exc).__name__}: {message or 'probe failed'}"


def resolve_backend(
    requested: str = "auto",
    *,
    params: StreamerParams,
    state_dir: str | Path,
    uart_port: str | None = None,
    uart_baud: int = DEFAULT_UART_BAUD,
    port_provider: Callable[[], Iterable[object]] | None = None,
    probe: Callable[[str, float], None] | None = None,
    on_candidates: Callable[[tuple[str, ...], tuple[str, ...]], None] | None = None,
) -> BackendResolution:
    """Resolve one physical backend, probing UART first unless JTAG was demanded."""

    choice = str(requested).strip().lower()
    if choice not in BACKEND_CHOICES:
        raise ValueError(f"backend must be one of {BACKEND_CHOICES}, not {requested!r}")
    if choice == "jtag-axi":
        return BackendResolution(
            choice,
            "jtag-axi",
            None,
            "explicit jtag-axi backend; UART probe skipped",
        )
    if not isinstance(params, StreamerParams):
        raise TypeError("params must be StreamerParams")

    ports, setup_attempts = _uart_candidates(uart_port, port_provider)
    if on_candidates is not None:
        on_candidates(ports, setup_attempts)
    attempts = list(setup_attempts)
    probe_fn = probe
    if probe_fn is None:
        probe_fn = lambda port, timeout: _probe_uart_port(
            port,
            timeout,
            params=params,
            state_dir=state_dir,
            baud=int(uart_baud),
        )
    for port in ports:
        try:
            probe_fn(port, UART_PROBE_TIMEOUT)
        except Exception as exc:
            attempts.append(f"{port}: {_probe_failure_reason(exc)}")
            continue
        attempts.append(f"{port}: geometry fingerprint matched")
        selected = "explicit uart" if choice == "uart" else "auto"
        megabaud = int(uart_baud) / 1_000_000
        return BackendResolution(
            choice,
            "uart",
            port,
            f"{selected} selected UART {port}@{megabaud:g}M after the geometry handshake",
            tuple(attempts),
            ports,
        )

    failure = "; ".join(attempts) if attempts else "no UART ports detected"
    if choice == "uart":
        # An explicitly demanded UART never degrades into JTAG: that would hide
        # the cable/driver fault the operator asked this run to prove.
        raise BackendResolutionError(
            f"explicit UART backend failed; no matching device: {failure}",
            attempts=attempts,
        )
    return BackendResolution(
        choice,
        "jtag-axi",
        None,
        f"auto fallback to jtag-axi after the UART probe: {failure}",
        tuple(attempts),
        ports,
    )


def open_deployed_session(
    backend: str,
    *,
    target: PulseTarget,
    state_dir: str | Path,
    params: StreamerParams,
    clock_hz: float,
    uart_port: str | None,
    uart_baud: int,
) -> _DeployedStreamerSessionPort:
    if backend == "jtag-axi":
        transport = VivadoAxiRegisterTransport(
            state_dir=state_dir,
        )
    elif backend == "uart":
        if not uart_port:
            raise ValueError("the uart backend requires the port resolve_backend() proved")
        transport = UartRegisterTransport(
            state_dir=state_dir,
            port=uart_port,
            baud=uart_baud,
        )
    else:
        raise ValueError(
            "open_deployed_session takes a resolved backend ('jtag-axi' or 'uart'); "
            f"call resolve_backend() first, not {backend!r}"
        )
    return _CurrentDeployedStreamerSession(
        transport,
        device_lease=InterprocessDeviceLease(),
        deployed_target=target,
        params=params,
        clock_hz=clock_hz,
    )


def _log_uart_candidates(
    candidates: tuple[str, ...],
    _setup_attempts: tuple[str, ...],
) -> None:
    # Emitted before the probe runs so a slow candidate list is visible while it
    # is still being walked; every attempt outcome is reported by the resolution.
    logging.info(
        "UART probe candidates: %s",
        ", ".join(candidates) if candidates else "none",
    )


def _log_backend_resolution(resolution: BackendResolution) -> None:
    for attempt in resolution.attempts:
        logging.info("UART probe: %s", attempt)
    logging.info(
        "backend resolved: requested=%s selected=%s uart_port=%s (%s)",
        resolution.requested,
        resolution.backend,
        resolution.uart_port or "-",
        resolution.reason,
    )
    if resolution.backend == "jtag-axi":
        logging.info(
            "JTAG-to-AXI keeps a resident vivado.exe (~1-2 GB) and pays"
            " milliseconds per transaction; prefer the UART bridge when available"
        )


def build_server_runtime(
    manifest: PulseTargetManifest,
    *,
    backend: str = "auto",
    state_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 18861,
    uart_port: str | None = None,
    uart_baud: int = DEFAULT_UART_BAUD,
    params: StreamerParams,
    clock_hz: float,
    start: bool = False,
) -> PulseServerRuntime:
    if not isinstance(manifest, PulseTargetManifest):
        raise TypeError("manifest must be PulseTargetManifest")
    target = manifest.target
    if not isinstance(params, StreamerParams):
        raise TypeError("params must be StreamerParams")
    geometry = params
    clock = float(clock_hz)
    if clock != FROZEN_CLOCK_HZ:
        raise ValueError(
            f"deployment clock must match the frozen RTL ({FROZEN_CLOCK_HZ:g} Hz)"
        )
    resolution = resolve_backend(
        backend,
        params=geometry,
        state_dir=state_dir,
        uart_port=uart_port,
        uart_baud=uart_baud,
        on_candidates=_log_uart_candidates,
    )
    _log_backend_resolution(resolution)
    session = open_deployed_session(
        resolution.backend,
        target=target,
        state_dir=state_dir,
        params=geometry,
        clock_hz=clock,
        uart_port=resolution.uart_port,
        uart_baud=uart_baud,
    )
    try:
        service = build_service_for_session(
            manifest,
            session,
            params=geometry,
            clock_hz=clock,
        )
    except BaseException:
        session.close()
        raise
    bring_up_frozen_session(session)
    try:
        rpc_server = serve_pulse_execution_service(
            service,
            host=host,
            port=port,
            start=False,
        )
    except BaseException as primary:
        try:
            session.safe_state()
        except BaseException as error:
            primary.add_note(
                "pulse RPC startup SAFE also failed: "
                f"{type(error).__name__}: {error}"
            )
        try:
            session.close()
        except BaseException as error:
            primary.add_note(
                "pulse RPC startup session close also failed: "
                f"{type(error).__name__}: {error}"
            )
        raise
    runtime = PulseServerRuntime(service, rpc_server, session)
    if start:
        try:
            rpc_server.start()
        finally:
            runtime.close()
    return runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the current-only pulse execution server")
    parser.add_argument("--target", required=True, help="canonical zlc_pulse.PulseTarget file")
    parser.add_argument(
        "--xdc",
        required=True,
        help="server-side constraints file owning deployed signal/package-pin bindings",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="auto",
        help="transport policy: probe UART first, or demand one explicit backend",
    )
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18861)
    parser.add_argument(
        "--uart-port",
        help="probe only this UART port instead of every enumerated port",
    )
    parser.add_argument("--uart-baud", type=int, default=DEFAULT_UART_BAUD)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="report the deployment and transport policy without touching hardware",
    )
    return parser


def _report_config(
    arguments: argparse.Namespace,
    manifest: PulseTargetManifest,
    params: StreamerParams,
    clock_hz: float,
    config_path: Path,
) -> None:
    """Print the startup facts without opening a transport or probing a port."""

    target = manifest.target
    validate_deployed_target(target, params)
    candidates, setup_attempts = _uart_candidates(arguments.uart_port)
    print(f"python={sys.executable}")
    print(f"backend={arguments.backend}")
    print(f"uart_port={arguments.uart_port or 'auto-discover'}")
    print(f"uart_ports_detected={','.join(candidates) if candidates else 'none'}")
    if setup_attempts:
        print(f"uart_port_note={'; '.join(setup_attempts)}")
    print(f"uart_baud={arguments.uart_baud}")
    print(f"listen_bind={arguments.host}:{arguments.port}")
    print(f"config={config_path}")
    print(f"clock_hz={clock_hz:g}")
    print(f"geometry={params}")
    print(f"geometry_fingerprint={hex(build_fingerprint(params) & 0xFFFFFFFF)}")
    print(f"target_abi={target.abi_fingerprint}")
    print(f"published_ports={len(manifest.ports)}")
    print(f"raw_lanes={len(target.raw_lanes)}")


def main(argv: list[str] | None = None) -> int:
    # This executable owns its process, so expose the RPyC lifecycle in the
    # launcher window.  Its ``server started`` record is emitted only after the
    # socket enters LISTEN; accepted/welcome/goodbye also make GUI connections
    # observable instead of leaving a silent console.
    logging.basicConfig(
        level=logging.INFO,
        format="ZLC pulse server: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    arguments = build_arg_parser().parse_args(argv)
    target = load_pulse_target(arguments.target)
    manifest = pulse_target_manifest_from_xdc(target, arguments.xdc)
    params, clock_hz, config_path = _load_deployed_streamer_config()
    if arguments.check_config:
        _report_config(arguments, manifest, params, clock_hz, config_path)
        return 0
    logging.info("deployment geometry loaded from %s", config_path)
    try:
        build_server_runtime(
            manifest,
            backend=arguments.backend,
            state_dir=arguments.state_dir,
            host=arguments.host,
            port=arguments.port,
            uart_port=arguments.uart_port,
            uart_baud=arguments.uart_baud,
            params=params,
            clock_hz=clock_hz,
            start=True,
        )
    except BackendResolutionError as error:
        # The operator demanded one transport; report why it is unreachable
        # instead of raising a traceback that buries the per-port evidence.
        logging.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "APPROVED_DEPLOYED_TARGET_ABI",
    "BACKEND_CHOICES",
    "DEFAULT_UART_BAUD",
    "UART_PROBE_TIMEOUT",
    "BackendResolution",
    "BackendResolutionError",
    "PulseServerRuntime",
    "bring_up_frozen_session",
    "build_arg_parser",
    "build_server_runtime",
    "build_service_for_session",
    "main",
    "open_deployed_session",
    "resolve_backend",
    "validate_deployed_target",
]
