"""Standalone composition root for the current frozen-bitstream pulse server."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fpga.pulse_streamer.host.image import (
    StreamerParams,
    default_clock_hz,
    default_params,
)

from .artifact import CompiledPulseArtifact
from .deployment import APPROVED_DEPLOYED_TARGET_ABI, validate_deployed_target
from .evidence import PulseBackendCompletion
from .server import PulseExecutionService, serve_pulse_execution_service
from .target import PulseTarget, load_pulse_target
from .transport import (
    DeployedStreamerSession as CurrentDeployedStreamerSession,
    InterprocessDeviceLease,
    UartRegisterTransport,
    VivadoAxiRegisterTransport,
)


class DeployedStreamerSession(Protocol):
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
    session: DeployedStreamerSession
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
                safety_error: BaseException | None = None
                try:
                    self.service.safe_state()
                except BaseException as error:
                    safety_error = error
                try:
                    # A successful session.close() is itself a verified SAFE retry
                    # followed by transport revocation, so it recovers a transient
                    # first safe_state failure.
                    self.session.close()
                    self._session_closed = True
                    safety_error = None
                except BaseException as error:
                    if safety_error is not None:
                        error.add_note(
                            "the preceding explicit safe_state also failed: "
                            f"{type(safety_error).__name__}: {safety_error}"
                        )
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
    target: PulseTarget,
    session: DeployedStreamerSession,
    *,
    params: StreamerParams | None = None,
    clock_hz: float | None = None,
) -> PulseExecutionService:
    geometry = params or default_params()
    clock = float(default_clock_hz() if clock_hz is None else clock_hz)
    validate_deployed_target(target, geometry)
    if getattr(session, "params", None) != geometry:
        raise ValueError("hardware session geometry differs from server deployment geometry")
    if float(getattr(session, "clock_hz", 0.0)) != clock:
        raise ValueError("hardware session clock differs from server deployment clock")
    return PulseExecutionService(
        target,
        clock_hz=clock,
        backend=session,
        params=geometry,
    )


def bring_up_frozen_session(session: DeployedStreamerSession) -> None:
    """Verify the approved deployment without synthesizing or programming hardware."""

    session.start()
    layout_verified = False
    try:
        session.check_register_layout()
        layout_verified = True
        session.clear_host_config()
        session.transport_self_test()
        session.safe_state()
    except BaseException:
        if layout_verified:
            try:
                session.safe_state()
            except BaseException:
                pass
        session.close()
        raise


def open_deployed_session(
    backend: str,
    *,
    target: PulseTarget,
    state_dir: str | Path,
    params: StreamerParams,
    clock_hz: float,
    uart_port: str | None,
    uart_baud: int,
) -> DeployedStreamerSession:
    if backend == "jtag-axi":
        transport = VivadoAxiRegisterTransport(
            state_dir=state_dir,
        )
    elif backend == "uart":
        if not uart_port:
            raise ValueError("--uart-port is required for the uart backend")
        transport = UartRegisterTransport(
            state_dir=state_dir,
            port=uart_port,
            baud=uart_baud,
        )
    else:
        raise ValueError("backend must be 'jtag-axi' or 'uart'")
    return CurrentDeployedStreamerSession(
        transport,
        device_lease=InterprocessDeviceLease(),
        deployed_target=target,
        params=params,
        clock_hz=clock_hz,
    )


def build_server_runtime(
    target: PulseTarget,
    *,
    backend: str,
    state_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 18861,
    uart_port: str | None = None,
    uart_baud: int = 3_000_000,
    params: StreamerParams | None = None,
    clock_hz: float | None = None,
    start: bool = False,
) -> PulseServerRuntime:
    geometry = params or default_params()
    clock = float(default_clock_hz() if clock_hz is None else clock_hz)
    session = open_deployed_session(
        backend,
        target=target,
        state_dir=state_dir,
        params=geometry,
        clock_hz=clock,
        uart_port=uart_port,
        uart_baud=uart_baud,
    )
    try:
        service = build_service_for_session(
            target,
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
    except BaseException:
        try:
            session.safe_state()
        finally:
            session.close()
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
    parser.add_argument("--backend", required=True, choices=("jtag-axi", "uart"))
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18861)
    parser.add_argument("--uart-port")
    parser.add_argument("--uart-baud", type=int, default=3_000_000)
    return parser


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
    build_server_runtime(
        target,
        backend=arguments.backend,
        state_dir=arguments.state_dir,
        host=arguments.host,
        port=arguments.port,
        uart_port=arguments.uart_port,
        uart_baud=arguments.uart_baud,
        start=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "APPROVED_DEPLOYED_TARGET_ABI",
    "PulseServerRuntime",
    "bring_up_frozen_session",
    "build_arg_parser",
    "build_server_runtime",
    "build_service_for_session",
    "main",
    "open_deployed_session",
    "validate_deployed_target",
]
