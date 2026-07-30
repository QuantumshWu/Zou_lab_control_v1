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
    FROZEN_CLOCK_HZ,
    StreamerParams,
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
    UartRegisterTransport,
    VivadoAxiRegisterTransport,
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
            raise ValueError("--uart-port is required for the uart backend")
        transport = UartRegisterTransport(
            state_dir=state_dir,
            port=uart_port,
            baud=uart_baud,
        )
    else:
        raise ValueError("backend must be 'jtag-axi' or 'uart'")
    return _CurrentDeployedStreamerSession(
        transport,
        device_lease=InterprocessDeviceLease(),
        deployed_target=target,
        params=params,
        clock_hz=clock_hz,
    )


def build_server_runtime(
    manifest: PulseTargetManifest,
    *,
    backend: str,
    state_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 18861,
    uart_port: str | None = None,
    uart_baud: int = 3_000_000,
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
    manifest = pulse_target_manifest_from_xdc(target, arguments.xdc)
    params, clock_hz, config_path = _load_deployed_streamer_config()
    logging.info("deployment geometry loaded from %s", config_path)
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
