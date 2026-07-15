"""Composition-private adapter from installed sequencers to the typed pulse Port."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fpga.pulse_streamer.host.image import StreamerParams, build_fingerprint
from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
from Zou_lab_control.neutral_atom.ports import PortCatalog
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SessionClosedAck,
    SessionCloseCommand,
)
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.timing.pulse import (
    CompletePulseCommand,
    FirePulseCommand,
    PreparePulseCommand,
    PulseFiredAck,
    PulsePreparedAck,
    PulseTerminalAck,
    PulseTerminalEvidenceKind,
    PulseTerminalReceipt,
    SequencerCapabilitySnapshot,
    SimulatedPulseReceipt,
)
from zlc_pulse import (
    PreparedPulseRef,
    PulsePortSpec,
    PulseTarget,
    RemotePulseExecutionClient,
    build_pulse_playback,
)
from zlc_storage import canonical_digest, canonical_text as _text

from ._endpoint_binding import require_current_endpoint_binding as _require_binding


def _pulse_target_from_port_catalog(catalog: PortCatalog) -> PulseTarget:
    """Project the installed sequencer's typed topology at the composition boundary."""

    if not isinstance(catalog, PortCatalog):
        raise TypeError("virtual sequencer port_catalog must be PortCatalog")
    return PulseTarget(
        tuple(catalog.raw_lanes),
        tuple(
            PulsePortSpec(
                key=port.key,
                kind=port.kind,
                lanes=tuple(port.lanes),
                label=port.label,
                bus_index=port.bus_index,
                width=port.width,
                encoding=port.encoding,
                safe_value=port.safe_value,
                latch_clock=port.latch_clock,
            )
            for port in catalog.ports
        ),
    )


@dataclass
class _EndpointSession:
    session_id: str
    artifact_digest: str
    request: object
    operation_epoch: int
    prepared: bool = False
    fired: bool = False
    completed: bool = False
    closed: bool = False
    prepared_ref: PreparedPulseRef | None = None
    physical_operation_in_flight: bool = False
    close_acknowledged: bool = False


class _SequencerSessionOwner:
    """The single finite sequencer transition and acknowledgement owner."""

    def __init__(
        self,
        backend: VirtualSequencerExecutionEndpoint | RemotePulseExecutionEndpoint,
    ) -> None:
        self._backend = backend
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._safe_state_lock = threading.Lock()
        self._binding_instance_id: str | None = None
        self._capability_fingerprint: str | None = None
        self._session: _EndpointSession | None = None
        self._last_prepare_session_id: str | None = None
        self._operation_epoch = 0

    def capability_probe(self, binding: BoundDevice) -> SequencerCapabilitySnapshot:
        with self._condition:
            if self._session is not None and (
                self._session.physical_operation_in_flight
                or not self._session.closed
                or not self._session.close_acknowledged
            ):
                raise RuntimeError("cannot probe sequencer capability during an active session")
            snapshot = self._backend._backend_capability_snapshot(binding)
            if not isinstance(snapshot, SequencerCapabilitySnapshot):
                raise TypeError("sequencer backend returned another capability type")
            if snapshot.binding_stamp != binding.binding_stamp:
                raise ValueError("sequencer capability identity differs from binding")
            self._binding_instance_id = binding.binding_instance_id
            self._capability_fingerprint = snapshot.capability_fingerprint
            return snapshot

    def execute_command(self, binding: BoundDevice, command: object) -> object:
        if isinstance(command, PreparePulseCommand):
            return self._prepare(binding, command)
        if isinstance(command, FirePulseCommand):
            return self._fire(binding, command)
        if isinstance(command, CompletePulseCommand):
            return self._complete(binding, command)
        raise TypeError(f"sequencer endpoint rejects command {type(command).__name__}")

    def _prepare(
        self,
        binding: BoundDevice,
        command: PreparePulseCommand,
    ) -> PulsePreparedAck:
        with self._condition:
            self._validate_binding(binding)
            current = self._session
            if current is not None:
                if current.physical_operation_in_flight:
                    raise RuntimeError(
                        "sequencer endpoint still has a physical operation in flight"
                    )
                if not current.closed or not current.close_acknowledged:
                    raise RuntimeError(
                        "sequencer endpoint previous session is not terminally closed"
                    )
            self._last_prepare_session_id = command.session_id
            if command.capability_fingerprint != self._capability_fingerprint:
                raise ValueError("sequencer capability fingerprint differs")
            self._backend._backend_validate_prepare(command)
            self._operation_epoch += 1
            operation_epoch = self._operation_epoch
            provisional = _EndpointSession(
                command.session_id,
                command.request.artifact_digest,
                command.request,
                operation_epoch,
                physical_operation_in_flight=True,
            )
            self._session = provisional
        try:
            prepared_ref = self._backend._backend_prepare(provisional)
        except BaseException as error:
            self._seal_after_physical_failure(provisional, error)
            self._finish_physical_operation(provisional)
            raise
        try:
            with self._lock:
                self._validate_binding(binding)
                if self._superseded(provisional, operation_epoch):
                    raise RuntimeError("sequencer prepare was superseded by interrupt")
                provisional.prepared = True
                provisional.prepared_ref = prepared_ref
                fingerprint = self._capability_fingerprint
                assert fingerprint is not None
            acknowledgement = PulsePreparedAck(
                command.session_id,
                binding.binding_instance_id,
                command.request.artifact_digest,
                fingerprint,
            )
        except BaseException as error:
            self._seal_after_physical_failure(provisional, error)
            self._finish_physical_operation(provisional)
            raise
        self._finish_physical_operation(provisional)
        return acknowledgement

    def _fire(self, binding: BoundDevice, command: FirePulseCommand) -> PulseFiredAck:
        with self._condition:
            session = self._active_session(binding, command.session_id)
            if session.physical_operation_in_flight:
                raise RuntimeError("sequencer session already has an operation in flight")
            if not session.prepared or session.fired:
                raise RuntimeError("sequencer session is not ready for FIRE")
            if command.artifact_digest != session.artifact_digest:
                raise ValueError("FIRE artifact digest differs from prepared session")
            operation_epoch = session.operation_epoch
            session.physical_operation_in_flight = True
        try:
            self._backend._backend_fire(session)
        except BaseException as error:
            self._seal_after_physical_failure(session, error)
            self._finish_physical_operation(session)
            raise
        try:
            with self._lock:
                self._validate_binding(binding)
                if self._superseded(session, operation_epoch):
                    raise RuntimeError("sequencer FIRE was superseded by interrupt")
                session.fired = True
            acknowledgement = PulseFiredAck(
                command.session_id,
                binding.binding_instance_id,
                session.artifact_digest,
            )
        except BaseException as error:
            self._seal_after_physical_failure(session, error)
            self._finish_physical_operation(session)
            raise
        self._finish_physical_operation(session)
        return acknowledgement

    def _complete(
        self,
        binding: BoundDevice,
        command: CompletePulseCommand,
    ) -> PulseTerminalAck:
        with self._condition:
            session = self._active_session(binding, command.session_id)
            if session.physical_operation_in_flight:
                raise RuntimeError("sequencer session already has an operation in flight")
            if not session.fired or session.completed:
                raise RuntimeError("sequencer session is not awaiting completion")
            if command.artifact_digest != session.artifact_digest:
                raise ValueError("completion artifact digest differs")
            operation_epoch = session.operation_epoch
            session.physical_operation_in_flight = True
        try:
            receipt = self._backend._backend_complete(session, command.timeout_seconds)
        except BaseException as error:
            self._seal_after_physical_failure(session, error)
            self._finish_physical_operation(session)
            raise
        try:
            with self._lock:
                self._validate_binding(binding)
                if self._superseded(session, operation_epoch):
                    raise RuntimeError(
                        "sequencer completion was superseded by interrupt"
                    )
                session.completed = True
            acknowledgement = PulseTerminalAck(
                command.session_id,
                binding.binding_instance_id,
                receipt,
            )
        except BaseException as error:
            self._seal_after_physical_failure(session, error)
            self._finish_physical_operation(session)
            raise
        self._finish_physical_operation(session)
        return acknowledgement

    def close_session(
        self,
        binding: BoundDevice,
        command: SessionCloseCommand,
    ) -> SessionClosedAck:
        deadline = time.monotonic() + command.timeout_seconds
        with self._condition:
            self._validate_binding(binding)
            session = self._session
            if session is None:
                if command.session_id != self._last_prepare_session_id:
                    raise RuntimeError("sequencer cleanup session id is unknown")
            elif session.session_id != command.session_id:
                if not (
                    session.close_acknowledged
                    and command.session_id == self._last_prepare_session_id
                ):
                    raise RuntimeError("sequencer cleanup belongs to another session")
                session = None
            self._operation_epoch += 1
            if session is not None:
                session.closed = True
            was_in_flight = bool(
                session is not None and session.physical_operation_in_flight
            )
        snapshot = self._set_safe_state()
        if time.monotonic() >= deadline:
            raise TimeoutError("sequencer close exceeded its bounded deadline")
        if session is not None and not self._wait_until_joined(session, deadline):
            raise TimeoutError("sequencer physical operation did not join before close")
        if was_in_flight:
            snapshot = self._set_safe_state()
            if time.monotonic() >= deadline:
                raise TimeoutError("sequencer close exceeded its bounded deadline")
        safe, digest = self._backend._backend_close_evidence(
            command.session_id,
            snapshot,
        )
        acknowledgement = SessionClosedAck(
            command.session_id,
            binding.binding_instance_id,
            safe,
            safe,
            safe,
            digest,
        )
        with self._condition:
            if session is not None:
                if session.physical_operation_in_flight:
                    raise RuntimeError("sequencer close lost its joined state")
                session.close_acknowledged = True
        return acknowledgement

    def interrupt(self) -> str:
        with self._condition:
            self._operation_epoch += 1
            if self._session is not None:
                self._session.closed = True
        snapshot = self._set_safe_state()
        return self._backend._backend_interrupt_digest(snapshot)

    def _validate_binding(self, binding: BoundDevice) -> None:
        _require_binding(binding, "sequencer", self._binding_instance_id)

    def _active_session(self, binding: BoundDevice, session_id: str) -> _EndpointSession:
        self._validate_binding(binding)
        session = self._session
        if session is None or session.session_id != session_id or session.closed:
            raise RuntimeError("sequencer command belongs to another or closed session")
        return session

    def _superseded(self, session: _EndpointSession, operation_epoch: int) -> bool:
        return (
            operation_epoch != self._operation_epoch
            or self._session is not session
            or session.closed
        )

    def _seal_after_physical_failure(
        self,
        session: _EndpointSession,
        primary: BaseException,
    ) -> None:
        with self._condition:
            session.closed = True
            self._operation_epoch += 1
        try:
            self._set_safe_state()
        except BaseException as secondary:
            record_secondary_failure(
                primary,
                "sequencer fail-safe transition also failed",
                secondary,
            )

    def _finish_physical_operation(self, session: _EndpointSession) -> None:
        with self._condition:
            session.physical_operation_in_flight = False
            self._condition.notify_all()

    def _wait_until_joined(
        self,
        session: _EndpointSession,
        deadline: float,
    ) -> bool:
        with self._condition:
            while session.physical_operation_in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _set_safe_state(self) -> object:
        with self._safe_state_lock:
            return self._backend._backend_set_safe_state()


class VirtualSequencerExecutionEndpoint:
    """Typed finite execution endpoint for the in-process hardware simulator."""

    def __init__(
        self,
        sequencer: VirtualSequencer,
        *,
        max_blocking_call_seconds: float = 10.0,
        params: StreamerParams | None = None,
    ) -> None:
        if type(sequencer) is not VirtualSequencer:
            raise TypeError("this endpoint is specific to VirtualSequencer")
        if max_blocking_call_seconds <= 0:
            raise ValueError("max_blocking_call_seconds must be positive")
        self._sequencer = sequencer
        self._timeout = float(max_blocking_call_seconds)
        self._params = params or StreamerParams()
        self._target = _pulse_target_from_port_catalog(sequencer.port_catalog)
        self._geometry = build_fingerprint(self._params) & 0xFFFFFFFF
        self._owner = _SequencerSessionOwner(self)

    def _backend_capability_snapshot(
        self, binding: BoundDevice
    ) -> SequencerCapabilitySnapshot:
        fingerprint = canonical_digest(
            {
                "contract": "zlc.virtual-sequencer-execution",
                "target_abi_fingerprint": self._target.abi_fingerprint,
                "clock_hz": float(self._sequencer.clock_hz),
                "geometry_fingerprint": self._geometry,
                "max_blocking_call_seconds": self._timeout,
                "terminal_evidence_kind": PulseTerminalEvidenceKind.SIMULATED.value,
            }
        )
        return SequencerCapabilitySnapshot(
            binding_stamp=binding.binding_stamp,
            target=self._target,
            clock_hz=float(self._sequencer.clock_hz),
            geometry_fingerprint=self._geometry,
            max_blocking_call_seconds=self._timeout,
            terminal_evidence_kind=PulseTerminalEvidenceKind.SIMULATED,
            server_connection_generation=None,
            capability_fingerprint=fingerprint,
        )

    def _backend_validate_prepare(self, command: PreparePulseCommand) -> None:
        artifact = command.request.artifact
        if artifact.target_abi_fingerprint != self._target.abi_fingerprint:
            raise ValueError("compiled pulse target differs from live sequencer")
        if artifact.target_ir.clock_hz != float(self._sequencer.clock_hz):
            raise ValueError("compiled pulse clock differs from live sequencer")
        if artifact.wire_image.geometry_fingerprint != self._geometry:
            raise ValueError("compiled wire geometry differs from live sequencer")

    def _backend_prepare(self, session: _EndpointSession) -> None:
        artifact = session.request.artifact
        playback = build_pulse_playback(
            artifact,
            name=session.request.document.name,
        )
        prepared_program = self._sequencer.prepare_compiled_playback(
            artifact,
            playback,
        )
        if prepared_program is not artifact.target_ir:
            raise RuntimeError(
                "virtual adapter did not retain the exact compiled TargetIR"
            )

    def _backend_fire(self, session: _EndpointSession) -> None:
        self._sequencer.fire_compiled_playback(session.artifact_digest)

    def _backend_complete(
        self,
        session: _EndpointSession,
        timeout_seconds: float,
    ) -> PulseTerminalReceipt:
        artifact = session.request.artifact
        deadline = time.monotonic() + timeout_seconds
        if not self._sequencer.wait_compiled_playback(
            session.artifact_digest,
            max(0.0, deadline - time.monotonic()),
        ):
            raise TimeoutError("sequencer did not reach logical terminal")
        delay_wait = (
            artifact.max_configured_output_delay_ticks
            / float(artifact.target_ir.clock_hz)
            * float(self._sequencer.sleep_scale)
        )
        if delay_wait > 0:
            if delay_wait > max(0.0, deadline - time.monotonic()):
                raise TimeoutError(
                    "sequencer output-delay tail exceeds completion deadline"
                )
            time.sleep(delay_wait)
        counts = tuple(
            (schedule.channel, schedule.total)
            for schedule in artifact.trigger_schedules
        )
        playback = build_pulse_playback(artifact)
        return SimulatedPulseReceipt(
            session.artifact_digest,
            "Zou_lab_control.VirtualSequencer",
            counts,
            playback.logical_duration,
            artifact.max_configured_output_delay_ticks / artifact.target_ir.clock_hz,
        )

    def _backend_set_safe_state(self) -> object:
        self._sequencer.set_safe_state()
        return dict(self._sequencer.snapshot())

    def _backend_close_evidence(
        self,
        session_id: str,
        snapshot: object,
    ) -> tuple[bool, str]:
        if not isinstance(snapshot, dict):
            raise TypeError("virtual sequencer safe-state snapshot must be a mapping")
        safe = snapshot.get("state") == "safe"
        return safe, canonical_digest(
            {"session_id": session_id, "state": snapshot.get("state")}
        )

    def _backend_interrupt_digest(self, snapshot: object) -> str:
        if not isinstance(snapshot, dict):
            raise TypeError("virtual sequencer safe-state snapshot must be a mapping")
        return canonical_digest(
            {"operation": "SAFE_STATE", "state": snapshot.get("state")}
        )


class RemotePulseExecutionEndpoint:
    """Typed target endpoint over one current, non-reconnecting pulse RPC owner."""

    def __init__(
        self,
        client: RemotePulseExecutionClient,
        *,
        endpoint_label: str,
        max_blocking_call_seconds: float | None = None,
    ) -> None:
        if not isinstance(client, RemotePulseExecutionClient):
            raise TypeError("remote endpoint requires RemotePulseExecutionClient")
        _text(endpoint_label, "endpoint_label")
        limit = (
            client.transport_timeout_seconds * 0.9
            if max_blocking_call_seconds is None
            else float(max_blocking_call_seconds)
        )
        if limit <= 0 or limit >= client.transport_timeout_seconds:
            raise ValueError("max blocking call must be shorter than the client transport backstop")
        snapshot = client.snapshot()
        self._client = client
        self._endpoint_label = endpoint_label
        self._timeout = limit
        self._target = snapshot.target
        self._clock_hz = snapshot.clock_hz
        self._geometry = snapshot.geometry_fingerprint
        self._server_connection_generation = snapshot.connection_generation
        self._owner = _SequencerSessionOwner(self)

    def _backend_capability_snapshot(
        self, binding: BoundDevice
    ) -> SequencerCapabilitySnapshot:
        snapshot = self._validate_server_connection_generation()
        if snapshot.state not in {"IDLE", "SAFE", "DONE"}:
            raise RuntimeError(
                f"remote pulse server is not ready for capability probe: {snapshot.state}"
            )
        fingerprint = canonical_digest(
            {
                "contract": "zlc.remote-pulse-execution",
                "endpoint_label": self._endpoint_label,
                "server_connection_generation": self._server_connection_generation,
                "target_abi_fingerprint": self._target.abi_fingerprint,
                "clock_hz": self._clock_hz,
                "geometry_fingerprint": self._geometry,
                "max_blocking_call_seconds": self._timeout,
                "terminal_evidence_kind": (
                    PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS.value
                ),
            }
        )
        return SequencerCapabilitySnapshot(
            binding_stamp=binding.binding_stamp,
            target=self._target,
            clock_hz=self._clock_hz,
            geometry_fingerprint=self._geometry,
            max_blocking_call_seconds=self._timeout,
            terminal_evidence_kind=PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS,
            server_connection_generation=self._server_connection_generation,
            capability_fingerprint=fingerprint,
        )

    def _backend_validate_prepare(self, command: PreparePulseCommand) -> None:
        self._validate_server_connection_generation()
        if command.timeout_seconds > self._timeout:
            raise ValueError("prepare timeout exceeds sequencer capability")
        artifact = command.request.artifact
        if artifact.target_abi_fingerprint != self._target.abi_fingerprint:
            raise ValueError("compiled pulse target differs from remote sequencer")
        if artifact.target_ir.clock_hz != self._clock_hz:
            raise ValueError("compiled pulse clock differs from remote sequencer")
        if artifact.wire_image.geometry_fingerprint != self._geometry:
            raise ValueError("compiled wire geometry differs from remote sequencer")

    def _backend_prepare(
        self,
        session: _EndpointSession,
    ) -> PreparedPulseRef:
        return self._client.prepare(session.request.artifact)

    def _backend_fire(self, session: _EndpointSession) -> None:
        reference = session.prepared_ref
        if reference is None:
            raise RuntimeError("remote sequencer session has no prepared reference")
        self._client.fire(reference)

    def _backend_complete(
        self,
        session: _EndpointSession,
        timeout_seconds: float,
    ) -> PulseTerminalReceipt:
        if timeout_seconds > self._timeout:
            raise ValueError("completion timeout exceeds sequencer capability")
        reference = session.prepared_ref
        if reference is None:
            raise RuntimeError("remote sequencer session has no prepared reference")
        return self._client.complete(reference, timeout=timeout_seconds)

    def _backend_set_safe_state(self) -> object:
        return self._client.safe_state()

    def _backend_close_evidence(
        self,
        session_id: str,
        snapshot: object,
    ) -> tuple[bool, str]:
        return True, canonical_digest(
            {
                "session_id": session_id,
                "server_connection_generation": self._server_connection_generation,
                "state": snapshot.state,
                "backend": snapshot.backend,
            }
        )

    def _backend_interrupt_digest(self, snapshot: object) -> str:
        return canonical_digest(
            {
                "operation": "SAFE_STATE",
                "server_connection_generation": self._server_connection_generation,
                "state": snapshot.state,
            }
        )

    def _validate_server_connection_generation(self):
        snapshot = self._client.snapshot()
        if snapshot.connection_generation != self._server_connection_generation:
            raise RuntimeError("remote pulse server connection generation changed")
        if (
            snapshot.target != self._target
            or snapshot.clock_hz != self._clock_hz
            or snapshot.geometry_fingerprint != self._geometry
        ):
            raise RuntimeError("remote pulse server capability changed within one connection")
        return snapshot

__all__ = [
    "RemotePulseExecutionEndpoint",
    "VirtualSequencerExecutionEndpoint",
]
