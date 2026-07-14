"""Composition-private adapter from installed sequencers to the typed pulse Port."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fpga.pulse_streamer.host.image import StreamerParams, build_fingerprint
from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
from Zou_lab_control.neutral_atom.ports import PortCatalog
from zlc_neutral_atom.runtime import (
    BoundDevice,
    SafetyOperation,
    SessionClosedAck,
    SessionCloseCommand,
)
from zlc_neutral_atom.timing import (
    BoundPulsePort,
    CompletePulseCommand,
    FirePulseCommand,
    PreparePulseCommand,
    PulseFiredAck,
    PulsePreparedAck,
    PulseTerminalAck,
    PulseTerminalEvidenceKind,
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
from .legacy_runtime import LegacyDeviceRegistry, TargetDeviceEndpoint


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


@dataclass(frozen=True)
class SequencerBindingRequest:
    role: str = "sequencer"

    def __post_init__(self) -> None:
        _text(self.role, "sequencer role")


@dataclass(frozen=True)
class SequencerDescription:
    target: PulseTarget
    clock_hz: float
    geometry_fingerprint: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        if self.clock_hz <= 0:
            raise ValueError("clock_hz must be positive")


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
        self._lock = threading.RLock()
        self._binding_id: str | None = None
        self._generation: str | None = None
        self._capability_fingerprint: str | None = None
        self._session: _EndpointSession | None = None
        self._last_prepare_session_id: str | None = None
        self._operation_epoch = 0

    @property
    def target_endpoint(self) -> TargetDeviceEndpoint:
        return TargetDeviceEndpoint(
            self.execute_command,
            self.capability_probe,
            self.close_session,
            self.describe,
            {SafetyOperation.SAFE_STATE: self.interrupt},
        )

    def describe(self, binding: BoundDevice) -> SequencerDescription:
        with self._lock:
            self._validate_binding(binding)
            return SequencerDescription(
                self._target,
                float(self._sequencer.clock_hz),
                self._geometry,
            )

    def capability_probe(self, binding: BoundDevice) -> SequencerCapabilitySnapshot:
        with self._lock:
            if self._session is not None and not self._session.closed:
                raise RuntimeError("cannot probe sequencer capability during an active session")
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
            self._binding_id = binding.binding_id
            self._generation = binding.connection_generation
            self._capability_fingerprint = fingerprint
            return SequencerCapabilitySnapshot(
                binding.binding_id,
                binding.stable_device_identity,
                binding.connection_generation,
                self._target,
                float(self._sequencer.clock_hz),
                self._geometry,
                self._timeout,
                PulseTerminalEvidenceKind.SIMULATED,
                fingerprint,
            )

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
        with self._lock:
            self._validate_binding(binding)
            self._last_prepare_session_id = command.session_id
            if self._session is not None and not self._session.closed:
                raise RuntimeError("sequencer endpoint already owns an active session")
            if command.capability_fingerprint != self._capability_fingerprint:
                raise ValueError("sequencer capability fingerprint differs")
            artifact = command.request.artifact
            if artifact.target_abi_fingerprint != self._target.abi_fingerprint:
                raise ValueError("compiled pulse target differs from live sequencer")
            if artifact.target_ir.clock_hz != float(self._sequencer.clock_hz):
                raise ValueError("compiled pulse clock differs from live sequencer")
            if artifact.wire_image.geometry_fingerprint != self._geometry:
                raise ValueError("compiled wire geometry differs from live sequencer")
            self._operation_epoch += 1
            operation_epoch = self._operation_epoch
            provisional = _EndpointSession(
                command.session_id,
                command.request.artifact_digest,
                command.request,
                operation_epoch,
            )
            self._session = provisional
        try:
            playback = build_pulse_playback(
                artifact,
                name=command.request.document.name,
            )
            prepared_program = self._sequencer.prepare_compiled_playback(
                artifact,
                playback,
            )
            if prepared_program is not artifact.target_ir:
                raise RuntimeError(
                    "virtual adapter did not retain the exact compiled TargetIR"
                )
        except BaseException:
            self._sequencer.set_safe_state()
            with self._lock:
                provisional.closed = True
            raise
        with self._lock:
            self._validate_binding(binding)
            superseded = (
                operation_epoch != self._operation_epoch
                or self._session is not provisional
                or provisional.closed
            )
            if not superseded:
                provisional.prepared = True
        if superseded:
            self._sequencer.set_safe_state()
            raise RuntimeError("virtual pulse prepare was superseded by interrupt")
        with self._lock:
            return PulsePreparedAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                command.request.artifact_digest,
                self._capability_fingerprint,
            )

    def _fire(self, binding: BoundDevice, command: FirePulseCommand) -> PulseFiredAck:
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if not session.prepared or session.fired:
                raise RuntimeError("sequencer session is not ready for FIRE")
            if command.artifact_digest != session.artifact_digest:
                raise ValueError("FIRE artifact digest differs from prepared session")
            operation_epoch = session.operation_epoch
        self._sequencer.fire_compiled_playback(command.artifact_digest)
        with self._lock:
            superseded = (
                operation_epoch != self._operation_epoch
                or self._session is not session
                or session.closed
            )
            if not superseded:
                session.fired = True
        if superseded:
            self._sequencer.set_safe_state()
            raise RuntimeError("virtual pulse FIRE was superseded by interrupt")
        with self._lock:
            return PulseFiredAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                session.artifact_digest,
            )

    def _complete(
        self,
        binding: BoundDevice,
        command: CompletePulseCommand,
    ) -> PulseTerminalAck:
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if not session.fired or session.completed:
                raise RuntimeError("sequencer session is not awaiting completion")
            if command.artifact_digest != session.artifact_digest:
                raise ValueError("completion artifact digest differs")
            artifact = session.request.artifact
            operation_epoch = session.operation_epoch
        deadline = time.monotonic() + command.timeout_seconds
        if not self._sequencer.wait_compiled_playback(
            command.artifact_digest,
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
        with self._lock:
            superseded = (
                operation_epoch != self._operation_epoch
                or self._session is not session
                or session.closed
            )
            if not superseded:
                session.completed = True
        if superseded:
            self._sequencer.set_safe_state()
            raise RuntimeError("virtual pulse completion was superseded by interrupt")
        with self._lock:
            playback = build_pulse_playback(artifact)
            return PulseTerminalAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                SimulatedPulseReceipt(
                    session.artifact_digest,
                    "Zou_lab_control.VirtualSequencer",
                    counts,
                    playback.logical_duration,
                    artifact.max_configured_output_delay_ticks
                    / artifact.target_ir.clock_hz,
                ),
            )

    def close_session(
        self,
        binding: BoundDevice,
        command: SessionCloseCommand,
    ) -> SessionClosedAck:
        with self._lock:
            self._validate_binding(binding)
            session = self._session
            if session is None:
                if command.session_id != self._last_prepare_session_id:
                    raise RuntimeError("sequencer cleanup session id is unknown")
            elif session.session_id != command.session_id:
                raise RuntimeError("sequencer cleanup belongs to another session")
            self._operation_epoch += 1
            if session is not None:
                session.closed = True
        self._sequencer.set_safe_state()
        with self._lock:
            snapshot = dict(self._sequencer.snapshot())
            safe = snapshot.get("state") == "safe"
            return SessionClosedAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                safe,
                safe,
                safe,
                canonical_digest(
                    {
                        "session_id": command.session_id,
                        "state": snapshot.get("state"),
                    }
                ),
            )

    def interrupt(self) -> str:
        with self._lock:
            self._operation_epoch += 1
            if self._session is not None:
                self._session.closed = True
        self._sequencer.set_safe_state()
        return canonical_digest({"operation": "SAFE_STATE", "state": "safe"})

    def _validate_binding(self, binding: BoundDevice) -> None:
        _require_binding(binding, "sequencer", self._binding_id, self._generation)

    def _active_session(self, binding: BoundDevice, session_id: str) -> _EndpointSession:
        self._validate_binding(binding)
        session = self._session
        if session is None or session.session_id != session_id or session.closed:
            raise RuntimeError("sequencer command belongs to another or closed session")
        return session


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
        self._server_generation = snapshot.connection_generation
        self._lock = threading.RLock()
        self._binding_id: str | None = None
        self._generation: str | None = None
        self._capability_fingerprint: str | None = None
        self._session: _EndpointSession | None = None
        self._last_prepare_session_id: str | None = None
        self._operation_epoch = 0

    @property
    def target_endpoint(self) -> TargetDeviceEndpoint:
        return TargetDeviceEndpoint(
            self.execute_command,
            self.capability_probe,
            self.close_session,
            self.describe,
            {SafetyOperation.SAFE_STATE: self.interrupt},
        )

    def describe(self, binding: BoundDevice) -> SequencerDescription:
        with self._lock:
            self._validate_binding(binding)
            self._validate_remote_generation()
            return SequencerDescription(self._target, self._clock_hz, self._geometry)

    def capability_probe(self, binding: BoundDevice) -> SequencerCapabilitySnapshot:
        with self._lock:
            if self._session is not None and not self._session.closed:
                raise RuntimeError("cannot probe sequencer capability during an active session")
            snapshot = self._validate_remote_generation()
            if snapshot.state not in {"IDLE", "SAFE", "DONE"}:
                raise RuntimeError(
                    f"remote pulse server is not ready for capability probe: {snapshot.state}"
                )
            fingerprint = canonical_digest(
                {
                    "contract": "zlc.remote-pulse-execution",
                    "endpoint_label": self._endpoint_label,
                    "server_connection_generation": self._server_generation,
                    "target_abi_fingerprint": self._target.abi_fingerprint,
                    "clock_hz": self._clock_hz,
                    "geometry_fingerprint": self._geometry,
                    "max_blocking_call_seconds": self._timeout,
                    "terminal_evidence_kind": (
                        PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS.value
                    ),
                }
            )
            self._binding_id = binding.binding_id
            self._generation = binding.connection_generation
            self._capability_fingerprint = fingerprint
            return SequencerCapabilitySnapshot(
                binding.binding_id,
                binding.stable_device_identity,
                binding.connection_generation,
                self._target,
                self._clock_hz,
                self._geometry,
                self._timeout,
                PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS,
                fingerprint,
            )

    def execute_command(self, binding: BoundDevice, command: object) -> object:
        if isinstance(command, PreparePulseCommand):
            return self._prepare(binding, command)
        if isinstance(command, FirePulseCommand):
            return self._fire(binding, command)
        if isinstance(command, CompletePulseCommand):
            return self._complete(binding, command)
        raise TypeError(f"sequencer endpoint rejects command {type(command).__name__}")

    def _prepare(self, binding: BoundDevice, command: PreparePulseCommand) -> PulsePreparedAck:
        with self._lock:
            self._validate_binding(binding)
            self._validate_remote_generation()
            self._last_prepare_session_id = command.session_id
            if self._session is not None and not self._session.closed:
                raise RuntimeError("sequencer endpoint already owns an active session")
            if command.capability_fingerprint != self._capability_fingerprint:
                raise ValueError("sequencer capability fingerprint differs")
            if command.timeout_seconds > self._timeout:
                raise ValueError("prepare timeout exceeds sequencer capability")
            artifact = command.request.artifact
            if artifact.target_abi_fingerprint != self._target.abi_fingerprint:
                raise ValueError("compiled pulse target differs from remote sequencer")
            if artifact.target_ir.clock_hz != self._clock_hz:
                raise ValueError("compiled pulse clock differs from remote sequencer")
            if artifact.wire_image.geometry_fingerprint != self._geometry:
                raise ValueError("compiled wire geometry differs from remote sequencer")
            self._operation_epoch += 1
            operation_epoch = self._operation_epoch
            provisional = _EndpointSession(
                command.session_id,
                command.request.artifact_digest,
                command.request,
                operation_epoch,
            )
            self._session = provisional
        try:
            reference = self._client.prepare(artifact)
        except BaseException:
            self._client.safe_state()
            with self._lock:
                provisional.closed = True
            raise
        with self._lock:
            self._validate_binding(binding)
            superseded = (
                operation_epoch != self._operation_epoch
                or self._session is not provisional
                or provisional.closed
            )
            if not superseded:
                provisional.prepared = True
                provisional.prepared_ref = reference
        if superseded:
            self._client.safe_state()
            raise RuntimeError("remote pulse prepare was superseded by interrupt")
        with self._lock:
            return PulsePreparedAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                command.request.artifact_digest,
                self._capability_fingerprint,
            )

    def _fire(self, binding: BoundDevice, command: FirePulseCommand) -> PulseFiredAck:
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if not session.prepared or session.fired or session.prepared_ref is None:
                raise RuntimeError("sequencer session is not ready for FIRE")
            if command.artifact_digest != session.artifact_digest:
                raise ValueError("FIRE artifact digest differs from prepared session")
            reference = session.prepared_ref
            operation_epoch = session.operation_epoch
        self._client.fire(reference)
        with self._lock:
            superseded = (
                operation_epoch != self._operation_epoch
                or self._session is not session
                or session.closed
            )
            if not superseded:
                session.fired = True
        if superseded:
            self._client.safe_state()
            raise RuntimeError("remote pulse FIRE was superseded by interrupt")
        with self._lock:
            return PulseFiredAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                session.artifact_digest,
            )

    def _complete(self, binding: BoundDevice, command: CompletePulseCommand) -> PulseTerminalAck:
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if not session.fired or session.completed or session.prepared_ref is None:
                raise RuntimeError("sequencer session is not awaiting completion")
            if command.artifact_digest != session.artifact_digest:
                raise ValueError("completion artifact digest differs")
            if command.timeout_seconds > self._timeout:
                raise ValueError("completion timeout exceeds sequencer capability")
            reference = session.prepared_ref
            operation_epoch = session.operation_epoch
        completion = self._client.complete(reference, timeout=command.timeout_seconds)
        with self._lock:
            superseded = (
                operation_epoch != self._operation_epoch
                or self._session is not session
                or session.closed
            )
            if not superseded:
                session.completed = True
        if superseded:
            self._client.safe_state()
            raise RuntimeError("remote pulse completion was superseded by interrupt")
        with self._lock:
            return PulseTerminalAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                completion,
            )

    def close_session(self, binding: BoundDevice, command: SessionCloseCommand) -> SessionClosedAck:
        with self._lock:
            self._validate_binding(binding)
            session = self._session
            if session is None:
                if command.session_id != self._last_prepare_session_id:
                    raise RuntimeError("sequencer cleanup session id is unknown")
            elif session.session_id != command.session_id:
                raise RuntimeError("sequencer cleanup belongs to another session")
            self._operation_epoch += 1
            if session is not None:
                session.closed = True
        snapshot = self._client.safe_state()
        with self._lock:
            digest = canonical_digest(
                {
                    "session_id": command.session_id,
                    "server_generation": self._server_generation,
                    "state": snapshot.state,
                    "backend": snapshot.backend,
                }
            )
            return SessionClosedAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                True,
                True,
                True,
                digest,
            )

    def interrupt(self) -> str:
        with self._lock:
            self._operation_epoch += 1
            if self._session is not None:
                self._session.closed = True
        snapshot = self._client.safe_state()
        return canonical_digest(
            {
                "operation": "SAFE_STATE",
                "server_generation": self._server_generation,
                "state": snapshot.state,
            }
        )

    def _validate_binding(self, binding: BoundDevice) -> None:
        _require_binding(binding, "sequencer", self._binding_id, self._generation)

    def _validate_remote_generation(self):
        snapshot = self._client.snapshot()
        if snapshot.connection_generation != self._server_generation:
            raise RuntimeError("remote pulse server connection generation changed")
        if (
            snapshot.target != self._target
            or snapshot.clock_hz != self._clock_hz
            or snapshot.geometry_fingerprint != self._geometry
        ):
            raise RuntimeError("remote pulse server capability changed within one connection")
        return snapshot

    def _active_session(self, binding: BoundDevice, session_id: str) -> _EndpointSession:
        self._validate_binding(binding)
        session = self._session
        if session is None or session.session_id != session_id or session.closed:
            raise RuntimeError("sequencer command belongs to another or closed session")
        return session


def bind_sequencer_port(
    device_set: object,
    registry: LegacyDeviceRegistry,
    request: SequencerBindingRequest = SequencerBindingRequest(),
) -> BoundPulsePort:
    if not isinstance(registry, LegacyDeviceRegistry):
        raise TypeError("registry must be LegacyDeviceRegistry")
    if not isinstance(request, SequencerBindingRequest):
        raise TypeError("request must be SequencerBindingRequest")
    devices = dict(getattr(device_set, "devices", {}) or {})
    try:
        sequencer = devices[request.role]
    except KeyError as exc:
        raise KeyError(f"sequencer role {request.role!r} is absent from installation") from exc
    registration = registry.registration_for(sequencer)
    target = registration.target_endpoint
    if target is None:
        raise RuntimeError(f"sequencer role {request.role!r} has no typed execution endpoint")
    binding = registry.binding_for(sequencer)
    attestation = registry.broker.verify_capability(binding)
    description = target.describe(binding)
    if not isinstance(description, SequencerDescription):
        raise TypeError("sequencer endpoint returned the wrong description type")
    capability = attestation.snapshot
    if not isinstance(capability, SequencerCapabilitySnapshot):
        raise TypeError("sequencer capability attestation has the wrong snapshot type")
    if (
        capability.target != description.target
        or capability.clock_hz != description.clock_hz
        or capability.geometry_fingerprint != description.geometry_fingerprint
    ):
        raise RuntimeError("sequencer description and capability differ")
    # Session close owns the one idempotent SAFE transition.  The registration's
    # SAFE_STATE callback remains available for legacy/recovery and out-of-band
    # interrupt paths, but normal cleanup must not issue the same drive twice.
    return BoundPulsePort(attestation, ())


__all__ = [
    "bind_sequencer_port",
    "SequencerBindingRequest",
    "SequencerDescription",
    "RemotePulseExecutionEndpoint",
    "VirtualSequencerExecutionEndpoint",
]
