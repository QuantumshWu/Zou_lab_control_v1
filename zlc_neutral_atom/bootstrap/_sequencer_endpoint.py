"""Composition-owned sequencer endpoint for the typed pulse Port."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fpga.pulse_streamer.host.image import StreamerParams, build_fingerprint
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
    PulseScanProgress,
    PulseTerminalAck,
    PulseTerminalEvidenceKind,
    PulseTerminalReceipt,
    SequencerCapabilitySnapshot,
    SimulatedPulseReceipt,
)
from zlc_pulse import (
    PulseExecutionForm,
    PulseTargetManifest,
    PreparedPulseRef,
    RemotePulseExecutionClient,
    build_pulse_playback,
    resident_scan_point_capacity,
    validate_resident_scan_capacity,
)
from zlc_storage import (
    canonical_digest,
    canonical_text as _text,
    positive_real as _positive_real,
)

from ._endpoint_binding import require_current_endpoint_binding as _require_binding
from ._virtual_hardware import VirtualSequencer


@dataclass
class _EndpointSession:
    session_id: str
    run_id: str
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
        *,
        max_blocking_call_seconds: float,
    ) -> None:
        self._backend = backend
        self._max_blocking_call_seconds = _positive_real(
            max_blocking_call_seconds,
            "max_blocking_call_seconds",
        )
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

    def observe_scan_progress(
        self,
        binding: BoundDevice,
        session_id: str,
        run_id: str,
        artifact_digest: str,
        point_count: int,
    ) -> PulseScanProgress:
        """Read one exact continuous-scan cursor without owning its timing."""

        def unavailable(state: str, reason: str) -> PulseScanProgress:
            return PulseScanProgress.unavailable(
                run_id=run_id,
                artifact_digest=artifact_digest,
                point_count=point_count,
                backend_state=state,
                reason=reason,
            )

        with self._condition:
            self._validate_binding(binding)
            session = self._session
            if (
                session is None
                or session.session_id != session_id
                or session.run_id != run_id
                or session.artifact_digest != artifact_digest
            ):
                return unavailable("SUPERSEDED", "pulse session is no longer current")
            if len(session.request.artifact.target_ir.scan_points) != point_count:
                return unavailable("MISMATCH", "scan table cardinality differs")
            if (
                session.request.artifact.execution_form
                is not PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS
            ):
                return unavailable(
                    "UNAVAILABLE",
                    "progress observation is defined only for continuous scan tables",
                )
            if session.closed or session.completed:
                return unavailable("TERMINAL", "pulse session is no longer active")
            if not session.fired:
                return unavailable("PREPARED", "pulse scan has not fired")
            operation_epoch = session.operation_epoch
        try:
            progress = self._backend._backend_observe_scan_progress(session)
        except Exception as error:
            return unavailable(
                "UNAVAILABLE",
                f"scan-progress observation failed: {type(error).__name__}",
            )
        if not isinstance(progress, PulseScanProgress):
            raise TypeError("sequencer backend returned another progress type")
        with self._condition:
            if self._superseded(session, operation_epoch):
                return unavailable("SUPERSEDED", "pulse session changed during observation")
        if (
            progress.run_id != run_id
            or progress.artifact_digest != artifact_digest
            or progress.point_count != point_count
        ):
            raise RuntimeError("sequencer backend observed another pulse execution")
        return progress

    def wait_continuous_failure(
        self,
        binding: BoundDevice,
        session_id: str,
        run_id: str,
        artifact_digest: str,
        timeout: float,
    ) -> str | None:
        """Wait on backend failure evidence without issuing a timing command."""

        with self._condition:
            self._validate_binding(binding)
            session = self._session
            if (
                session is None
                or session.session_id != session_id
                or session.run_id != run_id
                or session.artifact_digest != artifact_digest
                or session.closed
            ):
                return None
            if not session.request.artifact.target_ir.repeat_forever:
                raise RuntimeError("failure wait requires a continuous pulse session")
            if not session.fired:
                raise RuntimeError("continuous pulse session has not fired")
            operation_epoch = session.operation_epoch
        try:
            failure = self._backend._backend_wait_continuous_failure(
                session,
                timeout,
            )
        except Exception as error:
            failure = (
                "continuous failure notification failed: "
                f"{type(error).__name__}: {error}"
            )
        with self._condition:
            if self._superseded(session, operation_epoch):
                return None
        if failure is not None:
            _text(failure, "continuous pulse failure")
        return failure

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
                previously_cleaned = (
                    current.closed and current.close_acknowledged
                )
                same_run_completed_segment = (
                    current.completed
                    and not current.closed
                    and current.run_id == command.run_id
                )
                if not (previously_cleaned or same_run_completed_segment):
                    if current.run_id != command.run_id and not previously_cleaned:
                        raise RuntimeError(
                            "sequencer endpoint cannot cross runs before cleanup"
                        )
                    raise RuntimeError(
                        "sequencer endpoint previous segment is not completed"
                    )
            self._last_prepare_session_id = command.session_id
            if command.capability_fingerprint != self._capability_fingerprint:
                raise ValueError("sequencer capability fingerprint differs")
            self._backend._backend_validate_prepare(command)
            self._operation_epoch += 1
            operation_epoch = self._operation_epoch
            provisional = _EndpointSession(
                command.session_id,
                command.run_id,
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
                provisional.prepared_ref = prepared_ref
                provisional.prepared = True
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
        snapshot = self._set_safe_state(deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError("sequencer close exceeded its bounded deadline")
        if session is not None and not self._wait_until_joined(session, deadline):
            raise TimeoutError("sequencer physical operation did not join before close")
        if was_in_flight:
            snapshot = self._set_safe_state(deadline)
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

    def _set_safe_state(self, deadline: float | None = None) -> object:
        if deadline is None:
            deadline = time.monotonic() + self._max_blocking_call_seconds
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._safe_state_lock.acquire(timeout=remaining):
            raise TimeoutError("sequencer safe-state deadline elapsed")
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("sequencer safe-state deadline elapsed")
            return self._backend._backend_set_safe_state(remaining)
        finally:
            self._safe_state_lock.release()


class _OwnedSequencerEndpoint:
    """Public composition callbacks backed by one private session owner."""

    _owner: _SequencerSessionOwner

    def capability_probe(self, binding: BoundDevice) -> SequencerCapabilitySnapshot:
        return self._owner.capability_probe(binding)

    def execute_command(self, binding: BoundDevice, command: object) -> object:
        return self._owner.execute_command(binding, command)

    def observe_scan_progress(
        self,
        binding: BoundDevice,
        session_id: str,
        run_id: str,
        artifact_digest: str,
        point_count: int,
    ) -> PulseScanProgress:
        return self._owner.observe_scan_progress(
            binding,
            session_id,
            run_id,
            artifact_digest,
            point_count,
        )

    def wait_continuous_failure(
        self,
        binding: BoundDevice,
        session_id: str,
        run_id: str,
        artifact_digest: str,
        timeout: float,
    ) -> str | None:
        return self._owner.wait_continuous_failure(
            binding,
            session_id,
            run_id,
            artifact_digest,
            timeout,
        )

    def close_session(
        self,
        binding: BoundDevice,
        command: SessionCloseCommand,
    ) -> SessionClosedAck:
        return self._owner.close_session(binding, command)

    def interrupt(self) -> str:
        return self._owner.interrupt()


class VirtualSequencerExecutionEndpoint(_OwnedSequencerEndpoint):
    """Typed finite execution endpoint for the in-process hardware simulator."""

    def __init__(
        self,
        sequencer: VirtualSequencer,
        manifest: PulseTargetManifest,
        *,
        max_blocking_call_seconds: float = 10.0,
        params: StreamerParams | None = None,
    ) -> None:
        if type(sequencer) is not VirtualSequencer:
            raise TypeError("this endpoint is specific to VirtualSequencer")
        if not isinstance(manifest, PulseTargetManifest):
            raise TypeError("manifest must be PulseTargetManifest")
        if manifest.target != sequencer.target:
            raise ValueError("virtual manifest target differs from sequencer target")
        self._sequencer = sequencer
        self._timeout = _positive_real(
            max_blocking_call_seconds,
            "max_blocking_call_seconds",
        )
        self._params = params or StreamerParams()
        self._manifest = manifest
        self._target = manifest.target
        self._geometry = build_fingerprint(self._params) & 0xFFFFFFFF
        self._owner = _SequencerSessionOwner(
            self,
            max_blocking_call_seconds=self._timeout,
        )

    def _backend_capability_snapshot(
        self, binding: BoundDevice
    ) -> SequencerCapabilitySnapshot:
        fingerprint = canonical_digest(
            {
                "contract": "zlc.virtual-sequencer-execution",
                "target_abi_fingerprint": self._target.abi_fingerprint,
                "manifest_fingerprint": self._manifest.fingerprint,
                "clock_hz": float(self._sequencer.clock_hz),
                "geometry_fingerprint": self._geometry,
                "resident_scan_point_capacity": resident_scan_point_capacity(
                    self._params
                ),
                "max_blocking_call_seconds": self._timeout,
                "terminal_evidence_kind": PulseTerminalEvidenceKind.SIMULATED.value,
            }
        )
        return SequencerCapabilitySnapshot(
            binding_stamp=binding.binding_stamp,
            manifest=self._manifest,
            clock_hz=float(self._sequencer.clock_hz),
            geometry_fingerprint=self._geometry,
            resident_scan_point_capacity=resident_scan_point_capacity(
                self._params
            ),
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
        validate_resident_scan_capacity(artifact, self._params)

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

    def _backend_observe_scan_progress(
        self,
        session: _EndpointSession,
    ) -> PulseScanProgress:
        point_index, backend_state, reason = self._sequencer.observe_scan_cursor(
            session.artifact_digest
        )
        if point_index is None:
            assert reason is not None
            return PulseScanProgress.unavailable(
                run_id=session.run_id,
                artifact_digest=session.artifact_digest,
                point_count=len(session.request.artifact.target_ir.scan_points),
                backend_state=backend_state,
                reason=reason,
            )
        return PulseScanProgress(
            session.run_id,
            session.artifact_digest,
            len(session.request.artifact.target_ir.scan_points),
            point_index,
            backend_state,
        )

    def _backend_wait_continuous_failure(
        self,
        session: _EndpointSession,
        timeout: float,
    ) -> str | None:
        time.sleep(timeout)
        snapshot = self._sequencer.snapshot()
        output = self._sequencer.output_artifact
        if (
            snapshot.get("state") == "running"
            and output is not None
            and output.fingerprint == session.artifact_digest
        ):
            return None
        return "virtual continuous sequencer stopped outside its session owner"

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
            "zlc_neutral_atom.target.VirtualSequencer",
            counts,
            playback.logical_duration,
            artifact.max_configured_output_delay_ticks / artifact.target_ir.clock_hz,
        )

    def _backend_set_safe_state(self, _timeout_seconds: float) -> object:
        self._sequencer.set_safe_state()
        return dict(self._sequencer.snapshot())

    def _backend_close_evidence(
        self,
        session_id: str,
        snapshot: object,
    ) -> tuple[bool, str]:
        if not isinstance(snapshot, dict):
            raise TypeError("virtual sequencer safe-state snapshot must be a mapping")
        safe = (
            snapshot.get("state") == "safe"
            and snapshot.get("prepared_program") is None
            and self._sequencer.firing is None
            and self._sequencer.last_fired is None
        )
        return safe, canonical_digest(
            {
                "session_id": session_id,
                "state": snapshot.get("state"),
                "prepared_program": snapshot.get("prepared_program"),
                "firing": self._sequencer.firing is not None,
                "last_fired": self._sequencer.last_fired is not None,
            }
        )

    def _backend_interrupt_digest(self, snapshot: object) -> str:
        if not isinstance(snapshot, dict):
            raise TypeError("virtual sequencer safe-state snapshot must be a mapping")
        return canonical_digest(
            {"operation": "SAFE_STATE", "state": snapshot.get("state")}
        )


class RemotePulseExecutionEndpoint(_OwnedSequencerEndpoint):
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
        limit = _positive_real(
            (
                client.transport_timeout_seconds * 0.9
                if max_blocking_call_seconds is None
                else max_blocking_call_seconds
            ),
            "max_blocking_call_seconds",
        )
        if limit >= client.transport_timeout_seconds:
            raise ValueError(
                "max blocking call must be shorter than the client transport backstop"
            )
        snapshot = client.snapshot()
        self._client = client
        self._endpoint_label = _text(endpoint_label, "endpoint_label")
        self._timeout = limit
        self._manifest = snapshot.manifest
        self._target = self._manifest.target
        self._clock_hz = snapshot.clock_hz
        self._geometry = snapshot.geometry_fingerprint
        self._resident_scan_point_capacity = snapshot.resident_scan_point_capacity
        self._server_connection_generation = snapshot.connection_generation
        self._owner = _SequencerSessionOwner(
            self,
            max_blocking_call_seconds=self._timeout,
        )

    def _backend_capability_snapshot(
        self,
        binding: BoundDevice,
    ) -> SequencerCapabilitySnapshot:
        snapshot = self._validate_server_connection_generation()
        if snapshot.state not in {"IDLE", "SAFE", "DONE"}:
            raise RuntimeError(
                "remote pulse server is not ready for capability probe: "
                f"{snapshot.state}"
            )
        fingerprint = canonical_digest(
            {
                "contract": "zlc.remote-pulse-execution",
                "endpoint_label": self._endpoint_label,
                "server_connection_generation": (
                    self._server_connection_generation
                ),
                "target_abi_fingerprint": self._target.abi_fingerprint,
                "manifest_fingerprint": self._manifest.fingerprint,
                "clock_hz": self._clock_hz,
                "geometry_fingerprint": self._geometry,
                "resident_scan_point_capacity": self._resident_scan_point_capacity,
                "max_blocking_call_seconds": self._timeout,
                "terminal_evidence_kind": (
                    PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS.value
                ),
            }
        )
        return SequencerCapabilitySnapshot(
            binding_stamp=binding.binding_stamp,
            manifest=self._manifest,
            clock_hz=self._clock_hz,
            geometry_fingerprint=self._geometry,
            resident_scan_point_capacity=self._resident_scan_point_capacity,
            max_blocking_call_seconds=self._timeout,
            terminal_evidence_kind=(
                PulseTerminalEvidenceKind.HARDWARE_RAW_REGISTERS
            ),
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

    def _backend_prepare(self, session: _EndpointSession) -> PreparedPulseRef:
        return self._client.prepare(session.request.artifact)

    def _backend_fire(self, session: _EndpointSession) -> None:
        reference = session.prepared_ref
        if reference is None:
            raise RuntimeError("remote sequencer session has no prepared reference")
        self._client.fire(reference)

    def _backend_observe_scan_progress(
        self,
        session: _EndpointSession,
    ) -> PulseScanProgress:
        snapshot = self._validate_server_connection_generation()
        point_count = len(session.request.artifact.target_ir.scan_points)

        def unavailable(reason: str) -> PulseScanProgress:
            backend_state = snapshot.backend.get("state")
            return PulseScanProgress.unavailable(
                run_id=session.run_id,
                artifact_digest=session.artifact_digest,
                point_count=point_count,
                backend_state=(
                    backend_state if isinstance(backend_state, str) else snapshot.state
                ),
                reason=reason,
            )

        reference = session.prepared_ref
        if reference is None or snapshot.prepared_ref != reference:
            return unavailable("remote server no longer owns the prepared artifact")
        backend = snapshot.backend
        if snapshot.state != "RUNNING" or backend.get("state") != "RUNNING":
            return unavailable("remote sequencer is not reporting an active scan")
        if backend.get("prepared_artifact_digest") != session.artifact_digest:
            return unavailable("remote backend artifact identity differs")
        if backend.get("scan_points") != point_count:
            return unavailable("remote backend scan table cardinality differs")
        sample_count = backend.get("cursor_sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
        ):
            return unavailable("remote backend has not sampled a scan cursor yet")
        point_index = backend.get("last_confirmed_cursor")
        if (
            isinstance(point_index, bool)
            or not isinstance(point_index, int)
            or not 0 <= point_index < point_count
        ):
            return unavailable("remote backend cursor is outside the scan table")
        return PulseScanProgress(
            session.run_id,
            session.artifact_digest,
            point_count,
            point_index,
            "RUNNING",
        )

    def _backend_wait_continuous_failure(
        self,
        session: _EndpointSession,
        timeout: float,
    ) -> str | None:
        reference = session.prepared_ref
        if reference is None:
            return "remote continuous sequencer lost its prepared reference"
        return self._client.wait_continuous_failure(reference, timeout=timeout)

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

    def _backend_set_safe_state(self, timeout_seconds: float) -> object:
        return self._client.safe_state(timeout=timeout_seconds)

    def _backend_close_evidence(
        self,
        session_id: str,
        snapshot: object,
    ) -> tuple[bool, str]:
        state = getattr(snapshot, "state", None)
        prepared_ref = getattr(snapshot, "prepared_ref", None)
        safe = state == "SAFE" and prepared_ref is None
        return safe, canonical_digest(
            {
                "session_id": session_id,
                "server_connection_generation": (
                    self._server_connection_generation
                ),
                "state": state,
                "prepared_ref": None if prepared_ref is None else "present",
                "backend": getattr(snapshot, "backend", None),
            }
        )

    def _backend_interrupt_digest(self, snapshot: object) -> str:
        return canonical_digest(
            {
                "operation": "SAFE_STATE",
                "server_connection_generation": (
                    self._server_connection_generation
                ),
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
            or snapshot.resident_scan_point_capacity
            != self._resident_scan_point_capacity
        ):
            raise RuntimeError(
                "remote pulse server capability changed within one connection"
            )
        return snapshot


__all__ = [
    "RemotePulseExecutionEndpoint",
    "VirtualSequencerExecutionEndpoint",
]
