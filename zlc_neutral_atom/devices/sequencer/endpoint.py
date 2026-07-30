"""Composition-owned sequencer endpoint for the typed pulse Port."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SessionClosedAck,
    SessionCloseCommand,
    require_current_endpoint_binding as _require_binding,
)
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.devices.sequencer.port import (
    CompletePulseCommand,
    FirePulseCommand,
    PreparePulseCommand,
    PulseFiredAck,
    PulsePreparedAck,
    PulseScanProgress,
    PulseTerminalAck,
    SequencerCapabilitySnapshot,
)
from zlc_pulse import (
    PulseExecutionForm,
    PreparedPulseRef,
)
from zlc_storage import (
    canonical_text as _text,
    positive_real as _positive_real,
)


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
        backend: _OwnedSequencerEndpoint,
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
        snapshot = self._set_safe_state(deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError("sequencer close exceeded its bounded deadline")
        if session is not None and not self._wait_until_joined(session, deadline):
            raise TimeoutError("sequencer physical operation did not join before close")
        safe = self._backend._backend_safe_state_confirmed(snapshot)
        acknowledgement = SessionClosedAck(
            command.session_id,
            binding.binding_instance_id,
            safe,
            safe,
            safe,
        )
        with self._condition:
            if session is not None:
                if session.physical_operation_in_flight:
                    raise RuntimeError("sequencer close lost its joined state")
                session.close_acknowledged = True
        return acknowledgement

    def interrupt(self) -> None:
        with self._condition:
            self._operation_epoch += 1
            if self._session is not None:
                self._session.closed = True
        snapshot = self._set_safe_state()
        if not self._backend._backend_safe_state_confirmed(snapshot):
            raise RuntimeError("sequencer interrupt did not reach verified SAFE state")

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

    def interrupt(self) -> None:
        self._owner.interrupt()


__all__: list[str] = []
