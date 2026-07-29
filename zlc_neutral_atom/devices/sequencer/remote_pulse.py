"""Typed sequencer endpoint over the current pulse RPC client."""

from __future__ import annotations

from zlc_neutral_atom.runtime.ports import BoundDevice
from zlc_neutral_atom.devices.sequencer.port import (
    PreparePulseCommand,
    PulseScanProgress,
    PulseTerminalEvidenceKind,
    PulseTerminalReceipt,
    SequencerCapabilitySnapshot,
)
from zlc_pulse import (
    PreparedPulseRef,
    PulseServerSnapshot,
    RemotePulseExecutionClient,
)
from zlc_storage import (
    canonical_digest,
    canonical_text as _text,
    positive_real as _positive_real,
)

from .endpoint import (
    _EndpointSession,
    _OwnedSequencerEndpoint,
    _SequencerSessionOwner,
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
                "server_connection_generation": self._server_connection_generation,
                "target_abi_fingerprint": self._target.abi_fingerprint,
                "manifest_fingerprint": self._manifest.fingerprint,
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
            manifest=self._manifest,
            clock_hz=self._clock_hz,
            geometry_fingerprint=self._geometry,
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
            return PulseScanProgress.unavailable(
                run_id=session.run_id,
                artifact_digest=session.artifact_digest,
                point_count=point_count,
                backend_state=snapshot.physical_state,
                reason=reason,
            )

        reference = session.prepared_ref
        if reference is None or snapshot.prepared_ref != reference:
            return unavailable("remote server no longer owns the prepared artifact")
        if snapshot.state != "RUNNING" or snapshot.physical_state != "RUNNING":
            return unavailable("remote sequencer is not reporting an active scan")
        if snapshot.physical_prepared_artifact_digest != session.artifact_digest:
            return unavailable("remote backend artifact identity differs")
        if snapshot.physical_scan_point_count != point_count:
            return unavailable("remote backend scan table cardinality differs")
        if snapshot.physical_underflow_observed:
            return unavailable("remote backend observed a scan underflow")
        if snapshot.physical_cursor_sample_count < 1:
            return unavailable("remote backend has not sampled a scan cursor yet")
        point_index = snapshot.physical_scan_cursor
        if point_index is None or not 0 <= point_index < point_count:
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
        snapshot: PulseServerSnapshot,
    ) -> tuple[bool, str]:
        safe = (
            snapshot.state == "SAFE"
            and snapshot.prepared_ref is None
            and snapshot.safe_readback_confirmed
        )
        return safe, canonical_digest(
            {
                "session_id": session_id,
                "server_connection_generation": self._server_connection_generation,
                "state": snapshot.state,
                "prepared_ref": (
                    None if snapshot.prepared_ref is None else "present"
                ),
                "physical_state": snapshot.physical_state,
                "physical_prepared_artifact_digest": (
                    snapshot.physical_prepared_artifact_digest
                ),
                "safe_status_word": snapshot.safe_status_word,
                "safe_clock_enable_words": snapshot.safe_clock_enable_words,
            }
        )

    def _backend_interrupt_digest(self, snapshot: PulseServerSnapshot) -> str:
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
            raise RuntimeError(
                "remote pulse server capability changed within one connection"
            )
        return snapshot


__all__ = ["RemotePulseExecutionEndpoint"]
