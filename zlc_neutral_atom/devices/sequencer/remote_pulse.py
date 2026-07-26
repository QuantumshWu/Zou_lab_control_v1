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
from zlc_pulse import PreparedPulseRef, RemotePulseExecutionClient
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
                "server_connection_generation": self._server_connection_generation,
                "state": state,
                "prepared_ref": None if prepared_ref is None else "present",
                "backend": getattr(snapshot, "backend", None),
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
            raise RuntimeError(
                "remote pulse server capability changed within one connection"
            )
        return snapshot


__all__ = ["RemotePulseExecutionEndpoint"]
