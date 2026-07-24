"""Typed pulse endpoint for the installation-owned virtual sequencer."""

from __future__ import annotations

import time

from fpga.pulse_streamer.host.image import StreamerParams, build_fingerprint
from zlc_neutral_atom.devices.sequencer.endpoint import (
    _EndpointSession,
    _OwnedSequencerEndpoint,
    _SequencerSessionOwner,
)
from zlc_neutral_atom.runtime.ports import BoundDevice
from zlc_neutral_atom.devices.sequencer.port import (
    PreparePulseCommand,
    PulseScanProgress,
    PulseTerminalEvidenceKind,
    PulseTerminalReceipt,
    SequencerCapabilitySnapshot,
    SimulatedPulseReceipt,
)
from zlc_pulse import (
    PulseTargetManifest,
    build_pulse_playback,
    resident_scan_point_capacity,
    validate_resident_scan_capacity,
)
from zlc_storage import canonical_digest, positive_real as _positive_real

from .apparatus import VirtualSequencer


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
            resident_scan_point_capacity=resident_scan_point_capacity(self._params),
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


__all__ = ["VirtualSequencerExecutionEndpoint"]
