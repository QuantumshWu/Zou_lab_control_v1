"""Camera Measurement's named live-event projection.

The raw Camera stream remains owned by Camera Measurement.  The ordinary path
maps declared ``frame_i`` outputs onto source-neutral ordered events.  A target
composition may additionally inject a producer-owned pulse-association
authority; only then does the resulting source expose the stronger associated
cursor used by PulseScan.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from zlc_neutral_atom.devices.camera.contract import (
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.devices.sequencer.port import (
    PulseTerminalAck,
    PulseTerminalEvidenceKind,
    pulse_terminal_ack_to_tree,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CameraMeasurementRequest,
    camera_frame_output_index,
)
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationEvidence,
    SignalAssociationRequest,
    SignalAssociationScheduleRequirement,
    SignalAssociationUnavailable,
    SignalEvent,
    SignalEventCursor,
    SignalOutputProjection,
    StreamSignalEventSource,
)
from zlc_neutral_atom.runtime.streams import AcquisitionStream
from zlc_storage import canonical_digest, canonical_text, encode, sha256_text


_VIRTUAL_CAMERA_ASSOCIATION_SCHEMA = (
    "zlc_neutral_atom.camera-measurement.virtual-pulse-association"
)


@runtime_checkable
class CameraSignalAssociationAuthority(Protocol):
    """Optional target-owned physical association seam.

    This is intentionally separate from ``CameraAdapter``.  A composition may
    provide it only when its concrete producer can observe FIRE and camera
    ordinals at their shared physical boundary.
    """

    def arm_signal_event_association(
        self,
        association_id: str,
        cause_digest: str,
        expected_trigger_count: int,
        trigger_group_size: int,
        expected_group_count: int,
    ) -> tuple[object, int]: ...

    def bind_signal_event_association(
        self,
        token: object,
        *,
        artifact_digest: str,
        trigger_counts: tuple[tuple[str, int], ...],
        terminal_evidence_digest: str,
    ) -> tuple[str, int, int]: ...

    def finish_signal_event_association(
        self,
        token: object,
    ) -> tuple[str, int, int, str]: ...

    def cancel_signal_event_association(self, token: object) -> None: ...


@dataclass(slots=True)
class _CameraAssociationToken:
    request: SignalAssociationRequest
    authority_token: object
    physical_start_ordinal: int
    physical_end_ordinal: int | None = None
    trigger_channel: str | None = None
    terminal_evidence_digest: str | None = None
    delivered_count: int = 0
    bound: bool = False


class _CameraAssociatedSignalEventCursor:
    """Select one frame phase from an authority-admitted physical FIRE group."""

    __slots__ = (
        "_authority",
        "_binding_instance_id",
        "_capability_fingerprint",
        "_closed",
        "_cursor",
        "_frames_per_cycle",
        "_output_name",
        "_phase",
        "_token",
        "_trigger_channel",
    )

    def __init__(
        self,
        cursor: SignalEventCursor[CameraSample],
        *,
        authority: CameraSignalAssociationAuthority,
        output_name: str,
        phase: int,
        frames_per_cycle: int,
        trigger_channel: str,
        capability_fingerprint: str,
        binding_instance_id: str,
    ) -> None:
        self._cursor = cursor
        self._authority = authority
        self._output_name = canonical_text(output_name, "camera output name")
        self._phase = phase
        self._frames_per_cycle = frames_per_cycle
        self._trigger_channel = canonical_text(
            trigger_channel,
            "camera trigger channel",
        )
        self._capability_fingerprint = sha256_text(
            capability_fingerprint,
            "camera capability fingerprint",
        )
        self._binding_instance_id = canonical_text(
            binding_instance_id,
            "camera binding instance id",
        )
        self._token: _CameraAssociationToken | None = None
        self._closed = False

    @property
    def value_schema(self):
        return self._cursor.value_schema

    @property
    def stream_id(self):
        return self._cursor.stream_id

    @property
    def stream_generation(self):
        return self._cursor.stream_generation

    @property
    def start_sequence(self) -> int:
        return self._cursor.start_sequence

    def arm_signal_association(self, request: SignalAssociationRequest) -> object:
        self._ensure_open()
        if not isinstance(request, SignalAssociationRequest):
            raise TypeError("camera association requires SignalAssociationRequest")
        if self._token is not None:
            raise RuntimeError("camera association cursor already owns a token")
        physical_count = request.expected_event_count * self._frames_per_cycle
        authority_token, physical_start = (
            self._authority.arm_signal_event_association(
                request.association_id,
                request.cause_digest,
                physical_count,
                self._frames_per_cycle,
                request.expected_event_count,
            )
        )
        if (
            isinstance(physical_start, bool)
            or not isinstance(physical_start, int)
            or physical_start < 0
        ):
            self._authority.cancel_signal_event_association(authority_token)
            raise TypeError(
                "camera association authority returned an invalid start ordinal"
            )
        if physical_start % self._frames_per_cycle:
            self._authority.cancel_signal_event_association(authority_token)
            raise RuntimeError(
                "camera association starts inside a declared readout cycle"
            )
        token = _CameraAssociationToken(
            request,
            authority_token,
            physical_start,
        )
        self._token = token
        return token

    def bind_signal_association(
        self,
        token: object,
        terminal_evidence: object,
    ) -> None:
        current = self._require_token(token)
        if not isinstance(terminal_evidence, PulseTerminalAck):
            raise TypeError("camera association requires PulseTerminalAck")
        if terminal_evidence.evidence_kind is not PulseTerminalEvidenceKind.SIMULATED:
            raise ValueError(
                "this association owner proves only the virtual in-process trigger wire"
            )
        if terminal_evidence.session_id != current.request.cause_id:
            raise ValueError("pulse terminal belongs to another association cause")
        if terminal_evidence.artifact_digest != current.request.cause_digest:
            raise ValueError("pulse terminal belongs to another compiled artifact")
        terminal_digest = canonical_digest(
            pulse_terminal_ack_to_tree(terminal_evidence)
        )
        channel, start, end = self._authority.bind_signal_event_association(
            current.authority_token,
            artifact_digest=terminal_evidence.artifact_digest,
            trigger_counts=(
                terminal_evidence.expected_trigger_counts_from_completed_schedule
            ),
            terminal_evidence_digest=terminal_digest,
        )
        if canonical_text(channel, "associated trigger channel") != self._trigger_channel:
            raise RuntimeError("association authority returned another trigger channel")
        expected_end = (
            current.physical_start_ordinal
            + current.request.expected_event_count * self._frames_per_cycle
        )
        if start != current.physical_start_ordinal or end != expected_end:
            raise RuntimeError(
                "association authority returned another physical ordinal interval"
            )
        current.physical_end_ordinal = end
        current.trigger_channel = channel
        current.terminal_evidence_digest = terminal_digest
        current.bound = True

    def next_associated_signal(
        self,
        token: object,
        timeout: float | None = None,
    ) -> SignalEvent:
        current = self._require_token(token)
        if not current.bound:
            raise RuntimeError("camera association is not bound to a terminal")
        if current.delivered_count >= current.request.expected_event_count:
            raise RuntimeError("camera association exhausted its selected events")
        expected_sequence = (
            current.physical_start_ordinal
            + self._phase
            + current.delivered_count * self._frames_per_cycle
        )
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            event = self._cursor.next(remaining)
            sequence = event.event_ref.sequence
            if sequence < expected_sequence:
                continue
            if sequence != expected_sequence:
                raise RuntimeError(
                    "camera associated event sequence left its physical ordinal group"
                )
            current.delivered_count += 1
            return event

    def finish_signal_association(
        self,
        token: object,
    ) -> SignalAssociationEvidence:
        current = self._require_token(token)
        if not current.bound:
            raise RuntimeError("camera association is not bound to a terminal")
        if current.delivered_count != current.request.expected_event_count:
            raise RuntimeError(
                "camera association cannot finish before every selected event"
            )
        channel, start, end, terminal_digest = (
            self._authority.finish_signal_event_association(
                current.authority_token
            )
        )
        if (
            channel != current.trigger_channel
            or start != current.physical_start_ordinal
            or end != current.physical_end_ordinal
            or terminal_digest != current.terminal_evidence_digest
        ):
            raise RuntimeError(
                "camera association finish differs from its bound physical interval"
            )
        request = current.request
        canonical_evidence = encode(
            {
                "schema": _VIRTUAL_CAMERA_ASSOCIATION_SCHEMA,
                "association_id": request.association_id,
                "cause_id": request.cause_id,
                "cause_digest": request.cause_digest,
                "expected_event_count": request.expected_event_count,
                "terminal_evidence_digest": terminal_digest,
                "camera_capability_fingerprint": self._capability_fingerprint,
                "camera_binding_instance_id": self._binding_instance_id,
                "stream_id": self.stream_id.value,
                "stream_generation": self.stream_generation.value,
                "output_name": self._output_name,
                "trigger_channel": channel,
                "physical_start_ordinal": start,
                "physical_end_ordinal": end,
                "physical_trigger_count": end - start,
                "frames_per_cycle": self._frames_per_cycle,
                "selected_phase": self._phase,
                "evidence_kind": "SIMULATED_IN_PROCESS_TRIGGER_WIRE",
            }
        )
        evidence = SignalAssociationEvidence(
            request,
            terminal_digest,
            _VIRTUAL_CAMERA_ASSOCIATION_SCHEMA,
            canonical_evidence,
        )
        self._token = None
        return evidence

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        token = self._token
        self._token = None
        try:
            if token is not None:
                self._authority.cancel_signal_event_association(
                    token.authority_token
                )
        finally:
            self._cursor.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("camera association cursor is closed")

    def _require_token(self, token: object) -> _CameraAssociationToken:
        self._ensure_open()
        current = self._token
        if current is None or current is not token:
            raise RuntimeError("camera association token is not current")
        return current


class CameraAssociatedSignalEventSource:
    """Camera source whose target composition can prove virtual pulse groups."""

    __slots__ = (
        "_association_cursor_opened",
        "_association_running",
        "_authority",
        "_binding_lock",
        "_binding_instance_id",
        "_capability_fingerprint",
        "_frames_per_cycle",
        "_phases",
        "_source",
        "_trigger_channel",
    )

    def __init__(
        self,
        source: StreamSignalEventSource[CameraSample],
        *,
        authority: CameraSignalAssociationAuthority,
        phases: dict[str, int],
        frames_per_cycle: int,
        trigger_channel: str,
        capability_fingerprint: str,
        binding_instance_id: str,
    ) -> None:
        if not isinstance(authority, CameraSignalAssociationAuthority):
            raise TypeError("camera association authority has an incomplete contract")
        self._source = source
        self._authority = authority
        self._binding_lock = threading.Lock()
        self._association_cursor_opened = False
        self._association_running = False
        self._phases = dict(phases)
        self._frames_per_cycle = frames_per_cycle
        self._trigger_channel = canonical_text(
            trigger_channel,
            "camera trigger channel",
        )
        self._capability_fingerprint = sha256_text(
            capability_fingerprint,
            "camera capability fingerprint",
        )
        self._binding_instance_id = canonical_text(
            binding_instance_id,
            "camera binding instance id",
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._source.output_names

    def value_schema(self, output_name: str):
        return self._source.value_schema(output_name)

    def open_signal_cursor(self, output_name: str):
        return self._source.open_signal_cursor(output_name)

    def bind_capability_fingerprint(self, capability_fingerprint: str) -> None:
        """Freeze the endpoint-read working point before association starts."""

        fingerprint = sha256_text(
            capability_fingerprint,
            "camera capability fingerprint",
        )
        with self._binding_lock:
            if self._association_cursor_opened:
                raise RuntimeError(
                    "camera capability cannot change after association cursor open"
                )
            self._capability_fingerprint = fingerprint

    def mark_association_running(self) -> None:
        """Publish readiness only after the Camera endpoint acknowledged arm."""

        with self._binding_lock:
            self._association_running = True

    def mark_association_stopped(self) -> None:
        """Withdraw readiness at the beginning of Camera cleanup."""

        with self._binding_lock:
            self._association_running = False

    def _require_association_running(self) -> None:
        with self._binding_lock:
            if not self._association_running:
                raise SignalAssociationUnavailable(
                    "Camera signal association requires an already-running, "
                    "armed Camera producer"
                )

    def signal_association_schedule_requirement(
        self,
        output_name: str,
    ) -> SignalAssociationScheduleRequirement:
        self._require_association_running()
        name = canonical_text(output_name, "camera output name")
        if name not in self._phases:
            raise KeyError(f"camera has no signal output {name!r}")
        return SignalAssociationScheduleRequirement((self._trigger_channel,))

    def open_associated_signal_cursor(self, output_name: str):
        self._require_association_running()
        name = canonical_text(output_name, "camera output name")
        try:
            phase = self._phases[name]
        except KeyError as error:
            raise KeyError(f"camera has no signal output {name!r}") from error
        with self._binding_lock:
            self._association_cursor_opened = True
            capability_fingerprint = self._capability_fingerprint
        cursor = self._source.open_signal_cursor(name)
        try:
            return _CameraAssociatedSignalEventCursor(
                cursor,
                authority=self._authority,
                output_name=name,
                phase=phase,
                frames_per_cycle=self._frames_per_cycle,
                trigger_channel=self._trigger_channel,
                capability_fingerprint=capability_fingerprint,
                binding_instance_id=self._binding_instance_id,
            )
        except BaseException:
            cursor.close()
            raise


def camera_signal_event_source(
    stream: AcquisitionStream[CameraSample],
    request: CameraMeasurementRequest,
    payload_contract: CameraSampleContract,
    *,
    association_authority: CameraSignalAssociationAuthority | None = None,
    trigger_channel: str | None = None,
    capability_fingerprint: str | None = None,
    binding_instance_id: str | None = None,
) -> StreamSignalEventSource[CameraSample] | CameraAssociatedSignalEventSource:
    """Expose each declared READOUT_EVENT phase as one named live output."""

    if not isinstance(stream, AcquisitionStream):
        raise TypeError("camera signal source requires an AcquisitionStream")
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("camera signal source requires CameraMeasurementRequest")
    if not isinstance(payload_contract, CameraSampleContract):
        raise TypeError("camera signal source requires CameraSampleContract")
    if stream.payload_contract_fingerprint != payload_contract.fingerprint:
        raise ValueError("camera stream payload contract differs from the adapter")

    phase_count = request.frames_per_cycle
    outputs: dict[str, SignalOutputProjection[CameraSample]] = {}
    phases: dict[str, int] = {}
    for output_name in request.output_names:
        phase = camera_frame_output_index(output_name)
        phases[output_name] = phase

        def project(envelope, *, selected_phase: int = phase):
            if envelope.sequence % phase_count != selected_phase:
                return None
            payload = envelope.payload
            payload_contract.validate(payload)
            return payload.image

        outputs[output_name] = SignalOutputProjection(
            payload_contract.value_schema,
            project,
        )
    source = StreamSignalEventSource(stream, outputs)
    if association_authority is None:
        if any(
            value is not None
            for value in (
                trigger_channel,
                capability_fingerprint,
                binding_instance_id,
            )
        ):
            raise ValueError(
                "camera association facts require an association authority"
            )
        return source
    if (
        trigger_channel is None
        or capability_fingerprint is None
        or binding_instance_id is None
    ):
        raise ValueError("camera association authority requires complete binding facts")
    return CameraAssociatedSignalEventSource(
        source,
        authority=association_authority,
        phases=phases,
        frames_per_cycle=phase_count,
        trigger_channel=trigger_channel,
        capability_fingerprint=capability_fingerprint,
        binding_instance_id=binding_instance_id,
    )


__all__ = [
    "CameraAssociatedSignalEventSource",
    "CameraSignalAssociationAuthority",
    "camera_signal_event_source",
]
