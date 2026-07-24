"""Physical camera command protocol and bound device authority."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol, TypeVar

from zlc_storage import (
    canonical_text as _canonical_text,
    exact_mapping as _exact_tree,
    nonnegative_integer as _nonnegative_int,
    positive_integer as _positive_int,
    positive_real as _positive_finite,
    sha256_text as _sha256,
)

from zlc_neutral_atom.devices.camera.contract import (
    CameraCapabilityEvidence,
    CameraPhysicalFacts,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SafetyInterrupt,
    SafetyOperation,
    VerifiedDeviceCapability,
    admit_bound_capability,
    cleanup_device_session,
)
from zlc_neutral_atom.runtime.resources import DeviceBindingStamp, ResourceClaim
from zlc_neutral_atom.runtime.run import RunContext
from zlc_neutral_atom.runtime.streams import PayloadContract


PayloadT = TypeVar("PayloadT")


class CapturePayloadContract(PayloadContract[PayloadT], Protocol[PayloadT]):
    def source_ordinal(self, payload: PayloadT) -> int: ...

    def captured_at(self, payload: PayloadT) -> float: ...

    def correlation_id(self, payload: PayloadT) -> str: ...




@dataclass(frozen=True)
class CaptureCapabilitySnapshot:
    binding_stamp: DeviceBindingStamp
    payload_contract: CapturePayloadContract
    camera_capability_evidence: CameraCapabilityEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("binding_stamp must be DeviceBindingStamp")
        evidence = self.camera_capability_evidence
        if not isinstance(evidence, CameraCapabilityEvidence):
            raise TypeError(
                "camera_capability_evidence must be CameraCapabilityEvidence"
            )
        if getattr(self.payload_contract, "fingerprint", None) != (
            evidence.payload_contract_fingerprint
        ):
            raise ValueError(
                "camera capability payload contract differs from its fingerprint"
            )
        if evidence.physical_facts.camera_identity != (
            self.binding_stamp.physical_identity.stable_device_identity
        ):
            raise ValueError(
                "camera physical identity differs from capability stable identity"
            )

    @property
    def capability_fingerprint(self) -> str:
        return self.camera_capability_evidence.fingerprint

    @property
    def settings_fingerprint(self) -> str:
        return self.camera_capability_evidence.settings_fingerprint

    @property
    def capture_spec_owner_fingerprint(self) -> str:
        return self.camera_capability_evidence.capture_spec_owner_fingerprint

    @property
    def max_blocking_call_seconds(self) -> float:
        return self.camera_capability_evidence.max_blocking_call_seconds

    @property
    def camera_physical_facts(self) -> CameraPhysicalFacts:
        return self.camera_capability_evidence.physical_facts


@dataclass(frozen=True)
class ConfigureCameraExposureCommand:
    """Apply/read back one exposure under a cleanup-closeable lease."""

    session_id: str
    exposure_seconds: float
    baseline_settings_fingerprint: str

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "exposure_seconds",
            _positive_finite(self.exposure_seconds, "exposure_seconds"),
        )
        _sha256(
            self.baseline_settings_fingerprint,
            "baseline_settings_fingerprint",
        )


@dataclass(frozen=True)
class CameraExposureConfiguredAck:
    session_id: str
    binding_instance_id: str
    requested_exposure_seconds: float
    applied_exposure_seconds: float
    required_external_trigger_interval_seconds: float
    settings_fingerprint: str
    capability_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_instance_id"):
            _canonical_text(getattr(self, name), name)
        for name in (
            "requested_exposure_seconds",
            "applied_exposure_seconds",
        ):
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), name),
            )
        interval = float(self.required_external_trigger_interval_seconds)
        if not math.isfinite(interval) or interval < 0.0:
            raise ValueError(
                "required_external_trigger_interval_seconds must be finite "
                "and non-negative"
            )
        object.__setattr__(
            self,
            "required_external_trigger_interval_seconds",
            interval,
        )
        _sha256(self.settings_fingerprint, "settings_fingerprint")
        _sha256(self.capability_fingerprint, "capability_fingerprint")

@dataclass(frozen=True)
class PrepareCaptureCommand:
    session_id: str
    capture_spec_payload: bytes
    capture_spec_owner_fingerprint: str
    capture_spec_fingerprint: str
    capability_fingerprint: str
    settings_fingerprint: str
    expected_total_events: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        for name in (
            "capture_spec_owner_fingerprint",
            "capture_spec_fingerprint",
            "capability_fingerprint",
            "settings_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        if not isinstance(self.capture_spec_payload, bytes) or not self.capture_spec_payload:
            raise ValueError("capture_spec_payload must be non-empty bytes")
        if hashlib.sha256(self.capture_spec_payload).hexdigest() != self.capture_spec_fingerprint:
            raise ValueError("capture spec payload digest differs")
        object.__setattr__(
            self,
            "expected_total_events",
            _positive_int(self.expected_total_events, "expected_total_events"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_finite(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CapturePreparedAck:
    session_id: str
    binding_instance_id: str
    settings_fingerprint: str
    capability_fingerprint: str
    capture_spec_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_instance_id"):
            _canonical_text(getattr(self, name), name)
        _sha256(self.settings_fingerprint, "settings_fingerprint")
        _sha256(self.capability_fingerprint, "capability_fingerprint")
        _sha256(self.capture_spec_fingerprint, "capture_spec_fingerprint")


@dataclass(frozen=True)
class StartCaptureCommand:
    session_id: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_finite(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CaptureStartedAck:
    session_id: str
    binding_instance_id: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_instance_id"):
            _canonical_text(getattr(self, name), name)


@dataclass(frozen=True)
class ReadCaptureCommand:
    session_id: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_finite(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CapturedPayloadAck:
    session_id: str
    binding_instance_id: str
    payload: object

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_instance_id"):
            _canonical_text(getattr(self, name), name)


@dataclass(frozen=True)
class CompleteCaptureCommand:
    session_id: str
    expected_total_events: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        _canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "expected_total_events",
            _positive_int(self.expected_total_events, "expected_total_events"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_finite(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CaptureTerminalAck:
    session_id: str
    binding_instance_id: str
    produced_count: int
    drained_count: int
    source_stopped: bool
    no_more_frames: bool
    joined: bool
    ordered_metadata_digest: str
    settings_fingerprint: str
    capability_fingerprint: str
    capture_spec_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_instance_id"):
            _canonical_text(getattr(self, name), name)
        object.__setattr__(
            self,
            "produced_count",
            _nonnegative_int(self.produced_count, "produced_count"),
        )
        object.__setattr__(
            self,
            "drained_count",
            _nonnegative_int(self.drained_count, "drained_count"),
        )
        for name in ("source_stopped", "no_more_frames", "joined"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        _sha256(self.ordered_metadata_digest, "ordered_metadata_digest")
        _sha256(self.settings_fingerprint, "settings_fingerprint")
        _sha256(self.capability_fingerprint, "capability_fingerprint")
        _sha256(self.capture_spec_fingerprint, "capture_spec_fingerprint")


def capture_terminal_ack_to_tree(value: CaptureTerminalAck) -> dict[str, object]:
    if not isinstance(value, CaptureTerminalAck):
        raise TypeError("value must be CaptureTerminalAck")
    return {
        "session_id": value.session_id,
        "binding_instance_id": value.binding_instance_id,
        "produced_count": value.produced_count,
        "drained_count": value.drained_count,
        "source_stopped": value.source_stopped,
        "no_more_frames": value.no_more_frames,
        "joined": value.joined,
        "ordered_metadata_digest": value.ordered_metadata_digest,
        "settings_fingerprint": value.settings_fingerprint,
        "capability_fingerprint": value.capability_fingerprint,
        "capture_spec_fingerprint": value.capture_spec_fingerprint,
    }


def capture_terminal_ack_from_tree(tree: object) -> CaptureTerminalAck:
    data = _exact_tree(
        tree,
        {
            "session_id",
            "binding_instance_id",
            "produced_count",
            "drained_count",
            "source_stopped",
            "no_more_frames",
            "joined",
            "ordered_metadata_digest",
            "settings_fingerprint",
            "capability_fingerprint",
            "capture_spec_fingerprint",
        },
        "capture terminal acknowledgement",
        discriminator=None,
    )
    return CaptureTerminalAck(**data)

@dataclass(frozen=True)
class BoundCapturePort:
    capability_attestation: VerifiedDeviceCapability

    def __post_init__(self) -> None:
        admit_bound_capability(
            self.capability_attestation,
            CaptureCapabilitySnapshot,
        )
        if not self.device.session_cleanup_capable:
            raise ValueError("capture port requires session-specific cleanup capability")
        if not any(
            operation in self.device.interrupt_capabilities
            for operation in (SafetyOperation.ABORT, SafetyOperation.DISARM)
        ):
            raise ValueError("capture port requires a thread-safe ABORT or DISARM interrupt")

    @property
    def device(self) -> BoundDevice:
        return self.capability_attestation.device

    @property
    def capability(self) -> CaptureCapabilitySnapshot:
        snapshot = self.capability_attestation.snapshot
        assert isinstance(snapshot, CaptureCapabilitySnapshot)
        return snapshot

    @property
    def resource_claim(self) -> ResourceClaim:
        return ResourceClaim(self.device.key)

    @property
    def interrupt_operations(self) -> tuple[SafetyInterrupt, ...]:
        preferred = tuple(
            operation
            for operation in (SafetyOperation.ABORT, SafetyOperation.DISARM)
            if operation in self.device.interrupt_capabilities
        )
        return tuple(SafetyInterrupt(self.device.key, operation) for operation in preferred)

    def cleanup(self, context: RunContext, session_id: str) -> CleanupReport:
        device = context.cleanup_device(self.device.key)
        return cleanup_device_session(
            device,
            session_id,
            self.capability.max_blocking_call_seconds,
        )

    def verify_idle(self, _context: RunContext) -> CleanupReport:
        return CleanupReport.complete()



__all__ = [
    "BoundCapturePort",
    "CameraExposureConfiguredAck",
    "CaptureCapabilitySnapshot",
    "CapturePayloadContract",
    "CapturePreparedAck",
    "CaptureStartedAck",
    "CaptureTerminalAck",
    "CapturedPayloadAck",
    "CompleteCaptureCommand",
    "ConfigureCameraExposureCommand",
    "PrepareCaptureCommand",
    "ReadCaptureCommand",
    "StartCaptureCommand",
    "capture_terminal_ack_from_tree",
    "capture_terminal_ack_to_tree",
]
