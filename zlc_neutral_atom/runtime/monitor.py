"""Typed authority for a hardware-paced free-running camera monitor.

The monitor port is deliberately distinct from finite exact capture authority:
it may publish display-only samples, but it cannot mint an exact reservation or
sealed artifact.  Both ports still use the same composition-owned camera
endpoint, capability evidence, session cleanup, and physical DISARM boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_neutral_atom.acquisition.camera import CameraAcquisitionMode, CameraSample
from zlc_storage import (
    canonical_text,
    positive_integer,
    positive_real,
    sha256_text,
)

from .capture import CaptureCapabilitySnapshot
from .cleanup import CleanupReport
from .ports import (
    BoundDevice,
    SafetyInterrupt,
    SafetyOperation,
    VerifiedDeviceCapability,
    admit_bound_capability,
    cleanup_device_session,
    verify_cleanup_device_safe_state,
)
from .resources import ClaimMode, ResourceClaim
from .run import RunContext


class CameraMonitorInterrupted(RuntimeError):
    """An in-flight monitor operation was superseded by an external interrupt."""


@dataclass(frozen=True)
class CameraMonitorCapabilitySnapshot(CaptureCapabilitySnapshot):
    """Broker-attested camera facts qualified only for monitor acquisition."""

    acquisition_mode: CameraAcquisitionMode

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.acquisition_mode is not CameraAcquisitionMode.FREE_RUNNING:
            raise ValueError("camera monitor capability requires FREE_RUNNING mode")


@dataclass(frozen=True)
class PrepareCameraMonitorCommand:
    session_id: str
    capability_fingerprint: str
    settings_fingerprint: str
    max_inflight_frames: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        sha256_text(self.capability_fingerprint, "capability_fingerprint")
        sha256_text(self.settings_fingerprint, "settings_fingerprint")
        object.__setattr__(
            self,
            "max_inflight_frames",
            positive_integer(self.max_inflight_frames, "max_inflight_frames"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CameraMonitorPreparedAck:
    session_id: str
    binding_instance_id: str
    settings_fingerprint: str
    capability_fingerprint: str

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        canonical_text(self.binding_instance_id, "binding_instance_id")
        sha256_text(self.settings_fingerprint, "settings_fingerprint")
        sha256_text(self.capability_fingerprint, "capability_fingerprint")


@dataclass(frozen=True)
class StartCameraMonitorCommand:
    session_id: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CameraMonitorStartedAck:
    session_id: str
    binding_instance_id: str

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        canonical_text(self.binding_instance_id, "binding_instance_id")


@dataclass(frozen=True)
class ReadCameraMonitorCommand:
    session_id: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass(frozen=True)
class CameraMonitorPayloadAck:
    session_id: str
    binding_instance_id: str
    payload: CameraSample

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        canonical_text(self.binding_instance_id, "binding_instance_id")
        if not isinstance(self.payload, CameraSample):
            raise TypeError("camera monitor acknowledgement payload must be CameraSample")


@dataclass(frozen=True)
class BoundCameraMonitorPort:
    """Drive authority restricted to one continuous display-only monitor."""

    capability_attestation: VerifiedDeviceCapability
    cleanup_operations: tuple[SafetyOperation, ...]

    def __post_init__(self) -> None:
        admit_bound_capability(
            self.capability_attestation,
            CameraMonitorCapabilitySnapshot,
        )
        if not self.device.session_cleanup_capable:
            raise ValueError("camera monitor requires session-specific cleanup")
        if not any(
            operation in self.device.interrupt_capabilities
            for operation in (SafetyOperation.ABORT, SafetyOperation.DISARM)
        ):
            raise ValueError("camera monitor requires a thread-safe stop interrupt")
        operations = tuple(self.cleanup_operations)
        if len(set(operations)) != len(operations):
            raise ValueError("camera monitor cleanup operations cannot repeat")
        if any(
            operation not in (SafetyOperation.ABORT, SafetyOperation.DISARM)
            for operation in operations
        ):
            raise ValueError("camera monitor cleanup may only ABORT or DISARM")
        if any(operation not in self.device.safety_capabilities for operation in operations):
            raise ValueError("camera monitor cleanup operation is absent from the device")
        object.__setattr__(self, "cleanup_operations", operations)

    @property
    def device(self) -> BoundDevice:
        return self.capability_attestation.device

    @property
    def capability(self) -> CameraMonitorCapabilitySnapshot:
        snapshot = self.capability_attestation.snapshot
        assert isinstance(snapshot, CameraMonitorCapabilitySnapshot)
        return snapshot

    @property
    def resource_claim(self) -> ResourceClaim:
        return ResourceClaim(self.device.key, ClaimMode.EXCLUSIVE)

    @property
    def interrupt_operations(self) -> tuple[SafetyInterrupt, ...]:
        return tuple(
            SafetyInterrupt(self.device.key, operation)
            for operation in (SafetyOperation.ABORT, SafetyOperation.DISARM)
            if operation in self.device.interrupt_capabilities
        )

    def cleanup(self, context: RunContext, session_id: str) -> CleanupReport:
        return cleanup_device_session(
            context.cleanup_device(self.device.key),
            self.cleanup_operations,
            session_id,
            self.capability.max_blocking_call_seconds,
            termination_failure_reason="camera monitor did not stop and join",
            termination_recovery_action="recover the monitor camera before reuse",
            verification_failure_reason="camera monitor safe-state verification failed",
            verification_recovery_action="inspect and recover the monitor camera",
        )

    def verify_idle(self, context: RunContext) -> CleanupReport:
        return verify_cleanup_device_safe_state(
            context.cleanup_device(self.device.key),
            failure_reason="unopened camera monitor is not in a safe state",
            recovery_action="inspect and recover the monitor camera before reuse",
        )


__all__ = [
    "BoundCameraMonitorPort",
    "CameraMonitorCapabilitySnapshot",
    "CameraMonitorInterrupted",
    "CameraMonitorPayloadAck",
    "CameraMonitorPreparedAck",
    "CameraMonitorStartedAck",
    "PrepareCameraMonitorCommand",
    "ReadCameraMonitorCommand",
    "StartCameraMonitorCommand",
]
