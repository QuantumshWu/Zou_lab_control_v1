"""Physical authority for a hardware-paced camera monitor.

The monitor port is deliberately distinct from finite exact capture authority:
it may publish display-only samples, but it cannot mint an exact reservation or
sealed artifact.  Both ports still use the same composition-owned camera
endpoint, capability evidence, session cleanup, and physical DISARM boundary.

A monitor source may be sensor-clocked (``FREE_RUNNING``) or passively observe
an already external-triggered camera (``EXTERNAL_TRIGGERED``).  The latter does
not grant trigger authority: it only arms the camera and drains records emitted
by independent hardware timing.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraSample,
)
from zlc_storage import (
    canonical_text,
    positive_integer,
    positive_real,
    sha256_text,
)

from .capture_port import CaptureCapabilitySnapshot
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SafetyInterrupt,
    SafetyOperation,
    VerifiedDeviceCapability,
    admit_bound_capability,
    cleanup_device_session,
)
from zlc_neutral_atom.runtime.resources import ResourceClaim
from zlc_neutral_atom.runtime.run import RunContext


class CameraMonitorInterrupted(RuntimeError):
    """An in-flight monitor operation was superseded by an external interrupt."""


@dataclass(frozen=True)
class CameraMonitorCapabilitySnapshot(CaptureCapabilitySnapshot):
    """The same camera capability, with its live acquisition mode attached.

    ``capability_fingerprint`` remains the fingerprint inherited from
    :class:`CaptureCapabilitySnapshot`: capture and live observation are two
    operations on one physical camera, not two device identities.  The mode is
    still checked by the monitor command path, but must not mint a second
    capability digest that a finite capture artifact cannot reconstruct from
    the camera's persisted physical evidence.
    """

    acquisition_mode: CameraAcquisitionMode

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.acquisition_mode, CameraAcquisitionMode):
            raise TypeError("acquisition_mode must be CameraAcquisitionMode")

@dataclass(frozen=True)
class PrepareCameraMonitorCommand:
    session_id: str
    capability_fingerprint: str
    settings_fingerprint: str
    buffer_frame_count: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        sha256_text(self.capability_fingerprint, "capability_fingerprint")
        sha256_text(self.settings_fingerprint, "settings_fingerprint")
        object.__setattr__(
            self,
            "buffer_frame_count",
            positive_integer(self.buffer_frame_count, "buffer_frame_count"),
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
class CameraMonitorNoFrameAck:
    """A passive external-trigger poll completed without a hardware frame."""

    session_id: str
    binding_instance_id: str

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        canonical_text(self.binding_instance_id, "binding_instance_id")


@dataclass(frozen=True)
class BoundCameraMonitorPort:
    """Drive authority restricted to one continuous display-only monitor."""

    capability_attestation: VerifiedDeviceCapability

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
        return ResourceClaim(self.device.key)

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
            session_id,
            self.capability.max_blocking_call_seconds,
        )

    def verify_idle(self, _context: RunContext) -> CleanupReport:
        return CleanupReport.complete()


__all__ = [
    "BoundCameraMonitorPort",
    "CameraMonitorCapabilitySnapshot",
    "CameraMonitorInterrupted",
    "CameraMonitorNoFrameAck",
    "CameraMonitorPayloadAck",
    "CameraMonitorPreparedAck",
    "CameraMonitorStartedAck",
    "PrepareCameraMonitorCommand",
    "ReadCameraMonitorCommand",
    "StartCameraMonitorCommand",
]
