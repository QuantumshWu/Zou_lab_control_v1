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

from dataclasses import dataclass, field

from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraSample,
)
from zlc_storage import (
    canonical_text,
    positive_integer,
    positive_real,
)

from .capture_port import (
    CameraExposureConfiguredAck,
    CaptureCapabilitySnapshot,
    _admit_exposure_leased_capability,
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
from zlc_neutral_atom.runtime.resources import ResourceClaim
from zlc_neutral_atom.runtime.run import RunContext


class CameraMonitorInterrupted(RuntimeError):
    """An in-flight monitor operation was superseded by an external interrupt."""


@dataclass(frozen=True)
class CameraMonitorCapabilitySnapshot(CaptureCapabilitySnapshot):
    """The same typed camera capability with its live mode attached."""

    acquisition_mode: CameraAcquisitionMode

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.acquisition_mode, CameraAcquisitionMode):
            raise TypeError("acquisition_mode must be CameraAcquisitionMode")

@dataclass(frozen=True)
class PrepareCameraMonitorCommand:
    session_id: str
    buffer_frame_count: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
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

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "session_id")
        canonical_text(self.binding_instance_id, "binding_instance_id")


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
    _leased_capability: CameraMonitorCapabilitySnapshot | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _exposure_session_id: str | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        admit_bound_capability(
            self.capability_attestation,
            CameraMonitorCapabilitySnapshot,
        )
        leased = self._leased_capability
        if leased is None:
            if self._exposure_session_id is not None:
                raise ValueError("exposure session requires a leased capability")
        else:
            if not isinstance(leased, CameraMonitorCapabilitySnapshot):
                raise TypeError(
                    "camera monitor lease requires CameraMonitorCapabilitySnapshot"
                )
            _admit_exposure_leased_capability(
                self.capability_attestation.snapshot,
                leased,
                self._exposure_session_id,
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
        snapshot = self._leased_capability
        if snapshot is None:
            snapshot = self.capability_attestation.snapshot
        assert isinstance(snapshot, CameraMonitorCapabilitySnapshot)
        return snapshot

    def with_configured_exposure(
        self,
        acknowledgement: CameraExposureConfiguredAck,
    ) -> "BoundCameraMonitorPort":
        """Bind the endpoint-read monitor working point to this active Run."""

        if not isinstance(acknowledgement, CameraExposureConfiguredAck):
            raise TypeError(
                "acknowledgement must be CameraExposureConfiguredAck"
            )
        capability = acknowledgement.capability
        if not isinstance(capability, CameraMonitorCapabilitySnapshot):
            raise TypeError(
                "camera monitor exposure acknowledgement lost its monitor mode"
            )
        if acknowledgement.binding_instance_id != self.device.binding_instance_id:
            raise ValueError("exposure acknowledgement belongs to another device")
        return type(self)(
            self.capability_attestation,
            _leased_capability=capability,
            _exposure_session_id=acknowledgement.session_id,
        )

    def require_current_capability(self) -> CameraMonitorCapabilitySnapshot:
        """Validate the stable binding behind this baseline or leased view."""

        baseline = self.capability_attestation.snapshot
        if self.device.validate_capability(self.capability_attestation) is not baseline:
            raise RuntimeError("camera monitor capability attestation snapshot changed")
        capability = self.capability
        if self._leased_capability is not None:
            _admit_exposure_leased_capability(
                baseline,
                capability,
                self._exposure_session_id,
            )
        return capability

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
