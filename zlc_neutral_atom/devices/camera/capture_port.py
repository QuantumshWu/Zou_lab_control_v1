"""Physical camera command protocol and bound device authority."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Protocol, Self, TypeVar, runtime_checkable

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
    capability: CaptureCapabilitySnapshot

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
        capability = self.capability
        if not isinstance(capability, CaptureCapabilitySnapshot):
            raise TypeError("capability must be CaptureCapabilitySnapshot")
        if capability.binding_stamp.binding_instance_id != self.binding_instance_id:
            raise ValueError("configured capability belongs to another binding")
        if capability.settings_fingerprint != self.settings_fingerprint:
            raise ValueError("configured capability settings fingerprint differs")
        if capability.capability_fingerprint != self.capability_fingerprint:
            raise ValueError("configured capability fingerprint differs")
        if not math.isclose(
            capability.camera_physical_facts.exposure_seconds,
            self.applied_exposure_seconds,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("configured capability exposure differs from readback")


@runtime_checkable
class _CameraExposurePort(Protocol):
    @property
    def device(self) -> BoundDevice: ...

    @property
    def capability(self) -> CaptureCapabilitySnapshot: ...

    def with_configured_exposure(
        self,
        acknowledgement: CameraExposureConfiguredAck,
    ) -> Self: ...

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
    settings_fingerprint: str
    capability_fingerprint: str
    capture_spec_fingerprint: str
    expected_total_events: int
    buffer_frame_count: int
    source_ordinal_baseline: int

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_instance_id"):
            _canonical_text(getattr(self, name), name)
        for name in (
            "settings_fingerprint",
            "capability_fingerprint",
            "capture_spec_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "expected_total_events",
            _positive_int(self.expected_total_events, "expected_total_events"),
        )
        object.__setattr__(
            self,
            "buffer_frame_count",
            _positive_int(self.buffer_frame_count, "buffer_frame_count"),
        )
        object.__setattr__(
            self,
            "source_ordinal_baseline",
            _nonnegative_int(
                self.source_ordinal_baseline,
                "source_ordinal_baseline",
            ),
        )
        if self.buffer_frame_count != self.expected_total_events:
            raise ValueError(
                "finite camera arm buffer must cover the exact event cardinality"
            )


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
    _leased_capability: CaptureCapabilitySnapshot | None = field(
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
            CaptureCapabilitySnapshot,
        )
        leased = self._leased_capability
        if leased is None:
            if self._exposure_session_id is not None:
                raise ValueError("exposure session requires a leased capability")
        else:
            _admit_exposure_leased_capability(
                self.capability_attestation.snapshot,
                leased,
                self._exposure_session_id,
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
        snapshot = self._leased_capability
        if snapshot is None:
            snapshot = self.capability_attestation.snapshot
        assert isinstance(snapshot, CaptureCapabilitySnapshot)
        return snapshot

    def with_configured_exposure(
        self,
        acknowledgement: CameraExposureConfiguredAck,
    ) -> "BoundCapturePort":
        """Bind one endpoint-read working point to the active exposure lease."""

        if not isinstance(acknowledgement, CameraExposureConfiguredAck):
            raise TypeError(
                "acknowledgement must be CameraExposureConfiguredAck"
            )
        if acknowledgement.binding_instance_id != self.device.binding_instance_id:
            raise ValueError("exposure acknowledgement belongs to another device")
        return type(self)(
            self.capability_attestation,
            _leased_capability=acknowledgement.capability,
            _exposure_session_id=acknowledgement.session_id,
        )

    def require_current_capability(self) -> CaptureCapabilitySnapshot:
        """Validate the stable binding behind this baseline or leased view."""

        baseline = self.capability_attestation.snapshot
        if self.device.validate_capability(self.capability_attestation) is not baseline:
            raise RuntimeError("capture capability attestation snapshot changed")
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


def _admit_exposure_leased_capability(
    baseline: object,
    leased: CaptureCapabilitySnapshot,
    session_id: str | None,
) -> None:
    """Require a run-scoped exposure view to differ only in read-back timing."""

    if not isinstance(baseline, CaptureCapabilitySnapshot):
        raise TypeError("baseline must be CaptureCapabilitySnapshot")
    if not isinstance(leased, CaptureCapabilitySnapshot):
        raise TypeError("leased must be CaptureCapabilitySnapshot")
    if session_id is None:
        raise ValueError("leased capability requires an exposure session")
    _canonical_text(session_id, "exposure session_id")
    if leased.binding_stamp != baseline.binding_stamp:
        raise ValueError("leased capability binding differs from baseline")
    if leased.payload_contract is not baseline.payload_contract:
        raise ValueError("exposure lease changed the camera payload owner")
    baseline_evidence = baseline.camera_capability_evidence
    leased_evidence = leased.camera_capability_evidence
    if replace(
        leased_evidence,
        physical_facts=baseline_evidence.physical_facts,
    ) != baseline_evidence:
        raise ValueError("exposure lease changed non-physical capability evidence")
    leased_facts = leased_evidence.physical_facts
    expected_facts = replace(
        baseline_evidence.physical_facts,
        exposure_seconds=leased_facts.exposure_seconds,
        required_external_trigger_interval_seconds=(
            leased_facts.required_external_trigger_interval_seconds
        ),
        opaque_frame_settings_fingerprint=(
            leased_facts.opaque_frame_settings_fingerprint
        ),
    )
    if leased_facts != expected_facts:
        raise ValueError("exposure lease changed another camera working-point fact")


def configure_camera_exposure(
    context: RunContext,
    port: _CameraExposurePort,
    session_id: str,
    exposure_seconds: float,
):
    """Apply/read back one run-scoped exposure and return the leased Port view."""

    if not isinstance(context, RunContext):
        raise TypeError("context must be RunContext")
    if not isinstance(port, _CameraExposurePort):
        raise TypeError("port must implement the camera exposure Port contract")
    session_id = _canonical_text(session_id, "exposure session_id")
    exposure_seconds = _positive_finite(exposure_seconds, "exposure_seconds")
    acknowledgement = context.device(port.device.key).execute(
        ConfigureCameraExposureCommand(
            session_id,
            exposure_seconds,
            port.capability.settings_fingerprint,
        )
    )
    if not isinstance(acknowledgement, CameraExposureConfiguredAck):
        raise TypeError(
            "camera exposure configure returned another acknowledgement"
        )
    if acknowledgement.session_id != session_id:
        raise RuntimeError("camera exposure acknowledgement session differs")
    if acknowledgement.binding_instance_id != port.device.binding_instance_id:
        raise RuntimeError("camera exposure acknowledgement binding differs")
    if not math.isclose(
        acknowledgement.applied_exposure_seconds,
        exposure_seconds,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise RuntimeError("camera applied exposure differs from the request")
    configured = port.with_configured_exposure(acknowledgement)
    if not isinstance(configured, type(port)):
        raise TypeError("camera exposure Port changed its concrete authority type")
    return configured



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
    "configure_camera_exposure",
    "PrepareCaptureCommand",
    "ReadCaptureCommand",
    "StartCaptureCommand",
    "capture_terminal_ack_from_tree",
    "capture_terminal_ack_to_tree",
]
