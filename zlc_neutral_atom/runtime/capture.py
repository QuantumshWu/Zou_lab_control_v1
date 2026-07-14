"""Opaque capture-port/session authority over typed DeviceBroker commands."""

from __future__ import annotations

import hashlib
import math
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypeVar

import numpy as np
from zlc_storage import (
    canonical_digest,
    canonical_text as _canonical_text,
    exact_mapping as _exact_tree,
    nonnegative_integer as _nonnegative_int,
    positive_integer as _positive_int,
    positive_real as _positive_finite,
    sha256_text as _sha256,
)

from zlc_data import (
    AxisId,
    CoordinateFrameId,
    DatasetSchema,
    StreamGenerationId,
)
from zlc_neutral_atom.readout.contracts import (
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    ReadoutBindingKey,
    camera_output_shape_yx,
    normalize_camera_count_dtype,
    normalize_camera_geometry,
    validate_camera_spatial_axes,
)

from ._failure import record_secondary_failure, safe_error_summary
from .dataset import (
    DatasetCellAddress,
    DatasetCellKeyContract,
    DatasetEventAdapter,
    FrozenDatasetEdge,
    OrderedDatasetMetadataHasher,
    SealedDatasetArtifact,
    dataset_cell_key_fingerprint,
)
from .ports import (
    BoundDevice,
    SafetyInterrupt,
    SafetyOperation,
    VerifiedDeviceCapability,
)
from .resources import ClaimMode, HazardClaim, ResourceClaim
from .run import CleanupReport, RunContext
from .streams import (
    AcquisitionProducer,
    AcquisitionStream,
    EndOfStream,
    EventSpanRef,
    ExactConsumerReadiness,
    ExactReservation,
    OrderedEventSpanHasher,
    ProcessorStageProvenance,
    ProducerFlowControl,
    ReservationState,
    SourceFailed,
    StreamError,
    StreamId,
    TraceBinding,
    TraceContext,
)


PayloadT = TypeVar("PayloadT")
_COMPLETION_TOKEN = object()
_PROCESSOR_INPUT_BINDING_TOKEN = object()


@dataclass(frozen=True)
class CameraPhysicalFacts:
    """Typed camera facts minted inside a broker capability probe.

    These facts and the opaque settings fingerprint are frozen from one adapter
    snapshot.  A later binding may name event indices, but it cannot change the
    physical trigger wiring, exposure, gain, readout mode, geometry, identity,
    dtype, or unit attested by this capability.

    ``required_external_trigger_interval_seconds == 0`` means the adapter has
    no additional measured lower bound to assert.  It is not a claim that the
    physical camera has unlimited trigger bandwidth.  A ``None`` integration
    start offset is likewise an explicit unqualified fact; consumers must not
    reinterpret it as zero.
    """

    camera_identity: str
    sensor_identity: str
    optical_path: str
    capture_trigger_channels: tuple[str, ...]
    sensor_shape_yx: tuple[int, int]
    roi_origin_yx: tuple[int, int]
    roi_shape_yx: tuple[int, int]
    binning_yx: tuple[int, int]
    spatial_y_axis_id: AxisId
    spatial_x_axis_id: AxisId
    coordinate_frame: CoordinateFrameId
    dtype: np.dtype
    count_unit: str
    exposure_seconds: float
    required_external_trigger_interval_seconds: float
    external_trigger_integration_start_offset_seconds: float | None
    gain: float
    readout_mode: str
    opaque_frame_settings_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("camera_identity", "sensor_identity", "optical_path"):
            _canonical_text(getattr(self, name), name)
        if isinstance(self.capture_trigger_channels, (str, bytes)):
            raise TypeError("capture_trigger_channels must be a tuple of channel names")
        try:
            trigger_channels = tuple(self.capture_trigger_channels)
        except TypeError as exc:
            raise TypeError(
                "capture_trigger_channels must be a tuple of channel names"
            ) from exc
        trigger_channels = tuple(
            _canonical_text(channel, "capture trigger channel")
            for channel in trigger_channels
        )
        if len(trigger_channels) != len(set(trigger_channels)):
            raise ValueError("capture_trigger_channels must be unique")
        object.__setattr__(
            self,
            "capture_trigger_channels",
            trigger_channels,
        )
        sensor, origin, roi, binning = normalize_camera_geometry(
            sensor_shape_yx=self.sensor_shape_yx,
            roi_origin_yx=self.roi_origin_yx,
            roi_shape_yx=self.roi_shape_yx,
            binning_yx=self.binning_yx,
        )
        object.__setattr__(self, "sensor_shape_yx", sensor)
        object.__setattr__(self, "roi_origin_yx", origin)
        object.__setattr__(self, "roi_shape_yx", roi)
        object.__setattr__(self, "binning_yx", binning)
        validate_camera_spatial_axes(
            self.spatial_y_axis_id,
            self.spatial_x_axis_id,
            self.coordinate_frame,
        )
        dtype = normalize_camera_count_dtype(self.dtype, "dtype")
        object.__setattr__(self, "dtype", dtype)
        _canonical_text(self.count_unit, "count_unit")
        object.__setattr__(
            self,
            "exposure_seconds",
            _positive_finite(self.exposure_seconds, "exposure_seconds"),
        )
        interval = self.required_external_trigger_interval_seconds
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(float(interval))
            or float(interval) < 0
        ):
            raise ValueError(
                "required_external_trigger_interval_seconds must be finite "
                "and non-negative"
            )
        object.__setattr__(
            self,
            "required_external_trigger_interval_seconds",
            float(interval),
        )
        integration_offset = self.external_trigger_integration_start_offset_seconds
        if integration_offset is not None:
            if (
                isinstance(integration_offset, bool)
                or not isinstance(integration_offset, (int, float))
                or not math.isfinite(float(integration_offset))
                or float(integration_offset) < 0
            ):
                raise ValueError(
                    "external_trigger_integration_start_offset_seconds must be "
                    "finite, non-negative, or None"
                )
            integration_offset = float(integration_offset)
        object.__setattr__(
            self,
            "external_trigger_integration_start_offset_seconds",
            integration_offset,
        )
        if (
            isinstance(self.gain, bool)
            or not isinstance(self.gain, (int, float))
            or not math.isfinite(float(self.gain))
            or float(self.gain) < 0
        ):
            raise ValueError("gain must be finite and non-negative")
        object.__setattr__(self, "gain", float(self.gain))
        _canonical_text(self.readout_mode, "readout_mode")
        _sha256(
            self.opaque_frame_settings_fingerprint,
            "opaque_frame_settings_fingerprint",
        )

    @property
    def output_shape_yx(self) -> tuple[int, int]:
        return camera_output_shape_yx(self.roi_shape_yx, self.binning_yx)

    def event_setting(self, event_index: int) -> CameraEventReadoutSetting:
        return CameraEventReadoutSetting(
            _nonnegative_int(event_index, "event_index"),
            self.exposure_seconds,
            self.gain,
            self.readout_mode,
            self.opaque_frame_settings_fingerprint,
        )

    def validate_capture_trigger_channel(self, channel: str) -> None:
        """Require one selected pulse channel to belong to this physical wiring."""

        channel = _canonical_text(channel, "capture trigger channel")
        if channel not in self.capture_trigger_channels:
            raise ValueError(
                f"capture trigger channel {channel!r} is not wired to camera "
                f"{self.camera_identity!r}; attested channels are "
                f"{self.capture_trigger_channels!r}"
            )

    def require_single_capture_trigger_channel(self, channel: str) -> None:
        """Require exact capture to have one, and only one, physical edge source."""

        self.validate_capture_trigger_channel(channel)
        if self.capture_trigger_channels != (channel,):
            raise ValueError(
                "exact triggered capture requires the camera to have exactly one "
                f"attested trigger channel {channel!r}; physical wiring is "
                f"{self.capture_trigger_channels!r}"
            )

    def validate_descriptor(self, descriptor: CameraCaptureDescriptor) -> None:
        if not isinstance(descriptor, CameraCaptureDescriptor):
            raise TypeError("descriptor must be CameraCaptureDescriptor")
        expected = {
            "camera_identity": self.camera_identity,
            "sensor_identity": self.sensor_identity,
            "optical_path": self.optical_path,
            "sensor_shape_yx": self.sensor_shape_yx,
            "roi_origin_yx": self.roi_origin_yx,
            "roi_shape_yx": self.roi_shape_yx,
            "binning_yx": self.binning_yx,
            "spatial_y_axis_id": self.spatial_y_axis_id,
            "spatial_x_axis_id": self.spatial_x_axis_id,
            "coordinate_frame": self.coordinate_frame,
            "dtype": self.dtype,
            "count_unit": self.count_unit,
        }
        mismatches = tuple(
            name for name, value in expected.items() if getattr(descriptor, name) != value
        )
        for setting in descriptor.event_settings:
            if setting != self.event_setting(setting.event_index):
                mismatches += (f"event_settings[{setting.event_index}]",)
        if mismatches:
            raise ValueError(
                "camera descriptor differs from attested physical facts: "
                + ", ".join(mismatches)
            )

    @property
    def fingerprint(self) -> str:
        """Digest of this owner-minted physical-facts value."""

        return canonical_digest(camera_physical_facts_to_tree(self))


_CAMERA_PHYSICAL_FACTS_SCHEMA = "zlc_neutral_atom.CameraPhysicalFacts"
_CAMERA_CAPABILITY_EVIDENCE_SCHEMA = "zlc_neutral_atom.CameraCapabilityEvidence"


def camera_physical_facts_to_tree(value: CameraPhysicalFacts) -> dict[str, object]:
    if not isinstance(value, CameraPhysicalFacts):
        raise TypeError("value must be CameraPhysicalFacts")
    return {
        "schema": _CAMERA_PHYSICAL_FACTS_SCHEMA,
        "camera_identity": value.camera_identity,
        "sensor_identity": value.sensor_identity,
        "optical_path": value.optical_path,
        "capture_trigger_channels": list(value.capture_trigger_channels),
        "sensor_shape_yx": list(value.sensor_shape_yx),
        "roi_origin_yx": list(value.roi_origin_yx),
        "roi_shape_yx": list(value.roi_shape_yx),
        "binning_yx": list(value.binning_yx),
        "spatial_y_axis_id": value.spatial_y_axis_id.value,
        "spatial_x_axis_id": value.spatial_x_axis_id.value,
        "coordinate_frame": value.coordinate_frame.value,
        "dtype": value.dtype.str,
        "count_unit": value.count_unit,
        "exposure_seconds": value.exposure_seconds,
        "required_external_trigger_interval_seconds": (
            value.required_external_trigger_interval_seconds
        ),
        "external_trigger_integration_start_offset_seconds": (
            value.external_trigger_integration_start_offset_seconds
        ),
        "gain": value.gain,
        "readout_mode": value.readout_mode,
        "opaque_frame_settings_fingerprint": (
            value.opaque_frame_settings_fingerprint
        ),
    }


def camera_physical_facts_from_tree(tree: object) -> CameraPhysicalFacts:
    fields = {
        "schema",
        "camera_identity",
        "sensor_identity",
        "optical_path",
        "capture_trigger_channels",
        "sensor_shape_yx",
        "roi_origin_yx",
        "roi_shape_yx",
        "binning_yx",
        "spatial_y_axis_id",
        "spatial_x_axis_id",
        "coordinate_frame",
        "dtype",
        "count_unit",
        "exposure_seconds",
        "required_external_trigger_interval_seconds",
        "external_trigger_integration_start_offset_seconds",
        "gain",
        "readout_mode",
        "opaque_frame_settings_fingerprint",
    }
    data = _exact_tree(tree, fields, _CAMERA_PHYSICAL_FACTS_SCHEMA)
    return CameraPhysicalFacts(
        camera_identity=data["camera_identity"],
        sensor_identity=data["sensor_identity"],
        optical_path=data["optical_path"],
        capture_trigger_channels=data["capture_trigger_channels"],
        sensor_shape_yx=data["sensor_shape_yx"],
        roi_origin_yx=data["roi_origin_yx"],
        roi_shape_yx=data["roi_shape_yx"],
        binning_yx=data["binning_yx"],
        spatial_y_axis_id=AxisId(data["spatial_y_axis_id"]),
        spatial_x_axis_id=AxisId(data["spatial_x_axis_id"]),
        coordinate_frame=CoordinateFrameId(data["coordinate_frame"]),
        dtype=np.dtype(data["dtype"]),
        count_unit=data["count_unit"],
        exposure_seconds=data["exposure_seconds"],
        required_external_trigger_interval_seconds=data[
            "required_external_trigger_interval_seconds"
        ],
        external_trigger_integration_start_offset_seconds=data[
            "external_trigger_integration_start_offset_seconds"
        ],
        gain=data["gain"],
        readout_mode=data["readout_mode"],
        opaque_frame_settings_fingerprint=data[
            "opaque_frame_settings_fingerprint"
        ],
    )


@dataclass(frozen=True)
class CameraCapabilityEvidence:
    """Canonical facts whose digest is the broker capability fingerprint.

    This value is minted from the same frozen adapter snapshot as
    :class:`CameraPhysicalFacts`.  Persisting it lets the durable boundary
    recompute the terminal capability fingerprint instead of trusting a second
    caller-provided digest.
    """

    adapter_type: str
    source_id: str
    settings_fingerprint: str
    payload_contract_fingerprint: str
    capture_spec_owner_fingerprint: str
    flow_control: ProducerFlowControl
    max_source_burst_events: int
    driver_ring_bytes: int
    adapter_record_retention_bytes: int
    max_blocking_call_seconds: float
    max_capture_spec_bytes: int
    physical_facts: CameraPhysicalFacts

    def __post_init__(self) -> None:
        _canonical_text(self.adapter_type, "adapter_type")
        _canonical_text(self.source_id, "source_id")
        for name in (
            "settings_fingerprint",
            "payload_contract_fingerprint",
            "capture_spec_owner_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        if not isinstance(self.flow_control, ProducerFlowControl):
            raise TypeError("flow_control must be ProducerFlowControl")
        object.__setattr__(
            self,
            "max_source_burst_events",
            _positive_int(self.max_source_burst_events, "max_source_burst_events"),
        )
        object.__setattr__(
            self,
            "driver_ring_bytes",
            _positive_int(self.driver_ring_bytes, "driver_ring_bytes"),
        )
        object.__setattr__(
            self,
            "adapter_record_retention_bytes",
            _nonnegative_int(
                self.adapter_record_retention_bytes,
                "adapter_record_retention_bytes",
            ),
        )
        object.__setattr__(
            self,
            "max_blocking_call_seconds",
            _positive_finite(
                self.max_blocking_call_seconds,
                "max_blocking_call_seconds",
            ),
        )
        object.__setattr__(
            self,
            "max_capture_spec_bytes",
            _positive_int(self.max_capture_spec_bytes, "max_capture_spec_bytes"),
        )
        if not isinstance(self.physical_facts, CameraPhysicalFacts):
            raise TypeError("physical_facts must be CameraPhysicalFacts")
        if (
            self.physical_facts.opaque_frame_settings_fingerprint
            != self.settings_fingerprint
        ):
            raise ValueError("camera physical facts differ from capability settings")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(camera_capability_evidence_to_tree(self))


def camera_capability_evidence_to_tree(
    value: CameraCapabilityEvidence,
) -> dict[str, object]:
    if not isinstance(value, CameraCapabilityEvidence):
        raise TypeError("value must be CameraCapabilityEvidence")
    return {
        "schema": _CAMERA_CAPABILITY_EVIDENCE_SCHEMA,
        "adapter_type": value.adapter_type,
        "source_id": value.source_id,
        "settings_fingerprint": value.settings_fingerprint,
        "payload_contract_fingerprint": value.payload_contract_fingerprint,
        "capture_spec_owner_fingerprint": value.capture_spec_owner_fingerprint,
        "flow_control": value.flow_control.value,
        "max_source_burst_events": value.max_source_burst_events,
        "driver_ring_bytes": value.driver_ring_bytes,
        "adapter_record_retention_bytes": value.adapter_record_retention_bytes,
        "max_blocking_call_seconds": value.max_blocking_call_seconds,
        "max_capture_spec_bytes": value.max_capture_spec_bytes,
        "physical_facts_fingerprint": value.physical_facts.fingerprint,
        "physical_facts": camera_physical_facts_to_tree(value.physical_facts),
    }


def camera_capability_evidence_from_tree(tree: object) -> CameraCapabilityEvidence:
    fields = {
        "schema",
        "adapter_type",
        "source_id",
        "settings_fingerprint",
        "payload_contract_fingerprint",
        "capture_spec_owner_fingerprint",
        "flow_control",
        "max_source_burst_events",
        "driver_ring_bytes",
        "adapter_record_retention_bytes",
        "max_blocking_call_seconds",
        "max_capture_spec_bytes",
        "physical_facts_fingerprint",
        "physical_facts",
    }
    data = _exact_tree(tree, fields, _CAMERA_CAPABILITY_EVIDENCE_SCHEMA)
    facts = camera_physical_facts_from_tree(data["physical_facts"])
    if facts.fingerprint != _sha256(
        data["physical_facts_fingerprint"],
        "physical_facts_fingerprint",
    ):
        raise ValueError("camera physical-facts fingerprint differs from evidence")
    return CameraCapabilityEvidence(
        adapter_type=data["adapter_type"],
        source_id=data["source_id"],
        settings_fingerprint=data["settings_fingerprint"],
        payload_contract_fingerprint=data["payload_contract_fingerprint"],
        capture_spec_owner_fingerprint=data["capture_spec_owner_fingerprint"],
        flow_control=ProducerFlowControl(data["flow_control"]),
        max_source_burst_events=data["max_source_burst_events"],
        driver_ring_bytes=data["driver_ring_bytes"],
        adapter_record_retention_bytes=data["adapter_record_retention_bytes"],
        max_blocking_call_seconds=data["max_blocking_call_seconds"],
        max_capture_spec_bytes=data["max_capture_spec_bytes"],
        physical_facts=facts,
    )


class CapturePayloadContract(Protocol[PayloadT]):
    fingerprint: str
    max_retained_nbytes: int

    def snapshot(self, payload: PayloadT) -> PayloadT: ...

    def retained_nbytes(self, payload: PayloadT) -> int: ...

    def digest(self, payload: PayloadT) -> str: ...

    def source_ordinal(self, payload: PayloadT) -> int: ...

    def captured_at(self, payload: PayloadT) -> float: ...

    def correlation_id(self, payload: PayloadT) -> str: ...


@dataclass(frozen=True)
class FrozenCaptureSpec:
    """Canonical owner bytes; runtime never executes an arbitrary spec codec."""

    owner_fingerprint: str
    payload: bytes
    digest: str = ""

    def __post_init__(self) -> None:
        _sha256(self.owner_fingerprint, "capture spec owner fingerprint")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("capture spec payload must be non-empty immutable bytes")
        payload = bytes(self.payload)
        digest = hashlib.sha256(payload).hexdigest()
        if self.digest and self.digest != digest:
            raise ValueError("capture spec digest differs from canonical payload")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "digest", digest)


@dataclass(frozen=True)
class CaptureCapabilitySnapshot:
    binding_id: str
    stable_device_identity: str
    connection_generation: str
    capability_fingerprint: str
    settings_fingerprint: str
    payload_contract_fingerprint: str
    capture_spec_owner_fingerprint: str
    flow_control: ProducerFlowControl
    max_source_burst_events: int
    driver_ring_bytes: int
    adapter_record_retention_bytes: int
    max_blocking_call_seconds: float
    max_capture_spec_bytes: int
    camera_capability_evidence: CameraCapabilityEvidence | None = None

    def __post_init__(self) -> None:
        for field in (
            "binding_id",
            "stable_device_identity",
            "connection_generation",
        ):
            _canonical_text(getattr(self, field), field)
        for field in (
            "capability_fingerprint",
            "settings_fingerprint",
            "payload_contract_fingerprint",
            "capture_spec_owner_fingerprint",
        ):
            _sha256(getattr(self, field), field)
        if not isinstance(self.flow_control, ProducerFlowControl):
            raise TypeError("flow_control must be ProducerFlowControl")
        object.__setattr__(
            self,
            "max_source_burst_events",
            _positive_int(self.max_source_burst_events, "max_source_burst_events"),
        )
        object.__setattr__(
            self,
            "driver_ring_bytes",
            _positive_int(self.driver_ring_bytes, "driver_ring_bytes"),
        )
        object.__setattr__(
            self,
            "adapter_record_retention_bytes",
            _nonnegative_int(
                self.adapter_record_retention_bytes,
                "adapter_record_retention_bytes",
            ),
        )
        object.__setattr__(
            self,
            "max_capture_spec_bytes",
            _positive_int(self.max_capture_spec_bytes, "max_capture_spec_bytes"),
        )
        object.__setattr__(
            self,
            "max_blocking_call_seconds",
            _positive_finite(
                self.max_blocking_call_seconds,
                "max_blocking_call_seconds",
            ),
        )
        evidence = self.camera_capability_evidence
        if evidence is not None:
            if not isinstance(evidence, CameraCapabilityEvidence):
                raise TypeError(
                    "camera_capability_evidence must be CameraCapabilityEvidence or None"
                )
            if evidence.fingerprint != self.capability_fingerprint:
                raise ValueError(
                    "camera capability fingerprint differs from canonical evidence"
                )
            if evidence.physical_facts.camera_identity != self.stable_device_identity:
                raise ValueError(
                    "camera physical identity differs from capability stable identity"
                )
            for snapshot_field, evidence_field in (
                ("settings_fingerprint", "settings_fingerprint"),
                ("payload_contract_fingerprint", "payload_contract_fingerprint"),
                ("capture_spec_owner_fingerprint", "capture_spec_owner_fingerprint"),
                ("flow_control", "flow_control"),
                ("max_source_burst_events", "max_source_burst_events"),
                ("driver_ring_bytes", "driver_ring_bytes"),
                (
                    "adapter_record_retention_bytes",
                    "adapter_record_retention_bytes",
                ),
                ("max_blocking_call_seconds", "max_blocking_call_seconds"),
                ("max_capture_spec_bytes", "max_capture_spec_bytes"),
            ):
                if getattr(self, snapshot_field) != getattr(evidence, evidence_field):
                    raise ValueError(
                        f"camera capability {snapshot_field} differs from evidence"
                    )

    @property
    def camera_physical_facts(self) -> CameraPhysicalFacts | None:
        evidence = self.camera_capability_evidence
        return None if evidence is None else evidence.physical_facts


@dataclass(frozen=True)
class CaptureRuntimeProfile:
    required_consumer_lag_events: int
    transport_memory_limit_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_consumer_lag_events",
            _nonnegative_int(
                self.required_consumer_lag_events,
                "required_consumer_lag_events",
            ),
        )
        object.__setattr__(
            self,
            "transport_memory_limit_bytes",
            _positive_int(
                self.transport_memory_limit_bytes,
                "transport_memory_limit_bytes",
            ),
        )


@dataclass(frozen=True)
class CameraCaptureProvenance:
    """Owner-derived physical facts for one raw camera capture binding.

    No readout event is selected here.  A later calibration/analysis request
    combines this raw descriptor with an explicit ``CalibrationCaptureLayout``
    to derive a ``FrameContract``.  ``camera_arm_spec_fingerprint`` is only the
    frozen camera arm/capture-spec digest; it is not an FPGA pulse-schedule digest.
    """

    descriptor: CameraCaptureDescriptor
    binding: ReadoutBindingKey
    active_settings_fingerprint: str
    binding_id: str
    connection_generation: str
    capability_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CameraCaptureDescriptor):
            raise TypeError("descriptor must be CameraCaptureDescriptor")
        if not isinstance(self.binding, ReadoutBindingKey):
            raise TypeError("binding must be ReadoutBindingKey")
        _sha256(
            self.active_settings_fingerprint,
            "active_settings_fingerprint",
        )
        if any(
            setting.opaque_frame_settings_fingerprint
            != self.active_settings_fingerprint
            for setting in self.descriptor.event_settings
        ):
            raise ValueError(
                "descriptor event settings differ from active settings evidence"
            )
        if self.descriptor.camera_arm_spec_fingerprint is None:
            raise ValueError(
                "camera descriptor requires camera_arm_spec_fingerprint"
            )
        _canonical_text(self.binding_id, "binding_id")
        _canonical_text(self.connection_generation, "connection_generation")
        _sha256(self.capability_fingerprint, "capability_fingerprint")

    @property
    def camera_arm_spec_fingerprint(self) -> str:
        value = self.descriptor.camera_arm_spec_fingerprint
        assert value is not None
        return value

    def validate_schema(self, schema: DatasetSchema) -> None:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        self.descriptor.validate_schema(schema)


@dataclass(frozen=True)
class CaptureStreamContract:
    stream_id: StreamId
    source_id: str
    dataset_schema: DatasetSchema
    payload_contract: CapturePayloadContract
    event_adapter: DatasetEventAdapter
    expected_cells: tuple[DatasetCellAddress, ...]
    capability: CaptureCapabilitySnapshot
    runtime_profile: CaptureRuntimeProfile
    capture_spec_owner_fingerprint: str
    camera_provenance: CameraCaptureProvenance | None = None
    dataset_edge: FrozenDatasetEdge = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        _canonical_text(self.source_id, "source_id")
        if not isinstance(self.dataset_schema, DatasetSchema):
            raise TypeError("dataset_schema must be DatasetSchema")
        if not isinstance(self.capability, CaptureCapabilitySnapshot):
            raise TypeError("capability must be CaptureCapabilitySnapshot")
        if not isinstance(self.runtime_profile, CaptureRuntimeProfile):
            raise TypeError("runtime_profile must be CaptureRuntimeProfile")
        edge = FrozenDatasetEdge(
            self.dataset_schema,
            self.event_adapter,
            tuple(self.expected_cells),
        )
        object.__setattr__(self, "expected_cells", edge.expected_cells)
        object.__setattr__(self, "dataset_edge", edge)
        object.__setattr__(self, "event_adapter", edge.event_adapter)
        if edge.source_payload_contract is not self.payload_contract:
            raise ValueError("DatasetEventAdapter must share CapturePayloadContract owner")
        object.__setattr__(self, "payload_contract", edge.payload_contract)
        if edge.value_schema is not self.dataset_schema.cell_schema:
            raise ValueError("DatasetEventAdapter schema differs from DatasetSchema")
        _sha256(edge.payload_contract_fingerprint, "payload contract fingerprint")
        _sha256(
            self.capture_spec_owner_fingerprint,
            "capture spec owner fingerprint",
        )
        for member in (
            "snapshot",
            "retained_nbytes",
            "digest",
            "source_ordinal",
            "captured_at",
            "correlation_id",
        ):
            if not callable(getattr(self.payload_contract, member, None)):
                raise TypeError(f"payload_contract.{member} must be callable")
        _positive_int(
            edge.payload_max_retained_nbytes,
            "payload contract max_retained_nbytes",
        )
        metadata_contract = edge.metadata_contract
        _sha256(edge.metadata_contract_fingerprint, "metadata contract fingerprint")
        _sha256(edge.operator_fingerprint, "event adapter operator fingerprint")
        _nonnegative_int(
            edge.metadata_max_retained_nbytes,
            "metadata contract max_retained_nbytes",
        )
        for member in ("snapshot", "validate", "retained_nbytes", "digest"):
            if not callable(getattr(metadata_contract, member, None)):
                raise TypeError(f"metadata_contract.{member} must be callable")
        if self.capability.payload_contract_fingerprint != edge.payload_contract_fingerprint:
            raise ValueError("capture capability and payload contract fingerprints differ")
        if (
            self.capability.capture_spec_owner_fingerprint
            != self.capture_spec_owner_fingerprint
        ):
            raise ValueError("capture capability and spec owner fingerprints differ")
        if self.camera_provenance is not None:
            # A raw camera CaptureArtifact is defined by the acquisition owner,
            # not by schema equality.  A caller-defined adapter may preserve the
            # same schema while numerically changing every pixel, so require the
            # exact owner implementation as well as its canonical operator id.
            from zlc_neutral_atom.acquisition.camera import CameraDatasetEventAdapter
            from zlc_neutral_atom.camera_operator import (
                CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT,
            )

            if type(self.event_adapter) is not CameraDatasetEventAdapter or (
                edge.operator_fingerprint != CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT
            ):
                raise ValueError(
                    "raw camera provenance requires the owner identity event adapter"
                )
            if not isinstance(self.camera_provenance, CameraCaptureProvenance):
                raise TypeError(
                    "camera_provenance must be CameraCaptureProvenance or None"
                )
            self.camera_provenance.validate_schema(self.dataset_schema)
            if self.camera_provenance.binding.value != self.source_id:
                raise ValueError(
                    "camera provenance binding differs from capture source_id"
                )
            if (
                self.camera_provenance.active_settings_fingerprint
                != self.capability.settings_fingerprint
            ):
                raise ValueError(
                    "camera readout active settings differ from capture capability"
                )
            if (
                self.camera_provenance.descriptor.camera_identity
                != self.capability.stable_device_identity
            ):
                raise ValueError(
                    "camera readout identity differs from capture capability"
                )
            if self.camera_provenance.binding_id != self.capability.binding_id:
                raise ValueError("camera provenance binding_id differs from capability")
            if (
                self.camera_provenance.connection_generation
                != self.capability.connection_generation
            ):
                raise ValueError(
                    "camera provenance connection generation differs from capability"
                )
            if (
                self.camera_provenance.capability_fingerprint
                != self.capability.capability_fingerprint
            ):
                raise ValueError(
                    "camera provenance capability fingerprint differs from capability"
                )
            physical_facts = self.capability.camera_physical_facts
            if physical_facts is None:
                raise ValueError(
                    "camera provenance requires broker-attested physical facts"
                )
            physical_facts.validate_descriptor(self.camera_provenance.descriptor)
            evidence = self.capability.camera_capability_evidence
            assert evidence is not None
            if evidence.source_id != self.source_id:
                raise ValueError(
                    "camera capability evidence source differs from capture source_id"
                )
        if self.estimated_transport_bytes > self.runtime_profile.transport_memory_limit_bytes:
            raise MemoryError(
                f"capture transport budget {self.estimated_transport_bytes} exceeds "
                f"limit {self.runtime_profile.transport_memory_limit_bytes}"
            )

    @property
    def total_events(self) -> int:
        return len(self.expected_cells)

    @property
    def max_inflight_events(self) -> int:
        return min(
            self.total_events,
            self.capability.max_source_burst_events
            + self.runtime_profile.required_consumer_lag_events,
        )

    @property
    def max_inflight_bytes(self) -> int:
        return self.max_inflight_events * self.dataset_edge.payload_max_retained_nbytes

    @property
    def estimated_transport_bytes(self) -> int:
        return (
            self.capability.driver_ring_bytes
            + self.capability.adapter_record_retention_bytes
            + self.max_inflight_bytes
            + self.dataset_edge.payload_max_retained_nbytes
            + self.dataset_edge.metadata_max_retained_nbytes
            + self.capability.max_capture_spec_bytes
        )


@dataclass(frozen=True)
class PrepareCaptureCommand:
    session_id: str
    run_id: str
    source_id: str
    capture_spec_payload: bytes
    capture_spec_owner_fingerprint: str
    capture_spec_fingerprint: str
    capability_fingerprint: str
    settings_fingerprint: str
    expected_total_events: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        for name in ("session_id", "run_id", "source_id"):
            _canonical_text(getattr(self, name), name)
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
    binding_id: str
    connection_generation: str
    settings_fingerprint: str
    capability_fingerprint: str
    capture_spec_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_id", "connection_generation"):
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
    binding_id: str
    connection_generation: str

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_id", "connection_generation"):
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
    binding_id: str
    connection_generation: str
    payload: object

    def __post_init__(self) -> None:
        for name in ("session_id", "binding_id", "connection_generation"):
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
    binding_id: str
    connection_generation: str
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
        for name in ("session_id", "binding_id", "connection_generation"):
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


class CaptureSessionState(str, Enum):
    NEW = "NEW"
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    STARTING = "STARTING"
    STARTED = "STARTED"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CaptureCompletion:
    __slots__ = (
        "_session",
        "_eos",
        "_terminal",
        "_trace_binding",
        "_source_dataset_schema",
        "_source_cell_schedule",
        "_camera_provenance",
        "_camera_capability_evidence",
        "_camera_arm_spec",
        "_source_event_adapter_operator_fingerprint",
        "_chain_contract_digest",
        "_source_event_span",
        "_processor_stages",
        "_terminal_reservation",
        "_direct_terminal_consumer",
    )

    def __init__(
        self,
        authority: object,
        *,
        session: "CaptureSession",
        terminal: CaptureTerminalAck,
    ) -> None:
        if authority is not _COMPLETION_TOKEN:
            raise PermissionError("CaptureCompletion can only be minted by CaptureSession")
        readiness = session._exact_consumer_readiness
        if readiness is None:
            raise RuntimeError("capture completion has no exact consumer readiness")
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_eos", None)
        object.__setattr__(self, "_terminal", terminal)
        object.__setattr__(self, "_trace_binding", session._trace_binding)
        object.__setattr__(
            self,
            "_source_dataset_schema",
            session._contract.dataset_schema,
        )
        object.__setattr__(
            self,
            "_source_cell_schedule",
            session._contract.expected_cells,
        )
        object.__setattr__(
            self,
            "_camera_provenance",
            session._contract.camera_provenance,
        )
        object.__setattr__(
            self,
            "_camera_capability_evidence",
            session._contract.capability.camera_capability_evidence,
        )
        object.__setattr__(self, "_camera_arm_spec", session._capture_spec)
        object.__setattr__(
            self,
            "_source_event_adapter_operator_fingerprint",
            session._contract.dataset_edge.operator_fingerprint,
        )
        object.__setattr__(
            self,
            "_chain_contract_digest",
            readiness.chain_contract_digest,
        )
        source_reservation = readiness._source_reservation
        if source_reservation._stream is not session._stream:
            raise RuntimeError(
                "capture completion readiness belongs to another source stream"
            )
        source_event_span_hasher = session._source_event_span_hasher
        if source_event_span_hasher is None:
            raise RuntimeError("capture completion has no source event span owner")
        source_event_span = source_event_span_hasher.seal(
            source_reservation.end_sequence
        )
        if (
            source_event_span.stream_id != session._stream.stream_id
            or source_event_span.generation != session._stream.generation
            or source_event_span.start_sequence != source_reservation.start_sequence
            or source_event_span.end_sequence != source_reservation.end_sequence
        ):
            raise RuntimeError("capture source event span differs from its reservation")
        object.__setattr__(self, "_source_event_span", source_event_span)
        object.__setattr__(self, "_processor_stages", readiness.processor_stages)
        # Process-local identity capability.  PipelineResult uses it to prove
        # that its SealedDatasetArtifact came from this readiness chain's live
        # terminal DatasetBuilder, including through processor chains.
        object.__setattr__(
            self,
            "_terminal_reservation",
            readiness._terminal_reservation,
        )
        direct_terminal = (
            readiness._source_reservation is readiness._terminal_reservation
        )
        if direct_terminal != (not readiness.processor_stages):
            raise RuntimeError(
                "capture readiness terminal identity differs from its stage chain"
            )
        object.__setattr__(self, "_direct_terminal_consumer", direct_terminal)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CaptureCompletion is immutable")

    @property
    def eos(self) -> EndOfStream:
        if self._eos is None:
            raise RuntimeError("capture completion has not been committed")
        return self._eos

    @property
    def terminal(self) -> CaptureTerminalAck:
        return self._terminal

    @property
    def trace_binding(self) -> TraceBinding:
        return self._trace_binding

    @property
    def source_dataset_schema(self) -> DatasetSchema:
        return self._source_dataset_schema

    @property
    def source_cell_schedule(self) -> tuple[DatasetCellAddress, ...]:
        return self._source_cell_schedule

    @property
    def camera_provenance(self) -> CameraCaptureProvenance | None:
        return self._camera_provenance

    @property
    def camera_capability_evidence(self) -> CameraCapabilityEvidence | None:
        return self._camera_capability_evidence

    @property
    def camera_arm_spec(self) -> FrozenCaptureSpec:
        return self._camera_arm_spec

    @property
    def source_event_adapter_operator_fingerprint(self) -> str:
        return self._source_event_adapter_operator_fingerprint

    @property
    def chain_contract_digest(self) -> str:
        return self._chain_contract_digest

    @property
    def direct_terminal_consumer(self) -> bool:
        return self._direct_terminal_consumer

    @property
    def source_stream_id(self) -> StreamId:
        return self._source_event_span.stream_id

    @property
    def source_generation(self) -> StreamGenerationId:
        return self._source_event_span.generation

    @property
    def source_start_sequence(self) -> int:
        return self._source_event_span.start_sequence

    @property
    def source_end_sequence(self) -> int:
        return self._source_event_span.end_sequence

    @property
    def source_event_span(self) -> EventSpanRef:
        return self._source_event_span

    @property
    def processor_stages(self) -> tuple[ProcessorStageProvenance, ...]:
        return self._processor_stages

    def _commit_pipeline_result(self, dataset: SealedDatasetArtifact) -> None:
        """Atomically consume dataset and completion live authority."""

        session = self._session
        if session is None:
            raise RuntimeError("capture completion authority was already consumed")
        session._commit_pipeline_authority(self, dataset)

    def _validate_pipeline_authority(
        self,
    ) -> tuple["CaptureSession", ExactReservation]:
        """Validate the final no-fail authority commit before any owner mutates."""

        session = self._session
        reservation = self._terminal_reservation
        if session is None or reservation is None:
            raise RuntimeError("capture completion authority was already consumed")
        session._assert_owner_thread()
        if not session.owns_completion(self):
            raise RuntimeError("capture completion authority is absent or differs")
        return session, reservation


@dataclass(frozen=True)
class BoundCapturePort:
    capability_attestation: VerifiedDeviceCapability
    cleanup_operations: tuple[SafetyOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capability_attestation, VerifiedDeviceCapability):
            raise TypeError("capability_attestation must be broker-minted")
        if not isinstance(self.capability_attestation.device, BoundDevice):
            raise TypeError("capture capability has no BoundDevice")
        if not isinstance(
            self.capability_attestation.snapshot,
            CaptureCapabilitySnapshot,
        ):
            raise TypeError("capture capability attestation has the wrong snapshot type")
        if (
            self.capability_attestation.device.validate_capability(
                self.capability_attestation
            )
            is not self.capability_attestation.snapshot
        ):
            raise RuntimeError("capture capability attestation snapshot changed")
        if self.capability.binding_id != self.device.binding_id:
            raise ValueError("capture capability binding_id differs from BoundDevice")
        if self.capability.stable_device_identity != self.device.stable_device_identity:
            raise ValueError("capture capability stable identity differs from BoundDevice")
        if self.capability.connection_generation != self.device.connection_generation:
            raise ValueError("capture capability generation differs from BoundDevice")
        if not self.device.session_cleanup_capable:
            raise ValueError("capture port requires session-specific cleanup capability")
        if not any(
            operation in self.device.interrupt_capabilities
            for operation in (SafetyOperation.ABORT, SafetyOperation.DISARM)
        ):
            raise ValueError("capture port requires a thread-safe ABORT or DISARM interrupt")
        operations = tuple(self.cleanup_operations)
        if len(set(operations)) != len(operations):
            raise ValueError("capture cleanup operations cannot contain duplicates")
        if any(
            operation not in (SafetyOperation.ABORT, SafetyOperation.DISARM)
            for operation in operations
        ):
            raise ValueError("capture cleanup pre-steps may only ABORT or DISARM")
        if any(operation not in self.device.safety_capabilities for operation in operations):
            raise ValueError("capture cleanup operation is absent from BoundDevice")
        object.__setattr__(self, "cleanup_operations", operations)

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
        return ResourceClaim(self.device.key, ClaimMode.EXCLUSIVE)

    @property
    def hazard_claim(self) -> HazardClaim:
        return HazardClaim(
            self.device.key,
            self.device.stable_device_identity,
            self.device.connection_generation,
        )

    @property
    def interrupt_operations(self) -> tuple[SafetyInterrupt, ...]:
        preferred = tuple(
            operation
            for operation in (SafetyOperation.ABORT, SafetyOperation.DISARM)
            if operation in self.device.interrupt_capabilities
        )
        return tuple(SafetyInterrupt(self.device.key, operation) for operation in preferred)

    def open_session(
        self,
        contract: CaptureStreamContract,
        trace_binding: TraceBinding,
        capture_spec: FrozenCaptureSpec,
    ) -> "CaptureSession":
        if contract.capability is not self.capability:
            raise ValueError("CaptureStreamContract must share BoundCapturePort capability")
        return CaptureSession(self, contract, trace_binding, capture_spec)

    def cleanup(self, context: RunContext, session_id: str) -> CleanupReport:
        errors: list[BaseException] = []
        device = context.cleanup_device(self.device.key)
        for operation in self.cleanup_operations:
            try:
                device.perform(operation)
            except BaseException as error:
                errors.append(error)
        try:
            closed = device.close_session(
                session_id,
                self.capability.max_blocking_call_seconds,
            )
            if not (closed.source_stopped and closed.no_more_work and closed.joined):
                raise RuntimeError("capture session stop/drain/join acknowledgement failed")
        except BaseException as error:
            errors.append(error)
        if errors:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="capture cleanup did not acknowledge every required operation",
                recovery_action="verify camera session termination and physical safe state",
                errors=tuple(errors),
            )
        try:
            proof = device.verify_safe_state()
        except BaseException as error:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="capture device safe-state verification failed",
                recovery_action="inspect and recover the camera before reuse",
                errors=(error,),
            )
        return CleanupReport.safe((proof,))

    def verify_idle(self, context: RunContext) -> CleanupReport:
        """Verify safety when no physical capture prepare was ever attempted."""

        try:
            proof = context.cleanup_device(self.device.key).verify_safe_state()
        except BaseException as error:
            return CleanupReport.unsafe(
                (self.device.key,),
                reason="unopened capture device safe-state verification failed",
                recovery_action="inspect and recover the camera before reuse",
                errors=(error,),
            )
        return CleanupReport.safe((proof,))


@dataclass
class _CaptureProcessorInputAuthority:
    capture_contract: CaptureStreamContract
    stream: AcquisitionStream
    payload_contract: CapturePayloadContract
    join_key_contract: DatasetCellKeyContract
    input_edge: FrozenDatasetEdge
    stream_generation: StreamGenerationId
    session_reservation: ExactReservation | None = None


_CAPTURE_PROCESSOR_INPUT_AUTHORITIES = weakref.WeakKeyDictionary()
_CAPTURE_PROCESSOR_INPUT_AUTHORITIES_LOCK = threading.RLock()


class CaptureProcessorInputBinding:
    """Sealed process-local input authority shared with one capture session.

    The binding exposes the exact owners already frozen by ``CaptureSession``;
    it never reconstructs an equivalent payload or join-key contract.  That
    identity is required by exact stream processors, while the redundant
    fingerprint checks fail closed if a future construction path wires
    semantically different contracts together.
    """

    __slots__ = (
        "_authority",
        "_capture_contract",
        "_stream",
        "_payload_contract",
        "_join_key_contract",
        "_input_edge",
        "_stream_generation",
        "__weakref__",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("CaptureProcessorInputBinding is sealed")

    def __init__(
        self,
        authority: object,
        *,
        capture_contract: CaptureStreamContract,
        stream: AcquisitionStream,
    ) -> None:
        if authority is not _PROCESSOR_INPUT_BINDING_TOKEN:
            raise PermissionError(
                "CaptureProcessorInputBinding can only be minted by CaptureSession"
            )
        if not isinstance(capture_contract, CaptureStreamContract):
            raise TypeError("capture_contract must be CaptureStreamContract")
        if not isinstance(stream, AcquisitionStream):
            raise TypeError("stream must be AcquisitionStream")
        input_edge = capture_contract.dataset_edge
        if not isinstance(input_edge, FrozenDatasetEdge):
            raise TypeError("capture contract has no FrozenDatasetEdge")
        payload_contract = capture_contract.payload_contract
        join_key_contract = stream._join_key_contract
        if not isinstance(join_key_contract, DatasetCellKeyContract):
            raise TypeError("capture stream has no DatasetCellKeyContract owner")
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_capture_contract", capture_contract)
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_payload_contract", payload_contract)
        object.__setattr__(self, "_join_key_contract", join_key_contract)
        object.__setattr__(self, "_input_edge", input_edge)
        object.__setattr__(self, "_stream_generation", stream.generation)
        with _CAPTURE_PROCESSOR_INPUT_AUTHORITIES_LOCK:
            _CAPTURE_PROCESSOR_INPUT_AUTHORITIES[self] = (
                _CaptureProcessorInputAuthority(
                    capture_contract,
                    stream,
                    payload_contract,
                    join_key_contract,
                    input_edge,
                    stream.generation,
                )
            )
        self._validate_identity()

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CaptureProcessorInputBinding is immutable")

    def __reduce__(self):
        raise TypeError("CaptureProcessorInputBinding is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("CaptureProcessorInputBinding is process-local")

    def _require_authority(self) -> None:
        try:
            authority = object.__getattribute__(self, "_authority")
        except AttributeError:
            raise PermissionError(
                "CaptureProcessorInputBinding was not minted by CaptureSession"
            ) from None
        if (
            type(self) is not CaptureProcessorInputBinding
            or authority is not _PROCESSOR_INPUT_BINDING_TOKEN
        ):
            raise PermissionError(
                "CaptureProcessorInputBinding was not minted by CaptureSession"
            )

    def _authority_record(self) -> _CaptureProcessorInputAuthority:
        with _CAPTURE_PROCESSOR_INPUT_AUTHORITIES_LOCK:
            record = _CAPTURE_PROCESSOR_INPUT_AUTHORITIES.get(self)
        if not isinstance(record, _CaptureProcessorInputAuthority):
            raise PermissionError(
                "CaptureProcessorInputBinding was not registered by CaptureSession"
            )
        return record

    def _validate_identity(self) -> _CaptureProcessorInputAuthority:
        self._require_authority()
        record = self._authority_record()
        capture_contract = object.__getattribute__(self, "_capture_contract")
        stream = object.__getattribute__(self, "_stream")
        payload_contract = object.__getattribute__(self, "_payload_contract")
        join_key_contract = object.__getattribute__(self, "_join_key_contract")
        input_edge = object.__getattribute__(self, "_input_edge")
        stream_generation = object.__getattribute__(self, "_stream_generation")
        if (
            capture_contract is not record.capture_contract
            or stream is not record.stream
            or payload_contract is not record.payload_contract
            or join_key_contract is not record.join_key_contract
            or input_edge is not record.input_edge
            or stream_generation is not record.stream_generation
        ):
            raise PermissionError(
                "CaptureProcessorInputBinding owner graph changed after minting"
            )
        if capture_contract.dataset_edge is not input_edge:
            raise RuntimeError("processor input edge differs from capture contract owner")
        if input_edge.schema is not capture_contract.dataset_schema:
            raise RuntimeError("processor input edge differs from capture schema owner")
        if input_edge.event_adapter is not capture_contract.event_adapter:
            raise RuntimeError("processor input edge differs from capture adapter owner")
        if input_edge.expected_cells is not capture_contract.expected_cells:
            raise RuntimeError("processor input edge differs from capture cell schedule")
        if capture_contract.payload_contract is not payload_contract:
            raise RuntimeError("processor payload differs from capture contract owner")
        if input_edge.payload_contract is not payload_contract:
            raise RuntimeError("processor payload differs from dataset edge owner")
        if stream._payload_contract is not payload_contract:
            raise RuntimeError("processor payload differs from capture stream owner")
        if stream._join_key_contract is not join_key_contract:
            raise RuntimeError("processor join key differs from capture stream owner")
        if stream.stream_id != capture_contract.stream_id:
            raise RuntimeError("processor stream id differs from capture contract")
        if stream.generation != stream_generation:
            raise RuntimeError("processor stream generation differs from its minting witness")
        if (
            record.session_reservation is not None
            and record.session_reservation._stream is not stream
        ):
            raise RuntimeError("processor reservation belongs to another stream generation")
        payload_fingerprint = input_edge.payload_contract_fingerprint
        if payload_contract.fingerprint != payload_fingerprint:
            raise RuntimeError("processor payload owner fingerprint changed")
        if stream.payload_contract_fingerprint != payload_fingerprint:
            raise RuntimeError("processor stream payload fingerprint differs")
        if (
            capture_contract.capability.payload_contract_fingerprint
            != payload_fingerprint
        ):
            raise RuntimeError("processor capability payload fingerprint differs")
        if stream.max_payload_bytes != input_edge.payload_max_retained_nbytes:
            raise RuntimeError("processor payload byte bound differs from input edge")
        if join_key_contract.fingerprint != input_edge.key_contract_fingerprint:
            raise RuntimeError("processor join-key fingerprint differs from input edge")
        input_edge.validate_stream(stream)
        return record

    @property
    def capture_contract(self) -> CaptureStreamContract:
        self._validate_identity()
        return self._capture_contract

    @property
    def stream(self) -> AcquisitionStream:
        self._validate_identity()
        return self._stream

    @property
    def payload_contract(self) -> CapturePayloadContract:
        self._validate_identity()
        return self._payload_contract

    @property
    def join_key_contract(self) -> DatasetCellKeyContract:
        self._validate_identity()
        return self._join_key_contract

    @property
    def input_edge(self) -> FrozenDatasetEdge:
        self._validate_identity()
        return self._input_edge

    def _bind_session_reservation(
        self,
        authority: object,
        reservation: ExactReservation,
    ) -> None:
        """Record the one reservation minted through ``CaptureSession`` itself."""

        if authority is not _PROCESSOR_INPUT_BINDING_TOKEN:
            raise PermissionError("capture reservation binding is session-owned")
        if type(reservation) is not ExactReservation:
            raise TypeError("reservation must be an exact ExactReservation")
        record = self._validate_identity()
        stream = self._stream
        if reservation._stream is not stream:
            raise PermissionError("reservation belongs to another capture stream")
        with stream._condition:
            if self._validate_identity() is not record:
                raise PermissionError("capture processor input authority changed")
            if record.session_reservation is not None:
                raise RuntimeError("capture session reservation was already bound")
            if stream._reservations.get(reservation._token) is not reservation:
                raise RuntimeError("capture session reservation is not registered")
            if reservation.state is not ReservationState.RESERVED:
                raise RuntimeError("capture session reservation is not RESERVED")
            record.session_reservation = reservation

    def require_reservation(self, reservation: ExactReservation) -> None:
        """Require the active reservation minted by this exact CaptureSession."""

        if type(reservation) is not ExactReservation:
            raise TypeError("reservation must be an exact ExactReservation")
        record = self._validate_identity()
        stream = self._stream
        if reservation is not record.session_reservation:
            raise PermissionError(
                "reservation was not minted by this CaptureSession"
            )
        with stream._condition:
            if self._validate_identity() is not record:
                raise PermissionError("capture processor input authority changed")
            if stream._reservations.get(reservation._token) is not reservation:
                raise RuntimeError(
                    "reservation is no longer registered with the capture stream"
                )
            if reservation.state is not ReservationState.ACTIVE:
                raise RuntimeError("capture session reservation is not ACTIVE")


class CaptureSession:
    """One owner of producer, device session id, ordinal, and terminal receipt."""

    def __init__(
        self,
        port: BoundCapturePort,
        contract: CaptureStreamContract,
        trace_binding: TraceBinding,
        capture_spec: FrozenCaptureSpec,
    ) -> None:
        if not isinstance(trace_binding, TraceBinding):
            raise TypeError("trace_binding must be TraceBinding")
        if trace_binding.source_id != contract.source_id:
            raise ValueError("TraceBinding source differs from CaptureStreamContract")
        if not isinstance(capture_spec, FrozenCaptureSpec):
            raise TypeError("capture_spec must be FrozenCaptureSpec")
        if capture_spec.owner_fingerprint != contract.capture_spec_owner_fingerprint:
            raise ValueError("capture spec owner differs from CaptureStreamContract")
        if len(capture_spec.payload) > contract.capability.max_capture_spec_bytes:
            raise MemoryError("capture spec exceeds device capability byte bound")
        self._port = port
        self._contract = contract
        self._trace_binding = trace_binding
        self._capture_spec = capture_spec
        self._capture_spec_digest = capture_spec.digest
        self._session_id = uuid.uuid4().hex
        stream, producer = AcquisitionStream.create(
            contract.stream_id,
            contract.payload_contract,
            flow_control=contract.capability.flow_control,
            retention_events=contract.max_inflight_events,
            retention_bytes=contract.max_inflight_bytes,
            join_key_contract=DatasetCellKeyContract(contract.dataset_schema),
        )
        processor_input_binding = CaptureProcessorInputBinding(
            _PROCESSOR_INPUT_BINDING_TOKEN,
            capture_contract=contract,
            stream=stream,
        )
        self._stream = stream
        self._producer: AcquisitionProducer = producer
        self._processor_input_binding = processor_input_binding
        self._state = CaptureSessionState.NEW
        self._delivered = 0
        self._metadata_hasher = OrderedDatasetMetadataHasher(
            contract.dataset_edge.metadata_contract_fingerprint
        )
        self._completion: CaptureCompletion | None = None
        self._reservation: ExactReservation | None = None
        self._source_event_span_hasher: OrderedEventSpanHasher | None = None
        self._exact_consumer_readiness: ExactConsumerReadiness | None = None
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._owner_thread_id = threading.get_ident()
        self._hardware_prepare_attempted = False

    @property
    def stream(self) -> AcquisitionStream:
        return self._stream

    @property
    def processor_input_binding(self) -> CaptureProcessorInputBinding:
        with self._lock:
            if self._state is not CaptureSessionState.NEW:
                raise RuntimeError(
                    "processor input binding is only available while capture session is NEW"
                )
            binding = self._processor_input_binding
            binding._validate_identity()
            return binding

    @property
    def state(self) -> CaptureSessionState:
        with self._lock:
            return self._state

    @property
    def session_id(self) -> str:
        return self._session_id

    def reserve_exact(self) -> ExactReservation:
        """Mint the only formal reservation that may authorize this capture."""

        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                if self._state is not CaptureSessionState.NEW:
                    raise RuntimeError("exact reservation must precede capture prepare")
                if self._reservation is not None:
                    raise RuntimeError("capture session already has an exact reservation")
            reservation = self._stream.reserve(
                total_events=self._contract.total_events,
                max_inflight_events=self._contract.max_inflight_events,
                max_inflight_bytes=self._contract.max_inflight_bytes,
                trace_binding=self._trace_binding,
            )
            source_event_span_hasher = OrderedEventSpanHasher(
                self._stream.stream_id,
                self._stream.generation,
                reservation.start_sequence,
            )
            try:
                self._processor_input_binding._bind_session_reservation(
                    _PROCESSOR_INPUT_BINDING_TOKEN,
                    reservation,
                )
            except BaseException:
                reservation.abort()
                reservation.release()
                raise
            with self._lock:
                self._reservation = reservation
                self._source_event_span_hasher = source_event_span_hasher
            return reservation

    def bind_exact_consumer(self, readiness: ExactConsumerReadiness) -> None:
        """Bind a complete required exact chain before any capture command."""

        with self._operation_lock:
            self._assert_owner_thread()
            if not isinstance(readiness, ExactConsumerReadiness):
                raise TypeError("readiness must be ExactConsumerReadiness")
            with self._lock:
                if self._state is not CaptureSessionState.NEW:
                    raise RuntimeError("exact consumer readiness must precede capture prepare")
                reservation = self._reservation
                if reservation is None:
                    raise RuntimeError("capture session has no exact reservation")
                if self._exact_consumer_readiness is not None:
                    raise RuntimeError("capture session already has exact consumer readiness")
            self._validate_readiness(readiness, reservation)
            readiness._claim_binding(self)
            with self._lock:
                self._exact_consumer_readiness = readiness

    def prepare(self, context: RunContext) -> None:
        with self._operation_lock:
            self._assert_owner_thread()
            if context.run_id.value != self._trace_binding.run_id:
                raise ValueError("RunContext differs from the frozen TraceBinding")
            with self._lock:
                if self._state is not CaptureSessionState.NEW:
                    raise RuntimeError("capture session can only be prepared once")
                reservation = self._reservation
                readiness = self._exact_consumer_readiness
            if reservation is None:
                raise RuntimeError("capture cannot prepare without its exact reservation")
            if readiness is None:
                raise RuntimeError("capture has no exact consumer readiness proof")
            self._validate_readiness(readiness, reservation)
            self._validate_current_capability()
            with self._lock:
                self._hardware_prepare_attempted = True
                self._state = CaptureSessionState.PREPARING
            try:
                ack = context.device(self._port.device.key).execute(
                    PrepareCaptureCommand(
                        session_id=self._session_id,
                        run_id=context.run_id.value,
                        source_id=self._contract.source_id,
                    capture_spec_payload=self._capture_spec.payload,
                    capture_spec_owner_fingerprint=(
                        self._capture_spec.owner_fingerprint
                    ),
                        capture_spec_fingerprint=self._capture_spec_digest,
                        capability_fingerprint=self._port.capability.capability_fingerprint,
                        settings_fingerprint=self._port.capability.settings_fingerprint,
                        expected_total_events=self._contract.total_events,
                        timeout_seconds=(
                            self._port.capability.max_blocking_call_seconds
                        ),
                    )
                )
                self._validate_ack(ack, CapturePreparedAck)
                if (
                    ack.settings_fingerprint
                    != self._port.capability.settings_fingerprint
                    or ack.capability_fingerprint
                    != self._port.capability.capability_fingerprint
                    or ack.capture_spec_fingerprint != self._capture_spec_digest
                ):
                    raise RuntimeError(
                        "capture prepare acknowledgement differs from frozen contract"
                    )
            except BaseException as error:
                self._poison(
                    SourceFailed(f"capture prepare failed: {safe_error_summary(error)}")
                )
                raise
            with self._lock:
                self._state = CaptureSessionState.PREPARED

    def start(self, context: RunContext) -> None:
        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                if self._state is not CaptureSessionState.PREPARED:
                    raise RuntimeError("capture session is not prepared")
                reservation = self._reservation
                if reservation is None:
                    raise RuntimeError("capture cannot start without its exact reservation")
                if reservation.state is not ReservationState.ACTIVE:
                    raise RuntimeError("capture exact reservation is not active")
                readiness = self._exact_consumer_readiness
                if readiness is None:
                    raise RuntimeError("capture has no exact consumer readiness proof")
                self._validate_readiness(readiness, reservation)
                self._state = CaptureSessionState.STARTING
            try:
                ack = context.device(self._port.device.key).execute(
                    StartCaptureCommand(
                        self._session_id,
                        self._port.capability.max_blocking_call_seconds,
                    )
                )
                self._validate_ack(ack, CaptureStartedAck)
            except BaseException as error:
                self._poison(
                    SourceFailed(f"capture start failed: {safe_error_summary(error)}")
                )
                raise
            with self._lock:
                self._state = CaptureSessionState.STARTED

    def _validate_readiness(
        self,
        readiness: ExactConsumerReadiness,
        reservation: ExactReservation,
    ) -> None:
        readiness.validate_source(
            reservation=reservation,
            trace_binding=self._trace_binding,
            payload_contract_fingerprint=(
                self._contract.dataset_edge.payload_contract_fingerprint
            ),
            join_key_contract_fingerprint=dataset_cell_key_fingerprint(
                self._contract.dataset_schema
            ),
            source_contract_digest=self._contract.dataset_edge.consumer_contract_digest,
            source_schedule_digest=self._contract.dataset_edge.schedule_digest,
            source_key_sequence_digest=(
                self._contract.dataset_edge.exact_key_sequence_digest
            ),
            total_events=self._contract.total_events,
        )

    def capture_next(
        self,
        context: RunContext,
    ) -> None:
        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                if self._state is not CaptureSessionState.STARTED:
                    raise RuntimeError("capture session is not started")
                expected_ordinal = self._delivered
                if expected_ordinal >= self._contract.total_events:
                    raise RuntimeError("capture already delivered its frozen event budget")
                join_key = self._contract.expected_cells[expected_ordinal]
            try:
                ack = context.device(self._port.device.key).execute(
                    ReadCaptureCommand(
                        self._session_id,
                        self._port.capability.max_blocking_call_seconds,
                    )
                )
                self._validate_ack(ack, CapturedPayloadAck)
                payload_contract = self._contract.payload_contract
                payload = ack.payload
                actual_ordinal = _nonnegative_int(
                    payload_contract.source_ordinal(payload),
                    "payload source ordinal",
                )
                if actual_ordinal != expected_ordinal:
                    raise StreamError(
                        f"payload ordinal {actual_ordinal} differs from expected "
                        f"{expected_ordinal}"
                    )
                captured_at = payload_contract.captured_at(payload)
                if not math.isfinite(float(captured_at)):
                    raise ValueError("payload captured_at must be finite")
                correlation_id = payload_contract.correlation_id(payload)
                _canonical_text(correlation_id, "payload correlation_id")
                envelope = self._producer.emit(
                    payload,
                    captured_at=float(captured_at),
                    trace=TraceContext(
                        run_id=self._trace_binding.run_id,
                        source_id=self._trace_binding.source_id,
                        correlation_id=correlation_id,
                    ),
                    join_key=join_key,
                )
                source_event_span_hasher = self._source_event_span_hasher
                if source_event_span_hasher is None:
                    raise RuntimeError("capture has no source event span owner")
                source_event_span_hasher.update(envelope.ref)
                stored = envelope.payload
                if _nonnegative_int(
                    payload_contract.source_ordinal(stored),
                    "stored payload source ordinal",
                ) != actual_ordinal:
                    raise StreamError("payload snapshot changed the physical source ordinal")
                if float(payload_contract.captured_at(stored)) != float(captured_at):
                    raise StreamError("payload snapshot changed captured_at")
                if payload_contract.correlation_id(stored) != correlation_id:
                    raise StreamError("payload snapshot changed correlation_id")
                metadata_contract = self._contract.dataset_edge.metadata_contract
                metadata = metadata_contract.snapshot(stored)
                metadata_contract.validate(metadata)
                metadata_digest = _sha256(
                    metadata_contract.digest(metadata),
                    "payload metadata digest",
                )
            except BaseException as error:
                self._poison(
                    SourceFailed(
                        "captured payload failed validation/publish: "
                        + safe_error_summary(error)
                    )
                )
                raise
            with self._lock:
                self._metadata_hasher.update(metadata_digest)
                self._delivered += 1

    def complete(self, context: RunContext) -> CaptureCompletion:
        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                if self._state is not CaptureSessionState.STARTED:
                    raise RuntimeError("capture session is not started")
                if self._delivered != self._contract.total_events:
                    raise RuntimeError(
                        "cannot complete before every scheduled payload is delivered"
                    )
                self._state = CaptureSessionState.COMPLETING
            try:
                ack = context.device(self._port.device.key).execute(
                    CompleteCaptureCommand(
                        self._session_id,
                        self._contract.total_events,
                        self._port.capability.max_blocking_call_seconds,
                    )
                )
                self._validate_ack(ack, CaptureTerminalAck)
                if (
                    ack.produced_count != self._contract.total_events
                    or ack.drained_count != self._contract.total_events
                    or not ack.source_stopped
                    or not ack.no_more_frames
                    or not ack.joined
                ):
                    raise StreamError(
                        "capture terminal counters/stop/drain/join proof failed"
                    )
                expected_digest = self._metadata_hasher.digest()
                if ack.ordered_metadata_digest != expected_digest:
                    raise StreamError("capture terminal metadata digest differs")
                if (
                    ack.settings_fingerprint
                    != self._port.capability.settings_fingerprint
                    or ack.capability_fingerprint
                    != self._port.capability.capability_fingerprint
                    or ack.capture_spec_fingerprint != self._capture_spec_digest
                ):
                    raise StreamError("capture terminal contract binding differs")
                completion = CaptureCompletion(
                    _COMPLETION_TOKEN,
                    session=self,
                    terminal=ack,
                )
                eos = self._producer.finish()
                object.__setattr__(completion, "_eos", eos)
            except BaseException as error:
                self._poison(
                    SourceFailed(
                        "capture terminal validation failed: "
                        + safe_error_summary(error)
                    )
                )
                raise
            with self._lock:
                self._state = CaptureSessionState.COMPLETED
                self._completion = completion
            return completion

    def fail(self, error: BaseException) -> None:
        """Poison software authority; physical termination belongs to cleanup."""

        with self._operation_lock:
            self._assert_owner_thread()
            failure = SourceFailed(
                "capture session aborted: " + safe_error_summary(error)
            )
            self._poison(failure)

    def cleanup(self, context: RunContext) -> CleanupReport:
        """Poison unfinished data, then execute cleanup-capable stop/drain/join."""

        with self._operation_lock:
            self._assert_owner_thread()
            with self._lock:
                completed = self._state is CaptureSessionState.COMPLETED
            if not completed:
                self._poison(SourceFailed("capture terminated during cleanup"))
            report: CleanupReport | None = None
            port_error: BaseException | None = None
            try:
                report = (
                    self._port.cleanup(context, self._session_id)
                    if self._hardware_prepare_attempted
                    else self._port.verify_idle(context)
                )
            except BaseException as error:
                port_error = error
            release_errors: list[BaseException] = []
            reservation = self._reservation
            if reservation is not None and reservation.state is not ReservationState.RELEASED:
                try:
                    if not reservation.consumer_bound:
                        if reservation.state is not ReservationState.COMPLETED:
                            reservation.abort(cancelled=context.cancellation.is_cancelled)
                        reservation.release()
                    elif reservation.state in (
                        ReservationState.COMPLETED,
                        ReservationState.FAILED,
                        ReservationState.CANCELLED,
                    ):
                        reservation.release()
                    else:
                        raise RuntimeError(
                            "exact consumer did not terminate its reservation"
                        )
                except BaseException as error:
                    release_errors.append(error)
            if reservation is None or reservation.state is ReservationState.RELEASED:
                completion = self._completion
                if completion is not None:
                    try:
                        self._consume_completion_authority(completion)
                    except BaseException as error:
                        release_errors.append(error)
                with self._lock:
                    self._reservation = None
                    self._exact_consumer_readiness = None
                    self._source_event_span_hasher = None
            if port_error is not None:
                for error in release_errors:
                    record_secondary_failure(
                        port_error,
                        "exact reservation teardown also failed",
                        error,
                    )
                raise port_error
            assert report is not None
            if not release_errors:
                return report
            return CleanupReport(
                safety_proofs=report.safety_proofs,
                decisions=report.decisions,
                errors=(*report.errors, *release_errors),
            )

    def _validate_current_capability(self) -> None:
        if (
            self._port.device.validate_capability(
                self._port.capability_attestation
            )
            is not self._port.capability
        ):
            raise RuntimeError("capture capability attestation snapshot changed")

    def owns_completion(self, completion: CaptureCompletion) -> bool:
        with self._lock:
            return (
                isinstance(completion, CaptureCompletion)
                and completion._session is self
                and self._completion is completion
            )

    def _consume_completion_authority(
        self,
        completion: CaptureCompletion,
    ) -> None:
        """Single no-allocation commit that breaks the completed owner graph."""

        self._assert_owner_thread()
        with self._lock:
            if (
                not isinstance(completion, CaptureCompletion)
                or completion._session is not self
                or completion._terminal_reservation is None
                or self._completion is not completion
            ):
                raise RuntimeError("capture completion authority is absent or differs")
            self._completion = None
            self._exact_consumer_readiness = None
            self._source_event_span_hasher = None
            object.__setattr__(completion, "_session", None)
            object.__setattr__(completion, "_terminal_reservation", None)

    def _commit_pipeline_authority(
        self,
        completion: CaptureCompletion,
        dataset: SealedDatasetArtifact,
    ) -> None:
        """One owner-lane commit for all process-local result capabilities."""

        self._assert_owner_thread()
        if not isinstance(dataset, SealedDatasetArtifact):
            raise TypeError("dataset must be SealedDatasetArtifact")
        with self._lock:
            reservation = completion._terminal_reservation
            if (
                completion._session is not self
                or reservation is None
                or self._completion is not completion
                or not dataset._belongs_to_terminal_reservation(reservation)
            ):
                raise RuntimeError("pipeline result authority is absent or differs")
            # Final no-fail ownership commit.  All type, identity, and graph
            # checks precede these direct reference clears under the one
            # CaptureSession owner lock.
            object.__setattr__(dataset, "_terminal_reservation", None)
            self._completion = None
            self._exact_consumer_readiness = None
            self._source_event_span_hasher = None
            object.__setattr__(completion, "_session", None)
            object.__setattr__(completion, "_terminal_reservation", None)

    def _poison(self, error: StreamError) -> None:
        with self._lock:
            if self._state is CaptureSessionState.COMPLETED:
                raise RuntimeError("completed capture session cannot fail")
            self._state = CaptureSessionState.FAILED
        try:
            self._producer.fail(error)
        except BaseException:
            pass

    def _validate_ack(self, ack: object, expected_type: type) -> None:
        if not isinstance(ack, expected_type):
            raise TypeError(
                f"capture device returned {type(ack).__name__}, expected {expected_type.__name__}"
            )
        if ack.session_id != self._session_id:
            raise RuntimeError("capture acknowledgement session_id differs")
        if ack.binding_id != self._port.device.binding_id:
            raise RuntimeError("capture acknowledgement binding_id differs")
        if ack.connection_generation != self._port.device.connection_generation:
            raise RuntimeError("capture acknowledgement generation differs")

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("CaptureSession operation left its owner I/O lane")

__all__ = [
    "BoundCapturePort",
    "CaptureCapabilitySnapshot",
    "CaptureCompletion",
    "CapturePayloadContract",
    "CapturePreparedAck",
    "CaptureProcessorInputBinding",
    "CaptureRuntimeProfile",
    "FrozenCaptureSpec",
    "CaptureSession",
    "CaptureSessionState",
    "CaptureStartedAck",
    "CaptureStreamContract",
    "CaptureTerminalAck",
    "CameraCapabilityEvidence",
    "CameraCaptureProvenance",
    "CameraPhysicalFacts",
    "camera_capability_evidence_from_tree",
    "camera_capability_evidence_to_tree",
    "camera_physical_facts_from_tree",
    "camera_physical_facts_to_tree",
    "CapturedPayloadAck",
    "CompleteCaptureCommand",
    "PrepareCaptureCommand",
    "ReadCaptureCommand",
    "StartCaptureCommand",
]
