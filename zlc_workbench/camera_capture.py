"""Composition-owned bridge from concrete cameras to the exact capture runtime."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from Zou_lab_control.neutral_atom.devices.base import CameraDevice, CameraFrameRecord
from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera
from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera
from Zou_lab_control.neutral_atom.devices.virtual import VirtualCamera, VirtualMotCamera
from zlc_data import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    SPATIAL_X,
    SPATIAL_Y,
    VALID,
    ValidityContract,
    Value,
    ValueSchema,
)
from zlc_neutral_atom.acquisition import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraCaptureSpec,
    CameraDatasetEventAdapter,
    CameraFrameMetadata,
    CameraSample,
    CameraSampleContract,
    decode_camera_capture_spec,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.readout.contracts import (
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    ReadoutBindingKey,
)
from zlc_neutral_atom.runtime import (
    BoundCapturePort,
    BoundDevice,
    BoundMeasurement,
    CaptureCapabilitySnapshot,
    CapturePreparedAck,
    CaptureRuntimeProfile,
    CaptureStartedAck,
    CaptureStreamContract,
    CaptureTerminalAck,
    CapturedPayloadAck,
    CompleteCaptureCommand,
    DatasetCellAddress,
    MeasurementDefinition,
    OrderedDatasetMetadataHasher,
    PrepareCaptureCommand,
    ProducerFlowControl,
    ReadCaptureCommand,
    SafetyOperation,
    SessionClosedAck,
    SessionCloseCommand,
    StartCaptureCommand,
    StreamId,
)
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.runtime.capture import (
    CameraCapabilityEvidence,
    CameraCaptureProvenance,
    CameraPhysicalFacts,
)
from zlc_storage import (
    canonical_digest,
    canonical_text as _canonical_text,
    positive_integer as _positive_int,
    sha256_text as _sha256,
)

from ._endpoint_binding import require_current_endpoint_binding as _require_binding
from .legacy_runtime import LegacyDeviceRegistry, TargetDeviceEndpoint


_SUPPORTED_CAMERA_TYPES = (QCMOSCamera, PylonCamera, VirtualCamera, VirtualMotCamera)

def _camera_dtype(camera: CameraDevice) -> np.dtype:
    if type(camera) in (QCMOSCamera, VirtualCamera):
        return np.dtype("<u2")
    if type(camera) is VirtualMotCamera:
        return np.dtype("u1")
    if type(camera) is PylonCamera:
        pixel_format = str(camera.pixel_format).strip().lower()
        if pixel_format == "mono8":
            return np.dtype("u1")
        if pixel_format in {"mono10", "mono12", "mono16"}:
            return np.dtype("<u2")
        raise ValueError(
            f"Pylon pixel format {camera.pixel_format!r} has no exact scalar frame contract"
        )
    raise TypeError(f"unsupported camera adapter type {type(camera).__qualname__}")


def _camera_mode(camera: CameraDevice) -> CameraAcquisitionMode:
    return (
        CameraAcquisitionMode.FREE_RUNNING
        if camera._free_run
        else CameraAcquisitionMode.EXTERNAL_TRIGGERED
    )


def _settings_tree(camera: CameraDevice) -> dict[str, object]:
    shape = camera.frame_shape
    if shape is None or len(shape) != 2 or any(int(size) < 1 for size in shape):
        raise RuntimeError("camera frame shape is unavailable after connection handshake")
    snapshot = dict(camera.snapshot())
    # Snapshot methods also expose live observations for UI.  Only frozen
    # acquisition settings belong in the settings fingerprint: a pulse may
    # legitimately change these observations while the camera configuration
    # remains identical.
    for presentation_or_state_field in ("open", "last_sequence", "last_levels"):
        snapshot.pop(presentation_or_state_field, None)
    snapshot.update(
        {
            "adapter_type": f"{type(camera).__module__}.{type(camera).__qualname__}",
            "frame_shape": tuple(int(size) for size in shape),
            "sensor_shape": (
                None
                if camera.sensor_shape is None
                else tuple(int(size) for size in camera.sensor_shape)
            ),
            "frame_dtype": _camera_dtype(camera).str,
            "acquisition_mode": _camera_mode(camera).value,
            # Passive installation wiring remains authoritative even while a
            # camera is temporarily free-running and therefore has no active
            # trigger channel.
            "capture_trigger_channels": tuple(camera.capture_trigger_channels),
            "effective_trigger_channels": tuple(camera.effective_trigger_channels),
        }
    )
    if type(camera) is QCMOSCamera:
        applied_exposure = snapshot.get("applied_exposure_seconds")
        minimum_interval = snapshot.get(
            "required_external_trigger_interval_seconds"
        )
        applied_readout_speed = snapshot.pop("applied_readout_speed", None)
        applied_sensor_mode = snapshot.pop("applied_sensor_mode", None)
        applied_trigger_global_exposure = snapshot.pop(
            "applied_trigger_global_exposure",
            None,
        )
        if (
            applied_exposure is None
            or minimum_interval is None
            or applied_readout_speed is None
            or applied_sensor_mode is None
            or applied_trigger_global_exposure is None
        ):
            raise RuntimeError(
                "qCMOS physical working-point readback is unavailable"
            )
        snapshot.update(
            {
                "readout_speed": int(applied_readout_speed),
                "sensor_mode": int(applied_sensor_mode),
                "trigger_global_exposure": int(
                    applied_trigger_global_exposure
                ),
                # E0 has not qualified the edge-to-integration latency for the
                # deployed qCMOS mode.  A DCAM timing-rate readback is not proof
                # of this separate physical fact.
                "external_trigger_integration_start_offset_seconds": None,
            }
        )
    else:
        # These adapters currently expose no independently measured trigger-rate
        # lower bound.  Zero means exactly that -- no additional known lower
        # bound -- rather than claiming unlimited physical trigger bandwidth.
        snapshot["applied_exposure_seconds"] = float(snapshot["exposure"])
        snapshot["required_external_trigger_interval_seconds"] = 0.0
        snapshot["external_trigger_integration_start_offset_seconds"] = (
            0.0 if type(camera) is VirtualCamera else None
        )
    return snapshot


def _readout_mode_from_settings(
    camera: CameraDevice,
    settings: dict[str, object],
) -> str:
    if type(camera) is QCMOSCamera:
        return (
            f"qcmos:readout-speed={int(settings['readout_speed'])}:"
            f"sensor-mode={settings.get('sensor_mode')}:"
            f"global-exposure={settings.get('trigger_global_exposure')}"
        )
    if type(camera) is PylonCamera:
        return (
            f"pylon:pixel-format={settings['pixel_format']}:"
            f"trigger-source={settings['trigger_source']}"
        )
    return f"{type(camera).__name__}:mode={settings['acquisition_mode']}"


def _physical_facts(
    camera: CameraDevice,
    binding: BoundDevice,
    source_id: str,
    payload_contract: CameraSampleContract,
    settings_tree: dict[str, object],
    settings_fingerprint: str,
) -> CameraPhysicalFacts:
    sensor = settings_tree.get("sensor_shape")
    if sensor is None:
        raise RuntimeError(
            "camera sensor geometry is unavailable during capability probe"
        )
    sensor_y, sensor_x = (int(size) for size in sensor)
    roi = settings_tree.get("roi")
    if roi is None:
        origin_yx = (0, 0)
        roi_shape_yx = (sensor_y, sensor_x)
    else:
        x, width, y, height = (int(value) for value in roi)
        origin_yx = (y, x)
        roi_shape_yx = (height, width)
    y_axis, x_axis = payload_contract.value_schema.data_axes
    gain_value = settings_tree.get("gain", 1.0)
    gain = float(gain_value)
    facts = CameraPhysicalFacts(
        camera_identity=binding.stable_device_identity,
        sensor_identity=f"{binding.stable_device_identity}/sensor",
        optical_path=f"installation-role/{source_id}",
        capture_trigger_channels=settings_tree["capture_trigger_channels"],
        sensor_shape_yx=(sensor_y, sensor_x),
        roi_origin_yx=origin_yx,
        roi_shape_yx=roi_shape_yx,
        binning_yx=(1, 1),
        spatial_y_axis_id=y_axis.axis_id,
        spatial_x_axis_id=x_axis.axis_id,
        coordinate_frame=y_axis.coordinate_frame,
        dtype=payload_contract.value_schema.dtype,
        count_unit=payload_contract.value_schema.value_unit,
        exposure_seconds=float(settings_tree["applied_exposure_seconds"]),
        required_external_trigger_interval_seconds=float(
            settings_tree["required_external_trigger_interval_seconds"]
        ),
        external_trigger_integration_start_offset_seconds=settings_tree[
            "external_trigger_integration_start_offset_seconds"
        ],
        # These adapters expose no independent gain control; unity is their
        # explicit digital-count gain policy, frozen into the same snapshot.
        gain=gain,
        readout_mode=_readout_mode_from_settings(camera, settings_tree),
        opaque_frame_settings_fingerprint=settings_fingerprint,
    )
    if facts.output_shape_yx != payload_contract.value_schema.data_shape:
        raise RuntimeError(
            "camera physical geometry differs from the frozen payload schema"
        )
    return facts


def _value_schema(camera: CameraDevice, source_id: str) -> ValueSchema:
    shape = camera.frame_shape
    if shape is None:
        raise RuntimeError("camera frame shape is unavailable")
    height, width = (int(size) for size in shape)
    y = AxisSpec(
        AxisId(f"{source_id}.y"),
        "ROI-local output y",
        SPATIAL_Y,
        height,
        tuple(range(height)),
        unit="pixel",
        coordinate_frame=CoordinateFrameId(
            f"{source_id}.roi-local-output-pixels"
        ),
    )
    x = AxisSpec(
        AxisId(f"{source_id}.x"),
        "ROI-local output x",
        SPATIAL_X,
        width,
        tuple(range(width)),
        unit="pixel",
        coordinate_frame=CoordinateFrameId(
            f"{source_id}.roi-local-output-pixels"
        ),
    )
    return ValueSchema(
        (y, x),
        ValidityContract.value(),
        _camera_dtype(camera),
        "count",
    )


@dataclass
class _EndpointSession:
    session_id: str
    spec_fingerprint: str
    expected_frames: int
    payload_contract: CameraSampleContract
    metadata_hasher: OrderedDatasetMetadataHasher
    started: bool = False
    drained_count: int = 0
    terminal: object | None = None
    closed: bool = False


_DESCRIPTION_TOKEN = object()


@dataclass(frozen=True)
class CameraCaptureDescription:
    _authority: object
    source_id: str
    payload_contract: CameraSampleContract
    settings_fingerprint: str
    physical_facts: CameraPhysicalFacts

    def __post_init__(self) -> None:
        if self._authority is not _DESCRIPTION_TOKEN:
            raise PermissionError(
                "CameraCaptureDescription can only be minted by CameraCaptureEndpoint"
            )
        _canonical_text(self.source_id, "source_id")
        if not isinstance(self.payload_contract, CameraSampleContract):
            raise TypeError("payload_contract must be CameraSampleContract")
        _sha256(self.settings_fingerprint, "settings_fingerprint")
        if not isinstance(self.physical_facts, CameraPhysicalFacts):
            raise TypeError("physical_facts must be CameraPhysicalFacts")
        if (
            self.physical_facts.opaque_frame_settings_fingerprint
            != self.settings_fingerprint
        ):
            raise ValueError("physical facts and settings fingerprint differ")

    @property
    def capture_trigger_channels(self) -> tuple[str, ...]:
        """The one broker-attested physical trigger-wiring tuple."""

        return self.physical_facts.capture_trigger_channels

    def event_setting(self, event_index: int) -> CameraEventReadoutSetting:
        return self.physical_facts.event_setting(event_index)

    def mint_descriptor(
        self,
        schema: DatasetSchema,
        event_settings: tuple[CameraEventReadoutSetting, ...],
        *,
        camera_arm_spec_fingerprint: str,
    ) -> CameraCaptureDescriptor:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        settings = tuple(event_settings)
        if any(not isinstance(item, CameraEventReadoutSetting) for item in settings):
            raise TypeError("event_settings must contain CameraEventReadoutSetting")
        readout_axes = tuple(
            axis for axis in schema.point_axes if axis.role == READOUT_EVENT
        )
        if len(readout_axes) > 1:
            raise ValueError("camera schema has multiple READOUT_EVENT axes")
        event_axis = readout_axes[0] if readout_axes else None
        expected_indices = (
            (0,) if event_axis is None else tuple(range(event_axis.size))
        )
        if tuple(item.event_index for item in settings) != expected_indices:
            raise ValueError(
                "event_settings must explicitly cover every READOUT_EVENT index"
            )
        for setting in settings:
            if setting != self.physical_facts.event_setting(setting.event_index):
                raise ValueError(
                    "event setting differs from broker-attested camera settings"
                )
        facts = self.physical_facts
        descriptor = CameraCaptureDescriptor(
            camera_identity=facts.camera_identity,
            sensor_identity=facts.sensor_identity,
            optical_path=facts.optical_path,
            sensor_shape_yx=facts.sensor_shape_yx,
            roi_origin_yx=facts.roi_origin_yx,
            roi_shape_yx=facts.roi_shape_yx,
            binning_yx=facts.binning_yx,
            spatial_y_axis_id=facts.spatial_y_axis_id,
            spatial_x_axis_id=facts.spatial_x_axis_id,
            coordinate_frame=facts.coordinate_frame,
            dtype=facts.dtype,
            count_unit=facts.count_unit,
            readout_event_axis_id=(None if event_axis is None else event_axis.axis_id),
            event_settings=settings,
            camera_arm_spec_fingerprint=_sha256(
                camera_arm_spec_fingerprint,
                "camera_arm_spec_fingerprint",
            ),
        )
        return descriptor


@dataclass(frozen=True)
class CameraCaptureBindingRequest:
    """Composition request for one finite camera dataset binding."""

    role: str
    repeat_axis: AxisSpec
    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout
    expected_cells: tuple[DatasetCellAddress, ...]
    mode: CameraAcquisitionMode
    required_consumer_lag_events: int
    transport_memory_limit_bytes: int
    event_settings: tuple[CameraEventReadoutSetting, ...] | None = None

    def __post_init__(self) -> None:
        _canonical_text(self.role, "camera role")
        if not isinstance(self.repeat_axis, AxisSpec) or self.repeat_axis.role != REPEAT:
            raise ValueError("repeat_axis must have the repeat role")
        points = tuple(self.point_axes)
        if any(not isinstance(axis, AxisSpec) for axis in points):
            raise TypeError("point_axes must contain AxisSpec values")
        object.__setattr__(self, "point_axes", points)
        if not isinstance(self.point_layout, PointLayout):
            raise TypeError("point_layout must be PointLayout")
        if self.point_layout.logical_shape != tuple(axis.size for axis in points):
            raise ValueError("point_layout shape differs from point axes")
        # The runtime FrozenDatasetEdge is the sole complete-permutation owner.
        # This request only snapshots the caller's declarative sequence.
        object.__setattr__(self, "expected_cells", tuple(self.expected_cells))
        if not isinstance(self.mode, CameraAcquisitionMode):
            raise TypeError("mode must be CameraAcquisitionMode")
        lag = self.required_consumer_lag_events
        if isinstance(lag, bool) or not isinstance(lag, Integral) or lag < 0:
            raise ValueError("required_consumer_lag_events must be non-negative int")
        object.__setattr__(self, "required_consumer_lag_events", int(lag))
        object.__setattr__(
            self,
            "transport_memory_limit_bytes",
            _positive_int(
                self.transport_memory_limit_bytes,
                "transport_memory_limit_bytes",
            ),
        )
        if self.event_settings is not None:
            settings = tuple(self.event_settings)
            if any(
                not isinstance(item, CameraEventReadoutSetting) for item in settings
            ):
                raise TypeError(
                    "event_settings must contain CameraEventReadoutSetting values"
                )
            if tuple(item.event_index for item in settings) != tuple(
                sorted(item.event_index for item in settings)
            ):
                raise ValueError("event_settings must use canonical event-index order")
            object.__setattr__(self, "event_settings", settings)


class CameraCaptureEndpoint:
    """One raw camera's typed command endpoint, private to composition."""

    def __init__(
        self,
        camera: CameraDevice,
        source_id: str,
        *,
        max_inflight_frames: int | None = None,
        max_blocking_call_seconds: float | None = None,
        max_capture_spec_bytes: int = 4096,
    ) -> None:
        if type(camera) not in _SUPPORTED_CAMERA_TYPES:
            raise TypeError(
                f"camera capture endpoint has no exact adapter for {type(camera).__qualname__}"
            )
        self._camera = camera
        self._source_id = _canonical_text(source_id, "source_id")
        self._max_inflight_frames = _positive_int(
            max_inflight_frames or camera.recent_capacity,
            "max_inflight_frames",
        )
        timeout = (
            self._default_timeout(camera)
            if max_blocking_call_seconds is None
            else float(max_blocking_call_seconds)
        )
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("max_blocking_call_seconds must be finite and positive")
        self._max_blocking_call_seconds = timeout
        self._max_capture_spec_bytes = _positive_int(
            max_capture_spec_bytes,
            "max_capture_spec_bytes",
        )
        self._lock = threading.RLock()
        self._payload_contract: CameraSampleContract | None = None
        self._settings_fingerprint: str | None = None
        self._capability_fingerprint: str | None = None
        self._binding_id: str | None = None
        self._generation: str | None = None
        self._description: CameraCaptureDescription | None = None
        self._session: _EndpointSession | None = None
        self._last_prepare_session_id: str | None = None

    @staticmethod
    def _default_timeout(camera: CameraDevice) -> float:
        if type(camera) is QCMOSCamera:
            return float(camera.config.timeout_ms) / 1000.0
        if type(camera) is PylonCamera:
            return float(camera.timeout)
        return 2.0

    @property
    def target_endpoint(self) -> TargetDeviceEndpoint:
        return TargetDeviceEndpoint(
            self.execute_command,
            self.capability_probe,
            self.close_session,
            self.describe,
            {SafetyOperation.DISARM: self.interrupt},
        )

    def describe(self, binding: BoundDevice) -> CameraCaptureDescription:
        with self._lock:
            self._validate_binding(binding)
            if self._description is None:
                raise RuntimeError("camera capability has not been probed")
            return self._description

    def payload_contract(self, binding: BoundDevice) -> CameraSampleContract:
        with self._lock:
            self._validate_binding(binding)
            if self._payload_contract is None:
                raise RuntimeError("camera capability has not been probed")
            return self._payload_contract

    def settings_fingerprint(self, binding: BoundDevice) -> str:
        with self._lock:
            self._validate_binding(binding)
            if self._settings_fingerprint is None:
                raise RuntimeError("camera capability has not been probed")
            return self._settings_fingerprint

    def capability_probe(self, binding: BoundDevice) -> CaptureCapabilitySnapshot:
        with self._lock:
            if self._session is not None and not self._session.closed:
                raise RuntimeError("cannot probe camera capability during a capture session")
            settings_tree = _settings_tree(self._camera)
            settings = canonical_digest(settings_tree)
            payload_contract = CameraSampleContract(
                _value_schema(self._camera, self._source_id)
            )
            frame_bytes = int(np.prod(payload_contract.value_schema.data_shape)) * int(
                payload_contract.value_schema.dtype.itemsize
            )
            physical_facts = _physical_facts(
                self._camera,
                binding,
                self._source_id,
                payload_contract,
                settings_tree,
                settings,
            )
            driver_ring_bytes = self._max_inflight_frames * frame_bytes
            adapter_record_retention_bytes = max(
                self._max_inflight_frames,
                int(self._camera.recent_capacity),
            ) * frame_bytes
            capability_evidence = CameraCapabilityEvidence(
                adapter_type=(
                    f"{type(self._camera).__module__}."
                    f"{type(self._camera).__qualname__}"
                ),
                source_id=self._source_id,
                settings_fingerprint=settings,
                payload_contract_fingerprint=payload_contract.fingerprint,
                capture_spec_owner_fingerprint=(
                    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT
                ),
                flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
                max_source_burst_events=self._max_inflight_frames,
                driver_ring_bytes=driver_ring_bytes,
                adapter_record_retention_bytes=(
                    adapter_record_retention_bytes
                ),
                max_blocking_call_seconds=self._max_blocking_call_seconds,
                max_capture_spec_bytes=self._max_capture_spec_bytes,
                physical_facts=physical_facts,
            )
            capability = capability_evidence.fingerprint
            self._payload_contract = payload_contract
            self._settings_fingerprint = settings
            self._capability_fingerprint = capability
            self._binding_id = binding.binding_id
            self._generation = binding.connection_generation
            snapshot = CaptureCapabilitySnapshot(
                binding.binding_id,
                binding.stable_device_identity,
                binding.connection_generation,
                capability,
                settings,
                payload_contract.fingerprint,
                CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
                ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
                self._max_inflight_frames,
                driver_ring_bytes,
                adapter_record_retention_bytes,
                self._max_blocking_call_seconds,
                self._max_capture_spec_bytes,
                capability_evidence,
            )
            self._description = CameraCaptureDescription(
                _DESCRIPTION_TOKEN,
                self._source_id,
                payload_contract,
                settings,
                physical_facts,
            )
            return snapshot

    def execute_command(self, binding: BoundDevice, command: object) -> object:
        if isinstance(command, PrepareCaptureCommand):
            return self._prepare(binding, command)
        if isinstance(command, StartCaptureCommand):
            return self._start(binding, command)
        if isinstance(command, ReadCaptureCommand):
            return self._read(binding, command)
        if isinstance(command, CompleteCaptureCommand):
            return self._complete(binding, command)
        raise TypeError(f"camera endpoint rejects command {type(command).__name__}")

    def _prepare(
        self,
        binding: BoundDevice,
        command: PrepareCaptureCommand,
    ) -> CapturePreparedAck:
        with self._lock:
            self._validate_binding(binding)
            self._last_prepare_session_id = command.session_id
            if self._session is not None and not self._session.closed:
                raise RuntimeError("camera endpoint already owns an active session")
            payload_contract = self.payload_contract(binding)
            settings = self.settings_fingerprint(binding)
            if canonical_digest(_settings_tree(self._camera)) != settings:
                raise RuntimeError("camera settings changed after capability probe")
            if command.capture_spec_owner_fingerprint != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT:
                raise ValueError("camera capture spec owner differs")
            spec = decode_camera_capture_spec(command.capture_spec_payload)
            if spec.expected_frames != command.expected_total_events:
                raise ValueError("camera spec cardinality differs from prepared run")
            if spec.settings_fingerprint != settings or command.settings_fingerprint != settings:
                raise ValueError("camera capture settings fingerprint differs")
            if command.capability_fingerprint != self._capability_fingerprint:
                raise ValueError("camera capability fingerprint differs")
            if spec.mode is not _camera_mode(self._camera):
                raise ValueError("camera acquisition mode differs from live adapter")
            if spec.mode is CameraAcquisitionMode.FREE_RUNNING:
                raise ValueError(
                    "free-running cameras are monitor sources, not finite exact captures"
                )
            self._session = _EndpointSession(
                command.session_id,
                command.capture_spec_fingerprint,
                spec.expected_frames,
                payload_contract,
                OrderedDatasetMetadataHasher(
                    payload_contract.metadata_contract.fingerprint
                ),
            )
            return CapturePreparedAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                settings,
                self._capability_fingerprint,
                command.capture_spec_fingerprint,
            )

    def _start(
        self,
        binding: BoundDevice,
        command: StartCaptureCommand,
    ) -> CaptureStartedAck:
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if session.started:
                raise RuntimeError("camera session already started")
            expected = session.expected_frames
        self._camera.arm(
            expected,
            max_inflight_frames=min(expected, self._max_inflight_frames),
            timeout=command.timeout_seconds,
        )
        try:
            with self._lock:
                session = self._active_session(binding, command.session_id)
                if (
                    canonical_digest(_settings_tree(self._camera))
                    != self._settings_fingerprint
                ):
                    raise RuntimeError(
                        "camera settings changed between prepare and armed start"
                    )
                session.started = True
        except BaseException as primary:
            try:
                self._camera.finish_record_capture()
            except BaseException as secondary:
                try:
                    primary.add_note(
                        "camera disarm after armed-start validation also failed: "
                        f"{type(secondary).__name__}: {secondary}"
                    )
                except BaseException:
                    pass
            raise
        return CaptureStartedAck(
            command.session_id,
            binding.binding_id,
            binding.connection_generation,
        )

    def _read(
        self,
        binding: BoundDevice,
        command: ReadCaptureCommand,
    ) -> CapturedPayloadAck:
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if not session.started or session.closed:
                raise RuntimeError("camera session is not readable")
            expected_ordinal = session.drained_count
            if expected_ordinal >= session.expected_frames:
                raise RuntimeError("camera session exhausted its finite frame budget")
            payload_contract = session.payload_contract
        records = self._camera.read_frame_records(
            1,
            timeout=command.timeout_seconds,
            exact=True,
        )
        if len(records) != 1:
            raise RuntimeError("camera returned a short read for an exact capture")
        record = records[0]
        payload = self._sample(record, command.session_id, payload_contract)
        if payload.metadata.source_ordinal != expected_ordinal:
            raise RuntimeError(
                f"camera ordinal {payload.metadata.source_ordinal} differs from "
                f"expected {expected_ordinal}"
            )
        metadata_digest = payload_contract.metadata_contract.digest(payload.metadata)
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if session.closed or session.drained_count != expected_ordinal:
                raise RuntimeError("camera session changed during frame read")
            session.metadata_hasher.update(metadata_digest)
            session.drained_count += 1
        return CapturedPayloadAck(
            command.session_id,
            binding.binding_id,
            binding.connection_generation,
            payload,
        )

    @staticmethod
    def _sample(
        record: CameraFrameRecord,
        session_id: str,
        payload_contract: CameraSampleContract,
    ) -> CameraSample:
        if not isinstance(record, CameraFrameRecord):
            raise TypeError("camera adapter returned a non-record payload")
        metadata = CameraFrameMetadata(
            record.source_ordinal,
            record.produced_count,
            record.frame_stamp,
            record.camera_stamp,
            record.timestamp_seconds,
            record.timestamp_microseconds,
            record.host_received_at_ns,
            record.driver_buffer_index,
            f"{session_id}:{record.source_ordinal}",
        )
        sample = CameraSample(
            Value(record.image, VALID, payload_contract.value_schema),
            metadata,
        )
        payload_contract.validate(sample)
        return sample

    def _complete(
        self,
        binding: BoundDevice,
        command: CompleteCaptureCommand,
    ) -> CaptureTerminalAck:
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if not session.started or session.closed:
                raise RuntimeError("camera session cannot complete from its current state")
            if command.expected_total_events != session.expected_frames:
                raise ValueError("terminal frame cardinality differs from prepared session")
            if session.drained_count != session.expected_frames:
                raise RuntimeError("camera cannot complete before every frame is drained")
        terminal = self._camera.finish_record_capture()
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if canonical_digest(_settings_tree(self._camera)) != self._settings_fingerprint:
                raise RuntimeError("camera settings changed during capture")
            session.terminal = terminal
            session.closed = True
            return CaptureTerminalAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                terminal.produced_count,
                session.drained_count,
                terminal.source_stopped,
                terminal.no_more_frames,
                terminal.joined,
                session.metadata_hasher.digest(),
                self._settings_fingerprint,
                self._capability_fingerprint,
                session.spec_fingerprint,
            )

    def close_session(
        self,
        binding: BoundDevice,
        command: SessionCloseCommand,
    ) -> SessionClosedAck:
        with self._lock:
            self._validate_binding(binding)
            session = self._session
            if session is None:
                if command.session_id != self._last_prepare_session_id:
                    raise RuntimeError("camera cleanup session id is unknown")
            elif session.session_id != command.session_id:
                raise RuntimeError("camera cleanup belongs to another session")
            # Always re-enter the idempotent terminal boundary for a known
            # session.  An out-of-band interrupt may already have stopped the
            # source from another thread, but only this run-owner thread can
            # release the RLock acquired by arm().
            needs_terminal_join = session is not None or bool(
                self._camera._recent_state()["armed"]
            )
        terminal = (
            self._camera.finish_record_capture()
            if needs_terminal_join
            else None
        )
        with self._lock:
            session = self._session
            if session is not None:
                if terminal is not None:
                    session.terminal = terminal
                session.closed = True
            state = self._camera._recent_state()
            with state["cond"]:
                stopped = not bool(state["armed"])
                no_more = stopped and not bool(state["pending"])
            return SessionClosedAck(
                command.session_id,
                binding.binding_id,
                binding.connection_generation,
                stopped,
                no_more,
                stopped,
                canonical_digest(
                    {
                        "session_id": command.session_id,
                        "source_stopped": stopped,
                        "no_more_frames": no_more,
                    }
                ),
            )

    def interrupt(self) -> str:
        state = self._camera._recent_state()
        with state["cond"]:
            armed = bool(state["armed"])
        terminal = self._camera.finish_record_capture() if armed else None
        with self._lock:
            if self._session is not None and terminal is not None:
                self._session.terminal = terminal
                self._session.closed = True
        return canonical_digest(
            {
                "operation": "DISARM",
                "source_id": self._source_id,
                "source_stopped": terminal is None or terminal.source_stopped,
            }
        )

    def _validate_binding(self, binding: BoundDevice) -> None:
        _require_binding(binding, "camera", self._binding_id, self._generation)

    def _active_session(
        self,
        binding: BoundDevice,
        session_id: str,
    ) -> _EndpointSession:
        self._validate_binding(binding)
        session = self._session
        if session is None or session.session_id != session_id:
            raise RuntimeError("camera command belongs to another session")
        return session


def bind_camera_measurement(
    device_set: object,
    registry: LegacyDeviceRegistry,
    request: CameraCaptureBindingRequest,
) -> BoundMeasurement:
    """Resolve one installation role into a generation-pinned camera binding."""

    if not isinstance(registry, LegacyDeviceRegistry):
        raise TypeError("registry must be LegacyDeviceRegistry")
    if not isinstance(request, CameraCaptureBindingRequest):
        raise TypeError("request must be CameraCaptureBindingRequest")
    devices = dict(getattr(device_set, "devices", {}) or {})
    try:
        camera = devices[request.role]
    except KeyError as exc:
        raise KeyError(f"camera role {request.role!r} is absent from installation") from exc
    if not isinstance(camera, CameraDevice):
        raise TypeError(f"installation role {request.role!r} is not a camera")
    registration = registry.registration_for(camera)
    target = registration.target_endpoint
    if target is None:
        raise RuntimeError(f"camera role {request.role!r} has no target capture endpoint")
    binding = registry.binding_for(camera)
    attestation = registry.broker.verify_capability(binding)
    description = target.describe(binding)
    if not isinstance(description, CameraCaptureDescription):
        raise TypeError("camera target endpoint returned the wrong description type")
    if description.source_id != request.role:
        raise ValueError("camera endpoint source id differs from installation role")
    capability = attestation.snapshot
    if not isinstance(capability, CaptureCapabilitySnapshot):
        raise TypeError("camera capability attestation has the wrong snapshot type")
    if capability.settings_fingerprint != description.settings_fingerprint:
        raise RuntimeError("camera description and capability settings differ")
    if capability.camera_physical_facts is not description.physical_facts:
        raise RuntimeError(
            "camera description does not share broker-attested physical facts"
        )
    payload_contract = description.payload_contract
    dataset_schema = DatasetSchema(
        request.repeat_axis,
        request.point_axes,
        request.point_layout,
        payload_contract.value_schema,
    )
    expected_cells = request.expected_cells
    capture_spec = freeze_camera_capture_spec(
        CameraCaptureSpec(
            request.mode,
            len(expected_cells),
            description.settings_fingerprint,
        )
    )
    readout_axes = tuple(
        axis for axis in dataset_schema.point_axes if axis.role == READOUT_EVENT
    )
    if len(readout_axes) > 1:
        raise ValueError("camera dataset has multiple READOUT_EVENT axes")
    event_count = 1 if not readout_axes else readout_axes[0].size
    if request.event_settings is None:
        if event_count != 1:
            raise ValueError(
                "multi-event camera capture requires explicit event_settings"
            )
        event_settings = (description.event_setting(0),)
    else:
        event_settings = request.event_settings
    descriptor = description.mint_descriptor(
        dataset_schema,
        event_settings,
        camera_arm_spec_fingerprint=capture_spec.digest,
    )
    camera_provenance = CameraCaptureProvenance(
        descriptor=descriptor,
        binding=ReadoutBindingKey(request.role),
        active_settings_fingerprint=description.settings_fingerprint,
        binding_id=capability.binding_id,
        connection_generation=capability.connection_generation,
        capability_fingerprint=capability.capability_fingerprint,
    )
    capture_contract = CaptureStreamContract(
        StreamId(f"camera.{request.role}.frames"),
        request.role,
        dataset_schema,
        payload_contract,
        CameraDatasetEventAdapter(payload_contract),
        expected_cells,
        capability,
        CaptureRuntimeProfile(
            request.required_consumer_lag_events,
            request.transport_memory_limit_bytes,
        ),
        CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
        camera_provenance,
    )
    definition = MeasurementDefinition(
        DefinitionKey(
            "zlc_neutral_atom",
            f"camera-{request.role}",
        ),
        f"Camera {request.role}",
        "zlc.camera-capture-request",
        "zlc.camera-capture-binding",
        CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
        dataset_schema.fingerprint,
    )
    return BoundMeasurement(
        definition,
        BoundCapturePort(attestation, ()),
        capture_contract,
        capture_spec,
    )


__all__ = [
    "CameraCaptureBindingRequest",
    "CameraCaptureDescription",
    "CameraCaptureEndpoint",
    "bind_camera_measurement",
]
