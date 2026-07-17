"""Composition-owned camera endpoint for the exact neutral-atom runtime."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from zlc_data import (
    AxisSpec,
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
from zlc_neutral_atom.adapter_sdk import (
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_neutral_atom.acquisition import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CAMERA_MEASUREMENT_DEFINITION,
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
    camera_roi_local_spatial_identity,
)
from zlc_neutral_atom.runtime.capture import (
    BoundCapturePort,
    CaptureCapabilitySnapshot,
    CapturePreparedAck,
    CaptureStartedAck,
    CameraCaptureContract,
    CaptureTerminalAck,
    CapturedPayloadAck,
    CompleteCaptureCommand,
    PrepareCaptureCommand,
    ReadCaptureCommand,
    StartCaptureCommand,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellSchedule,
    FrozenDatasetEdge,
    OrderedDatasetMetadataHasher,
)
from zlc_neutral_atom.runtime.pipeline import BoundMeasurement
from zlc_neutral_atom.runtime.monitor import (
    CameraMonitorCapabilitySnapshot,
    CameraMonitorInterrupted,
    CameraMonitorPayloadAck,
    CameraMonitorPreparedAck,
    CameraMonitorStartedAck,
    PrepareCameraMonitorCommand,
    ReadCameraMonitorCommand,
    StartCameraMonitorCommand,
)
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    CleanupStepAck,
    SafeStateAck,
    SafetyOperation,
    SessionClosedAck,
    SessionCloseCommand,
)
from zlc_neutral_atom.runtime.streams import ProducerFlowControl, StreamId
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


def _physical_facts(
    working_point: CameraWorkingPoint,
    binding: BoundDevice,
    source_id: str,
    payload_contract: CameraSampleContract,
) -> CameraPhysicalFacts:
    y_axis, x_axis = payload_contract.value_schema.data_axes
    stable_identity = binding.binding_stamp.physical_identity.stable_device_identity
    facts = CameraPhysicalFacts(
        camera_identity=stable_identity,
        sensor_identity=f"{stable_identity}/sensor",
        optical_path=f"installation-role/{source_id}",
        capture_trigger_channels=working_point.capture_trigger_channels,
        sensor_shape_yx=working_point.sensor_shape_yx,
        roi_origin_yx=working_point.roi_origin_yx,
        roi_shape_yx=working_point.roi_shape_yx,
        binning_yx=working_point.binning_yx,
        spatial_y_axis_id=y_axis.axis_id,
        spatial_x_axis_id=x_axis.axis_id,
        coordinate_frame=y_axis.coordinate_frame,
        dtype=payload_contract.value_schema.dtype,
        count_unit=payload_contract.value_schema.value_unit,
        exposure_seconds=working_point.exposure_seconds,
        required_external_trigger_interval_seconds=(
            working_point.required_external_trigger_interval_seconds
        ),
        external_trigger_integration_start_offset_seconds=(
            working_point.external_trigger_integration_start_offset_seconds
        ),
        gain=working_point.gain,
        readout_mode=working_point.readout_mode,
        opaque_frame_settings_fingerprint=working_point.settings_fingerprint,
    )
    if facts.output_shape_yx != payload_contract.value_schema.data_shape:
        raise RuntimeError(
            "camera physical geometry differs from the frozen payload schema"
        )
    return facts


def _value_schema(working_point: CameraWorkingPoint, source_id: str) -> ValueSchema:
    try:
        height, width = (int(size) for size in working_point.frame_shape_yx)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("camera frame shape is unavailable") from exc
    y_axis_id, x_axis_id, coordinate_frame = camera_roi_local_spatial_identity(
        source_id
    )
    y = AxisSpec(
        y_axis_id,
        "ROI-local output y",
        SPATIAL_Y,
        height,
        tuple(range(height)),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    x = AxisSpec(
        x_axis_id,
        "ROI-local output x",
        SPATIAL_X,
        width,
        tuple(range(width)),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    return ValueSchema(
        (y, x),
        ValidityContract.value(),
        working_point.dtype,
        working_point.count_unit,
    )


@dataclass(frozen=True)
class _AppliedCameraWorkingPoint:
    """One owner-read configuration used by schema, timing, and admission."""

    settings_fingerprint: str
    payload_contract: CameraSampleContract
    physical_facts: CameraPhysicalFacts


@dataclass
class _EndpointSession:
    session_id: str
    spec_fingerprint: str
    expected_frames: int | None
    payload_contract: CameraSampleContract
    metadata_hasher: OrderedDatasetMetadataHasher
    max_inflight_frames: int | None = None
    started: bool = False
    drained_count: int = 0
    terminal: CameraCaptureTerminalRecord | None = None
    terminal_attempt: "_TerminalAttempt | None" = None
    metadata_presence: tuple[bool, bool, bool] | None = None
    last_produced_count: int | None = None
    last_frame_stamp: int | None = None
    last_camera_stamp: int | None = None
    last_captured_at: float | None = None
    closed: bool = False
    superseded: bool = False


@dataclass
class _TerminalAttempt:
    """The one physical terminalization attempt owned by a capture session."""

    done: threading.Event
    outcome: dict[str, object]
    worker: threading.Thread | None = None
    owner_joined: bool = False


def _terminal_proved(record: CameraCaptureTerminalRecord) -> bool:
    return record.source_stopped and record.no_more_frames and record.joined


@dataclass(frozen=True)
class CameraCaptureBindingRequest:
    """Composition request for one finite camera dataset binding."""

    role: str
    repeat_axis: AxisSpec
    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout
    cell_schedule: DatasetCellSchedule
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
        if not isinstance(self.cell_schedule, DatasetCellSchedule):
            raise TypeError("cell_schedule must be DatasetCellSchedule")
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
        camera: CameraAdapter,
        source_id: str,
        *,
        max_source_burst_events: int | None = None,
        max_blocking_call_seconds: float | None = None,
        max_capture_spec_bytes: int = 4096,
        exact_external_trigger_qualification_digest: str | None = None,
        acquisition_mode: CameraAcquisitionMode = CameraAcquisitionMode.EXTERNAL_TRIGGERED,
    ) -> None:
        if not isinstance(camera, CameraAdapter):
            raise TypeError("camera must implement the adapter_sdk CameraAdapter contract")
        self._camera = camera
        self._source_id = _canonical_text(source_id, "source_id")
        adapter_capacity = _positive_int(
            camera.max_pending_records,
            "camera max_pending_records",
        )
        self._adapter_record_capacity = adapter_capacity
        self._max_source_burst_events = _positive_int(
            adapter_capacity
            if max_source_burst_events is None
            else max_source_burst_events,
            "max_source_burst_events",
        )
        if self._max_source_burst_events > adapter_capacity:
            raise ValueError(
                "max_source_burst_events exceeds camera max_pending_records"
            )
        timeout = (
            float(camera.timeout)
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
        if exact_external_trigger_qualification_digest is not None:
            _sha256(
                exact_external_trigger_qualification_digest,
                "exact_external_trigger_qualification_digest",
            )
        self._exact_external_trigger_qualification_digest = (
            exact_external_trigger_qualification_digest
        )
        if not isinstance(acquisition_mode, CameraAcquisitionMode):
            raise TypeError("acquisition_mode must be CameraAcquisitionMode")
        self._acquisition_mode = acquisition_mode
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._capability: CaptureCapabilitySnapshot | None = None
        self._working_point: _AppliedCameraWorkingPoint | None = None
        self._operation_epoch = 0
        self._command_operation_token: object | None = None
        self._physical_operations_inflight = 0
        self._session: _EndpointSession | None = None

    def payload_contract(self, binding: BoundDevice) -> CameraSampleContract:
        with self._lock:
            self._validate_binding(binding)
            payload_contract = self._capability_snapshot().payload_contract
            if not isinstance(payload_contract, CameraSampleContract):
                raise RuntimeError("camera capability payload contract is invalid")
            return payload_contract

    def settings_fingerprint(self, binding: BoundDevice) -> str:
        with self._lock:
            self._validate_binding(binding)
            return self._capability_snapshot().settings_fingerprint

    def capability_probe(self, binding: BoundDevice) -> CaptureCapabilitySnapshot:
        with self._lock:
            if self._session is not None and not self._session.closed:
                raise RuntimeError("cannot probe camera capability during a capture session")
            if self._physical_operations_inflight:
                raise RuntimeError(
                    "cannot probe camera capability while a physical operation is in flight"
                )
            if self._capability is not None:
                self._validate_binding(binding)
            working_point = self._read_working_point(binding)
            settings = working_point.settings_fingerprint
            payload_contract = working_point.payload_contract
            physical_facts = working_point.physical_facts
            retained_frame_bytes = payload_contract.max_retained_nbytes
            driver_ring_bytes = (
                self._max_source_burst_events * retained_frame_bytes
            )
            adapter_record_retention_bytes = (
                self._adapter_record_capacity * retained_frame_bytes
            )
            capability_evidence = CameraCapabilityEvidence(
                adapter_type=(
                    f"{type(self._camera).__module__}."
                    f"{type(self._camera).__qualname__}"
                ),
                source_id=self._source_id,
                payload_contract_fingerprint=payload_contract.fingerprint,
                capture_spec_owner_fingerprint=(
                    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT
                ),
                flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
                max_source_burst_events=self._max_source_burst_events,
                driver_ring_bytes=driver_ring_bytes,
                adapter_record_retention_bytes=(
                    adapter_record_retention_bytes
                ),
                max_blocking_call_seconds=self._max_blocking_call_seconds,
                max_capture_spec_bytes=self._max_capture_spec_bytes,
                physical_facts=physical_facts,
                exact_external_trigger_qualification_digest=(
                    self._exact_external_trigger_qualification_digest
                ),
            )
            stamp = binding.binding_stamp
            snapshot = self._make_capability_snapshot(
                stamp,
                payload_contract,
                capability_evidence,
            )
            self._working_point = working_point
            self._capability = snapshot
            return snapshot

    def _make_capability_snapshot(
        self,
        binding_stamp,
        payload_contract: CameraSampleContract,
        capability_evidence: CameraCapabilityEvidence,
    ) -> CaptureCapabilitySnapshot:
        return CaptureCapabilitySnapshot(
            binding_stamp=binding_stamp,
            payload_contract=payload_contract,
            camera_capability_evidence=capability_evidence,
        )

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
            if self._physical_operations_inflight:
                raise RuntimeError(
                    "camera endpoint still owns an in-flight physical operation"
                )
            if self._session is not None and not self._session.closed:
                raise RuntimeError("camera endpoint already owns an active session")
            capability = self._capability_snapshot()
            payload_contract = self.payload_contract(binding)
            settings = capability.settings_fingerprint
            self._require_working_point_unchanged(binding)
            if command.capture_spec_owner_fingerprint != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT:
                raise ValueError("camera capture spec owner differs")
            spec = decode_camera_capture_spec(command.capture_spec_payload)
            if spec.expected_frames != command.expected_total_events:
                raise ValueError("camera spec cardinality differs from prepared run")
            if spec.settings_fingerprint != settings or command.settings_fingerprint != settings:
                raise ValueError("camera capture settings fingerprint differs")
            if command.capability_fingerprint != capability.capability_fingerprint:
                raise ValueError("camera capability fingerprint differs")
            if spec.mode is not self._acquisition_mode:
                raise ValueError("camera acquisition mode differs from live adapter")
            if self._acquisition_mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
                raise ValueError(
                    "free-running cameras are monitor sources, not finite exact captures"
                )
            evidence = capability.camera_capability_evidence
            if evidence.exact_external_trigger_qualification_digest is None:
                raise ValueError(
                    "real exact capture requires E0-qualified ordered one-frame-per-trigger evidence"
                )
            self._operation_epoch += 1
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
                binding.binding_instance_id,
                settings,
                capability.capability_fingerprint,
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
            max_inflight = min(
                expected,
                self._capability_snapshot()
                .camera_capability_evidence.max_source_burst_events,
            )
            operation_epoch = self._operation_epoch
            operation_token = self._begin_command_operation()
        arm_returned = False
        try:
            try:
                self._camera.arm(
                    expected,
                    # The camera ring retains only the capability-qualified
                    # maximum outstanding burst.  The host exact stream/dataset
                    # retention owns the complete run; allocating that cardinality
                    # again in the driver would violate the preflight byte budget.
                    max_inflight_frames=max_inflight,
                    timeout=command.timeout_seconds,
                )
                arm_returned = True
                with self._lock:
                    self._require_current_operation(
                        binding,
                        session,
                        operation_epoch,
                    )
                    self._require_working_point_unchanged(binding)
                    session.started = True
                    return CaptureStartedAck(
                        command.session_id,
                        binding.binding_instance_id,
                    )
            except BaseException as primary:
                terminal: CameraCaptureTerminalRecord | None = None
                # Once arm() returned, this same owner thread must re-enter the
                # terminal boundary even when an interrupt already stopped the
                # source.  It consumes the cached record and releases arm()'s
                # thread-owned RLock before the operation is declared joined.
                if arm_returned:
                    try:
                        terminal = self._terminalize_with_deadline(
                            session,
                            command.timeout_seconds,
                            require_owner_join=True,
                        )
                    except BaseException as secondary:
                        try:
                            primary.add_note(
                                "camera disarm after armed-start validation also failed: "
                                f"{type(secondary).__name__}: {secondary}"
                            )
                        except BaseException:
                            pass
                with self._lock:
                    if self._session is session:
                        if terminal is not None:
                            session.terminal = terminal
                        session.superseded = True
                raise
        finally:
            self._end_command_operation(operation_token)

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
            operation_epoch = self._operation_epoch
            operation_token = self._begin_command_operation()
        try:
            records = self._camera.read_frame_records(
                1,
                timeout=command.timeout_seconds,
                exact=True,
            )
            with self._lock:
                self._require_current_operation(
                    binding,
                    session,
                    operation_epoch,
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
                self._validate_frame_sequence(session, payload.metadata)
                metadata_digest = payload_contract.metadata_contract.digest(
                    payload.metadata
                )
                if session.drained_count != expected_ordinal:
                    raise RuntimeError("camera session changed during frame read")
                session.metadata_hasher.update(metadata_digest)
                session.drained_count += 1
                return CapturedPayloadAck(
                    command.session_id,
                    binding.binding_instance_id,
                    payload,
                )
        except BaseException:
            with self._lock:
                if self._session is session:
                    session.superseded = True
            raise
        finally:
            self._end_command_operation(operation_token)

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
            operation_epoch = self._operation_epoch
            operation_token = self._begin_command_operation()
        terminal: CameraCaptureTerminalRecord | None = None
        try:
            terminal = self._terminalize_with_deadline(
                session,
                command.timeout_seconds,
                require_owner_join=True,
            )
            with self._lock:
                self._require_current_operation(
                    binding,
                    session,
                    operation_epoch,
                )
                capability = self._capability_snapshot()
                self._require_working_point_unchanged(binding)
                if not _terminal_proved(terminal):
                    raise RuntimeError(
                        "camera terminal readback did not prove stop, drain, and join"
                    )
                session.terminal = terminal
                session.closed = True
                return CaptureTerminalAck(
                    command.session_id,
                    binding.binding_instance_id,
                    terminal.produced_count,
                    session.drained_count,
                    terminal.source_stopped,
                    terminal.no_more_frames,
                    terminal.joined,
                    session.metadata_hasher.digest(),
                    capability.settings_fingerprint,
                    capability.capability_fingerprint,
                    session.spec_fingerprint,
                )
        except BaseException:
            with self._lock:
                if self._session is session:
                    if terminal is not None:
                        session.terminal = terminal
                    session.superseded = True
            raise
        finally:
            self._end_command_operation(operation_token)

    def close_session(
        self,
        binding: BoundDevice,
        command: SessionCloseCommand,
    ) -> SessionClosedAck:
        with self._condition:
            self._validate_binding(binding)
            session = self._session
            if session is None:
                raise RuntimeError("camera cleanup session id is unknown")
            if session.session_id != command.session_id:
                raise RuntimeError("camera cleanup belongs to another session")
            self._operation_epoch += 1
            if session is not None:
                session.superseded = True
            deadline = time.monotonic() + command.timeout_seconds
            while self._physical_operations_inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "camera session physical operation did not join before cleanup timeout"
                    )
                self._condition.wait(remaining)
            # Always re-enter the idempotent terminal boundary for a known
            # session.  An out-of-band interrupt may already have stopped the
            # source from another thread, but only this run-owner thread can
            # release the RLock acquired by arm().
            terminal_budget = deadline - time.monotonic()
            if terminal_budget <= 0.0:
                raise TimeoutError(
                    "camera session exhausted cleanup timeout before terminal readback"
                )
        terminal = self._terminalize_with_deadline(
            session,
            terminal_budget,
            require_owner_join=True,
        )
        with self._lock:
            self._validate_binding(binding)
            if self._session is not session:
                raise RuntimeError("camera cleanup session is no longer current")
            session.terminal = terminal
            session.closed = _terminal_proved(terminal)
            return SessionClosedAck(
                command.session_id,
                binding.binding_instance_id,
                terminal.source_stopped,
                terminal.no_more_frames,
                terminal.joined,
                canonical_digest(
                    {
                        "session_id": command.session_id,
                        "produced_count": terminal.produced_count,
                        "source_stopped": terminal.source_stopped,
                        "no_more_frames": terminal.no_more_frames,
                        "joined": terminal.joined,
                    }
                ),
            )

    def interrupt(self) -> str:
        with self._condition:
            self._operation_epoch += 1
            session = self._session
            if session is not None:
                session.superseded = True
            self._physical_operations_inflight += 1
        terminal: CameraCaptureTerminalRecord | None = None
        try:
            if session is not None and (
                session.started
                or session.terminal_attempt is not None
                or session.terminal is not None
            ):
                terminal = self._terminalize_with_deadline(
                    session,
                    self._max_blocking_call_seconds,
                    require_owner_join=False,
                )
            with self._lock:
                if (
                    self._session is session
                    and session is not None
                    and terminal is not None
                ):
                    session.terminal = terminal
            return canonical_digest(
                {
                    "operation": "DISARM",
                    "source_id": self._source_id,
                    "terminal": (
                        None
                        if terminal is None
                        else {
                            "produced_count": terminal.produced_count,
                            "source_stopped": terminal.source_stopped,
                            "no_more_frames": terminal.no_more_frames,
                            "joined": terminal.joined,
                        }
                    ),
                }
            )
        finally:
            with self._condition:
                self._physical_operations_inflight -= 1
                self._condition.notify_all()

    def cleanup(self) -> CleanupStepAck:
        """Stop the physical source and acknowledge the declared DISARM step."""

        return CleanupStepAck(SafetyOperation.DISARM, self.interrupt())

    def verify_safe_state(self) -> SafeStateAck:
        """Read the adapter-owned armed/pending state without changing it."""

        armed, pending = self._camera.capture_state()
        if armed:
            raise RuntimeError(
                f"camera {self._source_id!r} still owns an armed acquisition"
            )
        if pending:
            raise RuntimeError(
                f"camera {self._source_id!r} still retains pending frames"
            )
        return SafeStateAck(
            canonical_digest(
                {"source_id": self._source_id, "armed": False, "pending": 0}
            )
        )

    def _terminalize_with_deadline(
        self,
        session: _EndpointSession,
        timeout_seconds: float,
        *,
        require_owner_join: bool,
    ) -> CameraCaptureTerminalRecord:
        """Bound an SDK stop without transferring the arm owner's RLock.

        The backend stop runs on an isolated I/O worker.  On timely completion
        this calling thread re-enters the idempotent terminal boundary: when it
        is the original arm owner that second call consumes the cached terminal
        record and releases the thread-owned acquisition RLock.  A timed-out
        worker may finish later, but no acknowledgement is minted and the
        superseded session remains unavailable until a later bounded cleanup
        consumes that frozen result.
        """

        if not isinstance(session, _EndpointSession):
            raise TypeError("camera terminalization requires an endpoint session")
        with self._condition:
            if self._session is not session:
                raise RuntimeError("camera terminalization session is no longer current")
            attempt = session.terminal_attempt
            if attempt is None:
                attempt = _TerminalAttempt(threading.Event(), {})
                session.terminal_attempt = attempt
                self._physical_operations_inflight += 1

                def terminalize() -> None:
                    try:
                        attempt.outcome["terminal"] = (
                            self._camera.finish_record_capture()
                        )
                    except BaseException as error:
                        attempt.outcome["error"] = error
                    finally:
                        attempt.done.set()
                        with self._condition:
                            self._physical_operations_inflight -= 1
                            self._condition.notify_all()

                attempt.worker = threading.Thread(
                    target=terminalize,
                    name=f"camera-terminal-{self._source_id}",
                    daemon=True,
                )
                try:
                    attempt.worker.start()
                except BaseException:
                    # No backend call ran, so this is the one case where the
                    # session may discard the attempt and let a later bounded
                    # cleanup create a new physical terminalization worker.
                    session.terminal_attempt = None
                    self._physical_operations_inflight -= 1
                    self._condition.notify_all()
                    raise
            self._physical_operations_inflight += 1
        try:
            if not attempt.done.wait(float(timeout_seconds)):
                raise TimeoutError(
                    "camera backend terminalization exceeded its blocking-call budget"
                )
            error = attempt.outcome.get("error")
            if isinstance(error, BaseException):
                raise error
            terminal = attempt.outcome.get("terminal")
            if not isinstance(terminal, CameraCaptureTerminalRecord):
                raise RuntimeError("camera terminal worker returned no terminal record")
            if not require_owner_join:
                return terminal
            with self._condition:
                if not attempt.owner_joined:
                    owner_terminal = self._camera.finish_record_capture()
                    if owner_terminal != terminal:
                        raise RuntimeError(
                            "camera terminal readback changed across owner join"
                        )
                    attempt.owner_joined = True
                session.terminal = terminal
                return terminal
        finally:
            with self._condition:
                self._physical_operations_inflight -= 1
                self._condition.notify_all()

    def _validate_binding(self, binding: BoundDevice) -> None:
        capability = self._capability_snapshot()
        _require_binding(
            binding,
            "camera",
            capability.binding_stamp.binding_instance_id,
        )

    def _capability_snapshot(self) -> CaptureCapabilitySnapshot:
        capability = self._capability
        if capability is None:
            raise RuntimeError("camera capability has not been probed")
        return capability

    def _read_working_point(
        self,
        binding: BoundDevice,
    ) -> _AppliedCameraWorkingPoint:
        working_point = self._camera.capture_working_point()
        if not isinstance(working_point, CameraWorkingPoint):
            raise TypeError("camera adapter returned a non-working-point value")
        if working_point.acquisition_mode != self._acquisition_mode.value:
            raise RuntimeError(
                "camera working point acquisition mode differs from its endpoint"
            )
        payload_contract = CameraSampleContract(
            _value_schema(working_point, self._source_id)
        )
        physical_facts = _physical_facts(
            working_point,
            binding,
            self._source_id,
            payload_contract,
        )
        return _AppliedCameraWorkingPoint(
            working_point.settings_fingerprint,
            payload_contract,
            physical_facts,
        )

    def _require_working_point_unchanged(self, binding: BoundDevice) -> None:
        expected = self._working_point
        if expected is None:
            raise RuntimeError("camera working point has not been probed")
        observed = self._read_working_point(binding)
        if observed != expected:
            raise RuntimeError("camera applied working point changed during capture")

    @staticmethod
    def _validate_frame_sequence(
        session: _EndpointSession,
        metadata: CameraFrameMetadata,
    ) -> None:
        presence = (
            metadata.frame_stamp is not None,
            metadata.camera_stamp is not None,
            metadata.timestamp_seconds is not None,
        )
        if session.metadata_presence is None:
            session.metadata_presence = presence
        elif presence != session.metadata_presence:
            raise RuntimeError("camera metadata availability changed within one arm epoch")
        produced = metadata.produced_count
        if produced is not None:
            if produced < metadata.source_ordinal + 1:
                raise RuntimeError("camera produced-count trails the delivered ordinal")
            if (
                session.last_produced_count is not None
                and produced < session.last_produced_count
            ):
                raise RuntimeError("camera produced-count moved backwards")
        for label, current, previous in (
            ("frame stamp", metadata.frame_stamp, session.last_frame_stamp),
            ("camera stamp", metadata.camera_stamp, session.last_camera_stamp),
        ):
            if current is not None and previous is not None and current <= previous:
                raise RuntimeError(f"camera {label} is not strictly increasing")
        captured_at = metadata.captured_at
        if session.last_captured_at is not None and captured_at < session.last_captured_at:
            raise RuntimeError("camera capture timestamp moved backwards")
        session.last_produced_count = produced
        session.last_frame_stamp = metadata.frame_stamp
        session.last_camera_stamp = metadata.camera_stamp
        session.last_captured_at = captured_at

    def _begin_command_operation(self) -> object:
        if self._command_operation_token is not None:
            raise RuntimeError(
                "camera endpoint already owns an in-flight physical command"
            )
        token = object()
        self._command_operation_token = token
        self._physical_operations_inflight += 1
        return token

    def _end_command_operation(self, token: object) -> None:
        with self._condition:
            if self._command_operation_token is not token:
                raise RuntimeError("camera endpoint physical-operation token differs")
            self._command_operation_token = None
            self._physical_operations_inflight -= 1
            self._condition.notify_all()

    def _require_current_operation(
        self,
        binding: BoundDevice,
        session: _EndpointSession,
        operation_epoch: int,
    ) -> None:
        self._validate_binding(binding)
        if (
            self._session is not session
            or self._operation_epoch != operation_epoch
            or session.superseded
            or session.closed
        ):
            raise RuntimeError("camera endpoint operation was superseded")

    def _active_session(
        self,
        binding: BoundDevice,
        session_id: str,
    ) -> _EndpointSession:
        self._validate_binding(binding)
        session = self._session
        if session is None or session.session_id != session_id:
            raise RuntimeError("camera command belongs to another session")
        if session.superseded:
            raise RuntimeError("camera session was superseded")
        return session


class CameraMonitorEndpoint(CameraCaptureEndpoint):
    """The FREE_RUNNING command face of the shared camera endpoint owner.

    It deliberately has no exact prepare/read/complete commands.  The adapter
    owns exposure cadence; the Run owner merely drains ordered records until
    cancellation invokes the same bounded DISARM/session-close recipe used by
    finite capture.
    """

    def __init__(
        self,
        camera: CameraAdapter,
        source_id: str,
        *,
        max_source_burst_events: int | None = None,
        max_blocking_call_seconds: float | None = None,
        max_capture_spec_bytes: int = 4096,
    ) -> None:
        super().__init__(
            camera,
            source_id,
            max_source_burst_events=max_source_burst_events,
            max_blocking_call_seconds=max_blocking_call_seconds,
            max_capture_spec_bytes=max_capture_spec_bytes,
            acquisition_mode=CameraAcquisitionMode.FREE_RUNNING,
        )

    def _make_capability_snapshot(
        self,
        binding_stamp,
        payload_contract: CameraSampleContract,
        capability_evidence: CameraCapabilityEvidence,
    ) -> CameraMonitorCapabilitySnapshot:
        return CameraMonitorCapabilitySnapshot(
            binding_stamp=binding_stamp,
            payload_contract=payload_contract,
            camera_capability_evidence=capability_evidence,
            acquisition_mode=CameraAcquisitionMode.FREE_RUNNING,
        )

    def execute_command(self, binding: BoundDevice, command: object) -> object:
        if isinstance(command, PrepareCameraMonitorCommand):
            return self._prepare_monitor(binding, command)
        if isinstance(command, StartCameraMonitorCommand):
            return self._start_monitor(binding, command)
        if isinstance(command, ReadCameraMonitorCommand):
            return self._read_monitor(binding, command)
        raise TypeError(f"camera monitor endpoint rejects command {type(command).__name__}")

    def _prepare_monitor(
        self,
        binding: BoundDevice,
        command: PrepareCameraMonitorCommand,
    ) -> CameraMonitorPreparedAck:
        if command.timeout_seconds > self._max_blocking_call_seconds:
            raise ValueError("camera monitor timeout exceeds the endpoint blocking bound")
        with self._lock:
            self._validate_binding(binding)
            if self._physical_operations_inflight:
                raise RuntimeError("camera monitor still owns a physical operation")
            if self._session is not None and not self._session.closed:
                raise RuntimeError("camera endpoint already owns an active session")
            capability = self._capability_snapshot()
            if not isinstance(capability, CameraMonitorCapabilitySnapshot):
                raise TypeError("camera endpoint has no monitor capability")
            self._require_working_point_unchanged(binding)
            if command.settings_fingerprint != capability.settings_fingerprint:
                raise ValueError("camera monitor settings fingerprint differs")
            if command.capability_fingerprint != capability.capability_fingerprint:
                raise ValueError("camera monitor capability fingerprint differs")
            if command.max_inflight_frames > capability.max_source_burst_events:
                raise ValueError("camera monitor inflight budget exceeds capability")
            payload_contract = self.payload_contract(binding)
            self._operation_epoch += 1
            self._session = _EndpointSession(
                command.session_id,
                capability.capability_fingerprint,
                None,
                payload_contract,
                OrderedDatasetMetadataHasher(
                    payload_contract.metadata_contract.fingerprint
                ),
                max_inflight_frames=command.max_inflight_frames,
            )
            return CameraMonitorPreparedAck(
                command.session_id,
                binding.binding_instance_id,
                capability.settings_fingerprint,
                capability.capability_fingerprint,
            )

    def _start_monitor(
        self,
        binding: BoundDevice,
        command: StartCameraMonitorCommand,
    ) -> CameraMonitorStartedAck:
        if command.timeout_seconds > self._max_blocking_call_seconds:
            raise ValueError("camera monitor timeout exceeds the endpoint blocking bound")
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if session.started:
                raise RuntimeError("camera monitor session already started")
            max_inflight = session.max_inflight_frames
            if max_inflight is None:
                raise RuntimeError("camera monitor has no inflight budget")
            operation_epoch = self._operation_epoch
            operation_token = self._begin_command_operation()
        armed = False
        try:
            try:
                self._camera.arm(
                    None,
                    max_inflight_frames=max_inflight,
                    timeout=command.timeout_seconds,
                )
                armed = True
                with self._lock:
                    self._require_current_operation(binding, session, operation_epoch)
                    self._require_working_point_unchanged(binding)
                    session.started = True
                    return CameraMonitorStartedAck(
                        command.session_id,
                        binding.binding_instance_id,
                    )
            except BaseException as primary:
                if armed:
                    try:
                        terminal = self._terminalize_with_deadline(
                            session,
                            command.timeout_seconds,
                            require_owner_join=True,
                        )
                        with self._lock:
                            if self._session is session:
                                session.terminal = terminal
                    except BaseException as secondary:
                        try:
                            primary.add_note(
                                "camera monitor stop after start failure also failed: "
                                f"{type(secondary).__name__}: {secondary}"
                            )
                        except BaseException:
                            pass
                with self._lock:
                    if self._session is session:
                        session.superseded = True
                raise
        finally:
            self._end_command_operation(operation_token)

    def _read_monitor(
        self,
        binding: BoundDevice,
        command: ReadCameraMonitorCommand,
    ) -> CameraMonitorPayloadAck:
        if command.timeout_seconds > self._max_blocking_call_seconds:
            raise ValueError("camera monitor timeout exceeds the endpoint blocking bound")
        with self._lock:
            session = self._active_session(binding, command.session_id)
            if not session.started or session.closed:
                raise RuntimeError("camera monitor session is not readable")
            expected_ordinal = session.drained_count
            payload_contract = session.payload_contract
            operation_epoch = self._operation_epoch
            operation_token = self._begin_command_operation()
        try:
            records = self._camera.read_frame_records(
                1,
                timeout=command.timeout_seconds,
                exact=False,
            )
            with self._lock:
                self._require_current_operation(binding, session, operation_epoch)
                if len(records) != 1:
                    raise RuntimeError("camera monitor returned a short read")
                payload = self._sample(records[0], command.session_id, payload_contract)
                if payload.metadata.source_ordinal != expected_ordinal:
                    raise RuntimeError(
                        "camera monitor lost or reordered a pre-broker record: "
                        f"got {payload.metadata.source_ordinal}, expected {expected_ordinal}"
                    )
                produced = payload.metadata.produced_count
                if (
                    produced is not None
                    and session.last_produced_count is not None
                    and produced != session.last_produced_count + 1
                ):
                    raise RuntimeError(
                        "camera monitor produced-count gap proves pre-broker loss"
                    )
                # Frame/camera stamps have adapter-specific scales and rollover
                # rules; the generic record contract only promises monotonicity.
                # Contiguous delivery is proved here by the owner-assigned source
                # ordinal and, when available, produced_count.  A real adapter
                # may promote unit-step stamp semantics only after qualification.
                self._validate_frame_sequence(session, payload.metadata)
                if session.drained_count != expected_ordinal:
                    raise RuntimeError("camera monitor session changed during frame read")
                session.drained_count += 1
                return CameraMonitorPayloadAck(
                    command.session_id,
                    binding.binding_instance_id,
                    payload,
                )
        except BaseException as error:
            with self._lock:
                interrupted = (
                    self._session is not session
                    or session.superseded
                    or self._operation_epoch != operation_epoch
                )
                if self._session is session:
                    session.superseded = True
            if interrupted:
                raise CameraMonitorInterrupted(
                    "camera monitor read was superseded by an external interrupt"
                ) from error
            raise
        finally:
            self._end_command_operation(operation_token)


def bind_camera_measurement(
    port: BoundCapturePort,
    request: CameraCaptureBindingRequest,
) -> BoundMeasurement:
    """Bind one already-resolved camera Port to a finite dataset request.

    Role resolution and raw-device ownership belong to the installation
    composition root; this binder receives only the resolved typed Port.
    """

    if not isinstance(port, BoundCapturePort):
        raise TypeError("port must be BoundCapturePort")
    if not isinstance(request, CameraCaptureBindingRequest):
        raise TypeError("request must be CameraCaptureBindingRequest")
    capability = port.capability
    evidence = capability.camera_capability_evidence
    if evidence.source_id != request.role:
        raise ValueError("camera endpoint source id differs from installation role")
    payload_contract = capability.payload_contract
    if not isinstance(payload_contract, CameraSampleContract):
        raise TypeError("camera capability payload contract has the wrong type")
    facts = evidence.physical_facts
    dataset_schema = DatasetSchema(
        request.repeat_axis,
        request.point_axes,
        request.point_layout,
        payload_contract.value_schema,
    )
    cell_schedule = request.cell_schedule
    capture_spec = freeze_camera_capture_spec(
        CameraCaptureSpec(
            request.mode,
            len(cell_schedule),
            evidence.settings_fingerprint,
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
        event_settings = (facts.event_setting(0),)
    else:
        event_settings = request.event_settings
    expected_indices = (0,) if not readout_axes else tuple(range(event_count))
    if tuple(item.event_index for item in event_settings) != expected_indices:
        raise ValueError(
            "event_settings must explicitly cover every READOUT_EVENT index"
        )
    for setting in event_settings:
        if setting != facts.event_setting(setting.event_index):
            raise ValueError(
                "event setting differs from broker-attested camera settings"
            )
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
        readout_event_axis_id=(
            None if not readout_axes else readout_axes[0].axis_id
        ),
        event_settings=event_settings,
        camera_arm_spec_fingerprint=_sha256(
            capture_spec.digest,
            "camera_arm_spec_fingerprint",
        ),
    )
    camera_provenance = CameraCaptureProvenance(
        descriptor=descriptor,
        binding=ReadoutBindingKey(request.role),
        binding_stamp=capability.binding_stamp,
        capability_fingerprint=capability.capability_fingerprint,
    )
    capture_contract = CameraCaptureContract(
        stream_id=StreamId(f"camera.{request.role}.frames"),
        dataset_edge=FrozenDatasetEdge(
            dataset_schema,
            CameraDatasetEventAdapter(payload_contract),
            cell_schedule,
        ),
        capability=capability,
        required_consumer_lag_events=request.required_consumer_lag_events,
        transport_memory_limit_bytes=request.transport_memory_limit_bytes,
        camera_provenance=camera_provenance,
    )
    return BoundMeasurement(
        CAMERA_MEASUREMENT_DEFINITION,
        port,
        capture_contract,
        capture_spec,
    )


__all__ = [
    "CameraCaptureBindingRequest",
    "CameraCaptureEndpoint",
    "CameraMonitorEndpoint",
    "bind_camera_measurement",
]
