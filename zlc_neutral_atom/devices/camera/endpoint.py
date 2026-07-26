"""Composition-owned camera endpoint for the exact neutral-atom runtime."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

import numpy as np
from zlc_data import (
    AxisSpec,
    SPATIAL_X,
    SPATIAL_Y,
    VALID,
    ValidityContract,
    Value,
    ValueSchema,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
    camera_roi_local_spatial_identity,
)
from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
    CameraAcquisitionMode,
    CameraFrameMetadata,
    CameraSample,
    CameraSampleContract,
    decode_camera_capture_spec,
)
from zlc_neutral_atom.devices.camera.capture_port import (
    CameraExposureConfiguredAck,
    CaptureCapabilitySnapshot,
    CapturePreparedAck,
    CaptureStartedAck,
    CaptureTerminalAck,
    CapturedPayloadAck,
    CompleteCaptureCommand,
    ConfigureCameraExposureCommand,
    PrepareCaptureCommand,
    ReadCaptureCommand,
    StartCaptureCommand,
)
from zlc_neutral_atom.runtime.dataset import (
    OrderedDatasetMetadataHasher,
)
from zlc_neutral_atom.devices.camera.monitor import (
    CameraMonitorCapabilitySnapshot,
    CameraMonitorInterrupted,
    CameraMonitorNoFrameAck,
    CameraMonitorPayloadAck,
    CameraMonitorPreparedAck,
    CameraMonitorStartedAck,
    PrepareCameraMonitorCommand,
    ReadCameraMonitorCommand,
    StartCameraMonitorCommand,
)
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SessionClosedAck,
    SessionCloseCommand,
    require_current_endpoint_binding as _require_binding,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCapabilityEvidence,
    CameraPhysicalFacts,
)
from zlc_storage import (
    canonical_digest,
    canonical_text as _canonical_text,
    sha256_text as _sha256,
)


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
    source_group_sizes: tuple[int, ...] | None
    payload_contract: CameraSampleContract
    metadata_hasher: OrderedDatasetMetadataHasher
    buffer_frame_count: int | None = None
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
class _ExposureLease:
    session_id: str
    baseline_working_point: _AppliedCameraWorkingPoint


@dataclass
class _TerminalAttempt:
    """The one physical terminalization attempt owned by a capture session."""

    done: threading.Event
    outcome: dict[str, object]
    worker: threading.Thread | None = None
    owner_joined: bool = False


def _terminal_proved(record: CameraCaptureTerminalRecord) -> bool:
    return record.source_stopped and record.no_more_frames and record.joined


class CameraCaptureEndpoint:
    """One raw camera's typed command endpoint, private to composition."""

    def __init__(
        self,
        camera: CameraAdapter,
        source_id: str,
        *,
        max_blocking_call_seconds: float | None = None,
        exact_external_trigger_qualification_digest: str | None = None,
        acquisition_mode: CameraAcquisitionMode = CameraAcquisitionMode.EXTERNAL_TRIGGERED,
    ) -> None:
        if not isinstance(camera, CameraAdapter):
            raise TypeError("camera must implement the CameraAdapter contract")
        self._camera = camera
        self._source_id = _canonical_text(source_id, "source_id")
        timeout = (
            float(camera.timeout)
            if max_blocking_call_seconds is None
            else float(max_blocking_call_seconds)
        )
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("max_blocking_call_seconds must be finite and positive")
        self._max_blocking_call_seconds = timeout
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
        self._exposure_lease: _ExposureLease | None = None

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
            if self._exposure_lease is not None:
                raise RuntimeError("cannot probe camera capability during an exposure lease")
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
                max_blocking_call_seconds=self._max_blocking_call_seconds,
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
        if isinstance(command, ConfigureCameraExposureCommand):
            return self._configure_exposure(binding, command)
        if isinstance(command, PrepareCaptureCommand):
            return self._prepare(binding, command)
        if isinstance(command, StartCaptureCommand):
            return self._start(binding, command)
        if isinstance(command, ReadCaptureCommand):
            return self._read(binding, command)
        if isinstance(command, CompleteCaptureCommand):
            return self._complete(binding, command)
        raise TypeError(f"camera endpoint rejects command {type(command).__name__}")

    def _configure_exposure(
        self,
        binding: BoundDevice,
        command: ConfigureCameraExposureCommand,
    ) -> CameraExposureConfiguredAck:
        with self._lock:
            self._validate_binding(binding)
            if self._physical_operations_inflight:
                raise RuntimeError("camera still owns an in-flight physical operation")
            if self._session is not None and not self._session.closed:
                raise RuntimeError("camera exposure cannot change during an active arm")
            lease = self._exposure_lease
            if lease is None:
                baseline = self._working_point
                if baseline is None:
                    raise RuntimeError("camera working point has not been probed")
                lease = _ExposureLease(command.session_id, baseline)
                self._exposure_lease = lease
            elif lease.session_id != command.session_id:
                raise RuntimeError("camera already owns another exposure lease")
            baseline = lease.baseline_working_point
            if command.baseline_settings_fingerprint != baseline.settings_fingerprint:
                raise ValueError("camera exposure baseline fingerprint differs")
            configure = getattr(self._camera, "configure_exposure_seconds", None)
            token = self._begin_command_operation()
        try:
            if not callable(configure):
                raise RuntimeError(
                    "camera adapter does not implement exposure configure/readback"
                )
            configure(command.exposure_seconds)
            observed = self._read_working_point(binding)
            if not np.isclose(
                observed.physical_facts.exposure_seconds,
                command.exposure_seconds,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "camera applied exposure differs from the requested duration"
                )
            if observed.payload_contract != baseline.payload_contract:
                raise RuntimeError(
                    "camera exposure change altered the frame payload contract"
                )
            # Exposure does not change the frame schema.  Retain the exact
            # broker-attested payload owner so a run-scoped exposure lease can
            # be consumed by the already-admitted Dataset/Figure topology.
            observed = replace(
                observed,
                payload_contract=baseline.payload_contract,
            )
            expected_facts = replace(
                baseline.physical_facts,
                exposure_seconds=observed.physical_facts.exposure_seconds,
                required_external_trigger_interval_seconds=(
                    observed.physical_facts
                    .required_external_trigger_interval_seconds
                ),
                opaque_frame_settings_fingerprint=(
                    observed.settings_fingerprint
                ),
            )
            if observed.physical_facts != expected_facts:
                raise RuntimeError(
                    "camera exposure command altered another physical working-point fact"
                )
            evidence = replace(
                self._capability_snapshot().camera_capability_evidence,
                physical_facts=observed.physical_facts,
            )
            capability = self._make_capability_snapshot(
                binding.binding_stamp,
                observed.payload_contract,
                evidence,
            )
            with self._lock:
                if self._exposure_lease is not lease:
                    raise RuntimeError("camera exposure lease was superseded")
                self._working_point = observed
                self._capability = capability
            required_interval = (
                observed.physical_facts
                .required_external_trigger_interval_seconds
            )
            if required_interval is None:
                raise RuntimeError(
                    "camera exposure readback lacks an external-trigger interval"
                )
            return CameraExposureConfiguredAck(
                command.session_id,
                binding.binding_instance_id,
                command.exposure_seconds,
                observed.physical_facts.exposure_seconds,
                required_interval,
                observed.settings_fingerprint,
                capability.capability_fingerprint,
                capability,
            )
        finally:
            self._end_command_operation(token)

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
                spec.source_group_sizes,
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
            if expected is None:
                raise RuntimeError("finite camera session lost its frame cardinality")
            operation_epoch = self._operation_epoch
            operation_token = self._begin_command_operation()
        arm_returned = False
        try:
            try:
                self._camera.arm(
                    expected,
                    source_group_sizes=session.source_group_sizes,
                    buffer_frame_count=expected,
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
                raise RuntimeError("camera session exhausted its expected frame count")
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
            lease = self._exposure_lease
            if lease is not None and lease.session_id == command.session_id:
                return self._close_exposure_lease(binding, command, lease)
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
            terminal_seconds = deadline - time.monotonic()
            if terminal_seconds <= 0.0:
                raise TimeoutError(
                    "camera session exhausted cleanup timeout before terminal readback"
                )
        terminal = self._terminalize_with_deadline(
            session,
            terminal_seconds,
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

    def _close_exposure_lease(
        self,
        binding: BoundDevice,
        command: SessionCloseCommand,
        lease: _ExposureLease,
    ) -> SessionClosedAck:
        """Restore the leased baseline through the existing cleanup lane."""

        if self._session is not None and not self._session.closed:
            raise RuntimeError(
                "camera exposure lease cannot close before its capture session"
            )
        deadline = time.monotonic() + command.timeout_seconds
        while self._physical_operations_inflight:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "camera exposure lease did not reach an idle adapter"
                )
            self._condition.wait(remaining)
        configure = getattr(self._camera, "configure_exposure_seconds", None)
        baseline = lease.baseline_working_point
        if callable(configure):
            configure(baseline.physical_facts.exposure_seconds)
        observed = self._read_working_point(binding)
        if observed != baseline:
            raise RuntimeError(
                "camera did not restore the leased physical working point"
            )
        evidence = replace(
            self._capability_snapshot().camera_capability_evidence,
            physical_facts=baseline.physical_facts,
        )
        capability = self._make_capability_snapshot(
            binding.binding_stamp,
            baseline.payload_contract,
            evidence,
        )
        self._working_point = baseline
        self._capability = capability
        self._exposure_lease = None
        return SessionClosedAck(
            command.session_id,
            binding.binding_instance_id,
            True,
            True,
            True,
            canonical_digest(
                {
                    "session_id": command.session_id,
                    "operation": "restore-camera-exposure",
                    "settings_fingerprint": baseline.settings_fingerprint,
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
                    "camera backend terminalization exceeded its blocking-call deadline"
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
    """The display-only command face of the shared camera endpoint owner.

    The adapter owns exposure cadence for a FREE_RUNNING camera.  For an
    EXTERNAL_TRIGGERED camera, independent hardware owns trigger timing and the
    monitor merely drains the resulting ordered records.  Exact capture
    commands remain available through ``CameraCaptureEndpoint`` and retain all
    of their qualification and finite-cardinality checks.
    """

    def __init__(
        self,
        camera: CameraAdapter,
        source_id: str,
        *,
        max_blocking_call_seconds: float | None = None,
        exact_external_trigger_qualification_digest: str | None = None,
        acquisition_mode: CameraAcquisitionMode = CameraAcquisitionMode.FREE_RUNNING,
        monitor_acquisition_mode: CameraAcquisitionMode | None = None,
    ) -> None:
        super().__init__(
            camera,
            source_id,
            max_blocking_call_seconds=max_blocking_call_seconds,
            exact_external_trigger_qualification_digest=(
                exact_external_trigger_qualification_digest
            ),
            acquisition_mode=acquisition_mode,
        )
        if monitor_acquisition_mode is None:
            monitor_acquisition_mode = acquisition_mode
        if not isinstance(monitor_acquisition_mode, CameraAcquisitionMode):
            raise TypeError(
                "monitor_acquisition_mode must be CameraAcquisitionMode"
            )
        self._monitor_acquisition_mode = monitor_acquisition_mode

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
            acquisition_mode=self._monitor_acquisition_mode,
        )

    def execute_command(self, binding: BoundDevice, command: object) -> object:
        if isinstance(command, PrepareCameraMonitorCommand):
            return self._prepare_monitor(binding, command)
        if isinstance(command, StartCameraMonitorCommand):
            return self._start_monitor(binding, command)
        if isinstance(command, ReadCameraMonitorCommand):
            return self._read_monitor(binding, command)
        return super().execute_command(binding, command)

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
            payload_contract = self.payload_contract(binding)
            self._operation_epoch += 1
            self._session = _EndpointSession(
                command.session_id,
                capability.capability_fingerprint,
                None,
                None,
                payload_contract,
                OrderedDatasetMetadataHasher(
                    payload_contract.metadata_contract.fingerprint
                ),
                buffer_frame_count=command.buffer_frame_count,
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
            buffer_frame_count = session.buffer_frame_count
            if buffer_frame_count is None:
                raise RuntimeError("camera monitor has no physical buffer geometry")
            operation_epoch = self._operation_epoch
            operation_token = self._begin_command_operation()
        armed = False
        try:
            try:
                self._camera.arm(
                    None,
                    source_group_sizes=None,
                    buffer_frame_count=buffer_frame_count,
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
    ) -> CameraMonitorPayloadAck | CameraMonitorNoFrameAck:
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
                if not records:
                    capability = self._capability_snapshot()
                    if (
                        not isinstance(capability, CameraMonitorCapabilitySnapshot)
                        or capability.acquisition_mode
                        is not CameraAcquisitionMode.EXTERNAL_TRIGGERED
                    ):
                        raise RuntimeError(
                            "free-running camera monitor produced no frame before "
                            "its hardware call deadline"
                        )
                    return CameraMonitorNoFrameAck(
                        command.session_id,
                        binding.binding_instance_id,
                    )
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


__all__ = [
    "CameraCaptureEndpoint",
    "CameraMonitorEndpoint",
]
