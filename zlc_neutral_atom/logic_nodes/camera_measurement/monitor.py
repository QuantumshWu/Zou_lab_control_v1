"""Continuous application branch of Camera Measurement.

The monitor publishes one atomic latest camera cycle.  Figure selection, ROI
projection, fitting, and other display analysis deliberately do not live here:
they are independent per-panel view branches over the public frame outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import uuid
from typing import Callable, Protocol

import numpy as np

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DatasetSchema,
    PointTable,
    READOUT_EVENT,
    REPEAT,
    Valid,
    ValidityContract,
    ValidityMode,
    Value,
    ValuePayloadContract,
    ValueSchema,
)
from zlc_data.value import expand_component_validity
from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraFrameMetadata,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CAMERA_MEASUREMENT_KEY,
    CameraMeasurementRequest,
    project_camera_monitor_outputs,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.signal_source import (
    CameraAssociatedSignalEventSource,
    CameraSignalAssociationAuthority,
    camera_signal_event_source,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.output_binding import (
    CameraFrameOutputBinding,
)
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.dataset_output import LiveDatasetOutput
from zlc_neutral_atom.runtime._failure import safe_error_summary
from zlc_neutral_atom.runtime.cancellation import CancellationRequested
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.cleanup import run_cleanup_steps
from zlc_neutral_atom.runtime.dataset import (
    FrozenDatasetEdge,
    MonitorDataset,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.devices.camera.monitor import (
    BoundCameraMonitorPort,
    CameraMonitorInterrupted,
    CameraMonitorNoFrameAck,
    CameraMonitorPayloadAck,
    CameraMonitorPreparedAck,
    CameraMonitorStartedAck,
    PrepareCameraMonitorCommand,
    ReadCameraMonitorCommand,
    StartCameraMonitorCommand,
)
from zlc_neutral_atom.devices.camera.capture_port import (
    configure_camera_exposure,
)
from zlc_neutral_atom.runtime.run import RunContext, RunHandle, RunPlan
from zlc_neutral_atom.runtime.signal_source import (
    SignalAssociationScheduleRequirement,
    SignalEventAssociationCursor,
    SignalEventCursor,
)
from zlc_neutral_atom.runtime.streams import (
    AcquisitionProducer,
    AcquisitionStream,
    EventRef,
    StreamError,
    StreamId,
)


_MONITOR_REPEAT_AXIS_ID = AxisId("camera-monitor.repeat")
_MONITOR_READOUT_EVENT_AXIS_ID = AxisId("camera-monitor.readout-event")


@dataclass(frozen=True)
class _CameraCyclePayload:
    value: Value
    metadata: tuple[CameraFrameMetadata, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.value, Value):
            raise TypeError("camera cycle value must be Value")
        metadata = tuple(self.metadata)
        if any(not isinstance(item, CameraFrameMetadata) for item in metadata):
            raise TypeError("camera cycle metadata must contain CameraFrameMetadata")
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class _CameraCycleMetadataContract:
    frame_count: int

    def snapshot(
        self,
        payload: _CameraCyclePayload,
    ) -> tuple[CameraFrameMetadata, ...]:
        if not isinstance(payload, _CameraCyclePayload):
            raise TypeError("camera cycle metadata requires _CameraCyclePayload")
        self.validate(payload.metadata)
        return payload.metadata

    def validate(self, metadata: object) -> None:
        if not isinstance(metadata, tuple) or len(metadata) != self.frame_count:
            raise ValueError("camera cycle metadata cardinality differs")
        if any(not isinstance(item, CameraFrameMetadata) for item in metadata):
            raise TypeError("camera cycle metadata contains an invalid frame")
        if any(not np.isfinite(item.captured_at) for item in metadata):
            raise ValueError("camera cycle captured_at values must be finite")


@dataclass(frozen=True)
class _CameraCycleContract:
    frame_contract: CameraSampleContract
    value_schema: ValueSchema
    metadata_contract: _CameraCycleMetadataContract

    def __post_init__(self) -> None:
        if not isinstance(self.frame_contract, CameraSampleContract):
            raise TypeError("frame_contract must be CameraSampleContract")
        if not isinstance(self.value_schema, ValueSchema):
            raise TypeError("value_schema must be ValueSchema")
        if not isinstance(self.metadata_contract, _CameraCycleMetadataContract):
            raise TypeError("metadata_contract has an invalid type")

    def snapshot(self, payload: _CameraCyclePayload) -> _CameraCyclePayload:
        self.validate(payload)
        return payload

    def validate(self, payload: _CameraCyclePayload) -> None:
        if not isinstance(payload, _CameraCyclePayload):
            raise TypeError("camera cycle contract requires _CameraCyclePayload")
        ValuePayloadContract(self.value_schema).validate(payload.value)
        self.metadata_contract.validate(payload.metadata)


@dataclass(frozen=True)
class _CameraCycleDatasetEventAdapter:
    payload_contract: _CameraCycleContract

    @property
    def value_schema(self) -> ValueSchema:
        return self.payload_contract.value_schema

    @property
    def metadata_contract(self) -> _CameraCycleMetadataContract:
        return self.payload_contract.metadata_contract

    def value(self, payload: _CameraCyclePayload) -> Value:
        return payload.value


def _camera_cycle_edge(
    frame_contract: CameraSampleContract,
    frames_per_cycle: int,
) -> tuple[FrozenDatasetEdge[_CameraCyclePayload], _CameraCycleContract]:
    event_axis = AxisSpec(
        _MONITOR_READOUT_EVENT_AXIS_ID,
        "readout event within latest camera cycle",
        READOUT_EVENT,
        frames_per_cycle,
        tuple(range(frames_per_cycle)),
    )
    frame_schema = frame_contract.value_schema
    frame_component_ids = frame_schema.validity_contract.component_axis_ids
    cycle_schema = ValueSchema(
        (event_axis, *frame_schema.data_axes),
        ValidityContract.components(event_axis.axis_id, *frame_component_ids),
        frame_schema.dtype,
        frame_schema.value_unit,
    )
    contract = _CameraCycleContract(
        frame_contract,
        cycle_schema,
        _CameraCycleMetadataContract(frames_per_cycle),
    )
    dataset_schema = DatasetSchema(
        AxisSpec(
            _MONITOR_REPEAT_AXIS_ID,
            "monitor repeat",
            REPEAT,
            1,
            (0,),
        ),
        PointTable(1),
        None,
        cycle_schema,
    )
    return (
        FrozenDatasetEdge(
            dataset_schema,
            _CameraCycleDatasetEventAdapter(contract),
        ),
        contract,
    )


def _camera_cycle_payload(
    samples: tuple[CameraSample, ...],
    contract: _CameraCycleContract,
) -> _CameraCyclePayload:
    if len(samples) != contract.metadata_contract.frame_count:
        raise ValueError("camera cycle sample cardinality differs")
    for sample in samples:
        contract.frame_contract.validate(sample)
    frame_schema = contract.frame_contract.value_schema
    values = np.stack(tuple(sample.image.values for sample in samples), axis=0)
    if frame_schema.validity_contract.mode is ValidityMode.VALUE:
        validity = np.asarray(
            tuple(isinstance(sample.image.validity, Valid) for sample in samples),
            dtype=np.bool_,
        )
    else:
        validity = np.stack(
            tuple(
                expand_component_validity(sample.image.validity, frame_schema)
                for sample in samples
            ),
            axis=0,
        )
    cycle_value = Value(
        values,
        ComponentValidity(
            contract.value_schema.validity_contract.component_axis_ids,
            validity,
        ),
        contract.value_schema,
    )
    return _CameraCyclePayload(
        cycle_value,
        tuple(sample.metadata for sample in samples),
    )


@dataclass(frozen=True)
class CameraMonitorViewSpec:
    """One admitted latest-cycle cell with declared frame cardinality."""

    block_id: BlockId
    dataset_edge: FrozenDatasetEdge[CameraSample]

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.dataset_edge, FrozenDatasetEdge):
            raise TypeError("dataset_edge must be FrozenDatasetEdge")
        if self.dataset_edge.cell_schedule is not None:
            raise ValueError("camera monitor view requires a schedule-free dataset edge")


class CameraMonitorViewPort(Protocol):
    """Workbench-owned sink; binding transfers the live dataset lifetime."""

    @property
    def spec(self) -> CameraMonitorViewSpec: ...

    @property
    def terminal(self) -> bool: ...

    def bind(
        self,
        dataset: "CameraMonitorLiveDataset",
    ) -> None: ...

    def updated(self) -> None: ...

    def notification_failed(self, message: str) -> None: ...

    def fail(self, message: str) -> None: ...

    def source_terminal(self) -> None: ...


class CameraMonitorLiveDataset:
    """Narrow latest-cycle owner presented to a live view."""

    def __init__(self, raw: MonitorDataset[_CameraCyclePayload]) -> None:
        if not isinstance(raw, MonitorDataset):
            raise TypeError("raw must be MonitorDataset")
        self.raw = raw
        self._closed = False
        self._lock = threading.RLock()

    def publish_latest(
        self,
        payload: _CameraCyclePayload,
        producer: AcquisitionProducer[_CameraCyclePayload],
        *,
        direct_parent_refs: tuple[EventRef, ...],
    ) -> None:
        """Atomically replace the visible cycle after all fallible work succeeds."""

        if not isinstance(payload, _CameraCyclePayload):
            raise TypeError("camera latest publication requires _CameraCyclePayload")
        if not isinstance(producer, AcquisitionProducer):
            raise TypeError("camera latest publication requires AcquisitionProducer")
        with self._lock:
            self._ensure_open()
            replacement = self.raw.prepare_latest_cell_replacement(payload)
            try:
                envelope = producer.emit(
                    payload,
                    captured_at=payload.metadata[-1].captured_at,
                    direct_parent_refs=direct_parent_refs,
                )
            except BaseException:
                self.raw.abort_latest_cell_replacement(replacement)
                raise
            self.raw.commit_latest_cell_replacement(
                replacement,
                envelope,
                timeout=0.0,
            )

    def freeze_current(self) -> MonitorDatasetSnapshot:
        with self._lock:
            self._ensure_open()
            return self.raw.freeze_current()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.raw.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("camera monitor live dataset is closed")


@dataclass
class _CameraMonitorTransaction:
    port: BoundCameraMonitorPort
    view: CameraMonitorViewPort
    dataset: CameraMonitorLiveDataset
    stream: AcquisitionStream[CameraSample]
    producer: AcquisitionProducer[CameraSample]
    cycle_producer: AcquisitionProducer[_CameraCyclePayload]
    cycle_contract: _CameraCycleContract
    session_id: str
    buffer_frame_count: int
    operation_deadline_seconds: float
    association_source: CameraAssociatedSignalEventSource | None = None
    pending_samples: list[CameraSample] = field(default_factory=list)
    pending_refs: list[EventRef] = field(default_factory=list)
    prepare_attempted: bool = False
    view_notifications_enabled: bool = True

    def execute(self, context: RunContext) -> None:
        device = context.device(self.port.device.key)
        capability = self.port.capability
        self.prepare_attempted = True
        prepared = device.execute(
            PrepareCameraMonitorCommand(
                self.session_id,
                capability.capability_fingerprint,
                capability.settings_fingerprint,
                self.buffer_frame_count,
                self.operation_deadline_seconds,
            )
        )
        if not isinstance(prepared, CameraMonitorPreparedAck):
            raise TypeError(
                "camera monitor prepare returned an unexpected acknowledgement"
            )
        self._validate_ack(prepared.session_id, prepared.binding_instance_id)
        if (
            prepared.settings_fingerprint != capability.settings_fingerprint
            or prepared.capability_fingerprint != capability.capability_fingerprint
        ):
            raise RuntimeError(
                "camera monitor prepare acknowledgement changed capability"
            )
        context.checkpoint()
        started = device.execute(
            StartCameraMonitorCommand(
                self.session_id,
                self.operation_deadline_seconds,
            )
        )
        if not isinstance(started, CameraMonitorStartedAck):
            raise TypeError(
                "camera monitor start returned an unexpected acknowledgement"
            )
        self._validate_ack(started.session_id, started.binding_instance_id)
        if self.association_source is not None:
            self.association_source.mark_association_running()
        context.set_phase("monitoring-camera")
        while True:
            context.checkpoint()
            try:
                response = device.execute(
                    ReadCameraMonitorCommand(
                        self.session_id,
                        self.operation_deadline_seconds,
                    )
                )
            except CameraMonitorInterrupted:
                context.checkpoint()
                raise
            if isinstance(response, CameraMonitorNoFrameAck):
                self._validate_ack(
                    response.session_id,
                    response.binding_instance_id,
                )
                # Passive externally-triggered live acquisition has no frame
                # when hardware is idle.  Keep the Run armed and retain the
                # last visible front; this is not a timeout or data event.
                continue
            if not isinstance(response, CameraMonitorPayloadAck):
                raise TypeError(
                    "camera monitor read returned an unexpected acknowledgement"
                )
            self._validate_ack(response.session_id, response.binding_instance_id)
            payload = response.payload
            capability.payload_contract.validate(payload)
            metadata = payload.metadata
            envelope = self.producer.emit(
                payload,
                captured_at=metadata.captured_at,
            )
            self.pending_samples.append(payload)
            self.pending_refs.append(envelope.ref)
            frame_count = self.cycle_contract.metadata_contract.frame_count
            if len(self.pending_samples) == frame_count:
                context.checkpoint()
                self._publish_cycle()
                self.pending_samples.clear()
                self.pending_refs.clear()
                self._notify_view_updated()
            elif len(self.pending_samples) > frame_count:
                raise RuntimeError("camera monitor accumulated an oversized cycle")

    def _publish_cycle(self) -> None:
        cycle = _camera_cycle_payload(
            tuple(self.pending_samples),
            self.cycle_contract,
        )
        self.dataset.publish_latest(
            cycle,
            self.cycle_producer,
            direct_parent_refs=tuple(self.pending_refs),
        )

    def _notify_view_updated(self) -> None:
        if not self.view_notifications_enabled:
            return
        try:
            self.view.updated()
        except BaseException as error:
            self.view_notifications_enabled = False
            try:
                self.view.notification_failed(
                    "camera monitor view notification failed: "
                    f"{safe_error_summary(error)}"
                )
            except BaseException:
                pass

    def cleanup(
        self,
        context: RunContext,
        primary: BaseException | None,
        *,
        restore_exposure: Callable[[], CleanupReport] | None = None,
    ) -> CleanupReport:
        software_errors: list[BaseException] = []
        cancelled = isinstance(primary, CancellationRequested)
        stream_failure = StreamError(safe_error_summary(primary)) if primary else None
        self.pending_samples.clear()
        self.pending_refs.clear()
        if self.association_source is not None:
            self.association_source.mark_association_stopped()
        for producer in (self.producer, self.cycle_producer):
            try:
                if primary is None or cancelled:
                    producer.finish()
                else:
                    assert stream_failure is not None
                    producer.fail(stream_failure)
            except BaseException as error:
                software_errors.append(error)
        device_cleanup = (
            (lambda: self.port.cleanup(context, self.session_id))
            if self.prepare_attempted
            else (lambda: self.port.verify_idle(context))
        )
        steps = [device_cleanup]
        if restore_exposure is not None:
            steps.append(restore_exposure)
        report = run_cleanup_steps(*steps)
        terminal_error: BaseException | None
        if report.errors:
            terminal_error = report.errors[0]
        elif software_errors:
            terminal_error = software_errors[0]
        elif cancelled:
            terminal_error = None
        else:
            terminal_error = primary
        try:
            if terminal_error is None:
                self.view.source_terminal()
            else:
                self.view.fail(safe_error_summary(terminal_error))
        except BaseException as error:
            software_errors.append(error)
            try:
                self.dataset.close()
            except BaseException as close_error:
                software_errors.append(close_error)
        if not software_errors:
            return report
        return CleanupReport.complete(errors=(*report.errors, *software_errors))

    def _validate_ack(self, session_id: str, binding_instance_id: str) -> None:
        if session_id != self.session_id:
            raise RuntimeError(
                "camera monitor acknowledgement belongs to another session"
            )
        if binding_instance_id != self.port.device.binding_instance_id:
            raise RuntimeError("camera monitor acknowledgement binding differs")


class PreparedLiveCameraMeasurement:
    """One-shot application command that never exposes a Port or raw device."""

    __slots__ = (
        "_edge",
        "_cycle_contract",
        "_cycle_stream",
        "_cycle_producer",
        "_frame_schema",
        "_lock",
        "_active_output_bindings",
        "_port",
        "_request",
        "_signal_source",
        "_start_run",
        "_started",
        "_stream",
        "_producer",
    )

    def __init__(
        self,
        request: CameraMeasurementRequest,
        port: BoundCameraMonitorPort,
        start_run: Callable[[RunPlan], RunHandle],
        *,
        association_authority: CameraSignalAssociationAuthority | None = None,
    ) -> None:
        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("request must be CameraMeasurementRequest")
        if request.repeat != 0:
            raise ValueError("live Camera measurement requires repeat=0")
        if not isinstance(port, BoundCameraMonitorPort):
            raise TypeError("port must be BoundCameraMonitorPort")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        capability = port.capability
        source_id = capability.camera_capability_evidence.source_id
        if source_id != request.camera_ref.role:
            raise ValueError(
                "camera monitor capability source differs from requested role"
            )
        self._edge, self._cycle_contract = _camera_cycle_edge(
            capability.payload_contract,
            request.frames_per_cycle,
        )
        self._frame_schema = capability.payload_contract.value_schema
        self._request = request
        self._port = port
        self._stream, self._producer = AcquisitionStream.create(
            StreamId(f"camera-monitor:{request.camera_ref.role}"),
            capability.payload_contract,
        )
        self._cycle_stream, self._cycle_producer = AcquisitionStream.create(
            StreamId(f"camera-monitor-cycle:{request.camera_ref.role}"),
            self._cycle_contract,
        )
        self._signal_source = camera_signal_event_source(
            self._stream,
            request,
            capability.payload_contract,
            association_authority=association_authority,
            trigger_channel=(
                None
                if association_authority is None
                else _association_trigger_channel(capability)
            ),
            operation_deadline_seconds=(
                None
                if association_authority is None
                else capability.max_blocking_call_seconds
            ),
        )
        self._start_run = start_run
        self._lock = threading.Lock()
        self._active_output_bindings: dict[str, CameraFrameOutputBinding] | None = None
        self._started = False

    @property
    def view_schema(self) -> DatasetSchema:
        return self._edge.schema

    @property
    def request(self) -> CameraMeasurementRequest:
        return self._request

    @property
    def definition_key(self):
        return CAMERA_MEASUREMENT_KEY

    @property
    def dataset_output_declarations(self):
        return self._request.output_declarations

    def live_dataset_outputs(
        self,
        frozen: MonitorDatasetSnapshot,
    ) -> dict[str, LiveDatasetOutput]:
        """Name and split one monitor front by the frozen Camera request."""

        return project_camera_monitor_outputs(frozen, self._request)

    def value_schema(self, output_name: str) -> ValueSchema:
        """Return the declared schema without inspecting a first live frame."""

        return self._signal_source.value_schema(output_name)

    def open_signal_cursor(
        self,
        output_name: str,
    ) -> SignalEventCursor[CameraSample]:
        """Follow this running Camera producer without acquiring its device."""

        return self._signal_source.open_signal_cursor(output_name)

    def dataset_output_binding(self, output_name: str) -> CameraFrameOutputBinding:
        """Return the endpoint-read binding for one currently active frame signal."""

        if output_name not in self._request.output_names:
            raise KeyError(f"Camera has no output {output_name!r}")
        with self._lock:
            bindings = self._active_output_bindings
            if bindings is None:
                raise RuntimeError(
                    "Camera frame binding is unavailable until monitor preflight "
                    "has read back the active working point"
                )
            return bindings[output_name]

    def start_with_view(
        self,
        *,
        factory: Callable[[CameraMonitorViewSpec], CameraMonitorViewPort],
        lifecycle_owner: object | None = None,
    ) -> RunHandle:
        if not callable(factory):
            raise TypeError("factory must be callable")
        self._claim_start()
        spec = CameraMonitorViewSpec(
            BlockId(f"camera-monitor-{uuid.uuid4().hex}"),
            self._edge,
        )
        view = factory(spec)
        if getattr(view, "spec", None) is not spec:
            try:
                view.fail(
                    "camera monitor view did not retain its exact admitted spec"
                )
            except BaseException:
                pass
            raise ValueError(
                "camera monitor view must retain the admitted spec by identity"
            )
        plan = _compile_camera_monitor_plan(
            self._request,
            self._port,
            view,
            self._stream,
            self._producer,
            self._cycle_stream,
            self._cycle_producer,
            self._cycle_contract,
            (
                self._signal_source
                if isinstance(
                    self._signal_source,
                    CameraAssociatedSignalEventSource,
                )
                else None
            ),
            activate_output_bindings=self._activate_output_bindings,
            deactivate_output_bindings=self._deactivate_output_bindings,
        )
        plan = plan.with_lifecycle(
            owner=self if lifecycle_owner is None else lifecycle_owner,
            preemptible=True,
        )
        try:
            return self._start_run(plan)
        except BaseException as error:
            try:
                self._producer.fail(StreamError(safe_error_summary(error)))
            except BaseException:
                pass
            try:
                self._cycle_producer.fail(StreamError(safe_error_summary(error)))
            except BaseException:
                pass
            try:
                view.fail(safe_error_summary(error))
            except BaseException:
                pass
            raise

    def _claim_start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedLiveCameraMeasurement is one-shot")
            self._started = True

    def _activate_output_bindings(self, port: BoundCameraMonitorPort) -> None:
        capability = port.capability
        if capability.payload_contract.value_schema != self._frame_schema:
            raise RuntimeError(
                "configured Camera working point changed the declared frame schema"
            )
        binding = ReadoutBindingKey(self._request.camera_ref.role)
        values = {
            output.name: CameraFrameOutputBinding(
                output=output,
                readout_event_index=index,
                readout_binding=binding,
                capability_evidence=capability.camera_capability_evidence,
                binding_stamp=capability.binding_stamp,
                frame_schema=capability.payload_contract.value_schema,
                event_stream_id=self._stream.stream_id,
                event_stream_generation=self._stream.generation,
            )
            for index, output in enumerate(self._request.output_declarations)
        }
        with self._lock:
            if self._active_output_bindings is not None:
                raise RuntimeError("Camera frame bindings are already active")
            self._active_output_bindings = values

    def _deactivate_output_bindings(self) -> None:
        with self._lock:
            self._active_output_bindings = None


class PreparedAssociatedLiveCameraMeasurement(PreparedLiveCameraMeasurement):
    """Live Camera command with an explicitly composed pulse-association owner."""

    __slots__ = ()

    def signal_association_schedule_requirement(
        self,
        output_name: str,
    ) -> SignalAssociationScheduleRequirement:
        source = self._signal_source
        if not isinstance(source, CameraAssociatedSignalEventSource):
            raise RuntimeError(
                "associated Camera command lost its composition authority"
            )
        return source.signal_association_schedule_requirement(output_name)

    def open_associated_signal_cursor(
        self,
        output_name: str,
    ) -> SignalEventAssociationCursor:
        source = self._signal_source
        if not isinstance(source, CameraAssociatedSignalEventSource):
            raise RuntimeError(
                "associated Camera command lost its composition authority"
            )
        return source.open_associated_signal_cursor(output_name)


def _association_trigger_channel(capability) -> str:
    """Admit only a Q0-qualified, single-wire external-trigger producer."""

    if capability.acquisition_mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
        raise ValueError("free-running Camera Measurement cannot associate pulse events")
    evidence = capability.camera_capability_evidence
    if evidence.exact_external_trigger_qualification_digest is None:
        raise ValueError(
            "Camera association requires explicit exact-trigger qualification"
        )
    if evidence.exact_external_trigger_quiet_window_seconds is None:
        raise ValueError(
            "Camera association requires a qualified quiet-window fact"
        )
    channels = evidence.physical_facts.capture_trigger_channels
    if len(channels) != 1:
        raise ValueError("Camera association requires exactly one trigger channel")
    return channels[0]


def _compile_camera_monitor_plan(
    request: CameraMeasurementRequest,
    port: BoundCameraMonitorPort,
    view: CameraMonitorViewPort,
    stream: AcquisitionStream[CameraSample],
    producer: AcquisitionProducer[CameraSample],
    cycle_stream: AcquisitionStream[_CameraCyclePayload],
    cycle_producer: AcquisitionProducer[_CameraCyclePayload],
    cycle_contract: _CameraCycleContract,
    association_source: CameraAssociatedSignalEventSource | None,
    *,
    activate_output_bindings: Callable[[BoundCameraMonitorPort], None],
    deactivate_output_bindings: Callable[[], None],
) -> RunPlan[_CameraMonitorTransaction, None, None]:
    spec = getattr(view, "spec", None)
    if not isinstance(spec, CameraMonitorViewSpec):
        raise TypeError("camera monitor view has no CameraMonitorViewSpec")

    exposure_session_id = (
        uuid.uuid4().hex if request.exposure_seconds is not None else None
    )
    exposure_attempted = False
    if not callable(activate_output_bindings) or not callable(
        deactivate_output_bindings
    ):
        raise TypeError("Camera output binding lifecycle callbacks must be callable")

    def preflight(context: RunContext) -> _CameraMonitorTransaction:
        nonlocal exposure_attempted
        tap = None
        raw_dataset = None
        dataset = None
        try:
            active_port = port
            exposure = request.exposure_seconds
            if exposure is not None:
                assert exposure_session_id is not None
                exposure_attempted = True
                active_port = configure_camera_exposure(
                    context,
                    port,
                    exposure_session_id,
                    exposure,
                )
            activate_output_bindings(active_port)
            tap = cycle_stream.monitor()
            raw_dataset = MonitorDataset.latest_cell(
                spec.block_id,
                tap,
                spec.dataset_edge,
            )
            dataset = CameraMonitorLiveDataset(raw_dataset)
            view.bind(dataset)
            return _CameraMonitorTransaction(
                active_port,
                view,
                dataset,
                stream,
                producer,
                cycle_producer,
                cycle_contract,
                uuid.uuid4().hex,
                request.frames_per_cycle,
                active_port.capability.max_blocking_call_seconds,
                association_source,
            )
        except BaseException as error:
            deactivate_output_bindings()
            try:
                view.fail(safe_error_summary(error))
            except BaseException:
                pass
            if dataset is not None:
                try:
                    dataset.close()
                except BaseException:
                    pass
            elif raw_dataset is not None:
                try:
                    raw_dataset.close()
                except BaseException:
                    pass
            elif tap is not None:
                try:
                    tap.close()
                except BaseException:
                    pass
            try:
                producer.fail(StreamError(safe_error_summary(error)))
            except BaseException:
                pass
            try:
                cycle_producer.fail(StreamError(safe_error_summary(error)))
            except BaseException:
                pass
            raise

    def execute(context: RunContext, prepared: _CameraMonitorTransaction) -> None:
        prepared.execute(context)
        raise RuntimeError("continuous camera monitor returned without cancellation")

    def cleanup(
        context: RunContext,
        prepared: _CameraMonitorTransaction | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        deactivate_output_bindings()
        restore_exposure = None
        if exposure_attempted:
            assert exposure_session_id is not None
            restore_exposure = lambda: port.cleanup(
                context,
                exposure_session_id,
            )
        if prepared is None:
            try:
                producer.fail(
                    StreamError(
                        safe_error_summary(
                            primary
                            if primary is not None
                            else RuntimeError("camera monitor ended before preflight")
                        )
                    )
                )
            except BaseException:
                pass
            steps = [lambda: port.verify_idle(context)]
            if restore_exposure is not None:
                steps.append(restore_exposure)
            report = run_cleanup_steps(*steps)
            failure = primary
            if failure is None and report.errors:
                failure = report.errors[0]
            if failure is not None:
                try:
                    view.fail(safe_error_summary(failure))
                except BaseException:
                    pass
            return report
        return prepared.cleanup(
            context,
            primary,
            restore_exposure=restore_exposure,
        )

    return RunPlan(
        name=f"Camera monitor {request.camera_ref.role}",
        resource_claims=(port.resource_claim,),
        bound_devices=(port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=lambda _context, result: result,
        interrupt_operations=port.interrupt_operations,
    )


def prepare_live_camera_measurement(
    request: CameraMeasurementRequest,
    *,
    monitor_port: BoundCameraMonitorPort,
    start_run: Callable[[RunPlan], RunHandle],
    association_authority: CameraSignalAssociationAuthority | None = None,
) -> PreparedLiveCameraMeasurement:
    prepared_type = (
        PreparedAssociatedLiveCameraMeasurement
        if association_authority is not None
        else PreparedLiveCameraMeasurement
    )
    return prepared_type(
        request,
        monitor_port,
        start_run,
        association_authority=association_authority,
    )


__all__ = [
    "CameraMonitorLiveDataset",
    "CameraMonitorViewPort",
    "CameraMonitorViewSpec",
    "PreparedLiveCameraMeasurement",
    "PreparedAssociatedLiveCameraMeasurement",
    "prepare_live_camera_measurement",
]
