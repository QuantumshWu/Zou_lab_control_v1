"""Continuous raw-camera monitor application shared by notebook and Workbench.

The monitor owns acquisition and a bounded newest-first frame history.  Figure
selection, ROI projection, fitting, and other display analysis deliberately do
not live here: they are independent per-panel view branches over this raw data.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import uuid
from typing import Callable, Protocol

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    DatasetSchema,
    MONITOR_HISTORY,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
)
from zlc_neutral_atom.acquisition.camera import (
    CameraDatasetEventAdapter,
    CameraSample,
)
from zlc_neutral_atom.camera_measurement import (
    CameraMeasurementDescriptor,
    CameraMeasurementRequest,
)
from zlc_neutral_atom.runtime._failure import safe_error_summary
from zlc_neutral_atom.runtime.cancellation import CancellationRequested
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.dataset import (
    FrozenDatasetEdge,
    MonitorDataset,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.runtime.monitor import (
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
from zlc_neutral_atom.runtime.run import RunContext, RunHandle, RunPlan
from zlc_neutral_atom.runtime.streams import (
    AcquisitionProducer,
    AcquisitionStream,
    ProducerFlowControl,
    StreamError,
    StreamId,
    TraceContext,
)


_MONITOR_REPEAT_AXIS_ID = AxisId("camera-monitor.repeat")
_MONITOR_HISTORY_AXIS_ID = AxisId("camera-monitor.history")
_MONITOR_READOUT_EVENT_AXIS_ID = AxisId("camera-monitor.readout-event")
_STREAM_RETENTION_EVENTS = 1
_TAP_BACKLOG_EVENTS = 1


@dataclass(frozen=True)
class CameraMonitorViewSpec:
    """One admitted bounded rolling raw-frame window."""

    block_id: BlockId
    dataset_edge: FrozenDatasetEdge[CameraSample]

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.dataset_edge, FrozenDatasetEdge):
            raise TypeError("dataset_edge must be FrozenDatasetEdge")
        if self.dataset_edge.cell_schedule is not None:
            raise ValueError("camera monitor view requires a schedule-free dataset edge")
        schema = self.dataset_edge.schema
        if (
            schema.repeat_axis.size != 1
            or len(schema.point_axes) != 2
            or schema.point_axes[0].role != MONITOR_HISTORY
            or schema.point_axes[1].role != READOUT_EVENT
            or schema.point_layout
            != PointLayout.rect_c(
                (schema.point_axes[0].size, schema.point_axes[1].size)
            )
        ):
            raise ValueError(
                "camera monitor requires (R=1, MONITOR_HISTORY, READOUT_EVENT) storage"
            )


class CameraMonitorViewPort(Protocol):
    """Workbench-owned sink; binding transfers the live dataset lifetime."""

    @property
    def spec(self) -> CameraMonitorViewSpec: ...

    @property
    def terminal(self) -> bool: ...

    def bind(
        self,
        dataset: "CameraMonitorLiveDataset",
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None: ...

    def updated(self) -> None: ...

    def notification_failed(self, message: str) -> None: ...

    def fail(self, message: str) -> None: ...

    def source_terminal(self) -> None: ...


class CameraMonitorLiveDataset:
    """Narrow raw-history owner presented to a live view."""

    def __init__(self, raw: MonitorDataset[CameraSample]) -> None:
        if not isinstance(raw, MonitorDataset):
            raise TypeError("raw must be MonitorDataset")
        self.raw = raw
        self._closed = False
        self._lock = threading.RLock()

    def ingest_next(
        self,
        checkpoint: Callable[[], None] | None = None,
    ) -> bool:
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("checkpoint must be callable or None")
        with self._lock:
            self._ensure_open()
            if checkpoint is not None:
                checkpoint()
            before = self.raw.revision
            after = self.raw.ingest_next(timeout=0.0)
            return after.revision.value > before.value

    def materialize(self) -> MonitorDatasetSnapshot:
        with self._lock:
            self._ensure_open()
            return self.raw.materialize(None)

    def finish(self) -> None:
        with self._lock:
            self._ensure_open()

    def fail(self, error: StreamError) -> None:
        if not isinstance(error, StreamError):
            raise TypeError("error must be StreamError")

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
    session_id: str
    operation_deadline_seconds: float
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
                capability.max_source_burst_events,
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
            self.producer.emit(
                payload,
                captured_at=metadata.captured_at,
                trace=TraceContext(
                    context.run_id.value,
                    capability.camera_capability_evidence.source_id,
                    metadata.correlation_id,
                ),
            )
            if self.dataset.ingest_next(context.checkpoint):
                self._notify_view_updated()

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
    ) -> CleanupReport:
        software_errors: list[BaseException] = []
        cancelled = isinstance(primary, CancellationRequested)
        stream_failure = StreamError(safe_error_summary(primary)) if primary else None
        try:
            if primary is None or cancelled:
                self.producer.finish()
            else:
                assert stream_failure is not None
                self.producer.fail(stream_failure)
        except BaseException as error:
            software_errors.append(error)
        try:
            if primary is None or cancelled:
                self.dataset.finish()
            else:
                assert stream_failure is not None
                self.dataset.fail(stream_failure)
        except BaseException as error:
            software_errors.append(error)
        try:
            report = (
                self.port.cleanup(context, self.session_id)
                if self.prepare_attempted
                else self.port.verify_idle(context)
            )
        except BaseException as error:
            try:
                self.view.fail(safe_error_summary(error))
            finally:
                raise
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
        "_descriptor",
        "_edge",
        "_lock",
        "_port",
        "_request",
        "_start_run",
        "_started",
    )

    def __init__(
        self,
        request: CameraMeasurementRequest,
        port: BoundCameraMonitorPort,
        start_run: Callable[[RunPlan], RunHandle],
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
        schema = DatasetSchema(
            AxisSpec(
                _MONITOR_REPEAT_AXIS_ID,
                "monitor storage repeat",
                REPEAT,
                1,
                (0,),
            ),
            (
                AxisSpec(
                    _MONITOR_HISTORY_AXIS_ID,
                    "newest-first monitor history",
                    MONITOR_HISTORY,
                    request.history_capacity,
                ),
                AxisSpec(
                    _MONITOR_READOUT_EVENT_AXIS_ID,
                    "readout event within monitor cycle",
                    READOUT_EVENT,
                    request.frames_per_cycle,
                    tuple(range(request.frames_per_cycle)),
                ),
            ),
            PointLayout.rect_c(
                (request.history_capacity, request.frames_per_cycle)
            ),
            capability.payload_contract.value_schema,
        )
        self._edge = FrozenDatasetEdge(
            schema,
            CameraDatasetEventAdapter(capability.payload_contract),
        )
        self._descriptor = CameraMeasurementDescriptor(
            "Camera",
            request.camera_ref.role,
            schema,
            str(port.resource_claim.key),
        )
        self._request = request
        self._port = port
        self._start_run = start_run
        self._lock = threading.Lock()
        self._started = False

    @property
    def descriptor(self) -> CameraMeasurementDescriptor:
        return self._descriptor

    @property
    def view_schema(self) -> DatasetSchema:
        return self._edge.schema

    @property
    def request(self) -> CameraMeasurementRequest:
        return self._request

    def start_with_view(
        self,
        *,
        factory: Callable[[CameraMonitorViewSpec], CameraMonitorViewPort],
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
        plan = _compile_camera_monitor_plan(self._request, self._port, view)
        try:
            return self._start_run(plan)
        except BaseException as error:
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


def _compile_camera_monitor_plan(
    request: CameraMeasurementRequest,
    port: BoundCameraMonitorPort,
    view: CameraMonitorViewPort,
) -> RunPlan[_CameraMonitorTransaction, None, None]:
    spec = getattr(view, "spec", None)
    if not isinstance(spec, CameraMonitorViewSpec):
        raise TypeError("camera monitor view has no CameraMonitorViewSpec")

    def preflight(context: RunContext) -> _CameraMonitorTransaction:
        producer = None
        tap = None
        raw_dataset = None
        dataset = None
        try:
            stream, producer = AcquisitionStream.create(
                StreamId(f"camera-monitor:{request.camera_ref.role}"),
                port.capability.payload_contract,
                flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
                retention_events=_STREAM_RETENTION_EVENTS,
            )
            tap = stream.monitor(max_events=_TAP_BACKLOG_EVENTS)
            raw_dataset = MonitorDataset.append_window(
                spec.block_id,
                tap,
                spec.dataset_edge,
            )
            dataset = CameraMonitorLiveDataset(raw_dataset)
            view.bind(
                dataset,
                run_id=context.run_id.value,
                causation_domain_id=stream.generation.value,
            )
            return _CameraMonitorTransaction(
                port,
                view,
                dataset,
                stream,
                producer,
                uuid.uuid4().hex,
                port.capability.max_blocking_call_seconds,
            )
        except BaseException as error:
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
            if producer is not None:
                try:
                    producer.fail(StreamError(safe_error_summary(error)))
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
        if prepared is None:
            report = port.verify_idle(context)
            if primary is not None:
                try:
                    view.fail(safe_error_summary(primary))
                except BaseException:
                    pass
            return report
        return prepared.cleanup(context, primary)

    return RunPlan(
        name=f"Camera monitor {request.camera_ref.role}",
        resource_claims=(port.resource_claim,),
        bound_devices=(port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=lambda _context, result: result,
        interrupt_operations=port.interrupt_operations,
        requires_final_commit=False,
    )


def prepare_live_camera_measurement(
    request: CameraMeasurementRequest,
    *,
    monitor_port: BoundCameraMonitorPort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedLiveCameraMeasurement:
    return PreparedLiveCameraMeasurement(request, monitor_port, start_run)


__all__ = [
    "CameraMonitorLiveDataset",
    "CameraMonitorViewPort",
    "CameraMonitorViewSpec",
    "PreparedLiveCameraMeasurement",
    "prepare_live_camera_measurement",
]
