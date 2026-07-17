"""Free-running camera monitor application seam for notebook and Workbench."""

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
    REPEAT,
)
from zlc_neutral_atom.acquisition.camera import (
    CameraDatasetEventAdapter,
    CameraSample,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.runtime._failure import safe_error_summary
from zlc_neutral_atom.runtime.cancellation import CancellationRequested
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.dataset import (
    FrozenDatasetEdge,
    MonitorDataset,
    dataset_storage_nbytes,
    mutable_dataset_storage_nbytes,
)
from zlc_neutral_atom.runtime.monitor import (
    BoundCameraMonitorPort,
    CameraMonitorPayloadAck,
    CameraMonitorPreparedAck,
    CameraMonitorInterrupted,
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
from zlc_storage import (
    canonical_text,
    nonnegative_integer,
    positive_integer,
    positive_real,
)


_MONITOR_REPEAT_AXIS_ID = AxisId("camera-monitor.repeat")
_MONITOR_HISTORY_AXIS_ID = AxisId("camera-monitor.history")
_STREAM_RETENTION_EVENTS = 1
_TAP_BACKLOG_EVENTS = 1


@dataclass(frozen=True)
class CameraMonitorRequest:
    """Declarative request for one hardware-paced, display-only camera monitor."""

    camera_ref: DeviceRef
    memory_limit_bytes: int = 256 << 20
    io_timeout_seconds: float = 2.0
    history_capacity: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        object.__setattr__(
            self,
            "memory_limit_bytes",
            positive_integer(self.memory_limit_bytes, "memory_limit_bytes"),
        )
        object.__setattr__(
            self,
            "io_timeout_seconds",
            positive_real(self.io_timeout_seconds, "io_timeout_seconds"),
        )
        object.__setattr__(
            self,
            "history_capacity",
            positive_integer(self.history_capacity, "history_capacity"),
        )


@dataclass(frozen=True)
class CameraMonitorDescriptor:
    name: str
    camera_role: str
    output_schema: DatasetSchema
    resource_claim: str
    base_peak_bytes: int

    def __post_init__(self) -> None:
        canonical_text(self.name, "camera monitor name")
        canonical_text(self.camera_role, "camera monitor role")
        if not isinstance(self.output_schema, DatasetSchema):
            raise TypeError("camera monitor output_schema must be DatasetSchema")
        canonical_text(self.resource_claim, "resource_claim")
        object.__setattr__(
            self,
            "base_peak_bytes",
            positive_integer(self.base_peak_bytes, "base_peak_bytes"),
        )

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.output_schema.physical_shape

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_schema.fingerprint


@dataclass(frozen=True)
class CameraMonitorViewSpec:
    """One admitted bounded rolling image window; it can never be sealed."""

    block_id: BlockId
    dataset_edge: FrozenDatasetEdge[CameraSample]
    downstream_peak_bytes: int

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
            or len(schema.point_axes) != 1
            or schema.point_axes[0].role != MONITOR_HISTORY
            or schema.point_layout != PointLayout.rect_c((schema.point_axes[0].size,))
        ):
            raise ValueError(
                "camera monitor requires (R=1, dense MONITOR_HISTORY) storage"
            )
        object.__setattr__(
            self,
            "downstream_peak_bytes",
            nonnegative_integer(
                self.downstream_peak_bytes,
                "downstream_peak_bytes",
            ),
        )


class CameraMonitorViewPort(Protocol):
    """Workbench-owned sink; binding transfers the MonitorDataset lifetime."""

    @property
    def spec(self) -> CameraMonitorViewSpec: ...

    @property
    def terminal(self) -> bool: ...

    def bind(
        self,
        dataset: MonitorDataset[CameraSample],
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None: ...

    def updated(self) -> None: ...

    def fail(self, message: str) -> None: ...

    def source_terminal(self) -> None: ...


@dataclass
class _CameraMonitorTransaction:
    port: BoundCameraMonitorPort
    view: CameraMonitorViewPort
    dataset: MonitorDataset[CameraSample]
    stream: AcquisitionStream[CameraSample]
    producer: AcquisitionProducer[CameraSample]
    session_id: str
    io_timeout_seconds: float
    prepare_attempted: bool = False

    def execute(self, context: RunContext) -> None:
        device = context.device(self.port.device.key)
        capability = self.port.capability
        # Prepare may create the hardware session before an acknowledgement is
        # returned or validated.  From this point onward cleanup must target
        # this session rather than merely checking an allegedly idle device.
        self.prepare_attempted = True
        prepared = device.execute(
            PrepareCameraMonitorCommand(
                self.session_id,
                capability.capability_fingerprint,
                capability.settings_fingerprint,
                capability.max_source_burst_events,
                self.io_timeout_seconds,
            )
        )
        if not isinstance(prepared, CameraMonitorPreparedAck):
            raise TypeError("camera monitor prepare returned an unexpected acknowledgement")
        self._validate_ack(prepared.session_id, prepared.binding_instance_id)
        if (
            prepared.settings_fingerprint != capability.settings_fingerprint
            or prepared.capability_fingerprint != capability.capability_fingerprint
        ):
            raise RuntimeError("camera monitor prepare acknowledgement changed capability")
        context.checkpoint()
        started = device.execute(
            StartCameraMonitorCommand(self.session_id, self.io_timeout_seconds)
        )
        if not isinstance(started, CameraMonitorStartedAck):
            raise TypeError("camera monitor start returned an unexpected acknowledgement")
        self._validate_ack(started.session_id, started.binding_instance_id)
        context.set_phase("monitoring-camera")
        while True:
            context.checkpoint()
            try:
                response = device.execute(
                    ReadCameraMonitorCommand(
                        self.session_id,
                        self.io_timeout_seconds,
                    )
                )
            except CameraMonitorInterrupted:
                # A cancellation interrupt supersedes an in-flight read.  Prefer
                # the user's cancellation truth over the adapter's resulting
                # typed interruption symptom.  Real source failures do not enter
                # this branch and therefore cannot be washed into CANCELLED.
                context.checkpoint()
                raise
            if not isinstance(response, CameraMonitorPayloadAck):
                raise TypeError("camera monitor read returned an unexpected acknowledgement")
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
            self.dataset.ingest_next(timeout=0.0)
            self.view.updated()

    def cleanup(
        self,
        context: RunContext,
        primary: BaseException | None,
    ) -> CleanupReport:
        software_errors: list[BaseException] = []
        cancelled = isinstance(primary, CancellationRequested)
        try:
            if primary is None or cancelled:
                self.producer.finish()
            else:
                self.producer.fail(StreamError(safe_error_summary(primary)))
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
        elif report.decisions:
            terminal_error = RuntimeError(
                "camera monitor cleanup reported an unsafe terminal state: "
                f"{report.decisions[0].reason}"
            )
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
        return CleanupReport(
            safety_proofs=report.safety_proofs,
            decisions=report.decisions,
            errors=(*report.errors, *software_errors),
        )

    def _validate_ack(self, session_id: str, binding_instance_id: str) -> None:
        if session_id != self.session_id:
            raise RuntimeError("camera monitor acknowledgement belongs to another session")
        if binding_instance_id != self.port.device.binding_instance_id:
            raise RuntimeError("camera monitor acknowledgement binding differs")


class PreparedCameraMonitor:
    """One-shot application command that never exposes a Port or raw device."""

    __slots__ = (
        "_descriptor",
        "_edge",
        "_lock",
        "_memory_limit_bytes",
        "_port",
        "_request",
        "_start_run",
        "_started",
    )

    def __init__(
        self,
        request: CameraMonitorRequest,
        port: BoundCameraMonitorPort,
        start_run: Callable[[RunPlan], RunHandle],
    ) -> None:
        if not isinstance(request, CameraMonitorRequest):
            raise TypeError("request must be CameraMonitorRequest")
        if not isinstance(port, BoundCameraMonitorPort):
            raise TypeError("port must be BoundCameraMonitorPort")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        capability = port.capability
        source_id = capability.camera_capability_evidence.source_id
        if source_id != request.camera_ref.role:
            raise ValueError("camera monitor capability source differs from requested role")
        if request.io_timeout_seconds > capability.max_blocking_call_seconds:
            raise ValueError(
                "camera monitor I/O timeout exceeds the adapter blocking-call bound"
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
            ),
            PointLayout.rect_c((request.history_capacity,)),
            capability.payload_contract.value_schema,
        )
        self._edge = FrozenDatasetEdge(
            schema,
            CameraDatasetEventAdapter(capability.payload_contract),
        )
        base_peak = _base_monitor_peak_bytes(port, self._edge)
        if base_peak > request.memory_limit_bytes:
            raise MemoryError(
                f"camera monitor base peak {base_peak} exceeds limit "
                f"{request.memory_limit_bytes}"
            )
        self._descriptor = CameraMonitorDescriptor(
            "Camera monitor",
            request.camera_ref.role,
            schema,
            str(port.resource_claim.key),
            base_peak,
        )
        self._request = request
        self._port = port
        self._start_run = start_run
        self._memory_limit_bytes = request.memory_limit_bytes
        self._lock = threading.Lock()
        self._started = False

    @property
    def descriptor(self) -> CameraMonitorDescriptor:
        return self._descriptor

    @property
    def view_schema(self) -> DatasetSchema:
        return self._edge.schema

    def start_with_view(
        self,
        *,
        downstream_peak_bytes: int,
        factory: Callable[[CameraMonitorViewSpec], CameraMonitorViewPort],
    ) -> RunHandle:
        if not callable(factory):
            raise TypeError("factory must be callable")
        downstream = nonnegative_integer(
            downstream_peak_bytes,
            "downstream_peak_bytes",
        )
        required = self._descriptor.base_peak_bytes + downstream
        if required > self._memory_limit_bytes:
            raise MemoryError(
                f"camera monitor peak budget {required} exceeds limit "
                f"{self._memory_limit_bytes}"
            )
        self._claim_start()
        spec = CameraMonitorViewSpec(
            BlockId(f"camera-monitor-{uuid.uuid4().hex}"),
            self._edge,
            downstream,
        )
        view = factory(spec)
        if getattr(view, "spec", None) is not spec:
            try:
                view.fail("camera monitor view did not retain its exact admitted spec")
            except BaseException:
                pass
            raise ValueError("camera monitor view must retain the admitted spec by identity")
        plan = _compile_camera_monitor_plan(
            self._request,
            self._port,
            view,
        )
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
                raise RuntimeError("PreparedCameraMonitor is one-shot")
            self._started = True


def _base_monitor_peak_bytes(
    port: BoundCameraMonitorPort,
    edge: FrozenDatasetEdge[CameraSample],
) -> int:
    capability = port.capability
    payload = capability.payload_contract.max_retained_nbytes
    schema = edge.schema
    history_cells = schema.repeat_axis.size * schema.point_layout.storage_size
    immutable_snapshot = dataset_storage_nbytes(schema)
    mutable_materializer = mutable_dataset_storage_nbytes(schema)
    # A non-trivial append window is presented newest-first.  Once its ring has
    # advanced, NumPy gathers values, validity, and the written mask before the
    # immutable DataBlock makes its own owner copy.  Admission must cover that
    # worst case rather than only the canonical first frame.
    reorder_scratch = mutable_materializer if history_cells > 1 else 0
    return (
        capability.driver_ring_bytes
        + capability.adapter_record_retention_bytes
        + (_STREAM_RETENTION_EVENTS * payload)
        + (_TAP_BACKLOG_EVENTS * payload)
        + mutable_materializer
        + immutable_snapshot
        + reorder_scratch
        + (history_cells * edge.metadata_max_retained_nbytes)
    )


def _compile_camera_monitor_plan(
    request: CameraMonitorRequest,
    port: BoundCameraMonitorPort,
    view: CameraMonitorViewPort,
) -> RunPlan[_CameraMonitorTransaction, None, None]:
    spec = getattr(view, "spec", None)
    if not isinstance(spec, CameraMonitorViewSpec):
        raise TypeError("camera monitor view has no CameraMonitorViewSpec")

    def preflight(context: RunContext) -> _CameraMonitorTransaction:
        stream = None
        producer = None
        tap = None
        dataset = None
        try:
            stream, producer = AcquisitionStream.create(
                StreamId(f"camera-monitor:{request.camera_ref.role}"),
                port.capability.payload_contract,
                flow_control=ProducerFlowControl.NON_BACKPRESSURE_CAPTURED,
                retention_events=_STREAM_RETENTION_EVENTS,
                retention_bytes=(
                    _STREAM_RETENTION_EVENTS
                    * port.capability.payload_contract.max_retained_nbytes
                ),
            )
            tap = stream.monitor(
                max_events=_TAP_BACKLOG_EVENTS,
                max_bytes=(
                    _TAP_BACKLOG_EVENTS
                    * port.capability.payload_contract.max_retained_nbytes
                ),
            )
            dataset = MonitorDataset.append_window(spec.block_id, tap, spec.dataset_edge)
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
                request.io_timeout_seconds,
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
        timeout_seconds=None,
        requires_final_commit=False,
    )


def prepare_camera_monitor(
    request: CameraMonitorRequest,
    *,
    monitor_port: BoundCameraMonitorPort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedCameraMonitor:
    return PreparedCameraMonitor(request, monitor_port, start_run)


__all__ = [
    "CameraMonitorDescriptor",
    "CameraMonitorRequest",
    "CameraMonitorViewPort",
    "CameraMonitorViewSpec",
    "PreparedCameraMonitor",
    "prepare_camera_monitor",
]
