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
    ReductionMethod,
    Selection,
    StreamGenerationId,
    ValueSchema,
    ValidityPolicy,
    selection_from_tree,
    selection_to_tree,
)
from zlc_neutral_atom.acquisition.camera import (
    CameraDatasetEventAdapter,
    CameraSample,
    CameraSampleContract,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.processing.roi_monitor import (
    RoiScalarBinding,
    RoiScalarDatasetEventAdapter,
    RoiScalarMetadata,
    RoiScalarMetadataContract,
    RoiScalarSample,
    RoiScalarSampleContract,
    RoiScalarStreamProjection,
    max_roi_scalar_reduction_scratch_nbytes,
    roi_scalar_output_schema,
)
from zlc_neutral_atom.runtime._failure import safe_error_summary
from zlc_neutral_atom.runtime.cancellation import CancellationRequested
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.control import (
    ControlCommand,
    ControlReceipt,
    create_control_topic,
)
from zlc_neutral_atom.runtime.dataset import (
    FrozenDatasetEdge,
    MonitorDataset,
    MonitorDatasetSnapshot,
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
    MonitorUpdate,
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
_ROI_SCALAR_REPEAT_AXIS_ID = AxisId("camera-monitor.roi-scalar-repeat")
_ROI_SCALAR_HISTORY_AXIS_ID = AxisId("camera-monitor.roi-scalar-history")
_STREAM_RETENTION_EVENTS = 1
_TAP_BACKLOG_EVENTS = 1


@dataclass(frozen=True)
class CameraMonitorRequest:
    """Declarative request for one hardware-paced, display-only camera monitor."""

    camera_ref: DeviceRef
    memory_limit_bytes: int = 256 << 20
    io_timeout_seconds: float = 2.0
    history_capacity: int = 8
    roi: Selection | None = None
    roi_reduction: ReductionMethod = ReductionMethod.MEAN
    roi_validity_policy: ValidityPolicy = ValidityPolicy.REQUIRE_ALL
    scalar_history_capacity: int = 300

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
        if self.roi is not None:
            if not isinstance(self.roi, Selection):
                raise TypeError("roi must be zlc_data.Selection or None")
            object.__setattr__(
                self,
                "roi",
                selection_from_tree(selection_to_tree(self.roi)),
            )
        if not isinstance(self.roi_reduction, ReductionMethod):
            raise TypeError("roi_reduction must be zlc_data.ReductionMethod")
        if self.roi_reduction not in (
            ReductionMethod.MEAN,
            ReductionMethod.SUM,
            ReductionMethod.MAX,
        ):
            raise ValueError("camera monitor ROI reduction must be MEAN, SUM, or MAX")
        if not isinstance(self.roi_validity_policy, ValidityPolicy):
            raise TypeError("roi_validity_policy must be zlc_data.ValidityPolicy")
        if self.roi_validity_policy is ValidityPolicy.MIN_COUNT:
            raise ValueError("camera monitor ROI MIN_COUNT is not implemented")
        object.__setattr__(
            self,
            "scalar_history_capacity",
            positive_integer(
                self.scalar_history_capacity,
                "scalar_history_capacity",
            ),
        )


@dataclass(frozen=True)
class CameraMonitorDescriptor:
    name: str
    camera_role: str
    output_schema: DatasetSchema
    scalar_output_schema: DatasetSchema | None
    resource_claim: str
    base_peak_bytes: int

    def __post_init__(self) -> None:
        canonical_text(self.name, "camera monitor name")
        canonical_text(self.camera_role, "camera monitor role")
        if not isinstance(self.output_schema, DatasetSchema):
            raise TypeError("camera monitor output_schema must be DatasetSchema")
        if self.scalar_output_schema is not None and not isinstance(
            self.scalar_output_schema,
            DatasetSchema,
        ):
            raise TypeError("scalar_output_schema must be DatasetSchema or None")
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
    scalar_block_id: BlockId | None
    scalar_dataset_edge: FrozenDatasetEdge[RoiScalarSample] | None
    roi_binding: RoiScalarBinding | None
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
        scalar_values = (
            self.scalar_block_id,
            self.scalar_dataset_edge,
            self.roi_binding,
        )
        if any(value is None for value in scalar_values) and not all(
            value is None for value in scalar_values
        ):
            raise ValueError("ROI scalar block, edge, and binding must appear together")
        if self.scalar_dataset_edge is not None:
            assert self.scalar_block_id is not None and self.roi_binding is not None
            scalar_schema = self.scalar_dataset_edge.schema
            if self.scalar_dataset_edge.cell_schedule is not None:
                raise ValueError("ROI scalar view requires a schedule-free dataset edge")
            if (
                scalar_schema.repeat_axis.size != 1
                or len(scalar_schema.point_axes) != 1
                or scalar_schema.point_axes[0].role != MONITOR_HISTORY
                or scalar_schema.cell_schema is not self.roi_binding.output_schema
                or scalar_schema.cell_schema.data_axes
            ):
                raise ValueError("ROI scalar view requires one dense scalar history")
        object.__setattr__(
            self,
            "downstream_peak_bytes",
            nonnegative_integer(
                self.downstream_peak_bytes,
                "downstream_peak_bytes",
            ),
        )


@dataclass(frozen=True)
class CameraMonitorRoiState:
    """Currently applied ROI branch; pending commands live only in receipts."""

    binding: RoiScalarBinding | None
    control_revision: int | None
    scalar_block_id: BlockId | None
    scalar_dataset_edge: FrozenDatasetEdge[RoiScalarSample] | None
    scalar_generation: StreamGenerationId | None
    state_revision: int = 0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_revision",
            nonnegative_integer(self.state_revision, "state_revision"),
        )
        if self.failure_reason is not None:
            object.__setattr__(
                self,
                "failure_reason",
                canonical_text(self.failure_reason, "ROI state failure reason"),
            )
        scalar_values = (
            self.scalar_block_id,
            self.scalar_dataset_edge,
            self.scalar_generation,
        )
        if self.binding is None:
            if any(value is not None for value in scalar_values):
                raise ValueError("raw-only ROI state cannot retain scalar identity")
            if self.control_revision is not None:
                object.__setattr__(
                    self,
                    "control_revision",
                    positive_integer(self.control_revision, "control_revision"),
                )
            if self.state_revision == 0 and (
                self.control_revision is not None or self.failure_reason is not None
            ):
                raise ValueError("initial ROI state cannot report applied or failed state")
            return
        if not isinstance(self.binding, RoiScalarBinding):
            raise TypeError("binding must be RoiScalarBinding or None")
        object.__setattr__(
            self,
            "control_revision",
            positive_integer(self.control_revision, "control_revision"),
        )
        if not isinstance(self.scalar_block_id, BlockId):
            raise TypeError("applied ROI state requires scalar_block_id")
        edge = self.scalar_dataset_edge
        if not isinstance(edge, FrozenDatasetEdge):
            raise TypeError("applied ROI state requires scalar_dataset_edge")
        if not isinstance(self.scalar_generation, StreamGenerationId):
            raise TypeError("applied ROI state requires scalar_generation")
        if edge.cell_schedule is not None:
            raise ValueError("ROI scalar state requires a schedule-free dataset edge")
        if edge.schema.cell_schema is not self.binding.output_schema:
            raise ValueError("ROI scalar state edge differs from the applied output schema")
        if self.state_revision == 0:
            raise ValueError("applied ROI state requires a positive state_revision")
        if self.failure_reason is not None:
            raise ValueError("applied ROI state cannot contain failure_reason")


class CameraMonitorViewPort(Protocol):
    """Workbench-owned sink; binding transfers the MonitorDataset lifetime."""

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


@dataclass(frozen=True)
class CameraMonitorSnapshot:
    """One atomically frozen raw/scalar pair from the same source event."""

    raw: MonitorDatasetSnapshot
    scalar: MonitorDatasetSnapshot | None
    scalar_metadata: RoiScalarMetadata | None

    def __post_init__(self) -> None:
        if not isinstance(self.raw, MonitorDatasetSnapshot):
            raise TypeError("raw must be MonitorDatasetSnapshot")
        if (self.scalar is None) != (self.scalar_metadata is None):
            raise ValueError("scalar snapshot and metadata must appear together")
        if self.scalar is None:
            return
        if not isinstance(self.scalar, MonitorDatasetSnapshot):
            raise TypeError("scalar must be MonitorDatasetSnapshot")
        if not isinstance(self.scalar_metadata, RoiScalarMetadata):
            raise TypeError("scalar_metadata must be RoiScalarMetadata")
        if self.raw.head is None or self.scalar.head is None:
            raise ValueError("joined camera snapshot requires two event heads")
        if self.scalar_metadata.source_event_ref != self.raw.head:
            raise ValueError("ROI scalar head belongs to another raw camera event")


@dataclass
class _RoiScalarBranch:
    binding: RoiScalarBinding
    control_revision: int
    block_id: BlockId
    edge: FrozenDatasetEdge[RoiScalarSample]
    stream: AcquisitionStream[RoiScalarSample]
    producer: AcquisitionProducer[RoiScalarSample]
    dataset: MonitorDataset[RoiScalarSample]

    def state(self, state_revision: int) -> CameraMonitorRoiState:
        return CameraMonitorRoiState(
            self.binding,
            self.control_revision,
            self.block_id,
            self.edge,
            self.stream.generation,
            state_revision,
        )


class CameraMonitorLiveDataset:
    """Stable raw owner with one revisioned, replaceable ROI scalar branch."""

    def __init__(
        self,
        raw: MonitorDataset[CameraSample],
        *,
        projection: RoiScalarStreamProjection,
        input_contract: CameraSampleContract,
        scalar_edges: tuple[FrozenDatasetEdge[RoiScalarSample], ...],
        scalar_stream_id: StreamId,
        max_reduction_scratch_nbytes: int,
        initial_scalar_block_id: BlockId | None = None,
        initial_scalar_edge: FrozenDatasetEdge[RoiScalarSample] | None = None,
        initial_binding: RoiScalarBinding | None = None,
    ) -> None:
        if not isinstance(raw, MonitorDataset):
            raise TypeError("raw must be MonitorDataset")
        if not isinstance(projection, RoiScalarStreamProjection):
            raise TypeError("projection must be RoiScalarStreamProjection")
        if not isinstance(input_contract, CameraSampleContract):
            raise TypeError("input_contract must be CameraSampleContract")
        edges = tuple(scalar_edges)
        if not edges or any(not isinstance(edge, FrozenDatasetEdge) for edge in edges):
            raise TypeError("scalar_edges must contain FrozenDatasetEdge values")
        edge_by_schema: dict[str, FrozenDatasetEdge[RoiScalarSample]] = {}
        for edge in edges:
            if edge.cell_schedule is not None or edge.schema.cell_schema.data_axes:
                raise ValueError("ROI control edges must be schedule-free scalar histories")
            fingerprint = edge.schema.cell_schema.fingerprint
            if fingerprint in edge_by_schema:
                raise ValueError("scalar_edges must contain unique value schemas")
            edge_by_schema[fingerprint] = edge
        if not isinstance(scalar_stream_id, StreamId):
            raise TypeError("scalar_stream_id must be StreamId")
        scratch = nonnegative_integer(
            max_reduction_scratch_nbytes,
            "max_reduction_scratch_nbytes",
        )
        initial_values = (
            initial_scalar_block_id,
            initial_scalar_edge,
            initial_binding,
        )
        if any(value is None for value in initial_values) and not all(
            value is None for value in initial_values
        ):
            raise ValueError("initial scalar block, edge, and binding must appear together")
        if initial_binding is not None:
            assert initial_scalar_block_id is not None and initial_scalar_edge is not None
            if not isinstance(initial_scalar_block_id, BlockId):
                raise TypeError("initial_scalar_block_id must be BlockId")
            if not any(edge is initial_scalar_edge for edge in edges):
                raise ValueError("initial_scalar_edge is absent from scalar_edges")
            if initial_scalar_edge.schema.cell_schema is not initial_binding.output_schema:
                raise ValueError("initial ROI binding differs from its dataset edge")
        self.raw = raw
        self._projection = projection
        self._input_contract = input_contract
        self._edge_by_schema = edge_by_schema
        self._scalar_stream_id = scalar_stream_id
        self._max_reduction_scratch_nbytes = scratch
        self._initial_scalar_block_id = initial_scalar_block_id
        self._initial_scalar_edge = initial_scalar_edge
        self._initial_revision = 1 if initial_binding is not None else None
        self._branch: _RoiScalarBranch | None = None
        self._roi_state = CameraMonitorRoiState(None, None, None, None, None)
        self._projection_usable = True
        self._controls_terminated = False
        self._lock = threading.RLock()
        self._closed = False
        self._roi_topic, self._roi_consumer = create_control_topic(
            self._snapshot_roi_candidate
        )
        self._initial_roi_receipt = None
        if initial_binding is not None:
            # The declarative initial ROI is the first real command.  It is not
            # reported as applied until its first scalar event commits.
            self._initial_roi_receipt = self._roi_topic.publish(initial_binding)

    @property
    def initial_roi_receipt(self) -> ControlReceipt | None:
        """Revision-one receipt for a declarative request ROI, if present."""

        return self._initial_roi_receipt

    def prepare_roi_control(
        self,
        selection: Selection | None,
        reduction: ReductionMethod = ReductionMethod.MEAN,
        validity_policy: ValidityPolicy = ValidityPolicy.REQUIRE_ALL,
    ) -> RoiScalarBinding | None:
        """Validate and freeze one candidate without accepting a revision."""

        if selection is None:
            return None
        if not isinstance(selection, Selection):
            raise TypeError("selection must be zlc_data.Selection or None")
        if not isinstance(reduction, ReductionMethod):
            raise TypeError("reduction must be zlc_data.ReductionMethod")
        if not isinstance(validity_policy, ValidityPolicy):
            raise TypeError("validity_policy must be zlc_data.ValidityPolicy")
        owned_selection = selection_from_tree(selection_to_tree(selection))
        output = roi_scalar_output_schema(self._input_contract, reduction)
        try:
            edge = self._edge_by_schema[output.fingerprint]
        except KeyError as error:
            raise ValueError("ROI reduction output schema was not admitted") from error
        binding = RoiScalarBinding(
            self._input_contract,
            owned_selection,
            reduction,
            validity_policy,
            edge.schema.cell_schema,
        )
        if binding.reduction_scratch_nbytes > self._max_reduction_scratch_nbytes:
            raise MemoryError("ROI candidate exceeds the admitted reduction scratch bound")
        return binding

    def submit_roi_control(
        self,
        candidate: RoiScalarBinding | None,
    ) -> ControlReceipt:
        """Accept one latest-wins candidate for the next source-shot boundary."""

        # Publishing must not wait behind a reduction that currently owns the
        # data transaction lock.  ControlTopic linearizes publish vs terminal
        # shutdown and snapshots the candidate before accepting a revision.
        return self._roi_topic.publish(candidate)

    def current_roi_state(self) -> CameraMonitorRoiState:
        with self._lock:
            return self._roi_state

    def ingest_next(
        self,
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("checkpoint must be callable or None")
        with self._lock:
            self._ensure_open()
            self.raw.ingest_next(timeout=0.0)
            if not self._projection_usable:
                return
            try:
                update = self._projection.take_next(timeout=0.0)
            except BaseException as error:
                self._projection_usable = False
                self._drop_branch_locked(
                    f"ROI processor ingress failed: {safe_error_summary(error)}"
                )
                self._terminate_controls_locked(
                    f"ROI processor ingress terminated: {safe_error_summary(error)}"
                )
                return

            try:
                self._checkpoint(checkpoint)
                command = self._roi_consumer.take_latest()
                if command is None:
                    self._process_applied_branch_locked(update, checkpoint)
                    return
                self._apply_command_locked(command, update, checkpoint)
            except CancellationRequested:
                self._terminate_controls_locked("camera monitor source was cancelled")
                raise

    def materialize(self) -> CameraMonitorSnapshot:
        with self._lock:
            self._ensure_open()
            return self._materialize_locked()

    def finish(self) -> None:
        with self._lock:
            self._ensure_open()
            self._terminate_controls_locked("camera monitor source finished")
            branch, self._branch = self._branch, None
            if branch is not None:
                self._set_raw_roi_state_locked(branch.control_revision)
                self._retire_branch_locked(branch)

    def fail(self, error: StreamError) -> None:
        if not isinstance(error, StreamError):
            raise TypeError("error must be StreamError")
        with self._lock:
            if self._closed:
                return
            self._terminate_controls_locked(
                f"camera monitor source failed: {safe_error_summary(error)}"
            )
            self._drop_branch_locked(
                f"camera monitor source failed: {safe_error_summary(error)}"
            )

    def close(self) -> None:
        errors: list[BaseException] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._terminate_controls_locked("camera monitor live dataset closed")
            branch, self._branch = self._branch, None
            if branch is not None:
                self._set_raw_roi_state_locked(branch.control_revision)
                try:
                    self._retire_branch_locked(branch)
                except BaseException as error:
                    errors.append(error)
            for close in (self._projection.close, self.raw.close):
                try:
                    close()
                except BaseException as error:
                    errors.append(error)
        if errors:
            raise errors[0]

    def _materialize_locked(self) -> CameraMonitorSnapshot:
        raw = self.raw.materialize(None)
        branch = self._branch
        if branch is None:
            return CameraMonitorSnapshot(raw, None, None)
        scalar = branch.dataset.materialize(None)
        if scalar.head is None:
            raise RuntimeError("ROI scalar dataset has no head event")
        try:
            head_index = scalar.event_refs.index(scalar.head)
        except ValueError as error:  # MonitorDatasetSnapshot already guards this
            raise RuntimeError("ROI scalar head has no aligned metadata") from error
        metadata = scalar.cell_metadata[head_index]
        if not isinstance(metadata, RoiScalarMetadata):
            raise TypeError("ROI scalar dataset head metadata has another type")
        if metadata.binding_fingerprint != branch.binding.fingerprint:
            raise RuntimeError("ROI scalar metadata differs from the admitted binding")
        if metadata.control_revision != branch.control_revision:
            raise RuntimeError("ROI scalar metadata differs from the applied control revision")
        return CameraMonitorSnapshot(raw, scalar, metadata)

    def _snapshot_roi_candidate(
        self,
        candidate: RoiScalarBinding | None,
    ) -> RoiScalarBinding | None:
        if candidate is None:
            return None
        if not isinstance(candidate, RoiScalarBinding):
            raise TypeError("ROI control candidate must be RoiScalarBinding or None")
        if candidate.input_contract.fingerprint != self._input_contract.fingerprint:
            raise ValueError("ROI control candidate belongs to another camera contract")
        return self.prepare_roi_control(
            candidate.selection,
            candidate.reduction,
            candidate.validity_policy,
        )

    def _apply_command_locked(
        self,
        command: ControlCommand[RoiScalarBinding | None],
        update: MonitorUpdate[CameraSample],
        checkpoint: Callable[[], None] | None,
    ) -> None:
        candidate = command.value
        if candidate is None:
            self._checkpoint(checkpoint)
            branch, self._branch = self._branch, None
            self._set_raw_roi_state_locked(command.revision)
            self._roi_consumer.applied(command)
            if branch is not None:
                self._retire_branch_locked(branch)
            return

        try:
            payload = self._projection.project(update, candidate, command.revision)
        except CancellationRequested:
            raise
        except BaseException as error:
            self._roi_consumer.rejected(
                command,
                f"ROI candidate failed: {safe_error_summary(error)}",
            )
            self._process_applied_branch_locked(update, checkpoint)
            return

        # The reduction may be the longest downstream operation.  Cancellation
        # that won while it was running must terminalize this in-flight command
        # before any scalar commit or APPLIED acknowledgement can become stale.
        self._checkpoint(checkpoint)

        branch = self._branch
        if branch is not None and branch.edge.schema.cell_schema is candidate.output_schema:
            try:
                replacement = branch.dataset.prepare_append_replacement(payload)
            except BaseException as error:
                self._roi_consumer.rejected(
                    command,
                    f"ROI retarget staging failed: {safe_error_summary(error)}",
                )
                self._process_applied_branch_locked(update, checkpoint)
                return
            try:
                self._checkpoint(checkpoint)
            except CancellationRequested:
                try:
                    branch.dataset.abort_append_replacement(replacement)
                except BaseException:
                    pass
                raise
            sequence_before_publish = branch.stream.next_sequence
            if sequence_before_publish != replacement.expected_sequence:
                try:
                    branch.dataset.abort_append_replacement(replacement)
                except BaseException:
                    pass
                reason = (
                    "ROI retarget staged watermark differs from the exclusive "
                    "producer sequence"
                )
                self._roi_consumer.rejected(command, reason)
                self._drop_branch_locked(reason)
                return
            try:
                envelope = self._projection.publish(
                    update,
                    replacement.payload,
                    branch.producer,
                )
            except CancellationRequested:
                try:
                    branch.dataset.abort_append_replacement(replacement)
                except BaseException:
                    pass
                if branch.stream.next_sequence != sequence_before_publish:
                    # Cancellation arrived through a wrapper only after the
                    # exclusive producer had irreversibly published.  Never
                    # resume the old consumer across that consumed sequence.
                    self._drop_branch_locked(
                        "ROI retarget publication committed before cancellation"
                    )
                raise
            except BaseException as error:
                publication_committed = (
                    branch.stream.next_sequence != sequence_before_publish
                )
                try:
                    branch.dataset.abort_append_replacement(replacement)
                except BaseException as abort_error:
                    reason = (
                        "ROI scalar replacement abort failed: "
                        f"{safe_error_summary(abort_error)}"
                    )
                    self._roi_consumer.rejected(command, reason)
                    self._drop_branch_locked(reason)
                    return
                if publication_committed:
                    reason = (
                        "ROI retarget publish failed after authoritative sequence "
                        f"advance: {safe_error_summary(error)}"
                    )
                    self._roi_consumer.rejected(command, reason)
                    self._drop_branch_locked(reason)
                    return
                self._roi_consumer.rejected(
                    command,
                    f"ROI retarget publish failed: {safe_error_summary(error)}",
                )
                self._process_applied_branch_locked(update, checkpoint)
                return
            try:
                branch.dataset.commit_append_replacement(
                    replacement,
                    envelope,
                    timeout=0.0,
                )
            except BaseException as error:
                reason = (
                    "ROI retarget authoritative commit failed: "
                    f"{safe_error_summary(error)}"
                )
                self._roi_consumer.rejected(command, reason)
                # Publication already consumed a sequence.  The old binding can
                # no longer resume without manufacturing a provenance gap.
                self._drop_branch_locked(reason)
                return
            # The publish/finalize pair is the irreversible data-plane half of
            # this control transaction.  Once it succeeds, binding/state/APPLIED
            # must follow without another cancellation point; a later Stop wins
            # only after this revision has become one coherent transaction.
            branch.binding = candidate
            branch.control_revision = command.revision
            self._roi_state = branch.state(self._next_roi_state_revision_locked())
            self._roi_consumer.applied(command)
            return

        replacement: _RoiScalarBranch | None = None
        try:
            replacement = self._create_branch_locked(candidate, command.revision)
            self._checkpoint(checkpoint)
            self._projection.publish(update, payload, replacement.producer)
            self._checkpoint(checkpoint)
            replacement.dataset.ingest_next(timeout=0.0)
        except CancellationRequested:
            if replacement is not None:
                self._retire_branch_locked(replacement)
            raise
        except BaseException as error:
            if replacement is not None:
                self._retire_branch_locked(
                    replacement,
                    error=StreamError(safe_error_summary(error)),
                )
            self._roi_consumer.rejected(
                command,
                f"ROI migration failed: {safe_error_summary(error)}",
            )
            self._process_applied_branch_locked(update, checkpoint)
            return

        assert replacement is not None
        previous, self._branch = self._branch, replacement
        self._roi_state = replacement.state(self._next_roi_state_revision_locked())
        self._roi_consumer.applied(command)
        if previous is not None:
            self._retire_branch_locked(
                previous,
                replacement=replacement.stream.generation,
            )

    def _process_applied_branch_locked(
        self,
        update: MonitorUpdate[CameraSample],
        checkpoint: Callable[[], None] | None,
    ) -> None:
        branch = self._branch
        if branch is None:
            return
        try:
            payload = self._projection.project(
                update,
                branch.binding,
                branch.control_revision,
            )
        except CancellationRequested:
            raise
        except BaseException as error:
            self._drop_branch_locked(
                f"ROI scalar branch failed: {safe_error_summary(error)}"
            )
            return
        self._checkpoint(checkpoint)
        try:
            self._projection.publish(update, payload, branch.producer)
            self._checkpoint(checkpoint)
            branch.dataset.ingest_next(timeout=0.0)
        except CancellationRequested:
            raise
        except BaseException as error:
            self._drop_branch_locked(
                f"ROI scalar branch failed: {safe_error_summary(error)}"
            )

    def _create_branch_locked(
        self,
        binding: RoiScalarBinding,
        control_revision: int,
    ) -> _RoiScalarBranch:
        edge = self._edge_by_schema[binding.output_schema.fingerprint]
        if edge.schema.cell_schema is not binding.output_schema:
            raise ValueError("ROI binding did not retain the admitted schema owner")
        use_initial_identity = (
            self._initial_revision == control_revision
            and self._initial_scalar_block_id is not None
            and self._initial_scalar_edge is edge
        )
        block_id = (
            self._initial_scalar_block_id
            if use_initial_identity
            else BlockId(f"camera-monitor-roi-scalar-{uuid.uuid4().hex}")
        )
        assert block_id is not None
        stream = None
        producer = None
        tap = None
        dataset = None
        try:
            stream, producer = AcquisitionStream.create(
                self._scalar_stream_id,
                edge.payload_contract,
                flow_control=ProducerFlowControl.BACKPRESSURE_CAPABLE,
                retention_events=_STREAM_RETENTION_EVENTS,
                retention_bytes=(
                    _STREAM_RETENTION_EVENTS * edge.payload_max_retained_nbytes
                ),
            )
            tap = stream.monitor(
                max_events=_TAP_BACKLOG_EVENTS,
                max_bytes=_TAP_BACKLOG_EVENTS * edge.payload_max_retained_nbytes,
            )
            dataset = MonitorDataset.append_window(block_id, tap, edge)
            return _RoiScalarBranch(
                binding,
                control_revision,
                block_id,
                edge,
                stream,
                producer,
                dataset,
            )
        except BaseException:
            try:
                if dataset is not None:
                    dataset.close()
                elif tap is not None:
                    tap.close()
            except BaseException:
                pass
            if producer is not None:
                try:
                    producer.fail(StreamError("ROI scalar branch construction failed"))
                except BaseException:
                    pass
            raise

    def _drop_branch_locked(self, reason: str) -> None:
        branch, self._branch = self._branch, None
        if branch is None:
            return
        self._set_raw_roi_state_locked(
            branch.control_revision,
            failure_reason=reason,
        )
        self._retire_branch_locked(branch, error=StreamError(reason))

    def _set_raw_roi_state_locked(
        self,
        control_revision: int,
        *,
        failure_reason: str | None = None,
    ) -> None:
        self._roi_state = CameraMonitorRoiState(
            None,
            control_revision,
            None,
            None,
            None,
            self._next_roi_state_revision_locked(),
            failure_reason,
        )

    def _next_roi_state_revision_locked(self) -> int:
        return self._roi_state.state_revision + 1

    @staticmethod
    def _retire_branch_locked(
        branch: _RoiScalarBranch,
        *,
        replacement: StreamGenerationId | None = None,
        error: StreamError | None = None,
    ) -> None:
        try:
            if error is not None:
                branch.producer.fail(error)
            elif replacement is not None:
                branch.producer.supersede(replacement)
            else:
                branch.producer.finish()
        except BaseException:
            pass
        try:
            branch.dataset.close()
        except BaseException:
            pass

    def _terminate_controls_locked(self, reason: str) -> None:
        if self._controls_terminated:
            return
        self._controls_terminated = True
        self._roi_consumer.terminate(reason)

    @staticmethod
    def _checkpoint(checkpoint: Callable[[], None] | None) -> None:
        if checkpoint is not None:
            checkpoint()

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
    io_timeout_seconds: float
    prepare_attempted: bool = False
    view_notifications_enabled: bool = True

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
            self.dataset.ingest_next(context.checkpoint)
            self._notify_view_updated()

    def _notify_view_updated(self) -> None:
        """Disable a failed display notification without failing acquisition."""

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
        "_roi_binding",
        "_roi_control_edges",
        "_max_roi_reduction_scratch_nbytes",
        "_scalar_edge",
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
        metadata_contract = RoiScalarMetadataContract(
            capability.payload_contract.metadata_contract
        )
        control_edges: list[FrozenDatasetEdge[RoiScalarSample]] = []
        seen_output_schemas: set[str] = set()
        scratch_bounds: list[int] = []
        for reduction in (
            ReductionMethod.MEAN,
            ReductionMethod.SUM,
            ReductionMethod.MAX,
        ):
            try:
                output_schema = roi_scalar_output_schema(
                    capability.payload_contract,
                    reduction,
                )
                scratch_bounds.append(
                    max_roi_scalar_reduction_scratch_nbytes(
                        capability.payload_contract,
                        reduction,
                    )
                )
            except TypeError:
                # MAX has no meaningful order for complex camera values.  The
                # other two families still form a complete admitted catalog.
                continue
            if output_schema.fingerprint in seen_output_schemas:
                continue
            seen_output_schemas.add(output_schema.fingerprint)
            scalar_contract = RoiScalarSampleContract(
                output_schema,
                metadata_contract,
            )
            scalar_schema = DatasetSchema(
                AxisSpec(
                    _ROI_SCALAR_REPEAT_AXIS_ID,
                    "ROI scalar monitor storage repeat",
                    REPEAT,
                    1,
                    (0,),
                ),
                (
                    AxisSpec(
                        _ROI_SCALAR_HISTORY_AXIS_ID,
                        "newest-first ROI scalar history",
                        MONITOR_HISTORY,
                        request.scalar_history_capacity,
                    ),
                ),
                PointLayout.rect_c((request.scalar_history_capacity,)),
                output_schema,
            )
            control_edges.append(
                FrozenDatasetEdge(
                    scalar_schema,
                    RoiScalarDatasetEventAdapter(scalar_contract),
                )
            )
        if not control_edges or not scratch_bounds:
            raise ValueError("camera contract admits no ROI scalar reduction schema")
        self._roi_control_edges = tuple(control_edges)
        self._max_roi_reduction_scratch_nbytes = max(scratch_bounds)
        self._roi_binding = (
            None
            if request.roi is None
            else RoiScalarBinding(
                capability.payload_contract,
                request.roi,
                request.roi_reduction,
                request.roi_validity_policy,
                _edge_for_output_schema(
                    self._roi_control_edges,
                    roi_scalar_output_schema(
                        capability.payload_contract,
                        request.roi_reduction,
                    ),
                ).schema.cell_schema,
            )
        )
        if self._roi_binding is None:
            self._scalar_edge = None
        else:
            self._scalar_edge = _edge_for_output_schema(
                self._roi_control_edges,
                self._roi_binding.output_schema,
            )
        base_peak = _base_monitor_peak_bytes(
            port,
            self._edge,
            self._roi_control_edges,
            self._max_roi_reduction_scratch_nbytes,
        )
        if base_peak > request.memory_limit_bytes:
            raise MemoryError(
                f"camera monitor base peak {base_peak} exceeds limit "
                f"{request.memory_limit_bytes}"
            )
        self._descriptor = CameraMonitorDescriptor(
            "Camera monitor",
            request.camera_ref.role,
            schema,
            None if self._scalar_edge is None else self._scalar_edge.schema,
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

    @property
    def scalar_view_schema(self) -> DatasetSchema | None:
        return None if self._scalar_edge is None else self._scalar_edge.schema

    @property
    def roi_control_schemas(self) -> tuple[DatasetSchema, ...]:
        return tuple(edge.schema for edge in self._roi_control_edges)

    @property
    def request(self) -> CameraMonitorRequest:
        return self._request

    @property
    def roi_binding(self) -> RoiScalarBinding | None:
        return self._roi_binding

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
            (
                None
                if self._scalar_edge is None
                else BlockId(f"camera-monitor-roi-scalar-{uuid.uuid4().hex}")
            ),
            self._scalar_edge,
            self._roi_binding,
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
            roi_control_edges=self._roi_control_edges,
            max_roi_reduction_scratch_nbytes=(
                self._max_roi_reduction_scratch_nbytes
            ),
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
    scalar_edges: tuple[FrozenDatasetEdge[RoiScalarSample], ...],
    max_roi_reduction_scratch_nbytes: int,
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
    raw_peak = (
        capability.driver_ring_bytes
        + capability.adapter_record_retention_bytes
        + (_STREAM_RETENTION_EVENTS * payload)
        # Raw display and the stable ROI processor ingress each own a monitor
        # tap from the unchanged source generation.
        + (_TAP_BACKLOG_EVENTS * payload)
        + (_TAP_BACKLOG_EVENTS * payload)
        + mutable_materializer
        + immutable_snapshot
        + reorder_scratch
        + (history_cells * edge.metadata_max_retained_nbytes)
    )
    edges = tuple(scalar_edges)
    if not edges:
        raise ValueError("scalar_edges cannot be empty")
    branch_peaks = sorted(
        (_scalar_branch_peak_bytes(scalar_edge) for scalar_edge in edges),
        reverse=True,
    )
    # A schema migration commits its replacement before superseding the old
    # branch.  Same-schema retarget likewise owns old plus shadow append
    # storage until its first replacement event commits.  Two full branch
    # peaks conservatively close both overlap shapes with one admitted bound.
    scalar_overlap = (
        sum(branch_peaks[:2])
        if len(branch_peaks) > 1
        else 2 * branch_peaks[0]
    )
    return (
        raw_peak
        + scalar_overlap
        + nonnegative_integer(
            max_roi_reduction_scratch_nbytes,
            "max_roi_reduction_scratch_nbytes",
        )
    )


def _scalar_branch_peak_bytes(
    edge: FrozenDatasetEdge[RoiScalarSample],
) -> int:
    scalar_schema = edge.schema
    scalar_cells = (
        scalar_schema.repeat_axis.size * scalar_schema.point_layout.storage_size
    )
    scalar_mutable = mutable_dataset_storage_nbytes(scalar_schema)
    scalar_snapshot = dataset_storage_nbytes(scalar_schema)
    scalar_reorder = scalar_mutable if scalar_cells > 1 else 0
    scalar_payload = edge.payload_max_retained_nbytes
    return (
        (_STREAM_RETENTION_EVENTS * scalar_payload)
        + (_TAP_BACKLOG_EVENTS * scalar_payload)
        + scalar_mutable
        + scalar_snapshot
        + scalar_reorder
        + (scalar_cells * edge.metadata_max_retained_nbytes)
    )


def _edge_for_output_schema(
    edges: tuple[FrozenDatasetEdge[RoiScalarSample], ...],
    output_schema: ValueSchema,
) -> FrozenDatasetEdge[RoiScalarSample]:
    if not isinstance(output_schema, ValueSchema):
        raise TypeError("output_schema must be ValueSchema")
    for edge in edges:
        if edge.schema.cell_schema.fingerprint == output_schema.fingerprint:
            return edge
    raise ValueError("ROI output schema was not admitted")


def _compile_camera_monitor_plan(
    request: CameraMonitorRequest,
    port: BoundCameraMonitorPort,
    view: CameraMonitorViewPort,
    *,
    roi_control_edges: tuple[FrozenDatasetEdge[RoiScalarSample], ...],
    max_roi_reduction_scratch_nbytes: int,
) -> RunPlan[_CameraMonitorTransaction, None, None]:
    spec = getattr(view, "spec", None)
    if not isinstance(spec, CameraMonitorViewSpec):
        raise TypeError("camera monitor view has no CameraMonitorViewSpec")

    def preflight(context: RunContext) -> _CameraMonitorTransaction:
        stream = None
        producer = None
        raw_tap = None
        processor_tap = None
        raw_dataset = None
        projection = None
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
            raw_tap = stream.monitor(
                max_events=_TAP_BACKLOG_EVENTS,
                max_bytes=(
                    _TAP_BACKLOG_EVENTS
                    * port.capability.payload_contract.max_retained_nbytes
                ),
            )
            raw_dataset = MonitorDataset.append_window(
                spec.block_id,
                raw_tap,
                spec.dataset_edge,
            )
            processor_tap = stream.monitor(
                max_events=_TAP_BACKLOG_EVENTS,
                max_bytes=(
                    _TAP_BACKLOG_EVENTS
                    * port.capability.payload_contract.max_retained_nbytes
                ),
            )
            projection = RoiScalarStreamProjection(processor_tap)
            dataset = CameraMonitorLiveDataset(
                raw_dataset,
                projection=projection,
                input_contract=port.capability.payload_contract,
                scalar_edges=roi_control_edges,
                scalar_stream_id=StreamId(
                    f"camera-monitor-roi-scalar:{request.camera_ref.role}"
                ),
                max_reduction_scratch_nbytes=(
                    max_roi_reduction_scratch_nbytes
                ),
                initial_scalar_block_id=spec.scalar_block_id,
                initial_scalar_edge=spec.scalar_dataset_edge,
                initial_binding=spec.roi_binding,
            )
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
            else:
                for close in (
                    None if projection is None else projection.close,
                    None if processor_tap is None else processor_tap.close,
                    None if raw_dataset is None else raw_dataset.close,
                    None if raw_tap is None else raw_tap.close,
                ):
                    if close is None:
                        continue
                    try:
                        close()
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
    "CameraMonitorLiveDataset",
    "CameraMonitorRequest",
    "CameraMonitorRoiState",
    "CameraMonitorSnapshot",
    "CameraMonitorViewPort",
    "CameraMonitorViewSpec",
    "PreparedCameraMonitor",
    "prepare_camera_monitor",
]
