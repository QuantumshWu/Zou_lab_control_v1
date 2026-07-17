"""Private composition seam for one finite pulse-triggered camera acquisition.

This module owns the one place where installation identity, physical trigger
wiring, compiled pulse cardinality, named sampling axes, the exact cell plan,
and the bound camera measurement are joined.  It is deliberately a concrete
composition helper rather than a public workflow or a generic graph builder.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
)
from zlc_neutral_atom.acquisition import CameraAcquisitionMode, CameraSampleContract
from zlc_neutral_atom.runtime.pipeline import BoundMeasurement
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
)
from zlc_neutral_atom.scan.contracts import (
    ApiSlotSegmentedProgram,
    ScanPointTable,
)
from zlc_neutral_atom.timing.capture_plan import (
    CompiledCaptureCellPlan,
    compile_capture_cell_plan,
)
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding
from zlc_neutral_atom.timing.pulse import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
)
from zlc_neutral_atom.timing.segmented import (
    ApiSlotPointDescriptor,
    admit_api_slot_segmented_control_memory,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseDocument,
    PulseExecutionForm,
    bind_pulse_document_target,
    compile_pulse_artifact,
    compiled_pulse_retained_upper_bound_bytes,
)
from zlc_storage import canonical_text, positive_integer as _positive_int

from ._camera_endpoint import (
    CameraCaptureBindingRequest,
    bind_camera_measurement,
)

def _canonical_grouping(
    value: tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    try:
        grouping = tuple(tuple(pair) for pair in value)
    except TypeError as exc:
        raise TypeError(
            "within_point_grouping must contain integer (repeat, event) pairs"
        ) from exc
    if any(
        len(pair) != 2
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in pair
        )
        for pair in grouping
    ):
        raise ValueError(
            "within_point_grouping must contain non-negative two-item pairs"
        )
    return grouping


@dataclass(frozen=True, slots=True)
class TriggeredCameraLayout:
    """Named sampling intent; the compiled schedule supplies scan cardinality.

    Ordinary captures may leave ``scan_axes`` absent and receive one explicit
    ordinal axis.  A scan authority instead supplies its physical axes and
    sparse/rectangular row mapping together; this binding never infers those
    semantics from the numeric table shape.
    """

    repeat_axis: AxisSpec
    readout_event_axis_id: AxisId
    ordinal_scan_axis_id: AxisId | None = None
    readout_events_per_repeat: int | None = None
    within_point_grouping: tuple[tuple[int, int], ...] | None = None
    scan_axes: tuple[AxisSpec, ...] | None = None
    scan_point_layout: PointLayout | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repeat_axis, AxisSpec)
            or self.repeat_axis.role != REPEAT
        ):
            raise ValueError("repeat_axis must be an AxisSpec with repeat role")
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if self.readout_events_per_repeat is not None:
            object.__setattr__(
                self,
                "readout_events_per_repeat",
                _positive_int(
                    self.readout_events_per_repeat,
                    "readout_events_per_repeat",
                ),
            )
        object.__setattr__(
            self,
            "within_point_grouping",
            _canonical_grouping(self.within_point_grouping),
        )
        if self.scan_axes is None:
            if self.scan_point_layout is not None:
                raise ValueError("scan_point_layout requires declared scan_axes")
            if not isinstance(self.ordinal_scan_axis_id, AxisId):
                raise TypeError(
                    "ordinary capture requires one ordinal_scan_axis_id"
                )
            axis_ids = {
                self.repeat_axis.axis_id,
                self.ordinal_scan_axis_id,
                self.readout_event_axis_id,
            }
            if len(axis_ids) != 3:
                raise ValueError("triggered-camera sampling AxisIds must be distinct")
        else:
            if self.ordinal_scan_axis_id is not None:
                raise ValueError(
                    "physical scan_axes replace the ordinal scan axis identity"
                )
            axes = tuple(self.scan_axes)
            if not axes or any(
                not isinstance(axis, AxisSpec) or axis.role != SCAN_POINT
                for axis in axes
            ):
                raise ValueError("scan_axes must contain SCAN_POINT AxisSpec values")
            if len({axis.axis_id for axis in axes}) != len(axes):
                raise ValueError("scan_axes must have unique AxisIds")
            if len(
                {
                    self.repeat_axis.axis_id,
                    self.readout_event_axis_id,
                    *(axis.axis_id for axis in axes),
                }
            ) != len(axes) + 2:
                raise ValueError("triggered-camera sampling AxisIds must be distinct")
            if not isinstance(self.scan_point_layout, PointLayout):
                raise TypeError("declared scan_axes require a PointLayout")
            if self.scan_point_layout.logical_shape != tuple(
                axis.size for axis in axes
            ):
                raise ValueError("scan PointLayout shape differs from scan_axes")
            object.__setattr__(self, "scan_axes", axes)


@dataclass(frozen=True, slots=True)
class TriggeredCameraBinding:
    """Run-local, generation-pinned result of the concrete composition join."""

    pulse_port: BoundPulsePort
    pulse_request: FinitePulseExecutionRequest
    trigger_channel: str
    measurement: BoundMeasurement
    cell_plan: CompiledCaptureCellPlan

    @property
    def compiled_artifact(self) -> CompiledPulseArtifact:
        return self.pulse_request.artifact

    @property
    def expected_frames(self) -> int:
        return self.cell_plan.total_events


@dataclass(frozen=True, slots=True)
class ApiSlotSegmentedCameraBinding:
    """Target-bound API program and its one-arm ordered segment authority."""

    pulse_port: BoundPulsePort
    trigger_channel: str
    measurement: BoundMeasurement
    program: ApiSlotSegmentedProgram
    point_descriptors: tuple[ApiSlotPointDescriptor, ...]
    compiled_artifacts: tuple[CompiledPulseArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_port, BoundPulsePort):
            raise TypeError("pulse_port must be BoundPulsePort")
        canonical_text(self.trigger_channel, "trigger_channel")
        if not isinstance(self.measurement, BoundMeasurement):
            raise TypeError("measurement must be BoundMeasurement")
        if not isinstance(self.program, ApiSlotSegmentedProgram):
            raise TypeError("program must be ApiSlotSegmentedProgram")
        descriptors = tuple(self.point_descriptors)
        artifacts = tuple(self.compiled_artifacts)
        point_count = self.program.point_count
        if len(descriptors) != point_count or any(
            not isinstance(value, ApiSlotPointDescriptor)
            for value in descriptors
        ):
            raise ValueError("point_descriptors must contain one descriptor per API row")
        if len(artifacts) != point_count or any(
            not isinstance(value, CompiledPulseArtifact) for value in artifacts
        ):
            raise ValueError("compiled_artifacts must contain one artifact per API row")
        if tuple(value.point_ordinal for value in descriptors) != tuple(
            range(point_count)
        ):
            raise ValueError("point descriptors differ from API table row order")
        if self.measurement.capture_contract.total_events != self.program.segment_count:
            raise ValueError("camera event count differs from API R by P cardinality")
        for point_index, descriptor in enumerate(descriptors):
            if descriptor.pulse_request.artifact is not artifacts[point_index]:
                raise ValueError("point descriptor uses another point artifact")
            if descriptor.pulse_binding.trigger_channel != self.trigger_channel:
                raise ValueError("point descriptor uses another trigger channel")
        object.__setattr__(self, "point_descriptors", descriptors)
        object.__setattr__(self, "compiled_artifacts", artifacts)

    @property
    def point_table(self) -> ScanPointTable:
        return self.program.point_table

    @property
    def dataset_schema(self) -> DatasetSchema:
        return self.measurement.capture_contract.dataset_schema

    @property
    def expected_frames(self) -> int:
        return self.program.segment_count


def _axis(axis_id: AxisId, role, size: int) -> AxisSpec:
    return AxisSpec(axis_id, axis_id.value, role, size, tuple(range(size)))


def bind_api_slot_segmented_camera_acquisition(
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    *,
    program: ApiSlotSegmentedProgram,
    trigger_channel: str | None,
    repeat_axis_id: AxisId,
    readout_event_axis_id: AxisId,
    transport_memory_limit_bytes: int,
    memory_limit_bytes: int,
) -> ApiSlotSegmentedCameraBinding:
    """Precompile P static rows and bind one R-major/P-fast camera arm."""

    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    if not isinstance(program, ApiSlotSegmentedProgram):
        raise TypeError("program must be ApiSlotSegmentedProgram")
    if not isinstance(repeat_axis_id, AxisId):
        raise TypeError("repeat_axis_id must be AxisId")
    if not isinstance(readout_event_axis_id, AxisId):
        raise TypeError("readout_event_axis_id must be AxisId")
    transport_memory_limit_bytes = _positive_int(
        transport_memory_limit_bytes,
        "transport_memory_limit_bytes",
    )
    memory_limit_bytes = _positive_int(memory_limit_bytes, "memory_limit_bytes")

    bound_program = ApiSlotSegmentedProgram(
        bind_pulse_document_target(
            program.document,
            pulse_port.capability.target,
        ),
        program.table,
        program.segmentation_rationale,
    )
    point_count = bound_program.point_count
    repeat_count = bound_program.repeat_count
    control_retained = admit_api_slot_segmented_control_memory(
        point_count,
        repeat_count,
        memory_limit_bytes,
    )

    camera_capability = camera_port.capability
    camera_evidence = camera_capability.camera_capability_evidence
    camera_facts = camera_evidence.physical_facts
    camera_payload_contract = camera_capability.payload_contract
    if not isinstance(camera_payload_contract, CameraSampleContract):
        raise TypeError("camera capability payload contract has the wrong type")
    if trigger_channel is None:
        if len(camera_facts.capture_trigger_channels) != 1:
            raise ValueError(
                "exact capture requires exactly one physical camera trigger channel"
            )
        selected_trigger = camera_facts.capture_trigger_channels[0]
    else:
        selected_trigger = canonical_text(trigger_channel, "trigger_channel")
    camera_facts.require_single_capture_trigger_channel(selected_trigger)

    # Point axes/layout and resolved documents are intentionally derived only
    # after the R*P control count is admitted.  The input table itself is
    # already the caller's frozen P data; the camera contract remains the sole
    # owner of its exact transport-memory formula below.
    point_table = bound_program.point_table
    if repeat_axis_id == readout_event_axis_id or repeat_axis_id in {
        axis.axis_id for axis in point_table.point_axes
    } or readout_event_axis_id in {
        axis.axis_id for axis in point_table.point_axes
    }:
        raise ValueError("segmented camera sampling AxisIds must be distinct")

    point_documents = bound_program.resolved_point_documents
    if len(point_documents) != point_count:
        raise RuntimeError("resolved API point documents do not cover P")

    artifacts_list: list[CompiledPulseArtifact] = []
    compiled_retained = control_retained
    for document in point_documents:
        artifact = compile_pulse_artifact(
            document,
            clock_hz=pulse_port.capability.clock_hz,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channels=(selected_trigger,),
            live_target=pulse_port.capability.target,
        )
        compiled_retained += compiled_pulse_retained_upper_bound_bytes(artifact)
        if compiled_retained > memory_limit_bytes:
            raise MemoryError(
                "API segmented compiled point artifacts require more than "
                f"{memory_limit_bytes} bytes"
            )
        artifacts_list.append(artifact)
    artifacts = tuple(artifacts_list)
    for artifact in artifacts:
        schedules = tuple(
            schedule
            for schedule in artifact.trigger_schedules
            if schedule.channel == selected_trigger
        )
        if len(schedules) != 1 or schedules[0].total != 1:
            raise ValueError(
                "each STATIC_ONCE API segment must emit exactly one camera trigger"
            )
    requests = tuple(
        FinitePulseExecutionRequest(document, artifact)
        for document, artifact in zip(point_documents, artifacts, strict=True)
    )

    repeat_axis = _axis(repeat_axis_id, REPEAT, repeat_count)
    event_axis = _axis(readout_event_axis_id, READOUT_EVENT, 1)
    point_axes = (*point_table.point_axes, event_axis)
    point_layout = PointLayout.from_mapping(
        tuple(axis.size for axis in point_axes),
        tuple(
            (*point_table.point_layout.multi_index(point_index), 0)
            for point_index in range(point_count)
        ),
    )
    dataset_schema = DatasetSchema(
        repeat_axis,
        point_axes,
        point_layout,
        camera_payload_contract.value_schema,
    )
    addresses = tuple(
        DatasetCellAddress(
            repeat_index,
            point_layout.storage_index(
                (*point_table.point_layout.multi_index(point_index), 0)
            ),
        )
        for repeat_index in range(repeat_count)
        for point_index in range(point_count)
    )
    cell_schedule = DatasetCellSchedule.from_cells(dataset_schema, addresses)
    measurement = bind_camera_measurement(
        camera_port,
        CameraCaptureBindingRequest(
            camera_evidence.source_id,
            repeat_axis,
            point_axes,
            point_layout,
            cell_schedule,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            0,
            transport_memory_limit_bytes,
            (camera_facts.event_setting(0),),
        ),
    )

    local_repeat_axis = _axis(repeat_axis_id, REPEAT, 1)
    local_schema = DatasetSchema(
        local_repeat_axis,
        (event_axis,),
        PointLayout.rect_c((1,)),
        camera_payload_contract.value_schema,
    )
    local_scan_layout = PointLayout.rect_c(())
    pulse_bindings = tuple(
        PulseCaptureBinding(
            artifact,
            selected_trigger,
            compile_capture_cell_plan(
                artifact,
                selected_trigger,
                local_schema,
                readout_event_axis_id=readout_event_axis_id,
                scan_point_layout=local_scan_layout,
            ),
        )
        for artifact in artifacts
    )
    point_descriptors = tuple(
        ApiSlotPointDescriptor(
            point_index,
            requests[point_index],
            pulse_bindings[point_index],
        )
        for point_index in range(point_count)
    )
    return ApiSlotSegmentedCameraBinding(
        pulse_port,
        selected_trigger,
        measurement,
        bound_program,
        point_descriptors,
        artifacts,
    )


def bind_triggered_camera_acquisition(
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    *,
    pulse_document: PulseDocument,
    execution_form: PulseExecutionForm,
    trigger_channel: str | None,
    layout: TriggeredCameraLayout,
    transport_memory_limit_bytes: int,
) -> TriggeredCameraBinding:
    """Bind one exact finite pulse/camera acquisition without starting hardware."""

    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    if not isinstance(pulse_document, PulseDocument):
        raise TypeError("pulse_document must be PulseDocument")
    if not isinstance(execution_form, PulseExecutionForm):
        raise TypeError("execution_form must be PulseExecutionForm")
    if execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
        raise ValueError("triggered camera acquisition requires a finite pulse form")
    if not isinstance(layout, TriggeredCameraLayout):
        raise TypeError("layout must be TriggeredCameraLayout")
    transport_memory_limit_bytes = _positive_int(
        transport_memory_limit_bytes,
        "transport_memory_limit_bytes",
    )

    document = bind_pulse_document_target(
        pulse_document,
        pulse_port.capability.target,
    )
    camera_capability = camera_port.capability
    camera_evidence = camera_capability.camera_capability_evidence
    camera_role = camera_evidence.source_id
    camera_facts = camera_evidence.physical_facts
    camera_payload_contract = camera_capability.payload_contract
    if not isinstance(camera_payload_contract, CameraSampleContract):
        raise TypeError("camera capability payload contract has the wrong type")
    if trigger_channel is None:
        if len(camera_facts.capture_trigger_channels) != 1:
            raise ValueError(
                "exact capture requires exactly one physical camera trigger channel"
            )
        selected_trigger = camera_facts.capture_trigger_channels[0]
    else:
        selected_trigger = canonical_text(trigger_channel, "trigger_channel")
    camera_facts.require_single_capture_trigger_channel(
        selected_trigger
    )

    artifact = compile_pulse_artifact(
        document,
        clock_hz=pulse_port.capability.clock_hz,
        execution_form=execution_form,
        trigger_channels=(selected_trigger,),
        live_target=pulse_port.capability.target,
    )
    schedule = artifact.trigger_schedules[0]
    if schedule.total < 1:
        raise ValueError("compiled pulse emits no camera trigger edge")
    point_counts = [0] * schedule.point_count
    for edge in schedule.iter_edges():
        point_counts[edge.point_index] += 1
    per_point_counts = tuple(point_counts)
    if len(set(per_point_counts)) != 1:
        raise ValueError("camera trigger count must be uniform across scan points")
    per_point = per_point_counts[0]
    repeat_major_points = (
        execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
        and layout.scan_axes is not None
    )
    events_per_repeat = layout.readout_events_per_repeat
    if events_per_repeat is None:
        if not repeat_major_points and layout.repeat_axis.size != 1:
            raise ValueError(
                "readout_events_per_repeat is required when repeat_count exceeds one"
            )
        events_per_repeat = per_point
    expected_per_point = (
        events_per_repeat
        if repeat_major_points
        else layout.repeat_axis.size * events_per_repeat
    )
    if expected_per_point != per_point:
        raise ValueError(
            "declared repeat/event layout differs from per-point trigger count"
        )

    repeat_axis = layout.repeat_axis
    if layout.scan_axes is None:
        assert layout.ordinal_scan_axis_id is not None
        scan_axes = (
            (_axis(layout.ordinal_scan_axis_id, SCAN_POINT, schedule.point_count),)
            if schedule.point_count > 1
            else ()
        )
        scan_point_layout = PointLayout.rect_c(
            tuple(axis.size for axis in scan_axes)
        )
    else:
        scan_axes = layout.scan_axes
        assert layout.scan_point_layout is not None
        scan_point_layout = layout.scan_point_layout
        expected_execution_points = (
            layout.repeat_axis.size * scan_point_layout.storage_size
            if repeat_major_points
            else scan_point_layout.storage_size
        )
        if repeat_major_points and (
            schedule.loop_count != 1 or not schedule.full_point_loop
        ):
            raise ValueError(
                "autonomous scan requires one complete repeat-major finite table"
            )
        if expected_execution_points != schedule.point_count:
            raise ValueError(
                "declared scan PointLayout rows differ from compiled pulse points"
            )
    event_axis = _axis(
        layout.readout_event_axis_id,
        READOUT_EVENT,
        events_per_repeat,
    )
    point_axes = (*scan_axes, event_axis)
    point_layout = PointLayout.from_mapping(
        tuple(axis.size for axis in point_axes),
        tuple(
            (*scan_point_layout.multi_index(scan_row), event_index)
            for scan_row in range(scan_point_layout.storage_size)
            for event_index in range(event_axis.size)
        ),
    )
    dataset_schema = DatasetSchema(
        repeat_axis,
        point_axes,
        point_layout,
        camera_payload_contract.value_schema,
    )
    cell_plan = compile_capture_cell_plan(
        artifact,
        selected_trigger,
        dataset_schema,
        readout_event_axis_id=event_axis.axis_id,
        scan_point_layout=scan_point_layout,
        within_point_grouping=layout.within_point_grouping,
    )
    measurement = bind_camera_measurement(
        camera_port,
        CameraCaptureBindingRequest(
            camera_role,
            repeat_axis,
            point_axes,
            point_layout,
            cell_plan.cell_schedule,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            0,
            transport_memory_limit_bytes,
            tuple(
                camera_facts.event_setting(index)
                for index in range(events_per_repeat)
            ),
        )
    )
    pulse_request = FinitePulseExecutionRequest(document, artifact)
    return TriggeredCameraBinding(
        pulse_port,
        pulse_request,
        selected_trigger,
        measurement,
        cell_plan,
    )


__all__: list[str] = []
