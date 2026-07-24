"""API-slot segmented camera binding owned by Pulse Scan."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import AxisId, AxisSpec, DatasetSchema, PointLayout, READOUT_EVENT, REPEAT
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraSampleContract,
)
from zlc_neutral_atom.logic_nodes.camera_capture.pipeline import BoundMeasurement
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress, DatasetCellSchedule
from zlc_neutral_atom.timing.capture_plan import compile_capture_cell_plan
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding
from zlc_neutral_atom.devices.sequencer.port import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    bind_pulse_document_target,
    compile_pulse_artifact,
)
from zlc_storage import canonical_text

from zlc_neutral_atom.logic_nodes.camera_measurement.binding import (
    CameraCaptureBindingRequest,
    bind_camera_measurement,
)
from .contracts import ApiSlotSegmentedProgram, ScanPointTable
from .segmented import ApiSlotPointDescriptor


def _axis(axis_id: AxisId, role, size: int) -> AxisSpec:
    return AxisSpec(axis_id, axis_id.value, role, size, tuple(range(size)))


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

def bind_api_slot_segmented_camera_acquisition(
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    *,
    program: ApiSlotSegmentedProgram,
    trigger_channel: str | None,
    repeat_axis_id: AxisId,
    readout_event_axis_id: AxisId,
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
    for document in point_documents:
        artifact = compile_pulse_artifact(
            document,
            clock_hz=pulse_port.capability.clock_hz,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channels=(selected_trigger,),
            live_target=pulse_port.capability.target,
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

__all__ = [
    "ApiSlotSegmentedCameraBinding",
    "bind_api_slot_segmented_camera_acquisition",
]
