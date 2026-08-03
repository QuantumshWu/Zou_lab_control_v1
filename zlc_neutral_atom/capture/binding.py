"""Physical camera-Dataset binding and finite pulse/camera composition.

This module owns the one place where installation identity, physical trigger
wiring, compiled pulse cardinality, named sampling axes, the exact cell plan,
and the bound camera capture are joined.  It is deliberately a concrete
composition helper rather than a public workflow or a generic graph builder.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraCaptureDescriptor,
    CameraCaptureSpec,
    CameraDatasetEventAdapter,
    CameraEventReadoutSetting,
    CameraSampleContract,
    ReadoutBindingKey,
)
from .pipeline import BoundCameraCapture
from .session import CameraCaptureContract, CameraCaptureProvenance
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.timing.capture_plan import (
    CompiledCaptureCellPlan,
    compile_capture_cell_plan,
)
from zlc_neutral_atom.devices.sequencer.port import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
)
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseDocument,
    PulseExecutionForm,
    bind_pulse_document_target,
    compile_pulse_artifact,
)
from zlc_neutral_atom.runtime.dataset import DatasetCellSchedule, FrozenDatasetEdge
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_storage import (
    canonical_text,
    positive_integer as _positive_int,
)


@dataclass(frozen=True, slots=True)
class CameraCaptureBindingRequest:
    """Physical intent for one finite camera-backed Dataset binding."""

    camera_instance_id: str
    dataset_schema: DatasetSchema
    cell_schedule: DatasetCellSchedule
    mode: CameraAcquisitionMode
    event_settings: tuple[CameraEventReadoutSetting, ...] | None = None

    def __post_init__(self) -> None:
        canonical_text(self.camera_instance_id, "camera_instance_id")
        if not isinstance(self.dataset_schema, DatasetSchema):
            raise TypeError("dataset_schema must be DatasetSchema")
        if not isinstance(self.cell_schedule, DatasetCellSchedule):
            raise TypeError("cell_schedule must be DatasetCellSchedule")
        self.cell_schedule.validate_schema(self.dataset_schema)
        if not isinstance(self.mode, CameraAcquisitionMode):
            raise TypeError("mode must be CameraAcquisitionMode")
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


def _source_group_sizes(
    request: CameraCaptureBindingRequest,
) -> tuple[int, ...]:
    """Derive the sole frame-group truth from the frozen Dataset schedule."""

    dataset_schema = request.dataset_schema
    event_columns = tuple(
        column
        for column in dataset_schema.point_table.columns
        if column.role == READOUT_EVENT
    )
    if not event_columns:
        return (1,) * len(request.cell_schedule)
    if len(event_columns) != 1:
        raise ValueError("camera Dataset has multiple READOUT_EVENT columns")
    event_column = event_columns[0]
    event_values = tuple(dict.fromkeys(event_column.values))
    if event_values != tuple(range(len(event_values))):
        raise ValueError("READOUT_EVENT values must be canonical zero-based indices")
    event_count = len(event_values)
    groups: list[int] = []
    current_identity: tuple[int, tuple[int, ...]] | None = None
    expected_event_index = 0
    for address in request.cell_schedule:
        event_index = event_column.values[address.point_ordinal]
        if not isinstance(event_index, int):
            raise TypeError("READOUT_EVENT value must be an integer")
        base_point_ordinal, event_remainder = divmod(
            address.point_ordinal,
            event_count,
        )
        if event_index != event_remainder:
            raise ValueError("camera Dataset rows are not base-major/event-minor")
        identity = (
            address.repeat_index,
            (base_point_ordinal,),
        )
        if identity != current_identity:
            if current_identity is not None and expected_event_index != event_count:
                raise ValueError(
                    "camera cell schedule splits an incomplete READOUT_EVENT group"
                )
            current_identity = identity
            expected_event_index = 0
        if event_index != expected_event_index:
            raise ValueError(
                "camera cell schedule must order each READOUT_EVENT group from zero"
            )
        expected_event_index += 1
        if expected_event_index == event_count:
            groups.append(event_count)
    if current_identity is None or expected_event_index != event_count:
        raise ValueError("camera cell schedule ends inside a READOUT_EVENT group")
    return tuple(groups)


def bind_camera_capture(
    port: BoundCapturePort,
    request: CameraCaptureBindingRequest,
) -> BoundCameraCapture:
    """Bind one resolved camera Port to a physical finite Dataset contract."""

    if not isinstance(port, BoundCapturePort):
        raise TypeError("port must be BoundCapturePort")
    if not isinstance(request, CameraCaptureBindingRequest):
        raise TypeError("request must be CameraCaptureBindingRequest")
    capability = port.capability
    evidence = capability.camera_capability_evidence
    payload_contract = capability.payload_contract
    if not isinstance(payload_contract, CameraSampleContract):
        raise TypeError("camera capability payload contract has the wrong type")
    dataset_schema = request.dataset_schema
    if dataset_schema.cell_schema is not payload_contract.value_schema:
        raise ValueError(
            "camera Dataset must share the capability-owned ValueSchema instance"
        )
    facts = evidence.physical_facts
    cell_schedule = request.cell_schedule
    source_group_sizes = _source_group_sizes(request)
    capture_spec = CameraCaptureSpec(
        request.mode,
        len(cell_schedule),
        source_group_sizes,
    )
    readout_columns = tuple(
        column
        for column in dataset_schema.point_table.columns
        if column.role == READOUT_EVENT
    )
    if len(readout_columns) > 1:
        raise ValueError("camera Dataset has multiple READOUT_EVENT columns")
    event_count = (
        1
        if not readout_columns
        else len(tuple(dict.fromkeys(readout_columns[0].values)))
    )
    if request.event_settings is None:
        if event_count != 1:
            raise ValueError(
                "multi-event camera capture requires explicit event_settings"
            )
        event_settings = (facts.event_setting(0),)
    else:
        event_settings = request.event_settings
    expected_indices = (0,) if not readout_columns else tuple(range(event_count))
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
            None if not readout_columns else readout_columns[0].coordinate_id
        ),
        event_settings=event_settings,
    )
    camera_provenance = CameraCaptureProvenance(
        descriptor=descriptor,
        binding=ReadoutBindingKey(request.camera_instance_id),
        binding_stamp=capability.binding_stamp,
    )
    capture_contract = CameraCaptureContract(
        stream_id=StreamId(f"camera.{request.camera_instance_id}.frames"),
        dataset_edge=FrozenDatasetEdge(
            dataset_schema,
            CameraDatasetEventAdapter(payload_contract),
            cell_schedule,
        ),
        capability=capability,
        camera_provenance=camera_provenance,
    )
    return BoundCameraCapture(
        port,
        capture_contract,
        capture_spec,
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
    """Named sampling intent; PointTable is the sole authored scan-row truth."""

    repeat_axis: AxisSpec
    readout_event_axis_id: AxisId
    ordinal_scan_axis_id: AxisId | None = None
    readout_events_per_repeat: int | None = None
    within_point_grouping: tuple[tuple[int, int], ...] | None = None
    scan_point_table: PointTable | None = None
    scan_grid_topology: GridTopology | None = None

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
        if self.scan_point_table is None:
            if self.scan_grid_topology is not None:
                raise ValueError("GridTopology requires a declared PointTable")
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
                    "a declared PointTable replaces the ordinal scan identity"
                )
            if not isinstance(self.scan_point_table, PointTable):
                raise TypeError("scan_point_table must be PointTable or None")
            columns = self.scan_point_table.columns
            if any(column.role != SCAN_POINT for column in columns):
                raise ValueError("scan PointTable columns must use SCAN_POINT")
            if len(
                {
                    self.repeat_axis.axis_id,
                    self.readout_event_axis_id,
                    *(column.coordinate_id for column in columns),
                }
            ) != len(columns) + 2:
                raise ValueError("triggered-camera sampling AxisIds must be distinct")
            if self.scan_grid_topology is not None and not isinstance(
                self.scan_grid_topology,
                GridTopology,
            ):
                raise TypeError("scan_grid_topology must be GridTopology or None")


@dataclass(frozen=True, slots=True)
class TriggeredCameraBinding:
    """Run-local, generation-pinned result of the concrete composition join."""

    pulse_port: BoundPulsePort
    pulse_request: FinitePulseExecutionRequest
    trigger_channel: str
    capture: BoundCameraCapture
    cell_plan: CompiledCaptureCellPlan

    @property
    def compiled_artifact(self) -> CompiledPulseArtifact:
        return self.pulse_request.artifact

    @property
    def expected_frames(self) -> int:
        return self.cell_plan.total_events




def _ordinary_point_table(axis_id: AxisId, count: int) -> PointTable:
    if count == 1:
        return PointTable(1)
    return PointTable(
        count,
        (
            PointColumn(
                axis_id,
                axis_id.value,
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(range(count)),
            ),
        ),
    )


def _expand_readout_events(
    point_table: PointTable,
    topology: GridTopology | None,
    event_axis_id: AxisId,
    event_count: int,
) -> tuple[PointTable, GridTopology | None]:
    columns = tuple(
        PointColumn(
            column.coordinate_id,
            column.name,
            column.role,
            column.value_kind,
            tuple(
                value
                for value in column.values
                for _event in range(event_count)
            ),
            column.unit,
            column.coordinate_frame,
        )
        for column in point_table.columns
    )
    event_column = PointColumn(
        event_axis_id,
        event_axis_id.value,
        READOUT_EVENT,
        PointColumn.NUMERIC,
        tuple(
            event
            for _point in range(point_table.row_count)
            for event in range(event_count)
        ),
    )
    expanded_table = PointTable(
        point_table.row_count * event_count,
        (*columns, event_column),
    )
    if topology is None:
        return expanded_table, None
    expanded_topology = GridTopology(
        (*topology.dimension_ids, event_axis_id),
        (*topology.coordinate_domains, tuple(range(event_count))),
        tuple(
            (*cell, event)
            for cell in topology.row_to_cell
            for event in range(event_count)
        ),
    )
    return expanded_table, expanded_topology




def bind_triggered_camera_acquisition(
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    *,
    pulse_document: PulseDocument,
    execution_form: PulseExecutionForm,
    trigger_channel: str | None,
    layout: TriggeredCameraLayout,
    camera_instance_id: str,
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
    if execution_form in (
        PulseExecutionForm.CONTINUOUS_MONITOR,
        PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    ):
        raise ValueError("triggered camera acquisition requires a finite pulse form")
    if not isinstance(layout, TriggeredCameraLayout):
        raise TypeError("layout must be TriggeredCameraLayout")
    camera_instance_id = canonical_text(
        camera_instance_id,
        "camera_instance_id",
    )
    document = bind_pulse_document_target(
        pulse_document,
        pulse_port.capability.target,
    )
    camera_capability = camera_port.capability
    camera_evidence = camera_capability.camera_capability_evidence
    camera_facts = camera_evidence.physical_facts
    camera_payload_contract = camera_capability.payload_contract
    if not isinstance(camera_payload_contract, CameraSampleContract):
        raise TypeError("camera capability payload contract has the wrong type")
    if trigger_channel is None:
        raise ValueError(
            "exact capture requires the pulse-owned trigger_channel in its binding"
        )
    selected_trigger = canonical_text(trigger_channel, "trigger_channel")

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
        and layout.scan_point_table is not None
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
    if layout.scan_point_table is None:
        assert layout.ordinal_scan_axis_id is not None
        base_point_table = _ordinary_point_table(
            layout.ordinal_scan_axis_id,
            schedule.point_count,
        )
        base_topology = None
    else:
        base_point_table = layout.scan_point_table
        base_topology = layout.scan_grid_topology
        expected_execution_points = (
            layout.repeat_axis.size * base_point_table.row_count
            if repeat_major_points
            else base_point_table.row_count
        )
        if repeat_major_points and (
            schedule.loop_count != 1 or not schedule.full_point_loop
        ):
            raise ValueError(
                "autonomous scan requires one complete repeat-major finite table"
            )
        if expected_execution_points != schedule.point_count:
            raise ValueError(
                "declared PointTable rows differ from compiled pulse points"
            )
    point_table, topology = _expand_readout_events(
        base_point_table,
        base_topology,
        layout.readout_event_axis_id,
        events_per_repeat,
    )
    dataset_schema = DatasetSchema(
        repeat_axis,
        point_table,
        topology,
        camera_payload_contract.value_schema,
    )
    cell_plan = compile_capture_cell_plan(
        artifact,
        selected_trigger,
        dataset_schema,
        readout_event_axis_id=layout.readout_event_axis_id,
        base_point_count=base_point_table.row_count,
        within_point_grouping=layout.within_point_grouping,
    )
    camera_capture = bind_camera_capture(
        camera_port,
        CameraCaptureBindingRequest(
            camera_instance_id,
            dataset_schema,
            cell_plan.cell_schedule,
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
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
        camera_capture,
        cell_plan,
    )


__all__ = [
    "bind_camera_capture",
    "TriggeredCameraBinding",
    "TriggeredCameraLayout",
    "CameraCaptureBindingRequest",
    "bind_triggered_camera_acquisition",
]
