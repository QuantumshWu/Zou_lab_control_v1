"""Bind a resolved camera capture Port to the Camera Measurement contract."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import AxisSpec, DatasetSchema, PointLayout, READOUT_EVENT, REPEAT
from zlc_neutral_atom.devices.camera.contract import (
    CameraAcquisitionMode,
    CameraCaptureSpec,
    CameraDatasetEventAdapter,
    CameraSampleContract,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CAMERA_MEASUREMENT_DEFINITION,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    ReadoutBindingKey,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.logic_nodes.camera_capture.session import (
    CameraCaptureContract,
    CameraCaptureProvenance,
)
from zlc_neutral_atom.runtime.dataset import DatasetCellSchedule, FrozenDatasetEdge
from zlc_neutral_atom.logic_nodes.camera_capture.pipeline import BoundMeasurement
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_storage import canonical_text as _canonical_text
from zlc_storage import sha256_text as _sha256


@dataclass(frozen=True)
class CameraCaptureBindingRequest:
    """Composition request for one finite camera dataset binding."""

    role: str
    repeat_axis: AxisSpec
    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout
    cell_schedule: DatasetCellSchedule
    mode: CameraAcquisitionMode
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
    dataset_schema: DatasetSchema,
) -> tuple[int, ...]:
    """Derive the sole frame-group truth from the frozen dataset schedule."""

    request.cell_schedule.validate_schema(dataset_schema)
    event_positions = tuple(
        index
        for index, axis in enumerate(dataset_schema.point_axes)
        if axis.role == READOUT_EVENT
    )
    if not event_positions:
        return (1,) * len(request.cell_schedule)
    if len(event_positions) != 1:
        raise ValueError("camera dataset has multiple READOUT_EVENT axes")
    event_position = event_positions[0]
    event_count = dataset_schema.point_axes[event_position].size
    groups: list[int] = []
    current_identity: tuple[int, tuple[int, ...]] | None = None
    expected_event_index = 0
    for address in request.cell_schedule:
        multi_index = request.point_layout.multi_index(address.point_storage_index)
        event_index = multi_index[event_position]
        identity = (
            address.repeat_index,
            multi_index[:event_position] + multi_index[event_position + 1 :],
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


def bind_camera_measurement(
    port: BoundCapturePort,
    request: CameraCaptureBindingRequest,
) -> BoundMeasurement:
    """Bind one already-resolved camera Port to a finite dataset request."""

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
    source_group_sizes = _source_group_sizes(request, dataset_schema)
    capture_spec = freeze_camera_capture_spec(
        CameraCaptureSpec(
            request.mode,
            len(cell_schedule),
            source_group_sizes,
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
        camera_provenance=camera_provenance,
    )
    return BoundMeasurement(
        CAMERA_MEASUREMENT_DEFINITION,
        port,
        capture_contract,
        capture_spec,
    )


__all__ = ["CameraCaptureBindingRequest", "bind_camera_measurement"]
