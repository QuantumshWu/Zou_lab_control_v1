"""Owner tree mappings embedded by durable neutral-atom artifacts."""

from __future__ import annotations

from typing import Any

from zlc_data import AxisId, CoordinateFrameId, value_schema_from_tree, value_schema_to_tree
from zlc_storage import exact_mapping as _exact_map

from .contracts import (
    CalibrationCaptureLayout,
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    FrameContract,
    ReadoutBindingKey,
)


def readout_binding_key_to_tree(value: ReadoutBindingKey) -> dict[str, Any]:
    if not isinstance(value, ReadoutBindingKey):
        raise TypeError("value must be ReadoutBindingKey")
    return {"value": value.value}


def readout_binding_key_from_tree(tree: Any) -> ReadoutBindingKey:
    data = _exact_map(
        tree,
        {"value"},
        "readout binding key",
        discriminator=None,
    )
    return ReadoutBindingKey(data["value"])


def _camera_event_readout_setting_to_tree(
    value: CameraEventReadoutSetting,
) -> dict[str, Any]:
    if not isinstance(value, CameraEventReadoutSetting):
        raise TypeError("value must be CameraEventReadoutSetting")
    return {
        "event_index": value.event_index,
        "exposure_seconds": value.exposure_seconds,
        "gain": value.gain,
        "readout_mode": value.readout_mode,
        "opaque_frame_settings_fingerprint": (
            value.opaque_frame_settings_fingerprint
        ),
    }


def _camera_event_readout_setting_from_tree(tree: Any) -> CameraEventReadoutSetting:
    data = _exact_map(
        tree,
        {
            "event_index",
            "exposure_seconds",
            "gain",
            "readout_mode",
            "opaque_frame_settings_fingerprint",
        },
        "camera event readout setting",
        discriminator=None,
    )
    return CameraEventReadoutSetting(
        data["event_index"],
        data["exposure_seconds"],
        data["gain"],
        data["readout_mode"],
        data["opaque_frame_settings_fingerprint"],
    )


_CAMERA_FRAME_FIELDS = {
    "camera_identity",
    "sensor_identity",
    "optical_path",
    "sensor_shape_yx",
    "roi_origin_yx",
    "roi_shape_yx",
    "binning_yx",
    "spatial_y_axis_id",
    "spatial_x_axis_id",
    "coordinate_frame",
    "dtype",
    "count_unit",
}

_CAPTURE_FIELDS = _CAMERA_FRAME_FIELDS | {
    "readout_event_axis_id",
    "event_settings",
    "camera_arm_spec_fingerprint",
}


def _camera_frame_facts_to_tree(
    value: CameraCaptureDescriptor | FrameContract,
) -> dict[str, Any]:
    return {
        "camera_identity": value.camera_identity,
        "sensor_identity": value.sensor_identity,
        "optical_path": value.optical_path,
        "sensor_shape_yx": list(value.sensor_shape_yx),
        "roi_origin_yx": list(value.roi_origin_yx),
        "roi_shape_yx": list(value.roi_shape_yx),
        "binning_yx": list(value.binning_yx),
        "spatial_y_axis_id": value.spatial_y_axis_id.value,
        "spatial_x_axis_id": value.spatial_x_axis_id.value,
        "coordinate_frame": value.coordinate_frame.value,
        "dtype": value.dtype.str,
        "count_unit": value.count_unit,
    }


def _camera_frame_facts_from_tree(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "camera_identity": data["camera_identity"],
        "sensor_identity": data["sensor_identity"],
        "optical_path": data["optical_path"],
        "sensor_shape_yx": data["sensor_shape_yx"],
        "roi_origin_yx": data["roi_origin_yx"],
        "roi_shape_yx": data["roi_shape_yx"],
        "binning_yx": data["binning_yx"],
        "spatial_y_axis_id": AxisId(data["spatial_y_axis_id"]),
        "spatial_x_axis_id": AxisId(data["spatial_x_axis_id"]),
        "coordinate_frame": CoordinateFrameId(data["coordinate_frame"]),
        "dtype": data["dtype"],
        "count_unit": data["count_unit"],
    }


def camera_capture_descriptor_to_tree(value: CameraCaptureDescriptor) -> dict[str, Any]:
    if not isinstance(value, CameraCaptureDescriptor):
        raise TypeError("value must be CameraCaptureDescriptor")
    return {
        **_camera_frame_facts_to_tree(value),
        "readout_event_axis_id": (
            None if value.readout_event_axis_id is None else value.readout_event_axis_id.value
        ),
        "event_settings": [
            _camera_event_readout_setting_to_tree(item) for item in value.event_settings
        ],
        "camera_arm_spec_fingerprint": value.camera_arm_spec_fingerprint,
    }


def camera_capture_descriptor_from_tree(tree: Any) -> CameraCaptureDescriptor:
    data = _exact_map(
        tree,
        _CAPTURE_FIELDS,
        "camera capture descriptor",
        discriminator=None,
    )
    event_axis = data["readout_event_axis_id"]
    value = CameraCaptureDescriptor(
        **_camera_frame_facts_from_tree(data),
        readout_event_axis_id=(
            None
            if event_axis is None
            else AxisId(event_axis)
        ),
        event_settings=tuple(
            _camera_event_readout_setting_from_tree(item)
            for item in data["event_settings"]
        ),
        camera_arm_spec_fingerprint=data["camera_arm_spec_fingerprint"],
    )
    return value


_FRAME_FIELDS = _CAMERA_FRAME_FIELDS | {
    "binding",
    "exposure_seconds",
    "gain",
    "readout_mode",
    "opaque_frame_settings_fingerprint",
    "frame_schema",
}


def frame_contract_to_tree(value: FrameContract) -> dict[str, Any]:
    if not isinstance(value, FrameContract):
        raise TypeError("value must be FrameContract")
    return {
        "binding": readout_binding_key_to_tree(value.binding),
        **_camera_frame_facts_to_tree(value),
        "exposure_seconds": value.exposure_seconds,
        "gain": value.gain,
        "readout_mode": value.readout_mode,
        "opaque_frame_settings_fingerprint": value.opaque_frame_settings_fingerprint,
        "frame_schema": value_schema_to_tree(value.frame_schema),
    }


def frame_contract_from_tree(tree: Any) -> FrameContract:
    data = _exact_map(
        tree,
        _FRAME_FIELDS,
        "frame contract",
        discriminator=None,
    )
    value = FrameContract(
        binding=readout_binding_key_from_tree(data["binding"]),
        **_camera_frame_facts_from_tree(data),
        exposure_seconds=data["exposure_seconds"],
        gain=data["gain"],
        readout_mode=data["readout_mode"],
        opaque_frame_settings_fingerprint=data["opaque_frame_settings_fingerprint"],
        frame_schema=value_schema_from_tree(data["frame_schema"]),
    )
    return value


def calibration_capture_layout_to_tree(
    value: CalibrationCaptureLayout,
) -> dict[str, Any]:
    if not isinstance(value, CalibrationCaptureLayout):
        raise TypeError("value must be CalibrationCaptureLayout")
    return {
        "readout_event_axis_id": value.readout_event_axis_id.value,
        "reference_event_indices": list(value.reference_event_indices),
        "readout_event_index": value.readout_event_index,
    }


def calibration_capture_layout_from_tree(tree: Any) -> CalibrationCaptureLayout:
    data = _exact_map(
        tree,
        {"readout_event_axis_id", "reference_event_indices", "readout_event_index"},
        "calibration capture layout",
        discriminator=None,
    )
    return CalibrationCaptureLayout(
        readout_event_axis_id=AxisId(data["readout_event_axis_id"]),
        reference_event_indices=tuple(data["reference_event_indices"]),
        readout_event_index=data["readout_event_index"],
    )


__all__ = [
    "calibration_capture_layout_from_tree",
    "calibration_capture_layout_to_tree",
    "camera_capture_descriptor_from_tree",
    "camera_capture_descriptor_to_tree",
    "frame_contract_from_tree",
    "frame_contract_to_tree",
    "readout_binding_key_from_tree",
    "readout_binding_key_to_tree",
]
