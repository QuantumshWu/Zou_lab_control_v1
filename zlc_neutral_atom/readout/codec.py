"""Strict current-version codecs for neutral-atom readout contracts."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import numpy as np

from zlc_data import AxisId, CoordinateFrameId, value_schema_from_tree, value_schema_to_tree
from zlc_storage import decode, encode

from .contracts import (
    CalibrationCaptureLayout,
    CameraCaptureDescriptor,
    CameraEventReadoutSetting,
    FrameContract,
    ReadoutBindingKey,
)


READOUT_BINDING_KEY_SCHEMA = "zlc_neutral_atom.readout-binding-key.v1"
CAMERA_EVENT_READOUT_SETTING_SCHEMA = "zlc_neutral_atom.camera-event-readout-setting.v1"
CAMERA_CAPTURE_DESCRIPTOR_SCHEMA = "zlc_neutral_atom.camera-capture-descriptor.v1"
FRAME_CONTRACT_SCHEMA = "zlc_neutral_atom.frame-contract.v1"
CALIBRATION_CAPTURE_LAYOUT_SCHEMA = "zlc_neutral_atom.calibration-capture-layout.v1"


class ReadoutCodecError(ValueError):
    """Raised when bytes do not have one current canonical typed meaning."""


T = TypeVar("T")


def _exact_map(tree: Any, fields: set[str], schema_id: str) -> dict[str, Any]:
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError(f"{schema_id} must contain exactly {sorted(fields)}")
    if tree["schema"] != schema_id:
        raise ValueError(f"expected schema {schema_id!r}, got {tree['schema']!r}")
    return tree


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _float(value: Any, field: str) -> float:
    if not isinstance(value, float):
        raise ValueError(f"{field} must use the canonical float representation")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _pair(value: Any, field: str) -> tuple[int, int]:
    items = _list(value, field)
    if len(items) != 2:
        raise ValueError(f"{field} must have exactly two entries")
    return (_integer(items[0], f"{field}[0]"), _integer(items[1], f"{field}[1]"))


def _dtype(value: Any) -> np.dtype:
    try:
        return np.dtype(_text(value, "dtype"))
    except (TypeError, ValueError) as exc:
        raise ValueError("dtype is not a NumPy dtype") from exc


def _canonical_tree(original: Any, projected: Any, schema_id: str) -> None:
    if original != projected:
        raise ReadoutCodecError(f"{schema_id} tree is typed but non-canonical")


def _encode_typed(value: T, projector: Callable[[T], dict[str, Any]]) -> bytes:
    return encode(projector(value))


def _decode_typed(
    payload: bytes | bytearray | memoryview,
    parser: Callable[[Any], T],
    projector: Callable[[T], dict[str, Any]],
    schema_id: str,
) -> T:
    raw = bytes(payload)
    result = parser(decode(raw))
    if _encode_typed(result, projector) != raw:
        raise ReadoutCodecError(
            f"{schema_id} payload uses a non-canonical typed representation"
        )
    return result


def readout_binding_key_to_tree(value: ReadoutBindingKey) -> dict[str, Any]:
    if not isinstance(value, ReadoutBindingKey):
        raise TypeError("value must be ReadoutBindingKey")
    return {"schema": READOUT_BINDING_KEY_SCHEMA, "value": value.value}


def readout_binding_key_from_tree(tree: Any) -> ReadoutBindingKey:
    data = _exact_map(tree, {"schema", "value"}, READOUT_BINDING_KEY_SCHEMA)
    value = ReadoutBindingKey(_text(data["value"], "value"))
    _canonical_tree(tree, readout_binding_key_to_tree(value), READOUT_BINDING_KEY_SCHEMA)
    return value


def encode_readout_binding_key(value: ReadoutBindingKey) -> bytes:
    return _encode_typed(value, readout_binding_key_to_tree)


def decode_readout_binding_key(payload: bytes | bytearray | memoryview) -> ReadoutBindingKey:
    return _decode_typed(
        payload,
        readout_binding_key_from_tree,
        readout_binding_key_to_tree,
        READOUT_BINDING_KEY_SCHEMA,
    )


def camera_event_readout_setting_to_tree(
    value: CameraEventReadoutSetting,
) -> dict[str, Any]:
    if not isinstance(value, CameraEventReadoutSetting):
        raise TypeError("value must be CameraEventReadoutSetting")
    return {
        "schema": CAMERA_EVENT_READOUT_SETTING_SCHEMA,
        "event_index": value.event_index,
        "exposure_seconds": value.exposure_seconds,
        "gain": value.gain,
        "readout_mode": value.readout_mode,
        "opaque_frame_settings_fingerprint": (
            value.opaque_frame_settings_fingerprint
        ),
    }


def camera_event_readout_setting_from_tree(tree: Any) -> CameraEventReadoutSetting:
    data = _exact_map(
        tree,
        {
            "schema",
            "event_index",
            "exposure_seconds",
            "gain",
            "readout_mode",
            "opaque_frame_settings_fingerprint",
        },
        CAMERA_EVENT_READOUT_SETTING_SCHEMA,
    )
    value = CameraEventReadoutSetting(
        _integer(data["event_index"], "event_index"),
        _float(data["exposure_seconds"], "exposure_seconds"),
        _float(data["gain"], "gain"),
        _text(data["readout_mode"], "readout_mode"),
        (
            None
            if data["opaque_frame_settings_fingerprint"] is None
            else _text(
                data["opaque_frame_settings_fingerprint"],
                "opaque_frame_settings_fingerprint",
            )
        ),
    )
    _canonical_tree(
        tree,
        camera_event_readout_setting_to_tree(value),
        CAMERA_EVENT_READOUT_SETTING_SCHEMA,
    )
    return value


def encode_camera_event_readout_setting(value: CameraEventReadoutSetting) -> bytes:
    return _encode_typed(value, camera_event_readout_setting_to_tree)


def decode_camera_event_readout_setting(
    payload: bytes | bytearray | memoryview,
) -> CameraEventReadoutSetting:
    return _decode_typed(
        payload,
        camera_event_readout_setting_from_tree,
        camera_event_readout_setting_to_tree,
        CAMERA_EVENT_READOUT_SETTING_SCHEMA,
    )


_CAPTURE_FIELDS = {
    "schema",
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
    "readout_event_axis_id",
    "event_settings",
    "camera_arm_spec_fingerprint",
}


def camera_capture_descriptor_to_tree(value: CameraCaptureDescriptor) -> dict[str, Any]:
    if not isinstance(value, CameraCaptureDescriptor):
        raise TypeError("value must be CameraCaptureDescriptor")
    return {
        "schema": CAMERA_CAPTURE_DESCRIPTOR_SCHEMA,
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
        "readout_event_axis_id": (
            None if value.readout_event_axis_id is None else value.readout_event_axis_id.value
        ),
        "event_settings": [
            camera_event_readout_setting_to_tree(item) for item in value.event_settings
        ],
        "camera_arm_spec_fingerprint": value.camera_arm_spec_fingerprint,
    }


def camera_capture_descriptor_from_tree(tree: Any) -> CameraCaptureDescriptor:
    data = _exact_map(tree, _CAPTURE_FIELDS, CAMERA_CAPTURE_DESCRIPTOR_SCHEMA)
    event_axis = data["readout_event_axis_id"]
    arm_spec_fingerprint = data["camera_arm_spec_fingerprint"]
    value = CameraCaptureDescriptor(
        camera_identity=_text(data["camera_identity"], "camera_identity"),
        sensor_identity=_text(data["sensor_identity"], "sensor_identity"),
        optical_path=_text(data["optical_path"], "optical_path"),
        sensor_shape_yx=_pair(data["sensor_shape_yx"], "sensor_shape_yx"),
        roi_origin_yx=_pair(data["roi_origin_yx"], "roi_origin_yx"),
        roi_shape_yx=_pair(data["roi_shape_yx"], "roi_shape_yx"),
        binning_yx=_pair(data["binning_yx"], "binning_yx"),
        spatial_y_axis_id=AxisId(_text(data["spatial_y_axis_id"], "spatial_y_axis_id")),
        spatial_x_axis_id=AxisId(_text(data["spatial_x_axis_id"], "spatial_x_axis_id")),
        coordinate_frame=CoordinateFrameId(
            _text(data["coordinate_frame"], "coordinate_frame")
        ),
        dtype=_dtype(data["dtype"]),
        count_unit=_text(data["count_unit"], "count_unit"),
        readout_event_axis_id=(
            None
            if event_axis is None
            else AxisId(_text(event_axis, "readout_event_axis_id"))
        ),
        event_settings=tuple(
            camera_event_readout_setting_from_tree(item)
            for item in _list(data["event_settings"], "event_settings")
        ),
        camera_arm_spec_fingerprint=(
            None
            if arm_spec_fingerprint is None
            else _text(arm_spec_fingerprint, "camera_arm_spec_fingerprint")
        ),
    )
    _canonical_tree(
        tree,
        camera_capture_descriptor_to_tree(value),
        CAMERA_CAPTURE_DESCRIPTOR_SCHEMA,
    )
    return value


def encode_camera_capture_descriptor(value: CameraCaptureDescriptor) -> bytes:
    return _encode_typed(value, camera_capture_descriptor_to_tree)


def decode_camera_capture_descriptor(
    payload: bytes | bytearray | memoryview,
) -> CameraCaptureDescriptor:
    return _decode_typed(
        payload,
        camera_capture_descriptor_from_tree,
        camera_capture_descriptor_to_tree,
        CAMERA_CAPTURE_DESCRIPTOR_SCHEMA,
    )


_FRAME_FIELDS = {
    "schema",
    "binding",
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
        "schema": FRAME_CONTRACT_SCHEMA,
        "binding": readout_binding_key_to_tree(value.binding),
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
        "exposure_seconds": value.exposure_seconds,
        "gain": value.gain,
        "readout_mode": value.readout_mode,
        "opaque_frame_settings_fingerprint": value.opaque_frame_settings_fingerprint,
        "frame_schema": value_schema_to_tree(value.frame_schema),
    }


def frame_contract_from_tree(tree: Any) -> FrameContract:
    data = _exact_map(tree, _FRAME_FIELDS, FRAME_CONTRACT_SCHEMA)
    value = FrameContract(
        binding=readout_binding_key_from_tree(data["binding"]),
        camera_identity=_text(data["camera_identity"], "camera_identity"),
        sensor_identity=_text(data["sensor_identity"], "sensor_identity"),
        optical_path=_text(data["optical_path"], "optical_path"),
        sensor_shape_yx=_pair(data["sensor_shape_yx"], "sensor_shape_yx"),
        roi_origin_yx=_pair(data["roi_origin_yx"], "roi_origin_yx"),
        roi_shape_yx=_pair(data["roi_shape_yx"], "roi_shape_yx"),
        binning_yx=_pair(data["binning_yx"], "binning_yx"),
        spatial_y_axis_id=AxisId(_text(data["spatial_y_axis_id"], "spatial_y_axis_id")),
        spatial_x_axis_id=AxisId(_text(data["spatial_x_axis_id"], "spatial_x_axis_id")),
        coordinate_frame=CoordinateFrameId(
            _text(data["coordinate_frame"], "coordinate_frame")
        ),
        dtype=_dtype(data["dtype"]),
        count_unit=_text(data["count_unit"], "count_unit"),
        exposure_seconds=_float(data["exposure_seconds"], "exposure_seconds"),
        gain=_float(data["gain"], "gain"),
        readout_mode=_text(data["readout_mode"], "readout_mode"),
        opaque_frame_settings_fingerprint=(
            None
            if data["opaque_frame_settings_fingerprint"] is None
            else _text(
                data["opaque_frame_settings_fingerprint"],
                "opaque_frame_settings_fingerprint",
            )
        ),
        frame_schema=value_schema_from_tree(data["frame_schema"]),
    )
    _canonical_tree(tree, frame_contract_to_tree(value), FRAME_CONTRACT_SCHEMA)
    return value


def encode_frame_contract(value: FrameContract) -> bytes:
    return _encode_typed(value, frame_contract_to_tree)


def decode_frame_contract(payload: bytes | bytearray | memoryview) -> FrameContract:
    return _decode_typed(
        payload,
        frame_contract_from_tree,
        frame_contract_to_tree,
        FRAME_CONTRACT_SCHEMA,
    )


def calibration_capture_layout_to_tree(
    value: CalibrationCaptureLayout,
) -> dict[str, Any]:
    if not isinstance(value, CalibrationCaptureLayout):
        raise TypeError("value must be CalibrationCaptureLayout")
    return {
        "schema": CALIBRATION_CAPTURE_LAYOUT_SCHEMA,
        "readout_event_axis_id": value.readout_event_axis_id.value,
        "reference_event_indices": list(value.reference_event_indices),
        "readout_event_index": value.readout_event_index,
    }


def calibration_capture_layout_from_tree(tree: Any) -> CalibrationCaptureLayout:
    data = _exact_map(
        tree,
        {"schema", "readout_event_axis_id", "reference_event_indices", "readout_event_index"},
        CALIBRATION_CAPTURE_LAYOUT_SCHEMA,
    )
    value = CalibrationCaptureLayout(
        readout_event_axis_id=AxisId(
            _text(data["readout_event_axis_id"], "readout_event_axis_id")
        ),
        reference_event_indices=tuple(
            _integer(index, "reference_event_indices entry")
            for index in _list(data["reference_event_indices"], "reference_event_indices")
        ),
        readout_event_index=_integer(data["readout_event_index"], "readout_event_index"),
    )
    _canonical_tree(
        tree,
        calibration_capture_layout_to_tree(value),
        CALIBRATION_CAPTURE_LAYOUT_SCHEMA,
    )
    return value


def encode_calibration_capture_layout(value: CalibrationCaptureLayout) -> bytes:
    return _encode_typed(value, calibration_capture_layout_to_tree)


def decode_calibration_capture_layout(
    payload: bytes | bytearray | memoryview,
) -> CalibrationCaptureLayout:
    return _decode_typed(
        payload,
        calibration_capture_layout_from_tree,
        calibration_capture_layout_to_tree,
        CALIBRATION_CAPTURE_LAYOUT_SCHEMA,
    )


__all__ = [
    "CALIBRATION_CAPTURE_LAYOUT_SCHEMA",
    "CAMERA_CAPTURE_DESCRIPTOR_SCHEMA",
    "CAMERA_EVENT_READOUT_SETTING_SCHEMA",
    "FRAME_CONTRACT_SCHEMA",
    "READOUT_BINDING_KEY_SCHEMA",
    "ReadoutCodecError",
    "calibration_capture_layout_from_tree",
    "calibration_capture_layout_to_tree",
    "camera_capture_descriptor_from_tree",
    "camera_capture_descriptor_to_tree",
    "camera_event_readout_setting_from_tree",
    "camera_event_readout_setting_to_tree",
    "decode_calibration_capture_layout",
    "decode_camera_capture_descriptor",
    "decode_camera_event_readout_setting",
    "decode_frame_contract",
    "decode_readout_binding_key",
    "encode_calibration_capture_layout",
    "encode_camera_capture_descriptor",
    "encode_camera_event_readout_setting",
    "encode_frame_contract",
    "encode_readout_binding_key",
    "frame_contract_from_tree",
    "frame_contract_to_tree",
    "readout_binding_key_from_tree",
    "readout_binding_key_to_tree",
]
