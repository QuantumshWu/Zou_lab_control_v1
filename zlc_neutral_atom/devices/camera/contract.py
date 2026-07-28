"""Canonical physical camera contracts, values, and codecs.

This lower owner contains only hardware-facing camera facts, immutable frame
payloads, and their canonical serialization. Runtime orchestration and logic
nodes depend on these contracts; this module never imports either layer.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from zlc_data import (
    READOUT_EVENT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    ComponentValidity,
    CoordinateFrameId,
    DatasetSchema,
    Invalid,
    PointColumn,
    Valid,
    Value,
    ValuePayloadContract,
    ValueSchema,
    immutable_array,
)
from zlc_storage import (
    canonical_digest,
    canonical_text,
    canonical_text as _canonical_text,
    decode,
    encode,
    exact_mapping as _exact_map,
    exact_mapping as _exact_tree,
    integer,
    integer as _integer,
    nonnegative_integer as _nonnegative_int,
    nonnegative_integer as _nonnegative_integer,
    nonnegative_real as _nonnegative_real,
    positive_integer as _positive_integer,
    positive_real as _positive_finite,
    positive_real as _positive_real,
    sha256_text,
    sha256_text as _sha256,
)


class CameraFrameFactsLike(Protocol):
    camera_identity: str
    sensor_identity: str
    optical_path: str
    sensor_shape_yx: tuple[int, int]
    roi_origin_yx: tuple[int, int]
    roi_shape_yx: tuple[int, int]
    binning_yx: tuple[int, int]
    spatial_y_axis_id: AxisId
    spatial_x_axis_id: AxisId
    coordinate_frame: CoordinateFrameId
    dtype: np.dtype
    count_unit: str


CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT = canonical_digest(
    {
        "owner": (
            "zlc_neutral_atom.devices.camera.contract."
            "CameraDatasetEventAdapter"
        ),
        "operator": "camera-sample.image-identity",
    }
)


def _pair(value: object, field: str, *, positive: bool) -> tuple[int, int]:
    try:
        pair = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field} must be a two-integer tuple") from exc
    if len(pair) != 2:
        raise ValueError(f"{field} must have exactly two entries in Y,X order")
    validator = _positive_integer if positive else _nonnegative_integer
    return tuple(
        validator(item, f"{field}[{index}]")
        for index, item in enumerate(pair)
    )  # type: ignore[return-value]


def normalize_camera_count_dtype(value: object, field: str) -> np.dtype:
    """Return the canonical little-endian dtype admitted for camera counts."""

    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a NumPy dtype") from exc
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real integer or floating dtype")
    if dtype.kind in "iu" and dtype.itemsize not in (1, 2, 4, 8):
        raise TypeError(f"{field} has an unsupported integer width")
    if dtype.kind == "f" and dtype.itemsize not in (2, 4, 8):
        raise TypeError(f"{field} has an unsupported floating width")
    return dtype.newbyteorder("<")


def normalize_camera_geometry(
    *,
    sensor_shape_yx: object,
    roi_origin_yx: object,
    roi_shape_yx: object,
    binning_yx: object,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Validate and normalize one unbinned sensor/ROI/binning geometry."""

    sensor = _pair(sensor_shape_yx, "sensor_shape_yx", positive=True)
    origin = _pair(roi_origin_yx, "roi_origin_yx", positive=False)
    roi = _pair(roi_shape_yx, "roi_shape_yx", positive=True)
    binning = _pair(binning_yx, "binning_yx", positive=True)
    if any(start + extent > limit for start, extent, limit in zip(origin, roi, sensor)):
        raise ValueError("camera ROI lies outside the declared sensor geometry")
    if any(extent % factor for extent, factor in zip(roi, binning)):
        raise ValueError("roi_shape_yx must be exactly divisible by binning_yx")
    return sensor, origin, roi, binning

def validate_camera_spatial_axes(
    spatial_y_axis_id: object,
    spatial_x_axis_id: object,
    coordinate_frame: object,
) -> None:
    """Validate the named spatial-axis facts shared by camera frame values."""

    if not isinstance(spatial_y_axis_id, AxisId):
        raise TypeError("spatial_y_axis_id must be AxisId")
    if not isinstance(spatial_x_axis_id, AxisId):
        raise TypeError("spatial_x_axis_id must be AxisId")
    if spatial_y_axis_id == spatial_x_axis_id:
        raise ValueError("spatial Y and X axes must have different identities")
    if not isinstance(coordinate_frame, CoordinateFrameId):
        raise TypeError("coordinate_frame must be CoordinateFrameId")


def camera_output_shape_yx(
    roi_shape_yx: tuple[int, int],
    binning_yx: tuple[int, int],
) -> tuple[int, int]:
    """Derive output-pixel geometry from a previously validated ROI/binning."""

    return tuple(
        extent // factor for extent, factor in zip(roi_shape_yx, binning_yx)
    )  # type: ignore[return-value]


def validate_camera_frame_schema_facts(
    *,
    spatial_y_axis_id: AxisId,
    spatial_x_axis_id: AxisId,
    coordinate_frame: CoordinateFrameId,
    output_shape_yx: tuple[int, int],
    dtype: np.dtype,
    count_unit: str,
    frame_schema: ValueSchema,
) -> None:
    if not isinstance(frame_schema, ValueSchema):
        raise TypeError("frame_schema must be ValueSchema")
    axes = frame_schema.data_axes
    if tuple(axis.axis_id for axis in axes) != (
        spatial_y_axis_id,
        spatial_x_axis_id,
    ):
        raise ValueError("frame data axes must be exactly descriptor Y,X axes in Y,X order")
    if tuple(axis.role for axis in axes) != (SPATIAL_Y, SPATIAL_X):
        raise ValueError("frame data-axis roles must be spatial-y, spatial-x")
    if tuple(axis.size for axis in axes) != output_shape_yx:
        raise ValueError("frame data-axis sizes differ from ROI/binning output geometry")
    if any(axis.coordinate_frame != coordinate_frame for axis in axes):
        raise ValueError("frame spatial axes differ from the descriptor coordinate frame")
    if any(axis.unit != "pixel" for axis in axes):
        raise ValueError("frame spatial axes must use the canonical 'pixel' unit")
    if any(
        axis.coordinates is None
        or len(axis.coordinates) != axis.size
        or any(
            coordinate != index
            for index, coordinate in enumerate(axis.coordinates)
        )
        for axis in axes
    ):
        raise ValueError(
            "frame spatial axes must use ROI-local output-pixel coordinates 0..N-1"
        )
    if frame_schema.dtype != dtype:
        raise ValueError("frame dtype differs from the camera descriptor")
    if frame_schema.value_unit != count_unit:
        raise ValueError("frame count unit differs from the camera descriptor")


@dataclass(frozen=True, order=True)
class ReadoutBindingKey:
    """Stable logical composition key for one readout path."""

    value: str

    def __post_init__(self) -> None:
        _canonical_text(self.value, "ReadoutBindingKey")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CameraEventReadoutSetting:
    """Typed camera settings for one capture-schedule event index."""

    event_index: int
    exposure_seconds: float
    gain: float
    readout_mode: str
    opaque_frame_settings_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_index",
            _nonnegative_integer(self.event_index, "event_index"),
        )
        object.__setattr__(
            self,
            "exposure_seconds",
            _positive_real(self.exposure_seconds, "exposure_seconds"),
        )
        object.__setattr__(self, "gain", _nonnegative_real(self.gain, "gain"))
        _canonical_text(self.readout_mode, "readout_mode")
        _sha256(
            self.opaque_frame_settings_fingerprint,
            "opaque_frame_settings_fingerprint",
            optional=True,
        )


@dataclass(frozen=True)
class CameraCaptureDescriptor:
    """Complete camera geometry and per-event schedule retained in capture lineage.

    ROI origin/shape and sensor shape use unbinned sensor pixels in Y,X order.
    ``camera_arm_spec_fingerprint`` is the exact frozen camera arm/capture-spec
    digest.  It is not an FPGA pulse-schedule digest.  Per-event untyped evidence
    lives on :class:`CameraEventReadoutSetting` instead.
    """

    camera_identity: str
    sensor_identity: str
    optical_path: str
    sensor_shape_yx: tuple[int, int]
    roi_origin_yx: tuple[int, int]
    roi_shape_yx: tuple[int, int]
    binning_yx: tuple[int, int]
    spatial_y_axis_id: AxisId
    spatial_x_axis_id: AxisId
    coordinate_frame: CoordinateFrameId
    dtype: np.dtype
    count_unit: str
    readout_event_axis_id: AxisId | None
    event_settings: tuple[CameraEventReadoutSetting, ...]
    camera_arm_spec_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("camera_identity", "sensor_identity", "optical_path"):
            _canonical_text(getattr(self, name), name)
        sensor, origin, roi, binning = normalize_camera_geometry(
            sensor_shape_yx=self.sensor_shape_yx,
            roi_origin_yx=self.roi_origin_yx,
            roi_shape_yx=self.roi_shape_yx,
            binning_yx=self.binning_yx,
        )
        object.__setattr__(self, "sensor_shape_yx", sensor)
        object.__setattr__(self, "roi_origin_yx", origin)
        object.__setattr__(self, "roi_shape_yx", roi)
        object.__setattr__(self, "binning_yx", binning)
        validate_camera_spatial_axes(
            self.spatial_y_axis_id,
            self.spatial_x_axis_id,
            self.coordinate_frame,
        )
        if self.readout_event_axis_id is not None and not isinstance(
            self.readout_event_axis_id,
            AxisId,
        ):
            raise TypeError("readout_event_axis_id must be AxisId or None")
        if self.readout_event_axis_id in {self.spatial_y_axis_id, self.spatial_x_axis_id}:
            raise ValueError("readout-event and spatial axes must have different identities")
        object.__setattr__(
            self,
            "dtype",
            normalize_camera_count_dtype(self.dtype, "dtype"),
        )
        _canonical_text(self.count_unit, "count_unit")
        settings = tuple(self.event_settings)
        if not settings:
            raise ValueError("event_settings cannot be empty")
        if any(not isinstance(item, CameraEventReadoutSetting) for item in settings):
            raise TypeError("event_settings must contain CameraEventReadoutSetting values")
        settings = tuple(sorted(settings, key=lambda item: item.event_index))
        if len({item.event_index for item in settings}) != len(settings):
            raise ValueError("event_settings event_index values must be unique")
        if self.readout_event_axis_id is None and tuple(
            item.event_index for item in settings
        ) != (0,):
            raise ValueError("a capture without a readout-event axis requires event index 0")
        object.__setattr__(self, "event_settings", settings)
        _sha256(
            self.camera_arm_spec_fingerprint,
            "camera_arm_spec_fingerprint",
            optional=True,
        )

    @property
    def output_shape_yx(self) -> tuple[int, int]:
        return camera_output_shape_yx(self.roi_shape_yx, self.binning_yx)

    def setting(self, event_index: int) -> CameraEventReadoutSetting:
        index = _nonnegative_integer(event_index, "event_index")
        for setting in self.event_settings:
            if setting.event_index == index:
                return setting
        raise ValueError(f"capture descriptor has no setting for event index {index}")

    def _event_column(self, schema: DatasetSchema) -> tuple[int, PointColumn] | None:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        readout_columns = tuple(
            (position, column)
            for position, column in enumerate(schema.point_table.columns)
            if column.role == READOUT_EVENT
        )
        if len(readout_columns) > 1:
            raise ValueError("DatasetSchema must not contain multiple READOUT_EVENT columns")
        if self.readout_event_axis_id is None:
            if readout_columns:
                raise ValueError("descriptor and schema disagree about a READOUT_EVENT column")
            return None
        matching = tuple(
            (position, column)
            for position, column in enumerate(schema.point_table.columns)
            if column.coordinate_id == self.readout_event_axis_id
        )
        if len(matching) != 1 or matching[0][1].role != READOUT_EVENT:
            raise ValueError("descriptor READOUT_EVENT id is absent or has the wrong role")
        if readout_columns != matching:
            raise ValueError("schema contains a different READOUT_EVENT column")
        return matching[0]

    def validate_schema(self, schema: DatasetSchema) -> None:
        event_column = self._event_column(schema)
        validate_camera_frame_schema_facts(
            spatial_y_axis_id=self.spatial_y_axis_id,
            spatial_x_axis_id=self.spatial_x_axis_id,
            coordinate_frame=self.coordinate_frame,
            output_shape_yx=self.output_shape_yx,
            dtype=self.dtype,
            count_unit=self.count_unit,
            frame_schema=schema.cell_schema,
        )
        if event_column is None:
            expected_indices = (0,)
        else:
            declared = tuple(dict.fromkeys(event_column[1].values))
            expected_indices = tuple(range(len(declared)))
            if declared != expected_indices:
                raise ValueError(
                    "READOUT_EVENT values must be canonical zero-based indices"
                )
        if tuple(item.event_index for item in self.event_settings) != expected_indices:
            raise ValueError("event_settings must cover every READOUT_EVENT index exactly once")

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


def camera_event_readout_setting_to_tree(
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


def camera_event_readout_setting_from_tree(tree: Any) -> CameraEventReadoutSetting:
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


CAMERA_FRAME_FACT_FIELDS = frozenset({
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
})

_CAPTURE_FIELDS = CAMERA_FRAME_FACT_FIELDS | {
    "readout_event_axis_id",
    "event_settings",
    "camera_arm_spec_fingerprint",
}


def camera_frame_facts_to_tree(
    value: CameraFrameFactsLike,
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


def camera_frame_facts_from_tree(data: dict[str, Any]) -> dict[str, Any]:
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
        **camera_frame_facts_to_tree(value),
        "readout_event_axis_id": (
            None if value.readout_event_axis_id is None else value.readout_event_axis_id.value
        ),
        "event_settings": [
            camera_event_readout_setting_to_tree(item) for item in value.event_settings
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
        **camera_frame_facts_from_tree(data),
        readout_event_axis_id=(
            None
            if event_axis is None
            else AxisId(event_axis)
        ),
        event_settings=tuple(
            camera_event_readout_setting_from_tree(item)
            for item in data["event_settings"]
        ),
        camera_arm_spec_fingerprint=data["camera_arm_spec_fingerprint"],
    )
    return value

@dataclass(frozen=True)
class FrozenCaptureSpec:
    """Canonical owner bytes; runtime never executes an arbitrary spec codec."""

    owner_fingerprint: str
    payload: bytes
    digest: str = ""

    def __post_init__(self) -> None:
        _sha256(self.owner_fingerprint, "capture spec owner fingerprint")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("capture spec payload must be non-empty immutable bytes")
        payload = bytes(self.payload)
        digest = hashlib.sha256(payload).hexdigest()
        if self.digest and self.digest != digest:
            raise ValueError("capture spec digest differs from canonical payload")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "digest", digest)


def frozen_capture_spec_to_tree(value: FrozenCaptureSpec) -> dict[str, object]:
    if not isinstance(value, FrozenCaptureSpec):
        raise TypeError("value must be FrozenCaptureSpec")
    return {
        "owner_fingerprint": value.owner_fingerprint,
        "payload": value.payload,
    }


def frozen_capture_spec_from_tree(tree: object) -> FrozenCaptureSpec:
    data = _exact_tree(
        tree,
        {"owner_fingerprint", "payload"},
        "frozen capture spec",
        discriminator=None,
    )
    return FrozenCaptureSpec(data["owner_fingerprint"], data["payload"])

_CAMERA_CAPTURE_SPEC_SCHEMA = "zlc_neutral_atom.camera-capture-spec"
CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT = canonical_digest(
    {
        "owner": "zlc_neutral_atom.devices.camera.contract",
        "schema": _CAMERA_CAPTURE_SPEC_SCHEMA,
    }
)


class CameraAcquisitionMode(str, Enum):
    EXTERNAL_TRIGGERED = "EXTERNAL_TRIGGERED"
    FREE_RUNNING = "FREE_RUNNING"


@dataclass(frozen=True)
class CameraCaptureSpec:
    """Request-owned settings that are frozen before runtime preflight."""

    mode: CameraAcquisitionMode
    expected_frames: int
    source_group_sizes: tuple[int, ...]
    settings_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CameraAcquisitionMode):
            raise TypeError("mode must be CameraAcquisitionMode")
        frames = _integer(
            self.expected_frames,
            "expected_frames",
            nonnegative=True,
        )
        assert frames is not None
        if frames == 0:
            raise ValueError("expected_frames must be positive")
        object.__setattr__(self, "expected_frames", frames)
        groups = tuple(
            _integer(size, "source_group_sizes item", nonnegative=True)
            for size in self.source_group_sizes
        )
        if not groups or any(size == 0 for size in groups):
            raise ValueError("source_group_sizes must contain positive integers")
        if sum(groups) != frames:
            raise ValueError("source_group_sizes must exactly cover expected_frames")
        object.__setattr__(self, "source_group_sizes", groups)
        _sha256(self.settings_fingerprint, "settings_fingerprint")


def camera_capture_spec_to_bytes(spec: CameraCaptureSpec) -> bytes:
    if not isinstance(spec, CameraCaptureSpec):
        raise TypeError("spec must be CameraCaptureSpec")
    return encode(
        {
            "schema": _CAMERA_CAPTURE_SPEC_SCHEMA,
            "mode": spec.mode.value,
            "expected_frames": spec.expected_frames,
            "source_group_sizes": list(spec.source_group_sizes),
            "settings_fingerprint": spec.settings_fingerprint,
        }
    )


def freeze_camera_capture_spec(spec: CameraCaptureSpec) -> FrozenCaptureSpec:
    return FrozenCaptureSpec(
        CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
        camera_capture_spec_to_bytes(spec),
    )


def camera_capture_spec_from_bytes(payload: bytes) -> CameraCaptureSpec:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be canonical bytes")
    decoded = decode(payload)
    if not isinstance(decoded, dict) or set(decoded) != {
        "schema",
        "mode",
        "expected_frames",
        "source_group_sizes",
        "settings_fingerprint",
    }:
        raise ValueError("camera capture spec has an unknown field set")
    if decoded["schema"] != _CAMERA_CAPTURE_SPEC_SCHEMA:
        raise ValueError("camera capture spec schema differs")
    try:
        mode = CameraAcquisitionMode(decoded["mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("camera capture mode is unknown") from exc
    return CameraCaptureSpec(
        mode,
        decoded["expected_frames"],
        tuple(decoded["source_group_sizes"]),
        decoded["settings_fingerprint"],
    )


def decode_camera_capture_spec(value: FrozenCaptureSpec | bytes) -> CameraCaptureSpec:
    """Decode exactly the current camera capture-spec schema."""

    if isinstance(value, FrozenCaptureSpec):
        if value.owner_fingerprint != CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT:
            raise ValueError("camera capture spec owner fingerprint differs")
        payload = value.payload
    elif isinstance(value, bytes):
        payload = value
    else:
        raise TypeError("value must be FrozenCaptureSpec or canonical bytes")
    return camera_capture_spec_from_bytes(payload)

@dataclass(frozen=True)
class CameraFrameMetadata:
    """One frame's frozen physical observations plus run correlation identity."""

    source_ordinal: int
    produced_count: int | None
    frame_stamp: int | None
    camera_stamp: int | None
    timestamp_seconds: int | None
    timestamp_microseconds: int | None
    host_received_at_ns: int
    driver_buffer_index: int | None
    correlation_id: str

    def __post_init__(self) -> None:
        for field in ("source_ordinal", "host_received_at_ns"):
            value = _integer(getattr(self, field), field, nonnegative=True)
            assert value is not None
            if field == "host_received_at_ns" and value == 0:
                raise ValueError("host_received_at_ns must be positive")
            object.__setattr__(self, field, value)
        for field in (
            "produced_count",
            "timestamp_seconds",
            "timestamp_microseconds",
            "driver_buffer_index",
        ):
            object.__setattr__(
                self,
                field,
                _integer(
                    getattr(self, field),
                    field,
                    optional=True,
                    nonnegative=True,
                ),
            )
        for field in ("frame_stamp", "camera_stamp"):
            object.__setattr__(
                self,
                field,
                _integer(getattr(self, field), field, optional=True),
            )
        if (
            self.timestamp_microseconds is not None
            and self.timestamp_microseconds >= 1_000_000
        ):
            raise ValueError("timestamp_microseconds must be less than 1_000_000")
        if (self.timestamp_seconds is None) != (self.timestamp_microseconds is None):
            raise ValueError("camera timestamp seconds and microseconds must appear together")
        _canonical_text(self.correlation_id, "correlation_id")

    @property
    def captured_at(self) -> float:
        if self.timestamp_seconds is not None:
            assert self.timestamp_microseconds is not None
            return float(self.timestamp_seconds) + self.timestamp_microseconds * 1e-6
        return self.host_received_at_ns * 1e-9


def camera_frame_metadata_to_tree(metadata: CameraFrameMetadata) -> dict[str, object]:
    """Current canonical primitive form owned by the camera domain."""

    if not isinstance(metadata, CameraFrameMetadata):
        raise TypeError("metadata must be CameraFrameMetadata")
    return {
        "source_ordinal": metadata.source_ordinal,
        "produced_count": metadata.produced_count,
        "frame_stamp": metadata.frame_stamp,
        "camera_stamp": metadata.camera_stamp,
        "timestamp_seconds": metadata.timestamp_seconds,
        "timestamp_microseconds": metadata.timestamp_microseconds,
        "host_received_at_ns": metadata.host_received_at_ns,
        "driver_buffer_index": metadata.driver_buffer_index,
        "correlation_id": metadata.correlation_id,
    }


def camera_frame_metadata_from_tree(tree: object) -> CameraFrameMetadata:
    fields = {
        "source_ordinal",
        "produced_count",
        "frame_stamp",
        "camera_stamp",
        "timestamp_seconds",
        "timestamp_microseconds",
        "host_received_at_ns",
        "driver_buffer_index",
        "correlation_id",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("camera frame metadata has an unknown field set")
    return CameraFrameMetadata(**tree)


@dataclass(frozen=True)
class CameraSample:
    image: Value
    metadata: CameraFrameMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.image, Value):
            raise TypeError("image must be a zlc_data.Value")
        if not isinstance(self.metadata, CameraFrameMetadata):
            raise TypeError("metadata must be CameraFrameMetadata")


@dataclass(frozen=True)
class CameraFrameMetadataContract:
    @property
    def fingerprint(self) -> str:
        return canonical_digest({"contract": "zlc.camera-frame-metadata"})

    def snapshot(self, payload: CameraSample) -> CameraFrameMetadata:
        if not isinstance(payload, CameraSample):
            raise TypeError("metadata snapshot requires CameraSample")
        self.validate(payload.metadata)
        return payload.metadata

    def validate(self, metadata: object) -> None:
        if not isinstance(metadata, CameraFrameMetadata):
            raise TypeError("metadata must be CameraFrameMetadata")
        if not math.isfinite(metadata.captured_at):
            raise ValueError("captured_at must be finite")

    def digest(self, metadata: object) -> str:
        self.validate(metadata)
        assert isinstance(metadata, CameraFrameMetadata)
        return canonical_digest(camera_frame_metadata_to_tree(metadata))


@dataclass(frozen=True)
class CameraSampleContract:
    value_schema: ValueSchema
    metadata_contract: CameraFrameMetadataContract = CameraFrameMetadataContract()

    def __post_init__(self) -> None:
        if not isinstance(self.value_schema, ValueSchema):
            raise TypeError("value_schema must be ValueSchema")
        if not isinstance(self.metadata_contract, CameraFrameMetadataContract):
            raise TypeError("metadata_contract must be CameraFrameMetadataContract")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "contract": "zlc.camera-sample",
                "value_schema_fingerprint": self.value_schema.fingerprint,
                "metadata_contract_fingerprint": self.metadata_contract.fingerprint,
            }
        )

    def snapshot(self, payload: CameraSample) -> CameraSample:
        self.validate(payload)
        return payload

    def validate(self, payload: CameraSample) -> None:
        if not isinstance(payload, CameraSample):
            raise TypeError("payload must be CameraSample")
        ValuePayloadContract(self.value_schema).validate(payload.image)
        self.metadata_contract.validate(payload.metadata)

    def digest(self, payload: CameraSample) -> str:
        """Bind one physical frame's pixels and acquisition metadata together."""

        self.validate(payload)
        return self.digest_components(
            payload.image.values,
            payload.image.validity,
            payload.metadata,
        )

    def digest_components(
        self,
        image_values: np.ndarray,
        image_validity: Valid | Invalid | ComponentValidity,
        metadata: CameraFrameMetadata,
    ) -> str:
        """Digest a durable frame cell through the transient payload owner."""

        self.metadata_contract.validate(metadata)
        return canonical_digest(
            {
                "schema": "zlc.camera-sample-content",
                "image": ValuePayloadContract(self.value_schema).digest_content(
                    image_values,
                    image_validity,
                ),
                "metadata": self.metadata_contract.digest(metadata),
            }
        )

    @staticmethod
    def source_ordinal(payload: CameraSample) -> int:
        return payload.metadata.source_ordinal

    @staticmethod
    def captured_at(payload: CameraSample) -> float:
        return payload.metadata.captured_at

    @staticmethod
    def correlation_id(payload: CameraSample) -> str:
        return payload.metadata.correlation_id


@dataclass(frozen=True)
class CameraDatasetEventAdapter:
    payload_contract: CameraSampleContract
    operator_fingerprint: str = CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT

    def __post_init__(self) -> None:
        if not isinstance(self.payload_contract, CameraSampleContract):
            raise TypeError("payload_contract must be CameraSampleContract")
        if self.operator_fingerprint != CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT:
            raise ValueError(
                "CameraDatasetEventAdapter operator identity cannot be overridden"
            )

    @property
    def value_schema(self) -> ValueSchema:
        return self.payload_contract.value_schema

    @property
    def metadata_contract(self) -> CameraFrameMetadataContract:
        return self.payload_contract.metadata_contract

    def value(self, payload: CameraSample) -> Value:
        return payload.image

@dataclass(frozen=True)
class CameraPhysicalFacts:
    """Typed camera facts minted inside a broker capability probe.

    These facts and the opaque settings fingerprint are frozen from one adapter
    snapshot.  A later binding may name event indices, but it cannot change the
    physical trigger wiring, exposure, gain, readout mode, geometry, identity,
    dtype, or unit attested by this capability.

    ``required_external_trigger_interval_seconds is None`` means no sufficient
    safe interval has been qualified for exact capture and preflight must
    reject.  Zero is reserved for a source (such as the deterministic virtual
    camera) that is explicitly qualified with no positive lower bound.  A
    ``None`` integration start offset is likewise an explicit unqualified fact;
    consumers must not reinterpret it as zero.
    """

    camera_identity: str
    sensor_identity: str
    optical_path: str
    capture_trigger_channels: tuple[str, ...]
    sensor_shape_yx: tuple[int, int]
    roi_origin_yx: tuple[int, int]
    roi_shape_yx: tuple[int, int]
    binning_yx: tuple[int, int]
    spatial_y_axis_id: AxisId
    spatial_x_axis_id: AxisId
    coordinate_frame: CoordinateFrameId
    dtype: np.dtype
    count_unit: str
    exposure_seconds: float
    required_external_trigger_interval_seconds: float | None
    external_trigger_integration_start_offset_seconds: float | None
    gain: float
    readout_mode: str
    opaque_frame_settings_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("camera_identity", "sensor_identity", "optical_path"):
            _canonical_text(getattr(self, name), name)
        if isinstance(self.capture_trigger_channels, (str, bytes)):
            raise TypeError("capture_trigger_channels must be a tuple of channel names")
        try:
            trigger_channels = tuple(self.capture_trigger_channels)
        except TypeError as exc:
            raise TypeError(
                "capture_trigger_channels must be a tuple of channel names"
            ) from exc
        trigger_channels = tuple(
            _canonical_text(channel, "capture trigger channel")
            for channel in trigger_channels
        )
        if len(trigger_channels) != len(set(trigger_channels)):
            raise ValueError("capture_trigger_channels must be unique")
        object.__setattr__(
            self,
            "capture_trigger_channels",
            trigger_channels,
        )
        sensor, origin, roi, binning = normalize_camera_geometry(
            sensor_shape_yx=self.sensor_shape_yx,
            roi_origin_yx=self.roi_origin_yx,
            roi_shape_yx=self.roi_shape_yx,
            binning_yx=self.binning_yx,
        )
        object.__setattr__(self, "sensor_shape_yx", sensor)
        object.__setattr__(self, "roi_origin_yx", origin)
        object.__setattr__(self, "roi_shape_yx", roi)
        object.__setattr__(self, "binning_yx", binning)
        validate_camera_spatial_axes(
            self.spatial_y_axis_id,
            self.spatial_x_axis_id,
            self.coordinate_frame,
        )
        dtype = normalize_camera_count_dtype(self.dtype, "dtype")
        object.__setattr__(self, "dtype", dtype)
        _canonical_text(self.count_unit, "count_unit")
        object.__setattr__(
            self,
            "exposure_seconds",
            _positive_finite(self.exposure_seconds, "exposure_seconds"),
        )
        interval = self.required_external_trigger_interval_seconds
        if interval is not None:
            if (
                isinstance(interval, bool)
                or not isinstance(interval, (int, float))
                or not math.isfinite(float(interval))
                or float(interval) < 0
            ):
                raise ValueError(
                    "required_external_trigger_interval_seconds must be finite, "
                    "non-negative, or None"
                )
            interval = float(interval)
        object.__setattr__(
            self,
            "required_external_trigger_interval_seconds",
            interval,
        )
        integration_offset = self.external_trigger_integration_start_offset_seconds
        if integration_offset is not None:
            if (
                isinstance(integration_offset, bool)
                or not isinstance(integration_offset, (int, float))
                or not math.isfinite(float(integration_offset))
                or float(integration_offset) < 0
            ):
                raise ValueError(
                    "external_trigger_integration_start_offset_seconds must be "
                    "finite, non-negative, or None"
                )
            integration_offset = float(integration_offset)
        object.__setattr__(
            self,
            "external_trigger_integration_start_offset_seconds",
            integration_offset,
        )
        if (
            isinstance(self.gain, bool)
            or not isinstance(self.gain, (int, float))
            or not math.isfinite(float(self.gain))
            or float(self.gain) < 0
        ):
            raise ValueError("gain must be finite and non-negative")
        object.__setattr__(self, "gain", float(self.gain))
        _canonical_text(self.readout_mode, "readout_mode")
        _sha256(
            self.opaque_frame_settings_fingerprint,
            "opaque_frame_settings_fingerprint",
        )

    @property
    def output_shape_yx(self) -> tuple[int, int]:
        return camera_output_shape_yx(self.roi_shape_yx, self.binning_yx)

    def event_setting(self, event_index: int) -> CameraEventReadoutSetting:
        return CameraEventReadoutSetting(
            _nonnegative_int(event_index, "event_index"),
            self.exposure_seconds,
            self.gain,
            self.readout_mode,
            self.opaque_frame_settings_fingerprint,
        )

    def validate_capture_trigger_channel(self, channel: str) -> None:
        """Require one selected pulse channel to belong to this physical wiring."""

        channel = _canonical_text(channel, "capture trigger channel")
        if channel not in self.capture_trigger_channels:
            raise ValueError(
                f"capture trigger channel {channel!r} is not wired to camera "
                f"{self.camera_identity!r}; attested channels are "
                f"{self.capture_trigger_channels!r}"
            )

    def require_single_capture_trigger_channel(self, channel: str) -> None:
        """Require exact capture to have one, and only one, physical edge source."""

        self.validate_capture_trigger_channel(channel)
        if self.capture_trigger_channels != (channel,):
            raise ValueError(
                "exact triggered capture requires the camera to have exactly one "
                f"attested trigger channel {channel!r}; physical wiring is "
                f"{self.capture_trigger_channels!r}"
            )

    def validate_descriptor(self, descriptor: CameraCaptureDescriptor) -> None:
        if not isinstance(descriptor, CameraCaptureDescriptor):
            raise TypeError("descriptor must be CameraCaptureDescriptor")
        expected = {
            "camera_identity": self.camera_identity,
            "sensor_identity": self.sensor_identity,
            "optical_path": self.optical_path,
            "sensor_shape_yx": self.sensor_shape_yx,
            "roi_origin_yx": self.roi_origin_yx,
            "roi_shape_yx": self.roi_shape_yx,
            "binning_yx": self.binning_yx,
            "spatial_y_axis_id": self.spatial_y_axis_id,
            "spatial_x_axis_id": self.spatial_x_axis_id,
            "coordinate_frame": self.coordinate_frame,
            "dtype": self.dtype,
            "count_unit": self.count_unit,
        }
        mismatches = tuple(
            name for name, value in expected.items() if getattr(descriptor, name) != value
        )
        for setting in descriptor.event_settings:
            if setting != self.event_setting(setting.event_index):
                mismatches += (f"event_settings[{setting.event_index}]",)
        if mismatches:
            raise ValueError(
                "camera descriptor differs from attested physical facts: "
                + ", ".join(mismatches)
            )

    @property
    def fingerprint(self) -> str:
        """Digest of this owner-minted physical-facts value."""

        return canonical_digest(_camera_physical_facts_to_tree(self))


_CAMERA_PHYSICAL_FACTS_SCHEMA = "zlc_neutral_atom.CameraPhysicalFacts"
_CAMERA_CAPABILITY_EVIDENCE_SCHEMA = "zlc_neutral_atom.CameraCapabilityEvidence"


def _camera_physical_facts_to_tree(value: CameraPhysicalFacts) -> dict[str, object]:
    if not isinstance(value, CameraPhysicalFacts):
        raise TypeError("value must be CameraPhysicalFacts")
    return {
        "schema": _CAMERA_PHYSICAL_FACTS_SCHEMA,
        "camera_identity": value.camera_identity,
        "sensor_identity": value.sensor_identity,
        "optical_path": value.optical_path,
        "capture_trigger_channels": list(value.capture_trigger_channels),
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
        "required_external_trigger_interval_seconds": (
            value.required_external_trigger_interval_seconds
        ),
        "external_trigger_integration_start_offset_seconds": (
            value.external_trigger_integration_start_offset_seconds
        ),
        "gain": value.gain,
        "readout_mode": value.readout_mode,
        "opaque_frame_settings_fingerprint": (
            value.opaque_frame_settings_fingerprint
        ),
    }


def _camera_physical_facts_from_tree(tree: object) -> CameraPhysicalFacts:
    fields = {
        "schema",
        "camera_identity",
        "sensor_identity",
        "optical_path",
        "capture_trigger_channels",
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
        "required_external_trigger_interval_seconds",
        "external_trigger_integration_start_offset_seconds",
        "gain",
        "readout_mode",
        "opaque_frame_settings_fingerprint",
    }
    data = _exact_tree(tree, fields, _CAMERA_PHYSICAL_FACTS_SCHEMA)
    return CameraPhysicalFacts(
        camera_identity=data["camera_identity"],
        sensor_identity=data["sensor_identity"],
        optical_path=data["optical_path"],
        capture_trigger_channels=data["capture_trigger_channels"],
        sensor_shape_yx=data["sensor_shape_yx"],
        roi_origin_yx=data["roi_origin_yx"],
        roi_shape_yx=data["roi_shape_yx"],
        binning_yx=data["binning_yx"],
        spatial_y_axis_id=AxisId(data["spatial_y_axis_id"]),
        spatial_x_axis_id=AxisId(data["spatial_x_axis_id"]),
        coordinate_frame=CoordinateFrameId(data["coordinate_frame"]),
        dtype=np.dtype(data["dtype"]),
        count_unit=data["count_unit"],
        exposure_seconds=data["exposure_seconds"],
        required_external_trigger_interval_seconds=data[
            "required_external_trigger_interval_seconds"
        ],
        external_trigger_integration_start_offset_seconds=data[
            "external_trigger_integration_start_offset_seconds"
        ],
        gain=data["gain"],
        readout_mode=data["readout_mode"],
        opaque_frame_settings_fingerprint=data[
            "opaque_frame_settings_fingerprint"
        ],
    )


@dataclass(frozen=True)
class CameraCapabilityEvidence:
    """Canonical facts whose digest is the broker capability fingerprint.

    This value is minted from the same frozen adapter snapshot as
    :class:`CameraPhysicalFacts`.  Persisting it lets the durable boundary
    recompute the terminal capability fingerprint instead of trusting a second
    caller-provided digest.
    """

    adapter_type: str
    source_id: str
    payload_contract_fingerprint: str
    capture_spec_owner_fingerprint: str
    max_blocking_call_seconds: float
    physical_facts: CameraPhysicalFacts
    exact_external_trigger_qualification_digest: str | None = None

    def __post_init__(self) -> None:
        _canonical_text(self.adapter_type, "adapter_type")
        _canonical_text(self.source_id, "source_id")
        for name in (
            "payload_contract_fingerprint",
            "capture_spec_owner_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "max_blocking_call_seconds",
            _positive_finite(
                self.max_blocking_call_seconds,
                "max_blocking_call_seconds",
            ),
        )
        if not isinstance(self.physical_facts, CameraPhysicalFacts):
            raise TypeError("physical_facts must be CameraPhysicalFacts")
        qualification = self.exact_external_trigger_qualification_digest
        if qualification is not None:
            _sha256(
                qualification,
                "exact_external_trigger_qualification_digest",
            )
            if (
                self.physical_facts.required_external_trigger_interval_seconds
                is None
            ):
                raise ValueError(
                    "exact external-trigger qualification requires trigger-interval readback"
                )

    @property
    def settings_fingerprint(self) -> str:
        """The settings digest owned by the frozen physical-facts snapshot."""

        return self.physical_facts.opaque_frame_settings_fingerprint

    @property
    def fingerprint(self) -> str:
        return canonical_digest(camera_capability_evidence_to_tree(self))


def camera_capability_evidence_to_tree(
    value: CameraCapabilityEvidence,
) -> dict[str, object]:
    if not isinstance(value, CameraCapabilityEvidence):
        raise TypeError("value must be CameraCapabilityEvidence")
    return {
        "schema": _CAMERA_CAPABILITY_EVIDENCE_SCHEMA,
        "adapter_type": value.adapter_type,
        "source_id": value.source_id,
        "payload_contract_fingerprint": value.payload_contract_fingerprint,
        "capture_spec_owner_fingerprint": value.capture_spec_owner_fingerprint,
        "max_blocking_call_seconds": value.max_blocking_call_seconds,
        "physical_facts": _camera_physical_facts_to_tree(value.physical_facts),
        "exact_external_trigger_qualification_digest": (
            value.exact_external_trigger_qualification_digest
        ),
    }


def camera_capability_evidence_from_tree(tree: object) -> CameraCapabilityEvidence:
    fields = {
        "schema",
        "adapter_type",
        "source_id",
        "payload_contract_fingerprint",
        "capture_spec_owner_fingerprint",
        "max_blocking_call_seconds",
        "physical_facts",
        "exact_external_trigger_qualification_digest",
    }
    data = _exact_tree(tree, fields, _CAMERA_CAPABILITY_EVIDENCE_SCHEMA)
    facts = _camera_physical_facts_from_tree(data["physical_facts"])
    return CameraCapabilityEvidence(
        adapter_type=data["adapter_type"],
        source_id=data["source_id"],
        payload_contract_fingerprint=data["payload_contract_fingerprint"],
        capture_spec_owner_fingerprint=data["capture_spec_owner_fingerprint"],
        max_blocking_call_seconds=data["max_blocking_call_seconds"],
        physical_facts=facts,
        exact_external_trigger_qualification_digest=data[
            "exact_external_trigger_qualification_digest"
        ],
    )

def camera_roi_local_spatial_identity(
    source_id: object,
) -> tuple[AxisId, AxisId, CoordinateFrameId]:
    """Derive the canonical ROI-local output-pixel identity for one camera."""

    source = canonical_text(source_id, "camera source_id")
    return (
        AxisId(f"{source}.y"),
        AxisId(f"{source}.x"),
        CoordinateFrameId(f"{source}.roi-local-output-pixels"),
    )


@dataclass(frozen=True, eq=False)
class CameraFrameRecord:
    """One adapter-owned frame copied out of a reusable driver buffer."""

    image: np.ndarray
    source_ordinal: int
    produced_count: int | None
    frame_stamp: int | None
    camera_stamp: int | None
    timestamp_seconds: int | None
    timestamp_microseconds: int | None
    host_received_at_ns: int
    driver_buffer_index: int | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        ordinal = integer(self.source_ordinal, "source_ordinal", nonnegative=True)
        assert ordinal is not None
        object.__setattr__(self, "source_ordinal", ordinal)
        for name in (
            "produced_count",
            "timestamp_seconds",
            "timestamp_microseconds",
            "driver_buffer_index",
        ):
            object.__setattr__(
                self,
                name,
                integer(getattr(self, name), name, optional=True, nonnegative=True),
            )
        for name in ("frame_stamp", "camera_stamp"):
            object.__setattr__(
                self,
                name,
                integer(getattr(self, name), name, optional=True),
            )
        host_received_at_ns = integer(
            self.host_received_at_ns,
            "host_received_at_ns",
            nonnegative=True,
        )
        assert host_received_at_ns is not None
        if host_received_at_ns == 0:
            raise ValueError("host_received_at_ns must be positive")
        object.__setattr__(self, "host_received_at_ns", host_received_at_ns)
        if (
            self.timestamp_microseconds is not None
            and self.timestamp_microseconds >= 1_000_000
        ):
            raise ValueError("timestamp_microseconds must be less than 1_000_000")
        if (self.timestamp_seconds is None) != (self.timestamp_microseconds is None):
            raise ValueError("camera timestamp seconds and microseconds must appear together")
        source = np.asarray(self.image)
        image = immutable_array(
            source,
            dtype=source.dtype.newbyteorder("<"),
            shape=source.shape,
        )
        object.__setattr__(self, "image", image)


@dataclass(frozen=True)
class CameraCaptureTerminalRecord:
    """Adapter readback proving a finite source has stopped and drained."""

    produced_count: int
    source_stopped: bool
    no_more_frames: bool
    joined: bool

    def __post_init__(self) -> None:
        produced_count = integer(
            self.produced_count,
            "produced_count",
            nonnegative=True,
        )
        assert produced_count is not None
        object.__setattr__(self, "produced_count", produced_count)
        if any(
            type(getattr(self, name)) is not bool
            for name in ("source_stopped", "no_more_frames", "joined")
        ):
            raise TypeError("terminal proof flags must be bool")


@dataclass(frozen=True)
class CameraWorkingPoint:
    """One adapter-read physical working point frozen for capability minting.

    The adapter owns ``settings_fingerprint`` and physical readback only.  It
    cannot grant itself exact-capture qualification; installation/Q0 composition
    supplies that separate authority to the endpoint.  The endpoint converts
    these primitive facts once into its authoritative camera-domain values.
    """

    settings_fingerprint: str
    acquisition_mode: str
    frame_shape_yx: tuple[int, int]
    sensor_shape_yx: tuple[int, int]
    roi_origin_yx: tuple[int, int]
    roi_shape_yx: tuple[int, int]
    binning_yx: tuple[int, int]
    dtype: np.dtype
    count_unit: str
    capture_trigger_channels: tuple[str, ...]
    exposure_seconds: float
    required_external_trigger_interval_seconds: float | None
    external_trigger_integration_start_offset_seconds: float | None
    gain: float
    readout_mode: str

    def __post_init__(self) -> None:
        sha256_text(self.settings_fingerprint, "settings_fingerprint")
        object.__setattr__(
            self,
            "acquisition_mode",
            canonical_text(self.acquisition_mode, "acquisition_mode"),
        )
        for name in (
            "frame_shape_yx",
            "sensor_shape_yx",
            "roi_origin_yx",
            "roi_shape_yx",
            "binning_yx",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(
            self,
            "count_unit",
            canonical_text(self.count_unit, "count_unit"),
        )
        if not isinstance(self.capture_trigger_channels, tuple):
            raise TypeError("capture_trigger_channels must be a tuple")
        object.__setattr__(
            self,
            "readout_mode",
            canonical_text(self.readout_mode, "readout_mode"),
        )


@runtime_checkable
class CameraAdapter(Protocol):
    """Record interface consumed by the composition-owned camera endpoint.

    Runtime structural checks only reject missing members; they do not prove
    thread safety, hardware identity, or exact-trigger qualification.  Each
    concrete adapter must pass its contract kit before a composition may bind it.
    This first seam deliberately does not make a real camera READY.
    """

    @property
    def timeout(self) -> float: ...

    def capture_working_point(self) -> CameraWorkingPoint: ...

    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        """Arm one source-owned capture contract.

        ``frames=None`` and ``source_group_sizes=None`` means a hardware-paced
        monitor.  ``buffer_frame_count`` is the physical driver-ring geometry
        for this arm, not a software retention policy.  For a finite capture it
        must equal ``frames`` exactly; for a monitor it is the declared
        ``history_cycles * frames_per_cycle``.  A finite capture also receives
        the ordered, frozen frame groups derived from the measurement cell
        schedule; adapters must validate that the groups exactly cover
        ``frames``.  Trigger transport may verify this contract but never
        defines it.
        """

        ...

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> Sequence[CameraFrameRecord]:
        """Read ordered records; terminalization from another thread must unblock it."""

        ...

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        """Stop, drain, and freeze one stable terminal record.

        The endpoint may first call this from its bounded terminal worker while
        the arm-owner is blocked in a read, then call it again from the arm-owner
        to complete owner-affine teardown.  Both calls must be thread-safe and
        return the same frozen record; the first call must unblock any read.
        Adapters that cannot meet this two-phase contract require an owner-lane
        host and are not eligible for this endpoint yet.
        """

        ...

    def capture_state(self) -> tuple[bool, int]: ...


@runtime_checkable
class CameraAssociationProgress(Protocol):
    """Physical produced-frame counter required by exact live association.

    This is intentionally separate from :class:`CameraAdapter`: finite capture
    and display-only monitoring do not need it.  A hardware endpoint may expose
    live Camera→Pulse association only when its adapter can read the actual
    produced count without consuming a frame.  That read closes the otherwise
    unsafe gap between the monitor's drained ordinal and frames already waiting
    in the driver ring before FPGA FIRE.
    """

    def observed_produced_count(self) -> int: ...



__all__ = [
    "CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT",
    "CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT",
    "CAMERA_FRAME_FACT_FIELDS",
    "CameraAcquisitionMode",
    "CameraAdapter",
    "CameraAssociationProgress",
    "CameraCapabilityEvidence",
    "CameraCaptureDescriptor",
    "CameraCaptureSpec",
    "CameraCaptureTerminalRecord",
    "CameraDatasetEventAdapter",
    "CameraEventReadoutSetting",
    "CameraFrameFactsLike",
    "CameraFrameMetadata",
    "CameraFrameMetadataContract",
    "CameraFrameRecord",
    "CameraPhysicalFacts",
    "CameraSample",
    "CameraSampleContract",
    "CameraWorkingPoint",
    "FrozenCaptureSpec",
    "ReadoutBindingKey",
    "camera_capability_evidence_from_tree",
    "camera_capability_evidence_to_tree",
    "camera_capture_descriptor_from_tree",
    "camera_capture_descriptor_to_tree",
    "camera_capture_spec_from_bytes",
    "camera_capture_spec_to_bytes",
    "camera_event_readout_setting_from_tree",
    "camera_event_readout_setting_to_tree",
    "camera_frame_facts_from_tree",
    "camera_frame_facts_to_tree",
    "camera_frame_metadata_from_tree",
    "camera_frame_metadata_to_tree",
    "camera_output_shape_yx",
    "camera_roi_local_spatial_identity",
    "decode_camera_capture_spec",
    "freeze_camera_capture_spec",
    "frozen_capture_spec_from_tree",
    "frozen_capture_spec_to_tree",
    "normalize_camera_count_dtype",
    "normalize_camera_geometry",
    "readout_binding_key_from_tree",
    "readout_binding_key_to_tree",
    "validate_camera_frame_schema_facts",
    "validate_camera_spatial_axes",
]
