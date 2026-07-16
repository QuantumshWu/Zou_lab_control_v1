"""Closed, immutable contracts for camera readout and calibration capture roles.

Capture descriptors retain the complete acquisition schedule.  Frame contracts
retain only the physical facts that make one selected image usable by a readout
model.  Calibration row pairing is always a named logical-context join; physical
row order is never treated as semantic correspondence.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
import math

import numpy as np

from zlc_data import (
    READOUT_EVENT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    DatasetSchema,
    ValueSchema,
)
from zlc_storage import (
    canonical_text as _canonical_text,
    nonnegative_integer as _nonnegative_integer,
    nonnegative_real as _nonnegative_real,
    positive_integer as _positive_integer,
    positive_real as _positive_real,
    sha256_text as _sha256,
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


def _minimum_coordinate_separation(coordinates_xy: np.ndarray) -> float:
    """Return the nearest 2D coordinate separation with only O(site) scratch."""

    site_count = len(coordinates_xy)
    if site_count < 2:
        return math.inf
    minimum = math.inf
    for site in range(site_count - 1):
        deltas = coordinates_xy[site + 1 :] - coordinates_xy[site]
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        minimum = min(minimum, float(np.min(distances)))
    return minimum


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


def camera_roi_local_spatial_identity(
    source_id: object,
) -> tuple[AxisId, AxisId, CoordinateFrameId]:
    """Derive the one canonical ROI-local output-pixel identity for a camera."""

    source = _canonical_text(source_id, "camera source_id")
    return (
        AxisId(f"{source}.y"),
        AxisId(f"{source}.x"),
        CoordinateFrameId(f"{source}.roi-local-output-pixels"),
    )


def camera_output_shape_yx(
    roi_shape_yx: tuple[int, int],
    binning_yx: tuple[int, int],
) -> tuple[int, int]:
    """Derive output-pixel geometry from a previously validated ROI/binning."""

    return tuple(
        extent // factor for extent, factor in zip(roi_shape_yx, binning_yx)
    )  # type: ignore[return-value]


def _validate_frame_schema_facts(
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

    def _event_axis(self, schema: DatasetSchema) -> tuple[int, AxisSpec] | None:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        readout_axes = tuple(
            (position, axis)
            for position, axis in enumerate(schema.point_axes)
            if axis.role == READOUT_EVENT
        )
        if len(readout_axes) > 1:
            raise ValueError("DatasetSchema must not contain multiple READOUT_EVENT axes")
        if self.readout_event_axis_id is None:
            if readout_axes:
                raise ValueError("descriptor and schema disagree about a READOUT_EVENT axis")
            return None
        matching = tuple(
            (position, axis)
            for position, axis in enumerate(schema.point_axes)
            if axis.axis_id == self.readout_event_axis_id
        )
        if len(matching) != 1 or matching[0][1].role != READOUT_EVENT:
            raise ValueError("descriptor READOUT_EVENT AxisId is absent or has the wrong role")
        if readout_axes != matching:
            raise ValueError("schema contains a different READOUT_EVENT axis")
        return matching[0]

    def validate_schema(self, schema: DatasetSchema) -> None:
        event_axis = self._event_axis(schema)
        _validate_frame_schema_facts(
            spatial_y_axis_id=self.spatial_y_axis_id,
            spatial_x_axis_id=self.spatial_x_axis_id,
            coordinate_frame=self.coordinate_frame,
            output_shape_yx=self.output_shape_yx,
            dtype=self.dtype,
            count_unit=self.count_unit,
            frame_schema=schema.cell_schema,
        )
        if event_axis is None:
            expected_indices = (0,)
        else:
            expected_indices = tuple(range(event_axis[1].size))
        if tuple(item.event_index for item in self.event_settings) != expected_indices:
            raise ValueError("event_settings must cover every READOUT_EVENT index exactly once")


@dataclass(frozen=True, slots=True, eq=False)
class _CalibrationCaptureJoin:
    """Compact owner-produced join reused by preflight, analysis, and commit."""

    repeat_axis_id: AxisId
    repeat_count: int
    context_axis_ids: tuple[AxisId, ...]
    _context_indices: tuple[tuple[int, ...], ...]
    _selected_point_storage_rows: tuple[tuple[int, ...], ...]

    @property
    def group_count(self) -> int:
        return self.repeat_count * len(self._context_indices)

    def rows(self) -> Iterator[tuple[int, tuple[int, ...], int]]:
        for repeat_index in range(self.repeat_count):
            for selected_rows in self._selected_point_storage_rows:
                yield (
                    repeat_index,
                    selected_rows[:-1],
                    selected_rows[-1],
                )

    def contexts(self) -> Iterator[tuple[tuple[AxisId, int], ...]]:
        for repeat_index in range(self.repeat_count):
            for indices in self._context_indices:
                yield (
                    (self.repeat_axis_id, repeat_index),
                    *tuple(
                        (axis_id, int(index))
                        for axis_id, index in zip(
                            self.context_axis_ids,
                            indices,
                            strict=True,
                        )
                    ),
                )

    def matches_contexts(
        self,
        observed: tuple[tuple[tuple[AxisId, int], ...], ...],
    ) -> bool:
        if len(observed) != self.group_count:
            return False
        return all(
            expected == actual
            for expected, actual in zip(self.contexts(), observed, strict=True)
        )


@dataclass(frozen=True)
class CalibrationCaptureLayout:
    """Explicit calibration event roles joined by all other logical axes."""

    readout_event_axis_id: AxisId
    reference_event_indices: tuple[int, ...]
    readout_event_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        references = tuple(
            _nonnegative_integer(index, "reference_event_indices entry")
            for index in self.reference_event_indices
        )
        if not references:
            raise ValueError("at least one reference event index is required")
        if len(set(references)) != len(references):
            raise ValueError("reference_event_indices must be unique")
        references = tuple(sorted(references))
        readout = _nonnegative_integer(
            self.readout_event_index,
            "readout_event_index",
        )
        if readout in references:
            raise ValueError("reference and readout event indices must be disjoint")
        object.__setattr__(self, "reference_event_indices", references)
        object.__setattr__(self, "readout_event_index", readout)

    def _axis_position(self, schema: DatasetSchema) -> tuple[int, AxisSpec]:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        readout_axes = tuple(
            (position, axis)
            for position, axis in enumerate(schema.point_axes)
            if axis.role == READOUT_EVENT
        )
        if len(readout_axes) != 1:
            raise ValueError("calibration schema must have exactly one READOUT_EVENT point axis")
        position, axis = readout_axes[0]
        if axis.axis_id != self.readout_event_axis_id:
            raise ValueError("calibration layout names a different READOUT_EVENT AxisId")
        selected = self.reference_event_indices + (self.readout_event_index,)
        if any(index >= axis.size for index in selected):
            raise ValueError("calibration event index is outside the READOUT_EVENT axis")
        return position, axis

    def _resolve(self, schema: DatasetSchema) -> _CalibrationCaptureJoin:
        """Join selected events by named context, never by filtered row position."""

        event_position, _event_axis = self._axis_position(schema)
        context_positions = tuple(
            position
            for position in range(len(schema.point_axes))
            if position != event_position
        )
        context_axis_ids = tuple(
            schema.point_axes[position].axis_id for position in context_positions
        )
        selected_events = self.reference_event_indices + (self.readout_event_index,)
        event_slots = {
            event_index: slot for slot, event_index in enumerate(selected_events)
        }
        rows_by_context: dict[tuple[int, ...], list[int | None]] = {}
        for storage_row in range(schema.point_layout.storage_size):
            logical = schema.point_layout.multi_index(storage_row)
            slot = event_slots.get(logical[event_position])
            if slot is None:
                continue
            context = tuple(logical[position] for position in context_positions)
            rows = rows_by_context.get(context)
            if rows is None:
                rows = [None] * len(selected_events)
                rows_by_context[context] = rows
            rows[slot] = storage_row
        if not rows_by_context:
            raise ValueError("calibration selected events have no physical context rows")
        context_indices = tuple(sorted(rows_by_context))
        selected_rows = []
        for context in context_indices:
            rows = rows_by_context[context]
            if any(row is None for row in rows):
                raise ValueError(
                    "reference/readout events do not have identical logical context sets"
                )
            selected_rows.append(tuple(int(row) for row in rows if row is not None))
        return _CalibrationCaptureJoin(
            schema.repeat_axis.axis_id,
            schema.repeat_axis.size,
            context_axis_ids,
            context_indices,
            tuple(selected_rows),
        )

    def _memory_upper_bounds(
        self,
        schema: DatasetSchema,
    ) -> tuple[int, int, int]:
        """Return group-count, join-build, and retained-join upper bounds."""

        event_position, _event_axis = self._axis_position(schema)
        context_rank = len(schema.point_axes) - 1
        selected_events = self.reference_event_indices + (self.readout_event_index,)
        event_count = len(selected_events)
        logical_contexts = math.prod(
            axis.size
            for position, axis in enumerate(schema.point_axes)
            if position != event_position
        )
        layout = schema.point_layout
        if layout.storage_size == math.prod(layout.logical_shape):
            event_row_counts = [logical_contexts] * event_count
        else:
            event_slots = {
                event_index: slot for slot, event_index in enumerate(selected_events)
            }
            event_row_counts = [0] * event_count
            for storage_row in range(layout.storage_size):
                slot = event_slots.get(layout.multi_index(storage_row)[event_position])
                if slot is not None:
                    event_row_counts[slot] += 1
        complete_contexts = max(event_row_counts)
        build_contexts = min(logical_contexts, sum(event_row_counts))
        # Conservative CPython object-graph allowances.  The build bound also
        # covers an invalid sparse source in which every selected row opens a
        # different incomplete context before the join rejects it.
        retained = complete_contexts * (
            512 + 64 * (context_rank + event_count)
        )
        build_peak = (
            build_contexts * (1024 + 128 * (context_rank + event_count))
            + retained
        )
        return (
            schema.repeat_axis.size * complete_contexts,
            build_peak,
            retained,
        )

@dataclass(frozen=True)
class FrameContract:
    """Single-frame facts determining calibration-model applicability."""

    binding: ReadoutBindingKey
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
    exposure_seconds: float
    gain: float
    readout_mode: str
    opaque_frame_settings_fingerprint: str | None
    frame_schema: ValueSchema

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ReadoutBindingKey):
            raise TypeError("binding must be ReadoutBindingKey")
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
        dtype = normalize_camera_count_dtype(self.dtype, "dtype")
        object.__setattr__(self, "dtype", dtype)
        _canonical_text(self.count_unit, "count_unit")
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
        _validate_frame_schema_facts(
            spatial_y_axis_id=self.spatial_y_axis_id,
            spatial_x_axis_id=self.spatial_x_axis_id,
            coordinate_frame=self.coordinate_frame,
            output_shape_yx=camera_output_shape_yx(roi, binning),
            dtype=dtype,
            count_unit=self.count_unit,
            frame_schema=self.frame_schema,
        )

    @classmethod
    def _from_schema_impl(
        cls,
        binding: ReadoutBindingKey,
        descriptor: CameraCaptureDescriptor,
        schema: DatasetSchema,
        *,
        readout_event_index: int,
        require_physical_event_witness: bool,
    ) -> "FrameContract":
        if not isinstance(binding, ReadoutBindingKey):
            raise TypeError("binding must be ReadoutBindingKey")
        if not isinstance(descriptor, CameraCaptureDescriptor):
            raise TypeError("descriptor must be CameraCaptureDescriptor")
        descriptor.validate_schema(schema)
        index = _nonnegative_integer(
            readout_event_index,
            "readout_event_index",
        )
        event_axis = descriptor._event_axis(schema)
        if event_axis is None:
            if index != 0:
                raise ValueError("a capture without READOUT_EVENT axis only has event index 0")
        else:
            if index >= event_axis[1].size:
                raise ValueError("readout_event_index is outside the READOUT_EVENT axis")
            if require_physical_event_witness and not np.any(
                schema.point_layout.axis_indices(event_axis[0]) == index
            ):
                raise ValueError("selected readout event is absent from the physical PointLayout")
        setting = descriptor.setting(index)
        return cls(
            binding=binding,
            camera_identity=descriptor.camera_identity,
            sensor_identity=descriptor.sensor_identity,
            optical_path=descriptor.optical_path,
            sensor_shape_yx=descriptor.sensor_shape_yx,
            roi_origin_yx=descriptor.roi_origin_yx,
            roi_shape_yx=descriptor.roi_shape_yx,
            binning_yx=descriptor.binning_yx,
            spatial_y_axis_id=descriptor.spatial_y_axis_id,
            spatial_x_axis_id=descriptor.spatial_x_axis_id,
            coordinate_frame=descriptor.coordinate_frame,
            dtype=descriptor.dtype,
            count_unit=descriptor.count_unit,
            exposure_seconds=setting.exposure_seconds,
            gain=setting.gain,
            readout_mode=setting.readout_mode,
            opaque_frame_settings_fingerprint=(
                setting.opaque_frame_settings_fingerprint
            ),
            frame_schema=schema.cell_schema,
        )

    @classmethod
    def _resolve_calibration_capture(
        cls,
        binding: ReadoutBindingKey,
        descriptor: CameraCaptureDescriptor,
        schema: DatasetSchema,
        layout: CalibrationCaptureLayout,
    ) -> tuple["FrameContract", _CalibrationCaptureJoin]:
        """Resolve the named sparse join once and derive its frame contract."""

        if not isinstance(layout, CalibrationCaptureLayout):
            raise TypeError("layout must be CalibrationCaptureLayout")
        if not isinstance(descriptor, CameraCaptureDescriptor):
            raise TypeError("descriptor must be CameraCaptureDescriptor")
        if descriptor.readout_event_axis_id != layout.readout_event_axis_id:
            raise ValueError("capture descriptor and calibration layout name different event axes")
        contract = cls._from_schema_impl(
            binding,
            descriptor,
            schema,
            readout_event_index=layout.readout_event_index,
            require_physical_event_witness=False,
        )
        return contract, layout._resolve(schema)

    def assert_compatible(
        self,
        binding: ReadoutBindingKey,
        descriptor: CameraCaptureDescriptor,
        schema: DatasetSchema,
        *,
        readout_event_index: int,
    ) -> None:
        observed = type(self)._from_schema_impl(
            binding,
            descriptor,
            schema,
            readout_event_index=readout_event_index,
            require_physical_event_witness=True,
        )
        if observed == self:
            return
        mismatches = tuple(
            item.name
            for item in fields(FrameContract)
            if getattr(observed, item.name) != getattr(self, item.name)
        )
        raise ValueError("readout frame contract mismatch: " + ", ".join(mismatches))


__all__ = [
    "CalibrationCaptureLayout",
    "CameraCaptureDescriptor",
    "CameraEventReadoutSetting",
    "FrameContract",
    "ReadoutBindingKey",
    "camera_roi_local_spatial_identity",
    "camera_output_shape_yx",
    "normalize_camera_count_dtype",
    "normalize_camera_geometry",
    "validate_camera_spatial_axes",
]
