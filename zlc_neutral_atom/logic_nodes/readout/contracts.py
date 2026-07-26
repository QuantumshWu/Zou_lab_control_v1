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

from zlc_neutral_atom.devices.camera.contract import (
    CameraCaptureDescriptor,
    CameraPhysicalFacts,
    ReadoutBindingKey,
    camera_output_shape_yx,
    normalize_camera_count_dtype,
    normalize_camera_geometry,
    validate_camera_frame_schema_facts,
    validate_camera_spatial_axes,
)


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
        validate_camera_frame_schema_facts(
            spatial_y_axis_id=self.spatial_y_axis_id,
            spatial_x_axis_id=self.spatial_x_axis_id,
            coordinate_frame=self.coordinate_frame,
            output_shape_yx=camera_output_shape_yx(roi, binning),
            dtype=dtype,
            count_unit=self.count_unit,
            frame_schema=self.frame_schema,
        )

    @classmethod
    def from_camera_working_point(
        cls,
        binding: ReadoutBindingKey,
        physical_facts: CameraPhysicalFacts,
        frame_schema: ValueSchema,
    ) -> "FrameContract":
        """Derive the complete readout contract from endpoint-read live facts."""

        if not isinstance(binding, ReadoutBindingKey):
            raise TypeError("binding must be ReadoutBindingKey")
        if not isinstance(physical_facts, CameraPhysicalFacts):
            raise TypeError("physical_facts must be CameraPhysicalFacts")
        if not isinstance(frame_schema, ValueSchema):
            raise TypeError("frame_schema must be ValueSchema")
        return cls(
            binding=binding,
            camera_identity=physical_facts.camera_identity,
            sensor_identity=physical_facts.sensor_identity,
            optical_path=physical_facts.optical_path,
            sensor_shape_yx=physical_facts.sensor_shape_yx,
            roi_origin_yx=physical_facts.roi_origin_yx,
            roi_shape_yx=physical_facts.roi_shape_yx,
            binning_yx=physical_facts.binning_yx,
            spatial_y_axis_id=physical_facts.spatial_y_axis_id,
            spatial_x_axis_id=physical_facts.spatial_x_axis_id,
            coordinate_frame=physical_facts.coordinate_frame,
            dtype=physical_facts.dtype,
            count_unit=physical_facts.count_unit,
            exposure_seconds=physical_facts.exposure_seconds,
            gain=physical_facts.gain,
            readout_mode=physical_facts.readout_mode,
            opaque_frame_settings_fingerprint=(
                physical_facts.opaque_frame_settings_fingerprint
            ),
            frame_schema=frame_schema,
        )

    def assert_compatible_working_point(
        self,
        binding: ReadoutBindingKey,
        physical_facts: CameraPhysicalFacts,
        frame_schema: ValueSchema,
    ) -> None:
        """Reject a live source unless every calibration-relevant fact agrees."""

        observed = type(self).from_camera_working_point(
            binding,
            physical_facts,
            frame_schema,
        )
        if observed == self:
            return
        mismatches = tuple(
            item.name
            for item in fields(FrameContract)
            if getattr(observed, item.name) != getattr(self, item.name)
        )
        raise ValueError("readout frame contract mismatch: " + ", ".join(mismatches))

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
    "FrameContract",
]
