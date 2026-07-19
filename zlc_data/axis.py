"""Stable axis identities and metadata for named multidimensional data."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

import numpy as np
from zlc_storage.canonical import canonical_text as _nonempty_text

from ._diagnostic import bounded_integer_diagnostic


def _canonical_numeric_coordinate(value: Any, field: str) -> int | float:
    """Give numerically equal coordinates one in-memory and wire identity."""

    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
        raise TypeError(f"{field} must be a Python or NumPy int/float scalar")
    if isinstance(scalar, int):
        return int(scalar)
    numeric = float(scalar)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return int(numeric) if numeric.is_integer() else numeric


@dataclass(frozen=True, order=True)
class AxisId:
    value: str

    def __post_init__(self) -> None:
        _nonempty_text(self.value, "AxisId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class AxisRoleId:
    value: str

    def __post_init__(self) -> None:
        _nonempty_text(self.value, "AxisRoleId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class CoordinateFrameId:
    value: str

    def __post_init__(self) -> None:
        _nonempty_text(self.value, "CoordinateFrameId")

    def __str__(self) -> str:
        return self.value


REPEAT = AxisRoleId("repeat")
SCAN_POINT = AxisRoleId("scan-point")
READOUT_EVENT = AxisRoleId("readout-event")
MONITOR_HISTORY = AxisRoleId("monitor-history")
SPATIAL_X = AxisRoleId("spatial-x")
SPATIAL_Y = AxisRoleId("spatial-y")
SPECTRAL = AxisRoleId("spectral")
HISTOGRAM_BIN = AxisRoleId("histogram-bin")
SITE = AxisRoleId("site")
COMPONENT = AxisRoleId("component")


@dataclass(frozen=True)
class AxisSpec:
    axis_id: AxisId
    name: str
    role: AxisRoleId
    size: int
    coordinates: tuple[Any, ...] | None = None
    unit: str | None = None
    coordinate_frame: CoordinateFrameId | None = None
    index_origin: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        if not isinstance(self.role, AxisRoleId):
            raise TypeError("role must be AxisRoleId")
        _nonempty_text(self.name, "axis name")
        if isinstance(self.size, bool) or not isinstance(self.size, Integral) or self.size <= 0:
            raise ValueError("axis size must be a positive integer")
        object.__setattr__(self, "size", int(self.size))
        if self.coordinates is not None:
            coordinates = []
            for coordinate in self.coordinates:
                scalar = coordinate.item() if isinstance(coordinate, np.generic) else coordinate
                if isinstance(scalar, bool):
                    raise TypeError("axis coordinates cannot be boolean")
                if isinstance(scalar, (int, float)):
                    scalar = _canonical_numeric_coordinate(scalar, "axis coordinate")
                elif scalar is not None and not isinstance(scalar, str):
                    raise TypeError(
                        "axis coordinates must be scalar int/float/str/null values"
                    )
                coordinates.append(scalar)
            coordinates = tuple(coordinates)
            if len(coordinates) != self.size:
                raise ValueError(
                    f"axis coordinates length {len(coordinates)} does not match size {self.size}"
                )
            object.__setattr__(self, "coordinates", coordinates)
        if self.unit is not None:
            _nonempty_text(self.unit, "axis unit")
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame, CoordinateFrameId
        ):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")
        if (
            isinstance(self.index_origin, bool)
            or not isinstance(self.index_origin, Integral)
            or self.index_origin < 0
        ):
            raise ValueError("index_origin must be a non-negative integer")
        object.__setattr__(self, "index_origin", int(self.index_origin))
        if self.coordinates is not None and self.index_origin != 0:
            raise ValueError("index_origin is only valid for an implicit-coordinate axis")

    def coordinate_at(self, index: int) -> Any:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("axis index must be an integer")
        index = int(index)
        if not 0 <= index < self.size:
            raise IndexError(
                f"axis index {bounded_integer_diagnostic(index)} is outside "
                f"[0, {bounded_integer_diagnostic(self.size)})"
            )
        if self.coordinates is None:
            return self.index_origin + index
        return self.coordinates[index]
