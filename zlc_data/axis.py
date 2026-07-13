"""Stable axis identities and metadata for named multidimensional data."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

import numpy as np
from zlc_storage.canonical import canonical_text as _nonempty_text


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
            coordinates = tuple(
                coordinate.item() if isinstance(coordinate, np.generic) else coordinate
                for coordinate in self.coordinates
            )
            if len(coordinates) != self.size:
                raise ValueError(
                    f"axis coordinates length {len(coordinates)} does not match size {self.size}"
                )
            if any(
                coordinate is not None
                and not isinstance(coordinate, (bool, int, float, str))
                for coordinate in coordinates
            ):
                raise TypeError("axis coordinates must be scalar bool/int/float/str/null values")
            if any(isinstance(coordinate, float) and not math.isfinite(coordinate) for coordinate in coordinates):
                raise ValueError("axis coordinates must be finite; missing points belong in validity")
            object.__setattr__(self, "coordinates", coordinates)
        if self.unit is not None:
            _nonempty_text(self.unit, "axis unit")
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame, CoordinateFrameId
        ):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")
