"""Public plot-kind and named-axis view vocabulary.

The data producer owns axes and point topology.  This module only states how a
declared source participates in a view.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlotKind(str, Enum):
    CURVE = "curve"
    IMAGE = "image"
    HISTOGRAM = "histogram"
    ROLLING = "rolling"
    FACET_GRID = "facet_grid"
    PULSE_TIMELINE = "pulse_timeline"


class AxisDomain(str, Enum):
    REPEAT = "repeat"
    POINT_ROW = "point_row"
    POINT_COORDINATE = "point_coordinate"
    POINT_DIMENSION = "point_dimension"
    DATA = "data"


@dataclass(frozen=True, slots=True)
class AxisRef:
    """A plot-side reference to one producer-declared data source."""

    domain: AxisDomain
    axis_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, AxisDomain):
            raise TypeError("domain must be AxisDomain")
        if self.domain in {AxisDomain.REPEAT, AxisDomain.POINT_ROW}:
            if self.axis_id is not None:
                raise ValueError(f"{self.domain.value} does not accept axis_id")
            return
        if not isinstance(self.axis_id, str) or not self.axis_id.strip():
            raise ValueError(f"{self.domain.value} requires a non-empty axis_id")
        object.__setattr__(self, "axis_id", self.axis_id.strip())

    def __str__(self) -> str:
        """Return the plot-owner label used by every parameter frontend."""

        if self.domain is AxisDomain.REPEAT:
            return "Repeat"
        if self.domain is AxisDomain.POINT_ROW:
            return "Point row"
        prefix = {
            AxisDomain.POINT_COORDINATE: "Point",
            AxisDomain.POINT_DIMENSION: "Grid",
            AxisDomain.DATA: "Data",
        }[self.domain]
        return f"{prefix} · {self.axis_id}"

    @classmethod
    def repeat(cls) -> "AxisRef":
        return cls(AxisDomain.REPEAT)

    @classmethod
    def point_rows(cls) -> "AxisRef":
        return cls(AxisDomain.POINT_ROW)

    @classmethod
    def point(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.POINT_COORDINATE, axis_id)

    @classmethod
    def point_dimension(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.POINT_DIMENSION, axis_id)

    @classmethod
    def data(cls, axis_id: str) -> "AxisRef":
        return cls(AxisDomain.DATA, axis_id)


__all__ = [
    "AxisDomain",
    "AxisRef",
    "PlotKind",
]
