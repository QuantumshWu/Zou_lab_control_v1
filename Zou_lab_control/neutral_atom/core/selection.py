"""Plot-independent data-selection contracts.

Selectors live in a frontend, but the meaning of a selection does not.  A
rectangle on an image, an x range on a trace, a value range on a distribution,
and a rectangle around trap sites are all the same operation: bounds over named
coordinates, optionally scoped to one grid cell.  This module is deliberately
free of Qt and Matplotlib so processors and saved-figure tools consume the exact
same selection as a live plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AxisRange:
    """Inclusive finite bounds for one named coordinate axis."""

    axis: str
    low: float
    high: float

    def __post_init__(self) -> None:
        axis = str(self.axis).strip()
        if not axis:
            raise ValueError("selection axis name must not be empty")
        low, high = sorted((float(self.low), float(self.high)))
        if not np.isfinite([low, high]).all():
            raise ValueError(f"selection bounds for {axis!r} must be finite")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def mask(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.isfinite(array) & (array >= self.low) & (array <= self.high)


@dataclass(frozen=True)
class Selection:
    """Bounds in a named coordinate frame, optionally scoped to a grid cell.

    ``scope`` is an immutable path such as ``("facet:repeat", "cell:2")``.  It
    prevents a cell-local drag from being misapplied to every subplot while
    remaining independent of any concrete grid widget.
    """

    ranges: tuple[AxisRange, ...] = ()
    frame: str = "data"
    scope: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        ranges = tuple(self.ranges)
        names = [item.axis for item in ranges]
        if len(set(names)) != len(names):
            raise ValueError(f"selection axes must be unique, got {names}")
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "frame", str(self.frame or "data"))
        object.__setattr__(self, "scope", tuple(str(v) for v in self.scope))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def x(cls, low: float, high: float, **kwargs) -> "Selection":
        return cls((AxisRange("x", low, high),), **kwargs)

    @classmethod
    def value(cls, low: float, high: float, **kwargs) -> "Selection":
        return cls((AxisRange("value", low, high),), **kwargs)

    @classmethod
    def rectangle(cls, x0: float, x1: float, y0: float, y1: float, **kwargs) -> "Selection":
        return cls((AxisRange("x", x0, x1), AxisRange("y", y0, y1)), **kwargs)

    @property
    def empty(self) -> bool:
        return not self.ranges

    def bounds(self, axis: str) -> tuple[float, float] | None:
        for item in self.ranges:
            if item.axis == str(axis):
                return (item.low, item.high)
        return None

    def mask(self, coordinates: Mapping[str, object], *, length: int | None = None) -> np.ndarray:
        """Vectorized mask over coordinate arrays.

        Missing coordinates are a contract error, never a silent fallback to all
        data.  An empty selection intentionally selects every finite row.
        """

        arrays = {str(name): np.asarray(value) for name, value in coordinates.items()}
        sizes = {int(array.size) for array in arrays.values() if array.ndim > 0}
        if length is None:
            if len(sizes) > 1:
                raise ValueError(f"selection coordinate sizes disagree: {sorted(sizes)}")
            length = next(iter(sizes), 1)
        length = int(length)
        if length < 0:
            raise ValueError("selection length must be non-negative")
        out = np.ones(length, dtype=bool)
        for item in self.ranges:
            if item.axis not in arrays:
                raise KeyError(
                    f"selection needs coordinate {item.axis!r}; available: {sorted(arrays)}")
            values = np.asarray(arrays[item.axis], dtype=float).reshape(-1)
            if values.size != length:
                raise ValueError(
                    f"selection coordinate {item.axis!r} has {values.size} values, expected {length}")
            out &= item.mask(values)
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "scope": list(self.scope),
            "ranges": [
                {"axis": item.axis, "low": item.low, "high": item.high}
                for item in self.ranges
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "Selection":
        payload = dict(payload or {})
        return cls(
            tuple(AxisRange(str(item["axis"]), float(item["low"]), float(item["high"]))
                  for item in payload.get("ranges", ())),
            frame=str(payload.get("frame", "data")),
            scope=tuple(str(v) for v in payload.get("scope", ())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SelectedData:
    """The concrete rows selected from one dataset."""

    coordinates: Mapping[str, np.ndarray]
    values: np.ndarray
    indices: np.ndarray
    selection: Selection = Selection()
    value_shape: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values[:, None]
        coords = {str(key): np.asarray(value).reshape(-1)
                  for key, value in self.coordinates.items()}
        indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        n = int(values.shape[0])
        if indices.size != n:
            raise ValueError(f"selected indices have {indices.size} rows, values have {n}")
        bad = {key: value.size for key, value in coords.items() if value.size != n}
        if bad:
            raise ValueError(f"selected coordinate lengths do not match values: {bad}, values={n}")
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "value_shape", tuple(int(v) for v in self.value_shape) or (1,))

    @property
    def count(self) -> int:
        return int(self.values.shape[0])


def select_rows(
    coordinates: Mapping[str, object],
    values,
    selection: Selection | None = None,
    *,
    finite: bool = True,
    value_shape: Sequence[int] = (1,),
) -> SelectedData:
    """Apply one selection to coordinates and values with no plot-specific branch."""

    selection = selection or Selection()
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    n = int(array.shape[0])
    coords = {str(key): np.asarray(value).reshape(-1) for key, value in coordinates.items()}
    mask = selection.mask(coords, length=n)
    if finite:
        mask &= np.isfinite(array.reshape(n, -1)).all(axis=1)
        for value in coords.values():
            mask &= np.isfinite(np.asarray(value, dtype=float))
    indices = np.flatnonzero(mask)
    return SelectedData(
        {key: value[indices] for key, value in coords.items()},
        array[indices],
        indices,
        selection,
        tuple(int(v) for v in value_shape),
    )


__all__ = ["AxisRange", "Selection", "SelectedData", "select_rows"]
