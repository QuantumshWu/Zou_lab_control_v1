"""Compact coordinate contract for dense, axis-aligned 2-D rasters.

A camera image has ``H*W`` values but only four coordinate parameters: shape, origin, and pixel
spacing.  Expanding that fact into an ``(H*W, 2)`` float table costs 36.9 MiB for 1920x1200 before
any fit or render begins.  :class:`RegularRaster` keeps the geometry in O(1); coordinates are
materialised only at an explicit array/export boundary or for selected sample indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class RegularRaster:
    """Row-major regular 2-D coordinate grid.

    ``shape`` is ``(height, width)``; ``origin`` is the coordinate of pixel ``[0, 0]`` as
    ``(x, y)``; ``spacing`` is positive ``(dx, dy)``.  Values are deliberately not stored here:
    the same immutable geometry can describe every frame in a live stream.
    """

    shape: tuple[int, int]
    origin: tuple[float, float] = (0.0, 0.0)
    spacing: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if len(tuple(self.shape)) != 2:
            raise ValueError(f"regular raster shape must be (height, width), got {self.shape!r}")
        height, width = (int(value) for value in self.shape)
        if height < 1 or width < 1:
            raise ValueError(f"regular raster dimensions must be positive, got {(height, width)}")
        ox, oy = (float(value) for value in self.origin)
        dx, dy = (float(value) for value in self.spacing)
        if not np.isfinite((ox, oy, dx, dy)).all() or dx <= 0.0 or dy <= 0.0:
            raise ValueError(
                f"regular raster origin/spacing must be finite with positive spacing, got "
                f"origin={(ox, oy)}, spacing={(dx, dy)}")
        object.__setattr__(self, "shape", (height, width))
        object.__setattr__(self, "origin", (ox, oy))
        object.__setattr__(self, "spacing", (dx, dy))

    @property
    def point_count(self) -> int:
        return int(self.shape[0] * self.shape[1])

    @property
    def array_shape(self) -> tuple[int, int]:
        """Shape of the equivalent coordinate table, without allocating it."""

        return (self.point_count, 2)

    @property
    def x_axis(self) -> np.ndarray:
        return self.origin[0] + self.spacing[0] * np.arange(self.shape[1], dtype=float)

    @property
    def y_axis(self) -> np.ndarray:
        return self.origin[1] + self.spacing[1] * np.arange(self.shape[0], dtype=float)

    def coordinates(self, indices: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        """Return coordinate rows for ``indices`` or, explicitly, the complete grid."""

        if indices is None:
            flat = np.arange(self.point_count, dtype=np.int64)
        else:
            flat = np.asarray(indices, dtype=np.int64).reshape(-1)
            if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= self.point_count):
                raise IndexError("regular raster coordinate index outside the grid")
        rows, cols = np.divmod(flat, self.shape[1])
        out = np.empty((flat.size, 2), dtype=float)
        out[:, 0] = self.origin[0] + cols * self.spacing[0]
        out[:, 1] = self.origin[1] + rows * self.spacing[1]
        return out

    def coordinate_mapping(self, indices=None) -> dict[str, np.ndarray]:
        rows = self.coordinates(indices)
        return {"x": rows[:, 0], "y": rows[:, 1]}

    def __len__(self) -> int:
        return self.point_count

    @property
    def ndim(self) -> int:
        return 2

    @property
    def table_shape(self) -> tuple[int, int]:
        return self.array_shape

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += self.point_count
            return self.coordinates([index])[0]
        return self.coordinates()[key]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = self.coordinates()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array


__all__ = ["RegularRaster"]
