"""Explicit mapping between logical point axes and physical storage order."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np


class PointLayoutMode(str, Enum):
    RECT_C = "RECT_C"
    RECT_F = "RECT_F"
    EXPLICIT = "EXPLICIT"


@dataclass(frozen=True)
class PointLayout:
    logical_shape: tuple[int, ...]
    mode: PointLayoutMode
    storage_size: int
    storage_to_multi: tuple[tuple[int, ...], ...] | None = None
    _multi_to_storage: Mapping[tuple[int, ...], int] | None = field(
        init=False, repr=False, compare=False, hash=False, default=None
    )

    def __post_init__(self) -> None:
        logical_shape = tuple(self.logical_shape)
        if any(isinstance(size, bool) or not isinstance(size, Integral) or size <= 0 for size in logical_shape):
            raise ValueError("logical_shape entries must be positive integers")
        logical_shape = tuple(int(size) for size in logical_shape)
        object.__setattr__(self, "logical_shape", logical_shape)
        if not isinstance(self.mode, PointLayoutMode):
            raise TypeError("mode must be PointLayoutMode")
        if isinstance(self.storage_size, bool) or not isinstance(self.storage_size, Integral):
            raise TypeError("storage_size must be an integer")
        object.__setattr__(self, "storage_size", int(self.storage_size))
        if self.storage_size <= 0:
            raise ValueError("storage_size must be positive")

        if self.mode in (PointLayoutMode.RECT_C, PointLayoutMode.RECT_F):
            expected = math.prod(logical_shape)
            if self.storage_size != expected:
                raise ValueError(
                    f"rectangular storage_size {self.storage_size} does not match {expected}"
                )
            if self.storage_to_multi is not None:
                raise ValueError("rectangular layouts derive their mapping and cannot store one")
            return

        if self.storage_to_multi is None:
            raise ValueError("EXPLICIT layout requires storage_to_multi")
        mapping = tuple(tuple(index) for index in self.storage_to_multi)
        if len(mapping) != self.storage_size:
            raise ValueError("EXPLICIT storage_size must equal mapping length")
        if len(set(mapping)) != len(mapping):
            raise ValueError("EXPLICIT mapping cannot contain duplicate logical points")
        for multi in mapping:
            self._validate_multi_index(multi)
        mapping = tuple(tuple(int(index) for index in multi) for multi in mapping)
        object.__setattr__(self, "storage_to_multi", mapping)
        object.__setattr__(
            self,
            "_multi_to_storage",
            MappingProxyType({multi: index for index, multi in enumerate(mapping)}),
        )

    @classmethod
    def rect_c(cls, logical_shape: tuple[int, ...]) -> "PointLayout":
        shape = tuple(logical_shape)
        return cls(shape, PointLayoutMode.RECT_C, math.prod(shape))

    @classmethod
    def rect_f(cls, logical_shape: tuple[int, ...]) -> "PointLayout":
        shape = tuple(logical_shape)
        return cls(shape, PointLayoutMode.RECT_F, math.prod(shape))

    @classmethod
    def explicit(
        cls,
        logical_shape: tuple[int, ...],
        storage_to_multi: tuple[tuple[int, ...], ...],
    ) -> "PointLayout":
        mapping = tuple(tuple(index) for index in storage_to_multi)
        return cls(tuple(logical_shape), PointLayoutMode.EXPLICIT, len(mapping), mapping)

    def multi_index(self, storage_index: int) -> tuple[int, ...]:
        self._validate_storage_index(storage_index)
        if self.mode is PointLayoutMode.EXPLICIT:
            assert self.storage_to_multi is not None
            return self.storage_to_multi[storage_index]
        order = "C" if self.mode is PointLayoutMode.RECT_C else "F"
        return tuple(int(index) for index in np.unravel_index(storage_index, self.logical_shape, order=order))

    def storage_index(self, multi_index: tuple[int, ...]) -> int:
        multi = tuple(multi_index)
        self._validate_multi_index(multi)
        if self.mode is PointLayoutMode.EXPLICIT:
            assert self._multi_to_storage is not None
            try:
                return self._multi_to_storage[multi]
            except KeyError as exc:
                raise KeyError(f"logical point {multi} is not present in sparse layout") from exc
        order = "C" if self.mode is PointLayoutMode.RECT_C else "F"
        return int(np.ravel_multi_index(multi, self.logical_shape, order=order))

    def _validate_storage_index(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < self.storage_size:
            raise IndexError(f"storage index {index!r} is outside [0, {self.storage_size})")

    def _validate_multi_index(self, multi: tuple[int, ...]) -> None:
        if len(multi) != len(self.logical_shape):
            raise ValueError(
                f"multi-index rank {len(multi)} does not match logical rank {len(self.logical_shape)}"
            )
        for index, size in zip(multi, self.logical_shape):
            if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < size:
                raise ValueError(f"multi-index {multi} is outside logical shape {self.logical_shape}")
