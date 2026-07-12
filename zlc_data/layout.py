"""Explicit mappings between named logical axes and physical storage order."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np


class AxisLayoutMode(str, Enum):
    RECT_C = "RECT_C"
    RECT_F = "RECT_F"
    EXPLICIT = "EXPLICIT"
    PRODUCT = "PRODUCT"


# Dataset schemas retain the domain-specific spelling while transforms and fit
# results use the same single layout-mode implementation.
PointLayoutMode = AxisLayoutMode


@dataclass(frozen=True)
class AxisLayout:
    """A finite physical row table over a logical Cartesian axis space.

    Rectangular layouts derive every logical row.  ``EXPLICIT`` stores only the
    rows that actually exist, so a sparse or even empty result never has to be
    densified or represented as an invalid value.
    """

    logical_shape: tuple[int, ...]
    mode: AxisLayoutMode
    storage_size: int
    storage_to_multi: tuple[tuple[int, ...], ...] | None = None
    factors: tuple["AxisLayout", ...] | None = None
    _multi_to_storage: Mapping[tuple[int, ...], int] | None = field(
        init=False, repr=False, compare=False, hash=False, default=None
    )

    def __post_init__(self) -> None:
        logical_shape = tuple(self.logical_shape)
        if any(isinstance(size, bool) or not isinstance(size, Integral) or size <= 0 for size in logical_shape):
            raise ValueError("logical_shape entries must be positive integers")
        logical_shape = tuple(int(size) for size in logical_shape)
        object.__setattr__(self, "logical_shape", logical_shape)
        if not isinstance(self.mode, AxisLayoutMode):
            raise TypeError("mode must be AxisLayoutMode")
        if isinstance(self.storage_size, bool) or not isinstance(self.storage_size, Integral):
            raise TypeError("storage_size must be an integer")
        object.__setattr__(self, "storage_size", int(self.storage_size))
        if self.storage_size < 0:
            raise ValueError("storage_size must be non-negative")

        if self.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F):
            expected = math.prod(logical_shape)
            if self.storage_size != expected:
                raise ValueError(
                    f"rectangular storage_size {self.storage_size} does not match {expected}"
                )
            if self.storage_to_multi is not None:
                raise ValueError("rectangular layouts derive their mapping and cannot store one")
            if self.factors is not None:
                raise ValueError("rectangular layouts cannot store product factors")
            return

        if self.mode is AxisLayoutMode.PRODUCT:
            if self.storage_to_multi is not None:
                raise ValueError("PRODUCT layout cannot store an explicit mapping")
            factors = () if self.factors is None else tuple(self.factors)
            if len(factors) < 2 or any(not isinstance(factor, AxisLayout) for factor in factors):
                raise ValueError("PRODUCT layout requires AxisLayout factors")
            if any(factor.mode is AxisLayoutMode.PRODUCT for factor in factors):
                raise ValueError("PRODUCT factors must be flattened")
            if any(
                factor.logical_shape == () and factor.storage_size == 1
                for factor in factors
            ):
                raise ValueError("PRODUCT identity factors are non-canonical")
            if any(
                factor.mode is AxisLayoutMode.RECT_F
                and len(factor.logical_shape) <= 1
                for factor in factors
            ):
                raise ValueError("rank-one RECT_F PRODUCT factors are non-canonical")
            if any(
                left.mode is AxisLayoutMode.RECT_C
                and right.mode is AxisLayoutMode.RECT_C
                for left, right in zip(factors, factors[1:])
            ):
                raise ValueError("adjacent RECT_C PRODUCT factors are non-canonical")
            expected_shape = tuple(size for factor in factors for size in factor.logical_shape)
            expected_storage = math.prod(factor.storage_size for factor in factors)
            if logical_shape != expected_shape or self.storage_size != expected_storage:
                raise ValueError("PRODUCT layout shape/storage do not match its factors")
            object.__setattr__(self, "factors", factors)
            return

        if self.storage_to_multi is None:
            raise ValueError("EXPLICIT layout requires storage_to_multi")
        if self.factors is not None:
            raise ValueError("EXPLICIT layout cannot store product factors")
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
    def rect_c(cls, logical_shape: tuple[int, ...]) -> "AxisLayout":
        shape = tuple(logical_shape)
        return cls(shape, AxisLayoutMode.RECT_C, math.prod(shape))

    @classmethod
    def rect_f(cls, logical_shape: tuple[int, ...]) -> "AxisLayout":
        shape = tuple(logical_shape)
        return cls(shape, AxisLayoutMode.RECT_F, math.prod(shape))

    @classmethod
    def explicit(
        cls,
        logical_shape: tuple[int, ...],
        storage_to_multi: tuple[tuple[int, ...], ...],
    ) -> "AxisLayout":
        mapping = tuple(tuple(index) for index in storage_to_multi)
        return cls(tuple(logical_shape), AxisLayoutMode.EXPLICIT, len(mapping), mapping)

    @classmethod
    def product(cls, *factors: "AxisLayout") -> "AxisLayout":
        flattened: list[AxisLayout] = []
        for factor in factors:
            if not isinstance(factor, AxisLayout):
                raise TypeError("PRODUCT factors must be AxisLayout values")
            if factor.mode is AxisLayoutMode.PRODUCT:
                assert factor.factors is not None
                flattened.extend(factor.factors)
            else:
                flattened.append(factor)
        flattened = [
            factor
            for factor in flattened
            if not (factor.logical_shape == () and factor.storage_size == 1)
        ]
        if not flattened:
            return AxisLayout.rect_c(())
        merged: list[AxisLayout] = []
        for factor in flattened:
            if factor.mode is AxisLayoutMode.RECT_F and len(factor.logical_shape) <= 1:
                factor = AxisLayout.rect_c(factor.logical_shape)
            if (
                merged
                and merged[-1].mode is AxisLayoutMode.RECT_C
                and factor.mode is AxisLayoutMode.RECT_C
            ):
                merged[-1] = AxisLayout.rect_c(
                    merged[-1].logical_shape + factor.logical_shape
                )
            else:
                merged.append(factor)
        flattened = merged
        if len(flattened) == 1:
            return flattened[0]
        shape = tuple(size for factor in flattened for size in factor.logical_shape)
        return cls(
            shape,
            AxisLayoutMode.PRODUCT,
            math.prod(factor.storage_size for factor in flattened),
            None,
            tuple(flattened),
        )

    def multi_index(self, storage_index: int) -> tuple[int, ...]:
        self._validate_storage_index(storage_index)
        if self.mode is AxisLayoutMode.EXPLICIT:
            assert self.storage_to_multi is not None
            return self.storage_to_multi[storage_index]
        if self.mode is AxisLayoutMode.PRODUCT:
            assert self.factors is not None
            factor_storage = tuple(factor.storage_size for factor in self.factors)
            physical = np.unravel_index(storage_index, factor_storage, order="C")
            return tuple(
                coordinate
                for factor, factor_index in zip(self.factors, physical)
                for coordinate in factor.multi_index(int(factor_index))
            )
        order = "C" if self.mode is AxisLayoutMode.RECT_C else "F"
        return tuple(int(index) for index in np.unravel_index(storage_index, self.logical_shape, order=order))

    def storage_index(self, multi_index: tuple[int, ...]) -> int:
        multi = tuple(multi_index)
        self._validate_multi_index(multi)
        if self.mode is AxisLayoutMode.EXPLICIT:
            assert self._multi_to_storage is not None
            try:
                return self._multi_to_storage[multi]
            except KeyError as exc:
                raise KeyError(f"logical point {multi} is not present in sparse layout") from exc
        if self.mode is AxisLayoutMode.PRODUCT:
            assert self.factors is not None
            offset = 0
            factor_storage_indices: list[int] = []
            for factor in self.factors:
                width = len(factor.logical_shape)
                factor_storage_indices.append(
                    factor.storage_index(multi[offset : offset + width])
                )
                offset += width
            return int(
                np.ravel_multi_index(
                    tuple(factor_storage_indices),
                    tuple(factor.storage_size for factor in self.factors),
                    order="C",
                )
            )
        order = "C" if self.mode is AxisLayoutMode.RECT_C else "F"
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


@dataclass(frozen=True)
class PointLayout(AxisLayout):
    """Dataset point-axis specialization of :class:`AxisLayout`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.storage_size == 0:
            raise ValueError("PointLayout storage_size must be positive")
        if self.mode is AxisLayoutMode.PRODUCT:
            raise ValueError("PointLayout cannot be PRODUCT")

    @classmethod
    def product(cls, *factors: AxisLayout) -> AxisLayout:
        raise TypeError("PointLayout does not support PRODUCT composition")


__all__ = ["AxisLayout", "AxisLayoutMode", "PointLayout", "PointLayoutMode"]
