"""Explicit mappings between named logical axes and physical storage order."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np
from zlc_storage.canonical import integer, nonnegative_integer, positive_integer

from ._diagnostic import (
    exact_index_tuple_text,
    exact_integer_text,
)


class AxisLayoutMode(str, Enum):
    RECT_C = "RECT_C"
    RECT_F = "RECT_F"
    EXPLICIT = "EXPLICIT"
    PRODUCT = "PRODUCT"


def _rectangular_mode_for_mapping(
    logical_shape: tuple[int, ...],
    mapping: tuple[tuple[int, ...], ...],
) -> AxisLayoutMode | None:
    if len(mapping) != math.prod(logical_shape):
        return None
    for mode, order in (
        (AxisLayoutMode.RECT_C, "C"),
        (AxisLayoutMode.RECT_F, "F"),
    ):
        if all(
            multi
            == tuple(
                int(index)
                for index in np.unravel_index(row, logical_shape, order=order)
            )
            for row, multi in enumerate(mapping)
        ):
            return mode
    return None


@dataclass(frozen=True, eq=False)
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
        logical_shape = tuple(
            positive_integer(size, "logical_shape entry")
            for size in self.logical_shape
        )
        object.__setattr__(self, "logical_shape", logical_shape)
        if not isinstance(self.mode, AxisLayoutMode):
            raise TypeError("mode must be AxisLayoutMode")
        if (
            self.mode is AxisLayoutMode.RECT_F
            and sum(size > 1 for size in logical_shape) <= 1
        ):
            object.__setattr__(self, "mode", AxisLayoutMode.RECT_C)
        object.__setattr__(
            self,
            "storage_size",
            nonnegative_integer(self.storage_size, "storage_size"),
        )

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
        mapping = tuple(self._validate_multi_index(multi) for multi in mapping)
        rectangular_mode = _rectangular_mode_for_mapping(logical_shape, mapping)
        if rectangular_mode is not None:
            if (
                rectangular_mode is AxisLayoutMode.RECT_F
                and sum(size > 1 for size in logical_shape) <= 1
            ):
                rectangular_mode = AxisLayoutMode.RECT_C
            object.__setattr__(self, "mode", rectangular_mode)
            object.__setattr__(self, "storage_to_multi", None)
            return
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
    def from_mapping(
        cls,
        logical_shape: tuple[int, ...],
        storage_to_multi: tuple[tuple[int, ...], ...],
    ) -> "AxisLayout":
        """Canonicalize a physical row mapping without losing sparse holes."""

        shape = tuple(logical_shape)
        mapping = tuple(tuple(index) for index in storage_to_multi)
        return cls.explicit(shape, mapping)

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
        shape = tuple(size for factor in flattened for size in factor.logical_shape)
        if any(factor.storage_size == 0 for factor in flattened):
            return cls.explicit(shape, ())
        merged: list[AxisLayout] = []
        for factor in flattened:
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
        nontrivial = tuple(factor for factor in flattened if factor.storage_size > 1)
        if len(nontrivial) <= 1 and all(
            factor.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F)
            for factor in flattened
        ):
            mode = (
                nontrivial[0].mode if nontrivial else AxisLayoutMode.RECT_C
            )
            factory = cls.rect_f if mode is AxisLayoutMode.RECT_F else cls.rect_c
            return factory(shape)
        if len(flattened) == 1:
            return flattened[0]
        return cls(
            shape,
            AxisLayoutMode.PRODUCT,
            math.prod(factor.storage_size for factor in flattened),
            None,
            tuple(flattened),
        )

    def multi_index(self, storage_index: int) -> tuple[int, ...]:
        storage_index = self._validate_storage_index(storage_index)
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
        multi = self._validate_multi_index(tuple(multi_index))
        if self.mode is AxisLayoutMode.EXPLICIT:
            assert self._multi_to_storage is not None
            try:
                return self._multi_to_storage[multi]
            except KeyError as exc:
                raise KeyError(
                    "logical point "
                    f"{exact_index_tuple_text(multi)} is not present in "
                    "sparse layout"
                ) from exc
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

    def axis_indices(self, position: int) -> np.ndarray:
        """Return physical-row logical indices for one axis without densifying holes."""

        normalized_position = integer(position, "layout axis position")
        assert normalized_position is not None
        position = normalized_position
        if not 0 <= position < len(self.logical_shape):
            raise IndexError("layout axis position is out of range")
        if self.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F):
            stride = (
                math.prod(self.logical_shape[position + 1 :])
                if self.mode is AxisLayoutMode.RECT_C
                else math.prod(self.logical_shape[:position])
            )
            result = (
                np.arange(self.storage_size, dtype=np.int64) // stride
            ) % self.logical_shape[position]
        elif self.mode is AxisLayoutMode.EXPLICIT:
            assert self.storage_to_multi is not None
            result = np.fromiter(
                (multi[position] for multi in self.storage_to_multi),
                dtype=np.int64,
                count=self.storage_size,
            )
        else:
            assert self.factors is not None
            axis_offset = 0
            result = None
            for factor_index, factor in enumerate(self.factors):
                next_offset = axis_offset + len(factor.logical_shape)
                if position < next_offset:
                    child = factor.axis_indices(position - axis_offset)
                    after = math.prod(
                        item.storage_size for item in self.factors[factor_index + 1 :]
                    )
                    before = math.prod(
                        item.storage_size for item in self.factors[:factor_index]
                    )
                    physical = np.tile(
                        np.repeat(np.arange(factor.storage_size, dtype=np.int64), after),
                        before,
                    )
                    result = child[physical]
                    break
                axis_offset = next_offset
            if result is None:  # pragma: no cover - validated shape guarantees a factor
                raise RuntimeError("PRODUCT axis resolution failed")
        result.setflags(write=False)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AxisLayout):
            return NotImplemented
        return (
            self.logical_shape == other.logical_shape
            and self.mode is other.mode
            and self.storage_size == other.storage_size
            and self.storage_to_multi == other.storage_to_multi
            and self.factors == other.factors
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.logical_shape,
                self.mode,
                self.storage_size,
                self.storage_to_multi,
                self.factors,
            )
        )

    def _validate_storage_index(self, index: int) -> int:
        normalized = integer(index, "storage index")
        assert normalized is not None
        if not 0 <= normalized < self.storage_size:
            raise IndexError(
                "storage index "
                f"{exact_integer_text(normalized)} is outside [0, "
                f"{exact_integer_text(self.storage_size)})"
            )
        return normalized

    def _validate_multi_index(self, multi: tuple[int, ...]) -> tuple[int, ...]:
        if len(multi) != len(self.logical_shape):
            raise ValueError(
                f"multi-index rank {len(multi)} does not match logical rank {len(self.logical_shape)}"
            )
        normalized = tuple(
            integer(index, "multi-index component") for index in multi
        )
        assert all(index is not None for index in normalized)
        resolved = tuple(int(index) for index in normalized)
        for index, size in zip(resolved, self.logical_shape):
            if not 0 <= index < size:
                raise ValueError(
                    "multi-index "
                    f"{exact_index_tuple_text(resolved)} is outside logical "
                    f"shape {exact_index_tuple_text(self.logical_shape)}"
                )
        return resolved


@dataclass(frozen=True, eq=False)
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


__all__ = ["AxisLayout", "AxisLayoutMode", "PointLayout"]
