"""Named validity contracts and immutable masks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._arrays import immutable_bool_array
from .axis import AxisId


class ValidityMode(str, Enum):
    VALUE = "VALUE"
    COMPONENTS = "COMPONENTS"


@dataclass(frozen=True)
class ValidityContract:
    mode: ValidityMode
    component_axis_ids: tuple[AxisId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ValidityMode):
            raise TypeError("mode must be ValidityMode")
        axis_ids = tuple(self.component_axis_ids)
        if any(not isinstance(axis_id, AxisId) for axis_id in axis_ids):
            raise TypeError("component_axis_ids must contain AxisId values")
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("component_axis_ids must be unique")
        if self.mode is ValidityMode.VALUE and axis_ids:
            raise ValueError("VALUE validity cannot declare component axes")
        if self.mode is ValidityMode.COMPONENTS and not axis_ids:
            raise ValueError("COMPONENTS validity requires at least one component axis")
        object.__setattr__(self, "component_axis_ids", axis_ids)

    @classmethod
    def value(cls) -> "ValidityContract":
        return cls(ValidityMode.VALUE)

    @classmethod
    def components(cls, *axis_ids: AxisId) -> "ValidityContract":
        return cls(ValidityMode.COMPONENTS, tuple(axis_ids))


@dataclass(frozen=True)
class Valid:
    pass


@dataclass(frozen=True)
class Invalid:
    pass


VALID = Valid()
INVALID = Invalid()


@dataclass(frozen=True, eq=False)
class CellValidity:
    mask: np.ndarray
    __hash__ = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError("CellValidity mask must have shape (R, P)")
        object.__setattr__(self, "mask", immutable_bool_array(mask, shape=mask.shape))


@dataclass(frozen=True, eq=False)
class ComponentValidity:
    """Validity over exactly the named axes of one :class:`Value`.

    Dataset carrier axes are deliberately not implicit here.  A materialized
    dataset uses :class:`DatasetComponentValidity`, whose leading repeat and
    physical-point dimensions are part of that distinct carrier contract.
    """

    axis_ids: tuple[AxisId, ...]
    mask: np.ndarray
    __hash__ = None

    def __post_init__(self) -> None:
        axis_ids = tuple(self.axis_ids)
        if not axis_ids:
            raise ValueError("ComponentValidity requires at least one named axis")
        if any(not isinstance(axis_id, AxisId) for axis_id in axis_ids):
            raise TypeError("ComponentValidity axis_ids must contain AxisId values")
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("ComponentValidity axis_ids must be unique")
        mask = np.asarray(self.mask)
        if mask.ndim != len(axis_ids):
            raise ValueError(
                "ComponentValidity mask rank must equal its named-axis count"
            )
        object.__setattr__(self, "axis_ids", axis_ids)
        object.__setattr__(self, "mask", immutable_bool_array(mask, shape=mask.shape))

@dataclass(frozen=True, eq=False)
class DatasetComponentValidity:
    """Physical dataset validity over ``(R, P, *named component axes)``.

    ``P`` is the dataset's physical point-storage index, not another guessed
    logical axis.  Keeping this representation separate from
    :class:`ComponentValidity` makes both ranks explicit and prevents a mask
    meant for a single Value from being mistaken for a whole Dataset mask (or
    vice versa).
    """

    axis_ids: tuple[AxisId, ...]
    mask: np.ndarray
    __hash__ = None

    def __post_init__(self) -> None:
        axis_ids = tuple(self.axis_ids)
        if not axis_ids:
            raise ValueError("DatasetComponentValidity requires a named component axis")
        if any(not isinstance(axis_id, AxisId) for axis_id in axis_ids):
            raise TypeError(
                "DatasetComponentValidity axis_ids must contain AxisId values"
            )
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("DatasetComponentValidity axis_ids must be unique")
        mask = np.asarray(self.mask)
        if mask.ndim != 2 + len(axis_ids):
            raise ValueError(
                "DatasetComponentValidity mask rank must be 2 + named-axis count"
            )
        object.__setattr__(self, "axis_ids", axis_ids)
        object.__setattr__(self, "mask", immutable_bool_array(mask, shape=mask.shape))
