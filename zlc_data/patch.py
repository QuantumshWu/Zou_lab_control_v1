"""Atomic internal dataset deltas keyed by named dataset cells."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ._arrays import immutable_array
from .validity import ComponentValidity, Invalid, Valid
from .value import BlockId, DatasetRevision


ValueValidity = Valid | Invalid | ComponentValidity


@dataclass(frozen=True, eq=False)
class DataPatch:
    """One immutable values+validity transaction for a materialized dataset.

    ``DataPatch`` is deliberately not a stream payload.  It is the compact delta
    shared only by a DatasetBuilder, its snapshot store, and persistence code.
    """

    block_id: BlockId
    base_revision: DatasetRevision
    result_revision: DatasetRevision
    target_cells: tuple[tuple[int, int], ...]
    values: np.ndarray
    validity_patch: tuple[ValueValidity, ...]
    schema_fingerprint: str
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.base_revision, DatasetRevision):
            raise TypeError("base_revision must be DatasetRevision")
        if not isinstance(self.result_revision, DatasetRevision):
            raise TypeError("result_revision must be DatasetRevision")
        if self.result_revision.value != self.base_revision.value + 1:
            raise ValueError("result_revision must immediately follow base_revision")
        cells = tuple(tuple(cell) for cell in self.target_cells)
        for cell in cells:
            if len(cell) != 2 or any(
                isinstance(index, bool) or not isinstance(index, Integral) or index < 0
                for index in cell
            ):
                raise ValueError("target_cells must contain non-negative (repeat, point) indices")
        cells = tuple((int(repeat), int(point)) for repeat, point in cells)
        if not cells:
            raise ValueError("DataPatch requires at least one target cell")
        if len(set(cells)) != len(cells):
            raise ValueError("target_cells must be unique within one DataPatch")
        validity = tuple(self.validity_patch)
        if len(validity) != len(cells):
            raise ValueError("validity_patch must contain one validity value per target cell")
        if any(not isinstance(item, (Valid, Invalid, ComponentValidity)) for item in validity):
            raise TypeError("validity_patch contains an unsupported validity value")
        if (
            not isinstance(self.schema_fingerprint, str)
            or len(self.schema_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.schema_fingerprint)
        ):
            raise ValueError("schema_fingerprint must be a lowercase SHA-256 digest")
        array = np.asarray(self.values)
        if array.shape[0:1] != (len(cells),):
            raise ValueError("values leading dimension must match target_cells")
        object.__setattr__(self, "target_cells", cells)
        object.__setattr__(self, "validity_patch", validity)
        object.__setattr__(
            self,
            "values",
            immutable_array(array, dtype=array.dtype, shape=array.shape),
        )


__all__ = ["DataPatch", "ValueValidity"]
