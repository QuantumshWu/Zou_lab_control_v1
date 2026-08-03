"""Small strict validators shared by backend-neutral plotting records."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import math
import numpy as np


def finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def integer(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    optional: bool = False,
) -> int | None:
    if value is None and optional:
        return None
    suffix = " or None" if optional else ""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer{suffix}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    result = value.strip()
    if not result and not allow_empty:
        raise ValueError(f"{field} must be non-empty")
    return result


def nonempty_text(value: object, field: str) -> str:
    return text(value, field)


def optional_nonempty_text(value: object, field: str) -> str | None:
    return None if value is None else nonempty_text(value, field)


def readonly_copy(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return independent array storage under a read-only view contract."""

    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result
