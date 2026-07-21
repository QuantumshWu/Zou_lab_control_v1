"""Strict numeric validators shared by the timing, topology and runtime layers.

One rejection vocabulary for every deserializer and compile entry: a value that is not
finite, not integral, or out of range raises ``ValueError`` naming the offending field.
Silently coercing instead can turn a stale document into a DIFFERENT hardware program,
which is why these live in the headless value layer and everyone imports the same five.
"""

from __future__ import annotations

import numpy as np

__all__ = ["finite_float", "nonnegative_int", "positive_int", "positive_float",
           "nonnegative_float"]


def finite_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite, not a boolean.")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out

def nonnegative_int(value, name: str) -> int:
    out = finite_float(value, name)
    if int(out) != out or out < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(out)

def positive_int(value, name: str) -> int:
    out = nonnegative_int(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return out

def positive_float(value, name: str) -> float:
    out = finite_float(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be > 0.")
    return out

def nonnegative_float(value, name: str) -> float:
    out = finite_float(value, name)
    if out < 0:
        raise ValueError(f"{name} must be >= 0.")
    return out
