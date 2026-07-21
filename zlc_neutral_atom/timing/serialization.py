"""Strict helpers for current-version JSON/wire objects.

Deserializers in the timing, topology, and runtime layers all cross the same
trust boundary.  They must reject missing and unknown fields consistently;
silently filling an old shape with defaults can turn a stale document into a
different hardware program.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Iterable


def require_exact_fields(
    payload: Mapping[str, object],
    required: Iterable[str],
    *,
    optional: Iterable[str] = (),
    what: str,
) -> None:
    """Require one mapping with exactly the declared current-schema fields."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{what} must be an object")
    required_fields = frozenset(str(field) for field in required)
    optional_fields = frozenset(str(field) for field in optional)
    overlap = required_fields & optional_fields
    if overlap:
        raise RuntimeError(
            f"invalid {what} schema declaration; fields are both required and optional: "
            f"{sorted(overlap)}")
    actual = frozenset(payload)
    missing = required_fields - actual
    unexpected = actual - required_fields - optional_fields
    if missing or unexpected:
        raise ValueError(
            f"invalid {what} fields: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}")


def require_array(value: object, *, what: str) -> list[object]:
    """Return a serialized JSON array, rejecting strings/mappings/iterators."""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{what} must be an ordered array")
    return list(value)


def require_object(value: object, *, what: str) -> Mapping[str, object]:
    """Return a serialized JSON object without coercing another container."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be an object")
    return value


def require_bool(value: object, *, what: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{what} must be a boolean")
    return value


def require_int(value: object, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{what} must be an integer")
    return int(value)


def require_number(value: object, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{what} must be a number")
    return float(value)


def require_string(value: object, *, what: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string")
    return value


__all__ = [
    "require_array",
    "require_bool",
    "require_exact_fields",
    "require_int",
    "require_number",
    "require_object",
    "require_string",
]
