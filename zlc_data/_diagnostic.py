"""Exact diagnostic text for integer authority values."""

from __future__ import annotations

from numbers import Integral


def exact_integer_text(value: object) -> str:
    """Return a complete, reversible integer representation.

    Python may reject very large decimal conversions through its interpreter
    safety limit.  Hexadecimal has no such limit and still preserves every bit,
    so it is the exact fallback rather than an application-level truncation.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        return repr(value)
    integer = int(value)
    try:
        return str(integer)
    except ValueError:
        return hex(integer)


def exact_index_tuple_text(values: tuple[object, ...]) -> str:
    """Format one logical index tuple without losing any component value."""

    labels = tuple(exact_integer_text(value) for value in values)
    if len(labels) == 1:
        return f"({labels[0]},)"
    return f"({', '.join(labels)})"


__all__ = ["exact_index_tuple_text", "exact_integer_text"]
