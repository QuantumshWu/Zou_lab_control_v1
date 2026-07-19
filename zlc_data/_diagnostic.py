"""Bounded diagnostics for exact integer authority values."""

from __future__ import annotations

from numbers import Integral


def bounded_integer_diagnostic(value: object) -> str:
    """Format an integer without triggering unbounded decimal conversion."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        return f"<{type(value).__name__}>"
    integer = int(value)
    bits = integer.bit_length()
    if bits > 4096:
        sign = "negative " if integer < 0 else ""
        return f"<{sign}integer; {bits} bits>"
    return str(integer)


def bounded_index_tuple_diagnostic(values: tuple[object, ...]) -> str:
    """Format one logical index tuple with bounded per-component labels."""

    labels = tuple(bounded_integer_diagnostic(value) for value in values)
    if len(labels) == 1:
        return f"({labels[0]},)"
    return f"({', '.join(labels)})"


__all__ = ["bounded_index_tuple_diagnostic", "bounded_integer_diagnostic"]
