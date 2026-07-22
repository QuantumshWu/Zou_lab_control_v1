"""One pure presentation policy for the formal pulse repeat wording/brackets.

The pulse domain owns the authored periods and optional finite inner bracket.
The formal editor presents that compact table as a continuously repeating
frame.  This module derives only ephemeral UI facts from period positions; it
is neither an authoring model nor a persistence format.
"""

from __future__ import annotations


def pulse_repeat_presentation(
    period_count: int,
    repeat: tuple[int, int, int] | None,
) -> tuple[str, tuple[tuple[int, int, str], ...]]:
    """Return ``(status wording, bracket index spans)`` for the formal UI.

    Bracket ends are inclusive period indexes.  The whole authored frame is an
    infinite outer loop unless a finite bracket already covers that complete
    frame; a partial finite bracket is therefore nested inside ``×∞``.
    """

    if isinstance(period_count, bool) or not isinstance(period_count, int):
        raise TypeError("period_count must be an integer")
    if period_count <= 0:
        raise ValueError("repeat presentation requires at least one period")

    outer = (0, period_count - 1, "×∞")
    if repeat is None:
        return "repeat ∞", (outer,)

    start, end, count = repeat
    if any(isinstance(value, bool) or not isinstance(value, int) for value in repeat):
        raise TypeError("repeat indexes/count must be integers")
    if start < 0 or end < start or end >= period_count:
        raise ValueError("repeat span is outside the authored period table")
    if count <= 0:
        raise ValueError("repeat count must be positive")

    inner_text = f"P{start + 1}-P{end + 1} x{count}"
    inner = (start, end, f"×{count}")
    if start == 0 and end == period_count - 1:
        return f"repeat {inner_text}", (inner,)
    return f"repeat ∞ + {inner_text}", (outer, inner)


__all__ = ["pulse_repeat_presentation"]
