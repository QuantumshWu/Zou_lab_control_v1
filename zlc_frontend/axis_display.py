"""Small, renderer-neutral labels for declared axes and scalar coordinates."""

from __future__ import annotations

import math
from numbers import Real


def axis_label(axis) -> str:
    """Return main's one public ``name (unit)`` spelling."""

    name = str(axis.name)
    unit = axis.unit
    if unit is None:
        return name
    # Storage uses the canonical physical spelling; Main's authored plot
    # chrome uses the established compact display spelling.
    # Helvetica Light does not contain U+0393.  Keep the canonical data unit
    # ``Γ`` untouched, but render its established scientific symbol through
    # mathtext so raster/vector outputs cannot silently substitute tofu.
    visible_unit = {
        "pixel": "px",
        "Γ": r"$\Gamma$",
    }.get(str(unit), str(unit))
    return f"{name} ({visible_unit})"


def compact_coordinate_label(value: object) -> str:
    """Format a declared coordinate without changing its physical value."""

    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            if number == 0.0:
                number = 0.0
            return f"{number:.6g}"
    return str(value)


__all__ = ["axis_label", "compact_coordinate_label"]
