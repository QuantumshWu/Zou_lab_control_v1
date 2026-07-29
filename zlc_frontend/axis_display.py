"""Small, renderer-neutral labels for declared axes and scalar coordinates."""

from __future__ import annotations

def axis_label(axis) -> str:
    """Return main's one public ``name (unit)`` spelling."""

    name = str(axis.name)
    unit = axis.unit
    if unit is None:
        return name
    # Storage uses the canonical physical spelling; authored plot chrome uses
    # only main's established compact alias for pixel.  This is a vocabulary
    # rule, not a per-renderer font workaround.
    visible_unit = {
        "pixel": "px",
    }.get(str(unit), str(unit))
    return f"{name} ({visible_unit})"
__all__ = ["axis_label"]
