"""The frontend panel-size vocabulary and its one parser.

A panel size is a spelling both shells and the render layer must agree on -- the
console lays cards out by it, the pulse GUI sizes its preview by it, and the figure
layer turns it into a data region.  The parser is pure string arithmetic over the
preset tuple: no geometry, no style tokens, no imports at all.

This module is headless despite living in :mod:`zlc_frontend`: shells may import the
vocabulary without importing a renderer or Qt.  The rendered SIZE of a panel
(``panel_display_size``) genuinely depends on figure margins and stays with layout;
this module owns only the presentation spelling and validation.
"""

from __future__ import annotations

__all__ = ["PANEL_SIZES", "panel_size_cells"]


PANEL_SIZES = ("1x2", "2x2", "4x2", "1x4", "2x4", "4x4", "4x8", "8x4", "8x8")


def panel_size_cells(size: str) -> tuple[int, int]:
    """Parse a panel size ("rows x cols" in half-units) against the preset list."""

    key = str(size).strip().lower().replace(" ", "")
    if key not in PANEL_SIZES:
        raise ValueError(f"unknown panel size {size!r}; choose from {', '.join(PANEL_SIZES)}.")
    rows, cols = key.split("x")
    return int(rows), int(cols)
