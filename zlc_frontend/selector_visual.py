"""Backend-neutral visual contract for every Figure selector overlay.

Matplotlib and Qt render the same established Main selector art.  This module
owns that art in design units; backends may convert it, but may not restate the
values.  It intentionally contains no Qt or Matplotlib import.
"""

from __future__ import annotations

from .plot_layout import DESIGN_DPI, PANEL_DISPLAY_SCALE
from .typography import FONT_FAMILY


SELECTOR_COLOR = "#808080"       # Matplotlib ``grey``
SELECTOR_ALPHA_FRACTION = 0.8
SELECTOR_LINE_PT = 1.0
SELECTOR_FONT_PT = 6.5
SELECTOR_HANDLE_PT = SELECTOR_FONT_PT / 2.0
SELECTOR_DOT_PT = 2.0
SELECTOR_FONT_FAMILY = FONT_FAMILY


def selector_pixels(points: float) -> float:
    """Convert selector points at the established live-panel effective DPI."""

    return float(points) * float(DESIGN_DPI) * float(PANEL_DISPLAY_SCALE) / 72.0


SELECTOR_ALPHA = round(SELECTOR_ALPHA_FRACTION * 255)
SELECTOR_LINE_PX = selector_pixels(SELECTOR_LINE_PT)
SELECTOR_FONT_PX = round(selector_pixels(SELECTOR_FONT_PT))
SELECTOR_HANDLE_PX = round(selector_pixels(SELECTOR_HANDLE_PT), 1)
SELECTOR_DOT_PX = round(selector_pixels(SELECTOR_DOT_PT))


__all__ = [
    "SELECTOR_ALPHA",
    "SELECTOR_ALPHA_FRACTION",
    "SELECTOR_COLOR",
    "SELECTOR_DOT_PT",
    "SELECTOR_DOT_PX",
    "SELECTOR_FONT_FAMILY",
    "SELECTOR_FONT_PT",
    "SELECTOR_FONT_PX",
    "SELECTOR_HANDLE_PT",
    "SELECTOR_HANDLE_PX",
    "SELECTOR_LINE_PT",
    "SELECTOR_LINE_PX",
    "selector_pixels",
]
