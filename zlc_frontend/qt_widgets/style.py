"""Qt-chrome design tokens owned by :mod:`zlc_frontend.qt_widgets`.

This module is intentionally free of Matplotlib and domain/runtime imports.  It
defines the visual vocabulary used by reusable Qt controls and the few
domain-specific QPainter surfaces composed by Workbench windows.
"""

from __future__ import annotations


ACCENT = "#77AADD"
HOVER = "#004578"
BG = "#F3F3F3"
TEXT = "#323130"
HINT = "#F0A150"
PLACEHOLDER = "#A19F9D"
DIVIDER = "#E1DFDD"
GREEN = "#7FC2AD"
RED = "#CD7380"
ORANGE = "#D69A6E"
ORANGE_TINT = "#F6E3D4"
ORANGE_DARK = "#8A4B1F"
API_VIOLET = "#9B86C9"
API_VIOLET_DARK = "#5A4A8A"
YELLOW = "#E5C85B"
GREY = "#A2A2A2"

MUTED_LABEL_STYLE = f"color: {GREY}; background: transparent; border: none;"

# ---- Selector overlay art: the REFERENCE's matplotlib selectors, verbatim ---- #
# The reference (frontend/selectors.py) draws its area/cross selectors as GREY
# SOLID lines at alpha 0.8 with white square handles (legend.fontsize/2 pt), a
# lines.markersize pt crosshair dot, and UNBOXED coordinate labels at
# legend.fontsize pt.  The pixel values below are those point sizes at the
# panel's effective dpi (DESIGN_DPI 300 x PANEL_DISPLAY_SCALE 0.7 = 210 dpi;
# px = pt x 210 / 72).  qt_widgets stays matplotlib-free (charter C12), so a
# contract test pins these literals to the render-style rcParams instead of an
# import -- the two layers cannot drift without a red test.
SELECTOR_COLOR = "#808080"       # matplotlib 'grey'
SELECTOR_ALPHA = 204             # alpha 0.8
SELECTOR_LINE_PX = 2.9           # lines.linewidth   1.0 pt @ 210 dpi
SELECTOR_FONT_PX = 19            # legend.fontsize   6.5 pt @ 210 dpi
SELECTOR_HANDLE_PX = 9           # legend.fontsize/2 3.25 pt @ 210 dpi (square side)
SELECTOR_DOT_PX = 6              # lines.markersize  2.0 pt @ 210 dpi (dot diameter)
RADIUS = 4
CARD_TITLE_PX = 32
CARD_PAD = 10
FONT = "Segoe UI"
FONT_SIZE = 12
PADDING_V = 1
PADDING_H = 1
EDIT_PADDING_H = 4
WINDOW_PAD = 14
TITLE_LEFT_INSET = WINDOW_PAD
COMBO_WIDTH = 16
COMBO_TRI_SIZE = 8
STEP_WIDTH = 6

FLUENT_SCALE_MIN = 0.72
FLUENT_SCALE_MAX = 1.25
AUTO_SCALE_BASIS = (1280, 790)
AUTO_SCALE_MARGIN = (48, 88)
WINDOW_FALLBACK_PX = (1280, 760)
WINDOW_FALLBACK_MIN_PX = (960, 620)
WINDOW_MIN_PX = (980, 640)
WINDOW_MIN_FLOOR_PX = (820, 560)
WINDOW_MARGIN_PX = (40, 48)
WINDOW_MARGIN_FLOOR_PX = (28, 32)
WINDOW_TITLEBAR_PX = 36
WINDOW_TITLEBAR_FLOOR_PX = 28
WINDOW_MAX_FLOOR_PX = (360, 320)
WINDOW_SCREEN_FRACTION = 0.90

# PulseTimelineWidget is a Qt/QPainter surface, not a Matplotlib renderer.
TIMELINE_BACKGROUND = "#FCFCFD"
TIMELINE_EMPTY_TEXT = "#657080"
TIMELINE_TITLE_TEXT = "#354052"
TIMELINE_AXIS_TEXT = "#7A8494"
TIMELINE_PERIOD_BACKGROUNDS = ("#EDF3FF", "#F4F7FB")
TIMELINE_PERIOD_TEXT = "#536078"
TIMELINE_GRID = "#D8DDE6"
TIMELINE_ROW_TEXT = "#253047"
TIMELINE_ACTIVE_FILL_RGBA = (72, 128, 232, 34)
TIMELINE_TRACE = "#2867C7"
TIMELINE_REPEAT = "#8B5CF6"

RASTER_PLACEHOLDER_BACKGROUND = "#111111"
RASTER_PLACEHOLDER_TEXT = "#BBBBBB"

def raster_placeholder_stylesheet() -> str:
    """Return the one Qt placeholder-surface stylesheet."""

    return (
        f"background: {RASTER_PLACEHOLDER_BACKGROUND}; "
        f"color: {RASTER_PLACEHOLDER_TEXT};"
    )


__all__ = [
    "ACCENT",
    "API_VIOLET",
    "API_VIOLET_DARK",
    "BG",
    "CARD_PAD",
    "CARD_TITLE_PX",
    "COMBO_TRI_SIZE",
    "COMBO_WIDTH",
    "DIVIDER",
    "EDIT_PADDING_H",
    "FLUENT_SCALE_MAX",
    "FLUENT_SCALE_MIN",
    "FONT",
    "FONT_SIZE",
    "GREEN",
    "GREY",
    "HINT",
    "HOVER",
    "MUTED_LABEL_STYLE",
    "ORANGE",
    "ORANGE_DARK",
    "ORANGE_TINT",
    "PADDING_H",
    "PADDING_V",
    "PLACEHOLDER",
    "RASTER_PLACEHOLDER_BACKGROUND",
    "RASTER_PLACEHOLDER_TEXT",
    "RADIUS",
    "RED",
    "STEP_WIDTH",
    "TEXT",
    "TIMELINE_ACTIVE_FILL_RGBA",
    "TIMELINE_AXIS_TEXT",
    "TIMELINE_BACKGROUND",
    "TIMELINE_EMPTY_TEXT",
    "TIMELINE_GRID",
    "TIMELINE_PERIOD_BACKGROUNDS",
    "TIMELINE_PERIOD_TEXT",
    "TIMELINE_REPEAT",
    "TIMELINE_ROW_TEXT",
    "TIMELINE_TITLE_TEXT",
    "TIMELINE_TRACE",
    "TITLE_LEFT_INSET",
    "WINDOW_PAD",
    "WINDOW_SCREEN_FRACTION",
    "YELLOW",
    "raster_placeholder_stylesheet",
]
