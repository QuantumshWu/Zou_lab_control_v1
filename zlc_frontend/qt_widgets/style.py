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
GRAPHITE = "#808080"
SURFACE = "#FFFFFF"

MUTED_LABEL_STYLE = f"color: {GREY}; background: transparent; border: none;"

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
    "GRAPHITE",
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
    "RADIUS",
    "RED",
    "STEP_WIDTH",
    "SURFACE",
    "TEXT",
    "TITLE_LEFT_INSET",
    "WINDOW_PAD",
    "WINDOW_SCREEN_FRACTION",
    "YELLOW",
]
