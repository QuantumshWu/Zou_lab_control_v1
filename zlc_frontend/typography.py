"""Pure font identity shared by the Matplotlib and QPainter plot backends."""

from __future__ import annotations

from pathlib import Path


FONT_FAMILY = "Helvetica Light"
FONT_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "helvetica-light-587ebe5a59211.ttf"
)
SANS_SERIF = (FONT_FAMILY, "Arial")


__all__ = ["FONT_FAMILY", "FONT_PATH", "SANS_SERIF"]
