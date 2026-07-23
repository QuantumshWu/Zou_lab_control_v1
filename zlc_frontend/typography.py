"""Pure typography identity shared by render and Qt presentation owners."""

from __future__ import annotations

from pathlib import Path


FONT_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "helvetica-light-587ebe5a59211.ttf"
)
FONT_FAMILY = "Helvetica Light"
SANS_SERIF = (FONT_FAMILY, "Arial")


__all__ = ["FONT_FAMILY", "FONT_PATH", "SANS_SERIF"]
