"""Immutable encoded-raster document crossing worker/presenter boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text

from .image_raster import (
    estimate_encoded_png_front_peak_nbytes,
    png_raster_size,
)


@dataclass(frozen=True, slots=True)
class EncodedRasterPage:
    key: str
    title: str
    png_bytes: bytes

    def __post_init__(self) -> None:
        canonical_text(self.key, "raster page key")
        canonical_text(self.title, "raster page title")
        if not isinstance(self.png_bytes, bytes):
            raise TypeError("raster page payload must be owned immutable bytes")
        png_raster_size(self.png_bytes)


@dataclass(frozen=True, slots=True)
class EncodedRasterDocument:
    summary: str
    pages: tuple[EncodedRasterPage, ...]

    def __post_init__(self) -> None:
        canonical_text(self.summary, "raster summary")
        pages = tuple(self.pages)
        if not pages or any(not isinstance(page, EncodedRasterPage) for page in pages):
            raise TypeError("raster document must contain EncodedRasterPage values")
        if len({page.key for page in pages}) != len(pages):
            raise ValueError("raster page keys must be unique")
        object.__setattr__(self, "pages", pages)

    @property
    def source_front_peak_nbytes(self) -> int:
        return sum(
            estimate_encoded_png_front_peak_nbytes(page.png_bytes)
            for page in self.pages
        )


__all__ = ["EncodedRasterDocument", "EncodedRasterPage"]
