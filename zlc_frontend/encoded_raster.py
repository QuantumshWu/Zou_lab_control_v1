"""Immutable encoded-raster document crossing worker/presenter boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from zlc_storage import canonical_text

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_raster_size(payload: bytes) -> tuple[int, int]:
    """Validate one owned PNG front and return its declared pixel geometry."""

    if not isinstance(payload, bytes):
        raise TypeError("PNG payload must be owned immutable bytes")
    if (
        not payload.startswith(_PNG_SIGNATURE)
        or len(payload) < 24
        or payload[12:16] != b"IHDR"
    ):
        raise ValueError("payload must contain a PNG raster with an IHDR")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    if width <= 0 or height <= 0:
        raise ValueError("PNG raster dimensions must be positive")
    return width, height


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


def encode_raster_buffer_png(raster) -> bytes:
    """Encode one frontend-owned RGBA raster without reimplementing save paths.

    Worker renderers deliberately hand immutable :class:`RasterBuffer` values
    to their presenters.  Reports and other headless consumers that need a PNG
    must use this one encoder instead of acquiring Matplotlib Figures or
    copying the Pillow bridge into each product package.
    """

    from .render import RasterBuffer

    if not isinstance(raster, RasterBuffer):
        raise TypeError("raster must be RasterBuffer")
    from PIL import Image

    target = BytesIO()
    Image.frombytes(
        "RGBA",
        (raster.width, raster.height),
        raster.pixels,
    ).save(target, format="PNG")
    payload = target.getvalue()
    if png_raster_size(payload) != (raster.width, raster.height):
        raise RuntimeError("encoded PNG geometry differs from its RasterBuffer")
    return payload


__all__ = [
    "EncodedRasterDocument",
    "EncodedRasterPage",
    "encode_raster_buffer_png",
    "png_raster_size",
]
