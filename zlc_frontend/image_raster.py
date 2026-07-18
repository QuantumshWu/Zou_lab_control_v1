"""Pure image-to-raster path for high-rate live display."""

from __future__ import annotations

import numpy as np

from zlc_storage import nonnegative_integer, positive_integer

from .figure import EvaluatedImage
from .render import PixelFormat, RasterBuffer


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


def estimate_encoded_png_front_peak_nbytes(
    payload: bytes,
    *,
    presentation_size: tuple[int, int] | None = None,
) -> int:
    """Bound encoded bytes plus source and physical presentation RGBA fronts."""

    width, height = png_raster_size(payload)
    if presentation_size is None:
        presentation_width, presentation_height = width, height
    else:
        if not isinstance(presentation_size, tuple) or len(presentation_size) != 2:
            raise TypeError("presentation_size must be a (width, height) tuple")
        presentation_width = positive_integer(
            presentation_size[0],
            "presentation width",
        )
        presentation_height = positive_integer(
            presentation_size[1],
            "presentation height",
        )
    return (
        len(payload)
        + width * height * 4
        + presentation_width * presentation_height * 4
    )


def estimate_gray8_raster_peak_nbytes(
    height: int,
    width: int,
    *,
    retained_fronts: int = 1,
) -> int:
    """Bound raster scratch, returned bytes, and retained live fronts.

    The caller-owned :class:`EvaluatedImage` is deliberately excluded.  The
    terms are a finite mask, float32 normalization workspace, uint8 workspace,
    returned immutable bytes, and ``retained_fronts`` presenter/interaction
    fronts.  A live rectangle hold uses two: the latest board front plus the
    older target-panel raster kept visible under the pointer.
    """

    height = positive_integer(height, "height")
    width = positive_integer(width, "width")
    retained_fronts = nonnegative_integer(retained_fronts, "retained_fronts")
    pixels = height * width
    scratch_and_result = pixels * (
        np.dtype(bool).itemsize
        + np.dtype(np.float32).itemsize
        + 2 * np.dtype(np.uint8).itemsize
    )
    return scratch_and_result + retained_fronts * pixels * np.dtype(np.uint8).itemsize


def rasterize_image_gray8(
    image: EvaluatedImage,
) -> RasterBuffer:
    """Map one evaluated ``(Y, X)`` image to owned GRAY8 without reorientation."""

    if not isinstance(image, EvaluatedImage):
        raise TypeError("image must be EvaluatedImage")
    values = image.values
    if values.dtype.kind == "c":
        raise TypeError("complex images require an explicit display transform")

    valid = np.empty(values.shape, dtype=bool)
    np.isfinite(values, out=valid)
    np.logical_and(valid, image.validity, out=valid)
    pixels = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = _automatic_range(values, valid)
        if high == low:
            np.copyto(pixels, 128, where=valid)
        else:
            scaled = np.empty(values.shape, dtype=np.float32)
            np.subtract(values, low, out=scaled, casting="unsafe")
            np.multiply(scaled, 254.0 / (high - low), out=scaled)
            np.add(scaled, 1.0, out=scaled)
            np.clip(scaled, 1.0, 255.0, out=scaled)
            np.copyto(pixels, scaled, where=valid, casting="unsafe")

    height, width = values.shape
    return RasterBuffer(
        width=width,
        height=height,
        stride_bytes=width,
        pixel_format=PixelFormat.GRAY8,
        pixels=pixels.tobytes(order="C"),
    )


def _automatic_range(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    if values.dtype.kind in "iu":
        info = np.iinfo(values.dtype)
        low = np.min(values, where=valid, initial=info.max)
        high = np.max(values, where=valid, initial=info.min)
    elif values.dtype.kind == "b":
        low = np.min(values, where=valid, initial=True)
        high = np.max(values, where=valid, initial=False)
    else:
        low = np.min(values, where=valid, initial=np.inf)
        high = np.max(values, where=valid, initial=-np.inf)
    return float(low), float(high)


__all__ = [
    "estimate_encoded_png_front_peak_nbytes",
    "estimate_gray8_raster_peak_nbytes",
    "png_raster_size",
    "rasterize_image_gray8",
]
