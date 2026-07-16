"""Pure image-to-raster path for high-rate live display."""

from __future__ import annotations

import numpy as np

from zlc_storage import positive_integer

from .figure import EvaluatedImage
from .render import PixelFormat, RasterBuffer


def estimate_gray8_raster_peak_nbytes(height: int, width: int) -> int:
    """Bound raster scratch, returned bytes, and one previous live front.

    The caller-owned :class:`EvaluatedImage` is deliberately excluded.  The
    five terms are a finite mask, float32 normalization workspace, uint8
    workspace, returned immutable bytes, and the prior presenter front.
    """

    height = positive_integer(height, "height")
    width = positive_integer(width, "width")
    pixels = height * width
    return pixels * (
        np.dtype(bool).itemsize
        + np.dtype(np.float32).itemsize
        + 3 * np.dtype(np.uint8).itemsize
    )


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


__all__ = ["estimate_gray8_raster_peak_nbytes", "rasterize_image_gray8"]
