"""Pure image-to-indexed-raster path for high-rate live display."""

from __future__ import annotations

import numpy as np

from zlc_storage import nonnegative_integer, positive_integer

from .figure import EvaluatedImage
from .image_display import (
    ImageDisplayState,
    ImageRelimMode,
    deadband_image_color_limits,
    validated_image_range,
)
from .render import PixelFormat, RasterBuffer


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DISPLAY_COORDINATE_ENTRY_BYTES = 128
_INDEXED8_VALID_CODE_SPAN = 254


def indexed8_code_for_value(
    value: float,
    color_limits: tuple[float, float],
) -> int:
    """Return the exact valid INDEXED8 code used by the raster worker."""

    low, high = color_limits
    if value <= low:
        return 1
    if value >= high:
        return 255
    return 1 + int((value - low) / (high - low) * _INDEXED8_VALID_CODE_SPAN)


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


def estimate_indexed8_raster_peak_nbytes(
    height: int,
    width: int,
    *,
    value_itemsize: int,
    retained_fronts: int = 1,
    retained_sample_fronts: int = 0,
) -> int:
    """Bound indexed-raster scratch plus retained raster/sample fronts.

    The candidate :class:`EvaluatedImage` remains part of the caller's view
    evaluation budget.  This function adds the finite mask, stable float64
    normalization workspace, uint8 workspace, returned immutable bytes,
    compact code histogram, ``retained_fronts`` presenter/interaction rasters,
    and ``retained_sample_fronts`` older exact value+validity planes.  Every
    retained indexed GUI front counts both its immutable RasterBuffer bytes and
    the private pixel plane Qt creates when ``QImage.setColorTable()`` detaches.
    A live interaction normally retains the latest board front plus the older
    target panel held under the pointer.
    """

    height = positive_integer(height, "height")
    width = positive_integer(width, "width")
    value_itemsize = positive_integer(value_itemsize, "value_itemsize")
    retained_fronts = nonnegative_integer(retained_fronts, "retained_fronts")
    retained_sample_fronts = nonnegative_integer(
        retained_sample_fronts,
        "retained_sample_fronts",
    )
    pixels = height * width
    scratch_and_result = pixels * (
        np.dtype(bool).itemsize
        + np.dtype(np.float64).itemsize
        + 2 * np.dtype(np.uint8).itemsize
    )
    retained_rasters = (
        2 * retained_fronts * pixels * np.dtype(np.uint8).itemsize
    )
    retained_samples = retained_sample_fronts * (
        pixels * (value_itemsize + np.dtype(bool).itemsize)
        + _DISPLAY_COORDINATE_ENTRY_BYTES * (height + width)
    )
    # bincount(0..255), the immutable tuple copy, and one 256-entry QRgb LUT
    # are fixed-size next to a multi-megapixel frame.  Keep an explicit 64 KiB
    # allowance for the overlapping NumPy counts, Python tuples/ints and LUT
    # objects rather than hiding them in a per-pixel multiplier.
    indexed_metadata = 64 * 1024
    return scratch_and_result + retained_rasters + retained_samples + indexed_metadata


def rasterize_image_indexed8(
    image: EvaluatedImage,
    state: ImageDisplayState,
    *,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: ImageRelimMode | None,
) -> tuple[
    RasterBuffer,
    tuple[float, float] | None,
    tuple[int, ...],
    tuple[float, float],
]:
    """Rasterize directly in the effective authored colour window.

    Quantizing the full camera range and narrowing it later through a Qt LUT
    cannot recover the levels discarded by the first 8-bit conversion.  This
    pure worker-side path resolves the dead-banded limits first and maps all
    255 valid codes directly across that range; exact samples remain in the
    separately retained :class:`EvaluatedImage`.
    """

    if not isinstance(state, ImageDisplayState):
        raise TypeError("state must be ImageDisplayState")
    if previous_relim_mode is not None and not isinstance(
        previous_relim_mode,
        ImageRelimMode,
    ):
        raise TypeError("previous_relim_mode must be ImageRelimMode or None")
    if not isinstance(image, EvaluatedImage):
        raise TypeError("image must be EvaluatedImage")
    values = image.values
    if values.dtype.kind == "c":
        raise TypeError("complex images require an explicit display transform")

    valid = np.empty(values.shape, dtype=bool)
    np.isfinite(values, out=valid)
    np.logical_and(valid, image.validity, out=valid)
    pixels = np.zeros(values.shape, dtype=np.uint8)
    value_range: tuple[float, float] | None = None
    valid_count = int(np.count_nonzero(valid))
    if valid_count:
        low, high = _automatic_range(values, valid)
        value_range = (low, high)
    if value_range is None:
        if (
            state.relim_mode is ImageRelimMode.FIXED
            and state.fixed_color_limits is not None
        ):
            color_limits = state.fixed_color_limits
        elif current_color_limits is not None:
            color_limits = validated_image_range(
                current_color_limits,
                "current_color_limits",
            )
        else:
            color_limits = (0.0, 1.0)
    else:
        color_limits = deadband_image_color_limits(
            state,
            current_color_limits,
            value_range[0],
            value_range[1],
            force=(
                previous_relim_mode is None
                or previous_relim_mode is not state.relim_mode
            ),
        )

    if valid_count:
        map_low, map_high = color_limits
        # Own exactly one float64 workspace.  np.interp would allocate a second
        # full float64 input conversion for uint16/float32 camera planes,
        # invalidating the pre-arm peak-memory bound.
        scaled = np.empty(values.shape, dtype=np.float64)
        np.copyto(scaled, values, casting="unsafe")
        span = map_high - map_low
        with np.errstate(over="ignore", invalid="ignore"):
            np.subtract(scaled, map_low, out=scaled)
            np.divide(scaled, span, out=scaled)
            np.multiply(scaled, float(_INDEXED8_VALID_CODE_SPAN), out=scaled)
            np.add(scaled, 1.0, out=scaled)
            np.clip(scaled, 1.0, 255.0, out=scaled)
        np.copyto(pixels, scaled, where=valid, casting="unsafe")
        # np.bincount converts uint8 to an intp-sized full plane.  End the
        # float64 workspace lifetime first so those two 8N allocations cannot
        # overlap.
        del scaled

        # Match main's distribution band: the IMAGE saturates outside clim,
        # while the histogram excludes those samples rather than inventing
        # false peaks exactly at the low/high handles.
        np.greater_equal(values, map_low, out=valid, where=valid)
        np.less_equal(values, map_high, out=valid, where=valid)

    histogram_codes = pixels[valid]
    histogram_valid_count = int(histogram_codes.size)
    counts = np.bincount(histogram_codes, minlength=256)
    del histogram_codes
    histogram = tuple(int(value) for value in counts[1:256])
    if sum(histogram) != histogram_valid_count:
        raise RuntimeError(
            "indexed image histogram differs from its in-range valid samples"
        )
    height, width = values.shape
    raster = RasterBuffer(
        width=width,
        height=height,
        stride_bytes=width,
        pixel_format=PixelFormat.INDEXED8,
        pixels=pixels.tobytes(order="C"),
    )
    return raster, value_range, histogram, color_limits


def _automatic_range(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    if values.dtype.kind in "iu":
        info = np.iinfo(values.dtype)
        native_low = np.min(values, where=valid, initial=info.max)
        native_high = np.max(values, where=valid, initial=info.min)
        low, high = float(native_low), float(native_high)
        if native_low != native_high and low == high:
            raise TypeError(
                "integer image range is not distinguishable in the float64 "
                "display contract; apply an explicit display transform"
            )
    elif values.dtype.kind == "b":
        low = np.min(values, where=valid, initial=True)
        high = np.max(values, where=valid, initial=False)
    else:
        low = np.min(values, where=valid, initial=np.inf)
        high = np.max(values, where=valid, initial=-np.inf)
    return float(low), float(high)


__all__ = [
    "estimate_encoded_png_front_peak_nbytes",
    "estimate_indexed8_raster_peak_nbytes",
    "indexed8_code_for_value",
    "png_raster_size",
    "rasterize_image_indexed8",
]
