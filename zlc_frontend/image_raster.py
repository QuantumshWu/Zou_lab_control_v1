"""Pure image-to-indexed-raster path for high-rate live display."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .display_range import (
    RelimMode,
    deadband_display_range,
    validated_display_range,
)
from .figure import EvaluatedImage
from .image_display import ImageDisplayState
from .render import PixelFormat, RasterBuffer


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_INDEXED8_VALID_CODE_SPAN = 254


def evaluated_image_data_range(
    images: Iterable[EvaluatedImage],
) -> tuple[float, float] | None:
    """Pool finite, valid IMAGE samples without concatenating their planes.

    This is the toolkit-neutral data-range primitive shared by indexed live
    rasterization and canonical multi-panel renderers.  Invalid components and
    non-finite floating values cannot influence a display range.
    """

    low = high = None
    for image in images:
        if not isinstance(image, EvaluatedImage):
            raise TypeError("images must contain only EvaluatedImage values")
        values = image.values
        if values.dtype.kind not in "biuf":
            raise TypeError("image display values must be real numeric arrays")
        if values.dtype.kind in "biu":
            valid = image.validity
        else:
            valid = np.empty(values.shape, dtype=bool)
            np.isfinite(values, out=valid)
            np.logical_and(valid, image.validity, out=valid)
        if not bool(np.any(valid)):
            continue
        cell_low, cell_high = _automatic_range(values, valid)
        low = cell_low if low is None else min(low, cell_low)
        high = cell_high if high is None else max(high, cell_high)
    return None if low is None else (low, high)


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


def rasterize_image_indexed8(
    image: EvaluatedImage,
    state: ImageDisplayState,
    *,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
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
        RelimMode,
    ):
        raise TypeError("previous_relim_mode must be RelimMode or None")
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
            state.relim_mode is RelimMode.FIXED
            and state.fixed_color_limits is not None
        ):
            color_limits = state.fixed_color_limits
        elif current_color_limits is not None:
            color_limits = validated_display_range(
                current_color_limits,
                "current_color_limits",
            )
        else:
            color_limits = (0.0, 1.0)
    else:
        color_limits = deadband_display_range(
            state.relim_mode,
            current_color_limits,
            value_range[0],
            value_range[1],
            fixed_range=state.fixed_color_limits,
            force=(
                previous_relim_mode is None
                or previous_relim_mode is not state.relim_mode
            ),
        )

    if valid_count:
        map_low, map_high = color_limits
        # Own one float64 workspace.  np.interp would allocate another full
        # conversion for uint16/float32 camera planes without improving output.
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
    "evaluated_image_data_range",
    "indexed8_code_for_value",
    "png_raster_size",
    "rasterize_image_indexed8",
]
