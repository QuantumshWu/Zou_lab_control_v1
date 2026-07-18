"""Shared pure range policy for authored live-panel displays.

The three relim modes and their hysteresis are value semantics shared by image
colour limits and curve y limits.  This module deliberately knows neither
panel kind nor rendering toolkit; callers supply the authored fixed range.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import math
from typing import TypeAlias

from zlc_storage import finite_real


DisplayRange: TypeAlias = tuple[float, float]


class RelimMode(str, Enum):
    """Closed range behaviours supported by live numeric displays."""

    TIGHT = "tight"
    NORMAL = "normal"
    FIXED = "fixed"


def validated_display_range(
    value: object,
    field: str,
    *,
    allow_degenerate: bool = False,
) -> DisplayRange:
    """Return one canonical finite ordered two-endpoint range."""

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{field} must be a two-item tuple")
    low = finite_real(value[0], f"{field} low")
    high = finite_real(value[1], f"{field} high")
    if low > high or (low == high and not allow_degenerate):
        relation = "cannot exceed" if allow_degenerate else "must be below"
        raise ValueError(f"{field} low {relation} high")
    if not math.isfinite(high - low):
        raise ValueError(f"{field} span must be finite")
    return low, high


def optional_display_range(value: object, field: str) -> DisplayRange | None:
    """Validate an optional authored range without assigning panel semantics."""

    if value is None:
        return None
    return validated_display_range(value, field)


def display_range_form_values(
    value: DisplayRange | None,
) -> tuple[float | None, float | None]:
    """Project one optional range into a form's two scalar fields."""

    return (None, None) if value is None else value


def display_range_from_form(
    values: Mapping[str, object],
    low_key: str,
    high_key: str,
    field: str,
) -> DisplayRange | None:
    """Read one all-or-none range from an already exact form mapping."""

    if not isinstance(values, Mapping):
        raise TypeError("display range form values must be a mapping")
    low = values[low_key]
    high = values[high_key]
    if low is None and high is None:
        return None
    if low is None or high is None:
        raise ValueError(f"{field} requires both minimum and maximum")
    return validated_display_range((low, high), field)


def _data_range(data_min: object, data_max: object) -> DisplayRange:
    low = finite_real(data_min, "data minimum")
    high = finite_real(data_max, "data maximum")
    if low > high:
        raise ValueError("data minimum cannot exceed data maximum")
    return low, high


def target_display_range(
    mode: RelimMode,
    data_min: object,
    data_max: object,
    *,
    fixed_range: DisplayRange | None = None,
) -> DisplayRange:
    """Derive the established tight/normal/fixed target range.

    ``NORMAL`` anchors non-negative data at zero with 20% headroom.  Negative
    data deliberately falls back to the symmetric tight policy without
    mutating the authored mode.  Constant data always receives a deterministic
    non-degenerate window.
    """

    if not isinstance(mode, RelimMode):
        raise TypeError("mode must be RelimMode")
    if mode is RelimMode.FIXED:
        if fixed_range is None:
            raise ValueError("fixed relim mode requires fixed_range")
        return validated_display_range(fixed_range, "fixed_range")

    low, high = _data_range(data_min, data_max)
    span = (high - low) or (abs(high) or 1.0)
    if not math.isfinite(span):
        raise ValueError("data range span must be finite")
    if mode is RelimMode.NORMAL and low >= 0.0:
        target = (0.0, high * 1.2 if high else 1.0)
    else:
        target = (low - 0.1 * span, high + 0.1 * span)
    return validated_display_range(target, "derived display range")


def deadband_display_range(
    mode: RelimMode,
    current_range: DisplayRange | None,
    data_min: object,
    data_max: object,
    *,
    fixed_range: DisplayRange | None = None,
    force: bool = False,
) -> DisplayRange:
    """Return current or target limits using the mature live-view hysteresis.

    Growth is immediate whenever data would clip.  ``NORMAL`` holds while its
    high value occupies 70--100% of the current ceiling.  ``TIGHT`` holds while
    no value clips and data has not vacated more than 35% at either side.
    ``force`` bypasses both dead bands after an authored mode change.
    """

    if not isinstance(mode, RelimMode):
        raise TypeError("mode must be RelimMode")
    if not isinstance(force, bool):
        raise TypeError("force must be bool")
    if mode is RelimMode.FIXED:
        return target_display_range(
            mode,
            data_min,
            data_max,
            fixed_range=fixed_range,
        )

    low, high = _data_range(data_min, data_max)
    target = target_display_range(mode, low, high)
    if current_range is None or force:
        return target
    current_low, current_high = validated_display_range(
        current_range,
        "current_range",
    )

    effective_normal = mode is RelimMode.NORMAL and low >= 0.0
    if effective_normal:
        if (
            current_low == 0.0
            and current_high > 0.0
            and 0.7 * current_high <= high <= current_high
        ):
            return current_low, current_high
        return target

    span = current_high - current_low
    clips = low < current_low or high > current_high
    too_empty = (
        high < current_high - 0.35 * span
        or low > current_low + 0.35 * span
    )
    if not clips and not too_empty:
        return current_low, current_high
    return target


__all__ = [
    "DisplayRange",
    "RelimMode",
    "deadband_display_range",
    "display_range_form_values",
    "display_range_from_form",
    "optional_display_range",
    "target_display_range",
    "validated_display_range",
]
