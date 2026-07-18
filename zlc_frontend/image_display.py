"""Pure authored display state and colour-range policy for pixel images.

This module owns no Qt, Matplotlib, raster storage, or data-projection intent.
It projects one immutable image-display value into the shared headless form
contract and implements the established tight/normal/fixed range hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import TypeAlias

from zlc_storage import exact_mapping, finite_real, nonnegative_integer

from .form import FormChoice, FormFieldProps, FormSpec
from .image_view import ImageViewportTransform


ImageRange: TypeAlias = tuple[float, float]


class ImageRelimMode(str, Enum):
    """Closed colour-limit behaviours supported by a pixel image."""

    TIGHT = "tight"
    NORMAL = "normal"
    FIXED = "fixed"


class ImageColormap(str, Enum):
    """Closed colormap names sampled by the optional render owner."""

    INFERNO = "inferno"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    GRAY = "gray"
    COOLWARM = "coolwarm"


def validated_image_range(
    value: object,
    field: str,
    *,
    allow_degenerate: bool = False,
) -> ImageRange:
    """Return the canonical finite two-endpoint IMAGE range."""

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


def _optional_finite_range(value: object, field: str) -> ImageRange | None:
    if value is None:
        return None
    return validated_image_range(value, field)


@dataclass(frozen=True, slots=True)
class ImageDisplayState:
    """One authored image-panel display value.

    Dynamic effective limits are deliberately absent: they describe a rendered
    front, not authored state, and therefore never advance ``revision``.
    """

    revision: int = 0
    relim_mode: ImageRelimMode = ImageRelimMode.TIGHT
    colormap: ImageColormap = ImageColormap.GRAY
    x_view: ImageRange | None = None
    y_view: ImageRange | None = None
    fixed_color_limits: ImageRange | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision",
            nonnegative_integer(self.revision, "image display revision"),
        )
        if not isinstance(self.relim_mode, ImageRelimMode):
            raise TypeError("relim_mode must be ImageRelimMode")
        if not isinstance(self.colormap, ImageColormap):
            raise TypeError("colormap must be ImageColormap")
        for field in ("x_view", "y_view", "fixed_color_limits"):
            object.__setattr__(
                self,
                field,
                _optional_finite_range(getattr(self, field), field),
            )
        if (
            self.relim_mode is ImageRelimMode.FIXED
            and self.fixed_color_limits is None
        ):
            raise ValueError("fixed relim_mode requires fixed_color_limits")


def image_viewport_for_display_state(
    state: ImageDisplayState,
    reference: ImageViewportTransform,
) -> ImageViewportTransform:
    """Return the one transform represented by authored x/y view pins.

    ``state.revision`` is the presentation revision; the function never
    invents another counter.  Equal revisions must already describe the same
    bounds, while a newer state reuses only the reference axes.
    """

    if not isinstance(state, ImageDisplayState):
        raise TypeError("state must be ImageDisplayState")
    if not isinstance(reference, ImageViewportTransform):
        raise TypeError("reference must be ImageViewportTransform")
    expected = reference.normalized_bounds_for_optional_coordinate_views(
        state.x_view,
        state.y_view,
    )
    if state.revision < reference.viewport_revision:
        raise ValueError("image display revision cannot precede its viewport")
    if state.revision == reference.viewport_revision:
        if any(
            not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-12)
            for actual, wanted in zip(
                reference.visible_bounds,
                expected,
                strict=True,
            )
        ):
            raise ValueError(
                "image display coordinate views differ from viewport bounds"
            )
        return reference
    return ImageViewportTransform(reference.axes, state.revision, expected)


def image_display_for_viewport(
    base: ImageDisplayState,
    viewport: ImageViewportTransform,
) -> ImageDisplayState:
    """Commit one gesture transform into the same authored display value."""

    if not isinstance(base, ImageDisplayState):
        raise TypeError("base must be ImageDisplayState")
    if not isinstance(viewport, ImageViewportTransform):
        raise TypeError("viewport must be ImageViewportTransform")
    x_view, y_view = viewport.optional_coordinate_views_for_normalized_bounds()
    if viewport.viewport_revision < base.revision:
        raise ValueError("viewport revision cannot precede image display state")
    candidate = replace(
        base,
        revision=viewport.viewport_revision,
        x_view=x_view,
        y_view=y_view,
    )
    if viewport.viewport_revision == base.revision:
        if candidate != base:
            raise ValueError(
                "viewport changed coordinate views without advancing revision"
            )
        return base
    return candidate


_IMAGE_DISPLAY_FORM = FormSpec(
    (
        FormFieldProps(
            "relim_mode",
            "choice",
            "Color limits",
            default=ImageRelimMode.TIGHT,
            choices=tuple(
                FormChoice(mode.value.title(), mode) for mode in ImageRelimMode
            ),
        ),
        FormFieldProps(
            "colormap",
            "choice",
            "Colormap",
            default=ImageColormap.GRAY,
            choices=tuple(
                FormChoice(colormap.value, colormap) for colormap in ImageColormap
            ),
        ),
        FormFieldProps("x_min", "float", "X minimum", default=None),
        FormFieldProps("x_max", "float", "X maximum", default=None),
        FormFieldProps("y_min", "float", "Y minimum", default=None),
        FormFieldProps("y_max", "float", "Y maximum", default=None),
        FormFieldProps("color_min", "float", "Color minimum", default=None),
        FormFieldProps("color_max", "float", "Color maximum", default=None),
    )
)


def image_display_form_spec() -> FormSpec:
    """Return the one immutable Setting/Edit form projection."""

    return _IMAGE_DISPLAY_FORM


def _range_form_values(value: ImageRange | None) -> tuple[float | None, float | None]:
    return (None, None) if value is None else value


def image_display_form_values(state: ImageDisplayState) -> dict[str, object]:
    """Project authored state to the form's exact typed keys."""

    if not isinstance(state, ImageDisplayState):
        raise TypeError("state must be ImageDisplayState")
    x_min, x_max = _range_form_values(state.x_view)
    y_min, y_max = _range_form_values(state.y_view)
    color_min, color_max = _range_form_values(state.fixed_color_limits)
    return {
        "relim_mode": state.relim_mode,
        "colormap": state.colormap,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "color_min": color_min,
        "color_max": color_max,
    }


def _range_from_form(
    values: dict[str, object],
    low_key: str,
    high_key: str,
    field: str,
) -> ImageRange | None:
    low = values[low_key]
    high = values[high_key]
    if low is None and high is None:
        return None
    if low is None or high is None:
        raise ValueError(f"{field} requires both minimum and maximum")
    return validated_image_range((low, high), field)


def image_display_from_form(
    base: ImageDisplayState,
    values: dict[str, object],
    *,
    current_color_limits: ImageRange | None = None,
) -> ImageDisplayState:
    """Apply one exact Setting/Edit projection to ``base``.

    Entering FIXED freezes the effective limits visible on the submitted front,
    even if an older dormant fixed range remains in the form.  Later edits while
    already FIXED consume the explicit color fields.  A semantic no-op returns
    ``base`` unchanged; every real authored change advances its revision once.
    """

    if not isinstance(base, ImageDisplayState):
        raise TypeError("base must be ImageDisplayState")
    values = exact_mapping(
        values,
        frozenset(_IMAGE_DISPLAY_FORM.keys),
        "image display form",
        discriminator=None,
    )
    relim_mode = values["relim_mode"]
    colormap = values["colormap"]
    if not isinstance(relim_mode, ImageRelimMode):
        raise TypeError("relim_mode form value must be ImageRelimMode")
    if not isinstance(colormap, ImageColormap):
        raise TypeError("colormap form value must be ImageColormap")

    x_view = _range_from_form(values, "x_min", "x_max", "x_view")
    y_view = _range_from_form(values, "y_min", "y_max", "y_view")
    submitted_fixed = _range_from_form(
        values,
        "color_min",
        "color_max",
        "fixed_color_limits",
    )
    if relim_mode is ImageRelimMode.FIXED:
        if base.relim_mode is not ImageRelimMode.FIXED:
            if current_color_limits is None:
                raise ValueError(
                    "entering fixed relim_mode requires current_color_limits"
                )
            fixed_color_limits = validated_image_range(
                current_color_limits,
                "current_color_limits",
            )
        else:
            if submitted_fixed is None:
                raise ValueError(
                    "fixed relim_mode requires color minimum and maximum"
                )
            fixed_color_limits = submitted_fixed
    else:
        fixed_color_limits = submitted_fixed

    candidate = ImageDisplayState(
        revision=base.revision,
        relim_mode=relim_mode,
        colormap=colormap,
        x_view=x_view,
        y_view=y_view,
        fixed_color_limits=fixed_color_limits,
    )
    if candidate == base:
        return base
    return replace(candidate, revision=base.revision + 1)


def _data_range(data_min: object, data_max: object) -> tuple[float, float]:
    low = finite_real(data_min, "data minimum")
    high = finite_real(data_max, "data maximum")
    if low > high:
        raise ValueError("data minimum cannot exceed data maximum")
    return low, high


def target_image_color_limits(
    state: ImageDisplayState,
    data_min: object,
    data_max: object,
) -> ImageRange:
    """Derive the established tight/normal/fixed target for one frame."""

    if not isinstance(state, ImageDisplayState):
        raise TypeError("state must be ImageDisplayState")
    if state.relim_mode is ImageRelimMode.FIXED:
        assert state.fixed_color_limits is not None
        return state.fixed_color_limits

    low, high = _data_range(data_min, data_max)
    span = (high - low) or (abs(high) or 1.0)
    if not math.isfinite(span):
        raise ValueError("data range span must be finite")
    if state.relim_mode is ImageRelimMode.NORMAL and low >= 0.0:
        target = (0.0, high * 1.2 if high else 1.0)
    else:
        target = (low - 0.1 * span, high + 0.1 * span)
    return validated_image_range(target, "derived color limits")


def deadband_image_color_limits(
    state: ImageDisplayState,
    current_color_limits: ImageRange | None,
    data_min: object,
    data_max: object,
    *,
    force: bool = False,
) -> ImageRange:
    """Return current or target limits using the mature live-view hysteresis.

    Growth is immediate whenever data would clip.  NORMAL holds while its high
    value occupies 70--100% of the current ceiling.  TIGHT holds while no value
    clips and the data has not vacated more than 35% at either side.  ``force``
    bypasses both dead bands for an authored mode change.
    """

    if not isinstance(state, ImageDisplayState):
        raise TypeError("state must be ImageDisplayState")
    if not isinstance(force, bool):
        raise TypeError("force must be bool")
    if state.relim_mode is ImageRelimMode.FIXED:
        assert state.fixed_color_limits is not None
        return state.fixed_color_limits
    low, high = _data_range(data_min, data_max)
    target = target_image_color_limits(state, low, high)
    if current_color_limits is None or force:
        return target
    current_low, current_high = validated_image_range(
        current_color_limits,
        "current_color_limits",
    )

    effective_normal = state.relim_mode is ImageRelimMode.NORMAL and low >= 0.0
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
    too_empty = high < current_high - 0.35 * span or low > current_low + 0.35 * span
    if not clips and not too_empty:
        return current_low, current_high
    return target


__all__ = [
    "ImageColormap",
    "ImageDisplayState",
    "ImageRange",
    "ImageRelimMode",
    "deadband_image_color_limits",
    "image_display_form_spec",
    "image_display_for_viewport",
    "image_display_form_values",
    "image_display_from_form",
    "image_viewport_for_display_state",
    "target_image_color_limits",
    "validated_image_range",
]
