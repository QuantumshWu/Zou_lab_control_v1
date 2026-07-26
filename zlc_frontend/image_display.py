"""Pure authored display state and colour-range policy for pixel images.

This module owns no Qt, Matplotlib, raster storage, or data-projection intent.
It projects one immutable image-display value into the shared headless form
contract and implements the established tight/normal/fixed range hysteresis.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from zlc_storage import exact_mapping, nonnegative_integer

from .display_range import (
    DisplayRange,
    RelimMode,
    deadband_display_range,
    display_range_form_values,
    display_range_from_form,
    optional_display_range,
    validated_display_range,
)
from .figure import EvaluatedImage
from .form import FormChoice, FormFieldProps, FormSpec
from .image_view import ImageViewportTransform


class ImageColormap(str, Enum):
    """Closed colormap names sampled by the optional render owner."""

    INFERNO = "inferno"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    GRAY = "gray"
    COOLWARM = "coolwarm"


@dataclass(frozen=True, slots=True)
class ImageDisplayState:
    """One authored image-panel display value.

    Dynamic effective limits are deliberately absent: they describe a rendered
    front, not authored state, and therefore never advance ``revision``.
    """

    revision: int = 0
    relim_mode: RelimMode = RelimMode.TIGHT
    colormap: ImageColormap = ImageColormap.GRAY
    x_view: DisplayRange | None = None
    y_view: DisplayRange | None = None
    fixed_color_limits: DisplayRange | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision",
            nonnegative_integer(self.revision, "image display revision"),
        )
        if not isinstance(self.relim_mode, RelimMode):
            raise TypeError("relim_mode must be RelimMode")
        if not isinstance(self.colormap, ImageColormap):
            raise TypeError("colormap must be ImageColormap")
        for field in ("x_view", "y_view", "fixed_color_limits"):
            object.__setattr__(
                self,
                field,
                optional_display_range(getattr(self, field), field),
            )
        if (
            self.relim_mode is RelimMode.FIXED
            and self.fixed_color_limits is None
        ):
            raise ValueError("fixed relim_mode requires fixed_color_limits")


def _automatic_image_range(
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
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


def evaluated_image_data_range(
    images: Iterable[EvaluatedImage],
) -> tuple[float, float] | None:
    """Pool finite valid IMAGE values without creating a display raster."""

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
        cell_low, cell_high = _automatic_image_range(values, valid)
        low = cell_low if low is None else min(low, cell_low)
        high = cell_high if high is None else max(high, cell_high)
    return None if low is None else (low, high)


def resolve_image_color_limits(
    image: EvaluatedImage,
    state: ImageDisplayState,
    *,
    current_color_limits: DisplayRange | None,
    previous_relim_mode: RelimMode | None,
) -> tuple[tuple[float, float] | None, tuple[float, float]]:
    """Resolve one image's observed range and effective authored clim.

    This function returns display facts only.  It never allocates a pixel
    plane, palette, or histogram as a side effect.
    """

    if not isinstance(image, EvaluatedImage):
        raise TypeError("image must be EvaluatedImage")
    if not isinstance(state, ImageDisplayState):
        raise TypeError("state must be ImageDisplayState")
    if previous_relim_mode is not None and not isinstance(
        previous_relim_mode,
        RelimMode,
    ):
        raise TypeError("previous_relim_mode must be RelimMode or None")
    data_range = evaluated_image_data_range((image,))
    return resolve_image_color_limits_from_range(
        data_range,
        state,
        current_color_limits=current_color_limits,
        previous_relim_mode=previous_relim_mode,
    )


def resolve_image_color_limits_from_range(
    data_range: tuple[float, float] | None,
    state: ImageDisplayState,
    *,
    current_color_limits: DisplayRange | None,
    previous_relim_mode: RelimMode | None,
) -> tuple[tuple[float, float] | None, tuple[float, float]]:
    """Resolve authored clim from an already-owned exact image range."""

    if data_range is not None:
        data_range = validated_display_range(data_range, "image data_range")
    if not isinstance(state, ImageDisplayState):
        raise TypeError("state must be ImageDisplayState")
    if previous_relim_mode is not None and not isinstance(
        previous_relim_mode,
        RelimMode,
    ):
        raise TypeError("previous_relim_mode must be RelimMode or None")
    if data_range is None:
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
            data_range[0],
            data_range[1],
            fixed_range=state.fixed_color_limits,
            force=(
                previous_relim_mode is None
                or previous_relim_mode is not state.relim_mode
            ),
        )
    return data_range, color_limits


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
        if expected != reference.visible_bounds:
            raise ValueError(
                "image display coordinate views differ from viewport bounds"
            )
        return reference
    return reference._replacement_with_visible_bounds(expected, state.revision)


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
            default=RelimMode.TIGHT,
            choices=tuple(
                FormChoice(mode.value.title(), mode) for mode in RelimMode
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


def image_display_form_values(state: ImageDisplayState) -> dict[str, object]:
    """Project authored state to the form's exact typed keys."""

    if not isinstance(state, ImageDisplayState):
        raise TypeError("state must be ImageDisplayState")
    x_min, x_max = display_range_form_values(state.x_view)
    y_min, y_max = display_range_form_values(state.y_view)
    color_min, color_max = display_range_form_values(state.fixed_color_limits)
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


def image_display_from_form(
    base: ImageDisplayState,
    values: dict[str, object],
    *,
    current_color_limits: DisplayRange | None = None,
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
    if not isinstance(relim_mode, RelimMode):
        raise TypeError("relim_mode form value must be RelimMode")
    if not isinstance(colormap, ImageColormap):
        raise TypeError("colormap form value must be ImageColormap")

    x_view = display_range_from_form(values, "x_min", "x_max", "x_view")
    y_view = display_range_from_form(values, "y_min", "y_max", "y_view")
    submitted_fixed = display_range_from_form(
        values,
        "color_min",
        "color_max",
        "fixed_color_limits",
    )
    if relim_mode is RelimMode.FIXED:
        if base.relim_mode is not RelimMode.FIXED:
            if current_color_limits is None:
                raise ValueError(
                    "entering fixed relim_mode requires current_color_limits"
                )
            fixed_color_limits = validated_display_range(
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


__all__ = [
    "evaluated_image_data_range",
    "ImageColormap",
    "ImageDisplayState",
    "image_display_form_spec",
    "image_display_for_viewport",
    "image_display_form_values",
    "image_display_from_form",
    "image_viewport_for_display_state",
    "resolve_image_color_limits",
    "resolve_image_color_limits_from_range",
]
