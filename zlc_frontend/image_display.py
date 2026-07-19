"""Pure authored display state and colour-range policy for pixel images.

This module owns no Qt, Matplotlib, raster storage, or data-projection intent.
It projects one immutable image-display value into the shared headless form
contract and implements the established tight/normal/fixed range hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from zlc_storage import exact_mapping, nonnegative_integer

from .display_range import (
    DisplayRange,
    RelimMode,
    display_range_form_values,
    display_range_from_form,
    optional_display_range,
    validated_display_range,
)
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
    candidate = ImageViewportTransform(reference.axes, state.revision, expected)
    if state.revision == reference.viewport_revision:
        if candidate != reference:
            raise ValueError(
                "image display coordinate views differ from viewport bounds"
            )
        return reference
    return candidate


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
    "ImageColormap",
    "ImageDisplayState",
    "image_display_form_spec",
    "image_display_for_viewport",
    "image_display_form_values",
    "image_display_from_form",
    "image_viewport_for_display_state",
]
