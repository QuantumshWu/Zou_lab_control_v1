"""Pure authored state and exact interaction mapping for numeric curves."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real

import numpy as np

from zlc_storage import (
    canonical_text,
    exact_mapping,
    finite_real,
    nonnegative_integer,
)

from .display_range import (
    DisplayRange,
    RelimMode,
    display_range_form_values,
    display_range_from_form,
    optional_display_range,
    validated_display_range,
)
from .figure.model import EvaluatedAxis
from .form import FormChoice, FormFieldProps, FormSpec


@dataclass(frozen=True, slots=True)
class CurveDisplayState:
    """One authored curve-panel display value.

    ``x_view`` is the sole user viewport pin.  Curves never author a y
    viewport: ``relim_mode`` and ``fixed_y_limits`` own the value axis.
    Dynamic accepted y limits belong to a rendered front, not this state.
    """

    revision: int = 0
    relim_mode: RelimMode = RelimMode.NORMAL
    x_view: DisplayRange | None = None
    fixed_y_limits: DisplayRange | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision",
            nonnegative_integer(self.revision, "curve display revision"),
        )
        if not isinstance(self.relim_mode, RelimMode):
            raise TypeError("relim_mode must be RelimMode")
        object.__setattr__(
            self,
            "x_view",
            optional_display_range(self.x_view, "x_view"),
        )
        object.__setattr__(
            self,
            "fixed_y_limits",
            optional_display_range(self.fixed_y_limits, "fixed_y_limits"),
        )
        if self.relim_mode is RelimMode.FIXED and self.fixed_y_limits is None:
            raise ValueError("fixed relim_mode requires fixed_y_limits")


_CURVE_DISPLAY_FORM = FormSpec(
    (
        FormFieldProps(
            "relim_mode",
            "choice",
            "Y limits",
            default=RelimMode.NORMAL,
            choices=tuple(
                FormChoice(mode.value.title(), mode) for mode in RelimMode
            ),
        ),
        FormFieldProps("x_min", "float", "X minimum", default=None),
        FormFieldProps("x_max", "float", "X maximum", default=None),
        FormFieldProps("y_min", "float", "Y minimum", default=None),
        FormFieldProps("y_max", "float", "Y maximum", default=None),
    )
)


def curve_display_form_spec() -> FormSpec:
    """Return the exact shared Setting/Edit projection for a curve."""

    return _CURVE_DISPLAY_FORM


def curve_display_form_values(state: CurveDisplayState) -> dict[str, object]:
    if not isinstance(state, CurveDisplayState):
        raise TypeError("state must be CurveDisplayState")
    x_min, x_max = display_range_form_values(state.x_view)
    y_min, y_max = display_range_form_values(state.fixed_y_limits)
    return {
        "relim_mode": state.relim_mode,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def curve_display_from_form(
    base: CurveDisplayState,
    values: dict[str, object],
    *,
    current_y_limits: DisplayRange | None = None,
) -> CurveDisplayState:
    """Apply one exact form projection and advance only semantic changes.

    The first transition into ``FIXED`` freezes the limits painted on the
    submitted front.  A dormant older fixed range is never substituted.
    """

    if not isinstance(base, CurveDisplayState):
        raise TypeError("base must be CurveDisplayState")
    values = exact_mapping(
        values,
        frozenset(_CURVE_DISPLAY_FORM.keys),
        "curve display form",
        discriminator=None,
    )
    relim_mode = values["relim_mode"]
    if not isinstance(relim_mode, RelimMode):
        raise TypeError("relim_mode form value must be RelimMode")
    x_view = display_range_from_form(values, "x_min", "x_max", "x_view")
    submitted_fixed = display_range_from_form(
        values,
        "y_min",
        "y_max",
        "fixed_y_limits",
    )
    if relim_mode is RelimMode.FIXED:
        if base.relim_mode is not RelimMode.FIXED:
            if current_y_limits is None:
                raise ValueError(
                    "entering fixed relim_mode requires current_y_limits"
                )
            fixed_y_limits = validated_display_range(
                current_y_limits,
                "current_y_limits",
            )
        else:
            if submitted_fixed is None:
                raise ValueError(
                    "fixed relim_mode requires y minimum and maximum"
                )
            fixed_y_limits = submitted_fixed
    else:
        fixed_y_limits = submitted_fixed

    candidate = CurveDisplayState(
        revision=base.revision,
        relim_mode=relim_mode,
        x_view=x_view,
        fixed_y_limits=fixed_y_limits,
    )
    if candidate == base:
        return base
    return replace(candidate, revision=base.revision + 1)


def curve_display_with_x_view(
    base: CurveDisplayState,
    x_view: DisplayRange | None,
) -> CurveDisplayState:
    """Commit a gesture's sole authoritative curve viewport pin."""

    if not isinstance(base, CurveDisplayState):
        raise TypeError("base must be CurveDisplayState")
    x_view = optional_display_range(x_view, "x_view")
    if x_view == base.x_view:
        return base
    return replace(base, revision=base.revision + 1, x_view=x_view)


def _normalized_plot_bounds(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError("plot_bounds must be (left, top, right, bottom)")
    left, top, right, bottom = (
        finite_real(item, f"plot_bounds[{index}]")
        for index, item in enumerate(value)
    )
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError("plot_bounds must be a nonempty top-origin unit rectangle")
    return left, top, right, bottom


def numeric_curve_coordinates(axis: EvaluatedAxis) -> tuple[Real, ...]:
    """Return finite strictly monotonic numeric curve coordinates.

    Uniform spacing is deliberately not required.  Non-numeric, repeated, or
    direction-changing axes are still renderable by the generic Figure path,
    but cannot enter the exact interactive curve path.
    """

    if not isinstance(axis, EvaluatedAxis):
        raise TypeError("axis must be EvaluatedAxis")
    direction = 0
    previous: float | None = None
    for index, value in enumerate(axis.coordinates):
        scalar = value.item() if isinstance(value, np.generic) else value
        if isinstance(scalar, bool) or not isinstance(scalar, Real):
            raise TypeError(
                f"curve x coordinate {index} must be a numeric scalar"
            )
        numeric = float(scalar)
        if not math.isfinite(numeric):
            raise ValueError(f"curve x coordinate {index} must be finite")
        if previous is not None:
            delta = numeric - previous
            if delta == 0.0:
                raise ValueError(
                    "interactive curve x coordinates must be strictly monotonic"
                )
            step_direction = 1 if delta > 0.0 else -1
            if direction and step_direction != direction:
                raise ValueError(
                    "interactive curve x coordinates must be strictly monotonic"
                )
            direction = step_direction
        previous = numeric
    if previous is None:
        raise ValueError("interactive curve axis must not be empty")
    # EvaluatedAxis already owns an immutable tuple.  Return that exact owner
    # after validation rather than allocating list/tuple/np.diff scratch before
    # the caller's rendering path.
    return axis.coordinates


def curve_home_x_limits(axis: EvaluatedAxis) -> DisplayRange:
    """Return main's exact coordinate span for a numeric curve axis."""

    coordinates = numeric_curve_coordinates(axis)
    low = min(coordinates)
    high = max(coordinates)
    if low == high:
        half_span = max(0.5, 0.05 * abs(low))
        return validated_display_range(
            (low - half_span, high + half_span),
            "single-point curve home limits",
        )
    return validated_display_range((low, high), "curve home limits")


@dataclass(frozen=True, slots=True)
class NumericDisplayAxis:
    """Presentation-only identity for one numeric display axis.

    Unlike :class:`EvaluatedAxis`, this value carries no data-domain role,
    storage indices, coordinate frame, or authority axis id.  Document-backed
    surfaces such as the pulse preview can therefore share the exact numeric
    viewport math without fabricating a dataset axis that does not exist.
    """

    key: str
    label: str
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", canonical_text(self.key, "axis key"))
        object.__setattr__(self, "label", canonical_text(self.label, "axis label"))
        unit = self.unit
        if unit is not None:
            unit = canonical_text(unit, "axis unit")
        object.__setattr__(self, "unit", unit)


@dataclass(frozen=True, slots=True)
class NumericViewportTransform:
    """Exact draw-frozen mapping shared by numeric raster surfaces.

    ``plot_bounds`` is normalized to the complete raster with a top-left
    origin, matching Qt pointer coordinates.  ``x_limits`` and ``y_limits``
    are the actual limits read back after drawing; ``home_x_limits`` is the
    surface-domain home independent of an authored x pin.
    """

    x_axis: NumericDisplayAxis
    display_revision: int
    plot_bounds: tuple[float, float, float, float]
    x_limits: DisplayRange
    y_limits: DisplayRange
    home_x_limits: DisplayRange

    def __post_init__(self) -> None:
        if not isinstance(self.x_axis, NumericDisplayAxis):
            raise TypeError("x_axis must be NumericDisplayAxis")
        self._validate_common()

    def _validate_common(self) -> None:
        object.__setattr__(
            self,
            "display_revision",
            nonnegative_integer(
                self.display_revision,
                "curve display revision",
            ),
        )
        object.__setattr__(
            self,
            "plot_bounds",
            _normalized_plot_bounds(self.plot_bounds),
        )
        for field in ("x_limits", "y_limits", "home_x_limits"):
            object.__setattr__(
                self,
                field,
                validated_display_range(getattr(self, field), field),
            )

    def contains_widget_normalized(self, x: object, y: object) -> bool:
        x = finite_real(x, "widget x")
        y = finite_real(y, "widget y")
        left, top, right, bottom = self.plot_bounds
        return left <= x <= right and top <= y <= bottom

    def widget_normalized_to_data(
        self,
        x: object,
        y: object,
        *,
        require_inside: bool = True,
    ) -> tuple[float, float]:
        x = finite_real(x, "widget x")
        y = finite_real(y, "widget y")
        if not isinstance(require_inside, bool):
            raise TypeError("require_inside must be bool")
        if require_inside and not self.contains_widget_normalized(x, y):
            raise ValueError("widget point lies outside the curve plot")
        left, top, right, bottom = self.plot_bounds
        x_fraction = (x - left) / (right - left)
        y_fraction = (y - top) / (bottom - top)
        x_low, x_high = self.x_limits
        y_low, y_high = self.y_limits
        return (
            x_low + x_fraction * (x_high - x_low),
            y_high - y_fraction * (y_high - y_low),
        )

    def data_to_widget_normalized(
        self,
        x: object,
        y: object,
    ) -> tuple[float, float]:
        x = finite_real(x, "data x")
        y = finite_real(y, "data y")
        left, top, right, bottom = self.plot_bounds
        x_low, x_high = self.x_limits
        y_low, y_high = self.y_limits
        return (
            left + (x - x_low) / (x_high - x_low) * (right - left),
            top + (y_high - y) / (y_high - y_low) * (bottom - top),
        )

    def zoomed_x_limits(self, anchor_x: object, factor: object) -> DisplayRange:
        anchor = finite_real(anchor_x, "zoom anchor x")
        factor = finite_real(factor, "zoom factor")
        if factor <= 0.0:
            raise ValueError("zoom factor must be positive")
        low, high = self.x_limits
        return validated_display_range(
            (
                anchor + (low - anchor) * factor,
                anchor + (high - anchor) * factor,
            ),
            "zoomed x limits",
        )

    def panned_x_limits(
        self,
        press_widget_x: object,
        current_widget_x: object,
        *,
        start_x_limits: DisplayRange | None = None,
    ) -> DisplayRange:
        press = finite_real(press_widget_x, "press widget x")
        current = finite_real(current_widget_x, "current widget x")
        start = (
            self.x_limits
            if start_x_limits is None
            else validated_display_range(start_x_limits, "start_x_limits")
        )
        left, _top, right, _bottom = self.plot_bounds
        span = start[1] - start[0]
        shift = -(current - press) / (right - left) * span
        return validated_display_range(
            (start[0] + shift, start[1] + shift),
            "panned x limits",
        )

    def selection_x_span(
        self,
        first_widget_x: object,
        second_widget_x: object,
    ) -> DisplayRange:
        first = self.widget_normalized_to_data(
            first_widget_x,
            0.5 * (self.plot_bounds[1] + self.plot_bounds[3]),
            require_inside=False,
        )[0]
        second = self.widget_normalized_to_data(
            second_widget_x,
            0.5 * (self.plot_bounds[1] + self.plot_bounds[3]),
            require_inside=False,
        )[0]
        return validated_display_range(
            (min(first, second), max(first, second)),
            "numeric selection x span",
        )


@dataclass(frozen=True, slots=True)
class CurveViewportTransform(NumericViewportTransform):
    """Dataset-backed numeric viewport retaining the authority curve axis."""

    x_axis: EvaluatedAxis

    def __post_init__(self) -> None:
        if not isinstance(self.x_axis, EvaluatedAxis):
            raise TypeError("x_axis must be EvaluatedAxis")
        numeric_curve_coordinates(self.x_axis)
        self._validate_common()

__all__ = [
    "CurveDisplayState",
    "CurveViewportTransform",
    "NumericDisplayAxis",
    "NumericViewportTransform",
    "curve_display_form_spec",
    "curve_display_form_values",
    "curve_display_from_form",
    "curve_display_with_x_view",
    "curve_home_x_limits",
    "numeric_curve_coordinates",
]
