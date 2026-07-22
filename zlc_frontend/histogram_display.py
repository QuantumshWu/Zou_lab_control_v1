"""Pure authored state, binning, and viewport mapping for histograms.

This module owns the display-only histogram projection.  It deliberately has
no Qt or Matplotlib dependency: render workers consume its frozen values and
GUI code can only map gestures through the exact transform published with a
rendered front.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import numpy as np

from zlc_data import immutable_array
from zlc_storage import exact_mapping, finite_real, nonnegative_integer

from .display_range import (
    DisplayRange,
    RelimMode,
    display_range_form_values,
    display_range_from_form,
    optional_display_range,
    validated_display_range,
)
from .form import FormChoice, FormFieldProps, FormSpec


DEFAULT_HISTOGRAM_BINS = 60
MIN_HISTOGRAM_BINS = 5
MAX_HISTOGRAM_BINS = 500
_COUNT_SHRINK_DEADBAND = 0.60


class HistogramCountScale(str, Enum):
    """Closed count-axis scales supported by the histogram renderer."""

    LINEAR = "linear"
    LOG = "log"


def _histogram_bin_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("histogram bin_count must be an integer")
    result = int(value)
    if not MIN_HISTOGRAM_BINS <= result <= MAX_HISTOGRAM_BINS:
        raise ValueError(
            "histogram bin_count must be between "
            f"{MIN_HISTOGRAM_BINS} and {MAX_HISTOGRAM_BINS}"
        )
    return result


def _validated_count_limits(
    value: object,
    field: str,
    *,
    scale: HistogramCountScale,
) -> DisplayRange:
    result = validated_display_range(value, field)
    if scale is HistogramCountScale.LOG and result[0] <= 0.0:
        raise ValueError(f"{field} lower limit must be positive for log count scale")
    return result


@dataclass(frozen=True, slots=True)
class HistogramDisplayState:
    """The sole authored display state for one histogram panel."""

    revision: int = 0
    relim_mode: RelimMode = RelimMode.TIGHT
    count_scale: HistogramCountScale = HistogramCountScale.LINEAR
    bin_count: int = DEFAULT_HISTOGRAM_BINS
    x_view: DisplayRange | None = None
    fixed_count_limits: DisplayRange | None = None
    # ZERO OR MORE vertical threshold cut lines (the design's frozen histogram
    # selector row): authored values in the histogram's VALUE coordinate, drawn
    # by the renderer and dragged live on the board.  The reference keeps them
    # as plain display state on the figure -- never an analysis authority.
    thresholds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision",
            nonnegative_integer(self.revision, "histogram display revision"),
        )
        if not isinstance(self.relim_mode, RelimMode):
            raise TypeError("relim_mode must be RelimMode")
        if not isinstance(self.count_scale, HistogramCountScale):
            raise TypeError("count_scale must be HistogramCountScale")
        object.__setattr__(self, "bin_count", _histogram_bin_count(self.bin_count))
        object.__setattr__(
            self,
            "x_view",
            optional_display_range(self.x_view, "x_view"),
        )
        fixed = self.fixed_count_limits
        if fixed is not None:
            fixed = validated_display_range(fixed, "fixed_count_limits")
            object.__setattr__(self, "fixed_count_limits", fixed)
            if (
                self.relim_mode is RelimMode.FIXED
                and self.count_scale is HistogramCountScale.LOG
            ):
                _validated_count_limits(
                    fixed,
                    "fixed_count_limits",
                    scale=self.count_scale,
                )
        if self.relim_mode is RelimMode.FIXED:
            if fixed is None:
                raise ValueError("fixed relim_mode requires fixed_count_limits")
        object.__setattr__(
            self,
            "thresholds",
            tuple(
                finite_real(value, "histogram threshold")
                for value in self.thresholds
            ),
        )


_HISTOGRAM_DISPLAY_FORM = FormSpec(
    (
        FormFieldProps(
            "relim_mode",
            "choice",
            "Count limits",
            default=RelimMode.TIGHT,
            choices=tuple(
                FormChoice(mode.value.title(), mode) for mode in RelimMode
            ),
        ),
        FormFieldProps(
            "count_scale",
            "choice",
            "Count scale",
            default=HistogramCountScale.LINEAR,
            choices=tuple(
                FormChoice(scale.value.title(), scale)
                for scale in HistogramCountScale
            ),
        ),
        FormFieldProps(
            "bin_count",
            "int",
            "Bins",
            default=DEFAULT_HISTOGRAM_BINS,
            minimum=MIN_HISTOGRAM_BINS,
            maximum=MAX_HISTOGRAM_BINS,
        ),
        FormFieldProps("x_min", "float", "X minimum", default=None),
        FormFieldProps("x_max", "float", "X maximum", default=None),
        FormFieldProps("count_min", "float", "Count minimum", default=None),
        FormFieldProps("count_max", "float", "Count maximum", default=None),
    )
)


def histogram_display_form_spec() -> FormSpec:
    """Return the exact shared Setting/Edit projection for a histogram."""

    return _HISTOGRAM_DISPLAY_FORM


def histogram_display_form_values(
    state: HistogramDisplayState,
) -> dict[str, object]:
    if not isinstance(state, HistogramDisplayState):
        raise TypeError("state must be HistogramDisplayState")
    x_min, x_max = display_range_form_values(state.x_view)
    count_min, count_max = display_range_form_values(state.fixed_count_limits)
    return {
        "relim_mode": state.relim_mode,
        "count_scale": state.count_scale,
        "bin_count": state.bin_count,
        "x_min": x_min,
        "x_max": x_max,
        "count_min": count_min,
        "count_max": count_max,
    }


def histogram_display_from_form(
    base: HistogramDisplayState,
    values: dict[str, object],
    *,
    current_count_limits: DisplayRange | None = None,
) -> HistogramDisplayState:
    """Apply one exact form and advance the revision on semantic change.

    The first transition into ``FIXED`` freezes the count limits painted on
    the submitted front.  A dormant older fixed range is never substituted.
    """

    if not isinstance(base, HistogramDisplayState):
        raise TypeError("base must be HistogramDisplayState")
    values = exact_mapping(
        values,
        frozenset(_HISTOGRAM_DISPLAY_FORM.keys),
        "histogram display form",
        discriminator=None,
    )
    relim_mode = values["relim_mode"]
    count_scale = values["count_scale"]
    if not isinstance(relim_mode, RelimMode):
        raise TypeError("relim_mode form value must be RelimMode")
    if not isinstance(count_scale, HistogramCountScale):
        raise TypeError("count_scale form value must be HistogramCountScale")
    bin_count = _histogram_bin_count(values["bin_count"])
    x_view = display_range_from_form(values, "x_min", "x_max", "x_view")
    submitted_fixed = display_range_from_form(
        values,
        "count_min",
        "count_max",
        "fixed_count_limits",
    )
    if relim_mode is RelimMode.FIXED:
        if base.relim_mode is not RelimMode.FIXED:
            if current_count_limits is None:
                raise ValueError(
                    "entering fixed relim_mode requires current_count_limits"
                )
            fixed_count_limits = _validated_count_limits(
                current_count_limits,
                "current_count_limits",
                scale=count_scale,
            )
        else:
            if submitted_fixed is None:
                raise ValueError(
                    "fixed relim_mode requires count minimum and maximum"
                )
            fixed_count_limits = _validated_count_limits(
                submitted_fixed,
                "fixed_count_limits",
                scale=count_scale,
            )
    else:
        fixed_count_limits = submitted_fixed

    candidate = HistogramDisplayState(
        revision=base.revision,
        relim_mode=relim_mode,
        count_scale=count_scale,
        bin_count=bin_count,
        x_view=x_view,
        fixed_count_limits=fixed_count_limits,
        # The display form does not edit thresholds; the authored cut lines
        # ride along unchanged (they are dragged on the board, not typed).
        thresholds=base.thresholds,
    )
    if candidate == base:
        return base
    return replace(candidate, revision=base.revision + 1)


def histogram_display_with_x_view(
    base: HistogramDisplayState,
    x_view: DisplayRange | None,
) -> HistogramDisplayState:
    """Commit one gesture's sole authored histogram viewport pin."""

    if not isinstance(base, HistogramDisplayState):
        raise TypeError("base must be HistogramDisplayState")
    x_view = optional_display_range(x_view, "x_view")
    if x_view == base.x_view:
        return base
    return replace(base, revision=base.revision + 1, x_view=x_view)


def histogram_display_with_thresholds(
    base: HistogramDisplayState,
    thresholds: tuple[float, ...],
) -> HistogramDisplayState:
    """Commit one authored threshold set (a board drag step or an explicit
    ``set_thresholds``-style write, the reference's single mutation point)."""

    if not isinstance(base, HistogramDisplayState):
        raise TypeError("base must be HistogramDisplayState")
    candidate = replace(
        base,
        thresholds=tuple(
            finite_real(value, "histogram threshold") for value in thresholds
        ),
    )
    if candidate == base:
        return base
    return replace(candidate, revision=base.revision + 1)


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


@dataclass(frozen=True, slots=True)
class HistogramViewportTransform:
    """Exact draw-frozen mapping for one interactive histogram raster."""

    display_revision: int
    plot_bounds: tuple[float, float, float, float]
    x_limits: DisplayRange
    count_limits: DisplayRange
    home_x_limits: DisplayRange
    count_scale: HistogramCountScale
    relim_mode: RelimMode
    x_limits_are_auto: bool
    bin_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "display_revision",
            nonnegative_integer(
                self.display_revision,
                "histogram display revision",
            ),
        )
        object.__setattr__(
            self,
            "plot_bounds",
            _normalized_plot_bounds(self.plot_bounds),
        )
        if not isinstance(self.count_scale, HistogramCountScale):
            raise TypeError("count_scale must be HistogramCountScale")
        if not isinstance(self.relim_mode, RelimMode):
            raise TypeError("relim_mode must be RelimMode")
        if not isinstance(self.x_limits_are_auto, bool):
            raise TypeError("x_limits_are_auto must be bool")
        object.__setattr__(self, "bin_count", _histogram_bin_count(self.bin_count))
        object.__setattr__(
            self,
            "x_limits",
            validated_display_range(self.x_limits, "x_limits"),
        )
        object.__setattr__(
            self,
            "home_x_limits",
            validated_display_range(self.home_x_limits, "home_x_limits"),
        )
        object.__setattr__(
            self,
            "count_limits",
            _validated_count_limits(
                self.count_limits,
                "count_limits",
                scale=self.count_scale,
            ),
        )
        if self.x_limits_are_auto and self.x_limits != self.home_x_limits:
            raise ValueError("automatic histogram x limits must equal the home range")

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
            raise ValueError("widget point lies outside the histogram plot")
        left, top, right, bottom = self.plot_bounds
        x_fraction = (x - left) / (right - left)
        count_fraction = (y - top) / (bottom - top)
        x_low, x_high = self.x_limits
        count_low, count_high = self.count_limits
        if self.count_scale is HistogramCountScale.LOG:
            log_low = math.log(count_low)
            log_high = math.log(count_high)
            count = math.exp(log_high - count_fraction * (log_high - log_low))
        else:
            count = count_high - count_fraction * (count_high - count_low)
        return x_low + x_fraction * (x_high - x_low), count

    def data_to_widget_normalized(
        self,
        x: object,
        count: object,
    ) -> tuple[float, float]:
        x = finite_real(x, "data x")
        count = finite_real(count, "data count")
        left, top, right, bottom = self.plot_bounds
        x_low, x_high = self.x_limits
        count_low, count_high = self.count_limits
        if self.count_scale is HistogramCountScale.LOG:
            if count <= 0.0:
                raise ValueError("data count must be positive for log count scale")
            count_fraction = (math.log(count_high) - math.log(count)) / (
                math.log(count_high) - math.log(count_low)
            )
        else:
            count_fraction = (count_high - count) / (count_high - count_low)
        return (
            left + (x - x_low) / (x_high - x_low) * (right - left),
            top + count_fraction * (bottom - top),
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
        middle_y = 0.5 * (self.plot_bounds[1] + self.plot_bounds[3])
        first = self.widget_normalized_to_data(
            first_widget_x,
            middle_y,
            require_inside=False,
        )[0]
        second = self.widget_normalized_to_data(
            second_widget_x,
            middle_y,
            require_inside=False,
        )[0]
        return validated_display_range(
            (min(first, second), max(first, second)),
            "histogram selection x span",
        )


def histogram_count_limits(
    state: HistogramDisplayState,
    peak_count: object,
    *,
    current_count_limits: DisplayRange | None = None,
    previous_relim_mode: RelimMode | None = None,
    previous_count_scale: HistogramCountScale | None = None,
) -> DisplayRange:
    """Resolve count limits with the established shrink-only deadband.

    Growth is immediate if the current front would clip a count.  Shrinkage is
    delayed while the newly derived ceiling remains at least 60% of the current
    ceiling.  A mode or scale change always applies the new policy immediately.
    """

    if not isinstance(state, HistogramDisplayState):
        raise TypeError("state must be HistogramDisplayState")
    peak = finite_real(peak_count, "histogram peak count", minimum=0.0)
    if previous_relim_mode is not None and not isinstance(
        previous_relim_mode,
        RelimMode,
    ):
        raise TypeError("previous_relim_mode must be RelimMode or None")
    if previous_count_scale is not None and not isinstance(
        previous_count_scale,
        HistogramCountScale,
    ):
        raise TypeError(
            "previous_count_scale must be HistogramCountScale or None"
        )

    if state.relim_mode is RelimMode.FIXED:
        assert state.fixed_count_limits is not None
        return _validated_count_limits(
            state.fixed_count_limits,
            "fixed_count_limits",
            scale=state.count_scale,
        )
    if state.count_scale is HistogramCountScale.LOG:
        target = (0.5, max(3.0 * peak, 1.0))
    elif state.relim_mode is RelimMode.TIGHT:
        target = (0.0, max(1.1 * peak, 1.0))
    else:
        target = (0.0, max(1.2 * peak, 1.0))
    target = _validated_count_limits(
        target,
        "derived histogram count limits",
        scale=state.count_scale,
    )

    force = (
        previous_relim_mode is None
        or previous_count_scale is None
        or previous_relim_mode is not state.relim_mode
        or previous_count_scale is not state.count_scale
    )
    if current_count_limits is None or force:
        return target
    current = _validated_count_limits(
        current_count_limits,
        "current_count_limits",
        scale=state.count_scale,
    )
    if current[0] != target[0]:
        return target
    if peak > current[1]:
        return target
    if target[1] >= _COUNT_SHRINK_DEADBAND * current[1]:
        return current
    return target


def _immutable_histogram_array(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype).newbyteorder("<")
    source = np.asarray(values, dtype=dtype)
    return immutable_array(source, dtype=dtype, shape=source.shape)


def _histogram_series(series_samples: object) -> tuple[np.ndarray, ...]:
    if not isinstance(series_samples, tuple):
        raise TypeError("histogram series_samples must be a tuple")
    if not series_samples:
        raise ValueError("histogram series_samples must not be empty")
    result = []
    for index, samples in enumerate(series_samples):
        values = np.asarray(samples)
        if values.ndim != 1:
            raise ValueError(
                f"histogram series {index} samples must be one-dimensional"
            )
        if values.dtype.kind not in "biuf":
            raise TypeError(
                "histogram samples must have a real numeric or boolean dtype"
            )
        if values.size and not bool(np.all(np.isfinite(values))):
            raise ValueError("valid histogram samples must all be finite")
        result.append(values)
    return tuple(result)


def _histogram_projection_edges(
    values_by_series: tuple[np.ndarray, ...],
    bins: int,
    value_range: DisplayRange | None,
) -> np.ndarray:
    if all(values.dtype.kind == "b" for values in values_by_series):
        if value_range is not None and validated_display_range(
            value_range,
            "histogram projection value_range",
        ) != (-0.5, 1.5):
            raise ValueError(
                "boolean histogram value_range must preserve false/true bins"
            )
        return np.asarray((-0.5, 0.5, 1.5), dtype=np.float64)
    if value_range is not None:
        low, high = validated_display_range(
            value_range,
            "histogram projection value_range",
        )
        return np.linspace(low, high, bins + 1, dtype=np.float64)
    nonempty = tuple(values for values in values_by_series if values.size)
    if not nonempty:
        representative = np.asarray((), dtype=np.float64)
    else:
        minima = tuple(values.min().item() for values in nonempty)
        maxima = tuple(values.max().item() for values in nonempty)
        representative = np.asarray((min(minima), max(maxima)))
    return np.histogram_bin_edges(representative, bins=bins)


def histogram_projection_home_x_limits(
    series_samples: tuple[np.ndarray, ...],
    bins: int = DEFAULT_HISTOGRAM_BINS,
    *,
    value_range: DisplayRange | None = None,
) -> DisplayRange:
    """Derive the exact owner-defined bin domain without materializing counts."""

    values_by_series = _histogram_series(series_samples)
    bins = _histogram_bin_count(bins)
    return histogram_home_x_limits(
        _histogram_projection_edges(values_by_series, bins, value_range)
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class HistogramBinProjection:
    """One exact-sample-bound, shared-edge display projection.

    Counts and edges are computed by this constructor and cannot be supplied
    independently by a caller.  The retained sample-array identities let the
    payload boundary prove that Agg and Qt consume the projection derived from
    those exact evaluated series without a second binning pass.
    """

    series_samples: tuple[np.ndarray, ...]
    bin_counts: tuple[np.ndarray, ...]
    bin_edges: np.ndarray
    requested_bin_count: int

    def __init__(
        self,
        series_samples: tuple[np.ndarray, ...],
        bins: int = DEFAULT_HISTOGRAM_BINS,
        *,
        value_range: DisplayRange | None = None,
    ) -> None:
        values_by_series = _histogram_series(series_samples)
        bins = _histogram_bin_count(bins)
        edges = _histogram_projection_edges(values_by_series, bins, value_range)

        immutable_edges = _immutable_histogram_array(
            np.asarray(edges, dtype=np.float64),
            np.dtype(np.float64),
        )
        counts = []
        for values in values_by_series:
            histogram_values = (
                values.astype(np.uint8, copy=False)
                if values.dtype.kind == "b"
                else values
            )
            series_counts = np.histogram(
                histogram_values,
                bins=immutable_edges,
            )[0]
            if int(np.sum(series_counts, dtype=np.int64)) != int(values.size):
                raise ValueError(
                    "histogram projection range did not retain every sample"
                )
            counts.append(
                _immutable_histogram_array(
                    np.asarray(series_counts, dtype=np.int64),
                    np.dtype(np.int64),
                )
            )
        object.__setattr__(self, "series_samples", values_by_series)
        object.__setattr__(self, "bin_counts", tuple(counts))
        object.__setattr__(self, "bin_edges", immutable_edges)
        object.__setattr__(self, "requested_bin_count", bins)


def histogram_home_x_limits(edges: np.ndarray) -> DisplayRange:
    """Return the exact finite bin domain used as the histogram home view."""

    values = np.asarray(edges)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("histogram edges must be a one-dimensional boundary array")
    if values.dtype.kind not in "iuf" or not bool(np.all(np.isfinite(values))):
        raise ValueError("histogram edges must be finite real numbers")
    if not bool(np.all(np.diff(values.astype(np.float64)) > 0.0)):
        raise ValueError("histogram edges must be strictly increasing")
    return validated_display_range(
        (values[0].item(), values[-1].item()),
        "histogram home x limits",
    )


__all__ = [
    "DEFAULT_HISTOGRAM_BINS",
    "HistogramCountScale",
    "HistogramBinProjection",
    "HistogramDisplayState",
    "HistogramViewportTransform",
    "MAX_HISTOGRAM_BINS",
    "MIN_HISTOGRAM_BINS",
    "histogram_count_limits",
    "histogram_display_form_spec",
    "histogram_display_form_values",
    "histogram_display_from_form",
    "histogram_display_with_x_view",
    "histogram_home_x_limits",
    "histogram_projection_home_x_limits",
]
