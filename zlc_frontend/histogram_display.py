"""Pure authored state, binning, and viewport mapping for histograms.

This module owns the display-only histogram projection.  It deliberately has
no Qt or Matplotlib dependency: render workers consume its frozen values and
GUI code can only map gestures through the exact transform published with a
rendered front.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math

import numpy as np

from zlc_data import (
    AxisSourceRef,
)
from zlc_data._arrays import immutable_array
from zlc_data.codec import axis_source_ref_from_tree, axis_source_ref_to_tree
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
from .fit_projection import canonical_panel_focus_address
from .numeric_viewport import (
    data_x_to_normalized_widget,
    normalized_plot_bounds,
    normalized_plot_contains,
    normalized_widget_x_to_data,
    panned_x_limits as _panned_x_limits,
    zoomed_x_limits as _zoomed_x_limits,
)


DEFAULT_HISTOGRAM_BINS = 60
MIN_HISTOGRAM_BINS = 5
_COUNT_SHRINK_DEADBAND = 0.60


def _histogram_bin_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("histogram bin_count must be an integer")
    result = int(value)
    if result < MIN_HISTOGRAM_BINS:
        raise ValueError(
            f"histogram bin_count must be at least {MIN_HISTOGRAM_BINS}"
        )
    return result


def _validated_count_limits(
    value: object,
    field: str,
    *,
    log_count_axis: bool,
) -> DisplayRange:
    if not isinstance(log_count_axis, bool):
        raise TypeError("log_count_axis must be bool")
    result = validated_display_range(value, field)
    if log_count_axis and result[0] <= 0.0:
        raise ValueError(f"{field} lower limit must be positive for log count scale")
    return result


@dataclass(frozen=True, slots=True)
class HistogramDisplayState:
    """The sole authored display state for one histogram panel."""

    revision: int = 0
    relim_mode: RelimMode = RelimMode.TIGHT
    log_count_axis: bool = False
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
        if not isinstance(self.log_count_axis, bool):
            raise TypeError("log_count_axis must be bool")
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
                and self.log_count_axis
            ):
                _validated_count_limits(
                    fixed,
                    "fixed_count_limits",
                    log_count_axis=self.log_count_axis,
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


@dataclass(frozen=True, slots=True)
class HistogramCellThresholds:
    """One Grid cell's display-only threshold set.

    ``address`` is the sparse source-aware cell identity painted in the overview,
    not a dense storage row and not a transient ``LatestNonempty`` resolution.
    """

    address: tuple[tuple[AxisSourceRef, int], ...]
    thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "address",
            canonical_panel_focus_address(self.address),
        )
        object.__setattr__(
            self,
            "thresholds",
            tuple(
                finite_real(value, "histogram cell threshold")
                for value in self.thresholds
            ),
        )


def _validated_cell_thresholds(
    value: object,
) -> tuple[HistogramCellThresholds, ...]:
    entries = tuple(value)
    if any(not isinstance(item, HistogramCellThresholds) for item in entries):
        raise TypeError(
            "cell_thresholds must contain HistogramCellThresholds values"
        )
    if len({item.address for item in entries}) != len(entries):
        raise ValueError("histogram cell threshold addresses must be unique")
    return tuple(
        sorted(
            entries,
            key=lambda item: item.address,
        )
    )


@dataclass(frozen=True, slots=True)
class FacetedHistogramDisplayState:
    """One shared histogram style plus thresholds authored per logical cell."""

    display: HistogramDisplayState
    cell_thresholds: tuple[HistogramCellThresholds, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.display, HistogramDisplayState):
            raise TypeError("faceted histogram display must wrap HistogramDisplayState")
        if self.display.thresholds:
            raise ValueError(
                "faceted histogram thresholds belong to cells, not the shared display"
            )
        object.__setattr__(
            self,
            "cell_thresholds",
            _validated_cell_thresholds(self.cell_thresholds),
        )

    @property
    def revision(self) -> int:
        return self.display.revision

    def display_for(
        self,
        address: tuple[tuple[AxisSourceRef, int], ...],
    ) -> HistogramDisplayState:
        resolved = canonical_panel_focus_address(address)
        thresholds = next(
            (
                item.thresholds
                for item in self.cell_thresholds
                if item.address == resolved
            ),
            (),
        )
        return replace(self.display, thresholds=thresholds)


_HISTOGRAM_CELL_THRESHOLDS_SCHEMA = "zlc_frontend.HistogramCellThresholds"


def _focus_address_to_tree(
    value: tuple[tuple[AxisSourceRef, int], ...],
) -> list[dict[str, object]]:
    return [
        {
            "source": axis_source_ref_to_tree(source),
            "index": index,
        }
        for source, index in canonical_panel_focus_address(value)
    ]


def _focus_address_from_tree(
    value: object,
) -> tuple[tuple[AxisSourceRef, int], ...]:
    if not isinstance(value, list):
        raise ValueError("histogram cell threshold address must be a list")
    entries = []
    for raw in value:
        item = exact_mapping(
            raw,
            {"source", "index"},
            "histogram cell threshold address entry",
            discriminator=None,
        )
        entries.append(
            (
                axis_source_ref_from_tree(item["source"]),
                item["index"],
            )
        )
    return canonical_panel_focus_address(entries)


def histogram_cell_thresholds_to_tree(
    entries: tuple[HistogramCellThresholds, ...],
) -> dict[str, object]:
    """Encode the complete sparse threshold map through its frontend owner."""

    resolved = _validated_cell_thresholds(entries)
    return {
        "schema": _HISTOGRAM_CELL_THRESHOLDS_SCHEMA,
        "entries": [
            {
                "address": _focus_address_to_tree(item.address),
                "thresholds": list(item.thresholds),
            }
            for item in resolved
        ],
    }


def histogram_cell_thresholds_from_tree(
    tree: object,
) -> tuple[HistogramCellThresholds, ...]:
    """Decode a persisted sparse threshold map without positional inference."""

    data = exact_mapping(
        tree,
        {"schema", "entries"},
        _HISTOGRAM_CELL_THRESHOLDS_SCHEMA,
    )
    raw_entries = data["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("histogram cell threshold entries must be a list")
    entries = []
    for raw in raw_entries:
        item = exact_mapping(
            raw,
            {"address", "thresholds"},
            "histogram cell threshold entry",
            discriminator=None,
        )
        raw_thresholds = item["thresholds"]
        if not isinstance(raw_thresholds, list):
            raise ValueError("histogram cell thresholds must be a list")
        entries.append(
            HistogramCellThresholds(
                _focus_address_from_tree(item["address"]),
                tuple(raw_thresholds),
            )
        )
    return _validated_cell_thresholds(entries)


def faceted_histogram_display_with_thresholds(
    base: FacetedHistogramDisplayState,
    address: tuple[tuple[AxisSourceRef, int], ...],
    thresholds: tuple[float, ...],
) -> FacetedHistogramDisplayState:
    """Return one-cell candidate state; the host CAS-checks its painted origin."""

    if not isinstance(base, FacetedHistogramDisplayState):
        raise TypeError("base must be FacetedHistogramDisplayState")
    candidate = HistogramCellThresholds(address, tuple(thresholds))
    by_address = {
        item.address: item for item in base.cell_thresholds
    }
    if candidate.thresholds:
        by_address[candidate.address] = candidate
    else:
        by_address.pop(candidate.address, None)
    entries = _validated_cell_thresholds(by_address.values())
    if entries == base.cell_thresholds:
        return base
    return FacetedHistogramDisplayState(
        replace(base.display, revision=base.display.revision + 1),
        entries,
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
            "log_count_axis",
            "bool",
            "Log count",
            default=False,
        ),
        FormFieldProps(
            "bin_count",
            "int",
            "Bins",
            default=DEFAULT_HISTOGRAM_BINS,
            minimum=MIN_HISTOGRAM_BINS,
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
        "log_count_axis": state.log_count_axis,
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
    log_count_axis = values["log_count_axis"]
    if not isinstance(relim_mode, RelimMode):
        raise TypeError("relim_mode form value must be RelimMode")
    if not isinstance(log_count_axis, bool):
        raise TypeError("log_count_axis form value must be bool")
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
                log_count_axis=log_count_axis,
            )
        else:
            if submitted_fixed is None:
                raise ValueError(
                    "fixed relim_mode requires count minimum and maximum"
                )
            fixed_count_limits = _validated_count_limits(
                submitted_fixed,
                "fixed_count_limits",
                log_count_axis=log_count_axis,
            )
    else:
        fixed_count_limits = submitted_fixed

    candidate = HistogramDisplayState(
        revision=base.revision,
        relim_mode=relim_mode,
        log_count_axis=log_count_axis,
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


@dataclass(frozen=True, slots=True)
class HistogramViewportTransform:
    """Exact draw-frozen mapping for one interactive histogram raster."""

    display_revision: int
    plot_bounds: tuple[float, float, float, float]
    x_limits: DisplayRange
    count_limits: DisplayRange
    home_x_limits: DisplayRange
    log_count_axis: bool
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
            normalized_plot_bounds(self.plot_bounds),
        )
        if not isinstance(self.log_count_axis, bool):
            raise TypeError("log_count_axis must be bool")
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
                log_count_axis=self.log_count_axis,
            ),
        )
        if self.x_limits_are_auto and self.x_limits != self.home_x_limits:
            raise ValueError("automatic histogram x limits must equal the home range")

    def contains_widget_normalized(self, x: object, y: object) -> bool:
        return normalized_plot_contains(self.plot_bounds, x, y)

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
        count_fraction = (y - top) / (bottom - top)
        count_low, count_high = self.count_limits
        if self.log_count_axis:
            log_low = math.log(count_low)
            log_high = math.log(count_high)
            count = math.exp(log_high - count_fraction * (log_high - log_low))
        else:
            count = count_high - count_fraction * (count_high - count_low)
        return normalized_widget_x_to_data(self.plot_bounds, self.x_limits, x), count

    def data_to_widget_normalized(
        self,
        x: object,
        count: object,
    ) -> tuple[float, float]:
        x = finite_real(x, "data x")
        count = finite_real(count, "data count")
        left, top, right, bottom = self.plot_bounds
        count_low, count_high = self.count_limits
        if self.log_count_axis:
            if count <= 0.0:
                raise ValueError("data count must be positive for log count scale")
            count_fraction = (math.log(count_high) - math.log(count)) / (
                math.log(count_high) - math.log(count_low)
            )
        else:
            count_fraction = (count_high - count) / (count_high - count_low)
        return (
            data_x_to_normalized_widget(self.plot_bounds, self.x_limits, x),
            top + count_fraction * (bottom - top),
        )

    def zoomed_x_limits(self, anchor_x: object, factor: object) -> DisplayRange:
        return _zoomed_x_limits(self.x_limits, anchor_x, factor)

    def panned_x_limits(
        self,
        press_widget_x: object,
        current_widget_x: object,
        *,
        start_x_limits: DisplayRange | None = None,
    ) -> DisplayRange:
        start = (
            self.x_limits
            if start_x_limits is None
            else validated_display_range(start_x_limits, "start_x_limits")
        )
        return _panned_x_limits(
            self.plot_bounds,
            start,
            press_widget_x,
            current_widget_x,
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
    previous_log_count_axis: bool | None = None,
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
    if previous_log_count_axis is not None and not isinstance(
        previous_log_count_axis,
        bool,
    ):
        raise TypeError("previous_log_count_axis must be bool or None")

    if state.relim_mode is RelimMode.FIXED:
        assert state.fixed_count_limits is not None
        return _validated_count_limits(
            state.fixed_count_limits,
            "fixed_count_limits",
            log_count_axis=state.log_count_axis,
        )
    if state.log_count_axis:
        target = (0.5, max(3.0 * peak, 1.0))
    elif state.relim_mode is RelimMode.TIGHT:
        target = (0.0, max(1.1 * peak, 1.0))
    else:
        target = (0.0, max(1.2 * peak, 1.0))
    target = _validated_count_limits(
        target,
        "derived histogram count limits",
        log_count_axis=state.log_count_axis,
    )

    force = (
        previous_relim_mode is None
        or previous_log_count_axis is None
        or previous_relim_mode is not state.relim_mode
        or previous_log_count_axis is not state.log_count_axis
    )
    if current_count_limits is None or force:
        return target
    current = _validated_count_limits(
        current_count_limits,
        "current_count_limits",
        log_count_axis=state.log_count_axis,
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


def _histogram_projection_digest(
    requested_bin_count: int,
    bin_edges: np.ndarray,
    bin_counts: tuple[np.ndarray, ...],
) -> str:
    """Digest exact bins once on their worker-owned construction boundary."""

    digest = hashlib.sha256(b"zlc_frontend.HistogramBinProjection\0")
    digest.update(int(requested_bin_count).to_bytes(8, "little", signed=False))
    for values in (bin_edges, *bin_counts):
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(int(values.ndim).to_bytes(2, "little", signed=False))
        for size in values.shape:
            digest.update(int(size).to_bytes(8, "little", signed=False))
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False, init=False)
class HistogramBinProjection:
    """One exact-sample-bound, shared-edge display projection.

    Counts and edges are computed by this constructor and cannot be supplied
    independently by a caller.  The retained sample-array identities let the
    payload boundary prove that Agg and Qt consume the projection derived from
    those exact evaluated series without a second binning pass.  It remains a
    display value until an explicit Fit command freezes its named SAMPLE axes
    and these exact edges into the canonical terminal ``HistogramSpec``.
    """

    series_samples: tuple[np.ndarray, ...]
    bin_counts: tuple[np.ndarray, ...]
    bin_edges: np.ndarray
    requested_bin_count: int
    projection_digest: str

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
        object.__setattr__(
            self,
            "projection_digest",
            _histogram_projection_digest(bins, immutable_edges, tuple(counts)),
        )

    @classmethod
    def _from_committed_edges(
        cls,
        series_samples: tuple[np.ndarray, ...],
        *,
        bins: int,
        bin_edges: tuple[float, ...] | np.ndarray,
    ) -> HistogramBinProjection:
        """Rebuild exact saved-Fit bars from their committed boundaries."""

        values_by_series = _histogram_series(series_samples)
        requested = _histogram_bin_count(bins)
        immutable_edges = _immutable_histogram_array(
            np.asarray(bin_edges, dtype=np.float64),
            np.dtype(np.float64),
        )
        histogram_home_x_limits(immutable_edges)
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
                    "committed histogram edges did not retain every exact sample"
                )
            counts.append(
                _immutable_histogram_array(
                    np.asarray(series_counts, dtype=np.int64),
                    np.dtype(np.int64),
                )
            )
        projection = object.__new__(cls)
        object.__setattr__(projection, "series_samples", values_by_series)
        object.__setattr__(projection, "bin_counts", tuple(counts))
        object.__setattr__(projection, "bin_edges", immutable_edges)
        object.__setattr__(projection, "requested_bin_count", requested)
        object.__setattr__(
            projection,
            "projection_digest",
            _histogram_projection_digest(
                requested,
                immutable_edges,
                tuple(counts),
            ),
        )
        return projection


def _windowed_histogram_projection(
    series_samples: tuple[np.ndarray, ...],
    bins: int,
    *,
    visible_range: DisplayRange,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Build renderer-private robust bars for Main-equivalent Grid thumbnails."""

    values_by_series = _histogram_series(series_samples)
    bins = _histogram_bin_count(bins)
    low, high = validated_display_range(
        visible_range,
        "windowed histogram visible_range",
    )
    # The live Grid's visible bars intentionally retain Main's robust
    # uniformly-spaced thumbnail geometry, including boolean payloads.
    edges = _immutable_histogram_array(
        np.linspace(low, high, bins + 1, dtype=np.float64),
        np.dtype(np.float64),
    )
    visible_counts = []
    for values in values_by_series:
        histogram_values = (
            values.astype(np.uint8, copy=False)
            if values.dtype.kind == "b"
            else values
        )
        counts = np.histogram(histogram_values, bins=edges)[0]
        underflow = int(np.count_nonzero(histogram_values < edges[0]))
        overflow = int(np.count_nonzero(histogram_values > edges[-1]))
        visible = int(np.sum(counts, dtype=np.int64))
        if visible + underflow + overflow != int(values.size):
            raise RuntimeError("windowed histogram did not account for every sample")
        visible_counts.append(
            _immutable_histogram_array(
                np.asarray(counts, dtype=np.int64),
                np.dtype(np.int64),
            )
        )
    # Under/overflow are construction-time conservation facts only.  Grid
    # thumbnail bars stay display-only and never become formal Fit authority.
    return tuple(visible_counts), edges


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
    "FacetedHistogramDisplayState",
    "HistogramCellThresholds",
    "HistogramBinProjection",
    "HistogramDisplayState",
    "HistogramViewportTransform",
    "MIN_HISTOGRAM_BINS",
    "faceted_histogram_display_with_thresholds",
    "histogram_count_limits",
    "histogram_cell_thresholds_from_tree",
    "histogram_cell_thresholds_to_tree",
    "histogram_display_form_spec",
    "histogram_display_form_values",
    "histogram_display_from_form",
    "histogram_display_with_x_view",
    "histogram_home_x_limits",
    "histogram_projection_home_x_limits",
]
