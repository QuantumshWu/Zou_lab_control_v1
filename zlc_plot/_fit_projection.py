"""Projection and fit semantics shared by sessions and live workers."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

import numpy as np

from ._dataset_bridge import _ReadonlyDataset as DatasetSnapshot

from ._pulse_time import pulse_time_scale
from .config import PlotLibraryDefaults
from .data_view import CurveData, HistogramData, ImageData
from .fit import (
    FitModelSpec,
    FitParameterDisplay,
    FitResult,
    FitTarget,
    RegularImageFitInput,
    UnitRelation,
)
from .kinds import AxisRef
from .resolver import index_parameter_specs, indices_from_parameters
from .primitives import PulseTimelineData
from ._fit_scene import (
    FitOverlay,
    FitPolyline,
    FitRadialGlyph,
)
from .selectors import (
    CrosshairPoint,
    NumericRange,
    RectangleRange,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
)
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSpec,
    PulseTimelinePlot,
    RollingPlot,
)
from .state import DisplayState
from ._validation import integer, readonly_copy


class FitScope(str, Enum):
    SELECTOR = "selector"
    VIEWPORT = "viewport"
    ALL = "all"


_FIT_SELECTOR_KINDS = frozenset((
    SelectorKind.AREA,
    SelectorKind.X_RANGE,
    SelectorKind.THRESHOLD,
))

_DEFAULT_FIT_SELECTOR_PRIORITY = (
    SelectorKind.AREA,
    SelectorKind.X_RANGE,
)


@dataclass(frozen=True, slots=True)
class FitAuthority:
    selector: SelectorState | None
    viewport: RectangleRange | None
    focused_facet_index: int | None


@dataclass(frozen=True, slots=True, eq=False)
class FitSelection:
    data_revision: int
    scope: FitScope
    coordinates: tuple[np.ndarray, ...]
    observations: np.ndarray
    selected_indices: np.ndarray | None
    source_revisions: tuple[int, ...] = ()
    facet_index: int | None = None
    selector_kind: SelectorKind | None = None
    regular_image: RegularImageFitInput | None = None
    _authority: FitAuthority | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, FitScope):
            raise TypeError("fit selection scope must be FitScope")
        object.__setattr__(
            self,
            "data_revision",
            integer(
                self.data_revision,
                "fit selection data_revision",
                minimum=0,
            ),
        )
        revisions = tuple(self.source_revisions) or (self.data_revision,)
        if any(
            isinstance(revision, bool)
            or not isinstance(revision, (int, np.integer))
            or int(revision) < 0
            for revision in revisions
        ):
            raise TypeError("source_revisions must contain non-negative integers")
        object.__setattr__(self, "source_revisions", tuple(map(int, revisions)))
        regular_image = self.regular_image
        if regular_image is not None and not isinstance(
            regular_image,
            RegularImageFitInput,
        ):
            raise TypeError("regular_image must be RegularImageFitInput or None")
        object.__setattr__(
            self,
            "coordinates",
            tuple(readonly_copy(value, dtype=float) for value in self.coordinates),
        )
        if regular_image is None:
            observations = readonly_copy(self.observations, dtype=float)
        else:
            observations = np.asarray(regular_image.observations).view()
            observations.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        selected = self.selected_indices
        if selected is not None:
            selected = readonly_copy(selected, dtype=np.int64)
        elif regular_image is None:
            raise ValueError("non-image fit selections require selected_indices")
        object.__setattr__(self, "selected_indices", selected)
        object.__setattr__(
            self,
            "facet_index",
            integer(
                self.facet_index,
                "fit facet_index",
                minimum=0,
                optional=True,
            ),
        )
        selector_kind = self.selector_kind
        if selector_kind is not None:
            if not isinstance(selector_kind, SelectorKind):
                raise TypeError("fit selector_kind must be SelectorKind or None")
            if selector_kind not in _FIT_SELECTOR_KINDS:
                raise ValueError("crosshair selectors cannot define a fit")

    @property
    def sample_count(self) -> int:
        if self.regular_image is None:
            return int(np.asarray(self.observations).size)
        valid = self.regular_image.valid_mask
        return (
            int(self.regular_image.observations.size)
            if valid is None
            else int(np.count_nonzero(valid))
        )


@dataclass(frozen=True, slots=True)
class HistogramProjection:
    bin_count: int
    edges: np.ndarray

    def __post_init__(self) -> None:
        bin_count = integer(
            self.bin_count,
            "histogram projection bin_count",
            minimum=1,
        )
        edges = readonly_copy(self.edges, dtype=float).reshape(-1)
        if edges.size != bin_count + 1:
            raise ValueError("histogram projection has the wrong edge count")
        if not bool(np.all(np.isfinite(edges))) or bool(
            np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("histogram projection edges must be finite and increasing")
        object.__setattr__(self, "bin_count", bin_count)
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True, slots=True)
class ProjectionContext:
    """Immutable owner or worker state consumed by one projection operation."""

    display_state: DisplayState
    selector_snapshot: SelectorSnapshot
    viewport: RectangleRange | None = None
    focused_facet_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.display_state, DisplayState):
            raise TypeError("projection context requires DisplayState")
        if not isinstance(self.selector_snapshot, SelectorSnapshot):
            raise TypeError("projection context requires SelectorSnapshot")
        if self.viewport is not None and not isinstance(self.viewport, RectangleRange):
            raise TypeError("projection viewport must be RectangleRange or None")
        object.__setattr__(
            self,
            "focused_facet_index",
            integer(
                self.focused_facet_index,
                "focused facet index",
                minimum=0,
                optional=True,
            ),
        )

    def selector_state(self, kind: SelectorKind) -> SelectorState:
        if not isinstance(kind, SelectorKind):
            raise TypeError("selector kind must be SelectorKind")
        for state in self.selector_snapshot.states:
            if state.kind is kind:
                return state
        raise KeyError(kind)


class FitProjection:
    """Projection state shared by owner-thread sessions and frozen workers."""

    @staticmethod
    def _validate_input(
        data: DatasetSnapshot | PulseTimelineData,
        spec: PlotSpec,
    ) -> None:
        if isinstance(spec, PulseTimelinePlot):
            if not isinstance(data, PulseTimelineData):
                raise TypeError("PulseTimelinePlot requires PulseTimelineData")
        elif isinstance(
            spec,
            (CurvePlot, ImagePlot, HistogramPlot, RollingPlot, FacetGridPlot),
        ):
            if not isinstance(data, DatasetSnapshot):
                raise TypeError(f"{type(spec).__name__} requires DatasetSnapshot")
        else:
            raise TypeError("unsupported plot specification")

    def __init__(
        self,
        *,
        data: DatasetSnapshot | PulseTimelineData,
        revision: int,
        spec: PlotSpec,
        context: ProjectionContext,
        defaults: PlotLibraryDefaults,
        histogram_projection: HistogramProjection | None,
        rolling_history: tuple[CurveData, ...] = (),
    ) -> None:
        if not isinstance(context, ProjectionContext):
            raise TypeError("context must be ProjectionContext")
        self._spec = spec
        self._context = context
        self._defaults = defaults
        if histogram_projection is not None and not isinstance(
            histogram_projection,
            HistogramProjection,
        ):
            raise TypeError(
                "histogram_projection must be HistogramProjection or None"
            )
        self._histogram_projection = histogram_projection
        history = tuple(rolling_history)
        if any(not isinstance(item, CurveData) for item in history):
            raise TypeError("rolling_history must contain CurveData")
        self._rolling_history = history
        self._view = None
        self._payload = None
        selected_revision = integer(revision, "projection revision", minimum=0)
        self._validate_input(data, self._spec)
        if isinstance(data, DatasetSnapshot) and selected_revision != data.revision:
            raise ValueError("DatasetSnapshot revision must equal projection revision")
        self._data = data
        self._revision = selected_revision

    def _with_context(self, context: ProjectionContext) -> "FitProjection":
        """Return a shallow immutable-data view bound to one context snapshot."""

        if not isinstance(context, ProjectionContext):
            raise TypeError("context must be ProjectionContext")
        selected = copy(self)
        selected._context = context
        return selected

    def _fork_frozen(
        self,
        *,
        data: DatasetSnapshot | PulseTimelineData,
        revision: int,
        context: ProjectionContext,
    ) -> "FitProjection":
        """Capture immutable worker inputs using this projection's configuration."""

        return FitProjection(
            data=data,
            revision=revision,
            spec=self._spec,
            context=context,
            defaults=self._defaults,
            histogram_projection=self._histogram_projection,
            rolling_history=self._rolling_history,
        )

    def _reproject(
        self,
        *,
        context: ProjectionContext,
        payload_only: bool = False,
    ) -> None:
        """Atomically rebuild derived view/payload state for one context."""

        if not isinstance(context, ProjectionContext):
            raise TypeError("context must be ProjectionContext")
        previous = (
            self._context,
            self._view,
            self._payload,
            self._histogram_projection,
            self._rolling_history,
        )
        try:
            old_indices = {
                name: value
                for name, value in self._context.display_state.values.items()
                if name.startswith("index__")
            }
            new_indices = {
                name: value
                for name, value in context.display_state.values.items()
                if name.startswith("index__")
            }
            if isinstance(self._spec, RollingPlot) and old_indices != new_indices:
                self._rolling_history = ()
            self._context = context
            if payload_only:
                self._build_payload_from_view()
            else:
                self._build_view_and_payload()
        except BaseException:
            (
                self._context,
                self._view,
                self._payload,
                self._histogram_projection,
                self._rolling_history,
            ) = previous
            raise

    @property
    def display_state(self) -> DisplayState:
        return self._context.display_state

    @property
    def data_revision(self) -> int:
        return self._revision

    @property
    def data(self) -> DatasetSnapshot | PulseTimelineData:
        return self._data

    @property
    def viewport(self) -> RectangleRange | None:
        return self._context.viewport

    @property
    def view(self) -> Any:
        return self._view

    @property
    def payload(self) -> Any:
        return self._payload

    @property
    def _viewport(self) -> RectangleRange | None:
        return self._context.viewport

    @property
    def _focused_facet_index(self) -> int | None:
        return self._context.focused_facet_index

    def _fit_target(self) -> FitTarget | None:
        semantic = self._semantic_spec()
        if isinstance(semantic, (CurvePlot, RollingPlot)):
            return FitTarget.SERIES
        if isinstance(semantic, HistogramPlot):
            return FitTarget.HISTOGRAM
        if isinstance(semantic, ImagePlot):
            return FitTarget.IMAGE
        return None

    def _fit_model_units_compatible(self, model: FitModelSpec) -> bool:
        if self._view is None:
            return False
        try:
            sources = (
                (self._value_quantity(),)
                if self._is_histogram_plot()
                else (
                    (self._rolling_x_quantity(),)
                    if isinstance(self._spec, RollingPlot)
                    else (self._coordinate(self._x_ref()),)
                    if model.independent_arity == 1
                    else (
                        self._coordinate(self._x_ref()),
                        self._coordinate(self._y_axis_ref()),
                    )
                )
            )
            return all(
                source.canonical_unit.compatible_with(
                    self._fit_relation_quantity(relation).canonical_unit
                )
                for source, relation in zip(
                    sources,
                    model.coordinate_relations,
                    strict=True,
                )
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _require_fit_model_compatible(self, model: FitModelSpec) -> None:
        target = self._fit_target()
        if target is None or target not in model.targets:
            name = "none" if target is None else target.value
            raise ValueError(
                f"fit model {model.model_id!r} is not authored for {name} plots"
            )
        if not self._fit_model_units_compatible(model):
            raise ValueError(
                f"fit model {model.model_id!r} is incompatible with the plot axes"
            )

    def _build_view_and_payload(self) -> None:
        if not isinstance(self._data, DatasetSnapshot):
            self._view = None
            self._payload = self._data
            return
        from .data_view import DataView

        values = self.display_state.values
        overrides: dict[AxisRef, object] = {}
        x_ref = getattr(self._spec, "x", None)
        y_ref = getattr(self._spec, "y", None)
        if isinstance(self._spec, FacetGridPlot):
            x_ref = getattr(self._spec.cell, "x", None)
            y_ref = getattr(self._spec.cell, "y", None)
            facet_unit = values.get("facet_display_unit")
            if facet_unit is not None:
                overrides[self._spec.facet] = facet_unit
        if x_ref is not None and values.get("x_display_unit") is not None:
            overrides[x_ref] = values.get("x_display_unit")
        if y_ref is not None and values.get("y_display_unit") is not None:
            overrides[y_ref] = values.get("y_display_unit")
        value_unit = values.get("value_display_unit")
        self._view = DataView(
            self._data,
            axis_display_units=overrides,
            value_display_unit=value_unit,
            indices=indices_from_parameters(
                index_parameter_specs(self._data.schema, self._spec)[1],
                self.display_state.values,
            ),
        )
        self._build_payload_from_view()

    def _build_payload_from_view(self) -> None:
        """Reproject the plot payload without rebuilding unit-aware DataView state."""

        if not isinstance(self._data, DatasetSnapshot):
            self._payload = self._data
            return
        if self._view is None:
            raise RuntimeError("dataset payload projection requires a DataView")
        state = self.display_state
        if isinstance(self._spec, CurvePlot):
            group = () if self._spec.group is None else (self._spec.group,)
            self._payload = self._view.curve(
                self._spec.x,
                group_by=group,
                aggregation=self._spec.reduction,
            )
        elif isinstance(self._spec, ImagePlot):
            self._payload = self._view.image(
                self._spec.x,
                self._spec.y,
                aggregation=self._spec.reduction,
            )
        elif isinstance(self._spec, HistogramPlot):
            bins = self._histogram_bins(self._view, state)
            self._payload = self._view.histogram(bins=bins)
        elif isinstance(self._spec, RollingPlot):
            group = () if self._spec.group is None else (self._spec.group,)
            payload = self._view.rolling(
                group_by=group,
                aggregation=self._spec.reduction,
            )
            self._payload = self._append_rolling_payload(
                payload, window=int(state["window"])
            )
        elif isinstance(self._spec, FacetGridPlot):
            cell = self._spec.cell
            bins = (
                self._histogram_bins(self._view, state)
                if isinstance(cell, HistogramPlot)
                else None
            )
            self._payload = self._view.facet(
                self._spec,
                bins=bins,
            )

    def _append_rolling_payload(
        self,
        payload: CurveData,
        *,
        window: int,
    ) -> CurveData:
        """Append one compact promoted-revision sample and build the window."""

        if window <= 0:
            raise ValueError("rolling window must be positive")
        history = list(self._rolling_history)
        if history and payload.revision < history[-1].revision:
            raise ValueError("rolling source revision moved backwards")
        if history and payload.revision == history[-1].revision:
            history[-1] = payload
        else:
            history.append(payload)
        history = history[-window:]
        self._rolling_history = tuple(history)
        revisions = tuple(item.revision for item in history)
        if not history:
            return payload
        keys = tuple(series.group_key for series in history[0].series)
        if any(tuple(series.group_key for series in item.series) != keys for item in history):
            raise ValueError("rolling group domain changed within one PlotSession")
        age = np.arange(1 - len(history), 1, dtype=float)
        x_unit = history[-1].series[0].x.canonical_unit if keys else None
        prepared: list[Any] = []
        for series_index, key in enumerate(keys):
            frames = tuple(item.series[series_index] for item in history)
            template = frames[-1]
            canonical = np.asarray(
                [np.asarray(item.y.canonical).reshape(-1)[0] for item in frames]
            )
            displayed = template.y.canonical_unit.convert_value_to(
                canonical,
                template.y.display_unit,
            )
            valid = np.asarray(
                [bool(np.asarray(item.valid).reshape(-1)[0]) for item in frames],
                dtype=bool,
            )
            counts = np.asarray(
                [int(np.asarray(item.counts).reshape(-1)[0]) for item in frames],
                dtype=np.int64,
            )
            assert x_unit is not None
            x = replace(template.x, canonical=age, display=age)
            y = replace(template.y, canonical=canonical, display=displayed)
            prepared.append(
                replace(template, x=x, y=y, valid=valid, counts=counts, plot_x=age)
            )
        return CurveData(
            revision=payload.revision,
            generation=payload.generation,
            x_ref=None,
            group_by=payload.group_by,
            series=tuple(prepared),
            source_revisions=revisions,
        )

    def _histogram_bins(
        self,
        view: Any,
        state: DisplayState,
    ) -> np.ndarray:
        """Return stable display-unit edges for one histogram projection."""

        count = int(state["bin_count"])
        samples = view.samples
        canonical = np.asarray(samples.value.canonical).reshape(-1)
        valid = np.asarray(samples.valid_mask, dtype=bool).reshape(-1)
        finite = valid & np.isfinite(canonical)
        values = np.asarray(canonical[finite], dtype=float)
        previous = self._histogram_projection
        mode = str(state["relim_mode"])
        retain_domain = (
            previous is not None
            and previous.bin_count == count
            and mode != "tight"
        )
        if retain_domain:
            assert previous is not None
            low = float(previous.edges[0])
            high = float(previous.edges[-1])
            if values.size:
                data_low = float(np.min(values))
                data_high = float(np.max(values))
                if data_low < low or data_high > high:
                    envelope_low = min(low, data_low)
                    envelope_high = max(high, data_high)
                    padding = (
                        self._defaults.projection.histogram_domain_padding_fraction
                        * (envelope_high - envelope_low)
                    )
                    if data_low < low:
                        low = data_low - padding
                    if data_high > high:
                        high = data_high + padding
        else:
            if values.size:
                data_low = float(np.min(values))
                data_high = float(np.max(values))
            else:
                data_low, data_high = 0.0, 1.0
            if data_low == data_high:
                half = max(abs(data_low) * 0.05, 0.5)
                data_low -= half
                data_high += half
            low, high = data_low, data_high
            if mode != "tight":
                padding = (
                    self._defaults.projection.histogram_domain_padding_fraction
                    * (high - low)
                )
                low -= padding
                high += padding

        selected = HistogramProjection(
            count,
            np.linspace(low, high, count + 1, dtype=float),
        )
        if previous is None or not (
            previous.bin_count == selected.bin_count
            and np.array_equal(previous.edges, selected.edges)
        ):
            previous = selected
            self._histogram_projection = previous
        assert previous is not None
        return np.asarray(
            samples.value.canonical_unit.convert_value_to(
                previous.edges,
                samples.value.display_unit,
            ),
            dtype=float,
        )

    def _facet_mask(self, facet_index: int | None = None) -> np.ndarray:
        if self._view is None:
            raise TypeError("facet masking requires DatasetSnapshot")
        if not isinstance(self._spec, FacetGridPlot):
            return np.ones(self._view.samples.shape, dtype=bool)
        cells = tuple(getattr(self._payload, "cells", ()))
        selected = self._focused_facet_index if facet_index is None else facet_index
        if selected is None or selected < 0 or selected >= len(cells):
            raise IndexError("facet index is outside the current grid")
        target = cells[selected].facet_value_canonical
        coordinate = np.asarray(self._coordinate(self._spec.facet).canonical)
        return np.asarray(np.equal(coordinate, target), dtype=bool)

    def _focused_payload(self, facet_index: int | None = None) -> Any:
        if not isinstance(self._spec, FacetGridPlot):
            return self._payload
        cells = tuple(getattr(self._payload, "cells", ()))
        selected = self._focused_facet_index if facet_index is None else facet_index
        if selected is None or selected < 0 or selected >= len(cells):
            raise IndexError("facet index is outside the current grid")
        return cells[selected].payload

    def _rolling_visible_mask(self) -> np.ndarray:
        if self._view is None:
            raise TypeError("rolling masking requires DatasetSnapshot")
        if not isinstance(self._spec, RollingPlot):
            return np.ones(self._view.samples.shape, dtype=bool)
        # Rolling history is plot-private and no longer aliases a Dataset
        # coordinate.  Dataset masks apply only to the current compact source
        # sample; history-domain selection is handled on CurveData itself.
        return np.ones(self._view.samples.shape, dtype=bool)

    def _crosshair_sample_mask(
        self,
        state: SelectorState,
        valid: np.ndarray,
        point_transform: Callable[[np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        """Materialize the nearest valid plotted sample in display space."""

        if self._view is None or not isinstance(state.value, CrosshairPoint):
            raise TypeError("crosshair sample lookup requires DatasetSnapshot")
        displayed = self._display_selector_state(state)
        assert isinstance(displayed.value, CrosshairPoint)
        target = displayed.value
        samples = self._view.samples
        semantic = self._semantic_spec()
        candidate = np.asarray(valid, dtype=bool) & np.isfinite(
            np.asarray(samples.value.canonical, dtype=float)
        )
        candidate &= self._rolling_visible_mask()
        if isinstance(semantic, HistogramPlot):
            x_values = np.asarray(samples.value.display, dtype=float)
            y_values = np.full(samples.shape, target.y, dtype=float)
        else:
            x_coordinate = self._coordinate(self._x_ref())
            x_values = (
                self._coordinate_values_to_display(
                    np.asarray(x_coordinate.canonical, dtype=float),
                    self._x_ref(),
                )
                if isinstance(self._spec, RollingPlot)
                else np.asarray(x_coordinate.display, dtype=float)
            )
            y_values = (
                np.asarray(self._coordinate(self._y_axis_ref()).display, dtype=float)
                if isinstance(semantic, ImagePlot)
                else np.asarray(samples.value.display, dtype=float)
            )
        candidate &= np.isfinite(x_values) & np.isfinite(y_values)
        flat_indices = np.flatnonzero(candidate.reshape(-1))
        result = np.zeros(samples.shape, dtype=bool)
        if flat_indices.size == 0:
            return result

        points = np.column_stack(
            (
                x_values.reshape(-1)[flat_indices],
                y_values.reshape(-1)[flat_indices],
            )
        )
        target_point = np.asarray((target.x, target.y), dtype=float)
        if point_transform is not None:
            try:
                transformed = np.asarray(
                    point_transform(np.vstack((points, target_point))),
                    dtype=float,
                )
                if transformed.shape != (points.shape[0] + 1, 2):
                    raise ValueError("point transform returned the wrong shape")
                points, target_point = transformed[:-1], transformed[-1]
            except (TypeError, ValueError):
                pass
        finite = np.all(np.isfinite(points), axis=1)
        if not np.any(finite):
            return result
        flat_indices = flat_indices[finite]
        delta = np.abs(points[finite] - target_point)
        if isinstance(semantic, ImagePlot):
            nearest = int(np.argmin(np.hypot(delta[:, 0], delta[:, 1])))
        else:
            nearest = int(np.lexsort((delta[:, 1], delta[:, 0]))[0])
        result.reshape(-1)[flat_indices[nearest]] = True
        return result

    def _selector_mask(
        self,
        state: SelectorState,
        *,
        point_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        samples = self._view.samples
        mask = np.zeros(samples.shape, dtype=bool)
        selected_positions = self._view.selected_positions()
        mask.reshape(-1)[selected_positions] = np.asarray(
            samples.valid_mask
        ).reshape(-1)[selected_positions]
        mask &= self._facet_mask(state.facet_index)
        value = np.asarray(samples.value.canonical)
        if state.kind is SelectorKind.X_RANGE:
            source = self._x_selector_source()
            coordinate = np.asarray(
                self._coordinate(source).canonical
                if isinstance(source, AxisRef)
                else source.canonical
            )
            assert isinstance(state.value, NumericRange)
            mask &= (coordinate >= state.value.low) & (coordinate <= state.value.high)
        elif state.kind is SelectorKind.AREA:
            assert isinstance(state.value, RectangleRange)
            x_source = self._x_selector_source()
            x = np.asarray(
                self._coordinate(x_source).canonical
                if isinstance(x_source, AxisRef)
                else x_source.canonical
            )
            mask &= (x >= state.value.x.low) & (x <= state.value.x.high)
            semantic = self._semantic_spec()
            if isinstance(semantic, ImagePlot):
                y = np.asarray(
                    self._coordinate(self._y_axis_ref()).canonical
                )
                mask &= (y >= state.value.y.low) & (y <= state.value.y.high)
            elif not isinstance(semantic, HistogramPlot):
                # A one-dimensional curve's vertical display coordinate is
                # the observation itself.  AREA therefore filters both the
                # independent coordinate and the canonical observation; this
                # The raw-sample mask is materialised only by selector_data().
                mask &= (value >= state.value.y.low) & (value <= state.value.y.high)
        elif state.kind is SelectorKind.THRESHOLD:
            mask &= value >= float(state.value)
        elif state.kind is SelectorKind.CROSSHAIR:
            return self._crosshair_sample_mask(state, mask, point_transform)
        return mask & np.isfinite(value)

    def _fit_selector(
        self,
        selector_kind: SelectorKind | None = None,
    ) -> SelectorState | None:
        if selector_kind is not None and not isinstance(selector_kind, SelectorKind):
            raise TypeError("selector_kind must be SelectorKind or None")
        if (
            selector_kind is not None
            and selector_kind not in _FIT_SELECTOR_KINDS
        ):
            raise ValueError("crosshair selectors cannot define a fit")

        states = {
            state.kind: state for state in self._context.selector_snapshot.states
        }

        def usable(state: SelectorState | None) -> bool:
            return bool(
                state is not None
                and state.kind in _FIT_SELECTOR_KINDS
                and state.facet_index == self._focused_facet_index
            )

        if selector_kind is not None:
            selected = states.get(selector_kind)
            if selected is None:
                raise KeyError(selector_kind)
            if not usable(selected):
                raise ValueError(
                    "fit selector must belong to the focused facet and contain "
                    "a numeric selection"
                )
            return selected

        for kind in _DEFAULT_FIT_SELECTOR_PRIORITY:
            selected = states.get(kind)
            if usable(selected):
                return selected
        return None

    def _fit_selection_authority(
        self,
        selector_kind: SelectorKind | None,
    ) -> FitAuthority:
        """Resolve the selector-or-viewport precedence once for all fit paths."""

        selector = self._fit_selector(selector_kind)
        viewport = None
        if selector is None and self._viewport is not None:
            viewport = (
                self._viewport_in_canonical()
                if self._view is not None
                else self._viewport
            )
        return FitAuthority(
            selector,
            viewport,
            self._focused_facet_index,
        )

    def fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
    ) -> FitSelection:
        if self._view is None:
            raise TypeError("fit is available only for DatasetSnapshot plots")
        if not isinstance(model, FitModelSpec):
            raise TypeError("model must be FitModelSpec")
        self._require_fit_model_compatible(model)
        if isinstance(self._spec, RollingPlot):
            self._rolling_fit_scale()
        payload = self._focused_payload(self._focused_facet_index)
        if isinstance(payload, CurveData):
            return self._curve_fit_selection(
                model,
                selector_kind=selector_kind,
                payload=payload,
            )
        if (
            isinstance(payload, ImageData)
            and model.model_id == "radial_gaussian_center"
        ):
            return self._regular_image_fit_selection(
                model,
                selector_kind=selector_kind,
                payload=payload,
            )
        if self._is_histogram_plot():
            if not isinstance(payload, HistogramData):
                raise RuntimeError("histogram projection did not produce histogram data")
            return self._histogram_fit_selection(
                model,
                selector_kind=selector_kind,
                payload=payload,
            )
        if not isinstance(payload, ImageData):
            raise TypeError("unsupported fit projection payload")
        return self._image_fit_selection(
            model,
            selector_kind=selector_kind,
            payload=payload,
        )

    def _all_facet_fit_selections(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
    ) -> tuple[tuple[Any, ...], tuple[FitSelection, ...]]:
        """Freeze every painted FacetData cell through the ordinary fit path."""

        if not isinstance(self._spec, FacetGridPlot):
            raise TypeError("fit_all_facets requires FacetGridPlot")
        cells = tuple(getattr(self._payload, "cells", ()))
        if not cells:
            raise ValueError("cannot fit an empty FacetGrid")
        values: list[Any] = []
        selections: list[FitSelection] = []
        for index, cell in enumerate(cells):
            # One authored selector geometry applies uniformly to all cells;
            # its facet address is presentation ownership, not a different
            # selection algorithm.  Re-address it before invoking the exact
            # same focused-cell projection/packing implementation.
            selectors = SelectorSnapshot(
                tuple(
                    replace(state, facet_index=index)
                    if state.facet_index is not None
                    else state
                    for state in self._context.selector_snapshot.committed
                ),
                (
                    None
                    if self._context.selector_snapshot.candidate is None
                    else replace(
                        self._context.selector_snapshot.candidate,
                        facet_index=index,
                    )
                ),
            )
            projection = self._with_context(replace(
                self._context,
                selector_snapshot=selectors,
                focused_facet_index=index,
            ))
            values.append(cell.facet_value_canonical)
            selections.append(projection.fit_selection(
                model,
                selector_kind=selector_kind,
            ))
        return tuple(values), tuple(selections)

    def _curve_fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        payload: CurveData,
    ) -> FitSelection:
        """Fit the first painted series, with scope applied to that series."""

        if self._view is None:
            raise TypeError("curve fitting requires DatasetSnapshot")
        if model.independent_arity != 1:
            raise ValueError("curve fit models require exactly one independent axis")
        series = tuple(payload.series)
        if not series:
            raise ValueError("painted curve has no series")
        source = series[0]
        x_canonical = np.asarray(source.x.canonical, dtype=float).reshape(-1)
        y_canonical = np.asarray(source.y.canonical, dtype=float).reshape(-1)
        valid = (
            np.asarray(source.valid, dtype=bool).reshape(-1)
            & np.isfinite(x_canonical)
            & np.isfinite(y_canonical)
        )

        authority = self._fit_selection_authority(selector_kind)
        active = authority.selector
        if active is not None:
            value = active.value
            if active.kind is SelectorKind.X_RANGE:
                assert isinstance(value, NumericRange)
                valid &= (x_canonical >= value.low) & (x_canonical <= value.high)
            elif active.kind is SelectorKind.AREA:
                assert isinstance(value, RectangleRange)
                valid &= (x_canonical >= value.x.low) & (
                    x_canonical <= value.x.high
                )
                valid &= (y_canonical >= value.y.low) & (
                    y_canonical <= value.y.high
                )
            elif active.kind is SelectorKind.THRESHOLD:
                valid &= y_canonical >= float(value)
            else:
                raise ValueError("selected geometry cannot define a curve fit domain")
            scope = FitScope.SELECTOR
        elif authority.viewport is not None:
            viewport = authority.viewport
            valid &= (x_canonical >= viewport.x.low) & (
                x_canonical <= viewport.x.high
            )
            valid &= (y_canonical >= viewport.y.low) & (
                y_canonical <= viewport.y.high
            )
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL

        indices = np.flatnonzero(valid)
        coordinates = self._fit_coordinate_values_to_solver(
            x_canonical[valid],
            source.x,
            model.coordinate_relations[0],
        )
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(coordinates,),
            observations=y_canonical[valid],
            selected_indices=indices,
            source_revisions=(
                tuple(
                    revision
                    for revision, selected in zip(
                        payload.source_revisions,
                        valid,
                        strict=True,
                    )
                    if selected
                )
                if isinstance(self._spec, RollingPlot)
                else (self.data_revision,)
            ),
            facet_index=self._focused_facet_index,
            selector_kind=None if active is None else active.kind,
            _authority=authority,
        )

    def _histogram_fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        payload: HistogramData,
    ) -> FitSelection:
        """Fit the exact bins painted by the current histogram projection."""

        if bool(self.display_state["density"]) or bool(
            self.display_state["cumulative"]
        ):
            raise ValueError(
                "histogram fitting requires count projection; set density=False "
                "and cumulative=False"
            )
        canonical = np.asarray(payload.centers.canonical, dtype=float).reshape(-1)
        counts = np.asarray(payload.counts, dtype=float).reshape(-1)
        valid = np.isfinite(canonical) & np.isfinite(counts)

        authority = self._fit_selection_authority(selector_kind)
        active = authority.selector
        if active is not None:
            if active.kind is SelectorKind.X_RANGE:
                value = active.value
                assert isinstance(value, NumericRange)
                valid &= (canonical >= value.low) & (canonical <= value.high)
            elif active.kind is SelectorKind.AREA:
                value = active.value
                assert isinstance(value, RectangleRange)
                valid &= (canonical >= value.x.low) & (canonical <= value.x.high)
                valid &= (counts >= value.y.low) & (counts <= value.y.high)
            elif active.kind is SelectorKind.THRESHOLD:
                valid &= canonical >= float(active.value)
            else:
                raise ValueError(
                    "selected geometry cannot define a histogram fit domain"
                )
            scope = FitScope.SELECTOR
        elif authority.viewport is not None:
            viewport = authority.viewport
            valid &= (canonical >= viewport.x.low) & (canonical <= viewport.x.high)
            valid &= (counts >= viewport.y.low) & (counts <= viewport.y.high)
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL

        indices = np.flatnonzero(valid)
        model_centers = self._fit_coordinate_values_to_solver(
            canonical[valid],
            payload.centers,
            model.coordinate_relations[0],
        )
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(model_centers,),
            observations=counts[valid],
            selected_indices=indices,
            facet_index=self._focused_facet_index,
            selector_kind=None if active is None else active.kind,
            _authority=authority,
        )

    def _regular_image_fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        payload: ImageData,
    ) -> FitSelection:
        """Freeze a regular image through axis slices and compact validity."""

        x_canonical = np.asarray(payload.x.canonical, dtype=float).reshape(-1)
        y_canonical = np.asarray(payload.y.canonical, dtype=float).reshape(-1)
        authority = self._fit_selection_authority(selector_kind)
        active = authority.selector
        x_slice = slice(None)
        y_slice = slice(None)
        minimum_observation: float | None = None

        def axis_slice(
            coordinates: np.ndarray,
            low: float,
            high: float,
        ) -> slice:
            positions = np.flatnonzero(
                (coordinates >= low) & (coordinates <= high)
            )
            if positions.size == 0:
                return slice(0, 0)
            return slice(int(positions[0]), int(positions[-1]) + 1)

        if active is not None:
            value = active.value
            if active.kind is SelectorKind.X_RANGE:
                assert isinstance(value, NumericRange)
                x_slice = axis_slice(x_canonical, value.low, value.high)
            elif active.kind is SelectorKind.AREA:
                assert isinstance(value, RectangleRange)
                x_slice = axis_slice(x_canonical, value.x.low, value.x.high)
                y_slice = axis_slice(y_canonical, value.y.low, value.y.high)
            elif active.kind is SelectorKind.THRESHOLD:
                minimum_observation = float(value)
            else:
                raise ValueError("selected geometry cannot define an image fit domain")
            scope = FitScope.SELECTOR
        elif authority.viewport is not None:
            viewport = authority.viewport
            x_slice = axis_slice(
                x_canonical,
                viewport.x.low,
                viewport.x.high,
            )
            y_slice = axis_slice(
                y_canonical,
                viewport.y.low,
                viewport.y.high,
            )
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL

        x_solver = self._fit_coordinate_values_to_solver(
            x_canonical[x_slice],
            payload.x,
            model.coordinate_relations[0],
        ).reshape(-1)
        y_solver = self._fit_coordinate_values_to_solver(
            y_canonical[y_slice],
            payload.y,
            model.coordinate_relations[1],
        ).reshape(-1)
        observations = np.asarray(payload.z.canonical)[y_slice, x_slice]
        valid = np.asarray(payload.valid, dtype=bool)[y_slice, x_slice]
        valid_mask = valid
        if valid.size and all(stride == 0 for stride in valid.strides):
            valid_mask = None if bool(valid.reshape(-1)[0]) else valid

        regular = RegularImageFitInput(
            x_solver,
            y_solver,
            observations,
            valid_mask=valid_mask,
            minimum_observation=minimum_observation,
        )
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(x_solver, y_solver),
            observations=observations,
            selected_indices=None,
            facet_index=self._focused_facet_index,
            selector_kind=None if active is None else active.kind,
            regular_image=regular,
            _authority=authority,
        )

    def _image_fit_selection(
        self,
        model: FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        payload: ImageData,
    ) -> FitSelection:
        """Fit the painted image projection without returning to raw samples."""

        if model.independent_arity != 2:
            raise ValueError("image fit models require exactly two independent axes")
        x_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.x.canonical, dtype=float),
            payload.x,
            model.coordinate_relations[0],
        ).reshape(-1)
        y_solver = self._fit_coordinate_values_to_solver(
            np.asarray(payload.y.canonical, dtype=float),
            payload.y,
            model.coordinate_relations[1],
        ).reshape(-1)
        valid, observations, scope, active, authority = self._image_fit_domain(
            payload,
            selector_kind,
        )
        valid &= np.isfinite(x_solver)[None, :] & np.isfinite(y_solver)[:, None]
        selected = valid.reshape(-1)
        x_solver_grid = np.broadcast_to(x_solver[None, :], observations.shape)
        y_solver_grid = np.broadcast_to(y_solver[:, None], observations.shape)
        return FitSelection(
            data_revision=self.data_revision,
            scope=scope,
            coordinates=(
                x_solver_grid.reshape(-1)[selected],
                y_solver_grid.reshape(-1)[selected],
            ),
            observations=observations.reshape(-1)[selected],
            selected_indices=np.flatnonzero(selected),
            facet_index=self._focused_facet_index,
            selector_kind=None if active is None else active.kind,
            _authority=authority,
        )

    def _image_fit_domain(
        self,
        payload: ImageData,
        selector_kind: SelectorKind | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        FitScope,
        SelectorState | None,
        FitAuthority,
    ]:
        """Resolve one canonical mask over the already projected image."""

        x = np.asarray(payload.x.canonical, dtype=float).reshape(-1)
        y = np.asarray(payload.y.canonical, dtype=float).reshape(-1)
        observations = np.asarray(payload.z.canonical)
        valid = (
            np.asarray(payload.valid, dtype=bool)
            & np.isfinite(observations)
            & np.isfinite(x)[None, :]
            & np.isfinite(y)[:, None]
        )
        authority = self._fit_selection_authority(selector_kind)
        active = authority.selector
        if active is not None:
            value = active.value
            if active.kind is SelectorKind.X_RANGE:
                assert isinstance(value, NumericRange)
                valid &= (x[None, :] >= value.low) & (x[None, :] <= value.high)
            elif active.kind is SelectorKind.AREA:
                assert isinstance(value, RectangleRange)
                valid &= (x[None, :] >= value.x.low) & (
                    x[None, :] <= value.x.high
                )
                valid &= (y[:, None] >= value.y.low) & (
                    y[:, None] <= value.y.high
                )
            elif active.kind is SelectorKind.THRESHOLD:
                valid &= observations >= float(value)
            else:
                raise ValueError("selected geometry cannot define an image fit domain")
            scope = FitScope.SELECTOR
        elif authority.viewport is not None:
            viewport = authority.viewport
            valid &= (x[None, :] >= viewport.x.low) & (
                x[None, :] <= viewport.x.high
            )
            valid &= (y[:, None] >= viewport.y.low) & (
                y[:, None] <= viewport.y.high
            )
            scope = FitScope.VIEWPORT
        else:
            scope = FitScope.ALL
        return valid, observations, scope, active, authority

    def _viewport_in_canonical(self) -> RectangleRange:
        assert self._viewport is not None
        return RectangleRange(
            self._display_range_to_canonical(
                self._viewport.x, self._x_selector_source()
            ),
            self._viewport.y
            if self._is_histogram_plot()
            else self._display_range_to_canonical(
                self._viewport.y, self._y_ref_or_value()
            ),
        )

    def _fit_relation_quantity(self, relation: UnitRelation) -> Any:
        """Resolve one model unit relation to the plot's authoritative quantity."""

        if relation is UnitRelation.VALUE:
            return self._value_quantity()
        if relation is UnitRelation.AXIS_0:
            return (
                self._value_quantity()
                if self._is_histogram_plot()
                else self._rolling_x_quantity()
                if isinstance(self._spec, RollingPlot)
                else self._coordinate(self._x_ref())
            )
        if relation is UnitRelation.AXIS_1:
            return self._coordinate(self._y_axis_ref())
        return None

    def _fit_coordinate_values_to_solver(
        self,
        values: np.ndarray,
        source_quantity: Any,
        solver_relation: UnitRelation,
    ) -> np.ndarray:
        """Convert a painted coordinate's canonical values into model units."""

        target_quantity = self._fit_relation_quantity(solver_relation)
        if target_quantity is None:
            raise ValueError(
                "fit coordinate relations must identify a plot coordinate axis"
            )
        source_unit = source_quantity.canonical_unit
        target_unit = target_quantity.canonical_unit
        if not source_unit.compatible_with(target_unit):
            raise ValueError(
                "fit model coordinate axes require compatible canonical units"
            )
        return np.asarray(
            source_unit.convert_value_to(values, target_unit),
            dtype=float,
        )

    def _fit_overlay_curve_domain(
        self,
        result: FitResult,
        selection: FitSelection,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the full painted one-dimensional domain in solver/display units."""

        payload = self._focused_payload(selection.facet_index)
        if self._is_histogram_plot() and hasattr(payload, "centers"):
            centers = payload.centers
            canonical = self._fit_coordinate_values_to_solver(
                np.asarray(centers.canonical, dtype=float).reshape(-1),
                centers,
                result.model.coordinate_relations[0],
            )
            return canonical, np.asarray(centers.display, dtype=float).reshape(-1)

        series = tuple(getattr(payload, "series", ()))
        if not series:
            raise RuntimeError("one-dimensional fit overlay requires a painted series")
        if isinstance(self._spec, RollingPlot):
            rolling_canonical, rolling_display = self._rolling_axis_domains()
            canonical = self._fit_coordinate_values_to_solver(
                np.asarray(rolling_canonical, dtype=float).reshape(-1),
                series[0].x,
                result.model.coordinate_relations[0],
            )
            return canonical, np.asarray(rolling_display, dtype=float).reshape(-1)

        x = series[0].x
        canonical = self._fit_coordinate_values_to_solver(
            np.asarray(x.canonical, dtype=float).reshape(-1),
            x,
            result.model.coordinate_relations[0],
        )
        return canonical, np.asarray(x.display, dtype=float).reshape(-1)

    def _fit_solver_coordinate_to_display(
        self,
        values: np.ndarray,
        solver_relation: UnitRelation,
        display_relation: UnitRelation,
    ) -> np.ndarray:
        source_quantity = self._fit_relation_quantity(solver_relation)
        target_quantity = self._fit_relation_quantity(display_relation)
        if source_quantity is None or target_quantity is None:
            raise ValueError("fit coordinate display requires physical axis relations")
        if isinstance(self._spec, RollingPlot) and (
            solver_relation is UnitRelation.AXIS_0
            and display_relation is UnitRelation.AXIS_0
        ):
            canonical, displayed = self._rolling_axis_domains()
            order = np.argsort(canonical)
            return np.asarray(
                np.interp(values, canonical[order], displayed[order]),
                dtype=float,
            )
        source_unit = source_quantity.canonical_unit
        target_unit = target_quantity.display_unit
        if not source_unit.compatible_with(target_unit):
            raise ValueError("fit coordinate solver and display units are incompatible")
        return np.asarray(
            source_unit.convert_value_to(values, target_unit),
            dtype=float,
        )

    @staticmethod
    def _fit_crossover_coordinate(
        coordinates: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        x = np.asarray(coordinates, dtype=float).reshape(-1)
        left = np.asarray(first, dtype=float).reshape(-1)
        right = np.asarray(second, dtype=float).reshape(-1)
        finite = np.isfinite(x) & np.isfinite(left) & np.isfinite(right)
        x, left, right = x[finite], left[finite], right[finite]
        if x.size == 0:
            raise ValueError("fit crossover requires finite component curves")
        difference = left - right
        crossings = np.flatnonzero(
            (difference[:-1] == 0.0)
            | (difference[1:] == 0.0)
            | (np.signbit(difference[:-1]) != np.signbit(difference[1:]))
        )
        if crossings.size:
            strength = np.minimum(
                np.abs(left[crossings]) + np.abs(right[crossings]),
                np.abs(left[crossings + 1]) + np.abs(right[crossings + 1]),
            )
            index = int(crossings[int(np.argmax(strength))])
            denominator = difference[index + 1] - difference[index]
            fraction = (
                0.5
                if denominator == 0.0
                else float(np.clip(-difference[index] / denominator, 0.0, 1.0))
            )
            return float(x[index] + fraction * (x[index + 1] - x[index]))
        scale = np.abs(left) + np.abs(right) + np.finfo(float).eps
        return float(x[int(np.argmin(np.abs(difference) / scale))])

    def _fit_overlay_polylines(
        self,
        result: FitResult,
        selection: FitSelection,
    ) -> tuple[tuple[FitPolyline, ...], float | None]:
        if not result.success or result.model.independent_arity != 1:
            return (), None
        presentation = result.model.presentation
        canonical, display_x = self._fit_overlay_curve_domain(result, selection)
        if not presentation.components:
            fitted = result.model.evaluate(
                (canonical,),
                result.parameter_values,
            ).reshape(-1)
            fitted_display = (
                fitted
                if self._is_histogram_plot()
                else self._convert_coordinate_array_to_display(
                    fitted,
                    self._value_quantity(),
                )
            )
            role = "total" if self._is_histogram_plot() else "primary"
            return (
                FitPolyline(display_x, fitted_display, role=role),
            ), None

        source = np.asarray(canonical, dtype=float).reshape(-1)
        finite = source[np.isfinite(source)]
        if finite.size < 2:
            return (), None
        sample_count = self._defaults.style.artists.fit_component_sample_count
        dense = np.linspace(float(np.min(finite)), float(np.max(finite)), sample_count)
        display_x = self._fit_solver_coordinate_to_display(
            dense,
            result.model.coordinate_relations[0],
            UnitRelation.AXIS_0,
        )
        component_values: dict[str, np.ndarray] = {}
        for component in presentation.components:
            component_values[component.component_id] = (
                result.model.evaluate_component(
                    component.component_id,
                    (dense,),
                    result.parameter_values,
                ).reshape(-1)
            )
        converted_components = {
            name: (
                values
                if self._is_histogram_plot()
                else self._convert_coordinate_array_to_display(
                    values,
                    self._value_quantity(),
                )
            )
            for name, values in component_values.items()
        }
        ordered_components = tuple(
            component_values[component.component_id]
            for component in presentation.components
        )
        total = ordered_components[0].copy()
        for component_values_array in ordered_components[1:]:
            total += component_values_array
        if not self._is_histogram_plot():
            total = self._convert_coordinate_array_to_display(
                total,
                self._value_quantity(),
            )
        polylines = tuple(
            FitPolyline(
                display_x,
                converted_components[component.component_id],
                role="component",
                component_index=index,
            )
            for index, component in enumerate(presentation.components)
        ) + (FitPolyline(display_x, total, role="total"),)
        crossover = presentation.crossover_components
        if crossover is None:
            return polylines, None
        canonical_threshold = self._fit_crossover_coordinate(
            dense,
            component_values[crossover[0]],
            component_values[crossover[1]],
        )
        threshold = float(
            self._fit_solver_coordinate_to_display(
                np.asarray((canonical_threshold,), dtype=float),
                result.model.coordinate_relations[0],
                UnitRelation.AXIS_0,
            )[0]
        )
        return polylines, threshold

    def _fit_overlay_glyph(
        self,
        result: FitResult,
        parameter_display: tuple[FitParameterDisplay, ...],
    ) -> FitRadialGlyph | None:
        glyph = result.model.presentation.radial_glyph
        if glyph is None or not result.success:
            return None
        center_indices = tuple(
            result.model.parameter_index(name) for name in glyph.center_parameters
        )
        radius_index = result.model.parameter_index(glyph.radius_parameter)
        center_x = parameter_display[center_indices[0]].value
        center_y = parameter_display[center_indices[1]].value
        radius_x = abs(parameter_display[radius_index].value)
        radius_spec = result.model.parameters[radius_index]
        radius_y, _unit = self._display_fit_parameter_value(
            radius_spec,
            float(result.parameter_values[radius_index]),
            difference=True,
            display_relation=UnitRelation.AXIS_1,
        )
        return FitRadialGlyph(center_x, center_y, radius_x, abs(radius_y))

    def _make_fit_overlay(
        self,
        result: FitResult,
        selection: FitSelection,
    ) -> FitOverlay:
        parameter_display = self._display_fit_parameters(result)
        polylines, suggested_threshold = self._fit_overlay_polylines(
            result,
            selection,
        )
        return FitOverlay(
            polylines=polylines,
            radial_glyph=self._fit_overlay_glyph(result, parameter_display),
            suggested_threshold=suggested_threshold,
            success=result.success,
            formula=result.model.formula or "",
            parameter_display=parameter_display,
            diagnostic=result.message,
            facet_index=selection.facet_index,
        )

    def _display_fit_parameters(
        self,
        result: FitResult,
    ) -> tuple[FitParameterDisplay, ...]:
        """Convert fit values and uncertainties into the painted units."""

        rows: list[FitParameterDisplay] = []
        for index, (spec, raw) in enumerate(zip(
            result.model.parameters,
            result.parameter_values,
            strict=True,
        )):
            value, unit = self._display_fit_parameter_value(spec, float(raw))
            error = None
            if result.covariance_valid:
                error, _error_unit = self._display_fit_parameter_value(
                    spec,
                    float(result.standard_errors[index]),
                    difference=True,
                )
                error = abs(error)
            rows.append(
                FitParameterDisplay(
                    name=spec.name,
                    label=spec.display_label or spec.name,
                    value=value,
                    standard_error=error,
                    unit=unit,
                )
            )
        return tuple(rows)

    def _display_fit_parameter_value(
        self,
        spec: Any,
        value: float,
        *,
        difference: bool = False,
        display_relation: UnitRelation | None = None,
    ) -> tuple[float, str]:
        relation = spec.unit_relation if display_relation is None else display_relation
        solver_relation = spec.solver_unit_relation
        if relation is UnitRelation.RADIAN:
            if solver_relation is not UnitRelation.RADIAN:
                raise ValueError("radian display requires a radian solver parameter")
            return value, "rad"
        if relation is UnitRelation.VALUE and self._is_histogram_plot():
            if solver_relation is not UnitRelation.VALUE:
                raise ValueError("histogram count parameters require value solver units")
            return value, "count"
        if isinstance(self._spec, RollingPlot) and relation in {
            UnitRelation.AXIS_0,
            UnitRelation.INVERSE_AXIS_0,
        }:
            if solver_relation is not relation:
                raise ValueError("rolling fit parameters cannot cross unit relations")
            canonical, display = self._rolling_axis_domains()
            if canonical.size <= 1:
                return value, "point"
            scale = self._rolling_fit_scale()
            if relation is UnitRelation.INVERSE_AXIS_0:
                return value / scale, "1/point"
            if spec.affine_point and not difference:
                order = np.argsort(canonical)
                return float(np.interp(value, canonical[order], display[order])), "point"
            return value * scale, "point"

        if relation is UnitRelation.INVERSE_AXIS_0:
            if solver_relation is not UnitRelation.INVERSE_AXIS_0:
                raise ValueError("inverse-axis parameters cannot cross unit relations")
            quantity = self._fit_relation_quantity(UnitRelation.AXIS_0)
            if quantity is None:
                return value, ""
            canonical_unit = quantity.canonical_unit
            display_unit = quantity.display_unit
            converted = value * float(display_unit.scale) / float(canonical_unit.scale)
            return converted, self._inverse_unit_symbol(display_unit.symbol)

        source_quantity = self._fit_relation_quantity(solver_relation)
        target_quantity = self._fit_relation_quantity(relation)
        if source_quantity is None or target_quantity is None:
            return value, ""
        canonical_unit = source_quantity.canonical_unit
        display_unit = target_quantity.display_unit
        if not canonical_unit.compatible_with(display_unit):
            raise ValueError("fit parameter solver and display units are incompatible")
        if spec.affine_point and not difference:
            converted = float(
                np.asarray(
                    canonical_unit.convert_value_to((value,), display_unit),
                    dtype=float,
                ).reshape(-1)[0]
            )
        else:
            converted = value * float(canonical_unit.scale) / float(display_unit.scale)
        unit = "" if display_unit.symbol == "1" else display_unit.symbol
        return converted, unit

    @staticmethod
    def _inverse_unit_symbol(symbol: str) -> str:
        return {
            "s": "Hz",
            "ms": "kHz",
            "us": "MHz",
            "µs": "MHz",
            "μs": "MHz",
            "ns": "GHz",
        }.get(symbol, "" if symbol == "1" else f"1/{symbol}")

    def _fit_parameter_canonical_units(
        self,
        model: FitModelSpec,
    ) -> tuple[str | None, ...]:
        """Return the physical solver unit of every published parameter."""

        units: list[str | None] = []
        for parameter in model.parameters:
            relation = parameter.solver_unit_relation
            if relation is UnitRelation.RADIAN:
                symbol = "rad"
            elif relation is UnitRelation.VALUE and self._is_histogram_plot():
                symbol = "count"
            elif isinstance(self._spec, RollingPlot) and relation in {
                UnitRelation.AXIS_0,
                UnitRelation.INVERSE_AXIS_0,
            }:
                symbol = (
                    "point"
                    if relation is UnitRelation.AXIS_0
                    else "1/point"
                )
            elif relation is UnitRelation.INVERSE_AXIS_0:
                quantity = self._fit_relation_quantity(UnitRelation.AXIS_0)
                symbol = (
                    ""
                    if quantity is None
                    else self._inverse_unit_symbol(
                        quantity.canonical_unit.symbol
                    )
                )
            else:
                quantity = self._fit_relation_quantity(relation)
                symbol = "" if quantity is None else quantity.canonical_unit.symbol
            units.append(None if symbol in {"", "1"} else symbol)
        return tuple(units)

    def _semantic_spec(self) -> Any:
        return self._spec.cell if isinstance(self._spec, FacetGridPlot) else self._spec

    def _is_histogram_plot(self) -> bool:
        return isinstance(self._semantic_spec(), HistogramPlot)

    def _x_ref(self) -> AxisRef:
        semantic = self._semantic_spec()
        ref = getattr(semantic, "x", None)
        if not isinstance(ref, AxisRef):
            raise TypeError("this plot has no coordinate x axis")
        return ref

    def _x_selector_source(self) -> AxisRef | Any:
        if self._is_histogram_plot():
            return self._value_quantity()
        if isinstance(self._spec, RollingPlot):
            return self._rolling_x_quantity()
        return self._x_ref()

    def _y_axis_ref(self) -> AxisRef:
        semantic = self._semantic_spec()
        ref = getattr(semantic, "y", None)
        if not isinstance(ref, AxisRef):
            raise TypeError("the selected fit model requires a plot y-coordinate axis")
        return ref

    def _coordinate(self, ref: AxisRef) -> Any:
        if self._view is None:
            raise TypeError("coordinate access requires DatasetSnapshot")
        return self._view.coordinate(ref)

    def _value_quantity(self) -> Any:
        if self._view is None:
            raise TypeError("value access requires DatasetSnapshot")
        return self._view.samples.value

    def _y_ref_or_value(self) -> AxisRef | Any:
        semantic = self._semantic_spec()
        ref = getattr(semantic, "y", None)
        return ref if isinstance(ref, AxisRef) else self._value_quantity()

    def _display_scalar_to_canonical(
        self, value: float, source: AxisRef | Any
    ) -> float:
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        unit = quantity.display_unit
        return float(np.asarray(unit.to_canonical([value])).reshape(-1)[0])

    def _canonical_scalar_to_display(
        self, value: float, source: AxisRef | Any
    ) -> float:
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        canonical = quantity.canonical_unit
        display = quantity.display_unit
        return float(np.asarray(canonical.convert_value_to([value], display)).reshape(-1)[0])

    def _display_range_to_canonical(
        self, value: NumericRange, source: AxisRef | Any
    ) -> NumericRange:
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        return NumericRange(
            self._display_scalar_to_canonical(value.low, quantity),
            self._display_scalar_to_canonical(value.high, quantity),
        )

    def _canonical_range_to_display(
        self, value: NumericRange, source: AxisRef | Any
    ) -> NumericRange:
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        return NumericRange(
            self._canonical_scalar_to_display(value.low, quantity),
            self._canonical_scalar_to_display(value.high, quantity),
        )

    @staticmethod
    def _convert_coordinate_array_to_display(values: np.ndarray, quantity: Any) -> np.ndarray:
        return np.asarray(
            quantity.canonical_unit.convert_value_to(values, quantity.display_unit),
            dtype=float,
        )

    def _rolling_axis_domains(self) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(self._spec, RollingPlot):
            raise TypeError("rolling axis conversion requires RollingPlot")
        series = tuple(getattr(self._payload, "series", ()))
        if not series:
            raise ValueError("rolling data has no series")
        canonical = np.asarray(series[0].x.canonical, dtype=float).reshape(-1)
        plot_x = series[0].plot_x
        if plot_x is None:
            raise RuntimeError("rolling payload has no plotted x coordinates")
        display = np.asarray(plot_x, dtype=float).reshape(-1)
        return canonical, display

    def _rolling_x_quantity(self) -> Any:
        if not isinstance(self._spec, RollingPlot):
            raise TypeError("rolling x quantity requires RollingPlot")
        series = tuple(getattr(self._payload, "series", ()))
        if not series:
            raise ValueError("rolling data has no series")
        return series[0].x

    def _rolling_fit_scale(self) -> float:
        """Return the exact affine canonical-to-point scale used by a fit."""

        canonical, _display = self._rolling_axis_domains()
        if canonical.size <= 1:
            return 1.0
        steps = np.diff(canonical)
        if (
            not np.all(np.isfinite(steps))
            or np.any(steps == 0.0)
            or not np.allclose(steps, steps[0], rtol=1.0e-9, atol=1.0e-12)
        ):
            raise ValueError(
                "rolling fit requires a uniformly spaced x coordinate"
            )
        return abs(1.0 / float(steps[0]))

    def _pulse_x_factor(self) -> float:
        if not isinstance(self._spec, PulseTimelinePlot) or not isinstance(
            self._data, PulseTimelineData
        ):
            raise TypeError("pulse time conversion requires PulseTimelinePlot")
        factor, _unit = pulse_time_scale(
            self._data,
            self.display_state.values.get("x_display_unit"),
        )
        return factor

    def _pulse_source_range_to_display(
        self, value: NumericRange
    ) -> NumericRange:
        factor = self._pulse_x_factor()
        return NumericRange(value.low * factor, value.high * factor)

    def _canonical_x_scalar_to_display(self, value: float) -> float:
        if isinstance(self._spec, RollingPlot):
            canonical, display = self._rolling_axis_domains()
            order = np.argsort(canonical)
            return float(np.interp(value, canonical[order], display[order]))
        source = self._x_selector_source()
        quantity = self._coordinate(source) if isinstance(source, AxisRef) else source
        return self._canonical_scalar_to_display(value, quantity)

    def _coordinate_values_to_display(
        self, values: np.ndarray, ref: AxisRef
    ) -> np.ndarray:
        return self._convert_coordinate_array_to_display(
            values, self._coordinate(ref)
        )

    def _area_canonical_to_display(
        self,
        value: RectangleRange,
    ) -> RectangleRange:
        if self._view is not None:
            x = self._canonical_range_to_display(
                value.x,
                self._x_selector_source(),
            )
            y = (
                value.y
                if self._is_histogram_plot()
                else self._canonical_range_to_display(
                    value.y,
                    self._y_ref_or_value(),
                )
            )
            return RectangleRange(x, y)
        if isinstance(self._spec, PulseTimelinePlot):
            return RectangleRange(
                self._pulse_source_range_to_display(value.x),
                value.y,
            )
        return value

    def _display_selector_state(self, state: SelectorState) -> SelectorState:
        value = state.value
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(value, NumericRange)
            value = self._canonical_range_to_display(
                value, self._x_selector_source()
            )
        elif state.kind is SelectorKind.AREA:
            assert isinstance(value, RectangleRange)
            value = self._area_canonical_to_display(value)
        elif state.kind is SelectorKind.CROSSHAIR:
            assert isinstance(value, CrosshairPoint)
            value = CrosshairPoint(
                self._canonical_x_scalar_to_display(value.x),
                value.y
                if self._is_histogram_plot()
                else self._canonical_scalar_to_display(value.y, self._y_ref_or_value()),
            )
        elif state.kind is SelectorKind.THRESHOLD:
            value = self._canonical_scalar_to_display(float(value), self._value_quantity())
        return replace(state, value=value)

    def _selector_state_or_none(
        self,
        kind: SelectorKind,
    ) -> SelectorState | None:
        try:
            return self._context.selector_state(kind)
        except KeyError:
            return None

__all__ = [
    "FitAuthority",
    "FitProjection",
    "FitScope",
    "FitSelection",
    "HistogramProjection",
    "ProjectionContext",
]
