"""High-level, renderer-neutral contract for every interactive plot panel.

Applications provide a typed source, placement, and authored display values.
This module alone resolves the plot vocabulary into one immutable contract and
opens the worker-owned rendering session.  TaskConsole, Calibration, notebook
windows, and report exporters must not instantiate Matplotlib composers or
reimplement kind/view/display policy themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from zlc_data import MONITOR_HISTORY, DatasetSchema, FitResultBatch, OwnedSnapshot
from zlc_storage import canonical_text

from .curve_display import CurveDisplayState
from .display_range import RelimMode
from .figure import GRID_INTENTS, ViewIntent, ViewSpec
from .histogram_display import (
    FacetedHistogramDisplayState,
    HistogramCountScale,
    HistogramDisplayState,
    HistogramFitMode,
    histogram_cell_thresholds_from_tree,
)
from .image_display import ImageColormap, ImageDisplayState
from .figure_source import FigureSource
from .meter_display import MeterDisplayState
from .panel_params import resolved_panel_param
from .panel_policy import (
    HISTOGRAM_CELL_THRESHOLDS_PARAM,
    HISTOGRAM_THRESHOLDS_PARAM,
    VIEW_SPEC_PARAM,
    panel_view_intents,
)
from .plot_kind import PLOT_KIND_SPEC_BY_KEY
from .plot_layout import PanelSurfaceGeometry, panel_surface_geometry
from .panel_render import PanelProvenance
from .site_map import SiteMapPresentation

if TYPE_CHECKING:
    from .data_figure import DataFigure
    from .panel_render import FacetedPanelFocus, FacetedPanelResult
    from .render import BoardFrame


PlotDisplayState: TypeAlias = (
    CurveDisplayState
    | ImageDisplayState
    | HistogramDisplayState
    | FacetedHistogramDisplayState
    | MeterDisplayState
)


@dataclass(frozen=True, slots=True)
class PlotPanelContract:
    """One complete static presentation contract for a plot surface.

    ``pixel_size`` is derived from the named frontend size and DPR.  It is not
    a QWidget measurement and therefore cannot drift between a live card,
    editor, report, and exported figure.
    """

    panel_id: str
    kind: str
    title: str
    value_label: str
    size_name: str = "2x2"
    pixel_ratio: float = 1.0
    view: ViewSpec | None = None
    rolling_distribution: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", canonical_text(self.panel_id, "panel_id"))
        kind = canonical_text(self.kind, "plot kind")
        if kind not in PLOT_KIND_SPEC_BY_KEY:
            raise ValueError(f"unknown plot kind {kind!r}")
        if kind == "pulse":
            raise ValueError("pulse documents use their dedicated Figure contract")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "value_label", str(self.value_label))
        geometry = panel_surface_geometry(
            self.size_name,
            pixel_ratio=self.pixel_ratio,
        )
        object.__setattr__(self, "size_name", geometry.size_name)
        object.__setattr__(self, "pixel_ratio", geometry.pixel_ratio)
        if self.view is not None and not isinstance(self.view, ViewSpec):
            raise TypeError("plot panel view must be ViewSpec or None")
        plot_panel_intent(kind, self.view)
        if bool(self.rolling_distribution) and kind != "monitor":
            raise ValueError("side distribution belongs only to rolling monitor panels")

    @property
    def intent(self) -> ViewIntent | None:
        return plot_panel_intent(self.kind, self.view)

    @property
    def faceted(self) -> bool:
        return self.kind == "grid"

    @property
    def surface_geometry(self) -> PanelSurfaceGeometry:
        return panel_surface_geometry(
            self.size_name,
            pixel_ratio=self.pixel_ratio,
        )

    @property
    def logical_size(self) -> tuple[int, int]:
        return self.surface_geometry.logical_size

    @property
    def pixel_size(self) -> tuple[int, int]:
        return self.surface_geometry.raster_size

    @property
    def session_identity(self) -> tuple[object, ...]:
        """Facts whose change requires another worker-owned Agg session."""

        return (
            self.kind,
            self.title,
            self.value_label,
            self.size_name,
            self.pixel_size,
            self.pixel_ratio,
            self.view,
            bool(self.rolling_distribution),
        )


def plot_panel_input(
    kind: str,
    snapshot: OwnedSnapshot,
    presentation: object | None = None,
) -> FigureSource:
    """Bind one immutable source under the plot kind's input contract."""

    key = canonical_text(kind, "plot kind")
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("plot panel input requires OwnedSnapshot")
    validate_plot_panel_schema(key, snapshot.block.schema)
    if key == "sites":
        if not isinstance(presentation, SiteMapPresentation):
            raise TypeError("SiteMap plot requires SiteMapPresentation")
        return FigureSource(snapshot, presentation)
    return FigureSource(snapshot)


def validate_plot_panel_schema(kind: str, schema: DatasetSchema) -> None:
    """Validate source semantics that distinguish otherwise similar views.

    A Meter reads one scalar projection from the supplied dataset.  A rolling
    Monitor is different: its x domain must already be an explicit
    ``MONITOR_HISTORY`` point axis produced by the data owner.  The frontend
    never turns successive unrelated snapshots into a hidden GUI-side buffer.
    """

    key = canonical_text(kind, "plot kind")
    if key not in PLOT_KIND_SPEC_BY_KEY:
        raise ValueError(f"unknown plot kind {key!r}")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("plot panel schema must be DatasetSchema")
    if key != "monitor":
        return
    history_axes = tuple(
        axis for axis in schema.point_axes if axis.role == MONITOR_HISTORY
    )
    if len(history_axes) != 1:
        raise ValueError(
            "rolling monitor requires exactly one explicit MONITOR_HISTORY "
            "point axis; use Meter for a scalar dataset"
        )


@dataclass(frozen=True, slots=True)
class PlotPanelComposeRequest:
    """Dynamic facts frozen for one worker compose operation."""

    source: FigureSource
    display: PlotDisplayState
    provenance: PanelProvenance
    focus: FacetedPanelFocus | None = None
    fit_result: FitResultBatch | None = None
    fit_result_identity: str | None = None
    histogram_projection_value_range: tuple[float, float] | None = None
    check_cancelled: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, FigureSource):
            raise TypeError("plot panel source must be FigureSource")
        if not isinstance(self.provenance, PanelProvenance):
            raise TypeError("plot panel provenance must be PanelProvenance")
        if not isinstance(
            self.display,
            (
                CurveDisplayState,
                ImageDisplayState,
                HistogramDisplayState,
                FacetedHistogramDisplayState,
                MeterDisplayState,
            ),
        ):
            raise TypeError("plot panel display state has another type")
        if self.focus is not None:
            from .panel_render import FacetedPanelFocus

            if not isinstance(self.focus, FacetedPanelFocus):
                raise TypeError("plot panel focus must be FacetedPanelFocus or None")
        if (self.fit_result is None) != (self.fit_result_identity is None):
            raise ValueError("fit result and identity must be supplied together")
        if self.fit_result is not None and not isinstance(self.fit_result, FitResultBatch):
            raise TypeError("fit_result must be FitResultBatch or None")
        if self.fit_result_identity is not None:
            canonical_text(self.fit_result_identity, "fit_result_identity")
        if self.check_cancelled is not None and not callable(self.check_cancelled):
            raise TypeError("check_cancelled must be callable or None")
        value_range = self.histogram_projection_value_range
        if value_range is not None:
            if not isinstance(
                self.display,
                (HistogramDisplayState, FacetedHistogramDisplayState),
            ):
                raise ValueError(
                    "only histogram composition accepts a projection value range"
                )
            from .display_range import validated_display_range

            object.__setattr__(
                self,
                "histogram_projection_value_range",
                validated_display_range(
                    value_range,
                    "histogram projection value range",
                ),
            )


@dataclass(frozen=True, slots=True)
class PlotPanelComposeResult:
    """Exactly one ordinary or faceted frontend result."""

    frame: BoardFrame | None
    faceted: FacetedPanelResult | None
    figure: DataFigure | None

    def __post_init__(self) -> None:
        ordinary = self.frame is not None
        faceted = self.faceted is not None
        if ordinary == faceted:
            raise ValueError("plot compose result requires one ordinary or faceted front")
        from .data_figure import DataFigure
        from .panel_render import FacetedPanelResult
        from .render import BoardFrame

        if ordinary:
            if not isinstance(self.frame, BoardFrame):
                raise TypeError("ordinary plot result requires BoardFrame")
            if self.figure is not None and not isinstance(self.figure, DataFigure):
                raise TypeError("ordinary plot figure must be DataFigure or None")
            return
        if not isinstance(self.faceted, FacetedPanelResult):
            raise TypeError("faceted plot result requires FacetedPanelResult")
        if not isinstance(self.figure, DataFigure):
            raise TypeError("faceted plot result requires DataFigure")
        if self.figure is not self.faceted.figure:
            raise ValueError("faceted result and Figure owner disagree")


def plot_panel_intent(kind: str, view: ViewSpec | None = None) -> ViewIntent | None:
    """Resolve one plot kind to its frontend-owned typed view intent."""

    key = canonical_text(kind, "plot kind")
    if key == "sites":
        if view is not None:
            raise ValueError("SiteMap panels do not accept ViewSpec")
        return None
    if key == "grid":
        if not isinstance(view, ViewSpec) or view.intent not in GRID_INTENTS:
            raise ValueError("grid plot requires an explicit faceted ViewSpec")
        from .figure import grid_facet_axis

        grid_facet_axis(view)
        return view.intent
    try:
        intent = panel_view_intents()[key]
    except KeyError as error:
        raise ValueError(f"plot kind {key!r} has no dataset view intent") from error
    if view is not None and view.intent is not intent:
        raise ValueError("plot kind and ViewSpec intent disagree")
    return intent


def plot_panel_view_from_params(
    kind: str,
    params: Mapping[str, object],
) -> ViewSpec | None:
    """Decode the sole persisted ViewSpec under Plot Panel policy."""

    if not isinstance(params, Mapping):
        raise TypeError("plot panel params must be a mapping")
    raw = params.get(VIEW_SPEC_PARAM)
    if raw is None:
        return None
    key = canonical_text(kind, "plot kind")
    if key == "sites":
        raise ValueError("SiteMap panels cannot persist a generic ViewSpec")
    from .figure import view_spec_from_tree

    view = view_spec_from_tree(raw)
    if key == "grid":
        plot_panel_intent(key, view)
    else:
        plot_panel_intent(key, view)
    return view


def plot_panel_view_for_schema(
    kind: str,
    params: Mapping[str, object],
    schema,
    *,
    default_grid: bool = False,
) -> ViewSpec | None:
    """Read a current-generation view, optionally deriving Grid's default."""

    view = plot_panel_view_from_params(kind, params)
    if view is not None and view.schema_fingerprint != schema.fingerprint:
        view = None
    if view is None and str(kind) == "grid" and bool(default_grid):
        from .figure import suggest_default_grid_view

        view = suggest_default_grid_view(schema).spec
    return view


def plot_panel_value_label(
    signal_key: str,
    axis_labels: Mapping[str, object] | None,
    short_labels: Mapping[str, object] | None,
) -> str:
    """Resolve visible plot chrome without exposing a routing key by default."""

    key = canonical_text(signal_key, "plot signal key")
    for labels in (axis_labels, short_labels):
        if labels is None:
            continue
        if not isinstance(labels, Mapping):
            raise TypeError("plot signal labels must be mappings")
        label = str(labels.get(key, "")).strip()
        if label:
            return label
    return key.rsplit("/", 1)[-1].strip() or "Signal"


def _display_selection(view: ViewSpec):
    """Merge every persisted display-only term for a safe re-suggestion."""

    if not isinstance(view, ViewSpec):
        raise TypeError("view must be ViewSpec")
    from zlc_data import Selection

    terms = tuple(
        term
        for selection in view.display_selections
        for term in selection.terms
    )
    return None if not terms else Selection(terms)


def _repeat_mode_from_view(view: ViewSpec, schema):
    """Read the authored repeat policy from one typed ViewSpec."""

    from .figure import AxisViewRole, DisplayReductionMethod, RepeatViewMode

    binding = view.binding(schema.repeat_axis.axis_id)
    if binding.role is AxisViewRole.REDUCED:
        return (
            RepeatViewMode.MEAN
            if binding.reduction.method is DisplayReductionMethod.MEAN
            else RepeatViewMode.SUM
        )
    try:
        return {
            AxisViewRole.BATCH: RepeatViewMode.BATCH,
            AxisViewRole.FACET: RepeatViewMode.FACET,
            AxisViewRole.SAMPLE: RepeatViewMode.SAMPLE,
            AxisViewRole.SELECTED: RepeatViewMode.LATEST,
        }[binding.role]
    except KeyError as error:
        raise ValueError(
            f"unsupported repeat binding {binding.role.value}"
        ) from error


def _view_preferences(
    view: ViewSpec,
    schema,
    *,
    repeat_mode=None,
    facet_axis_ids=None,
):
    """Re-author view preferences without treating a ViewSpec as authority."""

    from .figure import AxisViewRole, ViewPreferences

    by_role = {
        role: tuple(
            binding.axis_id for binding in view.axis_bindings if binding.role is role
        )
        for role in (
            AxisViewRole.X,
            AxisViewRole.IMAGE_X,
            AxisViewRole.IMAGE_Y,
            AxisViewRole.BATCH,
            AxisViewRole.FACET,
            AxisViewRole.SAMPLE,
        )
    }
    repeat_id = schema.repeat_axis.axis_id
    facets = (
        tuple(facet_axis_ids)
        if facet_axis_ids is not None
        else tuple(
            axis_id
            for axis_id in by_role[AxisViewRole.FACET]
            if axis_id != repeat_id
        )
    )
    return ViewPreferences(
        repeat_mode=(
            repeat_mode
            if repeat_mode is not None
            else _repeat_mode_from_view(view, schema)
        ),
        x_axis_id=next(iter(by_role[AxisViewRole.X]), None),
        image_x_axis_id=next(iter(by_role[AxisViewRole.IMAGE_X]), None),
        image_y_axis_id=next(iter(by_role[AxisViewRole.IMAGE_Y]), None),
        batch_axis_ids=tuple(
            axis_id
            for axis_id in by_role[AxisViewRole.BATCH]
            if axis_id != repeat_id
        ),
        facet_axis_ids=facets,
        sample_axis_ids=tuple(
            axis_id
            for axis_id in by_role[AxisViewRole.SAMPLE]
            if axis_id != repeat_id
        ),
    )


def plot_panel_repeat_modes(
    kind: str,
    schema,
    intent: ViewIntent,
    current_view: ViewSpec | None,
):
    """Return repeat policies renderable by this exact panel host."""

    from .figure import RepeatViewMode, dataset_contract_for, grid_facet_axis

    if str(kind) == "sites":
        return ()
    modes = dataset_contract_for(intent).repeat_modes
    ordinary = tuple(mode for mode in modes if mode is not RepeatViewMode.FACET)
    if str(kind) != "grid" or current_view is None:
        return ordinary
    return (
        (RepeatViewMode.FACET,)
        if grid_facet_axis(current_view) == schema.repeat_axis.axis_id
        else ordinary
    )


def plot_panel_selected_repeat_mode(
    schema,
    intent: ViewIntent,
    current_view: ViewSpec | None,
    presented_view: ViewSpec | None = None,
):
    """Resolve the control selection from authored or visible Figure state."""

    from .figure import dataset_contract_for

    view = current_view
    if (
        view is None
        and presented_view is not None
        and presented_view.schema_fingerprint == schema.fingerprint
    ):
        view = presented_view
    if view is not None:
        return _repeat_mode_from_view(view, schema)
    return dataset_contract_for(intent).default_repeat_mode


def plot_panel_view_with_repeat_mode(
    kind: str,
    schema,
    intent: ViewIntent,
    current_view: ViewSpec | None,
    repeat_mode,
):
    """Re-resolve a panel view after one explicit repeat-policy edit."""

    from .figure import (
        ViewPreferences,
        grid_facet_axis,
        resolve_grid_view,
        suggest_view,
    )

    if repeat_mode not in plot_panel_repeat_modes(
        kind,
        schema,
        intent,
        current_view,
    ):
        raise ValueError(
            f"repeat mode {repeat_mode.value} is not renderable by this panel"
        )
    if str(kind) == "grid":
        if current_view is None:
            raise ValueError("Grid repeat policy needs a committed facet view")
        return resolve_grid_view(
            schema,
            intent,
            grid_facet_axis(current_view),
            current_view=current_view,
            repeat_mode=repeat_mode,
        )
    preferences = (
        ViewPreferences(repeat_mode=repeat_mode)
        if current_view is None
        else _view_preferences(
            current_view,
            schema,
            repeat_mode=repeat_mode,
        )
    )
    selection = (
        None
        if current_view is None
        else _display_selection(current_view)
    )
    return suggest_view(
        schema,
        intent,
        selection,
        preferences=preferences,
    )


def plot_panel_display_state(
    contract: PlotPanelContract,
    params: Mapping[str, object],
    *,
    revision: int,
    x_view=None,
    y_view=None,
    focus=None,
) -> PlotDisplayState:
    """Resolve authored panel parameters into the renderer's sole state type."""

    if not isinstance(contract, PlotPanelContract):
        raise TypeError("contract must be PlotPanelContract")
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    mode = RelimMode(str(params.get("relim", RelimMode.TIGHT.value)))
    fixed = None
    if mode is RelimMode.FIXED:
        fixed = (float(params.get("fixed_lo", 0.0)), float(params.get("fixed_hi", 1.0)))
    intent = contract.intent
    if intent is ViewIntent.CURVE:
        return CurveDisplayState(
            revision=revision,
            relim_mode=mode,
            fixed_y_limits=fixed,
            x_view=x_view,
        )
    if intent is ViewIntent.HISTOGRAM:
        param_kind = "hist" if contract.faceted else contract.kind
        count_scale = resolved_panel_param(param_kind, params, "count_scale")
        fit_mode = resolved_panel_param(param_kind, params, "fit_mode")
        if not isinstance(count_scale, HistogramCountScale):
            raise TypeError("histogram count_scale lost its typed value")
        if not isinstance(fit_mode, HistogramFitMode):
            raise TypeError("histogram fit_mode lost its typed value")
        display = HistogramDisplayState(
            revision=revision,
            relim_mode=mode,
            count_scale=count_scale,
            bin_count=int(resolved_panel_param(param_kind, params, "bin_count")),
            fit_mode=fit_mode,
            fixed_count_limits=fixed,
            x_view=x_view,
            thresholds=(
                ()
                if contract.faceted
                else tuple(params.get(HISTOGRAM_THRESHOLDS_PARAM, ()))
            ),
        )
        if not contract.faceted:
            return display
        raw = params.get(HISTOGRAM_CELL_THRESHOLDS_PARAM)
        return FacetedHistogramDisplayState(
            display,
            () if raw is None else histogram_cell_thresholds_from_tree(raw),
        )
    if intent is ViewIntent.METER:
        return MeterDisplayState(
            0 if focus is None else int(focus.panel_index),
            None if focus is None else focus.selection,
            revision,
        )
    param_kind = "2d" if contract.faceted else contract.kind
    colormap = resolved_panel_param(param_kind, params, "colormap")
    if not isinstance(colormap, ImageColormap):
        raise TypeError("image colormap lost its typed value")
    return ImageDisplayState(
        revision=revision,
        relim_mode=mode,
        colormap=colormap,
        fixed_color_limits=fixed,
        x_view=x_view,
        y_view=y_view,
    )


class PlotPanelSession:
    """Worker-owned executor for one immutable :class:`PlotPanelContract`."""

    def __init__(self, contract: PlotPanelContract) -> None:
        if not isinstance(contract, PlotPanelContract):
            raise TypeError("contract must be PlotPanelContract")
        self.contract = contract
        self._composer = None

    def compose(self, request: PlotPanelComposeRequest) -> PlotPanelComposeResult:
        if not isinstance(request, PlotPanelComposeRequest):
            raise TypeError("request must be PlotPanelComposeRequest")
        contract = self.contract
        if contract.kind == "sites":
            if request.source.site_map is None:
                raise ValueError("SiteMap plot requires one joined SiteMapPresentation")
            if not isinstance(request.display, ImageDisplayState):
                raise TypeError("SiteMap display must be ImageDisplayState")
            if request.focus is not None or request.fit_result is not None:
                raise ValueError("SiteMap plot does not accept facet focus or Fit")
            if self._composer is None:
                from .site_map_render import SiteMapComposer

                self._composer = SiteMapComposer(
                    contract.panel_id,
                    surface_geometry=contract.surface_geometry,
                    title=contract.title,
                    value_label=contract.value_label,
                )
            frame = self._composer.compose(
                request.source.site_map,
                display=request.display,
            )
            return PlotPanelComposeResult(frame, None, None)

        if request.source.site_map is not None:
            raise ValueError("ordinary dataset plot cannot carry SiteMapPresentation")
        if not contract.faceted and request.focus is not None:
            raise ValueError("ordinary plot panels do not accept facet focus")
        if self._composer is None:
            from .panel_render import PanelComposer

            self._composer = PanelComposer(
                contract.panel_id,
                intent=contract.intent,
                size_name=contract.size_name,
                pixel_ratio=contract.pixel_ratio,
                label=contract.title,
                value_label=contract.value_label,
                view=contract.view,
                rolling_trace=contract.kind == "monitor",
                rolling_distribution=contract.rolling_distribution,
            )
        if contract.faceted:
            result = self._composer.compose_faceted(
                request.source.snapshot,
                display=request.display,
                provenance=request.provenance,
                focus=request.focus,
                fit_result=request.fit_result,
                fit_result_identity=request.fit_result_identity,
                check_cancelled=request.check_cancelled,
            )
            return PlotPanelComposeResult(None, result, result.figure)
        frame, figure = self._composer.compose_with_figure(
            request.source.snapshot,
            display=request.display,
            provenance=request.provenance,
            fit_result=request.fit_result,
            fit_result_identity=request.fit_result_identity,
            histogram_projection_value_range=(
                request.histogram_projection_value_range
            ),
            check_cancelled=request.check_cancelled,
        )
        return PlotPanelComposeResult(frame, None, figure)

    def compose_data_figure(
        self,
        figure: DataFigure,
        request: PlotPanelComposeRequest,
    ) -> PlotPanelComposeResult:
        """Paint an already-evaluated single panel without a second owner.

        This is the saved/archive counterpart to :meth:`compose`: the input
        ``DataFigure`` retains its exact evaluated arrays, while the same
        PlotPanel composer owns pixels, style, display continuity, and Fit
        overlays.
        """

        from .data_figure import DataFigure

        if not isinstance(figure, DataFigure):
            raise TypeError("figure must be DataFigure")
        if not isinstance(request, PlotPanelComposeRequest):
            raise TypeError("request must be PlotPanelComposeRequest")
        contract = self.contract
        if contract.faceted or contract.kind == "sites":
            raise ValueError(
                "already-evaluated single-panel composition rejects grid/SiteMap"
            )
        source_ref = request.source.snapshot.ref
        inputs = figure.evaluated.inputs
        if len(inputs) != 1 or inputs[0].ref != source_ref:
            raise ValueError("Figure and PlotPanel request name another source")
        if self._composer is None:
            from .panel_render import PanelComposer

            self._composer = PanelComposer(
                contract.panel_id,
                intent=contract.intent,
                size_name=contract.size_name,
                pixel_ratio=contract.pixel_ratio,
                label=contract.title,
                value_label=contract.value_label,
                view=contract.view,
            )
        frame, visible = self._composer.compose_data_figure(
            figure,
            display=request.display,
            provenance=request.provenance,
            fit_result=request.fit_result,
            fit_result_identity=request.fit_result_identity,
            histogram_projection_value_range=(
                request.histogram_projection_value_range
            ),
            check_cancelled=request.check_cancelled,
        )
        return PlotPanelComposeResult(frame, None, visible)

    def compose_data_figure_grid(
        self,
        figure: DataFigure,
        request: PlotPanelComposeRequest,
    ) -> PlotPanelComposeResult:
        """Paint an already-evaluated grid without rebuilding its evaluation."""

        from .data_figure import DataFigure

        if not isinstance(figure, DataFigure):
            raise TypeError("figure must be DataFigure")
        if not isinstance(request, PlotPanelComposeRequest):
            raise TypeError("request must be PlotPanelComposeRequest")
        contract = self.contract
        if not contract.faceted:
            raise ValueError("already-evaluated grid composition requires grid kind")
        if request.focus is not None:
            raise ValueError("already-evaluated grid composition is overview-only")
        if request.source.site_map is not None:
            raise ValueError("dataset grid cannot carry SiteMapPresentation")
        if request.histogram_projection_value_range is not None:
            raise ValueError(
                "grid overview does not accept one focused histogram value range"
            )
        source_ref = request.source.snapshot.ref
        inputs = figure.evaluated.inputs
        if len(inputs) != 1 or inputs[0].ref != source_ref:
            raise ValueError("Figure and PlotPanel request name another source")
        if self._composer is None:
            from .panel_render import PanelComposer

            self._composer = PanelComposer(
                contract.panel_id,
                intent=contract.intent,
                size_name=contract.size_name,
                pixel_ratio=contract.pixel_ratio,
                label=contract.title,
                value_label=contract.value_label,
                view=contract.view,
            )
        result = self._composer.compose_data_figure_faceted(
            figure,
            display=request.display,
            fit_result=request.fit_result,
            fit_result_identity=request.fit_result_identity,
            check_cancelled=request.check_cancelled,
        )
        return PlotPanelComposeResult(None, result, result.figure)

    def close(self) -> None:
        composer, self._composer = self._composer, None
        if composer is not None:
            composer.close()


__all__ = [
    "PlotDisplayState",
    "PlotPanelComposeRequest",
    "PlotPanelComposeResult",
    "PlotPanelContract",
    "PlotPanelSession",
    "plot_panel_display_state",
    "plot_panel_intent",
    "plot_panel_input",
    "plot_panel_repeat_modes",
    "plot_panel_selected_repeat_mode",
    "plot_panel_value_label",
    "plot_panel_view_for_schema",
    "plot_panel_view_from_params",
    "plot_panel_view_with_repeat_mode",
]
