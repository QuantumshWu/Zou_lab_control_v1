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
from .figure import AxisViewRole, GRID_INTENTS, ViewIntent, ViewSpec
from .histogram_display import (
    FacetedHistogramDisplayState,
    HistogramDisplayState,
    histogram_cell_thresholds_from_tree,
)
from .image_display import ImageDisplayState
from .figure_source import FigureSource
from .meter_display import MeterDisplayState
from .panel_params import panel_display_state_from_params
from .panel_size import DEFAULT_PANEL_SIZE
from .plot_kind import PlotKind
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


# Canonical persisted keys belong beside the sole PlotPanel decoder.  The
# Workbench composition owner imports them when committing authored values;
# a separate policy module would only mirror this vocabulary.
VIEW_SPEC_PARAM = "view_spec"
HISTOGRAM_THRESHOLDS_PARAM = "histogram_thresholds"
HISTOGRAM_CELL_THRESHOLDS_PARAM = "histogram_cell_thresholds"


def _dataset_view_intent_for_kind(kind: PlotKind) -> ViewIntent | None:
    if not isinstance(kind, PlotKind):
        raise TypeError("figure kind must be PlotKind")
    return {
        PlotKind.IMAGE: ViewIntent.IMAGE,
        PlotKind.CURVE: ViewIntent.CURVE,
        PlotKind.METER: ViewIntent.METER,
        PlotKind.ROLLING: ViewIntent.CURVE,
        PlotKind.HISTOGRAM: ViewIntent.HISTOGRAM,
    }.get(kind)


def _validate_figure_view(kind: PlotKind, view: ViewSpec | None) -> ViewIntent | None:
    if not isinstance(kind, PlotKind):
        raise TypeError("figure kind must be PlotKind")
    if view is not None and not isinstance(view, ViewSpec):
        raise TypeError("figure view must be ViewSpec or None")
    if kind in (PlotKind.SITE_MAP, PlotKind.PULSE):
        if view is not None:
            raise ValueError(f"{kind.value} Figure does not accept Dataset ViewSpec")
        return None
    if kind is PlotKind.GRID:
        if view is None or view.intent not in GRID_INTENTS:
            raise ValueError("Grid Figure requires an explicit faceted ViewSpec")
        from .figure import grid_facet_source

        grid_facet_source(view)
        return view.intent
    intent = _dataset_view_intent_for_kind(kind)
    if intent is None:
        raise ValueError(f"plot kind {kind.value!r} has no Dataset view intent")
    if view is not None and view.intent is not intent:
        raise ValueError("plot kind and ViewSpec intent disagree")
    return intent


@dataclass(frozen=True, slots=True)
class FigureIntent:
    """Generation-static Figure semantics shared by every presentation host."""

    kind: PlotKind
    title: str
    value_label: str
    view: ViewSpec | None = None
    rolling_distribution: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PlotKind):
            raise TypeError("FigureIntent kind must be PlotKind")
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "value_label", str(self.value_label))
        if not isinstance(self.rolling_distribution, bool):
            raise TypeError("rolling_distribution must be bool")
        _validate_figure_view(self.kind, self.view)
        if self.rolling_distribution and self.kind is not PlotKind.ROLLING:
            raise ValueError("side distribution belongs only to rolling Figures")

    @property
    def view_intent(self) -> ViewIntent | None:
        return _validate_figure_view(self.kind, self.view)

    @property
    def faceted(self) -> bool:
        return self.kind is PlotKind.GRID

    @property
    def rolling_trace(self) -> bool:
        return self.kind is PlotKind.ROLLING


def figure_intent_from_view(
    view: ViewSpec,
    *,
    title: str,
    value_label: str,
) -> FigureIntent:
    """Build the one typed Figure intent for an already-resolved Dataset view."""

    if not isinstance(view, ViewSpec):
        raise TypeError("resolved Figure view must be ViewSpec")
    faceted = any(
        binding.role is AxisViewRole.FACET for binding in view.source_bindings
    )
    kind = (
        PlotKind.GRID
        if faceted
        else {
            ViewIntent.IMAGE: PlotKind.IMAGE,
            ViewIntent.CURVE: PlotKind.CURVE,
            ViewIntent.HISTOGRAM: PlotKind.HISTOGRAM,
            ViewIntent.METER: PlotKind.METER,
        }[view.intent]
    )
    return FigureIntent(kind, title, value_label, view=view)


@dataclass(frozen=True, slots=True)
class PlotPanelContract:
    """One Figure intent bound to a panel identity and raster surface.

    ``pixel_size`` is derived from the named frontend size and DPR.  It is not
    a QWidget measurement and therefore cannot drift between a live card,
    editor, report, and exported figure.
    """

    panel_id: str
    figure: FigureIntent
    size_name: str = DEFAULT_PANEL_SIZE
    pixel_ratio: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", canonical_text(self.panel_id, "panel_id"))
        if not isinstance(self.figure, FigureIntent):
            raise TypeError("plot panel contract requires FigureIntent")
        if self.figure.kind is PlotKind.PULSE:
            raise ValueError("pulse documents use their dedicated Figure contract")
        geometry = panel_surface_geometry(
            self.size_name,
            pixel_ratio=self.pixel_ratio,
        )
        object.__setattr__(self, "size_name", geometry.size_name)
        object.__setattr__(self, "pixel_ratio", geometry.pixel_ratio)

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
            self.figure,
            self.size_name,
            self.pixel_size,
            self.pixel_ratio,
        )


def plot_panel_input(
    kind: PlotKind,
    snapshot: OwnedSnapshot,
    presentation: object | None = None,
) -> FigureSource:
    """Bind one immutable source under the plot kind's input contract."""

    if not isinstance(kind, PlotKind):
        raise TypeError("plot panel kind must be PlotKind")
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("plot panel input requires OwnedSnapshot")
    validate_plot_panel_schema(kind, snapshot.block.schema)
    if kind is PlotKind.SITE_MAP:
        if not isinstance(presentation, SiteMapPresentation):
            raise TypeError("SiteMap plot requires SiteMapPresentation")
        return FigureSource(snapshot, presentation)
    return FigureSource(snapshot)


def validate_plot_panel_schema(kind: PlotKind, schema: DatasetSchema) -> None:
    """Validate source semantics that distinguish otherwise similar views.

    A Meter reads one scalar projection from the supplied dataset.  A rolling
    Monitor is different: its x domain must already be an explicit
    ``MONITOR_HISTORY`` point axis produced by the data owner.  The frontend
    never turns successive unrelated snapshots into a hidden GUI-side buffer.
    """

    if not isinstance(kind, PlotKind):
        raise TypeError("plot panel kind must be PlotKind")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("plot panel schema must be DatasetSchema")
    if kind is not PlotKind.ROLLING:
        return
    history_columns = tuple(
        column
        for column in schema.point_table.columns
        if column.role == MONITOR_HISTORY
    )
    if len(history_columns) != 1:
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


def plot_panel_view_from_params(
    kind: PlotKind,
    params: Mapping[str, object],
) -> ViewSpec | None:
    """Decode the sole persisted ViewSpec under Plot Panel policy."""

    if not isinstance(params, Mapping):
        raise TypeError("plot panel params must be a mapping")
    raw = params.get(VIEW_SPEC_PARAM)
    if raw is None:
        return None
    if not isinstance(kind, PlotKind):
        raise TypeError("plot panel kind must be PlotKind")
    if kind is PlotKind.SITE_MAP:
        raise ValueError("SiteMap panels cannot persist a generic ViewSpec")
    from .figure import view_spec_from_tree

    view = view_spec_from_tree(raw)
    _validate_figure_view(kind, view)
    return view


def plot_panel_view_for_schema(
    kind: PlotKind,
    params: Mapping[str, object],
    schema,
    *,
    default_grid: bool = False,
) -> ViewSpec | None:
    """Read a current-generation view, optionally deriving Grid's default."""

    view = plot_panel_view_from_params(kind, params)
    if view is not None and view.schema_fingerprint != schema.fingerprint:
        view = None
    if view is None and kind is PlotKind.GRID and bool(default_grid):
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


def plot_panel_display_state(
    contract: PlotPanelContract,
    params: Mapping[str, object],
    *,
    revision: int,
    focus=None,
    home_view: bool = False,
) -> PlotDisplayState:
    """Resolve authored panel parameters into the renderer's sole state type."""

    if not isinstance(contract, PlotPanelContract):
        raise TypeError("contract must be PlotPanelContract")
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    figure = contract.figure
    intent = figure.view_intent
    raw_cell_thresholds = params.get(HISTOGRAM_CELL_THRESHOLDS_PARAM)
    return panel_display_state_from_params(
        figure.kind,
        params,
        revision=revision,
        cell_intent=intent if figure.faceted else None,
        focus=focus,
        thresholds=(
            ()
            if figure.faceted
            else tuple(params.get(HISTOGRAM_THRESHOLDS_PARAM, ()))
        ),
        cell_thresholds=(
            ()
            if raw_cell_thresholds is None
            else histogram_cell_thresholds_from_tree(raw_cell_thresholds)
        ),
        home_view=home_view,
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
        figure = contract.figure
        if figure.kind is PlotKind.SITE_MAP:
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
                    title=figure.title,
                    value_label=figure.value_label,
                )
            frame = self._composer.compose(
                request.source.site_map,
                display=request.display,
            )
            return PlotPanelComposeResult(frame, None, None)

        if request.source.site_map is not None:
            raise ValueError("ordinary dataset plot cannot carry SiteMapPresentation")
        if not figure.faceted and request.focus is not None:
            raise ValueError("ordinary plot panels do not accept facet focus")
        if self._composer is None:
            from .panel_render import PanelComposer

            self._composer = PanelComposer(
                contract.panel_id,
                intent=figure.view_intent,
                size_name=contract.size_name,
                pixel_ratio=contract.pixel_ratio,
                label=figure.title,
                value_label=figure.value_label,
                view=figure.view,
                rolling_trace=figure.rolling_trace,
                rolling_distribution=figure.rolling_distribution,
            )
        if figure.faceted:
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
        intent = contract.figure
        if intent.faceted or intent.kind is PlotKind.SITE_MAP:
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
                intent=intent.view_intent,
                size_name=contract.size_name,
                pixel_ratio=contract.pixel_ratio,
                label=intent.title,
                value_label=intent.value_label,
                view=intent.view,
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
        intent = contract.figure
        if not intent.faceted:
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
                intent=intent.view_intent,
                size_name=contract.size_name,
                pixel_ratio=contract.pixel_ratio,
                label=intent.title,
                value_label=intent.value_label,
                view=intent.view,
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
    "FigureIntent",
    "PlotDisplayState",
    "PlotPanelComposeRequest",
    "PlotPanelComposeResult",
    "PlotPanelContract",
    "PlotPanelSession",
    "figure_intent_from_view",
    "plot_panel_display_state",
    "plot_panel_input",
    "plot_panel_value_label",
    "plot_panel_view_for_schema",
    "plot_panel_view_from_params",
]
