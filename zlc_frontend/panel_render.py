"""Compose ONE panel's raster front from a frozen snapshot.  Worker-side only.

A console card and a Workbench narrow window both need the same thing: given an
immutable snapshot of a dataset plus the operator's display state, produce the
front a board can paint.  The derivation -- which view a schema admits, what
document wraps it, how the display state resolves colour limits -- is written
ONCE here, so two hosts cannot drift into two answers for the same data.

Nothing here touches Qt.  The product is a :class:`~zlc_frontend.render.BoardFrame`
of immutable bytes, which is what lets a megapixel frame be rasterized on a
worker while the GUI thread stays responsive; the host presents it unchanged.

Why this exists next to the Workbench board controller: that
controller owns ONE source's change listener and drives a fixed image+scalar
board.  A console board is N independent cards reading ONE per-tick freeze, so
its panels cannot each own the source.  Both, however, must rasterize a frame
the same way -- that shared step is what this module holds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from zlc_data import AxisSourceRef
from .curve_display import CurveDisplayState
from .data_figure import (
    DataFigure,
    FacetedOverviewArtifact,
)
from .figure import (
    DatasetDescriptor,
    DatasetId,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedInput,
    EvaluatedMeter,
    EvaluatedProjectionIdentity,
    FigureDocument,
    FigureLayer,
    ResolvedDataset,
    ResolvedDatasetMap,
    SuggestionStatus,
    ViewIntent,
    ViewSpec,
    dataset_contract_for,
    suggest_view,
    validate_view_spec,
)
from .histogram_display import (
    FacetedHistogramDisplayState,
    HistogramBinProjection,
    HistogramDisplayState,
)
from .image_display import (
    ImageDisplayState,
    image_viewport_for_display_state,
    resolve_image_color_limits,
)
from .fit_projection import canonical_panel_focus_address
from .panel_size import DEFAULT_PANEL_SIZE
from .plot_layout import panel_surface_geometry

__all__ = [
    "PanelComposer",
    "FacetedPanelFocus",
    "FacetedPanelResult",
    "PanelRenderError",
    "view_for_schema",
]


class PanelRenderError(RuntimeError):
    """A snapshot that cannot be shown as asked, with the reason the host shows."""


@dataclass(frozen=True, slots=True)
class FacetedPanelFocus:
    """One exact display cell chosen from a previously painted overview."""

    panel_index: int
    address: tuple[tuple[AxisSourceRef, int], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.panel_index, bool)
            or not isinstance(self.panel_index, int)
            or self.panel_index < 0
        ):
            raise ValueError("faceted focus panel_index must be non-negative")
        object.__setattr__(
            self,
            "address",
            canonical_panel_focus_address(self.address),
        )


@dataclass(frozen=True, slots=True)
class FacetedPanelResult:
    """One complete faceted compose result.

    Exactly one surface is present: immutable PNG + hit regions for overview,
    or one ordinary ``BoardFrame`` for the selected cell.  ``figure`` remains
    the complete faceted DataFigure authority in both cases; a focused
    ``BoardFrame`` already carries its exact selected-cell payload.  Keeping
    the Figure complete prevents display focus from silently replacing Fit
    batch axes with one scalar cell.
    """

    figure: DataFigure
    overview: FacetedOverviewArtifact | None = None
    frame: object | None = None
    focus: FacetedPanelFocus | None = None

    def __post_init__(self) -> None:
        from .render import BoardFrame

        if not isinstance(self.figure, DataFigure):
            raise TypeError("faceted result figure must be DataFigure")
        overview = self.overview is not None
        focused = self.frame is not None
        if overview == focused:
            raise ValueError(
                "faceted result requires exactly one overview or focus front"
            )
        if overview:
            if not isinstance(self.overview, FacetedOverviewArtifact):
                raise TypeError(
                    "faceted overview must be FacetedOverviewArtifact"
                )
            if self.overview.figure is not self.figure:
                raise ValueError(
                    "faceted result and overview must share one Figure owner"
                )
            if self.focus is not None:
                raise ValueError("faceted overview cannot carry focus")
        else:
            if not isinstance(self.frame, BoardFrame):
                raise TypeError("faceted focus frame must be BoardFrame")
            if self.focus is None:
                raise ValueError("faceted focus result requires its exact focus")


def view_for_schema(
    schema,
    intent: ViewIntent,
    selection=None,
    *,
    view: ViewSpec | None = None,
):
    """The ViewSpec this schema admits for ``intent``.

    The metadata-only suggester derives a complete display default from named
    axis roles and schema declaration order.  The returned ``ViewSpec`` records
    every chosen display, sample, slider, batch, and reduction role explicitly;
    ndarray shape, values, and AxisId spelling never participate.  A schema with
    an unknown role still fails instead of receiving a guessed projection.
    """

    if view is not None:
        if not isinstance(view, ViewSpec):
            raise TypeError("view must be ViewSpec or None")
        if view.schema_fingerprint != schema.fingerprint:
            raise PanelRenderError("saved panel view belongs to a different dataset schema")
        if view.intent is not intent:
            raise PanelRenderError("saved panel view belongs to a different view intent")
        try:
            validate_view_spec(schema, view, dataset_contract_for(intent))
        except (TypeError, ValueError, IndexError) as error:
            raise PanelRenderError(f"saved panel view is invalid: {error}") from error
        return view
    suggestion = suggest_view(schema, intent, selection)
    if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
        raise PanelRenderError(
            f"{intent.value.lower()} view needs an explicit axis choice for this data"
        )
    return suggestion.spec


class PanelComposer:
    """One panel's worker-side compose: frozen snapshot + display state -> BoardFrame.

    Stateful on purpose, and the state is exactly the display continuity a live
    panel needs: the colour/count limits already shown and the relim mode they
    were resolved under.  Recomputing limits from scratch every tick is what
    makes a live image flicker between frames; carrying them is what makes
    ``normal`` mean "hold what you have unless the data leaves it".

    One composer per panel per source.  A source change builds a new one, so a
    new stream can never inherit the previous one's colour window.
    """

    def __init__(
        self,
        panel_id: str,
        *,
        intent: ViewIntent = ViewIntent.IMAGE,
        size_name: str = DEFAULT_PANEL_SIZE,
        pixel_ratio: float = 1.0,
        selection=None,
        label: str = "",
        value_label: str = "Signal",
        view: ViewSpec | None = None,
        rolling_trace: bool = False,
        rolling_distribution: bool = False,
    ) -> None:
        from .figure import dataset_contract_for

        try:
            dataset_contract_for(intent)
        except ValueError as error:
            raise PanelRenderError(str(error)) from error
        self._panel_id = str(panel_id)
        self._intent = intent
        self._surface_geometry = panel_surface_geometry(
            size_name,
            pixel_ratio=pixel_ratio,
        )
        self._selection = selection
        if view is not None and not isinstance(view, ViewSpec):
            raise TypeError("view must be ViewSpec or None")
        self._view = view
        self._label = str(label or panel_id)
        self._value_label = str(value_label or "Signal")
        self._rolling_distribution = bool(rolling_distribution)
        self._rolling_trace = bool(rolling_trace or rolling_distribution)
        self._dataset_id = DatasetId(self._panel_id)
        self._document: FigureDocument | None = None
        self._document_fingerprint = None
        self._source_figure_ref = None
        self._source_figure: DataFigure | None = None
        self._faceted_focus: FacetedPanelFocus | None = None
        self._faceted_focus_source: DataFigure | None = None
        self._faceted_focus_figure: DataFigure | None = None
        self._faceted_renderer_document: FigureDocument | None = None
        self._faceted_renderer = None
        self._faceted_overview_renderer = None
        self._renderer = None
        self._image_home_viewport = None
        self._image_color_cache_key = None
        self._image_color_cache_value = None
        # Display continuity, carried between ticks (see the class docstring).
        self._color_limits: tuple[float, float] | None = None
        self._image_relim_mode = None
        self._curve_y_limits = (0.0, 1.0)
        self._curve_relim_mode = None
        self._histogram_count_limits = None
        self._histogram_relim_mode = None
        self._histogram_log_count_axis = None
        self._sequence = 0

    # ----------------------------------------------------------------- view
    @property
    def intent(self) -> ViewIntent:
        return self._intent

    def document_for(self, schema) -> FigureDocument:
        """This panel's one-layer document for ``schema``, rebuilt when it changes.

        The document is keyed on the schema fingerprint rather than rebuilt per
        tick: a document identity that changed every frame would make every
        presentation look like a new panel to anything that tracks panel
        revisions.
        """

        fingerprint = schema.fingerprint
        if self._document is not None and fingerprint == self._document_fingerprint:
            return self._document
        view = view_for_schema(
            schema,
            self._intent,
            self._selection,
            view=self._view,
        )
        document = FigureDocument(
            f"panel-{self._panel_id}",
            0,
            (DatasetDescriptor(self._dataset_id, self._label, fingerprint),),
            (FigureLayer(self._panel_id, self._dataset_id, view),),
        )
        self._document = document
        self._document_fingerprint = fingerprint
        self._source_figure_ref = None
        self._source_figure = None
        self._discard_faceted_overview()
        self._discard_faceted_focus()
        self._image_home_viewport = None
        self._image_color_cache_key = None
        self._image_color_cache_value = None
        # The Agg surface is thread-affine to this composer's worker.  A schema
        # change replaces the view, so retire the old surface on that same
        # worker before forgetting it; merely dropping the Python reference
        # leaves the Figure/Canvas artist cycle alive until a later GC pass.
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        return document

    def close(self) -> None:
        """Release the worker-owned Agg surface on its owner thread."""

        renderer, self._renderer = self._renderer, None
        if renderer is not None:
            renderer.close()
        self._document = None
        self._document_fingerprint = None
        self._source_figure_ref = None
        self._source_figure = None
        self._discard_faceted_overview()
        self._discard_faceted_focus()
        self._image_home_viewport = None
        self._image_color_cache_key = None
        self._image_color_cache_value = None

    # -------------------------------------------------------------- compose
    def compose(self, snapshot, *, display):
        """Rasterize ONE frozen snapshot into a single-panel BoardFrame.

        ``snapshot`` is an ``OwnedSnapshot`` -- the immutable (ref, block) pair a
        monitor dataset materialises.  The returned frame carries the immutable
        raster plus the typed payload the board needs for interaction, so a
        later ROI drag reads the exact values that were shown rather than
        re-deriving them from a newer frame.
        """

        frame, _figure = self.compose_with_figure(
            snapshot,
            display=display,
        )
        return frame

    def compose_with_figure(
        self,
        snapshot,
        *,
        display,
        fit_result=None,
        fit_result_identity: str | None = None,
        histogram_projection_value_range: tuple[float, float] | None = None,
        check_cancelled=None,
    ) -> tuple[object, DataFigure]:
        """Rasterize once and return the exact already-evaluated figure too."""

        block = getattr(snapshot, "block", None)
        ref = getattr(snapshot, "ref", None)
        if block is None or ref is None:
            raise PanelRenderError("a panel front needs an owned (ref, block) snapshot")
        document = self.document_for(block.schema)
        # A display gesture changes only authored presentation state.  The
        # source revision is immutable, so evaluating its full image again for
        # every wheel/pan event is both redundant and observably expensive for
        # camera frames.  Retain exactly the last resolved revision; a new ref
        # replaces it, while this composer never accumulates historical data.
        figure = self._source_figure_for(document, snapshot)
        fit_result = self._validated_transient_fit(
            figure,
            fit_result,
            fit_result_identity,
        )
        evaluated = figure.evaluated
        layers = evaluated.layers
        if len(layers) != 1 or len(layers[0].cells) != 1:
            raise PanelRenderError("a panel document must evaluate to one cell")
        series = layers[0].cells[0].series
        if not series:
            raise PanelRenderError("a panel document evaluated no series")
        raster, payload = self._rasterize(
            figure,
            series,
            display,
            ref,
            fit_result=fit_result,
            fit_result_identity=fit_result_identity,
            histogram_projection_value_range=histogram_projection_value_range,
            check_cancelled=check_cancelled,
        )
        return (
            self._frame_for(
                ref,
                raster,
                payload,
            ),
            figure,
        )

    def compose_faceted(
        self,
        snapshot,
        *,
        display,
        focus: FacetedPanelFocus | None = None,
        fit_result=None,
        fit_result_identity: str | None = None,
        check_cancelled=None,
    ) -> FacetedPanelResult:
        """Compose one complete typed grid or one exact focused cell.

        The full ``DataFigure`` is evaluated once from one immutable snapshot.
        Overview encoding consumes that same evaluation for every cell.  Focus
        derives one typed panel from it without re-resolving the dataset, so a
        live grid never becomes N independently-latest cells.
        """

        if focus is not None and not isinstance(focus, FacetedPanelFocus):
            raise TypeError("focus must be FacetedPanelFocus or None")
        block = getattr(snapshot, "block", None)
        ref = getattr(snapshot, "ref", None)
        if block is None or ref is None:
            raise PanelRenderError(
                "a faceted panel needs an owned (ref, block) snapshot"
            )
        document = self.document_for(block.schema)
        source_figure = self._source_figure_for(document, snapshot)
        fit_result = self._validated_transient_fit(
            source_figure,
            fit_result,
            fit_result_identity,
        )
        layers = source_figure.evaluated.layers
        if (
            len(layers) != 1
            or len(layers[0].cells) <= 1
            or self._intent
            not in (
                ViewIntent.CURVE,
                ViewIntent.HISTOGRAM,
                ViewIntent.IMAGE,
                ViewIntent.METER,
            )
        ):
            raise PanelRenderError(
                "a grid requires one multi-cell CURVE, HISTOGRAM, IMAGE, or METER view"
            )
        histogram_projection = None
        histogram_fit_overlays = ()
        if self._intent is ViewIntent.HISTOGRAM and fit_result is not None:
            adjusted, histogram_projection, histogram_fit_overlays = (
                source_figure._histogram_fit_presentation(
                    fit_result,
                    result_identity=fit_result_identity,
                    display_state=display,
                    check_cancelled=check_cancelled,
                )
            )
            if adjusted != display:
                raise PanelRenderError(
                    "histogram Fit grid was not seeded from its committed bins"
                )
        if focus is None:
            if check_cancelled is not None:
                check_cancelled()
            rendered_figure = source_figure
            if fit_result is not None:
                rendered_figure = source_figure.with_fit_results(
                    {source_figure.document.layers[0].layer_id: fit_result}
                )
            renderer = self._faceted_overview_agg()
            try:
                raster, regions = renderer.render(
                    rendered_figure.document,
                    rendered_figure.evaluated,
                    dict(rendered_figure.fit_results),
                    display_state=display,
                    histogram_projection=histogram_projection,
                    histogram_fit_overlays=histogram_fit_overlays,
                )
            except BaseException:
                if self._faceted_overview_renderer is renderer:
                    self._faceted_overview_renderer = None
                    renderer.close()
                raise
            if check_cancelled is not None:
                check_cancelled()
            if len(regions) != len(layers[0].cells):
                raise PanelRenderError(
                    "grid hit regions do not cover every evaluated cell"
                )
            overview = FacetedOverviewArtifact(
                source_figure,
                raster,
                regions,
                self._surface_geometry.logical_size,
            )
            return FacetedPanelResult(source_figure, overview=overview)

        focused = self._focused_figure_for(source_figure, focus)
        focused_display = display
        if self._intent is ViewIntent.HISTOGRAM:
            if not isinstance(display, FacetedHistogramDisplayState):
                raise PanelRenderError(
                    "histogram grid requires per-cell threshold state"
                )
            focused_display = display.display_for(focus.address)
        raster, payload = self._rasterize_focused(
            focused,
            focused_display,
            fit_result=fit_result,
            fit_result_identity=fit_result_identity,
            histogram_projection=histogram_projection,
            histogram_fit_overlays=(
                ()
                if not histogram_fit_overlays
                else histogram_fit_overlays[focus.panel_index]
            ),
            check_cancelled=check_cancelled,
        )
        frame = self._frame_for(
            ref,
            raster,
            payload,
        )
        return FacetedPanelResult(
            source_figure,
            frame=frame,
            focus=focus,
        )

    def compose_data_figure(
        self,
        figure: DataFigure,
        *,
        display,
        fit_result=None,
        fit_result_identity: str | None = None,
        histogram_projection_value_range: tuple[float, float] | None = None,
        check_cancelled=None,
    ) -> tuple[object, DataFigure]:
        """Render one already-evaluated Figure through this panel surface.

        Saved-figure and live-panel hosts differ in where their immutable
        ``DataFigure`` comes from, not in how it is painted.  This seam keeps
        the archive's exact evaluated arrays (no second evaluation/copy) while
        reusing this composer's sole Agg/style/continuity owner.
        """

        document, _evaluated, _layer, evaluated_input = self._bind_data_figure(
            figure,
            faceted=False,
        )

        overlay_result = fit_result
        base = figure
        if figure.has_fit_overlays:
            if fit_result is not None or fit_result_identity is None:
                raise PanelRenderError(
                    "saved Figure Fit replay requires its exact result identity"
                )
            results = tuple(figure.fit_results.values())
            if len(results) != 1:
                raise PanelRenderError("single panel Figure requires one Fit result")
            overlay_result = results[0]
            base = figure.with_fit_results(None)
        else:
            overlay_result = self._validated_transient_fit(
                base,
                fit_result,
                fit_result_identity,
            )
        series = base.evaluated.layers[0].cells[0].series
        if not series:
            raise PanelRenderError("existing panel Figure evaluated no series")
        raster, payload = self._rasterize(
            base,
            series,
            display,
            evaluated_input.ref,
            fit_result=overlay_result,
            fit_result_identity=fit_result_identity,
            histogram_projection_value_range=histogram_projection_value_range,
            check_cancelled=check_cancelled,
        )
        return (
            self._frame_for(
                evaluated_input.ref,
                raster,
                payload,
            ),
            base,
        )

    def compose_data_figure_faceted(
        self,
        figure: DataFigure,
        *,
        display,
        fit_result=None,
        fit_result_identity: str | None = None,
        check_cancelled=None,
    ) -> FacetedPanelResult:
        """Render an already-evaluated multi-cell Figure as one grid overview.

        Archive replay must preserve the exact arrays and facet addresses that
        were loaded.  Re-resolving the source snapshot would produce an
        equivalent-looking but second ``EvaluatedFigureData`` authority.  This
        method is the grid counterpart of :meth:`compose_data_figure`: it feeds
        that existing evaluation into the same persistent Plot Panel renderer.
        """

        document, evaluated, layer, _evaluated_input = self._bind_data_figure(
            figure,
            faceted=True,
        )
        overlay_result = fit_result
        base = figure
        if figure.has_fit_overlays:
            if fit_result is not None or fit_result_identity is None:
                raise PanelRenderError(
                    "saved Figure Fit replay requires its exact result identity"
                )
            results = tuple(figure.fit_results.values())
            if len(results) != 1:
                raise PanelRenderError("faceted Figure requires one Fit result")
            overlay_result = results[0]
            base = figure.with_fit_results(None)
        else:
            overlay_result = self._validated_transient_fit(
                base,
                fit_result,
                fit_result_identity,
            )
        visible = (
            base
            if overlay_result is None
            else base.with_fit_results({layer.layer_id: overlay_result})
        )
        histogram_projection = None
        histogram_fit_overlays = ()
        if self._intent is ViewIntent.HISTOGRAM and overlay_result is not None:
            adjusted, histogram_projection, histogram_fit_overlays = (
                visible._histogram_fit_presentation(
                    overlay_result,
                    result_identity=fit_result_identity,
                    display_state=display,
                    check_cancelled=check_cancelled,
                )
            )
            if adjusted != display:
                raise PanelRenderError(
                    "histogram Fit grid was not seeded from its committed bins"
                )
        renderer = self._faceted_overview_agg()
        try:
            if check_cancelled is not None:
                check_cancelled()
            raster, regions = renderer.render(
                document,
                evaluated,
                dict(visible.fit_results),
                display_state=display,
                histogram_projection=histogram_projection,
                histogram_fit_overlays=histogram_fit_overlays,
            )
        except BaseException:
            if self._faceted_overview_renderer is renderer:
                self._faceted_overview_renderer = None
                renderer.close()
            raise
        if check_cancelled is not None:
            check_cancelled()
        if len(regions) != len(evaluated.layers[0].cells):
            raise PanelRenderError(
                "grid hit regions do not cover every evaluated cell"
            )
        overview = FacetedOverviewArtifact(
            visible,
            raster,
            regions,
            self._surface_geometry.logical_size,
        )
        return FacetedPanelResult(visible, overview=overview)

    def _bind_data_figure(
        self,
        figure: DataFigure,
        *,
        faceted: bool,
    ):
        """Validate and bind one existing Figure without evaluating it again."""

        if not isinstance(figure, DataFigure):
            raise TypeError("figure must be DataFigure")
        document = figure.document
        evaluated = figure.evaluated
        cells = () if len(evaluated.layers) != 1 else evaluated.layers[0].cells
        expected_cells = len(cells) > 1 if faceted else len(cells) == 1
        if (
            len(document.layers) != 1
            or len(document.datasets) != 1
            or len(evaluated.layers) != 1
            or not expected_cells
            or len(evaluated.inputs) != 1
        ):
            shape = "multiple cells" if faceted else "one cell"
            raise PanelRenderError(
                f"an existing panel Figure requires one layer, input, and {shape}"
            )
        layer = document.layers[0]
        if layer.view.intent is not self._intent:
            raise PanelRenderError("Figure intent differs from this panel surface")
        if self._view is not None and layer.view != self._view:
            raise PanelRenderError("Figure view differs from this panel contract")
        evaluated_input = evaluated.inputs[0]
        if evaluated_input.dataset_id != layer.dataset_id:
            raise PanelRenderError("Figure layer and evaluated input disagree")

        document_changed = (
            self._document != document or self._dataset_id != evaluated_input.dataset_id
        )
        if document_changed:
            renderer, self._renderer = self._renderer, None
            if renderer is not None:
                renderer.close()
            self._discard_faceted_overview()
            self._discard_faceted_focus()
            self._document = document
            self._document_fingerprint = document.datasets[0].schema_fingerprint
            self._dataset_id = evaluated_input.dataset_id
            self._color_limits = None
            self._image_relim_mode = None
            self._curve_y_limits = (0.0, 1.0)
            self._curve_relim_mode = None
            self._histogram_count_limits = None
            self._histogram_relim_mode = None
            self._histogram_log_count_axis = None
            self._image_home_viewport = None
            self._image_color_cache_key = None
            self._image_color_cache_value = None
        return document, evaluated, layer, evaluated_input

    def _source_figure_for(self, document, snapshot) -> DataFigure:
        """Resolve and evaluate exactly one immutable source revision once."""

        ref = snapshot.ref
        if (
            self._source_figure_ref == ref
            and self._source_figure is not None
            and self._source_figure.document is document
        ):
            return self._source_figure
        # A source revision changes the exact evaluated values, not the focus
        # selection or renderer topology.  Discard only the derived frozen
        # focus; the worker-owned Agg surface remains the same owner.
        self._faceted_focus_source = None
        self._faceted_focus_figure = None
        figure = DataFigure(
            document,
            ResolvedDatasetMap((ResolvedDataset(self._dataset_id, snapshot),)),
        )
        self._source_figure_ref = ref
        self._source_figure = figure
        self._image_color_cache_key = None
        self._image_color_cache_value = None
        return figure

    def _focused_figure_for(
        self,
        source_figure: DataFigure,
        focus: FacetedPanelFocus,
    ) -> DataFigure:
        """Return one display-only focus without rebuilding it per gesture."""

        if (
            self._faceted_focus_source is source_figure
            and self._faceted_focus == focus
            and self._faceted_focus_figure is not None
        ):
            return self._faceted_focus_figure
        if self._faceted_focus != focus:
            self._discard_faceted_focus()
        try:
            focused = source_figure.focused_typed_panel(
                focus.panel_index,
                expected_address=focus.address,
                expected_intent=self._intent,
            )
        except (TypeError, ValueError, IndexError, RuntimeError) as error:
            raise PanelRenderError(f"grid focus is stale: {error}") from error
        self._faceted_focus = focus
        self._faceted_focus_source = source_figure
        self._faceted_focus_figure = focused
        return focused

    def _discard_faceted_focus(self) -> None:
        """Retire the one focus surface; no historical focus cache is kept."""

        renderer, self._faceted_renderer = self._faceted_renderer, None
        if renderer is not None:
            renderer.close()
        self._faceted_renderer_document = None
        self._faceted_focus = None
        self._faceted_focus_source = None
        self._faceted_focus_figure = None

    def _frame_for(
        self,
        ref,
        raster,
        payload,
    ):
        """Stamp one already-rendered front with its exact source facts."""

        from .render import (
            BoardFrame,
            CoherenceStamp,
            PanelFrame,
            SourceIdentity,
        )

        self._sequence += 1
        stamp = CoherenceStamp((EvaluatedInput(self._dataset_id, ref),))
        source = SourceIdentity(
            self._dataset_id,
            ref.block_id,
            ref.stream_generation,
            ref.schema_fingerprint,
        )
        return BoardFrame(
            f"panel-board-{self._panel_id}",
            0,
            self._sequence,
            # One panel, so it is its own coherence group: nothing else on this
            # board has to have been frozen with it.
            (PanelFrame(self._panel_id, self._panel_id, source, stamp, raster, payload),),
        )

    # ------------------------------------------------------------ per intent
    @staticmethod
    def _validated_transient_fit(figure, fit_result, fit_result_identity):
        """Validate one exact draft pair without attaching saved Figure state."""

        if (fit_result is None) != (fit_result_identity is None):
            raise PanelRenderError(
                "fit result and fit result identity must be supplied together"
            )
        if fit_result is None:
            return None
        from zlc_data import FitResultBatch

        if not isinstance(fit_result, FitResultBatch):
            raise PanelRenderError("panel fit result has another type")
        if not isinstance(fit_result_identity, str) or not fit_result_identity:
            raise PanelRenderError("panel fit result identity must be non-empty")
        if fit_result.source_ref != figure.evaluated.inputs[0].ref:
            raise PanelRenderError("fit result belongs to another Figure source")
        return fit_result

    def _rasterize(
        self,
        figure,
        series,
        display,
        ref,
        *,
        fit_result=None,
        fit_result_identity: str | None,
        histogram_projection_value_range: tuple[float, float] | None = None,
        check_cancelled=None,
    ):
        evaluated = figure.evaluated
        data = series[0].data
        if self._intent is ViewIntent.IMAGE:
            if not isinstance(data, EvaluatedImage):
                raise PanelRenderError("this signal does not evaluate to an image")
            fit_overlay = (
                None
                if fit_result is None
                else figure.transient_single_panel_radial_fit_overlay(
                    fit_result,
                    result_identity=fit_result_identity,
                    check_cancelled=check_cancelled,
                )
            )
            return self._image_front(
                data,
                display,
                ref,
                projection_identity=self._projection_identity(
                    evaluated,
                    evaluated.layers[0],
                    evaluated.layers[0].cells[0],
                    series[0],
                ),
                fit_overlay=fit_overlay,
            )
        if self._intent is ViewIntent.CURVE:
            if any(not isinstance(item.data, EvaluatedCurve) for item in series):
                raise PanelRenderError("this signal does not evaluate to a curve")
            overlays = ()
            if fit_result is not None:
                from .fit_curve_projection import (
                    materialize_curve_fit_overlay_plan,
                )

                overlays = materialize_curve_fit_overlay_plan(
                    figure.transient_single_panel_curve_fit_overlay_plan(
                        fit_result,
                        result_identity=fit_result_identity,
                    ),
                    check_cancelled=check_cancelled,
                )
            return self._curve_front(evaluated, display, fit_overlays=overlays)
        if self._intent is ViewIntent.HISTOGRAM:
            if not isinstance(data, EvaluatedHistogram):
                raise PanelRenderError("this signal does not evaluate to a histogram")
            overlays = ()
            if fit_result is None:
                projection = self._histogram_projection(
                    evaluated,
                    display,
                    value_range=histogram_projection_value_range,
                )
            else:
                adjusted, projection, overlays_by_cell = (
                    figure._histogram_fit_presentation(
                        fit_result,
                        result_identity=fit_result_identity,
                        display_state=display,
                        check_cancelled=check_cancelled,
                    )
                )
                if adjusted != display or len(overlays_by_cell) != 1:
                    raise PanelRenderError(
                        "histogram Fit display was not seeded from its committed bins"
                    )
                overlays = overlays_by_cell[0]
                if (
                    histogram_projection_value_range is not None
                    and histogram_projection_value_range
                    != (float(projection.bin_edges[0]), float(projection.bin_edges[-1]))
                ):
                    raise PanelRenderError(
                        "histogram Fit range differs from its committed bins"
                    )
            return self._histogram_front(
                evaluated,
                display,
                projection=projection,
                fit_overlays=overlays,
            )
        if self._intent is ViewIntent.METER:
            if not isinstance(data, EvaluatedMeter):
                raise PanelRenderError("this signal does not evaluate to a meter")
            return self._agg().render_meter(
                evaluated,
                display_revision=getattr(display, "revision", 0),
            )
        raise PanelRenderError(f"no panel renderer for view intent {self._intent!r}")

    def _rasterize_focused(
        self,
        figure: DataFigure,
        display,
        *,
        fit_result=None,
        fit_result_identity: str | None = None,
        histogram_projection=None,
        histogram_fit_overlays=(),
        check_cancelled=None,
    ):
        """Use the existing single-panel renderer for one typed grid cell."""

        if self._intent is ViewIntent.IMAGE:
            layer = figure.evaluated.layers[0]
            series = layer.cells[0].series
            if len(series) != 1 or not isinstance(
                series[0].data,
                EvaluatedImage,
            ):
                raise PanelRenderError(
                    "focused image grid cell must contain exactly one image"
                )
            descriptor = figure.document.datasets[0]
            resolved = figure.datasets.resolve(descriptor.dataset_id)
            fit_overlay = None
            if fit_result is not None:
                fit_overlay = figure.transient_single_panel_radial_fit_overlay(
                    fit_result,
                    result_identity=fit_result_identity,
                    check_cancelled=check_cancelled,
                )
            return self._image_front(
                series[0].data,
                display,
                resolved.ref,
                projection_identity=self._projection_identity(
                    figure.evaluated,
                    layer,
                    layer.cells[0],
                    series[0],
                ),
                fit_overlay=fit_overlay,
            )

        renderer = self._faceted_agg(figure)
        try:
            if self._intent is ViewIntent.CURVE:
                overlays = ()
                if fit_result is not None:
                    from .fit_curve_projection import (
                        materialize_curve_fit_overlay_plan,
                    )

                    overlays = materialize_curve_fit_overlay_plan(
                        figure.transient_single_panel_curve_fit_overlay_plan(
                            fit_result,
                            result_identity=fit_result_identity,
                        ),
                        check_cancelled=check_cancelled,
                    )
                raster, payload = renderer.render_interactive_curve(
                    figure.evaluated,
                    display,
                    current_y_limits=self._curve_y_limits,
                    previous_relim_mode=(
                        display.relim_mode
                        if self._curve_relim_mode is None
                        else self._curve_relim_mode
                    ),
                    fit_overlays=overlays,
                )
                self._curve_y_limits = payload.viewport.y_limits
                self._curve_relim_mode = display.relim_mode
                return raster, payload
            if self._intent is ViewIntent.HISTOGRAM:
                if fit_result is None:
                    if histogram_projection is not None or histogram_fit_overlays:
                        raise PanelRenderError(
                            "focused histogram projection has no Fit result"
                        )
                    projection = self._histogram_projection(
                        figure.evaluated,
                        display,
                    )
                    overlays = ()
                else:
                    if histogram_projection is None:
                        raise PanelRenderError(
                            "focused histogram Fit lacks its root projection"
                        )
                    series = figure.evaluated.layers[0].cells[0].series
                    projection = HistogramBinProjection._from_committed_edges(
                        tuple(item.data.samples for item in series),
                        bins=display.bin_count,
                        bin_edges=histogram_projection.bin_edges,
                    )
                    root_overlays = tuple(histogram_fit_overlays)
                    if len(root_overlays) != len(series):
                        raise PanelRenderError(
                            "focused histogram Fit overlays differ from its series"
                        )
                    overlays = tuple(
                        replace(
                            overlay,
                            series_batch_address=item.batch_address,
                        )
                        for overlay, item in zip(
                            root_overlays,
                            series,
                            strict=True,
                        )
                    )
                raster, payload = renderer.render_interactive_histogram(
                    figure.evaluated,
                    display,
                    current_count_limits=self._histogram_count_limits,
                    previous_relim_mode=self._histogram_relim_mode,
                    previous_log_count_axis=self._histogram_log_count_axis,
                    bin_projection=projection,
                    fit_overlays=overlays,
                )
                self._histogram_count_limits = payload.viewport.count_limits
                self._histogram_relim_mode = display.relim_mode
                self._histogram_log_count_axis = display.log_count_axis
                return raster, payload
            if self._intent is ViewIntent.METER:
                if not isinstance(display, MeterDisplayState):
                    raise TypeError("focused meter grid requires MeterDisplayState")
                if fit_result is not None or fit_result_identity is not None:
                    raise PanelRenderError("METER display cannot carry a Fit overlay")
                return renderer.render_meter(
                    figure.evaluated,
                    display_revision=display.revision,
                )
        except BaseException:
            # A newly created surface may have acquired only part of its artist
            # topology before Matplotlib rejected the front.  Never retain that
            # half-prepared surface as the stable owner for a later gesture.
            if self._faceted_renderer is renderer:
                self._faceted_renderer = None
                self._faceted_renderer_document = None
                renderer.close()
            raise
        raise PanelRenderError(
            f"no focused renderer for view intent {self._intent!r}"
        )

    def _faceted_agg(self, figure: DataFigure):
        """One persistent Agg owner for the current exact focused cell."""

        from .matplotlib_render import SinglePanelAggRenderer

        if (
            self._faceted_renderer is not None
            and self._faceted_renderer_document == figure.document
        ):
            return self._faceted_renderer
        if self._faceted_renderer is not None:
            self._faceted_renderer.close()
        renderer = SinglePanelAggRenderer(
            figure.document,
            width=self._surface_geometry.raster_size[0],
            height=self._surface_geometry.raster_size[1],
            dpi=self._surface_geometry.dpi,
            size_name=self._surface_geometry.size_name,
            value_label=self._value_label,
            title=self._label,
        )
        self._faceted_renderer = renderer
        self._faceted_renderer_document = figure.document
        return renderer

    def _faceted_overview_agg(self):
        """One persistent board-level Agg owner for the live grid overview."""

        if self._faceted_overview_renderer is None:
            from .matplotlib_render import FacetedPanelAggRenderer

            self._faceted_overview_renderer = FacetedPanelAggRenderer(
                size_name=self._surface_geometry.size_name,
                width=self._surface_geometry.raster_size[0],
                height=self._surface_geometry.raster_size[1],
                dpi=self._surface_geometry.dpi,
                title=self._label,
                value_label=self._value_label,
            )
        return self._faceted_overview_renderer

    def _discard_faceted_overview(self) -> None:
        renderer, self._faceted_overview_renderer = (
            self._faceted_overview_renderer,
            None,
        )
        if renderer is not None:
            renderer.close()

    def _agg(self):
        from .matplotlib_render import SinglePanelAggRenderer

        if self._renderer is None:
            if self._document is None:
                raise PanelRenderError("the panel has no document to render")
            self._renderer = SinglePanelAggRenderer(
                self._document,
                width=self._surface_geometry.raster_size[0],
                height=self._surface_geometry.raster_size[1],
                dpi=self._surface_geometry.dpi,
                rolling_trace=self._rolling_trace,
                rolling_distribution=self._rolling_distribution,
                value_label=self._value_label,
                title=self._label,
                size_name=self._surface_geometry.size_name,
            )
        if not isinstance(self._renderer, SinglePanelAggRenderer):
            raise PanelRenderError("panel renderer family changed without a source reset")
        return self._renderer

    def _image_agg(self):
        from .matplotlib_render import ImagePanelAggRenderer

        if self._renderer is None:
            self._renderer = ImagePanelAggRenderer(
                width=self._surface_geometry.raster_size[0],
                height=self._surface_geometry.raster_size[1],
                dpi=self._surface_geometry.dpi,
                size_name=self._surface_geometry.size_name,
            )
        if not isinstance(self._renderer, ImagePanelAggRenderer):
            raise PanelRenderError("panel renderer family changed without a source reset")
        return self._renderer

    def _image_front(
        self,
        data: EvaluatedImage,
        display: ImageDisplayState,
        ref,
        *,
        projection_identity: EvaluatedProjectionIdentity,
        fit_overlay=None,
    ):
        from .render import ImagePanelPayload

        color_cache_key = (
            data,
            display.relim_mode,
            display.fixed_color_limits,
            self._color_limits,
            self._image_relim_mode,
        )
        if color_cache_key == self._image_color_cache_key:
            data_range, color_limits = self._image_color_cache_value
        else:
            data_range, color_limits = resolve_image_color_limits(
                data,
                display,
                current_color_limits=self._color_limits,
                previous_relim_mode=self._image_relim_mode,
            )
        self._color_limits = color_limits
        self._image_relim_mode = display.relim_mode
        # Cache the post-resolution continuity state, which is the state the
        # next display-only gesture will actually observe.
        self._image_color_cache_key = (
            data,
            display.relim_mode,
            display.fixed_color_limits,
            self._color_limits,
            self._image_relim_mode,
        )
        self._image_color_cache_value = (data_range, color_limits)
        home_viewport = self._image_home_viewport
        if home_viewport is None:
            from .image_view import image_viewport_for_evaluated_image

            home_viewport = image_viewport_for_evaluated_image(data)
            self._image_home_viewport = home_viewport
        viewport = image_viewport_for_display_state(display, home_viewport)
        raster, raster_geometry = self._image_agg().render(
            data,
            viewport,
            display,
            color_limits=color_limits,
            data_range=data_range,
            title=self._label,
            value_label=self._value_label,
            projection_identity=projection_identity,
            fit_overlay=fit_overlay,
        )
        payload = ImagePanelPayload(
            image=data,
            evaluated_input=projection_identity.evaluated_input,
            # The viewport revision tracks the display revision because a
            # viewport is a property OF a display state; a pair that drifted
            # would let a pan land on a colour window that no longer exists.
            viewport=viewport,
            data_range=data_range,
            colormap=display.colormap,
            color_limits=color_limits,
            raster_geometry=raster_geometry,
            fit_overlay=fit_overlay,
        )
        return raster, payload

    @staticmethod
    def _projection_identity(evaluated, layer, cell, series):
        matches = tuple(
            item for item in evaluated.inputs if item.dataset_id == layer.dataset_id
        )
        if len(matches) != 1:
            raise PanelRenderError(
                "image projection has no unique evaluated dataset input"
            )
        return EvaluatedProjectionIdentity(
            evaluated.document_id,
            evaluated.document_revision,
            matches[0],
            layer.layer_id,
            layer.resolutions,
            cell.facet_address,
            series.batch_address,
            series.data,
        )

    def _curve_front(
        self,
        evaluated,
        display: CurveDisplayState,
        *,
        fit_overlays=(),
    ):
        raster, payload = self._agg().render_interactive_curve(
            evaluated,
            display,
            current_y_limits=self._curve_y_limits,
            previous_relim_mode=(
                display.relim_mode
                if self._curve_relim_mode is None
                else self._curve_relim_mode
            ),
            fit_overlays=fit_overlays,
        )
        self._curve_y_limits = payload.viewport.y_limits
        self._curve_relim_mode = display.relim_mode
        return raster, payload

    def _histogram_front(
        self,
        evaluated,
        display: HistogramDisplayState,
        *,
        projection: HistogramBinProjection,
        fit_overlays=(),
    ):
        raster, payload = self._agg().render_interactive_histogram(
            evaluated,
            display,
            current_count_limits=self._histogram_count_limits,
            previous_relim_mode=self._histogram_relim_mode,
            previous_log_count_axis=self._histogram_log_count_axis,
            bin_projection=projection,
            fit_overlays=fit_overlays,
        )
        self._histogram_count_limits = payload.viewport.count_limits
        self._histogram_relim_mode = display.relim_mode
        self._histogram_log_count_axis = display.log_count_axis
        return raster, payload

    @staticmethod
    def _histogram_projection(
        evaluated,
        display: HistogramDisplayState,
        *,
        value_range: tuple[float, float] | None = None,
    ) -> HistogramBinProjection:
        """Freeze full-sample bins; x-view zoom never changes Fit authority."""

        if not isinstance(display, HistogramDisplayState):
            raise TypeError("histogram projection requires HistogramDisplayState")
        if len(evaluated.layers) != 1 or len(evaluated.layers[0].cells) != 1:
            raise PanelRenderError("histogram projection requires one logical panel")
        series = evaluated.layers[0].cells[0].series
        histograms = tuple(item.data for item in series)
        if not histograms or any(
            not isinstance(item, EvaluatedHistogram) for item in histograms
        ):
            raise PanelRenderError("histogram projection requires histogram series")
        return HistogramBinProjection(
            tuple(item.samples for item in histograms),
            bins=display.bin_count,
            value_range=value_range,
        )
