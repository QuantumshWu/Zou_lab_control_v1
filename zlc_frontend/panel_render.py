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

from dataclasses import dataclass
import math

from zlc_data import Selection

from .curve_display import CurveDisplayState
from .data_figure import DataFigure, FigurePanelRegion
from .figure import (
    DatasetDescriptor,
    DatasetId,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedInput,
    EvaluatedMeter,
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
    HistogramDisplayState,
)
from .image_display import (
    ImageDisplayState,
    image_viewport_for_display_state,
    resolve_image_color_limits,
)

__all__ = [
    "PanelComposer",
    "FacetedPanelFocus",
    "FacetedPanelResult",
    "PanelProvenance",
    "PanelRenderError",
    "view_for_schema",
]


class PanelRenderError(RuntimeError):
    """A snapshot that cannot be shown as asked, with the reason the host shows."""


@dataclass(frozen=True)
class PanelProvenance:
    """Where one composed front came from -- the facts a coherence stamp needs.

    The host supplies these because only the host knows them: the run that
    produced the data, the causation domain it belongs to, and the digest of the
    event the snapshot froze.  A composer that invented them would be attesting
    to a lineage it cannot see.
    """

    run_id: object
    epoch_id: object
    join_digest: str


@dataclass(frozen=True, slots=True)
class FacetedPanelFocus:
    """One exact display cell chosen from a previously painted overview."""

    panel_index: int
    selection: Selection

    def __post_init__(self) -> None:
        if (
            isinstance(self.panel_index, bool)
            or not isinstance(self.panel_index, int)
            or self.panel_index < 0
        ):
            raise ValueError("faceted focus panel_index must be non-negative")
        if not isinstance(self.selection, Selection):
            raise TypeError("faceted focus selection must be Selection")


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
    overview_png: bytes | None = None
    regions: tuple[FigurePanelRegion, ...] = ()
    frame: object | None = None
    focus: FacetedPanelFocus | None = None

    def __post_init__(self) -> None:
        from .render import BoardFrame

        if not isinstance(self.figure, DataFigure):
            raise TypeError("faceted result figure must be DataFigure")
        regions = tuple(self.regions)
        object.__setattr__(self, "regions", regions)
        overview = self.overview_png is not None
        focused = self.frame is not None
        if overview == focused:
            raise ValueError(
                "faceted result requires exactly one overview or focus front"
            )
        if overview:
            if not isinstance(self.overview_png, bytes):
                raise TypeError("faceted overview must be owned PNG bytes")
            if self.focus is not None:
                raise ValueError("faceted overview cannot carry focus")
            if len(regions) <= 1 or any(
                not isinstance(item, FigurePanelRegion) for item in regions
            ):
                raise ValueError(
                    "faceted overview requires multiple exact panel regions"
                )
        else:
            if not isinstance(self.frame, BoardFrame):
                raise TypeError("faceted focus frame must be BoardFrame")
            if self.focus is None:
                raise ValueError("faceted focus result requires its exact focus")
            if regions:
                raise ValueError("faceted focus does not carry overview regions")


def view_for_schema(
    schema,
    intent: ViewIntent,
    selection=None,
    *,
    view: ViewSpec | None = None,
):
    """The ViewSpec this schema admits for ``intent``.

    Raises rather than guessing: a schema that needs an explicit axis choice is
    a question for the operator, and silently picking an axis would put a
    plausible-looking wrong picture on the board.
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
        size: tuple[int, int] = (800, 520),
        size_name: str = "2x2",
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
        self._size = (int(size[0]), int(size[1]))
        from zlc_frontend.panel_size import panel_size_cells

        panel_size_cells(size_name)
        self._size_name = str(size_name)
        pixel_ratio = float(pixel_ratio)
        if not math.isfinite(pixel_ratio) or pixel_ratio <= 0.0:
            raise ValueError("pixel_ratio must be positive and finite")
        self._pixel_ratio = pixel_ratio
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
        self._histogram_count_scale = None
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
        self._image_home_viewport = None
        self._image_color_cache_key = None
        self._image_color_cache_value = None

    # -------------------------------------------------------------- compose
    def compose(self, snapshot, *, display, provenance: PanelProvenance):
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
            provenance=provenance,
        )
        return frame

    def compose_with_figure(
        self,
        snapshot,
        *,
        display,
        provenance: PanelProvenance,
        fit_result=None,
        fit_result_identity: str | None = None,
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
        if self._source_figure_ref == ref and self._source_figure is not None:
            figure = self._source_figure
        else:
            figure = DataFigure(
                document,
                ResolvedDatasetMap((ResolvedDataset(self._dataset_id, snapshot),)),
            )
            self._source_figure_ref = ref
            self._source_figure = figure
            self._image_color_cache_key = None
            self._image_color_cache_value = None
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
        )
        return (
            self._frame_for(
                document,
                ref,
                raster,
                payload,
                display,
                provenance,
            ),
            figure,
        )

    def compose_faceted(
        self,
        snapshot,
        *,
        display,
        provenance: PanelProvenance,
        focus: FacetedPanelFocus | None = None,
        fit_result=None,
        fit_result_identity: str | None = None,
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
        datasets = ResolvedDatasetMap(
            (ResolvedDataset(self._dataset_id, snapshot),)
        )
        source_figure = DataFigure(document, datasets)
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
            )
        ):
            raise PanelRenderError(
                "a grid requires one multi-cell CURVE, HISTOGRAM, or IMAGE view"
            )
        if focus is None:
            from .plot_layout import LIVE_PANEL_DPI

            options = {
                "size": self._size_name,
                "width": self._size[0],
                "height": self._size[1],
                "dpi": LIVE_PANEL_DPI * self._pixel_ratio,
                "display_state": display,
                "title": self._label,
                "value_label": self._value_label,
            }
            if fit_result is None:
                payload, regions = (
                    source_figure.to_panel_png_bytes_with_panel_regions(
                        **options
                    )
                )
            else:
                payload, regions = (
                    source_figure.transient_fit_to_panel_png_bytes_with_panel_regions(
                        fit_result,
                        **options,
                    )
                )
            if len(regions) != len(layers[0].cells):
                raise PanelRenderError(
                    "grid hit regions do not cover every evaluated cell"
                )
            return FacetedPanelResult(
                source_figure,
                overview_png=payload,
                regions=regions,
            )

        try:
            focused = source_figure.focused_typed_panel(
                focus.panel_index,
                expected_selection=focus.selection,
                expected_intent=self._intent,
            )
        except (TypeError, ValueError, IndexError, RuntimeError) as error:
            raise PanelRenderError(f"grid focus is stale: {error}") from error
        focused_display = display
        if self._intent is ViewIntent.HISTOGRAM:
            if not isinstance(display, FacetedHistogramDisplayState):
                raise PanelRenderError(
                    "histogram grid requires per-cell threshold state"
                )
            focused_display = display.display_for(focus.selection)
        raster, payload = self._rasterize_focused(
            focused,
            focused_display,
            fit_result=fit_result,
            fit_result_identity=fit_result_identity,
        )
        frame = self._frame_for(
            focused.document,
            ref,
            raster,
            payload,
            focused_display,
            provenance,
        )
        return FacetedPanelResult(
            source_figure,
            frame=frame,
            focus=focus,
        )

    def _frame_for(
        self,
        document,
        ref,
        raster,
        payload,
        display,
        provenance: PanelProvenance,
    ):
        """Stamp one already-rendered front with its exact source facts."""

        from .render import (
            BoardFrame,
            CoherenceStamp,
            PanelFrame,
            PanelPresentationIdentity,
            SourceIdentity,
        )

        self._sequence += 1
        presentation = PanelPresentationIdentity(
            self._panel_id,
            document.document_id,
            document.revision,
            0,
            getattr(display, "revision", 0) or 0,
        )
        stamp = CoherenceStamp(
            provenance.run_id,
            provenance.epoch_id,
            "single-source-event-payload",
            ref.schema_fingerprint,
            provenance.join_digest,
            (EvaluatedInput(self._dataset_id, ref),),
            (presentation,),
        )
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
                )
            )
            return self._image_front(
                data,
                display,
                ref,
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
                    )
                )
            return self._curve_front(evaluated, display, fit_overlays=overlays)
        if self._intent is ViewIntent.HISTOGRAM:
            if not isinstance(data, EvaluatedHistogram):
                raise PanelRenderError("this signal does not evaluate to a histogram")
            return self._histogram_front(evaluated, display)
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
                )
            return self._image_front(
                series[0].data,
                display,
                resolved.ref,
                fit_overlay=fit_overlay,
            )

        from .matplotlib_render import SinglePanelAggRenderer

        renderer = SinglePanelAggRenderer(
            figure.document,
            width=self._size[0],
            height=self._size[1],
            dpi=self._live_dpi(),
            size_name=self._size_name,
            value_label=self._value_label,
            title=self._label,
        )
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
                        )
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
                raster, payload = renderer.render_interactive_histogram(
                    figure.evaluated,
                    display,
                    current_count_limits=self._histogram_count_limits,
                    previous_relim_mode=self._histogram_relim_mode,
                    previous_count_scale=self._histogram_count_scale,
                )
                self._histogram_count_limits = payload.viewport.count_limits
                self._histogram_relim_mode = display.relim_mode
                self._histogram_count_scale = display.count_scale
                return raster, payload
        finally:
            renderer.close()
        raise PanelRenderError(
            f"no focused renderer for view intent {self._intent!r}"
        )

    def _agg(self):
        from .matplotlib_render import SinglePanelAggRenderer

        if self._renderer is None:
            if self._document is None:
                raise PanelRenderError("the panel has no document to render")
            self._renderer = SinglePanelAggRenderer(
                self._document, width=self._size[0], height=self._size[1],
                dpi=self._live_dpi(),
                rolling_trace=self._rolling_trace,
                rolling_distribution=self._rolling_distribution,
                value_label=self._value_label,
                title=self._label,
                size_name=self._size_name,
            )
        if not isinstance(self._renderer, SinglePanelAggRenderer):
            raise PanelRenderError("panel renderer family changed without a source reset")
        return self._renderer

    def _image_agg(self):
        from .matplotlib_render import ImagePanelAggRenderer

        if self._renderer is None:
            self._renderer = ImagePanelAggRenderer(
                width=self._size[0],
                height=self._size[1],
                dpi=self._live_dpi(),
                size_name=self._size_name,
            )
        if not isinstance(self._renderer, ImagePanelAggRenderer):
            raise PanelRenderError("panel renderer family changed without a source reset")
        return self._renderer

    def _live_dpi(self) -> float:
        from .plot_layout import LIVE_PANEL_DPI

        return LIVE_PANEL_DPI * self._pixel_ratio

    def _image_front(
        self,
        data: EvaluatedImage,
        display: ImageDisplayState,
        ref,
        *,
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
            distribution_identity=ref,
            fit_overlay=fit_overlay,
        )
        payload = ImagePanelPayload(
            image=data,
            evaluated_input=EvaluatedInput(self._dataset_id, ref),
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

    def _histogram_front(self, evaluated, display: HistogramDisplayState):
        raster, payload = self._agg().render_interactive_histogram(
            evaluated,
            display,
            current_count_limits=self._histogram_count_limits,
            previous_relim_mode=self._histogram_relim_mode,
            previous_count_scale=self._histogram_count_scale,
        )
        self._histogram_count_limits = payload.viewport.count_limits
        self._histogram_relim_mode = display.relim_mode
        self._histogram_count_scale = display.count_scale
        return raster, payload
