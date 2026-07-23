"""Compose ONE panel's raster front from a frozen snapshot.  Worker-side only.

A console card and a Workbench narrow window both need the same thing: given an
immutable snapshot of a dataset plus the operator's display state, produce the
front a board can paint.  The derivation -- which view a schema admits, what
document wraps it, how the display state resolves colour limits -- is written
ONCE here, so two hosts cannot drift into two answers for the same data.

Nothing here touches Qt.  The product is a :class:`~zlc_frontend.render.BoardFrame`
of immutable bytes, which is what lets a megapixel frame be rasterized on a
worker while the GUI thread stays responsive; the host presents it unchanged.

Why this exists next to :class:`zlc_workbench.live.LiveBoardController`: that
controller owns ONE source's change listener and drives a fixed image+scalar
board.  A console board is N independent cards reading ONE per-tick freeze, so
its panels cannot each own the source.  Both, however, must rasterize a frame
the same way -- that shared step is what this module holds.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    FigureEvaluator,
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
from .histogram_display import HistogramDisplayState
from .image_display import ImageDisplayState

__all__ = [
    "PanelComposer",
    "FacetedPanelFocus",
    "FacetedPanelResult",
    "PanelProvenance",
    "PanelRenderError",
    "display_state_for_intent",
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
    or one ordinary ``BoardFrame`` for the selected cell.  ``figure`` is the
    exact typed value behind that visible surface and crosses the worker
    boundary only after its evaluation is complete.
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


def display_state_for_intent(intent: ViewIntent):
    """The display-state type an intent's rasterizer consumes.

    One mapping, so a host storing "the panel's display state" and this module
    reading it cannot disagree about which knobs a kind even has.
    """

    if intent is ViewIntent.IMAGE:
        return ImageDisplayState
    if intent is ViewIntent.CURVE:
        return CurveDisplayState
    if intent is ViewIntent.HISTOGRAM:
        return HistogramDisplayState
    if intent is ViewIntent.METER:
        return None
    raise PanelRenderError(f"no display state for view intent {intent!r}")


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
        selection=None,
        label: str = "",
        view: ViewSpec | None = None,
    ) -> None:
        from .figure import dataset_contract_for

        try:
            dataset_contract_for(intent)
        except ValueError as error:
            raise PanelRenderError(str(error)) from error
        self._panel_id = str(panel_id)
        self._intent = intent
        self._size = (int(size[0]), int(size[1]))
        self._selection = selection
        if view is not None and not isinstance(view, ViewSpec):
            raise TypeError("view must be ViewSpec or None")
        self._view = view
        self._label = str(label or panel_id)
        self._dataset_id = DatasetId(self._panel_id)
        self._evaluator = FigureEvaluator()
        self._document: FigureDocument | None = None
        self._document_fingerprint = None
        self._renderer = None
        # Display continuity, carried between ticks (see the class docstring).
        self._color_limits: tuple[float, float] | None = None
        self._image_relim_mode = None
        self._curve_y_limits = None
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

    # -------------------------------------------------------------- compose
    def compose(self, snapshot, *, display, provenance: PanelProvenance):
        """Rasterize ONE frozen snapshot into a single-panel BoardFrame.

        ``snapshot`` is an ``OwnedSnapshot`` -- the immutable (ref, block) pair a
        monitor dataset materialises.  The returned frame carries the immutable
        raster plus the typed payload the board needs for interaction, so a
        later ROI drag reads the exact values that were shown rather than
        re-deriving them from a newer frame.
        """

        block = getattr(snapshot, "block", None)
        ref = getattr(snapshot, "ref", None)
        if block is None or ref is None:
            raise PanelRenderError("a panel front needs an owned (ref, block) snapshot")
        document = self.document_for(block.schema)
        evaluated = self._evaluator.evaluate(
            document,
            ResolvedDatasetMap((ResolvedDataset(self._dataset_id, snapshot),)),
        )
        layers = evaluated.layers
        if len(layers) != 1 or len(layers[0].cells) != 1:
            raise PanelRenderError("a panel document must evaluate to one cell")
        series = layers[0].cells[0].series
        if not series:
            raise PanelRenderError("a panel document evaluated no series")
        raster, payload = self._rasterize(
            evaluated, series, display, ref, block.schema,
        )
        return self._frame_for(
            document,
            ref,
            raster,
            payload,
            display,
            provenance,
        )

    def compose_faceted(
        self,
        snapshot,
        *,
        display,
        provenance: PanelProvenance,
        focus: FacetedPanelFocus | None = None,
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
        figure = DataFigure(document, datasets)
        layers = figure.evaluated.layers
        if (
            len(layers) != 1
            or len(layers[0].cells) <= 1
            or self._intent
            not in (
                ViewIntent.CURVE,
                ViewIntent.HISTOGRAM,
                ViewIntent.METER,
            )
        ):
            raise PanelRenderError(
                "a grid requires one multi-cell CURVE, HISTOGRAM, or METER view"
            )
        if focus is None:
            payload, regions = figure.to_png_bytes_with_panel_regions()
            if len(regions) != len(layers[0].cells):
                raise PanelRenderError(
                    "grid hit regions do not cover every evaluated cell"
                )
            return FacetedPanelResult(
                figure,
                overview_png=payload,
                regions=regions,
            )

        try:
            focused = figure.focused_typed_panel(
                focus.panel_index,
                expected_selection=focus.selection,
                expected_intent=self._intent,
            )
        except (TypeError, ValueError, IndexError, RuntimeError) as error:
            raise PanelRenderError(f"grid focus is stale: {error}") from error
        raster, payload = self._rasterize_focused(focused, display)
        frame = self._frame_for(
            focused.document,
            ref,
            raster,
            payload,
            display,
            provenance,
        )
        return FacetedPanelResult(
            focused,
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
    def _rasterize(self, evaluated, series, display, ref, schema):
        data = series[0].data
        if self._intent is ViewIntent.IMAGE:
            if not isinstance(data, EvaluatedImage):
                raise PanelRenderError("this signal does not evaluate to an image")
            return self._image_front(data, display, ref, schema)
        if self._intent is ViewIntent.CURVE:
            if any(not isinstance(item.data, EvaluatedCurve) for item in series):
                raise PanelRenderError("this signal does not evaluate to a curve")
            return self._curve_front(evaluated, display)
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

    def _rasterize_focused(self, figure: DataFigure, display):
        """Use the existing single-panel renderer for one typed grid cell."""

        from .matplotlib_render import SinglePanelAggRenderer

        renderer = SinglePanelAggRenderer(
            figure.document,
            width=self._size[0],
            height=self._size[1],
        )
        try:
            if self._intent is ViewIntent.CURVE:
                raster, payload = renderer.render_interactive_curve(
                    figure.evaluated,
                    display,
                    current_y_limits=self._curve_y_limits,
                    previous_relim_mode=self._curve_relim_mode,
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
            if self._intent is ViewIntent.METER:
                return renderer.render_meter(
                    figure.evaluated,
                    display_revision=getattr(display, "revision", 0),
                )
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
            )
        return self._renderer

    def _image_front(self, data: EvaluatedImage, display: ImageDisplayState, ref, schema):
        from .image_raster import rasterize_image_indexed8
        from .image_view import ImageViewportTransform
        from .render import ImagePanelPayload
        from .render_style import indexed_colormap

        raster, data_range, histogram, color_limits = rasterize_image_indexed8(
            data,
            display,
            current_color_limits=self._color_limits,
            previous_relim_mode=self._image_relim_mode,
        )
        self._color_limits = color_limits
        self._image_relim_mode = display.relim_mode
        payload = ImagePanelPayload(
            image=data,
            evaluated_input=EvaluatedInput(self._dataset_id, ref),
            # The viewport revision tracks the display revision because a
            # viewport is a property OF a display state; a pair that drifted
            # would let a pan land on a colour window that no longer exists.
            viewport=ImageViewportTransform(
                self._image_axis_specs(data, schema),
                viewport_revision=display.revision,
            ),
            data_range=data_range,
            histogram_counts=histogram,
            base_palette=indexed_colormap(display.colormap.value),
            color_limits=color_limits,
        )
        return raster, payload

    @staticmethod
    def _image_axis_specs(data: EvaluatedImage, schema):
        """The (y, x) declared AxisSpecs behind an evaluated image.

        Matched by axis id rather than by position: a viewport maps pointer
        coordinates back onto the DECLARED axes, so pairing it with anything but
        the axes the image was actually drawn from would put a drag rectangle on
        the wrong pixels.
        """

        by_id = {axis.axis_id: axis for axis in schema.cell_schema.data_axes}
        try:
            return (by_id[data.y_axis.axis_id], by_id[data.x_axis.axis_id])
        except KeyError as error:
            raise PanelRenderError(
                "the evaluated image names an axis its schema does not declare"
            ) from error

    def _curve_front(self, evaluated, display: CurveDisplayState):
        raster, payload = self._agg().render_interactive_curve(
            evaluated,
            display,
            current_y_limits=self._curve_y_limits,
            previous_relim_mode=self._curve_relim_mode,
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
