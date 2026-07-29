"""One Qt owner for an interactive Figure surface.

``SinglePanelHost`` and ``FacetedPanelHost`` are deliberately small raster/
gesture primitives.  A product surface needs one more invariant: the pixels,
the exact :class:`~zlc_frontend.data_figure.DataFigure` used to make them, and
the Area/Cross outputs resolved from a completed gesture must be promoted in
one GUI-thread transaction.  Historically every Workbench window rebuilt that
transaction itself.  This host owns it once without learning anything about a
Measurement, Processor, archive, or application shell.

Rendering remains worker-owned.  The host accepts only immutable completed
fronts and never imports Matplotlib.  Setting/Edit chrome remains a projection
of the frontend display contract; callers may place that chrome wherever their
window layout requires, but they do not reinterpret a gesture or a Figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PyQt5 import QtCore, QtWidgets

from zlc_data import (
    CoordinateRangeSelection,
    IndexRangeSelection,
    Selection,
)

from ..data_figure import DataFigure
from ..figure_outputs import FigureAreaCommit, FigureCrossCommit
from ..panel_render import FacetedPanelResult
from ..curve_display import CurveDisplayState
from ..histogram_display import (
    FacetedHistogramDisplayState,
    HistogramDisplayState,
)
from ..image_display import ImageDisplayState
from ..meter_display import MeterDisplayState
from ..plot_panel import PlotDisplayState, PlotPanelContract
from ..render import (
    BoardFrame,
    CurvePanelPayload,
    ImagePanelPayload,
    PanelFrame,
    SourceIdentity,
)
from ..selector import (
    CrossGesture,
    CurveRangeGesture,
    HistogramRangeGesture,
    ImageColorLimitsCommit,
    ImageViewportCommit,
    RectangleGesture,
)
from .board import QtRasterBoard
from .faceted_panel_host import FacetedPanelHost
from .panel_host import SinglePanelHost


@dataclass(frozen=True, slots=True)
class FigureSurfaceContext:
    """Exact semantic owners promoted with one accepted raster front."""

    figure: DataFigure | None
    display: PlotDisplayState
    contract: PlotPanelContract
    source_identity: SourceIdentity
    selector_figure: DataFigure | None = None

    def __post_init__(self) -> None:
        if self.figure is not None and not isinstance(self.figure, DataFigure):
            raise TypeError("Figure surface figure must be DataFigure or None")
        if not isinstance(self.contract, PlotPanelContract):
            raise TypeError("Figure surface contract must be PlotPanelContract")
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
            raise TypeError("Figure surface display has another type")
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be SourceIdentity")
        if self.selector_figure is not None and not isinstance(
            self.selector_figure, DataFigure
        ):
            raise TypeError("selector_figure must be DataFigure or None")
        if self.selector_figure is not None and self.figure is None:
            raise ValueError("selector_figure requires a complete Figure owner")
        if self.selector_figure is not None and not self.contract.figure.faceted:
            raise ValueError("selector_figure override belongs only to a focused grid")
        semantic_figure = self.selector_figure or self.figure
        if semantic_figure is not None and not _identity_matches_figure_input(
            self.source_identity,
            semantic_figure,
        ):
            raise ValueError(
                "Figure surface source identity belongs to another Figure input"
            )

    @classmethod
    def for_frame(
        cls,
        frame: BoardFrame,
        *,
        figure: DataFigure | None,
        display: PlotDisplayState,
        contract: PlotPanelContract,
        selector_figure: DataFigure | None = None,
    ) -> "FigureSurfaceContext":
        """Bind semantics to the one producer generation painted in ``frame``."""

        return cls(
            figure=figure,
            display=display,
            contract=contract,
            source_identity=_frame_source_identity(frame),
            selector_figure=selector_figure,
        )

    @classmethod
    def for_figure(
        cls,
        figure: DataFigure,
        *,
        display: PlotDisplayState,
        contract: PlotPanelContract,
    ) -> "FigureSurfaceContext":
        """Bind a raster overview to its single dataset generation."""

        if not isinstance(figure, DataFigure):
            raise TypeError("overview context requires DataFigure")
        inputs = tuple(figure.evaluated.inputs)
        if len(inputs) != 1:
            raise ValueError("interactive overview requires one dataset source")
        source = inputs[0]
        ref = source.ref
        return cls(
            figure=figure,
            display=display,
            contract=contract,
            source_identity=SourceIdentity(
                source.dataset_id,
                ref.block_id,
                ref.stream_generation,
                ref.schema_fingerprint,
            ),
        )


class FigureOutputAuthority(QtCore.QObject):
    """One Area/Cross command state shared by every view of one Figure panel."""

    changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._area: FigureAreaCommit | None = None
        self._cross: FigureCrossCommit | None = None

    @property
    def area_commit(self) -> FigureAreaCommit | None:
        return self._area

    @property
    def cross_commit(self) -> FigureCrossCommit | None:
        return self._cross

    def set_area(self, commit: FigureAreaCommit | None) -> None:
        if commit is not None and not isinstance(commit, FigureAreaCommit):
            raise TypeError("Area authority requires FigureAreaCommit or None")
        if commit != self._area:
            self._area = commit
            self.changed.emit()

    def set_cross(self, commit: FigureCrossCommit | None) -> None:
        if commit is not None and not isinstance(commit, FigureCrossCommit):
            raise TypeError("Cross authority requires FigureCrossCommit or None")
        if commit != self._cross:
            self._cross = commit
            self.changed.emit()

    def clear(self, *, notify: bool = True) -> None:
        changed = self._area is not None or self._cross is not None
        self._area = None
        self._cross = None
        if changed and notify:
            self.changed.emit()


class FigureSurfaceHost(QtWidgets.QWidget):
    """Stable ordinary/faceted Figure presenter and selector-output owner.

    The widget subtree is constructed exactly once.  ``faceted`` selects the
    surface topology, not a temporary rendering state.  Grid overview/focus
    swaps therefore retain the same host and the same selector switch.

    ``present_frame`` and ``present_faceted`` promote pixels and
    :class:`FigureSurfaceContext` atomically.  Completed Area/Cross gestures
    are then resolved against that admitted context inside this owner; shells
    consume :attr:`area_commit`/:attr:`cross_commit` or the
    ``figureOutputsChanged`` signal and never duplicate selector mapping.
    """

    rangeSelected = QtCore.pyqtSignal(object)
    rectangleSelected = QtCore.pyqtSignal(object)
    crossSelected = QtCore.pyqtSignal(object)
    viewCommitted = QtCore.pyqtSignal(object)
    colorLimitsCommitted = QtCore.pyqtSignal(object)
    thresholdsCommitted = QtCore.pyqtSignal(object)
    focusRequested = QtCore.pyqtSignal(int, object)
    overviewRequested = QtCore.pyqtSignal()
    panelDoubleClicked = QtCore.pyqtSignal(str)
    figureOutputsChanged = QtCore.pyqtSignal()
    interactionRejected = QtCore.pyqtSignal(str)
    interactionStarted = QtCore.pyqtSignal(object)
    interactionFinished = QtCore.pyqtSignal()

    def __init__(
        self,
        panel_id: str,
        *,
        faceted: bool = False,
        empty_text: str = "",
        output_authority: FigureOutputAuthority | None = None,
        panel_ids: tuple[str, ...] | None = None,
        columns: int = 1,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel_id = str(panel_id)
        self._faceted = bool(faceted)
        self._dynamic = panel_ids is not None
        if self._dynamic and self._faceted:
            raise ValueError("dynamic multi-panel surface cannot be faceted")
        self._context: FigureSurfaceContext | None = None
        self._fit_projection_key: tuple[object, ...] | None = None
        self._dynamic_area_candidates: dict[
            str, tuple[float, float, float, float]
        ] = {}
        if output_authority is not None and not isinstance(
            output_authority, FigureOutputAuthority
        ):
            raise TypeError("output_authority must be FigureOutputAuthority or None")
        self._output_authority = output_authority or FigureOutputAuthority(self)
        self._closed = False

        if self._dynamic:
            self._presenter = QtRasterBoard(
                tuple(panel_ids or ()),
                self,
                columns=columns,
                empty_text=empty_text,
            )
        else:
            self._presenter = (
                FacetedPanelHost(
                    self._panel_id,
                    empty_text=empty_text,
                    parent=self,
                )
                if self._faceted
                else SinglePanelHost(
                    self._panel_id,
                    empty_text=empty_text,
                    parent=self,
                )
            )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._presenter)

        # A surface transaction starts on pointer press, not on the first
        # motion-derived commit.  Forward the low-level lifecycle so an
        # application owner can pin the exact value/Figure ancestry that is
        # still painted while a live producer continues publishing.
        self.board.interactionStarted.connect(self.interactionStarted.emit)
        self.board.interactionFinished.connect(self.interactionFinished.emit)

        self._output_authority.changed.connect(self.figureOutputsChanged.emit)

        if self._dynamic:
            self._presenter.crossSelected.connect(self.crossSelected.emit)
            self._presenter.imagePanelLeftDoubleClicked.connect(
                self.panelDoubleClicked.emit
            )
        else:
            self._presenter.rangeSelected.connect(self._accept_range)
            self._presenter.rectangleSelected.connect(self._accept_rectangle)
            self._presenter.crossSelected.connect(self._accept_cross)
            self._presenter.viewCommitted.connect(self.viewCommitted.emit)
            self._presenter.colorLimitsCommitted.connect(
                self.colorLimitsCommitted.emit
            )
            self._presenter.thresholdsCommitted.connect(
                self.thresholdsCommitted.emit
            )
        if isinstance(self._presenter, FacetedPanelHost):
            self._presenter.focusRequested.connect(self.focusRequested.emit)
            self._presenter.overviewRequested.connect(
                self.overviewRequested.emit
            )

    @property
    def presenter(self) -> SinglePanelHost | FacetedPanelHost | QtRasterBoard:
        """Low-level raster primitive for owner-only pending-intent receipts."""

        return self._presenter

    @property
    def faceted(self) -> bool:
        return self._faceted

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @property
    def board(self):
        """Read-only access to the single underlying Qt event target."""

        return self._presenter if self._dynamic else self._presenter.board

    @property
    def front_frame(self) -> BoardFrame | None:
        return self._presenter.front_frame

    @property
    def has_front(self) -> bool:
        return self._presenter.has_front

    @property
    def context(self) -> FigureSurfaceContext | None:
        return self._context

    @property
    def area_commit(self) -> FigureAreaCommit | None:
        return self._output_authority.area_commit

    @property
    def cross_commit(self) -> FigureCrossCommit | None:
        return self._output_authority.cross_commit

    @property
    def selectors_enabled(self) -> bool:
        return self._presenter.selectors_enabled

    @property
    def showing_overview(self) -> bool:
        return bool(
            isinstance(self._presenter, FacetedPanelHost)
            and self._presenter.showing_overview
        )

    @property
    def overview_artifact(self):
        if not isinstance(self._presenter, FacetedPanelHost):
            return None
        return self._presenter.overview_artifact

    def set_selectors_enabled(self, enabled: bool) -> None:
        self._require_open()
        self._presenter.set_selectors_enabled(bool(enabled))

    def set_interaction_ready(self, ready: bool) -> None:
        self._require_open()
        if self._dynamic:
            self._presenter.set_interaction_readiness(
                image=bool(ready),
                curve=False,
            )
            return
        self._presenter.set_interaction_ready(bool(ready))

    def set_logical_size(self, logical_size: tuple[int, int]) -> None:
        self._require_open()
        if self._dynamic:
            self._presenter.setFixedSize(*logical_size)
        else:
            self._presenter.set_logical_size(logical_size)
        self.setFixedSize(*logical_size)

    def present_image_grid(
        self,
        frame: BoardFrame,
        *,
        columns: int,
        logical_size: tuple[int, int],
    ) -> None:
        """Present and bind one coherent dynamic IMAGE board."""

        self._require_open()
        if not self._dynamic:
            raise RuntimeError("ordinary Figure surface cannot present an image grid")
        if not isinstance(frame, BoardFrame) or not frame.panels:
            raise TypeError("image grid requires a non-empty BoardFrame")
        panel_ids = tuple(panel.panel_id for panel in frame.panels)
        if any(
            not isinstance(panel.display_payload, ImagePanelPayload)
            for panel in frame.panels
        ):
            raise TypeError("dynamic Figure grid accepts only IMAGE panels")
        painted = self._presenter.front_frame
        needs_stage = (
            not self._presenter.has_front
            or self._presenter.panel_ids != panel_ids
            or (
                painted is not None
                and (painted.board_id, painted.layout_generation)
                != (frame.board_id, frame.layout_generation)
            )
        )
        self._presenter.setUpdatesEnabled(False)
        try:
            if needs_stage:
                self._presenter.stage_layout(
                    panel_ids,
                    board_id=frame.board_id,
                    layout_generation=frame.layout_generation,
                    columns=columns,
                )
            self._presenter.present(frame)
            self.set_logical_size(logical_size)
            for panel_id in tuple(self._presenter.panel_ids):
                self._presenter.unbind_rectangle_selector(panel_id)
            for panel in frame.panels:
                self._presenter.bind_rectangle_selector(
                    panel.panel_id,
                    panel.display_payload.viewport,
                    self._accept_dynamic_rectangle,
                    enabled=self._presenter.selectors_enabled,
                    interaction_callback=self._accept_dynamic_interaction,
                )
                bounds = self._dynamic_area_candidates.get(panel.panel_id)
                if bounds is not None:
                    self._presenter.set_image_rectangle_candidate(
                        bounds,
                        panel_id=panel.panel_id,
                    )
        finally:
            self._presenter.setUpdatesEnabled(True)
        self._presenter.updateGeometry()
        self._presenter.update()

    def present_frame(
        self,
        frame: BoardFrame,
        *,
        context: FigureSurfaceContext,
        logical_size: tuple[int, int] | None = None,
    ) -> None:
        """Atomically promote one ordinary/focused frame and its Figure facts."""

        self._require_open()
        if self._dynamic:
            raise RuntimeError("dynamic Figure grid requires present_image_grid")
        self._validate_context(context)
        semantic_figure = context.selector_figure or context.figure
        projection_key = (
            None
            if semantic_figure is None
            else _figure_projection_identity(semantic_figure, frame)
        )
        geometry_changes = logical_size is not None and (
            self.width(), self.height()
        ) != logical_size
        if geometry_changes:
            self.setUpdatesEnabled(False)
        try:
            # The raster primitive, this complete Figure surface, and the
            # semantic context are one accepted front.  In particular, do not
            # resize this outer host when a size request is authored: retain
            # the old front at its old extent until the worker returns the
            # raster composed for ``logical_size``, then promote all three in
            # this GUI-thread transaction.
            self._presenter.present_frame(frame, logical_size=logical_size)
            if logical_size is not None:
                self.setFixedSize(*logical_size)
            self._promote_context(context, projection_key)
        finally:
            if geometry_changes:
                self.setUpdatesEnabled(True)
                self.update()

    def present_faceted(
        self,
        result: FacetedPanelResult,
        *,
        context: FigureSurfaceContext,
        logical_size: tuple[int, int],
    ) -> None:
        """Atomically promote one coherent grid overview or focused child."""

        self._require_open()
        if self._dynamic:
            raise RuntimeError("dynamic Figure grid has no faceted overview")
        if not isinstance(self._presenter, FacetedPanelHost):
            raise RuntimeError("ordinary Figure surface cannot present a grid")
        if not isinstance(result, FacetedPanelResult):
            raise TypeError("faceted presentation requires FacetedPanelResult")
        if context.figure is not result.figure:
            raise ValueError("faceted pixels and Figure context have another owner")
        self._validate_context(context)
        if (
            result.overview is not None
            and result.overview.logical_size != logical_size
        ):
            raise ValueError(
                "faceted overview and PlotPanel contract have different geometry"
            )
        semantic_figure = context.selector_figure or context.figure
        projection_key = (
            None
            if result.overview is not None or semantic_figure is None
            else _figure_projection_identity(semantic_figure, result.frame)
        )
        geometry_changes = (
            self.width(), self.height()
        ) != logical_size
        if geometry_changes:
            self.setUpdatesEnabled(False)
        try:
            if result.overview is not None:
                self._presenter.present_overview(result.overview)
            else:
                self._presenter.present_frame(
                    result.frame,
                    logical_size=logical_size,
                )
            self.setFixedSize(*logical_size)
            self._promote_context(context, projection_key)
        finally:
            if geometry_changes:
                self.setUpdatesEnabled(True)
                self.update()

    def present_overview(
        self,
        artifact,
        *,
        context: FigureSurfaceContext,
    ) -> None:
        """Copy an already-admitted coherent overview into another view."""

        self._require_open()
        if not isinstance(self._presenter, FacetedPanelHost):
            raise RuntimeError("ordinary Figure surface cannot present an overview")
        self._validate_context(context)
        if context.figure is not artifact.figure:
            raise ValueError("overview and Figure context have another owner")
        logical_size = artifact.logical_size
        geometry_changes = (
            self.width(), self.height()
        ) != logical_size
        if geometry_changes:
            self.setUpdatesEnabled(False)
        try:
            self._presenter.present_overview(artifact)
            self.setFixedSize(*logical_size)
            self._promote_context(context, None)
        finally:
            if geometry_changes:
                self.setUpdatesEnabled(True)
                self.update()

    def visible_interaction_origin(self):
        if self._dynamic:
            raise RuntimeError("dynamic Figure grid requires a panel id")
        return self._presenter.visible_interaction_origin()

    def visible_image_origin(self, panel_id: str):
        if not self._dynamic:
            if panel_id != self._panel_id:
                return None
            return self.visible_interaction_origin()
        return self._presenter.visible_image_origin(panel_id)

    def image_selector_fault(self, panel_id: str):
        if not self._dynamic:
            if panel_id != self._panel_id:
                raise ValueError("panel id belongs to another Figure surface")
            return self._presenter.selector_fault
        return self._presenter.image_selector_fault(panel_id)

    def set_image_panel_ready(self, panel_id: str, ready: bool) -> None:
        if not self._dynamic:
            if panel_id != self._panel_id:
                raise ValueError("panel id belongs to another Figure surface")
            self._presenter.set_interaction_ready(bool(ready))
            return
        self._presenter.set_image_interaction_readiness(panel_id, bool(ready))

    def discard_pending_interaction(self, origin) -> bool:
        if self._dynamic:
            return self._presenter.discard_pending_image_interaction(origin)
        return self._presenter.discard_pending_interaction(origin)

    def unbind_interaction(self) -> None:
        if self._dynamic:
            for panel_id in tuple(self._presenter.panel_ids):
                self._presenter.unbind_rectangle_selector(panel_id)
            return
        self._presenter.unbind_interaction()

    def selection_for_rectangle_gesture(self, gesture):
        return self._presenter.selection_for_rectangle_gesture(gesture)

    def selection_for_curve_range_gesture(self, gesture):
        return self._presenter.selection_for_curve_range_gesture(gesture)

    def set_area_selection_candidate(self, selection: Selection | None) -> None:
        """Paint one already-authoritative Area through the current payload.

        This is used when Fit authoring reopens.  Named-axis-to-coordinate
        conversion belongs beside gesture-to-named-axis conversion, not in an
        archive or Workbench window.
        """

        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("Area candidate must be Selection or None")
        frame = self.front_frame
        if frame is None or len(frame.panels) != 1:
            raise RuntimeError("Area candidate requires one exact visible panel")
        payload = frame.panels[0].display_payload
        if isinstance(payload, ImagePanelPayload):
            self.board.set_selector_applied_selection(
                selection,
                panel_id=self._panel_id,
            )
            return
        if not isinstance(payload, CurvePanelPayload):
            raise TypeError("current Figure surface has no Fit Area candidate")
        span = None if selection is None else _curve_span_for_selection(
            selection,
            payload,
        )
        self.board.set_numeric_range_candidate(span, panel_id=self._panel_id)

    def set_rectangle_candidate(self, normalized_bounds) -> None:
        if self._dynamic:
            raise RuntimeError("dynamic Figure rectangle requires its panel gesture")
        self._presenter.set_rectangle_candidate(normalized_bounds)

    def set_range_candidate(self, x_span) -> None:
        if self._dynamic:
            raise RuntimeError("dynamic Figure grid has no numeric Area")
        self._presenter.set_range_candidate(x_span)

    def clear_outputs(self) -> None:
        self._output_authority.clear()

    def install_fit_overlays(
        self,
        source_figure: DataFigure,
        source_frame: BoardFrame,
        overlay_panel: PanelFrame,
    ) -> Literal["CURRENT", "LAGGING", "INCOMPATIBLE"]:
        """Install a worker-materialized Fit result without composing base pixels.

        ``source_figure`` and ``source_frame`` are the exact immutable
        single-panel front frozen when the Fit was submitted.  The host
        compares the authoritative View/facet while the board compares producer
        and draw-geometry semantics with the currently painted front, then
        paints only the existing backend-neutral vector primitives.  A
        different revision of the same compatible producer is visibly
        LAGGING; an incompatible result is not retained or painted.
        """

        self._require_open()
        if not isinstance(source_figure, DataFigure):
            raise TypeError("source_figure must be DataFigure")
        if self._dynamic:
            return "INCOMPATIBLE"
        if not isinstance(source_frame, BoardFrame) or len(source_frame.panels) != 1:
            raise TypeError("source_frame must contain one exact panel")
        if not _identity_matches_figure_input(
            _frame_source_identity(source_frame),
            source_figure,
        ):
            raise ValueError("Fit source Figure and frame have another input")
        source_projection = _figure_projection_identity(source_figure, source_frame)
        current_projection = self._fit_projection_key
        if current_projection is None or self.front_frame is None:
            self.clear_fit_overlays()
            return "INCOMPATIBLE"
        status = self.board._install_fit_overlays(
            source_frame,
            overlay_panel,
            source_projection_key=source_projection,
            current_projection_key=current_projection,
        )
        return status

    def clear_fit_overlays(self) -> None:
        """Clear only this surface's transient Fit result layer."""

        self._require_open()
        self.board._clear_fit_overlays()

    def clear(self) -> None:
        self._require_open()
        self._presenter.clear()
        self._context = None
        self._fit_projection_key = None
        if self._dynamic:
            self._dynamic_area_candidates.clear()
        self.clear_outputs()

    def close_surface(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context = None
        self._fit_projection_key = None
        clear = getattr(self._presenter, "clear", None)
        if callable(clear):
            clear()

    @QtCore.pyqtSlot(object)
    def _accept_dynamic_rectangle(self, gesture) -> None:
        if not isinstance(gesture, RectangleGesture):
            return
        if gesture.normalized_bounds is None:
            self._dynamic_area_candidates.pop(gesture.panel_id, None)
        else:
            self._dynamic_area_candidates[gesture.panel_id] = (
                gesture.normalized_bounds
            )
        self._presenter.set_image_rectangle_candidate(
            gesture.normalized_bounds,
            panel_id=gesture.panel_id,
        )
        self.rectangleSelected.emit(gesture)

    @QtCore.pyqtSlot(object)
    def _accept_dynamic_interaction(self, commit) -> None:
        if isinstance(commit, ImageViewportCommit):
            self.viewCommitted.emit(commit)
        elif isinstance(commit, ImageColorLimitsCommit):
            self.colorLimitsCommitted.emit(commit)
        else:
            self.interactionRejected.emit(
                "dynamic IMAGE surface received another interaction contract"
            )

    def _validate_context(self, context: FigureSurfaceContext) -> None:
        if not isinstance(context, FigureSurfaceContext):
            raise TypeError("context must be FigureSurfaceContext")
        if context.contract.panel_id != self._panel_id:
            raise ValueError("Figure context belongs to another surface")
        if context.contract.figure.faceted != self._faceted:
            raise ValueError("Figure context has another surface topology")

    def _promote_context(
        self,
        context: FigureSurfaceContext,
        projection_key: tuple[object, ...] | None,
    ) -> None:
        previous = self._context
        self._context = context
        self._fit_projection_key = projection_key
        previous_identity = _source_identity(previous)
        current_identity = _source_identity(context)
        if projection_key is None:
            self.board._clear_fit_overlays()
        else:
            self.board._reconcile_fit_projection(projection_key)
        if previous_identity != current_identity:
            self.clear_outputs()

    def _selector_figure(self) -> DataFigure | None:
        context = self._context
        if context is None:
            raise RuntimeError("selector surface has no admitted Figure context")
        return context.selector_figure or context.figure

    @QtCore.pyqtSlot(object)
    def _accept_range(self, gesture) -> None:
        if not isinstance(gesture, (CurveRangeGesture, HistogramRangeGesture)):
            self.rangeSelected.emit(gesture)
            return
        try:
            commit = self._presenter.area_commit_for_range_gesture(
                gesture,
                figure=self._selector_figure(),
            )
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self.interactionRejected.emit(str(error))
            commit = None
        self._set_area(commit)
        self.rangeSelected.emit(gesture)

    @QtCore.pyqtSlot(object)
    def _accept_rectangle(self, gesture) -> None:
        if not isinstance(gesture, RectangleGesture):
            self.rectangleSelected.emit(gesture)
            return
        try:
            commit = self._presenter.area_commit_for_rectangle_gesture(
                gesture,
                figure=self._selector_figure(),
            )
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self.interactionRejected.emit(str(error))
            commit = None
        self._set_area(commit)
        self._presenter.set_rectangle_candidate(gesture.normalized_bounds)
        self.rectangleSelected.emit(gesture)

    @QtCore.pyqtSlot(object)
    def _accept_cross(self, gesture) -> None:
        self.crossSelected.emit(gesture)
        if not isinstance(gesture, CrossGesture):
            return
        figure = self._selector_figure()
        if figure is None:
            return
        try:
            commit = self._presenter.cross_commit_for_gesture(
                gesture,
                figure=figure,
            )
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self.interactionRejected.emit(str(error))
            return
        self._output_authority.set_cross(commit)

    def _set_area(self, commit: FigureAreaCommit | None) -> None:
        self._output_authority.set_area(commit)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("FigureSurfaceHost is closed")


def _source_identity(context: FigureSurfaceContext | None):
    return None if context is None else context.source_identity


def _figure_projection_identity(
    figure: DataFigure,
    frame: BoardFrame,
) -> tuple[object, ...]:
    """Validate and return a precomputed typed overlay projection key."""

    if not isinstance(figure, DataFigure):
        raise TypeError("projection identity requires DataFigure")
    return figure._transient_fit_projection_key(frame)


def _frame_source_identity(frame: BoardFrame) -> SourceIdentity:
    if not isinstance(frame, BoardFrame) or not frame.panels:
        raise TypeError("Figure surface context requires a non-empty BoardFrame")
    identities = tuple(panel.source_identity for panel in frame.panels)
    if any(not isinstance(item, SourceIdentity) for item in identities):
        raise TypeError("interactive Figure surface requires a dataset source")
    source = identities[0]
    if any(item != source for item in identities[1:]):
        raise ValueError("one Figure surface cannot mix producer generations")
    return source


def _identity_matches_figure_input(
    identity: SourceIdentity,
    figure: DataFigure,
) -> bool:
    matches = tuple(
        item
        for item in figure.evaluated.inputs
        if item.dataset_id == identity.dataset_id
    )
    if len(matches) != 1:
        return False
    ref = matches[0].ref
    return (
        ref.block_id == identity.block_id
        and ref.stream_generation == identity.stream_generation
        and ref.schema_fingerprint == identity.schema_fingerprint
    )


def _curve_span_for_selection(
    selection: Selection,
    payload: CurvePanelPayload,
) -> tuple[float, float]:
    axis = payload.viewport.x_axis
    matches = tuple(
        term for term in selection.terms if term.axis_id == axis.axis_id
    )
    if len(matches) != 1:
        raise ValueError("curve Area does not name the displayed x axis")
    term = matches[0]
    if isinstance(term, CoordinateRangeSelection):
        return float(term.lower), float(term.upper)
    if isinstance(term, IndexRangeSelection):
        coordinates = axis.coordinates
        if term.stop > len(coordinates):
            raise IndexError("curve Area exceeds displayed coordinates")
        low = float(coordinates[term.start])
        high = float(coordinates[term.stop - 1])
        return min(low, high), max(low, high)
    raise ValueError("curve Area must preserve a non-empty range")


__all__ = [
    "FigureOutputAuthority",
    "FigureSurfaceContext",
    "FigureSurfaceHost",
]
