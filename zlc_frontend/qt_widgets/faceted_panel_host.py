"""One stable overview/focus host for a typed faceted figure.

The overview is one immutable raster composed from one ``DataFigure`` revision.
It therefore cannot tear across cells.  A double-click resolves one of the
overview's exact ``FigurePanelRegion`` values and asks the owner to render that
cell.  The focused cell uses :class:`SinglePanelHost`, so zoom/pan/selectors and
their exact interaction origin have the same owner as every other typed panel.

Both surfaces are constructed once.  Moving between overview and focus only
changes the current stack page; no QWidget, renderer, or selector family is
rebuilt.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..data_figure import FacetedOverviewArtifact
from ..render import BoardFrame
from .frozen_raster import FrozenRasterView
from .panel_host import SinglePanelHost


class FacetedPanelHost(QtWidgets.QWidget):
    """Atomic grid overview plus one exact interactive focus surface."""

    focusRequested = QtCore.pyqtSignal(int, object)
    overviewRequested = QtCore.pyqtSignal()

    rangeSelected = QtCore.pyqtSignal(object)
    viewCommitted = QtCore.pyqtSignal(object)
    thresholdsCommitted = QtCore.pyqtSignal(object)
    crossSelected = QtCore.pyqtSignal(object)
    rectangleSelected = QtCore.pyqtSignal(object)
    colorLimitsCommitted = QtCore.pyqtSignal(object)

    def __init__(
        self,
        panel_id: str,
        *,
        empty_text: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel_id = str(panel_id)
        self._overview_artifact: FacetedOverviewArtifact | None = None
        self._selectors_on = False

        self._stack = QtWidgets.QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setSpacing(0)

        self._overview = FrozenRasterView(
            f"{self._panel_id}-overview",
            empty_text=empty_text,
        )
        self._focus = SinglePanelHost(
            self._panel_id,
            empty_text=empty_text,
        )
        self._stack.addWidget(self._overview)
        self._stack.addWidget(self._focus)
        self._stack.setCurrentWidget(self._overview)

        self._overview.normalizedDoubleClicked.connect(
            self._resolve_overview_hit
        )
        self._focus.board.installEventFilter(self)
        self._focus.rangeSelected.connect(self.rangeSelected.emit)
        self._focus.viewCommitted.connect(self.viewCommitted.emit)
        self._focus.thresholdsCommitted.connect(
            self.thresholdsCommitted.emit
        )
        self._focus.crossSelected.connect(self.crossSelected.emit)
        self._focus.rectangleSelected.connect(self.rectangleSelected.emit)
        self._focus.colorLimitsCommitted.connect(
            self.colorLimitsCommitted.emit
        )

    @property
    def board(self):
        """The sole interactive board, used only while a cell is focused."""

        return self._focus.board

    @property
    def front_frame(self) -> BoardFrame | None:
        """The exact focused frame; overview is a raw raster and returns ``None``."""

        if self._stack.currentWidget() is not self._focus:
            return None
        return self._focus.front_frame

    @property
    def overview_artifact(self) -> FacetedOverviewArtifact | None:
        """The indivisible overview currently accepted by this host."""

        return self._overview_artifact

    @property
    def showing_overview(self) -> bool:
        return self._stack.currentWidget() is self._overview

    def present_overview(
        self,
        artifact: FacetedOverviewArtifact,
    ) -> None:
        """Atomically replace the complete overview and its exact hit map."""

        if not isinstance(artifact, FacetedOverviewArtifact):
            raise TypeError("faceted host requires FacetedOverviewArtifact")
        geometry_changes = (
            self.width(), self.height()
        ) != artifact.logical_size
        if geometry_changes:
            self.setUpdatesEnabled(False)
        try:
            self._overview.present_raster(artifact.raster)
            self._overview_artifact = artifact
            self.set_logical_size(artifact.logical_size)
            self._stack.setCurrentWidget(self._overview)
        finally:
            if geometry_changes:
                self.setUpdatesEnabled(True)
                self.update()

    def present_frame(
        self,
        frame: BoardFrame,
        *,
        logical_size: tuple[int, int] | None = None,
    ) -> None:
        """Present one exact focused cell through the shared panel host."""

        self._focus.present_frame(frame, logical_size=logical_size)
        if logical_size is not None:
            self.setFixedSize(*logical_size)
        self._focus.set_selectors_enabled(self._selectors_on)
        self._stack.setCurrentWidget(self._focus)
        self._focus.board.setFocus(QtCore.Qt.OtherFocusReason)

    def set_logical_size(self, logical_size: tuple[int, int]) -> None:
        """Give overview and focused-cell views the same authored plot extent."""

        self._focus.set_logical_size(logical_size)
        self.setFixedSize(*logical_size)

    def set_selectors_enabled(self, on: bool) -> None:
        self._selectors_on = bool(on)
        self._focus.set_selectors_enabled(self._selectors_on)

    def visible_interaction_origin(self):
        if self._stack.currentWidget() is not self._focus:
            return None
        return self._focus.visible_interaction_origin()

    def selection_for_rectangle_gesture(self, gesture):
        """Resolve an Area only against the exact focused cell front."""

        if self._stack.currentWidget() is not self._focus:
            raise RuntimeError(
                "faceted overview has no interactive rectangle selection"
            )
        return self._focus.selection_for_rectangle_gesture(gesture)

    def selection_for_curve_range_gesture(self, gesture):
        """Resolve an Area only against the exact focused cell front."""

        if self._stack.currentWidget() is not self._focus:
            raise RuntimeError(
                "faceted overview has no interactive curve selection"
            )
        return self._focus.selection_for_curve_range_gesture(gesture)

    def area_commit_for_range_gesture(self, gesture):
        if self._stack.currentWidget() is not self._focus:
            raise RuntimeError("faceted overview has no interactive Area")
        return self._focus.area_commit_for_range_gesture(gesture)

    def area_commit_for_rectangle_gesture(self, gesture):
        if self._stack.currentWidget() is not self._focus:
            raise RuntimeError("faceted overview has no interactive Area")
        return self._focus.area_commit_for_rectangle_gesture(gesture)

    def cross_commit_for_gesture(self, gesture):
        if self._stack.currentWidget() is not self._focus:
            raise RuntimeError("faceted overview has no interactive Cross")
        return self._focus.cross_commit_for_gesture(gesture)

    def discard_pending_interaction(self, origin) -> bool:
        return self._focus.discard_pending_interaction(origin)

    def clear(self) -> None:
        self._overview_artifact = None
        self._overview.clear()
        self._focus.clear()
        self._stack.setCurrentWidget(self._overview)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        """Match Main's focused-grid return gestures without extra chrome."""

        if watched is self._focus.board and not self.showing_overview:
            if (
                event.type() == QtCore.QEvent.MouseButtonDblClick
                and event.button() == QtCore.Qt.LeftButton
            ) or (
                event.type() == QtCore.QEvent.KeyPress
                and event.key() == QtCore.Qt.Key_Escape
            ):
                self.overviewRequested.emit()
                return True
        return super().eventFilter(watched, event)

    @QtCore.pyqtSlot(float, float)
    def _resolve_overview_hit(self, x: float, y: float) -> None:
        if not self.showing_overview:
            return
        hits = tuple(
            (index, region)
            for index, region in enumerate(
                ()
                if self._overview_artifact is None
                else self._overview_artifact.regions
            )
            if region.contains(x, y)
        )
        if len(hits) != 1:
            return
        index, region = hits[0]
        if region.focus_selection is None:  # constructor/present closes this
            raise RuntimeError("faceted overview region lost its selection")
        self.focusRequested.emit(index, region.focus_selection)


__all__ = ["FacetedPanelHost"]
