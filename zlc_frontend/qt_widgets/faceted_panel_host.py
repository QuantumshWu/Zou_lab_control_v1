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

from ..data_figure import FigurePanelRegion
from ..render import BoardFrame
from .fluent import FluentButton, scaled_px
from .frozen_raster import FrozenRasterView
from .panel_host import SinglePanelHost


class FacetedPanelHost(QtWidgets.QWidget):
    """Atomic grid overview plus one exact interactive focus surface."""

    focusRequested = QtCore.pyqtSignal(int, object)
    overviewRequested = QtCore.pyqtSignal()

    rangeSelected = QtCore.pyqtSignal(object)
    viewCommitted = QtCore.pyqtSignal(object)
    thresholdsCommitted = QtCore.pyqtSignal(object)
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
        self._regions: tuple[FigurePanelRegion, ...] = ()
        self._overview_png: bytes | None = None
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

        self._overview_button = FluentButton("Overview", self)
        self._overview_button.setObjectName("facetedPanelOverviewButton")
        self._overview_button.setToolTip("Return to the complete coherent grid")
        self._overview_button.clicked.connect(self.overviewRequested.emit)
        self._overview_button.hide()

        self._overview.normalizedDoubleClicked.connect(
            self._resolve_overview_hit
        )
        self._focus.rangeSelected.connect(self.rangeSelected.emit)
        self._focus.viewCommitted.connect(self.viewCommitted.emit)
        self._focus.thresholdsCommitted.connect(
            self.thresholdsCommitted.emit
        )
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
        """The exact focused frame; overview is encoded and returns ``None``."""

        if self._stack.currentWidget() is not self._focus:
            return None
        return self._focus.front_frame

    @property
    def overview_png(self) -> bytes | None:
        """The exact currently cached overview bytes."""

        return self._overview_png

    @property
    def showing_overview(self) -> bool:
        return self._stack.currentWidget() is self._overview

    def present_overview(
        self,
        png_bytes: bytes,
        regions: tuple[FigurePanelRegion, ...],
    ) -> None:
        """Atomically replace the complete overview and its exact hit map."""

        if not isinstance(png_bytes, bytes):
            raise TypeError("faceted overview must be owned PNG bytes")
        resolved = tuple(regions)
        if len(resolved) <= 1 or any(
            not isinstance(region, FigurePanelRegion) for region in resolved
        ):
            raise ValueError(
                "faceted overview requires multiple FigurePanelRegion values"
            )
        if len({region.key for region in resolved}) != len(resolved):
            raise ValueError("faceted overview region keys must be unique")
        if any(region.focus_selection is None for region in resolved):
            raise ValueError("faceted overview regions require exact selections")
        self._overview.present_encoded(png_bytes, image_format="PNG")
        self._overview_png = png_bytes
        self._regions = resolved
        self._stack.setCurrentWidget(self._overview)
        self._overview_button.hide()

    def present_frame(self, frame: BoardFrame) -> None:
        """Present one exact focused cell through the shared panel host."""

        self._focus.present_frame(frame)
        self._focus.set_selectors_enabled(self._selectors_on)
        self._stack.setCurrentWidget(self._focus)
        self._place_overview_button()
        self._overview_button.show()
        self._overview_button.raise_()

    def set_selectors_enabled(self, on: bool) -> None:
        self._selectors_on = bool(on)
        self._focus.set_selectors_enabled(self._selectors_on)

    def visible_interaction_origin(self):
        if self._stack.currentWidget() is not self._focus:
            return None
        return self._focus.visible_interaction_origin()

    def discard_pending_interaction(self, origin) -> bool:
        return self._focus.discard_pending_interaction(origin)

    def clear(self) -> None:
        self._regions = ()
        self._overview_png = None
        self._overview.clear()
        self._focus.clear()
        self._stack.setCurrentWidget(self._overview)
        self._overview_button.hide()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._place_overview_button()

    def _place_overview_button(self) -> None:
        hint = self._overview_button.sizeHint()
        self._overview_button.resize(hint)
        inset = scaled_px(8, minimum=5)
        self._overview_button.move(
            max(inset, self.width() - hint.width() - inset),
            inset,
        )

    @QtCore.pyqtSlot(float, float)
    def _resolve_overview_hit(self, x: float, y: float) -> None:
        if not self.showing_overview:
            return
        hits = tuple(
            (index, region)
            for index, region in enumerate(self._regions)
            if region.contains(x, y)
        )
        if len(hits) != 1:
            return
        index, region = hits[0]
        if region.focus_selection is None:  # constructor/present closes this
            raise RuntimeError("faceted overview region lost its selection")
        self.focusRequested.emit(index, region.focus_selection)


__all__ = ["FacetedPanelHost"]
