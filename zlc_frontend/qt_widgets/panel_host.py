"""One reusable interactive single-panel surface over :class:`QtRasterBoard`.

Every window that shows ONE rendered panel and wants the unified selector
family on it -- the pulse editor's preview, an Edit-tab snapshot, a standalone
viewer -- needs the same four things: the coherent-frame identity boilerplate
around :meth:`QtRasterBoard.present`, the gesture binding matched to the
payload's kind, the board-wide Selectors switch, and the logical-size pinning
that keeps a DPR-scaled raster blitting 1:1.  Writing those per window is how
a second selector stack starts; this host owns them ONCE.

The host stays display-only and domain-free (C12: Qt widgets never see
matplotlib or a domain package).  The window keeps only its own facts: how to
render its picture (any callable producing ``(RasterBuffer, payload)``), when
its content changed, and what to do with a completed gesture -- delivered as
Qt signals carrying the same typed intents the unified owner emits.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..render import (
    BoardFrame, CoherenceStamp, CurvePanelPayload, HistogramPanelPayload,
    ImagePanelPayload, PanelFrame, PanelPresentationIdentity,
    PulsePanelPayload, RasterBuffer, SourceIdentity)
from ..selector import (
    CurveRangeGesture, CurveViewportCommit, HistogramRangeGesture,
    HistogramThresholdCommit, HistogramViewportCommit, ImageColorLimitsCommit,
    ImageViewportCommit)
from .board import QtRasterBoard


class SinglePanelHost(QtWidgets.QWidget):
    """A QtRasterBoard hosting exactly one interactive panel.

    ``present_panel(raster, payload, ...)`` wraps the payload in a coherent
    one-panel :class:`BoardFrame` (provenance, stamp, presentation identity all
    derived from the caller's content key and revisions) and presents it.  The
    FIRST payload's type picks the gesture family -- pulse and curve speak the
    CURVE intent vocabulary, histogram its own -- and completed gestures come
    back as signals:

    * ``rangeSelected(object)``  -- an area drag's x-span (or ``None`` for a
      degenerate click).  The host already echoed the display-only candidate
      onto the board; the window decides what the selection MEANS.
    * ``viewCommitted(object)``  -- a wheel-zoom / pan commit's candidate
      viewport.  The window re-renders at ``candidate.x_limits`` and calls
      :meth:`present_panel` again with ``display_revision =
      candidate.display_revision`` so the accepted front matches the intent.
    """

    rangeSelected = QtCore.pyqtSignal(object)
    viewCommitted = QtCore.pyqtSignal(object)
    thresholdsCommitted = QtCore.pyqtSignal(object)
    # image family only: a completed DISPLAY ONLY rectangle (RectangleGesture)
    # and a clim-rail commit's fixed colour limits.  The window echoes a
    # rectangle candidate through ``board.set_image_rectangle_candidate``.
    rectangleSelected = QtCore.pyqtSignal(object)
    colorLimitsCommitted = QtCore.pyqtSignal(object)

    def __init__(self, panel_id: str = "panel", *,
                 group: str | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel_id = str(panel_id)
        self._group = str(group or panel_id)
        self._board = QtRasterBoard((self._panel_id,), columns=1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._board)
        self._bound_kind: str | None = None
        self._sequence = -1
        # The design's "Selectors default OFF" rule: the host remembers the
        # operator's switch itself (exactly like a console card) so a switch
        # flipped BEFORE the first present is replayed once the binding exists
        # -- the board refuses to arm with no healthy binding, and a refusal
        # inside a Qt slot would kill the process.
        self._selectors_on = False

    @property
    def board(self) -> QtRasterBoard:
        """The underlying unified interaction owner (read-mostly access)."""

        return self._board

    def set_selectors_enabled(self, on: bool) -> None:
        """The Selectors switch, same semantics as every console card: remember
        the desired state and gate the CURRENT binding now; a binding created
        later (first present) inherits it."""

        self._selectors_on = bool(on)
        if self._bound_kind is not None:
            self._board.set_selectors_enabled(self._selectors_on)

    def present_panel(self, raster: RasterBuffer, payload, *,
                      pixel_ratio: float = 1.0) -> tuple[int, int]:
        """Present one rendered panel as a coherent frame; returns the LOGICAL size.

        Every identity fact is DERIVED from the payload itself -- the
        provenance is ``payload.evaluated_input`` (its revision is the content
        revision, its schema fingerprint the stamp fingerprint) and the panel
        revision is the viewport's display revision -- so a window cannot hand
        the board a frame whose identity disagrees with its own payload.
        ``pixel_ratio`` is the screen ratio the raster was rendered at: the
        widget pins to the LOGICAL size so the whole-cell blit lands 1:1 on
        device pixels.
        """

        provenance = payload.evaluated_input
        fingerprint = provenance.ref.schema_fingerprint
        content_revision = int(provenance.ref.revision.value)
        display_revision = int(payload.viewport.display_revision)
        presentation = PanelPresentationIdentity(
            self._panel_id, self._group, content_revision, 0, display_revision)
        stamp = CoherenceStamp(
            self._group,
            f"{self._panel_id}-epoch-{display_revision}",
            f"{self._panel_id}-frame-{display_revision}",
            fingerprint,
            fingerprint,
            (provenance,),
            (presentation,),
        )
        source = SourceIdentity(
            provenance.dataset_id,
            provenance.ref.block_id,
            provenance.ref.stream_generation,
            fingerprint,
        )
        panel = PanelFrame(
            self._panel_id, self._group, source, stamp, raster, payload)
        self._sequence += 1
        self._board.present(BoardFrame(
            f"{self._group}-board", 0, self._sequence, (panel,)))
        self._ensure_binding(payload)
        ratio = float(pixel_ratio) or 1.0
        logical = (int(round(raster.width / ratio)),
                   int(round(raster.height / ratio)))
        self._board.setFixedSize(logical[0], logical[1])
        return logical

    def present_frame(self, frame: BoardFrame) -> None:
        """Present one ALREADY-COHERENT frame (e.g. a worker compose product).

        ``present_panel`` derives the identity boilerplate for windows that
        render their own picture; a console card's worker hands over a complete
        :class:`BoardFrame` whose identity the composer already minted.  Both
        entrances funnel into the SAME board and the SAME gesture binding, so
        the selector family stays owned once regardless of who built the frame.
        """

        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        if len(frame.panels) != 1 or frame.panels[0].panel_id != self._panel_id:
            raise ValueError(
                "SinglePanelHost requires its one configured panel"
            )
        self._board.present(frame)
        self._ensure_binding(frame.panels[0].display_payload)

    # ------------------------------------------------------------------ #
    # gesture plumbing
    # ------------------------------------------------------------------ #

    def _ensure_binding(self, payload) -> None:
        """Bind the gesture family matching the FIRST payload's kind, once.

        The binding is created READY (the panel was just presented, so its
        provenance is current) and the board-wide arm state is then set from
        the host's remembered switch -- so the surface comes up matching the
        switch (default OFF) instead of coming up live behind the operator.
        """

        if self._bound_kind is not None:
            return
        if isinstance(payload, PulsePanelPayload):
            self._board.bind_pulse_interaction(self._panel_id, self._on_intent)
            self._bound_kind = "pulse"
        elif isinstance(payload, CurvePanelPayload):
            self._board.bind_curve_interaction(self._panel_id, self._on_intent)
            self._bound_kind = "curve"
        elif isinstance(payload, HistogramPanelPayload):
            self._board.bind_histogram_interaction(
                self._panel_id, self._on_intent)
            self._bound_kind = "histogram"
        elif isinstance(payload, ImagePanelPayload):
            # The image family separates the operator's switch from readiness:
            # bind unarmed, then declare the just-presented provenance current
            # (the host's one source IS the frame the caller handed over).
            self._board.bind_rectangle_selector(
                self._panel_id,
                payload.viewport,
                self.rectangleSelected.emit,
                enabled=False,
                interaction_callback=self._on_intent,
            )
            self._board.set_interaction_readiness(
                image=True, curve=False, histogram=False, pulse=False)
            self._bound_kind = "image"
        if self._bound_kind is not None:
            self._board.set_selectors_enabled(self._selectors_on)

    def _echo_range_candidate(self, x_span) -> None:
        if self._bound_kind == "pulse":
            self._board.set_pulse_range_candidate(x_span)
        elif self._bound_kind == "curve":
            self._board.set_curve_range_candidate(x_span)
        elif self._bound_kind == "histogram":
            self._board.set_histogram_range_candidate(x_span)

    def _on_intent(self, intent) -> None:
        if isinstance(intent, (CurveRangeGesture, HistogramRangeGesture)):
            self._echo_range_candidate(intent.x_span)
            self.rangeSelected.emit(intent.x_span)
            return
        if isinstance(intent, (CurveViewportCommit, HistogramViewportCommit)):
            self.viewCommitted.emit(intent.viewport)
            return
        if isinstance(intent, HistogramThresholdCommit):
            self.thresholdsCommitted.emit(intent.thresholds)
            return
        if isinstance(intent, ImageViewportCommit):
            self.viewCommitted.emit(intent.viewport)
            return
        if isinstance(intent, ImageColorLimitsCommit):
            self.colorLimitsCommitted.emit(intent.color_limits)
