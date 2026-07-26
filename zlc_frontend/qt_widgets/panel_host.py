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

import math

from PyQt5 import QtCore, QtWidgets

from ..render import (
    BoardFrame, CoherenceStamp, CurvePanelPayload, HistogramPanelPayload,
    DocumentPresentationStamp, ImagePanelPayload, PanelFrame,
    PanelPresentationIdentity,
    PulsePanelPayload, RasterBuffer, SiteMapPanelPayload, SourceIdentity)
from ..figure_outputs import (
    FigureAreaCommit,
    FigureCrossCommit,
    HistogramValueRangeSelection,
    bind_area_data_commit,
    bind_cross_data_commit,
)
from ..selector import (
    CrossGesture, CurveRangeGesture, CurveViewportCommit, HistogramRangeGesture,
    HistogramThresholdCommit, HistogramViewportCommit, ImageColorLimitsCommit,
    ImageViewportCommit, PanelInteractionOrigin, RectangleGesture)
from .board import QtRasterBoard


class SinglePanelHost(QtWidgets.QWidget):
    """A QtRasterBoard hosting exactly one interactive panel.

    ``present_panel(raster, payload, ...)`` wraps the payload in a coherent
    one-panel :class:`BoardFrame` and presents it.  Dataset-backed payloads must
    arrive with the evaluator's exact :class:`CoherenceStamp`; this display
    adapter never invents run, epoch, or join identity.  The
    FIRST payload's type picks the gesture family -- pulse and curve speak the
    CURVE intent vocabulary, histogram its own -- and completed gestures come
    back as signals:

    * ``rangeSelected(object)``  -- the complete range gesture, including the
      exact painted origin.  The host already echoed its display-only span onto
      the board; the window decides what the selection MEANS.
    * ``viewCommitted(object)``  -- the complete wheel-zoom / pan commit.  The
      window first CAS-checks ``commit.origin``, re-renders at
      ``commit.viewport.x_limits``, and calls
      :meth:`present_panel` again with ``display_revision =
      commit.viewport.display_revision`` so the accepted front matches the
      intent.  No signal strips provenance down to a tuple or transform.
    """

    rangeSelected = QtCore.pyqtSignal(object)
    viewCommitted = QtCore.pyqtSignal(object)
    thresholdsCommitted = QtCore.pyqtSignal(object)
    crossSelected = QtCore.pyqtSignal(object)
    # image family only: a completed DISPLAY ONLY rectangle (RectangleGesture)
    # and a clim-rail commit's fixed colour limits.  The window echoes a
    # rectangle candidate through ``board.set_image_rectangle_candidate``.
    rectangleSelected = QtCore.pyqtSignal(object)
    colorLimitsCommitted = QtCore.pyqtSignal(object)

    def __init__(self, panel_id: str = "panel", *,
                 group: str | None = None,
                 empty_text: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel_id = str(panel_id)
        self._group = str(group or panel_id)
        self._board = QtRasterBoard(
            (self._panel_id,), columns=1, empty_text=empty_text)
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
        self._board.crossSelected.connect(self.crossSelected.emit)

    @property
    def board(self) -> QtRasterBoard:
        """The underlying unified interaction owner (read-mostly access)."""

        return self._board

    @property
    def front_frame(self):
        """The board's painted front (or None) -- same fact as the board's."""

        return self._board.front_frame

    @property
    def has_front(self) -> bool:
        return self._board.has_front

    @property
    def selector_fault(self) -> RuntimeError | None:
        """Return the current single-panel interaction fault, if any."""

        if self._bound_kind == "image":
            return self._board.image_selector_fault(self._panel_id)
        if self._bound_kind == "curve":
            return self._board.curve_selector_fault
        if self._bound_kind == "histogram":
            return self._board.histogram_selector_fault
        if self._bound_kind == "pulse":
            return self._board.pulse_selector_fault
        return None

    @property
    def selectors_enabled(self) -> bool:
        return self._board.selectors_enabled

    def set_interaction_ready(self, ready: bool) -> None:
        """Set readiness for this host's one bound gesture family."""

        if not isinstance(ready, bool):
            raise TypeError("interaction readiness must be bool")
        self._board.set_interaction_readiness(
            image=ready and self._bound_kind == "image",
            curve=ready and self._bound_kind == "curve",
            histogram=ready and self._bound_kind == "histogram",
            pulse=ready and self._bound_kind == "pulse",
        )

    def set_rectangle_candidate(self, normalized_bounds) -> None:
        """Echo one IMAGE rectangle on the exact single-panel binding."""

        if self._bound_kind != "image":
            raise RuntimeError("rectangle candidate requires an image binding")
        self._board.set_image_rectangle_candidate(normalized_bounds)

    def set_range_candidate(self, x_span) -> None:
        """Echo one completed numeric Area on this exact binding."""

        if self._bound_kind == "curve":
            self._board.set_curve_range_candidate(x_span)
            return
        if self._bound_kind == "histogram":
            self._board.set_histogram_range_candidate(x_span)
            return
        raise RuntimeError("range candidate requires a numeric binding")

    def set_selectors_enabled(self, on: bool) -> None:
        """The Selectors switch, same semantics as every console card: remember
        the desired state and gate the CURRENT binding now; a binding created
        later (first present) inherits it."""

        self._selectors_on = bool(on)
        if self._bound_kind is not None:
            self._board.set_selectors_enabled(self._selectors_on)

    def clear(self) -> None:
        """Forget the current front/binding while keeping this host alive.

        The host is a stable piece of window chrome.  Callers that change the
        presented signal or payload family need to retire the old interaction
        binding before the next immutable frame arrives; they do not need to
        delete and recreate this QWidget subtree.
        """

        self._unbind_current_interaction()
        self._board.clear()

    def _unbind_current_interaction(self) -> None:
        """Retire only the current gesture family, preserving the raster front."""

        if self._bound_kind == "pulse":
            self._board.unbind_pulse_interaction(self._panel_id)
        elif self._bound_kind == "curve":
            self._board.unbind_curve_interaction(self._panel_id)
        elif self._bound_kind == "histogram":
            self._board.unbind_histogram_interaction(self._panel_id)
        elif self._bound_kind == "image":
            self._board.unbind_rectangle_selector(self._panel_id)
        self._bound_kind = None

    def unbind_interaction(self) -> None:
        """Retire this surface's gesture binding without clearing its pixels."""

        self._unbind_current_interaction()

    def visible_interaction_origin(self) -> PanelInteractionOrigin | None:
        """Return the exact painted origin for this host's bound family.

        Family dispatch belongs here because the host owns the binding kind.
        Callers therefore perform one origin CAS without reaching into four
        separate board APIs or guessing the payload family from a transform.
        """

        if self._bound_kind == "image":
            return self._board.visible_image_origin(self._panel_id)
        if self._bound_kind == "curve":
            return self._board.visible_curve_origin(self._panel_id)
        if self._bound_kind == "histogram":
            return self._board.visible_histogram_origin(self._panel_id)
        if self._bound_kind == "pulse":
            return self._board.visible_pulse_origin(self._panel_id)
        return None

    def selection_for_rectangle_gesture(self, gesture):
        """Resolve one IMAGE rectangle through this host's exact held front.

        Gesture ownership and gesture-to-data conversion are one boundary.
        Callers that publish Figure outputs must not reach through the host to
        its private board, because doing so would make single and faceted
        surfaces expose different selector-output routes.
        """

        if self._bound_kind != "image":
            raise RuntimeError(
                "rectangle selection requires an image interaction binding"
            )
        return self._board.selection_for_rectangle_gesture(gesture)

    def selection_for_curve_range_gesture(self, gesture):
        """Resolve one CURVE range through this host's exact held front."""

        if self._bound_kind != "curve":
            raise RuntimeError(
                "curve selection requires a curve interaction binding"
            )
        return self._board.selection_for_curve_range_gesture(gesture)

    def area_commit_for_range_gesture(
        self,
        gesture: CurveRangeGesture | HistogramRangeGesture,
        *,
        figure,
    ) -> FigureAreaCommit | None:
        """Resolve a completed numeric Area against this exact painted front."""

        if not isinstance(gesture, (CurveRangeGesture, HistogramRangeGesture)):
            raise TypeError("numeric Area requires a curve or histogram gesture")
        if gesture.x_span is None:
            return None
        if self.visible_interaction_origin() != gesture.origin:
            raise RuntimeError("numeric Area gesture belongs to a stale front")
        source_identity = gesture.origin.source_identity
        if not isinstance(source_identity, SourceIdentity):
            raise TypeError("numeric Area requires a dataset source")
        selection = (
            HistogramValueRangeSelection(*gesture.x_span)
            if isinstance(gesture, HistogramRangeGesture)
            else self.selection_for_curve_range_gesture(gesture)
        )
        return bind_area_data_commit(source_identity, selection, figure)

    def area_commit_for_rectangle_gesture(
        self,
        gesture: RectangleGesture,
        *,
        figure,
    ) -> FigureAreaCommit | None:
        """Resolve a completed image Area against this exact painted front."""

        if not isinstance(gesture, RectangleGesture):
            raise TypeError("image Area requires RectangleGesture")
        if gesture.normalized_bounds is None:
            return None
        selection = self.selection_for_rectangle_gesture(gesture)
        return bind_area_data_commit(gesture.source_identity, selection, figure)

    def cross_commit_for_gesture(
        self,
        gesture: CrossGesture,
        *,
        figure,
    ) -> FigureCrossCommit | None:
        """Resolve one locked Cross to the exact value the Figure displayed."""

        if not isinstance(gesture, CrossGesture):
            raise TypeError("Cross output requires CrossGesture")
        if gesture.point is None:
            return None
        if self.visible_interaction_origin() != gesture.origin:
            raise RuntimeError("Cross gesture belongs to a stale front")
        source_identity = gesture.origin.source_identity
        if not isinstance(source_identity, SourceIdentity):
            raise TypeError("Cross output requires a dataset source")
        front = self.front_frame
        if front is None or len(front.panels) != 1:
            raise RuntimeError("Cross output requires one exact painted panel")
        return bind_cross_data_commit(
            source_identity,
            gesture.point,
            figure,
            front.panels[0].display_payload,
        )

    def discard_pending_interaction(self, origin: PanelInteractionOrigin) -> bool:
        """Release only the exact failed display commit for this host."""

        if not isinstance(origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        if self._bound_kind == "image":
            return self._board.discard_pending_image_interaction(origin)
        if self._bound_kind == "curve":
            return self._board.discard_pending_curve_interaction(origin)
        if self._bound_kind == "histogram":
            return self._board.discard_pending_histogram_interaction(origin)
        if self._bound_kind == "pulse":
            return self._board.discard_pending_pulse_interaction(origin)
        return False

    def present_panel(self, raster: RasterBuffer, payload, *,
                      coherence_stamp: CoherenceStamp | None = None,
                      pixel_ratio: float = 1.0) -> tuple[int, int]:
        """Present one rendered panel as a coherent frame; returns the LOGICAL size.

        Dataset panels require the evaluator's exact ``coherence_stamp``;
        presentation code cannot derive a typed join or runtime lineage from a
        display payload.  A pulse-document panel derives only its local
        :class:`DocumentPresentationStamp` and never fabricates dataset/run/
        join/schema identity.  In both families the panel revision is checked
        against the viewport's display revision by :class:`PanelFrame`.
        ``pixel_ratio`` is the screen ratio the raster was rendered at: the
        widget pins to the LOGICAL size so the whole-cell blit lands 1:1 on
        device pixels.
        """

        if isinstance(payload, ImagePanelPayload):
            display_revision = int(payload.viewport.viewport_revision)
        elif isinstance(
            payload, (CurvePanelPayload, HistogramPanelPayload, PulsePanelPayload)
        ):
            display_revision = int(payload.viewport.display_revision)
        else:
            raise TypeError(
                "SinglePanelHost requires an interactive image, curve, "
                "histogram, or pulse payload"
            )
        ratio = float(pixel_ratio)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("pixel_ratio must be positive and finite")
        logical = (
            int(round(raster.width / ratio)),
            int(round(raster.height / ratio)),
        )
        if any(value <= 0 for value in logical):
            raise ValueError("pixel_ratio resolves the raster to an empty widget")
        if isinstance(payload, PulsePanelPayload):
            if coherence_stamp is not None:
                raise TypeError(
                    "pulse document panels do not accept a dataset coherence stamp"
                )
            source = payload.document_input
            presentation = PanelPresentationIdentity(
                self._panel_id,
                source.document_id,
                source.document_revision,
                0,
                display_revision,
            )
            stamp = DocumentPresentationStamp(source, (presentation,))
        else:
            if not isinstance(coherence_stamp, CoherenceStamp):
                raise TypeError(
                    "dataset panels require the evaluator's exact coherence_stamp"
                )
            provenance = payload.evaluated_input
            fingerprint = provenance.ref.schema_fingerprint
            stamp = coherence_stamp
            source = SourceIdentity(
                provenance.dataset_id,
                provenance.ref.block_id,
                provenance.ref.stream_generation,
                fingerprint,
            )
        panel = PanelFrame(
            self._panel_id, self._group, source, stamp, raster, payload)
        self._sequence += 1
        self.present_frame(
            BoardFrame(f"{self._group}-board", 0, self._sequence, (panel,)),
            logical_size=logical,
        )
        return logical

    def set_logical_size(self, logical_size: tuple[int, int]) -> None:
        """Pin this complete panel surface to its authored logical pixel size.

        The worker raster may be denser because of screen DPR, but neither Qt
        nor a containing card may invent another plot extent.  Pulse Preview
        reaches this owner through :meth:`present_panel`; worker-rendered
        TaskConsole panels call it when their ``PanelConfig.size`` changes.
        """

        width, height = self._validated_logical_size(logical_size)
        self._board.setFixedSize(width, height)
        self.setFixedSize(width, height)

    @staticmethod
    def _validated_logical_size(
        logical_size: tuple[int, int],
    ) -> tuple[int, int]:
        if (
            not isinstance(logical_size, tuple)
            or len(logical_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in logical_size
            )
        ):
            raise ValueError(
                "logical_size must be a pair of positive integer pixels"
            )
        return logical_size

    def present_frame(
        self,
        frame: BoardFrame,
        *,
        logical_size: tuple[int, int] | None = None,
    ) -> None:
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
        payload = frame.panels[0].display_payload
        target_kind = self._binding_kind(payload)
        if logical_size is not None:
            logical_size = self._validated_logical_size(logical_size)
        geometry_changes = logical_size is not None and (
            self.width(), self.height()
        ) != logical_size
        if geometry_changes:
            self.setUpdatesEnabled(False)
        try:
            if self._bound_kind == target_kind:
                self._board.present(frame)
            else:
                self._board.present_with_single_panel_interaction(
                    frame,
                    panel_id=self._panel_id,
                    kind=target_kind,
                    interaction_callback=self._on_intent,
                    rectangle_callback=self.rectangleSelected.emit,
                    selectors_enabled=self._selectors_on,
                )
                self._bound_kind = target_kind
            if logical_size is not None:
                self.set_logical_size(logical_size)
        finally:
            if geometry_changes:
                self.setUpdatesEnabled(True)
                self.update()

    # ------------------------------------------------------------------ #
    # gesture plumbing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _binding_kind(payload) -> str | None:
        if isinstance(payload, PulsePanelPayload):
            return "pulse"
        if isinstance(payload, CurvePanelPayload):
            return "curve"
        if isinstance(payload, HistogramPanelPayload):
            return "histogram"
        if isinstance(payload, (ImagePanelPayload, SiteMapPanelPayload)):
            return "image"
        return None

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
            self.rangeSelected.emit(intent)
            return
        if isinstance(intent, (CurveViewportCommit, HistogramViewportCommit)):
            self.viewCommitted.emit(intent)
            return
        if isinstance(intent, HistogramThresholdCommit):
            self.thresholdsCommitted.emit(intent)
            return
        if isinstance(intent, ImageViewportCommit):
            self.viewCommitted.emit(intent)
            return
        if isinstance(intent, ImageColorLimitsCommit):
            self.colorLimitsCommitted.emit(intent)
