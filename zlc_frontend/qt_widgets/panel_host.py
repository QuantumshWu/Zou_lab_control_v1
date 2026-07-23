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
from ..selector import (
    CurveRangeGesture, CurveViewportCommit, HistogramRangeGesture,
    HistogramThresholdCommit, HistogramViewportCommit, ImageColorLimitsCommit,
    ImageViewportCommit, PanelInteractionOrigin)
from .board import QtRasterBoard


class SinglePanelHost(QtWidgets.QWidget):
    """A QtRasterBoard hosting exactly one interactive panel.

    ``present_panel(raster, payload, ...)`` wraps the payload in a coherent
    one-panel :class:`BoardFrame` (provenance, stamp, presentation identity all
    derived from the caller's content key and revisions) and presents it.  The
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

    @property
    def board(self) -> QtRasterBoard:
        """The underlying unified interaction owner (read-mostly access)."""

        return self._board

    @property
    def front_frame(self):
        """The board's painted front (or None) -- same fact as the board's."""

        return self._board.front_frame

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
                      pixel_ratio: float = 1.0) -> tuple[int, int]:
        """Present one rendered panel as a coherent frame; returns the LOGICAL size.

        Every identity fact is DERIVED from the payload itself.  Dataset panels
        mint a :class:`CoherenceStamp`; a pulse-document panel mints a
        :class:`DocumentPresentationStamp` and never fabricates dataset/run/
        join/schema identity.  In both families the panel revision is the
        viewport's display revision, so a window cannot hand the board a frame
        whose identity disagrees with its payload.
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
            provenance = payload.evaluated_input
            fingerprint = provenance.ref.schema_fingerprint
            content_revision = int(provenance.ref.revision.value)
            presentation = PanelPresentationIdentity(
                self._panel_id,
                self._group,
                content_revision,
                0,
                display_revision,
            )
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
        self._retire_incompatible_binding(payload)
        self._board.present(BoardFrame(
            f"{self._group}-board", 0, self._sequence, (panel,)))
        self._ensure_binding(payload)
        self.set_logical_size(logical)
        return logical

    def set_logical_size(self, logical_size: tuple[int, int]) -> None:
        """Pin this complete panel surface to its authored logical pixel size.

        The worker raster may be denser because of screen DPR, but neither Qt
        nor a containing card may invent another plot extent.  Pulse Preview
        reaches this owner through :meth:`present_panel`; worker-rendered
        TaskConsole panels call it when their ``PanelConfig.size`` changes.
        """

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
        width, height = logical_size
        self._board.setFixedSize(width, height)
        self.setFixedSize(width, height)

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
        payload = frame.panels[0].display_payload
        self._retire_incompatible_binding(payload)
        self._board.present(frame)
        self._ensure_binding(payload)

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

    def _retire_incompatible_binding(self, payload) -> None:
        """Unbind an old gesture family before the new front is validated."""

        if self._bound_kind != self._binding_kind(payload):
            self._unbind_current_interaction()

    def _ensure_binding(self, payload) -> None:
        """Bind the gesture family matching the currently presented payload.

        A family change retires only the old interaction binding; the stable
        host and newly presented raster remain in place.  The new binding is
        created READY (the panel was just presented, so its provenance is
        current) and inherits the host's remembered selector switch.
        """

        target_kind = self._binding_kind(payload)
        if self._bound_kind == target_kind:
            return
        if self._bound_kind is not None:
            raise RuntimeError("incompatible interaction binding was not retired")
        if target_kind == "pulse":
            self._board.bind_pulse_interaction(self._panel_id, self._on_intent)
            self._bound_kind = "pulse"
        elif target_kind == "curve":
            self._board.bind_curve_interaction(self._panel_id, self._on_intent)
            self._bound_kind = "curve"
        elif target_kind == "histogram":
            self._board.bind_histogram_interaction(
                self._panel_id, self._on_intent)
            self._bound_kind = "histogram"
        elif target_kind == "image":
            # The image family separates the operator's switch from readiness:
            # bind unarmed, then declare the just-presented provenance current
            # (the host's one source IS the frame the caller handed over).
            image_payload = (
                payload.background
                if isinstance(payload, SiteMapPanelPayload)
                else payload
            )
            self._board.bind_rectangle_selector(
                self._panel_id,
                image_payload.viewport,
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
