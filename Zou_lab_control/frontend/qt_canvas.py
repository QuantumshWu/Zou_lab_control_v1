"""Embedding matplotlib figures in Qt windows: the ONE display-scale wrapper.

Figures keep the single frontend font/geometry system (style.DEFAULT_STYLE,
dpi=300) -- font sizes are NEVER forked per host.  How large a figure APPEARS in
a Qt window is a display concern, handled here.

The frontend's figures have SPEC-OWNED geometry: ``create_axes_fixed`` pins the
axes at fixed inch offsets, so the figure's size-in-inches is part of the design
and must NEVER change.  The three invariants this canvas maintains, at every
screen scale (Windows display scaling, QT_SCALE_FACTOR, screen moves):

1. ``figure.get_size_inches()`` stays at its construction value forever.
2. ``figure.dpi = design_dpi x REAL screen ratio`` -- genuine high-DPI screens
   get the standard "retina" supersampling, nothing else touches the dpi.
3. The widget's LOGICAL size is ``design_px x display_scale`` -- our
   ``display_scale`` is a pure display zoom on top, with exact interaction
   coordinates (the backend converts mouse positions through the same ratio).

The stock Qt backend instead re-derives the figure size FROM the widget size on
every resize/ratio sync, which warps the fixed-inches axes layout the moment
the ratio changes (panels collapsed into a corner on scaled Windows screens) --
so ``resizeEvent`` here deliberately never touches the figure.

Every Qt host that embeds a figure must use :class:`EmbeddedFigureCanvas`
(``display_scale=1.0`` shows 1:1); it also stops wheel events from leaking into
a surrounding QScrollArea, so in-plot zoom never scrolls the page.
"""

from __future__ import annotations

from .style import DESIGN_DPI

try:
    from PyQt5 import QtCore, QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvasQTAgg
except Exception:  # pragma: no cover - depends on the local matplotlib install
    _FigureCanvasQTAgg = None


if _FigureCanvasQTAgg is None:  # pragma: no cover - matplotlib-qt missing
    EmbeddedFigureCanvas = None
else:

    class EmbeddedFigureCanvas(_FigureCanvasQTAgg):
        """Matplotlib Qt canvas with a display scale and wheel isolation."""

        def __init__(self, figure, *, display_scale: float = 1.0, isolate_wheel: bool = True,
                     render_scale: float = 1.0):
            # both must exist BEFORE super().__init__: the base class reads
            # devicePixelRatioF() (overridden below) during construction.
            self._zlc_ratio = 1.0 / max(0.1, float(display_scale))
            # render_scale < 1 renders the Agg buffer at a LOWER dpi (cheaper text raster)
            # and Qt scales it up to the SAME widget size -- it is factored into BOTH
            # figure.dpi and the device-pixel-ratio so they cancel and the display size /
            # fixed-inches axes layout are byte-identical (only slightly softer).
            self._zlc_render_scale = max(0.05, float(render_scale))
            self._zlc_inches = tuple(float(v) for v in figure.get_size_inches())
            # isolate_wheel=True: in-plot wheel zoom never leaks to a surrounding
            # scroll area (interactive plots).  False: the wheel PROPAGATES, so a
            # display-only panel (Monitor card, NO selectors) lets the dashboard
            # board scroll under the cursor instead of swallowing the wheel.
            self._zlc_isolate_wheel = bool(isolate_wheel)
            # Pinned (the Edit snapshot): once ``pin_size()`` is called, every resync re-applies
            # ``setFixedSize`` (min==max) instead of ``setMinimumSize``, so a deferred resync can never
            # raise the floor past the fixed ceiling and balloon the figure (#4 edit-resize).  Default
            # unpinned: a live card is held by its OWN ``setFixedSize`` wrapper, so it only needs a floor.
            self._zlc_pinned = False
            # The figure's DESIGN dpi is the ONE canonical ``DESIGN_DPI`` -- the same dpi the layout
            # geometry was authored against (``create_axes_fixed`` sizes the figure as
            # ``pixels / design_dpi(fig)``, with ``design_dpi`` == DESIGN_DPI), so the displayed size
            # ``inches x _zlc_design_dpi x display_scale`` is correct iff the two dpis MATCH.  We must
            # NOT read it from ``figure.dpi`` / ``figure._original_dpi``: under ``%matplotlib widget``
            # (ipympl) on a hi-DPI screen the Qt/ipympl backend BOOSTS ``figure.dpi`` by the screen
            # ratio, and on the FIRST figure of a session ``_original_dpi`` is still unset (None) while
            # ``figure.dpi`` is already boosted (e.g. 750) -- so the old ``_original_dpi or figure.dpi``
            # captured 750 while the axes were laid out at 300, rendering the first task-takeover panel
            # ~2.5x too big; the SECOND task (``_original_dpi`` now 300) was correct ("first run wrong,
            # second right").  Pinning to the constant makes it size-correct on the FIRST run, every run.
            self._zlc_design_dpi = float(DESIGN_DPI)
            super().__init__(figure)
            # undo any backend dpi inflation so design_dpi(fig) (and _zlc_sync) see the true design.
            figure._original_dpi = self._zlc_design_dpi
            # the backend syncs only in showEvent / on screen signals (never
            # offscreen) -- establish the invariants NOW
            self._zlc_sync()
            # ...and AGAIN on the next event-loop tick.  The construction sync above runs before this
            # canvas has been inserted into its parent layout, so the screen ratio it reads + the
            # sizeHint/minimumSize it publishes can be the pre-layout values; the FIRST task-takeover
            # panel of a just-opened console is built straight onto the board, so without this its card
            # keeps the stale size until some later relayout (the SECOND task run) -- the "first wrong,
            # second right" symptom.  singleShot(0) re-syncs once the widget is in its real layout.
            QtCore.QTimer.singleShot(0, self._zlc_resync)

        def _zlc_resync(self) -> None:
            # Re-establish the size invariants once the widget is genuinely on screen / laid out, and
            # tell the parent layout the (now-correct) sizeHint/minimumSize changed so the CARD around
            # us is re-measured -- not just the canvas.  Guarded: the deferred call may fire after the
            # C++ widget was torn down (panel closed) -> skip silently.
            try:
                self._zlc_sync()                       # 1) set the figure to the correct design dpi
            except RuntimeError:
                return
            # 2) Re-PUBLISH the size floor from the now-settled sizeHint.  The host (PanelCard) pins
            # ``setMinimumSize(sizeHint())`` at BUILD time; if that build ran before the layout settled
            # the floor is stale-too-big and would pin the canvas at the wrong size -- the exact "first
            # wrong, second right" trap (a resize cannot go below a stale minimum).  Re-assert it from
            # the SAME source (sizeHint) to clear any stale floor...  When PINNED (Edit snapshot) min and
            # max move TOGETHER to the settled sizeHint -- so resync corrects the size but can never let
            # the floor exceed the ceiling and balloon the figure (#4 "改参数图变大小").
            if self._zlc_pinned:
                self.setFixedSize(self.sizeHint())
            else:
                self.setMinimumSize(self.sizeHint())
            self._zlc_sync()                           # 3) ...then resize again, now free of that floor
            self.updateGeometry()                      # 4) let the parent CARD re-measure us
            self.draw_idle()

        def pin_size(self) -> None:
            """Lock this canvas to a FIXED size (min==max) == its design ``sizeHint``, and KEEP it
            locked across every later resync / showEvent.  The Edit snapshot uses this so a param edit
            (which rebuilds the snapshot) can never let a deferred ``_zlc_resync`` raise the minimum
            past the fixed maximum and balloon the figure -- min and max always move together to the
            settled design size.  Idempotent; replaces a bare ``setFixedSize`` that resync would undo."""
            self._zlc_pinned = True
            self.setFixedSize(self.sizeHint())

        def showEvent(self, event):  # noqa: N802 - Qt naming
            # Re-establish the size invariants + redraw the FIRST time the canvas actually becomes
            # visible.  A panel built on a not-yet-shown board (the FIRST task-takeover Monitor panel
            # of a just-opened console) can take its construction-time sync before the window settled
            # on its real screen; showEvent fires once it is genuinely on screen, so re-syncing here
            # makes that first panel render at the correct size without needing a second task run.
            # Idempotent: when the construction sync was already correct this only repaints.
            super().showEvent(event)
            self._zlc_resync()

        # ------------------------------------------------------------- ratio math
        def devicePixelRatioF(self):  # noqa: N802 - Qt naming
            # The backend derives the render-buffer size, sizeHint, mouse-event
            # coordinates and the painter's image scaling from this one ratio.  The
            # render_scale rides in here too: figure.dpi carries the SAME factor, so the
            # buffer shrinks (cheaper raster) while widget size = buffer / this is unchanged.
            return (super().devicePixelRatioF() or 1.0) * self._zlc_ratio * self._zlc_render_scale

        def _set_device_pixel_ratio(self, ratio):
            # Reroute every stock sync (showEvent, screen/dpi-change signals)
            # through our math: the stock implementation magnifies figure.dpi by
            # the FULL ratio ("retina": same on-screen size) and then re-derives
            # the figure from the widget -- both break the spec-owned geometry.
            if getattr(self, "_device_pixel_ratio", None) == ratio:
                return False
            self._zlc_sync()
            return True

        def _zlc_sync(self) -> None:
            real = super().devicePixelRatioF() or 1.0
            rs = self._zlc_render_scale
            figure = self.figure
            # invariant 1: the design inches NEVER change (fixed-inches axes)
            figure.set_size_inches(*self._zlc_inches, forward=False)
            # invariant 2: retina supersampling by the REAL screen ratio, times the live
            # render_scale (rs<1 -> smaller/cheaper buffer; rs cancels in the widget size).
            figure._set_dpi(self._zlc_design_dpi * real * rs, forward=False)
            self._device_pixel_ratio = real * self._zlc_ratio * rs
            # invariant 3: logical widget size = design px x display_scale
            width_px, height_px = map(float, figure.bbox.max)
            self.resize(round(width_px / self._device_pixel_ratio),
                        round(height_px / self._device_pixel_ratio))

        def resizeEvent(self, event):  # noqa: N802 - Qt naming
            # spec-owned figure: NEVER re-derive the figure geometry from the
            # widget size (the stock handler does, and any transient mismatch
            # warps the fixed-inches axes layout).  Accept the size, repaint.
            QtWidgets.QWidget.resizeEvent(self, event)
            self.draw_idle()

        # ------------------------------------------------------------- behaviour
        def wheelEvent(self, event):  # noqa: N802 - Qt naming
            if not self._zlc_isolate_wheel:
                # display-only panel: let the wheel bubble up to the scroll area
                # so the dashboard board scrolls under the cursor.
                event.ignore()
                return
            # interactive plot: in-plot wheel zoom must never double as a page scroll
            super().wheelEvent(event)
            event.accept()


def panel_canvas(figure, *, isolate_wheel: bool = True):
    """The canvas for a dashboard panel figure: the panel display scale is a
    frontend design constant, not a host knob.  ``isolate_wheel=False`` lets a
    display-only (Monitor) panel's wheel scroll the board instead of being
    swallowed; interactive (Edit) panels keep the default isolation."""

    if EmbeddedFigureCanvas is None:  # pragma: no cover - matplotlib-qt missing
        raise RuntimeError("matplotlib Qt canvas is not available")
    from .live import PANEL_DISPLAY_SCALE
    from .style import LIVE_RENDER_SCALE
    # Live panels render at LIVE_RENDER_SCALE x design dpi (150 dpi) for speed; the display
    # size is unchanged (Qt upscales the smaller buffer).  Saved figures use savefig.dpi.
    return EmbeddedFigureCanvas(figure, display_scale=PANEL_DISPLAY_SCALE,
                                isolate_wheel=isolate_wheel, render_scale=LIVE_RENDER_SCALE)


__all__ = ["EmbeddedFigureCanvas", "panel_canvas"]
