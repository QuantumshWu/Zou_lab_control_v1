"""Embedding matplotlib figures in Qt windows: the ONE display-scale wrapper.

Figures keep the single frontend font/geometry system (render_style.DEFAULT_STYLE,
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

import logging
import threading

from zlc_frontend.render_style import DESIGN_DPI
from zlc_frontend.render import RasterBuffer
import zlc_frontend.qt_widgets as _qt_widgets
RENDER_THREAD_NAME = _qt_widgets.render_loop.RENDER_THREAD_NAME

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigureCanvasAgg
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvasQTAgg
except Exception:  # pragma: no cover - depends on the local matplotlib install
    _FigureCanvasQTAgg = None


if _FigureCanvasQTAgg is None:  # pragma: no cover - matplotlib-qt missing
    EmbeddedFigureCanvas = None
else:

    class EmbeddedFigureCanvas(_FigureCanvasQTAgg):
        """Matplotlib Qt canvas with a display scale and wheel isolation."""

        #: Identity marker for ``live._is_embedded_canvas`` -- the ONE predicate the two-phase
        #: compose()/present()/blit render paths key off.  A duck-checked class attribute (not an
        #: isinstance) because live.py deliberately never imports Qt; inherited by subclasses, so
        #: a subclassed canvas keeps the blit path.
        _zlc_embedded = True

        def __init__(self, figure, *, display_scale: float = 1.0, isolate_wheel: bool = True):
            # Must exist BEFORE super().__init__: the base class reads devicePixelRatioF()
            # (overridden below) during construction.  ``display_scale`` is the ONE display
            # knob -- the on-screen zoom on top of the design size.  The Agg buffer is ALWAYS
            # rendered at the matching resolution (figure.dpi = design_dpi x real screen ratio
            # x display_scale), so the blit onto the widget is 1:1 crisp: nothing is ever
            # rendered small and stretched up.  (There is deliberately NO separate render-scale:
            # a coarser buffer would only ever be a permanent softness, so display == render.)
            self._zlc_display_scale = max(0.05, float(display_scale))
            self._zlc_inches = tuple(float(v) for v in figure.get_size_inches())
            # isolate_wheel=True: in-plot wheel zoom never leaks to a surrounding
            # scroll area (interactive plots).  False: the wheel PROPAGATES, so a
            # display-only panel (Monitor card, NO selectors) lets the dashboard
            # board scroll under the cursor instead of swallowing the wheel.
            self._zlc_isolate_wheel = bool(isolate_wheel)
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
            # Fix the widget to its DPR-INVARIANT design size ONCE (min==max).  Neither a deferred
            # resync, a screen-ratio change, nor the host can re-pin it from the racy DPR-derived
            # sizeHint and balloon/shrink the figure (#5 "图偶尔变大") -- DPR only scales the render
            # BUFFER, never this logical size.  This is the SINGLE owner of the canvas size; the host
            # PanelCard no longer pins setMinimumSize(sizeHint()) on top (the two-pin race is gone).
            self.setFixedSize(self._zlc_design_size())
            # Render the Agg buffer NOW (synchronously): the canvas is never blank in the window between
            # insertion and a deferred first paint (draw_idle leaves the buffer unrendered -- #5 "图偶尔
            # 消失").  A freshly-built canvas swapped into the holder already carries a painted frame.
            self._zlc_draw_if_needed()
            # ...and AGAIN on the next event-loop tick.  The construction sync above runs before this
            # canvas has been inserted into its parent layout, so the screen ratio it reads can be the
            # pre-layout value; the FIRST task-takeover panel of a just-opened console is built straight
            # onto the board, so without this its figure dpi (render buffer) stays pre-screen until some
            # later relayout.  singleShot(0) re-syncs the BUFFER (not the size) once genuinely laid out.
            QtCore.QTimer.singleShot(0, self._zlc_resync)

        def _zlc_design_size(self) -> "QtCore.QSize":
            """The widget's LOGICAL size = design_inches x design_dpi x display_scale -- a CONSTANT that
            does NOT read the screen device-pixel-ratio.  The stock ``sizeHint`` is
            ``figure.bbox / device_pixel_ratio``; the DPR cancels in that ratio ONLY when figure.dpi and
            device_pixel_ratio are read at the SAME instant -- mid screen-ratio-change (a rebuild burst,
            a monitor move) they desync and sizeHint balloons.  This product is invariant, so pinning the
            canvas min==max to it is race-free.  (DPR still scales the render BUFFER via figure.dpi.)"""
            w_in, h_in = self._zlc_inches
            scale = self._zlc_display_scale
            return QtCore.QSize(max(1, round(w_in * self._zlc_design_dpi * scale)),
                                max(1, round(h_in * self._zlc_design_dpi * scale)))

        def _zlc_resync(self) -> None:
            # Re-establish the figure dpi (render buffer) once the widget is genuinely on screen -- the
            # construction sync ran before the widget knew its real screen ratio.  The widget SIZE is the
            # DPR-invariant design constant, fixed min==max at construction, so a resync corrects the
            # buffer for the real screen but can NEVER re-pin a wrong size and balloon/shrink the figure
            # ("first wrong, second right" is gone).  Guarded: the deferred call may fire after the C++
            # widget was torn down (panel closed) -> skip silently.
            self._zlc_wait_render()                    # sync mutates figure dpi/size: own the figure first
            try:
                self._zlc_sync()                       # figure dpi -> the REAL screen ratio (crisp buffer)
            except RuntimeError:
                return
            self.setFixedSize(self._zlc_design_size())  # idempotent (a monitor move only shifts display_scale)
            self.updateGeometry()                       # let the parent CARD re-measure us
            self._zlc_draw_if_needed()                  # re-render ONLY if the dpi/content actually changed

        def pin_size(self) -> None:
            """Lock the canvas to its design size.  Now equivalent to the default -- the canvas is already
            setFixedSize to the DPR-invariant :meth:`_zlc_design_size` at construction and re-asserts it
            on every resync -- so this is idempotent.  Kept for the Edit-snapshot caller's intent."""
            self.setFixedSize(self._zlc_design_size())

        def refresh_design_size(self) -> None:
            """Re-read the FIGURE's current size and re-pin the widget to the matching design size --
            for the one legitimate figure-resize path: a grid enlarging a cell into a standalone 2x2
            panel ON ITS OWN figure (and back).  Construction cached ``_zlc_inches`` once and every
            resync forces the figure BACK to it, so without this refresh the enlarged view's 2x2
            buffer stretched over the old grid-sized widget (the "double-click zoom is as big as the
            whole grid" bug).  The plot layer calls it through duck-typing (a non-Qt canvas simply
            lacks the method and follows the figure natively)."""
            self._zlc_inches = tuple(float(v) for v in self.figure.get_size_inches())
            self._zlc_resync()
            self.updateGeometry()

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
            # The backend derives the render-buffer size, sizeHint, mouse-event coordinates and
            # the painter's image scaling from this one ratio.  figure.dpi already carries
            # display_scale (see _zlc_sync), so the buffer is exactly the widget's on-screen
            # device pixels and this is just the REAL screen ratio -- a 1:1 crisp blit.
            return super().devicePixelRatioF() or 1.0

        def _set_device_pixel_ratio(self, ratio):
            # Reroute every stock sync (showEvent, screen/dpi-change signals)
            # through our math: the stock implementation magnifies figure.dpi by
            # the FULL ratio ("retina": same on-screen size) and then re-derives
            # the figure from the widget -- both break the spec-owned geometry.
            if getattr(self, "_device_pixel_ratio", None) == ratio:
                return False
            self._zlc_wait_render()   # a mid-stream monitor move mutates figure dpi: own the figure first
            self._zlc_sync()
            return True

        def draw(self):  # noqa: N802 - overrides FigureCanvasQTAgg.draw
            # Every real Agg render funnels through here: record the dpi it rendered at + that the buffer
            # is now valid, so :meth:`_zlc_draw_if_needed` can SKIP the redundant lifecycle re-renders.
            # Ownership barrier FIRST: this is the one funnel every GUI-thread render takes (draw_idle
            # callbacks, tab-switch showEvent redraws, the paintEvent fallback), so it must never
            # rasterise into an Agg buffer the render worker is composing.  Free when idle / unhooked.
            self._zlc_wait_render()
            super().draw()
            self._zlc_drawn_dpi = self.figure.dpi
            self._zlc_painted_once = True
            # A canvas in FRONT-buffer mode (live console panel, rendered by the render thread) keeps
            # its front image in sync with any DIRECT GUI-thread render too (a CrossSelector click's
            # draw_idle, a Setting re-render), or the paintEvent would blit a stale frame over it.
            if getattr(self, "_zlc_front", None) is not None:
                self._zlc_snapshot_front()

        def _zlc_draw_agg_only(self):
            """Rasterise the figure into the Agg buffer WITHOUT the stock QtAgg ``draw()``'s
            ``self.update()`` (a Qt widget call).  This is the compose-phase render: the two-phase
            board flips the screen only in present(), and it is the ONLY draw the render thread may
            use -- Qt widget state must never be touched off the GUI thread."""
            _FigureCanvasAgg.draw(self)
            self._zlc_drawn_dpi = self.figure.dpi
            self._zlc_painted_once = True

        # ------------------------------------------------------------- front buffer
        # FRONT-buffer protocol (render-thread decoupling): while the render thread owns a live
        # panel's figure, the GUI paintEvent must not read (or lazily re-render) the Agg buffer the
        # worker is writing.  present() therefore deep-copies the finished buffer into ``_zlc_front``
        # and paintEvent blits THAT.  Interaction drags (selector blits bypass draw()/present()) drop
        # the front first so the live Agg buffer shows again -- the figure is GUI-owned for the whole
        # drag (see selectors.begin_figure_interaction).
        #
        # The front is a ``RasterBuffer``, which is the ONE value the target architecture lets cross a
        # worker/GUI boundary; ``zlc_frontend/render.py`` names the alternative this used to be --
        # "No live Figure, Artist, mutable or aliased ndarray view, or QImage storage crosses this
        # module's boundary".  A stored QImage is not merely off-vocabulary: it is a Qt object owned by
        # whichever thread built it, so it could never be the hand-off once the renderer stops being
        # this very widget.  Owned bytes can.  paintEvent images them for the duration of one blit.
        def _zlc_snapshot_front(self) -> None:
            renderer = getattr(self, "renderer", None)
            if renderer is None:
                return
            self._zlc_front = RasterBuffer.from_agg_rgba(
                int(renderer.width), int(renderer.height), self.buffer_rgba())

        @staticmethod
        def _zlc_front_image(front) -> "QtGui.QImage":
            """Image one owned raster for exactly one blit.

            The QImage does not copy: it VIEWS ``front.pixels``, which is immutable and outlives the
            paint because the caller holds the RasterBuffer.  Passing the stride explicitly keeps this
            honest for any future non-tight raster instead of silently mis-reading rows.
            """

            return QtGui.QImage(front.pixels, front.width, front.height,
                                front.stride_bytes, QtGui.QImage.Format_RGBA8888)

        def _zlc_drop_front(self) -> None:
            self._zlc_front = None

        def _zlc_present(self) -> None:
            """Publish the composed Agg buffer: snapshot it as the front image and schedule the
            Qt paint.  GUI thread only (present phase).  A figure the operator is mid-drag on is
            GUI-owned (``begin_figure_interaction`` dropped the front so paint reads the live Agg
            blits): restoring a front image now would freeze the selector feedback for the whole
            drag, so the present is skipped -- the panel catches up on release (owed beat)."""
            if getattr(self.figure, "_zlc_interacting", False):
                return
            self._zlc_snapshot_front()
            self.update()

        def _zlc_draw_if_needed(self) -> None:
            # Render the Agg buffer ONLY when it is genuinely out of date: never rendered, the figure
            # CONTENT changed (matplotlib's figure.stale), or the buffer dpi changed (a real screen-ratio
            # change).  The construct -> singleShot resync -> showEvent resync -> paint lifecycle otherwise
            # each did a full SYNCHRONOUS render of the SAME content at the SAME dpi -- ~6 pixel-identical
            # re-renders of a 36-cell grid (~125 ms each at a 3x screen).  Renders synchronously when it
            # IS due, so paintEvent always blits a valid buffer (no deferred-draw blank frame).
            if (getattr(self, "renderer", None) is None or self.figure.stale
                    or self.figure.dpi != getattr(self, "_zlc_drawn_dpi", None)):
                self.draw()

        def _zlc_sync(self) -> None:
            real = super().devicePixelRatioF() or 1.0
            ds = self._zlc_display_scale
            figure = self.figure
            # IDEMPOTENT: only mutate the figure when a value actually changes, so a repeated resync at
            # the SAME screen ratio (construct + singleShot + showEvent all fire post-layout) leaves
            # figure.stale untouched and _zlc_draw_if_needed can skip the pixel-identical re-render.
            # invariant 1: the design inches NEVER change (fixed-inches axes)
            if tuple(figure.get_size_inches()) != self._zlc_inches:
                figure.set_size_inches(*self._zlc_inches, forward=False)
            # invariant 2: the buffer dpi = design dpi x REAL screen ratio (retina supersampling)
            # x display_scale, so the Agg buffer is EXACTLY the widget's on-screen device pixels
            # (a 1:1 crisp blit -- never rendered small and stretched up).
            new_dpi = self._zlc_design_dpi * real * ds
            if figure.dpi != new_dpi:
                figure._set_dpi(new_dpi, forward=False)
            self._device_pixel_ratio = real
            # invariant 3: logical widget size = design px x display_scale -- the DPR-FREE constant
            # (NOT figure.bbox / device_pixel_ratio, which desyncs when the screen ratio changes
            # mid-sync and balloons the size).  After construction the canvas is setFixedSize to this
            # same value, so this resize() is the pre-fix construction sizing and a no-op thereafter.
            self.resize(self._zlc_design_size())

        def paintEvent(self, event):  # noqa: N802 - Qt naming
            # EXPLICIT stretch-blit of the rendered Agg buffer over the ENTIRE fixed widget rect.
            # The stock backend paints the buffer as a QImage tagged with a devicePixelRatio and lets
            # Qt scale it -- but Qt's image-DPR semantics are only well-defined for ratios >= 1, and a
            # display_scale below 1 (0.7) once left the buffer blitted 1:1 into the TOP-LEFT corner with
            # the right/bottom of the widget blank (the "enlarged view is not centred" bug -- content
            # occupied ~71% of the canvas).  Drawing it ourselves over ``rect()`` needs no image-DPR at
            # all: the widget size is the DPR-invariant design constant, the buffer is whatever dpi the
            # sync chose, and the smooth-transform scale maps one onto the other exactly.
            del event
            front = getattr(self, "_zlc_front", None)
            if front is not None:
                # Front-buffer mode: blit the last PRESENTED frame.  Never touch the Agg buffer here
                # -- the render thread may be composing the next frame into it right now.
                painter = QtGui.QPainter(self)
                painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
                painter.drawImage(self.rect(), self._zlc_front_image(front))
                painter.end()
                return
            # Fallback path (no front image -- a GUI-thread-only host, or before the first present):
            # the Agg buffer is about to be read on the GUI thread, so wait out any in-flight compose
            # first.  If the barrier TIMES OUT (a wedged render), a worker may STILL be composing into
            # the buffer -- reading it now is exactly the torn colorbar (#9), so skip the blit and log
            # (Qt keeps the previous pixels: a stale frame beats a torn one).
            if not self._zlc_wait_render():
                logging.getLogger(__name__).warning(
                    "paintEvent: render barrier timed out; skipping Agg blit to avoid a torn buffer read")
                return
            self._zlc_draw_if_needed()          # render the buffer only if it is out of date (else reuse)
            renderer = self.renderer
            # Own a copy first -- never a live view into the Agg buffer that a subsequent compose could
            # overwrite mid-blit.  Same construction as the front, so the two paths cannot drift.
            owned = RasterBuffer.from_agg_rgba(
                int(renderer.width), int(renderer.height), self.buffer_rgba())
            image = self._zlc_front_image(owned)
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.drawImage(self.rect(), image)
            painter.end()

        def resizeEvent(self, event):  # noqa: N802 - Qt naming
            # spec-owned figure: NEVER re-derive the figure geometry from the widget size (the stock
            # handler does, and any transient mismatch warps the fixed-inches axes layout).  The Agg
            # BUFFER is dpi-based -- independent of widget size (paintEvent stretch-blits it over rect())
            # -- so a resize needs only a RE-BLIT, not a re-render: update() repaints the existing buffer,
            # NOT draw_idle() (which re-renders the whole heavy figure for a pixel-identical result).
            QtWidgets.QWidget.resizeEvent(self, event)
            self.update()

        # ------------------------------------------------------------- behaviour
        def _zlc_wait_render(self) -> bool:
            # The render-thread ownership barrier: a host that composes this canvas off the GUI
            # thread hangs its barrier here (``_zlc_render_barrier``); every mouse path into the
            # figure (press, double-click, wheel zoom) waits for the in-flight compose FIRST, so
            # a selector/zoom callback never mutates artists the worker is rasterising.  Free when
            # idle; unset on GUI-thread-only hosts (pulse_gui, Edit snapshots).  Returns True when
            # the loop is settled (or there is no barrier), False ONLY on a pathological barrier
            # timeout -- the paintEvent fallback then must NOT read the Agg buffer (a wedged worker
            # may still be writing it).
            # Short-circuit when this IS the render worker: a compose-phase stock ``draw()`` already
            # OWNS the figure, so it must never wait on its own in-flight job (the drag self-deadlock).
            if threading.current_thread().name == RENDER_THREAD_NAME:
                return True
            barrier = getattr(self, "_zlc_render_barrier", None)
            if callable(barrier):
                return bool(barrier())
            return True

        def mousePressEvent(self, event):  # noqa: N802 - Qt naming
            self._zlc_wait_render()
            super().mousePressEvent(event)

        def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt naming
            self._zlc_wait_render()
            super().mouseDoubleClickEvent(event)

        def wheelEvent(self, event):  # noqa: N802 - Qt naming
            if not self._zlc_isolate_wheel:
                # display-only panel: let the wheel bubble up to the scroll area
                # so the dashboard board scrolls under the cursor.
                event.ignore()
                return
            # interactive plot: in-plot wheel zoom must never double as a page scroll
            self._zlc_wait_render()
            super().wheelEvent(event)
            event.accept()


def panel_canvas(figure, *, isolate_wheel: bool = True):
    """The canvas for a dashboard panel figure: the panel display scale is a
    frontend design constant (``style.PANEL_DISPLAY_SCALE``), not a host knob.
    ``isolate_wheel=False`` lets a display-only (Monitor) panel's wheel scroll the
    board instead of being swallowed; interactive (Edit) panels keep the default."""

    if EmbeddedFigureCanvas is None:  # pragma: no cover - matplotlib-qt missing
        raise RuntimeError("matplotlib Qt canvas is not available")
    from zlc_frontend.render_style import PANEL_DISPLAY_SCALE
    # The ONE display knob: the Agg buffer is rendered at exactly the widget's on-screen
    # device pixels (figure.dpi carries display_scale), so the blit is 1:1 -- no softness.
    # Saved figures use savefig.dpi, independently of this.
    return EmbeddedFigureCanvas(figure, display_scale=PANEL_DISPLAY_SCALE,
                                isolate_wheel=isolate_wheel)


__all__ = ["EmbeddedFigureCanvas", "panel_canvas"]
