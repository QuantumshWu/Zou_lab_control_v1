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
    from PyQt5 import QtCore, QtGui, QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvasQTAgg
except Exception:  # pragma: no cover - depends on the local matplotlib install
    _FigureCanvasQTAgg = None


if _FigureCanvasQTAgg is None:  # pragma: no cover - matplotlib-qt missing
    EmbeddedFigureCanvas = None
else:

    class EmbeddedFigureCanvas(_FigureCanvasQTAgg):
        """Matplotlib Qt canvas with a display scale and wheel isolation."""

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
            self.draw()
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
            try:
                self._zlc_sync()                       # figure dpi -> the REAL screen ratio (crisp buffer)
            except RuntimeError:
                return
            self.setFixedSize(self._zlc_design_size())  # idempotent (a monitor move only shifts display_scale)
            self.updateGeometry()                       # let the parent CARD re-measure us
            self.draw()                                 # SYNCHRONOUS: a resync never leaves a blank frame

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
            self._zlc_sync()
            return True

        def _zlc_sync(self) -> None:
            real = super().devicePixelRatioF() or 1.0
            ds = self._zlc_display_scale
            figure = self.figure
            # invariant 1: the design inches NEVER change (fixed-inches axes)
            figure.set_size_inches(*self._zlc_inches, forward=False)
            # invariant 2: the buffer dpi = design dpi x REAL screen ratio (retina supersampling)
            # x display_scale, so the Agg buffer is EXACTLY the widget's on-screen device pixels
            # (a 1:1 crisp blit -- never rendered small and stretched up).
            figure._set_dpi(self._zlc_design_dpi * real * ds, forward=False)
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
            renderer = getattr(self, "renderer", None)
            if renderer is None or not getattr(self, "_zlc_painted_once", False):
                self.draw()                     # ensure a rendered buffer before the first blit
                self._zlc_painted_once = True
                renderer = self.renderer
            w, h = int(renderer.width), int(renderer.height)
            image = QtGui.QImage(self.buffer_rgba(), w, h, QtGui.QImage.Format_RGBA8888)
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.drawImage(self.rect(), image)
            painter.end()

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
    frontend design constant (``style.PANEL_DISPLAY_SCALE``), not a host knob.
    ``isolate_wheel=False`` lets a display-only (Monitor) panel's wheel scroll the
    board instead of being swallowed; interactive (Edit) panels keep the default."""

    if EmbeddedFigureCanvas is None:  # pragma: no cover - matplotlib-qt missing
        raise RuntimeError("matplotlib Qt canvas is not available")
    from .style import PANEL_DISPLAY_SCALE
    # The ONE display knob: the Agg buffer is rendered at exactly the widget's on-screen
    # device pixels (figure.dpi carries display_scale), so the blit is 1:1 -- no softness.
    # Saved figures use savefig.dpi, independently of this.
    return EmbeddedFigureCanvas(figure, display_scale=PANEL_DISPLAY_SCALE,
                                isolate_wheel=isolate_wheel)


__all__ = ["EmbeddedFigureCanvas", "panel_canvas"]
