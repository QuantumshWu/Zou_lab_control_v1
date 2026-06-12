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

try:
    from PyQt5 import QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvasQTAgg
except Exception:  # pragma: no cover - depends on the local matplotlib install
    _FigureCanvasQTAgg = None


if _FigureCanvasQTAgg is None:  # pragma: no cover - matplotlib-qt missing
    EmbeddedFigureCanvas = None
else:

    class EmbeddedFigureCanvas(_FigureCanvasQTAgg):
        """Matplotlib Qt canvas with a display scale and wheel isolation."""

        def __init__(self, figure, *, display_scale: float = 1.0):
            # both must exist BEFORE super().__init__: the base class reads
            # devicePixelRatioF() (overridden below) during construction.
            self._zlc_ratio = 1.0 / max(0.1, float(display_scale))
            self._zlc_inches = tuple(float(v) for v in figure.get_size_inches())
            super().__init__(figure)
            # the backend syncs only in showEvent / on screen signals (never
            # offscreen) -- establish the invariants NOW
            self._zlc_sync()

        # ------------------------------------------------------------- ratio math
        def devicePixelRatioF(self):  # noqa: N802 - Qt naming
            # The backend derives the render-buffer size, sizeHint, mouse-event
            # coordinates and the painter's image scaling from this one ratio.
            return (super().devicePixelRatioF() or 1.0) * self._zlc_ratio

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
            figure = self.figure
            # invariant 1: the design inches NEVER change (fixed-inches axes)
            figure.set_size_inches(*self._zlc_inches, forward=False)
            # invariant 2: retina supersampling by the REAL screen ratio only
            figure._set_dpi(figure._original_dpi * real, forward=False)
            self._device_pixel_ratio = real * self._zlc_ratio
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
            # in-plot wheel zoom must never double as a page scroll
            super().wheelEvent(event)
            event.accept()


def panel_canvas(figure):
    """The canvas for a dashboard panel figure: the panel display scale is a
    frontend design constant, not a host knob."""

    if EmbeddedFigureCanvas is None:  # pragma: no cover - matplotlib-qt missing
        raise RuntimeError("matplotlib Qt canvas is not available")
    from .live import PANEL_DISPLAY_SCALE
    return EmbeddedFigureCanvas(figure, display_scale=PANEL_DISPLAY_SCALE)


__all__ = ["EmbeddedFigureCanvas", "panel_canvas"]
