"""Embedding matplotlib figures in Qt windows: the ONE display-scale wrapper.

Figures keep the single frontend font/geometry system (style.DEFAULT_STYLE,
dpi=300) -- font sizes are NEVER forked per host.  How large a figure APPEARS in
a Qt window is a display concern, handled here through matplotlib's standard
device-pixel-ratio path: the canvas reports an inflated ratio, so the figure
renders at its full pixel size and Qt shows it scaled by ``display_scale`` --
exactly like an OS high-DPI screen.  Interaction coordinates stay exact (the Qt
backend converts mouse positions through the same ratio) and the result is
supersampled, never blurry.

Every Qt host that embeds a figure should use :class:`EmbeddedFigureCanvas`
(``display_scale=1.0`` shows 1:1); it also stops wheel events from leaking into
a surrounding QScrollArea, so in-plot zoom never scrolls the page.
"""

from __future__ import annotations

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvasQTAgg
except Exception:  # pragma: no cover - depends on the local matplotlib install
    _FigureCanvasQTAgg = None


if _FigureCanvasQTAgg is None:  # pragma: no cover - matplotlib-qt missing
    EmbeddedFigureCanvas = None
else:

    class EmbeddedFigureCanvas(_FigureCanvasQTAgg):
        """Matplotlib Qt canvas with a display scale and wheel isolation."""

        def __init__(self, figure, *, display_scale: float = 1.0):
            # must exist BEFORE super().__init__: the base class reads
            # devicePixelRatioF() (overridden below) during construction.
            self._zlc_ratio = 1.0 / max(0.1, float(display_scale))
            super().__init__(figure)
            # The backend normally syncs its cached ratio in showEvent (via the
            # window's screen signals) -- too late here, and never offscreen.
            # Sync NOW.  The sync re-derives the FIGURE size from the widget's
            # current logical size (inflating the figure by our ratio), so pin
            # the figure back to its own pixel size and give the widget the
            # corresponding logical size; the resulting resizeEvent re-derives
            # the figure at exactly its original pixels.  Later screen changes
            # still re-sync through the stock signals (times our factor).
            width_px, height_px = map(float, figure.bbox.max)
            self._update_pixel_ratio()
            ratio = self.devicePixelRatioF() or 1.0
            figure.set_size_inches(width_px / figure.dpi, height_px / figure.dpi, forward=False)
            self.resize(round(width_px / ratio), round(height_px / ratio))

        def devicePixelRatioF(self):  # noqa: N802 - Qt naming
            # The backend derives the render-buffer size, sizeHint, mouse-event
            # coordinates and the painter's image scaling from this one ratio.
            return (super().devicePixelRatioF() or 1.0) * self._zlc_ratio

        def _set_device_pixel_ratio(self, ratio):
            # The stock implementation also MAGNIFIES figure.dpi by the ratio --
            # the "retina" semantics: same on-screen size, more pixels.  Our
            # display_scale wants the OPPOSITE: keep the figure's design pixels
            # and show them SMALLER.  So inflate the dpi only by the REAL screen
            # ratio (preserving genuine high-DPI behaviour) and never by our
            # display factor, which therefore acts as a pure display zoom.
            if getattr(self, "_device_pixel_ratio", None) == ratio:
                return False
            real = super().devicePixelRatioF() or 1.0
            self.figure._set_dpi(self.figure._original_dpi * real, forward=False)
            self._device_pixel_ratio = ratio
            return True

        def wheelEvent(self, event):  # noqa: N802 - Qt naming
            # in-plot wheel zoom must never double as a page scroll
            super().wheelEvent(event)
            event.accept()


__all__ = ["EmbeddedFigureCanvas"]
