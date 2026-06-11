"""Reusable PyQt Fluent-style widgets for Zou_lab_control frontends.

The visual constants and widget shapes follow the original Confocal_GUIv2
Fluent layer: pale blue accent, Segoe UI text, white cards, small radii, and
soft shadows.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

try:  # Optional, but used when available to match Confocal_GUIv2 windows.
    from qframelesswindow import FramelessWindow, StandardTitleBar
except Exception:  # pragma: no cover - depends on optional desktop package.
    FramelessWindow = QtWidgets.QWidget
    StandardTitleBar = None


ACCENT = "#77AADD"
HOVER = "#004578"
BG = "#F3F3F3"
TEXT = "#323130"
HINT = "#F0a150"
PLACEHOLDER = "#A19F9D"
# Light neutral hairline used to delineate cards that opt out of a drop shadow
# (shadows are costly to repaint while scrolling).
DIVIDER = "#E1DFDD"
GREEN = "#7FC2AD"
RED = "#CD7380"
ORANGE = "#D69A6E"
# A scan-bound field is painted a pale orange so the saturated-orange scan dot
# (the inline spinbox-style button) stands out against it instead of vanishing.
ORANGE_TINT = "#F6E3D4"
ORANGE_DARK = "#8A4B1F"
YELLOW = "#E5C85B"
GREY = "#A2A2A2"
RADIUS = 4
FONT = "Segoe UI"
FONT_SIZE = 12
PADDING_V = 1
PADDING_H = 1
EDIT_PADDING_H = 4
COMBO_WIDTH = 16
COMBO_TRI_SIZE = 8
STEP_WIDTH = 6

_QT_APP = None
_FLUENT_SCALE = 1.0
# Matches one float token; used by align_to_resolution to snap numbers inside a value.
_FLOAT_TOKEN_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def fluent_scale() -> float:
    return _FLUENT_SCALE


def set_fluent_scale(scale: float | None = None) -> float:
    """Set the scale used by subsequently created Fluent widgets."""

    global _FLUENT_SCALE
    if scale is None:
        scale = 1.0
    _FLUENT_SCALE = max(0.72, min(1.25, float(scale)))
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.setFont(QtGui.QFont(FONT, fluent_font_size()))
    return _FLUENT_SCALE


def scaled_px(value: int | float, *, minimum: int = 1) -> int:
    return max(int(minimum), int(round(float(value) * _FLUENT_SCALE)))


def fluent_font_size() -> int:
    return max(8, int(round(FONT_SIZE * _FLUENT_SCALE)))


def fluent_text_width(metrics: QtGui.QFontMetrics, text: str) -> int:
    """Return text width on old and new PyQt5 builds."""

    if hasattr(metrics, "horizontalAdvance"):
        return int(metrics.horizontalAdvance(text))
    return int(metrics.width(text))


def _radius() -> int:
    return scaled_px(RADIUS)


def ensure_qt_app() -> QtWidgets.QApplication:
    """Return a QApplication, creating and keeping one alive when needed."""

    global _QT_APP
    app = QtWidgets.QApplication.instance()
    if app is not None:
        _QT_APP = app
        return app
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # Silence the harmless Windows "Unable to open default EUDC font" Qt warning.
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
    _QT_APP = QtWidgets.QApplication(sys.argv)
    _QT_APP.setFont(QtGui.QFont(FONT, fluent_font_size()))
    return _QT_APP


def fluent_widget_stylesheet() -> str:
    return f'QWidget {{ background: {BG}; color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; }}'


def fluent_scrollbar_stylesheet(selector: str = "QScrollBar") -> str:
    """Return shared Fluent scrollbar CSS for scroll areas and popup lists."""

    thickness = scaled_px(12, minimum=10)
    return f"""
    {selector}:vertical {{
        background: transparent;
        border: none;
        width: {thickness}px;
        margin: 0px;
    }}
    {selector}:horizontal {{
        background: transparent;
        border: none;
        height: {thickness}px;
        margin: 0px;
    }}
    {selector}::handle:vertical, {selector}::handle:horizontal {{
        background: #C8C6C4;
        border: none;
        border-radius: {_radius()}px;
        min-height: {scaled_px(28)}px;
        min-width: {scaled_px(28)}px;
    }}
    {selector}::handle:vertical:hover, {selector}::handle:horizontal:hover {{
        background: {ACCENT};
    }}
    {selector}::add-line, {selector}::sub-line {{
        width: 0px;
        height: 0px;
        border: none;
        background: transparent;
    }}
    {selector}::add-page, {selector}::sub-page {{
        background: transparent;
    }}
    """


def status_dot_stylesheet(color: str, *, radius: int = 8) -> str:
    return f"background:{color}; border-radius:{scaled_px(radius)}px;"


# CachedDropShadow caches BAKED SHADOWS keyed by the widget's SILHOUETTE -- the
# geometry that determines the shadow (size + corner radii, or the tab-bar step
# for the tab widget) -- so the expensive Gaussian blur runs once per shape, not
# on every repaint.  The live widget content is rasterised fresh each paint (via
# Qt's own sourcePixmap machinery, which also keeps NESTED effects working).
# Contract: a shadowed widget must paint an opaque silhouette matching its
# declared path; widgets with other shapes need a silhouette provider like the
# tab widget's (see _tab_widget_silhouette).
_SHADOW_PIXMAP_CACHE: dict[tuple, QtGui.QPixmap] = {}


def _baked_silhouette_shadow(key: tuple, path: QtGui.QPainterPath, width: int, height: int,
                             margin: int, blur: int, alpha: int, offset: int,
                             dpr: float = 1.0) -> QtGui.QPixmap:
    """Shadow+silhouette composite for an opaque white shape, cached by ``key``.

    ``path`` is in widget LOGICAL coordinates; ``width``/``height`` are the
    padded effective rect in DEVICE pixels (logical * dpr).  The bake scales
    the path geometry, blur and offset by ``dpr`` so the shadow is rendered at
    the screen's native resolution; the returned pixmap is a plain dpr=1
    device-pixel image (blitted 1:1 under an identity transform).  Baked by
    running the real ``QGraphicsDropShadowEffect`` once in a throwaway scene,
    so the blur kernel/rounding are exactly the stock effect's.
    """
    cached = _SHADOW_PIXMAP_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_SHADOW_PIXMAP_CACHE) > 256:   # bound a long session full of resizes
        _SHADOW_PIXMAP_CACHE.clear()
    scene = QtWidgets.QGraphicsScene()
    item = scene.addPath(path, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtGui.QColor("white")))
    effect = QtWidgets.QGraphicsDropShadowEffect()
    effect.setBlurRadius(blur)
    effect.setColor(QtGui.QColor(0, 0, 0, alpha))
    effect.setOffset(0, offset)
    item.setGraphicsEffect(effect)
    # The dpr-tagged pixmap makes the painter rasterise at DEVICE resolution
    # while the scene/path/effect stay in logical units -- the same machinery
    # the stock effect goes through on a high-DPI screen.
    log_w, log_h = width / dpr, height / dpr
    pixmap = QtGui.QPixmap(width, height)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    scene.render(painter, QtCore.QRectF(0, 0, log_w, log_h),
                 QtCore.QRectF(-margin, -margin, log_w, log_h))
    painter.end()
    _SHADOW_PIXMAP_CACHE[key] = pixmap
    return pixmap


def _tab_widget_silhouette(widget: QtWidgets.QTabWidget):
    """Silhouette of a FluentTabWidget: tab strip (rounded tops) + pane.

    The widget is NOT an opaque rounded rect -- the area right of the tabs is
    transparent (the window background shows through), so a rounded-rect bake
    would paint a white band there and cast shadow where the stock effect casts
    none.  The 2 px notches between individual tabs are washed out by the blur,
    so one strip spanning the tabs matches the stock shadow visually.
    """
    bar = widget.tabBar()
    geo = bar.geometry()
    last = bar.tabRect(bar.count() - 1) if bar.count() else QtCore.QRect(0, 0, 0, 0)
    strip_w = min(geo.width(), last.x() + last.width())
    w, h, r = widget.width(), widget.height(), _radius()
    pane_top = geo.height()
    key = ("tabs", w, h, pane_top, strip_w, geo.x(), geo.y(), r)
    pane = QtGui.QPainterPath()
    # WindingFill: overlapping subpaths must UNION (the default odd-even rule
    # would XOR the corner square into a hole inside the pane).
    pane.setFillRule(QtCore.Qt.WindingFill)
    pane.addRoundedRect(0.0, float(pane_top), float(w), float(h - pane_top), float(r), float(r))
    pane.addRect(0.0, float(pane_top), float(2 * r), float(2 * r))   # square NW corner (tabs sit on it)
    strip = QtGui.QPainterPath()
    strip.addRoundedRect(float(geo.x()), float(geo.y()), float(strip_w), float(geo.height() + r), float(r), float(r))
    return key, pane.united(strip).simplified()


class CachedDropShadow(QtWidgets.QGraphicsEffect):
    """Drop shadow with a silhouette-cached blur + live source painting.

    A stock ``QGraphicsDropShadowEffect`` re-rasterises the source widget and
    re-runs the Gaussian blur on EVERY paint -- scrolling a panel of shadowed
    cards spent ~90% of each frame in the blur.  Here the blurred shadow is
    baked once per silhouette (size + shape) by the real stock effect and
    blitted, while the live widget is rasterised through Qt's own
    ``sourcePixmap`` machinery and drawn on top (fresh content every paint ->
    no ghosting).

    Two hard-won correctness points (do not "simplify" these away):

    * ``draw`` must follow the stock structure -- ``sourcePixmap`` in DEVICE
      coordinates, paint under an identity world transform at the returned
      offset.  Calling ``drawSource`` instead breaks the paint context of any
      NESTED graphics effect (children render with a (0,0)/identity painter
      and their shadows disappear), which blanked every card shadow inside
      the shadowed tab widget.
    * The silhouette must be the widget's TRUE opaque outline.  ``silhouette``
      (when given) returns (cache-key, QPainterPath) for non-rounded-rect
      widgets such as the tab widget; the default is a rounded rect with this
      effect's corner radii.
    """

    def __init__(self, parent=None, *, radius: int, blur: int, alpha: int, offset: int,
                 corner_radii: tuple = None, silhouette=None):
        super().__init__(parent)
        self._radius = int(radius)
        self._blur = int(blur)
        self._alpha = int(alpha)
        self._offset = int(offset)
        self._corner_radii = corner_radii   # (tl, tr, br, bl) or None -> uniform radius
        self._silhouette = silhouette       # callable(widget) -> (key, QPainterPath)

    def boundingRectFor(self, rect: QtCore.QRectF) -> QtCore.QRectF:  # noqa: N802
        margin = float(self._blur + abs(self._offset))
        return rect.adjusted(-margin, -margin, margin, margin)

    def _default_silhouette(self, width: float, height: float):
        radii = self._corner_radii or (self._radius,) * 4
        tl, tr, br, bl = (float(v) for v in radii)
        path = QtGui.QPainterPath()
        if tl == tr == br == bl:
            path.addRoundedRect(0.0, 0.0, width, height, tl, tl)
        else:
            # Per-corner radii (FluentFrame's round=(...) option).
            path.moveTo(tl, 0.0)
            path.lineTo(width - tr, 0.0)
            path.arcTo(width - 2 * tr, 0.0, 2 * tr, 2 * tr, 90.0, -90.0)
            path.lineTo(width, height - br)
            path.arcTo(width - 2 * br, height - 2 * br, 2 * br, 2 * br, 0.0, -90.0)
            path.lineTo(bl, height)
            path.arcTo(0.0, height - 2 * bl, 2 * bl, 2 * bl, -90.0, -90.0)
            path.lineTo(0.0, tl)
            path.arcTo(0.0, 0.0, 2 * tl, 2 * tl, 180.0, -90.0)
            path.closeSubpath()
        key = ("rrect", int(width), int(height), tl, tr, br, bl)
        return key, path

    def draw(self, painter: QtGui.QPainter) -> None:  # noqa: N802
        if self._blur <= 0 and self._offset == 0:
            self.drawSource(painter)
            return
        # Stock-effect structure: sourcePixmap() triggers Qt's proper
        # (nested-safe) source rendering and yields the device-coordinate
        # offset to paint at.
        src, off = self.sourcePixmap(QtCore.Qt.DeviceCoordinates,
                                     QtWidgets.QGraphicsEffect.PadToEffectiveBoundingRect)
        if src.isNull():
            return
        # `src` is a device-resolution pixmap tagged with the screen dpr; the
        # returned offset is LOGICAL, and under the identity world transform a
        # dpr-tagged pixmap drawn at a logical position lands exactly on its
        # device pixels (the painter keeps the dpr device transform).  The bake
        # mirrors that: device-resolution pixels, logical geometry.
        dpr = float(src.devicePixelRatioF() or 1.0)
        margin = self._blur + abs(self._offset)
        widget = self.parent()
        if self._silhouette is not None and widget is not None:
            sil_key, path = self._silhouette(widget)
        else:
            sil_key, path = self._default_silhouette(
                src.width() / dpr - 2.0 * margin, src.height() / dpr - 2.0 * margin)
        key = sil_key + (self._blur, self._alpha, self._offset, round(dpr, 2))
        shadow = _baked_silhouette_shadow(key, path, src.width(), src.height(),
                                          margin, self._blur, self._alpha, self._offset, dpr)
        restore = painter.worldTransform()
        painter.setWorldTransform(QtGui.QTransform())
        painter.drawPixmap(off, shadow)
        painter.drawPixmap(off, src)
        painter.setWorldTransform(restore)


def add_fluent_shadow(widget: QtWidgets.QWidget, *, blur: int = 20, alpha: int = 50, offset: int = 0,
                      corner_radii: tuple = None, silhouette=None) -> None:
    shadow = CachedDropShadow(
        widget, radius=_radius(), blur=scaled_px(blur), alpha=alpha,
        offset=scaled_px(offset, minimum=0), corner_radii=corner_radii, silhouette=silhouette)
    widget.setGraphicsEffect(shadow)


@contextlib.contextmanager
def signals_blocked(*widgets: QtWidgets.QWidget | None):
    """Temporarily block Qt signals on each (non-None) widget, restoring after.

    Handy when updating many existing widgets' values in place (e.g. refreshing
    a form from a model) without each ``setText`` / ``setCurrentText`` firing a
    feedback signal -- the standard alternative to destroying + rebuilding the
    widgets.
    """

    saved = [(w, w.blockSignals(True)) for w in widgets if w is not None]
    try:
        yield
    finally:
        for widget, previous in saved:
            widget.blockSignals(previous)


@contextlib.contextmanager
def batched_updates(*widgets: QtWidgets.QWidget | None):
    """Suspend repaints on each widget for a bulk change, repainting once after.

    Wrap a teardown + rebuild of many child widgets in this so each ``addWidget``
    on the (visible) tree does not trigger its own relayout + repaint; the single
    repaint on exit is the dominant speed-up for large dynamic panels.
    """

    saved = [(w, w.updatesEnabled()) for w in widgets if w is not None]
    for widget, _previous in saved:
        widget.setUpdatesEnabled(False)
    try:
        yield
    finally:
        for widget, previous in saved:
            widget.setUpdatesEnabled(previous)


def format_compact_number(value: float, *, digits: int = 12) -> str:
    if not math.isfinite(float(value)):
        return str(value)
    text = f"{float(value):.{digits}g}"
    return text.replace("e+0", "e").replace("e+", "e")


def _confocal_float2str(value: float, *, length: int | None = None) -> str:
    """Match Confocal_GUIv2.helper.float2str for numeric spin boxes."""

    if length is not None:
        length = max(int(length), 5)
        sign = "-" if value < 0 else ""
        abs_val = abs(float(value))
        int_part = str(int(abs_val))
        dec_places = length - len(sign) - len(int_part) - 1
        if dec_places < 0:
            dec_places = 0
        text = f"{float(value):.{dec_places}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text[:length]
    return str(value)


def align_to_resolution(value: str | float, resolution: float, *, allow_any: bool = True) -> str:
    """Snap the numeric part of a simple value/x expression to ``resolution``."""

    if resolution is None or float(resolution) <= 0:
        return str(value)
    text = str(value).strip()
    if not text:
        return text

    def snap_number(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            number = float(token)
        except ValueError:
            return token
        snapped = round(number / float(resolution)) * float(resolution)
        if not allow_any and snapped <= 0:
            snapped = float(resolution)
        return format_compact_number(snapped)

    if any(char in text for char in "[](),"):
        return _FLOAT_TOKEN_RE.sub(snap_number, text)
    return snap_number(re.match(r".+", text))


class FluentStatusDot(QtWidgets.QLabel):
    def __init__(self, parent=None, *, color: str = GREY, size: int = 16):
        super().__init__(parent)
        self._size = scaled_px(size)
        self.setFixedSize(self._size, self._size)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        self.setStyleSheet(status_dot_stylesheet(color, radius=max(1, self._size // 2)))


class FluentLabel(QtWidgets.QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f'QLabel {{ color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; background: transparent; }}')


class FluentFrame(QtWidgets.QFrame):
    def __init__(self, parent=None, *, shadow: bool = True, round: tuple[str, ...] = ("NW", "NE", "SE", "SW")):
        super().__init__(parent)
        corners = {corner.upper() for corner in round}
        top_left = _radius() if "NW" in corners else 0
        top_right = _radius() if "NE" in corners else 0
        bottom_right = _radius() if "SE" in corners else 0
        bottom_left = _radius() if "SW" in corners else 0
        self.setStyleSheet(
            f"""
            QFrame {{
                background: white;
                border: none;
                border-top-left-radius: {top_left}px;
                border-top-right-radius: {top_right}px;
                border-bottom-right-radius: {bottom_right}px;
                border-bottom-left-radius: {bottom_left}px;
            }}
            """
        )
        if shadow:
            add_fluent_shadow(self, corner_radii=(top_left, top_right, bottom_right, bottom_left))


class FluentGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title: str = "", parent=None, *, shadow: bool = True):
        super().__init__(title, parent)
        # A soft drop shadow (QGraphicsDropShadowEffect) is expensive to repaint
        # -- it rasterises + blurs the whole widget every frame, which makes a
        # tall, scrolled panel stutter.  When shadow is off we draw a light 1px
        # border instead so the card is still delineated, for a fraction of the
        # paint cost.
        border = "none" if shadow else f"1px solid {DIVIDER}"
        self.setStyleSheet(
            f"""
            QGroupBox {{
                background: white;
                border: {border};
                border-radius: {_radius()}px;
                margin-top: 0px;
                padding-top: {scaled_px(32)}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                background: {BG};
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                border-radius: {_radius()}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            """
        )
        if shadow:
            add_fluent_shadow(self)


class FluentButton(QtWidgets.QPushButton):
    def __init__(self, text: str = "", parent=None, *, color: str = ACCENT):
        super().__init__(text, parent)
        self._current_bg = None
        self.setMinimumHeight(scaled_px(30, minimum=22))
        # Default sizing is set ONCE here, not inside set_color -- callers that
        # opt into Expanding (e.g. a button grid) would otherwise have their
        # policy silently reset to Fixed the next time the colour changes, making
        # the button collapse to its text width (the "Save button shrinks after
        # load" bug).
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        bg = QtGui.QColor(color).name(QtGui.QColor.HexRgb)
        if bg == self._current_bg:
            return
        self._current_bg = bg
        hover = QtGui.QColor(bg).darker(184).name(QtGui.QColor.HexRgb)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: {bg};
                color: white;
                border: none;
                border-radius: {_radius()}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:disabled {{ background: {PLACEHOLDER}; color: {BG}; }}
            """
        )


class FluentLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._res_step: float | None = None
        self._allow_any = True
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.setStyleSheet(
            f"""
            QLineEdit {{
                background: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {_radius()}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
            QLineEdit:disabled {{ background: {BG}; color: {PLACEHOLDER}; }}
            """
        )
        self.setText(str(text))
        self.editingFinished.connect(self._snap_to_resolution)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        text = str(text)
        if self.text() == text:
            return   # no-op guard: skip the (expensive) re-set + cursor reset when unchanged
        super().setText(text)
        if not self.hasFocus():
            self.setCursorPosition(0)

    def set_resolution(self, step: float | None) -> None:
        self._res_step = None if step in (None, 0) else float(step)

    def set_allow_any(self, allow_any: bool = True) -> None:
        self._allow_any = bool(allow_any)

    def set_numeric_validator(
        self,
        kind: str = "float",
        *,
        bottom: float | None = None,
        top: float | None = None,
        decimals: int = 12,
    ) -> None:
        """Restrict typed input to a number (like the Confocal GUI's FloatLineEdit).

        ``kind="float"`` accepts only digits, a decimal point, ``e``/``E`` and a sign
        (scientific notation); ``kind="int"`` accepts only an integer.  Optional
        ``bottom``/``top`` bound the value (an int field with a ``top`` also blocks
        clearly-too-big entries as you type).  Other characters are simply rejected at
        the keystroke, so the field can never hold non-numeric junk."""

        if kind == "int":
            validator: QtGui.QValidator = QtGui.QIntValidator(self)
            if bottom is not None:
                validator.setBottom(int(bottom))
            if top is not None:
                validator.setTop(int(top))
        else:
            validator = QtGui.QDoubleValidator(self)
            validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
            validator.setDecimals(int(decimals))
            if bottom is not None:
                validator.setBottom(float(bottom))
            if top is not None:
                validator.setTop(float(top))
        # Force a dot decimal separator regardless of the system locale.
        validator.setLocale(QtCore.QLocale.c())
        self.setValidator(validator)

    def _snap_to_resolution(self) -> None:
        if not self._res_step:
            return
        before = self.text()
        after = align_to_resolution(before, self._res_step, allow_any=self._allow_any)
        if after != before:
            self.setText(after)


class FloatLineEdit(FluentLineEdit):
    pass


class FluentComboBox(QtWidgets.QComboBox):
    """A non-editable Fluent combo that paints its own current text.

    QComboBox's editable-lineEdit display does not render reliably on the Qt
    ``offscreen`` platform (and is finicky to style), so the current item text
    is drawn directly in :meth:`paintEvent`.  This also makes the text always
    visible for screenshots and avoids cursor/clipping quirks.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.setEditable(False)
        self.setStyleSheet(
            f"""
            QComboBox {{
                background-color: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {_radius()}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
                padding: 0px;
            }}
            QComboBox:disabled {{
                background-color: {BG};
                border: 1px solid {PLACEHOLDER};
                color: {PLACEHOLDER};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: border;
                subcontrol-position: right;
                width: {scaled_px(COMBO_WIDTH)}px;
                border: none;
                background-color: {ACCENT};
                border-top-right-radius: {_radius()}px;
                border-bottom-right-radius: {_radius()}px;
            }}
            QComboBox::drop-down:disabled {{ background-color: {PLACEHOLDER}; }}
            QComboBox::drop-down:hover {{ background-color: {HOVER}; }}
            QComboBox::down-arrow {{ image: none; width: 0px; height: 0px; }}
            QComboBox QAbstractItemView {{
                background: white;
                border: 1px solid {PLACEHOLDER};
                color: {TEXT};
                selection-background-color: {ACCENT};
                selection-color: white;
                font: {fluent_font_size()}pt "{FONT}";
                outline: none;
            }}
            """
        )

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        option = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = ""  # suppress the style's own label so we don't double-draw
        self.style().drawComplexControl(QtWidgets.QStyle.CC_ComboBox, option, painter, self)

        drop_width = scaled_px(COMBO_WIDTH)
        pad = scaled_px(EDIT_PADDING_H)
        # A styled QLineEdit insets its text by the frame width (~2 px) on top of
        # the stylesheet padding, so a combo that paints text at `pad` alone sits
        # a couple of pixels further left than the line edits / spin boxes beside
        # it.  Add the frame allowance so e.g. the "ns" unit lines up exactly with
        # the duration value above it.
        text_inset = pad + scaled_px(2)
        text_width = max(0, self.width() - drop_width - text_inset - pad)
        text_rect = QtCore.QRect(text_inset, 0, text_width, self.height())
        painter.setPen(QtGui.QColor(TEXT if self.isEnabled() else PLACEHOLDER))
        painter.setFont(QtGui.QFont(FONT, fluent_font_size()))
        metrics = painter.fontMetrics()
        text = metrics.elidedText(self.currentText(), QtCore.Qt.ElideRight, text_width)
        painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, text)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#FFFFFF"))
        cx = int(self.width() - drop_width / 2)
        cy = int(self.height() / 2)
        size = scaled_px(COMBO_TRI_SIZE)
        points = [
            QtCore.QPoint(cx - size // 2, cy - size // 4),
            QtCore.QPoint(cx + size // 2, cy - size // 4),
            QtCore.QPoint(cx, cy + size // 4),
        ]
        painter.drawPolygon(QtGui.QPolygon(points))
        painter.end()

    def showPopup(self):
        # Size the dropdown to its widest item so options like "Edge"/"Ramp"/
        # "Hold" or "ns"/"us"/"ms" are never clipped when the combo itself is
        # narrow (the popup defaults to the combo width otherwise).
        view = self.view()
        if view is not None and self.count():
            metrics = view.fontMetrics()
            widest = 0
            for index in range(self.count()):
                try:
                    advance = metrics.horizontalAdvance(self.itemText(index))
                except AttributeError:  # pragma: no cover - very old Qt
                    advance = metrics.width(self.itemText(index))
                widest = max(widest, advance)
            view.setMinimumWidth(widest + scaled_px(COMBO_WIDTH) + scaled_px(EDIT_PADDING_H) * 2)
        super().showPopup()

    def wheelEvent(self, event):
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class FluentTabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabBar().setExpanding(False)
        self.setUsesScrollButtons(True)
        self.setStyleSheet(
            f"""
            QTabWidget::pane {{
                background: white;
                margin: 0px;
                padding: 0px;
                border: none;
                border-top-right-radius: {_radius()}px;
                border-bottom-left-radius: {_radius()}px;
                border-bottom-right-radius: {_radius()}px;
            }}
            QTabWidget QStackedWidget {{
                background: white;
                border: none;
                margin: 0px;
                padding: 0px;
                border-top-right-radius: {_radius()}px;
                border-bottom-left-radius: {_radius()}px;
                border-bottom-right-radius: {_radius()}px;
            }}
            QTabWidget {{
                margin: 0px;
                padding: 0px;
            }}
            QTabWidget::tab-bar {{
                margin: 0px;
                padding: 0px;
            }}
            QTabBar::tab {{
                background: {BG};
                color: {TEXT};
                border: none;
                border-top-left-radius: {_radius()}px;
                border-top-right-radius: {_radius()}px;
                min-width: {scaled_px(82, minimum=68)}px;
                height: {scaled_px(30, minimum=24)}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(PADDING_H)}px;
                margin-right: {scaled_px(2)}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QTabBar::tab:selected {{
                background: white;
                color: {TEXT};
            }}
            QTabBar::tab:!selected:hover {{
                background: {ACCENT};
                color: white;
            }}
            """
        )
        add_fluent_shadow(self, blur=10, alpha=50, offset=2,
                          silhouette=_tab_widget_silhouette)


class FluentSwitch(QtWidgets.QAbstractButton):
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFont(QtGui.QFont(FONT, fluent_font_size()))
        self._offset = float(scaled_px(3, minimum=2))
        self._animating = False
        self._anim = QtCore.QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.stateChanged.connect(self._on_anim_state_changed)
        self.toggled.connect(self._start_animation)
        self.setMinimumSize(scaled_px(126, minimum=96), scaled_px(30, minimum=24))

    def sizeHint(self) -> QtCore.QSize:
        text_w = fluent_text_width(QtGui.QFontMetrics(self.font()), self.text())
        return QtCore.QSize(max(self.minimumWidth(), scaled_px(60) + text_w), self.minimumHeight())

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        track_w = scaled_px(60, minimum=48)
        track_h = min(self.height(), scaled_px(30, minimum=24))
        y = int((self.height() - track_h) / 2)
        track_color = ACCENT if self.isChecked() and self.isEnabled() else (PLACEHOLDER if self.isEnabled() else BG)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(track_color)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(0, y, track_w, track_h, track_h / 2, track_h / 2)

        margin = scaled_px(3, minimum=2)
        thumb_d = max(1, track_h - margin * 2)
        offset = self._offset if self._animating else self._checked_offset(track_w, thumb_d, margin)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#FFFFFF")))
        painter.drawEllipse(int(offset), y + margin, thumb_d, thumb_d)

        if self.text():
            painter.setPen(QtGui.QColor(TEXT))
            text_rect = QtCore.QRect(track_w + scaled_px(8), 0, max(0, self.width() - track_w - scaled_px(8)), self.height())
            painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.text())
        painter.end()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)

    def _checked_offset(self, track_w: int | None = None, thumb_d: int | None = None, margin: int | None = None) -> float:
        track_w = scaled_px(60, minimum=48) if track_w is None else int(track_w)
        track_h = min(self.height() or scaled_px(30, minimum=24), scaled_px(30, minimum=24))
        margin = scaled_px(3, minimum=2) if margin is None else int(margin)
        thumb_d = max(1, track_h - margin * 2) if thumb_d is None else int(thumb_d)
        return float(track_w - thumb_d - margin if self.isChecked() else margin)

    def _start_animation(self, checked: bool) -> None:
        del checked
        end_pos = self._checked_offset()
        if not self.isEnabled() or self.width() <= 0 or self.height() <= 0:
            self._anim.stop()
            self._offset = end_pos
            self.update()
            return
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(end_pos)
        self._anim.start()

    def _on_anim_state_changed(self, new_state, _old_state) -> None:
        self._animating = new_state == QtCore.QAbstractAnimation.Running
        if not self._animating:
            self._offset = self._checked_offset()
            self.update()

    def getOffset(self) -> float:
        return float(self._offset)

    def setOffset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = QtCore.pyqtProperty(float, fget=getOffset, fset=setOffset)


def fluent_spinbox_stylesheet(selector: str) -> str:
    return f"""
    {selector} {{
        background: white;
        border: 1px solid {PLACEHOLDER};
        border-radius: {_radius()}px;
        padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
        font: {fluent_font_size()}pt "{FONT}";
        color: {TEXT};
    }}
    {selector}:focus {{ border: 1px solid {ACCENT}; }}
    {selector}:disabled {{ background: {BG}; color: {PLACEHOLDER}; }}
    {selector}::up-button, {selector}::down-button {{
        subcontrol-origin: border;
        width: {scaled_px(COMBO_WIDTH)}px;
        border: none;
        background-color: {ACCENT};
    }}
    {selector}::up-button {{
        subcontrol-position: top right;
        border-top-right-radius: {_radius()}px;
    }}
    {selector}::down-button {{
        subcontrol-position: bottom right;
        border-bottom-right-radius: {_radius()}px;
    }}
    {selector}::up-button:hover, {selector}::down-button:hover {{
        background-color: {HOVER};
    }}
    {selector}::up-arrow, {selector}::down-arrow {{ image: none; }}
    """


class FluentSpinBox(QtWidgets.QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.setMinimumHeight(scaled_px(30, minimum=22))
        # Left-align with zero internal text margins so the text's left inset is
        # exactly the stylesheet ``padding`` (EDIT_PADDING_H) -- identical to a
        # plain FluentLineEdit / FluentScanLineEdit.  Center alignment made the
        # left gap content-dependent and looked narrower than the line edits.
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lineEdit().setTextMargins(0, 0, 0, 0)
        self.setStyleSheet(fluent_spinbox_stylesheet("QSpinBox"))


class FluentInputDialog(QtWidgets.QDialog):
    """Small Confocal-style numeric input dialog used by step editors."""

    def __init__(self, prompt: str, default: float, parent=None):
        super().__init__(parent, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self.setFont(QtGui.QFont(FONT, fluent_font_size()))
        self.setStyleSheet("QDialog { background: white; }")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(scaled_px(12), scaled_px(12), scaled_px(12), scaled_px(12))
        layout.setSpacing(scaled_px(8, minimum=5))

        layout.addWidget(FluentLabel(prompt, self))
        self._edit = FluentLineEdit(str(default), self)
        layout.addWidget(self._edit)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        ok = FluentButton("OK", self)
        cancel = FluentButton("Cancel", self)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)

    def getValue(self):
        if self.exec_() == QtWidgets.QDialog.Accepted:
            try:
                return float(self._edit.text()), True
            except ValueError:
                return None, False
        return None, False


class FluentDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """Confocal_GUIv2-style double spinbox with an inline step editor."""

    def __init__(self, length=5, allow_minus: bool = False, parent=None):
        if isinstance(length, QtWidgets.QWidget) and parent is None:
            parent = length
            length = 5
        base_class = type(self).mro()[1]
        if "Confocal_GUIv2" in getattr(base_class, "__module__", ""):
            super().__init__(length=length, allow_minus=allow_minus, parent=parent)
            self.setMinimumHeight(scaled_px(30, minimum=22))
            return
        super().__init__(parent)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.setMinimumHeight(scaled_px(30, minimum=22))
        # See FluentSpinBox: left-align + zero text margins for left-padding
        # parity with the line edits.
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lineEdit().setTextMargins(0, 0, 0, 0)
        self.setStyleSheet(fluent_spinbox_stylesheet("QDoubleSpinBox"))

        self._step_btn = QtWidgets.QToolButton(self)
        self._step_btn.setText(".")
        self._step_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._step_btn.setStyleSheet(
            f"""
            QToolButton {{
                background: {ACCENT};
                color: white;
                border: none;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QToolButton:hover {{
                background: {HOVER};
            }}
            """
        )
        self._step_btn.clicked.connect(self._on_edit_step)

        self.length = max(int(length), 5)
        if allow_minus:
            self.setRange(1 - 10 ** (self.length - 1), 10**self.length - 1)
        else:
            self.setRange(10 ** -(self.length - 2), 10**self.length - 1)
        self.setSingleStep(1)
        self.setDecimals(self.length - 2)
        self.lineEdit().setMaxLength(self.length)

    def setValue(self, value: float) -> None:
        rounded = float(_confocal_float2str(value, length=self.length))
        super().setValue(rounded)

    def setSingleStep(self, step: float) -> None:
        rounded = float(_confocal_float2str(step, length=self.length))
        super().setSingleStep(rounded)

    def stepBy(self, steps: int) -> None:
        current = self.value()
        step = self.singleStep()
        target = float(_confocal_float2str(current + steps * step, length=self.length))
        if self.minimum() <= target <= self.maximum():
            super(FluentDoubleSpinBox, self).setValue(target)

    def textFromValue(self, value: float) -> str:
        return _confocal_float2str(value, length=self.length)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        btn_w = scaled_px(COMBO_WIDTH)
        step_w = scaled_px(STEP_WIDTH)
        self._step_btn.setGeometry(self.width() - btn_w - step_w, 0, step_w, self.height())

    def _on_edit_step(self) -> None:
        current = self.singleStep()
        dialog = FluentInputDialog("Edit step", current, self)
        dialog._edit.setMaxLength(self.lineEdit().maxLength())
        value, ok = dialog.getValue()
        if ok and value is not None:
            self.setSingleStep(value)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        size = scaled_px(COMBO_TRI_SIZE)
        btn_w = scaled_px(COMBO_WIDTH)
        cx = self.width() - btn_w / 2
        height = self.height()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#FFFFFF")))

        y_up = height * 0.25
        up_points = [
            QtCore.QPoint(int(cx - size / 2), int(y_up + size / 4)),
            QtCore.QPoint(int(cx + size / 2), int(y_up + size / 4)),
            QtCore.QPoint(int(cx), int(y_up - size / 4)),
        ]
        painter.drawPolygon(QtGui.QPolygon(up_points))

        y_down = height * 0.75
        down_points = [
            QtCore.QPoint(int(cx - size / 2), int(y_down - size / 4)),
            QtCore.QPoint(int(cx + size / 2), int(y_down - size / 4)),
            QtCore.QPoint(int(cx), int(y_down + size / 4)),
        ]
        painter.drawPolygon(QtGui.QPolygon(down_points))
        painter.end()


class FluentCheckBox(QtWidgets.QCheckBox):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self.setStyleSheet(
            f"""
            QCheckBox {{
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
                spacing: {scaled_px(6)}px;
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: {scaled_px(16)}px;
                height: {scaled_px(16)}px;
                border-radius: {scaled_px(8)}px;
                border: 1px solid {PLACEHOLDER};
                background: white;
            }}
            QCheckBox::indicator:checked {{
                background: {ACCENT};
                border: 1px solid {ACCENT};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {HOVER};
            }}
            QCheckBox:disabled {{ color: {PLACEHOLDER}; }}
            QCheckBox::indicator:disabled {{ border: 1px solid {BG}; background: {BG}; }}
            QCheckBox::indicator:checked:disabled {{ background: {PLACEHOLDER}; border: 1px solid {PLACEHOLDER}; }}
            """
        )


class FluentScrollArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet(
            f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollArea > QWidget > QWidget {{ background: transparent; }}
            {fluent_scrollbar_stylesheet("QScrollBar")}
            """
        )


def apply_fluent_scrollbars(widget: "QtWidgets.QAbstractScrollArea") -> None:
    """Give any scroll-area-based widget (QPlainTextEdit, QTextEdit, QTableView, ...) the
    SAME Fluent scrollbar look that FluentScrollArea uses, sourced from the one shared
    ``fluent_scrollbar_stylesheet``.  Appends to (never clobbers) the widget's existing
    stylesheet, so its own background/border/padding rules are preserved.  Use this when
    the widget cannot be a FluentScrollArea (e.g. a QPlainTextEdit, which IS already a
    scroll area and must not be double-nested)."""

    existing = widget.styleSheet()
    widget.setStyleSheet((existing + "\n" if existing else "") + fluent_scrollbar_stylesheet("QScrollBar"))


class FluentWindow(FramelessWindow):
    """Frameless Confocal-style wrapper for PyQt frontends.

    The original Confocal GUI uses ``qframelesswindow`` with a 32 px custom
    titlebar.  This wrapper keeps that shape when the optional package is
    installed and falls back to a regular QWidget otherwise.
    """

    hidden = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        widget: QtWidgets.QWidget | None = None,
        widget_class: type | None = None,
        widget_kwargs: dict | None = None,
        title: str = "",
        hide_on_close: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._hide_on_close = bool(hide_on_close)
        self.setObjectName("FluentWindow")
        self.setStyleSheet(f"QWidget#FluentWindow {{ background: {BG}; }}")

        if StandardTitleBar is not None:
            title_bar = StandardTitleBar(self)
            title_bar.setTitle(title)
            title_bar.iconLabel.setFixedSize(0, 0)
            title_bar.titleLabel.setStyleSheet(
                f"""
                QLabel {{
                    background: transparent;
                    color: {TEXT};
                    font: {fluent_font_size()}pt "{FONT}";
                    padding: 0 {scaled_px(4)}px;
                }}
                """
            )
            self.setTitleBar(title_bar)
            top_margin = scaled_px(32)
        else:
            self.setWindowTitle(title)
            top_margin = 0

        if widget is not None:
            self.loaded = widget
            self.loaded.setParent(self)
        elif widget_class is not None:
            kwargs = dict(widget_kwargs or {})
            kwargs.setdefault("parent", self)
            self.loaded = widget_class(**kwargs)
        else:
            raise ValueError("FluentWindow needs widget or widget_class.")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, top_margin, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.loaded)
        if StandardTitleBar is not None:
            self.titleBar.raise_()
        self.loaded.adjustSize()
        self.resize(max(scaled_px(900, minimum=680), self.loaded.width()), self.loaded.height() + top_margin)

    def closeEvent(self, event):
        self.hidden.emit()
        if self._hide_on_close:
            event.ignore()
            self.hide()
            return
        super().closeEvent(event)

    def hideEvent(self, event):
        self.hidden.emit()
        super().hideEvent(event)


def run_fluent_window(
    *,
    widget_class: type | None = None,
    widget_kwargs: dict | None = None,
    widget: QtWidgets.QWidget | None = None,
    title: str = "",
    in_GUI: bool = False,
    window_handle: FluentWindow | None = None,
) -> FluentWindow:
    """Open a Confocal-style Fluent window and keep the QApplication alive."""

    app = ensure_qt_app()
    window = window_handle or FluentWindow(
        widget=widget,
        widget_class=widget_class,
        widget_kwargs=widget_kwargs,
        title=title,
        hide_on_close=not in_GUI,
    )

    screen = app.primaryScreen()
    if screen is not None:
        screen_geo = screen.availableGeometry()
        frame = window.frameGeometry()
        frame.moveCenter(screen_geo.center())
        window.move(frame.topLeft())
    window.show()

    if in_GUI:
        return window

    loop = QtCore.QEventLoop()
    window.hidden.connect(loop.quit)
    loop.exec_()
    return window


# ---------------------------------------------------------------------------
# Layout primitives -- structural guards against alignment / overlap / cutoff.
#
# Pages should compose these instead of hand-rolling fixed-width QHBoxLayouts.
# Labels elide and gain tooltips instead of overflowing; the label column width
# is computed from the actual text; fields share one row height so rows align.
# ---------------------------------------------------------------------------


class Metrics:
    """Scaled spacing/size tokens.  Call methods to read current pixel values."""

    @staticmethod
    def margin() -> int:
        return scaled_px(8, minimum=5)

    @staticmethod
    def gap_row() -> int:
        return scaled_px(6, minimum=4)

    @staticmethod
    def gap_item() -> int:
        return scaled_px(5, minimum=3)

    @staticmethod
    def gap_tight() -> int:
        return scaled_px(3, minimum=2)

    @staticmethod
    def row_h() -> int:
        return scaled_px(28, minimum=22)

    @staticmethod
    def dot() -> int:
        return scaled_px(15, minimum=12)


def measure_text_width(texts, *, padding: int = 16, minimum: int = 0, maximum: int | None = None) -> int:
    """Return a label-column width that fits the widest of ``texts`` at the current scale."""

    metrics = QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))
    widest = max([fluent_text_width(metrics, str(text)) for text in texts] + [0])
    width = widest + scaled_px(padding)
    if minimum:
        width = max(width, scaled_px(minimum))
    if maximum is not None:
        width = min(width, scaled_px(maximum))
    return int(width)


class ElidedLabel(QtWidgets.QLabel):
    """A label that elides with ``...`` and exposes the full text as a tooltip."""

    def __init__(self, text: str = "", parent=None, *, mode=QtCore.Qt.ElideRight, align=QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft):
        super().__init__(parent)
        self._full = str(text)
        self._mode = mode
        self.setAlignment(align)
        self.setStyleSheet(f'QLabel {{ color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; background: transparent; }}')
        self.setMinimumWidth(scaled_px(8))
        self.setText(str(text))

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self._full = str(text)
        self.setToolTip(self._full)
        self._elide()

    def text(self) -> str:  # noqa: N802 - Qt API name
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        available = max(0, self.width() - scaled_px(2))
        shown = metrics.elidedText(self._full, self._mode, available) if available > scaled_px(4) else self._full
        super().setText(shown)


class FluentScanDot(QtWidgets.QAbstractButton):
    """Small round toggle that marks a field as a scan parameter.

    Unbound: a hollow grey dot.  Bound: a filled orange dot showing its 1-based
    slot number.  Click toggles; the consumer connects ``toggled``.
    """

    def __init__(self, parent=None, *, tooltip: str = "Click to scan this field"):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._number: int | None = None
        diameter = Metrics.dot()
        self.setFixedSize(diameter, diameter)
        self.setToolTip(tooltip)

    def set_number(self, number: int | None) -> None:
        self._number = None if number is None else int(number)
        self.update()

    def number(self) -> int | None:
        return self._number

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = min(self.width(), self.height()) / 2.0 - max(1.0, scaled_px(1))
        if self.isChecked():
            painter.setBrush(QtGui.QColor(ORANGE))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(center, radius, radius)
            if self._number is not None:
                painter.setPen(QtGui.QColor("#FFFFFF"))
                font = QtGui.QFont(FONT, max(6, fluent_font_size() - 5))
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, str(self._number))
        else:
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(PLACEHOLDER), max(1, scaled_px(1))))
            painter.drawEllipse(center, radius, radius)
            painter.setBrush(QtGui.QColor(PLACEHOLDER))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(center, radius * 0.42, radius * 0.42)
        painter.end()


def mark_scan_field(widget: QtWidgets.QWidget, *, bound: bool) -> None:
    """Apply (or clear) the orange disabled look on a field bound to a scan slot."""

    widget.setProperty("zlcScanBound", bool(bound))
    widget.setEnabled(not bound)
    if bound:
        widget.setStyleSheet(
            f"""
            QLineEdit, QComboBox {{
                background: {ORANGE_TINT};
                color: {ORANGE_DARK};
                border: 1px solid {ORANGE};
                border-radius: {_radius()}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            """
        )
    else:
        widget.setStyleSheet("")


def _muted_line_style() -> str:
    """Read-only / inactive look for a field that must stay *enabled* (so its
    embedded scan dot keeps receiving clicks)."""

    return (
        f"QLineEdit {{ background: {BG}; color: {PLACEHOLDER}; "
        f"border: 1px solid {PLACEHOLDER}; border-radius: {_radius()}px; "
        f"padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px; "
        f'font: {fluent_font_size()}pt "{FONT}"; }}'
    )


class FluentScanLineEdit(FluentLineEdit):
    """A line edit with a round 'scan' toggle embedded on the right edge.

    The dot behaves like a spinbox's inline button: it sits inside the field
    (vertically centered) and stays clickable even when the field is bound.
    Binding turns the field orange + read-only and shows the slot number on the
    dot; clicking the dot emits :attr:`scanClicked`.
    """

    scanClicked = QtCore.pyqtSignal()

    def __init__(self, text: str = "", parent=None, *, tooltip: str = "Click the dot to scan this field"):
        super().__init__(text, parent)
        self._base_style = self.styleSheet()
        self._dot = FluentScanDot(self, tooltip=tooltip)
        self._dot.clicked.connect(self.scanClicked)
        self._bound = False
        self._reserve_right()

    def _dot_size(self) -> int:
        return Metrics.dot()

    def _reserve_right(self) -> None:
        # Left margin 0: the stylesheet ``padding`` (EDIT_PADDING_H) already sets
        # the left text inset, so the text/number left edge lines up with a plain
        # FluentLineEdit / spinbox.  Only reserve space on the RIGHT for the dot.
        margin = self._dot_size() + scaled_px(3)
        self.setTextMargins(0, 0, margin, 0)

    def _place_dot(self) -> None:
        diameter = self._dot_size()
        x = self.width() - diameter - scaled_px(4)
        y = (self.height() - diameter) // 2
        self._dot.setGeometry(int(x), int(y), int(diameter), int(diameter))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_dot()

    def showEvent(self, event):
        super().showEvent(event)
        self._place_dot()

    @property
    def dot(self) -> "FluentScanDot":
        return self._dot

    def set_scan_bound(self, bound: bool, number: int | None = None) -> None:
        bound = bool(bound)
        # no-op guard: re-applying the (expensive) stylesheet/dot/readonly when neither the
        # bound flag nor the badge number changed is wasted work (Qt re-polishes the style).
        if self._bound == bound and getattr(self, "_scan_number", "\x00unset") == number:
            return
        self._bound = bound
        self._scan_number = number
        self._dot.setChecked(self._bound)
        self._dot.set_number(number if self._bound else None)
        self.setReadOnly(self._bound)
        if self._bound:
            self.setStyleSheet(
                f"""
                QLineEdit {{
                    background: {ORANGE_TINT};
                    color: {ORANGE_DARK};
                    border: 1px solid {ORANGE};
                    border-radius: {_radius()}px;
                    padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                    font: {fluent_font_size()}pt "{FONT}";
                }}
                """
            )
        else:
            self.setStyleSheet(self._base_style)
        self._reserve_right()
        self.update()

    def set_editable(self, editable: bool) -> None:
        """Toggle text editability *without disabling the widget*.

        Disabling a ``QLineEdit`` also disables its child scan dot, which would
        make the dot un-clickable -- you could no longer bind/unbind a scan
        slot.  So an inactive field (e.g. a ``hold`` bus segment) instead goes
        read-only with a muted style while staying enabled, keeping the dot
        live.  The bound (orange) state owns its own appearance, so this is a
        no-op while bound.
        """

        editable = bool(editable)
        if self._bound:
            return
        if getattr(self, "_editable", None) == editable:
            return   # no-op guard: skip the readonly/stylesheet re-apply when unchanged
        self._editable = editable
        self.setReadOnly(not editable)
        self.setStyleSheet(self._base_style if editable else _muted_line_style())
        self._reserve_right()
        self.update()


class FluentLabeledField(QtWidgets.QWidget):
    """A ``label : widget [suffix]`` row with a shared, aligned height."""

    def __init__(self, label: str, widget: QtWidgets.QWidget, *, label_width: int | None = None, suffix: QtWidgets.QWidget | None = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Metrics.gap_item())
        self.label = ElidedLabel(str(label))
        self.label.setFixedHeight(Metrics.row_h())
        if label_width is not None:
            self.label.setFixedWidth(int(label_width))
        layout.addWidget(self.label)
        widget.setFixedHeight(Metrics.row_h())
        layout.addWidget(widget, 1)
        self.field = widget
        self.suffix = suffix
        if suffix is not None:
            layout.addWidget(suffix, 0, QtCore.Qt.AlignVCenter)


class FluentFormGrid(QtWidgets.QWidget):
    """A multi-column form whose first (label) column is shared and aligned.

    Every row keeps the same height and the same label-column width, so a stack
    of forms lines up without per-row fixed geometry.
    """

    def __init__(self, parent=None, *, label_width: int | None = None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(Metrics.gap_item())
        self.grid.setVerticalSpacing(Metrics.gap_row())
        self.grid.setColumnStretch(1, 1)
        self._rows = 0
        if label_width is not None:
            self.grid.setColumnMinimumWidth(0, int(label_width))

    def add_row(self, label, *widgets) -> ElidedLabel | QtWidgets.QWidget:
        cell = ElidedLabel(str(label)) if isinstance(label, str) else label
        cell.setFixedHeight(Metrics.row_h())
        self.grid.addWidget(cell, self._rows, 0)
        for column, widget in enumerate(widgets, start=1):
            widget.setFixedHeight(Metrics.row_h())
            self.grid.addWidget(widget, self._rows, column)
        self._rows += 1
        return cell


__all__ = [
    "ACCENT",
    "BG",
    "FONT",
    "FONT_SIZE",
    "GREEN",
    "GREY",
    "HINT",
    "HOVER",
    "ORANGE",
    "PLACEHOLDER",
    "RADIUS",
    "RED",
    "TEXT",
    "YELLOW",
    "FluentButton",
    "FluentCheckBox",
    "FluentComboBox",
    "FluentDoubleSpinBox",
    "FluentFrame",
    "FluentGroupBox",
    "FluentInputDialog",
    "FluentLabel",
    "FluentLineEdit",
    "FluentSpinBox",
    "FluentSwitch",
    "FluentTabWidget",
    "FloatLineEdit",
    "FluentScrollArea",
    "FluentStatusDot",
    "FluentWindow",
    "add_fluent_shadow",
    "align_to_resolution",
    "ensure_qt_app",
    "fluent_font_size",
    "fluent_scale",
    "fluent_scrollbar_stylesheet",
    "apply_fluent_scrollbars",
    "fluent_spinbox_stylesheet",
    "fluent_text_width",
    "fluent_widget_stylesheet",
    "format_compact_number",
    "run_fluent_window",
    "scaled_px",
    "set_fluent_scale",
    "status_dot_stylesheet",
    "Metrics",
    "ElidedLabel",
    "FluentScanDot",
    "FluentScanLineEdit",
    "FluentLabeledField",
    "FluentFormGrid",
    "measure_text_width",
    "mark_scan_field",
]
