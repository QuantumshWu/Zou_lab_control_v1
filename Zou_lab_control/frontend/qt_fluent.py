"""Reusable PyQt Fluent-style widgets for Zou_lab_control frontends.

The visual constants and widget shapes follow the original Confocal_GUIv2
Fluent layer: pale blue accent, Segoe UI text, white cards, small radii, and
flat 1 px DIVIDER card borders (no drop shadows -- see ``stroke_card_border``).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

try:  # Provided by the PyQt5-Frameless-Window distribution (module qframelesswindow).
    from qframelesswindow import FramelessWindow, StandardTitleBar
except ImportError:  # pragma: no cover - depends on the desktop package.
    # Degrade to a plain top-level window, but make the missing dependency LOUD:
    # a silent fallback once hid a requirements typo and shipped a frameless-less
    # GUI on the real machine.  Install "PyQt5-Frameless-Window" to get the
    # Fluent title bar.
    import warnings

    warnings.warn(
        "qframelesswindow not installed (pip install PyQt5-Frameless-Window); "
        "falling back to a plain window without the Fluent title bar.",
        RuntimeWarning,
        stacklevel=2,
    )
    FramelessWindow = QtWidgets.QWidget
    StandardTitleBar = None


ACCENT = "#77AADD"
HOVER = "#004578"
BG = "#F3F3F3"
TEXT = "#323130"
HINT = "#F0a150"
PLACEHOLDER = "#A19F9D"
# Light neutral hairline that delineates every flat fluent card (a painted 1 px border, no drop
# shadow -- shadows were costly to re-rasterise + blur on every repaint; see stroke_card_border).
DIVIDER = "#E1DFDD"
GREEN = "#7FC2AD"
RED = "#CD7380"
ORANGE = "#D69A6E"
# A scan-bound field is painted a pale orange so the saturated-orange scan dot
# (the inline spinbox-style button) stands out against it instead of vanishing.
ORANGE_TINT = "#F6E3D4"
ORANGE_DARK = "#8A4B1F"
# API-slot (a1/a2...) marker palette -- a VIOLET distinct from the orange scan colour, so
# the two slot kinds never read alike.  Unlike a scan slot, an API slot does NOT hide the
# field value, so there is no tint/fill: only a violet dot + a thin violet border mark it.
API_VIOLET = "#9B86C9"
API_VIOLET_DARK = "#5A4A8A"
YELLOW = "#E5C85B"
GREY = "#A2A2A2"
RADIUS = 4
# A "card" = FluentGroupBox: it reserves a CARD_TITLE_PX-tall strip at the top for the grey
# QGroupBox::title chip, and insets its content by CARD_PAD on the L / R / bottom edges.  These
# are the card's FORMAT and live HERE in the component library (the single source) -- callers
# place content in a card and read these tokens; they never hand-pick card padding themselves.
CARD_TITLE_PX = 32
CARD_PAD = 10
FONT = "Segoe UI"
FONT_SIZE = 12
PADDING_V = 1
PADDING_H = 1
EDIT_PADDING_H = 4
# WINDOW_PAD: the ONE window content-padding / spacing unit.  EVERY GUI insets its window body by
# ``window_pad(1)`` on all FOUR sides (so the padding to every window edge is identical and equal
# left/right/top/bottom), and inter-card gaps are ``window_pad`` multiples (0.5x / 1x / 2x) -- so all
# GUIs share one rhythm.  A card's left edge therefore lines up with the window-title text, because
# FluentWindow pins the title to this same column (see ``_position_window_title``).  Change the rhythm
# in ONE place here; never hand-pick a window margin or a top-level card gap in a GUI module.
WINDOW_PAD = 14


def window_pad(mult: float = 1.0) -> int:
    """The window padding / card-gap at ``mult`` x the single :data:`WINDOW_PAD` unit, display-scaled."""
    return scaled_px(round(WINDOW_PAD * mult))


# The window title pins to the same left column as the body content (one shared left edge).
TITLE_LEFT_INSET = WINDOW_PAD
COMBO_WIDTH = 16
COMBO_TRI_SIZE = 8
STEP_WIDTH = 6

_QT_APP = None
_QT_LOOP_ENABLED = False
_FLUENT_SCALE = 1.0
# The fluent scale is clamped to a usable band: below the min the controls become
# unreadable, above the max they waste space.  ONE source -- set_fluent_scale and
# the scale-sharing contract test both read these (the test must not re-type the
# band, or the two could silently disagree).
FLUENT_SCALE_MIN = 0.72
FLUENT_SCALE_MAX = 1.25
# Matches one float token; used by align_to_resolution to snap numbers inside a value.
_FLOAT_TOKEN_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def fluent_scale() -> float:
    return _FLUENT_SCALE


# The fluent design-basis window: the automatic scale fits this logical size
# into the primary screen's available geometry.  EVERY GUI window must resolve
# its automatic scale through this ONE rule -- two windows on the same screen
# must end up with identical control sizes (pulse editor == task console).
_AUTO_SCALE_BASIS = (1280, 790)
_AUTO_SCALE_MARGIN = (48, 88)

# Initial-window sizing for the two top-level GUIs (task console + pulse editor).
# Both windows fit themselves to the screen with the SAME rule; it used to be
# copied verbatim into each file (every literal duplicated), so a tweak in one
# silently diverged the other.  These are the ONE source -- see
# screen_fit_window_size below.  (height, width) where a pair is (w, h).
_WINDOW_FALLBACK_PX = (1280, 760)       # size when no screen can be queried
_WINDOW_FALLBACK_MIN_PX = (960, 620)    # scaled_px floor for the fallback
_WINDOW_MIN_PX = (980, 640)             # minimum window before clamping to screen
_WINDOW_MIN_FLOOR_PX = (820, 560)       # scaled_px floor for that minimum
_WINDOW_MARGIN_PX = (40, 48)            # (w, h) breathing room left around the screen
_WINDOW_MARGIN_FLOOR_PX = (28, 32)      # scaled_px floor for those margins
_WINDOW_TITLEBAR_PX = 36                # title-bar height allowance
_WINDOW_TITLEBAR_FLOOR_PX = 28          # scaled_px floor for the title bar
_WINDOW_MAX_FLOOR_PX = (360, 320)       # absolute (w, h) floor for the screen-fit max

#: The fraction of the primary screen each top-level GUI (task console + pulse editor) opens at.
#: ONE editable knob both windows read through ``screen_fit_window_size``, so they open the SAME
#: size on the same screen (it used to be 0.90 in the pulse editor but a stray 0.84 in the console).
#: User-tunable, like the other uppercase constants in this module.
WINDOW_SCREEN_FRACTION = 0.90


def resolve_fluent_auto_scale(app: QtWidgets.QApplication | None = None) -> float:
    """The shared automatic fluent scale for the current primary screen."""

    app = app or QtWidgets.QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return 1.0
    available = screen.availableGeometry()
    basis_w, basis_h = _AUTO_SCALE_BASIS
    margin_w, margin_h = _AUTO_SCALE_MARGIN
    return min(
        1.0,
        max(0.1, (available.width() - margin_w) / basis_w),
        max(0.1, (available.height() - margin_h) / basis_h),
    )


def set_fluent_scale(scale: float | None = None) -> float:
    """Set the PROCESS-GLOBAL scale used by subsequently created Fluent widgets.

    ``None`` means AUTOMATIC: resolve from the primary screen via
    :func:`resolve_fluent_auto_scale` (never a silent 1.0 -- a fixed fallback
    is how two GUIs ended up with different control sizes on the same screen).

    The fluent scale is ONE process-wide value BY DESIGN: every GUI window on a
    screen must agree on control sizes (the scale-sharing contract), so the
    per-window ``scale=`` parameters of the ``show_*`` launchers all write THIS
    value -- there is no per-window scale.  Changing it while FluentWindows are
    already open therefore also retunes the app font and every widget those
    windows create from now on; that conflict is logged as a warning.  Requesting
    the value already in force (e.g. the automatic ``scale=None`` path resolving
    to the same screen fit) stays silent."""

    global _FLUENT_SCALE
    if scale is None:
        scale = resolve_fluent_auto_scale()
    new_scale = max(FLUENT_SCALE_MIN, min(FLUENT_SCALE_MAX, float(scale)))
    app = QtWidgets.QApplication.instance()
    if new_scale != _FLUENT_SCALE and app is not None:
        live_windows = [w for w in app.topLevelWidgets() if isinstance(w, FluentWindow)]
        if live_windows:
            logging.getLogger(__name__).warning(
                "set_fluent_scale(%.3g): the fluent scale is process-global; changing it from %.3g "
                "with %d FluentWindow(s) open also resizes the app font and every widget those "
                "windows create from now on.",
                new_scale, _FLUENT_SCALE, len(live_windows))
    _FLUENT_SCALE = new_scale
    if app is not None:
        app.setFont(QtGui.QFont(FONT, fluent_font_size()))
    return _FLUENT_SCALE


def scaled_px(value: int | float, *, minimum: int = 1) -> int:
    return max(int(minimum), int(round(float(value) * _FLUENT_SCALE)))


def screen_fit_window_size(window_ratio: float) -> QtCore.QSize:
    """Initial window size for a top-level GUI, fit to the primary screen.

    The task console and the pulse editor share this ONE rule (it was duplicated
    verbatim in both, every literal copied -- so a change in one silently diverged
    the other, the same failure class as the scale rule).  ``window_ratio`` is the
    fraction of the screen the window aims for; the result is clamped to a usable
    minimum and to what the screen can actually show."""

    app = QtWidgets.QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return QtCore.QSize(
            scaled_px(_WINDOW_FALLBACK_PX[0], minimum=_WINDOW_FALLBACK_MIN_PX[0]),
            scaled_px(_WINDOW_FALLBACK_PX[1], minimum=_WINDOW_FALLBACK_MIN_PX[1]))
    available = screen.availableGeometry()
    titlebar_allowance = scaled_px(_WINDOW_TITLEBAR_PX, minimum=_WINDOW_TITLEBAR_FLOOR_PX)
    margin_w = scaled_px(_WINDOW_MARGIN_PX[0], minimum=_WINDOW_MARGIN_FLOOR_PX[0])
    margin_h = scaled_px(_WINDOW_MARGIN_PX[1], minimum=_WINDOW_MARGIN_FLOOR_PX[1])
    max_w = max(_WINDOW_MAX_FLOOR_PX[0], available.width() - margin_w)
    max_h = max(_WINDOW_MAX_FLOOR_PX[1], available.height() - titlebar_allowance - margin_h)
    min_w = min(scaled_px(_WINDOW_MIN_PX[0], minimum=_WINDOW_MIN_FLOOR_PX[0]), max_w)
    min_h = min(scaled_px(_WINDOW_MIN_PX[1], minimum=_WINDOW_MIN_FLOOR_PX[1]), max_h)
    desired_w = min(max_w, int(available.width() * window_ratio))
    desired_h = min(max_h, int(available.height() * window_ratio) - titlebar_allowance)
    return QtCore.QSize(max(min_w, desired_w), max(min_h, desired_h))


def center_window_on_primary_screen(window: QtWidgets.QWidget, app: QtWidgets.QApplication | None = None) -> None:
    """Centre a top-level GUI window on the primary screen's available area.

    The pulse editor and the task console share this ONE rule -- it used to live only in the
    pulse launcher, so the console opened wherever the window manager dropped it and the two
    GUIs looked inconsistent.  Single-sourced here next to ``screen_fit_window_size`` (which
    sizes the same windows) so the open POSITION cannot silently diverge from the open SIZE."""
    app = app or QtWidgets.QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return
    available = screen.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(available.center())
    window.move(frame.topLeft())


def retain_window(window: QtWidgets.QWidget, *extras) -> None:
    """Keep a launched top-level GUI ``window`` alive on the QApplication, pruned when it dies.

    The ``show_*`` launchers (task console / pulse editor / figure viewer) must hold a Python
    reference to the window they open -- a notebook cell often drops the return value, and with
    no reference the window would be garbage-collected mid-display.  Each launcher used to
    append to its OWN app-level list and never remove anything, so every reopened standalone
    window (with its canvases and figures) was retained forever.  This is the ONE registry
    (``app._zlc_retained_windows``) with the ONE pruning rule (the destroyed-connect pattern of
    Confocal-GUIv2's ``_on_child_closed``):

    * ``window.destroyed`` always removes the entry;
    * a genuine close (the ``closed`` signal, when the window has one) also removes it -- but
      ONLY for windows that really close on X.  ``hide_on_close`` windows hide instead and MUST
      stay retained, so a later session call restores the SAME interface.

    ``extras`` are for objects the caller needs kept alive that the window does NOT own; they
    ride the window's lifetime.  A widget the window parents (``FluentWindow(widget=...)``
    re-parents it) needs no extra retention -- do not double-retain children."""

    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("retain_window needs a QApplication (call ensure_qt_app first).")
    registry = getattr(app, "_zlc_retained_windows", None)
    if registry is None:
        registry = []
        app._zlc_retained_windows = registry
    if extras:
        window._zlc_retained_extras = tuple(extras)   # live and die with the window
    registry.append(window)

    def _prune(*_args) -> None:
        try:
            registry.remove(window)
        except ValueError:
            pass  # already pruned (e.g. destroyed after a genuine close)

    window.destroyed.connect(_prune)
    closed = getattr(window, "closed", None)
    if closed is not None and not getattr(window, "_hide_on_close", False):
        closed.connect(_prune)


def fluent_font_size() -> int:
    return max(8, int(round(FONT_SIZE * _FLUENT_SCALE)))


def fluent_text_width(metrics: QtGui.QFontMetrics, text: str) -> int:
    """Return text width on old and new PyQt5 builds."""

    if hasattr(metrics, "horizontalAdvance"):
        return int(metrics.horizontalAdvance(text))
    return int(metrics.width(text))


def _radius() -> int:
    return scaled_px(RADIUS)


def _popup_gap() -> int:
    """The few-px OUTER gap a Fluent popup sits OFF the control that anchored it -- the ONE source for
    every below-the-anchor Fluent surface: the FluentComboBox drop-down, the Setting FluentPopup, the
    FluentTabWidget overflow menu, the FluentLineEdit context menu.  Sharing this is what makes the
    ``...`` overflow list and a right-click menu open with the SAME breathing room as a combo drop-down,
    instead of one flush against its anchor and another floating off it."""
    return scaled_px(4, minimum=3)


def _enable_ipython_qt_loop() -> None:
    """Under IPython/Jupyter, install the Qt event-loop hook (the ``%gui qt`` integration) so a
    Qt window opened from a cell keeps PROCESSING events -- it paints and stays responsive
    instead of appearing frozen -- because the kernel pumps Qt between cells.  Without this a
    ``show_pulse_gui()`` / ``show_task_console()`` / ``exp.pulse_gui()`` window just hangs blank
    in a notebook.  No-op outside IPython (a script or pytest runs its own loop / is headless),
    and done once per process.  ``%matplotlib widget`` is independent of this -- they coexist."""

    global _QT_LOOP_ENABLED
    if _QT_LOOP_ENABLED:
        return
    try:
        from IPython import get_ipython
    except Exception:        # pragma: no cover - IPython not installed (plain script)
        return
    ip = get_ipython()
    if ip is None:           # not in an interactive shell (script / pytest)
        return
    try:
        ip.run_line_magic("gui", "qt")     # same hook a user typing %gui qt would install
        _QT_LOOP_ENABLED = True
    except Exception:        # pragma: no cover - headless kernel / no display
        pass


def ensure_qt_app() -> QtWidgets.QApplication:
    """Return a QApplication, creating and keeping one alive when needed.

    Also enables the IPython Qt event loop (``%gui qt``) when running in a notebook, so a window
    opened from a cell stays live instead of hanging blank."""

    global _QT_APP
    app = QtWidgets.QApplication.instance()
    if app is not None:
        _QT_APP = app
        _enable_ipython_qt_loop()
        return app
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # Silence the harmless Windows "Unable to open default EUDC font" Qt warning.
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
    _QT_APP = QtWidgets.QApplication(sys.argv)
    _QT_APP.setFont(QtGui.QFont(FONT, fluent_font_size()))
    _enable_ipython_qt_loop()
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


# Fluent cards are delineated by a flat 1 px ``DIVIDER`` border, NOT a drop shadow.  A
# ``QGraphicsDropShadowEffect`` re-rasterised the whole widget subtree + re-ran a Gaussian blur on EVERY
# paint (~250 ms of a 35-site grid card's cold load at 3x dpr); a 1 px border costs nothing and reads the
# same "raised card" boundary.  Each card owns its edge: ``FluentGroupBox`` paints a CONTINUOUS border
# UNDER its title pill (so the pill never notches it); ``FluentTabWidget`` has NO border at all -- a plain
# white rounded card (any border read as boxing the pivot tabs, or as a line between the tabs and the
# content below); ``FluentFrame`` paints its edge via this one helper (a ``QFrame`` stylesheet border
# would cascade onto child QFrames -- painting does not), exactly like ``FluentPopup`` strokes its own edge.
def stroke_card_border(widget: QtWidgets.QWidget, *, radius: int | None = None) -> None:
    """Stroke a 1 px antialiased ``DIVIDER`` rounded border hugging ``widget``'s edge, in a paintEvent.

    Cascade-proof (painted, not a stylesheet rule) so a container that nests its own widget type does
    not border every child.  Call from the widget's ``paintEvent`` AFTER its body is painted."""
    r = float(_radius() if radius is None else radius)
    painter = QtGui.QPainter(widget)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    pen = QtGui.QPen(QtGui.QColor(DIVIDER))
    pen.setWidthF(1.0)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    # inset by half the pen width so the 1 px stroke sits fully inside the widget, unclipped
    rect = QtCore.QRectF(widget.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
    painter.drawRoundedRect(rect, r, r)
    painter.end()


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


#: SI engineering prefixes, matching Confocal_GUIv2.helper.float2str_eng's ``si_map`` (one source of
#: the n/µ/m/k/M/G/T substitution).  Read-only value DISPLAYS (a device read-back, a range label)
#: engineering-scale through :func:`eng_mantissa_prefix`; the editable spin box stays plain decimal
#: (confocal does NOT SI-scale its editor -- so does ``FluentDoubleSpinBox``).
_SI_PREFIX = {-9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}


def eng_mantissa_prefix(value: float, *, length: int = 6) -> tuple[str, str]:
    """Split ``value`` into ``(mantissa_string, si_prefix)`` for a READ-ONLY engineering display,
    matching Confocal_GUIv2's ``float2str_eng``: a plain decimal when ``0.001 < |x| < 1000`` (prefix
    ``""``), otherwise the mantissa in ``[1, 1000)`` times an SI prefix (``6.83e9 -> ("6.83", "G")``,
    ``5e-3 -> ("5", "m")``).  Returned split (not glued) so a caller can attach the prefix to the UNIT
    -- ``f"{mant} {prefix}{unit}"`` reads ``"6.83 GHz"``.  ``length`` caps the mantissa width."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value), ""
    if not math.isfinite(v):
        return ("nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")), ""
    if v == 0.0:
        return "0", ""
    length = max(int(length), 5)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if 0.001 < a < 1000.0:                       # the human-comfortable band: plain decimal, no prefix
        mant, exp = a, 0
    else:
        exp = int(math.floor(math.log10(a) / 3.0) * 3)
        exp = max(-9, min(12, exp))              # clamp to the prefixes we actually have
        mant = a / (10.0 ** exp)
    dec = max(0, length - len(sign) - len(str(int(mant))) - 1)   # room for sign + int part + '.'
    text = f"{mant:.{dec}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{sign}{text}", _SI_PREFIX[exp]


def float2str_eng(value: float, *, length: int = 6) -> str:
    """Engineering-scaled string (mantissa glued to its SI prefix): ``6.83e9 -> "6.83G"`` -- the
    prefix-attached-to-number form for a range label; :func:`eng_mantissa_prefix` keeps them split
    when the prefix must ride the unit instead."""
    mant, prefix = eng_mantissa_prefix(value, length=length)
    return f"{mant}{prefix}"


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
    def __init__(self, parent=None, *, bordered: bool = True, round: tuple[str, ...] = ("NW", "NE", "SE", "SW")):
        super().__init__(parent)
        # ``bordered`` = a delineated flat card (1 px DIVIDER edge); False = a plain, edgeless region
        # (a header / button strip that must not read as a boxed card).  The edge is PAINTED, never a
        # stylesheet ``QFrame { border }`` rule -- that would cascade onto every child QFrame.
        self._bordered = bool(bordered)
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

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        if self._bordered:
            stroke_card_border(self)


class FluentPopup(QtWidgets.QFrame):
    """A top-level ``Qt.Popup`` with a CLEAN rounded white-card chrome.

    A rounded card on a popup must NOT be done with a stylesheet ``border-radius``
    on an opaque top-level window: the square window stays opaque behind the arc,
    so the corner triangles outside the radius show a square white nub past the
    rounded border (and Windows adds a native popup shadow) -- that nub reads as
    the stray line at the bottom corners.  WA_TranslucentBackground alone is not
    enough either: it stops the stylesheet background painting at all, leaving the
    card see-through.

    So this paints the card itself: a TRANSLUCENT, frameless popup whose
    paintEvent strokes ONE antialiased rounded rect (fill + 1 px border).  The
    corners are truly rounded with nothing outside them, and because the border is
    painted (not a stylesheet rule) it can never cascade onto child widgets."""

    def __init__(self, parent=None, *, radius: float | None = None,
                 border: str = DIVIDER, fill: str = "white"):
        super().__init__(parent, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint
                         | QtCore.Qt.NoDropShadowWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._radius = float(_radius() if radius is None else radius)
        self._border = QtGui.QColor(border)
        self._fill = QtGui.QColor(fill)
        # A Qt.Popup auto-closes on ANY outside mouse PRESS -- including the press on
        # the very button that toggles it.  An anchor button therefore needs to know
        # the popup was just auto-dismissed so its release does not RE-open it; set
        # this hook to be notified on every hide (auto or explicit).
        self._on_hidden = None

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if callable(self._on_hidden):
            self._on_hidden()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(self._border)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(self._fill)
        # inset by the half pen width so the 1 px stroke hugs the edge unclipped
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(rect, self._radius, self._radius)


class FluentGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        # Flat card delineated by a CONTINUOUS 1 px DIVIDER border painted in ``paintEvent`` (below) --
        # NOT Qt's own ``QGroupBox`` frame, which cuts a NOTCH in the top border where the title sits
        # (the "border broken at the rounded title corner" look).  The stylesheet keeps ONLY a
        # transparent body + the grey title pill; the white body and the unbroken border are painted
        # UNDER the pill, so the pill draws on top and covers only its own segment of the full border.
        # ``padding-top`` still reserves the title strip for child layout.
        self.setStyleSheet(
            f"""
            QGroupBox {{
                background: transparent;
                border: none;
                border-radius: {_radius()}px;
                margin-top: 0px;
                padding-top: {scaled_px(CARD_TITLE_PX)}px;
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

    def set_outline(self, color: str | None) -> None:
        """Select (color) or clear (None) this card: its continuous edge border switches to a 2 px
        accent (from the resting 1 px DIVIDER), painted in paintEvent.  Never via stylesheet -- an
        unscoped ``border:`` cascades to every CHILD widget, so each inner checkbox/lineedit grew its
        own box."""
        value = str(color) if color else None
        if getattr(self, "_zlc_outline", None) != value:
            self._zlc_outline = value
            self.update()

    def paintEvent(self, event) -> None:
        # Paint the white body + a CONTINUOUS rounded border, THEN let Qt draw the grey title pill +
        # text on top (super) -- so the pill covers its segment of the border, never the reverse, and
        # the border is unbroken all the way round.  1 px DIVIDER at rest; 2 px accent when selected.
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        radius = float(_radius())
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("white"))
        painter.drawRoundedRect(QtCore.QRectF(self.rect()), radius, radius)
        outline = getattr(self, "_zlc_outline", None)
        width = 2.0 if outline else 1.0
        pen = QtGui.QPen(QtGui.QColor(outline or DIVIDER))
        pen.setWidthF(width)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        half = width / 2.0
        painter.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(half, half, -half, -half), radius, radius)
        painter.end()
        super().paintEvent(event)


class FluentButton(QtWidgets.QPushButton):
    def __init__(self, text: str = "", parent=None, *, color: str = ACCENT):
        super().__init__(text, parent)
        self._current_bg = None
        # Confocal-style dirty state (see set_dirty): the CLEAN base colour is
        # remembered so a dirty colour never leaks into it, and toggling the
        # trailing '*' is width-stable.
        self._dirty = False
        self._dirty_color = None   # remembered so a dirty colour can always be reverted
        self._base_color = QtGui.QColor(color).name(QtGui.QColor.HexRgb)
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
        # Remember the CLEAN base colour only while not dirty, so set_dirty's
        # transient dirty colour can always be reverted.
        if not self._dirty:
            self._base_color = bg
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

    def set_dirty(self, dirty: bool = True, *, dirty_color: str | None = None) -> None:
        """Confocal-style dirty indicator, reusable on any FluentButton.

        A trailing ``*`` means "pressing this would apply / recompute something
        new" (exactly the ``On Pulse*`` / ``Save*`` convention used by the editor
        title bar).  By default ONLY the star toggles and the base colour is left
        unchanged (the action-button rule -- e.g. ``On Pulse`` stays GREEN).  Pass
        ``dirty_color`` for the save-button rule (e.g. YELLOW while dirty, base
        colour when clean).  Idempotent and width-stable -- the base label/colour
        are remembered so repeated calls never accumulate stars or lose the colour."""
        dirty = bool(dirty)
        if dirty_color is not None:
            self._dirty_color = dirty_color   # persist so clean can revert it
        base = self.text()
        if base.endswith("*"):
            base = base[:-1]
        self._dirty = dirty
        QtWidgets.QPushButton.setText(self, base + ("*" if dirty else ""))
        if self._dirty_color is not None:
            self.set_color(self._dirty_color if dirty else self._base_color)

    def is_dirty(self) -> bool:
        return self._dirty


class FluentTriStateToggle(QtWidgets.QAbstractButton):
    """Capsule-style MUTUALLY-EXCLUSIVE multi-state toggle -- a faithful port of the Confocal-GUIv2
    ``TriStateToggleSwitch`` (the "on start: single / replace / parallel" capsule).  ALL option labels
    stay visible and a white thumb slides to overlay the active one; clicking cycles to the next state.
    The track is the placeholder grey, the thumb white, the labels the standard text colour (sizes /
    colours read from this module's tokens + ``scaled_px``, so it scales with DPI).

    It exposes the SAME ``currentText`` / ``setCurrentText`` / ``activated(int)`` surface as
    :class:`FluentComboBox`, so a param form's choice handler swaps a toggle for a combo with no other
    change (it only decides which to BUILD; read / write / wiring are identical)."""

    activated = QtCore.pyqtSignal(int)          # mirrors QComboBox.activated(int): a USER pick (a cycle)

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self._options = [str(o) for o in options] or [""]
        self._state = 0
        self._offset = 0.0
        self._animating = False
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFont(QtGui.QFont(FONT, fluent_font_size()))
        self._anim = QtCore.QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.stateChanged.connect(self._on_anim_state_changed)
        # A FIXED size policy pins the capsule to its sizeHint (= widest label per
        # segment): a settings row adds the control with ``addWidget(control, 1)`` so a
        # default (Minimum) horizontal policy lets the row stretch the toggle to fill the
        # whole row -- the segments would balloon far past their text.  Fixed both ways
        # keeps each segment exactly as wide as its label needs.
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumSize(self.sizeHint())

    def _seg_width(self) -> float:
        return self.width() / max(1, len(self._options))

    def sizeHint(self) -> QtCore.QSize:
        fm = QtGui.QFontMetrics(self.font())
        pad = scaled_px(18, minimum=12)
        seg = max((fluent_text_width(fm, o) for o in self._options), default=0) + pad
        return QtCore.QSize(int(seg * len(self._options)), scaled_px(30, minimum=24))

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        seg = self._seg_width()
        margin = scaled_px(3, minimum=2)
        # 1) capsule track (placeholder grey; muted when disabled)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(PLACEHOLDER if self.isEnabled() else BG)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, h / 2, h / 2)
        # 2) white thumb overlaying the active segment (animated x while sliding)
        x = self._offset if self._animating else self._state * seg
        th = max(1, h - 2 * margin)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#FFFFFF")))
        painter.drawRoundedRect(int(x + margin), int(margin), int(seg - 2 * margin), int(th), th / 2, th / 2)
        # 3) labels on top, centred per segment
        painter.setPen(QtGui.QColor(TEXT if self.isEnabled() else PLACEHOLDER))
        for i, label in enumerate(self._options):
            painter.drawText(QtCore.QRectF(i * seg, 0, seg, h), QtCore.Qt.AlignCenter, label)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Click-to-segment: the active state is the segment the user actually clicked,
        # not the "next" one (cycling forces a user to walk past intermediate states to
        # reach the one they want).  Map the press x to a segment index, clamped in range.
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        seg = self._seg_width()
        idx = int(event.x() // seg) if seg > 0 else 0
        idx = max(0, min(idx, len(self._options) - 1))
        if idx != self._state:
            prev = self._state
            self._state = idx
            self._animate(prev, idx)
            self.activated.emit(self._state)
        event.accept()

    def _animate(self, prev_idx: int, new_idx: int) -> None:
        seg = self._seg_width()
        if not self.isEnabled() or self.width() <= 0:
            self._offset = new_idx * seg
            self.update()
            return
        self._anim.stop()
        self._anim.setStartValue(prev_idx * seg)
        self._anim.setEndValue(new_idx * seg)
        self._anim.start()

    def _on_anim_state_changed(self, new_state, _old) -> None:
        self._animating = new_state == QtCore.QAbstractAnimation.Running
        if not self._animating:
            self._offset = self._state * self._seg_width()
            self.update()

    def getOffset(self) -> float:
        return float(self._offset)

    def setOffset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = QtCore.pyqtProperty(float, fget=getOffset, fset=setOffset)

    # ---- QComboBox-compatible surface (so a choice handler is widget-agnostic) ----
    def currentText(self) -> str:
        return self._options[self._state] if self._options else ""

    def setCurrentText(self, text: str) -> None:
        for i, opt in enumerate(self._options):
            if opt == str(text):
                self._state = i
                self._offset = i * self._seg_width()
                self.update()
                return

    def state(self) -> int:                       # confocal parity: the active index
        return self._state


def _apply_fluent_context_menu(widget, event) -> None:
    """Pop ``widget``'s right-click menu with the Fluent rounded-card chrome instead of the platform's
    opaque SQUARE "black box" menu.  The ONE shared context-menu builder for every editable text widget
    (``FluentLineEdit``, ``FluentReadoutMultiline``, ``FluentCodeEdit``): it REUSES Qt's own standard
    context menu -- so Undo / Cut / Copy / Paste / Delete / Select-All stay wired to THAT widget -- and
    only swaps the container, re-homing the actions into a :class:`_FluentRoundedMenu` (ownership moved
    via ``setParent`` before the throwaway standard menu is disposed, so its teardown cannot delete them
    out from under the Fluent menu).  A read-only widget's standard menu is just Copy / Select-All; it is
    styled identically -- no widget special-cases the menu.  ``_FluentRoundedMenu`` resolves at call time,
    so this may sit above its definition."""
    std = widget.createStandardContextMenu()
    menu = _FluentRoundedMenu(widget)
    for act in list(std.actions()):
        act.setParent(menu)
        menu.addAction(act)
    std.deleteLater()
    menu.exec_(event.globalPos())
    event.accept()


class FluentLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._res_step: float | None = None
        self._allow_any = True
        self.setMinimumHeight(scaled_px(30, minimum=22))
        self._apply_style()
        self.setText(str(text))
        self.editingFinished.connect(self._snap_to_resolution)

    def _apply_style(self) -> None:
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

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _apply_fluent_context_menu(self, event)

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
        """Restrict typed input to a number (a numeric validator on the line edit).

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


class FluentReadoutEdit(FluentLineEdit):
    """A read-only BUT selectable / copyable text field -- the THIRD line-edit kind.

    Three visually distinct text widgets, so a user tells at a glance what a field is:

      * :class:`FluentLineEdit`    -- WHITE fill + accent focus ring: an EDITABLE input.
      * :class:`FluentLabel`       -- transparent, no border: pure DISPLAY (cannot select).
      * :class:`FluentReadoutEdit` -- light-grey fill + faint border, NO focus ring: you can
        select + Ctrl-C the text but NOT edit it (a command result, a resolved file path, a
        readout value).

    Inherits FluentLineEdit's sizing / API (``setText`` / ``text``) so it drops into the same
    rows; it just paints the read-only look and refuses edits."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setReadOnly(True)
        self.setCursor(QtCore.Qt.IBeamCursor)        # the I-beam signals "selectable text"

    def _apply_style(self) -> None:
        # grey fill (not white -> not an input) + a faint border (more than a bare label) +
        # full-contrast text; NO accent focus border, because it cannot be edited.
        self.setStyleSheet(
            f"""
            QLineEdit {{
                background: {BG};
                border: 1px solid {DIVIDER};
                border-radius: {_radius()}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QLineEdit:focus {{ border: 1px solid {DIVIDER}; }}
            """
        )


class FluentReadoutMultiline(QtWidgets.QPlainTextEdit):
    """A MULTI-LINE :class:`FluentReadoutEdit` -- the read-only-but-selectable text field for a
    value TOO LONG for one line (a resolved file path, a device-metadata blob, a data shape).

    Same look/role as :class:`FluentReadoutEdit` (grey fill + faint border + full-contrast text, NO
    accent focus ring: you can select + Ctrl-C but not edit), but it SOFT-WRAPS a long value at the
    field width (``WrapAtWordBoundaryOrAnywhere``) and AUTO-SIZES its height to the WHOLE wrapped
    content -- so every line is shown in full and nothing is ever truncated the way a single-line edit
    clips it.  There is NO row cap and NO inner scrollbar: the field grows to fit all its lines and the
    outer scroll area (the Info column's ``FluentScrollArea``) scrolls the whole form -- an inner
    scrollbar on a field that already shows everything is exactly the "why scroll when it's all here"
    confusion.  Mirrors confocal_gui's ``DynamicPlainTextEdit`` (scrollbars OFF, count the VISUAL
    wrapped lines, size to fit), on the house Fluent readout chrome.  All colours / radius / font come
    from the style tokens (one source), never re-picked."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse
                                     | QtCore.Qt.TextSelectableByKeyboard)
        self.setCursor(QtCore.Qt.IBeamCursor)          # the I-beam signals "selectable text"
        # Soft-wrap a long value at the field width (break inside a long token like a path too), so it
        # wraps to as many lines as it needs instead of overflowing / clipping (confocal idiom).
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        # BOTH scrollbars OFF: width-wrapped -> never sideways, and the field auto-sizes to fit EVERY
        # line so a vertical scrollbar would never have anything hidden to scroll to (confocal idiom).
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        apply_fluent_scrollbars(self)
        # Match the document's inner margin to the single-line edit's vertical padding (PADDING_V): the
        # default QPlainTextEdit documentMargin (4 px) makes one text row's content box TALLER than the
        # FluentReadoutEdit row, so a one-line field could not both equal that height AND fully contain
        # its text.  At PADDING_V the row's content box lines up with the line edit's, so the anchored
        # height formula (single-row height + one lineSpacing per extra line) shows every line in full.
        self.document().setDocumentMargin(scaled_px(PADDING_V))
        self._apply_style()
        if text:
            self.setPlainText(str(text))
        # Recompute height when the text changes AND when the wrapped layout size changes (a resize
        # re-wraps the same text into a different line count) -- the two triggers confocal uses.
        self.textChanged.connect(self._adjust_height)
        self.document().documentLayout().documentSizeChanged.connect(lambda *_: self._adjust_height())
        self._adjust_height()

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _apply_fluent_context_menu(self, event)   # Fluent right-click (Copy / Select-All), not the native menu

    def _apply_style(self) -> None:
        # The SAME readout chrome as FluentReadoutEdit: grey fill (not white -> not an input) + a faint
        # border + full-contrast text; NO accent focus border, because it cannot be edited.
        self.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background: {BG};
                border: 1px solid {DIVIDER};
                border-radius: {_radius()}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QPlainTextEdit:focus {{ border: 1px solid {DIVIDER}; }}
            """
        )

    def setText(self, text: str) -> None:  # noqa: N802 - match FluentReadoutEdit's line-edit API
        """Quack like a line edit (``setText`` / ``text``) so it drops into the same rows/fillers."""
        self.setPlainText(str(text))

    def text(self) -> str:
        return self.toPlainText()

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        # Wrapping depends on the viewport width, so a width change re-wraps -> re-measure the height.
        super().resizeEvent(event)
        self._adjust_height()

    def _visual_line_count(self) -> int:
        """The number of VISUAL lines the text occupies once wrapped at the current width (a wrapped
        long line counts as several) -- confocal's DynamicPlainTextEdit line-count walk."""
        doc = self.document()
        doc.setTextWidth(self.viewport().width())     # wrap at the live viewport width
        doc.adjustSize()                              # force relayout so line counts are current
        layout = doc.documentLayout()
        count = 0
        block = doc.begin()
        while block.isValid():
            if layout is not None:
                layout.blockBoundingRect(block)       # ensure the block's layout exists
            block_layout = block.layout()
            count += block_layout.lineCount() if block_layout is not None else 1
            block = block.next()
        return max(1, count)

    def _adjust_height(self) -> None:
        """Size the field to fit ALL its wrapped content (no row cap -> every line shown, no inner
        scroll).

        A ONE-line value must land at EXACTLY the single-line :class:`FluentReadoutEdit` height so an
        Info row with a short value looks identical whether it is a line edit or this multi-line field.
        So the height is ANCHORED to that single-row height -- ``scaled_px(30, minimum=22)``, the SAME
        expression :class:`FluentLineEdit` uses for its own row -- and each EXTRA wrapped line adds one
        ``lineSpacing``.  That base row already reserves the doc margin + PADDING_V + border of one text
        line, and every added line contributes exactly its own text height, so the field grows to show
        every line in full with none clipped (the QPlainTextEdit ``frameWidth`` is NOT used: with a
        stylesheet 1 px border it over-reports and would inflate a one-line field past the line edit).

        Re-entrancy guard: measuring the wrapped line count calls ``document().adjustSize()`` and the
        resize re-lays-out the document, either of which can emit ``documentSizeChanged`` -- a connected
        trigger of this method -- so an unguarded call recurses.  The flag lets the outermost call settle
        the height; a nested emission is dropped, and the trailing ``documentSizeChanged`` from the final
        resize re-runs it once cleanly (converging, since a fixed WIDTH gives a stable wrapped count)."""
        if getattr(self, "_adjusting_height", False):
            return
        self._adjusting_height = True
        try:
            lines = self._visual_line_count()
            line_height = self.fontMetrics().lineSpacing()
            base_row = scaled_px(30, minimum=22)    # == FluentLineEdit's single-row height (one source)
            total = base_row + (lines - 1) * line_height
            if total != self.height():
                self.setFixedHeight(total)
            # Absorb any residual: if the wrapped content is still a hair taller than the anchored
            # estimate (sub-pixel wrap rounding differs run-to-run), the scrollbar reports how many
            # pixels are hidden -- add exactly those so the LAST line's glyphs are never clipped.  A
            # one-line value has no residual, so it stays exactly the single-row height.
            residual = self.verticalScrollBar().maximum()
            if residual > 0:
                self.setFixedHeight(total + residual)
            self.updateGeometry()
        finally:
            self._adjusting_height = False


class FluentPathEdit(QtWidgets.QWidget):
    """A path field + a ``Browse…`` button -- the ONE reusable control for every path
    parameter (a data folder, a saved pulse-template ``.json``, a calibration file).

    The user can TYPE a path or click Browse to pick it in the native file/folder
    dialog, so a path is never a bare line-edit you must hand-type into.  ``mode="dir"``
    opens a folder chooser; ``mode="file"`` an open-file chooser filtered by
    ``file_filter``.  Quacks like a line edit (``text`` / ``setText`` /
    ``setPlaceholderText`` / ``setToolTip``) and emits ``changed(str)`` so a form can
    treat it exactly like a ``FluentLineEdit``."""

    changed = QtCore.pyqtSignal(str)

    def __init__(self, text: str = "", *, mode: str = "file", caption: str = "Choose",
                 file_filter: str = "All files (*)", base_dir: str = "", parent=None):
        super().__init__(parent)
        self._mode = "dir" if str(mode) == "dir" else "file"
        self._caption = str(caption)
        self._filter = str(file_filter)
        # The folder the Browse dialog opens in when the field doesn't point at an existing
        # path -- e.g. ``pulses`` for a pulse template, ``calibrations`` for a data folder --
        # so the operator lands where the files actually live, not the process CWD.
        self._base_dir = str(base_dir)
        self.setStyleSheet("background: transparent;")
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(scaled_px(6, minimum=4))
        self.edit = FluentLineEdit(str(text))
        # a modest floor so the field + button still fit a narrow form column (the
        # edit grows with the column via the stretch below); never a 150px floor that
        # pushes the Browse button off a narrow Edit tab.
        self.edit.setMinimumWidth(scaled_px(96, minimum=72))
        self.edit.textChanged.connect(lambda t: self.changed.emit(str(t)))
        self.browse = FluentButton("Browse…", color=GREY)
        # size the button to its OWN label (+ padding) at the live DPR so "Browse…"
        # is never clipped (a fixed 80px clipped it to "owse" at some scales).
        self.browse.setFixedWidth(
            fluent_text_width(self.browse.fontMetrics(), "Browse…") + scaled_px(22, minimum=16))
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.edit, 1)
        row.addWidget(self.browse, 0)

    def _browse(self) -> None:
        start = self._dialog_start()
        if self._mode == "dir":
            picked = QtWidgets.QFileDialog.getExistingDirectory(self, self._caption, start)
        else:
            picked, _ = QtWidgets.QFileDialog.getOpenFileName(self, self._caption, start, self._filter)
        if picked:
            self.edit.setText(picked)   # fires textChanged -> changed

    def _dialog_start(self) -> str:
        """Where the Browse dialog opens: the current path if it's a real file/dir or sits in
        a real directory the user typed; otherwise ``base_dir`` (the folder these files live
        in, created if needed) so a BARE default like ``imaging_template.json`` still lands in
        ``pulses/`` rather than the CWD; otherwise the CWD."""
        import os
        cur = self.edit.text().strip()
        if cur and os.path.exists(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent and os.path.isdir(parent):       # user typed a real-dir-qualified path
            return cur
        if self._base_dir:
            base = os.path.abspath(self._base_dir)
            if not os.path.isdir(base):
                try:
                    os.makedirs(base, exist_ok=True)
                except OSError:
                    return os.getcwd()
            return base
        return os.getcwd()

    def text(self) -> str:
        return self.edit.text()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self.edit.setText("" if text is None else str(text))

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self.edit.setPlaceholderText(str(text))

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt API name
        self.edit.setToolTip(str(text))
        self.browse.setToolTip(str(text))


class FluentSectionLabel(QtWidgets.QLabel):
    """Bold, own-line section header for confocal-style vertical settings popups.

    Confocal's per-plot settings page uses ONE QGroupBox to mark sections; we use a
    bold dim label per section instead so multiple sections stack cleanly in a
    single VBox without nested borders.  Reuse this for every settings popup so the
    visual rhythm is identical across GUIs."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        # Bold must go in the STYLESHEET (it is what actually renders): setting a stylesheet AFTER
        # setFont re-polishes the widget and drops a QFont-only bold, so a section ended up the same
        # weight as a row label (the "can't tell a section from a row" bug).  Set the stylesheet
        # (with font-weight: bold) FIRST, then setFont last so widget.font().bold() also stays True.
        self.setStyleSheet(f"color: {TEXT}; background: transparent; border: none; font-weight: bold;")
        font = self.font()
        font.setBold(True)
        font.setPointSize(fluent_font_size())
        self.setFont(font)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.setContentsMargins(0, scaled_px(4), 0, 0)


def setting_label_width(labels, *, minimum: int = 72) -> int:
    """The ONE label-column width for a settings/edit/param form: wide enough that the LONGEST
    of ``labels`` shows in full, never below ``minimum`` (scaled).  Every form computes its
    column with THIS function, so each form's rows align and the width rule is identical across
    the whole frontend (a form with longer labels simply gets a wider column).  This is the single
    source of the "[label | control]" column width -- callers never hand-pick a px value."""
    metrics = QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))
    widest = 0
    for text in labels:
        try:
            widest = max(widest, metrics.horizontalAdvance(str(text)))
        except AttributeError:  # pragma: no cover - very old Qt
            widest = max(widest, metrics.width(str(text)))
    return max(scaled_px(int(minimum), minimum=48), widest + scaled_px(10))


class FluentSettingRow(QtWidgets.QWidget):
    """One settings row -- the ONE param-row primitive for the whole frontend:
    ``[grey fixed-width label | control flush-left, expanding]``.

    THE design system (one logic everywhere): a form is built from exactly two primitives --
    :class:`FluentSectionLabel` (bold, dark, own-line) for a GROUP header, and this row (grey
    label, fixed-width column) for each control.  The section/row hierarchy is shown by
    weight+colour ONLY -- never by indentation, never by a third colour.  Share ONE column width
    per form via :func:`setting_label_width` so the rows align.  Do NOT use a ``QFormLayout`` (its
    labels render dark + auto-width = a different style) and do NOT use a bare ``FluentLabel`` as a
    row/section label.  Mirrors confocal_gui's ``[QLabel | widget]`` row template."""

    def __init__(
        self,
        label: str,
        control: QtWidgets.QWidget,
        *,
        label_width: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(scaled_px(6, minimum=4))
        lbl = QtWidgets.QLabel(label)
        lbl.setFixedWidth(int(label_width) if label_width is not None else scaled_px(60, minimum=48))
        lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lbl.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self._label = lbl                       # introspected by the layout-uniformity contract test
        h.addWidget(lbl)
        # A control that FILLS the cell (a combo / line-edit / spin / switch -- Expanding/Preferred
        # horizontal policy) is stretched so its left edge sits flush against the label column.  A
        # control pinned to its sizeHint (FIXED horizontal policy -- the FluentTriStateToggle capsule,
        # a FluentButton) must NOT be stretched: a stretch cell CENTRES a non-filling widget, so the
        # toggle would float in the middle of the row instead of left-aligning under its label.  Add it
        # left-aligned with a trailing stretch instead (confocal's TriStateToggleSwitch row idiom), so
        # EVERY control -- filling or fixed -- starts at the SAME left edge.
        if control.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Fixed:
            h.addWidget(control, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            h.addStretch(1)
        else:
            h.addWidget(control, 1)


class _RoundedPopupCard(QtCore.QObject):
    """Paints a FluentPopup-style rounded card (white fill + 1 px border) onto the
    widget it filters, suppressing that widget's default opaque square paint.

    Used on a QComboBox's top-level drop-down container: the container is a private
    C++ widget we cannot subclass, and assigning ``container.paintEvent`` does
    nothing (Qt's C++ dispatch never calls a Python instance attribute).  An event
    filter is the one reliable hook -- on the Paint event it strokes the card and
    returns True so the default (square, opaque) paint is skipped; the translucent
    container then shows nothing outside the rounded arc (no bottom-right nub)."""

    def __init__(self, radius: float, border: str, *, combo=None, gap: int = 0, parent=None):
        super().__init__(parent)
        self._radius = float(radius)
        self._border = QtGui.QColor(border)
        self._combo = combo            # the owning FluentComboBox (for the outer-gap offset)
        self._gap = int(gap)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        et = event.type()
        if et == QtCore.QEvent.Paint and isinstance(obj, QtWidgets.QWidget):
            painter = QtGui.QPainter(obj)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            pen = QtGui.QPen(self._border)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor("white"))
            rect = QtCore.QRectF(obj.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.drawRoundedRect(rect, self._radius, self._radius)
            painter.end()
            return True
        if et == QtCore.QEvent.Move and self._combo is not None and isinstance(obj, QtWidgets.QWidget):
            # ALWAYS BELOW: the dropdown must open downward off the box, never flip above it.
            # Qt drops the popup flush against the combo (and near the screen bottom would flip it
            # UPWARD) in a deferred step a showPopup-time move() can't beat; re-apply the downward
            # offset every time it is repositioned so the dropdown sits a few px BELOW the box -- the
            # same gap the Setting FluentPopup uses below its button.  Self-converging: once at target
            # the guard no-ops, so the corrective move does not loop.  (Overrun past the screen bottom
            # is handled by clamping the popup HEIGHT to the below-the-box space so it scrolls -- see
            # FluentComboBox.showPopup / _anchor_available_height -- not by flipping it up.)
            below = self._combo.mapToGlobal(QtCore.QPoint(0, self._combo.height())).y()
            geo = obj.geometry()
            target = below + self._gap
            if abs(geo.top() - target) > 1:
                obj.move(geo.left(), target)
            return False
        return False


class _FluentRoundedMenu(QtWidgets.QMenu):
    """A ``QMenu`` with the SAME antialiased rounded-card chrome as every other Fluent popup
    (the FluentPopup Setting card, the FluentComboBox drop-down), reusable frontend-owned.

    A stylesheet ``border-radius`` alone is NOT enough on a menu: its window stays opaque + SQUARE
    behind the arc, so the corners outside the radius show a white nub (and the platform adds a native
    shadow) -- reading as a hard-cornered popup beside the antialiased round ones.  So the window is
    made TRANSLUCENT + frameless and this PAINTS the card itself (fill + 1 px border) via the same
    ``drawRoundedRect`` single source, BEFORE the items -- unlike :class:`_RoundedPopupCard` (which
    filters a container that owns no items and so can suppress its paint), a menu draws its own items,
    so the card is composited UNDER them (super paints the labels on top) rather than replacing them."""

    def __init__(self, parent=None, *, radius: float | None = None,
                 border: str = DIVIDER, fill: str = "white"):
        super().__init__(parent)
        self._radius = float(_radius() if radius is None else radius)
        self._border = QtGui.QColor(border)
        self._fill = QtGui.QColor(fill)
        # Translucent frameless window so nothing opaque/square sits behind the painted arc.
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint
                            | QtCore.Qt.NoDropShadowWindowHint)
        # Transparent menu body (the card is painted, not a stylesheet background) + a rounded ACCENT
        # pill on the selected/hover row -- the same item chrome the combo drop-down uses.
        # The item has a horizontal MARGIN inside the menu's own padding so the selected/hover pill's
        # rounded rect sits fully WITHIN the card's rounded corners (a pill flush to the padding edge
        # gets clipped by the arc -> a square-cornered grey band bleeding past the corner).  The
        # ``::indicator`` is collapsed to nothing (image:none + 0x0) so a (formerly checkable) action
        # never reserves a tick gutter that shifts the labels right / draws an un-Fluent check box.
        self.setStyleSheet(
            f"""
            QMenu {{ background: transparent; border: none;
                     padding: {scaled_px(4, minimum=3)}px;
                     font: {fluent_font_size()}pt "{FONT}"; color: {TEXT}; }}
            QMenu::item {{ padding: {scaled_px(5, minimum=4)}px {scaled_px(14, minimum=10)}px;
                          margin: 0px {scaled_px(2, minimum=2)}px;
                          border-radius: {_radius()}px; }}
            QMenu::item:selected {{ background: {ACCENT}; color: white; }}
            QMenu::indicator {{ image: none; width: 0px; height: 0px; }}
            """
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Paint the rounded card FIRST (fill + 1 px antialiased border), then let QMenu draw its items
        # on top -- same drawRoundedRect recipe as FluentPopup / _RoundedPopupCard (one visual source).
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(self._border)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(self._fill)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(rect, self._radius, self._radius)
        painter.end()
        super().paintEvent(event)


class _FluentMessageDialog(QtWidgets.QDialog):
    """A modal message / warning box with the Fluent rounded-card chrome and NO native title bar -- so
    no platform window frame and no stray Python app icon in the corner (the reason a raw ``QMessageBox``
    reads off-brand here).  Frameless + translucent, it PAINTS the same rounded white card (fill + 1 px
    DIVIDER border, the shared ``drawRoundedRect`` recipe) as every other Fluent surface, stacks a bold
    title (``FluentSectionLabel``), the wrapped message (``FluentLabel``) and a single Fluent OK button,
    and centres on its parent.  ``fluent_message`` is the ONE entry point -- use it in place of
    ``QMessageBox.information`` / ``.warning``."""

    def __init__(self, parent, title: str, text: str, *, kind: str = "info"):
        super().__init__(parent, QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setModal(True)
        self._radius = float(_radius())
        pad = scaled_px(16, minimum=10)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(scaled_px(10, minimum=6))
        outer.addWidget(FluentSectionLabel(str(title)))
        body = FluentLabel(str(text))
        body.setWordWrap(True)
        body.setMaximumWidth(scaled_px(360, minimum=240))
        outer.addWidget(body)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        # A warning wears the ORANGE accent (the same destructive/attention hue Remove uses); a plain
        # info uses the resting ACCENT -- the ONE palette, never a new colour.
        ok = FluentButton("OK", color=(ORANGE if str(kind) == "warning" else ACCENT))
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        outer.addLayout(row)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor(DIVIDER))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor("white"))
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(rect, self._radius, self._radius)


def fluent_message(parent, title: str, text: str, *, kind: str = "info") -> None:
    """Show a modal Fluent message dialog (``kind`` = ``"info"`` | ``"warning"``) -- the ONE on-brand
    replacement for ``QMessageBox.information`` / ``.warning`` (no native frame, no stray app icon).
    Centres on ``parent``.  Blocks until the user clicks OK (or presses Escape)."""
    dlg = _FluentMessageDialog(parent, title, text, kind=kind)
    dlg.adjustSize()
    if parent is not None:
        centre = parent.mapToGlobal(parent.rect().center())
        dlg.move(centre.x() - dlg.width() // 2, centre.y() - dlg.height() // 2)
    dlg.exec_()


#: Max rows a Fluent drop-down shows before it SCROLLS (the shared fluent scrollbar) -- one source for
#: both the flat combo and the collapsible tree picker, so a long signal list never overflows the screen.
_COMBO_MAX_VISIBLE_ITEMS = 12


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
        # Cap the drop-down at a sensible number of visible rows so a long signal list SCROLLS (via the
        # shared fluent scrollbar already styled below) instead of growing unbounded / overflowing the
        # screen.  Qt shows its vertical scrollbar once the item count exceeds this (#H3w-4).
        self.setMaxVisibleItems(_COMBO_MAX_VISIBLE_ITEMS)
        self._popup_styled = False
        # The drop-down list is a top-level popup.  It must look EXACTLY like the
        # FluentPopup card (the Setting popup): one antialiased rounded rect with a
        # 1 px border, a few-px gap before the items, and the SHARED fluent
        # scrollbar.  A stylesheet ``border-radius`` on the popup is NOT enough --
        # the popup's window stays opaque + square behind the arc, so the corner
        # triangles outside the radius show a white/grey nub (the "stray line at the
        # bottom-right" the user hit) and Windows adds a native popup shadow.  So
        # the popup CONTAINER is made translucent + frameless and PAINTS the card
        # itself (see showPopup), and the item view below is transparent so that
        # painted card shows through.
        view = QtWidgets.QListView(self)
        # The item view must be GENUINELY transparent so the container's painted
        # rounded card shows through (a stylesheet ``background: transparent`` alone
        # does not stop QAbstractScrollArea's viewport from filling its Base colour,
        # which would paint an opaque SQUARE over the rounded card).
        view.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        view.viewport().setAutoFillBackground(False)
        view.viewport().setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        view_palette = view.palette()
        view_palette.setColor(QtGui.QPalette.Base, QtCore.Qt.transparent)
        view_palette.setColor(QtGui.QPalette.Window, QtCore.Qt.transparent)
        view.setPalette(view_palette)
        self.setView(view)
        self._gap = _popup_gap()   # the ONE Fluent popup outer-gap (shared with the overflow menu, etc.)
        # Configure the drop-down's top-level container NOW (setView created it but it
        # is not shown yet) -- WA_TranslucentBackground only takes effect if it is set
        # BEFORE the native window is created, so this cannot wait until showPopup.
        self._configure_popup_container()
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
                background: transparent;
                border: none;
                padding: {self._gap}px;
                color: {TEXT};
                selection-background-color: {ACCENT};
                selection-color: white;
                font: {fluent_font_size()}pt "{FONT}";
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: {scaled_px(3, minimum=2)}px {scaled_px(6, minimum=4)}px;
                border-radius: {_radius()}px;
                color: {TEXT};
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {HOVER};
                color: white;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {ACCENT};
                color: white;
            }}
            QComboBox QAbstractItemView::corner {{ background: transparent; border: none; }}
            """
            + fluent_scrollbar_stylesheet("QComboBox QAbstractItemView QScrollBar")
        )

    def _configure_popup_container(self) -> None:
        """Make the drop-down's top-level container a translucent, frameless card
        that PAINTS one rounded rect (border + fill) -- the SAME chrome FluentPopup
        uses for the Setting popup.

        The rounded card is painted by an EVENT FILTER on the container, not by
        assigning ``container.paintEvent`` (Qt's C++ event dispatch never calls a
        Python instance attribute, so a monkey-patched paintEvent silently does
        nothing and the container keeps its default opaque SQUARE).  The filter
        intercepts the Paint event, strokes the rounded card and returns True so the
        default square paint is suppressed; the transparent item view paints its
        rows on top.  The few-px gap before the items is the view's stylesheet
        ``padding`` (the translucent card shows through it)."""
        container = self.view().window()
        if container is None or container is self or self._popup_styled:
            return
        self._popup_styled = True
        container.setWindowFlags(container.windowFlags()
                                 | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint)
        container.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        container.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        layout = container.layout()
        if layout is not None:
            layout.setContentsMargins(0, 0, 0, 0)
        self._card_filter = _RoundedPopupCard(float(_radius()), DIVIDER,
                                              combo=self, gap=self._gap, parent=container)
        container.installEventFilter(self._card_filter)

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
        text = metrics.elidedText(self._display_text(), QtCore.Qt.ElideRight, text_width)
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

    def showPopup(self):  # noqa: N802 - Qt naming
        # Reconfigure the popup container (translucent + rounded-card event filter)
        # in case Qt rebuilt it since __init__.  The OUTER gap (dropdown sits a few
        # px off the box) AND the ALWAYS-DOWNWARD placement are enforced by the
        # container's Move event filter (_RoundedPopupCard), which beats Qt's
        # deferred flush-positioning that a showPopup-time move() cannot.
        self._configure_popup_container()
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
        self._place_popup_below()

    def _place_popup_below(self) -> None:
        """Force the open drop-down to sit BELOW the box (never flipped above it).

        Qt positions the popup container in a deferred flush AFTER ``super().showPopup()`` and, near the
        screen bottom, would flip it UPWARD.  We move the container to the combo's bottom-left + the outer
        gap right away (and ``_RoundedPopupCard``'s Move filter re-applies the same downward target on
        Qt's later flush, so the box can never end up above).  The x is clamped into the screen so a combo
        at the right edge does not push the popup off-screen horizontally; the height is clamped to the
        below-the-box space elsewhere so an overrun SCROLLS rather than flipping up."""
        view = self.view()
        container = view.window() if view is not None else None
        if container is None or container is self:
            return
        below = self.mapToGlobal(QtCore.QPoint(0, self.height() + self._gap))
        x = below.x()
        screen = self.screen() if hasattr(self, "screen") else QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            x = max(avail.left(), min(x, avail.right() - container.width()))
        container.move(x, below.y())

    def _display_text(self) -> str:
        """The text painted in the COLLAPSED combo (the seam a tree combo overrides to show a
        fuller producer-qualified name).  Default: the current item's own text."""
        return self.currentText()

    def wheelEvent(self, event):
        view = self.view()
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


class _ExpandableTreeView(QtWidgets.QTreeView):
    """A ``QTreeView`` whose PARENT (header) rows toggle expansion on a click ANYWHERE on the row
    -- the text as well as the disclosure triangle -- so a signal-group header is easy to open
    (G2 follow-up: "click the name OR the arrow"), exactly once (the click is swallowed so Qt's
    native triangle toggle never double-fires).  LEAF rows fall through to the normal click path
    (``clicked`` -> ``FluentTreeComboBox._on_tree_clicked`` selects them).  Reusable: the behaviour
    lives on the widget, not in any caller."""

    def mousePressEvent(self, event):  # noqa: N802 - Qt naming
        index = self.indexAt(event.pos())
        model = index.model() if index.isValid() else None
        item = model.itemFromIndex(index) if isinstance(model, QtGui.QStandardItemModel) else None
        # a producer header = has children + carries no leaf payload (UserRole) -> toggle on any click
        if item is not None and item.hasChildren() and item.data(QtCore.Qt.UserRole) is None:
            self.setExpanded(index, not self.isExpanded(index))
            return                                # swallow: no native double-toggle, no header "select"
        super().mousePressEvent(event)


class FluentTreeComboBox(FluentComboBox):
    """A signal picker whose drop-down is a COLLAPSIBLE TREE (G2): one expandable parent row per
    producing node (collapsed by default -- click the name OR the triangle to reveal its signals),
    each leaf a selectable signal showing its shape + ready/waiting state.  The popup GROWS to fit a
    freshly-expanded parent's children (instead of scrolling them); collapsed, the box paints the
    fuller ``producer · signal`` name (frame-title aligned, G3) so you can read the source without
    opening it.  Built on the standard QComboBox-with-QTreeView idiom (a leaf's ``currentData`` is
    the bare signal name); the populate/read helpers live in task_console so the data shape stays there."""

    #: emitted (with the bare signal name) when the user picks a leaf
    signalPicked = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        tree = _ExpandableTreeView(self)         # parent rows toggle on a name OR triangle click
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)            # show the expand/collapse triangles
        tree.setItemsExpandable(True)
        tree.setUniformRowHeights(True)
        tree.setExpandsOnDoubleClick(False)      # single click on the triangle expands
        tree.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        tree.viewport().setAutoFillBackground(False)
        tree.viewport().setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        pal = tree.palette()
        pal.setColor(QtGui.QPalette.Base, QtCore.Qt.transparent)
        pal.setColor(QtGui.QPalette.Window, QtCore.Qt.transparent)
        tree.setPalette(pal)
        self._model = QtGui.QStandardItemModel(self)
        self.setModel(self._model)
        self.setView(tree)
        self._configure_popup_container()
        # {bare signal name -> producer-qualified label}, built when the tree is populated.  The collapsed
        # box derives its text LIVE from this + the CURRENT selection (see _display_text) -- there is NO
        # cached "_display" to go stale: a pick via the native view click, _on_tree_clicked, OR a
        # programmatic set_signal_tree all change currentData, so the box always tracks the real pick (#1).
        self._full_by_bare: dict[str, str] = {}
        # selecting a leaf in the tree drives the combo's current item + emits signalPicked
        tree.clicked.connect(self._on_tree_clicked)
        # When a parent expands/collapses, RE-GROW the open popup so the children are fully visible
        # (Qt sizes the popup once at open and never re-fits a tree expand) -- the box "变长".
        tree.expanded.connect(self._resize_popup_to_contents)
        tree.collapsed.connect(self._resize_popup_to_contents)

    def showPopup(self):  # noqa: N802 - Qt naming
        super().showPopup()
        self._resize_popup_to_contents()         # fit the initial (collapsed) height on open

    def _popup_pad(self) -> int:
        return 2 * self._gap                                     # the translucent card's top/bottom gap

    def _desired_popup_height(self) -> int:
        """Pixel height the popup should be to show every CURRENTLY-VISIBLE row (expanded subtrees
        included), clamped to the screen.  Pure computation (no widget mutation) so it is unit-
        testable: counts visible rows structurally x ``sizeHintForRow`` (the delegate's height,
        valid the instant the expand signal fires -- ``rowHeight`` under-reports freshly-shown
        children before they lay out)."""
        view = self.view()
        if not isinstance(view, QtWidgets.QTreeView):
            return 0
        model = self._model
        rows = 0

        def _visit(parent_index):
            nonlocal rows
            for r in range(model.rowCount(parent_index)):
                rows += 1
                idx = model.index(r, 0, parent_index)
                if view.isExpanded(idx):
                    _visit(idx)

        _visit(QtCore.QModelIndex())
        if rows <= 0:
            return 0
        row_h = view.sizeHintForRow(0)                          # uniform-row height from the delegate
        if row_h <= 0:
            row_h = view.fontMetrics().height() + scaled_px(8)
        desired = rows * row_h + 2 * view.frameWidth() + self._popup_pad()
        # Clamp to the space ACTUALLY available at the anchor (the larger of below-the-box / above-the-box,
        # whichever side the popup opens), NOT the whole screen -- otherwise a popup anchored low overruns
        # the screen bottom.  When the content exceeds that space the popup stays at the boundary and the
        # tree SCROLLS (the shared fluent scrollbar), exactly like the Setting popup (#issue-4).
        avail = self._anchor_available_height()
        if avail > 0:
            desired = min(desired, avail)
        return int(desired)

    def _anchor_available_height(self) -> int:
        """Pixels available for the popup BELOW the combo box, down to the screen bottom.

        The popup ALWAYS opens downward (``_place_popup_below`` forces it; the dropdown never flips above
        the box).  So the grow-vs-scroll cap is always the space from the box's bottom edge (+ the outer
        gap) to the screen bottom: the popup grows to fit its visible rows up to THIS height, then SCROLLS
        the overflow (the shared fluent scrollbar) -- it never overruns the screen and never flips up."""
        screen = self.screen() if hasattr(self, "screen") else QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return 0
        avail = screen.availableGeometry()
        bottom_g = self.mapToGlobal(QtCore.QPoint(0, self.height())).y()   # the box's bottom edge, global
        return int(max(0, avail.bottom() - (bottom_g + self._gap)))

    def _resize_popup_to_contents(self, *_) -> None:
        """Re-grow/shrink the OPEN popup so a freshly-expanded parent's children are fully visible
        instead of scrolled.  Forces BOTH the view and its container height (``setFixedHeight``) so
        the container's own layout can't keep the view short."""
        view = self.view()
        if not isinstance(view, QtWidgets.QTreeView):
            return
        container = view.window()
        # NOT gated on ``view.isVisible()``: during the FIRST showPopup (and its first parent-expand)
        # the popup's visibility has not propagated yet -- ``view.isVisible()`` is transiently False
        # even though super().showPopup() already POSITIONED the container and ``_desired_popup_height``
        # computes the right (full) height.  Gating on it dropped the very first open's/expand's grow,
        # so the picker opened CLIPPED until the SECOND open.  The container guard (a real top-level
        # popup window the combo owns, never ``self``) is the correct gate -- it holds whether or not
        # the transient visible flag has caught up.  ``_resize_popup_to_contents`` is only ever reached
        # during the open lifecycle (showPopup + the tree expand/collapse signals), so resizing a
        # not-yet-flagged-visible-but-positioned container is safe.
        if container is None or container is self:
            return
        desired = self._desired_popup_height()
        if desired <= 0:
            return
        view.setFixedHeight(max(0, desired - self._popup_pad()))   # force the view tall enough...
        container.setFixedHeight(max(scaled_px(40), desired))      # ...and the container to match

    def _display_text(self) -> str:
        """The collapsed text = the producer-qualified label of the CURRENT pick, derived LIVE from the
        selection (no cached field): look the current bare signal up in the populate-time map.  So the
        box tracks EVERY selection change -- native view click, _on_tree_clicked, or set_signal_tree --
        and never shows the raw verbose leaf text (#1)."""
        full = self._full_by_bare.get(self.current_signal(), "")
        return full or self.currentText()

    def _on_tree_clicked(self, index) -> None:
        item = self._model.itemFromIndex(index)
        if item is None or item.data(QtCore.Qt.UserRole) is None:
            return                                # a parent header: let the triangle expand it
        parent = item.parent()
        if parent is None:
            return
        self.setRootModelIndex(parent.index())
        self.setCurrentIndex(item.row())
        self.setRootModelIndex(QtCore.QModelIndex())
        self.hidePopup()
        # repaint() NOT update(): the box lives inside the Setting Qt.Popup, and closing the tree's own
        # child popup (hidePopup) leaves the SCHEDULED async repaint of update() unprocessed -- so the box
        # showed the STALE pick until the whole Setting was closed + reopened (#1).  repaint() forces the
        # (live-derived) producer-qualified label to paint NOW.
        self.repaint()
        # Emit BOTH the explicit signalPicked AND the standard ``activated`` so any existing
        # ``combo.activated.connect(...)`` wiring fires (setCurrentIndex alone does not emit it).
        self.activated.emit(self.currentIndex())
        self.signalPicked.emit(self.current_signal())

    def set_signal_tree(self, groups, *, current="", none_label=None) -> None:
        """Populate the tree from ``groups`` = ``[(producer, [(leaf_label, bare_name, full_label)])]``
        and select ``current`` (a bare signal name).  Parents are non-selectable + bold + collapsed;
        ``none_label`` adds a leading selectable empty row."""
        self._model.clear()
        sel_parent_row = sel_child_row = None
        self._full_by_bare = {}                  # {bare -> producer-qualified label} for live _display_text
        base = 0
        if none_label is not None:
            none_item = QtGui.QStandardItem(str(none_label))
            none_item.setData("", QtCore.Qt.UserRole)
            none_item.setData("", QtCore.Qt.UserRole + 1)
            none_item.setEditable(False)
            self._model.appendRow(none_item)
            base = 1                                  # producers start one row below the none-row
        for gi, (producer, leaves) in enumerate(groups):
            parent = QtGui.QStandardItem(str(producer))
            parent.setSelectable(False)
            parent.setEditable(False)
            font = parent.font(); font.setBold(True); parent.setFont(font)
            for ci, (leaf_label, bare, full_label) in enumerate(leaves):
                child = QtGui.QStandardItem(str(leaf_label))
                child.setData(str(bare), QtCore.Qt.UserRole)
                child.setData(str(full_label), QtCore.Qt.UserRole + 1)
                child.setEditable(False)
                self._full_by_bare[str(bare)] = str(full_label)   # for the live collapsed-box label
                if current and str(bare) == str(current):
                    # the parent's FINAL model row is base+gi (parent.row() is -1 until appended)
                    sel_parent_row, sel_child_row = base + gi, ci
                parent.appendRow(child)
            self._model.appendRow(parent)
        view = self.view()
        if isinstance(view, QtWidgets.QTreeView):
            view.collapseAll()
        if sel_parent_row is not None:
            parent_index = self._model.index(sel_parent_row, 0)
            self.setRootModelIndex(parent_index)
            self.setCurrentIndex(sel_child_row)
            self.setRootModelIndex(QtCore.QModelIndex())
        elif none_label is not None:
            self.setCurrentIndex(0)
        # repaint() not update(): a rebuild while the Setting is open (a refresh after a pick) must show
        # the producer-qualified label IMMEDIATELY, not on a deferred async paint that the Qt.Popup may
        # not flush until it is closed + reopened (#1).  (A no-op on a not-yet-shown combo.)
        self.repaint()

    def current_signal(self) -> str:
        """The selected leaf's bare signal name ('' for none / a header)."""
        data = self.currentData(QtCore.Qt.UserRole)
        return str(data).strip() if data else ""


class _FluentTabCloseButton(QtWidgets.QToolButton):
    """The close affordance for a closable tab: a small "x" glyph that darkens on
    hover with a subtle round highlight.

    It draws the "x" as TEXT (``QToolButton``), not a custom ``paintEvent`` on a
    ``QAbstractButton`` -- the latter does not composite as a ``QTabBar`` tab
    button (it renders blank), so the close mark was invisible.  A QToolButton's
    text is centred and the hover background is the button rect rounded to a
    circle, so glyph and highlight are concentric by construction.  Frontend-
    owned, reused via :meth:`FluentTabWidget.add_closable_tab`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        size = scaled_px(18, minimum=14)
        self.setFixedSize(size, size)
        self.setText("✕")                       # heavy multiplication x
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setToolTip("Close this tab")
        font = self.font()
        font.setPixelSize(max(9, int(size * 0.62)))
        font.setBold(True)
        self.setFont(font)
        radius = size // 2
        # Clearly-visible grey at rest (a near-white "x" disappears on the tab
        # bar), darkening to TEXT on hover/press with a concentric round
        # highlight -- subtle, not the harsh native red X.
        self.setStyleSheet(
            f"""
            QToolButton {{ border: none; background: transparent; color: #6E6E6E; padding: 0px; }}
            QToolButton:hover {{ background: rgba(0, 0, 0, 38); border-radius: {radius}px; color: {TEXT}; }}
            QToolButton:pressed {{ background: rgba(0, 0, 0, 64); border-radius: {radius}px; color: {TEXT}; }}
            """
        )


class _FluentTabBar(QtWidgets.QTabBar):
    """Pivot tab bar that fits every tab WITHOUT scroll arrows and WITHOUT clipping a close
    ``x``, while keeping short tabs at full width.

    When the natural tab widths overflow the bar, the WIDEST tabs are capped to a shared
    width (water-filling) and ``ElideRight`` trims only their labels: ``Monitor`` / ``Logic``
    stay full, long Edit titles trim with a trailing ``...``, and every tab's close ``x``
    stays painted because the cap reserves the right-side button slot.  This is the
    squeeze-free replacement for the old ``width//count`` layout that crammed EVERY tab
    (short ones included) to an equal sliver.  ``sizeHint`` still reports the NATURAL total
    so the QTabWidget grants the bar its full window-capped width (``tabSizeHint`` caps to
    that ACTUAL width); reporting the capped sum instead would loop back and collapse the
    bar.  No tab scrolls off, so no scroll arrows appear; the corner ``...`` overflow menu
    lists every FULL title.  Reusable, frontend-owned."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Pivot tabs carry their own accent UNDERLINE; QTabBar's default "base" -- a line the
        # style strokes across the WHOLE bar width (and so grows with the tab count) -- would
        # show as a thin line on the bar's top edge, above the labels.  Turn it off.
        self.setDrawBase(False)

    def _natural_widths(self) -> list:
        # Explicit base call (NOT a no-arg super(), which fails inside a comprehension frame).
        base = QtWidgets.QTabBar.tabSizeHint
        return [base(self, i).width() for i in range(self.count())]

    def _shrink_cap(self):
        """Water-filling width cap: the largest ``cap`` such that capping every tab at it
        sums to <= the bar width.  Tabs already narrower than ``cap`` (Monitor / Logic) keep
        their natural width; only the wide Edit tabs are capped + elided.  None when all fit."""
        avail = self.width()
        widths = sorted(self._natural_widths())
        n = len(widths)
        if n == 0 or avail <= 0 or sum(widths) <= avail:
            return None
        # The cap floor is JUST the close-x slot + a sliver of label, NOT a readable-label
        # width: a larger floor would make n*floor exceed a narrow bar, so the rightmost tab
        # (and its close x) would spill past the edge -- the "x cut off" bug.  Floored only
        # this low, the tabs always sum to <= the bar (the cap falls to avail/n first), so a
        # close x is never clipped; the heavily-elided labels stay reachable via the ... menu.
        floor = scaled_px(34, minimum=28)
        prefix = 0
        for k in range(n):
            cap = (avail - prefix) / (n - k)         # share the remainder among the wide tabs
            if cap <= widths[k]:
                return max(int(cap), floor)
            prefix += widths[k]                      # this tab fits under the cap -> keeps natural
        return floor

    def tabSizeHint(self, index):  # noqa: N802 - Qt naming
        base = QtWidgets.QTabBar.tabSizeHint
        hint = base(self, index)
        cap = self._shrink_cap()
        if cap is None or hint.width() <= cap:
            return hint                              # fits (or is a short tab) -> natural width
        return QtCore.QSize(cap, hint.height())

    def sizeHint(self):  # noqa: N802 - Qt naming
        hint = QtWidgets.QTabBar.sizeHint(self)
        widths = self._natural_widths()
        return QtCore.QSize(sum(widths), hint.height()) if widths else hint

    def is_overflowing(self) -> bool:
        """True when the natural tab widths exceed the bar (some labels are being elided) --
        the cue to offer the ``...`` overflow menu of full titles."""
        widths = self._natural_widths()
        avail = self.width()
        return bool(widths) and avail > 0 and sum(widths) > avail


class FluentTabWidget(QtWidgets.QTabWidget):
    """Pivot-underline tab widget that also supports CLOSABLE tabs.

    Permanent tabs (Monitor / Logic) are added with :meth:`add_permanent_tab`
    (no close button); on-demand tabs (e.g. a per-panel Edit) are added with
    :meth:`add_closable_tab`, which shows the native close affordance and emits
    :attr:`tab_close_requested` with the page widget when its X is clicked, so a
    host can tear the page down.  Reusable across GUIs (frontend-owned)."""

    tab_close_requested = QtCore.pyqtSignal(QtWidgets.QWidget)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QtWidgets.QTabWidget.North)
        self.setTabBar(_FluentTabBar(self))     # content-width pivot tabs; elides to fit, no arrows
        self.tabBar().setExpanding(False)
        # No left/right scroll ARROWS (the native chevrons crowd the last tab's close "x"
        # and read as browser chrome, not Fluent).  With scroll buttons OFF and ElideRight,
        # Qt shrinks the OVERFLOWING tabs by eliding their labels so ALL tabs fit side by
        # side -- short tabs (Monitor / Logic) stay full, long Edit titles trim with "..."
        # and every tab's close "x" stays fully painted (the reported "x cut off" is gone).
        # The corner ``...`` overflow button lists every FULL title for navigation.
        self.setUsesScrollButtons(False)
        self.tabBar().setElideMode(QtCore.Qt.ElideRight)
        # No NATIVE close buttons.  A plain ``addTab()`` tab is PERMANENT (no X),
        # so GUIs that use ``addTab`` directly (e.g. the pulse editor's
        # Edit / Preview / Scan, which must always exist) never get a close
        # affordance.  ``add_closable_tab`` opts in a CUSTOM, subtle close button
        # (a soft grey "x" that highlights on hover -- not the harsh native X).
        self.setStyleSheet(
            f"""
            QTabWidget {{
                background: transparent;
                margin: 0px;
                padding: 0px;
            }}
            QTabWidget::pane {{
                background: white;
                margin: 0px;
                padding: 0px;
                border: none;
                border-radius: {_radius()}px;
            }}
            QTabWidget QStackedWidget {{
                background: white;
                border: none;
                margin: 0px;
                padding: 0px;
                border-radius: {_radius()}px;
            }}
            QTabWidget::tab-bar {{
                margin: 0px;
                padding: 0px;
            }}
            QTabBar {{
                background: transparent;
            }}
            QTabBar::tab {{
                background: transparent;
                color: {GREY};
                border: none;
                border-bottom: {scaled_px(2, minimum=2)}px solid transparent;
                height: {scaled_px(32, minimum=26)}px;
                padding: 0px {scaled_px(16, minimum=12)}px;
                margin-right: {scaled_px(6)}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QTabBar::tab:selected {{
                color: {TEXT};
                border-bottom: {scaled_px(2, minimum=2)}px solid {ACCENT};
            }}
            QTabBar::tab:!selected:hover {{
                color: {TEXT};
            }}
            """
        )
        self._overflow_btn = self._build_overflow_button()
        # The corner widget sits FLUSH at the widget's right edge, so the button's rounded hover
        # background used to be CLIPPED on the right.  Wrap it in a transparent holder whose right
        # margin reserves the clearance, so the full hover pill is visible.
        corner = QtWidgets.QWidget(self)
        corner.setStyleSheet("background: transparent;")
        corner_lay = QtWidgets.QHBoxLayout(corner)
        corner_lay.setContentsMargins(0, 0, scaled_px(6, minimum=4), 0)
        corner_lay.setSpacing(0)
        corner_lay.addWidget(self._overflow_btn)
        self._overflow_corner = corner
        self.setCornerWidget(corner, QtCore.Qt.TopRightCorner)
        corner.hide()
        self.currentChanged.connect(lambda _i: self._update_overflow())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # The tab widget is a plain WHITE rounded card with NO border line ANYWHERE -- the pivot tabs sit
        # on the white top, the content flows straight below, and the card is delineated only by its white
        # body against the grey window.  (Any border here read as boxing the tabs at the top, or as a line
        # between the tabs and the content below -- neither is wanted; the tabs + content are ONE surface.)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("white"))
        painter.drawRoundedRect(QtCore.QRectF(self.rect()), float(_radius()), float(_radius()))
        painter.end()
        super().paintEvent(event)

    def _build_overflow_button(self) -> QtWidgets.QToolButton:
        """The Fluent overflow affordance: a subtle ``...`` button (top-right corner) that
        replaces the native scroll arrows.  Shown only when the tabs overflow the bar;
        clicking it lists EVERY tab and jumps to the picked one (scrolling it into view)."""
        btn = QtWidgets.QToolButton(self)
        btn.setText("⋯")                          # MIDLINE HORIZONTAL ELLIPSIS
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFocusPolicy(QtCore.Qt.NoFocus)
        btn.setToolTip("All tabs")
        # The menu is popped from the `clicked` handler (the button owns no QMenu), so the
        # default (DelayedPopup) mode is correct -- InstantPopup would suppress `clicked`.
        size = scaled_px(26, minimum=20)
        btn.setFixedSize(size, size)
        font = btn.font()
        font.setPixelSize(max(11, int(size * 0.72)))
        font.setBold(True)
        btn.setFont(font)
        radius = _radius()
        btn.setStyleSheet(
            f"""
            QToolButton {{ border: none; background: transparent; color: {GREY};
                           border-radius: {radius}px; padding: 0px; }}
            QToolButton:hover {{ background: rgba(0, 0, 0, 18); color: {TEXT}; }}
            QToolButton:pressed {{ background: rgba(0, 0, 0, 34); color: {TEXT}; }}
            QToolButton::menu-indicator {{ image: none; width: 0px; }}
            """
        )
        btn.clicked.connect(self._show_overflow_menu)
        return btn

    def _tabs_overflow(self) -> bool:
        bar = self.tabBar()
        # _FluentTabBar keeps natural tab widths and scrolls on overflow; ask it whether the
        # natural widths exceed the bar (some tabs scrolled off) -- the cue to offer the
        # ``...`` menu of every tab (Qt scrolls the picked one into view).
        if hasattr(bar, "is_overflowing"):
            return bar.is_overflowing()
        return bar.count() > 0 and bar.sizeHint().width() > bar.width() + 1

    def _update_overflow(self) -> None:
        """Show the overflow ``...`` button only while the tabs do not all fit.  Run on a
        0-ms timer after a resize / tab change so the tab bar has been re-laid-out first
        (its width is stale inside the resize/insert event itself).  Visibility toggles the
        HOLDER (the margin-bearing corner widget), so a hidden overflow reserves no corner space."""
        corner = getattr(self, "_overflow_corner", None)
        if corner is not None:
            corner.setVisible(self._tabs_overflow())

    def _schedule_overflow_update(self) -> None:
        QtCore.QTimer.singleShot(0, self._update_overflow)

    def _overflow_menu(self) -> QtWidgets.QMenu:
        """Build (but do not show) the Fluent overflow menu: one action per tab that selects it on
        trigger (Qt scrolls it into view).  Split out from the popup so the click->select wiring is
        testable without a blocking ``exec_``.  The actions are NOT checkable -- a checkable action
        makes Qt reserve an indicator gutter and draw its own (un-Fluent) tick/check box; the current
        tab is already the visibly-active pivot, so no tick is needed on the list.

        The card is rounded the SAME way every other Fluent popup is (the FluentPopup Setting card,
        the FluentComboBox drop-down): a stylesheet ``border-radius`` alone leaves the menu's opaque
        SQUARE window behind the arc, so the corners outside the radius show a square white nub (and
        Windows adds a native shadow) -- reading as a hard-cornered popup next to the antialiased
        round ones.  So this uses :class:`_FluentRoundedMenu`, which is TRANSLUCENT + frameless and
        PAINTS one antialiased rounded rect (white fill + 1 px border) behind its items -- the shared
        rounded-card single source, composited so the labels still draw on top."""
        menu = _FluentRoundedMenu(self)
        for i in range(self.count()):
            act = menu.addAction(self.tabText(i))
            act.triggered.connect(lambda _c, idx=i: self.setCurrentIndex(idx))
        return menu

    def _show_overflow_menu(self) -> None:
        """Pop the overflow list below the ``...`` button -- the single overflow navigation,
        in place of left/right scroll arrows.  It sits the SAME few-px ``_popup_gap`` off the button
        that the combo drop-down / Setting popup use below THEIR anchor, so the overflow list reads as
        one of the family instead of flush against the tab strip.  (A ``QMenu.exec_`` position is the
        top-left Qt honours as-is -- unlike a combo popup, it is NOT re-flushed -- so the gap goes
        straight into the anchor point; no Move-event filter is needed here.)"""
        menu = self._overflow_menu()
        btn = self._overflow_btn
        menu.exec_(btn.mapToGlobal(QtCore.QPoint(0, btn.height() + _popup_gap())))

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._schedule_overflow_update()

    def showEvent(self, event):  # noqa: N802 - Qt naming
        super().showEvent(event)
        self._schedule_overflow_update()

    def tabInserted(self, index):  # noqa: N802 - Qt naming
        super().tabInserted(index)
        self._schedule_overflow_update()

    def tabRemoved(self, index):  # noqa: N802 - Qt naming
        super().tabRemoved(index)
        self._schedule_overflow_update()

    def add_permanent_tab(self, widget: QtWidgets.QWidget, title: str) -> int:
        """Add a fixture tab with NO close button (e.g. Monitor).  A plain
        ``addTab`` is already permanent, so this is just an explicit alias."""
        return self.addTab(widget, title)

    def add_closable_tab(self, widget: QtWidgets.QWidget, title: str, *,
                         focus: bool = True) -> int:
        """Add a tab WITH a subtle custom close button.  Clicking it emits
        :attr:`tab_close_requested` with ``widget`` so the host can tear it down."""
        index = self.addTab(widget, title)
        self.tabBar().setTabButton(index, QtWidgets.QTabBar.RightSide,
                                   self._make_close_button(widget))
        if focus:
            self.setCurrentIndex(index)
        return index

    def _make_close_button(self, widget: QtWidgets.QWidget) -> "_FluentTabCloseButton":
        """A custom-PAINTED close affordance whose "x" is guaranteed centred in
        its hover circle (a text "x" in a QToolButton drifts off-centre)."""
        btn = _FluentTabCloseButton(self)
        btn.clicked.connect(lambda: self.tab_close_requested.emit(widget))
        return btn


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
        # content = track + the 8 px track-to-label gap + the label + a few px of air; the gap
        # MUST be budgeted here too (hitButton already counts it) or a labelled switch packed
        # tight in a toolbar paints its last glyph under the next widget (the header "Selectors"
        # clip at 1.5x display scale).
        text_w = fluent_text_width(QtGui.QFontMetrics(self.font()), self.text())
        content_w = scaled_px(60) + (scaled_px(8) + text_w + scaled_px(4) if self.text() else 0)
        return QtCore.QSize(max(self.minimumWidth(), content_w), self.minimumHeight())

    def hitButton(self, pos) -> bool:
        # Only the visible switch TRACK (and its label, if any) toggles.  The widget reserves a wider
        # minimum width for column alignment, so without this the dead padding to the right of the
        # 60 px track -- which fills the form row's control cell -- would flip the switch on a click
        # that never landed on it (a click anywhere "on the row" toggling is the reported bug).
        track_w = scaled_px(60, minimum=48)
        hit_w = track_w
        if self.text():
            hit_w = track_w + scaled_px(8) + fluent_text_width(QtGui.QFontMetrics(self.font()), self.text())
        return 0 <= int(pos.x()) <= int(hit_w) and 0 <= int(pos.y()) <= self.height()

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
        # QAbstractButton already gates toggling on the left button + enabled state,
        # so the release is forwarded unconditionally.
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


class _WheelFocusGuardMixin:
    """Scroll-over protection for numeric spin controls: the wheel only changes the value
    AFTER the widget has been clicked into (has keyboard focus).  Without focus the wheel
    event is ignored and bubbles up to the parent scroll area, so scrolling a Setting / Edit
    page that happens to pass over a spinbox scrolls the PAGE instead of silently nudging the
    number -- the误触 the user hit.  Mirrors ``FluentComboBox.wheelEvent`` (ignore -> bubble)
    so every numeric control behaves the same way.  ``StrongFocus`` is what makes a left click
    grab focus (Tab too); the real gate is the ``hasFocus`` check below, not the policy."""

    def _install_wheel_focus_guard(self) -> None:
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()         # not focused -> let the parent scroll area scroll the page
            return
        super().wheelEvent(event)  # focused (clicked into) -> normal value step


class FluentSpinBox(_WheelFocusGuardMixin, QtWidgets.QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._install_wheel_focus_guard()
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


class FluentCodeEdit(QtWidgets.QPlainTextEdit):
    """A monospace code/expression editor with the house Fluent chrome (white card, light
    border, Consolas font, Fluent scrollbars).  ONE source for the look so every code editor
    -- the pulse Scan-tab generator + table view, the panel source-expression popup -- renders
    identically; changing the recipe here changes them all."""

    def __init__(self, text: str = "", parent=None, *, read_only: bool = False):
        super().__init__(parent)
        if text:
            self.setPlainText(text)
        self.setReadOnly(read_only)
        pad = scaled_px(4)
        self.setStyleSheet(
            f'QPlainTextEdit {{ background: white; color: {TEXT}; border: 1px solid {PLACEHOLDER}; '
            f'border-radius: {scaled_px(4)}px; font: {fluent_font_size()}pt "Consolas", "Courier New", '
            f'monospace; padding: {pad}px; }}')
        apply_fluent_scrollbars(self)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _apply_fluent_context_menu(self, event)   # Fluent right-click (undo/cut/copy/paste), not the native menu


class FluentFloatingEditor(QtWidgets.QDialog):
    """A NON-modal floating multi-line editor: a prompt label over a large
    :class:`FluentCodeEdit`, with **Apply + Close BELOW the box**.

    Non-modal on purpose: the panel/frame behind it stays VISIBLE and INTERACTIVE, and
    Apply updates it LIVE (you watch the plot change without dismissing the editor).  It is
    parented to the caller's TOP-LEVEL window, so it shares that window's screen + device
    scale -- this is what prevents the "one widget scaled, the other not" DPI bug a
    differently-parented top-level dialog hits.  ``applied`` fires with the current text on
    every Apply; the caller writes it back and re-renders."""

    applied = QtCore.pyqtSignal(str)

    def __init__(self, prompt: str, text: str = "", parent=None, *, title: str = "",
                 min_size: tuple[int, int] = (560, 260)):
        # Qt.Window: a real top-level the user can move/resize, but OWNED by ``parent`` so it
        # rides the parent's screen/scale and is destroyed when the parent closes.
        super().__init__(parent, QtCore.Qt.Window | QtCore.Qt.WindowTitleHint
                         | QtCore.Qt.WindowCloseButtonHint)
        if title:
            self.setWindowTitle(title)
        self.setModal(False)                      # NON-modal: frame stays usable behind it
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setFont(QtGui.QFont(FONT, fluent_font_size()))
        self.setStyleSheet("QDialog { background: white; }")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(scaled_px(12), scaled_px(12), scaled_px(12), scaled_px(12))
        layout.setSpacing(scaled_px(8, minimum=5))

        layout.addWidget(FluentLabel(prompt, self))
        self._edit = FluentCodeEdit(str(text), self)
        self._edit.setMinimumSize(scaled_px(int(min_size[0]), minimum=int(min_size[0] * 0.8)),
                                  scaled_px(int(min_size[1]), minimum=int(min_size[1] * 0.7)))
        layout.addWidget(self._edit, 1)

        # Apply + Close UNDER the editor (Apply = write back + re-render, keep open; Close =
        # dismiss).  Apply is the comfortable place to commit a long expression.
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        apply = FluentButton("Apply", self, color=GREEN)
        close = FluentButton("Close", self, color=GREY)
        apply.clicked.connect(lambda: self.applied.emit(self._edit.toPlainText()))
        close.clicked.connect(self.close)
        btn_row.addWidget(apply)
        btn_row.addWidget(close)
        layout.addLayout(btn_row)

    def text(self) -> str:
        return self._edit.toPlainText()


class FluentDoubleSpinBox(_WheelFocusGuardMixin, QtWidgets.QDoubleSpinBox):
    """Confocal_GUIv2-style double spinbox with an inline step editor."""

    def __init__(self, length=5, allow_minus: bool = False, parent=None):
        super().__init__(parent)
        self._install_wheel_focus_guard()
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

    hidden = QtCore.pyqtSignal()   # fires on close OR hide/minimize (event-loop quit)
    closed = QtCore.pyqtSignal()   # fires ONLY on a genuine close, never on hide/minimize
    close_requested = QtCore.pyqtSignal()  # fires on the X/close (before hide-or-destroy), NOT on minimize

    def __init__(
        self,
        *,
        widget: QtWidgets.QWidget,
        title: str = "",
        hide_on_close: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._hide_on_close = bool(hide_on_close)
        self.setObjectName("FluentWindow")
        self.setStyleSheet(f"QWidget#FluentWindow {{ background: {BG}; }}")

        self._zlc_title_bar = None
        self._zlc_title = None
        if StandardTitleBar is not None:
            title_bar = StandardTitleBar(self)
            title_bar.iconLabel.setFixedSize(0, 0)
            # FluentWindow OWNS the title rendering.  StandardTitleBar positions its
            # own title label via an internal layout whose left inset is version
            # dependent AND is re-activated by any later title set -- e.g. the pulse
            # GUI's _set_gui_title() calls titleBar.setTitle() right after
            # setWindowTitle(), which kept undoing an external alignment (console,
            # set once, looked fine; pulse, re-set at runtime, floated to the edge).
            # So HIDE the built-in label and draw the title ourselves with a FREE
            # child label pinned to the body content column -- it is NOT in the title
            # bar layout, so no title-setting path and no qframelesswindow version
            # can move it.  Fix once here => every FluentWindow GUI is fixed alike.
            try:
                title_bar.titleLabel.hide()
            except Exception:  # pragma: no cover - defensive across versions
                pass
            self.setTitleBar(title_bar)
            self._zlc_title_bar = title_bar
            self.setWindowTitle(title)
            self._zlc_title = QtWidgets.QLabel(title, title_bar)
            self._zlc_title.setObjectName("zlcWindowTitle")
            self._zlc_title.setStyleSheet(
                f'QLabel#zlcWindowTitle {{ background: transparent; color: {TEXT};'
                f' font: {fluent_font_size()}pt "{FONT}"; }}')
            self._zlc_title.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self._zlc_title.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
            self._zlc_title.show()
            self.windowTitleChanged.connect(self._zlc_title.setText)
            top_margin = scaled_px(32)
        else:
            self.setWindowTitle(title)
            top_margin = 0

        if widget is None:
            raise ValueError("FluentWindow needs a constructed widget.")
        self.loaded = widget
        self.loaded.setParent(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, top_margin, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.loaded)
        if StandardTitleBar is not None:
            self.titleBar.raise_()
        self.loaded.adjustSize()
        self.resize(max(scaled_px(900, minimum=680), self.loaded.width()), self.loaded.height() + top_margin)

    def _position_window_title(self) -> None:
        """Keep FluentWindow's own title label pinned to the body content column.

        The label spans from ``TITLE_LEFT_INSET`` to just left of the first window
        control button, vertically centred in the 32 px bar.  Being a free child of
        the title bar (NOT in its layout), it is immune to title-set churn and to
        qframelesswindow's version-dependent default insets -- the title therefore
        sits on the SAME left column as the body content on every GUI and DPR."""
        lab = getattr(self, "_zlc_title", None)
        tb = getattr(self, "_zlc_title_bar", None)
        if lab is None or tb is None:
            return
        try:
            x = scaled_px(TITLE_LEFT_INSET)
            right = tb.width()
            for child in tb.findChildren(QtWidgets.QAbstractButton):
                if child.isVisible() and child.width() > 0:
                    right = min(right, child.x())
            lab.setGeometry(x, 0, max(0, right - x - scaled_px(8)), tb.height())
            lab.raise_()
        except Exception:  # pragma: no cover - defensive across qframelesswindow versions
            pass

    def showEvent(self, event):  # noqa: N802 - Qt naming
        super().showEvent(event)
        self._position_window_title()
        # the title bar / buttons may finish their geometry one event-loop tick
        # after show; re-pin then so the right edge is computed against real button
        # positions
        QtCore.QTimer.singleShot(0, self._position_window_title)

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._position_window_title()

    def closeEvent(self, event):
        self.close_requested.emit()   # the X was pressed (NOT a minimize); fires whether we hide or destroy
        self.hidden.emit()
        if self._hide_on_close:
            event.ignore()
            self.hide()
            return
        self.closed.emit()   # a genuine close (NOT hide/minimize): hosts release resources here
        super().closeEvent(event)

    def hideEvent(self, event):
        self.hidden.emit()
        super().hideEvent(event)


def launch_fluent_window(
    widget: QtWidgets.QWidget,
    *,
    title: str,
    hide_on_close: bool = False,
    fixed_size: bool = True,
    size: tuple | None = None,
    wire=None,
) -> FluentWindow:
    """The ONE top-level-GUI launcher sequence: wrap ``widget`` in a :class:`FluentWindow`, run the
    caller's ``wire(window)`` (per-GUI signal wiring between construction and show: close ->
    teardown, title propagation), size it, centre it on the primary screen, show it, and retain it
    on the ONE app registry (:func:`retain_window`).

    EVERY ``show_*`` entry point (pulse editor / task console / figure viewer / device manager)
    routes through here -- the fourth hand-copied launcher (the device manager) had silently
    dropped ensure_qt_app / the shared scale / centring / retention, exactly the drift class this
    helper removes (same failure family as the duplicated scale rule, commit 57a1d25).

    ``fixed_size=True`` locks the window to its content hint (these GUIs size their own body from
    the screen, so the window must not be user-resizable past it); ``size=(w, h)`` (with
    ``fixed_size=False``) is the explicit-size path for a body whose hint would collapse (the
    device manager's scrolling list).  The caller must still :func:`ensure_qt_app` BEFORE
    constructing ``widget`` (a QWidget needs the QApplication; the widget ctor also resolves the
    shared fluent scale) -- it is re-asserted here so the sequence stays safe for a caller that
    forgot."""
    app = ensure_qt_app()
    window = FluentWindow(widget=widget, title=title, hide_on_close=hide_on_close)
    if wire is not None:
        wire(window)
    if fixed_size:
        window.adjustSize()
        window.setFixedSize(window.size())
    elif size is not None:
        window.resize(int(size[0]), int(size[1]))
    center_window_on_primary_screen(window, app)
    window.show()
    retain_window(window)   # the ONE app-level registry; pruned when the window dies
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
        self._api = False        # api-slot state (violet); scan state is isChecked() (orange)
        diameter = Metrics.dot()
        self.setFixedSize(diameter, diameter)
        self.setToolTip(tooltip)

    def set_number(self, number: int | None) -> None:
        self._number = None if number is None else int(number)
        self.update()

    def set_api(self, api: bool) -> None:
        """Mark the dot as an API slot (violet) -- distinct from a scan slot (orange,
        ``setChecked``).  An API-slot field keeps its value, so only the dot recolours."""
        self._api = bool(api)
        self.update()

    def nextCheckState(self) -> None:  # noqa: N802 - Qt API name
        # The dot's checked (scan) state is DRIVEN BY THE MODEL via set_scan_bound, not by the
        # click toggle: a click only emits ``clicked`` to drive the none->scan->api->off cycle,
        # which then re-applies the correct visual.  Without this, the auto-toggle would leave a
        # just-cleared dot stuck "checked" (orange) when the cycle landed on API or off.
        pass

    def number(self) -> int | None:
        return self._number

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = min(self.width(), self.height()) / 2.0 - max(1.0, scaled_px(1))
        if self.isChecked() or self._api:
            painter.setBrush(QtGui.QColor(API_VIOLET if self._api else ORANGE))
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


def _bound_field_style(*, selector: str, text: str, border: str, fill: str | None = None) -> str:
    """The ONE stylesheet recipe for a scan/api BOUND field: a coloured border + text (+ optional
    background ``fill``) with the shared radius / padding / font.  ``selector`` differs (a scan mark
    also covers combos, an api mark only the line edit); ``fill`` is ``None`` for the api border-only
    look.  Qt ignores CSS whitespace, so this single canonical form renders identically to the
    previously hand-written orange (multi-line) / api (inline) stylesheets (#D1)."""
    bg = f"background: {fill}; " if fill else ""
    return (
        f"{selector} {{ {bg}color: {text}; border: 1px solid {border}; "
        f"border-radius: {_radius()}px; "
        f"padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px; "
        f'font: {fluent_font_size()}pt "{FONT}"; }}'
    )


def mark_scan_field(widget: QtWidgets.QWidget, *, bound: bool) -> None:
    """Apply (or clear) the orange disabled look on a field bound to a scan slot."""

    widget.setProperty("zlcScanBound", bool(bound))
    widget.setEnabled(not bound)
    if bound:
        widget.setStyleSheet(_bound_field_style(
            selector="QLineEdit, QComboBox", text=ORANGE_DARK, border=ORANGE, fill=ORANGE_TINT))
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
        if bound:                      # scan and api are mutually exclusive on a field
            self._api_bound = False
        self._dot.set_api(False)
        self._dot.setChecked(self._bound)
        self._dot.set_number(number if self._bound else None)
        self.setReadOnly(self._bound)
        if self._bound:
            self.setStyleSheet(_bound_field_style(
                selector="QLineEdit", text=ORANGE_DARK, border=ORANGE, fill=ORANGE_TINT))
        else:
            self.setStyleSheet(self._base_style)
        self._reserve_right()
        self.update()

    def set_api_bound(self, bound: bool, number: int | None = None) -> None:
        """Mark (or clear) this field as an API slot ``a{number}``.

        UNLIKE :meth:`set_scan_bound`, an API slot KEEPS the field's value visible and
        EDITABLE -- only a thin violet border + the violet dot mark it (the value is a real
        number the API can also set by name).  Scan and API are mutually exclusive."""
        bound = bool(bound)
        if getattr(self, "_api_bound", False) == bound and getattr(self, "_api_number", "\x00unset") == number:
            return
        self._api_bound = bound
        self._api_number = number
        if bound:                      # leaving any scan state
            self._bound = False
            self._scan_number = None
            self._dot.setChecked(False)
            self.setReadOnly(False)
        self._dot.set_api(bound)
        self._dot.set_number(number if bound else None)
        if bound:
            self.setStyleSheet(_bound_field_style(
                selector="QLineEdit", text=API_VIOLET_DARK, border=API_VIOLET, fill=None))
        elif not getattr(self, "_bound", False):
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
    "FluentScrollArea",
    "FluentStatusDot",
    "FluentWindow",
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
    "launch_fluent_window",
    "retain_window",
    "resolve_fluent_auto_scale",
    "scaled_px",
    "window_pad",
    "WINDOW_PAD",
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
