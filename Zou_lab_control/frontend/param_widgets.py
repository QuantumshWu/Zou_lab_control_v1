"""Legacy five-operation parameter facade over the current frontend owners.

The knowledge "a :class:`ParamDecl` of kind ``K`` is built / read / seeded /
validated / refreshed as widget ``W``" used to live in 5-7 parallel ladders inside
``task_console.py`` (the measurement form's build / collect / seed / required /
refresh / set_running loops) PLUS a SECOND, smaller ladder behind a parallel
``ParamSpec`` declaration class for plot-panel params.  Adding a kind meant editing
every ladder; forgetting one was a silent bug.

This module originally collapsed all of that to one handler per kind.  During
migration, ordinary ``float``/``int``/``bool``/``text``/``choice`` rows now project
to :class:`zlc_frontend.form.FormFieldProps` and delegate to
``zlc_frontend.qt_widgets.FORM_WIDGET_HANDLERS``.  Therefore bounds, parsing,
typed choices, and non-quantizing float edits have one current owner.  This file
retains the old five-operation call surface while unported consumers still exist,
plus genuinely legacy composite/dynamic controls.  Those five operations are:

  * ``build(decl, value, ctx)``  -- construct the Qt widget, seed it with ``value``,
    and wire ``ctx.on_change`` (re-validation) + ``ctx.instant_apply`` (the Setting /
    Edit "apply on edit" path) from the supplied :class:`ParamWidgetContext`.
  * ``read(widget)``             -- coerce the widget's current value BY KIND.  Never
    ``eval``s free text (the confocal-GUI lesson): a ``text`` / ``path`` value is
    taken verbatim.
  * ``write(widget, value)``     -- seed / prefill the widget from a saved value.
  * ``is_empty(widget)``         -- True when a ``required`` field is unset (a blank
    line-edit / unpicked combo); always False for a control that always holds a value
    (a spin box, a switch).
  * ``refresh(widget, providers)`` -- repopulate a DYNAMIC control (a signal /
    pulse-template combo) from live providers; a no-op for a static kind.

Do not add an ordinary scalar widget implementation here.  Add it to the current
headless/Qt form owners, then add only a projection entry while legacy consumers
remain.  A legacy-only composite kind still needs one local handler registered in
:data:`PARAM_WIDGETS`.  ``tests/test_param_widget_registry.py`` mechanically
enforces coverage and the five-op facade.

This is a FRONTEND module: it may import Qt and the frontend's own fluent widgets.
``ParamDecl`` itself stays dependency-free in ``operations`` -- this registry is the
GUI-side consumer that interprets a declaration, exactly as the docstring of
``ParamDecl`` says ("the spec consumer validates / coerces by ``kind``").
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PyQt5 import QtCore, QtWidgets

from Zou_lab_control._paths import display_path

from zlc_frontend.form import FormChoice, FormFieldProps
from zlc_frontend.qt_widgets import (
    FORM_WIDGET_HANDLERS,
    GREY,
    FluentDoubleSpinBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentTreeComboBox,
    FluentTriStateToggle,
    eng_mantissa_prefix,
    scaled_px,
    signals_blocked as _signals_blocked,
)


def format_reading(decl, value) -> str:
    """The ONE way a live read-back value is rendered for display (a device viewer's "Current" line,
    a latest-value meter), so every read-out reads the same and the unit substitution is never
    re-implemented per call site.

    A numeric quantity is engineering-scaled with its unit SI-prefixed -- ``6.83e9`` Hz -> ``"6.83 GHz"``,
    ``5e-3`` s -> ``"5 ms"`` (confocal ``float2str_eng``, the prefix riding the unit); anything else is
    ``str``'d.  The declared ``unit`` (if any) is appended.  The editable spin box stays plain decimal
    (confocal does not SI-scale its editor) -- this is the read-out side only."""
    if getattr(decl, "is_numeric", False) and isinstance(value, (int, float)) and not isinstance(value, bool):
        mantissa, prefix = eng_mantissa_prefix(float(value))
        unit = getattr(decl, "unit", "") or ""
        return f"{mantissa} {prefix}{unit}".rstrip() if unit else f"{mantissa}{prefix}"
    text = str(value)
    unit = getattr(decl, "unit", "") or ""
    return f"{text} {unit}" if unit else text


def _noop() -> None:
    """The default ``on_change`` -- a form that does not re-validate (e.g. a read-only
    source view) leaves it unset; handlers always have something callable to connect."""
    return None


# --------------------------------------------------------------------------- context


@dataclass
class ParamWidgetContext:
    """The small bundle a handler needs to BUILD a widget, kept decoupled from the
    console.  A form fills in only the callbacks its widgets actually use:

    ``on_change``      a NO-ARG callable connected to every widget's change signal so
                       the form re-validates (enables / disables Start) on each edit.
    ``instant_apply``  optional ``(key, value) -> None`` for the "apply on edit" path
                       (the Setting popup / plot Edit tab push the value straight into
                       ``config.params`` as the user edits); the measurement form
                       leaves this None and reads back on Start via :meth:`read`.
    ``signals_provider`` / ``sources_provider`` / ``formats_provider``
                       the live-hub-signal name list + producer / format maps the
                       grouped signal pickers (kinds ``signal`` / ``signal_expr``) need.
    ``signal_expr_factory`` / ``pulse_slots_factory``
                       zero-arg factories that build the two COMPOSITE widgets that
                       live in ``task_console`` (``_SignalExprWidget`` /
                       ``_PulseSlotsWidget``) -- injected so this module needn't import
                       them (it stays a leaf the console depends on, not vice versa).
    """

    on_change: Callable[[], None] = _noop
    instant_apply: Optional[Callable[[str, Any], None]] = None
    signals_provider: Optional[Callable[[], Any]] = None
    sources_provider: Optional[Callable[[], Any]] = None
    formats_provider: Optional[Callable[[], Any]] = None
    labels_provider: Optional[Callable[[], Any]] = None
    signal_expr_factory: Optional[Callable[[str], QtWidgets.QWidget]] = None
    pulse_slots_factory: Optional[Callable[[], QtWidgets.QWidget]] = None

    def names(self) -> list[str]:
        if callable(self.signals_provider):
            try:
                return [str(n) for n in self.signals_provider()]
            except Exception:
                return []
        return []

    def sources(self) -> dict:
        return self.sources_provider() if callable(self.sources_provider) else {}

    def formats(self) -> dict:
        return self.formats_provider() if callable(self.formats_provider) else {}

    def labels(self) -> dict:
        """The SHORT-name map ({full hub name -> short name}) the grouped picker uses as ``labels`` so a
        leaf reads "frame_0", NOT the prefix-stripped "0" -- so a ``signal``-kind picker renders the SAME
        as the plot Setting / signal_expr pickers (#combo-parity, ``coerce_short_labels`` below)."""
        return coerce_short_labels(self.labels_provider)


#: Minimum spacing between two live "apply on edit" writes of the SAME key (ms).  200 ms = at most
#: 5 writes/second, matching the device viewer's 200 ms read-back poll: a mouse-wheel scroll on a
#: spin box in "Live" mode fires a valueChanged per tick, and an EXPENSIVE / blocking apply (a live
#: device set-point) on every one of those can wedge the GUI -- this caps the rate.  A module-level
#: constant (the repo's *_DEBOUNCE_MS convention) so it stays a single source.
LIVE_WRITE_MIN_INTERVAL_MS = 200


class RateLimitedApply:
    """Rate-limit an ``(key, value) -> None`` apply-on-edit callback -- LEADING + TRAILING, per key.

    A fast mouse-wheel scroll fires ``valueChanged`` per tick; routing an expensive / blocking apply
    (a live device write, a re-render) through every one can freeze the GUI.  This wraps such a
    callback so, PER KEY: the FIRST edit applies immediately (responsive -- you see the value move),
    further edits within ``interval_ms`` are coalesced, and the LATEST value is applied when the
    window elapses.  A pure trailing debounce (the repo's ``*_DEBOUNCE_MS`` timers) would give no live
    feedback until the scroll stops; a pure leading throttle would DROP the final value.  Leading +
    trailing gives both: at most one apply per window AND the final value always lands.

    Timers are parented to ``parent`` so they die with it; :meth:`flush` applies any pending values
    NOW (teardown / an explicit Apply / a test that must observe the trailing edge without pumping the
    event loop).  Reusable: any apply-on-edit path can wrap its callback in this."""

    def __init__(self, apply: Callable[[str, Any], None], *, parent: QtCore.QObject,
                 interval_ms: int = LIVE_WRITE_MIN_INTERVAL_MS) -> None:
        self._apply = apply
        self._parent = parent
        self._interval_ms = max(1, int(interval_ms))
        self._pending: dict = {}        # key -> latest value awaiting the trailing edge
        self._timers: dict = {}         # key -> its single-shot QTimer (window open while active)

    def __call__(self, key, value) -> None:
        timer = self._timers.get(key)
        if timer is None or not timer.isActive():
            self._apply(key, value)                      # leading edge: apply now, open the window
            if timer is None:
                timer = QtCore.QTimer(self._parent)
                timer.setSingleShot(True)
                timer.timeout.connect(lambda k=key: self._flush_key(k))
                self._timers[key] = timer
            self._pending.pop(key, None)
            timer.start(self._interval_ms)
        else:
            self._pending[key] = value                   # inside the window: keep only the latest

    def _flush_key(self, key) -> None:
        if key in self._pending:
            self._apply(key, self._pending.pop(key))     # trailing edge: apply the final value
            self._timers[key].start(self._interval_ms)   # there was activity -> keep the window open
        # else the window closes; the next edit is a fresh leading edge

    def flush(self) -> None:
        """Apply every pending trailing value immediately (teardown / explicit Apply / a test)."""
        for key in list(self._pending):
            self._apply(key, self._pending.pop(key))


@dataclass
class RefreshProviders:
    """What :meth:`ParamWidgetHandler.refresh` needs to repopulate a dynamic control.

    ``signals`` / ``sources`` / ``formats`` populate the grouped signal pickers.
    ``repopulate`` is a per-widget hook the form supplies for the ``pulse_slots``
    composite whose choices come from a sibling template
    field -- the inter-field reactivity stays owned by the form (it knows the sibling
    layout); the handler just calls it.
    """

    signals: list[str]
    sources: dict
    formats: dict
    labels: dict = field(default_factory=dict)   # short-name map -> picker leaf "frame_0" not "0"
    repopulate: Optional[Callable[[QtWidgets.QWidget], None]] = None


# --------------------------------------------------------------------- abstract handler


class ParamWidgetHandler(ABC):
    """One param kind's widget contract.  ABSTRACT: a concrete handler that omits ANY
    of the five operations cannot be instantiated (the abstractmethods below), which is
    exactly the guard ``test_param_widget_registry`` relies on."""

    @abstractmethod
    def build(self, decl, value, ctx: ParamWidgetContext) -> QtWidgets.QWidget:
        """Construct + seed + wire the widget for ``decl`` at initial ``value``."""

    @abstractmethod
    def read(self, widget) -> Any:
        """Coerce the widget's current value BY KIND (never eval free text)."""

    @abstractmethod
    def write(self, widget, value) -> None:
        """Seed / prefill the widget from a saved value (shape mismatch -> ignore)."""

    @abstractmethod
    def is_empty(self, widget) -> bool:
        """True when a ``required`` field is unset (for missing-required validation)."""

    @abstractmethod
    def refresh(self, widget, providers: RefreshProviders) -> None:
        """Repopulate a dynamic control from live providers; no-op for static kinds."""


# A static control (spin / switch / static combo / line edit) is never "empty" in the
# required-field sense -- it always holds a value -- and has nothing to repopulate.
class _StaticMixin:
    def is_empty(self, widget) -> bool:  # noqa: D401 - simple predicate
        return False

    def refresh(self, widget, providers: RefreshProviders) -> None:
        return None


def _wire(widget_signal, ctx: ParamWidgetContext, decl, reader: Callable[[], Any]) -> None:
    """Connect a widget's change signal to BOTH the form's re-validation
    (``ctx.on_change``) AND the optional instant-apply path (``ctx.instant_apply``),
    so ONE wiring rule serves the read-on-Start form and the apply-on-edit Setting."""
    def _on(*_a):
        if ctx.instant_apply is not None:
            ctx.instant_apply(decl.key, reader())
        ctx.on_change()
    widget_signal.connect(_on)


# ------------------------------------------------------------------------------- scalars


_CURRENT_FIELD_ATTR = "_zlc_current_form_field"


def _integral_bound(value, *, key: str, name: str) -> int:
    """Translate a legacy numeric bound without silently truncating it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"legacy int field {key!r} {name} must be numeric")
    result = int(value)
    if result != value:
        raise ValueError(f"legacy int field {key!r} {name} must be integral")
    return result


def _legacy_scalar_projection(decl, *, value=None, current_kind=None) -> FormFieldProps:
    """Losslessly project an old scalar declaration into the current form owner.

    This is deliberately an adapter, not a second widget registry.  The current
    :mod:`zlc_frontend` field and handler contracts perform all scalar typing,
    bounds checking, non-quantizing float editing, and ordinary-choice identity.
    Legacy-only render facts that the current scalar contract cannot express must
    either be handled explicitly (the segmented choice below) or fail here.
    """
    kind = str(current_kind or decl.kind)
    if kind not in FORM_WIDGET_HANDLERS:
        raise ValueError(f"legacy scalar adapter cannot project kind {kind!r}")

    default = decl.default
    minimum = maximum = None
    choices = ()
    if kind == "int":
        minimum = _integral_bound(decl.lo, key=decl.key, name="minimum")
        maximum = _integral_bound(decl.hi, key=decl.key, name="maximum")
    elif kind == "float":
        minimum = decl.lo
        maximum = decl.hi
    elif kind == "choice":
        choices = tuple(
            FormChoice(
                label="(none)" if decl.kind == "device" and item == "" else str(item),
                value=item,
            )
            for item in decl.choices
        )
        if decl.kind == "device" and default is None and any(
            choice.value == "" for choice in choices
        ):
            default = ""
    elif kind == "bool" and default is None:
        # The old BoolHandler's declared no-default state was concretely False.
        default = False

    if kind in {"int", "float"}:
        if decl.blank_allowed and default is not None:
            raise ValueError(
                f"legacy field {decl.key!r} requests a blank-able editor but has a "
                "concrete default; current FormFieldProps cannot express both facts"
            )
        if not decl.blank_allowed and default is None:
            # Current scalar handlers select a bounded spin only for a concrete
            # initial value.  Runtime controls historically carry that value in
            # ``build(..., value, ...)`` rather than ParamDecl.default.  Preserve
            # the spin contract without inventing a new legacy widget owner.
            candidate = value
            if candidate is None:
                candidate = min(max(0, minimum), maximum)
            default = candidate

    return FormFieldProps(
        key=str(decl.key),
        kind=kind,
        label=str(decl.label or decl.key),
        default=default,
        required=bool(decl.required),
        unit=str(decl.unit),
        description=str(decl.tooltip),
        minimum=minimum,
        maximum=maximum,
        choices=choices,
    )


class _CurrentScalarAdapter(ParamWidgetHandler):
    """Keep the old five-op call surface while delegating scalar truth to current."""

    def __init__(self, kind: str, *, legacy_kind: str | None = None) -> None:
        if kind not in FORM_WIDGET_HANDLERS:
            raise ValueError(f"unknown current form kind: {kind!r}")
        self._kind = kind
        self._legacy_kind = legacy_kind or kind
        self._handler = FORM_WIDGET_HANDLERS[kind]

    def build(self, decl, value, ctx):
        if decl.kind != self._legacy_kind:
            raise TypeError(
                f"{self._legacy_kind} adapter cannot build legacy kind {decl.kind!r}"
            )
        field = _legacy_scalar_projection(
            decl, value=value, current_kind=self._kind
        )
        initial = field.default if value is None else value
        holder: dict[str, QtWidgets.QWidget] = {}

        def _on_change() -> None:
            widget = holder.get("widget")
            if widget is None:
                raise RuntimeError("current form handler emitted while still building")
            if ctx.instant_apply is not None:
                ctx.instant_apply(decl.key, self._handler.read(field, widget))
            ctx.on_change()

        widget = self._handler.build(field, initial, _on_change)
        setattr(widget, _CURRENT_FIELD_ATTR, field)
        holder["widget"] = widget
        return widget

    def _field(self, widget) -> FormFieldProps:
        field = getattr(widget, _CURRENT_FIELD_ATTR, None)
        if not isinstance(field, FormFieldProps) or field.kind != self._kind:
            raise TypeError(
                f"widget was not built by the legacy {self._kind} scalar adapter"
            )
        return field

    def read(self, widget):
        field = self._field(widget)
        return self._handler.read(field, widget)

    def write(self, widget, value):
        field = self._field(widget)
        self._handler.write(field, widget, value)

    def is_empty(self, widget) -> bool:
        field = self._field(widget)
        return self._handler.is_empty(field, widget)

    def refresh(self, widget, providers: RefreshProviders) -> None:
        del providers
        field = self._field(widget)
        self._handler.refresh(field, widget)


class ChoiceHandler(ParamWidgetHandler):
    """Current typed combo by default; one explicit legacy segmented render adapter."""

    def __init__(self) -> None:
        self._ordinary = _CurrentScalarAdapter("choice")

    def build(self, decl, value, ctx):
        if not bool(getattr(decl, "segmented", False)):
            return self._ordinary.build(decl, value, ctx)
        choices = tuple(str(choice) for choice in decl.choices)
        if not choices or len(set(choices)) != len(choices):
            raise ValueError(
                f"legacy segmented choice {decl.key!r} needs unique choices"
            )
        widget = FluentTriStateToggle(choices)
        widget._zlc_legacy_segmented_choices = choices
        current = decl.default if value is None else value
        if current is not None:
            text = str(current)
            if text not in choices:
                raise ValueError(
                    f"legacy segmented choice {decl.key!r} value is not declared"
                )
            widget.setCurrentText(text)
        widget.setToolTip(decl.tooltip)
        _wire(widget.activated, ctx, decl, lambda: widget.currentText())
        return widget

    def read(self, widget):
        if isinstance(widget, FluentTriStateToggle):
            return widget.currentText()
        return self._ordinary.read(widget)

    def write(self, widget, value):
        if isinstance(widget, FluentTriStateToggle):
            text = str(value)
            choices = getattr(widget, "_zlc_legacy_segmented_choices", ())
            if text not in choices:
                raise ValueError("value is not a declared legacy segmented choice")
            widget.setCurrentText(text)
            return
        self._ordinary.write(widget, value)

    def is_empty(self, widget) -> bool:
        if isinstance(widget, FluentTriStateToggle):
            return not widget.currentText()
        return self._ordinary.is_empty(widget)

    def refresh(self, widget, providers: RefreshProviders) -> None:
        if isinstance(widget, FluentTriStateToggle):
            return None
        self._ordinary.refresh(widget, providers)


class JsonHandler(ParamWidgetHandler):
    """A JSON literal in a line edit -- a list / mapping / number (e.g. a device config's
    channel list ``["ch00", "ch01"]``).  Typed: parsed with ``json.loads``, NEVER ``eval``'d;
    blank = ``None`` (the consumer's own default applies).  The instant-apply path only
    forwards a value while the text PARSES (mid-edit garbage keeps the last good value);
    the explicit :meth:`read` raises ``ValueError`` so a form's gather can report the field
    by name and block its Apply."""

    @staticmethod
    def _dumps(value) -> str:
        return "" if value is None else json.dumps(value)

    def build(self, decl, value, ctx):
        seed = value if value is not None else decl.default
        edit = FluentLineEdit(self._dumps(seed))
        edit.setMinimumWidth(scaled_px(160, minimum=120))
        edit.setPlaceholderText('JSON, e.g. ["ch11"] / {"x": 1}')
        edit.setToolTip(decl.tooltip)

        def _on(*_a):
            if ctx.instant_apply is not None:
                try:
                    ctx.instant_apply(decl.key, self.read(edit))
                except ValueError:
                    pass                       # unparseable mid-edit: hold the last good value
            ctx.on_change()

        edit.textChanged.connect(_on)
        return edit

    def read(self, widget):
        text = widget.text().strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON ({exc.msg})") from exc

    def write(self, widget, value):
        widget.setText(self._dumps(value))

    def is_empty(self, widget) -> bool:
        return not widget.text().strip()

    def refresh(self, widget, providers: RefreshProviders) -> None:
        return None


class DeviceRefHandler(_CurrentScalarAdapter):
    """A ``$device:<entry>`` cross-reference to another entry of the SAME device config (a
    constructor arg that takes a built device).  Same combo semantics as ``choice`` -- the
    CONFIG EDITOR fills ``choices`` from the working config's other entry names (the decl
    itself may carry none; only the editor knows the live config) -- but an unset required
    reference counts as MISSING (a choice is never empty)."""

    def __init__(self) -> None:
        super().__init__("choice", legacy_kind="device")

    def is_empty(self, widget) -> bool:
        return not str(self.read(widget) or "").strip()


class PathHandler(_StaticMixin, ParamWidgetHandler):
    """A filesystem path: a line edit + Browse button (the one reusable picker).  Taken
    verbatim, never eval'd; the displayed path is absolute / project-anchored."""

    def build(self, decl, value, ctx):
        seed = decl.default if value is None else value
        picker = FluentPathEdit(
            display_path(seed),
            mode=getattr(decl, "path_mode", "file"),
            caption=f"Choose {decl.key}",
            file_filter=getattr(decl, "file_filter", "All files (*)"),
            base_dir=display_path(getattr(decl, "base_dir", "")))
        picker.setToolTip(decl.tooltip)
        _wire(picker.changed, ctx, decl, lambda: picker.text())
        return picker

    def read(self, widget):
        return widget.text()

    def write(self, widget, value):
        widget.setText(display_path(value))

    def is_empty(self, widget) -> bool:
        return not widget.text().strip()


# ----------------------------------------------------------------------- axis_range


def _make_axis_spin(decl, value=None) -> FluentDoubleSpinBox:
    """Build one coordinate editor for the legacy three-part axis-range control."""
    digits = max(5, len(str(int(abs(decl.hi) + 1))) + 4)
    spin = FluentDoubleSpinBox(length=digits, allow_minus=float(decl.lo) < 0)
    spin.setRange(float(decl.lo), float(decl.hi))
    if value is not None:
        spin.setValue(float(value))
    spin.setToolTip(decl.tooltip)
    return spin


class AxisRangeHandler(_StaticMixin, ParamWidgetHandler):
    """A swept range rendered as three boxes ``[min] to [max] / [points] pts``.  The
    widget is a container whose ``min_spin`` / ``max_spin`` / ``pts_spin`` attributes
    carry the three controls, so ``read`` / ``write`` reach them by NAME (no positional
    tuple)."""

    def build(self, decl, value, ctx):
        default = decl.default if decl.default is not None else (0.0, 1.0, 2)
        seed = value if (value is not None) else default
        try:
            lo, hi, points = seed
        except (TypeError, ValueError):
            lo, hi, points = default
        lo_spin = _make_axis_spin(decl, value=lo)
        hi_spin = _make_axis_spin(decl, value=hi)
        pts_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
        pts_spin.setDecimals(0)
        pts_spin.setRange(2, 100000)
        pts_spin.setValue(int(points))
        pts_spin.setToolTip("Number of scan points (>= 2).")

        host = QtWidgets.QWidget()
        host.setStyleSheet("background: transparent;")
        row = QtWidgets.QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(scaled_px(4, minimum=3))
        row.addWidget(lo_spin)
        row.addWidget(_grey_label("to"))
        row.addWidget(hi_spin)
        row.addWidget(_grey_label("/"))
        row.addWidget(pts_spin)
        row.addWidget(_grey_label("pts"))
        host.min_spin = lo_spin
        host.max_spin = hi_spin
        host.pts_spin = pts_spin
        for spin in (lo_spin, hi_spin, pts_spin):
            _wire(spin.valueChanged, ctx, decl, lambda: self.read(host))
        return host

    def read(self, widget):
        return (float(widget.min_spin.value()), float(widget.max_spin.value()),
                int(widget.pts_spin.value()))

    def write(self, widget, value):
        if isinstance(value, (tuple, list)) and len(value) == 3:
            widget.min_spin.setValue(float(value[0]))
            widget.max_spin.setValue(float(value[1]))
            widget.pts_spin.setValue(int(value[2]))


def _grey_label(text: str) -> FluentLabel:
    label = FluentLabel(text)
    label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
    return label


# ----------------------------------------------------------------- dynamic combos
# The grouped-signal-picker helper cluster LIVES HERE (the leaf every form module depends on;
# task_console / figure_viewer forward-import it).  It used to live in task_console with lazy
# back-imports from this module -- a cycle that contradicted this file's leaf contract.  Pure
# Qt-combo utilities: only qt_widgets (FluentTreeComboBox, signals_blocked) + plain data.


def _common_token_prefix(names) -> str:
    """The longest common UNDERSCORE-token prefix of ``names`` -- e.g. both
    ``analysis_rate`` and ``analysis_score`` share ``analysis_``.
    Empty for fewer than two names or no shared leading token.  Used to strip the producer
    prefix the hub prepends from a grouped signal picker's labels (the producer is the group
    header, so its name need not repeat in every signal)."""
    import os.path as _op
    names = [str(n) for n in names]
    if len(names) < 2:
        return ""
    common = _op.commonprefix(names)
    cut = common.rfind("_")
    return common[: cut + 1] if cut >= 0 else ""


def signal_state(name, formats) -> str:
    """A signal has exactly TWO states (G3): "ready" when it is PUBLISHED on the hub right now
    (so it has a live shape in ``formats``), else "waiting" -- it is declared by a node that has
    not started / not produced yet.  No more none/unbound/error/mid-run/unpublished clutter."""
    return "ready" if formats.get(str(name)) else "waiting"


def _signal_short_label(name, group, labels) -> str:
    """The leaf label for ``name`` under its producer node in the picker nest: the SHORT signal name --
    the producing node's prefix stripped (``temperature_survival`` -> ``survival``, ``frame`` ->
    ``frame``), passed in via ``labels`` (the ``short_names_provider`` map, built from each running
    node's prefix; #design: the nest already names the producer, so the leaf is the short NAME, NOT the
    verbose SignalSpec axis label like ``camera image``).  For a signal with no mapped short name (a
    declared-but-not-running node), fall back to the shared-token prefix stripped from the group, else
    the bare name."""
    short = (labels or {}).get(str(name))
    if short:
        return str(short)
    strip = _common_token_prefix(group)
    if strip and name.startswith(strip) and len(name) > len(strip):
        return name[len(strip):]
    return str(name)


def grouped_signal_items(names, sources, formats, labels=None) -> list:
    """``[(display, bare_name | None)]`` for a signal picker, GROUPED by producing node: a
    non-selectable bold header per node (``bare_name`` is ``None``), then that node's signals
    indented beneath it -- shown by their HUMAN label (the SignalSpec the producing node declares),
    with the SHAPE and the two-state tag (``    Loading rate  [(35,)] ready`` / ``    Survival
    waiting``).  ``data`` stays the BARE signal name (the binding key); only the DISPLAY is humanised.
    The ONE source every signal picker shares (plot panel AND logic-node source)."""
    names = sorted(str(n) for n in (names or []))
    sources = dict(sources or {})
    formats = dict(formats or {})
    labels = dict(labels or {})
    by_producer: dict[str, list[str]] = {}
    for name in names:
        for p in ([str(p) for p in (sources.get(name) or [])] or ["(unbound)"]):
            by_producer.setdefault(p, []).append(name)
    items: list[tuple[str, str | None]] = []
    for producer in sorted(by_producer, key=lambda p: (p == "(unbound)", p.lower())):
        group = by_producer[producer]
        items.append((producer, None))            # group header (rendered disabled + bold)
        for name in group:
            short = _signal_short_label(name, group, labels)
            fmt = formats.get(name)
            state = signal_state(name, formats)
            shape = f"  [{fmt}]" if fmt else ""
            items.append((f"    {short}{shape}  {state}", name))
    return items


def signal_tree_groups(names, sources, formats, labels=None) -> list:
    """``[(producer, [(leaf_label, bare_name, full_label)])]`` for the COLLAPSIBLE tree picker
    (G2): one expandable group per producing node; each leaf's ``leaf_label`` shows the HUMAN signal
    label + shape + ready/waiting state (in the tree), and its ``full_label`` is the producer-
    qualified ``"<producer> · <label>"`` painted when the combo is COLLAPSED (frame-title aligned,
    G3).  Built from the same producer grouping + ``_signal_short_label`` as
    :func:`grouped_signal_items` -- ONE source, so neither ever shows a raw ``temperature_survival``."""
    names = sorted(str(n) for n in (names or []))
    sources = dict(sources or {})
    formats = dict(formats or {})
    labels = dict(labels or {})
    by_producer: dict[str, list[str]] = {}
    for name in names:
        for p in ([str(p) for p in (sources.get(name) or [])] or ["(unbound)"]):
            by_producer.setdefault(p, []).append(name)
    groups: list = []
    for producer in sorted(by_producer, key=lambda p: (p == "(unbound)", p.lower())):
        group = by_producer[producer]
        leaves = []
        for name in group:
            short = _signal_short_label(name, group, labels)
            fmt = formats.get(name)
            shape = f"  [{fmt}]" if fmt else ""
            leaf_label = f"{short}{shape}  {signal_state(name, formats)}"
            leaves.append((leaf_label, name, f"{producer} · {short}"))
        groups.append((producer, leaves))
    return groups


def fill_grouped_signal_combo(combo, *, names, sources, formats, current, none_label=None, labels=None) -> None:
    """Populate ``combo`` with every live hub signal GROUPED by producing node (via
    :func:`grouped_signal_items`): bold non-selectable headers, indented signals (data = the
    BARE name).  ``none_label`` adds a leading empty choice; a not-yet-published ``current``
    is kept selectable.  Read the pick back with ``currentData()`` (the bare name) -- the
    visible label is indented.  Shared by the plot panel's slot picker and the logic-node
    source field, so the nested picker is identical everywhere."""
    cur = str(current or "")
    # A configured input may NAME a signal that is declared but not published yet -- for example a
    # PulseScan y supplied by a not-yet-started external producer.  The
    # binding is by NAME, resolved at RUN time, so keep such a name in the pool: BOTH the tree and the
    # flat picker then render it as a "waiting" leaf AND read it back.  Single-sources the docstring's
    # "kept selectable" promise across both branches -- the tree branch used to drop a not-listed name,
    # so ``read_editable_combo`` returned '' and the configured input vanished (e.g. a Start that then
    # built the node with an empty y-expression input -> every point NaN).  ``signal_state`` renders the
    # added name honestly as "waiting" (it has no live shape in ``formats``).
    names = list(names or [])
    if cur and cur not in {str(n) for n in names}:
        names = [*names, cur]
    if isinstance(combo, FluentTreeComboBox):
        # The collapsible-tree picker (G2): one expandable producer group, leaves = signals.
        with _signals_blocked(combo):
            combo.set_signal_tree(signal_tree_groups(names, sources, formats, labels),
                                  current=cur, none_label=none_label)
        return
    with _signals_blocked(combo):
        combo.clear()
        if none_label is not None:
            combo.addItem(none_label, "")
        items = grouped_signal_items(names, sources, formats, labels)
        for label, name in items:
            if name is None:                      # group header: visible but not selectable
                combo.addItem(label, None)
                item = combo.model().item(combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
                    font = item.font(); font.setBold(True); item.setFont(font)
                continue
            combo.addItem(label, name)            # indented signal; data is the bare name
        idx = combo.findData(cur)
        # No match: select the leading none-row if there is one, else leave it BLANK (index -1)
        # -- never auto-land on a disabled group HEADER (data None), whose label would otherwise
        # read back as if it were the chosen signal.
        if idx < 0 and none_label is not None:
            idx = 0
        combo.setCurrentIndex(idx)


def read_editable_combo(combo) -> str:
    """Read an EDITABLE combo that pairs a display LABEL with a bare-value ``data`` (the grouped
    signal picker, the pulse-param picker).  Returns the selected item's data when the visible
    text still matches that item (a real pick) -- else the typed text (a not-yet-published custom
    name).  A plain ``currentData()`` would return the STALE previously-selected data after the
    user types a new name into the line edit (Qt does not move currentIndex on free text), so the
    fresh name would be silently dropped; a disabled header (data ``None``) falls through to ''."""
    if isinstance(combo, FluentTreeComboBox):
        return combo.current_signal()             # the tree picker stores the bare name on the leaf
    idx = combo.currentIndex()
    text = combo.currentText()
    if idx >= 0 and text == combo.itemText(idx):
        data = combo.itemData(idx)
        if data is not None:
            return str(data).strip()
    return text.strip()


def coerce_short_labels(provider) -> dict:
    """Normalise a ``{full hub name -> short name}`` callback into the ``labels`` map every grouped
    signal picker feeds ``fill_grouped_signal_combo``: callable-guard, ``str()`` both ends, drop empty
    short names, swallow any provider exception to ``{}``.  The ONE source the signal_expr / plot
    Setting slot / form signal pickers share so they render IDENTICALLY (#combo-parity) instead of four
    hand-copied dict comprehensions (the 4th of which had already dropped the try/except)."""
    if not callable(provider):
        return {}
    try:
        return {str(n): str(s) for n, s in dict(provider()).items() if s}
    except Exception:
        return {}


class SignalHandler(ParamWidgetHandler):
    """A hub-signal NAME (a processor's input): the collapsible-tree grouped picker."""

    def build(self, decl, value, ctx):
        combo = FluentTreeComboBox()
        cur = "" if (value is None and decl.default is None) else str(value if value is not None else decl.default)
        fill_grouped_signal_combo(combo, names=ctx.names(), sources=ctx.sources(),
                                  formats=ctx.formats(), labels=ctx.labels(), current=cur)
        combo.setToolTip(decl.tooltip)
        _wire(combo.activated, ctx, decl, lambda: read_editable_combo(combo))
        return combo

    def read(self, widget):
        return read_editable_combo(widget)

    def write(self, widget, value):
        cur = "" if value is None else str(value)
        idx = widget.findData(cur)
        if idx >= 0:
            widget.setCurrentIndex(idx)
        else:
            widget.setCurrentText(cur)

    def is_empty(self, widget) -> bool:
        return not read_editable_combo(widget)

    def refresh(self, widget, providers: RefreshProviders) -> None:
        current = read_editable_combo(widget)
        fill_grouped_signal_combo(widget, names=providers.signals, sources=providers.sources,
                                  formats=providers.formats, labels=providers.labels, current=current)


class SignalExprHandler(ParamWidgetHandler):
    """A multi-slot signal picker + ``value = ...`` expression (the COMPOSITE
    ``_SignalExprWidget``).  Value is ``{"inputs": [...], "source": "value = ..."}``."""

    def build(self, decl, value, ctx):
        if ctx.signal_expr_factory is None:
            raise RuntimeError("signal_expr kind needs ctx.signal_expr_factory")
        widget = ctx.signal_expr_factory(decl.row_label())      # single source: label + (unit) [+ *]
        seed = value if value is not None else (decl.default if decl.default is not None else {})
        widget.set_value(seed)
        widget.setToolTip(decl.tooltip)
        # route through the ONE wiring rule (re-validate AND, where a form enables it, instant-apply),
        # so a composite widget never silently misses the apply-on-edit path a scalar has
        _wire(widget.changed, ctx, decl, lambda: self.read(widget))
        return widget

    def read(self, widget):
        return widget.values_dict()

    def write(self, widget, value):
        widget.set_value(value)

    def is_empty(self, widget) -> bool:
        # a signal_expr always carries a usable {"inputs", "source"} (defaults to
        # value=signal); it is never "missing".
        return False

    def refresh(self, widget, providers: RefreshProviders) -> None:
        widget.rebuild_combos()


class PulseSlotsHandler(ParamWidgetHandler):
    """An auto-generated per-slot sub-form for a pulse template (the COMPOSITE
    ``_PulseSlotsWidget``), repopulated from a sibling ``path`` field.  Value is
    ``{"program_id": "...", "api": {...}, "sweep_kind": "scan_slot"|"api_slot",
    "program": "..."}``: fixed API overrides plus exactly one selected sweep program.
    ``program_id`` prevents an override from one template leaking into another template that
    happens to reuse an internal slot index."""

    def build(self, decl, value, ctx):
        if ctx.pulse_slots_factory is None:
            raise RuntimeError("pulse_slots kind needs ctx.pulse_slots_factory")
        widget = ctx.pulse_slots_factory()
        widget.setToolTip(decl.tooltip)
        # the ONE wiring rule (re-validate + optional instant-apply), like every scalar handler
        _wire(widget.changed, ctx, decl, lambda: self.read(widget))
        return widget

    def read(self, widget):
        return widget.values_dict()

    def write(self, widget, value):
        # The auto-form rebuilds from the template path, so the SLOT ROWS come from the template,
        # not the blob -- but the saved fixed values + active program DO round-trip: stash
        # them on the widget so the next repopulation (driven by the seeded template field) restores
        # them.  The form calls refresh()/repopulate AFTER seeding so this stash is consumed.
        if hasattr(widget, "seed_value"):
            widget.seed_value(value)

    def is_empty(self, widget) -> bool:
        value = self.read(widget)
        return not str(value.get("sweep_kind") or "").strip() \
            or not str(value.get("program") or "").strip()

    def refresh(self, widget, providers: RefreshProviders) -> None:
        if providers.repopulate is not None:
            providers.repopulate(widget)


# --------------------------------------------------------------------------- the registry


PARAM_WIDGETS: dict[str, ParamWidgetHandler] = {
    "float": _CurrentScalarAdapter("float"),
    "int": _CurrentScalarAdapter("int"),
    "axis_range": AxisRangeHandler(),
    "bool": _CurrentScalarAdapter("bool"),
    "choice": ChoiceHandler(),
    "text": _CurrentScalarAdapter("text"),
    "json": JsonHandler(),
    "device": DeviceRefHandler(),
    "path": PathHandler(),
    "signal": SignalHandler(),
    "signal_expr": SignalExprHandler(),
    "pulse_slots": PulseSlotsHandler(),
}

# A composite kind carries its OWN section header (it spans the full width with no outer
# row label); a scalar kind is a labelled row.  ONE list both form loops read, so the
# row-vs-span decision is declared here next to the handlers, not re-spelled per form.
SPAN_KINDS: frozenset = frozenset({"signal_expr", "pulse_slots"})


__all__ = [
    "PARAM_WIDGETS",
    "SPAN_KINDS",
    "ParamWidgetContext",
    "ParamWidgetHandler",
    "RefreshProviders",
]
