"""ONE param-kind -> Qt-widget registry, shared by EVERY param form.

The knowledge "a :class:`ParamDecl` of kind ``K`` is built / read / seeded /
validated / refreshed as widget ``W``" used to live in 5-7 parallel ladders inside
``task_console.py`` (the measurement form's build / collect / seed / required /
refresh / set_running loops) PLUS a SECOND, smaller ladder behind a parallel
``ParamSpec`` declaration class for plot-panel params.  Adding a kind meant editing
every ladder; forgetting one was a silent bug.

This module collapses all of that to ONE handler per kind.  A handler implements
the five operations every form needs:

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

Adding a new ParamDecl kind is now: add it to ``ParamDecl``'s whitelist, add ONE
handler here, register it in :data:`PARAM_WIDGETS`.  ``tests/
test_param_widget_registry.py`` mechanically enforces that the registry covers
every whitelisted kind and that a handler missing any of the five ops cannot
instantiate.

This is a FRONTEND module: it may import Qt and the frontend's own fluent widgets.
``ParamDecl`` itself stays dependency-free in ``operations`` -- this registry is the
GUI-side consumer that interprets a declaration, exactly as the docstring of
``ParamDecl`` says ("the spec consumer validates / coerces by ``kind``").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PyQt5 import QtWidgets

from Zou_lab_control._paths import display_path

from .qt_fluent import (
    GREY,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentSwitch,
    FluentTreeComboBox,
    FluentTriStateToggle,
    scaled_px,
)


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
        as the plot Setting / signal_expr pickers (#combo-parity)."""
        try:
            return {str(n): str(s) for n, s in dict(self.labels_provider()).items() if s} \
                if callable(self.labels_provider) else {}
        except Exception:
            return {}


@dataclass
class RefreshProviders:
    """What :meth:`ParamWidgetHandler.refresh` needs to repopulate a dynamic control.

    ``signals`` / ``sources`` / ``formats`` populate the grouped signal pickers.
    ``repopulate`` is a per-widget hook the form supplies for a DEPENDENT combo
    (``pulse_param`` / ``pulse_slots``) whose choices come from a SIBLING template
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


def _make_spin(decl, *, integer: bool, value=None) -> FluentDoubleSpinBox:
    """A bounded spin box for a float / int param (range + width from the decl) -- the
    SAME construction the measurement form's ``_spin`` used."""
    digits = max(5, len(str(int(abs(decl.hi) + 1))) + (0 if integer else 4))
    spin = FluentDoubleSpinBox(length=digits, allow_minus=float(decl.lo) < 0)
    if integer:
        spin.setDecimals(0)
    spin.setRange(float(decl.lo), float(decl.hi))
    if value is None:
        value = decl.default
    if value is not None:
        spin.setValue(int(value) if integer else float(value))
    spin.setToolTip(decl.tooltip)
    return spin


class FloatHandler(_StaticMixin, ParamWidgetHandler):
    """A bounded float.  An OPTIONAL float (``default is None and not required``) renders as a
    line edit so "leave blank = use the library / device default" stays expressible (mirrors
    :class:`IntHandler`); a defaulted / required float uses a spin box.  ``read`` / ``write`` /
    ``is_empty`` branch on which one was built (the line edit can be blank -> ``None``)."""

    @staticmethod
    def _is_optional(decl) -> bool:
        return decl.default is None and not decl.required

    def build(self, decl, value, ctx):
        if self._is_optional(decl):
            edit = FluentLineEdit("" if value is None else f"{float(value):g}")
            edit.setMinimumWidth(scaled_px(96, minimum=80))
            edit.setPlaceholderText("(default)")
            edit.setToolTip(decl.tooltip)

            def _read_opt():
                text = edit.text().strip()
                return float(text) if text else None

            _wire(edit.textChanged, ctx, decl, _read_opt)
            return edit
        spin = _make_spin(decl, integer=False, value=value)
        _wire(spin.valueChanged, ctx, decl, lambda: float(spin.value()))
        return spin

    def read(self, widget):
        if isinstance(widget, FluentLineEdit):
            text = widget.text().strip()
            return float(text) if text else None
        return float(widget.value())

    def write(self, widget, value):
        if isinstance(widget, FluentLineEdit):
            widget.setText("" if value is None else f"{float(value):g}")
        else:
            widget.setValue(float(value))

    def is_empty(self, widget) -> bool:
        # only the optional-float line edit can be blank; a spin box always has a number
        return isinstance(widget, FluentLineEdit) and not widget.text().strip()


class IntHandler(_StaticMixin, ParamWidgetHandler):
    """A bounded int.  An OPTIONAL int (``default is None and not required``) renders as
    a line edit so "leave blank = all" stays expressible; a defaulted / required int
    uses a spin box.  ``read`` / ``write`` / ``is_empty`` branch on which one was built
    (the line edit can be blank; the spin box always holds a number)."""

    @staticmethod
    def _is_optional(decl) -> bool:
        return decl.default is None and not decl.required

    def build(self, decl, value, ctx):
        if self._is_optional(decl):
            edit = FluentLineEdit("" if value is None else str(int(value)))
            edit.setMinimumWidth(scaled_px(96, minimum=80))
            edit.setPlaceholderText("(all)")
            edit.setToolTip(decl.tooltip)

            def _read_opt():
                text = edit.text().strip()
                return int(text) if text else None

            _wire(edit.textChanged, ctx, decl, _read_opt)
            return edit
        spin = _make_spin(decl, integer=True, value=value)
        _wire(spin.valueChanged, ctx, decl, lambda: int(spin.value()))
        return spin

    def read(self, widget):
        if isinstance(widget, FluentLineEdit):
            text = widget.text().strip()
            return int(text) if text else None
        return int(widget.value())

    def write(self, widget, value):
        if isinstance(widget, FluentLineEdit):
            widget.setText("" if value is None else str(int(value)))
        else:
            widget.setValue(int(value))

    def is_empty(self, widget) -> bool:
        # only the optional-int line edit can be blank; a spin box always has a number
        return isinstance(widget, FluentLineEdit) and not widget.text().strip()


class BoolHandler(_StaticMixin, ParamWidgetHandler):
    def build(self, decl, value, ctx):
        sw = FluentSwitch("")
        sw.setChecked(bool(decl.default if value is None else value))
        sw.setToolTip(decl.tooltip)
        _wire(sw.toggled, ctx, decl, lambda: bool(sw.isChecked()))
        return sw

    def read(self, widget):
        return bool(widget.isChecked())

    def write(self, widget, value):
        widget.setChecked(bool(value))


class ChoiceHandler(_StaticMixin, ParamWidgetHandler):
    """One of ``decl.choices``.  ``decl.segmented`` renders a confocal-style capsule multi-state toggle
    (:class:`FluentTriStateToggle`) instead of a combo box -- both expose ``currentText`` /
    ``setCurrentText`` / ``activated``, so ``read`` / ``write`` / wiring stay widget-agnostic (only
    ``build`` decides which control to construct)."""

    def build(self, decl, value, ctx):
        choices = [str(c) for c in decl.choices]
        segmented = bool(getattr(decl, "segmented", False))
        widget = FluentTriStateToggle(choices) if segmented else FluentComboBox()
        if not segmented:
            widget.addItems(choices)
        cur = decl.default if value is None else value
        if cur is not None and str(cur) in choices:
            widget.setCurrentText(str(cur))
        widget.setToolTip(decl.tooltip)
        _wire(widget.activated, ctx, decl, lambda: widget.currentText())
        return widget

    def read(self, widget):
        return widget.currentText()

    def write(self, widget, value):
        widget.setCurrentText(str(value))


class TextHandler(_StaticMixin, ParamWidgetHandler):
    """A free string (line edit).  Taken VERBATIM, never eval'd."""

    def build(self, decl, value, ctx):
        seed = "" if value is None else str(value)
        edit = FluentLineEdit(seed)
        edit.setMinimumWidth(scaled_px(160, minimum=120))
        edit.setPlaceholderText(decl.tooltip[:48] if decl.tooltip else "")
        edit.setToolTip(decl.tooltip)
        _wire(edit.textChanged, ctx, decl, lambda: edit.text().strip())
        return edit

    def read(self, widget):
        return widget.text()

    def write(self, widget, value):
        widget.setText("" if value is None else str(value))

    def is_empty(self, widget) -> bool:
        return not widget.text().strip()


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
        lo_spin = _make_spin(decl, integer=False, value=lo)
        hi_spin = _make_spin(decl, integer=False, value=hi)
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


# These three helpers come from task_console; importing them at module scope would be a
# cycle (task_console imports THIS module).  They are tiny and stable, so we import them
# lazily inside the handlers -- the SAME single-source helpers every picker uses.
def _grouped_fill(combo, *, names, sources, formats, current, labels=None):
    from .task_console import fill_grouped_signal_combo
    fill_grouped_signal_combo(combo, names=names, sources=sources, formats=formats,
                              labels=labels or {}, current=current)


def _read_editable(combo) -> str:
    from .task_console import read_editable_combo
    return read_editable_combo(combo)


class SignalHandler(ParamWidgetHandler):
    """A hub-signal NAME (a processor's input): the collapsible-tree grouped picker."""

    def build(self, decl, value, ctx):
        combo = FluentTreeComboBox()
        cur = "" if (value is None and decl.default is None) else str(value if value is not None else decl.default)
        _grouped_fill(combo, names=ctx.names(), sources=ctx.sources(), formats=ctx.formats(),
                      labels=ctx.labels(), current=cur)
        combo.setToolTip(decl.tooltip)
        _wire(combo.activated, ctx, decl, lambda: _read_editable(combo))
        return combo

    def read(self, widget):
        return _read_editable(widget)

    def write(self, widget, value):
        cur = "" if value is None else str(value)
        idx = widget.findData(cur)
        if idx >= 0:
            widget.setCurrentIndex(idx)
        else:
            widget.setCurrentText(cur)

    def is_empty(self, widget) -> bool:
        return not _read_editable(widget)

    def refresh(self, widget, providers: RefreshProviders) -> None:
        current = _read_editable(widget)
        _grouped_fill(widget, names=providers.signals, sources=providers.sources,
                      formats=providers.formats, labels=providers.labels, current=current)


class PulseParamHandler(ParamWidgetHandler):
    """WHICH pulse-template parameter to sweep: a DEPENDENT editable combo repopulated
    from a sibling ``path`` field.  The repopulation (which reads the sibling template)
    is owned by the form (``providers.repopulate``); the handler only builds / reads /
    seeds the editable combo."""

    def build(self, decl, value, ctx):
        combo = FluentComboBox()
        combo.setEditable(True)
        combo.setToolTip(decl.tooltip)
        if value is not None:
            combo.setCurrentText("" if value is None else str(value))
        _wire(combo.activated, ctx, decl, lambda: _read_editable(combo))
        return combo

    def read(self, widget):
        return _read_editable(widget)

    def write(self, widget, value):
        widget.setCurrentText("" if value is None else str(value))

    def is_empty(self, widget) -> bool:
        return not _read_editable(widget)

    def refresh(self, widget, providers: RefreshProviders) -> None:
        if providers.repopulate is not None:
            providers.repopulate(widget)


class SignalExprHandler(ParamWidgetHandler):
    """A multi-slot signal picker + ``value = ...`` expression (the COMPOSITE
    ``_SignalExprWidget``).  Value is ``{"inputs": [...], "source": "value = ..."}``."""

    def build(self, decl, value, ctx):
        if ctx.signal_expr_factory is None:
            raise RuntimeError("signal_expr kind needs ctx.signal_expr_factory")
        title = (decl.label or decl.key) + (f" ({decl.unit})" if decl.unit else "")
        widget = ctx.signal_expr_factory(title)
        seed = value if value is not None else (decl.default if decl.default is not None else {})
        widget.set_value(seed)
        widget.setToolTip(decl.tooltip)
        widget.changed.connect(ctx.on_change)
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
    ``{"api": {...}, "scan_mode": "none"|"api"|"scan", "scan_code": "...", "extra_delay": ...}``
    -- the api fixed values plus a single Scan-mode toggle that picks what one shared scan table
    sweeps (the build dispatches on ``scan_mode``)."""

    def build(self, decl, value, ctx):
        if ctx.pulse_slots_factory is None:
            raise RuntimeError("pulse_slots kind needs ctx.pulse_slots_factory")
        widget = ctx.pulse_slots_factory()
        widget.setToolTip(decl.tooltip)
        widget.changed.connect(ctx.on_change)
        return widget

    def read(self, widget):
        return widget.values_dict()

    def write(self, widget, value):
        # The auto-form rebuilds from the template path, so the SLOT ROWS come from the template,
        # not the blob -- but the saved api VALUES + scan_mode + active program DO round-trip: stash
        # them on the widget so the next repopulation (driven by the seeded template field) restores
        # them.  The form calls refresh()/repopulate AFTER seeding so this stash is consumed.
        if hasattr(widget, "seed_value"):
            widget.seed_value(value)

    def is_empty(self, widget) -> bool:
        return False

    def refresh(self, widget, providers: RefreshProviders) -> None:
        if providers.repopulate is not None:
            providers.repopulate(widget)


# --------------------------------------------------------------------------- the registry


PARAM_WIDGETS: dict[str, ParamWidgetHandler] = {
    "float": FloatHandler(),
    "int": IntHandler(),
    "axis_range": AxisRangeHandler(),
    "bool": BoolHandler(),
    "choice": ChoiceHandler(),
    "text": TextHandler(),
    "path": PathHandler(),
    "signal": SignalHandler(),
    "signal_expr": SignalExprHandler(),
    "pulse_param": PulseParamHandler(),
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
