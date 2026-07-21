"""The measurement/processor/task parameter panel: one auto-generated form per node.

Pure Qt, per the placement axiom.  The form rows are built from the node's declared
parameters through the ONE param-widget registry, so this panel never hand-rolls a widget
ladder of its own.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from PyQt5 import QtCore, QtWidgets

from . import param_widgets as _param_widgets
from .fluent import (
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentSectionLabel,
    FluentSettingRow,
    GREEN,
    GREY,
    ORANGE,
    RED,
    scaled_px,
    setting_label_width,
)

PARAM_WIDGETS = _param_widgets.PARAM_WIDGETS
SPAN_KINDS = _param_widgets.SPAN_KINDS
ParamWidgetContext = _param_widgets.ParamWidgetContext
RefreshProviders = _param_widgets.RefreshProviders
coerce_short_labels = _param_widgets.coerce_short_labels
fill_grouped_signal_combo = _param_widgets.fill_grouped_signal_combo

__all__ = ["MeasurementPanel"]


class MeasurementPanel(QtWidgets.QWidget):
    """A measurement panel's Edit-tab Measurement section: pick a measurement, set
    its declared parameters in an AUTO-GENERATED + validated form, Start (one
    click) to stream the scan into a Monitor panel, Stop to cooperatively cancel.

    The form is built entirely from a :class:`MeasurementSpec`'s ``params``
    (the single source of truth -- the API default and this control derive from
    one declaration): each :class:`ParamDecl` becomes a widget chosen by its
    ``kind`` (float/int -> spin box, axis_range -> min/max/points triplet,
    bool -> checkbox, choice -> combo).  Values are read back BY KIND and
    coerced -- there is NO free-text eval (the confocal-GUI lesson).  ``required``
    params with an empty value disable Start with a hint.
    """

    start_requested = QtCore.pyqtSignal(object)    # emits THIS MeasurementPanel (read spec + values off it)
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, measurements: Sequence[object], parent=None, *, single: bool = False,
                 controls: bool = True, signals_provider=None, sources_provider=None, formats_provider=None,
                 short_names_provider=None, acquisition_params: Sequence[object] = ()):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._specs = list(measurements)
        # Extra ParamDecls APPENDED to every spec's auto-form (the ONE acquisition knob an acquisition
        # node owns -- ``repeat`` (0 = ∞) -- declared ONCE and auto-rendered through the SAME
        # ParamDecl path as every measurement param, so it is never a hand-placed widget, #H3l).
        self._acquisition_params = tuple(acquisition_params)
        self._single = bool(single)               # bound to ONE spec (per-panel Edit) -> hide the picker
        self._controls = bool(controls)           # False = no Start/Stop (e.g. a plot Edit's read-only-ish source form)
        # A ``kind="signal"`` param (a processor's input) renders the SAME nested-by-producer
        # signal picker the plot panels use: names_provider gives live signal names, and the
        # sources/formats providers give the producing node + shape for the grouping.
        self._signals_provider = signals_provider
        self._sources_provider = sources_provider
        self._formats_provider = formats_provider
        # The short-name map (full hub name -> short name) the signal picker uses as its ``labels`` so a
        # leaf reads "frame_0" not the prefix-stripped "0" -- threaded into the signal_expr widget so THIS
        # form's source picker renders IDENTICALLY to the plot Setting picker (#combo-parity).
        self._short_names_provider = short_names_provider
        self._widgets: dict[str, object] = {}     # param key -> widget (or tuple for axis_range)
        self._running = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(scaled_px(6, minimum=4))

        # measurement type picker (hidden in single-spec mode: the panel already
        # knows which measurement it edits, so the picker would be noise)
        pick_row = QtWidgets.QHBoxLayout()
        pick_row.setSpacing(scaled_px(6, minimum=4))
        self.type_combo = FluentComboBox()
        self.type_combo.setMinimumWidth(scaled_px(220, minimum=160))
        for index, spec in enumerate(self._specs):
            self.type_combo.addItem(str(spec.name), index)
        self.type_combo.setToolTip("Pick a measurement; its scanned parameters appear below.")
        self.type_combo.activated.connect(lambda *_: self._rebuild_form())
        self._pick_label = self._tag("measurement")
        pick_row.addWidget(self._pick_label)
        pick_row.addWidget(self.type_combo, 1)
        root.addLayout(pick_row)
        if self._single:
            self._pick_label.hide()
            self.type_combo.hide()

        # Auto-generated parameter form (rebuilt when the type changes).  It is a plain VBox of
        # the SHARED row primitives -- FluentSettingRow (grey fixed-width label | control) for a
        # scalar param, FluentSectionLabel for a composite's header -- exactly like the Setting
        # popup + plot Edit.  NOT a QFormLayout (whose labels render dark + auto-width = a 3rd
        # style); the whole frontend uses ONE label/row logic now.
        self.form = QtWidgets.QVBoxLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(scaled_px(5, minimum=3))
        self._form_label_w = scaled_px(72, minimum=56)   # recomputed per spec in _rebuild_form
        root.addLayout(self.form)

        # Start / Stop + status
        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(scaled_px(6, minimum=4))
        self.start_button = FluentButton("Start", color=GREEN)
        self.start_button.setToolTip("Build the measurement and stream its scan into a Monitor result panel.")
        self.start_button.clicked.connect(self._on_start)
        self.stop_button = FluentButton("Stop", color=ORANGE)
        self.stop_button.setToolTip("Cooperatively stop the running scan.")
        self.stop_button.clicked.connect(lambda: self.stop_requested.emit())
        self.stop_button.setEnabled(False)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        action_row.addStretch(1)
        # A plot Edit embeds this form to show its SOURCE node's parameters (#2) but
        # drives its own Apply -- the node is Started from its Logic-tab Edit, not here.
        if self._controls:
            root.addLayout(action_row)
        else:
            self.start_button.hide()
            self.stop_button.hide()

        self.status = FluentLabel("")
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        root.addWidget(self.status)

        if self._specs:
            self._rebuild_form()

    @staticmethod
    def _tag(text):
        label = FluentLabel(text)
        label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        return label

    def current_spec(self):
        index = self.type_combo.currentData()
        if index is None or not self._specs:
            return None
        return self._specs[int(index)]

    # ------------------------------------------------------------- form build
    def _clear_form(self) -> None:
        self._widgets = {}       # param key -> widget
        self._handlers = {}      # param key -> ParamWidgetHandler (kept in lockstep with _widgets)
        self._decls = {}         # param key -> ParamDecl
        while self.form.count():
            item = self.form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _add_row(self, label: str, control: QtWidgets.QWidget) -> None:
        """ONE param row = FluentSettingRow (grey fixed-width label | control), sharing this
        form's column width -- the SAME primitive the Setting popup + plot Edit use."""
        self.form.addWidget(FluentSettingRow(label, control, label_width=self._form_label_w))

    def _add_span(self, widget: QtWidgets.QWidget) -> None:
        """A composite control (signal_expr / pulse_slots) that carries its OWN FluentSectionLabel
        header spans the full width -- no outer row label (its header IS the label)."""
        self.form.addWidget(widget)

    def _param_context(self) -> ParamWidgetContext:
        """The bundle every PARAM_WIDGETS handler builds against: re-validation on edit
        + the signal providers + factories for the two composite widgets that live in
        this module.  ``instant_apply`` stays None -- the measurement form reads back on
        Start (``collect_values``), it does NOT push each edit into a live config."""
        return ParamWidgetContext(
            on_change=self._refresh_start_enabled,
            signals_provider=self._signals_provider,
            sources_provider=self._sources_provider,
            formats_provider=self._formats_provider,
            labels_provider=self._short_names_provider,
        )

    def _rebuild_form(self) -> None:
        """Rebuild the parameter form for the currently selected measurement.

        ONE thin loop: each ParamDecl's widget is built by its registry handler
        (``PARAM_WIDGETS[kind].build``) and stored by key alongside its HANDLER -- so the
        collect / seed / required / refresh / set_running loops dispatch through the
        handler, never index a positional tuple.  The only form-owned logic left is the
        layout (row vs full-width span) and the dependent-combo wiring (a sibling-field
        lookup the form must own)."""
        self._clear_form()
        # Pulse-slot declarations are kept per key so the template-dependent widget can
        # find its sibling ``path`` field and repopulate from the template it names.
        self._pulse_slots_decls: dict[str, object] = {}
        self._handlers: dict[str, object] = {}    # param key -> ParamWidgetHandler
        self._decls: dict[str, object] = {}        # param key -> ParamDecl
        spec = self.current_spec()
        if spec is None:
            return
        # the spec's declared params PLUS the auto-injected acquisition knob (repeat, 0 = ∞) --
        # ONE list so both the label-width and the widget loop see the same declarations.
        # ``display=False`` marks a NON-FORM payload (a structured value another surface injects --
        # e.g. the Analysis spec's fit_request the panel's Analysis section round-trips): it is not
        # renderable as a usable control, so the manual form skips it (its saved value still rides
        # ``row.node.values`` untouched -- the _start_logic_node merge preserves non-form keys).
        decls = [d for d in list(spec.params) + list(self._acquisition_params)
                 if getattr(d, "display", True)]
        # ONE label-column width for this form: fit the widest SCALAR-row label (composites carry
        # their own section header, so they are not row labels).  Single rule via setting_label_width.
        scalar_labels = [
            d.row_label() for d in decls if d.kind not in SPAN_KINDS      # single source: label + (unit) [+ *]
        ]
        self._form_label_w = setting_label_width(scalar_labels or [""], minimum=72)
        ctx = self._param_context()
        for decl in decls:
            kind = decl.kind
            # Show the READABLE label ("Pulse template" / "Signal (y)" / "Output name"), not the
            # raw build-call key ("template" / "y" / "y_name") -- the key is unreadable in a form
            # an experimenter actually uses (#H3); the tooltip still carries the full meaning.
            label_text = decl.row_label()           # single source: label + (unit) [+ *]
            handler = PARAM_WIDGETS[kind]
            # build the widget seeded from the decl's default (a saved value is applied later by
            # seed_values); the handler wires ctx.on_change (re-validate) onto its change signals.
            widget = handler.build(decl, decl.default, ctx)
            if kind in SPAN_KINDS:
                self._add_span(widget)              # FULL-WIDTH span (label is the widget's header, #H3b)
            else:
                self._add_row(label_text, widget)
            self._widgets[decl.key] = widget
            self._handlers[decl.key] = handler
            self._decls[decl.key] = decl
            if kind == "pulse_slots":
                self._pulse_slots_decls[decl.key] = decl
        # Wire each pulse_slots widget to its template path after the build loop so the
        # source field exists regardless of declaration order.
        for key, decl in self._pulse_slots_decls.items():
            src = self._sibling_path_widget(decl)
            if src is not None:
                src.changed.connect(lambda *_a, k=key: self._repopulate_pulse_slots(k))
            self._repopulate_pulse_slots(key)
        self._refresh_start_enabled()

    def _sibling_path_widget(self, decl):
        """The ``path`` widget named in ``decl.depends_on`` (the template a dependent
        pulse_slots field introspects), or None."""
        dep = getattr(decl, "depends_on", "")
        if dep and self._decls.get(dep) is not None and self._decls[dep].kind == "path":
            return self._widgets.get(dep)
        return None

    def _repopulate_pulse_slots(self, key: str) -> None:
        """Rebuild the auto-form rows of a ``pulse_slots`` widget from the pulse template
        named in its ``depends_on`` path field: one numeric input per API slot, one points-
        expression input per scan slot.  Preserves any value the operator already typed (the
        widget seeds each row from its current value when the slot survives the reload)."""
        widget = self._widgets.get(key)
        decl = self._pulse_slots_decls.get(key)
        if widget is None or decl is None:
            return
        src = self._sibling_path_widget(decl)
        path = src.text() if src is not None else ""
        # The template is READ by the domain and arrives as plain rows: the render
        # layer may not import the pulse compiler (see zlc_frontend.domain_ports).
        from zlc_frontend.domain_ports import pulse_template_rows
        try:
            rows = pulse_template_rows(path)
        except Exception:
            widget.rebuild([], [], program_id="")
            return
        widget.rebuild(list(rows.api_rows), list(rows.scan_rows),
                       api_columns=rows.api_columns, scan_columns=rows.scan_columns,
                       hardware_program=rows.program, program_id=rows.program_id)

    # ------------------------------------------------------------- value read
    def collect_values(self) -> dict[str, object]:
        """Read every parameter back BY KIND (no eval) into a build kwargs dict --
        each value is its handler's ``read`` of the widget (the coercion lives in
        PARAM_WIDGETS, one rule per kind)."""
        return {key: self._handlers[key].read(widget) for key, widget in self._widgets.items()}

    def refresh_on_show(self) -> None:
        """Re-poll providers and rebuild every dynamic control, so
        switching back to a tab whose providers have changed shows up-to-date choices.
        E.g. a processor's source dropdown was empty when first opened (no signal published
        yet); after a measurement publishes one, returning to this tab shows it -- without
        rebuilding the form, just refilling the existing combos -- so the current selection
        round-trips."""
        names: list[str] = []
        if callable(self._signals_provider):
            try: names = [str(n) for n in self._signals_provider()]
            except Exception: names = []
        sources = self._sources_provider() if callable(self._sources_provider) else {}
        formats = self._formats_provider() if callable(self._formats_provider) else {}
        labels = coerce_short_labels(self._short_names_provider)
        for key, widget in list(self._widgets.items()):
            kind = self._decls[key].kind
            # pulse_slots repopulates via the form's per-key hook (it reads the sibling
            # template); a signal picker refills from live providers.
            if kind == "pulse_slots":
                repopulate = lambda _w, k=key: self._repopulate_pulse_slots(k)
            else:
                repopulate = None
            providers = RefreshProviders(signals=names, sources=sources, formats=formats,
                                         labels=labels, repopulate=repopulate)
            self._handlers[key].refresh(widget, providers)
        self._refresh_start_enabled()

    def set_axis_range(self, lo: float, hi: float) -> bool:
        """Fill the FIRST axis_range param's Min/Max from a selected region
        (confocal _read_range: a plot selection becomes the next scan's range).
        Returns True if an axis_range param was found and set."""
        for key, widget in self._widgets.items():
            if self._decls[key].kind == "axis_range":
                widget.min_spin.setValue(float(min(lo, hi)))
                widget.max_spin.setValue(float(max(lo, hi)))
                return True
        return False

    def _missing_required(self) -> list[str]:
        """Required params whose value is empty -- delegated to each kind's handler
        (``is_empty``): only a blank line-edit / unpicked combo is "missing"; a spin
        box / switch always holds a value."""
        spec = self.current_spec()
        if spec is None:
            return []
        missing = []
        for decl in spec.params:
            if not decl.required:
                continue
            widget = self._widgets.get(decl.key)
            if widget is None or self._handlers[decl.key].is_empty(widget):
                missing.append(decl.label)
        return missing

    def _refresh_start_enabled(self, *_args) -> None:
        if self._running:
            return
        missing = self._missing_required()
        self.start_button.setEnabled(not missing)
        if missing:
            self.set_status("set required: " + ", ".join(missing), error=True)
        elif not self.status.text().startswith(("running", "done", "fit", "failed", "T")):
            self.set_status("ready", error=False)

    # ------------------------------------------------------------- run state
    def _on_start(self) -> None:
        spec = self.current_spec()
        if spec is None or self._missing_required():
            return
        self.start_requested.emit(self)

    def seed_values(self, values: Mapping[str, object]) -> None:
        """Pre-fill the form widgets from a saved ``{key: value}`` dict (e.g. the
        params the panel last ran with), so a per-panel Edit reopens on the
        last-used values rather than the declared defaults.  Unknown keys and
        shape mismatches are ignored -- the declared form is the source of truth."""
        for key, widget in self._widgets.items():
            if key not in values:
                continue
            # each kind's handler seeds its own widget (ignoring a shape mismatch / bad value)
            try:
                self._handlers[key].write(widget, values[key])
            except (TypeError, ValueError):
                continue
        # A pulse_slots auto-form rebuilds from its template field too: repopulate AFTER seeding so
        # the rebuild runs with the stash write() left (saved fixed values + hardware program) and
        # the round-trip restores them, regardless of seed order.
        for key in getattr(self, "_pulse_slots_decls", {}):
            self._repopulate_pulse_slots(key)
        self._refresh_start_enabled()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.start_button.setEnabled(not running and not self._missing_required())
        self.stop_button.setEnabled(bool(running))
        self.type_combo.setEnabled(not running)
        # _widgets[key] is now the widget itself (a handler-owned control), not a positional
        # tuple -- disable each one directly.
        for widget in self._widgets.values():
            widget.setEnabled(not running)

    def set_status(self, text: str, *, error: bool) -> None:
        self.status.setText(str(text)[:200])
        self.status.setStyleSheet(f"color: {RED if error else GREY}; background: transparent; border: none;")
