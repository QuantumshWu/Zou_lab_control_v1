"""Task console: a configurable dashboard with a clean VIEW / LOGIC split.

Two permanent tabs:

* **Monitor** -- the drag-and-snap board of PLOT panels.  A plot is a PURE VIEW,
  fully decoupled from acquisition: it shows nothing until you pick a hub signal in
  its Setting, and it NEVER builds, owns or starts a measurement.  Each panel card
  is JUST the plot frame (the frontend canvas + a thin border that doubles as the
  DRAG HANDLE -- grab the border, drop the card, it snaps to the layout grid).
  Everything else (title, size, signal source, fit, limits, save) lives behind the
  gear button / the panel's own Edit tab.

* **Logic** -- the list of LOGIC NODES (measurement / processor / task).  A logic
  node is the thing that PRODUCES data: added STOPPED, it shows as a row with a
  status colour dot (grey=stopped, green=running, red=error) + its name + an Edit
  button.  Its Edit tab carries its auto-generated parameter form + Start / Stop
  (no curve fit -- fitting is a plotter concern).  Start builds the node from
  the form values with display SUPPRESSED (it only publishes signals to the hub --
  never opens a matplotlib plot) and runs it; Stop cancels it.  You then Add plot
  panels on the Monitor board pointed at the signals it publishes.

A plot panel's data source is a one-line expression evaluated against the named
signals of a :class:`~Zou_lab_control.neutral_atom.core.signals.SignalHub`
(the same trusted-local-code posture as the pulse GUI's Scan tab):

    value = frame_0                     # show emCCD event 0 of the camera cycle
    value = occupied - b_occupied       # arbitrary math across signals (two detectors)
    value = history('counts', 200).ravel()

Layouts (plot panels + logic nodes + positions + sizes + expressions + params)
save/load as ONE JSON and are machine-portable: all plot geometry is owned by
frontend.panel_plot and never part of the layout.
"""

from __future__ import annotations

import hashlib
import inspect
import math as _math
import os
import re
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from .live import (
    panel_plot,
    recommended_grid_size,
    region_binding,
)
from zlc_frontend import board_layout as _layout
from zlc_data.console_records import (
    ADDABLE_PANEL_KINDS,
    PANEL_INPUT_FORMAT,
    panel_allows_multi_slot,
    panel_input_slots,
    CONSOLE_STATE_SCHEMA,
    BLANK_SOURCE as _BLANK_SOURCE,
    DEFAULT_UPDATE_MS,
    LOGIC_KINDS,
    LOGIC_NODE_CONFIG_FIELDS as _LOGIC_NODE_CONFIG_FIELDS,
    LogicNodeConfig as _LogicNodeConfig,
    PANEL_INPUT_SLOTS,
    PANEL_KINDS,
    PANEL_SINGLE_SLOT_KINDS,
    PanelConfig as _PanelConfig,
    TASK_CONSOLE_STATE_FIELDS as _TASK_CONSOLE_STATE_FIELDS,
    UPDATE_INTERVALS,
    layout_record,
)
from zlc_frontend.qt_widgets import LogicNodeRow as _LogicNodeRow
from zlc_data.panel_size import PANEL_SIZES, panel_size_cells
from zlc_data.shape_text import slot_label   # the ONE human slot-label formatter (period/channel)
from zlc_frontend.console_state import (
    TASK_FILES_ENV as _TASK_FILES_ENV,
    TaskConsoleState as _TaskConsoleState,
    default_console_state as _default_console_state,
    resolve_task_state as _resolve_task_state,
    task_files_dir as _task_files_dir,
)
# Submodules of the package are reached as ATTRIBUTES of this ONE binding: they are
# deliberately absent from the facade's __all__, and the package guard forbids anyone
# outside importing ``zlc_frontend.qt_widgets.<sub>`` as a path.
import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import PulseSlotsWidget, SignalExprWidget
from zlc_frontend.qt_widgets import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    TEXT,
    WINDOW_SCREEN_FRACTION,
    YELLOW,
    fluent_message,
    FluentButton,
    FluentCodeEdit,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    setting_label_width,
    FluentStatusDot,
    FluentStatusStrip,
    FluentSwitch,
    FluentTabWidget,
    ensure_qt_app,
    launch_fluent_window,
    fluent_widget_stylesheet,
    scaled_px,
    window_pad,
    screen_fit_window_size,
    set_fluent_scale,
    signals_blocked as _signals_blocked,
)

try:  # guarded like pulse_gui: the console degrades without matplotlib-qt
    import matplotlib.pyplot as plt
    from .qt_canvas import panel_canvas
except Exception:  # pragma: no cover - depends on the local matplotlib install
    plt = None
    panel_canvas = None

# Single source for turning a path param into an unambiguous, project-anchored display
# string (an absolute path, never a bare CWD-relative name) -- the same seam the analysis
# layer uses to resolve paths, so the field shows exactly the file/folder that is used.
from zlc_storage.paths import display_path

# The ONE param-kind -> widget registry every form in this module dispatches through: a
# ParamDecl of kind K is built / read / seeded / validated / refreshed by PARAM_WIDGETS[K].
# Adding a kind is one handler there + one whitelist entry on ParamDecl, not 5-7 ladders here.
from .param_widgets import PARAM_WIDGETS, SPAN_KINDS, ParamWidgetContext, RefreshProviders
# The grouped-signal-picker cluster lives in param_widgets (the leaf; see the note above
# strip_node_prefix) -- forward imports only, so the leaf never back-imports this module.
from .param_widgets import (
    coerce_short_labels,
    read_editable_combo,
)

# ParamDecl is the ONE declarative param record both the measurement form and the plot
# panels use (PANEL_PARAMS).  Importing it here (frontend -> neutral_atom is allowed; the
# reverse is not) lets the panel params be real ParamDecls validated by the kind whitelist,
# instead of a parallel ParamSpec class with its own smaller ladder.

# The default mid-run buffer key -- the spec layer's ONE spelling (TaskSpec.mid_run_key's
# default), imported so the console's spec-less fallbacks can never drift from it.
from zlc_data.vocabulary import DEFAULT_MID_RUN_KEY
from zlc_storage import exact_mapping


#: Now :data:`zlc_frontend.console_state.TASK_FILES_ENV`; the old name stays an alias.
TASK_FILES_ENV = _TASK_FILES_ENV

# ---- RESERVED expression-namespace keys (each spelled ONCE; every writer/reader shares these).
#: The running task's typed mid-run tensor: injected off-hub by the console each tick, read by the
#: task's dedicated panel (its source is ``value = {TASK_FRAME_KEY}``) -- never a hub signal.
TASK_FRAME_KEY = "__task_frame__"
# The panel card, its board, the card-geometry bridge and their constants now live in
# ``zlc_workbench.task_console.plot_bridge`` -- the sanctioned Qt x matplotlib zone (C20):
# a card HOLDS its matplotlib plotter/canvas inside a Qt group box, which neither
# qt_widgets (no matplotlib) nor zlc_data (no toolkit) may hold.  These names ARE the
# moved objects (not copies), so every caller, isinstance check and test keyed on the
# shell path keeps working while the shell is taken apart.
import zlc_workbench.task_console.plot_bridge as _plot_bridge

SIG_VERSIONS_KEY = _plot_bridge.SIG_VERSIONS_KEY
SIG_VALID_KEY = _plot_bridge.SIG_VALID_KEY
COORD_FRAMES_KEY = _plot_bridge.COORD_FRAMES_KEY
_BLANK_SOURCE = _plot_bridge._BLANK_SOURCE
GRID_UNIT = _plot_bridge.GRID_UNIT
GAP = _plot_bridge.GAP
_RELIM_MODES = _plot_bridge._RELIM_MODES
_RELIM_PARAM = _plot_bridge._RELIM_PARAM
REBUILD_DEBOUNCE_MS = _plot_bridge.REBUILD_DEBOUNCE_MS
_MONITOR_UNSET = _plot_bridge._MONITOR_UNSET
_card_y_is_view_axis = _plot_bridge._card_y_is_view_axis
_opaque_white_composite = _plot_bridge._opaque_white_composite
_board_metrics = _plot_bridge._board_metrics
_cell_size = _plot_bridge._cell_size
_card_size = _plot_bridge._card_size
_aabb = _plot_bridge._aabb
_overlaps_with_gap = _plot_bridge._overlaps_with_gap
_first_free_slot = _plot_bridge._first_free_slot
_board_width = _plot_bridge._board_width
pack = _plot_bridge.pack
drop_index = _plot_bridge.drop_index
_unit_df_for = _plot_bridge._unit_df_for
_GridFocus = _plot_bridge._GridFocus
PanelCard = _plot_bridge.PanelCard
_PanelBoard = _plot_bridge._PanelBoard

#: The display suffix marking a task's SYNTHETIC mid-run entry in the picker / Logic legend
#: (never part of a hub name) -- one spelling shared by the declared and the running paths.
MID_RUN_TAG = " (mid-run)"

# WHICH hardware a logic node references is the NODE's own ``referenced_devices()`` declaration.
# Every such node needs runtime authority; ``occupied_devices()`` is only the narrower subset the
# authority turns into EXCLUSIVE claims while observe-only references receive OBSERVE claims.
# There is no second frontend kind-string table and no global "stop everything" rule.

#: Every panel + logic-node name is "<base> #N" with N counting from 1 (G1), so two panels /
#: nodes of the same kind are always told apart -- in the card title, the Edit tab, the frame
#: title, and the signal-flow grouping.  ONE source of that scheme for panels and nodes alike.
# The six pure value/text helpers this module used to define now live with their owners
# (S5-shell(z)); the names below are the shell's plain aliases for them.  ``_GridFocus`` and
# ``_StopAttempt`` deliberately did NOT go: the census calls them render-free because it reads
# IMPORTS, but they HOLD live objects (a grid plotter + its canvas + mpl callback ids; a running
# daemon thread), so their home is wherever those objects live.
from zlc_data.shape_text import indexed_unique_name, strip_node_prefix  # noqa: E402
from zlc_data.param_decl import acquisition_param_decls as _acquisition_param_decls  # noqa: E402
from zlc_frontend.form import (  # noqa: E402
    lenient_float as _safe_float,
    python_to_text as _py_to_text,
    text_to_python as _text_to_py,
)
#: The expression-namespace help text, read straight from its owner.  The shell used to wrap it in
#: a function guarding a module-level cache -- an indirection whose only job was to memoise an
#: imported CONSTANT, which is not a thing that needs memoising.


# The grouped-signal-picker helper cluster (signal_state / grouped_signal_items /
# signal_tree_groups / fill_grouped_signal_combo / read_editable_combo / coerce_short_labels,
# #combo-parity) moved DOWN to param_widgets.py -- the leaf this module already imports -- so the
# leaf's old lazy back-imports of it are gone (no frontend cycle).  strip_node_prefix stays here:
# it is the Logic tab's rule (shared vocabulary, not a picker widget helper).


#: Now :class:`zlc_frontend.qt_widgets.PulseSlotsWidget`; the old name stays a plain
#: alias so nothing that referenced it has to learn a new one.
_PulseSlotsWidget = PulseSlotsWidget

#: Now :class:`zlc_frontend.qt_widgets.SignalExprWidget`; the old name stays a
#: plain alias so nothing that referenced it has to learn a new one.
_SignalExprWidget = SignalExprWidget


# Every Monitor-board panel is a PURE VIEW: it is fully decoupled
# from acquisition -- it shows a hub signal and carries the full plotter Edit (the
# whole DataFigure fit set + manual limits + the display knobs), but it NEVER
# builds / owns / starts a node.  The things that PRODUCE data
# (measurement / processor / task) are LOGIC NODES on the separate Logic tab, not
# board panels (see LogicNodeConfig / LogicNodeRow).

# The KINDS of LOGIC NODE the Logic tab hosts -- the node layers (measurement /
# processor / task).  "camera" is the continuous camera Measurement (a live frame
# stream); "measurement" a swept measurement; "processor" a reactive transform;
# "task" a one-shot orchestration.  A node is added STOPPED and Start/Stop'd from
# its own Edit -- it only ever publishes to the hub (display suppressed).

# The panel-param CATALOG and its four resolvers now live in ``zlc_frontend.panel_params``
# (S5-shell(y)); the names below are the shell's plain aliases for them.
from zlc_frontend.panel_params import (  # noqa: E402
    CMAPS,
    PANEL_PARAMS,
    GRID_TITLE_PARAMS,
    panel_display_decls as _panel_display_decls,
    panel_param_default as _panel_param_default,
    resolved_cmap as _resolved_cmap,
    resolved_param as _resolved_param,
)






# The per-panel Analysis controls now live in ``zlc_frontend.qt_widgets.analysis_controls``.
# These names ARE the moved objects (not copies), so every caller, isinstance check and test
# keyed on the shell path keeps working while the shell is taken apart.
_general_fit_models_for_kind = _qt_widgets.analysis_controls._general_fit_models_for_kind
_FitFixSeedEditor = _qt_widgets.analysis_controls._FitFixSeedEditor
_apply_analysis_state_to_widgets = _qt_widgets.analysis_controls._apply_analysis_state_to_widgets
AnalysisControls = _qt_widgets.analysis_controls.AnalysisControls















#: The placement-only stand-in now lives with the packer it serves.
_GeomProxy = _layout.GeomProxy














#: Image containers the Edit-tab Save offers, in menu order (first = default).  A DATA-layer choice of
#: the output FILE FORMAT (not an art knob): the figure's geometry / dpi / typography are unchanged; only
#: the container the ``DataFigure.save(image_ext=...)`` writes changes.  Lowercase to match matplotlib's
#: ``savefig`` extensions and confocal's ``save_type`` naming (jpg / png).  The matching ``.npz`` data
#: file is format-independent -- identical arrays + ``info`` for every choice.
SAVE_IMAGE_FORMATS: tuple[str, ...] = ("png", "pdf", "jpg")





# ====================================================================== state


#: The ONE console-record validator now lives in :mod:`zlc_data.console_records`;
#: panel, logic-node and console-state records all read it from there.
_layout_record = layout_record


#: Both console records now live in :mod:`zlc_data.console_records`; the old names
#: stay as plain aliases, so every existing caller and export keeps working.
PanelConfig = _PanelConfig
LogicNodeConfig = _LogicNodeConfig


#: The saved-layout record, its JSON file and the name->layout resolver now live in
#: :mod:`zlc_frontend.console_state` (L315 keeps a domain-schema reader out of storage;
#: L327 lets the frontend depend on storage).  The old names stay as plain aliases.
TaskConsoleState = _TaskConsoleState
default_console_state = _default_console_state
resolve_task_state = _resolve_task_state









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


class PanelEditor(QtWidgets.QWidget):
    """One panel's processing tab (confocal per-plot control card).

    Opened from a panel's ``Edit...`` button as its OWN closable tab, so several
    panels can be edited side by side.  Holds a FROZEN snapshot of that panel's
    current data plus the heavy controls that do NOT belong in the lightweight
    Setting popup AND do NOT duplicate it (Setting owns source / size / colormap
    / relim / unit):

      * curve fit -- a structured model request gated by the plot family,
        shared with the live overlay and :class:`DataFigure`;
      * manual x/y limits + Save Fig;
      * for a panel that came from a measurement, that measurement's
        AUTO-generated parameter form (the same ParamDecl form the launcher
        uses, bound to the one spec) whose Start RE-RUNS the measurement into
        this very Monitor panel with the edited params.

    The whole page lives in a scroll area, so the snapshot never pushes the
    fit/limits row off-screen (the old single-editor cutoff)."""

    def __init__(self, card: "PanelCard", console: "TaskConsole", parent=None):
        super().__init__(parent)
        self.card = card
        self.console = console
        self.setStyleSheet("background: transparent;")
        # Which kind's PANEL_PARAMS this page baked its rows from -- a grid's RESOLVED per-cell
        # kind can change later (a facet / sub-plot pick), and refresh_on_show then rebuilds this
        # page so the rows never lie (the Edit mirror of _sync_settings_param_rows).
        self._param_kind_built = card._param_kind() if card is not None else None
        self._plotter = None
        self._canvas = None
        self._df = None
        # A plot panel's Edit NEVER carries a measurement/processor param form or
        # Start/Stop -- a plot is a pure VIEW, decoupled from acquisition (the
        # node lives on the Logic tab).  meas_panel stays None so any leftover
        # callers no-op.
        self.meas_panel = None
        self._node = None                       # the node that produces this panel's data
        self._node_widgets: dict = {}           # acquisition-param name -> editable field
        self._node_now_labels: dict = {}        # acquisition-param name -> "now: X" reference
        self.fit_combo = None
        self.xmin = self.xmax = self.ymin = self.ymax = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        scroll.setWidget(page)
        col = QtWidgets.QVBoxLayout(page)
        m = scaled_px(10, minimum=6)
        col.setContentsMargins(m, m, m, m)
        col.setSpacing(scaled_px(6, minimum=4))

        def section(text):
            col.addWidget(FluentSectionLabel(text))   # the ONE section style (bold dark), like everywhere

        def labeled(text):
            lab = FluentLabel(text)
            lab.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            return lab

        # A plot panel is a PURE VIEW: its Edit carries ONLY the plotter controls
        # (acquisition of whatever node already publishes the signal it reads --
        # read-only-ish, snapshot, fit, manual limits, save).  It NEVER builds or
        # starts a node -- that lives on the Logic tab.

        # ---- Panel: rename this panel right here in the Edit (kept in sync with the Setting
        # popup's title field; both go through the sealed title API).  Saves a trip back to
        # the Setting popup just to relabel.
        section("Panel")
        self.title_edit = FluentLineEdit(card.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.setToolTip("Rename this panel (also the default save name).")
        self.title_edit.textChanged.connect(self._edit_title)
        col.addWidget(FluentSettingRow("title", self.title_edit, label_width=scaled_px(96, minimum=72)))

        # ---- Acquisition: the editable parameters of the DATA SOURCE behind this
        # panel.  A panel is a VIEW; the LOGIC NODE whose published signals it reads
        # declares what its source exposes via acquisition_parameters() -- a
        # raw-frame panel reads the camera Measurement (exposure / ROI), while a
        # derived panel reads its processor.  Each field is PREFILLED with
        # the CURRENT value (with a "now: X" reference); Apply pushes the edit to
        # that source IN PLACE (it does not start anything -- the node is started
        # from its own Logic-tab Edit).
        self._node = console._producing_node(card)
        for name, current in console._node_params(self._node):
            if not self._node_widgets:
                section("Acquisition")
            edit = FluentLineEdit(_py_to_text(current))
            # EXPAND to fill the row (stretch=1), not a fixed 110 px: a value
            # like a 4-int ROI [1648, 64, 1144, 64] was clipped while the right
            # half of the page sat empty.  The edit takes the slack; the live
            # "now:" reference trails it.  (cutoff is a core rule.)
            edit.setMinimumWidth(scaled_px(150, minimum=120))
            self._node_widgets[name] = edit
            now = labeled(f"now: {_py_to_text(current)}")
            self._node_now_labels[name] = now
            holder = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(scaled_px(6, minimum=4))
            hl.addWidget(edit, 1)
            hl.addWidget(now, 0)
            # widest node-param labels ("threshold_method" / "readout_exposure", 16 chars)
            # need ~170 px; 150 clipped a trailing char, so 170 fits the longest name in full.
            col.addWidget(FluentSettingRow(name, holder, label_width=scaled_px(170, minimum=140)))
        if self._node_widgets:
            self.node_apply_button = FluentButton("Apply", color=ACCENT)
            self.node_apply_button.setToolTip(
                "Apply the edited acquisition parameters to the data source in place\n"
                "(reconfigure the camera live) -- the panel keeps streaming.")
            self.node_apply_button.clicked.connect(self._restart_node)
            col.addWidget(self.node_apply_button)

        # ---- Source: a plot's signal ALWAYS comes from a measurement/processor (#2),
        # so show THAT node's full declarative parameter form here -- prefilled from its
        # current values (defaults via the spec).  Shown when the node exposes no live
        # acquisition_parameters above (a scanned measurement / a processor); Apply
        # rebuilds + restarts the source node with the edited params (it is Started /
        # Stopped from its OWN Logic-tab Edit, so this form carries no Start/Stop).
        self._source_row = None
        self.source_form = None
        if not self._node_widgets:
            self._source_row = console._producing_row(card)
            source_spec = (console._spec_for_logic(self._source_row.node)
                           if self._source_row is not None else None)
            if source_spec is not None:
                section(f"Source: {source_spec.name}")
                # pass the signal providers so a processor's source parameter renders
                # the SAME nested-by-producer picker the logic-node Edit uses --
                # not a flat/empty combo (every signal picker is the nested form, everywhere).
                self.source_form = MeasurementPanel(
                    [source_spec], single=True, controls=False,
                    signals_provider=getattr(console, "_signal_names", None),
                    sources_provider=getattr(console, "_signal_providers", None),
                    formats_provider=getattr(console, "_signal_formats", None),
                    short_names_provider=getattr(console, "_signal_short_names", None))
                self.source_form.seed_values(self._source_row.node.values or {})
                col.addWidget(self.source_form)
                self.source_apply_button = FluentButton("Apply", color=ACCENT)
                self.source_apply_button.setToolTip(
                    "Apply these parameters to the source node (rebuild + restart it);\n"
                    "the plot keeps reading its published signal.")
                self.source_apply_button.clicked.connect(self._apply_source_form)
                col.addWidget(self.source_apply_button)

        # ---- Parameters: the PLOT's own tunable API params as GUI controls,
        # auto-discovered from the kind's declarative specs.  Each edit re-renders
        # the LIVE panel AND this snapshot.  This is where "the params the plot
        # call exposes" live -- NOT in the basic Setting popup (which keeps only
        # source / size / colormap / relim / unit), so the two never duplicate.
        functional = [s for s in _panel_display_decls(card.config.kind, card._param_kind()) if not s.display]
        if functional:
            section("Parameters")
            for s in functional:
                widget = card._make_param_widget(s, apply=self._edit_param)
                col.addWidget(FluentSettingRow(s.label, widget, label_width=scaled_px(96, minimum=72)))

        # ---- Display: the plot's DataFigure VIEW knobs (colormap / relim / unit) -- the
        # SAME controls as the Setting popup, present HERE too so the Edit tab is the FULL
        # data_figure UI (whatever DataFigure can do has a control here).  They write the
        # SAME config.params keys and drive the SAME live card via the card's handlers
        # (single source -- Setting and Edit never drift), then re-snapshot this Edit canvas.
        self.ed_cmap = self.ed_relim = self.ed_unit_button = self.ed_fixed_row = None
        self.ed_params: dict[str, QtWidgets.QWidget] = {}
        section("Display")
        disp_lw = scaled_px(96, minimum=72)
        # The SAME declarative display knobs as the Setting popup -- the per-kind colormap / toggles
        # PLUS the relim chooser -- rendered through the card's SHARED _emit_param_rows, so the Edit
        # tab auto-exposes EVERY plot display ParamDecl with no hand-wiring (#H3v-4b).  Each edit
        # routes via _edit_param -> the live card (config.params + re-render) THEN re-snapshots here.
        # relim AND repeat_mode: the SAME declarative display knobs the Setting popup renders (relim
        # via _RELIM_PARAM, repeat_mode via _repeat_param_specs) are present HERE too, so the Edit tab
        # is the FULL data_figure UI (whatever the Setting can tune, the Edit can too).  Both route
        # through _edit_param -> card._set_param -- the ONE writer -- so Setting and Edit never drift.
        display_specs = ([s for s in _panel_display_decls(card.config.kind, card._param_kind()) if s.display]
                         + [_RELIM_PARAM] + list(card._repeat_param_specs()))
        self.ed_params = card._emit_param_rows(display_specs, col.addWidget, self._edit_param, disp_lw)
        self.ed_cmap = self.ed_params.get("cmap")        # named back-refs (kept for tests / clarity)
        self.ed_relim = self.ed_params.get("relim")
        # fixed lo/hi: the IDENTICAL bespoke [lo | hi] row the Setting popup builds (shared helper),
        # shown only when relim == "fixed"; _edit_param toggles it when relim changes.  EXCEPTION --
        # a 2d/sites panel's value axis IS the COLOUR limit (clim), not a y-axis: its manual pin
        # belongs with the ranges in the Limits section as an always-editable "colour range" row
        # (built below), NOT here.  Building it in BOTH places would put two inputs on the ONE
        # fixed_lo/hi source = exactly the drift the single-source rule forbids, so for an image kind
        # the Display fixed row is absent and self.ed_fixed_* stay None (every reader is None-guarded);
        # the relim chooser (tight/normal/fixed clim MODE) still lives here.
        if card.config.kind in ("2d", "sites"):
            self.ed_fixed_row = self.ed_fixed_lo = self.ed_fixed_hi = None
        else:
            self.ed_fixed_row, self.ed_fixed_lo, self.ed_fixed_hi = \
                card._make_fixed_lim_row(self._edit_fixed_lim, disp_lw)
            col.addWidget(self.ed_fixed_row)
            # The lo/hi row stays PERMANENTLY in the layout; only its inputs enable when relim ==
            # "fixed".  A setVisible toggle on a row that sits ABOVE the snapshot canvas in this shared
            # scroll page reflowed the unit row + the whole canvas DOWN by the row's height on every
            # relim change -- the reported Edit-tab jump.  Greying in place leaves every widget put.
            self._sync_fixed_lim_enabled(card._relim())
        # x-axis unit cycle -- the IDENTICAL row the Setting popup builds (shared helper), minus the
        # Setting-only current-unit label; the callback re-routes through the card's _on_unit_cycle.
        ed_unit_row, self.ed_unit_button, _ = \
            card._make_unit_cycle_row(self._edit_unit_cycle, disp_lw, with_label=False)
        col.addWidget(ed_unit_row)

        # ---- Processing: frozen snapshot + Refresh (Save is its own row below) -------------
        section("Processing")
        head = QtWidgets.QHBoxLayout()
        head.addWidget(labeled("frozen snapshot of current data"), 1)
        self.refresh_button = FluentButton("Refresh", color=GREY)
        self.refresh_button.setToolTip("Re-snapshot the panel's current data")
        self.refresh_button.clicked.connect(self.rebuild)
        head.addWidget(self.refresh_button)
        col.addLayout(head)

        self.canvas_holder = QtWidgets.QVBoxLayout()
        self.canvas_holder.setContentsMargins(0, 0, 0, 0)
        col.addLayout(self.canvas_holder)

        # Fit + manual limits are a panel concern, so they always build here.  A LOGIC NODE's Edit (on the
        # Logic tab, a LogicNodeEditor) deliberately carries no curve fit -- fitting
        # a curve is a plotter action, so you Add a Plot panel pointed at the
        # signals the node publishes for that.  A PLOT keeps the FULL DataFigure
        # model set (#176).
        # DataFigure picks the 1D / 2D path from the snapshot, so a 2D image
        # fits the 2D-Gaussian "2D center" model and lines fit the rest.  Same
        # [label | control] row idiom as the sections above (aligned label
        # column), so Fit / limits don't read as a cramped one-line jumble.
        proc_lw = scaled_px(96, minimum=72)

        def _inline(*widgets, trailing=None):
            host = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(scaled_px(6, minimum=4))
            for w in widgets:
                row.addWidget(w, 0)
            row.addStretch(1)
            if trailing is not None:
                row.addWidget(trailing, 0)
            return host

        # ---- Analysis: the SAME control set the Setting popup builds, through the ONE
        # :class:`AnalysisControls` builder (``surface='edit'`` -- adds the explicit Fit / Clear
        # buttons + gains the action combo incl. ROI + a result row).  Every action routes through the
        # ONE card mutator (card.set_fit_request), so the Edit and Setting surfaces are both views of
        # config.params['fit_request'] and can never disagree (#8).  ``fit_combo`` / ``ed_fix_seed``
        # stay the attribute names tests key off; ``do_fit`` / ``clear_fit`` delegate to the composite.
        self.fit_combo = None
        self.ed_fix_seed = None
        self._analysis_controls = None
        if self.card is not None:
            controls = AnalysisControls(self.card, surface="edit", label_w=proc_lw)
            if controls.empty:
                controls.deleteLater()
            else:
                section("Analysis")
                col.addWidget(controls)
                self._analysis_controls = controls
                self.fit_combo = controls.model_combo
                self.ed_fix_seed = controls.fix_seed

        # x/y-range pins (#3): the VIEW window, applied to the LIVE panel AND every grid cell (global,
        # not the snapshot only).  The boxes EDIT THE STORED PIN (``view_xlim``/``view_ylim``), NOT the
        # live range -- they hold the pin value (or are empty = autoscale) and are only re-seeded on
        # build / tab-show / Clear, so typing is NEVER clobbered by the refresh tick.  The live window
        # is shown as the grey PLACEHOLDER (``refresh_limit_hints``), a non-destructive hint Qt draws
        # only while a box is empty.  A y-range row exists ONLY where y is a VIEW axis
        # (``_card_y_is_view_axis``: an image, whose x AND y are pixel coordinates -- pinning both is
        # what makes an ROI crop real, since the image keeps aspect='equal' and an x-only pin would
        # letterbox).  On a 1d/dis panel the y VALUE axis is already owned by the relim family
        # (tight/normal/fixed in the Display section), so it gets no y row.  An image's VALUE axis is
        # the COLOUR limit -- a genuinely distinct quantity -- pinned by its own "colour range" row.
        section("Limits")
        self.xmin = FluentLineEdit(""); self.xmax = FluentLineEdit("")
        self.ymin = self.ymax = None
        if _card_y_is_view_axis(card):
            self.ymin = FluentLineEdit(""); self.ymax = FluentLineEdit("")
        # an image's value axis IS the colour limit: give 2d/sites an always-editable clim pin (built
        # into the Limits row below); every other kind has none (its value axis = the relim y).
        self.clo = self.chi = None
        if card.config.kind in ("2d", "sites"):
            self.clo = FluentLineEdit(""); self.chi = FluentLineEdit("")
        for w in (self.xmin, self.xmax) + ((self.ymin, self.ymax) if self.ymin is not None else ()):
            w.setFixedWidth(scaled_px(88, minimum=68))   # wide enough not to clip "-0.4960"
            w.returnPressed.connect(self.apply_limits)
        apply_btn = FluentButton("Apply lim", color=ACCENT)
        apply_btn.clicked.connect(self.apply_limits)
        # Clear releases the window pins back to autoscale -- the Limits counterpart of the Fit/Clear
        # pair above (without it a pinned window could never be undone: clearing the boxes + Apply
        # only errored 'bad limits', so the operator was stuck at whatever range they once set).
        clear_lim_btn = FluentButton("Clear", color=GREY)
        clear_lim_btn.clicked.connect(self.clear_limits)
        lim_row = _inline(self.xmin, self.xmax, trailing=apply_btn)
        lim_row.layout().addWidget(clear_lim_btn, 0)
        col.addWidget(FluentSettingRow("x range", lim_row, label_width=proc_lw))
        if self.ymin is not None:
            # ONE Apply/Clear pair drives BOTH rows (x + y are one view window, applied together).
            col.addWidget(FluentSettingRow("y range", _inline(self.ymin, self.ymax), label_width=proc_lw))

        # colour range (2d/sites only): an image's value axis is its COLOUR limit, so it gets an
        # always-editable clim pin here, mirroring the x-range row.  Apply writes the ONE clim source
        # (relim="fixed" + the card's apply_fixed_lims -> live card + snapshot + save); Auto releases
        # it to the autoscaled clim (relim="normal").  So this row and the Display relim chooser never
        # hold two copies of the pin -- they are two faces of the same fixed_lo/hi source.
        if self.clo is not None:
            for w in (self.clo, self.chi):
                w.setFixedWidth(scaled_px(88, minimum=68))
                w.returnPressed.connect(self.apply_clim)
            clim_apply = FluentButton("Apply", color=ACCENT)
            clim_apply.clicked.connect(self.apply_clim)
            clim_auto = FluentButton("Auto", color=GREY)
            clim_auto.clicked.connect(self.clear_clim)
            clim_row = _inline(self.clo, self.chi, trailing=clim_apply)
            clim_row.layout().addWidget(clim_auto, 0)
            col.addWidget(FluentSettingRow("colour range", clim_row, label_width=proc_lw))

        # ---- Command: a one-line REPL on the panel's DataFigure (confocal runs the same
        # `data_figure.<fn>(...)` form).  Type e.g. `data_figure.xlim(0, 10)`,
        # `df.fit('gaussian')`, `ax.set_title('x')` -- it runs on the snapshot and
        # the result/exception shows below.  Trusted-local-tool posture (same as the Scan
        # tab / fit args): only run code you wrote.
        section("Command")
        self.cmd_input = FluentLineEdit("")
        self.cmd_input.setPlaceholderText("data_figure.xlim(0, 10)")
        self.cmd_input.setStyleSheet(self.cmd_input.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
        self.cmd_input.setToolTip(
            "Run a line of Python on this panel's figure.  Names: data_figure / df, fig, ax,\n"
            "plotter, np.  e.g. df.fit('gaussian') ; ax.set_title('x') ; data_figure.xlim(0,10)")
        cmd_run = FluentButton("Run", color=ACCENT)
        cmd_run.clicked.connect(self._run_command)
        self.cmd_input.returnPressed.connect(self._run_command)
        # the input FILLS the row (stretch 1) so it is not a tiny box; Run trails it.
        run_host = QtWidgets.QWidget()
        run_row = QtWidgets.QHBoxLayout(run_host)
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(scaled_px(6, minimum=4))
        run_row.addWidget(self.cmd_input, 1)
        run_row.addWidget(cmd_run, 0)
        col.addWidget(FluentSettingRow("run", run_host, label_width=proc_lw))
        # result: a read-only-but-COPYABLE field (FluentReadoutEdit) so you can select +
        # Ctrl-C the value/error -- a plain label can't be copied.
        self.cmd_result = FluentReadoutEdit("")
        self.cmd_result.setPlaceholderText("result / error appears here (select to copy)")
        self.cmd_result.setToolTip("The command's result or error — read-only, but select + Ctrl-C to copy.")
        col.addWidget(FluentSettingRow("result", self.cmd_result, label_width=proc_lw))

        # ---- Save: the ONE place to save from now (Setting no longer has Save).  The path
        # field is FULL WIDTH (its own row) so a long path is never cut off -- the reusable
        # FluentPathEdit (Browse picks a folder), prefilled with the LAST place a panel saved
        # (remembered on the console for the kernel session) or tasks/.  An "auto-name" SWITCH
        # decides the filename: ON (default) appends ``_<plot-kind>_<timestamp>`` so saves are
        # unique; OFF writes the path VERBATIM (you control the exact name, overwrites).  A
        # read-only preview shows the actual file that will be written -- the full name, not
        # just the folder.
        section("Save")
        self.save_dir_edit = FluentPathEdit(
            self.console._last_save_dir or str(_task_files_dir()),
            mode="dir", caption="Choose where to save", base_dir=str(_task_files_dir()))
        self.save_dir_edit.setToolTip(
            "Where to save (folder, or a full path base).  Remembered across saves this\n"
            "kernel session.  With auto-name OFF this is the exact output path.")
        col.addWidget(FluentSettingRow("path", self.save_dir_edit, label_width=proc_lw))
        # trailing spaces: FluentSwitch.sizeHint reserves the track width but paints the
        # label past track+gap, clipping the last few px -- the pad absorbs that (scales
        # with DPR since it is text), so the real label stays whole.
        self.save_autoname = FluentSwitch("auto-name (type + time)   ")
        self.save_autoname.setChecked(True)
        self.save_autoname.setToolTip(
            "ON: append _<plot-kind>_<timestamp> to the path (unique files).\n"
            "OFF: write the path verbatim (you set the exact name; overwrites).")
        # image FORMAT: a DATA-layer choice of the output CONTAINER (png / pdf / jpg), NOT an art
        # knob -- the figure's geometry/dpi/typography are unchanged, only the file it lands in.
        # It drives ``DataFigure.save(image_ext=...)`` (the ONE save path) and the file suffix;
        # the .npz payload is format-independent (same data + info for every choice).  Lowercase
        # extensions match confocal's ``save_type`` convention (jpg/png).
        self.save_format_combo = FluentComboBox()
        self.save_format_combo.addItems(list(SAVE_IMAGE_FORMATS))
        self.save_format_combo.setCurrentText(SAVE_IMAGE_FORMATS[0])
        self.save_format_combo.setFixedWidth(scaled_px(72, minimum=56))
        self.save_format_combo.setToolTip(
            "Image container for the saved figure (png / pdf / jpg).\n"
            "The matching .npz data file is the same for every format.")
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        self.save_button.setToolTip("Save the edited figure (png / pdf / jpg) + matching data (npz).")
        col.addWidget(FluentSettingRow(
            "name", _inline(self.save_autoname, self.save_format_combo, trailing=self.save_button),
            label_width=proc_lw))
        # The previewed file path is a read-only-but-COPYABLE field (NOT a word-wrap label):
        # a long absolute path has no spaces to wrap on, so a word-wrap label would force its
        # own width to the full path and DRAG the whole Edit panel wider (the bug).  A
        # FluentReadoutEdit is a QLineEdit -- its size hint is content-INDEPENDENT (it scrolls
        # a long path instead of widening), so the panel width never tracks the path length,
        # and the resolved name stays selectable + Ctrl-C-able.
        self.save_preview = FluentReadoutEdit("")
        self.save_preview.setToolTip("The exact file that will be written — read-only, select to copy.")
        self.save_preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        col.addWidget(FluentSettingRow("file", self.save_preview, label_width=proc_lw))
        self.save_dir_edit.changed.connect(lambda *_: self._update_save_preview())
        self.save_autoname.toggled.connect(lambda *_: self._update_save_preview())
        self.save_format_combo.currentTextChanged.connect(lambda *_: self._update_save_preview())
        self.save_button.clicked.connect(self.save)
        self._update_save_preview()

        self.status = FluentLabel("")
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        # A status line must NEVER drive the page WIDTH: a QLabel's sizeHint width tracks its text, so a
        # long message (e.g. "saved … → C:\very\long\path") would balloon the Edit column into a
        # horizontal scroll.  Ignored horizontal policy makes the label take the available width and CLIP
        # instead -- the layout width is owned by the canvas/rows, never by a log string (#layout-fixed).
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        col.addWidget(self.status)
        col.addStretch(1)

        self.rebuild()

    # ------------------------------------------------------------- snapshot
    def teardown(self) -> None:
        if self._canvas is not None:
            self.canvas_holder.removeWidget(self._canvas)
            self._canvas.deleteLater()
        if self._plotter is not None and plt is not None and self._plotter.fig is not None:
            plt.close(self._plotter.fig)
        self._canvas = None
        self._plotter = None
        self._df = None

    def rebuild(self) -> None:
        """Snapshot the bound card's CURRENT data, MIRRORING the Monitor frame.

        The snapshot uses the card's OWN size (not a forced 2x4) so it looks
        exactly like the Monitor panel and can never overflow the Edit page (the
        Monitor card already fits) -- and unlike the Monitor card it is built
        INTERACTIVE (default selectors on), so zoom / region-select work here."""
        card = self.card
        if card is None or panel_canvas is None:
            self.status.setText("open the panel with data first")
            return
        # Snapshot the MOST RECENT frame: pull the latest hub data into the Monitor
        # card first (one synchronous tick), so Refresh mirrors what the camera just
        # produced -- not the last timer-tick render (which can lag by a beat).  This
        # refresh is BEST-EFFORT: a mid-session signal that has no retained state at the
        # current display provenance raises SignalHistoryGap, and that must NOT abort the
        # Edit tab's construction -- opening the tab must never depend on a clean hub tick
        # (#1: 'edit tab 点不开', intermittent).  On failure we snapshot whatever the
        # plotter already holds; a later timer tick refreshes it.
        try:
            self.console.refresh_once()
        except Exception:
            pass
        if card.plotter is None:
            self.status.setText("open the panel with data first")
            return
        src = card.plotter
        kind = card.config.kind
        size = card.config.size          # mirror the Monitor frame, never force 2x4
        title = card.config.title or PANEL_KINDS[kind]
        view = card._view_kwargs(kind)   # ONE source: relim mode + fixed lo/hi -- for sites too (#4)
        cmap = _resolved_cmap(kind, card.config.params)   # operator pick, else the kind default (ONE resolver)
        try:
            if kind == "pulse":
                # A pulse panel's snapshot rebuilds through the SAME renderer the live card uses
                # (``build_pulse_preview_plot`` via the console's ``_pulse_state`` provider) -- NEVER the
                # else (1d/monitor) branch, which reads ``src.data_x``/``data_y`` a PulseSequenceFigure does
                # not carry (that was the blank-Edit-tab bug).  ``include_always_off`` comes from the card's
                # own params (same source as the live card), falling back to the node's recorded value.
                from .live import build_pulse_preview_plot
                resolved = self.console._pulse_state(card.config.inputs[0]) if card.config.inputs else None
                if resolved is None:
                    raise ValueError("pulse panel has no producing node to snapshot")
                state, node_include_off = resolved
                include_off = bool(card.config.params.get("include_always_off", node_include_off))
                new_plotter, _channels, _repeat = build_pulse_preview_plot(
                    state, include_always_off=include_off, size=size)   # Edit tab: interactive (default)
            elif kind == "grid":
                # A grid panel's snapshot rebuilds through the SAME builders the live card uses --
                # never the array else branch.  A FACET grid re-slices the card's last good data
                # through the ONE shared builder (_build_facet_plotter -- interactive here, like
                # every Edit snapshot); a recipe grid rebuilds via the _grid_recipe provider.
                if card._facet():
                    if card._last_namespace is None:
                        raise ValueError("facet grid has no data yet to snapshot")
                    new_plotter = card._build_facet_plotter(
                        card._signal_then_repeat(card._last_namespace), interactions=True)
                else:
                    from .live import build_grid_figure
                    recipe = self.console._grid_recipe(card.config.inputs[0]) if card.config.inputs else None
                    if recipe is None:
                        raise ValueError("grid panel has no producing node to snapshot")
                    recipe = card._grid_recipe_with_params(recipe)   # fold the panel's live display knobs in
                    new_plotter = build_grid_figure(recipe, interactions=True, size=size, display=False)
            elif kind == "2d":
                new_plotter = panel_plot(np.array(src.data_x, dtype=float),
                                         np.array(src.data_y[:, 0], dtype=float), kind="2d",
                                         size=size, cmap=cmap, **view,
                                         labels=tuple(src.labels), title=title)
            elif kind == "sites":
                new_plotter = panel_plot(np.array(src.data_x[:, :2], dtype=float),
                                         np.array(src.data_y[:, 0], dtype=float), kind="sites",
                                         size=size, image=getattr(src, "background", None),
                                         roi_radius=getattr(src, "roi_radius", 3.0), cmap=cmap, **view,
                                         labels=tuple(src.labels), title=title)
            elif kind == "hist":
                # Same declared-default resolver as the live build (_resolved_param) -- the Edit
                # snapshot can never render a different default than the Monitor card.
                new_plotter = panel_plot(np.array(src.values, dtype=float), kind="hist", size=size,
                                         bins=int(_resolved_param(kind, card.config.params, "bins")),
                                         fit=str(_resolved_param(kind, card.config.params, "fit")),
                                         ylog=bool(_resolved_param(kind, card.config.params, "ylog")), **view,
                                         labels=tuple(src.labels), title=title)
            else:  # 1d / monitor -> a line snapshot (monitor honours its show_dist toggle)
                extra = ({"show_dist": bool(_resolved_param(kind, card.config.params, "show_dist"))}
                         if kind == "monitor" else {})
                # Pass the live plotter's FULL data_y (all columns), not just column 0: a 1d ``create``
                # repeat_mode collapses to a ``(points, R)`` array = one line per repeat, and the snapshot
                # must draw EVERY column exactly like the live card (Live1D plots one line per column) --
                # else the Edit-tab preview would ignore repeat_mode and always show a single line.
                new_plotter = panel_plot(np.array(src.data_x[:, 0], dtype=float),
                                         np.array(src.data_y, dtype=float),
                                         kind=kind, size=size, **view, **extra,
                                         labels=tuple(src.labels), title=title)
        except Exception as exc:
            self.status.setText(f"could not snapshot: {str(exc).splitlines()[0][:120]}")
            return                       # build failed -> keep the LAST good snapshot, never blank (#4 "图有时消失")
        # Build-then-swap: the new plotter built cleanly, so ONLY NOW tear down the old snapshot and
        # install the new -- a failed rebuild above left the previous figure intact.
        # A grid snapshot may be showing an ENLARGED cell (GridPlot.focus): re-enlarge the SAME cell
        # on the fresh snapshot, so a knob that genuinely needs a re-snapshot still returns the user
        # to the zoomed view -- the generic mirror of the card's own refocus-across-rebuild rule.
        refocus_k = getattr(self._plotter, "_focused", None)
        self.teardown()
        self._plotter = new_plotter
        if refocus_k is not None and hasattr(self._plotter, "focus"):
            self._plotter.focus(int(refocus_k))
        self._canvas = panel_canvas(self._plotter.fig)
        # The Edit page lives in a SCROLL AREA: the page SCROLLS when the figure is taller than the
        # viewport, and the snapshot must never squish (clip) NOR grow.  The canvas already setFixedSize's
        # itself to its DPR-invariant design size at construction (and re-asserts it on every resync), so
        # pin_size() is now idempotent -- the deferred _zlc_resync / showEvent can no longer re-open the
        # size and balloon the figure across param edits (that was the setMinimumSize(sizeHint()) race).
        self._canvas.pin_size()
        self.canvas_holder.addWidget(self._canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self._canvas.draw()      # SYNCHRONOUS: the snapshot shows its frame at once (never a blank)
        self._df = None
        # carry the persisted x-axis unit cycle onto the fresh snapshot (relim + cmap are
        # already baked into panel_plot above; unit is applied post-build, like the live card).
        self._apply_snapshot_unit()
        self._seed_limit_boxes()        # boxes = the stored pin (or empty=auto); grey hint = live range
        self.refresh_limit_hints()
        # confocal-style auto-range: interacting with this (interactive) Edit plot
        # writes the region the user marked back to the source's parameters.  The
        # selector / zoom is a GENERIC interface -- it only ever yields the
        # rectangle (endpoints) / view limits in plot coords; the SOURCE converts
        # that to its own format (a measurement -> scan range; a camera node ->
        # ROI; a 2-D scan -> axis ranges), so no device shape leaks into the GUI.
        # The SAME callback is bound to BOTH the zoom/scroll AND the area selector:
        # ZOOM/PAN updates from the view limits, the area selector OVERRIDES when
        # drawn (precedence in the writeback).  (Wiring only the area selector --
        # the earlier bug -- meant zoom did nothing.)
        area = getattr(self._plotter, "area", None)
        zoom = getattr(self._plotter, "zoom", None)
        # A plot is a pure view that only ever yields the marked rectangle (endpoints) in plot coords;
        # the PRODUCING node converts that to its own parameter -- so no device/measurement shape leaks
        # into the GUI, and the SAME rule serves every source:
        #   * a 2D IMAGE  -> the upstream camera node's ROI (endpoints -> sub-array), on zoom AND select;
        #   * a 1D CURVE of a scanning measurement -> that measurement's scan x-range (its axis_range
        #     param), on a DELIBERATE drag-select only (a mere zoom must not restage the scan).
        # The 1-D case is gated on the node DECLARING a scan range (``_node_scan_range_key``), so EVERY
        # measurement with an axis_range param gets the linkage by construction (never wired per node).
        if kind == "2d" and self._node is not None:
            if area is not None:
                area.callback = self._read_region
            if zoom is not None:
                zoom.callback = self._read_region
        elif self._node is not None and self.console._node_scan_range_key(self._node) is not None:
            if area is not None:
                area.callback = self._read_x_range
        self.status.setText("snapshot of current data — fit / set limits / save are frozen here"
                            if self.fit_combo is not None else
                            "snapshot of current data — Save Fig is frozen here")

    def _edit_param(self, key: str, value) -> None:
        """A plot-param edit from the Edit tab (declarative display knob OR functional param): apply to
        the LIVE panel via the card's ONE _set_param (config.params + live re-render -- single source,
        identical to the Setting popup), reveal the Edit tab's OWN fixed lo/hi row when relim flips to
        ``fixed``, then re-snapshot this canvas so the change shows in both surfaces (#H3v-4b)."""
        if self.card is not None:
            self.card._set_param(key, value)
        if key == "relim" and getattr(self, "ed_fixed_row", None) is not None:
            self._sync_fixed_lim_enabled(str(value))   # enable lo/hi in fixed WITHOUT moving the page
            if str(value) == "fixed" and self.card is not None:
                # mirror the freeze-current-view seed _set_param just wrote (config.params is the
                # one source) into THIS tab's lo/hi inputs -- setText does not re-fire editingFinished
                for edit, pkey in ((self.ed_fixed_lo, "fixed_lo"), (self.ed_fixed_hi, "fixed_hi")):
                    if edit is not None and pkey in self.card.config.params:
                        edit.setText(f"{float(self.card.config.params[pkey]):g}")
        if key == "relim":
            # a 2d/sites image has no Display fixed row (its clim lives in the Limits colour-range row):
            # re-seed THOSE boxes here so picking relim in the chooser fills/empties them to match the
            # pin -- runs even when ed_fixed_row is None, unlike the block above (#2 colour range).
            self._seed_clim_boxes()
        # EVERY knob first tries the snapshot's own in-place apply (BaseLivePlot.apply_param handles
        # the relim family for every kind; a GridPlot stores + forwards to its focused cell and NEVER
        # asks for a rebuild) -- rebuild() is only the fallback for a knob the snapshot truly cannot
        # apply, because it tears down + recreates the whole snapshot (which would throw away a
        # focused grid cell = the "changing lim bounces the enlarged cell back to the grid" bug).
        snap = getattr(self, "_plotter", None)
        if snap is not None and snap.apply_param(key, value):
            if getattr(self, "_canvas", None) is not None:
                self._canvas.draw_idle()
            return
        self.rebuild()

    def _sync_fixed_lim_enabled(self, relim: str) -> None:
        """The Edit tab's fixed lo/hi row is ALWAYS in the layout -- only its INPUTS enable when
        ``relim == "fixed"``.  A ``setVisible`` toggle on a row that sits above the snapshot canvas in
        the shared scroll page reflowed the unit row + the whole canvas down by the row's height on
        every relim change (the reported Edit-tab jump); greying the inputs in place leaves every
        widget put.  (The Setting popup keeps its reveal-and-grow-down: it is a compact floating card
        with nothing beneath the row, so growing downward moves nothing.)"""
        if self.ed_fixed_row is not None:
            # ALWAYS shown here (the shared _make_fixed_lim_row hides it by default for the Setting
            # popup's reveal-on-fixed) -- so the footprint is constant and the canvas never moves.
            self.ed_fixed_row.setVisible(True)
        fixed = str(relim) == "fixed"
        for w in (self.ed_fixed_lo, self.ed_fixed_hi):
            if w is not None:
                w.setEnabled(fixed)

    def _edit_fixed_lim(self) -> None:
        """The Edit tab's fixed lo/hi committed (#H2): apply to the LIVE card through its ONE
        ``apply_fixed_lims`` path (config.params + in-place apply_param; a zoomed grid also stores
        onto the parked grid), then push onto THIS tab's SNAPSHOT in place too (the same apply_param
        relim family; a GridPlot forwards to its focused cell) -- never a re-snapshot, which would
        bounce a focused grid cell back to the grid."""
        if self.card is None:
            return
        lo = _safe_float(self.ed_fixed_lo.text(), 0.0)
        hi = _safe_float(self.ed_fixed_hi.text(), 1.0)
        self.card.apply_fixed_lims(lo, hi)
        snap = getattr(self, "_plotter", None)
        if snap is not None and snap.apply_param("fixed_lo", lo) and snap.apply_param("fixed_hi", hi):
            if getattr(self, "_canvas", None) is not None:
                self._canvas.draw_idle()
            return
        self.rebuild()

    def _edit_unit_cycle(self) -> None:
        if self.card is not None:
            self.card._on_unit_cycle()            # config.params["unit_index"] + live cycle
        self.rebuild()

    def _edit_title(self, text: str) -> None:
        """Rename the panel from the Edit tab: route through the card's sealed title handler
        (config.title + live plot title via style.apply_title) and keep the Setting popup's
        title field in sync, then re-snapshot + refresh the save preview."""
        if self.card is None:
            return
        self.card._on_title(str(text))            # config.title + sealed live title
        edit = getattr(self.card, "title_edit", None)
        if edit is not None and edit.text() != text:
            with _signals_blocked(edit):
                edit.setText(str(text))
        self._update_save_preview()               # default save name follows the title
        self.rebuild()

    def _run_command(self) -> None:
        """A one-line REPL on this panel's DataFigure (confocal's ``data_figure.<fn>(...)``).
        Names exposed: ``data_figure``/``df``, ``fig``, ``ax``, ``plotter``, ``np``.  An
        expression shows its repr; a statement shows ``ok``; an error shows its message.
        SECURITY: runs arbitrary local Python -- trusted-input tool, like the Scan tab."""
        if self._plotter is None or not hasattr(self, "cmd_input"):
            return
        text = self.cmd_input.text().strip()
        if not text:
            return
        try:
            df = self._df_for()
            ns = {"data_figure": df, "df": df, "fig": self._plotter.fig,
                  "ax": getattr(self._plotter, "ax", None), "plotter": self._plotter, "np": np}
            try:
                value = eval(text, ns)            # noqa: S307 - local experiment tool, trusted input
                msg = "ok" if value is None else repr(value)
            except SyntaxError:
                exec(text, ns)                    # noqa: S102 - a statement, not an expression
                msg = "ok"
            if self._canvas is not None:
                self._canvas.draw_idle()
            self.cmd_result.setText(str(msg)[:300])
        except Exception as exc:
            self.cmd_result.setText(f"error: {str(exc).splitlines()[0][:200]}")

    def _apply_snapshot_unit(self) -> None:
        """Cycle the Edit snapshot's x-axis unit ``unit_index`` times from its original,
        mirroring the live card's :meth:`PanelCard._apply_unit` (shared :func:`_unit_df_for`).
        Called on every rebuild so the persisted unit survives a Refresh / re-snapshot."""
        if self._plotter is None or self.card is None:
            return
        index = int(self.card.config.params.get("unit_index", 0) or 0)
        if index <= 0:
            return
        try:
            df = _unit_df_for(self._plotter)
            if not getattr(df, "conversion_map", None):
                return
            for _ in range(index % max(1, len(df.conversion_map))):
                df.change_unit()
            if self._canvas is not None:
                self._canvas.draw_idle()
        except Exception:
            pass

    def _selected_rect(self):
        """The area rectangle if one is drawn, ELSE the current view box (x AND y
        view limits).  Returns (xlo, xhi, ylo, yhi) sorted, or all None -- so
        scroll-zoom alone sets the region, with a drag-rectangle overriding it."""
        plotter = self._plotter
        if plotter is None or plotter.ax is None:
            return (None, None, None, None)
        area = getattr(plotter, "area", None)
        if area is not None and area.range[0] is not None:
            x1, x2, y1, y2 = (float(v) for v in area.range)
            return (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        xlo, xhi = sorted(float(v) for v in plotter.ax.get_xlim())
        ylo, yhi = sorted(float(v) for v in plotter.ax.get_ylim())
        return (xlo, xhi, ylo, yhi)

    def _read_region(self) -> None:
        """Confocal ``_read_range`` for a 2D panel, GENERIC over the source.

        ZOOM/PAN or an area SELECTION yields the marked rectangle as endpoints
        (x_min, x_max, y_min, y_max) in the panel's axis coordinates (the area
        selection overrides the view box when drawn).  Those ENDPOINTS -- the only
        thing the selector knows -- are handed to the producing source via
        ``region_to_acquisition_parameters``; the SOURCE converts them to its own
        Acquisition parameters (a camera node -> a ROI rectangle; a 2-D scan ->
        axis ranges).  Whatever fields it names are filled in the Edit form, then
        Apply pushes them.  The frontend encodes NO device-specific shape."""
        if self._node is None or self._plotter is None:
            return
        convert = getattr(self._node, "region_to_acquisition_parameters", None)
        if convert is None:
            return
        x1, x2, y1, y2 = self._selected_rect()
        if x1 is None:
            return
        params = convert(x1, x2, y1, y2) or {}
        filled = {}
        for name, value in params.items():
            edit = self._node_widgets.get(name)
            if edit is not None:
                edit.setText(_py_to_text(value))
                filled[name] = value
        if filled:
            self.status.setText(f"region from view: {filled} — Apply to use it")

    def _read_x_range(self) -> None:
        """Confocal ``_read_range`` for a 1-D CURVE panel, GENERIC over the source.

        A drag-selected x-interval on a scanning measurement's plot becomes that measurement's NEXT
        scan x-range.  The selector knows only the endpoints (plot coords); they are staged onto the
        PRODUCING measurement's OWN Logic-tab Edit form (its first ``axis_range`` param, via
        ``MeasurementPanel.set_axis_range``), so ANY measurement that declares a scan range gets this by
        construction and the plot stays a pure view with no measurement internals.  ``Start`` on that
        form then re-runs the scan over the marked range (confocal's mark-then-rerun)."""
        if self._node is None:                     # the plotter-dependence lives in _selected_rect (None-safe)
            return
        x1, x2, y1, y2 = self._selected_rect()
        if x1 is None:
            return
        form = self.console._form_for_node(self._node)
        if form is not None and form.set_axis_range(x1, x2):
            self.status.setText(f"scan range from view: [{x1:g}, {x2:g}] — Start to use it")

    def refresh_node_now_labels(self) -> None:
        """Update the 'now: <value>' references beside each Acquisition field to the
        source's CURRENT values.  The console calls this each tick for the visible
        Edit tab (one general hook, not a per-field signal), so after the loop
        applies a queued edit the references catch up on their own -- no manual
        wiring per parameter, and the frozen snapshot / Refresh / Fit controls are
        untouched."""
        if self._node is None or not self._node_now_labels:
            return
        for name, current in self.console._node_params(self._node):
            label = self._node_now_labels.get(name)
            if label is not None:
                label.setText(f"now: {_py_to_text(current)}")

    def _restart_node(self) -> None:
        """Apply the edited Acquisition params to the data source, routed through the
        node's safe entry (queued + applied BETWEEN shots in the loop's own thread,
        never a GUI-thread stop/start); Apply also STARTs an idle source so it goes
        live.

        Re-snapshot timing is the subtle part.  An IDLE node applies + publishes the
        new-param frame SYNCHRONOUSLY (inside ``apply_acquisition_parameters``), so we
        refresh + re-snapshot right away.  A RUNNING node only publishes the new-param
        frame on its NEXT loop iteration, so reading the hub now would snapshot the
        STALE pre-edit frame -- so we wait for the node's acquisition epoch to advance
        (the first frame computed with the new params is on the hub) and re-snapshot
        THAT, polled off the GUI critical path (no acquire from this thread)."""
        if self._node is None:
            return
        new_params = {name: _text_to_py(w.text()) for name, w in self._node_widgets.items()}
        running = bool(getattr(self._node, "running", False))
        epoch0 = int(getattr(self._node, "acquisition_epoch", lambda: 0)())
        try:
            self.console._restart_node(self._node, new_params)
        except Exception as exc:
            self.status.setText(f"apply failed: {str(exc).splitlines()[0][:120]}")
            return
        self.refresh_node_now_labels()        # reuse the same refresh the tick uses
        if running:
            # queued: the loop has not yet produced a new-param frame -- defer the snapshot
            self.status.setText("acquisition parameters queued — Monitor updates on the next frame")
            self._await_fresh_frame(self._node, epoch0)
        else:
            # idle: apply published a fresh frame synchronously, so it is already here
            self.console.refresh_once()
            self.status.setText("acquisition started with the new parameters")
            self.rebuild()

    def _apply_source_form(self) -> None:
        """Apply the SOURCE node's edited parameter form (#2) to the producing node --
        rebuild + restart it (or live where it accepts), then re-snapshot."""
        if self._source_row is None or self.source_form is None:
            return
        try:
            values = self.source_form.collect_values()
        except Exception as exc:
            self.status.setText(f"apply failed: {str(exc).splitlines()[0][:120]}")
            return
        if not self.console._apply_source_params(self._source_row, values):
            self.status.setText("locked: a task is running — Stop it first")
            return
        self.console.refresh_once()
        self.status.setText("source parameters applied")
        self.rebuild()

    def _await_fresh_frame(self, node, epoch0: int, *, _tries: int = 0) -> None:
        """Poll (off the GUI critical path) until the running node has published its
        FIRST frame computed with the just-queued params -- detected by its
        ``acquisition_epoch`` advancing past ``epoch0`` -- then refresh + re-snapshot
        the Edit panel.  Bounded so a wedged/slow node never spins forever (it falls
        back to a refresh after the cap).  Aborts if the editor moved to another node
        or was torn down."""
        if self._node is not node or node is None:
            return
        epoch = int(getattr(node, "acquisition_epoch", lambda: epoch0 + 1)())
        if epoch > epoch0 or _tries >= 600:    # ~600 * 25 ms = 15 s safety cap
            self.console.refresh_once()
            self.refresh_node_now_labels()
            self.rebuild()
            return
        from PyQt5 import QtCore
        QtCore.QTimer.singleShot(25, lambda: self._await_fresh_frame(node, epoch0, _tries=_tries + 1))

    def _df_for(self):
        # Route through the plotter's OWN to_data_figure() -- the SINGLE source that already knows a
        # GridPlot is N per-cell DataFigures (returns a _GridData composite) while every other kind is a
        # flat DataFigure.  Building DataFigure(self._plotter) directly would collapse a grid to cell-0's
        # axes over placeholder arrays, so fit/limits would touch one subplot with garbage; going through
        # the override makes fit_targets()/xlim()/ylim() fan out over every cell.  Cached so repeated
        # Fit/Clear reuse the SAME per-cell handles (clear_fit clears this instance, not a fresh one).
        if self._df is None:
            self._df = self._plotter.to_data_figure()
        return self._df

    def do_fit(self) -> None:
        """Apply a curve fit from the Edit tab through the ONE :class:`AnalysisControls` composite (which
        funnels into card.set_fit_request), so the live overlay, the hub node, and the Setting popup's
        Analysis controls all follow the same config.params['fit_request'] state.  No-op when the panel
        family offers no fit (kept as a public method the tests + Fit button call)."""
        controls = getattr(self, "_analysis_controls", None)
        if controls is None:
            return
        controls.do_fit()
        if self.fit_combo is not None and self.fit_combo.currentData():
            self.status.setText(f"fit {self.fit_combo.currentText()}: applied (see panel)")

    def clear_fit(self) -> None:
        controls = getattr(self, "_analysis_controls", None)
        if controls is not None:
            controls.clear_fit()
        elif self.card is not None:
            self.card.set_fit_request(None)
        self.status.setText("fit cleared")

    def _refresh_edit_analysis(self) -> None:
        """Re-derive the Edit tab's Analysis controls from the card's stored state -- so the Edit surface
        always shows the SAME fit the Setting popup set, the two being pure views of the single source (#8)."""
        controls = getattr(self, "_analysis_controls", None)
        if controls is not None:
            controls.derive()

    def refresh_on_show(self) -> None:
        """When this panel's Edit tab becomes visible, refresh anything that may have changed
        since it was last shown: re-seed the display knobs from config.params, snap the manual-limit
        fields to current view, re-read the producing source's 'now' acquisition values, and refresh
        any embedded source form's dynamic combos (the ONE hook the tab-switch handler calls -- nothing
        special-cases PanelEditor versus LogicNodeEditor)."""
        # A grid's param rows follow its RESOLVED per-cell kind: when a facet / sub-plot pick changed
        # it since this page was built, rebuild the page in place (fresh rows + snapshot) -- the Edit
        # mirror of the Setting popup's _sync_settings_param_rows.
        card = self.card
        if card is not None and card.config.kind == "grid" \
                and card._param_kind() != self._param_kind_built:
            self.console._refresh_panel_editor(card)
            return
        self._refresh_display_params()
        self._refresh_edit_analysis()   # re-seed the fit model + fix/seed from config.params['fit_request']
        self._seed_limit_boxes()        # re-seed the pin from config.params (may have changed in Setting)
        self.refresh_limit_hints()
        self.refresh_node_now_labels()
        hook = getattr(getattr(self, "source_form", None), "refresh_on_show", None)
        if callable(hook):
            hook()

    def _refresh_display_params(self) -> None:
        """Re-seed the Edit tab's display-knob controls (``ed_params``) from the live card's
        ``config.params`` -- the SINGLE source of truth -- so switching back to this tab shows the
        CURRENT values even when they were changed in the Setting popup (which writes the same
        config.params).  Each control is re-seeded through its kind's ``PARAM_WIDGETS.write`` (the card
        records the kinds while building both surfaces' rows), signals blocked so re-seeding does not
        re-fire ``_edit_param``.  The #6 mirror of ``PanelCard.refresh_on_show`` -- both surfaces are a
        VIEW of config.params, refreshed on show, never private copies that drift."""
        if self.card is None:
            return
        kinds = getattr(self.card, "_param_kinds", {})
        params = self.card.config.params
        for key, widget in self.ed_params.items():
            kind = kinds.get(key)
            if kind is None or key not in params:
                continue
            with _signals_blocked(widget):
                try:
                    PARAM_WIDGETS[kind].write(widget, params[key])
                except (TypeError, ValueError):
                    continue

    def _limits_ax(self):
        """The axes the x-range boxes are a live VIEW of (#3): the LIVE card's plotter when it exists (so a
        zoom / pan on the panel reflects here), else the Edit-tab snapshot.  For a grid the x-window is
        shared across cells, so cell-0's axes is representative."""
        for plotter in (getattr(self.card, "plotter", None) if self.card is not None else None, self._plotter):
            if plotter is None:
                continue
            ax = getattr(plotter, "ax", None)
            if ax is None:
                axes = getattr(plotter, "site_axes", None)     # a grid: cells share one x-window
                ax = axes[0] if axes else None
            if ax is not None:
                return ax
        return None

    def _seed_limit_boxes(self) -> None:
        """Put the STORED x-window pin (``view_xlim`` in ``config.params``) into the boxes.  The boxes
        EDIT THE PIN, never the live autoscaled range, so their text never wanders on its own: empty
        boxes mean 'no pin' (autoscale), a value means 'pinned there'.  Called on build / tab-show /
        after Clear -- NEVER on the refresh tick (the root cause of the old 'I type and it rewrites'
        bug was ``fill_limits`` doing ``setText`` every tick).  The live range is shown separately as a
        non-destructive grey hint (:meth:`refresh_limit_hints`)."""
        if self.xmin is None:
            return                          # no Limits controls on this editor instance
        for key, lo_box, hi_box in self._limit_axes():
            pin = self.card.config.params.get(key) if self.card is not None else None
            lo = hi = ""
            if pin is not None:
                try:
                    lo, hi = f"{float(pin[0]):.6g}", f"{float(pin[1]):.6g}"
                except (TypeError, ValueError, IndexError):
                    lo = hi = ""
            with _signals_blocked(lo_box, hi_box):
                lo_box.setText(lo); hi_box.setText(hi)
        self._seed_clim_boxes()          # keep the 2d/sites colour-range boxes in step with the clim pin

    def refresh_limit_hints(self) -> None:
        """Refresh ONLY the grey PLACEHOLDER of the x-range boxes to the panel's current x-window -- a
        non-destructive live reference.  Qt shows a placeholder ONLY while the box is empty, so this can
        never overwrite a pinned value or the operator's in-progress typing.  This is the tick-safe
        successor of the old ``fill_limits`` (whose per-tick ``setText`` clobbered input): the tick calls
        THIS, so the boxes stay a live VIEW of the x-window (unpinned) while remaining freely editable."""
        if self.xmin is None:
            return                          # no Limits controls on this editor instance
        # colour-range (2d/sites) live hint: show the current clim as the grey PLACEHOLDER so an empty
        # box (= Auto) still tells the operator what range the image is using -- the clim counterpart of
        # the x-window hint below.  Non-destructive: Qt draws a placeholder only while the box is empty,
        # so a pinned/typed value is never overwritten.
        if getattr(self, "clo", None) is not None and self.card is not None \
                and self.card.plotter is not None and hasattr(self.card.plotter, "current_lims"):
            try:
                clo, chi = self.card.plotter.current_lims()
                self.clo.setPlaceholderText(f"{clo:.6g}"); self.chi.setPlaceholderText(f"{chi:.6g}")
            except Exception:
                pass
        ax = self._limits_ax()
        if ax is None:
            return
        xlo, xhi = ax.get_xlim()
        self.xmin.setPlaceholderText(f"{xlo:.6g}"); self.xmax.setPlaceholderText(f"{xhi:.6g}")
        if self.ymin is not None:           # y hint only where y is a view axis (an image family)
            ylo, yhi = ax.get_ylim()
            self.ymin.setPlaceholderText(f"{ylo:.6g}"); self.ymax.setPlaceholderText(f"{yhi:.6g}")

    def _limit_axes(self) -> list[tuple[str, object, object]]:
        """The (param key, lo box, hi box) rows this Edit's Limits section carries -- x always, y only
        on an image family (the y row exists iff ``_card_y_is_view_axis`` at build).  The ONE list
        apply/clear/seed/hints iterate, so adding an axis row can never miss a handler."""
        axes = [("view_xlim", self.xmin, self.xmax)]
        if self.ymin is not None:
            axes.append(("view_ylim", self.ymin, self.ymax))
        return axes

    def apply_limits(self) -> None:
        """Apply the typed view-window pins -- PER AXIS: a filled pair pins that axis, an empty pair
        releases it (so 'clear the y boxes + Apply' un-pins y while keeping x).  Each pin is an ORDINARY
        display knob (``view_xlim``/``view_ylim``) applied through the SAME ``_edit_param`` entry as
        bins / fit / relim -> the LIVE card (whose ``apply_param`` fans it to EVERY cell and re-asserts
        it after each tick's autoscale) + the snapshot + ``config.params`` + save.  So Apply reaches the
        RUNNING grid and STICKS (#3), not a one-shot on a throwaway ``_GridData`` the next redraw
        autoscaled away."""
        if self.xmin is None:
            return
        applied, cleared = [], []
        for key, lo_box, hi_box in self._limit_axes():
            lo_text, hi_text = lo_box.text().strip(), hi_box.text().strip()
            if not lo_text and not hi_text:
                self._edit_param(key, None)          # empty pair = release THIS axis's pin
                cleared.append(key[5])               # 'x' / 'y'
                continue
            try:
                lo, hi = float(lo_text), float(hi_text)
            except ValueError as exc:
                self.status.setText(f"bad limits: {str(exc).splitlines()[0][:100]}")
                return
            self._edit_param(key, (lo, hi))
            applied.append(key[5])
        parts = ([f"{'/'.join(applied)} range applied"] if applied else []) + \
                ([f"{'/'.join(cleared)} range cleared"] if cleared else [])
        self.status.setText("; ".join(parts) + " (all subplots)")

    def clear_limits(self) -> None:
        """Release EVERY view-window pin -> autoscale.  ``None`` is the SAME stored display knob
        ``apply_limits`` writes, routed through the ONE ``_edit_param`` entry: BaseLivePlot.apply_param /
        GridCell.consume_param already read ``None`` as 'no pin', so the live card, the snapshot, and the
        save all drop the pin.  EMPTIES the boxes (empty = no pin) so the grey placeholder hint takes
        over showing the now-auto range -- the Limits counterpart of :meth:`clear_fit`."""
        if self.xmin is None:
            return
        for key, lo_box, hi_box in self._limit_axes():
            self._edit_param(key, None)
            with _signals_blocked(lo_box, hi_box):
                lo_box.setText(""); hi_box.setText("")
        self.refresh_limit_hints()
        self.status.setText("view range cleared (auto)")

    def _sync_relim_combo(self, value: str) -> None:
        """Reflect a PROGRAMMATIC relim change (the colour-range Apply/Auto) in the Display relim combo
        WITHOUT re-firing its handler, so the chooser and the colour-range row always agree on whether
        the clim is pinned.  No-op when the combo is absent (a non-image Edit that has no relim row)."""
        combo = self.ed_params.get("relim") if getattr(self, "ed_params", None) else None
        if combo is None:
            return
        idx = combo.findText(value)
        if idx >= 0:
            with _signals_blocked(combo):
                combo.setCurrentIndex(idx)

    def apply_clim(self) -> None:
        """Pin a 2d/sites panel's COLOUR limit to the typed lo/hi.  Routes through the ONE clim source the
        relim family owns (relim="fixed" + the card's ``apply_fixed_lims``), so the live card (every cell,
        re-asserted each tick), this tab's snapshot, ``config.params`` and Save all move together -- there
        is no second hand-copied clim path.  Both boxes empty = Auto (release the pin)."""
        if self.clo is None:
            return
        if not self.clo.text().strip() and not self.chi.text().strip():
            self.clear_clim()
            return
        try:
            lo, hi = float(self.clo.text()), float(self.chi.text())
        except ValueError as exc:
            self.status.setText(f"bad colour range: {str(exc).splitlines()[0][:80]}")
            return
        self._edit_param("relim", "fixed")          # make the fixed clim take effect (seeds from view) ...
        if self.card is not None:
            self.card.apply_fixed_lims(lo, hi)      # ... then overwrite with the typed lo/hi (final state)
        snap = getattr(self, "_plotter", None)      # push onto THIS tab's snapshot too (as _edit_fixed_lim)
        if snap is not None:
            snap.apply_param("fixed_lo", lo); snap.apply_param("fixed_hi", hi)
            if getattr(self, "_canvas", None) is not None:
                self._canvas.draw_idle()
        self._sync_relim_combo("fixed")
        # _edit_param("relim","fixed") re-seeded the boxes from the FROZEN current view; re-seed now that
        # apply_fixed_lims has written the TYPED lo/hi so the boxes show what was actually pinned.
        self._seed_clim_boxes()
        self.status.setText("colour range applied")

    def clear_clim(self) -> None:
        """Release the colour-limit pin back to the autoscaled clim (relim="normal") -- the image
        counterpart of :meth:`clear_limits`.  Empties the boxes so the grey placeholder hint takes over
        showing the now-auto clim."""
        if self.clo is None:
            return
        self._edit_param("relim", "normal")
        with _signals_blocked(self.clo, self.chi):
            self.clo.setText(""); self.chi.setText("")
        self._sync_relim_combo("normal")
        self.refresh_limit_hints()
        self.status.setText("colour range cleared (auto)")

    def _seed_clim_boxes(self) -> None:
        """Put the STORED clim pin (``fixed_lo/hi``, in force only while relim=="fixed") into the
        colour-range boxes.  Like :meth:`_seed_limit_boxes` for x: the boxes edit the PIN, so empty = Auto
        and a value = pinned; re-seeded on build / tab-show / relim change, NEVER on the tick."""
        if getattr(self, "clo", None) is None:
            return
        p = self.card.config.params if self.card is not None else {}
        lo = hi = ""
        if str(p.get("relim")) == "fixed":
            try:
                lo, hi = f"{float(p.get('fixed_lo')):.6g}", f"{float(p.get('fixed_hi')):.6g}"
            except (TypeError, ValueError):
                lo = hi = ""
        with _signals_blocked(self.clo, self.chi):
            self.clo.setText(lo); self.chi.setText(hi)

    def _save_stem(self, timestamp: str | None) -> Path:
        """The output file stem (no extension) the Save section resolves to.

        ``path`` (the picker) is the base.  With auto-name ON the file is
        ``<base-or-base/title>_<plot-kind>_<timestamp>`` (unique); a ``timestamp`` of None
        yields the literal ``<time>`` placeholder for the read-only preview.  With auto-name
        OFF the base is used VERBATIM (its extension stripped), so the operator sets the exact
        name.  ONE resolver, shared by :meth:`save` and :meth:`_update_save_preview`."""
        title = (self.card.config.title or self.card.config.kind).strip() or "panel"
        kind = self.card.config.kind
        text = self.save_dir_edit.text().strip() if hasattr(self, "save_dir_edit") else ""
        base = Path(text) if text else Path(self.console._last_save_dir or _task_files_dir())
        # a bare folder (blank default / trailing sep / an existing dir) -> the file is
        # <folder>/<title>; otherwise the path already names the file stem.
        if not text or str(base).endswith(("/", "\\")) or (base.is_dir() and not base.suffix):
            base = base / title
        base = base.with_suffix("")               # we always append our own .png / .npz
        if hasattr(self, "save_autoname") and self.save_autoname.isChecked():
            return base.parent / f"{base.name}_{kind}_{timestamp or '<time>'}"   # unique
        return base                                # verbatim (operator-set name; overwrites)

    def _save_image_ext(self) -> str:
        """The image container the next Save writes (``png`` / ``pdf`` / ``jpg``).

        ONE reader of the format picker, shared by :meth:`save` and :meth:`_update_save_preview`, so the
        previewed suffix and the file actually written never disagree.  Falls back to the first
        :data:`SAVE_IMAGE_FORMATS` entry when the picker is absent (e.g. a non-plot editor) or empty."""
        combo = getattr(self, "save_format_combo", None)
        ext = combo.currentText().strip().lower() if combo is not None else ""
        return ext or SAVE_IMAGE_FORMATS[0]

    def _update_save_preview(self) -> None:
        """Show the actual file (full path) the next Save writes -- not just the folder."""
        if not hasattr(self, "save_preview"):
            return
        self.save_preview.setText(
            f"{display_path(str(self._save_stem(None)))}.{self._save_image_ext()} + .npz")

    def _save_view_state(self) -> dict[str, object]:
        """The panel's DISPLAY knobs, DERIVED from the ONE source (``config.params`` keyed by the ONE
        ``PANEL_PARAMS`` catalog), folded into saved ``info['view']`` so ``load_figure(...).plot()`` and
        the figure viewer reopen the figure AS SEEN.  Captures BOTH:

        * the cross-kind view knobs that live OUTSIDE ``PANEL_PARAMS`` (relim mode + fixed lo/hi from
          ``card._relim`` / the fixed bounds, unit index, repeat mode);
        * EVERY kind-specific knob the panel actually renders from -- looped from
          ``PANEL_PARAMS[_param_kind()]`` (a hist(-grid)'s bins/fit/ylog, a 2d/sites(-grid)'s colormap,
          a monitor's length/show_dist), the SAME declarative catalog the render path (``_build_plot`` /
          ``_build_facet_plotter``) and the Setting/Edit UI read.  Each stores the panel's EFFECTIVE
          value (operator's pick else the declared default), so the npz is self-describing and reopens
          exactly as drawn.

        This kills the old divergence where the save HARD-LISTED only 6 keys and silently dropped every
        other rendered knob (bins/fit/ylog/length/show_dist) -- a knob added to ``PANEL_PARAMS`` is now
        saved automatically, so *display == reopened figure* by construction.  ``_param_kind()`` yields a
        grid's per-cell ``sub_plot_kind``, so a grid captures its cells' knobs through the very same loop
        (the same thing ``_grid_recipe_with_params`` already does).  ``cmap`` is one of these declared
        keys: ``params.get('cmap', decl.default)`` is exactly the operator-pick-else-kind-default the
        ``_resolved_cmap`` resolver gives, so an image/site-map save still records the real colormap name
        it drew with; a kind with no colormap param simply omits the key (never a stray ``''``)."""
        params = self.card.config.params
        view: dict[str, object] = {
            "relim": self.card._relim(),
            "fixed_lo": _safe_float(str(params.get("fixed_lo", 0.0)), 0.0),
            "fixed_hi": _safe_float(str(params.get("fixed_hi", 1.0)), 1.0),
            "unit_index": int(params.get("unit_index", 0) or 0),
            "repeat_mode": self.card._repeat_mode_value(),
        }
        # the view-window pins (#3) are cross-kind view knobs too: persist them so a reopened figure keeps
        # the operator's window (only when actually set -- a never-pinned panel omits them, staying
        # autoscale; view_ylim only ever exists on the image family, whose store gate wrote it).
        for key in ("view_xlim", "view_ylim"):
            pin = params.get(key)
            if pin:
                view[key] = [float(pin[0]), float(pin[1])]
        # The serialized request is the one cross-surface fit state.  It contains
        # model + Selection + typed options and is safe to round-trip as data.
        if params.get("fit_request"):
            view["fit_request"] = dict(params["fit_request"])
        for decl in _panel_display_decls(self.card.config.kind, self.card._param_kind()):
            view[decl.key] = params.get(decl.key, decl.default)
        return view

    def save(self) -> None:
        if self._plotter is None:
            return
        try:
            df = self._df_for()
            # BIND the figure to its data SOURCE, then let ``DataFigure.save`` capture ``info['signals']``
            # + ``info['provenance']`` through the ONE frontend-neutral core (the SAME path a notebook
            # ``p.save()`` runs) -- the console only supplies its resolvers (a stopped node's lingering
            # signal still resolves via ``_node_for_signal``, incl. the ``_last_node`` fallback).  The
            # GUI-only display blocks (source/kind/size/view) stay ``extra_info`` (they are console state,
            # not signal/provenance).
            inputs = list(self.card.config.inputs or [])
            value_node = self.console._node_for_signal(inputs[0]) if inputs else None
            df.bind_source(self.console.hub, value_node, inputs=inputs,
                           resolve_node=self.console._node_for_signal, session=self.console.session)
            stem = self._save_stem(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            stem.parent.mkdir(parents=True, exist_ok=True)
            # the operator-chosen container (png / pdf / jpg) drives BOTH the file suffix and
            # ``DataFigure.save(image_ext=...)`` -- the ONE format reader keeps them in lockstep.
            ext = self._save_image_ext()
            # a path WITH a suffix makes DataFigure.save write it VERBATIM (no extra timestamp),
            # so this resolver is the single source of the output name.
            out = df.save(stem.with_suffix(f".{ext}"), image_ext=ext,
                          extra_info={"source": self.card.config.source,
                                      "kind": self.card.config.kind,
                                      "size": self.card.config.size,
                                      "view": self._save_view_state()})
            self.console._last_save_dir = str(stem.parent)   # remember where (kernel session)
            self._update_save_preview()
            # Show the LEAF folder, not the full absolute path (the full path is one click away in the
            # tooltip + the save-preview field) -- a short message that fits, with the width-safe status
            # policy as the backstop so even a long folder name can never widen the page.
            self.status.setText(f"saved {out['figure'].name} + {out['data'].name} → …/{stem.parent.name}")
            self.status.setToolTip(str(stem.parent))
        except Exception as exc:
            self.status.setText(f"save failed: {str(exc).splitlines()[0][:120]}")




# ====================================================================== logic tab
#: Now :class:`zlc_frontend.qt_widgets.LogicNodeRow`; old name kept as a plain alias.
LogicNodeRow = _LogicNodeRow


class LogicNodeEditor(QtWidgets.QWidget):
    """One logic node's Edit tab (closable): its auto-generated PARAM FORM + Start /
    Stop + a status line.  NO curve fit -- fitting a curve is a plotter concern (add
    a Plot panel on the Monitor board pointed at the signals this node publishes).

    The param form reuses :class:`MeasurementPanel` (single-spec): a camera /
    measurement / processor / task all expose ``.name`` + ``.params`` (ParamDecls),
    so the same form engine + Start / Stop signals drive every logic kind.  The
    camera live Measurement's spec is ``readout.camera_spec()`` (its ParamDecls are
    the camera's exposure / frames-per-cycle)."""

    def __init__(self, row: "LogicNodeRow", console: "TaskConsole", spec, parent=None):
        super().__init__(parent)
        self.row = row
        self.console = console
        self.spec = spec
        self.setStyleSheet("background: transparent;")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        scroll.setWidget(page)
        col = QtWidgets.QVBoxLayout(page)
        m = scaled_px(10, minimum=6)
        col.setContentsMargins(m, m, m, m)
        col.setSpacing(scaled_px(6, minimum=4))

        col.addWidget(FluentSectionLabel(row.node.title))
        # The auto-generated parameter form + Start / Stop (reused MeasurementPanel,
        # which already carries start_requested(self) / stop_requested + the typed,
        # no-eval form).  A spec drives a real ParamDecl form; the camera (spec is
        # None) shows nothing here but Start/Stop still build/run the camera node.
        # ``repeat`` (0 = ∞) is the ONE MEASUREMENT-layer acquisition knob -- the plot can NEVER tell a
        # measurement how many times to run (#H3l).  It is a DECLARED ParamDecl auto-injected into the
        # SAME auto-form as every other param (never a hand-placed widget; 0 is the ∞ sentinel, the same
        # semantics as the scan-repeat count -- no separate Free-run toggle).  An acquisition node
        # (measurement / camera) gets it; a processor / task does not.  How the repeats are DISPLAYED is
        # the PLOT's "repeat mode" Setting.  A camera defaults to ∞ (repeat=0, a live monitor); a scan
        # defaults to a single finite sweep (repeat=1).
        acquisition = (_acquisition_param_decls(repeat_default=(0 if row.node.kind == "camera" else 1))
                       if row.node.kind in ("measurement", "camera") else ())
        names_provider = getattr(console, "_signal_names", None)
        if row.node.kind == "processor" and callable(names_provider):
            # A reactive processor's source picker must not offer the node's OWN outputs -- picking
            # one is the self-feedback loop Processor.__init__ rejects loud at Start; hide it here
            # so the misclick cannot happen (declared keys, #prebind, so it holds before the first
            # run too).  This filter applies only to PROCESSOR rows; acquisition measurements use
            # their own explicitly bounded input contract (see ``_reactive_ring``).
            def names_provider(_base=names_provider, _console=console, _row=row):
                own = {str(k) for k in _console._declared_signal_keys(_row)}
                return [n for n in _base() if str(n) not in own]
        self.form = MeasurementPanel([spec] if spec is not None else [], single=True,
                                     signals_provider=names_provider,
                                     sources_provider=getattr(console, "_signal_providers", None),
                                     formats_provider=getattr(console, "_signal_formats", None),
                                     short_names_provider=getattr(console, "_signal_short_names", None),
                                     acquisition_params=acquisition)
        self.form.seed_values(row.node.values or {})
        self.form.start_requested.connect(lambda *_: self.console._start_logic_node(self.row))
        self.form.stop_requested.connect(lambda: self.console._stop_logic_node(self.row))
        col.addWidget(self.form)
        # (The node's published-signals + shapes are shown on its Logic-tab ROW card,
        # the single place for that legend -- not duplicated here.)
        col.addStretch(1)

    def collect_values(self) -> dict:
        return self.form.collect_values()                # repeat (0 = ∞) comes back like any param

    def set_running(self, running: bool) -> None:
        self.form.set_running(running)

    def set_status(self, text: str, *, error: bool) -> None:
        self.form.set_status(text, error=error)

    def teardown(self) -> None:
        # No matplotlib resources here (a logic node never plots), so teardown is a
        # no-op -- present so the console can treat it like a PanelEditor.
        pass

    def refresh_on_show(self) -> None:
        """When switching back to this Edit tab, refresh the form's dynamic combos so a
        signal that was not yet published when this tab was last open now shows up.
        Delegates to the form's own ``refresh_on_show`` (the one hook every form honours)."""
        hook = getattr(self.form, "refresh_on_show", None)
        if callable(hook):
            hook()


# ====================================================================== console
class _StopAttempt:
    """One console-owned invocation of an arbitrary node's cooperative stop method.

    The node may violate its timeout contract.  Running it on this private daemon thread keeps the
    Qt owner deadline authoritative; an overdue attempt remains attached to the node and can only
    resolve ownership when this exact invocation actually returns and the node reports not running.
    """

    def __init__(self, node, timeout: float):
        self.node = node
        self.timeout = max(0.0, float(timeout))
        self.result = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"TaskConsoleStop-{type(node).__name__}-{id(node):x}",
            daemon=True,
        )

    def _run(self) -> None:
        try:
            self.result = self.node.stop(timeout=self.timeout)
        except BaseException as exc:
            self.error = exc

    def start(self) -> None:
        self.thread.start()


class TaskConsole(QtWidgets.QWidget):
    """The dashboard window body: header bar + drag-and-snap panel board."""

    def __init__(
        self,
        *,
        hub,
        state: TaskConsoleState | None = None,
        running_nodes: Sequence[object] = (),
        measurements: Sequence[object] = (),
        processors: Sequence[object] = (),
        tasks: Sequence[object] = (),
        session: object | None = None,
        runtime_fence: object | None = None,
        runtime_fence_provider=None,
        scale: float | None = None,
        window_ratio: float = WINDOW_SCREEN_FRACTION,
        window_px: tuple[int, int] | None = None,
        embedded: bool = False,
    ):
        ensure_qt_app()
        set_fluent_scale(scale)
        super().__init__()
        self.hub = hub
        # EMBEDDED mode (the figure viewer hosts a whole TaskConsole in one pane): a stripped,
        # RESIZABLE console -- no Logic tab, no whole-board Pause/Save image/Save/Load buttons (a
        # loaded static figure has nothing to acquire, freeze, or persist as a layout) -- and, crucially,
        # NOT size-frozen: the parent layout stretches it so the gravity board reads the REAL viewport
        # width and reflows into 2+ columns.  Standalone (embedded=False) behaviour is byte-for-byte
        # unchanged.  The four omitted buttons are set to None so every reference guards on existence.
        self.embedded = bool(embedded)
        # Nodes that are CURRENTLY running (their owner thread is publishing to
        # the hub).  Populated on Start (``_start_logic_node``) and drained on Stop;
        # an externally-supplied, already-running node may be adopted here too.
        # This is DISTINCT from ``self.logic_nodes`` (the Logic-tab ROWS): a row is a
        # declaration that exists whether or not its node is running.
        adopted_nodes = list(running_nodes)
        self._require_bounded_stop_nodes(adopted_nodes)
        if runtime_fence is None and any(
            tuple(getattr(node, "referenced_devices", lambda: ())())
            or tuple(getattr(node, "occupied_devices", lambda: ())())
            or tuple(getattr(node, "lifecycle_devices", lambda: ())())
            for node in adopted_nodes
        ):
            raise RuntimeError(
                "device-bearing running_nodes require a runtime authority"
            )
        self.running_nodes = [] if runtime_fence is not None else adopted_nodes
        self._stop_attempts: dict[int, _StopAttempt] = {}
        # Production session launchers inject zlc_workbench.LegacyRuntimeFence.  The
        # frontend keeps it duck-typed so the target package DAG remains one-way:
        # workbench composes frontend, frontend never imports workbench.
        if runtime_fence_provider is not None and not callable(runtime_fence_provider):
            raise TypeError("runtime_fence_provider must be callable or None")
        self._legacy_runtime_fence = runtime_fence
        self._runtime_fence_provider = runtime_fence_provider
        self._legacy_handles: dict[int, object] = {}
        self._legacy_handle_fences: dict[int, object] = {}
        self._starting_nodes: dict[int, object] = {}
        self._pending_fenced_starts: dict[int, set[str]] = {}
        self._panel_teardown_phases: dict[int, set[str]] = {}
        # The declarative measurement catalog: each becomes an addable LOGIC NODE
        # (a swept measurement) on the Logic tab; with none, only the camera /
        # processors / tasks are offered (and only plot kinds without a session).
        self.measurements = list(measurements)
        # The declarative DATA-PROCESSING catalog (reactive transform nodes) and the
        # TASK catalog (one-shot orchestrations): each is an addable LOGIC NODE.
        self.processors = list(processors)
        self.tasks = list(tasks)
        # The connected experiment session (optional): with it, the camera live
        # Measurement is offered too (readout.camera_measurement).
        self.session = session
        # The camera row's display name, read ONCE from readout.camera_spec().name when the
        # Add-Panel dropdown is populated (the spec owns it; "" until/without a session camera).
        self._camera_title = ""
        self.state = state or default_console_state()
        self.window_ratio = float(window_ratio)
        self._window_px = window_px
        self.cards: list[PanelCard] = []
        # The Logic-tab nodes.  Each entry maps a LogicNodeRow -> the live state for
        # that node: its built node (None until Started) + its Edit tab.
        self.logic_nodes: list[LogicNodeRow] = []
        self._logic_nodes: dict[int, object] = {}      # id(row) -> node (or None when stopped)
        # id(row) -> the LAST built node, kept even after Stop (unlike _logic_nodes, which
        # is None'd on stop) -- so a finished/stopped node's signals that LINGER in the hub
        # still show WHICH node produced them in the signal picker (a stopped scan's
        # `readout_fidelity`, a stopped camera's `frame`).  Cleared only on row removal.
        self._last_node: dict[int, object] = {}
        self._logic_editors: dict[int, "LogicNodeEditor"] = {}  # id(row) -> Edit tab
        # The panel<->analysis association is a PERSISTED SINGLE SOURCE, never a runtime dict: the
        # panel's config stores its region signal name (``params['region_signal']``, written once at
        # creation and never re-derived) + the drawn region payload (``params['region']``), and its
        # Analysis row is DERIVED as the row whose ``values['region']`` equals that name
        # (:meth:`_panel_analysis_row`).  Save/load therefore re-associates for free, ids never leak,
        # and two panels can never cross-link (a fresh name is deduped against every persisted name).
        # id(card) -> the published version of its fit node's params last pushed to the overlay, so an
        # unchanged fit re-draws nothing (the overlay is DISPLAY-only, driven by the node's publishes).
        self._fit_overlay_pushed: dict[int, int] = {}
        self._building = False
        self._address: str | None = None
        # The folder the LAST panel "Save Fig" wrote into -- so a panel's Edit-tab save
        # picker reopens at the same place across panels and across reopens (remembered
        # for the life of the process / Jupyter kernel, like the pulse GUI's save dir).
        # None until the first save -> the picker defaults to the tasks/ folder.
        self._last_save_dir: str | None = None
        # No-measurement / no-processor sentinels kept so older callers / tests that
        # probe "is there a global measurement launcher" still read None (there is
        # no global form -- a measurement is a Logic node now).
        self.measurement_panel = None
        self.measurement_group = None
        # Per-panel editors: one PanelEditor per opened PLOT panel, hosted as a
        # closable tab (keyed by id(card)).
        self._panel_editors: dict[int, "PanelEditor"] = {}
        # A running TASK takes over the console (confocal-style): its mid-run output
        # occupies a FIXED panel so the operator watches the work in progress, and all
        # other actions are LOCKED until it finishes / is stopped.  ``_running_task_row``
        # is the LogicNodeRow of the task currently running (None when idle);
        # ``_task_card`` is its dedicated mid-run Monitor panel.
        self._running_task_row: "LogicNodeRow | None" = None
        self._task_card: "PanelCard | None" = None
        self._task_mid_key = DEFAULT_MID_RUN_KEY   # which output-buffer key the task panel shows
        self._task_output_node = None          # running Task whose typed TaskOutput feeds the panel
        self._task_card_tensor = None          # immutable latest SignalTensor, including validity
        self._task_locked = False              # True while a task runs -> all other actions blocked
        # During whole-console shutdown a task panel cannot be destroyed while the render worker may
        # still own its Figure.  Node termination therefore detaches the task state and parks the card
        # here; only a confirmed RenderLoop join permits the UI-teardown phase to remove it.
        self._defer_task_card_teardown = False
        self._deferred_task_card: "PanelCard | None" = None

        # Multi-rate refresh: the timer ticks at the BASE interval (the smallest panel
        # update_ms, which divides every other so the rates co-align); each panel redraws
        # every update_ms // base ticks.  _tick_count counts base ticks since the last re-base.
        self._tick_count = 0
        self._base_interval_ms = int(self.state.interval_ms)
        # Display reads each signal at its own latest value.  A producer that owns
        # related view signals publishes them atomically, so companion values remain
        # shot-coherent without console-side joins.

        # The ONE background render thread: every steady-tick compose (numpy prep + matplotlib
        # artist updates + Agg rasterisation) runs there; the GUI thread only schedules, presents
        # the finished front buffers, and serves interaction -- so a slow render lowers the plot
        # frame rate, never the UI's responsiveness (see frontend/render_loop.py for the ownership
        # protocol).  Structural builds (Qt widgets) come back to the GUI via _on_render_batch.
        # Created BEFORE the UI build: load_state constructs panel cards, which take the barrier.
        # L596: an un-migrated shared-Figure RenderLoop is only allowed to live INSIDE the
        # SerializedLegacyAggBridge island, so the shell never holds the raw loop.  The bridge is
        # fail-closed: a handoff that does not settle raises instead of letting this thread touch a
        # Figure the worker may still own.  Every access below goes through _render_barrier().
        from zlc_workbench.legacy import SerializedLegacyAggBridge

        RenderLoop = _qt_widgets.render_loop.RenderLoop
        self._render = SerializedLegacyAggBridge(RenderLoop(self._on_render_batch, parent=self))

        self._build_ui()
        self.load_state(self.state)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._recompute_tick_interval()          # base = min panel update_ms (sets the interval)
        self._timer.start()
        self._paused = False                     # Pause button: freeze EVERY plot's display at once

    # ------------------------------------------------------------------ UI
    def _target_console_size(self) -> QtCore.QSize:
        """Window size from the primary screen (the shared GUI sizing rule)."""

        if self._window_px is not None:
            return QtCore.QSize(int(self._window_px[0]), int(self._window_px[1]))
        return screen_fit_window_size(self.window_ratio)

    def _build_ui(self) -> None:
        self.setWindowTitle("TaskConsole@Zou lab")
        self.setStyleSheet(fluent_widget_stylesheet())
        # Standalone: a fixed window (the shared GUI sizing rule).  EMBEDDED: the parent (the figure
        # viewer's right pane) owns the size, so we set only a MINIMUM and expand into whatever the
        # layout gives -- that lets the gravity board read the live viewport width and reflow into
        # multiple columns instead of being pinned to a frozen width that stacks every card in one column.
        if self.embedded:
            self.setMinimumSize(self._target_console_size())
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        else:
            self.setFixedSize(self._target_console_size())
        root = QtWidgets.QVBoxLayout(self)
        # SINGLE SOURCE for every window-edge inset: ``window_pad(1)`` (== ``scaled_px(WINDOW_PAD)``).
        # The window title (qt_widgets draws it at ``scaled_px(TITLE_LEFT_INSET)`` == ``scaled_px(WINDOW_PAD)``)
        # pins to this SAME left column, so the body's left edge lines up under the "task_console@zoulab"
        # title text -- one shared left edge.
        margin = window_pad(1)
        # EMBEDDED: no top/bottom inset -- the host (the figure viewer) already frames the console with
        # its own root margin, so the console's FIRST card (the header) sits flush at the pane top and its
        # visible top/bottom edges line up with the Info card beside it.  STANDALONE: the top AND bottom
        # insets are the SAME ``window_pad(1)`` as left/right, so the padding to all four window edges is
        # identical (the bottom gap now matches the sides).
        v_margin = 0 if self.embedded else window_pad(1)
        # The tab card carries a flat 1 px border (no drop shadow), so no bottom-bleed headroom is
        # reserved -- top and bottom insets are both the plain ``v_margin`` (0 embedded, so the header
        # lines up flush with the Info card beside it).
        root.setContentsMargins(margin, v_margin, margin, v_margin)
        # A clear GAP separates the three rows -- the header card, the (hidden) task banner and
        # the tab card -- so they read as DISTINCT rounded cards on the grey window background.
        # The gap is HALF the window pad (``window_pad(0.5)``), so every spacing on screen is a clean
        # multiple of the ONE ``WINDOW_PAD`` unit.  The header is flat (no drop shadow) and the tab bar
        # draws no top base line, so this gap reads as clean card separation rather than a hard line.
        root.setSpacing(window_pad(0.5))

        # FLAT header (no drop shadow): the shadow's soft bottom edge cast a thin grey line
        # into the gap right above the tab strip (the "line above the tabs").  The tab widget
        # below carries its own card, so the header needs no elevation -- it is a plain top bar.
        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))
        header.setSpacing(scaled_px(7, minimum=4))

        self.status_dot = FluentStatusDot(size=16)
        self.status_dot.set_color(GREEN)
        self.name_edit = FluentLineEdit(self.state.name)
        self.name_edit.setPlaceholderText("task name")
        self.name_edit.setFixedWidth(scaled_px(150, minimum=110))
        self.name_edit.textChanged.connect(self._mark_dirty)
        # Persistent telemetry belongs to the original header: it is orthogonal to the event/status
        # channel below and therefore never disappears behind a task message or node error.
        self.summary = FluentLabel("")
        self.summary.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.kind_combo = FluentComboBox()
        # Add Panel offers EXACTLY the four kinds the user designs with -- nothing
        # invented, no composite:
        #   * "Plot: X"        -- a pure VIEW; shows nothing until you wire a hub
        #                         signal in its Setting (it NEVER starts anything);
        #   * "Measurement: X" -- an acquisition logic node (the camera live stream,
        #                         or a swept measurement) added to the LOGIC tab;
        #   * "Processor: Y"   -- a reactive transform logic node (the "func" layer);
        #   * "Task: Z"        -- a one-shot orchestration logic node (e.g. calibrate).
        # A logic node (measurement/processor/task) is added STOPPED to the Logic tab;
        # you Start/Stop it from its own Edit.  A plot is added to the Monitor board.
        readout = getattr(self.session, "readout", None)
        # The dropdown offers only the ADDABLE plot kinds (``panel=True``) -- the ones you add a
        # BLANK panel of and wire live.  ``pulse`` is a real panel kind too, but it is not added
        # blank live (it comes from a saved recipe / a fired sequence via the seed path), so it is
        # not listed here.
        for key, label in ADDABLE_PANEL_KINDS.items():
            self.kind_combo.addItem(f"Plot: {label}", key)
        # MEASUREMENT layer: the continuous camera (a live frame stream) first, then
        # the swept measurements from the catalog.
        if readout is not None and hasattr(readout, "camera_measurement") and hasattr(readout, "camera_spec"):
            # The camera row's DISPLAY name is the authoritative spec's own name
            # (readout.camera_spec().name) -- never a re-typed literal, so a spec rename
            # reaches the dropdown AND the row title (_add_panel) automatically.  Resolved
            # ONCE here (camera_spec() re-enumerates devices; keep it out of hot paths).
            self._camera_title = str(readout.camera_spec().name)
            self.kind_combo.addItem(f"Measurement: {self._camera_title}", ("camera", "live"))
        for spec in self.measurements:
            self.kind_combo.addItem(f"Measurement: {spec.name}", ("measurement", spec.name))
        # PROCESSOR layer (the "func" nodes).
        for spec in self.processors:
            self.kind_combo.addItem(f"Processor: {spec.name}", ("processor", spec.name))
        # TASK layer: one-shot orchestrations from the auto-discovered task catalog.
        for spec in self.tasks:
            self.kind_combo.addItem(f"Task: {spec.name}", ("task", spec.name))
        self.kind_combo.setFixedWidth(scaled_px(170, minimum=130))
        add_button = FluentButton("Add Panel", color=ACCENT)
        add_button.clicked.connect(self._add_panel)
        # "Selectors" switch: the Monitor board is display-only BY DEFAULT (every panel builds
        # its selector layer but parks it inactive; the wheel scrolls the board) -- the original
        # anti-misclick rule.  Flip ON to arm the full selector layer (zoom/pan, area, cross,
        # draggable threshold/clim lines) on every dashboard panel IN PLACE -- no rebuild, and a
        # panel added while ON inherits it (see _attach_card).  A DISPLAY control like Pause, so
        # it stays OUT of the running-task lockout.
        self.selectors_switch = FluentSwitch("Selectors")
        self.selectors_switch.setChecked(False)
        self.selectors_switch.setToolTip(
            "OFF: panels are display-only (wheel scrolls the board).\n"
            "ON: zoom / area / cross / draggable lines work on every panel\n"
            "(the wheel zooms inside a plot, like the Edit tab).")
        self.selectors_switch.toggled.connect(self._toggle_selectors)
        # Whole-console layout + monitor controls: Save/Load persist the LAYOUT (json); Pause/Resume
        # freezes every plot at once (a display freeze -- acquisition keeps running); Save image grabs
        # the WHOLE board region to a PNG.  EMBEDDED (figure-viewer host) has nothing to acquire, freeze
        # or persist as a standalone layout, so all four are OMITTED -- set to None so every reference
        # (``_lockable_header``, ``_mark_dirty``, ``_toggle_pause`` ...) guards on existence.
        if self.embedded:
            self.save_button = None
            load_button = None
            self.pause_button = None
            self.save_image_button = None
            self.devices_button = None
        else:
            self.save_button = FluentButton("Save", color=ACCENT)
            self.save_button.clicked.connect(self.save_to_file)
            load_button = FluentButton("Load", color=ORANGE)
            load_button.clicked.connect(self.load_from_file)
            self.pause_button = FluentButton("Pause", color=ORANGE)
            self.pause_button.clicked.connect(self._toggle_pause)
            self.save_image_button = FluentButton("Save image", color=ACCENT)
            self.save_image_button.clicked.connect(self._save_board_image)
            # The READ-ONLY device viewer: one tab per loaded device (its snapshot + live runtime
            # read-backs), NO config editor / add / remove -- the console must not let an operator
            # mutate the running device set (the full editor is the notebook's exp.device_manager()).
            # A launcher (like Selectors/Pause), OUT of the running-task lockout: viewing is read-only.
            self.devices_button = FluentButton("Devices", color=GREY)
            self.devices_button.setToolTip("Open the read-only device viewer: each loaded device's "
                                           "snapshot + live state (no editing -- use the notebook's "
                                           "device manager to change devices).")
            self.devices_button.clicked.connect(self._open_device_viewer)

        for widget in (self.status_dot, self.name_edit):
            header.addWidget(widget)
        header.addWidget(self.summary, 1)
        # Add Panel stays even when embedded (a loaded figure is re-wired + extra panels added), and so
        # does the Selectors switch (inspecting a loaded figure is exactly when the selectors help);
        # the four whole-console buttons only when they exist.
        for widget in (self.kind_combo, add_button, self.selectors_switch, self.devices_button,
                       self.pause_button, self.save_image_button, self.save_button, load_button):
            if widget is not None:
                header.addWidget(widget)
        # header controls disabled while a task runs (the lockout, #5); kept as a group
        # so ``_set_task_running`` flips them all at once.  Pause stays OUT of the lockout (you may
        # want to freeze the display even mid-task); Save image is locked like the other mutators.
        # None entries (embedded: the button was never made) are dropped so the lockout loop is safe.
        self._lockable_header = tuple(w for w in (self.kind_combo, add_button, self.save_image_button,
                                                  self.save_button, load_button, self.name_edit)
                                      if w is not None)
        root.addWidget(header_frame)

        # PERSISTENT status strip -- the ONE always-visible line between the header and the
        # tabs.  Its content switches by PRIORITY (node error > running task's progress >
        # display-behind advisory; idle is empty, see _update_summary); the Stop-task
        # action shows only while a task owns the console.  This replaces BOTH old mechanisms:
        # the transient orange task banner (its show/hide shifted the whole layout under the
        # pointer).  Header telemetry remains visible independently above it.
        self.status_strip = FluentStatusStrip(action_text="Stop task")
        self.status_strip.action_clicked.connect(self._stop_running_task)
        root.addWidget(self.status_strip)

        # TWO permanent tabs:
        #   * Monitor -- the live drag-and-snap board of PLOT panels (pure views);
        #   * Logic   -- the list of LOGIC NODES (measurement / processor / task),
        #               each a row with a status dot + Edit.
        # A plot panel's Edit and a logic node's Edit each open their OWN closable
        # tab beside these two.
        self.tabs = FluentTabWidget()
        dash_tab = QtWidgets.QWidget()
        dash_tab.setStyleSheet("background: transparent;")
        dash_layout = QtWidgets.QVBoxLayout(dash_tab)
        dash_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(True)
        self.board = _PanelBoard()
        self.scroll.setWidget(self.board)
        dash_layout.addWidget(self.scroll, 1)
        # Re-pack the gravity board when the viewport WIDTH changes (window resized): cards wrap at
        # the live viewport width, so narrowing the window must reflow them into fewer columns (else
        # they would just clip behind a horizontal scrollbar).  Watch the viewport's resize via an
        # event filter and re-arrange only on an actual width change (no-op recursion guard).
        self._board_view_w = 0
        self.scroll.viewport().installEventFilter(self)
        self.tabs.add_permanent_tab(dash_tab, "Monitor")
        # A one-shot re-arrange AFTER the first show, when the scroll viewport finally has its real
        # width (0 during construction): so an EMBEDDED console (hosted in the figure viewer) packs
        # its board against the true pane width immediately, not a stale 0.
        QtCore.QTimer.singleShot(0, self._arrange_if_cards)

        # Logic tab: a scrolled top-packed column of LogicNodeRow cards.  OMITTED when embedded (the
        # figure viewer hosts pure VIEWS of a loaded static figure -- no acquisition logic to run), so
        # ``logic_scroll`` / ``logic_layout`` / ``logic_hint`` stay None and their few users guard.
        if self.embedded:
            self.logic_scroll = None
            self.logic_layout = None
            self.logic_hint = None
        else:
            logic_tab = QtWidgets.QWidget()
            logic_tab.setStyleSheet("background: transparent;")
            logic_outer = QtWidgets.QVBoxLayout(logic_tab)
            logic_outer.setContentsMargins(0, 0, 0, 0)
            self.logic_scroll = FluentScrollArea()
            self.logic_scroll.setWidgetResizable(True)
            self.logic_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)   # #2: never scroll sideways
            self.logic_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            logic_body = QtWidgets.QWidget()
            logic_body.setStyleSheet("background: transparent;")
            self.logic_layout = QtWidgets.QVBoxLayout(logic_body)
            lm = scaled_px(10, minimum=6)
            self.logic_layout.setContentsMargins(lm, lm, lm, lm)
            self.logic_layout.setSpacing(scaled_px(8, minimum=5))
            self.logic_hint = FluentLabel(
                "No logic nodes yet.  Add a Measurement / Processor / Task from the header "
                "(it starts STOPPED); open its Edit to set parameters and Start it.")
            self.logic_hint.setWordWrap(True)
            self.logic_hint.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self.logic_layout.addWidget(self.logic_hint)
            self.logic_layout.addStretch(1)
            self.logic_scroll.setWidget(logic_body)
            logic_outer.addWidget(self.logic_scroll, 1)
            self.tabs.add_permanent_tab(logic_tab, "Logic")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.tab_close_requested.connect(self._on_editor_tab_closed)
        # The tab card (Monitor / Logic / Edit) carries a flat 1 px border (no drop shadow), so the row
        # spacing above it is the whole gap -- no extra top-bleed headroom is topped up.
        root.addWidget(self.tabs, 1)

    # ------------------------------------------------------------------ state
    def read_state(self) -> TaskConsoleState:
        # Flush every OPEN logic-node Edit form into its config first, so Save captures
        # the CURRENT parameter values even when the node was never Started (#4: a
        # layout persists Edit params, not just geometry).  Plot params already write
        # through to ``card.config.params`` live, so panels need no flush.
        for row in self.logic_nodes:
            editor = self._logic_editors.get(id(row))
            if editor is None:
                continue
            try:
                row.node.values = editor.collect_values()
            except Exception:
                pass
        return TaskConsoleState(
            name=self.name_edit.text().strip() or "task",
            interval_ms=self.state.interval_ms,
            panels=[card.config for card in self.cards],
            logic=[row.node for row in self.logic_nodes],
        )

    def load_state(
        self,
        state: TaskConsoleState,
        *,
        timeout: float = 5.0,
        _replacement_running_nodes: Sequence[object] | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("state-load timeout must be a non-negative number")
        if not isinstance(state, TaskConsoleState):
            raise TypeError("state must be TaskConsoleState")
        # Loading consumes no caller-owned objects.  Teardown intentionally mutates
        # the live PanelConfig/LogicNodeConfig graph, so freeze a private desired
        # graph before the first stop/remove and install only that graph.
        desired_state = TaskConsoleState.from_dict(state.to_dict())
        replacement_nodes = None
        if _replacement_running_nodes is not None:
            replacement_nodes = list(_replacement_running_nodes)
            self._require_bounded_stop_nodes(replacement_nodes)
        deadline = time.monotonic() + float(timeout)
        self._building = True
        try:
            for row in list(self.logic_nodes):
                if not self._stop_logic_node(
                    row,
                    _silent=True,
                    timeout=max(0.0, deadline - time.monotonic()),
                ):
                    raise RuntimeError(
                        "cannot load a new console state while a logic-node owner is still active"
                    )
            for row in list(self.logic_nodes):
                if not self._remove_logic_node(row, _rebuild=False):
                    raise RuntimeError("stopped logic node could not be removed")
            for card in list(self.cards):
                if not self._remove_panel(
                    card,
                    timeout=max(0.0, deadline - time.monotonic()),
                ):
                    raise RuntimeError(
                        "cannot load a new console state while a render worker still owns a panel"
                    )
            if replacement_nodes is not None:
                self.running_nodes = replacement_nodes
            self.state = desired_state
            self.name_edit.setText(desired_state.name)
            for config in desired_state.panels:
                self._attach_card(self._new_panel_card(config))
            for node in desired_state.logic:
                self._attach_logic_node(node)    # always STOPPED -- Start is manual
            # Replay every panel's persisted REGION as its control signal, so a loaded Analysis row
            # (still STOPPED) sees the saved selection the moment the user Starts it -- never a silent
            # whole-frame fallback -- and the derived panel<->row association is live immediately.
            from zlc_data.plot_region import Selection, region_bins, region_tensor
            for card in self.cards:
                name = str(card.config.params.get("region_signal") or "")
                payload = card.config.params.get("region")
                if not name or not isinstance(payload, Mapping):
                    continue
                selection = Selection.from_dict(payload)
                tensor = region_tensor(selection, bins=region_bins(selection))
                self.hub.register_signal(name, tensor.schema)   # idempotent: byte-stable schema
                self.hub.publish({name: tensor})
            self._fit_overlay_pushed.clear()   # loaded panels re-pull their overlays from scratch
            self._arrange()
        finally:
            self._building = False
        for card in self.cards:            # force every panel to redraw on its next beat
            card._render_version = -1
        self._recompute_tick_interval()    # the loaded panels' rates set the timer base
        self._update_summary()

    def reseed(self, state: TaskConsoleState, *, running_nodes: Sequence[object] = ()) -> None:
        """Reload the board to a NEW state IN PLACE: reuse this console's WHOLE widget tree (tabs, board,
        header, hub, refresh timer) and rebuild only its PANELS (:meth:`load_state`), swapping the
        producing-node set.  The single reseed entry for a host that shows one console and re-points it
        at different data -- the figure_viewer's Browse re-load: swapping the seeded panel must NOT tear
        the console down and construct a fresh one, which re-realizes the entire Qt widget tree (~0.35 s
        of the perceived load).  The caller owns STOPPING the previous nodes (the console only reads the
        hub); a reused hub's stale signals are overwritten by same-named republishes or GC'd as orphans."""
        self.load_state(state, _replacement_running_nodes=running_nodes)

    def _new_panel_card(self, config: PanelConfig) -> PanelCard:
        """Build a PanelCard wired to the console's signal providers -- the ONE place the
        provider block lives, so adding/renaming a provider is a single edit here instead of three
        parallel edits at every PanelCard construction site (load_state / _add_panel / a task run) (#A1)."""
        return PanelCard(
            config, parent=self.board,
            names_provider=self._signal_names, sources_provider=self._signal_providers,
            formats_provider=self._signal_formats, axes_provider=self._signal_axes,
            sites_inputs_provider=self._sites_inputs, curve_x_provider=self._curve_x,
            structure_provider=self._signal_structure, pulse_state_provider=self._pulse_state,
            grid_recipe_provider=self._grid_recipe,
            short_names_provider=self._signal_short_names, live_namespace_provider=self._expression_namespace,
            render_barrier=self._render_barrier, area_select_sink=self._on_panel_area_select,
            selection_clear_sink=self._on_panel_selection_clear, fit_node_sink=self._sync_fit_node)

    def _attach_card(self, card: PanelCard) -> None:
        card.setParent(self.board)
        card.show()
        card.changed.connect(self._mark_dirty)
        # Picking a signal / editing the source expression changes which signal the panel
        # reads -> refresh the frame-title legend NOW (it is self-guarded, so this is cheap and
        # a no-op when nothing changed), instead of lagging a tick behind the pick.
        card.changed.connect(self._refresh_signal_info)
        card.dropped.connect(self._snap_dropped_card)   # drag-release only: snap BEFORE the re-pack
        card.layout_changed.connect(self._arrange)
        card.update_interval_changed.connect(self._recompute_tick_interval)
        card.remove_requested.connect(self._remove_panel)
        card.edit_requested.connect(self._edit_card)
        # a panel added (or loaded) while the header's "Selectors" switch is ON inherits it --
        # the guard covers construction order (state panels may attach before the header exists).
        switch = getattr(self, "selectors_switch", None)
        if switch is not None:
            card.set_selectors_enabled(switch.isChecked())
        self.cards.append(card)
        self._recompute_tick_interval()        # a new panel's rate may change the timer base

    # ----------------------------------------------------------------- control
    # ---- producing-node discovery: a panel's data comes from a logic node; its Edit
    # exposes THAT node's acquisition parameters (e.g. a raw-frame panel is
    # produced by the camera measurement -> exposure / roi).
    @staticmethod
    def _referenced_signals(source: str) -> set:
        """The hub signal names a panel's source expression reads (AST Name nodes
        minus the namespace builtins) -- used to map the panel to its node.  The
        helper names come from the ONE signal_expr declaration (NAMESPACE_HELPERS),
        so a helper added there can never surface here as a phantom hub signal;
        ``value``/``signal`` are the expression-contract tokens (DEFAULT_SOURCE),
        not namespace helpers, and are excluded separately."""
        import ast

        from zlc_data.signal_expr import NAMESPACE_HELPERS
        try:
            tree = ast.parse(str(source or ""), mode="exec")
        except SyntaxError:
            return set()
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        return names - set(NAMESPACE_HELPERS) - {"value", "signal"}

    def _card_reads(self, card: "PanelCard") -> set:
        """The REAL hub signal names a panel reads: its picked input(s) (``config.inputs``)
        plus any hub signal its expression names directly.  The pseudo ``signal`` token in
        ``value = signal`` is NOT a hub name -- it stands for ``config.inputs[0]`` -- so it
        is excluded by :meth:`_referenced_signals` and the picked input is unioned in
        instead.  (Without this the signal-flow legend showed ``signal ← (no running
        source)`` for a panel that IS wired to a running node.)"""
        reads = set(self._referenced_signals(card.config.source))
        reads |= {str(n) for n in getattr(card.config, "inputs", ()) if n}
        return reads

    def _producing_node(self, card: "PanelCard"):
        """The running node whose published signals the panel's source reads (None if
        the expression touches no published signal, e.g. a pure constant)."""
        refs = self._card_reads(card)
        if not refs:
            return None
        for node in self.running_nodes:
            published = node.published_signals() if hasattr(node, "published_signals") else frozenset()
            if refs & set(published):
                return node
        return None

    def _producing_row(self, card: "PanelCard"):
        """The Logic-tab ROW whose RUNNING node produces this panel's signal (None if
        the producing node has no row -- e.g. a node passed via ``running_nodes=`` from
        a notebook).  Lets a plot's Edit show the SOURCE measurement/processor's full
        parameter form (#2)."""
        node = self._producing_node(card)
        if node is None:
            return None
        for row in self.logic_nodes:
            if self._logic_nodes.get(id(row)) is node:
                return row
        return None

    def _node_scan_range_key(self, node) -> str | None:
        """The first ``axis_range`` param key of ``node``'s spec -- i.e. the node's swept x-axis, or
        None when it does not scan a range.  The ONE data-driven test for "a selection on this node's
        1-D plot can set its scan range", so the plot-selection -> scan-range linkage is enforced for
        EVERY measurement that declares a scan range, never wired per measurement.

        ``node`` is a BUILT running node, so the spec is resolved via its Logic row's config
        (``_spec_for_logic`` reads a ``LogicNodeConfig``, not a built node)."""
        if node is None:
            return None
        for row in self.logic_nodes:
            if self._logic_nodes.get(id(row)) is node:
                spec = self._spec_for_logic(row.node)
                for param in getattr(spec, "params", ()) or ():
                    if getattr(param, "kind", "") == "axis_range":
                        return param.key
                return None
        return None

    def _form_for_node(self, node):
        """The producing node's OWN Logic-tab Edit FORM (the :class:`MeasurementPanel` that carries its
        scan range), OPENING its editor if not already open so a staged range always has somewhere to
        land.  The seam a 1-D plot selection uses to reach the measurement's scan-range param
        (node -> row -> editor.form), keeping the plot itself decoupled from the form's internals."""
        for row in self.logic_nodes:
            if self._logic_nodes.get(id(row)) is node:
                if self._logic_editors.get(id(row)) is None:
                    self._edit_logic_node(row)          # lazily create + show the node's Edit tab
                editor = self._logic_editors.get(id(row))
                return editor.form if editor is not None else None
        return None

    def _apply_source_params(self, row: "LogicNodeRow", values: dict) -> None:
        """Apply edited SOURCE parameters (from a plot Edit's source form, #2) to the
        producing node: live where the node accepts it (a camera's exposure/ROI), else
        rebuild + restart the node with the new values.  Keeps the node's own Logic-tab
        Edit form in sync so the two never diverge.  Returns False if a task lock
        dropped the edit (so the caller does not falsely report success)."""
        if self._task_locked:
            return False
        row.node.values = dict(values)
        editor = self._logic_editors.get(id(row))
        if editor is not None:
            editor.form.seed_values(values)               # keep the node's own Edit consistent
        node = self._logic_nodes.get(id(row))
        if node is None:
            return True                                    # not running -> values remembered for next Start
        try:
            live_keys = set(node.acquisition_parameters())
        except Exception:
            live_keys = set()
        if any(k not in live_keys for k in values) or not live_keys:
            self._start_logic_node(row)                   # rebuild + restart with the new values
        elif hasattr(node, "apply_acquisition_parameters"):
            node.apply_acquisition_parameters(**{k: v for k, v in values.items() if k in live_keys})
        return True

    @staticmethod
    def _node_params(node) -> list:
        """``[(name, current_value)]`` for the editable parameters of the data
        SOURCE behind ``node`` -- a camera's exposure/ROI, or the node's own
        analysis settings.  The node declares them via ``acquisition_parameters()``
        (the source decides what is tunable, not __init__ reflection)."""
        if node is None or not hasattr(node, "acquisition_parameters"):
            return []
        return list(node.acquisition_parameters().items())

    @staticmethod
    def _node_label(node) -> str:
        """Short LAYER name for a node (camera / detect / calibrate / a
        measurement's curve), NOT the Python class name -- the dashboard speaks in
        the architecture's layers, so no class name ever leaks into the UI."""
        label = getattr(node, "display_label", None)
        if label:
            return str(label)
        return str(getattr(node, "prefix", "") or type(node).__name__)

    def _provider_nodes(self) -> list:
        """Every node that can PROVIDE data or signals to a panel -- the RUNNING nodes plus the last
        build of each Logic-tab node (``_last_node``), kept past Stop.  The ONE source of "which nodes
        count": signal resolution (:meth:`_node_for_signal`), the picker (:meth:`_signal_providers`),
        AND the pulse/grid/sitemap/curve auxiliary-data providers all read this, so a stopped node's
        panel keeps rendering its lingering state instead of erroring "needs a producing node"."""
        return [*self.running_nodes, *self._last_node.values()]

    def _signal_providers(self) -> dict:
        """``name -> [node labels]`` for every signal a node produces, so the picker shows
        WHICH measurement / processor / camera each signal comes from.

        Covers RUNNING nodes AND the last build of every Logic-tab node (``_last_node``,
        kept past Stop): a finished scan's ``readout_fidelity`` or a stopped camera's
        ``frame`` LINGER in the hub, and the picker must still name their source node rather
        than show a bare signal.  Running nodes are listed first; a name carried by more
        than one node lists every source node (so an ambiguous pick can be flagged)."""
        providers: dict[str, list] = {}
        seen: set[int] = set()
        for node in self._provider_nodes():
            if node is None or id(node) in seen or not hasattr(node, "published_signals"):
                continue
            seen.add(id(node))
            label = self._node_label(node)
            for name in node.published_signals():
                bucket = providers.setdefault(str(name), [])
                if label not in bucket:
                    bucket.append(label)
        # DECLARED (not-yet-started) Logic-tab nodes: list the signals they WILL publish too,
        # tagged by their node title, so a Monitor can be wired to a node's output BEFORE that
        # node is started (#6: connect first, start later).  Skip rows already covered above.
        for row in self.logic_nodes:
            node = self._logic_nodes.get(id(row))
            if node is not None and getattr(node, "running", False):
                continue
            label = str(getattr(row.node, "title", "") or getattr(row.node, "name", "") or row.node.kind)
            for key in self._declared_signal_keys(row):
                bucket = providers.setdefault(str(key), [])
                if label not in bucket:
                    bucket.append(label)
        return providers

    def _is_control_signal(self, name: str) -> bool:
        """Whether ``name`` is a CONTROL-plane hub signal (a panel's drawn region) -- classified by
        the schema role the ONE region encoder stamps (``core.selection.CONTROL_ROLE``), the single
        choke point the picker pool filter AND the orphan-GC exemption both read."""
        from zlc_data.plot_region import CONTROL_ROLE
        try:
            schema = self.hub.schema(str(name))
        except KeyError:
            return False
        return dict(getattr(schema, "metadata", None) or {}).get("role") == CONTROL_ROLE

    def _signal_names(self) -> list[str]:
        """Every signal a picker may offer: PUBLISHED hub signals UNION the DECLARED outputs
        of stopped Logic-tab nodes (the latter not on the hub yet).  The ONE name source for
        every signal picker, so a not-yet-started node's output is selectable (#6).  CONTROL-plane
        signals (a panel's region) are NOT bindable data and are filtered here -- one choke point,
        so the flat combo, the tree combo, the expression popup and the pulse-scan y all agree (#7)."""
        names = {str(n) for n in self.hub.names()}
        names.update(self._signal_providers().keys())
        return sorted(n for n in names if not self._is_control_signal(n))

    def _signal_formats(self) -> dict:
        """``name -> standardized array shape`` for every LIVE hub signal, read straight
        off the most recent published VALUE (``shape_text.describe_shape``) -- AUTO from real
        data, never a hand-typed name->format map that could drift from what a node
        actually emits.  Lets the signal picker show each signal's SHAPE, not just its
        name (e.g. ``occupied  [(35,)]``)."""
        from zlc_data.shape_text import describe_shape
        out: dict[str, str] = {}
        for name in self.hub.names():
            try:
                st = self._signal_structure(name) or {}
                out[str(name)] = describe_shape(self.hub.latest(name), points_shape=st.get("points_shape"),
                                                data_shape=st.get("data_shape"), grid_shape=st.get("grid_shape"))
            except Exception:
                continue
        return out

    def _signal_axes(self) -> dict:
        """``{signal name: (axis_label, unit)}`` from each PROVIDER node's ``output_specs`` -- so a plot
        reads its y-axis label/unit from the producing measurement (the SignalSpec it declares), not a
        hard-coded per-kind string.  Reads the SAME ``_provider_nodes()`` source as every other signal-
        resolution site (running nodes + the last build kept past Stop), so a stopped node's panel keeps
        its declared axis label/unit instead of falling back to a default -- its signal still lingers in
        the hub, so its axis metadata must linger with it."""
        out: dict[str, tuple[str, str]] = {}
        for node in self._provider_nodes():
            if not hasattr(node, "output_specs"):
                continue
            try:
                for spec in node.output_specs():
                    out[str(spec.name)] = (spec.axis_label, spec.unit)
            except Exception:
                continue
        task_spec = self._task_mid_run_spec()
        if task_spec is not None:
            out[TASK_FRAME_KEY] = (task_spec.axis_label, task_spec.unit)
        return out

    def _signal_short_names(self) -> dict:
        """``{full hub signal: SHORT name}`` = each PROVIDER node's published signals with that node's
        prefix stripped (``temperature_survival`` -> ``survival``, ``frame`` -> ``frame``).  The picker
        nest binds this so the leaf shows the short name -- the SAME rule the Logic tab uses
        (``strip_node_prefix``), never the verbose SignalSpec axis label.  Reads the SAME
        ``_provider_nodes()`` source as every other signal-resolution site so a stopped node's lingering
        signal keeps its short name instead of showing the verbose full name."""
        out: dict[str, str] = {}
        for node in self._provider_nodes():
            pfx = str(getattr(node, "prefix", "") or "")
            try:
                for full in node.published_signals():
                    out[str(full)] = strip_node_prefix(str(full), pfx)
            except Exception:
                continue
        return out

    def _sites_inputs(self, occ_signal) -> tuple[str | None, str | None]:
        """For a site-map panel's ONE occupancy signal, return ``(centres_signal,
        image_signal)`` resolved from the SAME producing node -- so the site map pulls its
        centres + frame underlay from the one node that made the occupancy (rings + underlay
        = same shot).  This is the #6 "one signal" wiring: the user picks occupancy, the
        rest auto.

        The producing node is found among actual running nodes first, reading the
        centres/underlay output names from ``sitemap_centers_key`` and
        ``sitemap_image_key``.  A configured-but-not-yet-started Logic-tab row is
        the fallback, resolved from its spec metadata."""
        occ = str(occ_signal or "")
        if not occ:
            return (None, None)
        # 1) the producing node (ground truth: whatever publishes this signal) -- running OR kept
        # past Stop, so a stopped site-map panel still resolves its centres/underlay.
        for node in self._provider_nodes():
            ck = getattr(node, "sitemap_centers_key", "")
            if not ck:
                continue
            try:
                names = set(node.published_signals())
            except Exception:
                names = set()
            if occ in names:
                prefix = getattr(node, "prefix", "")
                ik = getattr(node, "sitemap_image_key", "")
                return (prefix + ck, (prefix + ik) if ik else None)
        # 2) a configured Logic-tab row whose node has not started yet (spec metadata)
        for row in self.logic_nodes:
            spec = self._spec_for_logic(row.node)
            meta = getattr(spec, "metadata", {}) or {}
            if not meta.get("centers_key"):
                continue
            node = self._logic_nodes.get(id(row))
            prefix = getattr(node, "prefix", "") if node is not None else ""
            names = set(node.published_signals()) if node is not None else set()
            default_occ = prefix + str(getattr(spec, "default_value_key", "") or "")
            if occ in names or (default_occ and occ == default_occ):
                ck, ik = meta.get("centers_key"), meta.get("image_key")
                return ((prefix + ck) if ck else None, (prefix + ik) if ik else None)
        return (None, None)

    def _curve_x(self, y_signal) -> str | None:
        """For a 1d plot wired to a scan's y CURVE, the companion x-axis signal resolved
        from the SAME producing node (the scan node exposes ``y_signal``/``x_signal``).  So
        wiring a 1d panel to ``temperature_survival`` draws it vs ``temperature_t_off`` with
        the right x-axis -- the user picks ONE signal (#3, same idea as the site map)."""
        y = str(y_signal or "")
        if not y:
            return None
        for node in self._provider_nodes():
            if getattr(node, "y_signal", None) == y:
                return getattr(node, "x_signal", None)
        return None

    def _pulse_state(self, value_signal):
        """For a PULSE panel's ``value`` signal, resolve ``(PulseTableState, include_always_off)`` off the
        SAME producing node (the loaded-figure node CARRIES the reproduction state as an attribute --
        the float-only hub cannot hold the object).  This is the SAME "auxiliary data from the producing
        node" wiring the site map uses for its centres / frame; the hub value is only a numeric
        placeholder.  ``None`` when no running node produces this signal with a pulse state."""
        name = str(value_signal or "")
        if not name:
            return None
        for node in self._provider_nodes():
            try:
                published = set(node.published_signals())
            except Exception:
                published = set()
            if name in published and getattr(node, "pulse_state", None) is not None:
                return (node.pulse_state, bool(getattr(node, "pulse_include_always_off", True)))
        return None

    def _grid_recipe(self, value_signal):
        """For a GRID panel's ``value`` signal, resolve its replay RECIPE (a dict) off the SAME producing
        node -- the loaded-figure node CARRIES the recipe as an attribute (the float-only hub cannot hold
        the dict), exactly as :meth:`_pulse_state` resolves a pulse panel's state.  ``None`` when no running
        node produces this signal with a grid recipe."""
        name = str(value_signal or "")
        if not name:
            return None
        for node in self._provider_nodes():
            try:
                published = set(node.published_signals())
            except Exception:
                published = set()
            if name in published and getattr(node, "grid_recipe", None) is not None:
                return node.grid_recipe
        return None

    def _live_node_formats(self, node) -> list[tuple[str, str, str]]:
        """``[(name, shape, description)]`` for a RUNNING node -- one ROW per output, each
        shape read off a real value via ``shape_text.describe_shape`` (auto, never hand-typed)
        and each description from the node's ``output_specs`` (what the signal MEANS).  A
        measurement / processor publishes to the hub under its prefix; a TASK is OFF the
        hub, so it documents what it streams mid-run (its ``output`` buffer) + what it
        produces (its ``result`` keys), shapes filled in as the values appear."""
        from zlc_data.shape_text import describe_shape
        specs = {s.name: s for s in node.output_specs()} if hasattr(node, "output_specs") else {}

        def desc(name: str) -> str:
            spec = specs.get(name)
            return spec.description if spec is not None else ""

        def schema_of(key: str):
            spec = specs.get(key)
            if spec is None:
                return None
            try:
                return spec.to_schema()
            except Exception:
                return None

        rows: list[tuple[str, str, str]] = []
        if getattr(node, "layer", "") == "task":
            buf = getattr(node, "output", None)
            for key in getattr(node, "mid_run", ()):
                if key in ("progress", "stage"):          # progress %/text live on the banner
                    continue
                value = buf.latest(key) if buf else None
                # #12: feed the declared schema so a task signal renders the SAME `R × P × (data)`
                # a measurement/processor signal does -- not the raw one-outer-paren fallback.
                rows.append((f"{key}{MID_RUN_TAG}", self._describe_from_schema(value, schema_of(key)), desc(key)))
            result = getattr(node, "result", None) or {}
            for key in getattr(node, "provides", ()):
                value = result.get(key) if isinstance(result, dict) else None
                rows.append((f"{key} (result)", self._describe_from_schema(value, schema_of(key)), desc(key)))
            return rows
        # published_signals() are HUB names (incl. the node's disambiguating prefix when two
        # nodes would collide).  Show the SHORT natural name (strip the prefix) because the
        # Logic row is already titled by the node.  ``output_specs`` (and so ``desc``) is
        # keyed by the FULL published name: look descriptions up by ``full``.
        pfx = str(getattr(node, "prefix", "") or "")
        for full in sorted(node.published_signals()):
            short = strip_node_prefix(full, pfx)             # ONE rule, shared with the picker nest
            try:
                # SAME schema-driven formatter as the task branch (#12): a hub signal and a task
                # output of the same logical shape render byte-identically -- no drift possible.
                # ZERO-COPY: read the tensor's ``.data`` (a reference -- only ``.shape`` is used) rather
                # than ``hub.latest`` (which .copy()s the whole 2.3 MB frame every tick just to format a
                # shape string, #4-E).
                shape = self._describe_from_schema(self.hub.latest_tensor(full).data, self.hub.schema(full))
            except Exception:
                shape = "—"
            rows.append((short, shape, desc(full)))
        return rows

    def _update_row_publishes(self, row: "LogicNodeRow") -> None:
        """Fill a Logic-tab row's "publishes:" legend (ONE signal per line: name, shape,
        meaning).  Shapes are AUTO-EXTRACTED from the real published VALUES
        (``shape_text.describe_shape``) and the meaning from the node's ``output_specs`` --
        never a hand-typed map.  Running node: live shapes off the hub (measurement /
        processor) or its mid-run buffer + result (task).  Stopped node: the NAMES it
        will produce (shape ``—`` until it runs)."""
        node = self._logic_nodes.get(id(row))
        if node is not None and getattr(node, "running", False):
            row.set_publishes(self._live_node_formats(node))
            return
        # The legend shows the SHORT name (strip the node prefix), exactly like the running path
        # (_live_node_formats); _declared_signal_keys now returns the FULL published names (#prebind).
        pfx = self._declared_node_prefix(row)
        row.set_publishes([(strip_node_prefix(k, pfx), "—", "") for k in self._declared_signal_keys(row)])

    def _declared_node_prefix(self, row: "LogicNodeRow") -> str:
        """The hub-signal PREFIX the row's node publishes under -- so a declared name == the
        published name (a binding made before Start, or restored from a saved layout, re-attaches
        the instant the producer starts, #prebind).

        The prefix is allocated ONCE, at Start, by the shared per-instance rule
        (``_logic_node_prefix``) and then STICKS as the instance's identity: a row that has (ever)
        been built reads its node's own ``prefix`` back -- re-running the collision rule later
        would drift with hub state (a sibling stopped by the device exclusion before it ever
        published leaves no trace, so a recomputation would 'un-collide' a name that IS published).
        Only a never-started row PREDICTS with the same rule.  A task is off the hub (its mid-run
        entry is a synthetic display tag, not a hub name) -> ``""``."""
        if row.node.kind == "task":
            return ""
        built = self._logic_nodes.get(id(row)) or (getattr(self, "_last_node", {}) or {}).get(id(row))
        if built is not None and hasattr(built, "prefix"):
            return str(built.prefix or "")
        return self._logic_node_prefix(row.node)

    def _declared_signal_keys(self, row: "LogicNodeRow") -> list[str]:
        """The FULL hub names a STOPPED Logic-tab node WILL publish once started -- IDENTICAL to the
        running node's ``published_signals()`` (same node prefix, #prebind), so the picker offers, and
        a Monitor binding stores, the SAME name the node later emits.  That makes a "connect first,
        start later" binding (and a save->load of one) re-attach automatically when the producer
        publishes that exact name.  The bare keys come from the ONE kind ladder
        (:meth:`_node_bare_keys` -- shared with the prefix collision check, so what the picker
        declares is exactly what is checked and published); a task is off the hub and instead
        shows a SYNTHETIC mid-run tag."""
        if row.node.kind == "task":                        # off-hub one-shot (never a hub signal)
            spec = self._spec_for_logic(row.node)
            return [f"{getattr(spec, 'mid_run_key', DEFAULT_MID_RUN_KEY)}{MID_RUN_TAG}"]
        pfx = self._declared_node_prefix(row)              # prepend the node prefix -> == published_signals()
        return [f"{pfx}{k}" for k in self._node_bare_keys(row.node)]

    def _reactive_ring(self, start_node) -> list[str] | None:
        """The signal-name path of an INDIRECT reactive ring that starting ``start_node`` would
        close (A consumes B's output while B consumes A's), or None.  Walked over REACTIVE edges
        only -- ``Processor.consumes`` (the versions that WAKE a node) -- because only an all-
        reactive ring is self-sustaining.  PulseScan consumes an external y through one bounded,
        cursor-ordered await per point; measurements therefore contribute outputs but never a
        reactive out-edge here.

        The graph is the RUNNING nodes only (``self.running_nodes`` -- GUI rows AND notebook-
        injected ``running_nodes=`` processors alike, which have no row at all).  A runtime ring
        can only exist between running nodes, and every candidate passes through here at ITS
        start, so whichever start CLOSES a ring is rejected -- start order still cannot smuggle
        one in.  Stopped rows deliberately contribute nothing: their stored values may be stale
        against what their next Start will actually build (rejecting a legal start on stale
        edges is worse than re-checking honestly at that later start).  The DIRECT self-loop is
        rejected at the base (Processor.__init__) -- one guard per scope.

        A reactive node is recognised by what it DECLARES, not by its class: ``consumes``
        is assigned by ``Processor.__init__`` alone, and the base already probes it the
        same way (``LogicNode._collect_provenance``: "probed by attribute, not by type").
        So the ring walk asks the node, and this layer needs no logic import at all."""
        if not getattr(start_node, "consumes", ()):
            return None
        producer = {}
        for node in self.running_nodes:
            if node is start_node or not hasattr(node, "published_signals"):
                continue
            try:
                for key in node.published_signals():
                    producer.setdefault(str(key), node)
            except Exception:
                continue
        targets = {str(s) for s in start_node.published_signals()}
        seen = set()
        stack = [(str(n), (str(n),)) for n in start_node.consumes]
        while stack:
            name, path = stack.pop()
            node = producer.get(name)
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            inputs = getattr(node, "consumes", ())
            for nxt in (str(n) for n in inputs):
                if nxt in targets:
                    return [*path, nxt]          # the ring closes on this node's own output
                stack.append((nxt, (*path, nxt)))
        return None

    def _node_for_signal(self, name: str):
        """The producing node for signal ``name``: a RUNNING node first, else the last build of a
        Logic-tab node (``_last_node``, kept past Stop) so a stopped node's lingering signal still
        resolves.  None if none publishes it."""
        for node in self._provider_nodes():
            if node is not None and hasattr(node, "published_signals"):
                try:
                    if name in node.published_signals():
                        return node
                except Exception:
                    continue
        return None

    @staticmethod
    def _schema_structure(schema) -> dict[str, object]:
        """Project one authoritative ``SignalSchema`` into the plot structure mapping."""

        ps = tuple(schema.point_shape)
        ds = tuple(schema.data_shape)
        metadata = dict(schema.metadata)
        grid = tuple(metadata.get("grid_shape", ()))
        if not grid and len(ps) == 2:
            grid = ps
        return {
            "points_shape": ps,
            "data_shape": ds,
            "grid_shape": grid,
            "ring": int(schema.repeat_capacity or 1),
            "metadata": metadata,
        }

    def _describe_from_schema(self, value, schema) -> str:
        """The ONE declared-shape string for any legend/picker row.  Projects a ``SignalSchema``
        through :meth:`_schema_structure` (the single grid rule) and renders it in the canonical
        ``R × P × (data)`` grammar, so a TASK output, a hub signal, a running node and a
        not-yet-published declared signal ALL read identically -- never the raw ``(R×P×data)``
        one-outer-paren spelling ``describe_shape`` falls back to when a caller forgets the schema
        (issue #12: calibration / mot-field task rows).  A canonical block's leading axis is R;
        with no value yet, R is the schema's declared repeat capacity.  No schema -> the raw
        value-only ``describe_shape`` (a scalar result still reads ``scalar``)."""
        from zlc_data.shape_text import contract_shape_label, describe_shape
        if schema is None:
            return describe_shape(value)
        st = self._schema_structure(schema)
        ps, ds, gs = st["points_shape"], st["data_shape"], st["grid_shape"]
        if value is not None:
            return describe_shape(value, points_shape=ps, data_shape=ds, grid_shape=gs)
        return contract_shape_label(int(st["ring"] or 1), ps, ds, gs)

    def _task_mid_run_spec(self):
        """The declared ``SignalSpec`` behind the running task panel's reserved source."""

        node = self._task_output_node
        if node is None or not hasattr(node, "output_specs"):
            return None
        try:
            return next(
                (spec for spec in node.output_specs() if str(spec.name) == self._task_mid_key),
                None,
            )
        except Exception:
            return None

    def _task_mid_run_schema(self):
        """The TaskOutput schema, available both before and after its first numeric publish."""

        node = self._task_output_node
        if node is None:
            return None
        try:
            return node.output.schema(self._task_mid_key)
        except KeyError:
            spec = self._task_mid_run_spec()
            return spec.to_schema() if spec is not None else None

    def _task_mid_run_structure(self):
        """Plot structure for the off-hub TaskOutput, without downgrading it to raw data."""

        schema = self._task_mid_run_schema()
        node = self._task_output_node
        if schema is None or node is None:
            return None
        result = self._schema_structure(schema)
        # A multidimensional TaskOutput point_shape is already the authoritative
        # scan geometry; unrelated domain geometry must not override it.
        if len(schema.point_shape) > 1:
            result["grid_shape"] = tuple(schema.point_shape)

        names = tuple(str(n) for n in (getattr(node, "scan_names", ()) or ()))
        arrays = tuple(getattr(node, "scan_arrays", ()) or ())
        if names:
            result["param_names"] = list(names)
        elif schema.metadata.get("coordinate_names"):
            result["param_names"] = [str(n) for n in schema.metadata["coordinate_names"]]
        if arrays:
            point_shape = tuple(result["grid_shape"] or result["points_shape"])
            if len(arrays) != len(point_shape):
                raise ValueError(
                    f"task declares {len(arrays)} coordinate arrays for point_shape {point_shape}.")
            coordinates = []
            for axis, (values, size) in enumerate(zip(arrays, point_shape)):
                values = np.asarray(values).reshape(-1)
                if values.size != size:
                    raise ValueError(
                        f"task coordinate axis {axis} has {values.size} values; expected {size}.")
                coordinates.append(values.tolist())
            result["points_coords"] = coordinates
        return result

    def _signal_structure(self, name: str):
        """Read the authoritative Hub or TaskOutput ``SignalSchema`` for plotting."""

        name = str(name)
        if name == TASK_FRAME_KEY:
            return self._task_mid_run_structure()
        try:
            schema = self.hub.schema(name)
        except KeyError:
            return None
        result = self._schema_structure(schema)
        metadata = dict(schema.metadata)
        coordinate_names = tuple(metadata.get("coordinate_signals", ()))
        if coordinate_names:
            arrays = []
            for coordinate_name in coordinate_names:
                coordinate = np.asarray(self.hub.latest(str(coordinate_name)), dtype=float)
                coordinate_schema = self.hub.schema(str(coordinate_name))
                expected = (coordinate.shape[0], coordinate_schema.point_count, 1)
                if coordinate_schema.data_shape != (1,) or tuple(coordinate.shape) != expected:
                    raise ValueError(
                        f"coordinate signal {coordinate_name!r} must be canonical (R,P,1); "
                        f"got {coordinate.shape}")
                arrays.append(coordinate[-1, :, 0].tolist())
            result["param_names"] = [str(value) for value in
                                     metadata.get("axis_order", coordinate_names)]
            result["points_coords"] = arrays
        return result

    def _refresh_signal_info(self) -> None:
        """Give every panel a legend (shown in its frame TITLE -- the grey strip, the old footer
        was removed) naming, FOR EACH signal the
        panel actually reads, WHICH node + layer produces it -- e.g.
        ``occupied ← occupancy [processor]``.  It lists only the signals this panel
        uses (not the producing node's whole output set), so the title answers
        exactly "this plot's value comes from which measurement/processor".  A read
        published by more than one running node is flagged ambiguous.  Self-guarded:
        recomputes only when the sources / nodes / published names change."""
        providers = self._signal_providers()
        sig = (tuple(sorted((k, len(v)) for k, v in providers.items())),
               tuple((id(c), c.config.source, tuple(c.config.inputs or ())) for c in self.cards))
        if sig == getattr(self, "_signal_info_sig", None):
            return
        self._signal_info_sig = sig
        # GC the hub of ORPHANED signals (#5 unbound): the AUTHORITATIVE invariant -- a hub signal that NO
        # live producer still publishes is stale and must leave, else it piles up as an "(unbound)" picker
        # entry run-after-run.  ``providers`` is the single source of "who publishes what" (running nodes +
        # stopped-but-kept rows via _last_node + declared rows); a hub name absent from it has no owner ->
        # purge.  A STOPPED node's signals are KEPT (its row is still a provider via _last_node -- a finished
        # scan stays plottable).  (A node error is surfaced via instance attrs + the banner, never as a hub
        # signal, so there is no reserved health channel to exempt here.)  This runs only when the provider
        # map / card sources CHANGED (the guard above), which a SWITCH does even with no rebuild: a live
        # ``frames_per_cycle`` 3->1 shrinks a running camera's published set {frame_0,1,2}->{frame_0},
        # dropping frame_1/frame_2 from ``providers`` -> they are caught here (the eager purges in
        # _start/_remove_logic_node cover the rebuild / remove paths synchronously; THIS catches every
        # remaining path, switches included).
        # CONTROL-plane signals (a panel's ``<slug>_region``) are published by the CONSOLE, not by a
        # LogicNode, so they have no ``providers`` entry -- exempt them BY THEIR DECLARED ROLE
        # (``schema.metadata['role'] == 'control'``, stamped by the ONE region encoder), never by a
        # parallel name list.  Their lifecycle is explicit: _remove_panel_analysis removes them with
        # the analysis (and _remove_panel with the panel), so nothing lingers.
        orphans = [n for n in self.hub.names()
                   if n not in providers and not self._is_control_signal(n)]
        if orphans:
            self.hub.remove_signals(orphans)
        for card in self.cards:
            reads = sorted(self._card_reads(card))
            parts: list[str] = []
            for name in reads:
                src = self._node_for_signal(name)
                if src is not None:
                    layer = str(getattr(src, "layer", ""))
                    tag = self._node_label(src)
                    tag = f"{tag} [{layer}]" if layer and layer != "node" else tag
                    note = "  ⚠ also from another node" if len(providers.get(name, [])) > 1 else ""
                    # The producing node is named to the RIGHT of the arrow, so the signal need
                    # not repeat its prefix -- the ONE strip_node_prefix rule the nested combo uses.
                    short = strip_node_prefix(name, getattr(src, "prefix", ""))
                    parts.append(f"{short} ← {tag}{note}")
                else:
                    parts.append(f"{name} ← (no running source)")
            if not reads:
                parts.append("(no signal set — pick one in Setting: value = <signal>)")
            # one read per line: "<signal> ← <node> [layer]" -- the value's origin only,
            # not the producing node's full output list.
            card.set_signal_info("\n".join(parts))

    def _restart_node(self, node, new_params: dict):
        """Apply edited acquisition parameters to the producing node so the
        Monitor re-acquires under them.  This goes through the node's SAFE entry
        ``apply_acquisition_parameters``: while the acquisition loop runs it owns
        the source, so the edit is queued and the loop applies it BETWEEN shots
        (the source re-arms in its owner thread -- a streaming camera picks up the
        new ROI/exposure -- with no GUI-thread stall and no second ``acquire()``
        racing on one camera; an in-place reconfigure of a running stream would be
        ignored, while stopping/starting the thread from here would block the GUI
        and could deadlock).  An idle node is reconfigured and stepped once.
        Returns the node.  Raises if the node has no editable acquisition params."""
        if node is None or not hasattr(node, "apply_acquisition_parameters"):
            raise RuntimeError("this panel's data source exposes no editable acquisition parameters")
        node.apply_acquisition_parameters(**new_params)
        # Edit's Apply makes the source LIVE: if its acquisition loop is not
        # running (a fresh node, or one the user stopped), start it so the Monitor
        # streams under the new params.  start() is idempotent, so a node that is
        # already running just keeps its loop (the edit was queued above).
        if not getattr(node, "running", False) and hasattr(node, "start"):
            fence = self._current_runtime_fence()
            if fence is None:
                if tuple(getattr(node, "referenced_devices", lambda: ())()):
                    raise RuntimeError(
                        "device-bearing nodes require a runtime authority"
                    )
                node.start()
            else:
                handle = fence.start(node)
                self._legacy_handles[id(node)] = handle
                self._legacy_handle_fences[id(node)] = fence
                row = next(
                    (
                        candidate
                        for candidate in self.logic_nodes
                        if self._logic_nodes.get(id(candidate)) is node
                        or self._last_node.get(id(candidate)) is node
                    ),
                    None,
                )
                if row is not None:
                    self._starting_nodes[id(row)] = node
                    self._pending_fenced_starts[id(node)] = set()
                    try:
                        handle.wait_started(0.05)
                    except Exception:
                        pass
                    if handle.started:
                        self._finalize_fenced_start(row, node)
                else:
                    # Row-less injected sources have no UI poll owner.  Their launcher
                    # performs the admission wait; a stopped row-less source cannot be
                    # restarted implicitly from a panel edit.
                    try:
                        handle.wait_started(0.05)
                    except TimeoutError as exc:
                        handle.cancel("row-less source start did not cross admission")
                        raise RuntimeError(
                            "row-less device source restart requires its composition launcher"
                        ) from exc
                    if node not in self.running_nodes:
                        self.running_nodes.append(node)
        return node

    def install_runtime_fence(self, runtime_fence: object) -> None:
        """Replace a static stopped generation for non-provider test/standalone hosts."""

        required = ("start", "stop", "handle_for")
        if not all(callable(getattr(runtime_fence, name, None)) for name in required):
            raise TypeError("runtime_fence must provide start(), stop(), and handle_for()")
        if self.running_nodes or self._starting_nodes:
            raise RuntimeError("cannot replace runtime fence while nodes are active")
        self._legacy_runtime_fence = runtime_fence

    def _enroll_composed_nodes(
        self,
        nodes: Sequence[object],
        runtime_fence: object | None,
    ) -> None:
        """Start composition-provided nodes through the same runtime authority.

        A device-bearing node is never adopted while already running: it first has
        to acknowledge termination, then the fence owns its fresh start.  The
        launcher and offscreen composition tests call this one entry so enrollment
        semantics cannot drift.
        """

        for node in nodes:
            if not hasattr(node, "start"):
                continue
            if runtime_fence is None:
                if tuple(getattr(node, "referenced_devices", lambda: ())()):
                    raise RuntimeError(
                        "device-bearing running_nodes require a runtime authority"
                    )
                if not getattr(node, "running", False):
                    node.start()
                if node not in self.running_nodes:
                    self.running_nodes.append(node)
                continue
            if getattr(node, "running", False):
                if not self._stop_unmanaged_node_confirmed(node, timeout=2.0):
                    raise RuntimeError(
                        "passed running node could not terminate for LegacyRuntimeFence enrollment"
                    )
            handle = runtime_fence.start(node)
            self._legacy_handles[id(node)] = handle
            self._legacy_handle_fences[id(node)] = runtime_fence
            handle.wait_started(2.0)
            if node not in self.running_nodes:
                self.running_nodes.append(node)

    def _current_runtime_fence(self):
        provider = self._runtime_fence_provider
        fence = provider() if provider is not None else self._legacy_runtime_fence
        if fence is None:
            return None
        required = ("start", "stop", "handle_for")
        if not all(callable(getattr(fence, name, None)) for name in required):
            raise TypeError("runtime fence provider returned an invalid authority")
        return fence

    def _edit_card(self, card: "PanelCard") -> None:
        """Open (or focus) this PLOT panel's OWN closable Edit tab (a PanelEditor); a
        running task locks every other action, so this no-ops then.

        Opens even BEFORE the panel has data -- the panel's plot params /
        acquisition / fit / limits are editable straight away.  The snapshot
        section just shows "waiting for data" until a plot exists (the data is
        produced by a Logic-tab node, not by this Edit)."""
        if card is None or self._task_locked:
            return
        # the Edit snapshot reads the live plotter's data arrays -- own the figure first
        if not self._render_barrier():
            self._render_handoff_failed("Edit")
            return
        existing = self._panel_editors.get(id(card))
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            existing.rebuild()
            return
        editor = PanelEditor(card, self)
        self._panel_editors[id(card)] = editor
        title = (card.config.title or PANEL_KINDS[card.config.kind]).strip() or "panel"
        self.tabs.add_closable_tab(editor, title)

    def _close_panel_editor(self, card: "PanelCard") -> None:
        """Close a card's Edit tab if open (called when the card is removed)."""
        editor = self._panel_editors.pop(id(card), None)
        if editor is None:
            return
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        editor.teardown()
        editor.setParent(None)
        editor.deleteLater()

    def _refresh_panel_editor(self, card: "PanelCard") -> None:
        """Rebuild a card's OPEN Edit tab in place (same tab slot, same selection) -- used when the
        panel's resolved param kind changed underneath it (a grid's facet / sub-plot pick), so the
        page's baked param rows follow instead of lying."""
        old = self._panel_editors.get(id(card))
        if old is None:
            return
        index = self.tabs.indexOf(old)
        was_current = self.tabs.currentWidget() is old
        self._close_panel_editor(card)
        editor = PanelEditor(card, self)
        self._panel_editors[id(card)] = editor
        title = (card.config.title or PANEL_KINDS[card.config.kind]).strip() or "panel"
        self.tabs.add_closable_tab(editor, title)
        new_index = self.tabs.indexOf(editor)
        if 0 <= index < new_index:
            self.tabs.tabBar().moveTab(new_index, index)     # keep the tab where the user left it
        if was_current:
            self.tabs.setCurrentWidget(editor)
        editor.rebuild()

    def _on_editor_tab_closed(self, widget) -> None:
        """X on a PanelEditor / LogicNodeEditor tab: tear it down + drop it from the
        registry.  The permanent Monitor / Logic tabs carry no X, so they never
        arrive here.  A logic node's Edit only closes the TAB -- the node keeps
        running and stays in the Logic list (reopen its Edit from its row)."""
        index = self.tabs.indexOf(widget)
        if index >= 0:
            self.tabs.removeTab(index)
        for key, editor in list(self._panel_editors.items()):
            if editor is widget:
                del self._panel_editors[key]
        for key, editor in list(self._logic_editors.items()):
            if editor is widget:
                del self._logic_editors[key]
        if hasattr(widget, "teardown"):
            widget.teardown()
        widget.setParent(None)
        widget.deleteLater()

    def _on_tab_changed(self, _index: int) -> None:
        # ONE rule: when a tab becomes visible, give whatever's inside a chance to refresh its
        # dynamic content -- any widget that implements ``refresh_on_show()`` runs it.  No
        # special-casing per tab type (PanelEditor / LogicNodeEditor / MeasurementPanel all
        # honour the same hook), so a brand new editor automatically participates by
        # implementing ``refresh_on_show`` -- nothing here changes.  This kills the bug where
        # a processor's source dropdown stayed empty after switching tabs.
        widget = self.tabs.currentWidget()
        hook = getattr(widget, "refresh_on_show", None)
        if callable(hook):
            try:
                hook()
            except Exception:
                pass

    def _arrange_if_cards(self) -> None:
        """Re-pack the board (deferred one-shot on show) ONLY when it holds cards -- a no-op on an
        empty board, so an embedded console with nothing loaded yet does not thrash."""
        if getattr(self, "cards", None):
            self._arrange()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        # Re-pack the gravity board when its scroll viewport WIDTH changes (window resized): the pack
        # width is demand-driven (``_pack_width`` = the viewport width grown to the furthest card's
        # reach), so WIDENING lets more columns fit and NARROWING re-flows cards that now fit while any
        # card the user parked past the edge keeps the board wide + a horizontal scrollbar.  Guard on an
        # actual width change (no redundant packs).
        if (getattr(self, "scroll", None) is not None and obj is self.scroll.viewport()
                and event.type() == QtCore.QEvent.Resize):
            w = self.scroll.viewport().width()
            if w != getattr(self, "_board_view_w", 0) and getattr(self, "cards", None):
                self._board_view_w = w
                self._arrange()
        return super().eventFilter(obj, event)

    def _snap_dropped_card(self, card: PanelCard) -> None:
        """Reorder a JUST-DROPPED card to the ORDER position nearest its raw drop point
        (:func:`drop_index`): dropping ONTO a card's slot DISPLACES it (insert before it, so it and
        everything after shift down and re-pack), dropping past the last card appends to the bottom.
        Connected to ``PanelCard.dropped`` -- the drag-release path ONLY; the ``layout_changed`` the
        card emits right after re-packs the new order (:func:`pack` recomputes every pixel), so resize
        / Add never reorder.  The ORDER (``self.cards``) is the layout's single source of truth."""
        if card not in self.cards:
            return
        others = [c.config for c in self.cards if c is not card]
        idx = drop_index(card.config, others, self._pack_width([c.config for c in self.cards]))
        self.cards.remove(card)
        self.cards.insert(idx, card)

    def _arrange(self) -> None:
        # ONE order-driven north-west pack (:func:`pack`): each card, IN ``self.cards`` ORDER, floats
        # to the first free top-left slot.  A drag has already reordered ``self.cards`` (via
        # _snap_dropped_card) so the drop is honoured, while gravity stays strictly top-left and
        # deterministic -- a resize/click re-packs to the SAME arrangement (no surprise reflow) and an
        # Add (appended last) lands in the next bottom slot, never a middle hole (#2).
        board_w = self._pack_width([c.config for c in self.cards])
        if pack([c.config for c in self.cards], board_w):
            self._mark_dirty()
        # Repack as one atomic frame: board.arrange moves EVERY card (card.move) + resizes the board,
        # so without a freeze each card's move paints on its own and a multi-card reflow reads as a
        # cascade of little jumps.  The SAME updates-disabled primitive the resized card uses in
        # _on_size, applied one level up so the whole reflow composites in a single flush.
        self.board.setUpdatesEnabled(False)
        try:
            self.board.arrange(self.cards)
        finally:
            self.board.setUpdatesEnabled(True)
        self._update_summary()

    def _pack_width(self, configs: Sequence["PanelConfig"]) -> int:
        """The width :func:`pack` wraps at = the live scroll-viewport width (pack clamps it up to
        one-card-wide for a card wider than the viewport).  Strict NW gravity means the board only ever
        grows DOWN (vertical scroll) -- it never parks a card in an off-screen column, so there is no
        demand-grown horizontal extent to track any more (that was the parking the user asked us to
        stop, #2).  No viewport yet (pre-show) -> the headless two-wide fallback."""
        vw = self.scroll.viewport().width() if hasattr(self, "scroll") else 0
        return vw if vw else _board_width(configs)

    # ------------------------------------------------------------------ actions
    def _add_panel(self) -> None:
        """Header "Add Panel": add either a BLANK plot panel (a plot kind) or a
        STOPPED logic node (camera / measurement / processor / task)."""
        if self._task_locked:
            return
        data = self.kind_combo.currentData()
        # A node LAYER -> a STOPPED logic node on the Logic tab (camera /
        # measurement / processor / task).  It publishes nothing until Started.
        if isinstance(data, tuple) and len(data) == 2 and data[0] in LOGIC_KINDS:
            kind, name = data
            # The camera's row title = the spec's display name resolved at populate time
            # (readout.camera_spec().name -- ONE source); every other kind's name IS its
            # catalog spec's name already.
            title = self._camera_title if kind == "camera" else str(name)
            self._add_logic_node(LogicNodeConfig(kind=kind, name=name, title=title), focus=True)
            return
        # Otherwise a PLOT kind -> a BLANK pure-view panel on the Monitor board
        # (decoupled: it shows nothing until a signal is picked in its Setting).
        kind = data or "1d"
        # Every panel gets a unique "<kind> #N" title (G1) so two of the same kind never share
        # one name in the card header / Edit tab / frame title.
        title = indexed_unique_name(PANEL_KINDS[str(kind)], {c.config.title for c in self.cards})
        config = PanelConfig(kind=str(kind), title=title, row=GAP, col=GAP, size="1x2")
        if str(kind) == "grid":
            # An Add-Panel grid IS the axis-expander: default to faceting the repeat axis so binding
            # a signal shows cells at once ("(recipe)" is only meaningful for a LOADED figure's grid).
            config.params["facet"] = "repeat"
        # APPEND the new card LAST in order (``_attach_card`` adds it to the end of ``self.cards``);
        # the order-driven :func:`pack` in ``_arrange`` then lands it in the next free BOTTOM slot,
        # never a middle hole (#2).  No pixel seed needed -- pack recomputes every card's position from
        # the order alone.
        card = self._new_panel_card(config)
        self._attach_card(card)
        self._arrange()
        self._mark_dirty()

    def _remove_panel(
        self,
        card: PanelCard,
        *,
        timeout: float = 5.0,
        render_already_stopped: bool = False,
        allow_task_owned: bool = False,
    ) -> bool:
        # User removal is blocked while a task owns the console.  The task lifecycle uses the
        # explicit ``allow_task_owned`` capability; it never temporarily unlocks the UI merely to
        # obtain teardown authority.
        if self._task_locked and not allow_task_owned:
            return False
        phases = self._panel_teardown_phases.setdefault(id(card), set())
        if "detached" in phases:
            return True
        if card not in self.cards and "removed" not in phases:
            return False
        if not render_already_stopped:
            if not self._render_barrier(max(0.0, float(timeout))):
                return False
        if "analysis" not in phases:
            self._remove_panel_analysis(card)  # analysis row + region signal go with the panel (#1/#7)
            phases.add("analysis")
        if "editor" not in phases:
            card.settings_popup.hide()
            self._close_panel_editor(card)     # drop this card's Edit tab too
            phases.add("editor")
        # Shutdown is the acknowledgement that the card no longer owns render/selector resources.
        # It must succeed before the card disappears from the registry; otherwise a retry would see
        # "not in cards" and falsely report completion while teardown never happened.
        if "shutdown" not in phases:
            card.shutdown()
            phases.add("shutdown")
        if "removed" not in phases:
            self.cards.remove(card)
            phases.add("removed")
        if "qt_detached" not in phases:
            card.setParent(None)
            card.deleteLater()
            phases.add("qt_detached")
        if "arranged" not in phases:
            self._arrange()
            self._recompute_tick_interval()    # removing the fastest panel can slow the base
            self._mark_dirty()
            phases.add("arranged")
        phases.add("detached")
        self._panel_teardown_phases.pop(id(card), None)
        return True

    # ====================================================================== logic nodes
    def _spec_for_logic(self, node: LogicNodeConfig):
        """The declarative spec a logic node builds from -- it carries the node's
        editable ParamDecls (so the Edit auto-form renders them) and a ``build``.
        The camera's spec comes from ``readout.camera_spec()`` (its params are the
        camera's exposure / frames-per-cycle), the others from the catalogs."""
        if node.kind == "camera":
            readout = getattr(self.session, "readout", None)
            return readout.camera_spec() if readout is not None and hasattr(readout, "camera_spec") else None
        if node.kind == "measurement":
            return next((s for s in self.measurements if getattr(s, "name", None) == node.name), None)
        if node.kind == "processor":
            return next((s for s in self.processors if getattr(s, "name", None) == node.name), None)
        if node.kind == "task":
            return next((s for s in self.tasks if getattr(s, "name", None) == node.name), None)
        return None

    def _unique_logic_title(self, title: str) -> str:
        """Make a logic-node title ``"<base> #N"`` and UNIQUE among the existing Logic rows
        (G1) -- so two same-kind nodes are told apart in the Logic
        rows AND get distinct per-instance signal prefixes (see _logic_node_prefix).  Re-indexes
        a loaded title's root, so an already-clean saved layout round-trips."""
        return indexed_unique_name(title, {str(r.node.title) for r in self.logic_nodes})

    def _node_bare_keys(self, node: LogicNodeConfig) -> list[str]:
        """The SHORT (un-prefixed) output names a logic node emits -- the ONE derivation shared by
        the prefix collision check (:meth:`_logic_node_prefix`) and the declared picker names
        (:meth:`_declared_signal_keys`), so what is checked for collision is exactly what will be
        published.  Processor: the spec's ``result_keys``; measurement: its x/y curve keys (or, when
        the spec exposes a ``declared_keys`` deriver, the REAL published names for its form values --
        pulse-scan overrides ``x_key="param"`` with the semantic scan coordinate, so the deriver keeps
        declared == published); camera: ``frame_i`` per emCCD event (``frames_per_cycle``, the same
        helper ``CameraMeasurement.published_signals`` uses so they can never drift)."""
        spec = self._spec_for_logic(node)
        declared = (getattr(spec, "metadata", None) or {}).get("declared_keys")
        if callable(declared):
            try:
                return [str(k) for k in declared(node.values or {})]
            except Exception:
                pass                                       # fall back to the static spec keys below
        keys = [str(k) for k in (getattr(spec, "result_keys", ()) or [])]
        if not keys and node.kind == "measurement":
            keys = [k for k in (getattr(spec, "x_key", ""), getattr(spec, "y_key", "")) if k]
        if not keys and node.kind == "camera":
            from zlc_data.shape_text import camera_frame_keys
            keys = camera_frame_keys((node.values or {}).get("frames_per_cycle", 1))
        return keys

    def _logic_node_base_prefix(self, node: LogicNodeConfig) -> str:
        """The KIND-semantic default prefix a logic node publishes under when nothing collides.
        Measurement: ``f"{spec.key}_"`` so every signal self-describes its quantity
        (``temperature_t_off`` -- one name, derived, #H3r).  Camera / processor: ``""`` -- the short
        natural names (``frame_0`` / ``occupied``); the PRODUCER is shown by the signal-flow legend
        and never baked into the signal (which camera INSTANCE, let alone which DEVICE, is not the
        signal's business -- a consumer binds an instance's output, not a piece of hardware)."""
        if node.kind == "measurement":
            spec = self._spec_for_logic(node)
            return f"{spec.key}_" if spec is not None else ""
        return ""

    def _logic_node_prefix(self, node: LogicNodeConfig) -> str:
        """The hub-signal prefix for a logic node -- the ONE rule for EVERY kind (camera /
        measurement / processor): the kind-semantic base prefix (:meth:`_logic_node_base_prefix`)
        by default, upgraded to a per-INSTANCE slug (from the row's unique title, ``occupancy_2_``)
        only when the base-prefixed names would COLLIDE with another node's signals.  A signal's
        namespace is the logic-node INSTANCE that produces it: two rows of the same kind can never
        overwrite each other, and no device / backend identity ever leaks into a signal name."""
        keys = self._node_bare_keys(node)
        base = self._logic_node_base_prefix(node)
        # Collision is checked against EVERY signal live in the hub -- running nodes AND a STOPPED
        # node's lingering signals -- not just running_nodes (#2): otherwise a new same-kind node added
        # after an earlier one STOPPED (its signals deliberately linger) would see no running collision,
        # take the empty prefix, and CLOBBER the stopped node's lingering data on the hub.
        running: set[str] = set(self.hub.signal_versions())
        for n in self.running_nodes:
            try:
                running.update(str(s) for s in n.published_signals())  # just-started, maybe not in hub yet
            except Exception:
                pass
        # A RESTART reclaims its OWN lingering signals -- they are not a collision (#issue-1): without
        # this, restarting a node sees its own STOPPED outputs still in the hub, takes a fresh
        # numbered prefix, and every panel bound to the original names goes UNBOUND.  Its
        # prior built node survives Stop in _last_node tagged with instance_label == this node's title.
        # (The same rule is what lets a row SWITCH its camera device and restart: the row keeps its
        # signal names, the new device's frames simply flow under them -- panels follow seamlessly.)
        for prev in (getattr(self, "_last_node", {}) or {}).values():
            if prev is not None and str(getattr(prev, "instance_label", "")) == str(node.title):
                try:
                    running.difference_update(str(s) for s in prev.published_signals())
                except Exception:
                    pass
        if not keys or not any((base + key) in running for key in keys):
            return base                              # no collision (incl. own restart) -> the base names
        from zlc_data.shape_text import measurement_slug
        slug = measurement_slug(node.title or node.name) or str(node.kind) or "node"
        prefix, k = f"{slug}_", 2
        while prefix in {getattr(n, "prefix", "") for n in self.running_nodes} \
                or any((prefix + key) in running for key in keys):
            prefix = f"{slug}_{k}_"
            k += 1
        return prefix

    def _attach_logic_node(self, node: LogicNodeConfig, *, focus: bool = False) -> "LogicNodeRow | None":
        """Add a STOPPED logic-node row to the Logic tab (no node built yet).  A NO-OP when embedded:
        the figure viewer has no Logic tab (``logic_layout`` is None), so there is nowhere to host the
        row -- the console is a pure view host, its acquisition is whatever produced the loaded figure."""
        if self.logic_layout is None:                        # embedded: no Logic tab to attach to
            return None
        node.title = self._unique_logic_title(node.title)   # distinct rows + distinct signal prefixes
        row = LogicNodeRow(node)
        row.edit_requested.connect(self._edit_logic_node)
        row.remove_requested.connect(self._remove_logic_node)
        row.start_requested.connect(self._start_logic_node)   # Start/Stop act on the row itself (#5)
        row.stop_requested.connect(self._stop_logic_node)
        # insert ABOVE the trailing stretch (the hint + stretch are the last 2 items)
        self.logic_layout.insertWidget(self.logic_layout.count() - 1, row)
        self.logic_nodes.append(row)
        self._logic_nodes[id(row)] = None
        self.logic_hint.hide()
        self._update_row_publishes(row)                       # show its outputs + shapes up front
        if focus and hasattr(self.tabs, "setCurrentWidget"):
            self._edit_logic_node(row)
        return row

    def _add_logic_node(self, node: LogicNodeConfig, *, focus: bool = True) -> "LogicNodeRow":
        row = self._attach_logic_node(node, focus=focus)
        self._mark_dirty()
        return row

    # ------------------------------------------------------- selector -> analysis chain
    def _on_panel_area_select(self, card: "PanelCard", selection) -> None:
        """Dispatch a drag-selection by the panel's ARMED analysis (a single drag has one meaning): a
        fit-on panel (``fit_request`` present) RETARGETS its fit to the new selection through the ONE
        card mutator; else the ROI action crops; else just report the count."""
        if card.config.params.get("fit_request"):
            card.set_fit_request(card._retarget_fit_request(selection))
            return
        action = str(card.config.params.get("selection_action") or "none")
        if action == "roi":
            self._apply_roi_selection(card, selection)
        else:
            try:
                count = card.plotter.to_data_figure().selected_data(selection).count
                card.set_status(f"selected {count} data points", error=False)
            except Exception as exc:
                card.set_status(f"selection invalid: {str(exc).splitlines()[0][:100]}", error=True)

    def _analysis_node_title(self, card: "PanelCard") -> str:
        """A per-PANEL analysis-node title derived from the panel's OWN title (``"2D image #1
        analysis"``), made unique among the Logic rows.  The node prefix derives from this title
        (:meth:`_logic_node_prefix`), so two panels on the SAME source get DISTINCT nodes AND distinct
        published output names -- the root fix for the old one-node-per-source sharing (#3)."""
        base = f"{card.config.title or 'panel'} analysis"
        return indexed_unique_name(base, {str(r.node.title) for r in self.logic_nodes})

    def _panel_analysis_row(self, card: "PanelCard") -> "LogicNodeRow | None":
        """This panel's ONE Analysis row, DERIVED from the persisted single source: the row whose
        ``values['region']`` equals the panel's stored ``params['region_signal']``.  No runtime dict,
        no id() key -- save/load re-associates for free, and a hand-removed row simply derives None."""
        name = str(card.config.params.get("region_signal") or "")
        if not name:
            return None
        for row in self.logic_nodes:
            if str((row.node.values or {}).get("region") or "") == name:
                return row
        return None

    def _remove_panel_analysis(self, card: "PanelCard") -> None:
        """Tear down THIS panel's analysis whole: stop + remove its Analysis row, THEN remove its
        region signal from the hub and drop the persisted association (``region_signal`` /
        ``region``).  The ONE teardown seam every analysis-off path funnels through -- clearing a
        fit, switching the Analysis action to none, turning the Selectors switch off, and panel
        removal -- symmetric for fit and ROI, and symmetric for the region itself (#7: a cleared
        analysis leaves NO orphan control signal behind).  Node first, region second: a running
        consumer must never see its region vanish."""
        row = self._panel_analysis_row(card)
        if row is not None:
            if not self._remove_logic_node(row):
                raise RuntimeError(
                    "cannot remove analysis while its owner thread is still active"
                )
        region = card.config.params.pop("region_signal", None)
        card.config.params.pop("region", None)
        if region:
            self.hub.remove_signals([str(region)])
        self._fit_overlay_pushed.pop(id(card), None)   # a fresh fit re-pushes from version -1
        if card.plotter is not None and hasattr(card.plotter, "apply_published_fit"):
            card.plotter.apply_published_fit(None)     # drop any live fit overlay too

    def _region_signal_names_in_use(self) -> set:
        """Every region name that may NOT be handed to a new panel: live hub names, every Logic row's
        consumed region (running or stopped, loaded or fresh), and every panel's persisted name.  The
        dedup scope that makes cross-linking structurally impossible (two panels can never share a
        region name, even across save/load/rename)."""
        names = {str(n) for n in self.hub.registered_names()}
        for row in self.logic_nodes:
            region = str((row.node.values or {}).get("region") or "")
            if region:
                names.add(region)
        for other in self.cards:
            region = str(other.config.params.get("region_signal") or "")
            if region:
                names.add(region)
        return names

    def _publish_region(self, card: "PanelCard", selection) -> str:
        """Publish THIS panel's drawn Selection as its ``<slug>_region`` hub CONTROL signal -- the ONE
        control input its Analysis node consumes.  The name is minted ONCE (deduped against every
        persisted region name -- :meth:`_region_signal_names_in_use`) and stored in
        ``config.params['region_signal']``; the drawn payload is stored in ``config.params['region']``
        (both persist with the panel, so save/load replays the region and re-associates the row).  A
        re-drag republishes the SAME name with a byte-stable schema (:func:`region_tensor` -- fixed
        shape, ``role='control'``), so a retarget can never fork the schema or gap a running consumer."""
        from zlc_data.plot_region import region_doc, region_tensor
        from zlc_data.shape_text import measurement_slug
        name = str(card.config.params.get("region_signal") or "")
        if not name:
            slug = measurement_slug(card.config.title) or "panel"
            taken = self._region_signal_names_in_use()
            name = f"{slug}_region"
            k = 2
            while name in taken:
                name = f"{slug}_{k}_region"
                k += 1
            card.config.params["region_signal"] = name
        bins = int(card.config.params.get("bins", 50)) if card.config.kind == "hist" else None
        card.config.params["region"] = region_doc(selection, bins=bins)
        tensor = region_tensor(selection, bins=bins)
        self.hub.register_signal(name, tensor.schema)     # idempotent: the schema is byte-stable
        self.hub.publish({name: tensor})
        return name

    def _published_fit_result(self, node):
        """Build a :class:`FitResult` from a FitProcessor's PUBLISHED parameters on the hub (its first
        cell) so the panel overlay DRAWS from solved params with no Qt-thread solve (#6).  ``None`` when
        the node has no published result yet."""
        from zlc_data.curve_fitting import FitResult, fit_model
        model = fit_model(node.fit_request.model)
        prefix = node.prefix

        def _scalar(key):
            return float(np.asarray(self.hub.latest(prefix + key)).reshape(-1)[0])

        try:
            params = np.array([_scalar(f"fit_{name}") for name in model.names], dtype=float)
            valid = bool(np.asarray(self.hub.latest(prefix + "fit_valid")).reshape(-1)[0])
        except Exception:
            return None
        if not np.isfinite(params).all():
            valid = False
        quality = {}
        for qkey, sig in (("rmse", "fit_rmse"), ("r2", "fit_r2")):
            try:
                quality[qkey] = _scalar(sig)
            except Exception:
                quality[qkey] = float("nan")
        try:
            n_points = int(_scalar("fit_points"))
        except Exception:
            n_points = 0
        return FitResult(model.key, model.names, params, None, valid,
                         "ok" if valid else "invalid", quality, n_points,
                         node.fit_request.coordinate_frame)

    def _update_fit_overlays(self) -> None:
        """GUI thread (after each coherent present): push every fit panel's per-panel FitProcessor
        PUBLISHED result to its plotter's DISPLAY-only overlay AND to the fit result line on BOTH surfaces
        (the Setting popup + the open Edit tab).  This is how a fit reaches the plot now -- the overlay
        reconstructs the curve/dot from published parameters, never solving on the Qt thread (#6).  Gated
        on the node's published version so an unchanged fit costs nothing (and the result labels are
        change-gated on top, so no per-tick setText thrash)."""
        from .live import GridPlot
        for card in self.cards:
            plotter = card.plotter
            is_grid = isinstance(plotter, GridPlot)
            if plotter is None or not (is_grid or hasattr(plotter, "apply_published_fit")):
                continue
            row = self._panel_analysis_row(card)
            node = self._logic_nodes.get(id(row)) if row is not None else None
            # A node publishes fit parameters when it SAYS it does.  An Analysis node inherits BOTH
            # strategies and its ``provides`` dispatches on the CURRENT action (analysis.py keeps
            # declared == published per action), so the declaration already answers "is this fitting
            # right now" -- and it names the very signal the version gate below reads, so the test and
            # the gate cannot drift apart.
            if node is None:                       # no analysis node bound to this panel yet
                continue
            fit_valid = f"{node.prefix}fit_valid"   # every LogicNode carries prefix/published_signals
            if fit_valid not in node.published_signals():
                continue
            version = self.hub.signal_versions().get(fit_valid, -1)
            if is_grid:
                # #6b: a facet grid fit is the SAME worker node, publishing per-cell params.  Push them
                # to the grid, which reconstructs each cell's curve (solve=False) -- so a grid fit NEVER
                # solves on the Qt thread either.  Gate on the version, but ALSO push once when the grid
                # is not yet in console (display-only) mode, so it stops solving in place from the start.
                if self._fit_overlay_pushed.get(id(card)) == version \
                        and getattr(plotter, "_published_cell_popt", None) is not None:
                    continue
                self._fit_overlay_pushed[id(card)] = version
                model, cell_popts = self._published_cell_fits(node)
                plotter.apply_published_cell_fits(cell_popts, model=model)
                card._set_fit_result_text(self._published_fit_result(node))
                continue
            # Skip only when the version is unchanged AND this very plotter already carries the pushed
            # overlay: a legitimate structural rebuild swaps in a FRESH plotter (which never had
            # apply_published_fit called -- no _fit_overlay_result attribute), and without this second
            # gate an unchanged fit version would leave the new plotter overlay-less until the next fit
            # publish (the #3 risks' one-line hygiene fix, keyed on the plotter instead of a card hook).
            if self._fit_overlay_pushed.get(id(card)) == version \
                    and hasattr(plotter, "_fit_overlay_result"):
                continue
            self._fit_overlay_pushed[id(card)] = version
            result = self._published_fit_result(node)
            plotter.apply_published_fit(result)
            # #6 stale-result fix: the node's SOLVED result now reaches the fit result line on BOTH
            # surfaces (previously only overlays updated, so the Setting result row said 'not fitted'
            # forever).  Both writers change-gate, so this is free when the text is unchanged.
            card._set_fit_result_text(result)
            editor = self._panel_editors.get(id(card))
            edit_controls = getattr(editor, "_analysis_controls", None) if editor is not None else None
            if edit_controls is not None:
                edit_controls.set_result(card._fit_result_text(result))

    def _published_cell_fits(self, node) -> tuple[str, dict]:
        """Read a facet FitProcessor node's PUBLISHED per-cell params off the hub as ``(model_key,
        {cell_index: (p0, p1, ...)})`` in the model's parameter order -- the per-cell counterpart of
        :meth:`_published_fit_result`, so a grid draws every cell from solved params with no Qt solve
        (#6b).  A cell whose fit is invalid (NaN / not converged) is omitted (that cell draws nothing)."""
        from zlc_data.curve_fitting import fit_model
        model = fit_model(node.fit_request.model)
        prefix = node.prefix
        try:
            valid = np.asarray(self.hub.latest(prefix + "fit_valid")).reshape(-1)
            params = [np.asarray(self.hub.latest(prefix + f"fit_{name}")).reshape(-1)
                      for name in model.names]
        except Exception:
            return model.key, {}
        cell_popts: dict[int, tuple] = {}
        for k in range(int(valid.size)):
            if not bool(valid[k]):
                continue
            vec = tuple(float(p[k]) for p in params)
            if all(np.isfinite(v) for v in vec):
                cell_popts[k] = vec
        return model.key, cell_popts

    def _apply_panel_analysis(self, card: "PanelCard", *, action: str, selection, extra_values=None,
                              node_params=None) -> None:
        """The ONE create/retarget seam for BOTH analysis actions: publish the panel's region signal,
        then land ``action`` (+ per-action values) on the panel's Analysis row.

        * row exists -> RETARGET IT (running or stopped): update ``row.node.values`` (the persisted
          truth) and, when the node is LIVE, queue ``set_acquisition_parameters`` so the worker
          switches action / model between shots.  A STOPPED row is never deleted and never
          auto-started -- stop means frozen; the status hints at the Logic tab (its lingering
          outputs stay viewable, and Start replays the freshly republished region).
        * no row -> CREATE one Analysis row (the single catalog spec) and Start it.

        The old delete-and-recreate branches (which silently destroyed a user-stopped row, purged
        its lingering signals, and re-started against their intent) are gone: fit<->roi is a
        parameter switch on the SAME node/row/region."""
        from zlc_data.vocabulary import ANALYSIS_SPEC_NAME
        from zlc_data.signal_expr import DEFAULT_SOURCE
        signal = str(card.config.inputs[0]) if card.config.inputs else ""
        if not signal:
            card.set_status("pick a signal in Setting before an analysis", error=True)
            return
        if self._task_locked:
            card.set_status("console is locked by a running task", error=True)
            return
        bound = region_binding(
            card.config.kind, selection,
            structure=card._bound_structure(),
            coordinates=card._selection_coordinates_for_binding(),
            origin=selection.metadata.get("origin", (0.0, 0.0)))
        region_name = self._publish_region(card, bound)
        values = {"action": str(action), "region": region_name, **dict(extra_values or {})}
        verb = "fit" if action == "fit" else "ROI"
        row = self._panel_analysis_row(card)
        if row is not None:
            row.node.values = {**dict(row.node.values or {}), **values}
            editor = self._logic_editors.get(id(row))
            if editor is not None and hasattr(getattr(editor, "form", None), "seed_values"):
                editor.form.seed_values(values)
            node = self._logic_nodes.get(id(row))
            if node is not None:
                node.apply_acquisition_parameters(**dict(node_params or {}), action=str(action))
                card.set_status(f"{verb} -> {row.node.title}", error=False)
            else:
                card.set_status(f"{row.node.title} is stopped — Start it from the Logic tab",
                                error=False)
            self._mark_dirty()
            return
        cfg = LogicNodeConfig(
            kind="processor", name=ANALYSIS_SPEC_NAME,
            title=self._analysis_node_title(card),
            values={"source": {"inputs": [signal], "source": DEFAULT_SOURCE}, **values})
        new_row = self._add_logic_node(cfg, focus=False)   # stay on the Monitor board -- no tab jump
        self._start_logic_node(new_row)
        started = self._logic_nodes.get(id(new_row)) is not None
        card.set_status(f"{verb} -> {cfg.title}" + ("" if started else " (start failed — see Logic tab)"),
                        error=not started)

    def _sync_fit_node(self, card: "PanelCard", request) -> None:
        """The console's ONE fit sink (wired to :attr:`PanelCard.fit_node_sink`): land the panel's fit
        on its per-panel Analysis node, which solves on its WORKER and publishes the model's parameters
        as signals (``fit_x0``/``fit_sigma``/... -- consumable by a Monitor or a scan loss, and read
        back by the panel's DISPLAY-only overlay).  EVERY fit family is a node -- 2-D image centre,
        1-D peak, hist gaussian, AND a facet grid (which the node fits PER CELL and publishes as
        ``(1,1,N)`` param vectors) -- so no fit ever solves on the Qt thread (#6b).  The drawn selection
        travels on the panel's region signal; the config request carries only model + fixed/initial."""
        from zlc_data.curve_fitting import FitRequest
        from dataclasses import replace as _dc_replace
        from zlc_data.plot_region import Selection
        req = FitRequest.from_dict(request) if isinstance(request, Mapping) else request
        payload = _dc_replace(req, selection=Selection()).to_dict()
        # A facet grid fit is the SAME per-panel node, made facet-aware: the node slices with the ONE
        # shared rule (zlc_data.facet.facet_cells) GridPlot displays and fits every cell on its worker, and
        # the panel reconstructs the published per-cell params for DISPLAY (no in-place solve, #6b).
        extra = {"fit_request": payload}
        if card.config.kind == "grid" and card._facet() is not None:
            points_shape, _ = card._facet_value_shapes()
            extra.update(facet=card._facet(), sub_plot_kind=card._resolved_sub_kind(),
                         repeat_mode=card._repeat_mode_value(), points_shape=list(points_shape))
        self._apply_panel_analysis(card, action="fit", selection=req.selection,
                                   extra_values=extra, node_params=extra)

    def _on_panel_selection_clear(self, card: "PanelCard", action: str) -> None:
        """The ONE selection-teardown seam, symmetric for BOTH analyses AND the region itself: an
        explicit CLEAR (Analysis action -> none, fit Clear, Selectors off, panel removal) stops +
        removes the panel's Analysis row and its region control signal whole
        (:meth:`_remove_panel_analysis`).  ``action`` is informational only.  (A re-drag on a STOPPED
        row is NOT a clear -- it retargets in place, see :meth:`_apply_panel_analysis`.)"""
        self._remove_panel_analysis(card)

    def _apply_roi_selection(self, card: "PanelCard", selection) -> None:
        """Turn a drag on ANY plot kind into the panel's ROI analysis.

        The region is bound to the consumed block's own axes through the ONE per-kind resolver
        (:func:`live.region_binding`, the inverse of ``coerce_panel_value``): an image rectangle, a 1-D
        x-range, a distribution count-range and a site-centre rectangle all become a serializable
        Selection whose ``metadata['binding']`` says which axes it spans.  That Selection -- NEVER pixel
        endpoints -- rides the panel's region signal into its per-panel Analysis node
        (:meth:`_apply_panel_analysis`), so two panels on the same source own DISTINCT nodes with
        distinct output names (#3) and the sealed seam holds (``neutral_atom`` never imports frontend)."""
        signal = str(card.config.inputs[0]) if card.config.inputs else ""
        if signal:
            try:
                self.hub.schema(signal)
            except KeyError:
                card.set_status("ROI source has no registered signal schema", error=True)
                return
        reduce = str(card.config.params.get("roi_reduce") or "mean")
        self._apply_panel_analysis(card, action="roi", selection=selection,
                                   extra_values={"reduce": reduce}, node_params={"reduce": reduce})

    def _edit_logic_node(self, row: "LogicNodeRow") -> None:
        """Open (or focus) a logic node's OWN closable Edit tab (param form + Start/Stop)."""
        if self._task_locked:
            return
        existing = self._logic_editors.get(id(row))
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            return
        spec = self._spec_for_logic(row.node)
        editor = LogicNodeEditor(row, self, spec)
        # reflect the live run state on the form (a Started node reopened keeps Stop enabled)
        node = self._logic_nodes.get(id(row))
        editor.set_running(bool(getattr(node, "running", False)))
        self._logic_editors[id(row)] = editor
        self.tabs.add_closable_tab(editor, row.node.title)

    def _start_logic_node(self, row: "LogicNodeRow") -> None:
        """Build the node FROM the node's current param-form values with display
        SUPPRESSED (it only publishes to the hub -- never opens a matplotlib plot),
        register it in ``self.running_nodes``, and ``node.start()`` (data-paced).  Sets
        the node's status dot green; on build/run error -> red + the error on the status
        line.  Reuses the SAME node-build paths the real readout / notebook use."""
        if self._task_locked:
            return                                 # a task owns the console -- no other Start
        editor = self._logic_editors.get(id(row))
        try:
            values = editor.collect_values() if editor is not None else dict(row.node.values)
        except Exception:
            values = dict(row.node.values)
        if editor is not None:
            # The form's collect_values returns only its DECLARED widgets, so a stored value the form
            # does not own (a drag-set ROI ``selection`` is DATA, not a form field) would be dropped on
            # an Edit-triggered restart.  Preserve those: form values WIN for keys it owns, stored fills
            # the rest -- the same "don't lose a param the UI can't show" rule the retarget seam relies on.
            values = {**dict(row.node.values or {}), **values}
        # BUILD-VALIDATE-COMMIT (#A2): build the NEW node FIRST -- a bad param edit must never
        # kill the run that is already going (the old order stopped the running node, then failed
        # the build, leaving nothing running); and nothing is registered until start() succeeds,
        # so a failed start can never leave a half-started ghost in running_nodes / the hub.
        try:
            node = self._build_logic_node(row.node, values)
            self._require_bounded_stop_nodes([node])
        except Exception as exc:
            row.set_state("error", status=f"build failed: {str(exc).splitlines()[0][:80]}")
            if editor is not None:
                editor.set_status(f"build failed: {str(exc).splitlines()[0][:140]}", error=True)
            return
        # VALIDATE: an INDIRECT reactive ring (A consumes B's output while B consumes A's) never
        # computes anything real -- fresh it starves both nodes silently, primed it becomes a
        # cross-thread full-speed republish ping-pong.  The DIRECT self-loop is rejected at the
        # base (Processor.__init__, single source); the ring spans nodes only the board can see,
        # so the graph walk lives here, before commit.
        ring = self._reactive_ring(node)
        if ring:
            path = " -> ".join(ring)
            row.set_state("error", status=f"signal loop: {path}"[:90])
            if editor is not None:
                editor.set_status(
                    f"signal loop rejected: {path} -- a reactive ring never advances "
                    "(each node waits on the other).  Re-pick an upstream source.", error=True)
            return
        row.node.values = dict(values)            # remember for the next Edit reopen + save
        # Label the built node with its ROW TITLE so its provider label MATCHES the declared row's
        # (the DEFAULT camera has prefix="" -> display_label would otherwise fall back to
        # node_label="camera", which differs from the row title "Camera (live frames)", listing
        # `frame` under TWO sources = the "two cameras" bug).  One label per node => one entry in
        # the signal picker (#H3n).
        node.instance_label = str(getattr(row.node, "title", "") or getattr(node, "instance_label", ""))
        # Capture THIS row's PREVIOUS published signals so a source/param change that drops some of
        # them (fewer emCCD events -> fewer frame_i, a different processor source, ...) UNLINKS the now
        # orphan signals from the hub instead of leaving them as stale "(unbound)" picker entries (#5).
        _prev = self._logic_nodes.get(id(row)) or self._last_node.get(id(row))
        old_sigs = set()
        if _prev is not None and hasattr(_prev, "published_signals"):
            try:
                old_sigs = {str(s) for s in _prev.published_signals()}
            except Exception:
                old_sigs = set()
        # The build is good -- NOW stand the previous run of THIS node down (never pile up), and
        # apply the device-OCCUPANCY mutual exclusion over ALL running nodes (#A3): each node
        # declares the hardware INSTANCES it drives (``occupied_devices``, from its own
        # ``_occupies`` attribute names), and two nodes conflict iff those sets intersect by
        # identity.  Only the conflicting nodes stop; everyone on disjoint hardware keeps
        # running -- the monitor camera's live view stays up while the main camera's
        # calibration or measurement starts.  Declared on the NODE, so a notebook-injected
        # ``running_nodes=`` node (which has no row) obeys the SAME rule as a GUI row; a
        # reactive processor occupies nothing and is never stopped.
        if not self._stop_logic_node(row, _silent=True):
            row.set_state("running", status="restart blocked: previous owner still active")
            if editor is not None:
                editor.set_running(True)
                editor.set_status(
                    "restart blocked: previous owner thread did not terminate", error=True
                )
            return
        referenced = tuple(getattr(node, "referenced_devices", lambda: ())())
        fence = self._current_runtime_fence()
        if referenced and fence is None:
            row.set_state(
                "error",
                status="start failed: device-bearing nodes require a runtime authority",
            )
            if editor is not None:
                editor.set_running(False)
                editor.set_status(
                    "start failed: device-bearing nodes require a runtime authority",
                    error=True,
                )
            return
        if not self._timer.isActive():
            self._timer.start()
        try:
            # A restart of THIS logical row keeps its signal names.  Hand the stopped instance's
            # schema-ownership proof to the replacement before its worker can publish: if switching
            # devices/ROI changes data_shape, the replacement's first frame then takes the normal
            # explicit schema-version path (which drops incompatible history) instead of looking like
            # an unrelated producer trying to clobber the lingering signal.  The node base validates
            # same hub + stopped owner + exact current definition; the console never relaxes shape
            # compatibility and never reaches into the hub's schema internals.
            if _prev is not None:
                node._inherit_output_schema_ownership(_prev)
            if fence is None:
                node.start()
            else:
                handle = fence.start(node)
                self._legacy_handles[id(node)] = handle
                self._legacy_handle_fences[id(node)] = fence
        except Exception as exc:
            row.set_state("error", status=f"start failed: {str(exc).splitlines()[0][:80]}")
            if editor is not None:
                editor.set_running(False)
                editor.set_status(f"start failed: {str(exc).splitlines()[0][:140]}", error=True)
            return
        # COMMIT: the node is genuinely running -- only now does it enter the registries, and only
        # now are the previous build's orphan signals unlinked (a failed start leaves the old
        # signals in the hub untouched, exactly like a plain Stop, so nothing is lost).
        # #5: unlink any signal the PREVIOUS build published that this new build no longer does AND no
        # other running node owns -> a switched/rebuilt node leaves NO orphan "(unbound)" signal behind.
        fenced = self._legacy_handles.get(id(node))
        if fenced is None:
            self._logic_nodes[id(row)] = node
            if node not in self.running_nodes:
                self.running_nodes.append(node)
            self._last_node[id(row)] = node       # survives Stop, for signal-source labelling
            self._remove_replaced_orphans(node, old_sigs)
        else:
            # Resource admission is authoritative immediately, but node.start() runs only
            # after the hazard journal is durable on the Run owner.  Delay schema-owner
            # replacement/orphan deletion and task takeover until that start receipt exists.
            self._starting_nodes[id(row)] = node
            self._pending_fenced_starts[id(node)] = set(old_sigs)
            # The ordinary local-journal path crosses this boundary in a few milliseconds.
            # Preserve the console's immediate Start feel with one strictly bounded handoff;
            # a slow/failed journal remains an explicit asynchronous "starting" state and
            # never becomes a provider merely because this budget elapsed.
            try:
                fenced.wait_started(0.05)
            except TimeoutError:
                pass
            except Exception:
                pass
            if fenced.started:
                self._finalize_fenced_start(row, node)
        row.set_state("running", status="starting" if fenced is not None else "running")
        self._update_row_publishes(row)            # now show the LIVE node's published shapes
        if editor is not None:
            editor.set_running(True)
            editor.set_status("starting" if fenced is not None else "running", error=False)
        self.status_dot.set_color(GREEN)
        self._mark_dirty()
        # A TASK (one-shot orchestration) TAKES OVER the console (confocal-style): show
        # its mid-run output in a dedicated Monitor panel + LOCK every other action.
        if fenced is None and getattr(node, "layer", "") == "task":
            self._set_task_running(row, node)

    def _remove_replaced_orphans(self, node, old_sigs: set[str]) -> None:
        try:
            keep = {str(s) for s in node.published_signals()}
        except Exception:
            keep = set()
        for other in self.running_nodes:
            if other is node:
                continue
            try:
                keep.update(str(s) for s in other.published_signals())
            except Exception:
                pass
        orphan = set(old_sigs) - keep
        if orphan:
            self.hub.remove_signals(orphan)

    def _finalize_fenced_start(self, row: "LogicNodeRow", node) -> None:
        pending = self._pending_fenced_starts.pop(id(node), None)
        if pending is None:
            return
        if self._starting_nodes.get(id(row)) is not node:
            raise RuntimeError("pending fenced node lost its row ownership")
        self._starting_nodes.pop(id(row), None)
        self._logic_nodes[id(row)] = node
        if node not in self.running_nodes:
            self.running_nodes.append(node)
        old_sigs = pending
        self._last_node[id(row)] = node
        self._remove_replaced_orphans(node, old_sigs)
        if getattr(node, "layer", "") == "task":
            self._set_task_running(row, node)

    def _task_mid_run_config(self, spec, node, *, title: str) -> "PanelConfig":
        """The task's mid-run panel, DECLARED as a plain ``PanelConfig`` -- so it is built by the SAME
        ``_new_panel_card`` path a manually Added / a saved-and-reloaded panel uses, never a bespoke
        widget.  The task declares only DATA: its spec's ``default_kind`` + ``mid_run_key`` and, for a
        SCANNING task, its ``grid_shape`` (a plain tuple).  The console maps that to a panel whose size
        comes from the ONE :func:`recommended_grid_size` rule (a few cells -> ``2x2``, never a magic
        size) and whose cmap the ONE ``_resolved_cmap`` resolver fills at draw (render == Setting).

        A scanning task (``default_kind='grid'`` + a >=2-D grid_shape) shows a live facet grid over its
        LAST scan axis -- one 2-D map per outer-axis plane, filling in point-by-point (the SAME facet
        machinery a pulse-scan grid panel uses; the panel's ``sub_plot_kind`` auto-derives from the
        remaining axes, so it is not hand-set here).  Anything else is a plain frame panel.  Every task
        panel reads the reserved ``__task_frame__`` the console injects each tick from the task's OWN
        typed TaskOutput (off the hub, #6).  Its SignalSchema reaches the generic panel through
        ``_signal_structure`` exactly like a Hub tensor; no shape is copied into panel params."""
        kind = str(getattr(spec, "default_kind", "2d") or "2d")
        source = f"value = {TASK_FRAME_KEY}"    # the reserved off-hub key (one spelling, TASK_FRAME_KEY)
        gshape = tuple(int(n) for n in (getattr(node, "grid_shape", ()) or ()))
        if kind == "grid" and len(gshape) >= 2:
            facet_axis = len(gshape) - 1                 # facet the LAST scan axis -> one plane per cell
            return PanelConfig(kind="grid", title=title, source=source,
                               size=recommended_grid_size(gshape[facet_axis]),
                               params={"facet": f"points:{facet_axis}"})
        if kind == "grid":
            kind = "1d"                                  # a 0/1-D "scan" is a plain task curve, not a grid
        return PanelConfig(kind=kind, title=title, source=source)

    def _set_task_running(self, row: "LogicNodeRow", node) -> None:
        """Engage task-run mode: open the task's dedicated mid-run panel (declared by
        :meth:`_task_mid_run_config`, built through the generic panel path) and LOCK all other
        controls so the only actions are Stop / wait (#5, confocal task semantics)."""
        spec = self._spec_for_logic(row.node)
        self._task_mid_key = str(getattr(spec, "mid_run_key", DEFAULT_MID_RUN_KEY))
        # Bind the typed source BEFORE constructing the generic PanelCard: construction may ask its
        # structure provider for the declared schema before the task's first numeric publish.
        self._task_output_node = node
        self._task_card_tensor = None
        config = self._task_mid_run_config(spec, node, title=f"Task: {row.node.title}")
        card = self._new_panel_card(config)
        self._attach_card(card)
        self._task_card = card
        self._running_task_row = row
        self._arrange()
        # Assemble the strip's task line BEFORE engaging the lock: _apply_task_lock flips the
        # strip immediately, so the text must already be there (never one tick of stale idle text).
        self._update_task_status_text(node)
        self._apply_task_lock(True)

    def _update_task_status_text(self, node) -> None:
        """Assemble the running task's one-line progress text for the persistent status strip
        (its display -- and its priority against a node error -- is _update_summary's job)."""
        row = self._running_task_row
        if row is None:
            return
        pct = int(round(float(getattr(node.output, "progress", 0.0)) * 100))
        # the task's current STAGE (e.g. "reference frame 23/30", "fitting per-site
        # thresholds") so the operator sees what step the calibration is on, not just %.
        stage = node.output.latest("stage")
        stage_txt = f"  —  {stage}" if stage else ""
        self._task_status_text = (f"⏳  Task running: {row.node.title}  —  {pct}%{stage_txt}  "
                                  "(all other controls locked until it finishes / you Stop it)")

    def _apply_task_lock(self, locked: bool) -> None:
        """Disable / re-enable every mutating control while a task runs.  The strip's Stop
        action stays enabled (it lives outside the locked groups); the strip itself is
        PERSISTENT -- only its content/action flip, so the layout never jumps."""
        self._task_locked = bool(locked)
        if not locked:
            self._task_status_text = None
        self.status_strip.set_action_visible(bool(locked))
        for widget in self._lockable_header:
            widget.setEnabled(not locked)
        self._update_summary()                       # flip the strip's line NOW, not next tick

    def _stop_running_task(self) -> None:
        """Banner 'Stop task' button: cooperatively stop the running task."""
        row = self._running_task_row
        if row is not None:
            self._stop_logic_node(row)

    def _clear_task_running(self, *, timeout: float = 2.0) -> bool:
        """Leave task-run mode (task finished OR stopped): drop the lock + banner and REMOVE the
        transient mid-run panel the task owned.  Finish and Stop take the SAME path (#C) -- a task's
        plot panel is transient (it only showed the work in progress), so it is auto-removed when the
        task ends; the operator's own panels are never touched (only ``_task_card`` is removed)."""
        card = self._task_card
        if card is not None and card in self.cards:
            if self._defer_task_card_teardown:
                if self._deferred_task_card not in (None, card):
                    raise RuntimeError("more than one task card was deferred during shutdown")
                self._deferred_task_card = card
            elif not self._remove_panel(
                card,
                timeout=timeout,
                allow_task_owned=True,
            ):
                return False
        self._running_task_row = None
        self._task_card = None
        self._apply_task_lock(False)
        self._task_card_tensor = None
        self._task_output_node = None
        return True

    def _refresh_task_panel(self) -> None:
        """Pump the running task's mid-run output (from its OWN buffer) into the
        dedicated panel + banner each tick, and leave task-run mode once it finishes."""
        if self._task_card is None:
            return
        node = self._task_output_node
        if node is None:
            return
        try:
            self._task_card_tensor = node.output.latest_tensor(self._task_mid_key)
        except KeyError:
            pass
        self._update_task_status_text(node)
        # The mid-run panel refresh touches its figure on the GUI thread, so it may only run while
        # the render worker is idle (the worker never composes _task_card -- it is not in
        # self.cards -- but the full-hub snapshot + single-card render must not overlap a batch).
        # While busy, simply skip: the task's output buffer keeps the latest frame and the very
        # next idle tick catches up.
        if not self._render.busy:
            self._task_card.refresh(self._expression_namespace())
        # NB: leaving task-run mode on finish is handled in ONE place -- _poll_logic_nodes
        # (the canonical node-lifecycle tick, which runs every tick regardless of whether
        # a mid-run panel exists), so a self-finishing task always releases the lock.

    @staticmethod
    def _repeat_value(values: dict):
        """The acquisition knob from the form -> ``repeat:int`` (#H3n, ONE knob with 0 = ∞).  ``repeat``
        is the depth of the measurement's repeat axis: ``K`` keeps a K-deep block (K passes/frames
        averaged) then STOPS; ``0`` rolls a 1-deep ring forever (a live monitor).  There is NO separate
        free-run toggle -- 0 IS infinite, the SAME semantics as the pulse-scan / scan-repeat count, so
        every measurement reads one number.  Pops the key so it never leaks into the build kwargs."""
        rv = values.pop("repeat", 0)
        try:
            return max(0, int(float(rv)))
        except (TypeError, ValueError):
            return 0

    def _build_logic_node(self, node: LogicNodeConfig, values: dict):
        """Build the node for a logic node, DISPLAY SUPPRESSED (publish-only).

        Reuses the SAME build paths as the real readout / notebook, and asks each SPEC to
        assemble its own live node (so the console never imports a concrete na node class to
        pick one by a metadata string):
          * camera      -> readout.camera_spec().build(hub, ...)
          * measurement -> spec.make_node(hub, prefix=, repeat=, **values)  (the spec's scan tier
                           picks PulseScanNode vs ScannedMeasurementNode)
          * processor   -> spec.make_node(...) (reactive) or spec.make_run(...) (one-shot)
          * task        -> spec.build(hub, **values)
        None of these ever opens a matplotlib plot -- they only publish to the hub."""
        kind = node.kind
        if kind == "camera":
            spec = self._spec_for_logic(node)
            if spec is None:
                raise RuntimeError("no session camera available")
            # Same build path as the notebook: readout.camera_spec().build(hub, repeat=, ...).  ``repeat``
            # = how many recent frames the camera keeps & AVERAGES then STOPS, or 0 = ∞ (roll the latest
            # frame forever).  The camera FILLS that ring and publishes the WHOLE block EVERY frame -- it
            # never averages / suppresses frames at the measurement (that was the lag, #H3l); the PLOT
            # reduces the block via repeat_mode.  Pass ``repeat`` straight INTO build (its ``_build``
            # forwards it to CameraMeasurement, whose ctor set_repeat is the single source) -- the same
            # ONE path the notebook takes, instead of building then re-setting it from a second code path.
            # The signal prefix is the ONE per-instance rule every kind shares (_logic_node_prefix):
            # a lone camera row publishes the short ``frame_i``; a second row that would collide gets
            # its row-title slug.  Which DEVICE the row images is a build parameter (``values``), never
            # part of the signal name -- a consumer binds this instance's output, not a sensor.
            repeat = self._repeat_value(values)                     # pops "repeat" off ``values``
            return spec.build(self.hub, prefix=self._logic_node_prefix(node), repeat=repeat, **values)
        spec = self._spec_for_logic(node)
        if spec is None:
            raise RuntimeError(f"no catalog spec named {node.name!r} for a {kind} node")
        if kind == "measurement":
            # The SPEC owns the assembly (its ProcessorSpec.make_node counterpart): ask it for the live
            # node.  ``repeat`` (re-run the whole scan N times, 0 = ∞) is the MEASUREMENT knob -- pop it
            # and hand the count to make_node; the node only FILLS its raw ``(R,P,*data_shape)`` block,
            # HOW the repeats are displayed is the PLOT's ``repeat_mode`` (#H3l).  The signal prefix is
            # the shared per-instance rule: normally the measurement's slug (spec.key) so every signal
            # is ``<slug>_<quantity>`` (temperature_t_off -- one name, derived), upgraded to the row
            # title's slug only when a SECOND row of the same measurement would collide.
            repeat = self._repeat_value(values)
            return spec.make_node(self.hub, prefix=self._logic_node_prefix(node), repeat=repeat, **values)
        if kind == "processor":
            # REACTIVE processor (the "func" layer): a live node consuming hub signals
            # and republishing derived ones -- e.g. judging occupancy from each frame.
            if getattr(spec, "reactive", False):
                # The SAME per-instance prefix rule (two occupancy judges publish DISTINCT signals, #2).
                built = spec.make_node(self.hub, prefix=self._logic_node_prefix(node), **values)
                built.instance_label = node.title
                return built
            # ONE-SHOT processor: runs once over saved/grabbed frames and self-stops.
            # Hardware goes in ONLY for the ctx device roles the SPEC declares
            # (ProcessorSpec.devices -- the spec owns whether its run drives hardware):
            # a live-grab action borrows the running nodes' device instances; a
            # saved-data action (devices=(), e.g. Readout fidelity over a folder)
            # receives None for both, so it occupies nothing and starting it can never
            # stop an unrelated live node through the device-occupancy exclusion.
            def _borrow(role: str):
                # A ProcessorRun DRIVES its declared roles.  Borrow only an instance the
                # source node itself holds EXCLUSIVE; a lifecycle/wiring record or OBSERVE
                # reference is not a drive capability even if stored as a raw object.
                for n in self.running_nodes:
                    dev = getattr(n, role, None)
                    occupied = tuple(
                        getattr(n, "occupied_devices", lambda: ())()
                    )
                    if dev is not None and any(dev is item for item in occupied):
                        return dev
                return None

            roles = {str(r) for r in (getattr(spec, "devices", ()) or ())}
            readout = getattr(self.session, "readout", None)
            # Symmetric with the reactive branch above: the SPEC builds its own node.
            return spec.make_run(self.hub, readout=readout,
                                 camera=_borrow("camera") if "camera" in roles else None,
                                 sequencer=_borrow("sequencer") if "sequencer" in roles else None,
                                 params=values)
        if kind == "task":
            return spec.build(self.hub, **values)
        raise RuntimeError(f"unknown logic kind {kind!r}")

    def _render_barrier(self, timeout: float = 5.0) -> bool:
        """Own every Figure, or say NO -- the one door to the render island.

        The bridge raises when the handoff does not settle; the panels were handed a
        bool-returning callable long before the bridge existed, so the exception is
        turned back into False HERE rather than at nine call sites.  What changes is
        what the console does with a False: it now aborts the operation instead of
        proceeding to touch a Figure the render worker may still own.
        """

        from zlc_workbench.legacy import LegacyHandoffTimeout

        try:
            self._render.settle(timeout)
        except LegacyHandoffTimeout:
            return False
        except BaseException:                      # bridge poisoned/closed, owner-thread misuse
            return False
        return True

    def _render_handoff_failed(self, what: str) -> None:
        """Tell the operator WHICH action was refused, on the permanent status strip."""

        try:
            self.status_strip.show_message(
                f"{what} cancelled: the render worker did not hand back the figures in time.",
                severity="error")
        except BaseException:
            pass

    @staticmethod
    def _supports_bounded_stop(node) -> bool:
        """Whether ``node.stop(timeout=...)`` is a callable, bounded ownership handoff."""

        stop = getattr(node, "stop", None)
        if not callable(stop):
            return False
        try:
            parameters = inspect.signature(stop).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == "timeout"
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
            for parameter in parameters
        )

    @classmethod
    def _require_bounded_stop_nodes(cls, nodes: Sequence[object]) -> None:
        invalid = [node for node in nodes if not cls._supports_bounded_stop(node)]
        if invalid:
            names = ", ".join(type(node).__name__ for node in invalid)
            raise TypeError(
                "running nodes must implement stop(*, timeout=...) with confirmed termination; "
                f"unbounded nodes: {names}"
            )

    def _stop_node_confirmed(self, node, *, timeout: float = 2.0) -> bool:
        """Request stop without ever translating an unknown join into success."""

        fence = self._legacy_handle_fences.get(id(node)) or self._current_runtime_fence()
        if fence is not None:
            handle = self._legacy_handles.get(id(node)) or fence.handle_for(node)
            if handle is not None:
                if handle.snapshot().state.terminal:
                    return True
                receipt = fence.stop(node, timeout=max(0.0, float(timeout)))
                return bool(receipt.terminated)
            # A production console never adopts an already-running raw node.  The
            # launcher has one explicit enrollment boundary that first calls the
            # unmanaged helper below and then restarts through the fence.
            return not bool(getattr(node, "running", False))
        return self._stop_unmanaged_node_confirmed(node, timeout=timeout)

    def _stop_unmanaged_node_confirmed(self, node, *, timeout: float = 2.0) -> bool:
        """One-time migration enrollment helper; never used after fenced start."""

        if not self._supports_bounded_stop(node):
            return False
        budget = max(0.0, float(timeout))
        key = id(node)
        attempt = self._stop_attempts.get(key)
        if attempt is None or attempt.node is not node:
            attempt = _StopAttempt(node, budget)
            self._stop_attempts[key] = attempt
            attempt.start()
        attempt.thread.join(budget)
        if attempt.thread.is_alive():
            return False
        self._stop_attempts.pop(key, None)
        if attempt.error is not None or attempt.result is False:
            return False
        try:
            return not bool(node.running)
        except BaseException:
            return attempt.result is True

    def _stop_logic_node(
        self,
        row: "LogicNodeRow",
        *,
        _silent: bool = False,
        timeout: float = 2.0,
    ) -> bool:
        """Stop a logic node's node (``node.stop()``) and grey its dot."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        node = self._logic_nodes.get(id(row)) or self._starting_nodes.get(id(row))
        if node is not None:
            if not self._stop_node_confirmed(
                node, timeout=max(0.0, deadline - time.monotonic())
            ):
                editor = self._logic_editors.get(id(row))
                if not _silent:
                    row.set_state("running", status="stop pending: owner thread still active")
                    if editor is not None:
                        editor.set_running(True)
                        editor.set_status(
                            "stop pending: owner thread still active", error=True
                        )
                return False
            if node in self.running_nodes:
                self.running_nodes.remove(node)
            self._legacy_handles.pop(id(node), None)
            self._legacy_handle_fences.pop(id(node), None)
            self._pending_fenced_starts.pop(id(node), None)
            self._starting_nodes.pop(id(row), None)
        # A task's transient Figure is part of the same ownership handoff.  Ordinary Stop/finish
        # waits for a render barrier; whole-console shutdown explicitly defers it until the central
        # RenderLoop join.  A failed barrier keeps the task row/card references so the next tick or
        # close retry can finish the handoff instead of destroying a Figure still in use.
        if row is self._running_task_row and not self._clear_task_running(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            row.set_state("stopped", status="panel teardown pending: render worker still active")
            editor = self._logic_editors.get(id(row))
            if editor is not None:
                editor.set_running(False)
                editor.set_status(
                    "panel teardown pending: render worker still owns the Figure", error=True
                )
            return False
        self._logic_nodes[id(row)] = None
        editor = self._logic_editors.get(id(row))
        if not _silent:
            row.set_state("stopped", status="stopped")
            self._update_row_publishes(row)        # back to the spec-declared outputs
            if editor is not None:
                editor.set_running(False)
                editor.set_status("stopped", error=False)
        return True

    def _remove_logic_node(self, row: "LogicNodeRow", *, _rebuild: bool = True) -> bool:
        """Stop + remove a logic node (its node is stopped, its row + Edit drop)."""
        # a running task locks the console: the per-row Remove button must no-op too
        # (every other mutating entry guards on this).  Internal teardown (load_state)
        # passes _rebuild=False and is never reached while locked.
        if self._task_locked and _rebuild:
            return False
        # Capture this node's published signals so REMOVE can PURGE them from the hub -- a removed
        # node's signals are stale and must leave, else they pile up run-after-run as "多余 signal" in
        # every picker (#2).  STOPPING keeps them (a finished scan stays plottable / a panel can be
        # wired before the next run); only REMOVING purges.  Use ``_last_node`` (the last built node,
        # retained THROUGH stop) not ``_logic_nodes`` (None'd on stop): the common flow is STOP-then-
        # REMOVE, where the live ref is already gone but the lingering hub signals are precisely the
        # ones to purge.
        gone = (
            self._logic_nodes.get(id(row))
            or self._starting_nodes.get(id(row))
            or self._last_node.get(id(row))
        )
        gone_sigs: set[str] = set()
        if gone is not None and hasattr(gone, "published_signals"):
            try:
                gone_sigs = {str(s) for s in gone.published_signals()}
            except Exception:
                gone_sigs = set()
        if not self._stop_logic_node(row, _silent=True):
            return False
        editor = self._logic_editors.pop(id(row), None)
        if editor is not None:
            idx = self.tabs.indexOf(editor)
            if idx >= 0:
                self.tabs.removeTab(idx)
            editor.teardown()
            editor.setParent(None)
            editor.deleteLater()
        self._logic_nodes.pop(id(row), None)
        self._starting_nodes.pop(id(row), None)
        self._last_node.pop(id(row), None)
        # (No per-panel handle to drop: the panel<->analysis association is DERIVED from the row's
        # values['region'] vs the panel's persisted region_signal -- removing the row derives None,
        # and the next drag creates a fresh row consuming the SAME persisted region name.)
        if row in self.logic_nodes:
            self.logic_nodes.remove(row)
        self.logic_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        if not self.logic_nodes:
            self.logic_hint.show()
        # PURGE the removed node's signals that NO OTHER live node still owns (a shared/prefixed name
        # another running node also publishes is kept), so the hub sheds the stale names instead of
        # accumulating them across runs.
        if gone_sigs:
            keep: set[str] = set()
            for other in self.running_nodes:
                try:
                    keep.update(str(s) for s in other.published_signals())
                except Exception:
                    pass
            purge = gone_sigs - keep
            if purge:
                self.hub.remove_signals(purge)
        if _rebuild:
            self._mark_dirty()
        return True

    def _poll_logic_nodes(self) -> None:
        """Each tick: reflect each running node's state on its row + Edit (a
        one-shot that finished -> stopped; a node that errored -> red)."""
        for row in self.logic_nodes:
            node = self._logic_nodes.get(id(row)) or self._starting_nodes.get(id(row))
            if node is None:
                continue
            editor = self._logic_editors.get(id(row))
            fenced = self._legacy_handles.get(id(node))
            if fenced is not None:
                snapshot = fenced.snapshot()
                if not fenced.started and not snapshot.state.terminal:
                    blocked = snapshot.recovery_instruction
                    status = f"start blocked: {blocked}" if blocked else "starting"
                    # The Run still owns claims and can still be cancelled/recovered, so
                    # keep Stop enabled even when the journal failure is shown as an error.
                    row.set_state("running", status=status[:90])
                    if editor is not None:
                        editor.set_running(True)
                        editor.set_status(status[:140], error=bool(blocked))
                    continue
                if fenced.started:
                    self._finalize_fenced_start(row, node)
                else:
                    # Terminal-before-start keeps the previous stopped provider and
                    # its lingering signals exactly as an ordinary failed start does.
                    self._pending_fenced_starts.pop(id(node), None)
                    self._starting_nodes.pop(id(row), None)
                if snapshot.state.name == "CANCELLING":
                    row.set_state("running", status="stop pending: cleanup/safety not complete")
                    if editor is not None:
                        editor.set_running(True)
                        editor.set_status(
                            "stop pending: cleanup/safety not complete", error=True
                        )
                    continue
                if snapshot.state.terminal:
                    if snapshot.state.name == "FAILED":
                        message = snapshot.primary_error or "runtime cleanup failed"
                        row.set_state("error", status=f"error: {message[:60]}")
                        if editor is not None:
                            editor.set_running(False)
                            editor.set_status(f"error: {message}", error=True)
                    elif snapshot.state.name == "SUCCEEDED":
                        row.set_state("stopped", status="done")
                        if editor is not None:
                            editor.set_running(False)
                            editor.set_status("done", error=False)
                    else:
                        row.set_state("stopped", status="stopped")
                        if editor is not None:
                            editor.set_running(False)
                            editor.set_status("stopped", error=False)
                    self._stop_logic_node(row, _silent=True, timeout=0.0)
                    continue
            error = getattr(node, "last_error", None)
            running = bool(getattr(node, "running", False))
            finished = bool(getattr(node, "finished", False))
            # A one-shot node is TERMINATED once its loop has stopped AND it has either
            # finished (ran once) or errored -- both outcomes end the node, so both must
            # release the console lock.  A run() that raises sets finished=True in a
            # ``finally`` (logic.py), so a failed Task lands here too; binding the unlock
            # to error as well covers any node that surfaces an error WITHOUT finishing.
            terminated = (not running) and (finished or bool(error))
            if error:
                row.set_state("error", status=f"error: {str(error)[:60]}")
                if editor is not None:
                    editor.set_running(False)
                    editor.set_status(f"error: {error}", error=True)
            elif finished and not running:
                row.set_state("stopped", status="done")
                if editor is not None:
                    editor.set_running(False)
                    editor.set_status("done", error=False)
            elif running:
                # surface scan progress when the node reports it -- ``total_points`` spans ALL
                # repeat passes (n_points x repeat), so a repeated scan counts 52/60, not 12/20.
                done = getattr(node, "points_done", None)
                total = getattr(node, "total_points", None) or getattr(node, "n_points", None)
                if done is not None and total:
                    row.set_state("running", status=f"running {done}/{total}")
                else:
                    row.set_state("running", status="running")
                # refresh the published shapes as the node's first values land (a reactive
                # processor publishes nothing until it consumes a frame); set_publishes is
                # self-guarded so this is a no-op once the shapes stop changing.
                self._update_row_publishes(row)
                # (The "xN" repeat tag is computed by each plot panel itself while it reduces the raw
                # canonical (R,P,*data_shape) block per its repeat_mode -- the console no longer pushes a
                # repeat counter onto plotters, since the panel is decoupled and owns the reduction.)
            # A node that ENDS ON ITS OWN -- a task finishing / erroring, a finite measurement
            # taking its last repeat -- must reach the SAME terminal state the Stop button
            # produces: ``_stop_logic_node`` is the ONE lifecycle endpoint (drop from
            # ``running_nodes`` + null the live ref + release the task lock).  Leaving the dead
            # node in ``running_nodes`` breaks the documented shot-clock invariant ("a lingering
            # signal of a STOPPED node is excluded" -- _display_shot) and lets a finished node
            # keep winning provider/structure resolution over live ones.  ``_silent`` keeps the
            # row/editor text set above (done / error), which is richer than Stop's "stopped".
            if terminated:
                self._stop_logic_node(row, _silent=True)

    def _mark_dirty(self, *_args) -> None:
        if self._building:
            return
        if self.save_button is not None:            # embedded: no layout Save button to flag dirty
            self.save_button.set_dirty(True, dirty_color=YELLOW)
        self._update_summary()

    # ------------------------------------------------------------------ refresh
    def _display_shot(self) -> int | None:
        """The ONE coherent display shot for the WHOLE board this tick (the global shot clock): the newest
        source-shot that EVERY displayed, still-running-producer signal has reached -- the ``min`` of their
        latest provenance ids.  :meth:`_expression_namespace` reads :meth:`SignalHub.snapshot_at` at this
        shot, so every panel draws ONE physical shot and they can never stagger: two 2-D images on ``frame_0``
        / ``frame_2`` and a sitemap fed by ``frame_1``->occupancy all show the same clock cycle; a fast camera
        frame is held back to match the slower reactive occupancy (the user's hard requirement -- coherence
        over freshness, "同一个clock周期，不要错位").

        Crucially this holds back ONLY relative to OTHER co-displayed producers: a LONE camera-repeat panel's
        bound set is just ``frame_0``, so its ``min`` is its OWN latest -> it stays fully live (repeat /
        repeat_mode never stutter).  The hold appears exactly when a slower producer is shown ALONGSIDE it.

        Scope = signals that are BOTH bound by a live panel AND published by a node STILL RUNNING: a lingering
        signal of a STOPPED node is excluded (it would freeze the clock at its last shot), and a free-running
        :data:`NO_LINEAGE` scalar (a loading rate) never constrains.  ``None`` (-> latest of each, i.e.
        :meth:`snapshot_latest`) when nothing displayed carries a lineage yet."""
        from zlc_data.vocabulary import NO_LINEAGE             # single source of the sentinel
        live: set[str] = set()
        for node in self.running_nodes:
            # Exclude a FAULTED node's straggling signals from the min, the SAME principle as the
            # STOPPED-node exclusion above: a running-but-erroring analysis node (its provenance frozen
            # at its last good shot) would otherwise pin the whole board's clock and freeze every panel
            # (#4-C).  A slow-but-HEALTHY node (consecutive_errors == 0) is kept, so coherence still
            # holds it back -- only a dead one is dropped.
            if int(getattr(node, "consecutive_errors", 0) or 0) > 0:
                continue
            try:
                live |= {str(s) for s in node.published_signals()}
            except Exception:
                continue
        if not live:
            return None
        shots: list[int] = []
        for card in self.cards:
            for name in self._card_reads(card):
                if name in live:
                    p = self.hub.latest_provenance(name)
                    if p != NO_LINEAGE:
                        shots.append(int(p))
        return min(shots) if shots else None

    def _expression_namespace(self, disp="__current__") -> dict[str, object]:
        """The board's shared shot-coherent GUI namespace.  ``disp`` pins the display shot (the
        render thread passes the one its batch was scheduled against); by default it is computed
        fresh (the single-card refresh / test paths)."""
        if disp == "__current__":
            disp = self._display_shot()
        return self._expression_namespace_at(disp)

    def _expression_namespace_at(self, disp) -> dict[str, object]:
        # Shot-COHERENT read at the board's global display shot (#shot-clock): every signal resolves to its
        # value AT that shot, so a frame_0 2-D, a frame_2 2-D and a frame_1->occupancy sitemap can never show
        # different shots -- the faster camera is held back to the slower co-displayed producer.  A signal
        # with no sample at that shot (a free-running scalar, or a producer not yet there) falls back to its
        # latest, never blanking a panel.  display_shot None -> latest of each (snapshot_latest).  A LONE
        # fast panel is NOT held back: its own signal is the min (see _display_shot).  The off-hub task
        # tensor is appended below with the same canonical data + validity interface.
        # The helpers (np/numpy/math/history/latest/names/shot) come from the ONE signal_expr builder
        # layered on this view's snapshot -- a panel expression and a node-side expression (pulse-scan
        # y, processor source) can never diverge in capability (GUI == node).
        from zlc_data.signal_expr import hub_namespace
        from zlc_data.signal_tensor import SignalHistoryGap
        try:
            tensors = self.hub.snapshot_at(disp, tensors=True)
        except SignalHistoryGap:
            # A lagging DERIVED signal (a slow per-panel worker fit/roi) can drag ``disp`` below a fast
            # frame's bounded history, so no coherent state exists at ``disp`` for that one signal.  Fall
            # back to its LATEST for THAT signal only -- the documented "a producer not yet there falls
            # back to its latest, never blanking a panel" behavior -- keeping every other signal
            # shot-coherent.  Per-signal so one straggler can't blank the whole board.  Only reached when
            # a gap actually occurs (a slow worker node behind the display shot), never on the happy path.
            tensors = {}
            for name in self.hub.names():
                try:
                    tensors.update(self.hub.snapshot_at(disp, tensors=True, names=[name]))
                except SignalHistoryGap:
                    tensors.update(self.hub.snapshot_at(None, tensors=True, names=[name]))
        namespace = hub_namespace(self.hub, {name: tensor.data for name, tensor in tensors.items()})
        valid = {name: tensor.valid for name, tensor in tensors.items()}
        task_tensor = self._task_card_tensor
        namespace[TASK_FRAME_KEY] = task_tensor.data if task_tensor is not None else None
        if task_tensor is not None:
            valid[TASK_FRAME_KEY] = task_tensor.valid
        namespace[SIG_VALID_KEY] = valid
        # Per-signal publish counters (reserved key) so a rolling monitor can tell
        # a new sample of its own source from an unrelated node's version bump.
        versions = self.hub.signal_versions()
        if self._task_output_node is not None:
            versions[TASK_FRAME_KEY] = int(self._task_output_node.output.version)
        namespace[SIG_VERSIONS_KEY] = versions
        # Coordinate frames (reserved key): {signal_name: [x, w, y, h]} from any
        # node whose acquisition source declares a ROI.  A 2D panel reads its
        # source signal's frame so the image axes are the REAL camera pixel
        # coordinates (ROI), not 0..N -- and an area-select maps back to the ROI.
        namespace[COORD_FRAMES_KEY] = self._coord_frames()
        return namespace

    def _coord_frames(self) -> dict[str, list]:
        """Map each node-published signal to its source's spatial ``region``
        endpoints ``[x_min, x_max, y_min, y_max]`` (the acquisition-layer format)
        when the node declares one (a camera frame), so a panel can put its axes in
        real source pixels.  A panel reads index 0 (x_min) and index 2 (y_min) as
        the axis origin, so this is robust to endpoints vs any position+size form.

        Reads :meth:`_provider_nodes` -- the ONE source of "which nodes count" (running nodes first,
        then each Logic row's last build kept past Stop) -- with ``setdefault`` so a RUNNING provider
        wins, exactly the first-match semantics :meth:`_node_for_signal` uses.  The coordinate frame is
        a property of the signal's producing node, NOT of its run state: a STOPPED camera's lingering
        ``frame_0`` keeps its real pixel frame, so a bound 2-D panel's ``_needs_structural_build``
        region comparison stays constant across stop/start and the panel is NEVER rebuilt (zoom /
        selector geometry / clim all survive).  The old running-only scan flipped the frame to None on
        Stop and back on Start -- two spurious full rebuilds that wiped the view (#3)."""
        frames: dict[str, list] = {}
        seen: set[int] = set()
        for node in self._provider_nodes():
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            try:
                region = node.acquisition_parameters().get("region")
            except Exception:
                region = None
            if not region:
                continue
            try:
                sigs = node.published_signals() if hasattr(node, "published_signals") else ()
            except Exception:
                sigs = ()
            for s in sigs:
                frames.setdefault(str(s), list(region))
        return frames

    def _recompute_tick_interval(self) -> None:
        """Re-base the shared timer to the SMALLEST panel ``update_ms`` (which divides every
        other rate in :data:`UPDATE_INTERVALS`, so the rates co-align) and reset the tick
        phase.  Called on add / remove / per-panel rate change, so the timer ticks no faster
        than the fastest panel actually needs."""
        base = min((c.config.update_ms for c in self.cards), default=DEFAULT_UPDATE_MS)
        self._base_interval_ms = max(min(UPDATE_INTERVALS), int(base))
        self._tick_count = 0                      # re-sync the phase to the new base
        if getattr(self, "_timer", None) is not None:
            self._timer.setInterval(self._base_interval_ms)

    def _toggle_pause(self) -> None:
        """Pause / Resume the WHOLE monitor: freeze or unfreeze every plot's live display at once.
        This is a DISPLAY freeze -- acquisition keeps running (Stop a Logic node to halt that)."""
        self._paused = not self._paused
        self.pause_button.setText("Resume" if self._paused else "Pause")
        self.pause_button.set_color(GREEN if self._paused else ORANGE)
        # A paused display freezes every panel: _tick returns early while paused (no card advances its
        # _render_version), and the LAST live tick already presented ONE coherent shot (the two-phase
        # compose-all-then-present-all render), so the frozen board is a single consistent shot.  On
        # Resume every stale panel's frame key differs from what it drew, so each recomposes on its
        # next update_ms beat (same-beat panels catch up together at the shared coherent shot).
        if not self._paused:
            self._tick()

    def _toggle_selectors(self, on: bool) -> None:
        """Header "Selectors" switch: arm (ON) or park (OFF) the selector layer of EVERY dashboard
        panel in place (``PanelCard.set_selectors_enabled`` -> ``BaseLivePlot.set_selectors_active``)
        -- a pure display gate, no rebuild, no effect on acquisition or the Edit tabs."""
        for card in self.cards:
            card.set_selectors_enabled(bool(on))

    def _open_device_viewer(self) -> None:
        """Header "Devices" button: open the READ-ONLY device viewer (one tab per loaded device --
        its snapshot + live runtime read-backs, NO config editing / add / remove).  Routes through
        the session's ONE-per-session ``device_viewer()`` facade (``na._gui.open_device_viewer`` ->
        ``show_device_viewer``): the console offers only a safe look at the running device set, never
        the full config editor (that is the notebook's ``exp.device_manager()`` entry) -- so a
        running experiment's devices can never be mutated from here."""
        session = getattr(self, "session", None)
        opener = getattr(session, "device_viewer", None)
        if callable(opener):
            opener()

    def _save_board_image(self) -> None:
        """Save the WHOLE monitor board (every panel, composited in its laid-out position) to a PNG.
        Unlike a per-plot save (data png+npz), this is a raster of the board region, so it is a plain
        image.  Reuses ``_last_save_dir`` so it shares the per-panel save's remembered folder."""
        default_dir = self._last_save_dir or str(_task_files_dir())
        default = str(Path(default_dir) / time.strftime("monitor_%Y_%m_%d_%H_%M_%S.png", time.localtime()))
        path, _sel = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save monitor image", default, "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        # Grab ONLY the MINIMAL rectangle that bounds the laid-out cards (union of their geometries +
        # one GAP margin), NOT the whole board widget -- the board is sized to fill the scroll viewport
        # (setWidgetResizable), so grabbing it would bake in a big empty plot area below/right of the
        # panels.  Clamp to the board so the sub-rect is valid; empty board -> the whole (tiny) board.
        # board.grab() forces a synchronous paint of EVERY canvas -- settle the render loop first
        # (the file dialog above kept ticks running, so a batch may be in flight right now).
        if not self._render_barrier():
            self._render_handoff_failed("Save image")
            return
        cards = list(self.cards)
        if cards:
            rect = cards[0].geometry()
            for card in cards[1:]:
                rect = rect.united(card.geometry())
            rect = rect.adjusted(-GAP, -GAP, GAP, GAP).intersected(self.board.rect())
            pm = self.board.grab(rect)
        else:
            pm = self.board.grab()
        # _PanelBoard is transparent -> composite onto an opaque white canvas so the PNG isn't see-through.
        _opaque_white_composite(pm).save(path)
        self._last_save_dir = str(Path(path).parent)
        self._update_summary()

    def _panel_frame_key(self, card, disp, sigvers: Mapping[str, int]):
        """The identity of the COHERENT FRAME this panel would draw right now.  It is the board display
        shot ``disp`` -- which pins EVERY lineage / shot-clock signal (that signal's value AT that shot is
        fixed) -- plus the publish counters of the panel's FREE-RUNNING (no-lineage) inputs (a loading
        rate, which ``disp`` does not constrain).  The panel's buffer is stale exactly when this changes:
        when the coherent clock ADVANCES -- so ALL panels go stale together and recompose from the ONE
        shared snapshot, and the three emCCD frames of a pulse can NEVER freeze on different shots -- or
        when one of the panel's own free-running scalars ticks.  ``sigvers`` is one shared
        ``hub.signal_versions()`` snapshot.

        (Successor of the per-signal gate, whose flaw was ignoring ``disp``: a panel whose OWN signal had
        already published stayed frozen a shot behind while a slower co-displayed producer advanced the
        clock -> the reported "the three frames are not one shot" desync.  Blit makes recomposing every
        panel on a clock tick cheap, so coherence costs nothing.)"""
        from zlc_data.vocabulary import NO_LINEAGE             # single source of the sentinel
        inputs = card.config.inputs or ()
        free = tuple((str(n), sigvers.get(str(n), 0)) for n in inputs
                     if self.hub.latest_provenance(str(n)) == NO_LINEAGE)
        return (disp, free)

    def _tick(self) -> None:
        # poll the logic nodes EVERY base tick (even when no new signal arrived) so a
        # one-shot node's run-complete / error transition is never missed if the node
        # self-stops between version bumps.
        self._poll_logic_nodes()
        self._refresh_signal_info()   # cheap + self-guarded: tracks source/node changes
        # A running task's mid-run output is OFF the hub (#6), so it does NOT bump the
        # hub version -- refresh its dedicated panel here every tick.
        self._refresh_task_panel()
        # PAUSE = freeze the board: skip the render below (no card advances its _render_version) while
        # node polling / banners stay alive.  On Resume the coherent clock has moved on, so every stale
        # panel's frame key differs from what it drew and the gate below recomposes them together.
        if self._paused:
            self._update_summary()
            return
        self._tick_count += 1
        # A render batch in flight (composing, or finished but its GUI pass still queued) blocks this
        # tick's submission -- this refusal IS the frame-skip back-pressure (no budget, no rotor: the
        # GUI thread no longer executes any compose, so there is nothing to ration on it).  But a beat
        # that falls on a busy tick is OWED (``card._beat_owed``): the next idle tick serves it
        # regardless of the modulo.  Without that, submissions phase-lock -- a heavy fast-beat panel
        # occupies exactly the ticks a slow-beat panel's beats land on, starving it FOREVER under
        # sustained overload (the fairness the deleted rotor used to provide).
        busy = self._render.busy
        disp = self._display_shot()            # the ONE coherent display shot for the whole board this tick
        sigvers = self.hub.signal_versions()   # ONE per-signal-counter snapshot for the whole tick
        elapsed = self._tick_count * self._base_interval_ms
        # COLLECT every panel whose coherent frame changed and whose beat is due (or owed).  The panel's
        # OWN beat (update_ms) gates WHEN it recomposes; the coherent clock decides WHAT it shows (the
        # shared snapshot at ``disp``).  Panels on the same beat flip to the same shot in the same
        # batch (the three emCCD frames of a pulse never split); a panel on a slower beat holds its
        # previous coherent shot until its beat comes round -- that is what choosing a slower update
        # interval MEANS.  A mid-drag panel is skipped whole (a live recompose under a drag stomps
        # the widget blit backgrounds); its beat is owed too, so it catches up right on release.
        batch = []
        for card in self.cards:
            key = self._panel_frame_key(card, disp, sigvers)
            if key == card._render_version:
                continue                       # nothing this panel shows changed
            if elapsed % card.config.update_ms != 0 and not card._beat_owed:
                continue                       # beat not due (and none owed)
            fig = getattr(card.plotter, "fig", None)
            if busy or (fig is not None and getattr(fig, "_zlc_interacting", False)):
                card._beat_owed = True         # due but unservable this tick -> next idle tick serves it
                continue
            card._beat_owed = False
            batch.append((card, key))
        if batch:
            # The worker builds the shared namespace ITSELF (the full-hub snapshot copy is real
            # work -- off the GUI thread with everything else) and composes each panel in place;
            # panels needing a STRUCTURAL build come back untouched for the GUI pass.
            self._render.submit(
                lambda b=batch, d=disp: self._compose_batch(b, d))
        # keep the visible Edit tab's 'now:' acquisition references live, so a queued
        # parameter edit shows as applied once the loop picks it up.
        editor = self.tabs.currentWidget()
        if isinstance(editor, PanelEditor):
            editor.refresh_node_now_labels()
            editor.refresh_limit_hints()  # #3a: tick updates only the grey placeholder hint, never the text
        self._update_summary()

    def _compose_batch(self, batch, disp):
        """RENDER THREAD: build the ONE shared shot-coherent namespace and compose every batched
        panel into its offscreen Agg buffer.  Returns the per-panel outcome for the GUI pass."""
        namespace = self._expression_namespace(disp)
        composed, structural = [], []
        for card, key in batch:
            try:
                done = card.compose(namespace, offthread=True)
            except Exception:
                done = True            # compose() latches errors itself; never kill the batch
            (composed if done else structural).append((card, key))
        return {"namespace": namespace, "composed": composed, "structural": structural}

    def _on_render_batch(self, result) -> None:
        """GUI THREAD (queued from the render thread): finish the batch -- run the structural
        builds the worker handed back, then PRESENT every panel TOGETHER so the board flips ONE
        coherent shot (never frame_0 at shot S beside frame_2 at S-1)."""
        if not isinstance(result, dict):
            return                     # a job that raised whole (already latched per panel) -- drop
        namespace = result["namespace"]
        # Membership gate FIRST: a card removed (or shut down) while the batch was in flight must
        # not get a structural compose / status flush either -- composing into a torn-down card
        # resurrects Qt widgets its removal already deleted.
        alive = lambda c: c in self.cards or c is self._task_card
        structural = [(c, k) for c, k in result["structural"] if alive(c)]
        composed = [(c, k) for c, k in result["composed"] if alive(c)]
        for card, _key in structural:
            try:
                card.compose(namespace)          # the full GUI compose: builds plotter + canvas
            except Exception:
                pass                             # compose latches its own status
        for card, _key in composed:
            card.flush_deferred_status()
        for card, key in structural + composed:
            card._render_version = key
            card.present()
        self._update_fit_overlays()   # push each fit panel's node params to its DISPLAY-only overlay (#6)

    def refresh_once(self) -> None:
        """Synchronous FULL refresh (tests / notebooks / Resume catch-up): COMPOSE every card from the
        ONE coherent snapshot, then PRESENT them all together -- the same two-phase coherence as the live
        tick, just unconditional (ignores per-panel beat and the frame-key gate).  Holds the render
        barrier first: this GUI-thread compose must own every figure."""
        if not self._render_barrier():
            self._render_handoff_failed("Refresh")
            return
        self._poll_logic_nodes()
        self._refresh_signal_info()
        self._refresh_task_panel()
        disp = self._display_shot()
        sigvers = self.hub.signal_versions()
        # compose EVERY card at the one coherent snapshot, THEN present them together (tests / notebooks /
        # Resume catch-up) so a frozen board is a single consistent shot, never a torn mix.
        namespace = self._expression_namespace()
        for card in self.cards:
            card.compose(namespace)
            card._render_version = self._panel_frame_key(card, disp, sigvers)
        for card in self.cards:
            card.present()
        self._update_fit_overlays()   # push each fit panel's node params to its DISPLAY-only overlay (#6)
        editor = self.tabs.currentWidget()
        if isinstance(editor, PanelEditor):
            editor.refresh_node_now_labels()
            editor.refresh_limit_hints()  # #3a: tick updates only the grey placeholder hint, never the text
        self._update_summary()

    def _note_display_drops(self) -> int:
        """Shots the hub's bounded ring dropped since the last banner refresh because acquisition -- which
        is deliberately never rate-capped -- outran the display.  Returns the count dropped THIS refresh
        (0 = the display is keeping up).  Acquisition is never throttled or affected; this only reports lost
        DISPLAY frames, for the amber heads-up.

        Measured CONSUMER-side, which is the only sound place: the hub silently rotates its ring on every
        long run, so a hub-side drop counter would fire constantly and mean nothing.  What matters is how
        many shots were published SINCE THIS display last read, versus the ring depth -- if that exceeds the
        ring, the oldest of them rolled off before any rolling-history panel could read them."""
        shot = self.hub.shot
        last = getattr(self, "_last_seen_shot", None)
        self._last_seen_shot = shot
        if last is None:                       # first call: establish the baseline, report nothing
            return 0
        overrun = (shot - last) - self.hub.history_len
        return overrun if overrun > 0 else 0

    def _update_summary(self) -> None:
        try:
            n_signals = len(self.hub.names())
        except Exception:
            n_signals = 0
        telemetry = f"{len(self.cards)} panels | {n_signals} signals | shot {self.hub.shot}"
        if telemetry != getattr(self, "_summary_text", None):
            self._summary_text = telemetry
            self.summary.setText(telemetry)
        dropped = self._note_display_drops()
        # The persistent strip's ONE priority ladder (every tick): a wedged node must never fail
        # silently, so a red node error outranks even the running task's progress line; the
        # display-behind advisory is amber (the RUN is unaffected -- acquisition is unthrottled
        # and always shows latest); idle shows the board summary.  The strip itself change-gates
        # text/style, so this per-tick call never repolishes an unchanged state.
        faulted = [n for n in self.running_nodes if getattr(n, "last_error", None)]
        if faulted:
            node = faulted[0]
            who = (getattr(node, "prefix", "") or self._node_label(node) or "node").rstrip("_:")
            n = int(getattr(node, "consecutive_errors", 1))
            self.status_strip.show_message(
                f"⚠ NODE ERROR ({who}, ×{n}): {node.last_error}"[:300], severity="error")
        elif getattr(self, "_task_status_text", None):
            self.status_strip.show_message(self._task_status_text, severity="task")
        elif dropped:
            self.status_strip.show_message(
                f"⚠ display behind: dropped {dropped} shot(s) faster than the "
                f"{self.hub.history_len}-deep buffer -- acquisition unaffected", severity="warning")
        else:
            # Idle telemetry already lives in the header.  Keep the status surface empty (but at its
            # fixed height, so layout never jumps) until an event worth the operator's attention exists.
            self.status_strip.show_message("", severity="info")

    # ------------------------------------------------------------------ files
    def save_to_file(self) -> None:
        if self._task_locked:
            return
        try:
            state = self.read_state()
            start = self._address or str(_task_files_dir() / f"{state.name}.json")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save task layout", start, "Task layout (*.json)")
            if not path:
                return
            state.save(path)
            self._address = path
            if self.save_button is not None:
                self.save_button.set_dirty(False)
            self._message(f"Saved: {path}")
        except Exception as exc:
            self._message(f"Save failed: {exc}")

    def load_from_file(self) -> None:
        if self._task_locked:
            return
        try:
            start = str(Path(self._address).parent) if self._address else str(_task_files_dir())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load task layout", start, "Task layout (*.json)")
            if not path:
                return
            state = TaskConsoleState.load(path)
            self._address = path
            self.load_state(state)
            if self.save_button is not None:
                self.save_button.set_dirty(False)
        except Exception as exc:
            self._message(f"Load failed: {exc}")

    def _message(self, text: str) -> None:
        if os.environ.get("QT_QPA_PLATFORM", "") == "offscreen":
            self.status_strip.show_message(str(text), severity="info")
            return
        fluent_message(self, "Task console", str(text), kind="info")  # pragma: no cover

    # ------------------------------------------------------------------ teardown
    def stop_all_nodes(self, timeout: float = 2.0) -> bool:
        """Stop every running node through THE lifecycle endpoint (``_stop_logic_node``) but KEEP
        the panels + editors intact.  This is the notebook close = "hide" path: closing the window
        halts all running processes, yet the layout is preserved so reopening the session-bound
        console (``exp.task_console()``) restores the SAME interface rather than a blank new one.
        Distinct from :meth:`shutdown`, which also tears the cards/editors down.

        Going row-by-row through ``_stop_logic_node`` (never bare ``node.stop()``) is load-bearing:
        a bare stop leaves the dead node in ``running_nodes`` and the row painted "running" -- the
        zombie then freezes the WHOLE board's shot clock (``_display_shot`` takes the min over
        bound-and-live signals, and a dead node's provenance never advances) and the reopened
        window shows live-looking rows whose images never update."""
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("node shutdown must run on the TaskConsole Qt owner thread")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("node shutdown timeout must be a non-negative number")
        deadline = time.monotonic() + float(timeout)
        stopped = True
        row_node_ids = {
            id(node)
            for row in self.logic_nodes
            for node in (
                self._logic_nodes.get(id(row)) or self._starting_nodes.get(id(row)),
            )
            if node is not None
        }
        for row in list(self.logic_nodes):
            if (
                self._logic_nodes.get(id(row)) is not None
                or self._starting_nodes.get(id(row)) is not None
            ):
                remaining = max(0.0, deadline - time.monotonic())
                stopped = self._stop_logic_node(row, timeout=remaining) and stopped
        # Row-less injected nodes (show_task_console(running_nodes=[...])) have no row to
        # repaint but must leave the shot clock's live set all the same.
        for node in list(self.running_nodes):
            if id(node) in row_node_ids:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            confirmed = self._stop_node_confirmed(node, timeout=remaining)
            stopped = confirmed and stopped
            if confirmed and node in self.running_nodes:
                self.running_nodes.remove(node)
        if not stopped:
            self.status_strip.show_message(
                "Close delayed: a logic-node owner thread is still active",
                severity="error",
            )
        return stopped

    def stop_nodes_using(self, affected_ids, timeout: float = 2.0) -> bool:
        """Stop exactly the running nodes that reference one of the ``affected_ids`` devices --
        the session's fine-grained device-change hook (``load_config`` swapping specific device
        INSTANCES).  A node is affected iff its :meth:`~LogicNode.referenced_devices` (EXCLUSIVE
        drivers AND OBSERVE records, unwrapped to real identity) intersect the swapped set, so
        reinitialising the camera stops every camera view / occupancy path riding it while a scan
        on the untouched sequencer keeps running.  Goes through the SAME ``_stop_logic_node``
        endpoint as every other stop (no zombie left to freeze the shot clock, #close-reopen)."""
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("node shutdown must run on the TaskConsole Qt owner thread")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("node shutdown timeout must be a non-negative number")
        deadline = time.monotonic() + float(timeout)
        affected = {int(i) for i in affected_ids}
        if not affected:
            return True

        def _touches(node) -> bool:
            references = tuple(node.referenced_devices()) + tuple(
                getattr(node, "lifecycle_devices", lambda: ())()
            )
            return any(id(d) in affected for d in references)

        stopped = True
        row_node_ids = {
            id(node)
            for row in self.logic_nodes
            for node in (
                self._logic_nodes.get(id(row)) or self._starting_nodes.get(id(row)),
            )
            if node is not None
        }
        for row in list(self.logic_nodes):
            node = self._logic_nodes.get(id(row)) or self._starting_nodes.get(id(row))
            if node is not None:
                try:
                    touches = _touches(node)
                except BaseException:
                    stopped = False
                    continue
                if touches:
                    remaining = max(0.0, deadline - time.monotonic())
                    stopped = self._stop_logic_node(row, timeout=remaining) and stopped
        for node in list(self.running_nodes):        # row-less injected nodes
            if id(node) in row_node_ids:
                continue
            try:
                touches = _touches(node)
            except BaseException:
                stopped = False
                continue
            if touches:
                remaining = max(0.0, deadline - time.monotonic())
                confirmed = self._stop_node_confirmed(node, timeout=remaining)
                stopped = confirmed and stopped
                if confirmed and node in self.running_nodes:
                    self.running_nodes.remove(node)
        return stopped

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Stop the refresh timer and every running node's owner thread, then release
        the editors/cards.  IDEMPOTENT -- it is reached from both the window close
        (``show_task_console`` installs it as the pre-close guard) and an explicit
        ``with show_task_console(...)`` / re-run, which can both fire.

        Stopping a node sets its cooperative-cancel event (M5), so a node thread
        blocked in ``camera.acquire`` unwinds and the camera is released -- this is
        what keeps a closed dashboard from leaving a live acquire thread (and a held
        camera / RPyC connection) behind, which previously wedged the kernel.  True
        means all render ownership is joined and teardown completed; False keeps the
        window alive so a later close can retry the unresolved handoff."""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("shutdown timeout must be positive")
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("TaskConsole shutdown must run on its Qt owner thread")
        state = getattr(self, "_shutdown_state", "RUNNING")
        if state == "TERMINATED" or getattr(self, "_shut", False):
            return True
        if state in {
            "STOPPING_NODES",
            "WAITING_RENDER",
            "CLOSING_RESOURCES",
            "TEARING_DOWN_UI",
        }:
            return False
        deadline = time.monotonic() + float(timeout)
        self._timer.stop()
        if state in {"RUNNING", "BLOCKED_NODE_OWNERSHIP"}:
            self._shutdown_state = "STOPPING_NODES"
            self._defer_task_card_teardown = True
            if not self.stop_all_nodes(timeout=max(0.0, deadline - time.monotonic())):
                self._shutdown_state = "BLOCKED_NODE_OWNERSHIP"
                return False
        # Figure-owning editors/cards remain intact until the render worker's real thread
        # termination is confirmed.  A timed-out join is OWNERSHIP_UNRESOLVED, not permission
        # to clear pending state and destroy the Figure underneath the worker.
        self._shutdown_state = "WAITING_RENDER"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._shutdown_state = "BLOCKED_RENDER_OWNERSHIP"
            return False
        try:
            self._render.close(max(0.0, float(remaining)))
            render_stopped = bool(self._render.closed)
        except BaseException as exc:
            render_stopped = False
            self._shutdown_error = exc
        if not render_stopped:
            self._shutdown_state = "BLOCKED_RENDER_OWNERSHIP"
            self.status_strip.show_message(
                "Close delayed: render worker still owns a Figure; retry after it stops",
                severity="error",
            )
            return False
        if not getattr(self, "_on_close_done", False):
            self._shutdown_state = "CLOSING_RESOURCES"
            on_close = getattr(self, "_on_close", None)
            if on_close is not None:
                try:
                    close_result = on_close()
                    if close_result is False:
                        raise RuntimeError("resource close reported unresolved ownership")
                except BaseException as exc:
                    self._shutdown_error = exc
                    self._shutdown_state = "BLOCKED_RESOURCE_CLOSE"
                    self.status_strip.show_message(
                        f"Close delayed: device/resource release failed: {exc}",
                        severity="error",
                    )
                    return False
            self._on_close_done = True
        self._shutdown_state = "TEARING_DOWN_UI"
        try:
            deferred_task_card = self._deferred_task_card
            if deferred_task_card is not None:
                if not self._remove_panel(
                    deferred_task_card,
                    timeout=0.0,
                    render_already_stopped=True,
                    allow_task_owned=True,
                ):
                    raise RuntimeError("deferred task panel teardown was not acknowledged")
                self._deferred_task_card = None
            for key, editor in list(self._panel_editors.items()):
                editor.teardown()
                self._panel_editors.pop(key, None)
            for key, editor in list(self._logic_editors.items()):
                editor.teardown()
                self._logic_editors.pop(key, None)
            completed_cards = getattr(self, "_shutdown_cards_done", set())
            self._shutdown_cards_done = completed_cards
            for card in self.cards:
                if id(card) in completed_cards:
                    continue
                card.shutdown()
                completed_cards.add(id(card))
        except BaseException as exc:
            self._shutdown_error = exc
            self._shutdown_state = "BLOCKED_UI_TEARDOWN"
            self.status_strip.show_message(
                f"Close delayed: UI teardown failed: {exc}", severity="error"
            )
            return False
        self._shutdown_state = "TERMINATED"
        self._shut = True
        return True

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self.shutdown():
            event.ignore()
            return
        super().closeEvent(event)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self.shutdown():
            raise RuntimeError(
                "TaskConsole shutdown is blocked because the render worker still owns a Figure"
            )
        return False


def show_task_console(
    *,
    hub,
    state: TaskConsoleState | None = None,
    task: str | None = None,
    running_nodes: Sequence[object] = (),
    measurements: Sequence[object] = (),
    processors: Sequence[object] = (),
    tasks: Sequence[object] = (),
    session: object | None = None,
    runtime_fence: object | None = None,
    runtime_fence_provider=None,
    scale: float | None = None,
    window_ratio: float = WINDOW_SCREEN_FRACTION,
    title: str = "TaskConsole@Zou lab",
    on_close=None,
    hide_on_close: bool = False,
):
    """Open the console in a Fluent window (mirrors ``show_pulse_gui``: the body
    sizes itself from the primary screen; the window wraps it exactly).

    ``task`` loads a layout YOU saved (``tasks/<name>.json``) or a JSON path;
    without one the console opens empty.

    ``measurements`` is the declarative measurement catalog
    (``exp.readout.measurement_specs()`` -- an AUTO-DISCOVERED list, built-ins +
    any ``@measurement`` / ``register_measurement``): pass it and every spec is
    listed in the header's Add Panel dropdown; picking one + Add Panel creates a
    result panel and opens its Edit tab (auto-generated parameter form + one-click
    Start).  Omit it and the dropdown carries only plot kinds.

    Teardown: closing the window stops the refresh timer and every running node's owner
    thread (the cooperative-cancel path), so no acquire thread is left holding the
    camera -- closing the dashboard genuinely releases it.  Pass ``on_close`` (e.g.
    ``exp.close``) to ALSO disconnect/safe the devices the caller owns when the
    window closes.  In a notebook the returned console is also a context manager
    (``with show_task_console(...) as console:``) and exposes ``console.shutdown()``
    for deterministic teardown on a cell re-run."""

    ensure_qt_app()          # the console is a QWidget: the app must exist BEFORE its ctor
    if state is None and task is not None:
        state = resolve_task_state(task)
    console = TaskConsole(hub=hub, state=state, running_nodes=running_nodes, measurements=measurements,
                          processors=processors, tasks=tasks, session=session,
                          runtime_fence=runtime_fence,
                          runtime_fence_provider=runtime_fence_provider, scale=scale,
                          window_ratio=window_ratio)
    console._on_close = on_close
    # A passed-in node should stream the moment the window opens.  Enrollment is
    # centralized on the console body so every composition surface uses the same
    # stop-unmanaged -> fenced-start transition.
    console._enroll_composed_nodes(running_nodes, runtime_fence)
    # Closing the window must stop the node owner threads (else they keep running, blocked in
    # camera.acquire holding the camera / RPyC link, wedging the kernel).  The console is a CHILD
    # of the window so its own closeEvent never fires on a window close -- we wire the window's
    # signals instead.  Minimising NEVER stops the nodes (only the X / a genuine close does).
    def _wire_close(window) -> None:
        if hide_on_close:
            # Session-bound (notebook) console: the X HIDES the window (keeps the panel layout so a
            # later exp.task_console() restores the SAME interface) and stops every running node so
            # the devices are released.  The hide guard runs on X, never on minimize.
            window.set_hide_guard(console.stop_all_nodes)
        else:
            # Standalone window (.bat / explicit): the X fully tears the console down.
            window.set_close_guard(console.shutdown)
    # ONE launcher sequence (launch_fluent_window: wrap -> wire -> size -> centre -> show ->
    # retain), shared with every other show_* GUI so the steps cannot drift per-launcher.
    launch_fluent_window(console, title=title, hide_on_close=hide_on_close, wire=_wire_close)
    return console


__all__ = [
    "LogicNodeConfig",
    "PanelConfig",
    "TaskConsole",
    "TaskConsoleState",
    "default_console_state",
    "show_task_console",
]
