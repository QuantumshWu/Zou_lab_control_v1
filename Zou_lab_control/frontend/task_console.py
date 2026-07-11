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

import json
import hashlib
import math as _math
import os
import re
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from .live import (
    DEFAULT_HIST_FIT,
    PANEL_SIZES,
    PLOT_KINDS,
    coerce_panel_value,
    kind_supports_roi,
    normalize_facet,
    panel_display_size,
    panel_plot,
    panel_size_cells,
    recommended_grid_size,
    region_binding,
    site_ring_radius,
)
from .style import PALETTE          # the ONE colour source -- panel cmap DEFAULTS reference it (never a literal)
from .pulse_gui import slot_label   # the ONE human slot-label formatter (period/channel, #H3s-F2)
from .qt_fluent import (
    ACCENT,
    CARD_PAD,
    CARD_TITLE_PX,
    GREEN,
    GREY,
    ORANGE,
    RED,
    TEXT,
    WINDOW_SCREEN_FRACTION,
    YELLOW,
    _popup_gap,
    fluent_message,
    fluent_scrollbar_thickness,
    FluentButton,
    FluentCodeEdit,
    FluentComboBox,
    FluentTreeComboBox,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentReadoutEdit,
    FluentPopup,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    setting_label_width,
    FluentStatusDot,
    FluentStatusStrip,
    FluentSwitch,
    FluentTabWidget,
    FluentFloatingEditor,
    ensure_qt_app,
    launch_fluent_window,
    fluent_text_width,
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
from Zou_lab_control._paths import display_path

# The ONE param-kind -> widget registry every form in this module dispatches through: a
# ParamDecl of kind K is built / read / seeded / validated / refreshed by PARAM_WIDGETS[K].
# Adding a kind is one handler there + one whitelist entry on ParamDecl, not 5-7 ladders here.
from .param_widgets import PARAM_WIDGETS, SPAN_KINDS, ParamWidgetContext, RefreshProviders
# The grouped-signal-picker cluster lives in param_widgets (the leaf; see the note above
# strip_node_prefix) -- forward imports only, so the leaf never back-imports this module.
from .param_widgets import (
    coerce_short_labels,
    fill_grouped_signal_combo,
    read_editable_combo,
)

# ParamDecl is the ONE declarative param record both the measurement form and the plot
# panels use (PANEL_PARAMS).  Importing it here (frontend -> neutral_atom is allowed; the
# reverse is not) lets the panel params be real ParamDecls validated by the kind whitelist,
# instead of a parallel ParamSpec class with its own smaller ladder.
from Zou_lab_control.neutral_atom.core.params import ParamDecl

# The default mid-run buffer key -- the spec layer's ONE spelling (TaskSpec.mid_run_key's
# default), imported so the console's spec-less fallbacks can never drift from it.
from Zou_lab_control.neutral_atom.operations.task import DEFAULT_MID_RUN_KEY


TASK_FILES_ENV = "ZLC_TASK_DIR"

# ---- RESERVED expression-namespace keys (each spelled ONCE; every writer/reader shares these).
#: The running task's typed mid-run tensor: injected off-hub by the console each tick, read by the
#: task's dedicated panel (its source is ``value = {TASK_FRAME_KEY}``) -- never a hub signal.
TASK_FRAME_KEY = "__task_frame__"
#: Per-signal publish counters ({name: version}) so a rolling monitor tells a new sample of
#: its OWN source from an unrelated node's version bump.
SIG_VERSIONS_KEY = "__sig_versions__"
#: Shot-coherent per-signal physical validity masks ({name: (R,P) bool}).
SIG_VALID_KEY = "__sig_valid__"
#: Coordinate frames ({signal_name: [x0, x1, y0, y1]}) from any node whose acquisition source
#: declares a ROI -- a 2D panel puts its axes in real camera pixels.
COORD_FRAMES_KEY = "__coord_frames__"

#: The display suffix marking a task's SYNTHETIC mid-run entry in the picker / Logic legend
#: (never part of a hub name) -- one spelling shared by the declared and the running paths.
MID_RUN_TAG = " (mid-run)"

# WHICH hardware a logic node drives is the NODE's own ``occupied_devices()`` declaration
# (operations/logic.py LogicNode, from its ``_occupies`` attribute names) -- the console's
# device-occupancy mutual exclusion in ``_start_logic_node`` intersects it across every running
# node, GUI rows and notebook-injected ``running_nodes=`` alike; nodes on disjoint hardware
# coexist, so there is no second kind-string table here and no global "stop everything" rule.

# Console PANEL kinds.  EVERY plot kind in the ONE table ``live.PLOT_KINDS`` is a console panel
# kind -- it renders through the SAME ``PanelCard`` (``_build_plot`` dispatches on the kind: a 2D
# frame, a site map, a histogram, a 1-D curve, a pulse timeline ...), so a saved figure of ANY kind
# seeds a normal ``PanelCard`` and reads its ``value`` off a hub signal.  The ``panel`` flag is NOT
# "can this be a panel"; it is ONLY "is this offered in the live Add-Panel dropdown" (see
# ADDABLE_PANEL_KINDS below) -- ``pulse`` is ``panel=False`` because you do not add a blank pulse
# panel live (it is reproduced from a saved recipe / a fired sequence), but it IS a real panel kind
# that seeds + renders through PanelCard exactly like every other.  All the per-kind panel tables
# below are derived from the WHOLE table so pulse (and any future kind) works on the seed path with
# no parallel literal to keep in sync.
_PANEL_KINDS: tuple = tuple(PLOT_KINDS)

#: ``key -> label`` for EVERY console panel kind -- the panel/card/frame title + ``PanelConfig`` kind
#: validation (so a saved ``pulse`` figure seeds a normal panel).  Insertion order = the table order.
PANEL_KINDS: dict[str, str] = {pk.key: pk.label for pk in _PANEL_KINDS}

#: The subset offered in the live Add-Panel dropdown (``panel=True``): the kinds you add a BLANK panel
#: of and then wire to a signal.  ``pulse`` is excluded (you do not add a blank pulse panel live), but
#: it is still a full panel kind on the SEED path (a saved pulse figure).  Insertion order = menu order.
ADDABLE_PANEL_KINDS: dict[str, str] = {pk.key: pk.label for pk in _PANEL_KINDS if pk.panel}

#: Every panel + logic-node name is "<base> #N" with N counting from 1 (G1), so two panels /
#: nodes of the same kind are always told apart -- in the card title, the Edit tab, the frame
#: title, and the signal-flow grouping.  ONE source of that scheme for panels and nodes alike.
_INDEX_SUFFIX_RE = re.compile(r"^(.*?)\s*#\d+$")


def indexed_unique_name(base: str, taken) -> str:
    """``"<root> #N"`` with the smallest ``N >= 1`` not already in ``taken``.  Any ``#k`` already
    on ``base`` is stripped first, so re-indexing a loaded ``"1D vector #2"`` re-derives a clean
    number rather than nesting (idempotent for an already-clean layout)."""
    text = str(base or "panel").strip() or "panel"
    m = _INDEX_SUFFIX_RE.match(text)
    root = (m.group(1).strip() if m else text) or "panel"
    taken = set(taken)
    n = 1
    while f"{root} #{n}" in taken:
        n += 1
    return f"{root} #{n}"


def _safe_float(text, fallback: float) -> float:
    """Parse a numeric line-edit, falling back on blank/garbage (the ONE parser the fixed-lim
    lo/hi inputs share between the Setting popup and the Edit tab)."""
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return float(fallback)

# What signal SHAPE each plot kind expects as its ``value`` -- shown in the panel's Setting so
# it is clear which signals fit (e.g. a Site map wants a per-site vector, a 2D image wants a
# frame).  DERIVED from each ``PlotKind.input_format`` in the ONE table; the per-kind contract
# is declared THERE (everything else about the source -- the multi-slot picker + ``value = ...``
# expression -- is universal and kind-agnostic).  Shown in the Setting + enforced in _coerce.
PANEL_INPUT_FORMAT: dict[str, str] = {pk.key: pk.input_format for pk in _PANEL_KINDS}

# The STARTING slot(s) each plot kind opens with (label, default-signal, tooltip).  Every kind
# uses the SAME source MECHANISM (a signal picker + a ``value = ...`` expression box); whether a
# kind can GROW extra slots (+signal / −signal) is declared by PANEL_SINGLE_SLOT_KINDS below
# (the site map is single-slot).  A plot reads its picked input(s) as ``signal`` / ``signal[i]``.
# Each slot = (label, default-signal-name, tooltip).  The DEFAULT (a single blank ``signal``
# slot) lives here; the per-kind overrides (e.g. the site map's "occupancy" slot) come from each
# ``PlotKind.input_slots`` in the ONE table, so PANEL_INPUT_SLOTS is derived, not hand-listed.
_DEFAULT_SLOTS = (("signal", "", "the hub signal to plot"),)
PANEL_INPUT_SLOTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    pk.key: pk.input_slots for pk in _PANEL_KINDS if pk.input_slots
}

# The per-kind declaration of which plot kinds take EXACTLY ONE signal (no +signal / −signal
# slot-growing).  The signal-expression MECHANISM (the picker + ``value = ...`` box + evaluator)
# is universal -- every kind has it -- but a SINGLE-slot kind cannot add more slots because its
# auxiliary data is resolved from signal[0]: the site map pulls its ring centres + frame underlay
# from signal[0]'s producing node, so a 2nd signal slot would be meaningless.  DERIVED from each
# ``PlotKind.single_slot`` flag in the ONE table; PanelCard reads it (no inline per-kind check).
PANEL_SINGLE_SLOT_KINDS: frozenset = frozenset(pk.key for pk in _PANEL_KINDS if pk.single_slot)


def panel_input_slots(kind: str) -> tuple[tuple[str, str, str], ...]:
    """The input slots for a plot kind -- ``[(label, default_signal, tooltip)]``.  The
    SINGLE source of how many signals a plot consumes and what each means."""
    return PANEL_INPUT_SLOTS.get(str(kind), _DEFAULT_SLOTS)


def panel_allows_multi_slot(kind: str) -> bool:
    """Whether a plot kind can grow extra signal slots (+signal / −signal).  Data-driven from
    ``PANEL_SINGLE_SLOT_KINDS`` -- the site map is single-slot (its centres/underlay come from
    signal[0]); every other kind is multi-slot.  Read by PanelCard so the slot UI is declared
    in ONE place, never an inline ``kind == 'sites'`` check scattered through the widget."""
    return str(kind) not in PANEL_SINGLE_SLOT_KINDS


# The grouped-signal-picker helper cluster (signal_state / grouped_signal_items /
# signal_tree_groups / fill_grouped_signal_combo / read_editable_combo / coerce_short_labels,
# #combo-parity) moved DOWN to param_widgets.py -- the leaf this module already imports -- so the
# leaf's old lazy back-imports of it are gone (no frontend cycle).  strip_node_prefix stays here:
# it is the Logic tab's rule (shared vocabulary, not a picker widget helper).


def strip_node_prefix(full: str, prefix: str) -> str:
    """The SHORT signal name = the hub name minus its producing node's disambiguating prefix
    (``judge_occupancy_rate`` -> ``rate``, ``temperature_survival`` -> ``survival``, ``frame`` ->
    ``frame``).  The ONE rule the Logic tab AND the signal picker share, so the nest leaf is ALWAYS the
    short name -- never the full prefixed key, never the verbose axis label."""
    full = str(full)
    prefix = str(prefix or "")
    return full[len(prefix):] if (prefix and full.startswith(prefix) and len(full) > len(prefix)) else full


def _is_number(v) -> bool:
    """True when ``v`` can be read as a finite float (a saved numeric param), else False."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class _PulseSlotsWidget(QtWidgets.QWidget):
    """Structured editor for the two PulseScan execution strategies.

    A scan-slot sweep uploads one complete FPGA table; an API-slot sweep submits one finite pulse
    per row.  The selector changes the meaning and columns of a single program editor.  Each
    strategy keeps its own in-memory buffer because those column spaces are not interchangeable.
    Selecting a pulse template seeds the scan-slot buffer from that template's persisted program;
    the API-slot buffer is generated from its API fields.

    A saved task override is tied to ``program_id``.  It is restored only when that exact template
    is selected; values named ``a1`` or a code buffer can therefore never leak from one template
    into another template whose opaque internal slot happens to have the same index."""

    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._api_widgets: dict[str, QtWidgets.QWidget] = {}
        from ..neutral_atom.operations.measurement import SWEEP_API_SLOT, SWEEP_SCAN_SLOT
        self._scan_slot_kind = SWEEP_SCAN_SLOT
        self._api_slot_kind = SWEEP_API_SLOT
        self._program_code = None
        self._sweep_combo = None
        self._sweep_kind = ""
        self._program_buffers = {SWEEP_SCAN_SLOT: "", SWEEP_API_SLOT: ""}
        self._columns: dict[str, list] = {SWEEP_SCAN_SLOT: [], SWEEP_API_SLOT: []}
        self._specs: dict[str, list] = {SWEEP_SCAN_SLOT: [], SWEEP_API_SLOT: []}
        self._available = {SWEEP_SCAN_SLOT: False, SWEEP_API_SLOT: False}
        self._program_id = ""
        self._pending_program_id = ""
        self._pending_api: dict[str, str] = {}
        self._pending_sweep_kind = ""
        self._pending_program = ""
        self._box = QtWidgets.QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(scaled_px(6, minimum=4))
        self._api_box = QtWidgets.QVBoxLayout()
        self._api_box.setContentsMargins(0, 0, 0, 0)
        self._api_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._api_box)
        self._selector_box = QtWidgets.QVBoxLayout()
        self._selector_box.setContentsMargins(0, 0, 0, 0)
        self._selector_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._selector_box)
        self._program_box = QtWidgets.QVBoxLayout()
        self._program_box.setContentsMargins(0, 0, 0, 0)
        self._program_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._program_box)

    @staticmethod
    def _drop_layout(layout) -> None:
        """Tear down every child widget + nested layout under ``layout`` (rebuilt from scratch)."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None); w.deleteLater()
            child = item.layout()
            if child is not None:
                _PulseSlotsWidget._drop_layout(child)

    def rebuild(self, api_rows, scan_rows, *, hardware_program: str = "",
                program_id: str = "") -> None:
        """Rebuild from one pulse template.

        ``api_rows`` entries are ``(handle, coordinate, kind, target, unit, current)``;
        ``scan_rows`` entries are ``(coordinate, kind, target, unit, label)``.
        """

        program_id = str(program_id or "")
        same_program = bool(program_id and program_id == self._program_id)
        restore_saved = bool(program_id and program_id == self._pending_program_id)

        remembered_api = {}
        if same_program:
            remembered_api = {name: widget.text().strip()
                              for name, widget in self._api_widgets.items()}
            self._stash_program()

        self._drop_layout(self._api_box)
        self._drop_layout(self._selector_box)
        self._drop_layout(self._program_box)
        self._api_widgets = {}
        self._program_code = None
        self._sweep_combo = None
        self._program_id = program_id

        self._api_box.addWidget(FluentSectionLabel("API parameters"))
        if api_rows:
            labels = [slot_label(kind, target)
                      for _handle, _coord, kind, target, _unit, _current in api_rows]
            label_width = setting_label_width(labels, minimum=72)
            for handle, _coordinate, kind, target, unit, current in api_rows:
                label = slot_label(kind, target)
                seed = (self._pending_api.get(handle) if restore_saved else None)
                if seed is None and same_program:
                    seed = remembered_api.get(handle)
                if seed is None:
                    seed = f"{float(current):g}"
                edit = FluentLineEdit(seed, self)
                edit.setMinimumWidth(scaled_px(120, minimum=96))
                edit.setPlaceholderText(str(unit))
                edit.setToolTip(
                    f"Resting value for {label} ({unit}).  In an API-slot sweep, the program "
                    "overrides this handle once per row.")
                edit.textChanged.connect(self.changed)
                self._api_box.addWidget(FluentSettingRow(label, edit, label_width=label_width))
                self._api_widgets[str(handle)] = edit
        else:
            note = FluentLabel("(this template has no API parameter)", self)
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self._api_box.addWidget(note)

        from ..neutral_atom.timing import scan_column_spec
        self._columns[self._api_slot_kind] = [
            (coordinate, slot_label(kind, target), str(unit or ""))
            for _handle, coordinate, kind, target, unit, _current in api_rows
        ]
        self._specs[self._api_slot_kind] = [
            scan_column_spec(coordinate, "dac" if kind == "dac" else "duration",
                             unit=(unit or "ns"))
            for _handle, coordinate, kind, _target, unit, _current in api_rows
        ]
        self._columns[self._scan_slot_kind] = []
        self._specs[self._scan_slot_kind] = []
        for coordinate, kind, target, unit, stored_label in scan_rows:
            display = stored_label or slot_label(kind, target)
            display_unit = "ns ticks" if kind == "duration" else (
                "integer code (LSB)" if kind == "dac" else str(unit or ""))
            self._columns[self._scan_slot_kind].append((coordinate, display, display_unit))
            self._specs[self._scan_slot_kind].append(
                scan_column_spec(coordinate, kind, unit=(unit or "ns")))

        self._available = {
            self._scan_slot_kind: bool(scan_rows),
            self._api_slot_kind: bool(api_rows),
        }
        if not same_program:
            self._program_buffers = {
                self._scan_slot_kind: str(hardware_program or ""),
                self._api_slot_kind: "",
            }
        elif not self._program_buffers[self._scan_slot_kind].strip():
            self._program_buffers[self._scan_slot_kind] = str(hardware_program or "")

        default_kind = self._scan_slot_kind if scan_rows else (
            self._api_slot_kind if api_rows else "")
        if restore_saved and self._available.get(self._pending_sweep_kind, False):
            self._sweep_kind = self._pending_sweep_kind
            self._program_buffers[self._sweep_kind] = self._pending_program
        elif not same_program or not self._available.get(self._sweep_kind, False):
            self._sweep_kind = default_kind

        self._build_sweep_selector()
        self._render_program()
        self._pending_program_id = ""
        self._pending_api = {}
        self._pending_sweep_kind = ""
        self._pending_program = ""
        self.changed.emit()

    def _build_sweep_selector(self) -> None:
        combo = FluentComboBox()
        choices = (
            ("Scan slots (hardware table)", self._scan_slot_kind),
            ("API slots (one pulse per point)", self._api_slot_kind),
        )
        for label, kind in choices:
            combo.addItem(label, kind)
            item = combo.model().item(combo.count() - 1)
            if item is not None:
                item.setEnabled(bool(self._available[kind]))
        index = combo.findData(self._sweep_kind)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.currentIndexChanged.connect(self._on_sweep_changed)
        combo.setToolTip(
            "Scan slots upload one complete FPGA table. API slots resolve and submit one finite "
            "pulse per program row.")
        self._sweep_combo = combo
        self._selector_box.addWidget(FluentSettingRow(
            "Sweep", combo, label_width=setting_label_width(["Sweep"], minimum=72)))

    def _on_sweep_changed(self, *_args) -> None:
        if self._sweep_combo is None:
            return
        kind = str(self._sweep_combo.currentData() or "")
        if kind == self._sweep_kind or not self._available.get(kind, False):
            return
        self._stash_program()
        self._sweep_kind = kind
        self._render_program()
        self.changed.emit()

    def _stash_program(self) -> None:
        if self._program_code is not None and self._sweep_kind:
            self._program_buffers[self._sweep_kind] = self._program_code.toPlainText()

    def _render_program(self) -> None:
        self._drop_layout(self._program_box)
        self._program_code = None
        if not self._sweep_kind or not self._available.get(self._sweep_kind, False):
            note = FluentLabel(
                "(bind at least one scan slot or API slot in the Pulse GUI)", self)
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self._program_box.addWidget(note)
            return

        title = "Hardware scan-slot program" if self._sweep_kind == self._scan_slot_kind \
            else "API-slot sweep program"
        self._program_box.addWidget(FluentSectionLabel(title))
        columns = self._columns[self._sweep_kind]
        legend = ["Columns of scan_table (one row = one point, columns advance in lockstep):"]
        legend.extend(f"  {name}: {display}  [{unit}]" for name, display, unit in columns)
        legend_label = FluentLabel("\n".join(legend), self)
        legend_label.setWordWrap(True)
        legend_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self._program_box.addWidget(legend_label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(scaled_px(6, minimum=4))
        btn_row.addWidget(FluentLabel("template:", self))
        for template in ("column_stack", "grid"):
            button = FluentButton(template, color=GREY)
            button.clicked.connect(lambda *_a, value=template: self._insert_template(value))
            btn_row.addWidget(button, 0)
        btn_row.addStretch(1)
        self._program_box.addLayout(btn_row)

        from ..neutral_atom.timing import scan_table_template
        seed = str(self._program_buffers[self._sweep_kind] or "").strip()
        if not seed:
            seed = scan_table_template("column_stack", self._specs[self._sweep_kind])
        self._program_buffers[self._sweep_kind] = seed
        editor = FluentCodeEdit(seed)
        editor.setMinimumHeight(scaled_px(120, minimum=90))
        editor.setToolTip(
            "Python assigning an (N_points x n_columns) array to scan_table. Values use each "
            "selected slot's native unit.")
        editor.textChanged.connect(self.changed)
        self._program_box.addWidget(editor)
        self._program_code = editor

    def _insert_template(self, template: str) -> None:
        from ..neutral_atom.timing import scan_table_template
        if self._program_code is not None and self._sweep_kind:
            self._program_code.setPlainText(
                scan_table_template(template, self._specs[self._sweep_kind]))

    def values_dict(self) -> dict:
        """Return the sole structured PulseScan form value."""

        api: dict[str, float] = {}
        for name, widget in self._api_widgets.items():
            text = widget.text().strip()
            if not text:
                continue
            try:
                api[name] = float(text)
            except ValueError:
                continue
        program = self._program_code.toPlainText() if self._program_code is not None else ""
        return {
            "program_id": self._program_id,
            "api": api,
            "sweep_kind": self._sweep_kind,
            "program": program,
        }

    def seed_value(self, value) -> None:
        """Queue a saved override, applied only to the exact matching pulse program."""

        if not isinstance(value, Mapping):
            return
        for name, item in dict(value.get("api") or {}).items():
            self._pending_api[str(name)] = f"{float(item):g}" if _is_number(item) else str(item)
        self._pending_sweep_kind = str(value.get("sweep_kind") or "")
        self._pending_program = str(value.get("program") or "")
        self._pending_program_id = str(value.get("program_id") or "")

# The ONE description of a source expression's namespace -- owned by operations.signal_expr
# (the single source the analysis layer + GUI share), fetched lazily so the frontend module
# import stays off neutral_atom's import graph (every other neutral_atom use here is lazy too).
_SOURCE_EXPR_HELP_CACHE: str | None = None


def SOURCE_EXPR_HELP() -> str:
    """The expression-namespace help text (a callable so it stays a single source -- the literal
    lives once in ``operations.signal_expr.SIGNAL_EXPR_HELP``)."""
    global _SOURCE_EXPR_HELP_CACHE
    if _SOURCE_EXPR_HELP_CACHE is None:
        from ..neutral_atom.operations.signal_expr import SIGNAL_EXPR_HELP
        _SOURCE_EXPR_HELP_CACHE = SIGNAL_EXPR_HELP
    return _SOURCE_EXPR_HELP_CACHE

class _SignalExprWidget(QtWidgets.QWidget):
    """A multi-slot hub-signal picker + a ``value = ...`` expression -- the SAME source control a
    plot panel uses, packaged as a measurement/processor param (ParamDecl kind ``signal_expr``).

    Pick one or more live hub signals (read as ``signal`` / ``signal[i]``) and combine them in a
    one-line expression.  So ANY "source" field -- a processor's frame, a pulse-scan's y -- can
    subscribe to several running nodes' signals, never just one bare name.  Output schema:
    ``{"inputs": [name, ...], "source": "value = ..."}`` (the :class:`SignalExpr` value).  Reuses
    the SAME primitives as the panel Setting (``fill_grouped_signal_combo`` / ``read_editable_combo``
    / the ``FluentFloatingEditor``) + the shared seed rule, so the logic is single-source."""

    changed = QtCore.pyqtSignal()

    def __init__(self, *, signals_provider=None, sources_provider=None, formats_provider=None,
                 labels_provider=None, title: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._signals_provider = signals_provider
        self._sources_provider = sources_provider
        self._formats_provider = formats_provider
        # The SHORT-name map (``short_names_provider``: {full hub name -> short name}), passed as the
        # picker's ``labels`` so a leaf shows "frame_0" -- NOT the prefix-stripped "0/1/2" the
        # _common_token_prefix fallback yields without it.  This is what makes THIS source picker render
        # IDENTICALLY to the plot-panel Setting picker (same fill_grouped_signal_combo + same labels).
        self._labels_provider = labels_provider
        self._inputs: list[str] = ["frame_0"]
        self._editor = None
        # ONE label-column width for this widget's rows (signal / signal[i] / value), via the SAME
        # setting_label_width rule every form uses -- so it aligns + follows the one logic.
        self._label_w = setting_label_width(["signal[0]", "value"], minimum=64)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(scaled_px(4, minimum=3))
        # Spans the FULL form width; its title is a FluentSectionLabel header (the ONE section
        # style -- same as the Setting popup sections), with the signal slots + value rows stacking
        # flush beneath (no indent: section-vs-row hierarchy is weight+colour, never indentation).
        if title:
            root.addWidget(FluentSectionLabel(title))
        self._slot_box = QtWidgets.QVBoxLayout()
        self._slot_box.setContentsMargins(0, 0, 0, 0)
        self._slot_box.setSpacing(scaled_px(4, minimum=3))
        root.addLayout(self._slot_box)
        self.slot_combos: list = []
        # +/- buttons sit UNDER the label column (an empty label), so they line up with the
        # control column of the rows above instead of floating full-width.
        btn_inner = QtWidgets.QWidget()
        btn_row = QtWidgets.QHBoxLayout(btn_inner)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(scaled_px(6, minimum=4))
        self._add_btn = FluentButton("+ signal", color=GREY)
        self._add_btn.setToolTip("Add another signal slot (read as signal[i] in the expression).")
        self._add_btn.clicked.connect(self._add_slot)
        self._rm_btn = FluentButton("− signal", color=GREY)
        self._rm_btn.setToolTip("Remove the last signal slot.")
        self._rm_btn.clicked.connect(self._remove_slot)
        btn_row.addWidget(self._add_btn, 0)
        btn_row.addWidget(self._rm_btn, 0)
        btn_row.addStretch(1)
        # +/- buttons under a blank label so they line up with the signal-slot control column
        root.addWidget(FluentSettingRow("", btn_inner, label_width=self._label_w))
        self._source_edit = FluentLineEdit("value = signal")
        self._source_edit.setMinimumWidth(scaled_px(160, minimum=120))
        self._source_edit.setToolTip(SOURCE_EXPR_HELP())
        self._source_edit.textChanged.connect(self.changed)
        self._edit_btn = FluentButton("Edit…", color=GREY)
        self._edit_btn.setFixedWidth(scaled_px(56, minimum=44))
        self._edit_btn.setToolTip("Open a large floating editor for this expression")
        self._edit_btn.clicked.connect(self._open_editor)
        expr_inner = QtWidgets.QWidget()
        expr_row = QtWidgets.QHBoxLayout(expr_inner)
        expr_row.setContentsMargins(0, 0, 0, 0)
        expr_row.setSpacing(scaled_px(6, minimum=4))
        expr_row.addWidget(self._source_edit, 1)
        expr_row.addWidget(self._edit_btn, 0)
        # the expression on its OWN labelled "value" row, aligned to the same label column as the
        # signal slots above -- so the whole source control reads as one tidy grid (#4)
        root.addWidget(FluentSettingRow("value", expr_inner, label_width=self._label_w))
        self._rebuild_slots()

    def _names(self) -> list:
        try:
            return [str(n) for n in self._signals_provider()] if callable(self._signals_provider) else []
        except Exception:
            return []

    def _sources(self) -> dict:
        try:
            return self._sources_provider() if callable(self._sources_provider) else {}
        except Exception:
            return {}

    def _formats(self) -> dict:
        try:
            return self._formats_provider() if callable(self._formats_provider) else {}
        except Exception:
            return {}

    def _labels(self) -> dict:
        return coerce_short_labels(self._labels_provider)

    def _rebuild_slots(self) -> None:
        for combo in self.slot_combos:
            combo.setParent(None); combo.deleteLater()
        self.slot_combos = []
        while self._slot_box.count():
            item = self._slot_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None); w.deleteLater()
        n = max(1, len(self._inputs))
        for i in range(n):
            combo = FluentTreeComboBox()             # collapsible-tree signal picker (G2)
            label = f"signal[{i}]" if n > 1 else "signal"
            cur = self._inputs[i] if i < len(self._inputs) else ""
            fill_grouped_signal_combo(combo, names=self._names(), sources=self._sources(),
                                      formats=self._formats(), labels=self._labels(), current=cur)
            combo.activated.connect(lambda *_a, idx=i: self._on_pick(idx))
            self.slot_combos.append(combo)
            self._slot_box.addWidget(FluentSettingRow(label, combo, label_width=self._label_w))
        self._rm_btn.setEnabled(n > 1)

    def _collect_inputs(self) -> None:
        if self.slot_combos:
            self._inputs = [read_editable_combo(c) for c in self.slot_combos]

    def _on_pick(self, idx: int) -> None:
        self._collect_inputs()
        # Single slot: the picker IS "read this signal" -> point the source at it (value = signal).
        # Multi-slot: the expression is user-authored across slots, so a pick just rebinds slot idx.
        if len(self.slot_combos) <= 1 and idx == 0:
            self._source_edit.blockSignals(True)
            self._source_edit.setText("value = signal")
            self._source_edit.blockSignals(False)
        self.changed.emit()

    def _add_slot(self) -> None:
        from ..neutral_atom.operations.signal_expr import seed_source_for_slots
        self._collect_inputs()
        self._inputs.append("")
        self._source_edit.blockSignals(True)
        self._source_edit.setText(seed_source_for_slots(len(self._inputs), self._source_edit.text()))
        self._source_edit.blockSignals(False)
        self._rebuild_slots()
        self.changed.emit()

    def _remove_slot(self) -> None:
        if len(self._inputs) <= 1:
            return
        from ..neutral_atom.operations.signal_expr import seed_source_for_slots
        self._collect_inputs()
        self._inputs.pop()
        self._source_edit.blockSignals(True)
        self._source_edit.setText(seed_source_for_slots(len(self._inputs), self._source_edit.text()))
        self._source_edit.blockSignals(False)
        self._rebuild_slots()
        self.changed.emit()

    def _open_editor(self) -> None:
        if self._editor is not None:
            self._editor.raise_(); self._editor.activateWindow(); return
        self._editor = FluentFloatingEditor(SOURCE_EXPR_HELP(), self._source_edit.text(),
                                            self.window(), title="Edit signal expression")
        self._editor.applied.connect(lambda text: self._source_edit.setText(" ".join(text.split("\n")).strip()))
        self._editor.destroyed.connect(self._clear_editor)
        self._editor.show()

    def _clear_editor(self, *_a) -> None:
        self._editor = None

    def rebuild_combos(self) -> None:
        """Refill every slot combo from the providers (a tab re-show: a signal published since
        the form opened becomes pickable), keeping the current pick."""
        for combo in self.slot_combos:
            cur = read_editable_combo(combo)
            fill_grouped_signal_combo(combo, names=self._names(), sources=self._sources(),
                                      formats=self._formats(), labels=self._labels(), current=cur)

    def values_dict(self) -> dict:
        """The current ``{"inputs": [...], "source": "value = ..."}`` (the signal_expr value)."""
        self._collect_inputs()
        inputs = [str(n) for n in self._inputs if str(n).strip()]
        return {"inputs": inputs, "source": self._source_edit.text().strip() or "value = signal"}

    def set_value(self, value) -> None:
        """Seed from a ``{"inputs", "source"}`` dict (a default / saved value)."""
        from ..neutral_atom.operations.signal_expr import SignalExpr
        expr = SignalExpr.from_value(value)
        self._inputs = list(expr.inputs) or ["frame_0"]
        self._source_edit.blockSignals(True)
        self._source_edit.setText(expr.source)
        self._source_edit.blockSignals(False)
        self._rebuild_slots()


# Every Monitor-board panel is a PURE VIEW (role "plot"): it is fully decoupled
# from acquisition -- it shows a hub signal and carries the full plotter Edit (the
# whole DataFigure fit set + manual limits + the display knobs), but it NEVER
# builds / owns / starts a node.  The things that PRODUCE data
# (measurement / processor / task) are LOGIC NODES on the separate Logic tab, not
# board panels (see LogicNodeConfig / LogicNodeRow).  ``role`` stays on PanelConfig
# (always "plot") only so an older saved layout round-trips cleanly.
PANEL_ROLES = ("plot",)

# The KINDS of LOGIC NODE the Logic tab hosts -- the node layers (measurement /
# processor / task).  "camera" is the continuous camera Measurement (a live frame
# stream); "measurement" a swept measurement; "processor" a reactive transform;
# "task" a one-shot orchestration.  A node is added STOPPED and Start/Stop'd from
# its own Edit -- it only ever publishes to the hub (display suppressed).
LOGIC_KINDS = ("camera", "measurement", "processor", "task")

CMAPS = ("inferno", "viridis", "magma", "plasma", "gray", "coolwarm")

# A fresh plot panel is BLANK: a pure view is fully decoupled from acquisition, so
# it shows nothing until the user picks a hub signal in its Setting (signal_combo)
# -- it must NOT auto-bind to any node's signal.  An empty source is the blank
# state; ``refresh`` treats it (and a source that produces None) as "pick a signal"
# rather than an error, so a blank panel sits quietly until wired.
_BLANK_SOURCE = ""


# A plot panel's per-kind params are REAL ``ParamDecl``s -- the SAME declarative record the
# measurement form uses -- so they render through the SAME PARAM_WIDGETS registry and are
# validated by the SAME kind whitelist (a typo'd kind raises at construction, instead of a
# parallel ParamSpec silently degrading to a text box).  ``display`` (a ParamDecl DATA flag,
# not an art knob) places the param: True = a basic display knob in the Setting popup (the
# colormap chooser); False = a functional plot-API param in the panel's Edit tab -- so the two
# surfaces never duplicate.  Adding a panel param is ONE ParamDecl here; adding a KIND is one
# handler in param_widgets + one whitelist entry on ParamDecl.
PANEL_PARAMS: dict[str, tuple[ParamDecl, ...]] = {
    "2d": (
        ParamDecl(key="cmap", label="colormap", kind="choice", default=PALETTE["cmap_scan"], choices=CMAPS,
                  tooltip="Image colormap", display=True),
    ),
    "sites": (
        # A site map is a binary occupancy OVERLAY (faint ring = empty, bold ring =
        # occupied) on the camera FRAME.  The colormap applies to that frame underlay
        # (its counts colorbar); the rings carry no scale.  It takes ONE signal input --
        # the per-site occupancy (PANEL_INPUT_SLOTS["sites"], picked in the Setting's Source
        # section); its ring CENTRES and the frame UNDERLAY auto-resolve from that signal's
        # producing node, so they are NOT extra slots or params here.
        ParamDecl(key="cmap", label="colormap", kind="choice", default=PALETTE["cmap_camera"], choices=CMAPS,
                  tooltip="Colormap for the camera-frame underlay", display=True),
    ),
    # Pure DISPLAY knobs (history / bins / fit / log axis / colormap) live in the lightweight
    # Setting popup (display=True): they only change how the SAME data is drawn, so they belong with
    # size / relim where an operator reaches for them.  Only acquisition / measurement-API params
    # (none on these display-only kinds) would be display=False and live in the Edit tab.
    "1d": (),
    "monitor": (
        ParamDecl(key="length", label="history", kind="int", default=300, lo=20, hi=10_000,
                  display=True, tooltip="Rolling history length (shots kept on screen)"),
        # The side distribution is ONE plot kind's toggle (not a separate "no-dist" kind):
        # ON shows the histogram band beside the trace, OFF gives the bare rolling line.
        ParamDecl(key="show_dist", label="side distribution", kind="bool", default=True, display=True,
                  tooltip="Show the side distribution histogram beside the rolling trace"),
    ),
    "hist": (
        ParamDecl(key="bins", label="bins", kind="int", default=60, lo=5, hi=500, display=True,
                  tooltip="Histogram bins"),
        # The fit is a confocal-style capsule tri-toggle (none / single / double), NOT a forced default:
        # the operator picks which fit to draw on whatever data the source provides.  "double" is the
        # dark/bright readout convention.  ``segmented=True`` renders it as the TriStateToggleSwitch
        # capsule (sliding thumb) instead of a combo box.
        ParamDecl(key="fit", label="fit", kind="choice", choices=("none", "single", "double"),
                  default=DEFAULT_HIST_FIT, segmented=True, display=True,
                  tooltip="Distribution fit (drives the display directly -- no auto-decision):\n"
                          "  none   = no fit curve\n"
                          "  single = one Gaussian\n"
                          "  double = the dark/bright two-Gaussian readout (fidelity stat shown only "
                          "when the two peaks cleanly separate, else 'fit F=N/A')"),
        # A log count axis makes a SPARSE bright tail (rare high occupancy) visible -- on a linear
        # axis a handful of bright shots vanish under the dark peak.  Default OFF (linear).
        ParamDecl(key="ylog", label="log count axis", kind="bool", default=False, display=True,
                  tooltip="Log-scale the count (y) axis -- reveals a sparse bright tail"),
    ),
    # A pulse panel (seeded from a saved pulse figure) has ONE display knob: whether to draw the
    # always-off channel rows.  The seed restores the saved value; toggling it re-renders the timeline.
    "pulse": (
        ParamDecl(key="include_always_off", label="show off rows", kind="bool", default=True, display=True,
                  tooltip="Draw channel rows that stay OFF the whole sequence (and idle DAC buses)"),
    ),
    # NOTE: there is DELIBERATELY no ``"grid"`` entry.  A grid panel's params are its per-site
    # ``sub_plot_kind``'s params (a hist grid -> the ``"hist"`` bins/fit/ylog, a 2d grid -> the ``"2d"``
    # colormap), resolved dynamically by ``PanelCard._param_kind`` -- so the Setting/Edit UI ALWAYS matches
    # what each cell actually is, instead of a hard-coded hist set that lied for a kernel grid (#4).
}


# Grid-ONLY per-cell title knob (#5): a grid panel ADDS this to its sub_plot_kind's ``PANEL_PARAMS`` so the
# operator can edit the per-cell title TEMPLATE from the Edit tab.  ``display=False`` => the Edit tab (a
# functional knob), not the lightweight Setting popup.  It flows through the SAME ``store_display_param`` ->
# ``GridCell.consume_param`` path every grid display knob uses, and round-trips through the saved view -- so
# ``{id}`` (the facet-aware identifier), ``{popt[i]}`` (a fit param), ``{fid}`` (readout fidelity) are all
# reachable.  (There is no font-SIZE knob: the cell title auto-tracks the xy tick-label size -- _cell_title_pt.)
GRID_TITLE_PARAMS: tuple[ParamDecl, ...] = (
    ParamDecl(key="title_template", label="cell title", kind="text", default="{id}", display=False,
              tooltip="Per-cell title template.  {id}=facet identifier (site / repeat / scan value); "
                      "{k}=cell index; {popt[i]}=a fit parameter; {fid}=readout fidelity.  "
                      "e.g. '{id}  F={popt[2]:.2f}'"),
)


def _panel_display_decls(kind: str, param_kind: str) -> tuple[ParamDecl, ...]:
    """The FULL ParamDecl list a panel's Setting / Edit UI + save / recipe enumerate: the kind's own
    ``PANEL_PARAMS`` plus, for a GRID, the grid-generic per-cell title knobs (#5).  The ONE place the two
    are combined, so every enumeration site shows the SAME set and a grid's title template / size are
    edited, applied, saved and reopened through the very same path as bins / cmap."""
    decls = PANEL_PARAMS.get(param_kind, ())
    return decls + GRID_TITLE_PARAMS if kind == "grid" else decls


def _panel_param_default(kind: str, key: str) -> object:
    """The declared default of a panel kind's param, from the ONE ``PANEL_PARAMS`` catalog -- so a
    kind's colormap default (``2d`` -> ``inferno``, ``sites`` -> ``gray``) has a SINGLE source and is
    never hand-typed at a consume site.  Returns ``None`` for a kind/key with no declared param."""
    for decl in PANEL_PARAMS.get(str(kind), ()):  # noqa: SIM110 - explicit loop is clearer than any()
        if decl.key == key:
            return decl.default
    return None


def _resolved_param(kind: str, params: Mapping[str, object], key: str) -> object:
    """The value a panel of ``kind`` actually renders for ``key``: the operator's stored
    ``params[key]`` when PRESENT, else the kind's declared default from ``PANEL_PARAMS``
    (:func:`_panel_param_default`) -- so a consume site (plot build / Edit snapshot) never
    hand-types a declared default, and changing a declaration changes the render AND the
    Setting/Edit UI together (they read the same decl).  Presence is ``key in params``,
    never a truthiness test: ``False`` / ``0`` are legal stored values for a bool/int knob."""
    store = params or {}
    return store[key] if key in store else _panel_param_default(kind, key)


def _resolved_cmap(kind: str, params: Mapping[str, object]) -> str:
    """The colormap a panel of ``kind`` actually draws with: the operator's picked ``params['cmap']``
    if set, else the kind's declared default from ``PANEL_PARAMS`` (``_panel_param_default``).  Returns
    an empty string for a kind that declares no cmap param (1-D / hist / monitor draw no image), so a
    caller can store ``''`` for "no colormap" and a colormap-drawing kind always resolves a real name.
    This is the SINGLE resolver for both the plot-build sites and the save's recorded view state."""
    picked = str((params or {}).get("cmap") or "").strip()
    if picked:
        return picked
    default = _panel_param_default(kind, "cmap")
    return str(default) if default else ""

def _card_y_is_view_axis(card) -> bool:
    """Does THIS panel's y axis take a view-window pin -- an image, where x AND y are spatial
    (pixel) coordinates and the value lives on the colour limit?  Reads the LIVE object's own
    ``y_is_view_axis`` flag -- the plot class's for a flat panel, the CELL family's for a grid
    (so a grid of image cells offers the pin and a hist/1d grid does not) -- falling back to the
    kind's plot class in ``PLOT_KIND_BY_KEY`` before a plotter exists.  The ONE resolver every
    Edit surface keys the "y range" row off, mirroring how the plot side gates ``view_ylim`` in
    ``apply_param`` / ``consume_param`` on the same flag: UI and apply can never disagree."""
    from .live import PLOT_KIND_BY_KEY

    plotter = getattr(card, "plotter", None)
    cell = getattr(plotter, "cell_renderer", None)
    if cell is not None:
        return bool(getattr(cell, "y_is_view_axis", False))
    if plotter is not None:
        return bool(getattr(plotter, "y_is_view_axis", False))
    pk = PLOT_KIND_BY_KEY.get(str(getattr(card.config, "kind", "")))
    return bool(getattr(pk.cls, "y_is_view_axis", False)) if pk is not None else False


def _general_fit_models_for_kind(kind: str) -> list:
    """The general curve-fit models offered for a panel of ``kind``: resolve kind -> render_family ->
    the ONE :func:`live.general_fit_models` capability table (keyed off ``render_family``, the single
    source).  A kind's OWN built-in fit no longer suppresses the general one -- a histogram
    (render_family ``1d``) is offered the 1-D family here ALONGSIDE its bimodal ``fit`` knob, while a
    site map (render_family ``auto``) is offered nothing (its occupancy rings are not a fittable
    curve).  Both Setting and Edit read this ONE adapter; the fit engine stays the sole owner of model
    keys, families, and parameter names."""
    from .live import PLOT_KIND_BY_KEY, general_fit_models
    pk = PLOT_KIND_BY_KEY.get(str(kind))
    return general_fit_models(pk.render_family) if pk is not None else []


def build_fit_request(model, selection, *, fixed=None, initial=None, coordinate_frame=None):
    """The ONE structured-fit-request builder every surface funnels through (the Setting Analysis
    section, the Edit Analysis section, a drag retarget): a typed :class:`core.FitRequest` carrying the
    model, the current selection, and the optional per-parameter ``fixed`` clamps / full-vector
    ``initial`` seeds -- NO free-text argument string is ever evaluated.  ``coordinate_frame`` defaults
    to the selection's own frame."""
    from ..neutral_atom.core.fitting import FitRequest
    return FitRequest(
        str(model),
        selection=selection,
        fixed=dict(fixed or {}),
        initial=None if not initial else tuple(float(v) for v in initial),
        coordinate_frame=coordinate_frame or getattr(selection, "frame", "data"),
    )


class _FitFixSeedEditor(QtWidgets.QWidget):
    """Compact typed per-parameter fix/seed editor for a fit model -- the TYPED replacement for the
    reference GUI's free-text ``Fit params:`` box.  For each parameter of the selected model it shows a
    narrow ``fix`` field (clamp the parameter at this value) and a ``seed`` field (initial guess); the
    values are reported as :attr:`FitRequest.fixed` / :attr:`FitRequest.initial` through the ONE
    :func:`build_fit_request`, so "fixing a parameter" works again with no ``eval``.  ``initial`` is
    only reported when EVERY parameter carries a seed (the core takes a full seed vector or none); a
    fixed parameter always overrides its seed.  Reused verbatim on both the Setting popup and the Edit
    tab, so the two surfaces can never diverge."""

    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(scaled_px(6, minimum=4))
        self._grid.setVerticalSpacing(scaled_px(4, minimum=2))
        self._rows: dict[str, tuple] = {}
        self._model = ""

    def set_model(self, model: str) -> None:
        """Rebuild the rows for ``model``'s parameters (a no-op if the model is unchanged)."""
        model = str(model or "")
        if model == self._model and self._rows:
            return
        self._model = model
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows = {}
        if not model:
            return
        from ..neutral_atom.core.fitting import fit_model
        try:
            names = fit_model(model).names
        except Exception:
            names = ()
        for row, name in enumerate(names):
            label = FluentLabel(name)
            label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            fix = FluentLineEdit("")
            fix.setPlaceholderText("fix")
            fix.setFixedWidth(scaled_px(64, minimum=48))
            fix.setToolTip(f"Clamp {name} at this value (leave blank to fit it freely)")
            seed = FluentLineEdit("")
            seed.setPlaceholderText("seed")
            seed.setFixedWidth(scaled_px(64, minimum=48))
            seed.setToolTip(f"Initial guess for {name} (used only when every parameter is seeded)")
            fix.editingFinished.connect(self.changed)
            seed.editingFinished.connect(self.changed)
            self._grid.addWidget(label, row, 0)
            self._grid.addWidget(fix, row, 1)
            self._grid.addWidget(seed, row, 2)
            self._rows[name] = (fix, seed)

    def values(self):
        """Return ``(fixed: dict[name -> float], initial: tuple | None)`` parsed from the fields."""
        fixed: dict[str, float] = {}
        seeds: list[float] = []
        all_seeded = bool(self._rows)
        for name, (fix, seed) in self._rows.items():
            fix_text = fix.text().strip()
            if fix_text:
                try:
                    fixed[name] = float(fix_text)
                except ValueError:
                    pass
            seed_text = seed.text().strip()
            try:
                seeds.append(float(seed_text))
            except ValueError:
                all_seeded = False
                seeds.append(0.0)
        initial = tuple(seeds) if (all_seeded and seeds) else None
        return fixed, initial

    def seed_from_request(self, request) -> None:
        """Re-seed the fields from a stored request's ``fixed`` / ``initial`` (blocking signals)."""
        from ..neutral_atom.core.fitting import FitRequest
        req = (FitRequest.from_dict(request) if isinstance(request, Mapping)
               else request if isinstance(request, FitRequest) else None)
        initial = None if req is None else req.initial
        fixed = {} if req is None else dict(req.fixed)
        for index, (name, (fix, seed)) in enumerate(self._rows.items()):
            with _signals_blocked(fix):
                fix.setText("" if name not in fixed else f"{float(fixed[name]):g}")
            with _signals_blocked(seed):
                seed.setText("" if (initial is None or index >= len(initial))
                             else f"{float(initial[index]):g}")


def _apply_analysis_state_to_widgets(card, *, action_combo=None, model_combo=None,
                                     fix_seed=None, result_label=None) -> None:
    """Derive ONE surface's Analysis controls from the card's state -- ``config.params['fit_request']``
    presence (the fit) + ``selection_action`` (the ROI).  Shared VERBATIM by the Setting popup and the
    Edit tab so the two surfaces are pure views of the same single source and can never disagree (#8).
    Signals are blocked while re-seeding so a derive never re-fires the action/model handlers."""
    params = card.config.params
    request = params.get("fit_request")
    active = "fit" if request else str(params.get("selection_action") or "none")
    model = str(request.get("model")) if isinstance(request, Mapping) and request.get("model") else ""
    if action_combo is not None:
        index = action_combo.findData(active)
        with _signals_blocked(action_combo):
            action_combo.setCurrentIndex(index if index >= 0 else 0)
    if model_combo is not None:
        if model:
            model_index = model_combo.findData(model)
            if model_index >= 0:
                with _signals_blocked(model_combo):
                    model_combo.setCurrentIndex(model_index)
        # Only GATE the model picker on the action when there IS an action combo driving on/off (the
        # Setting popup); on the Edit tab the picker + Fit button IS the entry, so it stays enabled.
        if action_combo is not None:
            model_combo.setEnabled(active == "fit")
    if fix_seed is not None:
        current_model = model_combo.currentData() if model_combo is not None else model
        fix_seed.set_model(str(current_model or ""))
        fix_seed.seed_from_request(request)
        if action_combo is not None:
            fix_seed.setEnabled(active == "fit")
    if result_label is not None:
        result_label.setText(card._fit_result_text())


def _py_to_text(value) -> str:
    """A Python value as an editable one-line string (confocal python2str): a
    tuple/list keeps its literal form, scalars use repr.  Round-trips through
    ``_text_to_py``."""
    if value is None:
        return ""
    return repr(value)


def _text_to_py(text: str):
    """Parse an edited acquisition-parameter field back to a Python value
    (confocal str2python): literal first, then a plain float, else the string."""
    import ast
    raw = str(text).strip()
    if not raw:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            return float(raw)
        except ValueError:
            return raw

# Board layout (raw px).  The board is a pure PIXEL plane of card AABBs -- there is NO column
# grid.  WIDTH still scales with the size (``cols // 2`` base-widths so 1x4 is wider than 1x2);
# HEIGHT HUGS the plot -- the card is exactly tall enough for its figure + chrome, with NO blank
# padding below (every size hugs like 1x2, #H3i-3).  ``PanelConfig.col`` is the card's pixel X and
# ``row`` is the card's pixel Y; :func:`pack` is the order-driven TOP-LEFT GRAVITY packer that places
# every card at the first free NW slot in list order.  The CARD'S FORMAT (rounded corners, shadow, grey title strip,
# content padding) belongs to the FluentGroupBox COMPONENT (qt_fluent.CARD_PAD / CARD_TITLE_PX,
# the single source); this module only lays cards out.
GRID_UNIT = 8
# The ONE spacing setting (#H3s-F8).  GAP is the UNIFORM clear distance between any two cards on
# every side -- top, bottom, left, right -- AND the board margin from the (0, 0) origin.  It equals
# the HORIZONTAL inter-card gap the user likes: two cards on adjacent base-columns pitched by
# ``_cell_size()[0] + GAP`` sit exactly GAP px apart (and a multi-column card's internal columns are
# joined by the SAME GAP, see ``_card_size``).  Reusing this one existing spacing constant (no new
# public art/geom knob, per frontend/AGENTS.md F2/F3); change this one number to retune all board
# spacing.
GAP = GRID_UNIT

def _cell_size() -> tuple[int, int]:
    """The base CARD WIDTH unit = a 1x2 panel's card: the figure (1 row x 2 cols) plus the card
    chrome (L/R border + grey title strip + bottom border).  Every card width is a whole number of
    these base widths joined by ``GAP``, so widths stay on one rhythm (height hugs the plot)."""

    width = panel_display_size("1x2")[0] + 2 * CARD_PAD
    height = scaled_px(CARD_TITLE_PX) + scaled_px(2) + panel_display_size("1x2")[1] + CARD_PAD
    return (width, height)


def _card_size(size: str) -> tuple[int, int]:
    """Outer card size: WIDTH = ``cols // 2`` base widths joined by ``GAP`` (1x4 is wider than 1x2),
    HEIGHT HUGS the plot (title strip + the size's own figure height + bottom pad, NO blank padding
    -- every size hugs like 1x2).  So width still scales with the size's columns, but height is
    just tall enough for the figure -- the card's bottom edge sits right under the plot."""

    rows, cols = panel_size_cells(size)
    w_units = max(1, cols // 2)
    cw, _ch = _cell_size()
    width = w_units * cw + (w_units - 1) * GAP
    # Height = the SAME chrome the cell uses, around THIS size's figure (not a cell multiple) -> hug.
    height = scaled_px(CARD_TITLE_PX) + scaled_px(2) + panel_display_size(size)[1] + CARD_PAD
    return (width, height)


def _opaque_white_composite(pm):
    """Composite a (possibly transparent, possibly HiDPI) grabbed pixmap onto an opaque WHITE canvas of
    the SAME size AND devicePixelRatio, so the saved PNG is not see-through AND has no blank margin.

    The dpr match is the crux: on a HiDPI screen ``QWidget.grab`` returns a pixmap whose PHYSICAL size is
    ``logical × dpr`` but whose LOGICAL size is ``logical``.  A plain ``QPixmap(pm.size())`` is dpr=1, so
    ``drawPixmap`` paints the pixmap at its smaller LOGICAL size into the top-left and leaves the rest
    blank -- the giant white margin around the panels the user saw on a scaled display.  Carrying the
    pixmap's dpr makes the canvas's logical size equal the pixmap's, so the composite fills it exactly."""
    canvas = QtGui.QPixmap(pm.size())
    canvas.setDevicePixelRatio(pm.devicePixelRatio())
    canvas.fill(QtGui.QColor("#FFFFFF"))
    painter = QtGui.QPainter(canvas)
    painter.drawPixmap(0, 0, pm)
    painter.end()
    return canvas


def _aabb(cfg) -> tuple[int, int, int, int]:
    """The card's pixel AABB ``(x0, y0, x1, y1)`` -- top-left ``(col, row)`` plus its size."""
    w, h = _card_size(cfg.size)
    return (cfg.col, cfg.row, cfg.col + w, cfg.row + h)


def _overlaps_with_gap(box: tuple[int, int, int, int], placed) -> bool:
    """True when ``box`` (an ``(x0, y0, x1, y1)`` AABB), EXPANDED by ``GAP`` on all sides, intersects
    any already-placed card.  Equivalently: the clear distance between ``box`` and a placed card is
    < GAP on the axis where they overlap, so leaving a card exactly GAP away counts as clear."""
    x0, y0, x1, y1 = box
    for p in placed:
        px0, py0, px1, py1 = _aabb(p)
        if x0 < px1 + GAP and px0 < x1 + GAP and y0 < py1 + GAP and py0 < y1 + GAP:
            return True
    return False


class _GeomProxy:
    """A throwaway geometry stand-in (size + packed pixel top-left) that :func:`pack` can place
    WITHOUT mutating a real ``PanelConfig``.  :func:`drop_index` packs proxies to probe where a
    trial order would put a card, so the ONE packer is also the drag oracle (no second rule)."""

    __slots__ = ("size", "col", "row")

    def __init__(self, size: str, col: int = 0, row: int = 0):
        self.size, self.col, self.row = str(size), int(col), int(row)


def _first_free_slot(cfg, placed, board_w: int) -> tuple[int, int]:
    """The TOP-MOST then LEFT-MOST free ``(col, row)`` where ``cfg`` fits clear of every ``placed`` card
    (GAP apart, inside ``board_w``) -- the per-card north-west placement :func:`pack` applies to EVERY
    card in order (so the board tiles the top row left-to-right, wraps to the next shelf, and never
    leaves a middle hole).  Candidate points are GAP (origin) + each placed card's right/bottom edge
    (``+GAP``) and its left/top edge (so a card can tuck under a wider one); swept by y then x, first
    feasible wins."""
    w, _h = _card_size(cfg.size)
    xs = {GAP}
    ys = {GAP}
    for p in placed:
        px0, py0, px1, py1 = _aabb(p)
        xs.add(px1 + GAP)
        ys.add(py1 + GAP)
        xs.add(px0)            # also align left edges, so a card can tuck under a wider one
        ys.add(py0)
    max_x = max(GAP, board_w - GAP - w)
    cand_x = sorted(x for x in xs if GAP <= x <= max_x) or [GAP]
    for y in sorted(ys):
        for x in cand_x:
            if not _overlaps_with_gap((x, y, x + w, y + _h), placed):
                return (x, y)
    # No candidate fit (should not happen -- placing past the lowest card always clears).
    bottom = max((py1 for *_rest, py1 in (_aabb(p) for p in placed)), default=0)
    return (GAP, bottom + GAP if placed else GAP)


def _min_board_width(configs: Sequence["PanelConfig"]) -> int:
    """The NARROWEST a board may pack to: one WIDEST card plus both GAP margins.  A viewport thinner
    than this still has to fit the widest card, so we clamp up to it -- but NOT to the cards' current
    right-extent: clamping to the extent would RATCHET (once cards spread wide the board could never
    pack narrower), so narrowing the window would never reflow into a single column.  At one-card
    width the gravity packer simply stacks every card in one column, which is the correct reflow."""
    widest = max((_card_size(c.size)[0] for c in configs), default=_card_size("1x2")[0])
    return widest + 2 * GAP


def _board_width(configs: Sequence["PanelConfig"]) -> int:
    """A fallback packing width for callers without a live viewport (the pure-function tests): two
    of the WIDEST card side by side plus the GAP margins, so cards CAN pack side by side.  The real
    GUI passes the scroll viewport width to :func:`pack` instead, so the board wraps at the edge."""
    widest = max((_card_size(c.size)[0] for c in configs), default=_card_size("1x2")[0])
    return max(2 * widest + 3 * GAP, _min_board_width(configs))


def pack(order: Sequence["PanelConfig"], board_w: int | None = None) -> bool:
    """The ONE board packer: place each card, IN THE GIVEN LIST ORDER, at the TOP-MOST then LEFT-MOST
    GAP-clear slot (:func:`_first_free_slot`).  Strict north-west gravity as a PURE function of the
    ORDER (the board's single source of truth), the sizes, and ``board_w`` -- it does NOT read any
    card's current pixel position, so it is deterministic and idempotent.

    That is what fixes issue #2.  The old packer floated each card up-left from WHERE IT WAS, gated by
    a fuzzy majority-overlap test and seeded from a pixel-sorted reading order -- so a resize/click
    could converge to a DIFFERENT fixed point (a surprising "reflow" that read as violating top-left
    gravity), and an Add seeded into the first middle HOLE.  Here placement depends only on order:
    the first card lands at ``(GAP, GAP)``; every later card fills the first free NW slot clearing all
    already-placed cards by GAP within ``board_w`` (else drops to a new shelf below).  An Add appended
    LAST therefore always lands in the next bottom slot -- never a middle hole -- and re-packing a
    settled board moves nothing.  A drop REORDERS the list (:func:`drop_index`); pack recomputes every
    pixel from the new order.  ``board_w`` None -> a two-wide headless fallback; a given width is
    honoured but clamped up to one-card-wide.  Returns True if any card's ``(col, row)`` changed."""
    order = list(order)
    board_w = _board_width(order) if board_w is None else max(board_w, _min_board_width(order))
    placed: list = []
    moved = False
    for cfg in order:
        col, row = _first_free_slot(cfg, placed, board_w)
        if (cfg.col, cfg.row) != (col, row):
            cfg.col, cfg.row = col, row
            moved = True
        placed.append(cfg)
    return moved


def drop_index(cfg, others: Sequence["PanelConfig"], board_w: int | None = None) -> int:
    """The ORDER index at which to insert a card DROPPED at its raw pixel ``(cfg.col, cfg.row)`` among
    ``others`` (already in order), so it lands NEAREST the drop point under :func:`pack` gravity.

    This is the user's Z1 rule expressed through the ONE packer instead of a separate placement math:
    for every candidate insertion index we pack a trial order (proxies, so the real configs are never
    mutated) and measure where the dropped card ends up; the index whose resulting top-left is closest
    to the raw drop wins.  A drop near an existing card's slot lands ON it (index before it -> that card
    and everything after shift DOWN the order and re-pack = "displace"); a drop past the last card lands
    at the bottom (append).  Ties -> the earliest index, so dropping squarely onto a card displaces it.
    Pure geometry (no Qt).  Board width None -> the same headless fallback :func:`pack` uses."""
    board_w = _board_width(list(others) + [cfg]) if board_w is None else board_w
    drop_x, drop_y = int(round(cfg.col)), int(round(cfg.row))
    proxies = [_GeomProxy(o.size) for o in others]
    best_i, best_d = 0, None
    for k in range(len(proxies) + 1):
        probe = _GeomProxy(cfg.size)
        trial = proxies[:k] + [probe] + proxies[k:]
        pack(trial, board_w)
        d = (probe.col - drop_x) ** 2 + (probe.row - drop_y) ** 2
        if best_d is None or d < best_d:
            best_d, best_i = d, k
    return best_i


def _task_files_dir() -> Path:
    root = os.environ.get(TASK_FILES_ENV)
    path = Path(root) if root else Path(__file__).resolve().parents[2] / "tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path


#: relim modes (confocal_gui combo_relim naming) -- the SINGLE source for both the Setting
#: popup combo and the Edit-tab combo, so the two never list different options.
#: "fixed" (#8) pins the y-axis / colour-limit to operator-set ``fixed_lo``/``fixed_hi`` bounds
#: (the lo/hi inputs reveal only in that mode); tight/normal autoscale as before.
_RELIM_MODES = ("tight", "normal", "fixed")

#: The relim mode as a declarative ``ParamDecl`` -- so a "plot"-role panel's relim chooser renders
#: through the SAME _make_param_widget / PARAM_WIDGETS path every other plot param uses (one source,
#: auto-injected into BOTH the Setting popup and the Edit tab, #H3v-4b).  Edits route through
#: ``_set_param`` (which pushes the mode onto the live plotter + reveals the fixed lo/hi row).
_RELIM_PARAM = ParamDecl(
    key="relim", label="relim", kind="choice", default="tight", choices=_RELIM_MODES, display=True,
    tooltip="Relim mode (confocal_gui combo_relim naming):\n"
            "  tight  = autoscale hugs the data\n"
            "  normal = autoscale with the matplotlib default margin\n"
            "  fixed  = pin the y-axis / colour-limit to the lo/hi below")

#: Per-panel display refresh intervals (ms) the operator can pick from.  A FIXED, harmonic set
#: (100·{1,2,4,8}) so the SMALLEST selected interval divides every other -- the console timer
#: runs at that base (the GCD) and each panel refreshes every ``update_ms // base`` ticks.  The
#: payoff is PHASE ALIGNMENT: panels that share a beat fire on the SAME tick and read the SAME
#: hub snapshot, so a 2-D frame and its site-map stay shot-coherent; a fast panel (100 ms) just
#: refreshes more often in between (a live-1D alignment monitor).  Limiting the choices to this
#: set is what makes the synchronisation exact -- arbitrary per-panel rates could never co-align.
UPDATE_INTERVALS = (100, 200, 400, 800)
DEFAULT_UPDATE_MS = 400

#: Image containers the Edit-tab Save offers, in menu order (first = default).  A DATA-layer choice of
#: the output FILE FORMAT (not an art knob): the figure's geometry / dpi / typography are unchanged; only
#: the container the ``DataFigure.save(image_ext=...)`` writes changes.  Lowercase to match matplotlib's
#: ``savefig`` extensions and confocal's ``save_type`` naming (jpg / png).  The matching ``.npz`` data
#: file is format-independent -- identical arrays + ``info`` for every choice.
SAVE_IMAGE_FORMATS: tuple[str, ...] = ("png", "pdf", "jpg")

#: Debounce window (ms) for coalescing a burst of STRUCTURE-knob edits into ONE plotter rebuild.  A
#: fast scroll on a rebuild-class param (history length, colormap, ...) fires many _set_param calls; each
#: restarts this timer, so the (heavy) teardown+rebuild runs ONCE after the burst settles, on the latest
#: values -- removing the per-tick rebuild race.  Short enough that a single deliberate edit still feels
#: instant; long enough that a wheel spin coalesces.
REBUILD_DEBOUNCE_MS = 90


def _unit_df_for(plotter):
    """A :class:`DataFigure` bound to ``plotter``'s figure/axes whose unit is inferred from
    the LIVE x-axis label (NOT ``live_plot=``, which would mask a unit-bearing label like
    'Detuning (GHz)').  ``change_unit`` operates on ``ax.lines``, so cycling rewrites that
    figure's curve in place.  Shared by the Setting popup (the live card) and the Edit-tab
    snapshot so the x-axis unit cycle is ONE implementation."""
    from .data_figure import DataFigure
    ax = getattr(plotter, "ax", None)
    unit = DataFigure._infer_unit(ax.get_xlabel()) if ax is not None else None
    return DataFigure(fig=plotter.fig, ax=ax, data_x=plotter.data_x, data_y=plotter.data_y,
                      labels=getattr(plotter, "labels", None), unit=unit)


# ====================================================================== state
class PanelConfig:
    """One panel: kind + a size PRESET + its pixel top-left on the board (``col`` = pixel x,
    ``row`` = pixel y).  ``_compact`` re-packs these top-left under gravity (no column grid)."""

    def __init__(
        self,
        *,
        kind: str,
        title: str = "",
        row: int = 0,
        col: int = 0,
        size: str = "2x2",
        source: str | None = None,
        params: Mapping[str, object] | None = None,
        inputs: Sequence[str] | None = None,
        role: str = "plot",
    ):
        if kind not in PANEL_KINDS:
            raise ValueError(f"unknown panel kind {kind!r}; choose from {sorted(PANEL_KINDS)}.")
        if role not in PANEL_ROLES:
            raise ValueError(f"unknown panel role {role!r}; choose from {list(PANEL_ROLES)}.")
        panel_size_cells(size)              # validate against the limited preset list
        self.kind = str(kind)
        self.title = str(title)
        self.row = max(0, int(row))    # pixel y of the card top-left (no column grid)
        self.col = max(0, int(col))    # pixel x of the card top-left (no column grid)
        self.size = str(size)
        # The per-slot signal names (signal[0], signal[1], ...): one hub signal per input
        # slot of this plot kind.  Defaults to each slot's default signal so a freshly
        # added panel already names what it wants; a saved layout restores its picks.
        slots = panel_input_slots(self.kind)
        if inputs is None:
            self.inputs = [d for _, d, _ in slots]
        else:
            self.inputs = [str(s) for s in inputs]
            if len(self.inputs) < len(slots):       # pad to the kind's slot count
                self.inputs += [d for _, d, _ in slots[len(self.inputs):]]
        # A pure-view plot is BLANK until a signal is picked (decoupled from acquisition).
        # When the input already names a signal, the default source is ``value = signal``;
        # an empty input leaves it blank ("pick a signal").  A saved layout keeps its
        # stored source verbatim.
        if source is not None:
            self.source = str(source)
        elif self.inputs and self.inputs[0]:
            self.source = "value = signal"
        else:
            self.source = _BLANK_SOURCE
        # A source written as a bare ``value = <hub signal>`` (a saved layout or a direct
        # pick) NAMES the input, so reflect it in the input slot -- the picker and the source
        # must never disagree.  The canonical ``value = signal`` and an expression
        # (``value = np.log(f)``) do NOT match this and leave the input alone.  The ``len == 1``
        # guard mirrors :func:`is_identity_source`'s SAME guard (a bare name only names the input
        # when there is exactly ONE slot -- a 2-slot ``value = counts`` is a custom expression, not a
        # naming): so once this backfill runs the two agree BY CONSTRUCTION, and the structure-
        # passthrough gate (:meth:`_bound_structure`, which calls is_identity_source) never disagrees
        # with the picker on what "just names a signal" means.
        from ..neutral_atom.operations.signal_expr import IDENTITY_SOURCE_RE
        m = IDENTITY_SOURCE_RE.fullmatch(self.source.strip())
        if m and m.group(1) != "signal" and len(self.inputs) == 1:
            self.inputs[0] = m.group(1)
        self.params = dict(params or {})
        self.role = str(role)

    @property
    def update_ms(self) -> int:
        """This panel's display refresh interval (ms), one of :data:`UPDATE_INTERVALS`.
        Stored in ``params`` (so it round-trips with the saved layout); an out-of-set value
        falls back to :data:`DEFAULT_UPDATE_MS` so the timer base stays harmonic."""
        ms = int(self.params.get("update_ms", DEFAULT_UPDATE_MS) or DEFAULT_UPDATE_MS)
        return ms if ms in UPDATE_INTERVALS else DEFAULT_UPDATE_MS

    def to_dict(self) -> dict[str, object]:
        # ``row``/``col`` are the card's pixel top-left (no column grid).  They are only a SEED for
        # the gravity packer, which re-packs the whole board on load, so a layout's reading order
        # (top-to-bottom, left-to-right) is what round-trips -- exact pixels are recomputed.
        return {
            "kind": self.kind,
            "title": self.title,
            "row": self.row,
            "col": self.col,
            "size": self.size,
            "source": self.source,
            "params": dict(self.params),
            "inputs": list(self.inputs),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PanelConfig":
        return cls(
            kind=str(payload["kind"]),
            title=str(payload.get("title", "")),
            row=int(payload.get("row", 0)),
            col=int(payload.get("col", 0)),
            size=str(payload.get("size", "2x2")),
            source=payload.get("source"),
            params=payload.get("params") or {},
            inputs=payload.get("inputs"),
            role=str(payload.get("role", "plot")),
        )


class LogicNodeConfig:
    """One LOGIC NODE: which node it is + the param values to build it with.

    A logic node lives on the Logic tab, NOT the Monitor board, and is the thing
    that PRODUCES data.  ``kind`` is one of :data:`LOGIC_KINDS` (camera /
    measurement / processor / task); ``name`` is the catalog spec's name (the
    camera's is ``"live"``; its display TITLE comes from ``readout.camera_spec().name``).
    ``values`` is the last param-form ``{key: value}`` it was built / run with, so
    reopening its Edit restores them.  A node is always added STOPPED -- nothing
    runs until Start in its Edit."""

    def __init__(self, *, kind: str, name: str, title: str = "",
                 values: Mapping[str, object] | None = None):
        if kind not in LOGIC_KINDS:
            raise ValueError(f"unknown logic kind {kind!r}; choose from {list(LOGIC_KINDS)}.")
        self.kind = str(kind)
        self.name = str(name)
        self.title = str(title) or str(name)
        self.values = dict(values or {})

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "name": self.name, "title": self.title,
                "values": dict(self.values)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogicNodeConfig":
        return cls(
            kind=str(payload["kind"]),
            name=str(payload.get("name", "")),
            title=str(payload.get("title", "")),
            values=payload.get("values") or {},
        )


class TaskConsoleState:
    """The whole console layout: serialised as ONE machine-portable JSON file."""

    schema = "Zou_lab_control.frontend.TaskConsoleState"
    version = 3

    def __init__(
        self,
        *,
        name: str = "task",
        interval_ms: int = 400,
        panels: Sequence[PanelConfig | Mapping[str, object]] | None = None,
        logic: Sequence[LogicNodeConfig | Mapping[str, object]] | None = None,
    ):
        self.name = str(name)
        self.interval_ms = max(50, int(interval_ms))
        self.panels = [
            panel if isinstance(panel, PanelConfig) else PanelConfig.from_dict(panel)
            for panel in (panels or [])
        ]
        # The Logic-tab nodes (measurement / processor / task), saved alongside the
        # plot panels so a layout restores the whole dashboard -- nodes always come
        # back STOPPED (the layout records what to build, not a running thread).
        self.logic = [
            node if isinstance(node, LogicNodeConfig) else LogicNodeConfig.from_dict(node)
            for node in (logic or [])
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "name": self.name,
            "interval_ms": self.interval_ms,
            "panels": [panel.to_dict() for panel in self.panels],
            "logic": [node.to_dict() for node in self.logic],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TaskConsoleState":
        return cls(
            name=str(payload.get("name", "task")),
            interval_ms=int(payload.get("interval_ms", 400)),
            panels=payload.get("panels") or [],
            logic=payload.get("logic") or [],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TaskConsoleState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != cls.schema:
            raise ValueError(f"{path} is not a task-console layout (schema={payload.get('schema')!r}).")
        return cls.from_dict(payload)


def default_console_state() -> TaskConsoleState:
    """The console opens EMPTY.  You build the dashboard yourself (Add Panel) and
    Save it to a file; you reuse it by Loading that file back (the Load button /
    ``--task`` / ``show_task_console(task=)``)."""

    return TaskConsoleState(name="task", panels=[])


def resolve_task_state(task: str) -> TaskConsoleState:
    """Resolve a task layout by FILE PATH or a saved tasks/<name>.json."""

    text = str(task).strip()
    path = Path(text)
    if path.suffix.lower() == ".json" and path.exists():
        return TaskConsoleState.load(path)
    saved = _task_files_dir() / f"{text}.json"
    if saved.exists():
        return TaskConsoleState.load(saved)
    raise ValueError(
        f"unknown task {task!r}: not a layout file and not a saved layout in {_task_files_dir()}.")


# ====================================================================== panels
_MONITOR_UNSET = object()   # sentinel: a monitor panel that has never rolled yet


class _GridFocus:
    """Parked GRID state while one of its cells is ENLARGED into a standalone plot-kind figure in the
    console.  Holds the grid plotter + its (detached, not closed) canvas so :meth:`PanelCard._unfocus_grid_cell`
    can swap the grid back, the focused cell index (to mirror a threshold drag back onto it), and the mpl
    callback ids on the focus canvas so they are disconnected on return / teardown."""

    def __init__(self, *, grid, grid_canvas, k: int):
        self.grid = grid
        self.grid_canvas = grid_canvas
        self.k = int(k)
        self.click_cid = None
        self.key_cid = None
        #: A Setting edit was persisted onto the (parked) grid while it was zoomed -> its buffer is stale,
        #: so unfocus must re-render.  Left False the common "just look and go back" case keeps the grid's
        #: already-rendered buffer (fit + thumbnails) and unfocus is an instant re-blit, never a re-render.
        self.dirty = False


class PanelCard(FluentGroupBox):
    """One dashboard panel: a TITLED frame (title strip = the panel KIND + the signal-source
    legend, top-left) holding the frontend canvas, and a text
    "Setting" button on the title strip (top-right).  The frame border is the DRAG
    HANDLE (the matplotlib canvas keeps all its own interactions); the card
    spans whole layout slots -- a 2-row card is exactly two
    1-row cards plus the gap."""

    changed = QtCore.pyqtSignal()          # any config edit (console marks dirty)
    layout_changed = QtCore.pyqtSignal()   # size/slot change (console re-arranges)
    dropped = QtCore.pyqtSignal(object)    # drag-release ONLY (console snaps the drop to its nearest anchor)
    update_interval_changed = QtCore.pyqtSignal()  # per-panel refresh rate change (console re-bases the timer)
    remove_requested = QtCore.pyqtSignal(object)
    edit_requested = QtCore.pyqtSignal(object)   # "Edit…" -> open the panel's Edit tab

    def __init__(self, config: PanelConfig, parent=None, *, names_provider=None,
                 sources_provider=None, formats_provider=None, axes_provider=None,
                 sites_inputs_provider=None, curve_x_provider=None,
                 structure_provider=None, short_names_provider=None,
                 live_namespace_provider=None, pulse_state_provider=None,
                 grid_recipe_provider=None, render_barrier=None, area_select_sink=None,
                 selection_clear_sink=None, fit_node_sink=None):
        # Titled frame: the title strip carries the panel KIND (top-left) and the
        # Setting button (top-right), so the card is delineated like the rest.
        super().__init__(PANEL_KINDS[config.kind], parent)
        self.config = config
        self.names_provider = names_provider   # callable -> live signal names (Setting combo)
        # callable -> {signal name: [source node labels]}, so the picker can show
        # WHICH measurement/processor each signal comes from (not just bare names).
        self.sources_provider = sources_provider
        # callable -> {signal name: array-format}, so the picker also shows each
        # signal's SHAPE (e.g. occupied -> per-site (N,)).
        self.formats_provider = formats_provider
        # callable -> {signal name: (axis_label, unit)} from the PRODUCING node's
        # SignalSpec, so a plot reads its y-axis label/unit from the measurement that
        # makes the signal (confocal: the measurement owns its labels), not a per-kind
        # hard-coded string.
        self.axes_provider = axes_provider
        # callable -> {full hub signal: SHORT name} (the producing node's prefix stripped), so the
        # picker NEST shows the short name (frame / survival / rate) -- never the full prefixed key
        # nor the verbose SignalSpec axis label.  ONE rule, shared with the Logic tab.
        self.short_names_provider = short_names_provider
        # callable(occupancy_signal) -> (centres_signal, image_signal): the site map takes
        # ONE signal (occupancy) and resolves its centres + frame underlay from the SAME
        # producing node (via that node's spec metadata), so rings + underlay are one shot.
        self.sites_inputs_provider = sites_inputs_provider
        # callable(y_signal) -> companion x_signal: a 1d plot wired to a scan's y curve
        # draws it vs the swept x from the SAME producing node (one signal pick, #3).
        self.curve_x_provider = curve_x_provider
        # callable(signal) -> (points_shape, data_shape) from the producing node's output contract,
        # so the plot auto-reshapes by the DATA dimensionality (1-D data -> multiple lines; 2-D data
        # -> reshape/imshow) instead of guessing from sizes (#H3o).
        self.structure_provider = structure_provider
        # callable() -> the console's CURRENT shared per-tick namespace (a fresh hub snapshot).  An
        # IMMEDIATE re-render (signal switch / colormap / resize) must draw from THIS -- the SAME shot
        # every other panel is on this tick -- NOT the panel's own stale ``_last_namespace`` (a PAST
        # tick).  Otherwise after switching a panel's source it shows an OLDER shot than its siblings,
        # so e.g. a 2D image and the sitemap display different shots (#shot-coherence-on-switch).
        self.live_namespace_provider = live_namespace_provider
        # callable(value_signal) -> (PulseTableState, include_always_off): a PULSE panel resolves its
        # reproduction state off the SAME producing node (the object is carried on the node, not the
        # float-only hub) -- the SAME "aux data from the producing node" wiring the site map uses.
        self.pulse_state_provider = pulse_state_provider
        # callable(value_signal) -> grid recipe dict: a GRID panel resolves its replay recipe off the SAME
        # producing node the SAME way a pulse panel resolves its state (a dict carried on the node, the
        # float-only hub cannot hold it).
        self.grid_recipe_provider = grid_recipe_provider
        # callable() -> waits until the console's render worker is idle.  EVERY GUI-thread path
        # that mutates this card's figure/plotter (a Setting edit, a source apply, the coalesced
        # rebuild, the selectors switch, teardown) must hold this barrier first -- while a batch
        # is in flight the render thread owns the figure (see frontend/render_loop.py).  None
        # (a standalone/test card) makes it a no-op via _wait_render_idle.
        self.render_barrier = render_barrier
        # callable(card, (x_min, x_max, y_min, y_max)) -> the console's ROI-chain sink: a
        # rectangle drawn on this LIVE image panel retargets/creates a RoiProcessor consuming
        # this panel's signal (the "draw a box -> get roi_frame/roi_value signals" gesture).
        self.area_select_sink = area_select_sink
        # callable(card, action) -> the console's ONE selection-teardown sink, symmetric for BOTH
        # analyses (#10): leaving/clearing an analysis STOPS + removes the hub node it created for this
        # card's signal -- a FitProcessor for "fit" (clearing the curve fit / picking a non-fit action),
        # a RoiProcessor for "roi" (switching the ROI action off).  Both have a full create-on-apply /
        # remove-on-clear lifecycle, so neither lingers as an orphan republishing after the operator
        # turned its analysis off.
        self.selection_clear_sink = selection_clear_sink
        # callable(card, request) -> the console's ONE fit-node sink: create OR retarget the hub
        # FitProcessor that publishes a 2-D image fit's parameters as signals (fit_x0/... ), for the
        # non-grid image case.  Its teardown counterpart is ``selection_clear_sink(card, "fit")``.  The
        # ONE mutator :meth:`set_fit_request` calls this, so the overlay + the hub node stay in lockstep.
        self.fit_node_sink = fit_node_sink
        self.plotter = None
        self.canvas = None
        # The console header's "Selectors" switch state for THIS card (set via
        # ``set_selectors_enabled``; default OFF = the historical display-only Monitor board).
        # Every plotter (re)build parks its selector layer to this flag (``_apply_selectors_state``),
        # so a fresh figure always inherits the switch instead of coming up live.
        self._selectors_on = False
        # Last DATA selection made on this panel.  It is plot-independent and
        # serializable; selecting never implies an ROI or a fit.  The explicit
        # ``selection_action`` config decides what a later release does.
        self._active_selection = None
        # GRID focus-zoom state (console path).  A GRID panel is display-only (interactions=False), so its
        # own double-click handler is dormant; THIS card catches the double-click on the grid canvas and
        # swaps ``self.plotter`` / ``self.canvas`` to a STANDALONE plot-kind figure of the clicked cell
        # (``GridPlot.build_focus_plotter`` -- the SAME builder the notebook path uses).  While focused,
        # ``self.plotter`` IS that standalone figure, so a lim / fit / relim edit reaches it through the
        # ORDINARY _set_param / _apply_lim_to_plotter path (no bespoke code) and the live tick is gated OFF
        # so the grid never rebuilds over it (no bounce-back).  ``None`` = showing the grid; else a
        # ``_GridFocus`` holding the parked grid plotter/canvas + the focused cell index + the click/key cids.
        self._grid_focus = None
        # The TOP spacer that centres the stock-size (2x2) enlarged cell vertically inside a larger grid
        # region while focused -- inserted by _focus_grid_cell, removed on unfocus/teardown.
        self._focus_top_stretch = None
        # The cell index to AUTO RE-FOCUS after a teardown+rebuild (size / source / structure-param
        # change) that happened WHILE a grid cell was enlarged: _teardown_plot records the focused index
        # here, _build_plot's grid branch re-focuses it -- so a rebuild-class edit lands the operator back
        # on the SAME enlarged cell (at the new size/params) instead of silently bouncing to the main
        # grid view (#no-focus-bounce).  ``None`` = nothing to restore.  An EXPLICIT unfocus (double-click
        # / Esc) never passes through _teardown_plot, so it never re-focuses.
        self._pending_refocus_k: int | None = None
        self._value_shape: tuple[int, ...] | None = None
        # A STRUCTURE knob (bins / colormap / fit ...) sets this to force the NEXT _render to REBUILD
        # the plotter -- WITHOUT pre-tearing it down.  _build_plot build-then-swaps, so a failed rebuild
        # keeps the OLD figure (the card never goes blank) -- the root fix for the "scroll bins and the
        # figure occasionally vanishes" race, where a pre-null teardown could latch plotter=None.
        self._force_rebuild = False
        # ANTI-RACING rebuild debounce: a STRUCTURE-knob edit that needs a rebuild schedules it on this
        # single-shot timer instead of rebuilding inline.  A fast burst of edits (scrolling the history
        # spin / a colormap combo) RESTARTS the timer, so only ONE rebuild runs after the burst settles
        # -- the latest config.params values.  This is GENERAL (every rebuild-class param, not history
        # only): coalescing the rebuilds removes the race where per-tick teardown+rebuild outran the Qt
        # holder reflow (the figure flickered / grew / vanished).  _build_plot itself is atomic
        # build-then-swap at a fixed canvas size, so even a rebuild that DOES run never blanks the card.
        self._rebuild_timer = QtCore.QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(REBUILD_DEBOUNCE_MS)
        self._rebuild_timer.timeout.connect(self._run_pending_rebuild)
        # The LAST namespace this panel rendered from (a reference to the hub
        # snapshot of that tick).  A display-only change (cmap / relim / source pick)
        # tears the plotter down and normally waits for the NEXT hub tick to rebuild --
        # but a STOPPED measurement freezes hub.version, so _tick's gate never calls
        # refresh again and the panel would stay blank (white).  Re-rendering from this
        # cached namespace makes such a change take effect immediately, stopped or not.
        self._last_namespace: Mapping[str, object] | None = None
        # ``_repeat_cur`` = how many repeats of the measurement's block currently have data (drives
        # the plotter's "xN" ylabel); set by _signal_then_repeat when it reduces the repeat axis.
        self._repeat_cur: int = 1
        # The hub version at this panel's LAST render -- the per-panel multi-rate refresh
        # (see TaskConsole._tick) skips a panel on its beat when nothing new was published
        # since, so a slow panel does not redraw stale data and a fast one only when needed.
        self._render_version = -1
        # Fairness under overload: True when this panel's beat fell on a tick the render
        # worker was busy (or the panel was mid-drag) -- the next idle tick serves it
        # regardless of the beat modulo, so a slow-beat panel can never phase-lock onto
        # busy ticks and starve behind a heavy fast-beat sibling (see TaskConsole._tick).
        self._beat_owed = False
        self._compiled_source = config.source
        # Monitor roll-gate: remembers the per-signal version of this panel's
        # source at the last roll, so an unrelated node's version bump does not
        # append a duplicate point.  `_MONITOR_UNSET` = never rolled yet.
        self._last_monitor_key: object = _MONITOR_UNSET
        self._ref_src: tuple | None = None      # (compiled_source, inputs) the names were derived from
        self._ref_names: frozenset = frozenset()
        # 2D coordinate frame: the ROI the axes were built from, so a ROI that
        # SHIFTS (same shape, new origin) still triggers an axes rebuild.
        self._roi_built: list | None = None
        self._drag_offset: QtCore.QPoint | None = None
        self.setCursor(QtCore.Qt.OpenHandCursor)   # the frame border drags

        holder = QtWidgets.QVBoxLayout(self)
        # The card's content padding is the component-library token CARD_PAD (L/R + bottom); the
        # grey title strip is above (the FluentGroupBox padding-top), and the bottom pad makes the
        # height proportional (see _card_size).  No footer -- the signal source lives in the title.
        holder.setContentsMargins(CARD_PAD, scaled_px(2), CARD_PAD, CARD_PAD)
        holder.setSpacing(0)
        self.canvas_holder = holder
        # The transient status (the Setting tooltip / button colour) and the persistent SIGNAL
        # legend (which node each read comes from) -- the legend goes into the frame TITLE; the
        # per-shot status no longer takes panel space.
        self._status_text = ""
        self._signal_info = ""

        self._build_settings()

        self.setting_button = FluentButton("Setting", color=GREY)
        self.setting_button.setParent(self)
        self.setting_button.setFixedSize(scaled_px(74, minimum=64), scaled_px(26, minimum=22))
        self.setting_button.setToolTip("Panel settings")
        self.setting_button.clicked.connect(self._open_settings)

        self._apply_fixed_size()
        self.set_status("waiting for data…", error=False)

    # ------------------------------------------------------------- geometry
    def _apply_fixed_size(self) -> None:
        """Card size = whole layout slots (the footer absorbs the slack)."""
        self.setFixedSize(*_card_size(self.config.size))
        self._place_setting_button()

    def _place_setting_button(self) -> None:
        if hasattr(self, "setting_button"):
            # top-right, on the title strip (the title kind sits top-left).
            self.setting_button.move(
                self.width() - self.setting_button.width() - scaled_px(8),
                scaled_px(4))
            self.setting_button.raise_()

    # ------------------------------------------------------------- settings UI
    def _build_settings(self) -> None:
        """Confocal-style VERTICAL settings popup.

        Layout idiom borrowed from ``Confocal_GUIv2/gui/gui_individual.py``'s
        ``BaseLivePlotGUI`` settings page: one section header per group (bold,
        own line), then one control per row underneath the header as a
        ``[fixed-width label | control]`` cell.  Five sections stacked top-to-
        bottom:

        * **Source** -- pick a signal or write an expression + ``Apply``.
        * **Display** -- size + (for image kinds) colormap (the COLORSET
          chooser; there is no separate cbar show/hide toggle), plus each
          kind's declarative ``PANEL_PARAMS`` widget.
        * **Unit** -- ``Unit`` cycle button + current unit label.
        * **Limits** -- ``auto``/``manual`` mode combo + a 3-column mini-grid
          of ``[axis | lo | hi]`` rows (axis labels left, lo/hi headers top).
        * **Panel** -- title edit, then a single row of
          ``Remove | Edit… | <stretch> | Save Fig``.

        Every edit is LIVE-applied -- there is no global ``Apply`` button for
        the popup itself (Source has its own Apply because the expression is
        validated separately).  Lifted helpers ``FluentSectionLabel`` and
        ``FluentSettingRow`` (in ``qt_fluent.py``) own the visual rhythm so
        future settings popups stay identical."""

        # The rounded card is painted by FluentPopup (translucent, frameless),
        # NOT a stylesheet border-radius on an opaque popup -- that left a
        # square white nub past the arc at the corners + a native popup shadow.
        # Painting the card also means no border stylesheet rule can cascade
        # onto child labels/controls.
        popup = FluentPopup(self)
        # The sections (Source / Display / Unit / Limits / Panel) can stack taller than the
        # screen when a panel has many signals -> wrap them in a FluentScrollArea so the popup
        # never exceeds the visible area (it scrolls instead of running off / overflowing the
        # panel).  The scroll viewport is transparent so FluentPopup's painted rounded card
        # still shows through; the height is capped in _open_settings.
        popup_outer = QtWidgets.QVBoxLayout(popup)
        popup_outer.setContentsMargins(0, 0, 0, 0)
        self._settings_scroll = FluentScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._settings_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        # NEVER a horizontal scrollbar -- the popup widens to fit its content instead (the rows
        # are narrow; only the HEIGHT is capped, below).  Vertical scroll only when the sections
        # are taller than the panel they belong to.
        self._settings_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._settings_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        popup_content = QtWidgets.QWidget()
        popup_content.setStyleSheet("background: transparent;")
        self._settings_scroll.setWidget(popup_content)
        popup_outer.addWidget(self._settings_scroll)
        root = QtWidgets.QVBoxLayout(popup_content)
        pad = scaled_px(10)
        # RESERVE the vertical scrollbar's width on the RIGHT so the scrollbar never sits ON TOP
        # of a control (it would otherwise occlude the combo / value boxes when content scrolls).
        # The popup anchors its RIGHT edge at the gear, so this extra width grows the popup
        # LEFTWARD into the empty space there -- exactly "extend the settings frame to the left".
        scrollbar_w = fluent_scrollbar_thickness()   # ONE source for the bar width (not a hand-typed 12)
        root.setContentsMargins(pad, pad, pad + scrollbar_w + scaled_px(4), pad)
        root.setSpacing(scaled_px(10, minimum=6))

        # fixed left-label column, SIZED TO CONTENT so nothing truncates: the longest
        # label is normally "colormap", but a multi-input plot adds slot rows like
        # "signal[0] occupancy" -- measure every label this popup will show and widen the
        # column to the widest (floored at 80 px), so all rows still align and read fully.
        base_slots = panel_input_slots(self.config.kind)
        # EVERY plot kind uses the SAME source MECHANISM: a signal picker + a ``value = ...``
        # expression (the reusable SignalExpr).  Whether a kind can GROW extra slots
        # (+signal / −signal) is data-driven from the per-kind declaration
        # (``panel_allows_multi_slot`` / PANEL_SINGLE_SLOT_KINDS), NOT an inline ``kind == 'sites'``
        # check: a site map takes EXACTLY ONE occupancy signal (its ring centres + frame underlay
        # resolve from signal[0]'s producing node, so a 2nd slot is meaningless) -> single picker,
        # no +/-, but it STILL has the expression box.  The per-kind value SHAPE is enforced in
        # _coerce (PANEL_INPUT_FORMAT): sites = a per-site (N,) vector, 2D = an (H×W) frame, etc.
        self._multi_slot = panel_allows_multi_slot(self.config.kind)
        n_slots = max(1, len(base_slots), len(self.config.inputs)) if self._multi_slot else max(1, len(base_slots))
        slot_labels = [(f"signal[{i}]" if n_slots > 1 else "signal") for i in range(n_slots)]
        slot_tips = [base_slots[i][2] if i < len(base_slots)
                     else f"an added signal slot — read as signal[{i}] in the expression"
                     for i in range(n_slots)]
        fm = self.fontMetrics()
        widest = max((fluent_text_width(fm, t) for t in [*slot_labels, "colormap", "threshold"]), default=0)
        label_w = max(scaled_px(80, minimum=56), widest + scaled_px(10))

        def section_box(title: str) -> QtWidgets.QVBoxLayout:
            """Header label + inner VBox; rows added to the inner VBox stack
            tightly under the header (own-line, vertical, confocal style)."""
            root.addWidget(FluentSectionLabel(title))
            inner = QtWidgets.QVBoxLayout()
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(scaled_px(6, minimum=4))
            root.addLayout(inner)
            return inner

        # ---- Source --------------------------------------------------------
        sec = section_box("Source")
        # Tell the experimenter what SHAPE of signal this plot kind expects, so the
        # signal picker / expression below is unambiguous (single source PANEL_INPUT_FORMAT).
        accepts = PANEL_INPUT_FORMAT.get(self.config.kind)
        if accepts:
            accepts_label = FluentLabel(f"accepts {accepts}")
            accepts_label.setWordWrap(True)
            accepts_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            sec.addWidget(accepts_label)
        # ONE signal picker: a combobox of the live hub signals.  Picking one sets the
        # source to ``value = signal`` (the canonical form -- the picked signal IS ``signal``
        # in the expression).  A site map needs only its occupancy signal; its ring centres
        # + frame underlay are resolved from the SAME producing node, never extra slots.
        self.slot_combos: list = []
        for i in range(n_slots):
            combo = FluentTreeComboBox()             # collapsible-tree signal picker (G2)
            combo.setToolTip(slot_tips[i] + (
                "\nSets the source to `value = signal`." if n_slots == 1
                else f"\nRead as signal[{i}] in the expression (e.g. value = signal[0] - signal[1])."))
            combo.activated.connect(lambda _ix, idx=i: self._on_slot_pick(idx))
            self.slot_combos.append(combo)
            sec.addWidget(FluentSettingRow(slot_labels[i], combo, label_width=label_w))
        self.signal_combo = self.slot_combos[0]    # the (first) signal picker
        # Add / remove signal slots (every kind except the site map): so a panel can combine
        # several signals -- e.g. plot the DIFFERENCE of two occupancies with signal[0]-signal[1].
        if self._multi_slot:
            self.add_slot_button = FluentButton("+ signal", color=GREY)
            self.add_slot_button.setToolTip("Add another signal slot (read as signal[i] in the expression).")
            self.add_slot_button.clicked.connect(self._add_signal_slot)
            self.remove_slot_button = FluentButton("− signal", color=GREY)
            self.remove_slot_button.setToolTip("Remove the last signal slot.")
            self.remove_slot_button.clicked.connect(self._remove_signal_slot)
            self.remove_slot_button.setEnabled(n_slots > 1)
            slot_btn_row = QtWidgets.QHBoxLayout()
            slot_btn_row.setContentsMargins(0, 0, 0, 0)
            slot_btn_row.setSpacing(scaled_px(6, minimum=4))
            slot_btn_row.addWidget(self.add_slot_button, 0)
            slot_btn_row.addWidget(self.remove_slot_button, 0)
            slot_btn_row.addStretch(1)
            sec.addLayout(slot_btn_row)

        self.source_edit = FluentLineEdit(self.config.source)
        self.source_edit.setMinimumWidth(scaled_px(280, minimum=220))
        self.source_edit.setStyleSheet(
            self.source_edit.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
        self.source_edit.setToolTip(SOURCE_EXPR_HELP())
        # An inline one-liner is cramped for a real expression; "Edit…" pops a LARGE floating
        # multi-line editor (a modal card, so it does NOT reflow the panel layout) prefilled
        # with the source -- comfortable for typing math across signals -- and writes it back.
        # The "Edit…" wording matches the panel's other Fluent "Edit…" buttons (one house style).
        self.expand_button = FluentButton("Edit…", color=GREY)
        self.expand_button.setFixedWidth(scaled_px(56, minimum=44))
        self.expand_button.setToolTip("Open a large floating editor for this expression")
        self.expand_button.clicked.connect(self._open_expr_editor)
        self.apply_button = FluentButton("Apply", color=GREEN)
        self.apply_button.setFixedWidth(scaled_px(64, minimum=52))
        self.apply_button.clicked.connect(self._apply_source)
        self.source_edit.textChanged.connect(lambda: self.apply_button.set_dirty(True))
        self.source_edit.returnPressed.connect(self._apply_source)
        expr_row = QtWidgets.QHBoxLayout()
        expr_row.setContentsMargins(0, 0, 0, 0)
        expr_row.setSpacing(scaled_px(6, minimum=4))
        expr_row.addWidget(self.source_edit, 1)
        expr_row.addWidget(self.expand_button, 0)
        expr_row.addWidget(self.apply_button, 0)
        sec.addLayout(expr_row)

        self.status = FluentLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        sec.addWidget(self.status)

        # ---- Display -------------------------------------------------------
        sec = section_box("Display")
        self.size_combo = FluentComboBox()
        self.size_combo.addItems(list(PANEL_SIZES))
        self.size_combo.setCurrentText(self.config.size)
        self.size_combo.setToolTip("Panel size preset (height x width half-units; 2x2 = the stock plot region)")
        self.size_combo.currentTextChanged.connect(self._on_size)
        sec.addWidget(FluentSettingRow("size", self.size_combo, label_width=label_w))

        # The plot's DISPLAY knobs are DECLARATIVE ParamDecls rendered through the SAME _make_param_widget
        # / PARAM_WIDGETS path everywhere (#H3v-4b): the per-kind colormap / toggles (display=True) for
        # EVERY role, plus -- on a "plot"-role panel -- the relim chooser.  Adding a plot display
        # ParamDecl here makes it appear in the Edit tab too with NO hand-wiring (both call
        # _emit_param_rows).  (size + colormap stay for every role; relim/repeat/unit/update are
        # plotter-only conveniences, "plot" role only -- restyle a measurement by Adding a Plot panel.)
        self.param_widgets: dict[str, QtWidgets.QWidget] = {}
        # {key: kind} for every declarative Setting control, so refresh_on_show can re-seed each widget
        # from config.params through its kind's PARAM_WIDGETS.write (one source -- no per-key handwiring).
        self._param_kinds: dict[str, str] = {}
        # remember which kind's params this popup baked -- when a grid panel's RESOLVED per-cell
        # kind changes later (facet / sub-plot pick, a signal bind), _sync_settings_param_rows
        # rebuilds the popup so the rows below are never a stale bake of the old kind.
        self._settings_param_kind = self._param_kind()
        display_specs = [s for s in PANEL_PARAMS.get(self._settings_param_kind, ()) if s.display]
        if self.config.role == "plot":
            display_specs = display_specs + [_RELIM_PARAM]
        self.param_widgets.update(
            self._emit_param_rows(display_specs, sec.addWidget, self._set_param, label_w))
        self.lim_combo = self.param_widgets.get("relim")     # named back-ref (relim is now declarative)

        if self.config.role == "plot":
            # ``repeat_mode`` (the DISPLAY collapse) is the ONLY repeat knob the plot owns: how to
            # collapse a measurement's repeat axis for display (average / add / replace / roll / create).
            # How MANY repeats lives on the MEASUREMENT (``repeat``, 0 = ∞, auto-injected by
            # _acquisition_param_decls -- #H3l), NOT here.  Rendered through the SAME _make_param_widget
            # path as every other param (no hand-placed widget); edits route through _set_param.
            for spec in self._repeat_param_specs():
                widget = self._make_param_widget(spec)
                self.param_widgets[spec.key] = widget
                self._param_kinds[spec.key] = spec.kind    # remember for refresh_on_show re-seed
                sec.addWidget(FluentSettingRow(spec.label, widget, label_width=label_w))

            # fixed lo/hi inputs (#8): ONE bespoke [lo | hi] row (the single special-cased control,
            # shown only when relim == "fixed") -- built by the shared helper the Edit tab also uses.
            self.fixed_lim_row, self.fixed_lo_edit, self.fixed_hi_edit = \
                self._make_fixed_lim_row(self._on_fixed_lim_edited, label_w)
            sec.addWidget(self.fixed_lim_row)

            # FACET chooser (grid panels only): which axis of the bound (R,P,*data_shape)
            # block expands into the cells -- the grid-as-axis-expander declaration.  Options derive
            # from the producing node's declared structure and refresh on every Setting open (the
            # signal-combo rule); "(recipe)" keeps the loaded-figure snapshot behaviour.  Beside it
            # the SUB PLOT chooser picks what each cell draws (auto = derive from what the slice
            # leaves; else an explicit hist / 2d / 1d) -- the params section below always shows the
            # RESOLVED kind's knobs (see _sync_settings_param_rows).
            self.facet_combo = None
            self.sub_kind_combo = None
            if self.config.kind == "grid":
                self.facet_combo = FluentComboBox()
                self.facet_combo.setToolTip(
                    "Expand ONE axis of the bound block into the grid cells: each repeat / scan-axis "
                    "entry / data-axis entry becomes its own cell ((recipe) = a loaded figure's snapshot)")
                self.facet_combo.activated.connect(self._on_facet_changed)
                sec.addWidget(FluentSettingRow("facet", self.facet_combo, label_width=label_w))
                self._refresh_facet_combo()
                from .live import GRID_CELL_BY_KIND
                self.sub_kind_combo = FluentComboBox()
                self.sub_kind_combo.addItem("auto", "")
                for cell_kind in GRID_CELL_BY_KIND:     # the cell families, ONE source
                    self.sub_kind_combo.addItem(cell_kind, cell_kind)
                self.sub_kind_combo.setToolTip(
                    "What each cell draws: auto derives it from what the facet slice leaves "
                    "(a 2-D frame -> 2d, an ordered axis -> 1d, bare samples -> hist); "
                    "an explicit pick overrides")
                self.sub_kind_combo.activated.connect(self._on_sub_kind_changed)
                sec.addWidget(FluentSettingRow("sub plot", self.sub_kind_combo, label_width=label_w))
                self._refresh_sub_kind_combo()

            # unit cycle: a single row [Unit button | <stretch> | current unit text] under the "unit"
            # label (one-control-per-row rhythm) -- the IDENTICAL row the Edit tab builds via the helper.
            unit_row, self.unit_button, self.unit_label = \
                self._make_unit_cycle_row(self._on_unit_cycle, label_w, with_label=True)
            sec.addWidget(unit_row)

            # per-panel display refresh rate (this panel only).  A fixed, harmonic set so
            # panels that share a beat stay frame-coherent (see UPDATE_INTERVALS); a fast rate
            # suits a live-1D alignment monitor.  Changing it re-bases the console timer.
            self.update_combo = FluentComboBox()
            for ms in UPDATE_INTERVALS:
                self.update_combo.addItem(f"{ms} ms", ms)
            idx = self.update_combo.findData(self.config.update_ms)
            self.update_combo.setCurrentIndex(idx if idx >= 0 else self.update_combo.findData(DEFAULT_UPDATE_MS))
            self.update_combo.setToolTip(
                "How often THIS panel redraws.  Every rate shares one base tick, so panels that\n"
                "share a beat refresh on the SAME tick from the same data -- a 2-D frame and its\n"
                "site-map stay shot-coherent.  A fast 100 ms suits a live-1D alignment monitor.")
            self.update_combo.currentIndexChanged.connect(self._on_update_interval)
            sec.addWidget(FluentSettingRow("update", self.update_combo, label_width=label_w))

            # ---- Analysis --------------------------------------------------
            # ONE picker for what a drag-selection DOES, spanning BOTH analyses this section owns: a
            # general CURVE FIT (its whole state = config.params['fit_request'] presence) and an ROI
            # crop (selection_action == 'roi').  The combo is a pure VIEW -- it DERIVES its selected
            # item from that state on every open (_refresh_analysis_controls), so the Setting picker
            # and the Edit picker can never disagree (#8); picking "curve fit" toggles fit_request
            # (never writes selection_action='fit'), "ROI" arms the crop, "none" clears both.  The
            # section is named "Analysis" because it is no longer fit-only nor roi-only (#3 naming).
            self._build_analysis_section(section_box, label_w)

        # ---- Panel ---------------------------------------------------------
        sec = section_box("Panel")
        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.textChanged.connect(self._on_title)
        sec.addWidget(FluentSettingRow("title", self.title_edit, label_width=label_w))

        # Action row: Remove on the left (destructive, ORANGE) + Edit… to open the
        # panel's Edit tab.  Saving lives ONLY in the Edit tab now (it owns the folder
        # picker + the full DataFigure controls) -- the lightweight Setting popup no
        # longer carries a Save button, so there is one place to save from.
        remove = FluentButton("Remove", color=ORANGE)
        remove.setFixedWidth(scaled_px(72, minimum=58))
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        edit_button = FluentButton("Edit…", color=ACCENT)
        edit_button.setFixedWidth(scaled_px(64, minimum=52))
        edit_button.setToolTip("Open this panel's Edit tab: colormap / unit / relim, curve fit, limits, save")
        edit_button.clicked.connect(lambda: (self.settings_popup.hide(), self.edit_requested.emit(self)))
        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(scaled_px(6, minimum=4))
        action_row.addWidget(remove, 0)
        action_row.addWidget(edit_button, 0)
        action_row.addStretch(1)
        sec.addLayout(action_row)

        self.settings_popup = popup
        # Setting-frame height high-water mark (#H3i-2): the popup GROWS to fit its content and
        # GROWS when the panel size grows, but NEVER shrinks back within a session.  Reset here so a
        # REBUILT popup (its content changed, e.g. a +/- signal slot) fits fresh, then grows again.
        self._settings_h_hwm = 0
        # A Qt.Popup auto-closes on the press that lands on the Setting button; record
        # WHEN so the button's release does not immediately re-open it (real toggle).
        self._settings_dismissed_at = 0.0
        popup._on_hidden = self._note_settings_dismissed

    def _note_settings_dismissed(self) -> None:
        self._settings_dismissed_at = time.monotonic()

    def _make_param_widget(self, spec: ParamDecl, *, apply=None) -> QtWidgets.QWidget:
        """One widget per declarative ParamDecl; edits apply INSTANTLY.

        ``apply`` overrides where the edit goes (default ``self._set_param``); the
        Edit tab passes its own callback so a functional-param edit re-renders the
        live panel AND its snapshot.  Dispatches through the SAME PARAM_WIDGETS
        registry the measurement form uses (one handler per kind), with the edit
        wired as the ``instant_apply`` path so a plot param applies on each edit."""

        cb = apply if apply is not None else self._set_param
        current = self.config.params.get(spec.key, spec.default)
        ctx = ParamWidgetContext(instant_apply=cb)
        return PARAM_WIDGETS[spec.kind].build(spec, current, ctx)

    def _emit_param_rows(self, specs, add, apply, label_w) -> dict:
        """Render each declarative ParamDecl in ``specs`` as a ``[label | control]`` row through the
        SAME _make_param_widget / PARAM_WIDGETS path the measurement form uses, appending it via the
        ``add`` callback.  Returns ``{key: widget}`` so a caller can keep a named back-reference.  BOTH
        the Setting popup AND the Edit tab call this for a plot's display knobs, so adding a plot
        ParamDecl shows up in both surfaces with NO hand-wiring (#H3v-4b)."""
        out = {}
        for spec in specs:
            widget = self._make_param_widget(spec, apply=apply)
            out[spec.key] = widget
            self._param_kinds[spec.key] = spec.kind        # remember the kind for refresh_on_show re-seed
            add(FluentSettingRow(spec.label, widget, label_width=label_w))
        return out

    def _make_fixed_lim_row(self, apply_cb, label_w):
        """The fixed lo/hi inputs as ONE bespoke ``[lo | hi]`` row (the single display knob kept
        special-cased rather than declarative -- a 2-box combined control PARAM_WIDGETS has no kind for),
        shown only when relim == "fixed".  Built by this ONE helper so the Setting popup and the Edit tab
        get the IDENTICAL row instead of two hand-copied blocks (#H3v-4b).  Returns ``(row, lo, hi)``."""
        lo = FluentLineEdit(str(self.config.params.get("fixed_lo", 0.0)))
        hi = FluentLineEdit(str(self.config.params.get("fixed_hi", 1.0)))
        for ed in (lo, hi):
            ed.setMinimumWidth(scaled_px(64, minimum=52))
            ed.editingFinished.connect(apply_cb)
        lo.setToolTip("Fixed lower limit (used only when relim = fixed)")
        hi.setToolTip("Fixed upper limit (used only when relim = fixed)")
        host = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(scaled_px(6, minimum=4))
        lay.addWidget(lo, 1)
        lay.addWidget(hi, 1)
        row = FluentSettingRow("lo / hi", host, label_width=label_w)
        row.setVisible(self._relim() == "fixed")
        return row, lo, hi

    def _make_unit_cycle_row(self, on_click, label_w, *, with_label: bool):
        """The x-axis unit-cycle row as ONE bespoke ``[Unit button | <stretch> [| current-unit text]]``
        row.  Built by this ONE helper (like ``_make_fixed_lim_row``) so the Setting popup and the Edit
        tab get the IDENTICAL row -- same button width, same flush-left idiom, same SINGLE tooltip --
        instead of two hand-copied blocks whose tooltips had already drifted apart.  ``with_label=True``
        (Setting) also builds the live current-unit ``self.unit_label`` (refreshed by ``_on_unit_cycle``);
        Edit passes ``with_label=False``.  Returns ``(row, button, label_or_None)``."""
        button = FluentButton("Unit", color=GREY)
        button.setFixedWidth(scaled_px(70, minimum=56))
        button.setToolTip(
            "Cycle the x-axis unit (GHz/nm/MHz or ns/us/ms) where the axis label\n"
            "declares a convertible unit; otherwise a no-op")
        button.clicked.connect(on_click)
        host = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(scaled_px(6, minimum=4))
        lay.addWidget(button, 0)
        lay.addStretch(1)
        label = None
        if with_label:
            label = FluentLabel(self._current_unit_text())
            label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lay.addWidget(label, 0)
        row = FluentSettingRow("unit", host, label_width=label_w)
        return row, button, label

    def _apply_lim_to_plotter(self) -> None:
        """Push the relim mode + fixed lo/hi onto the LIVE plotter through the ONE in-place entry
        every surface uses -- ``apply_param``'s relim family (BaseLivePlot applies mode + lo/hi and
        re-relims now; a GridPlot stores them and fans out to its THUMBNAILS and any focused cell) --
        never a bare attribute poke, which a grid would silently ignore.  Shared by the declarative
        relim edit (_set_param) and the rebuild-time re-apply (_apply_display_params)."""
        if self.plotter is None:
            return
        # Coalesce: each apply_param ends in BaseLivePlot.draw() (a SYNCHRONOUS draw_idle+flush_events, and
        # for a grid a whole-N-cell thumbnail rebuild), so three back-to-back applies did three full renders
        # of the same final state.  suspend_draws makes those inner draws no-ops; the ONE trailing
        # canvas.draw_idle() below paints the final result once (identical pixels, ~3x fewer renders).
        with self.plotter.suspend_draws():
            self.plotter.apply_param("relim", self._relim())
            self.plotter.apply_param("fixed_lo", float(self.config.params.get("fixed_lo", 0.0)))
            self.plotter.apply_param("fixed_hi", float(self.config.params.get("fixed_hi", 1.0)))
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _kind_repeat_modes(self) -> list:
        """The repeat modes THIS plot kind offers (its ``PLOT_KINDS`` entry, else the image
        default) -- the ONE lookup shared by the param spec + the value clamp (#A3)."""
        from .live import PLOT_KIND_BY_KEY, IMAGE_REPEAT_MODES
        pk = PLOT_KIND_BY_KEY.get(self.config.kind)
        return list(pk.repeat_modes) if (pk and pk.repeat_modes) else list(IMAGE_REPEAT_MODES)

    def _relim(self) -> str:
        """The panel's relim mode, defaulting to ``_RELIM_PARAM.default`` -- the ONE place that
        default lives, so every reader agrees instead of re-typing the 'tight' literal (#A3)."""
        return str(self.config.params.get("relim", _RELIM_PARAM.default))

    def _repeat_param_specs(self) -> tuple:
        """The PLOT's repeat-DISPLAY param (``repeat_mode``), DECLARED so the Setting auto-renders it
        through the same widget path as every other param.  This is the ONLY repeat knob on the plot:
        the COUNT (``repeat``) belongs to the MEASUREMENT (the plot cannot tell a measurement how many
        times to run) -- the plot just chooses how to DISPLAY the repeats the measurement produced.
        Each plot KIND exposes the repeat modes meaningful for it (#issue-1), read from the one
        ``PLOT_KINDS`` table: the BASE verbs (average/add/replace) are GENERIC across EVERY kind (one
        ``reduce_repeat`` collapses the axis the same way); a trace adds ``roll``; a DISTRIBUTION adds
        ``pool`` (bin every repeat together) and defaults to it.  2d/sites omit per-repeat ``create``.
        No kind offers a mode it would silently ignore; the default is the kind's first (canonical) mode."""
        modes = self._kind_repeat_modes()
        return (
            ParamDecl(key="repeat_mode", label="repeat mode", kind="choice", default=modes[0],
                      choices=tuple(modes),
                      tooltip=self._repeat_mode_tooltip()),
        )

    def _repeat_mode_value(self) -> str:
        """The stored repeat_mode, CLAMPED to this kind's valid modes (migrates an old saved layout
        whose mode -- e.g. a histogram saved with 'average' before #issue-1 -- is no longer offered)."""
        modes = self._kind_repeat_modes()
        cur = str(self.config.params.get("repeat_mode", modes[0]))
        return cur if cur in modes else modes[0]

    def _bound_is_occupancy(self) -> bool:
        """Whether this panel's bound signal is an OCCUPANCY vector (a Judge-occupancy output) --
        driven off the signal's ROLE, not a hardcoded panel-kind string: its producing node resolves
        a site-map centres/underlay for it (only a Judge-occupancy processor does), via the same
        ``sites_inputs_provider`` the site map already uses (#H3s-F5).  So the occupancy meaning shows
        for an occupancy signal on ANY plot kind, and stays generic for everything else."""
        occ = self.config.inputs[0] if self.config.inputs else ""
        if not occ or not callable(self.sites_inputs_provider):
            return False
        try:
            centers_name, _ = self.sites_inputs_provider(occ)
        except Exception:
            return False
        return bool(centers_name)

    def _repeat_mode_tooltip(self) -> str:
        """The repeat-mode tooltip, SPECIALISED for an occupancy signal (#H3s-F5): when the bound
        signal is per-site occupancy, ``average`` is the per-site LOADING PROBABILITY (mean of the N
        shots' 0/1), ``add`` the total loads, ``replace`` the latest shot, ``roll`` the last N --
        the experiment meaning, not a generic array verb.  Generic otherwise, driven off the signal's
        role rather than the panel kind."""
        if self.config.kind == "hist":
            return ("How to combine the measurement's repeats into the distribution:\n"
                    "  pool    = bin EVERY repeat's samples into ONE histogram (all repeats together)\n"
                    "  average = bin the per-point MEAN over the repeats (one histogram)\n"
                    "  add     = bin the per-point SUM over the repeats\n"
                    "  replace = bin only the newest repeat's samples\n"
                    "  create  = ONE filled histogram per repeat overlaid (each a different colour; the "
                    "first repeat also draws the fit/threshold/stats)")
        if self._bound_is_occupancy():
            return ("How to combine the N shots of per-site occupancy for display:\n"
                    "  average = per-site LOADING PROBABILITY = mean of the N shots' 0/1\n"
                    "  add     = total loads per site over the N shots\n"
                    "  replace = the latest shot's 0/1\n"
                    "  roll    = the last N shots (rolling)\n"
                    "  create  = draw EVERY shot as its own line (1-D only)")
        return ("How to combine the measurement's repeats for display (decoupled from\n"
                "the signal):\n"
                "  average = mean over the repeats that have data (noise reduction)\n"
                "  add     = sum over repeats\n"
                "  replace = show the latest repeat\n"
                "  roll    = show the newest repeat (rolling)\n"
                "  create  = draw EVERY repeat as its own line (1-D only)")

    def refresh_on_show(self) -> None:
        """Re-seed every Setting control from ``config.params`` -- the SINGLE source of truth for a
        panel's params -- so the Setting popup shows the CURRENT values whenever it opens, even if they
        were changed elsewhere (the Edit tab writes the same config.params).  Each widget is re-seeded
        through its kind's ``PARAM_WIDGETS.write`` (one entry point, no per-key handwiring), with its
        change signals blocked so re-seeding does not re-fire ``_set_param`` (which would needlessly
        rebuild).  This is the #6 fix: a control is a VIEW of config.params, refreshed on show, never a
        private copy that drifts from the other surface."""
        for key, widget in self.param_widgets.items():
            kind = self._param_kinds.get(key)
            if kind is None or key not in self.config.params:
                continue
            with _signals_blocked(widget):
                try:
                    PARAM_WIDGETS[kind].write(widget, self.config.params[key])
                except (TypeError, ValueError):
                    continue

    def _open_settings(self) -> None:
        # Click-to-open / click-again-to-close TOGGLE.  A Qt.Popup already auto-closes
        # on the mouse PRESS that lands on this button, so by the time the button's
        # release fires the popup is hidden -- naively re-opening it would make the
        # button never close it.  So: if visible, hide; and if it was auto-dismissed
        # within the last moment (this very click closed it), do NOT re-open.
        popup = self.settings_popup
        if popup.isVisible():
            popup.hide()
            return
        if time.monotonic() - self._settings_dismissed_at < 0.25:
            return
        self._sync_settings_param_rows()   # a grid's resolved kind may have changed since the last bake
        popup = self.settings_popup        # (the sync may have swapped in a fresh popup)
        self.refresh_on_show()          # Setting controls are a VIEW of config.params -- refresh on open (#6)
        self._refresh_signal_combo()
        self._refresh_facet_combo()     # facet choices re-derive from the CURRENT node structure
        self._refresh_sub_kind_combo()
        anchor = self.setting_button.mapToGlobal(
            QtCore.QPoint(self.setting_button.width(), self.setting_button.height()))
        self._size_settings_popup()                        # height: show-all, grow-not-shrink (#H3i-2)
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        top_y = anchor.y() + _popup_gap()   # the ONE below-anchor Fluent popup gap (combo / overflow share it)
        x = anchor.x() - popup.width()
        if avail is not None:
            x = max(avail.left(), min(x, avail.right() - popup.width()))
        popup.move(x, top_y)
        popup.show()
        popup.raise_()

    def _size_settings_popup(self) -> None:
        """Size the Setting frame: EXPAND to show all its content, UNTIL it reaches the PLOT PANEL's
        own bottom edge -- then the FluentScrollArea scrolls the overflow.  So the cap is the panel
        boundary (`panel_bottom - top_y`), NOT the screen: a tall panel gets a tall frame, a short
        panel scrolls, and either way the frame stays within the panel.  GROW with the panel size,
        clamped to the live cap (so shrinking the panel re-clamps the frame to the smaller panel)."""
        popup = getattr(self, "settings_popup", None)
        if popup is None:
            return
        popup.adjustSize()
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        anchor_y = self.setting_button.mapToGlobal(QtCore.QPoint(0, self.setting_button.height())).y()
        top_y = anchor_y + _popup_gap()      # match the same gap the open path (above) uses to place it
        content = self._settings_scroll.widget()
        content_h = (content.sizeHint().height() if content is not None else popup.height()) + 2 * scaled_px(10)
        # The cap is the PLOT PANEL's own bottom edge (the popup opens just below the gear near the
        # panel top and grows DOWN); content past it scrolls.
        panel_bottom = self.mapToGlobal(QtCore.QPoint(0, self.height())).y()
        cap = max(scaled_px(140), panel_bottom - top_y)
        if avail is not None:                              # last-resort: never escape the physical screen
            cap = min(cap, avail.bottom() - top_y)
        want = min(content_h, cap)                         # grow to the content, capped at the panel bottom
        self._settings_h_hwm = max(int(self._settings_h_hwm), int(want))   # grow within a session...
        h = min(self._settings_h_hwm, cap)                 # ...but always re-clamp to the LIVE panel cap
        popup.setMaximumHeight(int(cap))                   # content beyond the panel bottom scrolls
        popup.resize(popup.width(), max(scaled_px(140), int(h)))

    def _fill_slot_combo(self, combo, current: str) -> None:
        """Populate ONE slot combobox with ``(none)`` + every live hub signal GROUPED by its
        producing node (the shared :func:`fill_grouped_signal_combo`): bold non-selectable
        headers, indented signals.  Keeps the slot's current pick selected even when its source
        node is not running yet (so a saved layout's ``signal[1]`` survives a restart)."""
        fill_grouped_signal_combo(
            combo, names=(self.names_provider() if callable(self.names_provider) else []),
            sources=(self.sources_provider() if callable(self.sources_provider) else {}),
            formats=(self.formats_provider() if callable(self.formats_provider) else {}),
            labels=self._signal_short_names_map(),
            current=current, none_label="(none)")

    def _signal_short_names_map(self) -> dict:
        """``{full hub signal: SHORT name}`` (the producing node's prefix stripped) from the
        ``short_names_provider``, so the picker nest leaf is the short name (``frame`` / ``survival`` /
        ``rate``) -- NOT the full prefixed key and NOT the verbose SignalSpec axis label (``camera
        image``).  The picker nest already names the producer node, so the leaf is the short NAME."""
        return coerce_short_labels(self.short_names_provider)

    def _refresh_signal_combo(self) -> None:
        """Refresh the signal picker with the hub's current signals, each labelled with the
        measurement/processor that PRODUCES it (``occupied — occupancy``) so the input is
        filled by origin, not a bare name.  It keeps ``config.inputs[0]`` selected; the source
        expression reads it as ``signal`` (``value = signal``)."""
        for i, combo in enumerate(getattr(self, "slot_combos", [])):
            cur = self.config.inputs[i] if i < len(self.config.inputs) else ""
            self._fill_slot_combo(combo, cur)

    def _on_slot_pick(self, idx: int) -> None:
        """The signal picker changed: write its bare signal name into ``config.inputs[idx]``
        and point the source at ``value = signal`` (a blank pick blanks the panel).  The
        picker IS the "plot this signal" control; the expression box stays the advanced
        override for a custom ``value = ...``.  Then re-apply so the pick takes effect now."""
        name = str(self.slot_combos[idx].currentData() or "")   # bare name (not the labelled text)
        while len(self.config.inputs) <= idx:
            self.config.inputs.append("")
        self.config.inputs[idx] = name
        # Single-slot: the picker IS the "plot this signal" control -> point the source at it
        # (value = signal).  Multi-slot: the expression (value = signal[0] - signal[1]) is
        # user-authored and combines the slots, so a pick just rebinds slot idx -- don't clobber it.
        if len(self.slot_combos) <= 1 and idx == 0:
            self.source_edit.setText("value = signal" if name else _BLANK_SOURCE)
        self._apply_source()                      # picking a slot applies instantly
        self._sync_settings_param_rows()          # a grid's resolved cell kind follows the new bind

    def _add_signal_slot(self) -> None:
        """Add a signal slot (signal[i]) so the panel can combine more signals.  When growing
        from one slot to two, seed the canonical two-signal expression so the panel is usable
        at once; the user can edit it.  Rebuilds the Setting popup so the new row appears."""
        self.config.inputs.append("")
        if len(self.config.inputs) == 2 and self.config.source.strip() in ("value = signal", "", _BLANK_SOURCE.strip()):
            self.config.source = "value = signal[0] - signal[1]"
            self._compiled_source = self.config.source
        self._rebuild_settings_popup()
        self._apply_source()

    def _remove_signal_slot(self) -> None:
        """Remove the LAST signal slot (never below one).  Back at a single slot, restore the
        canonical ``value = signal`` so the picker drives the plot again."""
        if len(self.config.inputs) <= 1:
            return
        self.config.inputs.pop()
        if len(self.config.inputs) == 1:
            self.config.source = "value = signal" if self.config.inputs[0] else _BLANK_SOURCE
            self._compiled_source = self.config.source
        self._rebuild_settings_popup()
        self._apply_source()

    def _rebuild_settings_popup(self, *, reopen: bool = True) -> None:
        """Rebuild the Setting popup (the slot count and the per-kind param rows are fixed at build
        time, so adding/removing a slot or a resolved-kind change rebuilds it).  ``reopen`` shows the
        fresh popup at once (the user was in it); False rebuilds silently for the next open."""
        old = getattr(self, "settings_popup", None)
        if old is not None:
            old.hide()
            old.deleteLater()
        self._build_settings()                    # builds a fresh self.settings_popup + combos
        if reopen:
            self._settings_dismissed_at = 0.0
            self._open_settings()                 # reopen (positioned + height-capped)

    def _sync_settings_param_rows(self) -> None:
        """A grid panel's per-kind Setting rows (bins/fit/ylog for hist cells, the colormap for 2d
        cells, ...) follow the RESOLVED per-cell kind: whenever the resolve changes -- a facet or
        sub-plot pick, a signal bind, a load -- the popup is rebuilt so the rows are the new kind's
        PANEL_PARAMS, never a stale bake of the kind the popup happened to open with."""
        if self.config.kind != "grid":
            return
        if self._param_kind() == getattr(self, "_settings_param_kind", None):
            return
        self._rebuild_settings_popup(reopen=self.settings_popup.isVisible())

    # ------------------------------------------------ basic display toggles
    def _current_unit_text(self) -> str:
        n = int(self.config.params.get("unit_index", 0) or 0)
        return "unit" if n == 0 else f"unit x{n}"

    def _on_unit_cycle(self) -> None:
        """Cycle the x-axis unit ONE step on the live panel and remember the new
        index (reuses DataFigure.change_unit).  Cycling once on the current axis
        (rather than re-applying index-from-original) keeps repeated clicks
        correct; a rebuild re-derives the same state from the stored index.  When
        the axis label carries no convertible unit the cycle is a no-op."""
        if self.plotter is None:
            return
        self._wait_render_idle()       # change_unit rewrites live x data/labels: own the figure first
        df = self._unit_df()
        length = len(df.conversion_map) if getattr(df, "conversion_map", None) else 0
        if length:
            df.change_unit()
            self.config.params["unit_index"] = (
                int(self.config.params.get("unit_index", 0) or 0) + 1) % length
            self.changed.emit()
        if hasattr(self, "unit_label"):
            self.unit_label.setText(self._current_unit_text())
        if self.canvas is not None:
            self.canvas.draw_idle()

    def apply_fixed_lims(self, lo: float, hi: float) -> None:
        """Persist + apply the fixed lo/hi NOW through the ONE in-place entry every surface uses
        (``apply_param``'s relim family): on a ZOOMED grid the pin is ALSO stored on the parked grid
        (thumbnails carry it when the zoom returns); otherwise it lands on the live plotter directly
        (a GridPlot fans it out to its thumbnails itself).  The Setting popup's lo/hi inputs and the
        Edit tab's both route here -- no second hand-copied push path."""
        self._wait_render_idle()       # relim + axis mutation on the live plotter: own the figure first
        self.config.params["fixed_lo"], self.config.params["fixed_hi"] = float(lo), float(hi)
        if self._grid_focus is not None:
            self._set_focused_grid_param("fixed_lo", float(lo))
            self._set_focused_grid_param("fixed_hi", float(hi))
        elif self.plotter is not None:
            with self.plotter.suspend_draws():   # both applies paint the SAME final state -> one render below
                self.plotter.apply_param("fixed_lo", float(lo))
                self.plotter.apply_param("fixed_hi", float(hi))
        if self.canvas is not None:
            self.canvas.draw_idle()
        self.changed.emit()

    def _on_fixed_lim_edited(self) -> None:
        """The Setting popup's fixed lo/hi inputs committed (#8): read + apply via the ONE path."""
        self.apply_fixed_lims(_safe_float(self.fixed_lo_edit.text(), 0.0),
                              _safe_float(self.fixed_hi_edit.text(), 1.0))

    def _on_update_interval(self, _idx: int) -> None:
        """Persist THIS panel's display refresh interval (``config.params["update_ms"]``,
        one of :data:`UPDATE_INTERVALS`) and ask the console to re-base its timer so the new
        rate co-aligns with the others.  No plot rebuild -- only the refresh cadence changes."""
        ms = int(self.update_combo.currentData() or DEFAULT_UPDATE_MS)
        if ms == int(self.config.params.get("update_ms", DEFAULT_UPDATE_MS) or DEFAULT_UPDATE_MS):
            return
        self.config.params["update_ms"] = ms
        self.update_interval_changed.emit()    # console re-bases the shared timer
        self.changed.emit()                    # mark the layout dirty

    # --------------------------------------------- basic display application
    def _param_kind(self) -> str:
        """The plot kind whose :data:`PANEL_PARAMS` drive THIS panel's Setting / Edit param UI.

        For every ordinary kind that is ``config.kind``.  For a GRID panel it is the grid's per-site
        ``sub_plot_kind`` -- so a hist grid shows the ``"hist"`` bins/fit/ylog knobs and a 2d (kernel) grid
        shows the ``"2d"`` colormap chooser, INSTEAD of the one hard-coded hist set a fixed ``"grid"`` entry
        used to give every grid regardless of what its cells actually were (#4).  Resolved from the built grid
        (``plotter.sub_plot_kind``); before the grid is built, peeked off its producing node's recipe."""
        if self.config.kind != "grid":
            return self.config.kind
        sub = getattr(self.plotter, "sub_plot_kind", None)
        if sub:
            return str(sub)
        if self._facet():
            return self._resolved_sub_kind()     # a facet grid's kind derives from the slice, not a recipe
        recipe = self._grid_recipe_or_none()
        if recipe:
            return str(recipe.get("sub_plot_kind") or "hist")
        return "hist"

    def _grid_recipe_or_none(self):
        """The bound signal's grid RECIPE (a loaded saved-figure's reproduction dict) when its
        producing node carries one -- else None.  The ONE probe _param_kind and the facet choices
        share (a "(saved figure)" option only exists when there IS a saved figure to show)."""
        if not (callable(self.grid_recipe_provider) and self.config.inputs and self.config.inputs[0]):
            return None
        try:
            return self.grid_recipe_provider(self.config.inputs[0])
        except Exception:
            return None

    def _grid_recipe_with_params(self, recipe: Mapping[str, object]) -> dict:
        """Fold THIS panel's live grid DISPLAY knobs (the per-site ``sub_plot_kind``'s params -- ``bins`` /
        ``fit`` / ``ylog`` for a hist grid, ``cmap`` for a 2d grid) from ``config.params`` into the producing
        node's grid recipe, so a rebuild redraws the grid with the operator's current Setting choices (not just
        the recipe's saved defaults).  The panel's params WIN over the recipe's stored ``display_params`` (the
        operator's live pick is the source of truth); ``bins`` also overrides the recipe's top-level ``bins`` so
        the thumbnails re-bin.  A copy -- the node's recipe is untouched."""
        out = dict(recipe)
        # The panel's CURRENT title wins over the recipe's saved one -- same "live pick beats stored
        # default" rule as the display knobs below.  build_grid_figure reads recipe['title'], so without
        # this an Edit-tab title edit on a RECIPE grid (a loaded figure) rebuilt the snapshot from the
        # stale saved title and never followed the edit (the facet branch already passes config.title to
        # _build_facet_plotter; this makes the recipe branch agree).  ONE injection point -> both the live
        # card's recipe rebuild and the Edit snapshot rebuild pick up the edited title.
        out["title"] = self.config.title or out.get("title") or ""
        display = dict(out.get("display_params") or {})
        for decl in _panel_display_decls(self.config.kind, self._param_kind()):   # sub-kind knobs + grid title (#5)
            if decl.key in self.config.params:
                display[decl.key] = self.config.params[decl.key]
        # The relim family is display state too (it lives beside PANEL_PARAMS as _RELIM_PARAM + the
        # bespoke lo/hi row): folded in like every other knob, so a rebuilt grid / Edit snapshot
        # replays the operator's lim onto the thumbnails and the focus seed -- never a silent revert.
        for lim_key in ("relim", "fixed_lo", "fixed_hi"):
            if lim_key in self.config.params:
                display[lim_key] = self.config.params[lim_key]
        if "bins" in display:
            out["bins"] = int(display["bins"])          # top-level bins drives the thumbnail binning
        out["display_params"] = display
        return out

    # --------------------------------------------------------------- facet (the grid as an axis-expander)
    def _facet(self) -> str | None:
        """The panel's facet declaration (``config.params["facet"]``): which axis of the bound
        canonical ``(R,P,*data_shape)`` block expands into the grid cells.  A missing value means a
        non-faceted recipe grid; every present value is one canonical facet token."""
        value = self.config.params.get("facet")
        if value is None:
            return None
        normalize_facet(value)
        return str(value)

    def _facet_value_shapes(self) -> tuple[tuple, tuple]:
        """(points multi-D shape, data shape) from the producing node's declared structure (#H3o) --
        the points axis is stored FLAT, its multi-D form lives in ``grid_shape`` when the scan
        declared one.  The ONE source the facet choices, the auto sub-kind and the slicer share."""
        st = self._bound_structure() or {}
        gs = tuple(int(n) for n in (st.get("grid_shape") or ()))
        ps = tuple(int(n) for n in (st.get("points_shape") or ()))
        ds = tuple(int(n) for n in (st.get("data_shape") or ()))
        return (gs or ps), ds

    def _resolved_sub_kind(self) -> str:
        """The facet grid's per-cell kind: the operator's explicit ``sub_plot_kind`` param when set,
        else the ONE auto rule (``default_sub_plot_kind``: what each cell has left after the slice)."""
        from .live import GRID_CELL_BY_KIND, default_sub_plot_kind
        sub = str(self.config.params.get("sub_plot_kind") or "")
        if sub in GRID_CELL_BY_KIND:
            return sub
        points_shape, data_shape = self._facet_value_shapes()
        return default_sub_plot_kind(
            self._facet() or "repeat", points_shape=points_shape, data_shape=data_shape)

    def _facet_cells(self, value):
        """Slice the bound block into the per-cell inputs through the ONE rule (live.facet_cells)."""
        from .live import facet_cells, normalize_facet
        block = np.asarray(value)                 # NATIVE dtype (uint8 camera block stays uint8);
        if block.dtype.kind not in "iubf":        # only non-numeric results normalize to float
            block = np.asarray(block, dtype=float)
        if block.ndim < 2:
            raise ValueError(
                "a facet grid slices the bound signal's canonical (R,P,*data_shape) block; got shape "
                f"{block.shape} -- bind a measurement's block signal (not a scalar).")
        pts, _ = self._facet_value_shapes()
        if int(np.prod(pts, dtype=np.int64) if pts else 0) != int(block.shape[1]):
            pts = ()                              # declared shape does not match this block -> flat points
        cells = facet_cells(block, self._facet(), sub_plot_kind=self._resolved_sub_kind(),
                            points_shape=pts, repeat_mode=self._repeat_mode_value())
        # A LIVE finite block carries only the repeats that HOLD data (it grows 1..ring as shots
        # land), so a facet=repeat grid would see its cell count change every shot -- a full
        # build-then-swap per shot for the WHOLE fill window, and a >MAX_GRID_CELLS repeat would
        # only error at shot ring+1.  Pad to the producer's declared ring with ZERO-memory NaN
        # placeholders (broadcast views -- never a materialised full-size frame), so the grid holds
        # a constant ring cells from the first shot: not-yet-taken cells render blank (NaN), filled
        # cells stream through the in-place update_cells fast path, and the cell-count cap fires
        # immediately.  Only the repeat facet pads: a points/data facet's cell count is a declared
        # shape, not the fill state.
        spec = normalize_facet(self._facet())
        if spec is not None and spec[0] == "repeat" and cells:
            st = self._bound_structure()
            ring = int((st or {}).get("ring", 0) or 0)
            if ring > len(cells):
                blank = np.broadcast_to(np.float32(np.nan), np.shape(cells[0]))
                cells = list(cells) + [blank] * (ring - len(cells))
        return cells

    def _build_facet_plotter(self, value, *, interactions: bool):
        """Build this panel's FACET grid through the ONE factory + replay its persisted per-kind
        display knobs (bins / fit / ylog / cmap) AND the relim family (BaseLivePlot.apply_param owns
        them all).  The ONE builder the live card (display-only) and the Edit-tab snapshot
        (interactive) share -- so the two can never drift."""
        from .live import facet_axis_labels, facet_cell_labels, grid as build_facet_grid
        sub = self._resolved_sub_kind()
        cells = self._facet_cells(value)
        # Per-cell TITLE identifiers (#5): the console pre-slices (facet=None to the factory), so it hands
        # the labels EXPLICITLY, derived from the bound facet + its points shape through the ONE
        # ``facet_cell_labels`` source -- so a repeat / scan / site grid's cells read 'rep k' / a scan
        # coordinate / 's k' instead of a hardcoded site tag, from the same source the notebook path uses.
        points_shape, data_shape = self._facet_value_shapes()
        # The swept axis NAMES + per-cell COORDINATES are metadata of the producing SCAN NODE and the
        # facet spec -- NOT of the value expression -- so they are fetched UNGATED: even a transforming
        # value (``np.log(f)``, ``a-b``) still faceted on scan axis i, so cell k is still scan point k
        # of axis i and its coordinate is known.  (``_bound_structure`` is identity-gated because it
        # drives the value's SHAPE reshape, a DIFFERENT concern; the scan names/coordinates never
        # change under a value transform -- gating them there was the root cause of the ``pt k``
        # fallback.)  ``param_names`` are needed for EVERY facet group -- a repeat / data facet's cells
        # keep the scan axes as their remaining x/y (#6) -- while the per-cell COORDS label only a
        # POINTS facet's cells (``Bz=1.2`` instead of a bare ``pt k``).
        from .live import normalize_facet
        coords = names = None
        spec = normalize_facet(self._facet())
        if self.config.inputs and callable(self.structure_provider):
            try:
                raw = self.structure_provider(self.config.inputs[0])
            except Exception:
                raw = None
            if raw:
                names = raw.get("param_names")
                if spec and spec[0] == "points":
                    all_coords = raw.get("points_coords") or []
                    axis = int(spec[1])
                    if axis < len(all_coords) and len(all_coords[axis]) == len(cells):
                        coords = all_coords[axis]
        cell_labels = facet_cell_labels(self._facet(), len(cells), points_shape=points_shape,
                                        coords=coords, param_names=names)
        plotter = build_facet_grid(
            cells, sub_plot_kind=sub, size=self.config.size, cell_labels=cell_labels,
            # figure-level x/y axis names FOLLOW the facet (#6): the ONE facet_axis_labels rule maps
            # the REMAINING axes to a cell's x/y (scan param names / the value's declared axis
            # label), degrading to the kind's stock defaults when the metadata is unknown.  The
            # console pre-slices (facet=None to the factory), so it hands the labels explicitly --
            # the same derivation the notebook ``grid(value, facet=...)`` path runs inside.
            labels=facet_axis_labels(
                self._facet(), sub, points_shape=points_shape, data_shape=data_shape,
                param_names=names, value_label=self._source_axis_label()),
            display=False, interactions=interactions, title=self.config.title or "")
        # Fold the persisted display knobs in with the draws SUSPENDED.  Each ``apply_param`` otherwise
        # forces a synchronous N-cell repaint, so ~10 keys = up to 10 full grid ``draw()``s per build --
        # the draw-per-mutation anti-pattern, byte-identical to the loop in ``build_grid_figure`` above
        # (which already suspends).  The ONE first render happens via the caller's ``canvas.draw`` (the
        # plotter is built ``display=False``), exactly as the non-facet grid path.
        with plotter.suspend_draws():
            for key in ([d.key for d in _panel_display_decls("grid", sub)]
                        + ["relim", "fixed_lo", "fixed_hi", "view_xlim", "view_ylim", "fit_request"]):
                if key == "cmap":
                    # Inject the RESOLVED cmap (operator's pick ELSE the sub-kind's declared PANEL_PARAMS
                    # default) through the ONE ``_resolved_cmap`` resolver -- the SAME source the Setting
                    # popup shows.  Without this a grid whose params carry no explicit cmap fell through to
                    # ImageCell's own default (grey), so the live grid drew grey while the Setting said the
                    # default: render == Setting now, one source, no silent divergence.
                    plotter.apply_param("cmap", _resolved_cmap(sub, self.config.params))
                elif key in self.config.params:
                    plotter.apply_param(key, self.config.params[key])
        return plotter

    def _facet_choices(self) -> list[tuple[str | None, str, bool]]:
        """The facet dropdown's ``[(stored value, display text, enabled)]`` -- derived from the
        producing node's declared axis structure, with the axis LENGTH shown so the operator picks
        by meaning ('scan axis 0 (5)').  An axis longer than :data:`MAX_GRID_CELLS` is listed but
        DISABLED (the grid factory refuses it -- the UI would freeze); the "(saved figure)" row
        exists only when the bound node actually carries a saved grid recipe to replay."""
        from .live import MAX_GRID_CELLS
        out = []
        if self._grid_recipe_or_none() is not None:
            out.append((None, "(saved figure)", True))
        out.append(("repeat", "repeat", True))

        def _axis(value, text, n):
            ok = int(n) <= MAX_GRID_CELLS
            out.append((value, text if ok else f"{text} – too many", ok))

        points_shape, data_shape = self._facet_value_shapes()
        if len(points_shape) > 1:
            for i, n in enumerate(points_shape):
                _axis(f"points:{i}", f"scan axis {i} ({n})", n)
        elif points_shape and points_shape[0] > 1:
            _axis("points:0", f"scan axis 0 ({points_shape[0]})", points_shape[0])
        for i, n in enumerate(data_shape):
            if n > 1:
                _axis(f"data:{i}", f"data axis {i} ({n})", n)
        return out

    def _refresh_facet_combo(self) -> None:
        """Refill the facet dropdown from the CURRENT node structure (a Setting open re-derives it,
        like the signal combo) keeping the stored pick selected; over-limit axes come out greyed."""
        combo = getattr(self, "facet_combo", None)
        if combo is None:
            return
        current = self._facet()
        with _signals_blocked(combo):
            combo.clear()
            index = 0
            for j, (value, text, enabled) in enumerate(self._facet_choices()):
                combo.addItem(text, value)
                if not enabled:
                    item = combo.model().item(combo.count() - 1)
                    if item is not None:
                        item.setEnabled(False)
                if value == current:
                    index = j
            combo.setCurrentIndex(index)

    def _on_facet_changed(self, index: int) -> None:
        """The operator picked a facet axis: persist + rebuild.  A facet change is a STRUCTURE change
        (the cell count and even the per-cell kind may differ), so it goes through the ordinary reset
        path -- the generic refocus rule returns to the enlarged cell when it survives the rebuild."""
        raw_value = self.facet_combo.itemData(int(index))
        value = None if raw_value is None else str(raw_value)
        if value is not None:
            normalize_facet(value)
        if value == self._facet():
            return
        self._wait_render_idle()   # the teardown+rebuild below must own the figure (ownership protocol)
        if value is None:
            self.config.params.pop("facet", None)
        else:
            self.config.params["facet"] = value
        self._reset_plot()
        self._render_version = -1
        self._rerender_last()
        self._sync_settings_param_rows()   # the resolved per-cell kind may have changed with the axis
        self.changed.emit()

    def _refresh_sub_kind_combo(self) -> None:
        """Re-select the stored ``sub_plot_kind`` pick ("" = auto) on the Setting's sub-plot chooser."""
        combo = getattr(self, "sub_kind_combo", None)
        if combo is None:
            return
        with _signals_blocked(combo):
            combo.setCurrentIndex(max(0, combo.findData(str(self.config.params.get("sub_plot_kind") or ""))))

    def _on_sub_kind_changed(self, index: int) -> None:
        """The operator picked what each cell draws (auto / hist / 2d / 1d): persist + rebuild -- a
        per-cell kind change is a structure change exactly like a facet change, and the Setting's
        param rows follow the new resolve."""
        value = str(self.sub_kind_combo.itemData(int(index)) or "")
        if value == str(self.config.params.get("sub_plot_kind") or ""):
            return
        self._wait_render_idle()   # the teardown+rebuild below must own the figure (ownership protocol)
        if value:
            self.config.params["sub_plot_kind"] = value
        else:
            self.config.params.pop("sub_plot_kind", None)      # auto: derive from the slice
        self._reset_plot()
        self._render_version = -1
        self._rerender_last()
        self._sync_settings_param_rows()
        self.changed.emit()

    def _apply_display_params(self) -> None:
        """Apply the persisted Setting toggles (relim mode + unit cycle) to
        the current plotter -- called after every rebuild and on each edit.

        The persisted knobs: ``config.params["relim"]`` (confocal naming:
        ``tight`` / ``normal``), ``config.params["unit_index"]`` (the x-axis unit
        cycle count), and the view-window pins ``view_xlim`` / ``view_ylim`` (the
        Edit tab's range rows, #3) -- re-applied here so a rebuild / reopen keeps
        the operator's window instead of snapping back to autoscale.  (The y VALUE
        axis of a 1d panel is relim-owned; ``view_ylim`` exists only on the image
        family, whose ``apply_param`` gates it on ``y_is_view_axis``.)"""
        if self.plotter is None:
            return
        self._apply_lim_to_plotter()                     # relim family via the ONE in-place entry
        self._apply_unit()
        self._apply_view_lims()
        if "fit_request" in self.config.params:
            self.plotter.apply_param("fit_request", self.config.params.get("fit_request"))

    def _apply_view_lims(self) -> None:
        """Re-apply the persisted view-window pins (#3) to the LIVE plotter through the SAME
        ``apply_param`` entry every display knob uses -- a grid stores them and re-asserts them on every
        cell after each tick (:meth:`GridPlot._apply_view_knobs`), a flat panel sets its one axes -- so
        the operator's Apply survives a rebuild / reopen AND sticks across the live autoscale (never the
        old one-shot ``to_data_figure().xlim`` the next redraw wiped).  Absent (the default) leaves the
        plot's own autoscale untouched; a ``view_ylim`` on a non-image kind (a stale recipe) is refused
        by the plot's own ``y_is_view_axis`` gate."""
        if self.plotter is None:
            return
        for key in ("view_xlim", "view_ylim"):
            pin = self.config.params.get(key)
            if not pin:
                continue
            try:
                self.plotter.apply_param(key, (float(pin[0]), float(pin[1])))
            except Exception:
                pass                                      # a stale pin from a re-interpreted kind: ignore

    def _unit_df(self):
        """A DataFigure bound to the live card's figure/axes for the x-axis unit cycle
        (shared impl: :func:`_unit_df_for`)."""
        return _unit_df_for(self.plotter)

    def _unit_cycle_len(self) -> int:
        """Length of the x-axis unit cycle for the CURRENT plotter (0 if none)."""
        if self.plotter is None:
            return 0
        try:
            cmap = getattr(self._unit_df(), "conversion_map", None)
        except Exception:
            return 0
        return len(cmap) if cmap else 0

    def _apply_unit(self) -> bool:
        """Cycle the x-axis unit ``unit_index`` times from its original (reuse
        DataFigure.change_unit).  Returns True if any conversion was applied."""
        if self.plotter is None:
            return False
        index = int(self.config.params.get("unit_index", 0) or 0)
        if index <= 0:
            return False
        try:
            df = self._unit_df()
            if not getattr(df, "conversion_map", None):
                return False
            for _ in range(index % max(1, len(df.conversion_map))):
                df.change_unit()
            return True
        except Exception:
            return False

    def _refresh_title(self) -> None:
        """Compose the grey frame TITLE: the panel KIND + WHERE its signal comes from (the
        legend the console computes), e.g. ``1D vector — occupied ← Judge occupancy``.  This is
        the ordinary QGroupBox title -- the grey chip, alignment and font are the frame's own."""
        head = PANEL_KINDS[self.config.kind]
        info = " · ".join(p for p in self._signal_info.splitlines() if p.strip())
        self.setTitle(f"{head} — {info}" if info else head)

    def set_signal_info(self, info: str) -> None:
        """Set the signal legend (computed by the console: which node each read comes from).
        Shown in the frame TITLE (the grey strip), replacing the old footer legend."""
        info = str(info or "")
        if info == self._signal_info:
            return
        self._signal_info = info
        self._refresh_title()

    def set_status(self, text: str, *, error: bool) -> None:
        # No per-shot status line in the panel any more (it needed a footer, which broke the
        # height proportion).  The status lives in the Setting popup + the Setting-button
        # tooltip; an error turns the Setting button red.  Restyle only on the ok<->error edge.
        # THREAD-SAFE by deferral: compose() may run on the console's render thread, and Qt
        # widget calls (setText/setStyleSheet) are GUI-thread-only -- park the newest status and
        # let the GUI flush it when the batch is presented (flush_deferred_status).  Keeping the
        # deferral INSIDE the one status funnel means compose never forks per thread.
        if QtCore.QThread.currentThread() is not self.thread():
            self._deferred_status = (str(text), bool(error))
            return
        self._status_text = str(text)
        if hasattr(self, "status"):
            self.status.setText(str(text)[:200])
        self.setting_button.setToolTip(f"Panel settings — {text}" if text else "Panel settings")
        if error is not getattr(self, "_status_error", None):
            self._status_error = bool(error)
            colour = RED if error else GREY
            if hasattr(self, "status"):
                self.status.setStyleSheet(f"color: {colour}; background: transparent; border: none;")
            self.setting_button.set_color(colour)

    def flush_deferred_status(self) -> None:
        """Apply the newest status a render-thread compose parked (GUI thread, present phase)."""
        pending = self.__dict__.pop("_deferred_status", None)
        if pending is not None:
            self.set_status(pending[0], error=pending[1])

    # ------------------------------------------------------------- drag to grid
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # only the border frame starts a drag; the canvas consumes its own events
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.raise_()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None:
            new_pos = self.mapToParent(event.pos() - self._drag_offset)
            self.move(max(0, new_pos.x()), max(0, new_pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None:
            self._drag_offset = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            # Record the raw drop pixel as this card's (col, row); the console then REORDERS the card
            # to the ORDER position nearest that drop (:func:`drop_index` via ``dropped`` -- a drop onto
            # a card's slot displaces it, a drop past the last card appends to the bottom), and
            # :func:`pack` recomputes every pixel top-left from the new order.
            col, row = max(0, self.x()), max(0, self.y())
            if (col, row) != (self.config.col, self.config.row):
                self.config.col, self.config.row = col, row
                self.changed.emit()
            self.dropped.emit(self)         # drag-release ONLY: the console snaps the drop seed
            self.layout_changed.emit()      # re-pack (even when dropped back near the same spot)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._place_setting_button()

    # ------------------------------------------------------------- config edits
    def _on_title(self, text: str) -> None:
        """Update the plot title via the FRONTEND'S sealed title API.

        ``BaseLivePlot.title`` is the single source of truth and
        ``BaseLivePlot._apply_title`` routes through ``style.apply_title``,
        which paints the title at ``title_fontsize()`` with the correct pad.
        Calling ``ax.set_title(text)`` directly bypasses both -- matplotlib
        then uses its OWN ``axes.titlesize`` rcParam (NOT the frontend's
        ``axes.labelsize``-derived size), so the title visibly shrinks /
        grows after every edit.  Always go through the sealed API."""
        self.config.title = str(text)
        if self.plotter is not None and getattr(self.plotter, "ax", None) is not None:
            self._wait_render_idle()   # in-place artist mutation: own the figure first
            self.plotter.title = self.config.title
            self.plotter._apply_title()
            if self.canvas is not None:
                self.canvas.draw_idle()
        self.changed.emit()

    def _on_size(self, size: str) -> None:
        self._wait_render_idle()   # the teardown+regrow+rebuild below must own the figure
        self.config.size = str(size)
        # ATOMIC relayout transaction.  A size change is THREE synchronous steps: _reset_plot tears the
        # plotter down (canvas -> None, the holder goes momentarily EMPTY), _apply_fixed_size grows the
        # card to the new preset, _rerender_last rebuilds the canvas.  If ANY event-loop slice runs
        # between the grow and the rebuild -- a closing size-combo popup, matplotlib's own draw() -- the
        # user sees one frame of a BIG EMPTY card before the image lands = the "resize jump" (reproduced:
        # a 994x653 card with only the title strip painted, canvas still None).  Freezing THIS card's
        # repaints for the whole mutation (updates disabled -> the Agg buffer still renders, only the Qt
        # blit defers) makes the paint system composite one clean final frame: card at the new size WITH
        # the new-size canvas in place (verified: zero intermediate paintEvents).  The layout_changed
        # emit -- and so the sibling gravity repack it drives -- runs INSIDE the freeze, so this card's
        # final POSITION + size + image all land together; finally re-enables even if a rebuild raises.
        self.setUpdatesEnabled(False)
        try:
            self._reset_plot()
            self._apply_fixed_size()
            # If the Setting frame is open, GROW it immediately to match the new (taller) panel -- but
            # the high-water mark means a SMALLER size never snaps it shorter (#H3i-2).
            if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
                self._size_settings_popup()
            self._rerender_last()   # re-draw at the new size NOW, even if the source is stopped (else the
                                    # torn-down panel stays blank until the next hub tick -- same as _set_param)
            self.changed.emit()
            self.layout_changed.emit()
        finally:
            self.setUpdatesEnabled(True)

    def current_selection(self):
        """Return this panel's selection in the displayed coordinate frame."""

        from ..neutral_atom.core.selection import Selection

        selection = self._active_selection
        if selection is None:
            payload = self.config.params.get("selection")
            if isinstance(payload, Mapping):
                selection = Selection.from_dict(payload)
        if selection is None and self.plotter is not None:
            selection = self.plotter.to_data_figure().selection()
        if selection is None:
            selection = Selection()
        metadata = dict(selection.metadata)
        roi = getattr(self, "_roi_built", None)
        if roi and len(roi) >= 4:
            metadata["origin"] = [float(roi[0]), float(roi[2])]
        scope = selection.scope
        if self._grid_focus is not None:
            scope = (*scope, f"cell:{int(self._grid_focus.k)}")
        return Selection(selection.ranges, frame=selection.frame, scope=scope, metadata=metadata)

    def _selection_coordinates_for_binding(self) -> dict:
        """The plot's per-axis selection COORDINATES (a site map's centres, a 1-D panel's curve x) that
        :func:`live.region_binding` needs to bind an ROI drag to the consumed block's axes -- read from
        the SAME DataFigure adapter the selection itself comes from.  Empty when there is no plotter (a
        2-D / hist ROI needs none: pixel index / sample value carry no extra coordinate)."""
        if self.plotter is None:
            return {}
        try:
            df = self.plotter.to_data_figure()
        except Exception:
            return {}
        return {str(key): np.asarray(value)
                for key, value in dict(getattr(df, "_selection_coordinates", {})).items()}

    # ---- Analysis (curve fit + ROI): ONE state, ONE mutator --------------------------------------
    def _build_analysis_section(self, section_box, label_w) -> None:
        """Build the Setting popup's "Analysis" section (the fit + ROI umbrella).  A single ``action``
        combo picks what a drag does -- ``none`` / ``curve fit`` / ``ROI`` (each gated by capability) --
        followed, when a curve fit is offered, by the model chooser + the compact fix/seed editor + a
        result line.  Every widget DERIVES its value from state (:meth:`_refresh_analysis_controls`); no
        widget owns a private copy of "is fitting"."""
        models = _general_fit_models_for_kind(self._param_kind())
        # ROI is generic over EVERY plot kind whose value is a data array (kind_supports_roi -- the ONE
        # source): an image rectangle, a 1-D x-range, a distribution count-range and a site-centre
        # rectangle all reduce to roi_value + roi_frame.  (No longer gated on the y=view-axis image
        # test, which wrongly restricted ROI to 2-D image panels.)
        offers_roi = kind_supports_roi(self._param_kind())
        self.analysis_combo = self.fit_model_combo = self.fit_fix_seed = self.fit_result_label = None
        if not (models or offers_roi):
            return
        ana = section_box("Analysis")
        self.analysis_combo = FluentComboBox()
        self.analysis_combo.addItem("none", "none")
        if models:
            self.analysis_combo.addItem("curve fit", "fit")
        if offers_roi:
            self.analysis_combo.addItem("ROI", "roi")
        self.analysis_combo.setToolTip(
            "What a drag-selection on this panel does:\n"
            "  none      = just report the selected points"
            + ("\n  curve fit = fit the chosen model to the selection (the result overlays the plot;\n"
               "              a 2-D image centre fit ALSO publishes fit_x0/fit_y0/... as hub signals)"
               if models else "")
            + ("\n  ROI       = reduce the selected region to one scalar (publishes roi_value; also\n"
               "              roi_frame -- an image crop, or a 1-D / distribution / site sub-view)"
               if offers_roi else ""))
        self.analysis_combo.currentIndexChanged.connect(self._on_analysis_action_changed)
        ana.addWidget(FluentSettingRow("action", self.analysis_combo, label_width=label_w))
        self.fit_fix_seed_row = self.fit_result_row = None
        if models:
            self.fit_model_combo = FluentComboBox()
            for model in models:
                self.fit_model_combo.addItem(model.formula, model.key)
            self.fit_model_combo.setToolTip(
                "Curve-fit model for this panel's plot family (a 2d image offers the 2D-Gaussian\n"
                "'2D center'; a 1d / monitor / distribution offers the peak/decay models).")
            self.fit_model_combo.currentIndexChanged.connect(self._on_fit_model_changed)
            ana.addWidget(FluentSettingRow("model", self.fit_model_combo, label_width=label_w))
            # The compact per-parameter fix/seed editor + the result line only OCCUPY space while a
            # curve fit is actually on -- their rows stay in the layout but hide when the action is not
            # "fit" (the reveal-and-grow-down pattern the relim fixed lo/hi row uses), so the default
            # popup stays short and grows only when the operator turns the fit on.
            self.fit_fix_seed = _FitFixSeedEditor()
            self.fit_fix_seed.changed.connect(self._on_fit_fix_seed_changed)
            self.fit_fix_seed_row = FluentSettingRow("fix / seed", self.fit_fix_seed, label_width=label_w)
            ana.addWidget(self.fit_fix_seed_row)
            self.fit_result_label = FluentLabel("not fitted")
            self.fit_result_label.setWordWrap(True)
            self.fit_result_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self.fit_result_row = FluentSettingRow("result", self.fit_result_label, label_width=label_w)
            ana.addWidget(self.fit_result_row)
        self._refresh_analysis_controls()

    def _default_fit_model(self) -> str:
        """The model a fresh curve fit uses: the stored request's model FIRST (single source once fit
        is on), else this panel family's first offered model."""
        saved = self.config.params.get("fit_request")
        if isinstance(saved, Mapping) and saved.get("model"):
            return str(saved["model"])
        models = _general_fit_models_for_kind(self._param_kind())
        return models[0].key if models else "gaussian"

    def _build_fit_request_from_widgets(self, model_combo, fix_seed, selection):
        """Build a fresh fit request from a surface's OWN model combo + fix/seed editor + the selection
        (through the ONE :func:`build_fit_request`).  Used when a curve fit is first turned on from the
        Setting popup or the Edit tab -- each surface passes ITS widgets, so neither reads the other."""
        model = (str(model_combo.currentData()) if (model_combo is not None and model_combo.currentData())
                 else self._default_fit_model())
        fixed, initial = fix_seed.values() if fix_seed is not None else ({}, None)
        return build_fit_request(model, selection, fixed=fixed, initial=initial,
                                 coordinate_frame=selection.frame)

    def _retarget_fit_request(self, selection):
        """Rebuild the ACTIVE fit's request onto a NEW selection, preserving its model + fixed/initial
        (the stored request is the single source) -- the drag-to-retarget path."""
        from ..neutral_atom.core.fitting import FitRequest
        saved = self.config.params.get("fit_request")
        req = FitRequest.from_dict(saved) if isinstance(saved, Mapping) else None
        model = req.model if req is not None else self._default_fit_model()
        fixed = dict(req.fixed) if req is not None else {}
        initial = req.initial if req is not None else None
        return build_fit_request(model, selection, fixed=fixed, initial=initial,
                                 coordinate_frame=selection.frame)

    def set_fit_request(self, request) -> None:
        """The ONE fit mutator.  Its argument's presence is the single 'fit is on' state: it stores the
        request (or None) as ``config.params['fit_request']``, applies it to the live plotter overlay in
        place, re-derives any open Analysis control from that key (Setting + result line), and syncs the
        console's hub FitProcessor node -- create/retarget for a 2-D image fit, remove otherwise.  Every
        fit path (the Analysis combo, an Edit Fit action, a drag retarget) funnels through here, so the
        overlay, the two surfaces, and the hub node can never hold divergent fit state (#8)."""
        payload = request.to_dict() if request is not None else None
        self._set_param("fit_request", payload)          # store + apply overlay in place (apply_param)
        self._refresh_analysis_controls()
        if request is None:
            if callable(self.selection_clear_sink):
                self.selection_clear_sink(self, "fit")   # remove the hub FitProcessor this fit created
        elif callable(self.fit_node_sink):
            self.fit_node_sink(self, request)            # create/retarget the hub node (2-D image only)

    def _select_analysis_action(self, action, *, model_combo=None, fix_seed=None) -> None:
        """Apply a chosen post-drag action from EITHER surface: ``fit`` turns the curve fit ON (build +
        apply a request from the given model/fix-seed + the current selection), ``roi`` arms the crop,
        ``none`` clears both.  fit-vs-roi are the TWO independent analyses -- picking one clears the
        other so a single drag has one meaning; leaving ROI stops its node (#10)."""
        action = str(action or "none")
        if action == "fit":
            self.config.params["selection_action"] = "none"     # roi off; the fit lives in fit_request
            self.set_fit_request(
                self._build_fit_request_from_widgets(model_combo, fix_seed, self.current_selection()))
            return
        if self.config.params.get("fit_request"):
            self.set_fit_request(None)                          # leaving a fit removes overlay + node
        old = str(self.config.params.get("selection_action") or "none")
        self.config.params["selection_action"] = action
        if old == "roi" and action != old and callable(self.selection_clear_sink):
            self.selection_clear_sink(self, old)
        self._refresh_analysis_controls()
        self.changed.emit()

    def _on_analysis_action_changed(self, _index: int) -> None:
        combo = getattr(self, "analysis_combo", None)
        if combo is not None:
            self._select_analysis_action(combo.currentData(),
                                         model_combo=getattr(self, "fit_model_combo", None),
                                         fix_seed=getattr(self, "fit_fix_seed", None))

    def _on_fit_model_changed(self, _index: int) -> None:
        combo = getattr(self, "fit_model_combo", None)
        if combo is None:
            return
        editor = getattr(self, "fit_fix_seed", None)
        if editor is not None:
            editor.set_model(str(combo.currentData() or ""))
        if self.config.params.get("fit_request"):     # re-apply the active fit with the new model
            self.set_fit_request(self._build_fit_request_from_widgets(
                combo, editor, self.current_selection()))

    def _on_fit_fix_seed_changed(self) -> None:
        if self.config.params.get("fit_request"):     # re-apply the active fit with the new fix/seed
            self.set_fit_request(self._build_fit_request_from_widgets(
                getattr(self, "fit_model_combo", None), getattr(self, "fit_fix_seed", None),
                self.current_selection()))

    def _refresh_analysis_controls(self) -> None:
        """Re-derive THIS card's Setting Analysis controls from state (fit_request presence +
        selection_action) -- called from :meth:`set_fit_request`, the section build, and Setting show.
        The Edit tab derives its OWN copy the same way on show, so both are pure views (#8)."""
        _apply_analysis_state_to_widgets(
            self,
            action_combo=getattr(self, "analysis_combo", None),
            model_combo=getattr(self, "fit_model_combo", None),
            fix_seed=getattr(self, "fit_fix_seed", None),
            result_label=getattr(self, "fit_result_label", None))
        # The fix/seed editor + result line only take up space while a fit is on: reveal-and-grow-down
        # (the SAME pattern as the relim fixed lo/hi row), so the default popup stays short.  Set
        # visibility explicitly (isVisible() is unreliable before the popup is shown) and re-size only
        # when the popup is on screen (the build path sizes the popup itself afterwards).
        active = bool(self.config.params.get("fit_request"))
        for row in (getattr(self, "fit_fix_seed_row", None), getattr(self, "fit_result_row", None)):
            if row is not None:
                row.setVisible(active)
        if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
            self._size_settings_popup()

    def _fit_result_text(self, result=None) -> str:
        """The one-line fit-result string (shared by the Setting result label + the Edit status)."""
        result = result if result is not None else getattr(self.plotter, "_last_fit_result", None)
        if result is None:
            return "not fitted"
        if result.valid:
            quality = result.quality
            return (f"ok · {result.n_points} points · R²={quality.get('r2', float('nan')):.4g} · "
                    f"RMSE={quality.get('rmse', float('nan')):.4g}")
        return f"invalid · {result.status}"

    def _set_fit_result_text(self, result=None) -> None:
        label = getattr(self, "fit_result_label", None)
        if label is not None:
            label.setText(self._fit_result_text(result))

    def _set_param(self, key: str, value) -> None:
        """A declarative parameter edit: store, apply, mark dirty.

        Most params (colormap / bins / toggles …) change the plot's STRUCTURE, so they tear the plotter
        down + re-render.  ``relim`` is the exception: it is a LIVE axis adjustment (the dead-band-aware
        autoscale mode), so it pushes onto the existing plotter WITHOUT a teardown and reveals the fixed
        lo/hi row only in ``fixed`` mode -- the SAME effect the old hand-wired ``_on_relim_mode`` had,
        now reached through this one declarative path (#H3v-4b)."""
        if self.config.params.get(key) == value:
            return
        self._wait_render_idle()
        self.config.params[key] = value
        if key == "relim":
            # Flipping INTO ``fixed`` FREEZES the current view: seed lo/hi from what the plot shows
            # NOW (the plotter's live limits), never the stale/default 0..1 pair -- pinning a counts
            # histogram or a camera image to 0..1 empties it (every bar/pixel outside the range),
            # which reads as "the enlarged plot just died".  The operator then types exact bounds.
            if str(value) == "fixed" and self.plotter is not None \
                    and hasattr(self.plotter, "current_lims"):
                lo, hi = self.plotter.current_lims()
                self.config.params["fixed_lo"], self.config.params["fixed_hi"] = lo, hi
                for edit, val in ((getattr(self, "fixed_lo_edit", None), lo),
                                  (getattr(self, "fixed_hi_edit", None), hi)):
                    if edit is not None:
                        edit.setText(f"{val:g}")     # setText does NOT re-fire editingFinished
            if getattr(self, "fixed_lim_row", None) is not None:        # the Setting popup's lo/hi row
                self.fixed_lim_row.setVisible(str(value) == "fixed")    # (the Edit tab toggles its own)
                # Revealing/hiding the lo/hi row CHANGES the popup's content height.  The Setting popup
                # is a free-floating top-level Qt.Popup whose window size is fixed at open time, so a bare
                # setVisible reflows the rows INSIDE that fixed window -> the layout "jumps".  Re-run the
                # ONE sizing rule (_size_settings_popup: top-left anchored, grow-down, never re-position,
                # high-water so it never snaps shorter) so the window grows DOWNWARD to fit the new row --
                # smooth single-direction expand, not a jump (#fixed-lim-row-jump).
                if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
                    self._size_settings_popup()
            # A ZOOMED grid: route through the focused-param path so the mode lands on the enlarged
            # view AND is stored on the parked grid (the thumbnails carry it when the zoom returns);
            # otherwise apply to the live plotter directly (a GridPlot fans out to its cells itself).
            if self._grid_focus is not None:
                self._set_focused_grid_param(key, value)
                if str(value) == "fixed":
                    # land the just-seeded lo/hi on the parked grid too: its own flip-seed could only
                    # use the cells' auto envelope, but the operator froze the ENLARGED view -- the
                    # config values (read from that view above) are the authority.
                    self._set_focused_grid_param("fixed_lo", float(self.config.params.get("fixed_lo", 0.0)))
                    self._set_focused_grid_param("fixed_hi", float(self.config.params.get("fixed_hi", 1.0)))
            else:
                self._apply_lim_to_plotter()
            self.changed.emit()
            return
        # A GRID cell is currently ENLARGED (``_grid_focus`` set): a param edit must apply to the FOCUS view
        # and persist onto the PARKED grid, and NEVER mark a grid rebuild -- a grid rebuild swaps the focus
        # canvas out and bounces back to the thumbnail (the "adjusting a param jumps back to the main grid
        # view instead of staying zoomed" bug, #4).  Route through the dedicated focus-param path.
        if self._grid_focus is not None:
            self._set_focused_grid_param(key, value)
            self.changed.emit()
            return
        # Display-only knobs the plotter applies IN PLACE (e.g. log y-scale / bimodal-fit toggle): like
        # ``relim``, NO teardown -- a teardown + rebuild reflows the Qt holder so the plot visibly
        # resizes/flashes (#dis-resize).  Fall back to the rebuild path only for knobs it doesn't handle.
        if self.plotter is not None and self.plotter.apply_param(key, value):
            self.changed.emit()
            return
        # A STRUCTURE knob the plotter can't apply in place (e.g. a 2D colormap, a new history length):
        # mark a rebuild and schedule it on the DEBOUNCE timer rather than rebuilding inline.  A fast
        # burst of edits (scroll) restarts the timer, so the heavy teardown+rebuild runs ONCE after the
        # burst on the latest config.params -- not once per tick (the race source).  _build_plot itself
        # build-then-swaps at a fixed canvas size, so even the single rebuild never blanks the card.
        self._force_rebuild = True
        self._rebuild_timer.start()   # coalesce; _run_pending_rebuild fires after the burst settles
        self.changed.emit()

    def _run_pending_rebuild(self) -> None:
        """The debounce timer fired: perform the ONE coalesced rebuild for the latest config.params.
        Re-renders from the live/last namespace (so a stopped producer still rebuilds) -- _build_plot
        build-then-swaps atomically, so the card never blanks even if this rebuild raises."""
        if not self._force_rebuild:
            return
        self._wait_render_idle()
        self._rerender_last()

    def _apply_source(self) -> None:
        self._wait_render_idle()
        self.config.source = self.source_edit.text()
        self._compiled_source = self.config.source
        self._reset_plot()      # output shape may change with the expression
        # Force the console's NEXT tick to re-render this panel against a FRESH hub namespace (the
        # version gate honours -1), so re-picking a signal recovers a panel that previously errored
        # on a now-available signal -- never "remove the panel to recover" (#H3w-2).
        self._render_version = -1
        self._rerender_last()   # also re-evaluate on the last GOOD data at once (stopped node)
        self.apply_button.set_dirty(False)
        self.changed.emit()

    def _open_expr_editor(self) -> None:
        """Pop a NON-modal floating editor for the panel's source expression (the shared
        :class:`FluentFloatingEditor`): the panel behind stays VISIBLE and live, and the
        editor's Apply (below the box) writes the text back + re-renders so you watch the plot
        change.  Parented to this panel's TOP-LEVEL window so it shares the same screen scale
        (no DPI-mismatch shrink).  Re-clicking just raises the existing one."""
        existing = getattr(self, "_expr_editor", None)
        if existing is not None:
            existing.raise_(); existing.activateWindow(); return
        editor = FluentFloatingEditor(SOURCE_EXPR_HELP(), self.source_edit.text(), self.window(),
                                      title="Edit panel source expression")

        def _apply(text: str) -> None:
            # the source is one Python line -> collapse any newlines to spaces, write back to
            # the inline field, and apply live (the panel behind updates while the editor stays).
            self.source_edit.setText(" ".join(text.split("\n")).strip())
            self._apply_source()

        editor.applied.connect(_apply)
        editor.destroyed.connect(lambda *_: setattr(self, "_expr_editor", None))
        self._expr_editor = editor
        editor.show()

    def _rerender_last(self) -> None:
        """Re-render from the LAST namespace instead of waiting for the next hub tick.

        A display-only change (colormap / relim / a new source pick) tears the plotter
        down via :meth:`_reset_plot`; normally the next refresh rebuilds it.  But when the
        producing measurement is STOPPED the hub version is frozen, so ``_tick`` never calls
        ``refresh`` again -- the torn-down panel would stay blank (white).  Replaying the
        cached namespace rebuilds the plot at once.  No-op before the first render.

        Prefer the console's CURRENT shared namespace (a fresh hub snapshot = the SAME shot every
        other panel is on this tick) over this panel's own ``_last_namespace`` (a PAST tick): after a
        SIGNAL SWITCH or Stop/Start, replaying the stale cache would draw an OLDER shot than the
        siblings, so a 2D image and the sitemap would show different shots (#shot-coherence-on-switch).
        Fall back to the cached namespace only when no live one is available (or it lacks the new
        source -- e.g. a stopped producer whose frozen value lives only in the cache)."""
        live = None
        if callable(self.live_namespace_provider):
            try:
                live = self.live_namespace_provider()
            except Exception:
                live = None
        if live is not None and all(name in live for name in (self.config.inputs or ()) if name):
            self.refresh(live)
        elif self._last_namespace is not None:
            self.refresh(self._last_namespace)

    # ------------------------------------------------------------- data path
    def compose(self, namespace: dict[str, object], *, offthread: bool = False) -> bool:
        """Evaluate this panel's source against ``namespace`` and render the plot INTO ITS OFFSCREEN
        BUFFER -- phase 1 of the board's two-phase render: nothing is pushed to the screen here; the
        board presents every composed panel together in phase 2 so the whole board only ever shows ONE
        coherent shot.  Every failure lands on the gear/status -- a bad expression in one panel must
        never break the console or its siblings.

        ``offthread=True`` is the render-thread entry: identical logic, but a value that needs a
        STRUCTURAL (re)build -- a Qt-widget operation only the GUI thread may run -- is NOT rendered;
        the call returns False and the GUI thread re-runs the full compose for this panel when the
        batch lands.  In-place updates (the steady live stream) return True fully composed; status
        writes defer themselves inside set_status.  Returns True on the GUI thread always."""

        # A BLANK panel (a freshly added pure view, source not yet wired) sits
        # quietly with a hint -- it is decoupled, so it shows nothing until the
        # user picks a signal in its Setting.  Not an error.
        if not str(self._compiled_source).strip():
            self.set_status("pick a signal in Setting", error=False)
            return True
        try:
            # The three decoupled stages (#H3l): signal (per physical P slice) -> repeat reduce
            # (the measurement's own repeat axis, per the plot's repeat_mode) -> data-axis split.  The
            # measurement OWNS ``repeat``; the plot only chooses how to display it.
            value = self._signal_then_repeat(namespace)
            if offthread and self._needs_structural_build(value, namespace):
                return False
            self._render(value, namespace)
        except Exception as exc:
            # A failed eval (e.g. the bound signal is not in this namespace yet -- a still-WAITING
            # producer) must NOT latch the panel: keep the LAST GOOD namespace (set only on success
            # below), so re-picking a signal / a display change replays valid data, and a fresh hub
            # tick re-renders the moment the signal arrives -- never "remove the panel to recover"
            # (#H3w-2).
            self.set_status(str(exc).splitlines()[0][:160] or type(exc).__name__, error=True)
            return True
        # Remember the LAST SUCCESSFULLY-rendered namespace so a later display-only change (cmap /
        # relim / a source re-pick) can re-render immediately from it even when the producing
        # measurement is stopped (hub version frozen, no fresh tick) -- see _rerender_last.  Set ONLY
        # on success, so an error never overwrites it with a namespace that re-errors on replay.
        self._last_namespace = namespace
        shot = namespace.get("shot")
        self.set_status(f"shot {int(shot)}" if isinstance(shot, (int, float)) else "ok", error=False)
        return True

    def present(self) -> None:
        """Phase 2 of the board's two-phase render: flush this panel's composed buffer to the screen
        (schedule its Qt paint).  The board calls this for every panel it composed THIS tick, so they
        all repaint in ONE frame -- the board never shows a torn mix of shots."""
        if self.plotter is not None and self.canvas is not None:
            self.plotter.present()

    def refresh(self, namespace: dict[str, object]) -> None:
        """Compose + present this ONE panel now -- a single-card refresh (a source re-pick, the running
        task's mid-run panel).  The board (TaskConsole tick) instead splits these two phases across ALL
        its panels for cross-panel shot coherence; a lone card just composes then presents itself."""
        self.compose(namespace)
        self.present()

    def _signal_expr(self):
        """This panel's source as the ONE reusable :class:`SignalExpr` (the slot rule + the
        ``value = ...`` contract live there, shared with processors / pulse-scan).  Lazy import
        keeps the frontend module off neutral_atom's import graph (every neutral_atom use here
        is lazy)."""
        from ..neutral_atom.operations.signal_expr import SignalExpr
        return SignalExpr(self.config.inputs, self._compiled_source)

    def _signal_then_repeat(self, namespace: Mapping[str, object]):
        """Evaluate and reduce canonical signal tensors without rank inference."""
        from .live import reduce_repeat, repeats_with_data
        # A pulse panel's ``value`` is a STRUCTURED object (a sequence / PulseTableState), not an array --
        # it has no repeat axis and must NOT be float-coerced.  Read the bound signal (or the ``value =
        # ...`` expression result) as-is and hand it straight to _render / _build_plot, exactly as the
        # array pipeline hands a reduced block to the other kinds -- the kind (not the shape) decides how
        # its own data is consumed (a 2d reshapes to an image, sites uses centres, pulse uses the sequence).
        if self.config.kind == "pulse":
            self._repeat_cur = 1
            return self._signal_expr().evaluate(namespace)
        mode = self._repeat_mode_value()
        structure = self._bound_structure()
        if structure is not None:
            block = self._signal_expr().evaluate(namespace)
            b = self._validate_canonical_block(block, structure, self.config.inputs[0])
            valid = (namespace.get(SIG_VALID_KEY, {}) or {}).get(self.config.inputs[0])
            self._repeat_cur = repeats_with_data(b, valid=valid)
            if self.config.kind == "grid":
                return b
            return reduce_repeat(
                b, mode, valid=valid, hist=(self.config.kind == "hist"))

        block, had_repeat, valid = self._eval_signal_per_slice(namespace)
        b = np.asarray(block)
        if b.dtype.kind not in "iubf":
            b = np.asarray(b, dtype=float)
        if not had_repeat:
            self._repeat_cur = 1
            return b
        self._repeat_cur = repeats_with_data(b, valid=valid)
        return reduce_repeat(b, mode, valid=valid, hist=(self.config.kind == "hist"))

    @staticmethod
    def _validate_canonical_block(value, structure, name="signal") -> np.ndarray:
        from ..neutral_atom.core.signal_tensor import canonical_physical_shape
        array = np.asarray(value)
        point_shape = tuple(int(n) for n in structure["points_shape"])
        data_shape = tuple(int(n) for n in structure["data_shape"])
        # (points, *data_shape) = the per-slice tail of the ONE canonical (R,P,*data) shape (single source).
        expected_tail = canonical_physical_shape(1, point_shape, data_shape)[1:]
        points = expected_tail[0]
        if array.ndim != 1 + len(expected_tail) or tuple(array.shape[1:]) != expected_tail:
            raise ValueError(
                f"signal {name!r} violates its canonical schema: expected (R,{points},"
                f"{','.join(map(str, data_shape))}), got {array.shape}")
        return array

    def _eval_signal_per_slice(self, namespace: Mapping[str, object]):
        """Evaluate a transformed expression once per declared R slice."""
        expr = self._signal_expr()
        sig = expr.signal_for(namespace)
        slots = sig if isinstance(sig, list) else [sig]
        structured: dict[str, tuple[np.ndarray, Mapping[str, object]]] = {}
        for name, value in zip(self.config.inputs, slots):
            structure = self.structure_provider(name) if callable(self.structure_provider) and name else None
            if structure is not None:
                structured[f"slot:{len(structured)}"] = (
                    self._validate_canonical_block(value, structure, name), structure)
        raw_names = []
        for name in expr.direct_names():
            if name == "signal" or name not in namespace:
                continue
            structure = self.structure_provider(name) if callable(self.structure_provider) else None
            if structure is not None:
                structured[f"raw:{name}"] = (
                    self._validate_canonical_block(namespace[name], structure, name), structure)
                raw_names.append(name)
        if not structured:
            ns = dict(namespace); ns["signal"] = sig
            return expr.exec_in(ns), False, None
        repeats = {array.shape[0] for array, _structure in structured.values()}
        if len(repeats) != 1:
            raise ValueError(f"expression inputs have incompatible repeat sizes {sorted(repeats)}")
        repeat = repeats.pop()
        valid_by_name = namespace.get(SIG_VALID_KEY, {}) or {}
        repeat_valid = np.ones(repeat, dtype=bool)
        for name in (*self.config.inputs, *raw_names):
            mask = valid_by_name.get(name)
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.ndim != 2 or mask.shape[0] != repeat:
                    raise ValueError(f"signal {name!r} validity must be (R,P); got {mask.shape}")
                repeat_valid &= mask.any(axis=1)

        def _run(r):
            ns = dict(namespace)
            for name in raw_names:
                ns[name] = np.asarray(namespace[name])[r]
            sliced_slots = []
            for name, value in zip(self.config.inputs, slots):
                structure = self.structure_provider(name) if callable(self.structure_provider) and name else None
                sliced_slots.append(np.asarray(value)[r] if structure is not None else value)
            ns["signal"] = sliced_slots[0] if len(sliced_slots) == 1 else sliced_slots
            return np.asarray(expr.exec_in(ns), dtype=float)

        return np.stack([_run(r) for r in range(repeat)], axis=0), True, repeat_valid[:, None]

    def _bound_structure(self):
        """The producing node's ``{points_shape, data_shape, grid_shape}`` for this panel's bound
        signal -- authoritative for any IDENTITY source: the canonical ``value = signal`` OR a bare
        ``value = <the one input's name>`` (:func:`is_identity_source`), because naming the signal
        passes it through unchanged so the node's declared structure still describes it.  A transforming
        expression (``value = signal[0]-signal[1]``, ``value = np.log(f)``) rewrites the core shape (#H3o),
        so it returns ``None`` -> the reshape falls back to shape inference.  (The old gate compared only
        against ``value = signal``, so a named frame/scan binding like ``value = frame_judged`` lost its
        structure -> a 2-D frame grid collapsed to a histogram and facet axes vanished.)"""
        from ..neutral_atom.operations.signal_expr import is_identity_source
        if not is_identity_source(self._compiled_source, self.config.inputs):
            return None
        if not (self.config.inputs and callable(self.structure_provider)):
            return None
        try:
            return self.structure_provider(self.config.inputs[0])
        except Exception:
            return None

    def _coerce(self, value):
        # The per-kind reshape (image / lines / samples / one-value-per-site) lives WITH the plots
        # now -- ``live.coerce_panel_value`` owns every plot kind's INPUT contract.  The console only
        # GATHERS the inputs (value + the bound node's structure + params + repeat mode) and dispatches,
        # so the wiring layer holds ZERO per-kind reshape logic (a plot kind's input shape can change
        # without touching task_console).
        return coerce_panel_value(
            self.config.kind, value,
            structure=self._bound_structure(),
            params=self.config.params,
            repeat_mode=self._repeat_mode_value())

    def _sites_aux(self, namespace: Mapping[str, object]):
        """The site map's centres + camera underlay, resolved from the SAME producing node as its
        ONE occupancy signal (``config.inputs[0]``).  The occupancy's producing node (a
        Judge-occupancy processor) publishes occupied + centres + the judged frame together,
        so the console's ``sites_inputs_provider`` maps the occupancy signal -> (centres,
        image) signal names from that node's spec metadata, and rings + underlay are always
        the same shot.  The user therefore picks just ONE signal."""
        occ = self.config.inputs[0] if self.config.inputs else ""
        centers_name, image_name = (None, None)
        if callable(self.sites_inputs_provider) and occ:
            try:
                centers_name, image_name = self.sites_inputs_provider(occ)
            except Exception:
                centers_name, image_name = (None, None)
        centers = namespace.get(centers_name) if centers_name else None
        if centers is None:
            raise ValueError(
                f"site map needs the centres from `{occ}`'s producing node -- point it at an occupancy "
                "signal from a Judge-occupancy processor (it publishes occupied + centres + frame).")
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 4 or centers.shape[:2] != (1, 1) or centers.shape[-1] != 2:
            raise ValueError(
                "centres signal must be canonical (1,1,N,2) with data_shape=(N,2); "
                f"got {centers.shape}")
        centers = centers[0, 0]
        image = namespace.get(image_name) if image_name else None
        if image is not None:
            image = np.asarray(image, dtype=float)
            if image.ndim != 4 or image.shape[1] != 1:
                raise ValueError(
                    "site underlay must be canonical (R,1,H,W) with data_shape=(H,W); "
                    f"got {image.shape}")
            from .live import reduce_repeat
            # Reduce only the declared repeat axis.  P remains explicit until we
            # select its sole point; no squeeze/rank guess is involved.
            mode = self._repeat_mode_value()
            valid = (namespace.get(SIG_VALID_KEY, {}) or {}).get(image_name)
            image = reduce_repeat(
                image[:, 0], "replace" if mode == "create" else mode, valid=valid)
            image = np.asarray(image).reshape(image.shape[-2:])
        return centers[:, :2], image

    def _co_names(self) -> frozenset:
        """The hub-signal names this panel reads (cached) -- for the monitor roll-gate
        and the 2D coordinate-frame (ROI) lookup.  This is BOTH the identifiers the source
        expression names directly AND the picked input: the default ``value = signal`` form
        references the pseudo ``signal`` (not the real name), so ``config.inputs`` is folded
        in or version-gating would miss the input's updates."""
        key = (self._compiled_source, tuple(self.config.inputs))
        if key != self._ref_src:               # (re)derive on source OR slot change
            self._ref_src = key
            self._ref_names = self._signal_expr().co_names()   # names referenced + picked slots
        return self._ref_names

    def _source_coord_frame(self, namespace: Mapping[str, object] | None):
        """The ROI ([x, w, y, h]) of the camera signal this 2D panel's source
        reads, or None -- so the image axes can be real pixel coordinates."""
        frames = (namespace or {}).get(COORD_FRAMES_KEY)
        if not isinstance(frames, dict) or not frames:
            return None
        for name in self._co_names():
            if name in frames:
                return frames[name]
        return None

    def _monitor_source_key(self, namespace: Mapping[str, object] | None):
        """Version key of the signals this panel's source reads, or None when
        none are detectable.  None => caller rolls every tick (safe fallback)."""
        versions = (namespace or {}).get(SIG_VERSIONS_KEY)
        if not isinstance(versions, dict) or not versions:
            return None
        refs = [n for n in self._co_names() if n in versions]
        if not refs:
            return None
        return tuple(sorted((n, versions[n]) for n in refs))

    def _wait_render_idle(self) -> None:
        """Hold until the console's render worker is idle -- a GUI path must OWN the figure before
        mutating it (frontend/render_loop.py ownership protocol).  No-op on a standalone card and
        free when the worker is idle.  Never call from the render thread itself."""
        if callable(self.render_barrier):
            self.render_barrier()

    def set_selectors_enabled(self, on: bool) -> None:
        """The console header's "Selectors" switch for THIS card: remember the desired state and
        gate the CURRENT plotter now (in place -- no rebuild, no flash).  Every later (re)build /
        focus swap re-applies it through ``_apply_selectors_state``, so a fresh figure always
        inherits the switch."""
        self._wait_render_idle()
        was_on = self._selectors_on
        self._selectors_on = bool(on)
        self._apply_selectors_state()
        # Turning the Selectors switch OFF tears down this panel's analysis node (#1): the operator has
        # stopped driving it, so its FitProcessor / RoiProcessor must not keep republishing.  Routed
        # through the ONE console teardown seam (``selection_clear_sink`` -> ``_remove_panel_analysis``),
        # symmetric with panel removal and an Analysis action change; a no-op if this panel owns no node.
        if was_on and not on and callable(self.selection_clear_sink):
            self.selection_clear_sink(self, "selectors")

    def _apply_selectors_state(self) -> None:
        """Gate the current plotter's selector layer to the card's switch state (the ONE apply
        point every build / focus / tick path converges on).  Delegates to
        ``BaseLivePlot.set_selectors_active`` -- a safe no-op for a tool-less plot (the grid
        thumbnails) -- and aligns the canvas WHEEL policy with it: selectors ON isolates the
        wheel (in-plot scroll-zoom, the Edit-tab behaviour), OFF returns it to the board scroll.
        Only a plotter that actually HAS tools flips the wheel, so a grid card keeps scrolling
        the board either way."""
        plotter = self.plotter
        if plotter is None or not hasattr(plotter, "set_selectors_active"):
            return
        on = bool(self._selectors_on)
        plotter.set_selectors_active(on)
        canvas = self.canvas
        if canvas is not None and hasattr(canvas, "_zlc_isolate_wheel"):
            canvas._zlc_isolate_wheel = on and bool(plotter.interaction_handles())
        if canvas is not None:
            # Ownership barrier for every mouse path into this canvas (press / double-click /
            # wheel): wait for the render worker's in-flight batch before selector callbacks
            # mutate artists.  Hung here because this is the ONE apply point every build / focus
            # swap converges on, so a fresh or enlarged canvas always carries it (idempotent).
            canvas._zlc_render_barrier = self.render_barrier
        # Every family exposes the same DATA-selection callback.  Its DataFigure
        # adapter determines whether the gesture means x, xy, or value bounds;
        # selecting alone never implies either fit or ROI.
        bundles = list(getattr(getattr(plotter, "fig", None), "_zlc_grid_tools", ()) or ())
        if bundles:
            for cell_index, bundle in enumerate(bundles):
                area = getattr(bundle, "area", None)
                if area is not None:
                    area.callback = (lambda k=cell_index: self._forward_area_select(k)) \
                        if on and callable(self.area_select_sink) else None
        else:
            tools = getattr(getattr(plotter, "fig", None), "_zlc_tools", None)
            area = getattr(tools, "area", None)
            if area is not None:
                area.callback = self._forward_area_select \
                    if on and callable(self.area_select_sink) else None

    def _forward_area_select(self, cell_index: int | None = None) -> None:
        """Persist and forward one plot-independent selection.

        The displayed coordinate frame is retained for fitting.  Conversion to
        local frame indices belongs exclusively to the explicit ROI action.
        """
        bundles = list(getattr(getattr(self.plotter, "fig", None), "_zlc_grid_tools", ()) or ())
        tools = bundles[int(cell_index)] if cell_index is not None and int(cell_index) < len(bundles) \
            else getattr(getattr(self.plotter, "fig", None), "_zlc_tools", None)
        rng = list(getattr(getattr(tools, "area", None), "range", None) or [None] * 4)
        if any(v is None for v in rng):
            return
        data_figure = self.plotter.to_data_figure()
        target = data_figure.cell(int(cell_index)) if cell_index is not None else data_figure
        selection = target.selection()
        metadata = dict(selection.metadata)
        roi = self._roi_built
        if roi and len(roi) >= 4:
            metadata["origin"] = [float(roi[0]), float(roi[2])]
        scope = selection.scope
        if self._grid_focus is not None and not any(
                item == f"cell:{int(self._grid_focus.k)}" for item in scope):
            scope = (*scope, f"cell:{int(self._grid_focus.k)}")
        self._active_selection = type(selection)(
            selection.ranges, frame=selection.frame, scope=scope, metadata=metadata)
        self.config.params["selection"] = self._active_selection.to_dict()
        self.changed.emit()
        self.area_select_sink(self, self._active_selection)

    def _needs_structural_build(
        self,
        value,
        namespace: Mapping[str, object] | None,
        *,
        value_is_coerced: bool = False,
    ) -> bool:
        """Whether rendering ``value`` requires (re)building the plotter -- a Qt-widget operation
        ONLY the GUI thread may run.  False -> a pure in-place artist update (thread-agnostic).
        The ONE dirtiness rule: ``_render`` dispatches on it AND the render thread probes it
        first (``compose(offthread=True)``) so a structural panel is handed back to the board
        instead of touching Qt off-thread.  ``value`` is the raw ``_signal_then_repeat`` result
        (coerced here -- idempotent and zero-copy on the live path)."""
        if self._grid_focus is not None:
            return False                    # an enlarged cell's feed is in-place (or a silent skip)
        kind = self.config.kind
        if kind == "pulse":
            return True                     # a pulse panel rebuilds its structured figure every refresh
        if self.plotter is None or self._force_rebuild:
            return True
        if not value_is_coerced:
            value = self._coerce(value)
        if kind == "grid":
            if self._facet():
                # A cell-count change (the scan restructured) rebuilds through build-then-swap.
                return len(self._facet_cells(value)) != getattr(self.plotter, "n_cells", -1)
            return False                    # a recipe grid is a snapshot: never a per-tick rebuild
        if kind in ("2d", "1d", "sites") and tuple(np.shape(value)) != self._value_shape:
            return True
        if kind == "2d":
            # A ROI that *shifts* without resizing keeps the frame shape, but the
            # image's pixel coordinates moved -- rebuild so the axes track the ROI.
            roi = self._source_coord_frame(namespace)
            return (list(roi) if roi else None) != getattr(self, "_roi_built", None)
        return False

    def _render(self, value, namespace: Mapping[str, object] | None = None) -> None:
        # PERFORMANCE: push data with draw=False and queue ONE draw_idle per
        # panel -- rendering happens in Qt's paint pass (coalesced per frame),
        # never synchronously inside the refresh tick.  hist accepts any sample
        # count (HistogramFigure.update rebins itself), so a growing history
        # must NOT rebuild the whole plot every shot.
        #
        # A ``dis`` bins EXACTLY the array it is bound to -- it does NOT reach into any
        # calibration / processor to transform its input (decoupling: downstream never
        # knows an upstream node's internals).  Bind it to a processor's per-site COUNTS to
        # see the bimodal readout; bind it to a raw frame and it histograms the frame's
        # pixels, honestly.  Whatever the source gives the dis is what the dis bins.
        # A GRID cell is ENLARGED into a standalone plot-kind figure (``_grid_focus`` set): the live tick
        # must NOT rebuild the grid over it, or the enlarged view (and any lim edit on it) would bounce back
        # to the thumbnail.  A LIVE facet grid still feeds the ENLARGED view (update_cells' host-focus
        # channel touches ONLY the focus plotter; the thumbnails redraw once on unfocus); a recipe grid
        # stays a gated snapshot.
        if self._grid_focus is not None:
            if self.config.kind == "grid" and self._facet():
                parked = self._grid_focus
                per_cell = self._facet_cells(self._coerce(value))
                if len(per_cell) == parked.grid.n_cells:
                    parked.grid.update_cells(per_cell, focus=(self.plotter, parked.k))
                    # a hist focus feed can grow NEW threshold draggers inside update_core --
                    # re-park them to the switch (idempotent, cheap) so OFF stays display-only.
                    self._apply_selectors_state()
                    if self.plotter is not None:
                        self.plotter.compose()
            return
        # A recipe grid is a frozen artifact, not a view of this tick's scalar placeholder.
        # Once built, steady refreshes return before the canonical signal coercer (which correctly
        # rejects the numeric placeholder as grid data).  The FIRST tick and an explicit rebuild must
        # continue into ``_build_plot`` so the recipe carried by the producing node actually becomes a
        # plotter; the placeholder is intentionally ignored by that build branch.
        recipe_grid = self.config.kind == "grid" and not self._facet()
        if recipe_grid and self.plotter is not None and not self._force_rebuild:
            return
        if not recipe_grid:
            value = self._coerce(value)
        kind = self.config.kind
        if self._needs_structural_build(value, namespace, value_is_coerced=True):
            # STRUCTURAL (re)build -- creates the plotter + its Qt canvas, so this branch runs on
            # the GUI thread only (compose(offthread=True) hands such panels back to the board).
            # A pulse panel renders a STRUCTURED figure (a whole timeline) from its sequence
            # object; there is no in-place array ``update`` for it, so it (re)builds the
            # PulseSequenceFigure through _build_plot every refresh -- cheap (a static,
            # rarely-changing recipe) and always faithful.
            self._build_plot(value, namespace)             # build-then-swap: a failed build keeps the old figure
            self._force_rebuild = False                    # cleared ONLY after a clean build (else retry next tick)
            if kind == "grid":
                if self._facet():
                    # Every (re)build is followed by the Setting-rows sync: the build is the ONE
                    # point every kind-changing path converges on (a pick, a load, an expression edit).
                    self._sync_settings_param_rows()
                return
            if kind != "pulse":
                self._value_shape = (1,) if isinstance(value, float) else tuple(np.shape(value))
            if kind == "monitor":
                # The build already plotted this value as the first point; record
                # its source version so the next UNRELATED bump won't duplicate it.
                self._last_monitor_key = self._monitor_source_key(namespace)
            return
        # IN-PLACE update -- pure numpy + matplotlib artist work into the offscreen Agg buffer,
        # thread-agnostic (this is what the console's render thread executes every steady tick).
        if kind == "grid":
            if self._facet():
                # A LIVE facet grid: every tick moves the cells IN PLACE (update_cells -- the grid
                # counterpart of a standalone kind's update).
                self.plotter.update_cells(self._facet_cells(value))
                if self.canvas is not None:
                    self.plotter.compose()
            # a RECIPE grid is a SNAPSHOT built from its recipe: NEVER a per-tick redraw (#3).
            return
        if kind == "2d":
            self.plotter.update(np.asarray(value).ravel(), draw=False)
        elif kind == "monitor":
            # Roll ONE point per new sample of this monitor's own source.  With
            # several nodes, an unrelated node bumps hub.version and refreshes
            # every panel; without this gate the monitor would append a duplicate
            # point each time.  When the source's referenced signals are
            # undetectable (e.g. it reads via history("name")) we fall back to
            # rolling every tick -- never worse than before.
            key = self._monitor_source_key(namespace)
            if key is not None and key == self._last_monitor_key:
                return
            self._last_monitor_key = key
            self.plotter.roll(value, draw=False)
        elif kind == "sites":
            self.plotter.update(value, draw=False)
            if namespace is not None:           # refresh the camera underlay too
                _, image = self._sites_aux(namespace)
                self.plotter.set_background(image)
        elif kind == "1d" and self.config.params.get("xy") and np.ndim(value) == 2:
            # an x-y curve whose point count is unchanged (e.g. a finished scan
            # refreshed again): the x column is fixed at build time, so only the
            # y column updates in place (Live1D plots data_y vs its data_x[:, 0]).
            self.plotter.update(np.asarray(value)[:, 1], draw=False)
        elif kind == "1d":
            # a reduced scan curve refreshed in place: push the (points, ncols) y AND the current
            # repeat count so the "xN" ylabel tracks how many repeats have data.  Colours cycle by
            # column index inside Live1D (confocal-exact), so the panel no longer styles per-line.
            self.plotter.update(value, repeat_cur=int(getattr(self, "_repeat_cur", 1)), draw=False)
        else:  # hist / 1d-vector
            self.plotter.update(value, draw=False)
        # an in-place update can grow NEW selector handles (a hist whose threshold count changed
        # rebuilds its draggers inside update_core) -- re-park them to the switch state.  Idempotent
        # and cheap (a few attribute checks), so it never weighs on the live tick.
        self._apply_selectors_state()
        # COMPOSE this panel's buffer (blit: ~0.6 ms restore chrome bg + redraw data artists); nothing is
        # pushed to the screen here.  The board PRESENTS every composed buffer TOGETHER in phase 2 so the
        # whole board flips ONE coherent shot; a lone single-card refresh() presents right after.  A chrome
        # change full-renders into the buffer inside compose().
        if self.canvas is not None:
            self.plotter.compose()

    def _source_axis_label(self) -> str | None:
        """The y-axis label for this panel's sourced signal, taken from the PRODUCING
        node's :class:`SignalSpec` (``axes_provider``) -- so a plot of ``rate`` is
        labelled "loading rate" by the measurement that makes it, not a per-kind string.
        ``None`` when the source is a free expression or its source node declares no axis."""
        source = (self.config.source or "").strip()
        if not source.startswith("value ="):
            return None
        name = source[len("value ="):].strip()
        # The default source ``value = signal`` plots the picked input: resolve ``signal`` to
        # the real hub-signal name (``config.inputs[0]``) so its producing node's axis label
        # is found.
        if name == "signal":
            name = self.config.inputs[0] if self.config.inputs else ""
        return self._axis_label_for(name)

    def _axis_label_for(self, name: str | None) -> str | None:
        """The axis label a producing node declares for hub signal ``name`` (via
        ``axes_provider`` -> the node's :class:`SignalSpec`), or None."""
        if not name or not callable(self.axes_provider):
            return None
        try:
            axes = dict(self.axes_provider())
        except Exception:
            return None
        entry = axes.get(str(name))
        return entry[0] if entry else None

    def _panel_labels(self, xlabel: str, ylabel: str, zlabel: str = "") -> tuple[str, str, str]:
        """The (x, y, z) axis labels this panel draws.  A REPRODUCED figure seeds the SAVED labels as
        ``xlabel`` / ``ylabel`` / ``zlabel`` panel params (figure_viewer._seed_state, from the ONE
        ``SavedFigure.axis_labels`` source) so a reopened panel draws the SAME axes it was saved with --
        the params are then the single source; a LIVE panel carries no such params and keeps the kind's
        reconstructed defaults passed in here.  A NON-EMPTY override wins (``axis_labels`` only ever seeds
        non-empty labels); an absent OR empty param falls back to the default, so this never blanks a live
        axis (matching the xy-curve branch's own ``params.get('ylabel', '') or label``)."""
        p = self.config.params
        return (str(p.get("xlabel") or xlabel), str(p.get("ylabel") or ylabel), str(p.get("zlabel") or zlabel))

    # ------------------------------------------------------------- plot lifecycle
    def _fixed_lim_kwargs(self) -> dict:
        """``fixed_lo``/``fixed_hi`` kwargs for the plotter, ONLY when the lim mode is
        "fixed" (#8) -- otherwise omitted so the plotter keeps its tight/normal autoscale."""
        if self._relim() != "fixed":
            return {}
        return {"fixed_lo": float(self.config.params.get("fixed_lo", 0.0)),
                "fixed_hi": float(self.config.params.get("fixed_hi", 1.0))}

    def _view_kwargs(self, kind: str) -> dict:
        """The relim / limit view kwargs EVERY relim-capable kind passes to its plotter -- the ONE
        source for BOTH the live build (``_build_plot``) and the Edit snapshot (``PanelEditor.rebuild``),
        so no kind can silently omit relim/fixed.  This is exactly what regressed: the ``sites`` panel
        omitted these in both builders, so a Site map's lim mode / fixed lo-hi never re-rendered (#4).
        For a histogram the relim/fixed pins the VALUE (x) axis (the count y-axis is always auto, #3);
        every other kind pins its value axis (1D y-axis / 2D·sites clim)."""
        return {"relim_mode": self._relim(), **self._fixed_lim_kwargs()}

    def _build_plot(self, value, namespace: Mapping[str, object] | None = None) -> None:
        if panel_canvas is None:
            raise RuntimeError("matplotlib Qt canvas is not available")
        kind = self.config.kind
        size = self.config.size
        label = self.config.title or PANEL_KINDS[kind]
        if kind == "pulse":
            # A pulse panel renders its full timeline through the ONE pulse renderer in the PLOT LAYER
            # (``live.build_pulse_preview_plot`` -- the SAME one the editor + the reopened-recipe path use;
            # NEVER imported from the pulse_gui app -- see test_pulse_render_single_source), so a seeded
            # pulse figure is faithful (every digital channel / analog bus trace / repeat bracket), not a
            # flattened line.  The reproduction STATE (a PulseTableState) is an OBJECT the float-only hub
            # cannot carry, so it is resolved off the SAME producing node via ``pulse_state_provider`` --
            # the SAME "aux data from the producing node" wiring the site map uses for its centres/frame
            # (the hub ``value`` here is only a numeric placeholder).  The figure keeps its own spec-owned
            # geometry (frontend-owned); its own repeat brackets carry ×N, so there is no repeat_mode.
            from .live import build_pulse_preview_plot

            resolved = self.pulse_state_provider(self.config.inputs[0]) \
                if (callable(self.pulse_state_provider) and self.config.inputs) else None
            if resolved is None:
                raise ValueError(
                    "pulse panel needs a pulse figure's producing node -- point it at a loaded pulse "
                    "figure's fig_value signal (it carries the PulseTableState to reproduce).")
            state, node_include_off = resolved
            # The "show off rows" toggle is the panel's own display param (seeded from the saved value);
            # fall back to the node's recorded value when the param is unset -- so toggling re-renders.
            include_off = bool(self.config.params.get("include_always_off", node_include_off))
            # size drives the geometry like every other kind (config.size); interactions=True builds the
            # selector layer, then _apply_selectors_state (end of this build) PARKS it to the header's
            # "Selectors" switch -- OFF (default) keeps the Monitor card display-only exactly as before,
            # ON arms zoom / area / cross in place (the same rule the sites / 2d / hist / 1d branches
            # apply below).  The Edit tab stays the always-interactive surface.
            plotter, _channels, _repeat = build_pulse_preview_plot(
                state, include_always_off=include_off, size=size, interactions=True)
        elif kind == "grid":
            # A grid panel renders its per-site distribution / kernel grid through the ONE grid builder in
            # the PLOT LAYER (``live.build_grid_figure`` -- the SAME one ``na.load_figure(npz).plot()`` and
            # the report use).  Its reproduction RECIPE (a dict) is an OBJECT the float-only hub cannot
            # carry, so it is resolved off the producing node via ``grid_recipe_provider`` -- the SAME "aux
            # data from the producing node" wiring the pulse panel + site map use (the hub ``value`` here is
            # only a numeric placeholder).  Its geometry (cell count-driven size) is frontend-owned.
            # Build every cell's selector bundle; the card disables GridPlot's
            # notebook focus callback below so its own focus host remains the sole
            # double-click owner.
            facet = self._facet()
            if facet:
                # A FACET grid: the bound block IS the data -- slice it along the declared axis
                # (facet_cells, the ONE rule) and build through the ONE shared builder (the Edit-tab
                # snapshot uses the same one); every tick after this first build moves the cells in
                # place (update_cells in _render).
                plotter = self._build_facet_plotter(value, interactions=True)
            else:
                from .live import build_grid_figure

                recipe = self.grid_recipe_provider(self.config.inputs[0]) \
                    if (callable(self.grid_recipe_provider) and self.config.inputs) else None
                if recipe is None:
                    raise ValueError(
                        "grid panel needs a grid figure's producing node (a loaded grid figure's "
                        "fig_value signal) -- or pick a facet in Setting to expand a measurement "
                        "block's axis into cells.")
                recipe = self._grid_recipe_with_params(recipe)   # fold the panel's live display knobs in
                plotter = build_grid_figure(recipe, interactions=True, size=size, display=False)
            click_cid = getattr(plotter, "_click_cid", None)
            if click_cid is not None:
                plotter.fig.canvas.mpl_disconnect(click_cid)
                plotter._click_cid = None
        elif kind == "sites":
            centers, image = self._sites_aux(namespace or {})
            vec = np.asarray(value, dtype=float).reshape(-1)
            if len(vec) != len(centers):
                raise ValueError(
                    f"site-map value has {len(vec)} entries but the centers signal has {len(centers)} sites")
            plotter = panel_plot(
                centers, vec, kind="sites", size=size, interactions=True,
                image=image, roi_radius=site_ring_radius(centers),
                cmap=_resolved_cmap("sites", self.config.params),   # operator pick, else the kind default (ONE resolver)
                **self._view_kwargs("sites"),
                labels=self._panel_labels("Camera x (px)", "Camera y (px)", label),
                title=self.config.title or None)
        elif kind == "2d":
            arr = np.asarray(value, dtype=float)
            ny, nx = arr.shape
            # Coordinate axes ARE the source's pixel space: when the producing node
            # declares a spatial region, its endpoints [x_min, x_max, y_min, y_max]
            # give the axis ORIGIN (index 0 = x_min, index 2 = y_min); the image x/y
            # are the REAL pixels (x_min..x_min+nx), not 0..N -- so the axes match
            # the camera window and a selection maps straight back to a new region.
            roi = self._source_coord_frame(namespace)
            self._roi_built = list(roi) if roi else None
            if roi and len(roi) >= 4:
                x0, y0 = float(roi[0]), float(roi[2])
                xlabel, ylabel = "Camera x (px)", "Camera y (px)"
            else:
                x0, y0 = 0.0, 0.0
                xlabel, ylabel = "X (px)", "Y (px)"
            from ..neutral_atom.core.raster import RegularRaster
            data_x = RegularRaster((ny, nx), origin=(x0, y0))
            plotter = panel_plot(
                data_x, arr.ravel(), kind="2d", size=size, interactions=True,
                cmap=_resolved_cmap("2d", self.config.params),   # operator pick, else the kind default (ONE resolver)
                **self._view_kwargs("2d"),
                # x / y / colour-bar all via the ONE _panel_labels source: a LIVE 2D panel labels its
                # colour bar with the bound signal's own axis label (a camera frame's "Counts"); a
                # REPRODUCED 2D figure overrides x / y / colour bar with the SAVED labels seeded into the
                # panel params (_seed_state), so the reopened image draws the axes it was saved with.
                labels=self._panel_labels(xlabel, ylabel, self._source_axis_label() or ""),
                title=self.config.title or None)
        elif kind == "monitor":
            # Declared PANEL_PARAMS defaults via the ONE _resolved_param resolver (never a
            # re-typed consume-site literal); the >=20 floor mirrors the decl's lo bound.
            length = max(20, int(_resolved_param(kind, self.config.params, "length")))
            history = np.full(length, np.nan)
            plotter = panel_plot(
                np.arange(length, dtype=float), history, kind="monitor", size=size, interactions=True,
                show_dist=bool(_resolved_param(kind, self.config.params, "show_dist")),
                labels=self._panel_labels("Shots ago", self._source_axis_label() or label),
                **self._view_kwargs("monitor"),
                title=self.config.title or None)
            plotter.roll(float(value), draw=False)
        elif kind == "hist":
            plotter = panel_plot(
                np.asarray(value, dtype=float), kind="hist", size=size, interactions=True,
                bins=int(_resolved_param(kind, self.config.params, "bins")),
                ylog=bool(_resolved_param(kind, self.config.params, "ylog")),
                fit=str(_resolved_param(kind, self.config.params, "fit")),
                **self._view_kwargs("hist"),                # relim/fixed pins the VALUE (x) axis (#3)
                labels=self._panel_labels("Value", "Shots"), title=self.config.title or None)
        else:  # 1d
            arr = np.asarray(value, dtype=float)
            if self.config.params.get("xy") and arr.ndim == 2 and arr.shape[1] == 2:
                # an x-y curve (col0 = x, col1 = y): plot y vs the supplied x
                # (a scanned measurement's result panel uses this -- value =
                # column_stack([x_key, y_key]) grows the curve over the scan).
                data_x, vec = arr[:, 0], arr[:, 1]
                xlabel = str(self.config.params.get("xlabel", "X"))
                ylabel = str(self.config.params.get("ylabel", "")) or label
            else:
                # A reduced scan curve is ``(points, ncols)`` -- KEEP it 2-D so each column draws as
                # its own line (the one-dimensional data_shape, or one line per repeat in ``create`` mode); a
                # plain vector stays 1-D (one line).  npts = the point count either way.
                vec = arr if arr.ndim == 2 else arr.reshape(-1)
                npts = arr.shape[0] if arr.ndim == 2 else vec.size
                ylabel = self._source_axis_label() or label
                # A scan's y curve: draw it vs the companion x signal from the SAME
                # producing node (its axis label/unit come with it) -- so a 1d plot wired
                # to ``temperature_survival`` reads "Trap-off time (s)" on x (#3).  Falls
                # back to a per-site index when there is no companion x.
                x_name = None
                if self.config.inputs and callable(self.curve_x_provider):
                    try:
                        x_name = self.curve_x_provider(self.config.inputs[0])
                    except Exception:
                        x_name = None
                x_vals = (namespace or {}).get(x_name) if x_name else None
                x_arr = None if x_vals is None else np.asarray(x_vals, dtype=float).reshape(-1)
                if x_arr is not None and x_arr.size == npts:
                    data_x, xlabel = x_arr, (self._axis_label_for(x_name) or "X")
                else:
                    data_x, xlabel = np.arange(npts, dtype=float), "Site"
            plotter = panel_plot(
                data_x, vec, kind="1d", size=size, interactions=True,
                labels=self._panel_labels(xlabel, ylabel),
                **self._view_kwargs("1d"),
                title=self.config.title or None)
            # Colours cycle by column index inside Live1D (confocal-exact: grey, skyblue, ...; a lone
            # line is grey, every line alpha=1 / linewidth=1) -- no per-line styling needed here.
            # The plot shows the repeat count as a "xN" ylabel suffix (the panel computed it while
            # reducing the measurement's repeat axis).  Apply it NOW (an in-place update with the same
            # data) so the "xN" shows on the first build too -- not only after the next live tick.
            rc = int(getattr(self, "_repeat_cur", 1))
            plotter.repeat_cur = rc
            if rc != 1:
                plotter.update(plotter.data_y, repeat_cur=rc, draw=False)
        # ATOMIC build-then-swap: the new plotter is now fully built.  Capture the OLD canvas/figure,
        # INSERT the new canvas FIRST, then remove the old -- so the holder is NEVER empty during the
        # swap (no one-frame blank).  A build error above (e.g. the sites length check) returns before
        # here, leaving the previous figure untouched (#4 "图有时消失").  The card is setFixedSize, so the
        # swap cannot reflow its geometry -- the plot never visibly resizes/jumps either (#5 "图变大").
        old_canvas, old_plotter = self.canvas, self.plotter
        self.plotter = plotter
        # Re-apply the persisted Setting toggles (unit + x/y limits + x-window) to the FRESH plotter NOW
        # -- BEFORE wrapping it in the canvas -- so the canvas's construction render draws the FINAL
        # content ONCE.  Applying them AFTER the canvas (as it used to) rendered pre-knob content, then
        # re-rendered: 2+ full re-renders of a heavy 36-cell grid.  Each apply_param itself redraws the
        # whole figure, so suspend the plotter's per-knob draws; the canvas's own render is the single one.
        with self.plotter.suspend_draws():
            self._apply_display_params()
        # Monitor cards default to display-only: the selector layer is built but PARKED
        # inactive (see _apply_selectors_state below, gated by the header's "Selectors"
        # switch) and the wheel scrolls the board (isolate_wheel=False) instead of being
        # swallowed.  Switch ON arms the selectors in place -- Edit-tab behaviour on the
        # live board; _apply_selectors_state also flips the wheel policy to match.
        self.canvas = panel_canvas(self.plotter.fig, isolate_wheel=False)
        # The canvas OWNS its size: EmbeddedFigureCanvas setFixedSize's itself to its DPR-invariant
        # design size at construction (and renders its buffer synchronously), so the host no longer pins
        # setMinimumSize(sizeHint()) on top -- that two-pin + DPR-derived-sizeHint race is what ballooned
        # the figure.  The card is setFixedSize to hold the canvas + the proportional bottom padding the
        # trailing stretch absorbs.
        add_stretch = self.canvas_holder.count() == 0       # first build (no canvas, no stretch yet)
        # canvas pins to the TOP of the content (right below the grey title strip); the trailing
        # stretch is the proportional bottom padding (collapses to ~0 for a 1-row card).
        self.canvas_holder.insertWidget(0, self.canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        if add_stretch:
            self.canvas_holder.addStretch(1)
        # NOW retire the old canvas/figure -- the new one is already in the holder, so there is no blank
        # window between remove and insert (the swap is atomic from the user's view).
        if old_canvas is not None:
            self.canvas_holder.removeWidget(old_canvas)
            old_canvas.setParent(None)
            old_canvas.deleteLater()
        if old_plotter is not None and plt is not None and old_plotter.fig is not None:
            plt.close(old_plotter.fig)
        # park (or arm) the fresh plotter's selector layer to the header's "Selectors" switch --
        # a rebuild must inherit the current state, never come up with live selectors while OFF.
        self._apply_selectors_state()
        # The canvas already rendered the FINAL content at construction (the display knobs were applied to
        # the plotter ABOVE, before it was wrapped).  Re-render only if the selector-state pass dirtied the
        # figure -- otherwise a no-op, not a second full render of the heavy grid.
        self.canvas._zlc_draw_if_needed()
        self._place_setting_button()
        # A GRID panel is display-only, so its own focus-zoom is dormant -- THIS card catches the double-click
        # on the grid canvas and swaps to the clicked cell's standalone plot-kind figure (build_focus_plotter).
        if kind == "grid" and self._grid_focus is None:
            self._connect_grid_focus_click(self.plotter)
            # A rebuild that interrupted an ENLARGED cell (recorded by _teardown_plot) re-focuses the SAME
            # cell on the fresh grid -- a size/structure edit while zoomed stays zoomed (#no-focus-bounce).
            k = self._pending_refocus_k
            self._pending_refocus_k = None
            if k is not None and 0 <= k < int(getattr(self.plotter, "n_cells", 0)):
                self._focus_grid_cell(self.plotter, k)
        else:
            self._pending_refocus_k = None      # a non-grid rebuild has no cell to restore

    def _connect_grid_focus_click(self, grid) -> None:
        """Wire a double-click on the GRID canvas to enlarge the clicked cell into its standalone plot-kind
        figure (and Esc / another double-click to return).  The grid panel is display-only, so it has no
        selectors; this is the ONE handler the console adds for the focus-zoom -- it reuses the grid's own
        ``site_axes`` for hit-testing and :meth:`GridPlot.build_focus_plotter` for the enlarged view."""
        canvas = getattr(grid.fig, "canvas", None)
        if canvas is None:
            return
        self._grid_click_cid = canvas.mpl_connect("button_press_event", self._on_grid_canvas_click)

    def _on_grid_canvas_click(self, event) -> None:
        # Only a LEFT double-click toggles focus (mirror GridPlot._on_click -- some backends emit wheel /
        # middle as dblclick button 2/4/5).
        if not getattr(event, "dblclick", False) or getattr(event, "button", 1) != 1:
            return
        if self._grid_focus is None:
            grid = self.plotter
            axes = list(getattr(grid, "site_axes", []) or [])
            if event.inaxes in axes:
                self._focus_grid_cell(grid, axes.index(event.inaxes))
        else:
            self._unfocus_grid_cell()

    def _on_grid_focus_key(self, event) -> None:
        if getattr(event, "key", None) == "escape" and self._grid_focus is not None:
            self._unfocus_grid_cell()

    def _focus_grid_cell(self, grid, k: int) -> None:
        """Enlarge grid cell ``k`` into its STANDALONE plot-kind figure and SWAP it into this card's canvas,
        parking the grid (plotter + canvas) to swap back on unfocus.  The enlarged view is the STOCK ``2x2``
        panel of the cell's kind -- the SAME size the notebook path (:meth:`GridPlot.focus`) enlarges to,
        never the grid's own (possibly larger) preset -- CENTRED in the card's plot region, with the
        STANDARD relim view kwargs so a subsequent lim / fit / relim edit reaches it through the ordinary
        _set_param / _apply_lim_to_plotter path; the live tick is gated off (``_grid_focus`` set), so it
        never bounces back to the thumbnail."""
        if panel_canvas is None:
            return
        # the enlarged view is a standalone panel of the cell's per-site kind -> its standard relim view
        # kwargs.  interactions=True builds its selector layer; _apply_selectors_state (below) parks it
        # to the header's "Selectors" switch -- OFF (default) keeps the enlarged view display-only as
        # before, ON arms zoom / area / cross / the threshold drag (the same rule every _build_plot
        # branch applies).  The Edit tab stays the always-interactive surface.
        sub_kind = str(getattr(grid, "sub_plot_kind", "hist"))
        focus = grid.build_focus_plotter(
            k, size="2x2", interactions=True, **self._view_kwargs(sub_kind))
        focus_canvas = panel_canvas(focus.fig, isolate_wheel=False)
        # atomic swap-in (mirror _build_plot): insert the focus canvas, remove the grid canvas but do NOT
        # close the grid figure -- it is parked for the return.  A TOP stretch balances the holder's
        # trailing one, so the stock-size enlarged view sits at the CENTRE of the (larger) grid region
        # instead of pinned to the top-left.
        grid_canvas = self.canvas
        self.canvas_holder.insertWidget(0, focus_canvas, alignment=QtCore.Qt.AlignHCenter)
        # SAME stretch factor (1) as the holder's trailing addStretch(1): equal factors split the
        # spare height evenly above/below -- a factor-0 spacer would win nothing against the
        # trailing factor-1 stretch and the enlarged view would pin to the top, not the centre.
        self.canvas_holder.insertStretch(0, 1)
        self._focus_top_stretch = self.canvas_holder.itemAt(0)
        if grid_canvas is not None:
            self.canvas_holder.removeWidget(grid_canvas)
            grid_canvas.setParent(None)
        self.plotter = focus
        self.canvas = focus_canvas
        self._grid_focus = _GridFocus(grid=grid, grid_canvas=grid_canvas, k=int(k))
        self._grid_focus.key_cid = focus_canvas.mpl_connect("key_press_event", self._on_grid_focus_key)
        self._grid_focus.click_cid = focus_canvas.mpl_connect("button_press_event", self._on_grid_canvas_click)
        self._apply_display_params()        # re-apply unit / manual lims to the fresh focus plotter
        self._apply_selectors_state()       # park/arm the enlarged view's selectors to the switch
        focus_canvas.draw()
        self._place_setting_button()

    def _set_focused_grid_param(self, key: str, value) -> None:
        """Apply a param edit while a grid cell is ENLARGED (#4): keep it on the FOCUS view AND persist it onto
        the parked grid, and NEVER rebuild the grid (which would swap the focus canvas out and bounce back to
        the thumbnail).  The focus plotter (``self.plotter``) applies the sub-kind's knob in place; a knob it
        can't apply in place re-focuses the SAME cell -- rebuilding ONLY the enlarged view, staying zoomed."""
        parked = self._grid_focus
        if parked is None:
            return
        # Persist on the parked grid WITHOUT drawing (store_display_param): the stored params travel into
        # the save recipe and re-seed build_focus_plotter; the (invisible) thumbnails are NOT synchronously
        # repainted here -- _unfocus_grid_cell redraws them on return, so a Setting edit while zoomed costs
        # only the enlarged view's own in-place apply, never an N-cell repaint stall.
        try:
            if parked.grid.store_display_param(str(key), value):
                parked.dirty = True         # a thumbnail-affecting knob changed -> unfocus must repaint the grid
        except Exception:
            pass
        # Apply to the enlarged view in place.  A knob it can't apply live re-focuses the same cell ONLY
        # when the knob actually belongs to the per-site kind's own param set (bins/fit/ylog/cmap ...) --
        # an unrelated panel knob (e.g. ``repeat_mode``, which has no effect on a snapshot grid's enlarged
        # cell) is just stored, so it never triggers a visible rebuild/flash of the enlarged view.
        if self.plotter is not None and not self.plotter.apply_param(str(key), value):
            if any(d.key == str(key) for d in _panel_display_decls(self.config.kind, self._param_kind())):
                self._refocus_current_cell()   # rebuild ONLY the enlarged view, staying zoomed
        elif self.canvas is not None:
            self.canvas.draw_idle()

    def _remove_focus_stretch(self) -> None:
        """Drop the TOP spacer that centred the enlarged cell in the grid region (no-op when absent)."""
        stretch = self._focus_top_stretch
        self._focus_top_stretch = None
        if stretch is not None:
            self.canvas_holder.removeItem(stretch)

    def _refocus_current_cell(self) -> None:
        """Rebuild the ENLARGED cell view in place (same cell, current display params) -- used when the focus
        plotter cannot apply a knob live.  Swaps ONLY the focus canvas; the grid stays parked and
        ``_grid_focus`` stays set, so the view never bounces back to the thumbnail."""
        parked = self._grid_focus
        if parked is None or panel_canvas is None:
            return
        grid, k = parked.grid, parked.k
        old_focus, old_canvas = self.plotter, self.canvas
        sub_kind = str(getattr(grid, "sub_plot_kind", "hist"))
        focus = grid.build_focus_plotter(k, size="2x2", interactions=True,
                                         **self._view_kwargs(sub_kind))    # reads grid._display_params
        # (interactions=True + _apply_selectors_state below: the rebuilt enlarged view parks/arms its
        # selector layer to the header's "Selectors" switch, same as _focus_grid_cell)
        focus_canvas = panel_canvas(focus.fig, isolate_wheel=False)
        # replace the old focus canvas IN PLACE (the centring top stretch stays where it is)
        at = self.canvas_holder.indexOf(old_canvas) if old_canvas is not None else 0
        self.canvas_holder.insertWidget(max(0, at), focus_canvas, alignment=QtCore.Qt.AlignHCenter)
        if old_canvas is not None:
            self.canvas_holder.removeWidget(old_canvas)
            old_canvas.setParent(None)
            old_canvas.deleteLater()
        self.plotter = focus
        self.canvas = focus_canvas
        parked.key_cid = focus_canvas.mpl_connect("key_press_event", self._on_grid_focus_key)
        parked.click_cid = focus_canvas.mpl_connect("button_press_event", self._on_grid_canvas_click)
        self._apply_display_params()
        self._apply_selectors_state()
        focus_canvas.draw()
        if old_focus is not None and plt is not None and getattr(old_focus, "fig", None) is not None:
            plt.close(old_focus.fig)
        self._place_setting_button()

    def _unfocus_grid_cell(self) -> None:
        """Return from the enlarged cell to the grid: mirror any threshold drag back onto the grid cell, swap
        the parked grid canvas back in, and close the focus figure."""
        parked = self._grid_focus
        if parked is None:
            return
        grid, grid_canvas, k = parked.grid, parked.grid_canvas, parked.k
        focus, focus_canvas = self.plotter, self.canvas
        # copy an enlarged-view threshold cut back onto the grid cell (the grid thumbnail + save recipe read
        # it), noting whether it ACTUALLY changed -- an unfocus that edited nothing needs no repaint.
        cell = getattr(grid, "cell_renderer", None)
        threshold_changed = False
        if cell is not None and hasattr(cell, "sync_threshold_from_focus"):
            try:
                threshold_changed = bool(cell.sync_threshold_from_focus(k, focus))
            except Exception:
                pass
        # The parked grid kept its FULLY-RENDERED buffer (fit + x-window + thumbnails) untouched while zoomed,
        # so swapping it back shows it AS IT WAS.  Re-render ONLY when the zoom changed something the grid
        # shows -- a Setting edit persisted onto it (parked.dirty) or a threshold dragged in the focus view.
        # Otherwise just re-blit the valid buffer: the fit STAYS (fixing the "fit vanished after zoom-in-and-
        # back" bug the old unconditional redraw caused) AND unfocus is instant, never an N-cell re-render.
        needs_redraw = bool(parked.dirty) or threshold_changed
        self._grid_focus = None
        self._remove_focus_stretch()
        # swap the grid canvas back in FIRST (never a blank holder), then retire the focus canvas + figure
        if grid_canvas is not None:
            self.canvas_holder.insertWidget(0, grid_canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.plotter = grid
        self.canvas = grid_canvas
        if focus_canvas is not None:
            self.canvas_holder.removeWidget(focus_canvas)
            focus_canvas.setParent(None)
            focus_canvas.deleteLater()
        if focus is not None and plt is not None and focus.fig is not None:
            plt.close(focus.fig)
        if hasattr(grid, "discard_focus_fit"):
            grid.discard_focus_fit()        # release the (now-closed) enlarged cell's fit tracker entry
        if grid_canvas is not None:
            if needs_redraw and hasattr(grid, "_redraw_thumbnails"):
                try:
                    grid._redraw_thumbnails()       # self-contained: re-applies the fit / x-window too
                except Exception:
                    pass
                grid_canvas.draw()
            else:
                grid_canvas.update()                # valid buffer -> re-blit only (fit preserved, instant)
        self._place_setting_button()

    def _reset_plot(self) -> None:
        """Drop the plot so the next refresh rebuilds it (size/params/source changed)."""
        self._teardown_plot()
        self._value_shape = None

    def _teardown_plot(self) -> None:
        # If a grid cell is ENLARGED, the parked grid holds a SECOND figure/canvas -- close it too so a
        # teardown while focused leaks neither figure (self.plotter/self.canvas are the FOCUS view here).
        # Record the focused index so the NEXT _build_plot re-focuses the SAME cell: a rebuild-class edit
        # (size / source / structure param) made while zoomed lands back on the enlarged cell, never
        # silently bouncing to the main grid (#no-focus-bounce).
        parked = self._grid_focus
        self._grid_focus = None
        if parked is not None:
            self._pending_refocus_k = int(parked.k)
            self._remove_focus_stretch()
            if parked.grid_canvas is not None:
                parked.grid_canvas.setParent(None)
                parked.grid_canvas.deleteLater()
            if parked.grid is not None and plt is not None and getattr(parked.grid, "fig", None) is not None:
                plt.close(parked.grid.fig)
        canvas, plotter = self.canvas, self.plotter
        self.canvas = None
        self.plotter = None
        if canvas is not None:
            self.canvas_holder.removeWidget(canvas)
            # setParent(None) BEFORE deleteLater: removeWidget only detaches from the LAYOUT -- the
            # widget stays parented (and painted!) until the deferred delete runs, so the old figure
            # lingered on screen under the replacement (the ghost/overlap the size-change showed).
            canvas.setParent(None)
            canvas.deleteLater()
        if plotter is not None and plt is not None and plotter.fig is not None:
            plt.close(plotter.fig)

    def shutdown(self) -> None:
        self._wait_render_idle()            # the worker must not be composing into this figure
        self._rebuild_timer.stop()          # never fire a coalesced rebuild into a torn-down card
        self._force_rebuild = False
        self._teardown_plot()


def _acquisition_param_decls(repeat_default: int = 0) -> tuple:
    """The ONE acquisition knob EVERY measurement-layer node owns, declared ONCE (#H3n): ``Repeat`` =
    the depth of the repeat axis = how many passes/photos the data block keeps and AVERAGES, then STOPS
    -- with ``0`` = ∞ (roll forever, a live monitor showing the latest).  ONE number, 0 = infinite (the
    SAME semantics as the scan-repeat count) -- there is NO separate Free-run toggle.  ``repeat_default``
    is 0 for a CAMERA (a live monitor streams forever by default -- set Repeat=N to take exactly N
    photos) and 1 for a scan (run the sweep once; set 0 to keep re-running it live).  A real ``ParamDecl``
    so it auto-renders through the SAME form path as every measurement param."""
    from ..neutral_atom.core.params import ParamDecl
    return (
        ParamDecl(key="repeat", label="Repeat (0 = ∞)", kind="int", default=max(0, int(repeat_default)),
                  lo=0, hi=100000,
                  tooltip="How many passes/photos to keep & AVERAGE then STOP, or 0 = ∞ (roll forever, "
                          "a live monitor showing the latest).  A scan re-runs the whole sweep this many "
                          "times; a camera takes this many photos -- averaging them is a long exposure "
                          "that recovers the full site map.  How the repeats are DISPLAYED is the plot "
                          "panel's 'repeat mode' Setting (average / add / replace / roll / create)."),
    )


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
            signal_expr_factory=lambda title: _SignalExprWidget(
                signals_provider=self._signals_provider,
                sources_provider=self._sources_provider,
                formats_provider=self._formats_provider,
                labels_provider=self._short_names_provider,
                title=title),
            pulse_slots_factory=lambda: _PulseSlotsWidget(),
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
        # decl + widget kept per key so the dependent-combo wiring (pulse_param / pulse_slots)
        # can find its sibling ``path`` field and repopulate from the template it names.
        self._pulse_param_decls: dict[str, object] = {}
        self._pulse_slots_decls: dict[str, object] = {}
        self._handlers: dict[str, object] = {}    # param key -> ParamWidgetHandler
        self._decls: dict[str, object] = {}        # param key -> ParamDecl
        spec = self.current_spec()
        if spec is None:
            return
        # the spec's declared params PLUS the auto-injected acquisition knob (repeat, 0 = ∞) --
        # ONE list so both the label-width and the widget loop see the same declarations.
        decls = list(spec.params) + list(self._acquisition_params)
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
            if kind == "pulse_param":
                self._pulse_param_decls[decl.key] = decl
            elif kind == "pulse_slots":
                self._pulse_slots_decls[decl.key] = decl
        # Wire each pulse_param combo to its source template field (done AFTER the build loop
        # so the source field exists regardless of declaration order), then fill it once.  This
        # is the form's only inter-field reactivity: changing the template repopulates the
        # scan-target choices.
        for key, decl in self._pulse_param_decls.items():
            src = self._sibling_path_widget(decl)
            if src is not None:
                src.changed.connect(lambda *_a, k=key: self._repopulate_pulse_param(k))
            self._repopulate_pulse_param(key)
        # Wire each pulse_slots widget to its template path (same dependency pattern as
        # pulse_param): when the template changes, rebuild the auto-form's per-slot rows.
        for key, decl in self._pulse_slots_decls.items():
            src = self._sibling_path_widget(decl)
            if src is not None:
                src.changed.connect(lambda *_a, k=key: self._repopulate_pulse_slots(k))
            self._repopulate_pulse_slots(key)
        self._refresh_start_enabled()

    def _sibling_path_widget(self, decl):
        """The ``path`` widget named in ``decl.depends_on`` (the template a dependent
        pulse_param / pulse_slots field introspects), or None."""
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
        try:
            from ..neutral_atom.operations.measurements.pulse_scan import (
                _resolve_probe_template,
                _semantic_api_names,
                _semantic_scan_names,
            )
            state = _resolve_probe_template(path)
        except Exception:
            widget.rebuild([], [], program_id="")
            return
        api_names = _semantic_api_names(state)
        api_rows = [
            (slot.name, api_names[index], str(slot.kind), str(slot.target), str(slot.unit),
             float(state._read_api_field(slot)))
            for index, slot in enumerate(state.api_slots)
        ]
        scan_names = _semantic_scan_names(state)
        scan_rows = [(scan_names[i], str(s.kind), str(s.target), str(s.unit), s.label)
                     for i, s in enumerate(state.scan_slots)]
        code = str(getattr(state, "scan_code", "") or "")
        if not code.strip() and getattr(state, "scan_table", None):
            code = "scan_table = np.array(" + repr([list(row) for row in state.scan_table]) + ", dtype=float)"
        payload = state.to_dict()
        program_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        widget.rebuild(api_rows, scan_rows, hardware_program=code, program_id=program_id)

    def _repopulate_pulse_param(self, key: str) -> None:
        """Fill a ``pulse_param`` combo from the pulse template named in its ``depends_on``
        path field: each item's text is the human label, its data the ``"kind:target"`` token
        the measurement consumes.  Preserves the current selection across a reload (the editable
        combo lets an unknown saved target round-trip).  A bad/empty path -> empty combo (Start
        stays disabled via ``required``)."""
        combo = self._widgets.get(key)
        decl = self._pulse_param_decls.get(key)
        if combo is None or decl is None:
            return
        src = self._sibling_path_widget(decl)
        path = src.text() if src is not None else ""
        keep = combo.currentData() or combo.currentText()
        items: list[tuple[str, str]] = []   # (label, "kind:target")
        try:
            # lazy import (frontend stays off neutral_atom's import-time graph; this is a
            # GUI action) -- reuse the ONE template resolver + the single-source enumerator.
            from ..neutral_atom.operations.logic import CalibrateReadoutTask
            from ..neutral_atom.timing import enumerate_pulse_params
            state = CalibrateReadoutTask._resolve_template(path)
            items = [(label, f"{kind}:{target}") for kind, target, label in enumerate_pulse_params(state)]
        except Exception:
            items = []
        combo.blockSignals(True)
        combo.clear()
        for label, token in items:
            combo.addItem(label, token)
        tokens = [token for _label, token in items]
        if keep in tokens:
            combo.setCurrentIndex(tokens.index(keep))
        elif keep:
            combo.setCurrentText(str(keep))          # round-trip a saved token not in this template
        elif items:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    # ------------------------------------------------------------- value read
    def collect_values(self) -> dict[str, object]:
        """Read every parameter back BY KIND (no eval) into a build kwargs dict --
        each value is its handler's ``read`` of the widget (the coercion lives in
        PARAM_WIDGETS, one rule per kind)."""
        return {key: self._handlers[key].read(widget) for key, widget in self._widgets.items()}

    def refresh_on_show(self) -> None:
        """Re-poll providers and rebuild every dynamic combo (signals + pulse_params), so
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
            # a DEPENDENT combo (pulse_param / pulse_slots) repopulates via the form's per-key
            # hook (it reads the sibling template); a signal picker refills from live providers.
            if kind == "pulse_param":
                repopulate = lambda _w, k=key: self._repopulate_pulse_param(k)
            elif kind == "pulse_slots":
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
        # A pulse_param combo's choices come from its (also-seeded) template field, so
        # repopulate AFTER seeding so the saved "kind:target" token lands on a real item
        # (the editable combo round-trips it even if the template changed).
        for key in getattr(self, "_pulse_param_decls", {}):
            self._repopulate_pulse_param(key)
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
        # starts a node -- that lives on the Logic tab.  ``is_plot`` is always
        # True (a board panel's role is always "plot"); it is kept as a local for
        # the section gates below so the structure stays legible.
        is_plot = card.config.role == "plot"

        # ---- Panel: rename this panel right here in the Edit (kept in sync with the Setting
        # popup's title field; both go through the sealed title API).  Saves a trip back to
        # the Setting popup just to relabel.
        if is_plot:
            section("Panel")
            self.title_edit = FluentLineEdit(card.config.title)
            self.title_edit.setPlaceholderText("panel title…")
            self.title_edit.setToolTip("Rename this panel (also the default save name).")
            self.title_edit.textChanged.connect(self._edit_title)
            col.addWidget(FluentSettingRow("title", self.title_edit, label_width=scaled_px(96, minimum=72)))

        # ---- Acquisition: the editable parameters of the DATA SOURCE behind this
        # panel.  A panel is a VIEW; the LOGIC NODE whose published signals it reads
        # declares what its source exposes via acquisition_parameters() -- a
        # raw-frame panel reads the camera Measurement (exposure / ROI), an
        # occupancy panel reads the OccupancyProcessor.  Each field is PREFILLED with
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
                # pass the signal providers so a signal-kind param (e.g. the OccupancyProcessor's
                # 'source') renders the SAME nested-by-producer picker the logic-node Edit uses --
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
        if is_plot and functional:
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
        if is_plot:
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

        # Fit + manual limits are a PLOT concern (and a plot panel is always
        # role "plot", so they always build here).  A LOGIC NODE's Edit (on the
        # Logic tab, a LogicNodeEditor) deliberately carries no curve fit -- fitting
        # a curve is a plotter action, so you Add a Plot panel pointed at the
        # signals the node publishes for that.  A PLOT keeps the FULL DataFigure
        # model set (#176).
        if is_plot:
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

            # ---- Analysis: the SAME curve fit the Setting popup's Analysis section offers, listing ONLY
            # the models that match this panel's plot family (a hist now offers the 1-D family ALONGSIDE
            # its own bimodal knob -- they are two parallel fits).  Every action routes through the ONE
            # card mutator (card.set_fit_request), so the Edit fit combo and the Setting Analysis combo
            # are both views of config.params['fit_request'] and can never disagree (#8).  ``fit_combo``
            # stays the attribute name the model picker + tests key off; ``do_fit`` no-ops when None.
            fit_models = _general_fit_models_for_kind(
                self.card._param_kind() if self.card is not None else self.config.kind)
            self.fit_combo = None
            self.ed_fix_seed = None
            if fit_models:
                section("Analysis")
                self.fit_combo = FluentComboBox()
                for model in fit_models:
                    self.fit_combo.addItem(model.formula, model.key)
                self.fit_combo.setFixedWidth(scaled_px(150, minimum=120))
                self.fit_combo.setToolTip(
                    "Curve-fit model for this panel's plot family.  A 2d image offers the 2D-Gaussian\n"
                    "'2D center'; a 1d / monitor / distribution offers the peak/decay models.")
                self.fit_combo.currentIndexChanged.connect(self._on_edit_fit_model_changed)
                fit_btn = FluentButton("Fit", color=ACCENT)
                fit_btn.clicked.connect(self.do_fit)
                clear_btn = FluentButton("Clear", color=GREY)
                clear_btn.clicked.connect(self.clear_fit)
                # model row: the picker on the left, the Fit / Clear actions on the right.
                model_row = _inline(self.fit_combo, trailing=fit_btn)
                model_row.layout().addWidget(clear_btn, 0)
                col.addWidget(FluentSettingRow("model", model_row, label_width=proc_lw))
                # the compact typed per-parameter fix/seed editor (#1b): the SAME reusable widget the
                # Setting popup builds, writing into the SAME FitRequest.fixed / .initial via build_fit_request.
                self.ed_fix_seed = _FitFixSeedEditor()
                self.ed_fix_seed.changed.connect(self._on_edit_fit_fix_seed_changed)
                col.addWidget(FluentSettingRow("fix / seed", self.ed_fix_seed, label_width=proc_lw))
                self._refresh_edit_analysis()

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
        """Apply a curve fit from the Edit tab through the ONE card mutator, so the live overlay (a
        DISPLAY of the per-panel FitProcessor node's published params), the hub node, and the Setting
        popup's Analysis controls all follow the same config.params['fit_request'] state."""
        if self.fit_combo is None or self.card is None or not self.fit_combo.currentData():
            return
        selection = self.card.current_selection()
        request = self.card._build_fit_request_from_widgets(
            self.fit_combo, getattr(self, "ed_fix_seed", None), selection)
        self.card.set_fit_request(request)                 # per-panel fit node + live overlay + Setting widgets
        self.status.setText(f"fit {self.fit_combo.currentText()}: applied (see panel)")

    def clear_fit(self) -> None:
        if self.card is not None:
            self.card.set_fit_request(None)                # clears live overlay + removes the per-panel node
        self.status.setText("fit cleared")

    def _on_edit_fit_model_changed(self, _index: int) -> None:
        if self.fit_combo is None:
            return
        if getattr(self, "ed_fix_seed", None) is not None:
            self.ed_fix_seed.set_model(str(self.fit_combo.currentData() or ""))
        if self.card is not None and self.card.config.params.get("fit_request"):
            self.do_fit()                                   # re-apply the active fit with the new model

    def _on_edit_fit_fix_seed_changed(self) -> None:
        if self.card is not None and self.card.config.params.get("fit_request"):
            self.do_fit()

    def _refresh_edit_analysis(self) -> None:
        """Re-derive the Edit tab's Analysis controls (fit model combo + fix/seed) from the card's
        stored ``fit_request`` -- so the Edit picker always shows the SAME fit the Setting popup set,
        the two surfaces being pure views of the single source (#8)."""
        if self.card is None or self.fit_combo is None:
            return
        _apply_analysis_state_to_widgets(
            self.card, action_combo=None, model_combo=self.fit_combo,
            fix_seed=getattr(self, "ed_fix_seed", None), result_label=None)

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
            return                          # no Limits section (a non-"plot" role)
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
            return                          # no Limits section (a non-"plot" role)
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


class _PanelBoard(QtWidgets.QWidget):
    """Absolute-positioned canvas the cards live on (drag + snap layout)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

    def arrange(self, cards: Sequence[PanelCard]) -> None:
        # ``col``/``row`` ARE the card's pixel top-left (gravity-packed by :func:`pack`); place verbatim
        # and reserve a GAP margin past the lowest-right card so the board scrolls cleanly.
        max_x = max_y = 0
        for card in cards:
            x, y = card.config.col, card.config.row
            card.move(x, y)
            max_x = max(max_x, x + card.width())
            max_y = max(max_y, y + card.height())
        self.setMinimumSize(max_x + GAP, max_y + GAP)


# ====================================================================== logic tab
class LogicNodeRow(FluentFrame):
    """One LOGIC NODE's CARD on the Logic tab: a status dot + name + (kind) + status on
    the top line with Start / Stop / Edit / Remove, and a second line listing the
    signals it PUBLISHES with their array shape (``occupied [per-site (N,)], rate
    [scalar]``).  The dot follows the run state (grey=stopped / green=running /
    red=error), confocal's tab-icon colour map applied to a card.  Start / Stop act
    here directly; the full param form is in the node's Edit tab
    (:class:`LogicNodeEditor`)."""

    edit_requested = QtCore.pyqtSignal(object)     # "Edit" -> open the node's Edit tab
    remove_requested = QtCore.pyqtSignal(object)
    start_requested = QtCore.pyqtSignal(object)    # "Start" -> build + run the node
    stop_requested = QtCore.pyqtSignal(object)     # "Stop"  -> stop the node

    # confocal gui_combine colour map (INIT=grey / RUNNING=green / STOP/ERROR=red).
    STATE_COLORS = {"stopped": GREY, "running": GREEN, "error": RED}

    def __init__(self, node: LogicNodeConfig, parent=None):
        super().__init__(parent)
        self.node = node
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(scaled_px(12), scaled_px(8), scaled_px(12), scaled_px(8))
        outer.setSpacing(scaled_px(4, minimum=3))
        # --- top line: status + name + (kind) + Start / Stop / Edit / Remove --------
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(scaled_px(10, minimum=6))
        self.dot = FluentStatusDot(size=14)
        self.dot.set_color(GREY)
        self.name_label = FluentLabel(node.title)
        self.kind_label = FluentLabel(f"({node.kind})")
        self.kind_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.status_label = FluentLabel("stopped")
        self.status_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.start_button = FluentButton("Start", color=GREEN)
        self.start_button.setFixedWidth(scaled_px(60, minimum=48))
        self.start_button.clicked.connect(lambda: self.start_requested.emit(self))
        self.stop_button = FluentButton("Stop", color=ORANGE)
        self.stop_button.setFixedWidth(scaled_px(56, minimum=46))
        self.stop_button.clicked.connect(lambda: self.stop_requested.emit(self))
        self.stop_button.setEnabled(False)
        edit_button = FluentButton("Edit", color=ACCENT)
        edit_button.setFixedWidth(scaled_px(56, minimum=46))
        edit_button.clicked.connect(lambda: self.edit_requested.emit(self))
        remove = FluentButton("Remove", color=GREY)
        remove.setFixedWidth(scaled_px(82, minimum=66))
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        top.addWidget(self.dot, 0)
        top.addWidget(self.name_label, 0)
        top.addWidget(self.kind_label, 0)
        top.addWidget(self.status_label, 1)
        for b in (self.start_button, self.stop_button, edit_button, remove):
            top.addWidget(b, 0)
        outer.addLayout(top)
        # --- published-signals legend: one signal per line (name | shape | meaning) ---
        # Monospace so the name/shape columns ALIGN down the rows (a readable table, not a
        # run-on line).
        self.publishes_label = FluentLabel("")
        # WRAP, never extend the row horizontally: a logic-node card lives in a vertical list with NO
        # horizontal scroll (#2).  The publishes legend is name + shape only (short, fits) -- the longer
        # per-signal meaning lives in the tooltip, so nothing forces the card wider than the column.
        self.publishes_label.setWordWrap(True)
        self.publishes_label.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none; "
            "font-family: Consolas, 'DejaVu Sans Mono', monospace;")
        outer.addWidget(self.publishes_label)

    def set_state(self, state: str, *, status: str = "") -> None:
        """Reflect the node's run state on the dot + status text + Start/Stop enable."""
        self.dot.set_color(self.STATE_COLORS.get(state, GREY))
        self.status_label.setText(status or state)
        colour = RED if state == "error" else GREY
        self.status_label.setStyleSheet(f"color: {colour}; background: transparent; border: none;")
        running = state == "running"
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def set_publishes(self, rows) -> None:
        """Show the node's outputs as a SHORT table -- ONE signal per line, ``name`` + ``shape`` only::

            publishes:
              occupied   (35,)
              rate       scalar

        The per-signal MEANING goes in the label's tooltip (hover), NOT inline -- so the card never
        grows wider than its column and the Logic list needs no horizontal scroll (#2).  ``rows`` is
        ``[(name, shape, description)]`` (shapes AUTO-EXTRACTED via ``logic.describe_shape``; meanings
        from the node's ``output_specs``); a pending shape (``—``) just means no value yet."""
        rows = list(rows)
        if rows:
            name_w = max(len(str(n)) for n, _, _ in rows)
            shape_w = max(len(str(s)) for _, s, _ in rows)
            lines = [f"  {str(n):<{name_w}}  {str(s):<{shape_w}}".rstrip() for n, s, _ in rows]
            text = "publishes:\n" + "\n".join(lines)
            tip = "\n".join(f"{n} {s} — {d}" for n, s, d in rows if d)   # meanings on hover, off the card
        else:
            text, tip = "publishes: (nothing on the hub)", ""
        if text != self.publishes_label.text():       # skip churn: shapes refresh each tick
            self.publishes_label.setText(text)
            self.publishes_label.setToolTip(tip)


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
        self.running_nodes = list(running_nodes)
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
        # id(card) -> the ONE analysis LogicNodeRow this panel owns (a FitProcessor XOR a RoiProcessor).
        # An analysis node is keyed by the PANEL, never by the source signal: two panels on the SAME
        # source therefore own DISTINCT nodes with distinct output names, and create/retarget/teardown all
        # pivot on the card (the single fix for the old source-signal keying, which made every panel on one
        # source share -- and asymmetrically fail to tear down -- one node).
        self._panel_analysis: dict[int, "LogicNodeRow"] = {}
        # id(card) -> the panel's ``<slug>_region`` hub signal name (the drawn Selection AS a signal).
        # STABLE per panel: a re-drag republishes the SAME name, and the analysis node CONSUMES it, so a
        # region is a real, reusable per-panel signal -- retarget is a republish, not a set_selection poke.
        self._panel_region_name: dict[int, str] = {}
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

        # Multi-rate refresh: the timer ticks at the BASE interval (the smallest panel
        # update_ms, which divides every other so the rates co-align); each panel redraws
        # every update_ms // base ticks.  _tick_count counts base ticks since the last re-base.
        self._tick_count = 0
        self._base_interval_ms = int(self.state.interval_ms)
        # Display reads each signal at its OWN latest value (one snapshot per tick).  Cross-signal coherence
        # where it matters is FREE: a readout processor co-publishes occupied + centres + frame_judged in one
        # publish, so their latest values are always the same physical shot (sitemap rings == frame_judged
        # underlay == frame_judged 2D).  A live camera repeat block shows its newest, fullest ring -> live.

        # The ONE background render thread: every steady-tick compose (numpy prep + matplotlib
        # artist updates + Agg rasterisation) runs there; the GUI thread only schedules, presents
        # the finished front buffers, and serves interaction -- so a slow render lowers the plot
        # frame rate, never the UI's responsiveness (see frontend/render_loop.py for the ownership
        # protocol).  Structural builds (Qt widgets) come back to the GUI via _on_render_batch.
        # Created BEFORE the UI build: load_state constructs panel cards, which take the barrier.
        from .render_loop import RenderLoop
        self._render_loop = RenderLoop(self._on_render_batch, parent=self)

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
        # The window title (qt_fluent draws it at ``scaled_px(TITLE_LEFT_INSET)`` == ``scaled_px(WINDOW_PAD)``)
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

    def load_state(self, state: TaskConsoleState) -> None:
        self._building = True
        try:
            self.state = state
            for card in list(self.cards):
                self._close_panel_editor(card)   # drop any open Edit tab for this card
                card.shutdown()
                card.setParent(None)
                card.deleteLater()
            self.cards = []
            for row in list(self.logic_nodes):
                self._remove_logic_node(row, _rebuild=False)
            self.name_edit.setText(state.name)
            for config in state.panels:
                self._attach_card(self._new_panel_card(config))
            for node in state.logic:
                self._attach_logic_node(node)    # always STOPPED -- Start is manual
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
        self.running_nodes = list(running_nodes)
        self.load_state(state)

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
            render_barrier=self._render_loop.barrier, area_select_sink=self._on_panel_area_select,
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

        from Zou_lab_control.neutral_atom.operations.signal_expr import NAMESPACE_HELPERS
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

    def _signal_names(self) -> list[str]:
        """Every signal a picker may offer: PUBLISHED hub signals UNION the DECLARED outputs
        of stopped Logic-tab nodes (the latter not on the hub yet).  The ONE name source for
        every signal picker, so a not-yet-started node's output is selectable (#6)."""
        names = {str(n) for n in self.hub.names()}
        names.update(self._signal_providers().keys())
        return sorted(names)

    def _signal_formats(self) -> dict:
        """``name -> standardized array shape`` for every LIVE hub signal, read straight
        off the most recent published VALUE (``logic.describe_shape``) -- AUTO from real
        data, never a hand-typed name->format map that could drift from what a node
        actually emits.  Lets the signal picker show each signal's SHAPE, not just its
        name (e.g. ``occupied  [(35,)]``)."""
        from Zou_lab_control.neutral_atom.operations.logic import describe_shape
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

        The producing node is found among the ACTUAL running nodes first (the live-readout /
        notebook path wires a reactive OccupancyProcessor straight into ``running_nodes``
        without a Logic-tab row), reading the centres/underlay output names off the node
        itself (``sitemap_centers_key`` / ``sitemap_image_key``).  A configured-but-not-yet-
        started Logic-tab row is the fallback, resolved from its spec metadata."""
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
        shape read off a real value via ``logic.describe_shape`` (auto, never hand-typed)
        and each description from the node's ``output_specs`` (what the signal MEANS).  A
        measurement / processor publishes to the hub under its prefix; a TASK is OFF the
        hub, so it documents what it streams mid-run (its ``output`` buffer) + what it
        produces (its ``result`` keys), shapes filled in as the values appear."""
        from Zou_lab_control.neutral_atom.operations.logic import describe_shape
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
        # nodes would collide).  Show the SHORT natural name (strip the prefix) -- "rate", not
        # "judge_occupancy_rate" -- because the Logic row is already titled by the node.  NOTE
        # output_specs (and so ``desc``) is keyed by the FULL published name (occupancy's
        # ``p + "occupied"``, the camera's prefixed frame_i): look descriptions up by ``full``.
        pfx = str(getattr(node, "prefix", "") or "")
        for full in sorted(node.published_signals()):
            short = strip_node_prefix(full, pfx)             # ONE rule, shared with the picker nest
            try:
                # SAME schema-driven formatter as the task branch (#12): a hub signal and a task
                # output of the same logical shape render byte-identically -- no drift possible.
                shape = self._describe_from_schema(self.hub.latest(full), self.hub.schema(full))
            except Exception:
                shape = "—"
            rows.append((short, shape, desc(full)))
        return rows

    def _update_row_publishes(self, row: "LogicNodeRow") -> None:
        """Fill a Logic-tab row's "publishes:" legend (ONE signal per line: name, shape,
        meaning).  Shapes are AUTO-EXTRACTED from the real published VALUES
        (``logic.describe_shape``) and the meaning from the node's ``output_specs`` --
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
        rejected at the base (Processor.__init__) -- one guard per scope."""
        from ..neutral_atom.operations.logic import Processor
        if not isinstance(start_node, Processor) or not getattr(start_node, "consumes", ()):
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
            inputs = node.consumes if isinstance(node, Processor) else ()
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
        from Zou_lab_control.neutral_atom.operations.logic import contract_shape_label, describe_shape
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
        # A multidimensional TaskOutput point_shape is already the authoritative scan geometry.
        # Do not borrow an unrelated node ``grid_shape`` (CalibrateReadoutTask uses that name for
        # trap-site layout while its frame output correctly has point_shape=(1,)).
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
        # The per-panel ``<slug>_region`` signals are published by the CONSOLE (a panel's drawn
        # selection), not by a LogicNode, so they have no ``providers`` entry -- exempt them, else the GC
        # would purge a live region out from under its analysis node (a SignalHistoryGap).  They are
        # cleaned when their panel is removed (see _remove_panel).
        region_names = set(self._panel_region_name.values())
        orphans = [n for n in self.hub.names() if n not in providers and n not in region_names]
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
                    # not repeat its prefix: show "rate ← occupancy", not "judge_occupancy_rate ←
                    # occupancy" -- the ONE strip_node_prefix rule the nested combo uses.
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
            node.start()
        return node

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
        self._render_loop.barrier()
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

    def _remove_panel(self, card: PanelCard) -> None:
        # blocked while a task owns the console (the lock is released BEFORE
        # _clear_task_running drops the transient task panel, so this never blocks that).
        if self._task_locked:
            return
        self._render_loop.barrier()   # the worker must not be composing into the card we tear down
        if card in self.cards:
            self._remove_panel_analysis(card)  # tear down the analysis node this panel owned (#1)
            region = self._panel_region_name.pop(id(card), None)
            if region:
                self.hub.remove_signals([region])   # the panel's region signal is gone with the panel
            card.settings_popup.hide()
            self._close_panel_editor(card)     # drop this card's Edit tab too
            self.cards.remove(card)
            card.shutdown()
            card.setParent(None)
            card.deleteLater()
            self._arrange()
            self._recompute_tick_interval()    # removing the fastest panel can slow the base
            self._mark_dirty()

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
        (G1) -- so two same-kind nodes (two ``Judge occupancy``) are told apart in the Logic
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
            from Zou_lab_control.neutral_atom.operations.logic import camera_frame_keys
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
        # this, restarting the only occupancy node sees its own STOPPED 'occupied' still in the hub, takes
        # a fresh 'judge_occupancy_2_' prefix, and every panel bound to 'occupied' goes UNBOUND.  Its
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
        from Zou_lab_control.neutral_atom.operations.measurement import measurement_slug
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

    def _analysis_node_title(self, card: "PanelCard", verb: str) -> str:
        """A per-PANEL analysis-node title derived from the panel's OWN title (``"2D image #1 fit"``),
        made unique among the Logic rows.  The node prefix derives from this title
        (:meth:`_logic_node_prefix`), so two panels on the SAME source get DISTINCT nodes AND distinct
        published output names -- the root fix for the old one-node-per-source sharing (#3)."""
        base = f"{card.config.title or 'panel'} {verb}"
        return indexed_unique_name(base, {str(r.node.title) for r in self.logic_nodes})

    def _panel_analysis_row(self, card: "PanelCard") -> "LogicNodeRow | None":
        """This panel's ONE analysis row (a FitProcessor XOR a RoiProcessor), or ``None`` -- pruning a
        stale handle whose row was removed from the Logic tab by hand."""
        row = self._panel_analysis.get(id(card))
        if row is not None and row not in self.logic_nodes:
            self._panel_analysis.pop(id(card), None)
            return None
        return row

    def _remove_panel_analysis(self, card: "PanelCard") -> None:
        """Stop + remove the hub node THIS panel's analysis created (its FitProcessor or RoiProcessor),
        keyed by the PANEL.  The ONE teardown seam every analysis-off path funnels through -- clearing a
        fit, switching the Analysis action, turning the Selectors switch off, and panel removal -- so an
        analysis node is never left republishing after its panel stopped driving it (#1: symmetric for
        BOTH fit and ROI, unlike the old fit-only clear that left the ROI node consuming forever)."""
        row = self._panel_analysis.pop(id(card), None)
        if row is not None and row in self.logic_nodes:
            self._remove_logic_node(row)
        self._fit_overlay_pushed.pop(id(card), None)   # a fresh fit re-pushes from version -1
        if card.plotter is not None and hasattr(card.plotter, "apply_published_fit"):
            card.plotter.apply_published_fit(None)     # drop any live fit overlay too

    def _publish_region(self, card: "PanelCard", selection) -> str:
        """Publish THIS panel's drawn Selection as its ``<slug>_region`` hub signal (the drawn bounds as
        a tiny ``(K,2)`` tensor + the binding/frame/bins in the schema metadata), the ONE control input
        an analysis node consumes.  A re-drag republishes the SAME name (stable per panel), so retarget
        is a republish -- there is no ``set_selection`` plumbing.  Returns the region signal name."""
        from ..neutral_atom.core.selection import encode_region
        from ..neutral_atom.core.signals import SignalSchema, SignalTensor
        from ..neutral_atom.operations.measurement import measurement_slug
        name = self._panel_region_name.get(id(card))
        if name is None:
            slug = measurement_slug(card.config.title) or "panel"
            existing = set(self.hub.registered_names())
            name = f"{slug}_region"
            k = 2
            while name in existing:
                name = f"{slug}_{k}_region"
                k += 1
            self._panel_region_name[id(card)] = name
        bins = int(card.config.params.get("bins", 50)) if card.config.kind == "hist" else None
        values, metadata = encode_region(selection, bins=bins)
        rows = int(values.shape[0])
        schema = SignalSchema(point_shape=(1,), data_shape=(rows, 2), dtype=np.float64,
                              repeat_capacity=1, label="region", metadata=metadata)
        try:
            replace = not self.hub.schema(name).same_definition(schema)
        except KeyError:
            replace = False
        self.hub.register_signal(name, schema, replace=replace)
        self.hub.publish({name: SignalTensor(values.reshape(1, 1, rows, 2), schema)})
        return name

    def _published_fit_result(self, node):
        """Build a :class:`FitResult` from a FitProcessor's PUBLISHED parameters on the hub (its first
        cell) so the panel overlay DRAWS from solved params with no Qt-thread solve (#6).  ``None`` when
        the node has no published result yet."""
        from ..neutral_atom.core.fitting import FitResult, fit_model
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
        PUBLISHED result to its plotter's DISPLAY-only overlay.  This is how a fit reaches the plot now --
        the overlay reconstructs the curve/dot from published parameters, never solving on the Qt thread
        (#6).  Gated on the node's published version so an unchanged fit costs nothing."""
        from ..neutral_atom.operations.processors.fit import FitProcessor
        for card in self.cards:
            plotter = card.plotter
            if plotter is None or not hasattr(plotter, "apply_published_fit"):
                continue
            row = self._panel_analysis.get(id(card))
            node = self._logic_nodes.get(id(row)) if row is not None else None
            if not isinstance(node, FitProcessor):
                continue
            version = self.hub.signal_versions().get(node.prefix + "fit_valid", -1)
            if self._fit_overlay_pushed.get(id(card)) == version:
                continue
            self._fit_overlay_pushed[id(card)] = version
            plotter.apply_published_fit(self._published_fit_result(node))

    def _sync_fit_node(self, card: "PanelCard", request) -> None:
        """The console's ONE fit-node sink (wired to :attr:`PanelCard.fit_node_sink`): create OR retarget
        a per-panel FitProcessor that fits on its WORKER and publishes its parameters as signals
        (``fit_x0``/``fit_sigma``/... -- consumable by a Monitor or a scan loss, and read back by the
        panel's DISPLAY-only overlay).  EVERY fit family becomes a node -- a 2-D image centre, a 1-D
        peak, a hist gaussian -- so no fit ever solves on the Qt thread (#6).  The node is owned by THIS
        PANEL (``_panel_analysis[id(card)]``), never keyed by the source signal (#3); its region rides on
        the panel's ``<slug>_region`` signal (the drawn selection), republished here.  A grid focus fits
        per-cell in place (no hub node).  The teardown counterpart is :meth:`_remove_panel_analysis`."""
        from ..neutral_atom.core.fitting import FitRequest
        from dataclasses import replace as _dc_replace
        from ..neutral_atom.core.selection import Selection
        req = FitRequest.from_dict(request) if isinstance(request, Mapping) else request
        if card.config.kind == "grid":
            card._set_fit_result_text()      # a facet grid fits per-cell in place, not as a hub node
            self._remove_panel_analysis(card)
            return
        signal = str(card.config.inputs[0]) if card.config.inputs else ""
        if not signal or self._task_locked:
            return
        # The drawn selection travels as the region signal (annotated with this panel kind's binding +
        # bins); the CONFIG request carries only the model + fixed/initial (its selection is stripped --
        # the region is the single source of the selection).
        bound = region_binding(
            card.config.kind, req.selection,
            structure=card._bound_structure(),
            coordinates=card._selection_coordinates_for_binding(),
            origin=req.selection.metadata.get("origin", (0.0, 0.0)))
        region_name = self._publish_region(card, bound)
        payload = _dc_replace(req, selection=Selection()).to_dict()
        from ..neutral_atom.operations.processors.fit import FIT_SPEC_NAME, FitProcessor
        row = self._panel_analysis_row(card)
        node = self._logic_nodes.get(id(row)) if row is not None else None
        if isinstance(node, FitProcessor):
            node.set_fit_request(payload)                      # retarget model/fixed; region republished
            row.node.values = {**dict(row.node.values or {}), "fit_request": payload}
            editor = self._logic_editors.get(id(row))
            if editor is not None and hasattr(getattr(editor, "form", None), "seed_values"):
                editor.form.seed_values({"fit_request": payload})
            self._mark_dirty()
            card.set_status(f"fit -> {row.node.title}", error=False)
            return
        self._remove_panel_analysis(card)                     # was a ROI node (fit <-> roi are exclusive)
        from ..neutral_atom.operations.signal_expr import DEFAULT_SOURCE
        cfg = LogicNodeConfig(
            kind="processor", name=FIT_SPEC_NAME,
            title=self._analysis_node_title(card, "fit"),
            values={"source": {"inputs": [signal], "source": DEFAULT_SOURCE},
                    "fit_request": payload, "region": region_name})
        new_row = self._add_logic_node(cfg, focus=False)
        self._panel_analysis[id(card)] = new_row
        self._start_logic_node(new_row)
        started = self._logic_nodes.get(id(new_row)) is not None
        card.set_status(f"fit -> {cfg.title}" + ("" if started else " (start failed — see Logic tab)"),
                        error=not started)

    def _on_panel_selection_clear(self, card: "PanelCard", action: str) -> None:
        """The ONE selection-teardown seam, symmetric for BOTH analyses (#1): leaving/clearing a panel's
        analysis STOPS + REMOVES the hub node it created -- a FitProcessor (``fit_x0``/... stop
        publishing) or a RoiProcessor (``roi_frame``/``roi_value`` stop updating).  Because the panel owns
        AT MOST ONE analysis node keyed by the CARD (``_panel_analysis``), the removal is unambiguous and
        symmetric: ``action`` (``"fit"`` / ``"roi"`` / ``"selectors"``) is informational only.  This is
        the root fix for the old fit-only teardown that left the ROI node consuming forever, AND for the
        old source-signal keying that removed EVERY panel's node on one source instead of just this one."""
        self._remove_panel_analysis(card)

    def _apply_roi_selection(self, card: "PanelCard", selection) -> None:
        """Turn a drag on ANY plot kind into a Selection-driven RoiProcessor owned by THIS panel.

        The region is bound to the consumed block's own axes through the ONE per-kind resolver
        (:func:`live.region_binding`, the inverse of ``coerce_panel_value``): an image rectangle, a 1-D
        x-range, a distribution count-range and a site-centre rectangle all become a serializable
        Selection whose ``metadata['binding']`` says which axes it spans.  That Selection -- NEVER pixel
        endpoints -- travels to a RoiProcessor CREATED for or RETARGETED on this panel
        (``_panel_analysis[id(card)]``), so two panels on the same source own DISTINCT ROI nodes with
        distinct output names (#3) and the sealed seam holds (``neutral_atom`` never imports frontend)."""
        signal = str(card.config.inputs[0]) if card.config.inputs else ""
        if not signal or self._task_locked:
            return
        try:
            self.hub.schema(signal)
        except KeyError:
            card.set_status("ROI source has no registered signal schema", error=True)
            return
        bound = region_binding(
            card.config.kind, selection,
            structure=card._bound_structure(),
            coordinates=card._selection_coordinates_for_binding(),
            origin=selection.metadata.get("origin", (0.0, 0.0)))
        region_name = self._publish_region(card, bound)             # the drawn region AS a per-panel signal
        from ..neutral_atom.operations.processors.roi import ROI_SPEC_NAME, RoiProcessor
        row = self._panel_analysis_row(card)
        node = self._logic_nodes.get(id(row)) if row is not None else None
        if isinstance(node, RoiProcessor):
            # retarget = the republished region above; the reactive node re-computes on its worker.
            self._mark_dirty()
            card.set_status(f"ROI -> {row.node.title}", error=False)
            return
        self._remove_panel_analysis(card)                               # was a fit node (fit <-> roi exclusive)
        from ..neutral_atom.operations.signal_expr import DEFAULT_SOURCE
        cfg = LogicNodeConfig(
            kind="processor", name=ROI_SPEC_NAME,
            title=self._analysis_node_title(card, "ROI"),
            values={"source": {"inputs": [signal], "source": DEFAULT_SOURCE},
                    "region": region_name})
        new_row = self._add_logic_node(cfg, focus=False)   # stay on the Monitor board -- no tab jump
        self._panel_analysis[id(card)] = new_row
        self._start_logic_node(new_row)
        started = self._logic_nodes.get(id(new_row)) is not None
        card.set_status(f"ROI -> {cfg.title}" + ("" if started else " (start failed -- see Logic tab)"),
                        error=not started)

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
        self._stop_logic_node(row, _silent=True)
        claimed = {id(d) for d in getattr(node, "occupied_devices", lambda: ())()}
        if claimed:
            for other in list(self.running_nodes):
                if other is node:
                    continue
                theirs = {id(d) for d in getattr(other, "occupied_devices", lambda: ())()}
                if not (claimed & theirs):
                    continue                             # disjoint hardware -> coexist
                other_row = next((r for r in self.logic_nodes
                                  if self._logic_nodes.get(id(r)) is other), None)
                if other_row is not None:
                    self._stop_logic_node(other_row)     # full UI path: dot / editor / registry
                else:
                    # a row-less injected node: stop it directly and drop it from the registry
                    try:
                        other.stop()
                    except Exception:
                        pass
                    try:
                        self.running_nodes.remove(other)
                    except ValueError:
                        pass
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
            node.start()
        except Exception as exc:
            row.set_state("error", status=f"start failed: {str(exc).splitlines()[0][:80]}")
            if editor is not None:
                editor.set_running(False)
                editor.set_status(f"start failed: {str(exc).splitlines()[0][:140]}", error=True)
            return
        # COMMIT: the node is genuinely running -- only now does it enter the registries, and only
        # now are the previous build's orphan signals unlinked (a failed start leaves the old
        # signals in the hub untouched, exactly like a plain Stop, so nothing is lost).
        self._logic_nodes[id(row)] = node
        self._last_node[id(row)] = node           # survives Stop, for signal-source labelling
        if node not in self.running_nodes:
            self.running_nodes.append(node)
        # #5: unlink any signal the PREVIOUS build published that this new build no longer does AND no
        # other running node owns -> a switched/rebuilt node leaves NO orphan "(unbound)" signal behind.
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
        orphan = old_sigs - keep
        if orphan:
            self.hub.remove_signals(orphan)
        row.set_state("running", status="running")
        self._update_row_publishes(row)            # now show the LIVE node's published shapes
        if editor is not None:
            editor.set_running(True)
            editor.set_status("running", error=False)
        self.status_dot.set_color(GREEN)
        self._mark_dirty()
        # A TASK (one-shot orchestration) TAKES OVER the console (confocal-style): show
        # its mid-run output in a dedicated Monitor panel + LOCK every other action.
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

    def _clear_task_running(self) -> None:
        """Leave task-run mode (task finished OR stopped): drop the lock + banner and REMOVE the
        transient mid-run panel the task owned.  Finish and Stop take the SAME path (#C) -- a task's
        plot panel is transient (it only showed the work in progress), so it is auto-removed when the
        task ends; the operator's own panels are never touched (only ``_task_card`` is removed)."""
        self._running_task_row = None
        card, self._task_card = self._task_card, None
        self._apply_task_lock(False)                          # unlock BEFORE remove (remove no-ops while locked)
        if card is not None and card in self.cards:
            self._remove_panel(card)
        self._task_card_tensor = None
        self._task_output_node = None

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
        if not self._render_loop.busy:
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
          * processor   -> spec.make_node(...) (reactive) or a ProcessorRun (one-shot)
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
            from Zou_lab_control.neutral_atom.devices.base import ReadOnlyDevice
            from Zou_lab_control.neutral_atom.operations.logic import ProcessorRun

            def _borrow(role: str):
                # A ProcessorRun DRIVES its declared roles (EXCLUSIVE), so only a DRIVABLE
                # instance qualifies: an OBSERVE record on a running node (e.g. a camera
                # measurement's read-only sequencer proxy) would PermissionError on the
                # first prepare/fire -- skip those, never unwrap them.
                for n in self.running_nodes:
                    dev = getattr(n, role, None)
                    if dev is not None and not isinstance(dev, ReadOnlyDevice):
                        return dev
                return None

            roles = {str(r) for r in (getattr(spec, "devices", ()) or ())}
            readout = getattr(self.session, "readout", None)
            return ProcessorRun(self.hub, spec, readout=readout,
                                 camera=_borrow("camera") if "camera" in roles else None,
                                 sequencer=_borrow("sequencer") if "sequencer" in roles else None,
                                 params=values)
        if kind == "task":
            return spec.build(self.hub, **values)
        raise RuntimeError(f"unknown logic kind {kind!r}")

    def _stop_logic_node(self, row: "LogicNodeRow", *, _silent: bool = False) -> None:
        """Stop a logic node's node (``node.stop()``) and grey its dot."""
        node = self._logic_nodes.get(id(row))
        if node is not None:
            try:
                node.stop()
            except Exception:
                pass
            if node in self.running_nodes:
                self.running_nodes.remove(node)
        self._logic_nodes[id(row)] = None
        editor = self._logic_editors.get(id(row))
        if not _silent:
            row.set_state("stopped", status="stopped")
            self._update_row_publishes(row)        # back to the spec-declared outputs
            if editor is not None:
                editor.set_running(False)
                editor.set_status("stopped", error=False)
        # Leaving task-run mode when the running task is stopped (releases the lockout).
        if row is self._running_task_row:
            self._clear_task_running()

    def _remove_logic_node(self, row: "LogicNodeRow", *, _rebuild: bool = True) -> None:
        """Stop + remove a logic node (its node is stopped, its row + Edit drop)."""
        # a running task locks the console: the per-row Remove button must no-op too
        # (every other mutating entry guards on this).  Internal teardown (load_state)
        # passes _rebuild=False and is never reached while locked.
        if self._task_locked and _rebuild:
            return
        # Capture this node's published signals so REMOVE can PURGE them from the hub -- a removed
        # node's signals are stale and must leave, else they pile up run-after-run as "多余 signal" in
        # every picker (#2).  STOPPING keeps them (a finished scan stays plottable / a panel can be
        # wired before the next run); only REMOVING purges.  Use ``_last_node`` (the last built node,
        # retained THROUGH stop) not ``_logic_nodes`` (None'd on stop): the common flow is STOP-then-
        # REMOVE, where the live ref is already gone but the lingering hub signals are precisely the
        # ones to purge.
        gone = self._logic_nodes.get(id(row)) or self._last_node.get(id(row))
        gone_sigs: set[str] = set()
        if gone is not None and hasattr(gone, "published_signals"):
            try:
                gone_sigs = {str(s) for s in gone.published_signals()}
            except Exception:
                gone_sigs = set()
        self._stop_logic_node(row, _silent=True)
        editor = self._logic_editors.pop(id(row), None)
        if editor is not None:
            idx = self.tabs.indexOf(editor)
            if idx >= 0:
                self.tabs.removeTab(idx)
            editor.teardown()
            editor.setParent(None)
            editor.deleteLater()
        self._logic_nodes.pop(id(row), None)
        self._last_node.pop(id(row), None)
        # Drop any per-panel analysis handle pointing at this row -- so a panel whose ROI/fit row is
        # removed by hand (or by _remove_panel_analysis) re-creates a fresh node on the next drag
        # instead of retargeting a dead handle (#3 ownership stays exact).
        for card_id in [cid for cid, r in self._panel_analysis.items() if r is row]:
            self._panel_analysis.pop(card_id, None)
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

    def _poll_logic_nodes(self) -> None:
        """Each tick: reflect each running node's state on its row + Edit (a
        one-shot that finished -> stopped; a node that errored -> red)."""
        for row in self.logic_nodes:
            node = self._logic_nodes.get(id(row))
            if node is None:
                continue
            editor = self._logic_editors.get(id(row))
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
        from ..neutral_atom.core.signals import NO_LINEAGE     # single source of the sentinel (lazy: off na import graph)
        live: set[str] = set()
        for node in self.running_nodes:
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
        from Zou_lab_control.neutral_atom.operations.signal_expr import hub_namespace
        from Zou_lab_control.neutral_atom.core.signal_tensor import SignalHistoryGap
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
        the axis origin, so this is robust to endpoints vs any position+size form."""
        frames: dict[str, list] = {}
        for node in self.running_nodes:
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
                frames[str(s)] = list(region)
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
        self._render_loop.barrier()
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
        from ..neutral_atom.core.signals import NO_LINEAGE     # single source of the sentinel (lazy)
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
        busy = self._render_loop.busy
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
            self._render_loop.submit(
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
        self._render_loop.barrier()
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
    def stop_all_nodes(self) -> None:
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
        for row in list(self.logic_nodes):
            if self._logic_nodes.get(id(row)) is not None:
                self._stop_logic_node(row)
        # Row-less injected nodes (show_task_console(running_nodes=[...])) have no row to
        # repaint but must leave the shot clock's live set all the same.
        for node in list(self.running_nodes):
            try:
                node.stop()
            except Exception:
                pass
            if node in self.running_nodes:
                self.running_nodes.remove(node)

    def stop_nodes_using(self, affected_ids) -> None:
        """Stop exactly the running nodes that reference one of the ``affected_ids`` devices --
        the session's fine-grained device-change hook (``load_config`` swapping specific device
        INSTANCES).  A node is affected iff its :meth:`~LogicNode.referenced_devices` (EXCLUSIVE
        drivers AND OBSERVE records, unwrapped to real identity) intersect the swapped set, so
        reinitialising the camera stops every camera view / occupancy path riding it while a scan
        on the untouched sequencer keeps running.  Goes through the SAME ``_stop_logic_node``
        endpoint as every other stop (no zombie left to freeze the shot clock, #close-reopen)."""
        affected = {int(i) for i in affected_ids}
        if not affected:
            return

        def _touches(node) -> bool:
            try:
                return any(id(d) in affected for d in node.referenced_devices())
            except Exception:
                return False

        for row in list(self.logic_nodes):
            node = self._logic_nodes.get(id(row))
            if node is not None and _touches(node):
                self._stop_logic_node(row)
        for node in list(self.running_nodes):        # row-less injected nodes
            if _touches(node):
                try:
                    node.stop()
                except Exception:
                    pass
                if node in self.running_nodes:
                    self.running_nodes.remove(node)

    def shutdown(self) -> None:
        """Stop the refresh timer and every running node's owner thread, then release
        the editors/cards.  IDEMPOTENT -- it is reached from both the window close
        (``show_task_console`` wires ``window.hidden`` here) and an explicit
        ``with show_task_console(...)`` / re-run, which can both fire.

        Stopping a node sets its cooperative-cancel event (M5), so a node thread
        blocked in ``camera.acquire`` unwinds and the camera is released -- this is
        what keeps a closed dashboard from leaving a live acquire thread (and a held
        camera / RPyC connection) behind, which previously wedged the kernel."""
        if getattr(self, "_shut", False):
            return
        self._shut = True
        self._timer.stop()
        for node in list(self.running_nodes):
            try:
                node.stop()
            except Exception:
                pass
        for editor in list(self._panel_editors.values()):
            editor.teardown()
        self._panel_editors.clear()
        for editor in list(self._logic_editors.values()):
            editor.teardown()
        self._logic_editors.clear()
        # Stop the render worker BEFORE tearing panels down: any in-flight batch finishes against
        # still-alive figures, and no further batch can start touching a torn-down card.
        self._render_loop.stop()
        for card in self.cards:
            card.shutdown()
        on_close = getattr(self, "_on_close", None)
        if on_close is not None:
            try:
                on_close()
            except Exception:
                pass

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
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
                          processors=processors, tasks=tasks, session=session, scale=scale,
                          window_ratio=window_ratio)
    console._on_close = on_close
    # A passed-in node should stream the moment the window opens -- so the Monitor
    # is live without the caller having to remember node.start().  start() is
    # idempotent, so a node the caller already started (e.g. bring-up's
    # readout.start()) keeps its own loop; only NON-running nodes are launched
    # here.  (TaskConsole.__init__ deliberately does NOT do this, so
    # tests/notebooks keep deterministic manual stepping.)
    for node in running_nodes:
        if not getattr(node, "running", False) and hasattr(node, "start"):
            node.start()
    # Closing the window must stop the node owner threads (else they keep running, blocked in
    # camera.acquire holding the camera / RPyC link, wedging the kernel).  The console is a CHILD
    # of the window so its own closeEvent never fires on a window close -- we wire the window's
    # signals instead.  Minimising NEVER stops the nodes (only the X / a genuine close does).
    def _wire_close(window) -> None:
        if hide_on_close:
            # Session-bound (notebook) console: the X HIDES the window (keeps the panel layout so a
            # later exp.task_console() restores the SAME interface) and stops every running node so
            # the devices are released.  close_requested fires on the X, not on minimize.
            window.close_requested.connect(console.stop_all_nodes)
        else:
            # Standalone window (.bat / explicit): the X fully tears the console down.
            window.closed.connect(console.shutdown)
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
