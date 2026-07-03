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
    panel_display_size,
    panel_plot,
    panel_size_cells,
    site_ring_radius,
)
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
    FluentSwitch,
    FluentTabWidget,
    FluentFloatingEditor,
    FluentWindow,
    center_window_on_primary_screen,
    ensure_qt_app,
    fluent_tab_shadow_margin,
    fluent_text_width,
    fluent_widget_stylesheet,
    scaled_px,
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

# ParamDecl is the ONE declarative param record both the measurement form and the plot
# panels use (PANEL_PARAMS).  Importing it here (frontend -> operations is allowed; the
# reverse is not) lets the panel params be real ParamDecls validated by the kind whitelist,
# instead of a parallel ParamSpec class with its own smaller ladder.
from Zou_lab_control.neutral_atom.operations.measurement import ParamDecl


TASK_FILES_ENV = "ZLC_TASK_DIR"

# Logic-node kinds that DRIVE THE DEVICE (camera + sequencer): a camera live stream, a
# scanned measurement, or a one-shot task.  Starting any one of them first stops every
# OTHER running device-driver, so two never fight over the shared camera / pulse streamer
# (which deadlocks real hardware).  A reactive PROCESSOR (e.g. judge-occupancy) only reads
# hub signals -- it touches no device, so it is NOT in this set and keeps running.
DEVICE_DRIVING_KINDS: frozenset = frozenset({"camera", "measurement", "task"})

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


def _common_token_prefix(names) -> str:
    """The longest common UNDERSCORE-token prefix of ``names`` -- e.g. both
    ``judge_occupancy_rate`` and ``judge_occupancy_occupied`` share ``judge_occupancy_``.
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


def strip_node_prefix(full: str, prefix: str) -> str:
    """The SHORT signal name = the hub name minus its producing node's disambiguating prefix
    (``judge_occupancy_rate`` -> ``rate``, ``temperature_survival`` -> ``survival``, ``frame`` ->
    ``frame``).  The ONE rule the Logic tab AND the signal picker share, so the nest leaf is ALWAYS the
    short name -- never the full prefixed key, never the verbose axis label."""
    full = str(full)
    prefix = str(prefix or "")
    return full[len(prefix):] if (prefix and full.startswith(prefix) and len(full) > len(prefix)) else full


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
    # A configured input may NAME a signal that is declared but not published yet -- a node's own future
    # output (a pulse-scan reading its own ``frame_0``), or a not-yet-started producer's signal.  The
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


def _scan_modes() -> tuple[str, str, str]:
    """The ("none", "api", "scan") tuple -- fetched lazily from the ONE backend source
    (``operations.measurements.pulse_scan.SCAN_MODES``) so the name strings are typed once and the
    frontend module import stays off neutral_atom's import-time graph (every other na use is lazy)."""
    from ..neutral_atom.operations.measurements.pulse_scan import SCAN_MODES
    return SCAN_MODES


def _is_number(v) -> bool:
    """True when ``v`` can be read as a finite float (a saved numeric param), else False."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class _PulseSlotsWidget(QtWidgets.QWidget):
    """Auto-built sub-form for a pulse template's API + scan slots.

    One numeric row per API slot (``a1``, ``a2``, ...): the FIXED operator-set value, in the slot's
    own unit, seeded from the template's current value (these are ALWAYS shown and ALWAYS applied).

    Then ONE ``Scan:`` mode toggle -- ``[None | API | Scan]`` -- and ONE shared scan-table editor
    whose columns / legend / templates ADAPT to the selected mode:

    * **Scan** sweeps the bound HARDWARE scan slots (``s0``, ``s1``, ...) -- the columns are the
      hardware slots in ns ticks / LSB, snapped to the FPGA grid by the build.
    * **API** sweeps the API slots in SOFTWARE (``a1``, ``a2``, ...) -- the columns are the api
      slots in their native units (no snap); the ``Extra settle delay`` row is shown only here.
    * **None** fires a single fixed point at the api values -- the table is hidden.

    Each program is the SAME table model as the pulse GUI Scan tab (``column_stack`` / ``grid``
    templates): one ROW per point, one COLUMN per slot, advanced in LOCKSTEP.  A mode whose kind is
    ABSENT (no api slots -> API; no scan slots -> Scan) is disabled.  TWO remembered buffers (one
    per mode) so switching API<->Scan keeps each mode's last program -- the column MEANING differs,
    so they are never collapsed into one string.

    Output schema: ``{"api": {a1: float, ...}, "scan_mode": "none"|"api"|"scan",
    "scan_code": "<the active mode's program>", "extra_delay": float}`` (ONE ``scan_code`` = the
    active table; the build dispatches on ``scan_mode``)."""

    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._api_widgets: dict[str, QtWidgets.QWidget] = {}
        self._api_remembered: dict[str, str] = {}     # typed api values, kept across reloads
        # TWO remembered buffers -- one per sweep mode -- because the columns mean DIFFERENT things
        # (api native units vs hardware ns/LSB); switching API<->Scan must restore each mode's own
        # program.  Keyed by the SINGLE-source mode names (api, scan); None has no table.
        _none, _api, _scan = _scan_modes()
        self._scan_buffers: dict[str, str] = {_api: "", _scan: ""}
        self._extra_remembered: str = ""              # typed extra settle delay, kept across reloads
        self._scan_code = None                        # the ONE active-mode FluentCodeEdit (None in None mode)
        self._extra_delay = None                      # the extra-settle FluentLineEdit (API mode only)
        self._mode_combo = None                       # the Scan:[None|API|Scan] toggle
        self._mode = "none"                           # the currently selected mode
        # A pending SAVED state to restore on the next rebuild (a load / round-trip).  Kept apart
        # from the live-typed _api_remembered so rebuild's own stash-of-current-text never clobbers
        # it; consumed (cleared) once applied.
        self._pending_mode = None
        self._pending_api: dict[str, str] = {}
        self._api_columns: list[tuple[str, str, str]] = []   # the api-mode table columns (legend)
        self._scan_columns: list[tuple[str, str, str]] = []  # the scan-mode table columns (legend)
        self._api_specs: list = []                           # per-kind template column specs (api)
        self._scan_specs: list = []                          # per-kind template column specs (scan)
        self._have_api = False
        self._have_scan = False
        self._n_slots = 0                             # bound hardware scan slot count
        # Spans the FULL form width.  An always-on "API slots" header + the per-slot value rows live
        # in _api_box; the mode toggle + the single adaptive table live in _table_box (rebuilt on a
        # mode switch WITHOUT tearing down the api rows, so the value edits survive a toggle).
        self._box = QtWidgets.QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(scaled_px(6, minimum=4))
        self._api_box = QtWidgets.QVBoxLayout()
        self._api_box.setContentsMargins(0, 0, 0, 0)
        self._api_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._api_box)
        self._table_box = QtWidgets.QVBoxLayout()
        self._table_box.setContentsMargins(0, 0, 0, 0)
        self._table_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._table_box)
        # The whole-sweep count is the measurement's auto-injected ``Repeat (0 = ∞)`` knob -- a
        # pulse-scan pass IS a whole sweep, so the ONE repeat axis already counts sweeps (0 = sweep
        # forever).  There is NO separate "scan repeats" field here (it would double the same number).

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

    def rebuild(self, api_rows, scan_rows) -> None:
        """``api_rows`` = ``[(name, kind, target, unit, current_value), ...]``.
        ``scan_rows`` = ``[(name, kind, target, unit, label), ...]`` (the bound scan slots = the
        columns of the scan table)."""
        # Remember whatever the operator currently typed (api values + the active mode's program),
        # then rebuild both the api rows and the toggle + table.  A queued SAVED load (seed_value set
        # _pending_*) must NOT be clobbered by stashing the current (default) editor text, so skip the
        # stash in that case -- the pending buffers / values are the source of truth for this rebuild.
        pending_load = bool(self._pending_mode) or bool(self._pending_api)
        if not pending_load:
            for name, w in list(self._api_widgets.items()):
                self._api_remembered[name] = w.text().strip()
            self._stash_active_buffer()
            if self._extra_delay is not None:
                self._extra_remembered = self._extra_delay.text().strip()
        self._drop_layout(self._api_box)
        self._drop_layout(self._table_box)
        self._api_widgets = {}
        self._scan_code = None
        self._extra_delay = None
        self._mode_combo = None
        self._n_slots = len(scan_rows)
        self._have_api = bool(api_rows)
        self._have_scan = bool(scan_rows)

        # ---- API slots section: ALWAYS shown.  One FIXED value row per api slot (used in every
        # mode; in API mode a row may ALSO be the swept dimension, with this value as the seed).
        self._api_box.addWidget(FluentSectionLabel("API slots"))
        if api_rows:
            api_labels = [f"{name}  {slot_label(kind, target)}" for name, kind, target, _u, _c in api_rows]
            api_lw = setting_label_width(api_labels, minimum=72)   # same row rule as every form
            for name, kind, target, unit, current in api_rows:
                label = f"{name}  {slot_label(kind, target)}"
                # a saved value (a load) wins over the live-typed buffer wins over the template seed
                seed_text = (self._pending_api.get(name) or self._api_remembered.get(name)
                             or f"{float(current):g}")
                edit = FluentLineEdit(seed_text, self)
                edit.setMinimumWidth(scaled_px(120, minimum=96))
                edit.setPlaceholderText(f"{unit}")
                edit.setToolTip(f"Fixed value for API slot {name!r} (unit: {unit}) -- used in every "
                                "mode (in API mode it is the resting seed unless that slot is swept).")
                edit.textChanged.connect(self.changed)
                self._api_box.addWidget(FluentSettingRow(label, edit, label_width=api_lw))
                self._api_widgets[name] = edit
        else:
            api_note = FluentLabel("(no API slot -- in the pulse GUI Edit tab, click a duration / DAC "
                                   "cell to its API (purple) state to fix or sweep a value by name aN.)", self)
            api_note.setWordWrap(True)
            api_note.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self._api_box.addWidget(api_note)

        # the per-mode table columns (api native units; hardware scan ns/LSB) + the per-kind template
        # column specs (so a DAC column is seeded with its signed code range, not a duration's ns range).
        from ..neutral_atom.timing import scan_column_spec
        self._api_columns = [(name, slot_label(kind, target), (unit or "value"))
                             for name, kind, target, unit, _c in api_rows]
        self._api_specs = [scan_column_spec(name, ("dac" if kind == "dac" else "duration"), unit=(unit or "ns"))
                           for name, kind, target, unit, _c in api_rows]
        self._scan_columns = []
        self._scan_specs = []
        for i, (name, kind, target, unit, stored_label) in enumerate(scan_rows):
            disp = stored_label or slot_label(kind, target)
            u = "ns ticks" if kind == "duration" else ("integer code (LSB)" if kind == "dac" else (unit or ""))
            self._scan_columns.append((f"s{i}", disp, u))
            self._scan_specs.append(scan_column_spec(f"s{i}", kind, unit=(unit or "ns")))

        # Default mode: Scan if scan slots exist, else API if api slots exist, else None.  A SAVED
        # mode (set by seed_value on a load) wins -- but only if that kind is still available for the
        # loaded template (a saved "scan" with no scan slots falls back to the default).
        none, api, scan = _scan_modes()
        default_mode = scan if self._have_scan else (api if self._have_api else none)
        available = {none: True, api: self._have_api, scan: self._have_scan}
        self._mode = (self._pending_mode if self._pending_mode and available.get(self._pending_mode)
                      else default_mode)
        self._pending_mode = None
        self._pending_api = {}                         # consumed: a later template-driven rebuild is fresh
        self._build_mode_toggle()
        self._render_active_table()
        self.changed.emit()

    def _build_mode_toggle(self) -> None:
        """The ONE ``Scan:`` mode toggle -- a FluentComboBox of [None | API | Scan].  A mode whose
        kind is absent is added but disabled (so the toggle is the single place that picks the swept
        kind, and an impossible mode reads as unavailable rather than silently missing)."""
        none, api, scan = _scan_modes()
        labels = {none: "None", api: "API", scan: "Scan"}
        enabled = {none: True, api: self._have_api, scan: self._have_scan}
        combo = FluentComboBox()
        combo.setMinimumWidth(scaled_px(120, minimum=96))
        for mode in (none, api, scan):
            combo.addItem(labels[mode], mode)
            item = combo.model().item(combo.count() - 1)
            if item is not None and not enabled[mode]:
                item.setEnabled(False)
        idx = combo.findData(self._mode)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.currentIndexChanged.connect(self._on_mode_changed)
        combo.setToolTip("What to sweep: None = one fixed point; API = sweep the api slots in "
                         "software; Scan = sweep the bound hardware scan slots.  A mode is disabled "
                         "when the template has no slot of that kind.")
        self._mode_combo = combo
        self._table_box.addWidget(FluentSettingRow("Scan", combo,
                                                   label_width=setting_label_width(["Scan"], minimum=72)))

    def _on_mode_changed(self, *_a) -> None:
        if self._mode_combo is None:
            return
        new_mode = str(self._mode_combo.currentData() or "none")
        if new_mode == self._mode:
            return
        self._stash_active_buffer()                   # keep the program the operator just typed
        self._mode = new_mode
        self._render_active_table()
        self.changed.emit()

    def _stash_active_buffer(self) -> None:
        """Save the active code editor's text into its mode's buffer (so a switch preserves it)."""
        if self._scan_code is not None and self._mode in self._scan_buffers:
            self._scan_buffers[self._mode] = self._scan_code.toPlainText()
        if self._extra_delay is not None:
            self._extra_remembered = self._extra_delay.text().strip()

    def _render_active_table(self) -> None:
        """(Re)build the single adaptive table area below the toggle for the current mode: nothing
        for None; the api columns + extra-settle row for API; the hardware scan columns for Scan."""
        # Drop everything in the table box EXCEPT the first item (the mode toggle row).
        while self._table_box.count() > 1:
            item = self._table_box.takeAt(self._table_box.count() - 1)
            w = item.widget()
            if w is not None:
                w.setParent(None); w.deleteLater()
            child = item.layout()
            if child is not None:
                self._drop_layout(child)
        self._scan_code = None
        self._extra_delay = None
        none, api, scan = _scan_modes()
        if self._mode == none:
            note = FluentLabel("(None -- no sweep: the pulse fires ONCE at the fixed api values above.)",
                               self)
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self._table_box.addWidget(note)
            return
        columns = self._api_columns if self._mode == api else self._scan_columns
        self._render_scan_table_body(columns, mode=self._mode)
        if self._mode == api:
            # The inter-point software settle is meaningful only for the API sweep (the device
            # streams hardware scans itself), so its row appears ONLY here.
            self._extra_delay = FluentLineEdit(self._extra_remembered, self)
            self._extra_delay.setMinimumWidth(scaled_px(120, minimum=96))
            self._extra_delay.setPlaceholderText("0")
            self._extra_delay.setToolTip("Extra settle delay (seconds) the device holds AFTER each "
                                         "point's pulse finishes, before the next is loaded (0 = none).")
            self._extra_delay.textChanged.connect(self.changed)
            self._table_box.addWidget(FluentSettingRow("Extra settle delay (s)", self._extra_delay,
                                                       label_width=setting_label_width(["Extra settle delay (s)"])))

    def _render_scan_table_body(self, columns, *, mode: str) -> None:
        """Render the shared scan-table form for the active ``mode``: a per-column legend +
        column_stack / grid template buttons + the ``FluentCodeEdit`` that assigns an (N x n_cols)
        ``scan_table``.  ONE renderer, so the API and Scan tables are byte-identical in FORM; only
        the columns / legend / remembered buffer differ.  ``columns`` = ``[(col_var, display, unit),
        ...]``.  The editor seeds from this mode's remembered buffer, or the column_stack template."""
        from ..neutral_atom.timing import scan_table_template
        _none, api_mode, _scan = _scan_modes()
        specs = self._api_specs if mode == api_mode else self._scan_specs
        legend = ["Columns of scan_table (one row = one point, columns advance in lockstep):"]
        for var, disp, unit in columns:
            legend.append(f"  {var}: {disp}  [{unit}]")
        legend_label = FluentLabel("\n".join(legend), self)
        legend_label.setWordWrap(True)
        legend_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self._table_box.addWidget(legend_label)
        remembered = self._scan_buffers.get(mode, "").strip()
        seed = remembered or scan_table_template("column_stack", specs)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(scaled_px(6, minimum=4))
        btn_row.addWidget(FluentLabel("template:", self))
        cs = FluentButton("column_stack", color=GREY)
        cs.setToolTip("Insert the column_stack template (one column per slot), adapted to the columns.")
        cs.clicked.connect(lambda *_a: self._insert_template("column_stack"))
        gr = FluentButton("grid", color=GREY)
        gr.setToolTip("Insert the grid template (every combination of the axis arrays).")
        gr.clicked.connect(lambda *_a: self._insert_template("grid"))
        btn_row.addWidget(cs, 0)
        btn_row.addWidget(gr, 0)
        btn_row.addStretch(1)
        self._table_box.addLayout(btn_row)
        editor = FluentCodeEdit(seed)
        editor.setMinimumHeight(scaled_px(120, minimum=90))
        editor.setToolTip("Python that assigns an (N_points x n_cols) array to 'scan_table' "
                          "(one column per slot, in the slot's native unit).")
        editor.textChanged.connect(self.changed)
        self._table_box.addWidget(editor)
        self._scan_code = editor

    def _insert_template(self, template: str) -> None:
        from ..neutral_atom.timing import scan_table_template
        if self._scan_code is not None:
            _none, api_mode, _scan = _scan_modes()
            specs = self._api_specs if self._mode == api_mode else self._scan_specs
            self._scan_code.setPlainText(scan_table_template(template, specs))

    def values_dict(self) -> dict:
        """The current ``{"api": {name: float}, "scan_mode": "none"|"api"|"scan",
        "scan_code": "<python>", "extra_delay": float}`` snapshot.  ONE ``scan_code`` = the ACTIVE
        mode's program (the build dispatches on ``scan_mode``); an empty / blank api row is dropped
        (keeps the template's value).  The whole-sweep count is the measurement's auto-injected
        ``Repeat (0 = ∞)`` knob (a pulse-scan pass IS a sweep) -- not a field here."""
        api: dict[str, float] = {}
        for name, w in self._api_widgets.items():
            text = w.text().strip()
            if not text:
                continue
            try:
                api[name] = float(text)
            except ValueError:
                continue
        code = self._scan_code.toPlainText() if self._scan_code is not None else ""
        extra = 0.0
        if self._extra_delay is not None:
            try:
                extra = float(self._extra_delay.text().strip() or 0.0)
            except ValueError:
                extra = 0.0
        return {"api": api, "scan_mode": self._mode, "scan_code": code, "extra_delay": extra}

    def seed_value(self, value) -> None:
        """Seed from a SAVED ``{"api", "scan_mode", "scan_code", "extra_delay"}`` blob (a load /
        round-trip).  The auto-form is rebuilt from the template path (the slots themselves come
        from the template, not the blob), so we stash the saved api values + mode + active program
        into the per-mode buffers here; the next :meth:`rebuild` (triggered by the template field's
        repopulation) restores them.  ``_pending_mode`` is honoured by ``rebuild`` over the default
        when the saved mode is still available for the loaded template."""
        if not isinstance(value, Mapping):
            return
        for name, v in dict(value.get("api") or {}).items():
            self._pending_api[str(name)] = f"{float(v):g}" if _is_number(v) else str(v)
        mode = str(value.get("scan_mode") or "").strip().lower()
        code = str(value.get("scan_code") or "")
        none, api, scan = _scan_modes()
        if mode in self._scan_buffers and code.strip():
            self._scan_buffers[mode] = code
        elif not mode and code.strip():
            # a legacy blob (no scan_mode) carried only scan_code -- keep it for both buffers so
            # whichever mode the rebuild defaults to picks it up.
            self._scan_buffers[scan] = self._scan_buffers[api] = code
        self._pending_mode = mode if mode in (none, api, scan) else None
        self._extra_remembered = (f"{float(value['extra_delay']):g}"
                                  if _is_number(value.get("extra_delay")) else self._extra_remembered)

    def has_scan_rows(self) -> bool:
        return self._n_slots > 0

    def all_scan_rows_filled(self) -> bool:
        """The form has SOMETHING to run: a fixed point (None mode) is always runnable; a sweep mode
        needs its program filled (or the column_stack default the build supplies, which a blank
        editor seeds, so a non-None mode is considered ready)."""
        none, _api, _scan = _scan_modes()
        if self._mode == none:
            return True
        return self._scan_code is not None and bool(self._scan_code.toPlainText().strip())


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
        ParamDecl(key="cmap", label="colormap", kind="choice", default="inferno", choices=CMAPS,
                  tooltip="Image colormap", display=True),
    ),
    "sites": (
        # A site map is a binary occupancy OVERLAY (faint ring = empty, bold ring =
        # occupied) on the camera FRAME.  The colormap applies to that frame underlay
        # (its counts colorbar); the rings carry no scale.  It takes ONE signal input --
        # the per-site occupancy (PANEL_INPUT_SLOTS["sites"], picked in the Setting's Source
        # section); its ring CENTRES and the frame UNDERLAY auto-resolve from that signal's
        # producing node, so they are NOT extra slots or params here.
        ParamDecl(key="cmap", label="colormap", kind="choice", default="gray", choices=CMAPS,
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


def _panel_param_default(kind: str, key: str) -> object:
    """The declared default of a panel kind's param, from the ONE ``PANEL_PARAMS`` catalog -- so a
    kind's colormap default (``2d`` -> ``inferno``, ``sites`` -> ``gray``) has a SINGLE source and is
    never hand-typed at a consume site.  Returns ``None`` for a kind/key with no declared param."""
    for decl in PANEL_PARAMS.get(str(kind), ()):  # noqa: SIM110 - explicit loop is clearer than any()
        if decl.key == key:
            return decl.default
    return None


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

# Curve fits a panel's Edit tab can run on its frozen snapshot -- the FULL
# DataFigure model set (confocal's combo_fit plus gaussian), available for EVERY
# plot kind (DataFigure picks the 1D or 2D path from the snapshot, so a 2D image
# fits the 2D-Gaussian ``center`` model).  label -> DataFigure method name.  The
# Setting popup carries NO fit controls (basic display only); fitting lives in
# each panel's own Edit tab.
FIT_MODELS: dict[str, str] = {
    "Lorentzian": "lorent",
    "Gaussian": "gaussian",
    "Lorentzian (Zeeman)": "lorent_zeeman",
    "Rabi": "rabi",
    "Exp decay": "decay",
    "2D center": "center",
}


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
# ``row`` is the card's pixel Y; ``_compact`` is a TOP-LEFT GRAVITY packer that floats every card
# UP then LEFT until blocked.  The CARD'S FORMAT (rounded corners, shadow, grey title strip,
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

# The header status readout (panel/signal counts, or a node-error banner) is a borderless
# label: grey normally, red on a node error (the colour IS the danger cue -- no box border,
# which as a line-edit drew a grey line across the header bottom above the tabs).
_SUMMARY_STYLE = f"color: {GREY}; background: transparent; border: none;"
_SUMMARY_STYLE_DANGER = f"color: {RED}; background: transparent; border: none;"


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


def _spans_mostly_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    """True when two 1-D spans ``[lo_a, hi_a]`` and ``[lo_b, hi_b]`` overlap by MORE THAN HALF of the
    NARROWER span -- a "majority overlap", not a mere touch.  This is the robust "in the same band"
    test the gravity packer uses: two cards count as sharing a column (or a row) only when one sits
    mostly in front of the other, so a card dragged to sit BESIDE another with a slight edge overlap is
    NOT judged to be stacked with it.  Adjacent (GAP-apart, zero intersection) spans return False."""
    inter = min(hi_a, hi_b) - max(lo_a, lo_b)            # signed intersection length (negative = a gap)
    if inter <= 0:
        return False
    narrower = min(hi_a - lo_a, hi_b - lo_b)
    return inter * 2 > narrower                          # more than half the narrower span is covered


def _gravity_slot(cfg, placed, board_w: int) -> tuple[int, int]:
    """NORTH-WEST (top-left) gravity ANCHORED at the card's CURRENT ``(col, row)``: from where it is, the
    card falls toward the top-left corner -- it rises straight UP until a card above blocks it, then slides
    straight LEFT until a card to its left (or the boundary) blocks it -- alternating to a fixed point.  It
    only ever moves UP and LEFT and is BLOCKED by neighbours, so it NEVER teleports over an obstacle to a
    far free slot: a card dragged to the bottom-left, blocked above by a panel and on the left by the
    boundary, STAYS bottom-left -- it is NOT flung to a top-right gap (#2).  ``placed`` = the already-
    settled cards; x is clamped inside ``board_w`` so a card stays on the board.

    "In my column" / "in my row" use a MAJORITY-overlap test (:func:`_spans_mostly_overlap`), not any
    touch: a placed card blocks my RISE only when its x-span covers MORE THAN HALF of the narrower of
    our two widths, and blocks my SLIDE-LEFT only when its y-span covers more than half the narrower
    height.  So a card dragged to sit BESIDE another with a slight edge overlap rests at the other's
    right edge instead of sinking beneath it (the reported "drop B to A's right -> B falls under A"
    bug): only a card the dragged one MOSTLY covers counts as sharing its band.

    Rest-up = just below the lowest placed card whose x-span mostly overlaps this card's column (else the
    top margin); rest-left = just right of the right-most placed card whose y-span mostly overlaps this
    card's row (else the left margin).  Sliding left changes which cards share the row and rising changes
    the column band, so we alternate until neither moves.  (A FRESH card spawns at :func:`_first_free_slot`
    so an Add tiles into the board; thereafter this local gravity keeps it -- and a drag -- where put.)"""
    w, h = _card_size(cfg.size)
    max_x = max(GAP, board_w - GAP - w)
    x = min(max(GAP, int(round(cfg.col))), max_x)        # start from the card's CURRENT spot (clamped on-board)
    y = max(GAP, int(round(cfg.row)))
    for _ in range(8):                                    # alternate rise / slide-left to a fixed point
        ny = GAP                                          # rise: rest just under any column-overlapping card
        for p in placed:
            px0, py0, px1, py1 = _aabb(p)
            # p is "above me in my column" only if its width MOSTLY covers mine -- a slight edge overlap
            # (a card sitting mostly beside me) must NOT make me sink under it.
            if _spans_mostly_overlap(x, x + w, px0, px1):
                ny = max(ny, py1 + GAP)
        nx = GAP                                          # slide left: rest just right of any LEFT card in the row
        for p in placed:
            px0, py0, px1, py1 = _aabb(p)
            # p is "to my left in my row" only if its height MOSTLY covers mine (majority overlap at the
            # risen y) AND it is actually to the LEFT (px0 < x).  Only a card to the left blocks LEFTWARD
            # travel: a card to the RIGHT that merely shares the row band (e.g. a right-column card grown
            # TALLER on resize) must NOT shove this card right past it (the resize bug that flung every
            # left-column card down/across when one card was enlarged); a mere slight vertical sliver of
            # overlap must not block the slide either.
            if px0 < x and _spans_mostly_overlap(ny, ny + h, py0, py1):
                nx = max(nx, px1 + GAP)
        nx = min(nx, max_x)
        if (nx, ny) == (x, y):
            break
        x, y = nx, ny
    return (x, y)


def _snap_drop_anchor(cfg, placed, board_w: int) -> tuple[int, int]:
    """NEAREST-ANCHOR landing rule for a DRAG-DROP: the dragged card's raw drop TOP-LEFT
    (``cfg.col``, ``cfg.row``) competes over two candidate anchor sets and lands on the
    closest one (squared euclidean distance).

    A. CORNERS -- every OTHER placed card's top-left ``(c.col, c.row)``.  Winning means the
       drop DISPLACES that card: the dragged card is seeded exactly on its corner, and the
       subsequent :func:`_compact` (with the dragged card ``active``) lets it win the
       coincident slot while NW gravity re-settles the displaced card out of the way (below).
    B. VACANCIES -- the :func:`_first_free_slot` candidate lattice (x = GAP, each placed
       card's right edge + GAP, and each left edge; y = GAP and each bottom edge + GAP),
       kept only where the dragged card FITS: on the board (``GAP <= x <= max_x``, the same
       clamp as :func:`_gravity_slot`) and clear of every placed card by GAP
       (:func:`_overlaps_with_gap`).  Winning means the drop lands on that free anchor.

    Deterministic tie-break: distance, then y, then x, then corner-over-vacancy.  Pure
    geometry (no Qt).  Snapping only picks the SEED -- the ordinary :func:`_compact` NW
    gravity then packs from it, so the no-teleport law still holds: the geometrically
    NEAREST anchor wins, never a far free slot.  Applied ONLY on the drag-release path
    (resize / reflow / Add placement never snap)."""
    w, h = _card_size(cfg.size)
    max_x = max(GAP, board_w - GAP - w)
    drop_x, drop_y = int(round(cfg.col)), int(round(cfg.row))

    def _key(x: int, y: int, rank: int) -> tuple[int, int, int, int]:
        return ((x - drop_x) ** 2 + (y - drop_y) ** 2, y, x, rank)

    best = None
    for p in placed:                                     # A: corners (rank 0 wins a full tie)
        cand = _key(int(p.col), int(p.row), 0)
        best = cand if best is None or cand < best else best
    xs = {GAP}
    ys = {GAP}
    for p in placed:                                     # B: the _first_free_slot vacancy lattice
        px0, _py0, px1, py1 = _aabb(p)
        xs.add(px1 + GAP)
        xs.add(px0)                                      # align left edges (tuck under a wider card)
        ys.add(py1 + GAP)
    for x in xs:
        if not (GAP <= x <= max_x):
            continue
        for y in ys:
            if _overlaps_with_gap((x, y, x + w, y + h), placed):
                continue
            cand = _key(x, y, 1)
            best = cand if best is None or cand < best else best
    # ``placed`` empty -> A is empty but (GAP, GAP) is always a clear vacancy, so best is set.
    _d2, y, x, _rank = best
    return (x, y)


def _first_free_slot(cfg, placed, board_w: int) -> tuple[int, int]:
    """The TOP-MOST then LEFT-MOST free ``(col, row)`` where ``cfg`` fits clear of every placed card (GAP
    apart, inside ``board_w``).  Used ONLY to SEED where a freshly-Added panel spawns, so an Add TILES into
    the board (fills the top row left-to-right, then wraps to the next row) instead of starting a fresh
    column.  (Compaction + drag use :func:`_gravity_slot`'s local NW gravity, which then KEEPS a card here.)
    Candidate points are GAP (origin) + each placed card's right/bottom edge (``+GAP``) and its left/top
    edge (so a card can tuck under a wider one); swept by y then x, first feasible wins."""
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
    GUI passes the scroll viewport width to ``_compact`` instead, so the board wraps at the edge."""
    widest = max((_card_size(c.size)[0] for c in configs), default=_card_size("1x2")[0])
    return max(2 * widest + 3 * GAP, _min_board_width(configs))


def _compact(configs: Sequence["PanelConfig"], active: "PanelConfig | None" = None,
             board_w: int | None = None) -> bool:
    """TOP-LEFT GRAVITY packer over pixel AABBs (#H3s-F8).  Every card floats UP then LEFT until it
    is blocked by another card or the board edge; the clear distance to every neighbour and to the
    top-left origin is a UNIFORM ``GAP`` on all four sides (there is NO column grid -- ``col`` is a
    pixel x, ``row`` a pixel y).  So a card dropped low-right snaps up-left into the first free slot,
    and cards pack side by side until the board (``board_w``, the live scroll-viewport width) is full.

    Cards are placed in READING ORDER of their CURRENT positions -- ``(row, col)``, top-to-bottom
    then left-to-right -- with the just-moved ``active`` card winning any TIE (it sorts first when it
    shares a row/col with another, so it claims the contested slot and the others reflow around it).
    Placing in reading order is what makes the layout STABLE and DETERMINISTIC: the drop point only
    sets a card's reading-order rank, so dropping a card back where it was reproduces the previous
    packing exactly, and a settled board is a fixed point (a second pass moves nothing -> False).

    For each card we take the TOP-MOST then LEFT-MOST feasible slot (see ``_gravity_slot``): a card
    with nothing above or left of it lands at exactly ``(GAP, GAP)``.  ``board_w`` defaults to a
    two-wide fallback for headless callers.  Returns True if any card's ``(col, row)`` changed."""

    moved = False
    placed: list["PanelConfig"] = []
    # No live viewport -> a two-wide fallback so cards CAN pack side by side; a given viewport width
    # is honoured but never below one-card-wide (a too-narrow viewport must still fit the widest card).
    board_w = _board_width(configs) if board_w is None else max(board_w, _min_board_width(configs))
    # Reading order (row, then col); the active card wins a tie so a card dropped ONTO another's slot
    # claims it (the other reflows) instead of yielding -- a 0 sort key only on an exact coincidence.
    order = sorted(configs, key=lambda c: (c.row, c.col, 0 if c is active else 1))
    for config in order:
        col, row = _gravity_slot(config, placed, board_w)
        if (config.col, config.row) != (col, row):
            config.col, config.row = col, row
            moved = True
        placed.append(config)
    return moved


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
        # (``value = np.log(f)``) do NOT match this and leave the input alone.
        m = re.fullmatch(r"value\s*=\s*([A-Za-z_]\w*)", self.source.strip())
        if m and m.group(1) != "signal" and self.inputs:
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
    measurement / processor / task); ``name`` is the catalog spec's name (or
    ``"Camera (live frames)"`` for the camera).  ``values`` is the last param-form
    ``{key: value}`` it was built / run with, so reopening its Edit restores them.
    A node is always added STOPPED -- nothing runs until Start in its Edit."""

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
                 grid_recipe_provider=None):
        # Titled frame: the title strip carries the panel KIND (top-left) and the
        # Setting button (top-right), so the card is delineated like the rest.
        super().__init__(PANEL_KINDS[config.kind], parent, shadow=True)
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
        self.plotter = None
        self.canvas = None
        # The console header's "Selectors" switch state for THIS card (set via
        # ``set_selectors_enabled``; default OFF = the historical display-only Monitor board).
        # Every plotter (re)build parks its selector layer to this flag (``_apply_selectors_state``),
        # so a fresh figure always inherits the switch instead of coming up live.
        self._selectors_on = False
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
        scrollbar_w = self._settings_scroll.verticalScrollBar().sizeHint().width() or scaled_px(12)
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

            # FACET chooser (grid panels only): which axis of the bound (repeat, points, *data_dim)
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
        top_y = anchor.y() + scaled_px(2)
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
        top_y = anchor_y + scaled_px(2)
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

    def _on_relim_mode(self, mode: str) -> None:
        """Back-compat shim: drive a relim change through the declarative ``_set_param`` path (the relim
        chooser is now a ParamDecl, #H3v-4b).  Kept so callers that set relim by name -- notebook code,
        tests -- still work; ``_set_param`` does the persist + plotter push + fixed-row reveal."""
        self._set_param("relim", str(mode))

    def apply_fixed_lims(self, lo: float, hi: float) -> None:
        """Persist + apply the fixed lo/hi NOW through the ONE in-place entry every surface uses
        (``apply_param``'s relim family): on a ZOOMED grid the pin is ALSO stored on the parked grid
        (thumbnails carry it when the zoom returns); otherwise it lands on the live plotter directly
        (a GridPlot fans it out to its thumbnails itself).  The Setting popup's lo/hi inputs and the
        Edit tab's both route here -- no second hand-copied push path."""
        self.config.params["fixed_lo"], self.config.params["fixed_hi"] = float(lo), float(hi)
        if self._grid_focus is not None:
            self._set_focused_grid_param("fixed_lo", float(lo))
            self._set_focused_grid_param("fixed_hi", float(hi))
        elif self.plotter is not None:
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
        display = dict(out.get("display_params") or {})
        for decl in PANEL_PARAMS.get(self._param_kind(), ()):    # exactly the sub_plot_kind's knobs -- no hard-code
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
    def _facet(self) -> str:
        """The panel's facet declaration (``config.params["facet"]``): which axis of the bound
        ``(repeat, points, *data_dim)`` block expands into the grid cells.  Empty = a RECIPE grid
        (the loaded-figure snapshot, the pre-facet behaviour)."""
        return str(self.config.params.get("facet") or "")

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
        pts, dim = self._facet_value_shapes()
        return default_sub_plot_kind(self._facet() or "repeat", points_shape=pts, data_shape=dim)

    def _facet_cells(self, value):
        """Slice the bound block into the per-cell inputs through the ONE rule (live.facet_cells)."""
        from .live import facet_cells
        block = np.asarray(value, dtype=float)
        if block.ndim < 2:
            raise ValueError(
                "a facet grid slices the bound signal's (repeat, points, *data_dim) block; got shape "
                f"{block.shape} -- bind a measurement's block signal (not a scalar).")
        pts, _ = self._facet_value_shapes()
        if int(np.prod(pts, dtype=np.int64) if pts else 0) != int(block.shape[1]):
            pts = ()                              # declared shape does not match this block -> flat points
        return facet_cells(block, self._facet(), sub_plot_kind=self._resolved_sub_kind(),
                           points_shape=pts, repeat_mode=self._repeat_mode_value())

    def _build_facet_plotter(self, value, *, interactions: bool):
        """Build this panel's FACET grid through the ONE factory + replay its persisted per-kind
        display knobs (bins / fit / ylog / cmap) AND the relim family (BaseLivePlot.apply_param owns
        them all).  The ONE builder the live card (display-only) and the Edit-tab snapshot
        (interactive) share -- so the two can never drift."""
        from .live import grid as build_facet_grid
        sub = self._resolved_sub_kind()
        plotter = build_facet_grid(
            self._facet_cells(value), sub_plot_kind=sub, size=self.config.size,
            display=False, interactions=interactions, title=self.config.title or "")
        for key in [d.key for d in PANEL_PARAMS.get(sub, ())] + ["relim", "fixed_lo", "fixed_hi"]:
            if key in self.config.params:
                plotter.apply_param(key, self.config.params[key])
        return plotter

    def _facet_choices(self) -> list[tuple[str, str, bool]]:
        """The facet dropdown's ``[(stored value, display text, enabled)]`` -- derived from the
        producing node's declared axis structure, with the axis LENGTH shown so the operator picks
        by meaning ('scan axis 0 (5)').  An axis longer than :data:`MAX_GRID_CELLS` is listed but
        DISABLED (the grid factory refuses it -- the UI would freeze); the "(saved figure)" row
        exists only when the bound node actually carries a saved grid recipe to replay."""
        from .live import MAX_GRID_CELLS
        out = []
        if self._grid_recipe_or_none() is not None:
            out.append(("", "(saved figure)", True))
        out.append(("repeat", "repeat", True))

        def _axis(value, text, n):
            ok = int(n) <= MAX_GRID_CELLS
            out.append((value, text if ok else f"{text} – too many", ok))

        pts, dim = self._facet_value_shapes()
        if len(pts) > 1:
            for i, n in enumerate(pts):
                _axis(f"points:{i}", f"scan axis {i} ({n})", n)
        elif pts and pts[0] > 1:
            _axis("points:0", f"scan axis 0 ({pts[0]})", pts[0])
        for i, n in enumerate(dim):
            if n > 1:
                _axis(f"dim:{i}", f"data axis {i} ({n})", n)
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
        value = str(self.facet_combo.itemData(int(index)) or "")
        if value == self._facet():
            return
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

        Two persisted knobs: ``config.params["relim"]`` (confocal naming:
        ``tight`` / ``normal``) and ``config.params["unit_index"]`` (the
        x-axis unit cycle count).  There is no manual xlim/ylim path -- the
        Setting popup has no manual lim controls (zoom/pan + Edit… to the
        panel's Edit tab handle interactive ranging), and a colorbar visibility
        path is absent because the colormap chooser IS the colorset chooser."""
        if self.plotter is None:
            return
        self._apply_lim_to_plotter()                     # relim family via the ONE in-place entry
        self._apply_unit()

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
            # Record the raw drop pixel as this card's (col, row) seed; the console then SNAPS the
            # seed to the NEAREST anchor (another card's corner = displace it, or the closest free
            # lattice point -- :func:`_snap_drop_anchor`) via ``dropped``, and _compact gravity-packs
            # it top-left (it is the ``active`` card so it wins the anchor it snapped onto).
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
            self.plotter.title = self.config.title
            self.plotter._apply_title()
            if self.canvas is not None:
                self.canvas.draw_idle()
        self.changed.emit()

    def _on_size(self, size: str) -> None:
        self.config.size = str(size)
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

    def _set_param(self, key: str, value) -> None:
        """A declarative parameter edit: store, apply, mark dirty.

        Most params (colormap / bins / toggles …) change the plot's STRUCTURE, so they tear the plotter
        down + re-render.  ``relim`` is the exception: it is a LIVE axis adjustment (the dead-band-aware
        autoscale mode), so it pushes onto the existing plotter WITHOUT a teardown and reveals the fixed
        lo/hi row only in ``fixed`` mode -- the SAME effect the old hand-wired ``_on_relim_mode`` had,
        now reached through this one declarative path (#H3v-4b)."""
        if self.config.params.get(key) == value:
            return
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
        self._rerender_last()

    def _apply_source(self) -> None:
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
    def refresh(self, namespace: dict[str, object]) -> None:
        """Evaluate this panel's source against ``namespace`` and render the plot.
        Every failure lands on the gear/status -- a bad expression in one panel
        must never break the console or its siblings."""

        # A BLANK panel (a freshly added pure view, source not yet wired) sits
        # quietly with a hint -- it is decoupled, so it shows nothing until the
        # user picks a signal in its Setting.  Not an error.
        if not str(self._compiled_source).strip():
            self.set_status("pick a signal in Setting", error=False)
            return
        try:
            # The three decoupled stages (#H3l): signal (per data_points slice) -> repeat reduce
            # (the measurement's own repeat axis, per the plot's repeat_mode) -> dim split.  The
            # measurement OWNS ``repeat``; the plot only chooses how to display it.
            value = self._signal_then_repeat(namespace)
            self._render(value, namespace)
        except Exception as exc:
            # A failed eval (e.g. the bound signal is not in this namespace yet -- a still-WAITING
            # producer) must NOT latch the panel: keep the LAST GOOD namespace (set only on success
            # below), so re-picking a signal / a display change replays valid data, and a fresh hub
            # tick re-renders the moment the signal arrives -- never "remove the panel to recover"
            # (#H3w-2).
            self.set_status(str(exc).splitlines()[0][:160] or type(exc).__name__, error=True)
            return
        # Remember the LAST SUCCESSFULLY-rendered namespace so a later display-only change (cmap /
        # relim / a source re-pick) can re-render immediately from it even when the producing
        # measurement is stopped (hub version frozen, no fresh tick) -- see _rerender_last.  Set ONLY
        # on success, so an error never overwrites it with a namespace that re-errors on replay.
        self._last_namespace = namespace
        shot = namespace.get("shot")
        self.set_status(f"shot {int(shot)}" if isinstance(shot, (int, float)) else "ok", error=False)

    def _signal_expr(self):
        """This panel's source as the ONE reusable :class:`SignalExpr` (the slot rule + the
        ``value = ...`` contract live there, shared with processors / pulse-scan).  Lazy import
        keeps the frontend module off neutral_atom's import graph (every neutral_atom use here
        is lazy)."""
        from ..neutral_atom.operations.signal_expr import SignalExpr
        return SignalExpr(self.config.inputs, self._compiled_source)

    def _signal_then_repeat(self, namespace: Mapping[str, object]):
        """The decoupled plot pipeline.  The MEASUREMENT owns the repeat axis: it publishes a RAW
        block whose LEADING axis is the repeat (a 1-D scan's ``(repeat, points, dim)``, a camera's
        ``(repeat, H, W)``); a node with no repeat publishes a plain value (a curve / image / scalar).

        The plot runs the ``value = ...`` expression PER REPEAT (``signal`` = that repeat's whole
        core -- the (H, W) frame, or the (points, dim) curve -- so the user can write ``value =
        signal[0]`` etc.); ONLY the repeat axis is looped, so it stays decoupled.  Then ``repeat_mode``
        reduces the repeat axis (average = the long-exposure mean over the repeats that have data, add
        = sum, create = one line per repeat, ...).  ``_repeat_cur`` (how many repeats hold data) drives
        the plotter's "xN" ylabel."""
        from .live import reduce_repeat, repeats_with_data
        # A pulse panel's ``value`` is a STRUCTURED object (a sequence / PulseTableState), not an array --
        # it has no repeat axis and must NOT be float-coerced.  Read the bound signal (or the ``value =
        # ...`` expression result) as-is and hand it straight to _render / _build_plot, exactly as the
        # array pipeline hands a reduced block to the other kinds -- the kind (not the shape) decides how
        # its own data is consumed (a 2d reshapes to an image, sites uses centres, pulse uses the sequence).
        if self.config.kind == "pulse":
            self._repeat_cur = 1
            return self._signal_expr().evaluate(namespace)
        mode = self._repeat_mode_value()                          # clamped to this kind's valid modes (#issue-1)
        # core_ndim of the bound signal (a Judge-occupancy output declares its OWN per-signal structure,
        # so its clean ``(repeat, n_sites)`` block is collapsed by STRUCTURE, not an ndim guess; camera /
        # scan stay on the legacy ndim>=3 path with core_ndim=None) -- #H3s-F3.
        core_ndim = self._bound_core_ndim()
        block, had_repeat = self._eval_signal_per_slice(namespace, core_ndim=core_ndim)
        if isinstance(block, (list, tuple)):                      # a free expression returned a list
            self._repeat_cur = 1
            return np.asarray(block, dtype=float)
        b = np.asarray(block, dtype=float)
        if had_repeat and self.config.kind == "1d" and core_ndim is None and b.ndim == 2:
            b = b[:, :, None]                                     # legacy (repeat, points) -> add dim axis
        # a repeat block (axis 0 = repeat) -- structure-driven when core_ndim is declared, else ndim>=3
        if had_repeat and (b.ndim == 1 + core_ndim if core_ndim is not None else b.ndim >= 3):
            self._repeat_cur = repeats_with_data(b, core_ndim=core_ndim)   # scan / camera / occupancy block
            if self.config.kind == "grid" and self._facet():
                # A FACET grid owns the repeat axis ITSELF: facet=repeat slices it into cells, and any
                # other facet collapses the leftover repeats INSIDE facet_cells (per repeat_mode) --
                # reducing here would hand the slicer a block whose repeat axis is already gone
                # (facet=repeat would always show ONE cell).
                return b
            # ONE reducer dispatches on the chosen mode; the only kind input is hist=, because a histogram
            # has NO x-axis in its core (#iron-law): every non-pool reduction flattens to one sample set,
            # and 'create' keeps each repeat's whole core as a column (n_samples, R) -- a trace's create
            # instead keeps the points axis.  A trace/image block and a hist block can be the SAME ndim
            # ((R, points, dim) vs (R, 1, n_sites)), so the kind, not the shape, picks the layout.
            return reduce_repeat(b, mode, core_ndim=core_ndim, hist=(self.config.kind == "hist"))
        self._repeat_cur = 1                                      # no repeat axis -> nothing to reduce
        return b

    def _eval_signal_per_slice(self, namespace: Mapping[str, object], *, core_ndim=None):
        """Run the ``value`` expression once PER REPEAT and re-stack -> ``(repeat, *result), True``.
        Every signal the expression reads that carries a repeat axis (the bound ``signal`` AND any
        raw hub signal the source names directly, e.g. ``value = frame_0``) is presented as that
        repeat's whole core -- the (H, W) frame / the (points, dim) curve -- so the user can process
        it (``signal[0]``, ``frame.mean()``, ...) while the repeat axis stays OUTSIDE the expression.
        When nothing read has a repeat axis (a single frame / curve / scalar) the expression runs
        once -> ``value, False``.  The default ``value = signal`` on one bound block short-circuits.

        WHETHER the bound ``signal`` carries a repeat axis is STRUCTURE-driven when its producing signal
        declares ``core_ndim`` (a clean occupancy ``(repeat, n_sites)`` is ndim ``1 + 1``, #H3s-F3);
        camera / scan slots (core_ndim None) keep the ndim>=3 rule."""
        from ..neutral_atom.operations.signal_expr import DEFAULT_SOURCE

        def _slot_has_repeat(s) -> bool:
            if s is None:
                return False
            return np.ndim(s) == 1 + int(core_ndim) if core_ndim is not None else np.ndim(s) >= 3

        expr = self._signal_expr()
        # A panel reads EXACTLY the signal it is bound to -- no rewrite.  A camera's `frame` shows the
        # camera's own (repeat, 1, H, W) block (so average/create work), INDEPENDENT of any Judge
        # processor (a Judge is a separate reactive node; its `frame_judged` is its OWN output, bound
        # explicitly when wanted).  The site-map underlay coherence is handled separately (#_sites_aux).
        sig = expr.signal_for(namespace)
        slots = sig if isinstance(sig, list) else [sig]
        # raw hub signals the source NAMES directly (not via ``signal``) that carry a repeat axis -- these
        # are NOT the bound signal, so they keep the ndim>=3 rule (no per-signal core_ndim for them).
        raw_names = [n for n in expr.co_names()
                     if n != "signal" and np.ndim(namespace.get(n)) >= 3]
        sig_rep = bool(slots) and all(_slot_has_repeat(s) for s in slots)
        if not sig_rep and not raw_names:                         # no repeat axis -> evaluate once
            ns = dict(namespace); ns["signal"] = sig
            return expr.exec_in(ns), False
        if sig_rep and not raw_names and len(slots) == 1 \
                and str(self._compiled_source).strip() == DEFAULT_SOURCE:
            return np.asarray(slots[0], dtype=float), True        # identity: the block IS the value
        raw = {n: np.asarray(namespace[n], dtype=float) for n in raw_names}
        sl = [np.asarray(s, dtype=float) for s in slots] if sig_rep else None
        R = int((sl[0] if sig_rep else raw[raw_names[0]]).shape[0])

        def _run(r):
            ns = dict(namespace)
            for n, a in raw.items():
                ns[n] = a[r]                                       # slice each named block to this repeat
            ns["signal"] = (sl[0][r] if len(sl) == 1 else [a[r] for a in sl]) if sl is not None else sig
            return np.asarray(expr.exec_in(ns), dtype=float)

        return np.stack([_run(r) for r in range(R)], axis=0), True

    def _bound_structure(self):
        """The producing node's ``{points_shape, data_shape, grid_shape}`` for this panel's bound
        signal -- authoritative ONLY for the DEFAULT identity source (``value = signal``), because a
        custom expression rewrites the core's shape so the node's declared structure no longer
        describes it (#H3o).  ``None`` -> the reshape falls back to shape inference."""
        from ..neutral_atom.operations.signal_expr import DEFAULT_SOURCE
        if str(self._compiled_source).strip() != DEFAULT_SOURCE:
            return None
        if not (self.config.inputs and callable(self.structure_provider)):
            return None
        try:
            return self.structure_provider(self.config.inputs[0])
        except Exception:
            return None

    def _bound_core_ndim(self):
        """``core_ndim`` (= len(points)+len(data)) for the bound signal, fed to ``reduce_repeat`` /
        ``repeats_with_data`` so the LEADING repeat axis is collapsed by STRUCTURE (#H3s-F3) -- a clean
        occupancy ``(repeat, n_sites)`` (ndim 2, core_ndim 1) IS a repeat block where a bare ndim guess
        would miss it.  Returned ONLY when the bound signal declares its OWN per-signal structure (a
        Judge-occupancy output): for camera / scan signals (node-level structure, ``per_signal`` False)
        this returns ``None`` so ``reduce_repeat`` keeps the legacy ndim>=3 rule byte-identically.
        ``None`` too for a custom expression (``_bound_structure`` already gates on identity source)."""
        st = self._bound_structure()
        if st is None or not st.get("per_signal"):
            return None
        return int(st.get("core_ndim", 0))

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
        if centers.ndim != 2 or centers.shape[1] < 2:
            raise ValueError(f"centres signal must have shape (N, 2); got {centers.shape}")
        image = namespace.get(image_name) if image_name else None
        if image is not None:
            image = np.asarray(image, dtype=float)
            if image.ndim >= 3:        # a (repeat, H, W) frame_judged block -> ONE coherent (H,W) underlay
                from .live import reduce_repeat
                # The underlay obeys the SAME ``repeat_mode`` Setting as the occupancy rings (#H3u-3):
                # average = a long-exposure mean over the kept frames (coherent with the averaged
                # occupancy), replace/roll = the latest frame, add = the summed frame.  A 2-D image
                # cannot be per-repeat lines, so 'create' falls back to the latest (exactly as a 2-D
                # panel offers no 'create') -- the rings and the frame are always the same reduction.
                mode = self._repeat_mode_value()      # the ONE clamped reader (#A2)
                image = np.squeeze(reduce_repeat(image, "replace" if mode == "create" else mode))
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
        frames = (namespace or {}).get("__coord_frames__")
        if not isinstance(frames, dict) or not frames:
            return None
        for name in self._co_names():
            if name in frames:
                return frames[name]
        return None

    def _monitor_source_key(self, namespace: Mapping[str, object] | None):
        """Version key of the signals this panel's source reads, or None when
        none are detectable.  None => caller rolls every tick (safe fallback)."""
        versions = (namespace or {}).get("__sig_versions__")
        if not isinstance(versions, dict) or not versions:
            return None
        refs = [n for n in self._co_names() if n in versions]
        if not refs:
            return None
        return tuple(sorted((n, versions[n]) for n in refs))

    def set_selectors_enabled(self, on: bool) -> None:
        """The console header's "Selectors" switch for THIS card: remember the desired state and
        gate the CURRENT plotter now (in place -- no rebuild, no flash).  Every later (re)build /
        focus swap re-applies it through ``_apply_selectors_state``, so a fresh figure always
        inherits the switch."""
        self._selectors_on = bool(on)
        self._apply_selectors_state()

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
                    if self.canvas is not None:
                        self.canvas.draw_idle()
            return
        value = self._coerce(value)
        kind = self.config.kind
        # A pulse panel renders a STRUCTURED figure (a whole timeline) from its sequence object; there is
        # no in-place array ``update`` for it, so it (re)builds the PulseSequenceFigure through _build_plot
        # every refresh -- cheap (a static, rarely-changing recipe) and always faithful.
        if kind == "pulse":
            self._build_plot(value, namespace)
            self._force_rebuild = False
            return
        # A grid panel is either a LIVE facet grid (the bound block sliced along the declared axis,
        # cells moved in place each tick) or a RECIPE snapshot (a loaded figure -- built once, never a
        # per-tick redraw: re-rendering all N cells on every hub bump froze the console, #3).
        if kind == "grid":
            if self._facet():
                # A LIVE facet grid: first show builds; every later tick moves the cells IN PLACE
                # (update_cells -- the grid counterpart of a standalone kind's update).  A cell-count
                # change (the scan restructured) rebuilds through the ordinary build-then-swap.
                # Every (re)build is followed by the Setting-rows sync: the build is the ONE point
                # every kind-changing path converges on (a pick, a load, an expression edit).
                if self.plotter is None or self._force_rebuild:
                    self._build_plot(value, namespace)
                    self._force_rebuild = False
                    self._sync_settings_param_rows()
                    return
                per_cell = self._facet_cells(value)
                if len(per_cell) != getattr(self.plotter, "n_cells", -1):
                    self._build_plot(value, namespace)
                    self._force_rebuild = False
                    self._sync_settings_param_rows()
                    return
                self.plotter.update_cells(per_cell)
                if self.canvas is not None:
                    self.canvas.draw_idle()
                return
            # a RECIPE grid is a SNAPSHOT built from its recipe: (re)build only on the FIRST show or
            # when a display knob marked a rebuild -- NEVER a per-tick redraw (#3).
            if self.plotter is None or self._force_rebuild:
                self._build_plot(value, namespace)
                self._force_rebuild = False
            return
        rebuild = self.plotter is None or self._force_rebuild
        if not rebuild and kind in ("2d", "1d", "sites"):
            rebuild = tuple(np.shape(value)) != self._value_shape
        if not rebuild and kind == "2d":
            # A ROI that *shifts* without resizing keeps the frame shape, but the
            # image's pixel coordinates moved -- rebuild so the axes track the ROI.
            roi = self._source_coord_frame(namespace)
            rebuild = (list(roi) if roi else None) != getattr(self, "_roi_built", None)
        if rebuild:
            self._build_plot(value, namespace)             # build-then-swap: a failed build keeps the old figure
            self._force_rebuild = False                    # cleared ONLY after a clean build (else retry next tick)
            self._value_shape = (1,) if isinstance(value, float) else tuple(np.shape(value))
            if kind == "monitor":
                # The build already plotted this value as the first point; record
                # its source version so the next UNRELATED bump won't duplicate it.
                self._last_monitor_key = self._monitor_source_key(namespace)
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
        if self.canvas is not None:
            self.canvas.draw_idle()

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
            # A GRID stays interactions=False even under the "Selectors" switch: GridPlot's own
            # _attach_interactions wires a canvas-level double-click focus-zoom that would DOUBLE-FIRE
            # against this card's _on_grid_canvas_click (two competing focus mechanisms).  The grid's
            # interactive surface is its ENLARGED cell (see _focus_grid_cell), which builds WITH
            # selectors and follows the switch like any standalone panel.
            facet = self._facet()
            if facet:
                # A FACET grid: the bound block IS the data -- slice it along the declared axis
                # (facet_cells, the ONE rule) and build through the ONE shared builder (the Edit-tab
                # snapshot uses the same one); every tick after this first build moves the cells in
                # place (update_cells in _render).
                plotter = self._build_facet_plotter(value, interactions=False)
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
                plotter = build_grid_figure(recipe, interactions=False, size=size, display=False)
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
                labels=("Camera x (px)", "Camera y (px)", label),
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
            xx, yy = np.meshgrid(x0 + np.arange(nx, dtype=float), y0 + np.arange(ny, dtype=float))
            data_x = np.column_stack([xx.ravel(), yy.ravel()])
            plotter = panel_plot(
                data_x, arr.ravel(), kind="2d", size=size, interactions=True,
                cmap=_resolved_cmap("2d", self.config.params),   # operator pick, else the kind default (ONE resolver)
                **self._view_kwargs("2d"),
                labels=(xlabel, ylabel, ""), title=self.config.title or None)
        elif kind == "monitor":
            length = max(20, int(self.config.params.get("length", 300)))
            history = np.full(length, np.nan)
            plotter = panel_plot(
                np.arange(length, dtype=float), history, kind="monitor", size=size, interactions=True,
                show_dist=bool(self.config.params.get("show_dist", True)),
                labels=("Shots ago", self._source_axis_label() or label, "Z"),
                **self._view_kwargs("monitor"),
                title=self.config.title or None)
            plotter.roll(float(value), draw=False)
        elif kind == "hist":
            plotter = panel_plot(
                np.asarray(value, dtype=float), kind="hist", size=size, interactions=True,
                bins=int(self.config.params.get("bins", 60)), ylog=bool(self.config.params.get("ylog", False)),
                fit=str(self.config.params.get("fit", "double")),
                **self._view_kwargs("hist"),                # relim/fixed pins the VALUE (x) axis (#3)
                labels=("Value", "Shots", "Population"), title=self.config.title or None)
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
                # its own line (the dimension axis O2, or one line per repeat in ``create`` mode); a
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
                labels=(xlabel, ylabel, "Z"),
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
        # re-apply persisted Setting toggles (unit + manual x/y limits) to the
        # FRESH plotter every rebuild -- the panel rebuilds whenever its data
        # shape changes, so without this the toggle would silently revert.
        self._apply_display_params()
        # park (or arm) the fresh plotter's selector layer to the header's "Selectors" switch --
        # a rebuild must inherit the current state, never come up with live selectors while OFF.
        self._apply_selectors_state()
        self.canvas.draw()          # SYNCHRONOUS: the swapped-in canvas shows its FINAL frame at once
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
            parked.grid.store_display_param(str(key), value)
        except Exception:
            pass
        # Apply to the enlarged view in place.  A knob it can't apply live re-focuses the same cell ONLY
        # when the knob actually belongs to the per-site kind's own param set (bins/fit/ylog/cmap ...) --
        # an unrelated panel knob (e.g. ``repeat_mode``, which has no effect on a snapshot grid's enlarged
        # cell) is just stored, so it never triggers a visible rebuild/flash of the enlarged view.
        if self.plotter is not None and not self.plotter.apply_param(str(key), value):
            if any(d.key == str(key) for d in PANEL_PARAMS.get(self._param_kind(), ())):
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
        # copy an enlarged-view threshold cut back onto the grid cell (the grid thumbnail + save recipe read it)
        cell = getattr(grid, "cell_renderer", None)
        if cell is not None and hasattr(cell, "sync_threshold_from_focus"):
            try:
                cell.sync_threshold_from_focus(k, focus)
            except Exception:
                pass
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
        # the grid thumbnails may have picked up a new threshold -> redraw them, then show
        if hasattr(grid, "_redraw_thumbnails"):
            try:
                grid._redraw_thumbnails()
            except Exception:
                pass
        if grid_canvas is not None:
            grid_canvas.draw()
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
    from ..neutral_atom.operations.measurement import ParamDecl
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

    def _spin(self, decl, *, integer: bool, value=None):
        """A bounded spin box for a float/int param (range + unit from the decl)."""
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
        spin.valueChanged.connect(self._refresh_start_enabled)
        return spin

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
            (d.label or d.key) + (f" ({d.unit})" if d.unit else "") + (" *" if d.required else "")
            for d in decls if d.kind not in SPAN_KINDS
        ]
        self._form_label_w = setting_label_width(scalar_labels or [""], minimum=72)
        ctx = self._param_context()
        for decl in decls:
            kind = decl.kind
            # Show the READABLE label ("Pulse template" / "Signal (y)" / "Output name"), not the
            # raw build-call key ("template" / "y" / "y_name") -- the key is unreadable in a form
            # an experimenter actually uses (#H3); the tooltip still carries the full meaning.
            label_text = (decl.label or decl.key) + (f" ({decl.unit})" if decl.unit else "")
            if decl.required:
                label_text += " *"
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
            from ..neutral_atom.operations.measurements.pulse_scan import _resolve_probe_template
            state = _resolve_probe_template(path)
        except Exception:
            widget.rebuild([], [])
            return
        api_rows = [(s.name, str(s.kind), str(s.target), str(s.unit), float(state._read_api_field(s)))
                    for s in state.api_slots]
        scan_rows = [(f"s{i}", str(s.kind), str(s.target), str(s.unit), s.label)
                     for i, s in enumerate(state.scan_slots)]
        widget.rebuild(api_rows, scan_rows)

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
        # the rebuild runs with the stash write() left (saved api values + scan_mode + program) and
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

      * curve fit  -- model (gated by the panel's plot kind) + free-text args +
        Fit / Clear, run on the snapshot via :class:`DataFigure`;
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
        self.fit_cmd = None
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
        functional = [s for s in PANEL_PARAMS.get(card._param_kind(), ()) if not s.display]
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
            display_specs = ([s for s in PANEL_PARAMS.get(card._param_kind(), ()) if s.display]
                             + [_RELIM_PARAM] + list(card._repeat_param_specs()))
            self.ed_params = card._emit_param_rows(display_specs, col.addWidget, self._edit_param, disp_lw)
            self.ed_cmap = self.ed_params.get("cmap")        # named back-refs (kept for tests / clarity)
            self.ed_relim = self.ed_params.get("relim")
            # fixed lo/hi: the IDENTICAL bespoke [lo | hi] row the Setting popup builds (shared helper),
            # shown only when relim == "fixed"; _edit_param toggles it when relim changes.
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

            section("Fit")
            self.fit_combo = FluentComboBox()
            self.fit_combo.addItems(list(FIT_MODELS))
            self.fit_combo.setFixedWidth(scaled_px(150, minimum=120))
            self.fit_combo.setToolTip(
                "Curve-fit model (the full DataFigure set).  '2D center' fits a 2D image\n"
                "(2D Gaussian); the others fit the 1D trace.")
            fit_btn = FluentButton("Fit", color=ACCENT)
            fit_btn.clicked.connect(self.do_fit)
            clear_btn = FluentButton("Clear", color=GREY)
            clear_btn.clicked.connect(self.clear_fit)
            # model row: the picker on the left, the Fit / Clear actions on the right.
            model_row = _inline(self.fit_combo, trailing=fit_btn)
            model_row.layout().addWidget(clear_btn, 0)
            col.addWidget(FluentSettingRow("model", model_row, label_width=proc_lw))
            # args on its OWN full-width row (it needs the room; jamming it next to
            # the combo squeezed both).
            self.fit_cmd = FluentLineEdit("")
            self.fit_cmd.setPlaceholderText("p0=[1,0,1,0], B=0.1, is_fit=False")
            self.fit_cmd.setStyleSheet(self.fit_cmd.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
            self.fit_cmd.setToolTip(
                "Optional fit arguments injected into the call (trusted local code):\n"
                "p0=[...] initial guess; NAME=value fixes a named parameter; is_fit=False just overlays p0.")
            self.fit_cmd.returnPressed.connect(self.do_fit)
            col.addWidget(FluentSettingRow("args", self.fit_cmd, label_width=proc_lw))

            # manual x/y limits (DataFigure.xlim/ylim).  Together with the Display section
            # above (colormap / relim / unit) the Edit tab now covers the whole DataFigure
            # surface -- the same knobs as Setting plus the per-axis detail Setting omits.
            section("Limits")
            self.xmin = FluentLineEdit(""); self.xmax = FluentLineEdit("")
            self.ymin = FluentLineEdit(""); self.ymax = FluentLineEdit("")
            for w in (self.xmin, self.xmax, self.ymin, self.ymax):
                w.setFixedWidth(scaled_px(88, minimum=68))   # wide enough not to clip "-0.4960"
                w.returnPressed.connect(self.apply_limits)
            apply_btn = FluentButton("Apply lim", color=ACCENT)
            apply_btn.clicked.connect(self.apply_limits)
            col.addWidget(FluentSettingRow("x range", _inline(self.xmin, self.xmax), label_width=proc_lw))
            col.addWidget(FluentSettingRow("y range", _inline(self.ymin, self.ymax, trailing=apply_btn), label_width=proc_lw))

            # ---- Command: a one-line REPL on the panel's DataFigure (confocal runs the same
            # `data_figure.<fn>(...)` form).  Type e.g. `data_figure.xlim(0, 10)`,
            # `df.lorentzian(p0=[1,0,1,0])`, `ax.set_title('x')` -- it runs on the snapshot and
            # the result/exception shows below.  Trusted-local-tool posture (same as the Scan
            # tab / fit args): only run code you wrote.
            section("Command")
            self.cmd_input = FluentLineEdit("")
            self.cmd_input.setPlaceholderText("data_figure.xlim(0, 10)")
            self.cmd_input.setStyleSheet(self.cmd_input.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
            self.cmd_input.setToolTip(
                "Run a line of Python on this panel's figure.  Names: data_figure / df, fig, ax,\n"
                "plotter, np.  e.g. df.lorentzian(p0=[1,0,1,0]) ; ax.set_title('x') ; data_figure.xlim(0,10)")
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
        # produced -- not the last timer-tick render (which can lag by a beat).
        self.console.refresh_once()
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
                new_plotter = panel_plot(np.array(src.values, dtype=float), kind="hist", size=size,
                                         bins=int(card.config.params.get("bins", 60)),
                                         fit=str(card.config.params.get("fit", "double")),
                                         ylog=bool(card.config.params.get("ylog", False)), **view,
                                         labels=tuple(src.labels), title=title)
            else:  # 1d / monitor -> a line snapshot (monitor honours its show_dist toggle)
                extra = {"show_dist": bool(card.config.params.get("show_dist", True))} if kind == "monitor" else {}
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
        self.fill_limits()
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
        writeback = None
        # A plot is a pure view: the only writeback a plot Edit does is a 2D
        # region select back to its upstream camera node's ROI (a 1D scan-range
        # writeback belongs to the measurement's OWN Logic-tab Edit, not here).
        if kind == "2d" and self._node is not None:
            writeback = self._read_region
        if writeback is not None:
            if area is not None:
                area.callback = writeback
            if zoom is not None:
                zoom.callback = writeback
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
        from .data_figure import DataFigure
        if self._df is None:
            self._df = DataFigure(self._plotter)
        return self._df

    def do_fit(self) -> None:
        if self._plotter is None or self.fit_combo is None:
            return
        method = FIT_MODELS.get(self.fit_combo.currentText())
        if method is None:
            return
        cmd = self.fit_cmd.text().strip()
        try:
            df = self._df_for()
            if df.fit is not None:
                df.clear()
            # trusted-local-tool posture: the typed args are injected into the
            # fit call -- p0 / fixed-params / is_fit.
            call = f"_df.{method}({cmd})" if cmd else f"_df.{method}(is_display=True)"
            result, _ = eval(call, {"_df": df, "np": np})  # noqa: S307 - local experiment tool
            self._canvas.draw_idle()
            popt = getattr(result, "popt", None)
            names = getattr(result, "names", None) or []
            if popt is None:
                self.status.setText(f"fit {self.fit_combo.currentText()}: did not converge")
                return
            self.status.setText(f"fit {self.fit_combo.currentText()}: "
                                + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, popt)))
        except Exception as exc:
            self.status.setText(f"fit failed: {str(exc).splitlines()[0][:140]}")

    def clear_fit(self) -> None:
        if self._df is not None:
            try:
                self._df.clear()
            except Exception:
                pass
            self._df = None
            if self._canvas is not None:
                self._canvas.draw_idle()
            self.status.setText("fit cleared")

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
        self.fill_limits()
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

    def fill_limits(self) -> None:
        if self._plotter is None or self.xmin is None:
            return                          # no Limits section (a non-"plot" role)
        ax = getattr(self._plotter, "ax", None)
        if ax is None:
            return
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        with _signals_blocked(self.xmin, self.xmax, self.ymin, self.ymax):
            self.xmin.setText(f"{xlo:.6g}"); self.xmax.setText(f"{xhi:.6g}")
            self.ymin.setText(f"{ylo:.6g}"); self.ymax.setText(f"{yhi:.6g}")

    def apply_limits(self) -> None:
        if self._plotter is None:
            return
        try:
            df = self._df_for()
            df.xlim(float(self.xmin.text()), float(self.xmax.text()))
            df.ylim(float(self.ymin.text()), float(self.ymax.text()))
            self._canvas.draw_idle()
            self.status.setText("limits applied")
        except Exception as exc:
            self.status.setText(f"bad limits: {str(exc).splitlines()[0][:100]}")

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
        """The panel's DISPLAY knobs, read from ``config.params`` (the one source), folded into the
        saved ``info['view']`` so ``load_figure(...).plot()`` reopens the figure AS SEEN: relim mode
        + fixed lo/hi (``card._relim`` / the fixed bounds), unit index, colormap and repeat mode.
        Everything defaults, so a panel saved before a knob was touched still round-trips cleanly.

        ``cmap`` stores the RESOLVED colormap name -- the actual colormap the panel drew with -- via the
        ONE ``_resolved_cmap`` resolver (operator's pick, else the kind's declared ``PANEL_PARAMS``
        default: ``gray`` for a site map, ``inferno`` for a 2-D image).  So the npz records a REAL name an
        image/site-map save drew with, never an empty ``''`` a consumer would have to second-guess; kinds
        with no colormap param (1-D / hist / monitor) resolve to ``''`` -- they never feed it to
        matplotlib."""
        params = self.card.config.params
        return {
            "relim": self.card._relim(),
            "fixed_lo": _safe_float(str(params.get("fixed_lo", 0.0)), 0.0),
            "fixed_hi": _safe_float(str(params.get("fixed_hi", 1.0)), 1.0),
            "unit_index": int(params.get("unit_index", 0) or 0),
            "cmap": _resolved_cmap(str(self.card.config.kind or ""), params),
            "repeat_mode": self.card._repeat_mode_value(),
        }

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
        # ``col``/``row`` ARE the card's pixel top-left (gravity-packed by _compact); place verbatim
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
        super().__init__(parent, shadow=True)
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
        self.form = MeasurementPanel([spec] if spec is not None else [], single=True,
                                     signals_provider=getattr(console, "_signal_names", None),
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
        self._task_mid_key = "frame"           # which output-buffer key the task panel shows
        self._task_card_frame = None           # last mid-run frame (kept frozen after the task ends)
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
        margin = scaled_px(14)
        # EMBEDDED: no top/bottom inset -- the host (the figure viewer) already frames the console with
        # its own root margin, so the console's FIRST card (the header) sits flush at the pane top and its
        # visible top/bottom edges line up with the Info card beside it.  Standalone keeps the 8 px inset
        # off the window chrome.
        v_margin = 0 if self.embedded else scaled_px(8)
        # The tab card is the LAST child of this root layout, and its soft drop shadow bleeds
        # ``fluent_tab_shadow_margin`` px BELOW its geometry.  Reserve that bleed at the BOTTOM of the
        # console's OWN layout so the shadow is not clipped by the console's bottom edge.  (An EMBEDDED host
        # margin -- the figure viewer's console holder -- sits OUTSIDE the console widget and therefore
        # cannot un-clip a shadow that is already clipped INSIDE the console's tree: this is the real root
        # cause of the "bottom shadow cut off" the outer holder margin never fixed.)  The TOP inset stays
        # ``v_margin`` (0 embedded) so the header still lines up flush with the Info card beside it; the top
        # shadow bleed is topped up separately just above the tab widget (see ``root.addSpacing`` below).
        bottom_margin = max(v_margin, fluent_tab_shadow_margin())
        root.setContentsMargins(margin, v_margin, margin, bottom_margin)
        # A clear GAP separates the three rows -- the header card, the (hidden) task banner and
        # the tab card -- so they read as DISTINCT rounded cards on the grey window background.
        # The header is flat (no drop shadow) and the tab bar draws no top base line, so this
        # gap reads as clean card separation rather than a hard line.
        root.setSpacing(scaled_px(10, minimum=7))

        # FLAT header (no drop shadow): the shadow's soft bottom edge cast a thin grey line
        # into the gap right above the tab strip (the "line above the tabs").  The tab widget
        # below carries its own card, so the header needs no elevation -- it is a plain top bar.
        header_frame = FluentFrame(shadow=False)
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
        # A READ-ONLY status readout (panel/signal/shot counts, or a node-error banner),
        # not an input -- so it is a borderless label, never a bordered line-edit.  As a
        # disabled line-edit its box border drew a long grey line across the header bottom
        # that read as "a thin line above the tabs"; a label has no box.
        self.summary = FluentLabel("")
        self.summary.setStyleSheet(_SUMMARY_STYLE)

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
        if readout is not None and hasattr(readout, "camera_measurement"):
            self.kind_combo.addItem("Measurement: Camera (live frames)", ("camera", "live"))
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
            # The device manager: see every device the config loaded (by role-type) + Scan hardware --
            # the GUI face of na.load_devices/discover_devices, and the SAME device registry the per-
            # measurement Camera dropdowns read.  A launcher (like Selectors/Pause), OUT of the running-
            # task lockout: inspecting devices is read-only.
            self.devices_button = FluentButton("Devices", color=GREY)
            self.devices_button.setToolTip("Open the device manager: every loaded device grouped by "
                                           "role-type (Camera / Sequencer / …) + a Scan-hardware button.")
            self.devices_button.clicked.connect(self._open_device_manager)

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

        # Running-task LOCKOUT banner (confocal-style): a prominent strip shown ONLY
        # while a task runs.  It names the task + progress and offers the ONE allowed
        # action (Stop); every other control is disabled meanwhile (``_set_task_running``).
        # Hidden by default so an idle console looks unchanged.  A plain objectName-scoped
        # QWidget (flat fill, no cascade to children, no FluentFrame self-paint to fight).
        self.task_banner = QtWidgets.QWidget()
        self.task_banner.setObjectName("taskBanner")
        self.task_banner.setStyleSheet(f"#taskBanner {{ background-color: {ORANGE}; }}")
        banner = QtWidgets.QHBoxLayout(self.task_banner)
        banner.setContentsMargins(scaled_px(12), scaled_px(5), scaled_px(12), scaled_px(5))
        banner.setSpacing(scaled_px(8, minimum=4))
        self.task_banner_label = FluentLabel("")
        self.task_banner_label.setStyleSheet(
            "color: white; background: transparent; border: none; font-weight: bold;")
        self.task_stop_button = FluentButton("Stop task", color=RED)
        self.task_stop_button.clicked.connect(self._stop_running_task)
        banner.addWidget(self.task_banner_label, 1)
        banner.addWidget(self.task_stop_button, 0)
        self.task_banner.setVisible(False)
        root.addWidget(self.task_banner)

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
        # The tab card (Monitor / Logic / Edit) carries a soft drop shadow whose blur+offset bleeds
        # ``fluent_tab_shadow_margin`` past its TOP edge.  The generic row spacing above it is smaller than
        # that bleed, so the tab strip's top read as CUT OFF -- most visibly in the EMBEDDED figure viewer,
        # whose top inset is 0.  Top up the gap above the tabs to EXACTLY the shadow bleed (the ONE source
        # shared with the shadow itself and with the Info column's own tab-gap), so the whole top shadow is
        # visible at every display scale.  (Only the shortfall is added, so a standalone console's existing
        # gap is unchanged where it already suffices.)
        root.addSpacing(max(0, fluent_tab_shadow_margin() - root.spacing()))
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
            short_names_provider=self._signal_short_names, live_namespace_provider=self._expression_namespace)

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
        minus the namespace builtins) -- used to map the panel to its node."""
        import ast
        try:
            tree = ast.parse(str(source or ""), mode="exec")
        except SyntaxError:
            return set()
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        return names - {"np", "numpy", "math", "history", "latest", "names", "shot", "value", "signal"}

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
        for node in [*self.running_nodes, *self._last_node.values()]:
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
        """``{signal name: (axis_label, unit)}`` from each RUNNING node's
        ``output_specs`` -- so a plot reads its y-axis label/unit from the producing
        measurement (the SignalSpec it declares), not a hard-coded per-kind string."""
        out: dict[str, tuple[str, str]] = {}
        for node in self.running_nodes:
            if not hasattr(node, "output_specs"):
                continue
            try:
                for spec in node.output_specs():
                    out[str(spec.name)] = (spec.axis_label, spec.unit)
            except Exception:
                continue
        return out

    def _signal_short_names(self) -> dict:
        """``{full hub signal: SHORT name}`` = each RUNNING node's published signals with that node's
        prefix stripped (``temperature_survival`` -> ``survival``, ``frame`` -> ``frame``).  The picker
        nest binds this so the leaf shows the short name -- the SAME rule the Logic tab uses
        (``strip_node_prefix``), never the verbose SignalSpec axis label."""
        out: dict[str, str] = {}
        for node in self.running_nodes:
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
        # 1) the running producing node (ground truth: whatever publishes this signal now)
        for node in self.running_nodes:
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
        for node in self.running_nodes:
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
        for node in self.running_nodes:
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
        for node in self.running_nodes:
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

        rows: list[tuple[str, str, str]] = []
        if getattr(node, "layer", "") == "task":
            buf = getattr(node, "output", None)
            for key in getattr(node, "mid_run", ()):
                if key in ("progress", "stage"):          # progress %/text live on the banner
                    continue
                rows.append((f"{key} (mid-run)", describe_shape(buf.latest(key) if buf else None), desc(key)))
            result = getattr(node, "result", None) or {}
            for key in getattr(node, "provides", ()):
                value = result.get(key) if isinstance(result, dict) else None
                rows.append((f"{key} (result)", describe_shape(value), desc(key)))
            return rows
        # published_signals() are HUB names (incl. the node's disambiguating prefix when two
        # nodes would collide).  Show the SHORT natural name (strip the prefix) -- "rate", not
        # "judge_occupancy_rate" -- because the Logic row is already titled by the node; the
        # short name is also what output_specs (and so ``desc``) is keyed by.
        pfx = str(getattr(node, "prefix", "") or "")
        for full in sorted(node.published_signals()):
            short = strip_node_prefix(full, pfx)             # ONE rule, shared with the picker nest
            try:
                # PER-SIGNAL structure (#H3s-F3): a signal whose own SignalSpec declares a points/data
                # slot (occupied -> 5 × (35); frame_judged -> 5 × (96×128)) shows in CONTRACT form off
                # ITS structure.  Else the node's PRIMARY block (camera frame, scan y) shows in contract
                # form off the node-level triple; any other AUX signal (a static centers (35, 2) /
                # thresholds (35,), a per-trigger frame_i) shows its RAW numpy shape.
                spec = specs.get(full)
                primary = full == pfx + str(getattr(node, "primary_signal", "") or "")
                if spec is not None and getattr(spec, "has_structure", False):
                    ps, ds, gs = spec.points_shape, spec.data_shape, getattr(node, "grid_shape", ())
                elif primary:                               # node-level primary block (camera / scan)
                    ps = getattr(node, "points_shape", ()); ds = getattr(node, "data_shape", ())
                    gs = getattr(node, "grid_shape", ())
                else:                                       # an aux signal with no contract slot -> raw shape
                    ps, ds, gs = None, None, ()
                shape = describe_shape(self.hub.latest(full), points_shape=ps, data_shape=ds, grid_shape=gs)
            except Exception:
                shape = "—"
            rows.append((short, shape, desc(short)))
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
        """The hub-signal PREFIX the row's node will publish under once started -- the SAME prefix the
        build path uses, so a declared name == the published name (a binding made before Start, or
        restored from a saved layout, re-attaches the instant the producer starts, #prebind).
        Measurement: the slug ``f"{spec.key}_"`` (what ScannedMeasurementNode/PulseScanNode get at
        build); processor: the per-instance ``_logic_node_prefix``; camera/task: ``""`` (a camera
        publishes ``frame_i`` bare, a task's mid-run entry is a synthetic display tag, not a hub name)."""
        node = row.node
        if node.kind == "measurement":
            return f"{self._spec_for_logic(node).key}_"
        if node.kind == "processor":
            return self._logic_node_prefix(node)
        return ""

    def _declared_signal_keys(self, row: "LogicNodeRow") -> list[str]:
        """The FULL hub names a STOPPED Logic-tab node WILL publish once started -- IDENTICAL to the
        running node's ``published_signals()`` (same node prefix, #prebind), so the picker offers, and
        a Monitor binding stores, the SAME name the node later emits.  That makes a "connect first,
        start later" binding (and a save->load of one) re-attach automatically when the producer
        publishes that exact name.  A processor publishes its ``result_keys``; a camera one ``frame_i``
        PER emCCD event (``frames_per_cycle``); a measurement its ``x_key``/``y_key`` curve; a task
        streams a (synthetic, off-hub) mid-run key."""
        spec = self._spec_for_logic(row.node)
        keys = list(getattr(spec, "result_keys", ()) or [])
        if not keys and row.node.kind == "measurement":
            keys = [k for k in (getattr(spec, "x_key", ""), getattr(spec, "y_key", "")) if k]
        if not keys and row.node.kind == "camera":
            # ONE frame_i per emCCD event -- the SAME names CameraMeasurement.published_signals() will
            # publish once started (shared helper so they can NEVER drift; NOT the lumped "frame" the old
            # design had, which left a phantom "frame waiting" while the camera emits frame_0/1/2, #residue).
            from Zou_lab_control.neutral_atom.operations.logic import camera_frame_keys
            keys = camera_frame_keys(row.node.values.get("frames_per_cycle", 1))
        if not keys and row.node.kind == "task":          # off-hub one-shot, SYNTHETIC mid-run tag (never a hub signal)
            return [f"{getattr(spec, 'mid_run_key', 'frame')} (mid-run)"]
        pfx = self._declared_node_prefix(row)              # prepend the node prefix -> == published_signals()
        return [f"{pfx}{k}" for k in keys]

    def _node_for_signal(self, name: str):
        """The producing node for signal ``name``: a RUNNING node first, else the last build of a
        Logic-tab node (``_last_node``, kept past Stop) so a stopped node's lingering signal still
        resolves.  None if none publishes it."""
        for node in [*self.running_nodes, *self._last_node.values()]:
            if node is not None and hasattr(node, "published_signals"):
                try:
                    if name in node.published_signals():
                        return node
                except Exception:
                    continue
        return None

    def _signal_structure(self, name: str):
        """The output-contract structure for ONE signal ``name`` from its producing node (#H3o, #H3s-F3):
        ``{"points_shape", "data_shape", "grid_shape", "core_ndim"}`` -- so a plot knows whether the DATA
        is 1-D (multiple series -> lines) or 2-D (an image -> reshape/imshow), what the swept points are,
        the 2-D ``grid_shape`` for reshaping a flattened scan grid, and ``core_ndim`` (= len(points) +
        len(data)) so ``reduce_repeat`` collapses the LEADING repeat axis by STRUCTURE, never an ndim
        guess.  PER-SIGNAL first: a node may publish signals of different structure (occupancy's
        ``(repeat, n_sites)`` vs its static ``centers`` (N, 2)), so when the signal's own
        :class:`SignalSpec` declares a points/data slot it wins; only when it declares none does this
        fall back to the node-level triple (so other nodes are unaffected).  ``None`` when no producing
        node is found (a derived expression / a raw array): the consumer then uses shape heuristics."""
        node = self._node_for_signal(str(name or ""))
        if node is None:
            return None
        node_grid = tuple(int(n) for n in (getattr(node, "grid_shape", ()) or ()))
        # ``grid_shape`` un-flattens a 2-D scan's SWEPT POINTS into a map; whether it applies to THIS
        # signal is the ONE shared rule grid_for_points (operations layer) -- prod(grid)==prod(points) and
        # points non-empty -- the SAME fact describe_shape and _coerce use, so it can never drift.  A
        # per-site DATA signal (occupancy points=()) gets () and so can never be imshow'd as a (5x7)
        # heatmap; the site map stays frame + rings (the trap layout is camera-pixel centres, not a grid).
        from Zou_lab_control.neutral_atom.operations.logic import grid_for_points

        def _grid_for(points) -> tuple:
            return grid_for_points(node_grid, points)

        # PER-SIGNAL: the producing signal's own SignalSpec (occupied declares points=(), data=(n_sites,);
        # centers declares neither -> None) takes precedence over the node-level triple.
        spec = None
        if hasattr(node, "signal_spec"):
            try:
                spec = node.signal_spec(str(name))
            except Exception:
                spec = None
        if spec is not None and getattr(spec, "has_structure", False):
            ps = tuple(spec.points_shape or ())
            ds = tuple(spec.data_shape or ())
            return {"points_shape": ps, "data_shape": ds,
                    "grid_shape": _grid_for(ps),
                    "core_ndim": len(ps) + len(ds), "per_signal": True}
        # fall back to the node-level contract triple (camera / scan / a node with no per-signal spec).
        # ``per_signal`` False -> a consumer keeps the legacy ndim>=3 repeat detection (core_ndim NOT
        # fed to reduce_repeat) so camera / scan paths stay byte-identical.
        ps = tuple(getattr(node, "points_shape", ()) or ())
        ds = tuple(getattr(node, "data_shape", ()) or ())
        return {"points_shape": ps, "data_shape": ds,
                "grid_shape": _grid_for(ps),
                "core_ndim": len(ps) + len(ds), "per_signal": False}

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
        orphans = [n for n in self.hub.names() if n not in providers]
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
                    # occupancy" (same brevity the nested combo uses).
                    pfx = str(getattr(src, "prefix", "") or "")
                    short = name[len(pfx):] if (pfx and name.startswith(pfx) and len(name) > len(pfx)) else name
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
            node.start(rate_hz=getattr(node, "rate_hz", 5.0))
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
        """Snap a JUST-DROPPED card's raw pixel seed to its NEAREST anchor (:func:`_snap_drop_anchor`):
        another card's top-left corner (= displace that card) or the closest free lattice point.
        Connected to ``PanelCard.dropped`` -- the drag-release path ONLY, so resize / reflow / Add
        placement stay snap-free; the ``layout_changed`` the card emits right after re-packs as usual
        (the card is ``active`` in :func:`_compact`, so it wins the anchor it snapped onto)."""
        if card not in self.cards:
            return
        others = [c.config for c in self.cards if c is not card]
        board_w = self._pack_width([c.config for c in self.cards])
        card.config.col, card.config.row = _snap_drop_anchor(card.config, others, board_w)

    def _arrange(self) -> None:
        # Top-left gravity pack (see _compact): every card floats UP then LEFT until blocked, with a
        # uniform GAP on all four sides.  The just-dragged / resized card is the ``active`` one, so it
        # wins a contested slot and the others reflow.
        active = self.sender()
        active_cfg = active.config if isinstance(active, PanelCard) and active in self.cards else None
        # DEMAND-driven pack width: the viewport width, GROWN only far enough to hold the card that
        # currently reaches furthest right (its just-dropped column pixel + width).  So a board whose
        # cards all fit the viewport packs at the viewport width (one column when narrow, no sideways
        # scroll); the moment a card is dragged into a column past the viewport, the board grows to fit
        # it -- gravity then keeps it there (it no longer wraps back) and the scroll area shows a
        # horizontal scrollbar.  Move that card back and the reach shrinks, so the board returns to the
        # viewport width and the scrollbar disappears.
        board_w = self._pack_width([c.config for c in self.cards])
        if _compact([c.config for c in self.cards], active_cfg, board_w=board_w):
            self._mark_dirty()
        self.board.arrange(self.cards)
        self._update_summary()

    def _pack_width(self, configs: Sequence["PanelConfig"]) -> int:
        """The width the gravity packer wraps at: the live scroll-viewport width GROWN to the furthest
        right edge any card currently reaches (``col + width + GAP``).  Cards that all fit the viewport
        pack at the viewport width (no sideways scroll); a card dropped into a column past the viewport
        grows the board to hold it (so column 2 stays reachable + a horizontal scrollbar appears), and
        removing/moving it back shrinks the board to the viewport again -- expansion strictly on demand."""
        vw = self.scroll.viewport().width() if hasattr(self, "scroll") else 0
        reach = max((c.col + _card_size(c.size)[0] + GAP for c in configs), default=0)
        return max(vw, reach)

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
            title = "Camera (live frames)" if kind == "camera" else str(name)
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
        # SPAWN the new card at the first free TOP-LEFT slot (tile into the board: fill the top row
        # then wrap), NOT a fresh column -- the local NW gravity in _compact then keeps it there.  Seed
        # against the live VIEWPORT width so Add only ever tiles WITHIN the visible pane (it never
        # force-grows the board into an off-screen column -- that expansion is on-demand via a DRAG, see
        # _pack_width); a fallback width covers the pre-show case where the viewport is still 0.
        cfgs = [c.config for c in self.cards]
        vw = self.scroll.viewport().width() if hasattr(self, "scroll") else 0
        board_w = max(vw, _min_board_width(cfgs + [config])) if vw else _board_width(cfgs + [config])
        config.col, config.row = _first_free_slot(config, cfgs, board_w)
        card = self._new_panel_card(config)
        self._attach_card(card)
        self._arrange()
        self._mark_dirty()

    def _remove_panel(self, card: PanelCard) -> None:
        # blocked while a task owns the console (the lock is released BEFORE
        # _clear_task_running drops the transient task panel, so this never blocks that).
        if self._task_locked:
            return
        if card in self.cards:
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

    def _logic_node_prefix(self, node: LogicNodeConfig) -> str:
        """The hub-signal prefix for a logic node -- EMPTY by default, so a node publishes its
        SHORT natural signal names (``occupied`` / ``rate`` / ``frame`` ...), NOT a verbose
        ``judge_occupancy_rate``.  The producing node is shown by the signal-flow grouping +
        the frame-title legend, so the producer name need not be baked into every signal.

        Only a SECOND node whose OWN output keys would COLLIDE with signals an already-running
        node publishes gets a disambiguating slug prefix (``occupancy_2_occupied``), so the two
        never overwrite each other on the hub; the common single-instance case stays short."""
        spec = self._spec_for_logic(node)
        keys = {str(k) for k in (getattr(spec, "result_keys", ()) or ())}
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
        for prev in (getattr(self, "_last_node", {}) or {}).values():
            if prev is not None and str(getattr(prev, "instance_label", "")) == str(node.title):
                try:
                    running.difference_update(str(s) for s in prev.published_signals())
                except Exception:
                    pass
        if not keys or not (keys & running):
            return ""                                # no collision (incl. own restart) -> short names
        from Zou_lab_control.neutral_atom.operations.measurement import measurement_slug
        base = measurement_slug(node.title or node.name) or str(node.kind) or "node"
        prefix, k = f"{base}_", 2
        while prefix in {getattr(n, "prefix", "") for n in self.running_nodes} \
                or any((prefix + key) in running for key in keys):
            prefix = f"{base}_{k}_"
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
        register it in ``self.running_nodes``, and ``node.start(rate_hz=...)``.  Sets
        the node's status dot green; on build/run error -> red + the error on the status
        line.  Reuses the SAME node-build paths the real readout / notebook use."""
        if self._task_locked:
            return                                 # a task owns the console -- no other Start
        editor = self._logic_editors.get(id(row))
        try:
            values = editor.collect_values() if editor is not None else dict(row.node.values)
        except Exception:
            values = dict(row.node.values)
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
        # stop a previous run of THIS node so running nodes don't pile up
        self._stop_logic_node(row, _silent=True)
        # Starting ANY device-driving node (camera / measurement / task) first STOPS every
        # OTHER running device-driver, so two never fight over the shared camera + pulse
        # streamer (which deadlocks real hardware).  Reactive processors (judge-occupancy)
        # read only hub signals -- they touch no device, so they keep running (#6).
        if row.node.kind in DEVICE_DRIVING_KINDS:
            for other in list(self.logic_nodes):
                if other is row or self._logic_nodes.get(id(other)) is None:
                    continue
                if other.node.kind in DEVICE_DRIVING_KINDS:
                    self._stop_logic_node(other)
        try:
            node = self._build_logic_node(row.node, values)
        except Exception as exc:
            row.set_state("error", status=f"build failed: {str(exc).splitlines()[0][:80]}")
            if editor is not None:
                editor.set_status(f"build failed: {str(exc).splitlines()[0][:140]}", error=True)
            return
        row.node.values = dict(values)            # remember for the next Edit reopen + save
        # Label the built node with its ROW TITLE so its provider label MATCHES the declared row's
        # (the camera has prefix="" -> display_label would otherwise fall back to node_label="camera",
        # which differs from the row title "Camera (live frames)", listing `frame` under TWO sources
        # = the "two cameras" bug).  One label per node => one entry in the signal picker (#H3n).
        node.instance_label = str(getattr(row.node, "title", "") or getattr(node, "instance_label", ""))
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
        if not self._timer.isActive():
            self._timer.start()
        try:
            node.start(rate_hz=getattr(node, "rate_hz", 5.0))
        except Exception as exc:
            row.set_state("error", status=f"start failed: {str(exc).splitlines()[0][:80]}")
            if editor is not None:
                editor.set_running(False)
                editor.set_status(f"start failed: {str(exc).splitlines()[0][:140]}", error=True)
            return
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

    def _set_task_running(self, row: "LogicNodeRow", node) -> None:
        """Engage task-run mode: open a dedicated Monitor panel that shows the task's
        mid-run output (read off its OWN buffer, NOT the hub -- #6), and LOCK all other
        controls so the only actions are Stop / wait (#5, confocal task semantics)."""
        spec = self._spec_for_logic(row.node)
        self._task_mid_key = str(getattr(spec, "mid_run_key", "frame"))
        self._task_card_frame = None
        kind = str(getattr(spec, "default_kind", "2d") or "2d")
        title = f"Task: {row.node.title}"
        # The dedicated panel reads the reserved ``__task_frame__`` injected each tick
        # from the task's output buffer (see _tick) -- never a hub signal.
        config = PanelConfig(kind=kind, title=title, size="2x2",
                             source="value = __task_frame__")
        card = self._new_panel_card(config)
        self._attach_card(card)
        self._task_card = card
        self._running_task_row = row
        self._arrange()
        self._apply_task_lock(True)
        self._update_task_banner(node)

    def _update_task_banner(self, node) -> None:
        row = self._running_task_row
        if row is None:
            return
        pct = int(round(float(getattr(node.output, "progress", 0.0)) * 100))
        # the task's current STAGE (e.g. "reference frame 23/30", "fitting per-site
        # thresholds") so the operator sees what step the calibration is on, not just %.
        stage = node.output.latest("stage")
        stage_txt = f"  —  {stage}" if stage else ""
        self.task_banner_label.setText(f"⏳  Task running: {row.node.title}  —  {pct}%{stage_txt}  "
                                       "(all other controls locked until it finishes / you Stop it)")

    def _apply_task_lock(self, locked: bool) -> None:
        """Disable / re-enable every mutating control while a task runs.  The banner's
        Stop button stays enabled (it lives outside the locked groups)."""
        self._task_locked = bool(locked)
        self.task_banner.setVisible(bool(locked))
        for widget in self._lockable_header:
            widget.setEnabled(not locked)

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

    def _refresh_task_panel(self) -> None:
        """Pump the running task's mid-run output (from its OWN buffer) into the
        dedicated panel + banner each tick, and leave task-run mode once it finishes."""
        if self._task_card is None:
            return
        row = self._running_task_row
        node = self._logic_nodes.get(id(row)) if row is not None else None
        if node is None:
            return
        frame = node.output.latest(self._task_mid_key)
        if frame is not None:
            self._task_card_frame = frame
        self._update_task_banner(node)
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
            repeat = self._repeat_value(values)                     # pops "repeat" off ``values``
            return spec.build(self.hub, repeat=repeat, **values)
        spec = self._spec_for_logic(node)
        if spec is None:
            raise RuntimeError(f"no catalog spec named {node.name!r} for a {kind} node")
        if kind == "measurement":
            # The SPEC owns the assembly (its ProcessorSpec.make_node counterpart): ask it for the live
            # node.  ``repeat`` (re-run the whole scan N times, 0 = ∞) is the MEASUREMENT knob -- pop it
            # and hand the count to make_node; the node only FILLS its raw ``(ring, points, dim)`` block,
            # HOW the repeats are displayed is the PLOT's ``repeat_mode`` (#H3l).  WHICH concrete node
            # (decoupled PulseScanNode vs frame-reducing ScannedMeasurementNode) is the spec's scan TIER,
            # not a string the GUI parses; the node publishes under the measurement's slug (spec.key) so
            # every signal is ``<slug>_<quantity>`` (e.g. temperature_t_off) -- one name, derived.
            repeat = self._repeat_value(values)
            return spec.make_node(self.hub, prefix=f"{spec.key}_", repeat=repeat, **values)
        if kind == "processor":
            # REACTIVE processor (the "func" layer): a live node consuming hub signals
            # and republishing derived ones -- e.g. judging occupancy from each frame.
            if getattr(spec, "reactive", False):
                # Per-INSTANCE prefix + label so multiple same-kind processors (two occupancy
                # judges) publish DISTINCT signals and are told apart everywhere (#2).
                built = spec.make_node(self.hub, prefix=self._logic_node_prefix(node), **values)
                built.instance_label = node.title
                return built
            # ONE-SHOT processor: runs once over saved/grabbed frames and self-stops.
            from Zou_lab_control.neutral_atom.operations.logic import ProcessorRun
            camera = next((getattr(n, "camera", None) for n in self.running_nodes
                           if getattr(n, "camera", None) is not None), None)
            sequencer = next((getattr(n, "sequencer", None) for n in self.running_nodes
                              if getattr(n, "sequencer", None) is not None), None)
            readout = getattr(self.session, "readout", None)
            return ProcessorRun(self.hub, spec, readout=readout, camera=camera,
                                 sequencer=sequencer, params=values)
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
                # (repeat, points, dim) block per its repeat_mode -- the console no longer pushes a
                # repeat counter onto plotters, since the panel is decoupled and owns the reduction.)
            # A one-shot TASK that ENDS on its own -- finishing normally OR raising (which
            # sets finished=True in a finally) -- must release the console lockout here, the
            # SAME path the Stop button takes.  Finish == error == Stop: otherwise a task that
            # completes (or fails) leaves the dashboard locked forever (only a manual Stop
            # would reach _clear_task_running).
            if terminated and row is self._running_task_row:
                self._clear_task_running()                        # release lock + remove the task panel (#C)

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

    def _expression_namespace(self) -> dict[str, object]:
        namespace: dict[str, object] = {"np": np, "numpy": np, "math": _math}
        # Shot-COHERENT read at the board's global display shot (#shot-clock): every signal resolves to its
        # value AT that shot, so a frame_0 2-D, a frame_2 2-D and a frame_1->occupancy sitemap can never show
        # different shots -- the faster camera is held back to the slower co-displayed producer.  A signal
        # with no sample at that shot (a free-running scalar, or a producer not yet there) falls back to its
        # latest, never blanking a panel.  display_shot None -> latest of each (snapshot_latest).  A LONE
        # fast panel is NOT held back: its own signal is the min (see _display_shot).  (Task mid-run output
        # is intentionally absent here: it is off-hub via __task_frame__.)
        namespace.update(self.hub.snapshot_at(self._display_shot()))
        namespace["history"] = self.hub.history
        namespace["latest"] = self.hub.latest
        namespace["names"] = self.hub.names
        namespace["shot"] = self.hub.shot                # the hub's latest publish counter (for rolling monitors)
        # Per-signal publish counters (reserved key) so a rolling monitor can tell
        # a new sample of its own source from an unrelated node's version bump.
        namespace["__sig_versions__"] = self.hub.signal_versions()
        # Coordinate frames (reserved key): {signal_name: [x, w, y, h]} from any
        # node whose acquisition source declares a ROI.  A 2D panel reads its
        # source signal's frame so the image axes are the REAL camera pixel
        # coordinates (ROI), not 0..N -- and an area-select maps back to the ROI.
        namespace["__coord_frames__"] = self._coord_frames()
        # Reserved key for the running task's dedicated mid-run panel: the latest frame
        # from the task's OWN output buffer (NOT the hub -- #6), kept frozen after the
        # task ends until its transient panel is dropped.
        namespace["__task_frame__"] = self._task_card_frame
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
        # _render_version), and Resume redraws them all -- hub.version has moved on, so the version gate fires.
        if not self._paused:
            self._tick()                          # immediate catch-up redraw of every panel

    def _toggle_selectors(self, on: bool) -> None:
        """Header "Selectors" switch: arm (ON) or park (OFF) the selector layer of EVERY dashboard
        panel in place (``PanelCard.set_selectors_enabled`` -> ``BaseLivePlot.set_selectors_active``)
        -- a pure display gate, no rebuild, no effect on acquisition or the Edit tabs."""
        for card in self.cards:
            card.set_selectors_enabled(bool(on))

    def _open_device_manager(self) -> None:
        """Header "Devices" button: open the device-manager window (every loaded device by role-type
        + Scan hardware).  Routes through the session's ONE-per-session ``device_manager()`` facade
        (``na._gui.open_device_manager`` -> ``show_device_manager``) so it reuses the same singleton
        the notebook opens -- the console never builds the window itself."""
        session = getattr(self, "session", None)
        opener = getattr(session, "device_manager", None)
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

    def _tick(self) -> None:
        # poll the logic nodes EVERY base tick (even when no new signal arrived) so a
        # one-shot node's run-complete / error transition is never missed if the node
        # self-stops between version bumps.
        self._poll_logic_nodes()
        self._refresh_signal_info()   # cheap + self-guarded: tracks source/node changes
        # A running task's mid-run output is OFF the hub (#6), so it does NOT bump the
        # hub version -- refresh its dedicated panel here every tick.
        self._refresh_task_panel()
        # PAUSE = freeze EVERY plot's display: skip the per-card render below (so no card advances
        # its _render_version) while keeping node polling / banners alive.  On Resume, hub.version
        # has moved on but the cards' _render_version did not, so the version gate redraws them all.
        if self._paused:
            self._update_summary()
            return
        self._tick_count += 1
        version = self.hub.version
        elapsed = self._tick_count * self._base_interval_ms
        namespace = None
        for card in self.cards:
            # this panel's beat?  update_ms is a multiple of the base, so cards that share a beat fire on the
            # SAME tick -> they read the SAME namespace below (one snapshot, mutually consistent).
            if elapsed % card.config.update_ms != 0:
                continue
            if version == card._render_version:   # no new publish since this panel last drew -> nothing to do
                continue
            if namespace is None:                 # built ONCE per tick -> one snapshot (each signal latest)
                namespace = self._expression_namespace()
            card.refresh(namespace)
            card._render_version = version
        # keep the visible Edit tab's 'now:' acquisition references live, so a queued
        # parameter edit shows as applied once the loop picks it up.
        editor = self.tabs.currentWidget()
        if isinstance(editor, PanelEditor):
            editor.refresh_node_now_labels()
        self._update_summary()

    def refresh_once(self) -> None:
        """Synchronous FULL refresh (tests / notebooks): render every card now, regardless of
        its per-panel beat or the version gate."""
        self._poll_logic_nodes()
        self._refresh_signal_info()
        self._refresh_task_panel()
        version = self.hub.version
        # render EVERY card now at each signal's latest value (tests / notebooks / Resume catch-up).
        namespace = self._expression_namespace()
        for card in self.cards:
            card.refresh(namespace)
            card._render_version = version
        editor = self.tabs.currentWidget()
        if isinstance(editor, PanelEditor):
            editor.refresh_node_now_labels()
        self._update_summary()

    def _update_summary(self) -> None:
        try:
            n_signals = len(self.hub.names())
        except Exception:
            n_signals = 0
        # A wedged node must not fail silently: if any running node recorded an error,
        # raise a red banner naming it instead of just showing frozen data.
        faulted = [n for n in self.running_nodes if getattr(n, "last_error", None)]
        if faulted:
            node = faulted[0]
            who = (getattr(node, "prefix", "") or self._node_label(node) or "node").rstrip("_:")
            n = int(getattr(node, "consecutive_errors", 1))
            self.summary.setStyleSheet(_SUMMARY_STYLE_DANGER)
            self.summary.setText(f"⚠ NODE ERROR ({who}, ×{n}): {node.last_error}"[:200])
        else:
            self.summary.setStyleSheet(_SUMMARY_STYLE)
            # No single global refresh rate any more -- each panel sets its OWN update interval
            # (see UPDATE_INTERVALS), so the header no longer claims one "every N ms".
            self.summary.setText(
                f"{len(self.cards)} panels | {n_signals} signals | shot {self.hub.shot}")

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
            self.summary.setText(str(text))
            return
        QtWidgets.QMessageBox.information(self, "Task console", str(text))  # pragma: no cover

    # ------------------------------------------------------------------ teardown
    def stop_all_nodes(self) -> None:
        """Stop every running node's owner thread (so the camera / sequencer are released) but
        KEEP the panels + editors intact.  This is the notebook close = "hide" path: closing the
        window halts all running processes, yet the layout is preserved so reopening the
        session-bound console (``exp.task_console()``) restores the SAME interface rather than a
        blank new one.  Distinct from :meth:`shutdown`, which also tears the cards/editors down
        (the standalone-window / explicit-teardown path)."""
        for node in list(self.running_nodes):
            try:
                node.stop()
            except Exception:
                pass
        try:
            self._tick()   # one refresh so the stopped nodes' rows repaint with grey dots
        except Exception:
            pass

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

    app = ensure_qt_app()
    if state is None and task is not None:
        state = resolve_task_state(task)
    console = TaskConsole(hub=hub, state=state, running_nodes=running_nodes, measurements=measurements,
                          processors=processors, tasks=tasks, session=session, scale=scale,
                          window_ratio=window_ratio)
    console._on_close = on_close
    # A passed-in node should stream the moment the window opens -- so the Monitor
    # is live without the caller having to remember node.start().  start() is
    # idempotent, so a node the caller already started (e.g. bring-up's
    # readout.start(rate_hz=4)) keeps its own rate; only NON-running nodes
    # are launched here.  (TaskConsole.__init__ deliberately does NOT do this, so
    # tests/notebooks keep deterministic manual stepping.)
    for node in running_nodes:
        if not getattr(node, "running", False) and hasattr(node, "start"):
            node.start(rate_hz=getattr(node, "rate_hz", 5.0))
    window = FluentWindow(widget=console, title=title, hide_on_close=hide_on_close)
    # Closing the window must stop the node owner threads (else they keep running, blocked in
    # camera.acquire holding the camera / RPyC link, wedging the kernel).  The console is a CHILD
    # of the window so its own closeEvent never fires on a window close -- we wire the window's
    # signals instead.  Minimising NEVER stops the nodes (only the X / a genuine close does).
    if hide_on_close:
        # Session-bound (notebook) console: the X HIDES the window (keeps the panel layout so a
        # later exp.task_console() restores the SAME interface) and stops every running node so
        # the devices are released.  close_requested fires on the X, not on minimize.
        window.close_requested.connect(console.stop_all_nodes)
    else:
        # Standalone window (.bat / explicit): the X fully tears the console down.
        window.closed.connect(console.shutdown)
    window.adjustSize()
    window.setFixedSize(window.size())
    center_window_on_primary_screen(window, app)   # open centred, exactly like show_pulse_gui (consistency)
    window.show()
    if not hasattr(app, "_zlc_task_windows"):
        app._zlc_task_windows = []
    app._zlc_task_windows.append(window)
    return console


__all__ = [
    "LogicNodeConfig",
    "PanelConfig",
    "TaskConsole",
    "TaskConsoleState",
    "default_console_state",
    "show_task_console",
]
