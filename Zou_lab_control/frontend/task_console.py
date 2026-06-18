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

    value = frame                       # show the latest camera frame
    value = rate_grid - b_rate_grid     # arbitrary math across signals
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

from PyQt5 import QtCore, QtWidgets

from .live import (
    PANEL_SIZES,
    panel_display_size,
    panel_plot,
    panel_size_cells,
    site_ring_radius,
)
from .qt_fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    TEXT,
    YELLOW,
    FluentButton,
    FluentCheckBox,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPopup,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusDot,
    FluentTabWidget,
    FluentWindow,
    add_fluent_shadow,
    ensure_qt_app,
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


TASK_FILES_ENV = "ZLC_TASK_DIR"

PANEL_KINDS: dict[str, str] = {
    "2d": "2D image",
    "sites": "Site map",
    "1d": "1D vector",
    "monitor": "Rolling trace",
    "monitor_nodist": "Rolling trace (no dist)",
    "hist": "Distribution",
}

# What signal SHAPE each plot kind expects as its ``value`` -- shown in the panel's
# Setting so it is clear which signals fit (e.g. a Site map wants a per-site vector,
# a 2D image wants a frame).  Single source for the panel's input self-documentation.
PANEL_INPUT_FORMAT: dict[str, str] = {
    "2d": "a 2D array / camera frame (H×W)",
    "sites": "a per-site (N,) occupancy vector (+ a centers signal)",
    "1d": "a 1D vector (N,) or per-site array",
    "monitor": "a scalar per shot (rolling trace)",
    "monitor_nodist": "a scalar per shot (rolling trace)",
    "hist": "a 1D sample vector",
}

# How many SIGNAL INPUT SLOTS each plot kind has, and what each slot is.  A plot is a
# GENERIC multi-input view: it reads ``signal[0]``, ``signal[1]``, ... (a list) in its
# source expression, one hub signal picked per slot in the Setting.  Most kinds take ONE
# slot (the value); the site map takes THREE (occupancy + centers + image).  A future
# plot type with more inputs is just a longer slot list -- no per-kind params needed.
# Each slot = (label, default-signal-name, tooltip).
_DEFAULT_SLOTS = (("signal", "", "the hub signal to plot"),)
PANEL_INPUT_SLOTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "sites": (
        ("occupancy", "occupied", "per-site (N,) occupancy vector -- colours the rings (signal[0])"),
        ("centers", "centers", "(N, 2) site centres in camera px (signal[1])"),
        ("image", "frame", "camera-frame underlay, blank = none (signal[2])"),
    ),
}


def panel_input_slots(kind: str) -> tuple[tuple[str, str, str], ...]:
    """The input slots for a plot kind -- ``[(label, default_signal, tooltip)]``.  The
    SINGLE source of how many signals a plot consumes and what each means."""
    return PANEL_INPUT_SLOTS.get(str(kind), _DEFAULT_SLOTS)

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


class ParamSpec:
    """One declarative panel parameter: the Setting popup generates its widget
    from this spec (confocal style -- adding a parameter is ONE line here, the
    GUI, the JSON params and the plot rebuild all follow)."""

    def __init__(self, key: str, label: str, kind: str, default, *,
                 choices: Sequence[str] = (), lo: float = 0, hi: float = 1e9, tooltip: str = "",
                 display: bool = False):
        self.key = str(key)
        self.label = str(label)
        self.kind = str(kind)              # "choice" | "int" | "text"
        self.default = default
        self.choices = tuple(choices)
        self.lo, self.hi = lo, hi
        self.tooltip = str(tooltip)
        # display=True  -> a BASIC display knob, rendered in the Setting popup
        #                  (e.g. the colormap / colorset chooser).
        # display=False -> a FUNCTIONAL plot-API parameter, rendered as a GUI
        #                  control in the panel's Edit tab (length / bins /
        #                  centers / image), so Setting and Edit never duplicate.
        self.display = bool(display)

    def get(self, params: Mapping[str, object]):
        return params.get(self.key, self.default)


PANEL_PARAMS: dict[str, tuple[ParamSpec, ...]] = {
    "2d": (
        ParamSpec("cmap", "colormap", "choice", "inferno", choices=CMAPS, tooltip="Image colormap", display=True),
    ),
    "sites": (
        # A site map is a binary occupancy OVERLAY (faint ring = empty, bold ring =
        # occupied) on the camera FRAME.  The colormap applies to that frame underlay
        # (its counts colorbar); the rings carry no scale.  Its THREE signal inputs
        # (occupancy / centers / image) are SLOTS (PANEL_INPUT_SLOTS), picked in the
        # Setting's Source section as signal[0..2] -- NOT params here.
        ParamSpec("cmap", "colormap", "choice", "gray", choices=CMAPS,
                  tooltip="Colormap for the camera-frame underlay", display=True),
    ),
    # The colormap / colorset chooser is the only per-kind knob that stays in the
    # Setting popup (display=True); the FUNCTIONAL params below (length / bins /
    # centers / image) are rendered in each panel's Edit tab instead, so the two
    # surfaces never duplicate.
    "1d": (),
    "monitor": (
        ParamSpec("length", "history", "int", 300, lo=20, hi=10_000,
                  tooltip="Rolling history length (shots kept on screen)"),
    ),
    "monitor_nodist": (
        ParamSpec("length", "history", "int", 300, lo=20, hi=10_000,
                  tooltip="Rolling history length (shots kept on screen)"),
    ),
    "hist": (
        ParamSpec("bins", "bins", "int", 60, lo=5, hi=500, tooltip="Histogram bins"),
    ),
}

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

# Card geometry (raw px, matching the raw-px canvas).  The card is a TITLED frame
# (title strip = the panel kind, top-left; Setting button top-right) whose FOOTER
# absorbs the modulus difference: the figure margins do not halve with the data
# area, so the canvases alone can never tile -- the frame makes the CARDS tile
# instead.  A card spans rows x _pitch_y - gap, so a 2-row card is EXACTLY two
# 1-row cards plus the gap, and dragged cards snap onto one shared grid.
_CARD_PAD = 10        # horizontal padding canvas<->frame, and the drag border
_CARD_TITLE_H = 32    # the FluentGroupBox title strip (kind title top-left + Setting top-right)
_CARD_VPAD = 12       # vertical padding above + below the content
_FOOTER_MIN = 26      # the status + signal-legend line every card carries under its canvas
_GRID_GAP = 8


def _grid_pitch() -> tuple[int, int]:
    """(horizontal, vertical) snap pitch: one layout slot.

    Vertically a slot is the FULL 1-row card (canvas + title strip + pads +
    minimum footer) plus the gap -- taller cards stretch their footer to span
    whole slots, so mixed sizes tile exactly.  Horizontally a slot is half a
    2-column card plus the gap; 4-column cards span four slots and centre
    their canvas in the extra frame width."""

    half_w, half_h = panel_display_size("1x2")
    pitch_x = (half_w + 2 * _CARD_PAD + _GRID_GAP) // 2
    pitch_y = half_h + _CARD_TITLE_H + _CARD_VPAD + _FOOTER_MIN + _GRID_GAP
    return pitch_x, pitch_y


def _card_size(size: str) -> tuple[int, int]:
    """Outer card size for a panel size preset: whole slots minus the gap."""

    rows, cols = panel_size_cells(size)
    pitch_x, pitch_y = _grid_pitch()
    return (cols * pitch_x - _GRID_GAP, rows * pitch_y - _GRID_GAP)


def _slot_to_pos(row: int, col: int) -> tuple[int, int]:
    pitch_x, pitch_y = _grid_pitch()
    return (_GRID_GAP + int(col) * pitch_x, _GRID_GAP + int(row) * pitch_y)


def _pos_to_slot(x: int, y: int) -> tuple[int, int]:
    pitch_x, pitch_y = _grid_pitch()
    col = max(0, round((int(x) - _GRID_GAP) / pitch_x))
    row = max(0, round((int(y) - _GRID_GAP) / pitch_y))
    return int(row), int(col)


def _columns_overlap(a, b) -> bool:
    """True when two cards share at least one column (overlap horizontally)."""
    return a.col < b.col + b.cols and b.col < a.col + a.cols


def _compact(configs: Sequence["PanelConfig"], active: "PanelConfig | None" = None) -> bool:
    """Gravity layout: pull every card UP within its columns until it rests on
    the card above it (or the board top).

    This single rule is what keeps the board tidy after any drag, resize, add
    or remove.  It does BOTH jobs at once -- a card always settles strictly
    below every card it shares columns with (so nothing overlaps) AND it never
    floats above free space (so there are no vertical gaps).  The result is
    always a gap-free, top-packed grid; the previous push-DOWN-only resolver
    cleared overlaps but left holes and let the board drift downward over
    repeated drags.

    Cards settle in reading order (row, then col); ``active`` (the card the user
    just dropped) wins ties so it keeps its column/row while the others reflow
    around it.  Returns True when any card moved."""

    moved = False
    placed: list["PanelConfig"] = []
    for config in sorted(configs, key=lambda c: (c.row, c.col, 0 if c is active else 1)):
        target = 0
        for blocker in placed:
            if _columns_overlap(config, blocker):
                target = max(target, blocker.row + blocker.rows)
        if config.row != target:
            config.row = target
            moved = True
        placed.append(config)
    return moved


def _task_files_dir() -> Path:
    root = os.environ.get(TASK_FILES_ENV)
    path = Path(root) if root else Path(__file__).resolve().parents[2] / "tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ====================================================================== state
class PanelConfig:
    """One panel: kind + a size PRESET + the grid slot it is pinned to."""

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
        self.row = max(0, int(row))
        self.col = max(0, int(col))
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
        # When the primary slot already names a signal, the default source plots
        # ``signal[0]``; an empty primary slot leaves it blank ("pick a signal").  A saved
        # layout keeps its stored source verbatim.
        if source is not None:
            self.source = str(source)
        elif self.inputs and self.inputs[0]:
            self.source = "value = signal[0]"
        else:
            self.source = _BLANK_SOURCE
        # A source written as a bare ``value = <signal>`` (a saved layout or a direct pick)
        # NAMES the primary input, so reflect it in slot 0 -- the slot list and the source
        # must never disagree (the picker would otherwise show "(none)" while the panel
        # plots a signal).  ``value = signal[0]`` and an expression (``value = np.log(f)``)
        # do NOT match this and leave the slots alone.
        m = re.fullmatch(r"value\s*=\s*([A-Za-z_]\w*)", self.source.strip())
        if m and self.inputs:
            self.inputs[0] = m.group(1)
        self.params = dict(params or {})
        self.role = str(role)

    @property
    def rows(self) -> int:
        return panel_size_cells(self.size)[0]

    @property
    def cols(self) -> int:
        return panel_size_cells(self.size)[1]

    def to_dict(self) -> dict[str, object]:
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


class PanelCard(FluentGroupBox):
    """One dashboard panel: a TITLED frame (title strip = the panel KIND, top-left)
    holding the frontend canvas, a status + signal-legend footer, and a text
    "Setting" button on the title strip (top-right).  The frame border is the DRAG
    HANDLE (the matplotlib canvas keeps all its own interactions); the footer
    stretches so the card spans whole layout slots -- a 2-row card is exactly two
    1-row cards plus the gap."""

    changed = QtCore.pyqtSignal()          # any config edit (console marks dirty)
    layout_changed = QtCore.pyqtSignal()   # size/slot change (console re-arranges)
    remove_requested = QtCore.pyqtSignal(object)
    edit_requested = QtCore.pyqtSignal(object)   # "Edit…" -> open the panel's Edit tab

    def __init__(self, config: PanelConfig, parent=None, *, names_provider=None,
                 sources_provider=None, formats_provider=None, axes_provider=None):
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
        self.plotter = None
        self.canvas = None
        self._value_shape: tuple[int, ...] | None = None
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
        # _CARD_PAD insets the canvas from the frame on both sides (and is the drag
        # border); the title strip is above (padding_top), the footer below.
        holder.setContentsMargins(_CARD_PAD, scaled_px(2), _CARD_PAD, _CARD_VPAD // 2)
        holder.setSpacing(_CARD_VPAD // 2)
        self.canvas_holder = holder
        self.footer = FluentLabel("")
        # L/R padding so the legend never hugs the frame edge; word-wrap on +
        # newline-separated categories (see _refresh_signal_info) so a long
        # signal list breaks to its own lines instead of jamming into one strip.
        self.footer.setStyleSheet(
            f"color: {GREY}; background: transparent; padding: {scaled_px(2)}px {scaled_px(8)}px;")
        self.footer.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.footer.setWordWrap(True)
        # The footer carries the transient status AND a persistent SIGNAL legend
        # (what this panel READS from the hub, what its node PROVIDES, and any
        # name published by >1 node) so the hub namespace is never a mystery.
        self._status_text = ""
        self._signal_info = ""
        # added AFTER the canvas (in _build_plot); keep a stretch so the footer
        # pins under the canvas and absorbs the modulus slack below

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
        root = QtWidgets.QVBoxLayout(popup)
        pad = scaled_px(10)
        root.setContentsMargins(pad, pad, pad, pad)
        root.setSpacing(scaled_px(10, minimum=6))

        # fixed left-label column, SIZED TO CONTENT so nothing truncates: the longest
        # label is normally "colormap", but a multi-input plot adds slot rows like
        # "signal[0] occupancy" -- measure every label this popup will show and widen the
        # column to the widest (floored at 80 px), so all rows still align and read fully.
        slots = panel_input_slots(self.config.kind)
        slot_labels = [(f"signal[{i}] {sl}" if len(slots) > 1 else "signal")
                       for i, (sl, _d, _t) in enumerate(slots)]
        fm = self.fontMetrics()
        widest = max((fm.horizontalAdvance(t) for t in [*slot_labels, "colormap", "threshold"]), default=0)
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
        # ONE signal picker per INPUT SLOT (signal[0], signal[1], ...).  Most plots have a
        # single slot ("signal"); a multi-input plot (the site map) shows one combobox per
        # slot, each picking a hub signal.  The source expression reads them as
        # ``signal[0]``, ``signal[1]``, ...  -- so a plot with more inputs is just more
        # slots, no special params.
        self.slot_combos: list = []
        multi = len(slots) > 1
        for i, (slabel, _default, stip) in enumerate(slots):
            combo = FluentComboBox()
            combo.setToolTip(stip + ("\nReads into signal[%d] of the source expression." % i if multi else
                                     "\nSets the source to `value = signal[0]`."))
            combo.activated.connect(lambda _ix, idx=i: self._on_slot_pick(idx))
            self.slot_combos.append(combo)
            sec.addWidget(FluentSettingRow(slot_labels[i], combo, label_width=label_w))
        self.signal_combo = self.slot_combos[0]    # primary-slot alias

        self.source_edit = FluentLineEdit(self.config.source)
        self.source_edit.setMinimumWidth(scaled_px(280, minimum=220))
        self.source_edit.setStyleSheet(
            self.source_edit.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
        self.source_edit.setToolTip(
            "Panel data source: one line of Python evaluated against the live signals.\n"
            "Assign the result to `value`.  Namespace: every signal name (latest value),\n"
            "history(name, n), latest(name), names(), shot, np, math.")
        self.apply_button = FluentButton("Apply", color=GREEN)
        self.apply_button.setFixedWidth(scaled_px(64, minimum=52))
        self.apply_button.clicked.connect(self._apply_source)
        self.source_edit.textChanged.connect(lambda: self.apply_button.set_dirty(True))
        self.source_edit.returnPressed.connect(self._apply_source)
        expr_row = QtWidgets.QHBoxLayout()
        expr_row.setContentsMargins(0, 0, 0, 0)
        expr_row.setSpacing(scaled_px(6, minimum=4))
        expr_row.addWidget(self.source_edit, 1)
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

        # one ParamSpec widget per row -- this is where the kind's `cmap`
        # colormap lives (the colorbar COLORSET chooser; there is no separate
        # cbar show/hide toggle).
        self.param_widgets: dict[str, QtWidgets.QWidget] = {}
        for spec in PANEL_PARAMS.get(self.config.kind, ()):
            if not spec.display:
                continue            # FUNCTIONAL params live in the Edit tab, not here
            widget = self._make_param_widget(spec)
            self.param_widgets[spec.key] = widget
            sec.addWidget(FluentSettingRow(spec.label, widget, label_width=label_w))

        # relim mode + unit cycle are PURE plot-display knobs (axis autoscale style
        # + x-axis unit conversion).  They are "things only a plotter finds useful"
        # (#3), so they appear ONLY on a "plot"-role panel.  A measurement / task
        # panel's Setting keeps just source / size / colormap / title / actions --
        # if you want to restyle a measurement's curve, Add a Plot panel for it.
        # (size + colormap stay for every role: resizing a grid slot and choosing
        # an image colorset are not plotter-only conveniences.)
        if self.config.role == "plot":
            # relim mode (confocal_gui's combo_relim semantics EXACTLY: ``tight`` /
            # ``normal``).  Persisted as ``config.params["relim"]`` -- the SAME key
            # the Edit tab's ed_relim writes to, so Setting and Edit never
            # drift apart.  No "manual" mode, no x/y typed limits: live autoscale
            # is the right tool here; if you need to inspect a frozen range, use
            # zoom/pan on the canvas or Edit… into the panel's Edit tab.
            self.lim_combo = FluentComboBox()
            self.lim_combo.addItems(["tight", "normal"])
            self.lim_combo.setToolTip(
                "Relim mode (confocal_gui combo_relim naming):\n"
                "  tight  = autoscale hugs the data\n"
                "  normal = autoscale with the matplotlib default margin")
            self.lim_combo.setCurrentText(str(self.config.params.get("relim", "tight")))
            self.lim_combo.currentTextChanged.connect(self._on_relim_mode)
            sec.addWidget(FluentSettingRow("relim", self.lim_combo, label_width=label_w))

            # unit cycle: a single row [Unit button | <stretch> | current unit text]
            # under the "unit" label, so the layout rhythm stays one-control-per-row.
            self.unit_button = FluentButton("Unit", color=GREY)
            self.unit_button.setFixedWidth(scaled_px(70, minimum=56))
            self.unit_button.setToolTip(
                "Cycle the x-axis unit (GHz/nm/MHz or ns/us/ms) where the axis label\n"
                "declares one; persisted and re-applied to the live panel")
            self.unit_button.clicked.connect(self._on_unit_cycle)
            self.unit_label = FluentLabel(self._current_unit_text())
            self.unit_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self.unit_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            unit_inner = QtWidgets.QWidget()
            unit_inner_layout = QtWidgets.QHBoxLayout(unit_inner)
            unit_inner_layout.setContentsMargins(0, 0, 0, 0)
            unit_inner_layout.setSpacing(scaled_px(6, minimum=4))
            unit_inner_layout.addWidget(self.unit_button, 0)
            unit_inner_layout.addStretch(1)
            unit_inner_layout.addWidget(self.unit_label, 0)
            sec.addWidget(FluentSettingRow("unit", unit_inner, label_width=label_w))

        # ---- Panel ---------------------------------------------------------
        sec = section_box("Panel")
        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.textChanged.connect(self._on_title)
        sec.addWidget(FluentSettingRow("title", self.title_edit, label_width=label_w))

        # Action row: Remove on the left (destructive, ORANGE), Edit… opens
        # the panel's Edit tab for the heavier processing (fit / relim / per-axis
        # detail), Save Fig is the one-click export.  The stretch between Edit…
        # and Save Fig keeps the destructive button visually separated from the
        # routine "open / export" actions.
        remove = FluentButton("Remove", color=ORANGE)
        remove.setFixedWidth(scaled_px(72, minimum=58))
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        edit_button = FluentButton("Edit…", color=ACCENT)
        edit_button.setFixedWidth(scaled_px(64, minimum=52))
        edit_button.setToolTip("Open this panel's Edit tab: curve fit, command-line fit, relim, Save Fig")
        edit_button.clicked.connect(lambda: (self.settings_popup.hide(), self.edit_requested.emit(self)))
        save_fig = FluentButton("Save Fig", color=YELLOW)
        save_fig.setFixedWidth(scaled_px(80, minimum=64))
        save_fig.setToolTip("Save this panel as <title>.png + <title>.npz (timestamped) in tasks/")
        save_fig.clicked.connect(self._save_fig)
        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(scaled_px(6, minimum=4))
        action_row.addWidget(remove, 0)
        action_row.addWidget(edit_button, 0)
        action_row.addStretch(1)
        action_row.addWidget(save_fig, 0)
        sec.addLayout(action_row)

        self.settings_popup = popup
        # A Qt.Popup auto-closes on the press that lands on the Setting button; record
        # WHEN so the button's release does not immediately re-open it (real toggle).
        self._settings_dismissed_at = 0.0
        popup._on_hidden = self._note_settings_dismissed

    def _note_settings_dismissed(self) -> None:
        self._settings_dismissed_at = time.monotonic()

    def _make_param_widget(self, spec: ParamSpec, *, apply=None) -> QtWidgets.QWidget:
        """One widget per declarative ParamSpec; edits apply INSTANTLY.

        ``apply`` overrides where the edit goes (default ``self._set_param``); the
        Edit tab passes its own callback so a functional-param edit re-renders the
        live panel AND its snapshot.  Same builder for both surfaces (DRY)."""

        cb = apply if apply is not None else self._set_param
        current = self.config.params.get(spec.key, spec.default)
        if spec.kind == "choice":
            combo = FluentComboBox()
            combo.addItems(list(spec.choices))
            if str(current) in spec.choices:
                combo.setCurrentText(str(current))
            combo.setToolTip(spec.tooltip)
            combo.currentTextChanged.connect(lambda text, k=spec.key: cb(k, str(text)))
            return combo
        if spec.kind == "int":
            spin = FluentDoubleSpinBox(length=max(4, len(str(int(spec.hi)))), allow_minus=False)
            spin.setDecimals(0)
            spin.setRange(spec.lo, spec.hi)
            spin.setValue(int(current))
            spin.setToolTip(spec.tooltip)
            spin.valueChanged.connect(lambda v, k=spec.key: cb(k, int(v)))
            return spin
        # "text": a free-form value (blank allowed where documented)
        edit = FluentLineEdit(str(current))
        edit.setMinimumWidth(scaled_px(96, minimum=80))   # expands in its row; long signal names not cut off
        edit.setToolTip(spec.tooltip)
        edit.editingFinished.connect(lambda k=spec.key, w=edit: cb(k, w.text().strip()))
        return edit

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
        self._refresh_signal_combo()
        anchor = self.setting_button.mapToGlobal(
            QtCore.QPoint(self.setting_button.width(), self.setting_button.height()))
        popup.adjustSize()
        popup.move(anchor.x() - popup.width(), anchor.y() + scaled_px(2))
        popup.show()
        popup.raise_()

    def _signal_combo_items(self) -> list[tuple[str, str]]:
        """``[(display, bare_name)]`` for every LIVE hub signal, each labelled
        ``name — source  [shape]`` -- the SINGLE source every input-SLOT picker
        (signal[0], signal[1], ... including the site map's centers/image slots) shares,
        so a signal is picked the same way everywhere (by origin + shape, not typed)."""
        names: list[str] = []
        if callable(self.names_provider):
            try:
                names = sorted(str(n) for n in self.names_provider())
            except Exception:
                names = []
        sources: dict = {}
        if callable(self.sources_provider):
            try:
                sources = dict(self.sources_provider())
            except Exception:
                sources = {}
        formats: dict = {}
        if callable(self.formats_provider):
            try:
                formats = dict(self.formats_provider())
            except Exception:
                formats = {}
        items: list[tuple[str, str]] = []
        for name in names:
            producers = sources.get(name) or []
            fmt = formats.get(name)
            label = name
            if producers:
                label += f" — {', '.join(str(p) for p in producers)}"
            if fmt:
                label += f"  [{fmt}]"                 # show the signal's SHAPE too
            items.append((label, name))
        return items

    def _fill_slot_combo(self, combo, current: str) -> None:
        """Populate ONE slot combobox with ``(none)`` + every live hub signal (each
        labelled ``name — source [shape]`` via :meth:`_signal_combo_items`), keeping the
        slot's current pick selected even when its source node is not running yet (so a
        saved layout's ``signal[1]`` survives a restart)."""
        cur = str(current or "")
        with _signals_blocked(combo):
            combo.clear()
            combo.addItem("(none)", "")
            items = self._signal_combo_items()
            have = {bare for _, bare in items}
            for label, name in items:
                combo.addItem(label, name)         # text shows source + shape; data is the bare name
            if cur and cur not in have:             # preserve a not-yet-published name
                combo.addItem(f"{cur}  (not yet published)", cur)
            idx = combo.findData(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _refresh_signal_combo(self) -> None:
        """Refresh EVERY input-slot picker with the hub's current signals, each labelled
        with the measurement/processor that PRODUCES it (``occupied — occupancy``) so a
        slot is filled by origin, not a bare name.  Each slot keeps its own ``config.inputs[i]``
        selected; the source expression reads them as ``signal[0]``, ``signal[1]``, ..."""
        for i, combo in enumerate(getattr(self, "slot_combos", [])):
            cur = self.config.inputs[i] if i < len(self.config.inputs) else ""
            self._fill_slot_combo(combo, cur)

    def _on_slot_pick(self, idx: int) -> None:
        """A slot picker changed: write its bare signal name into ``config.inputs[idx]``.
        The PRIMARY slot (signal[0]) IS the "plot this signal" control, so it also points
        the source at ``value = signal[0]`` (a blank pick blanks the panel); the EXTRA
        slots (signal[1], signal[2], ... e.g. the site map's centres/image) only fill their
        input and leave the source alone.  The expression box stays the advanced override
        for a custom ``value = ...``.  Then re-apply so the pick takes effect now."""
        name = str(self.slot_combos[idx].currentData() or "")   # bare name (not the labelled text)
        while len(self.config.inputs) <= idx:
            self.config.inputs.append("")
        self.config.inputs[idx] = name
        if idx == 0:
            self.source_edit.setText("value = signal[0]" if name else _BLANK_SOURCE)
        self._apply_source()                      # picking a slot applies instantly

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
        """Persist + apply the relim mode (confocal_gui naming: ``tight`` /
        ``normal``).

        Writes to the SAME ``config.params["relim"]`` key the Edit tab's
        ``ed_relim`` writes to, so the two never drift apart."""
        self.config.params["relim"] = str(mode)
        if self.plotter is not None and hasattr(self.plotter, "relim_mode"):
            self.plotter.relim_mode = str(mode)
            # apply the switch NOW (2D colorbar / 1D y-axis), bypassing the relim
            # dead-band -- otherwise a 2D image's clim only changes on the next
            # frame (or never, for a static panel), so the toggle looks dead.
            if hasattr(self.plotter, "apply_relim_now"):
                self.plotter.apply_relim_now()
        if self.canvas is not None:
            self.canvas.draw_idle()
        self.changed.emit()

    # --------------------------------------------- basic display application
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
        if hasattr(self.plotter, "relim_mode"):
            self.plotter.relim_mode = str(self.config.params.get("relim", "tight"))
        self._apply_unit()

    def _unit_df(self):
        """A DataFigure bound to the current plotter's figure/axes whose unit is
        inferred from the LIVE x-axis label.  Built from fig/data (NOT live_plot)
        on purpose: DataFigure(live_plot=...) overrides the unit with the
        plotter's own ``unit`` (defaults to '1'), which would mask a unit-bearing
        label like 'Detuning (GHz)'.  change_unit operates on ax.lines, so the
        cycle still rewrites the live panel's curve in place."""
        from .data_figure import DataFigure
        ax = getattr(self.plotter, "ax", None)
        unit = DataFigure._infer_unit(ax.get_xlabel()) if ax is not None else None
        return DataFigure(fig=self.plotter.fig, ax=ax,
                          data_x=self.plotter.data_x, data_y=self.plotter.data_y,
                          labels=getattr(self.plotter, "labels", None), unit=unit)

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

    def _save_fig(self) -> None:
        """Save this panel: <title>.png + <title>.npz (timestamped) in tasks/."""
        if self.plotter is None:
            self.set_status("no plot to save yet", error=True)
            return
        try:
            from .data_figure import DataFigure
            df = DataFigure(self.plotter)
            stem = (self.config.title or PANEL_KINDS[self.config.kind]).strip() or self.config.kind
            out = df.save(_task_files_dir() / stem,
                          extra_info={"source": self.config.source, "kind": self.config.kind})
            self.set_status(f"saved {out['figure'].name}", error=False)
        except Exception as exc:
            self.set_status(f"save fig failed: {str(exc).splitlines()[0][:120]}", error=True)

    def _compose_footer(self) -> str:
        """status  ·  signal legend (reads / provides / duplicates), or the raw
        source when no legend has been computed yet."""
        info = self._signal_info or self.config.source
        # status on its own line; the signal legend (its own lines) below it
        return f"{self._status_text}\n{info}"[:240] if info else self._status_text[:240]

    def set_signal_info(self, info: str) -> None:
        """Set the persistent signal legend (computed by the console: what this
        panel READS from the hub, what its node PROVIDES, duplicate names)."""
        info = str(info or "")
        if info == self._signal_info:
            return
        self._signal_info = info
        self.footer.setText(self._compose_footer())

    def set_status(self, text: str, *, error: bool) -> None:
        # Text changes every tick ("shot N") -- but the COLOUR/stylesheet only
        # changes on the ok<->error transition.  Restyle only on that transition
        # (rebuilding the same stylesheet string every tick was pure waste);
        # appearance-neutral because the colour is identical when error is.
        self._status_text = str(text)
        self.status.setText(str(text)[:200])
        self.footer.setText(self._compose_footer())
        self.setting_button.setToolTip(f"Panel settings — {text}" if text else "Panel settings")
        if error is not getattr(self, "_status_error", None):
            self._status_error = bool(error)
            colour = RED if error else GREY
            self.status.setStyleSheet(f"color: {colour}; background: transparent; border: none;")
            self.footer.setStyleSheet(f"color: {colour}; background: transparent;")
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
            row, col = _pos_to_slot(self.x(), self.y())
            if (row, col) != (self.config.row, self.config.col):
                self.config.row, self.config.col = row, col
                self.changed.emit()
            self.layout_changed.emit()      # snap (even back to the same slot)
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
        self.changed.emit()
        self.layout_changed.emit()

    def _set_param(self, key: str, value) -> None:
        """A declarative parameter edit: store, rebuild the plot, mark dirty."""
        if self.config.params.get(key) == value:
            return
        self.config.params[key] = value
        self._reset_plot()
        self.changed.emit()

    def _apply_source(self) -> None:
        self.config.source = self.source_edit.text()
        self._compiled_source = self.config.source
        self._reset_plot()      # output shape may change with the expression
        self.apply_button.set_dirty(False)
        self.changed.emit()

    # ------------------------------------------------------------- data path
    def refresh(self, namespace: dict[str, object]) -> None:
        """Evaluate this panel's source against ``namespace`` and render the plot.
        Every failure lands on the gear/status -- a bad expression in one panel
        must never break the console or its siblings."""

        # Inject this panel's INPUT SLOTS as ``signal`` (a list): ``signal[0]`` is the
        # primary value, ``signal[1..]`` the extra inputs of a multi-input plot (the site
        # map's centers + image).  The source expression and _render read these, so a plot
        # with any number of inputs is just a longer slot list.
        namespace = self._with_signal_slots(namespace)
        # A BLANK panel (a freshly added pure view, source not yet wired) sits
        # quietly with a hint -- it is decoupled, so it shows nothing until the
        # user picks a signal in its Setting.  Not an error.
        if not str(self._compiled_source).strip():
            self.set_status("pick a signal in Setting", error=False)
            return
        try:
            value = self._evaluate(dict(namespace))
            self._render(value, namespace)
        except Exception as exc:
            self.set_status(str(exc).splitlines()[0][:160] or type(exc).__name__, error=True)
            return
        shot = namespace.get("shot")
        self.set_status(f"shot {int(shot)}" if isinstance(shot, (int, float)) else "ok", error=False)

    def _with_signal_slots(self, namespace: Mapping[str, object]) -> dict[str, object]:
        """Return a namespace copy with ``signal`` = the per-slot input values (one entry
        per :func:`panel_input_slots` slot, taken from ``config.inputs``); a slot whose
        signal is blank / not on the hub yet is ``None``.  So ``value = signal[0]`` plots
        the primary input and a multi-input plot reads ``signal[1]``, ``signal[2]``, ..."""
        ns = dict(namespace)
        ns["signal"] = [(ns.get(str(name)) if name else None) for name in self.config.inputs]
        return ns

    def _evaluate(self, namespace: dict[str, object]):
        # SECURITY: runs the panel's user-entered snippet as arbitrary Python.
        # Same trusted-local-tool posture as the pulse GUI Scan tab -- only run
        # layouts you wrote or trust.
        exec(self._compiled_source, namespace)  # noqa: S102 - local experiment tool, trusted input only
        if "value" not in namespace:
            raise ValueError("assign the panel data to a `value = ...` variable")
        return namespace["value"]

    def _coerce(self, value):
        kind = self.config.kind
        arr = np.asarray(value, dtype=float)
        if kind == "1d":
            # 1D panels are y-vs-index by default.  Only a panel EXPLICITLY
            # flagged xy=True (e.g. a curve view whose source is
            # value = column_stack([x_key, y_key])) reads an (N, 2) value as an
            # x-y curve (col0 = x, col1 = y).  A plain 1d panel that happens to
            # produce (N, 2) still flattens, so the x-y meaning is opt-in, not a
            # silent shape heuristic.
            if self.config.params.get("xy") and arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
                return arr
            flat = arr.reshape(-1)
            if flat.size < 1:
                raise ValueError("panel value is empty")
            return flat
        if kind == "2d":
            if arr.ndim != 2 or min(arr.shape) < 2:
                raise ValueError(f"2D panel needs a 2D array value (got shape {arr.shape})")
            # bound the point table so the grid scatter stays cheap per tick
            sy = max(1, int(np.ceil(arr.shape[0] / 192)))
            sx = max(1, int(np.ceil(arr.shape[1] / 192)))
            return arr[::sy, ::sx]
        if kind in ("monitor", "monitor_nodist"):
            flat = arr.reshape(-1)
            if flat.size != 1:
                raise ValueError(f"rolling-trace panel needs a scalar value (got shape {arr.shape})")
            return float(flat[0])
        flat = arr.reshape(-1)
        if flat.size < 1:
            raise ValueError("panel value is empty")
        if kind == "sites" and flat.size > 4096:
            raise ValueError(f"site-map panel needs one value per site (got {flat.size} values)")
        return flat

    def _sites_aux(self, namespace: Mapping[str, object]):
        """The site map's extra INPUT SLOTS: ``signal[1]`` = centres, ``signal[2]`` =
        optional camera underlay (``signal[0]`` is the occupancy = the panel value).  The
        namespace carries ``signal`` (the per-slot list) injected by ``_with_signal_slots``.
        """
        sig = namespace.get("signal") or []
        centers = sig[1] if len(sig) > 1 else None
        if centers is None:
            slot = self.config.inputs[1] if len(self.config.inputs) > 1 else "centers"
            raise ValueError(
                f"site map needs a centres signal in slot signal[1] (`{slot}`) with the (N, 2) "
                "site centres -- pick it in Setting -> centers")
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] < 2:
            raise ValueError(f"centres signal must have shape (N, 2); got {centers.shape}")
        image = sig[2] if len(sig) > 2 else None
        return centers[:, :2], (None if image is None else np.asarray(image, dtype=float))

    def _co_names(self) -> frozenset:
        """The hub-signal names this panel reads (cached) -- for the monitor roll-gate
        and the 2D coordinate-frame (ROI) lookup.  This is BOTH the identifiers the
        source expression names directly AND the slot inputs it pulls in as
        ``signal[i]`` (the default ``value = signal[0]`` form references ``signal``, not
        the real name, so the slot names must be folded in or version-gating misses)."""
        key = (self._compiled_source, tuple(self.config.inputs))
        if key != self._ref_src:               # (re)derive on source OR slot change
            self._ref_src = key
            try:
                names = set(compile(str(self._compiled_source), "<panel-source>", "exec").co_names)
            except Exception:
                names = set()
            names.update(str(n) for n in self.config.inputs if n)   # slot signal[i] -> real name
            self._ref_names = frozenset(names)
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

    def _render(self, value, namespace: Mapping[str, object] | None = None) -> None:
        # PERFORMANCE: push data with draw=False and queue ONE draw_idle per
        # panel -- rendering happens in Qt's paint pass (coalesced per frame),
        # never synchronously inside the refresh tick.  hist accepts any sample
        # count (HistogramFigure.update rebins itself), so a growing history
        # must NOT rebuild the whole plot every shot.
        value = self._coerce(value)
        kind = self.config.kind
        rebuild = self.plotter is None
        if not rebuild and kind in ("2d", "1d", "sites"):
            rebuild = tuple(np.shape(value)) != self._value_shape
        if not rebuild and kind == "2d":
            # A ROI that *shifts* without resizing keeps the frame shape, but the
            # image's pixel coordinates moved -- rebuild so the axes track the ROI.
            roi = self._source_coord_frame(namespace)
            rebuild = (list(roi) if roi else None) != getattr(self, "_roi_built", None)
        if rebuild:
            self._build_plot(value, namespace)
            self._value_shape = (1,) if isinstance(value, float) else tuple(np.shape(value))
            if kind in ("monitor", "monitor_nodist"):
                # The build already plotted this value as the first point; record
                # its source version so the next UNRELATED bump won't duplicate it.
                self._last_monitor_key = self._monitor_source_key(namespace)
            return
        if kind == "2d":
            self.plotter.update(np.asarray(value).ravel(), draw=False)
        elif kind in ("monitor", "monitor_nodist"):
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
        else:  # hist / 1d-vector
            self.plotter.update(value, draw=False)
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
        # The default source plots an input SLOT (``value = signal[0]``): resolve it to the
        # real hub-signal name in that slot so its producing node's axis label is found.
        m = re.fullmatch(r"signal\[(\d+)\]", name)
        if m:
            i = int(m.group(1))
            name = self.config.inputs[i] if i < len(self.config.inputs) else ""
        if not name or not callable(self.axes_provider):
            return None
        try:
            axes = dict(self.axes_provider())
        except Exception:
            return None
        entry = axes.get(name)
        return entry[0] if entry else None

    # ------------------------------------------------------------- plot lifecycle
    def _build_plot(self, value, namespace: Mapping[str, object] | None = None) -> None:
        if panel_canvas is None:
            raise RuntimeError("matplotlib Qt canvas is not available")
        self._teardown_plot()
        kind = self.config.kind
        size = self.config.size
        label = self.config.title or PANEL_KINDS[kind]
        if kind == "sites":
            centers, image = self._sites_aux(namespace or {})
            vec = np.asarray(value, dtype=float).reshape(-1)
            if len(vec) != len(centers):
                raise ValueError(
                    f"site-map value has {len(vec)} entries but the centers signal has {len(centers)} sites")
            self.plotter = panel_plot(
                centers, vec, kind="sites", size=size, interactions=False,
                image=image, roi_radius=site_ring_radius(centers),
                cmap=str(self.config.params.get("cmap", "gray")),
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
            self.plotter = panel_plot(
                data_x, arr.ravel(), kind="2d", size=size, interactions=False,
                cmap=str(self.config.params.get("cmap", "inferno")),
                relim_mode=str(self.config.params.get("relim", "tight")),
                labels=(xlabel, ylabel, ""), title=self.config.title or None)
        elif kind in ("monitor", "monitor_nodist"):
            length = max(20, int(self.config.params.get("length", 300)))
            history = np.full(length, np.nan)
            self.plotter = panel_plot(
                np.arange(length, dtype=float), history, kind=kind, size=size, interactions=False,
                labels=("Shots ago", self._source_axis_label() or label, "Z"),
                relim_mode=str(self.config.params.get("relim", "tight")),
                title=self.config.title or None)
            self.plotter.roll(float(value), draw=False)
        elif kind == "hist":
            self.plotter = panel_plot(
                np.asarray(value, dtype=float), kind="hist", size=size, interactions=False,
                bins=int(self.config.params.get("bins", 60)),
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
                vec = arr.reshape(-1)
                data_x = np.arange(len(vec), dtype=float)
                xlabel, ylabel = "Site", self._source_axis_label() or label
            self.plotter = panel_plot(
                data_x, vec, kind="1d", size=size, interactions=False,
                labels=(xlabel, ylabel, "Z"),
                relim_mode=str(self.config.params.get("relim", "tight")),
                title=self.config.title or None)
        # Monitor cards are display-only: NO selectors (interactions=False above)
        # and the wheel scrolls the board (isolate_wheel=False) instead of being
        # swallowed.  Interactive zoom / select lives in the Edit tab.
        self.canvas = panel_canvas(self.plotter.fig, isolate_wheel=False)
        # Pin the canvas to its DESIGN size so the surrounding QVBoxLayout can never
        # squish it below the panel region: the signal-legend footer word-wraps to
        # several lines for a rich node chain, and without a floor it would steal
        # the canvas's height (breaking the canvas-fits-card contract).  The card
        # is setFixedSize to exactly hold canvas + chrome, so the footer keeps its
        # _FOOTER_MIN slot and a long legend elides rather than shrinking the plot.
        self.canvas.setMinimumSize(self.canvas.sizeHint())
        self.canvas_holder.insertWidget(0, self.canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        if self.canvas_holder.indexOf(self.footer) < 0:
            # footer (status line) sits DIRECTLY under the canvas, then the
            # stretch absorbs the multi-row tiling slack below it -- so the slack
            # reads as a clean bottom margin rather than a gap floating between
            # the plot and its status line.
            self.canvas_holder.addWidget(self.footer)
            self.canvas_holder.addStretch(1)
        # re-apply persisted Setting toggles (unit + manual x/y limits) to the
        # FRESH plotter every rebuild -- the panel rebuilds whenever its data
        # shape changes, so without this the toggle would silently revert.
        self._apply_display_params()
        self.canvas.draw_idle()
        self._place_setting_button()

    def _reset_plot(self) -> None:
        """Drop the plot so the next refresh rebuilds it (size/params/source changed)."""
        self._teardown_plot()
        self._value_shape = None

    def _teardown_plot(self) -> None:
        canvas, plotter = self.canvas, self.plotter
        self.canvas = None
        self.plotter = None
        if canvas is not None:
            self.canvas_holder.removeWidget(canvas)
            canvas.deleteLater()
        if plotter is not None and plt is not None and plotter.fig is not None:
            plt.close(plotter.fig)

    def shutdown(self) -> None:
        self._teardown_plot()


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
                 controls: bool = True):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._specs = list(measurements)
        self._single = bool(single)               # bound to ONE spec (per-panel Edit) -> hide the picker
        self._controls = bool(controls)           # False = no Start/Stop (e.g. a plot Edit's read-only-ish source form)
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

        # auto-generated parameter form (rebuilt when the type changes)
        self.form = QtWidgets.QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.form.setHorizontalSpacing(scaled_px(8, minimum=5))
        self.form.setVerticalSpacing(scaled_px(5, minimum=3))
        # Fields fill the available width (the Edit page is wide): an expanding
        # control -- a free-form value edit -- grows so its value is never cut off
        # and the right-side space is used; fixed-size spins keep their natural
        # width.  (cutoff is a core rule.)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
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
        self._widgets = {}
        while self.form.count():
            item = self.form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

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

    def _rebuild_form(self) -> None:
        """Rebuild the parameter form for the currently selected measurement."""
        self._clear_form()
        spec = self.current_spec()
        if spec is None:
            return
        for decl in spec.params:
            kind = decl.kind
            # Show the REAL parameter name (the key threaded into the build call, e.g.
            # ``data_dir`` / ``calibration_frames``), not a hand-written prettified label
            # -- so the form is unambiguously the node's actual parameters.
            label_text = decl.key + (f" ({decl.unit})" if decl.unit else "")
            if decl.required:
                label_text += " *"
            if kind == "axis_range":
                lo, hi, points = decl.default if decl.default is not None else (0.0, 1.0, 2)
                lo_spin = self._spin(decl, integer=False, value=lo)
                hi_spin = self._spin(decl, integer=False, value=hi)
                pts_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
                pts_spin.setDecimals(0)
                pts_spin.setRange(2, 100000)
                pts_spin.setValue(int(points))
                pts_spin.setToolTip("Number of scan points (>= 2).")
                pts_spin.valueChanged.connect(self._refresh_start_enabled)
                triplet = QtWidgets.QWidget()
                triplet.setStyleSheet("background: transparent;")
                trow = QtWidgets.QHBoxLayout(triplet)
                trow.setContentsMargins(0, 0, 0, 0)
                trow.setSpacing(scaled_px(4, minimum=3))
                trow.addWidget(lo_spin); trow.addWidget(self._tag("to")); trow.addWidget(hi_spin)
                trow.addWidget(self._tag("/")); trow.addWidget(pts_spin); trow.addWidget(self._tag("pts"))
                self.form.addRow(label_text, triplet)
                self._widgets[decl.key] = ("axis_range", lo_spin, hi_spin, pts_spin)
            elif kind == "bool":
                check = FluentCheckBox("")
                check.setChecked(bool(decl.default))
                check.setToolTip(decl.tooltip)
                check.toggled.connect(self._refresh_start_enabled)
                self.form.addRow(label_text, check)
                self._widgets[decl.key] = ("bool", check)
            elif kind == "choice":
                combo = FluentComboBox()
                combo.addItems([str(c) for c in decl.choices])
                if decl.default is not None and str(decl.default) in [str(c) for c in decl.choices]:
                    combo.setCurrentText(str(decl.default))
                combo.setToolTip(decl.tooltip)
                combo.activated.connect(lambda *_: self._refresh_start_enabled())
                self.form.addRow(label_text, combo)
                self._widgets[decl.key] = ("choice", combo)
            elif kind == "int":
                # an optional int param (default None, not required) shows blank
                # via a line edit so "leave blank" stays expressible; bounded ints
                # with a default use a spin box.
                if decl.default is None and not decl.required:
                    edit = FluentLineEdit("")
                    edit.setMinimumWidth(scaled_px(96, minimum=80))   # grows with the form field; no cutoff
                    edit.setPlaceholderText("(all)")
                    edit.setToolTip(decl.tooltip)
                    edit.textChanged.connect(self._refresh_start_enabled)
                    self.form.addRow(label_text, edit)
                    self._widgets[decl.key] = ("int_opt", edit)
                else:
                    spin = self._spin(decl, integer=True)
                    self.form.addRow(label_text, spin)
                    self._widgets[decl.key] = ("int", spin)
            elif kind == "text":
                edit = FluentLineEdit("" if decl.default is None else str(decl.default))
                edit.setMinimumWidth(scaled_px(160, minimum=120))   # grows with the field; no cutoff
                edit.setPlaceholderText(decl.tooltip[:48] if decl.tooltip else "")
                edit.setToolTip(decl.tooltip)
                edit.textChanged.connect(self._refresh_start_enabled)
                self.form.addRow(label_text, edit)
                self._widgets[decl.key] = ("text", edit)
            else:  # float
                spin = self._spin(decl, integer=False)
                self.form.addRow(label_text, spin)
                self._widgets[decl.key] = ("float", spin)
        self._refresh_start_enabled()

    # ------------------------------------------------------------- value read
    def collect_values(self) -> dict[str, object]:
        """Read every parameter back BY KIND (no eval) into a build kwargs dict."""
        values: dict[str, object] = {}
        for key, entry in self._widgets.items():
            tag = entry[0]
            if tag == "axis_range":
                _, lo_spin, hi_spin, pts_spin = entry
                values[key] = (float(lo_spin.value()), float(hi_spin.value()), int(pts_spin.value()))
            elif tag == "bool":
                values[key] = bool(entry[1].isChecked())
            elif tag == "choice":
                values[key] = entry[1].currentText()
            elif tag == "int":
                values[key] = int(entry[1].value())
            elif tag == "int_opt":
                text = entry[1].text().strip()
                values[key] = int(text) if text else None
            elif tag == "text":
                values[key] = entry[1].text()
            else:  # float
                values[key] = float(entry[1].value())
        return values

    def set_axis_range(self, lo: float, hi: float) -> bool:
        """Fill the FIRST axis_range param's Min/Max from a selected region
        (confocal _read_range: a plot selection becomes the next scan's range).
        Returns True if an axis_range param was found and set."""
        for entry in self._widgets.values():
            if entry[0] == "axis_range":
                _, lo_spin, hi_spin, _pts = entry
                lo_spin.setValue(float(min(lo, hi)))
                hi_spin.setValue(float(max(lo, hi)))
                return True
        return False

    def _missing_required(self) -> list[str]:
        """Required params whose value is empty (only the optional-int line edit
        can be blank; spin boxes always hold a number)."""
        spec = self.current_spec()
        if spec is None:
            return []
        missing = []
        for decl in spec.params:
            if not decl.required:
                continue
            entry = self._widgets.get(decl.key)
            if entry is None:
                missing.append(decl.label)
            elif entry[0] in ("int_opt", "text") and not entry[1].text().strip():
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
        for key, entry in self._widgets.items():
            if key not in values:
                continue
            val = values[key]
            tag = entry[0]
            try:
                if tag == "axis_range" and isinstance(val, (tuple, list)) and len(val) == 3:
                    entry[1].setValue(float(val[0])); entry[2].setValue(float(val[1]))
                    entry[3].setValue(int(val[2]))
                elif tag == "bool":
                    entry[1].setChecked(bool(val))
                elif tag == "choice":
                    entry[1].setCurrentText(str(val))
                elif tag == "int":
                    entry[1].setValue(int(val))
                elif tag == "int_opt":
                    entry[1].setText("" if val is None else str(int(val)))
                elif tag == "text":
                    entry[1].setText("" if val is None else str(val))
                elif tag == "float":
                    entry[1].setValue(float(val))
            except (TypeError, ValueError):
                continue
        self._refresh_start_enabled()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.start_button.setEnabled(not running and not self._missing_required())
        self.stop_button.setEnabled(bool(running))
        self.type_combo.setEnabled(not running)
        for entry in self._widgets.values():
            for widget in entry[1:]:
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
            lab = FluentLabel(text)
            lab.setStyleSheet(f"color: {GREY}; background: transparent; border: none; font-weight: bold;")
            col.addWidget(lab)

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
            # widest node-param label is "calibration_frames" (18 chars) --
            # 150 clipped its trailing 's'; 170 fits the longest name in full.
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
                self.source_form = MeasurementPanel([source_spec], single=True, controls=False)
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
        functional = [s for s in PANEL_PARAMS.get(card.config.kind, ()) if not s.display]
        if is_plot and functional:
            section("Parameters")
            for s in functional:
                widget = card._make_param_widget(s, apply=self._edit_param)
                col.addWidget(FluentSettingRow(s.label, widget, label_width=scaled_px(96, minimum=72)))

        # ---- Processing: frozen snapshot + fit + limits + save -------------
        section("Processing")
        head = QtWidgets.QHBoxLayout()
        head.addWidget(labeled("frozen snapshot of current data"), 1)
        self.refresh_button = FluentButton("Refresh", color=GREY)
        self.refresh_button.setToolTip("Re-snapshot the panel's current data")
        self.refresh_button.clicked.connect(self.rebuild)
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        self.save_button.setToolTip("Save the edited figure (png) + data (npz), timestamped, into tasks/")
        self.save_button.clicked.connect(self.save)
        head.addWidget(self.refresh_button)
        head.addWidget(self.save_button)
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

            # manual x/y limits (DataFigure.xlim/ylim) -- NOT in Setting, so they
            # live here.  Unit + relim are deliberately ABSENT (Setting owns them).
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

        self.status = FluentLabel("")
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
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
        self.teardown()
        src = card.plotter
        kind = card.config.kind
        size = card.config.size          # mirror the Monitor frame, never force 2x4
        title = card.config.title or PANEL_KINDS[kind]
        relim = str(card.config.params.get("relim", "tight"))
        cmap = str(card.config.params.get("cmap", "gray" if kind == "sites" else "inferno"))
        try:
            if kind == "2d":
                self._plotter = panel_plot(np.array(src.data_x, dtype=float),
                                           np.array(src.data_y[:, 0], dtype=float), kind="2d",
                                           size=size, cmap=cmap, relim_mode=relim,
                                           labels=tuple(src.labels), title=title)
            elif kind == "sites":
                self._plotter = panel_plot(np.array(src.data_x[:, :2], dtype=float),
                                           np.array(src.data_y[:, 0], dtype=float), kind="sites",
                                           size=size, image=getattr(src, "background", None),
                                           roi_radius=getattr(src, "roi_radius", 3.0), cmap=cmap,
                                           labels=tuple(src.labels), title=title)
            elif kind == "hist":
                self._plotter = panel_plot(np.array(src.values, dtype=float), kind="hist", size=size,
                                           bins=int(card.config.params.get("bins", 60)),
                                           labels=tuple(src.labels), title=title)
            else:  # 1d / monitor / monitor_nodist -> a line snapshot
                self._plotter = panel_plot(np.array(src.data_x[:, 0], dtype=float),
                                           np.array(src.data_y[:, 0], dtype=float),
                                           kind=kind, size=size, relim_mode=relim,
                                           labels=tuple(src.labels), title=title)
        except Exception as exc:
            self.status.setText(f"could not snapshot: {str(exc).splitlines()[0][:120]}")
            return
        self._canvas = panel_canvas(self._plotter.fig)
        # The Edit page lives in a scroll area; without a floor its QVBoxLayout
        # SQUISHES the canvas below the figure's design height (clipping the plot
        # and the y-axis label -- the snapshot then looks empty/broken).  Pin the
        # minimum to the figure's own size so the page SCROLLS instead of squishing.
        self._canvas.setMinimumSize(self._canvas.sizeHint())
        self.canvas_holder.addWidget(self._canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self._canvas.draw_idle()
        self._df = None
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
        """A plot-param edit from the Edit tab: apply to the LIVE panel (re-renders
        it) and re-snapshot here, so the change shows in both surfaces."""
        if self.card is not None:
            self.card._set_param(key, value)
        self.rebuild()

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

    def save(self) -> None:
        if self._plotter is None:
            return
        try:
            df = self._df_for()
            stem = (self.card.config.title or self.card.config.kind).strip() or "panel"
            out = df.save(_task_files_dir() / stem,
                          extra_info={"source": self.card.config.source,
                                      "kind": self.card.config.kind})
            self.status.setText(f"saved {out['figure'].name} + {out['data'].name}")
        except Exception as exc:
            self.status.setText(f"save failed: {str(exc).splitlines()[0][:120]}")


class _PanelBoard(QtWidgets.QWidget):
    """Absolute-positioned canvas the cards live on (drag + snap layout)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

    def arrange(self, cards: Sequence[PanelCard]) -> None:
        max_x = max_y = 0
        for card in cards:
            x, y = _slot_to_pos(card.config.row, card.config.col)
            card.move(x, y)
            max_x = max(max_x, x + card.width())
            max_y = max(max_y, y + card.height())
        self.setMinimumSize(max_x + _GRID_GAP, max_y + _GRID_GAP)


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
        self.publishes_label.setWordWrap(False)
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
        """Show the node's outputs as a readable TABLE -- ONE signal per line, the
        ``name`` / ``shape`` columns aligned, then the human meaning::

            publishes:
              occupied    (35,)     per-site single-shot occupancy (0 / 1)
              rate        scalar    running-mean loading rate over all sites

        ``rows`` is ``[(name, shape, description)]`` from the console (shapes
        AUTO-EXTRACTED from the real values via ``logic.describe_shape``; meanings from
        the node's ``output_specs``).  A pending shape (``—``) just means no value yet."""
        rows = list(rows)
        if rows:
            name_w = max(len(str(n)) for n, _, _ in rows)
            shape_w = max(len(str(s)) for _, s, _ in rows)
            lines = ["publishes:"]
            for name, shape, description in rows:
                line = f"  {str(name):<{name_w}}  {str(shape):<{shape_w}}"
                if description:
                    line += f"  {description}"
                lines.append(line.rstrip())
            text = "\n".join(lines)
        else:
            text = "publishes: (nothing on the hub)"
        if text != self.publishes_label.text():       # skip churn: shapes refresh each tick
            self.publishes_label.setText(text)


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
        self.form = MeasurementPanel([spec] if spec is not None else [], single=True)
        self.form.seed_values(row.node.values or {})
        self.form.start_requested.connect(lambda *_: self.console._start_logic_node(self.row))
        self.form.stop_requested.connect(lambda: self.console._stop_logic_node(self.row))
        col.addWidget(self.form)
        # (The node's published-signals + shapes are shown on its Logic-tab ROW card,
        # the single place for that legend -- not duplicated here.)
        col.addStretch(1)

    def collect_values(self) -> dict:
        return self.form.collect_values()

    def set_running(self, running: bool) -> None:
        self.form.set_running(running)

    def set_status(self, text: str, *, error: bool) -> None:
        self.form.set_status(text, error=error)

    def teardown(self) -> None:
        # No matplotlib resources here (a logic node never plots), so teardown is a
        # no-op -- present so the console can treat it like a PanelEditor.
        pass


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
        window_ratio: float = 0.84,
        window_px: tuple[int, int] | None = None,
    ):
        ensure_qt_app()
        set_fluent_scale(scale)
        super().__init__()
        self.hub = hub
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
        self._logic_nodes: dict[int, object] = {}      # id(row) -> node (or None)
        self._logic_editors: dict[int, "LogicNodeEditor"] = {}  # id(row) -> Edit tab
        self._last_version = -1
        self._building = False
        self._address: str | None = None
        # No-measurement / no-processor sentinels kept so older callers / tests that
        # probe "is there a global measurement launcher" still read None (there is
        # no global form -- a measurement is a Logic node now).
        self.measurement_panel = None
        self.measurement_group = None
        self.measurement_placeholder = None
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

        self._build_ui()
        self.load_state(self.state)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.state.interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------ UI
    def _target_console_size(self) -> QtCore.QSize:
        """Window size from the primary screen (the shared GUI sizing rule)."""

        if self._window_px is not None:
            return QtCore.QSize(int(self._window_px[0]), int(self._window_px[1]))
        return screen_fit_window_size(self.window_ratio)

    def _build_ui(self) -> None:
        self.setWindowTitle("TaskConsole@Zou lab")
        self.setStyleSheet(fluent_widget_stylesheet())
        self.setFixedSize(self._target_console_size())
        root = QtWidgets.QVBoxLayout(self)
        margin = scaled_px(14)
        root.setContentsMargins(margin, scaled_px(8), margin, scaled_px(8))
        root.setSpacing(scaled_px(7, minimum=5))

        header_frame = FluentFrame()
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
        self.summary = FluentLineEdit("")
        self.summary.setEnabled(False)

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
        for key, label in PANEL_KINDS.items():
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
        self.save_button = FluentButton("Save", color=ACCENT)
        self.save_button.clicked.connect(self.save_to_file)
        load_button = FluentButton("Load", color=ORANGE)
        load_button.clicked.connect(self.load_from_file)

        for widget in (self.status_dot, self.name_edit):
            header.addWidget(widget)
        header.addWidget(self.summary, 1)
        for widget in (self.kind_combo, add_button, self.save_button, load_button):
            header.addWidget(widget)
        # header controls disabled while a task runs (the lockout, #5); kept as a group
        # so ``_set_task_running`` flips them all at once.
        self._lockable_header = (self.kind_combo, add_button, self.save_button, load_button,
                                 self.name_edit)
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
        self.tabs.add_permanent_tab(dash_tab, "Monitor")

        # Logic tab: a scrolled top-packed column of LogicNodeRow cards.
        logic_tab = QtWidgets.QWidget()
        logic_tab.setStyleSheet("background: transparent;")
        logic_outer = QtWidgets.QVBoxLayout(logic_tab)
        logic_outer.setContentsMargins(0, 0, 0, 0)
        self.logic_scroll = FluentScrollArea()
        self.logic_scroll.setWidgetResizable(True)
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
                self._attach_card(PanelCard(config, parent=self.board,
                                            names_provider=self.hub.names,
                                            sources_provider=self._signal_providers, formats_provider=self._signal_formats, axes_provider=self._signal_axes))
            for node in state.logic:
                self._attach_logic_node(node)    # always STOPPED -- Start is manual
            self._arrange()
        finally:
            self._building = False
        self._last_version = -1            # force a refresh on the next tick
        self._update_summary()

    def _attach_card(self, card: PanelCard) -> None:
        card.setParent(self.board)
        card.show()
        card.changed.connect(self._mark_dirty)
        card.layout_changed.connect(self._arrange)
        card.remove_requested.connect(self._remove_panel)
        card.edit_requested.connect(self._edit_card)
        self.cards.append(card)

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
        return names - {"np", "numpy", "math", "history", "latest", "names", "shot", "value"}

    def _producing_node(self, card: "PanelCard"):
        """The running node whose published signals the panel's source reads (None if
        the expression touches no published signal, e.g. a pure constant)."""
        refs = self._referenced_signals(card.config.source)
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
        """``name -> [node labels]`` for every signal currently published, so a
        name carried by MORE THAN ONE node can be flagged as ambiguous."""
        providers: dict[str, list] = {}
        for node in self.running_nodes:
            if not hasattr(node, "published_signals"):
                continue
            label = self._node_label(node)
            for name in node.published_signals():
                providers.setdefault(str(name), []).append(label)
        return providers

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
                out[str(name)] = describe_shape(self.hub.latest(name))
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
        for full in sorted(node.published_signals()):     # already hub names (incl. prefix)
            try:
                shape = describe_shape(self.hub.latest(full))
            except Exception:
                shape = "—"
            rows.append((str(full), shape, desc(str(full))))
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
        spec = self._spec_for_logic(row.node)
        keys = list(getattr(spec, "result_keys", ()) or [])
        if not keys and row.node.kind == "camera":
            keys = ["frame"]
        if not keys and row.node.kind == "task":          # off-hub one-shot, mid-run stream
            keys = [f"{getattr(spec, 'mid_run_key', 'frame')} (mid-run)"]
        row.set_publishes([(k, "—", "") for k in keys])

    def _node_for_signal(self, name: str):
        """The first running node that PUBLISHES signal ``name`` (None if none does)."""
        for node in self.running_nodes:
            if hasattr(node, "published_signals") and name in node.published_signals():
                return node
        return None

    def _refresh_signal_info(self) -> None:
        """Give every panel a legend (shown in its footer) naming, FOR EACH signal the
        panel actually reads, WHICH node + layer produces it -- e.g.
        ``occupied ← occupancy [processor]``.  It lists only the signals this panel
        uses (not the producing node's whole output set), so the footer answers
        exactly "this plot's value comes from which measurement/processor".  A read
        published by more than one running node is flagged ambiguous.  Self-guarded:
        recomputes only when the sources / nodes / published names change."""
        providers = self._signal_providers()
        sig = (tuple(sorted((k, len(v)) for k, v in providers.items())),
               tuple((id(c), c.config.source) for c in self.cards))
        if sig == getattr(self, "_signal_info_sig", None):
            return
        self._signal_info_sig = sig
        for card in self.cards:
            reads = sorted(self._referenced_signals(card.config.source))
            parts: list[str] = []
            for name in reads:
                src = self._node_for_signal(name)
                if src is not None:
                    layer = str(getattr(src, "layer", ""))
                    tag = self._node_label(src)
                    tag = f"{tag} [{layer}]" if layer and layer != "node" else tag
                    note = "  ⚠ also from another node" if len(providers.get(name, [])) > 1 else ""
                    parts.append(f"{name} ← {tag}{note}")
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
        # re-snapshot a PanelEditor's frozen limits when its tab is shown
        widget = self.tabs.currentWidget()
        if isinstance(widget, PanelEditor):
            widget.fill_limits()

    def _arrange(self) -> None:
        # gravity: the just-dragged / resized card keeps its dropped slot (it
        # wins ties) and every other card falls UP to fill gaps, so the board
        # stays gap-free, top-packed and overlap-free (see _compact).
        active = self.sender()
        active_cfg = active.config if isinstance(active, PanelCard) and active in self.cards else None
        if _compact([c.config for c in self.cards], active_cfg):
            self._mark_dirty()
        self.board.arrange(self.cards)
        self._update_summary()

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
        rows = max((c.config.row + c.config.rows for c in self.cards), default=0)
        config = PanelConfig(kind=str(kind), row=rows, col=0, size="1x2")
        card = PanelCard(config, parent=self.board, names_provider=self.hub.names,
                         sources_provider=self._signal_providers, formats_provider=self._signal_formats, axes_provider=self._signal_axes)
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

    def _attach_logic_node(self, node: LogicNodeConfig, *, focus: bool = False) -> "LogicNodeRow":
        """Add a STOPPED logic-node row to the Logic tab (no node built yet)."""
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
        # stop a previous run of THIS node so running nodes don't pile up
        self._stop_logic_node(row, _silent=True)
        try:
            node = self._build_logic_node(row.node, values)
        except Exception as exc:
            row.set_state("error", status=f"build failed: {str(exc).splitlines()[0][:80]}")
            if editor is not None:
                editor.set_status(f"build failed: {str(exc).splitlines()[0][:140]}", error=True)
            return
        row.node.values = dict(values)            # remember for the next Edit reopen + save
        self._logic_nodes[id(row)] = node
        if node not in self.running_nodes:
            self.running_nodes.append(node)
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
        card = PanelCard(config, parent=self.board, names_provider=self.hub.names,
                         sources_provider=self._signal_providers, formats_provider=self._signal_formats, axes_provider=self._signal_axes)
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
        """Leave task-run mode (task finished or stopped): drop the lock + banner and
        the transient mid-run panel (its job was to show the work in progress)."""
        self._running_task_row = None
        card, self._task_card = self._task_card, None
        self._apply_task_lock(False)
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

    def _build_logic_node(self, node: LogicNodeConfig, values: dict):
        """Build the node for a logic node, DISPLAY SUPPRESSED (publish-only).

        Reuses the SAME build paths as the real readout / notebook:
          * camera      -> readout.camera_measurement(hub)
          * measurement -> spec.build(**values) wrapped in a ScannedMeasurementNode
          * processor   -> a ProcessorRun driving the spec once
          * task        -> spec.build(hub, **values)
        None of these ever opens a matplotlib plot -- they only publish to the hub."""
        kind = node.kind
        if kind == "camera":
            spec = self._spec_for_logic(node)
            if spec is None:
                raise RuntimeError("no session camera available")
            # Same build path as the notebook: readout.camera_spec().build(hub, ...).
            return spec.build(self.hub, **values)
        spec = self._spec_for_logic(node)
        if spec is None:
            raise RuntimeError(f"no catalog spec named {node.name!r} for a {kind} node")
        if kind == "measurement":
            measurement = spec.build(**values)
            from Zou_lab_control.neutral_atom.operations.logic import ScannedMeasurementNode
            return ScannedMeasurementNode(
                self.hub, measurement, x_key=spec.x_key, y_key=spec.y_key, grid_shape=spec.grid_shape)
        if kind == "processor":
            # REACTIVE processor (the "func" layer): a live node consuming hub signals
            # and republishing derived ones -- e.g. judging occupancy from each frame.
            if getattr(spec, "reactive", False):
                return spec.make_node(self.hub, **values)
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
        if row in self.logic_nodes:
            self.logic_nodes.remove(row)
        self.logic_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        if not self.logic_nodes:
            self.logic_hint.show()
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
                # A one-shot TASK that finishes ON ITS OWN (not via the Stop button)
                # must ALSO release the console lockout here -- otherwise a calibration
                # that completes normally leaves the dashboard locked forever (only a
                # manual Stop reaches _clear_task_running).
                if row is self._running_task_row:
                    self._clear_task_running()
            elif running:
                # surface scan progress when the node reports it
                done = getattr(node, "points_done", None)
                total = getattr(node, "n_points", None)
                if done is not None and total:
                    row.set_state("running", status=f"running {done}/{total}")
                else:
                    row.set_state("running", status="running")
                # refresh the published shapes as the node's first values land (a reactive
                # processor publishes nothing until it consumes a frame); set_publishes is
                # self-guarded so this is a no-op once the shapes stop changing.
                self._update_row_publishes(row)

    def _mark_dirty(self, *_args) -> None:
        if self._building:
            return
        self.save_button.set_dirty(True, dirty_color=YELLOW)
        self._update_summary()

    # ------------------------------------------------------------------ refresh
    def _expression_namespace(self) -> dict[str, object]:
        namespace: dict[str, object] = {"np": np, "numpy": np, "math": _math}
        namespace.update(self.hub.snapshot_latest())
        namespace["history"] = self.hub.history
        namespace["latest"] = self.hub.latest
        namespace["names"] = self.hub.names
        namespace["shot"] = self.hub.shot
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

    def _tick(self) -> None:
        # poll the logic nodes EVERY tick (even when no new signal arrived) so a
        # one-shot node's run-complete / error transition is never missed if the
        # node self-stops between version bumps.
        self._poll_logic_nodes()
        self._refresh_signal_info()   # cheap + self-guarded: tracks source/node changes
        # A running task's mid-run output is OFF the hub (#6), so it does NOT bump the
        # hub version -- refresh its dedicated panel here, before the version gate below.
        self._refresh_task_panel()
        version = self.hub.version
        if version == self._last_version:
            return
        self._last_version = version
        namespace = self._expression_namespace()
        for card in self.cards:
            card.refresh(namespace)
        # keep the visible Edit tab's 'now:' acquisition references live, so a
        # queued parameter edit shows as applied once the loop picks it up (one
        # general hook on the current tab -- no per-field wiring).
        editor = self.tabs.currentWidget()
        if isinstance(editor, PanelEditor):
            editor.refresh_node_now_labels()
        self._update_summary()

    def refresh_once(self) -> None:
        """Synchronous refresh (tests / notebooks): one tick regardless of the timer."""
        self._last_version = -1
        self._tick()

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
            who = (getattr(node, "prefix", "") or self._node_label(node) or "node").rstrip(":")
            n = int(getattr(node, "consecutive_errors", 1))
            self.summary.set_danger(True)
            self.summary.setText(f"⚠ NODE ERROR ({who}, ×{n}): {node.last_error}"[:200])
        else:
            self.summary.set_danger(False)
            self.summary.setText(
                f"{len(self.cards)} panels | {n_signals} signals | shot {self.hub.shot}"
                f" | every {self.state.interval_ms} ms")

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
            self.save_button.set_dirty(False)
        except Exception as exc:
            self._message(f"Load failed: {exc}")

    def _message(self, text: str) -> None:
        if os.environ.get("QT_QPA_PLATFORM", "") == "offscreen":
            self.summary.setText(str(text))
            return
        QtWidgets.QMessageBox.information(self, "Task console", str(text))  # pragma: no cover

    # ------------------------------------------------------------------ teardown
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
    window_ratio: float = 0.84,
    title: str = "TaskConsole@Zou lab",
    on_close=None,
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
    window = FluentWindow(widget=console, title=title, hide_on_close=False)
    # Closing the window must tear the console down: the console is a CHILD of the
    # window, so its own closeEvent never fires on a window close -- without this the
    # node owner threads keep running (blocked in camera.acquire, holding the camera /
    # RPyC link), which wedges the kernel.  `closed` fires only on a genuine close, so
    # minimising the dashboard does NOT stop the nodes.
    window.closed.connect(console.shutdown)
    window.adjustSize()
    window.setFixedSize(window.size())
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
