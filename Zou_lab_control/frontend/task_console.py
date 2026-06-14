"""Task console: a configurable grid dashboard of live experiment panels.

Each panel card is JUST the plot frame: the frontend panel canvas plus a thin
border that doubles as the DRAG HANDLE -- grab the border, drop the card, and
it snaps to the layout grid (half-unit pitch, so half-height cards stack flush
against full ones).  Everything else (title, size preset, kind parameters, the
data-source expression, status, remove) lives behind the small gear button in
the panel's top-right corner, confocal-settings style, so the frame size is
fully controlled by the frontend's modular plot region.

A panel's data source is a one-line expression evaluated against the named
signals of a :class:`~Zou_lab_control.neutral_atom.core.signals.SignalHub`
(the same trusted-local-code posture as the pulse GUI's Scan tab):

    value = frame                       # show the latest camera frame
    value = rate_grid - b_rate_grid     # arbitrary math across signals
    value = history('counts', 200).ravel()

Layouts (panels + positions + sizes + expressions + params) save/load as ONE
JSON and are machine-portable: all plot geometry is owned by
frontend.panel_plot and never part of the layout.
"""

from __future__ import annotations

import json
import math as _math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from PyQt5 import QtCore, QtWidgets

from .live import (
    PANEL_SIZES,
    panel_display_size,
    panel_plot,
    panel_size_cells,
)
from .qt_fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    TEXT,
    YELLOW,
    DIVIDER,
    FluentButton,
    FluentCheckBox,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentStatusDot,
    FluentTabWidget,
    FluentWindow,
    add_fluent_shadow,
    ensure_qt_app,
    fluent_widget_stylesheet,
    scaled_px,
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
    "hist": "Distribution",
}

CMAPS = ("inferno", "viridis", "magma", "plasma", "gray", "coolwarm")

_DEFAULT_SOURCES = {
    "2d": "value = frame",
    "sites": "value = occupied",
    "1d": "value = rate_sites",
    "monitor": "value = rate",
    "hist": "value = history('counts', 200).ravel()",
}


class ParamSpec:
    """One declarative panel parameter: the Setting popup generates its widget
    from this spec (confocal style -- adding a parameter is ONE line here, the
    GUI, the JSON params and the plot rebuild all follow)."""

    def __init__(self, key: str, label: str, kind: str, default, *,
                 choices: Sequence[str] = (), lo: float = 0, hi: float = 1e9, tooltip: str = ""):
        self.key = str(key)
        self.label = str(label)
        self.kind = str(kind)              # "choice" | "int" | "signal"
        self.default = default
        self.choices = tuple(choices)
        self.lo, self.hi = lo, hi
        self.tooltip = str(tooltip)

    def get(self, params: Mapping[str, object]):
        return params.get(self.key, self.default)


PANEL_PARAMS: dict[str, tuple[ParamSpec, ...]] = {
    "2d": (
        ParamSpec("cmap", "colormap", "choice", "inferno", choices=CMAPS, tooltip="Image colormap"),
    ),
    "sites": (
        ParamSpec("centers", "centers", "signal", "centers",
                  tooltip="Signal holding the (N, 2) site centers in camera px"),
        ParamSpec("image", "image", "signal", "frame",
                  tooltip="Signal for the camera-frame underlay (blank for none)"),
        ParamSpec("cmap", "colormap", "choice", "viridis", choices=CMAPS, tooltip="Site-value colormap"),
    ),
    # relim is NOT a Setting param: it lives in the Control tab (the processing
    # panel) -- _editor_set_relim writes it back to config.params["relim"] and
    # rebuilds, so a live card's relim is still settable there.
    "1d": (),
    "monitor": (
        ParamSpec("length", "history", "int", 300, lo=20, hi=10_000,
                  tooltip="Rolling history length (shots kept on screen)"),
    ),
    "hist": (
        ParamSpec("bins", "bins", "int", 60, lo=5, hi=500, tooltip="Histogram bins"),
    ),
}

# Curve-fit overlays the CONTROL tab can run on a frozen panel snapshot (pair
# with "Pause Plot" to freeze the trace, fit, and inspect).  label -> DataFigure
# method name; only the line kinds (1d / monitor) support fitting -- hist already
# auto-fits a bimodal, 2d/sites don't apply.  The Setting popup carries NO fit
# controls (basic display only); fitting lives in the Control tab.
FIT_MODELS: dict[str, str] = {
    "Lorentzian": "lorent",
    "Gaussian": "gaussian",
    "Exp decay": "decay",
    "Rabi": "rabi",
}
FITTABLE_KINDS = ("1d", "monitor")

# Basic display toggles the Setting popup persists into config.params and
# RE-APPLIES on every _build_plot (the panel rebuilds each time its data shape
# changes, so a toggle that was not re-applied would silently revert):
#  - colorbar   : show/hide the colorbar band (image kinds only).
#  - unit_index : how many times to cycle the x-axis unit from its original
#                 (reuses DataFigure.change_unit's GHz/nm/MHz | ns/us/ms cycle).
#  - xlim/ylim  : None = auto, or (lo, hi) = manual (autoscale off on that axis).
COLORBAR_KINDS = ("2d", "sites")     # the kinds that own a colorbar to toggle

# Card geometry (raw px, matching the raw-px canvas).  The card is a titled
# frame (title = panel kind) whose FOOTER absorbs the modulus difference: the
# figure margins do not halve with the data area, so the canvases alone can
# never tile -- the frame makes the CARDS tile instead.  A card spans
# rows x _pitch_y - gap, so a 2-row card is EXACTLY two 1-row cards plus the
# gap, and dragged cards snap onto one shared grid.
_CARD_PAD = 10        # horizontal padding canvas<->frame, and the drag border
_CARD_TITLE_H = 32    # the FluentGroupBox title strip
_CARD_VPAD = 12       # vertical padding above + below the content
_FOOTER_MIN = 26      # the status line every card carries under its canvas
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


def _slots_overlap(a, b) -> bool:
    return (a.col < b.col + b.cols and b.col < a.col + a.cols
            and a.row < b.row + b.rows and b.row < a.row + a.rows)


def _resolve_collisions(configs: Sequence["PanelConfig"], active: "PanelConfig") -> bool:
    """Push cards DOWN so no two overlap; ``active`` keeps its slot.

    Cards never overlap on the board: when a drag (or a size change) lands a
    card on occupied slots, the cards underneath move down just far enough to
    clear, cascading.  Pushes only ever increase ``row`` and never touch the
    active card, so the loop terminates.  Returns True when anything moved."""

    moved_any = False
    for _ in range(len(configs) * len(configs) + 1):
        ordered = sorted((c for c in configs if c is not active),
                         key=lambda c: (c.row, c.col))
        placed = [active]
        moved = False
        for config in ordered:
            for blocker in placed:
                if _slots_overlap(config, blocker):
                    config.row = blocker.row + blocker.rows
                    moved = moved_any = True
            placed.append(config)
        if not moved:
            break
    return moved_any


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
    ):
        if kind not in PANEL_KINDS:
            raise ValueError(f"unknown panel kind {kind!r}; choose from {sorted(PANEL_KINDS)}.")
        panel_size_cells(size)              # validate against the limited preset list
        self.kind = str(kind)
        self.title = str(title)
        self.row = max(0, int(row))
        self.col = max(0, int(col))
        self.size = str(size)
        self.source = str(source) if source is not None else _DEFAULT_SOURCES[self.kind]
        self.params = dict(params or {})

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
        )


class TaskConsoleState:
    """The whole console layout: serialised as ONE machine-portable JSON file."""

    schema = "Zou_lab_control.frontend.TaskConsoleState"
    version = 2

    def __init__(
        self,
        *,
        name: str = "task",
        interval_ms: int = 400,
        panels: Sequence[PanelConfig | Mapping[str, object]] | None = None,
    ):
        self.name = str(name)
        self.interval_ms = max(50, int(interval_ms))
        self.panels = [
            panel if isinstance(panel, PanelConfig) else PanelConfig.from_dict(panel)
            for panel in (panels or [])
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "name": self.name,
            "interval_ms": self.interval_ms,
            "panels": [panel.to_dict() for panel in self.panels],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TaskConsoleState":
        return cls(
            name=str(payload.get("name", "task")),
            interval_ms=int(payload.get("interval_ms", 400)),
            panels=payload.get("panels") or [],
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


def _atom_loading_monitor_state() -> TaskConsoleState:
    """The atom-loading dashboard: camera image + occupancy site map on top,
    counts distribution + loading-rate trace in the middle, per-site strip below."""

    return TaskConsoleState(
        name="atom_loading_monitor",
        panels=[
            PanelConfig(kind="2d", title="Loading image", row=0, col=0, size="2x2",
                        source="value = frame"),
            PanelConfig(kind="sites", title="Occupancy", row=0, col=2, size="2x2",
                        source="value = occupied"),
            PanelConfig(kind="hist", title="Counts distribution", row=2, col=0, size="1x2",
                        source="value = history('counts', 200).ravel()", params={"bins": 80}),
            PanelConfig(kind="monitor", title="Loading rate", row=2, col=2, size="1x2",
                        source="value = rate", params={"length": 300}),
            PanelConfig(kind="1d", title="Per-site loading rate", row=3, col=0, size="2x4",
                        source="value = rate_sites"),
        ],
    )


def _loading_rate_live_state() -> TaskConsoleState:
    """A lightweight rate-watch layout (no camera image): rate trace + counts
    distribution on top, per-site rate map + strip below."""

    return TaskConsoleState(
        name="loading_rate_live",
        panels=[
            PanelConfig(kind="monitor", title="Loading rate", row=0, col=0, size="1x2",
                        source="value = rate", params={"length": 500}),
            PanelConfig(kind="hist", title="Counts distribution", row=0, col=2, size="1x2",
                        source="value = history('counts', 200).ravel()", params={"bins": 80}),
            PanelConfig(kind="sites", title="Per-site rate", row=1, col=0, size="2x2",
                        source="value = rate_sites", params={"image": ""}),
            PanelConfig(kind="1d", title="Per-site loading rate", row=1, col=2, size="1x2",
                        source="value = rate_sites"),
        ],
    )


# Named, reusable dashboards ("task GUIs"): designed once, saved by name, and
# loaded back from the GUI preset picker, `--task <name>` or
# `show_task_console(task=...)`.  A JSON file in tasks/ with the same stem
# OVERRIDES the built-in (so the lab can evolve a stock layout in place).
BUILTIN_TASKS: dict[str, object] = {
    "atom_loading_monitor": _atom_loading_monitor_state,
    "loading_rate_live": _loading_rate_live_state,
}


def default_console_state() -> TaskConsoleState:
    return _atom_loading_monitor_state()


def list_task_presets() -> list[str]:
    """All loadable task names: built-ins + every tasks/*.json (deduplicated)."""

    names = set(BUILTIN_TASKS)
    try:
        names.update(path.stem for path in _task_files_dir().glob("*.json"))
    except Exception:
        pass
    return sorted(names)


def resolve_task_state(task: str) -> TaskConsoleState:
    """Resolve a task by FILE PATH, tasks/<name>.json, or built-in name."""

    text = str(task).strip()
    path = Path(text)
    if path.suffix.lower() == ".json" and path.exists():
        return TaskConsoleState.load(path)
    saved = _task_files_dir() / f"{text}.json"
    if saved.exists():
        return TaskConsoleState.load(saved)
    if text in BUILTIN_TASKS:
        return BUILTIN_TASKS[text]()
    raise ValueError(
        f"unknown task {task!r}: not a layout file, not in {_task_files_dir()}, and not one of "
        f"{', '.join(sorted(BUILTIN_TASKS))}.")


# ====================================================================== panels
class PanelCard(FluentGroupBox):
    """One dashboard panel: a titled frame (title = the panel KIND) holding the
    frontend canvas, a status footer, and a text "Setting" button on the title
    strip.  The frame border is the DRAG HANDLE (the matplotlib canvas keeps
    all its own interactions); the footer stretches so the card spans whole
    layout slots -- a 2-row card is exactly two 1-row cards plus the gap."""

    changed = QtCore.pyqtSignal()          # any config edit (console marks dirty)
    layout_changed = QtCore.pyqtSignal()   # size/slot change (console re-arranges)
    remove_requested = QtCore.pyqtSignal(object)
    edit_requested = QtCore.pyqtSignal(object)   # "Edit…" -> open the Control tab

    def __init__(self, config: PanelConfig, parent=None, *, names_provider=None):
        super().__init__(PANEL_KINDS[config.kind], parent, shadow=True)
        self.config = config
        self.names_provider = names_provider   # callable -> live signal names (Setting combo)
        self.plotter = None
        self.canvas = None
        self._value_shape: tuple[int, ...] | None = None
        self._compiled_source = config.source
        self._drag_offset: QtCore.QPoint | None = None
        self.setCursor(QtCore.Qt.OpenHandCursor)   # the frame border drags

        holder = QtWidgets.QVBoxLayout(self)
        holder.setContentsMargins(_CARD_PAD, scaled_px(2), _CARD_PAD, _CARD_VPAD // 2)
        holder.setSpacing(_CARD_VPAD // 2)
        self.canvas_holder = holder
        self.footer = FluentLabel("")
        self.footer.setStyleSheet(f"color: {GREY}; background: transparent;")
        self.footer.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
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
            self.setting_button.move(
                self.width() - self.setting_button.width() - scaled_px(8),
                scaled_px(4))
            self.setting_button.raise_()

    # ------------------------------------------------------------- settings UI
    def _build_settings(self) -> None:
        """The confocal-style settings popup, organised the way an experimenter
        thinks: WHAT to show (Source: pick a live signal, or write an
        expression), HOW to show it (Display: size + colormap + the basic
        display toggles: colorbar, unit, axis limits), and the panel itself
        (title, Remove).  Only BASIC display lives here -- the heavier
        processing (curve fit, command-line fit, relim, units cycle, Save Fig)
        is in the Control tab.  Every widget keeps the stock fluent size."""

        popup = QtWidgets.QFrame(self, QtCore.Qt.Popup)
        # SCOPE the border to the popup itself by objectName: a bare `QFrame { border }`
        # rule cascades to every QFrame-derived child -- and QLabel/QComboBox/QSpinBox
        # internals ARE QFrames, so the unscoped rule drew stray 1px lines around the
        # labels and controls inside.  `#id` keeps the border on the popup only.
        popup.setObjectName("zlcPanelSettings")
        popup.setStyleSheet(
            f"QFrame#zlcPanelSettings {{ background: white; border: 1px solid {DIVIDER};"
            f" border-radius: {scaled_px(6)}px; }}")
        form = QtWidgets.QVBoxLayout(popup)
        pad = scaled_px(10)
        form.setContentsMargins(pad, pad, pad, pad)
        form.setSpacing(scaled_px(6, minimum=4))

        def row(*widgets, stretch_first=True):
            line = QtWidgets.QHBoxLayout()
            line.setSpacing(scaled_px(6, minimum=4))
            for index, widget in enumerate(widgets):
                line.addWidget(widget, 1 if (stretch_first and index == 0) else 0)
            form.addLayout(line)

        def section(text):
            label = FluentLabel(text)
            label.setStyleSheet(f"color: {GREY}; background: transparent; border: none; font-weight: bold;")
            form.addWidget(label)

        def tag(text):
            label = FluentLabel(text)
            label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            return label

        # ---- Source: pick a signal, or write an expression -----------------
        section("Source")
        self.signal_combo = FluentComboBox()
        self.signal_combo.setMinimumWidth(scaled_px(150, minimum=120))
        self.signal_combo.setToolTip(
            "Pick a live signal to show it directly (sets the expression to\n"
            "`value = <signal>`).  Choose (expression) to write your own.")
        self.signal_combo.activated.connect(self._on_signal_pick)
        row(self.signal_combo, tag("signal"), stretch_first=False)

        self.source_edit = FluentLineEdit(self.config.source)
        self.source_edit.setMinimumWidth(scaled_px(340, minimum=280))
        self.source_edit.setStyleSheet(
            self.source_edit.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
        self.source_edit.setToolTip(
            "Panel data source: one line of Python evaluated against the live signals.\n"
            "Assign the result to `value`.  Namespace: every signal name (latest value),\n"
            "history(name, n), latest(name), names(), shot, np, math.")
        self.apply_button = FluentButton("Apply", color=GREEN)
        self.apply_button.clicked.connect(self._apply_source)
        self.source_edit.textChanged.connect(lambda: self.apply_button.set_dirty(True))
        self.source_edit.returnPressed.connect(self._apply_source)
        row(self.source_edit, self.apply_button)

        self.status = FluentLabel("")
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        form.addWidget(self.status)

        # ---- Display: size + the kind's declarative parameters -------------
        section("Display")
        self.size_combo = FluentComboBox()
        self.size_combo.addItems(list(PANEL_SIZES))
        self.size_combo.setCurrentText(self.config.size)
        self.size_combo.setFixedWidth(scaled_px(64, minimum=56))
        self.size_combo.setToolTip("Panel size preset (height x width half-units; 2x2 = the stock plot region)")
        self.size_combo.currentTextChanged.connect(self._on_size)
        display_row = [self.size_combo, tag("size")]
        self.param_widgets: dict[str, QtWidgets.QWidget] = {}
        for spec in PANEL_PARAMS.get(self.config.kind, ()):
            widget = self._make_param_widget(spec)
            self.param_widgets[spec.key] = widget
            display_row.extend([widget, tag(spec.label)])
        row(*display_row, stretch_first=False)

        # colorbar toggle (image kinds only): persisted + re-applied on rebuild
        self.cbar_check = None
        if self.config.kind in COLORBAR_KINDS:
            self.cbar_check = FluentCheckBox("colorbar")
            self.cbar_check.setChecked(bool(self.config.params.get("colorbar", True)))
            self.cbar_check.setToolTip("Show the colorbar band next to the image")
            self.cbar_check.toggled.connect(self._on_colorbar_toggle)
            row(self.cbar_check, stretch_first=False)

        # unit cycle (generic): reuse DataFigure.change_unit's unit families.
        # We store how many cycles from the original; the rebuild re-applies them.
        self.unit_button = FluentButton("Unit", color=GREY)
        self.unit_button.setToolTip(
            "Cycle the x-axis unit (GHz/nm/MHz or ns/us/ms) where the axis label\n"
            "declares one; persisted and re-applied to the live panel")
        self.unit_button.clicked.connect(self._on_unit_cycle)
        self.unit_label = tag(self._current_unit_text())
        row(self.unit_button, self.unit_label, stretch_first=False)

        # axis limits (generic): auto, or manual lo/hi per axis (autoscale off)
        self.lim_combo = FluentComboBox()
        self.lim_combo.addItems(["auto", "manual"])
        self.lim_combo.setFixedWidth(scaled_px(74, minimum=60))
        self.lim_combo.setToolTip("auto = autoscale to the data; manual = fix the x/y limits below")
        manual = self.config.params.get("xlim") is not None or self.config.params.get("ylim") is not None
        self.lim_combo.setCurrentText("manual" if manual else "auto")
        self.lim_combo.currentTextChanged.connect(self._on_lim_mode)
        row(self.lim_combo, tag("limits"), stretch_first=False)

        def _lim_edit(value):
            edit = FluentLineEdit("" if value is None else f"{float(value):.6g}")
            edit.setFixedWidth(scaled_px(72, minimum=56))
            edit.setEnabled(manual)
            edit.editingFinished.connect(self._on_lim_values)
            edit.returnPressed.connect(self._on_lim_values)
            return edit

        xlim = self.config.params.get("xlim") or (None, None)
        ylim = self.config.params.get("ylim") or (None, None)
        self.xmin_edit, self.xmax_edit = _lim_edit(xlim[0]), _lim_edit(xlim[1])
        self.ymin_edit, self.ymax_edit = _lim_edit(ylim[0]), _lim_edit(ylim[1])
        self.lim_edits = (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit)
        row(tag("x"), self.xmin_edit, self.xmax_edit,
            tag("y"), self.ymin_edit, self.ymax_edit, stretch_first=False)

        # ---- Panel: title + edit/save/remove --------------------------------
        section("Panel")
        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.textChanged.connect(self._on_title)
        remove = FluentButton("Remove", color=ORANGE)
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        row(self.title_edit, remove)
        # heavy controls (curve fit, command fit, relim, units cycle, Save Fig)
        # live in the Control tab, not this lightweight popup; Save Fig here is
        # the basic one-click export.
        edit_button = FluentButton("Edit…", color=ACCENT)
        edit_button.setToolTip("Open the Control tab: curve fit, command-line fit, x/y limits, units, relim, Save Fig")
        edit_button.clicked.connect(lambda: (self.settings_popup.hide(), self.edit_requested.emit(self)))
        save_fig = FluentButton("Save Fig", color=GREY)
        save_fig.setToolTip("Save this panel as <title>.png + <title>.npz (timestamped) in tasks/")
        save_fig.clicked.connect(self._save_fig)
        row(edit_button, save_fig, stretch_first=False)

        self.settings_popup = popup

    def _make_param_widget(self, spec: ParamSpec) -> QtWidgets.QWidget:
        """One widget per declarative ParamSpec; edits apply INSTANTLY."""

        current = self.config.params.get(spec.key, spec.default)
        if spec.kind == "choice":
            combo = FluentComboBox()
            combo.addItems(list(spec.choices))
            if str(current) in spec.choices:
                combo.setCurrentText(str(current))
            combo.setToolTip(spec.tooltip)
            combo.currentTextChanged.connect(lambda text, k=spec.key: self._set_param(k, str(text)))
            return combo
        if spec.kind == "int":
            spin = FluentDoubleSpinBox(length=max(4, len(str(int(spec.hi)))), allow_minus=False)
            spin.setDecimals(0)
            spin.setRange(spec.lo, spec.hi)
            spin.setValue(int(current))
            spin.setToolTip(spec.tooltip)
            spin.valueChanged.connect(lambda v, k=spec.key: self._set_param(k, int(v)))
            return spin
        # "signal": a free-form signal name (blank allowed where documented)
        edit = FluentLineEdit(str(current))
        edit.setFixedWidth(scaled_px(96, minimum=80))
        edit.setToolTip(spec.tooltip)
        edit.editingFinished.connect(lambda k=spec.key, w=edit: self._set_param(k, w.text().strip()))
        return edit

    def _open_settings(self) -> None:
        self._refresh_signal_combo()
        anchor = self.setting_button.mapToGlobal(
            QtCore.QPoint(self.setting_button.width(), self.setting_button.height()))
        popup = self.settings_popup
        popup.adjustSize()
        popup.move(anchor.x() - popup.width(), anchor.y() + scaled_px(2))
        popup.show()

    def _refresh_signal_combo(self) -> None:
        """Fill the signal picker with the hub's CURRENT signal names."""
        names = []
        if callable(self.names_provider):
            try:
                names = sorted(str(n) for n in self.names_provider())
            except Exception:
                names = []
        combo = self.signal_combo
        with _signals_blocked(combo):
            combo.clear()
            combo.addItem("(expression)")
            combo.addItems(names)
            # reflect a plain `value = <signal>` source in the picker
            source = (self.config.source or "").strip()
            picked = source[len("value ="):].strip() if source.startswith("value =") else ""
            combo.setCurrentText(picked if picked in names else "(expression)")

    def _on_signal_pick(self, index: int) -> None:
        name = self.signal_combo.currentText()
        if not name or name == "(expression)":
            return
        self.source_edit.setText(f"value = {name}")
        self._apply_source()                      # picking a signal applies instantly

    # ------------------------------------------------ basic display toggles
    def _on_colorbar_toggle(self, checked: bool) -> None:
        """Persist + apply the colorbar toggle (image kinds).  Applied LIVE on
        the current plotter and stored so the next rebuild keeps it."""
        self.config.params["colorbar"] = bool(checked)
        self._apply_display_params()
        if self.canvas is not None:
            self.canvas.draw_idle()
        self.changed.emit()

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
            df.change_unit()            # cycle the live curve + label in place
            self.config.params["unit_index"] = (
                int(self.config.params.get("unit_index", 0) or 0) + 1) % length
            self.changed.emit()
        if hasattr(self, "unit_label"):
            self.unit_label.setText(self._current_unit_text())
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _on_lim_mode(self, mode: str) -> None:
        manual = str(mode) == "manual"
        for edit in getattr(self, "lim_edits", ()):
            edit.setEnabled(manual)
        if not manual:
            self.config.params["xlim"] = None
            self.config.params["ylim"] = None
            self._reset_plot()          # rebuild so autoscale comes back
            self.changed.emit()
        else:
            self._on_lim_values()       # apply whatever is typed (may be partial)

    def _on_lim_values(self, *_args) -> None:
        if self.lim_combo.currentText() != "manual":
            return

        def pair(lo_edit, hi_edit):
            try:
                lo, hi = float(lo_edit.text()), float(hi_edit.text())
            except (TypeError, ValueError):
                return None
            return (lo, hi) if lo != hi else None

        self.config.params["xlim"] = pair(self.xmin_edit, self.xmax_edit)
        self.config.params["ylim"] = pair(self.ymin_edit, self.ymax_edit)
        self._apply_display_params()
        if self.canvas is not None:
            self.canvas.draw_idle()
        self.changed.emit()

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

    def set_status(self, text: str, *, error: bool) -> None:
        # Text changes every tick ("shot N") -- but the COLOUR/stylesheet only
        # changes on the ok<->error transition.  Restyle only on that transition
        # (rebuilding the same stylesheet string every tick was pure waste);
        # appearance-neutral because the colour is identical when error is.
        self.status.setText(str(text)[:200])
        self.footer.setText(f"{text}   ·   {self.config.source}"[:160])
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
        self.config.title = str(text)
        if self.plotter is not None and getattr(self.plotter, "ax", None) is not None:
            self.plotter.ax.set_title(self.config.title)
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
        """Evaluate this panel's source against ``namespace`` and feed the plot.
        Every failure lands on the gear/status -- a bad expression in one panel
        must never break the console or its siblings."""

        try:
            value = self._evaluate(dict(namespace))
            self._feed(value, namespace)
        except Exception as exc:
            self.set_status(str(exc).splitlines()[0][:160] or type(exc).__name__, error=True)
            return
        shot = namespace.get("shot")
        self.set_status(f"shot {int(shot)}" if isinstance(shot, (int, float)) else "ok", error=False)

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
            # flagged xy=True (the measurement result card built by
            # _ensure_result_panel) reads an (N, 2) value as an x-y curve
            # (col0 = x, col1 = y, e.g. value = column_stack([x_key, y_key])).
            # A plain 1d panel that happens to produce (N, 2) still flattens, so
            # the x-y meaning is opt-in, not a silent shape heuristic.
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
        if kind == "monitor":
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
        """The site-map panel's two auxiliary signals: centers + optional image."""
        centers_name = str(self.config.params.get("centers", "centers")).strip()
        image_name = str(self.config.params.get("image", "frame")).strip()
        centers = namespace.get(centers_name) if centers_name else None
        if centers is None:
            raise ValueError(
                f"site-map panel needs a `{centers_name or 'centers'}` signal with the (N, 2) site centers"
                " (set its name in Setting -> centers)")
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] < 2:
            raise ValueError(f"centers signal must have shape (N, 2); got {centers.shape}")
        image = namespace.get(image_name) if image_name else None
        return centers[:, :2], (None if image is None else np.asarray(image, dtype=float))

    def _feed(self, value, namespace: Mapping[str, object] | None = None) -> None:
        # PERFORMANCE: feed data with draw=False and queue ONE draw_idle per
        # panel -- rendering happens in Qt's paint pass (coalesced per frame),
        # never synchronously inside the refresh tick.  hist accepts any sample
        # count (HistogramFigure.update rebins itself), so a growing history
        # must NOT rebuild the whole plot every shot.
        value = self._coerce(value)
        kind = self.config.kind
        rebuild = self.plotter is None
        if not rebuild and kind in ("2d", "1d", "sites"):
            rebuild = tuple(np.shape(value)) != self._value_shape
        if rebuild:
            self._build_plot(value, namespace)
            self._value_shape = (1,) if isinstance(value, float) else tuple(np.shape(value))
            return
        if kind == "2d":
            self.plotter.update(np.asarray(value).ravel(), draw=False)
        elif kind == "monitor":
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
            spacing = 6.0
            if len(centers) > 1:
                deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
                deltas = deltas[deltas > 0]
                if deltas.size:
                    spacing = float(np.min(deltas))
            self.plotter = panel_plot(
                centers, vec, kind="sites", size=size,
                cmap=str(self.config.params.get("cmap", "viridis")),
                image=image, roi_radius=max(1.5, 0.3 * spacing),
                labels=("Camera x (px)", "Camera y (px)", label),
                title=self.config.title or None)
        elif kind == "2d":
            arr = np.asarray(value, dtype=float)
            ny, nx = arr.shape
            xx, yy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
            data_x = np.column_stack([xx.ravel(), yy.ravel()])
            self.plotter = panel_plot(
                data_x, arr.ravel(), kind="2d", size=size,
                cmap=str(self.config.params.get("cmap", "inferno")),
                labels=("X (px)", "Y (px)", ""), title=self.config.title or None)
        elif kind == "monitor":
            length = max(20, int(self.config.params.get("length", 300)))
            history = np.full(length, np.nan)
            self.plotter = panel_plot(
                np.arange(length, dtype=float), history, kind="monitor", size=size,
                labels=("Shots ago", label, "Z"),
                relim_mode=str(self.config.params.get("relim", "tight")),
                title=self.config.title or None)
            self.plotter.roll(float(value), draw=False)
        elif kind == "hist":
            self.plotter = panel_plot(
                np.asarray(value, dtype=float), kind="hist", size=size,
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
                xlabel, ylabel = "Site", label
            self.plotter = panel_plot(
                data_x, vec, kind="1d", size=size,
                labels=(xlabel, ylabel, "Z"),
                relim_mode=str(self.config.params.get("relim", "tight")),
                title=self.config.title or None)
        self.canvas = panel_canvas(self.plotter.fig)
        self.canvas_holder.insertWidget(0, self.canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        if self.canvas_holder.indexOf(self.footer) < 0:
            # footer (status line) sits DIRECTLY under the canvas, then the
            # stretch absorbs the multi-row tiling slack below it -- so the slack
            # reads as a clean bottom margin rather than a gap floating between
            # the plot and its status line.
            self.canvas_holder.addWidget(self.footer)
            self.canvas_holder.addStretch(1)
        # re-apply the persisted basic display toggles to the FRESH plotter:
        # the panel rebuilds whenever its data shape changes, so a toggle that
        # was not re-applied here would silently revert every such rebuild.
        self._apply_display_params()
        self.canvas.draw_idle()
        self._place_setting_button()

    # --------------------------------------------- basic display application
    def _apply_display_params(self) -> None:
        """Apply the persisted basic display toggles (colorbar, unit, x/y limits)
        to the current plotter -- called after every rebuild and on each edit."""
        if self.plotter is None:
            return
        if self.config.kind in COLORBAR_KINDS and hasattr(self.plotter, "set_colorbar_visible"):
            self.plotter.set_colorbar_visible(bool(self.config.params.get("colorbar", True)))
        self._apply_unit()
        self._apply_limits()

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

    def _apply_limits(self) -> None:
        """Apply manual x/y limits (autoscale off on the fixed axis); None = auto.

        For a live line panel (1d / monitor) the per-tick relim would re-autoscale
        y, so a manual ylim also pins relim_mode='off' and the plotter's frozen
        ylim_min/ylim_max to the requested range (relim then re-applies it)."""
        ax = getattr(self.plotter, "ax", None)
        if ax is None:
            return
        xlim = self.config.params.get("xlim")
        ylim = self.config.params.get("ylim")
        if xlim is not None:
            ax.set_autoscalex_on(False)
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if ylim is not None:
            lo, hi = float(ylim[0]), float(ylim[1])
            ax.set_autoscaley_on(False)
            ax.set_ylim(lo, hi)
            if hasattr(self.plotter, "relim_mode"):
                self.plotter.relim_mode = "off"
                self.plotter.ylim_min, self.plotter.ylim_max = lo, hi
        elif hasattr(self.plotter, "relim_mode") and self.plotter.relim_mode == "off":
            # ylim cleared back to auto: restore the panel's normal relim mode
            self.plotter.relim_mode = str(self.config.params.get("relim", "tight"))

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
    """The Control tab's Measurement section: pick a measurement, set its
    declared parameters in an AUTO-GENERATED + validated form, Start (one click)
    to stream the scan into a Monitor panel, Stop to cooperatively cancel.

    The form is built entirely from a :class:`MeasurementSpec`'s ``params``
    (the single source of truth -- the API default and this control derive from
    one declaration): each :class:`ParamDecl` becomes a widget chosen by its
    ``kind`` (float/int -> spin box, axis_range -> min/max/points triplet,
    bool -> checkbox, choice -> combo).  Values are read back BY KIND and
    coerced -- there is NO free-text eval (the confocal-GUI lesson).  ``required``
    params with an empty value disable Start with a hint.
    """

    start_requested = QtCore.pyqtSignal(object)    # emits the chosen MeasurementSpec
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, measurements: Sequence[object], parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._specs = list(measurements)
        self._widgets: dict[str, object] = {}     # param key -> widget (or tuple for axis_range)
        self._running = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(scaled_px(6, minimum=4))

        # measurement type picker
        pick_row = QtWidgets.QHBoxLayout()
        pick_row.setSpacing(scaled_px(6, minimum=4))
        self.type_combo = FluentComboBox()
        self.type_combo.setMinimumWidth(scaled_px(220, minimum=160))
        for index, spec in enumerate(self._specs):
            self.type_combo.addItem(str(spec.name), index)
        self.type_combo.setToolTip("Pick a measurement; its scanned parameters appear below.")
        self.type_combo.activated.connect(lambda *_: self._rebuild_form())
        pick_row.addWidget(self._tag("measurement"))
        pick_row.addWidget(self.type_combo, 1)
        root.addLayout(pick_row)

        # auto-generated parameter form (rebuilt when the type changes)
        self.form = QtWidgets.QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.form.setHorizontalSpacing(scaled_px(8, minimum=5))
        self.form.setVerticalSpacing(scaled_px(5, minimum=3))
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
        root.addLayout(action_row)

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
            label_text = decl.label + (f" ({decl.unit})" if decl.unit else "")
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
                    edit.setFixedWidth(scaled_px(96, minimum=80))
                    edit.setPlaceholderText("(all)")
                    edit.setToolTip(decl.tooltip)
                    edit.textChanged.connect(self._refresh_start_enabled)
                    self.form.addRow(label_text, edit)
                    self._widgets[decl.key] = ("int_opt", edit)
                else:
                    spin = self._spin(decl, integer=True)
                    self.form.addRow(label_text, spin)
                    self._widgets[decl.key] = ("int", spin)
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
            else:  # float
                values[key] = float(entry[1].value())
        return values

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
            elif entry[0] == "int_opt" and not entry[1].text().strip():
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
        self.start_requested.emit(spec)

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


# ====================================================================== console
class TaskConsole(QtWidgets.QWidget):
    """The dashboard window body: header bar + drag-and-snap panel board."""

    def __init__(
        self,
        *,
        hub,
        state: TaskConsoleState | None = None,
        feeds: Sequence[object] = (),
        measurements: Sequence[object] = (),
        scale: float | None = None,
        window_ratio: float = 0.84,
        window_px: tuple[int, int] | None = None,
    ):
        ensure_qt_app()
        set_fluent_scale(scale)
        super().__init__()
        self.hub = hub
        self.feeds = list(feeds)
        # The declarative measurement catalog (P5): with none, the Measurement
        # section stays a disabled placeholder; with specs it becomes the
        # auto-generated form + one-click Start.
        self.measurements = list(measurements)
        self._meas_feed = None          # the ScannedMeasurementFeed of the active run
        self._meas_spec = None
        self.state = state or default_console_state()
        self.window_ratio = float(window_ratio)
        self._window_px = window_px
        self.cards: list[PanelCard] = []
        self._last_version = -1
        self._building = False
        self._address: str | None = None
        # Control-tab state: a FROZEN snapshot plot of the card being edited,
        # plus a DataFigure for fit / limits / units / save.
        self._editor_card: PanelCard | None = None
        self._editor_plotter = None
        self._editor_canvas = None
        self._editor_df = None

        self._build_ui()
        self.load_state(self.state)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.state.interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------ UI
    def _target_console_size(self) -> QtCore.QSize:
        """Window size from the primary screen (the pulse-editor sizing rule)."""

        if self._window_px is not None:
            return QtCore.QSize(int(self._window_px[0]), int(self._window_px[1]))
        app = QtWidgets.QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return QtCore.QSize(scaled_px(1280, minimum=960), scaled_px(760, minimum=620))
        available = screen.availableGeometry()
        titlebar_allowance = scaled_px(36, minimum=28)
        margin_w = scaled_px(40, minimum=28)
        margin_h = scaled_px(48, minimum=32)
        max_w = max(360, available.width() - margin_w)
        max_h = max(320, available.height() - titlebar_allowance - margin_h)
        min_w = min(scaled_px(980, minimum=820), max_w)
        min_h = min(scaled_px(640, minimum=560), max_h)
        desired_w = min(max_w, int(available.width() * self.window_ratio))
        desired_h = min(max_h, int(available.height() * self.window_ratio) - titlebar_allowance)
        return QtCore.QSize(max(min_w, desired_w), max(min_h, desired_h))

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
        self.preset_combo = FluentComboBox()
        self.preset_combo.setFixedWidth(scaled_px(170, minimum=130))
        self.preset_combo.setToolTip(
            "Named task dashboards: built-ins plus every layout saved in tasks/.\n"
            "Picking one loads it (Save stores the current design under its name).")
        self._refresh_presets()
        self.preset_combo.activated.connect(self._on_preset_pick)
        self.summary = FluentLineEdit("")
        self.summary.setEnabled(False)

        self.kind_combo = FluentComboBox()
        for key, label in PANEL_KINDS.items():
            self.kind_combo.addItem(label, key)
        self.kind_combo.setFixedWidth(scaled_px(108, minimum=92))
        add_button = FluentButton("Add Panel", color=ACCENT)
        add_button.clicked.connect(self._add_panel)
        # Two independent freezes (see _toggle_pause / _toggle_measurement):
        #  Pause Plot  -> stop refreshing the panels; data still flows into the hub,
        #                 so you can freeze the display and fit / inspect carefully.
        #  Pause Meas. -> stop the feeds (the experiment data source) themselves.
        self.pause_button = FluentButton("Pause Plot", color=ACCENT)
        self.pause_button.setToolTip("Freeze the panel display (data keeps arriving) — fit / inspect, then Resume")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.meas_button = FluentButton("Pause Meas.", color=ACCENT)
        self.meas_button.setToolTip("Pause the measurement feeds (stop producing new shots)")
        self.meas_button.clicked.connect(self._toggle_measurement)
        self.meas_button.setEnabled(bool(self._controllable_feeds()))
        self.save_button = FluentButton("Save", color=ACCENT)
        self.save_button.clicked.connect(self.save_to_file)
        load_button = FluentButton("Load", color=ORANGE)
        load_button.clicked.connect(self.load_from_file)

        for widget in (self.status_dot, self.name_edit, self.preset_combo):
            header.addWidget(widget)
        header.addWidget(self.summary, 1)
        for widget in (self.kind_combo, add_button, self.pause_button, self.meas_button,
                       self.save_button, load_button):
            header.addWidget(widget)
        root.addWidget(header_frame)

        # Two tabs (like the pulse GUI): Monitor = the live drag-and-snap panel
        # grid; Control = the per-panel processing/save panel (curve fit,
        # command fit, x/y limits, units, relim, Save Fig) plus the Measurement
        # section (P5) -- the heavier controls that would overload the
        # lightweight Setting popup.
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
        self.tabs.addTab(dash_tab, "Monitor")
        self.tabs.addTab(self._build_editor_tab(), "Control")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, 1)

    # ------------------------------------------------------------------ state
    def read_state(self) -> TaskConsoleState:
        return TaskConsoleState(
            name=self.name_edit.text().strip() or "task",
            interval_ms=self.state.interval_ms,
            panels=[card.config for card in self.cards],
        )

    def load_state(self, state: TaskConsoleState) -> None:
        self._building = True
        try:
            self.state = state
            for card in self.cards:
                card.shutdown()
                card.setParent(None)
                card.deleteLater()
            self.cards = []
            self.name_edit.setText(state.name)
            for config in state.panels:
                self._attach_card(PanelCard(config, parent=self.board,
                                            names_provider=self.hub.names))
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
    def _build_editor_tab(self) -> QtWidgets.QWidget:
        """The Control tab: the per-panel PROCESSING / SAVE panel (confocal
        style).  A frozen snapshot of one panel plus the heavier controls
        (command-line fit, x/y limits, unit cycle, relim, Save Fig) that are
        kept off the lightweight Setting popup -- organised into labelled
        sections, with a Measurement section reserved at the top for P5."""
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        outer = QtWidgets.QVBoxLayout(page)
        m = scaled_px(8, minimum=5)
        outer.setContentsMargins(m, m, m, m)
        outer.setSpacing(scaled_px(6, minimum=4))

        def section(text):
            label = FluentLabel(text)
            label.setStyleSheet(f"color: {GREY}; background: transparent; border: none; font-weight: bold;")
            outer.addWidget(label)
            return label

        # ---- Measurement (P5): pick -> auto-form -> one-click Start ---------
        # With NO measurements the group stays a disabled placeholder so the
        # Control tab keeps its intended top-down shape (Measurement ->
        # Processing -> Axes); with specs the placeholder is replaced by the
        # auto-generated, validated parameter form + Start/Stop.
        self.measurement_group = FluentGroupBox("Measurement", shadow=False)
        meas_box = QtWidgets.QVBoxLayout(self.measurement_group)
        meas_box.setContentsMargins(scaled_px(10), scaled_px(8), scaled_px(10), scaled_px(8))
        self.measurement_placeholder = None
        self.measurement_panel = None
        if self.measurements:
            self.measurement_group.setEnabled(True)
            self.measurement_panel = MeasurementPanel(self.measurements)
            self.measurement_panel.start_requested.connect(self._start_measurement)
            self.measurement_panel.stop_requested.connect(self._stop_measurement)
            meas_box.addWidget(self.measurement_panel)
        else:
            self.measurement_group.setEnabled(False)
            self.measurement_placeholder = FluentLabel(
                "Measurement controls land here: pick a measurement, set its scanned\n"
                "parameters, Start to stream the scan into a Monitor panel.")
            self.measurement_placeholder.setStyleSheet(
                f"color: {GREY}; background: transparent; border: none;")
            meas_box.addWidget(self.measurement_placeholder)
        outer.addWidget(self.measurement_group)

        # ---- Processing: the bound panel snapshot + fit / axes / save ------
        section("Processing")
        head = QtWidgets.QHBoxLayout()
        self.ed_title = FluentLabel("Control — select a panel's Edit… button")
        self.ed_title.setStyleSheet(f"color: {TEXT}; background: transparent; border: none; font-weight: bold;")
        self.ed_refresh = FluentButton("Refresh", color=GREY)
        self.ed_refresh.setToolTip("Re-snapshot the panel's current data into the Control tab")
        self.ed_refresh.clicked.connect(lambda: self._edit_card(self._editor_card) if self._editor_card else None)
        self.ed_savefig = FluentButton("Save Fig", color=ACCENT)
        self.ed_savefig.setToolTip("Save the edited figure (png) + data (npz), timestamped, into tasks/")
        self.ed_savefig.clicked.connect(self._editor_save)
        head.addWidget(self.ed_title, 1)
        head.addWidget(self.ed_refresh)
        head.addWidget(self.ed_savefig)
        outer.addLayout(head)

        self.ed_canvas_holder = QtWidgets.QVBoxLayout()
        self.ed_canvas_holder.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.ed_canvas_holder)

        def labeled(text):
            lab = FluentLabel(text)
            lab.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            return lab

        section("Fit & axes")
        # Fit row: model + free-text command (confocal ":" params) + Fit + Clear
        fit_row = QtWidgets.QHBoxLayout()
        self.ed_fit_combo = FluentComboBox()
        self.ed_fit_combo.addItems(list(FIT_MODELS))
        self.ed_fit_combo.setFixedWidth(scaled_px(120, minimum=96))
        self.ed_fit_cmd = FluentLineEdit("")
        self.ed_fit_cmd.setPlaceholderText("fit args, e.g.  p0=[1,0,1,0], B=0.1, is_fit=False")
        self.ed_fit_cmd.setStyleSheet(self.ed_fit_cmd.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
        self.ed_fit_cmd.setToolTip(
            "Optional fit arguments injected into the call (trusted local code):\n"
            "p0=[...] initial guess; NAME=value fixes a named parameter; is_fit=False just overlays p0.")
        self.ed_fit_cmd.returnPressed.connect(self._editor_do_fit)
        ed_fit = FluentButton("Fit", color=ACCENT)
        ed_fit.clicked.connect(self._editor_do_fit)
        ed_clear = FluentButton("Clear", color=GREY)
        ed_clear.clicked.connect(self._editor_clear_fit)
        fit_row.addWidget(labeled("Fit"))
        fit_row.addWidget(self.ed_fit_combo)
        fit_row.addWidget(self.ed_fit_cmd, 1)
        fit_row.addWidget(ed_fit)
        fit_row.addWidget(ed_clear)
        outer.addLayout(fit_row)

        # Axes row: x/y limits + Apply + Unit cycle + relim
        ax_row = QtWidgets.QHBoxLayout()
        self.ed_xmin = FluentLineEdit(""); self.ed_xmax = FluentLineEdit("")
        self.ed_ymin = FluentLineEdit(""); self.ed_ymax = FluentLineEdit("")
        for w in (self.ed_xmin, self.ed_xmax, self.ed_ymin, self.ed_ymax):
            w.setFixedWidth(scaled_px(72, minimum=56))
            w.returnPressed.connect(self._editor_apply_limits)
        ed_apply = FluentButton("Apply lim", color=ACCENT)
        ed_apply.clicked.connect(self._editor_apply_limits)
        ed_unit = FluentButton("Unit", color=GREY)
        ed_unit.setToolTip("Cycle the x-axis unit (GHz/nm/MHz or ns/us/ms) where defined")
        ed_unit.clicked.connect(self._editor_unit)
        self.ed_relim = FluentComboBox()
        self.ed_relim.addItems(["tight", "normal"])
        self.ed_relim.setFixedWidth(scaled_px(74, minimum=60))
        self.ed_relim.activated.connect(self._editor_set_relim)
        ax_row.addWidget(labeled("x"))
        ax_row.addWidget(self.ed_xmin); ax_row.addWidget(self.ed_xmax)
        ax_row.addWidget(labeled("y"))
        ax_row.addWidget(self.ed_ymin); ax_row.addWidget(self.ed_ymax)
        ax_row.addWidget(ed_apply)
        ax_row.addWidget(ed_unit)
        ax_row.addWidget(labeled("relim"))
        ax_row.addWidget(self.ed_relim)
        ax_row.addStretch(1)
        outer.addLayout(ax_row)

        self.ed_status = FluentLabel("")
        self.ed_status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        outer.addWidget(self.ed_status)
        outer.addStretch(1)
        self._editor_controls = [self.ed_fit_combo, self.ed_fit_cmd, ed_fit, ed_clear,
                                 self.ed_xmin, self.ed_xmax, self.ed_ymin, self.ed_ymax,
                                 ed_apply, ed_unit, self.ed_relim, self.ed_savefig, self.ed_refresh]
        for w in self._editor_controls:
            w.setEnabled(False)
        return page

    def _edit_card(self, card: "PanelCard") -> None:
        """Bind the editor to a card (snapshot its current data) and show the tab."""
        if card is None or card.plotter is None:
            self._message("Open a panel with data first (the panel must be showing a plot).")
            return
        self._editor_card = card
        self._editor_rebuild()
        self.ed_title.setText(f"Control — {card.config.title or PANEL_KINDS[card.config.kind]}"
                              f"  ({PANEL_KINDS[card.config.kind]})")
        with _signals_blocked(self.ed_relim):
            self.ed_relim.setCurrentText(str(card.config.params.get("relim", "tight")))
        for w in self._editor_controls:
            w.setEnabled(True)
        self.tabs.setCurrentWidget(self.tabs.widget(1))

    def _editor_teardown(self) -> None:
        if self._editor_canvas is not None:
            self.ed_canvas_holder.removeWidget(self._editor_canvas)
            self._editor_canvas.deleteLater()
        if self._editor_plotter is not None and plt is not None and self._editor_plotter.fig is not None:
            plt.close(self._editor_plotter.fig)
        self._editor_canvas = None
        self._editor_plotter = None
        self._editor_df = None

    def _editor_rebuild(self) -> None:
        """Build a fresh, larger snapshot plot of the bound card's CURRENT data."""
        card = self._editor_card
        if card is None or card.plotter is None or panel_canvas is None:
            return
        self._editor_teardown()
        src = card.plotter
        kind = card.config.kind
        size = "2x4"
        title = card.config.title or PANEL_KINDS[kind]
        relim = str(card.config.params.get("relim", "tight"))
        cmap = str(card.config.params.get("cmap", "inferno" if kind == "2d" else "viridis"))
        try:
            if kind == "2d":
                self._editor_plotter = panel_plot(np.array(src.data_x, dtype=float),
                                                  np.array(src.data_y[:, 0], dtype=float), kind="2d",
                                                  size=size, cmap=cmap, labels=tuple(src.labels), title=title)
            elif kind == "sites":
                self._editor_plotter = panel_plot(np.array(src.data_x[:, :2], dtype=float),
                                                  np.array(src.data_y[:, 0], dtype=float), kind="sites",
                                                  size=size, cmap=cmap, image=getattr(src, "background", None),
                                                  roi_radius=getattr(src, "roi_radius", 3.0),
                                                  labels=tuple(src.labels), title=title)
            elif kind == "hist":
                self._editor_plotter = panel_plot(np.array(src.values, dtype=float), kind="hist", size=size,
                                                  bins=int(card.config.params.get("bins", 60)),
                                                  labels=tuple(src.labels), title=title)
            else:  # 1d / monitor -> a line snapshot
                self._editor_plotter = panel_plot(np.array(src.data_x[:, 0], dtype=float),
                                                  np.array(src.data_y[:, 0], dtype=float),
                                                  kind=kind, size=size, relim_mode=relim,
                                                  labels=tuple(src.labels), title=title)
        except Exception as exc:
            self.ed_status.setText(f"could not snapshot: {str(exc).splitlines()[0][:120]}")
            return
        self._editor_canvas = panel_canvas(self._editor_plotter.fig)
        self.ed_canvas_holder.addWidget(self._editor_canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self._editor_canvas.draw_idle()
        self._editor_df = None
        self._editor_fill_limits()
        self.ed_status.setText("snapshot of current data — fit / set limits / save are frozen here")

    def _editor_df_for(self):
        from .data_figure import DataFigure
        if self._editor_df is None:
            self._editor_df = DataFigure(self._editor_plotter)
        return self._editor_df

    def _editor_do_fit(self) -> None:
        if self._editor_plotter is None:
            return
        method = FIT_MODELS.get(self.ed_fit_combo.currentText())
        if method is None:
            return
        cmd = self.ed_fit_cmd.text().strip()
        try:
            df = self._editor_df_for()
            if df.fit is not None:
                df.clear()
            # trusted-local-tool posture (same as the Scan tab's exec): the args
            # the user types are injected into the fit call -- p0/fixed-params/is_fit.
            call = f"_df.{method}({cmd})" if cmd else f"_df.{method}(is_display=True)"
            result, _ = eval(call, {"_df": df, "np": np})  # noqa: S307 - local experiment tool
            self._editor_canvas.draw_idle()
            popt = getattr(result, "popt", None)
            names = getattr(result, "names", None) or []
            if popt is None:
                self.ed_status.setText(f"fit {self.ed_fit_combo.currentText()}: did not converge")
                return
            self.ed_status.setText(f"fit {self.ed_fit_combo.currentText()}: "
                                   + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, popt)))
        except Exception as exc:
            self.ed_status.setText(f"fit failed: {str(exc).splitlines()[0][:140]}")

    def _editor_clear_fit(self) -> None:
        if self._editor_df is not None:
            try:
                self._editor_df.clear()
            except Exception:
                pass
            self._editor_df = None
            if self._editor_canvas is not None:
                self._editor_canvas.draw_idle()
            self.ed_status.setText("fit cleared")

    def _editor_fill_limits(self) -> None:
        if self._editor_plotter is None:
            return
        ax = getattr(self._editor_plotter, "ax", None)
        if ax is None:
            return
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        with _signals_blocked(self.ed_xmin, self.ed_xmax, self.ed_ymin, self.ed_ymax):
            self.ed_xmin.setText(f"{xlo:.6g}"); self.ed_xmax.setText(f"{xhi:.6g}")
            self.ed_ymin.setText(f"{ylo:.6g}"); self.ed_ymax.setText(f"{yhi:.6g}")

    def _editor_apply_limits(self) -> None:
        if self._editor_plotter is None:
            return
        try:
            df = self._editor_df_for()
            df.xlim(float(self.ed_xmin.text()), float(self.ed_xmax.text()))
            df.ylim(float(self.ed_ymin.text()), float(self.ed_ymax.text()))
            self._editor_canvas.draw_idle()
            self.ed_status.setText("limits applied")
        except Exception as exc:
            self.ed_status.setText(f"bad limits: {str(exc).splitlines()[0][:100]}")

    def _editor_unit(self) -> None:
        if self._editor_plotter is None:
            return
        try:
            self._editor_df_for().change_unit()
            self._editor_canvas.draw_idle()
            self._editor_fill_limits()
            self.ed_status.setText("unit cycled")
        except Exception as exc:
            self.ed_status.setText(f"unit cycle n/a: {str(exc).splitlines()[0][:100]}")

    def _editor_set_relim(self, *_args) -> None:
        if self._editor_card is None:
            return
        # persist on the card config (remembered + saved) then rebuild the EDITOR
        # snapshot only -- do NOT tear the live card down here (that would null its
        # plotter and the editor reads from it).
        self._editor_card.config.params["relim"] = self.ed_relim.currentText()
        self._mark_dirty()
        self._editor_rebuild()

    def _editor_save(self) -> None:
        if self._editor_plotter is None:
            return
        try:
            df = self._editor_df_for()
            stem = (self._editor_card.config.title or self._editor_card.config.kind).strip() or "panel"
            out = df.save(_task_files_dir() / stem,
                          extra_info={"source": self._editor_card.config.source,
                                      "kind": self._editor_card.config.kind})
            self.ed_status.setText(f"saved {out['figure'].name} + {out['data'].name}")
        except Exception as exc:
            self.ed_status.setText(f"save failed: {str(exc).splitlines()[0][:120]}")

    def _on_tab_changed(self, _index: int) -> None:
        # refresh the editor snapshot when its tab is shown (data may have moved)
        if self.tabs.currentIndex() == 1 and self._editor_card is not None:
            self._editor_fill_limits()

    def _arrange(self) -> None:
        # the card that was just dragged / resized wins its slot; everyone it
        # now overlaps is pushed down (cards never overlap on the board)
        active = self.sender()
        if isinstance(active, PanelCard) and len(self.cards) > 1:
            if _resolve_collisions([c.config for c in self.cards], active.config):
                self._mark_dirty()
        self.board.arrange(self.cards)
        self._update_summary()

    # ------------------------------------------------------------------ actions
    def _add_panel(self) -> None:
        kind = self.kind_combo.currentData() or "1d"
        rows = max((c.config.row + c.config.rows for c in self.cards), default=0)
        config = PanelConfig(kind=str(kind), row=rows, col=0, size="1x2")
        card = PanelCard(config, parent=self.board, names_provider=self.hub.names)
        self._attach_card(card)
        self._arrange()
        self._mark_dirty()

    def _remove_panel(self, card: PanelCard) -> None:
        if card in self.cards:
            card.settings_popup.hide()
            self.cards.remove(card)
            card.shutdown()
            card.setParent(None)
            card.deleteLater()
            self._arrange()
            self._mark_dirty()

    def _toggle_pause(self) -> None:
        """Pause/resume the PLOT refresh: freezes the display while data keeps
        flowing into the hub, so you can fit and inspect a frozen trace."""
        if self._timer.isActive():
            self._timer.stop()
            self.pause_button.setText("Resume Plot")
            self.status_dot.set_color(YELLOW)
        else:
            self._timer.start()
            self.pause_button.setText("Pause Plot")
            self.status_dot.set_color(GREEN if self._feeds_running() else GREY)

    def _controllable_feeds(self) -> list:
        """Feeds the console can start/stop (have start+stop+running).

        A feed that reports ``finished`` (a completed ScannedMeasurementFeed) is
        excluded: re-starting it would clear its stop event and spawn a daemon
        thread whose first shot() raises StopIteration, leaving the loop spinning
        idle forever.  So "Resume Meas." never restarts an exhausted scan."""
        return [f for f in self.feeds
                if all(hasattr(f, attr) for attr in ("start", "stop", "running"))
                and not getattr(f, "finished", False)]

    def _feeds_running(self) -> bool:
        feeds = self._controllable_feeds()
        return bool(feeds) and any(getattr(f, "running", False) for f in feeds)

    def _toggle_measurement(self) -> None:
        """Pause/resume the MEASUREMENT: stop/start the feeds (the experiment data
        source) themselves -- distinct from freezing the plot."""
        feeds = self._controllable_feeds()
        if not feeds:
            return
        if self._feeds_running():
            for feed in feeds:
                try:
                    feed.stop()
                except Exception:
                    pass
            self.meas_button.setText("Resume Meas.")
            if self._timer.isActive():
                self.status_dot.set_color(GREY)   # plot live but no new data
        else:
            for feed in feeds:
                try:
                    feed.start(rate_hz=getattr(feed, "rate_hz", 5.0))
                except Exception:
                    pass
            self.meas_button.setText("Pause Meas.")
            self.status_dot.set_color(GREEN if self._timer.isActive() else YELLOW)

    # --------------------------------------------------------- measurement (P5)
    def _measurement_source(self, spec) -> str:
        """The result panel's data source: an (N, 2) x-y curve grown by the feed
        (col0 = x_key, col1 = y_key).  Both keys arrive together each shot."""
        return (f"value = np.column_stack([np.atleast_1d({spec.x_key}), "
                f"np.atleast_1d({spec.y_key})]) if '{spec.x_key}' in names() else np.zeros((1, 2))")

    def _ensure_result_panel(self, spec) -> PanelCard:
        """Find (or create) the 1-D Monitor panel that shows ``spec``'s curve.

        Reuses an existing card whose source already targets this spec's result
        signals so repeated Start clicks update one panel rather than piling up;
        otherwise adds a 1x2 line panel at the bottom of the board with the
        spec's axis labels."""
        source = self._measurement_source(spec)
        for card in self.cards:
            if card.config.kind == "1d" and card.config.params.get("measurement") == spec.name:
                return card
        rows = max((c.config.row + c.config.rows for c in self.cards), default=0)
        xlabel, ylabel = spec.result_labels
        config = PanelConfig(
            kind="1d", title=spec.name, row=rows, col=0, size="1x2", source=source,
            params={"measurement": spec.name, "xy": True,
                    "xlabel": xlabel, "ylabel": ylabel, "relim": "normal"},
        )
        card = PanelCard(config, parent=self.board, names_provider=self.hub.names)
        self._attach_card(card)
        self._arrange()
        self._mark_dirty()
        return card

    def _start_measurement(self, spec) -> None:
        """One-click Start: build the measurement, wrap it in a feed, ensure a
        Monitor result panel, register the feed, and run it."""
        panel = self.measurement_panel
        if panel is None:
            return
        if self._meas_feed is not None and getattr(self._meas_feed, "running", False):
            panel.set_status("a measurement is already running -- Stop it first", error=True)
            return
        try:
            values = panel.collect_values()
            measurement = spec.build(**values)
            # Lazy import (frontend stays decoupled from the neutral_atom backend
            # at import time; the feed only ever touches the measurement contract
            # + hub.publish, so it is guarded by test_virtual_equals_real_contract).
            from Zou_lab_control.neutral_atom.operations.feeds import ScannedMeasurementFeed

            feed = ScannedMeasurementFeed(
                self.hub, measurement, x_key=spec.x_key, y_key=spec.y_key, grid_shape=spec.grid_shape,
            )
        except Exception as exc:
            panel.set_status(f"build failed: {str(exc).splitlines()[0][:140]}", error=True)
            return

        # drop a previous (finished) measurement feed so feeds don't pile up
        if self._meas_feed is not None and self._meas_feed in self.feeds:
            self.feeds.remove(self._meas_feed)
        self._meas_feed = feed
        self._meas_spec = spec
        self.feeds.append(feed)
        self.meas_button.setEnabled(bool(self._controllable_feeds()))
        self._ensure_result_panel(spec)
        if not self._timer.isActive():           # make sure the curve actually refreshes
            self._timer.start()
            self.pause_button.setText("Pause Plot")
        panel.set_running(True)
        panel.set_status(f"running 0/{feed.n_points}…", error=False)
        self.status_dot.set_color(GREEN)
        try:
            feed.start(rate_hz=getattr(feed, "rate_hz", 5.0))
        except Exception as exc:
            panel.set_running(False)
            panel.set_status(f"start failed: {str(exc).splitlines()[0][:140]}", error=True)

    def _stop_measurement(self) -> None:
        feed = self._meas_feed
        if feed is None:
            return
        try:
            feed.stop()
        except Exception:
            pass
        if self.measurement_panel is not None:
            self.measurement_panel.set_running(False)
            done = getattr(feed, "points_done", 0)
            self.measurement_panel.set_status(f"stopped at {done}/{getattr(feed, 'n_points', 0)} points", error=False)

    def _poll_measurement(self) -> None:
        """Called each tick: surface progress, and once the scan completes show
        the result (for temperature, fit T from the form's capture radius)."""
        feed = self._meas_feed
        panel = self.measurement_panel
        if feed is None or panel is None:
            return
        if not getattr(feed, "finished", False):
            if getattr(feed, "running", False):
                panel.set_status(f"running {feed.points_done}/{feed.n_points}…", error=False)
            return
        # finished: report once, run the optional fit, then release the controls
        if not panel._running:
            return
        panel.set_running(False)
        self.status_dot.set_color(GREY if self._timer.isActive() else YELLOW)
        spec = self._meas_spec
        summary = f"done: {feed.points_done} points"
        fit_summary = self._fit_measurement(spec, feed, panel)
        panel.set_status(summary + (f" — {fit_summary}" if fit_summary else ""), error=False)

    def _fit_measurement(self, spec, feed, panel) -> str:
        """Run the spec's declared fit (temperature) on the completed curve and
        return a short result string; '' when the spec declares no fit."""
        meta = getattr(spec, "metadata", None) or {}
        if meta.get("fit") != "fit_temperature":
            return ""
        try:
            import numpy as _np

            x = _np.asarray(self.hub.latest(spec.x_key), dtype=float)
            y = _np.asarray(self.hub.latest(spec.y_key), dtype=float)
            param_key = meta.get("fit_param", "capture_radius")
            scale = float(meta.get("fit_param_scale", 1.0))
            radius = float(panel.collect_values().get(param_key, 0.0)) * scale
            from Zou_lab_control.neutral_atom.operations.temperature import fit_temperature

            fit = fit_temperature(x, y, capture_radius=radius)
            t_uK = getattr(fit, "temperature_uK", None)
            if t_uK is None:
                t_uK = getattr(fit, "temperature_K", float("nan")) * 1e6
            return f"T ≈ {t_uK:.1f} µK"
        except Exception as exc:
            return f"fit failed: {str(exc).splitlines()[0][:80]}"

    def _mark_dirty(self, *_args) -> None:
        if self._building:
            return
        self.save_button.set_dirty(True, dirty_color=YELLOW)
        self._update_summary()

    # ------------------------------------------------------------------ presets
    def _refresh_presets(self) -> None:
        with _signals_blocked(self.preset_combo):
            self.preset_combo.clear()
            self.preset_combo.addItem("Presets…")
            self.preset_combo.addItems(list_task_presets())
            self.preset_combo.setCurrentIndex(0)

    def _on_preset_pick(self, index: int) -> None:
        name = self.preset_combo.currentText()
        if not name or name == "Presets…":
            return
        try:
            state = resolve_task_state(name)
        except Exception as exc:
            self._message(f"Load preset failed: {exc}")
            return
        self._address = None
        self.load_state(state)
        self.save_button.set_dirty(False)
        self._refresh_presets()

    # ------------------------------------------------------------------ refresh
    def _expression_namespace(self) -> dict[str, object]:
        namespace: dict[str, object] = {"np": np, "numpy": np, "math": _math}
        namespace.update(self.hub.snapshot_latest())
        namespace["history"] = self.hub.history
        namespace["latest"] = self.hub.latest
        namespace["names"] = self.hub.names
        namespace["shot"] = self.hub.shot
        return namespace

    def _tick(self) -> None:
        # poll the measurement EVERY tick (even when no new signal arrived) so
        # the run-complete transition is never missed if the feed self-stops
        # between version bumps.
        self._poll_measurement()
        version = self.hub.version
        if version == self._last_version:
            return
        self._last_version = version
        namespace = self._expression_namespace()
        for card in self.cards:
            card.refresh(namespace)
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
        self.summary.setText(
            f"{len(self.cards)} panels | {n_signals} signals | shot {self.hub.shot}"
            f" | every {self.state.interval_ms} ms")

    # ------------------------------------------------------------------ files
    def save_to_file(self) -> None:
        try:
            state = self.read_state()
            start = self._address or str(_task_files_dir() / f"{state.name}.json")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save task layout", start, "Task layout (*.json)")
            if not path:
                return
            state.save(path)
            self._address = path
            self.save_button.set_dirty(False)
            self._refresh_presets()             # a layout saved into tasks/ is a named preset
            self._message(f"Saved: {path}")
        except Exception as exc:
            self._message(f"Save failed: {exc}")

    def load_from_file(self) -> None:
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
        self._timer.stop()
        for feed in self.feeds:
            try:
                feed.stop()
            except Exception:
                pass
        self._editor_teardown()
        for card in self.cards:
            card.shutdown()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)


def show_task_console(
    *,
    hub,
    state: TaskConsoleState | None = None,
    task: str | None = None,
    feeds: Sequence[object] = (),
    measurements: Sequence[object] = (),
    scale: float | None = None,
    window_ratio: float = 0.84,
    title: str = "TaskConsole@Zou lab",
):
    """Open the console in a Fluent window (mirrors ``show_pulse_gui``: the body
    sizes itself from the primary screen; the window wraps it exactly).

    ``task`` loads a NAMED dashboard: a built-in (``atom_loading_monitor``,
    ``loading_rate_live``), a layout saved in tasks/, or a JSON path.

    ``measurements`` is the declarative measurement catalog
    (``exp.readout.measurement_specs()``): pass it to turn the Control tab's
    Measurement section into the auto-generated parameter form + one-click Start;
    omit it and that section stays a disabled placeholder."""

    app = ensure_qt_app()
    if state is None and task is not None:
        state = resolve_task_state(task)
    console = TaskConsole(hub=hub, state=state, feeds=feeds, measurements=measurements,
                          scale=scale, window_ratio=window_ratio)
    window = FluentWindow(widget=console, title=title, hide_on_close=False)
    window.adjustSize()
    window.setFixedSize(window.size())
    window.show()
    if not hasattr(app, "_zlc_task_windows"):
        app._zlc_task_windows = []
    app._zlc_task_windows.append(window)
    return console


__all__ = [
    "PanelConfig",
    "TaskConsole",
    "TaskConsoleState",
    "default_console_state",
    "show_task_console",
]
