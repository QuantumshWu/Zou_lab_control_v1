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

CMAPS = ("inferno", "viridis", "magma", "plasma", "gray", "coolwarm")

_DEFAULT_SOURCES = {
    "2d": "value = frame",
    "sites": "value = occupied",
    "1d": "value = rate_sites",
    "monitor": "value = rate",
    "monitor_nodist": "value = rate",
    "hist": "value = history('counts', 200).ravel()",
}


class ParamSpec:
    """One declarative panel parameter: the Setting popup generates its widget
    from this spec (confocal style -- adding a parameter is ONE line here, the
    GUI, the JSON params and the plot rebuild all follow)."""

    def __init__(self, key: str, label: str, kind: str, default, *,
                 choices: Sequence[str] = (), lo: float = 0, hi: float = 1e9, tooltip: str = "",
                 display: bool = False):
        self.key = str(key)
        self.label = str(label)
        self.kind = str(kind)              # "choice" | "int" | "signal"
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
        ParamSpec("cmap", "colormap", "choice", "viridis", choices=CMAPS, tooltip="Site-value colormap", display=True),
        ParamSpec("centers", "centers", "signal", "centers",
                  tooltip="Signal holding the (N, 2) site centers in camera px"),
        ParamSpec("image", "image", "signal", "frame",
                  tooltip="Signal for the camera-frame underlay (blank for none)"),
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
            PanelConfig(kind="monitor_nodist", title="Loading rate", row=2, col=2, size="1x2",
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
            PanelConfig(kind="monitor_nodist", title="Loading rate", row=0, col=0, size="1x2",
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


def _single_live_state() -> TaskConsoleState:
    """The BARE default the console opens with: ONE rolling loading-rate trace
    (no side distribution).  Deliberately minimal -- you add panels from the
    header ("Add Panel") to build the task group you want.  The richer named
    dashboards (``atom_loading_monitor``, ``loading_rate_live``) stay one click
    away in the preset picker / ``--task``."""

    return TaskConsoleState(
        name="live",
        panels=[
            PanelConfig(kind="monitor_nodist", title="Loading rate", row=0, col=0, size="2x4",
                        source="value = rate", params={"length": 300}),
        ],
    )


def default_console_state() -> TaskConsoleState:
    return _single_live_state()


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
_MONITOR_UNSET = object()   # sentinel: a monitor panel that has never rolled yet


class PanelCard(FluentGroupBox):
    """One dashboard panel: a titled frame (title = the panel KIND) holding the
    frontend canvas, a status footer, and a text "Setting" button on the title
    strip.  The frame border is the DRAG HANDLE (the matplotlib canvas keeps
    all its own interactions); the footer stretches so the card spans whole
    layout slots -- a 2-row card is exactly two 1-row cards plus the gap."""

    changed = QtCore.pyqtSignal()          # any config edit (console marks dirty)
    layout_changed = QtCore.pyqtSignal()   # size/slot change (console re-arranges)
    remove_requested = QtCore.pyqtSignal(object)
    edit_requested = QtCore.pyqtSignal(object)   # "Edit…" -> open the panel's Edit tab

    def __init__(self, config: PanelConfig, parent=None, *, names_provider=None):
        super().__init__(PANEL_KINDS[config.kind], parent, shadow=True)
        self.config = config
        self.names_provider = names_provider   # callable -> live signal names (Setting combo)
        self.plotter = None
        self.canvas = None
        self._value_shape: tuple[int, ...] | None = None
        self._compiled_source = config.source
        # Monitor roll-gate: remembers the per-signal version of this panel's
        # source at the last roll, so an unrelated feed's version bump does not
        # append a duplicate point.  `_MONITOR_UNSET` = never rolled yet.
        self._last_monitor_key: object = _MONITOR_UNSET
        self._ref_src: str | None = None
        self._ref_names: frozenset = frozenset()
        # 2D coordinate frame: the ROI the axes were built from, so a ROI that
        # SHIFTS (same shape, new origin) still triggers an axes rebuild.
        self._roi_built: list | None = None
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

        # fixed left-label column.  Widest known label is "colormap" -- bumping
        # to 80 px gives it space without crowding the controls (one-char "x"/"y"
        # axis labels in the limits grid use the same column width, so the
        # x/y/lo/hi grid still aligns with the section rows above).
        label_w = scaled_px(80, minimum=56)

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
        self.signal_combo = FluentComboBox()
        self.signal_combo.setToolTip(
            "Pick a live signal to show it directly (sets the expression to\n"
            "`value = <signal>`).  Choose (expression) to write your own.")
        self.signal_combo.activated.connect(self._on_signal_pick)
        sec.addWidget(FluentSettingRow("signal", self.signal_combo, label_width=label_w))

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
        # "signal": a free-form signal name (blank allowed where documented)
        edit = FluentLineEdit(str(current))
        edit.setMinimumWidth(scaled_px(96, minimum=80))   # expands in its row; long signal names not cut off
        edit.setToolTip(spec.tooltip)
        edit.editingFinished.connect(lambda k=spec.key, w=edit: cb(k, w.text().strip()))
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

    def _co_names(self) -> frozenset:
        """The identifiers this panel's source expression references (cached).
        Used to map the source to its signal(s) -- for the monitor roll-gate and
        for the 2D coordinate-frame (ROI) lookup."""
        src = self._compiled_source
        if src != self._ref_src:               # (re)derive on source change
            self._ref_src = src
            try:
                self._ref_names = frozenset(compile(str(src), "<panel-source>", "exec").co_names)
            except Exception:
                self._ref_names = frozenset()
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
            # several feeds, an unrelated feed bumps hub.version and refreshes
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
                centers, vec, kind="sites", size=size, interactions=False,
                cmap=str(self.config.params.get("cmap", "viridis")),
                image=image, roi_radius=max(1.5, 0.3 * spacing),
                labels=("Camera x (px)", "Camera y (px)", label),
                title=self.config.title or None)
        elif kind == "2d":
            arr = np.asarray(value, dtype=float)
            ny, nx = arr.shape
            # Coordinate axes ARE the source's pixel space: when the producing feed
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
                labels=("Shots ago", label, "Z"),
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
                xlabel, ylabel = "Site", label
            self.plotter = panel_plot(
                data_x, vec, kind="1d", size=size, interactions=False,
                labels=(xlabel, ylabel, "Z"),
                relim_mode=str(self.config.params.get("relim", "tight")),
                title=self.config.title or None)
        # Monitor cards are display-only: NO selectors (interactions=False above)
        # and the wheel scrolls the board (isolate_wheel=False) instead of being
        # swallowed.  Interactive zoom / select lives in the Edit tab.
        self.canvas = panel_canvas(self.plotter.fig, isolate_wheel=False)
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

    def __init__(self, measurements: Sequence[object], parent=None, *, single: bool = False):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._specs = list(measurements)
        self._single = bool(single)               # bound to ONE spec (per-panel Edit) -> hide the picker
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
        self.meas_panel = None
        self._feed = None                       # the feed that produces this panel's data
        self._feed_widgets: dict = {}           # acquisition-param name -> editable field
        self._feed_now_labels: dict = {}        # acquisition-param name -> "now: X" reference

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

        # ---- Measurement (only when this panel came from a measurement) -----
        # The measurement's OWN API parameters, auto-generated as a GUI form from
        # its ParamDecls; Start RE-RUNS the scan into this Monitor panel with the
        # edited params.
        spec = console._spec_for_card(card)
        if spec is not None:
            section("Measurement")
            self.meas_panel = MeasurementPanel([spec], single=True)
            self.meas_panel.seed_values(card.config.params.get("measurement_values") or {})
            self.meas_panel.start_requested.connect(console._start_measurement)
            self.meas_panel.stop_requested.connect(console._stop_measurement)
            col.addWidget(self.meas_panel)

        # ---- Acquisition: the editable parameters of the DATA SOURCE behind this
        # panel.  A panel is a VIEW; the producing feed declares what its source
        # exposes via acquisition_parameters() -- a raw-frame panel's source is the
        # camera (exposure / ROI, reconfigured live), a loading-image panel's is
        # the LoadingFeed analysis (exposure / roi_radius / grid / ema / *_frames,
        # re-calibrated).  Each field is PREFILLED with the CURRENT value (with a
        # "now: X" reference), and Apply pushes the edits to the source in place.
        # Measurement panels use the Measurement form above instead (that IS their
        # source), so this section is skipped for them.
        if spec is None:
            self._feed = console._producing_feed(card)
            for name, current in console._feed_params(self._feed):
                if not self._feed_widgets:
                    section("Acquisition")
                edit = FluentLineEdit(_py_to_text(current))
                # EXPAND to fill the row (stretch=1), not a fixed 110 px: a value
                # like a 4-int ROI [1648, 64, 1144, 64] was clipped while the right
                # half of the page sat empty.  The edit takes the slack; the live
                # "now:" reference trails it.  (cutoff is a core rule.)
                edit.setMinimumWidth(scaled_px(150, minimum=120))
                self._feed_widgets[name] = edit
                now = labeled(f"now: {_py_to_text(current)}")
                self._feed_now_labels[name] = now
                holder = QtWidgets.QWidget()
                hl = QtWidgets.QHBoxLayout(holder)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setSpacing(scaled_px(6, minimum=4))
                hl.addWidget(edit, 1)
                hl.addWidget(now, 0)
                # widest feed-param label is "calibration_frames" (18 chars) --
                # 150 clipped its trailing 's'; 170 fits the longest name in full.
                col.addWidget(FluentSettingRow(name, holder, label_width=scaled_px(170, minimum=140)))
            if self._feed_widgets:
                self.feed_restart_button = FluentButton("Apply", color=ACCENT)
                self.feed_restart_button.setToolTip(
                    "Apply the edited acquisition parameters to the data source in place\n"
                    "(reconfigure the camera live, or re-calibrate) -- the panel keeps streaming.")
                self.feed_restart_button.clicked.connect(self._restart_feed)
                col.addWidget(self.feed_restart_button)

        # ---- Parameters: the PLOT's own tunable API params as GUI controls,
        # auto-discovered from the kind's declarative specs.  Each edit re-renders
        # the LIVE panel AND this snapshot.  This is where "the params the plot
        # call exposes" live -- NOT in the basic Setting popup (which keeps only
        # source / size / colormap / relim / unit), so the two never duplicate.
        functional = [s for s in PANEL_PARAMS.get(card.config.kind, ()) if not s.display]
        if functional:
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

        # Fit: the FULL DataFigure model set, available for EVERY kind (no
        # gating).  DataFigure picks the 1D / 2D path from the snapshot, so a 2D
        # image fits the 2D-Gaussian "2D center" model and lines fit the rest.
        # Same [label | control] row idiom as the sections above (aligned label
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
        cmap = str(card.config.params.get("cmap", "inferno" if kind == "2d" else "viridis"))
        try:
            if kind == "2d":
                self._plotter = panel_plot(np.array(src.data_x, dtype=float),
                                           np.array(src.data_y[:, 0], dtype=float), kind="2d",
                                           size=size, cmap=cmap, relim_mode=relim,
                                           labels=tuple(src.labels), title=title)
            elif kind == "sites":
                self._plotter = panel_plot(np.array(src.data_x[:, :2], dtype=float),
                                           np.array(src.data_y[:, 0], dtype=float), kind="sites",
                                           size=size, cmap=cmap, image=getattr(src, "background", None),
                                           roi_radius=getattr(src, "roi_radius", 3.0),
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
        # that to its own format (a measurement -> scan range; a camera feed ->
        # ROI; a 2-D scan -> axis ranges), so no device shape leaks into the GUI.
        # The SAME callback is bound to BOTH the zoom/scroll AND the area selector:
        # ZOOM/PAN updates from the view limits, the area selector OVERRIDES when
        # drawn (precedence in the writeback).  (Wiring only the area selector --
        # the earlier bug -- meant zoom did nothing.)
        area = getattr(self._plotter, "area", None)
        zoom = getattr(self._plotter, "zoom", None)
        writeback = None
        if self.meas_panel is not None:
            writeback = self._read_scan_range
        elif kind == "2d" and self._feed is not None:
            writeback = self._read_region
        if writeback is not None:
            if area is not None:
                area.callback = writeback
            if zoom is not None:
                zoom.callback = writeback
        self.status.setText("snapshot of current data — fit / set limits / save are frozen here")

    def _edit_param(self, key: str, value) -> None:
        """A plot-param edit from the Edit tab: apply to the LIVE panel (re-renders
        it) and re-snapshot here, so the change shows in both surfaces."""
        if self.card is not None:
            self.card._set_param(key, value)
        self.rebuild()

    def _selected_xrange(self):
        """Confocal precedence (the X view): the area selection if one is drawn,
        ELSE the current axis view limits (from zoom/pan).  Returns (lo, hi)
        sorted, or (None, None).  This is what makes scroll-zoom alone set the
        range, with a drag-rectangle overriding it."""
        plotter = self._plotter
        if plotter is None or plotter.ax is None:
            return (None, None)
        area = getattr(plotter, "area", None)
        if area is not None and area.range[0] is not None:
            return (float(area.range[0]), float(area.range[1]))
        lo, hi = sorted(float(v) for v in plotter.ax.get_xlim())
        return (lo, hi)

    def _selected_rect(self):
        """2D analogue of :meth:`_selected_xrange`: the area rectangle if drawn,
        ELSE the current view box (x AND y view limits).  Returns
        (xlo, xhi, ylo, yhi) sorted, or all None."""
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

    def _read_scan_range(self) -> None:
        """Confocal ``_read_range``: ZOOM/PAN or an area SELECTION on the Edit plot
        fills this measurement's scan x-range Min/Max -- the NEXT scan's range.
        Scroll-zoom to narrow, or drag a rectangle to pin it exactly (the
        selection overrides the view); then Start re-scans just that range.  No-op
        without a measurement form."""
        if self.meas_panel is None or self._plotter is None:
            return
        x1, x2 = self._selected_xrange()
        if x1 is None:
            return
        if self.meas_panel.set_axis_range(x1, x2):
            self.status.setText(f"scan range set from view: {x1:.4g} … {x2:.4g}")

    def _read_region(self) -> None:
        """Confocal ``_read_range`` for a 2D panel, GENERIC over the source.

        ZOOM/PAN or an area SELECTION yields the marked rectangle as endpoints
        (x_min, x_max, y_min, y_max) in the panel's axis coordinates (the area
        selection overrides the view box when drawn).  Those ENDPOINTS -- the only
        thing the selector knows -- are handed to the producing source via
        ``region_to_acquisition_parameters``; the SOURCE converts them to its own
        Acquisition parameters (a camera feed -> a ROI rectangle; a 2-D scan ->
        axis ranges).  Whatever fields it names are filled in the Edit form, then
        Apply pushes them.  The frontend encodes NO device-specific shape."""
        if self._feed is None or self._plotter is None:
            return
        convert = getattr(self._feed, "region_to_acquisition_parameters", None)
        if convert is None:
            return
        x1, x2, y1, y2 = self._selected_rect()
        if x1 is None:
            return
        params = convert(x1, x2, y1, y2) or {}
        filled = {}
        for name, value in params.items():
            edit = self._feed_widgets.get(name)
            if edit is not None:
                edit.setText(_py_to_text(value))
                filled[name] = value
        if filled:
            self.status.setText(f"region from view: {filled} — Apply to use it")

    def refresh_feed_now_labels(self) -> None:
        """Update the 'now: <value>' references beside each Acquisition field to the
        source's CURRENT values.  The console calls this each tick for the visible
        Edit tab (one general hook, not a per-field signal), so after the loop
        applies a queued edit the references catch up on their own -- no manual
        wiring per parameter, and the frozen snapshot / Refresh / Fit controls are
        untouched."""
        if self._feed is None or not self._feed_now_labels:
            return
        for name, current in self.console._feed_params(self._feed):
            label = self._feed_now_labels.get(name)
            if label is not None:
                label.setText(f"now: {_py_to_text(current)}")

    def _restart_feed(self) -> None:
        """Apply the edited Acquisition params to the data source.  The change is
        routed through the feed's safe entry: while the acquisition loop runs it is
        applied BETWEEN shots in the loop's own thread (the live Monitor keeps
        streaming, no GUI stall); Apply also START s an idle source so it goes live.
        The 'now:' references update via the console's per-tick refresh (reused
        here for an immediate read), and the Edit snapshot re-snapshots."""
        if self._feed is None:
            return
        new_params = {name: _text_to_py(w.text()) for name, w in self._feed_widgets.items()}
        running = bool(getattr(self._feed, "running", False))
        try:
            self.console._restart_feed(self._feed, new_params)
        except Exception as exc:
            self.status.setText(f"apply failed: {str(exc).splitlines()[0][:120]}")
            return
        # tick the console so an IDLE feed's freshly-published frame shows now; a
        # running feed updates on its own next loop iteration + timer beat.
        self.console.refresh_once()
        self.refresh_feed_now_labels()        # reuse the same refresh the tick uses
        self.status.setText(
            "acquisition parameters queued — Monitor updates on the next frame"
            if running else "acquisition started with the new parameters")
        self.rebuild()

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
        if self._plotter is None:
            return
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
        # The measurement form now lives in EACH measurement panel's own Edit tab
        # (no global Control launcher); _active_meas_panel is the form that owns
        # the running scan.  measurement_panel stays None (kept for the stop/poll
        # fallback) since there is no single shared form any more.
        self.measurement_panel = None
        self.measurement_group = None
        self.measurement_placeholder = None
        self.state = state or default_console_state()
        self.window_ratio = float(window_ratio)
        self._window_px = window_px
        self.cards: list[PanelCard] = []
        self._last_version = -1
        self._building = False
        self._address: str | None = None
        # Per-panel editors: one PanelEditor per opened panel, hosted as a
        # closable tab (keyed by id(card)).  The measurement panel that started
        # the active run (global launcher OR a per-panel editor) is tracked so
        # poll/stop route their status to the right place.
        self._panel_editors: dict[int, "PanelEditor"] = {}
        self._active_meas_panel = None

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
        # The measurement catalog is offered RIGHT HERE (no separate Control tab):
        # picking one + Add Panel creates a result panel and opens its Edit tab,
        # where the measurement's parameters + Start live.
        for spec in self.measurements:
            self.kind_combo.addItem(f"Measurement: {spec.name}", ("measurement", spec.name))
        self.kind_combo.setFixedWidth(scaled_px(132, minimum=104))
        add_button = FluentButton("Add Panel", color=ACCENT)
        add_button.clicked.connect(self._add_panel)
        self.save_button = FluentButton("Save", color=ACCENT)
        self.save_button.clicked.connect(self.save_to_file)
        load_button = FluentButton("Load", color=ORANGE)
        load_button.clicked.connect(self.load_from_file)

        for widget in (self.status_dot, self.name_edit, self.preset_combo):
            header.addWidget(widget)
        header.addWidget(self.summary, 1)
        for widget in (self.kind_combo, add_button, self.save_button, load_button):
            header.addWidget(widget)
        root.addWidget(header_frame)

        # ONE permanent tab: Monitor = the live drag-and-snap panel grid.
        # Measurements are launched from the header's Add Panel (no Control tab);
        # each panel's Edit... opens its OWN closable tab (a PanelEditor) carrying
        # that panel's params / fit / limits / save / re-run (+ Measurement form).
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
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.tab_close_requested.connect(self._on_editor_tab_closed)
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
            for card in list(self.cards):
                self._close_panel_editor(card)   # drop any open Edit tab for this card
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
    def _spec_for_card(self, card: "PanelCard"):
        """The MeasurementSpec a result panel came from (None for plain panels)."""
        name = card.config.params.get("measurement")
        if not name:
            return None
        return next((s for s in self.measurements if getattr(s, "name", None) == name), None)

    # ---- producing-feed discovery: a panel's data comes from a feed; its Edit
    # exposes THAT feed's acquisition parameters (e.g. a loading-image panel is
    # produced by a LoadingFeed -> exposure / roi_radius / grid_shape / ...).
    @staticmethod
    def _referenced_signals(source: str) -> set:
        """The hub signal names a panel's source expression reads (AST Name nodes
        minus the namespace builtins) -- used to map the panel to its feed."""
        import ast
        try:
            tree = ast.parse(str(source or ""), mode="exec")
        except SyntaxError:
            return set()
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        return names - {"np", "numpy", "math", "history", "latest", "names", "shot", "value"}

    def _producing_feed(self, card: "PanelCard"):
        """The feed whose published signals the panel's source reads (None if the
        expression touches no feed signal, e.g. a pure constant)."""
        refs = self._referenced_signals(card.config.source)
        if not refs:
            return None
        for feed in self.feeds:
            published = feed.published_signals() if hasattr(feed, "published_signals") else frozenset()
            if refs & set(published):
                return feed
        return None

    @staticmethod
    def _feed_params(feed) -> list:
        """``[(name, current_value)]`` for the editable parameters of the data
        SOURCE behind ``feed`` -- a camera's exposure/ROI, or the feed's own
        analysis settings.  The feed declares them via ``acquisition_parameters()``
        (the source decides what is tunable, not __init__ reflection)."""
        if feed is None or not hasattr(feed, "acquisition_parameters"):
            return []
        return list(feed.acquisition_parameters().items())

    def _restart_feed(self, feed, new_params: dict):
        """Apply edited acquisition parameters to the producing feed so the
        Monitor re-acquires under them.  This goes through the feed's SAFE entry
        ``apply_acquisition_parameters``: while the acquisition loop runs it owns
        the source, so the edit is queued and the loop applies it BETWEEN shots
        (the source re-arms in its owner thread -- a streaming camera picks up the
        new ROI/exposure -- with no GUI-thread stall and no second ``acquire()``
        racing on one camera; an in-place reconfigure of a running stream would be
        ignored, while stopping/starting the thread from here would block the GUI
        and could deadlock).  An idle feed is reconfigured and stepped once.
        Returns the feed.  Raises if the feed has no editable acquisition params."""
        if feed is None or not hasattr(feed, "apply_acquisition_parameters"):
            raise RuntimeError("this panel's data source exposes no editable acquisition parameters")
        feed.apply_acquisition_parameters(**new_params)
        # Edit's Apply makes the source LIVE: if its acquisition loop is not
        # running (a fresh feed, or one the user stopped), start it so the Monitor
        # streams under the new params.  start() is idempotent, so a feed that is
        # already running just keeps its loop (the edit was queued above).
        if not getattr(feed, "running", False) and hasattr(feed, "start"):
            feed.start(rate_hz=getattr(feed, "rate_hz", 5.0))
        return feed

    def _edit_card(self, card: "PanelCard") -> None:
        """Open (or focus) this panel's OWN closable Edit tab (a PanelEditor).

        Opens even BEFORE the panel has data -- a measurement panel's Edit is
        exactly where you set its parameters and press Start to PRODUCE the data,
        and a plain panel's parameters are editable straight away.  The snapshot
        section just shows "waiting for data" until a plot exists."""
        if card is None:
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
        if editor.meas_panel is not None and editor.meas_panel is self._active_meas_panel:
            self._active_meas_panel = None     # don't write status to a deleted form
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        editor.teardown()
        editor.setParent(None)
        editor.deleteLater()

    def _on_editor_tab_closed(self, widget) -> None:
        """X on a PanelEditor tab: tear it down + drop it from the registry.
        The permanent Monitor tab carries no X, so it never arrives here."""
        if not isinstance(widget, PanelEditor):
            return
        if widget.meas_panel is not None and widget.meas_panel is self._active_meas_panel:
            self._active_meas_panel = None     # don't write status to a deleted form
        index = self.tabs.indexOf(widget)
        if index >= 0:
            self.tabs.removeTab(index)
        for key, editor in list(self._panel_editors.items()):
            if editor is widget:
                del self._panel_editors[key]
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
        data = self.kind_combo.currentData()
        # A measurement entry: create (or focus) its result panel and open its
        # Edit tab, where the auto-generated parameter form + Start live.
        if isinstance(data, tuple) and len(data) == 2 and data[0] == "measurement":
            spec = next((s for s in self.measurements if s.name == data[1]), None)
            if spec is not None:
                card = self._ensure_result_panel(spec)
                self._edit_card(card)
            return
        kind = data or "1d"
        rows = max((c.config.row + c.config.rows for c in self.cards), default=0)
        config = PanelConfig(kind=str(kind), row=rows, col=0, size="1x2")
        card = PanelCard(config, parent=self.board, names_provider=self.hub.names)
        self._attach_card(card)
        self._arrange()
        self._mark_dirty()

    def _remove_panel(self, card: PanelCard) -> None:
        if card in self.cards:
            card.settings_popup.hide()
            self._close_panel_editor(card)     # drop this card's Edit tab too
            self.cards.remove(card)
            card.shutdown()
            card.setParent(None)
            card.deleteLater()
            self._arrange()
            self._mark_dirty()

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

    def _start_measurement(self, meas_panel) -> None:
        """Start (or RE-RUN) a measurement from the given MeasurementPanel -- a
        measurement panel's own Edit form.  Reads the spec +
        param values off that panel; the scan streams into the spec's result
        panel (created if missing, REUSED on re-run, so editing a panel's params
        and hitting Start restarts that same Monitor panel)."""
        spec = meas_panel.current_spec() if meas_panel is not None else None
        if spec is None:
            return
        if self._meas_feed is not None and getattr(self._meas_feed, "running", False):
            meas_panel.set_status("a measurement is already running -- Stop it first", error=True)
            return
        try:
            values = meas_panel.collect_values()
            measurement = spec.build(**values)
            # Lazy import (frontend stays decoupled from the neutral_atom backend
            # at import time; the feed only ever touches the measurement contract
            # + hub.publish, so it is guarded by test_virtual_equals_real_contract).
            from Zou_lab_control.neutral_atom.operations.feeds import ScannedMeasurementFeed

            feed = ScannedMeasurementFeed(
                self.hub, measurement, x_key=spec.x_key, y_key=spec.y_key, grid_shape=spec.grid_shape,
            )
        except Exception as exc:
            meas_panel.set_status(f"build failed: {str(exc).splitlines()[0][:140]}", error=True)
            return

        # drop a previous (finished) measurement feed so feeds don't pile up
        if self._meas_feed is not None and self._meas_feed in self.feeds:
            self.feeds.remove(self._meas_feed)
        self._meas_feed = feed
        self._meas_spec = spec
        self._active_meas_panel = meas_panel
        self.feeds.append(feed)
        card = self._ensure_result_panel(spec)
        card.config.params["measurement_values"] = dict(values)   # seed the next Edit reopen
        if not self._timer.isActive():           # make sure the curve actually refreshes
            self._timer.start()
        meas_panel.set_running(True)
        meas_panel.set_status(f"running 0/{feed.n_points}…", error=False)
        self.status_dot.set_color(GREEN)
        try:
            feed.start(rate_hz=getattr(feed, "rate_hz", 5.0))
        except Exception as exc:
            meas_panel.set_running(False)
            meas_panel.set_status(f"start failed: {str(exc).splitlines()[0][:140]}", error=True)

    def _stop_measurement(self) -> None:
        feed = self._meas_feed
        if feed is None:
            return
        try:
            feed.stop()
        except Exception:
            pass
        panel = self._active_meas_panel or self.measurement_panel
        if panel is not None:
            panel.set_running(False)
            done = getattr(feed, "points_done", 0)
            panel.set_status(f"stopped at {done}/{getattr(feed, 'n_points', 0)} points", error=False)

    def _poll_measurement(self) -> None:
        """Called each tick: surface progress, and once the scan completes show
        the result (for temperature, fit T from the form's capture radius)."""
        feed = self._meas_feed
        panel = self._active_meas_panel or self.measurement_panel
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
        # Per-signal publish counters (reserved key) so a rolling monitor can tell
        # a new sample of its own source from an unrelated feed's version bump.
        namespace["__sig_versions__"] = self.hub.signal_versions()
        # Coordinate frames (reserved key): {signal_name: [x, w, y, h]} from any
        # feed whose acquisition source declares a ROI.  A 2D panel reads its
        # source signal's frame so the image axes are the REAL camera pixel
        # coordinates (ROI), not 0..N -- and an area-select maps back to the ROI.
        namespace["__coord_frames__"] = self._coord_frames()
        return namespace

    def _coord_frames(self) -> dict[str, list]:
        """Map each feed-published signal to its source's spatial ``region``
        endpoints ``[x_min, x_max, y_min, y_max]`` (the acquisition-layer format)
        when the feed declares one (a camera frame), so a panel can put its axes in
        real source pixels.  A panel reads index 0 (x_min) and index 2 (y_min) as
        the axis origin, so this is robust to endpoints vs any position+size form."""
        frames: dict[str, list] = {}
        for feed in self.feeds:
            try:
                region = feed.acquisition_parameters().get("region")
            except Exception:
                region = None
            if not region:
                continue
            try:
                sigs = feed.published_signals() if hasattr(feed, "published_signals") else ()
            except Exception:
                sigs = ()
            for s in sigs:
                frames[str(s)] = list(region)
        return frames

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
        # keep the visible Edit tab's 'now:' acquisition references live, so a
        # queued parameter edit shows as applied once the loop picks it up (one
        # general hook on the current tab -- no per-field wiring).
        editor = self.tabs.currentWidget()
        if isinstance(editor, PanelEditor):
            editor.refresh_feed_now_labels()
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
        # A wedged feed must not fail silently: if any feed recorded an error,
        # raise a red banner naming it instead of just showing frozen data.
        faulted = [f for f in self.feeds if getattr(f, "last_error", None)]
        if faulted:
            f = faulted[0]
            who = (getattr(f, "prefix", "") or "feed").rstrip(":")
            n = int(getattr(f, "consecutive_errors", 1))
            self.summary.set_danger(True)
            self.summary.setText(f"⚠ FEED ERROR ({who}, ×{n}): {f.last_error}"[:200])
        else:
            self.summary.set_danger(False)
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
        for editor in list(self._panel_editors.values()):
            editor.teardown()
        self._panel_editors.clear()
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
    (``exp.readout.measurement_specs()`` -- an AUTO-DISCOVERED list, built-ins +
    any ``@measurement`` / ``register_measurement``): pass it and every spec is
    listed in the header's Add Panel dropdown; picking one + Add Panel creates a
    result panel and opens its Edit tab (auto-generated parameter form + one-click
    Start).  Omit it and the dropdown carries only plot kinds."""

    app = ensure_qt_app()
    if state is None and task is not None:
        state = resolve_task_state(task)
    console = TaskConsole(hub=hub, state=state, feeds=feeds, measurements=measurements,
                          scale=scale, window_ratio=window_ratio)
    # A passed-in producer feed should stream the moment the window opens -- so the
    # Monitor is live without the caller having to remember feed.start().  start()
    # is idempotent, so a feed the caller already started (e.g. bring-up's
    # LoadingFeed(...).start(rate_hz=4)) keeps its own rate; only NON-running feeds
    # are launched here.  (TaskConsole.__init__ deliberately does NOT do this, so
    # tests/notebooks keep deterministic manual stepping.)
    for feed in feeds:
        if not getattr(feed, "running", False) and hasattr(feed, "start"):
            feed.start(rate_hz=getattr(feed, "rate_hz", 5.0))
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
