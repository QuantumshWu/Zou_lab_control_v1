"""Task console: a configurable grid dashboard of live experiment panels.

Each panel is one live plot (2D image / 1D vector / rolling monitor / distribution)
in one of the LIMITED frontend size presets (``frontend.PANEL_SIZES``, "cols x
rows" grid cells), pinned to a grid position.  The plot itself comes from the
frontend's :func:`~Zou_lab_control.frontend.live.panel_plot` preset, so every
size shares one visual language (fixed dpi -> fixed title/label sizes), and the
console only owns wiring + layout.

A panel's data source is a one-line expression evaluated against the named
signals of a :class:`~Zou_lab_control.neutral_atom.core.signals.SignalHub`
(the same trusted-local-code posture as the pulse GUI's Scan tab):

    value = frame                       # show the latest camera frame
    value = rate_grid - b_rate_grid     # arbitrary math across signals
    value = history('counts', 200).ravel()

Layouts (panels + positions + sizes + expressions + params) save/load as ONE
JSON and are machine-portable: the grid-cell pixel geometry is a frontend
design constant, never part of the layout.
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
    PANEL_CELL_PX,
    PANEL_CHROME_PX,
    PANEL_GAP_PX,
    PANEL_SIZES,
    panel_plot,
    panel_size_cells,
)
from .qt_fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    YELLOW,
    FluentButton,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentStatusDot,
    FluentWindow,
    ensure_qt_app,
    fluent_widget_stylesheet,
    scaled_px,
    set_fluent_scale,
)

try:  # same guarded import as pulse_gui: the console degrades without matplotlib-qt
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvasQTAgg

    class FigureCanvas(_FigureCanvasQTAgg):
        """Canvas whose wheel events never leak into the surrounding scroll area."""

        def wheelEvent(self, event):  # noqa: N802 - Qt naming
            super().wheelEvent(event)
            event.accept()
except Exception:  # pragma: no cover - depends on the local matplotlib install
    plt = None
    FigureCanvas = None


TASK_FILES_ENV = "ZLC_TASK_DIR"

PANEL_KINDS: dict[str, str] = {
    "2d": "2D image",
    "1d": "1D vector",
    "monitor": "Rolling trace",
    "hist": "Distribution",
}

CMAPS = ("inferno", "viridis", "magma", "plasma", "gray", "coolwarm")

_DEFAULT_SOURCES = {
    "2d": "value = frame",
    "1d": "value = rate_sites",
    "monitor": "value = rate",
    "hist": "value = history('counts', 200).ravel()",
}

# PanelCard chrome (RAW px, matching the raw-px canvas geometry): these row
# heights are built into the card below and MUST stay <= PANEL_CHROME_PX, or
# the canvas would overflow its grid cells.
_HEADER_H = 30
_SOURCE_H = 30
_STATUS_H = 16
_CARD_PAD = 8
_CARD_SPACING = 5


def _task_files_dir() -> Path:
    root = os.environ.get(TASK_FILES_ENV)
    path = Path(root) if root else Path(__file__).resolve().parents[2] / "tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ====================================================================== state
class PanelConfig:
    """One panel: kind + a size PRESET + the grid cell it is pinned to."""

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
    def cols(self) -> int:
        return panel_size_cells(self.size)[0]

    @property
    def rows(self) -> int:
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


def default_console_state() -> TaskConsoleState:
    """The reference layout: 2x2 loading image, 2x1 distribution + 2x1 rate trace
    on the right, and a 4x2 per-site loading-rate strip along the bottom."""

    return TaskConsoleState(
        name="atom_loading",
        panels=[
            PanelConfig(kind="2d", title="Loading image", row=0, col=0, size="2x2",
                        source="value = frame"),
            PanelConfig(kind="hist", title="Counts distribution", row=0, col=2, size="2x1",
                        source="value = history('counts', 200).ravel()", params={"bins": 80}),
            PanelConfig(kind="monitor", title="Loading rate", row=1, col=2, size="2x1",
                        source="value = rate", params={"length": 300}),
            PanelConfig(kind="1d", title="Per-site loading rate", row=2, col=0, size="4x2",
                        source="value = rate_sites"),
        ],
    )


# ====================================================================== panels
class PanelCard(FluentGroupBox):
    """One dashboard panel, pinned to its grid cells: a compact header (title +
    position + size preset + kind params), a one-line source expression with
    Apply, the live plot canvas, and a status line."""

    changed = QtCore.pyqtSignal()          # any config edit (console marks dirty)
    layout_changed = QtCore.pyqtSignal()   # grid placement/size edit (console re-grids)
    remove_requested = QtCore.pyqtSignal(object)

    def __init__(self, config: PanelConfig, parent=None):
        super().__init__(PANEL_KINDS[config.kind], parent)
        self.config = config
        self.plotter = None
        self.canvas = None
        self._value_shape: tuple[int, ...] | None = None
        self._compiled_source = config.source

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(_CARD_PAD, _CARD_PAD, _CARD_PAD, _CARD_PAD)
        outer.setSpacing(_CARD_SPACING)

        # --- header (one compact row) ------------------------------------------
        head_row = QtWidgets.QWidget()
        head_row.setStyleSheet("background: transparent;")
        head_row.setFixedHeight(_HEADER_H)
        head = QtWidgets.QHBoxLayout(head_row)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(_CARD_SPACING)

        self.title_edit = FluentLineEdit(config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.setMinimumWidth(scaled_px(90))
        self.title_edit.textChanged.connect(self._on_title)
        head.addWidget(self.title_edit, 1)

        self.place_spins: dict[str, FluentDoubleSpinBox] = {}
        for key, label, tip in (("row", "R", "Grid row this panel is pinned to"),
                                ("col", "C", "Grid column this panel is pinned to")):
            tag = FluentLabel(label)
            tag.setToolTip(tip)
            head.addWidget(tag)
            spin = FluentDoubleSpinBox(length=2, allow_minus=False)
            spin.setDecimals(0)
            spin.setRange(0, 15)
            spin.setValue(getattr(config, key))
            spin.setFixedWidth(scaled_px(44, minimum=38))
            spin.valueChanged.connect(lambda _v, k=key: self._on_place(k))
            head.addWidget(spin)
            self.place_spins[key] = spin

        self.size_combo = FluentComboBox()
        self.size_combo.addItems(list(PANEL_SIZES))
        self.size_combo.setCurrentText(config.size)
        self.size_combo.setFixedWidth(scaled_px(58, minimum=50))
        self.size_combo.setToolTip("Panel size preset (grid cells, cols x rows)")
        self.size_combo.currentTextChanged.connect(self._on_size)
        head.addWidget(self.size_combo)

        self._build_param_widgets(head)

        remove = FluentButton("X", color=ORANGE)
        remove.setFixedSize(scaled_px(26, minimum=22), _HEADER_H - 4)
        remove.setToolTip("Remove this panel")
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        head.addWidget(remove)
        outer.addWidget(head_row)

        # --- one-line source expression + Apply (the Scan-tab Run pattern) -----
        source_row = QtWidgets.QWidget()
        source_row.setStyleSheet("background: transparent;")
        source_row.setFixedHeight(_SOURCE_H)
        src = QtWidgets.QHBoxLayout(source_row)
        src.setContentsMargins(0, 0, 0, 0)
        src.setSpacing(_CARD_SPACING)
        self.source_edit = FluentLineEdit(config.source)
        self.source_edit.setStyleSheet(
            self.source_edit.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
        self.source_edit.setToolTip(
            "Panel data source: one line of Python evaluated against the live signals.\n"
            "Assign the result to `value`.  Namespace: every signal name (latest value),\n"
            "history(name, n), latest(name), names(), shot, np, math.")
        self.apply_button = FluentButton("Apply", color=GREEN)
        self.apply_button.setFixedSize(scaled_px(60, minimum=52), _SOURCE_H - 4)
        self.apply_button.clicked.connect(self._apply_source)
        self.source_edit.textChanged.connect(lambda: self.apply_button.set_dirty(True))
        self.source_edit.returnPressed.connect(self._apply_source)
        src.addWidget(self.source_edit, 1)
        src.addWidget(self.apply_button)
        outer.addWidget(source_row)

        # --- plot area + status ------------------------------------------------
        self.canvas_holder = QtWidgets.QVBoxLayout()
        self.canvas_holder.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.canvas_holder, 1)
        self.status = FluentLabel("waiting for data…")
        self.status.setFixedHeight(_STATUS_H)
        self.status.setStyleSheet(f"color: {GREY}; background: transparent;")
        outer.addWidget(self.status)

        self._apply_fixed_size()

    # ------------------------------------------------------------- geometry
    def _apply_fixed_size(self) -> None:
        """Pin the card to EXACTLY its spanned grid cells (incl. swallowed gaps),
        so every card lines up on the console grid regardless of content."""
        cols, rows = self.config.cols, self.config.rows
        width = cols * PANEL_CELL_PX[0] + (cols - 1) * PANEL_GAP_PX
        height = rows * PANEL_CELL_PX[1] + (rows - 1) * PANEL_GAP_PX
        self.setFixedSize(width, height)

    # ------------------------------------------------------------- config edits
    def _on_title(self, text: str) -> None:
        self.config.title = str(text)
        if self.plotter is not None and getattr(self.plotter, "ax", None) is not None:
            self.plotter.ax.set_title(self.config.title)
            self.plotter.draw()
        self.changed.emit()

    def _on_place(self, key: str) -> None:
        setattr(self.config, key, int(self.place_spins[key].value()))
        self.changed.emit()
        self.layout_changed.emit()

    def _on_size(self, size: str) -> None:
        self.config.size = str(size)
        self._reset_plot()
        self._apply_fixed_size()
        self.changed.emit()
        self.layout_changed.emit()

    def _build_param_widgets(self, head: QtWidgets.QHBoxLayout) -> None:
        kind = self.config.kind
        if kind == "2d":
            self.cmap_combo = FluentComboBox()
            self.cmap_combo.addItems(list(CMAPS))
            current = str(self.config.params.get("cmap", "inferno"))
            if current in CMAPS:
                self.cmap_combo.setCurrentText(current)
            self.cmap_combo.setFixedWidth(scaled_px(82, minimum=70))
            self.cmap_combo.setToolTip("Colormap")
            self.cmap_combo.currentTextChanged.connect(self._on_param)
            head.addWidget(self.cmap_combo)
        elif kind == "monitor":
            self.length_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
            self.length_spin.setDecimals(0)
            self.length_spin.setRange(20, 10_000)
            self.length_spin.setValue(int(self.config.params.get("length", 300)))
            self.length_spin.setFixedWidth(scaled_px(58, minimum=50))
            self.length_spin.setToolTip("Rolling history length (shots kept on screen)")
            self.length_spin.valueChanged.connect(self._on_param)
            head.addWidget(self.length_spin)
        elif kind == "hist":
            self.bins_spin = FluentDoubleSpinBox(length=4, allow_minus=False)
            self.bins_spin.setDecimals(0)
            self.bins_spin.setRange(5, 500)
            self.bins_spin.setValue(int(self.config.params.get("bins", 60)))
            self.bins_spin.setFixedWidth(scaled_px(52, minimum=46))
            self.bins_spin.setToolTip("Histogram bins")
            self.bins_spin.valueChanged.connect(self._on_param)
            head.addWidget(self.bins_spin)

    def _on_param(self, *_args) -> None:
        kind = self.config.kind
        if kind == "2d":
            self.config.params["cmap"] = self.cmap_combo.currentText()
        elif kind == "monitor":
            self.config.params["length"] = int(self.length_spin.value())
        elif kind == "hist":
            self.config.params["bins"] = int(self.bins_spin.value())
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
        Every failure lands in the status line -- a bad expression in one panel
        must never break the console or its siblings."""

        try:
            value = self._evaluate(dict(namespace))
            self._feed(value)
        except Exception as exc:
            self.status.setStyleSheet(f"color: {RED}; background: transparent;")
            self.status.setText(str(exc).splitlines()[0][:160] or type(exc).__name__)
            return
        self.status.setStyleSheet(f"color: {GREY}; background: transparent;")
        shot = namespace.get("shot")
        self.status.setText(f"shot {int(shot)}" if isinstance(shot, (int, float)) else "ok")

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
        if kind == "2d":
            if arr.ndim != 2 or min(arr.shape) < 2:
                raise ValueError(f"2D panel needs a 2D array value (got shape {arr.shape})")
            # bound the point table so Live2DDis.fill_grid stays cheap per tick
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
        return flat

    def _feed(self, value) -> None:
        value = self._coerce(value)
        shape = (1,) if isinstance(value, float) else tuple(np.shape(value))
        if self.plotter is None or shape != self._value_shape:
            self._build_plot(value)
            self._value_shape = shape
            return
        kind = self.config.kind
        if kind == "2d":
            self.plotter.update(np.asarray(value).ravel())
        elif kind == "monitor":
            self.plotter.roll(value)
        elif kind == "hist":
            self.plotter.update(value)
        else:  # 1d
            self.plotter.update(value)

    # ------------------------------------------------------------- plot lifecycle
    def _build_plot(self, value) -> None:
        if FigureCanvas is None:
            raise RuntimeError("matplotlib Qt canvas is not available")
        self._teardown_plot()
        kind = self.config.kind
        size = self.config.size
        label = self.config.title or PANEL_KINDS[kind]
        if kind == "2d":
            arr = np.asarray(value, dtype=float)
            ny, nx = arr.shape
            xx, yy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
            data_x = np.column_stack([xx.ravel(), yy.ravel()])
            self.plotter = panel_plot(
                data_x, arr.ravel(), kind="2d", size=size,
                cmap=str(self.config.params.get("cmap", "inferno")),
                labels=("X (px)", "Y (px)", label), title=self.config.title or None)
        elif kind == "monitor":
            length = max(20, int(self.config.params.get("length", 300)))
            history = np.full(length, np.nan)
            self.plotter = panel_plot(
                np.arange(length, dtype=float), history, kind="monitor", size=size,
                labels=("Shots ago", label, "Z"), title=self.config.title or None,
                relim_mode="tight")
            self.plotter.roll(float(value))
        elif kind == "hist":
            self.plotter = panel_plot(
                np.asarray(value, dtype=float), kind="hist", size=size,
                bins=int(self.config.params.get("bins", 60)),
                labels=("Value", "Shots", "Population"), title=self.config.title or None)
        else:  # 1d
            vec = np.asarray(value, dtype=float).reshape(-1)
            self.plotter = panel_plot(
                np.arange(len(vec), dtype=float), vec, kind="1d", size=size,
                labels=("Site", label, "Z"), title=self.config.title or None,
                relim_mode="tight")
        if self.config.title:
            self.plotter.ax.set_title(self.config.title)
        self.canvas = FigureCanvas(self.plotter.fig)
        self.canvas.draw()
        self.canvas.setFixedSize(self.canvas.sizeHint())
        self.canvas_holder.addWidget(self.canvas, alignment=QtCore.Qt.AlignCenter)

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


# ====================================================================== console
class TaskConsole(QtWidgets.QWidget):
    """The dashboard window body: header bar + pinned panel grid + refresh timer."""

    def __init__(
        self,
        *,
        hub,
        state: TaskConsoleState | None = None,
        feeds: Sequence[object] = (),
        scale: float | None = None,
        window_ratio: float = 0.84,
        window_px: tuple[int, int] | None = None,
    ):
        ensure_qt_app()
        set_fluent_scale(scale)
        super().__init__()
        self.hub = hub
        self.feeds = list(feeds)
        self.state = state or default_console_state()
        self.window_ratio = float(window_ratio)
        self._window_px = window_px
        self.cards: list[PanelCard] = []
        self._last_version = -1
        self._building = False
        self._address: str | None = None

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
        self.summary = FluentLineEdit("")
        self.summary.setEnabled(False)

        self.kind_combo = FluentComboBox()
        for key, label in PANEL_KINDS.items():
            self.kind_combo.addItem(label, key)
        self.kind_combo.setFixedWidth(scaled_px(108, minimum=92))
        add_button = FluentButton("Add Panel", color=ACCENT)
        add_button.clicked.connect(self._add_panel)
        self.pause_button = FluentButton("Pause", color=ACCENT)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.save_button = FluentButton("Save", color=ACCENT)
        self.save_button.clicked.connect(self.save_to_file)
        load_button = FluentButton("Load", color=ORANGE)
        load_button.clicked.connect(self.load_from_file)

        for widget in (self.status_dot, self.name_edit):
            header.addWidget(widget)
        header.addWidget(self.summary, 1)
        for widget in (self.kind_combo, add_button, self.pause_button, self.save_button, load_button):
            header.addWidget(widget)
        root.addWidget(header_frame)

        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_body = QtWidgets.QWidget()
        self.grid_body.setStyleSheet("background: transparent;")
        self.grid = QtWidgets.QGridLayout(self.grid_body)
        gap = PANEL_GAP_PX
        self.grid.setContentsMargins(gap, gap, gap, gap)
        self.grid.setHorizontalSpacing(gap)
        self.grid.setVerticalSpacing(gap)
        self.scroll.setWidget(self.grid_body)
        root.addWidget(self.scroll, 1)

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
                self._attach_card(PanelCard(config))
            self._regrid()
        finally:
            self._building = False
        self._last_version = -1            # force a refresh on the next tick
        self._update_summary()

    def _attach_card(self, card: PanelCard) -> None:
        card.changed.connect(self._mark_dirty)
        card.layout_changed.connect(self._regrid)
        card.remove_requested.connect(self._remove_panel)
        self.cards.append(card)

    def _regrid(self) -> None:
        """Pin every card to its grid cells: fixed-size rows/columns make spanned
        cards line up exactly (a card's fixed size equals its spanned cells)."""

        for card in self.cards:
            self.grid.removeWidget(card)
        rows = max((c.config.row + c.config.rows for c in self.cards), default=1)
        cols = max((c.config.col + c.config.cols for c in self.cards), default=1)
        for r in range(rows):
            self.grid.setRowMinimumHeight(r, PANEL_CELL_PX[1])
        for c in range(cols):
            self.grid.setColumnMinimumWidth(c, PANEL_CELL_PX[0])
        for card in self.cards:
            cfg = card.config
            self.grid.addWidget(card, cfg.row, cfg.col, cfg.rows, cfg.cols,
                                QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.grid.setRowStretch(rows, 1)
        self.grid.setColumnStretch(cols, 1)
        self._update_summary()

    # ------------------------------------------------------------------ actions
    def _add_panel(self) -> None:
        kind = self.kind_combo.currentData() or "1d"
        rows = max((c.config.row + c.config.rows for c in self.cards), default=0)
        config = PanelConfig(kind=str(kind), row=rows, col=0, size="2x1")
        card = PanelCard(config)
        self._attach_card(card)
        self._regrid()
        self._mark_dirty()

    def _remove_panel(self, card: PanelCard) -> None:
        if card in self.cards:
            self.cards.remove(card)
            card.shutdown()
            self.grid.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
            self._regrid()
            self._mark_dirty()

    def _toggle_pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.pause_button.setText("Resume")
            self.status_dot.set_color(YELLOW)
        else:
            self._timer.start()
            self.pause_button.setText("Pause")
            self.status_dot.set_color(GREEN)

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
        return namespace

    def _tick(self) -> None:
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
        for card in self.cards:
            card.shutdown()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)


def show_task_console(
    *,
    hub,
    state: TaskConsoleState | None = None,
    feeds: Sequence[object] = (),
    scale: float | None = None,
    window_ratio: float = 0.84,
    title: str = "TaskConsole@Zou lab",
):
    """Open the console in a Fluent window (mirrors ``show_pulse_gui``: the body
    sizes itself from the primary screen; the window wraps it exactly)."""

    app = ensure_qt_app()
    console = TaskConsole(hub=hub, state=state, feeds=feeds, scale=scale, window_ratio=window_ratio)
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
