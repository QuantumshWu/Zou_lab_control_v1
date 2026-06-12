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
    YELLOW,
    DIVIDER,
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
    add_fluent_shadow,
    ensure_qt_app,
    fluent_widget_stylesheet,
    scaled_px,
    set_fluent_scale,
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


def default_console_state() -> TaskConsoleState:
    """The reference layout: 2x2 loading image, half-height distribution + rate
    trace stacked on the right, and a double-width per-site strip below."""

    return TaskConsoleState(
        name="atom_loading",
        panels=[
            PanelConfig(kind="2d", title="Loading image", row=0, col=0, size="2x2",
                        source="value = frame"),
            PanelConfig(kind="hist", title="Counts distribution", row=0, col=2, size="1x2",
                        source="value = history('counts', 200).ravel()", params={"bins": 80}),
            PanelConfig(kind="monitor", title="Loading rate", row=1, col=2, size="1x2",
                        source="value = rate", params={"length": 300}),
            PanelConfig(kind="1d", title="Per-site loading rate", row=2, col=0, size="2x4",
                        source="value = rate_sites"),
        ],
    )


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

    def __init__(self, config: PanelConfig, parent=None):
        super().__init__(PANEL_KINDS[config.kind], parent, shadow=False)
        self.config = config
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
        """The confocal-style settings popup; every widget keeps the stock
        fluent size (no squeezed rows)."""

        popup = QtWidgets.QFrame(self, QtCore.Qt.Popup)
        popup.setStyleSheet(
            f"QFrame {{ background: white; border: 1px solid {DIVIDER};"
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

        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.textChanged.connect(self._on_title)
        kind_tag = FluentLabel(PANEL_KINDS[self.config.kind])
        kind_tag.setStyleSheet(f"color: {GREY}; background: transparent;")
        row(self.title_edit, kind_tag)

        self.size_combo = FluentComboBox()
        self.size_combo.addItems(list(PANEL_SIZES))
        self.size_combo.setCurrentText(self.config.size)
        self.size_combo.setToolTip("Panel size preset (height x width half-units; 2x2 = the stock plot region)")
        self.size_combo.currentTextChanged.connect(self._on_size)
        size_tag = FluentLabel("size")
        size_tag.setStyleSheet(f"color: {GREY}; background: transparent;")
        param_widget = self._build_param_widget()
        if param_widget is not None:
            row(self.size_combo, size_tag, param_widget, stretch_first=False)
        else:
            row(self.size_combo, size_tag, stretch_first=False)

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
        self.status.setStyleSheet(f"color: {GREY}; background: transparent;")
        remove = FluentButton("Remove", color=ORANGE)
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        row(self.status, remove)

        self.settings_popup = popup

    def _build_param_widget(self):
        kind = self.config.kind
        if kind == "2d":
            self.cmap_combo = FluentComboBox()
            self.cmap_combo.addItems(list(CMAPS))
            current = str(self.config.params.get("cmap", "inferno"))
            if current in CMAPS:
                self.cmap_combo.setCurrentText(current)
            self.cmap_combo.setToolTip("Colormap")
            self.cmap_combo.currentTextChanged.connect(self._on_param)
            return self.cmap_combo
        if kind == "monitor":
            self.length_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
            self.length_spin.setDecimals(0)
            self.length_spin.setRange(20, 10_000)
            self.length_spin.setValue(int(self.config.params.get("length", 300)))
            self.length_spin.setToolTip("Rolling history length (shots kept on screen)")
            self.length_spin.valueChanged.connect(self._on_param)
            return self.length_spin
        if kind == "hist":
            self.bins_spin = FluentDoubleSpinBox(length=4, allow_minus=False)
            self.bins_spin.setDecimals(0)
            self.bins_spin.setRange(5, 500)
            self.bins_spin.setValue(int(self.config.params.get("bins", 60)))
            self.bins_spin.setToolTip("Histogram bins")
            self.bins_spin.valueChanged.connect(self._on_param)
            return self.bins_spin
        return None

    def _open_settings(self) -> None:
        anchor = self.setting_button.mapToGlobal(
            QtCore.QPoint(self.setting_button.width(), self.setting_button.height()))
        popup = self.settings_popup
        popup.adjustSize()
        popup.move(anchor.x() - popup.width(), anchor.y() + scaled_px(2))
        popup.show()

    def set_status(self, text: str, *, error: bool) -> None:
        self.status.setText(str(text)[:200])
        self.status.setStyleSheet(
            f"color: {RED if error else GREY}; background: transparent;")
        colour = RED if error else GREY
        self.setting_button.set_color(colour)
        self.setting_button.setToolTip(f"Panel settings — {text}" if text else "Panel settings")
        self.footer.setText(f"{text}   ·   {self.config.source}"[:160])
        self.footer.setStyleSheet(f"color: {colour}; background: transparent;")

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
        Every failure lands on the gear/status -- a bad expression in one panel
        must never break the console or its siblings."""

        try:
            value = self._evaluate(dict(namespace))
            self._feed(value)
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
        return flat

    def _feed(self, value) -> None:
        # PERFORMANCE: feed data with draw=False and queue ONE draw_idle per
        # panel -- rendering happens in Qt's paint pass (coalesced per frame),
        # never synchronously inside the refresh tick.  hist accepts any sample
        # count (HistogramFigure.update rebins itself), so a growing history
        # must NOT rebuild the whole plot every shot.
        value = self._coerce(value)
        kind = self.config.kind
        rebuild = self.plotter is None
        if not rebuild and kind in ("2d", "1d"):
            rebuild = tuple(np.shape(value)) != self._value_shape
        if rebuild:
            self._build_plot(value)
            self._value_shape = (1,) if isinstance(value, float) else tuple(np.shape(value))
            return
        if kind == "2d":
            self.plotter.update(np.asarray(value).ravel(), draw=False)
        elif kind == "monitor":
            self.plotter.roll(value, draw=False)
        else:  # hist / 1d
            self.plotter.update(value, draw=False)
        if self.canvas is not None:
            self.canvas.draw_idle()

    # ------------------------------------------------------------- plot lifecycle
    def _build_plot(self, value) -> None:
        if panel_canvas is None:
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
                labels=("X (px)", "Y (px)", ""), title=self.config.title or None)
        elif kind == "monitor":
            length = max(20, int(self.config.params.get("length", 300)))
            history = np.full(length, np.nan)
            self.plotter = panel_plot(
                np.arange(length, dtype=float), history, kind="monitor", size=size,
                labels=("Shots ago", label, "Z"), relim_mode="tight",
                title=self.config.title or None)
            self.plotter.roll(float(value), draw=False)
        elif kind == "hist":
            self.plotter = panel_plot(
                np.asarray(value, dtype=float), kind="hist", size=size,
                bins=int(self.config.params.get("bins", 60)),
                labels=("Value", "Shots", "Population"), title=self.config.title or None)
        else:  # 1d
            vec = np.asarray(value, dtype=float).reshape(-1)
            self.plotter = panel_plot(
                np.arange(len(vec), dtype=float), vec, kind="1d", size=size,
                labels=("Site", label, "Z"), relim_mode="tight",
                title=self.config.title or None)
        self.canvas = panel_canvas(self.plotter.fig)
        self.canvas_holder.insertWidget(0, self.canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        if self.canvas_holder.indexOf(self.footer) < 0:
            self.canvas_holder.addStretch(1)
            self.canvas_holder.addWidget(self.footer)
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
        self.board = _PanelBoard()
        self.scroll.setWidget(self.board)
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
                self._attach_card(PanelCard(config, parent=self.board))
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
        self.cards.append(card)

    def _arrange(self) -> None:
        self.board.arrange(self.cards)
        self._update_summary()

    # ------------------------------------------------------------------ actions
    def _add_panel(self) -> None:
        kind = self.kind_combo.currentData() or "1d"
        rows = max((c.config.row + c.config.rows for c in self.cards), default=0)
        config = PanelConfig(kind=str(kind), row=rows, col=0, size="1x2")
        card = PanelCard(config, parent=self.board)
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
