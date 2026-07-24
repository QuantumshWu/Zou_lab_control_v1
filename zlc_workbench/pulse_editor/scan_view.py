"""Frozen Scan tab for the formal Pulse GUI.

The widget owns only Qt editing buffers and presentation.  Current scan-table
validation, execution, progress observation, and hold/step intents belong to
the application controller behind its signals.
"""

from __future__ import annotations

import sys

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    YELLOW,
    FluentButton,
    FluentCodeEdit,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    signals_blocked,
)
from zlc_pulse import DEFAULT_SCAN_SWEEP_COUNT, MIN_SCAN_SWEEP_COUNT

from ._layout import px, row_height


def _format_scan_values(
    values: tuple[tuple[str, int | float], ...],
) -> str:
    return ", ".join(
        f"{name} = {float(value):g}" for name, value in values
    )


def format_scan_progress(
    progress: object | None,
    *,
    values: tuple[tuple[str, int | float], ...] = (),
) -> str:
    """Format the established operator-facing scan cursor text.

    Current sequencers expose the row they own, but the frozen RTL does not
    expose a completed-sweep counter.  The typed path therefore shows only
    facts backed by the device instead of inventing a sweep number.  The
    mapping path remains accepted by this pure presenter for transports that
    do expose the established sweep fields.
    """

    if not progress:
        return ""
    getter = getattr(progress, "get", None)
    try:
        if getter is None and hasattr(progress, "current_point_index"):
            point_value = getattr(progress, "current_point_index")
            if point_value is None:
                return ""
            n_points = int(getattr(progress, "point_count"))
            point = int(point_value)
            if n_points <= 0:
                return ""
            point_1 = min(max(point + 1, 1), n_points)
            text = f"Scan: point {point_1} / {n_points}"
            value_text = _format_scan_values(values)
            return text + (f"  {value_text}" if value_text else "")
        if getter is None:
            scanning = bool(getattr(progress, "scanning"))
            n_points = int(getattr(progress, "n_points"))
            point = int(getattr(progress, "point"))
            sweep = int(getattr(progress, "sweep"))
            n_repeats = int(getattr(progress, "n_repeats"))
        else:
            scanning = bool(getter("scanning"))
            n_points = int(getter("n_points", 0))
            point = int(getter("point", 0))
            sweep = int(getter("sweep", 0))
            n_repeats = int(getter("n_repeats", 0))
    except (AttributeError, TypeError, ValueError):
        return ""
    if not scanning or n_points <= 0:
        return ""
    point_1 = min(max(point + 1, 1), n_points)
    text = f"Scan: point {point_1} / {n_points} · sweep {sweep + 1}"
    if n_repeats > 0:
        text += f" / {n_repeats}"
    value_text = _format_scan_values(values)
    return text + (f"  {value_text}" if value_text else "")


def format_held_scan_point(
    held: tuple[
        int,
        int,
        tuple[tuple[str, int | float], ...],
    ]
    | None,
) -> str:
    """Format the frozen formal held-point readout from immutable state."""

    if held is None:
        return ""
    point, total, values = held
    return (
        f"held at point {int(point) + 1}/{int(total)}: "
        f"{_format_scan_values(values)}"
    )


class PulseScanView(QtWidgets.QWidget):
    """Exact formal Scan layout with no hardware or document authority."""

    repeatsChanged = QtCore.pyqtSignal(int)
    holdRequested = QtCore.pyqtSignal()
    stepRequested = QtCore.pyqtSignal(int)
    loadProgramRequested = QtCore.pyqtSignal()
    templateRequested = QtCore.pyqtSignal(str)
    runRequested = QtCore.pyqtSignal(str)
    saveArrayRequested = QtCore.pyqtSignal()
    progressRefreshRequested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QtWidgets.QVBoxLayout(self)
        margin = px(8, minimum=5)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(px(8, minimum=5))

        info = FluentFrame()
        info.setMinimumHeight(px(64, minimum=52))
        info_layout = QtWidgets.QVBoxLayout(info)
        info_layout.setContentsMargins(px(12), px(8), px(12), px(8))
        self.scan_slots_label = FluentLabel("")
        self.scan_slots_label.setWordWrap(True)
        info_layout.addWidget(self.scan_slots_label)

        run_row = QtWidgets.QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(px(6, minimum=4))
        run_row.addWidget(FluentLabel("Scan repeats (0 = ∞)"))
        self.scan_repeats_spin = FluentDoubleSpinBox(
            length=5,
            allow_minus=False,
        )
        self.scan_repeats_spin.setDecimals(0)
        self.scan_repeats_spin.setMinimum(MIN_SCAN_SWEEP_COUNT)
        self.scan_repeats_spin.setMaximum(sys.float_info.max)
        self.scan_repeats_spin.setValue(DEFAULT_SCAN_SWEEP_COUNT)
        self._committed_repeats = DEFAULT_SCAN_SWEEP_COUNT
        self.scan_repeats_spin.setFixedHeight(row_height())
        self.scan_repeats_spin.setToolTip(
            "How many WHOLE scan sweeps to play before stopping.  0 = sweep "
            "forever (the default); K ≥ 1 = K full sweeps then stop.  The "
            "device performs the finite stop."
        )
        self.scan_repeats_spin.editingFinished.connect(self._commit_repeats)
        run_row.addWidget(self.scan_repeats_spin)

        self.scan_hold_button = FluentButton("Stop ▸ hold point", color=RED)
        self.scan_hold_button.setFixedHeight(row_height())
        self.scan_hold_button.setToolTip(
            "Stop the running scan and HOLD the CURRENT scan point: reloads a "
            "single-point pulse built from that point's values and loops it "
            "forever.  Re-run the Scan to resume sweeping."
        )
        self.scan_hold_button.clicked.connect(self.holdRequested)
        run_row.addWidget(self.scan_hold_button)

        self.scan_step_back_button = FluentButton("◀ step", color=ORANGE)
        self.scan_step_back_button.setToolTip(
            "Step the HELD scan point one row BACK in the scan table (clamped "
            "at point 1, no wrap).  If the scan is still running, stops and "
            "holds first -- like Stop ▸ hold point."
        )
        self.scan_step_forward_button = FluentButton("step ▶", color=ORANGE)
        self.scan_step_forward_button.setToolTip(
            "Step the HELD scan point one row FORWARD in the scan table "
            "(clamped at the last point, no wrap).  If the scan is still "
            "running, stops and holds first -- like Stop ▸ hold point."
        )
        for button, delta in (
            (self.scan_step_back_button, -1),
            (self.scan_step_forward_button, 1),
        ):
            button.setFixedHeight(row_height())
            button.clicked.connect(
                lambda _checked=False, value=delta: self.stepRequested.emit(value)
            )
            run_row.addWidget(button)
        self.scan_progress_label = FluentLabel("")
        run_row.addWidget(self.scan_progress_label, 1)
        info_layout.addLayout(run_row)
        layout.addWidget(info)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(px(8, minimum=5))

        editor_box = FluentGroupBox("Generate the scan table (Python)")
        editor_layout = QtWidgets.QVBoxLayout(editor_box)
        editor_layout.setContentsMargins(
            px(8),
            px(28, minimum=24),
            px(8),
            px(8),
        )
        editor_layout.setSpacing(px(6, minimum=4))
        self.scan_code = FluentCodeEdit()
        self._code_dirty = False
        self._source_revision = -1

        template_buttons = QtWidgets.QHBoxLayout()
        template_buttons.setSpacing(px(6, minimum=4))
        self.scan_load_program_button = FluentButton(
            "Load Program",
            color=ACCENT,
        )
        self.scan_load_program_button.setToolTip(
            "Load a Python program (.py) into the editor."
        )
        self.scan_load_program_button.clicked.connect(self.loadProgramRequested)
        self.scan_column_template_button = FluentButton(
            "Template: column_stack",
            color=GREY,
        )
        self.scan_column_template_button.setToolTip(
            "Insert the column_stack template (one column per slot), adapted "
            "to the bound slots."
        )
        self.scan_column_template_button.clicked.connect(
            lambda: self.templateRequested.emit("column_stack")
        )
        self.scan_grid_template_button = FluentButton(
            "Template: grid",
            color=GREY,
        )
        self.scan_grid_template_button.setToolTip(
            "Insert the grid template (outer product of two arrays), adapted "
            "to the bound slots."
        )
        self.scan_grid_template_button.clicked.connect(
            lambda: self.templateRequested.emit("grid")
        )
        for button in (
            self.scan_load_program_button,
            self.scan_column_template_button,
            self.scan_grid_template_button,
        ):
            button.setFixedHeight(row_height())
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            template_buttons.addWidget(button, 1)
        editor_layout.addLayout(template_buttons)
        editor_layout.addWidget(self.scan_code, 1)

        code_buttons = QtWidgets.QHBoxLayout()
        code_buttons.setSpacing(px(6, minimum=4))
        self.scan_run_button = FluentButton("Run", color=GREEN)
        self.scan_run_button.setFixedHeight(row_height())
        self.scan_run_button.setToolTip(
            "Run the code; assign an N_points x N_slots array to 'scan_table'."
        )
        self.scan_run_button.clicked.connect(
            lambda: self.runRequested.emit(self.scan_code.toPlainText())
        )
        self.scan_code.textChanged.connect(self._on_code_changed)
        self.scan_save_array_button = FluentButton("Save Array", color=YELLOW)
        self.scan_save_array_button.setFixedHeight(row_height())
        self.scan_save_array_button.setToolTip(
            "Save the generated scan table to a .npy/.csv file."
        )
        self.scan_save_array_button.clicked.connect(self.saveArrayRequested)
        for button in (self.scan_run_button, self.scan_save_array_button):
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed,
            )
            code_buttons.addWidget(button, 1)
        editor_layout.addLayout(code_buttons)
        body.addWidget(editor_box, 3)

        preview_box = FluentGroupBox("Scan table")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(
            px(8),
            px(28, minimum=24),
            px(8),
            px(8),
        )
        preview_layout.setSpacing(px(6, minimum=4))
        self.scan_table_view = FluentCodeEdit(read_only=True)
        preview_layout.addWidget(self.scan_table_view, 1)
        footer = QtWidgets.QWidget()
        footer.setFixedHeight(row_height())
        footer.setStyleSheet("background: transparent;")
        preview_layout.addWidget(footer)
        body.addWidget(preview_box, 2)
        layout.addLayout(body, 1)

        self._scan_progress_timer = QtCore.QTimer(self)
        self._scan_progress_timer.setInterval(200)
        self._scan_progress_timer.timeout.connect(self._request_visible_progress)

    def _on_code_changed(self) -> None:
        self._code_dirty = True
        self.scan_run_button.set_dirty(True)

    def _request_visible_progress(self) -> None:
        if self.isVisible():
            self.progressRefreshRequested.emit()

    def set_progress_polling(self, enabled: bool) -> None:
        """Run the narrow progress watcher only for a visible active scan."""

        should_run = bool(enabled)
        if should_run and not self._scan_progress_timer.isActive():
            self._scan_progress_timer.start()
        elif not should_run and self._scan_progress_timer.isActive():
            self._scan_progress_timer.stop()

    def set_repeats(self, repeats: int) -> None:
        normalized = int(repeats)
        with signals_blocked(self.scan_repeats_spin):
            if int(self.scan_repeats_spin.value()) != normalized:
                self.scan_repeats_spin.setValue(normalized)
        self._committed_repeats = normalized

    def _commit_repeats(self) -> None:
        value = int(self.scan_repeats_spin.value())
        if value == self._committed_repeats:
            return
        self.repeatsChanged.emit(value)

    def set_slots_text(self, text: str) -> None:
        self.scan_slots_label.setText(str(text))

    def set_progress_text(self, text: str) -> None:
        self.scan_progress_label.setText(str(text))

    @property
    def code_dirty(self) -> bool:
        return self._code_dirty

    @property
    def source_revision(self) -> int:
        return self._source_revision

    def set_scan_code(
        self,
        source: str,
        *,
        dirty: bool = False,
        source_revision: int,
    ) -> None:
        with signals_blocked(self.scan_code):
            self.scan_code.setPlainText(str(source))
        self._source_revision = int(source_revision)
        self.set_run_dirty(dirty)

    def replace_scan_draft(self, source: str) -> None:
        """Replace the UI-owned draft without committing controller state."""

        with signals_blocked(self.scan_code):
            self.scan_code.setPlainText(str(source))
        self.set_run_dirty(True)

    def acknowledge_scan_draft(self, *, dirty: bool, source_revision: int) -> None:
        """Advance committed metadata without rebuilding the editor widget."""

        self._source_revision = int(source_revision)
        self.set_run_dirty(dirty)

    def set_scan_table_text(self, text: str) -> None:
        self.scan_table_view.setPlainText(str(text))

    def set_run_dirty(self, dirty: bool) -> None:
        self._code_dirty = bool(dirty)
        self.scan_run_button.set_dirty(self._code_dirty)

    def set_workspace_busy(self, busy: bool) -> None:
        """Disable only workspace admissions while its worker owns one intent."""

        enabled = not bool(busy)
        for button in (
            self.scan_load_program_button,
            self.scan_column_template_button,
            self.scan_grid_template_button,
            self.scan_run_button,
            self.scan_save_array_button,
        ):
            button.setEnabled(enabled)


__all__ = ["PulseScanView", "format_scan_progress"]
