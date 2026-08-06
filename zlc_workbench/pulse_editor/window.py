"""Formal Pulse GUI composition over current document/application owners.

The operator-facing surface is frozen: this module composes the mechanically
extracted Edit, Preview, and Scan pages without changing their controls,
wording, geometry, or workflow.  It owns only Qt coordination.  Pulse meaning,
file state, preview rendering, connection authority, and Run lifecycle remain
behind :class:`PulseEditorController`.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, TimeoutError
from dataclasses import replace as dataclass_replace
import os
from pathlib import Path
import threading
from typing import Callable

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    GREEN,
    GREY,
    ORANGE,
    RED,
    WINDOW_SCREEN_FRACTION,
    YELLOW,
    FluentButton,
    FluentFrame,
    FluentLabel,
    FluentLineEdit,
    FluentStatusDot,
    FluentTabWidget,
    QtOwnerWake,
    ensure_qt_app,
    fluent_confirm,
    fluent_message,
    fluent_widget_stylesheet,
    launch_fluent_window,
    release_window,
    screen_fit_window_size,
    set_fluent_scale,
    wait_for_owner_retirement,
    window_pad,
)
from zlc_plot import PulseTimelinePlot, Qt5PlotWidget, RasterPlotHost
from zlc_workbench.retiring_hosts import RetiringPlotHosts
from zlc_neutral_atom.runtime.run import RunState
from zlc_pulse import (
    DestructivePulseTargetEditError,
    PulseDocument,
    PulseExecutionForm,
    scan_column_specs,
)

from ._layout import px
from .controller import (
    PulseEditorController,
    PulseFileUpdate,
    PulseOwnerUpdate,
    PulseEditorProjection,
    PulsePreviewPlot,
    PulsePreviewUpdate,
    PulseRuntimeUpdate,
    PulseScanProgressUpdate,
)
from .preview_view import PulsePreviewView
from .scan_view import (
    PulseScanView,
    format_held_scan_point,
    format_scan_progress,
)
from .schedule_view import PulseScheduleView
from .target_view import PulseTargetView

def _pulse_files_dir(root: Path) -> Path:
    directory = Path(root).expanduser()
    if not directory.is_absolute():
        raise ValueError("Pulse files root must be absolute")
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def _pulse_figure_dir(root: Path) -> Path:
    directory = Path(root).expanduser() / "figures" / "pulses"
    if not directory.is_absolute():
        raise ValueError("Pulse output root must be absolute")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_file_stem(value: str) -> str:
    result = []
    for character in str(value).strip():
        if character.isalnum() or character in ("-", "_"):
            result.append(character)
        elif character.isspace():
            result.append("_")
    return "".join(result).strip("_") or "pulse"


class PulseEditorWindowBody(QtWidgets.QWidget):
    """The one formal Pulse GUI body; no alternate editor surface exists."""

    def __init__(
        self,
        controller: PulseEditorController,
        *,
        pulses_root: Path,
        output_root: Path,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not isinstance(controller, PulseEditorController):
            raise TypeError("controller must be PulseEditorController")
        ensure_qt_app()
        super().__init__(parent)
        self._pulses_root = Path(pulses_root).expanduser().resolve()
        self._output_root = Path(output_root).expanduser().resolve()
        self._controller = controller
        self._window = None
        self._last_runtime: PulseRuntimeUpdate | None = None
        self._last_editor_projection: PulseEditorProjection | None = None
        self._last_scan_workspace = None
        self._last_scan_progress_key = None
        self._last_preview_key = None
        self._shown_preview_key: tuple[int, int, bool] | None = None
        self.preview_host: RasterPlotHost | None = None
        self.preview_widget: Qt5PlotWidget | None = None
        self._preview_host_unsubscribe: Callable[[], None] | None = None
        self._preview_spec: PulseTimelinePlot | None = None
        self._preview_plot: PulsePreviewPlot | None = None
        self._preview_size: str | None = None
        self._preview_operations: dict[Future, str | None] = {}
        self._retiring_hosts = RetiringPlotHosts()
        self._close_decided = False
        self._close_requested = False
        self._teardown_pending = False
        self._owner_retiring = False
        self._permanently_closed = False
        self._owner_closed = threading.Event()

        set_fluent_scale(None)
        self.setWindowTitle("PulseGUI@Zou lab")
        self.setFixedSize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
        self.setStyleSheet(fluent_widget_stylesheet())
        initial_editor = controller.editor_projection()
        initial_runtime = controller.runtime_update()
        initial_preview = controller.preview_update()
        self._build_ui(initial_editor, initial_runtime)
        self._wire_ui()

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._controller.set_notify(self._wake.request_owner_wake)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        # The clock only asks for an owner turn.  Advancing is ``pump``'s job
        # alone, and QtOwnerWake is the one authority that decides whether a
        # turn may start, so a tick raised inside a nested modal loop coalesces
        # instead of re-entering the owner.
        self._timer.timeout.connect(self._wake.request_owner_wake)
        self._apply_initial_state(initial_editor, initial_runtime, initial_preview)

    # ------------------------------------------------------------------
    # frozen composition
    # ------------------------------------------------------------------

    def _build_ui(
        self,
        editor: PulseEditorProjection,
        runtime: PulseRuntimeUpdate,
    ) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(
            window_pad(1),
            window_pad(1),
            window_pad(1),
            window_pad(1),
        )
        root.setSpacing(window_pad(0.5))

        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(px(12), px(6), px(12), px(6))
        header.setSpacing(px(8, minimum=5))
        self.status_dot = FluentStatusDot(size=16)
        self.label_name = FluentLabel("PulseGUI - Untitled*")
        self.label_name.setMinimumWidth(px(260, minimum=180))
        self.label_name.setAlignment(
            QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft
        )
        self.summary = FluentLineEdit("")
        self.summary.setEnabled(False)
        self.clear_all_button = FluentButton("Clear All", color=ORANGE)
        self.clear_all_button.setToolTip(
            "Reset the schedule: remove every period and every channel delay, "
            "leaving one blank 1 µs period with no channel on.\n"
            "The sequencer-owned PortCatalog and current visibility are kept."
        )
        header.addWidget(self.status_dot)
        header.addWidget(self.label_name)
        header.addWidget(self.summary, 1)
        header.addWidget(self.clear_all_button)
        root.addWidget(header_frame)

        self.tabs = FluentTabWidget()
        self.schedule_view = PulseScheduleView(
            editor.document,
            editor.target_manifest,
            display_visible_ports=editor.display_visible_ports,
            document_generation=editor.document_generation,
            revision=editor.editor_revision,
        )
        self.preview_view = PulsePreviewView()
        self.scan_view = PulseScanView()
        self.target_view = PulseTargetView(
            editor.target_manifest,
            editable=(
                runtime.target_descriptor is None
                and runtime.connection_state == "offline"
                and runtime.connection_mode == "offline"
            ),
            mode=runtime.connection_mode,
        )
        self.tabs.addTab(self.schedule_view, "Edit")
        self.tabs.addTab(self.preview_view, "Preview")
        self.tabs.addTab(self.scan_view, "Scan")
        self.tabs.addTab(self.target_view, "Target")
        root.addWidget(self.tabs, 1)

    def _wire_ui(self) -> None:
        view = self.schedule_view
        view.documentNameEdited.connect(self._edit_document_name)
        view.portLabelEdited.connect(self._edit_port_label)
        view.periodNameEdited.connect(self._edit_period_name)
        view.durationEdited.connect(self._edit_duration)
        view.digitalEdited.connect(self._edit_digital)
        view.analogEdited.connect(self._edit_analog)
        view.delayEdited.connect(self._edit_delay)
        view.bindingCycleRequested.connect(self._edit_binding)
        view.insertPeriodRequested.connect(self._insert_period)
        view.movePeriodRequested.connect(self._move_period)
        view.removePeriodRequested.connect(self._remove_period)
        view.repeatEdited.connect(self._edit_repeat)
        view.visiblePortsEdited.connect(self._edit_visible_ports)
        view.clearPortRequested.connect(self._clear_port)
        view.clearAllRequested.connect(self._clear_all)
        self.clear_all_button.clicked.connect(self._clear_all)
        view.runRequested.connect(self._run_from_edit)
        view.stopRequested.connect(
            lambda: self._invoke_runtime(self._controller.cancel)
        )
        view.syncRequested.connect(self._sync_applied)
        view.feedbackRequested.connect(self._message)
        view.saveRequested.connect(self._save_document)
        view.loadRequested.connect(self._load_document)
        view.connectionRequested.connect(
            self._invoke_connection
        )
        view.scanArrayLoadRequested.connect(self._load_scan_array)
        view.scanSourceEdited.connect(self._select_scan_source)
        self.target_view.feedbackRequested.connect(self._message)
        self.target_view.applyRequested.connect(self._apply_target_manifest)

        self.preview_view.includeOffToggled.connect(
            lambda checked: self._invoke_preview(
                self._controller.set_preview_include_off, checked
            )
        )
        self.preview_view.selectorsToggled.connect(
            self._set_preview_interaction
        )
        self.preview_view.sizeActivated.connect(
            self._set_preview_size
        )
        self.preview_view.saveFigureRequested.connect(self._save_preview)

        self.scan_view.repeatsChanged.connect(
            self._edit_scan_sweep_count
        )
        self.scan_view.holdRequested.connect(self._hold_scan_point)
        self.scan_view.stepRequested.connect(self._step_scan_point)
        self.scan_view.loadProgramRequested.connect(self._load_scan_program)
        self.scan_view.templateRequested.connect(self._replace_scan_template)
        self.scan_view.runRequested.connect(
            lambda source: self._invoke_scan_worker(
                self._controller.generate_scan_source, source
            )
        )
        self.scan_view.saveArrayRequested.connect(self._save_scan_array)
        self.scan_view.progressRefreshRequested.connect(
            self._request_scan_progress
        )
        self.tabs.currentChanged.connect(self._tab_changed)

    # ------------------------------------------------------------------
    # Qt intents
    # ------------------------------------------------------------------

    def _invoke_worker(self, action: Callable, *args, **kwargs):
        """Start one worker command and present only its immediate busy state."""

        try:
            result = action(*args, **kwargs)
        except BaseException as error:
            self._message(str(error))
            return None
        update = self._controller.runtime_update()
        self._apply_runtime_update(update)
        self._finish_close_if_ready(update)
        return result

    def _invoke_scan_worker(self, action: Callable, *args, **kwargs):
        """Start Scan work and update only the Scan workspace in place."""

        try:
            result = action(*args, **kwargs)
        except BaseException as error:
            self._message(str(error))
            return None
        workspace = self._controller.current_scan_workspace
        self._apply_scan_workspace_value(
            workspace,
            self._last_scan_workspace,
        )
        self._last_scan_workspace = workspace
        return result

    def _invoke_connection(self, mode: str, endpoint: str) -> None:
        """Apply a connection intent without projecting unrelated surfaces."""

        try:
            editor_boundary = self._controller.connect(mode, endpoint)
        except BaseException as error:
            self._message(str(error))
            return
        update = self._controller.runtime_update()
        self._apply_runtime_update(update)
        if editor_boundary:
            self._apply_editor_projection(self._controller.editor_projection())

    def _invoke_editor_boundary(self, action: Callable, *args, **kwargs):
        """Apply a command that intentionally replaces document/target authority."""

        previous = self._last_editor_projection
        if previous is None:
            raise RuntimeError("Pulse editor command preceded initial composition")
        try:
            result = action(*args, **kwargs)
        except BaseException as error:
            self._message(str(error))
            return None
        update = self._controller.runtime_update()
        self._apply_runtime_update(update)
        if result != previous.editor_revision:
            self._apply_editor_projection(self._controller.editor_projection())
        return result

    def _commit_local_edit(
        self,
        action: Callable[[], object],
        present: Callable[[object, object, object], None],
    ):
        """Commit one synchronous Qt intent and update only its known dependents."""

        previous_editor = self._last_editor_projection
        runtime = self._last_runtime
        if previous_editor is None or runtime is None:
            raise RuntimeError("Pulse edit preceded initial composition")
        before = self._controller.current_document
        try:
            result = action()
        except BaseException as error:
            self._message(str(error))
            return None
        if result is None:
            return None

        after = self._controller.current_document
        generation = self._controller.current_document_generation
        revision = self._controller.current_editor_revision
        self.schedule_view.accept_local_commit(
            document_generation=generation,
            revision=revision,
        )
        present(before, after, result)
        self.schedule_view.refresh_summary(after)
        self.summary.setText(self.schedule_view.summary_text())

        workspace = self._controller.current_scan_workspace
        if workspace != self._last_scan_workspace:
            self._apply_scan_workspace_value(workspace, self._last_scan_workspace)
            self._last_scan_workspace = workspace

        self._apply_file_and_run_state_values(
            name=after.name,
            path=previous_editor.path,
            file_state=previous_editor.file_state,
            dirty=self._controller.dirty,
            document=after,
            runtime=runtime,
        )
        visible = self._controller.current_display_visible_ports
        self._last_editor_projection = dataclass_replace(
            previous_editor,
            document=after,
            document_generation=generation,
            editor_revision=revision,
            dirty=self._controller.dirty,
            display_visible_ports=visible,
            scan_workspace=workspace,
        )
        if (
            revision != previous_editor.editor_revision
            and self.tabs.currentWidget() is self.preview_view
        ):
            self._controller.request_preview()
        return result

    def _edit_document_name(self, value: str) -> None:
        self._commit_local_edit(
            lambda: self._controller.rename_document(value),
            lambda _before, after, _result: self.schedule_view.apply_document_name(after),
        )

    def _edit_port_label(self, port: str, value: str) -> None:
        self._commit_local_edit(
            lambda: self._controller.rename_port(port, value),
            lambda _before, after, _result: self.schedule_view.apply_port_label(
                after, port
            ),
        )

    def _edit_period_name(self, period: str, value: str) -> None:
        self._commit_local_edit(
            lambda: self._controller.rename_period(period, value),
            lambda _before, after, _result: self.schedule_view.apply_period_name(
                after, period
            ),
        )

    def _edit_duration(self, period: str, value, unit: str) -> None:
        self._commit_local_edit(
            lambda: self._controller.set_period_duration(period, value, unit),
            lambda _before, after, _result: self.schedule_view.apply_duration(
                after, period
            ),
        )

    def _edit_digital(self, period: str, port: str, high: bool) -> None:
        self._commit_local_edit(
            lambda: self._controller.set_digital(period, port, high),
            lambda _before, after, _result: self.schedule_view.apply_digital(
                after, period, port
            ),
        )

    def _edit_analog(self, period: str, port: str, mode: str, value) -> None:
        def present(before, after, _result) -> None:
            if (
                before.scan_parameters,
                before.api_parameters,
            ) != (
                after.scan_parameters,
                after.api_parameters,
            ):
                self.schedule_view.apply_all_bindings(after)
            else:
                self.schedule_view.apply_analog_port(after, port)

        self._commit_local_edit(
            lambda: self._controller.set_analog(
                period,
                port,
                mode,
                value,
                cascade=True,
            ),
            present,
        )

    def _edit_delay(self, port: str, value, unit: str) -> None:
        def present(before, after, _result) -> None:
            if (
                before.scan_parameters,
                before.api_parameters,
            ) != (
                after.scan_parameters,
                after.api_parameters,
            ):
                self.schedule_view.apply_all_bindings(after)
            else:
                self.schedule_view.apply_delay(after, port)

        self._commit_local_edit(
            lambda: self._controller.set_delay(
                port,
                value,
                unit,
                cascade=True,
            ),
            present,
        )

    def _edit_binding(self, field) -> None:
        self._commit_local_edit(
            lambda: self._controller.cycle_binding(field),
            lambda _before, after, _result: self.schedule_view.apply_all_bindings(after),
        )

    def _insert_period(self, before: str | None) -> None:
        self._commit_local_edit(
            lambda: self._controller.add_period(
                before_period_id=before,
                cascade=True,
            ),
            lambda _before, after, _result: self.schedule_view.apply_period_structure(
                after
            ),
        )

    def _move_period(self, period: str, before: str | None) -> None:
        self._commit_local_edit(
            lambda: self._controller.move_period(
                period,
                before,
                cascade=True,
            ),
            lambda _old, after, _result: self.schedule_view.apply_period_structure(after),
        )

    def _remove_period(self, period: str) -> None:
        self._commit_local_edit(
            lambda: self._controller.remove_period(period, cascade=True),
            lambda _before, after, _result: self.schedule_view.apply_period_structure(
                after
            ),
        )

    def _edit_repeat(self, start, end, count: int) -> None:
        self._commit_local_edit(
            lambda: self._controller.set_repeat(start, end, count),
            lambda _before, after, _result: self.schedule_view.apply_period_structure(
                after
            ),
        )

    def _edit_visible_ports(self, ports) -> None:
        self._commit_local_edit(
            lambda: self._controller.set_visible_ports(ports),
            lambda _before, _after, visible: self.schedule_view.apply_visible_ports(
                visible
            ),
        )

    def _clear_port(self, port: str) -> None:
        self._commit_local_edit(
            lambda: self._controller.clear_port(port),
            lambda _before, after, _result: self.schedule_view.apply_port_clear(
                after, port
            ),
        )

    def _clear_all(self) -> None:
        self._commit_local_edit(
            self._controller.clear_all,
            lambda _before, after, _result: self.schedule_view.apply_clear_all(after),
        )

    def _edit_scan_sweep_count(self, repeats: int) -> None:
        self._commit_local_edit(
            lambda: self._controller.set_scan_sweep_count(int(repeats)),
            lambda _before, after, _result: self.scan_view.set_repeats(
                after.scan_sweep_count
            ),
        )

    def _invoke_runtime(self, action: Callable, *args, **kwargs):
        """Commit one same-thread Run command through the narrow runtime view."""

        try:
            result = action(*args, **kwargs)
        except BaseException as error:
            self._message(str(error))
            return None
        update = self._controller.runtime_update()
        self._apply_runtime_update(update)
        self._finish_close_if_ready(update)
        return result

    def _replace_scan_template(self, kind: str) -> None:
        """Unconditionally replace the local Scan draft with a fresh template."""

        try:
            source = self._controller.scan_template_source(kind)
        except BaseException as error:
            self._message(str(error))
            return
        self.scan_view.replace_scan_draft(source)

    def _select_scan_source(self, use_loaded: bool) -> None:
        """Commit a source toggle, restoring the displayed switch on rejection."""

        self._commit_local_edit(
            lambda: self._controller.select_scan_source(
                "loaded" if use_loaded else "generated"
            ),
            lambda _before, _after, _result: None,
        )
        workspace = self._controller.current_scan_workspace
        self.schedule_view.set_scan_source(
            use_loaded=workspace.selected_source == "loaded",
            path="" if workspace.loaded_path is None else str(workspace.loaded_path),
        )

    def _invoke_preview(self, action: Callable, *args, **kwargs):
        """Submit presentation work; the render completion is its only wake."""

        try:
            return action(*args, **kwargs)
        except BaseException as error:
            self._message(str(error))
            return None

    def _message(self, text: str) -> None:
        message = str(text)
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            self.summary.setText(message)
            self.preview_view.set_status(message)
            return
        fluent_message(self, "Pulse", message, kind="warning")

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.preview_view:
            self.preview_view.reset_preview_size_pin()
            self._apply_recommended_preview_size()
            self._invoke_preview(self._controller.request_preview)
        self._sync_runtime_watchers()

    def _apply_target_manifest(self, manifest) -> None:
        try:
            self._controller.apply_target_manifest(manifest)
        except DestructivePulseTargetEditError as error:
            references = "\n".join(
                f"• {value}" for value in error.impact.cleared_references
            )
            if not fluent_confirm(
                self,
                "Pulse Target",
                "The new target removes or changes ports used by this pulse. "
                "Apply and clear these references?\n\n" + references,
                confirm_text="Apply and clear",
                cancel_text="Keep draft",
            ):
                return
            try:
                self._controller.apply_target_manifest(manifest, cascade=True)
            except BaseException as error:
                self._message(str(error))
                return
            self._apply_editor_projection(self._controller.editor_projection())
            return
        except BaseException as error:
            self._message(str(error))
            return
        self._apply_editor_projection(self._controller.editor_projection())

    def _run_from_edit(self) -> None:
        document = self._controller.current_document
        if document.scan_parameters:
            form = (
                PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS
                if document.scan_sweep_count == 0
                else PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
            )
        else:
            form = PulseExecutionForm.CONTINUOUS_MONITOR
        self._invoke_runtime(
            self._controller.start,
            form,
            scan_sweep_count=max(1, document.scan_sweep_count),
        )

    def _sync_applied(self) -> None:
        self._invoke_editor_boundary(self._controller.sync_applied)

    def _hold_scan_point(self) -> None:
        self._invoke_runtime(self._controller.hold_scan_point)

    def _step_scan_point(self, delta: int) -> None:
        self._invoke_runtime(self._controller.step_scan_point, int(delta))

    def _scan_file_start(self) -> Path:
        path = self._controller.current_path
        return (
            path.parent
            if path is not None
            else _pulse_files_dir(self._pulses_root)
        )

    def _request_scan_progress(self) -> None:
        """Ask for one worker observation without waking the whole window."""

        try:
            self._controller.request_scan_progress()
        except BaseException as error:
            self._message(str(error))

    def _load_scan_array(self) -> None:
        if not self._controller.current_document.scan_parameters:
            self._message(
                "Bind at least one field to a scan slot (click a dot) before "
                "loading an array."
            )
            return
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load scan array",
            str(self._scan_file_start()),
            "Scan array (*.npy *.csv *.txt *.json)",
        )
        if path:
            self._invoke_scan_worker(self._controller.load_scan_array, Path(path))

    def _load_scan_program(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load scan program / table",
            str(self._scan_file_start()),
            "Scan program or current table (*.py *.txt *.npy *.csv *.json);;"
            "Python scan program (*.py *.txt);;"
            "Scan array (*.npy *.csv);;"
            "Current PulseDocument with scan table (*.json)",
        )
        if path:
            self._invoke_scan_worker(self._controller.load_scan_program, Path(path))

    def _save_scan_array(self) -> None:
        workspace = self._controller.current_scan_workspace
        document = self._controller.current_document
        if workspace.selected_table is None:
            self._message(
                "No scan table to save yet. Run code or load a file first."
            )
            return
        suggested = self._scan_file_start() / (
            f"{_safe_file_stem(document.name)}_scan.npy"
        )
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save scan array",
            str(suggested),
            "Scan array (*.npy *.csv)",
        )
        if path:
            self._invoke_scan_worker(self._controller.save_scan_array, Path(path))

    def _save_document(self) -> None:
        document = self._controller.current_document
        suggested = self._controller.current_path or (
            _pulse_files_dir(self._pulses_root)
            / f"{_safe_file_stem(document.name)}.json"
        )
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save pulse",
            str(suggested),
            "ZLC pulse (*.json)",
        )
        if path:
            self._invoke_worker(self._controller.save, Path(path), overwrite=True)

    def _load_document(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load pulse",
            str(_pulse_files_dir(self._pulses_root)),
            "ZLC pulse (*.json)",
        )
        if path:
            self._invoke_worker(self._controller.open_path, Path(path))

    def _save_preview(self) -> None:
        host = self.preview_host
        if host is None or self.preview_widget is None:
            self._message("Open Preview and wait for the pulse plot before exporting.")
            return
        document = self._controller.current_document
        suggested = _pulse_figure_dir(self._output_root) / (
            f"{_safe_file_stem(document.name)}.png"
        )
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save pulse figure",
            str(suggested),
            "Pulse figure (*.png)",
        )
        if not path:
            return
        try:
            target = Path(path).expanduser().resolve()
            if not target.suffix:
                target = target.with_suffix(".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            future = host.save(target)
        except BaseException as error:
            self._message(str(error))
            return
        self._track_preview_operation(
            future,
            success=f"Saved figure: {target.name}",
        )

    def _set_preview_interaction(self, enabled: bool) -> None:
        widget = self.preview_widget
        if widget is None:
            return
        try:
            widget.set_interaction_enabled(bool(enabled))
        except BaseException as error:
            self._message(str(error))

    def _set_preview_size(self, size: str) -> None:
        normalized = str(size)
        if normalized == self._preview_size:
            return
        host = self.preview_host
        if host is None:
            return
        try:
            future = host.set_size(normalized)
        except BaseException as error:
            self._message(str(error))
            return
        self._preview_size = normalized
        self._track_preview_operation(future)

    def _apply_recommended_preview_size(self) -> None:
        plot = self._preview_plot
        if plot is None or self.preview_view.preview_size_pinned:
            return
        self.preview_view.set_preview_size(plot.recommended_size, pinned=False)
        self._set_preview_size(plot.recommended_size)

    def _track_preview_operation(
        self,
        future: Future,
        *,
        success: str | None = None,
    ) -> None:
        self._preview_operations[future] = success

        def completed(_future: Future) -> None:
            if not self._permanently_closed:
                self._wake.request_owner_wake()

        future.add_done_callback(completed)

    def _drain_preview_operations(self) -> None:
        completed = tuple(
            future for future in self._preview_operations if future.done()
        )
        for future in completed:
            success = self._preview_operations.pop(future)
            try:
                future.result()
            except CancelledError:
                continue
            except BaseException as error:
                self._shown_preview_key = None
                diagnostic = f"Preview unavailable:\n{type(error).__name__}: {error}"
                self.preview_view.set_status(diagnostic)
                self.preview_view.preview_status.setToolTip(diagnostic)
            else:
                if success:
                    self.preview_view.set_status(success)

    def _preview_surface_changed(self) -> None:
        widget = self.preview_widget
        if widget is None:
            return
        front = widget.presented_front
        if front is None:
            return
        self.preview_view.mount_content(
            widget,
            logical_size=front.logical_size,
            wheel_target=widget,
        )

    def _mount_preview_if_ready(self) -> None:
        if self.preview_widget is not None or self._permanently_closed:
            return
        host = self.preview_host
        if host is None:
            return
        front = host.front
        if front is None:
            try:
                front = host.wait_for_front(timeout=0.0)
            except TimeoutError:
                QtCore.QTimer.singleShot(10, self._mount_preview_if_ready)
                return
            except BaseException as error:
                diagnostic = f"Preview unavailable:\n{type(error).__name__}: {error}"
                self.preview_view.set_status(diagnostic)
                self.preview_view.preview_status.setToolTip(diagnostic)
                return
        try:
            widget = Qt5PlotWidget(host, parent=self.preview_view)
            widget.set_interaction_enabled(
                self.preview_view.preview_selectors_switch.isChecked()
            )
            widget.errorOccurred.connect(self._message)
            widget.surfaceChanged.connect(
                lambda _identity: self._preview_surface_changed()
            )
        except BaseException as error:
            diagnostic = f"Preview unavailable:\n{type(error).__name__}: {error}"
            self.preview_view.set_status(diagnostic)
            self.preview_view.preview_status.setToolTip(diagnostic)
            return
        unsubscribe, self._preview_host_unsubscribe = (
            self._preview_host_unsubscribe,
            None,
        )
        if unsubscribe is not None:
            unsubscribe()
        self.preview_widget = widget
        self._preview_surface_changed()

    # ------------------------------------------------------------------
    # worker/runtime publications
    # ------------------------------------------------------------------

    def _owner_cycle(self) -> None:
        """Drain every owner-visible fact once; the only advancing entrance."""

        publication = self._controller.pump()
        if publication is not None:
            if not isinstance(publication, PulseOwnerUpdate):
                raise TypeError("Pulse owner published an invalid update")
            runtime = publication.runtime
            if runtime is not None:
                self._apply_runtime_update(runtime)
            if publication.editor is not None:
                self._apply_editor_projection(publication.editor)
            elif publication.scan_workspace is not None:
                self._apply_scan_workspace_value(
                    publication.scan_workspace,
                    self._last_scan_workspace,
                )
                self._last_scan_workspace = publication.scan_workspace
            if publication.file is not None:
                self._apply_file_update(publication.file)
            if publication.preview is not None:
                self._present_preview_update(publication.preview)
            if publication.scan_progress is not None:
                self._present_scan_progress(publication.scan_progress)
            if runtime is not None:
                self._finish_close_if_ready(runtime)
        self._drain_preview_operations()
        self._mount_preview_if_ready()
        self._advance_teardown()

    def _reconcile_editor_fields(
        self,
        editor: PulseEditorProjection,
        runtime: PulseRuntimeUpdate,
        previous_editor: PulseEditorProjection | None,
        previous_runtime: PulseRuntimeUpdate | None,
    ) -> bool:
        """Reconcile only document-derived widgets; return revision change."""

        document_key = (
            editor.document_generation,
            editor.editor_revision,
            editor.target_manifest.fingerprint,
            editor.display_visible_ports,
        )
        old_document_key = (
            None
            if previous_editor is None
            else (
                previous_editor.document_generation,
                previous_editor.editor_revision,
                previous_editor.target_manifest.fingerprint,
                previous_editor.display_visible_ports,
            )
        )
        if document_key != old_document_key:
            self.schedule_view.set_document(
                editor.document,
                editor.target_manifest,
                display_visible_ports=editor.display_visible_ports,
                document_generation=editor.document_generation,
                revision=editor.editor_revision,
            )
            self.scan_view.set_repeats(editor.document.scan_sweep_count)
        target_editable = (
            runtime.target_descriptor is None
            and runtime.connection_state == "offline"
            and runtime.connection_mode == "offline"
            and not runtime.file_busy
            and not runtime.run_busy
        )
        target_key = (
            editor.target_manifest.fingerprint,
            target_editable,
            runtime.connection_mode,
        )
        old_target_key = None
        if previous_editor is not None and previous_runtime is not None:
            old_editable = (
                previous_runtime.target_descriptor is None
                and previous_runtime.connection_state == "offline"
                and previous_runtime.connection_mode == "offline"
                and not previous_runtime.file_busy
                and not previous_runtime.run_busy
            )
            old_target_key = (
                previous_editor.target_manifest.fingerprint,
                old_editable,
                previous_runtime.connection_mode,
            )
        if target_key != old_target_key:
            self.target_view.set_manifest(
                editor.target_manifest,
                editable=target_editable,
                mode=runtime.connection_mode,
            )
        if document_key != old_document_key:
            self.summary.setText(self.schedule_view.summary_text())
        revision_changed = previous_editor is None or (
            editor.document_generation,
            editor.editor_revision,
        ) != (
            previous_editor.document_generation,
            previous_editor.editor_revision,
        )
        if revision_changed and self.tabs.currentWidget() is self.preview_view:
            self._controller.request_preview()
        return revision_changed

    def _apply_editor_projection(self, projection: PulseEditorProjection) -> None:
        """Present a synchronous editor commit without polling application state."""

        runtime = self._last_runtime
        previous_editor = self._last_editor_projection
        if runtime is None or previous_editor is None:
            # Construction installs the three narrow initial fronts before any
            # Qt intent can arrive.
            raise RuntimeError("Pulse editor projection preceded initial composition")
        self._reconcile_editor_fields(
            projection,
            runtime,
            previous_editor,
            runtime,
        )
        file_fields = (
            projection.path,
            projection.file_state,
            projection.dirty,
            projection.document.name,
            projection.document_generation,
            projection.editor_revision,
        )
        old_file_fields = (
            previous_editor.path,
            previous_editor.file_state,
            previous_editor.dirty,
            previous_editor.document.name,
            previous_editor.document_generation,
            previous_editor.editor_revision,
        )
        if file_fields != old_file_fields:
            self._apply_file_and_run_state(projection, runtime)
        if projection.scan_workspace != previous_editor.scan_workspace:
            self._apply_scan_workspace(
                projection,
                previous_editor,
            )
            self._last_scan_workspace = projection.scan_workspace
        self._last_editor_projection = projection

    def _apply_file_update(self, update: PulseFileUpdate) -> None:
        """Reconcile Save/Open header facts without touching Edit widgets."""

        editor = self._last_editor_projection
        runtime = self._last_runtime
        if editor is None or runtime is None:
            raise RuntimeError("Pulse file update preceded initial composition")
        key = (
            update.path,
            update.file_state,
            update.dirty,
            update.document_name,
            update.document_generation,
            update.editor_revision,
        )
        old_key = (
            editor.path,
            editor.file_state,
            editor.dirty,
            editor.document.name,
            editor.document_generation,
            editor.editor_revision,
        )
        if key == old_key:
            return
        self._apply_file_and_run_state_values(
            name=update.document_name,
            path=update.path,
            file_state=update.file_state,
            dirty=update.dirty,
            document=editor.document,
            runtime=runtime,
        )
        self._last_editor_projection = dataclass_replace(
            editor,
            path=update.path,
            file_state=update.file_state,
            dirty=update.dirty,
        )

    def _apply_initial_state(
        self,
        editor: PulseEditorProjection,
        runtime: PulseRuntimeUpdate,
        preview: PulsePreviewUpdate,
    ) -> None:
        """Compose the initial editor, runtime, and preview fronts once."""

        self._last_editor_projection = editor
        self._last_runtime = runtime
        self._last_scan_workspace = editor.scan_workspace
        self._reconcile_editor_fields(editor, runtime, None, None)
        self._apply_file_and_run_state(editor, runtime)
        self._apply_connection_state(runtime, None)
        self._present_preview_update(preview)
        self._apply_scan_workspace(editor, None)
        self._last_scan_progress_key = self._scan_progress_key(runtime)
        self._apply_scan_progress(runtime)
        self._sync_runtime_watchers()

    def _apply_runtime_update(self, update: PulseRuntimeUpdate) -> None:
        """Present changed Run facts without touching editor/Scan projections."""

        editor = self._last_editor_projection
        previous = self._last_runtime
        if editor is None:
            raise RuntimeError("Pulse runtime update preceded initial composition")
        self._reconcile_editor_fields(editor, update, editor, previous)
        previous_diagnostic = "" if previous is None else previous.diagnostic
        if update.diagnostic and update.diagnostic != previous_diagnostic:
            lowered = update.diagnostic.lower()
            if "failed" in lowered or "error" in lowered:
                self._message(update.diagnostic)
        applied_key = (
            None
            if update.applied_snapshot is None
            else (
                update.applied_snapshot.run_id,
                update.applied_snapshot.authoring_document_digest,
            )
        )
        previous_applied_key = (
            None
            if previous is None or previous.applied_snapshot is None
            else (
                previous.applied_snapshot.run_id,
                previous.applied_snapshot.authoring_document_digest,
            )
        )
        # Every fact ``_apply_file_and_run_state`` presents belongs in this key.
        # ``run_admission_pending`` drives the On Pulse control on its own, and
        # it moves while ``run_busy`` is already true, so leaving it out made
        # that transition visible only when some other fact happened to change.
        file_run_key = (
            update.run_snapshot,
            applied_key,
            update.file_busy,
            update.run_busy,
            update.run_admission_pending,
            update.connection_state,
        )
        previous_file_run_key = None if previous is None else (
            previous.run_snapshot,
            previous_applied_key,
            previous.file_busy,
            previous.run_busy,
            previous.run_admission_pending,
            previous.connection_state,
        )
        if file_run_key != previous_file_run_key:
            self._apply_file_and_run_state(editor, update)
        self._apply_connection_state(update, previous)
        progress_key = self._scan_progress_key(update)
        if progress_key != self._last_scan_progress_key:
            self._apply_scan_progress(update)
            self._last_scan_progress_key = progress_key
        self._last_runtime = update
        self._sync_runtime_watchers()

    def _finish_close_if_ready(
        self,
        runtime: PulseRuntimeUpdate,
    ) -> None:
        if not runtime.close_complete or self._window is None:
            return
        self._teardown_pending = True
        self._advance_teardown()

    def _advance_teardown(self) -> bool:
        """Stop the render surface *before* the window accepts its close.

        Once ``closeEvent`` is accepted there is no event left to defer, so a
        host that has not stopped yet could only be waited for by joining its
        thread on the Qt thread.  Retiring first turns that join into a level
        this owner re-reads on its next turn.
        """

        self._retiring_hosts.poll()
        if not self._teardown_pending or self._permanently_closed:
            return self._permanently_closed
        self._release_preview_surface()
        if self._retiring_hosts:
            self._sync_runtime_watchers()
            return False
        self._teardown_pending = False
        if self._owner_retiring or self._window is None:
            self._commit_permanent_close()
            return self._permanently_closed
        QtCore.QTimer.singleShot(0, self._window.close)
        return True

    @staticmethod
    def _scan_progress_key(snapshot) -> tuple[object, ...]:
        return (
            snapshot.scan_progress,
            snapshot.held_scan_point,
            None
            if snapshot.applied_snapshot is None
            else (
                snapshot.applied_snapshot.run_id,
                snapshot.applied_snapshot.artifact_digest,
            ),
        )

    def _present_scan_progress(self, update: PulseScanProgressUpdate) -> None:
        key = self._scan_progress_key(update)
        if key == self._last_scan_progress_key:
            return
        self._apply_scan_progress(update)
        self._last_scan_progress_key = key

    def _present_preview_update(self, update: PulsePreviewUpdate) -> None:
        key = (
            update.preview_error,
            None
            if update.plot is None
            else (
                update.plot.document_generation,
                update.plot.editor_revision,
                update.plot.include_off_rows,
            ),
        )
        if key != self._last_preview_key:
            self._apply_preview(update)
            self._last_preview_key = key

    def _sync_runtime_watchers(self) -> None:
        """Arm the owner clock only while some fact still needs advancing."""

        if self._controller.runtime_poll_required or self._retiring_hosts:
            if not self._timer.isActive():
                self._timer.start()
        elif self._timer.isActive():
            self._timer.stop()
        self.scan_view.set_progress_polling(
            self.tabs.currentWidget() is self.scan_view
            and self._controller.scan_progress_poll_required
        )

    def _apply_scan_workspace(
        self,
        editor: PulseEditorProjection,
        previous_editor: PulseEditorProjection | None,
    ) -> None:
        self._apply_scan_workspace_value(
            editor.scan_workspace,
            None if previous_editor is None else previous_editor.scan_workspace,
            replace_source=(
                previous_editor is None
                or editor.document_generation != previous_editor.document_generation
            ),
        )

    def _apply_scan_progress(
        self,
        runtime: PulseRuntimeUpdate | PulseScanProgressUpdate,
    ) -> None:
        """Update only the active scan cursor text, never the Scan draft."""

        held_text = format_held_scan_point(runtime.held_scan_point)
        progress_values: tuple[tuple[str, int | float], ...] = ()
        progress = runtime.scan_progress
        applied = runtime.applied_snapshot
        if (
            not held_text
            and progress is not None
            and progress.available
            and applied is not None
            and applied.source_document.scan_table is not None
        ):
            table = applied.source_document.scan_table
            point = progress.current_point_index
            if point is not None and 0 <= point < len(table.rows):
                display_names = tuple(
                    spec.name
                    for spec in scan_column_specs(applied.source_document)
                )
                progress_values = tuple(
                    zip(display_names, table.rows[point], strict=True)
                )
        self.scan_view.set_progress_text(
            held_text
            or format_scan_progress(progress, values=progress_values)
        )

    def _apply_scan_workspace_value(
        self,
        workspace,
        previous_workspace,
        *,
        replace_source: bool = False,
    ) -> None:
        loaded_path = "" if workspace.loaded_path is None else str(workspace.loaded_path)
        self.schedule_view.set_scan_source(
            use_loaded=workspace.selected_source == "loaded",
            path=loaded_path,
        )
        busy = workspace.busy_operation is not None
        self.schedule_view.set_scan_workspace_busy(busy)
        self.scan_view.set_workspace_busy(busy)

        if self.scan_view.scan_slots_label.text() != workspace.slots_text:
            self.scan_view.set_slots_text(workspace.slots_text)
        if self.scan_view.scan_table_view.toPlainText() != workspace.table_text:
            self.scan_view.set_scan_table_text(workspace.table_text)
        loaded_program_completed = (
            previous_workspace is not None
            and previous_workspace.busy_operation == "load-program"
            and workspace.busy_operation is None
            and workspace.source_revision != previous_workspace.source_revision
        )
        current_source = self.scan_view.scan_code.toPlainText()
        if replace_source or loaded_program_completed:
            self.scan_view.set_scan_code(
                workspace.source_text,
                dirty=workspace.source_dirty,
                source_revision=workspace.source_revision,
            )
        elif self.scan_view.code_dirty:
            # Reading the full code buffer is reserved for a real workspace
            # transition (Run/load/template result), never a key event/timer.
            if current_source == workspace.source_text:
                self.scan_view.acknowledge_scan_draft(
                    dirty=workspace.source_dirty,
                    source_revision=workspace.source_revision,
                )
        elif (
            self.scan_view.source_revision != workspace.source_revision
            or current_source != workspace.source_text
        ):
            self.scan_view.set_scan_code(
                workspace.source_text,
                dirty=workspace.source_dirty,
                source_revision=workspace.source_revision,
            )
        else:
            self.scan_view.set_run_dirty(workspace.source_dirty)

        previous_diagnostic = (
            ""
            if previous_workspace is None
            else previous_workspace.diagnostic
        )
        if workspace.diagnostic and workspace.diagnostic != previous_diagnostic:
            lowered = workspace.diagnostic.lower()
            if "error" in lowered or "failed" in lowered:
                self._message(workspace.diagnostic)
            elif self._preview_plot is None:
                self.preview_view.set_status(workspace.diagnostic)

    def _apply_file_and_run_state(
        self,
        editor: PulseEditorProjection,
        runtime: PulseRuntimeUpdate,
    ) -> None:
        self._apply_file_and_run_state_values(
            name=editor.document.name,
            path=editor.path,
            file_state=editor.file_state,
            dirty=editor.dirty,
            document=editor.document,
            runtime=runtime,
        )

    def _apply_file_and_run_state_values(
        self,
        *,
        name: str,
        path: Path | None,
        file_state: str,
        dirty: bool,
        document: PulseDocument,
        runtime: PulseRuntimeUpdate,
    ) -> None:
        local = path.name if path is not None else ""
        name = str(name).strip() or "pulse"
        if dirty:
            status = "unsaved" if local else "new"
            star = "*"
        else:
            status = file_state if local else "new"
            star = "" if local else "*"
        if local:
            self.label_name.setText(
                f"PulseGUI - {name} ({status}: {local}){star}"
            )
        else:
            self.label_name.setText(f"PulseGUI - {name} ({status}){star}")
        title = f"{name} - PulseGUI{star}"
        self.setWindowTitle(title)
        if self._window is not None:
            self._window.setWindowTitle(title)

        run = runtime.run_snapshot
        synchronized = runtime.is_document_applied(document)
        if runtime.run_busy and run is None:
            color = YELLOW
        elif run is not None and run.state is RunState.RUNNING:
            color = GREEN if synchronized else ORANGE
        elif run is not None and run.state is RunState.CANCELLING:
            color = RED
        elif run is not None and run.state is RunState.FAILED:
            color = RED
        elif runtime.connection_state == "ready" and dirty:
            color = ORANGE
        else:
            color = GREY if run is None else RED
        self.status_dot.set_color(color)
        self.schedule_view.set_control_state(
            running=bool(run is not None and run.state is RunState.RUNNING),
            synchronized=synchronized,
            file_dirty=dirty,
            file_busy=runtime.file_busy,
            run_admission_pending=runtime.run_admission_pending,
        )

    def _apply_connection_state(
        self,
        snapshot: PulseRuntimeUpdate,
        previous: PulseRuntimeUpdate | None,
    ) -> None:
        connection_key = (
            snapshot.connection_state,
            snapshot.connection_mode,
            snapshot.connection_endpoint,
        )
        previous_key = (
            None
            if previous is None
            else (
                previous.connection_state,
                previous.connection_mode,
                previous.connection_endpoint,
            )
        )
        if connection_key == previous_key:
            # The target/address controls are an operator edit buffer.  A periodic
            # presentation refresh must not overwrite an unsubmitted choice with
            # the controller's last committed connection.  Reconcile them only
            # when connection authority itself advances (initial, connecting,
            # ready, failure, or close).
            return
        mode = snapshot.connection_mode
        if mode not in ("virtual", "remote", "offline"):
            # Existing-installation composition supplies the actual mode before
            # this product is launched.  This fallback is deliberately only a
            # truthful display guard while that immutable fact is unavailable.
            mode = "offline" if snapshot.target_descriptor is None else "virtual"
        if snapshot.connection_state == "connecting":
            status = "Connecting…"
        elif snapshot.connection_state == "switching":
            status = "Switching safely…"
        elif snapshot.connection_state == "ready":
            status = (
                snapshot.connection_endpoint
                if mode == "remote"
                else "Virtual (sim)"
            )
        else:
            status = "Offline (edit only)"
        self.schedule_view.set_connection_state(
            mode,
            endpoint=snapshot.connection_endpoint,
            status=status,
        )
        if previous is None:
            return
        if (
            snapshot.connection_state == "ready"
            and previous.connection_state != "ready"
        ):
            if mode == "remote":
                self._message(
                    "Connected to sequencer server at "
                    f"{snapshot.connection_endpoint}."
                )
            else:
                self._message("Connected to a virtual (in-memory) sequencer.")
        elif (
            snapshot.connection_state == "offline"
            and mode == "offline"
            and previous.connection_mode != "offline"
            and not snapshot.diagnostic
        ):
            self._message("Offline: editing only, no backend calls.")

    def _apply_preview(
        self,
        snapshot: PulsePreviewUpdate,
    ) -> None:
        projected = snapshot.plot
        if projected is None:
            if snapshot.preview_error:
                diagnostic = f"Preview unavailable:\n{snapshot.preview_error}"
                self.preview_view.set_status(diagnostic)
                self.preview_view.preview_status.setToolTip(diagnostic)
            return
        key = (
            projected.document_generation,
            projected.editor_revision,
            projected.include_off_rows,
        )
        if key == self._shown_preview_key:
            return
        self._preview_plot = projected
        self.preview_view.set_status(projected.status)
        self.preview_view.preview_status.setToolTip(projected.status)
        if not self.preview_view.preview_size_pinned:
            self.preview_view.set_preview_size(
                projected.recommended_size,
                pinned=False,
            )

        host = self.preview_host
        if host is None:
            size = (
                self.preview_view.preview_size
                if self.preview_view.preview_size_pinned
                else projected.recommended_size
            )
            host = RasterPlotHost.from_plot(
                projected.data,
                projected.spec,
                size=size,
            )
            self.preview_host = host
            self._preview_spec = projected.spec
            self._preview_size = size
            self._preview_host_unsubscribe = host.subscribe_front(
                lambda _front: self._wake.request_owner_wake()
            )
            self.preview_view.show_placeholder("Rendering preview…")
            QtCore.QTimer.singleShot(0, self._mount_preview_if_ready)
        else:
            if projected.spec != self._preview_spec:
                self._track_preview_operation(host.replace_spec(projected.spec))
                self._preview_spec = projected.spec
            if not self.preview_view.preview_size_pinned:
                self._set_preview_size(projected.recommended_size)
            self._track_preview_operation(host.update_data(projected.data))
        self._shown_preview_key = key

    # ------------------------------------------------------------------
    # lifecycle / launcher contract
    # ------------------------------------------------------------------

    def _release_preview_surface(self) -> None:
        """Hand the preview host to the one owner of plot-host shutdown."""

        unsubscribe, self._preview_host_unsubscribe = (
            self._preview_host_unsubscribe,
            None,
        )
        if unsubscribe is not None:
            unsubscribe()
        # The host joins the retiring set BEFORE the widget cleanup edge, which
        # re-raises its first failure by contract.  Retiring afterwards would let
        # that raise strand a live host that nobody ever asked to stop, while an
        # empty set reports every surface already retired.
        host, self.preview_host = self.preview_host, None
        if host is not None:
            self._retiring_hosts.retire(host)
        widget, self.preview_widget = self.preview_widget, None
        if widget is not None:
            widget.close_adapter()
        self._preview_operations.clear()

    @property
    def worker_idle(self) -> bool:
        return (
            self._controller.worker_idle
            and not self._preview_operations
            and not self._retiring_hosts
        )

    @property
    def current_document(self):
        return self._controller.current_document

    @property
    def active_snapshot(self):
        return None if self._last_runtime is None else self._last_runtime.run_snapshot

    def request_close(self, *, discard_unsaved: bool = False) -> bool:
        if self._permanently_closed:
            return True
        if not self._close_decided:
            dirty = self._controller.dirty
            if dirty and not discard_unsaved:
                if not fluent_confirm(
                    self,
                    "Pulse",
                    "Close this pulse without saving the current edits?",
                    confirm_text="Close",
                    cancel_text="Cancel",
                ):
                    return False
            self._close_decided = True
        if not self._close_requested:
            self._close_requested = True
            self._controller.request_close()
        update = self._controller.runtime_update()
        self._apply_runtime_update(update)
        if not update.close_complete:
            return False
        # ``request_close`` is also the wrapper's close guard: refusing while
        # the render surface is still stopping is what keeps the window alive
        # long enough to retire it without joining a thread here.
        self._teardown_pending = True
        return self._advance_teardown()

    @property
    def permanently_closed(self) -> bool:
        return self._permanently_closed

    def restore_window(self) -> None:
        """Restore the same application-owned editor after an X-to-hide action."""

        if self._owner_retiring or self._permanently_closed:
            raise RuntimeError("Pulse editor is closing with its application")
        window = self._window
        if window is None:
            raise RuntimeError("Pulse editor window is unavailable")
        window.showNormal()
        window.raise_()
        window.activateWindow()

    def wait_owner_closed(self, timeout: float) -> bool:
        return wait_for_owner_retirement(
            self,
            self._owner_closed,
            timeout=timeout,
        )

    def request_owner_close(self) -> None:
        """Thread-safe invalidation when the borrowing application closes."""

        if self._owner_retiring or self._permanently_closed:
            return
        self._owner_retiring = True
        self._controller.retire_borrowed_authority()

    def _commit_permanent_close(self) -> None:
        """Release the wrapper after either standalone or borrowed close commits.

        Committing destroys the wrapper and with it the clock, so this entrance
        may not leave a host that has not stopped with nobody left to poll it.
        A still-retiring surface keeps the commit pending instead: the level
        that says so re-arms the clock, and ``_advance_teardown`` re-enters
        here from the next owner turn.
        """

        if self._permanently_closed:
            return
        self._release_preview_surface()
        if not self._retiring_hosts.poll():
            self._teardown_pending = True
            self._sync_runtime_watchers()
            return
        self._permanently_closed = True
        window = self._window
        self._window = None
        if window is None:
            self._owner_closed.set()
            return
        release_window(window)
        window.hide()
        window.deleteLater()
        self._owner_closed.set()

    def _attach_window(self, window) -> None:
        self._window = window


def launch_pulse_editor_window(
    controller: PulseEditorController,
    *,
    pulses_root: Path,
    output_root: Path,
    hide_on_close: bool = False,
):
    """Launch the one formal body through the shared Fluent window sequence."""

    ensure_qt_app()
    body = PulseEditorWindowBody(
        controller,
        pulses_root=pulses_root,
        output_root=output_root,
    )

    def wire(window) -> None:
        body._attach_window(window)
        if not hide_on_close:
            window.set_close_guard(body.request_close)
            window.closed.connect(body._commit_permanent_close)

    launch_fluent_window(
        body,
        title="PulseGUI@Zou lab",
        hide_on_close=hide_on_close,
        wire=wire,
    )
    return body


__all__ = ["PulseEditorWindowBody", "launch_pulse_editor_window"]
