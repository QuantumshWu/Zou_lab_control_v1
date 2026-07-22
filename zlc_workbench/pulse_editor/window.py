"""Formal Pulse GUI composition over current document/application owners.

The operator-facing surface is frozen: this module composes the mechanically
extracted Edit, Preview, and Scan pages without changing their controls,
wording, geometry, or workflow.  It owns only Qt coordination.  Pulse meaning,
file state, preview rendering, connection authority, and Run lifecycle remain
behind :class:`PulseEditorController`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    ACCENT,
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
    SinglePanelHost,
    ensure_qt_app,
    fluent_confirm,
    fluent_message,
    fluent_widget_stylesheet,
    launch_fluent_window,
    release_window,
    screen_fit_window_size,
    set_fluent_scale,
    window_pad,
)
from zlc_neutral_atom.runtime.run import RunState
from zlc_pulse import PulseExecutionForm
from zlc_storage.paths import project_path

from ._layout import px
from .controller import PulseEditorController, PulseEditorControllerSnapshot
from .preview_view import PulsePreviewView
from .scan_view import (
    PulseScanView,
    format_held_scan_point,
    format_scan_progress,
)
from .schedule_view import PulseScheduleView


_PULSE_FILES_ENV = "ZLC_PULSE_DIR"


def _pulse_files_dir() -> Path:
    configured = os.environ.get(_PULSE_FILES_ENV, "").strip()
    directory = (
        Path(configured).expanduser()
        if configured
        else project_path("pulses")
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


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
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not isinstance(controller, PulseEditorController):
            raise TypeError("controller must be PulseEditorController")
        ensure_qt_app()
        super().__init__(parent)
        self._controller = controller
        self._window = None
        self._last_snapshot: PulseEditorControllerSnapshot | None = None
        self._shown_preview_key: tuple[int, int, int] | None = None
        self._pending_preview_origin = None
        self._pending_preview_revision: int | None = None
        self._preview_window_handle = None
        self._observed_preview_screens: set[int] = set()
        self._close_decided = False
        self._close_requested = False
        self._owner_retiring = False
        self._permanently_closed = False

        set_fluent_scale(None)
        self.setWindowTitle("PulseGUI@Zou lab")
        self.setFixedSize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
        self.setStyleSheet(fluent_widget_stylesheet())
        self._build_ui(controller.snapshot())
        self._wire_ui()

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._controller.set_notify(self._wake.request_owner_wake)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._owner_cycle)
        self._timer.start()
        self._owner_cycle()

    # ------------------------------------------------------------------
    # frozen composition
    # ------------------------------------------------------------------

    def _build_ui(self, snapshot: PulseEditorControllerSnapshot) -> None:
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
            snapshot.document,
            document_generation=snapshot.document_generation,
            revision=snapshot.editor_revision,
        )
        self.preview_view = PulsePreviewView()
        self.scan_view = PulseScanView()
        self.tabs.addTab(self.schedule_view, "Edit")
        self.tabs.addTab(self.preview_view, "Preview")
        self.tabs.addTab(self.scan_view, "Scan")
        root.addWidget(self.tabs, 1)

        self.preview_host = SinglePanelHost(
            "pulse",
            group="pulse-preview",
            empty_text="Open Preview to render the pulse plot.",
        )
        self.preview_view.mount_content(
            self.preview_host,
            wheel_target=self.preview_host.board,
        )

    def _wire_ui(self) -> None:
        view = self.schedule_view
        view.documentNameEdited.connect(
            lambda value: self._invoke(self._controller.rename_document, value)
        )
        view.portLabelEdited.connect(
            lambda port, value: self._invoke(
                self._controller.rename_port, port, value
            )
        )
        view.periodNameEdited.connect(
            lambda period, value: self._invoke(
                self._controller.rename_period, period, value
            )
        )
        view.durationEdited.connect(
            lambda period, value, unit: self._invoke(
                self._controller.set_period_duration,
                period,
                value,
                unit,
            )
        )
        view.digitalEdited.connect(
            lambda period, port, high: self._invoke(
                self._controller.set_digital, period, port, high
            )
        )
        view.analogEdited.connect(
            lambda period, port, mode, value: self._invoke(
                self._controller.set_analog,
                period,
                port,
                mode,
                value,
                cascade=True,
            )
        )
        view.delayEdited.connect(
            lambda port, value, unit: self._invoke(
                self._controller.set_delay,
                port,
                value,
                unit,
                cascade=True,
            )
        )
        view.bindingCycleRequested.connect(
            lambda field: self._invoke(self._controller.cycle_binding, field)
        )
        view.insertPeriodRequested.connect(
            lambda before: self._invoke(
                self._controller.add_period,
                before_period_id=before,
                cascade=True,
            )
        )
        view.movePeriodRequested.connect(
            lambda period, before: self._invoke(
                self._controller.move_period,
                period,
                before,
                cascade=True,
            )
        )
        view.removePeriodRequested.connect(
            lambda period: self._invoke(
                self._controller.remove_period,
                period,
                cascade=True,
            )
        )
        view.repeatEdited.connect(
            lambda start, end, count: self._invoke(
                self._controller.set_repeat, start, end, count
            )
        )
        view.visiblePortsEdited.connect(
            lambda ports: self._invoke(self._controller.set_visible_ports, ports)
        )
        view.clearPortRequested.connect(
            lambda port: self._invoke(self._controller.clear_port, port)
        )
        view.clearAllRequested.connect(
            lambda: self._invoke(self._controller.clear_all)
        )
        self.clear_all_button.clicked.connect(
            lambda: self._invoke(self._controller.clear_all)
        )
        view.runRequested.connect(self._run_from_edit)
        view.stopRequested.connect(
            lambda: self._invoke(self._controller.cancel)
        )
        view.syncRequested.connect(self._sync_applied)
        view.feedbackRequested.connect(self._message)
        view.saveRequested.connect(self._save_document)
        view.loadRequested.connect(self._load_document)
        view.connectionRequested.connect(
            lambda mode, endpoint: self._invoke(
                self._controller.connect, mode, endpoint
            )
        )
        view.scanArrayLoadRequested.connect(self._load_scan_array)
        view.scanSourceEdited.connect(
            lambda use_loaded: self._invoke(
                self._controller.select_scan_source,
                "loaded" if use_loaded else "generated",
            )
        )

        self.preview_view.includeOffToggled.connect(
            lambda checked: self._invoke(
                self._controller.set_preview_include_off, checked
            )
        )
        self.preview_view.selectorsToggled.connect(
            self.preview_host.set_selectors_enabled
        )
        self.preview_view.sizeActivated.connect(
            lambda size: self._invoke(self._controller.set_preview_size, size)
        )
        self.preview_view.saveFigureRequested.connect(self._save_preview)
        self.preview_host.viewCommitted.connect(self._preview_view_committed)

        self.scan_view.repeatsChanged.connect(
            lambda repeats: self._invoke(
                self._controller.set_scan_sweep_count,
                int(repeats),
            )
        )
        self.scan_view.holdRequested.connect(self._hold_scan_point)
        self.scan_view.stepRequested.connect(self._step_scan_point)
        self.scan_view.loadProgramRequested.connect(self._load_scan_program)
        self.scan_view.templateRequested.connect(
            lambda kind: self._invoke(self._controller.set_scan_template, kind)
        )
        self.scan_view.codeChanged.connect(
            lambda: self._invoke(
                self._controller.update_scan_source,
                self.scan_view.scan_code.toPlainText(),
            )
        )
        self.scan_view.runRequested.connect(
            lambda source: self._invoke(
                self._controller.generate_scan_source, source
            )
        )
        self.scan_view.saveArrayRequested.connect(self._save_scan_array)
        self.scan_view.progressRefreshRequested.connect(
            lambda: self._invoke(self._controller.request_scan_progress)
        )
        self.tabs.currentChanged.connect(self._tab_changed)

    # ------------------------------------------------------------------
    # Qt intents
    # ------------------------------------------------------------------

    def _invoke(self, action: Callable, *args, **kwargs):
        try:
            result = action(*args, **kwargs)
        except BaseException as error:
            self._message(str(error))
            return None
        # Never rebuild the immutable editor tree re-entrantly from the signal
        # of a widget that belongs to that tree.  Queue one owner turn after the
        # complete mouse/key intent returns to Qt; the controller remains the
        # only document authority and the existing 40 ms timer is a backstop.
        wake = getattr(self, "_wake", None)
        if wake is not None:
            wake.request_owner_wake()
        else:
            QtCore.QTimer.singleShot(0, self._owner_cycle)
        return result

    def _message(self, text: str) -> None:
        message = str(text)
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            self.summary.setText(message)
            self.preview_view.set_status(message)
            return
        fluent_message(self, "Pulse", message, kind="warning")

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.preview_view:
            self._sync_preview_pixel_ratio()
            self.preview_view.reset_preview_size_pin()
            self._invoke(self._controller.reset_preview_size)
            self._invoke(self._controller.request_preview)

    def _current_preview_pixel_ratio(self) -> float:
        """Read the physical/logical ratio of the screen painting this window."""

        window = self._window
        ratio = (
            float(window.devicePixelRatioF())
            if window is not None
            else float(self.devicePixelRatioF())
        )
        if not math.isfinite(ratio) or ratio <= 0.0:
            return 1.0
        return ratio

    def _sync_preview_pixel_ratio(self, *_args) -> None:
        """Keep the worker raster at one source pixel per physical screen pixel."""

        if self._owner_retiring or self._permanently_closed:
            return
        self._invoke(
            self._controller.set_preview_pixel_ratio,
            self._current_preview_pixel_ratio(),
        )

    def _observe_preview_screen(self, screen) -> None:
        if screen is None or id(screen) in self._observed_preview_screens:
            return
        self._observed_preview_screens.add(id(screen))
        for name in (
            "logicalDotsPerInchChanged",
            "physicalDotsPerInchChanged",
        ):
            signal = getattr(screen, name, None)
            if signal is not None:
                signal.connect(self._sync_preview_pixel_ratio)

    def _preview_screen_changed(self, screen) -> None:
        self._observe_preview_screen(screen)
        # Qt finalises per-monitor DPR after the screen-change notification.
        QtCore.QTimer.singleShot(0, self._sync_preview_pixel_ratio)

    def _bind_preview_screen(self) -> None:
        window = self._window
        if window is None or self._permanently_closed:
            return
        handle = window.windowHandle()
        if handle is None:
            QtCore.QTimer.singleShot(0, self._bind_preview_screen)
            return
        if handle is not self._preview_window_handle:
            self._preview_window_handle = handle
            handle.screenChanged.connect(self._preview_screen_changed)
        self._observe_preview_screen(handle.screen())
        self._sync_preview_pixel_ratio()

    def _run_from_edit(self) -> None:
        snapshot = self._controller.snapshot()
        if snapshot.document.scan_parameters:
            form = (
                PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS
                if snapshot.document.scan_sweep_count == 0
                else PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
            )
        else:
            form = PulseExecutionForm.CONTINUOUS_MONITOR
        self._invoke(
            self._controller.start,
            form,
            scan_sweep_count=max(1, snapshot.document.scan_sweep_count),
        )

    def _sync_applied(self) -> None:
        action = getattr(self._controller, "sync_applied", None)
        if action is None:
            self._message("The sequencer has no applied pulse yet (nothing was prepared).")
            return
        self._invoke(action)

    def _hold_scan_point(self) -> None:
        action = getattr(self._controller, "hold_scan_point", None)
        if action is None:
            self._message("No running scan point is available to hold.")
            return
        self._invoke(action)

    def _step_scan_point(self, delta: int) -> None:
        action = getattr(self._controller, "step_scan_point", None)
        if action is None:
            self._message("No held scan point is available to step.")
            return
        self._invoke(action, int(delta))

    def _scan_file_start(self) -> Path:
        snapshot = self._controller.snapshot()
        return (
            snapshot.path.parent
            if snapshot.path is not None
            else _pulse_files_dir()
        )

    def _load_scan_array(self) -> None:
        if not self._controller.snapshot().document.scan_parameters:
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
            self._invoke(self._controller.load_scan_array, Path(path))

    def _load_scan_program(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load scan program / table",
            str(self._scan_file_start()),
            "Scan program or saved table (*.py *.txt *.npy *.csv *.json);;"
            "Python program (*.py *.txt);;"
            "Scan array (*.npy *.csv);;"
            "Saved pulse / program (*.json)",
        )
        if path:
            self._invoke(self._controller.load_scan_program, Path(path))

    def _save_scan_array(self) -> None:
        snapshot = self._controller.snapshot()
        if snapshot.scan_workspace.selected_table is None:
            self._message(
                "No scan table to save yet. Run code or load a file first."
            )
            return
        suggested = self._scan_file_start() / (
            f"{_safe_file_stem(snapshot.document.name)}_scan.npy"
        )
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save scan array",
            str(suggested),
            "Scan array (*.npy *.csv)",
        )
        if path:
            self._invoke(self._controller.save_scan_array, Path(path))

    def _save_document(self) -> None:
        snapshot = self._controller.snapshot()
        suggested = snapshot.path or (
            _pulse_files_dir()
            / f"{_safe_file_stem(snapshot.document.name)}.json"
        )
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save pulse",
            str(suggested),
            "ZLC pulse (*.json)",
        )
        if path:
            self._invoke(self._controller.save, Path(path), overwrite=True)

    def _load_document(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load pulse",
            str(_pulse_files_dir()),
            "ZLC pulse (*.json)",
        )
        if path:
            self._invoke(self._controller.open_path, Path(path))

    def _save_preview(self) -> None:
        snapshot = self._controller.snapshot()
        suggested = _pulse_files_dir() / (
            f"{_safe_file_stem(snapshot.document.name)}.png"
        )
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save pulse figure",
            str(suggested),
            "Pulse figure (*.png)",
        )
        if not path:
            return
        action = getattr(self._controller, "save_preview", None)
        if action is None:
            self._message("Pulse preview export is unavailable.")
            return
        self._invoke(action, Path(path))

    def _preview_view_committed(self, commit) -> None:
        visible = self.preview_host.visible_interaction_origin()
        origin = getattr(commit, "origin", None)
        if visible is None or origin != visible:
            if origin is not None:
                self.preview_host.discard_pending_interaction(origin)
            return
        viewport = getattr(commit, "viewport", None)
        if viewport is None:
            self.preview_host.discard_pending_interaction(origin)
            return
        x_limits = (
            None
            if viewport.x_limits == viewport.home_x_limits
            else viewport.x_limits
        )
        self._pending_preview_origin = origin
        self._pending_preview_revision = int(viewport.display_revision)
        try:
            self._controller.commit_preview_view(
                x_limits,
                presentation_revision=viewport.display_revision,
            )
        except BaseException as error:
            self.preview_host.discard_pending_interaction(origin)
            self._pending_preview_origin = None
            self._pending_preview_revision = None
            self._message(str(error))

    # ------------------------------------------------------------------
    # immutable snapshot presentation
    # ------------------------------------------------------------------

    def _owner_cycle(self) -> None:
        snapshot = self._controller.pump()
        self._apply_snapshot(snapshot)
        if snapshot.close_complete and self._window is not None:
            if self._owner_retiring:
                self._commit_owner_retirement()
            else:
                QtCore.QTimer.singleShot(0, self._window.close)

    def _apply_snapshot(self, snapshot: PulseEditorControllerSnapshot) -> None:
        previous = self._last_snapshot
        document_key = (
            snapshot.document_generation,
            snapshot.editor_revision,
        )
        old_document_key = (
            None
            if previous is None
            else (previous.document_generation, previous.editor_revision)
        )
        if document_key != old_document_key:
            self.schedule_view.set_document(
                snapshot.document,
                document_generation=snapshot.document_generation,
                revision=snapshot.editor_revision,
            )
            self.scan_view.set_repeats(snapshot.document.scan_sweep_count)
        self.summary.setText(self.schedule_view.summary_text())
        previous_diagnostic = "" if previous is None else previous.diagnostic
        if snapshot.diagnostic and snapshot.diagnostic != previous_diagnostic:
            lowered = snapshot.diagnostic.lower()
            if "failed" in lowered or "error" in lowered:
                self._message(snapshot.diagnostic)
        self._apply_file_and_run_state(snapshot)
        self._apply_connection_state(snapshot, previous)
        self._apply_preview(snapshot)
        self._apply_scan_workspace(snapshot, previous)
        self._last_snapshot = snapshot

    def _apply_scan_workspace(
        self,
        snapshot: PulseEditorControllerSnapshot,
        previous: PulseEditorControllerSnapshot | None,
    ) -> None:
        workspace = snapshot.scan_workspace
        loaded_path = "" if workspace.loaded_path is None else str(workspace.loaded_path)
        self.schedule_view.set_scan_source(
            use_loaded=workspace.selected_source == "loaded",
            path=loaded_path,
        )
        busy = workspace.busy_operation is not None
        self.schedule_view.set_scan_workspace_busy(busy)
        self.scan_view.set_workspace_busy(busy)

        held_text = format_held_scan_point(snapshot.held_scan_point)
        progress_values: tuple[tuple[str, int | float], ...] = ()
        progress = snapshot.scan_progress
        applied = snapshot.applied_snapshot
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
                progress_values = tuple(
                    zip(table.columns, table.rows[point], strict=True)
                )
        self.scan_view.set_progress_text(
            held_text
            or format_scan_progress(progress, values=progress_values)
        )

        if self.scan_view.scan_slots_label.text() != workspace.slots_text:
            self.scan_view.set_slots_text(workspace.slots_text)
        if self.scan_view.scan_table_view.toPlainText() != workspace.table_text:
            self.scan_view.set_scan_table_text(workspace.table_text)
        if self.scan_view.scan_code.toPlainText() != workspace.source_text:
            self.scan_view.set_scan_code(
                workspace.source_text,
                dirty=workspace.source_dirty,
            )
        else:
            self.scan_view.set_run_dirty(workspace.source_dirty)

        previous_diagnostic = (
            "" if previous is None else previous.scan_workspace.diagnostic
        )
        if workspace.diagnostic and workspace.diagnostic != previous_diagnostic:
            lowered = workspace.diagnostic.lower()
            if "error" in lowered or "failed" in lowered:
                self._message(workspace.diagnostic)
            elif not snapshot.preview_notice:
                self.preview_view.set_status(workspace.diagnostic)

    def _apply_file_and_run_state(
        self,
        snapshot: PulseEditorControllerSnapshot,
    ) -> None:
        local = snapshot.path.name if snapshot.path is not None else ""
        name = snapshot.document.name.strip() or "pulse"
        if snapshot.dirty:
            status = "unsaved" if local else "new"
            star = "*"
        else:
            status = snapshot.file_state if local else "new"
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

        run = snapshot.run_snapshot
        synchronized = (
            run is not None
            and run.state is RunState.RUNNING
            and snapshot.run_generation == snapshot.document_generation
            and snapshot.run_revision == snapshot.editor_revision
        )
        if snapshot.run_busy and run is None:
            color = YELLOW
        elif run is not None and run.state is RunState.RUNNING:
            color = GREEN if synchronized else ORANGE
        elif run is not None and run.state is RunState.CANCELLING:
            color = RED
        elif run is not None and run.state is RunState.FAILED:
            color = RED
        elif snapshot.connection_state == "ready" and snapshot.dirty:
            color = ORANGE
        else:
            color = GREY if run is None else RED
        self.status_dot.set_color(color)
        self.schedule_view.set_control_state(
            running=bool(run is not None and run.state is RunState.RUNNING),
            synchronized=synchronized,
            file_dirty=snapshot.dirty,
            file_busy=snapshot.file_busy,
            run_busy=snapshot.run_busy,
        )

    def _apply_connection_state(
        self,
        snapshot: PulseEditorControllerSnapshot,
        previous: PulseEditorControllerSnapshot | None,
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
        elif snapshot.connection_state == "ready":
            status = (
                snapshot.connection_endpoint
                or ("Remote server" if mode == "remote" else "Virtual (sim)")
            )
        elif snapshot.connection_state == "restart_required":
            status = "Restart required"
        elif snapshot.connection_state == "close_blocked":
            status = "SAFE close blocked"
        else:
            status = "Offline (edit only)"
        self.schedule_view.set_connection_state(
            mode,
            endpoint=snapshot.connection_endpoint,
            status=status,
        )

    def _settle_pending_preview_through(
        self,
        presentation_revision: int,
        *,
        failed: bool,
    ) -> None:
        """Resolve only the pending view intent reached by this worker answer.

        A completed intermediate may be older than the newest pointer motion.
        Its Qt presentation failure therefore cannot discard the newer intent
        merely because both originated from the same held gesture.
        """

        pending_revision = self._pending_preview_revision
        if (
            pending_revision is None
            or int(presentation_revision) < pending_revision
        ):
            return
        if failed and self._pending_preview_origin is not None:
            self.preview_host.discard_pending_interaction(
                self._pending_preview_origin
            )
        self._pending_preview_origin = None
        self._pending_preview_revision = None

    def _apply_preview(self, snapshot: PulseEditorControllerSnapshot) -> None:
        notice = snapshot.preview_notice
        rendered = snapshot.rendered_preview
        if rendered is None:
            if notice:
                self.preview_view.set_status(notice)
            elif snapshot.preview_error:
                if self._pending_preview_origin is not None:
                    self.preview_host.discard_pending_interaction(
                        self._pending_preview_origin
                    )
                    self._pending_preview_origin = None
                    self._pending_preview_revision = None
                self.preview_view.set_status(
                    f"Preview unavailable: {snapshot.preview_error.splitlines()[0][:120]}"
                )
            return
        key = (
            rendered.document_generation,
            rendered.editor_revision,
            rendered.presentation_revision,
        )
        if key == self._shown_preview_key:
            if notice:
                self.preview_view.set_status(notice)
            return
        try:
            logical_size = self.preview_host.present_panel(
                rendered.raster,
                rendered.payload,
                pixel_ratio=rendered.pixel_ratio,
            )
        except BaseException as error:
            self._settle_pending_preview_through(
                rendered.presentation_revision,
                failed=True,
            )
            self.preview_view.set_status(
                notice
                or f"Preview unavailable: {str(error).splitlines()[0][:120]}"
            )
            return
        self._settle_pending_preview_through(
            rendered.presentation_revision,
            failed=False,
        )
        self._shown_preview_key = key
        self.preview_view.mount_content(
            self.preview_host,
            logical_size=logical_size,
            wheel_target=self.preview_host.board,
        )
        self.preview_view.set_status(notice or rendered.status)
        if not self.preview_view.preview_size_pinned:
            self.preview_view.set_preview_size(rendered.size, pinned=False)

    # ------------------------------------------------------------------
    # lifecycle / launcher contract
    # ------------------------------------------------------------------

    @property
    def worker_idle(self) -> bool:
        return self._controller.worker_idle

    @property
    def current_document(self):
        return self._controller.snapshot().document

    @property
    def active_snapshot(self):
        return self._controller.snapshot().run_snapshot

    def request_close(self, *, discard_unsaved: bool = False) -> bool:
        if not self._close_decided:
            dirty = self._controller.snapshot().dirty
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
        snapshot = self._controller.pump()
        self._apply_snapshot(snapshot)
        return snapshot.close_complete

    @property
    def permanently_closed(self) -> bool:
        return self._permanently_closed

    def restore_window(self) -> None:
        """Restore the same notebook-owned editor after an X-to-hide action."""

        if self._owner_retiring or self._permanently_closed:
            raise RuntimeError("Pulse editor is closing with its Experiment")
        window = self._window
        if window is None:
            raise RuntimeError("Pulse editor window is unavailable")
        window.showNormal()
        window.raise_()
        window.activateWindow()

    def request_owner_close(self) -> None:
        """Thread-safe invalidation when the borrowing Experiment closes."""

        if self._owner_retiring or self._permanently_closed:
            return
        self._owner_retiring = True
        self._controller.retire_borrowed_authority()

    def _commit_owner_retirement(self) -> None:
        if self._permanently_closed:
            return
        self._permanently_closed = True
        window = self._window
        self._window = None
        if window is None:
            return
        release_window(window)
        window.hide()
        window.deleteLater()

    def _attach_window(self, window) -> None:
        self._window = window
        # ``wire`` runs before show, so the native QWindow/screen may not exist
        # until the next owner turn.  Bind then and re-render Preview at the
        # actual per-monitor DPR; the initial controller frame is only a
        # provisional logical-pixel result.
        QtCore.QTimer.singleShot(0, self._bind_preview_screen)


def launch_pulse_editor_window(
    controller: PulseEditorController,
    *,
    hide_on_close: bool = False,
):
    """Launch the one formal body through the shared Fluent window sequence."""

    ensure_qt_app()
    body = PulseEditorWindowBody(controller)

    def wire(window) -> None:
        body._attach_window(window)
        if not hide_on_close:
            window.set_close_guard(body.request_close)

    launch_fluent_window(
        body,
        title="PulseGUI@Zou lab",
        hide_on_close=hide_on_close,
        wire=wire,
    )
    return body


__all__ = ["PulseEditorWindowBody", "launch_pulse_editor_window"]
