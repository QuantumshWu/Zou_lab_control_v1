"""Qt PulseWorkbench composed from current pulse/application boundaries only."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

from Zou_lab_control.notebook.facade import Experiment, PulseFacade
from zlc_frontend.qt_board import QtOwnerWake
from zlc_neutral_atom.pulse_application import PulseTargetDescriptor
from zlc_neutral_atom.runtime.run import RunHandle, RunSnapshot, RunState
from zlc_pulse import (
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    PORT_DAC,
    PORT_DIGITAL,
    TIME_UNIT_TO_NS,
    AnalogStep,
    ApiParameter,
    DestructivePulseEditError,
    OutputDelay,
    PulseDocument,
    PulseExecutionForm,
    PulseFieldRef,
    PulseTarget,
    PulseTimelineDocument,
    RepeatRegion,
    ScanParameter,
    freeze_scan_table,
    insert_period,
    move_period,
    new_period,
    remove_period,
    replace_field_binding,
    replace_pulse_field,
    resolve_api_parameters,
    set_analog_action,
    set_digital_output,
    set_output_delay,
)
from zlc_workbench.pulse import PulseEditorSession, project_pulse_preview


_TIME_UNITS = tuple(TIME_UNIT_TO_NS)


def _period_label(document: PulseDocument, period_id: str) -> str:
    period = document.period_by_id[period_id]
    return period.name or period.period_id


def _field_label(document: PulseDocument, field: PulseFieldRef) -> str:
    if field.kind == FIELD_DURATION:
        return f"{_period_label(document, field.period_id)} · duration"
    if field.kind == FIELD_DAC:
        port = document.target.by_key[field.port]
        return f"{_period_label(document, field.period_id)} · {port.label} · DAC"
    port = document.target.by_key[field.port]
    return f"{port.label} · output delay"


class PulseTimelineWidget(QtWidgets.QWidget):
    """Small GUI-owned renderer for one immutable pulse timeline."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._timeline: PulseTimelineDocument | None = None
        self.setObjectName("pulseTimelineCanvas")
        self.setMinimumSize(640, 300)

    @property
    def timeline(self) -> PulseTimelineDocument | None:
        return self._timeline

    def set_timeline(self, timeline: PulseTimelineDocument | None) -> None:
        if timeline is not None and not isinstance(timeline, PulseTimelineDocument):
            raise TypeError("timeline must be PulseTimelineDocument or None")
        self._timeline = timeline
        rows = 1 if timeline is None else len(timeline.rows)
        self.setMinimumHeight(max(300, 86 + rows * 54))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#fcfcfd"))
        timeline = self._timeline
        if timeline is None:
            painter.setPen(QtGui.QColor("#657080"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No current preview")
            return

        left = 150.0
        right = max(left + 1.0, float(self.width() - 18))
        top = 54.0
        row_height = 54.0
        duration = float(timeline.duration_ticks)

        def x(tick: int) -> float:
            return left + (right - left) * float(tick) / duration

        painter.setPen(QtGui.QColor("#354052"))
        title = f"{timeline.title} · {timeline.reference_label}"
        painter.drawText(12, 22, title)
        painter.setPen(QtGui.QColor("#7a8494"))
        painter.drawText(int(left), 43, "0")
        painter.drawText(
            QtCore.QRectF(left, 28, right - left, 18),
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
            f"{timeline.duration_ticks} ticks",
        )

        period_colors = (QtGui.QColor("#edf3ff"), QtGui.QColor("#f4f7fb"))
        period_index = 0
        for annotation in timeline.annotations:
            if annotation.kind != "period":
                continue
            x0, x1 = x(annotation.start_tick), x(annotation.stop_tick)
            painter.fillRect(
                QtCore.QRectF(x0, top - 18, max(1.0, x1 - x0), 16),
                period_colors[period_index % 2],
            )
            if x1 - x0 >= 30:
                painter.setPen(QtGui.QColor("#536078"))
                painter.drawText(
                    QtCore.QRectF(x0 + 2, top - 18, x1 - x0 - 4, 16),
                    QtCore.Qt.AlignCenter,
                    annotation.label,
                )
            period_index += 1

        for row_index, row in enumerate(timeline.rows):
            y0 = top + row_index * row_height
            painter.setPen(QtGui.QColor("#d8dde6"))
            painter.drawLine(int(left), int(y0 + row_height - 5), int(right), int(y0 + row_height - 5))
            painter.setPen(QtGui.QColor("#253047"))
            painter.drawText(
                QtCore.QRectF(8, y0, left - 18, row_height - 7),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                row.label,
            )
            low, high = row.value_range

            def y(value: int) -> float:
                if high == low:
                    return y0 + row_height / 2
                fraction = (float(value) - low) / float(high - low)
                return y0 + row_height - 12 - fraction * (row_height - 24)

            path = QtGui.QPainterPath()
            first = True
            previous_stop = None
            previous_value = None
            for segment in row.segments:
                x0, x1 = x(segment.start_tick), x(segment.stop_tick)
                start_y, stop_y = y(segment.start_value), y(segment.stop_value)
                if row.unit == "logic" and segment.start_value:
                    painter.fillRect(
                        QtCore.QRectF(x0, y0 + 5, max(1.0, x1 - x0), row_height - 17),
                        QtGui.QColor(72, 128, 232, 34),
                    )
                if first:
                    path.moveTo(x0, start_y)
                    first = False
                elif previous_stop is not None and previous_value != segment.start_value:
                    path.lineTo(x0, start_y)
                path.lineTo(x1, stop_y)
                previous_stop = x1
                previous_value = segment.stop_value
            painter.setPen(QtGui.QPen(QtGui.QColor("#2867c7"), 2.0))
            painter.drawPath(path)

        painter.setPen(QtGui.QPen(QtGui.QColor("#8b5cf6"), 1.2, QtCore.Qt.DashLine))
        for annotation in timeline.annotations:
            if annotation.kind != "repeat":
                continue
            x0, x1 = x(annotation.start_tick), x(annotation.stop_tick)
            bracket_y = top + len(timeline.rows) * row_height + 2
            painter.drawLine(int(x0), int(bracket_y), int(x1), int(bracket_y))
            painter.drawText(
                QtCore.QRectF(x0, bracket_y + 1, max(45.0, x1 - x0), 18),
                QtCore.Qt.AlignCenter,
                annotation.label,
            )


class PulseWorkbenchWindow(QtWidgets.QMainWindow):
    """Nonblocking current PulseDocument editor, preview, and Run surface."""

    def __init__(
        self,
        pulse: PulseFacade | None,
        descriptor: PulseTargetDescriptor | None,
        editor: PulseEditorSession,
    ) -> None:
        super().__init__()
        if pulse is not None and not isinstance(pulse, PulseFacade):
            raise TypeError("pulse must be PulseFacade or None")
        if descriptor is not None and not isinstance(descriptor, PulseTargetDescriptor):
            raise TypeError("descriptor must be PulseTargetDescriptor or None")
        if (pulse is None) != (descriptor is None):
            raise ValueError("pulse facade and online target descriptor must appear together")
        if not isinstance(editor, PulseEditorSession):
            raise TypeError("editor must be PulseEditorSession")
        self._pulse = pulse
        self._descriptor = descriptor
        self._editor = editor
        self._pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="zlc-pulse-workbench",
        )
        self._lock = threading.Lock()
        self._tracked: set[Future] = set()
        self._pending_results: list[tuple[str, object, Future]] = []
        self._editor_generation = 0
        self._preview_inflight_token: tuple[int, int] | None = None
        self._preview_requested_token = (self._editor_generation, editor.revision)
        self._operation_generation = 0
        self._run_revision: int | None = None
        self._run_starting = False
        self._cancel_when_started = False
        self._handle: RunHandle | None = None
        self._last_snapshot: RunSnapshot | None = None
        self._owner_reaped = True
        self._reap_inflight = False
        self._save_inflight = False
        self._load_inflight = False
        self._scan_dirty = False
        self._refreshing = False
        self._diagnostic_text = ""
        self._closing = False
        self._allow_close = False
        self._discard_on_close = False
        self._pool_closed = False
        self._delay_rows: dict[str, tuple[QtWidgets.QTableWidgetItem, QtWidgets.QDoubleSpinBox, QtWidgets.QComboBox]] = {}

        self.setWindowTitle("Pulse Workbench")
        self.resize(1180, 760)
        self._build_ui()
        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._poll_run_snapshot)
        self._timer.start()
        self._refresh_editor()
        self._schedule_preview()
        self._pulse_status.setText(
            "Pulse: OFFLINE · execution disabled"
            if self._pulse is None
            else "Pulse: READY"
        )

    @property
    def editor_session(self) -> PulseEditorSession:
        return self._editor

    @property
    def current_document(self) -> PulseDocument:
        return self._editor.document

    @property
    def active_snapshot(self) -> RunSnapshot | None:
        return self._last_snapshot

    @property
    def timeline(self) -> PulseTimelineDocument | None:
        return self._timeline.timeline

    @property
    def worker_idle(self) -> bool:
        with self._lock:
            return not self._tracked and not self._pending_results

    def request_close(self, *, discard_unsaved: bool = False) -> None:
        self._discard_on_close = bool(discard_unsaved)
        self.close()

    def open_path(self, path: str | Path) -> None:
        if (
            self._save_inflight
            or self._load_inflight
            or self._run_busy
            or self._scan_dirty
        ):
            raise RuntimeError("cannot load a pulse while another operation is active")
        resolved = Path(path).expanduser().resolve()
        self._load_inflight = True
        self._set_diagnostic("")

        def load() -> PulseEditorSession:
            session = PulseEditorSession.load(resolved)
            if self._descriptor is not None:
                session.bind_target(self._descriptor.target)
            return session

        self._submit(
            "load",
            resolved,
            load,
        )
        self._update_controls()

    @property
    def _run_busy(self) -> bool:
        if self._run_starting:
            return True
        handle = self._handle
        return handle is not None and (
            not handle.snapshot().state.terminal or not self._owner_reaped
        )

    def _build_ui(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._new_action = file_menu.addAction("&New")
        self._open_action = file_menu.addAction("&Open…")
        self._save_action = file_menu.addAction("&Save")
        self._save_as_action = file_menu.addAction("Save &As…")
        file_menu.addSeparator()
        self._close_action = file_menu.addAction("&Close")

        central = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(central)
        identity = QtWidgets.QHBoxLayout()
        identity.addWidget(QtWidgets.QLabel("Pulse name"))
        self._document_name = QtWidgets.QLineEdit()
        self._document_name.setObjectName("pulseDocumentName")
        identity.addWidget(self._document_name, 1)
        self._target_label = QtWidgets.QLabel()
        self._target_label.setObjectName("pulseTargetSummary")
        identity.addWidget(self._target_label)
        outer.addLayout(identity)

        run_bar = QtWidgets.QHBoxLayout()
        self._run_once = QtWidgets.QPushButton("Run Once")
        self._run_once.setObjectName("runOnceButton")
        self._run_scan = QtWidgets.QPushButton("Run Scan")
        self._run_scan.setObjectName("runScanButton")
        self._hold = QtWidgets.QPushButton("On Pulse (HOLD)")
        self._hold.setObjectName("holdButton")
        self._stop = QtWidgets.QPushButton("Stop")
        self._stop.setObjectName("stopButton")
        self._nominal_reference = QtWidgets.QCheckBox(
            "Use nominal reference for Run Once / HOLD when scan-authored"
        )
        self._nominal_reference.setObjectName("nominalReferenceCheck")
        for widget in (self._run_once, self._run_scan, self._hold, self._stop):
            run_bar.addWidget(widget)
        run_bar.addWidget(self._nominal_reference)
        run_bar.addStretch(1)
        outer.addLayout(run_bar)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setObjectName("pulseTabs")
        self._tabs.addTab(self._build_edit_tab(), "Edit")
        self._tabs.addTab(self._build_preview_tab(), "Preview")
        self._tabs.addTab(self._build_scan_tab(), "Scan")
        outer.addWidget(self._tabs, 1)

        self._pulse_status = QtWidgets.QLabel()
        self._pulse_status.setObjectName("pulseStatus")
        self._preview_status = QtWidgets.QLabel()
        self._preview_status.setObjectName("pulsePreviewStatus")
        self._diagnostics = QtWidgets.QLabel()
        self._diagnostics.setObjectName("pulseDiagnostics")
        self._diagnostics.setWordWrap(True)
        outer.addWidget(self._pulse_status)
        outer.addWidget(self._preview_status)
        outer.addWidget(self._diagnostics)
        self.setCentralWidget(central)

        self._new_action.triggered.connect(self._new_document)
        self._open_action.triggered.connect(self._choose_open)
        self._save_action.triggered.connect(self._save)
        self._save_as_action.triggered.connect(self._save_as)
        self._close_action.triggered.connect(self.close)
        self._document_name.editingFinished.connect(self._commit_document_name)
        self._run_once.clicked.connect(
            lambda: self._start_execution(PulseExecutionForm.STATIC_ONCE)
        )
        self._run_scan.clicked.connect(
            lambda: self._start_execution(PulseExecutionForm.AUTONOMOUS_SCAN_ONCE)
        )
        self._hold.clicked.connect(
            lambda: self._start_execution(PulseExecutionForm.CONTINUOUS_MONITOR)
        )
        self._stop.clicked.connect(self._cancel_execution)
        self._nominal_reference.toggled.connect(self._update_controls)

    def _build_edit_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        period_side = QtWidgets.QWidget()
        period_layout = QtWidgets.QVBoxLayout(period_side)
        self._periods = QtWidgets.QListWidget()
        self._periods.setObjectName("pulsePeriodList")
        period_layout.addWidget(self._periods, 1)
        buttons = QtWidgets.QHBoxLayout()
        self._period_add = QtWidgets.QPushButton("Add")
        self._period_remove = QtWidgets.QPushButton("Remove")
        self._period_up = QtWidgets.QPushButton("←")
        self._period_down = QtWidgets.QPushButton("→")
        for button in (
            self._period_add,
            self._period_remove,
            self._period_up,
            self._period_down,
        ):
            buttons.addWidget(button)
        period_layout.addLayout(buttons)
        layout.addWidget(period_side, 0)

        details_scroll = QtWidgets.QScrollArea()
        details_scroll.setWidgetResizable(True)
        details = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details)
        form = QtWidgets.QFormLayout()
        self._period_name = QtWidgets.QLineEdit()
        self._period_name.setObjectName("pulsePeriodName")
        self._period_duration = QtWidgets.QDoubleSpinBox()
        self._period_duration.setObjectName("pulsePeriodDuration")
        self._period_duration.setDecimals(9)
        self._period_duration.setRange(1e-9, 1e15)
        self._period_unit = QtWidgets.QComboBox()
        self._period_unit.setObjectName("pulsePeriodUnit")
        self._period_unit.addItems(_TIME_UNITS)
        duration_row = QtWidgets.QHBoxLayout()
        duration_row.addWidget(self._period_duration, 1)
        duration_row.addWidget(self._period_unit)
        duration_widget = QtWidgets.QWidget()
        duration_widget.setLayout(duration_row)
        form.addRow("Period name", self._period_name)
        form.addRow("Duration", duration_widget)
        details_layout.addLayout(form)

        self._digital_table = QtWidgets.QTableWidget(0, 2)
        self._digital_table.setObjectName("pulseDigitalTable")
        self._digital_table.setHorizontalHeaderLabels(("Digital output", "On"))
        self._digital_table.horizontalHeader().setStretchLastSection(True)
        details_layout.addWidget(QtWidgets.QLabel("Digital outputs"))
        details_layout.addWidget(self._digital_table)

        self._dac_table = QtWidgets.QTableWidget(0, 3)
        self._dac_table.setObjectName("pulseDacTable")
        self._dac_table.setHorizontalHeaderLabels(("DAC output", "Action", "Target"))
        self._dac_table.horizontalHeader().setStretchLastSection(True)
        details_layout.addWidget(QtWidgets.QLabel("DAC actions (Hold preserves carried value)"))
        details_layout.addWidget(self._dac_table)

        self._visible_outputs = QtWidgets.QListWidget()
        self._visible_outputs.setObjectName("pulseVisibleOutputs")
        details_layout.addWidget(QtWidgets.QLabel("Preview rows"))
        details_layout.addWidget(self._visible_outputs)

        self._delay_table = QtWidgets.QTableWidget(0, 4)
        self._delay_table.setObjectName("pulseDelayTable")
        self._delay_table.setHorizontalHeaderLabels(("Use", "Output", "Delay", "Unit"))
        self._delay_table.horizontalHeader().setStretchLastSection(True)
        details_layout.addWidget(QtWidgets.QLabel("Physical output delays"))
        details_layout.addWidget(self._delay_table)

        repeat_box = QtWidgets.QGroupBox("Finite repeat region")
        repeat_layout = QtWidgets.QGridLayout(repeat_box)
        self._repeat_enabled = QtWidgets.QCheckBox("Repeat")
        self._repeat_start = QtWidgets.QComboBox()
        self._repeat_end = QtWidgets.QComboBox()
        self._repeat_count = QtWidgets.QSpinBox()
        self._repeat_count.setRange(2, 1_000_000)
        repeat_layout.addWidget(self._repeat_enabled, 0, 0)
        repeat_layout.addWidget(QtWidgets.QLabel("Start"), 0, 1)
        repeat_layout.addWidget(self._repeat_start, 0, 2)
        repeat_layout.addWidget(QtWidgets.QLabel("End"), 1, 1)
        repeat_layout.addWidget(self._repeat_end, 1, 2)
        repeat_layout.addWidget(QtWidgets.QLabel("Count"), 2, 1)
        repeat_layout.addWidget(self._repeat_count, 2, 2)
        details_layout.addWidget(repeat_box)
        details_layout.addStretch(1)
        details_scroll.setWidget(details)
        layout.addWidget(details_scroll, 1)

        self._periods.currentRowChanged.connect(self._refresh_selected_period)
        self._period_add.clicked.connect(self._add_period)
        self._period_remove.clicked.connect(self._remove_period)
        self._period_up.clicked.connect(lambda: self._move_period(-1))
        self._period_down.clicked.connect(lambda: self._move_period(1))
        self._period_name.editingFinished.connect(self._commit_period_name)
        self._period_duration.editingFinished.connect(self._commit_period_duration)
        self._period_unit.currentTextChanged.connect(self._commit_period_unit)
        self._digital_table.itemChanged.connect(self._commit_digital)
        self._visible_outputs.itemChanged.connect(self._commit_visible_outputs)
        self._delay_table.itemChanged.connect(self._delay_enabled_changed)
        self._repeat_enabled.toggled.connect(self._commit_repeat)
        self._repeat_start.currentIndexChanged.connect(self._commit_repeat)
        self._repeat_end.currentIndexChanged.connect(self._commit_repeat)
        self._repeat_count.valueChanged.connect(self._commit_repeat)
        return tab

    def _build_preview_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        note = QtWidgets.QLabel(
            "Preview is compiler-derived. Scan/API documents show the authored nominal reference; "
            "hardware execution never silently uses API nominal values."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self._timeline = PulseTimelineWidget()
        scroll.setWidget(self._timeline)
        layout.addWidget(scroll, 1)
        return tab

    def _build_scan_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        binding_box = QtWidgets.QGroupBox("Literal / Scan / API field intent")
        binding_layout = QtWidgets.QGridLayout(binding_box)
        self._field_selector = QtWidgets.QComboBox()
        self._field_selector.setObjectName("pulseFieldSelector")
        self._binding_status = QtWidgets.QLabel()
        self._bind_scan = QtWidgets.QPushButton("Bind as Scan")
        self._bind_api = QtWidgets.QPushButton("Bind as API")
        self._unbind = QtWidgets.QPushButton("Set Literal")
        binding_layout.addWidget(self._field_selector, 0, 0, 1, 3)
        binding_layout.addWidget(self._binding_status, 1, 0, 1, 3)
        binding_layout.addWidget(self._bind_scan, 2, 0)
        binding_layout.addWidget(self._bind_api, 2, 1)
        binding_layout.addWidget(self._unbind, 2, 2)
        layout.addWidget(binding_box)

        scan_box = QtWidgets.QGroupBox("Frozen autonomous scan table")
        scan_layout = QtWidgets.QVBoxLayout(scan_box)
        self._scan_table = QtWidgets.QTableWidget()
        self._scan_table.setObjectName("pulseScanTable")
        scan_layout.addWidget(self._scan_table)
        scan_buttons = QtWidgets.QHBoxLayout()
        self._scan_add_row = QtWidgets.QPushButton("Add row")
        self._scan_remove_rows = QtWidgets.QPushButton("Remove selected")
        self._scan_paste = QtWidgets.QPushButton("Paste TSV")
        self._scan_apply = QtWidgets.QPushButton("Apply frozen table")
        self._scan_discard = QtWidgets.QPushButton("Discard table edits")
        self._scan_apply.setObjectName("applyScanTableButton")
        for button in (
            self._scan_add_row,
            self._scan_remove_rows,
            self._scan_paste,
            self._scan_apply,
            self._scan_discard,
        ):
            scan_buttons.addWidget(button)
        scan_buttons.addStretch(1)
        scan_layout.addLayout(scan_buttons)
        layout.addWidget(scan_box, 1)

        api_box = QtWidgets.QGroupBox("Explicit API values for the next hardware Run")
        api_layout = QtWidgets.QVBoxLayout(api_box)
        self._api_table = QtWidgets.QTableWidget(0, 5)
        self._api_table.setObjectName("pulseApiTable")
        self._api_table.setHorizontalHeaderLabels(
            ("ParameterId", "Field", "Unit", "Run value", "Use")
        )
        self._api_table.horizontalHeader().setStretchLastSection(True)
        api_layout.addWidget(self._api_table)
        layout.addWidget(api_box)

        self._field_selector.currentIndexChanged.connect(self._refresh_binding_status)
        self._bind_scan.clicked.connect(self._bind_selected_scan)
        self._bind_api.clicked.connect(self._bind_selected_api)
        self._unbind.clicked.connect(self._unbind_selected_field)
        self._scan_table.itemChanged.connect(self._scan_table_changed)
        self._scan_add_row.clicked.connect(self._add_scan_row)
        self._scan_remove_rows.clicked.connect(self._remove_scan_rows)
        self._scan_paste.clicked.connect(self._paste_scan_table)
        self._scan_apply.clicked.connect(self._apply_scan_table)
        self._scan_discard.clicked.connect(self._discard_scan_table)
        self._api_table.itemChanged.connect(self._api_table_changed)
        return tab

    def _submit(self, kind: str, token: object, work) -> Future:
        future = self._pool.submit(work)
        with self._lock:
            self._tracked.add(future)

        def done(completed: Future) -> None:
            with self._lock:
                self._tracked.discard(completed)
                self._pending_results.append((kind, token, completed))
            self._wake.request_owner_wake()

        future.add_done_callback(done)
        return future

    def _schedule_preview(self) -> None:
        if self._closing:
            return
        revision, document = self._editor.snapshot()
        token = (self._editor_generation, revision)
        self._preview_requested_token = token
        self._timeline.set_timeline(None)
        self._preview_status.setText(f"Preview: COMPILING editor rev {revision}")
        if self._preview_inflight_token is not None:
            return
        self._preview_inflight_token = token
        self._submit(
            "preview",
            token,
            lambda: project_pulse_preview(document),
        )

    def _owner_cycle(self) -> None:
        try:
            self._drain_worker_results()
            self._poll_run_snapshot()
            self._maybe_finish_close()
        except BaseException as error:
            self._set_diagnostic(f"{type(error).__name__}: {error}")
            self._maybe_finish_close()

    def _drain_worker_results(self) -> None:
        with self._lock:
            pending, self._pending_results = self._pending_results, []
        for kind, token, future in pending:
            if kind == "preview":
                self._preview_inflight_token = None
                current_token = (self._editor_generation, self._editor.revision)
                if not self._closing and token == current_token:
                    try:
                        timeline = future.result()
                    except BaseException as error:
                        self._timeline.set_timeline(None)
                        self._preview_status.setText(
                            f"Preview: INVALID editor rev {token[1]}"
                        )
                        self._set_diagnostic(
                            f"Preview failed: {type(error).__name__}: {error}"
                        )
                    else:
                        self._timeline.set_timeline(timeline)
                        self._preview_status.setText(
                            f"Preview: READY editor rev {token[1]} · {timeline.reference_label}"
                        )
                if (
                    not self._closing
                    and self._preview_requested_token != token
                ):
                    self._schedule_preview()
                continue
            if kind == "start":
                generation, revision = token
                self._run_starting = False
                try:
                    handle = future.result()
                except BaseException as error:
                    if generation == self._operation_generation:
                        self._owner_reaped = True
                        self._run_revision = None
                        self._set_diagnostic(
                            f"Pulse start failed: {type(error).__name__}: {error}"
                        )
                        self._pulse_status.setText("Pulse: FAILED BEFORE ADMISSION")
                    self._update_controls()
                    continue
                if generation != self._operation_generation:
                    handle.cancel("stale PulseWorkbench start result")
                    continue
                self._handle = handle
                self._run_revision = revision
                self._last_snapshot = handle.snapshot()
                if self._closing or self._cancel_when_started:
                    handle.cancel("PulseWorkbench stop requested before admission returned")
                self._reconcile_snapshot(self._last_snapshot)
                continue
            if kind == "reap":
                self._reap_inflight = False
                try:
                    future.result()
                except BaseException as error:
                    self._set_diagnostic(
                        f"Pulse owner reap failed: {type(error).__name__}: {error}"
                    )
                else:
                    self._owner_reaped = True
                self._update_controls()
                continue
            if kind == "save":
                self._save_inflight = False
                try:
                    saved = future.result()
                except BaseException as error:
                    if token == (self._editor_generation, self._editor):
                        self._set_diagnostic(
                            f"Save failed: {type(error).__name__}: {error}"
                        )
                else:
                    if token == (self._editor_generation, self._editor):
                        self._set_diagnostic(f"Saved {saved}")
                        self._refresh_title()
                self._update_controls()
                continue
            if kind == "load":
                self._load_inflight = False
                try:
                    loaded = future.result()
                except BaseException as error:
                    self._set_diagnostic(
                        f"Open failed: {type(error).__name__}: {error}"
                    )
                else:
                    if not self._closing and not self._run_busy:
                        self._editor = loaded
                        self._editor_generation += 1
                        self._scan_dirty = False
                        self._refresh_editor()
                        self._schedule_preview()
                        suffix = " · rebound to online target" if loaded.dirty else ""
                        self._set_diagnostic(f"Opened {loaded.path}{suffix}")
                self._update_controls()

    def _refresh_editor(self, *, selected_period_id: str | None = None) -> None:
        document = self._editor.document
        if selected_period_id is None and self._periods.currentItem() is not None:
            selected_period_id = self._periods.currentItem().data(QtCore.Qt.UserRole)
        self._refreshing = True
        try:
            self._document_name.setText(document.name)
            self._target_label.setText(
                f"{(1e9 / document.time_step_ns) / 1e6:g} MHz · "
                f"{len(document.target.ports)} logical ports"
            )
            self._periods.clear()
            selected_row = 0
            for row, period in enumerate(document.periods):
                item = QtWidgets.QListWidgetItem(period.name or period.period_id)
                item.setData(QtCore.Qt.UserRole, period.period_id)
                self._periods.addItem(item)
                if period.period_id == selected_period_id:
                    selected_row = row
            self._periods.setCurrentRow(selected_row)
            self._refresh_repeat(document)
            self._refresh_fields(document)
            self._refresh_scan_table(document)
            self._refresh_api_table(document)
        finally:
            self._refreshing = False
        self._refresh_selected_period(self._periods.currentRow())
        self._refresh_binding_status()
        self._scan_dirty = False
        self._refresh_title()
        self._update_controls()

    def _refresh_selected_period(self, row: int) -> None:
        if self._refreshing:
            return
        document = self._editor.document
        if not 0 <= row < len(document.periods):
            return
        period = document.periods[row]
        self._refreshing = True
        try:
            self._period_name.setText(period.name)
            self._period_duration.setValue(float(period.duration))
            self._period_unit.setCurrentText(period.unit)
            self._refresh_outputs(document, period.period_id)
        finally:
            self._refreshing = False
        self._period_remove.setEnabled(len(document.periods) > 1)
        self._period_up.setEnabled(row > 0)
        self._period_down.setEnabled(row + 1 < len(document.periods))

    def _refresh_outputs(self, document: PulseDocument, period_id: str) -> None:
        period = document.period_by_id[period_id]
        digital_ports = tuple(
            port for port in document.target.ports if port.kind == PORT_DIGITAL
        )
        self._digital_table.setRowCount(len(digital_ports))
        for row, port in enumerate(digital_ports):
            name = QtWidgets.QTableWidgetItem(port.label)
            name.setData(QtCore.Qt.UserRole, port.key)
            name.setFlags(name.flags() & ~QtCore.Qt.ItemIsEditable)
            lane = document.target.raw_lanes.index(port.lanes[0])
            state = QtWidgets.QTableWidgetItem()
            state.setFlags(
                (state.flags() | QtCore.Qt.ItemIsUserCheckable)
                & ~QtCore.Qt.ItemIsEditable
            )
            state.setCheckState(
                QtCore.Qt.Checked if period.states[lane] else QtCore.Qt.Unchecked
            )
            self._digital_table.setItem(row, 0, name)
            self._digital_table.setItem(row, 1, state)

        dac_ports = tuple(port for port in document.target.ports if port.kind == PORT_DAC)
        steps = {step.port: step for step in period.analog_steps}
        self._dac_table.setRowCount(len(dac_ports))
        for row, port in enumerate(dac_ports):
            name = QtWidgets.QTableWidgetItem(port.label)
            name.setData(QtCore.Qt.UserRole, port.key)
            name.setFlags(name.flags() & ~QtCore.Qt.ItemIsEditable)
            self._dac_table.setItem(row, 0, name)
            mode = QtWidgets.QComboBox()
            mode.addItems(("Hold", "Edge", "Ramp"))
            step = steps.get(port.key)
            mode.setCurrentText("Hold" if step is None else step.mode.title())
            value = QtWidgets.QSpinBox()
            assert port.signed_range is not None
            value.setRange(*port.signed_range)
            value.setValue(port.safe_value if step is None else step.value)
            mode.currentTextChanged.connect(
                lambda _text, key=port.key, combo=mode, spin=value: self._commit_dac(
                    key, combo.currentText(), spin.value()
                )
            )
            value.valueChanged.connect(
                lambda _number, key=port.key, combo=mode, spin=value: self._commit_dac(
                    key, combo.currentText(), spin.value()
                )
            )
            self._dac_table.setCellWidget(row, 1, mode)
            self._dac_table.setCellWidget(row, 2, value)

        visible = set(document.visible_ports)
        self._visible_outputs.clear()
        for port in document.target.ports:
            if port.kind not in (PORT_DIGITAL, PORT_DAC):
                continue
            item = QtWidgets.QListWidgetItem(port.label)
            item.setData(QtCore.Qt.UserRole, port.key)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked if port.key in visible else QtCore.Qt.Unchecked
            )
            self._visible_outputs.addItem(item)

        output_ports = tuple(
            port
            for port in document.target.ports
            if port.kind in (PORT_DIGITAL, PORT_DAC)
        )
        delays = {delay.port: delay for delay in document.delays}
        self._delay_rows.clear()
        self._delay_table.setRowCount(len(output_ports))
        for row, port in enumerate(output_ports):
            delay = delays.get(port.key)
            enabled = QtWidgets.QTableWidgetItem()
            enabled.setData(QtCore.Qt.UserRole, port.key)
            enabled.setFlags(
                (enabled.flags() | QtCore.Qt.ItemIsUserCheckable)
                & ~QtCore.Qt.ItemIsEditable
            )
            enabled.setCheckState(
                QtCore.Qt.Checked if delay is not None else QtCore.Qt.Unchecked
            )
            name = QtWidgets.QTableWidgetItem(port.label)
            name.setFlags(name.flags() & ~QtCore.Qt.ItemIsEditable)
            value = QtWidgets.QDoubleSpinBox()
            value.setDecimals(9)
            value.setRange(-1e15, 1e15)
            value.setValue(0.0 if delay is None else float(delay.value))
            unit = QtWidgets.QComboBox()
            unit.addItems(_TIME_UNITS)
            unit.setCurrentText("ns" if delay is None else delay.unit)
            self._delay_table.setItem(row, 0, enabled)
            self._delay_table.setItem(row, 1, name)
            self._delay_table.setCellWidget(row, 2, value)
            self._delay_table.setCellWidget(row, 3, unit)
            self._delay_rows[port.key] = (enabled, value, unit)
            value.editingFinished.connect(
                lambda key=port.key: self._commit_delay(key)
            )
            unit.currentTextChanged.connect(
                lambda _text, key=port.key: self._commit_delay(key)
            )

    def _refresh_repeat(self, document: PulseDocument) -> None:
        repeat = document.repeat
        labels = [period.name or period.period_id for period in document.periods]
        ids = [period.period_id for period in document.periods]
        self._repeat_start.clear()
        self._repeat_end.clear()
        for label, period_id in zip(labels, ids):
            self._repeat_start.addItem(label, period_id)
            self._repeat_end.addItem(label, period_id)
        self._repeat_enabled.setChecked(repeat is not None)
        if repeat is None:
            self._repeat_start.setCurrentIndex(0)
            self._repeat_end.setCurrentIndex(len(ids) - 1)
            self._repeat_count.setValue(2)
        else:
            self._repeat_start.setCurrentIndex(ids.index(repeat.start_period_id))
            self._repeat_end.setCurrentIndex(ids.index(repeat.end_period_id))
            self._repeat_count.setValue(repeat.count)

    def _editable_fields(self, document: PulseDocument) -> tuple[PulseFieldRef, ...]:
        fields: list[PulseFieldRef] = []
        for period in document.periods:
            fields.append(PulseFieldRef(FIELD_DURATION, period.period_id))
            fields.extend(
                PulseFieldRef(FIELD_DAC, period.period_id, step.port)
                for step in period.analog_steps
            )
        fields.extend(
            PulseFieldRef(FIELD_DELAY, None, delay.port)
            for delay in document.delays
        )
        return tuple(fields)

    def _refresh_fields(self, document: PulseDocument) -> None:
        selected = self._field_selector.currentData()
        self._field_selector.clear()
        selected_index = 0
        for index, field in enumerate(self._editable_fields(document)):
            self._field_selector.addItem(_field_label(document, field), field)
            if field == selected:
                selected_index = index
        if self._field_selector.count():
            self._field_selector.setCurrentIndex(selected_index)
        self._refresh_binding_status()

    def _refresh_binding_status(self) -> None:
        if self._refreshing:
            return
        field = self._field_selector.currentData()
        document = self._editor.document
        scan = next((item for item in document.scan_parameters if item.field == field), None)
        api = next((item for item in document.api_parameters if item.field == field), None)
        if scan is not None:
            text = f"Current intent: Scan · {scan.parameter_id} [{scan.unit}]"
        elif api is not None:
            text = f"Current intent: API · {api.parameter_id} [{api.unit}]"
        else:
            text = "Current intent: Literal"
        self._binding_status.setText(text)
        self._bind_scan.setEnabled(field is not None and field.kind != FIELD_DELAY)
        self._bind_api.setEnabled(field is not None)
        self._unbind.setEnabled(scan is not None or api is not None)

    def _refresh_scan_table(self, document: PulseDocument) -> None:
        columns = tuple(parameter.parameter_id for parameter in document.scan_parameters)
        self._scan_table.clear()
        self._scan_table.setColumnCount(len(columns))
        self._scan_table.setHorizontalHeaderLabels(columns)
        table = document.scan_table
        rows = () if table is None else table.rows
        self._scan_table.setRowCount(len(rows))
        if table is not None:
            by_column = {column: index for index, column in enumerate(table.columns)}
            for row_index, row in enumerate(rows):
                for column_index, parameter_id in enumerate(columns):
                    self._scan_table.setItem(
                        row_index,
                        column_index,
                        QtWidgets.QTableWidgetItem(
                            str(row[by_column[parameter_id]])
                        ),
                    )

    def _refresh_api_table(self, document: PulseDocument) -> None:
        self._api_table.setRowCount(len(document.api_parameters))
        for row, parameter in enumerate(document.api_parameters):
            value, value_unit = document.field_value(parameter.field)
            values = (
                parameter.parameter_id,
                _field_label(document, parameter.field),
                parameter.unit,
                str(value) if value_unit == parameter.unit else "",
            )
            for column, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if column != 3:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self._api_table.setItem(row, column, item)
            confirmed = QtWidgets.QTableWidgetItem()
            confirmed.setFlags(
                (confirmed.flags() | QtCore.Qt.ItemIsUserCheckable)
                & ~QtCore.Qt.ItemIsEditable
            )
            confirmed.setCheckState(QtCore.Qt.Unchecked)
            self._api_table.setItem(row, 4, confirmed)

    def _apply_document(
        self,
        document: PulseDocument,
        *,
        selected_period_id: str | None = None,
        allow_scan_dirty: bool = False,
    ) -> bool:
        if self._scan_dirty and not allow_scan_dirty:
            self._set_diagnostic("Apply or discard the edited scan table first")
            return False
        previous = self._editor.revision
        revision = self._editor.replace_document(document)
        if revision == previous:
            return False
        self._refresh_editor(selected_period_id=selected_period_id)
        self._schedule_preview()
        if self._last_snapshot is not None:
            self._reconcile_snapshot(self._last_snapshot)
        return True

    def _apply_edit(self, operation, description: str, *, selected_period_id=None) -> None:
        if self._scan_dirty:
            self._set_diagnostic("Apply or discard the edited scan table first")
            return
        try:
            result = operation(False)
        except DestructivePulseEditError as error:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Confirm pulse edit",
                f"{description} also changes scan/API/repeat intent. Apply the complete change?\n\n"
                f"Removed scan: {', '.join(error.impact.removed_scan_parameters) or 'none'}\n"
                f"Removed API: {', '.join(error.impact.removed_api_parameters) or 'none'}",
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
            result = operation(True)
        except BaseException as error:
            self._set_diagnostic(f"{description} failed: {type(error).__name__}: {error}")
            return
        self._apply_document(result.document, selected_period_id=selected_period_id)

    def _selected_period_id(self) -> str | None:
        item = self._periods.currentItem()
        return None if item is None else item.data(QtCore.Qt.UserRole)

    def _commit_document_name(self) -> None:
        if self._refreshing:
            return
        document = self._editor.document
        try:
            changed = replace(document, name=self._document_name.text())
        except BaseException as error:
            self._set_diagnostic(f"Pulse name is invalid: {error}")
            self._refresh_editor()
            return
        self._apply_document(changed, selected_period_id=self._selected_period_id())

    def _commit_period_name(self) -> None:
        if self._refreshing:
            return
        period_id = self._selected_period_id()
        if period_id is None:
            return
        document = self._editor.document
        periods = tuple(
            replace(period, name=self._period_name.text())
            if period.period_id == period_id
            else period
            for period in document.periods
        )
        try:
            changed = replace(document, periods=periods)
        except BaseException as error:
            self._set_diagnostic(f"Period name is invalid: {error}")
            self._refresh_editor(selected_period_id=period_id)
            return
        self._apply_document(changed, selected_period_id=period_id)

    def _commit_period_duration(self) -> None:
        if self._refreshing:
            return
        period_id = self._selected_period_id()
        if period_id is None:
            return
        document = self._editor.document
        try:
            changed = replace_pulse_field(
                document,
                PulseFieldRef(FIELD_DURATION, period_id),
                self._period_duration.value(),
                unit=self._period_unit.currentText(),
            )
        except BaseException as error:
            self._set_diagnostic(f"Duration is not on the target clock grid: {error}")
            self._refresh_editor(selected_period_id=period_id)
            return
        self._apply_document(changed, selected_period_id=period_id)

    def _commit_period_unit(self, unit: str) -> None:
        if self._refreshing:
            return
        period_id = self._selected_period_id()
        if period_id is None:
            return
        document = self._editor.document
        period = document.period_by_id[period_id]
        converted = (
            float(period.duration)
            * TIME_UNIT_TO_NS[period.unit]
            / TIME_UNIT_TO_NS[unit]
        )
        try:
            changed = replace_pulse_field(
                document,
                PulseFieldRef(FIELD_DURATION, period_id),
                converted,
                unit=unit,
            )
        except BaseException as error:
            self._set_diagnostic(f"Duration unit change failed: {error}")
            self._refresh_editor(selected_period_id=period_id)
            return
        self._apply_document(changed, selected_period_id=period_id)

    def _add_period(self) -> None:
        document = self._editor.document
        selected = self._periods.currentRow()
        before = (
            document.periods[selected + 1].period_id
            if 0 <= selected + 1 < len(document.periods)
            else None
        )
        try:
            period = new_period(
                document,
                duration=document.time_step_ns,
                unit="ns",
            )
            result = insert_period(document, period=period, before=before)
        except BaseException as error:
            self._set_diagnostic(f"Add period failed: {error}")
            return
        self._apply_document(result.document, selected_period_id=period.period_id)

    def _remove_period(self) -> None:
        period_id = self._selected_period_id()
        if period_id is None:
            return
        document = self._editor.document
        self._apply_edit(
            lambda cascade: remove_period(document, period_id, cascade=cascade),
            "Remove period",
        )

    def _move_period(self, direction: int) -> None:
        row = self._periods.currentRow()
        document = self._editor.document
        if not 0 <= row < len(document.periods):
            return
        target = row + direction
        if not 0 <= target < len(document.periods):
            return
        period_id = document.periods[row].period_id
        if direction < 0:
            before = document.periods[target].period_id
        else:
            before = (
                document.periods[target + 1].period_id
                if target + 1 < len(document.periods)
                else None
            )
        self._apply_edit(
            lambda cascade: move_period(
                document,
                period_id=period_id,
                before=before,
                cascade=cascade,
            ),
            "Move period",
            selected_period_id=period_id,
        )

    def _commit_digital(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._refreshing or item.column() != 1:
            return
        period_id = self._selected_period_id()
        name = self._digital_table.item(item.row(), 0)
        if period_id is None or name is None:
            return
        try:
            changed = set_digital_output(
                self._editor.document,
                period_id,
                name.data(QtCore.Qt.UserRole),
                item.checkState() == QtCore.Qt.Checked,
            )
        except BaseException as error:
            self._set_diagnostic(f"Digital edit failed: {error}")
            self._refresh_editor(selected_period_id=period_id)
            return
        self._apply_document(changed, selected_period_id=period_id)

    def _commit_dac(self, port: str, mode: str, value: int) -> None:
        if self._refreshing:
            return
        period_id = self._selected_period_id()
        if period_id is None:
            return
        document = self._editor.document
        action = None if mode == "Hold" else AnalogStep(port, mode.lower(), value)
        self._apply_edit(
            lambda cascade: set_analog_action(
                document,
                period_id,
                port,
                action,
                cascade=cascade,
            ),
            "Change DAC action",
            selected_period_id=period_id,
        )

    def _commit_visible_outputs(self, _item) -> None:
        if self._refreshing:
            return
        visible = tuple(
            self._visible_outputs.item(row).data(QtCore.Qt.UserRole)
            for row in range(self._visible_outputs.count())
            if self._visible_outputs.item(row).checkState() == QtCore.Qt.Checked
        )
        if not visible:
            self._set_diagnostic("Preview requires at least one visible logical output")
            self._refresh_editor(selected_period_id=self._selected_period_id())
            return
        try:
            changed = replace(self._editor.document, visible_ports=visible)
        except BaseException as error:
            self._set_diagnostic(f"Preview row edit failed: {error}")
            self._refresh_editor(selected_period_id=self._selected_period_id())
            return
        self._apply_document(changed, selected_period_id=self._selected_period_id())

    def _delay_enabled_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._refreshing or item.column() != 0:
            return
        self._commit_delay(item.data(QtCore.Qt.UserRole))

    def _commit_delay(self, port: str) -> None:
        if self._refreshing:
            return
        enabled, value, unit = self._delay_rows[port]
        document = self._editor.document
        delay = (
            OutputDelay(port, value.value(), unit.currentText())
            if enabled.checkState() == QtCore.Qt.Checked
            else None
        )
        self._apply_edit(
            lambda cascade: set_output_delay(
                document,
                port,
                delay,
                cascade=cascade,
            ),
            "Change output delay",
            selected_period_id=self._selected_period_id(),
        )

    def _commit_repeat(self, *_args) -> None:
        if self._refreshing:
            return
        document = self._editor.document
        try:
            if not self._repeat_enabled.isChecked():
                repeat = None
            else:
                start = self._repeat_start.currentIndex()
                end = self._repeat_end.currentIndex()
                if start > end:
                    end = start
                repeat = RepeatRegion(
                    self._repeat_start.itemData(start),
                    self._repeat_end.itemData(end),
                    self._repeat_count.value(),
                )
            changed = replace(document, repeat=repeat)
        except BaseException as error:
            self._set_diagnostic(f"Repeat edit failed: {error}")
            self._refresh_editor(selected_period_id=self._selected_period_id())
            return
        self._apply_document(changed, selected_period_id=self._selected_period_id())

    def _selected_field(self) -> PulseFieldRef | None:
        value = self._field_selector.currentData()
        return value if isinstance(value, PulseFieldRef) else None

    def _bind_selected_scan(self) -> None:
        field = self._selected_field()
        if field is None or field.kind == FIELD_DELAY:
            return
        document = self._editor.document
        default_id = (
            f"{field.period_id}_duration"
            if field.kind == FIELD_DURATION
            else f"{field.period_id}_{field.port}"
        )
        parameter_id, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Scan ParameterId",
            "Stable ParameterId",
            text=default_id,
        )
        if not accepted:
            return
        label, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Scan label",
            "Display label",
            text=_field_label(document, field),
        )
        if not accepted:
            return
        _value, unit = document.field_value(field)
        try:
            binding = ScanParameter(parameter_id, field, label, unit)
        except BaseException as error:
            self._set_diagnostic(f"Scan binding is invalid: {error}")
            return
        self._apply_edit(
            lambda cascade: replace_field_binding(
                document,
                field,
                binding,
                cascade=cascade,
            ),
            "Change field binding",
            selected_period_id=self._selected_period_id(),
        )

    def _bind_selected_api(self) -> None:
        field = self._selected_field()
        if field is None:
            return
        document = self._editor.document
        default_id = "_".join(
            value for value in (field.period_id, field.port, field.kind) if value
        )
        parameter_id, accepted = QtWidgets.QInputDialog.getText(
            self,
            "API ParameterId",
            "Stable ParameterId",
            text=default_id,
        )
        if not accepted:
            return
        _value, unit = document.field_value(field)
        try:
            binding = ApiParameter(parameter_id, field, unit)
        except BaseException as error:
            self._set_diagnostic(f"API binding is invalid: {error}")
            return
        self._apply_edit(
            lambda cascade: replace_field_binding(
                document,
                field,
                binding,
                cascade=cascade,
            ),
            "Change field binding",
            selected_period_id=self._selected_period_id(),
        )

    def _unbind_selected_field(self) -> None:
        field = self._selected_field()
        if field is None:
            return
        document = self._editor.document
        self._apply_edit(
            lambda cascade: replace_field_binding(
                document,
                field,
                None,
                cascade=cascade,
            ),
            "Set field literal",
            selected_period_id=self._selected_period_id(),
        )

    def _scan_table_changed(self, _item) -> None:
        if self._refreshing:
            return
        self._scan_dirty = True
        self._preview_status.setText("Preview: CURRENT DOCUMENT · scan table edits not applied")
        self._update_controls()

    def _add_scan_row(self) -> None:
        if self._scan_table.columnCount() < 1:
            self._set_diagnostic("Bind at least one field as Scan first")
            return
        row = self._scan_table.rowCount()
        self._scan_table.insertRow(row)
        document = self._editor.document
        for column, parameter in enumerate(document.scan_parameters):
            value, unit = document.field_value(parameter.field)
            text = str(value) if unit == parameter.unit else ""
            if row and self._scan_table.item(row - 1, column) is not None:
                text = self._scan_table.item(row - 1, column).text()
            self._scan_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self._scan_dirty = True
        self._update_controls()

    def _remove_scan_rows(self) -> None:
        rows = sorted({index.row() for index in self._scan_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._scan_table.removeRow(row)
        if rows:
            self._scan_dirty = True
            self._update_controls()

    def _paste_scan_table(self) -> None:
        text = QtWidgets.QApplication.clipboard().text().strip()
        if not text:
            return
        rows = [line.split("\t") for line in text.splitlines() if line.strip()]
        if any(len(row) != self._scan_table.columnCount() for row in rows):
            self._set_diagnostic("Pasted TSV width must equal the named scan columns")
            return
        self._scan_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                self._scan_table.setItem(
                    row_index,
                    column_index,
                    QtWidgets.QTableWidgetItem(value.strip()),
                )
        self._scan_dirty = True
        self._update_controls()

    def _apply_scan_table(self) -> None:
        document = self._editor.document
        columns = tuple(parameter.parameter_id for parameter in document.scan_parameters)
        if not columns:
            self._set_diagnostic("Bind at least one field as Scan first")
            return
        if self._scan_table.rowCount() < 1:
            self._set_diagnostic("A frozen scan table requires at least one row")
            return
        try:
            rows = tuple(
                tuple(
                    float(self._scan_table.item(row, column).text())
                    for column in range(len(columns))
                )
                for row in range(self._scan_table.rowCount())
            )
            table, report = freeze_scan_table(document, columns, rows)
            changed = replace(document, scan_table=table, scan_recipe=None)
        except BaseException as error:
            self._set_diagnostic(f"Scan table is invalid: {type(error).__name__}: {error}")
            return
        self._scan_dirty = False
        self._apply_document(changed, allow_scan_dirty=True)
        self._set_diagnostic(
            "Frozen scan table applied"
            + (
                f" · {report.adjusted_cells} cells snapped to the target grid"
                if report.adjusted_cells
                else ""
            )
        )

    def _discard_scan_table(self) -> None:
        if not self._scan_dirty:
            return
        self._refreshing = True
        try:
            self._refresh_scan_table(self._editor.document)
        finally:
            self._refreshing = False
        self._scan_dirty = False
        self._set_diagnostic("Unapplied scan table edits discarded")
        self._update_controls()

    def _api_table_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._refreshing:
            return
        if item.column() == 3:
            self._refreshing = True
            try:
                self._api_table.item(item.row(), 4).setCheckState(QtCore.Qt.Unchecked)
            finally:
                self._refreshing = False
        self._update_controls()

    def _api_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in range(self._api_table.rowCount()):
            parameter_id = self._api_table.item(row, 0).text()
            confirmed = self._api_table.item(row, 4)
            if confirmed.checkState() != QtCore.Qt.Checked:
                raise ValueError(f"API parameter {parameter_id!r} is not confirmed")
            values[parameter_id] = float(self._api_table.item(row, 3).text())
        return values

    def _api_ready(self) -> bool:
        try:
            self._api_values()
        except (AttributeError, TypeError, ValueError):
            return False
        return True

    def _execution_document(self, form: PulseExecutionForm) -> PulseDocument:
        document = self._editor.document
        if document.api_parameters:
            document = resolve_api_parameters(document, self._api_values())
        if form in (
            PulseExecutionForm.STATIC_ONCE,
            PulseExecutionForm.CONTINUOUS_MONITOR,
        ) and document.scan_parameters:
            if not self._nominal_reference.isChecked():
                raise ValueError("enable the explicit nominal-reference option first")
            document = replace(
                document,
                scan_parameters=(),
                scan_table=None,
                scan_recipe=None,
            )
        return document

    def _start_execution(self, form: PulseExecutionForm) -> None:
        if self._pulse is None or self._run_busy or self._scan_dirty:
            return
        try:
            document = self._execution_document(form)
        except BaseException as error:
            self._set_diagnostic(f"Run request is incomplete: {error}")
            return
        self._operation_generation += 1
        generation = self._operation_generation
        revision = self._editor.revision
        pulse = self._pulse
        self._run_starting = True
        self._cancel_when_started = False
        self._owner_reaped = False
        self._handle = None
        self._run_revision = revision
        self._last_snapshot = None
        self._set_diagnostic("")
        self._pulse_status.setText(f"Pulse: STARTING editor rev {revision}")

        def start() -> RunHandle:
            request = pulse.request(document, form)
            return pulse.start(request)

        self._submit("start", (generation, revision), start)
        self._update_controls()

    def _cancel_execution(self) -> None:
        if self._run_starting and self._handle is None:
            self._cancel_when_started = True
            self._pulse_status.setText("Pulse: STOP REQUESTED · waiting for admission")
            self._update_controls()
            return
        handle = self._handle
        if handle is None or handle.snapshot().state.terminal:
            return
        outcome = handle.cancel("PulseWorkbench Stop")
        self._pulse_status.setText(f"Pulse: STOPPING · {outcome.value}")
        self._update_controls()

    def _poll_run_snapshot(self) -> None:
        handle = self._handle
        if handle is None:
            if self._closing:
                self._wake.request_owner_wake()
            return
        snapshot = handle.snapshot()
        if snapshot != self._last_snapshot:
            self._reconcile_snapshot(snapshot)
        if snapshot.state.terminal and not self._owner_reaped and not self._reap_inflight:
            self._reap_inflight = True
            self._submit("reap", self._operation_generation, handle.wait)

    def _reconcile_snapshot(self, snapshot: RunSnapshot) -> None:
        self._last_snapshot = snapshot
        revision = "?" if self._run_revision is None else str(self._run_revision)
        editor_suffix = (
            ""
            if self._run_revision == self._editor.revision
            else f" · editor rev {self._editor.revision} modified"
        )
        if snapshot.state is RunState.RUNNING and snapshot.phase == "holding-pulse":
            text = f"Pulse: HOLDING rev {revision}{editor_suffix}"
        elif snapshot.state is RunState.SUCCEEDED:
            safe = " · SAFE" if snapshot.safety_bundle_id is not None else ""
            text = f"Pulse: SUCCEEDED rev {revision}{safe}{editor_suffix}"
        elif snapshot.state is RunState.CANCELLED:
            safe = " · SAFE" if snapshot.safety_bundle_id is not None else ""
            text = f"Pulse: STOPPED rev {revision}{safe}{editor_suffix}"
        else:
            text = f"Pulse: {snapshot.state.value} / {snapshot.phase} rev {revision}{editor_suffix}"
        self._pulse_status.setText(text)
        diagnostics = []
        if snapshot.primary_error:
            diagnostics.append(f"Error: {snapshot.primary_error}")
        diagnostics.extend(f"Cleanup: {value}" for value in snapshot.cleanup_errors)
        if diagnostics:
            self._set_diagnostic("\n".join(diagnostics))
        self._update_controls()

    def _update_controls(self) -> None:
        document = self._editor.document
        executable = (
            self._pulse is not None
            and not self._run_busy
            and not self._scan_dirty
            and not self._load_inflight
            and self._api_ready()
            and not self._closing
        )
        nominal_ready = not document.scan_parameters or self._nominal_reference.isChecked()
        self._run_once.setEnabled(executable and nominal_ready)
        self._hold.setEnabled(executable and nominal_ready)
        self._run_scan.setEnabled(
            executable
            and document.scan_table is not None
            and bool(document.scan_table.rows)
        )
        handle = self._handle
        self._stop.setEnabled(
            not self._closing
            and (
                self._run_starting
                or (
                    handle is not None
                    and not handle.snapshot().state.terminal
                )
            )
            and not self._cancel_when_started
        )
        self._nominal_reference.setVisible(bool(document.scan_parameters))
        file_ready = (
            not self._save_inflight
            and not self._load_inflight
            and not self._scan_dirty
            and not self._closing
        )
        self._new_action.setEnabled(not self._run_busy and file_ready)
        self._open_action.setEnabled(not self._run_busy and file_ready)
        self._save_action.setEnabled(not self._save_inflight and file_ready)
        self._save_as_action.setEnabled(not self._save_inflight and file_ready)
        self._scan_apply.setEnabled(self._scan_dirty)
        self._scan_discard.setEnabled(self._scan_dirty)
        self._scan_add_row.setEnabled(bool(document.scan_parameters))
        self._scan_remove_rows.setEnabled(self._scan_table.rowCount() > 0)
        self._scan_paste.setEnabled(bool(document.scan_parameters))
        self._tabs.setTabEnabled(0, not self._scan_dirty)
        editor_enabled = not self._load_inflight and not self._closing
        self._tabs.setEnabled(editor_enabled)
        self._document_name.setEnabled(editor_enabled and not self._scan_dirty)
        self._refresh_binding_status()
        if self._scan_dirty:
            for button in (self._bind_scan, self._bind_api, self._unbind):
                button.setEnabled(False)

    def _refresh_title(self) -> None:
        path = self._editor.path
        suffix = " *" if self._editor.dirty else ""
        location = "unsaved" if path is None else str(path)
        self.setWindowTitle(f"Pulse Workbench — {self._editor.document.name} — {location}{suffix}")

    def _set_diagnostic(self, message: str) -> None:
        self._diagnostic_text = str(message)
        self._diagnostics.setText(self._diagnostic_text)

    def _confirm_discard(self) -> bool:
        if not self._editor.dirty and not self._scan_dirty:
            return True
        return (
            QtWidgets.QMessageBox.question(
                self,
                "Discard unsaved pulse?",
                "The current PulseDocument has unsaved changes. Discard them?",
            )
            == QtWidgets.QMessageBox.Yes
        )

    def _new_document(self) -> None:
        if (
            self._save_inflight
            or self._load_inflight
            or self._run_busy
            or self._scan_dirty
            or not self._confirm_discard()
        ):
            return
        current = self._editor.document
        self._editor = PulseEditorSession.new(
            current.target,
            time_step_ns=current.time_step_ns,
        )
        self._editor_generation += 1
        self._scan_dirty = False
        self._refresh_editor()
        self._schedule_preview()

    def _choose_open(self) -> None:
        if (
            self._save_inflight
            or self._load_inflight
            or self._run_busy
            or self._scan_dirty
            or not self._confirm_discard()
        ):
            return
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open current PulseDocument",
            "",
            "PulseDocument (*.json);;All files (*)",
        )
        if path:
            self.open_path(path)

    def _save(self) -> None:
        if self._save_inflight:
            return
        if self._editor.path is None:
            self._save_as()
            return
        self._submit_save(None, overwrite=False)

    def _save_as(self) -> None:
        if self._save_inflight:
            return
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save current PulseDocument",
            "",
            "PulseDocument (*.json);;All files (*)",
        )
        if path:
            self._submit_save(path, overwrite=True)

    def save_path(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Submit a nonblocking save without opening a file dialog."""

        self._submit_save(path, overwrite=overwrite)

    def _submit_save(self, path: str | Path | None, *, overwrite: bool) -> None:
        if self._save_inflight or self._load_inflight:
            raise RuntimeError("a file operation is already active")
        session = self._editor
        token = (self._editor_generation, session)
        self._save_inflight = True
        self._submit(
            "save",
            token,
            lambda: session.save(path, overwrite=overwrite),
        )
        self._update_controls()

    def closeEvent(self, event) -> None:
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        if not self._closing:
            if not self._discard_on_close and not self._confirm_discard():
                return
            self._closing = True
            self._new_action.setEnabled(False)
            self._open_action.setEnabled(False)
            self._save_action.setEnabled(False)
            self._save_as_action.setEnabled(False)
            self._run_once.setEnabled(False)
            self._run_scan.setEnabled(False)
            self._hold.setEnabled(False)
            self._tabs.setEnabled(False)
            self._document_name.setEnabled(False)
            self._pulse_status.setText("Pulse: CLOSING")
            if self._run_starting and self._handle is None:
                self._cancel_when_started = True
            handle = self._handle
            if handle is not None and not handle.snapshot().state.terminal:
                handle.cancel("PulseWorkbench window is closing")
        self._wake.request_owner_wake()

    def _maybe_finish_close(self) -> None:
        if not self._closing:
            return
        if self._run_starting:
            return
        handle = self._handle
        if handle is not None:
            snapshot = handle.snapshot()
            if not snapshot.state.terminal:
                return
            if not self._owner_reaped:
                if not self._reap_inflight:
                    self._reap_inflight = True
                    self._submit("reap", self._operation_generation, handle.wait)
                return
        with self._lock:
            if self._tracked or self._pending_results:
                return
        if not self._pool_closed:
            self._pool.shutdown(wait=False)
            self._pool_closed = True
        self._timer.stop()
        self._wake.detach()
        self._allow_close = True
        QtCore.QTimer.singleShot(0, self.close)


def _qt_application() -> tuple[QtWidgets.QApplication, bool]:
    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication([])
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("PulseWorkbench must be opened on the Qt GUI thread")
    return application, owns_application


def _show_window(
    window: PulseWorkbenchWindow,
    application: QtWidgets.QApplication,
    owns_application: bool,
) -> PulseWorkbenchWindow:
    if owns_application:
        window._application_owner = application
    window.show()
    return window


def open_pulse_workbench(
    experiment: Experiment,
    document: PulseDocument | None = None,
    *,
    path: str | Path | None = None,
) -> PulseWorkbenchWindow:
    """Open a PulseWorkbench using one existing Experiment authority."""

    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be Experiment")
    if document is not None and path is not None:
        raise ValueError("provide document or path, not both")
    application, owns_application = _qt_application()
    descriptor = experiment.pulse.target
    editor = (
        PulseEditorSession.new(
            descriptor.target,
            time_step_ns=descriptor.time_step_ns,
        )
        if document is None
        else PulseEditorSession(document)
    )
    editor.bind_target(descriptor.target)
    window = _show_window(
        PulseWorkbenchWindow(experiment.pulse, descriptor, editor),
        application,
        owns_application,
    )
    if path is not None:
        window.open_path(path)
    return window


def open_offline_pulse_workbench(
    target: PulseTarget,
    *,
    time_step_ns: int | float,
    document: PulseDocument | None = None,
    path: str | Path | None = None,
) -> PulseWorkbenchWindow:
    """Open authoring/preview only; no fake hardware Run is constructed."""

    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    if document is not None and path is not None:
        raise ValueError("provide document or path, not both")
    application, owns_application = _qt_application()
    editor = (
        PulseEditorSession.new(target, time_step_ns=time_step_ns)
        if document is None
        else PulseEditorSession(document)
    )
    window = _show_window(
        PulseWorkbenchWindow(None, None, editor),
        application,
        owns_application,
    )
    if path is not None:
        window.open_path(path)
    return window


__all__ = [
    "PulseTimelineWidget",
    "PulseWorkbenchWindow",
    "open_offline_pulse_workbench",
    "open_pulse_workbench",
]
