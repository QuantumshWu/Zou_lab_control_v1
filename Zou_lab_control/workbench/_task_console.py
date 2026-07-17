"""Qt composition for the current single-card SCAN_SLOT TaskConsole.

The domain values, edit revision, catalog projection, and persistence codec live
in :mod:`zlc_workbench.task_console`.  This module only owns Qt widgets and the
composition seam from one validated intent to the existing scan panel.
"""

from __future__ import annotations

import math
from pathlib import Path
import re

from PyQt5 import QtCore, QtWidgets

from Zou_lab_control.notebook.facade import Experiment
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
from zlc_neutral_atom.readout.calibration import ReadoutModelKind
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.occupancy import OCCUPANCY_STREAM_PROCESSOR_KEY
from zlc_neutral_atom.scan.contracts import AUTONOMOUS_SCAN_SLOT_TASK_KEY
from zlc_pulse import PulseDocument, load_pulse_document
from zlc_workbench.task_console import (
    ScanDisplayIntent,
    ScanEditConflict,
    ScanEditDraft,
    ScanEditorSession,
    TaskConsoleScanIntent,
    compose_task_console_catalog,
    describe_authoritative_transform,
    load_task_console_scan_intent,
    save_task_console_scan_intent,
    task_console_catalog_items,
)

from ._scan import ScanWorkbenchWindow


_DIRECT_SOURCE = "direct"
_OCCUPANCY_SOURCE = "occupancy"
_EMBEDDED_DOCUMENT = "(embedded PulseDocument)"


def _parse_number(text: str, field: str) -> int | float:
    value = text.strip()
    if not value:
        raise ValueError(f"{field} is required")
    try:
        parsed: int | float
        if re.fullmatch(r"[+-]?\d+", value):
            parsed = int(value)
        else:
            parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(float(parsed)):
        raise ValueError(f"{field} must be finite")
    return parsed


def _parse_positive_integer(text: str, field: str) -> int:
    parsed = _parse_number(text, field)
    if not isinstance(parsed, int) or parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _parse_positive_real(text: str, field: str) -> float:
    parsed = float(_parse_number(text, field))
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _intent_request(experiment: Experiment, intent: TaskConsoleScanIntent):
    common = dict(
        camera_role=intent.camera_role,
        sequencer_role=intent.sequencer_role,
        trigger_channel=intent.trigger_channel,
        api_values=intent.fixed_api_values,
        output_transform_spec=intent.output_transform_spec,
        transport_memory_limit_bytes=intent.transport_memory_limit_bytes,
        memory_limit_bytes=intent.memory_limit_bytes,
        timeout_seconds=intent.timeout_seconds,
    )
    if intent.processor_key is None:
        return experiment.readout.scan_request(intent.pulse_document, **common)
    assert intent.calibration_ref is not None
    return experiment.readout.occupancy_scan_request(
        intent.pulse_document,
        calibration_ref=intent.calibration_ref,
        model_kind=intent.model_kind,
        **common,
    )


class ScanIntentForm(QtWidgets.QWidget):
    """The one scan-specific editor used by both Setting and Edit surfaces."""

    applied = QtCore.pyqtSignal()

    def __init__(
        self,
        card: "TaskScanCard",
        *,
        object_prefix: str,
    ) -> None:
        super().__init__(card)
        self._card = card
        self._prefix = object_prefix
        self._base_revision: int | None = None
        self._document: PulseDocument | None = None
        self._document_source: str | None = None
        self._api_edits: dict[str, QtWidgets.QLineEdit] = {}
        self._output_transform_spec = None
        self.setObjectName(f"{object_prefix}ScanIntentForm")

        self._pulse_path = QtWidgets.QLineEdit(self)
        self._pulse_path.setObjectName(f"{object_prefix}PulseDocumentPath")
        self._pulse_path.setPlaceholderText("Current PulseDocument JSON")
        self._load_pulse = QtWidgets.QPushButton("Load JSON", self)
        self._load_pulse.setObjectName(f"{object_prefix}LoadPulseDocumentButton")
        self._browse_pulse = QtWidgets.QPushButton("Browse…", self)
        self._browse_pulse.setObjectName(f"{object_prefix}BrowsePulseDocumentButton")
        pulse_row = QtWidgets.QHBoxLayout()
        pulse_row.addWidget(self._pulse_path, 1)
        pulse_row.addWidget(self._load_pulse)
        pulse_row.addWidget(self._browse_pulse)
        self._fingerprint = QtWidgets.QLabel("Fingerprint: —", self)
        self._fingerprint.setObjectName(f"{object_prefix}PulseFingerprint")
        self._fingerprint.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._scan_summary = QtWidgets.QLabel("SCAN_SLOT: no document", self)
        self._scan_summary.setObjectName(f"{object_prefix}ScanTableSummary")
        self._scan_summary.setWordWrap(True)

        pulse_box = QtWidgets.QGroupBox("PulseDocument · SCAN_SLOT", self)
        pulse_layout = QtWidgets.QVBoxLayout(pulse_box)
        pulse_layout.addLayout(pulse_row)
        pulse_layout.addWidget(self._fingerprint)
        pulse_layout.addWidget(self._scan_summary)

        self._api_box = QtWidgets.QGroupBox("Whole-run API constants", self)
        self._api_layout = QtWidgets.QFormLayout(self._api_box)
        self._api_empty = QtWidgets.QLabel("Load a PulseDocument", self._api_box)
        self._api_empty.setObjectName(f"{object_prefix}ApiConstantsState")
        self._api_layout.addRow(self._api_empty)

        self._authority_summary = QtWidgets.QLabel(self)
        self._authority_summary.setObjectName(
            f"{object_prefix}AuthoritativeTransformSummary"
        )
        self._authority_summary.setWordWrap(True)
        self._authority_summary.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self._clear_authority = QtWidgets.QPushButton(
            "Clear user transform",
            self,
        )
        self._clear_authority.setObjectName(
            f"{object_prefix}ClearAuthoritativeTransformButton"
        )
        authority_box = QtWidgets.QGroupBox(
            "Authoritative output transform",
            self,
        )
        authority_layout = QtWidgets.QVBoxLayout(authority_box)
        authority_layout.addWidget(self._authority_summary)
        authority_layout.addWidget(self._clear_authority)

        self._source = QtWidgets.QComboBox(self)
        self._source.setObjectName(f"{object_prefix}ScanSource")
        self._source.addItem("Direct camera", _DIRECT_SOURCE)
        self._source.addItem("Camera → Occupancy counts", _OCCUPANCY_SOURCE)
        self._calibration_repository = QtWidgets.QLineEdit(self)
        self._calibration_repository.setObjectName(
            f"{object_prefix}CalibrationRepositoryId"
        )
        self._calibration_repository.setPlaceholderText("repository_id")
        self._calibration_digest = QtWidgets.QLineEdit(self)
        self._calibration_digest.setObjectName(
            f"{object_prefix}CalibrationManifestDigest"
        )
        self._calibration_digest.setPlaceholderText("64-character manifest digest")
        self._model_kind = QtWidgets.QComboBox(self)
        self._model_kind.setObjectName(f"{object_prefix}ReadoutModelKind")
        self._model_kind.addItem("Calibration default", None)
        for kind in ReadoutModelKind:
            self._model_kind.addItem(kind.value, kind)
        source_box = QtWidgets.QGroupBox("Source", self)
        source_layout = QtWidgets.QFormLayout(source_box)
        source_layout.addRow("Pipeline", self._source)
        source_layout.addRow("Calibration repository", self._calibration_repository)
        source_layout.addRow("Calibration digest", self._calibration_digest)
        source_layout.addRow("Readout model", self._model_kind)

        self._camera_role = QtWidgets.QLineEdit("camera", self)
        self._camera_role.setObjectName(f"{object_prefix}CameraRole")
        self._sequencer_role = QtWidgets.QLineEdit("sequencer", self)
        self._sequencer_role.setObjectName(f"{object_prefix}SequencerRole")
        self._trigger_channel = QtWidgets.QLineEdit(self)
        self._trigger_channel.setObjectName(f"{object_prefix}TriggerChannel")
        self._trigger_channel.setPlaceholderText("optional")
        binding_box = QtWidgets.QGroupBox("Device binding", self)
        binding_layout = QtWidgets.QFormLayout(binding_box)
        binding_layout.addRow("Camera role", self._camera_role)
        binding_layout.addRow("Sequencer role", self._sequencer_role)
        binding_layout.addRow("Trigger channel", self._trigger_channel)

        self._transport_budget = QtWidgets.QLineEdit(str(64 << 20), self)
        self._transport_budget.setObjectName(f"{object_prefix}TransportBudgetBytes")
        self._memory_budget = QtWidgets.QLineEdit(str(512 << 20), self)
        self._memory_budget.setObjectName(f"{object_prefix}MemoryBudgetBytes")
        self._deadline = QtWidgets.QLineEdit("30.0", self)
        self._deadline.setObjectName(f"{object_prefix}DeadlineSeconds")
        budget_box = QtWidgets.QGroupBox("Budgets and deadline", self)
        budget_layout = QtWidgets.QFormLayout(budget_box)
        budget_layout.addRow("Transport bytes", self._transport_budget)
        budget_layout.addRow("Pipeline bytes", self._memory_budget)
        budget_layout.addRow("Timeout seconds", self._deadline)

        self._site_mode = QtWidgets.QComboBox(self)
        self._site_mode.setObjectName(f"{object_prefix}SiteDisplayMode")
        self._site_mode.addItem("Auto", "auto")
        self._site_mode.addItem("Batch", "batch")
        self._site_mode.addItem("Select", "select")
        self._site_index = QtWidgets.QSpinBox(self)
        self._site_index.setObjectName(f"{object_prefix}SiteDisplayIndex")
        self._site_index.setRange(0, 2_147_483_647)
        display_box = QtWidgets.QGroupBox("Display only · SITE", self)
        display_layout = QtWidgets.QFormLayout(display_box)
        display_layout.addRow("Mode", self._site_mode)
        display_layout.addRow("Index", self._site_index)

        self._diagnostics = QtWidgets.QLabel("", self)
        self._diagnostics.setObjectName(f"{object_prefix}ScanEditorDiagnostics")
        self._diagnostics.setWordWrap(True)
        self._diagnostics.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._apply = QtWidgets.QPushButton("Apply", self)
        self._apply.setObjectName(f"{object_prefix}ApplyScanIntentButton")
        self._cancel = QtWidgets.QPushButton("Cancel", self)
        self._cancel.setObjectName(f"{object_prefix}CancelScanIntentButton")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._cancel)
        buttons.addWidget(self._apply)

        columns = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        left.addWidget(source_box)
        left.addWidget(binding_box)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(budget_box)
        right.addWidget(display_box)
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(pulse_box)
        layout.addWidget(self._api_box)
        layout.addWidget(authority_box)
        layout.addLayout(columns)
        layout.addWidget(self._diagnostics)
        layout.addLayout(buttons)
        layout.addStretch(1)

        self._browse_pulse.clicked.connect(self._browse_for_pulse)
        self._load_pulse.clicked.connect(self._load_path_from_edit)
        self._source.currentIndexChanged.connect(self._update_enabled_state)
        self._site_mode.currentIndexChanged.connect(self._update_enabled_state)
        self._clear_authority.clicked.connect(self._clear_authoritative_transform)
        self._apply.clicked.connect(self._apply_clicked)
        self._cancel.clicked.connect(self.cancel_edit)
        self._update_enabled_state()
        self._refresh_authority_summary()

    @property
    def editor_session(self) -> ScanEditorSession | None:
        """Both form instances expose the same card-owned edit session."""

        return self._card.editor_session

    @property
    def base_revision(self) -> int | None:
        return self._base_revision

    def begin_edit(self) -> None:
        session = self.editor_session
        if session is None:
            return
        draft = session.begin()
        self._base_revision = draft.base_revision
        self._populate(draft.intent)
        self._diagnostics.clear()

    def accept_revision(self, revision: int) -> None:
        session = self.editor_session
        if session is None:
            raise RuntimeError("applied revision has no current intent")
        snapshot = session.snapshot()
        if snapshot.revision != revision:
            raise RuntimeError("applied revision differs from the current edit session")
        self._populate(snapshot.intent)
        self._base_revision = revision
        self._diagnostics.setText(f"Applied revision {revision}")
        self.applied.emit()

    def cancel_edit(self) -> None:
        session = self.editor_session
        if session is None or self._base_revision is None:
            self._clear_unapplied()
            return
        current = session.snapshot()
        snapshot = session.cancel(ScanEditDraft(self._base_revision, current.intent))
        self._base_revision = snapshot.revision
        self._populate(snapshot.intent)
        self._diagnostics.setText("Unapplied edits discarded")

    def show_error(self, error: BaseException) -> None:
        self._diagnostics.setText(f"{type(error).__name__}: {error}")

    def load_pulse_path(self, path: str | Path) -> None:
        """Load through the PulseDocument owner's strict current loader."""

        document = load_pulse_document(path)
        # Reject non-SCAN_SLOT documents before disturbing the current draft.
        if document.scan_table is None or not document.scan_parameters:
            raise ValueError("PulseDocument has no frozen SCAN_SLOT table")
        self._document = document
        self._document_source = str(Path(path).expanduser().resolve())
        self._pulse_path.setText(self._document_source)
        self._output_transform_spec = None
        self._refresh_authority_summary()
        self._show_document(document, {})
        self._diagnostics.clear()

    def build_intent(self) -> TaskConsoleScanIntent:
        document = self._document
        if document is None:
            raise ValueError("a current PulseDocument JSON must be loaded")
        api_values = tuple(
            (
                parameter.parameter_id,
                _parse_number(
                    self._api_edits[parameter.parameter_id].text(),
                    f"API constant {parameter.parameter_id!r}",
                ),
            )
            for parameter in document.api_parameters
        )
        occupancy = self._source.currentData() == _OCCUPANCY_SOURCE
        calibration_ref = None
        model_kind = None
        display = ScanDisplayIntent()
        if occupancy:
            calibration_ref = CalibrationArtifactRef(
                self._calibration_repository.text().strip(),
                self._calibration_digest.text().strip(),
            )
            model_kind = self._model_kind.currentData()
            display = ScanDisplayIntent(
                self._site_mode.currentData(),
                self._site_index.value()
                if self._site_mode.currentData() == "select"
                else 0,
            )
        return TaskConsoleScanIntent(
            task_key=AUTONOMOUS_SCAN_SLOT_TASK_KEY,
            measurement_key=CAMERA_MEASUREMENT_KEY,
            processor_key=(OCCUPANCY_STREAM_PROCESSOR_KEY if occupancy else None),
            pulse_document=document,
            api_values=api_values,
            camera_role=self._camera_role.text().strip(),
            sequencer_role=self._sequencer_role.text().strip(),
            trigger_channel=self._trigger_channel.text().strip() or None,
            calibration_ref=calibration_ref,
            model_kind=model_kind,
            output_transform_spec=self._output_transform_spec,
            display_intent=display,
            transport_memory_limit_bytes=_parse_positive_integer(
                self._transport_budget.text(),
                "transport memory budget",
            ),
            memory_limit_bytes=_parse_positive_integer(
                self._memory_budget.text(),
                "pipeline memory budget",
            ),
            timeout_seconds=_parse_positive_real(
                self._deadline.text(),
                "timeout",
            ),
        )

    def _populate(self, intent: TaskConsoleScanIntent) -> None:
        previous_document = self._document
        self._document = intent.pulse_document
        if (
            self._document_source is None
            or previous_document is None
            or previous_document.fingerprint != intent.pulse_document.fingerprint
        ):
            self._document_source = None
        self._pulse_path.setText(self._document_source or _EMBEDDED_DOCUMENT)
        self._output_transform_spec = intent.output_transform_spec
        self._refresh_authority_summary()
        self._show_document(intent.pulse_document, intent.fixed_api_values)
        occupancy = intent.processor_key is not None
        self._source.setCurrentIndex(1 if occupancy else 0)
        self._camera_role.setText(intent.camera_role)
        self._sequencer_role.setText(intent.sequencer_role)
        self._trigger_channel.setText(intent.trigger_channel or "")
        self._transport_budget.setText(str(intent.transport_memory_limit_bytes))
        self._memory_budget.setText(str(intent.memory_limit_bytes))
        self._deadline.setText(str(intent.timeout_seconds))
        if occupancy:
            assert intent.calibration_ref is not None
            self._calibration_repository.setText(intent.calibration_ref.repository_id)
            self._calibration_digest.setText(intent.calibration_ref.manifest_digest)
            index = self._model_kind.findData(intent.model_kind)
            self._model_kind.setCurrentIndex(max(0, index))
            site_index = self._site_mode.findData(intent.display_intent.site_mode)
            self._site_mode.setCurrentIndex(max(0, site_index))
            self._site_index.setValue(intent.display_intent.site_index)
        else:
            self._calibration_repository.clear()
            self._calibration_digest.clear()
            self._model_kind.setCurrentIndex(0)
            self._site_mode.setCurrentIndex(0)
            self._site_index.setValue(0)
        self._update_enabled_state()

    def _show_document(
        self,
        document: PulseDocument,
        api_values: dict[str, int | float],
    ) -> None:
        table = document.scan_table
        assert table is not None
        self._fingerprint.setText(f"Fingerprint: {document.fingerprint}")
        self._scan_summary.setText(
            "SCAN_SLOT: columns="
            + ", ".join(table.columns)
            + f" · rows={len(table.rows)}"
        )
        self._clear_api_rows()
        if not document.api_parameters:
            self._api_empty = QtWidgets.QLabel(
                "No whole-run API constants declared",
                self._api_box,
            )
            self._api_empty.setObjectName(f"{self._prefix}ApiConstantsState")
            self._api_layout.addRow(self._api_empty)
            return
        for parameter in document.api_parameters:
            edit = QtWidgets.QLineEdit(self._api_box)
            edit.setObjectName(
                f"{self._prefix}ApiValue_{parameter.parameter_id}"
            )
            if parameter.parameter_id in api_values:
                edit.setText(str(api_values[parameter.parameter_id]))
            self._api_edits[parameter.parameter_id] = edit
            self._api_layout.addRow(
                f"{parameter.parameter_id} ({parameter.unit})",
                edit,
            )

    def _clear_api_rows(self) -> None:
        while self._api_layout.count():
            item = self._api_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._api_edits.clear()

    def _clear_unapplied(self) -> None:
        self._document = None
        self._document_source = None
        self._output_transform_spec = None
        self._refresh_authority_summary()
        self._base_revision = None
        self._pulse_path.clear()
        self._fingerprint.setText("Fingerprint: —")
        self._scan_summary.setText("SCAN_SLOT: no document")
        self._clear_api_rows()
        self._api_empty = QtWidgets.QLabel("Load a PulseDocument", self._api_box)
        self._api_layout.addRow(self._api_empty)
        self._source.setCurrentIndex(0)
        self._calibration_repository.clear()
        self._calibration_digest.clear()
        self._model_kind.setCurrentIndex(0)
        self._camera_role.setText("camera")
        self._sequencer_role.setText("sequencer")
        self._trigger_channel.clear()
        self._transport_budget.setText(str(64 << 20))
        self._memory_budget.setText(str(512 << 20))
        self._deadline.setText("30.0")
        self._site_mode.setCurrentIndex(0)
        self._site_index.setValue(0)
        self._update_enabled_state()
        self._diagnostics.setText("Unapplied draft cleared")

    def _clear_authoritative_transform(self) -> None:
        self._output_transform_spec = None
        self._refresh_authority_summary()
        self._diagnostics.setText(
            "User-authored authoritative Select/Reduce cleared"
        )

    def _refresh_authority_summary(self) -> None:
        self._authority_summary.setText(
            describe_authoritative_transform(self._output_transform_spec)
        )
        self._clear_authority.setEnabled(self._output_transform_spec is not None)

    def _browse_for_pulse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select current PulseDocument",
            "",
            "PulseDocument JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.load_pulse_path(path)
        except BaseException as error:
            self.show_error(error)

    def _load_path_from_edit(self) -> None:
        path = self._pulse_path.text().strip()
        if not path or path == _EMBEDDED_DOCUMENT:
            self.show_error(ValueError("choose a PulseDocument JSON path"))
            return
        try:
            self.load_pulse_path(path)
        except BaseException as error:
            self.show_error(error)

    def _apply_clicked(self) -> None:
        try:
            revision = self._card.apply_form(self)
        except BaseException as error:
            self.show_error(error)
            return
        self.accept_revision(revision)

    def _update_enabled_state(self) -> None:
        occupancy = self._source.currentData() == _OCCUPANCY_SOURCE
        for widget in (
            self._calibration_repository,
            self._calibration_digest,
            self._model_kind,
            self._site_mode,
        ):
            widget.setEnabled(occupancy)
        selecting = occupancy and self._site_mode.currentData() == "select"
        self._site_index.setEnabled(selecting)
        if not selecting:
            self._site_index.setValue(0)


class TaskScanCard(QtWidgets.QWidget):
    """One stopped/configurable card around the existing scan panel."""

    def __init__(
        self,
        experiment: Experiment,
        initial_intent: TaskConsoleScanIntent | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("taskConsoleScanCard")
        self._experiment = experiment
        self._session: ScanEditorSession | None = None
        self._panel: ScanWorkbenchWindow | None = None

        title = QtWidgets.QLabel("Task: Autonomous SCAN_SLOT", self)
        title.setObjectName("taskCardTitle")
        self._state = QtWidgets.QLabel("STOPPED · CONFIGURATION REQUIRED", self)
        self._state.setObjectName("taskCardState")
        self._settings = QtWidgets.QPushButton("Setting…", self)
        self._settings.setObjectName("taskSettingsButton")
        header = QtWidgets.QHBoxLayout()
        header.addWidget(title)
        header.addWidget(self._state)
        header.addStretch(1)
        header.addWidget(self._settings)

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setObjectName("taskCardTabs")
        self._run_page = QtWidgets.QWidget(self._tabs)
        self._run_page.setObjectName("taskRunTab")
        self._run_layout = QtWidgets.QVBoxLayout(self._run_page)
        self._placeholder = QtWidgets.QLabel(
            "Stopped. Configure this card in Edit or Setting.",
            self._run_page,
        )
        self._placeholder.setObjectName("taskPanelPlaceholder")
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._run_layout.addWidget(self._placeholder, 1)
        self._edit_form = ScanIntentForm(self, object_prefix="edit")
        self._tabs.addTab(self._run_page, "Run")
        self._tabs.addTab(self._edit_form, "Edit")

        self._settings_dialog = QtWidgets.QDialog(self)
        self._settings_dialog.setObjectName("taskSettingsDialog")
        self._settings_dialog.setWindowTitle("Autonomous SCAN_SLOT Setting")
        settings_layout = QtWidgets.QVBoxLayout(self._settings_dialog)
        self._settings_form = ScanIntentForm(self, object_prefix="setting")
        settings_layout.addWidget(self._settings_form)
        self._settings_form.applied.connect(self._settings_dialog.accept)

        self._diagnostics = QtWidgets.QLabel("", self)
        self._diagnostics.setObjectName("taskCardDiagnostics")
        self._diagnostics.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._diagnostics)

        self._settings.clicked.connect(self._open_settings)
        self._tabs.currentChanged.connect(self._tab_changed)
        self._state_timer = QtCore.QTimer(self)
        self._state_timer.setInterval(100)
        self._state_timer.timeout.connect(self._refresh_state)
        self._state_timer.start()

        if initial_intent is not None:
            self.load_intent(initial_intent)
        else:
            self._tabs.setCurrentWidget(self._edit_form)

    @property
    def editor_session(self) -> ScanEditorSession | None:
        return self._session

    @property
    def current_intent(self) -> TaskConsoleScanIntent | None:
        return None if self._session is None else self._session.snapshot().intent

    @property
    def panel(self) -> ScanWorkbenchWindow | None:
        return self._panel

    @property
    def idle(self) -> bool:
        return self._panel is None or self._panel.can_reconfigure

    def apply_form(self, form: ScanIntentForm) -> int:
        """Validate a complete replacement before touching applied state."""

        if form not in (self._edit_form, self._settings_form):
            raise ValueError("form does not belong to this card")
        intent = form.build_intent()
        request = _intent_request(self._experiment, intent)

        if self._session is None:
            session = ScanEditorSession(intent)
            draft = session.begin()
            panel = ScanWorkbenchWindow(
                self._experiment,
                request,
                display_intent=intent.display_intent,
            )
            snapshot = session.apply(draft)
            self._session = session
            self._install_panel(panel)
        else:
            if form.base_revision is None:
                raise ScanEditConflict("editor has no current base revision")
            draft = ScanEditDraft(form.base_revision, intent)
            if self._session.snapshot().revision != draft.base_revision:
                raise ScanEditConflict(
                    f"edit base revision {draft.base_revision} is stale; "
                    f"current revision is {self._session.snapshot().revision}"
                )
            panel = self._panel
            if panel is None:
                raise RuntimeError("applied edit session has no scan panel")
            if not panel.can_reconfigure:
                raise RuntimeError("scan panel must be stopped and idle before Apply")
            # ScanPanelController.reconfigure owns the atomic idle replacement;
            # the already-checked session apply cannot then conflict on this GUI
            # owner thread.
            panel.reconfigure(request, display_intent=intent.display_intent)
            snapshot = self._session.apply(draft)

        self._diagnostics.clear()
        self._state.setText("STOPPED")
        return snapshot.revision

    def load_intent(self, intent: TaskConsoleScanIntent) -> None:
        """Install one strict current intent without starting a Run."""

        if not isinstance(intent, TaskConsoleScanIntent):
            raise TypeError("intent must be TaskConsoleScanIntent")
        request = _intent_request(self._experiment, intent)
        if self._panel is None:
            panel = ScanWorkbenchWindow(
                self._experiment,
                request,
                display_intent=intent.display_intent,
            )
            self._install_panel(panel)
        else:
            if not self._panel.can_reconfigure:
                raise RuntimeError("scan panel must be stopped and idle before Load")
            self._panel.reconfigure(request, display_intent=intent.display_intent)
        self._session = ScanEditorSession(intent)
        self._edit_form.begin_edit()
        self._settings_form.begin_edit()
        self._diagnostics.clear()
        self._state.setText("STOPPED")

    def shutdown(self) -> None:
        self._state_timer.stop()
        if self._panel is not None:
            self._panel.shutdown()

    def _install_panel(self, panel: ScanWorkbenchWindow) -> None:
        if self._panel is not None:
            raise RuntimeError("scan card already owns a panel")
        self._panel = panel
        panel.setObjectName("embeddedScanWorkbench")
        self._run_layout.removeWidget(self._placeholder)
        self._placeholder.hide()
        self._run_layout.addWidget(panel, 1)
        panel.show()

    def _open_settings(self) -> None:
        self._settings_form.begin_edit()
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._edit_form:
            self._edit_form.begin_edit()

    def _refresh_state(self) -> None:
        if self._panel is None:
            self._state.setText("STOPPED · CONFIGURATION REQUIRED")
        elif self._panel.closed:
            self._state.setText("CLOSED")
        elif self._panel.can_reconfigure:
            self._state.setText("STOPPED")
        else:
            self._state.setText("ACTIVE")


class TaskConsoleWindow(QtWidgets.QWidget):
    """Current one-card TaskConsole; no legacy registry or generic graph."""

    def __init__(
        self,
        experiment: Experiment,
        initial_intent: TaskConsoleScanIntent | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be Experiment")
        if initial_intent is not None and not isinstance(
            initial_intent,
            TaskConsoleScanIntent,
        ):
            raise TypeError("initial_intent must be TaskConsoleScanIntent or None")
        self.setObjectName("taskConsoleWindow")
        self.setWindowTitle("Task Console")
        self.resize(1120, 820)
        self._experiment = experiment
        self._card: TaskScanCard | None = None
        self._closing = False

        self._catalog = QtWidgets.QComboBox(self)
        self._catalog.setObjectName("taskCatalogCombo")
        items = task_console_catalog_items(compose_task_console_catalog())
        task_items = tuple(item for item in items if item.group == "Task")
        if len(task_items) != 1:
            raise RuntimeError("TaskConsole requires exactly one current Task")
        for item in task_items:
            self._catalog.addItem(f"{item.group}: {item.title}", item.key)
        self._add = QtWidgets.QPushButton("Add Panel", self)
        self._add.setObjectName("addTaskPanelButton")
        self._save = QtWidgets.QPushButton("Save", self)
        self._save.setObjectName("saveTaskIntentButton")
        self._load = QtWidgets.QPushButton("Load", self)
        self._load.setObjectName("loadTaskIntentButton")
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._catalog, 1)
        controls.addWidget(self._add)
        controls.addStretch(1)
        controls.addWidget(self._save)
        controls.addWidget(self._load)

        self._card_host = QtWidgets.QWidget(self)
        self._card_host.setObjectName("taskCardHost")
        self._card_layout = QtWidgets.QVBoxLayout(self._card_host)
        self._empty = QtWidgets.QLabel("Add the Autonomous SCAN_SLOT task", self._card_host)
        self._empty.setObjectName("taskConsoleEmptyState")
        self._empty.setAlignment(QtCore.Qt.AlignCenter)
        self._card_layout.addWidget(self._empty, 1)
        self._status = QtWidgets.QLabel("", self)
        self._status.setObjectName("taskConsoleDiagnostics")
        self._status.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._card_host, 1)
        layout.addWidget(self._status)

        self._add.clicked.connect(self._add_card)
        self._save.clicked.connect(self._save_dialog)
        self._load.clicked.connect(self._load_dialog)
        self._close_timer = QtCore.QTimer(self)
        self._close_timer.setInterval(40)
        self._close_timer.timeout.connect(self._poll_close)

        if initial_intent is not None:
            self._create_card(initial_intent)

    @property
    def scan_card(self) -> TaskScanCard | None:
        return self._card

    @property
    def current_intent(self) -> TaskConsoleScanIntent | None:
        return None if self._card is None else self._card.current_intent

    def save_intent(self, path: str | Path) -> Path:
        intent = self.current_intent
        if intent is None:
            raise RuntimeError("there is no applied TaskConsole intent to save")
        destination = save_task_console_scan_intent(intent, path)
        self._status.setText(f"Saved current intent: {destination}")
        return destination

    def load_intent(self, path: str | Path) -> TaskConsoleScanIntent:
        if self._card is not None and not self._card.idle:
            raise RuntimeError("scan panel must be stopped and idle before Load")
        intent = load_task_console_scan_intent(path)
        if self._card is None:
            self._create_card(intent)
        else:
            self._card.load_intent(intent)
        self._status.setText(f"Loaded current intent stopped: {Path(path)}")
        return intent

    def _add_card(self) -> None:
        if self._catalog.currentData() != AUTONOMOUS_SCAN_SLOT_TASK_KEY:
            self._status.setText("Only Autonomous SCAN_SLOT is available")
            return
        self._create_card(None)

    def _create_card(self, intent: TaskConsoleScanIntent | None) -> None:
        if self._card is not None:
            raise RuntimeError("TaskConsole currently owns exactly one card")
        card = TaskScanCard(self._experiment, intent, self._card_host)
        self._card = card
        self._card_layout.removeWidget(self._empty)
        self._empty.hide()
        self._card_layout.addWidget(card, 1)
        self._add.setEnabled(False)
        self._catalog.setEnabled(False)
        self._status.clear()

    def _save_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save current TaskConsole intent",
            "task-console.json",
            "TaskConsole intent (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.save_intent(path)
        except BaseException as error:
            self._show_error(error)

    def _load_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load current TaskConsole intent",
            "",
            "TaskConsole intent (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.load_intent(path)
        except BaseException as error:
            self._show_error(error)

    def _show_error(self, error: BaseException) -> None:
        self._status.setText(f"{type(error).__name__}: {error}")

    def closeEvent(self, event) -> None:
        panel = None if self._card is None else self._card.panel
        if not self._closing:
            self._closing = True
            if self._card is not None:
                self._card.shutdown()
        if panel is None or panel.closed:
            self._close_timer.stop()
            event.accept()
            return
        event.ignore()
        if not self._close_timer.isActive():
            self.setEnabled(False)
            self._status.setText("Closing embedded scan panel…")
            self._close_timer.start()

    def _poll_close(self) -> None:
        panel = None if self._card is None else self._card.panel
        if panel is None or panel.closed:
            self._close_timer.stop()
            QtCore.QTimer.singleShot(0, self.close)


def open_task_console(
    experiment: Experiment,
    initial_intent: TaskConsoleScanIntent | None = None,
) -> TaskConsoleWindow:
    """Lazily open the current TaskConsole on the Qt owner thread."""

    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication([])
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("TaskConsole must be opened on the Qt GUI thread")
    window = TaskConsoleWindow(experiment, initial_intent)
    if owns_application:
        window._application_owner = application
    window.show()
    return window


__all__ = [
    "ScanIntentForm",
    "TaskConsoleWindow",
    "TaskScanCard",
    "open_task_console",
]
