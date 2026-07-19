"""Qt composition for the current single-card pulse-scan TaskConsole.

The domain values, edit revision, catalog projection, and persistence codec live
in :mod:`zlc_workbench.task_console`.  This module only owns Qt widgets and the
composition seam from one validated intent to the existing scan panel.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from zlc_frontend import FormFieldProps, FormSpec, parse_number_text
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentCheckBox,
    FluentComboBox,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentParameterForm,
    FluentScrollArea,
    FluentSettingRow,
    FluentSpinBox,
    FluentTabWidget,
    GREEN,
    GREY,
    ORANGE,
    WINDOW_SCREEN_FRACTION,
    apply_fluent_scrollbars,
    center_window_on_primary_screen,
    ensure_qt_app,
    release_window,
    retain_window,
    screen_fit_window_size,
    set_fluent_scale,
    setting_label_width,
)

from Zou_lab_control.notebook.facade import Experiment
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
from zlc_neutral_atom.readout.calibration import ReadoutModelKind
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.occupancy import OCCUPANCY_STREAM_PROCESSOR_KEY
from zlc_neutral_atom.scan.contracts import (
    PULSE_SCAN_TASK_KEY,
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)
from zlc_pulse import FIELD_DAC, PulseDocument, load_pulse_document
from zlc_frontend import describe_authoritative_transform
from zlc_workbench.task_console import (
    ScanDisplayIntent,
    ScanEditConflict,
    ScanEditDraft,
    ScanEditorSession,
    TaskConsoleScanIntent,
    compose_task_console_catalog,
    load_task_console_scan_intent,
    save_task_console_scan_intent,
    task_console_catalog_items,
    task_console_scan_binding_form_spec,
    task_console_scan_budget_form_spec,
)

from ._scan import ScanWorkbenchWindow


_DIRECT_SOURCE = "direct"
_OCCUPANCY_SOURCE = "occupancy"
_SCAN_SLOT_MODE = "scan-slot"
_API_SLOT_MODE = "api-slot-segmented"
_EMBEDDED_DOCUMENT = "(embedded PulseDocument)"


def _settings_card(
    title: str,
    rows: tuple[tuple[str, QtWidgets.QWidget], ...],
    parent: QtWidgets.QWidget,
) -> FluentGroupBox:
    """Compose one card from the shared setting-row design system."""

    box = FluentGroupBox(title, parent)
    layout = QtWidgets.QVBoxLayout(box)
    label_width = setting_label_width(label for label, _control in rows)
    for label, control in rows:
        layout.addWidget(
            FluentSettingRow(label, control, label_width=label_width, parent=box)
        )
    return box


def _form_card(
    title: str,
    form: FluentParameterForm,
    parent: QtWidgets.QWidget,
) -> FluentGroupBox:
    box = FluentGroupBox(title, parent)
    layout = QtWidgets.QVBoxLayout(box)
    layout.addWidget(form)
    return box


def _api_constant_form_spec(document: PulseDocument) -> FormSpec:
    """Project declared API leaves; PulseDocument remains the value authority."""

    fields = []
    for parameter in document.api_parameters:
        value, _unit = document.field_value(parameter.field)
        minimum = None
        maximum = None
        kind = "number"
        if parameter.field.kind == FIELD_DAC:
            port = document.target.by_key[parameter.field.port]
            assert port.signed_range is not None
            minimum, maximum = port.signed_range
            kind = "int"
        fields.append(
            FormFieldProps(
                parameter.parameter_id,
                kind,
                parameter.parameter_id,
                default=value,
                required=True,
                unit=parameter.unit,
                minimum=minimum,
                maximum=maximum,
                description="Whole-run PulseDocument API constant",
            )
        )
    return FormSpec(tuple(fields))


def _intent_request(experiment: Experiment, intent: TaskConsoleScanIntent):
    common = dict(
        camera_role=intent.camera_role,
        sequencer_role=intent.sequencer_role,
        trigger_channel=intent.trigger_channel,
        output_transform_spec=intent.output_transform_spec,
        transport_memory_limit_bytes=intent.transport_memory_limit_bytes,
        memory_limit_bytes=intent.memory_limit_bytes,
        timeout_seconds=intent.timeout_seconds,
    )
    program = intent.program
    if isinstance(program, AutonomousScanSlotProgram):
        common["api_values"] = dict(program.api_values)
        if intent.processor_key is None:
            return experiment.readout.scan_request(program.document, **common)
        assert intent.calibration_ref is not None
        return experiment.readout.occupancy_scan_request(
            program.document,
            calibration_ref=intent.calibration_ref,
            model_kind=intent.model_kind,
            **common,
        )
    if not isinstance(program, ApiSlotSegmentedProgram):
        raise TypeError("TaskConsole intent carries another scan program")
    common.update(
        api_table=program.table,
        segmentation_rationale=program.segmentation_rationale,
    )
    if intent.processor_key is None:
        return experiment.readout.api_scan_request(program.document, **common)
    assert intent.calibration_ref is not None
    return experiment.readout.api_occupancy_scan_request(
        program.document,
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
        self._api_form: FluentParameterForm | None = None
        self._output_transform_spec = None
        self.setObjectName(f"{object_prefix}ScanIntentForm")

        self._pulse_path = FluentPathEdit(
            mode="file",
            caption="Select current PulseDocument",
            file_filter="PulseDocument JSON (*.json);;All files (*)",
            parent=self,
        )
        self._pulse_path.setObjectName(f"{object_prefix}PulseDocumentPathField")
        self._pulse_path.edit.setObjectName(f"{object_prefix}PulseDocumentPath")
        self._pulse_path.setPlaceholderText("Current PulseDocument JSON")
        self._load_pulse = FluentButton("Load JSON", self, color=ORANGE)
        self._load_pulse.setObjectName(f"{object_prefix}LoadPulseDocumentButton")
        self._browse_pulse = self._pulse_path.browse
        self._browse_pulse.setObjectName(f"{object_prefix}BrowsePulseDocumentButton")
        pulse_row = QtWidgets.QHBoxLayout()
        pulse_row.addWidget(self._pulse_path, 1)
        pulse_row.addWidget(self._load_pulse)
        self._fingerprint = FluentLabel("Fingerprint: —", self)
        self._fingerprint.setObjectName(f"{object_prefix}PulseFingerprint")
        self._fingerprint.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._slot_mode = FluentComboBox(self)
        self._slot_mode.setObjectName(f"{object_prefix}PulseScanSlotMode")
        self._scan_summary = FluentLabel("Pulse scan: no document", self)
        self._scan_summary.setObjectName(f"{object_prefix}ScanTableSummary")
        self._scan_summary.setWordWrap(True)

        pulse_box = FluentGroupBox("PulseDocument · scan program", self)
        pulse_layout = QtWidgets.QVBoxLayout(pulse_box)
        pulse_layout.addLayout(pulse_row)
        pulse_layout.addWidget(self._fingerprint)
        pulse_layout.addWidget(
            FluentSettingRow(
                "Execution",
                self._slot_mode,
                label_width=setting_label_width(("Execution",)),
                parent=pulse_box,
            )
        )
        pulse_layout.addWidget(self._scan_summary)

        self._api_box = FluentGroupBox("Slot values", self)
        api_box_layout = QtWidgets.QVBoxLayout(self._api_box)
        self._slot_stack = QtWidgets.QStackedWidget(self._api_box)
        api_box_layout.addWidget(self._slot_stack)

        self._fixed_api_page = QtWidgets.QWidget(self._slot_stack)
        self._api_layout = QtWidgets.QVBoxLayout(self._fixed_api_page)
        self._api_empty = FluentLabel("Load a PulseDocument", self._fixed_api_page)
        self._api_empty.setObjectName(f"{object_prefix}ApiConstantsState")
        self._api_layout.addWidget(self._api_empty)
        self._slot_stack.addWidget(self._fixed_api_page)

        self._segmented_api_page = QtWidgets.QWidget(self._slot_stack)
        segmented_layout = QtWidgets.QVBoxLayout(self._segmented_api_page)
        self._api_table = QtWidgets.QTableWidget(0, 0, self._segmented_api_page)
        self._api_table.setObjectName(f"{object_prefix}ApiSegmentTable")
        self._api_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._api_table.horizontalHeader().setStretchLastSection(True)
        apply_fluent_scrollbars(self._api_table)
        segmented_layout.addWidget(self._api_table)
        row_buttons = QtWidgets.QHBoxLayout()
        self._add_api_row = FluentButton("Add row", self._segmented_api_page)
        self._add_api_row.setObjectName(f"{object_prefix}AddApiSegmentRow")
        self._remove_api_row = FluentButton(
            "Remove row",
            self._segmented_api_page,
            color=GREY,
        )
        self._remove_api_row.setObjectName(f"{object_prefix}RemoveApiSegmentRow")
        row_buttons.addWidget(self._add_api_row)
        row_buttons.addWidget(self._remove_api_row)
        row_buttons.addStretch(1)
        segmented_layout.addLayout(row_buttons)
        self._allow_host_gaps = FluentCheckBox(
            "Each API point is independent; variable host gaps are allowed",
            self._segmented_api_page,
        )
        self._allow_host_gaps.setObjectName(f"{object_prefix}AllowApiHostGaps")
        self._segmentation_rationale = FluentLineEdit(parent=self._segmented_api_page)
        self._segmentation_rationale.setObjectName(
            f"{object_prefix}ApiSegmentationRationale"
        )
        self._segmentation_rationale.setPlaceholderText(
            "Why independent finite segments preserve this experiment"
        )
        segmented_layout.addWidget(self._allow_host_gaps)
        segmented_layout.addWidget(
            FluentSettingRow(
                "Rationale",
                self._segmentation_rationale,
                label_width=setting_label_width(("Rationale",)),
                parent=self._segmented_api_page,
            )
        )
        self._slot_stack.addWidget(self._segmented_api_page)

        self._authority_summary = FluentLabel(parent=self)
        self._authority_summary.setObjectName(
            f"{object_prefix}AuthoritativeTransformSummary"
        )
        self._authority_summary.setWordWrap(True)
        self._authority_summary.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self._clear_authority = FluentButton(
            "Clear user transform",
            self,
            color=GREY,
        )
        self._clear_authority.setObjectName(
            f"{object_prefix}ClearAuthoritativeTransformButton"
        )
        authority_box = FluentGroupBox(
            "Authoritative output transform",
            self,
        )
        authority_layout = QtWidgets.QVBoxLayout(authority_box)
        authority_layout.addWidget(self._authority_summary)
        authority_layout.addWidget(self._clear_authority)

        self._source = FluentComboBox(self)
        self._source.setObjectName(f"{object_prefix}ScanSource")
        self._source.addItem("Direct camera", _DIRECT_SOURCE)
        self._source.addItem("Camera → Occupancy counts", _OCCUPANCY_SOURCE)
        self._calibration_repository = FluentLineEdit(parent=self)
        self._calibration_repository.setObjectName(
            f"{object_prefix}CalibrationRepositoryId"
        )
        self._calibration_repository.setPlaceholderText("repository_id")
        self._calibration_digest = FluentLineEdit(parent=self)
        self._calibration_digest.setObjectName(
            f"{object_prefix}CalibrationManifestDigest"
        )
        self._calibration_digest.setPlaceholderText("64-character manifest digest")
        self._model_kind = FluentComboBox(self)
        self._model_kind.setObjectName(f"{object_prefix}ReadoutModelKind")
        self._model_kind.addItem("Calibration default", None)
        for kind in ReadoutModelKind:
            self._model_kind.addItem(kind.value, kind)
        source_box = _settings_card(
            "Source",
            (
                ("Pipeline", self._source),
                ("Calibration repository", self._calibration_repository),
                ("Calibration digest", self._calibration_digest),
                ("Readout model", self._model_kind),
            ),
            self,
        )

        self._binding_form = FluentParameterForm(
            task_console_scan_binding_form_spec(
                card.device_catalog.roles("camera"),
                card.device_catalog.roles("sequencer"),
            ),
            parent=self,
        )
        self._binding_form.setObjectName(f"{object_prefix}DeviceBindingForm")
        self._binding_form.widget_for("camera_role").setObjectName(
            f"{object_prefix}CameraRole"
        )
        self._binding_form.widget_for("sequencer_role").setObjectName(
            f"{object_prefix}SequencerRole"
        )
        self._binding_form.widget_for("trigger_channel").setObjectName(
            f"{object_prefix}TriggerChannel"
        )
        binding_box = _form_card("Device binding", self._binding_form, self)

        self._budget_form = FluentParameterForm(
            task_console_scan_budget_form_spec(),
            parent=self,
        )
        self._budget_form.setObjectName(f"{object_prefix}RunBudgetForm")
        self._budget_form.widget_for(
            "transport_memory_limit_bytes"
        ).setObjectName(f"{object_prefix}TransportBudgetBytes")
        self._budget_form.widget_for("memory_limit_bytes").setObjectName(
            f"{object_prefix}MemoryBudgetBytes"
        )
        self._budget_form.widget_for("timeout_seconds").setObjectName(
            f"{object_prefix}DeadlineSeconds"
        )
        budget_box = _form_card(
            "Budgets and deadline",
            self._budget_form,
            self,
        )

        self._site_mode = FluentComboBox(self)
        self._site_mode.setObjectName(f"{object_prefix}SiteDisplayMode")
        self._site_mode.addItem("Auto", "auto")
        self._site_mode.addItem("Batch", "batch")
        self._site_mode.addItem("Select", "select")
        self._site_index = FluentSpinBox(self)
        self._site_index.setObjectName(f"{object_prefix}SiteDisplayIndex")
        self._site_index.setRange(0, 2_147_483_647)
        display_box = _settings_card(
            "Display only · SITE",
            (("Mode", self._site_mode), ("Index", self._site_index)),
            self,
        )

        self._diagnostics = FluentLabel("", self)
        self._diagnostics.setObjectName(f"{object_prefix}ScanEditorDiagnostics")
        self._diagnostics.setWordWrap(True)
        self._diagnostics.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._apply = FluentButton("Apply", self, color=GREEN)
        self._apply.setObjectName(f"{object_prefix}ApplyScanIntentButton")
        self._cancel = FluentButton("Cancel", self, color=GREY)
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

        content = QtWidgets.QWidget(self)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.addWidget(pulse_box)
        content_layout.addWidget(self._api_box)
        content_layout.addWidget(authority_box)
        content_layout.addLayout(columns)
        content_layout.addStretch(1)
        self._scroll = FluentScrollArea(self)
        self._scroll.setObjectName(f"{object_prefix}ScanIntentScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._scroll, 1)
        layout.addWidget(self._diagnostics)
        layout.addLayout(buttons)

        self._pulse_path.selected.connect(self._load_selected_pulse)
        self._load_pulse.clicked.connect(self._load_path_from_edit)
        self._slot_mode.currentIndexChanged.connect(self._slot_mode_changed)
        self._add_api_row.clicked.connect(lambda: self._append_api_row())
        self._remove_api_row.clicked.connect(
            lambda: self._remove_selected_api_rows()
        )
        self._source.currentIndexChanged.connect(self._update_enabled_state)
        self._site_mode.currentIndexChanged.connect(self._update_enabled_state)
        self._clear_authority.clicked.connect(self._clear_authoritative_transform)
        self._apply.clicked.connect(self._apply_clicked)
        self._cancel.clicked.connect(self.cancel_edit)
        self._slot_stack.setCurrentWidget(self._fixed_api_page)
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
        if (
            (document.scan_table is None or not document.scan_parameters)
            and not document.api_parameters
        ):
            raise ValueError("PulseDocument declares neither SCAN_SLOT nor API_SLOT")
        self._document = document
        self._document_source = str(Path(path).expanduser().resolve())
        self._pulse_path.setText(self._document_source)
        self._output_transform_spec = None
        self._refresh_authority_summary()
        self._show_document(document, None)
        self._diagnostics.clear()

    def build_intent(self) -> TaskConsoleScanIntent:
        document = self._document
        if document is None:
            raise ValueError("a current PulseDocument JSON must be loaded")
        mode = self._slot_mode.currentData()
        if mode == _SCAN_SLOT_MODE:
            api_values = (
                {}
                if self._api_form is None
                else self._api_form.read_all()
            )
            program = AutonomousScanSlotProgram(
                document,
                tuple(
                    (
                        parameter.parameter_id,
                        api_values[parameter.parameter_id],
                    )
                    for parameter in document.api_parameters
                ),
            )
        elif mode == _API_SLOT_MODE:
            if not self._allow_host_gaps.isChecked():
                raise ValueError(
                    "API_SLOT segmented execution requires explicit confirmation "
                    "that points are independent and host gaps are allowed"
                )
            columns = tuple(
                parameter.parameter_id for parameter in document.api_parameters
            )
            rows = tuple(
                tuple(
                    parse_number_text(
                        "" if self._api_table.item(row, column) is None else
                        self._api_table.item(row, column).text(),
                        f"API row {row} value {parameter_id!r}",
                    )
                    for column, parameter_id in enumerate(columns)
                )
                for row in range(self._api_table.rowCount())
            )
            program = ApiSlotSegmentedProgram(
                document,
                ApiSegmentTable(columns, rows),
                self._segmentation_rationale.text().strip(),
            )
        else:
            raise ValueError("select SCAN_SLOT or API_SLOT segmented execution")
        binding = self._binding_form.read_all()
        budgets = self._budget_form.read_all()
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
            task_key=PULSE_SCAN_TASK_KEY,
            measurement_key=CAMERA_MEASUREMENT_KEY,
            processor_key=(OCCUPANCY_STREAM_PROCESSOR_KEY if occupancy else None),
            program=program,
            camera_role=binding["camera_role"],
            sequencer_role=binding["sequencer_role"],
            trigger_channel=binding["trigger_channel"].strip() or None,
            calibration_ref=calibration_ref,
            model_kind=model_kind,
            output_transform_spec=self._output_transform_spec,
            display_intent=display,
            transport_memory_limit_bytes=budgets[
                "transport_memory_limit_bytes"
            ],
            memory_limit_bytes=budgets["memory_limit_bytes"],
            timeout_seconds=budgets["timeout_seconds"],
        )

    def _populate(self, intent: TaskConsoleScanIntent) -> None:
        program = intent.program
        document = program.document
        previous_document = self._document
        self._document = document
        if (
            self._document_source is None
            or previous_document is None
            or previous_document.fingerprint != document.fingerprint
        ):
            self._document_source = None
        self._pulse_path.setText(self._document_source or _EMBEDDED_DOCUMENT)
        self._output_transform_spec = intent.output_transform_spec
        self._refresh_authority_summary()
        self._show_document(document, program)
        occupancy = intent.processor_key is not None
        self._source.setCurrentIndex(1 if occupancy else 0)
        self._binding_form.populate(
            {
                "camera_role": intent.camera_role,
                "sequencer_role": intent.sequencer_role,
                "trigger_channel": intent.trigger_channel or "",
            }
        )
        self._budget_form.populate(
            {
                "transport_memory_limit_bytes": (
                    intent.transport_memory_limit_bytes
                ),
                "memory_limit_bytes": intent.memory_limit_bytes,
                "timeout_seconds": intent.timeout_seconds,
            }
        )
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
        program: AutonomousScanSlotProgram | ApiSlotSegmentedProgram | None,
    ) -> None:
        self._fingerprint.setText(f"Fingerprint: {document.fingerprint}")
        selected = (
            _API_SLOT_MODE
            if isinstance(program, ApiSlotSegmentedProgram)
            else _SCAN_SLOT_MODE
            if isinstance(program, AutonomousScanSlotProgram)
            else _SCAN_SLOT_MODE
            if document.scan_table is not None and document.scan_parameters
            else _API_SLOT_MODE
        )
        self._slot_mode.blockSignals(True)
        self._slot_mode.clear()
        if document.scan_table is not None and document.scan_parameters:
            self._slot_mode.addItem("Autonomous SCAN_SLOT", _SCAN_SLOT_MODE)
        if document.api_parameters:
            self._slot_mode.addItem(
                "API_SLOT · segmented existing",
                _API_SLOT_MODE,
            )
        index = self._slot_mode.findData(selected)
        if index < 0:
            raise ValueError("saved scan program is incompatible with PulseDocument")
        self._slot_mode.setCurrentIndex(index)
        self._slot_mode.blockSignals(False)
        self._reset_slot_values(selected, program)
        self._refresh_slot_summary()

    def _reset_slot_values(
        self,
        mode: str,
        program: AutonomousScanSlotProgram | ApiSlotSegmentedProgram | None = None,
    ) -> None:
        document = self._document
        if document is None:
            return
        self._clear_api_rows()
        api_values = (
            dict(program.api_values)
            if isinstance(program, AutonomousScanSlotProgram)
            else {
                parameter.parameter_id: document.field_value(parameter.field)[0]
                for parameter in document.api_parameters
            }
        )
        if not document.api_parameters:
            self._api_empty = FluentLabel(
                "No whole-run API constants declared",
                self._fixed_api_page,
            )
            self._api_empty.setObjectName(f"{self._prefix}ApiConstantsState")
            self._api_layout.addWidget(self._api_empty)
        else:
            self._api_form = FluentParameterForm(
                _api_constant_form_spec(document),
                api_values,
                parent=self._fixed_api_page,
            )
            self._api_form.setObjectName(f"{self._prefix}ApiConstantForm")
            for parameter in document.api_parameters:
                self._api_form.widget_for(parameter.parameter_id).setObjectName(
                    f"{self._prefix}ApiValue_{parameter.parameter_id}"
                )
            self._api_layout.addWidget(self._api_form)
        columns = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        self._api_table.clear()
        self._api_table.setColumnCount(len(columns))
        self._api_table.setHorizontalHeaderLabels(list(columns))
        rows = (
            program.table.rows
            if isinstance(program, ApiSlotSegmentedProgram)
            else (tuple(api_values[column] for column in columns),)
            if columns
            else ()
        )
        self._api_table.setRowCount(0)
        for row in rows:
            self._append_api_row(row)
        self._allow_host_gaps.setChecked(
            isinstance(program, ApiSlotSegmentedProgram)
        )
        self._segmentation_rationale.setText(
            program.segmentation_rationale
            if isinstance(program, ApiSlotSegmentedProgram)
            else ""
        )
        self._slot_stack.setCurrentWidget(
            self._segmented_api_page
            if mode == _API_SLOT_MODE
            else self._fixed_api_page
        )

    def _refresh_slot_summary(self) -> None:
        document = self._document
        if document is None:
            self._scan_summary.setText("Pulse scan: no document")
            return
        mode = self._slot_mode.currentData()
        if mode == _SCAN_SLOT_MODE:
            table = document.scan_table
            if table is None:
                self._scan_summary.setText("SCAN_SLOT: unavailable")
            else:
                self._scan_summary.setText(
                    "SCAN_SLOT · one autonomous FIRE · columns="
                    + ", ".join(table.columns)
                    + f" · points={len(table.rows)}"
                )
        else:
            self._scan_summary.setText(
                "API_SLOT_SEGMENTED_EXISTING · one finite hardware segment per "
                f"(repeat, point) · points={self._api_table.rowCount()}"
            )

    def _slot_mode_changed(self, _index: int = -1) -> None:
        mode = self._slot_mode.currentData()
        if mode not in (_SCAN_SLOT_MODE, _API_SLOT_MODE):
            return
        self._reset_slot_values(mode)
        self._refresh_slot_summary()

    def _append_api_row(self, values=None) -> None:
        document = self._document
        if document is None:
            return
        if values is None:
            values = tuple(
                document.field_value(parameter.field)[0]
                for parameter in document.api_parameters
            )
        values = tuple(values)
        if len(values) != self._api_table.columnCount():
            raise ValueError("API row width differs from declared parameters")
        row = self._api_table.rowCount()
        self._api_table.insertRow(row)
        for column, value in enumerate(values):
            self._api_table.setItem(
                row,
                column,
                QtWidgets.QTableWidgetItem(str(value)),
            )
        self._refresh_slot_summary()

    def _remove_selected_api_rows(self) -> None:
        selected = sorted(
            {index.row() for index in self._api_table.selectedIndexes()},
            reverse=True,
        )
        if not selected and self._api_table.rowCount():
            selected = [self._api_table.rowCount() - 1]
        for row in selected:
            self._api_table.removeRow(row)
        self._refresh_slot_summary()

    def _clear_api_rows(self) -> None:
        while self._api_layout.count():
            item = self._api_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._api_form = None

    def _clear_unapplied(self) -> None:
        self._document = None
        self._document_source = None
        self._output_transform_spec = None
        self._refresh_authority_summary()
        self._base_revision = None
        self._pulse_path.clear()
        self._fingerprint.setText("Fingerprint: —")
        self._scan_summary.setText("Pulse scan: no document")
        self._slot_mode.blockSignals(True)
        self._slot_mode.clear()
        self._slot_mode.blockSignals(False)
        self._clear_api_rows()
        self._api_empty = FluentLabel("Load a PulseDocument", self._fixed_api_page)
        self._api_layout.addWidget(self._api_empty)
        self._api_table.clear()
        self._api_table.setRowCount(0)
        self._api_table.setColumnCount(0)
        self._allow_host_gaps.setChecked(False)
        self._segmentation_rationale.clear()
        self._slot_stack.setCurrentWidget(self._fixed_api_page)
        self._source.setCurrentIndex(0)
        self._calibration_repository.clear()
        self._calibration_digest.clear()
        self._model_kind.setCurrentIndex(0)
        self._binding_form.populate(self._binding_form.spec.default_values())
        self._budget_form.populate(self._budget_form.spec.default_values())
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

    def _load_selected_pulse(self, path: str) -> None:
        try:
            self.load_pulse_path(path)
        except BaseException as error:
            self._pulse_path.setText(
                ""
                if self._document is None
                else self._document_source or _EMBEDDED_DOCUMENT
            )
            self.show_error(error)

    def _load_path_from_edit(self) -> None:
        path = self._pulse_path.text().strip()
        if not path or path == _EMBEDDED_DOCUMENT:
            self.show_error(ValueError("choose a PulseDocument JSON path"))
            return
        self._load_selected_pulse(path)

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

    finalReferenceChanged = QtCore.pyqtSignal(object)

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
        self._reported_final_reference = None

        title = FluentLabel("Task: Pulse scan", self)
        title.setObjectName("taskCardTitle")
        self._state = FluentLabel("STOPPED · CONFIGURATION REQUIRED", self)
        self._state.setObjectName("taskCardState")
        self._settings = FluentButton("Setting…", self, color=GREY)
        self._settings.setObjectName("taskSettingsButton")
        header = QtWidgets.QHBoxLayout()
        header.addWidget(title)
        header.addWidget(self._state)
        header.addStretch(1)
        header.addWidget(self._settings)

        self._tabs = FluentTabWidget(self)
        self._tabs.setObjectName("taskCardTabs")
        self._run_page = QtWidgets.QWidget(self._tabs)
        self._run_page.setObjectName("taskRunTab")
        self._run_layout = QtWidgets.QVBoxLayout(self._run_page)
        self._placeholder = FluentLabel(
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
        self._settings_dialog.setWindowTitle("Pulse scan Setting")
        settings_layout = QtWidgets.QVBoxLayout(self._settings_dialog)
        self._settings_form = ScanIntentForm(self, object_prefix="setting")
        settings_layout.addWidget(self._settings_form)
        self._settings_form.applied.connect(self._settings_dialog.accept)

        self._diagnostics = FluentLabel("", self)
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
    def device_catalog(self):
        return self._experiment.device_catalog

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
    def final_reference(self):
        return None if self._panel is None else self._panel.final_reference

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
        self._refresh_state()
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
        self._refresh_state()

    def shutdown(self) -> None:
        self._state_timer.stop()
        if self._panel is not None:
            self._panel.shutdown()

    def _install_panel(self, panel: ScanWorkbenchWindow) -> None:
        if self._panel is not None:
            raise RuntimeError("scan card already owns a panel")
        self._panel = panel
        panel.finalReferenceChanged.connect(self._forward_final_reference)
        panel.setObjectName("embeddedScanWorkbench")
        self._run_layout.removeWidget(self._placeholder)
        self._placeholder.hide()
        self._run_layout.addWidget(panel, 1)
        panel.show()

    def _forward_final_reference(self, reference) -> None:
        if reference != self._reported_final_reference:
            self._reported_final_reference = reference
            self.finalReferenceChanged.emit(reference)

    def _open_settings(self) -> None:
        self._settings_form.begin_edit()
        self._settings_dialog.resize(
            screen_fit_window_size(WINDOW_SCREEN_FRACTION)
        )
        self._settings_dialog.show()
        center_window_on_primary_screen(self._settings_dialog)
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
        self._forward_final_reference(self.final_reference)


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
        self._experiment = experiment
        self._card: TaskScanCard | None = None
        self._closing = False
        self._analysis_window = None
        self._analysis_source = None

        self._catalog = FluentComboBox(self)
        self._catalog.setObjectName("taskCatalogCombo")
        items = task_console_catalog_items(compose_task_console_catalog())
        task_items = tuple(item for item in items if item.group == "Task")
        if len(task_items) != 1:
            raise RuntimeError("TaskConsole requires exactly one current Task")
        for item in task_items:
            self._catalog.addItem(f"{item.group}: {item.title}", item.key)
        self._add = FluentButton("Add Panel", self)
        self._add.setObjectName("addTaskPanelButton")
        self._add_analysis = FluentButton("Add Analysis → Fit", self, color=GREEN)
        self._add_analysis.setObjectName("addTaskAnalysisButton")
        self._add_analysis.setEnabled(False)
        self._save = FluentButton("Save", self)
        self._save.setObjectName("saveTaskIntentButton")
        self._load = FluentButton("Load", self, color=ORANGE)
        self._load.setObjectName("loadTaskIntentButton")
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._catalog, 1)
        controls.addWidget(self._add)
        controls.addWidget(self._add_analysis)
        controls.addStretch(1)
        controls.addWidget(self._save)
        controls.addWidget(self._load)

        self._card_host = QtWidgets.QWidget(self)
        self._card_host.setObjectName("taskCardHost")
        self._card_layout = QtWidgets.QVBoxLayout(self._card_host)
        self._empty = FluentLabel("Add the Pulse scan task", self._card_host)
        self._empty.setObjectName("taskConsoleEmptyState")
        self._empty.setAlignment(QtCore.Qt.AlignCenter)
        self._card_layout.addWidget(self._empty, 1)
        self._status = FluentLabel("", self)
        self._status.setObjectName("taskConsoleDiagnostics")
        self._status.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._card_host, 1)
        layout.addWidget(self._status)

        self._add.clicked.connect(self._add_card)
        self._add_analysis.clicked.connect(self._open_analysis)
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

    @property
    def analysis_window(self):
        return self._analysis_window

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
        if self._catalog.currentData() != PULSE_SCAN_TASK_KEY:
            self._status.setText("Only Pulse scan is available")
            return
        self._create_card(None)

    def _create_card(self, intent: TaskConsoleScanIntent | None) -> None:
        if self._card is not None:
            raise RuntimeError("TaskConsole currently owns exactly one card")
        card = TaskScanCard(self._experiment, intent, self._card_host)
        self._card = card
        card.finalReferenceChanged.connect(self._sync_analysis_entry)
        self._card_layout.removeWidget(self._empty)
        self._empty.hide()
        self._card_layout.addWidget(card, 1)
        self._add.setEnabled(False)
        self._catalog.setEnabled(False)
        self._sync_analysis_entry(card.final_reference)
        self._status.clear()

    def _sync_analysis_entry(self, reference) -> None:
        self._add_analysis.setEnabled(
            not self._closing
            and reference is not None
        )

    def _open_analysis(self) -> None:
        card = self._card
        reference = None if card is None else card.final_reference
        if reference is None:
            self._status.setText(
                "Fit Analysis requires this card's current FINAL scan artifact"
            )
            self._sync_analysis_entry(None)
            return
        current = self._analysis_window
        if (
            current is not None
            and self._analysis_source == reference
            and not current.closed
        ):
            current.show()
            current.raise_()
            current.activateWindow()
            return
        try:
            window = self._experiment.fit_gui(reference)
        except BaseException as error:
            self._show_error(error)
            return
        self._analysis_window = window
        self._analysis_source = reference
        self._status.setText(
            "Opened exact Fit Analysis for the current FINAL scan artifact"
        )

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
            self._add_analysis.setEnabled(False)
            if self._card is not None:
                self._card.shutdown()
        if panel is None or panel.closed:
            self._close_timer.stop()
            release_window(self)
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

    application = ensure_qt_app()
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("TaskConsole must be opened on the Qt GUI thread")
    set_fluent_scale(None)
    window = TaskConsoleWindow(experiment, initial_intent)
    window.resize(screen_fit_window_size(WINDOW_SCREEN_FRACTION))
    retain_window(window)
    window.show()
    center_window_on_primary_screen(window, application)
    return window


__all__ = [
    "ScanIntentForm",
    "TaskConsoleWindow",
    "TaskScanCard",
    "open_task_console",
]
