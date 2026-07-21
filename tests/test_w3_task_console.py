"""Current Pulse scan TaskConsole intent, catalog, and edit authority."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import numpy as np
import Zou_lab_control.notebook as zlc

from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    SITE,
    AxisId,
    AxisSpec,
    ComponentValidity,
    DataTransformSpec,
    DatasetSchema,
    Selection,
    ValidityContract,
    ValueSchema,
    commit_transform,
)
from zlc_frontend.figure import AxisViewRole
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
from zlc_neutral_atom.readout.calibration import ReadoutModelKind
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.occupancy import OCCUPANCY_STREAM_PROCESSOR_KEY
from zlc_neutral_atom.readout.sitemap import load_packaged_sitemap_pulse
from zlc_neutral_atom.scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PULSE_SCAN_TASK_KEY,
    ScanPointTable,
    bind_scan_output_contract,
)
from zlc_pulse import (
    FrozenScanTable,
    RepeatRegion,
    ScanParameter,
    TIME_UNIT_TO_NS,
    load_pulse_document,
    replace_pulse_field,
)
from zlc_frontend import describe_authoritative_transform
from zlc_workbench.task_console import (
    SCAN_INTENT_DEFAULT_CAMERA_ROLE,
    SCAN_INTENT_DEFAULT_PIPELINE_MEMORY_BYTES,
    SCAN_INTENT_DEFAULT_SEQUENCER_ROLE,
    SCAN_INTENT_DEFAULT_TIMEOUT_SECONDS,
    SCAN_INTENT_DEFAULT_TRANSPORT_MEMORY_BYTES,
    ScanDisplayIntent,
    ScanEditConflict,
    ScanEditDraft,
    ScanEditorSession,
    TaskConsoleScanIntent,
    compose_task_console_catalog,
    decode_task_console_scan_intent,
    encode_task_console_scan_intent,
    load_task_console_scan_intent,
    save_task_console_scan_intent,
    task_console_catalog_items,
    task_console_scan_binding_form_spec,
    task_console_scan_budget_form_spec,
)
from zlc_workbench.progressive_scan import build_occupancy_progressive_spec


ROOT = Path(__file__).resolve().parents[1]


def _document():
    return load_pulse_document(ROOT / "pulses" / "mot_field_template.json")


def _api_values(document):
    return tuple(
        (
            parameter.parameter_id,
            document.field_value(parameter.field)[0],
        )
        for parameter in document.api_parameters
    )


def _intent(*, occupancy: bool = False):
    document = _document()
    return TaskConsoleScanIntent(
        task_key=PULSE_SCAN_TASK_KEY,
        measurement_key=CAMERA_MEASUREMENT_KEY,
        processor_key=(OCCUPANCY_STREAM_PROCESSOR_KEY if occupancy else None),
        program=AutonomousScanSlotProgram(document, _api_values(document)),
        camera_role="camera",
        sequencer_role="sequencer",
        calibration_ref=(
            CalibrationArtifactRef("calibration", "a" * 64)
            if occupancy
            else None
        ),
        model_kind=ReadoutModelKind.BOX if occupancy else None,
        display_intent=(
            ScanDisplayIntent("select", 1)
            if occupancy
            else ScanDisplayIntent()
        ),
    )


def _occupancy_document():
    """Turn the proven sitemap readout event into a two-point SCAN_SLOT."""

    document = load_packaged_sitemap_pulse()
    camera_port = next(
        port for port in document.target.ports if port.label == "emCCD"
    )
    trigger_index = document.target.raw_lanes.index(camera_port.lanes[0])
    segment = -1
    previous = 0
    periods = []
    for period in document.periods:
        states = list(period.states)
        current = int(states[trigger_index])
        if current and not previous:
            segment += 1
        states[trigger_index] = int(bool(current and segment == 1))
        periods.append(replace(period, states=tuple(states)))
        previous = current

    scanned_api = document.api_parameters[0]
    scanned_period = next(
        period
        for period in periods
        if period.period_id == scanned_api.field.period_id
    )
    parameter = ScanParameter(
        "reference_settle",
        scanned_api.field,
        "reference settle",
        scanned_api.unit,
    )
    start = scanned_period.duration
    step = 1 if isinstance(start, int) else 1e-6
    return replace(
        document,
        name="task-console-occupancy-scan",
        periods=tuple(periods),
        api_parameters=tuple(
            item for item in document.api_parameters if item is not scanned_api
        ),
        scan_parameters=(parameter,),
        scan_table=FrozenScanTable(
            (parameter.parameter_id,),
            ((start,), (start + step,)),
        ),
        repeat=RepeatRegion(
            periods[0].period_id,
            periods[-1].period_id,
            2,
        ),
    )


def _api_program():
    document = _occupancy_document()
    columns = tuple(item.parameter_id for item in document.api_parameters)
    baseline = tuple(
        document.field_value(parameter.field)[0]
        for parameter in document.api_parameters
    )
    varied = list(baseline)
    parameter = document.api_parameters[-1]
    one_tick = document.time_step_ns / TIME_UNIT_TO_NS[parameter.unit]
    varied_document = replace_pulse_field(
        document,
        parameter.field,
        float(varied[-1]) + one_tick,
        unit=parameter.unit,
    )
    varied[-1] = varied_document.field_value(parameter.field)[0]
    return ApiSlotSegmentedProgram(
        document,
        ApiSegmentTable(columns, (baseline, tuple(varied))),
        "Each readout is independent and permits an explicit host gap.",
    )


def _api_intent():
    return TaskConsoleScanIntent(
        task_key=PULSE_SCAN_TASK_KEY,
        measurement_key=CAMERA_MEASUREMENT_KEY,
        processor_key=None,
        program=_api_program(),
        camera_role="camera",
        sequencer_role="sequencer",
        timeout_seconds=20.0,
    )


def test_task_console_catalog_projects_every_static_definition_once():
    import zlc_neutral_atom.processing as processing_api
    import zlc_neutral_atom.processing.stream as stream_api
    import zlc_neutral_atom.runtime.pipeline as pipeline_api

    catalog = compose_task_console_catalog()
    items = task_console_catalog_items(catalog)

    assert len(catalog) == 3
    assert tuple(item.group for item in items) == (
        "Task",
        "Measurement",
        "Processor",
    )
    assert {item.key for item in items} == set(catalog.by_key)
    assert all(not callable(value) for definition in catalog for value in vars(definition).values())
    assert not hasattr(pipeline_api, "MeasurementDefinition")
    assert not hasattr(processing_api, "StreamProcessorDefinition")
    assert not hasattr(stream_api, "StreamProcessorDefinition")


def test_scan_scalar_form_specs_project_owner_defaults_and_strict_bounds_once():
    binding = task_console_scan_binding_form_spec(
        ("camera", "camera-secondary"),
        ("sequencer",),
    )
    budgets = task_console_scan_budget_form_spec()

    assert binding.keys == ("camera_role", "sequencer_role", "trigger_channel")
    assert binding.default_values() == {
        "camera_role": SCAN_INTENT_DEFAULT_CAMERA_ROLE,
        "sequencer_role": SCAN_INTENT_DEFAULT_SEQUENCER_ROLE,
        "trigger_channel": None,
    }
    assert tuple(choice.value for choice in binding.fields[0].choices) == (
        "camera",
        "camera-secondary",
    )
    assert budgets.default_values() == {
        "transport_memory_limit_bytes": (
            SCAN_INTENT_DEFAULT_TRANSPORT_MEMORY_BYTES
        ),
        "memory_limit_bytes": SCAN_INTENT_DEFAULT_PIPELINE_MEMORY_BYTES,
        "timeout_seconds": SCAN_INTENT_DEFAULT_TIMEOUT_SECONDS,
    }
    assert budgets.fields[0].minimum == 1
    assert budgets.fields[1].minimum == 1
    assert budgets.fields[2].minimum > 0.0


def test_scan_intent_round_trip_preserves_named_axes_and_display_authority(tmp_path):
    intent = _intent(occupancy=True)
    point_table = intent.program.point_table

    assert len(point_table.point_axes) == 3
    assert point_table.rows == intent.program.document.scan_table.rows
    assert decode_task_console_scan_intent(encode_task_console_scan_intent(intent)) == intent

    target = tmp_path / "scan-task.zlc"
    assert save_task_console_scan_intent(intent, target) == target.resolve()
    assert load_task_console_scan_intent(target) == intent
    assert intent.output_transform_spec is None
    assert intent.display_intent == ScanDisplayIntent("select", 1)


def test_api_scan_intent_save_load_uses_only_the_current_program_union(tmp_path):
    intent = _api_intent()
    assert decode_task_console_scan_intent(
        encode_task_console_scan_intent(intent)
    ) == intent
    target = tmp_path / "api-scan-task.zlc"
    save_task_console_scan_intent(intent, target)
    loaded = load_task_console_scan_intent(target)
    assert loaded == intent
    assert isinstance(loaded.program, ApiSlotSegmentedProgram)
    assert loaded.program.table.rows == intent.program.table.rows


def test_authoritative_transform_summary_is_visible_and_axis_named():
    operation = DataTransformSpec(
        (Selection.index(AxisId("site"), 2),)
    )
    assert describe_authoritative_transform(None) == (
        "None · no user-authored Select/Reduce"
    )
    assert describe_authoritative_transform(operation) == (
        "AUTHORITATIVE · select(site=index[2])"
    )


def test_scan_intent_requires_exact_whole_run_api_values_and_typed_source_chain():
    document = _occupancy_document()
    direct = replace(
        _intent(),
        program=AutonomousScanSlotProgram(document, _api_values(document)),
    )
    assert direct.program.api_values
    with pytest.raises(ValueError, match="exactly cover declared parameters"):
        replace(
            direct,
            program=replace(
                direct.program,
                api_values=direct.program.api_values[:-1],
            ),
        )
    with pytest.raises(ValueError, match="no SITE display choice"):
        replace(direct, display_intent=ScanDisplayIntent("batch"))
    with pytest.raises(ValueError, match="processor must be occupancy or absent"):
        replace(
            direct,
            processor_key=replace(
                OCCUPANCY_STREAM_PROCESSOR_KEY,
                stable_definition_id="invented",
            ),
            calibration_ref=CalibrationArtifactRef("calibration", "b" * 64),
        )


def test_setting_and_edit_share_one_optimistic_revision_without_last_write_wins():
    session = ScanEditorSession(_intent())
    setting = session.begin()
    edit = session.begin()

    applied = session.apply(
        ScanEditDraft(
            setting.base_revision,
            replace(setting.intent, camera_role="camera-secondary"),
        )
    )
    assert applied.revision == 1
    assert applied.intent.camera_role == "camera-secondary"
    with pytest.raises(ScanEditConflict, match="stale"):
        session.apply(edit)
    assert session.cancel(edit) == applied


def test_site_display_choice_changes_only_view_and_never_authoritative_axes():
    document = _document()
    points = ScanPointTable.from_pulse_document(document)
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 2, (0, 1))
    event = AxisSpec(
        AxisId("readout.event"),
        "readout event",
        READOUT_EVENT,
        1,
        ("image",),
    )
    site = AxisSpec(AxisId("site"), "site", SITE, 4, (0, 1, 2, 3))
    source = DatasetSchema(
        repeat,
        points.point_axes,
        points.point_layout,
        ValueSchema(
            (event, site),
            ValidityContract.components(site.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    transform = commit_transform(
        source,
        DataTransformSpec((Selection.index(event.axis_id, 0),)),
    )
    contract = bind_scan_output_contract(source, points, transform)

    selected = build_occupancy_progressive_spec(
        source,
        contract,
        identity="task-console-selected-site",
        display_intent=ScanDisplayIntent("select", 2),
    )
    batched = build_occupancy_progressive_spec(
        source,
        contract,
        identity="task-console-batched-sites",
        display_intent=ScanDisplayIntent("batch"),
    )

    assert selected.output_contract is contract
    assert batched.output_contract is contract
    assert contract.output_dataset_schema.cell_schema.data_axes == (site,)
    assert selected.document.layers[0].view.binding(site.axis_id).role is AxisViewRole.SELECTED
    assert batched.document.layers[0].view.binding(site.axis_id).role is AxisViewRole.BATCH
    assert "site=2" in selected.projection_summary
    assert "site=batch/4" in batched.projection_summary


def _run_task_console_product_e2e(tmp_path: Path):
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
    from Zou_lab_control.workbench._figure import DataFigureWindow
    from Zou_lab_control.workbench._task_console import ScanIntentForm
    from zlc_frontend.qt_widgets import GREEN, ORANGE, ensure_qt_app

    application = ensure_qt_app()

    def until(predicate, timeout=30.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            application.processEvents(QtCore.QEventLoop.AllEvents, 20)
            time.sleep(0.005)
        assert predicate()

    exp = zlc.connect("virtual", repository=tmp_path / "repository")
    unconfigured = None
    blank = None
    window = None
    analysis = None
    # The scan-intent editor is no longer what ``Experiment.task_console()`` opens -- that
    # entry now gives the operator's full console (C22).  This window survives as a
    # COMPONENT, so the tests that pin its behaviour ask for it by name.
    from Zou_lab_control.workbench import open_task_console

    try:
        unconfigured = open_task_console(exp)
        add = unconfigured.findChild(
            QtWidgets.QPushButton,
            "addTaskPanelButton",
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        unconfigured_timers = unconfigured.scan_card.findChildren(QtCore.QTimer)
        assert unconfigured_timers and any(
            timer.isActive() for timer in unconfigured_timers
        )
        unconfigured.close()
        until(lambda: not unconfigured.isVisible(), timeout=5.0)
        assert unconfigured not in application._zlc_retained_windows
        assert all(not timer.isActive() for timer in unconfigured_timers)
        unconfigured = None

        blank = open_task_console(exp)
        add = blank.findChild(QtWidgets.QPushButton, "addTaskPanelButton")
        add_analysis = blank.findChild(
            QtWidgets.QPushButton,
            "addTaskAnalysisButton",
        )
        assert add_analysis is not None
        assert add_analysis.text() == "Add Analysis → Fit"
        assert not add_analysis.isEnabled()
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        assert blank.scan_card is not None
        blank_edit = blank.findChild(ScanIntentForm, "editScanIntentForm")
        assert blank_edit is not None
        source = blank.findChild(QtWidgets.QComboBox, "editScanSource")
        calibration_repository = blank.findChild(
            QtWidgets.QLineEdit,
            "editCalibrationRepositoryId",
        )
        calibration_digest = blank.findChild(
            QtWidgets.QLineEdit,
            "editCalibrationManifestDigest",
        )
        model = blank.findChild(QtWidgets.QComboBox, "editReadoutModelKind")
        camera_role = blank.findChild(QtWidgets.QComboBox, "editCameraRole")
        trigger_channel = blank.findChild(
            QtWidgets.QLineEdit,
            "editTriggerChannel",
        )
        site_mode = blank.findChild(QtWidgets.QComboBox, "editSiteDisplayMode")
        assert all(
            widget is not None
            for widget in (
                source,
                calibration_repository,
                calibration_digest,
                model,
                camera_role,
                trigger_channel,
                site_mode,
            )
        )
        source.setCurrentIndex(source.findData("occupancy"))
        calibration_repository.setText("cancelled-repository")
        calibration_digest.setText("c" * 64)
        model.setCurrentIndex(model.findData(ReadoutModelKind.PER_SITE_PSF))
        trigger_channel.setText("cancelled-trigger")
        site_mode.setCurrentIndex(site_mode.findData("select"))
        blank_edit.cancel_edit()
        assert source.currentData() == "direct"
        assert not calibration_repository.text()
        assert not calibration_digest.text()
        assert model.currentData() is None
        assert camera_role.currentData() == "camera"
        assert not trigger_channel.text()
        assert site_mode.currentData() == "auto"

        blank.scan_card.load_intent(_intent(occupancy=True))
        blank_edit.begin_edit()
        source.setCurrentIndex(source.findData("direct"))
        revision = blank.scan_card.apply_form(blank_edit)
        blank_edit.accept_revision(revision)
        assert source.currentData() == "direct"
        assert not calibration_repository.text()
        assert not calibration_digest.text()
        assert model.currentData() is None

        api_intent = _api_intent()
        blank.scan_card.load_intent(api_intent)
        slot_mode = blank.findChild(
            QtWidgets.QComboBox,
            "editPulseScanSlotMode",
        )
        api_table = blank.findChild(
            QtWidgets.QTableWidget,
            "editApiSegmentTable",
        )
        allow_gaps = blank.findChild(
            QtWidgets.QCheckBox,
            "editAllowApiHostGaps",
        )
        rationale = blank.findChild(
            QtWidgets.QLineEdit,
            "editApiSegmentationRationale",
        )
        assert slot_mode.currentData() == "api-slot-segmented"
        assert api_table.rowCount() == len(api_intent.program.table.rows)
        assert allow_gaps.isChecked()
        assert rationale.text() == api_intent.program.segmentation_rationale
        api_table.item(0, 0).setText("999")
        allow_gaps.setChecked(False)
        rationale.setText("unapplied")
        blank_edit.cancel_edit()
        assert slot_mode.currentData() == "api-slot-segmented"
        assert api_table.item(0, 0).text() == str(api_intent.program.table.rows[0][0])
        assert allow_gaps.isChecked()
        assert rationale.text() == api_intent.program.segmentation_rationale

        api_panel = blank.scan_card.panel
        assert api_panel is not None
        assert "FINAL-ONLY" in api_panel.findChild(
            QtWidgets.QLabel,
            "scanMode",
        ).text()
        api_start = api_panel.findChild(QtWidgets.QPushButton, "startScanButton")
        QtTest.QTest.mouseClick(api_start, QtCore.Qt.LeftButton)
        until(lambda: api_panel.final_reference is not None)
        api_result = exp.readout.materialize_scan(api_panel.final_reference)
        assert api_result.values.shape[:2] == (2, 2)
        until(lambda: api_panel.worker_idle)
        until(add_analysis.isEnabled)

        QtTest.QTest.mouseClick(add_analysis, QtCore.Qt.LeftButton)
        analysis = blank.analysis_window
        assert isinstance(analysis, DataFigureWindow)
        until(
            lambda: analysis.worker_idle
            and analysis.raster_ready
            and bool(analysis.fit_models),
            timeout=45.0,
        )
        pane = analysis._fit_pane
        assert pane is not None and analysis._tabs.currentWidget() is pane
        assert pane.current_option().spec.committed_transform is None

        # Repeated activation of the same exact FINAL artifact focuses the one
        # shared DataFigure/Fit host instead of constructing a second Fit UI.
        QtTest.QTest.mouseClick(add_analysis, QtCore.Qt.LeftButton)
        assert blank.analysis_window is analysis
        QtTest.QTest.mouseClick(pane.fit_button, QtCore.Qt.LeftButton)
        until(
            lambda: analysis.worker_idle
            and analysis.draft_ready
            and analysis.raster_ready,
            timeout=45.0,
        )
        QtTest.QTest.mouseClick(pane.save_button, QtCore.Qt.LeftButton)
        until(
            lambda: analysis.worker_idle
            and analysis.saved_reference is not None
            and analysis.raster_ready,
            timeout=45.0,
        )
        saved_fit = exp.load_fit(analysis.saved_reference)
        assert saved_fit.source_artifact_ref == api_panel.final_reference
        analysis.close()
        until(lambda: analysis.closed and not analysis.isVisible(), timeout=10.0)
        analysis = None

        # A new Run revokes the old artifact action in the same Qt turn; the
        # previous FINAL ref must never remain an enabled implicit source.
        QtTest.QTest.mouseClick(api_start, QtCore.Qt.LeftButton)
        assert api_panel.final_reference is None
        assert not add_analysis.isEnabled()
        until(lambda: api_panel.final_reference is not None)
        until(lambda: api_panel.worker_idle)
        until(add_analysis.isEnabled)

        direct_document = _occupancy_document()
        direct_intent = TaskConsoleScanIntent(
            task_key=PULSE_SCAN_TASK_KEY,
            measurement_key=CAMERA_MEASUREMENT_KEY,
            processor_key=None,
            program=AutonomousScanSlotProgram(
                direct_document,
                _api_values(direct_document),
            ),
            camera_role="camera",
            sequencer_role="sequencer",
        )
        source.setCurrentIndex(source.findData("occupancy"))
        calibration_repository.setText("stale-repository")
        calibration_digest.setText("d" * 64)
        model.setCurrentIndex(model.findData(ReadoutModelKind.PER_SITE_PSF))
        blank.scan_card.load_intent(direct_intent)
        assert not add_analysis.isEnabled()
        assert source.currentData() == "direct"
        assert not calibration_repository.text()
        assert not calibration_digest.text()
        assert model.currentData() is None
        source.setCurrentIndex(source.findData("occupancy"))
        calibration_repository.setText("cancelled-again")
        calibration_digest.setText("e" * 64)
        model.setCurrentIndex(model.findData(ReadoutModelKind.PER_SITE_PSF))
        blank_edit.cancel_edit()
        assert source.currentData() == "direct"
        assert not calibration_repository.text()
        assert not calibration_digest.text()
        assert model.currentData() is None
        blank.close()
        until(lambda: not blank.isVisible(), timeout=5.0)
        assert blank not in application._zlc_retained_windows
        blank = None

        document = _occupancy_document()
        calibration_ref = exp.readout.sitemap(frames=6)
        intent = TaskConsoleScanIntent(
            task_key=PULSE_SCAN_TASK_KEY,
            measurement_key=CAMERA_MEASUREMENT_KEY,
            processor_key=OCCUPANCY_STREAM_PROCESSOR_KEY,
            program=AutonomousScanSlotProgram(document, _api_values(document)),
            camera_role="camera",
            sequencer_role="sequencer",
            calibration_ref=calibration_ref,
            model_kind=None,
            display_intent=ScanDisplayIntent("select", 1),
            timeout_seconds=20.0,
        )
        window = open_task_console(exp, intent)
        card = window.scan_card
        assert card is not None and card.current_intent == intent
        edit = window.findChild(ScanIntentForm, "editScanIntentForm")
        setting = window.findChild(ScanIntentForm, "settingScanIntentForm")
        assert edit is not None and setting is not None
        available = application.primaryScreen().availableGeometry()
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        assert available.contains(window.frameGeometry())
        edit_scroll = window.findChild(
            QtWidgets.QScrollArea,
            "editScanIntentScroll",
        )
        apply_button = window.findChild(
            QtWidgets.QPushButton,
            "editApplyScanIntentButton",
        )
        load_button = window.findChild(
            QtWidgets.QPushButton,
            "editLoadPulseDocumentButton",
        )
        assert edit_scroll is not None and apply_button is not None
        assert not edit_scroll.isAncestorOf(apply_button)
        assert apply_button._base_color == QtGui.QColor(GREEN).name(
            QtGui.QColor.HexRgb
        )
        assert load_button._base_color == QtGui.QColor(ORANGE).name(
            QtGui.QColor.HexRgb
        )
        summary = window.findChild(
            QtWidgets.QLabel,
            "editAuthoritativeTransformSummary",
        )
        clear_transform = window.findChild(
            QtWidgets.QPushButton,
            "editClearAuthoritativeTransformButton",
        )
        assert summary is not None and "no user-authored Select/Reduce" in summary.text()
        assert clear_transform is not None and not clear_transform.isEnabled()
        edit.begin_edit()
        assert edit.build_intent().model_kind is None
        committed_path_text = edit._pulse_path.text()
        committed_fingerprint = edit.build_intent().program.document.fingerprint
        invalid_pulse = tmp_path / "not-a-pulse.json"
        invalid_pulse.write_text("{}", encoding="utf-8")
        edit._pulse_path.setText(str(invalid_pulse))
        edit._load_pulse.click()
        assert edit._pulse_path.text() == committed_path_text
        assert edit.build_intent().program.document.fingerprint == committed_fingerprint
        setting.begin_edit()
        card.apply_form(edit)
        with pytest.raises(ScanEditConflict, match="stale"):
            card.apply_form(setting)

        panel = card.panel
        assert panel is not None and panel.can_reconfigure
        start = panel.findChild(QtWidgets.QPushButton, "startScanButton")
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        until(lambda: panel.final_reference is not None)
        reference = panel.final_reference
        materialized = exp.readout.materialize_scan(reference)
        assert materialized.values.ndim == 3
        assert materialized.values.shape[:2] == (2, 2)
        assert isinstance(materialized.validity, ComponentValidity)
        assert materialized.validity.mask.shape == materialized.values.shape
        until(lambda: panel.can_reconfigure)

        path = window.save_intent(tmp_path / "task-console.json")
        assert path.is_file()
        loaded = window.load_intent(path)
        assert loaded == window.current_intent
        assert panel.final_reference is None
        assert panel.can_reconfigure

        window.close()
        until(lambda: panel.closed and not window.isVisible(), timeout=10.0)
        assert window not in application._zlc_retained_windows
        window = None
    finally:
        if unconfigured is not None and unconfigured.isVisible():
            unconfigured.close()
        if blank is not None and blank.isVisible():
            blank.close()
        if analysis is not None:
            analysis.close()
            until(lambda: analysis.closed and not analysis.isVisible(), timeout=10.0)
        if window is not None:
            window.close()
            until(lambda: not window.isVisible(), timeout=10.0)
        exp.close()


def test_task_console_add_edit_run_save_load_and_close_current_product(tmp_path):
    code = (
        "from pathlib import Path; import runpy, sys; "
        "namespace = runpy.run_path(sys.argv[1]); "
        "namespace['_run_task_console_product_e2e'](Path(sys.argv[2]))"
    )
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, (
        f"TaskConsole product subprocess failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_standalone_task_console_launcher_owns_current_virtual_product(tmp_path):
    source = (ROOT / "task_console.py").read_text(encoding="utf-8")
    # The launcher goes through the ONE composition root, and knows nothing of how the window
    # is assembled -- which is what lets the assembly be rewritten without touching entries.
    assert "from zlc_workbench.task_console.app import open_task_console" in source
    assert "window = open_task_console(experiment, state=args.state," in source
    for forbidden in (
        "Zou_lab_control.frontend.task_console",
        "SignalHub",
        "show_task_console",
        "TaskConsoleState",
        "default_console_state",
        "resolve_task_state",
    ):
        assert forbidden not in source

    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["ZLC_TASK_CONSOLE_AUTO_CLOSE_MS"] = "20"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "task_console.py"),
            "--repository",
            str(tmp_path / "standalone"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, (
        f"TaskConsole launcher failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
