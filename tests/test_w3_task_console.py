"""Current SCAN_SLOT TaskConsole intent, catalog, and edit authority."""

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
    AUTONOMOUS_SCAN_SLOT_TASK_KEY,
    ScanPointTable,
    bind_scan_output_contract,
)
from zlc_pulse import (
    FrozenScanTable,
    RepeatRegion,
    ScanParameter,
    load_pulse_document,
)
from zlc_workbench.task_console import (
    ScanDisplayIntent,
    ScanEditConflict,
    ScanEditDraft,
    ScanEditorSession,
    TaskConsoleScanIntent,
    compose_task_console_catalog,
    decode_task_console_scan_intent,
    describe_authoritative_transform,
    encode_task_console_scan_intent,
    load_task_console_scan_intent,
    save_task_console_scan_intent,
    task_console_catalog_items,
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
        task_key=AUTONOMOUS_SCAN_SLOT_TASK_KEY,
        measurement_key=CAMERA_MEASUREMENT_KEY,
        processor_key=(OCCUPANCY_STREAM_PROCESSOR_KEY if occupancy else None),
        pulse_document=document,
        api_values=_api_values(document),
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


def test_scan_intent_round_trip_preserves_named_axes_and_display_authority(tmp_path):
    intent = _intent(occupancy=True)
    point_table = ScanPointTable.from_pulse_document(intent.pulse_document)

    assert len(point_table.point_axes) == 3
    assert point_table.rows == intent.pulse_document.scan_table.rows
    assert decode_task_console_scan_intent(encode_task_console_scan_intent(intent)) == intent

    target = tmp_path / "scan-task.zlc"
    assert save_task_console_scan_intent(intent, target) == target.resolve()
    assert load_task_console_scan_intent(target) == intent
    assert intent.output_transform_spec is None
    assert intent.display_intent == ScanDisplayIntent("select", 1)


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
        pulse_document=document,
        api_values=_api_values(document),
    )
    assert direct.api_values
    with pytest.raises(ValueError, match="exactly one whole-run value"):
        replace(direct, api_values=direct.api_values[:-1])
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
    from PyQt5 import QtCore, QtTest, QtWidgets
    from Zou_lab_control.workbench._task_console import ScanIntentForm

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

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
    try:
        unconfigured = exp.task_console()
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
        assert all(not timer.isActive() for timer in unconfigured_timers)
        unconfigured = None

        blank = exp.task_console()
        add = blank.findChild(QtWidgets.QPushButton, "addTaskPanelButton")
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
        camera_role = blank.findChild(QtWidgets.QLineEdit, "editCameraRole")
        site_mode = blank.findChild(QtWidgets.QComboBox, "editSiteDisplayMode")
        assert all(
            widget is not None
            for widget in (
                source,
                calibration_repository,
                calibration_digest,
                model,
                camera_role,
                site_mode,
            )
        )
        source.setCurrentIndex(source.findData("occupancy"))
        calibration_repository.setText("cancelled-repository")
        calibration_digest.setText("c" * 64)
        model.setCurrentIndex(model.findData(ReadoutModelKind.PER_SITE_PSF))
        camera_role.setText("cancelled-camera")
        site_mode.setCurrentIndex(site_mode.findData("select"))
        blank_edit.cancel_edit()
        assert source.currentData() == "direct"
        assert not calibration_repository.text()
        assert not calibration_digest.text()
        assert model.currentData() is None
        assert camera_role.text() == "camera"
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

        direct_document = _occupancy_document()
        direct_intent = TaskConsoleScanIntent(
            task_key=AUTONOMOUS_SCAN_SLOT_TASK_KEY,
            measurement_key=CAMERA_MEASUREMENT_KEY,
            processor_key=None,
            pulse_document=direct_document,
            api_values=_api_values(direct_document),
            camera_role="camera",
            sequencer_role="sequencer",
        )
        source.setCurrentIndex(source.findData("occupancy"))
        calibration_repository.setText("stale-repository")
        calibration_digest.setText("d" * 64)
        model.setCurrentIndex(model.findData(ReadoutModelKind.PER_SITE_PSF))
        blank.scan_card.load_intent(direct_intent)
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
        blank = None

        document = _occupancy_document()
        calibration_ref = exp.readout.sitemap(frames=6)
        intent = TaskConsoleScanIntent(
            task_key=AUTONOMOUS_SCAN_SLOT_TASK_KEY,
            measurement_key=CAMERA_MEASUREMENT_KEY,
            processor_key=OCCUPANCY_STREAM_PROCESSOR_KEY,
            pulse_document=document,
            api_values=_api_values(document),
            camera_role="camera",
            sequencer_role="sequencer",
            calibration_ref=calibration_ref,
            model_kind=None,
            display_intent=ScanDisplayIntent("select", 1),
            timeout_seconds=20.0,
        )
        window = exp.task_console(intent)
        card = window.scan_card
        assert card is not None and card.current_intent == intent
        edit = window.findChild(ScanIntentForm, "editScanIntentForm")
        setting = window.findChild(ScanIntentForm, "settingScanIntentForm")
        assert edit is not None and setting is not None
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
        window = None
    finally:
        if unconfigured is not None and unconfigured.isVisible():
            unconfigured.close()
        if blank is not None and blank.isVisible():
            blank.close()
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
