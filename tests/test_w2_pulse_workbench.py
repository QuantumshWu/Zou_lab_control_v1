"""Current W2 PulseWorkbench product and lifecycle oracles."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.notebook.facade import PulseFacade
from Zou_lab_control.workbench import (
    open_offline_pulse_workbench,
    open_pulse_workbench,
)
from zlc_frontend.qt_widgets import FluentFormGrid, GREEN, ORANGE
from zlc_neutral_atom.runtime.run import RunState
from zlc_pulse import (
    FIELD_DELAY,
    FIELD_DURATION,
    PORT_DAC,
    PORT_DIGITAL,
    ApiParameter,
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    PulseFieldRef,
    PulsePeriod,
    ScanParameter,
    save_pulse_document,
)


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    value = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("w2-pulse-workspace"),
    )
    yield value
    value.close()


def _until(application, predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _close(application, window) -> None:
    window.request_close(discard_unsaved=True)
    _until(application, lambda: not window.isVisible(), timeout=5.0)
    assert window.worker_idle
    assert window not in getattr(application, "_zlc_retained_windows", ())


def _document(experiment) -> PulseDocument:
    target = experiment.pulse.target
    return PulseDocument(
        "Workbench pulse",
        target.target,
        target.time_step_ns,
        (
            PulsePeriod(
                "idle",
                100,
                "ns",
                "idle",
                tuple(0 for _ in target.target.raw_lanes),
            ),
        ),
        visible_ports=tuple(
            port.key
            for port in target.target.ports
            if port.kind in (PORT_DIGITAL, PORT_DAC)
        ),
    )


def test_edit_preview_run_hold_stop_and_experiment_survives_window(
    experiment,
    application,
):
    document = _document(experiment)
    window = open_pulse_workbench(experiment, document)
    try:
        preview = window.findChild(QtWidgets.QLabel, "pulsePreviewStatus")
        status = window.findChild(QtWidgets.QLabel, "pulseStatus")
        name = window.findChild(QtWidgets.QLineEdit, "pulseDocumentName")
        run_once = window.findChild(QtWidgets.QPushButton, "runOnceButton")
        hold = window.findChild(QtWidgets.QPushButton, "holdButton")
        stop = window.findChild(QtWidgets.QPushButton, "stopButton")
        assert run_once._base_color == QtGui.QColor(GREEN).name(QtGui.QColor.HexRgb)
        assert stop._base_color == QtGui.QColor(ORANGE).name(QtGui.QColor.HexRgb)
        repeat_form = window.findChild(FluentFormGrid, "pulseRepeatForm")
        assert repeat_form is window._repeat_form
        assert repeat_form.grid.rowCount() == 3
        available = application.primaryScreen().availableGeometry()
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        assert available.contains(window.frameGeometry())
        _until(application, lambda: preview.text().startswith("Preview: READY"))
        assert window.timeline is not None

        name.setText("Edited next revision")
        QtTest.QTest.keyClick(name, QtCore.Qt.Key_Return)
        _until(application, lambda: window.current_document.name == "Edited next revision")
        _until(application, lambda: preview.text().startswith("Preview: READY editor rev 1"))

        QtTest.QTest.mouseClick(run_once, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.active_snapshot is not None
            and window.active_snapshot.state is RunState.SUCCEEDED
            and run_once.isEnabled(),
        )
        assert "SUCCEEDED" in status.text() and "SAFE" in status.text()
        assert not window.active_snapshot.final_committed
        assert available.contains(window.frameGeometry())

        QtTest.QTest.mouseClick(hold, QtCore.Qt.LeftButton)
        _until(application, lambda: "HOLDING" in status.text())
        name.setText("Edited while HOLD runs")
        QtTest.QTest.keyClick(name, QtCore.Qt.Key_Return)
        _until(application, lambda: "editor rev 2 modified" in status.text())
        assert stop.isEnabled()
        QtTest.QTest.mouseClick(stop, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.active_snapshot is not None
            and window.active_snapshot.state is RunState.CANCELLED
            and hold.isEnabled(),
        )
        assert "STOPPED" in status.text() and "SAFE" in status.text()
    finally:
        _close(application, window)

    request = experiment.pulse.request(document)
    assert experiment.pulse.inspect(request).scan_point_count == 0


def test_scan_and_api_values_are_explicit_without_mutating_editor_intent(
    experiment,
    application,
):
    document = _document(experiment)
    digital = next(
        port.key for port in document.target.ports if port.kind == PORT_DIGITAL
    )
    duration = PulseFieldRef(FIELD_DURATION, "idle")
    delay = PulseFieldRef(FIELD_DELAY, None, digital)
    document = replace(
        document,
        scan_parameters=(
            ScanParameter("idle_duration", duration, "Idle duration", "ns"),
        ),
        scan_table=FrozenScanTable(("idle_duration",), ((100,), (200,))),
        delays=(OutputDelay(digital, 0, "ns"),),
        api_parameters=(ApiParameter("trigger_delay", delay, "ns"),),
    )
    window = open_pulse_workbench(experiment, document)
    try:
        preview = window.findChild(QtWidgets.QLabel, "pulsePreviewStatus")
        run_once = window.findChild(QtWidgets.QPushButton, "runOnceButton")
        run_scan = window.findChild(QtWidgets.QPushButton, "runScanButton")
        hold = window.findChild(QtWidgets.QPushButton, "holdButton")
        nominal = window.findChild(QtWidgets.QCheckBox, "nominalReferenceCheck")
        api = window.findChild(QtWidgets.QTableWidget, "pulseApiTable")
        _until(application, lambda: preview.text().startswith("Preview: READY"))
        assert "nominal scan/API reference" in preview.text()
        assert not run_once.isEnabled() and not run_scan.isEnabled() and not hold.isEnabled()

        api.item(0, 3).setText("20")
        api.item(0, 4).setCheckState(QtCore.Qt.Checked)
        _until(application, run_scan.isEnabled)
        assert not run_once.isEnabled() and not hold.isEnabled()
        nominal.setChecked(True)
        _until(application, lambda: run_once.isEnabled() and hold.isEnabled())

        QtTest.QTest.mouseClick(run_scan, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.active_snapshot is not None
            and window.active_snapshot.state is RunState.SUCCEEDED
            and run_scan.isEnabled(),
        )
        assert window.current_document.api_parameters == document.api_parameters
        assert window.current_document.scan_table == document.scan_table
    finally:
        _close(application, window)


def test_authoritative_delay_editor_preserves_the_full_stored_float(
    experiment,
    application,
):
    document = _document(experiment)
    digital = next(
        port.key for port in document.target.ports if port.kind == PORT_DIGITAL
    )
    exact_value = -123456789012340.12
    document = replace(
        document,
        delays=(OutputDelay(digital, exact_value, "us"),),
    )
    window = open_pulse_workbench(experiment, document)
    try:
        field = window._delay_rows[digital][1]
        assert field.value() == exact_value
    finally:
        _close(application, window)


def test_failed_start_never_claims_safe(
    experiment,
    application,
    monkeypatch,
):
    def failed_start(self, request):
        raise RuntimeError("admission rejected")

    monkeypatch.setattr(PulseFacade, "start", failed_start)
    window = open_pulse_workbench(experiment, _document(experiment))
    try:
        status = window.findChild(QtWidgets.QLabel, "pulseStatus")
        run_once = window.findChild(QtWidgets.QPushButton, "runOnceButton")
        _until(application, run_once.isEnabled)
        QtTest.QTest.mouseClick(run_once, QtCore.Qt.LeftButton)
        _until(application, lambda: "FAILED BEFORE ADMISSION" in status.text())
        assert "SAFE" not in status.text()
        assert run_once.isEnabled()
    finally:
        _close(application, window)


def test_save_freezes_the_editor_and_blocks_session_replacement(
    experiment,
    application,
    tmp_path,
    monkeypatch,
):
    import zlc_workbench.pulse as pulse_module

    document = replace(_document(experiment), name="saved session")
    replacement = replace(document, name="must not replace during save")
    replacement_path = save_pulse_document(replacement, tmp_path / "replacement")
    destination = tmp_path / "blocked-save"
    entered = threading.Event()
    release = threading.Event()
    original_save = pulse_module.save_pulse_document

    def blocking_save(value, path):
        entered.set()
        assert release.wait(2.0)
        return original_save(value, path)

    monkeypatch.setattr(pulse_module, "save_pulse_document", blocking_save)
    window = open_pulse_workbench(experiment, document)
    session = window.editor_session
    try:
        window.save_path(destination)
        _until(application, entered.is_set)
        assert not window._new_action.isEnabled()
        assert not window._open_action.isEnabled()
        with pytest.raises(RuntimeError, match="another operation"):
            window.open_path(replacement_path)
        window._new_document()
        assert window.editor_session is session

        release.set()
        _until(application, lambda: window.editor_session.path is not None)
        assert window.editor_session is session
        assert window.current_document.name == "saved session"
        assert pulse_module.load_pulse_document(destination).name == "saved session"
    finally:
        release.set()
        _close(application, window)


def test_close_before_start_returns_is_nonblocking_and_reaps_the_run(
    experiment,
    application,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    original_start = PulseFacade.start

    def delayed_start(self, request):
        entered.set()
        assert release.wait(2.0)
        return original_start(self, request)

    monkeypatch.setattr(PulseFacade, "start", delayed_start)
    document = _document(experiment)
    window = open_pulse_workbench(experiment, document)
    hold = window.findChild(QtWidgets.QPushButton, "holdButton")
    _until(application, hold.isEnabled)
    QtTest.QTest.mouseClick(hold, QtCore.Qt.LeftButton)
    _until(application, entered.is_set)
    began = time.monotonic()
    window.request_close(discard_unsaved=True)
    assert time.monotonic() - began < 0.1
    release.set()
    _until(application, lambda: not window.isVisible(), timeout=5.0)
    assert window.worker_idle
    assert experiment.pulse.inspect(experiment.pulse.request(document)).scan_point_count == 0


def test_load_generation_rejects_an_old_revision_zero_preview(
    experiment,
    application,
    tmp_path,
    monkeypatch,
):
    import Zou_lab_control.workbench._pulse as pulse_window_module

    first = replace(_document(experiment), name="old revision zero")
    second = replace(_document(experiment), name="loaded revision zero")
    path = save_pulse_document(second, tmp_path / "loaded")
    entered = threading.Event()
    release = threading.Event()
    original_project = pulse_window_module.project_pulse_preview
    calls = 0

    def delayed_project(document):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(2.0)
        return original_project(document)

    monkeypatch.setattr(pulse_window_module, "project_pulse_preview", delayed_project)
    window = open_pulse_workbench(experiment, first)
    try:
        _until(application, entered.is_set)
        window.open_path(path)
        _until(application, lambda: window.current_document.name == "loaded revision zero")
        release.set()
        _until(
            application,
            lambda: window.timeline is not None
            and window.timeline.title == "loaded revision zero",
        )
        assert calls == 2
    finally:
        release.set()
        _close(application, window)


def test_offline_and_public_import_have_no_hardware_or_qt_backdoor(
    application,
    experiment,
):
    descriptor = experiment.pulse.target
    window = open_offline_pulse_workbench(
        descriptor.target,
        time_step_ns=descriptor.time_step_ns,
    )
    try:
        _until(application, lambda: window.timeline is not None)
        assert not window.findChild(QtWidgets.QPushButton, "runOnceButton").isEnabled()
        assert window.findChild(QtWidgets.QLabel, "pulseStatus").text().startswith(
            "Pulse: OFFLINE"
        )
    finally:
        _close(application, window)

    code = (
        "import sys; import Zou_lab_control.workbench, zlc_workbench; "
        "assert not any(n == 'PyQt5' or n.startswith('PyQt5.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    source = (ROOT / "Zou_lab_control" / "workbench" / "_pulse.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "BoundPulsePort",
        "RunPlan",
        "RemoteSequencer",
        "_prepare_pulse_for_workbench",
        "._authority_token",
        ".fire(",
        ".safe(",
    ):
        assert forbidden not in source


def test_standalone_virtual_launcher_owns_and_closes_its_experiment(tmp_path):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["ZLC_PULSE_GUI_AUTO_CLOSE_MS"] = "20"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pulse_gui.py"),
            "--repository",
            str(tmp_path / "standalone"),
        ],
        cwd=ROOT,
        env=environment,
        timeout=15,
        check=True,
    )
