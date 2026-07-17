"""M1 current free-running camera MonitorDataset product oracles."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import MONITOR_HISTORY, SPATIAL_X, SPATIAL_Y
from zlc_frontend.qt_widgets import QtImageBoard
from zlc_neutral_atom.runtime.run import RunState
from zlc_neutral_atom.bootstrap._virtual_hardware import (
    VirtualMonitorCamera,
    VirtualSequencer,
)
from zlc_pulse import (
    PulseExecutionForm,
    build_pulse_playback,
    compile_pulse_artifact,
    load_pulse_document,
)


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    exp = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("m1-camera-monitor"),
    )
    yield exp
    exp.close()


def _until(application, predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def _widgets(window):
    return (
        window.findChild(QtWidgets.QPushButton, "startButton"),
        window.findChild(QtWidgets.QPushButton, "stopButton"),
        window.findChild(QtWidgets.QLabel, "monitorStatus"),
        window.findChild(QtWidgets.QLabel, "monitorViewStatus"),
        window.findChild(QtWidgets.QLabel, "diagnostics"),
        window.findChild(QtImageBoard, "cameraMonitorImageBoard"),
    )


def _close_window(application, window) -> None:
    window.close()
    _until(application, lambda: not window.isVisible(), timeout=5.0)
    assert window not in getattr(application, "_zlc_retained_windows", ())


def test_public_monitor_preserves_axes_restarts_with_fresh_identity_and_clears_front(
    experiment,
    application,
):
    request = experiment.readout.camera_monitor_request()
    descriptor = experiment.readout.inspect_camera_monitor(request)
    assert descriptor.camera_role == "monitor_camera"
    assert descriptor.output_shape == (1, 1, 1200, 1920)

    window = experiment.readout.camera_monitor_gui(request)
    first_handle = None
    second_handle = None
    try:
        start, stop, status, view, diagnostics, board = _widgets(window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: board.has_front and view.text().startswith("View: LIVE"))
        first_handle = window._handle
        first = window._slot.freeze_current()[2]
        schema = first.snapshot.block.schema
        assert schema.physical_shape == (1, 1, 1200, 1920)
        assert tuple(axis.role for axis in schema.point_axes) == (MONITOR_HISTORY,)
        assert tuple(axis.role for axis in schema.cell_schema.data_axes) == (
            SPATIAL_Y,
            SPATIAL_X,
        )
        assert first.head is not None and first.event_refs == (first.head,)
        assert first.coverage.written_cells == 1
        assert first.coverage.total_cells == 1
        assert not window._slot._dataset._source._reservations
        first_identity = (
            first.snapshot.ref.block_id,
            first.snapshot.ref.stream_generation,
        )

        QtTest.QTest.mouseClick(stop, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: first_handle.snapshot().state is RunState.CANCELLED,
        )
        _until(application, lambda: "STOPPED" in status.text())
        _until(application, lambda: not board.has_front)
        _until(application, start.isEnabled)
        assert diagnostics.text().startswith("Stop: REQUESTED")

        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: board.has_front and view.text().startswith("View: LIVE"))
        second_handle = window._handle
        second = window._slot.freeze_current()[2]
        second_identity = (
            second.snapshot.ref.block_id,
            second.snapshot.ref.stream_generation,
        )
        assert second_identity != first_identity
        assert second.head is not None and second.event_refs == (second.head,)
        assert "missed=" in view.text() and "current_gap=" in view.text()
    finally:
        if window.isVisible():
            _close_window(application, window)
    assert first_handle is not None and first_handle.snapshot().state is RunState.CANCELLED
    assert second_handle is not None and second_handle.snapshot().state is RunState.CANCELLED
    assert experiment.readout.inspect_camera_monitor(request).output_shape == (
        1,
        1,
        1200,
        1920,
    )


def test_total_memory_rejection_happens_before_start_and_next_window_remains_usable(
    experiment,
    application,
):
    bad = experiment.readout.camera_monitor_gui(memory_limit_bytes=1)
    good = None
    try:
        start, _stop, status, _view, diagnostics, board = _widgets(bad)
        _until(application, lambda: "base peak" in diagnostics.text())
        assert status.text() == "Monitor: NOT READY"
        assert not start.isEnabled() and not board.has_front
        _close_window(application, bad)

        good = experiment.readout.camera_monitor_gui()
        start, _stop, _status, view, _diagnostics, board = _widgets(good)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: board.has_front and view.text().startswith("View: LIVE"))
    finally:
        if bad.isVisible():
            _close_window(application, bad)
        if good is not None and good.isVisible():
            _close_window(application, good)


def test_monitor_public_boundary_stays_typed_and_workbench_has_no_raw_authority():
    root = Path(__file__).parents[1]
    source = (root / "Zou_lab_control" / "workbench" / "_camera_monitor.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "RunPlan",
        "BoundCameraMonitorPort",
        "CameraMonitorEndpoint",
        "VirtualMonitorCamera",
        "_authority_token",
        "camera_monitor_port",
    ):
        assert forbidden not in source


def test_virtual_monitor_preserves_main_mot_coil_physics_and_sensor_clock():
    document = load_pulse_document(Path(__file__).parents[1] / "pulses" / "mot_field_template.json")
    wanted = {"da_bias_x": 7, "da_bias_y": -5, "da_bias_z": 11}
    document = replace(
        document,
        periods=tuple(
            replace(
                period,
                analog_steps=tuple(
                    replace(step, value=wanted.get(step.port, step.value))
                    for step in period.analog_steps
                ),
            )
            for period in document.periods
        ),
    )
    clock_hz = 1e9 / document.time_step_ns
    artifact = compile_pulse_artifact(
        document,
        clock_hz=clock_hz,
        execution_form=PulseExecutionForm.STATIC_REFERENCE_POINT,
        live_target=document.target,
    )
    sequencer = VirtualSequencer(document.target, clock_hz=clock_hz, sleep_scale=0.0)
    camera = VirtualMonitorCamera(
        sequencer,
        frame_shape=(60, 96),
        exposure=0.01,
        timeout=1.0,
        seed=4,
    )
    try:
        camera.arm(None, max_inflight_frames=2, timeout=1.0)
        safe = camera.read_frame_records(1, timeout=1.0, exact=False)[0].image
        assert camera.last_levels == {"da_x": 0.0, "da_y": 0.0, "da_z": 0.0}
        camera.finish_record_capture()

        playback = build_pulse_playback(artifact, name="mot-optimum")
        sequencer.prepare_compiled_playback(artifact, playback)
        sequencer.fire_compiled_playback(artifact.fingerprint)
        camera.arm(None, max_inflight_frames=2, timeout=1.0)
        optimum = camera.read_frame_records(1, timeout=1.0, exact=False)[0].image
        assert camera.last_levels == {"da_x": 7.0, "da_y": -5.0, "da_z": 11.0}
        assert camera.mot_efficiency(camera.last_levels) == pytest.approx(1.0)
        assert float(optimum.mean()) > float(safe.mean()) + 5.0
        camera.finish_record_capture()

        sequencer.set_safe_state()
        camera.arm(None, max_inflight_frames=2, timeout=1.0)
        camera.read_frame_records(1, timeout=1.0, exact=False)
        assert camera.last_levels == {"da_x": 0.0, "da_y": 0.0, "da_z": 0.0}
    finally:
        camera.close()
        sequencer.close()


def test_virtual_monitor_camera_is_sensor_clocked_and_never_silently_overwrites():
    # Isolate the deliberately overflowing source worker from the Qt process;
    # the oracle is adapter behavior, not pytest/Qt teardown ordering.
    code = r'''
import time
from zlc_neutral_atom.bootstrap._virtual_hardware import VirtualMonitorCamera, VirtualSequencer
from zlc_pulse import load_deployed_pulse_target

sequencer = VirtualSequencer(load_deployed_pulse_target(), clock_hz=1e8, sleep_scale=0.0)
camera = VirtualMonitorCamera(sequencer, frame_shape=(12, 20), exposure=0.002)
try:
    camera.arm(None, max_inflight_frames=1, timeout=1.0)
    time.sleep(0.03)
    first = camera.read_frame_records(1, timeout=1.0, exact=False)
    assert len(first) == 1 and first[0].image.shape == (12, 20)
    assert first[0].source_ordinal == 0
    try:
        camera.read_frame_records(1, timeout=1.0, exact=False)
    except RuntimeError as error:
        assert "monitor source failed" in str(error)
    else:
        raise AssertionError("bounded monitor queue silently overwrote")
finally:
    terminal = camera.finish_record_capture()
    camera.close()
    sequencer.close()
assert terminal.source_stopped and terminal.no_more_frames and terminal.joined
'''
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        check=True,
    )
