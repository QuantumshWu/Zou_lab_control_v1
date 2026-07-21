"""M1 current free-running camera MonitorDataset product oracles."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from zlc_data import MONITOR_HISTORY, SPATIAL_X, SPATIAL_Y
from zlc_frontend.display_range import RelimMode
from zlc_frontend.image_display import ImageColormap
from zlc_frontend.qt_widgets import ensure_qt_app  # noqa: F401
from zlc_frontend.qt_widgets import QtRasterBoard
from zlc_frontend.render import PixelFormat
from zlc_frontend.selector import ImageViewportCommit
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
    return ensure_qt_app()


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
        window.findChild(QtRasterBoard, "cameraMonitorImageBoard"),
    )


def _raw_image_binding(board: QtRasterBoard):
    assert tuple(board._image_bindings) == ("camera-monitor-image",)
    return board._image_bindings["camera-monitor-image"]


def _close_window(application, window) -> None:
    window.close()
    _until(application, lambda: not window.isVisible(), timeout=5.0)
    assert window not in getattr(application, "_zlc_retained_windows", ())


def _start_live_window(experiment, application):
    window = experiment.readout.camera_monitor_gui()
    start, _stop, _status, view, _diagnostics, board = _widgets(window)
    _until(application, start.isEnabled)
    QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
    _until(application, lambda: board.has_front and view.text().startswith("View: LIVE"))
    return window


def _live_display_identity(window) -> tuple[object, ...]:
    """Freeze identities that a display-only commit is forbidden to replace."""

    handle = window._handle
    slot = window._slot
    controller = window._board
    board = window._board_widget
    assert handle is not None and slot is not None and controller is not None
    assert isinstance(board, QtRasterBoard)
    front = board.front_frame
    assert front is not None
    model = controller.model
    return (
        handle,
        handle.snapshot().run_id,
        window._run_owner.generation,
        slot,
        slot.dataset_id,
        controller,
        model.board_id,
        model.layout_generation,
        model.panels,
        board,
        tuple(panel.panel_id for panel in front.panels),
        front.panels[0].source_identity,
    )


def _raw_revision(window):
    assert window._slot is not None
    return window._slot.freeze_current()[2].snapshot.ref.revision


def _choose_image_display_value(editor, key: str, value: object) -> None:
    widget = editor._form.widget_for(key)
    assert isinstance(widget, QtWidgets.QComboBox)
    index = widget.findData(value)
    assert index >= 0
    widget.setCurrentIndex(index)
    # Choice handlers intentionally treat only a user activation as an edit;
    # this helper supplies that one public form notification after selecting.
    editor._form.changed.emit(key)


def _wheel_image_down(board: QtRasterBoard) -> QtGui.QWheelEvent:
    target = board._selector_target()[0]
    position = target.center()
    event = QtGui.QWheelEvent(
        QtCore.QPointF(position),
        QtCore.QPointF(board.mapToGlobal(position)),
        QtCore.QPoint(),
        QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    board.wheelEvent(event)
    return event


def test_public_monitor_preserves_axes_restarts_with_fresh_identity_and_clears_front(
    experiment,
    application,
):
    request = experiment.readout.camera_monitor_request()
    descriptor = experiment.readout.inspect_camera_monitor(request)
    assert descriptor.camera_role == "monitor_camera"
    assert descriptor.output_shape == (1, 8, 1200, 1920)

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
        assert schema.physical_shape == (1, 8, 1200, 1920)
        assert tuple(axis.role for axis in schema.point_axes) == (MONITOR_HISTORY,)
        assert tuple(axis.role for axis in schema.cell_schema.data_axes) == (
            SPATIAL_Y,
            SPATIAL_X,
        )
        assert first.head is not None and first.event_refs[0] == first.head
        assert len(first.event_refs) == 8
        assert 1 <= first.coverage.written_cells <= 8
        assert first.coverage.total_cells == 8
        assert not window._slot._dataset.raw._source._reservations
        visible = board.front_frame
        assert visible is not None
        image_panel = visible.panels[0]
        payload = image_panel.display_payload
        assert image_panel.raster.pixel_format is PixelFormat.INDEXED8
        assert payload is not None
        assert payload.evaluated_input in image_panel.coherence_stamp.inputs
        assert payload.viewport.viewport_revision == 0
        assert payload.image.values.shape == (1200, 1920)
        assert payload.image.values.flags.writeable is False
        assert window._live.front_status.image_display_revision == 0
        assert window._live.front_status.image_color_limits is not None
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
        assert not view.text().startswith("View: FAILED")
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
        assert second.head is not None and second.event_refs[0] == second.head
        assert "missed=" in view.text() and "current_gap=" in view.text()
    finally:
        if window.isVisible():
            _close_window(application, window)
    assert first_handle is not None and first_handle.snapshot().state is RunState.CANCELLED
    assert second_handle is not None and second_handle.snapshot().state is RunState.CANCELLED
    assert experiment.readout.inspect_camera_monitor(request).output_shape == (
        1,
        8,
        1200,
        1920,
    )


def test_display_gesture_and_form_reconfigure_only_the_live_presentation(
    experiment,
    application,
):
    window = _start_live_window(experiment, application)
    try:
        board = _widgets(window)[-1]
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "roiSelectorSwitch",
        )
        assert selector is not None
        _until(application, selector.isEnabled)
        stable_identity = _live_display_identity(window)
        first_raw_revision = _raw_revision(window)

        selector.setChecked(True)
        assert _wheel_image_down(board).isAccepted()
        gesture_revision = window._image_display.revision
        assert gesture_revision == 1
        # The old pixels remain visible while the worker evaluates r1, but
        # they cannot author another gesture.  User intent remains checked and
        # is restored automatically when the exact r1 front is painted.
        assert selector.isChecked() and not selector.isEnabled()
        assert not board._selector_enabled
        assert board._image_interaction_is_pending(_raw_image_binding(board))
        assert board.visible_image_origin().presentation.panel_revision == 0
        _until(
            application,
            lambda: (
                board.visible_image_origin() is not None
                and board.visible_image_origin().presentation.panel_revision
                == gesture_revision
            ),
        )
        _until(application, selector.isEnabled)
        assert selector.isChecked() and board._selector_enabled
        assert not board._image_interaction_is_pending(_raw_image_binding(board))
        assert board.selector_fault is None
        _until(application, lambda: _raw_revision(window) > first_raw_revision)
        assert _live_display_identity(window) == stable_identity

        second_raw_revision = _raw_revision(window)
        editor = window._edit_image_display
        _choose_image_display_value(editor, "colormap", ImageColormap.VIRIDIS)
        editor._apply_button.click()
        form_revision = window._image_display.revision
        assert form_revision == gesture_revision + 1
        assert window._image_display.colormap is ImageColormap.VIRIDIS
        _until(
            application,
            lambda: (
                board.visible_image_origin() is not None
                and board.visible_image_origin().presentation.panel_revision
                == form_revision
            ),
        )
        _until(application, lambda: _raw_revision(window) > second_raw_revision)
        assert _live_display_identity(window) == stable_identity
    finally:
        _close_window(application, window)


def test_stale_painted_origin_and_live_refusal_never_publish_owner_display(
    experiment,
    application,
    monkeypatch,
):
    window = _start_live_window(experiment, application)
    try:
        board = _widgets(window)[-1]
        origin = board.visible_image_origin()
        viewport = window._image_viewport
        live = window._live
        assert origin is not None and viewport is not None and live is not None
        owner_state = window._image_display
        owner_viewport = viewport
        candidate = viewport.centered_zoom(
            (0.5, 0.5),
            0.75,
            viewport_revision=owner_state.revision + 1,
        )

        stale_origin = replace(origin, sequence=origin.sequence + 1)
        with pytest.raises(RuntimeError, match="no longer painted"):
            window._accept_image_interaction(
                ImageViewportCommit(stale_origin, candidate)
            )
        assert window._image_display is owner_state
        assert window._image_viewport is owner_viewport

        def refuse_reconfigure(_state, _viewport) -> None:
            raise RuntimeError("synchronous live display refusal")

        monkeypatch.setattr(live, "reconfigure_image_display", refuse_reconfigure)
        with pytest.raises(RuntimeError, match="synchronous live display refusal"):
            window._accept_image_interaction(
                ImageViewportCommit(origin, candidate)
            )
        assert window._image_display is owner_state
        assert window._image_viewport is owner_viewport
        assert live._configuration.image_display is owner_state
        assert live._configuration.image_viewport is owner_viewport
    finally:
        _close_window(application, window)


def test_entering_fixed_freezes_visible_painted_payload_not_ahead_status(
    experiment,
    application,
):
    window = _start_live_window(experiment, application)
    try:
        board = _widgets(window)[-1]
        payload = board.visible_image_payload()
        live = window._live
        assert payload is not None and live is not None
        painted_limits = payload.color_limits
        status = live.front_status
        assert status is not None
        ahead_limits = (
            painted_limits[0] - 12345.0,
            painted_limits[1] + 12345.0,
        )
        assert ahead_limits != painted_limits
        with live._lock:
            live._front_status = replace(
                status,
                sequence=status.sequence + 1,
                image_color_limits=ahead_limits,
            )

        editor = window._setting_image_display
        values = editor._form.read_all()
        values["relim_mode"] = RelimMode.FIXED
        window._apply_image_display_form(
            editor,
            editor.base_revision,
            values,
        )

        assert window._image_display.relim_mode is RelimMode.FIXED
        assert window._image_display.fixed_color_limits == painted_limits
        assert window._image_display.fixed_color_limits != ahead_limits
    finally:
        _close_window(application, window)


def test_setting_and_edit_share_spec_and_one_owner_cas_workflow(
    experiment,
    application,
):
    window = experiment.readout.camera_monitor_gui()
    try:
        start = _widgets(window)[0]
        _until(application, start.isEnabled)
        edit = window._edit_image_display
        setting = window._setting_image_display
        assert edit._form.spec is setting._form.spec
        assert edit._form.keys == setting._form.keys
        assert edit.base_revision == setting.base_revision == 0

        from zlc_frontend.qt_widgets import FluentPopup

        setting_button = window.findChild(
            QtWidgets.QPushButton,
            "displaySettingButton",
        )
        assert setting_button is not None
        setting_button.click()
        application.processEvents()
        popup = window._settings_popup
        assert isinstance(popup, FluentPopup) and popup.isVisible()
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            assert popup.width() <= int(available.width() * 0.55)
            assert popup.height() <= int(available.height() * 0.90)
            assert available.contains(popup.frameGeometry())
        button_top = setting_button.mapToGlobal(QtCore.QPoint(0, 0)).y()
        button_bottom = setting_button.mapToGlobal(
            QtCore.QPoint(0, setting_button.height() - 1)
        ).y()
        assert popup.frameGeometry().bottom() < button_top or popup.y() > button_bottom
        popup.hide()
        application.processEvents()

        # Keep one unapplied Setting draft while Edit commits from the same
        # optimistic base.  The submitting surface cleans; the other is stale.
        setting._form.widget_for("x_min").setText("10")
        assert setting._dirty and not setting._stale
        setting_draft = setting._form.read_all()
        _choose_image_display_value(edit, "colormap", ImageColormap.MAGMA)
        edit._apply_button.click()

        assert window._image_display.revision == 1
        assert window._image_display.colormap is ImageColormap.MAGMA
        assert edit.base_revision == 1 and not edit._dirty and not edit._stale
        assert setting.base_revision == 0 and setting._dirty and setting._stale
        assert setting._form.read_all() == setting_draft

        setting._cancel_button.click()
        assert setting.base_revision == 1
        assert not setting._dirty and not setting._stale
        assert setting._form.read_all() == edit._form.read_all()

        # Applying the exact current projection is a semantic no-op: it
        # acknowledges the submitter without manufacturing another revision.
        before = window._image_display
        edit._apply_button.click()
        assert window._image_display is before
        assert edit.base_revision == 1 and not edit._dirty and not edit._stale
    finally:
        _close_window(application, window)


def test_stop_start_preserves_authored_display_and_resets_selector_off(
    experiment,
    application,
):
    window = experiment.readout.camera_monitor_gui()
    first_handle = None
    try:
        start, stop, status, view, _diagnostics, board = _widgets(window)
        _until(application, start.isEnabled)
        editor = window._edit_image_display
        _choose_image_display_value(editor, "colormap", ImageColormap.PLASMA)
        editor._apply_button.click()
        authored = window._image_display
        assert authored.revision == 1

        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: board.has_front and view.text().startswith("View: LIVE"))
        first_handle = window._handle
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "roiSelectorSwitch",
        )
        assert selector is not None
        _until(application, selector.isEnabled)
        selector.setChecked(True)
        assert selector.isChecked()

        QtTest.QTest.mouseClick(stop, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: first_handle.snapshot().state is RunState.CANCELLED,
        )
        _until(application, lambda: "STOPPED" in status.text() and not board.has_front)
        _until(application, start.isEnabled)
        assert window._image_display == authored
        assert not selector.isChecked()

        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: board.has_front and view.text().startswith("View: LIVE"))
        payload = board.visible_image_payload()
        assert payload is not None
        assert window._image_display == authored
        assert payload.viewport.viewport_revision == authored.revision
        assert not selector.isChecked()
    finally:
        _close_window(application, window)
    assert first_handle is not None
    assert first_handle.snapshot().state is RunState.CANCELLED


def test_changed_display_commit_recovers_stopped_not_ready_preparation(
    experiment,
    application,
):
    window = experiment.readout.camera_monitor_gui()
    try:
        start = _widgets(window)[0]
        _until(application, start.isEnabled)
        editor = window._edit_image_display

        # Author a real coordinate pin, then reproduce the exact stopped state
        # left when a newly discovered schema rejects that retained pin.
        editor._form.widget_for("x_min").setText("10")
        editor._form.widget_for("x_max").setText("100")
        editor._apply_button.click()
        assert window._image_display.x_view == (10.0, 100.0)
        assert window._prepared is not None and window._live is None
        window._prepared = None
        window._prepare_inflight = False
        window._run_status.setText("Monitor: NOT READY")
        window._record_local_failure(
            "Preparation failed: retained x_view lies outside the new schema"
        )
        window._update_start_button()
        assert not start.isEnabled()

        # Clearing the rejected pin is one changed owner CAS.  It retries the
        # existing prepare path; no source Run is armed as a side effect.
        editor = window._edit_image_display
        editor._form.widget_for("x_min").clear()
        editor._form.widget_for("x_max").clear()
        editor._apply_button.click()
        assert window._image_display.x_view is None
        assert window._handle is None and window._live is None
        assert window._prepare_inflight
        assert window._run_status.text() == "Monitor: PREPARING"
        _until(application, lambda: window._prepared is not None and start.isEnabled())
        assert window._prepared.viewport.visible_bounds == (0.0, 0.0, 1.0, 1.0)
        assert window._prepared.viewport.viewport_revision == window._image_display.revision
        assert window._handle is None and window._live is None
    finally:
        _close_window(application, window)


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
