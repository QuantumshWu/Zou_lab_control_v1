"""Current W1 finite exact-capture Workbench product oracles."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.notebook.facade import _prepare_capture_for_workbench
from Zou_lab_control.workbench import open_capture_workbench
from zlc_data import BlockId, SPATIAL_X, SPATIAL_Y
from zlc_frontend.figure import DatasetId
from zlc_frontend.display_range import RelimMode
from zlc_frontend.image_display import ImageColormap
from zlc_frontend.qt_widgets import ensure_qt_app  # noqa: F401
from zlc_frontend.qt_widgets import GREEN, ORANGE, QtRasterBoard
from zlc_neutral_atom.monitor_application import (
    CameraMonitorLiveDataset,
    CameraMonitorRoiState,
)
from zlc_neutral_atom.runtime.control import ControlAckStatus, create_control_topic
import zlc_workbench.live as live_module
from zlc_workbench.live import LiveBoardController, LiveDatasetSlot


ROOT = Path(__file__).parents[1]
SINGLE_EVENT_PULSE = ROOT / "pulses" / "probe_template.json"
MULTI_EVENT_PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    exp = zlc.connect(
        "virtual",
        repository=tmp_path_factory.mktemp("w1-workspace"),
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
        window.findChild(QtWidgets.QLabel, "captureStatus"),
        window.findChild(QtWidgets.QLabel, "previewStatus"),
        window.findChild(QtWidgets.QLabel, "diagnostics"),
        window.findChild(QtRasterBoard, "captureImageBoard"),
    )


def _capture_image_binding(board: QtRasterBoard):
    assert tuple(board._image_bindings) == ("capture-image",)
    return board._image_bindings["capture-image"]


def _image_target(board: QtRasterBoard) -> QtCore.QRect:
    target = board._selector_target(_capture_image_binding(board))
    assert target is not None
    return target[0]


def _point(target: QtCore.QRect, x_fraction: float, y_fraction: float) -> QtCore.QPoint:
    return QtCore.QPoint(
        target.left() + int(x_fraction * max(1, target.width() - 1)),
        target.top() + int(y_fraction * max(1, target.height() - 1)),
    )


def _drag_move(board: QtRasterBoard, position: QtCore.QPoint, button) -> None:
    board.mouseMoveEvent(
        QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(position),
            QtCore.Qt.NoButton,
            button,
            QtCore.Qt.NoModifier,
        )
    )


def _wheel_image_down(board: QtRasterBoard) -> QtGui.QWheelEvent:
    position = _image_target(board).center()
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


def _choose_image_display_value(editor, key: str, value: object) -> None:
    widget = editor._form.widget_for(key)
    assert isinstance(widget, QtWidgets.QComboBox)
    index = widget.findData(value)
    assert index >= 0
    widget.setCurrentIndex(index)
    editor._form.changed.emit(key)


def _close_window(application, window) -> None:
    window.close()
    _until(application, lambda: not window.isVisible(), timeout=5.0)
    assert window not in getattr(application, "_zlc_retained_windows", ())


def test_public_workbench_repeats_with_new_preview_and_loads_exact_artifact(
    experiment,
    application,
):
    window = None
    first = None
    second = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, capture, preview, diagnostics, board = _widgets(window)
        stop = window.findChild(QtWidgets.QPushButton, "stopButton")
        assert start._base_color == QtGui.QColor(GREEN).name(QtGui.QColor.HexRgb)
        assert stop._base_color == QtGui.QColor(ORANGE).name(QtGui.QColor.HexRgb)
        available = application.primaryScreen().availableGeometry()
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        assert available.contains(window.frameGeometry())
        _until(application, start.isEnabled)

        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: window.final_reference is not None)
        first = window.final_reference
        assert capture.text().startswith("Capture: FINAL")
        assert preview.text().startswith("Preview: DISPLAY ONLY · capture FINAL")
        assert board.has_front and diagnostics.text() == ""
        assert available.contains(window.frameGeometry())
        _until(application, start.isEnabled)

        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None
            and window.final_reference != first,
        )
        second = window.final_reference
        assert capture.text().startswith("Capture: FINAL")
        assert board.has_front
        _close_window(application, window)
        assert experiment.inspect(request).output_shape == (1, 1, 96, 128)
    finally:
        if window is not None and window.isVisible():
            _close_window(application, window)
    assert first is not None and second is not None and first != second
    artifact = experiment.readout.load_capture(second)
    schema = artifact.frame_source.schema
    assert schema.physical_shape == (1, 1, 96, 128)
    assert tuple(axis.role for axis in schema.cell_schema.data_axes) == (
        SPATIAL_Y,
        SPATIAL_X,
    )


def test_final_preview_exposes_exact_image_handles_without_changing_authority(
    experiment,
    application,
):
    window = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, _capture, _preview, diagnostics, board = _widgets(window)
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "captureSelectorSwitch",
        )
        assert selector is not None and not selector.isChecked()
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None
            and board.has_front
            and selector.isEnabled(),
        )
        final_ref = window.final_reference
        handle = window._handle
        slot = window._slot
        live = window._live
        controller = window._board
        assert handle is not None and slot is not None
        assert live is not None and controller is not None
        model = controller.model
        front = board.front_frame
        payload = board.visible_image_payload()
        assert front is not None and payload is not None
        source = front.panels[0].source_identity
        evaluated_input = payload.evaluated_input

        selector.setChecked(True)
        assert board.selectors_enabled
        target = _image_target(board)
        drag_start = _point(target, 0.20, 0.25)
        drag_end = _point(target, 0.65, 0.70)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=drag_start)
        assert board._selector_hold is not None
        assert board._selector_hold.prepared is board._front[1][0]
        assert board._selector_hold.display_payload is payload
        held_revision = slot.freeze_current()[2].snapshot.ref.revision
        _drag_move(board, drag_end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=drag_end)
        assert board._selector_hold is None
        assert slot.freeze_current()[2].snapshot.ref.revision == held_revision
        assert window._rectangle_candidate is not None
        assert "DISPLAY ONLY" in window._interaction_status.text()

        center = target.center()
        QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=center)
        assert _capture_image_binding(board).cross is not None
        _drag_move(board, center, QtCore.Qt.NoButton)
        assert _capture_image_binding(board).hover is not None
        QtTest.QTest.mouseDClick(board, QtCore.Qt.RightButton, pos=center)
        assert _capture_image_binding(board).cross is None

        event = _wheel_image_down(board)
        assert event.isAccepted()
        _until(
            application,
            lambda: window._image_display.revision == 1
            and board.visible_image_origin() is not None
            and board.visible_image_origin().presentation.panel_revision == 1,
        )

        rail_target = board._clim_rail_target(_capture_image_binding(board))
        assert rail_target is not None
        rail, *_rest, painted = rail_target
        domain = board._color_rail_domain(painted)
        low, high = painted.color_limits
        start_y = board._rail_y(low, domain, rail)
        target_value = low + 0.20 * (high - low)
        end_y = board._rail_y(target_value, domain, rail)
        clim_start = QtCore.QPoint(rail.center().x(), int(round(start_y)))
        clim_end = QtCore.QPoint(rail.center().x(), int(round(end_y)))
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=clim_start)
        _drag_move(board, clim_end, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=clim_end)
        _until(
            application,
            lambda: window._image_display.revision == 2
            and board.visible_image_origin() is not None
            and board.visible_image_origin().presentation.panel_revision == 2,
        )
        assert window._image_display.relim_mode is RelimMode.FIXED

        current = board.front_frame
        current_payload = board.visible_image_payload()
        assert current is not None and current_payload is not None
        assert window.final_reference == final_ref
        assert window._handle is handle
        assert window._slot is slot and window._live is live
        assert window._board is controller and controller.model == model
        assert current.panels[0].source_identity == source
        assert current_payload.evaluated_input == evaluated_input
        assert diagnostics.text() == ""
        assert experiment.readout.load_capture(final_ref).frame_source.schema.physical_shape == (
            1,
            1,
            96,
            128,
        )
    finally:
        if window is not None:
            _close_window(application, window)


def test_capture_setting_edit_share_cas_and_repeat_preserves_only_authored_display(
    experiment,
    application,
):
    window = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, _capture, _preview, diagnostics, board = _widgets(window)
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "captureSelectorSwitch",
        )
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None
            and selector is not None
            and selector.isEnabled(),
        )
        first_ref = window.final_reference
        first_source = board.front_frame.panels[0].source_identity
        first_origin = board.visible_image_origin()
        assert first_origin is not None
        edit = window._edit_image_display
        setting = window._setting_image_display
        assert edit._form.spec is setting._form.spec
        assert edit.base_revision == setting.base_revision == 0

        setting._form.widget_for("x_min").setText("10")
        assert setting._dirty and not setting._stale
        setting_draft = setting._form.read_all()
        selector.setChecked(True)
        hold_point = _point(_image_target(board), 0.40, 0.45)
        QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=hold_point)
        held = board._selector_hold
        assert held is not None
        slot = window._slot
        assert slot is not None
        frozen_before = slot.freeze_current()[2]
        _choose_image_display_value(edit, "colormap", ImageColormap.MAGMA)
        edit._apply_button.click()
        assert window._image_display.revision == 1
        assert window._image_display.colormap is ImageColormap.MAGMA
        # Advancing the target display revision synchronously makes the painted
        # revision unready and releases the old hold before the commit returns.
        # A worker that starts earlier still sees hold/current as one aliased old
        # front.  This is the concrete R1/S1 topology used by the estimator; the
        # immutable source itself never advances.
        assert board._selector_hold is None
        assert board.front_frame.panels[0].source_identity == first_source
        frozen_after = slot.freeze_current()[2]
        assert frozen_after.snapshot.ref == frozen_before.snapshot.ref
        assert frozen_after.event_refs == frozen_before.event_refs
        assert setting.base_revision == 0 and setting._dirty and setting._stale
        assert setting._form.read_all() == setting_draft
        _until(
            application,
            lambda: board.visible_image_origin() is not None
            and board.visible_image_origin().presentation.panel_revision == 1,
        )
        setting._cancel_button.click()
        assert setting.base_revision == 1
        authored = window._image_display

        selector.setChecked(True)
        QtTest.QTest.mouseClick(
            board,
            QtCore.Qt.RightButton,
            pos=_image_target(board).center(),
        )
        assert _capture_image_binding(board).cross is not None
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        assert not selector.isChecked()
        assert window._rectangle_candidate is None
        assert _capture_image_binding(board).cross is None
        _until(
            application,
            lambda: window.final_reference is not None
            and window.final_reference != first_ref
            and board.has_front,
        )
        second_ref = window.final_reference
        payload = board.visible_image_payload()
        assert payload is not None
        assert window._image_display == authored
        assert payload.viewport.viewport_revision == authored.revision
        assert board.front_frame.panels[0].source_identity != first_source
        assert window._pending_image_interaction_origin is None
        from zlc_frontend.selector import ImageViewportCommit

        stale_viewport = payload.viewport.centered_zoom((0.5, 0.5), 0.9)
        with pytest.raises(RuntimeError, match="stale"):
            window._accept_image_interaction(
                ImageViewportCommit(first_origin, stale_viewport)
            )
        assert diagnostics.text() == ""
        assert experiment.readout.load_capture(second_ref).frame_source.schema.physical_shape == (
            1,
            1,
            96,
            128,
        )
    finally:
        if window is not None:
            _close_window(application, window)


def test_repeat_waits_for_the_previous_exact_raster_worker(
    experiment,
    application,
    monkeypatch,
):
    original = live_module.rasterize_image_indexed8
    entered = threading.Event()
    release = threading.Event()

    def blocked_raster(*args, **kwargs):
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("test did not release finite raster worker")
        return original(*args, **kwargs)

    monkeypatch.setattr(live_module, "rasterize_image_indexed8", blocked_raster)
    window = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, capture, _preview, diagnostics, board = _widgets(window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, entered.is_set)
        _until(application, lambda: window.final_reference is not None)
        first_ref = window.final_reference
        first_generation = window._run_owner.generation
        assert capture.text().startswith("Capture: FINAL")
        assert not window.worker_idle
        assert not start.isEnabled()

        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        application.processEvents()
        assert window._run_owner.generation == first_generation
        assert window.final_reference == first_ref

        release.set()
        _until(
            application,
            lambda: window.worker_idle and start.isEnabled() and board.has_front,
        )
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None
            and window.final_reference != first_ref,
        )
        assert window._run_owner.generation == first_generation + 1
        assert diagnostics.text() == ""
    finally:
        release.set()
        if window is not None:
            _close_window(application, window)


def test_capture_rejects_stale_and_failed_display_commits_without_touching_final(
    experiment,
    application,
    monkeypatch,
):
    window = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, _capture, _preview, diagnostics, board = _widgets(window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None and board.has_front,
        )
        final_ref = window.final_reference
        origin = board.visible_image_origin()
        payload = board.visible_image_payload()
        assert origin is not None and payload is not None

        editor = window._edit_image_display
        _choose_image_display_value(editor, "colormap", ImageColormap.PLASMA)
        editor._apply_button.click()
        assert window._image_display.revision == 1
        stale_viewport = payload.viewport.centered_zoom((0.5, 0.5), 0.9)
        from zlc_frontend.selector import ImageViewportCommit

        with pytest.raises(RuntimeError, match="stale"):
            window._accept_image_interaction(
                ImageViewportCommit(origin, stale_viewport)
            )
        _until(
            application,
            lambda: board.visible_image_origin() is not None
            and board.visible_image_origin().presentation.panel_revision == 1,
        )

        before = window._image_display
        assert window._live is not None
        monkeypatch.setattr(
            window._live,
            "reconfigure_image_display",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("reconfigure rejected")),
        )
        _choose_image_display_value(editor, "colormap", ImageColormap.INFERNO)
        editor._apply_button.click()
        assert window._image_display == before
        assert "reconfigure rejected" in diagnostics.text()
        assert window.final_reference == final_ref
        assert experiment.readout.load_capture(final_ref).frame_source.schema.physical_shape == (
            1,
            1,
            96,
            128,
        )
    finally:
        if window is not None:
            _close_window(application, window)


def test_display_rerender_fault_clears_only_pending_intent_and_preserves_final(
    experiment,
    application,
    monkeypatch,
):
    window = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, capture, preview, _diagnostics, board = _widgets(window)
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "captureSelectorSwitch",
        )
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None
            and selector is not None
            and selector.isEnabled(),
        )
        final_ref = window.final_reference
        source = board.front_frame.panels[0].source_identity

        def fail_raster(*_args, **_kwargs):
            raise RuntimeError("injected display rerender failure")

        monkeypatch.setattr(live_module, "rasterize_image_indexed8", fail_raster)
        selector.setChecked(True)
        _wheel_image_down(board)
        assert window._image_display.revision == 1
        assert board._image_interaction_is_pending(_capture_image_binding(board))
        _until(application, lambda: preview.text().startswith("Preview: FAILED"))
        assert not board._image_interaction_is_pending(
            _capture_image_binding(board)
        )
        assert window._pending_image_interaction_origin is None
        assert not selector.isEnabled()
        assert capture.text().startswith("Capture: FINAL")
        assert window.final_reference == final_ref
        assert board.front_frame.panels[0].source_identity == source
        assert experiment.readout.load_capture(final_ref).frame_source.schema.physical_shape == (
            1,
            1,
            96,
            128,
        )
    finally:
        if window is not None:
            _close_window(application, window)


def test_close_releases_an_active_finite_image_hold_nonblocking(
    experiment,
    application,
):
    request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
    window = open_capture_workbench(experiment, request)
    try:
        start, _capture, _preview, _diagnostics, board = _widgets(window)
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "captureSelectorSwitch",
        )
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None
            and selector is not None
            and selector.isEnabled(),
        )
        selector.setChecked(True)
        QtTest.QTest.mousePress(
            board,
            QtCore.Qt.LeftButton,
            pos=_point(_image_target(board), 0.25, 0.25),
        )
        assert board._selector_hold is not None
        began = time.monotonic()
        window.close()
        assert time.monotonic() - began < 0.1
        assert board._selector_hold is None
        _until(application, lambda: not window.isVisible(), timeout=5.0)
        assert not board.has_front
        assert window._pending_image_interaction_origin is None
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_close_during_pending_finite_rerender_never_late_publishes(
    experiment,
    application,
    monkeypatch,
):
    window = None
    release = threading.Event()
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, capture, _preview, _diagnostics, board = _widgets(window)
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "captureSelectorSwitch",
        )
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(
            application,
            lambda: window.final_reference is not None
            and selector is not None
            and selector.isEnabled(),
        )
        final_ref = window.final_reference
        original = live_module.rasterize_image_indexed8
        entered = threading.Event()

        def blocked_raster(*args, **kwargs):
            entered.set()
            if not release.wait(5.0):
                raise TimeoutError("test did not release finite rerender")
            return original(*args, **kwargs)

        monkeypatch.setattr(live_module, "rasterize_image_indexed8", blocked_raster)
        selector.setChecked(True)
        _wheel_image_down(board)
        _until(application, entered.is_set)
        assert not window.worker_idle

        began = time.monotonic()
        window.close()
        assert time.monotonic() - began < 0.1
        assert window.isVisible()
        assert not board.has_front
        assert window.final_reference == final_ref
        application.processEvents()
        assert not board.has_front

        release.set()
        _until(application, lambda: not window.isVisible(), timeout=5.0)
        assert window.worker_idle
        assert not board.has_front
        assert capture.text().startswith("Capture: CLOSING")
    finally:
        release.set()
        if window is not None and window.isVisible():
            _close_window(application, window)


def test_multi_event_capture_is_rejected_before_start(experiment, application):
    window = None
    try:
        request = experiment.readout.capture_request(MULTI_EVENT_PULSE)
        generic_ref = experiment.run(request)
        assert (
            experiment.readout.load_capture(generic_ref).frame_source.schema.physical_shape
            == (1, 3, 96, 128)
        )
        window = open_capture_workbench(experiment, request)
        start, capture, _preview, diagnostics, _board = _widgets(window)
        _until(application, lambda: "Preparation failed" in diagnostics.text())
        assert not start.isEnabled()
        assert capture.text() == "Capture: NOT READY · NOT FINAL"
        assert "exactly one readout event per repeat" in diagnostics.text()
    finally:
        if window is not None:
            _close_window(application, window)


def test_render_failure_does_not_change_final_capture(
    experiment,
    application,
    monkeypatch,
):
    import zlc_workbench.live as live_module

    def fail_raster(_image, _state, **_kwargs):
        raise RuntimeError("injected raster failure")

    monkeypatch.setattr(live_module, "rasterize_image_indexed8", fail_raster)
    window = None
    try:
        request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
        window = open_capture_workbench(experiment, request)
        start, capture, preview, _diagnostics, board = _widgets(window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        _until(application, lambda: window.final_reference is not None)
        _until(application, lambda: preview.text().startswith("Preview: FAILED"))
        assert capture.text().startswith("Capture: FINAL")
        assert not board.has_front
        assert (
            experiment.readout.load_capture(window.final_reference)
            .frame_source.schema.physical_shape
            == (1, 1, 96, 128)
        )
    finally:
        if window is not None:
            _close_window(application, window)


def test_close_during_start_is_nonblocking_and_does_not_close_experiment(
    experiment,
    application,
):
    request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
    window = open_capture_workbench(experiment, request)
    try:
        start, _capture, _preview, _diagnostics, _board = _widgets(window)
        _until(application, start.isEnabled)
        QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
        began = time.monotonic()
        window.close()
        assert time.monotonic() - began < 0.1
        _until(application, lambda: not window.isVisible(), timeout=5.0)
        assert window.worker_idle
        assert experiment.inspect(request).output_shape == (1, 1, 96, 128)
    finally:
        if window.isVisible():
            _close_window(application, window)


def test_slot_replays_update_or_failure_once_to_a_late_listener(experiment):
    request = experiment.readout.capture_request(SINGLE_EVENT_PULSE)
    prepared = _prepare_capture_for_workbench(experiment, request)
    try:
        slots = []

        def factory(spec):
            slot = LiveDatasetSlot(
                spec,
                dataset_id=DatasetId("late-update"),
            )
            slots.append(slot)
            return slot

        handle = prepared.start_with_preview(
            block_id=BlockId("late-update-block"),
            factory=factory,
        )
        handle.result(5.0)
        calls = []
        slots[0].set_change_listener(lambda: calls.append("update"))
        assert calls == ["update"]
        with pytest.raises(RuntimeError, match="already has"):
            slots[0].set_change_listener(lambda: None)
        slots[0].close()

        failed_slots = [
            LiveDatasetSlot(
                slots[0].spec,
                dataset_id=DatasetId("late-failure"),
            )
        ]
        failed_slots[0].fail("injected source failure")
        failures = []
        failed_slots[0].set_change_listener(lambda: failures.append("failure"))
        assert failures == ["failure"]
        assert failed_slots[0].failure is not None
        failed_slots[0].fail("second failure must not replay")
        assert failures == ["failure"]
        failed_slots[0].close()
    finally:
        for slot in locals().get("slots", ()):
            slot.close()
        for slot in locals().get("failed_slots", ()):
            slot.close()


def test_public_import_is_lazy_and_w1_has_no_runtime_boundary_leak():
    code = (
        "import sys; import Zou_lab_control.workbench, zlc_workbench; "
        "assert not any(n == 'PyQt5' or n.startswith('PyQt5.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    source = (ROOT / "Zou_lab_control" / "workbench" / "_capture.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "MinimalPipelineSpec",
        "RunPlan",
        "BoundCapturePort",
        "BoundPulsePort",
        "compile_capture_artifact_pipeline",
        "_authority_token",
    ):
        assert forbidden not in source


class _CameraControlDataset(CameraMonitorLiveDataset):
    def __init__(self, state: CameraMonitorRoiState) -> None:
        self.state = state
        self.closed = False
        self.slot = None
        self.topic, self.consumer = create_control_topic(lambda value: value)

    def current_roi_state(self):
        return self.state

    def submit_roi_control(self, candidate):
        receipt = self.topic.publish(candidate)
        if self.slot is not None:
            self.slot.close()
        return receipt

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.consumer.terminate("camera control dataset closed")


def _camera_control_slot(dataset):
    slot = object.__new__(LiveDatasetSlot)
    slot._lock = threading.Lock()
    slot._dataset = dataset
    slot._camera_roi_state = CameraMonitorRoiState(
        None,
        None,
        None,
        None,
        None,
    )
    slot._failure = None
    slot._notification_failure = None
    slot._terminal = False
    slot._withdrawn = False
    slot._closed = False
    slot._listener = None
    slot._listener_claimed = False
    slot._pending_change = False
    slot._retain_on_terminal = False
    return slot


def test_notification_failure_is_visible_without_detaching_source_dataset():
    dataset = _CameraControlDataset(
        CameraMonitorRoiState(None, None, None, None, None)
    )
    slot = _camera_control_slot(dataset)
    wakes = []
    slot.set_change_listener(lambda: wakes.append("wake"))
    slot.notification_failed("view change listener failed")
    assert slot.failure is None
    assert slot.notification_failure == "view change listener failed"
    assert wakes == ["wake"]
    assert slot._dataset is dataset
    assert not dataset.closed
    assert slot.current_camera_roi_state() == dataset.state

    slot.notification_failed("later failure must not replace first cause")
    assert slot.failure is None
    assert slot.notification_failure == "view change listener failed"
    assert wakes == ["wake"]
    assert slot._dataset is dataset

    visible_status = object()
    controller = object.__new__(LiveBoardController)
    controller._slot = slot
    controller._lock = threading.Lock()
    controller._closed = False
    controller._fault = None
    controller._candidate = object()
    controller._active = True
    controller._dirty = True
    controller._port = object()
    controller._sources = (object(),)
    controller._front_status = visible_status
    revoked = []

    def revoke(before_revoke=None):
        if before_revoke is not None and not before_revoke():
            return False
        revoked.append(True)
        return True

    controller._board = SimpleNamespace(
        revoke_pending_publication=revoke
    )
    controller._request_owner_wake = lambda: wakes.append("controller-fault")
    assert controller.admit_pending() is False
    assert controller.fault is not None
    assert controller._front_status is visible_status
    assert revoked == [True]
    assert slot._dataset is dataset and not dataset.closed

    slot.source_terminal()
    assert dataset.closed and slot.terminal


def test_camera_control_submit_returns_receipt_when_detach_loses_the_race():
    state = CameraMonitorRoiState(None, None, None, None, None)
    dataset = _CameraControlDataset(state)
    slot = _camera_control_slot(dataset)
    dataset.slot = slot
    receipt = slot.submit_camera_roi_control(None)
    assert receipt.snapshot().status is ControlAckStatus.TERMINATED
    assert dataset.closed and slot.terminal


@pytest.mark.parametrize("terminal", [False, True])
def test_camera_roi_state_survives_failure_or_terminal_detach(terminal):
    initial = CameraMonitorRoiState(None, 1, None, None, None, 1)
    final = CameraMonitorRoiState(
        None,
        1,
        None,
        None,
        None,
        2,
        "scalar branch failed",
    )
    dataset = _CameraControlDataset(initial)
    slot = _camera_control_slot(dataset)
    assert slot.current_camera_roi_state() is initial
    dataset.state = final
    if terminal:
        slot.source_terminal()
    else:
        slot.fail("camera source failed")
    assert slot.current_camera_roi_state() is final
    assert dataset.closed


def test_camera_roi_state_cache_rejects_conflicting_same_revision_truth():
    first = CameraMonitorRoiState(None, 1, None, None, None, 1)
    conflicting = CameraMonitorRoiState(None, 2, None, None, None, 1)
    dataset = _CameraControlDataset(first)
    slot = _camera_control_slot(dataset)
    assert slot.current_camera_roi_state() is first
    dataset.state = conflicting
    with pytest.raises(RuntimeError, match="without advancing state_revision"):
        slot.current_camera_roi_state()
    assert slot._camera_roi_state is first


def test_freeze_presentation_revokes_work_and_preserves_the_coherent_front_state():
    controller = object.__new__(LiveBoardController)
    controller._owner_thread = threading.get_ident()
    controller._lock = threading.Lock()
    controller._closed = False
    controller._candidate = object()
    controller._port = object()
    controller._sources = (object(),)
    controller._dirty = True
    controller._active = True
    controller._presentation_frozen = False
    frozen = []
    controller._board = SimpleNamespace(freeze_front=lambda: frozen.append(True))
    controller.freeze_presentation()
    assert controller._candidate is None
    assert controller._port is None and controller._sources is None
    assert not controller._dirty and not controller._active
    assert controller._presentation_frozen
    assert frozen == [True]


def test_scalar_control_change_reuses_layout_when_panel_ids_are_unchanged(monkeypatch):
    class CameraSpec:
        pass

    panels = (SimpleNamespace(panel_id="image"),)
    previous = SimpleNamespace(
        board_id="camera-board",
        layout_generation=4,
        panels=panels,
        image_document=object(),
        image_display=None,
        image_viewport=None,
        scalar_dataset_id=object(),
        scalar_documents=(SimpleNamespace(document_id="curve", revision=0),),
        scalar_binding_fingerprint="binding",
        scalar_control_revision=2,
    )
    replacement = SimpleNamespace(
        board_id=previous.board_id,
        layout_generation=previous.layout_generation,
        panels=panels,
        scalar_documents=(),
        presentations=(object(),),
    )
    requests = []
    retired = []
    monkeypatch.setattr(live_module, "CameraMonitorViewSpec", CameraSpec)
    monkeypatch.setattr(
        live_module,
        "_build_live_configuration",
        lambda **_kwargs: replacement,
    )
    controller = object.__new__(LiveBoardController)
    controller._owner_thread = threading.get_ident()
    controller._slot = SimpleNamespace(
        spec=CameraSpec(),
        failure=None,
        withdrawn=False,
    )
    controller._worker_thread_affine = True
    controller._scalar_dataset_generations = {}
    controller._scalar_generation_datasets = {}
    controller._lock = threading.Lock()
    controller._closed = False
    controller._fault = None
    controller._configuration_epoch = 8
    controller._configuration = previous
    controller._port = None
    controller._sources = None
    controller._candidate = None
    controller._dirty = False
    controller._active = False
    controller._presentation_frozen = False
    controller._board = SimpleNamespace(
        model=SimpleNamespace(
            board_id=previous.board_id,
            layout_generation=previous.layout_generation,
            panels=panels,
        )
    )
    controller._document = object()
    controller._submit = lambda work: retired.append(work)
    controller._request_snapshot = lambda: requests.append(None)

    controller.reconfigure_scalar(
        CameraMonitorRoiState(None, 3, None, None, None, 3),
        None,
        (),
        None,
        None,
    )

    assert controller._configuration is replacement
    assert controller._configuration_epoch == 9
    assert retired and requests == [None]
