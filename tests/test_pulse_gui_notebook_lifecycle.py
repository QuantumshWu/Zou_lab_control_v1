"""Human lifecycle contract for the notebook-owned formal Pulse GUI."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtTest

from Zou_lab_control.notebook.facade import connect
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_pulse import load_deployed_pulse_target, new_pulse_document


def _until(application, predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


def test_notebook_x_hides_and_reopens_same_dirty_editor_then_owner_retires(tmp_path):
    application = ensure_qt_app()
    experiment = connect("virtual", repository=tmp_path)
    body = experiment.pulse_gui()
    wrapper = body.window()
    try:
        editor = body.schedule_view.name_edit
        QtTest.QTest.mouseClick(editor, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        QtTest.QTest.keyClicks(editor, "operator unsaved pulse")
        _until(
            application,
            lambda: body.current_document.name == "operator unsaved pulse"
            and body._controller.snapshot().dirty,
        )

        # This is the same close event emitted by the title-bar X.  Notebook
        # windows hide immediately; no controller shutdown or dirty prompt runs.
        wrapper.close()
        _until(application, lambda: not wrapper.isVisible())
        assert not body._controller.snapshot().close_requested

        restored = experiment.pulse_gui()
        assert restored is body
        _until(application, wrapper.isVisible)
        assert restored.current_document.name == "operator unsaved pulse"
        assert restored._controller.snapshot().dirty

        replacement = new_pulse_document(
            load_deployed_pulse_target(),
            time_step_ns=20,
        )
        with pytest.raises(RuntimeError, match="already owns a Pulse editor"):
            experiment.pulse_gui(document=replacement)

        QtTest.QTest.mouseClick(
            body.schedule_view.fire_button,
            QtCore.Qt.LeftButton,
        )
        _until(
            application,
            lambda: body.active_snapshot is not None
            and body.active_snapshot.state.value == "RUNNING",
        )

        # Runtime safety belongs to Experiment.close, not to a Qt close event.
        # The editor detaches and drains after authority retirement begins.
        experiment.close()
        _until(
            application,
            lambda: body.permanently_closed and body.worker_idle,
        )
    finally:
        experiment.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
