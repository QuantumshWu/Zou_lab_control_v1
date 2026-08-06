"""Human lifecycle contract for the Experiment-owned formal Pulse GUI."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtTest

from Zou_lab_control.api import WorkspacePaths, connect
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_pulse import load_deployed_pulse_target, new_pulse_document


def _until(application, predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.005)
    assert predicate()


class _StillStoppingHost:
    """A plot host whose worker outlives its close request until released."""

    def __init__(self) -> None:
        self.attempts = 0
        self.stopped = False

    def close(self, *, timeout: float | None = None) -> bool:
        self.attempts += 1
        return self.stopped


def test_committing_the_close_waits_for_a_host_that_has_not_stopped(tmp_path):
    """The last entrance may not strand a worker with nobody left to poll it.

    Committing the permanent close destroys the wrapper, and with it the clock
    that re-attempts a retiring host.  Committing while one has not stopped
    therefore leaks its thread un-joined, so the commit waits on that level
    instead of overtaking it.
    """

    application = ensure_qt_app()
    experiment = connect(
        "virtual",
        workspace=WorkspacePaths.for_workspace((tmp_path / "workspace").resolve()),
    )
    body = experiment.pulse_gui()
    try:
        host = _StillStoppingHost()
        body._retiring_hosts.retire(host)
        assert bool(body._retiring_hosts)
        assert not body.worker_idle

        body._commit_permanent_close()
        assert not body.permanently_closed
        assert body._teardown_pending
        # A poller is left behind: the clock stays armed on the same level.
        assert body._timer.isActive()
        assert host.attempts > 1

        host.stopped = True
        body._commit_permanent_close()
        assert body.permanently_closed
        assert not body._retiring_hosts
        assert body.worker_idle
    finally:
        experiment.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_bound_x_hides_and_reopens_same_dirty_editor_then_owner_retires(tmp_path):
    application = ensure_qt_app()
    experiment = connect(
        "virtual",
        workspace=WorkspacePaths.for_workspace((tmp_path / "workspace").resolve()),
    )
    body = experiment.pulse_gui()
    wrapper = body.window()
    try:
        editor = body.schedule_view.name_edit
        QtTest.QTest.mouseClick(editor, QtCore.Qt.LeftButton)
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
        QtTest.QTest.keyClicks(editor, "operator unsaved pulse")
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key_Return)
        _until(
            application,
            lambda: body.current_document.name == "operator unsaved pulse"
            and body._controller.dirty,
        )

        # This is the same close event emitted by the title-bar X.  Notebook
        # windows hide immediately; no controller shutdown or dirty prompt runs.
        wrapper.close()
        _until(application, lambda: not wrapper.isVisible())
        assert not body._controller.runtime_update().close_requested

        restored = experiment.pulse_gui()
        assert restored is body
        _until(application, wrapper.isVisible)
        assert restored.current_document.name == "operator unsaved pulse"
        assert restored._controller.dirty

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
        assert body.schedule_view.fire_button.isEnabled()
        assert not body._last_runtime.run_admission_pending

        body.tabs.setCurrentWidget(body.preview_view)
        _until(application, lambda: body.preview_host is not None)

        released: list[tuple[object, bool]] = []
        commit = body._commit_permanent_close

        def record_release_order() -> None:
            released.append((body.preview_host, bool(body._retiring_hosts)))
            commit()

        body._commit_permanent_close = record_release_order

        # Runtime safety belongs to Experiment.close, not to a Qt close event.
        # The editor detaches and drains after authority retirement begins.
        experiment.close()
        _until(
            application,
            lambda: body.permanently_closed and body.worker_idle,
        )
        # The render thread is stopped before the wrapper is released: after
        # the window has accepted its close there is no event left to defer on,
        # so the only non-blocking place to wait for it is before that.
        assert released == [(None, False)]
        assert body._wake.fault is None
    finally:
        experiment.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
