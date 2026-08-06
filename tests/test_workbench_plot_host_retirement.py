"""One owner answers "has this plot host stopped?" for the whole workbench."""

from __future__ import annotations

from concurrent.futures import Future
import os
from pathlib import Path
import re
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

from zlc_data import (
    REPEAT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_plot import AxisRef, ImagePlot, PlotKind, RasterPlotHost
from zlc_workbench.data_figure.window import DataFigureWindow
from zlc_workbench.retiring_hosts import RetiringPlotHosts
from zlc_workbench.task_console.console_records import PanelConfig
from zlc_workbench.task_console.panel_card import PanelCard
from zlc_workbench.task_console.panel_editor import PanelEditor


ROOT = Path(__file__).resolve().parents[1]
RETIREMENT_OWNER = ROOT / "zlc_workbench" / "retiring_hosts.py"


class _ScriptedHost:
    """A host whose worker stops only after a scripted number of attempts.

    ``RasterPlotHost.close`` is a two-part contract — request the shutdown, and
    report whether the worker has stopped — and only the second part is what
    ``RetiringPlotHosts`` stores.  Scripting it is the only way to observe the
    not-yet-stopped window deterministically.
    """

    def __init__(self, attempts_before_stopped: int) -> None:
        self._remaining = attempts_before_stopped
        self.host_id = f"scripted-{id(self):x}"
        self.timeouts: list[object] = []

    def close(self, *, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        if self._remaining > 0:
            self._remaining -= 1
            return False
        return True


class _WakeRecorder:
    def __init__(self) -> None:
        self.wakes = 0
        self.detached = False

    def request_owner_wake(self) -> None:
        self.wakes += 1

    def detach(self) -> None:
        self.detached = True


class _FailingCloseAdapter:
    """``Qt5PlotWidget.close_adapter`` when a cleanup edge fails.

    Faithful to the real one: it latches itself closed *before* running the
    edges, so the failure is re-raised exactly once and every later call is the
    no-op early return (``zlc_plot/backends.py``).
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("cleanup edge failed")


def _failing_plot_widget(parent: QtWidgets.QWidget | None = None) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget(parent)
    widget.close_adapter = _FailingCloseAdapter()
    return widget


class _CardRetirement:
    """Only ``PanelCard``'s retirement seam, without its Qt widget tree."""

    _retire_host = PanelCard._retire_host
    _poll_retiring_hosts = PanelCard._poll_retiring_hosts
    _arm_retirement_wake = PanelCard._arm_retirement_wake

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[object]] = {}
        self._publication_by_host_revision: dict[str, dict[int, object]] = {}
        self._unresolved_revisions_by_host: dict[str, set[int]] = {}
        self._latest_host_revisions: dict[str, tuple[int, ...]] = {}
        self._latest_host_sequence: dict[str, int] = {}
        self._presented_revisions_by_host: dict[str, tuple[int, ...]] = {}
        self._retiring_hosts = RetiringPlotHosts()
        self._closing = False
        self._wake = _WakeRecorder()


class _EditorRetirement(QtWidgets.QWidget):
    """``PanelEditor``'s real shutdown bodies over the state they read.

    ``PanelEditor.__init__`` needs a whole TaskConsole; the shutdown path needs
    only the holders it empties, so the real methods run here verbatim.
    """

    shutdownFinished = QtCore.pyqtSignal()

    _retire_surface = PanelEditor._retire_surface
    _poll_retiring_hosts = PanelEditor._poll_retiring_hosts
    _arm_retirement_wake = PanelEditor._arm_retirement_wake
    teardown = PanelEditor.teardown
    _poll_shutdown = PanelEditor._poll_shutdown
    _finish_shutdown = PanelEditor._finish_shutdown

    def __init__(self) -> None:
        super().__init__()
        self._retiring_hosts = RetiringPlotHosts()
        self._closing = False
        self._closed = False
        self._wake = _WakeRecorder()
        self._host = None
        self._plot_widget = None
        self._initial_future: Future | None = None
        self._front_futures: set[Future] = set()
        self.canvas_holder = QtWidgets.QVBoxLayout()
        self.controls_holder = QtWidgets.QVBoxLayout()
        self.fit_holder = QtWidgets.QVBoxLayout()
        self._shutdown_timer = QtCore.QTimer(self)
        self._shutdown_timer.setInterval(25)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)
        self.finished = 0
        self.shutdownFinished.connect(self._count_finished)

    def _count_finished(self) -> None:
        self.finished += 1


def _image_snapshot() -> OwnedSnapshot:
    repeat = AxisSpec(AxisId("capture.repeat"), "repeat", REPEAT, 1, (0,))
    y = AxisSpec(AxisId("camera.y"), "y", SPATIAL_Y, 8, tuple(range(8)), "pixel")
    x = AxisSpec(AxisId("camera.x"), "x", SPATIAL_X, 10, tuple(range(10)), "pixel")
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema((y, x), ValidityContract.value(), np.dtype("<u2"), "count"),
    )
    values = np.arange(np.prod(schema.physical_shape), dtype="<u2").reshape(
        schema.physical_shape
    )
    block = DataBlock(
        BlockId("retirement-image"),
        DatasetRevision(3),
        values,
        CellValidity(np.ones((1, 1), dtype=bool)),
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("retirement-generation")), block)


def _wait_until(application: QtWidgets.QApplication, predicate, timeout: float = 12.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        application.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Qt condition did not become true")
        time.sleep(0.01)
    application.processEvents()


def test_retirement_is_membership_so_the_in_flight_fact_cannot_latch() -> None:
    """"Still retiring" is read off the set, so clearing it is not an event."""

    retiring = RetiringPlotHosts()
    slow = _ScriptedHost(attempts_before_stopped=2)

    assert not retiring

    retiring.retire(slow)
    assert retiring

    assert retiring.poll() is False
    assert retiring

    assert retiring.poll() is True
    assert not retiring

    # A drained owner keeps answering from the same collection, so no later
    # entrance can resurrect a stale "in flight" boolean.
    assert retiring.poll() is True


def test_retire_hands_back_nothing_so_no_caller_can_copy_the_fact() -> None:
    """The only answer is the collection; ``retire`` cannot seed a stale copy."""

    retiring = RetiringPlotHosts()

    assert retiring.retire(_ScriptedHost(attempts_before_stopped=1)) is None
    assert not hasattr(RetiringPlotHosts, "__len__")


def test_a_host_that_stopped_immediately_never_enters_the_set() -> None:
    retiring = RetiringPlotHosts()
    prompt = _ScriptedHost(attempts_before_stopped=0)

    retiring.retire(prompt)
    assert not retiring


def test_every_attempt_is_bounded_so_a_qt_owner_turn_may_call_it() -> None:
    retiring = RetiringPlotHosts()
    slow = _ScriptedHost(attempts_before_stopped=1)

    retiring.retire(slow)
    retiring.poll()

    assert slow.timeouts == [0.0, 0.0]


def test_a_real_plot_host_drains_through_poll_without_blocking() -> None:
    snapshot = _image_snapshot()
    host = RasterPlotHost.from_plot(
        snapshot,
        ImagePlot(AxisRef.data("camera.x"), AxisRef.data("camera.y")),
    )
    host.wait_for_front(timeout=5.0)

    retiring = RetiringPlotHosts()
    retiring.retire(host)

    deadline = time.monotonic() + 10.0
    while retiring:
        if retiring.poll():
            break
        if time.monotonic() >= deadline:
            raise AssertionError("a retiring plot host never reported stopped")
        time.sleep(0.005)

    assert not retiring
    assert host.close(timeout=0.0) is True


def test_data_figure_reports_closed_only_once_its_host_reports_stopped(
    tmp_path: Path,
) -> None:
    application = ensure_qt_app()
    pane = DataFigureWindow(
        _image_snapshot(),
        ImagePlot(AxisRef.data("camera.x"), AxisRef.data("camera.y")),
        output_root=tmp_path,
        embedded=True,
    )
    try:
        _wait_until(application, lambda: pane.plot_widget is not None)
        assert pane.closed is False

        _wait_until(application, pane.teardown)

        assert pane.closed is True
        assert not pane._retiring
    finally:
        pane.deleteLater()
        application.processEvents()


def test_data_figure_asks_its_host_to_stop_when_the_widget_cleanup_raises(
    tmp_path: Path,
) -> None:
    """``close_adapter`` re-raises by design; the host must still be asked.

    An entrance that dies before retiring leaves the set empty, and an empty
    set is indistinguishable from a drained one -- the window would report
    ``closed`` while its worker runs on, never having been asked to stop.
    """

    application = ensure_qt_app()
    pane = DataFigureWindow(
        _image_snapshot(),
        ImagePlot(AxisRef.data("camera.x"), AxisRef.data("camera.y")),
        output_root=tmp_path,
        embedded=True,
    )
    try:
        _wait_until(application, lambda: pane.plot_widget is not None)
        pane.plot_widget.close_adapter = _FailingCloseAdapter()

        with pytest.raises(RuntimeError, match="cleanup edge failed"):
            pane.teardown()

        # Only a host that was asked to close refuses new front subscribers.
        with pytest.raises(RuntimeError, match="closing"):
            pane.host.subscribe_front(lambda front: None)

        _wait_until(application, pane.teardown)
        assert pane.closed is True
        assert pane.host.close(timeout=0.0) is True
    finally:
        pane.deleteLater()
        application.processEvents()


def test_data_figure_arms_its_retry_before_the_cleanup_edge_can_raise(
    tmp_path: Path,
) -> None:
    """The retry timer is the only thing that re-enters teardown by itself.

    Arming it after ``close_adapter`` means an entrance that dies on that edge
    leaves a host in the set with nothing left to drain it: the window keeps a
    live worker until a human happens to press close a second time.
    """

    application = ensure_qt_app()
    pane = DataFigureWindow(
        _image_snapshot(),
        ImagePlot(AxisRef.data("camera.x"), AxisRef.data("camera.y")),
        output_root=tmp_path,
        embedded=True,
    )
    real_host = pane.host
    try:
        _wait_until(application, lambda: pane.plot_widget is not None)
        pane.plot_widget.close_adapter = _FailingCloseAdapter()
        # A worker that outlives its close request is the only state in which
        # the retry matters, and only a scripted host reaches it reliably.
        pane._host = _ScriptedHost(attempts_before_stopped=1_000)

        with pytest.raises(RuntimeError, match="cleanup edge failed"):
            pane.teardown()

        assert pane._retiring
        assert pane._shutdown_timer.isActive()
    finally:
        pane._shutdown_timer.stop()
        real_host.close()
        pane.deleteLater()
        application.processEvents()


def test_panel_card_shutdown_never_reports_done_when_the_widget_cleanup_raises() -> None:
    """``shutdown`` is the console's acknowledgement; an empty set is not one."""

    application = ensure_qt_app()
    card = PanelCard(
        PanelConfig(plot=PlotKind.IMAGE, title="retirement", signal="camera/frame"),
        signal_groups_provider=lambda selected: {},
    )
    try:
        stuck = _ScriptedHost(attempts_before_stopped=1_000)
        widget = _failing_plot_widget()
        card.canvas_holder.insertWidget(0, widget)
        card._plot_widget = widget
        card._host = stuck

        with pytest.raises(RuntimeError, match="cleanup edge failed"):
            card.shutdown()

        assert card.shutdown() is False
        assert card._retiring_hosts
        assert stuck.timeouts == [0.0, 0.0]
    finally:
        card.deleteLater()
        application.processEvents()


def test_panel_editor_never_finishes_shutdown_when_the_widget_cleanup_raises() -> None:
    """``shutdownFinished`` must never fire while a live host is still held."""

    application = ensure_qt_app()
    editor = _EditorRetirement()
    try:
        stuck = _ScriptedHost(attempts_before_stopped=1_000)
        widget = _failing_plot_widget()
        editor.canvas_holder.insertWidget(0, widget)
        editor._plot_widget = widget
        editor._host = stuck

        with pytest.raises(RuntimeError, match="cleanup edge failed"):
            editor.teardown()

        assert editor.teardown() is False
        assert editor.finished == 0
        assert editor._retiring_hosts
        assert stuck.timeouts == [0.0, 0.0]
    finally:
        editor._shutdown_timer.stop()
        editor.deleteLater()
        application.processEvents()


@pytest.mark.parametrize("surface", (_CardRetirement, _EditorRetirement))
def test_console_surfaces_arm_their_retry_wake_off_the_set_not_a_flag(
    surface,
) -> None:
    """The retry edge is re-armed from the level, so it cannot stop early."""

    application = ensure_qt_app()
    owner = surface()
    stuck = _ScriptedHost(attempts_before_stopped=1_000)

    assert owner._poll_retiring_hosts() is True
    application.processEvents()
    assert owner._wake.wakes == 0

    owner._retiring_hosts.retire(stuck)
    owner._arm_retirement_wake()
    _wait_until(application, lambda: owner._wake.wakes >= 1, timeout=3.0)

    # While the console drives the shutdown there is no wake owner left, so the
    # timer must not be re-armed behind it.
    owner._closing = True
    owner._wake.wakes = 0
    assert owner._poll_retiring_hosts() is False
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert owner._wake.wakes == 0


def test_panel_card_retire_host_releases_its_bookkeeping_and_the_host() -> None:
    application = ensure_qt_app()
    card = _CardRetirement()
    released: list[str] = []
    host = _ScriptedHost(attempts_before_stopped=1)
    card._subscriptions[host.host_id] = [lambda: released.append("unsubscribed")]
    card._latest_host_sequence[host.host_id] = 4

    card._retire_host(host)
    application.processEvents()

    assert released == ["unsubscribed"]
    assert host.host_id not in card._subscriptions
    assert host.host_id not in card._latest_host_sequence
    assert card._retiring_hosts
    assert card._poll_retiring_hosts() is True
    assert not card._retiring_hosts


def test_no_workbench_module_but_the_owner_attempts_a_plot_host_close() -> None:
    """The policy has one implementation, so it cannot be omitted by copying.

    A hand-copied "close(timeout=0.0) and retry" is what let the pulse editor
    silently keep a blocking bare ``close()`` on the Qt thread.  Grep is the
    only guard that survives the next new window.
    """

    bare_host_close = re.compile(r"\b\w*host\.close\(\s*\)", re.IGNORECASE)
    violations = []
    for path in sorted((ROOT / "zlc_workbench").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        if path != RETIREMENT_OWNER and "close(timeout=" in text:
            violations.append(f"{relative} attempts a plot-host close itself")
        for match in bare_host_close.finditer(text):
            violations.append(
                f"{relative} blocks the Qt thread on {match.group(0)}"
            )
    assert not violations, (
        "RetiringPlotHosts is the sole owner of plot-host shutdown:\n"
        + "\n".join(violations)
    )




def test_the_pulse_preview_host_is_retired_before_the_cleanup_edge_can_raise(
    tmp_path,
) -> None:
    """The fourth adopter owes the same order as the three siblings.

    ``close_adapter`` re-raises its first failing edge by contract, so retiring
    the host after it would strand a live worker while an empty retiring set
    reports every surface already gone.
    """

    from Zou_lab_control.api import WorkspacePaths
    from Zou_lab_control.workbench import open_pulse_editor
    from zlc_pulse import load_deployed_pulse_target, new_pulse_document

    application = ensure_qt_app()
    body = open_pulse_editor(
        workspace=WorkspacePaths.for_workspace((tmp_path / "workspace").resolve()),
        document=new_pulse_document(load_deployed_pulse_target(), time_step_ns=20),
    )
    try:
        stuck = _ScriptedHost(attempts_before_stopped=1_000)
        widget = _failing_plot_widget()
        body.preview_host = stuck
        body.preview_widget = widget

        with pytest.raises(RuntimeError, match="cleanup edge failed"):
            body._release_preview_surface()

        # The level moved before the raise: the host is stopping regardless.
        assert stuck.timeouts == [0.0]
        assert body.preview_host is None
        assert body._retiring_hosts
        assert not body.worker_idle

        widget.deleteLater()
    finally:
        body.deleteLater()
        application.processEvents()


def test_every_plot_window_owner_delegates_to_the_single_owner() -> None:
    owners = (
        Path("zlc_workbench/data_figure/window.py"),
        Path("zlc_workbench/task_console/panel_card.py"),
        Path("zlc_workbench/task_console/panel_editor.py"),
        Path("zlc_workbench/pulse_editor/window.py"),
    )
    for relative in owners:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "from zlc_workbench.retiring_hosts import RetiringPlotHosts" in text
        assert "RetiringPlotHosts()" in text
