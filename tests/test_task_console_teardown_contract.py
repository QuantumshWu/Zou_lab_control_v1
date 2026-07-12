"""Contract: closing the task-console window tears the console down (FRONTEND).

The console is a CHILD of the Fluent window, so its own ``closeEvent`` never fires
on a window close.  ``show_task_console`` therefore installs ``console.shutdown``
as the wrapper's pre-close guard.  This pins the contracts that keep a closed
dashboard from leaking node owner threads (each blocked in ``camera.acquire``,
holding the camera / RPyC link -- the leak that wedged the kernel):

1. a genuine CLOSE stops the refresh timer + every node's owner thread AND fires
   the optional ``on_close`` device-teardown hook;
2. a MINIMISE / hide (``hidden``, not ``closed``) must NOT stop the nodes;
3. teardown is idempotent (close + an explicit shutdown / cell re-run can both fire).

Built on the offscreen Qt platform -- it does NOT pull in the flaky demo GUI fixtures.
"""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


class _FakeNode:
    """Minimal node: records start/stop the way the real owner-thread node would."""

    def __init__(self):
        self.running = False
        self.starts = 0
        self.stops = 0

    def start(self):
        self.running = True
        self.starts += 1
        return self

    def stop(self, timeout=2.0):
        self.stops += 1
        self.running = False
        return True

    def step(self):
        pass

    def published_signals(self):
        return ()


def _open(on_close=None, *, hide_on_close=False, node=None):
    from Zou_lab_control.frontend.task_console import show_task_console
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    node = node or _FakeNode()
    console = show_task_console(
        hub=SignalHub(),
        running_nodes=[node],
        on_close=on_close,
        hide_on_close=hide_on_close,
    )
    window = ensure_qt_app()._zlc_retained_windows[-1]   # the ONE retain_window registry
    return console, window, node


def test_close_stops_nodes_timer_and_calls_on_close():
    calls = []
    console, window, node = _open(on_close=lambda: calls.append(1))
    assert node.running and node.starts == 1        # show_task_console auto-started it
    assert console._timer.isActive()
    assert window.close()                           # a genuine guarded close
    assert node.stops == 1 and not node.running     # node owner thread stopped
    assert not console._timer.isActive()            # refresh timer stopped
    assert calls == [1]                             # device-teardown hook fired


def test_minimise_does_not_stop_nodes():
    console, window, node = _open()
    window.hidden.emit()                            # hide / minimise -- NOT a close
    assert node.running and node.stops == 0         # nodes must survive a minimise


def test_shutdown_is_idempotent():
    console, window, node = _open()
    console.shutdown()
    console.shutdown()
    window.close()
    assert node.stops == 1                          # stopped exactly once


def test_closed_signal_is_notification_not_a_command_or_retain_bypass():
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    console, window, node = _open()
    registry = ensure_qt_app()._zlc_retained_windows
    assert window in registry
    window.closed.emit()
    assert node.running and node.stops == 0
    assert window in registry
    assert window.close()
    assert console._shut and window not in registry


class _StubbornNode(_FakeNode):
    def __init__(self):
        super().__init__()
        self.allow_stop = False

    def stop(self, timeout=2.0):
        self.stops += 1
        if not self.allow_stop:
            return False
        self.running = False
        return True


def test_close_and_x_to_hide_both_wait_for_confirmed_node_termination():
    calls = []
    node = _StubbornNode()
    console, window, _ = _open(on_close=lambda: calls.append(1), node=node)
    assert not window.close()
    assert node.running and not getattr(console, "_shut", False) and calls == [] and window.isVisible()
    node.allow_stop = True
    assert window.close()
    assert console._shut and calls == [1]

    node = _StubbornNode()
    console, window, _ = _open(hide_on_close=True, node=node)
    assert not window.close()
    assert node.running and window.isVisible()
    node.allow_stop = True
    window.close()  # X-to-hide intentionally ignores the QCloseEvent after a successful guard
    assert not node.running and not window.isVisible()
    assert not getattr(console, "_shut", False)


def test_resource_close_reentry_and_failure_are_retryable_and_once_only():
    holder = {}
    calls = []

    def reentrant_close():
        calls.append("reentrant")
        assert holder["console"].shutdown() is False

    console, window, _node = _open(on_close=reentrant_close)
    holder["console"] = console
    assert window.close()
    assert calls == ["reentrant"] and console._shut

    attempts = []

    def flaky_close():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("device close failed")

    console, window, _node = _open(on_close=flaky_close)
    assert not window.close()
    assert not getattr(console, "_shut", False)
    assert console._shutdown_state == "BLOCKED_RESOURCE_CLOSE"
    assert window.close()
    assert console._shut and attempts == [1, 1]


def test_logic_node_stop_preserves_live_thread_until_join_is_real():
    from Zou_lab_control.neutral_atom.operations.logic import LogicNode
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    entered = threading.Event()
    release = threading.Event()

    class _BlockingNode(LogicNode):
        def step(self):
            entered.set()
            release.wait(5.0)
            return {}

    node = _BlockingNode(SignalHub()).start()
    assert entered.wait(5.0)
    assert node.stop(timeout=0.01) is False
    assert node.running and node._thread is not None
    release.set()
    assert node.stop(timeout=5.0) is True
    assert not node.running and node._thread is None


def test_remove_and_device_reference_unknown_both_fail_closed():
    from Zou_lab_control.frontend.task_console import LogicNodeConfig

    node = _StubbornNode()
    console, window, _ = _open(node=node)
    row = console._attach_logic_node(
        LogicNodeConfig(kind="task", name="probe", title="probe")
    )
    assert row is not None
    console._logic_nodes[id(row)] = node
    console._last_node[id(row)] = node
    if node not in console.running_nodes:
        console.running_nodes.append(node)
    assert console._remove_logic_node(row) is False
    assert row in console.logic_nodes and console._logic_nodes[id(row)] is node

    def unknown_references():
        raise RuntimeError("cannot inspect device references")

    node.referenced_devices = unknown_references
    assert console.stop_nodes_using({123}) is False
    assert node.running
    node.allow_stop = True
    assert console._remove_logic_node(row) is True
    window.close()


def test_start_never_replaces_an_unconfirmed_conflicting_owner():
    device = object()

    class _ConflictingNode(_StubbornNode):
        def occupied_devices(self):
            return (device,)

    old = _ConflictingNode()
    with pytest.raises(RuntimeError, match="runtime authority"):
        _open(node=old)
    assert not old.running
    assert old.starts == 0


def test_unbounded_injected_node_is_rejected_before_admission():
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    class _UnboundedNode:
        running = True

        def stop(self):
            time.sleep(0.1)
            self.running = False

    class _CatchAllNode:
        running = True

        def stop(self, **_kwargs):
            self.running = False

    started = time.monotonic()
    with pytest.raises(TypeError, match=r"stop\(\*, timeout="):
        TaskConsole(
            hub=SignalHub(),
            state=default_console_state(),
            running_nodes=[_UnboundedNode()],
        )
    with pytest.raises(TypeError, match=r"stop\(\*, timeout="):
        TaskConsole(
            hub=SignalHub(),
            state=default_console_state(),
            running_nodes=[_CatchAllNode()],
        )
    assert time.monotonic() - started < 0.05


def test_node_ignoring_timeout_cannot_block_the_qt_owner_deadline():
    class _IgnoringTimeoutNode(_FakeNode):
        def stop(self, timeout=2.0):
            self.stops += 1
            time.sleep(0.12)
            self.running = False
            return True

    node = _IgnoringTimeoutNode()
    console, window, _ = _open(node=node)
    started = time.monotonic()
    assert console.shutdown(timeout=0.01) is False
    assert time.monotonic() - started < 0.05
    assert console._shutdown_state == "BLOCKED_NODE_OWNERSHIP"
    time.sleep(0.15)
    assert console.shutdown(timeout=5.0) is True
    window.close()


def test_running_task_card_waits_for_confirmed_render_join_before_teardown():
    from Zou_lab_control.frontend.task_console import LogicNodeConfig, PanelConfig

    console, window, node = _open()
    row = console._attach_logic_node(
        LogicNodeConfig(kind="task", name="probe", title="probe")
    )
    assert row is not None
    console._logic_nodes[id(row)] = node
    console._last_node[id(row)] = node
    card = console._new_panel_card(
        PanelConfig(kind="monitor", title="task", source="value = 0")
    )
    console._attach_card(card)
    console._running_task_row = row
    console._task_card = card
    console._task_output_node = node
    console._apply_task_lock(True)

    shutdown_calls = []
    real_card_shutdown = card.shutdown

    def tracked_card_shutdown():
        shutdown_calls.append(1)
        return real_card_shutdown()

    card.shutdown = tracked_card_shutdown
    real_render_stop = console._render_loop.stop
    render_attempts = []

    def delayed_render_stop(timeout=5.0):
        render_attempts.append(timeout)
        if len(render_attempts) == 1:
            return False
        return real_render_stop(timeout)

    console._render_loop.stop = delayed_render_stop
    assert console.shutdown(timeout=0.2) is False
    assert shutdown_calls == []
    assert card in console.cards and console._deferred_task_card is card
    assert console.shutdown(timeout=5.0) is True
    assert shutdown_calls == [1]
    assert card not in console.cards and console._deferred_task_card is None
    window.close()


def test_deferred_task_card_shutdown_failure_is_retried_before_detach():
    from Zou_lab_control.frontend.task_console import LogicNodeConfig, PanelConfig

    console, window, node = _open()
    row = console._attach_logic_node(
        LogicNodeConfig(kind="task", name="probe", title="probe")
    )
    assert row is not None
    console._logic_nodes[id(row)] = node
    console._last_node[id(row)] = node
    card = console._new_panel_card(
        PanelConfig(kind="monitor", title="task", source="value = 0")
    )
    console._attach_card(card)
    console._running_task_row = row
    console._task_card = card
    console._task_output_node = node
    console._apply_task_lock(True)

    calls = []
    real_shutdown = card.shutdown

    def flaky_shutdown():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("card teardown failed")
        return real_shutdown()

    card.shutdown = flaky_shutdown
    assert console.shutdown(timeout=5.0) is False
    assert console._shutdown_state == "BLOCKED_UI_TEARDOWN"
    assert calls == [1] and card in console.cards
    assert console._deferred_task_card is card
    assert console.shutdown(timeout=5.0) is True
    assert calls == [1, 1] and card not in console.cards
    assert console._deferred_task_card is None
    window.close()


def test_ordinary_task_stop_keeps_card_and_row_until_render_barrier_acknowledges():
    from Zou_lab_control.frontend.task_console import LogicNodeConfig, PanelConfig

    console, window, node = _open()
    row = console._attach_logic_node(
        LogicNodeConfig(kind="task", name="probe", title="probe")
    )
    assert row is not None
    console._logic_nodes[id(row)] = node
    console._last_node[id(row)] = node
    card = console._new_panel_card(
        PanelConfig(kind="monitor", title="task", source="value = 0")
    )
    console._attach_card(card)
    console._running_task_row = row
    console._task_card = card
    console._task_output_node = node
    console._apply_task_lock(True)

    shutdown_calls = []
    real_card_shutdown = card.shutdown
    card.shutdown = lambda: (shutdown_calls.append(1), real_card_shutdown())[1]
    real_barrier = console._render_loop.barrier
    barrier_calls = []

    def delayed_barrier(timeout=5.0):
        barrier_calls.append(timeout)
        if len(barrier_calls) == 1:
            return False
        return real_barrier(timeout)

    console._render_loop.barrier = delayed_barrier
    assert console._stop_logic_node(row, timeout=0.1) is False
    assert shutdown_calls == [] and card in console.cards
    assert console._logic_nodes[id(row)] is node
    assert console._running_task_row is row and console._task_card is card
    assert console._stop_logic_node(row, timeout=5.0) is True
    assert shutdown_calls == [1] and card not in console.cards
    assert console._logic_nodes[id(row)] is None
    window.close()


def test_load_state_keeps_panel_when_render_barrier_is_unresolved():
    from Zou_lab_control.frontend.task_console import PanelConfig, TaskConsoleState

    console, window, _node = _open()
    card = console._new_panel_card(
        PanelConfig(kind="monitor", title="old", source="value = 0")
    )
    console._attach_card(card)
    shutdown_calls = []
    real_shutdown = card.shutdown
    card.shutdown = lambda: (shutdown_calls.append(1), real_shutdown())[1]
    real_barrier = console._render_loop.barrier
    console._render_loop.barrier = lambda timeout=5.0: False
    replacement = _FakeNode()
    try:
        with pytest.raises(RuntimeError, match="render worker still owns"):
            console.reseed(
                TaskConsoleState(name="new", panels=[]), running_nodes=[replacement]
            )
        assert shutdown_calls == []
        assert card in console.cards and console.state.name != "new"
        assert console.running_nodes == [_node]
    finally:
        console._render_loop.barrier = real_barrier
        window.close()
