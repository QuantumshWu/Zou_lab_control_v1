"""Contract: closing the task-console window tears the console down (FRONTEND).

The console is a CHILD of the Fluent window, so its own ``closeEvent`` never fires
on a window close.  ``show_task_console`` therefore wires the window's ``closed``
signal to ``console.shutdown``.  This pins the three contracts that keeps a closed
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

    def stop(self):
        self.stops += 1
        self.running = False

    def step(self):
        pass

    def published_signals(self):
        return ()


def _open(on_close=None):
    from Zou_lab_control.frontend.task_console import show_task_console
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    node = _FakeNode()
    console = show_task_console(hub=SignalHub(), running_nodes=[node], on_close=on_close)
    window = ensure_qt_app()._zlc_retained_windows[-1]   # the ONE retain_window registry
    return console, window, node


def test_close_stops_nodes_timer_and_calls_on_close():
    calls = []
    console, window, node = _open(on_close=lambda: calls.append(1))
    assert node.running and node.starts == 1        # show_task_console auto-started it
    assert console._timer.isActive()
    window.closed.emit()                            # a genuine close
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
    window.closed.emit()
    assert node.stops == 1                          # stopped exactly once
