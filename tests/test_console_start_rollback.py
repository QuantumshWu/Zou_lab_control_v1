"""Contract: _start_logic_node is BUILD-VALIDATE-COMMIT (#A2) and the device-driver
mutual exclusion covers ALL running nodes (#A3).

A2: a failed BUILD leaves the currently-running node untouched (the old order stopped
it first, so one bad param edit killed a live acquisition); a failed START leaves the
registries clean (no half-started ghost in running_nodes / _logic_nodes, and the
previous build's hub signals are NOT unlinked).

A3: the exclusion's single source is ``running_nodes`` + each node's own
``drives_devices`` declaration -- a notebook-injected node (no GUI row) is stopped by
a GUI Start exactly like a row-owned driver, and a reactive processor
(drives_devices=False) keeps running.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.core.signals import SignalHub


def _console(exp, running_nodes=None):
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state

    ensure_qt_app()
    return TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                       measurements=exp.readout.measurement_specs(),
                       processors=exp.readout.processor_specs(),
                       tasks=exp.readout.task_specs(), window_px=(900, 600),
                       running_nodes=list(running_nodes or ()))


def _camera_row(con):
    """Add a camera logic row through the REAL Add-Panel path (same as the startstop tests)."""
    kc = con.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == ("camera", "live"))
    kc.setCurrentIndex(i)
    con._add_panel()
    con._timer.stop()                                    # deterministic: no background ticks
    return con.logic_nodes[-1]


class _FakeInjected:
    """A notebook-injected device driver: no GUI row, only the node contract."""

    drives_devices = True

    def __init__(self):
        self.stopped = False
        self.running = True

    def published_signals(self):
        return frozenset()

    def stop(self):
        self.stopped = True
        self.running = False


def test_failed_build_leaves_the_running_node_untouched():
    exp = na.connect("virtual")
    con = _console(exp)
    try:
        row = _camera_row(con)
        con._start_logic_node(row)
        node = con._logic_nodes[id(row)]
        assert node in con.running_nodes and node.running

        def _boom(*a, **k):
            raise RuntimeError("bad params")

        con._build_logic_node = _boom                        # the NEXT build fails
        con._start_logic_node(row)
        assert con._logic_nodes[id(row)] is node             # old node still registered...
        assert node in con.running_nodes and node.running    # ...and still acquiring
    finally:
        con.shutdown()
        exp.close()


def test_failed_start_registers_nothing_and_keeps_old_signals():
    from conftest import fire_live_imaging

    exp = na.connect("virtual")
    con = _console(exp)
    try:
        row = _camera_row(con)
        con._start_logic_node(row)
        old = con._logic_nodes[id(row)]
        fire_live_imaging(exp)                               # the camera is passive: fire, then step
        old.step()                                           # one synchronous frame into the hub
        old_signals = set(old.published_signals())
        assert old_signals and all(con.hub.latest(s) is not None for s in old_signals)

        class _DeadNode(_FakeInjected):
            instance_label = ""
            rate_hz = 5.0

            def start(self, **k):
                raise RuntimeError("wedged device")

        con._build_logic_node = lambda *a, **k: _DeadNode()
        con._start_logic_node(row)
        dead = [n for n in con.running_nodes if isinstance(n, _DeadNode)]
        assert not dead                                      # no half-started ghost (#A2 commit point)
        # the previous build's signals were NOT unlinked -- a failed start behaves like a plain Stop
        assert all(con.hub.latest(s) is not None for s in old_signals)
    finally:
        con.shutdown()
        exp.close()


def test_injected_device_driver_obeys_the_same_exclusion():
    exp = na.connect("virtual")
    injected = _FakeInjected()
    con = _console(exp, running_nodes=[injected])
    try:
        assert injected in con.running_nodes
        row = _camera_row(con)
        con._start_logic_node(row)                           # a GUI device driver starts
        assert injected.stopped                              # the row-less driver was stopped...
        assert injected not in con.running_nodes             # ...and dropped from the registry (#A3)
    finally:
        con.shutdown()
        exp.close()


def test_reactive_processor_keeps_running_across_a_driver_start():
    from Zou_lab_control.neutral_atom.operations.logic import Processor

    assert Processor.drives_devices is False                 # the node's own declaration (#6)

    class _FakeProcessor(_FakeInjected):
        drives_devices = False

    exp = na.connect("virtual")
    proc = _FakeProcessor()
    con = _console(exp, running_nodes=[proc])
    try:
        con._start_logic_node(_camera_row(con))
        assert not proc.stopped and proc in con.running_nodes
    finally:
        con.shutdown()
        exp.close()
