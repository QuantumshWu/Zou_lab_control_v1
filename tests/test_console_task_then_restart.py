"""Generic TaskConsole lifecycle regression, independent of retired catalog entries.

A finite task that stops by itself must reach the same terminal endpoint as the
Stop button: it leaves ``running_nodes``, clears the live row reference and
releases the task-owned transient panel.  A stopped measurement can then restart
under its original signal names while an existing monitor keeps rendering the
last frame.

The lifecycle belongs to TaskConsole rather than to any product catalog entry, so
a small test-only task exercises it directly.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Zou_lab_control.neutral_atom as na
from conftest import add_logic_row, fire_live_imaging, make_console, raw_device_set, tick
from Zou_lab_control.neutral_atom.operations.logic import SignalSpec, Task
from Zou_lab_control.neutral_atom.operations.task import TaskSpec


class _FiniteTask(Task):
    """A controllable one-shot with a valid task-panel frame and no hardware."""

    _devices = {}
    mid_run = ("frame",)

    def __init__(self, hub, entered: threading.Event, release: threading.Event, *, prefix: str):
        super().__init__(hub, prefix=prefix)
        self._entered = entered
        self._release = release

    def _bare_output_specs(self):
        return (
            SignalSpec(
                "frame",
                "test frame",
                points_shape=(1,),
                data_shape=(2, 2),
                repeat_capacity=1,
            ),
        )

    def run(self, out):
        out.publish(frame=np.ones((1, 1, 2, 2), dtype=float))
        self._entered.set()
        while not self._release.is_set() and not self._stop.wait(0.01):
            pass
        return {"done": True}


def _wait_until(predicate, message: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def test_self_finished_task_releases_console_and_camera_can_restart():
    from Zou_lab_control.frontend.task_console import PanelConfig

    entered = threading.Event()
    release = threading.Event()

    def build(hub, *, prefix="finite_task_"):
        return _FiniteTask(hub, entered, release, prefix=prefix)

    spec = TaskSpec(
        name="Finite lifecycle test",
        build=build,
        prefix="finite_task_",
        mid_run_key="frame",
        default_kind="2d",
    )

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        # Install only the test spec into this console instance, then use the real
        # Add-Panel path.  Production discovery remains untouched.
        con.tasks.append(spec)
        con.kind_combo.addItem(f"Task: {spec.name}", ("task", spec.name))

        camrow = add_logic_row(con, ("camera", "live"))
        con._start_logic_node(camrow)
        fire_live_imaging(exp)
        con._logic_nodes[id(camrow)].step()
        card = con._new_panel_card(
            PanelConfig(
                kind="2d",
                title="IMG",
                size="4x4",
                source="value = frame_0",
                params={},
            )
        )
        con._attach_card(card)
        tick(con)
        assert card._status_error is False, card._status_text

        # Stop through the real endpoint; the lingering frame remains a valid
        # provider for the panel while the finite task owns the console.
        assert con._stop_logic_node(camrow)
        assert con._logic_nodes[id(camrow)] is None

        taskrow = add_logic_row(con, ("task", spec.name))
        con._start_logic_node(taskrow)
        _wait_until(entered.is_set, "finite task did not start")
        con._poll_logic_nodes()
        task_node = con._logic_nodes.get(id(taskrow)) or con._starting_nodes.get(id(taskrow))
        assert task_node is not None
        assert con._task_locked is True
        assert con._running_task_row is taskrow
        assert con._task_card is not None

        release.set()
        _wait_until(lambda: task_node.finished and not task_node.running,
                    "finite task did not finish")
        con._poll_logic_nodes()

        assert task_node not in con.running_nodes
        assert con._logic_nodes.get(id(taskrow)) is None
        assert con._running_task_row is None
        assert con._task_locked is False

        # Restarting the same camera row reclaims its original signal name and
        # declares image structure before another hardware trigger arrives.
        con._start_logic_node(camrow)
        _wait_until(lambda: con._logic_nodes.get(id(camrow)) is not None,
                    "camera did not restart")
        cam2 = con._logic_nodes[id(camrow)]
        frame_spec = next(item for item in cam2.output_specs() if item.name == "frame_0")
        assert frame_spec.data_shape == raw_device_set(exp).camera.frame_shape
        assert set(cam2.published_signals()) == {"frame_0"}

        card._render_version = -1
        tick(con)
        assert card._status_error is False, card._status_text
    finally:
        release.set()
        con.shutdown()
        exp.close()


def test_failed_task_releases_console_lock_on_lifecycle_poll():
    built = []

    class _FailingTask(Task):
        _devices = {}

        def run(self, out):
            raise RuntimeError("test task failure")

    def build(hub, *, prefix="failing_task_"):
        node = _FailingTask(hub, prefix=prefix)
        built.append(node)
        return node

    spec = TaskSpec(
        name="Failing lifecycle test",
        build=build,
        prefix="failing_task_",
    )
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        con.tasks.append(spec)
        con.kind_combo.addItem(f"Task: {spec.name}", ("task", spec.name))
        row = add_logic_row(con, ("task", spec.name))
        con._start_logic_node(row)
        _wait_until(lambda: bool(built), "failing task was not built")
        node = built[0]
        _wait_until(lambda: node.finished and not node.running,
                    "failing task did not reach terminal state")

        assert con._task_locked is True
        con._poll_logic_nodes()
        assert con._task_locked is False
        assert con._running_task_row is None
        assert con.kind_combo.isEnabled() is True
    finally:
        con.shutdown()
        exp.close()
