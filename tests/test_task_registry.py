"""Contract: the TASK catalog is an OPEN registry (mirrors the processor registry).

The orchestration tier is pluggable the same way measurements / processors are:
the built-in calibrate task is auto-discovered, a notebook can ``register_task`` an
ad-hoc one, and two tasks sharing a hub ``prefix`` FAIL LOUD (their namespaced
signals would clobber each other on the shared SignalHub).  Routed through the real
``exp.readout.task_specs()`` so it exercises the same path the GUI uses.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def _virtual_readout():
    import Zou_lab_control.neutral_atom as na

    return na.connect("virtual", sitemap={"grid_shape": (2, 3)}).readout


def test_builtin_calibrate_task_is_discovered():
    specs = _virtual_readout().task_specs()
    by_name = {s.name: s for s in specs}
    assert "Calibrate readout" in by_name
    spec = by_name["Calibrate readout"]
    assert spec.mid_run_key == "frame"               # buffer key the dedicated panel shows
    assert spec.prefix == "cal_"


def test_build_returns_an_unrun_task_over_the_session():
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CalibrateReadoutTask

    readout = _virtual_readout()
    spec = next(s for s in readout.task_specs() if s.name == "Calibrate readout")
    hub = SignalHub()
    task = spec.build(hub)
    assert isinstance(task, CalibrateReadoutTask)
    assert task.layer == "task" and task.node_label == "calibrate"
    assert not task.finished                          # UNRUN until started


def test_register_and_unregister_roundtrip():
    from Zou_lab_control.neutral_atom.operations.task import TaskSpec
    from Zou_lab_control.neutral_atom.operations import task_registry as reg

    def factory(readout):
        return TaskSpec(name="Ad-hoc task", build=lambda hub, *, prefix="ad_": None,
                        mid_run_key="frame", prefix="ad_")

    reg.register_task(factory)
    try:
        names = [s.name for s in reg.discovered_task_specs(_virtual_readout())]
        assert "Ad-hoc task" in names and "Calibrate readout" in names
    finally:
        assert reg.unregister_task(factory) is True
    assert "Ad-hoc task" not in [s.name for s in reg.discovered_task_specs(_virtual_readout())]


def test_two_tasks_sharing_a_prefix_fail_loud():
    from Zou_lab_control.neutral_atom.operations.task import TaskSpec
    from Zou_lab_control.neutral_atom.operations import task_registry as reg

    # a second task colliding on the built-in calibrate task's "cal_" prefix
    def clasher(readout):
        return TaskSpec(name="Clasher", build=lambda hub, *, prefix="cal_": None, prefix="cal_")

    reg.register_task(clasher)
    try:
        with pytest.raises(ValueError):
            reg.discovered_task_specs(_virtual_readout())
    finally:
        reg.unregister_task(clasher)
