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


def test_built_node_prefix_matches_spec_prefix_for_every_task():
    """TaskSpec.prefix is the ONE namespace source: a BARE ``spec.build(hub)`` (the console's
    task branch passes no prefix) must yield a node carrying exactly ``spec.prefix``, for EVERY
    discovered task -- so the discovery collision guard (``collision_key``) always checks the
    token the built node actually carries.  A build-closure / ctor default drifting away from
    the spec's prefix fails here."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    readout = _virtual_readout()
    specs = readout.task_specs()
    assert specs, "no task specs discovered"
    for spec in specs:
        node = spec.build(SignalHub())
        assert node.prefix == spec.prefix, (
            f"task {spec.name!r}: bare build() produced prefix {node.prefix!r} but the spec "
            f"declares {spec.prefix!r} -- the closure/ctor default drifted from TaskSpec.prefix.")


def test_notebook_funnel_prefix_default_is_the_spec_constant():
    """The notebook funnel (``ReadoutSubsystem.calibrate_task``) defaults its ``prefix`` to the
    SAME constant object the Calibrate TaskSpec carries (``CAL_TASK_PREFIX``) -- one token,
    never a retyped literal in the subsystem layer."""
    import inspect

    from Zou_lab_control.neutral_atom.operations.tasks.calibrate import CAL_TASK_PREFIX
    from Zou_lab_control.neutral_atom.subsystems.readout import ReadoutSubsystem

    default = inspect.signature(ReadoutSubsystem.calibrate_task).parameters["prefix"].default
    assert default is CAL_TASK_PREFIX
    spec = next(s for s in _virtual_readout().task_specs() if s.name == "Calibrate readout")
    assert spec.prefix is CAL_TASK_PREFIX
