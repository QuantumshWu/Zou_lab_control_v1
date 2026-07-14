"""Contracts for registering and validating task-catalog entries."""

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




def test_register_and_unregister_roundtrip():
    from Zou_lab_control.neutral_atom.operations.task import TaskSpec
    from Zou_lab_control.neutral_atom.operations import task_registry as reg

    def factory(readout):
        return TaskSpec(
            name="Ad-hoc task",
            build=lambda hub, *, prefix="ad_": None,
            mid_run_key="frame",
            prefix="ad_",
        )

    reg.register_task(factory)
    try:
        names = [spec.name for spec in reg.discovered_task_specs(_virtual_readout())]
        assert "Ad-hoc task" in names
    finally:
        assert reg.unregister_task(factory) is True
    assert "Ad-hoc task" not in {
        spec.name for spec in reg.discovered_task_specs(_virtual_readout())
    }


def test_two_registered_tasks_sharing_a_prefix_fail_loud():
    from Zou_lab_control.neutral_atom.operations.task import TaskSpec
    from Zou_lab_control.neutral_atom.operations import task_registry as reg

    def first(readout):
        return TaskSpec(name="First", build=lambda hub, *, prefix="same_": None, prefix="same_")

    def second(readout):
        return TaskSpec(name="Second", build=lambda hub, *, prefix="same_": None, prefix="same_")

    reg.register_task(first)
    reg.register_task(second)
    try:
        with pytest.raises(ValueError):
            reg.discovered_task_specs(_virtual_readout())
    finally:
        reg.unregister_task(second)
        reg.unregister_task(first)


def test_built_node_prefix_matches_spec_prefix_for_every_builtin_task():
    """Every remaining built-in task keeps its catalog namespace contract."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    specs = _virtual_readout().task_specs()
    assert specs, "no usable task specs discovered"
    for spec in specs:
        node = spec.build(SignalHub())
        assert node.prefix == spec.prefix
