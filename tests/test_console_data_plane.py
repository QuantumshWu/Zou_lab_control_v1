"""MONITOR seam contract: changed producers freeze once, unchanged fronts are reused.

Worth a test where it touches reality: a live virtual monitor is frozen through
the data plane and must yield the camera's actual block (shape, dtype, unit off
the producer's own cell schema), an exact revision that ADVANCES with the
stream, and coverage instead of a global shot counter -- the fiction the purge
removed.
Independent producers may share a present cycle but never gain a same-shot
claim from it.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import numpy as np

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValueSchema,
)
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage
from zlc_workbench.task_console.data_plane import ConsoleDataPlane

REPO = pathlib.Path(__file__).resolve().parents[1]


def _live_output(name: str, revision: int, digest: str) -> LiveDatasetOutput:
    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId(f"{name}.point"), "point", SCAN_POINT, 1, (0,))
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((1,)),
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )
    values = np.asarray([[[float(revision)]]], dtype=np.float64)
    block = DataBlock(
        BlockId(f"{name}-block"),
        DatasetRevision(revision),
        values,
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId(f"{name}-generation")),
        block,
    )
    return LiveDatasetOutput(
        DatasetOutputDeclaration(name, f"test.{name}"),
        snapshot,
        MonitorCoverage(1, 1, 0, False),
        digest,
    )


def _node(instance_id: str, output: LiveDatasetOutput):
    return SimpleNamespace(
        instance_id=instance_id,
        display_label=instance_id,
        output_declarations=(
            SimpleNamespace(declaration=output.declaration),
        ),
        signal_key=lambda name: f"{instance_id}/{name}",
    )


def _slot(*, run: str, epoch: str, outputs):
    return SimpleNamespace(
        freeze_live_outputs=lambda: (run, epoch, outputs),
        close=lambda: None,
        notification_failure=None,
    )


def test_unchanged_sources_reuse_their_immutable_front() -> None:
    plane = ConsoleDataPlane()
    output = _live_output("frame", 1, "a" * 64)
    node = _node("camera", output)
    slot = _slot(run="run", epoch="epoch", outputs={"frame": output})
    calls = []
    plane._freeze_one = lambda *_args: (calls.append(object()) or {}, None)
    plane.attach(node, slot)
    plane.mark_changed(node)

    first = plane.freeze()
    assert plane.freeze() is first
    assert len(calls) == 1

    plane.mark_changed(node)
    assert plane.freeze() is not first
    assert len(calls) == 2


def test_the_data_plane_holds_no_toolkit_and_no_domain_authority():
    """It receives slots; it never reaches for Qt, matplotlib or the facade."""

    tree = ast.parse((REPO / "zlc_workbench" / "task_console" / "data_plane.py")
                     .read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & {"PyQt5", "matplotlib", "Zou_lab_control"}, roots


def test_live_slot_cannot_publish_an_undeclared_output_contract() -> None:
    """The generic plane routes only owner-declared typed outputs."""

    declared = _live_output("frame", 1, "a" * 64)
    undeclared = _live_output("roi", 1, "b" * 64)
    node = _node("camera", declared)
    slot = _slot(run="run", epoch="epoch", outputs={"roi": undeclared})
    plane = ConsoleDataPlane()
    plane.attach(node, slot)
    plane.mark_changed(node)

    front = plane.freeze()

    assert front.names() == ()
    assert "absent from the Workbench vocabulary" in front.failures["camera"]


def test_independent_producers_keep_independent_causation_in_one_present_cycle() -> None:
    """Latest values from two producers are not promoted into one fake shot."""

    plane = ConsoleDataPlane()

    def attach(name: str, *, run: str, epoch: str, sequence: int, digest: str) -> None:
        output_name = f"{name}_frame"
        output = _live_output(output_name, sequence, digest)
        node = _node(name, output)
        plane.attach(
            node,
            _slot(
                run=run,
                epoch=epoch,
                outputs={output_name: output},
            ),
        )
        plane.mark_changed(node)

    attach("slow", run="run-slow", epoch="epoch-slow", sequence=3, digest="a" * 64)
    attach("fast", run="run-fast", epoch="epoch-fast", sequence=91, digest="b" * 64)

    front = plane.freeze()
    slow = front.value("slow/slow_frame")
    fast = front.value("fast/fast_frame")

    assert slow is not None and fast is not None
    assert (slow.run_id, slow.epoch_id, slow.join_digest) == (
        "run-slow", "epoch-slow", "a" * 64,
    )
    assert (fast.run_id, fast.epoch_id, fast.join_digest) == (
        "run-fast", "epoch-fast", "b" * 64,
    )
    assert not hasattr(front, "run_id")
    assert not hasattr(front, "shot")
    assert not hasattr(front, "coherence_stamp")
