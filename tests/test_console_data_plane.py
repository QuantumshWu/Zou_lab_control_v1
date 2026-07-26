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
import threading
import time
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
from zlc_workbench.task_console.data_plane import ConsoleDataFront, ConsoleDataPlane

REPO = pathlib.Path(__file__).resolve().parents[1]


def _live_output(
    name: str,
    revision: int,
    digest: str,
    *,
    generation: str | None = None,
) -> LiveDatasetOutput:
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
        block.ref(
            StreamGenerationId(
                f"{name}-generation" if generation is None else generation
            )
        ),
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


class _GatedProcessorApplication:
    def __init__(
        self,
        output_names: tuple[str, ...],
        gates: dict[int, threading.Event],
    ) -> None:
        self._output_names = output_names
        self._gates = gates

    def evaluate(self, snapshot, _coverage, *, source_event_digest):
        revision = snapshot.ref.revision.value
        gate = self._gates.setdefault(revision, threading.Event())
        if not gate.wait(2.0):
            raise TimeoutError("test Processor gate did not open")
        outputs = {
            name: _live_output(
                name,
                revision,
                source_event_digest,
                generation=f"{name}-generation-{revision}",
            )
            for name in self._output_names
        }
        return SimpleNamespace(
            source_ref=snapshot.ref,
            source_event_digest=source_event_digest,
            outputs=outputs,
        )


class _GatedProcessorNode:
    def __init__(
        self,
        plane: ConsoleDataPlane,
        *,
        instance_id: str,
        source_name: str,
        declared_outputs: tuple[str, ...],
        gates: dict[int, threading.Event],
        published_outputs: tuple[str, ...] | None = None,
    ) -> None:
        self.instance_id = instance_id
        self.display_label = instance_id
        self._plane = plane
        self._source_name = source_name
        self._gates = gates
        self._published_outputs = (
            declared_outputs if published_outputs is None else published_outputs
        )
        self.output_declarations = tuple(
            SimpleNamespace(
                declaration=_live_output(name, 1, "a" * 64).declaration,
            )
            for name in declared_outputs
        )
        self.failure = None

    def signal_key(self, name: str) -> str:
        return f"{self.instance_id}/{name}"

    def _prepare_processor_application(self):
        return _GatedProcessorApplication(self._published_outputs, self._gates)

    def _validate_processor_source(self, source) -> None:
        if source.name != self._source_name:
            raise ValueError("wrong test source")

    @staticmethod
    def _processor_application_ready(_application) -> None:
        return None

    @staticmethod
    def _processor_work_started(_source) -> None:
        return None

    def _accept_processor_result(self, source, evaluation) -> None:
        self._plane.publish_processor(
            self,
            evaluation.outputs,
            source=source,
        )

    def _accept_processor_failure(self, error) -> None:
        self.failure = error

    @staticmethod
    def _accept_processor_cancelled() -> None:
        return None

    @staticmethod
    def _request_processor_owner_wake() -> None:
        return None


def _live_source_plane():
    plane = ConsoleDataPlane()
    state = {
        "frame": _live_output("frame", 1, "1" * 64),
        "frame_aux": _live_output("frame_aux", 1, "2" * 64),
    }
    node = SimpleNamespace(
        instance_id="camera",
        display_label="camera",
        output_declarations=tuple(
            SimpleNamespace(declaration=output.declaration)
            for output in state.values()
        ),
        signal_key=lambda name: f"camera/{name}",
    )
    slot = _slot(run="camera-run", epoch="camera-epoch", outputs={})
    slot.freeze_live_outputs = lambda: (
        "camera-run",
        "camera-epoch",
        dict(state),
    )
    plane.attach(node, slot)
    plane.mark_changed(node)
    first = plane.freeze()
    return plane, state, node, first


def _wait_for_signal_revision(
    plane: ConsoleDataPlane,
    name: str,
    revision: int,
) -> ConsoleDataFront:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        front = plane.freeze()
        value = front.value(name)
        if value is not None and value.snapshot.ref.revision.value == revision:
            return front
        time.sleep(0.005)
    raise AssertionError(f"{name} revision {revision} did not reach the data front")


def test_reactive_processor_wakes_only_for_its_declared_source() -> None:
    plane, _source_state, source_node, first = _live_source_plane()
    wakes: list[object] = []
    plane.bind_owner_wake(lambda: wakes.append(object()))

    gates = {1: threading.Event()}
    gates[1].set()
    processor = _GatedProcessorNode(
        plane,
        instance_id="occupancy",
        source_name="camera/frame",
        declared_outputs=("occupied",),
        gates=gates,
    )
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_source=first.value("camera/frame"),
    )

    thermometer_output = _live_output("temperature", 1, "b" * 64)
    thermometer = _node("thermometer", thermometer_output)
    plane.attach(
        thermometer,
        _slot(
            run="thermometer-run",
            epoch="thermometer-epoch",
            outputs={"temperature": thermometer_output},
        ),
    )

    plane.mark_changed(thermometer)
    assert wakes == []
    plane.mark_changed(source_node)
    assert len(wakes) == 1

    plane.cancel_latest_only_processor(processor)
    plane.mark_changed(source_node)
    assert len(wakes) == 1
    plane.close()


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


def test_source_and_processor_descendants_advance_as_one_local_component(
    request,
) -> None:
    """A slow Processor cannot expose source N beside derived N-1."""

    plane = ConsoleDataPlane()
    request.addfinalizer(plane.close)
    source_state = {
        "frame": _live_output("frame", 1, "1" * 64),
        "frame_aux": _live_output("frame_aux", 1, "3" * 64),
    }
    source_node = SimpleNamespace(
        instance_id="camera",
        display_label="camera",
        output_declarations=tuple(
            SimpleNamespace(declaration=output.declaration)
            for output in source_state.values()
        ),
        signal_key=lambda name: f"camera/{name}",
    )
    source_slot = _slot(
        run="camera-run",
        epoch="camera-epoch",
        outputs={},
    )
    source_slot.freeze_live_outputs = lambda: (
        "camera-run",
        "camera-epoch",
        dict(source_state),
    )
    plane.attach(source_node, source_slot)
    plane.mark_changed(source_node)
    first_source = plane.freeze().value("camera/frame")
    assert first_source is not None

    gates = {
        1: threading.Event(),
        2: threading.Event(),
        3: threading.Event(),
    }
    gates[1].set()

    class Application:
        def evaluate(self, snapshot, _coverage, *, source_event_digest):
            revision = snapshot.ref.revision.value
            if not gates[revision].wait(2.0):
                raise TimeoutError("test Processor gate did not open")
            output = _live_output(
                "occupied",
                revision,
                source_event_digest,
                generation=f"occupancy-generation-{revision}",
            )
            return SimpleNamespace(
                source_ref=snapshot.ref,
                source_event_digest=source_event_digest,
                outputs={"occupied": output},
            )

    class ProcessorNode:
        instance_id = "occupancy"
        display_label = "occupancy"

        def __init__(self) -> None:
            declaration = _live_output("occupied", 1, "a" * 64).declaration
            self.output_declarations = (
                SimpleNamespace(declaration=declaration),
            )
            self.failure = None

        @staticmethod
        def signal_key(name: str) -> str:
            return f"occupancy/{name}"

        @staticmethod
        def published_signals() -> tuple[str, ...]:
            return ("occupancy/occupied",)

        @staticmethod
        def _prepare_processor_application():
            return Application()

        @staticmethod
        def _validate_processor_source(source) -> None:
            if source.name != "camera/frame":
                raise ValueError("wrong test source")

        @staticmethod
        def _processor_application_ready(_application) -> None:
            return None

        @staticmethod
        def _processor_work_started(_source) -> None:
            return None

        def _accept_processor_result(self, source, evaluation) -> None:
            plane.publish_processor(
                self,
                evaluation.outputs,
                source=source,
            )

        def _accept_processor_failure(self, error) -> None:
            self.failure = error

        @staticmethod
        def _accept_processor_cancelled() -> None:
            return None

        @staticmethod
        def _request_processor_owner_wake() -> None:
            return None

    processor = ProcessorNode()
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_source=first_source,
    )

    def wait_for_revision(revision: int) -> ConsoleDataFront:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            front = plane.freeze()
            occupied = front.value("occupancy/occupied")
            if (
                occupied is not None
                and occupied.snapshot.ref.revision.value == revision
            ):
                return front
            time.sleep(0.005)
        raise AssertionError("Processor result did not reach the data front")

    first = wait_for_revision(1)
    assert first.value("camera/frame").snapshot.ref.revision.value == 1

    independent_state = {
        "output": _live_output("temperature", 1, "b" * 64),
    }
    independent = _node("thermometer", independent_state["output"])
    independent_slot = _slot(run="thermometer-run", epoch="thermometer-epoch", outputs={})
    independent_slot.freeze_live_outputs = lambda: (
        "thermometer-run",
        "thermometer-epoch",
        {"temperature": independent_state["output"]},
    )
    plane.attach(independent, independent_slot)
    plane.mark_changed(independent)
    assert plane.freeze().value("thermometer/temperature") is not None

    source_state["frame"] = _live_output("frame", 2, "2" * 64)
    source_state["frame_aux"] = _live_output("frame_aux", 2, "4" * 64)
    independent_state["output"] = _live_output("temperature", 2, "c" * 64)
    plane.mark_changed(source_node)
    plane.mark_changed(independent)
    staged = plane.freeze()
    assert staged.value("camera/frame").snapshot.ref.revision.value == 1
    assert staged.value("camera/frame_aux").snapshot.ref.revision.value == 1
    assert staged.value("occupancy/occupied").snapshot.ref.revision.value == 1
    assert staged.value("thermometer/temperature").snapshot.ref.revision.value == 2

    # Keep the source moving while revision 2 is in flight.  The lane must
    # retain only revision 3 as pending, but completed revision 2 still needs
    # to become one coherent presentation front instead of being starved by
    # the newer candidate forever.
    source_state["frame"] = _live_output("frame", 3, "5" * 64)
    source_state["frame_aux"] = _live_output("frame_aux", 3, "6" * 64)
    plane.mark_changed(source_node)
    assert plane.freeze().value("camera/frame").snapshot.ref.revision.value == 1
    entry = plane._processor_lane._entries[id(processor)]
    assert entry.work_source.snapshot.ref.revision.value == 2
    assert entry.pending_source.snapshot.ref.revision.value == 3

    gates[2].set()
    deadline = time.monotonic() + 2.0
    while not entry.work_future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert entry.work_future.done()

    source_state["frame"] = _live_output("frame", 4, "7" * 64)
    source_state["frame_aux"] = _live_output("frame_aux", 4, "8" * 64)
    plane.mark_changed(source_node)
    second = plane.freeze()
    assert second.value("camera/frame").snapshot.ref.revision.value == 2
    assert second.value("camera/frame_aux").snapshot.ref.revision.value == 2
    assert second.value("occupancy/occupied").snapshot.ref.revision.value == 2
    assert processor.failure is None

    gates[3].set()
    plane.cancel_latest_only_processor(processor)
    withdrawn = plane.freeze()
    assert withdrawn.value("camera/frame") is not None
    assert withdrawn.value("occupancy/occupied") is None


def test_processor_chain_presents_the_completed_causal_edge_closure() -> None:
    """A downstream completion may retain upstream N after upstream reached M."""

    plane, source_state, source_node, first = _live_source_plane()
    upstream_gates = {
        revision: threading.Event()
        for revision in range(1, 7)
    }
    downstream_gates = {
        revision: threading.Event()
        for revision in range(1, 7)
    }
    upstream_gates[1].set()
    downstream_gates[1].set()
    try:
        upstream = _GatedProcessorNode(
            plane,
            instance_id="upstream",
            source_name="camera/frame",
            declared_outputs=("classified",),
            gates=upstream_gates,
        )
        plane.attach_latest_only_processor(
            upstream,
            source_name="camera/frame",
            initial_source=first.value("camera/frame"),
        )
        upstream_first = _wait_for_signal_revision(
            plane,
            "upstream/classified",
            1,
        )

        downstream = _GatedProcessorNode(
            plane,
            instance_id="downstream",
            source_name="upstream/classified",
            declared_outputs=("decision",),
            gates=downstream_gates,
        )
        plane.attach_latest_only_processor(
            downstream,
            source_name="upstream/classified",
            initial_source=upstream_first.value("upstream/classified"),
        )
        _wait_for_signal_revision(plane, "downstream/decision", 1)

        source_state["frame"] = _live_output("frame", 2, "3" * 64)
        source_state["frame_aux"] = _live_output("frame_aux", 2, "4" * 64)
        plane.mark_changed(source_node)
        plane.freeze()
        source_state["frame"] = _live_output("frame", 3, "5" * 64)
        source_state["frame_aux"] = _live_output("frame_aux", 3, "6" * 64)
        plane.mark_changed(source_node)
        plane.freeze()

        upstream_entry = plane._processor_lane._entries[id(upstream)]
        assert upstream_entry.work_source.snapshot.ref.revision.value == 2
        assert upstream_entry.pending_source.snapshot.ref.revision.value == 3
        upstream_gates[2].set()
        deadline = time.monotonic() + 2.0
        while (
            not upstream_entry.work_future.done()
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert upstream_entry.work_future.done()

        source_state["frame"] = _live_output("frame", 4, "7" * 64)
        source_state["frame_aux"] = _live_output("frame_aux", 4, "8" * 64)
        plane.mark_changed(source_node)
        plane.freeze()
        upstream_gates[3].set()
        deadline = time.monotonic() + 2.0
        while (
            not upstream_entry.work_future.done()
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert upstream_entry.work_future.done()

        source_state["frame"] = _live_output("frame", 5, "9" * 64)
        source_state["frame_aux"] = _live_output("frame_aux", 5, "a" * 64)
        plane.mark_changed(source_node)
        plane.freeze()
        downstream_entry = plane._processor_lane._entries[id(downstream)]
        assert downstream_entry.work_source.snapshot.ref.revision.value == 2
        downstream_gates[2].set()
        deadline = time.monotonic() + 2.0
        while (
            not downstream_entry.work_future.done()
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert downstream_entry.work_future.done()

        completed = plane.freeze()
        assert completed.value("camera/frame").snapshot.ref.revision.value == 2
        assert completed.value("camera/frame_aux").snapshot.ref.revision.value == 2
        assert completed.value(
            "upstream/classified"
        ).snapshot.ref.revision.value == 2
        assert completed.value(
            "downstream/decision"
        ).snapshot.ref.revision.value == 2
    finally:
        for gate in (*upstream_gates.values(), *downstream_gates.values()):
            gate.set()
        plane.close()


def test_fan_out_can_bind_the_exact_staged_source_front() -> None:
    """A second Processor binds visible M even when the raw cache is newer."""

    plane, source_state, source_node, first = _live_source_plane()
    first_gates = {
        revision: threading.Event()
        for revision in range(1, 6)
    }
    second_gates = {
        revision: threading.Event()
        for revision in range(1, 6)
    }
    first_gates[1].set()
    try:
        first_processor = _GatedProcessorNode(
            plane,
            instance_id="first",
            source_name="camera/frame",
            declared_outputs=("classified",),
            gates=first_gates,
        )
        plane.attach_latest_only_processor(
            first_processor,
            source_name="camera/frame",
            initial_source=first.value("camera/frame"),
        )
        _wait_for_signal_revision(plane, "first/classified", 1)

        source_state["frame"] = _live_output("frame", 2, "3" * 64)
        source_state["frame_aux"] = _live_output("frame_aux", 2, "4" * 64)
        plane.mark_changed(source_node)
        plane.freeze()
        source_state["frame"] = _live_output("frame", 3, "5" * 64)
        source_state["frame_aux"] = _live_output("frame_aux", 3, "6" * 64)
        plane.mark_changed(source_node)
        plane.freeze()
        first_entry = plane._processor_lane._entries[id(first_processor)]
        first_gates[2].set()
        deadline = time.monotonic() + 2.0
        while not first_entry.work_future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert first_entry.work_future.done()

        source_state["frame"] = _live_output("frame", 4, "7" * 64)
        source_state["frame_aux"] = _live_output("frame_aux", 4, "8" * 64)
        plane.mark_changed(source_node)
        staged = plane.freeze()
        assert staged.value("camera/frame").snapshot.ref.revision.value == 2

        second_processor = _GatedProcessorNode(
            plane,
            instance_id="second",
            source_name="camera/frame",
            declared_outputs=("judged",),
            gates=second_gates,
        )
        plane.attach_latest_only_processor(
            second_processor,
            source_name="camera/frame",
            initial_source=staged.value("camera/frame"),
        )
        second_entry = plane._processor_lane._entries[id(second_processor)]
        retained = second_entry.pending_source_component
        assert retained.signals["camera/frame"].snapshot.ref.revision.value == 2
        assert retained.signals[
            "camera/frame_aux"
        ].snapshot.ref.revision.value == 2
    finally:
        for gate in (*first_gates.values(), *second_gates.values()):
            gate.set()
        plane.close()


def test_processor_publication_requires_every_declared_sibling() -> None:
    plane, _source_state, _source_node, first = _live_source_plane()
    gates = {1: threading.Event()}
    gates[1].set()
    try:
        processor = _GatedProcessorNode(
            plane,
            instance_id="partial",
            source_name="camera/frame",
            declared_outputs=("left", "right"),
            published_outputs=("left",),
            gates=gates,
        )
        plane.attach_latest_only_processor(
            processor,
            source_name="camera/frame",
            initial_source=first.value("camera/frame"),
        )
        deadline = time.monotonic() + 2.0
        while processor.failure is None and time.monotonic() < deadline:
            plane.freeze()
            time.sleep(0.005)
        assert isinstance(processor.failure, ValueError)
        assert "complete frozen output vocabulary" in str(processor.failure)
        front = plane.freeze()
        assert front.value("partial/left") is None
        assert front.value("partial/right") is None
    finally:
        plane.close()
