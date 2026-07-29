"""Current signal-plane contracts: exact publications and causal fronts."""

from __future__ import annotations

import ast
import os
import pathlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

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
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValueSchema,
)
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_neutral_atom.processing.signal_plane import (
    SignalDataPlane,
    SignalFront,
    SignalPublication,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage


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
        PointTable(
            1,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    point.coordinates,
                ),
            ),
        ),
        None,
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


def _node(instance_id: str, outputs: tuple[LiveDatasetOutput, ...]):
    return SimpleNamespace(
        instance_id=instance_id,
        dataset_output_declarations=tuple(output.declaration for output in outputs),
        signal_key=lambda name: f"{instance_id}/{name}",
    )


def _attach_live(
    plane: SignalDataPlane,
    node,
    state: dict[str, LiveDatasetOutput],
    *,
    run: str,
    epoch: str,
    calls: list[object] | None = None,
):
    def freeze_live_outputs():
        if calls is not None:
            calls.append(object())
        return run, epoch, dict(state)

    slot = SimpleNamespace(
        freeze_live_outputs=freeze_live_outputs,
        close=lambda: None,
        notification_failure=None,
    )
    plane.reserve(node)
    lifecycle = plane.begin_run_lifecycle(node)
    plane.bind_run_lifecycle(lifecycle, run, preemptible=True)
    plane.attach(node, slot)
    plane.mark_changed(node)
    return slot


def _live_source_plane():
    plane = SignalDataPlane()
    state = {
        "frame": _live_output("frame", 1, "1" * 64),
        "frame_aux": _live_output("frame_aux", 1, "2" * 64),
    }
    node = _node("camera", tuple(state.values()))
    _attach_live(
        plane,
        node,
        state,
        run="camera-run",
        epoch="camera-epoch",
    )
    first = plane.freeze()
    return plane, state, node, first


def test_event_route_freezes_generation_before_any_publication() -> None:
    """A passive trigger source is bindable before the event that FIRE creates."""

    plane = SignalDataPlane()
    output = _live_output("frame", 1, "1" * 64)
    node = _node("passive-camera", (output,))
    source = SimpleNamespace(
        value_schema=lambda _name: output.snapshot.block.schema.cell_schema,
        open_signal_cursor=lambda _name: object(),
    )
    slot = SimpleNamespace(
        freeze_live_outputs=lambda: ("camera-run", "camera-epoch", {"frame": output}),
        close=lambda: None,
        notification_failure=None,
    )
    try:
        generation = plane.reserve(node)
        plane.attach(node, slot, event_source=source)
        assert plane.latest_publication("passive-camera/frame") is None
        frozen = plane.signal_event_binding("passive-camera/frame")
        assert frozen == (generation, source, "frame", None)

        plane.retire(node)
        replacement = _node("passive-camera", (output,))
        replacement_generation = plane.reserve(replacement)
        plane.attach(replacement, slot, event_source=source)
        assert replacement_generation > generation
        with pytest.raises(RuntimeError, match="generation changed"):
            plane.signal_event_binding(
                "passive-camera/frame",
                expected_generation=generation,
            )
    finally:
        plane.close()


def test_preemption_keeps_a_parent_with_an_unadmitted_exact_descendant() -> None:
    plane = SignalDataPlane()
    parent = _node("parent", (_live_output("source", 1, "1" * 64),))
    child = _node("child", (_live_output("result", 1, "2" * 64),))
    parent_command = object()
    child_command = object()
    try:
        plane.reserve(parent)
        plane.bind_lifecycle_owner(parent, parent_command)
        parent_ref = plane.begin_run_lifecycle(parent_command)
        plane.bind_run_lifecycle(parent_ref, "parent-run", preemptible=True)

        plane.reserve(child)
        plane.bind_lifecycle_owner(
            child,
            child_command,
            parent_owners=(parent_command,),
        )

        assert plane.retire_preemptible_run_closure(("parent-run",)) is None
        child_ref = plane.begin_run_lifecycle(child_command)
        assert plane.abort_run_lifecycle(child_ref)
        plane.finish_run_lifecycle("parent-run")
    finally:
        plane.close()


def test_preemption_closes_live_slot_only_after_safe_run_release() -> None:
    plane = SignalDataPlane()
    output = _live_output("source", 1, "1" * 64)
    parent = _node("parent", (output,))
    closed = []
    slot = SimpleNamespace(
        freeze_live_outputs=lambda: ("parent-run", "parent-epoch", {"source": output}),
        close=lambda: closed.append(True),
        notification_failure=None,
    )
    command = object()
    try:
        plane.reserve(parent)
        plane.bind_lifecycle_owner(parent, command)
        lifecycle = plane.begin_run_lifecycle(command)
        plane.bind_run_lifecycle(lifecycle, "parent-run", preemptible=True)
        plane.attach(parent, slot)

        assert plane.retire_preemptible_run_closure(("parent-run",)) == (
            "parent-run",
        )
        assert closed == []
        assert len(plane) == 0

        assert plane.finish_preemptible_run_retirement(("parent-run",)) == ()
        assert closed == [True]
    finally:
        plane.close()


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
        plane: SignalDataPlane,
        *,
        instance_id: str,
        source_name: str,
        declared_outputs: tuple[str, ...],
        gates: dict[int, threading.Event],
        published_outputs: tuple[str, ...] | None = None,
    ) -> None:
        self.instance_id = instance_id
        self._plane = plane
        self._source_name = source_name
        self._gates = gates
        self._published_outputs = (
            declared_outputs if published_outputs is None else published_outputs
        )
        self.dataset_output_declarations = tuple(
            _live_output(name, 1, "a" * 64).declaration
            for name in declared_outputs
        )
        self.failure: Exception | None = None
        self.cancelled = False

    def signal_key(self, name: str) -> str:
        return f"{self.instance_id}/{name}"

    def prepare_processor_application(self):
        return _GatedProcessorApplication(self._published_outputs, self._gates)

    def validate_processor_source(self, source) -> None:
        if source.name != self._source_name:
            raise ValueError("wrong test source")

    @staticmethod
    def processor_application_ready(_application) -> None:
        return None

    @staticmethod
    def processor_work_started(_source) -> None:
        return None

    def accept_processor_result(
        self,
        _source,
        source_publication: SignalPublication,
        evaluation,
    ) -> None:
        self._plane.publish_processor(
            self,
            evaluation.outputs,
            source_publication=source_publication,
        )

    def accept_processor_failure(self, error) -> None:
        self.failure = error

    def accept_processor_cancelled(self) -> None:
        self.cancelled = True

    @staticmethod
    def request_processor_owner_wake() -> None:
        return None


def _wait_for_signal_revision(
    plane: SignalDataPlane,
    name: str,
    revision: int,
) -> SignalFront:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        front = plane.freeze()
        value = front.value(name)
        if value is not None and value.snapshot.ref.revision.value == revision:
            return front
        time.sleep(0.005)
    raise AssertionError(f"{name} revision {revision} did not reach the data front")


def test_unchanged_sources_reuse_their_exact_publication_and_front() -> None:
    plane = SignalDataPlane()
    output = _live_output("frame", 1, "a" * 64)
    state = {"frame": output}
    node = _node("camera", (output,))
    calls: list[object] = []
    _attach_live(
        plane,
        node,
        state,
        run="run",
        epoch="epoch",
        calls=calls,
    )
    try:
        first = plane.freeze()
        publication = first.publication("camera/frame")
        assert isinstance(publication, SignalPublication)
        assert publication.value("camera/frame") is first.value("camera/frame")
        assert plane.freeze() is first
        assert len(calls) == 1

        state["frame"] = _live_output("frame", 2, "b" * 64)
        plane.mark_changed(node)
        second = plane.freeze()
        assert second is not first
        assert second.publication("camera/frame") is not publication
        assert len(calls) == 2
    finally:
        plane.close()


def test_the_data_plane_holds_no_toolkit_or_domain_authority() -> None:
    tree = ast.parse(
        (REPO / "zlc_neutral_atom" / "processing" / "signal_plane.py").read_text(
            encoding="utf-8"
        )
    )
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & {"PyQt5", "matplotlib", "Zou_lab_control"}, roots


def test_live_slot_cannot_publish_an_undeclared_output_contract() -> None:
    declared = _live_output("frame", 1, "a" * 64)
    undeclared = _live_output("roi", 1, "b" * 64)
    node = _node("camera", (declared,))
    plane = SignalDataPlane()
    _attach_live(
        plane,
        node,
        {"roi": undeclared},
        run="run",
        epoch="epoch",
    )
    try:
        front = plane.freeze()
        assert front.names() == ()
        assert "undeclared output" in front.failures["camera"]
    finally:
        plane.close()


def test_independent_producers_keep_independent_publications() -> None:
    plane = SignalDataPlane()
    slow = _live_output("frame", 3, "a" * 64)
    fast = _live_output("frame", 91, "b" * 64, generation="fast-generation")
    slow_node = _node("slow", (slow,))
    fast_node = _node("fast", (fast,))
    _attach_live(
        plane,
        slow_node,
        {"frame": slow},
        run="run-slow",
        epoch="epoch-slow",
    )
    _attach_live(
        plane,
        fast_node,
        {"frame": fast},
        run="run-fast",
        epoch="epoch-fast",
    )
    try:
        front = plane.freeze()
        slow_publication = front.publication("slow/frame")
        fast_publication = front.publication("fast/frame")
        assert slow_publication is not None and fast_publication is not None
        assert slow_publication is not fast_publication
        assert slow_publication.parents == fast_publication.parents == ()
        assert front.value("slow/frame").run_id == "run-slow"
        assert front.value("fast/frame").run_id == "run-fast"
        assert not hasattr(front, "shot")
    finally:
        plane.close()


def test_processor_advances_with_its_exact_source_publication() -> None:
    plane, state, source_node, first = _live_source_plane()
    gates = {1: threading.Event(), 2: threading.Event()}
    gates[1].set()
    processor = _GatedProcessorNode(
        plane,
        instance_id="occupancy",
        source_name="camera/frame",
        declared_outputs=("occupied",),
        gates=gates,
    )
    plane.set_front_signals(
        {"camera/frame", "camera/frame_aux", "occupancy/occupied"}
    )
    first_publication = first.publication("camera/frame")
    assert first_publication is not None
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_publication=first_publication,
    )
    try:
        admitted = _wait_for_signal_revision(plane, "occupancy/occupied", 1)
        derived_publication = admitted.publication("occupancy/occupied")
        assert derived_publication is not None
        assert derived_publication.parents == (first_publication,)

        state["frame"] = _live_output("frame", 2, "3" * 64)
        state["frame_aux"] = _live_output("frame_aux", 2, "4" * 64)
        plane.mark_changed(source_node)
        staged = plane.freeze()
        assert staged.value("camera/frame").snapshot.ref.revision.value == 1
        assert staged.value("camera/frame_aux").snapshot.ref.revision.value == 1
        assert staged.value("occupancy/occupied").snapshot.ref.revision.value == 1

        gates[2].set()
        advanced = _wait_for_signal_revision(plane, "occupancy/occupied", 2)
        source_publication = advanced.publication("camera/frame")
        result_publication = advanced.publication("occupancy/occupied")
        assert source_publication is not None and result_publication is not None
        assert result_publication.parents == (source_publication,)
        assert advanced.value("camera/frame_aux").snapshot.ref.revision.value == 2
    finally:
        for gate in gates.values():
            gate.set()
        plane.close()


def test_source_retirement_removes_the_complete_publication_closure() -> None:
    plane, _state, source_node, first = _live_source_plane()
    gate = threading.Event()
    processor = _GatedProcessorNode(
        plane,
        instance_id="occupancy",
        source_name="camera/frame",
        declared_outputs=("occupied",),
        gates={1: gate},
    )
    publication = first.publication("camera/frame")
    assert publication is not None
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_publication=publication,
    )
    try:
        retired = plane.retire(source_node)
        assert retired == frozenset(
            {"camera/frame", "camera/frame_aux", "occupancy/occupied"}
        )
        assert plane.freeze().names() == ()
        gate.set()
        time.sleep(0.01)
        assert plane.freeze().names() == ()
    finally:
        gate.set()
        plane.close()


def test_fan_out_binds_the_same_exact_visible_publication() -> None:
    plane, _state, _source_node, first = _live_source_plane()
    source_publication = first.publication("camera/frame")
    assert source_publication is not None
    processors = []
    try:
        for name in ("left", "right"):
            gate = threading.Event()
            gate.set()
            processor = _GatedProcessorNode(
                plane,
                instance_id=name,
                source_name="camera/frame",
                declared_outputs=("value",),
                gates={1: gate},
            )
            processors.append(processor)
            plane.attach_latest_only_processor(
                processor,
                source_name="camera/frame",
                initial_publication=source_publication,
            )
        for name in ("left", "right"):
            front = _wait_for_signal_revision(plane, f"{name}/value", 1)
            publication = front.publication(f"{name}/value")
            assert publication is not None
            assert publication.parents == (source_publication,)
    finally:
        plane.close()


def test_processor_publication_requires_every_declared_sibling() -> None:
    plane, _state, _source_node, first = _live_source_plane()
    gate = threading.Event()
    gate.set()
    processor = _GatedProcessorNode(
        plane,
        instance_id="partial",
        source_name="camera/frame",
        declared_outputs=("left", "right"),
        published_outputs=("left",),
        gates={1: gate},
    )
    publication = first.publication("camera/frame")
    assert publication is not None
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_publication=publication,
    )
    try:
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


@pytest.mark.parametrize("failure_point", ("construct", "launch"))
def test_task_console_failure_and_retirement_release_the_shared_plane_wake(
    tmp_path,
    monkeypatch,
    failure_point: str,
) -> None:
    """Only a successfully composed console may retain the application wake."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import Zou_lab_control.api as zlc
    from zlc_frontend.qt_widgets import ensure_qt_app
    import zlc_workbench.task_console.window as window_module

    application = ensure_qt_app()
    assert application.platformName().lower() == "offscreen"
    workspace = zlc.WorkspacePaths.for_workspace(
        REPO,
        repository_root=tmp_path.resolve(),
    )
    experiment = zlc.connect("virtual", workspace=workspace)
    original_constructor = window_module.TaskConsole
    original_launcher = window_module.launch_fluent_window

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"deterministic {failure_point} failure")

    if failure_point == "construct":
        monkeypatch.setattr(window_module, "TaskConsole", fail)
    else:
        monkeypatch.setattr(window_module, "launch_fluent_window", fail)
    try:
        with pytest.raises(RuntimeError, match=failure_point):
            experiment.task_console()
    finally:
        monkeypatch.setattr(window_module, "TaskConsole", original_constructor)
        monkeypatch.setattr(
            window_module,
            "launch_fluent_window",
            original_launcher,
        )

    first = experiment.task_console()
    try:
        first.request_owner_close()
        assert first.wait_owner_closed(10.0)
        assert first.permanently_closed

        second = experiment.task_console()
        assert second is not first
        assert not second.permanently_closed
    finally:
        experiment.close()

    assert second.permanently_closed
