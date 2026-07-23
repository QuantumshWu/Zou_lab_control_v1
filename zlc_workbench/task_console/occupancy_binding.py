"""Typed binding and lifecycle for the reactive occupancy Processor."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable
import uuid

from zlc_data import dataset_revision_ref_to_tree
from zlc_data.console_records import console_signal_key
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.readout.calibration import ResolvedCalibration
from zlc_neutral_atom.readout.calibration_reference import (
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.readout.occupancy import apply_occupancy_snapshot
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_storage import canonical_digest, canonical_text

from .data_plane import ConsoleDataPlane, ConsoleSignalValue

__all__ = [
    "ConsoleProducerBinding",
    "OccupancyBindingIntent",
    "ReactiveOccupancyNode",
]


@dataclass(frozen=True, slots=True)
class OccupancyBindingIntent:
    """Exact producer/output keys selected in the processor form."""

    camera_frame_signal: str
    calibration_signal: str

    def __post_init__(self) -> None:
        canonical_text(self.camera_frame_signal, "camera_frame_signal")
        canonical_text(self.calibration_signal, "calibration_signal")
        if self.camera_frame_signal == self.calibration_signal:
            raise ValueError(
                "occupancy source and calibration must be distinct console outputs"
            )


@dataclass(frozen=True, slots=True)
class ConsoleProducerBinding:
    """One exact output resolved against one row in this TaskConsole."""

    signal_key: str
    producer_label: str
    definition_key: DefinitionKey
    output_name: str
    request: object
    run_node: object | None
    final_result_resolved: bool
    final_result: object | None

    def __post_init__(self) -> None:
        canonical_text(self.signal_key, "signal_key")
        canonical_text(self.producer_label, "producer_label")
        if not isinstance(self.definition_key, DefinitionKey):
            raise TypeError("definition_key must be DefinitionKey")
        canonical_text(self.output_name, "output_name")
        if not isinstance(self.final_result_resolved, bool):
            raise TypeError("final_result_resolved must be bool")
        if not self.final_result_resolved and self.final_result is not None:
            raise ValueError(
                "an unresolved producer cannot expose a FINAL result"
            )

    @property
    def running(self) -> bool:
        return bool(
            self.run_node is not None
            and getattr(self.run_node, "running", False)
        )


class ReactiveOccupancyNode:
    """Latest-revision Processor over one already-running Camera signal.

    The node owns no device and never starts the selected producer.  Its sole
    worker admits the frozen calibration once, then classifies each new
    immutable camera front.  Counts and occupied are admitted to the console
    data plane together and retain the camera revision/run/epoch lineage.
    """

    def __init__(
        self,
        spec,
        values,
        *,
        intent: OccupancyBindingIntent,
        source_node: object,
        initial_source: ConsoleSignalValue,
        resolve_calibration: Callable[[], ResolvedCalibration],
        data_plane: ConsoleDataPlane,
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(intent, OccupancyBindingIntent):
            raise TypeError("intent must be OccupancyBindingIntent")
        if source_node is None:
            raise ValueError("reactive occupancy requires a running Camera node")
        if not isinstance(initial_source, ConsoleSignalValue):
            raise TypeError("initial_source must be ConsoleSignalValue")
        if initial_source.name != intent.camera_frame_signal:
            raise ValueError("initial Camera value differs from the selected signal")
        source_handle = getattr(source_node, "handle", None)
        source_run_id = getattr(getattr(source_handle, "run_id", None), "value", None)
        if source_run_id != initial_source.run_id:
            raise ValueError(
                "initial Camera value does not belong to the selected producer instance"
            )
        if not callable(resolve_calibration):
            raise TypeError("resolve_calibration must be callable")
        if not isinstance(data_plane, ConsoleDataPlane):
            raise TypeError("data_plane must be ConsoleDataPlane")
        if not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable")
        self._spec = spec
        self.instance_label = str(spec.title)
        self._values = dict(values)
        self._request = intent
        self._source_node = source_node
        self._source_handle = source_handle
        self._initial_source: ConsoleSignalValue | None = initial_source
        self._source_run_id = initial_source.run_id
        self._source_epoch_id = initial_source.epoch_id
        self._resolve_calibration = resolve_calibration
        self._data_plane = data_plane
        self._request_owner_wake = request_owner_wake
        self._executor: ThreadPoolExecutor | None = None
        self._binding_future: Future | None = None
        self._work_future: Future | None = None
        self._work_source: ConsoleSignalValue | None = None
        self._calibration: ResolvedCalibration | None = None
        self._last_source_ref = None
        self._run_id = RunId(f"reactive-occupancy-{uuid.uuid4().hex}")
        self._state: RunState | None = None
        self._phase = "not started"
        self._error: str | None = None
        self._cancel_requested = False

    @property
    def spec(self):
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def display_label(self) -> str:
        return self._spec.title

    @property
    def layer(self) -> str:
        return self._spec.kind

    @property
    def prefix(self) -> str:
        return ""

    @property
    def request(self) -> OccupancyBindingIntent:
        return self._request

    @property
    def handle(self):
        return None

    @property
    def running(self) -> bool:
        return self._state is RunState.RUNNING

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def final_result(self):
        return None

    @property
    def final_result_resolved(self) -> bool:
        return False

    @property
    def final_projection_resolved(self) -> bool:
        return False

    @property
    def projected_final_signals(self):
        return None

    @property
    def final_projection_error(self) -> None:
        return None

    def signal_key(self, output_name: str) -> str:
        return console_signal_key(self.instance_label, output_name)

    def published_signals(self) -> tuple[str, ...]:
        return tuple(
            self.signal_key(output.name)
            for output in self._spec.declared_outputs
        )

    def start(self) -> None:
        if self.running:
            return
        if self._state is not None:
            raise RuntimeError("reactive occupancy nodes are one-shot")
        self._state = RunState.RUNNING
        self._phase = "admitting calibration"
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="console-reactive-occupancy",
        )
        self._binding_future = self._submit(self._resolve_calibration)

    def cancel(self, _reason: str = "Console user requested stop") -> None:
        if self._state is None or self._state.terminal:
            return
        self._cancel_requested = True
        self._phase = "stopping after current pure transform"
        self._finish_cancel_if_idle()

    def poll(self) -> RunSnapshot | None:
        if self._state is None:
            return None
        if self._state.terminal:
            return self._snapshot()

        binding = self._binding_future
        if binding is not None and binding.done():
            self._binding_future = None
            try:
                calibration = binding.result()
                if type(calibration) is not ResolvedCalibration:
                    raise TypeError(
                        "calibration resolver did not return ResolvedCalibration"
                    )
                calibration.reference
                self._calibration = calibration
                self._phase = "waiting for a new Camera revision"
            except BaseException as error:
                self._fail(error)

        work = self._work_future
        if self._state is RunState.RUNNING and work is not None and work.done():
            source = self._work_source
            self._work_future = None
            self._work_source = None
            try:
                if source is None:
                    raise RuntimeError("occupancy transform lost its source revision")
                counts, occupied = work.result()
                calibration = self._calibration
                if type(calibration) is not ResolvedCalibration:
                    raise RuntimeError("occupancy calibration was not admitted")
                reference = calibration.reference
                model = calibration.artifact.select_model()
                join_digest = canonical_digest(
                    {
                        "owner": "zlc-workbench.reactive-occupancy-join",
                        "source_revision": dataset_revision_ref_to_tree(
                            source.snapshot.ref
                        ),
                        "source_event": source.join_digest,
                        "calibration": calibration_artifact_ref_to_tree(reference),
                        "model_kind": model.kind.value,
                    }
                )
                declared = {
                    output.name: self.signal_key(output.name)
                    for output in self._spec.declared_outputs
                }
                if set(declared) != {"counts", "occupied"}:
                    raise RuntimeError(
                        "occupancy catalog must declare counts and occupied"
                    )
                self._data_plane.publish_processor(
                    self,
                    {
                        declared["counts"]: ConsoleSignalValue(
                            name=declared["counts"],
                            source=self.name,
                            snapshot=counts,
                            coverage=source.coverage,
                            run_id=source.run_id,
                            epoch_id=source.epoch_id,
                            join_digest=join_digest,
                        ),
                        declared["occupied"]: ConsoleSignalValue(
                            name=declared["occupied"],
                            source=self.name,
                            snapshot=occupied,
                            coverage=source.coverage,
                            run_id=source.run_id,
                            epoch_id=source.epoch_id,
                            join_digest=join_digest,
                        ),
                    },
                )
                self._last_source_ref = source.snapshot.ref
                self._phase = "waiting for a new Camera revision"
            except BaseException as error:
                self._fail(error)

        if self._state is not RunState.RUNNING:
            return self._snapshot()
        if self._cancel_requested:
            self._finish_cancel_if_idle()
            return self._snapshot()
        if self._calibration is None or self._work_future is not None:
            return self._snapshot()

        if (
            getattr(self._source_node, "handle", None) is not self._source_handle
            or not bool(getattr(self._source_node, "running", False))
        ):
            self._fail(RuntimeError("selected Camera producer instance has stopped"))
            return self._snapshot()
        source, self._initial_source = (
            self._initial_source,
            None,
        )
        if source is None:
            source = self._data_plane.freeze().value(
                self._request.camera_frame_signal
            )
        if source is None:
            self._fail(
                RuntimeError(
                    "selected Camera source no longer publishes a frame signal"
                )
            )
            return self._snapshot()
        if (
            source.run_id != self._source_run_id
            or source.epoch_id != self._source_epoch_id
        ):
            self._fail(
                RuntimeError(
                    "selected Camera signal now belongs to another Run/epoch; "
                    "restart Occupancy to bind the new producer generation"
                )
            )
            return self._snapshot()
        if source.snapshot.ref != self._last_source_ref:
            calibration = self._calibration
            self._work_source = source
            self._work_future = self._submit(
                lambda: apply_occupancy_snapshot(
                    source.snapshot,
                    calibration,
                )
            )
            self._phase = (
                "classifying Camera revision "
                f"{source.snapshot.ref.revision.value}"
            )
        return self._snapshot()

    def shutdown(self) -> None:
        self.cancel("TaskConsole is closing")

    def _submit(self, work: Callable[[], object]) -> Future:
        executor = self._executor
        if executor is None:
            raise RuntimeError("reactive occupancy worker is not running")
        future = executor.submit(work)
        future.add_done_callback(lambda _future: self._request_owner_wake())
        return future

    def _finish_cancel_if_idle(self) -> None:
        if self._binding_future is not None or self._work_future is not None:
            return
        self._state = RunState.CANCELLED
        self._phase = "cancelled"
        self._close_executor()

    def _fail(self, error: BaseException) -> None:
        self._error = f"{type(error).__name__}: {error}"
        self._state = RunState.FAILED
        self._phase = "failed"
        if self._binding_future is None and self._work_future is None:
            self._close_executor()

    def _close_executor(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _snapshot(self) -> RunSnapshot:
        state = self._state
        if state is None:
            raise RuntimeError("reactive occupancy has not started")
        return RunSnapshot(
            self._run_id,
            state,
            self._phase,
            False,
            None,
            self._error,
            (),
            None,
        )
